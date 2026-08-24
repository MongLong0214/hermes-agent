"""Regression test for /branch reporting create_failed after already committing (R-5).

``_handle_branch_command`` (gateway/slash_commands.py) wraps its
``self._session_db.create_session(...)`` call in a bare ``try/except
Exception`` and unconditionally returns ``gateway.branch.create_failed`` on
any exception from that call.

But ``SessionDB.create_session`` -> ``_insert_session_row`` ->
``_execute_write`` commits the INSERT transaction *before* running
best-effort post-commit maintenance (periodic WAL checkpoint / incremental
FTS merge) that is not fully exception-safe. If one of those later steps
raises, the exception surfaces from the same ``create_session()`` call even
though the branch row is already durably committed to state.db.

Reporting create_failed in that case is a lie: the caller is told branching
failed while a fully routable "shadow" session now exists that nobody was
told about (findable by /resume, list_sessions, etc).

This test drives the REAL ``_handle_branch_command`` against a REAL
SessionStore + SessionDB (SQLite in a tmp_path, no mocks on the DB/session
layer) and simulates exactly that failure shape: the underlying
``create_session`` genuinely performs its insert (so the row is truly
committed) and then raises, standing in for the post-commit maintenance
step raising. The reported outcome must match what actually happened.
"""

from __future__ import annotations

import unittest.mock as mock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, SessionStore
from hermes_state import AsyncSessionDB


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Real SessionStore backed by a real SessionDB (SQLite in tmp_path)."""
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    config = GatewayConfig()
    return SessionStore(sessions_dir=tmp_path, config=config)


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="170829464",
        chat_id="170829464",
        chat_type="dm",
        thread_id="544520",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_branch_runner(store: SessionStore):
    """Minimal GatewayRunner stub wired to a REAL session_store/session_db."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._agent_cache_lock = None
    runner.session_store = store
    runner._session_db = AsyncSessionDB(store._db)
    runner._pending_skills_reload_notes = {}
    return runner


class TestBranchCreateFailedReport:
    @pytest.mark.asyncio
    async def test_branch_does_not_report_create_failed_for_a_committed_row(self, store):
        source = _make_source()
        parent_entry = store.get_or_create_session(source)
        store._db.append_message(parent_entry.session_id, role="user", content="hello")

        runner = _make_branch_runner(store)

        real_create_session = store._db.create_session
        captured: dict[str, str] = {}

        def _commit_then_raise(session_id, source, **kwargs):
            # The real write path: this actually performs and commits the
            # INSERT (identical to production) before we simulate the
            # post-commit maintenance step raising.
            real_create_session(session_id, source, **kwargs)
            captured["id"] = session_id
            raise RuntimeError("simulated post-commit maintenance failure")

        with mock.patch.object(store._db, "create_session", side_effect=_commit_then_raise):
            result = await runner._handle_branch_command(_make_event("/branch"))

        new_session_id = captured.get("id")
        assert new_session_id, "create_session was never called"
        assert new_session_id != parent_entry.session_id

        row = store._db.get_session(new_session_id)
        assert row is not None, "the branch row must actually be committed to state.db"

        assert "Failed to create branch" not in result, (
            "handler reported create_failed for a session that was actually "
            f"committed to the store (row exists: {row!r}); the caller now "
            f"believes branch creation failed while a routable shadow "
            f"session exists that nobody was told about. got: {result!r}"
        )
