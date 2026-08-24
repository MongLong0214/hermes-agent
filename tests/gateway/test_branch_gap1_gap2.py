"""Regression tests for two gaps left in the R-5 /branch create_failed fix.

R-5 (fb9c3f39ba) made ``_handle_branch_command`` check ``get_session(new_id)``
before reporting ``create_failed`` when ``create_session`` raises, so a row
that was actually committed (e.g. a post-commit maintenance step raised
after the INSERT) is no longer misreported as a failure.

Two gaps remain in that fix:

Gap 1: the confirming ``get_session()`` call can itself raise (e.g. during
its own token-count flush or read). The current code treats *that*
exception exactly like "the row doesn't exist" and reports
``create_failed`` anyway -- which can be the same lie the R-5 fix exists to
prevent, just one level deeper: we genuinely don't know whether the row
exists, but we tell the caller it doesn't.

Gap 2: once past the create_session/get_session check, two further
sub-steps -- copying history (``append_messages_batch``) and setting the
title (``set_session_title``) -- each wrap their call in a bare
``except Exception: pass``. If either raises, the failure leaves zero
trace (not even a log line) and the command still returns its normal
"branched successfully" text.
"""

from __future__ import annotations

import logging
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


class TestBranchGap1ConfirmGetSessionRaises:
    """get_session() raising while confirming must not be read as "absent"."""

    @pytest.mark.asyncio
    async def test_get_session_raising_during_confirm_is_not_reported_as_create_failed(
        self, store
    ):
        source = _make_source()
        parent_entry = store.get_or_create_session(source)
        store._db.append_message(parent_entry.session_id, role="user", content="hello")

        runner = _make_branch_runner(store)

        real_create_session = store._db.create_session
        real_get_session = store._db.get_session
        captured: dict[str, str] = {}

        def _commit_then_raise(session_id, source, **kwargs):
            # Real insert, identical to production, then simulate the
            # post-commit maintenance step raising (same shape as the R-5
            # test).
            real_create_session(session_id, source, **kwargs)
            captured["id"] = session_id
            raise RuntimeError("simulated post-commit maintenance failure")

        def _get_session_raises_for_new_row(session_id):
            # Simulate get_session() itself raising during its own
            # flush/read -- but only for the branch row being confirmed, so
            # unrelated get_session calls elsewhere are unaffected.
            if captured.get("id") and session_id == captured["id"]:
                raise RuntimeError("simulated get_session flush/read failure")
            return real_get_session(session_id)

        with mock.patch.object(
            store._db, "create_session", side_effect=_commit_then_raise
        ), mock.patch.object(
            store._db, "get_session", side_effect=_get_session_raises_for_new_row
        ):
            with pytest.raises(RuntimeError):
                await runner._handle_branch_command(_make_event("/branch"))

        new_session_id = captured.get("id")
        assert new_session_id, "create_session was never called"

        # The row really was committed -- proving that reporting
        # create_failed (confirmed absent) would have been a lie.
        row = real_get_session(new_session_id)
        assert row is not None, "the branch row must actually be committed to state.db"


class TestBranchGap2SwallowedSubStepFailures:
    """History-copy and title-set failures must not vanish without a trace."""

    @pytest.mark.asyncio
    async def test_history_copy_failure_is_not_silently_swallowed(self, store, caplog):
        source = _make_source()
        parent_entry = store.get_or_create_session(source)
        store._db.append_message(parent_entry.session_id, role="user", content="hello")

        runner = _make_branch_runner(store)

        with mock.patch.object(
            store._db,
            "append_messages_batch",
            side_effect=RuntimeError("simulated history-copy failure"),
        ):
            with caplog.at_level(logging.ERROR, logger="gateway.run"):
                result = await runner._handle_branch_command(_make_event("/branch"))

        assert "Branched to" in result, f"expected the best-effort branch to still succeed, got: {result!r}"

        matching = [
            r for r in caplog.records
            if "simulated history-copy failure" in r.getMessage()
            or (r.exc_info and "simulated history-copy failure" in str(r.exc_info[1]))
        ]
        assert matching, (
            "history-copy failure left no trace in the logs -- it was "
            "silently swallowed even though the returned message claims success"
        )

    @pytest.mark.asyncio
    async def test_title_set_failure_is_not_silently_swallowed(self, store, caplog):
        source = _make_source()
        parent_entry = store.get_or_create_session(source)
        store._db.append_message(parent_entry.session_id, role="user", content="hello")

        runner = _make_branch_runner(store)

        with mock.patch.object(
            store._db,
            "set_session_title",
            side_effect=RuntimeError("simulated title-set failure"),
        ):
            with caplog.at_level(logging.ERROR, logger="gateway.run"):
                result = await runner._handle_branch_command(_make_event("/branch"))

        assert "Branched to" in result, f"expected the best-effort branch to still succeed, got: {result!r}"

        matching = [
            r for r in caplog.records
            if "simulated title-set failure" in r.getMessage()
            or (r.exc_info and "simulated title-set failure" in str(r.exc_info[1]))
        ]
        assert matching, (
            "title-set failure left no trace in the logs -- it was "
            "silently swallowed even though the returned message claims success"
        )
