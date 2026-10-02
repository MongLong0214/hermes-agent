"""A failed transcript read must not masquerade as an empty history (#100788).

The gateway restore path (``_handle_message``) already fails closed on
current main (#100910). This file covers the surviving half of PR #100887:
the slash-command handlers, which used to let ``TranscriptReadError``
propagate into the dispatch wrapper and reply with nothing at all.

Incident shape: a malformed ``state.db`` made every
``SessionStore.load_transcript`` raise; the except-block swallowed it and
returned ``[]``.  Restore then rebuilt the turn from "no history", so a
long-running chat silently restarted as a brand-new conversation and the
model happily answered as if nothing had ever been discussed.

Two guarantees under test:
  A. ``load_transcript`` raises ``TranscriptReadError`` on a read failure,
     while a genuinely empty session still returns ``[]``.
  B. Slash-command handlers that read the transcript reply with
     ``HISTORY_UNREADABLE`` instead of raising into the dispatch wrapper
     (which logs and sends nothing).

Offline: SQLite on tmp_path only, no network.
"""

import sqlite3

import pytest

from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_transcript import TranscriptReadError


@pytest.fixture
def store(tmp_path):
    return SessionStore(sessions_dir=tmp_path / "gw", config=GatewayConfig())


# --------------------------------------------------------------------------
# A. read failure != empty transcript (landed on main via #100910; kept as
#    the contract the slash-command handlers below rely on)
# --------------------------------------------------------------------------


class TestLoadTranscriptReadFailure:
    def test_read_failure_raises_instead_of_returning_empty(self, store, monkeypatch):
        db = store._db
        assert db is not None
        db.create_session("s1", "telegram", session_key="telegram:1")
        db.append_message("s1", "user", "the conversation we must not forget")

        boom = sqlite3.DatabaseError("database disk image is malformed")

        def _raise(*_args, **_kwargs):
            raise boom

        monkeypatch.setattr(db, "get_messages_as_conversation", _raise)

        with pytest.raises(TranscriptReadError) as excinfo:
            store.load_transcript("s1")

        assert excinfo.value.session_id == "s1"
        assert excinfo.value.__cause__ is boom

    def test_genuinely_empty_session_still_returns_empty_list(self, store):
        db = store._db
        assert db is not None
        db.create_session("s2", "telegram", session_key="telegram:2")

        assert store.load_transcript("s2") == []

    def test_no_db_still_returns_empty_list(self, store):
        # "No DB for this session" really is an empty transcript, not a
        # failure — that path must keep its [] contract.
        store._db = None
        assert store.load_transcript("nope") == []


# --------------------------------------------------------------------------
# C. L4-1: a reopen failure during RecoverableHandleCache backoff is NOT the
#    same as "no SQLite backing" (``store._db = None``, the JSONL-only test
#    idiom above). An existing routed session's history may be sitting
#    behind that unavailable store, so it must fail the same way a
#    corrupt-read does (A above) rather than invent an empty conversation.
# --------------------------------------------------------------------------


def _break_next_reopen(store, monkeypatch) -> None:
    """Evict the cached handle and make the next open attempt raise — the
    real RecoverableHandleCache backoff path, not a deliberately pinned
    ``store._db = None``."""
    import hermes_state_registry

    path = next(iter(store._db_handle_cache.handles))
    store._db_handle_cache.handles.pop(path)

    def _boom(*_a, **_kw):
        raise OSError("database is locked")

    monkeypatch.setattr(hermes_state_registry, "acquire", _boom)


class TestUnavailableStoreVsDeliberatelyAbsentStore:
    def test_reopen_failure_on_existing_session_raises_not_empty(self, store, monkeypatch):
        db = store._db
        assert db is not None
        db.create_session("s3", "telegram", session_key="telegram:3")
        db.append_message("s3", "user", "the operator's only copy of this conversation")

        _break_next_reopen(store, monkeypatch)

        with pytest.raises(TranscriptReadError) as excinfo:
            store.load_transcript("s3")
        assert excinfo.value.session_id == "s3"

    def test_reopen_failure_on_existing_session_fails_the_rewrite(self, store, monkeypatch):
        db = store._db
        assert db is not None
        db.create_session("s4", "telegram", session_key="telegram:4")
        db.append_message("s4", "user", "keep me")

        _break_next_reopen(store, monkeypatch)

        assert store.rewrite_transcript("s4", [{"role": "user", "content": "x"}]) is False

    def test_pinned_none_keeps_its_empty_and_success_contract(self, store):
        """Guardrail: distinguishing the two must not regress the deliberate no-DB path."""
        store._db = None
        assert store.load_transcript("nope") == []
        assert store.rewrite_transcript("nope", []) is True


# --------------------------------------------------------------------------
# B. slash-command handlers surface the failure instead of dying silently.
#    Before: the handler raised, base.py's dispatch wrapper logged
#    "Command '/x' dispatch failed" and the user got NO reply at all.
# --------------------------------------------------------------------------


class TestSlashCommandsOnUnreadableTranscript:

    @pytest.mark.asyncio
    async def test_btw_replies_history_unreadable_on_read_failure(self):
        """The except-branch must actually run: after the Sep 2026 decomposition the
        constant lived in slash_commands_status but was not imported by slash_commands,
        so this path raised NameError (#102117 follow-up)."""
        from unittest.mock import AsyncMock, MagicMock

        from gateway.slash_commands_status import HISTORY_UNREADABLE
        from tests.gateway.test_background_command import _make_event, _make_runner

        runner = _make_runner()
        store = AsyncMock()
        store.get_or_create_session.return_value = MagicMock(session_id="s1")
        store.load_transcript.side_effect = TranscriptReadError("malformed")
        store._store = runner.session_store
        runner._async_session_store = store

        result = await runner._handle_btw_command(_make_event(text="/btw what?"))
        assert result == HISTORY_UNREADABLE

