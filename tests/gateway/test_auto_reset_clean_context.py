"""Compression exhaustion keeps the session; only an explicit reset starts clean (#9893, #35809).

Exhaustion used to auto-reset the session and re-point the Telegram topic binding at the fresh one
(#9893 / #10063, #35809), dropping the conversation's continuity. It now ends the turn with a notice and
leaves the session, its history and its routing where they were; the user picks ``/compress`` or ``/new``.

* ``TestCompressionExhaustionKeepsSession`` — a whole gateway turn (``_handle_message_with_agent``) whose
  agent result is exhausted: no reset, no eviction, a notice that names both ways forward, and the /goal
  and /loop hooks told the turn was exhausted so they bound their retries instead of judging it.
* ``TestAutoResetLoadsCleanContext`` — an explicit ``SessionStore.reset_session`` still yields an EMPTY
  next-turn transcript and keeps the old history searchable.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource, SessionStore
from hermes_state import SessionDB

SESSION_KEY = "agent:main:telegram:dm:123:42"


def _make_store(tmp_path):
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    # Isolate the SQLite transcript store so we exercise per-session_id
    # transcripts without touching the developer's real state.db.
    store._db = SessionDB(db_path=tmp_path / "state.db")
    return store


def _make_source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1")


def _bloat(n):
    # Stand-in for an oversized transcript that could not be compressed any
    # further. Alternates roles so the fixture is a valid conversation:
    # load_transcript is a live-replay restore site and heals alternation
    # violations on load (#64934), so a degenerate all-user transcript would
    # be merged into one message.
    return [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": "x" * 2000,
        }
        for i in range(n)
    ]


def _topic_source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", user_id="u1", thread_id="42")


def _turn_runner(monkeypatch, tmp_path, agent_result):
    """A real GatewayRunner whose agent returns ``agent_result``; store, adapters and hooks are stubs."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_a, **_k: 100_000)
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._begin_session_run_generation = lambda _key: 1
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=SESSION_KEY, session_id="sess-bloated", created_at=datetime.now(),
        updated_at=datetime.now(), platform=Platform.TELEGRAM, chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = _bloat(40)
    runner.session_store.has_platform_message_id.return_value = False
    runner.session_store.transcript_tail_role.return_value = "user"
    runner._evict_cached_agent = MagicMock()
    runner._sync_telegram_topic_binding = MagicMock()
    runner._run_agent = AsyncMock(return_value=agent_result)
    runner._post_turn_goal_continuation = AsyncMock()
    runner._post_turn_loop_completion = AsyncMock()
    return runner


class TestCompressionExhaustionKeepsSession:
    @pytest.mark.asyncio
    async def test_exhausted_turn_keeps_the_session_says_so_and_tells_goal_and_loop(self, monkeypatch, tmp_path):
        reply = "This conversation has grown too long."
        runner = _turn_runner(monkeypatch, tmp_path, {
            "final_response": reply, "error": reply, "messages": [], "history_offset": 0,
            "failed": True, "completed": False, "partial": True, "compression_exhausted": True,
            "failure_reason": "context_overflow", "last_prompt_tokens": 0,
        })
        event = MessageEvent(text="one more thing", source=_topic_source(), message_id="m-1")

        response = await runner._handle_message_with_agent(event, _topic_source(), SESSION_KEY, 1)

        runner.session_store.reset_session.assert_not_called()
        runner._evict_cached_agent.assert_not_called()
        runner._sync_telegram_topic_binding.assert_not_called()
        assert response.startswith(reply)
        assert "/compress" in response and "/new" in response, "the user is told how to get unstuck"

        await runner._run_post_turn_hooks(agent_result=response, source=_topic_source(), is_internal=False, event=event)
        for hook in (runner._post_turn_goal_continuation, runner._post_turn_loop_completion):
            assert hook.await_args.kwargs.get("compression_exhausted") is True, "exhaustion copy is not work to judge"


# ---------------------------------------------------------------------------
# Behavioral contract: an explicit reset yields a clean next-turn transcript
# ---------------------------------------------------------------------------
class TestAutoResetLoadsCleanContext:
    """An explicit ``reset_session`` makes the NEXT turn load an EMPTY transcript
    for the new session_id and leaves the old one searchable."""

    def test_next_turn_transcript_is_empty_after_auto_reset(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()

        entry = store.get_or_create_session(source)
        session_key = entry.session_key
        bloated_sid = entry.session_id
        store._db.create_session(
            session_id=bloated_sid, source="telegram", user_id="u1"
        )
        store._db.replace_messages(bloated_sid, _bloat(120))
        assert len(store.load_transcript(bloated_sid)) == 120  # precondition

        new_entry = store.reset_session(session_key)
        assert new_entry is not None
        assert new_entry.session_id != bloated_sid

        resolved = store.get_or_create_session(source)
        assert resolved.session_id == new_entry.session_id
        loaded = store.load_transcript(resolved.session_id)

        assert loaded == [], (
            f"Auto-reset must yield an empty context, got {len(loaded)} "
            f"messages — the bloated compressed child leaked into the new session."
        )
        # The old transcript is still searchable, not destroyed.
        assert len(store.load_transcript(bloated_sid)) == 120
