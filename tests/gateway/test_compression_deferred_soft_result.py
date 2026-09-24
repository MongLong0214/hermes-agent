"""Gateway must treat ``compression_deferred`` as a soft result (#49874).

A lock-contended compression defer means a CONCURRENT compressor is actively shrinking the session. The
turn is not exhausted: the session stays intact, the reply carries no exhaustion notice, the /goal and /loop
hooks are not told it exhausted, and the next message retries normally — even when a stale
``compression_exhausted`` bit rides along (#69870). Driven through a whole gateway turn.
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
from gateway.session import SessionEntry, SessionSource

SESSION_KEY = "agent:main:telegram:dm:123"


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", user_id="u1")


def _turn_runner(monkeypatch, tmp_path, agent_result):
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
        session_key=SESSION_KEY, session_id="sess-deferred", created_at=datetime.now(),
        updated_at=datetime.now(), platform=Platform.TELEGRAM, chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_platform_message_id.return_value = False
    runner.session_store.transcript_tail_role.return_value = "user"
    runner._evict_cached_agent = MagicMock()
    runner._run_agent = AsyncMock(return_value=agent_result)
    runner._post_turn_goal_continuation = AsyncMock()
    runner._post_turn_loop_completion = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_deferred_result_leaves_session_reply_and_hooks_untouched(monkeypatch, tmp_path):
    reply = "Context compression is already running for this session. Please retry in a moment."
    runner = _turn_runner(monkeypatch, tmp_path, {
        "final_response": reply, "error": reply, "messages": [], "history_offset": 0, "failed": True,
        "compression_deferred": True, "compression_exhausted": True, "last_prompt_tokens": 0,
    })
    event = MessageEvent(text="hello", source=_source(), message_id="m-1")

    response = await runner._handle_message_with_agent(event, _source(), SESSION_KEY, 1)

    runner.session_store.reset_session.assert_not_called()
    runner._evict_cached_agent.assert_not_called()
    assert response == reply
    await runner._run_post_turn_hooks(agent_result=response, source=_source(), is_internal=False, event=event)
    assert runner._post_turn_loop_completion.await_args.kwargs.get("compression_exhausted", False) is False
