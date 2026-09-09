"""Schema incompatibility must be actionable without exposing exception text."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource
from hermes_state import IncompatibleSchemaError


@pytest.mark.parametrize("case", ["generations", "missing", "generic"])
def test_schema_incompatible_owner_message(monkeypatch, tmp_path, case):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length", lambda *a, **kw: 100_000
    )
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._session_db = MagicMock()
    runner._set_session_env = lambda context: None
    runner._recover_telegram_topic_thread_id = lambda source: None
    runner._cache_session_source = lambda key, source: None
    runner._is_session_run_current = lambda key, generation: True
    runner._reply_anchor_for_event = lambda event: None
    runner._get_guild_id = lambda event: None
    runner._should_send_voice_reply = lambda *a, **kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="schema-owner-message", session_id="schema-owner-message",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_platform_message_id.return_value = False
    secret = "PRIVATE_EXCEPTION_SENTINEL"
    if case == "generations":
        error = IncompatibleSchemaError(expected_generation=41, actual_generation=47)
    elif case == "missing":
        error = IncompatibleSchemaError()
    else:
        error = RuntimeError(secret)
        error.expected_generation = 41
        error.actual_generation = 47
        error.code = "STATE_DB_SCHEMA_INCOMPATIBLE"
    # Even typed exceptions may carry private build/path details in their text.
    error.args = (secret,)
    runner._run_agent = AsyncMock(side_effect=error)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="owner", chat_type="dm", user_id="owner"
    )
    response = asyncio.run(runner._handle_message_with_agent(
        MessageEvent(text="hello", source=source), source, "schema-owner-message", 1
    ))
    runner._run_agent.assert_awaited_once()
    assert secret not in response
    if case == "generic":
        assert "unexpected error" in response
        assert "compatible Hermes build" not in response
        assert "41" not in response and "47" not in response
    else:
        assert "schema" in response.lower()
        assert "compatible Hermes build" in response
        assert "/reset" not in response
        if case == "generations":
            assert "expected generation 41" in response
            assert "actual generation 47" in response
        else:
            assert not any(char.isdigit() for char in response)
            assert "None" not in response
