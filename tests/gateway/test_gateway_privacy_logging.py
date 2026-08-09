"""Privacy regressions for gateway operational logging."""

import asyncio
import logging
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway import run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


def test_inbound_info_log_omits_message_and_identifiers(caplog):
    source = SimpleNamespace(
        platform=SimpleNamespace(value="telegram"),
        user_name="private_username",
        user_id="987654321",
        chat_id="-1001234567890",
    )
    event = SimpleNamespace(
        text="private message body with medical details",
        reply_to_message_id="42",
        reply_to_text="private quoted body",
    )

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        gateway_run._log_inbound_message(event, source)

    info = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
    assert "platform=telegram" in info
    assert "chars=41" in info
    for private in (
        "private_username",
        "987654321",
        "-1001234567890",
        "private message body",
        "private quoted body",
    ):
        assert private not in info


def test_inbound_debug_log_uses_hashes_and_bounded_preview(caplog):
    source = SimpleNamespace(
        platform=SimpleNamespace(value="telegram"),
        user_name="private_username",
        user_id="987654321",
        chat_id="-1001234567890",
    )
    secret_tail = "TAIL-MUST-NOT-APPEAR"
    event = SimpleNamespace(
        text=("x" * 200) + secret_tail,
        reply_to_message_id=None,
        reply_to_text="",
    )

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        gateway_run._log_inbound_message(event, source)

    debug = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)
    assert "user_hash=" in debug and "chat_hash=" in debug
    assert "private_username" not in debug
    assert "987654321" not in debug
    assert "-1001234567890" not in debug
    assert secret_tail not in debug


def test_transcript_lag_is_info_not_unverified_fts_corruption(caplog):
    session_key = "agent:main:telegram:dm:private-chat-id"

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        gateway_run._log_transcript_lag(session_key, disk_count=10, memory_count=11)

    text = caplog.text
    assert "transcript_lag" in text
    assert "classification=unverified" in text
    assert "FTS" not in text
    assert "corrupt" not in text.lower()
    assert session_key not in text
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_response_ready_log_resolves_platform_after_inbound_privacy_refactor(caplog):
    source = SimpleNamespace(
        platform=SimpleNamespace(value="telegram"), chat_id="-1001234567890"
    )

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        gateway_run._log_response_ready(
            source,
            response_time=0.2,
            api_calls=1,
            response_length=7,
        )

    assert "response ready: platform=telegram" in caplog.text
    assert "chat_hash=" in caplog.text
    assert "-1001234567890" not in caplog.text


def test_successful_agent_path_logs_and_returns_response(monkeypatch, tmp_path, caplog):
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _generation: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001234567890:12345",
        session_id="sess-response-ready",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "intended response",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "intended response"},
            ],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "api_calls": 1,
            "failed": False,
        }
    )
    event = MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001234567890",
            chat_type="group",
            user_id="12345",
        ),
        message_id="msg-response-ready",
    )

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        response = asyncio.run(
            runner._handle_message_with_agent(
                event,
                event.source,
                "agent:main:telegram:group:-1001234567890:12345",
                1,
            )
        )

    assert response == "intended response"
    assert len(response) != 91
    response_logs = [
        record.getMessage()
        for record in caplog.records
        if "response ready:" in record.getMessage()
    ]
    assert response_logs
    assert "chat_hash=" in response_logs[-1]
    assert "-1001234567890" not in response_logs[-1]
