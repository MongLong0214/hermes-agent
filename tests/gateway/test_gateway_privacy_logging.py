"""Privacy regressions for gateway operational logging."""

import logging
from types import SimpleNamespace

from gateway import run as gateway_run


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
    source = SimpleNamespace(platform=SimpleNamespace(value="telegram"), chat_id="chat")

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        gateway_run._log_response_ready(
            source,
            response_time=0.2,
            api_calls=1,
            response_length=7,
        )

    assert "response ready: platform=telegram" in caplog.text
