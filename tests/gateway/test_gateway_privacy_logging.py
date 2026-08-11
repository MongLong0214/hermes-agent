"""Privacy regressions for gateway operational logging."""

import hashlib
import logging
from types import SimpleNamespace

from gateway import run as gateway_run


def test_inbound_logs_never_expose_bodies_or_correlatable_identifiers(caplog):
    source = SimpleNamespace(
        platform=SimpleNamespace(value="telegram"),
        user_name="private_username",
        user_id="sensitive-user-id-fixture",
        chat_id="sensitive-chat-id-fixture",
    )
    event = SimpleNamespace(
        text="private message body with medical details",
        reply_to_message_id="private-reply-id-42",
        reply_to_text="private quoted body",
    )

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        gateway_run._log_inbound_message(event, source)

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "platform=telegram" in text
    assert "chars=41" in text
    for private in (
        "private_username",
        "sensitive-user-id-fixture",
        "sensitive-chat-id-fixture",
        "private message body",
        "private quoted body",
        "private-reply-id-42",
        hashlib.sha256(b"private_username").hexdigest()[:12],
        hashlib.sha256(b"sensitive-chat-id-fixture").hexdigest()[:12],
    ):
        assert private not in text
    assert "preview=" not in text
    assert "user_hash=" not in text
    assert "chat_hash=" not in text


def test_transcript_lag_uses_only_non_identifying_counts_and_classification(caplog):
    session_key = "agent:main:telegram:dm:private-chat-id"
    session_digest = hashlib.sha256(session_key.encode()).hexdigest()[:12]

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        gateway_run._log_transcript_lag(disk_count=10, memory_count=11)

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "transcript_lag disk=10 memory=11 classification=unverified"
    ]
    text = "\n".join(messages)
    assert session_key not in text
    assert session_digest not in text
    assert "session" not in text
    assert "hash" not in text
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_response_ready_log_omits_raw_and_hashed_chat_identifier(caplog):
    chat_id = "sensitive-response-chat-id-fixture"
    source = SimpleNamespace(platform=SimpleNamespace(value="telegram"), chat_id=chat_id)

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        gateway_run._log_response_ready(
            source,
            response_time=0.2,
            api_calls=1,
            response_length=7,
        )

    assert "response ready: platform=telegram" in caplog.text
    assert chat_id not in caplog.text
    assert hashlib.sha256(chat_id.encode()).hexdigest()[:12] not in caplog.text
    assert "chat=" not in caplog.text
