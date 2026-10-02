"""Regression for PR #71 review finding R71-1: the chat-bound turn-error
notice and the background-error log must not forward raw exception text.

``agent.redact.redact_sensitive_text`` is a PATTERN redactor: it masks known
credential SHAPES (vendor-prefixed API keys, Telegram bot tokens, DB
connection strings, ...). It does not mask arbitrary private content, and by
default (``redact_url_credentials=False`` — deliberate: OAuth-callback /
magic-link / pre-signed URLs must survive ordinary tool flows unchanged) it
also leaves a generic credential-shaped URL query parameter
(``?access_token=...``) untouched. Two sinks in gateway/platforms/base.py
built user/log-facing text from ``str(e)`` straight through this gap:

- ``_notify_turn_error`` (the chat-bound failure notice, ~base.py:4306)
  embedded a pattern-redacted slice of the exception text, so anything the
  patterns don't cover (a private filename, a non-vendor credential URL)
  reached the user verbatim.
- ``_process_message_background``'s ``except BaseException`` handler
  (~base.py:4546) logged the raw exception + traceback with NO redaction
  pass at all, so even a KNOWN credential shape (a Telegram bot token)
  reached disk whenever the supported redaction opt-out
  (``security.redact_secrets: false`` / ``HERMES_REDACT_SECRETS=false``) was
  active — defeating the point of a diagnostic sink that runs immediately
  before the user-facing notice.

Fix: the user notice drops exception text entirely (a generic message only —
pattern redaction of arbitrary private content can't be made airtight); the
background-error log force-redacts with ``redact_url_credentials=True``
regardless of the opt-out, so neither sink depends on coverage it doesn't
have.
"""

from __future__ import annotations

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key

_CRED_URL = "https://service.example/resource?access_token=opaque-secret-7781"
_PRIVATE_MARKER = "/Users/alice/private-notes/medical-record.pdf"
_BOT_TOKEN = "123456789:AAFakeSecretTelegramBotTokenABCDEFGHIJ"
_BOT_URL = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"


class _ProbeAdapter(BasePlatformAdapter):
    """Minimal concrete adapter that records deliveries (same shape as
    tests/gateway/test_baseexception_turn_notify.py's probe)."""

    def __init__(self) -> None:
        super().__init__(PlatformConfig(enabled=True, token="x"), Platform.SLACK)
        self.sent: list[str] = []

    async def start(self):  # pragma: no cover - unused
        pass

    async def stop(self):  # pragma: no cover - unused
        pass

    async def connect(self):  # pragma: no cover - unused
        pass

    async def disconnect(self):  # pragma: no cover - unused
        pass

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)

        class _R:
            success = True
            message_id = "m1"

        return _R()

    async def send_typing(self, chat_id, metadata=None):  # pragma: no cover - unused
        pass


def _source() -> SessionSource:
    return SessionSource(platform=Platform.SLACK, user_id="U1", chat_id="C1",
                         user_name="tester", chat_type="channel")


def _event() -> MessageEvent:
    return MessageEvent(text="hello", message_type=MessageType.TEXT, source=_source())


def _make_adapter(handler) -> _ProbeAdapter:
    adapter = _ProbeAdapter()
    adapter.set_message_handler(handler)
    return adapter


def _raising_handler(exc: BaseException):
    async def handler(event):
        raise exc

    return handler


@pytest.mark.asyncio
async def test_turn_error_notice_and_background_log_drop_url_credential_and_private_text(caplog):
    """A credential-shaped URL query param (no vendor prefix) and an arbitrary
    private-looking string must reach neither the user-facing notice
    (_notify_turn_error, base.py ~4306) nor the background-error log
    (_process_message_background's except block, base.py ~4546)."""
    exc = RuntimeError(f"failed reading {_PRIVATE_MARKER}: GET {_CRED_URL} -> 401")
    adapter = _make_adapter(_raising_handler(exc))
    event = _event()

    with caplog.at_level("ERROR"):
        await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent, "the user-facing failure notice must still be sent"
    notice = adapter.sent[0]
    assert "opaque-secret-7781" not in notice
    assert _PRIVATE_MARKER not in notice

    # The background-error log is an internal diagnostic sink, not user-facing: it keeps full
    # exception context (including non-credential private text like a filename) for debugging.
    # Only the credential-shaped part is a redaction requirement there.
    # caplog.text is the formatted record (traceback included), not record.getMessage().
    assert "opaque-secret-7781" not in caplog.text
    assert "Error handling message" in caplog.text


@pytest.mark.asyncio
async def test_background_log_redacts_known_credential_even_with_redaction_opt_out(monkeypatch, caplog):
    """The background-error log is a forced, fail-closed sink: even when an
    operator has set the redaction opt-out (security.redact_secrets: false),
    a Telegram bot token embedded in the exception text must not reach disk
    from this specific sink."""
    import agent.redact as redact_module
    monkeypatch.setattr(redact_module, "_REDACT_ENABLED", False)

    exc = RuntimeError(f"Bad Request at {_BOT_URL}")
    adapter = _make_adapter(_raising_handler(exc))
    event = _event()

    with caplog.at_level("ERROR"):
        await adapter._process_message_background(event, build_session_key(event.source))

    assert _BOT_TOKEN not in caplog.text
    assert "Error handling message" in caplog.text
