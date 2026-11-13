"""Every record the Telegram adapter logs is secret-redacted before any handler sees it.

Call sites redact the error text they format, but ``exc_info=True`` makes logging append the raw
exception text (a Bot API URL in it carries the token), and a handler without the redacting
formatter (or a profile with ``security.redact_secrets: false``) would print the record as built.
The adapter's logger therefore force-redacts message and traceback itself, URL credentials included.
"""

import logging

import httpx
import pytest

from plugins.platforms.telegram import adapter as tg_adapter

_TOKEN = "123456789:AAFakeSecretTelegramBotTokenABCDEFGHIJ"
_POLL_URL = f"https://api.telegram.org/bot{_TOKEN}/getUpdates"


class _PlainCapture(logging.Handler):
    """A handler with the stock formatter: no redaction of its own, traceback included."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))


@pytest.fixture
def captured():
    logger = tg_adapter.logger
    handler, level = _PlainCapture(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


def _text(handler) -> str:
    return "\n".join(handler.lines)


def test_httpx_request_record_masks_the_bot_token(captured):
    # httpx's own request-line shape, with an ``httpx.URL`` (not a str) as the argument.
    tg_adapter.logger.info('HTTP Request: %s %s "%s %d %s"', "POST", httpx.URL(_POLL_URL), "HTTP/1.1", 200, "OK")

    text = _text(captured)
    assert _TOKEN.split(":", 1)[1] not in text
    assert "api.telegram.org/bot123456789:***/getUpdates" in text
    assert 'HTTP Request: POST' in text and '"HTTP/1.1 200 OK"' in text


def test_traceback_masks_the_bot_token(captured):
    try:
        try:
            raise httpx.ConnectError(f"All connection attempts failed for {_POLL_URL}")
        except httpx.ConnectError as cause:
            raise RuntimeError(f"getUpdates failed: {_POLL_URL}") from cause
    except RuntimeError:
        tg_adapter.logger.warning("[Telegram] polling failed", exc_info=True)

    text = _text(captured)
    assert _TOKEN.split(":", 1)[1] not in text
    assert "Traceback (most recent call last)" in text
    assert "httpx.ConnectError" in text and "RuntimeError: getUpdates failed" in text
    assert text.count("bot123456789:***") == 2  # the cause and the raised error both kept, masked


def test_record_args_are_still_formatted(captured):
    error = OSError(f"Network error connecting to {_POLL_URL}")
    tg_adapter.logger.warning("[%s] Connect attempt %d/%d failed: %s — retrying in %ds", "Telegram", 2, 8, error, 4)
    tg_adapter.logger.info("%(name)s polled %(count)d updates", {"name": "Telegram", "count": 3})
    tg_adapter.logger.info("plain line, no args")

    assert captured.lines == [
        "WARNING [Telegram] Connect attempt 2/8 failed: Network error connecting to "
        "https://api.telegram.org/bot123456789:***/getUpdates — retrying in 4s",
        "INFO Telegram polled 3 updates",
        "INFO plain line, no args",
    ]


@pytest.mark.parametrize("proxy", [
    "http://alice:s3cretPassw0rd@proxy.example:8080",
    "socks5://opaqueproxytoken123456@proxy.example:1080",
])
def test_proxy_url_userinfo_is_masked(captured, proxy):
    tg_adapter.logger.warning("[Telegram] proxy %s unreachable", proxy)

    text = _text(captured)
    assert "s3cretPassw0rd" not in text and "alice" not in text
    assert "opaqueproxytoken" not in text and "123456" not in text  # no head/tail of an opaque credential
    assert "://***:***@proxy.example:" in text


def test_a_failing_redaction_never_drops_the_record(captured, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("redactor unavailable")

    monkeypatch.setattr("agent.redact.redact_sensitive_text", boom)
    tg_adapter.logger.warning("[Telegram] token %s", _TOKEN)

    assert len(captured.lines) == 1
    assert _TOKEN not in captured.lines[0]  # fails closed to the placeholder, never the raw text


def test_an_unformattable_record_fails_closed(captured, capsys):
    # One argument short: formatting raises, and logging's own error dump would print the raw arguments.
    tg_adapter.logger.warning("[Telegram] token %s rejected by %s", _TOKEN)

    assert captured.lines == ["WARNING <telegram log record redacted>"]
    assert _TOKEN not in capsys.readouterr().err
