"""Regression tests for remaining unredacted Telegram transport-error sites.

``c3ab1424e`` added ``_redact_telegram_error_text()`` (built on
``agent.redact``'s bot-token stripping for
``api.telegram.org/bot<TOKEN>/...`` URLs) and applied it across the
send/edit transient-error paths. Four sites still built their message from
the raw exception:

- ``connect()``'s fatal-error path — the most severe: the raw text is
  passed to ``_set_fatal_error()``, which *persists* it via
  ``write_runtime_status()`` to a dashboard/admin-facing runtime status
  file, not just a log line. A transient network error during startup
  commonly embeds the request URL (``https://api.telegram.org/bot<TOKEN>/
  getMe``), so this could leak the live bot token into that surface.
- ``disconnect()``, ``send_document()``, ``send_video()`` — log-only,
  lower blast radius, but the same unredacted-exception pattern.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from plugins.platforms.telegram.adapter import TelegramAdapter

_SECRET_TOKEN = "123456789:AAFakeSecretTelegramBotTokenABCDEFGHIJ"
_SECRET_URL = f"https://api.telegram.org/bot{_SECRET_TOKEN}/getMe"


def _make_bare_adapter() -> TelegramAdapter:
    config = PlatformConfig(enabled=True, token=_SECRET_TOKEN, extra={})
    return TelegramAdapter(config)


def _make_connected_adapter() -> TelegramAdapter:
    """Adapter with a mock bot wired, past connect() — for send_* tests."""
    adapter = _make_bare_adapter()
    bot = MagicMock()
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot
    return adapter


@pytest.mark.asyncio
async def test_connect_failure_redacts_token_from_fatal_status(monkeypatch):
    """A connect()-time exception embedding the bot token URL must not reach
    the persisted fatal-error status or the log line unredacted."""
    adapter = _make_bare_adapter()

    def _boom(*_args, **_kwargs):
        raise RuntimeError(f"Network error connecting to {_SECRET_URL}")

    monkeypatch.setattr(adapter, "_acquire_platform_lock", _boom)
    monkeypatch.setattr(adapter, "_write_runtime_status_safe", lambda *a, **k: None)

    result = await adapter.connect()

    assert result is False
    assert adapter._fatal_error_message is not None
    assert _SECRET_TOKEN not in adapter._fatal_error_message
    assert "***" in adapter._fatal_error_message


@pytest.mark.asyncio
async def test_disconnect_failure_redacts_token_in_log(monkeypatch, caplog):
    """A disconnect()-time exception embedding the bot token URL must not
    reach the warning log unredacted."""
    adapter = _make_connected_adapter()
    adapter._app = SimpleNamespace(
        updater=SimpleNamespace(running=False),
        running=False,
        shutdown=AsyncMock(side_effect=RuntimeError(f"teardown failed: {_SECRET_URL}")),
    )
    adapter._release_platform_lock = lambda: None
    adapter._cancel_pending_delivery_tasks = AsyncMock()

    with caplog.at_level("WARNING"):
        await adapter.disconnect()

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert _SECRET_TOKEN not in logged
    assert "Error during Telegram disconnect" in logged


@pytest.mark.asyncio
async def test_send_document_failure_redacts_token_in_log(monkeypatch, caplog, tmp_path):
    """A send_document() transport exception embedding the bot token URL
    must not reach the warning log unredacted."""
    adapter = _make_connected_adapter()
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr(
        adapter,
        "_send_with_dm_topic_reply_anchor_retry",
        AsyncMock(side_effect=RuntimeError(f"upload failed: {_SECRET_URL}")),
    )
    fallback = AsyncMock(return_value=SimpleNamespace(success=False, error="fallback"))
    monkeypatch.setattr(BasePlatformAdapter, "send_document", fallback)

    with caplog.at_level("WARNING"):
        await adapter.send_document("123", str(file_path))

    # caplog.text is the formatted record, traceback included (exc_info=True).
    assert "Failed to send document" in caplog.text
    assert _SECRET_TOKEN not in caplog.text


@pytest.mark.asyncio
async def test_send_video_failure_redacts_token_in_log(monkeypatch, caplog, tmp_path):
    """A send_video() transport exception embedding the bot token URL must
    not reach the warning log unredacted."""
    adapter = _make_connected_adapter()
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake mp4 bytes")

    monkeypatch.setattr(
        adapter,
        "_send_with_dm_topic_reply_anchor_retry",
        AsyncMock(side_effect=RuntimeError(f"upload failed: {_SECRET_URL}")),
    )
    fallback = AsyncMock(return_value=SimpleNamespace(success=False, error="fallback"))
    monkeypatch.setattr(BasePlatformAdapter, "send_video", fallback)

    with caplog.at_level("WARNING"):
        await adapter.send_video("123", str(video_path))

    # caplog.text is the formatted record, traceback included (exc_info=True).
    assert "Failed to send video" in caplog.text
    assert _SECRET_TOKEN not in caplog.text


_SECRET_SEND_URL = f"https://api.telegram.org/bot{_SECRET_TOKEN}/sendMessage"


@pytest.mark.asyncio
async def test_send_update_prompt_failure_redacts_token_in_result_and_log(caplog):
    """A send_update_prompt() transport exception embedding the bot token URL
    must not reach the warning log or SendResult.error unredacted."""
    adapter = _make_connected_adapter()
    adapter._send_message_with_thread_fallback = AsyncMock(
        side_effect=RuntimeError(f"Timed out requesting {_SECRET_SEND_URL}")
    )

    with caplog.at_level("WARNING"):
        result = await adapter.send_update_prompt("123", "restart?")

    assert result.success is False
    assert _SECRET_TOKEN not in (result.error or "")
    assert "***" in (result.error or "")
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert _SECRET_TOKEN not in logged


@pytest.mark.asyncio
async def test_send_clarify_failure_redacts_token_in_result_and_log(caplog):
    """A send_clarify() transport exception embedding the bot token URL must
    not reach the warning log or SendResult.error unredacted."""
    adapter = _make_connected_adapter()
    adapter._send_message_with_thread_fallback = AsyncMock(
        side_effect=RuntimeError(f"Timed out requesting {_SECRET_SEND_URL}")
    )

    with caplog.at_level("WARNING"):
        result = await adapter.send_clarify("123", "q?", ["a", "b"], "cid", "sess")

    assert result.success is False
    assert _SECRET_TOKEN not in (result.error or "")
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert _SECRET_TOKEN not in logged


@pytest.mark.asyncio
async def test_delete_message_failure_redacts_token_in_log(caplog):
    """A delete_message() transport exception embedding the bot token URL
    must not reach the debug log unredacted."""
    adapter = _make_connected_adapter()
    adapter._bot.delete_message = AsyncMock(
        side_effect=RuntimeError(f"Bad Request: {_SECRET_SEND_URL}")
    )

    with caplog.at_level("DEBUG"):
        ok = await adapter.delete_message("123", "55")

    assert ok is False
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert _SECRET_TOKEN not in logged
    assert "***" in logged


def _secret_error(endpoint: str) -> RuntimeError:
    return RuntimeError(f"Bad Request at https://api.telegram.org/bot{_SECRET_TOKEN}/{endpoint}")


@pytest.mark.asyncio
async def test_inbound_photo_cache_failure_redacts_traceback_and_reply(monkeypatch, caplog):
    """The photo download failure logs with exc_info=True and the user reply also fails:
    neither formatted log line (traceback included) nor the reply may carry the token."""
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    adapter = _make_connected_adapter()
    monkeypatch.setattr(adapter, "_is_user_authorized_from_message", lambda msg: True)
    monkeypatch.setattr(adapter, "_should_process_message", lambda msg: True)
    monkeypatch.setattr(adapter, "warning_notifications_enabled", lambda *a, **k: True)
    event = MessageEvent(text="", message_type=MessageType.PHOTO,
                         source=SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", user_id="88"))
    monkeypatch.setattr(adapter, "_build_message_event", lambda *a, **k: event)
    monkeypatch.setattr(adapter, "_apply_telegram_group_observe_attribution", lambda ev: ev)
    adapter.handle_message = AsyncMock()
    replies = []

    async def reply_text(text, **_kwargs):
        replies.append(text)
        raise _secret_error("sendMessage")

    photo = SimpleNamespace(get_file=AsyncMock(side_effect=_secret_error("getFile")))
    msg = SimpleNamespace(photo=[photo], caption=None, sticker=None, voice=None, audio=None, video=None,
                          document=None, media_group_id=None, reply_text=reply_text)

    with caplog.at_level("WARNING"):
        await adapter._handle_media_message(SimpleNamespace(message=msg, update_id=1), None)

    assert "Failed to cache photo" in caplog.text
    assert "Failed to notify user about photo cache failure" in caplog.text
    assert "Traceback" in caplog.text
    assert _SECRET_TOKEN not in caplog.text
    assert replies and all(_SECRET_TOKEN not in r for r in replies)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,label", [("voice", "voice message"), ("audio", "audio file"), ("video", "video file")])
async def test_inbound_av_cache_failure_redacts_traceback(caplog, kind, label):
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    adapter = _make_connected_adapter()
    adapter._telegram_media_size_allowed = lambda source, label: (True, None)
    event = MessageEvent(text="", message_type=MessageType.VOICE,
                         source=SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", user_id="88"))
    source = SimpleNamespace(get_file=AsyncMock(side_effect=_secret_error("getFile")), file_size=10)
    msg = SimpleNamespace(reply_text=AsyncMock())

    with caplog.at_level("WARNING"):
        await adapter._cache_inbound_av(msg, event, source, label, kind, ".ogg", "audio/ogg")

    assert f"Failed to cache {kind}" in caplog.text
    assert "Traceback" in caplog.text
    assert _SECRET_TOKEN not in caplog.text


@pytest.mark.asyncio
async def test_outbound_send_image_failures_redact_traceback(monkeypatch, caplog):
    """URL send and upload fallback both fail; both log with exc_info=True (the second
    with the raw exception as its message)."""
    import tools.url_safety

    adapter = _make_connected_adapter()
    monkeypatch.setattr(tools.url_safety, "is_safe_url", lambda url: True)
    monkeypatch.setattr(tools.url_safety, "create_ssrf_safe_async_client",
                        MagicMock(side_effect=_secret_error("sendPhoto")))
    monkeypatch.setattr(adapter, "_send_media", AsyncMock(side_effect=_secret_error("sendPhoto")))
    monkeypatch.setattr(BasePlatformAdapter, "send_image",
                        AsyncMock(return_value=SimpleNamespace(success=False, error="fallback")))

    with caplog.at_level("WARNING"):
        await adapter.send_image("123", "https://example.com/a.png")

    assert "URL-based send_photo failed" in caplog.text
    assert "File upload send_photo also failed" in caplog.text
    assert _SECRET_TOKEN not in caplog.text


@pytest.mark.asyncio
async def test_markdown_fallback_redacts_raw_error_in_log(caplog):
    adapter = _make_connected_adapter()
    sent = SimpleNamespace(message_id=7)
    adapter._bot.send_message = AsyncMock(side_effect=[
        RuntimeError(f"Can't parse entities at https://api.telegram.org/bot{_SECRET_TOKEN}/sendMessage"), sent])

    with caplog.at_level("WARNING"):
        result = await adapter._send_chunk_markdown_or_plain("*hi", {"chat_id": 123})

    assert result is sent
    assert "MarkdownV2 parse failed" in caplog.text
    assert _SECRET_TOKEN not in caplog.text


@pytest.mark.asyncio
async def test_turn_error_notice_does_not_send_raw_exception(monkeypatch, caplog):
    """_notify_turn_error must not forward raw exception text (here a bot URL) to the chat,
    nor log it raw when the notice itself fails."""
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    adapter = _make_connected_adapter()
    monkeypatch.setattr(adapter, "warning_notifications_enabled", lambda *a, **k: True)
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="1"))
    event = MessageEvent(text="hi", message_type=MessageType.TEXT,
                         source=SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", user_id="88"))

    await adapter._notify_turn_error(event, _secret_error("sendMessage"))

    content = adapter.send.await_args.kwargs["content"]
    assert "Sorry, I encountered an error (RuntimeError)" in content
    assert _SECRET_TOKEN not in content

    adapter.send = AsyncMock(side_effect=_secret_error("sendMessage"))
    with caplog.at_level("ERROR"):
        await adapter._notify_turn_error(event, _secret_error("sendMessage"))
    assert "Failed to send error notification to user" in caplog.text
    assert _SECRET_TOKEN not in caplog.text
