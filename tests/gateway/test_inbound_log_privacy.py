"""The gateway's INFO ingress line identifies a turn without storing what the user wrote.

agent.log (and every watcher that reads it) receives INFO records, so the inbound line must carry
identifiers and lengths, never the message or the quoted reply text.
"""
import logging
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource

_PRIVATE = "zq-private-marker-7781 my bank PIN is 4242"
_QUOTED = "zq-quoted-marker-5519 the address you asked for"


@pytest.mark.asyncio
async def test_inbound_info_log_has_no_message_or_reply_text(caplog):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._hmwa_resolve_session = AsyncMock(return_value=None)  # stop right after the ingress line
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", user_id="88")
    event = MessageEvent(text=_PRIVATE, message_type=MessageType.TEXT, source=source, message_id="msg-7",
                         reply_to_message_id="41", reply_to_text=_QUOTED)

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        await runner._handle_message_with_agent(event, source, "agent:main:telegram:dm:123", 1)

    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("inbound message:")]
    assert len(lines) == 1
    assert "zq-private-marker" not in caplog.text and "zq-quoted-marker" not in caplog.text
    line = lines[0]
    assert "platform=telegram" in line and "chat=123" in line and "reply_to_id=41" in line
    assert f"msg_len={len(_PRIVATE)}" in line and f"reply_to_len={len(_QUOTED)}" in line
