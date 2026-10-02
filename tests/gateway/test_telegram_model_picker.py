"""Tests for Telegram model picker thread fallback."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class TestTelegramModelPicker:
    @pytest.mark.asyncio
    async def test_send_model_picker_escapes_dynamic_provider_label(self):
        adapter = _make_adapter()
        sent = {}

        async def mock_send_message(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(message_id=101)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        result = await adapter.send_model_picker(
            chat_id="12345",
            providers=[
                {"slug": "provider_one", "name": "Provider One", "total_models": 1, "is_current": True}
            ],
            current_model="model_1",
            current_provider="provider_one",
            session_key="s",
            on_model_selected=AsyncMock(),
            metadata={"thread_id": "99999"},
        )

        assert result.success is True
        assert "MARKDOWN_V2" in repr(sent["parse_mode"])
        assert "provider\\_one" in sent["text"]
        assert "`model_1`" in sent["text"]

    @pytest.mark.asyncio
    async def test_back_button_escapes_dynamic_provider_label(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")  # picker taps are auth-gated
        adapter = _make_adapter()
        adapter._model_picker_state["12345"] = {
            "providers": [{"slug": "provider_one", "name": "Provider One", "total_models": 1, "is_current": True}],
            "current_model": "model_1",
            "current_provider": "provider_one",
            "session_key": "s",
            "on_model_selected": AsyncMock(),
            "msg_id": 42,
        }

        query = AsyncMock()
        query.data = "mb"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = SimpleNamespace(id=111, first_name="Operator")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        await adapter._handle_model_picker_callback(query, "mb", "12345")

        edit_kwargs = query.edit_message_text.call_args[1]
        assert "MARKDOWN_V2" in repr(edit_kwargs["parse_mode"])
        assert "provider\\_one" in edit_kwargs["text"]
        assert "`model_1`" in edit_kwargs["text"]




def _picker_state(on_model_selected):
    return {
        "providers": [{"slug": "provider_one", "name": "Provider One", "total_models": 1, "is_current": True,
                       "models": ["model_1", "model_2"]}],
        "current_model": "model_1", "current_provider": "provider_one", "session_key": "s",
        "on_model_selected": on_model_selected, "msg_id": 42,
        "selected_provider": "provider_one", "model_list": ["model_1", "model_2"],
    }


def _group_tap(data: str, user_id: int):
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat_id = -1001234
    query.message.chat.type = "supergroup"
    query.message.message_thread_id = None
    query.from_user = SimpleNamespace(id=user_id, first_name="Member")
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    return SimpleNamespace(callback_query=query), query


@pytest.mark.asyncio
@pytest.mark.parametrize("data", ["mm:1", "mc:1", "mp:provider_one", "mx"])
async def test_unauthorized_group_member_cannot_drive_model_picker(monkeypatch, data):
    """A group member outside the allowlist who can see the picker must not switch the model or touch
    picker state; the choice picker and approval buttons apply the same gate."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")
    monkeypatch.setattr("hermes_cli.model_selection_guards.combined_selection_warning", lambda *a, **k: None)
    adapter = _make_adapter()
    on_model_selected = AsyncMock(return_value="switched")
    adapter._model_picker_state["-1001234"] = state = _picker_state(on_model_selected)
    snapshot = dict(state)

    update, query = _group_tap(data, user_id=999)
    await adapter._handle_callback_query(update, None)

    on_model_selected.assert_not_awaited()
    assert adapter._model_picker_state.get("-1001234") is state and state == snapshot
    query.edit_message_text.assert_not_awaited()
    query.answer.assert_awaited_once()

    # The allowlisted operator's tap on the same picker still switches.
    if data in ("mm:1", "mc:1"):
        update, query = _group_tap(data, user_id=111)
        await adapter._handle_callback_query(update, None)
        on_model_selected.assert_awaited_once_with("-1001234", "model_2", "provider_one")
