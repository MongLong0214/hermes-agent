"""The setup surfaces no longer offer Gemini TTS, and a stale Gemini TTS config shows as unavailable.

Google outbound speech is refused at runtime (tools.tts_tool), so the wizard must not ask for a
Gemini key or list Gemini, and the summary must not report a configured Gemini provider as ready.
"""

from types import SimpleNamespace

import pytest

from hermes_cli import setup_summary, setup_tts, tools_config


def _feats():
    return SimpleNamespace(tts=SimpleNamespace(managed_by_nous=False))


def test_the_tts_wizard_offers_no_gemini_choice_or_key_prompt():
    assert "gemini" not in [key for key, _label in setup_tts._TTS_PROVIDER_CHOICES]
    assert "gemini" not in setup_tts._TTS_API_KEY_PROVIDERS
    assert "gemini" not in setup_tts._TTS_PROVIDER_LABELS


def test_the_tools_picker_lists_no_gemini_tts_row():
    rows = tools_config.TOOL_CATEGORIES["tts"]["providers"]
    assert [row for row in rows if row.get("tts_provider") == "gemini"] == []
    assert "GEMINI_API_KEY" not in repr(rows)


def test_a_stale_gemini_tts_config_is_summarized_as_unavailable_without_a_key_check(monkeypatch):
    checked = []
    monkeypatch.setattr("hermes_cli.setup.cfg_get", lambda config, *keys, default=None: "gemini")
    monkeypatch.setattr("hermes_cli.setup.get_env_value", lambda name: checked.append(name) or "key")

    label, ok, hint = setup_summary._tts_row({}, _feats())

    assert (label, ok) == ("Text-to-Speech (Unavailable provider)", False)
    assert hint == "run 'hermes setup tts'"
    assert checked == []


def test_an_ordinary_tts_provider_is_summarized_as_before(monkeypatch):
    monkeypatch.setattr("hermes_cli.setup.cfg_get", lambda config, *keys, default=None: "openai")
    monkeypatch.setattr("hermes_cli.setup.get_env_value", lambda name: "key" if name == "OPENAI_API_KEY" else "")

    assert setup_summary._tts_row({}, _feats()) == ("Text-to-Speech (OpenAI)", True, None)


@pytest.mark.parametrize(("current", "label"), [
    ("gemini", "Unavailable provider (choose another)"),
    ("openai", "OpenAI TTS"),
    ("some-plugin", "some-plugin"),
])
def test_the_wizard_labels_the_current_provider(monkeypatch, current, label):
    shown = []
    monkeypatch.setattr("hermes_cli.setup.print_header", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.setup._info", lambda text, *a, **k: shown.append(text))
    monkeypatch.setattr("hermes_cli.setup.prompt_choice", lambda prompt, choices, default: default)
    monkeypatch.setattr(setup_tts.tool_backend_helpers, "managed_nous_tools_enabled", lambda: False)

    setup_tts._setup_tts_provider({"tts": {"provider": current}})

    assert shown[0] == f"Current: {label}"
