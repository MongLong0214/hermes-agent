"""Tests for the TTS plugin picker surface in hermes_cli/tools_config.py (issue #30398).

Covers ``_plugin_tts_providers()`` and the ``_visible_providers()``
integration that injects plugin rows into the Text-to-Speech category.

Mirrors the structure of existing image_gen / browser picker tests.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent import tts_registry
from agent.tts_provider import TTSProvider
from hermes_cli import setup, tools_config


class _FakeTTSProvider(TTSProvider):
    def __init__(self, name: str, schema: dict | None = None):
        self._name = name
        self._schema = schema

    @property
    def name(self) -> str:
        return self._name

    def synthesize(self, text, output_path, **kw):
        return output_path

    def get_setup_schema(self):
        if self._schema is not None:
            return self._schema
        return super().get_setup_schema()


@pytest.fixture(autouse=True)
def _reset_registry():
    tts_registry._reset_for_tests()
    yield
    tts_registry._reset_for_tests()


class TestPluginTTSProviders:
    """``_plugin_tts_providers()`` returns picker-row dicts."""




    def test_skips_providers_with_no_name(self):
        """Defense in depth: a provider with no .name attribute is skipped
        rather than crashing the picker."""

        class _NoName:
            display_name = "Bogus"
            def get_setup_schema(self):
                return {"name": "Bogus"}

        tts_registry._providers["bogus"] = _NoName()  # type: ignore[assignment]
        try:
            rows = tools_config._plugin_tts_providers()
            # Provider has no .name so the picker filters it out
            assert all(r.get("tts_plugin_name") != "bogus" for r in rows)
        finally:
            tts_registry._providers.pop("bogus", None)  # type: ignore[arg-type]


    def test_minimal_schema_uses_display_name(self):
        """A provider with no setup_schema override gets a row built from
        ``display_name`` and ``name`` only."""
        tts_registry.register_provider(_FakeTTSProvider(name="minimal"))
        rows = tools_config._plugin_tts_providers()
        assert len(rows) == 1
        assert rows[0]["name"] == "Minimal"  # display_name default
        assert rows[0]["tts_provider"] == "minimal"
        assert rows[0]["env_vars"] == []



class TestVisibleProvidersInjectsTTSPlugins:
    """``_visible_providers()`` injects plugin rows into the Text-to-Speech
    category alongside the hardcoded built-in rows."""

    def test_tts_category_includes_plugin_rows(self):
        tts_registry.register_provider(_FakeTTSProvider(name="cartesia"))

        tts_cat = tools_config.TOOL_CATEGORIES["tts"]
        visible = tools_config._visible_providers(tts_cat, config={})

        names = [row.get("name") for row in visible]
        # Hardcoded rows (sample — check at least one is present)
        assert "Microsoft Edge TTS" in names
        # Plugin row injected at the end
        assert "Cartesia" in names

        # Plugin row has tts_provider key for write-path compat
        plugin_rows = [r for r in visible if r.get("tts_plugin_name")]
        assert len(plugin_rows) == 1
        assert plugin_rows[0]["tts_provider"] == "cartesia"

    def test_other_categories_unaffected_by_tts_plugins(self):
        """Registering a TTS plugin must not leak into the Image Generation
        or Browser pickers."""
        tts_registry.register_provider(_FakeTTSProvider(name="cartesia"))

        img_cat = tools_config.TOOL_CATEGORIES["image_gen"]
        visible = tools_config._visible_providers(img_cat, config={})
        names = [row.get("name") for row in visible]
        assert "Cartesia" not in names


def _disable_nous_tts_picker(monkeypatch):
    monkeypatch.setattr(setup, "managed_nous_tools_enabled", lambda: False)
    monkeypatch.setattr(
        setup,
        "get_nous_subscription_features",
        lambda _config: SimpleNamespace(nous_auth_present=False),
    )


def test_setup_tts_picker_excludes_gemini_before_persistence(monkeypatch):
    _disable_nous_tts_picker(monkeypatch)
    config = {"tts": {"provider": "gemini"}}
    captured = {}

    def _choose_keep(_prompt, choices, default):
        captured["choices"] = choices
        return default

    def _persist_stale_config(_config):
        raise AssertionError("stale Gemini config must not be persisted")

    monkeypatch.setattr(setup, "prompt_choice", _choose_keep)
    monkeypatch.setattr(setup, "save_config", _persist_stale_config)

    setup._setup_tts_provider(config)

    assert all("gemini" not in choice.lower() for choice in captured["choices"])
    assert config == {"tts": {"provider": "gemini"}}


def test_setup_tts_picker_preserves_non_google_provider_selection(monkeypatch):
    _disable_nous_tts_picker(monkeypatch)
    config = {"tts": {"provider": "edge"}}
    persisted = []

    def _choose_edge(_prompt, choices, _default):
        assert "Edge TTS (free, cloud-based, no setup needed)" in choices
        assert "OpenAI TTS (good quality, needs API key)" in choices
        assert all("gemini" not in choice.lower() for choice in choices)
        return choices.index("Edge TTS (free, cloud-based, no setup needed)")

    monkeypatch.setattr(setup, "prompt_choice", _choose_edge)
    monkeypatch.setattr(setup, "save_config", lambda value: persisted.append(deepcopy(value)))

    setup._setup_tts_provider(config)

    assert persisted == [{"tts": {"provider": "edge"}}]

