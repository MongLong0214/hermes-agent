"""Gemini provider aliases: a Google endpoint is refused before any client is built, and a non-Google
endpoint builds a plain OpenAI-compatible client."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from openai import OpenAI

from agent.agent_runtime_helpers import create_openai_client
from agent.gemini_outbound_policy import GeminiOutboundDenied
from providers import get_provider_profile


def _agent(provider):
    return SimpleNamespace(
        provider=provider, model="gemini-flash-latest",
        _client_log_context=lambda: "test",
        _build_keepalive_http_client=lambda *a, **k: None,
    )


def test_gemini_aliases_refuse_the_google_endpoint_before_any_client():
    profile = get_provider_profile("gemini")
    for provider in (profile.name, *profile.aliases):
        with patch("agent.gemini_native_adapter.GeminiNativeClient") as native_client, \
             patch("agent.process_bootstrap.OpenAI") as openai_client:
            with pytest.raises(GeminiOutboundDenied):
                create_openai_client(
                    _agent(provider), {"api_key": "test-key", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
                    reason="fallback", shared=False,
                )
        native_client.assert_not_called()
        openai_client.assert_not_called()


def test_gemini_aliases_on_a_non_google_endpoint_build_an_openai_client():
    profile = get_provider_profile("gemini")
    for provider in (profile.name, *profile.aliases):
        client = create_openai_client(
            _agent(provider), {"api_key": "test-key", "base_url": "https://example.invalid/v1"},
            reason="fallback", shared=False,
        )
        try:
            assert isinstance(client, OpenAI), (provider, type(client))
        finally:
            client.close()
