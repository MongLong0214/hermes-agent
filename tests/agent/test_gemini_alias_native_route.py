"""Gemini provider aliases on a non-Google endpoint build a plain OpenAI-compatible client."""
from types import SimpleNamespace

from openai import OpenAI

from agent.agent_runtime_helpers import create_openai_client
from providers import get_provider_profile


def test_gemini_aliases_on_a_non_google_endpoint_build_an_openai_client():
    profile = get_provider_profile("gemini")
    for provider in (profile.name, *profile.aliases):
        agent = SimpleNamespace(
            provider=provider, model="gemini-flash-latest",
            _client_log_context=lambda: "test",
            _build_keepalive_http_client=lambda *a, **k: None,
        )
        client = create_openai_client(
            agent, {"api_key": "test-key", "base_url": "https://example.invalid/v1"},
            reason="fallback", shared=False,
        )
        try:
            assert isinstance(client, OpenAI), (provider, type(client))
        finally:
            client.close()
