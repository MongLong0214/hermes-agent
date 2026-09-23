"""Fail-closed main fallback admission for named custom provider routes."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import FailoverReason
from run_agent import AIAgent


def _agent_with_chain(chain):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="primary-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=chain,
        )
    agent.client = MagicMock()
    return agent


def _client():
    client = MagicMock()
    client.base_url = "https://inference-api.nousresearch.com/v1"
    client.api_key = "nous-key"
    return client


def _run_runtime_and_init(chain, config, secret_reads):
    """Run both admission paths and return their observable resolver effects."""
    runtime = _agent_with_chain(chain)
    runtime_cache_keys, runtime_key_reads, runtime_clients = [], [], []

    def runtime_key(entry):
        runtime_key_reads.append(entry["provider"])
        return "fallback-key"

    def runtime_client(provider, model=None, **_kwargs):
        runtime_clients.append((provider, model))
        return _client(), model

    with (
        patch("hermes_cli.runtime_provider.load_config", return_value=config),
        patch("hermes_cli.runtime_provider_custom.get_secret_str", side_effect=lambda *args: secret_reads.append(args[0]) or "secret"),
        patch("agent.chat_completion_helpers._fallback_entry_key", side_effect=lambda entry: runtime_cache_keys.append(entry["provider"]) or (entry["provider"], entry["model"])),
        patch("agent.chat_completion_helpers._should_skip_fallback_candidate", return_value=False),
        patch("hermes_cli.fallback_config.resolve_entry_api_key", side_effect=runtime_key),
        patch("agent.auxiliary_client.resolve_provider_client", side_effect=runtime_client),
        patch("agent.chat_completion_helpers._rebind_fallback_credential_pool"),
    ):
        assert runtime._try_activate_fallback(FailoverReason.rate_limit) is True

    init = SimpleNamespace(provider="primary", model="primary-model")
    init_key_reads, init_clients = [], []

    def init_key(entry):
        init_key_reads.append(entry["provider"])
        return "fallback-key"

    def init_client(provider, model=None, **_kwargs):
        init_clients.append((provider, model))
        return (None, None) if provider == "primary" else (_client(), model)

    from agent.agent_init import _routed_client_kwargs

    with (
        patch("hermes_cli.runtime_provider.load_config", return_value=config),
        patch("hermes_cli.runtime_provider_custom.get_secret_str", side_effect=lambda *args: secret_reads.append(args[0]) or "secret"),
        patch("hermes_cli.fallback_config.resolve_entry_api_key", side_effect=init_key),
        patch("agent.auxiliary_client.resolve_provider_client", side_effect=init_client),
    ):
        kwargs = _routed_client_kwargs(init, chain, None)

    assert runtime.provider == init.provider == "nous"
    assert kwargs == {
        "api_key": "nous-key",
        "base_url": "https://inference-api.nousresearch.com/v1",
        "default_headers": {},
    }
    return runtime_cache_keys, runtime_key_reads, runtime_clients, init_key_reads, init_clients


def test_named_google_and_vertex_custom_routes_are_rejected_before_secret_client_or_cache():
    """Configured named Google routes never enter fallback resolution in either path."""
    config = {
        "providers": {
            "google-relay": {
                "base_url": "https://us-central1-aiplatform.googleapis.com/v1",
                "key_env": "GOOGLE_RELAY_KEY",
            },
            "vertex-relay": {
                "base_url": "https://vertexai.googleapis.com/v1",
                "key_env": "VERTEX_RELAY_KEY",
            },
        },
    }
    chain = [
        {"provider": "google-relay", "model": "ordinary-model"},
        {"provider": "vertex-relay", "model": "ordinary-model"},
        {"provider": "nous", "model": "Hermes-4-70B"},
    ]
    secret_reads = []

    effects = _run_runtime_and_init(chain, config, secret_reads)

    assert secret_reads == []
    assert effects == (
        ["nous"], ["nous"], [("nous", "Hermes-4-70B")],
        ["nous"], [("primary", "primary-model"), ("nous", "Hermes-4-70B")],
    )


def test_unknown_named_route_fails_closed_but_benign_named_custom_remains_resolvable():
    """Only a declared non-Google named route reaches normal fallback resolution."""
    config = {
        "providers": {
            "unknown-relay": {
                "name": "unknown-relay",
                "key_env": "UNKNOWN_RELAY_KEY",
            },
            "benign-relay": {
                "name": "benign-relay",
                "base_url": "https://gateway.example.test/v1",
                "key_env": "BENIGN_RELAY_KEY",
            },
        },
    }
    unknown_chain = [
        {"provider": "unknown-relay", "model": "ordinary-model"},
        {"provider": "nous", "model": "Hermes-4-70B"},
    ]
    unknown_reads = []

    effects = _run_runtime_and_init(unknown_chain, config, unknown_reads)

    assert unknown_reads == []
    assert effects == (
        ["nous"], ["nous"], [("nous", "Hermes-4-70B")],
        ["nous"], [("primary", "primary-model"), ("nous", "Hermes-4-70B")],
    )

    from hermes_cli.runtime_provider import _get_named_custom_provider
    from hermes_cli.runtime_provider_custom import peek_named_custom_provider_route

    secret_reads = []
    with (
        patch("hermes_cli.runtime_provider.load_config", return_value=config),
        patch("hermes_cli.runtime_provider_custom.get_secret_str", side_effect=lambda *args: secret_reads.append(args[0]) or "secret"),
    ):
        assert peek_named_custom_provider_route("benign-relay") == {
            "base_url": "https://gateway.example.test/v1",
        }
        assert peek_named_custom_provider_route("unknown-relay") == {"base_url": ""}
        assert _get_named_custom_provider("benign-relay")["base_url"] == "https://gateway.example.test/v1"

    assert secret_reads == ["BENIGN_RELAY_KEY"]


@pytest.mark.parametrize("bad_url", [
    "https://relay.example.test:65536/v1",
    "https://relay.example.test:not-a-port/v1",
    "https://[",
])
def test_malformed_custom_fallback_urls_skip_before_runtime_or_init_resolution(bad_url):
    """Bad custom routes never reach cache, credential, or client resolution."""
    chain = [
        {"provider": "custom", "model": "ordinary-model", "base_url": bad_url},
        {"provider": "nous", "model": "Hermes-4-70B"},
    ]
    secret_reads = []

    effects = _run_runtime_and_init(chain, {}, secret_reads)

    assert secret_reads == []
    assert effects == (
        ["nous"], ["nous"], [("nous", "Hermes-4-70B")],
        ["nous"], [("primary", "primary-model"), ("nous", "Hermes-4-70B")],
    )


def test_named_route_peek_matches_resolver_endpoint_precedence_without_secret_reads():
    """A route-only lookup skips endpoint-less aliases exactly as the resolver does."""
    from hermes_cli.runtime_provider import _get_named_custom_provider
    from hermes_cli.runtime_provider_custom import peek_named_custom_provider_route

    config = {
        "providers": {
            "first": {"name": "shared-relay", "key_env": "FIRST_KEY"},
            "later": {
                "name": "shared-relay",
                "base_url": "https://later.example.test/v1",
                "key_env": "LATER_KEY",
            },
        },
        "custom_providers": [
            {
                "name": "shared-relay",
                "base_url": "https://legacy.example.test/v1",
                "key_env": "LEGACY_KEY",
            },
        ],
    }
    secret_reads = []

    with (
        patch("hermes_cli.runtime_provider.load_config", return_value=config),
        patch("hermes_cli.runtime_provider_custom.get_secret_str", side_effect=lambda *args: secret_reads.append(args[0]) or "secret"),
    ):
        assert peek_named_custom_provider_route("shared-relay") == {
            "base_url": "https://later.example.test/v1",
        }
        assert secret_reads == []
        resolved = _get_named_custom_provider("shared-relay")
        assert resolved is not None
        assert resolved["base_url"] == "https://later.example.test/v1"

    assert secret_reads == ["LATER_KEY"]

    legacy_config = {
        "providers": {"first": {"name": "shared-relay", "key_env": "FIRST_KEY"}},
        "custom_providers": [{
            "name": "shared-relay",
            "base_url": "https://legacy.example.test/v1",
            "key_env": "LEGACY_KEY",
        }],
    }
    secret_reads = []
    with (
        patch("hermes_cli.runtime_provider.load_config", return_value=legacy_config),
        patch("hermes_cli.runtime_provider_custom.get_secret_str", side_effect=lambda *args: secret_reads.append(args[0]) or "secret"),
    ):
        assert peek_named_custom_provider_route("shared-relay") == {
            "base_url": "https://legacy.example.test/v1",
        }
        assert secret_reads == []
        resolved = _get_named_custom_provider("shared-relay")
        assert resolved is not None
        assert resolved["base_url"] == "https://legacy.example.test/v1"

    assert secret_reads == []

    no_endpoint_config = {
        "providers": {"first": {"name": "shared-relay", "key_env": "FIRST_KEY"}},
        "custom_providers": [{"name": "shared-relay", "key_env": "LEGACY_KEY"}],
    }
    with patch("hermes_cli.runtime_provider.load_config", return_value=no_endpoint_config):
        assert peek_named_custom_provider_route("shared-relay") == {"base_url": ""}
        assert _get_named_custom_provider("shared-relay") is None
