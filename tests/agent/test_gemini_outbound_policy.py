"""Contracts for the Gemini / Google AI Studio / Vertex outbound default-deny.

The policy classifies a route from its non-secret facts only. The runtime resolver refuses a
Google-bound route before any rung reads a credential, skips a Google fallback entry before its key
is read, and keeps resolving every non-Google route (the default openai-codex route included).
"""

from types import SimpleNamespace

import pytest

from agent.gemini_outbound_policy import GeminiOutboundDenied, deny_gemini_outbound, is_gemini_outbound
from hermes_cli import runtime_provider
from hermes_cli.auth import AuthError
from tools.delegate_tool_config import _runtime_provider_credentials

STUDIO = "https://generativelanguage.googleapis.com/v1beta"
VERTEX = "https://us-central1-aiplatform.googleapis.com/v1"
ORDINARY = "https://ordinary.example.test/v1"


class _Tripwire:
    """Stands in for a credential read, pool, client constructor or agent: any use is recorded."""

    def __init__(self, name, touched):
        self._name, self._touched = name, touched

    def __call__(self, *_args, **_kwargs):
        self._touched.append(self._name)
        raise AssertionError(f"{self._name} was reached before the refusal")

    def __getattr__(self, attr):
        self._touched.append(f"{self._name}.{attr}")
        raise AssertionError(f"{self._name}.{attr} was read before the refusal")


# ── the classifier ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("route", [
    pytest.param({"canonical_provider": "gemini"}, id="provider-gemini"),
    pytest.param({"canonical_provider": "Google-AI-Studio"}, id="provider-alias-case"),
    pytest.param({"canonical_provider": "vertex"}, id="provider-vertex"),
    pytest.param({"canonical_provider": "gcp-vertex"}, id="provider-vertex-alias"),
    pytest.param({"routing_hint": "google"}, id="routing-hint"),
    pytest.param({"model": "gemini-2.5-flash"}, id="model-bare"),
    pytest.param({"model": "google/gemini-3-pro-preview"}, id="model-vendor-prefixed"),
    pytest.param({"model": "models:gemini_exp"}, id="model-colon-segment"),
    pytest.param({"model": "nano-banana-pro"}, id="model-nano-banana"),
    pytest.param({"base_url": STUDIO}, id="host-ai-studio"),
    pytest.param({"base_url": VERTEX}, id="host-vertex-regional-dash"),
    pytest.param({"base_url": "https://europe-west4.aiplatform.googleapis.com/v1"}, id="host-vertex-regional-dot"),
    pytest.param({"base_url": "generativelanguage.googleapis.com/v1beta"}, id="host-without-scheme"),
    pytest.param({"base_url": "https://GENERATIVELANGUAGE.googleapis.com./v1"}, id="host-case-trailing-dot"),
    pytest.param({"api_mode": "gemini_native"}, id="api-mode"),
])
def test_any_google_signal_is_denied(route):
    assert is_gemini_outbound(**route) is True
    with pytest.raises(GeminiOutboundDenied) as caught:
        deny_gemini_outbound(**route)
    assert str(caught.value) == GeminiOutboundDenied.public_message
    assert caught.value.code == "gemini_outbound_denied"


@pytest.mark.parametrize("route", [
    pytest.param({"canonical_provider": "openai-codex", "model": "gpt-6.1-sol",
                  "base_url": "https://chatgpt.com/backend-api/codex", "api_mode": "codex_responses"},
                 id="default-route"),
    pytest.param({"canonical_provider": "custom", "model": "ordinary-model", "base_url": ORDINARY}, id="custom"),
    pytest.param({"canonical_provider": "anthropic", "model": "claude-sonnet-4"}, id="anthropic"),
    pytest.param({"model": "gemma-3-27b"}, id="gemma-is-not-gemini"),
    pytest.param({"model": "my-gemini"}, id="gemini-only-mid-segment"),
    pytest.param({"base_url": "https://www.googleapis.com/oauth2/v4/token"}, id="non-inference-google-host"),
    pytest.param({"base_url": "https://evil-generativelanguage.googleapis.com.example/v1"}, id="lookalike-host"),
    pytest.param({}, id="no-facts"),
])
def test_non_google_routes_are_allowed(route):
    assert is_gemini_outbound(**route) is False
    deny_gemini_outbound(**route)


def test_endpoint_authority_lets_the_concrete_endpoint_decide():
    # A boundary holding the selected endpoint: google/gemini-* served by a relay stays allowed ...
    assert is_gemini_outbound(model="google/gemini-2.5-flash", base_url="https://openrouter.ai/api/v1",
                              endpoint_authority=True) is False
    # ... while the Google host itself is refused whatever the model is.
    assert is_gemini_outbound(model="ordinary-model", base_url=VERTEX, endpoint_authority=True) is True
    # Without an endpoint, provider / hint / api_mode decide and the model name is not consulted ...
    assert is_gemini_outbound(canonical_provider="openrouter", model="google/gemini-2.5-flash",
                              endpoint_authority=True) is False
    assert is_gemini_outbound(canonical_provider="gemini", endpoint_authority=True) is True
    # ... and with none of those, the default classification applies.
    assert is_gemini_outbound(model="gemini-2.5-flash", endpoint_authority=True) is True


def test_the_denial_carries_no_route_detail():
    exc = GeminiOutboundDenied()
    assert exc.args == (GeminiOutboundDenied.public_message,)
    assert "googleapis" not in str(exc) and "gemini-" not in str(exc)


# ── the runtime resolver ───────────────────────────────────────────────────────────────────

# (tripwired seams, config/stand-in patches, the boundary call)
_DENIED_ROUTES = [
    pytest.param(("hermes_cli.runtime_provider._ladder_rungs",), {},
                 lambda: runtime_provider.resolve_runtime_provider(requested="gemini"),
                 id="primary-resolver"),
    pytest.param(("hermes_cli.runtime_provider._ladder_rungs",), {},
                 lambda: runtime_provider.resolve_runtime_provider(requested="openrouter", target_model="gemini-2.5-pro"),
                 id="primary-gemini-model"),
    pytest.param(("hermes_cli.runtime_provider._ladder_rungs",), {},
                 lambda: runtime_provider.resolve_runtime_provider(requested="custom", explicit_base_url=VERTEX,
                                                                   explicit_api_key="k"),
                 id="primary-explicit-google-endpoint"),
    pytest.param(("hermes_cli.runtime_provider._ladder_rungs",),
                 {"hermes_cli.runtime_provider._get_model_config":
                  lambda: {"provider": "auto", "default": "gemini-2.5-flash"}},
                 lambda: runtime_provider.resolve_runtime_provider(requested="auto"),
                 id="auto-configured-gemini-default"),
    pytest.param(("hermes_cli.runtime_provider_custom.get_secret_str",),
                 {"hermes_cli.runtime_provider.load_config":
                  lambda: {"providers": {"relay": {"base_url": STUDIO, "key_env": "RELAY_TEST_KEY"}}}},
                 lambda: runtime_provider.resolve_runtime_provider(requested="custom:relay"),
                 id="primary-named-custom-endpoint"),
    pytest.param(("hermes_cli.runtime_provider_custom.get_secret_str",),
                 {"hermes_cli.runtime_provider.load_config":
                  lambda: {"providers": {"relay": {"base_url": STUDIO, "key_env": "RELAY_TEST_KEY"}},
                           "custom_providers": [{"name": "relay", "base_url": ORDINARY}]}},
                 lambda: runtime_provider.resolve_runtime_provider(requested="relay"),
                 id="named-custom-providers-before-legacy"),
    pytest.param(("hermes_cli.runtime_provider._ladder_rungs",), {},
                 lambda: _runtime_provider_credentials({"provider": "vertex", "model": "gemini-2.5-flash"}, None),
                 id="delegation"),
]


def _refusal(call) -> str:
    """The refusal a boundary produced: the typed denial, also when a caller chains it."""
    try:
        result = call()
    except Exception as exc:
        denial = exc if isinstance(exc, GeminiOutboundDenied) else exc.__cause__
        return str(denial) if isinstance(denial, GeminiOutboundDenied) else f"not a denial: {exc!r}"
    return f"not refused: {result!r}"


@pytest.mark.parametrize(("seams", "patches", "call"), _DENIED_ROUTES)
def test_google_bound_resolution_is_refused_before_any_rung(seams, patches, call, monkeypatch):
    touched = []
    for seam in seams:
        monkeypatch.setattr(seam, _Tripwire(seam, touched))
    for target, value in patches.items():
        monkeypatch.setattr(target, value)

    assert _refusal(call) == GeminiOutboundDenied.public_message
    assert touched == []


def test_auto_picking_a_google_provider_is_refused_before_its_runtime(monkeypatch):
    touched = []
    monkeypatch.setattr(runtime_provider, "_get_model_config", dict)
    monkeypatch.setattr(runtime_provider, "_resolve_requested_shortcuts", lambda *a: None)
    monkeypatch.setattr(runtime_provider, "_named_custom_rung", lambda *a: None)
    monkeypatch.setattr(runtime_provider, "_local_endpoint_bypass", lambda *a: None)
    monkeypatch.setattr(runtime_provider, "resolve_provider", lambda *a, **k: "gemini")
    monkeypatch.setattr(runtime_provider, "_resolve_explicit_runtime", _Tripwire("explicit_runtime", touched))
    monkeypatch.setattr(runtime_provider, "load_pool", _Tripwire("load_pool", touched))

    with pytest.raises(GeminiOutboundDenied):
        runtime_provider.resolve_runtime_provider(requested="auto")
    assert touched == []


def test_a_google_hosted_pool_entry_is_refused_before_its_key_is_read(monkeypatch):
    touched = []
    entry = SimpleNamespace(runtime_base_url=VERTEX)
    pool = SimpleNamespace(has_credentials=lambda: True, select=lambda model=None: entry)
    monkeypatch.setattr(runtime_provider, "load_pool", lambda provider: pool)
    monkeypatch.setattr(runtime_provider, "_pool_entry_api_key", _Tripwire("pool_entry_api_key", touched))

    with pytest.raises(GeminiOutboundDenied):
        runtime_provider._resolve_from_pool("custom", "custom", {}, None, None, None)
    assert touched == []


def test_a_rung_that_lands_on_a_google_host_is_refused_after_the_ladder(monkeypatch):
    monkeypatch.setattr(runtime_provider, "_get_model_config", dict)
    monkeypatch.setattr(runtime_provider, "_ladder_rungs", lambda *a: iter([{
        "provider": "custom", "requested_provider": "relay", "model": "ordinary-model", "base_url": STUDIO,
        "api_key": "k", "api_mode": "chat_completions", "source": "test"}]))

    with pytest.raises(GeminiOutboundDenied):
        runtime_provider.resolve_runtime_provider(requested="relay")


def test_a_google_fallback_entry_is_skipped_before_its_key_is_read(monkeypatch):
    from hermes_cli import fallback_config

    keys_read, requested = [], []
    google_entries = [{"provider": "gemini", "model": "gemini-2.5-flash"},
                      {"provider": "openrouter", "model": "google/gemini-2.5-pro"},
                      {"provider": "custom", "model": "ordinary-model", "base_url": VERTEX}]
    ordinary = {"provider": "custom", "model": "ordinary-model", "base_url": ORDINARY}

    def _resolve(**kwargs):
        requested.append(kwargs.get("requested"))
        if len(requested) == 1:
            raise AuthError("primary down", provider="openai-codex", code="missing_api_key")
        return {"provider": "custom", "base_url": kwargs.get("explicit_base_url"), "api_key": "k"}

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", _resolve)
    monkeypatch.setattr(fallback_config, "get_fallback_chain", lambda config: [*google_entries, ordinary])
    monkeypatch.setattr(fallback_config, "resolve_entry_api_key", lambda entry: keys_read.append(entry) or None)
    monkeypatch.setattr(fallback_config, "effective_runtime_provider", lambda entry, runtime: runtime["provider"])

    runtime, entry = runtime_provider.resolve_runtime_with_fallback({}, requested="openai-codex")

    assert entry is ordinary and runtime["base_url"] == ORDINARY
    assert keys_read == [ordinary]
    assert requested == ["openai-codex", "custom"]


def test_a_stale_google_model_config_does_not_block_switching_away(monkeypatch):
    monkeypatch.setattr(runtime_provider, "_get_model_config",
                        lambda: {"provider": "gemini", "default": "gemini-2.5-pro", "base_url": STUDIO,
                                 "api_mode": "gemini_native"})

    resolved = runtime_provider.resolve_runtime_provider(requested="openrouter", explicit_api_key="k")

    assert resolved["provider"] == "openrouter"
    assert "googleapis" not in resolved["base_url"]


def test_non_google_endpoint_resolves_and_its_google_host_twin_is_refused():
    def build(url):
        return runtime_provider.resolve_runtime_provider(
            requested="custom", explicit_base_url=url, explicit_api_key="k", target_model="ordinary-model")["base_url"]

    assert build(ORDINARY).rstrip("/") == ORDINARY
    with pytest.raises(GeminiOutboundDenied):
        build(STUDIO)


def test_a_named_custom_provider_on_an_ordinary_host_still_resolves(monkeypatch):
    monkeypatch.setattr(runtime_provider, "load_config",
                        lambda: {"providers": {"relay": {"base_url": ORDINARY, "api_key": "k"}}})

    resolved = runtime_provider.resolve_runtime_provider(requested="custom:relay")

    assert resolved["base_url"].rstrip("/") == ORDINARY


def test_the_default_openai_codex_route_is_not_refused(monkeypatch):
    runtime = {"provider": "openai-codex", "requested_provider": "openai-codex", "model": "gpt-6.1-sol",
               "base_url": "https://chatgpt.com/backend-api/codex", "api_key": "tok",
               "api_mode": "codex_responses", "source": "test"}
    monkeypatch.setattr(runtime_provider, "_get_model_config",
                        lambda: {"provider": "openai-codex", "default": "gpt-6.1-sol"})
    monkeypatch.setattr(runtime_provider, "_ladder_rungs", lambda *a: iter([dict(runtime)]))

    resolved = runtime_provider.resolve_runtime_provider(requested="openai-codex")

    assert (resolved["provider"], resolved["model"], resolved["base_url"]) == (
        "openai-codex", "gpt-6.1-sol", "https://chatgpt.com/backend-api/codex")
