"""Tests for Google AI Studio (Gemini) provider integration."""

import pytest
from unittest.mock import patch, MagicMock

from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider, resolve_api_key_provider_credentials
from hermes_cli.models import _PROVIDER_MODELS, _PROVIDER_LABELS, _PROVIDER_ALIASES, normalize_provider
from hermes_cli.model_normalize import normalize_model_for_provider, detect_vendor
from agent.model_metadata import get_model_context_length
from agent.models_dev import PROVIDER_TO_MODELS_DEV, list_agentic_models, _NOISE_PATTERNS


# ── Provider Registry ──

class TestGeminiProviderRegistry:
    def test_gemini_in_registry(self):
        assert "gemini" in PROVIDER_REGISTRY

    def test_gemini_config(self):
        pconfig = PROVIDER_REGISTRY["gemini"]
        assert pconfig.id == "gemini"
        assert pconfig.name == "Google AI Studio"
        assert pconfig.auth_type == "api_key"
        assert pconfig.inference_base_url == "https://generativelanguage.googleapis.com/v1beta"


# ── Provider Aliases ──

PROVIDER_ENV_VARS = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_BASE_URL",
    "GLM_API_KEY", "ZAI_API_KEY", "KIMI_API_KEY",
    "MINIMAX_API_KEY", "DEEPSEEK_API_KEY",
)

@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestGeminiAliases:
    def test_explicit_gemini(self):
        assert resolve_provider("gemini") == "gemini"


    def test_models_py_aliases(self):
        assert _PROVIDER_ALIASES.get("google") == "gemini"
        assert _PROVIDER_ALIASES.get("google-gemini") == "gemini"
        assert _PROVIDER_ALIASES.get("google-ai-studio") == "gemini"

    def test_normalize_provider(self):
        assert normalize_provider("google") == "gemini"
        assert normalize_provider("gemini") == "gemini"
        assert normalize_provider("google-ai-studio") == "gemini"


# ── Auto-detection ──

class TestGeminiAutoDetection:
    def test_auto_detects_google_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        assert resolve_provider("auto") == "gemini"


    def test_google_api_key_priority_over_gemini(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "primary-key")
        monkeypatch.setenv("GEMINI_API_KEY", "alias-key")
        creds = resolve_api_key_provider_credentials("gemini")
        assert creds["api_key"] == "primary-key"
        assert creds["source"] == "GOOGLE_API_KEY"


# ── Credential Resolution ──

class TestGeminiCredentials:
    def test_resolve_with_google_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "google-secret")
        creds = resolve_api_key_provider_credentials("gemini")
        assert creds["provider"] == "gemini"
        assert creds["api_key"] == "google-secret"
        assert creds["base_url"] == "https://generativelanguage.googleapis.com/v1beta"

    def test_resolve_with_gemini_api_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
        creds = resolve_api_key_provider_credentials("gemini")
        assert creds["api_key"] == "gemini-secret"


class TestGeminiOutboundPolicy:
    @pytest.mark.parametrize("base_url", ("http://[", None, 7))
    def test_malformed_or_non_string_base_url_route_facts_are_non_matches(self, base_url):
        from agent.gemini_outbound_policy import is_gemini_outbound

        assert not is_gemini_outbound(
            canonical_provider="custom",
            model="ordinary-model",
            base_url=base_url,
            api_mode="chat_completions",
            routing_hint="ordinary",
        )

    def test_an_independent_gemini_route_fact_still_denies_malformed_base_url(self):
        from agent.gemini_outbound_policy import is_gemini_outbound

        assert is_gemini_outbound(
            canonical_provider="gemini",
            model="ordinary-model",
            base_url="http://[",
            api_mode="chat_completions",
            routing_hint="ordinary",
        )


# ── Model Catalog ──

class TestGeminiModelCatalog:
    def test_provider_entry_exists(self):
        """Gemini provider has a model catalog entry. Specific model names
        are data that changes with Google releases and don't belong in tests.
        """
        assert "gemini" in _PROVIDER_MODELS
        assert len(_PROVIDER_MODELS["gemini"]) >= 1


# ── Model Normalization ──

class TestGeminiModelNormalization:


    def test_gemma_vendor_detection(self):
        assert detect_vendor("gemma-4-31b-it") == "google"


    def test_gemma_aggregator_prepends_vendor(self):
        result = normalize_model_for_provider("gemma-4-31b-it", "openrouter")
        assert result == "google/gemma-4-31b-it"


# ── Context Length ──

class TestGeminiContextLength:
    def test_gemma_4_31b_context(self):
        # Mock external API lookups to test against hardcoded defaults
        # (models.dev and OpenRouter may return different values like 262144).
        with patch("agent.models_dev.lookup_models_dev_context", return_value=None), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}):
            ctx = get_model_context_length("gemma-4-31b-it", provider="gemini")
        assert ctx == 256000


# ── Agent Init (no SyntaxError) ──

class TestGeminiAgentInit:

    @pytest.mark.parametrize(
        ("provider", "model"),
        (("gemini", "gemini-2.5-flash"), ("vertex", "google/gemini-2.5-flash")),
    )
    def test_gemini_agent_uses_chat_completions(self, provider, model):
        """Direct Gemini/Vertex AIAgent construction denies before credentials."""
        from agent.gemini_outbound_policy import GeminiOutboundDenied
        from run_agent import AIAgent

        with (
            patch(
                "agent.credential_pool.credential_pool_matches_provider",
                side_effect=AssertionError("credential pool matcher reached"),
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                side_effect=AssertionError("auxiliary resolver reached"),
            ),
        ):
            with pytest.raises(GeminiOutboundDenied) as exc_info:
                AIAgent(provider=provider, model=model, credential_pool=object())

        assert exc_info.type is GeminiOutboundDenied
        assert exc_info.value.code == "gemini_outbound_denied"
        assert str(exc_info.value) == "Gemini outbound requests are disabled."



    @pytest.mark.parametrize(
        ("provider", "model", "route_kwargs"),
        (
            ("gemini", "ordinary-model", {}),
            ("custom", "gemini-2.5-flash", {}),
            ("custom", "ordinary-model", {"explicit_base_url": "https://generativelanguage.googleapis.com/v1beta/openai"}),
            ("custom", "ordinary-model", {"api_mode": "gemini_native"}),
            ("custom", "ordinary-model", {"main_runtime": {"requested_provider": "vertex"}}),
        ),
    )
    def test_gemini_resolve_provider_client_uses_native_client(self, provider, model, route_kwargs):
        """Google route facts deny before effects; a custom model spelling alone does not."""
        from agent.auxiliary_client import resolve_provider_client
        from agent.gemini_outbound_policy import GeminiOutboundDenied

        safe_custom_model = provider == "custom" and model == "gemini-2.5-flash" and not route_kwargs
        with (
            patch("agent.auxiliary_client._validate_proxy_env_urls", side_effect=None if safe_custom_model else AssertionError("proxy validation reached")) as mock_proxy,
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", side_effect=AssertionError("credential resolver reached")) as mock_credentials,
            patch("agent.auxiliary_client._get_provider_chain", return_value=[] if safe_custom_model else None),
            patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)),
            patch("agent.gemini_native_adapter.GeminiNativeClient", side_effect=AssertionError("native Gemini client reached")) as mock_native,
        ):
            if safe_custom_model:
                client, resolved = resolve_provider_client(provider, model, **route_kwargs)
            else:
                with pytest.raises(GeminiOutboundDenied) as exc_info:
                    resolve_provider_client(provider, model, **route_kwargs)

        if safe_custom_model:
            assert client is None
            assert resolved is None
            assert mock_proxy.called
        else:
            assert exc_info.type is GeminiOutboundDenied
            assert exc_info.value.code == "gemini_outbound_denied"
            assert str(exc_info.value) == "Gemini outbound requests are disabled."
            assert vars(exc_info.value) == {}
            mock_proxy.assert_not_called()
        mock_credentials.assert_not_called()
        mock_native.assert_not_called()

    def test_aiagent_gemini_fallback_denies_before_scoped_secret_read(self):
        """Primary-none fallback policy is decided before fallback key resolution."""
        from agent import auxiliary_client as aux
        from agent.gemini_outbound_policy import GeminiOutboundDenied
        from run_agent import AIAgent

        resolver_calls = []
        real_resolver = aux.resolve_provider_client

        class FallbackSecretReadReached(BaseException):
            pass

        def primary_none_then_real(provider, *args, **kwargs):
            resolver_calls.append(provider)
            if provider == "ordinary-primary":
                return None, None
            return real_resolver(provider, *args, **kwargs)

        with (
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                side_effect=primary_none_then_real,
            ),
            patch(
                "agent.secret_scope.get_secret",
                side_effect=FallbackSecretReadReached("fallback scoped secret read reached"),
            ) as secret_reader,
            patch(
                "agent.auxiliary_client._validate_proxy_env_urls",
                side_effect=AssertionError("proxy validation reached"),
            ) as proxy,
            patch(
                "agent.auxiliary_client._create_openai_client",
                side_effect=AssertionError("client construction reached"),
            ) as client_factory,
        ):
            with pytest.raises(GeminiOutboundDenied) as exc_info:
                AIAgent(
                    provider="ordinary-primary",
                    model="ordinary-model",
                    fallback_model=[
                        {
                            "provider": "gemini",
                            "model": "gemini-2.5-flash",
                            "key_env": "FORBIDDEN_FALLBACK_KEY",
                        }
                    ],
                    quiet_mode=True,
                )

        assert exc_info.type is GeminiOutboundDenied
        assert exc_info.value.code == "gemini_outbound_denied"
        assert str(exc_info.value) == "Gemini outbound requests are disabled."
        assert vars(exc_info.value) == {}
        secret_reader.assert_not_called()
        proxy.assert_not_called()
        client_factory.assert_not_called()
        assert resolver_calls == ["ordinary-primary"]


# ── models.dev Integration ──

class TestGeminiModelsDev:
    def test_gemini_mapped_to_google(self):
        assert PROVIDER_TO_MODELS_DEV.get("gemini") == "google"




    def test_list_agentic_models_with_mock_data(self):
        """list_agentic_models filters correctly from mock models.dev data."""
        mock_data = {
            "google": {
                "models": {
                    "gemini-3-flash-preview": {"tool_call": True},
                    "gemini-2.5-pro": {"tool_call": True},
                    "gemini-embedding-001": {"tool_call": False},
                    "gemini-2.5-flash-preview-tts": {"tool_call": False},
                    "gemini-live-2.5-flash": {"tool_call": True},
                    "gemini-2.5-flash-preview-04-17": {"tool_call": True},
                    "gemma-4-31b-it": {"tool_call": True},
                }
            }
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=mock_data):
            result = list_agentic_models("gemini")
        assert "gemini-3-flash-preview" in result
        assert "gemini-2.5-pro" in result
        assert "gemma-4-31b-it" not in result
        # Filtered out:
        assert "gemini-embedding-001" not in result      # no tool_call
        assert "gemini-2.5-flash-preview-tts" not in result  # no tool_call
        assert "gemini-live-2.5-flash" not in result     # noise: live-
        assert "gemini-2.5-flash-preview-04-17" not in result  # noise: dated preview

