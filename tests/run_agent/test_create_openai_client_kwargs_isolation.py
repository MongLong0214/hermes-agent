"""Guardrail: _create_openai_client must not mutate its input kwargs.

#10933 injected an httpx.Client directly into the caller's ``client_kwargs``.
When the dict was ``self._client_kwargs``, the shared transport was torn down
after the first request_complete close and subsequent request-scoped clients
wrapped a closed transport, raising ``APIConnectionError('Connection error.')``
with cause ``RuntimeError: Cannot send a request, as the client has been closed``
on every retry. That PR has since been reverted, but the underlying issue
(#10324, connections hanging in CLOSE-WAIT) is still open, so another transport
tweak inside this function is likely. This test pins the contract that the
function must treat its input dict as read-only.
"""
from unittest.mock import MagicMock, patch

import pytest

from agent.gemini_outbound_policy import GeminiOutboundDenied
from run_agent import AIAgent


class _SideEffectReached(BaseException):
    """Prohibited TLS/proxy/client/cache effect before default-deny."""


class _LiveClient:
    is_closed = False


_GOOGLE_INFERENCE_URLS = (
    "https://generativelanguage.googleapis.com/v1beta",
    "https://aiplatform.googleapis.com/v1",
    "https://vertexai.googleapis.com/v1",
    "https://us-central1-aiplatform.googleapis.com/v1",
    "https://generativelanguage.googleapis.com.:443/v1beta",
)

_PROVIDER_ALIASES = (
    "gemini",
    "google",
    "google-gemini",
    "google-ai-studio",
    "vertex",
    "vertexai",
    "google-vertex",
    "vertex-ai",
    "gcp-vertex",
)


def _assert_stable_denial(exc_info):
    assert exc_info.type is GeminiOutboundDenied
    assert type(exc_info.value) is GeminiOutboundDenied
    assert exc_info.value.code == "gemini_outbound_denied"
    assert str(exc_info.value) == "Gemini outbound requests are disabled."
    assert vars(exc_info.value) == {}


def _bare_agent(**fields):
    agent = AIAgent.__new__(AIAgent)
    agent.provider = "custom"
    agent.requested_provider = "custom"
    agent.model = "ordinary-model"
    agent.api_mode = "chat_completions"
    agent.base_url = "https://api.example.test/v1"
    agent.api_key = "placeholder-key"
    agent.client = None
    agent._client_kwargs = {
        "api_key": "placeholder-key",
        "base_url": "https://api.example.test/v1",
    }
    for name, value in fields.items():
        setattr(agent, name, value)
    return agent


def _explode(*_args, **_kwargs):
    raise _SideEffectReached("prohibited Gemini/Vertex side effect reached")


@patch("run_agent.OpenAI")
def test_create_openai_client_does_not_mutate_input_kwargs(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    kwargs = {"api_key": "test-key", "base_url": "https://api.example.com/v1"}
    snapshot = dict(kwargs)

    agent._create_openai_client(kwargs, reason="test", shared=False)

    assert kwargs == snapshot, (
        f"_create_openai_client mutated input kwargs; expected {snapshot}, got {kwargs}"
    )


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("base_url", _GOOGLE_INFERENCE_URLS)
def test_create_openai_client_denies_google_inference_hosts_before_construction(
    shared, base_url
):
    agent = _bare_agent(provider="custom", requested_provider="custom", base_url="")
    kwargs = {"api_key": "placeholder-key", "base_url": base_url}
    snapshot = dict(kwargs)

    with (
        patch("agent.ssl_verify.resolve_httpx_verify", side_effect=_explode) as tls,
        patch("agent.auxiliary_client._validate_proxy_env_urls", side_effect=_explode) as proxy,
        patch("agent.auxiliary_client._validate_base_url", side_effect=_explode) as validate,
        patch.object(agent, "_build_keepalive_http_client", side_effect=_explode) as keepalive,
        patch("run_agent.OpenAI", side_effect=_explode) as openai_cls,
        patch("agent.gemini_native_adapter.GeminiNativeClient", side_effect=_explode) as native,
    ):
        with pytest.raises(GeminiOutboundDenied) as exc_info:
            agent._create_openai_client(kwargs, reason="test", shared=shared)

    _assert_stable_denial(exc_info)
    assert kwargs == snapshot
    tls.assert_not_called()
    proxy.assert_not_called()
    validate.assert_not_called()
    keepalive.assert_not_called()
    openai_cls.assert_not_called()
    native.assert_not_called()


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("provider", _PROVIDER_ALIASES)
def test_create_openai_client_denies_provider_aliases_without_endpoint(shared, provider):
    agent = _bare_agent(provider=provider, requested_provider=provider, base_url="")
    kwargs = {"api_key": "placeholder-key"}
    snapshot = dict(kwargs)

    with (
        patch("agent.ssl_verify.resolve_httpx_verify", side_effect=_explode) as tls,
        patch("agent.auxiliary_client._validate_proxy_env_urls", side_effect=_explode) as proxy,
        patch("run_agent.OpenAI", side_effect=_explode) as openai_cls,
    ):
        with pytest.raises(GeminiOutboundDenied) as exc_info:
            agent._create_openai_client(kwargs, reason="test", shared=shared)

    _assert_stable_denial(exc_info)
    assert kwargs == snapshot
    tls.assert_not_called()
    proxy.assert_not_called()
    openai_cls.assert_not_called()


@pytest.mark.parametrize("shared", [True, False])
def test_create_openai_client_allows_non_google_endpoint(shared):
    agent = _bare_agent(
        provider="openrouter",
        requested_provider="openrouter",
        model="google/gemini-2.5-flash",
        base_url="https://openrouter.ai/api/v1",
    )
    kwargs = {"api_key": "placeholder-key", "base_url": "https://openrouter.ai/api/v1"}
    snapshot = dict(kwargs)
    built = MagicMock(name="SafeOpenAI")

    with patch("run_agent.OpenAI", return_value=built) as openai_cls:
        client = agent._create_openai_client(kwargs, reason="test", shared=shared)

    assert client is built
    assert kwargs == snapshot
    openai_cls.assert_called_once()


def test_ensure_primary_openai_client_does_not_reuse_live_client_on_vertex_route():
    live = _LiveClient()
    agent = _bare_agent(
        provider="vertex",
        requested_provider="vertex",
        base_url="",
        client=live,
        _client_kwargs={"api_key": "placeholder-vertex-token"},
    )
    agent._create_openai_client = _explode
    agent._close_openai_client = _explode

    with pytest.raises(GeminiOutboundDenied) as exc_info:
        agent._ensure_primary_openai_client(reason="test")

    _assert_stable_denial(exc_info)
    assert agent.client is live


def test_create_request_openai_client_denies_before_cache_or_ensure():
    live = _LiveClient()
    cached = _LiveClient()
    kwargs = {
        "api_key": "placeholder-key",
        "base_url": "https://us-central1-aiplatform.googleapis.com/v1",
    }
    cache = {
        "client": cached,
        "kwargs": dict(kwargs),
        "poisoned": False,
        "in_use": False,
    }
    agent = _bare_agent(
        provider="custom",
        requested_provider="custom",
        base_url=kwargs["base_url"],
        client=live,
        _client_kwargs=dict(kwargs),
        _request_client_cache=cache,
    )
    agent._ensure_primary_openai_client = _explode
    agent._close_openai_client = _explode
    agent._create_openai_client = _explode

    with pytest.raises(GeminiOutboundDenied) as exc_info:
        agent._create_request_openai_client(reason="test")

    _assert_stable_denial(exc_info)
    assert agent.client is live
    assert cache["client"] is cached
    assert cache["poisoned"] is False
    assert cache["in_use"] is False
    assert cache["kwargs"] == kwargs


def test_replace_primary_openai_client_preserves_typed_gemini_denial():
    live = _LiveClient()
    agent = _bare_agent(
        provider="custom",
        requested_provider="custom",
        base_url="https://aiplatform.googleapis.com/v1",
        client=live,
        _client_kwargs={
            "api_key": "placeholder-key",
            "base_url": "https://aiplatform.googleapis.com/v1",
        },
    )

    with patch("run_agent.OpenAI", return_value=MagicMock()) as openai_cls:
        with pytest.raises(GeminiOutboundDenied) as exc_info:
            agent._replace_primary_openai_client(reason="test")

    _assert_stable_denial(exc_info)
    assert agent.client is live
    openai_cls.assert_not_called()


def test_ensure_primary_openai_client_preserves_typed_denial_on_rebuild():
    class _ClosedClient:
        is_closed = True

    closed = _ClosedClient()
    agent = _bare_agent(
        provider="gemini",
        requested_provider="gemini",
        base_url="",
        client=closed,
        _client_kwargs={"api_key": "placeholder-key"},
    )
    agent._close_openai_client = _explode

    with patch("run_agent.OpenAI", side_effect=_explode) as openai_cls:
        with pytest.raises(GeminiOutboundDenied) as exc_info:
            agent._ensure_primary_openai_client(reason="test")

    _assert_stable_denial(exc_info)
    assert agent.client is closed
    openai_cls.assert_not_called()
