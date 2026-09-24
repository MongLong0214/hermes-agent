"""Contracts for the Gemini / Google AI Studio / Vertex outbound default-deny.

Every boundary that can reach a Google-hosted inference or speech endpoint refuses the route
with the typed denial before it reads a credential or builds a client, and a Google model served
by a non-Google endpoint stays allowed where the boundary holds the concrete endpoint.
"""

from types import SimpleNamespace

import pytest

from agent import agent_runtime_helpers, auxiliary_client
from agent.gemini_native_adapter import GeminiNativeClient, probe_gemini_tier
from agent.gemini_outbound_policy import GeminiOutboundDenied
from agent.vertex_adapter import get_vertex_credentials
from hermes_cli import model_switch, runtime_provider
from hermes_constants import OPENROUTER_BASE_URL
from mini_swe_runner import MiniSWERunner
from tools import tts_streaming, tts_tool
from tools.delegate_tool_config import _runtime_provider_credentials
from trajectory_compressor import CompressionConfig, TrajectoryCompressor

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


def _trajectory(model, base_url):
    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    compressor.config = CompressionConfig(summarization_model=model, base_url=base_url,
                                          api_key_env="TRAJECTORY_TEST_KEY")
    compressor._async_client_api_key = "k"
    return compressor


def _agent(provider, base_url, model="ordinary-model"):
    return SimpleNamespace(provider=provider, model=model, base_url=base_url, api_mode="chat_completions",
                           requested_provider=provider, _client_log_context=lambda: "test",
                           _build_keepalive_http_client=lambda *a, **k: None)


# (tripwired seams, config/stand-in patches, the boundary call(tmp_path, tripwire_factory))
_DENIED_ROUTES = [
    pytest.param(("hermes_cli.runtime_provider._ladder_rungs",), {},
                 lambda tmp, trip: runtime_provider.resolve_runtime_provider(requested="gemini"),
                 id="primary-resolver"),
    pytest.param(("hermes_cli.runtime_provider_custom.get_secret_str",),
                 {"hermes_cli.runtime_provider.load_config":
                  lambda: {"providers": {"relay": {"base_url": STUDIO, "key_env": "RELAY_TEST_KEY"}}}},
                 lambda tmp, trip: runtime_provider.resolve_runtime_provider(requested="custom:relay"),
                 id="primary-named-custom-endpoint"),
    pytest.param(("hermes_cli.runtime_provider._ladder_rungs",), {},
                 lambda tmp, trip: _runtime_provider_credentials({"provider": "vertex", "model": "gemini-2.5-flash"}, None),
                 id="delegation"),
    pytest.param(("agent.auxiliary_client.load_pool", "agent.auxiliary_client.OpenAI",
                  "hermes_cli.auth.resolve_api_key_provider_credentials"), {},
                 lambda tmp, trip: auxiliary_client.resolve_provider_client("gemini", "gemini-2.5-flash"),
                 id="auxiliary-resolver"),
    pytest.param(("agent.auxiliary_client.load_pool", "agent.auxiliary_client._peek_pool_entry",
                  "agent.auxiliary_client.resolve_provider_client"), {},
                 lambda tmp, trip: auxiliary_client._get_cached_client("custom", "ordinary-model", base_url=VERTEX, api_key="k"),
                 id="auxiliary-client-cache"),
    pytest.param(("hermes_cli.runtime_provider.resolve_runtime_provider",), {},
                 lambda tmp, trip: model_switch.switch_model("gemini-2.5-flash", "custom", "ordinary-model",
                                                             explicit_provider="gemini"),
                 id="model-command"),
    pytest.param(("agent.agent_runtime_helpers.create_openai_client",), {},
                 lambda tmp, trip: agent_runtime_helpers.switch_model(trip("agent"), "gemini-2.5-flash", "gemini",
                                                                      api_key="k", base_url=STUDIO),
                 id="agent-switch-model"),
    pytest.param(("agent.process_bootstrap.OpenAI", "agent.agent_runtime_helpers._gemini_native_client"), {},
                 lambda tmp, trip: agent_runtime_helpers.create_openai_client(
                     _agent("custom", VERTEX), {"base_url": VERTEX, "api_key": "k"}, reason="test", shared=False),
                 id="client-construction"),
    pytest.param(("trajectory_compressor.os", "openai.OpenAI"), {},
                 lambda tmp, trip: _trajectory("gemini-2.5-flash", ORDINARY)._init_summarizer(),
                 id="trajectory-custom-endpoint"),
    pytest.param(("openai.AsyncOpenAI",), {},
                 lambda tmp, trip: _trajectory("ordinary-model", STUDIO)._get_async_client(),
                 id="trajectory-async-client"),
    pytest.param(("mini_swe_runner.MiniSWERunner._init_client",), {},
                 lambda tmp, trip: MiniSWERunner(model="gemini-2.5-flash"),
                 id="mini-swe"),
    pytest.param(("agent.gemini_native_adapter.httpx",), {},
                 lambda tmp, trip: GeminiNativeClient(api_key="k", base_url=STUDIO),
                 id="native-client"),
    pytest.param(("agent.gemini_native_adapter.httpx",), {},
                 lambda tmp, trip: probe_gemini_tier("k"),
                 id="native-tier-probe"),
    pytest.param(("agent.vertex_adapter._resolve_credentials_path", "agent.vertex_adapter._sa_snapshot"), {},
                 lambda tmp, trip: get_vertex_credentials(),
                 id="vertex-credentials"),
    pytest.param(("tools.tts_tool._resolve_provider_key",), {"tools.tts_tool._load_tts_config": lambda: {}},
                 lambda tmp, trip: tts_tool.text_to_speech_tool("hello", output_path=str(tmp / "speech" / "a.mp3"),
                                                                provider="gemini"),
                 id="tts-gemini"),
    pytest.param(("tools.tts_tool._resolve_provider_key", "tools.tts_tool_openai._resolve_openai_audio_client_config"),
                 {"tools.tts_tool._load_tts_config": lambda: {"provider": "openai", "openai": {"base_url": STUDIO}}},
                 lambda tmp, trip: tts_tool.text_to_speech_tool("hello", output_path=str(tmp / "speech" / "a.mp3")),
                 id="tts-openai-google-endpoint"),
    pytest.param(("tools.tts_streaming._gemini_key",), {},
                 lambda tmp, trip: tts_streaming.resolve_streaming_provider({"streaming": {"provider": "gemini"}}),
                 id="tts-streaming"),
]


def _refusal(call) -> str:
    """The owner-facing refusal a boundary produced: the typed denial (also when a caller chains
    it into its own error) or a ``/model`` result's message; anything else is reported as is."""
    try:
        result = call()
    except Exception as exc:
        denial = exc if isinstance(exc, GeminiOutboundDenied) else exc.__cause__
        return str(denial) if isinstance(denial, GeminiOutboundDenied) else f"not a denial: {exc!r}"
    return getattr(result, "error_message", None) or f"not refused: {result!r}"


@pytest.mark.parametrize(("seams", "patches", "call"), _DENIED_ROUTES)
def test_every_google_bound_boundary_refuses_before_credentials_or_client(seams, patches, call, tmp_path, monkeypatch):
    touched = []
    for seam in seams:
        monkeypatch.setattr(seam, _Tripwire(seam, touched))
    for target, value in patches.items():
        monkeypatch.setattr(target, value)

    refusal = _refusal(lambda: call(tmp_path, lambda name: _Tripwire(name, touched)))

    assert refusal == GeminiOutboundDenied.public_message
    assert touched == []
    assert not (tmp_path / "speech").exists()  # speech rows: no output directory was created


# (boundary taking the endpoint, the non-Google endpoint it must keep serving)
_ENDPOINT_PAIRS = [
    pytest.param(lambda url: runtime_provider.resolve_runtime_provider(
        requested="custom", explicit_base_url=url, explicit_api_key="k", target_model="ordinary-model")["base_url"],
        ORDINARY, id="resolver"),
    pytest.param(lambda url: auxiliary_client.resolve_provider_client(
        "custom", "google/gemini-2.5-flash", explicit_base_url=url, explicit_api_key="k")[0].base_url,
        "https://safe-relay.example/v1", id="auxiliary"),
    pytest.param(lambda url: agent_runtime_helpers.create_openai_client(
        _agent("custom", url, model="google/gemini-2.5-flash"), {"base_url": url, "api_key": "k"},
        reason="test", shared=False).base_url,
        OPENROUTER_BASE_URL, id="client-construction"),
]


@pytest.mark.parametrize(("build", "allowed_url"), _ENDPOINT_PAIRS)
def test_non_google_endpoint_resolves_and_its_google_host_twin_is_refused(build, allowed_url):
    assert str(build(allowed_url)).rstrip("/") == allowed_url.rstrip("/")

    with pytest.raises(GeminiOutboundDenied):
        build(STUDIO)
