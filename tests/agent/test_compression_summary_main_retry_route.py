"""L2-1: the main-model retry after a failed summary model must actually reach the main model.

Deployment shape: main ``gpt-6.1-sol`` on ``openai-codex`` and ``auxiliary.compression.model:
gpt-6-luna``. The agent builds its compressor with no ``summary_model_override``, so luna comes from
the task config. When luna fails, the compressor logs "Falling back to main model" and retries — but a
retry that names no model is resolved by the auxiliary task resolver to the configured luna again.

These tests run the REAL ``call_llm`` → ``_resolve_task_provider_model`` path; only the task config and
the final client are faked. A spy around the real resolver records the (provider, model) each summary
attempt resolved to; the client records the model every wire request carried.
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent import auxiliary_client
from agent.context_compressor import ContextCompressor

MAIN_PROVIDER = "openai-codex"
MAIN_MODEL = "gpt-6.1-sol"
SUMMARY_MODEL = "gpt-6-luna"


def _msgs():
    return [
        {"role": "user", "content": "please refactor the parser"},
        {"role": "assistant", "content": "working on it"},
    ]


def _ok_response(text):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=text, tool_calls=None), finish_reason="stop")])


class _RecordingClient:
    """Stands in for the provider client; records the route each request was resolved to."""

    def __init__(self, provider, attempts, failing_models):
        self.base_url = "https://chatgpt.com/backend-api/codex"
        self._provider = provider
        self._attempts = attempts
        self._failing = failing_models
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        model = kwargs.get("model")
        self._attempts.append((self._provider, model))
        if model in self._failing:
            raise ValueError(f"{model} rejected the summary request")
        return _ok_response(f"summary written by {model}")


def _resolver_spy(resolved):
    real = auxiliary_client._resolve_task_provider_model

    def _spy(*args, **kwargs):
        route = real(*args, **kwargs)
        resolved.append((route[0], route[1]))
        return route
    return _spy


def _run(failing_models, task_config):
    attempts, requests = [], []

    def _cached_client(provider, model=None, **_kw):
        return _RecordingClient(provider, requests, failing_models), model

    with patch("agent.context_compressor.get_model_context_length", return_value=272_000):
        c = ContextCompressor(
            model=MAIN_MODEL, provider=MAIN_PROVIDER, base_url="https://chatgpt.com/backend-api/codex",
            api_key="main-route-token", api_mode="codex_responses", quiet_mode=True,
            abort_on_summary_failure=False,
        )
    with patch("agent.auxiliary_client._get_auxiliary_task_config",
               side_effect=lambda task: dict(task_config) if task == "compression" else {}), \
            patch("agent.auxiliary_client._get_cached_client", side_effect=_cached_client), \
            patch("agent.auxiliary_client._resolve_task_provider_model", side_effect=_resolver_spy(attempts)):
        result = c._generate_summary(_msgs())
    return c, attempts, requests, result


def test_failed_configured_summary_model_retries_on_the_main_model():
    c, attempts, requests, result = _run(
        {SUMMARY_MODEL}, {"provider": MAIN_PROVIDER, "model": SUMMARY_MODEL},
    )

    assert attempts == [(MAIN_PROVIDER, SUMMARY_MODEL), (MAIN_PROVIDER, MAIN_MODEL)], attempts
    assert requests[-1] == (MAIN_PROVIDER, MAIN_MODEL), requests
    assert result is not None and f"summary written by {MAIN_MODEL}" in result
    assert c._last_aux_model_failure_model == SUMMARY_MODEL


def test_main_retry_uses_main_provider_when_summary_route_is_another_provider():
    _c, attempts, requests, result = _run(
        {"glm-5.3"}, {"provider": "openrouter", "model": "glm-5.3"},
    )

    assert attempts == [("openrouter", "glm-5.3"), (MAIN_PROVIDER, MAIN_MODEL)], attempts
    assert requests[-1] == (MAIN_PROVIDER, MAIN_MODEL), requests
    assert result is not None and f"summary written by {MAIN_MODEL}" in result


def test_main_retry_pin_does_not_leak_into_the_next_compression():
    """The main route is pinned for the retry call only; the next compression asks the configured route again."""
    attempts, requests = [], []

    def _cached_client(provider, model=None, **_kw):
        return _RecordingClient(provider, requests, set()), model

    with patch("agent.context_compressor.get_model_context_length", return_value=272_000):
        c = ContextCompressor(model=MAIN_MODEL, provider=MAIN_PROVIDER, quiet_mode=True)
    with patch("agent.auxiliary_client._get_auxiliary_task_config",
               side_effect=lambda task: {"provider": MAIN_PROVIDER, "model": SUMMARY_MODEL}
               if task == "compression" else {}), \
            patch("agent.auxiliary_client._get_cached_client", side_effect=_cached_client), \
            patch("agent.auxiliary_client._resolve_task_provider_model", side_effect=_resolver_spy(attempts)):
        c._generate_summary(_msgs())

    assert attempts == [(MAIN_PROVIDER, SUMMARY_MODEL)]
    assert requests == [(MAIN_PROVIDER, SUMMARY_MODEL)]
