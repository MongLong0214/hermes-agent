from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import openai
import pytest

from agent.gemini_outbound_policy import GeminiOutboundDenied


@pytest.mark.parametrize(
    ("model", "base_url"),
    (
        pytest.param(
            "gemini-2.5-pro",
            "https://api.example.test/v1",
            id="gemini-model-ordinary-url",
        ),
        pytest.param(
            "openai/gpt-4.1-mini",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            id="google-generativelanguage-host",
        ),
        pytest.param(
            "openai/gpt-4.1-mini",
            "https://us-central1-aiplatform.googleapis.com/v1",
            id="google-vertex-host",
        ),
    ),
)
def test_init_denies_gemini_outbound_before_credentials_or_client_creation(
    model, base_url
):
    """Gemini-family routes fail closed before credential or client access."""
    import mini_swe_runner

    with patch.object(
        mini_swe_runner.os,
        "getenv",
        side_effect=AssertionError("os.getenv must not be called for denied routes"),
    ) as mock_getenv, patch.object(
        openai,
        "OpenAI",
        side_effect=AssertionError("OpenAI must not be constructed for denied routes"),
    ) as mock_openai:
        with pytest.raises(GeminiOutboundDenied) as raised:
            mini_swe_runner.MiniSWERunner(
                model=model,
                base_url=base_url,
                api_key="test-key",
            )

    error = raised.value
    assert raised.type is GeminiOutboundDenied
    assert type(error) is GeminiOutboundDenied
    assert error.code == "gemini_outbound_denied"
    assert str(error) == "Gemini outbound requests are disabled."
    assert vars(error) == {}
    mock_getenv.assert_not_called()
    mock_openai.assert_not_called()


def test_init_allows_ordinary_non_google_openai_compatible_endpoint():
    """Ordinary non-Google routes retain the direct mocked OpenAI path."""
    import mini_swe_runner

    with patch("openai.OpenAI") as mock_openai:
        client = MagicMock()
        mock_openai.return_value = client

        runner = mini_swe_runner.MiniSWERunner(
            model="openai/gpt-4.1-mini",
            base_url="https://api.example.test/v1",
            api_key="test-key",
        )

    assert runner.client is client
    mock_openai.assert_called_once_with(
        base_url="https://api.example.test/v1",
        api_key="test-key",
    )


def test_run_task_kimi_omits_temperature():
    """Kimi models should NOT have client-side temperature overrides.

    The Kimi gateway selects the correct temperature server-side.
    """
    with patch("openai.OpenAI") as mock_openai:
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=[]))]
        )
        mock_openai.return_value = client

        from mini_swe_runner import MiniSWERunner

        runner = MiniSWERunner(
            model="kimi-for-coding",
            base_url="https://api.kimi.com/coding/v1",
            api_key="test-key",
            env_type="local",
            max_iterations=1,
        )
        runner._create_env = MagicMock()
        runner._cleanup_env = MagicMock()

        result = runner.run_task("2+2")

    assert result["completed"] is True
    assert "temperature" not in client.chat.completions.create.call_args.kwargs


def test_run_task_public_moonshot_kimi_k2_5_omits_temperature():
    """kimi-k2.5 on the public Moonshot API should not get a forced temperature."""
    with patch("openai.OpenAI") as mock_openai:
        client = MagicMock()
        client.base_url = "https://api.moonshot.ai/v1"
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=[]))]
        )
        mock_openai.return_value = client

        from mini_swe_runner import MiniSWERunner

        runner = MiniSWERunner(
            model="kimi-k2.5",
            base_url="https://api.moonshot.ai/v1",
            api_key="test-key",
            env_type="local",
            max_iterations=1,
        )
        runner._create_env = MagicMock()
        runner._cleanup_env = MagicMock()

        result = runner.run_task("2+2")

    assert result["completed"] is True
    assert "temperature" not in client.chat.completions.create.call_args.kwargs
