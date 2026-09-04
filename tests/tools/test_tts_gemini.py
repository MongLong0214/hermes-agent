"""Default-deny tests for Gemini-family TTS routes."""

from __future__ import annotations

import builtins
import struct
from types import SimpleNamespace

import pytest

from agent.gemini_outbound_policy import GeminiOutboundDenied


def _assert_stable_denial(exc_info: pytest.ExceptionInfo[GeminiOutboundDenied]) -> None:
    assert exc_info.type is GeminiOutboundDenied
    assert type(exc_info.value) is GeminiOutboundDenied
    assert exc_info.value.code == "gemini_outbound_denied"
    assert str(exc_info.value) == "Gemini outbound requests are disabled."
    assert vars(exc_info.value) == {}


class TestWrapPcmAsWav:
    def test_riff_header_structure(self):
        from tools.tts_tool import _wrap_pcm_as_wav

        pcm = b"\x01\x02\x03\x04" * 10
        wav = _wrap_pcm_as_wav(pcm, sample_rate=24000, channels=1, sample_width=2)

        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "
        assert struct.unpack("<H", wav[20:22])[0] == 1
        assert struct.unpack("<H", wav[22:24])[0] == 1
        assert struct.unpack("<I", wav[24:28])[0] == 24000
        assert struct.unpack("<H", wav[34:36])[0] == 16
        assert wav[36:40] == b"data"
        assert wav[44:] == pcm

    def test_header_size_is_44(self):
        from tools.tts_tool import _wrap_pcm_as_wav

        pcm = b"\xff" * 100
        assert len(_wrap_pcm_as_wav(pcm)) == 44 + len(pcm)


class TestGeminiOutboundTts:
    def test_sync_gemini_denies_before_secret_or_http(self, tmp_path, monkeypatch):
        from tools import tts_tool

        def _explode(*_args, **_kwargs):
            raise AssertionError("secret, config, env, or HTTP access reached")

        real_import = builtins.__import__

        def _guard_import(name, *args, **kwargs):
            if name == "requests":
                raise AssertionError("HTTP import reached")
            return real_import(name, *args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(tts_tool, "_resolve_provider_key", _explode)
            context.setattr(tts_tool, "get_env_value", _explode)
            context.setattr(builtins, "__import__", _guard_import)
            with pytest.raises(GeminiOutboundDenied) as exc_info:
                tts_tool._generate_gemini_tts(
                    "Hello", str(tmp_path / "denied.wav"), {"gemini": {}}
                )
        _assert_stable_denial(exc_info)

        def _deny_generator(*_args, **_kwargs):
            raise GeminiOutboundDenied()

        monkeypatch.setattr(tts_tool, "_generate_gemini_tts", _deny_generator)
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "gemini"})
        with pytest.raises(GeminiOutboundDenied) as single_exc:
            tts_tool._text_to_speech_single(
                "Hello", str(tmp_path / "single.wav"), provider="gemini"
            )
        _assert_stable_denial(single_exc)
        with pytest.raises(GeminiOutboundDenied) as multi_exc:
            tts_tool.text_to_speech_tool(
                "Hello", str(tmp_path / "multi.wav"), provider="gemini"
            )
        _assert_stable_denial(multi_exc)

    @pytest.mark.parametrize(
        ("model", "base_url"),
        [
            ("gemini-2.5-flash-tts", "https://api.openai.com/v1"),
            ("ordinary-tts-model", "https://aiplatform.googleapis.com/v1"),
        ],
        ids=["google-model", "vertex-base-url"],
    )
    def test_openai_compatible_google_or_vertex_route_denies_before_secret_or_client(
        self, tmp_path, monkeypatch, model, base_url
    ):
        from tools import tts_tool

        def _explode(*_args, **_kwargs):
            raise AssertionError("credential resolver or OpenAI client reached")

        monkeypatch.setattr(tts_tool, "_resolve_openai_audio_client_config", _explode)
        monkeypatch.setattr(tts_tool, "_import_openai_client", _explode)

        with pytest.raises(GeminiOutboundDenied) as exc_info:
            tts_tool._generate_openai_tts(
                "Hello",
                str(tmp_path / "denied.mp3"),
                {"openai": {"model": model, "base_url": base_url}},
            )
        _assert_stable_denial(exc_info)

    def test_openai_compatible_non_google_route_preserves_client_path(self, tmp_path, monkeypatch):
        from tools import tts_tool

        captured = {}

        class _Response:
            def stream_to_file(self, output_path):
                with open(output_path, "wb") as output:
                    output.write(b"audio")

        class _Client:
            def __init__(self, **kwargs):
                captured["client"] = kwargs
                self.audio = SimpleNamespace(
                    speech=SimpleNamespace(create=lambda **kwargs: _Response())
                )

            def close(self):
                captured["closed"] = True

        monkeypatch.setattr(
            tts_tool,
            "_resolve_openai_audio_client_config",
            lambda: ("local-key", "https://api.openai.com/v1", False),
        )
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: _Client)

        output_path = str(tmp_path / "allowed.mp3")
        assert tts_tool._generate_openai_tts(
            "Hello",
            output_path,
            {
                "openai": {
                    "model": "local-tts-model",
                    "base_url": "http://localhost:8080/v1",
                }
            },
        ) == output_path
        assert captured["client"] == {
            "api_key": "local-key",
            "base_url": "http://localhost:8080/v1",
        }
        assert captured["closed"] is True

    @pytest.mark.parametrize("entrypoint", ["single", "public"])
    @pytest.mark.parametrize(
        ("provider", "tts_config"),
        [
            ("gemini", {"provider": "gemini"}),
            (
                "openai",
                {
                    "provider": "openai",
                    "openai": {"model": "gemini-2.5-flash-tts"},
                },
            ),
            (
                "openai",
                {
                    "provider": "openai",
                    "openai": {"base_url": "https://aiplatform.googleapis.com/v1"},
                },
            ),
            (
                "deepinfra",
                {
                    "provider": "deepinfra",
                    "deepinfra": {"model": "gemini-2.5-flash-tts"},
                },
            ),
            (
                "deepinfra",
                {
                    "provider": "deepinfra",
                    "deepinfra": {"base_url": "https://vertexai.googleapis.com/v1"},
                },
            ),
        ],
        ids=[
            "fixed-gemini",
            "openai-gemini-model",
            "openai-vertex-base",
            "deepinfra-gemini-model",
            "deepinfra-vertex-base",
        ],
    )
    def test_public_dispatch_route_denies_before_output_or_client_side_effects(
        self, tmp_path, monkeypatch, entrypoint, provider, tts_config
    ):
        """Public TTS entrypoints must reject configured Google routes first."""
        from tools import tts_tool

        output_dir = tmp_path / f"{entrypoint}-{provider}-denied"

        def _explode(*_args, **_kwargs):
            raise AssertionError("output, credential, or OpenAI client side effect reached")

        monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(output_dir))
        monkeypatch.setattr(tts_tool, "_resolve_openai_audio_client_config", _explode)
        monkeypatch.setattr(tts_tool, "_resolve_provider_key", _explode)
        monkeypatch.setattr(tts_tool, "_import_openai_client", _explode)
        monkeypatch.setattr(tts_tool, "_generate_gemini_tts", _explode)
        monkeypatch.setattr(tts_tool.Path, "mkdir", _explode)
        monkeypatch.setattr(builtins, "open", _explode)
        if entrypoint == "public":
            monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: tts_config)

        with pytest.raises(GeminiOutboundDenied) as exc_info:
            if entrypoint == "single":
                tts_tool._text_to_speech_single(
                    "Hello", provider=provider, tts_config_override=tts_config
                )
            else:
                tts_tool.text_to_speech_tool("Hello", provider=provider)

        _assert_stable_denial(exc_info)
        assert not output_dir.exists()

    def test_public_deepinfra_dynamic_route_denies_before_output_credential_or_client(
        self, tmp_path, monkeypatch
    ):
        """A discovered DeepInfra Gemini model is rejected before dispatch output."""
        from hermes_cli import models
        from tools import tts_tool

        output_dir = tmp_path / "deepinfra-dynamic-denied"

        def _explode(*_args, **_kwargs):
            raise AssertionError("output, credential, or OpenAI client side effect reached")

        monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(output_dir))
        monkeypatch.setattr(tts_tool, "_resolve_provider_key", _explode)
        monkeypatch.setattr(tts_tool, "_import_openai_client", _explode)
        monkeypatch.setattr(tts_tool.Path, "mkdir", _explode)
        monkeypatch.setattr(builtins, "open", _explode)
        monkeypatch.setattr(models, "deepinfra_model_ids", lambda _surface: ["gemini-2.5-flash-tts"])
        monkeypatch.setattr(
            tts_tool,
            "_load_tts_config",
            lambda: {"provider": "deepinfra", "deepinfra": {}},
        )

        with pytest.raises(GeminiOutboundDenied) as exc_info:
            tts_tool.text_to_speech_tool("Hello", provider="deepinfra")

        _assert_stable_denial(exc_info)
        assert not output_dir.exists()


class TestGeminiInCheckRequirements:
    def test_gemini_cannot_satisfy_requirements_without_key_lookup(self, monkeypatch):
        from tools import tts_tool

        def _explode(*_args, **_kwargs):
            raise AssertionError("Gemini key lookup reached")

        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "gemini"})
        monkeypatch.setattr(tts_tool, "_resolve_provider_key", _explode)

        assert tts_tool.check_tts_requirements() is False
