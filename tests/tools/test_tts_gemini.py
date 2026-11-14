"""The PCM-to-WAV wrapper used by the Gemini TTS provider, and the Google outbound denial on the
TTS tool: a Gemini or Google-hosted speech route is refused before any key, client or output."""

import struct
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_BASE_URL",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)


class TestWrapPcmAsWav:
    def test_riff_header_structure(self):
        from tools.tts_tool_delivery import _wrap_pcm_as_wav

        pcm = b"\x01\x02\x03\x04" * 10
        wav = _wrap_pcm_as_wav(pcm, sample_rate=24000, channels=1, sample_width=2)

        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "
        # Audio format (PCM=1)
        assert struct.unpack("<H", wav[20:22])[0] == 1
        # Channels
        assert struct.unpack("<H", wav[22:24])[0] == 1
        # Sample rate
        assert struct.unpack("<I", wav[24:28])[0] == 24000
        # Bits per sample
        assert struct.unpack("<H", wav[34:36])[0] == 16
        assert wav[36:40] == b"data"
        assert wav[44:] == pcm


# ── Google outbound denial on the TTS tool ─────────────────────────────────────────────────

_STUDIO = "https://generativelanguage.googleapis.com/v1beta/openai"


class _Tripwire:
    def __init__(self, name, touched):
        self._name, self._touched = name, touched

    def __call__(self, *_args, **_kwargs):
        self._touched.append(self._name)
        raise AssertionError(f"{self._name} was reached before the refusal")


def _deny_type():
    from agent.gemini_outbound_policy import GeminiOutboundDenied
    return GeminiOutboundDenied


class TestGoogleOutboundDenial:
    @pytest.mark.parametrize("config", [
        pytest.param({"provider": "gemini"}, id="gemini-configured"),
        pytest.param({"provider": "openai", "openai": {"base_url": _STUDIO}}, id="openai-on-google-host"),
        pytest.param({"provider": "openai", "openai": {"model": "gemini-2.5-flash-preview-tts"}},
                     id="openai-with-gemini-model"),
        pytest.param({"provider": "deepinfra", "deepinfra": {"model": "gemini-tts"}}, id="deepinfra-pinned"),
    ])
    def test_a_google_bound_speech_route_is_refused_before_keys_clients_or_output(
            self, monkeypatch, tmp_path, config):
        from tools import tts_tool, tts_tool_openai

        touched = []
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: config)
        monkeypatch.setattr(tts_tool, "_resolve_provider_key", _Tripwire("_resolve_provider_key", touched))
        monkeypatch.setattr(tts_tool, "_import_openai_client", _Tripwire("_import_openai_client", touched))
        monkeypatch.setattr(tts_tool_openai, "_resolve_openai_audio_client_config",
                            _Tripwire("_resolve_openai_audio_client_config", touched))
        out = tmp_path / "speech" / "a.mp3"

        with pytest.raises(_deny_type()):
            tts_tool.text_to_speech_tool("hello", output_path=str(out))
        assert touched == []
        assert not out.parent.exists()

    def test_a_single_chunk_for_a_google_provider_is_refused_before_dispatch(self, monkeypatch, tmp_path):
        from tools import tts_tool

        touched = []
        monkeypatch.setattr(tts_tool, "_select_builtin_engine", _Tripwire("_select_builtin_engine", touched))
        with pytest.raises(_deny_type()):
            tts_tool._text_to_speech_single(
                "hello", str(tmp_path / "a.mp3"), provider="gemini", tts_config={},
                command_provider_config=None, want_opus=False, instructions=None)
        assert touched == []

    def test_openai_tts_checks_the_configured_google_host_before_the_auth_chain(self, monkeypatch, tmp_path):
        from tools import tts_tool_openai

        touched = []
        monkeypatch.setattr(tts_tool_openai, "_resolve_openai_audio_client_config",
                            _Tripwire("_resolve_openai_audio_client_config", touched))
        with pytest.raises(_deny_type()):
            tts_tool_openai._generate_openai_tts("hello", str(tmp_path / "a.mp3"), {"openai": {"base_url": _STUDIO}})
        assert touched == []

    def test_openai_tts_refuses_an_auth_chain_fallback_on_a_google_host_before_the_client(self, monkeypatch, tmp_path):
        from tools import tts_tool, tts_tool_openai

        touched = []
        monkeypatch.setattr(tts_tool_openai, "_resolve_openai_audio_client_config", lambda: ("k", _STUDIO, False))
        monkeypatch.setattr(tts_tool, "_import_openai_client", _Tripwire("_import_openai_client", touched))
        with pytest.raises(_deny_type()):
            tts_tool_openai._generate_openai_tts("hello", str(tmp_path / "a.mp3"), {})
        assert touched == []

    def test_a_pinned_gemini_deepinfra_model_is_refused_before_the_catalog_or_key(self, monkeypatch, tmp_path):
        from tools import tts_tool, tts_tool_openai

        touched = []
        monkeypatch.setattr("hermes_cli.models.deepinfra_model_ids", _Tripwire("deepinfra_model_ids", touched))
        monkeypatch.setattr(tts_tool, "_resolve_provider_key", _Tripwire("_resolve_provider_key", touched))
        with pytest.raises(_deny_type()):
            tts_tool_openai._generate_deepinfra_tts("hello", str(tmp_path / "a.mp3"), {"deepinfra": {"model": "gemini-tts"}})
        assert touched == []

    def test_a_catalog_picked_gemini_deepinfra_model_is_refused_before_the_key(self, monkeypatch, tmp_path):
        from tools import tts_tool, tts_tool_openai

        touched = []
        monkeypatch.setattr("hermes_cli.models.deepinfra_model_ids", lambda kind: ["gemini-tts-latest"])
        monkeypatch.setattr(tts_tool, "_resolve_provider_key", _Tripwire("_resolve_provider_key", touched))
        with pytest.raises(_deny_type()):
            tts_tool_openai._generate_deepinfra_tts("hello", str(tmp_path / "a.mp3"), {"deepinfra": {}})
        assert touched == []

    def test_an_explicit_google_endpoint_is_refused_before_the_client(self, monkeypatch, tmp_path):
        from tools import tts_tool, tts_tool_openai

        touched = []
        monkeypatch.setattr(tts_tool, "_import_openai_client", _Tripwire("_import_openai_client", touched))
        with pytest.raises(_deny_type()):
            tts_tool_openai._generate_openai_tts("hello", str(tmp_path / "a.mp3"), {}, api_key="k", base_url=_STUDIO)
        assert touched == []

    def test_the_gemini_generator_itself_refuses_before_its_key(self, monkeypatch, tmp_path):
        from tools import tts_tool
        from tools.tts_tool_providers import _generate_gemini_tts

        touched = []
        monkeypatch.setattr(tts_tool, "_resolve_provider_key", _Tripwire("_resolve_provider_key", touched))
        with pytest.raises(_deny_type()):
            _generate_gemini_tts("hello", str(tmp_path / "a.wav"), {})
        assert touched == []

    def test_gemini_is_never_available_and_reads_no_key(self, monkeypatch):
        from tools import tts_tool

        touched = []
        monkeypatch.setattr(tts_tool, "_resolve_provider_key", _Tripwire("_resolve_provider_key", touched))
        assert tts_tool._BUILTIN_REQUIREMENTS["gemini"]() is False
        assert touched == []

    def test_the_denial_is_not_turned_into_a_tool_failure_by_the_chunk_excepts(self, monkeypatch, tmp_path):
        from tools import tts_tool

        def _deny(*_a, **_k):
            raise _deny_type()()

        monkeypatch.setattr(tts_tool, "_preflight_tts_outbound_route", lambda provider, config: None)
        monkeypatch.setattr(tts_tool, "_select_builtin_engine", lambda provider: (provider, None))
        monkeypatch.setattr(tts_tool, "_synthesize_builtin", _deny)
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
        with pytest.raises(_deny_type()):
            tts_tool.text_to_speech_tool("hello", output_path=str(tmp_path / "a.mp3"))

    def test_openai_tts_on_an_ordinary_host_still_speaks(self, monkeypatch, tmp_path):
        from tools import tts_tool, tts_tool_openai

        sent = []

        class _Speech:
            def create(self, **kwargs):
                sent.append(kwargs)
                return MagicMock(stream_to_file=lambda path: open(path, "wb").write(b"ID3audio"))

        def _client(api_key, base_url):
            sent.append(base_url)
            return MagicMock(audio=MagicMock(speech=_Speech()))

        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: _client)
        monkeypatch.setattr(tts_tool_openai, "_resolve_openai_audio_client_config", lambda: ("k", None, False))
        out = tmp_path / "a.mp3"

        path = tts_tool_openai._generate_openai_tts(
            "hello", str(out), {"openai": {"base_url": "https://ordinary.example.test/v1"}})

        assert path == str(out) and out.read_bytes() == b"ID3audio"
        assert sent[0] == "https://ordinary.example.test/v1"
