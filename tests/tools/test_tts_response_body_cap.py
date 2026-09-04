"""Regression tests for bounded upstream TTS response reads."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tools import tts_tool


class StreamingResponse:
    def __init__(self, chunks, *, status_code=200, headers=None):
        self._chunks = list(chunks)
        self.status_code = status_code
        self.headers = headers or {}
        self.closed = False

    def iter_content(self, chunk_size=65536):
        del chunk_size
        yield from self._chunks

    def close(self):
        self.closed = True

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def small_tts_body_cap(monkeypatch):
    monkeypatch.setattr(tts_tool, "TTS_RESPONSE_BODY_LIMIT_BYTES", 8)


def test_xai_tts_rejects_oversized_audio_response(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    response = StreamingResponse([b"12345", b"6789"], headers={"Content-Type": "audio/mpeg"})
    output_path = tmp_path / "out.mp3"

    with patch("requests.post", return_value=response) as post:
        with pytest.raises(RuntimeError, match="xAI TTS response exceeds 8 bytes"):
            tts_tool._generate_xai_tts("hello", str(output_path), {})

    assert post.call_args.kwargs["stream"] is True
    assert response.closed is True
    assert not output_path.exists()


def test_gemini_tts_denies_before_response_or_output_side_effects(
    tmp_path, monkeypatch
):
    from agent.gemini_outbound_policy import GeminiOutboundDenied

    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    response = StreamingResponse([b'{"candidates":', b"[{}]}"], headers={"Content-Type": "application/json"})
    output_path = tmp_path / "out.wav"

    with patch("requests.post", return_value=response) as post, patch.object(
        response, "iter_content", wraps=response.iter_content
    ) as iter_content:
        with pytest.raises(GeminiOutboundDenied) as exc_info:
            tts_tool._generate_gemini_tts("hello", str(output_path), {})

    assert exc_info.type is GeminiOutboundDenied
    assert type(exc_info.value) is GeminiOutboundDenied
    assert exc_info.value.code == "gemini_outbound_denied"
    assert str(exc_info.value) == "Gemini outbound requests are disabled."
    assert vars(exc_info.value) == {}
    post.assert_not_called()
    iter_content.assert_not_called()
    assert response.closed is False
    assert not output_path.exists()
