"""Google outbound denial on the streaming TTS paths (tools.tts_streaming, tools.tts_tool_speaker).

A Gemini streamer, or an OpenAI-compatible streamer on a Google host, is refused before any key is
read or client built; the speaker refuses a Google-bound provider before probing a streamer, and a
denial raised on the synthesis executor or a prefetch thread reaches the caller without
deadlocking the drain. Fake streamers and a fake synth only: nothing is sent anywhere.
"""

import queue
import threading
from unittest.mock import MagicMock, patch

import pytest

import tools.tts_streaming as ts
from agent.gemini_outbound_policy import GeminiOutboundDenied
from tools import tts_tool
from tools import tts_tool_speaker
from tools.tts_tool_speaker import _StreamerPlayback, _SyncSentencePipeline, stream_tts_to_speaker

STUDIO = "https://generativelanguage.googleapis.com/v1beta/openai"


class _Tripwire:
    def __init__(self, name, touched):
        self._name, self._touched = name, touched

    def __call__(self, *_args, **_kwargs):
        self._touched.append(self._name)
        raise AssertionError(f"{self._name} was reached before the refusal")


def _text_queue(sentences):
    q = queue.Queue()
    for sentence in sentences:
        q.put(sentence)
    q.put(None)
    return q


def _run_speaker(q, **kwargs):
    """Run the speaker on a thread so a deadlock fails the test instead of hanging it."""
    stop, done, outcome = threading.Event(), threading.Event(), {}

    def _target():
        try:
            stream_tts_to_speaker(q, stop, done, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - the test inspects what escaped
            outcome["exc"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout=20)
    assert not worker.is_alive(), "stream_tts_to_speaker deadlocked"
    return outcome.get("exc"), done


# ── tools.tts_streaming ────────────────────────────────────────────────────────────────────


def test_a_pinned_gemini_streamer_is_refused_before_its_key_is_read(monkeypatch):
    touched = []
    monkeypatch.setattr(ts, "_gemini_key", _Tripwire("gemini_key", touched))

    with pytest.raises(GeminiOutboundDenied):
        ts.resolve_streaming_provider({"streaming": {"provider": "gemini"}})
    with pytest.raises(GeminiOutboundDenied):
        ts.resolve_streaming_provider({"provider": "gemini"})
    assert touched == []


def test_auto_streaming_never_considers_gemini(monkeypatch):
    touched = []
    monkeypatch.setattr(ts, "_gemini_key", _Tripwire("gemini_key", touched))
    for name in ("elevenlabs", "openai", "xai"):
        monkeypatch.setattr(ts._REGISTRY[name], "available", staticmethod(lambda: False))
    monkeypatch.setattr(ts, "_plugin_streamer", lambda name, cfg: None)

    assert "gemini" not in ts._PROVIDER_PRIORITY
    assert ts.resolve_streaming_provider({"provider": "edge", "streaming": {"provider": "auto"}}) is None
    assert touched == []


def test_an_openai_streamer_on_a_google_host_is_refused_before_its_key_is_read(monkeypatch):
    touched = []
    monkeypatch.setattr(ts.OpenAIStreamer, "available", staticmethod(_Tripwire("openai_available", touched)))

    with pytest.raises(GeminiOutboundDenied):
        ts.resolve_streaming_provider({"provider": "openai", "openai": {"base_url": STUDIO}})
    assert touched == []


def test_the_openai_streamer_refuses_a_google_host_before_building_its_client(monkeypatch):
    touched = []
    monkeypatch.setattr("openai.OpenAI", _Tripwire("OpenAI", touched))
    streamer = ts.OpenAIStreamer({}, {"base_url": STUDIO, "api_key": "k"})

    with pytest.raises(GeminiOutboundDenied):
        next(iter(streamer.stream("hello")))
    assert touched == []


def test_the_gemini_streamer_itself_refuses_before_its_key_or_request(monkeypatch):
    touched = []
    monkeypatch.setattr(ts, "_gemini_key", _Tripwire("gemini_key", touched))
    monkeypatch.setattr("requests.post", _Tripwire("requests.post", touched))

    with pytest.raises(GeminiOutboundDenied):
        ts.GeminiStreamer.available()
    with pytest.raises(GeminiOutboundDenied):
        next(iter(ts.GeminiStreamer({}, {}).stream("hello")))
    assert touched == []


def test_an_ordinary_openai_streamer_still_resolves(monkeypatch):
    monkeypatch.setattr(ts.OpenAIStreamer, "available", staticmethod(lambda: True))

    streamer = ts.resolve_streaming_provider(
        {"provider": "openai", "openai": {"base_url": "https://ordinary.example.test/v1"}})

    assert isinstance(streamer, ts.OpenAIStreamer)


# ── tools.tts_tool_speaker ─────────────────────────────────────────────────────────────────


def test_the_speaker_refuses_a_google_provider_before_probing_a_streamer(monkeypatch):
    touched = []
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "gemini"})
    monkeypatch.setattr(ts, "resolve_streaming_provider", _Tripwire("resolve_streaming_provider", touched))

    exc, done = _run_speaker(_text_queue(["Hello there, this is a sentence. "]))

    assert isinstance(exc, GeminiOutboundDenied)
    assert touched == [] and done.is_set()


def test_the_speaker_refuses_a_google_provider_override(monkeypatch):
    touched = []
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
    monkeypatch.setattr(ts, "resolve_streaming_provider", _Tripwire("resolve_streaming_provider", touched))

    exc, done = _run_speaker(_text_queue(["Hello there, this is a sentence. "]), provider="gemini")

    assert isinstance(exc, GeminiOutboundDenied)
    assert touched == [] and done.is_set()


def test_a_sync_synthesis_denial_reaches_the_caller_without_deadlocking_the_drain(monkeypatch):
    calls = []

    def _synth(text, output_path):
        calls.append(text)
        raise GeminiOutboundDenied()

    played = []
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
    monkeypatch.setattr(ts, "resolve_streaming_provider", lambda cfg, preferred=None: None)
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", _synth)
    monkeypatch.setattr("tools.voice_mode.play_audio_file", played.append)
    sentences = [f"This is complete sentence number {i} of many. " for i in range(8)]

    exc, done = _run_speaker(_text_queue(sentences))

    assert isinstance(exc, GeminiOutboundDenied)
    assert done.is_set() and played == []
    assert 1 <= len(calls) < len(sentences)  # later sentences are dropped once a denial is recorded


def test_the_sync_pipeline_records_the_denial_and_stops_synthesizing(monkeypatch):
    calls = []

    def _synth(text, output_path):
        calls.append(text)
        raise GeminiOutboundDenied()

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", _synth)
    pipeline = _SyncSentencePipeline(threading.Event(), lookahead=1)
    pipeline.speak("first")
    pipeline.close()

    assert isinstance(pipeline.denied, GeminiOutboundDenied)
    pipeline.speak("second")  # dropped: nothing is queued after a denial
    assert calls == ["first"]


def test_a_queued_sentence_is_not_synthesized_once_a_denial_is_recorded(monkeypatch):
    touched = []
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", _Tripwire("text_to_speech_tool", touched))
    pipeline = _SyncSentencePipeline(threading.Event(), lookahead=1)
    try:
        pipeline.denied = GeminiOutboundDenied()
        assert pipeline._synthesize_to_tmp("queued before the denial landed") is None
    finally:
        pipeline.close()
    assert touched == []


def test_streamer_playback_starts_no_stream_once_a_denial_is_recorded():
    touched = []

    class _Streamer:
        sample_rate = 24000

        def stream(self, text):
            touched.append(text)
            return iter(())

    playback = _StreamerPlayback.__new__(_StreamerPlayback)
    playback.streamer, playback.stop_event = _Streamer(), threading.Event()
    playback.denied = GeminiOutboundDenied()

    playback.speak("after the denial")
    assert touched == []


def test_a_prefetch_thread_denial_is_recorded_and_re_raised(monkeypatch):
    class _DeniesMidStream(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return True

        def stream(self, text):
            raise GeminiOutboundDenied()
            yield b""  # pragma: no cover - makes this a generator, so the raise happens on the prefetch thread

    sd = MagicMock()
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
    with patch("tools.tts_streaming.resolve_streaming_provider", return_value=_DeniesMidStream({}, {})), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd), \
         patch("tools.voice_mode.play_audio_file"):
        exc, done = _run_speaker(_text_queue(["This is the very first sentence here. ",
                                              "This is the second complete sentence. "]))

    assert isinstance(exc, GeminiOutboundDenied)
    assert done.is_set()


def test_a_streamer_that_refuses_on_start_raises_from_speak():
    class _RefusesOnStart:
        sample_rate = 24000

        def stream(self, text):
            raise GeminiOutboundDenied()

    playback = _StreamerPlayback.__new__(_StreamerPlayback)
    playback.streamer, playback.stop_event, playback.denied = _RefusesOnStart(), threading.Event(), None

    with pytest.raises(GeminiOutboundDenied):
        playback.speak("hello")


def test_an_ordinary_sync_failure_is_still_only_logged(monkeypatch):
    def _synth(text, output_path):
        raise RuntimeError("engine crashed")

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
    monkeypatch.setattr(ts, "resolve_streaming_provider", lambda cfg, preferred=None: None)
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", _synth)
    monkeypatch.setattr("tools.voice_mode.play_audio_file", lambda path: None)

    exc, done = _run_speaker(_text_queue(["This is the very first sentence here. "]))

    assert exc is None and done.is_set()


def test_an_ordinary_sync_provider_still_speaks(monkeypatch, tmp_path):
    played = []

    def _synth(text, output_path):
        with open(output_path, "wb") as fh:
            fh.write(b"ID3fake")
        return '{"success": true, "file_path": "%s"}' % output_path

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
    monkeypatch.setattr(ts, "resolve_streaming_provider", lambda cfg, preferred=None: None)
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", _synth)
    monkeypatch.setattr("tools.voice_mode.play_audio_file", played.append)

    exc, done = _run_speaker(_text_queue(["This is the very first sentence here. ",
                                          "This is the second complete sentence. "]))

    assert exc is None and done.is_set()
    assert len(played) == 2
