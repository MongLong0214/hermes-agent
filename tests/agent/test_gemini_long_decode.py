"""Gemini long decode under the Google outbound denial: the native client is refused before any stream or
hop is sent, and the hop-usage merge still splits billing from context."""

from __future__ import annotations

import pytest


def _refused_client(monkeypatch, scripts):
    """Build the native client over a transport that would answer ``scripts``; return what it sent.

    Google outbound is denied, so construction raises the typed denial before any HTTP client exists
    and nothing is sent: no stream to resume, no hop to bound."""
    from agent.gemini_native_adapter import GeminiNativeClient
    from agent.gemini_outbound_policy import GeminiOutboundDenied

    sent = []

    class DummyHTTP:
        def stream(self, method, url, json=None, headers=None, timeout=None):
            sent.append(json)
            raise AssertionError(f"streamed to Google: {scripts!r}")

        def post(self, url, json=None, headers=None, timeout=None):
            sent.append(json)
            raise AssertionError(f"posted to Google: {scripts!r}")

        def close(self):
            return None

    built = []
    monkeypatch.setattr("agent.gemini_native_adapter.httpx.Client", lambda *a, **k: built.append(1) or DummyHTTP())
    with pytest.raises(GeminiOutboundDenied):
        GeminiNativeClient(api_key="AQ.test", base_url="https://generativelanguage.googleapis.com/v1beta")
    assert built == []
    return sent


def _ld_event(text, token=None, finish=None):
    cand = {"content": {"role": "model", "parts": [{"text": text}]}}
    return {"candidates": [{**cand, **({"continuationToken": token} if token else {}), **({"finishReason": finish} if finish else {})}]}


def test_a_long_decode_stream_is_never_opened_so_nothing_resumes(monkeypatch):
    """Was: a drop after a checkpoint resends the same request plus that token. The client is refused."""
    streams = [[_ld_event("1\n"), _ld_event("2\n", token="ckpt-2"), "DROP"], [_ld_event("3\n", finish="STOP")]]
    assert _refused_client(monkeypatch, streams) == []


@pytest.mark.parametrize("streams", [
    [[_ld_event("1\n"), "DROP"]],  # before the first checkpoint
    [[_ld_event("1\n", token="ckpt-1"), _ld_event("2\n"), "DROP"]],  # "2" shown but not committed
    [[_ld_event("1\n", token="ckpt-1"), "DROP"], ["DROP"]],  # the bounded resend case
])
def test_a_dropped_stream_case_sends_no_request(monkeypatch, streams):
    assert _refused_client(monkeypatch, streams) == []


def test_no_long_decode_hop_is_ever_sent(monkeypatch):
    """Was: past ceil(budget / 32,768) + 1 hops the turn ends as ``length``. No hop is sent at all."""
    assert _refused_client(monkeypatch, ["CONTINUATION"] * 4) == []


def test_long_decode_usage_keeps_canonical_buckets_on_first_hop():
    """Context engines read the canonical buckets: input/cache_read follow the same first-hop split as
    prompt_tokens, so cache_read never exceeds the context-facing prompt."""
    from agent.gemini_long_decode import merge_hop_usage
    from agent.gemini_native_adapter import _usage_from_metadata

    hops = [{"promptTokenCount": 100, "cachedContentTokenCount": 40, "candidatesTokenCount": 32768},
            {"promptTokenCount": 32868, "cachedContentTokenCount": 32800, "candidatesTokenCount": 10}]
    usage = _usage_from_metadata(merge_hop_usage(hops))
    assert (usage.prompt_tokens, usage.prompt_tokens_details.cached_tokens) == (32968, 32840)  # billing: summed
    assert (usage.initial_prompt_tokens, usage.initial_cached_tokens) == (100, 40)
