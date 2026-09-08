"""Tests for Gemini free-tier detection and blocking."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


from agent.gemini_native_adapter import (
    gemini_http_error,
    is_free_tier_quota_error,
    probe_gemini_tier,
)


def _mock_response(status: int, headers: dict | None = None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.text = text
    return resp


def _run_probe(resp: MagicMock) -> str:
    with patch("agent.gemini_native_adapter.httpx.Client") as MC:
        inst = MagicMock()
        inst.post.return_value = resp
        MC.return_value.__enter__.return_value = inst
        return probe_gemini_tier("fake-key")


class TestProbeGeminiTier:
    def test_probe_denies_before_http_client_and_propagates_canonical_error(self, monkeypatch):
        import pytest

        import agent.gemini_native_adapter as adapter
        from agent.gemini_outbound_policy import GeminiOutboundDenied

        class ExplodingClient:
            def __init__(self, *args, **kwargs):
                raise AssertionError("httpx.Client must not be created")

        monkeypatch.setattr(adapter.httpx, "Client", ExplodingClient)

        with pytest.raises(GeminiOutboundDenied) as excinfo:
            adapter.probe_gemini_tier("fake-key")

        exc = excinfo.value
        assert type(exc) is GeminiOutboundDenied
        assert str(exc) == "Gemini outbound requests are disabled."
        assert exc.code == "gemini_outbound_denied"
        assert vars(exc) == {}








class TestIsFreeTierQuotaError:
    def test_detects_free_tier_marker(self):
        assert is_free_tier_quota_error(
            "Quota exceeded for metric: generate_content_free_tier_requests"
        )


    def test_no_free_tier_marker(self):
        assert not is_free_tier_quota_error("rate limited")


    def test_none(self):
        assert not is_free_tier_quota_error(None)  # type: ignore[arg-type]


class TestGeminiHttpErrorFreeTierGuidance:
    """gemini_http_error should append free-tier guidance for free-tier 429s."""

    class _FakeResp:
        def __init__(self, status: int, text: str):
            self.status_code = status
            self.headers: dict = {}
            self.text = text

    def test_free_tier_429_appends_guidance(self):
        body = (
            '{"error":{"code":429,"message":"Quota exceeded for metric: '
            "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
            'limit: 20","status":"RESOURCE_EXHAUSTED"}}'
        )
        err = gemini_http_error(self._FakeResp(429, body))
        msg = str(err)
        assert "free tier" in msg.lower()
        assert "aistudio.google.com/apikey" in msg

    def test_paid_429_has_no_billing_url(self):
        body = '{"error":{"code":429,"message":"Rate limited","status":"RESOURCE_EXHAUSTED"}}'
        err = gemini_http_error(self._FakeResp(429, body))
        assert "aistudio.google.com/apikey" not in str(err)


