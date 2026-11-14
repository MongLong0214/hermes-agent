"""Tests for Gemini free-tier detection and blocking."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.gemini_native_adapter import (
    gemini_http_error,
    is_free_tier_quota_error,
    probe_gemini_tier,
)
from agent.gemini_outbound_policy import GeminiOutboundDenied


def _mock_response(status: int, headers: dict | None = None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.text = text
    return resp


def _assert_probe_refused(resp: MagicMock) -> None:
    """The tier probe is Google outbound: refused before an HTTP client exists, whatever Google would answer."""
    with patch("agent.gemini_native_adapter.httpx.Client") as MC:
        MC.return_value.__enter__.return_value.post.return_value = resp
        with pytest.raises(GeminiOutboundDenied):
            probe_gemini_tier("fake-key")
    MC.assert_not_called()


class TestProbeGeminiTier:
    """The probe never runs: it would send the key to Google."""

    def test_a_free_tier_429_answer_is_never_requested(self):
        body = (
            '{"error":{"code":429,"message":"Quota exceeded for metric: '
            'generativelanguage.googleapis.com/generate_content_free_tier_requests, '
            'limit: 20"}}'
        )
        _assert_probe_refused(_mock_response(429, {}, body))

    def test_a_paid_200_answer_is_never_requested(self):
        _assert_probe_refused(_mock_response(200, {}, '{"candidates":[]}'))


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


