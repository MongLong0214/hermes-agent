"""Cron delivery failures are a fixed public value, never remediation text."""

import cron.scheduler as scheduler
from cron.scheduler import _summarize_cron_failure_for_delivery


def test_empty_chain_failure_closes_remediation_details(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", lambda: {"fallback_providers": []})
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])
    job = {"name": "semi-analyst-radar", "id": "aaa111"}
    raw_error = "Request timed out."
    msg = _summarize_cron_failure_for_delivery(job, raw_error)
    assert msg == "Cron delivery failed"
    assert raw_error not in msg
    assert "hermes fallback add" not in msg


def test_exhausted_chain_failure_closes_remediation_details(monkeypatch):
    monkeypatch.setattr(
        scheduler, "load_config",
        lambda: {"fallback_providers": [{"provider": "openrouter", "model": "x"}]},
    )
    monkeypatch.setattr(
        scheduler, "get_fallback_chain",
        lambda cfg: [{"provider": "openrouter", "model": "x"}],
    )
    job = {"name": "semi-analyst-radar", "id": "aaa111"}
    raw_error = "Request timed out."
    msg = _summarize_cron_failure_for_delivery(job, raw_error)
    assert msg == "Cron delivery failed"
    assert raw_error not in msg
    assert "hermes fallback add" not in msg


def test_rate_limit_empty_chain_closes_provider_details(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", lambda: {"fallback_providers": []})
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])
    job = {"name": "kz-coverage", "id": "bbb222"}
    raw_error = "HTTP 429: rate limit exceeded"
    msg = _summarize_cron_failure_for_delivery(job, raw_error)
    assert msg == "Cron delivery failed"
    assert raw_error not in msg
    assert "fallback" not in msg.lower()
