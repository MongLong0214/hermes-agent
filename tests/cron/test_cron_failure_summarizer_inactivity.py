"""Cron delivery failures close watchdog, provider, and lock diagnostics."""

import cron.scheduler as scheduler
from cron.scheduler import _summarize_cron_failure_for_delivery


def test_inactivity_timeout_is_closed_at_delivery_boundary():
    job = {"name": "Daily Repo Sweep", "id": "82d65bdd5ba9"}
    error = (
        "TimeoutError: Cron job 'Daily Repo Sweep' idle for 1239s "
        "(limit 600s) — last activity: terminal command running (30s elapsed)"
    )
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert msg == "Cron delivery failed"
    assert error not in msg
    assert "provider timeout" not in msg
    assert "fallback chain" not in msg.lower()
    assert "Daily Repo Sweep" not in msg


def test_genuine_provider_timeout_with_no_fallback_configured(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", lambda: {"fallback_providers": []})
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])
    job = {"name": "CI Autofix Poller", "id": "f7fe78574bda"}
    error = "Request timed out."
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert msg == "Cron delivery failed"
    assert error not in msg
    assert "fallback" not in msg.lower()


def test_genuine_provider_timeout_with_fallback_configured(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", lambda: {
        "fallback_providers": [{"provider": "openrouter", "model": "anthropic/claude-sonnet-5"}]
    })
    monkeypatch.setattr(
        scheduler,
        "get_fallback_chain",
        lambda cfg: [{"provider": "openrouter", "model": "anthropic/claude-sonnet-5"}],
    )
    job = {"name": "CI Autofix Poller", "id": "f7fe78574bda"}
    error = "Request timed out."
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert msg == "Cron delivery failed"
    assert error not in msg
    assert "fallback" not in msg.lower()


def test_fallback_chain_phrase_fails_open_on_config_error(monkeypatch):
    def _raise():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(scheduler, "load_config", _raise)
    assert scheduler._fallback_chain_phrase() == "Fallback chain was exhausted or unavailable."


def test_readtimeout_error_is_closed_at_delivery_boundary(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", lambda: {"fallback_providers": []})
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])
    job = {"name": "some-job", "id": "abc123"}
    error = "httpx.ReadTimeout: The read operation timed out"
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert msg == "Cron delivery failed"
    assert error not in msg
    assert "ReadTimeout" not in msg


def test_rate_limit_error_is_closed_at_delivery_boundary(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", lambda: {"fallback_providers": []})
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])
    job = {"name": "some-job", "id": "abc123"}
    error = "HTTP 429: weekly usage limit exceeded"
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert msg == "Cron delivery failed"
    assert error not in msg
    assert "weekly usage limit" not in msg


def test_terminal_cwd_lock_timeout_is_closed_at_delivery_boundary():
    """Lock-wait diagnostics are never public delivery text."""
    job = {"name": "Workdir Job", "id": "abc123def456"}
    error = (
        "TimeoutError: Timed out waiting for the TERMINAL_CWD write lock "
        "after 600s — another cron job (a workdir writer, or long-running "
        "readers) has held it for longer than the cron inactivity limit. "
        "If a workdir job is the holder, stagger its schedule or remove its "
        "workdir to unblock this job (#79768)."
    )
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert msg == "Cron delivery failed"
    assert error not in msg
    assert "provider timeout" not in msg
    assert "fallback chain" not in msg.lower()
    assert "working-directory lock" not in msg
    assert "Workdir Job" not in msg
