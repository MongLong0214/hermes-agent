"""Regression for sol-audit L6-1: ``--replace`` SIGKILLed the old gateway after a flat 10s even
though its own stop() path grants in-flight cron work up to the configured ``cron_drain_timeout``
(default 30s) plus a cleanup margin. A recurring job already advanced its next slot before
execution (cron/scheduler_tick.py) and restart recovery marks an abandoned execution ``unknown``
with no retry (cron/executions.py), so cutting the drain off mid-job silently loses that run.

This exercises ``gateway.run._start_gateway_replace_existing_instance`` directly: the old process
is simulated as still alive for more polls than the old hardcoded 20-attempt/10s window allowed,
but within the cron-drain-derived window, and the test asserts the replacement never force-kills
it (``terminate_pid(..., force=True)``) before it exits on its own.
"""
import asyncio
import math

import pytest


@pytest.mark.asyncio
async def test_replace_sigterm_wait_covers_configured_cron_drain(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import run as gateway_run
    from gateway.restart import CRON_DRAIN_CLEANUP_RESERVE_S
    from gateway.run_config_loaders import GatewayConfigLoadersMixin

    # A generous configured cron drain: the OLD hardcoded 10s SIGTERM-to-SIGKILL window would
    # force-kill the old process well before this elapses.
    configured_cron_drain_s = 20.0
    monkeypatch.setattr(
        GatewayConfigLoadersMixin, "_load_cron_drain_timeout",
        classmethod(lambda cls: configured_cron_drain_s),
    )

    # Ownership guard (#89315) fixture: a bound, same-home record for target pid 42, matching
    # tests/gateway/test_replace_child_reap.py's integration test.
    monkeypatch.setattr(
        "gateway.status._read_pid_record",
        lambda path=None: {
            "pid": 42, "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway", "run"],
            "start_time": 0, "hermes_home": str(tmp_path),
        },
    )
    monkeypatch.setattr(
        "gateway.status._get_process_start_time", lambda pid: 0 if pid == 42 else None)
    monkeypatch.setattr("gateway.status._snapshot_gateway_children", lambda pid: [])
    monkeypatch.setattr(
        "gateway.status.reap_gateway_children",
        lambda children, *, parent_pid, timeout=5.0: 0)
    monkeypatch.setattr("gateway.status.release_all_scoped_locks", lambda **kwargs: 0)
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)

    terminate_calls = []

    def fake_terminate_pid(pid, force=False, expected_start_time=None):
        terminate_calls.append((pid, force))

    monkeypatch.setattr("gateway.status.terminate_pid", fake_terminate_pid)

    # The old process "drains" for more polls than the OLD 20-attempt/10s cutoff allowed, then
    # exits on its own — simulating a cron job that finishes mid-drain rather than being SIGKILLed.
    exit_after_polls = 25
    poll_count = {"n": 0}

    def fake_pid_exists(pid):
        poll_count["n"] += 1
        return poll_count["n"] <= exit_after_polls

    monkeypatch.setattr("gateway.status._pid_exists", fake_pid_exists)

    # Real timing would make this test take ~12.5s; the poll COUNT already proves the window,
    # so collapse the sleep between polls instead of waiting for it.
    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(gateway_run.asyncio, "sleep", no_sleep)

    result = await gateway_run._start_gateway_replace_existing_instance(42, True)

    assert result is True, "the old process exited on its own; replacement should report success"
    assert all(not force for _, force in terminate_calls), (
        f"the old process was force-killed (SIGKILL) before its cron-drain window elapsed: "
        f"{terminate_calls}"
    )
    old_hardcoded_attempts = 20  # 20 * 0.5s = 10s, the pre-fix flat window
    assert poll_count["n"] > old_hardcoded_attempts, (
        f"only polled {poll_count['n']} times — at or below the OLD hardcoded "
        f"{old_hardcoded_attempts}-attempt window, so this run would not have caught the bug"
    )
    # The new window itself must actually be derived from the configured cron drain, not just
    # "bigger than 20": it should sit at ceil((cron_drain + cleanup_margin) / poll_interval).
    expected_attempts = math.ceil((configured_cron_drain_s + CRON_DRAIN_CLEANUP_RESERVE_S) / 0.5)
    assert exit_after_polls < expected_attempts, "test fixture must exit within the new window"


@pytest.mark.asyncio
async def test_replace_sigterm_wait_still_force_kills_a_genuinely_wedged_process(
    monkeypatch, tmp_path
):
    """The forced-kill path must stay bounded: a process that never exits is still SIGKILLed,
    just after the cron-drain-derived window instead of a flat 10s."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import run as gateway_run
    from gateway.run_config_loaders import GatewayConfigLoadersMixin

    monkeypatch.setattr(
        GatewayConfigLoadersMixin, "_load_cron_drain_timeout", classmethod(lambda cls: 1.0))

    monkeypatch.setattr(
        "gateway.status._read_pid_record",
        lambda path=None: {
            "pid": 42, "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway", "run"],
            "start_time": 0, "hermes_home": str(tmp_path),
        },
    )
    monkeypatch.setattr(
        "gateway.status._get_process_start_time", lambda pid: 0 if pid == 42 else None)
    monkeypatch.setattr("gateway.status._snapshot_gateway_children", lambda pid: [])
    monkeypatch.setattr(
        "gateway.status.reap_gateway_children",
        lambda children, *, parent_pid, timeout=5.0: 0)
    monkeypatch.setattr("gateway.status.release_all_scoped_locks", lambda **kwargs: 0)
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)

    terminate_calls = []
    # Alive forever through the graceful wait, dead only once force=True has been requested.
    forced = {"done": False}

    def fake_terminate_pid(pid, force=False, expected_start_time=None):
        terminate_calls.append((pid, force))
        if force:
            forced["done"] = True

    monkeypatch.setattr("gateway.status.terminate_pid", fake_terminate_pid)
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: not forced["done"])

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(gateway_run.asyncio, "sleep", no_sleep)

    result = await gateway_run._start_gateway_replace_existing_instance(42, True)

    assert result is True
    assert (42, True) in terminate_calls, "a genuinely wedged process must still be SIGKILLed"
