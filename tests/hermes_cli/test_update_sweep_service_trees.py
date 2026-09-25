"""``hermes update``'s gateway sweeps never signal a process a service manager owns.

Regression: the manual-gateway sweep and the post-restart survivor sweep spared only the PIDs the
service manager reports. On macOS that PID is the launchd job, the ``stderr_timestamp`` wrapper,
and the gateway runs as its child; both argvs match the gateway scan. So the sweep took the
supervised gateway for a manual one and sent it SIGUSR1 with a SIGTERM fallback after the launchd
restart had already replaced it, and a gateway of another install got a bare SIGTERM. The orphan
reaper was fixed for the same shape in 722ec84758; the sweeps now share its exclusion.

The process table is fake: nothing is spawned and nothing is signalled.
"""

from types import SimpleNamespace

import pytest

import gateway.status as gateway_status
import hermes_cli.gateway as gateway
from hermes_cli import update_cmd_fleet

# Service PID -> its descendants: this install's launchd job (500) runs gateway 501, which maps to
# the default profile's PID file; another install's job (600) runs gateway 601, which maps to none.
_SERVICE_TREES = {500: [501], 600: [601]}
_MANUAL_GATEWAY = 700
_GATEWAY_ARGV_PIDS = [500, 501, 600, 601, _MANUAL_GATEWAY]


@pytest.fixture
def process_table(monkeypatch, tmp_path):
    # The restart phase just kickstarted job 500: it respawns while the sweep's process scan runs, so
    # launchd reports job 500 and its gateway 501 exists only from the scan on.
    started = set()

    def scan(exclude_pids, all_profiles=False, include_restart_managers=False):
        started.update((500, 501))
        return [pid for pid in _GATEWAY_ARGV_PIDS if pid not in exclude_pids]

    def service_pids(all_profiles=False):
        return {pid for pid in _SERVICE_TREES if pid != 500 or pid in started}

    def children(pid):
        return [SimpleNamespace(pid=c) for c in _SERVICE_TREES.get(pid, []) if c != 501 or c in started]

    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway, "_get_service_pids", service_pids)
    monkeypatch.setattr(gateway_status, "_snapshot_gateway_children", children)
    monkeypatch.setattr(gateway, "_scan_gateway_pids", scan)
    mapped = [
        gateway.ProfileGatewayProcess("default", tmp_path / "default", 501),
        gateway.ProfileGatewayProcess("work", tmp_path / "work", _MANUAL_GATEWAY),
    ]
    monkeypatch.setattr(
        gateway,
        "find_profile_gateway_processes",
        lambda exclude_pids=None, **_kw: [proc for proc in mapped if proc.pid not in (exclude_pids or set())],
    )


def test_update_sweep_signals_only_gateways_outside_every_service_tree(process_table, monkeypatch):
    signalled = []
    monkeypatch.setattr(
        update_cmd_fleet,
        "_drain_or_signal_gateway_for_update",
        lambda pid, _budget, _label: signalled.append(pid) or True,
    )
    monkeypatch.setattr(update_cmd_fleet.os, "kill", lambda pid, _sig: signalled.append(pid))
    monkeypatch.setattr(gateway, "_prepare_profile_gateway_update_restart", lambda _profile, _pid: "detached")
    monkeypatch.setattr(gateway, "_wait_for_gateway_exit", lambda **_kw: None)
    out = update_cmd_fleet._GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[], restarted_services=[],
        failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set(),
    )

    update_cmd_fleet._restart_manual_gateways(out, 1.0)

    assert sorted(signalled) == [_MANUAL_GATEWAY]
    assert out.killed_pids == {_MANUAL_GATEWAY}


def test_survivor_sweep_never_force_kills_a_service_owned_process(process_table, monkeypatch):
    force_killed = []
    monkeypatch.setattr(
        gateway_status, "terminate_pid", lambda pid, force=False, expected_start_time=None: force_killed.append(pid)
    )
    monkeypatch.setattr(gateway_status, "get_process_start_time", lambda _pid: None)
    monkeypatch.setattr(update_cmd_fleet, "_time", SimpleNamespace(sleep=lambda _s: None))

    # killed_pids is written before the service jobs finish respawning, so a PID in it can since have
    # been reused by a service job's gateway; the SIGKILL still reaches only gateways outside every tree.
    update_cmd_fleet._force_kill_stuck_gateways({501, 601, _MANUAL_GATEWAY})

    assert force_killed == [_MANUAL_GATEWAY]
