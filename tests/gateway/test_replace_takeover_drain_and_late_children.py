"""Regressions for sol-audit R68-3 and R68-4 (both takeover paths: ``gateway run --replace`` and an
adapter's explicit credential-lock handoff, ``gateway.status.take_over_scoped_lock_holder``).

R68-3: the SIGTERM-to-SIGKILL wait did not cover the old instance's own stop budget. ``--replace``
read the *new* process's ``cron_drain_timeout`` and added the cleanup reserve, but the old
process's stop path waits ``max(restart_drain_timeout, cron floor)`` (gateway/restart.py's
``resolve_cron_drain_budget``) with the timeouts it captured at startup; the credential-lock path
still SIGKILLed after a flat 10 s. A simulated clock drives an old instance that would exit on its
own at 45 s.

R68-4: children the old gateway creates after the pre-SIGTERM snapshot (e.g. in its SIGTERM
handler, or during the drain) were never reaped and kept running reparented to init. The witness
is a real subprocess that spawns a child in its SIGTERM handler and exits.
"""
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway import status

_OLD_PID = 42
_OLD_START = 0
_NATURAL_EXIT_S = 45.0  # the old instance finishes its in-flight cron job and exits here
# Published-record contract (gateway.status): the env var a gateway stamps on its children.
_LINEAGE_ENV = "_HERMES_GATEWAY_LINEAGE"


def _hand_built_record(home: Path, *, pid: int = _OLD_PID, start_time: int = _OLD_START, **extra):
    """A gateway PID record as an older build wrote it (no published stop budget)."""
    return {"pid": pid, "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway", "run"],
            "start_time": start_time, "hermes_home": str(home), **extra}


def _published_pid_record(monkeypatch, home: Path, *, drain_s: float, cron_s: float) -> dict:
    """The PID record a gateway started with ``drain_s``/``cron_s`` writes when it claims its
    home, produced by the real claim path (locks and the host role stubbed out)."""
    from gateway import run as gateway_run

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv(_LINEAGE_ENV, "inherited-from-a-launcher")
    runner = types.SimpleNamespace(_restart_drain_timeout=drain_s, _cron_drain_timeout=cron_s)
    with patch("gateway.status.get_running_pid", return_value=None), \
         patch("gateway.status.acquire_gateway_runtime_lock", return_value=True), \
         patch.object(gateway_run, "_claim_host_gateway_role", lambda force=False: None), \
         patch("atexit.register", lambda *a, **k: None):
        assert gateway_run._start_gateway_claim_pid_file(force=False, runner=runner) is True
    record = json.loads((home / "gateway.pid").read_text())
    (home / "gateway.pid").unlink()
    assert record.get("lineage") and record["lineage"] == os.environ[_LINEAGE_ENV] != (
        "inherited-from-a-launcher"), "the claim must stamp a fresh lineage for its children"
    return record


class _SimulatedOldInstance:
    """Clock + liveness for an old gateway that exits on its own at ``exit_at`` unless SIGKILLed."""

    def __init__(self, exit_at: float):
        self.now = 0.0
        self.exit_at = exit_at
        self.signals = []

    def alive(self, pid) -> bool:
        return not any(force for _, force in self.signals) and self.now < self.exit_at

    def advance(self, seconds) -> None:
        self.now += max(0.0, float(seconds))

    def terminate(self, pid, force=False, expected_start_time=None):
        self.signals.append((self.now, force))


def _stub_replace_io(monkeypatch, record: dict) -> None:
    monkeypatch.setattr("gateway.status._read_pid_record", lambda path=None: dict(record))
    monkeypatch.setattr(
        "gateway.status._get_process_start_time",
        lambda pid: _OLD_START if pid == record["pid"] else None)
    monkeypatch.setattr("gateway.status.release_all_scoped_locks", lambda **kwargs: 0)
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)


async def _replace_simulated(
    monkeypatch, record: dict, exit_at: float = _NATURAL_EXIT_S,
) -> tuple[bool, _SimulatedOldInstance]:
    from gateway import run as gateway_run

    old = _SimulatedOldInstance(exit_at)
    _stub_replace_io(monkeypatch, record)
    monkeypatch.setattr("gateway.status._snapshot_gateway_children", lambda pid: [])
    monkeypatch.setattr(
        "gateway.status.reap_gateway_children", lambda children, *, parent_pid, timeout=5.0: 0)
    monkeypatch.setattr("gateway.status.terminate_pid", old.terminate)
    monkeypatch.setattr("gateway.status._pid_exists", old.alive)

    async def simulated_sleep(seconds):
        old.advance(seconds)

    monkeypatch.setattr(gateway_run.asyncio, "sleep", simulated_sleep)
    return await gateway_run._start_gateway_replace_existing_instance(record["pid"], True), old


def _as_old_target(record: dict, home: Path, pid: int = _OLD_PID) -> dict:
    """The published fields of ``record`` on the identity of the simulated old gateway."""
    return {**record, **_hand_built_record(home, pid=pid)}


@pytest.mark.asyncio
async def test_replace_waits_out_a_chat_drain_longer_than_the_cron_floor(monkeypatch, tmp_path):
    """Unchanged config ``restart_drain_timeout=50`` / ``cron_drain_timeout=30``: the old stop
    path grants cron work 50 s, so cron floor + reserve (40 s) cuts it off."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_RESTART_DRAIN_TIMEOUT", "50")
    monkeypatch.setenv("HERMES_CRON_DRAIN_TIMEOUT", "30")

    replaced, old = await _replace_simulated(monkeypatch, _hand_built_record(tmp_path))

    assert replaced is True
    assert old.signals == [(0.0, False)], f"force-killed before its natural exit: {old.signals}"


@pytest.mark.asyncio
async def test_replace_honours_the_budget_the_old_instance_started_with(monkeypatch, tmp_path):
    """The old instance captured ``restart_drain_timeout=50`` at startup (stop leash 110 s); the
    config was edited to 0 since. Sizing the wait from the edited value (a 60 s leash) would
    SIGKILL an old instance still inside its own budget at 90 s."""
    record = _as_old_target(
        _published_pid_record(monkeypatch, tmp_path, drain_s=50.0, cron_s=30.0), tmp_path)
    monkeypatch.setenv("HERMES_RESTART_DRAIN_TIMEOUT", "0")
    monkeypatch.setenv("HERMES_CRON_DRAIN_TIMEOUT", "30")

    replaced, old = await _replace_simulated(monkeypatch, record, exit_at=90.0)

    assert replaced is True
    assert old.signals == [(0.0, False)], f"force-killed before its natural exit: {old.signals}"


@pytest.mark.asyncio
async def test_replace_still_force_kills_an_instance_that_never_exits(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record = _hand_built_record(tmp_path)
    old_never_exits = float("inf")
    from gateway import run as gateway_run

    old = _SimulatedOldInstance(old_never_exits)
    _stub_replace_io(monkeypatch, record)
    monkeypatch.setattr("gateway.status._snapshot_gateway_children", lambda pid: [])
    monkeypatch.setattr(
        "gateway.status.reap_gateway_children", lambda children, *, parent_pid, timeout=5.0: 0)
    monkeypatch.setattr("gateway.status.terminate_pid", old.terminate)
    monkeypatch.setattr("gateway.status._pid_exists", old.alive)

    async def simulated_sleep(seconds):
        old.advance(seconds)

    monkeypatch.setattr(gateway_run.asyncio, "sleep", simulated_sleep)

    assert await gateway_run._start_gateway_replace_existing_instance(_OLD_PID, True) is True
    forced_at = [at for at, force in old.signals if force]
    # A legacy record (no published stop_leash_s) cannot prove its owner's drain budget, so the
    # takeover waits out a long conservative ceiling instead of a short guess (R68-3) -- but a
    # genuinely hung owner must still be force-killed eventually, not waited on forever.
    from gateway.status import _LEGACY_RECORD_ASSUMED_DRAIN_S
    assert forced_at and forced_at[0] < _LEGACY_RECORD_ASSUMED_DRAIN_S + 600, (
        f"the forced kill must stay bounded: {old.signals}")


@pytest.mark.parametrize("published, exit_at", [
    # An older build's record: no published leash; the wait still covers this short natural exit.
    (False, _NATURAL_EXIT_S),
    # The owner published a 110 s leash (restart_drain_timeout=50); this side's config says 0.
    (True, 90.0),
    # Round-2 R68-3: an older build's record cannot prove its owner's budget, so this side's 0 s
    # drain (a 62 s wait) must not be taken as the owner's: it started with 50 s, exits at 90 s.
    (False, 90.0),
    # Round-3 R68-3: the reviewer's own reproduction -- an older owner configured for a 600 s
    # drain naturally exits at 450 s; the previous 300 s floor force-killed it at ~362 s.
    (False, 450.0),
])
def test_credential_lock_takeover_waits_out_the_owners_drain(
    monkeypatch, tmp_path, published, exit_at,
):
    """The adapter-startup handoff (gateway/platforms/base.py) used twenty half-second polls."""
    target_home = tmp_path / "target"
    target_home.mkdir()
    record = _hand_built_record(target_home, pid=4242, start_time=123)
    if published:
        record = {**_published_pid_record(monkeypatch, target_home, drain_s=50.0, cron_s=30.0),
                  **record}
    (target_home / "gateway.pid").write_text(json.dumps(record))
    replacer_home = tmp_path / "replacer"
    replacer_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(replacer_home))
    monkeypatch.setenv("HERMES_RESTART_DRAIN_TIMEOUT", "0")

    old = _SimulatedOldInstance(exit_at)
    monkeypatch.setattr(status, "_pid_exists", old.alive)
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)
    monkeypatch.setattr(
        status, "_read_process_cmdline", lambda pid: "python -m hermes_cli.main gateway run")
    monkeypatch.setattr(status, "_snapshot_gateway_children", lambda pid: [])
    monkeypatch.setattr(
        status, "reap_gateway_children", lambda children, *, parent_pid, timeout=5.0: 0)
    monkeypatch.setattr(status, "terminate_pid", old.terminate)
    monkeypatch.setattr(status, "time", types.SimpleNamespace(
        **{name: getattr(time, name) for name in dir(time)
           if not name.startswith("_") and name != "sleep"},
        sleep=old.advance))

    assert status.take_over_scoped_lock_holder(record) == 4242
    assert old.signals == [(0.0, False)], f"force-killed before its natural exit: {old.signals}"


_OLD_GATEWAY_SCRIPT = r"""
import os, signal, subprocess, sys, time
out_path, ready_path, lineage_env = sys.argv[1], sys.argv[2], sys.argv[3]
sleeper = [sys.executable, "-c", "import time; time.sleep(120)"]

def on_term(signum, frame):
    # Created AFTER the replacer's pre-SIGTERM snapshot: a drain-time child.
    late = subprocess.Popen(sleeper)
    # A restart-safe worker is launched without the gateway's lineage, in its own session.
    worker_env = {k: v for k, v in os.environ.items() if k != lineage_env}
    worker = subprocess.Popen(sleeper, env=worker_env, start_new_session=True)
    with open(out_path, "w") as fh:
        fh.write(f"{late.pid} {worker.pid}")
    os._exit(0)

signal.signal(signal.SIGTERM, on_term)
open(ready_path, "w").write("ready")
while True:
    time.sleep(0.05)
"""


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _gone(pid: int) -> bool:
    import psutil
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX reap path")
# Real signals to processes this test spawned; once reparented they are outside the test subtree.
# The unique per-run lineage token keeps the reap away from every other process on the host.
@pytest.mark.live_system_guard_bypass
@pytest.mark.asyncio
async def test_replace_reaps_a_child_created_after_sigterm(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record = _hand_built_record(tmp_path, lineage=f"test-lineage-{os.getpid()}-{time.time_ns()}")
    out_path, ready_path = tmp_path / "children", tmp_path / "ready"
    # The real old gateway stamps this on itself when it claims its home (see the claim test
    # above); its children inherit it.
    env = {**os.environ, _LINEAGE_ENV: record["lineage"]}
    old = subprocess.Popen(
        [sys.executable, "-c", _OLD_GATEWAY_SCRIPT, str(out_path), str(ready_path),
         _LINEAGE_ENV], env=env)
    late_pid = worker_pid = None
    try:
        assert _wait_for(ready_path.exists), "the simulated old gateway never became ready"
        target = {**record, "pid": old.pid, "start_time": _OLD_START, "hermes_home": str(tmp_path)}
        _stub_replace_io(monkeypatch, target)
        # The replacer's poll must see the zombie as gone; Popen.poll() reaps it for _pid_exists.
        real_pid_exists = status._pid_exists
        monkeypatch.setattr(
            "gateway.status._pid_exists",
            lambda pid: old.poll() is None if pid == old.pid else real_pid_exists(pid))

        from gateway import run as gateway_run
        assert await gateway_run._start_gateway_replace_existing_instance(old.pid, True) is True

        late_pid, worker_pid = map(int, out_path.read_text().split())
        assert _wait_for(lambda: _gone(late_pid), timeout=10), (
            "a child the old gateway created after SIGTERM survived the replacement")
        assert not _gone(worker_pid), "a restart-safe worker (no gateway lineage) must survive"
    finally:
        if old.poll() is None:
            old.kill()
            old.wait(5)
        for pid in (late_pid, worker_pid):
            if pid:
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass


@pytest.mark.asyncio
async def test_replace_does_not_size_a_legacy_owners_deadline_from_its_own_config(monkeypatch, tmp_path):
    """Round-2 R68-3: a record from an older build publishes no stop leash. The old instance
    started with ``restart_drain_timeout=50`` (110 s leash) and exits on its own at 90 s; this
    side's config now says 0. Reusing it (a 62 s wait) SIGKILLs an instance inside its own budget."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_RESTART_DRAIN_TIMEOUT", "0")

    replaced, old = await _replace_simulated(monkeypatch, _hand_built_record(tmp_path), exit_at=90.0)

    assert replaced is True
    assert old.signals == [(0.0, False)], f"force-killed before its natural exit: {old.signals}"


# Round-2 R68-4 and ROUND1-ESCAPE-1, on both takeover paths and both record shapes. The old gateway
# runs with the environment a real one has (``_HERMES_GATEWAY=1`` is set at gateway/run.py import,
# HERMES_HOME names its home). Before SIGTERM it already has a plain child (must be reaped) and a
# restart-safe cron worker (``--external-worker-file``, the real launcher's argv) and a kanban worker
# (``HERMES_KANBAN_TASK``, kanban_db_dispatch's env) that must both survive;
# its SIGTERM handler creates one more child and exits at once (must be reaped).
_OLD_GATEWAY_WITH_EARLY_CHILDREN_SCRIPT = r"""
import os, signal, subprocess, sys, time
out_path, ready_path, lineage_env = sys.argv[1], sys.argv[2], sys.argv[3]
sleeper = [sys.executable, "-c", "import time; time.sleep(120)"]
early = subprocess.Popen(sleeper)
worker_env = {k: v for k, v in os.environ.items() if k != lineage_env}
worker = subprocess.Popen(sleeper + ["--external-worker-file", "payload.json"], env=worker_env,
                          start_new_session=True)
kanban = subprocess.Popen(sleeper, env={**worker_env, "HERMES_KANBAN_TASK": "t-1"},
                          start_new_session=True)

def on_term(signum, frame):
    late = subprocess.Popen(sleeper)
    with open(out_path, "w") as fh:
        fh.write(f"{early.pid} {worker.pid} {kanban.pid} {late.pid}")
    os._exit(0)

signal.signal(signal.SIGTERM, on_term)
open(ready_path, "w").write("ready")
while True:
    time.sleep(0.05)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX reap path")
# Real signals to processes this test spawned; a guard below refuses to reap anything else.
@pytest.mark.live_system_guard_bypass
@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["replace", "credential_lock"])
@pytest.mark.parametrize("with_lineage", [False, True], ids=["legacy-record", "lineage-record"])
async def test_takeover_reaps_late_children_and_spares_restart_safe_workers(
    monkeypatch, tmp_path, path, with_lineage,
):
    home = tmp_path / "home"
    home.mkdir()
    lineage = f"test-lineage-{os.getpid()}-{time.time_ns()}" if with_lineage else None
    out_path, ready_path = tmp_path / "children", tmp_path / "ready"
    env = {k: v for k, v in os.environ.items() if k != _LINEAGE_ENV}
    env.update({"_HERMES_GATEWAY": "1", "HERMES_HOME": str(home)})
    if lineage:
        env[_LINEAGE_ENV] = lineage
    old = subprocess.Popen(
        [sys.executable, "-c", _OLD_GATEWAY_WITH_EARLY_CHILDREN_SCRIPT, str(out_path),
         str(ready_path), _LINEAGE_ENV], env=env)
    spawned = []
    try:
        assert _wait_for(ready_path.exists), "the simulated old gateway never became ready"
        record = _hand_built_record(home, pid=old.pid)
        if lineage:
            record["lineage"] = lineage
        real_pid_exists, real_reap = status._pid_exists, status.reap_gateway_children

        def guarded_reap(children, *, parent_pid, timeout=5.0):
            mine = set(map(int, out_path.read_text().split()))
            stray = {c.pid for c in children} - mine
            assert not stray, f"the reap reached processes this test did not spawn: {stray}"
            return real_reap(children, parent_pid=parent_pid, timeout=timeout)

        monkeypatch.setattr(status, "reap_gateway_children", guarded_reap)
        monkeypatch.setattr(
            status, "_pid_exists",
            lambda pid: old.poll() is None if pid == old.pid else real_pid_exists(pid))
        if path == "replace":
            monkeypatch.setenv("HERMES_HOME", str(home))
            _stub_replace_io(monkeypatch, record)
            from gateway import run as gateway_run
            assert await gateway_run._start_gateway_replace_existing_instance(old.pid, True) is True
        else:
            (home / "gateway.pid").write_text(json.dumps(record))
            replacer_home = tmp_path / "replacer"
            replacer_home.mkdir()
            monkeypatch.setenv("HERMES_HOME", str(replacer_home))
            monkeypatch.setattr(
                status, "_scoped_lock_owner_state",
                lambda pid, start: "same" if old.poll() is None else "exited")
            monkeypatch.setattr(
                status, "_read_process_cmdline", lambda pid: "python -m hermes_cli.main gateway run")
            assert status.take_over_scoped_lock_holder(record) == old.pid

        early_pid, worker_pid, kanban_pid, late_pid = spawned[:] = map(
            int, out_path.read_text().split())
        if with_lineage:
            assert _wait_for(lambda: _gone(late_pid), timeout=10), (
                "a child the old gateway created after SIGTERM survived the takeover")
        else:
            # R68-5: a legacy record has no lineage token to prove ownership, so the late-child
            # scan now requires real OS ancestry (still a direct child of the old gateway's PID at
            # scan time) rather than env-marker + home + a coarse time window -- the latter could
            # match an unrelated process and kill it (see
            # test_takeover_spares_an_unrelated_process_with_matching_markers). By the time this
            # scan runs the old gateway has already exited and late_pid has typically been
            # reparented to init, so it is no longer provably this gateway's child and is correctly
            # left alone: a known, accepted narrowing of R68-4's best-effort reach in exchange for
            # never signalling a process this gateway did not actually spawn.
            time.sleep(0.5)
            assert not _gone(late_pid), (
                "a legacy-record late child with no provable ancestry must be left alone, not killed")
        assert _wait_for(lambda: _gone(early_pid), timeout=10), "a pre-SIGTERM child survived"
        assert not _gone(worker_pid), (
            "a restart-safe worker already running before SIGTERM must outlive the takeover")
        assert not _gone(kanban_pid), "a kanban worker must outlive the takeover"
    finally:
        if old.poll() is None:
            old.kill()
            old.wait(5)
        for pid in spawned:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["replace", "credential_lock"])
async def test_takeover_spares_an_unrelated_process_with_matching_markers(monkeypatch, tmp_path, path):
    """R68-5: a legacy record (no lineage token) cannot prove ownership through env markers and a
    coarse creation-time window alone -- an independently spawned process that merely happens to
    carry the same _HERMES_GATEWAY/HERMES_HOME markers, started in the same time window, must not
    be mistaken for one of the old gateway's own children and killed."""
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if k != _LINEAGE_ENV}
    env.update({"_HERMES_GATEWAY": "1", "HERMES_HOME": str(home)})
    ready_path = tmp_path / "old_ready"
    old = subprocess.Popen(
        [sys.executable, "-c",
         f"open({str(ready_path)!r}, 'w').close(); import time; time.sleep(60)"], env=env)
    # Independently spawned -- NOT a child of `old` -- but carries the same markers and starts in
    # the same window, which is exactly what the legacy env-only scan used to match on.
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], env=dict(env))
    try:
        assert _wait_for(ready_path.exists), "the simulated old gateway never became ready"
        record = _hand_built_record(home, pid=old.pid)
        real_pid_exists = status._pid_exists
        monkeypatch.setattr(
            status, "_pid_exists",
            lambda pid: old.poll() is None if pid == old.pid else real_pid_exists(pid))
        if path == "replace":
            monkeypatch.setenv("HERMES_HOME", str(home))
            _stub_replace_io(monkeypatch, record)
            from gateway import run as gateway_run
            assert await gateway_run._start_gateway_replace_existing_instance(old.pid, True) is True
        else:
            (home / "gateway.pid").write_text(json.dumps(record))
            replacer_home = tmp_path / "replacer"
            replacer_home.mkdir()
            monkeypatch.setenv("HERMES_HOME", str(replacer_home))
            monkeypatch.setattr(
                status, "_scoped_lock_owner_state",
                lambda pid, start: "same" if old.poll() is None else "exited")
            monkeypatch.setattr(
                status, "_read_process_cmdline", lambda pid: "python -m hermes_cli.main gateway run")
            assert status.take_over_scoped_lock_holder(record) == old.pid
        time.sleep(1.0)  # give any (wrong) reap attempt a moment to land
        assert not _gone(unrelated.pid), "an unrelated process with matching markers was killed"
    finally:
        for p in (old, unrelated):
            if p.poll() is None:
                p.kill()
                p.wait(5)


def test_restart_watcher_env_is_marked_restart_safe():
    """ROUND1-ESCAPE-1: the gateway's own detached restart watcher has neither _HERMES_GATEWAY nor
    a lineage token (by design -- it must outlive the gateway it is waiting to restart), so without
    its own marker it was indistinguishable from an ordinary stray child and got reaped, defeating
    the restart it exists to perform."""
    from gateway.run_shutdown import GatewayShutdownMixin
    from gateway.status import _RESTART_WATCHER_ENV, _is_restart_safe_worker_or_descendant

    watcher_env = GatewayShutdownMixin._restart_watcher_env()
    assert watcher_env.get(_RESTART_WATCHER_ENV) == "1"

    class _FakeProc:
        def __init__(self, env):
            self._env = env

        def parents(self):
            return []

        def cmdline(self):
            return ["python", "-c", "..."]

        def environ(self):
            return self._env

    assert _is_restart_safe_worker_or_descendant(_FakeProc(watcher_env))
    assert not _is_restart_safe_worker_or_descendant(_FakeProc({}))
