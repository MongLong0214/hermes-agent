"""The gateway pid scan and the orphan reaper's exemptions follow a process's real home.

Regression: a web server started on a temp HERMES_HOME runs the orphan reaper at startup. Its
current-profile scan claimed every ``gateway run`` with no profile flag and no ``HERMES_HOME=`` on
argv, so it SIGTERMed the operator's launchd gateway, whose home arrives through the plist
environment and never appears on argv. The reaper's service exemption covered the launchd job PID,
which is the stderr-timestamp wrapper, but not the gateway that wrapper runs.

Both tests spawn only their own children and read the process table. Nothing is signalled except
those children, and each child exits on its own once its parent is gone.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import hermes_cli.gateway as gateway

# Named ``hermes`` so ``python <dir>/hermes gateway run`` satisfies the canonical gateway matcher.
_GATEWAY_STUB = """
import os, time
open(os.environ["STUB_READY"], "w").close()
parent, deadline = os.getppid(), time.monotonic() + 120
while os.getppid() == parent and time.monotonic() < deadline:
    time.sleep(0.1)
"""

# Mirrors ``python -m hermes_cli.stderr_timestamp ... -- <gateway argv>``: the service manager's job
# PID is this wrapper, and the gateway is its child. Closing stdin makes it stop its child and exit.
_SERVICE_WRAPPER = """
import subprocess, sys
child = subprocess.Popen(sys.argv[sys.argv.index("--") + 1:])
print(child.pid, flush=True)
try:
    sys.stdin.read()
finally:
    child.terminate()
    child.wait(10)
"""


def _gateway_argv(tmp_path: Path) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / "hermes"
    script.write_text(_GATEWAY_STUB, encoding="utf-8")
    return [sys.executable, str(script), "gateway", "run"]


def _wait_for(path: Path, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + 15.0
    while not path.exists():
        if proc.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError("gateway stub never came up")
        time.sleep(0.05)


def _reap(proc: subprocess.Popen) -> None:
    if proc.stdin is not None:
        proc.stdin.close()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.mark.spawns_gateway_lookalike
def test_scan_claims_a_bare_gateway_only_for_the_home_its_environment_names(tmp_path, monkeypatch):
    home_a, home_b = tmp_path / "home-a", tmp_path / "home-b"
    home_a.mkdir()
    home_b.mkdir()
    ready = tmp_path / "gateway.ready"
    monkeypatch.setattr(gateway, "_get_service_pids", lambda all_profiles=False: set())
    proc = subprocess.Popen(
        _gateway_argv(tmp_path),
        env={**os.environ, "HERMES_HOME": str(home_a), "STUB_READY": str(ready)},
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for(ready, proc)
        monkeypatch.setenv("HERMES_HOME", str(home_b))
        assert proc.pid not in gateway.find_gateway_pids()
        monkeypatch.setenv("HERMES_HOME", str(home_a))
        assert proc.pid in gateway.find_gateway_pids()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.mark.spawns_gateway_lookalike
def test_reaper_exempts_the_gateway_a_service_wrapper_runs(tmp_path, monkeypatch):
    ready = tmp_path / "gateway.ready"
    wrapper_script = tmp_path / "service_wrapper.py"
    wrapper_script.write_text(_SERVICE_WRAPPER, encoding="utf-8")
    wrapper = subprocess.Popen(
        [sys.executable, str(wrapper_script), "--", *_gateway_argv(tmp_path)],
        env={**os.environ, "STUB_READY": str(ready)},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    try:
        gateway_pid = int(wrapper.stdout.readline())
        _wait_for(ready, wrapper)
        monkeypatch.setattr(gateway, "_get_service_pids", lambda all_profiles=False: {wrapper.pid})
        assert gateway_pid in gateway._reaper_exclusion_pids(None)
    finally:
        _reap(wrapper)
