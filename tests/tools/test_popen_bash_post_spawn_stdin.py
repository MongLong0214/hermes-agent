"""``_popen_bash`` stops the process it spawned when piping stdin fails after ``Popen``.

The ssh and docker backends spawn their client through ``_popen_bash``. A stdin failure after
``Popen`` surfaces as ``PostSpawnExecutionError`` (never retried), and since nothing will wait
on that client, ``_popen_bash`` stops it itself with the local backend's tree kill: the client
and what it started die, its pipes close and it is reaped. The client shares the caller's
process group, so the stop must never signal that group.
"""

import contextlib
import os
import signal
import subprocess
import time
import types

import psutil
import pytest

from tools.environments import base_output
from tools.environments.base import PostSpawnExecutionError

pytestmark = [
    pytest.mark.skipif(os.name == "nt", reason="POSIX process teardown"),
    # A child whose shell died first is reparented to init: the stop (and a red run's teardown)
    # must still kill it by PID. ``group_signals`` keeps every signal off our own group.
    pytest.mark.live_system_guard_bypass,
]

MARK = "hermes-e2p4b-stdin-marker"
# The shell backgrounds a child (renamed so it carries MARK too) and waits on it.
COMMAND = ["bash", "-c", f"exec -a {MARK}-child sleep 30 & wait # {MARK}"]


def _kill(target) -> None:
    with contextlib.suppress(psutil.Error):
        target.kill()


@pytest.fixture
def spawned(monkeypatch):
    """Record the real ``Popen`` objects ``_popen_bash`` creates and the children each one started
    before stdin failed; kill whatever is still running at teardown, by PID (never the group:
    these share the test runner's own group)."""
    record = types.SimpleNamespace(procs=[], pgids=[], children=[])
    real_popen = subprocess.Popen

    def _popen(args, *a, **kw):
        proc = real_popen(args, *a, **kw)
        record.procs.append(proc)
        record.pgids.append(os.getpgid(proc.pid))
        return proc

    monkeypatch.setattr(base_output, "subprocess", types.SimpleNamespace(
        Popen=_popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT, DEVNULL=subprocess.DEVNULL))
    yield record
    for child in record.children:
        _kill(child)
    for proc in record.procs:
        if proc.poll() is None:  # still ours: never signal a reaped (recyclable) pid
            with contextlib.suppress(psutil.Error):
                for child in psutil.Process(proc.pid).children(recursive=True):
                    _kill(child)
            with contextlib.suppress(OSError):
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
        for pipe in (proc.stdin, proc.stdout):
            if pipe is not None:
                with contextlib.suppress(Exception):
                    pipe.close()
    psutil.wait_procs(record.children, timeout=5)


@pytest.fixture
def broken_stdin(monkeypatch, spawned):
    """``_pipe_stdin`` fails once the shell has started its child (snapshotted for the asserts)."""
    def _broken(proc, data):
        deadline = time.monotonic() + 5
        while not (children := psutil.Process(proc.pid).children(recursive=True)):
            assert time.monotonic() < deadline, "the shell never started its child"
            time.sleep(0.02)
        spawned.children.extend(children)
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(base_output, "_pipe_stdin", _broken)


@pytest.fixture(autouse=True)
def group_signals(monkeypatch):
    """Record every group signal; one aimed at the test runner's own group is never delivered."""
    calls: list = []
    real_killpg = os.killpg

    def _killpg(pgid, sig):
        if sig != 0:
            calls.append((pgid, sig))
            if pgid == os.getpgrp():
                return None
        return real_killpg(pgid, sig)

    monkeypatch.setattr(os, "killpg", _killpg)
    return calls


def _group_alive(pgid) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_stdin_failure_is_post_spawn_and_stops_the_spawned_process(spawned, broken_stdin):
    with pytest.raises(PostSpawnExecutionError) as caught:
        base_output._popen_bash(COMMAND, "payload\n")

    assert isinstance(caught.value.__cause__, BrokenPipeError)
    (proc,) = spawned.procs
    assert proc.returncode is not None, "the spawned shell was left running or unreaped"
    assert proc.stdout.closed and proc.stdin.closed


def test_stdin_failure_stops_what_the_shell_started(spawned, broken_stdin):
    with pytest.raises(PostSpawnExecutionError):
        base_output._popen_bash(COMMAND, "payload\n")

    assert spawned.children
    _gone, alive = psutil.wait_procs(spawned.children, timeout=3)
    assert alive == [], f"left running: {alive}"


def test_stdin_failure_stop_never_signals_the_callers_group(spawned, broken_stdin, group_signals):
    with pytest.raises(PostSpawnExecutionError):
        base_output._popen_bash(COMMAND, "payload\n")

    (proc,) = spawned.procs
    assert spawned.pgids == [os.getpgrp()]  # the client was spawned into our group
    assert os.getpgrp() not in {pgid for pgid, _ in group_signals}
    assert proc.returncode is not None


def test_stdin_failure_stops_a_spawned_group_leader_by_its_group(spawned, broken_stdin, group_signals):
    """A client spawned into its own session is stopped through its group, and only that group."""
    with pytest.raises(PostSpawnExecutionError):
        base_output._popen_bash(COMMAND, "payload\n", start_new_session=True)

    (proc,) = spawned.procs
    assert (proc.pid, signal.SIGTERM) in group_signals
    assert {pgid for pgid, _ in group_signals} == {proc.pid}
    assert not _group_alive(proc.pid)
    assert proc.returncode is not None
