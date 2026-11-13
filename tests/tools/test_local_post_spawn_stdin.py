"""The local backend's stdin pipe failure after ``Popen`` is a post-spawn failure.

``LocalEnvironment._run_bash`` writes stdin (a piped sudo password) only after ``Popen``
succeeded, so a failure there comes from a shell that already exists and may already be
running. It must surface as ``PostSpawnExecutionError`` (never retried by the terminal tool),
and since nothing will wait on that process, ``_run_bash`` stops its group itself instead of
leaving it running and still registered with the owner-death supervisor.
"""

import contextlib
import json
import os
import signal
import subprocess
import threading
import time
import types

import pytest

import gateway.status as gateway_status
import hermes_cli.process_identity as process_identity
import tools.environments.local as local_mod
import tools.terminal_tool as terminal_tool
from tools.environments.base import PostSpawnExecutionError
from tools.environments.local import LocalEnvironment

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX process-group teardown")

MARK = "hermes-e2p4-stdin-marker"


@pytest.fixture
def supervised(monkeypatch):
    """Owner-death coverage ledger in place of the real supervisor process."""
    ledger = {"current": set(), "registered": []}

    def _update(verb, pgids):
        wanted = {int(p) for p in pgids}
        if verb == "register":
            ledger["registered"].extend(wanted)
            ledger["current"].update(wanted)
        else:
            ledger["current"].difference_update(wanted)

    monkeypatch.setattr(process_identity, "_update_supervised_pgids", _update)
    return ledger


@pytest.fixture
def spawned(monkeypatch):
    """Record the real ``Popen`` objects ``local`` creates for MARKed commands (other spawns,
    e.g. the session snapshot, pass through unrecorded); kill any still running at teardown."""
    procs: list = []
    real_popen = subprocess.Popen
    proxy = types.ModuleType("subprocess_proxy")
    proxy.__getattr__ = lambda name: getattr(subprocess, name)

    def _popen(args, *a, **kw):
        if getattr(proxy, "fail_before_spawn", None) is not None and MARK in " ".join(map(str, args)):
            raise proxy.fail_before_spawn
        proc = real_popen(args, *a, **kw)
        if MARK in " ".join(map(str, args)):
            procs.append(proc)
        return proc

    proxy.Popen = _popen
    proxy.fail_before_spawn = None
    monkeypatch.setattr(local_mod, "subprocess", proxy)
    yield types.SimpleNamespace(procs=procs, proxy=proxy)
    for proc in procs:
        if proc.poll() is None:  # still ours: never signal a reaped (recyclable) pid
            with contextlib.suppress(OSError):
                os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
        for pipe in (proc.stdin, proc.stdout):
            if pipe is not None:
                with contextlib.suppress(Exception):
                    pipe.close()


@pytest.fixture
def env(tmp_path, supervised, spawned):
    environment = LocalEnvironment(cwd=str(tmp_path), timeout=10)
    yield environment
    environment.cleanup()


@pytest.fixture
def broken_stdin(monkeypatch):
    def _broken(proc, data):
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(local_mod, "_pipe_stdin", _broken)


@pytest.fixture
def backoffs(monkeypatch):
    """Record the retry loop's backoff sleeps (test thread only) instead of sleeping."""
    calls: list = []
    real_sleep = time.sleep
    test_thread = threading.get_ident()

    def _sleep(seconds):
        if threading.get_ident() == test_thread:
            calls.append(seconds)
            return
        real_sleep(seconds)

    monkeypatch.setattr(terminal_tool.time, "sleep", _sleep)
    return calls


def _group_alive(pgid) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run_foreground(environment, command):
    plan = terminal_tool._ExecPlan(
        config={}, env_type="local", effective_task_id="t-e2p4", image="",
        cwd=environment.cwd, host_cwd=None, effective_timeout=10,
    )
    return json.loads(terminal_tool._run_foreground(
        command, environment, plan, task_id="t-e2p4", session_id=None,
        session_key="e2p4-local-stdin", workdir=None, approval_note=None, clear_interrupt=False,
    ))


def _with_sudo_stdin(environment):
    """Give every command a piped stdin, as a sudo password rewrite does."""
    environment._prepare_command = lambda command: (command, "pw\n")
    return environment


# 1. _pipe_stdin raises BrokenPipeError after Popen -> PostSpawnExecutionError
def test_stdin_failure_after_popen_is_post_spawn(env, spawned, broken_stdin):
    with pytest.raises(PostSpawnExecutionError) as caught:
        env._run_bash(f"sleep 30 # {MARK}", stdin_data="pw\n")

    assert isinstance(caught.value, RuntimeError)
    assert isinstance(caught.value.__cause__, BrokenPipeError)
    assert len(spawned.procs) == 1


def test_stdin_failure_stops_the_spawned_group_and_releases_it(env, spawned, supervised, broken_stdin):
    with pytest.raises(PostSpawnExecutionError):
        env._run_bash(f"sleep 30 # {MARK}", stdin_data="pw\n")

    (proc,) = spawned.procs
    assert proc.pid in supervised["registered"]  # it was covered once spawned
    assert proc.poll() is not None, "the spawned shell was left running"
    assert not _group_alive(proc.pid)
    assert proc.pid not in supervised["current"]  # no stale (recyclable) pgid left covered
    assert proc.stdout.closed


def test_terminal_tool_runs_a_stdin_failure_once(env, spawned, broken_stdin, backoffs):
    body = _run_foreground(_with_sudo_stdin(env), f"sleep 30 # {MARK}")

    assert len(spawned.procs) == 1
    assert backoffs == []
    assert body["status"] == "ambiguous", body
    assert "outcome unknown" in body["error"]
    assert spawned.procs[0].poll() is not None


# 2. a Popen failure (before spawn) is unchanged and retryable
def test_popen_failure_is_not_post_spawn(env, spawned, supervised):
    before = list(supervised["registered"])
    spawned.proxy.fail_before_spawn = FileNotFoundError(2, "No such file or directory", "bash")

    with pytest.raises(FileNotFoundError) as caught:
        env._run_bash(f"true # {MARK}", stdin_data="pw\n")

    assert not isinstance(caught.value, PostSpawnExecutionError)
    assert supervised["registered"] == before  # nothing spawned, nothing covered


def test_terminal_tool_still_retries_a_popen_failure(env, spawned, backoffs):
    spawned.proxy.fail_before_spawn = FileNotFoundError(2, "No such file or directory", "bash")
    executes = []
    real_execute = env.execute
    env.execute = lambda *a, **kw: (executes.append(a), real_execute(*a, **kw))[1]

    body = _run_foreground(_with_sudo_stdin(env), f"true # {MARK}")

    assert len(executes) == 4
    assert backoffs == [2, 4, 8]
    assert "status" not in body
    assert body["error"].startswith("Command execution failed: FileNotFoundError")


# 3. the pgid start-time check still decides which group the stop signals
@pytest.fixture
def group_signals(monkeypatch):
    calls: list = []
    real_killpg = os.killpg

    def _killpg(pgid, sig):
        if sig != 0:
            calls.append((pgid, sig))
        return real_killpg(pgid, sig)

    monkeypatch.setattr(os, "killpg", _killpg)
    return calls


def test_stdin_failure_stop_signals_the_spawned_group(env, spawned, broken_stdin, group_signals):
    with pytest.raises(PostSpawnExecutionError):
        env._run_bash(f"sleep 30 # {MARK}", stdin_data="pw\n")

    (proc,) = spawned.procs
    assert proc._hermes_pgid == proc.pid and proc._hermes_pgid_start is not None
    assert (proc.pid, signal.SIGTERM) in group_signals
    assert {pgid for pgid, _ in group_signals} == {proc.pid}
    assert os.getpgrp() not in {pgid for pgid, _ in group_signals}


def test_stdin_failure_stop_never_signals_a_recycled_leader(
        env, spawned, supervised, broken_stdin, group_signals, monkeypatch):
    """The leader's start time no longer matches the spawn-time baseline: a recycled PGID
    may belong to a stranger, so the stop must not signal that group (#43044)."""
    real_start_time = gateway_status.get_process_start_time
    reads: dict = {}

    def _start_time(pid):
        value = real_start_time(pid)
        reads[pid] = reads.get(pid, 0) + 1
        if reads[pid] > 1 and value is not None and any(p.pid == pid for p in spawned.procs):
            return value + 10**9
        return value

    monkeypatch.setattr(gateway_status, "get_process_start_time", _start_time)

    with pytest.raises(PostSpawnExecutionError):
        env._run_bash(f"sleep 30 # {MARK}", stdin_data="pw\n")

    (proc,) = spawned.procs
    assert reads.get(proc.pid, 0) >= 2  # baseline at spawn, then the check before signalling
    assert group_signals == []
    assert proc.pid in supervised["current"]  # a group we could not stop stays covered
