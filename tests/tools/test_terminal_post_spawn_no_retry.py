"""A terminal command that already spawned is never retried.

``_run_foreground`` retries a failing ``env.execute()`` up to 3 times with a 2/4/8 s backoff.
That is only safe for a failure that precedes process creation: once the shell exists it may
already have produced a side effect (``git push``, a sent mail), so a failure discovered after
spawn comes back as ``status: "ambiguous"`` and is never replayed.
"""

import json
import subprocess
import threading
import time
import types

import pytest

import tools.environments.base_output as base_output
import tools.terminal_tool as terminal_tool
from tools.environments.base import BaseEnvironment, EnvironmentConnectionError, PostSpawnExecutionError


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


def _plan():
    return terminal_tool._ExecPlan(
        config={}, env_type="fake", effective_task_id="t-postspawn", image="",
        cwd="/tmp", host_cwd=None, effective_timeout=5,
    )


def _run(env, command="echo hi"):
    return json.loads(terminal_tool._run_foreground(
        command, env, _plan(), task_id="t-postspawn", session_id=None,
        session_key="postspawn-no-retry", workdir=None, approval_note=None, clear_interrupt=False,
    ))


class _Env(BaseEnvironment):
    """``_run_bash`` records each spawn; ``spawn_error``/``wait_error`` fail before/after it."""

    def __init__(self, *, spawn_error=None, wait_error=None):
        super().__init__(cwd="/tmp", timeout=5)
        self.spawns: list = []
        self.spawn_attempts = 0
        self.execute_calls = 0
        self._spawn_error, self._wait_error = spawn_error, wait_error

    def execute(self, *args, **kwargs):
        self.execute_calls += 1
        return super().execute(*args, **kwargs)

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        self.spawn_attempts += 1
        if self._spawn_error is not None:
            raise self._spawn_error
        self.spawns.append(cmd_string)
        return object()

    def _wait_for_process(self, proc, timeout=120, **kwargs):
        raise self._wait_error

    def cleanup(self):
        pass


class _FakePopen:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.stdin = self.stdout = None


@pytest.fixture
def popen_calls(monkeypatch):
    """Count ``subprocess.Popen`` in ``base_output`` only; ``_pipe_stdin`` fails after it."""
    calls: list = []

    def _popen(*args, **kwargs):
        calls.append(args)
        return _FakePopen(*args, **kwargs)

    monkeypatch.setattr(base_output, "subprocess", types.SimpleNamespace(
        Popen=_popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT, DEVNULL=subprocess.DEVNULL))

    def _boom(proc, data):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(base_output, "_pipe_stdin", _boom)
    return calls


# 1. spawn, then _wait_for_process raises -> one execute, ambiguous, no backoff
def test_wait_failure_after_spawn_is_ambiguous_and_runs_once(backoffs):
    env = _Env(wait_error=OSError("disk full collecting output"))

    body = _run(env)

    assert env.execute_calls == 1
    assert len(env.spawns) == 1 and "echo hi" in env.spawns[0]
    assert backoffs == []
    assert body["status"] == "ambiguous", body
    assert "outcome unknown" in body["error"] and "disk full collecting output" in body["error"]


def test_execute_raises_post_spawn_error_with_the_original_cause():
    original = OSError("disk full collecting output")
    env = _Env(wait_error=original)

    with pytest.raises(PostSpawnExecutionError) as caught:
        env.execute("echo hi", timeout=5)

    assert caught.value.__cause__ is original


# 2. a stdin pipe failure after Popen in _popen_bash is post-spawn
def test_popen_bash_stdin_failure_after_popen_is_post_spawn(popen_calls):
    with pytest.raises(PostSpawnExecutionError) as caught:
        base_output._popen_bash(["bash", "-c", "true"], "payload\n")

    assert isinstance(caught.value, RuntimeError)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "can't start new thread" in str(caught.value)
    assert len(popen_calls) == 1


def test_popen_bash_without_stdin_does_not_wrap(popen_calls):
    proc = base_output._popen_bash(["bash", "-c", "true"])

    assert isinstance(proc, _FakePopen) and len(popen_calls) == 1


def test_popen_bash_stdin_failure_runs_the_command_once(popen_calls, backoffs):
    class _PopenEnv(_Env):
        def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
            return base_output._popen_bash(["bash", "-c", cmd_string], "payload\n")

    body = _run(_PopenEnv())

    assert len(popen_calls) == 1
    assert backoffs == []
    assert body["status"] == "ambiguous", body


# 3. a pre-spawn EnvironmentConnectionError is still retried 3 times
def test_pre_spawn_connection_error_still_retries(backoffs):
    env = _Env(spawn_error=EnvironmentConnectionError("ssh: host unreachable"))

    body = _run(env)

    assert env.spawn_attempts == 4 and env.spawns == []
    assert backoffs == [2, 4, 8]
    assert "status" not in body
    assert body["error"].startswith("Command execution failed: EnvironmentConnectionError")


# 4. an exception whose text contains "timeout" is exit 124 with no retry
def test_pre_spawn_timeout_text_returns_124_without_retry(backoffs):
    env = _Env(spawn_error=OSError("ssh: connect timeout"))

    body = _run(env)

    assert env.spawn_attempts == 1
    assert backoffs == []
    assert body["exit_code"] == 124 and "status" not in body


# 5. the ambiguous error text is redacted
def test_ambiguous_error_text_is_redacted(backoffs):
    secret = "Zq8vNotReal3xK"
    env = _Env(wait_error=OSError(f"push to https://deploy:{secret}@git.example.com/o/r.git failed"))

    raw = terminal_tool._run_foreground(
        "git push", env, _plan(), task_id="t-postspawn", session_id=None,
        session_key="postspawn-no-retry", workdir=None, approval_note=None, clear_interrupt=False,
    )

    assert secret not in raw
    body = json.loads(raw)
    assert body["status"] == "ambiguous"
    assert "https://deploy:***@git.example.com" in body["error"]
