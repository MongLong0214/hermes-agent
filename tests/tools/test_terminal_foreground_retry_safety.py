"""``_run_foreground`` must not replay a command whose shell already started.

Before this fix, the retry loop caught every non-timeout exception from ``env.execute()``
and retried unconditionally, even though the shell can already be running (or have already
run) by the time ``execute()`` fails — collecting output, updating cwd state, and so on.
Retrying then risks repeating a real side effect (``git push``, a sent mail). Only a failure
proven to precede process creation (raised before ``tools/environments/base.py``'s
``_spawn_and_wait`` records the spawned handle) may be retried; anything after must come back
as an "outcome unknown" error instead of being replayed.
"""

import time

import pytest

import tools.terminal_tool as terminal_tool
from tools.environments.base import BaseEnvironment


class _SpawnThenFailEnv(BaseEnvironment):
    """Minimal concrete backend: ``_run_bash`` records a side effect (the shell actually ran),
    then ``_wait_for_process`` raises — simulating a failure discovered only after the command
    was already spawned."""

    def __init__(self, side_effects: list):
        super().__init__(cwd="/tmp", timeout=5)
        self._side_effects = side_effects

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        self._side_effects.append(cmd_string)
        return object()  # opaque process handle; never polled before the failure below

    def _wait_for_process(self, proc, timeout=120, **kwargs):
        raise OSError("disk full collecting output")

    def cleanup(self):
        pass


def _plan():
    from tools.terminal_tool import _ExecPlan
    return _ExecPlan(
        config={}, env_type="fake", effective_task_id="t-postspawn", image="",
        cwd="/tmp", host_cwd=None, effective_timeout=5,
    )


def test_post_spawn_failure_is_not_retried_and_runs_exactly_once():
    side_effects: list = []
    env = _SpawnThenFailEnv(side_effects)

    start = time.monotonic()
    result = terminal_tool._run_foreground(
        "echo hi", env, _plan(),
        task_id="t-postspawn", session_id=None, session_key="postspawn-test",
        workdir=None, approval_note=None, clear_interrupt=False,
    )
    elapsed = time.monotonic() - start

    assert len(side_effects) == 1, "the shell must not be re-spawned after an ambiguous post-spawn failure"
    assert "echo hi" in side_effects[0]
    assert elapsed < 1.0, f"no retry backoff (2/4/8s) should run for a post-spawn failure, took {elapsed:.2f}s"

    import json
    body = json.loads(result)
    assert body["error"], body
    assert "outcome unknown" in body["error"].lower(), body
    assert body.get("status") == "ambiguous", body


def test_run_bash_raising_after_popen_is_not_retried(monkeypatch):
    """A failure raised by ``_run_bash`` itself — not only one from ``_wait_for_process``
    after ``_run_bash`` already returned — must also be classified as post-spawn when the
    underlying process was already created. Real ``LocalEnvironment``, real subprocess
    creation (counted at the actual ``subprocess.Popen`` boundary, per PR64-01's "Popen
    succeeding = process exists"), supplied stdin (mirroring the ``sudo_stdin`` merge every
    foreground command goes through), and an injected stdin writer-thread-start failure:
    before the fix this replayed the command (multiple real ``Popen`` calls instead of one)."""
    import contextlib
    import subprocess as subprocess_mod

    from tools.environments.local import LocalEnvironment
    import tools.environments.local as local_mod

    monkeypatch.setattr(terminal_tool.time, "sleep", lambda _seconds: None)  # skip 2/4/8s backoff

    env = LocalEnvironment()  # real init_session() bootstrap happens before any patching below

    # _prepare_command normally returns (command, sudo_stdin) — non-None sudo_stdin is how
    # an ordinary foreground command (one using sudo with a cached password) ends up with
    # non-None stdin_data reaching _run_bash. Stub it the same way rather than wiring the
    # real sudo-password plumbing, which is incidental to what's under test here.
    monkeypatch.setattr(LocalEnvironment, "_prepare_command", lambda self, command: (command, "dummy\n"))

    def _boom_pipe_stdin(proc, data):
        # Popen has already succeeded by the time _run_bash calls this — this
        # simulates the writer thread failing to start (RuntimeError: can't
        # start new thread), which happens AFTER the process exists.
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(local_mod, "_pipe_stdin", _boom_pipe_stdin)

    spawned: list = []
    real_popen = subprocess_mod.Popen

    def _counting_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(local_mod.subprocess, "Popen", _counting_popen)

    try:
        result = terminal_tool._run_foreground(
            "true", env, _plan(),
            task_id="t-runbash-postspawn", session_id=None, session_key="runbash-postspawn-test",
            workdir=None, approval_note=None, clear_interrupt=False,
        )
    finally:
        for proc in spawned:
            with contextlib.suppress(Exception):
                proc.kill()
                proc.wait(timeout=2)
        env.cleanup()

    assert len(spawned) == 1, (
        f"the shell must not be re-spawned after a post-Popen failure; "
        f"spawned {len(spawned)} real processes instead of 1"
    )

    import json
    body = json.loads(result)
    assert body.get("status") == "ambiguous", body

    import json
    body = json.loads(result)
    assert body.get("status") == "ambiguous", body


def test_pre_spawn_failure_still_retries(monkeypatch):
    """Control: a failure BEFORE the shell is spawned (the historical case this loop exists
    for) keeps retrying up to max_retries, unlike the post-spawn case above."""
    monkeypatch.setattr(terminal_tool.time, "sleep", lambda _seconds: None)  # skip the 2/4/8s backoff

    class _FailBeforeSpawnEnv(BaseEnvironment):
        def __init__(self):
            super().__init__(cwd="/tmp", timeout=5)
            self.attempts = 0

        def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
            self.attempts += 1
            raise OSError("connection refused")

        def cleanup(self):
            pass

    env = _FailBeforeSpawnEnv()
    result = terminal_tool._run_foreground(
        "echo hi", env, _plan(),
        task_id="t-prespawn", session_id=None, session_key="prespawn-test",
        workdir=None, approval_note=None, clear_interrupt=False,
    )
    assert env.attempts == 4  # initial attempt + 3 retries (max_retries=3)
    import json
    body = json.loads(result)
    assert body.get("status") != "ambiguous", body
