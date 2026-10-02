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
