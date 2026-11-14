"""A failed or timed-out child's background processes are named on its result before teardown.

A timed-out worker still owns its child, so the child's processes keep running until that worker exits: the
result must say so, rather than call them terminated and invite a relaunch beside the live original. The
child below spawns a real background process through the process registry under its own task id, and its
``close()`` stops what it owns, as an agent's close does.
"""

from __future__ import annotations

import shlex
import sys
import threading
import uuid
from types import SimpleNamespace

import pytest

from tools import delegate_tool, delegate_tool_child_run
from tools.process_registry import process_registry
from tools.process_registry_notifications import _process_accounting_lines


class _ProcessChild:
    def __init__(self, *, fail_after_spawn: bool = False, finish_at_once: bool = False) -> None:
        self.tool_progress_callback = None
        self._credential_pool = None
        self._delegate_saved_tool_names = []
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._subagent_id = None
        self.model = "test-model"
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.fail_after_spawn, self.finish_at_once = fail_after_spawn, finish_at_once
        self.spawned = threading.Event()
        self.interrupted = threading.Event()
        self.allow_finish = threading.Event()
        self.finished = threading.Event()
        self.closed = threading.Event()
        self.session = None
        self.task_id = None

    def run_conversation(self, *, task_id, **_kwargs):
        self.task_id = task_id
        # A sleeping interpreter whose argv carries a token, so a leftover is attributable to this test.
        command = f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(30)' e2p19-{uuid.uuid4().hex}"
        self.session = process_registry.spawn_local(command, task_id=task_id, owner_task_id=task_id)
        self.spawned.set()
        if self.fail_after_spawn:
            raise RuntimeError("child crashed after starting a watcher")
        if not self.finish_at_once:
            assert self.interrupted.wait(timeout=30)
            assert self.allow_finish.wait(timeout=30)
        self.finished.set()
        return {"final_response": "done", "completed": True, "api_calls": 1, "messages": []}

    def hard_interrupt(self, _reason=None):
        self.interrupted.set()

    def get_activity_summary(self):
        return {"api_call_count": 1}

    def close(self):
        if self.task_id:
            process_registry.kill_all(task_id=self.task_id, source="agent_close")
        self.closed.set()


@pytest.fixture()
def run_child(monkeypatch):
    """Run ``_run_single_child`` with a short inactivity cap; stop every process a child left, whatever happens."""
    import tools.process_registry as _pr

    monkeypatch.setattr(_pr, "_SYSTEMD_SCOPE_AVAILABLE", False)
    # Real pgids must not start the owner-death supervisor from a test.
    monkeypatch.setattr("hermes_cli.process_identity._update_supervised_pgids", lambda verb, pgids: None)
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 0.5)
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
    children = []
    from tools.daemon_pool import DaemonThreadPoolExecutor

    submit = DaemonThreadPoolExecutor.submit

    def submit_spawned(executor, *args, **kwargs):
        future = submit(executor, *args, **kwargs)
        assert children[-1].spawned.wait(timeout=30)  # the inactivity cap starts once the process is running
        return future

    monkeypatch.setattr(DaemonThreadPoolExecutor, "submit", submit_spawned)

    def run(child):
        children.append(child)
        parent = SimpleNamespace(session_id="parent-orphans", _current_task_id=None, _active_children=[child],
                                 _active_children_lock=threading.Lock())
        return delegate_tool._run_single_child(task_index=0, goal="watch CI", child=child, parent_agent=parent)

    yield run
    for child in children:
        child.allow_finish.set()
        if child.task_id:
            process_registry.kill_all(task_id=child.task_id, source="test")


def _alive(session) -> bool:
    return not session.exited and process_registry._is_host_pid_alive(session.pid)


def test_timed_out_child_reports_its_process_as_still_running_until_the_worker_exits(run_child):
    child = _ProcessChild()
    result = run_child(child)

    assert result["status"] == "timeout"
    sid = child.session.id
    assert [o["session_id"] for o in result.get("orphaned_processes", [])] == [sid], result
    assert result.get("orphaned_processes_close_deferred") is True
    notice = "\n".join(_process_accounting_lines(result))
    assert "still running" in notice and "abandoned worker exits" in notice and sid in notice
    assert "TERMINATED" not in notice and "Re-launch" not in notice
    assert _alive(child.session), "the notice must match a process that really is still running"

    child.allow_finish.set()
    assert child.closed.wait(timeout=10), "the deferred close runs once the worker exits"
    assert not _alive(child.session)


def test_processes_are_accounted_before_the_deferred_close(run_child, monkeypatch):
    """A worker that settles between the timeout and the deferred close makes add_done_callback run close() at
    once: the child's processes are killed in that call, so accounting after it would find nothing."""
    real_defer = delegate_tool_child_run._defer_close_after_timeout
    child = _ProcessChild()

    def settle_then_defer(child_, future):
        child.allow_finish.set()
        future.result(timeout=10)
        real_defer(child_, future)

    monkeypatch.setattr(delegate_tool_child_run, "_defer_close_after_timeout", settle_then_defer)
    result = run_child(child)

    assert result["status"] == "timeout" and child.closed.is_set()
    assert not _alive(child.session)
    assert [o["session_id"] for o in result.get("orphaned_processes", [])] == [child.session.id], result


@pytest.mark.parametrize("where", ["child raised", "parent tail raised"])
def test_failed_child_names_the_processes_teardown_terminates(run_child, monkeypatch, where):
    if where == "parent tail raised":
        def broken_entry(*_args, **_kwargs):
            raise RuntimeError("result entry could not be built")
        monkeypatch.setattr(delegate_tool, "_build_result_entry", broken_entry)
        child = _ProcessChild(finish_at_once=True)
    else:
        child = _ProcessChild(fail_after_spawn=True)

    result = run_child(child)

    assert result["status"] == "error"
    assert [o["session_id"] for o in result.get("orphaned_processes", [])] == [child.session.id], result
    assert "orphaned_processes_close_deferred" not in result
    notice = "\n".join(_process_accounting_lines(result))
    assert "TERMINATED" in notice and "still running" not in notice
    assert child.closed.is_set() and not _alive(child.session)
