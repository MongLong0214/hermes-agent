"""A rejected background batch retains synchronous cancellation ownership."""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
import uuid
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from agent.interrupt_control import InterruptControlMixin
from agent.turn_context import _bind_interrupt_scope
from tools import async_delegation
from tools.delegate_tool_dispatch import _Batch, _dispatch_background
from tools.interrupt import is_interrupted, set_interrupt
from tools.process_registry import process_registry


class _Parent(InterruptControlMixin):
    def __init__(self):
        self.session_id = "capacity-interrupt-parent"
        self._active_children = []
        self._active_children_lock = threading.Lock()
        self._execution_thread_id = None
        self._interrupt_requested = False
        self._hard_interrupt_requested = threading.Event()
        self.quiet_mode = True


class _ControlledChild(_Parent):
    """Replace model work while retaining real interrupt and worker-start semantics."""

    def __init__(self):
        super().__init__()
        self.session_id = "capacity-interrupt-child"
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._delegate_saved_tool_names = []
        self._credential_pool = None
        self._subagent_id = None
        self.tool_progress_callback = None
        self.model = "test-model"
        self.started = threading.Event()
        self.stop_received = threading.Event()
        self.unwinding = threading.Event()
        self.allow_finish = threading.Event()
        self.finished = threading.Event()
        self.closed = threading.Event()
        self.close_count = 0
        self.closed_while_running = False
        self.observed_interrupt = None

    def interrupt(self, message=None, **kwargs):
        accepted = super().interrupt(message, **kwargs)
        self.stop_received.set()
        return accepted

    def hard_interrupt(self, message=None, **kwargs):
        super().hard_interrupt(message, **kwargs)
        self.stop_received.set()

    def run_conversation(self, **_kwargs):
        # A stop can arrive before this thread exists. Use the real turn-start
        # binding so the pending agent interrupt must reach the tool thread too.
        _bind_interrupt_scope(self, lambda: SimpleNamespace(_set_interrupt=set_interrupt))
        self.started.set()
        try:
            assert self.stop_received.wait(30), "child never received cancellation"
            assert self._interrupt_requested
            assert is_interrupted(), "stop did not reach the child execution thread"
            self.observed_interrupt = (
                self._interrupt_message, self._hard_interrupt_requested.is_set(),
            )
            self.unwinding.set()
            assert self.allow_finish.wait(30), "test did not release child cleanup"
            return {
                "final_response": "", "completed": False, "interrupted": True,
                "api_calls": 0, "messages": [],
            }
        finally:
            self.clear_interrupt()
            self.finished.set()

    def get_activity_summary(self):
        return {"api_call_count": 0}

    def close(self):
        self.closed_while_running |= not self.finished.is_set()
        self.close_count += 1
        self.closed.set()


def _batch(parent, *children):
    tasks = [{"goal": f"wait until cancelled {i}"} for i in range(len(children))]
    parent._active_children.extend(children)
    return _Batch(
        task_list=tasks, children=[(i, tasks[i], child) for i, child in enumerate(children)], parent_agent=parent,
        creds={"model": children[0].model}, context=None, top_role="leaf", max_children=len(children),
        live_deleg_id=None, live_writers=[], live_paths=[], origin_wake_sid="",
        origin_ui_session_id="", origin_owner_transport=None,
        origin_owner_session_record=None, origin_session_history_delivery=False, overall_start=time.monotonic(),
    )


@pytest.fixture
def registry_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text(
        "delegation:\n  max_concurrent_children: 1\n  worktree_isolation: false\n",
        encoding="utf-8",
    )
    async_delegation._reset_for_tests()
    completion_queue = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", completion_queue)
    yield completion_queue
    # Test bodies release their gates and join their workers before registry teardown.
    if async_delegation._executor is not None:
        async_delegation._executor.shutdown(wait=True)
    async_delegation._reset_for_tests()


@pytest.mark.parametrize("rejection", ["capacity", "schedule_failure", "partial_schedule_failure"])
@pytest.mark.parametrize("stop_timing", ["running", "during_admission"])
@pytest.mark.parametrize("stop_kind", ["soft", "hard"])
def test_rejected_background_child_stops_with_parent(
    registry_state, monkeypatch, tmp_path, rejection, stop_timing, stop_kind,
):
    parent, child = _Parent(), _ControlledChild()
    background_child = pending_child = None
    if rejection == "partial_schedule_failure":
        # Three independent units: one accepted, one rejected, one not yet submitted.
        # The model-facing batch width is legal under the configured limit.
        (tmp_path / "config.yaml").write_text(
            "delegation:\n  max_concurrent_children: 3\n  worktree_isolation: false\n"
            "  independent_completions: true\n",
            encoding="utf-8",
        )
        background_child, pending_child = _ControlledChild(), _ControlledChild()
        background_child.session_id += "-background"
        pending_child.session_id += "-pending"
        batch = _batch(parent, background_child, child, pending_child)
    else:
        batch = _batch(parent, child)
    occupied = threading.Event()
    release_occupier = threading.Event()
    admission_started = threading.Event()
    continue_admission = threading.Event()
    outcome = Future()

    def occupy_slot():
        occupied.set()
        assert release_occupier.wait(30)
        return {"status": "completed", "summary": "slot released"}

    if rejection == "capacity":
        accepted = async_delegation.dispatch_async_delegation(
            goal="occupy the only slot", context=None, toolsets=None, role="leaf",
            model=child.model, session_key="other-session", runner=occupy_slot,
            max_async_children=1,
        )
        assert accepted["status"] == "dispatched"
        assert occupied.wait(5)
    elif rejection == "schedule_failure":
        class RejectingExecutor:
            def submit(self, *_args, **_kwargs):
                raise RuntimeError("executor shut down")

        monkeypatch.setattr(async_delegation, "_get_executor", lambda _n: RejectingExecutor())
    else:
        executor = async_delegation._get_executor(3)

        class PartiallyRejectingExecutor:
            submitted = 0

            def submit(self, *args, **kwargs):
                self.submitted += 1
                if self.submitted == 2:
                    raise RuntimeError("unit submission failed")
                return executor.submit(*args, **kwargs)

        partial_executor = PartiallyRejectingExecutor()
        monkeypatch.setattr(async_delegation, "_get_executor", lambda _n: partial_executor)

    dispatch = async_delegation.dispatch_async_delegation_batch
    admissions = 0
    accepted_ids = []

    def pause_admission(**kwargs):
        nonlocal admissions
        admissions += 1
        rejected_admission = 2 if background_child is not None else 1
        if admissions == rejected_admission:
            admission_started.set()
            assert continue_admission.wait(5)
        result = dispatch(**kwargs)
        if result.get("status") == "dispatched":
            accepted_ids.append(result["delegation_id"])
        return result

    monkeypatch.setattr(async_delegation, "dispatch_async_delegation_batch", pause_admission)

    def run_dispatch():
        try:
            outcome.set_result(json.loads(_dispatch_background(batch)))
        except BaseException as exc:
            outcome.set_exception(exc)

    worker = threading.Thread(target=run_dispatch, daemon=True)
    worker.start()
    try:
        assert admission_started.wait(5)
        if background_child is not None:
            assert background_child.started.wait(5)
        request_stop = parent.hard_interrupt if stop_kind == "hard" else parent.interrupt
        stop_message = "user correction or stop request"
        if stop_timing == "during_admission":
            request_stop(stop_message)
        continue_admission.set()
        assert child.started.wait(5)
        if stop_timing == "running":
            request_stop(stop_message)

        assert child.stop_received.wait(5), "fallback lost parent cancellation ownership"
        assert child.unwinding.wait(5)
        assert child.observed_interrupt == (stop_message, stop_kind == "hard")
        if background_child is not None:
            assert not background_child.stop_received.is_set()
            assert pending_child.stop_received.is_set(), "unsubmitted unit lost parent cancellation ownership"
            assert not pending_child.started.is_set()
        assert not outcome.done(), "dispatch returned while its child still owned resources"
        assert child.close_count == 0
        child.allow_finish.set()
        result = outcome.result(timeout=5)
        if background_child is None:
            assert "SYNCHRONOUSLY" in result["note"]
            assert result["results"][0]["status"] == "interrupted"
        else:
            assert result["status"] == "dispatched"
            assert result["inline_results"][0]["status"] == "interrupted"
            assert not background_child.stop_received.is_set()
            assert pending_child.unwinding.wait(5)
            assert pending_child.observed_interrupt == (stop_message, stop_kind == "hard")
        assert child.finished.is_set()
        assert child.close_count == 1
        assert not child.closed_while_running
        assert parent._active_children == []
    finally:
        continue_admission.set()
        if not child.finished.is_set():
            child.hard_interrupt("test teardown")
        child.allow_finish.set()
        worker.join(timeout=5)
        release_occupier.set()
        if background_child is not None:
            async_delegation.interrupt_for_session(parent_session_id=parent.session_id)
            for extra in (background_child, pending_child):
                extra.allow_finish.set()
                assert extra.closed.wait(5)
                assert extra.finished.is_set()
                assert extra.close_count == 1
                assert not extra.closed_while_running
            completed_ids = {registry_state.get(timeout=5)["delegation_id"] for _ in accepted_ids}
            assert completed_ids == set(accepted_ids)
        if rejection == "capacity":
            completion = registry_state.get(timeout=5)
            assert completion["delegation_id"] == accepted["delegation_id"]
        assert not worker.is_alive()


def test_accepted_background_child_keeps_registry_cancellation_ownership(registry_state):
    parent, child = _Parent(), _ControlledChild()
    try:
        result = json.loads(_dispatch_background(_batch(parent, child)))
        assert result["status"] == "dispatched"
        assert child.started.wait(5)
        parent.interrupt()
        # Parent interrupt fan-out is synchronous; observing it return establishes
        # that a detached child did not receive it without a timing-based wait.
        assert parent._interrupt_requested
        assert not child.stop_received.is_set()
        assert not child.finished.is_set()
        parent.hard_interrupt("stop the current parent turn")
        assert parent._hard_interrupt_requested.is_set()
        assert not child.stop_received.is_set()
        assert async_delegation.interrupt_for_session(parent_session_id=parent.session_id) == 1
        assert child.unwinding.wait(5)
        assert child.observed_interrupt[1] is True
        assert child.close_count == 0
        child.allow_finish.set()
        completion = registry_state.get(timeout=5)
        assert completion["delegation_id"] == result["delegation_id"]
        assert completion["results"][0]["status"] == "interrupted"
        assert child.finished.is_set()
        assert child.close_count == 1
        assert not child.closed_while_running
        assert parent._active_children == []
    finally:
        if not child.finished.is_set():
            child.hard_interrupt("test teardown")
        child.allow_finish.set()
        assert child.closed.wait(5)


class _QuickChild(_ControlledChild):
    """Finishes at once if it is ever run: a unit whose admission failed must never reach it."""

    def __init__(self):
        super().__init__()
        self.run_count = 0

    def run_conversation(self, **_kwargs):
        self.run_count += 1
        self.started.set()
        self.finished.set()
        return {"final_response": "ran", "completed": True, "api_calls": 0, "messages": []}


@pytest.fixture
def locked_ledger(monkeypatch):
    """The ledger on a shared-cache in-memory SQLite DB. ``lock()`` opens a write transaction on a second connection,
    so every ledger write -- the dispatch INSERT and the cleanup DELETE alike -- fails at once with the real
    ``database table is locked`` (SQLITE_LOCKED is not retried by busy_timeout); ``unlock()`` ends it."""
    uri = f"file:ledger-{uuid.uuid4().hex}?mode=memory&cache=shared"
    holder = sqlite3.connect(uri, uri=True, check_same_thread=False, isolation_level=None)
    async_delegation._initialize_schema(holder)
    monkeypatch.setattr(async_delegation, "_connect", lambda: sqlite3.connect(uri, uri=True, check_same_thread=False))

    def lock():
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("INSERT INTO async_delegations (delegation_id, origin_session, state, dispatched_at, updated_at, "
                       "delivery_state, delivery_attempts) VALUES ('lock-holder', 's', 'running', 0, 0, 'pending', 0)")

    def unlock():
        if holder.in_transaction:
            holder.execute("ROLLBACK")

    def rows():
        return [r[0] for r in holder.execute("SELECT delegation_id FROM async_delegations").fetchall()]

    yield SimpleNamespace(lock=lock, unlock=unlock, rows=rows)
    unlock()
    holder.close()


@pytest.mark.parametrize("failure", ["persistent", "transient"])
def test_unrecordable_background_dispatch_reports_the_db_failure_without_running(
    registry_state, locked_ledger, monkeypatch, failure,
):
    """PR65-R1: state.db cannot take the dispatch row. Persistent: the cleanup DELETE fails too and must not escape
    and strand the detached child. Transient: the cleanup succeeds, and the rejection must not be read as "pool at
    capacity" -- that ran the child synchronously inside the user's turn and hid the DB failure. Either way the tool
    reports the failure, the child never runs, it is disposed rather than left detached, and no slot is held."""
    from tools.delegate_tool_dispatch import _run_batch

    parent, child = _Parent(), _QuickChild()
    batch = _batch(parent, child)
    persist_errors = []
    real_persist = async_delegation._persist_dispatch

    def persist(record):
        try:
            return real_persist(record)
        except Exception as exc:
            persist_errors.append(exc)
            raise
        finally:
            if failure == "transient":
                locked_ledger.unlock()

    monkeypatch.setattr(async_delegation, "_persist_dispatch", persist)
    locked_ledger.lock()
    try:
        result = json.loads(_run_batch(batch, background=True))
    finally:
        locked_ledger.unlock()

    assert persist_errors and "locked" in str(persist_errors[0])
    assert result.get("status") == "error", result
    assert "locked" in result["error"]
    assert "capacity" not in json.dumps(result).lower()
    assert not child.started.is_set(), "a unit the ledger could not record ran inline anyway"
    assert async_delegation.active_count() == 0
    assert child not in parent._active_children and child.close_count == 1
    if failure == "transient":
        assert locked_ledger.rows() == [], "the failed dispatch left a ledger row behind"


def test_admission_that_raises_restores_parent_ownership(registry_state, monkeypatch):
    """PR65-R1: an exception out of async admission must not leave the already-detached child outside the parent's
    stop fan-out; the call reports the failure instead of raising past the restore or running the child."""
    from tools.delegate_tool_dispatch import _run_batch

    parent, child = _Parent(), _QuickChild()
    batch = _batch(parent, child)

    def raising_admission(**_kwargs):
        raise sqlite3.OperationalError("database table is locked")

    monkeypatch.setattr(async_delegation, "dispatch_async_delegation_batch", raising_admission)
    result = json.loads(_run_batch(batch, background=True))

    assert result["status"] == "error" and "locked" in result["error"], result
    assert not child.started.is_set()
    assert parent._active_children == [child], "admission raised and the detached child was never re-attached"


def test_schedule_failure_cleanup_survives_an_unavailable_ledger(registry_state, locked_ledger, monkeypatch):
    """PR65-R1 sibling site (the submit-failure cleanup): the executor refuses the unit and state.db is unavailable
    for the cleanup DELETE as well. That cleanup must not raise out of admission: the unit's slot is freed and the
    established scheduler-failure fallback still runs the child under the parent's ownership."""
    from tools.delegate_tool_dispatch import _run_batch

    parent, child = _Parent(), _QuickChild()

    class LockingRejectingExecutor:
        def submit(self, *_args, **_kwargs):
            locked_ledger.lock()
            raise RuntimeError("executor shut down")

    monkeypatch.setattr(async_delegation, "_get_executor", lambda _n: LockingRejectingExecutor())
    try:
        result = json.loads(_run_batch(_batch(parent, child), background=True))
    finally:
        locked_ledger.unlock()

    assert child.started.is_set() and result["results"][0]["status"] == "completed", result
    assert async_delegation.active_count() == 0
    assert child.close_count == 1 and parent._active_children == []


def test_submission_success_survives_a_stale_monitor_start_failure(registry_state, monkeypatch):
    """PR65-R3: ``executor.submit()`` succeeds -- the worker is really running -- before ``_ensure_stale_monitor``
    is ever called (tools/async_delegation.py:795-796). A failure in THAT call must not be reported as "could not
    be dispatched": the work already started, and a caller that retried on that message would dispatch a SECOND
    copy of the same work. Mirrors the before/after-the-real-action split already used for the persist/schedule-
    failure sites above: classify by where the exception originates, never a blanket catch around the dispatch."""
    from tools.delegate_tool_dispatch import _run_batch

    parent, child = _Parent(), _QuickChild()
    batch = _batch(parent, child)

    def failing_monitor():
        raise RuntimeError("could not start thread")

    monkeypatch.setattr(async_delegation, "_ensure_stale_monitor", failing_monitor)
    result = json.loads(_run_batch(batch, background=True))

    assert result["status"] == "dispatched", result
    assert "could not be dispatched" not in json.dumps(result).lower(), result
    completion = registry_state.get(timeout=5)
    assert completion["delegation_id"] == result["delegation_id"]
    assert completion["results"][0]["status"] == "completed"
    assert child.started.is_set() and child.finished.is_set()
    assert child.close_count == 1 and parent._active_children == []


def test_partial_batch_running_unit_is_not_reported_not_started(registry_state, monkeypatch, tmp_path):
    """PR65-R3 partial-batch sibling: a LATER unit of an independent-completions call whose worker already started
    must not land in ``not_started`` just because ``_ensure_stale_monitor`` then failed for it -- ``not_started``
    promises a task that never ran and invites the caller to resend it while the real one is still in flight."""
    (tmp_path / "config.yaml").write_text(
        "delegation:\n  max_concurrent_children: 3\n  worktree_isolation: false\n"
        "  independent_completions: true\n",
        encoding="utf-8",
    )
    from tools.delegate_tool_dispatch import _run_batch

    parent = _Parent()
    first, second = _QuickChild(), _QuickChild()
    first.session_id += "-first"
    second.session_id += "-second"
    batch = _batch(parent, first, second)

    real_monitor = async_delegation._ensure_stale_monitor
    calls: list = []

    def flaky_monitor():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("could not start thread")
        return real_monitor()

    monkeypatch.setattr(async_delegation, "_ensure_stale_monitor", flaky_monitor)
    result = json.loads(_run_batch(batch, background=True))

    assert result["status"] == "dispatched", result
    assert "not_started" not in result, result
    assert len(result.get("units", [])) == 2, result
    # Wait for both units' completions before reading ``started``: executor.submit() returning does not mean the
    # pool has run the worker yet, so checking ``started`` first would race the thread that sets it.
    completions = {registry_state.get(timeout=5)["delegation_id"] for _ in range(2)}
    assert completions == {u["delegation_id"] for u in result["units"]}
    assert first.started.is_set() and second.started.is_set()


def test_submit_exception_after_the_work_item_already_ran_is_not_executed_again(registry_state, monkeypatch):
    """PR65-R5 (ROUND1-ESCAPE): ThreadPoolExecutor.submit() enqueues the work item BEFORE it tries to
    start a worker thread, so a late exception from that attempt does not mean the task never ran --
    an existing or newly-created thread can still pick the queued item up and execute it for real.
    Treating that exception as a clean "nothing was submitted" rejection let the caller run the SAME
    task a second time inline. Reproduced here by letting the real submit() genuinely enqueue and run
    the work, then raising afterward -- exactly the shape of a late thread-start failure."""
    from tools.delegate_tool_dispatch import _run_batch

    parent, child = _Parent(), _QuickChild()
    batch = _batch(parent, child)

    executor = async_delegation._get_executor(2)
    real_submit = executor.submit
    raised = threading.Event()

    def failing_submit(fn, *a, **kw):
        future = real_submit(fn, *a, **kw)
        future.result(timeout=5)  # the real work item has already run by the time submit() "fails"
        raised.set()
        raise RuntimeError("injected: thread start failed after the work item was already queued")

    monkeypatch.setattr(executor, "submit", failing_submit)
    try:
        result = json.loads(_run_batch(batch, background=True))
    finally:
        monkeypatch.setattr(executor, "submit", real_submit)

    assert raised.is_set(), "the test did not exercise the late-submit-failure path"
    assert child.finished.wait(5), "the real worker thread should have run the task once"
    assert result.get("status") == "error", result
    assert "inline_results" not in result, "the already-run task must not also run inline"
    assert child.run_count == 1, "the task ran more than once"
