"""A background delegation unit that was not accepted is never also run inline.

A clean rejection (the pool is full, or the item provably never reached the pool's queue) may still
run synchronously. A rejection whose submission is unresolved, or that state.db could not record, is
reported to the caller instead: running it inline could duplicate work a worker already started, or
hide the database failure inside the user's turn.
"""

from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from hermes_cli.backend_retirement import retirement
from tests.tools.test_delegate_capacity_interrupt import (  # registry_state is a fixture
    _ControlledChild, _Parent, _batch, registry_state,
)
from tools import async_delegation, daemon_pool
from tools.delegate_tool_dispatch import _SYNC_FALLBACK_NOTES, _run_batch


class _QuickChild(_ControlledChild):
    """Finishes at once whenever it is run, and counts how many times that happened."""

    def __init__(self, suffix=""):
        super().__init__()
        self.session_id += suffix
        self.run_count = 0

    def run_conversation(self, **_kwargs):
        self.run_count += 1
        self.started.set()
        self.finished.set()
        return {"final_response": "ran", "completed": True, "api_calls": 0, "messages": []}


def _independent_units(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "delegation:\n  max_concurrent_children: 3\n  worktree_isolation: false\n"
        "  independent_completions: true\n",
        encoding="utf-8",
    )


def _dispatch_single(runner):
    return async_delegation.dispatch_async_delegation(
        goal="probe", context=None, toolsets=None, role="leaf", model=None,
        session_key="probe-session", runner=runner, max_async_children=1)


def _fail_after_the_item_ran(monkeypatch, executor, on_call):
    """Wrap the real submit(): the item is queued and run by a real worker, then submit() raises
    untagged -- the shape of a worker-start failure that lost the race to an existing worker."""
    real_submit = executor.submit
    calls = []

    def submit(fn, *args, **kwargs):
        calls.append(fn)
        future = real_submit(fn, *args, **kwargs)
        if len(calls) == on_call:
            future.result(timeout=5)
            raise RuntimeError("injected: worker start failed after the item was queued")
        return future

    monkeypatch.setattr(executor, "submit", submit)
    return calls


def test_admission_raise_after_the_unit_ran_is_uncertain_and_not_run_again(registry_state, monkeypatch):
    parent, child = _Parent(), _QuickChild()
    executor = async_delegation._get_executor(2)
    calls = _fail_after_the_item_ran(monkeypatch, executor, on_call=1)

    result = json.loads(_run_batch(_batch(parent, child), background=True))

    assert len(calls) == 1
    assert result.get("status") == "error", result
    assert result.get("outcome_uncertain") is True, result
    assert "background=false" in result["error"]
    assert "note" not in result and "inline_results" not in result, result
    assert child.run_count == 1, "a unit a worker already ran was run again inline"
    completion = registry_state.get(timeout=5)
    assert completion["results"][0]["status"] == "completed"
    assert async_delegation.active_count() == 0


def test_later_unit_admission_raise_is_reported_uncertain_not_run_inline(registry_state, monkeypatch, tmp_path):
    _independent_units(tmp_path)
    parent = _Parent()
    first, second = _QuickChild("-first"), _QuickChild("-second")
    executor = async_delegation._get_executor(3)
    calls = _fail_after_the_item_ran(monkeypatch, executor, on_call=2)

    result = json.loads(_run_batch(_batch(parent, first, second), background=True))

    assert len(calls) == 2
    assert result.get("status") == "dispatched", result
    assert "inline_results" not in result and "not_started" not in result, result
    assert [(u["task_index"], u["outcome_uncertain"]) for u in result["outcome_uncertain"]] == [(1, True)]
    completions = [registry_state.get(timeout=5) for _ in range(2)]
    assert all(c["results"][0]["status"] == "completed" for c in completions), completions
    assert (first.run_count, second.run_count) == (1, 1), "a unit ran more than once"


def test_admission_that_raises_is_reported_and_the_child_returns_to_the_parent(registry_state, monkeypatch):
    parent, child = _Parent(), _QuickChild()

    def raising_admission(**_kwargs):
        raise sqlite3.OperationalError("database table is locked")

    monkeypatch.setattr(async_delegation, "dispatch_async_delegation_batch", raising_admission)
    result = json.loads(_run_batch(_batch(parent, child), background=True))

    assert result.get("status") == "error" and "locked" in result["error"], result
    assert result.get("outcome_uncertain") is True, result
    assert child.run_count == 0
    assert parent._active_children == [child], "the detached child was never re-attached"


@pytest.mark.parametrize("insert", ["never_written", "written_then_failed"])
def test_unrecordable_dispatch_is_not_started_and_never_runs(registry_state, monkeypatch, insert):
    parent, child = _Parent(), _QuickChild()
    real_persist = async_delegation._persist_dispatch
    ids = []

    def persist(record):
        ids.append(record["delegation_id"])
        if insert == "written_then_failed":
            real_persist(record)
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(async_delegation, "_persist_dispatch", persist)
    result = json.loads(_run_batch(_batch(parent, child), background=True))

    assert result.get("status") == "error", result
    assert "locked" in result["error"] and "capacity" not in json.dumps(result).lower(), result
    assert "Retry" in result["error"] and "background=false" in result["error"], result
    assert [t["task_index"] for t in result["not_started"]] == [0], result
    assert child.run_count == 0, "a unit state.db could not record ran inline anyway"
    assert child not in parent._active_children and child.close_count == 1
    assert async_delegation._records == {}, "the rejected unit still holds a slot"
    assert retirement.active_count() == 0, "the rejected unit still holds a retirement reservation"
    assert async_delegation.get_durable_delegation(ids[0]) is None, "the failed dispatch left a ledger row"


def test_later_unrecordable_unit_is_not_started_and_never_runs(registry_state, monkeypatch, tmp_path):
    _independent_units(tmp_path)
    parent = _Parent()
    first, second = _QuickChild("-first"), _QuickChild("-second")
    real_persist = async_delegation._persist_dispatch
    persisted = []

    def persist(record):
        persisted.append(record["delegation_id"])
        if len(persisted) == 2:
            raise sqlite3.OperationalError("database is locked")
        return real_persist(record)

    monkeypatch.setattr(async_delegation, "_persist_dispatch", persist)
    result = json.loads(_run_batch(_batch(parent, first, second), background=True))

    assert result.get("status") == "dispatched", result
    assert "inline_results" not in result, result
    (entry,) = result["not_started"]
    assert entry["task_index"] == 1 and "background=false" in entry["error"], entry
    assert registry_state.get(timeout=5)["delegation_id"] == persisted[0]
    assert (first.run_count, second.run_count) == (1, 0)
    assert second.close_count == 1 and second not in parent._active_children


def test_submit_failure_before_enqueue_is_a_clean_rejection_without_a_queue_scan(registry_state, monkeypatch):
    ran = threading.Event()
    scans = []

    def failing_copy_context():
        raise MemoryError("injected: context capture")

    monkeypatch.setattr(daemon_pool, "copy_context", failing_copy_context)
    monkeypatch.setattr(async_delegation, "_discard_queued_work_item", lambda *a: scans.append(a))
    handle = _dispatch_single(lambda: ran.set() or {"summary": "must never run"})

    assert handle["status"] == "rejected" and handle.get("reason") is None, handle
    assert scans == [], "a proven pre-enqueue failure still scanned the queue"
    assert async_delegation._records == {}
    assert retirement.active_count() == 0
    assert not ran.is_set()


def test_submit_failure_after_enqueue_discards_the_queued_item(registry_state, monkeypatch):
    ran = threading.Event()
    executor = async_delegation._get_executor(1)
    real_start = threading.Thread.start
    raised = []

    def failing_start(self):
        if self.name == "async-delegate_0":
            raise RuntimeError("injected: OS refused to start the worker thread")
        return real_start(self)

    real_submit = executor.submit

    def recording_submit(*args, **kwargs):
        try:
            return real_submit(*args, **kwargs)
        except Exception as exc:
            raised.append(exc)
            raise

    monkeypatch.setattr(executor, "submit", recording_submit)
    monkeypatch.setattr(threading.Thread, "start", failing_start)
    try:
        handle = _dispatch_single(lambda: ran.set() or {"summary": "must never run"})
    finally:
        monkeypatch.setattr(threading.Thread, "start", real_start)

    assert raised and not getattr(raised[0], "never_enqueued", False)
    assert handle["status"] == "rejected" and handle.get("reason") is None, handle
    assert executor._work_queue.empty(), "the rejected item was left in the queue"
    assert async_delegation._records == {}
    assert retirement.active_count() == 0
    done = threading.Event()
    real_submit(done.set)  # a real worker now drains whatever is still queued
    assert done.wait(5)
    assert not ran.is_set(), "the discarded work item ran anyway"


def test_capacity_rejection_of_a_single_unit_still_runs_synchronously(registry_state):
    release = threading.Event()
    occupied = threading.Event()

    def occupy():
        occupied.set()
        assert release.wait(30)
        return {"status": "completed", "summary": "slot released"}

    accepted = _dispatch_single(occupy)
    assert accepted["status"] == "dispatched" and occupied.wait(5)
    parent, child = _Parent(), _QuickChild()
    try:
        result = json.loads(_run_batch(_batch(parent, child), background=True))
    finally:
        release.set()
    assert result["note"] == _SYNC_FALLBACK_NOTES["at_capacity"], result
    assert child.run_count == 1
    assert registry_state.get(timeout=5)["delegation_id"] == accepted["delegation_id"]
