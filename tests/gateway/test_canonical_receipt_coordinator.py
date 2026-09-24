"""Canonical receipt claiming uses only the bound cached actor's live DB handle."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading

import pytest

from gateway.canonical_surface import (
    CanonicalIngressEvent,
    CanonicalReceiptCoordinator,
    CanonicalSurfaceBinding,
    request_local_reply_sink,
)
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tools.process_registry import process_registry


def _binding(entry, source):
    return CanonicalSurfaceBinding(
        name="receipt-binding", session_key=entry.session_key, session_id=entry.session_id,
        telegram_chat_id=source.chat_id, telegram_chat_type=source.chat_type,
        telegram_user_id=source.user_id, telegram_thread_id=source.thread_id,
        allowed_author_ids=("author",), allowed_channel_ids=("channel",),
    )


def _event(text="hello"):
    return CanonicalIngressEvent("receipt-binding", "evt-1", "author", "channel", text)


def _runner(tmp_path, monkeypatch, home=None):
    home = home or tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user")
    entry = runner.session_store.get_or_create_session(source)
    db = runner.session_store._db

    class Agent:
        session_id = entry.session_id
        compression_in_place = True
        _session_db = db

        def __init__(self):
            self.calls = 0
            self.interrupted = None

        def interrupt(self, text=None):
            self.interrupted = text

        def run_conversation(self, text, *, conversation_history, task_id):
            self.calls += 1
            self._persist_user_message_idx = len(conversation_history)
            return {"completed": True, "session_id": task_id, "final_response": "done:" + text,
                    "messages": [*conversation_history, {"role": "user", "content": text},
                                 {"role": "assistant", "content": "done:" + text}]}

    agent = Agent()
    with runner._agent_cache_lock:
        runner._agent_cache[entry.session_key] = (agent, "exact", 0, entry.session_id)
    return runner, entry, source, db, agent


class _SaturatedPool(concurrent.futures.ThreadPoolExecutor):
    """Once ``holding`` is set, accepts work without handing it to a thread, like a full pool."""

    def __init__(self):
        super().__init__(max_workers=2)
        self.holding = False
        self.queued = []

    def submit(self, fn, /, *args, **kwargs):
        if not self.holding:
            return super().submit(fn, *args, **kwargs)
        future = concurrent.futures.Future()
        self.queued.append((future, fn, args, kwargs))
        return future

    def drain(self):
        """Hand every queued item to a thread the way ThreadPoolExecutor's work item does."""
        for future, fn, args, kwargs in self.queued:
            if future.set_running_or_notify_cancel():
                future.set_result(fn(*args, **kwargs))


def _mock_processes(monkeypatch, running, on_hold=None):
    """A process table whose reap kills exactly the ids started after the baseline."""
    reaped = threading.Event()

    def snapshot(_task_id):
        if on_hold is not None:
            on_hold()
        return frozenset(running)

    def kill_started_since(_task_id, baseline_ids, *, source):
        killed = running - set(baseline_ids)
        running.difference_update(killed)
        reaped.set()
        return len(killed)

    monkeypatch.setattr(process_registry, "snapshot_running_ids", snapshot)
    monkeypatch.setattr(process_registry, "kill_started_since", kill_started_since)
    return reaped


def _join_new_threads(before):
    for thread in set(threading.enumerate()) - before:
        thread.join(timeout=5)


def test_terminal_replay_runs_actor_once(tmp_path, monkeypatch):
    async def exercise():
        runner, entry, source, _db, agent = _runner(tmp_path, monkeypatch)
        try:
            coordinator = CanonicalReceiptCoordinator(runner)
            first = await coordinator.submit(_binding(entry, source), _event())
            second = await coordinator.submit(_binding(entry, source), _event())
            assert first == second
            assert first.status == "terminal"
            assert first.terminal_text == "done:hello"
            assert agent.calls == 1
        finally:
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())


def test_busy_lease_refuses_before_claim_and_later_retry_claims(tmp_path, monkeypatch):
    async def exercise():
        runner, entry, source, db, agent = _runner(tmp_path, monkeypatch)
        try:
            coordinator = CanonicalReceiptCoordinator(runner)
            original_acquire = runner._turn_leases.acquire
            original_claim = db.claim_meta_once
            claims = 0

            async def busy(*_args, **_kwargs):
                raise RuntimeError("lease busy")

            def count_claim(*args, **kwargs):
                nonlocal claims
                claims += 1
                return original_claim(*args, **kwargs)

            monkeypatch.setattr(runner._turn_leases, "acquire", busy)
            monkeypatch.setattr(db, "claim_meta_once", count_claim)
            with pytest.raises(ValueError, match="^canonical_turn_busy$"):
                await coordinator.submit(_binding(entry, source), _event())
            assert claims == 0
            assert agent.calls == 0

            monkeypatch.setattr(runner._turn_leases, "acquire", original_acquire)
            result = await coordinator.submit(_binding(entry, source), _event())
            assert result.status == "terminal"
            assert claims == 1
            assert agent.calls == 1
        finally:
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())


def test_normal_turn_slot_refuses_before_receipt_claim(tmp_path, monkeypatch):
    async def exercise():
        runner, entry, source, db, agent = _runner(tmp_path, monkeypatch)
        try:
            runner._session_state(entry.session_key).turn.agent = object()

            def no_claim(*_args, **_kwargs):
                raise AssertionError("occupied normal turn must refuse before receipt claim")

            monkeypatch.setattr(db, "claim_meta_once", no_claim)
            with pytest.raises(ValueError, match="^canonical_turn_busy$"):
                await CanonicalReceiptCoordinator(runner).submit(_binding(entry, source), _event())
            assert agent.calls == 0
        finally:
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())


def test_canonical_turn_uses_shared_running_slot_and_interrupt_generation(tmp_path, monkeypatch):
    async def exercise():
        runner, entry, source, _db, agent = _runner(tmp_path, monkeypatch)
        entered = threading.Event()
        released = threading.Event()
        try:
            def interrupt(_text):
                released.set()

            def blocked_run(text, *, conversation_history, task_id):
                agent.calls += 1
                entered.set()
                assert released.wait(timeout=2)
                return {"completed": False, "interrupted": True, "session_id": task_id,
                        "final_response": "", "messages": conversation_history}

            monkeypatch.setattr(agent, "interrupt", interrupt)
            monkeypatch.setattr(agent, "run_conversation", blocked_run)
            task = asyncio.create_task(
                CanonicalReceiptCoordinator(runner).submit(_binding(entry, source), _event())
            )
            while not entered.is_set():
                await asyncio.sleep(0)
            assert runner._is_session_running(entry.session_key)
            assert runner._session_state(entry.session_key).turn.agent is agent
            assert runner._session_state(entry.session_key).turn.event is None

            successor_generation = runner._interrupt_running_turn(
                entry.session_key, interrupt_reason="stop", invalidation_reason="stop",
            )
            successor = object()
            runner._session_state(entry.session_key).turn.agent = successor

            with pytest.raises(ValueError, match="^canonical_turn_interrupted$"):
                await task
            assert runner._is_session_run_current(entry.session_key, successor_generation)
            assert runner._session_state(entry.session_key).turn.agent is successor
        finally:
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())


def test_new_command_reaps_processes_spawned_by_canonical_turn_but_not_its_baseline(tmp_path, monkeypatch):
    async def exercise():
        runner, entry, source, _db, agent = _runner(tmp_path, monkeypatch)
        running = {"proc-q"}
        reaps = []
        entered, reaped, released = threading.Event(), threading.Event(), threading.Event()
        try:
            def snapshot(task_id):
                return frozenset(running) if task_id == entry.session_id else frozenset()

            def kill_started_since(task_id, baseline_ids, *, source):
                reaps.append((task_id, baseline_ids, source))
                killed = running - set(baseline_ids)
                running.difference_update(killed)
                reaped.set()
                return len(killed)

            def spawning_run(text, *, conversation_history, task_id):
                agent.calls += 1
                running.add("proc-p")
                entered.set()
                assert released.wait(timeout=5)
                return {"completed": False, "interrupted": True, "session_id": task_id,
                        "final_response": "", "messages": conversation_history}

            monkeypatch.setattr(process_registry, "snapshot_running_ids", snapshot)
            monkeypatch.setattr(process_registry, "kill_started_since", kill_started_since)
            monkeypatch.setattr(agent, "run_conversation", spawning_run)
            task = asyncio.create_task(
                CanonicalReceiptCoordinator(runner).submit(_binding(entry, source), _event())
            )
            assert await asyncio.to_thread(entered.wait, 5)

            runner._interrupt_running_turn(
                entry.session_key, interrupt_reason="new", invalidation_reason="new_command",
            )

            assert await asyncio.to_thread(reaped.wait, 5)
            assert reaps == [(entry.session_id, frozenset({"proc-q"}), "gateway_turn_interrupt")]
            assert running == {"proc-q"}
            released.set()
            with pytest.raises(ValueError, match="^canonical_turn_interrupted$"):
                await task
        finally:
            released.set()
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())


def test_completed_canonical_turn_releases_process_ownership(tmp_path, monkeypatch):
    async def exercise():
        runner, entry, source, _db, agent = _runner(tmp_path, monkeypatch)
        reaps = []
        try:
            monkeypatch.setattr(process_registry, "snapshot_running_ids", lambda _task_id: frozenset({"proc-q"}))
            monkeypatch.setattr(process_registry, "kill_started_since",
                                lambda *args, **kwargs: reaps.append(args) or 0)
            result = await CanonicalReceiptCoordinator(runner).submit(_binding(entry, source), _event())
            assert result.status == "terminal"
            assert agent._gateway_turn_process_task_id == ""
            assert agent._gateway_turn_process_baseline == frozenset()

            # A stale unwind skips the slot release, leaving the finished actor reachable to /stop and /new.
            runner._session_state(entry.session_key).turn.agent = agent
            before = set(threading.enumerate())
            runner._interrupt_running_turn(entry.session_key, interrupt_reason="stop", invalidation_reason="stop")
            for thread in set(threading.enumerate()) - before:
                thread.join(timeout=5)
            assert reaps == []
        finally:
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())


def test_cancel_while_turn_is_queued_never_runs_it_and_leaves_no_process_ownership(tmp_path, monkeypatch):
    async def exercise():
        runner, entry, source, _db, agent = _runner(tmp_path, monkeypatch)
        pool = _SaturatedPool()
        # Hold whichever pool the turn is handed to; the baseline snapshot is taken just before.
        asyncio.get_running_loop().set_default_executor(pool)
        monkeypatch.setattr(runner, "_get_executor", lambda: pool)
        running = {"proc-q"}
        _mock_processes(monkeypatch, running, on_hold=lambda: setattr(pool, "holding", True))
        try:
            task = asyncio.create_task(
                CanonicalReceiptCoordinator(runner).submit(_binding(entry, source), _event())
            )
            for _ in range(500):
                if pool.queued:
                    break
                await asyncio.sleep(0.01)
            assert pool.queued
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            pool.drain()
            assert agent.calls == 0

            # A same-task process no turn owns, then a /stop before the next turn republishes.
            running.add("proc-u")
            runner._session_state(entry.session_key).turn.agent = agent
            before = set(threading.enumerate())
            runner._interrupt_running_turn(entry.session_key, interrupt_reason="stop", invalidation_reason="stop")
            _join_new_threads(before)
            assert running == {"proc-q", "proc-u"}
        finally:
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())


def test_cancel_while_worker_runs_interrupts_it_and_holds_the_turn_until_it_exits(tmp_path, monkeypatch):
    async def exercise():
        runner, entry, source, _db, agent = _runner(tmp_path, monkeypatch)
        running = {"proc-q"}
        reaped = _mock_processes(monkeypatch, running)
        entered, interrupted, may_exit = threading.Event(), threading.Event(), threading.Event()
        try:
            def blocked_run(text, *, conversation_history, task_id):
                agent.calls += 1
                running.add("proc-p")
                entered.set()
                interrupted.wait(5)
                may_exit.wait(5)
                return {"completed": False, "interrupted": True, "session_id": task_id,
                        "final_response": "", "messages": conversation_history}

            monkeypatch.setattr(agent, "interrupt", lambda _text=None: interrupted.set())
            monkeypatch.setattr(agent, "run_conversation", blocked_run)
            coordinator = CanonicalReceiptCoordinator(runner)
            task = asyncio.create_task(coordinator.submit(_binding(entry, source), _event()))
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()

            assert await asyncio.to_thread(interrupted.wait, 2)
            assert await asyncio.to_thread(reaped.wait, 2)
            assert running == {"proc-q"}
            # The worker is still inside the actor: the turn keeps its slot, so a second turn refuses.
            with pytest.raises(ValueError, match="^canonical_turn_busy$"):
                await coordinator.submit(_binding(entry, source), _event("second"))
            assert not task.done()

            may_exit.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert agent.calls == 1
            assert not runner._is_session_running(entry.session_key)
            lease = await runner._turn_leases.acquire(
                entry.session_id, owner_key="probe", generation=0, timeout=0.5,
            )
            runner._turn_leases.release(lease)
        finally:
            interrupted.set()
            may_exit.set()
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())


def test_replaced_db_after_duplicate_read_refuses_terminal_replay(tmp_path, monkeypatch):
    async def exercise():
        runner, entry, source, db, agent = _runner(tmp_path, monkeypatch)
        try:
            coordinator = CanonicalReceiptCoordinator(runner)
            first = await coordinator.submit(_binding(entry, source), _event())
            assert first.status == "terminal"
            original_get_meta = db.get_meta

            def replace_after_read(*args, **kwargs):
                value = original_get_meta(*args, **kwargs)
                replacement = tmp_path / "replacement.db"
                replacement.touch()
                os.replace(replacement, db.db_path)
                return value

            monkeypatch.setattr(db, "get_meta", replace_after_read)
            with pytest.raises(ValueError, match="^canonical_binding_stale$"):
                await coordinator.submit(_binding(entry, source), _event())
            assert agent.calls == 1
        finally:
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())


def test_home_behind_a_symlink_binds_by_file_and_still_refuses_a_replaced_db(tmp_path, monkeypatch):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    home = tmp_path / "link" / "home"

    async def exercise():
        runner, entry, source, db, agent = _runner(tmp_path, monkeypatch, home=home)
        try:
            coordinator = CanonicalReceiptCoordinator(runner)
            first = await coordinator.submit(_binding(entry, source), _event())
            assert (first.status, first.terminal_text, agent.calls) == ("terminal", "done:hello", 1)
            original_get_meta = db.get_meta

            def replace_after_read(*args, **kwargs):
                value = original_get_meta(*args, **kwargs)
                replacement = home / "replacement.db"
                replacement.touch()
                os.replace(replacement, home / "state.db")
                return value

            monkeypatch.setattr(db, "get_meta", replace_after_read)
            with pytest.raises(ValueError, match="^canonical_binding_stale$"):
                await coordinator.submit(_binding(entry, source), _event())
            assert agent.calls == 1
        finally:
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())


def test_pending_after_interrupted_turn_never_reexecutes(tmp_path, monkeypatch):
    async def exercise():
        runner, entry, source, _db, agent = _runner(tmp_path, monkeypatch)
        try:
            coordinator = CanonicalReceiptCoordinator(runner)
            async def interrupted(*_args, **_kwargs):
                agent.calls += 1
                raise RuntimeError("interrupted")
            monkeypatch.setattr(runner, "run_bound_existing_turn", interrupted)
            with pytest.raises(RuntimeError, match="interrupted"):
                await coordinator.submit(_binding(entry, source), _event())
            result = await coordinator.submit(_binding(entry, source), _event())
            assert result.status == "pending"
            assert agent.calls == 1
        finally:
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())


def test_changed_payload_conflicts_without_actor_reexecution(tmp_path, monkeypatch):
    async def exercise():
        runner, entry, source, _db, agent = _runner(tmp_path, monkeypatch)
        try:
            coordinator = CanonicalReceiptCoordinator(runner)
            await coordinator.submit(_binding(entry, source), _event("first"))
            with pytest.raises(ValueError, match="^canonical_receipt_conflict$"):
                await coordinator.submit(_binding(entry, source), _event("changed"))
            assert agent.calls == 1
        finally:
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())


@pytest.mark.parametrize("bad", ["binding", "handle"], ids=["wrong-profile-binding", "mismatched-handle"])
def test_bad_binding_or_handle_refuses_before_claim_or_actor(tmp_path, monkeypatch, bad):
    async def exercise():
        runner, entry, source, db, agent = _runner(tmp_path, monkeypatch)
        try:
            coordinator = CanonicalReceiptCoordinator(runner)
            def no_write(*_args, **_kwargs):
                raise AssertionError("must not write SQLite")
            monkeypatch.setattr(db, "claim_meta_once", no_write)
            binding = _binding(entry, source)
            if bad == "binding":
                binding = CanonicalSurfaceBinding(**{**binding.__dict__, "session_key": "agent:other:telegram:dm:no"})
            else:
                agent._session_db = object()
            with pytest.raises(ValueError, match="^canonical_binding_stale$"):
                await coordinator.submit(binding, _event())
            assert agent.calls == 0
        finally:
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())


def test_actor_replacement_under_lease_refuses_before_execution(tmp_path, monkeypatch):
    async def exercise():
        runner, entry, source, _db, agent = _runner(tmp_path, monkeypatch)
        try:
            original = runner.run_bound_existing_turn
            async def replace_then_run(*args, **kwargs):
                with runner._agent_cache_lock:
                    runner._agent_cache[entry.session_key] = (object(), "exact", 0, entry.session_id)
                return await original(*args, **kwargs)
            monkeypatch.setattr(runner, "run_bound_existing_turn", replace_then_run)
            with pytest.raises(ValueError, match="^canonical_agent_replaced$"):
                await CanonicalReceiptCoordinator(runner).submit(_binding(entry, source), _event())
            assert agent.calls == 0
        finally:
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())


def test_foreign_held_lease_refuses_before_actor_invocation(tmp_path, monkeypatch):
    async def exercise():
        runner, entry, source, _db, agent = _runner(tmp_path, monkeypatch)
        try:
            event = _event()
            foreign = await runner._turn_leases.acquire(
                entry.session_id, owner_key="canonical:foreign", generation=1,
            )

            async def discard(_result):
                return None

            with pytest.raises(ValueError, match="^canonical_turn_busy$"):
                await runner.run_bound_existing_turn(
                    _binding(entry, source), event, entry,
                    reply_sink=request_local_reply_sink(discard), held_lease=foreign,
                )
            assert agent.calls == 0
        finally:
            runner._turn_leases.release(foreign)
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())


def test_actor_swapped_during_run_leaves_receipt_pending_without_terminal_cas(tmp_path, monkeypatch):
    async def exercise():
        runner, entry, source, db, agent = _runner(tmp_path, monkeypatch)
        try:
            class Replacement:
                session_id = entry.session_id
                compression_in_place = True
                _session_db = db

                def run_conversation(self, *_args, **_kwargs):
                    raise AssertionError("pending receipt must not retry")

            replacement = Replacement()
            original_run = agent.run_conversation
            cas_calls = 0

            def swap_during_run(*args, **kwargs):
                with runner._agent_cache_lock:
                    runner._agent_cache[entry.session_key] = (
                        replacement, "exact", 0, entry.session_id,
                    )
                return original_run(*args, **kwargs)

            original_cas = db.compare_and_set_meta

            def count_terminal_cas(*args, **kwargs):
                nonlocal cas_calls
                cas_calls += 1
                return original_cas(*args, **kwargs)

            monkeypatch.setattr(agent, "run_conversation", swap_during_run)
            monkeypatch.setattr(db, "compare_and_set_meta", count_terminal_cas)
            coordinator = CanonicalReceiptCoordinator(runner)
            with pytest.raises(ValueError, match="^canonical_agent_replaced$"):
                await coordinator.submit(_binding(entry, source), _event())
            assert agent.calls == 1
            assert cas_calls == 0
            pending = await coordinator.submit(_binding(entry, source), _event())
            assert pending.status == "pending"
            assert cas_calls == 0
        finally:
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())


def test_receipt_key_binds_session_key_but_not_rotating_session_id():
    event = _event()
    binding = CanonicalSurfaceBinding(
        name="receipt-binding", session_key="agent:one", session_id="old-session",
        telegram_chat_id="chat", telegram_chat_type="dm", telegram_user_id="user",
        telegram_thread_id=None, allowed_author_ids=("author",), allowed_channel_ids=("channel",),
    )
    other_key = CanonicalSurfaceBinding(**{**binding.__dict__, "session_key": "agent:two"})
    rotated_head = CanonicalSurfaceBinding(**{**binding.__dict__, "session_id": "new-session"})
    other_origin = CanonicalSurfaceBinding(**{**binding.__dict__, "telegram_thread_id": "other-thread"})

    assert CanonicalReceiptCoordinator._key(binding, event) != CanonicalReceiptCoordinator._key(other_key, event)
    assert CanonicalReceiptCoordinator._fingerprint(binding, event) != CanonicalReceiptCoordinator._fingerprint(other_key, event)
    assert CanonicalReceiptCoordinator._key(binding, event) == CanonicalReceiptCoordinator._key(rotated_head, event)
    assert CanonicalReceiptCoordinator._fingerprint(binding, event) == CanonicalReceiptCoordinator._fingerprint(rotated_head, event)
    assert CanonicalReceiptCoordinator._key(binding, event) != CanonicalReceiptCoordinator._key(other_origin, event)
    assert CanonicalReceiptCoordinator._fingerprint(binding, event) != CanonicalReceiptCoordinator._fingerprint(other_origin, event)


def test_stale_cached_tuple_session_refuses_before_receipt_claim(tmp_path, monkeypatch):
    async def exercise():
        runner, entry, source, db, agent = _runner(tmp_path, monkeypatch)
        try:
            with runner._agent_cache_lock:
                runner._agent_cache[entry.session_key] = (agent, "exact", 0, "stale-session")

            def no_claim(*_args, **_kwargs):
                raise AssertionError("stale cached tuple must fail before receipt claim")

            monkeypatch.setattr(db, "claim_meta_once", no_claim)
            with pytest.raises(ValueError, match="^canonical_binding_stale$"):
                await CanonicalReceiptCoordinator(runner).submit(_binding(entry, source), _event())
            assert agent.calls == 0
        finally:
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())


def test_terminal_cas_failure_leaves_pending(tmp_path, monkeypatch):
    async def exercise():
        runner, entry, source, db, agent = _runner(tmp_path, monkeypatch)
        try:
            monkeypatch.setattr(db, "compare_and_set_meta", lambda *_args, **_kwargs: False)
            coordinator = CanonicalReceiptCoordinator(runner)
            with pytest.raises(ValueError, match="^canonical_receipt_terminal_unconfirmed$"):
                await coordinator.submit(_binding(entry, source), _event())
            pending = await coordinator.submit(_binding(entry, source), _event())
            assert pending.status == "pending"
            assert agent.calls == 1
        finally:
            runner.session_store.close_all_db_handles()
    asyncio.run(exercise())
