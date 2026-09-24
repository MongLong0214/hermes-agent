"""Owned-turn runner for canonical receipts: publish, run, cancel, clear.

A canonical turn runs the bound cached actor on the gateway turn pool while the actor advertises
which processes the turn owns, so /stop, /new and eviction reap only what this turn spawned.
Cancelling the awaiting coroutine must never leave that advertisement behind for a turn that did
not run, nor release the turn while its worker is still inside the actor.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import suppress
from contextvars import copy_context
from typing import Any, Callable

from agent.interrupt_compat import request_hard_interrupt

_INTERRUPT_REASON_CANCELLED = "Canonical request cancelled"
_INTERRUPT_TOOL_REASON_CANCELLED = "canonical request cancelled"
_INVALIDATION_REASON_CANCELLED = "canonical_request_cancelled"


async def run_owned_turn(
    runner: Any, agent: Any, work: Callable[[], Any], *,
    session_key: str, task_id: str, run_generation: int | None,
) -> Any:
    """Run ``work`` on ``agent`` in the turn pool, owning the processes it spawns until it exits."""
    from tools.process_registry import process_registry

    # Published on the loop so /stop, /new and eviction reap only processes this turn spawns, even
    # when the interrupt lands before the worker thread starts.
    baseline = process_registry.snapshot_running_ids(task_id)
    agent._gateway_turn_process_task_id = task_id
    agent._gateway_turn_process_baseline = baseline

    def clear_ownership() -> None:
        # The identity check keeps a later turn's publication on this actor intact.
        if agent._gateway_turn_process_baseline is baseline:
            agent._gateway_turn_process_task_id = ""
            agent._gateway_turn_process_baseline = frozenset()

    def owned_work() -> Any:
        try:
            return work()
        finally:
            clear_ownership()

    try:
        future = runner._get_executor().submit(copy_context().run, owned_work)
    except BaseException:
        clear_ownership()
        raise
    worker = asyncio.wrap_future(future)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # cancel() succeeds only while the pool has not handed the item to a thread, and a
        # cancelled item never runs: this is the race-free "never started" answer.
        if future.cancel():
            clear_ownership()
            raise
        if not future.done():
            release_generation = _interrupt_owned_turn(
                runner, agent, session_key=session_key, task_id=task_id,
                baseline=baseline, run_generation=run_generation,
            )
            await _join_worker(worker)
            if release_generation is not None:
                runner._release_running_agent_state(session_key, run_generation=release_generation)
        raise


def _interrupt_owned_turn(
    runner: Any, agent: Any, *, session_key: str, task_id: str, baseline: frozenset,
    run_generation: int | None,
) -> int | None:
    """Interrupt this turn as /stop would; return the generation whose slot it must release."""
    if run_generation is None:
        # A standalone turn claims no running slot for /stop to find, so it is interrupted and
        # reaped here with the same two steps _interrupt_running_turn composes.
        from gateway.run import _reap_gateway_turn_processes

        request_hard_interrupt(
            agent, _INTERRUPT_REASON_CANCELLED, tool_reason=_INTERRUPT_TOOL_REASON_CANCELLED,
        )
        threading.Thread(
            target=copy_context().run,
            args=(_reap_gateway_turn_processes, task_id, baseline),
            kwargs={"source": "gateway_turn_interrupt"},
            name=f"gateway-turn-reaper-{task_id[:12]}",
            daemon=True,
        ).start()
        return None
    state = runner._peek_session_state(session_key)
    if (
        state is None
        or state.turn.agent is not agent
        or not runner._is_session_run_current(session_key, run_generation)
    ):
        # /stop, /new, eviction or shutdown already took the slot; they interrupted this actor
        # and, where they reap, reaped with this turn's baseline.
        return None
    return runner._interrupt_running_turn(
        session_key,
        interrupt_reason=_INTERRUPT_REASON_CANCELLED,
        invalidation_reason=_INVALIDATION_REASON_CANCELLED,
        tool_reason=_INTERRUPT_TOOL_REASON_CANCELLED,
    )


async def _join_worker(worker: asyncio.Future) -> None:
    """Wait until the worker thread has left the actor, however often the waiter is cancelled."""
    while not worker.done():
        with suppress(asyncio.CancelledError):
            await asyncio.wait((worker,))
    if not worker.cancelled():
        # The cancelled caller never reads the abandoned result; retrieve it so asyncio does not
        # report an unobserved exception.
        worker.exception()
