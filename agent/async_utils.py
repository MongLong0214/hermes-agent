"""Async/sync bridging helpers.

``asyncio.run_coroutine_threadsafe`` can raise ``RuntimeError`` (loop closed during a
shutdown race); the coroutine is then never awaited or closed, which triggers a
"coroutine was never awaited" RuntimeWarning and leaks its frame. The helpers here
close the coroutine on scheduling failure. ``future.result()`` failures are deliberately
NOT handled: once the loop accepts the coroutine its lifecycle belongs to the loop.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from typing import Any, Coroutine, Optional, Tuple


_DEFAULT_LOGGER = logging.getLogger(__name__)


def safe_schedule_threadsafe(
    coro: Coroutine[Any, Any, Any], loop: Optional[asyncio.AbstractEventLoop], *,
    logger: Optional[logging.Logger] = None,
    log_message: str = "Failed to schedule coroutine on loop", log_level: int = logging.DEBUG,
) -> Optional[Future]:
    """Schedule ``coro`` on ``loop`` from a sync context, leak-safe.

    Returns the Future on success, or ``None`` if the loop is missing or scheduling
    raised; in every failure path the coroutine is closed. Callers keep full control
    over the returned future (``.result(timeout=...)``, callbacks, fire-and-forget).
    """
    log = logger if logger is not None else _DEFAULT_LOGGER
    try:
        if loop is None:
            raise RuntimeError("loop is None")
        return asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception as exc:
        if asyncio.iscoroutine(coro):
            coro.close()
        log.log(log_level, "%s: %s", log_message, exc)
        return None


class DispatchGate:
    """One-shot handoff between a scheduled coroutine and a caller that may give up on it.

    ``future.cancel()`` on a ``run_coroutine_threadsafe`` future proves nothing about whether the
    coroutine ran or will run: the wrapper ``concurrent.futures.Future`` stays PENDING (so
    ``cancel()`` returns True) while the Task is genuinely in flight, and a cancel issued before
    the loop has even created the Task arrives AFTER that Task's first step was queued, so a
    blocked loop that resumes still runs the body once. The gate replaces both guesses with a
    single locked transition out of ``pending``: the coroutine's first action claims
    ``dispatched``; a caller that stops waiting claims ``abandoned``. Exactly one side wins, and a
    coroutine that loses never runs its body.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "pending"

    def _claim(self, state: str) -> bool:
        with self._lock:
            if self._state == "pending":
                self._state = state
            return self._state == state

    def claim_dispatch(self) -> bool:
        """Coroutine side: True = proceed; False = the caller already abandoned it."""
        return self._claim("dispatched")

    def abandon(self) -> bool:
        """Caller side: True = the coroutine's body has not run and now never will; False = it
        already began (its outcome is the coroutine's, not the caller's to replace)."""
        return self._claim("abandoned")


def safe_schedule_threadsafe_gated(
    coro: Coroutine[Any, Any, Any], loop: Optional[asyncio.AbstractEventLoop], *,
    logger: Optional[logging.Logger] = None,
    log_message: str = "Failed to schedule coroutine on loop", log_level: int = logging.DEBUG,
) -> Tuple[Optional[Future], DispatchGate]:
    """Like :func:`safe_schedule_threadsafe`, but ``coro`` runs only if it claims the returned
    :class:`DispatchGate` before the caller abandons it (see that class). A refused coroutine is
    closed unstarted and its Task ends cancelled."""
    gate = DispatchGate()

    async def _gated() -> Any:
        if not gate.claim_dispatch():
            coro.close()
            raise asyncio.CancelledError("abandoned before dispatch")
        return await coro

    gated = _gated()
    log = logger if logger is not None else _DEFAULT_LOGGER
    try:
        if loop is None:
            raise RuntimeError("loop is None")
        future = asyncio.run_coroutine_threadsafe(gated, loop)
    except Exception as exc:
        gated.close()
        coro.close()
        log.log(log_level, "%s: %s", log_message, exc)
        return None, gate
    return future, gate


def consume_detached_task_result(task: "asyncio.Future[Any]") -> None:
    """``add_done_callback`` for cancelled-and-detached tasks: observe the exception so the
    loop does not log "exception was never retrieved"; cancellation and terminal errors
    are swallowed because the task's owner already gave up on it."""
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass
