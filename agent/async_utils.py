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


def safe_schedule_threadsafe_with_dispatch_signal(
    coro: Coroutine[Any, Any, Any], loop: Optional[asyncio.AbstractEventLoop], *,
    logger: Optional[logging.Logger] = None,
    log_message: str = "Failed to schedule coroutine on loop", log_level: int = logging.DEBUG,
) -> Tuple[Optional[Future], threading.Event]:
    """Like :func:`safe_schedule_threadsafe`, plus a ``threading.Event`` the coroutine sets as its
    very first action, before any real work.

    ``future.cancel()`` on a ``run_coroutine_threadsafe`` future does NOT prove ``coro`` never
    started: the wrapper ``concurrent.futures.Future`` stays in the PENDING state (and therefore
    cancellable, with ``cancel()`` returning True) for as long as the underlying Task has not
    finished — including while it is genuinely in flight (``asyncio.futures._chain_future`` never
    calls ``set_running_or_notify_cancel`` on it). The returned event is the only trustworthy
    "did this ever run" signal: a Task cancelled before its first step throws ``CancelledError``
    into the coroutine without executing any of its body, so the event stays unset forever; once
    set it stays set, cancellation or not.
    """
    dispatch_began = threading.Event()

    async def _signal_then_run() -> Any:
        dispatch_began.set()
        return await coro

    tracked = _signal_then_run()
    log = logger if logger is not None else _DEFAULT_LOGGER
    try:
        if loop is None:
            raise RuntimeError("loop is None")
        future = asyncio.run_coroutine_threadsafe(tracked, loop)
    except Exception as exc:
        tracked.close()
        coro.close()
        log.log(log_level, "%s: %s", log_message, exc)
        return None, dispatch_began
    return future, dispatch_began


def consume_detached_task_result(task: "asyncio.Future[Any]") -> None:
    """``add_done_callback`` for cancelled-and-detached tasks: observe the exception so the
    loop does not log "exception was never retrieved"; cancellation and terminal errors
    are swallowed because the task's owner already gave up on it."""
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass
