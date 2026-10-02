"""Tests for agent.async_utils.safe_schedule_threadsafe."""

from __future__ import annotations

import asyncio
import gc
import warnings
from concurrent.futures import Future
from unittest.mock import patch


from agent.async_utils import safe_schedule_threadsafe, safe_schedule_threadsafe_gated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_unawaited_warnings(caught, *, coro_name: str = "") -> bool:
    """Return True if no "X was never awaited" warning slipped through.

    When *coro_name* is provided, only warnings naming that coroutine are
    counted
    """
    bad = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "was never awaited" in str(w.message)
        and (not coro_name or coro_name in str(w.message))
    ]
    return not bad


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSafeScheduleThreadsafe:
    def test_returns_future_on_success(self):
        loop = asyncio.new_event_loop()
        try:
            import threading
            ready = threading.Event()
            stop = threading.Event()

            def _runner():
                asyncio.set_event_loop(loop)
                ready.set()
                loop.run_until_complete(_wait_for_stop(stop))

            async def _wait_for_stop(ev):
                while not ev.is_set():
                    await asyncio.sleep(0.005)

            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            ready.wait(timeout=2)

            async def _sample():
                return 42

            fut = safe_schedule_threadsafe(_sample(), loop)
            assert isinstance(fut, Future)
            assert fut.result(timeout=2) == 42

            stop.set()
            t.join(timeout=2)
        finally:
            if loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
            loop.close()



    def test_scheduling_exception_closes_coroutine(self):
        """If run_coroutine_threadsafe raises, close the coroutine and return None."""
        # A loop that *looks* open but raises on submission
        loop = asyncio.new_event_loop()
        try:
            async def _sample():
                return "ok"

            coro = _sample()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with patch(
                    "agent.async_utils.asyncio.run_coroutine_threadsafe",
                    side_effect=RuntimeError("scheduler down"),
                ):
                    result = safe_schedule_threadsafe(coro, loop)
                del coro
                gc.collect()

            assert result is None
            assert _no_unawaited_warnings(caught, coro_name='_sample')
        finally:
            loop.close()


class TestSafeScheduleThreadsafeGated:
    """Regression for sol-audit L6-2 / R68-1 (duplicate live cron sends, cron/scheduler_delivery.py)."""

    def test_a_coroutine_that_ran_wins_the_gate(self):
        """The coroutine claims the gate as its first action, so once it ran, a caller that
        stops waiting can no longer abandon it."""
        import threading

        loop = asyncio.new_event_loop()
        ready = threading.Event()
        stop = threading.Event()

        def _runner():
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_until_complete(_wait_for_stop(stop))

        async def _wait_for_stop(ev):
            while not ev.is_set():
                await asyncio.sleep(0.005)

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        try:
            ready.wait(timeout=2)

            async def _sample():
                return 42

            fut, gate = safe_schedule_threadsafe_gated(_sample(), loop)
            assert isinstance(fut, Future)
            assert fut.result(timeout=2) == 42
            assert gate.abandon() is False
        finally:
            stop.set()
            t.join(timeout=2)
            if loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
            loop.close()

    def test_scheduling_exception_closes_both_coroutines(self):
        """The wrapper AND the real coroutine it closes over must both be closed on failure —
        closing only the wrapper (which never ran) leaves the inner one unawaited forever."""
        loop = asyncio.new_event_loop()
        try:
            async def _sample():
                return "ok"

            coro = _sample()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with patch(
                    "agent.async_utils.asyncio.run_coroutine_threadsafe",
                    side_effect=RuntimeError("scheduler down"),
                ):
                    result, gate = safe_schedule_threadsafe_gated(coro, loop)
                del coro
                gc.collect()

            assert result is None
            assert gate.abandon() is True
            assert _no_unawaited_warnings(caught)
        finally:
            loop.close()

    def test_abandoned_before_its_first_step_the_body_never_runs(self):
        """The caller abandons while the Task is still queued (loop not pumped yet); when the
        loop then runs, the coroutine is refused: its body never executes, the future ends
        cancelled, and nothing is left unawaited."""
        loop = asyncio.new_event_loop()
        ran = []
        try:
            async def _sample():
                ran.append(True)
                return "sent"

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fut, gate = safe_schedule_threadsafe_gated(_sample(), loop)
                assert gate.abandon() is True
                loop.run_until_complete(asyncio.sleep(0.01))
                gc.collect()

            assert ran == []
            assert fut.cancelled()
            assert _no_unawaited_warnings(caught)
        finally:
            loop.close()


