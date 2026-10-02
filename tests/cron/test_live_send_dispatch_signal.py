"""Regression for sol-audit L6-2: a timed-out live cron send could be duplicated.

``cron.scheduler_delivery._live_send_text`` used ``future.cancel()`` to decide whether a
``run_coroutine_threadsafe`` coroutine had ever started after its 60s confirmation timeout: False
("already running") skipped the standalone fallback, True ("never started") fell through to it.
That proof is invalid. ``asyncio.futures._chain_future`` never calls
``set_running_or_notify_cancel()`` on the wrapper ``concurrent.futures.Future``, so its state stays
PENDING — and therefore cancellable, with ``cancel()`` returning True — for as long as the
underlying Task has not finished, including while it is genuinely in flight. A real probe below
(mirroring the one the sol audit used) proves this directly against the stdlib.

``tests/cron/test_scheduler.py::TestDeliverResultTimeoutCancelsFuture`` cannot catch this: it
patches ``asyncio.run_coroutine_threadsafe`` itself and hand-sets ``future.cancel()``'s return
value, so it never exercises real ``run_coroutine_threadsafe``/Task semantics. This file uses a
REAL event loop running on a background thread and a real coroutine that signals it has started
before stalling past a shortened confirmation timeout.
"""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cron.scheduler import _deliver_result
from gateway.config import Platform


@pytest.fixture
def loop_thread():
    """A real asyncio loop pumped by a background thread (mirrors
    tests/cron/test_warning_transport_contract.py's fixture)."""
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(started.set)
        loop.run_forever()

    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(5)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


def test_run_coroutine_threadsafe_cancel_returns_true_after_dispatch_began(loop_thread):
    """Direct probe of the stdlib claim the fix relies on: cancel() is not proof of
    "never started" — it can return True well after the coroutine's first line ran."""
    dispatch_began = threading.Event()

    async def stalls_after_signalling():
        dispatch_began.set()
        await asyncio.sleep(5)
        return "sent"

    future = asyncio.run_coroutine_threadsafe(stalls_after_signalling(), loop_thread)
    with pytest.raises(TimeoutError):
        future.result(timeout=0.3)

    assert dispatch_began.is_set(), "the coroutine should have started well within 0.3s"
    assert future.cancel() is True, (
        "cancel() returning True here is exactly the false signal L6-2 found: the coroutine "
        "had already started (dispatch_began is set) yet cancel() still reports cancellable"
    )


class TestLiveAdapterCancelDoesNotProveNeverDispatched:
    """End-to-end through cron.scheduler._deliver_result -> _live_send_text with a real loop."""

    def test_in_flight_confirmation_timeout_does_not_duplicate_via_standalone(
        self, monkeypatch, loop_thread
    ):
        """A real coroutine that signals dispatch began, then stalls past a shortened
        confirmation timeout, must not trigger the standalone fallback — that would
        duplicate an accepted-but-slow Telegram send."""
        from cron import scheduler_delivery

        # Shrink both budgets so the test completes in well under a second regardless of
        # outcome; a shortened confirmation timeout is exactly what the finding calls for.
        monkeypatch.setattr(scheduler_delivery, "_LIVE_SEND_CONFIRM_TIMEOUT_S", 0.2)
        monkeypatch.setattr(scheduler_delivery, "_DISPATCH_SIGNAL_GRACE_S", 0.3)

        dispatch_started = threading.Event()

        async def stalling_send(chat_id, content, metadata=None):
            dispatch_started.set()
            await asyncio.sleep(5)  # outlives the test; the loop_thread fixture tears it down
            return MagicMock(success=True)

        adapter = SimpleNamespace(send=stalling_send)

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        job = {
            "id": "in-flight-job",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }

        standalone_send = AsyncMock(return_value={"success": True})

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("tools.send_message_tool._send_to_platform", new=standalone_send):
            result = _deliver_result(
                job, "Hello world",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop_thread,
            )

        assert dispatch_started.wait(1), "the live send coroutine never started"
        standalone_send.assert_not_awaited()
        assert result is None, f"expected the in-flight send to count as handled, got: {result!r}"

    def test_truly_wedged_dispatch_still_falls_back_to_standalone(self, monkeypatch):
        """Preserve the other half of the original fix: a coroutine that never gets a chance
        to run at all (loop wedged) must still fall through to standalone, or the cron
        message is silently dropped."""
        from cron import scheduler_delivery

        monkeypatch.setattr(scheduler_delivery, "_LIVE_SEND_CONFIRM_TIMEOUT_S", 0.2)
        monkeypatch.setattr(scheduler_delivery, "_DISPATCH_SIGNAL_GRACE_S", 0.2)

        # A loop that is created but never pumped: run_coroutine_threadsafe queues the
        # scheduling callback, but nothing ever executes it, so the coroutine's first line
        # (dispatch_began.set()) can never run — genuinely "never started".
        wedged_loop = asyncio.new_event_loop()

        async def never_runs(chat_id, content, metadata=None):
            raise AssertionError("must not run on a wedged loop")

        adapter = SimpleNamespace(send=never_runs)

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        job = {
            "id": "wedged-job",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }

        standalone_send = AsyncMock(return_value={"success": True})

        try:
            with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
                 patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
                 patch("tools.send_message_tool._send_to_platform", new=standalone_send):
                result = _deliver_result(
                    job, "Hello world",
                    adapters={Platform.TELEGRAM: adapter},
                    loop=wedged_loop,
                )
        finally:
            wedged_loop.close()

        standalone_send.assert_awaited_once()
        assert result is None, f"expected the standalone fallback to deliver, got: {result!r}"
