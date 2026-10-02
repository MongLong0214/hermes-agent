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

        # Shrink the budget so the test completes in well under a second regardless of outcome;
        # a shortened confirmation timeout is exactly what the finding calls for.
        monkeypatch.setattr(scheduler_delivery, "_LIVE_SEND_CONFIRM_TIMEOUT_S", 0.2)

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

    def test_fallback_after_timeout_excludes_a_late_live_dispatch(self, monkeypatch, loop_thread):
        """sol-audit R68-1: a RUNNING gateway loop that is blocked past the confirmation timeout
        still has the send queued. Once the scheduler falls back to standalone, releasing the loop
        must not let that queued coroutine reach the live adapter as well — exactly one of the two
        sends may happen. (An unpumped loop cannot witness this: ``live_adapter_ready`` requires a
        running loop, so the live path, and the race, would be skipped entirely.)"""
        from cron import scheduler_delivery

        monkeypatch.setattr(scheduler_delivery, "_LIVE_SEND_CONFIRM_TIMEOUT_S", 0.2)

        sends = []
        loop_blocked, release_loop = threading.Event(), threading.Event()

        def block_loop():
            loop_blocked.set()
            release_loop.wait(10)

        loop_thread.call_soon_threadsafe(block_loop)
        assert loop_blocked.wait(5), "the gateway loop never reached the blocking callback"

        async def live_send(chat_id, content, metadata=None):
            sends.append("live accepted")
            return SimpleNamespace(success=True, message_id="m-live", raw_response={})

        async def standalone_send(*args, **kwargs):
            sends.append("standalone accepted")
            return {"success": True, "message_id": "m-standalone"}

        scheduled = []
        real_run_coroutine_threadsafe = asyncio.run_coroutine_threadsafe

        def counting_run_coroutine_threadsafe(coro, loop):
            scheduled.append(loop)
            return real_run_coroutine_threadsafe(coro, loop)

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
        job = {
            "id": "blocked-loop-job",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }

        try:
            with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
                 patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
                 patch("asyncio.run_coroutine_threadsafe", new=counting_run_coroutine_threadsafe), \
                 patch("tools.send_message_tool._send_to_platform", new=standalone_send):
                result = _deliver_result(
                    job, "Hello world",
                    adapters={Platform.TELEGRAM: SimpleNamespace(send=live_send)},
                    loop=loop_thread,
                )
            assert sends == ["standalone accepted"], sends
        finally:
            release_loop.set()

        # Let the released loop run everything that was queued behind the block.
        for _ in range(3):
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0.05), loop_thread).result(5)

        assert scheduled == [loop_thread], "the live lane must actually have been attempted"
        assert result is None, f"expected the standalone fallback to deliver, got: {result!r}"
        assert sends == ["standalone accepted"], (
            f"fallback won, so the queued live send must refuse to dispatch; got {sends}")
