"""Telegram recovery contracts (``plugins/platforms/telegram/telegram_recovery.py``).

An ``initialize()`` that never finished still has its request transports closed, after it unwinds, when
its connect is cancelled (the runner's connect deadline) or its retry ladder is exhausted; and a final reply
refused while polling was degraded is redelivered once polling recovers inside the SAME adapter instance.
Transports are fake: each request holds a local ``socketpair`` as its pool, so the process's socket fd count
is the witness. Nothing contacts Telegram.
"""
import asyncio
import gc
import os
import socket
import stat
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.telegram import telegram_send_path_recovery
from plugins.platforms.telegram.adapter import TelegramAdapter


def _open_fds() -> int:
    """Open socket fds of this process (the fake pools are sockets; files the teardown path opens and
    closes on its own threads would only add noise)."""
    gc.collect()  # an unreachable socket from an earlier test closes on collection, not on our schedule
    count = 0
    for name in os.listdir("/dev/fd"):
        try:
            count += stat.S_ISSOCK(os.fstat(int(name)).st_mode)
        except OSError:  # the listing's own descriptor, already closed
            pass
    return count


class _Request:
    """PTB request double: ``initialize`` (re)opens its pool like ``HTTPXRequest`` rebuilding a closed
    client; ``shutdown`` closes it. The pool is a real local socket pair, never a network connection."""

    def __init__(self, events):
        self._events, self._pool = events, None

    @property
    def closed(self) -> bool:
        return self._pool is None

    async def initialize(self):
        if self._pool is None:
            self._pool = socket.socketpair()

    async def shutdown(self):
        if self._pool is not None:
            for sock in self._pool:
                sock.close()
            self._pool = None
        self._events.append("request closed")


def _app(initialize, requests):
    app = MagicMock()
    app.initialize = initialize
    app.shutdown = AsyncMock()  # PTB: no-op for an app that never finished initialize()
    app.start = AsyncMock()
    app.running = False
    app.updater.running = False
    app.updater.start_polling = AsyncMock()
    app.bot._request = requests
    return app


def _bare_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="***"))


@pytest.mark.asyncio
async def test_cancelled_connect_leaves_no_open_request_transport():
    """The runner's connect deadline cancels connect() inside a cancellation-shielded initialize(): the
    deadline helper abandons the child with no cleanup and ``app.shutdown()`` no-ops for it. Once the child
    unwinds its pools must be closed — the fd count returns to where it started."""
    events = []
    requests = (_Request(events), _Request(events))
    entered, release = asyncio.Event(), asyncio.Event()

    async def _shielded_initialize():
        for request in requests:
            await request.initialize()
        entered.set()
        while not release.is_set():  # an anyio-shielded httpcore scope swallows the cancel
            try:
                await release.wait()
            except asyncio.CancelledError:
                events.append("init cancelled")
        events.append("init exited")

    adapter = _bare_adapter()
    app = _app(_shielded_initialize, requests)
    adapter._app, adapter._bot = app, app.bot
    fds_before = _open_fds()
    ladder = asyncio.ensure_future(adapter._initialize_app_with_retries(MagicMock()))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert _open_fds() == fds_before + 4  # two request pools, one socket pair each

        ladder.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ladder
        real_cancel = adapter._cancel_pending_delivery_tasks

        async def _cancel_then_unwind():
            await real_cancel()
            release.set()  # the shielded child unwinds only now, late in disconnect()

        adapter._cancel_pending_delivery_tasks = _cancel_then_unwind
        await asyncio.wait_for(adapter.disconnect(), timeout=10)
    finally:
        release.set()  # a failed assertion must not leave a cancel-swallowing task for loop teardown

    assert all(request.closed for request in requests)
    assert _open_fds() == fds_before
    # Closed only after the child left initialize(): closing under it would let it reopen a pool no-one owns.
    assert events.index("init exited") < events.index("request closed")
    # The abandoned app never started, so it cannot run a getUpdates loop beside a successor.
    app.start.assert_not_awaited()
    app.updater.start_polling.assert_not_awaited()


@pytest.mark.asyncio
async def test_abandoned_initialize_owner_is_retained_until_its_close_completes():
    events = []
    requests = (_Request(events),)
    entered, release = asyncio.Event(), asyncio.Event()
    close_started, close_release = asyncio.Event(), asyncio.Event()

    async def _shielded_initialize():
        await requests[0].initialize()
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass
        events.append("init exited")

    real_shutdown = requests[0].shutdown

    async def _slow_shutdown():
        close_started.set()
        await close_release.wait()
        await real_shutdown()

    requests[0].shutdown = _slow_shutdown
    adapter = _bare_adapter()
    app = _app(_shielded_initialize, requests)
    adapter._app, adapter._bot = app, app.bot
    ladder = asyncio.ensure_future(adapter._initialize_app_with_retries(MagicMock()))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        ladder.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ladder

        owners = set(getattr(adapter, "_abandoned_initialize_owners", ()))
        assert len(owners) == 1, "nothing owns the abandoned initialize()"
        owner = next(iter(owners))
        # Detached from the adapter (disconnect() waits on the owner instead of racing it) and kept out of
        # the background-task set that base teardown cancels.
        assert adapter._app is None and adapter._bot is None
        assert owner not in adapter._background_tasks
        await adapter._cancel_pending_delivery_tasks()
        assert not owner.done()

        release.set()
        await asyncio.wait_for(close_started.wait(), timeout=5)
        assert not owner.done() and owner in adapter._abandoned_initialize_owners  # still closing
        assert not requests[0].closed

        close_release.set()
        await asyncio.wait_for(asyncio.shield(owner), timeout=5)
    finally:
        release.set()
        close_release.set()
    assert requests[0].closed
    assert owner not in adapter._abandoned_initialize_owners
    assert events == ["init exited", "request closed"]


@pytest.mark.asyncio
async def test_exhausted_init_ladder_leaves_no_open_request_transport(monkeypatch):
    requests = (_Request([]), _Request([]))

    async def _unreachable():
        for request in requests:  # Bot.initialize() reopens the shared requests every attempt
            await request.initialize()
        raise OSError("api.telegram.org unreachable")

    builder = MagicMock()
    builder.build.side_effect = lambda: _app(_unreachable, requests)
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock())
    adapter = _bare_adapter()
    adapter._app = builder.build()
    adapter._bot = adapter._app.bot
    fds_before = _open_fds()

    with pytest.raises(OSError):
        await adapter._initialize_app_with_retries(builder)
    assert _open_fds() == fds_before + 4  # the last attempt's pools are still open

    await asyncio.wait_for(adapter.disconnect(), timeout=10)

    assert all(request.closed for request in requests)
    assert _open_fds() == fds_before


# -- in-place redelivery after a degraded recovery -------------------------------------------------------

_REPLY = "the final answer"


def _live_adapter() -> TelegramAdapter:
    adapter = _bare_adapter()
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
    adapter._rich_send_disabled = True
    adapter._running = True
    adapter.set_owner_profile(None)
    return adapter


def _runner(adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._active_profile_name = lambda: "default"
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = runner.session_store = None
    runner._async_session_store = store
    runner._running = True
    runner._delivery_adapter_for = lambda source: adapter  # no reconnect: the same instance stays live
    adapter.gateway_runner = runner
    return runner


def _state(oid):
    with dl._connect() as conn:
        row = conn.execute("SELECT state FROM delivery_obligations WHERE obligation_id=?", (oid,)).fetchone()
    return row[0] if row else None


async def _drain(adapter):
    for _ in range(3):  # a sweep may re-arm itself once
        tasks = [t for t in adapter._background_tasks if not t.done()]
        if not tasks:
            return
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)


@pytest.mark.asyncio
async def test_reply_refused_while_degraded_is_redelivered_once_after_in_place_recovery(monkeypatch):
    monkeypatch.setattr(telegram_send_path_recovery, "_SWEEP_SETTLE_SECONDS", 0)
    adapter = _live_adapter()
    runner = _runner(adapter)
    generation, _ = adapter._begin_polling_generation()  # polling restarted in place: unproven until progress

    event = MessageEvent(
        text="hello agent", message_type=MessageType.TEXT, message_id="msg-7",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm"))
    oid = await adapter._record_delivery_obligation(event, "agent:main:telegram:dm:123", _REPLY, adapter, False)
    refused = await adapter.send(chat_id="123", content=_REPLY)
    assert refused.error == "send_path_degraded"
    await adapter._finalize_delivery_obligation(oid, refused, event, adapter)
    assert _state(oid) == "failed"
    adapter._bot.send_message.assert_not_awaited()

    assert adapter._record_polling_progress(generation) is True  # getUpdates proven again, same instance
    await _drain(adapter)

    assert _state(oid) == "delivered"
    assert adapter._bot.send_message.await_count == 1
    assert _REPLY in adapter._bot.send_message.await_args.kwargs["text"]
    # A later healthy round-trip is not a recovery edge, and the row can no longer be claimed.
    assert adapter._record_polling_progress(generation) is True
    await _drain(adapter)
    assert await runner._redeliver_failed_obligations_for_platform(Platform.TELEGRAM) == 0
    assert adapter._bot.send_message.await_count == 1
