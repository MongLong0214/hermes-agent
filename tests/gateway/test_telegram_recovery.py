"""Telegram recovery contracts (``plugins/platforms/telegram/telegram_recovery.py``).

A final reply refused while polling was degraded is redelivered once polling recovers inside the SAME
adapter instance, from the owning profile's ledger rows; and an ``initialize()`` that never finished still has
its httpx transports closed, after it unwinds, when its connect is cancelled or its ladder is exhausted.
"""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter

_REPLY = "the final answer"


def _adapter(profile=None) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
    adapter._rich_send_disabled = True
    adapter._running = True
    adapter.set_owner_profile(profile)
    return adapter


def _runner(adapter, profile=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    primary = MagicMock()
    primary.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: primary if profile else adapter}
    runner._profile_adapters = {profile: {Platform.TELEGRAM: adapter}} if profile else {}
    runner._active_profile_name = lambda: "default"
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = runner.session_store = None
    runner._async_session_store = store
    runner._running = True
    runner._delivery_adapter_for = lambda source: adapter  # no reconnect: the same instance stays live
    adapter.gateway_runner = runner
    return runner, primary


async def _refuse_while_degraded(adapter) -> str:
    """The ``send_final_ledgered`` bracket (minus its retry sleeps) for one reply refused while degraded."""
    event = MessageEvent(
        text="hello agent", message_type=MessageType.TEXT, message_id="msg-7",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm"))
    oid = await adapter._record_delivery_obligation(event, "agent:main:telegram:dm:123", _REPLY, adapter, False)
    refused = await adapter.send(chat_id="123", content=_REPLY)
    assert refused.error == "send_path_degraded"
    await adapter._finalize_delivery_obligation(oid, refused, event, adapter)
    return oid


def _state(oid):
    with dl._connect() as conn:
        row = conn.execute("SELECT state FROM delivery_obligations WHERE obligation_id=?", (oid,)).fetchone()
    return row[0] if row else None


async def _drain(adapter):
    tasks = [t for t in adapter._background_tasks if not t.done()]
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)


@pytest.mark.asyncio
async def test_reply_refused_while_degraded_is_redelivered_once_after_in_place_recovery():
    adapter = _adapter()
    runner, _primary = _runner(adapter)
    generation, _ = adapter._begin_polling_generation()  # polling restarted in place: unproven until progress

    oid = await _refuse_while_degraded(adapter)

    assert _state(oid) == "failed"
    assert dl.pending_retries() == []  # reconnect-only: no redelivery timer will ever claim it
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


@pytest.mark.asyncio
async def test_in_place_redelivery_serves_the_owning_profiles_rows(tmp_path, monkeypatch):
    from gateway.run import _profile_runtime_scope

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    launch_home = tmp_path / ".hermes"
    work_home = launch_home / "profiles" / "work"
    work_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    adapter = _adapter(profile="work")
    _, primary = _runner(adapter, profile="work")
    generation, _ = adapter._begin_polling_generation()

    with _profile_runtime_scope(work_home, {}):  # the secondary's turn owns the reply
        oid = await _refuse_while_degraded(adapter)
        assert _state(oid) == "failed"
    assert _state(oid) == "failed"  # one shared ledger in the launch state.db, keyed by adapter_profile

    # The progress edge arrives with no profile bound; the handoff must bind the owner's.
    assert adapter._record_polling_progress(generation) is True
    await _drain(adapter)

    assert adapter._bot.send_message.await_count == 1
    primary.send.assert_not_awaited()
    assert _state(oid) == "delivered"


def _attempts(oid):
    with dl._connect() as conn:
        return conn.execute("SELECT attempts FROM delivery_obligations WHERE obligation_id=?", (oid,)).fetchone()[0]


@pytest.mark.asyncio
async def test_degraded_again_after_the_claim_refunds_the_attempt_and_delivers_once():
    """Polling degrades between the in-place handoff's claim and its send: the adapter refuses locally
    (``send_path_degraded``, no request), so the claim goes back unspent. Flapping as many times as the
    attempts cap must not abandon the reply; the next real recovery delivers it once."""
    adapter = _adapter()
    runner, _primary = _runner(adapter)
    generation, _ = adapter._begin_polling_generation()
    oid = await _refuse_while_degraded(adapter)
    attempts_before = _attempts(oid)

    clear_resume = runner._clear_resume_pending_for_claimed_obligations
    flaps = {"left": dl.MAX_ATTEMPTS}

    async def degrade_after_claim(claimed, **kwargs):
        if flaps["left"]:
            flaps["left"] -= 1
            flaps["generation"], _ = adapter._begin_polling_generation()  # polling dies again
        return await clear_resume(claimed, **kwargs)

    runner._clear_resume_pending_for_claimed_obligations = degrade_after_claim
    for _ in range(dl.MAX_ATTEMPTS):
        assert adapter._record_polling_progress(generation) is True  # in-place recovery edge
        await _drain(adapter)
        generation = flaps["generation"]
        assert _state(oid) == "failed"
        assert _attempts(oid) == attempts_before  # a local refusal spends no attempt
        adapter._bot.send_message.assert_not_awaited()

    assert adapter._record_polling_progress(generation) is True  # recovers and stays up this time
    await _drain(adapter)
    assert _state(oid) == "delivered"
    assert adapter._bot.send_message.await_count == 1
    assert _REPLY in adapter._bot.send_message.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_recovery_edge_during_refund_commit_does_not_strand_the_reply(monkeypatch):
    """R71-2: the refund branch's handback (``_release_unsent_claims(reconnect_only=True)``) commits
    through a worker thread (``release_runtime_claim``). The previous test always lets that commit
    finish before the next recovery edge is recorded, which never exercises the ordering where the
    edge arrives WHILE the commit is still in flight.

    If polling's degraded-to-healthy edge fires in that window, ``_record_polling_progress``'s own
    recovery sweep (``_redeliver_failed_obligations_for_platform``, via ``schedule_in_place_redelivery``)
    claims only ``failed`` rows and the row is still ``attempting`` — the sweep finds nothing, and the
    edge does not fire again for this generation. Without a post-commit recheck, the reply would wait
    for the next restart or a manually requested sweep."""
    import threading

    from gateway import delivery_ledger as dl_module

    adapter = _adapter()
    runner, _primary = _runner(adapter)
    generation, _ = adapter._begin_polling_generation()
    oid = await _refuse_while_degraded(adapter)
    assert _state(oid) == "failed"

    generation, _ = adapter._begin_polling_generation()  # degrades again before the resend is attempted

    entered_commit = threading.Event()
    release_commit = threading.Event()
    real_release_runtime_claim = dl_module.release_runtime_claim

    def _paused_release_runtime_claim(*args, **kwargs):
        entered_commit.set()
        assert release_commit.wait(timeout=5), "test never released the paused handback commit"
        return real_release_runtime_claim(*args, **kwargs)

    monkeypatch.setattr(dl_module, "release_runtime_claim", _paused_release_runtime_claim)

    # This is the same call ``schedule_in_place_redelivery`` makes from a recovery edge: claim the
    # `failed` row and retry its send. The retry finds the adapter degraded again (the flap above) and
    # takes the new refund branch, whose handback pauses at the patched commit below.
    redeliver = asyncio.ensure_future(runner._redeliver_failed_obligations_for_platform(Platform.TELEGRAM))
    await asyncio.wait_for(asyncio.to_thread(entered_commit.wait), timeout=5)
    assert _state(oid) == "attempting"  # the refund's handback has not committed yet

    # The SAME degraded generation recovers while that handback commit is still in flight.
    assert adapter._record_polling_progress(generation) is True
    await _drain(adapter)  # the edge's own sweep runs now, while the row is still invisible to it
    adapter._bot.send_message.assert_not_awaited()

    release_commit.set()  # let the paused handback commit
    await asyncio.wait_for(redeliver, timeout=5)
    # The fix's post-commit recheck tracks its wake-up task on the RUNNER (same bookkeeping as
    # ``_rearm_after_interrupted_send``), not the adapter — drain both when present, so an unfixed
    # ``run_startup.py`` (which never creates ``runner._background_tasks``) fails on the state
    # assertion below rather than on a missing attribute here.
    if getattr(runner, "_background_tasks", None):
        await _drain(runner)
    await _drain(adapter)

    assert _state(oid) == "delivered"
    assert adapter._bot.send_message.await_count == 1
    assert _REPLY in adapter._bot.send_message.await_args.kwargs["text"]


class _Request:
    """PTB request double: ``initialize`` reopens a closed client, ``shutdown`` closes it."""

    def __init__(self, events):
        self.closed, self._events = False, events

    async def initialize(self):
        self.closed = False

    async def shutdown(self):
        self._events.append("request closed")
        self.closed = True


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


@pytest.mark.asyncio
async def test_cancelled_connect_closes_the_abandoned_init_transports_after_it_unwinds():
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

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    app = _app(_shielded_initialize, requests)
    adapter._app, adapter._bot = app, app.bot
    ladder = asyncio.ensure_future(adapter._initialize_app_with_retries(MagicMock()))
    await asyncio.wait_for(entered.wait(), timeout=5)

    ladder.cancel()  # the runner's connect deadline cancels connect() mid-initialize
    with pytest.raises(asyncio.CancelledError):
        await ladder
    asyncio.get_running_loop().call_later(0.2, release.set)
    await asyncio.wait_for(adapter.disconnect(), timeout=10)

    assert all(request.closed for request in requests)
    assert events.index("init exited") < events.index("request closed")
    # The abandoned app never started, so it cannot run a getUpdates loop beside a successor.
    app.start.assert_not_awaited()
    app.updater.start_polling.assert_not_awaited()


@pytest.mark.asyncio
async def test_disconnect_closes_transports_after_the_init_ladder_is_exhausted(monkeypatch):
    requests = (_Request([]), _Request([]))

    async def _unreachable():
        for request in requests:  # Bot.initialize() reopens the shared requests every attempt
            await request.initialize()
        raise OSError("api.telegram.org unreachable")

    builder = MagicMock()
    builder.build.side_effect = lambda: _app(_unreachable, requests)
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock())
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._app = builder.build()
    adapter._bot = adapter._app.bot

    with pytest.raises(OSError):
        await adapter._initialize_app_with_retries(builder)
    assert not any(request.closed for request in requests)  # the last attempt's pools are still open

    await asyncio.wait_for(adapter.disconnect(), timeout=10)

    assert all(request.closed for request in requests)


@pytest.mark.asyncio
async def test_recovery_edge_during_refund_commit_survives_cancellation(monkeypatch):
    """Same race as the test above, but the redelivery task itself is cancelled right after the
    handback commit resumes (not awaited to completion) — exactly what drops a reply if the shielded
    recheck/spawn in ``_finish_degraded_refund`` is removed."""
    import threading

    from gateway import delivery_ledger as dl_module

    adapter = _adapter()
    runner, _primary = _runner(adapter)
    generation, _ = adapter._begin_polling_generation()
    oid = await _refuse_while_degraded(adapter)
    assert _state(oid) == "failed"

    generation, _ = adapter._begin_polling_generation()

    entered_commit = threading.Event()
    release_commit = threading.Event()
    real_release_runtime_claim = dl_module.release_runtime_claim

    def _paused_release_runtime_claim(*args, **kwargs):
        entered_commit.set()
        assert release_commit.wait(timeout=5), "test never released the paused handback commit"
        return real_release_runtime_claim(*args, **kwargs)

    monkeypatch.setattr(dl_module, "release_runtime_claim", _paused_release_runtime_claim)

    redeliver = asyncio.ensure_future(runner._redeliver_failed_obligations_for_platform(Platform.TELEGRAM))
    await asyncio.wait_for(asyncio.to_thread(entered_commit.wait), timeout=5)
    assert adapter._record_polling_progress(generation) is True
    await _drain(adapter)
    adapter._bot.send_message.assert_not_awaited()

    release_commit.set()
    redeliver.cancel()  # cancel as the paused thread resumes, racing the shield's own completion
    try:
        await redeliver
    except asyncio.CancelledError:
        pass

    # The shielded recheck/spawn must still complete even though the task awaiting it was cancelled.
    for _ in range(200):
        if getattr(runner, "_background_tasks", None):
            break
        await asyncio.sleep(0.02)
    if getattr(runner, "_background_tasks", None):
        await _drain(runner)
    await _drain(adapter)

    assert _state(oid) == "delivered", "cancelling the caller must not strand the shielded recovery spawn"
