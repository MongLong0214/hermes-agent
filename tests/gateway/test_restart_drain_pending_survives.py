"""Regression tests for L1-2 — an accepted busy-queue follow-up must survive a turn finishing
while the gateway is draining for a restart.

``request_restart()`` sets ``self._draining = True`` immediately (gateway/run_shutdown.py), while
the currently-running turn is still allowed to finish and deliver its reply. The restart-drain
notice (gateway/run_busy.py's ``_send_busy_drain_notice``) queues the follow-up and promises it a
turn "after it comes back". When the active turn then completes, ``_run_agent_drain_pending``
(gateway/run_turn.py) dequeues that head from the adapter's pending slot and then — because the
gateway is draining — discarded it outright, with no copy left in the slot or in the FIFO
overflow for shutdown's flush/recovery to see. The accepted head vanished rather than surviving
the restart it was promised.

The fix leaves the head queued (handing it back to the slot, or ahead of whatever
``_promote_queued_event`` staged there) instead of discarding it, so shutdown's
``flush_pending_to_file`` / ``flush_overflow_to_file`` + ``recover_pending_to_db`` round trip sees
it exactly once.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from tests.gateway.restart_test_helpers import RestartTestAdapter


def _make_source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id="123456", chat_type="dm", user_id="u1")


def _make_runner_draining() -> GatewayRunner:
    """A bare runner mid-restart-drain — mirrors what request_restart() sets synchronously
    before the active turn finishes (gateway/run_shutdown.py:1609-1617)."""
    runner = object.__new__(GatewayRunner)
    runner._restart_requested = True
    runner._draining = True
    return runner


class TestAcceptedHeadSurvivesTurnCompletingDuringRestartDrain:
    @pytest.mark.asyncio
    async def test_head_is_not_lost_when_slot_had_only_the_head(self):
        """No overflow: the accepted head must come straight back to the (now-empty) slot."""
        runner = _make_runner_draining()
        adapter = RestartTestAdapter()
        source = _make_source()
        sk = build_session_key(source)
        head = MessageEvent(
            text="please also check the deploy", message_type=MessageType.TEXT,
            source=source, message_id="head-1",
        )
        adapter._pending_messages[sk] = head

        pending_event, pending = await GatewayRunner._run_agent_drain_pending(
            runner, {"ok": True}, adapter, source, sk
        )

        # A new turn must NOT start while draining.
        assert pending_event is None
        assert pending is None
        # But the accepted head must not vanish — it must still be visible to shutdown's flush
        # (the adapter's pending slot, same as before the drain attempt).
        assert adapter._pending_messages.get(sk) is head

    @pytest.mark.asyncio
    async def test_head_survives_and_overflow_order_is_preserved(self):
        """Slot held the head, overflow held a second message: restoring the head must not lose
        or duplicate the second message, and must not invert arrival order."""
        runner = _make_runner_draining()
        adapter = RestartTestAdapter()
        source = _make_source()
        sk = build_session_key(source)
        head = MessageEvent(text="first", message_type=MessageType.TEXT, source=source, message_id="head-1")
        second = MessageEvent(text="second", message_type=MessageType.TEXT, source=source, message_id="head-2")
        adapter._pending_messages[sk] = head
        runner._session_state(sk).conversation.queued_events.append(second)

        pending_event, pending = await GatewayRunner._run_agent_drain_pending(
            runner, {"ok": True}, adapter, source, sk
        )

        assert pending_event is None and pending is None
        assert adapter._pending_messages.get(sk) is head
        overflow = runner._session_state(sk).conversation.queued_events
        assert [e.text for e in overflow] == ["second"]

    @pytest.mark.asyncio
    async def test_flush_and_recover_round_trip_sees_head_exactly_once(self, tmp_path, monkeypatch):
        """End-to-end per the finding's suggested test: a turn finishes during request_restart(),
        then shutdown's flush + recovery must see the accepted head exactly once."""
        from gateway import shutdown_flush

        monkeypatch.setattr(shutdown_flush, "_get_flush_dir", lambda: tmp_path)

        runner = _make_runner_draining()
        adapter = RestartTestAdapter()
        source = _make_source()
        sk = build_session_key(source)
        head = MessageEvent(
            text="please also check the deploy", message_type=MessageType.TEXT,
            source=source, message_id="head-1",
        )
        adapter._pending_messages[sk] = head

        await GatewayRunner._run_agent_drain_pending(runner, {"ok": True}, adapter, source, sk)

        # Mirrors gateway/run_shutdown.py's _stop_release_runtime_state flush sequence.
        flushed = shutdown_flush.flush_pending_to_file(dict(adapter._pending_messages), reason="shutdown")
        flushed += shutdown_flush.flush_overflow_to_file(
            {k: list(v) for k, v in dict(runner._queued_events).items() if v}, reason="shutdown",
        )
        assert flushed == 1, "the accepted head must be flushed exactly once, not lost (0) or duplicated"

        mock_db = MagicMock()
        resolver = MagicMock(return_value=("sid-resolved", mock_db))
        recovered = shutdown_flush.recover_pending_to_db(mock_db, session_resolver=resolver)

        assert recovered == 1
        mock_db.append_message.assert_called_once()
        assert mock_db.append_message.call_args.kwargs["content"] == "please also check the deploy"


async def _settle_adapter_tasks(adapter) -> None:
    """Await every adapter background task, including drain tasks spawned while settling."""
    import asyncio

    for _ in range(10):
        tasks = [t for t in adapter._background_tasks if not t.done()]
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)


class TestAdapterKeepsRestoredHeadForShutdownFlush:
    """PR72-R1 — the runner hands the accepted head back to the adapter's pending slot, but the
    adapter's generic completion paths (the in-band drain at the end of
    ``_process_message_background`` and the late-arrival drain in ``_finish_session_task``) treat
    any slot occupant as work to dispatch. During a restart drain that dispatch reaches a runner
    which refuses new work (gateway/run_inbound.py ``_hm_dispatch_quick_and_plugin_commands``) and
    the head is lost one hop later. While the runner drains, the adapter must leave the slot for
    ``cancel_background_tasks``' flush instead."""

    _REFUSAL = "⏳ Gateway is restarting and is not accepting new work right now."

    @pytest.mark.asyncio
    async def test_running_turn_completion_does_not_dispatch_restored_head(self, tmp_path, monkeypatch):
        from gateway import shutdown_flush

        monkeypatch.setattr(shutdown_flush, "_get_flush_dir", lambda: tmp_path)
        runner = _make_runner_draining()
        adapter = RestartTestAdapter()
        adapter.gateway_runner = runner
        source = _make_source()
        sk = build_session_key(source)
        head = MessageEvent(
            text="please also check the deploy", message_type=MessageType.TEXT,
            source=source, message_id="head-1",
        )
        running = MessageEvent(text="the running turn", message_type=MessageType.TEXT,
                               source=source, message_id="run-1")
        handled = []

        async def handler(event):
            handled.append(event)
            if event is running:
                # The active turn finishes during the drain: the runner returns the head to the slot.
                await GatewayRunner._run_agent_drain_pending(runner, {"ok": True}, adapter, source, sk)
                assert adapter._pending_messages.get(sk) is head
                return None
            return self._REFUSAL  # what the draining runner answers for any new dispatch

        adapter.set_message_handler(handler)
        await adapter.handle_message(running)
        adapter._pending_messages[sk] = head  # accepted while the turn ran
        await _settle_adapter_tasks(adapter)

        assert handled == [running], "the restored head was dispatched to a runner that refuses it"
        assert adapter._pending_messages.get(sk) is head
        assert sk not in adapter._active_sessions, "the finished turn must still release its guard"

        # The real adapter shutdown flush, then boot recovery, sees the head exactly once.
        await adapter.cancel_background_tasks()
        mock_db = MagicMock()
        resolver = MagicMock(return_value=("sid-resolved", mock_db))
        assert shutdown_flush.recover_pending_to_db(mock_db, session_resolver=resolver) == 1
        assert mock_db.append_message.call_args.kwargs["content"] == "please also check the deploy"

    @pytest.mark.asyncio
    async def test_late_arrival_cleanup_does_not_dispatch_while_draining(self):
        import asyncio

        runner = _make_runner_draining()
        adapter = RestartTestAdapter()
        adapter.gateway_runner = runner
        source = _make_source()
        sk = build_session_key(source)
        late = MessageEvent(text="late follow-up", message_type=MessageType.TEXT,
                            source=source, message_id="late-1")
        handled = []

        async def handler(event):
            handled.append(event)
            return self._REFUSAL

        adapter.set_message_handler(handler)

        async def owner_task_cleanup():
            # The owner task's finally runs _finish_session_task with a late slot arrival.
            adapter._pending_messages[sk] = late
            adapter._finish_session_task(sk, guard)

        guard = asyncio.Event()
        adapter._active_sessions[sk] = guard
        task = asyncio.create_task(owner_task_cleanup())
        adapter._track_session_task(sk, task)
        await task
        await _settle_adapter_tasks(adapter)

        assert handled == [], "the late arrival was dispatched to a runner that refuses it"
        assert adapter._pending_messages.get(sk) is late
        assert sk not in adapter._active_sessions

    @pytest.mark.asyncio
    async def test_session_command_drain_does_not_dispatch_while_draining(self):
        import asyncio

        runner = _make_runner_draining()
        adapter = RestartTestAdapter()
        adapter.gateway_runner = runner
        source = _make_source()
        sk = build_session_key(source)
        queued = MessageEvent(text="queued follow-up", message_type=MessageType.TEXT,
                              source=source, message_id="q-1")
        handled = []

        async def handler(event):
            handled.append(event)
            return self._REFUSAL

        adapter.set_message_handler(handler)
        command_guard = asyncio.Event()
        adapter._active_sessions[sk] = command_guard
        adapter._pending_messages[sk] = queued

        await adapter._drain_pending_after_session_command(sk, command_guard)
        await _settle_adapter_tasks(adapter)

        assert handled == []
        assert adapter._pending_messages.get(sk) is queued

    @pytest.mark.asyncio
    async def test_not_draining_still_dispatches_queued_follow_up(self):
        """Control: outside a drain the generic handoff must keep running queued follow-ups."""
        runner = _make_runner_draining()
        runner._draining = False
        adapter = RestartTestAdapter()
        adapter.gateway_runner = runner
        source = _make_source()
        sk = build_session_key(source)
        follow = MessageEvent(text="next", message_type=MessageType.TEXT, source=source, message_id="n-1")
        running = MessageEvent(text="now", message_type=MessageType.TEXT, source=source, message_id="r-1")
        handled = []

        async def handler(event):
            handled.append(event)
            if event is running:
                adapter._pending_messages[sk] = follow
            return None

        adapter.set_message_handler(handler)
        await adapter.handle_message(running)
        await _settle_adapter_tasks(adapter)

        assert handled == [running, follow]
        assert sk not in adapter._pending_messages


def _flush_and_recover(tmp_path, monkeypatch, pending: dict) -> MagicMock:
    """Real shutdown serializer + boot recovery for *pending*; return the DB the rows went to."""
    from gateway import shutdown_flush

    monkeypatch.setattr(shutdown_flush, "_get_flush_dir", lambda: tmp_path)
    assert shutdown_flush.flush_pending_to_file(pending, reason="adapter_shutdown") == len(pending)
    mock_db = MagicMock()
    resolver = MagicMock(return_value=("sid-resolved", mock_db))
    assert shutdown_flush.recover_pending_to_db(mock_db, session_resolver=resolver) == len(pending)
    assert not list(tmp_path.glob("*.json")), "every flushed payload must be recovered, none left behind"
    return mock_db


class TestRestoredMediaHeadIsRecoverable:
    """PR72-R3 — a restored head that carries media serialised to ``{"text": ""}``: the voice
    transcript produced by the drain lives only in the event's STT cache, ``media_urls`` was never
    written, and recovery rejects empty text. The flush must keep enough to recover the turn."""

    @pytest.mark.asyncio
    async def test_voice_head_transcribed_during_drain_is_recovered(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock

        runner = _make_runner_draining()
        runner._enrich_message_with_transcription = AsyncMock(
            return_value=('[The user sent a voice message] "Please check the deploy"',
                          ["Please check the deploy"]))
        runner._should_echo_stt_transcripts = lambda: False
        adapter = RestartTestAdapter()
        source = _make_source()
        sk = build_session_key(source)
        voice = MessageEvent(
            text="", message_type=MessageType.VOICE, source=source, message_id="voice-1",
            media_urls=["/cache/audio/voice-1.ogg"], media_types=["audio/ogg"],
        )
        adapter._pending_messages[sk] = voice

        pending_event, pending = await GatewayRunner._run_agent_drain_pending(
            runner, {"ok": True}, adapter, source, sk)

        assert pending_event is None and pending is None
        assert adapter._pending_messages.get(sk) is voice
        runner._enrich_message_with_transcription.assert_awaited_once()
        db = _flush_and_recover(tmp_path, monkeypatch, dict(adapter._pending_messages))
        assert "Please check the deploy" in db.append_message.call_args.kwargs["content"]

    @pytest.mark.parametrize(
        ("message_type", "caption", "url", "mime", "expected"),
        [
            (MessageType.PHOTO, "", "/cache/img/a.jpg", "image/jpeg", ["/cache/img/a.jpg"]),
            (MessageType.PHOTO, "what is this?", "/cache/img/b.jpg", "image/jpeg",
             ["what is this?", "/cache/img/b.jpg"]),
            (MessageType.VIDEO, "", "/cache/video/c.mp4", "video/mp4", ["/cache/video/c.mp4"]),
            (MessageType.AUDIO, "", "/cache/audio/d.mp3", "audio/mpeg", ["/cache/audio/d.mp3"]),
            (MessageType.DOCUMENT, "summarise", "/cache/doc/e.pdf", "application/pdf",
             ["summarise", "/cache/doc/e.pdf"]),
            (MessageType.TEXT, "plain text stays as is", None, None, ["plain text stays as is"]),
        ],
    )
    def test_media_and_captioned_heads_round_trip(self, tmp_path, monkeypatch, message_type, caption,
                                                  url, mime, expected):
        source = _make_source()
        event = MessageEvent(
            text=caption, message_type=message_type, source=source, message_id="m-1",
            media_urls=[url] if url else [], media_types=[mime] if mime else [],
        )

        db = _flush_and_recover(tmp_path, monkeypatch, {build_session_key(source): event})

        content = db.append_message.call_args.kwargs["content"]
        for fragment in expected:
            assert fragment in content
        if not url:
            assert content == caption
