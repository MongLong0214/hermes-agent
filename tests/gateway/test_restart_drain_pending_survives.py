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
