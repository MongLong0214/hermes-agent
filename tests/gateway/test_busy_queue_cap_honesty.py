"""Regression tests for L1-1 — the busy-mode follow-up queue must never tell the operator a
message was queued (or that it interrupted the run) when the cap (``_BUSY_QUEUE_MAX_PENDING`` =
32) actually refused it.

Before this fix, ``_queue_or_replace_pending_event``'s cap check logged a drop and returned
without an acceptance result; every caller (human busy-path ack, restart-drain notice, ``/queue``,
and the ``/steer`` turn-boundary fallback) went on to send its normal "queued"/"interrupting"
acknowledgment regardless. ``/queue`` and ``/steer`` fallback additionally called the uncapped
``_enqueue_fifo`` directly, so they could grow the FIFO past 32 instead of being capped at all.

The fix enforces the cap once, inside ``_enqueue_fifo`` (the one admission point every path now
routes through), which returns ``True``/``False``; every acknowledging caller uses that result.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import build_session_key
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from tests.gateway.test_busy_session_ack import _make_adapter, _make_event, _make_runner


def _fill_to_cap(runner: GatewayRunner, adapter, session_key: str) -> None:
    """Fill the session's FIFO to exactly ``_BUSY_QUEUE_MAX_PENDING`` via the real admission
    point, so each setup call is itself proof the cap wasn't already wrong."""
    for i in range(runner._BUSY_QUEUE_MAX_PENDING):
        filler = MessageEvent(
            text=f"filler-{i}", message_type=MessageType.TEXT,
            source=MagicMock(platform=Platform.TELEGRAM), message_id=f"fill-{i}",
        )
        assert runner._enqueue_fifo(session_key, filler, adapter) is True
    assert runner._queue_depth(session_key, adapter=adapter) == runner._BUSY_QUEUE_MAX_PENDING


class TestHumanBusyFollowupRefusedWhenQueueFull:
    @pytest.mark.asyncio
    async def test_queue_mode_followup_is_honestly_refused_at_cap(self):
        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        adapter = _make_adapter()
        event = _make_event(text="one more, please")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        agent.get_activity_summary.return_value = {"seconds_since_activity": 1.0}
        runner._running_agents[sk] = agent

        _fill_to_cap(runner, adapter, sk)

        result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True  # handled — a reply was sent
        # The cap must still hold: the 33rd follow-up was NOT admitted.
        assert runner._queue_depth(sk, adapter=adapter) == runner._BUSY_QUEUE_MAX_PENDING
        content = adapter._send_with_retry.call_args.kwargs["content"]
        # The honest refusal, not the ordinary "queued" acknowledgment.
        assert "not queued" in content.lower() or "queue is full" in content.lower()
        assert "Queued for the next turn" not in content


class TestRestartDrainNoticeRefusedWhenQueueFull:
    @pytest.mark.asyncio
    async def test_restart_drain_notice_is_honest_at_cap(self):
        runner, _sentinel = _make_runner()
        runner._restart_requested = True
        adapter = _make_adapter()
        event = _make_event(text="are you still there?")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        _fill_to_cap(runner, adapter, sk)

        await runner._send_busy_drain_notice(event, sk, "queue")

        assert runner._queue_depth(sk, adapter=adapter) == runner._BUSY_QUEUE_MAX_PENDING
        content = adapter._send_with_retry.call_args.kwargs["content"]
        assert "queued for the next turn after it comes back" not in content
        assert "not queued" in content.lower() or "queue is full" in content.lower()


class TestQueueCommandBoundedByCap:
    @pytest.mark.asyncio
    async def test_queue_command_refuses_past_cap_instead_of_growing_fifo(self):
        runner, _sentinel = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="/queue one more please")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        _fill_to_cap(runner, adapter, sk)

        reply = await runner._busy_queue_command(event, sk, event.source)

        # Finding L1-1: /queue called the uncapped _enqueue_fifo directly and could grow the FIFO
        # past 32. Admission must stay bounded and say so honestly.
        assert runner._queue_depth(sk, adapter=adapter) == runner._BUSY_QUEUE_MAX_PENDING
        assert "Queued for the next turn" not in reply
        assert "not queued" in reply.lower() or "queue is full" in reply.lower()


class TestSteerFallbackBoundedByCap:
    @pytest.mark.asyncio
    async def test_steer_pending_sentinel_fallback_refuses_past_cap(self):
        runner, _sentinel = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="/steer one more please")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter
        runner._running_agents[sk] = _AGENT_PENDING_SENTINEL

        _fill_to_cap(runner, adapter, sk)

        reply = await runner._busy_steer_command(event, sk, event.source)

        # Finding L1-1: the /steer turn-boundary fallback also called the uncapped _enqueue_fifo.
        assert runner._queue_depth(sk, adapter=adapter) == runner._BUSY_QUEUE_MAX_PENDING
        assert "queued for the next turn" not in reply.lower()
        assert "not queued" in reply.lower() or "queue is full" in reply.lower()
