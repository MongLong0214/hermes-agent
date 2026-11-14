"""The busy follow-up queue refuses honestly at its cap (``_BUSY_QUEUE_MAX_PENDING``).

Ported from the fork's L1-1 regression (tests/gateway/test_busy_queue_cap_honesty.py), cut to the
two run_busy.py callers of ``_queue_or_replace_pending_event``: the restart-drain notice and the
in-process busy handler. Before, the cap check logged a drop and returned nothing, and both callers
went on to claim the message was queued (and the busy handler interrupted the running turn with no
replacement queued).
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import build_session_key
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session_ingress import QUEUE_FULL_REPLY
from tests.gateway.test_busy_session_ack import _make_adapter, _make_event, _make_runner


def _fill_to_cap(runner, adapter, session_key: str) -> None:
    for i in range(runner._BUSY_QUEUE_MAX_PENDING):
        filler = MessageEvent(
            text=f"filler-{i}", message_type=MessageType.TEXT,
            source=MagicMock(platform=Platform.TELEGRAM), message_id=f"fill-{i}",
        )
        runner._enqueue_fifo(session_key, filler, adapter)
    assert runner._queue_depth(session_key, adapter=adapter) == runner._BUSY_QUEUE_MAX_PENDING


def _draining_runner(text: str):
    runner, _sentinel = _make_runner()
    runner._restart_requested = True
    runner._draining = True
    adapter = _make_adapter()
    event = _make_event(text=text)
    sk = build_session_key(event.source)
    runner.adapters[event.source.platform] = adapter
    return runner, adapter, event, sk


def _sent(adapter) -> str:
    return adapter._send_with_retry.call_args.kwargs["content"]


class TestRestartDrainNotice:
    @pytest.mark.asyncio
    async def test_a_message_at_the_cap_is_refused_not_promised(self):
        runner, adapter, event, sk = _draining_runner("are you still there?")
        _fill_to_cap(runner, adapter, sk)

        await runner._send_busy_drain_notice(event, sk, "queue")

        assert runner._queue_depth(sk, adapter=adapter) == runner._BUSY_QUEUE_MAX_PENDING
        assert getattr(event, "_gateway_accepted", False) is not True
        assert _sent(adapter) == QUEUE_FULL_REPLY

    @pytest.mark.asyncio
    async def test_below_the_cap_the_follow_up_is_queued_and_survives_the_drain(self):
        from agent.i18n import t

        runner, adapter, event, sk = _draining_runner("one more, please")

        await runner._send_busy_drain_notice(event, sk, "queue")

        assert adapter._pending_messages[sk] is event and event._gateway_accepted is True
        assert _sent(adapter) == t("gateway.busy.drain_queued", action=runner._status_action_gerund())


class TestInProcessBusyFollowUp:
    @staticmethod
    def _busy_runner():
        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()
        event = _make_event(text="one more, please")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter
        agent = MagicMock()
        agent._active_children = []
        agent.get_activity_summary.return_value = {"seconds_since_activity": 1.0}
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 60
        return runner, adapter, event, sk, agent

    @pytest.mark.asyncio
    async def test_at_the_cap_it_is_refused_and_the_running_turn_is_kept(self):
        runner, adapter, event, sk, agent = self._busy_runner()
        _fill_to_cap(runner, adapter, sk)

        assert await runner._handle_active_session_busy_message(event, sk) is True

        assert runner._queue_depth(sk, adapter=adapter) == runner._BUSY_QUEUE_MAX_PENDING
        assert getattr(event, "_gateway_accepted", False) is not True
        assert _sent(adapter) == QUEUE_FULL_REPLY
        agent.interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_below_the_cap_nothing_changes(self):
        runner, adapter, event, sk, agent = self._busy_runner()

        assert await runner._handle_active_session_busy_message(event, sk) is True

        assert adapter._pending_messages[sk] is event and event._gateway_accepted is True
        assert _sent(adapter) != QUEUE_FULL_REPLY
        agent.interrupt.assert_called_once()
