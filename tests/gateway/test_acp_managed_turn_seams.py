"""The managed /acp turn's gateway seams: crash identity, the follow-up start record, the queued
first-response send, and the declared ``TurnContext.acp_admission``.

Drives the real ``_hmwa_prepare_turn``, ``_handle_message_with_agent``, ``_run_agent_queued_followup``
and the queued send under the temp home of ``test_acp_managed_ingress`` (real runner, session store,
binding and SessionDB; ACP's lane is a real unix socket). Settlement by the adapter's ledgered send is
the ingress seams' own subject; here only what reaches that send is checked.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from unittest.mock import AsyncMock

import pytest

from gateway import acp_managed_ingress as ingress
from gateway import acp_turn_receipts as receipts
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.session_transcript import TranscriptReadError
from gateway.turn_context import TurnContext
from tests.gateway.restart_test_helpers import RestartTestAdapter
from tests.gateway.test_acp_managed_ingress import (
    _ANSWERED, _admitted, _ledger_rows, _ledgered_adapter, _managed_event, _stored_status, _turn_ctx,
)
from tests.gateway.test_acp_managed_ingress import gw as acp_home

gw = acp_home  # the ACP ingress temp home, under the name the imported helpers take

_ORDINARY_ANSWER = {"final_response": "ordinary answer", "messages": []}


def _stub_prepare(gw, monkeypatch, at_history):
    """Stub ``_hmwa_prepare_turn``'s surroundings up to the history load, which calls
    ``at_history(event)`` and then stops the turn as an unreadable transcript."""
    runner = gw.runner
    monkeypatch.setattr(runner, "_hmwa_open_session", AsyncMock(return_value=(False, False)))
    monkeypatch.setattr(runner, "_set_session_env", lambda context: [])
    monkeypatch.setattr(runner, "_clear_session_env", lambda tokens: None)
    monkeypatch.setattr(runner, "_pinned_session_context_prompt", lambda *a, **k: "")
    monkeypatch.setattr(runner, "_hmwa_acquire_turn_lease", AsyncMock())
    events = []

    async def load(session_id):
        at_history(events[-1])
        raise TranscriptReadError(session_id)

    monkeypatch.setattr(runner.async_session_store, "load_transcript", load)
    return events


def _prepare(gw, events, event):
    events.append(event)
    key = gw.entry.session_key
    return asyncio.run(gw.runner._hmwa_prepare_turn(event, gw.source, gw.entry, key, key, 1))


class TestCrashIdentity:
    """An admitted /acp task runs only once its marker names it managed; startup recovery reads that."""

    def test_the_marker_is_bound_before_the_history_loads(self, gw, monkeypatch):
        seen = []
        events = _stub_prepare(gw, monkeypatch, lambda event: seen.append(ingress.crash_left_turn_is_managed(
            gw.runner, gw.entry.session_key, event._gateway_active_turn_token)))
        _prepare(gw, events, _managed_event(gw, _admitted(gw)))
        assert seen == [True]

    def test_an_ordinary_turn_binds_nothing_and_still_loads_its_history(self, gw, monkeypatch):
        bound, seen = [], []
        monkeypatch.setattr(ingress, "bind_turn_marker", lambda *a: bound.append(a) or False)
        events = _stub_prepare(gw, monkeypatch, lambda event: seen.append(ingress.crash_left_turn_is_managed(
            gw.runner, gw.entry.session_key, event._gateway_active_turn_token)))
        _prepare(gw, events, MessageEvent(text="hello", source=gw.source, message_id="56"))
        assert bound == [] and seen == [False]

    def test_a_marker_binding_failure_delivers_the_refusal(self, gw, monkeypatch):
        """Through the real handler: the refusal is the reply, the turn never runs, and the receipt is
        the definite non-run (not a failed turn)."""
        admission = _admitted(gw)
        event = _managed_event(gw, admission)
        events = _stub_prepare(gw, monkeypatch, lambda event: None)
        events.append(event)
        monkeypatch.setattr(gw.runner, "_hmwa_resolve_session",
                            AsyncMock(return_value=(gw.source, gw.entry, gw.entry.session_key)))
        monkeypatch.setattr(ingress, "bind_turn_marker", lambda *a: False)
        run = AsyncMock()
        monkeypatch.setattr(gw.runner, "_run_agent", run)
        reply = asyncio.run(gw.runner._handle_message_with_agent(event, gw.source, gw.entry.session_key, 1))
        assert "turn identity could not be recorded" in reply and run.await_count == 0
        assert event._acp_admission is None
        receipt = receipts.lookup(gw.db, 901)
        assert receipt.status == "ABORTED" and receipt.reason_code == "REFUSED_BEFORE_RUN"


class TestFollowUpStart:
    """Before an ordinary follow-up runs in-band in a managed chain, its start is recorded under the
    managed marker; an unrecordable start still runs the follow-up."""

    def _marked(self, gw):
        admission = _admitted(gw)
        token = gw.runner.session_store.mark_turn_active(gw.entry.session_key)
        assert ingress.bind_turn_marker(admission, token)
        return admission, token

    def _chain(self, gw, monkeypatch, admission, at_run=lambda: None):
        runner = gw.runner
        ran = []

        async def followup(**kwargs):
            ran.append(at_run())
            return dict(_ORDINARY_ANSWER)

        monkeypatch.setattr(runner, "_run_agent", followup)
        monkeypatch.setattr(runner, "_run_agent_deliver_first_response", AsyncMock())
        monkeypatch.setattr(runner, "_refresh_agent_cache_message_count", AsyncMock())
        monkeypatch.setattr(runner, "_delivery_adapter_for", lambda source: None)
        ctx = _turn_ctx(gw, admission, session_id=gw.entry.session_id, history=[],
                        inbound_message_id="55", event_message_id="55")
        result = asyncio.run(runner._run_agent_queued_followup(
            ctx, None, "queued follow-up", None, "managed answer",
            {"final_response": "managed answer", "messages": []}, None))
        return result, ran

    def test_the_start_is_recorded_before_the_follow_up_runs(self, gw, monkeypatch):
        admission, token = self._marked(gw)
        result, ran = self._chain(gw, monkeypatch, admission,
                                  lambda: ingress.followup_start(gw.runner, gw.entry.session_key, token))
        assert result["final_response"] == "ordinary answer"
        assert len(ran) == 1 and isinstance(ran[0], float)

    def test_an_unrecordable_start_still_runs_the_follow_up_in_band(self, gw, monkeypatch, caplog):
        admission, _ = self._marked(gw)
        monkeypatch.setattr(ingress, "mark_followup_start", lambda *a, **k: False)
        with caplog.at_level(logging.WARNING, logger="gateway.run"):
            result, ran = self._chain(gw, monkeypatch, admission)
        assert ran == [None] and result["final_response"] == "ordinary answer"
        assert "follow-up start not recorded" in caplog.text

    def test_an_ordinary_chain_records_no_start(self, gw, monkeypatch):
        calls = []
        monkeypatch.setattr(ingress, "mark_followup_start", lambda *a, **k: calls.append(a) or True)
        result, ran = self._chain(gw, monkeypatch, None)
        assert calls == [] and ran == [None] and result["final_response"] == "ordinary answer"


class _LedgerAdapter(RestartTestAdapter):
    """Records the event each ledgered final send was bracketed against."""

    def __init__(self):
        super().__init__()
        self.ledgered = []

    async def send_final_ledgered(self, event, session_key, text_content, metadata, **kwargs):
        self.ledgered.append((event, text_content))
        return SendResult(success=True, message_id="77"), self


class TestQueuedFirstResponse:
    """The queued first response carries the managed turn's admission into its ledgered send only when
    it is that turn's model answer; that send's row then settles the receipt."""

    def _deliver(self, gw, adapter, **kwargs):
        return asyncio.run(gw.runner._deliver_queued_first_response(
            "managed answer", source=gw.source, adapter=adapter, event_message_id="55",
            deliver_media=False, session_key=gw.entry.session_key, inbound_message_id="55", **kwargs))

    def test_the_ledgered_send_carries_the_admission(self, gw):
        admission, adapter = _admitted(gw), _LedgerAdapter()
        assert self._deliver(gw, adapter, acp_admission=admission) is True
        (event, text), = adapter.ledgered
        assert event._acp_admission is admission and text == "managed answer"
        assert event.ledger_message_id == "55"

    def test_an_ordinary_queued_send_carries_none(self, gw):
        adapter = _LedgerAdapter()
        assert self._deliver(gw, adapter) is True
        (event, _), = adapter.ledgered
        assert event._acp_admission is None

    @pytest.mark.parametrize("result,carried,status", [
        ({"final_response": "the answer", "api_calls": 1, **_ANSWERED}, True, "PENDING"),
        ({"final_response": "context overflow", "failed": True, "api_calls": 1}, False, "ABORTED"),
    ], ids=["answer", "notice"])
    def test_the_queued_site_hands_the_send_only_a_model_answers_admission(self, gw, monkeypatch, result,
                                                                          carried, status):
        admission = _admitted(gw)
        sent = AsyncMock(return_value=True)
        monkeypatch.setattr(gw.runner, "_deliver_queued_first_response", sent)
        monkeypatch.setattr(gw.runner, "_record_room_silence_decision", AsyncMock())
        ctx = _turn_ctx(gw, admission, session_id=gw.entry.session_id, history=[],
                        inbound_message_id="55", event_message_id="55", run_generation=1)
        asyncio.run(gw.runner._run_agent_deliver_first_response(ctx, None, result, result, None))
        assert sent.await_args.kwargs["acp_admission"] is (admission if carried else None)
        assert receipts.lookup(gw.db, 901).status == status  # the answer is settled by its send's row


    @pytest.mark.parametrize("result,content,correlated,status", [
        ({"final_response": "the answer", "api_calls": 1, **_ANSWERED}, "the answer", "901", "COMPLETED"),
        ({"final_response": "context overflow", "failed": True, "api_calls": 1}, "context overflow", None, "ABORTED"),
    ], ids=["answer", "notice"])
    def test_a_managed_answer_sent_first_settles_its_receipt(self, gw, monkeypatch, result, content, correlated,
                                                             status):
        """Through the real queued send and the adapter's ledgered final send: the answer's own row carries
        the update id and, delivered, settles the receipt; a notice goes out uncorrelated and the receipt
        is settled as a failed turn."""
        adapter = _ledgered_adapter(gw)
        monkeypatch.setattr(gw.runner, "_record_room_silence_decision", AsyncMock())
        ctx = _turn_ctx(gw, _admitted(gw), session_id=gw.entry.session_id, history=[],
                        inbound_message_id="55", event_message_id="55", run_generation=1)
        asyncio.run(gw.runner._run_agent_deliver_first_response(ctx, adapter, result, result, None))
        assert adapter.sent == [content] and _ledger_rows() == [(content, correlated)]
        assert _stored_status(gw, 901) == status


class TestDeclaredAdmission:
    def test_turn_context_declares_the_admission(self, gw):
        admission = _admitted(gw)
        assert "acp_admission" in {f.name for f in dataclasses.fields(TurnContext)}
        assert TurnContext().acp_admission is None
        assert TurnContext(source=gw.source, acp_admission=admission).acp_admission is admission
