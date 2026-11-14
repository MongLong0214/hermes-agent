"""U4 H2/H3 — the seam subset of ``/acp`` managed ingress: the adapter, the runner's ``_handle_message``
and startup, over the temp home, lane and helpers of ``test_acp_managed_ingress.py`` (split out of it)."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import acp_managed_ingress as ingress
from gateway import acp_turn_receipts as receipts
from gateway import delivery_ledger
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tests.gateway.test_acp_managed_ingress import (
    _OWN_BOT, _TURN, _WIRE_TURN, _Lane, _admitted, _allowed, _ledger_ready, _ledger_row, _ledger_rows,
    _ledgered_adapter, _managed_state, _marked_turn, _stored_status,
)
from tests.gateway.test_acp_managed_ingress import TestCommandShapes as _CommandShapes  # aliased: not re-collected
from tests.gateway.test_acp_managed_ingress import TestManagedAdmissionGate as _AdmissionGate
from tests.gateway.test_acp_managed_ingress import gw as gw_fixture
from tests.waiting import async_wait_until

gw = gw_fixture


# ── Seam subset ────────────────────────────────────────────────────────────────────────────────
# The adapter, the runner's ``_handle_message`` and startup are real; the ACP lane is the unix
# socket of ``test_acp_managed_ingress.py``, and only the agent run (and, where named, one gate) is a stand-in.


def _seam_managed_event(gw, admission, *, text=None):
    from gateway.platforms.event import MessageEvent
    event = MessageEvent(text=admission.task_text if text is None else text, source=gw.source, message_id="55")
    event._acp_admission = admission
    return event


def _send_final(adapter, gw, event, text):
    return asyncio.run(adapter.send_final_ledgered(event, gw.entry.session_key, text, {}, reply_to="55"))


def _failing_settle(*args, **kwargs):
    raise OSError("disk")


class TestLedgeredFinalSend:
    """The adapter's ``send_final_ledgered``: a managed answer is always ledgered with its update id,
    the delivered row (with the message ids the send occupied) settles the receipt, and no
    settlement-write failure or unrecorded row turns an answered update into a death before answer."""

    def test_the_delivered_final_text_is_the_evidence(self, gw):
        import hashlib

        admission = _admitted(gw, "/acp /deploy now")  # a task text starting with "/" is still ledgered
        adapter = _ledgered_adapter(gw)
        _send_final(adapter, gw, _seam_managed_event(gw, admission), "shaped final")
        receipt = receipts.lookup(gw.db, 901)
        assert adapter.sent == ["shaped final"] and _stored_status(gw, 901) == "COMPLETED"
        assert receipt.evidence_digest == "sha256:" + hashlib.sha256(b"shaped final").hexdigest()
        assert receipt.receipt_id.startswith("hermes-tg:") and receipt.receipt_identity == _TURN
        assert receipt.content == "shaped final" and receipt.delivery["message_ids"] == [77]

    @pytest.mark.parametrize("settlement_write_fails", [False, True])
    def test_get_content_bytes_hash_to_the_evidence_digest(self, gw, monkeypatch, settlement_write_fails):
        """ACP counts a COMPLETED answer as found only when sha256(content) equals evidenceDigest. Both come
        from the one ledgered text, whole, whether the receipt was settled or GET reads the row. A failed
        settlement write is caught by the send: the reply still counts as delivered."""
        import hashlib

        text = "배포 완료 — " + "가나다 ✅ " * 600  # non-ASCII, longer than one Telegram message
        admission = _admitted(gw)
        with monkeypatch.context() as m:
            if settlement_write_fails:
                m.setattr(receipts, "_settle", _failing_settle)
            result, _ = _send_final(_ledgered_adapter(gw), gw, _seam_managed_event(gw, admission), text)
        assert result.success
        body = receipts.lookup(gw.db, 901).to_response()
        assert body["status"] == "COMPLETED" and body["content"] == text
        assert body["evidenceDigest"] == "sha256:" + hashlib.sha256(body["content"].encode("utf-8")).hexdigest()

    def test_delivery_is_the_closed_six_field_shape_acp_parses(self, gw):
        from gateway.platforms.base import SendResult

        admission = _admitted(gw)
        adapter = _ledgered_adapter(gw)

        async def split_send(chat_id, content, reply_to=None, metadata=None):
            return SendResult(success=True, message_id="301", raw_response={"message_ids": ["301", "302", "303"]})

        adapter.send = split_send
        _send_final(adapter, gw, _seam_managed_event(gw, admission), "long answer")
        delivery = receipts.lookup(gw.db, 901).to_response()["delivery"]
        assert set(delivery) == {"obligation_id", "state", "content_digest", "chat_id",
                                 "reply_to_message_id", "message_ids"}
        assert delivery["state"] == "delivered" and delivery["message_ids"] == [301, 302, 303]
        assert delivery["chat_id"] == 100200300 and delivery["reply_to_message_id"] == 55
        assert delivery["content_digest"] == receipts.lookup(gw.db, 901).evidence_digest

    def test_a_delivery_without_message_ids_stays_in_doubt_and_is_never_aborted(self, gw, monkeypatch):
        from gateway.platforms.base import SendResult

        admission = _admitted(gw)
        adapter = _ledgered_adapter(gw)

        async def idless_send(chat_id, content, reply_to=None, metadata=None):
            return SendResult(success=True, message_id=None)

        adapter.send = idless_send
        _send_final(adapter, gw, _seam_managed_event(gw, admission), "answer")
        assert receipts.lookup(gw.db, 901).status == "PENDING"
        monkeypatch.setattr(ingress, "_PROCESS_OWNER", "a-later-process")
        assert ingress.sweep_at_startup(gw.runner) == []
        assert receipts.lookup(gw.db, 901).status == "PENDING"

    def test_a_failed_settlement_write_still_reads_completed_and_is_settled_at_startup(self, gw, monkeypatch, caplog):
        admission = _admitted(gw)
        with monkeypatch.context() as broken, caplog.at_level("WARNING", logger="gateway.platforms.base"):
            broken.setattr(receipts, "_settle", _failing_settle)
            _send_final(_ledgered_adapter(gw), gw, _seam_managed_event(gw, admission), "the answer")
        assert "ACP receipt settlement after delivery failed" in caplog.text  # the receipt is in doubt: say so
        assert _stored_status(gw, 901) == "PENDING"
        assert receipts.lookup(gw.db, 901).status == "COMPLETED"  # GET reads the delivered row
        monkeypatch.setattr(ingress, "_PROCESS_OWNER", "a-later-process")
        assert ingress.sweep_at_startup(gw.runner) == []
        assert _stored_status(gw, 901) == "COMPLETED"

    def test_an_undelivered_answer_is_never_redelivered_and_stays_in_doubt(self, gw, monkeypatch):
        admission = _admitted(gw)
        _send_final(_ledgered_adapter(gw, succeed=False), gw, _seam_managed_event(gw, admission), "the answer")
        [row] = receipts.ledger_answers(gw.db, 901)
        assert row["state"] == "failed"
        assert delivery_ledger.pending_retries() == []  # no timer for the managed row
        monkeypatch.setattr(ingress, "_PROCESS_OWNER", "a-later-process")
        assert ingress.sweep_at_startup(gw.runner) == []  # owed, not dead
        assert receipts.lookup(gw.db, 901).status == "PENDING"

    def test_an_unrecorded_obligation_withholds_the_managed_answer(self, gw, monkeypatch):
        """No ledger row, no managed send. The receipt is a definite non-delivery."""
        admission = _admitted(gw)
        adapter = _ledgered_adapter(gw)
        monkeypatch.setattr(delivery_ledger, "record_obligation",
                            lambda **k: (_ for _ in ()).throw(OSError("disk I/O error")))
        result, _ = _send_final(adapter, gw, _seam_managed_event(gw, admission), "managed answer")
        assert result.success is False and adapter.sent == [ingress.UNRECORDED]
        receipt = receipts.lookup(gw.db, 901)
        assert receipt.status == "ABORTED" and receipt.reason_code == "HERMES_ANSWER_NOT_RECORDED"

    def test_an_unrecorded_ordinary_answer_is_still_sent(self, gw, monkeypatch):
        from gateway.platforms.event import MessageEvent

        adapter = _ledgered_adapter(gw)
        monkeypatch.setattr(delivery_ledger, "record_obligation",
                            lambda **k: (_ for _ in ()).throw(OSError("disk I/O error")))
        _send_final(adapter, gw, MessageEvent(text="hi", source=gw.source, message_id="56"), "ordinary answer")
        assert adapter.sent == ["ordinary answer"]

    def test_a_recording_failure_after_an_uncertain_send_does_not_abort(self, gw, monkeypatch):
        """An earlier send failed after submission, so its delivery is unknown: the later recording failure
        withholds the new send but leaves the receipt in doubt."""
        from gateway.platforms.base import SendResult

        admission = _admitted(gw)
        event = _seam_managed_event(gw, admission)
        adapter = _ledgered_adapter(gw)

        async def uncertain_send(**kwargs):
            adapter.sent.append(kwargs["content"])
            return SendResult(success=False, error="request timed out after submission")

        adapter._send_with_retry = uncertain_send
        _send_final(adapter, gw, event, "managed answer")
        assert receipts.ledger_answers(gw.db, 901)[0]["state"] == "failed"
        monkeypatch.setattr(delivery_ledger, "record_obligation",
                            lambda **kwargs: (_ for _ in ()).throw(OSError("disk I/O error")))
        _send_final(adapter, gw, event, "managed answer")
        assert receipts.lookup(gw.db, 901).status == "PENDING"
        assert adapter.sent[-1] == ingress.UNRECORDED  # the new send itself is still withheld

    def test_a_same_text_handed_off_reply_never_overwrites_the_managed_row(self, gw):
        """Same session, same opening message id, same text: the handed-off reply has its own row."""
        event = _seam_managed_event(gw, _admitted(gw))
        _send_final(_ledgered_adapter(gw), gw, event, "X")
        event._acp_admission = None
        event._acp_handed_off = True
        _send_final(_ledgered_adapter(gw), gw, event, "X")
        assert _ledger_rows() == [("X", None), ("X", "901")]
        assert receipts.lookup(gw.db, 901).status == "COMPLETED"

    def test_a_handed_off_reply_to_a_slash_task_is_still_ledgered(self, gw):
        """The task text began with "/", but the reply answers the chain, not a slash command."""
        event = _seam_managed_event(gw, _admitted(gw, "/acp /deploy now"))
        event._acp_admission = None
        event._acp_handed_off = True
        _send_final(_ledgered_adapter(gw, succeed=False), gw, event, "ordinary follow-up answer")
        assert _ledger_rows() == [("ordinary follow-up answer", None)]


class TestAdapterEntry:
    """The adapter's own busy route, which never passes through the runner's ``_handle_message``."""

    @staticmethod
    def _msg(gw, text, mid):
        from gateway.platforms.event import MessageEvent, MessageType
        return MessageEvent(text=text, message_type=MessageType.TEXT, source=gw.source,
                            message_id=str(mid), platform_update_id=mid)

    def _busy_adapter(self, gw):
        adapter = _ledgered_adapter(gw)
        busy_calls = []

        async def busy(event, key):
            busy_calls.append(event.text)
            return True

        adapter.set_busy_session_handler(busy)
        key = adapter._event_session_key(self._msg(gw, "x", 1))
        adapter._active_sessions[key] = asyncio.Event()
        return adapter, key, busy_calls

    @pytest.mark.asyncio
    async def test_a_task_arriving_while_the_adapter_session_is_busy_is_refused_not_queued(self, gw):
        adapter, key, busy_calls = self._busy_adapter(gw)
        await adapter.handle_message(self._msg(gw, "/acp deploy", 2))
        assert adapter.sent == [ingress.BUSY] and busy_calls == [] and key not in adapter._pending_messages
        assert _managed_state(gw) == (0, 0, 0)
        await adapter.handle_message(self._msg(gw, "ordinary", 3))
        assert busy_calls == ["ordinary"]  # ordinary text keeps its busy route

    @pytest.mark.parametrize("text,expected", [
        ("/acp\ndeploy", "BUSY"), (f"/acp@{_OWN_BOT} deploy", "BUSY"),
        ("/acp", "EMPTY_TASK"), ("/acp  \n", "EMPTY_TASK")], ids=lambda v: repr(v))
    def test_the_adapter_busy_route_refuses_every_shape_without_queueing(self, gw, text, expected):
        adapter, key, busy_calls = self._busy_adapter(gw)
        adapter._current_bot_username = lambda: _OWN_BOT.lower()
        asyncio.run(adapter.handle_message(self._msg(gw, text, 2)))
        assert adapter.sent == [getattr(ingress, expected)] and busy_calls == []
        assert key not in adapter._pending_messages and _managed_state(gw) == (0, 0, 0)

    def test_a_closed_gate_refuses_on_the_adapter_busy_route_too(self, gw):
        _AdmissionGate()._set_gate("false")
        adapter, key, busy_calls = self._busy_adapter(gw)
        asyncio.run(adapter.handle_message(self._msg(gw, "/acp deploy", 2)))
        assert adapter.sent == [ingress.PAUSED] and busy_calls == [] and key not in adapter._pending_messages
        assert _managed_state(gw) == (0, 0, 0)


class TestUpdateIdentity:
    """An /acp message is always its own update. Telegram gives no evidence that tells a client split
    from a separate message, so nothing is ever joined to a task, and a task long enough to have been
    split is refused before ACP is asked (core subset)."""

    def _run(self, gw, chunks):
        adapter = _ledgered_adapter(gw)
        adapter._text_batch_delay_seconds = adapter._text_batch_split_delay_seconds = 0.05
        from gateway.platforms.event import MessageEvent, MessageType

        async def exercise():
            for text, mid in chunks:
                adapter._enqueue_text_event(MessageEvent(text=text, message_type=MessageType.TEXT,
                                                         source=gw.source, message_id=str(mid),
                                                         platform_update_id=mid))
            def settled():
                return (not adapter._pending_text_batches and not adapter._pending_text_batch_tasks
                        and all(task.done() for task in adapter._background_tasks))

            await async_wait_until(settled, timeout=5, what="every text batch dispatched")
            return adapter.dispatched
        return asyncio.run(exercise())

    @pytest.mark.parametrize("head", ["/acp deploy", "/acp " + "x" * 3990, "/acp " + "x" * 4091,
                                      "/acp " + "x" * 3995], ids=["short", "near_limit", "split_length", "exact_4000"])
    def test_an_ordinary_message_is_never_absorbed(self, gw, head):
        dispatched = self._run(gw, [("hello", 10), (head, 11), ("next message", 12)])
        assert sorted(dispatched, key=len) == sorted(["hello", head, "next message"], key=len)

    def test_ordinary_chunks_still_batch(self, gw):
        assert self._run(gw, [("hello", 10), ("there", 11)]) == ["hello\nthere"]


class TestHandleMessageSeam:
    """The seam in the real ``GatewayRunner._handle_message``: the ingress admission and the agent run are
    the only stand-ins; the managed check, busy fast-path, command dispatch, claim and gates are real."""

    @staticmethod
    def _wire(gw, monkeypatch, *, username=None):
        from gateway.platforms.event import MessageEvent

        runs = []

        async def admitted(event):
            return event, event.source, False

        async def run_agent(event, source, key, generation):
            runs.append(event.text)
            return "final reply"

        monkeypatch.setattr(gw.runner, "_hm_admit_event", admitted)
        monkeypatch.setattr(gw.runner, "_handle_message_with_agent", run_agent)
        monkeypatch.setattr(ingress, "socket_path", lambda: gw.sock)
        monkeypatch.setattr(ingress, "read_secret", lambda: "lane-secret")
        if username is not None:
            receiver = SimpleNamespace(_current_bot_username=lambda: username.lower())
            monkeypatch.setattr(gw.runner, "_intake_adapter_for", lambda source: receiver)

        def event(text, *, source=None, update_id=901, message_id="55"):
            return MessageEvent(text=text, source=source or gw.source, message_id=message_id,
                                platform_update_id=update_id)
        return runs, event

    @staticmethod
    def _send(gw, *messages, answer=None):
        async def exercise():
            async with _Lane(gw.sock, answer or _allowed(gw)) as lane:
                replies = [await gw.runner._handle_message(m) for m in messages]
            return replies, lane.envelopes
        return asyncio.run(exercise())

    def test_an_admitted_task_runs_on_its_text_and_ordinary_chat_never_asks_acp(self, gw, monkeypatch):
        runs, event = self._wire(gw, monkeypatch)
        (managed, ordinary), envelopes = self._send(gw, event("/acp deploy the fix"), event("how are you"))
        assert managed == ordinary == "final reply"
        assert runs == ["deploy the fix", "how are you"]
        assert len(envelopes) == 1  # only the /acp message went to ACP
        assert receipts.lookup(gw.db, 901).status == "PENDING"  # settled later by the delivery

    def test_a_redelivered_update_runs_once(self, gw, monkeypatch):
        """The same Telegram update handed over twice is admitted once: the second is the duplicate."""
        runs, event = self._wire(gw, monkeypatch)
        (first, second), envelopes = self._send(gw, event("/acp deploy the fix"), event("/acp deploy the fix"))
        assert first == "final reply" and second == ingress.DUPLICATE.format(status="PENDING")
        assert runs == ["deploy the fix"] and len(envelopes) == 2

    def test_an_admitted_answer_settles_its_receipt_on_delivery(self, gw, monkeypatch):
        _, event = self._wire(gw, monkeypatch)
        managed = event("/acp deploy the fix")
        [reply], _ = self._send(gw, managed)
        _send_final(_ledgered_adapter(gw), gw, managed, reply)
        assert _stored_status(gw, 901) == "COMPLETED" and _ledger_rows() == [("final reply", "901")]

    def test_a_denied_task_never_reaches_the_agent(self, gw, monkeypatch):
        runs, event = self._wire(gw, monkeypatch)
        [reply], _ = self._send(gw, event("/acp deploy"),
                                answer=lambda e: {"allowed": False, "reasonCode": "NOPE", "message": "x"})
        assert "NOPE" in reply and runs == []

    def test_a_task_arriving_during_a_busy_turn_is_refused_before_admission(self, gw, monkeypatch):
        """A queued /acp message would later drain as an ordinary turn with no admission."""
        runs, event = self._wire(gw, monkeypatch)
        queued = []

        async def queue(*args, **kwargs):
            queued.append(args)

        monkeypatch.setattr(gw.runner, "_hm_handle_running_session_message", queue)
        monkeypatch.setattr(gw.runner, "_is_session_running", lambda key: True)
        monkeypatch.setattr(gw.runner, "_hm_evict_reaped_agent", lambda key: None)
        [reply], envelopes = self._send(gw, event("/acp deploy"))
        assert reply == ingress.BUSY and queued == [] and runs == [] and envelopes == []
        assert _managed_state(gw) == (0, 0, 0)

    def test_an_external_drain_refuses_before_admission(self, gw, monkeypatch):
        runs, event = self._wire(gw, monkeypatch)
        gw.runner._external_drain_active = True
        [reply], envelopes = self._send(gw, event("/acp deploy"))
        assert runs == [] and envelopes == [] and reply not in (ingress.BUSY, "final reply")
        assert _managed_state(gw) == (0, 0, 0)

    def test_a_task_never_answers_a_pending_prompt(self, gw, monkeypatch):
        runs, event = self._wire(gw, monkeypatch)
        monkeypatch.setattr(gw.runner, "_hm_pending_reply_intercepts",
                            AsyncMock(return_value="answered the pending prompt"))
        (managed, ordinary), _ = self._send(gw, event("/acp deploy"), event("yes"))
        assert managed == "final reply" and runs == ["deploy"]
        assert ordinary == "answered the pending prompt"  # ordinary text still answers it

    def test_an_internal_event_is_never_managed(self, gw, monkeypatch):
        _, event = self._wire(gw, monkeypatch)

        async def internal(ev):
            return ev, ev.source, True

        monkeypatch.setattr(gw.runner, "_hm_admit_event", internal)
        message = event("/acp deploy")
        _, envelopes = self._send(gw, message)
        assert envelopes == [] and getattr(message, "_acp_admission", None) is None
        assert _managed_state(gw) == (0, 0, 0)

    def test_an_admitted_task_is_never_parked_behind_an_orphan(self, gw, monkeypatch):
        """The FIFO orphan rescue runs the oldest orphan as this turn: never for an admitted task."""
        runs, event = self._wire(gw, monkeypatch)
        parked = []
        monkeypatch.setattr(gw.runner, "_delivery_adapter_for", lambda source: SimpleNamespace())
        monkeypatch.setattr(gw.runner, "_rescue_orphaned_overflow",
                            lambda key, adapter: event("orphan", message_id="50"))
        monkeypatch.setattr(gw.runner, "_enqueue_fifo", lambda key, ev, adapter: parked.append(ev.text))
        self._send(gw, event("/acp deploy"), event("hello", message_id="56"))
        assert runs == ["deploy", "orphan"] and parked == ["hello"]

    def test_a_plugin_that_consumes_an_admitted_task_records_its_non_run(self, gw, monkeypatch):
        runs, event = self._wire(gw, monkeypatch)
        monkeypatch.setattr(gw.runner, "_hm_post_admission_consume",
                            AsyncMock(return_value=(True, "taken by a plugin")))
        managed = event("/acp deploy")
        [reply], _ = self._send(gw, managed, answer=_allowed(gw, turn=_WIRE_TURN))
        assert reply == "taken by a plugin" and runs == [] and managed._acp_admission is None
        _send_final(_ledgered_adapter(gw), gw, managed, reply)
        receipt = receipts.served(gw.db, 901)
        assert receipt.status == "ABORTED" and receipt.reason_code == "REFUSED_BEFORE_RUN"

    def test_the_session_limit_records_the_non_run_and_its_notice_never_certifies(self, gw, monkeypatch):
        runs, event = self._wire(gw, monkeypatch)
        monkeypatch.setattr(gw.runner, "_claim_active_session_slot", lambda key, source: (None, "too many sessions"))
        managed = event("/acp /deploy now")  # the event now carries a task text that starts with "/"
        [reply], _ = self._send(gw, managed, answer=_allowed(gw, turn=_WIRE_TURN))
        assert reply == "too many sessions" and runs == []
        _send_final(_ledgered_adapter(gw), gw, managed, reply)
        receipt = receipts.served(gw.db, 901)  # a delivered ACP row here would make this unprovable
        assert receipt.status == "ABORTED" and receipt.reason_code == "REFUSED_BEFORE_RUN"
        assert _ledger_rows() == [("too many sessions", None)]  # still an ordinary, ledgered reply

    def test_a_lease_timeout_is_a_definite_non_run(self, gw, monkeypatch):
        from gateway.turn_lease import TurnLeaseTimeoutError

        _, event = self._wire(gw, monkeypatch)

        async def lease_timeout(*a, **k):
            raise TurnLeaseTimeoutError(gw.entry.session_id, owner_key="k", generation=1, wait_seconds=1)

        monkeypatch.setattr(gw.runner, "_handle_message_with_agent", lease_timeout)
        managed = event("/acp deploy")
        [reply], _ = self._send(gw, managed)
        assert "not processed" in reply and managed._acp_admission is None
        receipt = receipts.lookup(gw.db, 901)
        assert receipt.status == "ABORTED" and receipt.reason_code == "REFUSED_BEFORE_RUN"


class TestAdmissionGateAtTheSeams:
    """``gateway.acp_managed_admission`` through ``_handle_message``: closed, an /acp message is refused as
    an ordinary reply with no managed write. Ordinary messages never read the gate."""

    def test_closed_gate_refuses_an_acp_task_before_any_managed_write(self, gw, monkeypatch):
        _AdmissionGate()._set_gate("false")
        runs, event = TestHandleMessageSeam._wire(gw, monkeypatch)
        managed = event("/acp deploy the fix")
        [reply], envelopes = TestHandleMessageSeam._send(gw, managed)
        assert reply == ingress.PAUSED and envelopes == [] and runs == []
        assert getattr(managed, "_acp_admission", None) is None
        adapter = _ledgered_adapter(gw)
        _send_final(adapter, gw, managed, reply)
        assert adapter.sent == [ingress.PAUSED] and _managed_state(gw) == (0, 0, 0)
        assert receipts.lookup(gw.db, 901).status == "NEVER_FOUND"

    @pytest.mark.parametrize("gate", [None, "true", "false"])
    def test_an_ordinary_message_is_processed_the_same_whatever_the_gate(self, gw, monkeypatch, gate):
        _AdmissionGate()._set_gate(gate)
        runs, event = TestHandleMessageSeam._wire(gw, monkeypatch)
        ordinary = event("how are you")
        [reply], envelopes = TestHandleMessageSeam._send(gw, ordinary)
        assert reply == "final reply" and runs == ["how are you"] and envelopes == []
        _send_final(_ledgered_adapter(gw), gw, ordinary, reply)
        assert _ledger_rows() == [("final reply", None)] and _managed_state(gw) == (0, 0, 0)

    def test_the_gate_is_read_per_message(self, gw, monkeypatch):
        runs, event = TestHandleMessageSeam._wire(gw, monkeypatch)
        gate = _AdmissionGate()

        async def exercise():
            replies = []
            async with _Lane(gw.sock, _allowed(gw)) as lane:
                for value, update_id, mid in (("false", 901, "55"), ("true", 902, "56"), ("off", 903, "57")):
                    gate._set_gate(value)
                    replies.append(await gw.runner._handle_message(
                        event(f"/acp task {update_id}", update_id=update_id, message_id=mid)))
            return replies, lane.envelopes

        replies, envelopes = asyncio.run(exercise())
        assert replies == [ingress.PAUSED, "final reply", ingress.PAUSED]
        assert runs == ["task 902"] and [e["update"]["update_id"] for e in envelopes] == [902]


class TestCommandShapesAtTheSeams:
    """On the bound chat from the owner, every ``/acp`` command shape is the managed path's alone: admitted
    on its task text, or refused (empty task, closed gate, busy) as an ordinary reply — never the generic
    slash dispatch's "Unknown command" (production 2026-10-09)."""

    ADMITTED = _CommandShapes.ADMITTED
    EMPTY = _CommandShapes.EMPTY
    NOT_OURS = _CommandShapes.NOT_OURS

    @staticmethod
    def _wire(gw, monkeypatch):
        return TestHandleMessageSeam._wire(gw, monkeypatch, username=_OWN_BOT)

    @pytest.mark.parametrize("text,task", ADMITTED, ids=lambda v: repr(v))
    def test_every_acp_command_shape_is_admitted_on_its_task_text(self, gw, monkeypatch, text, task):
        runs, event = self._wire(gw, monkeypatch)
        [reply], [envelope] = TestHandleMessageSeam._send(gw, event(text))
        assert reply == "final reply" and runs == [task]
        assert envelope["update"]["message"]["text"] == text  # ACP is shown the message as sent
        assert receipts.lookup(gw.db, 901).status == "PENDING"

    @pytest.mark.parametrize("text", EMPTY, ids=lambda v: repr(v))
    def test_an_acp_command_with_no_task_is_refused_without_any_managed_write(self, gw, monkeypatch, text):
        runs, event = self._wire(gw, monkeypatch)
        message = event(text)
        [reply], envelopes = TestHandleMessageSeam._send(gw, message)
        assert reply == ingress.EMPTY_TASK and envelopes == [] and runs == []
        assert getattr(message, "_acp_admission", None) is None and _managed_state(gw) == (0, 0, 0)

    @pytest.mark.parametrize("text", NOT_OURS, ids=lambda v: repr(v))
    def test_other_commands_and_other_bots_keep_todays_path(self, gw, monkeypatch, text):
        _, event = self._wire(gw, monkeypatch)
        message = event(text)
        [reply], envelopes = TestHandleMessageSeam._send(gw, message)
        assert envelopes == [] and _managed_state(gw) == (0, 0, 0)
        assert reply not in (ingress.EMPTY_TASK, ingress.PAUSED, ingress.BUSY)
        assert getattr(message, "_acp_admission", None) is None

    @pytest.mark.parametrize("who", ["other_chat", "non_owner"])
    @pytest.mark.parametrize("text", ["/acp deploy the fix", "/acp"], ids=lambda v: repr(v))
    def test_off_the_bound_chat_or_owner_nothing_changes(self, gw, monkeypatch, who, text):
        _, event = self._wire(gw, monkeypatch)
        source = SessionSource(platform=Platform.TELEGRAM, chat_type="dm",
                               chat_id="42" if who == "other_chat" else "100200300", user_id="42")
        message = event(text, source=source)
        [reply], envelopes = TestHandleMessageSeam._send(gw, message)
        assert envelopes == [] and _managed_state(gw) == (0, 0, 0)
        assert reply not in (ingress.EMPTY_TASK, ingress.PAUSED, ingress.BUSY)

    @pytest.mark.parametrize("text", [s for s, _ in ADMITTED] + EMPTY, ids=lambda v: repr(v))
    def test_a_closed_gate_refuses_every_shape_before_any_managed_write(self, gw, monkeypatch, text):
        _AdmissionGate()._set_gate("false")
        runs, event = self._wire(gw, monkeypatch)
        [reply], envelopes = TestHandleMessageSeam._send(gw, event(text))
        assert reply == ingress.PAUSED and envelopes == [] and runs == []
        assert _managed_state(gw) == (0, 0, 0)

    @pytest.mark.parametrize("text", ["/acp\ndeploy", "/acp"], ids=lambda v: repr(v))
    def test_the_runner_busy_route_refuses_without_queueing(self, gw, monkeypatch, text):
        runs, event = self._wire(gw, monkeypatch)
        queued = []

        async def queue(*args, **kwargs):
            queued.append(args)

        monkeypatch.setattr(gw.runner, "_hm_handle_running_session_message", queue)
        monkeypatch.setattr(gw.runner, "_is_session_running", lambda key: True)
        monkeypatch.setattr(gw.runner, "_hm_evict_reaped_agent", lambda key: None)
        reply = asyncio.run(gw.runner._handle_message(event(text)))
        assert queued == [] and runs == []
        assert reply == (ingress.EMPTY_TASK if text == "/acp" else ingress.BUSY)


class TestTelegramCommandMention:
    """``/acp@<this bot> <task>`` as the Telegram adapter hands it over: its mention cleanup keeps the
    space after the command, so the task is admitted instead of becoming the command ``/acptask``."""

    def test_the_adapter_cleaned_mention_form_is_admitted(self, gw, monkeypatch):
        from plugins.platforms.telegram.adapter import TelegramAdapter
        from plugins.platforms.telegram.telegram_context import group_trigger_text

        telegram = object.__new__(TelegramAdapter)
        telegram._bot = SimpleNamespace(id=999, username=_OWN_BOT)
        dm = SimpleNamespace(chat=SimpleNamespace(type="private"))
        text = group_trigger_text(telegram, dm, f"/acp@{_OWN_BOT} deploy the fix")
        assert text == "/acp deploy the fix"
        runs, event = TestCommandShapesAtTheSeams._wire(gw, monkeypatch)
        [reply], envelopes = TestHandleMessageSeam._send(gw, event(text))
        assert reply == "final reply" and runs == ["deploy the fix"] and len(envelopes) == 1


class TestFirstAfterCleanRestart:
    """A clean restart loads the session index lazily. An /acp task arriving before any ordinary inbound
    must still find the binding's existing session for its lineage check."""

    def test_an_acp_task_first_after_a_clean_restart_is_admitted(self, gw, monkeypatch):
        from gateway.platforms.event import MessageEvent

        gw.runner.session_store.close_all_db_handles()
        restarted = GatewayRunner(GatewayConfig(sessions_dir=gw.runner.session_store.sessions_dir))
        restarted.config.canonical_surface_bindings = gw.runner.config.canonical_surface_bindings
        assert restarted.session_store._loaded is False
        gw_restarted = SimpleNamespace(**{**vars(gw), "runner": restarted})
        runs, _ = TestHandleMessageSeam._wire(gw_restarted, monkeypatch)
        event = MessageEvent(text="/acp deploy the fix", source=gw.source, message_id="55", platform_update_id=901)
        try:
            [reply], envelopes = TestHandleMessageSeam._send(gw_restarted, event)
            assert reply == "final reply" and runs == ["deploy the fix"] and len(envelopes) == 1
            receipt = receipts.lookup(restarted.session_store._db_for_key(gw.entry.session_key), 901)
            assert receipt.status == "PENDING"
        finally:
            restarted.session_store.close_all_db_handles()


class TestCrashRecovery:
    """A crash-left managed turn is recognised by its bound active-turn marker, never by message time: it
    is neither resumed nor ledgered from its transcript. An ordinary follow-up its chain had started owes
    exactly what an ordinary turn begun at the follow-up's recorded start owes."""

    @staticmethod
    def _recover(gw):
        recovered = asyncio.run(gw.runner._recover_unclean_sessions())
        return recovered, [tuple(r) for r in _ledger_rows()]

    @staticmethod
    def _start(gw, admission, store):
        token = store._entries[gw.entry.session_key].active_turn_token
        assert ingress.mark_followup_start(admission, token, time.time())

    @pytest.mark.parametrize("has_answer", [False, True])
    def test_a_bound_managed_turn_is_neither_resumed_nor_ledgered(self, gw, has_answer):
        _, store = _marked_turn(gw, answer=has_answer)
        recovered, rows = self._recover(gw)
        entry = store._entries[gw.entry.session_key]
        assert recovered == (0, 0) and rows == [] and not entry.resume_pending and not entry.active_turn_token

    def test_an_unbound_turn_on_the_bound_session_keeps_ordinary_recovery(self, gw):
        _, store = _marked_turn(gw, bind=False)
        assert self._recover(gw) == ((1, 0), [])
        assert store._entries[gw.entry.session_key].resume_pending

    def test_a_follow_up_reply_after_its_start_is_owed_once(self, gw):
        admission, store = _marked_turn(gw, answer=True)
        self._start(gw, admission, store)
        store.append_to_transcript(gw.entry.session_id, {"role": "user", "content": "ordinary follow-up"})
        store.append_to_transcript(gw.entry.session_id, {"role": "assistant", "content": "ordinary answer"})
        assert self._recover(gw) == ((0, 1), [("ordinary answer", None)])
        assert not store._entries[gw.entry.session_key].resume_pending

    def test_no_recorded_start_owes_nothing(self, gw):
        _, store = _marked_turn(gw, answer=True)
        store.append_to_transcript(gw.entry.session_id, {"role": "user", "content": "later"})
        store.append_to_transcript(gw.entry.session_id, {"role": "assistant", "content": "later reply"})
        assert self._recover(gw) == ((0, 0), [])

    def test_only_the_managed_answer_before_the_start_owes_nothing(self, gw):
        admission, store = _marked_turn(gw, answer=True)
        self._start(gw, admission, store)
        assert self._recover(gw) == ((0, 0), [])

    def test_an_unanswered_follow_up_owes_nothing_and_is_not_resumed(self, gw):
        admission, store = _marked_turn(gw, answer=True)
        self._start(gw, admission, store)
        store.append_to_transcript(gw.entry.session_id, {"role": "user", "content": "ordinary follow-up"})
        assert self._recover(gw) == ((0, 0), [])
        assert not store._entries[gw.entry.session_key].resume_pending

    def test_a_same_text_managed_answer_does_not_hide_the_follow_up_reply(self, gw):
        admission, store = _marked_turn(gw, answer=True)
        delivery_ledger.record_obligation(obligation_id="ob-managed", session_key=gw.entry.session_key,
                                          platform="telegram", chat_id="100200300", thread_id=None,
                                          content="done", acp_update_id="901")
        delivery_ledger.mark_delivered("ob-managed", message_ids=["7"])
        self._start(gw, admission, store)
        store.append_to_transcript(gw.entry.session_id, {"role": "user", "content": "and the other?"})
        store.append_to_transcript(gw.entry.session_id, {"role": "assistant", "content": "done"})
        recovered, rows = self._recover(gw)
        assert recovered == (0, 1) and ("done", None) in rows

    def test_a_failed_managed_marker_cleanup_never_ledgers_its_answer(self, gw, monkeypatch):
        _marked_turn(gw, answer=True)

        async def unavailable(*args):
            raise OSError("temporary failure persisting marker cleanup")

        monkeypatch.setattr(gw.runner.async_session_store, "clear_turn_active", unavailable)
        assert self._recover(gw)[1] == []

    def test_a_turn_whose_identity_cannot_be_read_is_neither_ledgered_nor_cleared(self, gw, monkeypatch):
        """Held for the next start: the managed path never clears it, and its transcript reply is not owed."""
        _marked_turn(gw, answer=True)
        monkeypatch.setattr(ingress, "crash_left_turn_is_managed", lambda runner, key, token: None)
        cleared = []
        real_clear = gw.runner.async_session_store.clear_turn_active

        async def spy(key, token):
            cleared.append(key)
            return await real_clear(key, token)

        monkeypatch.setattr(gw.runner.async_session_store, "clear_turn_active", spy)
        assert self._recover(gw)[1] == [] and cleared == []

    def test_an_unreadable_follow_up_is_not_ledgered_and_recovery_goes_on(self, gw, monkeypatch):
        admission, store = _marked_turn(gw, answer=True)
        self._start(gw, admission, store)
        store.append_to_transcript(gw.entry.session_id, {"role": "user", "content": "ordinary follow-up"})
        store.append_to_transcript(gw.entry.session_id, {"role": "assistant", "content": "ordinary answer"})

        async def unreadable(*a, **k):
            raise OSError("disk")

        monkeypatch.setattr(gw.runner.async_session_store, "load_transcript", unreadable)
        assert self._recover(gw)[1] == []

    def test_an_earlier_turns_reply_is_never_adopted(self, gw):
        store = gw.runner.session_store
        store.append_to_transcript(gw.entry.session_id, {"role": "user", "content": "earlier question"})
        store.append_to_transcript(gw.entry.session_id, {"role": "assistant", "content": "earlier answer"})
        assert ingress.bind_turn_marker(_admitted(gw), store.mark_turn_active(gw.entry.session_key))
        assert self._recover(gw) == ((0, 0), [])

    def test_durable_managed_identity_survives_binding_removal(self, gw):
        _marked_turn(gw, answer=True)
        gw.runner.config.canonical_surface_bindings = {}
        recovered, rows = self._recover(gw)
        adapter = _ledgered_adapter(gw)
        gw.runner._obligation_adapter = AsyncMock(return_value=adapter)
        for row in delivery_ledger.sweep_recoverable():
            asyncio.run(gw.runner._redeliver_claimed_row(row, delivery_ledger.RECOVERED_MARKER))
        assert recovered == (0, 0) and rows == [] and adapter.sent == []


_DEAD_OWNER = (999_999_999, 1)


def _boot_runner(gw, adapter):
    """The fixture's runner as startup sees it: running, with ``adapter`` as its Telegram transport."""
    gw.runner.adapters = {Platform.TELEGRAM: adapter}
    gw.runner._running = True
    return gw.runner


def _rows_by_id():
    with delivery_ledger._connect() as conn:
        return {r[0]: (r[1], r[2], r[3]) for r in conn.execute(
            "SELECT obligation_id, state, attempts, last_error FROM delivery_obligations")}


class TestBootReconciliation:
    """At boot a PENDING receipt a dead process claimed is reconciled against the ledger before any
    redelivery: a delivered answer settles COMPLETED, an owed one stays in doubt and is never resent, and
    only an update with no ledgered answer is aborted. Ordinary rows are redelivered as before."""

    def test_a_pending_receipt_at_boot_is_reconciled_not_resent(self, gw, monkeypatch):
        _ledger_ready()
        for update_id in (901, 902, 903):
            _admitted(gw, update_id=update_id, turn=_WIRE_TURN)
        _ledger_row(gw, "acp-delivered", 901, "delivered", content="managed answer 901", owner=_DEAD_OWNER)
        delivery_ledger._update_state("acp-delivered", "delivered", delivered_message_ids='["77"]')
        _ledger_row(gw, "acp-owed", 902, "failed", content="managed answer 902", age=600, owner=_DEAD_OWNER)
        _ledger_row(gw, "ob-ordinary", None, "failed", content="ordinary answer", age=600, owner=_DEAD_OWNER)
        monkeypatch.setattr(ingress, "_PROCESS_OWNER", "a-later-process")
        adapter = _ledgered_adapter(gw)
        runner = _boot_runner(gw, adapter)
        asyncio.run(runner._await_startup_boot_sends(planned_restart_notification_pending=False))
        assert [_stored_status(gw, u) for u in (901, 902, 903)] == ["COMPLETED", "PENDING", "ABORTED"]
        assert receipts.lookup(gw.db, 903).reason_code == "HERMES_PROCESS_DIED_BEFORE_ANSWER"
        assert not any("managed answer" in text for text in adapter.sent)
        assert [text for text in adapter.sent if text.endswith("ordinary answer")] == adapter.sent != []
        assert _rows_by_id()["acp-owed"][0] == "failed"

    def test_a_failed_sweep_never_stops_the_boot_sends(self, gw, monkeypatch):
        _ledger_ready()
        _ledger_row(gw, "ob-ordinary", None, "failed", content="ordinary answer", age=600, owner=_DEAD_OWNER)
        monkeypatch.setattr(ingress, "sweep_at_startup", lambda runner: (_ for _ in ()).throw(OSError("disk")))
        adapter = _ledgered_adapter(gw)
        asyncio.run(_boot_runner(gw, adapter)._await_startup_boot_sends(planned_restart_notification_pending=False))
        assert len(adapter.sent) == 1 and adapter.sent[0].endswith("ordinary answer")


class TestRedeliveredAnswer:
    """A redelivered /acp answer settles its receipt from the row, with the message ids the send occupied.
    A settlement failure leaves the receipt in doubt, never the redelivery's own result."""

    @staticmethod
    def _claimed_acp_row(gw):
        _admitted(gw)
        _ledger_row(gw, "acp-1", 901, "failed", content="managed answer")
        delivery_ledger.mark_attempting("acp-1")
        [row] = [r for r in receipts.ledger_answers(gw.db, 901)]
        return {"obligation_id": row["obligation_id"], "platform": "telegram", "chat_id": "100200300",
                "thread_id": None, "content": "managed answer", "attempts": None}

    def test_a_redelivered_answer_settles_its_receipt(self, gw):
        row = self._claimed_acp_row(gw)
        gw.runner._obligation_adapter = AsyncMock(return_value=_ledgered_adapter(gw))
        assert asyncio.run(gw.runner._redeliver_claimed_row(row, delivery_ledger.RECOVERED_MARKER))
        assert _stored_status(gw, 901) == "COMPLETED"
        assert receipts.lookup(gw.db, 901).delivery["message_ids"] == [77]

    def test_a_failed_settlement_keeps_the_redelivery_delivered(self, gw, monkeypatch):
        row = self._claimed_acp_row(gw)
        gw.runner._obligation_adapter = AsyncMock(return_value=_ledgered_adapter(gw))
        monkeypatch.setattr(receipts, "_settle", _failing_settle)
        assert asyncio.run(gw.runner._redeliver_claimed_row(row, delivery_ledger.RECOVERED_MARKER))
        assert _stored_status(gw, 901) == "PENDING" and receipts.lookup(gw.db, 901).status == "COMPLETED"


class TestBootRedeliveryShutdown:
    """Shutdown during a boot redelivery batch: nothing more goes out, and the claims not yet sent go back
    without spending an attempt. One claimed from ``pending`` goes back to ``pending``: it provably never
    started, so the next boot sends it plainly."""

    @staticmethod
    def _record(names_and_states):
        for oid, state in names_and_states:
            delivery_ledger.record_obligation(obligation_id=oid, session_key="agent:main:telegram:dm:100200300",
                                              platform="telegram", chat_id="100200300", thread_id=None,
                                              content=f"answer {oid}")
            conn = delivery_ledger._connect()
            try:
                conn.execute("UPDATE delivery_obligations SET state = ?, attempts = ?, updated_at = ?,"
                             " owner_pid = ?, owner_started_at = ? WHERE obligation_id = ?",
                             (state, 1 if state == "failed" else 0, time.time() - 600, *_DEAD_OWNER, oid))
                conn.commit()
            finally:
                conn.close()

    def test_shutdown_hands_back_the_unsent_claims_unspent(self, gw):
        _ledger_ready()
        self._record([("ob-1", "pending"), ("ob-2", "pending"), ("ob-3", "failed")])
        adapter = _ledgered_adapter(gw)
        runner = _boot_runner(gw, adapter)
        real_send = adapter.send

        async def send_then_shut_down(chat_id, content, reply_to=None, metadata=None):
            runner._running = False
            return await real_send(chat_id, content, reply_to=reply_to, metadata=metadata)

        adapter.send = send_then_shut_down
        asyncio.run(runner._redeliver_pending_obligations())
        assert len(adapter.sent) == 1  # nothing more goes out once shutdown has begun
        rows = _rows_by_id()
        sent = next(oid for oid in rows if adapter.sent[0].endswith(f"answer {oid}"))
        assert rows[sent][0] == "delivered"
        unsent = {oid: rows[oid] for oid in rows if oid != sent}
        assert unsent == {oid: ("pending", 0, None) if oid != "ob-3" else ("failed", 1, unsent["ob-3"][2])
                          for oid in unsent}
        assert all(state != "attempting" for state, _, _ in rows.values())

    def test_a_cancelled_batch_hands_back_the_claims_after_the_cut_send(self, gw):
        """The boot task cancelled mid-send: the rows after the cut never started and go back unspent; the
        cut one may have reached the chat, so it is left for the next boot's sweep (sent with the marker)."""
        _ledger_ready()
        self._record([("ob-1", "pending"), ("ob-2", "pending")])
        adapter = _ledgered_adapter(gw)
        runner = _boot_runner(gw, adapter)
        started = []

        async def hanging_send(chat_id, content, reply_to=None, metadata=None):
            started.append(content)
            await asyncio.Event().wait()

        adapter.send = hanging_send

        async def exercise():
            task = asyncio.create_task(runner._redeliver_pending_obligations())
            await async_wait_until(lambda: started, timeout=5, what="the first send started")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(exercise())
        rows = _rows_by_id()
        cut = next(oid for oid in rows if started[0].endswith(f"answer {oid}"))
        (rest,) = set(rows) - {cut}
        assert rows[rest] == ("failed", 0, "send_path_degraded") and rows[cut][:2] == ("attempting", 1)
