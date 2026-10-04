"""U4 H2/H3 — ``/acp <task>`` admission through ACP's ``telegram-update.ingress.sock`` lane.

A real unix socket stands in for ACP #1062's lane (one NDJSON envelope in, one answer line out);
the runner, session store, binding and SessionDB are real under a temp home. CEO 36ade192: only the
explicit prefix is managed; a deny, a timeout or an unreadable answer never runs the turn and never
retries; a late or duplicate delivery never runs it twice; a failed settlement stays in doubt.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import acp_managed_ingress as ingress
from gateway import acp_turn_receipts as receipts
from gateway.canonical_surface import CanonicalSurfaceBinding
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource

_TURN = {
    "turnRequestId": "turn-1", "targetActorId": "actor-1", "promptDigest": "sha256:p",
    "bindingGeneration": 4, "targetBindingId": "bind-1", "targetAttestationId": "att-1",
    "executorSessionId": "ses-1", "executorSessionIncarnation": "inc-1",
}


@pytest.fixture
def gw(tmp_path, monkeypatch):
    home, user_home = tmp_path / "hermes-home", tmp_path / "user"
    home.mkdir()
    user_home.mkdir()
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setattr(Path, "home", lambda: user_home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="100200300", chat_type="dm", user_id="100200300")
    entry = runner.session_store.get_or_create_session(source)
    binding = CanonicalSurfaceBinding(
        name="acp-canonical-ceo", session_key=entry.session_key, session_id=entry.session_id,
        telegram_chat_id="100200300", telegram_chat_type="dm", telegram_user_id="100200300",
        telegram_thread_id=None, allowed_author_ids=("a",), allowed_channel_ids=("c",),
    )
    runner.config.canonical_surface_bindings = {binding.name: binding}
    db = runner.session_store._db_for_key(entry.session_key)
    from hermes_state_target_bind import _lineage_root_digest

    lineage = _lineage_root_digest(db._session_lineage_root_to_tip(entry.session_id)[0])
    sock_dir = Path(tempfile.mkdtemp(dir="/tmp", prefix="acp-"))  # AF_UNIX path length limit
    yield SimpleNamespace(runner=runner, source=source, entry=entry, db=db, lineage=lineage,
                          sock=sock_dir / "lane.sock")
    shutil.rmtree(sock_dir, ignore_errors=True)
    runner.session_store.close_all_db_handles()


def _event(text, *, update_id=901, message_id=55):
    return SimpleNamespace(text=text, platform_update_id=update_id, message_id=str(message_id), raw_message=None)


class _Lane:
    """ACP's lane: records every envelope, answers with ``answer(envelope)`` (None = never answer)."""

    def __init__(self, path, answer):
        self.path, self.answer, self.envelopes, self.server = path, answer, [], None

    async def __aenter__(self):
        async def handle(reader, writer):
            envelope = json.loads(await reader.readline())
            self.envelopes.append(envelope)
            reply = self.answer(envelope)
            if reply is None:
                await asyncio.sleep(5)
            else:
                writer.write(json.dumps(reply).encode() + b"\n")
                await writer.drain()
            writer.close()

        self.server = await asyncio.start_unix_server(handle, path=str(self.path))
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()


def _allowed(gw, *, lineage=None):
    def answer(envelope):
        return {"allowed": True, "replayed": False, "turn": dict(_TURN),
                "source": {"channel": "telegram", "nonce": f"update:{envelope['update']['update_id']}"},
                "targetBind": {"requested_session_id": gw.entry.session_id,
                               "lineage_root_digest": lineage or gw.lineage}}
    return answer


def _admit(gw, event, **kw):
    return ingress.admit(gw.runner, event, gw.source, gw.entry.session_key,
                         path=gw.sock, secret="lane-secret", **kw)


def test_only_an_explicit_prefix_on_the_bound_chat_is_managed(gw):
    assert ingress.is_managed(gw.runner, _event("/acp deploy the fix"), gw.source)
    assert not ingress.is_managed(gw.runner, _event("deploy the fix"), gw.source)
    assert not ingress.is_managed(gw.runner, _event("/acp"), gw.source)
    other = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42")
    assert not ingress.is_managed(gw.runner, _event("/acp deploy"), other)


def test_an_allowed_admission_claims_the_receipt_and_hands_back_the_task(gw):
    async def exercise():
        async with _Lane(gw.sock, _allowed(gw)) as lane:
            outcome = await _admit(gw, _event("/acp deploy the fix"))
        return outcome, lane.envelopes

    outcome, envelopes = asyncio.run(exercise())
    assert outcome.reply is None and outcome.admission.task_text == "deploy the fix"
    [envelope] = envelopes
    assert envelope["schema"] == "acp.telegram-external-update/v1" and envelope["binding"] == "acp-canonical-ceo"
    assert envelope["update"] == {"update_id": 901, "message": {
        "message_id": 55, "from": {"id": 100200300}, "chat": {"id": 100200300, "type": "private"},
        "text": "/acp deploy the fix"}}
    receipt = receipts.lookup(gw.db, 901)
    assert receipt.status == "PENDING" and receipt.receipt_identity == _TURN


def test_a_denied_admission_does_not_run_and_writes_nothing(gw):
    async def exercise():
        async with _Lane(gw.sock, lambda e: {"allowed": False, "reasonCode": "OWNER_NOT_ALLOWED", "message": "x"}):
            return await _admit(gw, _event("/acp deploy"))

    outcome = asyncio.run(exercise())
    assert outcome.admission is None and "OWNER_NOT_ALLOWED" in outcome.reply
    assert receipts.lookup(gw.db, 901).status == "NEVER_FOUND"


def test_a_timed_out_admission_does_not_run_retry_or_tombstone(gw):
    """ACP may have committed the claim before its answer was lost: the receipt stays absent so
    ACP keeps the turn in doubt; nothing here claims a non-run it cannot prove (CEO 3c058be8)."""
    async def exercise():
        async with _Lane(gw.sock, lambda e: None) as lane:
            outcome = await _admit(gw, _event("/acp deploy"), timeout=0.3)
        return outcome, lane.envelopes

    outcome, envelopes = asyncio.run(exercise())
    assert outcome.admission is None and outcome.reply == ingress.UNKNOWN
    assert len(envelopes) == 1  # asked once, never retried
    assert receipts.lookup(gw.db, 901).status == "NEVER_FOUND"


def test_an_unreachable_lane_or_missing_secret_is_unknown_and_never_runs(gw):
    outcome = asyncio.run(_admit(gw, _event("/acp deploy")))  # no server on the socket
    assert outcome.admission is None and outcome.reply == ingress.UNKNOWN
    no_secret = asyncio.run(ingress.admit(gw.runner, _event("/acp deploy"), gw.source,
                                          gw.entry.session_key, path=gw.sock, secret=""))
    assert no_secret.admission is None and no_secret.reply == ingress.UNKNOWN


def test_a_redelivered_update_is_not_run_twice(gw):
    """ACP answers a replayed envelope with the same identity; the receipt claim is what stops a
    second execution here."""
    async def exercise():
        async with _Lane(gw.sock, _allowed(gw)):
            first = await _admit(gw, _event("/acp deploy"))
            second = await _admit(gw, _event("/acp deploy"))
        return first, second

    first, second = asyncio.run(exercise())
    assert first.admission is not None
    assert second.admission is None and second.reply.startswith("This /acp update was already handled")


def test_a_lineage_mismatch_refuses_and_records_the_definite_non_run(gw):
    async def exercise():
        async with _Lane(gw.sock, _allowed(gw, lineage="sha256:" + "0" * 64)):
            return await _admit(gw, _event("/acp deploy"))

    outcome = asyncio.run(exercise())
    assert outcome.admission is None
    receipt = receipts.lookup(gw.db, 901)
    assert receipt.status == "ABORTED" and receipt.reason_code == "REFUSED_BEFORE_RUN"


def _admitted(gw, text="/acp deploy"):
    async def exercise():
        async with _Lane(gw.sock, _allowed(gw)):
            return await _admit(gw, _event(text))
    outcome = asyncio.run(exercise())
    assert outcome.admission is not None
    return outcome.admission


class TestRunSync:
    """A managed turn is not streamed (its answer must go out through the ledgered send) and runs
    only on the session ACP approved; settlement is no longer the turn's job."""

    def _run_sync(self, gw, monkeypatch, admission, *, agent=None):
        from gateway.run_turn_runner import TurnRunner
        from gateway.turn_context import TurnContext
        import gateway.run as gateway_run

        calls = []
        monkeypatch.setattr(gateway_run, "_current_max_iterations", lambda: 30)
        monkeypatch.setattr(TurnRunner, "_combined_ephemeral_prompt", lambda self: "")

        def stream(self, key):
            calls.append("stream")
            return None, None, None, False

        monkeypatch.setattr(TurnRunner, "_setup_stream_consumer", stream)
        agent = agent or SimpleNamespace(_session_db=gw.db, session_id=gw.entry.session_id)
        monkeypatch.setattr(TurnRunner, "_resolve_turn_agent", lambda self, *a, **k: (agent, False))
        monkeypatch.setattr(TurnRunner, "_wire_turn_agent_callbacks", lambda self, *a, **k: None)
        monkeypatch.setattr(TurnRunner, "_load_turn_history", lambda self, *a, **k: ([], None, []))
        monkeypatch.setattr(TurnRunner, "_prepare_turn_message", lambda self, *a, **k: (None, None))

        def run(self, *a, **k):
            calls.append("run")
            return {"final_response": "answer", "messages": [], "api_calls": 1}

        monkeypatch.setattr(TurnRunner, "_run_conversation_with_approval", run)
        monkeypatch.setattr(TurnRunner, "_finish_stream_consumer", lambda self, *a, **k: None)
        monkeypatch.setattr(TurnRunner, "_sync_session_after_run", lambda self, *a, **k: (False, "s", 0))
        monkeypatch.setattr(TurnRunner, "_append_auto_media_tags", lambda self, r, *a, **k: r)
        runner = SimpleNamespace(
            _resolve_session_agent_runtime=lambda **k: ("m", {"provider": "openrouter"}),
            _provider_routing=None, _resolve_session_reasoning_config=lambda **k: None,
            _resolve_session_service_tier=lambda **k: None, _resolve_turn_agent_config=lambda *a, **k: None,
        )
        ctx = TurnContext(source=gw.source, session_key=gw.entry.session_key,
                          user_config={}, message="deploy", acp_admission=admission)
        return TurnRunner(runner, ctx).run_sync(), calls

    def test_a_managed_turn_is_not_streamed_and_an_ordinary_one_is(self, gw, monkeypatch):
        _, managed = self._run_sync(gw, monkeypatch, _admitted(gw))
        _, ordinary = self._run_sync(gw, monkeypatch, None)
        assert managed == ["run"] and ordinary == ["stream", "run"]
        assert receipts.lookup(gw.db, 901).status == "PENDING"  # settled by delivery, not the turn

    def test_an_executor_on_another_session_is_refused_before_running(self, gw, monkeypatch):
        """R4: under the slot, the session about to run must be the one ACP approved."""
        other = SimpleNamespace(_session_db=gw.db, session_id="some-other-session")
        out, calls = self._run_sync(gw, monkeypatch, _admitted(gw), agent=other)
        assert "run" not in calls and "target mismatch" in out["final_response"]
        receipt = receipts.lookup(gw.db, 901)
        assert receipt.status == "ABORTED" and receipt.reason_code == "REFUSED_BEFORE_RUN"


def test_a_bound_chat_routed_to_another_session_is_refused_unasked(gw):
    """R4: a /acp from the bound chat that resolves to another session key never reaches ACP."""
    async def exercise():
        async with _Lane(gw.sock, _allowed(gw)) as lane:
            outcome = await ingress.admit(gw.runner, _event("/acp deploy"), gw.source,
                                          "agent:other:telegram:dm:100200300", path=gw.sock, secret="s")
        return outcome, lane.envelopes

    outcome, envelopes = asyncio.run(exercise())
    assert outcome.admission is None and "target mismatch" in outcome.reply and envelopes == []


def test_a_disabled_delivery_ledger_refuses_unasked(gw, monkeypatch):
    """Without the ledger there is no evidence an answer was delivered, so nothing is admitted."""
    monkeypatch.setattr("gateway.delivery_ledger.ledger_enabled", lambda config=None: False)

    async def exercise():
        async with _Lane(gw.sock, _allowed(gw)) as lane:
            return await _admit(gw, _event("/acp deploy")), lane.envelopes

    outcome, envelopes = asyncio.run(exercise())
    assert outcome.admission is None and "ledger" in outcome.reply and envelopes == []


class TestLedgerSettlement:
    """R2/R3 through the real delivery ledger (same state.db): the receipt settles from the answer
    that was actually sent, live or redelivered, and no settlement-write failure, crash or upgrade
    can turn an answered update into a death before answer."""

    def _adapter(self, gw, *, succeed=True):
        from gateway.config import PlatformConfig
        from gateway.platforms.base import BasePlatformAdapter, SendResult

        class _Adapter(BasePlatformAdapter):
            def __init__(self):
                super().__init__(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
                self.sent = []

            async def connect(self):
                return True

            async def disconnect(self):
                pass

            async def send(self, chat_id, content, reply_to=None, metadata=None):
                self.sent.append(content)
                return SendResult(success=succeed, message_id="77", error=None if succeed else "flood")

            async def get_chat_info(self, chat_id):
                return {"id": chat_id}

        adapter = _Adapter()
        adapter.gateway_runner = gw.runner
        return adapter

    def _managed_event(self, gw, admission):
        from gateway.platforms.event import MessageEvent
        event = MessageEvent(text=admission.task_text, source=gw.source, message_id="55")
        event._acp_admission = admission
        return event

    def test_the_delivered_final_text_is_the_evidence(self, gw):
        import hashlib

        admission = _admitted(gw, "/acp /deploy now")  # a task text starting with "/" is still ledgered
        adapter = self._adapter(gw)
        asyncio.run(adapter.send_final_ledgered(self._managed_event(gw, admission), gw.entry.session_key,
                                                "shaped final", {}, reply_to="55"))
        receipt = receipts.lookup(gw.db, 901)
        assert adapter.sent == ["shaped final"] and receipt.status == "COMPLETED"
        assert receipt.evidence_digest == "sha256:" + hashlib.sha256(b"shaped final").hexdigest()
        assert receipt.receipt_id.startswith("hermes-tg:") and receipt.receipt_identity == _TURN
        assert receipt.content == "shaped final"

    @pytest.mark.parametrize("settlement_write_fails", [False, True])
    def test_get_content_bytes_hash_to_the_evidence_digest(self, gw, monkeypatch, settlement_write_fails):
        """ACP's client counts a COMPLETED answer as found only when sha256(content) equals
        evidenceDigest. Both come from the one ledgered text, whole (a reply the adapter splits on
        the wire is ledgered unsplit), whether the receipt was settled or GET reads the row."""
        import hashlib

        text = "배포 완료 — " + "가나다 ✅ " * 600  # non-ASCII, longer than one Telegram message
        admission = _admitted(gw)
        with monkeypatch.context() as m:
            if settlement_write_fails:
                m.setattr(receipts, "_settle", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
            asyncio.run(self._adapter(gw).send_final_ledgered(
                self._managed_event(gw, admission), gw.entry.session_key, text, {}, reply_to="55"))
        body = receipts.lookup(gw.db, 901).to_response()
        assert body["status"] == "COMPLETED" and body["content"] == text
        assert body["evidenceDigest"] == "sha256:" + hashlib.sha256(body["content"].encode("utf-8")).hexdigest()

    def test_delivery_is_the_closed_six_field_shape_acp_parses(self, gw):
        """ACP's receipt port accepts delivery only as exactly these keys, ids as integers, at least
        one sent message; anything else reads NOT_FOUND there and the turn never settles."""
        from gateway.platforms.base import SendResult

        admission = _admitted(gw)
        adapter = self._adapter(gw)

        async def split_send(chat_id, content, reply_to=None, metadata=None):
            return SendResult(success=True, message_id="301", raw_response={"message_ids": ["301", "302", "303"]})

        adapter.send = split_send
        asyncio.run(adapter.send_final_ledgered(self._managed_event(gw, admission), gw.entry.session_key,
                                                "long answer", {}, reply_to="55"))
        delivery = receipts.lookup(gw.db, 901).to_response()["delivery"]
        assert set(delivery) == {"obligation_id", "state", "content_digest", "chat_id",
                                 "reply_to_message_id", "message_ids"}
        assert delivery["state"] == "delivered" and delivery["message_ids"] == [301, 302, 303]
        assert delivery["chat_id"] == 100200300 and delivery["reply_to_message_id"] == 55
        assert delivery["content_digest"] == receipts.lookup(gw.db, 901).evidence_digest

    def test_a_delivery_without_message_ids_stays_in_doubt_and_is_never_aborted(self, gw, monkeypatch):
        from gateway.platforms.base import SendResult

        admission = _admitted(gw)
        adapter = self._adapter(gw)

        async def idless_send(chat_id, content, reply_to=None, metadata=None):
            return SendResult(success=True, message_id=None)

        adapter.send = idless_send
        asyncio.run(adapter.send_final_ledgered(self._managed_event(gw, admission), gw.entry.session_key,
                                                "answer", {}, reply_to="55"))
        assert receipts.lookup(gw.db, 901).status == "PENDING"
        monkeypatch.setattr(ingress, "_PROCESS_OWNER", "a-later-process")
        assert ingress.sweep_at_startup(gw.runner) == []
        assert receipts.lookup(gw.db, 901).status == "PENDING"

    def test_a_queued_follow_up_never_certifies_the_managed_receipt(self, gw):
        admission = _admitted(gw)
        adapter = self._adapter(gw)

        async def exercise():
            await gw.runner._send_queued_final_text(adapter, gw.source, "follow-up answer", None, "56",
                                                    gw.entry.session_key, "56")
            assert receipts.lookup(gw.db, 901).status == "PENDING"
            await gw.runner._send_queued_final_text(adapter, gw.source, "managed answer", None, "55",
                                                    gw.entry.session_key, "55", acp_admission=admission)

        asyncio.run(exercise())
        assert receipts.lookup(gw.db, 901).content == "managed answer"

    def test_a_failed_settlement_write_still_reads_completed_and_is_settled_at_startup(self, gw, monkeypatch):
        admission = _admitted(gw)
        with monkeypatch.context() as broken:
            broken.setattr(receipts, "_settle", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
            asyncio.run(self._adapter(gw).send_final_ledgered(
                self._managed_event(gw, admission), gw.entry.session_key, "the answer", {}, reply_to="55"))
        assert receipts.decode(gw.db.get_meta(receipts.receipt_key(901)), 901).status == "PENDING"
        assert receipts.lookup(gw.db, 901).status == "COMPLETED"  # GET reads the delivered row
        monkeypatch.setattr(ingress, "_PROCESS_OWNER", "a-later-process")
        assert ingress.sweep_at_startup(gw.runner) == []
        assert receipts.decode(gw.db.get_meta(receipts.receipt_key(901)), 901).status == "COMPLETED"

    def test_an_undelivered_answer_is_never_redelivered_and_stays_in_doubt(self, gw, monkeypatch):
        """U4-04: no automatic sweep claims a managed row (recovery sends without the reply anchor
        and with recovery decoration, which the receipt could not truthfully certify), while an
        ordinary failed row beside it is still claimed by each of them."""
        import time as _time
        from gateway import delivery_ledger

        admission = _admitted(gw)
        asyncio.run(self._adapter(gw, succeed=False).send_final_ledgered(
            self._managed_event(gw, admission), gw.entry.session_key, "the answer", {}, reply_to="55"))
        [row] = receipts.ledger_answers(gw.db, 901)
        assert row["state"] == "failed"
        assert delivery_ledger.pending_retries() == []  # no timer for the managed row
        delivery_ledger.record_obligation(obligation_id="ob-ordinary", session_key=gw.entry.session_key,
                                          platform="telegram", chat_id="100200300", thread_id=None,
                                          content="x")
        delivery_ledger.mark_failed("ob-ordinary", "boom")
        assert [r["platform"] for r in delivery_ledger.pending_retries()] == ["telegram"]
        soon = _time.time() + 600  # past the retry backoff, inside the stale cutoff
        assert [r["obligation_id"] for r in delivery_ledger.sweep_failed_for_runtime("telegram", now=soon)] \
            == ["ob-ordinary"]
        delivery_ledger.mark_failed("ob-ordinary", "boom")
        monkeypatch.setattr(delivery_ledger, "_owner_alive", lambda *a: False)  # as seen by the next boot
        assert [r["obligation_id"] for r in delivery_ledger.sweep_recoverable(now=soon)] == ["ob-ordinary"]
        assert receipts.ledger_answers(gw.db, 901)[0]["state"] == "failed"
        monkeypatch.setattr(ingress, "_PROCESS_OWNER", "a-later-process")
        assert ingress.sweep_at_startup(gw.runner) == []  # owed, not dead
        assert receipts.lookup(gw.db, 901).status == "PENDING"

    def test_an_unrecorded_obligation_withholds_the_managed_answer(self, gw, monkeypatch):
        """U4-03: no ledger row, no managed send. The receipt is a definite non-delivery."""
        from gateway import delivery_ledger

        admission = _admitted(gw)
        adapter = self._adapter(gw)
        monkeypatch.setattr(delivery_ledger, "record_obligation",
                            lambda **k: (_ for _ in ()).throw(OSError("disk I/O error")))
        result, _ = asyncio.run(adapter.send_final_ledgered(
            self._managed_event(gw, admission), gw.entry.session_key, "managed answer", {}, reply_to="55"))
        assert result.success is False and adapter.sent == [ingress.UNRECORDED]
        receipt = receipts.lookup(gw.db, 901)
        assert receipt.status == "ABORTED" and receipt.reason_code == "HERMES_ANSWER_NOT_RECORDED"

    def test_an_unrecorded_ordinary_answer_is_still_sent(self, gw, monkeypatch):
        from gateway import delivery_ledger
        from gateway.platforms.event import MessageEvent

        adapter = self._adapter(gw)
        monkeypatch.setattr(delivery_ledger, "record_obligation",
                            lambda **k: (_ for _ in ()).throw(OSError("disk I/O error")))
        asyncio.run(adapter.send_final_ledgered(
            MessageEvent(text="hi", source=gw.source, message_id="56"), gw.entry.session_key,
            "ordinary answer", {}, reply_to="56"))
        assert adapter.sent == ["ordinary answer"]

    def test_only_an_update_with_no_ledgered_answer_is_a_death_before_answer(self, gw, monkeypatch):
        from gateway import delivery_ledger

        proof = {"proven_db_path": gw.db.db_path, "proven_db_identity": gw.db._db_file_identity}
        for update in (1, 2, 3):
            receipts.claim_pending(gw.db, update, message_id="m", turn_request_id="t",
                                   receipt_identity=_TURN, owner="dead-process", **proof)
        # 2: an older build recorded the answer's correlation on the receipt itself.
        raw = json.loads(gw.db.get_meta(receipts.receipt_key(2)))
        gw.db.compare_and_set_meta(receipts.receipt_key(2), gw.db.get_meta(receipts.receipt_key(2)),
                                   json.dumps({**raw, "obligation_id": "ob-old"}), **proof)
        # 3: answered, but the ledger gave up delivering it.
        delivery_ledger.record_obligation(obligation_id="ob-3", session_key=gw.entry.session_key,
                                          platform="telegram", chat_id="100200300", thread_id=None,
                                          content="x", acp_update_id="3")
        delivery_ledger._update_state("ob-3", "abandoned")
        assert sorted(ingress.sweep_at_startup(gw.runner)) == ["1", "3"]
        assert receipts.lookup(gw.db, 1).reason_code == "HERMES_PROCESS_DIED_BEFORE_ANSWER"
        assert receipts.lookup(gw.db, 2).status == "PENDING"
        assert receipts.lookup(gw.db, 3).reason_code == "HERMES_ANSWER_UNDELIVERABLE"


class TestAdapterEntry:
    """R1: the adapter's own busy and batching routes, which never pass through the runner's
    ``_handle_message`` guard."""

    def _adapter(self, gw):
        from gateway.config import PlatformConfig
        from gateway.platforms.base import BasePlatformAdapter, SendResult

        class _Adapter(BasePlatformAdapter):
            def __init__(self):
                super().__init__(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
                self.sent, self.dispatched = [], []

            async def connect(self):
                return True

            async def disconnect(self):
                pass

            async def send(self, chat_id, content, reply_to=None, metadata=None):
                self.sent.append(content)
                return SendResult(success=True, message_id="1")

            async def get_chat_info(self, chat_id):
                return {"id": chat_id}

            async def _dispatch_text_batch(self, event):
                self.dispatched.append(event.text)

        adapter = _Adapter()
        adapter.gateway_runner = gw.runner

        async def handler(event):
            return None
        adapter._message_handler = handler
        return adapter

    def _msg(self, gw, text, mid):
        from gateway.platforms.event import MessageEvent, MessageType
        return MessageEvent(text=text, message_type=MessageType.TEXT, source=gw.source,
                            message_id=str(mid), platform_update_id=mid)

    @pytest.mark.asyncio
    async def test_a_task_arriving_while_the_adapter_session_is_busy_is_refused_not_queued(self, gw):
        adapter = self._adapter(gw)
        busy_calls = []

        async def busy(event, key):
            busy_calls.append(event.text)
            return True

        adapter.set_busy_session_handler(busy)
        key = adapter._event_session_key(self._msg(gw, "x", 1))
        adapter._active_sessions[key] = asyncio.Event()
        await adapter.handle_message(self._msg(gw, "/acp deploy", 2))
        assert adapter.sent == [ingress.BUSY] and busy_calls == [] and key not in adapter._pending_messages
        await adapter.handle_message(self._msg(gw, "ordinary", 3))
        assert busy_calls == ["ordinary"]  # ordinary text keeps its busy route

class TestUpdateIdentity:
    """U4-01 (option A, CEO d34c076b): an /acp message is always its own update. Telegram gives no
    evidence that tells a client split from a separate message, so nothing is ever joined to a task,
    and a task long enough to have been split is refused before ACP is asked."""

    def _run(self, gw, chunks):
        adapter = TestAdapterEntry()._adapter(gw)
        adapter._text_batch_delay_seconds = adapter._text_batch_split_delay_seconds = 0.05
        from gateway.platforms.event import MessageEvent, MessageType

        async def exercise():
            for text, mid in chunks:
                adapter._enqueue_text_event(MessageEvent(text=text, message_type=MessageType.TEXT,
                                                         source=gw.source, message_id=str(mid),
                                                         platform_update_id=mid))
            for _ in range(100):
                if len(adapter.dispatched) >= len(chunks):
                    break
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.15)
            return adapter.dispatched
        return asyncio.run(exercise())

    @pytest.mark.parametrize("head", ["/acp deploy", "/acp " + "x" * 3990, "/acp " + "x" * 4091],
                             ids=["short", "near_limit", "split_length"])
    def test_an_ordinary_follow_up_is_never_absorbed(self, gw, head):
        dispatched = self._run(gw, [("hello", 10), (head, 11), ("next message", 12)])
        assert sorted(dispatched, key=len) == sorted(["hello", head, "next message"], key=len)

    def test_a_split_length_task_is_refused_before_acp_is_asked(self, gw):
        async def exercise():
            async with _Lane(gw.sock, _allowed(gw)) as lane:
                outcome = await _admit(gw, _event("/acp " + "x" * 3995))
            return outcome, lane.envelopes

        outcome, envelopes = asyncio.run(exercise())
        assert outcome.reply == ingress.TOO_LONG and outcome.admission is None and envelopes == []
        assert receipts.lookup(gw.db, 901).status == "NEVER_FOUND"

    def test_a_task_under_the_limit_is_admitted(self, gw):
        assert _admitted(gw, "/acp " + "x" * 3990).task_text == "x" * 3990


class TestQueuedChain:
    """U4-02: when a /queue chain ends on an ordinary follow-up, the outer final answers that
    follow-up; it must not carry the managed task's admission into the follow-up's obligation."""

    def test_the_terminal_handoff_drops_the_admission(self, gw):
        from gateway.platforms.event import MessageEvent
        from gateway.run_turn import GatewayTurnMixin

        admission = _admitted(gw)
        event = MessageEvent(text="deploy", source=gw.source, message_id="55")
        event._acp_admission = admission
        GatewayTurnMixin._adopt_queued_terminal(event, {"queued_terminal_inbound_id": "56"})
        assert event.ledger_message_id == "56" and event._acp_admission is None

    def test_a_failed_managed_answer_is_not_certified_by_the_follow_up(self, gw):
        """The reviewer's chain: the managed answer's send fails, the follow-up's outer final
        succeeds on the same event object. The receipt stays PENDING for its own answer."""
        from gateway.platforms.event import MessageEvent
        from gateway.run_turn import GatewayTurnMixin

        admission = _admitted(gw)
        settle = TestLedgerSettlement()
        event = MessageEvent(text="deploy", source=gw.source, message_id="55")
        event._acp_admission = admission

        async def exercise():
            await gw.runner._send_queued_final_text(settle._adapter(gw, succeed=False), gw.source,
                                                    "managed answer", None, "55", gw.entry.session_key,
                                                    "55", acp_admission=admission)
            GatewayTurnMixin._adopt_queued_terminal(event, {"queued_terminal_inbound_id": "56"})
            await settle._adapter(gw).send_final_ledgered(event, gw.entry.session_key,
                                                          "follow-up answer", {}, reply_to="55")

        asyncio.run(exercise())
        receipt = receipts.lookup(gw.db, 901)
        assert receipt.status == "PENDING"
        assert [r["content"] for r in receipts.ledger_answers(gw.db, 901)] == ["managed answer"]


class TestHandleMessageSeam:
    """The seam in the real ``GatewayRunner._handle_message``: ingress gates and the agent run are
    the only stand-ins; the managed check, busy fast-path, command dispatch and claim are real."""

    def _wire(self, gw, monkeypatch, lane_answer):
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

        def event(text):
            return MessageEvent(text=text, source=gw.source, message_id="55", platform_update_id=901)
        return runs, event

    def test_an_admitted_task_runs_on_its_text_and_ordinary_chat_never_asks_acp(self, gw, monkeypatch):
        runs, event = self._wire(gw, monkeypatch, None)

        async def exercise():
            async with _Lane(gw.sock, _allowed(gw)) as lane:
                managed = await gw.runner._handle_message(event("/acp deploy the fix"))
                ordinary = await gw.runner._handle_message(event("how are you"))
            return managed, ordinary, lane.envelopes

        managed, ordinary, envelopes = asyncio.run(exercise())
        assert managed == ordinary == "final reply"
        assert runs == ["deploy the fix", "how are you"]
        assert len(envelopes) == 1  # only the /acp message went to ACP
        assert receipts.lookup(gw.db, 901).status == "PENDING"  # settled later by the delivery

    def test_a_denied_task_never_reaches_the_agent(self, gw, monkeypatch):
        runs, event = self._wire(gw, monkeypatch, None)

        async def exercise():
            async with _Lane(gw.sock, lambda e: {"allowed": False, "reasonCode": "NOPE", "message": "x"}):
                return await gw.runner._handle_message(event("/acp deploy"))

        reply = asyncio.run(exercise())
        assert "NOPE" in reply and runs == []

    def test_a_task_arriving_during_a_busy_turn_is_refused_not_queued(self, gw, monkeypatch):
        """A queued /acp message would later drain as an ordinary turn with no admission."""
        runs, event = self._wire(gw, monkeypatch, None)
        queued = []

        async def queue(*args, **kwargs):
            queued.append(args)
            return None

        monkeypatch.setattr(gw.runner, "_hm_handle_running_session_message", queue)
        monkeypatch.setattr(gw.runner, "_is_session_running", lambda key: True)
        monkeypatch.setattr(gw.runner, "_hm_evict_reaped_agent", lambda key: None)
        reply = asyncio.run(gw.runner._handle_message(event("/acp deploy")))
        assert reply == ingress.BUSY and queued == [] and runs == []


class TestStore:
    """U4-05: receipts and the delivery ledger share one store, the launch home's state.db. A
    binding whose session a named profile owns is not managed at all."""

    def test_a_named_profile_binding_is_not_managed(self, gw, monkeypatch):
        assert ingress.bound_binding(gw.runner, gw.source) is not None
        monkeypatch.setattr(gw.runner.session_store, "_named_profile_for_key", lambda key: "work")
        assert ingress.bound_binding(gw.runner, gw.source) is None

    def test_a_store_other_than_the_ledgers_is_refused_unasked(self, gw, monkeypatch, tmp_path):
        from gateway import delivery_ledger

        monkeypatch.setattr(delivery_ledger, "_db_path", lambda: tmp_path / "elsewhere.db")

        async def exercise():
            async with _Lane(gw.sock, _allowed(gw)) as lane:
                outcome = await _admit(gw, _event("/acp deploy"))
            return outcome, lane.envelopes

        outcome, envelopes = asyncio.run(exercise())
        assert outcome.admission is None and "launch home" in outcome.reply and envelopes == []


class TestCrashRecovery:
    """Round-2 ROUND1-ESCAPE-01: a crash-left managed turn is recognised by its bound active-turn
    marker, never by message time. The persisted user row carries Telegram's authored time, which
    precedes the marker, exactly as production stores it."""

    def _crash_left(self, gw, *, bind, has_answer=False):
        import time as _time
        from datetime import datetime, timezone
        from gateway.platforms.event import MessageEvent

        admission = _admitted(gw)
        store = gw.runner.session_store
        event = MessageEvent(text="deploy", source=gw.source, message_id="55",
                             timestamp=datetime.fromtimestamp(_time.time() - 10, timezone.utc))
        token = store.mark_turn_active(gw.entry.session_key)
        if bind:
            assert ingress.bind_turn_marker(admission, token)
        started = store._entries[gw.entry.session_key].active_turn_started_at.timestamp()
        _, user_text, user_ts = gw.runner._hmwa_apply_message_timestamp(event, event.text)
        assert user_ts < started  # the production relationship the old timestamp check missed
        store.append_to_transcript(gw.entry.session_id, {
            "role": "user", "content": user_text, "timestamp": user_ts,
            "display_metadata": {"acp_update_id": str(admission.update_id)}})
        if has_answer:
            store.append_to_transcript(gw.entry.session_id, {"role": "assistant", "content": "managed answer"})
        return admission, store

    @pytest.mark.parametrize("has_answer", [False, True])
    def test_a_bound_managed_turn_is_neither_resumed_nor_ledgered(self, gw, has_answer):
        _, store = self._crash_left(gw, bind=True, has_answer=has_answer)
        recovered = asyncio.run(gw.runner._recover_unclean_sessions())
        entry = store._entries[gw.entry.session_key]
        assert recovered == (0, 0) and not entry.resume_pending and not entry.active_turn_token
        assert receipts.ledger_answers(gw.db, 901) == []

    def test_an_unbound_turn_on_the_bound_session_keeps_ordinary_recovery(self, gw):
        _, store = self._crash_left(gw, bind=False)
        assert asyncio.run(gw.runner._recover_unclean_sessions()) == (1, 0)
        assert store._entries[gw.entry.session_key].resume_pending

    def test_an_unreadable_marker_store_never_resumes_the_bound_session(self, gw, monkeypatch):
        _, store = self._crash_left(gw, bind=True)
        monkeypatch.setattr(type(gw.db), "get_meta", lambda self, key: (_ for _ in ()).throw(OSError("disk")))
        assert asyncio.run(gw.runner._recover_unclean_sessions()) == (0, 0)
        assert not store._entries[gw.entry.session_key].resume_pending

    def test_another_session_is_never_treated_as_managed(self, gw):
        assert ingress.crash_left_turn_is_managed(gw.runner, "agent:main:telegram:dm:5", "tok") is False

    def test_a_marker_that_cannot_be_bound_refuses_the_turn(self, gw, monkeypatch):
        admission = _admitted(gw)
        monkeypatch.setattr(type(gw.db), "claim_meta_once",
                            lambda self, *a, **k: (_ for _ in ()).throw(OSError("disk")))
        assert ingress.bind_turn_marker(admission, "tok") is False
        assert ingress.bind_turn_marker(admission, None) is False


# Round-2 review probes (hermes-agent-pr84 round-0002), kept as regression witnesses.

async def _outer_handoff(gw, monkeypatch, event, result):
    from unittest.mock import AsyncMock

    prepared = gw.runner._PreparedTurn([], "", event.text, event.text, None, None,
                                        gw.entry.session_id, "probe-owner")
    monkeypatch.setattr(gw.runner, "_hmwa_resolve_session",
                        AsyncMock(return_value=(gw.source, gw.entry, gw.entry.session_key)))
    monkeypatch.setattr(gw.runner, "_hmwa_prepare_turn", AsyncMock(return_value=(prepared, None)))
    monkeypatch.setattr(gw.runner, "_run_agent", AsyncMock(return_value=result))
    monkeypatch.setattr(gw.runner, "_is_session_run_current", lambda *a: True)
    monkeypatch.setattr(gw.runner, "_hmwa_stop_typing_for_turn", AsyncMock())
    monkeypatch.setattr(gw.runner, "_hmwa_shape_agent_response",
                        AsyncMock(return_value=(result["final_response"], False, [])))
    monkeypatch.setattr(gw.runner, "_hmwa_prepend_reasoning", lambda a, r, *args: r)
    monkeypatch.setattr(gw.runner, "_hmwa_runtime_footer_line", lambda *a: "")
    monkeypatch.setattr(gw.runner, "_hmwa_post_turn_hooks", AsyncMock())
    monkeypatch.setattr(gw.runner, "_hmwa_classify_turn_failure", lambda *a: (False, False, False))
    monkeypatch.setattr(gw.runner, "_hmwa_compression_exhaustion_notice", lambda a, r, *args: r)
    monkeypatch.setattr(gw.runner, "_hmwa_persist_turn_transcript", AsyncMock())
    monkeypatch.setattr(gw.runner, "_clear_session_env", lambda *a: None)
    monkeypatch.setattr(gw.runner, "_hmwa_deliver_turn_response",
                        AsyncMock(return_value=result["final_response"]))
    return await gw.runner._handle_message_with_agent(event, gw.source, gw.entry.session_key, 1)


class TestReviewRound2:
    def test_an_idless_terminal_follow_up_never_certifies_the_managed_turn(self, gw, monkeypatch):
        """U4-02: a leftover steer has no platform id; the outer final still answers it."""
        from unittest.mock import AsyncMock
        from gateway.platforms.event import MessageEvent
        from gateway.turn_context import TurnContext

        admission = _admitted(gw)
        # As in production, the managed turn's marker is bound before its chain runs.
        assert ingress.bind_turn_marker(admission, gw.runner.session_store.mark_turn_active(gw.entry.session_key))
        settle = TestLedgerSettlement()
        adapter = settle._adapter(gw)
        ctx = TurnContext(source=gw.source, session_key=gw.entry.session_key,
                          session_id=gw.entry.session_id, history=[], acp_admission=admission,
                          inbound_message_id="55", event_message_id="55")
        event = MessageEvent(text="deploy", source=gw.source, message_id="55")
        event._acp_admission = admission

        from gateway.platforms.base import SendResult
        real_send = adapter.send

        async def failing_send(chat_id, content, reply_to=None, metadata=None):
            adapter.sent.append(content)
            return SendResult(success=False, error="flood")

        async def first_response(*args):
            adapter.send = failing_send  # the managed answer's own send is rejected
            try:
                await gw.runner._send_queued_final_text(adapter, gw.source, "managed answer", None, "55",
                                                        gw.entry.session_key, "55", acp_admission=admission)
            finally:
                adapter.send = real_send

        followup = AsyncMock(return_value={"final_response": "follow-up answer", "messages": []})
        monkeypatch.setattr(gw.runner, "_run_agent_deliver_first_response", first_response)
        monkeypatch.setattr(gw.runner, "_run_agent", followup)
        monkeypatch.setattr(gw.runner, "_refresh_agent_cache_message_count", AsyncMock())
        monkeypatch.setattr(gw.runner, "_delivery_adapter_for", lambda source: adapter)

        async def exercise():
            result = await gw.runner._run_agent_queued_followup(
                ctx, adapter, "ordinary leftover steer", None,
                {"final_response": "managed answer"}, {"messages": []}, None)
            assert result["queued_terminal_inbound_id"] is None
            response = await _outer_handoff(gw, monkeypatch, event, result)
            await adapter.send_final_ledgered(event, gw.entry.session_key, response, {}, reply_to="55")

        asyncio.run(exercise())
        assert "managed answer" in adapter.sent and adapter.sent[-1] == "follow-up answer"
        assert receipts.lookup(gw.db, 901).status == "PENDING"
        assert [(r["content"], r["state"]) for r in receipts.ledger_answers(gw.db, 901)] == [
            ("managed answer", "failed")]

    def test_an_ordinary_follow_up_after_a_slash_task_is_still_ledgered(self, gw, monkeypatch):
        """Round-2 regression: the handed-off final is ledgered even though the task began with "/"."""
        from unittest.mock import AsyncMock
        from gateway import delivery_ledger
        from gateway.platforms.base import SendResult
        from gateway.platforms.event import MessageEvent

        admission = _admitted(gw, "/acp /deploy now")
        event = MessageEvent(text=admission.task_text, source=gw.source, message_id="55")
        event._acp_admission = admission
        adapter = TestLedgerSettlement()._adapter(gw)
        adapter._send_with_retry = AsyncMock(return_value=SendResult(success=False, error="flood_control:600"))

        async def exercise():
            result = {"final_response": "ordinary follow-up answer", "queued_terminal_inbound_id": "56"}
            response = await _outer_handoff(gw, monkeypatch, event, result)
            await adapter.send_final_ledgered(event, gw.entry.session_key, response, {}, reply_to="55")

        asyncio.run(exercise())
        with delivery_ledger._connect() as conn:
            rows = conn.execute("SELECT content, acp_update_id FROM delivery_obligations").fetchall()
        assert rows == [("ordinary follow-up answer", None)]

    def test_a_recording_failure_after_an_uncertain_send_does_not_abort(self, gw, monkeypatch):
        """U4-R01: an earlier send failed after submission; its delivery is unknown, so the later
        recording failure withholds the new send but leaves the receipt in doubt."""
        from gateway import delivery_ledger
        from gateway.platforms.base import SendResult

        admission = _admitted(gw)
        settle = TestLedgerSettlement()
        event = settle._managed_event(gw, admission)
        adapter = settle._adapter(gw)

        async def uncertain_send(**kwargs):
            adapter.sent.append(kwargs["content"])
            return SendResult(success=False, error="request timed out after submission")

        adapter._send_with_retry = uncertain_send
        asyncio.run(adapter.send_final_ledgered(event, gw.entry.session_key, "managed answer", {}, reply_to="55"))
        assert receipts.ledger_answers(gw.db, 901)[0]["state"] == "failed"
        monkeypatch.setattr(delivery_ledger, "record_obligation",
                            lambda **kwargs: (_ for _ in ()).throw(OSError("disk I/O error")))
        asyncio.run(adapter.send_final_ledgered(event, gw.entry.session_key, "managed answer", {}, reply_to="55"))
        assert receipts.lookup(gw.db, 901).status == "PENDING"
        assert adapter.sent[-1] == ingress.UNRECORDED  # the new send itself is still withheld

    def test_an_exact_4000_char_head_never_absorbs_a_follow_up(self, gw):
        head = "/acp " + "x" * 3995
        dispatched = TestUpdateIdentity()._run(gw, [(head, 11), ("ordinary", 12)])
        assert sorted(dispatched, key=len) == sorted([head, "ordinary"], key=len)


# Final-review probes (hermes-agent-pr84 round-0003), kept as regression witnesses.

def _marked_turn(gw, *, answer=False):
    import time as _time
    from datetime import datetime, timezone
    from gateway.platforms.event import MessageEvent

    admission = _admitted(gw)
    store = gw.runner.session_store
    token = store.mark_turn_active(gw.entry.session_key)
    assert ingress.bind_turn_marker(admission, token)
    event = MessageEvent(text="deploy", source=gw.source, message_id="55",
                         timestamp=datetime.fromtimestamp(_time.time() - 10, timezone.utc))
    _, text, ts = gw.runner._hmwa_apply_message_timestamp(event, event.text)
    store.append_to_transcript(gw.entry.session_id, {"role": "user", "content": text, "timestamp": ts,
                                                      "display_metadata": {"acp_update_id": str(admission.update_id)}})
    if answer:
        store.append_to_transcript(gw.entry.session_id, {"role": "assistant", "content": "managed answer"})
    return admission, store


class TestReviewRound3:
    def test_a_marker_binding_failure_delivers_the_refusal(self, gw, monkeypatch):
        """U4-R03: the real handler returns the refusal (not a crash) and never runs the turn."""
        from unittest.mock import AsyncMock
        from gateway.platforms.event import MessageEvent

        admission = _admitted(gw)
        event = MessageEvent(text="deploy", source=gw.source, message_id="55")
        event._acp_admission = admission
        runner = gw.runner
        monkeypatch.setattr(runner, "_hmwa_resolve_session",
                            AsyncMock(return_value=(gw.source, gw.entry, gw.entry.session_key)))
        monkeypatch.setattr(runner, "_hmwa_open_session", AsyncMock(return_value=(False, False)))
        monkeypatch.setattr(runner, "_set_session_env", lambda context: [])
        monkeypatch.setattr(runner, "_clear_session_env", lambda tokens: None)
        monkeypatch.setattr(runner, "_pinned_session_context_prompt", lambda *a: "")
        monkeypatch.setattr(runner, "_hmwa_acquire_turn_lease", AsyncMock())
        monkeypatch.setattr(ingress, "bind_turn_marker", lambda *a: False)
        run = AsyncMock()
        monkeypatch.setattr(runner, "_run_agent", run)
        reply = asyncio.run(runner._handle_message_with_agent(event, gw.source, gw.entry.session_key, 1))
        assert "turn identity could not be recorded" in reply and run.await_count == 0
        assert receipts.lookup(gw.db, 901).status == "ABORTED"

    def test_a_failed_managed_marker_cleanup_never_promotes_a_resume(self, gw, monkeypatch):
        _, store = _marked_turn(gw)

        async def unavailable(*args):
            raise OSError("temporary failure persisting marker cleanup")

        monkeypatch.setattr(gw.runner.async_session_store, "clear_turn_active", unavailable)
        assert asyncio.run(gw.runner._recover_unclean_sessions()) == (0, 0)
        assert not store._entries[gw.entry.session_key].resume_pending

    def test_durable_managed_identity_survives_binding_removal(self, gw):
        from unittest.mock import AsyncMock
        from gateway import delivery_ledger

        _marked_turn(gw, answer=True)
        gw.runner.config.canonical_surface_bindings = {}
        recovered = asyncio.run(gw.runner._recover_unclean_sessions())
        with delivery_ledger._connect() as conn:
            rows = conn.execute("SELECT content, acp_update_id FROM delivery_obligations").fetchall()
        adapter = TestLedgerSettlement()._adapter(gw)
        gw.runner._obligation_adapter = AsyncMock(return_value=adapter)
        for row in delivery_ledger.sweep_recoverable():
            asyncio.run(gw.runner._redeliver_claimed_row(row, delivery_ledger.RECOVERED_MARKER))
        assert recovered == (0, 0) and rows == [] and adapter.sent == []


class TestManagedTurnFollowups:
    """Third supplementary review (3775c2c60c): position-based chain recovery could not survive
    compaction and its deferral lost input. A managed turn now runs no follow-up under its marker:
    the adapter's end-of-task drain runs it as its own ordinary turn, so a crash-left managed marker
    never owes anything but stays suppressed."""

    def _ctx(self, gw, admission):
        from gateway.turn_context import TurnContext

        return TurnContext(source=gw.source, session_key=gw.entry.session_key, session_id=gw.entry.session_id,
                           history=[], acp_admission=admission, inbound_message_id="55", event_message_id="55")

    def _queued(self, gw, text="queued follow-up"):
        from gateway.platforms.event import MessageEvent, MessageType

        return MessageEvent(text=text, message_type=MessageType.TEXT, source=gw.source, message_id="56")

    def test_a_managed_turn_leaves_a_queued_event_for_the_adapter_drain(self, gw):
        adapter = TestLedgerSettlement()._adapter(gw)
        event = self._queued(gw)
        adapter._pending_messages[gw.entry.session_key] = event
        got = asyncio.run(gw.runner._run_agent_next_followup(
            self._ctx(gw, _admitted(gw)), {"messages": []}, adapter, gw.source, gw.entry.session_key))
        assert got == (None, None) and adapter._pending_messages[gw.entry.session_key] is event

    def test_an_ordinary_turn_still_drains_its_follow_up_in_band(self, gw):
        adapter = TestLedgerSettlement()._adapter(gw)
        adapter._pending_messages[gw.entry.session_key] = self._queued(gw)
        pending_event, pending = asyncio.run(gw.runner._run_agent_next_followup(
            self._ctx(gw, None), {"messages": []}, adapter, gw.source, gw.entry.session_key))
        assert pending == "queued follow-up" and gw.entry.session_key not in adapter._pending_messages

    def test_a_queued_event_is_not_duplicated_by_interrupt_or_steer_text(self, gw):
        """SUPP-08: an event-backed interrupt leaves its event in the slot; the interrupt text is the
        same input and must not be appended again (nor a leftover steer, as the in-band drain does)."""
        adapter = TestLedgerSettlement()._adapter(gw)
        event = self._queued(gw, "followup-event")
        adapter._pending_messages[gw.entry.session_key] = event
        result = {"messages": [], "interrupted": True, "interrupt_message": "followup-event",
                  "pending_steer": "leftover steer"}
        asyncio.run(gw.runner._run_agent_next_followup(
            self._ctx(gw, _admitted(gw)), result, adapter, gw.source, gw.entry.session_key))
        assert adapter._pending_messages[gw.entry.session_key] is event and event.text == "followup-event"
        assert event._acp_deferred_followup is True

    def test_an_empty_slot_takes_the_interrupt_text_before_a_leftover_steer(self, gw):
        adapter = TestLedgerSettlement()._adapter(gw)
        result = {"messages": [], "interrupted": True, "interrupt_message": "change of plan",
                  "pending_steer": "leftover steer"}
        asyncio.run(gw.runner._run_agent_next_followup(
            self._ctx(gw, _admitted(gw)), result, adapter, gw.source, gw.entry.session_key))
        assert adapter._pending_messages[gw.entry.session_key].text == "change of plan"

    def test_a_bare_steer_alone_becomes_a_queued_event(self, gw):
        """SUPP-06: a leftover /steer without any event is not lost."""
        adapter = TestLedgerSettlement()._adapter(gw)
        asyncio.run(gw.runner._run_agent_next_followup(
            self._ctx(gw, _admitted(gw)), {"messages": [], "pending_steer": "leftover steer"}, adapter,
            gw.source, gw.entry.session_key))
        assert adapter._pending_messages[gw.entry.session_key].text == "leftover steer"

    def test_an_empty_slot_promotes_the_overflow_head_in_order(self, gw):
        adapter = TestLedgerSettlement()._adapter(gw)
        q2, q3 = self._queued(gw, "Q2"), self._queued(gw, "Q3")
        gw.runner._session_state(gw.entry.session_key).conversation.queued_events.extend([q2, q3])
        asyncio.run(gw.runner._run_agent_next_followup(
            self._ctx(gw, _admitted(gw)), {"messages": []}, adapter, gw.source, gw.entry.session_key))
        assert adapter._pending_messages[gw.entry.session_key] is q2
        assert gw.runner._overflow_queue(gw.entry.session_key) == [q3]

    def test_a_deferred_follow_up_runs_as_the_head_of_its_line(self, gw, monkeypatch):
        """SUPP-09: Q1 (deferred from the slot) runs before the newer Q2/Q3 still in the overflow;
        orphan rescue must not park it behind them."""
        from gateway.platforms.event import MessageEvent

        q1 = MessageEvent(text="Q1", source=gw.source, message_id="61", platform_update_id=61)
        q1._acp_deferred_followup = True
        gw.runner._session_state(gw.entry.session_key).conversation.queued_events.extend(
            [MessageEvent(text="Q2", source=gw.source, message_id="62"),
             MessageEvent(text="Q3", source=gw.source, message_id="63")])
        ran = []

        async def admitted(event):
            return event, event.source, False

        async def run_agent(event, source, key, generation):
            ran.append(event.text)
            return "answer"

        monkeypatch.setattr(gw.runner, "_hm_admit_event", admitted)
        monkeypatch.setattr(gw.runner, "_handle_message_with_agent", run_agent)
        # The rescue only fires with a live adapter whose slot for this session is empty: exactly the
        # state after the adapter's end-of-task drain popped Q1.
        gw.runner.adapters[gw.source.platform] = TestLedgerSettlement()._adapter(gw)
        asyncio.run(gw.runner._handle_message(q1))
        assert ran == ["Q1"]
        assert [e.text for e in gw.runner._overflow_queue(gw.entry.session_key)] == ["Q2", "Q3"]

    def test_a_crash_left_managed_marker_owes_nothing_even_with_later_rows(self, gw):
        _, store = _marked_turn(gw, answer=True)
        store.append_to_transcript(gw.entry.session_id, {"role": "user", "content": "something later"})
        store.append_to_transcript(gw.entry.session_id, {"role": "assistant", "content": "a later reply"})
        recovered = asyncio.run(gw.runner._recover_unclean_sessions())
        from gateway import delivery_ledger
        with delivery_ledger._connect() as conn:
            rows = conn.execute("SELECT content FROM delivery_obligations").fetchall()
        assert recovered == (0, 0) and rows == []
        assert not store._entries[gw.entry.session_key].resume_pending

    def test_a_withheld_managed_answer_is_never_sent_whatever_the_prompt_metadata(self, gw):
        """ESCAPE-SUPP-01: the gateway fallback writer's prompt row carries no acp_update_id."""
        _, store = _marked_turn(gw, answer=True)
        store.rewrite_transcript(gw.entry.session_id, [
            {**m, "display_metadata": {"gateway_input_owner": "fallback"}} if m.get("role") == "user" else m
            for m in store.load_transcript(gw.entry.session_id)])
        assert asyncio.run(gw.runner._recover_unclean_sessions()) == (0, 0)

    def test_an_earlier_turns_reply_is_never_adopted(self, gw):
        """ESCAPE-SUPP-03."""
        store = gw.runner.session_store
        store.append_to_transcript(gw.entry.session_id, {"role": "user", "content": "earlier question"})
        store.append_to_transcript(gw.entry.session_id, {"role": "assistant", "content": "earlier answer"})
        admission = _admitted(gw)
        assert ingress.bind_turn_marker(admission, store.mark_turn_active(gw.entry.session_key))
        assert asyncio.run(gw.runner._recover_unclean_sessions()) == (0, 0)

    def test_a_managed_row_never_deduplicates_an_ordinary_crash_left_reply(self, gw):
        """ESCAPE-SUPP-04, at the ledger: same text, different deliveries."""
        from gateway import delivery_ledger

        delivery_ledger.record_obligation(obligation_id="ob-managed", session_key=gw.entry.session_key,
                                          platform="telegram", chat_id="100200300", thread_id=None,
                                          content="done", acp_update_id="901")
        delivery_ledger.record_crash_left_reply(obligation_id="ob-ordinary", session_key=gw.entry.session_key,
                                                platform="telegram", chat_id="100200300", thread_id=None,
                                                content="done", since=0.0)
        with delivery_ledger._connect() as conn:
            ids = sorted(r[0] for r in conn.execute("SELECT obligation_id FROM delivery_obligations"))
        assert ids == ["ob-managed", "ob-ordinary"]
