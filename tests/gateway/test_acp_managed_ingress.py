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
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1718881034", chat_type="dm", user_id="1718881034")
    entry = runner.session_store.get_or_create_session(source)
    binding = CanonicalSurfaceBinding(
        name="acp-canonical-ceo", session_key=entry.session_key, session_id=entry.session_id,
        telegram_chat_id="1718881034", telegram_chat_type="dm", telegram_user_id="1718881034",
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
        "message_id": 55, "from": {"id": 1718881034}, "chat": {"id": 1718881034, "type": "private"},
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
                                          "agent:other:telegram:dm:1718881034", path=gw.sock, secret="s")
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

    def test_an_undelivered_answer_waits_for_redelivery_then_completes(self, gw, monkeypatch):
        from gateway import delivery_ledger

        admission = _admitted(gw)
        asyncio.run(self._adapter(gw, succeed=False).send_final_ledgered(
            self._managed_event(gw, admission), gw.entry.session_key, "the answer", {}, reply_to="55"))
        monkeypatch.setattr(ingress, "_PROCESS_OWNER", "a-later-process")
        assert ingress.sweep_at_startup(gw.runner) == []  # owed, not dead
        assert receipts.lookup(gw.db, 901).status == "PENDING"
        [row] = receipts.ledger_answers(gw.db, 901)
        delivery_ledger.mark_delivered(row["obligation_id"])
        assert ingress.settle_delivered(gw.runner, row["obligation_id"])
        assert receipts.lookup(gw.db, 901).receipt_id == f"hermes-tg:{row['obligation_id']}"

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
                                          platform="telegram", chat_id="1718881034", thread_id=None,
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

    @pytest.mark.asyncio
    async def test_text_before_a_task_is_its_own_update_and_a_split_tail_joins_the_task(self, gw):
        """REG1: a >4096-char /acp paste arrives as a near-limit chunk plus continuations; the tail
        must join the task, and ordinary text batched just before it must not."""
        adapter = self._adapter(gw)
        adapter._text_batch_delay_seconds = adapter._text_batch_split_delay_seconds = 0.05
        head = "/acp " + "x" * 4095
        adapter._enqueue_text_event(self._msg(gw, "hello", 10))
        adapter._enqueue_text_event(self._msg(gw, head, 11))
        adapter._enqueue_text_event(self._msg(gw, "TAIL", 12))
        for _ in range(100):
            if len(adapter.dispatched) == 2:
                break
            await asyncio.sleep(0.02)
        assert sorted(adapter.dispatched, key=len) == ["hello", head + "\nTAIL"]


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
