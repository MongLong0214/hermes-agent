"""U4 H2/H3 — ``/acp <task>`` admission through ACP's ``telegram-update.ingress.sock`` lane.

A real unix socket stands in for ACP #1062's lane (one NDJSON envelope in, one answer line out);
the runner, session store, binding and SessionDB are real under a temp home. CEO 36ade192: only the
explicit prefix is managed; a deny, a timeout or an unreadable answer never runs the turn and never
retries; a late or duplicate delivery never runs it twice; a failed settlement stays in doubt.

The core subset drives ``gateway.acp_managed_ingress``, ``gateway.acp_turn_receipts`` and
``gateway.delivery_ledger`` directly. The seam subset below goes through the code that calls them:
the adapter's batching, busy route and ledgered final send, the runner's ``_handle_message``, and
startup (the receipt sweep, the boot redelivery, crash-left recovery). Answer certification brings
its own tests with it.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import acp_managed_ingress as ingress
from gateway import acp_turn_receipts as receipts
from gateway import delivery_ledger
from gateway.canonical_surface import CanonicalSurfaceBinding
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tests.waiting import async_wait_until

_TURN = {
    "turnRequestId": "turn-1", "targetActorId": "actor-1", "promptDigest": "sha256:p",
    "bindingGeneration": 4, "targetBindingId": "bind-1", "targetAttestationId": "att-1",
    "executorSessionId": "ses-1", "executorSessionIncarnation": "inc-1",
}
_WIRE_TURN = {
    "turnRequestId": "turn-wire-1", "targetActorId": "actor-ceo", "promptDigest": "sha256:" + "ab" * 32,
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


def _allowed(gw, *, lineage=None, turn=None):
    def answer(envelope):
        return {"allowed": True, "replayed": False, "turn": dict(turn or _TURN),
                "source": {"channel": "telegram", "nonce": f"update:{envelope['update']['update_id']}"},
                "targetBind": {"requested_session_id": gw.entry.session_id,
                               "lineage_root_digest": lineage or gw.lineage}}
    return answer


def _admit(gw, event, **kw):
    return ingress.admit(gw.runner, event, gw.source, gw.entry.session_key,
                         path=gw.sock, secret="lane-secret", **kw)


def _admit_on_lane(gw, event, answer=None, **kw):
    """(outcome, envelopes ACP's lane received) for one admission."""
    async def exercise():
        async with _Lane(gw.sock, answer or _allowed(gw)) as lane:
            outcome = await _admit(gw, event, **kw)
        return outcome, lane.envelopes
    return asyncio.run(exercise())


def _admitted(gw, text="/acp deploy", *, update_id=901, turn=None):
    outcome, _ = _admit_on_lane(gw, _event(text, update_id=update_id), _allowed(gw, turn=turn))
    assert outcome.admission is not None
    return outcome.admission


def _proof(gw):
    return {"proven_db_path": gw.db.db_path, "proven_db_identity": gw.db._db_file_identity}


def _managed_state(gw):
    """(turn markers, receipts, ACP-correlated ledger rows) in the launch-home store."""
    markers = gw.db._read_all("SELECT key FROM state_meta WHERE key LIKE 'acp_turn_marker:%'")
    receipt_rows = gw.db._read_all("SELECT key FROM state_meta WHERE key LIKE 'acp-tg-receipt%'")
    try:
        acp_rows = gw.db._read_all(
            "SELECT obligation_id FROM delivery_obligations WHERE acp_update_id IS NOT NULL")
    except sqlite3.OperationalError as exc:
        assert "no such table" in str(exc)
        acp_rows = []
    return len(markers), len(receipt_rows), len(acp_rows)


def _ledger_ready():
    """The gateway's startup ledger sweep opens the ledger (creating and migrating its schema)
    before any admission; a test home has to do the same."""
    assert delivery_ledger.sweep_recoverable(deliverable_platforms=set()) == []


def _ledger_row(gw, obligation_id, update_id, state, *, content="an answer", age=0.0, owner=None):
    """One ledger row, recorded by the real INSERT and then moved to *state*, *age* seconds old."""
    delivery_ledger.record_obligation(obligation_id=obligation_id, session_key=gw.entry.session_key,
                                      platform="telegram", chat_id="100200300", thread_id=None,
                                      content=content,
                                      acp_update_id=str(update_id) if update_id is not None else None)
    stamp = time.time() - age
    pid, started = owner if owner is not None else delivery_ledger._owner_stamp()
    conn = delivery_ledger._connect()
    try:
        conn.execute("UPDATE delivery_obligations SET state = ?, created_at = ?, updated_at = ?, owner_pid = ?,"
                     " owner_started_at = ? WHERE obligation_id = ?",
                     (state, stamp, stamp, pid, started, obligation_id))
        conn.commit()
    finally:
        conn.close()


def _ledger_ids():
    conn = delivery_ledger._connect()
    try:
        return {row[0]: row[1] for row in conn.execute("SELECT obligation_id, acp_update_id FROM delivery_obligations")}
    finally:
        conn.close()


def test_only_an_explicit_prefix_on_the_bound_chat_is_managed(gw):
    assert ingress.is_managed(gw.runner, _event("/acp deploy the fix"), gw.source)
    assert not ingress.is_managed(gw.runner, _event("deploy the fix"), gw.source)
    # A bare /acp on the bound chat is the managed path's to refuse (empty task), never the generic
    # slash dispatch's "unknown command".
    assert ingress.is_managed(gw.runner, _event("/acp"), gw.source)
    other = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42")
    assert not ingress.is_managed(gw.runner, _event("/acp deploy"), other)
    non_owner = SessionSource(platform=Platform.TELEGRAM, chat_id="100200300", chat_type="dm", user_id="42")
    assert not ingress.is_managed(gw.runner, _event("/acp deploy"), non_owner)


def test_an_allowed_admission_claims_the_receipt_and_hands_back_the_task(gw):
    outcome, envelopes = _admit_on_lane(gw, _event("/acp deploy the fix"))
    assert outcome.reply is None and outcome.admission.task_text == "deploy the fix"
    [envelope] = envelopes
    assert envelope["schema"] == "acp.telegram-external-update/v1" and envelope["binding"] == "acp-canonical-ceo"
    assert envelope["update"] == {"update_id": 901, "message": {
        "message_id": 55, "from": {"id": 100200300}, "chat": {"id": 100200300, "type": "private"},
        "text": "/acp deploy the fix"}}
    receipt = receipts.lookup(gw.db, 901)
    assert receipt.status == "PENDING" and receipt.receipt_identity == _TURN


def test_a_denied_admission_does_not_run_and_writes_nothing(gw):
    outcome, _ = _admit_on_lane(gw, _event("/acp deploy"),
                                lambda e: {"allowed": False, "reasonCode": "OWNER_NOT_ALLOWED", "message": "x"})
    assert outcome.admission is None and "OWNER_NOT_ALLOWED" in outcome.reply
    assert receipts.lookup(gw.db, 901).status == "NEVER_FOUND"


def test_a_timed_out_admission_does_not_run_retry_or_tombstone(gw):
    """ACP may have committed the claim before its answer was lost: the receipt stays absent so
    ACP keeps the turn in doubt; nothing here claims a non-run it cannot prove (CEO 3c058be8)."""
    outcome, envelopes = _admit_on_lane(gw, _event("/acp deploy"), lambda e: None, timeout=0.3)
    assert outcome.admission is None and outcome.reply == ingress.UNKNOWN
    assert len(envelopes) == 1  # asked once, never retried
    assert receipts.lookup(gw.db, 901).status == "NEVER_FOUND"


def test_an_unreachable_lane_or_missing_secret_is_unknown_and_never_runs(gw):
    outcome = asyncio.run(_admit(gw, _event("/acp deploy")))  # no server on the socket
    assert outcome.admission is None and outcome.reply == ingress.UNKNOWN
    no_secret = asyncio.run(ingress.admit(gw.runner, _event("/acp deploy"), gw.source,
                                          gw.entry.session_key, path=gw.sock, secret=""))
    assert no_secret.admission is None and no_secret.reply == ingress.UNKNOWN
    assert receipts.lookup(gw.db, 901).status == "NEVER_FOUND"


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
    assert second.reply == ingress.DUPLICATE.format(status="PENDING")
    # The first admission's claim stands: owner and identity are the first claimant's.
    stored = json.loads(gw.db.get_meta(receipts.receipt_key(901)))
    assert stored["owner"] == ingress.process_owner() and stored["receipt_identity"] == _TURN


def test_a_lineage_mismatch_refuses_and_records_the_definite_non_run(gw):
    outcome, _ = _admit_on_lane(gw, _event("/acp deploy"), _allowed(gw, lineage="sha256:" + "0" * 64))
    assert outcome.admission is None
    receipt = receipts.lookup(gw.db, 901)
    assert receipt.status == "ABORTED" and receipt.reason_code == "REFUSED_BEFORE_RUN"


def test_a_busy_session_refuses_and_records_the_definite_non_run(gw, monkeypatch):
    monkeypatch.setattr(gw.runner, "_is_session_running", lambda key: key == gw.entry.session_key)
    outcome, envelopes = _admit_on_lane(gw, _event("/acp deploy"))
    assert outcome.admission is None and outcome.reply == ingress.BUSY and len(envelopes) == 1
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
    outcome, envelopes = _admit_on_lane(gw, _event("/acp deploy"))
    assert outcome.admission is None and "ledger" in outcome.reply and envelopes == []


def test_a_split_length_task_is_refused_before_acp_is_asked(gw):
    outcome, envelopes = _admit_on_lane(gw, _event("/acp " + "x" * 3995))
    assert outcome.reply == ingress.TOO_LONG and outcome.admission is None and envelopes == []
    assert receipts.lookup(gw.db, 901).status == "NEVER_FOUND"
    assert _admitted(gw, "/acp " + "x" * 3990, update_id=902).task_text == "x" * 3990


class TestManagedAdmissionGate:
    """``gateway.acp_managed_admission``: closed, an /acp message is refused before anything else
    happens to it — no ACP request, no receipt, no turn marker, no ACP-correlated ledger row. Absent
    or true, managed admission is unchanged. The gate is re-read per message (no restart)."""

    _mtime = [1_900_000_000]

    def _set_gate(self, value):
        import os

        path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
        if value is None:
            path.unlink(missing_ok=True)
            return
        path.write_text(f"gateway:\n  acp_managed_admission: {value}\n", encoding="utf-8")
        self._mtime[0] += 10  # a distinct file signature, so the config cache re-reads it
        os.utime(path, (self._mtime[0], self._mtime[0]))

    @pytest.mark.parametrize("text", ["/acp deploy the fix", "/acp", "/acp " + "x" * 3995])
    def test_a_closed_gate_refuses_before_any_managed_write(self, gw, text):
        self._set_gate("false")
        outcome, envelopes = _admit_on_lane(gw, _event(text))
        assert outcome.admission is None and outcome.reply == ingress.PAUSED
        assert "managed admission paused" in outcome.reply and envelopes == []
        assert _managed_state(gw) == (0, 0, 0)
        assert receipts.lookup(gw.db, 901).status == "NEVER_FOUND"

    @pytest.mark.parametrize("value,enabled", [
        ("off", False), ("false", False), ("0", False), ("no", False), (" OFF ", False), ("False", False),
        ("on", True), ("true", True), ("yes", True), (False, False), (True, True)], ids=repr)
    def test_a_quoted_gate_value_is_read_as_its_word(self, value, enabled):
        """YAML reads a bare ``off`` as a boolean; a quoted one arrives as the string and must close the gate too."""
        assert ingress.admission_enabled({"gateway": {"acp_managed_admission": value}}) is enabled

    @pytest.mark.parametrize("gate", [None, "true"])
    def test_an_open_gate_keeps_managed_admission_unchanged(self, gw, gate):
        self._set_gate(gate)
        outcome, envelopes = _admit_on_lane(gw, _event("/acp deploy the fix"))
        assert outcome.admission is not None and len(envelopes) == 1
        assert receipts.lookup(gw.db, 901).status == "PENDING"

    def test_the_gate_is_read_per_message(self, gw):
        async def exercise():
            outcomes = []
            async with _Lane(gw.sock, _allowed(gw)) as lane:
                for gate, update_id in (("false", 901), ("true", 902), ("off", 903)):
                    self._set_gate(gate)
                    outcomes.append(await _admit(gw, _event("/acp go", update_id=update_id)))
            return outcomes, lane.envelopes

        outcomes, envelopes = asyncio.run(exercise())
        assert [o.reply for o in outcomes] == [ingress.PAUSED, None, ingress.PAUSED]
        assert [e["update"]["update_id"] for e in envelopes] == [902]
        assert [receipts.lookup(gw.db, u).status for u in (901, 902, 903)] == [
            "NEVER_FOUND", "PENDING", "NEVER_FOUND"]


_OWN_BOT = "Hermes_CEO_Bot"


class TestCommandShapes:
    """Production 2026-10-09: on the bound chat from the owner, any message whose command token is
    ``/acp`` — followed by whitespace or the end, or by ``@<this bot>`` then whitespace or the end —
    is the managed path's alone: admitted on its task text, or refused (empty task, closed gate)."""

    ADMITTED = [
        ("/acp\ndeploy the fix", "deploy the fix"),
        ("/acp\tdeploy the fix", "deploy the fix"),
        (f"/acp@{_OWN_BOT} deploy the fix", "deploy the fix"),
        (f"/acp@{_OWN_BOT.lower()}\ndeploy the fix", "deploy the fix"),
        ("/acp  deploy the fix", "deploy the fix"),
        ("/acp\n\ndeploy\nthe fix\n", "deploy\nthe fix"),
        ("/acp deploy the fix", "deploy the fix"),  # control: unchanged
    ]
    EMPTY = ["/acp", "/acp   ", "/acp\n", "/acp\t \n", f"/acp@{_OWN_BOT}", f"/acp@{_OWN_BOT.upper()}  "]
    NOT_OURS = ["/acpx deploy the fix", "/acp@other_bot deploy the fix", "/acp@ deploy the fix",
                f"/acp@{_OWN_BOT}x deploy the fix", "/acp/deploy the fix", "/ACP deploy the fix"]

    @staticmethod
    def _receiver(gw, monkeypatch, username=_OWN_BOT):
        receiver = SimpleNamespace(_current_bot_username=lambda: (username or "").lower())
        monkeypatch.setattr(gw.runner, "_intake_adapter_for", lambda source: receiver)

    def test_one_classifier_decides_every_shape(self):
        for text, task in self.ADMITTED:
            assert receipts.acp_command_task(text, _OWN_BOT) == task
            assert receipts.acp_managed_task_text(text, _OWN_BOT) == task
            assert receipts.is_acp_managed_message(SimpleNamespace(text=text), None, _OWN_BOT)
        for text in self.EMPTY:
            assert receipts.acp_command_task(text, _OWN_BOT) == ""
            assert receipts.acp_managed_task_text(text, _OWN_BOT) is None
            assert receipts.is_acp_managed_message(SimpleNamespace(text=text), None, _OWN_BOT)
        for text in self.NOT_OURS + [" /acp deploy", "can you use /acp for this?", "", None]:
            assert receipts.acp_command_task(text, _OWN_BOT) is None
            assert not receipts.is_acp_managed_message(SimpleNamespace(text=text), None, _OWN_BOT)
        # The mention form is ours only when this bot's own handle is known.
        assert receipts.acp_command_task(f"/acp@{_OWN_BOT} deploy", None) is None

    @pytest.mark.parametrize("text,task", ADMITTED, ids=lambda v: repr(v))
    def test_every_acp_command_shape_is_admitted_on_its_task_text(self, gw, monkeypatch, text, task):
        self._receiver(gw, monkeypatch)
        assert ingress.is_managed(gw.runner, _event(text), gw.source)
        outcome, [envelope] = _admit_on_lane(gw, _event(text))
        assert outcome.admission.task_text == task
        assert envelope["update"]["message"]["text"] == text  # ACP is shown the message as sent
        assert receipts.lookup(gw.db, 901).status == "PENDING"

    @pytest.mark.parametrize("text", EMPTY, ids=lambda v: repr(v))
    def test_admit_refuses_an_empty_task_before_asking_acp(self, gw, monkeypatch, text):
        self._receiver(gw, monkeypatch)
        assert ingress.is_managed(gw.runner, _event(text), gw.source)
        outcome, envelopes = _admit_on_lane(gw, _event(text))
        assert outcome.reply == ingress.EMPTY_TASK and outcome.admission is None and envelopes == []
        assert _managed_state(gw) == (0, 0, 0)

    @pytest.mark.parametrize("text", NOT_OURS, ids=lambda v: repr(v))
    def test_other_commands_and_other_bots_are_not_managed(self, gw, monkeypatch, text):
        self._receiver(gw, monkeypatch)
        assert not ingress.is_managed(gw.runner, _event(text), gw.source)

    def test_the_mention_form_needs_this_bots_known_handle(self, gw, monkeypatch):
        self._receiver(gw, monkeypatch, username=None)
        assert not ingress.is_managed(gw.runner, _event(f"/acp@{_OWN_BOT} deploy"), gw.source)
        assert ingress.is_managed(gw.runner, _event("/acp\ndeploy"), gw.source)


class TestStore:
    """U4-05: receipts and the delivery ledger share one store, the launch home's state.db. A
    binding whose session a named profile owns is not managed at all."""

    def test_a_named_profile_binding_is_not_managed(self, gw, monkeypatch):
        assert ingress.bound_binding(gw.runner, gw.source) is not None
        monkeypatch.setattr(gw.runner.session_store, "_named_profile_for_key", lambda key: "work")
        assert ingress.bound_binding(gw.runner, gw.source) is None

    def test_a_store_other_than_the_ledgers_is_refused_unasked(self, gw, monkeypatch, tmp_path):
        monkeypatch.setattr(delivery_ledger, "_db_path", lambda: tmp_path / "elsewhere.db")
        outcome, envelopes = _admit_on_lane(gw, _event("/acp deploy"))
        assert outcome.admission is None and "launch home" in outcome.reply and envelopes == []


class TestTurnMarker:
    """The durable active-turn marker that names a turn as an admitted /acp task, read back from the
    session's own store (never from the current binding configuration)."""

    def test_a_bound_marker_reads_managed_and_an_unbound_one_does_not(self, gw):
        admission = _admitted(gw)
        assert ingress.bind_turn_marker(admission, "tok-bound")
        assert ingress.bind_turn_marker(admission, "tok-bound") is False  # bound once
        assert ingress.crash_left_turn_is_managed(gw.runner, gw.entry.session_key, "tok-bound") is True
        assert ingress.crash_left_turn_is_managed(gw.runner, gw.entry.session_key, "tok-other") is False
        gw.runner.config.canonical_surface_bindings = {}  # an edited binding never un-manages a marked turn
        assert ingress.crash_left_turn_is_managed(gw.runner, gw.entry.session_key, "tok-bound") is True

    def test_an_unreadable_marker_store_is_unknown(self, gw, monkeypatch):
        monkeypatch.setattr(type(gw.db), "get_meta", lambda self, key: (_ for _ in ()).throw(OSError("disk")))
        assert ingress.crash_left_turn_is_managed(gw.runner, gw.entry.session_key, "tok") is None

    def test_another_session_is_never_treated_as_managed(self, gw):
        assert ingress.crash_left_turn_is_managed(gw.runner, "agent:main:telegram:dm:5", "tok") is False

    def test_a_marker_that_cannot_be_bound_refuses_the_turn(self, gw, monkeypatch):
        admission = _admitted(gw)
        monkeypatch.setattr(type(gw.db), "claim_meta_once",
                            lambda self, *a, **k: (_ for _ in ()).throw(OSError("disk")))
        assert ingress.bind_turn_marker(admission, "tok") is False
        assert ingress.bind_turn_marker(admission, None) is False


class TestLedgerSettlement:
    """R2/R3 through the real delivery ledger (same state.db): the receipt settles from the answer
    row that was actually delivered, and no settlement-write failure, crash or upgrade can turn an
    answered update into a death before answer."""

    @staticmethod
    def _deliver(obligation_id, *, content="the answer", message_ids=("77",), update_id="901"):
        delivery_ledger.record_obligation(obligation_id=obligation_id, session_key="agent:main:telegram:dm:100200300",
                                          platform="telegram", chat_id="100200300", thread_id=None,
                                          content=content, acp_update_id=update_id)
        delivery_ledger.mark_attempting(obligation_id)
        assert delivery_ledger.mark_delivered(obligation_id, message_ids=list(message_ids) if message_ids else None)

    def test_the_delivered_row_is_the_evidence_in_the_closed_six_field_shape(self, gw):
        import hashlib

        _admitted(gw)
        self._deliver("ob-1", content="long answer", message_ids=("301", "302", "303"))
        assert ingress.settle_delivered(gw.runner, "ob-1") is True
        receipt = receipts.lookup(gw.db, 901)
        digest = "sha256:" + hashlib.sha256(b"long answer").hexdigest()
        assert receipt.status == "COMPLETED" and receipt.content == "long answer"
        assert receipt.evidence_digest == digest and receipt.receipt_id == "hermes-tg:ob-1"
        assert receipt.receipt_identity == _TURN
        assert receipt.delivery == {"obligation_id": "ob-1", "state": "delivered", "content_digest": digest,
                                    "chat_id": 100200300, "reply_to_message_id": 55,
                                    "message_ids": [301, 302, 303]}

    def test_an_ordinary_answer_row_settles_nothing(self, gw):
        _admitted(gw)
        self._deliver("ob-ordinary", update_id=None)
        assert ingress.settle_delivered(gw.runner, "ob-ordinary") is False
        assert receipts.lookup(gw.db, 901).status == "PENDING"

    def test_a_failed_settlement_write_still_reads_completed_and_is_settled_at_startup(self, gw, monkeypatch):
        _admitted(gw)
        self._deliver("ob-1")
        with monkeypatch.context() as broken:
            broken.setattr(receipts, "_settle", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
            with pytest.raises(OSError):
                ingress.settle_delivered(gw.runner, "ob-1")
        assert receipts.decode(gw.db.get_meta(receipts.receipt_key(901)), 901).status == "PENDING"
        assert receipts.lookup(gw.db, 901).status == "COMPLETED"  # GET reads the delivered row
        monkeypatch.setattr(ingress, "_PROCESS_OWNER", "a-later-process")
        assert ingress.sweep_at_startup(gw.runner) == []
        assert receipts.decode(gw.db.get_meta(receipts.receipt_key(901)), 901).status == "COMPLETED"

    def test_a_delivery_without_message_ids_stays_in_doubt_and_is_never_aborted(self, gw, monkeypatch):
        _admitted(gw)
        self._deliver("ob-1", message_ids=())
        assert ingress.settle_delivered(gw.runner, "ob-1") is False
        assert receipts.lookup(gw.db, 901).status == "PENDING"
        monkeypatch.setattr(ingress, "_PROCESS_OWNER", "a-later-process")
        assert ingress.sweep_at_startup(gw.runner) == []
        assert receipts.lookup(gw.db, 901).status == "PENDING"

    def test_a_later_checkpoint_keeps_the_recorded_message_ids(self, gw):
        self._deliver("ob-1", message_ids=("301", "302"))
        delivery_ledger._update_state("ob-1", "delivered")  # a checkpoint without ids keeps the evidence
        [row] = receipts.ledger_answers(gw.db, 901)
        assert json.loads(row["message_ids"]) == ["301", "302"]

    def test_an_earlier_recorded_send_keeps_a_failed_turn_in_doubt(self, gw):
        admission = _admitted(gw)
        _ledger_row(gw, "ob-1", 901, "failed")
        assert ingress.abort_unrecorded(admission) is False
        assert ingress.abort_failed_turn(admission) is False
        assert receipts.lookup(gw.db, 901).status == "PENDING"

    def test_only_an_update_with_no_ledgered_answer_is_a_death_before_answer(self, gw):
        proof = _proof(gw)
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


class TestAcpRowsInTheLedger:
    """U4-04: an /acp answer row is never redelivered by an automatic sweep (recovery sends without
    the reply anchor and with recovery decoration, which the receipt could not truthfully certify) and
    is never pruned while its receipt may still read it, while an ordinary row beside it keeps every
    sweep and prune it had."""

    _DEAD = (999_999_999, 1)

    @staticmethod
    def _live_shaped_store():
        """The live store's table, created before this build's ledger first opens it."""
        conn = sqlite3.connect(delivery_ledger._db_path())
        try:
            conn.execute(
                """CREATE TABLE delivery_obligations (
                    obligation_id TEXT PRIMARY KEY, session_key TEXT NOT NULL, platform TEXT NOT NULL,
                    chat_id TEXT NOT NULL, thread_id TEXT, content TEXT NOT NULL, state TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    owner_pid INTEGER, owner_started_at INTEGER, last_error TEXT, adapter_profile TEXT,
                    acp_update_id TEXT, delivered_message_ids TEXT)""")
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row(obligation_id, update_id, state, *, age, owner, attempts=1):
        conn = sqlite3.connect(delivery_ledger._db_path())
        try:
            stamp = time.time() - age
            conn.execute(
                "INSERT INTO delivery_obligations VALUES (?, 'agent:main:telegram:dm:100200300', 'telegram',"
                " '100200300', NULL, 'an answer', ?, ?, ?, ?, ?, ?, 'boom', 'default', ?, NULL)",
                (obligation_id, state, attempts, stamp, stamp, owner[0], owner[1], update_id))
            conn.commit()
        finally:
            conn.close()

    def test_no_sweep_redelivers_an_acp_row(self, gw, monkeypatch):
        self._live_shaped_store()
        own = delivery_ledger._owner_stamp()
        self._row("acp-dead", "901", "failed", age=600, owner=self._DEAD)
        self._row("acp-pending", "902", "pending", age=600, owner=self._DEAD, attempts=0)
        self._row("acp-own", "903", "failed", age=600, owner=own)
        assert delivery_ledger.pending_retries() == []  # no timer for a managed row
        assert delivery_ledger.sweep_failed_for_runtime("telegram") == []
        assert delivery_ledger.sweep_recoverable() == []
        # An ordinary row beside each one is still claimed by the same sweeps.
        self._row("ob-own", None, "failed", age=600, owner=own)
        assert [r["platform"] for r in delivery_ledger.pending_retries()] == ["telegram"]
        assert [r["obligation_id"] for r in delivery_ledger.sweep_failed_for_runtime("telegram")] == ["ob-own"]
        self._row("ob-dead", None, "failed", age=600, owner=self._DEAD)
        assert [r["obligation_id"] for r in delivery_ledger.sweep_recoverable()] == ["ob-dead"]
        conn = sqlite3.connect(delivery_ledger._db_path())
        try:
            states = dict(conn.execute(
                "SELECT obligation_id, state FROM delivery_obligations WHERE acp_update_id IS NOT NULL"))
        finally:
            conn.close()
        assert states == {"acp-dead": "failed", "acp-pending": "pending", "acp-own": "failed"}

    def test_an_acp_row_outlives_the_retention_until_ninety_days(self, gw):
        day = 24 * 60 * 60
        _ledger_row(gw, "acp-8d", 901, "delivered", age=8 * day)
        _ledger_row(gw, "acp-8d-abandoned", 902, "abandoned", age=8 * day)
        _ledger_row(gw, "acp-91d", 903, "delivered", age=91 * day)
        _ledger_row(gw, "ob-8d", None, "delivered", age=8 * day)
        _ledger_row(gw, "ob-1d", None, "delivered", age=1 * day)
        _ledger_row(gw, "ob-new", None, "pending")  # this INSERT runs the retention prune
        assert _ledger_ids() == {"acp-8d": "901", "acp-8d-abandoned": "902", "ob-1d": None, "ob-new": None}

    def test_acp_rows_neither_count_toward_nor_are_cut_by_the_row_cap(self, gw, monkeypatch):
        monkeypatch.setattr(delivery_ledger, "_MAX_ROWS", 2)
        for n in range(3):  # the oldest delivered rows: first in line for the cap
            _ledger_row(gw, f"acp-{n}", 900 + n, "delivered", age=3600 * (10 + n))
        for n in range(3):
            _ledger_row(gw, f"ob-{n}", None, "delivered", age=60 * (n + 1))
        _ledger_row(gw, "ob-new", None, "pending")  # this INSERT runs the cap
        ids = _ledger_ids()
        assert sorted(k for k, v in ids.items() if v is not None) == ["acp-0", "acp-1", "acp-2"]
        assert sorted(k for k, v in ids.items() if v is None) == ["ob-0", "ob-new"]

    def test_a_crash_left_reply_is_never_deduplicated_against_an_acp_answer(self, gw):
        since = time.time() - 60
        _ledger_row(gw, "acp-1", 901, "delivered", content="same text")
        delivery_ledger.record_crash_left_reply(obligation_id="ob-crash", session_key=gw.entry.session_key,
                                                platform="telegram", chat_id="100200300", thread_id=None,
                                                content="same text", since=since)
        assert _ledger_ids() == {"acp-1": "901", "ob-crash": None}
        delivery_ledger.record_crash_left_reply(obligation_id="ob-crash-2", session_key=gw.entry.session_key,
                                                platform="telegram", chat_id="100200300", thread_id=None,
                                                content="same text", since=since)
        assert "ob-crash-2" not in _ledger_ids()  # an ordinary same-text row still deduplicates


class TestCrashAfterClaim:
    """H3: a receipt claimed by a process that died before answering is settled as a definite
    non-run at the next start, so ACP never waits on a turn that no longer exists."""

    def test_a_claim_left_by_a_dead_process_is_a_definite_non_run(self, gw, monkeypatch):
        _ledger_ready()
        _admitted(gw, turn=_WIRE_TURN)
        assert ingress.sweep_at_startup(gw.runner) == []  # this process may still answer it
        assert receipts.lookup(gw.db, 901).status == "PENDING"
        monkeypatch.setattr(ingress, "_PROCESS_OWNER", "a-later-process")
        assert ingress.sweep_at_startup(gw.runner) == ["901"]
        receipt = receipts.served(gw.db, 901)
        assert receipt.status == "ABORTED" and receipt.reason_code == "HERMES_PROCESS_DIED_BEFORE_ANSWER"
        assert receipt.receipt_id == "hermes-tg:aborted:901" and receipt.evidence_digest.startswith("sha256:")
        assert ingress.sweep_at_startup(gw.runner) == []  # terminal: never touched again

    def test_a_dead_claim_whose_answer_is_still_owed_is_left_to_its_redelivery(self, gw, monkeypatch):
        _admitted(gw)
        _ledger_row(gw, "acp-1", 901, "failed")
        monkeypatch.setattr(ingress, "_PROCESS_OWNER", "a-later-process")
        assert ingress.sweep_at_startup(gw.runner) == []
        assert receipts.lookup(gw.db, 901).status == "PENDING"


_MISSING = object()


class TestAbortedReceipt:
    """An ABORTED receipt is served with the receipt id and evidence digest ACP compares, computed
    from what the store preserved, or refused explicitly when that evidence is not whole."""

    def _abort(self, gw, code, update_id=901):
        """Settle one admitted update ABORTED through the real path that records *code*."""
        _ledger_ready()
        admission = _admitted(gw, "/acp deploy the fix", update_id=update_id, turn=_WIRE_TURN)
        if code == "REFUSED_BEFORE_RUN":
            ingress.abort_claimed(admission)
        elif code == "HERMES_TURN_FAILED":
            assert ingress.abort_failed_turn(admission)
        elif code == "HERMES_ANSWER_NOT_RECORDED":
            assert ingress.abort_unrecorded(admission)
        else:  # the startup sweep: the claiming process died
            if code == "HERMES_ANSWER_UNDELIVERABLE":  # an answer recorded, then abandoned undelivered
                _ledger_row(gw, f"abandoned-{update_id}", update_id, "abandoned")
            assert receipts.sweep_dead_owner_receipts(gw.db, resuming_owners=(), **_proof(gw)) == [str(update_id)]
        receipt = receipts.lookup(gw.db, update_id)
        assert receipt.status == "ABORTED" and receipt.reason_code == code

    @pytest.mark.parametrize("stored,wire", [
        ("901", 901), (str(2**53 - 1), 2**53 - 1), (55, 55),
        (str(2**53), str(2**53)), ("0901", "0901"), ("-5", "-5"), ("1.5", "1.5"), ("", ""), (" 9", " 9"),
        ("m1", "m1"), ("٣", "٣"), ("0", "0"), (None, None), (True, True)], ids=lambda v: repr(v))
    def test_only_a_positive_safe_integer_is_sent_as_a_number(self, stored, wire):
        body = receipts.TelegramTurnReceipt(status="COMPLETED", update_id=stored, message_id=stored).to_response()
        assert body["update_id"] == wire and body["message_id"] == wire
        assert type(body["update_id"]) is type(wire)

    def test_aborted_keeps_its_failure_code_and_pending_keeps_no_code(self):
        aborted = receipts.TelegramTurnReceipt(status="ABORTED", update_id="901", reason_code="REFUSED_BEFORE_RUN")
        assert aborted.to_response()["reasonCode"] == "REFUSED_BEFORE_RUN"
        assert receipts.TelegramTurnReceipt(status="PENDING", update_id="901").to_response()["reasonCode"] is None
        assert receipts.not_found("901").to_response()["reasonCode"] is None

    @pytest.mark.parametrize("code", ["REFUSED_BEFORE_RUN", "HERMES_TURN_FAILED", "HERMES_ANSWER_NOT_RECORDED",
                                      "HERMES_PROCESS_DIED_BEFORE_ANSWER", "HERMES_ANSWER_UNDELIVERABLE"])
    def test_every_abort_path_serves_a_stable_receipt(self, gw, code):
        self._abort(gw, code)
        stored = gw.db.get_meta(receipts.receipt_key(901))
        first, second = receipts.served(gw.db, 901), receipts.served(gw.db, 901)
        assert first.to_response() == second.to_response()  # stable: no clock, no randomness
        assert gw.db.get_meta(receipts.receipt_key(901)) == stored  # served from storage, never rewritten
        body = first.to_response()
        assert body["receiptId"] == "hermes-tg:aborted:901" and body["delivery"] is None
        assert body["update_id"] == 901 and body["message_id"] == 55 and body["reasonCode"] == code
        assert body["receiptIdentity"] == _WIRE_TURN

    def test_the_evidence_digest_binds_exactly_the_preserved_fields(self, gw):
        import dataclasses
        import hashlib

        self._abort(gw, "HERMES_TURN_FAILED")
        receipt = receipts.lookup(gw.db, 901)
        wire = receipts.aborted_on_the_wire(gw.db, receipt)
        preserved = {"schema": "hermes.gateway-turn-receipt.aborted-evidence/v1", "update_id": 901,
                     "message_id": "55", "turnRequestId": "turn-wire-1", "receiptIdentity": _WIRE_TURN,
                     "reasonCode": "HERMES_TURN_FAILED", "deliveredAnswer": False}
        canonical = json.dumps(preserved, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        assert wire.evidence_digest == "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert wire.receipt_id == "hermes-tg:aborted:901" and wire.delivery is None
        changed = [
            dataclasses.replace(receipt, reason_code="REFUSED_BEFORE_RUN"),
            dataclasses.replace(receipt, message_id="56"),
            dataclasses.replace(receipt, receipt_identity={**_WIRE_TURN, "executorSessionIncarnation": "inc-2"}),
            dataclasses.replace(receipt, turn_request_id="turn-wire-2",
                                receipt_identity={**_WIRE_TURN, "turnRequestId": "turn-wire-2"}),
            dataclasses.replace(receipt, completed_at=1.0),  # not preserved evidence: no effect
        ]
        digests = [receipts.aborted_on_the_wire(gw.db, r).evidence_digest for r in changed]
        assert len(set(digests[:4]) | {wire.evidence_digest}) == 5 and digests[4] == wire.evidence_digest

    def _stored_aborted(self, gw, *, message_id="55", turn="turn-wire-1", identity=_MISSING):
        """An ABORTED receipt as a claim with exactly these preserved fields, settled by the real
        ``settle_aborted`` (a field left out is absent from the stored record)."""
        identity = dict(_WIRE_TURN) if identity is _MISSING else identity
        record = {"v": 1, "state": "pending", "owner": "dead-owner"}
        for name, value in (("message_id", message_id), ("turn_request_id", turn), ("receipt_identity", identity)):
            if value is not _MISSING:
                record[name] = value
        assert gw.db.claim_meta_once(receipts.receipt_key(901), json.dumps(record), **_proof(gw))
        assert receipts.settle_aborted(gw.db, 901, reason_code="REFUSED_BEFORE_RUN", **_proof(gw))

    def _unprovable(self, gw, case):
        key = receipts.receipt_key(901)
        if case == "unreadable":  # RECEIPT_UNREADABLE: no turn identity survives to attest
            assert gw.db.claim_meta_once(key, "{not json", **_proof(gw))
        elif case == "no_identity":  # a claim that preserved no turn identity, later aborted
            _ledger_ready()
            assert gw.db.claim_meta_once(key, json.dumps({"v": 1, "state": "pending", "message_id": "55"}),
                                         **_proof(gw))
            assert receipts.settle_aborted(gw.db, 901, reason_code="REFUSED_BEFORE_RUN", **_proof(gw))
        elif case == "ledger_never_created":  # a claimed receipt where the ledger schema never existed
            self._stored_aborted(gw)
        else:
            self._abort(gw, "HERMES_PROCESS_DIED_BEFORE_ANSWER")
            if case != "ledger_unreadable":
                _ledger_row(gw, "delivered-901", 901, "delivered")
            if case in ("ledger_table_dropped", "ledger_column_renamed"):
                # Reviewer probe: a delivery recorded, then the ledger schema that recorded it is gone.
                conn = delivery_ledger._connect()
                try:
                    conn.execute("DROP TABLE delivery_obligations" if case == "ledger_table_dropped" else
                                 "ALTER TABLE delivery_obligations RENAME COLUMN acp_update_id TO acp_update_id_old")
                    conn.commit()
                finally:
                    conn.close()

    @pytest.mark.parametrize("case", ["unreadable", "no_identity", "delivered_answer", "ledger_unreadable",
                                      "ledger_table_dropped", "ledger_column_renamed", "ledger_never_created"])
    def test_an_unprovable_aborted_receipt_is_an_explicit_error_never_a_guess(self, gw, monkeypatch, case):
        from hermes_state import SessionDB

        self._unprovable(gw, case)
        if case == "ledger_unreadable":
            real = SessionDB._read_all

            def read_all(self, sql, *a, **k):
                if "delivery_obligations" in sql:
                    raise sqlite3.OperationalError("disk I/O error")
                return real(self, sql, *a, **k)
            monkeypatch.setattr(SessionDB, "_read_all", read_all)
        with pytest.raises(receipts.ReceiptContractError):
            receipts.served(gw.db, 901)

    @pytest.mark.parametrize("field,value", [
        ("identity", {}),
        ("identity", {k: v for k, v in _WIRE_TURN.items() if k != "executorSessionIncarnation"}),
        ("message_id", _MISSING),
        ("message_id", None), ("message_id", "m1"), ("message_id", "0055"), ("message_id", "-55"),
        ("turn", _MISSING), ("turn", "turn-wire-2"),
        ("identity", {**_WIRE_TURN, "extra": "x"}),
        ("identity", {**_WIRE_TURN, "targetActorId": ""}),
        ("identity", {**_WIRE_TURN, "executorSessionId": "ses\n1"}),
        ("identity", {**_WIRE_TURN, "targetBindingId": "b" * 513}),
        ("identity", {**_WIRE_TURN, "promptDigest": "sha256:" + "AB" * 32}),
        ("identity", {**_WIRE_TURN, "bindingGeneration": 0}),
        ("identity", {**_WIRE_TURN, "bindingGeneration": True}),
        ("identity", {**_WIRE_TURN, "bindingGeneration": 4.0}),
        ("identity", {**_WIRE_TURN, "bindingGeneration": "4"}),
        ("identity", {**_WIRE_TURN, "turnRequestId": None}),
    ], ids=lambda v: "missing" if v is _MISSING else repr(v)[:40])
    def test_an_aborted_receipt_without_the_evidence_acp_compares_is_refused(self, gw, field, value):
        _ledger_ready()
        self._stored_aborted(gw, **{field: value})
        with pytest.raises(receipts.ReceiptContractError):
            receipts.served(gw.db, 901)

    def test_the_same_stored_shape_with_its_evidence_whole_is_served(self, gw):
        """Control for the refusals above: the identical helper with every field preserved."""
        _ledger_ready()
        self._stored_aborted(gw)
        served = receipts.served(gw.db, 901)
        assert served.status == "ABORTED" and served.reason_code == "REFUSED_BEFORE_RUN"
        assert served.receipt_id == "hermes-tg:aborted:901"


# ── Seam subset ────────────────────────────────────────────────────────────────────────────────
# The adapter, the runner's ``_handle_message`` and startup are real; the ACP lane is the unix
# socket above, and only the agent run (and, where named, one gate) is a stand-in.

def _ledgered_adapter(gw, *, succeed=True):
    """A real ``BasePlatformAdapter`` over a fake transport that records every text it sends."""
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
            return SendResult(success=succeed, message_id="77", error=None if succeed else "flood")

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


def _managed_event(gw, admission, *, text=None):
    from gateway.platforms.event import MessageEvent
    event = MessageEvent(text=admission.task_text if text is None else text, source=gw.source, message_id="55")
    event._acp_admission = admission
    return event


def _send_final(adapter, gw, event, text):
    return asyncio.run(adapter.send_final_ledgered(event, gw.entry.session_key, text, {}, reply_to="55"))


def _ledger_rows():
    with delivery_ledger._connect() as conn:
        return sorted(conn.execute("SELECT content, acp_update_id FROM delivery_obligations").fetchall(),
                      key=lambda r: (r[1] or "", r[0]))


def _stored_status(gw, update_id):
    """The receipt as stored (``lookup`` would read a delivered row through a PENDING receipt)."""
    return receipts.decode(gw.db.get_meta(receipts.receipt_key(update_id)), update_id).status


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
        _send_final(adapter, gw, _managed_event(gw, admission), "shaped final")
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
            result, _ = _send_final(_ledgered_adapter(gw), gw, _managed_event(gw, admission), text)
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
        _send_final(adapter, gw, _managed_event(gw, admission), "long answer")
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
        _send_final(adapter, gw, _managed_event(gw, admission), "answer")
        assert receipts.lookup(gw.db, 901).status == "PENDING"
        monkeypatch.setattr(ingress, "_PROCESS_OWNER", "a-later-process")
        assert ingress.sweep_at_startup(gw.runner) == []
        assert receipts.lookup(gw.db, 901).status == "PENDING"

    def test_a_failed_settlement_write_still_reads_completed_and_is_settled_at_startup(self, gw, monkeypatch, caplog):
        admission = _admitted(gw)
        with monkeypatch.context() as broken, caplog.at_level("WARNING", logger="gateway.platforms.base"):
            broken.setattr(receipts, "_settle", _failing_settle)
            _send_final(_ledgered_adapter(gw), gw, _managed_event(gw, admission), "the answer")
        assert "ACP receipt settlement after delivery failed" in caplog.text  # the receipt is in doubt: say so
        assert _stored_status(gw, 901) == "PENDING"
        assert receipts.lookup(gw.db, 901).status == "COMPLETED"  # GET reads the delivered row
        monkeypatch.setattr(ingress, "_PROCESS_OWNER", "a-later-process")
        assert ingress.sweep_at_startup(gw.runner) == []
        assert _stored_status(gw, 901) == "COMPLETED"

    def test_an_undelivered_answer_is_never_redelivered_and_stays_in_doubt(self, gw, monkeypatch):
        admission = _admitted(gw)
        _send_final(_ledgered_adapter(gw, succeed=False), gw, _managed_event(gw, admission), "the answer")
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
        result, _ = _send_final(adapter, gw, _managed_event(gw, admission), "managed answer")
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
        event = _managed_event(gw, admission)
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
        event = _managed_event(gw, _admitted(gw))
        _send_final(_ledgered_adapter(gw), gw, event, "X")
        event._acp_admission = None
        event._acp_handed_off = True
        _send_final(_ledgered_adapter(gw), gw, event, "X")
        assert _ledger_rows() == [("X", None), ("X", "901")]
        assert receipts.lookup(gw.db, 901).status == "COMPLETED"

    def test_a_handed_off_reply_to_a_slash_task_is_still_ledgered(self, gw):
        """The task text began with "/", but the reply answers the chain, not a slash command."""
        event = _managed_event(gw, _admitted(gw, "/acp /deploy now"))
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
        TestManagedAdmissionGate()._set_gate("false")
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
        TestManagedAdmissionGate()._set_gate("false")
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
        TestManagedAdmissionGate()._set_gate(gate)
        runs, event = TestHandleMessageSeam._wire(gw, monkeypatch)
        ordinary = event("how are you")
        [reply], envelopes = TestHandleMessageSeam._send(gw, ordinary)
        assert reply == "final reply" and runs == ["how are you"] and envelopes == []
        _send_final(_ledgered_adapter(gw), gw, ordinary, reply)
        assert _ledger_rows() == [("final reply", None)] and _managed_state(gw) == (0, 0, 0)

    def test_the_gate_is_read_per_message(self, gw, monkeypatch):
        runs, event = TestHandleMessageSeam._wire(gw, monkeypatch)
        gate = TestManagedAdmissionGate()

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

    ADMITTED = TestCommandShapes.ADMITTED
    EMPTY = TestCommandShapes.EMPTY
    NOT_OURS = TestCommandShapes.NOT_OURS

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
        TestManagedAdmissionGate()._set_gate("false")
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


def _marked_turn(gw, *, bind=True, answer=False):
    """A turn the dead process left marked on the bound session; its user row carries Telegram's
    authored time, which precedes the marker, exactly as production stores it."""
    from datetime import UTC, datetime
    from gateway.platforms.event import MessageEvent

    admission = _admitted(gw)
    store = gw.runner.session_store
    token = store.mark_turn_active(gw.entry.session_key)
    if bind:
        assert ingress.bind_turn_marker(admission, token)
    event = MessageEvent(text="deploy", source=gw.source, message_id="55",
                         timestamp=datetime.fromtimestamp(time.time() - 10, UTC))
    _, text, ts = gw.runner._hmwa_apply_message_timestamp(event, event.text)
    assert ts < store._entries[gw.entry.session_key].active_turn_started_at.timestamp()
    store.append_to_transcript(gw.entry.session_id, {"role": "user", "content": text, "timestamp": ts,
                                                      "display_metadata": {"acp_update_id": str(admission.update_id)}})
    if answer:
        store.append_to_transcript(gw.entry.session_id, {"role": "assistant", "content": "managed answer"})
    return admission, store


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
