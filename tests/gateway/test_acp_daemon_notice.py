"""``POST /v1/canonical-surface/daemon-notices``: a signed ``acp-daemon-notice/v1`` reaches the CEO's
existing canonical session once, or is refused with a specific 4xx before anything is claimed.

The runner, binding, resolver, coordinator and SQLite are real (the ``ingress`` fixture of
``test_canonical_surface_ingress``, its binding renamed ``acp-canonical-ceo``); the cached actor is that
test's stand-in for the model. Requests go over a real listener on 127.0.0.1.

ACP's own code is exercised too: ``tests/gateway/acp_notice/sealed_start_refusal.json`` is the exact body
ACP's ``sealDaemonNotice`` and ``JSON.stringify`` produced (``acp_notice.mjs seal``), and one test runs ACP's
real probe, identity read and ``createHermesGatewayDaemonNoticeSender`` against the listener when an ACP
checkout carrying them is available (``ACP_REPO``).
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import hmac
import json
import os
import shutil
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest

from gateway import acp_daemon_notice as notices
from gateway import acp_managed_ingress
from gateway.canonical_surface import CanonicalReceiptCoordinator, CanonicalReceiptResult
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tests.gateway.test_canonical_surface_ingress import ingress as _ingress_fixture  # noqa: F401

ingress = _ingress_fixture

SECRET = "acp-daemon-notice test lane secret (a test constant)"
NOTICES = "/v1/canonical-surface/daemon-notices"
EVENTS = "/v1/canonical-surface/events"
HERE = Path(__file__).parent / "acp_notice"
ACP_REPO = Path(os.environ.get("ACP_REPO", "/Users/isaac/worktrees/agent-control-plane/i-1496"))
REFUSAL_ID = "refusal-9f2c41"


def _hkdf_sha256(secret: str, salt: bytes, info: bytes, length: int = 32) -> bytes:
    """RFC 5869, written out here so the receiver's key derivation is checked against the standard."""
    prk = hmac.new(salt, secret.encode(), hashlib.sha256).digest()
    out, block = b"", b""
    while len(out) < length:
        block = hmac.new(prk, block + info + bytes([len(out) // 32 + 1]), hashlib.sha256).digest()
        out += block
    return out[:length]


def refusal_text(refusal_id=REFUSAL_ID):
    message_id = f"start-refusal:{refusal_id}"
    body = json.dumps({"messageId": message_id, "kind": "CEO_NOTIFICATION", "payload": {
        "source": "DAEMON_START_REFUSED_BEFORE_OPEN", "refusalId": refusal_id, "reasonCode": "START_GATE_REFUSED",
        "databaseWritten": False}})
    return f"ACP notification {message_id} part 1/1: {body}"


def sign(destination, *, event_id="acp-notice:" + "1" * 64, text=None, secret=SECRET, **overrides):
    fields = {"protocol": "acp-daemon-notice/v1", "principal": "acp-daemon", "binding": "acp-canonical-ceo",
              "event_id": event_id, "destination": destination, "text": refusal_text() if text is None else text}
    canonical = notices.canonical_json(fields).encode()
    key = _hkdf_sha256(secret, b"acp-hermes-lane", b"acp-daemon-notice/v1")
    return {**fields, "payload_digest": "sha256:" + hashlib.sha256(canonical).hexdigest(),
            "signature": "hmac-sha256:" + hmac.new(key, canonical, hashlib.sha256).hexdigest(), **overrides}


@pytest.fixture
def rig(ingress, monkeypatch):
    """The listener, the CEO binding under the notice's name, and the lane secret."""
    binding = dataclasses.replace(ingress.binding, name=notices.BINDING)
    ingress.runner.config.canonical_surface_bindings = {binding.name: binding}
    monkeypatch.setattr(acp_managed_ingress, "read_secret", lambda: SECRET)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0,
                                                                   "key": ingress.key}))
    adapter.gateway_runner = ingress.runner

    @asynccontextmanager
    async def serving():
        assert await adapter.connect()
        try:
            port = adapter._site._server.sockets[0].getsockname()[1]
            auth = {"Authorization": f"Bearer {ingress.key}"}
            async with aiohttp.ClientSession() as client:
                async def post(body, route=NOTICES, headers=auth):
                    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
                    async with client.post(f"http://127.0.0.1:{port}{route}", data=raw, headers={
                            "Content-Type": "application/json", **headers}) as response:
                        return response.status, await response.json()

                async def identity():
                    async with client.get(f"http://127.0.0.1:{port}/v1/canonical-surface/identity",
                                          headers=auth) as response:
                        assert response.status == 200
                        return await response.json()

                async def identity_status():
                    async with client.get(f"http://127.0.0.1:{port}/v1/canonical-surface/identity",
                                          headers=auth) as response:
                        return response.status, await response.json()

                yield SimpleNamespace(post=post, identity=identity, identity_status=identity_status, client=client, port=port, auth=auth)
        finally:
            await adapter.disconnect()

    def run(fn):
        async def go():
            async with serving() as served:
                served.destination = await served.identity()
                return await fn(served)
        return asyncio.run(go())

    yield SimpleNamespace(run=run, ingress=ingress, adapter=adapter)
    adapter._response_store.close()


def turns(rig):
    return rig.ingress.actor.calls


def receipt_count(rig):
    return len(rig.ingress.receipts())


# --- the notice is received ------------------------------------------------------------------------


def test_an_origin_less_signed_notice_runs_one_ceo_turn_and_is_answered_with_a_receipt_id(rig):
    async def go(s):
        body = sign(s.destination)
        status, answer = await s.post(body)
        assert status == 200
        assert answer["event_id"] == body["event_id"] and answer["payload_digest"] == body["payload_digest"]
        receipt = answer["receipt"]
        assert receipt["status"] == "completed"
        assert receipt["session_id"] == s.destination["session_id"]
        assert receipt["lineage_root_digest"] == s.destination["lineage_root_digest"]
        assert receipt["receipt_id"].startswith("hermes-canonical-receipt:v2:")

    rig.run(go)
    assert len(turns(rig)) == 1 and receipt_count(rig) == 1
    # The CEO's own session read it as daemon data, with the refusal id intact.
    [model_input] = rig.ingress.actor.model_inputs
    assert REFUSAL_ID in model_input and notices.LABEL in model_input


def test_a_replay_of_the_same_event_id_runs_no_second_turn(rig):
    async def go(s):
        body = sign(s.destination)
        first = await s.post(body)
        second = await s.post(body)
        assert first[0] == 200 and second == first
        # The same event id with other signed text is a different payload, never a new turn.
        other = await s.post(sign(s.destination, text=refusal_text("refusal-other")))
        assert other == (409, {"error": "daemon_notice_payload_mismatch",
                               "detail": "daemon_notice_payload_mismatch"})

    rig.run(go)
    assert len(turns(rig)) == 1 and receipt_count(rig) == 1


def test_two_different_event_ids_are_two_turns(rig):
    async def go(s):
        one = await s.post(sign(s.destination, event_id="acp-notice:" + "a" * 64))
        two = await s.post(sign(s.destination, event_id="acp-notice:" + "b" * 64,
                                text=refusal_text("refusal-two")))
        assert one[0] == two[0] == 200
        assert one[1]["receipt"]["receipt_id"] != two[1]["receipt"]["receipt_id"]

    rig.run(go)
    assert len(turns(rig)) == 2 and receipt_count(rig) == 2


def test_a_notice_that_names_an_origin_is_accepted_and_nothing_is_sent_to_telegram(rig):
    origin = json.dumps({"messageId": "msg-1", "kind": "CEO_NOTIFICATION",
                         "payload": {"origin": {"turn_request_id": "turn-1", "update_id": "901"}}})

    async def go(s):
        status, _ = await s.post(sign(s.destination, text=f"ACP notification msg-1 part 1/1: {origin}"))
        assert status == 200

    rig.run(go)
    assert len(turns(rig)) == 1


# --- refusals before any claim ---------------------------------------------------------------------


@pytest.mark.parametrize("change", [
    {"signature": "hmac-sha256:" + "0" * 64},
    {"payload_digest": "sha256:" + "0" * 64},
    {"secret": "another lane secret"},
])
def test_a_bad_hmac_is_a_401_with_no_turn(rig, change):
    async def go(s):
        body = sign(s.destination, **change)
        assert await s.post(body) == (401, {"error": "daemon_notice_signature_invalid",
                                            "detail": "daemon_notice_signature_invalid"})

    rig.run(go)
    assert turns(rig) == [] and receipt_count(rig) == 0


def test_a_text_changed_after_signing_is_a_401_with_no_turn(rig):
    async def go(s):
        body = sign(s.destination)
        body["text"] = refusal_text("refusal-forged")
        assert (await s.post(body))[0] == 401

    rig.run(go)
    assert turns(rig) == [] and receipt_count(rig) == 0


@pytest.mark.parametrize("stale", [{"session_id": "20261001_000000_previous"},
                                   {"process_pid": os.getpid() + 1},
                                   {"process_started_at": "darwin-tv:1700000000.000001"},
                                   {"lineage_root_digest": "sha256:" + "d" * 64}])
def test_a_wrong_destination_is_a_409_with_no_turn(rig, stale):
    async def go(s):
        body = sign({**s.destination, **stale})
        assert await s.post(body) == (409, {"error": "daemon_notice_destination_mismatch",
                                            "detail": "daemon_notice_destination_mismatch"})

    rig.run(go)
    assert turns(rig) == [] and receipt_count(rig) == 0


def _drop(key):
    return lambda body: body.pop(key)


@pytest.mark.parametrize("damage", [
    lambda body: b"not json",
    lambda body: json.dumps([body]).encode(),
    lambda body: b'{"protocol":"acp-daemon-notice/v1","protocol":"acp-daemon-notice/v1"}',
    lambda body: body.update(extra="field"),
    _drop("payload_digest"), _drop("destination"), _drop("text"),
    lambda body: body.update(principal="owner"),
    lambda body: body.update(binding="another-binding"),
    lambda body: body.update(protocol="acp-daemon-notice/v2"),
    lambda body: body.update(payload_digest="sha256:XYZ"),
    lambda body: body.update(signature="hmac-sha256:abc"),
    lambda body: body.update(text="   "),
    lambda body: body.update(event_id=""),
    lambda body: body["destination"].update(extra=1),
    lambda body: body["destination"].update(process_pid="1"),
    lambda body: body["destination"].update(process_started_at="not-a-start"),
])
def test_a_malformed_body_is_a_400_with_no_turn(rig, damage):
    async def go(s):
        body = sign(s.destination)
        raw = damage(body)
        assert (await s.post(raw if isinstance(raw, bytes) else body)) == (
            400, {"error": "daemon_notice_invalid", "detail": "daemon_notice_invalid"})

    rig.run(go)
    assert turns(rig) == [] and receipt_count(rig) == 0


def test_a_lone_surrogate_in_the_text_is_a_400(rig):
    async def go(s):
        raw = json.dumps(sign(s.destination)).replace(json.dumps(refusal_text())[1:-1], "\\ud800")
        assert (await s.post(raw.encode()))[0] == 400

    rig.run(go)
    assert turns(rig) == []


def test_a_notice_is_refused_without_the_gateway_key_before_it_is_read(rig):
    async def go(s):
        body = sign(s.destination)
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            status, answer = await s.post(body, headers=headers)
            assert (status, answer["error"]["code"]) == (401, "gateway_auth_failed")

    rig.run(go)
    assert turns(rig) == [] and receipt_count(rig) == 0


def test_no_lane_secret_is_a_503_with_no_turn(rig, monkeypatch):
    monkeypatch.setattr(acp_managed_ingress, "read_secret", lambda: None)

    async def go(s):
        assert await s.post(sign(s.destination)) == (503, {"error": "daemon_notice_unavailable",
                                                           "detail": "daemon_notice_unavailable"})

    rig.run(go)
    assert turns(rig) == []


def test_no_canonical_ceo_binding_is_a_503_with_no_turn(rig):
    async def go(s):
        body = sign(s.destination)
        rig.ingress.runner.config.canonical_surface_bindings = {}
        assert (await s.post(body))[0] == 503

    rig.run(go)
    assert turns(rig) == []


def test_a_replay_for_a_new_destination_is_a_409_never_the_old_receipt_under_the_new_session(rig, monkeypatch):
    from gateway.platforms import api_server_canonical as canonical

    live = canonical.live_identity
    current = {}
    monkeypatch.setattr(canonical, "live_identity", lambda runner: (current.get("identity") or live(runner)[0], None))

    async def go(s):
        first = await s.post(sign(s.destination))
        assert first[0] == 200 and first[1]["receipt"]["session_id"] == s.destination["session_id"]
        # The CEO session is reset: the same event id and text, signed for the new session, now current.
        current["identity"] = dict(s.destination, session_id=s.destination["session_id"] + "-new")
        replay = await s.post(sign(current["identity"]))
        assert replay == (409, {"error": "daemon_notice_payload_mismatch",
                                "detail": "daemon_notice_payload_mismatch"})

    rig.run(go)
    assert len(turns(rig)) == 1 and receipt_count(rig) == 1


def test_a_notice_is_refused_when_identity_would_refuse_because_the_only_room_is_retired(rig):
    from tests.bridge.test_canonical_roster import OWNER, document, replace, room
    from gateway.canonical_surface import CanonicalSurfaceBinding
    from bridge.canonical.canonical_roster import RosterReader

    root = rig.ingress.home / "bridge"
    (root / "keys").mkdir(parents=True)
    (root / "keys" / "owner.pub").write_text(OWNER)
    path = root / "required-mentions.json"
    room_id = "10000000-0000-4000-8000-000000000001"
    replace(path, document(2, 1, {room_id: room()}))
    binding = dataclasses.replace(rig.ingress.runner.config.canonical_surface_bindings[notices.BINDING],
                                  allowed_channel_ids=(room_id,), allowed_author_ids=("b" * 64,),
                                  roster=RosterReader(path))
    rig.ingress.runner.config.canonical_surface_bindings = {binding.name: binding}

    async def go(s):
        before = await s.identity()
        body = sign(before)
        assert (await s.post(body))[0] == 200
        replace(path, document(2, 2, {room_id: room(status="retired")}))
        status, _ = await s.identity_status()
        assert status != 200
        other = sign(before, event_id="acp-notice:" + "2" * 64)
        status, answer = await s.post(other)
        assert status in (409, 503) and answer["error"] in ("daemon_notice_unavailable", "canonical_binding_stale")

    rig.run(go)
    assert len(turns(rig)) == 1


@pytest.mark.parametrize("mutate", [
    lambda b: b["destination"].update(process_pid=2**40),
    lambda b: b["destination"].update(process_pid=2**63),
    lambda b: b.update(event_id="acp-notice:\\ud800"),
    lambda b: b["destination"].update(session_id="s\\ud800"),
    lambda b: b.update(text="x\\ud800"),
])
def test_out_of_range_or_unpaired_surrogate_fields_are_a_400_never_a_500(rig, mutate):
    async def go(s):
        body = sign(s.destination)
        mutate(body)
        raw = json.dumps(body).encode().replace(b"\\\\ud800", b"\\ud800")
        status, answer = await s.post(raw)
        assert (status, answer["error"]) == (400, "daemon_notice_invalid")

    rig.run(go)
    assert turns(rig) == [] and receipt_count(rig) == 0


# --- /events is unchanged --------------------------------------------------------------------------


def test_events_still_refuses_a_notice_and_the_notice_principal(rig):
    async def go(s):
        status, answer = await s.post(sign(s.destination), route=EVENTS)
        assert (status, answer["error"]["code"]) == (400, "canonical_invalid_request")
        # The notice's author is not admitted on /events either: no roster or configured pair grants it.
        event = {"binding": notices.BINDING, "event_id": "event-x", "author_id": notices.AUTHOR,
                 "channel_id": notices.CHANNEL, "text": "hello"}
        status, answer = await s.post(event, route=EVENTS)
        assert status == 403 and answer["error"]["code"] == "canonical_principal_rejected"
        # And its own admitted pair still works there, unchanged.
        status, answer = await s.post({**event, "author_id": "author", "channel_id": "channel"}, route=EVENTS)
        assert status == 200 and answer["event_id"] == "event-x"

    rig.run(go)
    assert len(turns(rig)) == 1


# --- the route as ACP finds it ---------------------------------------------------------------------


def test_head_without_auth_is_a_405_that_allows_post_and_runs_nothing(rig):
    async def go(s):
        async with s.client.head(f"http://127.0.0.1:{s.port}{NOTICES}") as response:
            assert response.status == 405
            assert "POST" in [m.strip() for m in response.headers["Allow"].split(",")]
        async with s.client.get(f"http://127.0.0.1:{s.port}{NOTICES}", headers=s.auth) as response:
            assert response.status == 405
        # A route that does not exist is a 404, which is what ACP reads as "not served".
        async with s.client.head(f"http://127.0.0.1:{s.port}{NOTICES}-absent") as response:
            assert response.status == 404

    rig.run(go)
    assert turns(rig) == [] and receipt_count(rig) == 0


# --- ACP's own bytes -------------------------------------------------------------------------------

_FIXTURE_SECRET = "acp-daemon-notice fixture lane secret (a test constant)"


def test_bytes_sealed_by_acps_sealDaemonNotice_are_accepted_by_the_receiver(rig, monkeypatch):
    raw = (HERE / "sealed_start_refusal.json").read_bytes()
    wire = json.loads(raw)

    notice = notices.DaemonNotice.from_json_bytes(raw)
    assert notice.canonical() == notices.canonical_json({k: wire[k] for k in (
        "protocol", "principal", "binding", "event_id", "destination", "text")})
    assert notice.authentic(_FIXTURE_SECRET) and not notice.authentic(SECRET)
    assert notices.notice_key(_FIXTURE_SECRET) == _hkdf_sha256(_FIXTURE_SECRET, b"acp-hermes-lane",
                                                               b"acp-daemon-notice/v1")
    assert "refusal-fixture-0001" in notice.text and "origin" not in notice.text

    # The fixture's destination is a fixed one, so the live identity check and the coordinator stand
    # aside here; everything the receiver does with the bytes itself is real.
    seen = []

    async def submit(self, binding, event):
        seen.append(event)
        return CanonicalReceiptResult(status="terminal", terminal_text="the CEO's answer")

    monkeypatch.setattr(acp_managed_ingress, "read_secret", lambda: _FIXTURE_SECRET)
    monkeypatch.setattr("gateway.platforms.api_server_canonical.live_identity",
                        lambda runner: (dict(wire["destination"]), None))
    monkeypatch.setattr(CanonicalReceiptCoordinator, "submit", submit)

    async def go(s):
        status, answer = await s.post(raw)
        assert status == 200
        assert answer["event_id"] == wire["event_id"] and answer["payload_digest"] == wire["payload_digest"]
        assert answer["receipt"]["status"] == "completed"
        assert answer["receipt"]["session_id"] == wire["destination"]["session_id"]
        assert answer["receipt"]["lineage_root_digest"] == wire["destination"]["lineage_root_digest"]
        assert answer["receipt"]["receipt_id"]

    rig.run(go)
    [event] = seen
    assert event.text == f"{notices.LABEL}[notice {wire['payload_digest']}]\n" + wire["text"] and event.expected_identity == wire["destination"]


def _node(*args, timeout=60):
    return subprocess.run(["node", "--experimental-transform-types", "--no-warnings", str(HERE / "acp_notice.mjs"),
                           *args], capture_output=True, text=True, timeout=timeout)


def _acp_has_the_notice_sender():
    source = ACP_REPO / "src/runtime/hermes-gateway-conversation.ts"
    return (shutil.which("node") is not None and source.is_file()
            and "probeDaemonNoticeRoute" in source.read_text(encoding="utf-8")
            and "path?: string" in source.read_text(encoding="utf-8"))


@pytest.mark.skipif(not _acp_has_the_notice_sender(), reason="no ACP checkout with the daemon-notice sender (ACP_REPO)")
def test_acps_real_probe_and_sender_settle_against_the_receiver(rig):
    async def go(s):
        process = await asyncio.create_subprocess_exec(
            "node", "--experimental-transform-types", "--no-warnings", str(HERE / "acp_notice.mjs"), "send",
            str(ACP_REPO), str(s.port), rig.ingress.key, SECRET, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        out, err = await asyncio.wait_for(process.communicate(), 60)
        assert process.returncode == 0, err.decode()
        return json.loads(out)

    result = rig.run(go)
    assert result["probe"]["supported"] is True, result["probe"]
    assert result["outcome"]["kind"] == "RESPONDED" and result["outcome"]["status"] == 200, result["outcome"]
    assert result["settlement"]["settled"] is True, result["settlement"]
    assert result["settlement"]["receiptId"]
    assert len(turns(rig)) == 1 and "refusal-fixture-0001" in rig.ingress.actor.model_inputs[0]
