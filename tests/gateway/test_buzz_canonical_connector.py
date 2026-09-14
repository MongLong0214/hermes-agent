"""Native push wiring; loopback servers only, no live actors or sessions."""
import asyncio
import importlib
import json

import pytest
from aiohttp import web
import websockets


def test_native_push_reaches_authenticated_canonical_ingress():
    asyncio.run(_native_push())


def test_source_principals_are_checked_before_http():
    asyncio.run(_native_push(reject_sources=True))


def _options(module):
    key = "00" * 31 + "03"
    return dict(relay_url="wss://relay.example", endpoint="http://127.0.0.1:8642/v1/canonical-surface/events",
                binding="existing", recipient=module.public_key_hex(key), author="b" * 64,
                channel="room", since=1700000000, private_key=key, api_key="fixture-only")


@pytest.mark.parametrize("delta", [
    {"endpoint": "http://evil.example/v1/canonical-surface/events"},
    {"endpoint": "http://127.0.0.1:8642/other"}, {"endpoint": "http://user@127.0.0.1:8642/v1/canonical-surface/events"},
    {"endpoint": "http://127.0.0.1:8642/v1/canonical-surface/events?token=x"},
    {"api_key": ""}, {"api_key": "x\r\ny"}, {"recipient": "c" * 64},
    {"relay_url": "ws://remote.example"}, {"since": True}, {"since": -1},
    {"author": "bad"}, {"channel": ""}, {"binding": ""},
])
def test_configuration_refuses_unsafe_authority(delta):
    module = importlib.import_module("plugins.platforms.buzz.canonical_connector")
    with pytest.raises(ValueError, match="canonical_connector_config_invalid"):
        module.CanonicalConnector(**{**_options(module), **delta})


@pytest.mark.parametrize("status,bad_reply", [(401, False), (409, False), (503, False), (302, False), (200, True)])
def test_no_receipt_or_retry_on_refused_or_uncorrelated_response(status, bad_reply):
    asyncio.run(_native_push(status=status, bad_reply=bad_reply))


def test_configuration_check_never_connects(tmp_path, monkeypatch, capsys):
    module = importlib.import_module("plugins.platforms.buzz.canonical_connector")
    options = _options(module)
    private = tmp_path / "existing-key"
    private.write_text(options.pop("private_key"))
    options.pop("api_key")
    options["private_key_file"] = str(private)
    config = tmp_path / "candidate.json"
    config.write_text(json.dumps(options))
    monkeypatch.setattr("agent.secret_scope.get_secret", lambda name, default=None: "fixture-only")
    def forbidden(*args, **kwargs):
        raise AssertionError("check must not create network connections")
    monkeypatch.setattr(websockets, "connect", forbidden)
    assert module.main(["--config", str(config), "--check"]) == 0
    assert json.loads(capsys.readouterr().out) == {"configuration_valid": True, "activated": False}


async def _native_push(reject_sources=False, status=200, bad_reply=False):
    module = importlib.import_module("plugins.platforms.buzz.canonical_connector")
    received = []
    requests = []
    key = "00" * 31 + "03"
    recipient = module.public_key_hex(key)
    event = {"id": "a" * 64, "pubkey": "b" * 64, "kind": 9,
             "created_at": 1700000000, "tags": [["p", recipient], ["h", "room"]],
             "content": "A native question"}

    async def ingress(request):
        assert request.headers.get("Authorization") == "Bearer fixture-only"
        received.append(await request.json())
        if status != 200 or bad_reply:
            done.set()
        return web.json_response({"event_id": "wrong" if bad_reply else event["id"], "text": "answer"}, status=status,
                                 headers={"Location": "/must-not-follow"})

    app = web.Application()
    app.router.add_post("/v1/canonical-surface/events", ingress)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    async def relay(ws):
        await ws.send(json.dumps(["AUTH", "fixture-challenge"]))
        auth = json.loads(await ws.recv())
        assert auth[0] == "AUTH" and auth[1]["pubkey"] == recipient
        await ws.send(json.dumps(["OK", auth[1]["id"], True, ""]))
        req = json.loads(await ws.recv())
        requests.append(req)
        if reject_sources:
            for delta in (
                {"pubkey": "c" * 64}, {"kind": 1},
                {"tags": [["p", "c" * 64], ["h", "room"]]},
                {"tags": [["p", recipient], ["h", "other"]]},
                {"tags": [["p", recipient], ["h", "room"], ["h", "other"]]},
                {"content": ""}, {"created_at": 1699999999}, {"id": "bad"},
            ):
                await ws.send(json.dumps(["EVENT", req[1], {**event, **delta}]))
        await ws.send(json.dumps(["EVENT", req[1], event]))
        await asyncio.wait_for(done.wait(), 5)

    done = asyncio.Event()
    try:
        async with websockets.serve(relay, "127.0.0.1", 0) as server:
            relay_port = server.sockets[0].getsockname()[1]
            connector = module.CanonicalConnector(
                relay_url=f"ws://127.0.0.1:{relay_port}", endpoint=f"http://127.0.0.1:{port}/v1/canonical-surface/events",
                binding="existing", recipient=recipient, author="b" * 64,
                channel="room", since=1700000000, private_key=key, api_key="fixture-only",
            )
            receipts = []
            async def receipt(event_id):
                receipts.append(event_id)
                assert event_id == event["id"]
                done.set()
            if status != 200 or bad_reply:
                with pytest.raises(ValueError, match="canonical_delivery_refused"):
                    await asyncio.wait_for(connector.run(receipt), 10)
                assert receipts == []
            else:
                await asyncio.wait_for(connector.run(receipt), 10)
                assert receipts == [event["id"]]
        assert requests == [["REQ", "canonical-buzz", {"kinds": [9], "authors": ["b" * 64],
                            "#p": [recipient], "#h": ["room"], "since": 1700000000}]]
        assert received == [{"binding": "existing", "event_id": event["id"],
                             "author_id": event["pubkey"], "channel_id": "room", "text": event["content"]}]
    finally:
        await runner.cleanup()
