"""#100 r3 (H100-R3-01): a roster that becomes unreadable at any point while one event is decided must
never end the turn with that event silently undelivered: it is delivered, or the prompt fails so
buzz-acp resends it."""
from pathlib import Path

import pytest

import test_shim_batch_delivery as delivery
from test_shim_batch_delivery import signed, drive, bp, CH, NAME, _calls


def _outage_from(bridge, monkeypatch, start):
    read = bridge._roster_reader.read
    reads = 0

    def read_with_outage():
        nonlocal reads
        reads += 1
        if reads == start:
            Path(bridge.MENTIONS).write_text("{")
        return read()

    monkeypatch.setattr(bridge._roster_reader, "read", read_with_outage)


def _failed(bridge):
    return any("error" in frame for frame in bridge.frames)


@pytest.mark.parametrize("start", range(1, 8))
def test_owner_broadcast_is_delivered_or_retried_whenever_the_roster_breaks(tmp_path, monkeypatch, start):
    bridge = delivery.bridge.__wrapped__(tmp_path, monkeypatch)
    event = signed(1, tags=[["h", CH]])
    monkeypatch.setattr(bridge, "OWNER", event["pubkey"])
    _outage_from(bridge, monkeypatch, start)
    drive(bridge, bp.format_prompt(CH, NAME, [bp.Ev(event["id"], event["pubkey"], event["created_at"],
        event["content"], event["tags"], prompt_tag="owner-broadcast")], {}))
    delivered = [d["event_id"] for d in bridge.deliveries] == [event["id"]]
    assert delivered or (_failed(bridge) and not bridge.deliveries and not _calls(tmp_path)), bridge.frames


@pytest.mark.parametrize("start", range(1, 10))
def test_a_batch_never_hides_an_undelivered_owner_broadcast_behind_a_sibling(tmp_path, monkeypatch, start):
    bridge = delivery.bridge.__wrapped__(tmp_path, monkeypatch)
    owner = signed(1, tags=[["h", CH]])
    sibling = signed(2)
    monkeypatch.setattr(bridge, "OWNER", owner["pubkey"])
    delivery.install(bridge, owner, sibling)
    _outage_from(bridge, monkeypatch, start)
    drive(bridge, delivery.batch([owner, sibling]))
    ids = {d["event_id"] for d in bridge.deliveries}
    assert owner["id"] in ids or _failed(bridge), (ids, bridge.frames)
    assert sibling["id"] in ids or _failed(bridge), (ids, bridge.frames)
