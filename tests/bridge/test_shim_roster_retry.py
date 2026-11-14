"""#99 r2 (H99-02): roster outages request an ACP retry; successful batch siblings are never repeated."""
import io
from pathlib import Path
import urllib.error

import pytest

from test_shim_batch_delivery import bridge, signed, install, drive, batch, bp, _calls


@pytest.mark.parametrize('framing', [None, bp.STEER, bp.INTERRUPT], ids=['counted', 'steer', 'interrupt'])
def test_one_roster_refusal_retries_without_repeating_successful_batch_siblings(bridge, framing):
    events = [signed(n) for n in (1, 2, 3)]
    install(bridge, *events)
    held = events[0]['id']
    bridge.failures[held] = urllib.error.HTTPError('http://canonical.invalid', 503, 'unavailable', {},
        io.BytesIO(b'{"error":{"code":"canonical_roster_unavailable"}}'))
    blocks = batch(events) if framing is None else batch(events[1:], cancelled=events[:1], framing=framing)
    drive(bridge, blocks)
    assert [d['event_id'] for d in bridge.deliveries] == [e['id'] for e in events[1:]]
    assert len(_calls(bridge.tmp)) == 2
    assert bridge.frames[-1]['error']['code'] == -32000
    assert not bridge._INFLIGHT
    assert not list((bridge.tmp / 'published').glob('*.inflight'))
    bridge.failures.clear()
    bridge.frames.clear()
    drive(bridge, blocks, rpc_id=2)
    assert [d['event_id'] for d in bridge.deliveries] == [events[1]['id'], events[2]['id'], held]
    assert len(_calls(bridge.tmp)) == 3
    assert bridge.frames[-1]['result'] == {'stopReason': 'end_turn'}


def test_local_roster_outage_does_not_acknowledge_a_recovered_mention_batch(bridge):
    events = [signed(1), signed(2)]
    install(bridge, *events)
    roster = Path(bridge.MENTIONS)
    saved = roster.read_bytes()
    roster.write_text('{')
    drive(bridge, batch(events))
    assert not bridge.deliveries and not _calls(bridge.tmp)
    assert bridge.frames[-1]['error']['code'] == -32000
    roster.write_bytes(saved)
    bridge.frames.clear()
    drive(bridge, batch(events), rpc_id=2)
    assert [d['event_id'] for d in bridge.deliveries] == [e['id'] for e in events]
    assert len(_calls(bridge.tmp)) == 2
