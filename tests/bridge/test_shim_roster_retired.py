"""#99 r3 (H99-03): a roster whose rooms are all retired suppresses a batch like a single event, without an ACP retry."""
import json
from pathlib import Path

from test_shim_batch_delivery import bridge, signed, install, drive, batch, _calls


def test_retiring_all_rooms_suppresses_a_batch_without_requesting_retry(bridge):
    events = [signed(1), signed(2)]
    install(bridge, *events)
    roster = Path(bridge.MENTIONS)
    value = json.loads(roster.read_text())
    for room in value['channels'].values():
        room['status'] = 'retired'
    roster.write_text(json.dumps(value))
    snapshot, why = bridge._read_roster()
    assert snapshot is not None and why is None
    drive(bridge, batch([events[0]]))
    assert bridge.frames[-1]['result'] == {'stopReason': 'end_turn'}
    bridge.frames.clear()
    drive(bridge, batch(events), rpc_id=2)
    assert not bridge.deliveries and not _calls(bridge.tmp)
    assert bridge.frames[-1].get('result') == {'stopReason': 'end_turn'}, bridge.frames
