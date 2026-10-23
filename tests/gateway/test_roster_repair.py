"""#99 r2 review witnesses: primary reassignment keeps the CEO identity; local roster outages request an ACP retry."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.gateway.test_roster_ceo_routing import install
from tests.bridge.test_commitlore_room_peer import (
    gw, shim, bp, drive, mention, RF_CTO, LPM_CTO, ACP_ROOM, LPM_ROOM, ROOMS,
)
from tests.bridge.test_canonical_roster import document, room, replace, OWNER


def test_reassigned_primary_preserves_identity_and_accepts_current_peer(gw, shim, monkeypatch):
    binding, path = install(gw, shim, monkeypatch, 2)
    key, session = binding.session_key, binding.session_id
    drive(shim, mention('1' * 64, RF_CTO, ACP_ROOM, '@ceo initial'))
    replace(path, document(2, 2, {ACP_ROOM: room(LPM_CTO), LPM_ROOM: room(RF_CTO)}))
    assert binding.primary_pair() == (LPM_CTO, ACP_ROOM)
    [identity] = [h for method, route, h in gw.adapter._http_route_table()
                  if (method, route) == ('GET', '/v1/canonical-surface/identity')]
    response = asyncio.run(identity(SimpleNamespace(
        path_qs='/v1/canonical-surface/identity', headers={'Authorization': 'Bearer ' + gw.key}, transport=None)))
    assert response.status == 200 and json.loads(response.text)['session_id'] == session
    drive(shim, mention('2' * 64, LPM_CTO, ACP_ROOM, '@ceo new peer'))
    assert len(gw.runs) == 2 and len(shim.posts()) == 2
    assert (binding.session_key, binding.session_id) == (key, session)


@pytest.mark.parametrize('owner_broadcast', [False, True], ids=['mention', 'owner-broadcast'])
def test_local_roster_outage_requests_retry(gw, shim, monkeypatch, owner_broadcast):
    binding, path = install(gw, shim, monkeypatch, 2)
    frames = []
    monkeypatch.setattr(shim, 'write', frames.append)
    event = '3' * 64
    blocks = (bp.format_prompt(ACP_ROOM, ROOMS[ACP_ROOM][0], [bp.Ev(
        event, OWNER, 1_792_324_251, 'owner without mention', [['h', ACP_ROOM]],
        prompt_tag='owner-broadcast')], {OWNER: 'owner'}) if owner_broadcast
        else mention(event, RF_CTO, ACP_ROOM, '@ceo must survive local roster outage'))
    path.write_text('{')
    drive(shim, blocks, rpc_id=7)
    assert not gw.runs and not shim.posts() and not gw.received
    assert (Path(shim.PUBLISHED) / (event + '.held')).exists()
    assert any(frame.get('id') == 7 and 'error' in frame for frame in frames)
    assert not any(frame.get('id') == 7 and 'result' in frame for frame in frames)
    replace(path, document(2, 2, {ACP_ROOM: room(RF_CTO)}))
    frames.clear()
    drive(shim, blocks, rpc_id=8)
    assert len(gw.runs) == 1 and len(shim.posts()) == 1
    assert any(frame.get('id') == 8 and frame.get('result') == {'stopReason': 'end_turn'} for frame in frames)
