"""Bootstrap roster changes reach the existing CEO through the real shim, HTTP handler and SQLite."""
import asyncio
import json
from types import SimpleNamespace
from pathlib import Path
import urllib.error

import pytest

from tests.bridge.test_commitlore_room_peer import (
    gw, shim, drive, mention, RF_CTO, LPM_CTO, ACP_ROOM, LPM_ROOM, ROOMS, bp,
)
from tests.bridge.test_canonical_roster import document, room, replace, OWNER


def install(gw, shim, monkeypatch, version):
    binding = gw.runner.config.canonical_surface_bindings['acp-canonical-ceo']
    path = binding.roster.path
    path.parent.mkdir(parents=True)
    (path.parent / 'keys').mkdir()
    (path.parent / 'keys/owner.pub').write_text(OWNER)
    value = document(version, channels={ACP_ROOM: room(RF_CTO)})
    replace(path, value)
    monkeypatch.setattr(shim, 'MENTIONS', str(path))
    monkeypatch.setattr(shim, '_roster_reader', shim._roster_module.RosterReader(path))
    monkeypatch.setattr(shim, 'OWNER', OWNER)
    return binding, path


@pytest.mark.parametrize('version', [1, 2])
def test_hot_added_room_mentions_and_owner_broadcast_keep_one_session_and_pair_isolation(
        gw, shim, monkeypatch, version):
    binding, path = install(gw, shim, monkeypatch, version)
    key, session = binding.session_key, binding.session_id
    drive(shim, mention('1' * 64, RF_CTO, ACP_ROOM, '@ceo first'))
    replace(path, document(version, 2, {ACP_ROOM: room(RF_CTO), LPM_ROOM: room(LPM_CTO)}))
    for n, (author, broadcast) in enumerate([(LPM_CTO, False), (OWNER, False), (OWNER, True)], 2):
        event = f'{n:064x}'
        if broadcast:
            blocks = bp.format_prompt(LPM_ROOM, ROOMS[LPM_ROOM][0], [bp.Ev(
                event, author, 1_792_324_251, 'owner without mention', [['h', LPM_ROOM]],
                prompt_tag='owner-broadcast')], {OWNER: 'owner'})
        else:
            blocks = mention(event, author, LPM_ROOM, '@ceo new project')
        drive(shim, blocks, rpc_id=n)
        receipt = json.loads((Path(shim.PUBLISHED) / event).read_text())
        assert receipt['roster_generation'] == (2 if version == 2 else 0)
    assert len(gw.runs) == len(shim.posts()) == 4
    assert all(status == 200 for status, _ in gw.answers)
    assert (binding.session_key, binding.session_id) == (key, session)
    assert gw.runner.session_store.lookup_by_session_key_existing(key).session_id == session
    with gw.runner.session_store._db._lock:
        assert gw.runner.session_store._db._conn.execute('SELECT count(*) FROM sessions').fetchone()[0] == 1
    # Exercise gateway refusal directly too: the shim's retired-room suppression cannot mask a grant.
    for author, channel in [(RF_CTO, LPM_ROOM), (LPM_CTO, ACP_ROOM)]:
        with pytest.raises(urllib.error.HTTPError) as error:
            shim.submit('cross pair', f'{len(gw.received):064x}', author, channel)
        assert error.value.code == 403
    replace(path, document(version, 3, {ACP_ROOM: room(RF_CTO, 'retired'), LPM_ROOM: room(LPM_CTO)}))
    assert binding.primary_pair() == (LPM_CTO, LPM_ROOM)
    [identity] = [h for method, route, h in gw.adapter._http_route_table()
                  if (method, route) == ('GET', '/v1/canonical-surface/identity')]
    response = asyncio.run(identity(SimpleNamespace(
        path_qs='/v1/canonical-surface/identity', headers={'Authorization': 'Bearer ' + gw.key}, transport=None)))
    assert response.status == 200 and json.loads(response.text)['session_id'] == session
    for author in (RF_CTO, OWNER):
        assert not binding.admits(author, ACP_ROOM)  # overrides even the explicit config grant
        with pytest.raises(urllib.error.HTTPError) as error:
            shim.submit('retired', f'{len(gw.received):064x}', author, ACP_ROOM)
        assert error.value.code == 403
    before = len(shim.posts())
    drive(shim, mention('9' * 64, RF_CTO, ACP_ROOM, '@ceo retired'))
    assert not shim.publish_reply('a' * 64, ACP_ROOM, 'retired during turn')
    assert len(shim.posts()) == before
    assert len(gw.runs) == 4
    assert 'roster_generation=' in Path(shim.LOG).read_text()


def test_malformed_roster_refuses_gateway_and_shim_without_discarding_last_good(gw, shim, monkeypatch, caplog):
    binding, path = install(gw, shim, monkeypatch, 2)
    drive(shim, mention('1' * 64, RF_CTO, ACP_ROOM, '@ceo first'))
    previous = binding.roster.last_good
    shim_previous = shim._roster_reader.last_good
    path.write_text('{')
    with pytest.raises(urllib.error.HTTPError) as error:
        shim.submit('bad roster', '2' * 64, RF_CTO, ACP_ROOM)
    assert error.value.code == 503
    assert json.loads(error.value.read())['error']['code'] == 'canonical_roster_unavailable'
    drive(shim, mention('3' * 64, RF_CTO, ACP_ROOM, '@ceo no grant'))
    blocks = bp.format_prompt(ACP_ROOM, ROOMS[ACP_ROOM][0], [bp.Ev(
        '5' * 64, OWNER, 1_792_324_251, 'owner without mention', [['h', ACP_ROOM]],
        prompt_tag='owner-broadcast')], {OWNER: 'owner'})
    drive(shim, blocks)
    assert (Path(shim.PUBLISHED) / ('5' * 64 + '.held')).exists()
    assert len(gw.runs) == 1
    assert binding.roster.last_good is previous and shim._roster_reader.last_good is shim_previous
    assert 'roster' in caplog.text and 'roster unavailable' in Path(shim.LOG).read_text()
    replace(path, document(generation=2, channels={ACP_ROOM: room(RF_CTO)}))
    drive(shim, mention('4' * 64, RF_CTO, ACP_ROOM, '@ceo recovery'))
    assert len(gw.runs) == 2 and binding.roster.last_good.generation == 2


def test_reassigned_room_refuses_the_old_cto_despite_its_config_grant(gw, shim, monkeypatch):
    """#99 r1 H99-01: a room the roster lists is decided by the roster alone."""
    from tests.bridge.test_commitlore_room_peer import RF_CTO, LPM_CTO, ACP_ROOM, LPM_ROOM, drive, mention
    binding, path = install(gw, shim, monkeypatch, 2)
    drive(shim, mention("1" * 64, RF_CTO, ACP_ROOM, "@ceo initial"))
    turns = len(gw.runs)
    replace(path, document(2, 2, {ACP_ROOM: room(LPM_CTO), LPM_ROOM: room(RF_CTO)}))
    assert binding.roster.read().admits(RF_CTO, ACP_ROOM, shim.OWNER) is False
    with pytest.raises(urllib.error.HTTPError) as refused:
        shim.submit("old CTO must not be admitted to the reassigned room", "2" * 64, RF_CTO, ACP_ROOM)
    assert refused.value.code == 403
    assert len(gw.runs) == turns


def test_gateway_roster_outage_requests_an_acp_retry_and_the_resend_runs_once(gw, shim, monkeypatch):
    """#99 r1 H99-02: a 503 canonical_roster_unavailable claims nothing, so the prompt must fail
    (buzz-acp resends the same event id) instead of ending the turn as if delivered."""
    from tests.bridge.test_commitlore_room_peer import RF_CTO, ACP_ROOM, bp, mention
    binding, path = install(gw, shim, monkeypatch, 2)
    owner_pub = path.parent / "keys" / "owner.pub"
    saved = owner_pub.read_bytes()
    owner_pub.unlink()
    frames = []
    monkeypatch.setattr(shim, "write", frames.append)
    event = "3" * 64
    message = {"jsonrpc": "2.0", "id": 7, "method": "session/prompt",
               "params": bp.prompt_params("canonical:acp-canonical-ceo",
                                          mention(event, RF_CTO, ACP_ROOM, "@ceo must survive a roster outage"))}
    shim.handle(message)
    assert gw.answers[-1][0] == 503 and gw.answers[-1][1]["error"]["code"] == "canonical_roster_unavailable"
    assert not gw.runs and not shim.posts()
    errors = [frame for frame in frames if frame.get("id") == 7 and "error" in frame]
    assert errors and "retryable" in errors[0]["error"]["message"]
    assert not any(frame.get("id") == 7 and "result" in frame for frame in frames)
    assert (Path(shim.PUBLISHED) / (event + ".held")).exists()
    # The roster is readable again: buzz-acp's resend of the same event runs exactly one turn.
    owner_pub.write_bytes(saved)
    frames.clear()
    shim.handle(message)
    assert len(gw.runs) == 1
    assert any(frame.get("id") == 7 and frame.get("result") == {"stopReason": "end_turn"} for frame in frames)
