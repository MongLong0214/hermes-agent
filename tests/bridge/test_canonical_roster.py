"""Atomic roster replacements preserve pair isolation and never install malformed authority."""
import importlib.util
import json
from pathlib import Path

import pytest

# Standalone loading is also how the Python 3.9 shim consumes this module.
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('roster_test', ROOT / 'bridge/canonical/canonical_roster.py')
roster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roster)
A, B = '10000000-0000-4000-8000-000000000001', '20000000-0000-4000-8000-000000000002'
CEO, CTO, OTHER, OWNER = ('a' * 64, 'b' * 64, 'c' * 64, 'd' * 64)


def room(peer=CTO, status='active'):
    return {'relay_name': 'project', 'ceo': {'pubkey': CEO, 'handle': 'ceo'},
            'peer': {'pubkey': peer, 'handle': 'cto-project'}, 'acp_project_id': 'p',
            'classification': 'project', 'status': status}


def document(version=2, generation=1, channels=None):
    value = {'version': version, 'relay': 'wss://relay.invalid', 'channels': channels or {A: room()}}
    if version == 2:
        value['generation'] = generation
    return value


def replace(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value))
    tmp.replace(path)


@pytest.mark.parametrize('version', [1, 2])
def test_reader_reloads_atomic_addition_and_retirement_with_explicit_pairs(tmp_path, version):
    path = tmp_path / 'roster.json'
    value = document(version)
    if version == 1:
        del value['channels'][A]['status']
    replace(path, value)
    reader = roster.RosterReader(path)
    first = reader.read()
    assert first.admits(CTO, A, OWNER) and first.admits(OWNER, A, OWNER)
    assert not first.admits(OTHER, A, OWNER)
    assert reader.read() is first
    replace(path, document(version, 2, {A: room(), B: room(OTHER)}))
    second = reader.read()
    assert second.generation == (2 if version == 2 else 0)
    assert second.admits(OTHER, B, OWNER) and second.admits(OWNER, B, OWNER)
    assert not second.admits(CTO, B, OWNER) and not second.admits(OTHER, A, OWNER)
    replace(path, document(version, 3, {A: room(status='retired'), B: room(OTHER)}))
    third = reader.read()
    assert not third.admits(CTO, A, OWNER) and not third.admits(OWNER, A, OWNER)


@pytest.mark.parametrize('bad', ['json', 'pair', 'status', 'generation', 'rollback', 'reuse', 'duplicate', 'missing'])
def test_invalid_replacement_fails_closed_keeps_snapshot_and_recovers(tmp_path, bad):
    path = tmp_path / 'roster.json'
    replace(path, document())
    reader = roster.RosterReader(path)
    previous = reader.read()
    value = document(generation=2)
    if bad == 'pair':
        value['channels'][B] = room('bad-key')
    elif bad == 'status':
        value['channels'][A]['status'] = 'unknown'
    elif bad == 'generation':
        value['generation'] = True
    elif bad == 'rollback':
        value['generation'] = 0
    elif bad == 'reuse':
        value['generation'] = 1
        value['channels'][B] = room(OTHER)
    replace(path, value)
    if bad == 'json':
        path.write_text('{')
    elif bad == 'duplicate':
        path.write_text('{"channels":{},"channels":{}}')
    elif bad == 'missing':
        path.unlink()
    with pytest.raises(roster.RosterError, match='roster unavailable'):
        reader.read()
    assert reader.last_good is previous
    replace(path, document(generation=3))
    assert reader.read().generation == 3
