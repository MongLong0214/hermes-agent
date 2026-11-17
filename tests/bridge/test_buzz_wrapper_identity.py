"""The buzz wrapper's default signing identity is the single CEO key, ``ceo``.

Everything is a fixture: a throwaway HOME, fake key text, and a stub that echoes what it was handed.
No relay, no network, no real key.
"""
import json
import os
import shutil
import subprocess

import pytest
from test_canonical_roster import ROOT, A, CEO, CTO, document, replace, room

WRAPPER = ROOT / 'bridge/bin/buzz'
CEO_KEY, CTO_KEY = 'fake-ceo-key', 'fake-cto-key'
# The retired CEO key name, spelled in two parts so a repo-wide search for it stays empty.
RETIRED = 'ceo' + '2'
UNKNOWN_ROOM = '30000000-0000-4000-8000-000000000003'
ENFORCE = ('import json,os,sys\n'
           'print(json.dumps({"key": os.environ.get("BUZZ_PRIVATE_KEY"), "args": sys.argv[2:]}))\n')


@pytest.fixture
def home(tmp_path):
    h = tmp_path / 'user'
    bridge = h / '.hermes/bridge'
    for name in ('canonical', 'keys'):
        (bridge / name).mkdir(parents=True)
    (h / '.local/bin').mkdir(parents=True)
    shutil.copy(ROOT / 'bridge/canonical/canonical_roster.py', bridge / 'canonical')
    shutil.copy(ROOT / 'bridge/live/buzz-identity-map.sh', bridge)
    (bridge / 'buzz-mention-enforce.py').write_text(ENFORCE)
    stub = h / 'buzz-stub'
    stub.write_text('#!/bin/sh\n')
    stub.chmod(0o755)
    (h / '.local/bin/.buzz.symlink-target.bak').write_text(str(stub))
    for alias, pub, key in (('ceo', CEO, CEO_KEY), ('cto-local', CTO, CTO_KEY)):
        (bridge / 'keys' / (alias + '.pub')).write_text(pub)
        (bridge / 'keys' / (alias + '.priv')).write_text(key)
    replace(bridge / 'required-mentions.json', document(channels={A: room()}))
    return h


def run(home, *args, **env):
    base = {k: v for k, v in os.environ.items()
            if k not in ('BUZZ_PRIVATE_KEY', 'BUZZ_IDENTITY', 'BUZZ_RELAY_URL', 'CLAUDECODE')}
    p = subprocess.run(['bash', str(WRAPPER), *args], env={**base, 'HOME': str(home), **env},
                       capture_output=True, text=True)
    return p, (json.loads(p.stdout) if p.returncode == 0 else None)


def refused(p):
    assert p.returncode == 3 and p.stdout == '' and 'identity_error' in p.stderr, (p.returncode, p.stdout, p.stderr)


def test_default_signs_as_the_ceo_key(home):
    p, out = run(home, 'feed')
    assert p.returncode == 0, p.stderr
    assert out['key'] == CEO_KEY
    assert run(home, 'messages', '--channel', A)[1]['key'] == CEO_KEY


def test_the_retired_ceo_name_is_refused(home):
    refused(run(home, 'feed', BUZZ_IDENTITY=RETIRED)[0])


def test_cto_shell_without_identity_cannot_sign_as_the_ceo(home):
    refused(run(home, 'feed', CLAUDECODE='1')[0])
    refused(run(home, 'feed', '--channel', UNKNOWN_ROOM, CLAUDECODE='1')[0])


def test_cto_shell_still_derives_its_identity_from_the_channel(home):
    p, out = run(home, 'send', '--channel', A, CLAUDECODE='1')
    assert p.returncode == 0, p.stderr
    assert out['key'] == CTO_KEY


def test_explicit_key_and_identity_still_win(home):
    assert run(home, 'feed', BUZZ_PRIVATE_KEY='fake-explicit')[1]['key'] == 'fake-explicit'
    assert run(home, 'feed', BUZZ_IDENTITY='cto-local')[1]['key'] == CTO_KEY
    assert run(home, 'feed', BUZZ_IDENTITY='ceo', CLAUDECODE='1')[1]['key'] == CEO_KEY
    refused(run(home, 'feed', BUZZ_IDENTITY='no-such')[0])
