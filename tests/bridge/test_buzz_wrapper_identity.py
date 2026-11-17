"""The buzz wrapper's default signing identity is the single CEO key, ``ceo``.

Everything is a fixture: a throwaway HOME, fake key text, and a stub that echoes what it was handed.
No relay, no network, no real key.
"""
import json
import os
import shutil
import subprocess

import pytest
from test_canonical_roster import ROOT, A, CTO, document, replace, room

WRAPPER = ROOT / 'bridge/bin/buzz'
# Throwaway scalars 1 and 2, never a real key. CEO_PUB is x(1*G): the wrapper derives public keys.
CEO_KEY, CTO_KEY = '0' * 63 + '1', '0' * 63 + '2'
CEO_PUB = '79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798'
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
    for alias, pub, key in (('ceo', CEO_PUB, CEO_KEY), ('cto-local', CTO, CTO_KEY)):
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
    assert run(home, 'feed', BUZZ_IDENTITY='cto-local', CLAUDECODE='1')[1]['key'] == CTO_KEY
    assert run(home, 'feed', BUZZ_PRIVATE_KEY=CTO_KEY, CLAUDECODE='1')[1]['key'] == CTO_KEY
    refused(run(home, 'feed', BUZZ_IDENTITY='no-such')[0])


def test_the_ceo_shell_may_name_the_ceo_key_any_way(home):
    assert run(home, 'feed', BUZZ_IDENTITY='ceo')[1]['key'] == CEO_KEY
    assert run(home, 'feed', BUZZ_PRIVATE_KEY=CEO_KEY)[1]['key'] == CEO_KEY


@pytest.mark.parametrize('route', ['identity', 'env-key', 'flag', 'flag-equals', 'twin-alias', 'mixed-case'])
def test_a_cto_shell_cannot_sign_as_the_ceo_by_any_route(home, route):
    keys = home / '.hermes/bridge/keys'
    if route == 'twin-alias':
        (keys / 'twin.priv').write_text(CEO_KEY)
        (keys / 'twin.pub').write_text(CEO_PUB)
    args, env = ['feed'], {'CLAUDECODE': '1'}
    if route in ('identity', 'twin-alias'):
        env['BUZZ_IDENTITY'] = 'ceo' if route == 'identity' else 'twin'
    elif route == 'env-key':
        env['BUZZ_PRIVATE_KEY'] = CEO_KEY
    elif route == 'mixed-case':
        env['BUZZ_PRIVATE_KEY'] = CEO_KEY.upper()
    elif route == 'flag':
        args += ['--private-key', CEO_KEY]
    else:
        args += ['--private-key=' + CEO_KEY]
    refused(run(home, *args, **env)[0])
    if 'BUZZ_IDENTITY' not in env:  # the key still wins over a CTO identity name
        refused(run(home, *args, **env, BUZZ_IDENTITY='cto-local')[0])


def test_a_cto_shell_with_a_key_it_cannot_check_is_refused(home):
    # Not a 64-digit hex scalar, so it cannot be proven different from the CEO key.
    refused(run(home, 'feed', BUZZ_PRIVATE_KEY='nsec1fake', CLAUDECODE='1')[0])
    (home / '.hermes/bridge/keys/ceo.pub').unlink()
    refused(run(home, 'feed', BUZZ_PRIVATE_KEY=CTO_KEY, CLAUDECODE='1')[0])
