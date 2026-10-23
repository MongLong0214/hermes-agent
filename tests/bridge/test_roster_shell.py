"""Run staged launch and identity scripts without a relay, credentials or the user's home."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from test_canonical_roster import ROOT, A, B, OWNER, OTHER, CTO, document, room, replace


def test_staged_scripts_generate_unbounded_subscriptions_and_complete_atomic_map(tmp_path):
    home = tmp_path / 'user'
    bridge = home / '.hermes/bridge'
    for name in ('canonical', 'bin', 'keys'):
        (bridge / name).mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / 'bridge/canonical/canonical_roster.py', bridge / 'canonical')
    for name in ('run-ceo-bridge.sh', 'buzz-identity-map.sh'):
        shutil.copy(ROOT / 'bridge/live' / name, bridge)
    replace(bridge / 'required-mentions.json', document(generation=7, channels={A: room(), B: room(OTHER, 'retired')}))
    (bridge / 'keys/owner.pub').write_text(OWNER)
    (bridge / 'keys/cto-local.pub').write_text(CTO)
    # Pure fixture bytes, never copied from real signing material.
    (bridge / 'keys/ceo2.priv').write_text('fixture-only')
    (bridge / 'preflight.sh').write_text('exit 0\n')
    (bridge / 'relay-host.txt').write_text('relay.invalid:3000')
    (bridge / 'bin/buzz-acp').write_text('#!/bin/bash\nexec "' + sys.executable + '" -c '\
        + "'import os,json,sys; print(json.dumps({\"env\":{k:v for k,v in os.environ.items() if k.startswith(\"BUZZ_ACP_\")},\"args\":sys.argv[1:]}))' \"$@\"\n")
    (bridge / 'bin/buzz-acp').chmod(0o700)
    env = {**os.environ, 'HOME': str(home), 'API_SERVER_KEY': 'fixture-api',
           'BUZZ_ACP_CHANNELS': A, 'BUZZ_ACP_RESPOND_TO_ALLOWLIST': OTHER}
    run = subprocess.run(['bash', str(bridge / 'run-ceo-bridge.sh')], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    settings = json.loads(run.stdout)['env']
    assert settings['BUZZ_ACP_RESPOND_TO'] == 'anyone'
    assert settings['BUZZ_ACP_MULTIPLE_EVENT_HANDLING'] == 'queue'
    assert 'BUZZ_ACP_RESPOND_TO_ALLOWLIST' not in settings and 'BUZZ_ACP_CHANNELS' not in settings
    # TOML is generated data, not source inspection. Python 3.9 uses the installed tomli parser.
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    config = tomllib.loads((bridge / 'canonical/owner-subscriptions.toml').read_text())
    mentions, broadcasts = config['rules']
    assert mentions['channels'] == broadcasts['channels'] == 'all'
    assert mentions['require_mention'] and not broadcasts['require_mention']
    assert broadcasts['filter'] == 'author == "' + OWNER + '"'
    def map_run():
        return subprocess.run(['bash', str(bridge / 'buzz-identity-map.sh')], env=env, capture_output=True, text=True)
    mapped = map_run()
    assert mapped.returncode == 0 and mapped.stdout == A + ' cto-local\n'
    assert 'roster_generation=7' in mapped.stderr
    before = (bridge / 'channel-cto.map').read_bytes()
    (bridge / 'keys/ambiguous.pub').write_text(CTO)
    ambiguous = map_run()
    assert ambiguous.returncode != 0 and 'exactly one' in ambiguous.stderr
    assert (bridge / 'channel-cto.map').read_bytes() == before
    (bridge / 'keys/ambiguous.pub').unlink()
    (bridge / 'keys/cto-local.pub').unlink()
    assert map_run().returncode != 0
    assert (bridge / 'channel-cto.map').read_bytes() == before
    (bridge / 'required-mentions.json').write_text('{')
    failed = map_run()
    assert failed.returncode != 0 and 'roster unavailable' in failed.stderr
    assert (bridge / 'channel-cto.map').read_bytes() == before


@pytest.mark.parametrize('version', [1, 2])
def test_send_gate_uses_active_roster_without_requiring_known_profiles(tmp_path, version):
    bridge = tmp_path / 'bridge'
    (bridge / 'canonical').mkdir(parents=True)
    shutil.copy(ROOT / 'bridge/canonical/canonical_roster.py', bridge / 'canonical')
    gate = bridge / 'buzz-mention-enforce.py'
    shutil.copy(ROOT / 'bridge/live/buzz-mention-enforce.py', gate)
    value = document(version, 1, {A: room(), B: {**room(OTHER), 'peer': {'pubkey': OTHER, 'handle': 'cto-other'}}})
    replace(bridge / 'required-mentions.json', value)
    real = bridge / 'buzz-real'
    # No relay is contacted. This fixture exercises the actual gate subprocess and final CLI argv.
    real.write_text('#!' + sys.executable + '\n' + '''import json, sys
args = sys.argv[1:]
if args[:2] == ['users', 'get']:
    if '--pubkey' in args:
        print(json.dumps([{'pubkey': 'b'*64, 'display_name': 'cto-project'}]))
    else:
        print(json.dumps([{'pubkey': 'a'*64}]))
elif args[:2] == ['channels', 'members']:
    print(json.dumps([{'pubkey': 'a'*64}, {'pubkey': 'b'*64}]))
else:
    print(json.dumps({'args': args, 'body': sys.stdin.read()}))
''')
    real.chmod(0o700)
    env = {**os.environ, 'HOME': str(tmp_path), 'BUZZ_RELAY_URL': 'wss://relay.invalid'}
    def send(body):
        return subprocess.run([sys.executable, str(gate), str(real), 'messages', 'send',
                               '--channel', A, '--content', '-'], input=body, env=env,
                              capture_output=True, text=True)
    sent = send('answer')
    assert sent.returncode == 0, sent.stderr
    payload = json.loads(sent.stdout)
    assert payload['body'] == '@cto-project answer'
    assert payload['args'][payload['args'].index('--mention') + 1] == CTO
    assert send('@cto-other wrong recipient').returncode != 0
    value['generation'] = 2
    value['channels'][A]['status'] = 'retired'
    replace(bridge / 'required-mentions.json', value)
    refused = send('retired')
    assert refused.returncode != 0 and 'unregistered channel' in refused.stderr
    (bridge / 'required-mentions.json').write_text('{')
    assert send('malformed').returncode != 0
