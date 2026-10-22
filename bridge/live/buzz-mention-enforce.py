#!/usr/bin/env python3
"""Fail-closed project send gate. All other CLI operations pass unchanged.
Signer is resolved by the existing signed CLI's read-only own-profile endpoint;
identity labels never override an inherited signing key. No private key is read.
"""
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).parent / 'canonical'))
from canonical_roster import RosterReader


def fail(message):
    print(json.dumps({'error':'mention_policy','message':message,'retryable':False}),file=sys.stderr)
    raise SystemExit(1)


def split_options(args, valued, flags):
    values = {}; kept = []; i = 0
    while i < len(args):
        arg = args[i]; name, sep, value = arg.partition('=')
        if name in valued:
            if not sep:
                i += 1
                if i == len(args): fail('missing option value')
                value = args[i]
            values.setdefault(name,[]).append(value)
        elif arg in flags:
            kept.append(arg)
        else:
            fail('unsupported or ambiguous send option')
        i += 1
    return values, kept


def relay_origin(value):
    u = urlsplit(value)
    if u.scheme not in ('wss','https') or u.username or u.password or u.query or u.fragment or u.path not in ('','/'):
        fail('unregistered relay')
    return (u.hostname,u.port or 443)


def current_authority(real, prefix, channel, rule, target):
    # Snapshot is an allowlist/expected identity, never current authority.
    def query(*args):
        result = subprocess.run([real,*prefix,*args],stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15)
        if result.returncode: fail('current recipient verification unavailable')
        return json.loads(result.stdout)

    members = query('channels','members','--channel',channel)
    if not isinstance(members,list) or any(not isinstance(m,dict) for m in members):
        fail('current roster unavailable')
    keys = [m.get('pubkey') for m in members]
    if any(keys.count(rule[role]['pubkey']) != 1 for role in ('ceo','peer')):
        fail('current roster does not match expected pair')
    profiles = query('users','get','--pubkey',target['pubkey'])
    if (not isinstance(profiles,list) or len(profiles)!=1 or not isinstance(profiles[0],dict)
            or profiles[0].get('pubkey') != target['pubkey']
            or profiles[0].get('display_name') != target['handle']):
        fail('current counterpart profile does not match expected identity')

    # CEO 판정 2026-09-20: 이 선행 바인딩 검사만 제거한다.
    #
    # 이 검사는 "현재 ACP 바인딩이 있어야 그 바인딩을 복구·논의하는 메시지를 보낼 수 있다" 는
    # 순환 의존이었다 — ACTIVE PRIMARY_CTO 가 0 이면 방이 막히고, 방이 막히면 그것을 푸는 논의를
    # 방에서 할 수 없다. 단일 owner/canonical actor 운영에서 이 검사가 막는 대상(존재하지 않는
    # ACP 바인딩을 사칭하는 메시지)은 존재하지 않는다.
    #
    # 제거 범위는 이 검사뿐이다. 위의 signer/채널 로스터/상대 프로필 검증과, 아래 main() 의
    # reply-to·p-tag 검증 및 새 actor/fallback 금지는 그대로 유지한다.


def main():
    real, *args = sys.argv[1:]
    # Parse only root options. Never search arbitrary content for command words.
    i = 0
    globals_ = {'--relay','--private-key','--auth-tag','--format'}
    while i < len(args) and args[i].split('=',1)[0] in globals_:
        i += 1 if '=' in args[i] else 2
    if args[i:i+2] != ['messages','send']:
        os.execv(real,[real,*args])
    prefix = args[:i]
    if args[i+2:] in (['--help'],['-h']):
        os.execv(real,[real,*args])
    root, _ = split_options(prefix,globals_,set())
    if any(len(v)!=1 for v in root.values()): fail('duplicate root option')
    vals, flags = split_options(args[i+2:],{'--channel','--content','--mention','--reply-to','--kind','--file'}, {'--broadcast'})
    for key in ('--channel','--content'):
        if len(vals.get(key,[])) != 1: fail('exactly one channel and content required')
    if any(len(vals.get(k,[]))>1 for k in ('--reply-to','--kind')): fail('duplicate send option')
    registry = RosterReader(Path(__file__).parent/'required-mentions.json').read()
    channel = vals['--channel'][0]
    rule = registry.active(channel)
    if not rule: fail('unregistered channel')
    relay = root.get('--relay',[os.environ.get('BUZZ_RELAY_URL','')])[0]
    if relay_origin(relay) != relay_origin(registry.relay): fail('unregistered relay')
    # Force full JSON for the readback even if caller requests compact send output.
    lookup_prefix = []
    for k,v in root.items():
        if k != '--format': lookup_prefix += [k,v[0]]
    lookup = subprocess.run([real,*lookup_prefix,'users','get'],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15)
    if lookup.returncode: fail('cannot verify signing identity')
    profile = json.loads(lookup.stdout)
    if not isinstance(profile,list) or len(profile)!=1: fail('ambiguous signing identity')
    signer = profile[0].get('pubkey')
    if signer == rule['ceo']['pubkey']: target = rule['peer']
    elif signer == rule['peer']['pubkey']: target = rule['ceo']
    else: fail('signer not registered for channel')
    if any(p != target['pubkey'] for p in vals.get('--mention',[])):
        fail('explicit recipient does not match channel counterpart')
    content = vals['--content'][0]
    via_stdin = content == '-'
    body = sys.stdin.buffer.read().decode('utf-8') if via_stdin else content
    # Reject known wrong recipients in presentation text as well as explicit tags.
    known = {handle: {key} for handle, key in registry.known_profiles.items()}
    for entry in registry.channels.values():
        if entry['status'] != 'active':
            continue
        for role in ('ceo', 'peer'):
            identity = entry[role]
            known.setdefault(identity['handle'], set()).add(identity['pubkey'])
    for handle, pubkeys in known.items():
        if not pubkeys.intersection((signer,target['pubkey'])) and re.search(r'(?<![\w@-])@'+re.escape(handle)+r'(?![\w-])',body,re.IGNORECASE):
            fail('body names a different registered recipient')
    # A longer handle must not satisfy the required exact token.
    token = '@'+target['handle']
    if not re.search(r'(?<![\w@-])'+re.escape(token)+r'(?![\w-])',body):
        body = token+' '+body
    vals['--content'] = ['-' if via_stdin else body]
    vals['--mention'] = [target['pubkey']]
    forwarded = [*prefix,'messages','send',*flags]
    for k,values in vals.items():
        for v in values:
            forwarded.extend(['--mention',v] if k=='--mention' else [k+'='+v])
    # Revalidate after reading stdin, immediately before dispatch. These CLI/DB
    # reads and send are NOT atomic; relay-side enforcement owns that race.
    current_authority(real,lookup_prefix,channel,rule,target)
    if via_stdin:
        result = subprocess.run([real,*forwarded],input=body.encode('utf-8'))
        raise SystemExit(result.returncode)
    os.execv(real,[real,*forwarded])


if __name__ == '__main__':
    try:
        main()
    except (OSError,ValueError,KeyError,TypeError,sqlite3.Error,subprocess.TimeoutExpired):
        fail('policy or identity verification unavailable')
