#!/bin/bash
# The roster is authority; membership enumeration cannot safely infer a CTO identity.
# Consumers invoking this helper get a fresh complete map. An invalid roster leaves the
# previous file intact and exits nonzero, never publishing a silent partial result.
set -euo pipefail
B="$HOME/.hermes/bridge"
OUT="$B/channel-cto.map"
TMP="$(mktemp "$OUT.XXXXXX")"
trap 'rm -f "$TMP"' EXIT
python3 - "$B" > "$TMP" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1] + '/canonical')
from canonical_roster import RosterReader
bridge = Path(sys.argv[1])
roster = RosterReader(bridge / 'required-mentions.json').read()
identities = {}
for public_file in sorted((bridge / 'keys').glob('*.pub')):
    # Handles and local key aliases can differ (cto-clore vs cto-commitlore).
    # Public files alone resolve the signing alias; private keys are never enumerated or read.
    key = public_file.read_text().strip()
    identities.setdefault(key, []).append(public_file.stem)
rows = []
for channel, room in sorted(roster.channels.items()):
    if room['status'] != 'active':
        continue
    aliases = identities.get(room['peer']['pubkey'], [])
    if len(aliases) != 1:
        raise ValueError('roster peer must have exactly one local public identity: ' + channel)
    rows.append((channel, aliases[0]))
for channel, identity in rows:
    print(channel, identity)
print('buzz-identity-map: roster_generation=' + str(roster.generation), file=sys.stderr)
PY
mv "$TMP" "$OUT"
cat "$OUT"
