# CEO owner broadcasts (2026-10-20)

Runtime change: install `bridge/canonical/acp-canonical-shim.py` and switch the
single CEO buzz-acp process from `--subscribe mentions` to `--subscribe config
--config ~/.hermes/bridge/canonical/owner-subscriptions.toml` (equivalently,
`BUZZ_ACP_SUBSCRIBE=config` and `BUZZ_ACP_CONFIG=<absolute path>`).

Binary help/schema and filter symbols confirm config rules support channel,
author and mention gates. This expresses the requested union, so it is
preferred to `all`. Rules match once; keep the mention rule first. Config mode
ignores `--channels` and `--kinds`: explicitly copy their effective values into
BOTH rules. Generate the file with the existing launch environment loaded:

```sh
python3 - <<'PY' > ~/.hermes/bridge/canonical/owner-subscriptions.toml
import json, os, uuid
channels = [str(uuid.UUID(c.strip())) for c in os.environ['BUZZ_ACP_CHANNELS'].split(',')]
assert channels
kinds = [int(k.strip()) for k in os.environ.get('BUZZ_ACP_KINDS', '9,46010').split(',')]
owner = 'd046238dd1422955898205bd60507992cb101a1ea2ecfa6e52fd00d5ddca98c4'
for mention, tag in [(True, '@mention'), (False, 'owner-broadcast')]:
    print('[[rules]]')
    print('channels =', json.dumps(channels))
    print('kinds =', json.dumps(kinds))
    print('require_mention =', str(mention).lower())
    print('prompt_tag =', json.dumps(tag))
    if not mention:
        print('filter =', json.dumps('author == "' + owner + '"'))
PY
```

The binary's default kinds are 9 and 46010; preserve an explicit `--kinds`
override instead if present. Keep `BUZZ_ACP_RESPOND_TO=allowlist`, its existing
allowlist/owner, ignore-self, agent command, canonical binding and all other
settings. The shim silently drops other authors' broadcasts before any hold,
receipt, notice or submit, requires a paired channel for owner broadcasts, and
skips their published event IDs, including IDs previously delivered as mentions.
Mention receipt replays and in-flight exclusion retain their existing behavior;
the canonical receipt prevents a second turn and publication remains once per ID.

Deploy (not performed): using the earlier shim-cutover procedure, save the old
shim and launch settings, copy this repo's shim to
`~/.hermes/bridge/canonical/acp-canonical-shim.py`, generate the config above,
and change the existing buzz-acp launch flags/env. Let an active turn finish,
then stop/start ONLY that buzz-acp process and its shim child, with the same
credentials and canonical binding. Keep the gateway and CEO session running;
retain `published/`, receipts and session storage. Do not launch a second bridge.

Rollback: restore the saved shim and launch settings (`--subscribe mentions`,
or `BUZZ_ACP_SUBSCRIBE=mentions`), then cut over ONLY buzz-acp again. Preserve
the same binding, session and journals. No live files or processes were changed.

Validation: bridge suite 393 passed, 16 localhost tests skipped (no-network rule).
