#!/bin/bash
# Recover Buzz events the CEO's canonical shim refused or never received, by event ID.
# usage: ceo-shim-recover.sh <channel-uuid> <event-id> [<event-id> ...]
#
# Same relay, signing key, gateway key and binding as run-ceo-bridge.sh, so the recovered event is
# delivered exactly as the live shim would deliver it: the original is fetched once, its hash,
# signature and channel are verified, and an event already answered is skipped (rerun-safe).
# It never prints key material. Prints one "<event-id> <state>" line per event.
set -euo pipefail
B="$HOME/.hermes/bridge"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$B/bin:$HOME/.local/bin:/opt/homebrew/bin:${PATH:-}"
HOSTPORT="$(cat "$B/relay-host.txt" 2>/dev/null || echo "172.30.1.88:3000")"
export BUZZ_RELAY_URL="${BUZZ_RELAY_URL:-wss://${HOSTPORT}}"
export BUZZ_PRIVATE_KEY="$(cat "$B/keys/ceo2.priv")"
export BUZZ_ACP_AGENT_OWNER="$(cat "$B/keys/owner.pub")"
if [ -z "${API_SERVER_KEY:-}" ]; then
  API_SERVER_KEY=$(sed -n "s/^API_SERVER_KEY=//p" "$HOME/.hermes/.env" | head -1 | tr -d "\"'")
fi
export API_SERVER_KEY
export CANONICAL_BINDING="${CANONICAL_BINDING:-acp-canonical-ceo}"
unset CANONICAL_AUTHOR_ID CANONICAL_CHANNEL_ID
export CANONICAL_FORWARD=live
exec /usr/bin/python3 "$B/canonical/acp-canonical-shim.py" recover "$@"
