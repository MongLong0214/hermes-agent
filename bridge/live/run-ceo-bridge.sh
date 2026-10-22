#!/bin/bash
# CEO(Hermes) ↔ buzz 브릿지.
#
# buzz-acp가 릴레이의 메시지를 받아 stdio ACP로 Hermes에 넘기고, 응답을 채널에
# 되돌린다. 이 스크립트는 launchd가 상시 띄운다.
#
# 설계 메모
# - 모든 자산은 ~/.hermes/bridge 아래 영구 위치에 둔다. 예전에 세션 임시
#   디렉터리(/private/tmp/...)에 두었다가 세션 종료와 함께 사라질 뻔했다.
# - PATH에 /usr/bin이 반드시 있어야 한다. macOS 키체인 조회(`security`)가
#   거기 있고, 없으면 Hermes가 자격증명을 못 읽어 인증 실패로 오탐한다.
# - buzz CLI도 PATH에 있어야 한다. 에이전트가 답장을 발행할 때 쓴다.
set -euo pipefail

B="$HOME/.hermes/bridge"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$B/bin:$HOME/.local/bin:/opt/homebrew/bin:$HOME/.nvm/versions/node/v24.18.0/bin:${PATH:-}"

# Config rules cover present and future member rooms; the gateway enforces roster pairs.
# --channels is ignored in config subscription mode. Do not retain a stale environment list.
unset BUZZ_ACP_CHANNELS

# 릴레이 주소는 매 기동마다 자가 점검으로 결정한다. LAN IP가 바뀌면 커뮤니티
# 등록값도 함께 갱신되므로, DHCP 변경에 채널이 조용히 죽지 않는다.
bash "$B/preflight.sh" || true
HOSTPORT="$(cat "$B/relay-host.txt" 2>/dev/null || echo "172.30.1.88:3000")"
export BUZZ_RELAY_URL="${BUZZ_RELAY_URL:-wss://${HOSTPORT}}"
export BUZZ_PRIVATE_KEY="$(cat "$B/keys/ceo2.priv")"
export BUZZ_ACP_AGENT_OWNER="$(cat "$B/keys/owner.pub")"
# buzz-acp freezes the pubkey allowlist at startup. Let events reach the canonical gateway,
# which checks explicit (author, room) grants per request; private relay membership remains required.
export BUZZ_ACP_RESPOND_TO=anyone
unset BUZZ_ACP_RESPOND_TO_ALLOWLIST
# PR #98: queue preserves each arriving event; the shim splits and verifies bundled originals.
export BUZZ_ACP_MULTIPLE_EVENT_HANDLING=queue

# #96: owner broadcasts and CEO mentions in all present/future member rooms. The owner is
# one public identity, read from owner.pub; room authorization belongs to the live roster gate.
OWNER_SUBS="$B/canonical/owner-subscriptions.toml"
python3 - "$B" > "$OWNER_SUBS.tmp" <<'TOMLEOF'
import json, sys
sys.path.insert(0, sys.argv[1] + '/canonical')
from canonical_roster import RosterReader, read_owner
from pathlib import Path
bridge = Path(sys.argv[1])
roster = RosterReader(bridge / 'required-mentions.json').read()
owner = read_owner(bridge / 'keys' / 'owner.pub')
print('# roster_generation =', roster.generation)
for name, mention, tag in [('mentions', True, '@mention'), ('owner-broadcast', False, 'owner-broadcast')]:
    print('[[rules]]'); print('name =', json.dumps(name)); print('channels = "all"'); print('kinds = [9, 46010]')
    print('require_mention =', str(mention).lower()); print('prompt_tag =', json.dumps(tag))
    if not mention: print('filter =', json.dumps('author == "' + owner + '"'))
    print()
TOMLEOF
mv "$OWNER_SUBS.tmp" "$OWNER_SUBS"
export BUZZ_ACP_SUBSCRIBE=config
export BUZZ_ACP_CONFIG="$OWNER_SUBS"

# 제출 경로 (2026-09-27). buzz-acp 는 --agent-command 를 ACP stdio 로 띄운다. 그 자리에
# `hermes acp` 를 두면 **새 세션**이 뜬다 — 오너가 금지한 포크이고, 3,788 세션을 가진 CEO 의
# 전사가 답하지 않는다. 대신 canonical surface 로 제출하는 shim 을 둔다. 그 shim 은 세션을
# 소유하지 않으므로 포크를 회피하는 것이 아니라 **할 수 없다**.
if [ -z "${API_SERVER_KEY:-}" ]; then
  API_SERVER_KEY=$(sed -n "s/^API_SERVER_KEY=//p" "$HOME/.hermes/.env" | head -1 | tr -d "\"'")
fi
export API_SERVER_KEY
export CANONICAL_BINDING="${CANONICAL_BINDING:-acp-canonical-ceo}"
# Event headers supply author and room; these retired startup constants never supplied authority.
unset CANONICAL_AUTHOR_ID CANONICAL_CHANNEL_ID
# 2026-10-04: #627 canary 1·2 PASS (같은 thread 자동 답장 1회, 새 세션 0건) 뒤 CEO 사전 승인으로 live.
# 롤백: 이 줄을 지우고 buzz-acp 를 재시작하면 shim 기본값 quarantine 으로 돌아간다.
export CANONICAL_FORWARD=live
export RUST_LOG="${RUST_LOG:-buzz_acp=info}"

# 2026-09-27: heartbeat 를 0(buzz-acp 자신의 기본값)으로 되돌린다. 이 자리의 에이전트는 더 이상
# 자기 도구로 피드를 당겨올 수 있는 에이전트가 아니라, CEO 의 살아있는 세션으로 텍스트를 넘기는
# shim 이다. heartbeat 는 사람이 보내지 않은 지시를 그 세션에 owner 로서 집어넣는다. 같은 날 첫
# heartbeat 는 HTTP 409(그 순간 세션이 바빴다)로만 막혔다 — 그건 가드가 아니라 타이밍이다.
#
# 2026-10-04: 이 주석이 원래 exec 의 `\` 이어진 줄들 사이에 있었다. 주석이 명령을 끝내서 아래
# 인자 4개가 한 번도 buzz-acp 에 전달되지 않았다(실행 중 인자는 --agent-command, --agent-args
# 둘뿐이었다). 주석을 exec 밖으로 옮긴다. idle-timeout 은 300 이 아니라 900 이다: shim 은 CEO
# 턴의 HTTP 응답을 기다리는 동안(CANONICAL_TIMEOUT_SECS 900) stdout 에 아무것도 쓰지 않으므로,
# 300 이면 5 분 넘는 CEO 턴이 끊기고 답장도 게시되지 않는다. 턴 예산(900)과 같게 둔다.
# buzz-acp 는 idle_timeout < max_turn_duration 이 아니면 시작을 거부한다(900=900 으로 한 번 내려갔다),
# 그래서 max-turn 은 shim 의 900 이 먼저 끝나도록 960 이다.
exec "$B/bin/buzz-acp" \
  --agent-command "$B/canonical/acp-canonical-shim.py" \
  --agent-args=acp \
  --heartbeat-interval "${BUZZ_ACP_HEARTBEAT_INTERVAL:-0}" \
  --turn-liveness-secs "${BUZZ_ACP_TURN_LIVENESS_SECS:-15}" \
  --idle-timeout 900 \
  --max-turn-duration 960
