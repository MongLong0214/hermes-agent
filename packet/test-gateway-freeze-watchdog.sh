#!/usr/bin/env bash
set -u

base_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
watchdog="$base_dir/gateway-freeze-watchdog.sh"
plist_template="$base_dir/ai.hermes.gateway-freeze-watchdog.plist"
receipt="$base_dir/receipt.json"
passes=0
failures=0
roots=()

cleanup() {
  local root
  for root in "${roots[@]:-}"; do
    rm -rf -- "$root"
  done
}
trap cleanup EXIT

pass() {
  passes=$((passes + 1))
  printf 'ok %s\n' "$1"
}

fail() {
  failures=$((failures + 1))
  printf 'not ok %s\n' "$1"
}

assert_eq() {
  local name=$1 expected=$2 actual=$3
  if [[ "$actual" == "$expected" ]]; then
    pass "$name"
  else
    fail "$name"
    printf '  expected: %s\n  actual:   %s\n' "$expected" "$actual"
  fi
}

assert_no_state() {
  local name=$1 root=$2
  if [[ ! -s "$root/state" ]]; then
    pass "$name"
  else
    fail "$name"
  fi
}

make_root() {
  local root
  root=$(mktemp -d)
  roots+=("$root")
  printf 'gen-a|123|100\n' > "$root/generation"
  printf '400\n' > "$root/log_epoch"
  printf 'route-a|gen-a|300\n' > "$root/inbound"
  printf 'route-a|gen-a|300\n' > "$root/progress"
  cat > "$root/inspect" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "${FAKE_INSPECT_OUTPUT:-}"
exit "${FAKE_INSPECT_STATUS:-0}"
EOF
  chmod 700 "$root/inspect"
  printf '%s' "$root"
}

make_clock() {
  local root=$1
  cat > "$root/clock" <<'EOF'
#!/usr/bin/env bash
printf '1000\n'
EOF
  chmod 700 "$root/clock"
}

invoke() {
  local root=$1 digest=${2:-} now output
  if [[ -z "$digest" ]]; then
    digest=$(shasum -a 256 "$watchdog" 2>/dev/null | cut -d ' ' -f 1 || true)
  fi
  now=${WATCHDOG_NOW_FOR_TEST-1000}
  output=$(WATCHDOG_ROOT="$root" WATCHDOG_NOW="$now" WATCHDOG_TIME_CMD="${WATCHDOG_TIME_CMD:-}" \
    PATH="${WATCHDOG_TEST_PATH:-$PATH}" PROHIBITED_ACTION_LOG="${PROHIBITED_ACTION_LOG:-}" \
    WATCHDOG_INSPECT_CMD="$root/inspect" WATCHDOG_SCRIPT_PATH="$watchdog" \
    WATCHDOG_EXPECTED_SHA256="$digest" "$watchdog" 2>&1 || true)
  printf '%s' "$output"
}

render_plist() {
  local root=$1 digest rendered
  digest=$(shasum -a 256 "$watchdog" | cut -d ' ' -f 1)
  rendered="$root/rendered.plist"
  python3 - "$plist_template" "$rendered" "$watchdog" "$digest" "$root" <<'PY' || return 1
import plistlib
import sys
from pathlib import Path

template, rendered, script, digest, root = sys.argv[1:]
text = Path(template).read_text()
values = {
    "__WATCHDOG_SCRIPT__": script,
    "__WATCHDOG_PATH__": str(Path(root) / "fixture-bin"),
    "__WATCHDOG_ROOT__": root,
    "__WATCHDOG_INSPECT_CMD__": str(Path(root) / "inspect"),
    "__WATCHDOG_TIME_CMD__": str(Path(root) / "clock"),
    "__WATCHDOG_SCRIPT_PATH__": script,
    "__WATCHDOG_EXPECTED_SHA256__": digest,
    "__WATCHDOG_HASH_CMD__": "shasum",
}
for key, value in values.items():
    text = text.replace(key, value)
assert "__WATCHDOG_" not in text
Path(rendered).write_text(text)
with Path(rendered).open("rb") as stream:
    data = plistlib.load(stream)
assert data["Label"] == "ai.hermes.gateway-freeze-watchdog"
assert data["StartInterval"] == 300
assert data["ProgramArguments"] == [script]
expected_env = {
    "PATH",
    "WATCHDOG_ROOT",
    "WATCHDOG_INSPECT_CMD",
    "WATCHDOG_TIME_CMD",
    "WATCHDOG_SCRIPT_PATH",
    "WATCHDOG_EXPECTED_SHA256",
    "WATCHDOG_HASH_CMD",
}
assert set(data["EnvironmentVariables"]) == expected_env
assert data["EnvironmentVariables"]["WATCHDOG_EXPECTED_SHA256"] == digest
assert data["EnvironmentVariables"]["WATCHDOG_SCRIPT_PATH"] == script
PY
  plutil -lint "$rendered" >/dev/null
}

test_inspection_failure_is_unknown() {
  local root output
  root=$(make_root)
  output=$(FAKE_INSPECT_STATUS=1 invoke "$root")
  assert_eq "inspection failure is A_UNKNOWN with zero-strike decision" \
    "A_UNKNOWN B_UNKNOWN DECISION_UNKNOWN" "$output"
  assert_no_state "inspection failure creates no strike state" "$root"
}

test_zero_sockets_after_stale_log_is_a_suspect() {
  local root output
  root=$(make_root)
  rm -f -- "$root/progress"
  output=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  assert_eq "successful full inspection with zero sockets is A_SUSPECT" \
    "A_SUSPECT B_UNKNOWN DECISION_UNKNOWN" "$output"
  assert_no_state "zero-socket B_UNKNOWN creates no strike state" "$root"
}

test_matching_socket_is_healthy() {
  local root output
  root=$(make_root)
  output=$(FAKE_INSPECT_OUTPUT="ESTABLISHED gateway" invoke "$root")
  assert_eq "matching established socket is A_HEALTHY" \
    "A_HEALTHY B_SKIPPED DECISION_HEALTHY" "$output"
}

test_fresh_log_is_healthy() {
  local root output
  root=$(make_root)
  printf '401\n' > "$root/log_epoch"
  output=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  assert_eq "zero sockets with fresh log is A_HEALTHY" \
    "A_HEALTHY B_SKIPPED DECISION_HEALTHY" "$output"
}

test_same_route_post_inbound_stale_progress_starts_strike() {
  local root output state
  root=$(make_root)
  output=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  assert_eq "same-route post-inbound stale progress starts first strike" \
    "A_SUSPECT B_SUSPECT DECISION_FIRST_STRIKE" "$output"
  state=$(<"$root/state")
  assert_eq "first strike is bound to generation and observation class" \
    "gen-a|A_SOCKET_ZERO_B_PROGRESS_STALE|1" "$state"
}

test_invalid_progress_is_unknown() {
  local root output
  root=$(make_root)
  rm -f -- "$root/progress"
  output=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  assert_eq "missing progress is B_UNKNOWN" "A_SUSPECT B_UNKNOWN DECISION_UNKNOWN" "$output"
  assert_no_state "missing progress has zero strike" "$root"

  root=$(make_root)
  printf 'route-a|gen-a|bad\n' > "$root/progress"
  output=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  assert_eq "malformed progress is B_UNKNOWN" "A_SUSPECT B_UNKNOWN DECISION_UNKNOWN" "$output"

  root=$(make_root)
  printf 'route-a|gen-a|0\n' > "$root/progress"
  output=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  assert_eq "zero progress is B_UNKNOWN" "A_SUSPECT B_UNKNOWN DECISION_UNKNOWN" "$output"

  root=$(make_root)
  printf 'route-b|gen-a|300\n' > "$root/progress"
  output=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  assert_eq "wrong route progress is B_UNKNOWN" "A_SUSPECT B_UNKNOWN DECISION_UNKNOWN" "$output"

  root=$(make_root)
  printf 'route-a|gen-b|300\n' > "$root/progress"
  output=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  assert_eq "wrong generation progress is B_UNKNOWN" "A_SUSPECT B_UNKNOWN DECISION_UNKNOWN" "$output"

  root=$(make_root)
  printf 'route-a|gen-a|99\n' > "$root/progress"
  output=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  assert_eq "pre-process progress is B_UNKNOWN" "A_SUSPECT B_UNKNOWN DECISION_UNKNOWN" "$output"

  root=$(make_root)
  printf 'route-a|gen-a|299\n' > "$root/progress"
  output=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  assert_eq "pre-inbound progress is B_UNKNOWN" "A_SUSPECT B_UNKNOWN DECISION_UNKNOWN" "$output"
}

test_nonwedge_progress_is_healthy() {
  local root output
  root=$(make_root)
  printf 'route-a|gen-a|401\n' > "$root/progress"
  output=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  assert_eq "fresh progress is B_HEALTHY" "A_SUSPECT B_HEALTHY DECISION_HEALTHY" "$output"

  root=$(make_root)
  printf 'route-a|gen-a|301\n' > "$root/response"
  output=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  assert_eq "later response is B_HEALTHY" "A_SUSPECT B_HEALTHY DECISION_HEALTHY" "$output"
}

test_injected_clock_is_used() {
  local root output
  root=$(make_root)
  make_clock "$root"
  output=$(WATCHDOG_NOW_FOR_TEST= WATCHDOG_TIME_CMD="$root/clock" FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  assert_eq "injected clock supports deterministic inspection" \
    "A_SUSPECT B_SUSPECT DECISION_FIRST_STRIKE" "$output"
}

test_generation_drift_cancels_strike() {
  local root output state
  root=$(make_root)
  cat > "$root/drift" <<'EOF'
#!/usr/bin/env bash
printf 'gen-b|124|101\n' > "$1/generation"
EOF
  chmod 700 "$root/drift"
  output=$(WATCHDOG_BEFORE_STRIKE_CMD="$root/drift" FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  assert_eq "generation drift cancels the transition" \
    "A_UNKNOWN B_UNKNOWN DECISION_GENERATION_DRIFT" "$output"
  state=$(<"$root/state")
  assert_eq "generation drift clears prior strike state" "" "$state"
}

test_class_and_generation_bound_strikes() {
  local root output state
  root=$(make_root)
  printf 'gen-a|other|1\n' > "$root/state"
  output=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  assert_eq "different observation class starts a first strike" \
    "A_SUSPECT B_SUSPECT DECISION_FIRST_STRIKE" "$output"
  state=$(<"$root/state")
  assert_eq "different observation class is replaced deterministically" \
    "gen-a|A_SOCKET_ZERO_B_PROGRESS_STALE|1" "$state"
}

test_second_strike_emits_one_recommendation() {
  local root first first_recommendation second third recommendation
  root=$(make_root)
  first=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  if [[ -e "$root/recommendation" ]]; then
    first_recommendation="present"
  else
    first_recommendation="absent"
  fi
  second=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  third=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  recommendation=$(<"$root/recommendation")
  assert_eq "first strike has no recommendation decision" \
    "A_SUSPECT B_SUSPECT DECISION_FIRST_STRIKE" "$first"
  assert_eq "first strike writes no recommendation" "absent" "$first_recommendation"
  assert_eq "second strike emits fixed recommendation decision" \
    "A_SUSPECT B_SUSPECT DECISION_RESTART_RECOMMENDED" "$second"
  assert_eq "recommendation is fixed and metadata-minimal" "RESTART_RECOMMENDED" "$recommendation"
  assert_eq "later identical observation does not emit again" \
    "A_SUSPECT B_SUSPECT DECISION_ALREADY_RECORDED" "$third"
}

test_concurrent_transition_is_serialized() {
  local root first second entered release out first_pid
  root=$(make_root)
  entered="$root/entered"
  release="$root/release"
  out="$root/first.out"
  mkfifo "$entered" "$release"
  cat > "$root/barrier" <<'EOF'
#!/usr/bin/env bash
printf 'entered\n' > "$BARRIER_ENTERED"
IFS= read -r ignored < "$BARRIER_RELEASE"
EOF
  chmod 700 "$root/barrier"
  BARRIER_ENTERED="$entered" BARRIER_RELEASE="$release" WATCHDOG_BEFORE_STRIKE_CMD="$root/barrier" \
    FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root" > "$out" &
  first_pid=$!
  IFS= read -r ignored < "$entered"
  second=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  printf 'release\n' > "$release"
  wait "$first_pid"
  first=$(<"$out")
  assert_eq "concurrent invocation observes serialized busy decision" \
    "A_SUSPECT B_SUSPECT DECISION_BUSY" "$second"
  assert_eq "lock owner completes a single first transition" \
    "A_SUSPECT B_SUSPECT DECISION_FIRST_STRIKE" "$first"
}

start_transition_lock_holder() {
  local root=$1 entered=$2 release=$3 holder ignored
  holder="$root/lock-holder"
  cat > "$holder" <<'EOF'
#!/usr/bin/env bash
mkdir "$1/transition.lock"
printf 'fixture-holder\n' > "$1/transition.lock/owner"
printf 'entered\n' > "$LOCK_ENTERED"
IFS= read -r ignored < "$LOCK_RELEASE"
rm -f -- "$1/transition.lock/owner"
rmdir -- "$1/transition.lock"
EOF
  chmod 700 "$holder"
  LOCK_ENTERED="$entered" LOCK_RELEASE="$release" "$holder" "$root" &
  lock_holder_pid=$!
  IFS= read -r ignored < "$entered"
}

release_transition_lock_holder() {
  local release=$1
  printf 'release\n' > "$release"
  wait "$lock_holder_pid"
}

assert_contended_unknown_reset() {
  local name=$1 kind=$2 root entered release unknown next recommendation
  root=$(make_root)
  printf 'gen-a|A_SOCKET_ZERO_B_PROGRESS_STALE|1\n' > "$root/state"
  entered="$root/entered"
  release="$root/release"
  mkfifo "$entered" "$release"
  start_transition_lock_holder "$root" "$entered" "$release"
  case "$kind" in
    A) unknown=$(FAKE_INSPECT_STATUS=1 invoke "$root") ;;
    B)
      rm -f -- "$root/progress"
      unknown=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
      ;;
  esac
  assert_eq "$name contended unknown does not claim an uncommitted reset" \
    "A_UNKNOWN B_UNKNOWN DECISION_BUSY" "$unknown"
  release_transition_lock_holder "$release"
  if [[ "$kind" == B ]]; then
    printf 'route-a|gen-a|300\n' > "$root/progress"
  fi
  next=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  assert_eq "$name next same-generation suspect is only a first strike" \
    "A_SUSPECT B_SUSPECT DECISION_FIRST_STRIKE" "$next"
  if [[ -e "$root/recommendation" ]]; then
    recommendation="present"
  else
    recommendation="absent"
  fi
  assert_eq "$name contended unknown leaves no promotable recommendation" \
    "absent" "$recommendation"
}

test_contended_a_unknown_reset_invalidates_prior_strike() {
  assert_contended_unknown_reset "A_UNKNOWN" A
}

test_contended_b_unknown_reset_invalidates_prior_strike() {
  assert_contended_unknown_reset "B_UNKNOWN" B
}

test_concurrent_second_strike_emits_one_complete_recommendation() {
  local root first second entered release out first_pid recommendation receipt_shape state
  root=$(make_root)
  printf 'gen-a|A_SOCKET_ZERO_B_PROGRESS_STALE|1\n' > "$root/state"
  entered="$root/entered"
  release="$root/release"
  out="$root/first.out"
  mkfifo "$entered" "$release"
  cat > "$root/barrier" <<'EOF'
#!/usr/bin/env bash
printf 'entered\n' > "$BARRIER_ENTERED"
IFS= read -r ignored < "$BARRIER_RELEASE"
EOF
  chmod 700 "$root/barrier"
  BARRIER_ENTERED="$entered" BARRIER_RELEASE="$release" WATCHDOG_BEFORE_STRIKE_CMD="$root/barrier" \
    FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root" > "$out" &
  first_pid=$!
  IFS= read -r ignored < "$entered"
  second=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root")
  printf 'release\n' > "$release"
  wait "$first_pid"
  first=$(<"$out")
  recommendation=$(<"$root/recommendation")
  receipt_shape="$(wc -l < "$root/recommendation" | tr -d '[:space:]')|$recommendation"
  state=$(<"$root/state")
  assert_eq "one contender publishes the fixed restart recommendation" \
    "A_SUSPECT B_SUSPECT DECISION_RESTART_RECOMMENDED" "$first"
  assert_eq "other state-one contender cannot duplicate publication" \
    "A_SUSPECT B_SUSPECT DECISION_BUSY" "$second"
  assert_eq "concurrent second-strike receipt is one complete fixed line" \
    "1|RESTART_RECOMMENDED" "$receipt_shape"
  assert_eq "concurrent second-strike state is bounded at two" \
    "gen-a|A_SOCKET_ZERO_B_PROGRESS_STALE|2" "$state"
}

make_prohibited_action_guards() {
  local root=$1 guard command
  guard="$root/prohibited-action-guards"
  mkdir "$guard"
  for command in launchctl kill pkill killall hermes bridge bridgectl osascript terminal-notifier notify-send curl wget nc ncat socat cron crontab at systemctl service open; do
    cat > "$guard/$command" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$(basename "$0")" >> "$PROHIBITED_ACTION_LOG"
exit 99
EOF
    chmod 700 "$guard/$command"
  done
  printf '%s' "$guard"
}

test_observer_never_executes_prohibited_action_surfaces() {
  local root guard log
  root=$(make_root)
  guard=$(make_prohibited_action_guards "$root")
  log="$root/prohibited-actions.log"
  WATCHDOG_TEST_PATH="$guard:$PATH" PROHIBITED_ACTION_LOG="$log" \
    FAKE_INSPECT_STATUS=1 invoke "$root" >/dev/null
  WATCHDOG_TEST_PATH="$guard:$PATH" PROHIBITED_ACTION_LOG="$log" \
    FAKE_INSPECT_OUTPUT="ESTABLISHED gateway" invoke "$root" >/dev/null
  WATCHDOG_TEST_PATH="$guard:$PATH" PROHIBITED_ACTION_LOG="$log" \
    FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root" >/dev/null
  WATCHDOG_TEST_PATH="$guard:$PATH" PROHIBITED_ACTION_LOG="$log" \
    FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root" >/dev/null
  if [[ -e "$log" ]]; then
    fail "observer avoids lifecycle notification network scheduler and bridge commands"
  else
    pass "observer avoids lifecycle notification network scheduler and bridge commands"
  fi
}

test_bad_digest_is_not_managed() {
  local root output
  root=$(make_root)
  output=$(FAKE_INSPECT_OUTPUT="no matching sockets" invoke "$root" "0000000000000000000000000000000000000000000000000000000000000000")
  assert_eq "digest mismatch is NOT_MANAGED" "NOT_MANAGED" "$output"
}

test_rendered_plist_is_loadable_and_structural() {
  local root
  root=$(make_root)
  if render_plist "$root"; then
    pass "rendered plist has exact label cadence closed environment and source binding"
  else
    fail "rendered plist has exact label cadence closed environment and source binding"
  fi
}

test_candidate_receipt_is_truthful() {
  if python3 - "$receipt" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
assert data == {
    "artifact": "gateway-freeze-watchdog",
    "authority": "observer_only",
    "operational_liveness": "UNVERIFIED",
}
PY
  then
    pass "candidate receipt is observer-only with unverified operational liveness"
  else
    fail "candidate receipt is observer-only with unverified operational liveness"
  fi
}

if [[ -n "${WATCHDOG_TEST_FILTER:-}" ]]; then
  "$WATCHDOG_TEST_FILTER"
else
  test_inspection_failure_is_unknown
  test_zero_sockets_after_stale_log_is_a_suspect
  test_matching_socket_is_healthy
  test_fresh_log_is_healthy
  test_same_route_post_inbound_stale_progress_starts_strike
  test_invalid_progress_is_unknown
  test_nonwedge_progress_is_healthy
  test_injected_clock_is_used
  test_generation_drift_cancels_strike
  test_class_and_generation_bound_strikes
  test_second_strike_emits_one_recommendation
  test_concurrent_transition_is_serialized
  test_contended_a_unknown_reset_invalidates_prior_strike
  test_contended_b_unknown_reset_invalidates_prior_strike
  test_concurrent_second_strike_emits_one_complete_recommendation
  test_observer_never_executes_prohibited_action_surfaces
  test_bad_digest_is_not_managed
  test_rendered_plist_is_loadable_and_structural
  test_candidate_receipt_is_truthful
fi
printf 'TOTAL=%d PASS=%d FAIL=%d\n' "$((passes + failures))" "$passes" "$failures"
(( failures == 0 ))
