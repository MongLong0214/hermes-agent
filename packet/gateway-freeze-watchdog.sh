#!/usr/bin/env bash
set -u

emit() {
  printf '%s\n' "$1"
}

is_uint() {
  case "$1" in
    ''|*[!0-9]*) return 1 ;;
    *) return 0 ;;
  esac
}

is_token() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]]
}

is_sha256() {
  [[ "$1" =~ ^[0-9a-f]{64}$ ]]
}

read_generation() {
  local file=$1 line pipes extra
  [[ -f "$file" && -r "$file" ]] || return 1
  IFS= read -r line < "$file" || [[ -n "${line:-}" ]] || return 1
  pipes=${line//[^|]/}
  [[ "$pipes" == '||' ]] || return 1
  IFS='|' read -r generation_token process_pid process_start extra <<< "$line"
  [[ -z "$extra" ]] && is_token "$generation_token" && is_uint "$process_pid" && (( process_pid > 0 )) && is_uint "$process_start" || return 1
  generation_snapshot=$line
}

read_record() {
  local file=$1 line pipes extra
  [[ -f "$file" && -r "$file" ]] || return 1
  IFS= read -r line < "$file" || [[ -n "${line:-}" ]] || return 1
  pipes=${line//[^|]/}
  [[ "$pipes" == '||' ]] || return 1
  IFS='|' read -r record_route record_generation record_epoch extra <<< "$line"
  [[ -z "$extra" ]] && is_token "$record_route" && is_token "$record_generation" && is_uint "$record_epoch" && (( record_epoch > 0 )) || return 1
}

same_generation() {
  read_generation "$generation_file" && [[ "$generation_snapshot" == "$captured_generation" ]]
}

release_lock() {
  local owner_line
  if [[ "${owns_lock:-0}" == 1 && -d "$lock_dir" && -f "$lock_dir/owner" ]]; then
    IFS= read -r owner_line < "$lock_dir/owner" || owner_line=""
    if [[ "$owner_line" == "$owner_token" ]]; then
      rm -f -- "$lock_dir/owner"
      rmdir -- "$lock_dir" 2>/dev/null || true
    fi
  fi
  owns_lock=0
}

acquire_lock() {
  if mkdir "$lock_dir" 2>/dev/null; then
    printf '%s\n' "$owner_token" > "$lock_dir/owner"
    owns_lock=1
    return 0
  fi
  return 1
}

request_reset() {
  ( set -C; printf '%s\n' "$captured_generation" > "$reset_file" ) 2>/dev/null || [[ -e "$reset_file" ]]
}

clear_strikes() {
  clear_strikes_result=UNKNOWN
  if ! acquire_lock; then
    if request_reset; then
      clear_strikes_result=BUSY
    fi
    return 1
  fi
  if same_generation; then
    if [[ -e "$state_file" ]]; then
      : > "$state_file"
    fi
    if [[ -e "$reset_file" ]] && ! rm -f -- "$reset_file"; then
      release_lock
      return 1
    fi
    release_lock
    return 0
  fi
  : > "$state_file"
  release_lock
  clear_strikes_result=GENERATION_DRIFT
  return 1
}

clear_or_fail_closed() {
  if clear_strikes; then
    return 0
  fi
  emit "A_UNKNOWN B_UNKNOWN DECISION_${clear_strikes_result}"
  return 1
}

root=${WATCHDOG_ROOT:-}
inspect_cmd=${WATCHDOG_INSPECT_CMD:-}
script_path=${WATCHDOG_SCRIPT_PATH:-}
expected_sha=${WATCHDOG_EXPECTED_SHA256:-}
hash_cmd=${WATCHDOG_HASH_CMD:-shasum}
now=${WATCHDOG_NOW:-}

if [[ -z "$root" || -z "$inspect_cmd" || -z "$script_path" ]] || ! is_sha256 "$expected_sha"; then
  emit "NOT_MANAGED"
  exit 0
fi

actual_line=$("$hash_cmd" -a 256 "$script_path" 2>/dev/null || true)
actual_sha=${actual_line%% *}
if ! is_sha256 "$actual_sha" || [[ "$actual_sha" != "$expected_sha" ]]; then
  emit "NOT_MANAGED"
  exit 0
fi

if ! is_uint "$now"; then
  time_cmd=${WATCHDOG_TIME_CMD:-}
  if [[ -n "$time_cmd" ]]; then
    now=$("$time_cmd" 2>/dev/null || true)
  fi
fi
if ! is_uint "$now"; then
  emit "A_UNKNOWN B_UNKNOWN DECISION_UNKNOWN"
  exit 0
fi

generation_file="$root/generation"
state_file="$root/state"
recommendation_file="$root/recommendation"
lock_dir="$root/transition.lock"
owns_lock=0

if ! read_generation "$generation_file"; then
  emit "A_UNKNOWN B_UNKNOWN DECISION_UNKNOWN"
  exit 0
fi
captured_generation=$generation_snapshot
owner_token="${generation_token}.${process_pid}.$$"
reset_file="$root/reset.${generation_token}"
trap release_lock EXIT

if inspection=$("$inspect_cmd" "$process_pid" 2>/dev/null); then
  inspection_status=0
else
  inspection_status=$?
fi

if (( inspection_status != 0 )); then
  clear_or_fail_closed || exit 0
  emit "A_UNKNOWN B_UNKNOWN DECISION_UNKNOWN"
  exit 0
fi

if [[ "$inspection" == *ESTABLISHED*gateway* ]]; then
  clear_or_fail_closed || exit 0
  emit "A_HEALTHY B_SKIPPED DECISION_HEALTHY"
  exit 0
fi

if ! IFS= read -r log_epoch < "$root/log_epoch" || ! is_uint "$log_epoch" || (( now < log_epoch )); then
  clear_or_fail_closed || exit 0
  emit "A_UNKNOWN B_UNKNOWN DECISION_UNKNOWN"
  exit 0
fi

if (( now - log_epoch < 600 )); then
  clear_or_fail_closed || exit 0
  emit "A_HEALTHY B_SKIPPED DECISION_HEALTHY"
  exit 0
fi

if ! read_record "$root/inbound"; then
  clear_or_fail_closed || exit 0
  emit "A_SUSPECT B_UNKNOWN DECISION_UNKNOWN"
  exit 0
fi
inbound_route=$record_route
inbound_generation=$record_generation
inbound_epoch=$record_epoch

if [[ "$inbound_generation" != "$generation_token" ]] || (( inbound_epoch < process_start )); then
  clear_or_fail_closed || exit 0
  emit "A_SUSPECT B_UNKNOWN DECISION_UNKNOWN"
  exit 0
fi

if ! read_record "$root/progress"; then
  clear_or_fail_closed || exit 0
  emit "A_SUSPECT B_UNKNOWN DECISION_UNKNOWN"
  exit 0
fi

if [[ "$record_route" != "$inbound_route" || "$record_generation" != "$generation_token" ]] || (( record_epoch < process_start || record_epoch < inbound_epoch || now < record_epoch )); then
  clear_or_fail_closed || exit 0
  emit "A_SUSPECT B_UNKNOWN DECISION_UNKNOWN"
  exit 0
fi

progress_epoch=$record_epoch
if (( now - progress_epoch < 600 )); then
  clear_or_fail_closed || exit 0
  emit "A_SUSPECT B_HEALTHY DECISION_HEALTHY"
  exit 0
fi

if [[ -e "$root/response" ]]; then
  if ! read_record "$root/response" || [[ "$record_route" != "$inbound_route" || "$record_generation" != "$generation_token" ]]; then
    clear_or_fail_closed || exit 0
    emit "A_SUSPECT B_UNKNOWN DECISION_UNKNOWN"
    exit 0
  fi
  if (( record_epoch > inbound_epoch )); then
    clear_or_fail_closed || exit 0
    emit "A_SUSPECT B_HEALTHY DECISION_HEALTHY"
    exit 0
  fi
fi

if ! acquire_lock; then
  emit "A_SUSPECT B_SUSPECT DECISION_BUSY"
  exit 0
fi

before_strike_cmd=${WATCHDOG_BEFORE_STRIKE_CMD:-}
if [[ -n "$before_strike_cmd" ]]; then
  "$before_strike_cmd" "$root" >/dev/null 2>&1 || true
fi

if ! same_generation; then
  : > "$state_file"
  release_lock
  emit "A_UNKNOWN B_UNKNOWN DECISION_GENERATION_DRIFT"
  exit 0
fi

if [[ -e "$reset_file" ]]; then
  : > "$state_file"
  if ! rm -f -- "$reset_file"; then
    release_lock
    emit "A_UNKNOWN B_UNKNOWN DECISION_UNKNOWN"
    exit 0
  fi
fi

prior_generation=""
prior_class=""
prior_count=0
if [[ -f "$state_file" ]]; then
  IFS= read -r prior_line < "$state_file" || prior_line=""
  IFS='|' read -r prior_generation prior_class prior_count prior_extra <<< "$prior_line"
  if [[ -n "${prior_extra:-}" ]] || ! is_uint "${prior_count:-}"; then
    prior_generation=""
    prior_class=""
    prior_count=0
  fi
fi

observation_class="A_SOCKET_ZERO_B_PROGRESS_STALE"
if [[ "$prior_generation" == "$generation_token" && "$prior_class" == "$observation_class" && "$prior_count" == 1 ]]; then
  next_count=2
elif [[ "$prior_generation" == "$generation_token" && "$prior_class" == "$observation_class" && "$prior_count" == 2 ]]; then
  next_count=2
else
  next_count=1
fi

if ! same_generation; then
  : > "$state_file"
  release_lock
  emit "A_UNKNOWN B_UNKNOWN DECISION_GENERATION_DRIFT"
  exit 0
fi

if (( next_count == 1 )); then
  printf '%s|%s|1\n' "$generation_token" "$observation_class" > "$state_file"
  release_lock
  emit "A_SUSPECT B_SUSPECT DECISION_FIRST_STRIKE"
  exit 0
fi

if (( prior_count == 2 )); then
  release_lock
  emit "A_SUSPECT B_SUSPECT DECISION_ALREADY_RECORDED"
  exit 0
fi

printf '%s|%s|2\n' "$generation_token" "$observation_class" > "$state_file"
if ! same_generation; then
  : > "$state_file"
  release_lock
  emit "A_UNKNOWN B_UNKNOWN DECISION_GENERATION_DRIFT"
  exit 0
fi

if [[ ! -e "$recommendation_file" ]]; then
  ( set -C; printf '%s\n' "RESTART_RECOMMENDED" > "$recommendation_file" ) 2>/dev/null || true
fi

if [[ -f "$recommendation_file" ]]; then
  IFS= read -r recommendation < "$recommendation_file" || recommendation=""
else
  recommendation=""
fi
release_lock
if [[ "$recommendation" == "RESTART_RECOMMENDED" ]]; then
  emit "A_SUSPECT B_SUSPECT DECISION_RESTART_RECOMMENDED"
else
  emit "A_SUSPECT B_SUSPECT DECISION_UNKNOWN"
fi
