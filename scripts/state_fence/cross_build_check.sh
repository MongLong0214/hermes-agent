#!/usr/bin/env bash
# Cross-build receipt for the turn-fence cutover: what the fork (generation 29), this candidate and
# plain upstream do to one another's state.db. CI has no fork or plain-upstream build, so this is the
# only proof that the fork is refused after the cutover and that plain upstream neither writes a
# fenced store nor restamps it.
#
#   scripts/state_fence/cross_build_check.sh [--python PY] [--workdir DIR]
#                                            [--fork-rev REV] [--upstream-rev REV]
#
# Steps 1-4 run on one scratch store under a throwaway HOME/HERMES_HOME. Every child runs under
# `env -i` so no caller HERMES_HOME can point it at a real profile, with PYTHONPATH set to the one
# tree it tests (it asserts `hermes_state` imports from there) and no bytecode written.
#   1. The fork creates and writes the store.
#   2. This candidate (HEAD plus the working tree) migrates it and writes.
#   3. The fork reopens it and is refused; a fork raw write aborts on the fence.
#   4. Plain upstream opens it; its governed writes fail and the stored value is kept.
# Step 5 (first-open and optimize-storage timings on a copy of a live store) is not implemented;
# `--live-copy PATH` is refused, and nothing ever defaults to a live store.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY="$REPO/.venv/bin/python"
WORKDIR="${TMPDIR:-/tmp}"
FORK_REV=6c9d8d8074
UPSTREAM_REV=bcbfce90f8
LIVE_COPY=""

usage() { sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while (($#)); do
  case "$1" in
    --python) PY=$2; shift 2 ;;
    --workdir) WORKDIR=$2; shift 2 ;;
    --fork-rev) FORK_REV=$2; shift 2 ;;
    --upstream-rev) UPSTREAM_REV=$2; shift 2 ;;
    --live-copy) LIVE_COPY=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Step 5 stub. It needs an explicit path and has no default: a live state.db must never be opened by
# a candidate build. Whoever implements it copies the store with the backup API from a `mode=ro`
# URI, times only the copy, and never opens the given path for write. Until then it opens nothing.
if [[ -n "$LIVE_COPY" ]]; then
  echo "step 5 (timings on a live-store copy) is not implemented; refusing to open $LIVE_COPY" >&2
  exit 3
fi

fork_sha=$(git -C "$REPO" rev-parse "$FORK_REV^{commit}")
upstream_sha=$(git -C "$REPO" rev-parse "$UPSTREAM_REV^{commit}")
mkdir -p "$WORKDIR"
RUN=$(mktemp -d "$WORKDIR/cross-build.XXXXXX")
mkdir -p "$RUN/fork" "$RUN/upstream" "$RUN/home/.hermes"
git -C "$REPO" archive "$fork_sha" | tar -x -C "$RUN/fork"
git -C "$REPO" archive "$upstream_sha" | tar -x -C "$RUN/upstream"
DB="$RUN/home/.hermes/state.db"

cat > "$RUN/step.py" <<'PY'
import os
import re
import sqlite3
import sys
from pathlib import Path

step, tree, db = sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3])
import hermes_state

assert Path(hermes_state.__file__).resolve().is_relative_to(tree), hermes_state.__file__
assert Path(os.environ["HERMES_HOME"]) == db.parent, os.environ["HERMES_HOME"]
from hermes_state import SessionDB

_LITERAL = re.compile(r"hermes_turn_fence_generation\(\)\s*!=\s*(\d+)")


def facts():
    conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        fences = [sql for (sql,) in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'turn_fence_%'")]
        return {
            "stored": [v for (v,) in conn.execute("SELECT version FROM schema_version")],
            "fence_triggers": len(fences),
            "fence_literals": sorted({int(m) for sql in fences for m in _LITERAL.findall(sql)}),
            "schema_cookie": conn.execute("PRAGMA schema_version").fetchone()[0],
            "sessions": conn.execute("SELECT count(*) FROM sessions").fetchone()[0],
            "messages": conn.execute("SELECT count(*) FROM messages").fetchone()[0],
        }
    finally:
        conn.close()


def describe(exc):
    if exc is None:
        return "nothing raised"
    fields = {k: getattr(exc, k) for k in ("cause", "expected_generation", "actual_generation") if hasattr(exc, k)}
    return f"{type(exc).__module__}.{type(exc).__name__}: {exc}" + (f" {fields}" if fields else "")


def raised(fn):
    try:
        fn()
    except Exception as exc:
        return exc
    return None


def turn(session_db, session_id, text):
    session_db.create_session(session_id, "cli")
    session_db.append_message(session_id, "user", text)
    session_db.append_message(session_id, "assistant", f"ack: {text}")


def fork_writes():
    from hermes_state_common import TURN_FENCE_GENERATION
    print(f"  fork build generation: {TURN_FENCE_GENERATION}")
    session_db = SessionDB(db_path=db)
    try:
        turn(session_db, "xb-fork", "written by the fork")
    finally:
        session_db.close()
    after = facts()
    print(f"  after: {after}")
    return after["stored"] == [TURN_FENCE_GENERATION] and after["fence_literals"] == [TURN_FENCE_GENERATION]


def candidate_migrates():
    from hermes_state_fence import STORED_SCHEMA_VERSION
    before = facts()
    print(f"  candidate stored generation: {STORED_SCHEMA_VERSION}")
    print(f"  before: {before}")
    session_db = SessionDB(db_path=db)
    try:
        turn(session_db, "xb-cand", "written by the candidate")
    finally:
        session_db.close()
    after = facts()
    print(f"  after: {after}")
    return (after["stored"] == [STORED_SCHEMA_VERSION] and after["fence_literals"] == [STORED_SCHEMA_VERSION]
            and (after["sessions"], after["messages"]) == (before["sessions"] + 1, before["messages"] + 2))


def fork_refused():
    from hermes_state_common import SCHEMA_VERSION, register_turn_fence_generation
    before = facts()
    print(f"  before: {before}")
    opened = raised(lambda: SessionDB(db_path=db).close())
    print(f"  fork SessionDB open: {describe(opened)}")
    conn = sqlite3.connect(str(db), isolation_level=None)
    register_turn_fence_generation(conn)
    try:
        raw = raised(lambda: conn.execute("UPDATE sessions SET title = 'fork raw write' WHERE id = 'xb-cand'"))
    finally:
        conn.close()
    print(f"  fork raw write (fence function registered at the fork's generation): {describe(raw)}")
    after = facts()
    print(f"  after: {after}")
    # The fork spells its cause values its own way; compare against its class constant.
    return (type(opened).__name__ == "IncompatibleSchemaError"
            and getattr(opened, "cause", None) == getattr(type(opened), "BUILD_TOO_OLD", object())
            and (opened.expected_generation, opened.actual_generation) == (SCHEMA_VERSION, before["stored"][0])
            and isinstance(raw, sqlite3.IntegrityError) and after == before)


def upstream_is_fenced_out():
    before = facts()
    print(f"  before: {before}")
    session_db = None

    def open_store():
        nonlocal session_db
        session_db = SessionDB(db_path=db)

    opened = raised(open_store)
    print(f"  plain upstream SessionDB open: {'opened' if opened is None else describe(opened)}")
    writes = []
    if session_db is not None:
        try:
            writes.append(raised(lambda: session_db.create_session("xb-upstream", "cli")))
            print(f"  plain upstream create_session: {describe(writes[-1])}")
            writes.append(raised(lambda: session_db.append_message("xb-cand", "user", "written by plain upstream")))
            print(f"  plain upstream append_message: {describe(writes[-1])}")
        finally:
            session_db.close()
    after = facts()
    print(f"  after: {after}")
    refused = [isinstance(e, sqlite3.OperationalError) and "no such function: hermes_turn_fence_generation" in str(e)
               for e in writes]
    return (opened is None and len(refused) == 2 and all(refused) and after["stored"] == before["stored"]
            and (after["sessions"], after["messages"]) == (before["sessions"], before["messages"]))


STEPS = {"1": fork_writes, "2": candidate_migrates, "3": fork_refused, "4": upstream_is_fenced_out}
ok = STEPS[step]()
print(f"  => {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY

run_step() {
  local step=$1 label=$2 tree=$3
  echo "step $step: $label"
  env -i PATH=/usr/bin:/bin HOME="$RUN/home" HERMES_HOME="$RUN/home/.hermes" PYTHONPATH="$tree" \
    PYTHONDONTWRITEBYTECODE=1 TZ=UTC LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    "$PY" "$RUN/step.py" "$step" "$tree" "$DB" 2>&1
}

echo "interpreter: $PY ($("$PY" -c 'import sqlite3, sys; print(sys.version.split()[0], "sqlite", sqlite3.sqlite_version)'))"
echo "fork: $fork_sha  upstream: $upstream_sha"
echo "candidate: $(git -C "$REPO" rev-parse HEAD) + $(git -C "$REPO" status --porcelain | wc -l | tr -d ' ') changed paths"
echo "scratch: $RUN"
status=0
run_step 1 "the fork creates and writes the store" "$RUN/fork" || status=1
((status == 0)) && { run_step 2 "the candidate migrates it and writes" "$REPO" || status=1; }
((status == 0)) && { run_step 3 "the fork reopens it and is refused" "$RUN/fork" || status=1; }
((status == 0)) && { run_step 4 "plain upstream opens it and cannot write" "$RUN/upstream" || status=1; }
echo "step 5: not run (timings on a live-store copy are not implemented; see --live-copy)"
exit "$status"
