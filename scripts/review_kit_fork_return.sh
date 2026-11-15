#!/usr/bin/env bash
# Reviewer kit for hermes_state_fork_return: verify the return path with the fork's own binary.
#
#   scripts/review_kit_fork_return.sh FORK_TREE FORK_PYTHON CANDIDATE_PYTHON
#
#   FORK_TREE         a read-only copy of the fork tree at 420c58a1 (its root holds hermes_state.py)
#   FORK_PYTHON       the interpreter that tree runs with, normally FORK_TREE/venv/bin/python
#   CANDIDATE_PYTHON  an interpreter with this checkout's test dependencies (pytest included)
#
# 1. fork binary: creates a fresh store; the fork's legacy session-process authority objects are added as
#    its older generations wrote them (tests/hermes_state/test_fork_return.py FORK_AUTHORITY_DDL), then the
#    fork binary writes fork-era sessions, messages, a conversation boundary and an expired turn lease.
# 2. this checkout: migrates it to 1033 and writes what this build writes (candidate_writes in that test
#    module: a turn, a revoked and a reopened authority, a reclaimed lease, a run admission, a worker).
#    Then label probes: one new session per label of this checkout's alias table (lookup_labels), on a copy, each
#    looked up under every label by this checkout's CLI reader; one session is added to SOURCE for every label the
#    module returns (label_lookups equal on both sides).
# 3. this checkout: python -m hermes_state_fork_return SOURCE OUTPUT --report REPORT, then --verify OUTPUT (the
#    consumer's check: its commit marker names its bytes; a copy without the marker must fail it). Then lookups:
#    every session label the store holds (source, created_source) and every alias-table label, looked up by this
#    checkout's CLI reader on a copy of SOURCE and by the fork binary on a copy of OUTPUT (listings with and without
#    archived rows, search, count, each session's own row): the same sessions must come back for every label. Then
#    the fork binary's own probe sessions, one per label, on a copy of OUTPUT: for every label, the labels each
#    side's real lookups find a probe under must be the ones label_lookups names, so the module returns exactly the
#    labels both binaries find alike. A copy of SOURCE with a desktop/desktop session must refuse.
# 4. fork binary, on a copy of OUTPUT in a fresh HOME: open, read, write a turn, reopen a revoked session,
#    try the superseded lease holder, cross a conversation boundary; then a new process (restart) reads it
#    all back, and the fork's `hermes sessions list` reads it.
#
# Every process runs under env -i with HOME/HERMES_HOME in a new temp directory, no bytecode writes and
# proxies pointed at a closed port. Each step prints the interpreter (as invoked and resolved), sys.prefix
# and the file every hermes module was loaded from; a module loaded from outside FORK_TREE (or this
# checkout, for the candidate steps) fails the step. Nothing outside the temp directory is written;
# KEEP=1 keeps it. Exit status 0 only when every check passed.
set -euo pipefail

if [ $# -ne 3 ]; then
  sed -n '2,8p' "$0" >&2
  exit 2
fi
FORK_TREE=$(cd "$1" && pwd -P)
FORK_PYTHON=$2
CANDIDATE_PYTHON=$3
REPO=$(cd "$(dirname "$0")/.." && pwd -P)
[ -f "$FORK_TREE/hermes_state.py" ] || { echo "not a fork tree: $FORK_TREE" >&2; exit 2; }
[ -f "$REPO/hermes_state_fork_return.py" ] || { echo "not this checkout: $REPO" >&2; exit 2; }

W=$(mktemp -d "${TMPDIR:-/tmp}/review-kit-fork-return.XXXXXX")
if [ "${KEEP:-0}" = 1 ]; then echo "work directory kept: $W"; else trap 'rm -rf "$W"' EXIT; fi
mkdir -p "$W/steps" "$W/tmp" "$W/fork-home/.hermes" "$W/cand-home/.hermes" "$W/return-home/.hermes" \
  "$W/candidate" "$W/out"
echo "fork tree:        $FORK_TREE ($(git -C "$FORK_TREE" rev-parse HEAD 2>/dev/null || echo 'no git metadata'))"
echo "fork python:      $FORK_PYTHON -> $(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$FORK_PYTHON")"
echo "candidate:        $REPO ($(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo 'no git metadata'))"
echo "candidate python: $CANDIDATE_PYTHON"
echo "work directory:   $W"

run() {  # run HOME PYTHON ROOT STEP ARGS...: one isolated process with ROOT first on sys.path
  local home=$1 python=$2 root=$3; shift 3
  env -i HOME="$home" HERMES_HOME="$home/.hermes" TMPDIR="$W/tmp" PATH=/usr/bin:/bin TZ=UTC LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 \
    "$python" -I -B "$W/steps/boot.py" "$root" "$@" </dev/null
}

cat >"$W/steps/boot.py" <<'PY'
"""boot.py ROOT STEP ARGS...: ROOT first on sys.path, run STEP, then check where every hermes module came from."""
import json, os, runpy, sys
root, step = os.path.realpath(sys.argv[1]), sys.argv[2]
sys.path.insert(0, root)
sys.argv = [step, *sys.argv[3:]]
code, foreign = 1, []
try:
    runpy.run_path(step, run_name="__main__")
    code = 0
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
finally:
    ours = {m: getattr(sys.modules[m], "__file__", None) or "" for m in list(sys.modules)
            if m.split(".")[0].startswith(("hermes", "gateway", "agent", "tools", "cron", "acp_adapter", "tests"))}
    foreign = sorted(f for f in ours.values() if f and not os.path.realpath(f).startswith(root + os.sep))
    print("  interpreter:", json.dumps({"invoked": sys.executable, "resolved": os.path.realpath(sys.executable),
          "prefix": sys.prefix, "base_prefix": sys.base_prefix, "version": sys.version.split()[0],
          "hermes_state": ours.get("hermes_state"), "modules_outside_root": foreign}), flush=True)
sys.exit(code or (3 if foreign else 0))
PY

cat >"$W/steps/fork_create.py" <<'PY'
import sys
from pathlib import Path
from hermes_state import SessionDB
SessionDB(db_path=Path(sys.argv[1])).close()
PY

cat >"$W/steps/legacy_authorities.py" <<'PY'
"""The fork's legacy session-process authority objects on a fork store, as its older generations wrote them."""
import sqlite3, sys
from tests.hermes_state.test_fork_return import FORK_AUTHORITY_DDL, STATE_DB_ID
from tests.hermes_state.test_state_lineage_admission import live_fence_trigger_sql
conn = sqlite3.connect(sys.argv[1], isolation_level=None)
conn.execute("BEGIN")
for sql in FORK_AUTHORITY_DDL:
    conn.execute(sql)
for operation in ("INSERT", "UPDATE", "DELETE"):
    conn.execute(live_fence_trigger_sql("session_process_authorities", operation, 1030))
conn.execute("INSERT INTO state_meta (key, value) VALUES ('session_process_state_db_id', ?), "
             "('session_process_state_family', 'sessiondb-v1')", (STATE_DB_ID,))
conn.execute("COMMIT")
conn.close()
PY

cat >"$W/steps/fork_era.py" <<'PY'
import sys
from hermes_state import SessionDB
from hermes_state_fence import probe_store_lineage
KEY = "agent:p1:telegram:dm:1"
db = SessionDB(db_path=__import__("pathlib").Path(sys.argv[1]))
try:
    db.create_session("s1", source="cli")
    db.append_message("s1", role="user", content="hello fork store")
    db.append_message("s1", role="assistant", content="the needle is here")
    db.create_session("tg0", source="telegram", session_key=KEY)
    db.append_message("tg0", role="user", content="the first telegram conversation")
    db.end_session("tg0", "session_reset")  # a conversation boundary, written by the fork
    db.create_session("tg1", source="telegram", session_key=KEY)
    db.append_message("tg1", role="user", content="telegram before the upgrade")
    db.create_session("s3", source="cli")
    assert db.try_acquire_session_turn_lease("tg1", "pid=999999:turn=fork-era", ttl_seconds=0.1)
    print("  fork-era boundary:", db.latest_conversation_boundary(KEY, "telegram"))
finally:
    db.close()
lineage = probe_store_lineage(sys.argv[1])
print("  fork-era store:", lineage.lineage, lineage.stored, sorted(lineage.fence_literals))
assert (lineage.lineage, lineage.stored) == ("fenced", 1030), lineage
PY

cat >"$W/steps/candidate.py" <<'PY'
import hashlib, sqlite3, sys
from tests.hermes_state.test_fork_return import candidate_writes
candidate_writes(__import__("pathlib").Path(sys.argv[1]))  # migrates to 1033 with this build, then writes
conn = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
print("  candidate store:", conn.execute("SELECT version FROM schema_version").fetchone()[0],
      dict(conn.execute("SELECT status || '/' || COALESCE(outcome, ''), count(*) FROM session_admissions GROUP BY 1")))
conn.close()
PY

cat >"$W/steps/compare_shape.py" <<'PY'
"""The returned store against the fork-era store the fork binary wrote: same objects, columns, fences, stamp."""
import sys
from pathlib import Path
from tests.hermes_state.test_fork_return import _shape
fork, returned = _shape(Path(sys.argv[1])), _shape(Path(sys.argv[2]))
for key in ("objects", "columns", "fences", "stamp"):
    same = fork[key] == returned[key]
    print(f"  {key} equal to the fork's own store: {same}")
    if not same:
        a, b = fork[key], returned[key]
        print("    differs:", sorted(set(a) ^ set(b)) if isinstance(a, dict) else (a, b),
              [k for k in a if isinstance(a, dict) and k in b and a[k] != b[k]])
        sys.exit(1)
PY

cat >"$W/steps/fork_check.py" <<'PY'
"""fork_check.py DB PHASE: the fork binary opens, reads, writes (PHASE=write) or reads back (PHASE=restart)."""
import json, sqlite3, sys
from hermes_state import SessionDB
from hermes_state_fence import probe_store_lineage
KEY, DB, PHASE = "agent:p1:telegram:dm:1", sys.argv[1], sys.argv[2]
failed = []


def check(name, ok, value=None):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + ("" if value is None else f": {json.dumps(value, default=str)}"))
    ok or failed.append(name)


def raw(sql, *args):
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return [tuple(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


AUTH = "SELECT session_generation, status, authority_token FROM session_process_authorities WHERE session_id = ? ORDER BY 1"
lineage = probe_store_lineage(DB)
check("lineage is the fork's fenced 1030 store", (lineage.lineage, lineage.stored) == ("fenced", 1030),
      [lineage.lineage, lineage.stored, sorted(lineage.fence_literals)])
db = SessionDB(db_path=__import__("pathlib").Path(DB))
try:
    check("opened writable by the fork binary", True)
    tg1 = [m["content"] for m in db.get_messages("tg1")]
    check("this build's turn reads back", "telegram after the upgrade qv82" in tg1 and "done" in tg1, tg1)
    check("full-text search finds it", len(db.search_messages("qv82")) >= 1)
    check("sessions read back", all(db.get_session(s) for s in ("s1", "s2", "s3", "tg0", "tg1")))
    if PHASE == "write":
        s1_before = raw(AUTH, "s1")
        boundary = db.latest_conversation_boundary(KEY, "telegram")
        check("conversation boundary is this build's, not reset", boundary == 2, boundary)
        db.append_message("tg1", role="user", content="after the return ok7")
        db.append_message("tg1", role="assistant", content="the fork answered ok7")
        db.create_session("after", source="cli")
        check("the fork's trigger issues the new session's authority",
              [r[:2] for r in raw(AUTH, "after")] == [(1, "ISSUED")], raw(AUTH, "after")[:1])
        db.reopen_session("s1")
        s1 = raw(AUTH, "s1")
        check("a revoked holder's authority is not revived", s1_before[0][1] == "REVOKED" and s1[0] == s1_before[0]
              and [r[:2] for r in s1] == [(1, "REVOKED"), (2, "ISSUED")] and s1[1][2] != s1[0][2],
              [r[:2] for r in s1])
        check("the superseded fork-era lease holder cannot take the conversation",
              db.try_acquire_session_turn_lease("tg1", "pid=999999:turn=fork-era", ttl_seconds=60) is False,
              raw("SELECT holder FROM session_turn_leases"))
        db.end_session("tg1", "session_reset")
        boundary = db.latest_conversation_boundary(KEY, "telegram")
        check("the next boundary continues the counter (no conversation fork)", boundary == 3, boundary)
    else:
        check("the fork's own write reads back after restart",
              "after the return ok7" in tg1 and "the fork answered ok7" in tg1)
        check("its new session reads back", db.get_session("after") is not None)
        check("authorities read back", [r[:2] for r in raw(AUTH, "s1")] == [(1, "REVOKED"), (2, "ISSUED")])
        check("boundary reads back", db.latest_conversation_boundary(KEY, "telegram") == 3)
finally:
    db.close()
check("integrity_check", raw("PRAGMA integrity_check") == [("ok",)])
sys.exit(1 if failed else 0)
PY

cat >"$W/steps/lookups.py" <<'PY'
"""lookups.py MODE DB OUT [LABELS]: the sessions each label finds. MODE=candidate reads through this checkout's CLI
reader (a ``local`` row is listed under the surface that created it) and writes the labels it used; MODE=fork reads
with the fork binary's SessionDB, as the fork's CLI does, for the labels the candidate step wrote."""
import json, sqlite3, sys
from pathlib import Path
from hermes_state import SessionDB
MODE, DB, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
if MODE == "candidate":
    from hermes_state_fork_return import lookup_labels
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    held = {v for (v,) in conn.execute("SELECT source FROM sessions UNION SELECT created_source FROM sessions")}
    conn.close()
    labels = sorted((held | set(lookup_labels())) - {None})
else:
    labels = json.loads(Path(sys.argv[4]).read_text())["labels"]
db = SessionDB(db_path=DB)
try:
    reader = db
    if MODE == "candidate":
        from hermes_cli.gateway_session_store import SurfaceLabelReader
        reader = SurfaceLabelReader(db)

    def ids(rows):
        return sorted(r["id"] for r in rows)

    wide = dict(limit=100000, include_children=True, include_hidden=True)
    found = {label: {
        "list": ids(reader.list_sessions_rich(source=label, **wide)),
        "list_with_archived": ids(reader.list_sessions_rich(source=label, include_archived=True, **wide)),
        "search": ids(reader.search_sessions(source=label, limit=100000)),
        "count": reader.session_count(source=label),
    } for label in labels}
    every = sorted({i for f in found.values() for i in f["list_with_archived"]})
    rows = {}
    for i in every:
        row = db.get_session(i)
        rows[i] = row and {k: row.get(k) for k in ("id", "source", "session_key", "parent_session_id")}
finally:
    db.close()
OUT.write_text(json.dumps({"labels": labels, "found": found, "rows": rows}, sort_keys=True))
print(f"  {MODE}: {len(labels)} labels, {len(every)} sessions found")
PY

cat >"$W/steps/fork_cli.py" <<'PY'
import sys
sys.argv = ["hermes", "sessions", "list"]
from hermes_cli.main import main
main()
PY

cat >"$W/steps/label_probe.py" <<'PY'
"""label_probe.py MODE DB OUT [CANDIDATE_OUT]: a new session per label (MODE=candidate: this checkout's SessionDB and
its alias-table labels; MODE=fork: the fork binary, the candidate's labels), then every label's listing and search
through that side's own reader: the labels that find each probe session."""
import json, sys
from pathlib import Path
from hermes_state import SessionDB
MODE, DB, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
if MODE == "candidate":
    from hermes_state_fork_return import lookup_labels
    labels = sorted(set(lookup_labels()) | {"telegram"})
else:
    labels = json.loads(Path(sys.argv[4]).read_text())["labels"]
db = SessionDB(db_path=DB)
try:
    for label in labels:
        db.create_session(f"probe-{label}", source=label)
        db.append_message(f"probe-{label}", role="user", content=f"probe under {label}")
    reader = db
    if MODE == "candidate":
        from hermes_cli.gateway_session_store import SurfaceLabelReader
        reader = SurfaceLabelReader(db)
    probes = {f"probe-{label}": {"listed": [], "searched": []} for label in labels}
    for label in labels:
        for row in reader.list_sessions_rich(source=label, include_archived=True, limit=100000,
                                             include_children=True, include_hidden=True):
            if row["id"] in probes:
                probes[row["id"]]["listed"].append(label)
        for row in reader.search_sessions(source=label, limit=100000):
            if row["id"] in probes:
                probes[row["id"]]["searched"].append(label)
    for probe, found in probes.items():
        row = db.get_session(probe)
        found.update(source=row.get("source"), created_source=row.get("created_source"))
finally:
    db.close()
OUT.write_text(json.dumps({"labels": labels, "probes": probes}, sort_keys=True))
print(f"  {MODE}: {len(labels)} probe sessions")
PY

cat >"$W/steps/label_sessions.py" <<'PY'
"""label_sessions.py DB PROBES: on DB, one session for every probed label the module returns."""
import json, sys
from pathlib import Path
from hermes_state import SessionDB
from hermes_state_fork_return import label_lookups
probes = json.loads(Path(sys.argv[2]).read_text())["probes"]
returned = sorted(p["source"] for p in probes.values() if p["created_source"] in (None, p["source"])
                  and label_lookups(p["source"], p["created_source"])[0] == [p["source"]]
                  and label_lookups(p["source"], p["created_source"])[1] == [p["source"]])
db = SessionDB(db_path=Path(sys.argv[1]))
try:
    for label in returned:
        db.create_session(f"label-{label}", source=label)
        db.append_message(f"label-{label}", role="user", content=f"kept under {label}")
finally:
    db.close()
print("  a session for every label the module returns:", " ".join(returned))
PY

cat >"$W/steps/compare_probes.py" <<'PY'
"""compare_probes.py CANDIDATE FORK: each side's measured lookups against label_lookups, for every probe label."""
import json, sys
from hermes_state_fork_return import label_lookups
cand, fork = (json.load(open(p)) for p in sys.argv[1:3])
bad = []
for probe, c in sorted(cand["probes"].items()):
    f = fork["probes"][probe]
    this_build, the_fork = label_lookups(c["source"], c["created_source"])
    for side, found, model in (("this build", c, this_build), ("the fork", f, the_fork)):
        if sorted(found["listed"]) != model or sorted(found["searched"]) != model:
            bad.append((side, probe, found["listed"], found["searched"], model))
    returns = this_build == the_fork and c["created_source"] in (None, c["source"])
    print(f"  {c['source']}/{c['created_source']}: this build finds it under {sorted(c['listed'])}, the fork under "
          f"{sorted(f['listed'])} -> {'returned' if returns else 'refused'}")
for b in bad:
    print("  FAIL measured lookups differ from label_lookups:", json.dumps(b))
sys.exit(1 if bad else 0)
PY

cat >"$W/steps/desktop_session.py" <<'PY'
import sys
from pathlib import Path
from hermes_state import SessionDB
db = SessionDB(db_path=Path(sys.argv[1]))
try:
    db.create_session("d1", source="desktop")
    db.append_message("d1", role="user", content="made in the desktop app")
finally:
    db.close()
PY

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
step() { echo; echo "== $*"; }

step "1. fork binary: a fresh store, its legacy authorities, fork-era writes"
FORK_DB=$W/fork-home/.hermes/state.db
run "$W/fork-home" "$FORK_PYTHON" "$FORK_TREE" "$W/steps/fork_create.py" "$FORK_DB"
run "$W/cand-home" "$CANDIDATE_PYTHON" "$REPO" "$W/steps/legacy_authorities.py" "$FORK_DB"
run "$W/fork-home" "$FORK_PYTHON" "$FORK_TREE" "$W/steps/fork_era.py" "$FORK_DB"
cp "$FORK_DB" "$W/fork-era.db"

step "2. this build: migrate to 1033 and write"
SRC=$W/candidate/state.db
cp "$W/fork-era.db" "$SRC"
run "$W/cand-home" "$CANDIDATE_PYTHON" "$REPO" "$W/steps/candidate.py" "$SRC"
cp "$SRC" "$W/probe-source.db"
run "$W/cand-home" "$CANDIDATE_PYTHON" "$REPO" "$W/steps/label_probe.py" candidate "$W/probe-source.db" \
  "$W/probes-candidate.json"
run "$W/cand-home" "$CANDIDATE_PYTHON" "$REPO" "$W/steps/label_sessions.py" "$SRC" "$W/probes-candidate.json"
SRC_SHA=$(sha "$SRC")

step "3. this build: the return"
cat >"$W/steps/return.py" <<'PY'
import sys
from hermes_state_fork_return import main
sys.exit(main(sys.argv[1:]))
PY
run "$W/cand-home" "$CANDIDATE_PYTHON" "$REPO" "$W/steps/return.py" "$SRC" "$W/out/state.db" \
  --report "$W/report.json"
python3 - "$W/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("  report:", json.dumps({k: r.get(k) for k in ("published", "loss_detected", "differences", "source_unchanged",
      "refused", "output_lineage")}))
print("  removed tables:", json.dumps({k: [v["disposition"], v["rows"]] for k, v in r["tables"].items()
      if v["disposition"] != "preserved"}))
print("  admission inputs:", json.dumps(r["tables"]["session_admissions"].get("inputs")))
print("  creation labels:", json.dumps(r["columns"]["sessions.created_source"]))
print("  labels:", " ".join(f"{e['source']}/{e['created_source']}" for e in r["labels"]))
assert r["published"] and r["committed"] and r["succeeded"] and not r["loss_detected"], "report"
assert r["source_unchanged"] is True, "source_unchanged"
assert r["columns"]["sessions.created_source"]["condition_broken_rows"] == 0, "created_source"
PY
[ "$(sha "$SRC")" = "$SRC_SHA" ] && echo "  source bytes unchanged: $SRC_SHA" || { echo "  FAIL source changed"; exit 1; }
run "$W/cand-home" "$CANDIDATE_PYTHON" "$REPO" "$W/steps/return.py" --verify "$W/out/state.db"
mkdir "$W/unmarked" && cp "$W/out/state.db" "$W/unmarked/state.db"
if run "$W/cand-home" "$CANDIDATE_PYTHON" "$REPO" "$W/steps/return.py" --verify "$W/unmarked/state.db"; then
  echo "  FAIL an output without its commit marker verified"; exit 1
fi
echo "  an output without its commit marker does not verify"
run "$W/cand-home" "$CANDIDATE_PYTHON" "$REPO" "$W/steps/compare_shape.py" "$W/fork-era.db" "$W/out/state.db"

step "3b. lookups: this build on the source, the fork binary on the output, for every label"
cp "$SRC" "$W/lookup-source.db"
cp "$W/out/state.db" "$W/lookup-output.db"
run "$W/cand-home" "$CANDIDATE_PYTHON" "$REPO" "$W/steps/lookups.py" candidate "$W/lookup-source.db" \
  "$W/lookups-candidate.json"
run "$W/return-home" "$FORK_PYTHON" "$FORK_TREE" "$W/steps/lookups.py" fork "$W/lookup-output.db" \
  "$W/lookups-fork.json" "$W/lookups-candidate.json"
python3 - "$W/lookups-candidate.json" "$W/lookups-fork.json" "$SRC" <<'PY'
import json, sqlite3, sys
cand, fork = (json.load(open(p)) for p in sys.argv[1:3])
conn = sqlite3.connect(f"file:{sys.argv[3]}?mode=ro", uri=True)
print("  source labels:", " ".join(f"{i}={s}/{c}" for i, s, c in conn.execute(
    "SELECT id, source, coalesce(created_source, 'NULL') FROM sessions ORDER BY id")))
conn.close()
bad = [(label, kind, cand["found"][label][kind], fork["found"][label][kind]) for label in cand["labels"]
       for kind in cand["found"][label] if cand["found"][label][kind] != fork["found"][label][kind]]
bad += [("row", i, cand["rows"].get(i), fork["rows"].get(i)) for i in sorted(set(cand["rows"]) | set(fork["rows"]))
        if cand["rows"].get(i) != fork["rows"].get(i)]
for label in cand["labels"]:
    if cand["found"][label]["list_with_archived"]:
        print(f"  {label}: {cand['found'][label]} (fork: {fork['found'][label]})")
for b in bad:
    print("  FAIL lookup differs:", json.dumps(b))
kinds = len(next(iter(cand["found"].values())))
print(f"  {len(cand['labels'])} labels x {kinds} lookups, {len(cand['rows'])} session rows: "
      + ("same" if not bad else f"{len(bad)} differ"))
sys.exit(1 if bad else 0)
PY

step "3c. label probes: both binaries' own lookups against the module's, for every alias-table label"
cp "$W/out/state.db" "$W/probe-output.db"
run "$W/return-home" "$FORK_PYTHON" "$FORK_TREE" "$W/steps/label_probe.py" fork "$W/probe-output.db" \
  "$W/probes-fork.json" "$W/probes-candidate.json"
run "$W/cand-home" "$CANDIDATE_PYTHON" "$REPO" "$W/steps/compare_probes.py" "$W/probes-candidate.json" \
  "$W/probes-fork.json"
mkdir "$W/desktop" && cp "$SRC" "$W/desktop/state.db"
run "$W/cand-home" "$CANDIDATE_PYTHON" "$REPO" "$W/steps/desktop_session.py" "$W/desktop/state.db"
set +e
run "$W/cand-home" "$CANDIDATE_PYTHON" "$REPO" "$W/steps/return.py" "$W/desktop/state.db" "$W/desktop/out.db" \
  2>"$W/desktop/stderr"
code=$?
set -e
grep -o "sessions with source='desktop'[^:]*" "$W/desktop/stderr" | sed 's/^/  refused: /' || true
if [ "$code" != 2 ] || [ -e "$W/desktop/out.db" ] || ! grep -q "source='desktop'" "$W/desktop/stderr"; then
  echo "  FAIL a desktop/desktop session did not refuse (exit $code)"; cat "$W/desktop/stderr"; exit 1
fi

step "4. fork binary on a copy of the output: open, read, write"
RET_DB=$W/return-home/.hermes/state.db
cp "$W/out/state.db" "$RET_DB"
run "$W/return-home" "$FORK_PYTHON" "$FORK_TREE" "$W/steps/fork_check.py" "$RET_DB" write

step "5. fork binary, new process (restart): read back"
run "$W/return-home" "$FORK_PYTHON" "$FORK_TREE" "$W/steps/fork_check.py" "$RET_DB" restart

step "6. fork binary: hermes sessions list"
run "$W/return-home" "$FORK_PYTHON" "$FORK_TREE" "$W/steps/fork_cli.py" | tail -12

echo
echo "PASS: returned store opened, read, written and restarted by the fork binary at $FORK_TREE"
