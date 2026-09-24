#!/usr/bin/env python3
"""Regenerate the fork generation-29 state.db fixture from the fork's own code.

The fixture must be derived from the fork commit's builders, never from a live
store: a live state.db carries personal data and local paths, and opening it
from a candidate build risks migrating a production store.

Run from the repo root:

    python scripts/state_fence/gen_fork_fixture.py [--rev 6c9d8d8074] [--python PY]

``--python`` must be an interpreter that can import the fork tree's
dependencies (the live venv interpreter works). The script extracts the fork
tree with ``git archive``, runs the fork's ``SessionDB`` against a throwaway
HOME/HERMES_HOME, and writes:

    tests/hermes_state/fixtures/fork_gen29_schema.sql
    tests/hermes_state/fixtures/fork_gen29_rows.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_REV = "6c9d8d8074"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "tests" / "hermes_state" / "fixtures"

_CHILD = r'''
import json, os, sqlite3, sys, time
from pathlib import Path

fork = Path(sys.argv[1]).resolve()
home = Path(os.environ["HERMES_HOME"])
import hermes_state
assert Path(hermes_state.__file__).resolve().is_relative_to(fork), hermes_state.__file__
from hermes_state import SessionDB
from hermes_state_common import TURN_FENCE_GENERATION, register_turn_fence_generation
assert TURN_FENCE_GENERATION == 29, TURN_FENCE_GENERATION

db_path = home / "state.db"
db = SessionDB(db_path=db_path)
db.create_session("fx-alpha", "cli")
db.create_session("fx-beta", "telegram")
db.append_message("fx-alpha", "user", "hello fixture")
db.append_message("fx-alpha", "assistant", None, tool_calls=[
    {"id": "call_1", "type": "function",
     "function": {"name": "read_file", "arguments": "{}"}}])
db.append_message("fx-alpha", "tool", "tool output", tool_name="read_file", tool_call_id="call_1")
db.append_message("fx-alpha", "assistant", "done")
db.append_message("fx-beta", "user", "second session")
db.update_token_counts("fx-alpha", 11, 7, model="fixture-model")
db.save_gateway_routing_entry("fixture:key", json.dumps({"platform": "cli"}))
assert db.try_acquire_compression_lock("fx-beta", "fixture-holder", ttl_seconds=300.0)
db.end_session("fx-alpha", "user_exit")
db.end_session("fx-beta", "user_exit")
db.reopen_session("fx-beta")
db.close()

from tools import async_delegation
async_delegation._persist_dispatch({
    "delegation_id": "fx-deleg-1", "dispatched_at": time.time(),
    "session_key": "fixture:key", "origin_ui_session_id": "",
    "parent_session_id": "fx-alpha", "origin_session_id": "fx-alpha",
    "goal": "fixture goal",
})

conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
register_turn_fence_generation(conn)
virtual = {r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE 'CREATE VIRTUAL TABLE%'")}
shadow_suffixes = ("_data", "_idx", "_content", "_docsize", "_config")
shadow = {f"{v}{s}" for v in virtual for s in shadow_suffixes}
objects = [
    (t, n, s) for t, n, s in conn.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY rowid")
    if not n.startswith("sqlite_") and n not in shadow
]
rows = {}
for t, n, _ in objects:
    # Authority rows are derived by the fork's own triggers when the builder
    # replays sessions, so storing them would double-issue on replay.
    if t != "table" or n in virtual or n.startswith("session_process_"):
        continue
    cur = conn.execute(f'SELECT * FROM "{n}" ORDER BY rowid')
    cols = [d[0] for d in cur.description]
    data = cur.fetchall()
    if data:
        rows[n] = {"columns": cols, "rows": [list(r) for r in data]}
json.dump({
    "sqlite_version": sqlite3.sqlite_version,
    "objects": [[t, n, s] for t, n, s in objects],
    "rows": rows,
}, sys.stdout)
'''


def _extract(rev: str, dest: Path) -> None:
    archive = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "archive", rev],
        check=True, capture_output=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)


def _assert_anonymous(text: str) -> None:
    lowered = text.lower()
    home_name = Path.home().name.lower()
    for needle in ("/users/", "/home/", "/private/", "/var/folders", home_name):
        if needle and needle in lowered:
            raise SystemExit(f"fixture output contains a local path marker: {needle!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rev", default=DEFAULT_REV)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    sha = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", args.rev],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    scratch = Path(tempfile.mkdtemp(prefix="fork-fixture-"))
    try:
        fork = scratch / "fork"
        home = scratch / "home"
        hermes_home = home / ".hermes"
        fork.mkdir()
        hermes_home.mkdir(parents=True)
        _extract(sha, fork)
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(home),
            "HERMES_HOME": str(hermes_home),
            "PYTHONPATH": str(fork),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TZ": "UTC",
            "LANG": "C.UTF-8",
        }
        result = subprocess.run(
            [args.python, "-c", _CHILD, str(fork)],
            cwd=fork, env=env, check=True, capture_output=True, text=True,
        )
        dump = json.loads(result.stdout)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    header = (
        f"-- Fork generation-29 state.db DDL, generated from commit {sha}\n"
        f"-- by scripts/state_fence/gen_fork_fixture.py with SQLite {dump['sqlite_version']}.\n"
        "-- sqlite_master.sql in rowid order, minus sqlite_% and FTS shadow tables.\n"
    )
    schema = header + "".join(f"{sql};\n" for _, _, sql in dump["objects"])
    rows = json.dumps(
        {"fork_commit": sha, "sqlite_version": dump["sqlite_version"], "tables": dump["rows"]},
        indent=1, sort_keys=True,
    ) + "\n"
    _assert_anonymous(schema)
    _assert_anonymous(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "fork_gen29_schema.sql").write_text(schema, encoding="utf-8")
    (OUT_DIR / "fork_gen29_rows.json").write_text(rows, encoding="utf-8")
    print(f"wrote fixture from {sha}: {len(dump['objects'])} objects, "
          f"{sum(len(t['rows']) for t in dump['rows'].values())} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
