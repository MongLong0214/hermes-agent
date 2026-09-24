"""Cutover E2E: two profiles on fork-generation stores forward to each other through A2A, and both
end at this build's lineage with exact fences and every fork row kept; a store another build has
moved past refuses the next forward without a byte of DDL.

Real imports and real profile resolution (``Path.home`` + ``HERMES_HOME``, the test_profiles
pattern). The forward runs the forwarder's own steps: the real ``_state_db`` lookups and title write,
the real ``served_profile_child_env`` for the target home, and a child process. The one stand-in is
the child's program: ``hermes chat`` needs a model provider, and ``hermes`` from PATH may be a live
build, so the child is this interpreter writing its turn through ``SessionDB()`` resolved from the
env the forwarder built. That keeps the part under test (who opens, migrates and writes which
store) real.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.hermes_state.fence_store_probe import expected_fences, fence_triggers, read_ro, stored
from tests.hermes_state.fork_store_fixture import (
    CURRENT_STAMP, FORK_ROWS_JSON, FUTURE_STAMP, build_store, file_fingerprint, refence,
)

REPO = Path(__file__).resolve().parents[2]
# Fork rows the store must keep; the fork-derived authority tables are counted with them.
FIXTURE = json.loads(FORK_ROWS_JSON.read_text(encoding="utf-8"))["tables"]
DERIVED = ("session_process_authorities", "session_process_authority_events")
# Rows the migration owns: the lineage stamp is rewritten, and state_meta gains this build's keys.
MIGRATION_OWNED = ("schema_version", "state_meta")

_CHILD_TURN = r"""
import sys
from pathlib import Path
import uuid
import hermes_state

repo, expected_db, text = sys.argv[1:4]
assert Path(hermes_state.__file__).resolve().is_relative_to(Path(repo)), hermes_state.__file__
# Resolve the store the way `hermes chat` does, and refuse to open anything but the target's.
assert hermes_state._default_db_path() == Path(expected_db), hermes_state._default_db_path()
db = hermes_state.SessionDB()
try:
    session_id = f"a2a-{uuid.uuid4().hex[:12]}"
    db.create_session(session_id, "a2a")
    db.append_message(session_id, "user", text)
    db.append_message(session_id, "assistant", f"reply to {text}")
finally:
    db.close()
print(f"reply to {text}")
"""


def _forward(profile: str, context_id: str, text: str):
    """A2A forward to *profile*: title lookup, child turn, latest-a2a lookup, title write."""
    from plugins.platforms.a2a import adapter
    from tools.environments.local import served_profile_child_env

    title = f"a2a-{profile}-{adapter._safe_context_slug(context_id)}"
    resumed = adapter._state_db(profile, "SELECT id FROM sessions WHERE title = ? ORDER BY started_at DESC LIMIT 1",
                                (title,), "lookup")
    home = adapter._profile_home(profile)
    env = served_profile_child_env(target_home=home, inherit_credentials=True)
    start = time.time()
    proc = subprocess.run([sys.executable, "-c", _CHILD_TURN, str(REPO), str(Path(home) / "state.db"), text],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=120, check=False,
                          stdin=subprocess.DEVNULL)
    session_id = ""
    if proc.returncode == 0:
        session_id = adapter._state_db(
            profile, "SELECT id FROM sessions WHERE source = 'a2a' AND started_at >= ? ORDER BY started_at DESC LIMIT 1",
            (start - 2.0,), "latest")
        if session_id:
            adapter._state_db(profile, "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id), "title",
                              commit=True)
    return proc, resumed, session_id, title


def _count(db, table: str) -> int:
    return read_ro(db, f'SELECT count(*) FROM "{table}"')[0][0]


def _linked(db, table: str, session_ids) -> int:
    """Rows of *table* that belong to *session_ids* (the rows this test wrote)."""
    columns = {row[1] for row in read_ro(db, f'PRAGMA table_info("{table}")')}
    key = "id" if table == "sessions" else "session_id" if "session_id" in columns else None
    if key is None or not session_ids:
        return 0
    marks = ", ".join("?" for _ in session_ids)
    return read_ro(db, f'SELECT count(*) FROM "{table}" WHERE "{key}" IN ({marks})', tuple(session_ids))[0][0]


def _baseline(db) -> dict:
    return {t: _count(db, t) for t in (*FIXTURE, *DERIVED) if t not in MIGRATION_OWNED}


def _assert_cut_over_losslessly(db, baseline: dict, new_sessions: dict) -> None:
    assert stored(db) == CURRENT_STAMP
    assert fence_triggers(db) == expected_fences(db, CURRENT_STAMP)
    for table, before in baseline.items():
        assert _count(db, table) == before + _linked(db, table, list(new_sessions)), table
    for table, spec in FIXTURE.items():
        if table in MIGRATION_OWNED:
            continue
        names = ", ".join(f'"{c}"' for c in spec["columns"])
        kept = {tuple(r) for r in read_ro(db, f'SELECT {names} FROM "{table}"')}
        assert {tuple(r) for r in spec["rows"]} <= kept, table
    fixture_keys = {row[0] for row in FIXTURE["state_meta"]["rows"]}
    assert fixture_keys <= {row[0] for row in read_ro(db, "SELECT key FROM state_meta")}
    # Every turn this test wrote is whole: its session and each of its messages.
    for session_id, messages in new_sessions.items():
        assert _linked(db, "sessions", [session_id]) == 1, session_id
        assert _linked(db, "messages", [session_id]) == messages, session_id


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    """Default profile A and named profile B, both on fork-generation-29 stores."""
    # HOME stays the suite's: its live-system guard reads HOME/.hermes as the production root.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    a_home = tmp_path / ".hermes"
    b_home = a_home / "profiles" / "b"
    monkeypatch.setenv("HERMES_HOME", str(a_home))
    return build_store(a_home / "state.db", "fork29"), build_store(b_home / "state.db", "fork29")


def test_a2a_round_trip_cuts_both_fork_stores_over_and_a_newer_store_refuses(two_profiles, monkeypatch):
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    a_db, b_db = two_profiles
    baselines = {a_db: _baseline(a_db), b_db: _baseline(b_db)}
    written = {a_db: {}, b_db: {}}

    # 1. A's own turn, on the store A's profile resolves to.
    assert get_hermes_home() / "state.db" == a_db
    db = SessionDB(db_path=get_hermes_home() / "state.db")
    try:
        db.create_session("a-own", "cli")
        db.append_message("a-own", "user", "hello from A")
        db.append_message("a-own", "assistant", "A answers")
    finally:
        db.close()
    written[a_db]["a-own"] = 2
    assert stored(b_db) == 29  # B is untouched until something forwards to it

    # 2. A forwards to B: B's first writer bootstraps it to this build's lineage, and the forwarder's
    # own title write lands through the fenced connection.
    proc, resumed, b_session, b_title = _forward("b", "ctx-ab", "task for B")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert resumed == "" and b_session
    assert stored(b_db) == CURRENT_STAMP
    assert read_ro(b_db, "SELECT title FROM sessions WHERE id = ?", (b_session,)) == [(b_title,)]
    written[b_db][b_session] = 2

    # 3. B forwards back to A, as B's own gateway: "default" resolves from B's home to A.
    monkeypatch.setenv("HERMES_HOME", str(b_db.parent))
    proc, resumed, a_session, a_title = _forward("default", "ctx-ba", "task for A")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert resumed == "" and a_session
    assert read_ro(a_db, "SELECT title FROM sessions WHERE id = ?", (a_session,)) == [(a_title,)]
    written[a_db][a_session] = 2

    # 4. A reopens its store and keeps writing.
    monkeypatch.setenv("HERMES_HOME", str(a_db.parent))
    db = SessionDB(db_path=get_hermes_home() / "state.db")
    try:
        db.append_message("a-own", "user", "after the round trip")
        assert db.get_session(a_session)["title"] == a_title
    finally:
        db.close()
    written[a_db]["a-own"] += 1

    # 5. Both stores: this build's stamp, exact fences, every fork row kept, every new row whole.
    for store in (a_db, b_db):
        _assert_cut_over_losslessly(store, baselines[store], written[store])

    # 6. Another build moves B past this one. The next forward is refused at every step, and B is
    # left exactly as that build wrote it: no DDL, no schema cookie bump, no write.
    import sqlite3

    conn = sqlite3.connect(str(b_db), isolation_level=None)
    try:
        refence(conn, FUTURE_STAMP)
        conn.execute("UPDATE schema_version SET version = ?", (FUTURE_STAMP,))
    finally:
        conn.close()
    before = file_fingerprint(b_db)

    proc, resumed, refused_session, _ = _forward("b", "ctx-ab", "task B must refuse")
    from plugins.platforms.a2a import adapter
    retitled = adapter._state_db("b", "UPDATE sessions SET title = ? WHERE id = ?", ("refused", b_session),
                                 "title", commit=True)

    from hermes_state_errors import incompatible_schema_cause
    # The child dies on the open-time refusal; only its text crosses the process boundary.
    assert proc.returncode != 0 and incompatible_schema_cause(proc.stderr) == "BUILD_TOO_OLD", proc.stderr[-2000:]
    assert (resumed, refused_session, retitled) == ("", "", "")
    after = file_fingerprint(b_db)
    assert (after["sqlite_master"], after["schema_cookie"]) == (before["sqlite_master"], before["schema_cookie"])
    assert after == before
