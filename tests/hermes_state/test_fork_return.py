"""Returning this build's fenced 1033 state.db to the live fork's 1030 schema (hermes_state_fork_return).

The fixture is a fork 1030 store carrying the fork's session-process authority machinery (its DDL as the fork
wrote it), migrated to 1033 by this build's real migration and then written by this build: a turn with minted
uids, a revoked and a reopened authority, a turn lease, a conversation boundary, a runtime epoch and finished
runtime work. Fixtures are synthesized here; none is a copy of a real store.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_state_fence as fence
import hermes_state_fork_return as fork_return
from hermes_state import SessionDB
from tests.hermes_state.test_state_lineage_admission import live_fence_trigger_sql

OPERATIONS = ("INSERT", "UPDATE", "DELETE")
STATE_DB_ID = "ab" * 32
TG_KEY = "agent:p1:telegram:dm:1"

# The fork's session-process authority objects, as its legacy generations wrote them.
FORK_AUTHORITY_DDL = (
    "ALTER TABLE sessions ADD COLUMN session_generation INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE session_turn_leases ADD COLUMN owner_pid INTEGER",
    "ALTER TABLE session_turn_leases ADD COLUMN owner_pid_start REAL",
    """CREATE TABLE session_process_authorities (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    session_generation INTEGER NOT NULL CHECK (session_generation >= 1),
    state_db_id TEXT NOT NULL CHECK (length(state_db_id) = 64 AND state_db_id NOT GLOB '*[^0-9a-f]*'),
    state_family TEXT NOT NULL CHECK (state_family = 'sessiondb-v1'),
    authority_token TEXT NOT NULL CHECK (length(authority_token) = 64 AND authority_token NOT GLOB '*[^0-9a-f]*'),
    status TEXT NOT NULL CHECK (status IN ('ISSUED', 'CLOSED', 'REVOKED')),
    issued_at REAL NOT NULL,
    terminal_at REAL,
    PRIMARY KEY (session_id, session_generation)
)""",
    """CREATE TABLE session_process_authority_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    session_generation INTEGER NOT NULL CHECK (session_generation >= 1),
    state_db_id TEXT NOT NULL,
    state_family TEXT NOT NULL CHECK (state_family = 'sessiondb-v1'),
    event_type TEXT NOT NULL,
    reservation_id TEXT,
    occurred_at REAL NOT NULL
)""",
    """CREATE INDEX idx_session_process_authorities_current
    ON session_process_authorities(session_id, session_generation DESC)""",
    """CREATE TRIGGER session_process_authority_generation_insert_guard
BEFORE INSERT ON sessions
WHEN NEW.session_generation <> 0
BEGIN
    SELECT RAISE(ABORT, 'session authority generation must start at zero');
END""",
    """CREATE TRIGGER session_process_authority_generation_monotonic_guard
BEFORE UPDATE OF session_generation ON sessions
WHEN NEW.session_generation <> OLD.session_generation + 1
BEGIN
    SELECT RAISE(ABORT, 'session authority generation must be monotonic');
END""",
    """CREATE TRIGGER session_process_authority_sessions_issue
AFTER INSERT ON sessions
BEGIN
    UPDATE sessions SET session_generation = CASE
        WHEN session_generation < 1 THEN 1 ELSE session_generation END
    WHERE id = NEW.id;
    INSERT INTO session_process_authorities (
        session_id, session_generation, state_db_id, state_family, authority_token, status, issued_at)
    SELECT s.id, s.session_generation,
           (SELECT value FROM state_meta WHERE key = 'session_process_state_db_id'),
           (SELECT value FROM state_meta WHERE key = 'session_process_state_family'),
           lower(hex(randomblob(32))), 'ISSUED', s.started_at
    FROM sessions AS s WHERE s.id = NEW.id;
    INSERT INTO session_process_authority_events (
        session_id, session_generation, state_db_id, state_family, event_type, occurred_at)
    SELECT session_id, session_generation, state_db_id, state_family, 'SESSION_ISSUED', issued_at
    FROM session_process_authorities
    WHERE session_id = NEW.id
      AND session_generation = (SELECT session_generation FROM sessions WHERE id = NEW.id);
END""",
    """CREATE TRIGGER session_process_authority_sessions_close
AFTER UPDATE OF ended_at ON sessions
WHEN OLD.ended_at IS NULL AND NEW.ended_at IS NOT NULL
BEGIN
    UPDATE session_process_authorities
    SET status = CASE WHEN NEW.end_reason = 'authority_revoked' THEN 'REVOKED' ELSE 'CLOSED' END,
        terminal_at = NEW.ended_at
    WHERE session_id = NEW.id AND session_generation = NEW.session_generation AND status = 'ISSUED';
    INSERT INTO session_process_authority_events (
        session_id, session_generation, state_db_id, state_family, event_type, occurred_at)
    SELECT session_id, session_generation, state_db_id, state_family,
           CASE WHEN NEW.end_reason = 'authority_revoked' THEN 'SESSION_REVOKED' ELSE 'SESSION_CLOSED' END,
           NEW.ended_at
    FROM session_process_authorities
    WHERE session_id = NEW.id AND session_generation = NEW.session_generation
      AND status = CASE WHEN NEW.end_reason = 'authority_revoked' THEN 'REVOKED' ELSE 'CLOSED' END;
END""",
    """CREATE TRIGGER session_process_authority_sessions_reopen
AFTER UPDATE OF ended_at ON sessions
WHEN OLD.ended_at IS NOT NULL AND NEW.ended_at IS NULL
BEGIN
    UPDATE sessions SET session_generation = session_generation + 1 WHERE id = NEW.id;
    INSERT INTO session_process_authorities (
        session_id, session_generation, state_db_id, state_family, authority_token, status, issued_at)
    SELECT s.id, s.session_generation,
           (SELECT value FROM state_meta WHERE key = 'session_process_state_db_id'),
           (SELECT value FROM state_meta WHERE key = 'session_process_state_family'),
           lower(hex(randomblob(32))), 'ISSUED', strftime('%s', 'now')
    FROM sessions AS s WHERE s.id = NEW.id;
    INSERT INTO session_process_authority_events (
        session_id, session_generation, state_db_id, state_family, event_type, occurred_at)
    SELECT session_id, session_generation, state_db_id, state_family, 'SESSION_ISSUED', issued_at
    FROM session_process_authorities
    WHERE session_id = NEW.id
      AND session_generation = (SELECT session_generation FROM sessions WHERE id = NEW.id);
END""",
    """CREATE TRIGGER session_process_authority_events_append_only_delete
BEFORE DELETE ON session_process_authority_events
BEGIN
    SELECT RAISE(ABORT, 'session process authority events are append-only');
END""",
)

# Written out here, independently of the module's own list: what upstream 31-33 added to the fork's 30.
_UPSTREAM_ADDED_TRIGGERS = ("messages_display_revision_update", "messages_message_uid_insert",
                            "messages_transcript_revision_update", "sessions_runtime_ledger_delete_guard")
_UPSTREAM_ADDED_INDEXES = ("idx_messages_session_uid", "idx_sessions_local_created_started")
_UPSTREAM_ADDED_TABLES = ("worker_receipts", "worker_executions", "session_admissions", "runtime_epoch")
_UPSTREAM_ADDED_COLUMNS = (
    ("async_delegations", "owner_execution_id"), ("messages", "message_uid"), ("messages", "absorbed_message_uids"),
    ("messages", "tool_call_uid"), ("messages", "tool_call_uids"), ("sessions", "compression_overload_streak"),
    ("sessions", "runtime_revision"), ("sessions", "created_source"), ("sessions", "transcript_revision"),
    ("sessions", "runtime_generation"), ("sessions", "auto_archived"),
)


def _fork_fence(conn) -> None:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    for table in fence.CORE_GOVERNED_TABLES + tuple(t for t in fence.SESSION_PROCESS_GOVERNED_TABLES if t in tables):
        for operation in OPERATIONS:
            conn.execute(live_fence_trigger_sql(table, operation, 1030))


def _shape(path: Path) -> dict:
    """Every non-fence schema object's SQL plus every table's columns."""
    conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        objects = {(t, n): sql for t, n, sql in conn.execute("SELECT type, name, sql FROM sqlite_master")
                   if not n.startswith("turn_fence_")}
        columns = {n: [r[1:] for r in conn.execute(f'PRAGMA table_xinfo("{n}")')]
                   for (t, n) in objects if t == "table"}
        fences = dict(conn.execute("SELECT name, sql FROM sqlite_master WHERE name LIKE 'turn_fence_%'").fetchall())
        stamp = conn.execute("SELECT version FROM schema_version").fetchone()[0]
    finally:
        conn.close()
    return {"objects": objects, "columns": columns, "fences": fences, "stamp": stamp}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path, sql: str) -> list:
    conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        return [tuple(r) for r in conn.execute(sql)]
    finally:
        conn.close()


def fork_1030_store(path: Path) -> dict:
    """The fork's 1030 shape: this build's tables minus what upstream 31-33 added, plus the fork's authority
    machinery, fork-era rows, the fork's fences and stamp. Returns its shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=path)
    try:
        db.create_session("s1", source="cli")
        db.append_message("s1", role="user", content="hello fork store")
        db.append_message("s1", role="assistant", content="the needle is here")
        db.create_session("tg1", source="telegram", session_key=TG_KEY)
        db.append_message("tg1", role="user", content="telegram before the upgrade")
        db.create_session("s3", source="cli")
    finally:
        db.close()
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.execute("BEGIN")
        for name in _UPSTREAM_ADDED_TRIGGERS:
            conn.execute(f"DROP TRIGGER {name}")
        for name in _UPSTREAM_ADDED_INDEXES:
            conn.execute(f"DROP INDEX {name}")
        for name in _UPSTREAM_ADDED_TABLES:
            conn.execute(f"DROP TABLE {name}")
        for table, column in _UPSTREAM_ADDED_COLUMNS:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
        conn.execute("DELETE FROM state_meta WHERE key LIKE 'message_uid_backfill%' OR key LIKE 'legacy_refusal_repair%'")
        for sql in FORK_AUTHORITY_DDL:
            conn.execute(sql)
        conn.execute("INSERT INTO state_meta (key, value) VALUES ('session_process_state_db_id', ?), "
                     "('session_process_state_family', 'sessiondb-v1')", (STATE_DB_ID,))
        # Authorities for the fork-era sessions, as the fork's issue trigger left them.
        for (sid,) in conn.execute("SELECT id FROM sessions ORDER BY id").fetchall():
            conn.execute("UPDATE sessions SET session_generation = 1 WHERE id = ?", (sid,))
            conn.execute("INSERT INTO session_process_authorities VALUES (?, 1, ?, 'sessiondb-v1', ?, 'ISSUED', 1.0, NULL)",
                         (sid, STATE_DB_ID, hashlib.sha256(sid.encode()).hexdigest()))
            conn.execute("INSERT INTO session_process_authority_events (session_id, session_generation, state_db_id, "
                         "state_family, event_type, occurred_at) VALUES (?, 1, ?, 'sessiondb-v1', 'SESSION_ISSUED', 1.0)",
                         (sid, STATE_DB_ID))
        conn.execute("INSERT INTO conversation_generations VALUES ('telegram', ?, 3)", (TG_KEY,))
        conn.execute("INSERT INTO session_turn_leases VALUES ('tg1', 'pid=999999:turn=fork-era', 1.0, 2.0, 999999, 0.5)")
        conn.execute("UPDATE schema_version SET version = 1030")
        _fork_fence(conn)
        conn.execute("COMMIT")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    lineage = fence.probe_store_lineage(path)
    assert lineage.migratable, lineage
    return _shape(path)


def run_input(db, session_id: str, request_id: str, text: str, *, transcript: bool, outcome="completed",
              payload=None, cancel=False, stored=None, platform_message_id=None, principal="messaging:1") -> int:
    """One input through this build's real admission APIs: admitted, then cancelled before it runs, or claimed
    and settled with *outcome* (its turn's user message written first when *transcript*: *stored* as its content
    when the turn records more than the text, with the platform's message id when it has one). Returns the epoch."""
    from hermes_state_runtime import (admit_session_input, begin_runtime_epoch, cancel_session_input,
                                      claim_session_input, settle_session_input)

    epoch = begin_runtime_epoch(db, instance_id="candidate-1")
    admission = admit_session_input(db, epoch=epoch, principal_id=principal, session_id=session_id,
                                    request_id=request_id, payload=payload or {"text": text})
    if cancel:
        assert cancel_session_input(db, epoch=epoch, admission_id=admission["admission_id"])["outcome"] == "cancelled"
        return epoch
    claim = claim_session_input(db, epoch=epoch, session_id=session_id)
    assert claim["admission_id"] == admission["admission_id"]
    if transcript:
        db.append_message(session_id, role="user", content=text if stored is None else stored,
                          platform_message_id=platform_message_id)
        db.append_message(session_id, role="assistant", content=f"answered: {text}")
    settle_session_input(db, epoch=epoch, admission_id=claim["admission_id"], generation=claim["generation"],
                         outcome=outcome)
    return epoch


def candidate_writes(path: Path) -> None:
    """Migrate to 1033 with this build's real migration, then write what this build writes."""
    db = SessionDB(db_path=path)
    try:
        assert db._conn.execute("SELECT version FROM schema_version").fetchone()[0] == 1033
        db.append_message("tg1", role="user", content="telegram after the upgrade qv82")
        db.append_message("tg1", role="assistant", content="answered by 1033")
        db.create_session("s2", source="tui")  # created_source = source: the fork lists it under tui too
        db.end_session("s1", "authority_revoked")
        db.end_session("s3", "user_exit")
        db.reopen_session("s3")
        assert db.try_acquire_session_turn_lease("tg1", "pid=1:turn=candidate", ttl_seconds=3600)
        epoch = run_input(db, "tg1", "r1", "done", transcript=True)

        def finish(conn):
            conn.execute("INSERT INTO worker_executions VALUES ('x1', 'tg1', 'child', ?, 1, 'terminal', 'd', 1)", (epoch,))
            conn.execute("INSERT INTO worker_receipts VALUES ('x1', 1, 'p', '{}')")
            conn.execute("UPDATE conversation_generations SET generation = generation + 1 WHERE session_key = ?", (TG_KEY,))
            conn.execute("UPDATE sessions SET auto_archived = 1, archived = 1 WHERE id = 's2'")
        db._execute_write(finish)
    finally:
        db.close()
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    assert fence.probe_store_lineage(path).settled


@pytest.fixture
def stores(tmp_path):
    source = tmp_path / "home" / "state.db"
    fork_shape = fork_1030_store(source)
    candidate_writes(source)
    return source, fork_shape, tmp_path / "out" / "state.db"


def _export(source, output, **kwargs):
    output.parent.mkdir(parents=True, exist_ok=True)
    return fork_return.return_to_fork(source, output, **kwargs)


FORK_ROWS = {
    "authorities": "SELECT session_id, session_generation, authority_token, status, terminal_at "
                   "FROM session_process_authorities ORDER BY 1, 2",
    "events": "SELECT * FROM session_process_authority_events ORDER BY id",
    "leases": "SELECT * FROM session_turn_leases ORDER BY 1",
    "generations": "SELECT * FROM conversation_generations ORDER BY 1, 2",
    "sessions": "SELECT id, source, parent_session_id, ended_at, end_reason, archived, session_generation "
                "FROM sessions ORDER BY id",
    "messages": "SELECT id, session_id, role, content, active FROM messages ORDER BY id",
    "routing": "SELECT * FROM gateway_routing ORDER BY 1, 2",
}


# ── The return ──

def test_returned_store_is_the_forks_own_shape_with_every_row(stores):
    source, fork_shape, output = stores
    sha = _sha(source)
    before = {name: _rows(source, sql) for name, sql in FORK_ROWS.items()}
    report_path = output.parent / "report.json"

    output.parent.mkdir(parents=True)
    assert fork_return.main([str(source), str(output), "--report", str(report_path)]) == 0

    report = json.loads(report_path.read_text())
    assert report["published"] and not report["loss_detected"] and report["source_unchanged"]
    assert report["committed"] and report["succeeded"] and report["marker"] == f"{output}.fork-return"
    assert fork_return.verify_return(output)["sha256"] == _sha(output)  # the consumer's check: a committed return
    assert report["output_lineage"] == {"lineage": "fenced", "schema_version": 1030,
                                        "fence_generations": [1030], "fork_store": True}
    assert _sha(source) == sha
    # Round trip of the schema: 1030 -> this build's migration -> return is the fork's 1030 again.
    returned = _shape(output)
    assert returned["stamp"] == 1030
    assert returned["columns"] == fork_shape["columns"]
    assert returned["objects"] == fork_shape["objects"]
    assert returned["fences"] == fork_shape["fences"]
    # Every fork row, including what this build wrote while it owned the store.
    after = {name: _rows(output, sql) for name, sql in FORK_ROWS.items()}
    assert after == before
    assert ("s1", 1, hashlib.sha256(b"s1").hexdigest(), "REVOKED") == after["authorities"][0][:4]
    assert [r[1:4:2] for r in after["authorities"] if r[0] == "s3"] == [(1, "CLOSED"), (2, "ISSUED")]
    assert after["generations"] == [("telegram", TG_KEY, 4)]
    # This build reclaimed the fork's expired lease; the holder it superseded is not the one returned.
    assert [r[:2] for r in after["leases"]] == [("tg1", "pid=1:turn=candidate")]
    # What left, and how much of it was set.
    tables, columns = report["tables"], report["columns"]
    assert {k: (v["disposition"], v["rows"]) for k, v in tables.items() if v["disposition"] != "preserved"} == {
        "runtime_epoch": ("converted", 1), "session_admissions": ("converted", 1),
        "worker_executions": ("converted", 1), "worker_receipts": ("converted", 1),
        "schema_version": ("converted", 1)}
    assert all(v["absent_or_changed_rows"] == 0 for v in tables.values() if v["disposition"] == "preserved"
               and "absent_or_changed_rows" in v)
    n_messages = len(after["messages"])
    assert columns["messages.message_uid"]["non_default_rows"] == n_messages
    assert columns["sessions.created_source"]["non_default_rows"] >= 1
    assert columns["sessions.auto_archived"]["non_default_rows"] == 1
    # Every column of every removed table has its disposition too, with how many of its rows were set.
    removed = {f"{t}.{c}": "converted" for t in _UPSTREAM_ADDED_TABLES
               for (c,) in _rows(source, f"SELECT name FROM pragma_table_xinfo('{t}')")}
    assert len(removed) == 28
    assert {k: v["disposition"] for k, v in columns.items()} == {
        **{f"{t}.{c}": "converted" for t, c in _UPSTREAM_ADDED_COLUMNS}, **removed}
    assert columns["session_admissions.payload_json"]["non_default_rows"] == 1
    # The finished input is matched to its turn's user message, not assumed to be there.
    assert tables["session_admissions"]["inputs"] == {
        "matched_to_messages": 1, "lossless_return_impossible": 0, "no_rule": 0,
        "by_kind": {"text": {"matched": 1, "lossless_return_impossible": 0, "no_rule": 0}}}
    assert "message_uid_backfill" in report["state_meta"]
    assert _rows(output, "SELECT key FROM state_meta WHERE key LIKE 'message_uid_backfill%' "
                         "OR key LIKE 'legacy_refusal_repair%'") == []


def _connect_as(path: Path, generation: int) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None)
    conn.create_function("hermes_turn_fence_generation", 0, lambda: generation, deterministic=True)
    return conn


def test_the_forks_generation_writes_it_and_this_builds_cannot(stores):
    source, _, output = stores
    _export(source, output)
    conn = _connect_as(output, 1033)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="generation incompatible"):
            conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('stale', 'cli', 9.0)")
    finally:
        conn.close()
    conn = _connect_as(output, 1030)
    try:
        conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('after', 'cli', 9.0)")
        conn.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES ('after', 'user', 'x', 9.0)")
        # The fork's own trigger issues the new session's authority, as it would have before the upgrade.
        assert conn.execute("SELECT session_generation, status FROM session_process_authorities "
                            "WHERE session_id = 'after'").fetchall() == [(1, "ISSUED")]
        # A revoked holder's authority is never issued again: reopening issues generation 2 with a new token.
        revoked = conn.execute("SELECT authority_token FROM session_process_authorities WHERE session_id = 's1'").fetchone()[0]
        conn.execute("UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = 's1'")
        rows = conn.execute("SELECT session_generation, authority_token, status FROM session_process_authorities "
                            "WHERE session_id = 's1' ORDER BY 1").fetchall()
        assert [(g, s) for g, _, s in rows] == [(1, "REVOKED"), (2, "ISSUED")] and rows[1][1] != revoked
        # A conversation boundary continues the counter instead of re-issuing generation 1.
        conn.execute("INSERT INTO conversation_generations (source, session_key, generation) VALUES ('telegram', ?, 1) "
                     "ON CONFLICT(source, session_key) DO UPDATE SET generation = conversation_generations.generation + 1",
                     (TG_KEY,))
        assert conn.execute("SELECT generation FROM conversation_generations").fetchone()[0] == 5
        # The lease's holder is the one the source had; the fork-era holder this build superseded cannot extend it.
        assert conn.execute("UPDATE session_turn_leases SET expires_at = 1e12 WHERE conversation_id = 'tg1' "
                            "AND holder = 'pid=999999:turn=fork-era'").rowcount == 0
    finally:
        conn.close()


def test_the_returned_store_migrates_forward_again(stores):
    source, _, output = stores
    _export(source, output)
    SessionDB(db_path=output).close()
    assert fence.probe_store_lineage(output).settled
    # The backfill marker left with the column, so this build mints every row's uid again.
    assert _rows(output, "SELECT count(*) FROM messages WHERE message_uid IS NULL") == [(0,)]
    for name in ("authorities", "events", "generations", "messages"):
        assert _rows(output, FORK_ROWS[name]) == _rows(source, FORK_ROWS[name])


def _wal_bearing_copy(source: Path, directory: Path) -> Path:
    """A copy of *source* whose WAL holds a commit its main file lacks, as a crashed writer leaves it."""
    crashed = directory / "state.db"
    directory.mkdir()
    conn = sqlite3.connect(source, isolation_level=None)
    try:
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("INSERT INTO state_meta (key, value) VALUES ('only-in-the-wal', '1')")
        crashed.write_bytes(source.read_bytes())
        Path(f"{crashed}-wal").write_bytes(Path(f"{source}-wal").read_bytes())
    finally:
        conn.close()
    assert Path(f"{crashed}-wal").stat().st_size
    return crashed


def test_a_wal_bearing_source_is_read_through_a_copy(stores, tmp_path):
    source, _, output = stores
    crashed = _wal_bearing_copy(source, tmp_path / "crashed")
    shas = {p: _sha(p) for p in (crashed, Path(f"{crashed}-wal"))}

    _export(crashed, output)

    assert {p: _sha(p) for p in shas} == shas and not Path(f"{crashed}-shm").exists()
    assert _rows(output, "SELECT value FROM state_meta WHERE key = 'only-in-the-wal'") == [("1",)]
    assert sorted(p.name for p in output.parent.iterdir()) == ["state.db", "state.db.fork-return"]


# ── Refusals: nothing is published and the source is untouched ──

def _assert_refused(source, output, match: str) -> dict:
    sha = _sha(source)
    with pytest.raises(fork_return.ForkReturnRefused, match=match) as caught:
        _export(source, output)
    assert _sha(source) == sha
    assert not output.exists() and not list(output.parent.glob(".*return*"))
    assert not Path(f"{output}.fork-return").exists() and not caught.value.report.get("committed")
    assert not caught.value.report["published"] and caught.value.report["source_unchanged"] is True
    return caught.value.report


def _write_1033(path: Path, sql: str, *params) -> None:
    db = SessionDB(db_path=path)
    try:
        db._execute_write(lambda conn: conn.execute(sql, params))
    finally:
        db.close()


@pytest.mark.parametrize("sql, match", [
    ("UPDATE session_admissions SET status = 'queued', outcome = NULL", r"session_admissions holds unfinished work"),
    ("UPDATE worker_executions SET status = 'running'", r"worker_executions holds unfinished work"),
])
def test_unfinished_runtime_work_refuses(stores, sql, match):
    source, _, output = stores
    _write_1033(source, sql)
    _assert_refused(source, output, match)


@pytest.mark.parametrize("ddl, match", [
    (("CREATE TABLE hosted_room_attachments (id TEXT)", "INSERT INTO hosted_room_attachments VALUES ('a')"),
     r"table hosted_room_attachments \(1 rows\)"),
    (("CREATE TRIGGER sessions_newer_guard BEFORE DELETE ON sessions BEGIN SELECT 1; END",),
     r"trigger sessions_newer_guard on sessions"),
    (("ALTER TABLE messages ADD COLUMN newer_column TEXT",), r"column messages.newer_column"),
])
def test_an_object_the_fork_does_not_have_refuses(stores, ddl, match):
    source, _, output = stores
    conn = sqlite3.connect(source, isolation_level=None)
    try:
        for sql in ddl:
            conn.execute(sql)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    _assert_refused(source, output, match)


def test_an_empty_table_the_fork_does_not_have_is_dropped(stores):
    source, _, output = stores
    conn = sqlite3.connect(source, isolation_level=None)
    try:
        conn.execute("CREATE TABLE preview_markers (id TEXT)")
    finally:
        conn.close()
    report = _export(source, output)
    assert report["tables"]["preview_markers"]["disposition"] == "converted"
    assert not _rows(output, "SELECT 1 FROM sqlite_master WHERE name = 'preview_markers'")


def test_only_this_builds_settled_store_returns(tmp_path):
    fork = tmp_path / "fork" / "state.db"
    fork_1030_store(fork)
    _assert_refused(fork, tmp_path / "out" / "a.db", r"source is fenced at schema_version 1030")
    upstream = tmp_path / "upstream" / "state.db"
    upstream.parent.mkdir()
    SessionDB(db_path=upstream).close()
    _assert_refused(upstream, tmp_path / "out" / "b.db", r"source is upstream at schema_version 33")


# ── A conversion that loses or alters anything is measured and refused ──

_LOSSY = {
    "a row": ("DELETE FROM session_process_authorities WHERE session_id = 's3' AND session_generation = 1",
              r"the output differs", {"session_process_authorities":
                                      "1 source rows absent or changed; 5 source vs 4 output rows"}),
    "a value": ("UPDATE session_turn_leases SET holder = 'pid=999999:turn=fork-era' WHERE conversation_id = 'tg1'",
                r"the output differs", {"session_turn_leases":
                                        "1 source rows absent or changed; 1 source vs 1 output rows"}),
    "a storage class": ("UPDATE messages SET display_identity = 7.0 WHERE display_identity = 7",
                        r"the output differs", {"messages": "1 source rows absent or changed; 7 source vs 7 output rows"}),
    "a column": ("ALTER TABLE sessions ADD COLUMN runtime_generation INTEGER NOT NULL DEFAULT 0",
                 r"the output differs", {"sessions": "columns differ: ['runtime_generation']"}),
    "an index": ("DROP INDEX idx_sessions_source", r"the output differs",
                 {"(indexes)": {"missing": ["idx_sessions_source"], "unexpected": [], "rewritten": []}}),
    "a trigger body": ("DROP TRIGGER session_process_authority_events_append_only_delete; "
                       "CREATE TRIGGER session_process_authority_events_append_only_delete "
                       "BEFORE DELETE ON session_process_authority_events BEGIN SELECT 1; END",
                       r"the output differs", {"(triggers)": {"missing": [], "unexpected": [],
                                               "rewritten": ["session_process_authority_events_append_only_delete"]}}),
    "a fence": ("DROP TRIGGER turn_fence_session_process_authorities_delete",
                r"not exactly the fork's fenced 1030 store", None),
}


@pytest.mark.parametrize("name", list(_LOSSY))
def test_a_conversion_that_loses_or_alters_anything_refuses(stores, monkeypatch, name):
    source, _, output = stores
    sql, match, differences = _LOSSY[name]
    if name == "a storage class":
        _write_1033(source, "UPDATE messages SET display_identity = 7 WHERE id = (SELECT max(id) FROM messages)")
    convert = fork_return._convert

    def lossy(conn, plan):
        convert(conn, plan)
        conn.create_function("hermes_turn_fence_generation", 0, lambda: 1030)
        conn.executescript(sql)

    monkeypatch.setattr(fork_return, "_convert", lossy)
    report = _assert_refused(source, output, match)
    if differences is not None:
        assert report["loss_detected"] is True and report["differences"] == differences


def test_a_damaged_output_refuses(stores, monkeypatch):
    source, _, output = stores
    monkeypatch.setattr(fork_return, "_integrity", lambda conn: ["row 3 missing from index idx_messages_session"])
    _assert_refused(source, output, r"integrity_check")


def test_a_source_written_during_the_export_refuses(stores, monkeypatch):
    source, _, output = stores
    compare = fork_return._compare

    def concurrent_write(conn, plan, inventory):
        Path(f"{source}-wal").write_bytes(b"\0" * 32)  # a writer appended while the export read
        return compare(conn, plan, inventory)

    monkeypatch.setattr(fork_return, "_compare", concurrent_write)
    with pytest.raises(fork_return.ForkReturnRefused, match="the source changed while it was read"):
        _export(source, output)
    assert not output.exists()


def test_unusable_paths_refuse_before_reading(stores):
    source, _, output = stores
    with pytest.raises(fork_return.ForkReturnError, match="would be the source.s file"):
        fork_return.return_to_fork(source, Path(f"{source}-wal"))
    output.parent.mkdir()
    output.write_bytes(b"x")
    with pytest.raises(fork_return.ForkReturnError, match="refusing to overwrite"):
        fork_return.return_to_fork(source, output)
    Path(f"{source}-journal").write_bytes(b"x")
    with pytest.raises(fork_return.ForkReturnError, match="hot rollback journal"):
        fork_return.return_to_fork(source, output.parent / "other.db")
    assert fork_return.main([str(source), str(output.parent / "other.db")]) == 2


def test_this_builds_declared_schema_is_fully_accounted_for(tmp_path):
    """A fresh store of this build carries every table and column it declares: each one is either the fork's or
    has a disposition here, so an object added later refuses until it gets one."""
    fresh = tmp_path / "fresh" / "state.db"
    fresh.parent.mkdir()
    db = SessionDB(db_path=fresh)
    try:
        db.apply_telegram_topic_migration()
    finally:
        db.close()
    conn = sqlite3.connect(f"file:{fresh}?mode=ro", uri=True)
    try:
        plan = fork_return._plan(conn, fork_return._inventory(conn))
    finally:
        conn.close()
    assert [r for r in plan["refused"] if "schema_version" not in r] == []


# ── Paths: a conflict refuses before anything is written, and no existing file is ever touched ──

def _snapshot(root: Path) -> dict:
    """Every path under *root*: a directory, a link and its target, or a file's sha256."""
    entries = {}
    for directory, dirs, files in os.walk(root):
        for name in dirs + files:
            path = Path(directory) / name
            entries[str(path.relative_to(root))] = (
                ("link", os.readlink(path)) if path.is_symlink() else ("dir",) if path.is_dir() else ("file", _sha(path)))
    return entries


def _symlink(path: Path, target: Path) -> Path:
    path.symlink_to(target)
    return path


def _hardlink(path: Path, target: Path) -> Path:
    os.link(target, path)
    return path


_PATH_CONFLICTS = {
    "report is the source": (lambda t, s, o: {"report_path": s}, "refusing to overwrite"),
    "report is a symlink to the source": (lambda t, s, o: {"report_path": _symlink(t / "r.json", s)},
                                          "refusing to overwrite"),
    "report is a hardlink of the source": (lambda t, s, o: {"report_path": _hardlink(t / "r.json", s)},
                                           "refusing to overwrite"),
    "report is the output": (lambda t, s, o: {"report_path": o}, "would be the output's file"),
    "report is the output's WAL": (lambda t, s, o: {"report_path": Path(f"{o}-wal")}, "would be the output's file"),
    "report is the source's WAL": (lambda t, s, o: {"report_path": Path(f"{s}-wal")},
                                   "would be the source's file|refusing to overwrite"),
    "report is the output through a directory alias": (
        lambda t, s, o: {"report_path": _symlink(t / "alias", o.parent) / o.name}, "would be the output's file"),
    "report is the source's journal in another case": (
        lambda t, s, o: {"report_path": s.parent / f"{s.name.upper()}-JOURNAL"}, "would be the source's file"),
    "report is inside the source file": (lambda t, s, o: {"report_path": s / "r.json"}, "is not a directory"),
    "output is the source through a directory alias": (
        lambda t, s, o: {"output": _symlink(t / "alias", s.parent) / s.name}, "."),
    "output is inside the source file": (lambda t, s, o: {"output": s / "state.db"}, "is not a directory"),
    "output's directory does not exist": (lambda t, s, o: {"output": t / "missing" / "state.db"}, "does not exist"),
    "work directory is the source file": (lambda t, s, o: {"work_dir": s}, "is not a directory"),
}


@pytest.mark.parametrize("case", list(_PATH_CONFLICTS))
def test_a_path_conflict_refuses_before_anything_is_written(stores, tmp_path, case):
    source, _, output = stores
    output.parent.mkdir()
    make, match = _PATH_CONFLICTS[case]
    kwargs = {"output": output, **make(tmp_path, source, output)}
    before = _snapshot(tmp_path)
    with pytest.raises(fork_return.ForkReturnError, match=match):
        fork_return.return_to_fork(source, kwargs.pop("output"), **kwargs)
    assert _snapshot(tmp_path) == before


def test_scratch_never_takes_an_existing_path(stores, tmp_path):
    """A source stored where a predictable build copy would go, and user files where staging copies would go:
    the export runs and every one of them is left as it was."""
    source, _, output = stores
    output.parent.mkdir()
    planted = output.parent / f".{output.name}.return-build"
    planted.write_bytes(source.read_bytes())
    for name in (f".{output.name}.return-source", f".{output.name}.return-source-wal", "build.db", "source.db"):
        (output.parent / name).write_bytes(b"a user's file")
    before = _snapshot(tmp_path)

    report = fork_return.return_to_fork(planted, output)

    assert report["published"] and report["source_unchanged"] is True
    after = _snapshot(tmp_path)
    assert after.pop("out/state.db")[0] == after.pop("out/state.db.fork-return")[0] == "file" and after == before


@pytest.mark.parametrize("stage", ["_plan", "_convert", "_compare"])
def test_a_cancelled_export_removes_only_what_it_created(stores, tmp_path, monkeypatch, stage):
    source, _, output = stores
    crashed = _wal_bearing_copy(source, tmp_path / "crashed")  # so the staging copy exists too
    work = tmp_path / "work"
    work.mkdir()
    output.parent.mkdir()
    for directory in (work, output.parent):
        for name in (f".{output.name}.return-build", f".{output.name}.return-source"):
            (directory / name).write_bytes(b"a user's file")

    def cancel(*_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(fork_return, stage, cancel)
    before = _snapshot(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        fork_return.return_to_fork(crashed, output, work_dir=work, report_path=tmp_path / "report.json")
    after = _snapshot(tmp_path)
    assert after.pop("report.json")[0] == "file" and after == before
    report = json.loads((tmp_path / "report.json").read_text())
    assert not report["published"] and report["source_unchanged"] is True and "scratch_left" not in report


def test_a_terminated_export_removes_its_scratch(stores, tmp_path, monkeypatch):
    source, _, output = stores
    output.parent.mkdir()

    class Terminated(Exception):
        pass

    def unhandled(_signum, _frame):  # what SIGTERM does here if the export installs no handler of its own
        raise Terminated

    previous = signal.signal(signal.SIGTERM, unhandled)
    try:
        monkeypatch.setattr(fork_return, "_convert", lambda conn, plan: os.kill(os.getpid(), signal.SIGTERM))
        before = _snapshot(tmp_path)
        with pytest.raises(SystemExit) as caught:
            fork_return.main([str(source), str(output)])
        assert caught.value.code == 128 + signal.SIGTERM
        assert signal.getsignal(signal.SIGTERM) is unhandled
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert _snapshot(tmp_path) == before


def test_an_output_that_appears_during_the_export_is_never_replaced(stores, monkeypatch):
    source, _, output = stores
    compare = fork_return._compare

    def intruder(conn, plan, inventory):
        output.write_bytes(b"someone else's file")
        return compare(conn, plan, inventory)

    monkeypatch.setattr(fork_return, "_compare", intruder)
    with pytest.raises(fork_return.ForkReturnError, match="appeared during the export"):
        _export(source, output)
    assert output.read_bytes() == b"someone else's file"
    assert sorted(p.name for p in output.parent.iterdir()) == ["state.db"]


def test_source_unchanged_is_the_sources_bytes_read_again(stores, monkeypatch):
    """A write that keeps the source's size, inode and mtime is still a change; it is never reported unchanged."""
    source, _, output = stores
    compare = fork_return._compare

    def same_size_write(conn, plan, inventory):
        st = source.stat()
        with open(source, "r+b") as handle:  # the header's user_version: a write no stat() shows
            handle.seek(60)
            handle.write((1).to_bytes(4, "big"))
        os.utime(source, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert (source.stat().st_size, source.stat().st_ino, source.stat().st_mtime_ns) == (
            st.st_size, st.st_ino, st.st_mtime_ns)
        return compare(conn, plan, inventory)

    monkeypatch.setattr(fork_return, "_compare", same_size_write)
    with pytest.raises(fork_return.ForkReturnRefused, match="the source changed while it was read") as caught:
        _export(source, output)
    assert caught.value.report["source_unchanged"] is False and not output.exists()


def test_an_unreadable_source_is_never_reported_unchanged(stores, tmp_path, monkeypatch):
    source, _, output = stores
    read, calls = fork_return._source_state, []

    def unreadable_after_the_first_read(path):
        calls.append(path)
        if len(calls) > 1:
            raise PermissionError("Operation not permitted")
        return read(path)

    monkeypatch.setattr(fork_return, "_source_state", unreadable_after_the_first_read)
    with pytest.raises(fork_return.ForkReturnRefused, match="could not be read again to verify it"):
        _export(source, output, report_path=tmp_path / "report.json")
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["source_unchanged"] is False and report["source_check"].startswith("unverifiable")
    assert not report["published"] and not output.exists()


# ── Terminal admissions: input is data, matched to the kept transcript or refused ──

def _runtime_write(path: Path, act) -> None:
    db = SessionDB(db_path=path)
    try:
        act(db)
    finally:
        db.close()


def _set_outcome(db, request_id: str, outcome: str) -> None:
    db._execute_write(lambda conn: conn.execute(
        "UPDATE session_admissions SET outcome = ? WHERE request_id = ?", (outcome, request_id)))


_ONLY_IN_ITS_ADMISSION = {
    "cancelled before it ran": lambda db: run_input(db, "s3", "r9", "only copy zk41", transcript=False, cancel=True),
    "completed without its message": lambda db: run_input(db, "s3", "r9", "only copy zk41", transcript=False),
    "interrupted before its message": lambda db: run_input(db, "s3", "r9", "only copy zk41", transcript=False,
                                                           outcome="interrupted"),
    "left steering": lambda db: (run_input(db, "s3", "r9", "only copy zk41", transcript=True),
                                 _set_outcome(db, "r9", "steering")),
    "the second of two with one message": lambda db: (run_input(db, "s3", "r8", "same words", transcript=True),
                                                      run_input(db, "s3", "r9", "same words", transcript=False)),
    "cancelled, its words in an older message": lambda db: (
        db.append_message("s3", role="user", content="said before"),
        run_input(db, "s3", "r9", "said before", transcript=False, cancel=True)),
}


@pytest.mark.parametrize("case", list(_ONLY_IN_ITS_ADMISSION))
def test_input_that_exists_only_in_its_admission_refuses(stores, case):
    source, _, output = stores
    _runtime_write(source, _ONLY_IN_ITS_ADMISSION[case])
    report = _assert_refused(source, output, r"lossless return impossible: 1 session_admissions inputs exist only")
    assert report["tables"]["session_admissions"]["disposition"] == "refused"
    assert report["tables"]["session_admissions"]["inputs"]["lossless_return_impossible"] == 1


def test_inputs_each_matched_to_their_own_message_return(stores):
    source, _, output = stores
    _runtime_write(source, lambda db: (run_input(db, "s3", "r8", "same words", transcript=True),
                                       run_input(db, "s3", "r9", "same words", transcript=True, outcome="failed")))
    report = _export(source, output)
    assert report["published"] and report["tables"]["session_admissions"]["inputs"]["matched_to_messages"] == 3


def native_payload(monkeypatch, tmp_path, text: str, message_id, **event_fields) -> dict:
    """A native admission payload from the gateway's own builder (``snapshot_native``) for an A2A peer's message."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.run import GatewayRunner
    from gateway.session_envelope import snapshot_native
    from plugins.platforms.a2a.adapter import A2AAdapter

    for key in ("A2A_ALLOWED_USERS", "A2A_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "gateway-sessions"))
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    runner.adapters = {Platform("a2a"): adapter}
    source = adapter.build_source(chat_id="ctx-1", chat_name="a2a:peer", chat_type="dm", user_id="peer",
                                  user_name="peer")
    payload = snapshot_native(runner, MessageEvent(text=text, message_type=MessageType.TEXT, message_id=message_id,
                                                   source=source, **event_fields))
    assert set(payload) == {"text", "native_text_v1"} and payload["native_text_v1"]["event"]["message_id"] == message_id
    return payload


def api_payload(text, settings=None, **fields) -> dict:
    """An API admission payload as ``session_api_turn`` builds it (no settings unless given)."""
    return {"text": text, "api_turn_v1": {"history": None, "settings": settings or {}, **fields}}


# Finished inputs whose every key has a mapping, matched by the row the turn stored.
_MATCHED = {
    "api: exactly its text": lambda db: run_input(
        db, "s3", "r9", "an api question", transcript=True, payload=api_payload("an api question")),
    "api: a resumed unanswered turn adopts its failed attempt's row": lambda db: (
        run_input(db, "s3", "r8", "unanswered", transcript=True, outcome="failed", payload=api_payload("unanswered")),
        run_input(db, "s3", "r9", "unanswered", transcript=False,
                  payload=api_payload("unanswered", resume_unanswered=True))),
    "api: settings and relay metadata that carry nothing": lambda db: run_input(
        db, "s3", "r9", "an api question", transcript=True, payload=api_payload(
            "an api question", {"ephemeral_system_prompt": None, "model_options": {}}, relay_metadata={})),
}


@pytest.mark.parametrize("case", list(_MATCHED))
def test_a_finished_input_with_its_envelope_returns(stores, case):
    source, _, output = stores
    _runtime_write(source, _MATCHED[case])
    report = _export(source, output)
    inputs = report["tables"]["session_admissions"]["inputs"]
    assert report["committed"] and inputs["lossless_return_impossible"] == inputs["no_rule"] == 0
    assert inputs["matched_to_messages"] == len(_rows(source, "SELECT 1 FROM session_admissions"))
    assert set(inputs["by_kind"]) == {"text", "api_turn_v1"}


def test_an_api_input_in_no_row_refuses(stores):
    source, _, output = stores
    _runtime_write(source, lambda db: run_input(db, "s3", "r9", "an api question", transcript=False,
                                                payload=api_payload("an api question")))
    report = _assert_refused(source, output, r"lossless return impossible: 1 session_admissions inputs .*no kept "
                                             r"user message of its sessions is its input")
    assert report["tables"]["session_admissions"]["inputs"]["by_kind"]["api_turn_v1"]["lossless_return_impossible"] == 1


def test_an_api_turns_instructions_are_data_its_terminal_replay_reads(stores):
    """This build's terminal replay of a finished API turn projects settings.ephemeral_system_prompt as its
    ``instructions``; the fork has no place for it, so the export refuses rather than drop it."""
    from gateway.platforms.api_server_response_admissions import admitted_context
    from hermes_state_runtime import get_session_admission

    source, _, output = stores
    _runtime_write(source, lambda db: _api_turn(
        db, "responses:r9", "and now?", None, transcript=True,
        settings={"ephemeral_system_prompt": "answer as the night shift zx3"}))
    (admission_id,) = _rows(source, "SELECT admission_id FROM session_admissions WHERE request_id = 'responses:r9'")[0]
    db = SessionDB(db_path=source)
    try:
        row = get_session_admission(db, admission_id=admission_id)
    finally:
        db.close()
    assert row["status"] == "terminal"
    assert admitted_context((None, SimpleNamespace(session_id="s3"), row))["instructions"] == (
        "answer as the night shift zx3")
    report = _assert_refused(source, output, r"lossless return impossible: 1 session_admissions inputs .*"
                                             r"api_turn_v1.settings.ephemeral_system_prompt only its admission holds")
    assert report["tables"]["session_admissions"]["inputs"]["by_kind"]["api_turn_v1"]["lossless_return_impossible"] == 1


@pytest.mark.parametrize("metadata, match", [
    ({"trace": {"span": "zx4", "baggage": {"k": "v"}}}, r"no rule for api_turn_v1.relay_metadata.trace"),
    ({"client": "zx5"}, r"no rule for api_turn_v1.relay_metadata.client"),
], ids=["a nested subtree", "a scalar"])
def test_relay_metadata_has_no_mapping_and_refuses_at_its_path(stores, metadata, match):
    source, _, output = stores
    _runtime_write(source, lambda db: run_input(db, "s3", "r9", "q", transcript=True,
                                                payload=api_payload("q", relay_metadata=metadata)))
    report = _assert_refused(source, output, r"session_admissions rows no rule carries to the fork: .*" + match)
    assert report["tables"]["session_admissions"]["inputs"]["no_rule"] == 1


def test_a_native_turns_source_route_and_timestamp_have_no_place_in_the_fork(stores, monkeypatch, tmp_path):
    """Its user row is kept with its platform message id, but the sender, route and time it carried are on no kept
    row; this build's own defaults (is_bot, allow_gateway_control, reply_to_is_own_message) are not data."""
    source, _, output = stores
    _runtime_write(source, lambda db: run_input(
        db, "s3", "r9", "hi there", transcript=True, platform_message_id="m-7",
        payload=native_payload(monkeypatch, tmp_path, "hi there", "m-7")))
    report = _assert_refused(source, output, r"lossless return impossible: 1 session_admissions inputs .*\(native_text_v1,"
                                             r" completed; native_text_v1.event.message_id, native_text_v1.route, "
                                             r"native_text_v1.source.chat_id, .*native_text_v1.source.platform, .*"
                                             r"native_text_v1.timestamp only its admission holds\)")
    held = "; ".join(report["refused"])
    for default in ("source.is_bot", "event.allow_gateway_control", "event.reply_to_is_own_message", "source.thread_id"):
        assert f"native_text_v1.{default}" not in held


@pytest.mark.parametrize("value, data", [(True, False), (False, True)], ids=["its default", "not its default"])
def test_only_a_declared_default_carries_nothing(stores, monkeypatch, tmp_path, value, data):
    """``allow_gateway_control`` defaults to true: false is what the turn ran under, not an unset value."""
    source, _, output = stores
    _runtime_write(source, lambda db: run_input(
        db, "s3", "r9", "hi", transcript=True, payload=native_payload(
            monkeypatch, tmp_path, "hi", "m-7", allow_gateway_control=value)))
    report = _assert_refused(source, output, r"lossless return impossible")
    assert ("native_text_v1.event.allow_gateway_control" in "; ".join(report["refused"])) is data


def _with_unknown_envelope_key(payload: dict, key: str) -> dict:
    payload[key]["later"] = 1
    return payload


@pytest.mark.parametrize("act, match", [
    (lambda db, native: (run_input(db, "s3", "r9", "x", transcript=True), _set_outcome(db, "r9", "superseded")),
     r"session_admissions rows no rule carries to the fork: \w+ \(text\): outcome 'superseded'"),
    (lambda db, native: run_input(db, "s3", "r9", "op", transcript=True,
                                  payload={"text": "op", "local_operator_v1": {"profile_id": "p"}}),
     r"no rule carries to the fork: \w+ \(local_operator_v1\): payload \['local_operator_v1', 'text'\] has no rule"),
    (lambda db, native: run_input(db, "s3", "r9", "hi", transcript=True,
                                  payload=_with_unknown_envelope_key(native("hi", "m-7"), "native_text_v1")),
     r"\(native_text_v1\): no rule for native_text_v1.later"),
    (lambda db, native: run_input(db, "s3", "r9", "q", transcript=True,
                                  payload=_with_unknown_envelope_key(api_payload("q"), "api_turn_v1")),
     r"\(api_turn_v1\): no rule for api_turn_v1.later"),
    (lambda db, native: run_input(db, "s3", "r9", "parts", transcript=True,
                                  payload={"text": [{"type": "text", "text": "parts"}], "api_turn_v1": {
                                      "history": None, "settings": {}}}),
     r"\(api_turn_v1\): its text is a list"),
    (lambda db, native: run_input(db, "s3", "r9", "hi", transcript=True,
                                  payload={"text": "hi", "native_text_v1": {"route": "r"}}),
     r"\(native_text_v1\): payload \['native_text_v1', 'text'\] has no rule"),
], ids=["an unknown outcome", "an envelope with no rule", "an unknown native field", "an unknown api field",
        "multimodal api input", "a native envelope without its event"])
def test_an_admission_no_rule_carries_refuses(stores, monkeypatch, tmp_path, act, match):
    source, _, output = stores
    _runtime_write(source, lambda db: act(db, lambda text, mid: native_payload(monkeypatch, tmp_path, text, mid)))
    report = _assert_refused(source, output, match)
    assert report["tables"]["session_admissions"]["inputs"]["no_rule"] == 1


# ── Removed tables: every column and every row has a rule, or the export refuses ──

@pytest.mark.parametrize("sql, match", [
    ("ALTER TABLE worker_receipts ADD COLUMN unconvertible_payload TEXT; "
     "UPDATE worker_receipts SET unconvertible_payload = 'only here'",
     r"column worker_receipts.unconvertible_payload \(1 rows set\) has no representation in the fork"),
    ("ALTER TABLE session_admissions ADD COLUMN later TEXT",
     r"column session_admissions.later \(0 rows set\) has no representation in the fork"),
    ("ALTER TABLE runtime_epoch ADD COLUMN later INTEGER DEFAULT 3",
     r"column runtime_epoch.later \(0 rows set\) has no representation in the fork"),
    ("ALTER TABLE schema_version ADD COLUMN note TEXT", r"schema_version has columns \['note', 'version'\]"),
    ("INSERT INTO worker_receipts VALUES ('ghost', 1, 'p', '{}')", r"worker_receipts rows of no terminal worker: 1"),
    ("PRAGMA ignore_check_constraints = ON; UPDATE worker_executions SET kind = 'future'",
     r"worker_executions.kind holds values no rule carries to the fork: \{'future': 1\}"),
    ("ALTER TABLE worker_receipts DROP COLUMN payload_digest",
     r"worker_receipts lacks \['payload_digest'\]; its rows are not the shape this export reads"),
    ("""UPDATE worker_receipts SET result_json = '{"unknown_receipt_field": {"input": "only receipt z733"}}'""",
     r"worker_receipts answers no rule carries to the fork: x1/1: no rule for "
     r"worker_receipts.result_json.unknown_receipt_field"),
], ids=["a set unknown column", "an unset unknown column", "an unknown column with a default",
        "an unknown schema_version column", "a receipt of no worker", "an unknown value", "a missing column",
        "an unmapped key in a receipt's answer"])
def test_a_removed_tables_data_without_a_rule_refuses(stores, sql, match):
    source, _, output = stores
    conn = _connect_as(source, 1033)  # foreign keys off, as any raw writer may have left them
    try:
        conn.executescript(sql)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    _assert_refused(source, output, match)


# ── Every leaf of a finished payload: found in the kept rows, held nowhere but its admission, or refused ──

def _api_turn(db, request_id: str, text: str, history, settings=None, **kwargs) -> None:
    """An API turn as session_api_turn admits it, then settlement's own compaction (a Responses row keeps its
    history: its terminal replay rebuilds the previous_response_id snapshot from it)."""
    from gateway.session_api_media import compact_settled_api_payloads

    run_input(db, "s3", request_id, text, principal="api", payload=api_payload(text, settings, history=history),
              **kwargs)
    compact_settled_api_payloads(db)


def _history(*entries) -> list:
    return [{"role": role, "content": content} for role, content in entries]


_FIRST = (lambda db: _api_turn(db, "responses:r8", "first question", None, transcript=True))
_KEPT_CONVERSATION = _history(("user", "first question"), ("assistant", "answered: first question"))


def test_a_responses_turns_prior_input_no_kept_row_holds_refuses(stores):
    """The client sent prior turns in api_turn_v1.history that this store holds nowhere else; the turn's own text
    is a kept row. Matching the text alone would drop the history while reporting no loss."""
    source, _, output = stores
    _runtime_write(source, lambda db: _api_turn(db, "responses:r9", "and now?", _history(
        ("user", "unique prior input zq9"), ("assistant", "an answer given elsewhere zq9")), transcript=True))
    assert "unique prior input zq9" in _rows(source, "SELECT payload_json FROM session_admissions "
                                                     "WHERE request_id = 'responses:r9'")[0][0]
    report = _assert_refused(source, output, r"lossless return impossible: 1 session_admissions inputs .*"
                                             r"api_turn_v1.history\[0\] \(user\) is no kept message of its sessions")
    assert report["tables"]["session_admissions"]["inputs"]["by_kind"]["api_turn_v1"]["lossless_return_impossible"] == 1


def test_a_history_that_is_the_kept_conversation_before_it_returns(stores):
    source, _, output = stores
    _runtime_write(source, lambda db: (_FIRST(db), _api_turn(db, "responses:r9", "and now?", _KEPT_CONVERSATION,
                                                             transcript=True)))
    report = _export(source, output)
    assert report["published"] and report["tables"]["session_admissions"]["inputs"]["by_kind"]["api_turn_v1"] == {
        "matched": 2, "lossless_return_impossible": 0, "no_rule": 0}


@pytest.mark.parametrize("history, match", [
    (_KEPT_CONVERSATION[::-1], r"api_turn_v1.history\[1\] \(user\) is no kept message of its sessions before"),
    (_history(("user", "said after it")), r"api_turn_v1.history\[0\] \(user\) is no kept message of its sessions"),
], ids=["kept rows in another order", "a kept row written after its input"])
def test_a_history_out_of_its_order_refuses(stores, history, match):
    source, _, output = stores
    _runtime_write(source, lambda db: (_FIRST(db), _api_turn(db, "responses:r9", "and now?", history, transcript=True),
                                       db.append_message("s3", role="user", content="said after it")))
    _assert_refused(source, output, match)


def _nested(payload: dict, *path_and_value) -> dict:
    *path, key, value = path_and_value
    node = payload
    for name in path:
        node = node[name]
    node[key] = value
    return payload


_HELD_ONLY_BY_ITS_ADMISSION = {
    "native media": (lambda native: _nested(native("hi", "m-7"), "native_text_v1", "media",
                                            [{"path": "native-inputs/ab", "sha256": "ab"}]),
                     r"native_text_v1.media"),
    "native platform metadata": (lambda native: _nested(native("hi", "m-7"), "native_text_v1", "event", "metadata",
                                                        {"whatsapp_native_type": "locationMessage"}),
                                 r"native_text_v1.event.metadata"),
    "api media": (lambda native: _nested(api_payload("hi"), "api_turn_v1", "media", [{"sha256": "ab"}]),
                  r"api_turn_v1.media"),
}


@pytest.mark.parametrize("case", list(_HELD_ONLY_BY_ITS_ADMISSION))
def test_data_only_its_admission_holds_refuses(stores, monkeypatch, tmp_path, case):
    source, _, output = stores
    make, match = _HELD_ONLY_BY_ITS_ADMISSION[case]
    _runtime_write(source, lambda db: run_input(
        db, "s3", "r9", "hi", transcript=True, platform_message_id="m-7",
        payload=make(lambda text, mid: native_payload(monkeypatch, tmp_path, text, mid))))
    _assert_refused(source, output, r"lossless return impossible: 1 session_admissions inputs .*" + match
                    + r"(, [\w.]+)* only its admission holds")


@pytest.mark.parametrize("make, match", [
    (lambda native: _nested(native("hi", "m-7"), "native_text_v1", "event", "later", 1),
     r"no rule for native_text_v1.event.later"),
    (lambda native: _nested(native("hi", "m-7"), "native_text_v1", "source", "later", "x"),
     r"no rule for native_text_v1.source.later"),
    (lambda native: _nested(api_payload("hi"), "api_turn_v1", "settings", "later", 1),
     r"no rule for api_turn_v1.settings.later"),
    (lambda native: _nested(native("hi", "m-7"), "native_text_v1", "source", "a2a peer"),
     r"native_text_v1.source is not the shape its rule reads"),
    (lambda native: _nested(api_payload("hi"), "api_turn_v1", "resume_unanswered", "yes"),
     r"api_turn_v1.resume_unanswered is not the shape its rule reads"),
    (lambda native: _nested(api_payload("hi"), "api_turn_v1", "history", "hi"),
     r"api_turn_v1.history is not the shape its rule reads"),
    (lambda native: _nested(api_payload("hi"), "api_turn_v1", "history",
                            [{"role": "user", "content": "hi", "name": "x"}]),
     r"api_turn_v1.history is not the shape its rule reads"),
    (lambda native: _nested(api_payload("hi"), "api_turn_v1", "history",
                            [{"role": "user", "content": [{"type": "text"}]}]),
     r"api_turn_v1.history is not the shape its rule reads"),
], ids=["an unknown event field", "an unknown source field", "an unknown api setting", "a source not an object",
        "a resume flag not true", "a history not a list", "a history entry with more than role and content",
        "a multimodal history entry"])
def test_a_nested_leaf_no_rule_carries_refuses(stores, monkeypatch, tmp_path, make, match):
    source, _, output = stores
    _runtime_write(source, lambda db: run_input(
        db, "s3", "r9", "hi", transcript=True, platform_message_id="m-7",
        payload=make(lambda text, mid: native_payload(monkeypatch, tmp_path, text, mid))))
    report = _assert_refused(source, output, r"session_admissions rows no rule carries to the fork: .*" + match)
    assert report["tables"]["session_admissions"]["inputs"]["no_rule"] == 1


# ── Session labels: found under the same labels by this build's lookups and the fork's ──

# Every label this build's alias table (gateway.session_policy) resolves, and a platform outside it. The fork filters
# on sessions.source alone; this build lists ``gui`` and ``desktop`` as one conversation, so a session under either
# would leave or join a listing on return. The kit measures the fork side with the fork binary.
_LABELS_RETURNED = ("acp", "cli", "local", "oneshot", "telegram", "tool", "tui")
_LABELS_REFUSED = ("desktop", "gui")


def _labelled(db, session_id: str, label: str) -> None:
    db.create_session(session_id, source=label)
    db.append_message(session_id, role="user", content=f"made under {label}")


def test_every_label_of_the_alias_table_has_a_fixture():
    assert set(fork_return.lookup_labels()) <= set(_LABELS_RETURNED + _LABELS_REFUSED)


@pytest.mark.parametrize("label", _LABELS_RETURNED + _LABELS_REFUSED)
def test_a_session_label_returns_only_when_both_sides_find_it_under_the_same_labels(stores, label):
    source, _, output = stores
    _runtime_write(source, lambda db: _labelled(db, "s4", label))
    (created,) = _rows(source, "SELECT created_source FROM sessions WHERE id = 's4'")[0]
    this_build, fork = fork_return.label_lookups(label, created)
    if label in _LABELS_RETURNED:
        report = _export(source, output)
        assert report["committed"] and this_build == fork == [label]
        assert [(e["this_build_finds_under"], e["the_fork_finds_under"]) for e in report["labels"]
                if (e["source"], e["created_source"]) == (label, created)] == [([label], [label])]
    else:
        report = _assert_refused(source, output, rf"lossless return impossible: sessions with source='{label}', "
                                                 rf"created_source='{created}' are found under \['desktop', 'gui'\] "
                                                 rf"by this build's lookups and under \['{label}'\] by the fork's: "
                                                 rf"1 rows: s4")
        assert this_build == ["desktop", "gui"] and fork == [label]


def test_a_desktop_session_this_builds_gui_listing_finds_refuses(stores):
    """The counterexample: created through this build's SessionDB as desktop/desktop, its CLI reader finds it with
    ``list_sessions_rich(source='gui')``; the fork's equivalent query returns nothing."""
    from hermes_cli.gateway_session_store import SurfaceLabelReader

    source, _, output = stores
    _runtime_write(source, lambda db: _labelled(db, "d1", "desktop"))
    assert _rows(source, "SELECT source, created_source FROM sessions WHERE id = 'd1'") == [("desktop", "desktop")]
    db = SessionDB(db_path=source)
    try:
        assert [r["id"] for r in SurfaceLabelReader(db).list_sessions_rich(source="gui", limit=100)] == ["d1"]
    finally:
        db.close()
    _assert_refused(source, output, r"sessions with source='desktop', created_source='desktop' are found under "
                                    r"\['desktop', 'gui'\] by this build's lookups and under \['desktop'\]")


def test_label_lookups_are_what_this_builds_reader_finds(stores):
    """The module's account of this build's lookups is this build's reader: for one session per label, the labels
    whose listing and search find it are the ones label_lookups names."""
    from hermes_cli.gateway_session_store import SurfaceLabelReader

    source, _, _ = stores
    labels = sorted(set(_LABELS_RETURNED + _LABELS_REFUSED))
    _runtime_write(source, lambda db: [_labelled(db, f"probe-{label}", label) for label in labels])
    db = SessionDB(db_path=source)
    try:
        reader, finds = SurfaceLabelReader(db), {}
        wide = dict(limit=1000, include_children=True, include_hidden=True)
        for label in labels:
            for row in reader.list_sessions_rich(source=label, include_archived=True, **wide):
                finds.setdefault(row["id"], set()).add(label)
            searched = {r["id"] for r in reader.search_sessions(source=label, limit=1000)}
            assert {i for i, found in finds.items() if label in found and i.startswith("probe-")} == {
                i for i in searched if i.startswith("probe-")}
    finally:
        db.close()
    for label in labels:
        (created,) = _rows(source, f"SELECT created_source FROM sessions WHERE id = 'probe-{label}'")[0]
        assert sorted(finds[f"probe-{label}"]) == fork_return.label_lookups(label, created)[0], label


@pytest.mark.parametrize("source_label, created", [("local", "tui"), ("local", "unknown"), ("cli", "")])
def test_a_creation_label_other_than_its_source_refuses(stores, source_label, created):
    """The minimum safeguard: a creation label set and other than its source refuses, whatever the lookups say."""
    source, _, output = stores
    _runtime_write(source, lambda db: (db.create_session("s4", source=source_label),
                                       db.append_message("s4", role="user", content="made in the tui"),
                                       db._execute_write(lambda conn: conn.execute(
                                           "UPDATE sessions SET created_source = ? WHERE id = 's4'", (created,)))))
    report = _assert_refused(
        source, output, rf"lossless return impossible: sessions.created_source differs from source; the fork finds a "
                        rf"session by its source alone: 1 rows: id='s4', source='{source_label}', "
                        rf"created_source='{created}'")
    assert report["columns"]["sessions.created_source"]["condition_broken_rows"] == 1


def test_a_creation_label_equal_to_its_source_or_unset_returns(stores):
    source, _, output = stores
    labels = dict(_rows(source, "SELECT id, created_source FROM sessions"))
    assert labels["s2"] == "tui" and labels["s1"] is None  # this build's own session, and a fork-era one
    report = _export(source, output)
    assert report["published"] and report["columns"]["sessions.created_source"]["condition_broken_rows"] == 0


def test_a_message_uid_shared_by_two_rows_refuses(stores):
    """A compaction's carried tail keeps its original's uid: this build reads the two rows as one message."""
    source, _, output = stores
    db = SessionDB(db_path=source)
    try:
        db.append_message("s3", role="user", content="question one")
        db.append_message("s3", role="assistant", content="REPLY-ONE")
        tail = db.get_messages_as_conversation("s3")[-1]
        db.archive_and_compact("s3", [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}, tail], tail_count=1)
    finally:
        db.close()
    _assert_refused(source, output, r"messages.message_uid is shared by other rows.*: 2 rows: id=\d+, session_id='s3'")


def test_rules_written_for_other_schemas_refuse(stores, monkeypatch):
    source, _, output = stores
    monkeypatch.setitem(fork_return._RULES, "source_schema_version", 1034)
    _assert_refused(source, output, r"these conversion rules are for schema 1034 -> 1030 \(fork 420c58a13e\); this "
                                    r"build stores 1033")


# ── The verify/publish boundary: a change found at any point is a failure, and its output is taken back ──

def _after_publication(monkeypatch, output: Path, act) -> None:
    """Run *act* once, when the export removes its scratch with the output already published: the steps between
    publication and the export's end."""
    remove, fired = fork_return._remove_scratch, []

    def removing(directory):
        left = remove(directory)
        if output.exists() and not fired:
            fired.append(act())
        return left

    monkeypatch.setattr(fork_return, "_remove_scratch", removing)


def _append_to(source: Path, text: str) -> bool:
    db = SessionDB(db_path=source)
    try:
        db.append_message("s1", role="user", content=text)
    finally:
        db.close()
    return True


def _fail(source, output, report_path, code: int = 2) -> dict:
    assert fork_return.main([str(source), str(output), "--report", str(report_path)]) == code
    report = json.loads(report_path.read_text())
    assert report["succeeded"] is False
    return report


def test_a_source_written_after_publication_fails_and_takes_its_output_back(stores, monkeypatch, tmp_path):
    source, _, output = stores
    output.parent.mkdir()
    _after_publication(monkeypatch, output, lambda: _append_to(source, "committed during the export q77"))
    report = _fail(source, output, tmp_path / "report.json")
    assert not output.exists() and report["published"] is False and report["source_unchanged"] is False
    assert report["retraction"].startswith("retracted") and "the source changed while it was read" in report["refused"]
    assert _rows(source, "SELECT count(*) FROM messages WHERE content = 'committed during the export q77'") == [(1,)]
    assert not list(output.parent.iterdir())


def test_a_source_written_before_publication_publishes_nothing(stores, monkeypatch, tmp_path):
    """A change during the staging copy's removal, the last step before the source is read again for publication."""
    source, _, output = stores
    crashed = _wal_bearing_copy(source, tmp_path / "crashed")
    work = tmp_path / "work"
    work.mkdir()
    output.parent.mkdir()
    remove, fired = fork_return._remove_scratch, []

    def removing(directory):
        left = remove(directory)
        if directory.parent == work and not fired:
            fired.append(_append_to(crashed, "committed while staging q78"))
        return left

    monkeypatch.setattr(fork_return, "_remove_scratch", removing)
    assert fork_return.main([str(crashed), str(output), "--work-dir", str(work),
                             "--report", str(tmp_path / "report.json")]) == 2
    report = json.loads((tmp_path / "report.json").read_text())
    assert fired and not output.exists() and report["published"] is False and "retraction" not in report
    assert report["refused"] == ["the source changed while it was read"] and not list(output.parent.iterdir())


def test_an_output_replaced_after_publication_is_not_a_success_and_is_left_alone(stores, monkeypatch, tmp_path):
    source, _, output = stores
    output.parent.mkdir()

    def replace():
        output.unlink()
        output.write_bytes(b"another process's file")
        return True

    _after_publication(monkeypatch, output, replace)
    report = _fail(source, output, tmp_path / "report.json")
    assert output.read_bytes() == b"another process's file" and report["published"] is False
    assert any("was replaced or removed by another process" in r for r in report["refused"])


def test_an_output_written_in_place_after_publication_is_not_committed(stores, monkeypatch, tmp_path):
    """Same file, other bytes: another process wrote into the published output before the commit point."""
    source, _, output = stores
    output.parent.mkdir()

    def write_into():
        with open(output, "r+b") as handle:
            handle.seek(60)
            handle.write((7).to_bytes(4, "big"))
        return True

    _after_publication(monkeypatch, output, write_into)
    report = _fail(source, output, tmp_path / "report.json")
    assert any("was written by another process after it was published" in r for r in report["refused"])
    assert not output.exists() and not Path(f"{output}.fork-return").exists() and report["committed"] is False


def test_a_source_written_while_the_marker_is_staged_is_not_committed(stores, monkeypatch, tmp_path):
    source, _, output = stores
    output.parent.mkdir()
    stage = fork_return._stage_marker

    def staging(out, record):
        _append_to(source, "committed while the marker was staged q80")
        return stage(out, record)

    monkeypatch.setattr(fork_return, "_stage_marker", staging)
    report = _fail(source, output, tmp_path / "report.json")
    assert "the source changed while it was read" in report["refused"] and report["committed"] is False
    assert not output.exists() and not Path(f"{output}.fork-return").exists()


def test_an_output_written_through_its_wal_after_publication_is_not_committed(stores, monkeypatch, tmp_path):
    """Another connection commits into the published output's WAL and stays open: its main file is unchanged."""
    source, _, output = stores
    output.parent.mkdir()
    held, wal = [], []

    def write_through_wal():
        held.append(_connect_as(output, 1030))
        held[0].execute("INSERT INTO state_meta (key, value) VALUES ('written-after-publication', '1')")
        wal.append(Path(f"{output}-wal").stat().st_size)
        return True

    _after_publication(monkeypatch, output, write_through_wal)
    try:
        report = _fail(source, output, tmp_path / "report.json")
    finally:
        for conn in held:
            conn.close()
    assert wal[0] > 0 and any("-wal exists: the output" in r and "was opened by another process" in r
                              for r in report["refused"])
    assert not output.exists() and not Path(f"{output}.fork-return").exists() and report["committed"] is False


def test_a_retraction_never_moves_a_file_it_did_not_create(stores, monkeypatch, tmp_path):
    """The source changed after publication, and before the output is taken back another process replaced it."""
    source, _, output = stores
    output.parent.mkdir()
    _after_publication(monkeypatch, output, lambda: _append_to(source, "committed during the export q79"))
    retract, rename = fork_return._retract, os.rename

    def replaced_first(out, identity, report):
        out.unlink()
        out.write_bytes(b"another process's file")
        monkeypatch.setattr(os, "rename", lambda *a: pytest.fail(f"moved a file this run did not create: {a}"))
        try:
            return retract(out, identity, report)
        finally:
            monkeypatch.setattr(os, "rename", rename)

    monkeypatch.setattr(fork_return, "_retract", replaced_first)
    report = _fail(source, output, tmp_path / "report.json")
    assert output.read_bytes() == b"another process's file" and report["published"] is False
    assert "left untouched" in report["retraction"] and "residual" not in report


def test_a_file_swapped_in_while_retracting_is_put_back(stores, monkeypatch, tmp_path):
    """Replaced between the retraction's check and its move: what was moved is not this run's, so it goes back."""
    source, _, output = stores
    output.parent.mkdir()
    _after_publication(monkeypatch, output, lambda: _append_to(source, "committed during the export q80"))
    rename = os.rename

    def swapped_then_moved(src, dst):
        if Path(src) == output:
            output.unlink()
            output.write_bytes(b"another process's file")
        return rename(src, dst)

    monkeypatch.setattr(os, "rename", swapped_then_moved)
    report = _fail(source, output, tmp_path / "report.json")
    assert output.read_bytes() == b"another process's file" and report["published"] is False
    assert "another process's file; it was left in place" in report["retraction"]
    assert sorted(p.name for p in output.parent.iterdir()) == ["state.db"]


def test_an_output_that_cannot_be_taken_back_is_named_and_never_a_success(stores, monkeypatch, tmp_path):
    source, _, output = stores
    output.parent.mkdir()

    def write_then_block_moves():
        _append_to(source, "committed during the export q81")

        def refused(*_args):
            raise PermissionError("Operation not permitted")

        monkeypatch.setattr(os, "rename", refused)
        return True

    _after_publication(monkeypatch, output, write_then_block_moves)
    report = _fail(source, output, tmp_path / "report.json", code=3)
    assert output.exists() and report["published"] is True and report["residual"]["path"] == str(output)
    assert "not a verified return of the source" in report["residual"]["state"]
    assert sorted(p.name for p in output.parent.iterdir()) == ["state.db"]


def test_a_report_that_cannot_be_written_after_the_commit_leaves_the_committed_return(stores, monkeypatch, tmp_path):
    """The report is written after the commit point and is not the verdict: the committed output stays, the other
    process's file stays, and the exit status says the account is missing."""
    source, _, output = stores
    output.parent.mkdir()
    report_path = tmp_path / "report.json"
    _after_publication(monkeypatch, output, lambda: report_path.write_bytes(b"another process's report"))
    assert fork_return.main([str(source), str(output), "--report", str(report_path)]) == 4
    assert report_path.read_bytes() == b"another process's report"
    assert fork_return.verify_return(output)["sha256"] == _sha(output)


# ── The commit point: the marker's link; an interruption before it is never a success, one after it takes nothing back ──

_INTERRUPT = '''
import os, signal, sys
import hermes_state_fork_return as fork_return

point, signum = sys.argv[1], int(sys.argv[2])
marker = str(sys.argv[4]) + getattr(fork_return, "MARKER_SUFFIX", ".fork-return")
link, write_report = os.link, fork_return._write_report


def interrupt():
    os.kill(os.getpid(), signum)  # SIGKILL ends the process here; SIGTERM runs the export's own handler
    for _ in range(10 ** 8):
        pass
    raise AssertionError("the signal was not delivered")


def linking(src, dst, *args, **kwargs):
    if str(dst) == marker and point == "before the commit":
        interrupt()
    link(src, dst, *args, **kwargs)
    if str(dst) == marker and point == "after the commit":
        interrupt()


def reporting(path, report):
    write_report(path, report)
    if point == "after the report":
        interrupt()


os.link, fork_return._write_report = linking, reporting
sys.exit(fork_return.main(sys.argv[3:]))
'''


def _interrupted(tmp_path, source, output, point: str, signum: int) -> tuple:
    """The CLI in its own process, *signum* delivered at *point*: ``(exit status, report or None)``."""
    driver, report_path = tmp_path / "interrupt.py", tmp_path / "report.json"
    driver.write_text(_INTERRUPT)
    repo = Path(fork_return.__file__).resolve().parent
    done = subprocess.run([sys.executable, str(driver), point, str(int(signum)), str(source), str(output),
                           "--report", str(report_path)], cwd=repo, env={**os.environ, "PYTHONPATH": str(repo)},
                          capture_output=True, text=True, timeout=300, check=False)
    return done.returncode, json.loads(report_path.read_text()) if report_path.exists() else None


def _not_a_return(output: Path) -> None:
    assert not Path(f"{output}.fork-return").exists()
    with pytest.raises(fork_return.ForkReturnError, match="no commit marker"):
        fork_return.verify_return(output)
    assert fork_return.main(["--verify", str(output)]) == 2


def _a_return(output: Path) -> None:
    assert fork_return.verify_return(output)["sha256"] == _sha(output)
    assert fork_return.main(["--verify", str(output)]) == 0


def test_a_cancel_before_the_commit_point_takes_the_output_back(stores, tmp_path):
    source, _, output = stores
    output.parent.mkdir()
    code, report = _interrupted(tmp_path, source, output, "before the commit", signal.SIGTERM)
    assert code == 128 + signal.SIGTERM and not output.exists()
    assert report["published"] is report["committed"] is report["succeeded"] is False
    assert report["retraction"].startswith("retracted")
    _not_a_return(output)
    assert not list(output.parent.iterdir())


def test_a_kill_before_the_commit_point_leaves_no_return(stores, tmp_path):
    """SIGKILL runs no handler: the published output stays, without its marker, and no report is written. The
    consumer's rule reads that as no return."""
    source, _, output = stores
    output.parent.mkdir()
    code, report = _interrupted(tmp_path, source, output, "before the commit", signal.SIGKILL)
    assert code == -signal.SIGKILL and output.exists() and report is None
    _not_a_return(output)


@pytest.mark.parametrize("point", ["after the commit", "after the report"])
def test_a_cancel_after_the_commit_point_takes_nothing_back(stores, tmp_path, point):
    """The review's counterexample: a SIGTERM right after the success report was written took the output back while
    the report still said succeeded. Past the marker's link nothing is taken back, and the report says so."""
    source, _, output = stores
    output.parent.mkdir()
    code, report = _interrupted(tmp_path, source, output, point, signal.SIGTERM)
    assert code == 128 + signal.SIGTERM and output.exists()
    assert report["published"] is report["committed"] is report["succeeded"] is True and "retraction" not in report
    assert report.get("after_commit", "SystemExit").startswith("SystemExit")
    _a_return(output)


def test_a_kill_after_the_commit_point_is_a_return_without_its_report(stores, tmp_path):
    source, _, output = stores
    output.parent.mkdir()
    code, report = _interrupted(tmp_path, source, output, "after the commit", signal.SIGKILL)
    assert code == -signal.SIGKILL and report is None
    _a_return(output)


@pytest.mark.parametrize("change, match", [
    (lambda output: output.write_bytes(output.read_bytes() + b"\0"), "is not the file its commit marker names"),
    (lambda output: Path(f"{output}-journal").write_bytes(b"x"), "has been opened since it was committed"),
    (lambda output: Path(f"{output}.fork-return").write_text("{}"), "is not this tool's record"),
    (lambda output: Path(f"{output}.fork-return").unlink(), "no commit marker"),
], ids=["other bytes", "a journal beside it", "another record", "no marker"])
def test_only_the_bytes_its_marker_names_verify(stores, change, match):
    source, _, output = stores
    _export(source, output)
    _a_return(output)
    change(output)
    with pytest.raises(fork_return.ForkReturnError, match=match):
        fork_return.verify_return(output)
    assert fork_return.main(["--verify", str(output)]) == 2


def test_a_marker_that_appears_during_the_export_is_never_replaced(stores, monkeypatch):
    source, _, output = stores
    marker = Path(f"{output}.fork-return")
    _after_publication(monkeypatch, output, lambda: marker.write_text("another process's marker"))
    with pytest.raises(fork_return.ForkReturnError, match="refusing to overwrite .*fork-return"):
        _export(source, output)
    assert marker.read_text() == "another process's marker" and not output.exists()
