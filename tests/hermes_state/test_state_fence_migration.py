"""The live fork's fenced 1030 state.db migrates to this build's 1033 in one transaction.

A writable SessionDB open decodes the fork store to logical schema 30, takes its fences off, brings
the declared schema in (upstream 31-32 tables), runs this build's data migrations above 30 only,
fences every governed table at 1033 and stamps 1033, under the profile's ``gateway.lock`` lease.
Every connection that writes the store registers the fence UDF and is serialized per connection,
which is what keeps a fence callback from deadlocking a shared connection (GIL vs. db mutex).

Fixtures are synthesized here; none of them is a copy of a real store.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import hermes_state_fence as fence
from hermes_state import SessionDB
from tests.hermes_state.test_state_lineage_admission import live_fence_trigger_sql

REPO = Path(__file__).resolve().parents[2]
OPERATIONS = ("INSERT", "UPDATE", "DELETE")
FORK_GOVERNED = fence.CORE_GOVERNED_TABLES + ("session_process_authorities",)
TARGET_TABLES = ("runtime_epoch", "session_admissions", "worker_executions", "worker_receipts")
ALL_GOVERNED = FORK_GOVERNED + TARGET_TABLES
_JOIN_S = 60.0


def _rows(conn) -> dict:
    return {
        "sessions": [tuple(r) for r in conn.execute("SELECT id, source, parent_session_id FROM sessions ORDER BY id")],
        "messages": [tuple(r) for r in conn.execute("SELECT id, session_id, role, content FROM messages ORDER BY id")],
    }


def fork_1030_store(path: Path, *, legacy_shapes: bool = False) -> dict:
    """This build's real tables and FTS rows, minus the tables upstream schema 31-32 added and the
    message_uid backfill, plus a fork session-process table, stamped 1030 and fenced at 1030 with the
    fork's exact trigger text. ``legacy_shapes`` adds what _init_schema would rebuild or retire: the
    pre-scope gateway_routing and pre-task session_model_usage keys, and a retired CJK index whose
    trigger fails every write to messages."""
    db = SessionDB(db_path=path)
    try:
        db.create_session("s1", source="cli")
        db.append_message("s1", role="user", content="hello fork store")
        db.append_message("s1", role="assistant", content="the needle is here")
        db.create_session("c1", source="subagent", parent_session_id="s1")
        db.append_message("c1", role="user", content="child work")
    finally:
        db.close()
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.execute("UPDATE sessions SET model_config = '{}' WHERE id = 'c1'")  # the row v33 re-tags
        conn.execute("DELETE FROM state_meta WHERE key = 'message_uid_backfill'")
        conn.execute("UPDATE messages SET message_uid = NULL")
        if legacy_shapes:
            _add_legacy_shapes(conn)
        for table in reversed(TARGET_TABLES):
            conn.execute(f"DROP TABLE {table}")
        conn.execute("CREATE TABLE session_process_authorities (session_id TEXT PRIMARY KEY)")
        conn.execute("UPDATE schema_version SET version = 1030")
        for table in FORK_GOVERNED:
            for operation in OPERATIONS:
                conn.execute(live_fence_trigger_sql(table, operation, 1030))
        rows = _rows(conn)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    assert path.read_bytes()[18:20] == b"\x02\x02" and not Path(f"{path}-wal").exists()
    lineage = fence.probe_store_lineage(path)
    assert (lineage.lineage, lineage.stored, set(lineage.fence_literals)) == ("fenced", 1030, {1030})
    return rows


def _add_legacy_shapes(conn) -> None:
    conn.execute("DROP TABLE gateway_routing")
    conn.execute("CREATE TABLE gateway_routing (session_key TEXT PRIMARY KEY, entry_json TEXT NOT NULL, "
                 "updated_at REAL NOT NULL)")
    conn.execute("INSERT INTO gateway_routing VALUES ('route-1', '{}', 1.0)")
    usage = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'session_model_usage'").fetchone()[0]
    assert "billing_mode, task)" in usage
    conn.execute("DROP TABLE session_model_usage")
    conn.execute(usage.replace("billing_mode, task)", "billing_mode)"))
    conn.execute("INSERT INTO session_model_usage (session_id, model) VALUES ('s1', 'm')")
    conn.execute("CREATE TABLE messages_fts_cjk (content)")
    conn.execute("CREATE TRIGGER messages_fts_cjk_update AFTER UPDATE ON messages "
                 "BEGIN SELECT RAISE(ABORT, 'cjk tokenizer unavailable'); END")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fence_sql(conn) -> dict:
    return dict(conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'turn_fence_%'").fetchall())


def _stamp(conn) -> int:
    return conn.execute("SELECT version FROM schema_version").fetchone()[0]


@pytest.fixture
def fork_store(tmp_path):
    path = tmp_path / "home" / "state.db"
    return path, fork_1030_store(path)


@pytest.fixture
def migrated(fork_store):
    path, _ = fork_store
    SessionDB(db_path=path).close()
    return path


# ── Migration ──

def test_fork_1030_store_migrates_to_1033_keeping_every_row(fork_store, monkeypatch):
    path, before = fork_store
    gates = []
    real_migrations = SessionDB._run_data_migrations

    def recording(self, cursor, current_version, fts5_available):
        gates.append(current_version)
        return real_migrations(self, cursor, current_version, fts5_available)

    monkeypatch.setattr(SessionDB, "_run_data_migrations", recording)
    db = SessionDB(db_path=path)
    try:
        conn = db._conn
        assert _stamp(conn) == fence.STORED_SCHEMA_VERSION == 1033
        assert _fence_sql(conn) == {
            f"turn_fence_{table}_{op.lower()}": live_fence_trigger_sql(table, op, 1033)
            for table in ALL_GOVERNED for op in OPERATIONS
        }
        assert _rows(conn) == before
        assert [hit["session_id"] for hit in db.search_messages("needle")] == ["s1"]
        child = conn.execute("SELECT json_extract(model_config, '$._delegate_from') FROM sessions "
                             "WHERE id = 'c1'").fetchone()[0]
        assert child == "s1"  # the v33 data migration ran
    finally:
        db.close()
    # Decoded to logical 30, so only the chain's gates above 30 fire; the open's own pass then sees 1033.
    assert gates == [30, 1033]
    assert fence.probe_store_lineage(path).settled


def test_table_heals_and_the_retired_cjk_index_run_inside_the_migration_so_every_fence_survives(tmp_path):
    path = tmp_path / "home" / "state.db"
    before = fork_1030_store(path, legacy_shapes=True)
    db = SessionDB(db_path=path)
    try:
        conn = db._conn
        assert set(_fence_sql(conn)) == {f"turn_fence_{t}_{op.lower()}" for t in ALL_GOVERNED for op in OPERATIONS}
        assert fence.decode_store_lineage(conn.cursor()).settled
        assert db._live_pk_columns(conn.cursor(), "gateway_routing") == ["scope", "session_key"]
        assert "task" in db._live_pk_columns(conn.cursor(), "session_model_usage")
        assert [tuple(r) for r in conn.execute("SELECT scope, session_key FROM gateway_routing")] == [("", "route-1")]
        assert [tuple(r) for r in conn.execute("SELECT session_id, model FROM session_model_usage")] == [("s1", "m")]
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE message_uid IS NULL").fetchone()[0] == 0
        assert _rows(conn) == before
    finally:
        db.close()


def test_a_stamp_below_1033_under_this_builds_fences_is_refused_unchanged(migrated):
    conn = sqlite3.connect(migrated, isolation_level=None)
    conn.execute("UPDATE schema_version SET version = 1032")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    sha = _sha(migrated)
    with pytest.raises(fence.IncompatibleSchemaError) as caught:
        SessionDB(db_path=migrated)
    assert caught.value.cause == "FENCE_GENERATION_MISMATCH" and _sha(migrated) == sha


def test_migrating_a_store_a_sibling_already_migrated_runs_nothing(migrated):
    db = SessionDB(db_path=migrated)
    try:
        cookie = db._conn.execute("PRAGMA schema_version").fetchone()[0]
        assert fence.migrate_fenced_store(db, db._conn.cursor()) is False
        assert db._conn.execute("PRAGMA schema_version").fetchone()[0] == cookie
    finally:
        db.close()


def test_a_write_after_migration_passes_the_fence_and_a_bare_connection_is_refused(migrated):
    db = SessionDB(db_path=migrated)
    try:
        db.create_session("s2", source="cli")
        db.append_message("s2", role="user", content="written after the migration")
        assert db.get_session("s2") is not None
    finally:
        db.close()

    bare = sqlite3.connect(migrated)
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such function: hermes_turn_fence_generation"):
            bare.execute("INSERT INTO sessions (id, source, started_at) VALUES ('s3', 'cli', 1.0)")
        bare.create_function(fence.TURN_FENCE_FUNCTION, 0, lambda: 1030)  # the fork's generation
        with pytest.raises(sqlite3.DatabaseError, match="state DB generation incompatible"):
            bare.execute("INSERT INTO runtime_epoch (singleton, epoch, instance_id) VALUES (1, 1, 'x')")
    finally:
        bare.close()


def _fail_data_migrations(monkeypatch):
    def boom(self, cursor, current_version, fts5_available):
        raise RuntimeError("injected mid-migration")
    monkeypatch.setattr(SessionDB, "_run_data_migrations", boom)


def _fail_refencing(monkeypatch):
    real = fence.turn_fence_trigger_definitions

    def partial(tables, **kwargs):
        definitions = real(tables, **kwargs)
        definitions["turn_fence_injected"] = "SELECT injected_mid_migration()"  # after the real fences
        return definitions
    monkeypatch.setattr(fence, "turn_fence_trigger_definitions", partial)


def _miss_one_fence(monkeypatch):
    real = fence.turn_fence_trigger_definitions

    def missing(tables, **kwargs):
        definitions = real(tables, **kwargs)
        del definitions["turn_fence_worker_receipts_delete"]
        return definitions
    monkeypatch.setattr(fence, "turn_fence_trigger_definitions", missing)


@pytest.mark.parametrize("inject", [_fail_data_migrations, _fail_refencing, _miss_one_fence],
                         ids=["data_migrations", "refencing", "fence_missing"])
def test_an_error_mid_migration_leaves_the_file_byte_identical(fork_store, monkeypatch, inject):
    path, before = fork_store
    sha = _sha(path)
    with monkeypatch.context() as patched:
        inject(patched)
        with pytest.raises((RuntimeError, sqlite3.DatabaseError), match="injected|did not leave exactly"):
            SessionDB(db_path=path)

    assert _sha(path) == sha
    lineage = fence.probe_store_lineage(path)
    assert lineage.migratable and lineage.stored == 1030  # stamp and fork fences untouched
    db = SessionDB(db_path=path)  # and the next open migrates it whole
    try:
        assert _stamp(db._conn) == 1033 and _rows(db._conn) == before
    finally:
        db.close()


def test_the_1033_store_reopens_without_fence_ddl_or_lease(migrated, monkeypatch):
    statements, leases = [], []
    real_open, real_lease = SessionDB._open_writer_conn, fence._runtime_lease

    def traced(self):
        conn = real_open(self)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(SessionDB, "_open_writer_conn", traced)
    monkeypatch.setattr(fence, "_runtime_lease", lambda home: leases.append(home) or real_lease(home))
    probe = sqlite3.connect(migrated)
    fences_before = _fence_sql(probe)
    probe.close()

    db = SessionDB(db_path=migrated)
    try:
        assert _stamp(db._conn) == 1033 and _fence_sql(db._conn) == fences_before
    finally:
        db.close()
    fence_ddl = re.compile(r"\b(CREATE|DROP)\s+TRIGGER\b[^;]*turn_fence_", re.IGNORECASE)
    assert statements and not [s for s in statements if fence_ddl.search(s)]
    assert not [s for s in statements if "UPDATE SCHEMA_VERSION" in s.upper()]
    assert leases == []


def test_read_only_open_serves_the_migrated_store(migrated):
    reader = SessionDB(db_path=migrated, read_only=True)
    try:
        assert isinstance(reader._conn, fence.SerializedConnectionMixin)
        assert reader._conn.execute("SELECT hermes_turn_fence_generation()").fetchone()[0] == 1033
        assert [hit["session_id"] for hit in reader.search_messages("needle")] == ["s1"]
    finally:
        reader.close()


# ── Concurrency ──

def test_two_openers_at_once_migrate_exactly_once(fork_store, monkeypatch):
    from gateway import runtime_ownership

    path, before = fork_store
    results, errors = [], []
    contended = threading.Event()
    real_migrate, real_owner = fence._migrate_locked, runtime_ownership.live_runtime_owner

    def migrate_after_contention(db, cursor):
        contended.wait(timeout=10)  # the other opener has met this one's lease
        migrated = real_migrate(db, cursor)
        results.append(migrated)
        return migrated

    def noting_owner(home):
        contended.set()
        return real_owner(home)

    monkeypatch.setattr(fence, "_migrate_locked", migrate_after_contention)
    monkeypatch.setattr(runtime_ownership, "live_runtime_owner", noting_owner)
    start = threading.Barrier(2)

    def opener():
        try:
            start.wait(timeout=_JOIN_S)
            SessionDB(db_path=path).close()
        except BaseException as exc:  # reported below
            errors.append(exc)

    threads = [threading.Thread(target=opener, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=_JOIN_S)
    assert not [t for t in threads if t.is_alive()], "an opener hung"
    assert errors == []
    # The sibling met the lease, then found the store settled and never entered the migration.
    assert contended.is_set() and results == [True]
    conn = sqlite3.connect(path)
    try:
        assert _stamp(conn) == 1033 and _rows(conn) == before
    finally:
        conn.close()


_CHILD_OPENER = """
import sys
from pathlib import Path
import hermes_state_fence as fence
real = fence._migrate_locked
def spy(db, cursor):
    migrated = real(db, cursor)
    print("migrated" if migrated else "settled", flush=True)
    return migrated
fence._migrate_locked = spy
from hermes_state import SessionDB
print("ready", flush=True)
sys.stdin.readline()
SessionDB(db_path=Path(sys.argv[1])).close()
print("opened", flush=True)
"""


def test_two_processes_opening_at_once_migrate_exactly_once(fork_store):
    path, before = fork_store
    children = [subprocess.Popen([sys.executable, "-c", _CHILD_OPENER, str(path)], cwd=REPO, text=True,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                for _ in range(2)]
    try:
        assert [child.stdout.readline().strip() for child in children] == ["ready", "ready"]
        for child in children:
            child.stdin.write("go\n")
            child.stdin.flush()
        outputs = [child.communicate(timeout=120) for child in children]
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
    assert [child.returncode for child in children] == [0, 0], [err for _, err in outputs]
    lines = [line for out, _ in outputs for line in out.split()]
    assert lines.count("migrated") == 1 and "settled" not in lines and lines.count("opened") == 2
    conn = sqlite3.connect(path)
    try:
        assert _stamp(conn) == 1033 and _rows(conn) == before
    finally:
        conn.close()


# ── The admission lease ──

_LOCK_HOLDER = """
import sys
from pathlib import Path
from gateway.runtime_ownership import ProfileOwnership
owner = ProfileOwnership()
owner.reserve([Path(sys.argv[1])])
print("held", flush=True)
sys.stdin.read()
"""


def test_a_gateway_holding_the_profile_lock_blocks_the_migration_with_nothing_changed(fork_store):
    path, _ = fork_store
    holder = subprocess.Popen([sys.executable, "-c", _LOCK_HOLDER, str(path.parent)], cwd=REPO, text=True,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        assert holder.stdout.readline().strip() == "held"
        sha, listing = _sha(path), sorted(p.name for p in path.parent.iterdir())
        with pytest.raises(fence.ForwardSchemaMigrationAdmissionError) as caught:
            SessionDB(db_path=path)
        assert caught.value.code == "STATE_DB_FORWARD_MIGRATION_ADMISSION_BLOCKED"
        assert "a running gateway holds its runtime lock" in str(caught.value)
        assert _sha(path) == sha and sorted(p.name for p in path.parent.iterdir()) == listing
        assert fence.probe_store_lineage(path).migratable
    finally:
        holder.stdin.close()
        holder.wait(timeout=_JOIN_S)


def test_the_gateway_that_owns_the_profile_in_this_process_lends_its_lock(fork_store):
    from gateway.runtime_ownership import process_ownership

    path, _ = fork_store
    process_ownership.reserve([path.parent])
    try:
        SessionDB(db_path=path).close()
    finally:
        process_ownership.release(path.parent)
    assert fence.probe_store_lineage(path).settled


# ── UDF serialization ──

def test_fence_udf_runs_inside_the_serializer_and_a_shared_connection_does_not_deadlock(migrated, monkeypatch):
    holder, seen, errors = {}, [], []

    def generation():
        conn = holder.get("conn")
        if conn is not None:
            seen.append(conn._hermes_serial_lock._is_owned())
        return fence.TURN_FENCE_GENERATION

    monkeypatch.setattr(fence, "_turn_fence_generation", generation)
    db = SessionDB(db_path=migrated)
    holder["conn"] = conn = db._conn
    threads = []
    try:
        assert isinstance(conn, fence.SerializedConnectionMixin)
        db.create_session("s2", source="cli")
        assert seen and all(seen)

        def write(sid):
            conn.execute("INSERT INTO sessions (id, source, started_at) VALUES (?, 'cli', 1.0)", (sid,))
            for i in range(200):
                conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (f"{sid}-{i}", sid))

        def read():
            for _ in range(200):
                cursor = conn.execute("SELECT id, title FROM sessions")
                assert cursor.description
                cursor.fetchall()

        def guarded(fn, *args):
            def run():
                try:
                    fn(*args)
                except BaseException as exc:  # reported below
                    errors.append(exc)
            return run

        threads = [threading.Thread(target=guarded(write, f"w{i}"), daemon=True) for i in range(3)]
        threads += [threading.Thread(target=guarded(read), daemon=True) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=_JOIN_S)
        assert not [t for t in threads if t.is_alive()], "shared connection deadlocked"
        assert errors == []
    finally:
        if not [t for t in threads if t.is_alive()]:
            db.close()


def test_timeline_udfs_run_inside_the_same_serializer(migrated, monkeypatch):
    import hermes_state_timeline as timeline

    conns, seen = [], []
    real_register, real_preview = timeline._register_functions, timeline._prompt_preview

    def register(db, conn):
        conns.append(conn)
        real_register(db, conn)

    def preview(db, content, kind, summary):
        seen.append(conns[-1]._hermes_serial_lock._is_owned())
        return real_preview(db, content, kind, summary)

    monkeypatch.setattr(timeline, "_register_functions", register)
    monkeypatch.setattr(timeline, "_prompt_preview", preview)
    db = SessionDB(db_path=migrated)
    try:
        timeline.get_session_timeline(db, "s1")
    finally:
        db.close()
    assert seen and all(seen)


# ── The search-index stamp ──

def test_optimize_settle_advances_a_fenced_stamp_only_through_the_lineage_helper(migrated):
    db = SessionDB(db_path=migrated)
    try:
        db._conn.execute("UPDATE schema_version SET version = 1031")
        assert db._optimize_settle(db._conn) is None
        assert _stamp(db._conn) == 1033
    finally:
        db.close()


def test_optimize_settle_still_advances_an_upstream_stamp(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db._conn.execute("UPDATE schema_version SET version = 30")
        assert db._optimize_settle(db._conn) is None
        assert _stamp(db._conn) == 33
        assert _fence_sql(db._conn) == {}
    finally:
        db.close()
