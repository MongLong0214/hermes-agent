"""This (fork) build reads a generation-1033 state.db read-only, for recovery after a code rollback.

After cutover the v0.23.1 line stamps the store 1033 and fences every governed table at 1033, the
four tables its schema 31-32 added included. The default open refuses that store as BUILD_TOO_OLD and
must keep doing so; ``SessionDB.open_recovery_reader`` is the one path that serves it, and only
read-only.

The 1033 store is built to the candidate's DDL: ``fixtures/target_gen1033_schema.sql`` (its header
says how it was generated) runs on an empty file, then every row of a 1030 store this build created
is copied in on the columns both share, under a connection carrying the 1033 fence UDF, as a 1033
writer would. A session written afterwards fills target-only columns and tables (the writes made
after cutover). The candidate's data migrations are not replayed, so the copied rows are the source
rows exactly and "identical to the source" is a property of the reader alone.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_errors import IncompatibleSchemaError
from tests.hermes_state.fork_store_fixture import CORE_GOVERNED, OPERATIONS, fence_sql

TARGET_SCHEMA_SQL = Path(__file__).parent / "fixtures" / "target_gen1033_schema.sql"
TARGET_STAMP = 1033
TARGET_GOVERNED = ("runtime_epoch", "session_admissions", "worker_executions", "worker_receipts")
SOURCE_SESSIONS = ("s1", "c1")
TARGET_SESSION_COLUMNS = {
    "runtime_revision", "runtime_generation", "created_source", "compression_overload_streak",
    "auto_archived", "transcript_revision",
}
TARGET_MESSAGE_COLUMNS = {"message_uid", "absorbed_message_uids", "tool_call_uids", "tool_call_uid"}


def _build_source(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=path)
    try:
        db.create_session("s1", source="cli", model="m1")
        db.set_session_title("s1", "First title")
        db.append_message("s1", role="user", content="hello fork store")
        db.append_message("s1", role="assistant", content="the needle is here", tool_calls=[
            {"id": "t1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}])
        db.append_message("s1", role="tool", content="tool says hi", tool_call_id="t1", tool_name="lookup")
        db.append_message("s1", role="assistant", content="한국어 답변 needle")
        db.create_session("c1", source="subagent", parent_session_id="s1")
        db.append_message("c1", role="user", content="child work")
    finally:
        db.close()
    return path


def _connect_1033(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path.as_uri(), uri=True, isolation_level=None)
    conn.create_function("hermes_turn_fence_generation", 0, lambda: TARGET_STAMP)
    return conn


def _build_1033(source: Path, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect_1033(path)
    try:
        conn.executescript(TARGET_SCHEMA_SQL.read_text(encoding="utf-8"))
        conn.execute("ATTACH DATABASE ? AS src", (f"{source.as_uri()}?mode=ro&immutable=1",))
        conn.execute("BEGIN")
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM src.sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "AND name NOT LIKE 'messages_fts%' ORDER BY rowid")]
        for table in tables:
            target_cols = {r[1] for r in conn.execute(f'PRAGMA main.table_info("{table}")')}
            cols = ", ".join(f'"{r[1]}"' for r in conn.execute(f'PRAGMA src.table_info("{table}")')
                             if r[1] in target_cols)
            conn.execute(f'INSERT INTO main."{table}" ({cols}) SELECT {cols} FROM src."{table}" ORDER BY rowid')
        conn.execute("UPDATE schema_version SET version = ?", (TARGET_STAMP,))
        conn.execute("COMMIT")
        conn.execute("DETACH DATABASE src")
        # Written by the 1033 build after cutover: target-only columns and tables carry values.
        conn.execute("BEGIN")
        conn.execute(
            "INSERT INTO sessions (id, source, started_at, runtime_revision, runtime_generation, created_source, "
            "transcript_revision) VALUES ('n1', 'api_server', 1794540000.0, 3, 2, 'api_server', 2)")
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, message_uid) "
            "VALUES ('n1', 'user', 'written after cutover', 1794540001.0, 'uid-n1-1')")
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, message_uid) "
            "VALUES ('n1', 'assistant', 'the newer build answered', 1794540002.0, 'uid-n1-2')")
        conn.execute("INSERT INTO runtime_epoch (singleton, epoch, instance_id) VALUES (1, 7, 'instance-1033')")
        conn.execute("COMMIT")
        assert conn.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
    finally:
        conn.close()
    assert not Path(f"{path}-wal").exists()
    return path


def _alter(path: Path, *statements: str) -> None:
    conn = _connect_1033(path)
    try:
        for sql in statements:
            conn.execute(sql)
    finally:
        conn.close()


def _fingerprint(path: Path) -> tuple:
    sidecars = sorted(p.name for p in path.parent.glob(f"{path.name}-*"))
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None, sidecars


def _open_recovery(path: Path) -> SessionDB:
    return SessionDB.open_recovery_reader(path)


def _reads(db: SessionDB, session_ids) -> dict:
    out = {sid: {
        "session": db.get_session(sid),
        "messages": db.get_messages(sid),
        "display": db.get_messages(sid, include_compacted=True),
        "conversation": db.get_messages_as_conversation(sid, repair_alternation=True),
        "resume": db.get_resume_conversations(sid),
        "resume_id": db.resolve_resume_session_id(sid),
        "tip": db.get_compression_tip(sid),
        "title": db.get_session_title(sid),
    } for sid in session_ids}
    out["listed"] = sorted(s["id"] for s in db.list_sessions_rich())
    out["search"] = db.search_messages("needle")
    out["search_cjk"] = db.search_messages("한국어")
    return out


def _project(value, like):
    """*value* cut to the shape of *like*: dict keys *like* lacks (target-only columns) are dropped."""
    if isinstance(like, dict) and isinstance(value, dict):
        return {key: _project(value[key], sub) if key in value else "<missing>" for key, sub in like.items()}
    if isinstance(like, (list, tuple)) and isinstance(value, (list, tuple)) and len(like) == len(value):
        return type(like)(_project(v, sub) for v, sub in zip(value, like))
    return value


@pytest.fixture
def stores(tmp_path):
    source = _build_source(tmp_path / "source" / "state.db")
    return source, _build_1033(source, tmp_path / "target" / "state.db")


def test_a_1033_store_reads_back_identical_to_its_source(stores):
    source, target = stores
    with SessionDB(db_path=source, read_only=True) as db:
        expected = _reads(db, SOURCE_SESSIONS)
    before = _fingerprint(target)
    db = _open_recovery(target)
    try:
        recovered = _reads(db, SOURCE_SESSIONS)
        assert db.search_unavailable_reason is None
    finally:
        db.close()

    assert expected["s1"]["messages"] and expected["search"] and expected["search_cjk"]
    assert recovered["listed"] == sorted(expected["listed"] + ["n1"])
    del recovered["listed"], expected["listed"]
    assert _project(recovered, expected) == expected
    # The target-only columns are what the projection dropped, and nothing else.
    assert set(recovered["s1"]["session"]) - set(expected["s1"]["session"]) == TARGET_SESSION_COLUMNS
    assert set(recovered["s1"]["messages"][0]) - set(expected["s1"]["messages"][0]) == TARGET_MESSAGE_COLUMNS
    assert _fingerprint(target) == before


def test_rows_written_by_the_1033_build_are_served(stores):
    _, target = stores
    db = _open_recovery(target)
    try:
        session = db.get_session("n1")
        messages = db.get_messages("n1")
        conversation = db.get_messages_as_conversation("n1", repair_alternation=True)
        hits = db.search_messages("cutover")
    finally:
        db.close()
    assert (session["runtime_revision"], session["created_source"], session["transcript_revision"]) == (
        3, "api_server", 2)
    assert [(m["role"], m["content"], m["message_uid"]) for m in messages] == [
        ("user", "written after cutover", "uid-n1-1"), ("assistant", "the newer build answered", "uid-n1-2")]
    assert [(m["role"], m["content"]) for m in conversation] == [
        ("user", "written after cutover"), ("assistant", "the newer build answered")]
    assert [h["session_id"] for h in hits] == ["n1"]


def test_an_uncheckpointed_wal_frame_is_served(stores):
    """A 1033 store left with a -wal (the newer build stopped uncleanly) is read through it, not past it."""
    _, target = stores
    writer = _connect_1033(target)
    try:
        writer.execute("PRAGMA wal_autocheckpoint = 0")
        writer.execute("INSERT INTO messages (session_id, role, content, timestamp, message_uid) "
                       "VALUES ('n1', 'user', 'only in the wal', 1794540003.0, 'uid-n1-3')")
        assert Path(f"{target}-wal").stat().st_size > 0
        db = _open_recovery(target)
        try:
            assert [m["content"] for m in db.get_messages("n1")][-1] == "only in the wal"
        finally:
            db.close()
    finally:
        writer.close()


WRITES = {
    "append_message": lambda db: db.append_message("s1", role="user", content="refused"),
    "create_session": lambda db: db.create_session("w1", source="cli"),
    "set_session_title": lambda db: db.set_session_title("s1", "renamed"),
    "end_session": lambda db: db.end_session("s1", "user_exit"),
    "delete_session": lambda db: db.delete_session("c1"),
    "replace_messages": lambda db: db.replace_messages("s1", []),
}


@pytest.mark.parametrize("write", sorted(WRITES))
def test_every_write_api_is_refused_with_a_clear_error_and_the_file_is_unchanged(stores, write):
    _, target = stores
    before = _fingerprint(target)
    db = _open_recovery(target)
    try:
        import hermes_state_recovery as recovery

        with pytest.raises(recovery.StateRecoveryReadOnlyError, match="read-only recovery"):
            WRITES[write](db)
        assert [m["content"] for m in db.get_messages("s1")][0] == "hello fork store"
    finally:
        db.close()
    assert _fingerprint(target) == before


@pytest.mark.parametrize("sql", [
    "INSERT INTO state_meta (key, value) VALUES ('k', 'v')",  # an ungoverned table
    "UPDATE messages SET content = 'x'",
    "CREATE TABLE stray (a)",
])
def test_a_write_that_bypasses_the_write_path_is_refused_by_the_engine(stores, sql):
    _, target = stores
    before = _fingerprint(target)
    db = _open_recovery(target)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db._conn.execute(sql)
    finally:
        db.close()
    assert _fingerprint(target) == before


@pytest.mark.parametrize("read_only", [False, True])
def test_the_default_open_still_refuses_a_1033_store_as_build_too_old(stores, read_only):
    _, target = stores
    before = _fingerprint(target)
    with pytest.raises(IncompatibleSchemaError) as refused:
        SessionDB(db_path=target, read_only=read_only)
    assert refused.value.cause == "BUILD_TOO_OLD"
    assert _fingerprint(target) == before


def _refence(table: str, literal: int) -> list:
    return [stmt for op in OPERATIONS for stmt in (
        f"DROP TRIGGER turn_fence_{table}_{op.lower()}", fence_sql(table, op, literal))]


def _every_fence(literal: int) -> list:
    return [stmt for table in CORE_GOVERNED + TARGET_GOVERNED for stmt in _refence(table, literal)]


REFUSED = {
    "stamp_above_1033": (["UPDATE schema_version SET version = 1034"], "BUILD_TOO_OLD"),
    "fences_above_1033": (_every_fence(1034), "BUILD_TOO_OLD"),
    "stamp_1032": (["UPDATE schema_version SET version = 1032"], "FENCE_GENERATION_MISMATCH"),
    "core_fence_at_1030": (_refence("messages", 1030), "FENCE_GENERATION_MISMATCH"),
    "target_fence_missing": (["DROP TRIGGER turn_fence_worker_receipts_delete"], "FENCE_GENERATION_MISMATCH"),
    "target_table_unfenced": ([f"DROP TRIGGER turn_fence_runtime_epoch_{op.lower()}" for op in OPERATIONS],
                              "FENCE_GENERATION_MISMATCH"),
    "unowned_fence": (["CREATE TRIGGER turn_fence_extra_insert BEFORE INSERT ON messages "
                       "BEGIN SELECT hermes_turn_fence_generation(); END"], "FENCE_GENERATION_MISMATCH"),
    "altered_fence": (["DROP TRIGGER turn_fence_sessions_update",
                       fence_sql("sessions", "UPDATE", TARGET_STAMP).replace("!= 'integer'", "= 'text'")],
                      "FENCE_GENERATION_MISMATCH"),
    "two_stamps": (["INSERT INTO schema_version (version) VALUES (1033)"], "SCHEMA_VERSION_UNREADABLE"),
    "no_stamp": (["DROP TABLE schema_version"], "SCHEMA_VERSION_UNREADABLE"),
}


@pytest.mark.parametrize("kind", sorted(REFUSED))
def test_a_store_that_is_not_exactly_1033_is_refused_even_in_recovery_mode(stores, kind):
    _, target = stores
    statements, cause = REFUSED[kind]
    _alter(target, *statements)
    before = _fingerprint(target)
    with pytest.raises(IncompatibleSchemaError) as refused:
        _open_recovery(target)
    assert refused.value.cause == cause
    assert _fingerprint(target) == before


def test_the_fork_store_itself_is_refused_in_recovery_mode(stores):
    """Recovery mode reads 1033 only; this build's own 1030 store opens through the default path."""
    source, _ = stores
    before = _fingerprint(source)
    with pytest.raises(IncompatibleSchemaError) as refused:
        _open_recovery(source)
    assert refused.value.cause == "FENCE_GENERATION_MISMATCH"
    assert _fingerprint(source) == before


@pytest.mark.parametrize("shape", ["missing", "empty"])
def test_a_missing_or_empty_file_is_refused_and_not_created(tmp_path, shape):
    path = tmp_path / "state.db"
    if shape == "empty":
        path.write_bytes(b"")
    with pytest.raises((IncompatibleSchemaError, FileNotFoundError)):
        _open_recovery(path)
    assert _fingerprint(path) == ((hashlib.sha256(b"").hexdigest(), []) if shape == "empty" else (None, []))


def test_search_is_reported_unavailable_when_the_fts_layout_is_not_this_builds(stores):
    _, target = stores
    _alter(target, "DROP TABLE messages_fts",
           "CREATE VIRTUAL TABLE messages_fts USING fts5(content, tool_name, tool_calls, origin, "
           "content='messages_fts_src', content_rowid='id')")
    db = _open_recovery(target)
    try:
        import hermes_state_recovery as recovery

        assert "messages_fts" in db.search_unavailable_reason
        with pytest.raises(recovery.StateRecoverySearchUnavailableError, match="search is unavailable"):
            db.search_messages("needle")
        assert [m["content"] for m in db.get_messages("s1")][1] == "the needle is here"
    finally:
        db.close()
