"""state.db stores of each lineage, for the turn-fence contracts.

The fork store is replayed from ``fixtures/fork_gen29_schema.sql`` and
``fixtures/fork_gen29_rows.json`` (generated from the fork commit by
``scripts/state_fence/gen_fork_fixture.py``). Rows are inserted with a
generation-29 UDF registered, as a fork writer would, so the fork's own
triggers derive the authority rows and FTS content.

The fence trigger text is written out here rather than imported, so the
fixtures are an independent oracle for the production builder and can be
built on a base checkout that has no fence module.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from hermes_state_common import SCHEMA_VERSION

# The lineage scheme written out (base has no fence module). Every generation below is derived
# from SCHEMA_VERSION so an upstream bump moves them with it; only the fork's history (27-29) is
# literal. Kind names record the values at SCHEMA_VERSION 30.
FENCE_LINEAGE_BASE = 1000
CURRENT_STAMP = FENCE_LINEAGE_BASE + SCHEMA_VERSION
FUTURE_STAMP = CURRENT_STAMP + 1
UPSTREAM_FUTURE = SCHEMA_VERSION + 1

FIXTURES = Path(__file__).parent / "fixtures"
FORK_SCHEMA_SQL = FIXTURES / "fork_gen29_schema.sql"
FORK_ROWS_JSON = FIXTURES / "fork_gen29_rows.json"

CORE_GOVERNED = (
    "async_delegations", "compression_locks", "gateway_routing", "messages",
    "session_model_usage", "session_turn_leases", "sessions", "system_prompts",
)
AUTHORITY_GOVERNED = ("session_process_authorities", "session_process_reservations")
OPERATIONS = ("INSERT", "UPDATE", "DELETE")

KINDS = (
    "fork29", "fork29_touched", "fenced1030", "upstream29", "upstream30", "fresh",
    "stored1031", "upstream31", "two_row", "text_scalar", "unowned_fence",
)
# Kinds this build must refuse, with the IncompatibleSchemaError cause.
REFUSED_KINDS = {
    "stored1031": "BUILD_TOO_OLD",
    "upstream31": "BUILD_TOO_OLD",
    "two_row": "SCHEMA_VERSION_UNREADABLE",
    "text_scalar": "SCHEMA_VERSION_UNREADABLE",
    "unowned_fence": "FENCE_GENERATION_MISMATCH",
}


def fence_sql(table: str, operation: str, literal: int) -> str:
    return (
        f"CREATE TRIGGER turn_fence_{table}_{operation.lower()} BEFORE {operation} ON {table} "
        "BEGIN "
        "SELECT CASE "
        "WHEN typeof(hermes_turn_fence_generation()) != 'integer' "
        f"OR hermes_turn_fence_generation() != {literal} "
        "THEN RAISE(ABORT, 'state DB generation incompatible') "
        "END; "
        "END"
    )


def _connect_as(path: Path, generation) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), isolation_level=None)
    if generation is not None:
        conn.create_function("hermes_turn_fence_generation", 0, lambda: generation)
    return conn


def _tables(conn) -> set:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def refence(conn: sqlite3.Connection, literal: int, *, tables=None) -> None:
    """Replace every turn_fence_* trigger with the builder text at *literal* on *tables* (default: all governed present)."""
    present = _tables(conn)
    governed = tables or [t for t in CORE_GOVERNED + AUTHORITY_GOVERNED if t in present]
    conn.execute("BEGIN IMMEDIATE")
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'turn_fence_%'").fetchall():
        conn.execute(f'DROP TRIGGER "{name}"')
    for table in governed:
        for op in OPERATIONS:
            conn.execute(fence_sql(table, op, literal))
    conn.execute("COMMIT")


def _insert_rows(conn, table: str, spec: dict, *, overrides=None) -> None:
    cols = spec["columns"]
    marks = ", ".join("?" for _ in cols)
    names = ", ".join(f'"{c}"' for c in cols)
    overrides = overrides or {}
    for row in spec["rows"]:
        values = [overrides.get(c, v) for c, v in zip(cols, row)]
        conn.execute(f'INSERT INTO "{table}" ({names}) VALUES ({marks})', values)


def _replay_sessions(conn, spec: dict) -> None:
    """Insert sessions open at generation 0 and drive each through the fork's own close/reopen
    triggers to its recorded generation and end state, so authority rows are the fork's."""
    cols = spec["columns"]
    _insert_rows(conn, "sessions", spec, overrides={"session_generation": 0, "ended_at": None, "end_reason": None})
    for row in spec["rows"]:
        record = dict(zip(cols, row))
        sid = record["id"]
        while conn.execute("SELECT session_generation FROM sessions WHERE id = ?", (sid,)).fetchone()[0] < record["session_generation"]:
            conn.execute("UPDATE sessions SET ended_at = started_at, end_reason = 'user_exit' WHERE id = ?", (sid,))
            conn.execute("UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?", (sid,))
        if record["ended_at"] is not None:
            conn.execute("UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?",
                         (record["ended_at"], record["end_reason"], sid))


def _build_fork29(path: Path) -> None:
    rows = json.loads(FORK_ROWS_JSON.read_text(encoding="utf-8"))["tables"]
    conn = _connect_as(path, 29)
    try:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        conn.executescript(FORK_SCHEMA_SQL.read_text(encoding="utf-8"))
        conn.execute("BEGIN IMMEDIATE")
        # state_meta first: the fork's session triggers read its identity keys.
        order = ["state_meta", "schema_version", "sessions"] + sorted(
            t for t in rows if t not in {"state_meta", "schema_version", "sessions"})
        for table in order:
            if table == "sessions":
                _replay_sessions(conn, rows[table])
            elif table in rows:
                _insert_rows(conn, table, rows[table])
        conn.execute("COMMIT")
    finally:
        conn.close()


def _build_r_store(path: Path) -> None:
    """The store plain R (upstream) leaves. A fenced build's SessionDB writes it, so its fences and
    lineage stamp come off again: R declares neither (a no-op on a build without fences)."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=path)
    try:
        db.create_session("r-alpha", "cli")
        db.append_message("r-alpha", "user", "hello upstream")
    finally:
        db.close()
    conn = _connect_as(path, None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        for (name,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'turn_fence_%'").fetchall():
            conn.execute(f'DROP TRIGGER "{name}"')
        conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
        conn.execute("COMMIT")
    finally:
        conn.close()


def _raw(path: Path, *statements, generation=None) -> None:
    conn = _connect_as(path, generation)
    try:
        for sql in statements:
            conn.execute(sql)
    finally:
        conn.close()


def build_store(path: Path, kind: str) -> Path:
    """Create a ``state.db`` of *kind* at *path*: WAL journal mode, no sidecars left behind."""
    if kind not in KINDS:
        raise ValueError(kind)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "fresh":
        return path
    if kind.startswith("fork29") or kind == "fenced1030":
        _build_fork29(path)
        if kind == "fork29_touched":
            from hermes_state_schema import reconcile_state_schema

            conn = _connect_as(path, 29)
            try:
                reconcile_state_schema(conn)
            finally:
                conn.close()
        if kind == "fenced1030":
            conn = _connect_as(path, 29)
            try:
                refence(conn, CURRENT_STAMP)
                conn.execute("UPDATE schema_version SET version = ?", (CURRENT_STAMP,))
            finally:
                conn.close()
    else:
        _build_r_store(path)
        if kind == "upstream29":
            _raw(path, "UPDATE schema_version SET version = 29")
        elif kind == "upstream31":
            _raw(path, f"UPDATE schema_version SET version = {UPSTREAM_FUTURE}")
        elif kind == "two_row":
            _raw(path, f"INSERT INTO schema_version (version) VALUES ({SCHEMA_VERSION})")
        elif kind == "text_scalar":
            # Non-numeric: INTEGER affinity would silently coerce '30' back to an integer.
            _raw(path, "UPDATE schema_version SET version = 'v30'")
        elif kind == "stored1031":
            conn = _connect_as(path, None)
            try:
                refence(conn, FUTURE_STAMP, tables=list(CORE_GOVERNED))
            finally:
                conn.close()
            _raw(path, f"UPDATE schema_version SET version = {FUTURE_STAMP}")
        elif kind == "unowned_fence":
            _raw(path, "CREATE TRIGGER turn_fence_messages_insert BEFORE INSERT ON messages "
                       "BEGIN SELECT hermes_turn_fence_generation(); END")
    # WAL like the live stores (a runtime with the WAL-reset gate builds R stores in DELETE mode):
    # a mode=ro open of a WAL file is what mints stray -wal/-shm sidecars.
    _raw(path, "PRAGMA journal_mode=WAL")
    for suffix in ("-wal", "-shm", "-journal"):
        assert not Path(f"{path}{suffix}").exists(), f"fixture left {suffix}"
    return path


def isolate_home(tmp_path: Path, monkeypatch) -> Path:
    """Point HOME and HERMES_HOME at temp dirs (the suite only redirects HERMES_HOME); returns HERMES_HOME."""
    # Not HOME/.hermes: the suite's live-system guard treats that as the production root.
    home = Path(tmp_path) / "home"
    hermes_home = Path(tmp_path) / "hermes_home"
    home.mkdir(parents=True, exist_ok=True)
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def build_fork_gen29_home(tmp_path: Path, kind: str = "fork29") -> Path:
    """A HERMES_HOME-shaped directory whose state.db is a store of *kind*; returns the db path."""
    return build_store(Path(tmp_path) / "state.db", kind)


def file_fingerprint(path: Path) -> dict:
    """Everything a refused open must leave unchanged."""
    path = Path(path)
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        master = conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY rowid").fetchall()
        cookie = conn.execute("PRAGMA schema_version").fetchone()[0]
    finally:
        conn.close()
    header = path.read_bytes()[:100]
    return {
        "bytes": path.read_bytes(),
        "mtime_ns": path.stat().st_mtime_ns,
        "journal_mode_header": (header[18], header[19]),
        "sidecars": sorted(s for s in ("-wal", "-shm", "-journal") if Path(f"{path}{s}").exists()),
        "sqlite_master": master,
        "schema_cookie": cookie,
    }
