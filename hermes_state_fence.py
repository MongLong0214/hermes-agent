"""Turn-fence generation for state.db: the UDF, the fence trigger builders, and the
read-only lineage probe every state.db opener runs before touching the file.

A fence trigger aborts each governed INSERT/UPDATE/DELETE unless the writing
connection's ``hermes_turn_fence_generation()`` equals the literal the store was
fenced with, so a build that does not own a store's generation cannot write it
(a connection without the UDF fails with "no such function"). DDL is never
fenced, which is why the lineage decision below must be made with SELECTs alone
on a read-only connection, before any ``SCHEMA_SQL``, reconcile, journal-mode
change or write: a refused store must come out of the open byte-identical.

Stored ``schema_version`` values carry the lineage: upstream and the fork stamp
their own integers (< 1000); this line stamps ``FENCE_LINEAGE_BASE +
SCHEMA_VERSION`` so neither older line can misread it as its own.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from hermes_state_common import SCHEMA_VERSION
from hermes_state_errors import (
    SCHEMA_CAUSE_BUILD_TOO_OLD, SCHEMA_CAUSE_VERSION_UNREADABLE, IncompatibleSchemaError,
)
from hermes_state_fence_classifier import owned_turn_fence_literals

logger = logging.getLogger("hermes_state")

FENCE_LINEAGE_BASE = 1000
STORED_SCHEMA_VERSION = FENCE_LINEAGE_BASE + SCHEMA_VERSION
TURN_FENCE_GENERATION = STORED_SCHEMA_VERSION
FORK_LEGACY_GENERATIONS = frozenset({27, 28, 29})
FORK_BASE_UPSTREAM_GATE = 26

TURN_FENCE_FUNCTION = "hermes_turn_fence_generation"
FENCE_OPERATIONS = ("INSERT", "UPDATE", "DELETE")
CORE_GOVERNED_TABLES = (
    "async_delegations", "compression_locks", "gateway_routing", "messages",
    "session_model_usage", "session_turn_leases", "sessions", "system_prompts",
)
# Governed only where the table exists: the fork's AFTER INSERT ON sessions trigger
# inserts into session_process_authorities, so leaving that table on an old literal
# would abort every session insert.
SESSION_PROCESS_GOVERNED_TABLES = ("session_process_authorities", "session_process_reservations")
ALL_GOVERNED_TABLES = CORE_GOVERNED_TABLES + SESSION_PROCESS_GOVERNED_TABLES

LINEAGE_FRESH = "fresh"
LINEAGE_UPSTREAM = "upstream"
LINEAGE_FORK = "fork"
LINEAGE_FENCED = "fenced"
LINEAGE_DAMAGED = "damaged"

SCHEMA_INCOMPATIBLE_VERDICT_PREFIX = "schema_incompatible: "
_FENCE_REFUSAL_PHRASES = ("state db generation incompatible", f"no such function: {TURN_FENCE_FUNCTION}")


def _turn_fence_generation() -> int:
    return TURN_FENCE_GENERATION


def register_turn_fence_generation(conn: sqlite3.Connection) -> None:
    conn.create_function(TURN_FENCE_FUNCTION, 0, _turn_fence_generation, deterministic=True)


def turn_fence_trigger_name(table: str, operation: str) -> str:
    return f"turn_fence_{table}_{operation.lower()}"


def turn_fence_trigger_sql(table: str, operation: str, *, generation: Optional[int] = None) -> str:
    # Byte-compatible with the fork builder for its generations: ownership is proven by token equality.
    # The default resolves at call time, so every builder reads the one module binding of the generation.
    generation = TURN_FENCE_GENERATION if generation is None else generation
    return (
        f"CREATE TRIGGER {turn_fence_trigger_name(table, operation)} BEFORE {operation} ON {table} "
        "BEGIN "
        "SELECT CASE "
        f"WHEN typeof({TURN_FENCE_FUNCTION}()) != 'integer' "
        f"OR {TURN_FENCE_FUNCTION}() != {generation} "
        "THEN RAISE(ABORT, 'state DB generation incompatible') "
        "END; "
        "END"
    )


def turn_fence_trigger_definitions(governed, *, generation: Optional[int] = None) -> dict:
    return {
        turn_fence_trigger_name(table, op): turn_fence_trigger_sql(table, op, generation=generation)
        for table in governed for op in FENCE_OPERATIONS
    }


def schema_text(value) -> str:
    """sqlite_master text read as a BLOB: a store with invalid UTF-8 in its schema must decode, not
    raise, so the probe never fails an open that SessionDB's own FTS probes are built to survive."""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else (value or "")


def _existing_tables(cursor) -> set:
    return {schema_text(row[0]) for row in cursor.execute(
        "SELECT CAST(name AS BLOB) FROM sqlite_master WHERE type = 'table'").fetchall()}


def governed_tables(cursor) -> tuple:
    present = _existing_tables(cursor)
    return CORE_GOVERNED_TABLES + tuple(t for t in SESSION_PROCESS_GOVERNED_TABLES if t in present)


@dataclass(frozen=True)
class StoreLineage:
    lineage: str
    stored: Optional[int]
    gate: int
    fence_literals: frozenset = frozenset()

    @property
    def writable_by_this_build(self) -> bool:
        """False when owned fences carry a generation other than this build's, so every governed write aborts."""
        return not (self.fence_literals - {TURN_FENCE_GENERATION})


_FRESH = StoreLineage(LINEAGE_FRESH, None, 0)
_DAMAGED = StoreLineage(LINEAGE_DAMAGED, None, 0)


def decode_store_lineage(cursor) -> StoreLineage:
    """SELECT-only lineage decode on an open connection; raises IncompatibleSchemaError on refusal."""
    tables = _existing_tables(cursor)
    if "schema_version" not in tables:
        return _FRESH
    rows = cursor.execute(
        "SELECT CASE WHEN typeof(version) = 'integer' THEN version END, typeof(version) FROM schema_version").fetchall()
    # Zero rows is R's own mid-bootstrap shape (reconcile creates the table empty and
    # _init_schema then inserts the stamp), so it decodes as fresh, not as unreadable.
    if not rows:
        return _FRESH
    if len(rows) != 1 or rows[0][1] != "integer":
        raise IncompatibleSchemaError(
            cause=SCHEMA_CAUSE_VERSION_UNREADABLE, expected_generation=STORED_SCHEMA_VERSION,
            actual_generation=None, detail=f"{len(rows)} row(s), type {rows[0][1] if rows else 'none'}",
        )
    stored = int(rows[0][0])
    literals = frozenset(owned_turn_fence_literals(cursor).values())
    if stored >= FENCE_LINEAGE_BASE:
        if stored > STORED_SCHEMA_VERSION:
            raise IncompatibleSchemaError(
                cause=SCHEMA_CAUSE_BUILD_TOO_OLD, expected_generation=STORED_SCHEMA_VERSION, actual_generation=stored,
            )
        return StoreLineage(LINEAGE_FENCED, stored, stored - FENCE_LINEAGE_BASE, literals)
    if stored > SCHEMA_VERSION:
        raise IncompatibleSchemaError(
            cause=SCHEMA_CAUSE_BUILD_TOO_OLD, expected_generation=SCHEMA_VERSION, actual_generation=stored,
        )
    if _has_fork_markers(cursor, tables, literals):
        return StoreLineage(LINEAGE_FORK, stored, min(stored, FORK_BASE_UPSTREAM_GATE), literals)
    return StoreLineage(LINEAGE_UPSTREAM, stored, stored, literals)


def _has_fork_markers(cursor, tables: set, literals: frozenset) -> bool:
    if literals & FORK_LEGACY_GENERATIONS or "session_process_authorities" in tables:
        return True
    if "state_meta" not in tables:
        return False
    return cursor.execute(
        "SELECT 1 FROM state_meta WHERE key = 'session_process_state_family' LIMIT 1").fetchone() is not None


def _decode_at(uri: str, path: Path) -> StoreLineage:
    # Tracked, so a concurrent byte-level header probe cannot open()/close() the file under it.
    from hermes_state_dbfile import _connect_tracked_db

    conn = _connect_tracked_db(uri, tracking_path=path, uri=True, timeout=1.0, isolation_level=None)
    try:
        return decode_store_lineage(conn.cursor())
    except UnicodeDecodeError:
        # pysqlite could not decode SQLite's own error text: the schema holds invalid UTF-8, so
        # no statement on this store can run and its lineage is unknowable. That is damage, not
        # a lineage, and the caller's own probes are built to survive it (#98924).
        return _DAMAGED
    finally:
        conn.close()


def probe_store_lineage(path) -> StoreLineage:
    """Decode *path*'s lineage without writing a byte to it or creating a sidecar."""
    from hermes_state_holders import read_only_db_uri

    path = Path(path)
    try:
        if path.stat().st_size == 0:
            return _FRESH
    except FileNotFoundError:
        return _FRESH
    uri = read_only_db_uri(path)
    # A mode=ro open of a WAL-mode file with no sidecars creates -wal/-shm and leaves them
    # behind; immutable=1 reads the main file only, which is sound exactly when no sidecar
    # exists (no connection holds the store in WAL mode).
    if not Path(f"{path}-wal").exists() and not Path(f"{path}-journal").exists():
        try:
            return _decode_at(f"{uri}&immutable=1", path)
        except sqlite3.DatabaseError:
            # A rollback-journal writer that started after the sidecar check can tear an
            # immutable read; the locked read-only retry decides.
            pass
    return _decode_at(uri, path)


def validate_state_connection(conn: sqlite3.Connection) -> StoreLineage:
    """Lineage decode on an already-open connection (raw openers, after registration)."""
    return decode_store_lineage(conn.cursor())


def schema_incompatibility_verdict(path) -> Optional[str]:
    """``schema_incompatible: ...`` when this build must not repair or write *path*, else None.

    Covers both refusal causes and stores whose owned fences carry another generation:
    those are healthy for their writer, and the repair ladder would strip their fences.
    A store SQLite cannot read is damaged, not incompatible: None, so the caller's own
    probes name the damage."""
    try:
        lineage = probe_store_lineage(path)
    except IncompatibleSchemaError as exc:
        return f"{SCHEMA_INCOMPATIBLE_VERDICT_PREFIX}{exc}"
    except sqlite3.DatabaseError:
        return None
    if not lineage.writable_by_this_build:
        stored = ", ".join(str(g) for g in sorted(lineage.fence_literals - {TURN_FENCE_GENERATION}))
        return (f"{SCHEMA_INCOMPATIBLE_VERDICT_PREFIX}turn-fence generation {stored} on the store does not match "
                f"this build's generation {TURN_FENCE_GENERATION}; it was written by another Hermes line")
    return None


def fence_refusal_verdict(exc_or_text) -> Optional[str]:
    """Map a fence abort or a missing-UDF error to a schema_incompatible verdict, else None."""
    text = str(exc_or_text)
    if any(phrase in text.lower() for phrase in _FENCE_REFUSAL_PHRASES):
        return f"{SCHEMA_INCOMPATIBLE_VERDICT_PREFIX}{text}"
    return None


def is_schema_incompatible_verdict(reason: Optional[str]) -> bool:
    return bool(reason) and str(reason).startswith(SCHEMA_INCOMPATIBLE_VERDICT_PREFIX)


def open_fenced_state_connection(
    path,
    *,
    bootstrap: bool,
    connect: Callable[[], sqlite3.Connection],
    initialize: Optional[Callable[[sqlite3.Connection], None]] = None,
) -> sqlite3.Connection:
    """Open a raw (non-SessionDB) connection to a state.db: probe, register, validate, then initialize.

    The probe runs before ``connect`` so a refused store gets no journal-mode change or DDL
    from the caller's opener. ``bootstrap`` (an opener of its own profile's store) lets the
    SessionDB facade go first: it creates a fresh store's schema, so the raw opener never becomes
    the store's first schema writer, and it migrates a store whose fences carry another
    generation, which would otherwise abort every governed write the raw opener makes. That is
    the one migration, under SessionDB's lock and in its single transaction."""
    lineage = probe_store_lineage(path)
    if bootstrap and (lineage.lineage == LINEAGE_FRESH or not lineage.writable_by_this_build):
        from hermes_state import SessionDB

        SessionDB(db_path=Path(path)).close()
    conn = connect()
    try:
        register_turn_fence_generation(conn)
        validate_state_connection(conn)
        if initialize is not None:
            initialize(conn)
    except BaseException:
        conn.close()
        raise
    return conn


# ── In-place migration: fence swap + lineage stamp ─────────────────────────────


def _expected_definitions(cursor) -> dict:
    return turn_fence_trigger_definitions(governed_tables(cursor))


def _is_exact(owned: dict, expected: dict) -> bool:
    return set(owned) == set(expected) and all(literal == TURN_FENCE_GENERATION for literal in owned.values())


def fences_exact(cursor) -> bool:
    """True when the owned fences are exactly this build's declarations for the governed tables present."""
    return _is_exact(owned_turn_fence_literals(cursor), _expected_definitions(cursor))


def _in_write_transaction(cursor, body):
    # R's writer is autocommit, so each fence step brings its own transaction; a caller that
    # already holds one (optimize-storage's settle) keeps the atomicity it asked for.
    if cursor.connection.in_transaction:
        return body()
    cursor.execute("BEGIN IMMEDIATE")
    try:
        result = body()
    except BaseException:
        cursor.execute("ROLLBACK")
        raise
    cursor.execute("COMMIT")
    return result


def _write_lineage_stamp(cursor, lineage: StoreLineage, stamp: int) -> None:
    if lineage.stored is None:
        cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (stamp,))
    else:
        cursor.execute("UPDATE schema_version SET version = ? WHERE version < ?", (stamp, stamp))


def _create_missing(cursor, owned: dict, expected: dict) -> list:
    missing = [name for name in expected if name not in owned]
    for name in missing:
        cursor.execute(expected[name])
    return missing


def _warn_restored(names: list) -> None:
    if names:
        logger.warning("restored %d missing turn-fence trigger(s) on state.db (a non-fence build's DDL "
                       "removed them): %s", len(names), ", ".join(names))


def _swap_fences(cursor, owned: dict, expected: dict) -> None:
    for name in owned:
        cursor.execute('DROP TRIGGER "{}"'.format(name.replace('"', '""')))
    for sql in expected.values():
        cursor.execute(sql)
    if not fences_exact(cursor):
        raise RuntimeError("turn-fence trigger verification failed after the swap")


def apply_fence_delta(cursor) -> StoreLineage:
    """Make this build's fences exact and stamp ``FENCE_LINEAGE_BASE + gate``, atomically.

    Runs after ``SCHEMA_SQL`` + column reconcile (DDL, never fenced) and before any governed
    DML. Returns the lineage decoded under the write lock; its ``gate`` drives the data
    migrations. A settled store (fenced lineage, exact fences) takes no lock and runs no DDL.
    From the commit on, the fork refuses the store at open and its UDF cannot write it."""
    lineage = decode_store_lineage(cursor)
    if lineage.lineage == LINEAGE_FENCED and fences_exact(cursor):
        return lineage

    def migrate() -> StoreLineage:
        current = decode_store_lineage(cursor)  # another opener may have migrated since the read
        owned, expected = owned_turn_fence_literals(cursor), _expected_definitions(cursor)
        if not _is_exact(owned, expected):
            if current.lineage == LINEAGE_FENCED and all(v == TURN_FENCE_GENERATION for v in owned.values()):
                _warn_restored(_create_missing(cursor, owned, expected))
            else:
                _swap_fences(cursor, owned, expected)
        stamp = STORED_SCHEMA_VERSION if current.lineage == LINEAGE_FRESH else FENCE_LINEAGE_BASE + current.gate
        _write_lineage_stamp(cursor, current, stamp)
        return current

    return _in_write_transaction(cursor, migrate)


def schema_cookie(cursor) -> int:
    return cursor.execute("PRAGMA schema_version").fetchone()[0]


def restore_missing_fences(cursor, *, unchanged_since: Optional[int] = None) -> list:
    """Backstop after the heals and migrations: re-create any missing owned fence at this build's
    generation. Never drops anything; an unowned or colliding declaration still refuses.

    ``unchanged_since`` is the schema cookie read before the fence delta checked the fences:
    only DDL removes a trigger, and any committed or own DDL moves the cookie, so an equal
    cookie proves the fences are still the ones the delta found exact."""
    if unchanged_since is not None and schema_cookie(cursor) == unchanged_since:
        return []
    if fences_exact(cursor):
        return []

    def restore() -> list:
        owned, expected = owned_turn_fence_literals(cursor), _expected_definitions(cursor)
        if any(v != TURN_FENCE_GENERATION for v in owned.values()):
            raise RuntimeError("turn-fence triggers carry another generation after the fence delta")
        missing = _create_missing(cursor, owned, expected)
        _warn_restored(missing)
        return missing

    return _in_write_transaction(cursor, restore)


def table_fence_literals(cursor, table: str) -> dict:
    """Owned fence declarations on *table*, read before a rebuild RENAMEs it (after the RENAME
    SQLite rewrites their bodies onto the legacy name and they are no longer owned)."""
    names = set(turn_fence_trigger_definitions((table,)))
    return {name: literal for name, literal in owned_turn_fence_literals(cursor).items() if name in names}


def recreate_table_fences(cursor, table: str, literals: dict) -> None:
    """Re-declare *table*'s fences inside the rebuild's own transaction: the DROP of the legacy
    copy took them, and a commit without them would leave the table writable by any build."""
    for name, literal in literals.items():
        operation = name.rsplit("_", 1)[1].upper()
        cursor.execute(turn_fence_trigger_sql(table, operation, generation=literal))


def advance_lineage_stamp(cursor) -> bool:
    """Raise a fenced store's stamp to ``STORED_SCHEMA_VERSION``, forward only, while its fences
    are exact. Read-first, so a settled store takes no write lock; joins a caller's transaction."""
    if cursor.execute(
            "SELECT COUNT(*) = 1 AND SUM(typeof(version) = 'integer' AND version = ?) = 1 FROM schema_version",
            (STORED_SCHEMA_VERSION,)).fetchone()[0]:
        return False  # settled: skip the trigger parse the full decode needs
    lineage = decode_store_lineage(cursor)
    if lineage.lineage != LINEAGE_FENCED or lineage.stored >= STORED_SCHEMA_VERSION:
        return False

    def stamp() -> bool:
        current = decode_store_lineage(cursor)
        if current.lineage != LINEAGE_FENCED or current.stored >= STORED_SCHEMA_VERSION or not fences_exact(cursor):
            return False
        _write_lineage_stamp(cursor, current, STORED_SCHEMA_VERSION)
        return True

    return _in_write_transaction(cursor, stamp)
