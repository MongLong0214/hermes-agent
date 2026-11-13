"""Read-only recovery reader for a generation-1033 state.db (the store the v0.23.1 line writes).

After cutover the newer line stamps state.db ``FENCE_LINEAGE_BASE + 33`` and fences every governed
table at that generation, the four tables its schema 31-32 added included. Every default open of
this build refuses such a store (BUILD_TOO_OLD), which is right for a writer and stays so. A code
rollback must still serve the conversations the newer build wrote, so ``SessionDB.open_recovery_reader``
admits exactly that store, read-only:

* admission is SELECT-only and runs on the serving connection itself: exactly one integer stamp equal
  to 1033, and turn-fence triggers token-exact to the 1033 declaration on exactly the tables that build
  governs. Anything else raises IncompatibleSchemaError and leaves the file untouched;
* the connection is ``mode=ro``, ``immutable=1`` while no -wal/-journal exists (no connection holds
  the store, and a plain ``mode=ro`` open of a WAL file would leave a -shm and -wal behind), so a write
  that bypasses SessionDB's write path still fails inside SQLite;
* SessionDB's write path raises StateRecoveryReadOnlyError;
* search runs only while messages_fts (and, for the trigram route, messages_fts_trigram) is declared
  exactly as this build declares it; otherwise ``search_unavailable_reason`` says why and
  ``search_messages`` raises StateRecoverySearchUnavailableError.

An immutable reader does not see writes made after it opened: recovery assumes the newer build is stopped.
"""

from __future__ import annotations

import errno
import logging
import re
import sqlite3
from pathlib import Path
from typing import Optional

from hermes_state import SessionDB
from hermes_state_common import FTS_SQL, FTS_TRIGRAM_SQL, LEGACY_FTS_SQL, LEGACY_FTS_TRIGRAM_SQL
from hermes_state_dbfile import _connect_tracked_db
from hermes_state_errors import (
    SCHEMA_CAUSE_BUILD_TOO_OLD, SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, SCHEMA_CAUSE_VERSION_UNREADABLE,
    IncompatibleSchemaError,
)
from hermes_state_fence import (
    CORE_GOVERNED_TABLES, FENCE_LINEAGE_BASE, FENCE_OPERATIONS, SESSION_PROCESS_GOVERNED_TABLES,
    TURN_FENCE_FUNCTION, schema_text, turn_fence_trigger_name, turn_fence_trigger_sql,
)
from hermes_state_fence_classifier import _declared_literal, _tokens
from hermes_state_holders import read_only_db_uri

logger = logging.getLogger("hermes_state")

RECOVERY_STORED_SCHEMA_VERSION = FENCE_LINEAGE_BASE + 33
# Added by upstream schema 31-32: the 1033 build fences them, no fork build ever did.
RECOVERY_TARGET_GOVERNED_TABLES = ("runtime_epoch", "session_admissions", "worker_executions", "worker_receipts")

# {table: ((DDL that declares it, its external-content view or None), ...)}: the layouts this build creates.
_FTS_LAYOUTS = {
    "messages_fts": ((FTS_SQL, "messages_fts_src"), (LEGACY_FTS_SQL, None)),
    "messages_fts_trigram": ((FTS_TRIGRAM_SQL, "messages_fts_trigram_src"), (LEGACY_FTS_TRIGRAM_SQL, None)),
}


class StateRecoveryReadOnlyError(sqlite3.OperationalError):
    """A write on a recovery reader. An OperationalError so callers' existing write-failure handling applies."""


class StateRecoverySearchUnavailableError(RuntimeError):
    """The store's FTS layout is not one this build reads; sessions and messages are still served."""


def _refuse(cause: str, actual, detail: str) -> IncompatibleSchemaError:
    return IncompatibleSchemaError(cause=cause, expected_generation=RECOVERY_STORED_SCHEMA_VERSION,
                                   actual_generation=actual, detail=f"recovery reader: {detail}")


def _fence_templates(tables: set) -> dict:
    """{fence name: (template tokens, literal slot)} for every table the 1033 build governs in *tables*."""
    governed = (CORE_GOVERNED_TABLES + tuple(t for t in SESSION_PROCESS_GOVERNED_TABLES if t in tables)
                + RECOVERY_TARGET_GOVERNED_TABLES)
    templates = {}
    for table in governed:
        for operation in FENCE_OPERATIONS:
            tokens = _tokens(turn_fence_trigger_sql(table, operation, generation=0))
            templates[turn_fence_trigger_name(table, operation)] = (tokens, tokens.index(("word", "0")))
    return templates


def _verify_fences(cursor, tables: set) -> None:
    templates = _fence_templates(tables)
    seen = set()
    for raw_name, raw_sql in cursor.execute(
            "SELECT CAST(name AS BLOB), CAST(sql AS BLOB) FROM sqlite_master WHERE type = 'trigger'").fetchall():
        name, tokens = schema_text(raw_name).lower(), _tokens(schema_text(raw_sql))
        calls = any(token == ("word", TURN_FENCE_FUNCTION) and following == ("word", "(")
                    for token, following in zip(tokens, tokens[1:]))
        if not (calls or name.startswith("turn_fence_")):
            continue
        literal = _declared_literal(tokens, *templates[name]) if name in templates else None
        if literal is not None and literal > RECOVERY_STORED_SCHEMA_VERSION:
            raise _refuse(SCHEMA_CAUSE_BUILD_TOO_OLD, literal, f"trigger {name}")
        if literal != RECOVERY_STORED_SCHEMA_VERSION:
            raise _refuse(SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, literal, f"trigger {name} is not a 1033 fence")
        seen.add(name)
    missing = sorted(set(templates) - seen)
    if missing:
        raise _refuse(SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, None, f"missing 1033 fence(s): {', '.join(missing)}")


def verify_recovery_store(cursor) -> None:
    """SELECT-only: raise IncompatibleSchemaError unless the store is exactly a settled 1033 store."""
    tables = {schema_text(row[0]) for row in cursor.execute(
        "SELECT CAST(name AS BLOB) FROM sqlite_master WHERE type = 'table'").fetchall()}
    rows = cursor.execute(
        "SELECT CASE WHEN typeof(version) = 'integer' THEN version END, typeof(version) FROM schema_version"
    ).fetchall() if "schema_version" in tables else []
    if len(rows) != 1 or rows[0][1] != "integer":
        raise _refuse(SCHEMA_CAUSE_VERSION_UNREADABLE, None, f"{len(rows)} schema_version row(s)")
    stored = rows[0][0]
    if stored > RECOVERY_STORED_SCHEMA_VERSION:
        raise _refuse(SCHEMA_CAUSE_BUILD_TOO_OLD, stored, "it reads generation 1033 only")
    if stored != RECOVERY_STORED_SCHEMA_VERSION:
        raise _refuse(SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, stored,
                      "it reads generation 1033 only; a store this build owns opens with the default SessionDB")
    _verify_fences(cursor, tables)


def _declaration(ddl: str, kind: str, name: str) -> tuple:
    statement = re.search(rf"CREATE {kind} IF NOT EXISTS {name}\b[^;]*", ddl).group()
    return _tokens(statement.replace(" IF NOT EXISTS", "", 1))


def fts_layout_unavailable_reason(cursor, table: str) -> Optional[str]:
    """None when *table* (and its content view) is declared exactly as one of this build's layouts."""
    names = {table} | {view for _, view in _FTS_LAYOUTS[table] if view}
    stored = {schema_text(name): _tokens(schema_text(sql)) for name, sql in cursor.execute(
        f"SELECT CAST(name AS BLOB), CAST(sql AS BLOB) FROM sqlite_master WHERE type IN ('table', 'view') "
        f"AND name IN ({', '.join('?' for _ in names)})", tuple(names)).fetchall()}
    if table not in stored:
        return f"{table} is absent"
    for ddl, view in _FTS_LAYOUTS[table]:
        expected = {table: _declaration(ddl, "VIRTUAL TABLE", table)}
        if view:
            expected[view] = _declaration(ddl, "VIEW", view)
        if all(stored.get(name) == tokens for name, tokens in expected.items()):
            return None
    return f"{table} is not declared as this build declares it"


class RecoverySessionDB(SessionDB):
    """SessionDB over a generation-1033 store, read-only. See the module docstring."""

    def __init__(self, db_path) -> None:
        self.search_unavailable_reason: Optional[str] = None
        self._recovery_immutable = False
        super().__init__(db_path=Path(db_path), read_only=True)

    def _admit_read_only_lineage(self) -> None:
        # The lineage is verified on the serving connection itself (_connect_read_only), not on a probe.
        if not self.db_path.exists():
            raise FileNotFoundError(errno.ENOENT, "recovery reader: no state.db", str(self.db_path))
        self._recovery_immutable = not any(
            Path(f"{self.db_path}{suffix}").exists() for suffix in ("-wal", "-journal"))

    def _connect_read_only(self, timeout: float) -> sqlite3.Connection:
        uri = read_only_db_uri(self.db_path) + ("&immutable=1" if self._recovery_immutable else "")
        conn = _connect_tracked_db(uri, tracking_path=self.db_path, uri=True, check_same_thread=False,
                                   timeout=timeout, isolation_level=None)
        try:
            verify_recovery_store(conn.cursor())
        except BaseException:
            conn.close()
            raise
        conn.row_factory = sqlite3.Row
        return conn

    def _fts_table_probe(self, cursor: sqlite3.Cursor, table_name: str) -> Optional[bool]:
        # A layout this build did not create is never queried, not even by the open's probe.
        reason = fts_layout_unavailable_reason(cursor, table_name) if table_name in _FTS_LAYOUTS else None
        if reason is not None:
            if table_name == "messages_fts":
                self.search_unavailable_reason = reason
            return False
        return super()._fts_table_probe(cursor, table_name)

    def _open_read_only(self) -> None:
        super()._open_read_only()
        if not self._fts_enabled and self.search_unavailable_reason is None:
            self.search_unavailable_reason = "messages_fts is absent or FTS5 is unavailable on this runtime"
        logger.warning(
            "state.db at %s is open in read-only recovery mode (generation %d): every write is refused%s",
            self.db_path, RECOVERY_STORED_SCHEMA_VERSION,
            f"; search is unavailable ({self.search_unavailable_reason})" if self.search_unavailable_reason else "")

    def _execute_write(self, fn, patience_s: Optional[float] = None):
        raise StateRecoveryReadOnlyError(
            f"state.db at {self.db_path} is open in read-only recovery mode (generation "
            f"{RECOVERY_STORED_SCHEMA_VERSION}, written by a newer Hermes build); this build refuses every write to it")

    def search_messages(self, query: str, *args, **kwargs):
        if self.search_unavailable_reason is not None:
            raise StateRecoverySearchUnavailableError(
                f"state.db recovery reader: search is unavailable ({self.search_unavailable_reason})")
        return super().search_messages(query, *args, **kwargs)
