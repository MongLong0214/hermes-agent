"""Read-only lineage admission for state.db, decided before any connect-side DDL, pragma or write.

A fork build of this line stamps ``schema_version`` as ``FENCE_LINEAGE_BASE + SCHEMA_VERSION`` and
guards each governed table with turn-fence triggers whose generation literal names the build that
owns the store. Upstream and the older fork stamp plain integers (< 1000). This build cannot migrate
a fork or fenced store yet, so every open path asks :func:`refuse_unadmitted_lineage` first. It
decodes the lineage with SELECTs alone on a read-only connection (``immutable=1`` while no sidecar
exists), and a refused store comes out of the open byte-identical: no WAL switch, no ``SCHEMA_SQL``,
no chmod, no sidecar, no quarantine.

Fresh, missing, empty and pure-upstream stores (``schema_version`` <= ``SCHEMA_VERSION`` and no fork
marker) are admitted unchanged, so they open exactly as they did before this module existed.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hermes_state_common import SCHEMA_VERSION

FENCE_LINEAGE_BASE = 1000
STORED_SCHEMA_VERSION = FENCE_LINEAGE_BASE + SCHEMA_VERSION
FORK_LEGACY_GENERATIONS = frozenset({27, 28, 29})
FORK_BASE_UPSTREAM_GATE = 26

TURN_FENCE_FUNCTION = "hermes_turn_fence_generation"
FENCE_OPERATIONS = ("INSERT", "UPDATE", "DELETE")
CORE_GOVERNED_TABLES = (
    "async_delegations", "compression_locks", "gateway_routing", "messages",
    "session_model_usage", "session_turn_leases", "sessions", "system_prompts",
)
SESSION_PROCESS_GOVERNED_TABLES = ("session_process_authorities", "session_process_reservations")
ALL_GOVERNED_TABLES = CORE_GOVERNED_TABLES + SESSION_PROCESS_GOVERNED_TABLES

LINEAGE_FRESH = "fresh"
LINEAGE_UPSTREAM = "upstream"
LINEAGE_FORK = "fork"
LINEAGE_FENCED = "fenced"
LINEAGE_DAMAGED = "damaged"

SCHEMA_CAUSE_BUILD_TOO_OLD = "BUILD_TOO_OLD"
SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH = "FENCE_GENERATION_MISMATCH"
SCHEMA_CAUSE_VERSION_UNREADABLE = "SCHEMA_VERSION_UNREADABLE"
SCHEMA_INCOMPATIBLE_CAUSES = frozenset({
    SCHEMA_CAUSE_BUILD_TOO_OLD, SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, SCHEMA_CAUSE_VERSION_UNREADABLE,
})
MIGRATION_NOT_ADMITTED = "migration not admitted in this build"

# A fence mismatch fires in either direction (older or newer writer), so its text never says "newer".
_INCOMPATIBLE_SCHEMA_HEADS = {
    SCHEMA_CAUSE_BUILD_TOO_OLD: "Session state schema is newer than this Hermes build",
    SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH: "Session state turn-fence generation does not match this Hermes build",
    SCHEMA_CAUSE_VERSION_UNREADABLE: "Session state does not record exactly one integer schema version",
}
_INCOMPATIBLE_SCHEMA_GENERATIONS = {
    SCHEMA_CAUSE_BUILD_TOO_OLD: " (expected generation {expected}, actual generation {actual})",
    SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH: " (build generation {expected}, stored generation {actual})",
}


class IncompatibleSchemaError(RuntimeError):
    """This build must not open the store at all. Deliberately NOT an ``sqlite3.DatabaseError``: every
    DatabaseError path in the open/repair stack treats the file as damaged and may quarantine, rebuild
    FTS or VACUUM it, which would destroy a store that is healthy for the build that wrote it."""

    code = "STATE_DB_SCHEMA_INCOMPATIBLE"

    def __init__(self, *, cause: str, expected_generation, actual_generation, detail: str = ""):
        if cause not in SCHEMA_INCOMPATIBLE_CAUSES:
            raise ValueError(f"unknown schema incompatibility cause: {cause!r}")
        self.cause = cause
        self.expected_generation = expected_generation
        self.actual_generation = actual_generation
        self.detail = detail
        suffix = f" ({detail})" if detail else ""
        generations = _INCOMPATIBLE_SCHEMA_GENERATIONS.get(cause, "").format(
            expected=expected_generation, actual=actual_generation)
        super().__init__(
            f"{_INCOMPATIBLE_SCHEMA_HEADS[cause]}{generations}{suffix}; refusing to open it, and nothing was changed.")


def schema_text(value) -> str:
    """sqlite_master text read as a BLOB: invalid UTF-8 in a schema must decode, not raise."""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else (value or "")


def _existing_tables(cursor) -> set:
    return {schema_text(row[0]) for row in cursor.execute(
        "SELECT CAST(name AS BLOB) FROM sqlite_master WHERE type = 'table'").fetchall()}


# ── Fence ownership: token-exact match against the fork's declaration, never a substring ──

# Strings stay opaque and comments are dropped before calls are recognised.
_TOKEN = re.compile(
    r"--[^\n]*(?:\n|$)|/\*.*?(?:\*/|$)|"
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]|"
    r"[\w$]+|!=|<>|\S",
    re.DOTALL,
)
_LITERAL_SLOT = 0


def _tokens(sql: str) -> tuple:
    result = []
    for match in _TOKEN.finditer(sql):
        text = match.group()
        if text.startswith(("--", "/*")):
            continue
        if text[0] == "'":
            result.append(("string", text[1:-1].replace("''", "'")))
        elif text[0] in ('"', "`", "["):
            quote = text[0]
            value = text[1:-1]
            if quote != "[":
                value = value.replace(quote * 2, quote)
            result.append(("word", value.lower()))
        else:
            result.append(("word", text.lower()))
    return tuple(result)


def _fence_trigger_name(table: str, operation: str) -> str:
    return f"turn_fence_{table}_{operation.lower()}"


def _fence_declaration(table: str, operation: str, generation: int) -> str:
    # Recognition only: the fork's persisted declaration, compared token by token. Nothing here creates it.
    return (
        f"CREATE TRIGGER {_fence_trigger_name(table, operation)} BEFORE {operation} ON {table} "
        "BEGIN "
        "SELECT CASE "
        f"WHEN typeof({TURN_FENCE_FUNCTION}()) != 'integer' "
        f"OR {TURN_FENCE_FUNCTION}() != {generation} "
        "THEN RAISE(ABORT, 'state DB generation incompatible') "
        "END; "
        "END"
    )


def _templates() -> dict:
    """{canonical trigger name: (template tokens, index of the generation literal)}."""
    templates = {}
    for table in ALL_GOVERNED_TABLES:
        for operation in FENCE_OPERATIONS:
            tokens = _tokens(_fence_declaration(table, operation, _LITERAL_SLOT))
            templates[_fence_trigger_name(table, operation)] = (tokens, tokens.index(("word", str(_LITERAL_SLOT))))
    return templates


def _declared_literal(tokens: tuple, template: tuple, slot: int):
    if len(tokens) != len(template):
        return None
    for index, (token, expected) in enumerate(zip(tokens, template)):
        if index == slot:
            if token[0] != "word" or not token[1].isdigit():
                return None
        elif token != expected:
            return None
    return int(tokens[slot][1])


def owned_turn_fence_literals(cursor) -> dict:
    """``{trigger name: generation literal}`` for every fence trigger a known build wrote.

    A call to the fence function or a ``turn_fence_`` name makes a trigger a candidate. One that is not
    token-exact to a known declaration refuses the store: BUILD_TOO_OLD for an exact fence above this
    line's generation, FENCE_GENERATION_MISMATCH otherwise (unknown body, name collision, or a
    generation no build of this line ever wrote)."""
    templates = _templates()
    authority_names = {
        name for name in templates
        if any(name.startswith(f"turn_fence_{table}_") for table in SESSION_PROCESS_GOVERNED_TABLES)
    }
    owned = {}
    for raw_name, raw_sql in cursor.execute(
            "SELECT CAST(name AS BLOB), CAST(sql AS BLOB) FROM sqlite_master WHERE type = 'trigger'").fetchall():
        name, tokens = schema_text(raw_name), _tokens(schema_text(raw_sql))
        lowered = name.lower()
        calls = any(
            token == ("word", TURN_FENCE_FUNCTION) and following == ("word", "(")
            for token, following in zip(tokens, tokens[1:])
        )
        if not (calls or lowered.startswith("turn_fence_")):
            continue
        template = templates.get(lowered)
        literal = _declared_literal(tokens, *template) if template else None
        if literal is None:
            raise IncompatibleSchemaError(
                cause=SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, expected_generation=STORED_SCHEMA_VERSION,
                actual_generation=None, detail=f"unowned trigger declaration {name}",
            )
        if literal > STORED_SCHEMA_VERSION:
            raise IncompatibleSchemaError(
                cause=SCHEMA_CAUSE_BUILD_TOO_OLD, expected_generation=STORED_SCHEMA_VERSION,
                actual_generation=literal, detail=f"trigger {name}",
            )
        # The fork first governed the authority tables at generation 28.
        legacy_ok = literal in FORK_LEGACY_GENERATIONS and not (literal == 27 and lowered in authority_names)
        older_fenced = FENCE_LINEAGE_BASE + FORK_BASE_UPSTREAM_GATE < literal < STORED_SCHEMA_VERSION
        if literal != STORED_SCHEMA_VERSION and not (legacy_ok or older_fenced):
            raise IncompatibleSchemaError(
                cause=SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, expected_generation=STORED_SCHEMA_VERSION,
                actual_generation=literal, detail=f"trigger {name}",
            )
        owned[name] = literal
    return owned


# ── Lineage decode ──

@dataclass(frozen=True)
class StoreLineage:
    lineage: str
    stored: Optional[int]
    gate: int
    fence_literals: frozenset = frozenset()


_FRESH = StoreLineage(LINEAGE_FRESH, None, 0)
_DAMAGED = StoreLineage(LINEAGE_DAMAGED, None, 0)


def decode_store_lineage(cursor) -> StoreLineage:
    """SELECT-only lineage decode on an open connection; raises IncompatibleSchemaError on refusal."""
    tables = _existing_tables(cursor)
    if "schema_version" not in tables:
        return _FRESH
    rows = cursor.execute(
        "SELECT CASE WHEN typeof(version) = 'integer' THEN version END, typeof(version) FROM schema_version").fetchall()
    # Zero rows is the mid-bootstrap shape (the table exists, the stamp is not inserted yet): fresh.
    if not rows:
        return _FRESH
    if len(rows) != 1 or rows[0][1] != "integer":
        raise IncompatibleSchemaError(
            cause=SCHEMA_CAUSE_VERSION_UNREADABLE, expected_generation=STORED_SCHEMA_VERSION,
            actual_generation=None, detail=f"{len(rows)} row(s), type {rows[0][1]}",
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


def _construct_probe_connection(database, *, factory=sqlite3.Connection, **kwargs) -> sqlite3.Connection:
    # The factory itself, not sqlite3.connect: tests patch sqlite3.connect to wrap or count an opener's
    # own connections, and a wrapped connection cannot be tracked. This read belongs to no opener, so
    # every opener keeps seeing exactly the connections it saw before the probe existed.
    return factory(database, **kwargs)


def _decode_at(uri: str, path: Path) -> StoreLineage:
    from hermes_state import _READ_BUSY_TIMEOUT_S

    # A read: under DELETE journaling it waits out another process's commit on the SessionDB read budget.
    kwargs = {"uri": True, "timeout": _READ_BUSY_TIMEOUT_S, "isolation_level": None}
    try:
        from hermes_cli.sqlite_safe_read import connect_tracked
    except ImportError:
        conn = _construct_probe_connection(uri, **kwargs)  # installs without hermes_cli: untracked
    else:
        # Tracked, so a concurrent byte-level header probe cannot open()/close() the file under it.
        conn = connect_tracked(uri, tracking_path=path, connect_fn=_construct_probe_connection, **kwargs)
    try:
        return decode_store_lineage(conn.cursor())
    except UnicodeDecodeError:
        # pysqlite could not decode SQLite's own error text: the schema holds invalid UTF-8 and its
        # lineage is unknowable. That is damage, which the opener's own probes are built to survive (#98924).
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
    # A mode=ro open of a WAL-mode file with no sidecars creates -wal/-shm and leaves them behind;
    # immutable=1 reads the main file only, which is sound exactly when no sidecar exists.
    if not Path(f"{path}-wal").exists() and not Path(f"{path}-journal").exists():
        try:
            return _decode_at(f"{uri}&immutable=1", path)
        except sqlite3.DatabaseError:
            # A rollback-journal writer that started after the sidecar check can tear an immutable
            # read; the locked read-only retry decides.
            pass
    return _decode_at(uri, path)


def refuse_unadmitted_lineage(path) -> StoreLineage:
    """Admit only what this build opens as upstream does; raise IncompatibleSchemaError for the rest.

    Fresh, missing, empty and damaged stores pass (damage goes to the opener's existing probes), as do
    upstream stamps <= SCHEMA_VERSION without fork markers or fences. Fork and fenced lineages cannot
    be migrated by this build yet; newer stamps and unreadable versions refuse inside the decode."""
    lineage = probe_store_lineage(path)
    if lineage.lineage in (LINEAGE_FRESH, LINEAGE_DAMAGED):
        return lineage
    if lineage.lineage == LINEAGE_UPSTREAM and not lineage.fence_literals:
        return lineage
    actual = max(lineage.fence_literals) if lineage.fence_literals else lineage.stored
    raise IncompatibleSchemaError(
        cause=SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, expected_generation=STORED_SCHEMA_VERSION,
        actual_generation=actual, detail=f"{lineage.lineage} lineage at schema_version {lineage.stored}; "
        f"{MIGRATION_NOT_ADMITTED}",
    )
