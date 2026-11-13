"""Lineage admission and the one fork migration for state.db, decided before any connect-side DDL.

A fork build of this line stamps ``schema_version`` as ``FENCE_LINEAGE_BASE + SCHEMA_VERSION`` and
guards each governed table with turn-fence triggers whose generation literal names the build that
owns the store. Upstream and the older fork stamp plain integers (< 1000). Every open path asks
:func:`refuse_unadmitted_lineage` first. It decodes the lineage with SELECTs alone on a read-only
connection (``immutable=1`` while no sidecar exists), and a refused store comes out of the open
byte-identical: no WAL switch, no ``SCHEMA_SQL``, no chmod, no sidecar, no quarantine.

Admitted unchanged: fresh, missing, empty and pure-upstream stores (``schema_version`` <=
``SCHEMA_VERSION`` and no fork marker), which open exactly as before this module existed, and a store
this build fenced (``STORED_SCHEMA_VERSION`` with exactly this build's fences), which opens without
fence DDL. The live fork's store (1030 with exactly the fork's fences) is migrated by a writable
SessionDB open alone, in one transaction under the profile's ``gateway.lock`` lease
(:func:`fence_migration_admission`, :func:`migrate_fenced_store`); a writable raw opener hands it to
that open first (:func:`admit_raw_state_writer`), and a reader refuses it. Every other shape is refused.

A fence trigger aborts a governed write unless the writing connection's ``hermes_turn_fence_generation()``
equals the literal the store was fenced with, so every connection that writes a fenced store registers
the UDF (:func:`register_turn_fence_generation`) and is serialized per connection
(:func:`serialized_connection_factory`).
"""

from __future__ import annotations

import contextlib
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hermes_state_common import SCHEMA_VERSION
# Re-exported: the refusal API was published from this module first.
from hermes_state_errors import (
    SCHEMA_CAUSE_BUILD_TOO_OLD, SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, SCHEMA_CAUSE_VERSION_UNREADABLE,
    SCHEMA_INCOMPATIBLE_CAUSES, ForwardSchemaMigrationAdmissionError, IncompatibleSchemaError,
)

logger = logging.getLogger("hermes_state")

FENCE_LINEAGE_BASE = 1000
STORED_SCHEMA_VERSION = FENCE_LINEAGE_BASE + SCHEMA_VERSION
TURN_FENCE_GENERATION = STORED_SCHEMA_VERSION
FORK_LEGACY_GENERATIONS = frozenset({27, 28, 29})
FORK_BASE_UPSTREAM_GATE = 26
# The one fork store this build migrates: the live fork's schema 30 on the lineage base.
FORK_SCHEMA_VERSION = 30
FORK_STORED_SCHEMA_VERSION = FENCE_LINEAGE_BASE + FORK_SCHEMA_VERSION

TURN_FENCE_FUNCTION = "hermes_turn_fence_generation"
FENCE_OPERATIONS = ("INSERT", "UPDATE", "DELETE")
CORE_GOVERNED_TABLES = (
    "async_delegations", "compression_locks", "gateway_routing", "messages",
    "session_model_usage", "session_turn_leases", "sessions", "system_prompts",
)
# The fork's own tables, governed where they exist: its AFTER INSERT ON sessions trigger writes them.
SESSION_PROCESS_GOVERNED_TABLES = ("session_process_authorities", "session_process_reservations")
# Added by upstream schema 31-32. No fork build fenced them, so only this build's generation owns them.
TARGET_GOVERNED_TABLES = ("runtime_epoch", "session_admissions", "worker_executions", "worker_receipts")
ALL_GOVERNED_TABLES = CORE_GOVERNED_TABLES + SESSION_PROCESS_GOVERNED_TABLES + TARGET_GOVERNED_TABLES

LINEAGE_FRESH = "fresh"
LINEAGE_UPSTREAM = "upstream"
LINEAGE_FORK = "fork"
LINEAGE_FENCED = "fenced"
LINEAGE_DAMAGED = "damaged"

MIGRATION_NOT_ADMITTED = "migration not admitted in this build"
MIGRATION_NEEDS_WRITABLE_OPEN = "only a writable session-store open migrates it"
# How long an opener waits out a sibling's migration lease before refusing.
MIGRATION_LEASE_WAIT_S = 30.0
_MIGRATION_LEASE_POLL_S = 0.05


def schema_text(value) -> str:
    """sqlite_master text read as a BLOB: invalid UTF-8 in a schema must decode, not raise."""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else (value or "")


def _existing_tables(cursor) -> set:
    return {schema_text(row[0]) for row in cursor.execute(
        "SELECT CAST(name AS BLOB) FROM sqlite_master WHERE type = 'table'").fetchall()}


# ── Fence declarations and the UDF ──

def _fence_trigger_name(table: str, operation: str) -> str:
    return f"turn_fence_{table}_{operation.lower()}"


def _fence_declaration(table: str, operation: str, generation: int) -> str:
    # The fork's persisted declaration, character for character: ownership is proven by token
    # equality against it, and the migration writes exactly this text at this build's generation.
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


def _fence_names(tables) -> frozenset:
    return frozenset(_fence_trigger_name(table, op) for table in tables for op in FENCE_OPERATIONS)


def turn_fence_trigger_definitions(tables, *, generation: int = TURN_FENCE_GENERATION) -> dict:
    return {
        _fence_trigger_name(table, op): _fence_declaration(table, op, generation)
        for table in tables for op in FENCE_OPERATIONS
    }


def _fork_governed(tables) -> tuple:
    return CORE_GOVERNED_TABLES + tuple(t for t in SESSION_PROCESS_GOVERNED_TABLES if t in tables)


def _this_build_governed(tables) -> tuple:
    return _fork_governed(tables) + TARGET_GOVERNED_TABLES


def _turn_fence_generation() -> int:
    return TURN_FENCE_GENERATION


def register_turn_fence_generation(conn: sqlite3.Connection) -> None:
    """The UDF every governed write on a fenced store needs; without it the write fails "no such function"."""
    conn.create_function(TURN_FENCE_FUNCTION, 0, _turn_fence_generation, deterministic=True)


# ── Per-connection serialization for connections that run Python UDFs ──
#
# A fenced write runs ``hermes_turn_fence_generation()`` inside ``sqlite3_step``: thread A holds the
# connection mutex with the GIL released and waits for the GIL to run the callback, while thread B
# holds the GIL and waits for that mutex (bind, cursor description and fetch paths keep the GIL while
# taking it). The live fork's gateway sat in exactly this ABBA deadlock for 80+ minutes. At most one
# thread enters SQLite per connection, enforced at the connection factory; waiting on the RLock
# releases the GIL, so the callback thread can always finish and the cycle cannot form. Every Python
# UDF on such a connection (the fence UDF, hermes_state_timeline's) runs under that lock.
# ``interrupt()`` is deliberately not wrapped: it cancels another thread's in-flight statement.

class _SerializedCursor(sqlite3.Cursor):
    """Cursor whose SQLite entries run under the connection's RLock. Fetches are wrapped too: they
    step the VM while holding the GIL, which is the GIL-held mutex-wait leg of the deadlock."""

    def _serial(self):
        return self.connection._hermes_serial_lock

    def execute(self, *args, **kwargs):
        with self._serial():
            return super().execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        with self._serial():
            return super().executemany(*args, **kwargs)

    def executescript(self, *args, **kwargs):
        with self._serial():
            return super().executescript(*args, **kwargs)

    def fetchone(self):
        with self._serial():
            return super().fetchone()

    def fetchmany(self, *args, **kwargs):
        with self._serial():
            return super().fetchmany(*args, **kwargs)

    def fetchall(self):
        with self._serial():
            return super().fetchall()

    def __next__(self):
        with self._serial():
            return super().__next__()

    def close(self):
        with self._serial():
            return super().close()


class SerializedConnectionMixin:
    """Serialize every SQLite entry on one connection behind an RLock."""

    def __init__(self, *args, **kwargs):
        self._hermes_serial_lock = threading.RLock()
        super().__init__(*args, **kwargs)

    def cursor(self, factory=None):
        with self._hermes_serial_lock:
            return super().cursor(factory or _SerializedCursor)

    # Routed through self.cursor() instead of the C shortcuts: on Python 3.11 those build a plain
    # Cursor without calling the overridden cursor(), so ``conn.execute(...).fetchall()`` would fetch
    # unserialized.
    def execute(self, *args, **kwargs):
        with self._hermes_serial_lock:
            return self.cursor().execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        with self._hermes_serial_lock:
            return self.cursor().executemany(*args, **kwargs)

    def executescript(self, *args, **kwargs):
        with self._hermes_serial_lock:
            return self.cursor().executescript(*args, **kwargs)

    def commit(self):
        with self._hermes_serial_lock:
            return super().commit()

    def rollback(self):
        with self._hermes_serial_lock:
            return super().rollback()

    def close(self):
        with self._hermes_serial_lock:
            return super().close()

    def __exit__(self, *exc_info):
        with self._hermes_serial_lock:
            return super().__exit__(*exc_info)

    def create_function(self, *args, **kwargs):
        with self._hermes_serial_lock:
            return super().create_function(*args, **kwargs)

    def create_collation(self, *args, **kwargs):
        with self._hermes_serial_lock:
            return super().create_collation(*args, **kwargs)

    def create_aggregate(self, *args, **kwargs):
        with self._hermes_serial_lock:
            return super().create_aggregate(*args, **kwargs)

    def backup(self, *args, **kwargs):
        with self._hermes_serial_lock:
            return super().backup(*args, **kwargs)


class SerializedConnection(SerializedConnectionMixin, sqlite3.Connection):
    pass


_serialized_factory_cache: dict = {}


def serialized_connection_factory(factory: type = sqlite3.Connection) -> type:
    """Mix serialization into *factory* (mirrors ``sqlite_safe_read._tracking_factory``)."""
    if factory is sqlite3.Connection:
        return SerializedConnection
    if issubclass(factory, SerializedConnectionMixin):
        return factory
    cached = _serialized_factory_cache.get(factory)
    if cached is None:
        cached = type(f"Serialized{factory.__name__}", (SerializedConnectionMixin, factory), {})
        _serialized_factory_cache[factory] = cached
    return cached


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
    target_names = _fence_names(TARGET_GOVERNED_TABLES)
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
        if lowered in target_names and literal != STORED_SCHEMA_VERSION:
            raise IncompatibleSchemaError(
                cause=SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, expected_generation=STORED_SCHEMA_VERSION,
                actual_generation=literal, detail=f"trigger {name} on a table no fork build fenced",
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
    fence_names: frozenset = frozenset()  # lowercased names of the owned fence triggers
    tables: frozenset = frozenset()

    def _fenced_exactly(self, generation: int, governed: tuple) -> bool:
        return self.fence_literals == {generation} and self.fence_names == _fence_names(governed)

    @property
    def settled(self) -> bool:
        """This build's own store, which opens with no fence DDL: stamp 1033 and exactly this build's
        fences on every governed table, the 31-32 tables included."""
        return (self.lineage == LINEAGE_FENCED and self.stored == STORED_SCHEMA_VERSION
                and self._fenced_exactly(TURN_FENCE_GENERATION, _this_build_governed(self.tables)))

    @property
    def migratable(self) -> bool:
        """The live fork's store: stamp 1030 and exactly the fork's fences at 1030 on the tables it governed."""
        return (self.lineage == LINEAGE_FENCED and self.stored == FORK_STORED_SCHEMA_VERSION
                and self._fenced_exactly(FORK_STORED_SCHEMA_VERSION, _fork_governed(self.tables)))


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
    owned = owned_turn_fence_literals(cursor)
    literals = frozenset(owned.values())
    shape = {"fence_literals": literals, "fence_names": frozenset(name.lower() for name in owned),
             "tables": frozenset(tables)}
    if stored >= FENCE_LINEAGE_BASE:
        if stored > STORED_SCHEMA_VERSION:
            raise IncompatibleSchemaError(
                cause=SCHEMA_CAUSE_BUILD_TOO_OLD, expected_generation=STORED_SCHEMA_VERSION, actual_generation=stored,
            )
        return StoreLineage(LINEAGE_FENCED, stored, stored - FENCE_LINEAGE_BASE, **shape)
    if stored > SCHEMA_VERSION:
        raise IncompatibleSchemaError(
            cause=SCHEMA_CAUSE_BUILD_TOO_OLD, expected_generation=SCHEMA_VERSION, actual_generation=stored,
        )
    if _has_fork_markers(cursor, tables, literals):
        return StoreLineage(LINEAGE_FORK, stored, min(stored, FORK_BASE_UPSTREAM_GATE), **shape)
    return StoreLineage(LINEAGE_UPSTREAM, stored, stored, **shape)


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


def refuse_unadmitted_lineage(path, *, migrate: bool = False) -> StoreLineage:
    """Admit what this build opens; raise IncompatibleSchemaError for the rest.

    Fresh, missing, empty and damaged stores pass (damage goes to the opener's existing probes), as do
    upstream stamps <= SCHEMA_VERSION without fork markers or fences and this build's settled fenced
    store. The fork's 1030 store passes only with ``migrate`` (a writable SessionDB open, which then
    migrates it); newer stamps and unreadable versions refuse inside the decode."""
    lineage = probe_store_lineage(path)
    if lineage.lineage in (LINEAGE_FRESH, LINEAGE_DAMAGED):
        return lineage
    if lineage.lineage == LINEAGE_UPSTREAM and not lineage.fence_literals:
        return lineage
    if lineage.settled or (migrate and lineage.migratable):
        return lineage
    actual = max(lineage.fence_literals) if lineage.fence_literals else lineage.stored
    reason = MIGRATION_NEEDS_WRITABLE_OPEN if lineage.migratable else MIGRATION_NOT_ADMITTED
    raise IncompatibleSchemaError(
        cause=SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, expected_generation=STORED_SCHEMA_VERSION,
        actual_generation=actual, detail=f"{lineage.lineage} lineage at schema_version {lineage.stored}; {reason}",
    )


def admit_raw_state_writer(path) -> StoreLineage:
    """The lineage gate for a writable raw (non-SessionDB) opener of *path*, before it connects.

    Admits what :func:`refuse_unadmitted_lineage` admits. The fork's 1030 store is first opened by a
    writable SessionDB, which migrates it once under the ``gateway.lock`` lease (a sibling that migrated
    it meanwhile leaves nothing to do), then re-admitted: the raw opener's own WAL switch and
    ``initialize`` DDL only ever meet a settled store. A writable open that came back without migrating
    leaves the store refused, unchanged."""
    lineage = refuse_unadmitted_lineage(path, migrate=True)
    if not lineage.migratable:
        return lineage
    from hermes_state import SessionDB

    SessionDB(db_path=Path(path)).close()
    return refuse_unadmitted_lineage(path)


# ── The fork migration: 1030 -> 1033 under the admission lease, in one transaction ──

def _runtime_lease(home: Path):
    """``gateway.lock`` for the migration. A gateway that owns the profile in this process already holds
    it (a second descriptor here would conflict with that very lock), so its own lock serves."""
    from gateway.runtime_ownership import exclusive_maintenance, process_ownership
    from hermes_state_reconcile import BROWSE_RECONCILE_PURPOSE

    if process_ownership.owns(home):
        return contextlib.nullcontext()
    # The schema catch-up purpose: a gateway booting meanwhile waits this holder out
    # (reserve_waiting_out_transient_holders) instead of failing its start.
    return exclusive_maintenance([home], purpose=BROWSE_RECONCILE_PURPOSE)


@contextlib.contextmanager
def fence_migration_admission(path, lineage: StoreLineage):
    """Hold the profile's ``gateway.lock`` lease from before the open's first write to the end of the
    migration when *lineage* is the fork's store; yields whether the store still needs migrating.

    The fork's gateway holds that lock while it runs, so its store is never migrated under it: the open
    raises ForwardSchemaMigrationAdmissionError with nothing written. A sibling opener's lease is waited
    out (bounded); the re-admission under the lease then finds the store it migrated settled."""
    if not lineage.migratable:
        yield False
        return
    from gateway.runtime_ownership import OwnershipConflict, live_runtime_owner

    path = Path(path)
    lock_path = path.parent / "gateway.lock"
    deadline = time.monotonic() + MIGRATION_LEASE_WAIT_S
    with contextlib.ExitStack() as lease:
        while True:
            try:
                lease.enter_context(_runtime_lease(path.parent))
                break
            except OwnershipConflict as exc:
                if live_runtime_owner(path.parent) is not None:
                    raise ForwardSchemaMigrationAdmissionError(
                        stored=lineage.stored, lock_path=lock_path,
                        detail="a running gateway holds its runtime lock") from exc
                if time.monotonic() >= deadline:
                    raise ForwardSchemaMigrationAdmissionError(
                        stored=lineage.stored, lock_path=lock_path,
                        detail=f"another process held its runtime lock for over {MIGRATION_LEASE_WAIT_S:.0f}s",
                    ) from exc
                time.sleep(_MIGRATION_LEASE_POLL_S)
            except OSError as exc:
                raise ForwardSchemaMigrationAdmissionError(
                    stored=lineage.stored, lock_path=lock_path,
                    detail=f"its runtime lock cannot be used ({exc})") from exc
        # Re-admitted under the lease: the first read only chose this branch, and a sibling may have
        # migrated the store before the lease came free.
        yield refuse_unadmitted_lineage(path, migrate=True).migratable


def migrate_fenced_store(db, cursor) -> bool:
    """Migrate the fork's 1030 store on *db*'s writer connection to ``STORED_SCHEMA_VERSION``.

    One transaction, before ``_init_schema``: decode to logical schema 30 under the write lock, take the
    fork's fences off, bring the declared schema in (the 31-32 tables and columns, statement by
    statement), run this build's data migrations gated above 30 and only those, put this build's fences
    on every governed table, stamp. Any failure rolls all of it back, stamp included. False when the
    store needed nothing (a sibling migrated it since the probe)."""
    cursor.execute("BEGIN IMMEDIATE")
    try:
        migrated = _migrate_locked(db, cursor)
    except BaseException:
        with contextlib.suppress(sqlite3.Error):
            cursor.execute("ROLLBACK")
        raise
    cursor.execute("COMMIT")
    return migrated


def _migrate_locked(db, cursor) -> bool:
    from hermes_state_fts import drop_retired_cjk_index
    from hermes_state_schema import apply_declared_schema

    current = decode_store_lineage(cursor)
    if current.settled:
        return False
    if not current.migratable:
        raise IncompatibleSchemaError(
            cause=SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, expected_generation=STORED_SCHEMA_VERSION,
            actual_generation=current.stored,
            detail=f"{current.lineage} lineage at schema_version {current.stored}; {MIGRATION_NOT_ADMITTED}",
        )
    for name in sorted(current.fence_names):
        cursor.execute(f'DROP TRIGGER "{name}"')
    # _init_schema's own order (declared schema, retired CJK index, PK heals), joined to this transaction,
    # so its later run finds nothing to rebuild and cannot drop the fences put back below.
    apply_declared_schema(cursor)
    if drop_retired_cjk_index(cursor):
        logger.warning("Removed the retired messages_fts_cjk index while migrating state.db to generation %d",
                       STORED_SCHEMA_VERSION)
    db._heal_gateway_routing_pk(cursor)
    db._heal_session_model_usage_pk(cursor)
    db._run_data_migrations(cursor, current.gate, db._sqlite_supports_fts5(cursor))
    for sql in turn_fence_trigger_definitions(_this_build_governed(_existing_tables(cursor))).values():
        cursor.execute(sql)
    cursor.execute("UPDATE schema_version SET version = ?", (STORED_SCHEMA_VERSION,))
    if not decode_store_lineage(cursor).settled:
        raise RuntimeError("state.db fence migration did not leave exactly this build's fences and stamp")
    return True


def advance_schema_stamp(cursor) -> None:
    """Forward-only ``schema_version`` advance for a writer on an open store (optimize-storage's settle).

    The open admitted the store, so a fenced one carries this build's fences: its stamp moves only up to
    ``STORED_SCHEMA_VERSION``. Any other stamp keeps upstream's advance to ``SCHEMA_VERSION``. A plain
    stamp never lands on a fenced store, where it would read as an upstream store carrying fences."""
    target = STORED_SCHEMA_VERSION if decode_store_lineage(cursor).lineage == LINEAGE_FENCED else SCHEMA_VERSION
    cursor.execute("UPDATE schema_version SET version = ? WHERE version < ?", (target, target))
