"""One bounded, offline way back off the turn-fence generation barrier.

WHAT THIS IS NOT
    Not a compatibility fallback. Not an automatic trigger drop on open. Not a
    silent admission of an old writer. A binary from before the barrier being
    unable to write a fenced store is INTENTIONAL and required, and nothing in
    this module runs unless a person asks for it by name, on a store nobody is
    using.

WHY IT EXISTS ANYWAY
    Because the alternative that was actually on offer — an operator typing
    ``DROP TRIGGER`` nine times into whatever database they think is the right
    one — is not a rollback story. It has no backup, no check that the store is
    idle, no check that it is removing the surface it believes it is removing,
    and no atomicity: eight of nine dropped leaves a store fenced against some
    writes and not others, which is worse than either end state.

    So this is the same operation with the four things that were missing:

    1. FAIL-CLOSED ON LIVENESS. Refused while any conversation in the store is
       owned by a live turn, and refused if exclusive ownership of the file
       cannot be taken. Rolling back mid-turn hands the conversation to a
       binary that has never heard of the lease, which is the exact interleave
       the barrier was installed to stop.
    2. AN EXECUTABLE BACKUP. Not a sentence in a runbook. The store and its
       sidecars are copied, the copy is opened, and its ``sessions`` /
       ``messages`` / ``session_turn_leases`` rows are compared against the
       original before any DDL runs. A backup that cannot be read is not data
       preservation.
    3. A GENERATED TRIGGER SET. :func:`rollback_trigger_names` derives from
       ``TURN_FENCE_SURFACE``, the same declaration the triggers are created
       from. There is no list of names in this file. A surface that grows and a
       rollback that does not is how a store ends up half-fenced, and a
       hand-copied list is right exactly once.
    4. VERIFY, THEN MUTATE, ALL OR NOTHING. The installed surface is compared
       against the expected one INSIDE the exclusive transaction, before the
       first ``DROP``; anything unexpected refuses the whole operation, and the
       drops commit together or not at all.

THE RETURN LEG
    Rollback is not a one-way door. ``_init_schema`` creates the triggers with
    ``CREATE TRIGGER IF NOT EXISTS`` on every open, so a current binary
    reopening a rolled-back store puts the fence back and the old binary is
    refused again. That is a property of the store, not a favour this module
    does, and it is pinned in ``tests/state/test_turn_fence_rollback.py``.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any

import hermes_state_common
from hermes_state import SessionDB, SessionTurnLeaseLostError

#: The sidecars SQLite may have left beside the main file. Copied with it so the
#: backup is the whole store rather than the part that happened to be flushed.
_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")

#: The tables whose rows this operation must not change. Derived from the fence
#: surface rather than typed, for the same reason the trigger names are: the
#: thing a rollback must preserve is exactly the thing the fence was protecting.
def _protected_tables() -> tuple[str, ...]:
    return tuple(
        sorted({table for table, _op in hermes_state_common.TURN_FENCE_SURFACE})
    )


class TurnFenceRollbackRefused(RuntimeError):
    """The rollback did not run, and the store was not touched.

    Every raise site in this module happens BEFORE any DDL, or inside a
    transaction that is rolled back — so catching this means the store is
    exactly as it was found.
    """


def rollback_trigger_names() -> tuple[str, ...]:
    """The triggers a rollback removes, GENERATED from the declaration.

    Read through the module rather than bound at import so that moving
    ``TURN_FENCE_SURFACE`` moves this with it. That is not a testing
    convenience: it is the property being claimed. A rollback carrying its own
    nine names is correct on the day it is written and silently incomplete on
    the day the surface grows, and nothing would notice until a store was left
    fenced against some writes and not others.
    """
    declared = getattr(hermes_state_common, "TURN_FENCE_TRIGGERS", None)
    if declared:
        return tuple(declared)
    return tuple(
        hermes_state_common.turn_fence_trigger_name(table, operation)
        for table, operation in hermes_state_common.TURN_FENCE_SURFACE
    )


def _installed_fence_triggers(store_path: Path) -> list[str]:
    """The fence triggers a store carries, read WITHOUT opening it as a store.

    A plain read-only connection, deliberately. ``SessionDB.__init__`` runs
    schema init, and schema init creates the fence triggers with
    ``CREATE TRIGGER IF NOT EXISTS`` — so merely asking the question through
    SessionDB HEALS the answer. A store missing one trigger would be silently
    completed and then reported as carrying the expected surface, and the
    verify-before-mutate step would be checking a surface this tool had just
    installed rather than the one it found. That was not a hypothesis: it is
    what the first version of this module did, and
    ``test_rollback_verifies_the_installed_surface_before_changing_anything``
    is what noticed.
    """
    conn = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    try:
        return sorted(
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                "AND name LIKE 'hermes_turn_fence_%'"
            ).fetchall()
        )
    finally:
        conn.close()


def _refuse_unexpected_surface(store_path: Path, installed, expected) -> None:
    if list(installed) == list(expected):
        return
    missing = sorted(set(expected) - set(installed))
    unexpected = sorted(set(installed) - set(expected))
    raise TurnFenceRollbackRefused(
        f"{store_path} does not carry the fence surface this binary declares, "
        f"so the rollback would be removing something it does not understand. "
        f"missing={missing} unexpected={unexpected}. Nothing was changed"
    )


def _canonical_rows(conn: sqlite3.Connection) -> dict[str, list[tuple]]:
    """Every row of every fenced table, ordered, as plain tuples."""
    rows: dict[str, list[tuple]] = {}
    for table in _protected_tables():
        try:
            cursor = conn.execute(f'SELECT * FROM "{table}"')
        except sqlite3.DatabaseError as exc:
            raise TurnFenceRollbackRefused(
                f"could not read {table} to prove the rollback preserved it: {exc}"
            ) from exc
        rows[table] = sorted(tuple(row) for row in cursor.fetchall())
    return rows


def _refuse_if_live(store_path: Path) -> None:
    """Refuse unless every conversation in the store is free.

    Delegates to :meth:`SessionDB.offline_rebuild`, which is the one predicate
    in the tree for "is this store idle enough to rewrite wholesale". Asking
    the question a second way here would let the two answers drift, and the
    interesting drift is the one where this module says idle and the store
    disagrees.
    """
    db = SessionDB(db_path=store_path)
    try:
        with db.offline_rebuild(reason="turn-fence rollback"):
            pass
    except SessionTurnLeaseLostError as exc:
        raise TurnFenceRollbackRefused(
            f"refusing to roll back the turn fence on {store_path}: {exc}"
        ) from exc
    finally:
        db.close()


def _make_verified_backup(store_path: Path, backup_path: Path) -> dict[str, Any]:
    """Copy the store and READ THE COPY BACK before anything is changed.

    The verification is the point. ``shutil.copyfile`` succeeding says a file
    exists at the destination; it says nothing about whether that file opens,
    or whether the rows an operator would need to restore are in it. So the
    backup is opened and compared row-for-row against the source, and a
    mismatch refuses the rollback rather than being logged.
    """
    if backup_path.exists():
        raise TurnFenceRollbackRefused(
            f"the backup path {backup_path} already exists; refusing to "
            "overwrite it. Name a path that does not exist — a rollback whose "
            "backup step destroys an earlier backup is not data preservation"
        )
    if not backup_path.parent.is_dir():
        raise TurnFenceRollbackRefused(
            f"the backup directory {backup_path.parent} does not exist, so no "
            "backup can be written and the rollback will not proceed"
        )

    source_conn = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    try:
        expected = _canonical_rows(source_conn)
    finally:
        source_conn.close()

    copied: list[str] = []
    try:
        shutil.copyfile(store_path, backup_path)
        copied.append(backup_path.name)
        for suffix in _SIDECAR_SUFFIXES:
            sidecar = store_path.with_name(store_path.name + suffix)
            if sidecar.is_file():
                shutil.copyfile(
                    sidecar, backup_path.with_name(backup_path.name + suffix)
                )
                copied.append(backup_path.name + suffix)
    except OSError as exc:
        raise TurnFenceRollbackRefused(
            f"the backup of {store_path} could not be written to "
            f"{backup_path}: {exc}"
        ) from exc

    try:
        backup_conn = sqlite3.connect(f"file:{backup_path}?mode=ro", uri=True)
    except sqlite3.DatabaseError as exc:
        raise TurnFenceRollbackRefused(
            f"the backup at {backup_path} does not open: {exc}"
        ) from exc
    try:
        found = _canonical_rows(backup_conn)
    finally:
        backup_conn.close()

    if found != expected:
        differing = sorted(
            table for table in expected if expected[table] != found.get(table)
        )
        raise TurnFenceRollbackRefused(
            f"the backup at {backup_path} does not reproduce the store: "
            f"{', '.join(differing)} differ. Refusing to roll back — a backup "
            "that cannot be read back is not a backup"
        )

    return {
        "path": str(backup_path),
        "files": copied,
        "verified": True,
        "rows": {table: len(values) for table, values in expected.items()},
    }


def rollback_turn_fence(
    store_path: Path,
    *,
    backup_path: Path,
) -> dict[str, Any]:
    """Remove the turn-fence triggers from an IDLE store, offline, atomically.

    The order is the contract, and each step can only refuse:

    1. the surface the store carries is read on a READ-ONLY connection and
       compared to the generated one. First, before anything else, because
       every later step opens the store in a way that would repair it — see
       :func:`_installed_fence_triggers`;
    2. every conversation in the store is free (no live turn owns one);
    3. a backup is written and read back row-for-row;
    4. the file is taken EXCLUSIVE — a store another process is reading is not
       a store that is safe to downgrade;
    5. the surface is compared AGAIN inside that transaction, because steps 2
       and 3 opened the file and the answer from step 1 is a snapshot;
    6. every trigger is dropped in that same transaction, then committed.

    Returns a report. Raises :class:`TurnFenceRollbackRefused` — and leaves the
    store untouched — at every other outcome.
    """
    store_path = Path(store_path)
    backup_path = Path(backup_path)
    if not store_path.is_file():
        raise TurnFenceRollbackRefused(f"{store_path} is not a file")

    expected = sorted(rollback_trigger_names())
    _refuse_unexpected_surface(
        store_path, _installed_fence_triggers(store_path), expected
    )
    _refuse_if_live(store_path)
    backup = _make_verified_backup(store_path, backup_path)

    conn = sqlite3.connect(str(store_path), isolation_level=None, timeout=1.0)
    try:
        try:
            # EXCLUSIVE, not IMMEDIATE: a reader mid-query on a store whose
            # triggers are about to vanish is a reader whose next statement
            # prepares against a different schema. If the lock cannot be taken
            # the store is in use and this is not an offline rollback.
            conn.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            raise TurnFenceRollbackRefused(
                f"exclusive ownership of {store_path} could not be "
                f"established, so it is not offline: {exc}"
            ) from exc
        try:
            installed = sorted(
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                    "AND name LIKE 'hermes_turn_fence_%'"
                ).fetchall()
            )
            _refuse_unexpected_surface(store_path, installed, expected)
            for name in expected:
                conn.execute(f"DROP TRIGGER {name}")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()

    return {
        "store": str(store_path),
        "backup": backup,
        "generation": hermes_state_common.TURN_FENCE_GENERATION,
        "dropped_triggers": expected,
    }
