"""One bounded, offline way back off the turn-fence generation barrier.

WHAT THIS IS NOT
    Not a compatibility fallback. Not an automatic trigger drop on open. Not a
    silent admission of an old writer. A binary from before the barrier being
    unable to write a fenced store is INTENTIONAL and required, and nothing in
    this module runs unless a person asks for it by name, on a store nobody is
    using.

WHY IT EXISTS ANYWAY
    Because the alternative that was actually on offer — an operator typing
    ``DROP TRIGGER`` once per surface entry into whatever database they think
    is the right one — is not a rollback story. It has no backup, no check that
    the store is idle, no check that it is removing the surface it believes it
    is removing, and no atomicity: all-but-one dropped leaves a store fenced
    against some writes and not others, which is worse than either end state.

    How many that is has already changed once — the surface went from three
    tables to eight when the adjunct tables were closed — which is the whole
    reason nothing here counts them.

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
       from. There is no list of names in this file and no count of them. A
       surface that grows and a rollback that does not is how a store ends up
       half-fenced, and a hand-copied list is right exactly once — the surface
       has since grown from nine entries to twenty-four, and this file needed
       no edit for it.
    4. VERIFY, THEN MUTATE, ALL OR NOTHING. The installed surface is compared
       against the expected one INSIDE the exclusive transaction, before the
       first ``DROP``; anything unexpected refuses the whole operation, and the
       drops commit together or not at all.

WHO CALLS IT
    ``hermes sessions fence-rollback`` — :mod:`hermes_cli.session_fence_rollback_cmd`
    — and nothing else in the tree. A library function is not an operator
    surface: it has no exit code, no way to name which precondition stopped it,
    and no rehearsal, so the runbook entry for it is a Python one-liner. The
    verb supplies those three and re-derives none of this: it hands
    :func:`preflight_turn_fence_rollback` and :func:`rollback_turn_fence` a
    store and a backup path the operator typed, and reports what they return.

THE RETURN LEG
    Rollback is not a one-way door. ``_init_schema`` creates the triggers with
    ``CREATE TRIGGER IF NOT EXISTS`` on every open, so a current binary
    reopening a rolled-back store puts the fence back and the old binary is
    refused again. That is a property of the store, not a favour this module
    does, and it is pinned in ``tests/state/test_turn_fence_rollback.py``.
"""

from __future__ import annotations

import hashlib
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


#: The pre-flight checks, in the order they run. An operator reading a refusal
#: needs to know how far it got: "the surface matched and then a turn was live"
#: and "the file is not this fence at all" are different next moves.
PREFLIGHT_CHECKS = (
    "target_present",
    "surface_verified",
    "offline_verified",
    "backup_path_available",
)


def _blank_preflight() -> dict[str, bool]:
    return {name: False for name in PREFLIGHT_CHECKS}


class TurnFenceRollbackRefused(RuntimeError):
    """The rollback did not run, and the store was not touched.

    Every raise site in this module happens BEFORE any DDL, or inside a
    transaction that is rolled back — so catching this means the store is
    exactly as it was found.

    CARRIES A REASON, NOT JUST A SENTENCE
        Every raise site tags itself with a stable ``reason``. An operator — or
        the script driving them — must be able to tell WHICH refusal they hit,
        and the only thing a caller could otherwise key on is the message text,
        which is exactly the check that goes stale the day someone improves the
        wording. ``preflight`` says how far the pre-flight got before this
        refusal, which is the other half of the same question.
    """

    def __init__(self, message: str, *, reason: str = "refused") -> None:
        super().__init__(message)
        self.reason = reason
        self.preflight: dict[str, bool] = _blank_preflight()


def rollback_trigger_names() -> tuple[str, ...]:
    """The triggers a rollback removes, GENERATED from the declaration.

    Read through the module rather than bound at import so that moving
    ``TURN_FENCE_SURFACE`` moves this with it. That is not a testing
    convenience: it is the property being claimed. A rollback carrying its own
    copy of the names is correct on the day it is written and silently
    incomplete on the day the surface grows — which has now happened, from nine
    entries to twenty-four — and nothing would notice until a store was left
    fenced against some writes and not others.
    """
    declared = getattr(hermes_state_common, "TURN_FENCE_TRIGGERS", None)
    if declared:
        return tuple(declared)
    return tuple(
        hermes_state_common.turn_fence_trigger_name(table, operation)
        for table, operation in hermes_state_common.TURN_FENCE_SURFACE
    )


def _installed_fence_triggers(store_path: Path, *, report_as: Path = None) -> list[str]:
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

    ``mode=ro`` IS NOT "TOUCHES NOTHING". It makes the MAIN FILE read-only and
    says nothing about the sidecars: against a ``journal_mode=wal`` store with
    no ``-wal`` beside it, this connection CREATES ``-wal`` and ``-shm``, and
    whether they survive its close differs by SQLite build. So a caller that
    has promised to leave a directory as it found it must point this at a copy
    — see :func:`rehearse_turn_fence_rollback` — and pass *report_as* so the
    refusal still names the store the operator typed rather than the copy.
    """
    reported = Path(report_as) if report_as is not None else store_path
    try:
        conn = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    except sqlite3.DatabaseError as exc:
        raise TurnFenceRollbackRefused(
            f"{reported} could not be opened as a database at all: {exc}. "
            "Nothing was changed",
            reason="store-unreadable",
        ) from exc
    try:
        return sorted(
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                "AND name LIKE 'hermes_turn_fence_%'"
            ).fetchall()
        )
    except sqlite3.DatabaseError as exc:
        # A file an operator renamed, a truncated copy, a directory. This is a
        # DIFFERENT next move from "a database that is not this fence", so it
        # gets a different reason rather than being folded into the mismatch.
        raise TurnFenceRollbackRefused(
            f"{reported} does not read as a SQLite database: {exc}. "
            "Nothing was changed",
            reason="store-unreadable",
        ) from exc
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
        f"missing={missing} unexpected={unexpected}. Nothing was changed",
        reason="surface-mismatch",
    )


def _canonical_rows(conn: sqlite3.Connection) -> dict[str, list[tuple]]:
    """Every row of every fenced table, ordered, as plain tuples."""
    rows: dict[str, list[tuple]] = {}
    for table in _protected_tables():
        try:
            cursor = conn.execute(f'SELECT * FROM "{table}"')
        except sqlite3.DatabaseError as exc:
            raise TurnFenceRollbackRefused(
                f"could not read {table} to prove the rollback preserved it: {exc}",
                reason="store-unreadable",
            ) from exc
        rows[table] = sorted(tuple(row) for row in cursor.fetchall())
    return rows


def _live_holders_inside_the_transaction(conn, identity_path: Path) -> list[str]:
    """Conversations a live turn owns, decided ON *conn*, INSIDE its transaction.

    THE POINT IS THE BOUNDARY, NOT THE ANSWER
        Every writer in this program is required to check root, holder and
        epoch in the SAME transaction as the DML it is admitting. The rollback
        removes that fence, so it may not be held to a weaker standard than the
        thing it removes. Deciding liveness in the pre-flight and using that
        answer later leaves a window: a turn acquired after the check and
        before the ``DROP`` is invisible, and the fence comes off underneath a
        live conversation. That was reproduced, not theorised — a child that
        took the lease between the backup and ``BEGIN EXCLUSIVE`` watched all
        twenty-four triggers go, and the verb reported success.

        Called from inside ``BEGIN EXCLUSIVE``, this closes it from both sides.
        A contender that acquired BEFORE the lock has a row this reads and
        refuses on. A contender that tries AFTER it must write
        ``session_turn_leases`` to acquire, and cannot until this transaction
        ends.

        The file lock alone would not do it. An idle connection does not
        prevent ``BEGIN EXCLUSIVE``, so "I took the lock" says nothing about
        whether a conversation is owned; the exclusion has to be over the
        LEASE, which is why the rows are read here rather than inferred from
        the lock.

    THE PREDICATE IS NOT REIMPLEMENTED
        :meth:`SessionDB._turn_lease_row_is_free` is called directly, unbound,
        against a stand-in that carries only the one attribute it reads
        (``db_path``, for the process-local grant registry). Instantiating a
        ``SessionDB`` here is not an option — its schema init would write, and
        it would write to a store this transaction holds EXCLUSIVE — but a
        second copy of the predicate is worse than either. Two opinions about
        what a lease row means is the drift this program keeps paying for.
    """
    from types import SimpleNamespace

    from hermes_state import SessionDB

    previous = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT conversation_id, {SessionDB._TURN_LEASE_COLUMNS} "
            "FROM session_turn_leases "
            "WHERE holder IS NOT NULL AND holder != ''"
        ).fetchall()
    finally:
        conn.row_factory = previous

    stand_in = SimpleNamespace(db_path=Path(identity_path))
    return sorted(
        str(row["conversation_id"])
        for row in rows
        if not SessionDB._turn_lease_row_is_free(
            stand_in, str(row["conversation_id"]), row
        )
    )


def _refuse_if_live(store_path: Path, *, report_as: Path = None) -> None:
    """Refuse unless every conversation in the store is free.

    Delegates to :meth:`SessionDB.offline_rebuild`, which is the one predicate
    in the tree for "is this store idle enough to rewrite wholesale". Asking
    the question a second way here would let the two answers drift, and the
    interesting drift is the one where this module says idle and the store
    disagrees.

    NEVER POINTED AT THE OPERATOR'S STORE. ``SessionDB(db_path=…)`` runs schema
    init, and schema init WRITES — it re-creates triggers ``IF NOT EXISTS`` and
    normalises ``messages.active``. Asking "is this store idle" through it
    therefore initialises the store it is asking about, and a refusal that has
    already rewritten the file cannot claim the file is as it was found. That
    was measured: a ``live-turn`` refusal moved ``state.db``'s digest on both
    SQLite builds. So the caller hands this a byte-faithful COPY and names the
    real store in *report_as*.

    This is an early, cheap answer and NOT the authority. The decision the
    rollback acts on is :func:`_live_holders_inside_the_transaction`, taken
    under the same lock as the DDL.
    """
    reported = Path(report_as) if report_as is not None else store_path
    db = SessionDB(db_path=store_path)
    try:
        with db.offline_rebuild(reason="turn-fence rollback"):
            pass
    except SessionTurnLeaseLostError as exc:
        raise TurnFenceRollbackRefused(
            f"refusing to roll back the turn fence on {reported}: {exc}",
            reason="live-turn",
        ) from exc
    finally:
        db.close()


def _refuse_unusable_backup_path(backup_path: Path) -> None:
    """The backup destination, checked before the store is opened for writing.

    Split out of :func:`_make_verified_backup` so the pre-flight can answer
    "would this run proceed" without writing anything.

    THIS IS THE MESSAGE, NOT THE GUARANTEE. What actually makes "never
    clobber" true is the ``O_CREAT | O_EXCL`` acquisition in
    :func:`_copy_onto_a_destination_we_created`; a check here can only be
    stale by the time the write happens. It exists so the common case refuses
    early, cheaply, and with a sentence that says which file is in the way.

    A BACKUP IS A FAMILY OF FILES, so the whole family is checked. An operator
    with an orphaned ``backup.db-wal`` from a previous attempt and no
    ``backup.db`` is one WAL-mode store away from having that orphan
    overwritten, and "the path you named does not exist" would have been true
    and useless.
    """
    occupied = [
        str(candidate)
        for candidate in [backup_path]
        + [
            backup_path.with_name(backup_path.name + suffix)
            for suffix in _SIDECAR_SUFFIXES
        ]
        if candidate.exists()
    ]
    if occupied:
        raise TurnFenceRollbackRefused(
            f"the backup destination is already occupied by {occupied}; "
            "refusing to overwrite it. Name a path whose whole family — the "
            "file and its -wal/-shm/-journal siblings — does not exist. A "
            "rollback whose backup step destroys an earlier backup is not "
            "data preservation",
            reason="backup-exists",
        )
    if not backup_path.parent.is_dir():
        raise TurnFenceRollbackRefused(
            f"the backup directory {backup_path.parent} does not exist, so no "
            "backup can be written and the rollback will not proceed",
            reason="backup-directory-missing",
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class BoundTarget:
    """The store this run acts on, bound by INODE rather than by name.

    A PATH IS A NAME, NOT AN IDENTITY
        Every check in this module used to re-resolve ``store_path``, so the
        file inspected and the file mutated were only assumed to be the same
        one. They need not be: rename the store away between the pre-flight and
        the final open, drop a DIFFERENT valid fenced store at the same path,
        and every consistency check passes — the surface matches, the store is
        idle, the generation is right — while the backup describes the store
        that left and the twenty-four drops land on the store that arrived.
        Reproduced exactly that way.

        Re-checking the surface and the liveness inside the transaction does
        not help. Those protect CONSISTENCY, and the substitute is perfectly
        consistent. What was missing is IDENTITY.

        So the target is opened once, with ``O_NOFOLLOW`` so a symlink cannot
        stand in for it, and the descriptor is held for the whole operation.
        Holding it pins the inode: it cannot be recycled underneath us even if
        the name is unlinked. ``(st_dev, st_ino)`` from that descriptor is the
        identity every later step is measured against.

    A FAMILY, NOT A FILE — AND WHY POINT CHECKS ARE NOT ENOUGH
        Binding the main file while resolving ``-wal``/``-shm`` by pathname
        leaves the set unbound. Two stores can share a byte-identical
        checkpointed main file and carry DIFFERENT committed WAL contents, so a
        pathname swap that lands between two identity checks — and is undone
        before the second — produces a backup of A's main and B's WAL that
        reads back as B's rows, while the transaction commits against A. The
        point checks see A both times and report nothing.

        Two point observations with a window between them is the defect, not
        the fix. So :meth:`bind_family` opens EVERY member and the bytes are
        read from those descriptors; no name is resolved again during a copy.
        A member whose name resolves to an inode this did not bind, or a member
        that appears after binding, is a refusal.

    WHAT IT STILL CANNOT PROMISE, STATED AS A REFUSAL AND NOT AS A CAVEAT
        ``sqlite3`` opens by pathname and there is no descriptor-based open to
        hand it, so nothing here can prove the CONNECTION is on the bound
        inode. ``Connection.backup()`` would have derived the snapshot from the
        transaction itself; measured, it blocks indefinitely and
        uninterruptibly when the source connection holds ``BEGIN EXCLUSIVE``,
        so that door is shut.

        What is done instead: the connection is opened, the identity is
        verified, the family is bound UNDER THE EXCLUSIVE LOCK, and the bound
        family's main member is required to be the same inode this object
        bound at invocation. A swap that reaches the connection must therefore
        also survive that equality check, and the failure direction is always
        "refuse", never "write to the wrong file" or "claim a backup of one
        store while mutating another".
    """

    def __init__(self, path: Path) -> None:
        import os

        self.path = Path(path)
        flags = os.O_RDONLY
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_BINARY", 0)
        try:
            self.fd = os.open(self.path, flags)
        except FileNotFoundError as exc:
            raise TurnFenceRollbackRefused(
                f"{self.path} is not a file", reason="store-missing"
            ) from exc
        except OSError as exc:
            raise TurnFenceRollbackRefused(
                f"{self.path} could not be opened as a plain file — a symlink "
                f"or a special file cannot be a rollback target: {exc}",
                reason="store-unreadable",
            ) from exc
        info = os.fstat(self.fd)
        self.identity = (info.st_dev, info.st_ino)

    def verify(self, where: str) -> None:
        """Refuse unless *path* still names the inode this run bound."""
        import os

        try:
            now = os.stat(self.path)
        except OSError as exc:
            raise TurnFenceRollbackRefused(
                f"{self.path} no longer resolves to the file this rollback "
                f"bound at {where}: {exc}. Nothing was changed",
                reason="target-replaced",
            ) from exc
        if (now.st_dev, now.st_ino) != self.identity:
            raise TurnFenceRollbackRefused(
                f"{self.path} names a DIFFERENT file at {where} than the one "
                f"this rollback bound when it started (was "
                f"dev={self.identity[0]} ino={self.identity[1]}, now "
                f"dev={now.st_dev} ino={now.st_ino}). The store was replaced "
                "mid-operation; a backup of one file and drops against another "
                "is not a rollback. Nothing was changed",
                reason="target-replaced",
            )

    def bind_family(self):
        """Open EVERY member of the store's file family, atomically, by name once.

        Called under the exclusive lock, where no other process can write the
        store. What comes back reads exclusively through descriptors, so the
        pathname cannot be re-resolved underneath a copy — which is the ABA the
        main-file-only binding left open.

        The main member is required to be the inode bound at invocation; that
        is what ties the family to the target the operator named rather than to
        whatever the name means now.
        """
        import os

        members = {}
        try:
            for suffix in ("",) + _SIDECAR_SUFFIXES:
                candidate = self.path.with_name(self.path.name + suffix)
                flags = os.O_RDONLY
                flags |= getattr(os, "O_NOFOLLOW", 0)
                flags |= getattr(os, "O_BINARY", 0)
                try:
                    handle = os.open(candidate, flags)
                except FileNotFoundError:
                    if not suffix:
                        raise TurnFenceRollbackRefused(
                            f"{self.path} disappeared before its family could "
                            "be bound. Nothing was changed",
                            reason="target-replaced",
                        ) from None
                    continue
                except OSError as exc:
                    raise TurnFenceRollbackRefused(
                        f"{candidate} could not be bound as a plain file: "
                        f"{exc}. Nothing was changed",
                        reason="target-replaced",
                    ) from exc
                members[suffix] = handle
        except BaseException:
            for handle in members.values():
                try:
                    os.close(handle)
                except OSError:
                    pass
            raise

        main = os.fstat(members[""])
        if (main.st_dev, main.st_ino) != self.identity:
            for handle in members.values():
                try:
                    os.close(handle)
                except OSError:
                    pass
            raise TurnFenceRollbackRefused(
                f"{self.path} resolved to a different file when its family was "
                "bound than the one this rollback started on. The store was "
                "replaced mid-operation. Nothing was changed",
                reason="target-replaced",
            )
        return _BoundFamily(self.path, members)

    def open_for_reading(self):
        """A fresh read handle on the BOUND inode, never re-resolving the name.

        Rewound explicitly. ``os.dup`` shares the file OFFSET with the
        descriptor it copies, so a second reader would start wherever the first
        one stopped — at EOF — and hand back an empty file that looks like a
        successful copy. The byte-digest check caught exactly that; this is the
        cause it caught.
        """
        import os

        handle = os.fdopen(os.dup(self.fd), "rb", closefd=True)
        handle.seek(0)
        return handle

    def close(self) -> None:
        import os

        try:
            os.close(self.fd)
        except OSError:  # pragma: no cover - already closed
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


class _BoundFamily:
    """The store's whole file family, held open. Reads never touch a pathname."""

    def __init__(self, path: Path, members) -> None:
        self.path = Path(path)
        self.members = dict(members)

    def suffixes(self):
        return sorted(self.members, key=lambda suffix: (suffix != "", suffix))

    def open_member(self, suffix: str):
        import os

        handle = os.fdopen(os.dup(self.members[suffix]), "rb", closefd=True)
        handle.seek(0)
        return handle

    def digest(self, suffix: str) -> str:
        digest = hashlib.sha256()
        with self.open_member(suffix) as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    def close(self) -> None:
        import os

        for handle in self.members.values():
            try:
                os.close(handle)
            except OSError:  # pragma: no cover - already closed
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def _copy_family_onto_new_destinations(
    family: "_BoundFamily", destination: Path, *, what: str, collision_reason: str
) -> dict[str, Any]:
    """Copy every bound member to a destination this call creates exclusively.

    Sources are descriptors, destinations are ``O_CREAT | O_EXCL`` creations,
    and the verification compares the descriptor's bytes against the bytes that
    landed. No pathname is resolved on either side after the family was bound,
    so neither a swapped source nor a pre-existing destination can be involved
    without a refusal.
    """
    copied: list[str] = []
    try:
        for suffix in family.suffixes():
            target = destination.with_name(destination.name + suffix)
            _copy_onto_a_destination_we_created(
                None, target, source_handle=family.open_member(suffix)
            )
            copied.append(suffix or destination.name)
    except FileExistsError as exc:
        raise TurnFenceRollbackRefused(
            f"the {what} destination {exc.filename} already exists; refusing "
            "to overwrite it. The check is not the guarantee, the exclusive "
            "create is",
            reason=collision_reason,
        ) from exc
    except OSError as exc:
        raise TurnFenceRollbackRefused(
            f"the {what} of {family.path} could not be written to "
            f"{destination}: {exc}",
            reason="backup-unwritable",
        ) from exc

    drifted = [
        suffix or "(main)"
        for suffix in family.suffixes()
        if family.digest(suffix)
        != _sha256(destination.with_name(destination.name + suffix))
    ]
    if drifted:
        raise TurnFenceRollbackRefused(
            f"the {what} of {family.path} does not reproduce the bound store "
            f"({', '.join(drifted)} differ). Nothing was changed",
            reason="store-in-use",
        )
    return {"copy": str(destination), "files": copied}


def _copy_onto_a_destination_we_created(
    source: Path, destination: Path, *, source_handle=None
) -> None:
    """Copy *source* to a destination THIS call brought into existence.

    ``Path.exists()`` followed by ``shutil.copyfile`` cannot implement "must
    not already exist". They are two operations with a window between them, and
    ``copyfile`` truncates whatever it opens — so a destination created in that
    window is silently destroyed. Reproduced: a sentinel written between the
    check and the copy was gone, and the rollback reported success.

    ``O_CREAT | O_EXCL`` collapses the check and the create into one atomic
    operation the filesystem arbitrates: either this call made the file or it
    raises, and there is no third outcome to race. ``O_NOFOLLOW`` refuses a
    symlink sitting at the destination, which would otherwise redirect the
    write to a file the operator never named.

    The obligation covers EVERY destination — the main file and each sidecar —
    which is why this is the only way bytes are written here.
    """
    import os

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0)
    handle = os.open(destination, flags, 0o600)
    reader = source_handle if source_handle is not None else open(source, "rb")
    with reader, os.fdopen(handle, "wb") as writer:
        shutil.copyfileobj(reader, writer)


def _byte_copy_of_the_store(
    store_path: Path,
    destination: Path,
    *,
    what: str = "rehearsal",
    reason: str = "rehearsal-unwritable",
    collision_reason: str = "backup-exists",
    bound: "BoundTarget" = None,
) -> dict[str, Any]:
    """A copy of the store made with FILE I/O ONLY, verified byte-for-byte.

    WHY NOT :func:`_make_verified_backup`, WHICH ALREADY COPIES AND VERIFIES
        Because that one verifies by opening the SOURCE and comparing rows, and
        opening the source is the thing this exists to avoid. See
        :func:`_installed_fence_triggers`: ``mode=ro`` constrains the main file
        and not the sidecars, so any SQLite open of a WAL-mode store can add
        ``-wal`` and ``-shm`` to the operator's directory. A dry run that does
        that has changed the directory it promised to leave alone.

        So the verification here is on BYTES: every file is copied, then the
        source and the copy are both re-read and their digests compared. A
        byte-identical copy of the main file and every sidecar IS the store —
        including any ``-wal`` holding committed frames that have not been
        checkpointed, which is exactly what an ``immutable=1`` open would have
        silently skipped. Row-level verification still happens, one level down,
        when the rehearsal's own rollback writes its backup off this copy.

    A digest that moves between the two reads means something is writing to the
    store right now, which is not a store this operation may proceed on.
    """
    copied: list[str] = []
    names = [""] + [suffix for suffix in _SIDECAR_SUFFIXES]
    if bound is not None:
        bound.verify("the start of the copy")
    try:
        for suffix in names:
            source = store_path.with_name(store_path.name + suffix)
            if suffix and not source.is_file():
                continue
            _copy_onto_a_destination_we_created(
                source,
                destination.with_name(destination.name + suffix),
                # The MAIN file is read through the bound descriptor, so the
                # bytes in the copy come from the inode this run bound and not
                # from whatever the name resolves to now.
                source_handle=(
                    bound.open_for_reading() if bound is not None and not suffix
                    else None
                ),
            )
            copied.append(suffix or destination.name)
    except FileExistsError as exc:
        raise TurnFenceRollbackRefused(
            f"the {what} destination {exc.filename} already exists; refusing "
            "to overwrite it. A destination that appears between the check and "
            "the write is exactly what this refuses — the check is not the "
            "guarantee, the exclusive create is",
            reason=collision_reason,
        ) from exc
    except OSError as exc:
        raise TurnFenceRollbackRefused(
            f"the store at {store_path} could not be copied for the {what}: "
            f"{exc}",
            reason=reason,
        ) from exc

    drifted = []
    for suffix in names:
        source = store_path.with_name(store_path.name + suffix)
        mirror = destination.with_name(destination.name + suffix)
        if not source.is_file() and not mirror.is_file():
            continue
        if not source.is_file() or not mirror.is_file():
            drifted.append(suffix or "(main)")
            continue
        if _sha256(source) != _sha256(mirror):
            drifted.append(suffix or "(main)")
    if drifted:
        raise TurnFenceRollbackRefused(
            f"the {what} copy of {store_path} does not reproduce it "
            f"byte-for-byte ({', '.join(drifted)} differ), which means the "
            "store changed while it was being read. A copy of a moving store "
            "is a copy of nothing",
            reason="store-in-use",
        )
    return {"copy": str(destination), "files": copied}


def _refuse_if_this_process_owns_a_turn(
    store_path: Path, *, identity_path: Path = None
) -> None:
    """Close the ONE branch of the liveness predicate that is keyed by path.

    :meth:`SessionDB._turn_lease_row_is_free` frees a row whose ``owner_pid``
    is this process when this process holds no grant FOR THAT ``db_path`` — a
    turn that died without releasing. Every other branch reads the row.

    A rehearsal runs on a copy, and the copy has a different path. So a
    conversation THIS process is genuinely mid-turn on reads free on the copy
    and held on the original: the dry run would report "this would proceed"
    about a run that is going to refuse. Measured, not reasoned about — the
    first version of the rehearsal did exactly that.

    Asked here for the REAL path, against the conversations the store itself
    names, and only for the process-local registry. This is not a second
    opinion about liveness: the row-reading branches stay where they are, on
    :func:`_refuse_if_live`.

    The ROWS may be read from anywhere faithful — a rehearsal hands the copy —
    but *identity_path* is what the registry is keyed on, and defaults to the
    file the rows came from.
    """
    from hermes_state import live_turn_grant

    identity = Path(identity_path) if identity_path is not None else store_path

    conn = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT conversation_id FROM session_turn_leases "
            "WHERE holder IS NOT NULL AND holder != ''"
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise TurnFenceRollbackRefused(
            f"could not read the turn leases of {identity}: {exc}",
            reason="store-unreadable",
        ) from exc
    finally:
        conn.close()

    held = sorted(
        str(row[0])
        for row in rows
        if live_turn_grant(identity, str(row[0])) is not None
    )
    if held:
        raise TurnFenceRollbackRefused(
            f"refusing to roll back the turn fence on {identity}: this "
            f"process holds a live turn on {held}, so the real run would "
            "refuse. End those turns first",
            reason="live-turn",
        )


def _make_verified_backup(
    store_path: Path,
    backup_path: Path,
    *,
    bound: "BoundTarget" = None,
    family: "_BoundFamily" = None,
) -> dict[str, Any]:
    """Copy the store and READ THE COPY BACK before anything is changed.

    The verification is the point. ``shutil.copyfile`` succeeding says a file
    exists at the destination; it says nothing about whether that file opens,
    or whether the rows an operator would need to restore are in it. So the
    copy is made byte-for-byte, digest-checked in both directions, and then
    OPENED and read; a failure at any of those refuses the rollback rather than
    being logged.

    WHY THE SOURCE IS NEVER OPENED HERE, AND WHEN THIS RUNS
        An earlier version proved the copy by opening the SOURCE and comparing
        rows. That is a read-only connection on the operator's store, and
        ``mode=ro`` does not stop SQLite creating ``-wal``/``-shm`` beside a
        WAL-mode database — so the step that promised to change nothing was
        adding files. Byte equality across the main file and every sidecar is
        strictly stronger than the row comparison it replaces, and it needs no
        connection to the source at all.

        This runs INSIDE the caller's ``BEGIN EXCLUSIVE`` transaction, which is
        what makes the backup a copy of the state the rollback is about to act
        on rather than of an earlier, unprotected snapshot. No other writer can
        move the store between this copy and the ``DROP``s.
    """
    _refuse_unusable_backup_path(backup_path)

    if family is not None:
        # Sources are the descriptors bound under the exclusive lock, so the
        # backup is of the artifact the transaction is holding and not of
        # whatever the pathname family resolves to while the copy runs.
        snapshot = _copy_family_onto_new_destinations(
            family, backup_path, what="backup", collision_reason="backup-exists"
        )
    else:
        snapshot = _byte_copy_of_the_store(
            store_path,
            backup_path,
            what="backup",
            reason="backup-unwritable",
            collision_reason="backup-exists",
            bound=bound,
        )

    try:
        backup_conn = sqlite3.connect(f"file:{backup_path}?mode=ro", uri=True)
    except sqlite3.DatabaseError as exc:
        raise TurnFenceRollbackRefused(
            f"the backup at {backup_path} does not open: {exc}",
            reason="backup-unreadable",
        ) from exc
    try:
        found = _canonical_rows(backup_conn)
    except TurnFenceRollbackRefused as exc:
        raise TurnFenceRollbackRefused(
            f"the backup at {backup_path} is a byte-identical copy that does "
            f"not read back: {exc}. Refusing to roll back — a backup that "
            "cannot be read back is not a backup",
            reason="backup-unreadable",
        ) from exc
    finally:
        backup_conn.close()

    return {
        "path": str(backup_path),
        "files": snapshot["files"],
        "verified": True,
        "rows": {table: len(values) for table, values in found.items()},
    }




def preflight_turn_fence_rollback(
    store_path: Path,
    *,
    backup_path: Path,
    work_dir: Path,
    bound: "BoundTarget" = None,
) -> dict[str, Any]:
    """Everything that can refuse, decided WITHOUT touching the operator's store.

    ONE implementation, shared. :func:`rollback_turn_fence` calls this as its
    first act and the dry run calls it to answer "would this proceed", so a
    pre-flight that says yes and a run that then refuses cannot be two
    different opinions. A dry run that checks something other than what the
    real run checks is not a dry run.

    EVERY SQLITE OPEN HERE IS AGAINST A COPY
        The store is copied — main file and every sidecar, file I/O only,
        digest-checked both ways — into *work_dir*, and the surface probe, the
        lease read and the liveness check all run against that copy. This is
        not fastidiousness; it is the correction for a defect that appeared
        three times in this program under three disguises:

            C4b   reading the trigger surface THROUGH ``SessionDB`` healed the
                  answer, because schema init creates triggers IF NOT EXISTS —
                  verify-before-mutate was verifying its own repair
            C5    rehearsing on a copy still probed the original with
                  ``mode=ro``, which creates ``-wal``/``-shm`` beside a
                  WAL-mode store
            C5    the liveness question opened ``SessionDB`` on the store and
                  so INITIALISED the store it was asking about; a ``live-turn``
                  refusal moved ``state.db``'s digest on both SQLite builds

        Each time an inspection constructed the thing it was inspecting. The
        rule that falls out, and that this function exists to hold: a
        pre-flight must not build, migrate, heal or open-for-write the
        operator's artifact. Read a copy, decide, and only then construct
        anything.

    THIS IS NOT THE AUTHORITY ON LIVENESS
        It is the early, cheap, honest answer — it makes the common refusals
        (wrong target, half-installed surface, a turn already in flight, an
        occupied backup path) cost nothing and leave nothing. The decision the
        DDL acts on is taken again inside the exclusive transaction by
        :func:`_live_holders_inside_the_transaction`, because an answer from
        out here is a snapshot and a contender can acquire a lease after it.
    """
    progress = _blank_preflight()
    try:
        store_path = Path(store_path)
        backup_path = Path(backup_path)
        work_dir = Path(work_dir)
        if not store_path.is_file():
            raise TurnFenceRollbackRefused(
                f"{store_path} is not a file", reason="store-missing"
            )
        if bound is not None:
            bound.verify("the start of the pre-flight")
        progress["target_present"] = True

        try:
            work_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise TurnFenceRollbackRefused(
                f"the pre-flight directory {work_dir} could not be created, so "
                f"there is nowhere to inspect a copy of the store: {exc}",
                reason="rehearsal-unwritable",
            ) from exc

        copy = work_dir / "preflight.db"
        if bound is not None:
            # The family, not just the main file. This was the last place a
            # sidecar was still resolved by pathname after acquisition, which
            # is how an A-main/B-wal chimera could be assembled.
            with bound.bind_family() as family:
                _copy_family_onto_new_destinations(
                    family, copy, what="pre-flight",
                    collision_reason="rehearsal-unwritable",
                )
        else:
            _byte_copy_of_the_store(
                store_path, copy, what="pre-flight",
                collision_reason="rehearsal-unwritable",
            )

        expected = sorted(rollback_trigger_names())
        installed = _installed_fence_triggers(copy, report_as=store_path)
        _refuse_unexpected_surface(store_path, installed, expected)
        progress["surface_verified"] = True

        _refuse_if_this_process_owns_a_turn(copy, identity_path=store_path)
        _refuse_if_live(copy, report_as=store_path)
        progress["offline_verified"] = True

        _refuse_unusable_backup_path(backup_path)
        progress["backup_path_available"] = True
    except TurnFenceRollbackRefused as exc:
        exc.preflight = dict(progress)
        raise

    return {
        "store": str(store_path),
        "backup_path": str(backup_path),
        "copy": str(copy),
        "generation": hermes_state_common.TURN_FENCE_GENERATION,
        "installed_triggers": installed,
        "would_drop": expected,
        "preflight": dict(progress),
    }


def _commit_the_rollback(
    target_path: Path,
    backup_path: Path,
    expected,
    *,
    report_as: Path = None,
    bound: "BoundTarget" = None,
) -> dict[str, Any]:
    """The exclusion boundary: decide and act without letting go in between.

    ONE transaction covers the final liveness decision, the surface
    re-verification, the backup, and every ``DROP``. Nothing here trusts an
    answer computed outside it.

    EXCLUSIVE, not IMMEDIATE: a reader mid-query on a store whose triggers are
    about to vanish is a reader whose next statement prepares against a
    different schema. But taking the lock is NOT the liveness check — an idle
    connection does not prevent ``BEGIN EXCLUSIVE``, so the lock says nothing
    about whether a conversation is owned. It is the fence around the lease
    read, not a substitute for it.

    The backup is written INSIDE the boundary, after the liveness decision and
    before the first ``DROP``, so it is a copy of the state this transaction is
    about to act on rather than of an earlier snapshot that a contender may
    already have moved.
    """
    target_path = Path(target_path)
    reported = Path(report_as) if report_as is not None else target_path
    # IDENTITY, before the connection is opened by NAME. See `BoundTarget`: a
    # store renamed away and replaced by a different valid fenced store passes
    # every consistency check there is, because consistency is not identity.
    if bound is not None:
        bound.verify("the open of the final transaction")
    conn = sqlite3.connect(str(target_path), isolation_level=None, timeout=1.0)
    try:
        try:
            conn.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            raise TurnFenceRollbackRefused(
                f"exclusive ownership of {reported} could not be "
                f"established, so it is not offline: {exc}",
                reason="not-exclusive",
            ) from exc
        try:
            # 0. IDENTITY again, now that the lock is held: the connection
            #    above was opened by name, and the name could have moved
            #    between that call and this one.
            if bound is not None:
                bound.verify("the exclusive transaction")

            # 1. LIVENESS, here and not earlier. A lease acquired after the
            #    pre-flight and before this line is exactly the interleave the
            #    fence exists to stop, and the pre-flight cannot see it.
            owned = _live_holders_inside_the_transaction(conn, reported)
            if owned:
                raise TurnFenceRollbackRefused(
                    f"refusing to roll back the turn fence on {reported}: "
                    f"conversation(s) a live turn owns ({', '.join(owned)}) "
                    "were admitted between the pre-flight and this "
                    "transaction. Nothing was changed",
                    reason="live-turn",
                )

            # 2. The surface, again, for the same reason.
            installed = sorted(
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                    "AND name LIKE 'hermes_turn_fence_%'"
                ).fetchall()
            )
            _refuse_unexpected_surface(reported, installed, expected)

            # 3. The backup, of the state this transaction is holding. The
            #    WHOLE file family is bound here, under the lock, and every
            #    byte is read from those descriptors — binding only the main
            #    file while resolving -wal by name is what let an A-main/B-wal
            #    chimera through.
            if bound is not None:
                with bound.bind_family() as family:
                    backup = _make_verified_backup(
                        target_path, backup_path, family=family
                    )
            else:
                backup = _make_verified_backup(target_path, backup_path)

            # 4. All of them, or none.
            for name in expected:
                conn.execute(f"DROP TRIGGER {name}")
            # 5. IDENTITY one last time. If the name moved at any point in
            #    this block the drops are rolled back rather than committed.
            if bound is not None:
                bound.verify("the commit")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
    return backup


def rehearse_turn_fence_rollback(
    store_path: Path,
    *,
    backup_path: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Run the REAL rollback against a disposable copy. Only READ the store.

    The pre-flight already made a byte-faithful copy and decided every
    precondition on it; this drives :func:`_commit_the_rollback` — the same
    boundary the real run uses, in-transaction liveness check included —
    against that copy. What comes back is not a prediction; it is the
    operation, performed, somewhere else.

    The store itself is never opened by SQLite: it is read as bytes, once. See
    :func:`preflight_turn_fence_rollback` for why that is load-bearing rather
    than tidy.
    """
    store_path = Path(store_path)
    backup_path = Path(backup_path)
    work_dir = Path(work_dir)

    with BoundTarget(store_path) as bound:
        plan = preflight_turn_fence_rollback(
            store_path, backup_path=backup_path, work_dir=work_dir, bound=bound
        )
    copy = Path(plan["copy"])
    try:
        # The rehearsal's target is the copy this run just created in its own
        # private directory, so its identity is bound by construction — nothing
        # else knows the name.
        _commit_the_rollback(
            copy,
            work_dir / "rehearsal-backup.db",
            plan["would_drop"],
            report_as=store_path,
        )
    except TurnFenceRollbackRefused as exc:
        refused = TurnFenceRollbackRefused(
            f"the rehearsal on a copy of {store_path} was refused, so the "
            f"real run would be too: {exc}",
            reason=exc.reason,
        )
        refused.preflight = dict(plan["preflight"])
        raise refused from exc

    remaining = _installed_fence_triggers(copy, report_as=store_path)
    if remaining:
        raise TurnFenceRollbackRefused(
            f"the rehearsal ran on a copy of {store_path} and left "
            f"{len(remaining)} trigger(s) behind, so the real run would not be "
            "all-or-nothing either",
            reason="rehearsal-incomplete",
        )

    return {
        "store": str(store_path),
        "backup_path": str(backup_path),
        "generation": hermes_state_common.TURN_FENCE_GENERATION,
        "installed_triggers": plan["installed_triggers"],
        "would_drop": plan["would_drop"],
        "rehearsal": {"copy": str(copy), "work_dir": str(work_dir)},
        "preflight": dict(plan["preflight"]),
    }


def rollback_turn_fence(
    store_path: Path,
    *,
    backup_path: Path,
    work_dir: Path = None,
) -> dict[str, Any]:
    """Remove the turn-fence triggers from an IDLE store, offline, atomically.

    Two phases, and which one is authoritative matters:

    PRE-FLIGHT — :func:`preflight_turn_fence_rollback`, on a byte-faithful COPY
        The surface is the declared one; no conversation is owned; the backup
        destination is free. Every one of these can refuse, and a refusal here
        has not opened the operator's store with SQLite at all, so the file and
        the directory are exactly as they were found. Shared verbatim with the
        dry run.

    THE BOUNDARY — :func:`_commit_the_rollback`, on the store
        ``BEGIN EXCLUSIVE``, then liveness AGAIN, then the surface AGAIN, then
        the verified backup, then every ``DROP``, then ``COMMIT``. The
        pre-flight's answers are snapshots; these are decisions, and they are
        bound to the DML by one transaction because that is the standard this
        program holds every other writer to. The rollback removes that fence,
        so it does not get a weaker one.

    Returns a report. Raises :class:`TurnFenceRollbackRefused` at every other
    outcome, leaving rows and triggers as they were found; a refusal decided in
    the pre-flight leaves the file itself byte-identical too.

    *work_dir* holds the pre-flight copy. When it is ``None`` a private
    temporary directory is used and removed; ``residue`` in the report names
    what could not be removed, which the caller must surface rather than drop —
    the copy is an unfenced duplicate of every conversation in the store.
    """
    store_path = Path(store_path)
    backup_path = Path(backup_path)

    owned_work_dir = work_dir is None
    if owned_work_dir:
        import tempfile

        work_dir = Path(tempfile.mkdtemp(prefix="hermes-fence-preflight-"))
    else:
        work_dir = Path(work_dir)

    residue = None
    try:
        # ONE binding for the whole operation. Opened before the first check
        # and held past the last write, so "the file we inspected" and "the
        # file we mutated" are the same inode by construction rather than by
        # both happening to resolve the same name.
        with BoundTarget(store_path) as bound:
            plan = preflight_turn_fence_rollback(
                store_path, backup_path=backup_path, work_dir=work_dir,
                bound=bound,
            )
            expected = plan["would_drop"]
            backup = _commit_the_rollback(
                store_path, backup_path, expected, bound=bound
            )
    finally:
        if owned_work_dir:
            residue = sweep_work_dir(work_dir)

    report = {
        "store": str(store_path),
        "backup": backup,
        "generation": hermes_state_common.TURN_FENCE_GENERATION,
        "installed_triggers": plan["installed_triggers"],
        "preflight": plan["preflight"],
        "dropped_triggers": expected,
    }
    if residue is not None:
        report["residue"] = residue
    return report


def sweep_work_dir(work_dir: Path):
    """Remove a working directory and PROVE it is gone. ``None`` when it is.

    ``shutil.rmtree(..., ignore_errors=True)`` cannot distinguish "removed"
    from "failed, swallowed", and what is in here is a ROLLED-BACK — that is,
    UNFENCED — duplicate of every conversation in the store. Leaving one on
    disk while reporting success is a data-residue failure and an output-truth
    failure at once, so the removal is verified by looking, and what comes back
    is the caller's problem to surface.

    Reports a COUNT and the path. Never file names and never content: an
    operator needs to know where to go and that it is not empty.
    """
    work_dir = Path(work_dir)
    error = ""
    try:
        shutil.rmtree(work_dir)
    except OSError as exc:
        error = f" ({type(exc).__name__}: {exc})"
    except Exception as exc:  # pragma: no cover - defensive
        error = f" ({type(exc).__name__}: {exc})"
    if not work_dir.exists():
        return None
    try:
        remaining = sum(1 for entry in work_dir.rglob("*") if entry.is_file())
    except OSError:
        remaining = -1
    return {"work_dir": str(work_dir), "files": remaining, "error": error}
