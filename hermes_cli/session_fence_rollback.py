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

    1. FAIL-CLOSED ON OFFLINE AUTHORITY. The success path is permitted only on
       a detached artifact whose offline authority was established OUTSIDE this
       module, and no capability in this build can establish it — so the
       in-place run refuses, with a reason, every time. What was tried instead
       and does not work: a lease sweep reports what the store was told, not
       who has it open; ``BEGIN EXCLUSIVE`` in WAL mode excludes writers only;
       and a descriptor with ``O_NOFOLLOW`` pins an inode, not a pathname, and
       says nothing about who else holds one. All three are still here as
       refusals. None of them is evidence of offline, and none is read as such.
    2. AN EXECUTABLE BACKUP. Not a sentence in a runbook, and not a copy of
       files. SQLite writes it from the source connection, so it carries that
       connection's committed view wherever it lives — a row committed into an
       uncheckpointed ``-wal`` is in the store and not in the main file — and
       it is read back and compared against the source's own rows before any
       DDL runs. A backup that cannot be restored is not data preservation.
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
import os
import shutil
import sqlite3
import tempfile
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


#: Handed to :class:`_PrivateCopy` by the one function allowed to make one.
#: A module-private object cannot be named from outside this file, so the
#: capability is not constructible by anything that did not go through the
#: preparer — which is the difference between a docstring saying it may not be
#: minted and the type saying it cannot be.
_ONLY_THE_PREPARER = object()


class _PrivateCopy:
    """A store copy this run made, in a directory this run made. SEALED.

    THE QUESTION IS NEVER "IS IT MARKED", IT IS WHO MAY MINT THE MARK
        An earlier shape of this passed an ``OfflineAuthority`` object into the
        boundary. The docstring said it could not be minted on request; the
        constructor was public and took a path, so anything holding a
        ``Path`` could produce one and walk past the offline contract. The
        prose and the type said opposite things, and the type is what runs.

        So the capability is not a label attached to a path. It is only ever
        returned by :func:`prepare_the_private_copy`, in the SAME call that
        creates the directory, creates the file, and records what it created —
        and it carries the identity of both so the boundary can re-establish
        them rather than believe the object.

    WHY THIS IS AUTHORITY AND A LEASE SWEEP IS NOT
        Not because the file looks idle. Because of how it came to exist: this
        process made the directory with ``mkdtemp`` (0700, a name nothing else
        has) and made the file inside it with ``O_CREAT | O_EXCL``. "No other
        writer is attached" is the construction, not a deduction from the
        contents. That is the one thing in this build that can be established
        rather than inferred, and it is why the rehearsal may perform the
        operation while the operator's artifact may not be touched.
    """

    def __init__(self, sentinel, path: Path, work_dir: Path, identity) -> None:
        if sentinel is not _ONLY_THE_PREPARER:
            raise TypeError(
                "a private copy is not constructible: it is returned by "
                "prepare_the_private_copy, which is the call that creates the "
                "directory and the file it describes. An object that can be "
                "built from a pathname is a label, and a label is what the "
                "offline contract exists to refuse"
            )
        self.path = Path(path)
        self.work_dir = Path(work_dir)
        self.identity = identity

    def verify(self) -> None:
        """Re-establish, at the point of use, what the preparer recorded.

        Trusting the object would make the seal a formality: an object handed
        across a call is a claim about the past, and what the boundary needs is
        the present. Cheap enough to do again, and the failure direction is a
        refusal.
        """
        import stat as _stat

        if self.path.parent != self.work_dir:
            raise TurnFenceRollbackRefused(
                f"{self.path} is no longer inside the private directory it was "
                f"prepared in ({self.work_dir})",
                reason="offline-authority-unknown",
            )
        try:
            here = os.lstat(self.path)
            holder = os.lstat(self.work_dir)
        except OSError as exc:
            raise TurnFenceRollbackRefused(
                f"the private copy at {self.path} could not be re-examined: "
                f"{exc}",
                reason="offline-authority-unknown",
            ) from exc
        if (here.st_dev, here.st_ino) != self.identity:
            raise TurnFenceRollbackRefused(
                f"{self.path} is not the file this run prepared; the private "
                "copy was replaced",
                reason="target-replaced",
            )
        if not _stat.S_ISREG(here.st_mode) or here.st_nlink != 1:
            raise TurnFenceRollbackRefused(
                f"{self.path} has acquired a second name or is no longer a "
                f"plain file (links={here.st_nlink})",
                reason="target-untrusted-namespace",
            )
        if not _stat.S_ISDIR(holder.st_mode) or _stat.S_IMODE(holder.st_mode) & 0o077:
            raise TurnFenceRollbackRefused(
                f"the private directory {self.work_dir} is no longer owner-only "
                f"(mode {_stat.S_IMODE(holder.st_mode):o}), so the copy inside "
                "it is reachable by something this run did not account for",
                reason="offline-authority-unknown",
            )
        if holder.st_uid != os.getuid():
            raise TurnFenceRollbackRefused(
                f"the private directory {self.work_dir} is not owned by this "
                "process",
                reason="offline-authority-unknown",
            )


def prepare_the_private_copy(
    store_path: Path, *, work_dir: Path, bound: "BoundTarget" = None
) -> "_PrivateCopy":
    """Make the copy and return the capability over it, in ONE call.

    Split into "make a copy" and "declare it private" and the second half
    becomes a claim anyone can make about any file. Kept together, the thing
    returned is the thing created: the directory is this process's ``mkdtemp``
    (0700), the file inside it is an ``O_CREAT | O_EXCL`` create, and the
    identity recorded is the one that came back from the filesystem.

    This is also the pre-flight's copy. There is not a private one for the
    rehearsal and another for inspection — one artifact, prepared once, so the
    surface that was verified is the surface that is acted on.
    """
    work_dir = Path(work_dir)
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        private = Path(tempfile.mkdtemp(prefix="private-", dir=str(work_dir)))
    except OSError as exc:
        raise TurnFenceRollbackRefused(
            f"there is nowhere under {work_dir} to prepare a private copy of "
            f"{store_path}: {exc}",
            reason="rehearsal-unwritable",
        ) from exc
    copy = private / "preflight.db"
    _byte_copy_of_the_store(
        store_path, copy, what="pre-flight",
        collision_reason="rehearsal-unwritable",
        bound=bound,
    )
    info = os.lstat(copy)
    return _PrivateCopy(
        _ONLY_THE_PREPARER, copy, private, (info.st_dev, info.st_ino)
    )


def _canonical_store_paths() -> list:
    """EVERY path this build might mean by "the store", not the first one.

    Canonicalness is a SET, and reducing it to one candidate is how a real
    ``state.db`` gets classified as something nobody recognises. There are at
    least two independent answers and they disagree by design:

    * ``hermes_state._default_db_path()`` — which honours a re-pointed
      ``DEFAULT_DB_PATH`` above everything else;
    * ``get_hermes_home() / "state.db"`` — the profile's home as it stands right
      now.

    A profile override, a re-pointed default, or a test fixture makes those two
    different paths, and the store the operator is actually running on is
    whichever the process would open. If the artifact is ANY of them it is the
    live one.

    Both are asked of the modules that own them rather than reconstructed here,
    for the same reason the trigger names are generated: a second copy of
    somebody else's rule is right until they change it.

    The failure direction matters. Missing a candidate makes this say "I do not
    know what this is" about the operator's real store — which is safe only for
    as long as every path refuses anyway, and is precisely the judgement a
    future authority-bearing caller would trust to tell a canonical store from
    a detached artifact.
    """
    candidates = []
    try:
        from hermes_state import _default_db_path

        candidates.append(Path(_default_db_path()))
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        from hermes_constants import get_hermes_home

        candidates.append(Path(get_hermes_home()) / "state.db")
    except Exception:  # pragma: no cover - defensive
        pass
    seen = []
    for candidate in candidates:
        if candidate not in seen:
            seen.append(candidate)
    return seen


def _same_file(left: Path, right: Path) -> bool:
    try:
        a, b = os.stat(left), os.stat(right)
    except OSError:
        return False
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def disqualify_the_target(artifact: Path) -> None:
    """Refuse the targets that are provably wrong, before anything is opened.

    Three facts about the PATHNAME and the INODE, none of which needs the store
    to be read, and each of which is a different next move for the operator. It
    runs first precisely because it must not depend on the store being
    openable: an artifact with a hot journal beside it cannot be probed
    read-only without SQLite trying to roll it back, so a check that waited
    until after the pre-flight would never be reached on the case it is for.

    It only ever ADDS refusals. Nothing here can authorise anything, which is
    why using the canonical location is safe in this direction and would not be
    in the other — "outside HERMES_HOME therefore offline" is an inference, and
    "this IS the store the binary opens by default" is an identity.
    """
    artifact = Path(artifact)
    try:
        info = os.lstat(artifact)
    except OSError:
        # Missing, unreadable, a directory: named by the checks that follow,
        # which say more about it than this one could.
        return
    import stat as _stat

    if not _stat.S_ISREG(info.st_mode):
        return
    if info.st_nlink != 1:
        raise TurnFenceRollbackRefused(
            f"{artifact} has {info.st_nlink} hard links, so another name "
            "reaches the same file and nothing this run can see governs it. A "
            "rollback is not permitted on an artifact whose namespace is not "
            "bounded. Nothing was changed",
            reason="target-untrusted-namespace",
        )
    # BY IDENTITY, against every candidate. A string comparison is what a
    # symlink or an override defeats, which is the same lesson the A/B store
    # swap taught one layer down.
    for canonical in _canonical_store_paths():
        if _same_file(artifact, canonical):
            raise TurnFenceRollbackRefused(
                f"{artifact} IS a canonical store this binary opens on its own "
                f"({canonical}). The canonical store is the live one whatever "
                "it looks like at this instant, and a detached artifact is "
                "what this verb acts on. Nothing was changed",
                reason="canonical-store-target",
            )
    present = [
        str(artifact.with_name(artifact.name + suffix))
        for suffix in _SIDECAR_SUFFIXES
        if artifact.with_name(artifact.name + suffix).exists()
    ]
    if present:
        raise TurnFenceRollbackRefused(
            f"{artifact} has SQLite sidecars beside it ({', '.join(present)}), "
            "so it is attached to a connection or was interrupted while it "
            "was. A detached artifact has none of these. Nothing was changed",
            reason="target-not-quiesced",
        )


def establish_offline_authority(artifact: Path) -> OfflineAuthority:
    """The external route, and in this build it always refuses. Deliberately.

    WHAT WOULD HAVE TO BE TRUE
        The success path is permitted on a detached artifact whose offline
        authority was established elsewhere. The only producer of a detached
        ``state.db`` in this tree is
        :func:`hermes_cli.backup.create_quick_snapshot`, which copies through
        the SQLite backup API into a staging directory and renames it into
        place with a ``manifest.json`` beside it. That is a real detachment;
        the manifest is not a binding.

    WHY IT CANNOT BE READ AS ONE, MEASURED
        ``manifest.json`` records ``files[rel] = SIZE`` and nothing else — see
        ``hermes_cli/backup.py``, where ``manifest`` is a ``Dict[str, int]``.
        Replacing a snapshot's ``state.db`` with a DIFFERENT database of the
        same size leaves the id, the path and the size entry all agreeing while
        the contents are another store. That was run against this exact tree,
        not argued from the source. So a manifest entry cannot say which
        database the artifact is, and provenance built on it would be a
        capability in name only.

    WHAT HAPPENS INSTEAD
        This refuses, with its own reason, and the refusal is the deliverable.
        Binding contents needs the PRODUCER to record something that identifies
        them — a separate slice with its own evidence, not something to infer
        here. Until then a verb that refuses what it cannot prove is worth more
        than one whose success rests on a size field, and none of the shapes
        that would manufacture a success are permitted to stand in for the
        proof: not a flag, not a manifest path the caller chooses, not a lease
        or process scan, and not "the path is outside HERMES_HOME". Every one
        of those is an inference, and inference is the thing being withdrawn.
    """
    artifact = Path(artifact)
    disqualify_the_target(artifact)
    raise TurnFenceRollbackRefused(
        f"no capability in this build can establish that {artifact} is offline, "
        "so the rollback is refused. The only producer of a detached state.db "
        "here is the quick snapshot, whose manifest records file SIZE only — a "
        "same-size replacement satisfies it while the contents are a different "
        "database — so nothing available proves which store an artifact is or "
        "that it is detached. Nothing was changed",
        reason="offline-authority-unknown",
    )


#: The outcome states, in the only order they may be reached. The snapshot
#: states are INTERNAL: they describe a file in a private staging directory
#: that the operator will never see and that is removed either way. Only the
#: three ``backup-*`` states are claims about the artifact at the path the
#: operator named. ``committed`` and ``commit-unknown`` are alternatives at the
#: same depth: both terminal, and neither may follow the other.
_OUTCOME_RANK = {
    "not-started": 0,
    "preflight-passed": 1,
    "snapshot-created": 2,
    "snapshot-verified": 3,
    "backup-created": 4,
    "backup-verified": 5,
    "backup-durable": 6,
    "committing": 7,
    "committed": 8,
    "commit-unknown": 8,
}


class RollbackOutcome:
    """What the run ESTABLISHED, kept where a failure cannot take it back.

    A report returned at the end cannot describe a run that did not reach the
    end, and the shape this replaces inferred exactly that: no report meant
    nothing happened. It does not. ``COMMIT`` runs before the connection is
    closed and before anything is assembled, and an operator told "nothing was
    changed" after the fence came off stops looking for the store they now have
    to restore.

    So the caller owns this object and reads it on every path, including the
    ones where the call raised. Two rules make that safe:

    MONOTONIC. :meth:`advance` moves forward or refuses. A late failure sets the
    exit status and the primary reason; it may not rewrite a state an earlier
    step established.

    SEPARATE FACTS. ``changed``, ``backup_created``, ``backup_verified``,
    ``backup_durable`` and ``residue_present`` are independent observations, not
    renderings of one another. A run whose backup landed and whose commit is
    unknown has to be able to say both.

    ``changed`` IS THREE-STATE, and that is the point. ``False`` before the
    commit is attempted, ``True`` once it returned, and ``None`` while it is
    genuinely unknown — a ``COMMIT`` that raised may have landed. Rendering
    that as a boolean always loses the same state, and it is the state an
    operator most needs to see.
    """

    def __init__(self) -> None:
        self.outcome = "not-started"
        self.changed = False
        self.backup_created = False
        self.backup_verified = False
        self.backup_durable = False
        self.residue_present = False
        self.residue = None
        self.backup = None

    def advance(self, state: str) -> None:
        if state not in _OUTCOME_RANK:
            raise ValueError(f"unknown rollback outcome {state!r}")
        current = _OUTCOME_RANK[self.outcome]
        if _OUTCOME_RANK[state] < current:
            raise ValueError(
                f"refusing to move the outcome back from {self.outcome!r} to "
                f"{state!r}: a later step does not get to retract what an "
                "earlier one established"
            )
        if _OUTCOME_RANK[state] == current and state != self.outcome:
            raise ValueError(
                f"{self.outcome!r} and {state!r} are alternatives; neither may "
                "follow the other"
            )
        self.outcome = state
        if state == "backup-created":
            self.backup_created = True
        elif state == "backup-verified":
            self.backup_verified = True
        elif state == "backup-durable":
            self.backup_durable = True
        elif state in ("committing", "commit-unknown"):
            self.changed = None
        elif state == "committed":
            self.changed = True

    def facts(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "changed": self.changed,
            "backup_created": self.backup_created,
            "backup_verified": self.backup_verified,
            "backup_durable": self.backup_durable,
            "residue_present": self.residue_present,
            "residue": self.residue,
            "backup": self.backup,
        }


class BoundTarget:
    """The store this run acts on, bound by INODE rather than by name.

    A PATH IS A NAME, NOT AN IDENTITY
        Every check in this module used to re-resolve ``store_path``, so the
        file inspected and the file mutated were only assumed to be the same
        one. They need not be: rename the store away between the pre-flight and
        the final open, drop a DIFFERENT valid fenced store at the same path,
        and every consistency check passes — the surface matches, the store is
        idle, the generation is right — while the backup describes the store
        that left and the drops land on the store that arrived. Reproduced
        exactly that way.

        Re-checking the surface and the liveness inside the transaction does
        not help. Those protect CONSISTENCY, and the substitute is perfectly
        consistent. What was missing is IDENTITY.

        So the target is opened once, with ``O_NOFOLLOW`` so a symlink cannot
        stand in for it, and the descriptor is held for the whole operation.
        Holding it pins the inode: it cannot be recycled underneath us even if
        the name is unlinked. ``(st_dev, st_ino)`` from that descriptor is the
        identity every later step is measured against.

    WHAT IT IS NOT, STATED SO NOTHING BUILDS ON IT
        It is a defence against substitution of the named file, and that is the
        whole of it. It does not establish that the store is offline and cannot
        be extended to: a descriptor says nothing about who else has one, and
        ``sqlite3`` opens by pathname, so nothing here can prove the CONNECTION
        is even on the bound inode. What "offline" requires is
        :func:`establish_offline_authority`, which is a different question
        answered somewhere else.
    """

    def __init__(self, path: Path) -> None:
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

    def open_for_reading(self):
        """A fresh read handle on the BOUND inode, never re-resolving the name.

        Rewound explicitly. ``os.dup`` shares the file OFFSET with the
        descriptor it copies, so a second reader would start wherever the first
        one stopped — at EOF — and hand back an empty file that looks like a
        successful copy. The byte-digest check caught exactly that; this is the
        cause it caught.
        """
        handle = os.fdopen(os.dup(self.fd), "rb", closefd=True)
        handle.seek(0)
        return handle

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:  # pragma: no cover - already closed
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


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
    """
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

    This is how the PRE-FLIGHT gets something it can open without opening the
    operator's artifact. It is not how the backup is made — see
    :func:`_make_verified_backup`, which has a connection and uses it.

    ``mode=ro`` constrains the main file and not the sidecars, so any SQLite
    open of a WAL-mode store can add ``-wal`` and ``-shm`` to the operator's
    directory. A pre-flight that does that has changed the directory it
    promised to leave alone, so it reads bytes and inspects the result.

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


class _AcquiredDestinations:
    """Every name the backup could occupy, taken atomically, owned explicitly.

    WHY AN ACQUISITION AND NOT A CHECK
        ``VACUUM INTO`` and the online backup API both take a FILENAME. Neither
        can be handed a descriptor, so the call that writes the destination
        cannot also be the call that creates it, and "must not already exist"
        has to be re-established at the one place it can be: an
        ``O_CREAT | O_EXCL`` create of the final path, with the bytes then
        streamed through THAT descriptor. Anything else — check then write,
        reserve then let SQLite reopen the path — reopens the window where a
        file that appeared in between is destroyed.

    WHY THE WHOLE FAMILY
        A backup is a file and its ``-wal``/``-shm``/``-journal`` siblings. An
        operator whose previous attempt died leaves one of those behind with no
        main file, and "the path you named does not exist" is true of the name
        they typed and useless — the orphan is what the next reader picks up.

    WHAT CLEANUP MAY TOUCH
        Only what this object created, identified by the ``(st_dev, st_ino)``
        recorded at acquisition rather than by the name. The file that was
        already there is not this run's to remove, and a name that has since
        become a different file is not either.
    """

    def __init__(self, backup_path: Path) -> None:
        self.backup_path = Path(backup_path)
        self.handles: dict[str, int] = {}
        self.identities: dict[str, tuple] = {}
        self.identities_at_acquisition: dict[str, tuple] = {}

    def _member(self, suffix: str) -> Path:
        return self.backup_path.with_name(self.backup_path.name + suffix)

    def acquire(self) -> None:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_BINARY", 0)
        for suffix in ("",) + _SIDECAR_SUFFIXES:
            member = self._member(suffix)
            try:
                handle = os.open(member, flags, 0o600)
            except FileExistsError as exc:
                self.remove_only_what_we_created()
                raise TurnFenceRollbackRefused(
                    f"the backup destination {member} already exists; refusing "
                    "to overwrite it. Name a path whose whole family — the "
                    "file and its -wal/-shm/-journal siblings — is free. The "
                    "check is not the guarantee, the exclusive create is",
                    reason="backup-exists",
                ) from exc
            except OSError as exc:
                self.remove_only_what_we_created()
                raise TurnFenceRollbackRefused(
                    f"the backup destination {member} could not be created: "
                    f"{exc}",
                    reason="backup-unwritable",
                ) from exc
            self.handles[suffix] = handle
            info = os.fstat(handle)
            self.identities[suffix] = (info.st_dev, info.st_ino)
            # Kept past release, because the identity of the file this run
            # created is what any later cleanup has to match against.
            self.identities_at_acquisition[suffix] = (info.st_dev, info.st_ino)

    def write_the_main_member_from(self, snapshot: Path) -> None:
        """Stream *snapshot* through the descriptor this object created.

        The bytes are quiescent by now — the engine finished writing them into
        a directory nothing else can name — which is why a plain copy is sound
        here and was not sound as a way of making the snapshot itself.
        """
        handle = self.handles[""]
        with open(snapshot, "rb") as reader, os.fdopen(
            os.dup(handle), "wb", closefd=True
        ) as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
        os.fsync(handle)

    def release_the_sidecars(self) -> None:
        """Give back the sidecar names, so the backup is one file.

        They were taken to prove nothing was already there. Leaving them would
        put an empty ``-shm`` next to a database that has no connection.
        """
        for suffix in list(self.handles):
            if not suffix:
                continue
            self._release(suffix)

    def confirm_the_final_member(self, snapshot: Path) -> None:
        now = os.stat(self.backup_path)
        if (now.st_dev, now.st_ino) != self.identities[""]:
            raise TurnFenceRollbackRefused(
                f"{self.backup_path} is not the file this run created when it "
                "acquired the destination, so the backup that was written is "
                "not the backup at that name",
                reason="backup-unwritable",
            )
        if _sha256(self.backup_path) != _sha256(snapshot):
            raise TurnFenceRollbackRefused(
                f"{self.backup_path} does not reproduce the snapshot the "
                "engine wrote",
                reason="backup-unreadable",
            )

    def _release(self, suffix: str) -> None:
        handle = self.handles.pop(suffix, None)
        if handle is not None:
            try:
                os.close(handle)
            except OSError:  # pragma: no cover - already closed
                pass
        identity = self.identities.pop(suffix, None)
        member = self._member(suffix)
        try:
            info = os.lstat(member)
        except OSError:
            return
        if identity is not None and (info.st_dev, info.st_ino) != identity:
            return
        try:
            os.unlink(member)
        except OSError:  # pragma: no cover - defensive
            pass

    def close(self) -> None:
        for suffix in list(self.handles):
            handle = self.handles.pop(suffix, None)
            if handle is not None:
                try:
                    os.close(handle)
                except OSError:  # pragma: no cover - already closed
                    pass

    def remove_only_what_we_created(self) -> None:
        for suffix in list(self.handles):
            self._release(suffix)


def _fsync_the_directory(directory: Path) -> None:
    """Flush the DIRECTORY ENTRY, not only the bytes.

    A file whose contents are on the platter and whose name is not is a backup
    nothing can find after the crash this backup exists for.
    """
    handle = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _verify_the_snapshot_restores(
    snapshot: Path, reported: Path, expected_rows, expected_triggers
) -> dict[str, int]:
    """Read the snapshot back and require it to BE the pre-rollback state.

    "It opens" is not the claim. An earlier version proved a backup by opening
    it, and a file that opens can be missing every row an operator would need
    to restore. So the rows of every fenced table are compared against the ones
    the source connection held, and the fence surface has to still be on it —
    a backup taken after the drops restores a store with no fence.
    """
    try:
        conn = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
    except sqlite3.DatabaseError as exc:
        raise TurnFenceRollbackRefused(
            f"the backup for {reported} does not open: {exc}",
            reason="backup-unreadable",
        ) from exc
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        if not integrity or str(integrity[0]) != "ok":
            raise TurnFenceRollbackRefused(
                f"the backup for {reported} does not pass integrity_check: "
                f"{integrity!r}",
                reason="backup-unreadable",
            )
        found = _canonical_rows(conn)
        surface = sorted(
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                "AND name LIKE 'hermes_turn_fence_%'"
            ).fetchall()
        )
    except sqlite3.DatabaseError as exc:
        raise TurnFenceRollbackRefused(
            f"the backup for {reported} does not read back: {exc}",
            reason="backup-unreadable",
        ) from exc
    finally:
        conn.close()

    if found != expected_rows:
        differing = sorted(
            table
            for table in set(found) | set(expected_rows)
            if found.get(table) != expected_rows.get(table)
        )
        raise TurnFenceRollbackRefused(
            f"the backup for {reported} does not restore the state the source "
            f"connection holds ({', '.join(differing)} differ). A backup that "
            "cannot be restored is not a backup",
            reason="backup-not-restorable",
        )
    if surface != list(expected_triggers):
        raise TurnFenceRollbackRefused(
            f"the backup for {reported} does not carry the fence surface the "
            "store had before the rollback, so restoring it would put back an "
            "unfenced store",
            reason="backup-not-restorable",
        )
    return {table: len(values) for table, values in found.items()}


def _make_verified_backup(
    conn: sqlite3.Connection,
    store_path: Path,
    backup_path: Path,
    *,
    expected_rows,
    expected_triggers,
    outcome: "RollbackOutcome" = None,
) -> dict[str, Any]:
    """A LOGICAL backup, written by the engine on the source connection.

    WHY NOT A FILE COPY
        Because a file copy reproduces the bytes that happen to be on disk, and
        committed state does not have to be there — a row committed into an
        uncheckpointed ``-wal`` is in the store and not in the main file. The
        engine copies the CONNECTION's committed view, so it captures that
        wherever it lives, and it does it without this module having to reason
        about which files the store currently spans.

    THE SEQUENCE, AND WHY EACH STEP IS WHERE IT IS
        1. a private staging directory inside the destination's own parent, so
           the snapshot is built where no other name resolves to it and the
           copy at the end stays on one filesystem;
        2. the snapshot, on *conn*, outside any transaction — neither
           ``VACUUM INTO`` nor the online backup API may run inside one, which
           is why the decisions this backup rests on are taken in a transaction
           that has already closed and taken again in the one that acts;
        3. verification against the source's own rows BEFORE the destination is
           touched, so a backup that would not restore never becomes a file the
           operator can find;
        4. the destination family, acquired atomically;
        5. the bytes, through the descriptor this run created — the snapshot is
           quiescent by now, which is what makes a plain copy sound here;
        6. ``fsync`` of the file and then of its parent directory. Both, or the
           backup is durable and its NAME is not.

    WHAT MAY BE CLAIMED, AND WHEN
        The snapshot is a private intermediate. It is created and verified in a
        directory nothing else can name and it is removed either way, so
        advancing ``backup_created`` when IT appears publishes a fact about a
        file the operator has never been able to reach — and a collision at the
        final destination then leaves ``backup_created=true`` beside no backup
        at all. So the snapshot has its own internal states and the three
        ``backup-*`` facts are set together, after the final inode has been
        exclusively acquired, written, flushed and proved byte-equal to the
        snapshot that was verified. Before that moment there is no backup to
        claim.

    THE STAGING DIRECTORY IS SWEPT, NOT ASSUMED
        ``rmtree(..., ignore_errors=True)`` cannot tell "removed" from "failed,
        swallowed", and what is in there is an unfenced duplicate of every
        conversation in the store. This slice has already paid for that once,
        in the pre-flight directory; the staging lives somewhere else, which is
        exactly how the second instance stayed invisible. So it is swept and
        the removal is proved, and a failure to remove it refuses the run
        BEFORE the drops — while keeping a backup that is already durable,
        because both facts are true and the operator needs both.
    """
    _refuse_unusable_backup_path(backup_path)

    staging = None
    reservation = None
    report = None
    try:
        try:
            staging = Path(
                tempfile.mkdtemp(
                    prefix=".fence-rollback-backup-", dir=str(backup_path.parent)
                )
            )
        except OSError as exc:
            raise TurnFenceRollbackRefused(
                f"the backup of {store_path} has nowhere to be staged next to "
                f"{backup_path}: {exc}",
                reason="backup-unwritable",
            ) from exc

        snapshot = staging / "snapshot.db"
        try:
            conn.execute("VACUUM INTO ?", (str(snapshot),))
        except sqlite3.DatabaseError as exc:
            raise TurnFenceRollbackRefused(
                f"the engine could not write a backup of {store_path}: {exc}",
                reason="backup-unwritable",
            ) from exc
        if outcome is not None:
            outcome.advance("snapshot-created")

        rows = _verify_the_snapshot_restores(
            snapshot, store_path, expected_rows, expected_triggers
        )
        if outcome is not None:
            outcome.advance("snapshot-verified")

        reservation = _AcquiredDestinations(backup_path)
        reservation.acquire()
        reservation.write_the_main_member_from(snapshot)
        reservation.release_the_sidecars()
        _fsync_the_directory(backup_path.parent)
        reservation.confirm_the_final_member(snapshot)
        reservation.close()

        # NOW there is a backup: at the name the operator gave, byte-equal to a
        # snapshot that was read back and restored, and flushed with its
        # directory entry. All three facts became true together, so they are
        # set together.
        report = {
            "path": str(backup_path),
            "files": [backup_path.name],
            "verified": True,
            "durable": True,
            "rows": rows,
            # Carried so a later withdrawal can check that the file it is
            # about to unlink is still the one this run created.
            "identity": list(reservation.identities_at_acquisition[""]),
        }
        if outcome is not None:
            outcome.advance("backup-created")
            outcome.advance("backup-verified")
            outcome.advance("backup-durable")
            outcome.backup = report
    except BaseException:
        if reservation is not None:
            reservation.remove_only_what_we_created()
        raise
    finally:
        if staging is not None:
            residue = sweep_work_dir(staging)
            if residue is not None:
                if outcome is not None:
                    outcome.residue_present = True
                    outcome.residue = residue
                # The durable backup is NOT removed. It exists, it restores,
                # and telling the operator otherwise to keep the failure tidy
                # is the same lie as reporting success over residue.
                refused = TurnFenceRollbackRefused(
                    f"the backup of {store_path} landed, and its staging copy "
                    f"could not be removed: {residue['files']} file(s) remain "
                    f"under {residue['work_dir']}{residue['error']}. That copy "
                    "is a duplicate of every conversation in the store — "
                    "remove that directory. The rollback did not proceed",
                    reason="backup-staging-residue",
                )
                refused.residue = residue
                raise refused

    return report


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
        digest-checked both ways — into a private directory under *work_dir*,
        and the surface probe, the lease read and the liveness check all run
        against that copy. This is not fastidiousness; it is the correction for
        a defect that appeared three times in this program under three
        disguises:

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

    NONE OF THIS ESTABLISHES THAT THE STORE IS OFFLINE
        The lease read is an early, cheap refusal for the case where the store
        itself records a live turn, and it is worth having for that. It is not
        evidence of the absence of writers — a lease table says what the store
        was told, not who has it open — and it is not what permits the
        rollback. What permits it is which target reached the mutating
        boundary at all, and only a copy this run prepared ever does.
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

        prepared = prepare_the_private_copy(
            store_path, work_dir=work_dir, bound=bound
        )
        copy = prepared.path

        expected = sorted(rollback_trigger_names())
        installed = _installed_fence_triggers(copy, report_as=store_path)
        _refuse_unexpected_surface(store_path, installed, expected)
        progress["surface_verified"] = True

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
        "private_copy": prepared,
        "generation": hermes_state_common.TURN_FENCE_GENERATION,
        "installed_triggers": installed,
        "would_drop": expected,
        "preflight": dict(progress),
    }


def _decide_under_the_lock(conn, reported: Path, expected, *, bound) -> None:
    """Liveness, surface and identity — all three, or nothing runs.

    Held together by one ``BEGIN EXCLUSIVE`` so a contender cannot acquire a
    lease between the read and the verdict: to acquire it must write
    ``session_turn_leases``, and it cannot while this transaction is open.

    EXCLUSIVE is not a liveness proof and is not used as one. An idle
    connection does not prevent it, and in WAL mode it excludes writers only.
    It is the fence around the lease read, not a substitute for it. Neither is
    evidence that the artifact is detached, which is why nothing here decides
    that question: it is settled by WHICH TARGET reached this boundary at all.
    """
    if bound is not None:
        bound.verify("the exclusive transaction")

    owned = _live_holders_inside_the_transaction(conn, reported)
    if owned:
        raise TurnFenceRollbackRefused(
            f"refusing to roll back the turn fence on {reported}: "
            f"conversation(s) a live turn owns ({', '.join(owned)}) "
            "were admitted between the pre-flight and this "
            "transaction. Nothing was changed",
            reason="live-turn",
        )

    installed = sorted(
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND name LIKE 'hermes_turn_fence_%'"
        ).fetchall()
    )
    _refuse_unexpected_surface(reported, installed, expected)


def _commit_the_rollback(
    private_copy: "_PrivateCopy",
    backup_path: Path,
    expected,
    *,
    report_as: Path = None,
    outcome: "RollbackOutcome" = None,
) -> dict[str, Any]:
    """Decide, back up, then decide again and act. On a PRIVATE COPY, only.

    THE TARGET IS THE AUTHORITY, AND IT IS NOT A PARAMETER ANYONE CAN SUPPLY
        This takes a :class:`_PrivateCopy`, which only
        :func:`prepare_the_private_copy` can produce and which records the
        directory and the inode it created. A capability that can be passed can
        be forged; the previous shape took an ``OfflineAuthority`` built from a
        pathname, and a caller holding any path could walk past the offline
        contract with it. There is now nothing to forge — a pathname is not a
        private copy, and this refuses one.

    WHY NOT ONE TRANSACTION AROUND ALL OF IT
        Because the backup cannot be inside one. ``VACUUM INTO`` refuses to run
        in a transaction and the online backup API blocks indefinitely against
        a source holding ``BEGIN EXCLUSIVE`` — both measured. The earlier shape
        put a byte copy in there instead, which is what made the backup a copy
        of files rather than of the store.

        So the decisions are taken under the lock, the lock is released for the
        backup, and every decision is taken AGAIN under the lock that performs
        the drops. Nothing the DDL rests on was decided outside the transaction
        that runs it — the standard every other writer in this program is held
        to — and the window the backup sits in is one where the target is a
        file nothing else can name.

    A REFUSAL AFTER THE BACKUP TAKES THE BACKUP WITH IT, with one exception:
    staging residue, where the backup landed and something else went wrong.
    Deleting a good backup to keep a failure tidy is not cleanup.
    """
    if not isinstance(private_copy, _PrivateCopy):
        raise TurnFenceRollbackRefused(
            f"the rollback boundary was entered with {type(private_copy).__name__} "
            "rather than a private copy this run prepared. A pathname is a name, "
            "not a capability, and no target that was merely named has offline "
            "authority. Nothing was changed",
            reason="offline-authority-unknown",
        )
    private_copy.verify()
    target_path = private_copy.path
    backup_path = Path(backup_path)
    reported = Path(report_as) if report_as is not None else target_path
    conn = sqlite3.connect(str(target_path), isolation_level=None, timeout=1.0)
    try:
        # 1. DECIDE.
        try:
            conn.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            raise TurnFenceRollbackRefused(
                f"exclusive ownership of {reported} could not be "
                f"established, so it is not offline: {exc}",
                reason="not-exclusive",
            ) from exc
        try:
            _decide_under_the_lock(conn, reported, expected, bound=None)
            expected_rows = _canonical_rows(conn)
        finally:
            # ROLLBACK, not COMMIT, and not only for tidiness. The decision
            # phase writes nothing, so rolling back is what actually happened;
            # and it leaves exactly one COMMIT in this module, which is the one
            # whose outcome an operator has to be told about.
            conn.execute("ROLLBACK")
        if outcome is not None:
            outcome.advance("preflight-passed")

        # 2. BACK UP, on this connection, with no transaction open.
        backup = _make_verified_backup(
            conn, reported, backup_path,
            expected_rows=expected_rows,
            expected_triggers=list(expected),
            outcome=outcome,
        )

        # 3. DECIDE AGAIN, AND ACT — one transaction over both.
        try:
            conn.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            _remove_a_backup_this_run_wrote(
                backup_path, backup["identity"], outcome
            )
            raise TurnFenceRollbackRefused(
                f"exclusive ownership of {reported} could not be "
                f"re-established after the backup: {exc}",
                reason="not-exclusive",
            ) from exc
        try:
            private_copy.verify()
            _decide_under_the_lock(conn, reported, expected, bound=None)
            for name in expected:
                conn.execute(f"DROP TRIGGER {name}")
        except BaseException:
            conn.execute("ROLLBACK")
            _remove_a_backup_this_run_wrote(
                backup_path, backup["identity"], outcome
            )
            raise

        if outcome is not None:
            outcome.advance("committing")
        try:
            conn.execute("COMMIT")
        except BaseException as exc:
            # NOT rolled back, and not reported as unchanged. SQLite may have
            # committed and failed afterwards; "it raised" does not decide
            # which, and neither may this.
            if outcome is not None:
                outcome.advance("commit-unknown")
            raise TurnFenceRollbackRefused(
                f"the COMMIT of the rollback on {reported} raised ({exc}). "
                "Whether the transaction landed is UNKNOWN — do not assume the "
                f"fence is still installed. The verified backup is at "
                f"{backup_path}",
                reason="commit-unknown",
            ) from exc
        if outcome is not None:
            outcome.advance("committed")
    finally:
        conn.close()
    return backup


def _remove_a_backup_this_run_wrote(backup_path: Path, identity, outcome=None):
    """Unlink the backup ONLY while it is still the file this run created.

    A path is a name, not an identity — the same rule the A/B store swap
    established, and it binds harder here because deletion is irreversible. The
    backup has been published: something can replace it between the moment it
    landed and the moment a later failure decides to withdraw it, and unlinking
    by name then destroys a file that is not this run's.

    Nor may a failure to remove it be swallowed. An undeletable backup is
    residue like any other, and residue that is not reported is residue that is
    not found.
    """
    try:
        here = os.lstat(backup_path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        left = {"path": str(backup_path), "error": f"{type(exc).__name__}: {exc}"}
        if outcome is not None:
            outcome.residue_present = True
            outcome.residue = left
        return left
    if (here.st_dev, here.st_ino) != tuple(identity):
        left = {"path": str(backup_path), "error": "ownership-lost"}
        if outcome is not None:
            outcome.residue_present = True
            outcome.residue = left
        return left
    try:
        os.unlink(backup_path)
    except OSError as exc:
        left = {"path": str(backup_path), "error": f"{type(exc).__name__}: {exc}"}
        if outcome is not None:
            outcome.residue_present = True
            outcome.residue = left
        return left
    return None


def rehearse_turn_fence_rollback(
    store_path: Path,
    *,
    backup_path: Path,
    work_dir: Path,
    outcome: "RollbackOutcome" = None,
) -> dict[str, Any]:
    """Perform the REAL rollback on a disposable copy, then report the verdict.

    Two halves, and the order is the point.

    THE OPERATION HAPPENS. The pre-flight already prepared a byte-faithful copy
    in a private directory; this drives :func:`_commit_the_rollback` — the same
    boundary, the same backup, the same in-transaction decisions — against that
    copy. What comes back is not a prediction. The copy is the only artifact in
    this build with offline authority, and it has it because of how it was
    made, not because of anything read out of it.

    THEN THE VERDICT THE REAL RUN WILL GIVE. A dry run that reports "this would
    proceed" about an operation that then refuses is worse than having no dry
    run, because the operator types the real command on its say-so. So the
    authority over the OPERATOR's artifact is established last, and the
    refusal it raises carries what the rehearsal established — the operator
    learns what the rollback would do AND why it will not be permitted, from
    one invocation.

    The store itself is never opened by SQLite: it is read as bytes, once.
    """
    store_path = Path(store_path)
    backup_path = Path(backup_path)
    work_dir = Path(work_dir)
    outcome = outcome if outcome is not None else RollbackOutcome()

    with BoundTarget(store_path) as bound:
        plan = preflight_turn_fence_rollback(
            store_path, backup_path=backup_path, work_dir=work_dir, bound=bound
        )
    copy = Path(plan["copy"])
    try:
        _commit_the_rollback(
            plan["private_copy"],
            work_dir / "rehearsal-backup.db",
            plan["would_drop"],
            report_as=store_path,
            outcome=outcome,
        )
    except TurnFenceRollbackRefused as exc:
        refused = TurnFenceRollbackRefused(
            f"the rehearsal on a copy of {store_path} was refused, so the "
            f"real run would be too: {exc}",
            reason=exc.reason,
        )
        refused.preflight = dict(plan["preflight"])
        refused.established = {
            "installed_triggers": plan["installed_triggers"],
            "would_drop": plan["would_drop"],
            "rehearsal": dict(outcome.facts(), copy=str(copy)),
        }
        raise refused from exc

    remaining = _installed_fence_triggers(copy, report_as=store_path)
    if remaining:
        raise TurnFenceRollbackRefused(
            f"the rehearsal ran on a copy of {store_path} and left "
            f"{len(remaining)} trigger(s) behind, so the real run would not be "
            "all-or-nothing either",
            reason="rehearsal-incomplete",
        )

    established = {
        "store": str(store_path),
        "backup_path": str(backup_path),
        "generation": hermes_state_common.TURN_FENCE_GENERATION,
        "installed_triggers": plan["installed_triggers"],
        "would_drop": plan["would_drop"],
        "rehearsal": dict(outcome.facts(), copy=str(copy)),
        "preflight": dict(plan["preflight"]),
    }
    try:
        establish_offline_authority(store_path)
    except TurnFenceRollbackRefused as exc:
        exc.preflight = dict(plan["preflight"])
        exc.established = established
        raise
    return established  # pragma: no cover - establish_offline_authority always raises


def rollback_turn_fence(
    store_path: Path,
    *,
    backup_path: Path,
    work_dir: Path = None,
    outcome: "RollbackOutcome" = None,
) -> dict[str, Any]:
    """Remove the turn-fence triggers from a DETACHED artifact, offline.

    OFFLINE IS PROVEN, NOT INFERRED, AND IN THIS BUILD IT CANNOT BE PROVEN
        The pre-flight refuses a store that records a live turn and the
        decision phase refuses one it cannot lock, and neither of those is
        evidence that nothing is attached. What permits the operation is
        :func:`establish_offline_authority`, and it refuses every artifact —
        see its docstring for the measurement. So this function has no
        reachable success path today, and the mutating boundary is not even
        offered the operator's store: it takes a private copy this run made,
        which is a thing no caller can manufacture.

        That is the deliverable, not a gap in it. The alternative on offer was
        a success gated on an inference, and every inference available here has
        a counterexample.

    WHAT STILL RUNS, AND WHY IT IS NOT DEAD CODE
        Everything up to the authority: the pre-flight on a copy, the identity
        binding, the in-transaction liveness and surface decisions on the
        operator's own store. And the operation itself — backup, drops, commit
        — is driven on every invocation by
        :func:`rehearse_turn_fence_rollback`, against the private copy. A
        boundary that is only ever green because nothing reaches it is not
        verified, so nothing here is left to that.

    Raises :class:`TurnFenceRollbackRefused` on every outcome, leaving rows and
    triggers as they were found; a refusal decided in the pre-flight leaves the
    file itself byte-identical too. *outcome* is the caller's, so what a run
    established is readable even on the paths where this raises.
    """
    store_path = Path(store_path)
    backup_path = Path(backup_path)
    outcome = outcome if outcome is not None else RollbackOutcome()

    # BEFORE ANYTHING IS OPENED, COPIED OR CREATED. Under this contract the
    # answer is always "no", so every byte written, every SQLite handle taken
    # and every duplicate of the store made on the way to it is work done for a
    # run that cannot proceed — and opening the source is not free: it can put
    # -wal/-shm beside the artifact, which is a refusal that changed the
    # directory it is about to say it left alone. The inspection must not
    # construct what it inspects, and a decision placed after the side effect it
    # governs has not been made in time.
    disqualify_the_target(store_path)
    establish_offline_authority(store_path)

    owned_work_dir = work_dir is None
    if owned_work_dir:
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
                plan["private_copy"], backup_path, expected, outcome=outcome
            )
    finally:
        if owned_work_dir:
            residue = sweep_work_dir(work_dir)
            if residue is not None:
                outcome.residue_present = True
                outcome.residue = residue

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
