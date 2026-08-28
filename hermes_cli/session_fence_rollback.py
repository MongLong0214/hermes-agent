"""One bounded, offline way back off the turn-fence generation barrier.

WHAT THIS IS NOT
    Not a compatibility fallback. Not an automatic trigger drop on open. Not a
    silent admission of an old writer. A binary from before the barrier being
    unable to write a fenced store is INTENTIONAL and required, and nothing in
    this module runs unless a person asks for it by name, on a store nobody is
    using.

WHO CALLS IT, AND WHAT IS CURRENTLY REACHABLE
    ``hermes sessions fence-rollback`` — :mod:`hermes_cli.session_fence_rollback_cmd`
    — and nothing else in the tree. Both entry points below —
    :func:`rollback_turn_fence` and :func:`rehearse_turn_fence_rollback` —
    refuse before they observe the store: :func:`disqualify_the_target` and
    :func:`establish_offline_authority` run first, and the second always
    raises. The store is never opened, copied or written to.

THIS IS THE REACHABLE CONTRACT ONLY
    An earlier revision of this module also carried the backup/commit engine
    that a real rollback would need once an artifact's offline authority could
    actually be established: exclusive destination acquisition, a
    SQLite-verified backup, a pre-flight surface comparison, an in-transaction
    commit of the trigger drops, and a withdrawal ledger for cleanup. Every one
    of those ~2,600 lines sat behind :func:`establish_offline_authority`, which
    refuses unconditionally in this build — so none of it was reachable from
    the verb, and a test exercising it was maintenance evidence about code
    nothing could call, not a statement about behaviour. It has been removed
    from this reduction along with its dedicated pins; the refusal contract
    below is unchanged by the removal, because the engine never ran. The
    removed engine remains available in this repository's history (see the
    commit that introduced ``fix(c5): fence-rollback verb joins the
    composition with its Family A/B pins``) for whichever future slice
    actually establishes an artifact's offline authority.

WHY THIS EXISTS ANYWAY, EVEN AS A REFUSAL
    Because the alternative that was actually on offer — an operator typing
    ``DROP TRIGGER`` once per surface entry into whatever database they think
    is the right one — is not a rollback story: no backup, no check that the
    store is idle, no check that it is removing the surface it believes it is
    removing, and no atomicity. Naming the refusal, its reason, and the
    surface a real rollback would touch is worth more than either the manual
    route or a success gated on an inference that has a counterexample (see
    :func:`establish_offline_authority`).

THE RETURN LEG
    Rollback is not a one-way door. ``_init_schema`` creates the triggers with
    ``CREATE TRIGGER IF NOT EXISTS`` on every open, so a current binary
    reopening a rolled-back store puts the fence back and the old binary is
    refused again. That is a property of the store, not a favour this module
    does.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import hermes_state_common


class TurnFenceRollbackRefused(RuntimeError):
    """The rollback did not run, and the store was not touched.

    Every raise site in this module happens BEFORE any DDL — so catching this
    means the store is exactly as it was found.

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
        # EMPTY, not all-false. A refusal that never reached the pre-flight has
        # no pre-flight result, and a block of false checks would claim each of
        # them was performed and failed. Nothing in this build ever reaches a
        # pre-flight, so this stays empty on every path.
        self.preflight: dict[str, bool] = {}


def rollback_trigger_names() -> tuple[str, ...]:
    """The triggers a rollback would remove, GENERATED from the declaration.

    Read through the module rather than bound at import so that moving
    ``TURN_FENCE_SURFACE`` moves this with it. Unreachable on the only path
    this build has today (every invocation refuses first), kept because it is
    the generated fact a future success report would need and there is no
    hand-copied list here to go stale.
    """
    declared = getattr(hermes_state_common, "TURN_FENCE_TRIGGERS", None)
    if declared:
        return tuple(declared)
    return tuple(
        hermes_state_common.turn_fence_trigger_name(table, operation)
        for table, operation in hermes_state_common.TURN_FENCE_SURFACE
    )


#: The sidecars SQLite may have left beside the main file. Checked for, never
#: created, by :func:`disqualify_the_target`.
_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")

DISQUALIFICATION_REASONS = {
    "namespace": "target-untrusted-namespace",
    "canonical": "canonical-store-target",
    "not-quiesced": "target-not-quiesced",
    "unknown": "offline-authority-unknown",
}


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
    until after a pre-flight would never be reached on the case it is for.

    It only ever ADDS refusals. Nothing here can authorise anything, which is
    why using the canonical location is safe in this direction and would not be
    in the other — "outside HERMES_HOME therefore offline" is an inference, and
    "this IS the store the binary opens by default" is an identity.
    """
    artifact = Path(artifact)
    try:
        info = os.lstat(artifact)
    except FileNotFoundError:
        raise TurnFenceRollbackRefused(
            f"{artifact} is not a file", reason="store-missing"
        ) from None
    except OSError as exc:
        raise TurnFenceRollbackRefused(
            f"{artifact} could not be examined: {exc}", reason="store-unreadable"
        ) from exc
    import stat as _stat

    if not _stat.S_ISREG(info.st_mode):
        # A directory, a symlink, a device. Named here rather than left to a
        # later check, because there is no later check any more: nothing opens
        # the target, so a fact about the pathname is the only fact available.
        raise TurnFenceRollbackRefused(
            f"{artifact} is not a plain file, so it is not a rollback target",
            reason="store-unreadable",
        )
    if info.st_nlink != 1:
        raise TurnFenceRollbackRefused(
            f"{artifact} has {info.st_nlink} hard links, so another name "
            "reaches the same file and nothing this run can see governs it. A "
            "rollback is not permitted on an artifact whose namespace is not "
            "bounded. Nothing was changed",
            reason=DISQUALIFICATION_REASONS["namespace"],
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
                reason=DISQUALIFICATION_REASONS["canonical"],
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
            reason=DISQUALIFICATION_REASONS["not-quiesced"],
        )


def establish_offline_authority(artifact: Path) -> None:
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
        reason=DISQUALIFICATION_REASONS["unknown"],
    )


class RollbackOutcome:
    """What the run ESTABLISHED, kept where a failure cannot take it back.

    A report returned at the end cannot describe a run that did not reach the
    end. So the caller owns this object and reads it on every path, including
    the ones where the call raised.

    ``changed`` IS THREE-STATE, and that is the point. ``False`` before a
    commit is attempted, ``True`` once it returned, and ``None`` while it is
    genuinely unknown. In this build every path leaves it ``False``, because
    every path refuses before a commit is attempted.
    """

    def __init__(self) -> None:
        self.outcome = "not-started"
        self.changed = False
        self.backup_created = False
        self.backup_verified = False
        self.backup_durable = False
        # HISTORY above, PRESENCE here, and they are not the same question.
        self.backup_present = False
        self.backup_unlinked_by_this_run = False
        self.backup_absence_durable = False
        self.backup_withdrawn = False
        # A LIST, and never assigned over.
        self.residue = []
        self.backup = None

    @property
    def residue_present(self) -> bool:
        """Derived, so it cannot disagree with what was actually recorded."""
        return bool(self.residue)

    def note_residue(self, record: dict, *, incident: str) -> None:
        """Record ONE unresolved incident. Idempotent per incident.

        Unreachable in this build — nothing here creates anything that could
        leave residue — kept because :class:`RollbackOutcome` is the shared
        bookkeeping shape a future authority-bearing run would still use.
        """
        if record is None:
            return
        entry = dict(record, incident=incident)
        for position, existing in enumerate(self.residue):
            if existing.get("incident") == incident:
                self.residue[position] = entry
                return
        self.residue.append(entry)

    def facts(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "changed": self.changed,
            "backup_created": self.backup_created,
            "backup_verified": self.backup_verified,
            "backup_durable": self.backup_durable,
            "backup_present": self.backup_present,
            "backup_unlinked_by_this_run": self.backup_unlinked_by_this_run,
            "backup_absence_durable": self.backup_absence_durable,
            "backup_withdrawn": self.backup_withdrawn,
            "residue_present": bool(self.residue),
            "residue": list(self.residue),
            "backup": self.backup,
        }


def rehearse_turn_fence_rollback(
    store_path: Path,
    *,
    backup_path: Path,
    work_dir: Path = None,
    outcome: "RollbackOutcome" = None,
) -> dict[str, Any]:
    """Refuses before it derives anything. There is no rehearsal to report.

    Refuses, before a working directory is created, before the source is
    opened, before any image is taken. What comes back is a refusal and the
    reason for it, and the source is byte-for-byte and file-for-file as it was
    found. *work_dir* is accepted and unused: nothing here creates one.
    """
    store_path = Path(store_path)
    backup_path = Path(backup_path)
    outcome = outcome if outcome is not None else RollbackOutcome()

    # BEFORE ANYTHING IS CREATED, OPENED OR READ.
    disqualify_the_target(store_path)
    establish_offline_authority(store_path)
    raise AssertionError(  # pragma: no cover - establish_offline_authority raises
        "establish_offline_authority returned; it has no success path"
    )


def rollback_turn_fence(
    store_path: Path,
    *,
    backup_path: Path,
    work_dir: Path = None,
    outcome: "RollbackOutcome" = None,
) -> dict[str, Any]:
    """Remove the turn-fence triggers from a DETACHED artifact, offline.

    OFFLINE IS PROVEN, NOT INFERRED, AND IN THIS BUILD IT CANNOT BE PROVEN
        The pre-flight a real rollback would need refuses a store that records
        a live turn and refuses one it cannot lock, and neither of those is
        evidence that nothing is attached. What permits the operation is
        :func:`establish_offline_authority`, and it refuses every artifact —
        see its docstring for the measurement. So this function has no
        reachable success path today: it refuses BEFORE IT OBSERVES THE SOURCE
        AT ALL. ``disqualify_the_target`` and ``establish_offline_authority``
        are the only two statements that run; the second always raises.
        Nothing else runs: no pre-flight, no copy, no SQLite handle on the
        store, no working directory. That is deliberate and it is the
        acceptance property — opening the source is not free, because it can
        leave ``-wal``/``-shm`` beside the artifact, and a refusal that changed
        the directory it is about to say it left alone is not a refusal.

    Raises :class:`TurnFenceRollbackRefused` on every outcome, leaving the
    store byte-identical — including its directory listing, because nothing
    here opens it. *outcome* is the caller's, so what a run established is
    readable even on the paths where this raises. *backup_path* and
    *work_dir* are accepted for signature compatibility with the future
    authority-bearing implementation; neither is examined or used while this
    build fails closed.
    """
    store_path = Path(store_path)
    backup_path = Path(backup_path)
    outcome = outcome if outcome is not None else RollbackOutcome()

    # BEFORE ANYTHING IS OPENED, COPIED OR CREATED. Under this contract the
    # answer is always "no", so every byte written, every SQLite handle taken
    # and every duplicate of the store made on the way to it is work done for a
    # run that cannot proceed.
    disqualify_the_target(store_path)
    establish_offline_authority(store_path)
    raise AssertionError(  # pragma: no cover - establish_offline_authority raises
        "establish_offline_authority returned; it has no success path"
    )
