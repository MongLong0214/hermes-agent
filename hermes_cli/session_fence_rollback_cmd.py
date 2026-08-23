"""``hermes sessions fence-rollback`` — the operator surface over the rollback.

WHAT WAS MISSING, GIVEN THAT THE OPERATION ALREADY EXISTED
    :mod:`hermes_cli.session_fence_rollback` is bounded, offline, fail-closed
    and all-or-nothing, and none of that reaches a person at 3am. What reached
    them was a Python one-liner pasted into a runbook — or the thing the
    one-liner replaced, twenty-four hand-typed ``DROP TRIGGER`` statements
    against whatever database they believed was the right one.

    A one-liner is not a production rollback because it has:

    * no exit code a script or a runbook step can branch on;
    * no way to say WHICH precondition stopped it. "It failed" is the answer
      that sends somebody editing the store by hand;
    * no rehearsal. The first time the operator learns what it would do is
      after it has done it.

    So this module is four things and nothing else: a named target, a
    pre-flight whose result is reported, a rehearsal that provably changes
    nothing, and a machine-readable report on BOTH outcomes.

WHAT IT DELIBERATELY IS NOT
    Not a compatibility fallback, not an automatic trigger drop, not a
    discovery pass over the machine's stores. The old binary being unable to
    write a fenced store is intentional and required, and this verb does not
    soften it — it only gives the person who has decided to step back off the
    barrier a way to do it that can be audited afterwards.

    The target is named, never defaulted. A rollback that runs against "the
    store we would have opened anyway" is one wrong host away from destroying
    the fence on a store nobody meant to touch, and the invocation that does it
    looks identical in the shell history to the one they meant.

    The backup path is named too, for the same reason: the operator decides
    where a copy of their conversations lands.

OUTPUT CONTRACT
    stdout carries exactly one JSON document, on success and on refusal alike.
    Human-readable lines go to stderr. That split is what lets a runbook step
    do ``hermes sessions fence-rollback … | jq -r .refused.reason`` while a
    person watching the terminal still reads a sentence.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

_VERB = "sessions fence-rollback"


def add_fence_rollback_parser(sessions_subparsers):
    """Register the verb on the ``sessions`` subparsers. Called from ``main``.

    Lives here rather than inline in ``main()`` so the registration is a thing
    a test can build a parser around: "``--store`` is required" is a claim
    about this function, and checking it by reading ``main.py`` as text is the
    check that passes while the flag means something else.
    """
    parser = sessions_subparsers.add_parser(
        "fence-rollback",
        help=(
            "Rehearse the turn-fence rollback; a real run is refused until an "
            "offline artifact can be proven"
        ),
        description=(
            "Step a NAMED session store back off the turn-fence generation "
            "barrier, offline and all-or-nothing. THIS BUILD REFUSES EVERY "
            "REAL RUN: the rollback is permitted only on a detached artifact "
            "whose offline authority was established elsewhere, and nothing "
            "here can establish it — the only producer of a detached state.db "
            "records file size only, which a same-size replacement satisfies. "
            "So a real invocation reports which kind of target you named and "
            "changes nothing. --dry-run performs the whole operation on a "
            "private copy this command creates, which is the one artifact it "
            "can prove is offline, and reports what it did along with the "
            "refusal the real run gives. The store and the backup path are "
            "both explicit; this command never discovers a target. Prints one "
            "JSON report on stdout; human-readable lines go to stderr."
        ),
    )
    parser.add_argument(
        "--store",
        type=Path,
        required=True,
        help=(
            "The session database to EVALUATE and rehearse against. Required "
            "and never defaulted — name the file you mean. This build does not "
            "roll back any store you can name: the run is refused and the file "
            "is read as bytes, never opened as a database"
        ),
    )
    parser.add_argument(
        "--backup",
        type=Path,
        required=True,
        help=(
            "The destination RESERVED for the verified backup an "
            "authority-bearing run would write before dropping any trigger. "
            "NOT created by this build — nothing is written here. It is still "
            "required and still validated, because an occupied destination is "
            "a refusal you need to find out about now rather than at the "
            "moment a real run would be about to overwrite something. Must not "
            "already exist, including its -wal/-shm/-journal siblings"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Perform the rollback on a disposable copy of the store and report "
            "what it removed, alongside the refusal the real run gives. The "
            "copy and its backup live in a private directory and are removed "
            "before this exits; the store itself is read as bytes and never "
            "modified. This is the only form of the command that does anything"
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Existing directory to hold --dry-run's disposable copy of the "
            "store and its transient backup (defaults to the system temporary "
            "directory). Both are removed before this exits. Point it at a "
            "volume with room for a second copy of a large store"
        ),
    )
    return parser


def run_fence_rollback(args) -> int:
    """Dispatch entry point. Returns the process exit code."""
    store = Path(getattr(args, "store")).expanduser()
    backup = Path(getattr(args, "backup")).expanduser()
    dry_run = bool(getattr(args, "dry_run", False))
    work_parent = getattr(args, "work_dir", None)
    work_parent = Path(work_parent).expanduser() if work_parent else None

    if dry_run:
        return _report_rehearsal(store, backup, work_parent)
    return _report_rollback(store, backup)


def _report_rehearsal(store: Path, backup: Path, work_parent: Path = None) -> int:
    """``--dry-run``: rehearse on a copy, report the plan AND the verdict.

    THE COPY DOES NOT GO IN THE OPERATOR'S DIRECTORY. An earlier version put it
    beside the store, reasoning that a hidden subdirectory removed in a
    ``finally`` leaves the listing exactly as it found it. It does — until the
    process dies hard, and then a full UNFENCED duplicate of every conversation
    is sitting next to the original with nothing to draw attention to it.
    ``mkdtemp`` is 0700 and lives where the platform already expects disposable
    copies; ``--work-dir`` moves it for a store too large for that volume.

    THE REHEARSAL COMPLETING IS NOT THE REAL RUN PROCEEDING. The rollback is
    permitted only on an artifact whose offline authority was established
    elsewhere, and the copy this rehearsal makes is the only artifact that
    qualifies. So a dry run reports both: what the operation did on the copy,
    and the refusal the operator will get if they type the real command.
    """
    import tempfile

    from hermes_cli import session_fence_rollback as rollback

    if work_parent is not None and not work_parent.is_dir():
        return _emit_refusal(
            rollback.TurnFenceRollbackRefused(
                f"--work-dir {work_parent} is not an existing directory, so "
                "the dry run has nowhere to put its copy",
                reason="rehearsal-unwritable",
            ),
            store=store, backup=backup, dry_run=True,
        )
    try:
        work_dir = Path(tempfile.mkdtemp(
            prefix="hermes-fence-rehearsal-",
            dir=str(work_parent) if work_parent is not None else None,
        ))
    except OSError as exc:
        return _emit_refusal(
            _unexpected(exc), store=store, backup=backup, dry_run=True
        )
    outcome = rollback.RollbackOutcome()
    plan = None
    refusal = None
    try:
        try:
            plan = rollback.rehearse_turn_fence_rollback(
                store, backup_path=backup, work_dir=work_dir, outcome=outcome
            )
        except rollback.TurnFenceRollbackRefused as exc:
            refusal = exc
        except (sqlite3.DatabaseError, OSError) as exc:
            refusal = _unexpected(exc)
    finally:
        # ONE ENTRY POINT. The record goes into the outcome and is read back
        # from there; nothing downstream appends. Two writers observing one
        # sweep is what turned a single stuck directory into two identical
        # records, and a second entry point is how that returns.
        residue = rollback.sweep_work_dir(work_dir)
        if residue is not None:
            copies = sorted(work_dir.glob("private-*/preflight.db"))
            outcome.note_residue(
                rollback.describe_residue(
                    residue,
                    obligation="the rehearsal working directory",
                    outcome=outcome,
                    copy_path=str(copies[0]) if copies else None,
                ),
                incident=f"work-dir:{work_dir}",
            )

    # WHAT THE REHEARSAL ESTABLISHED, decided by the steps that established it.
    # Read from the outcome the run wrote into rather than from a report it may
    # never have returned: a fault after the rehearsal's COMMIT raises, and a
    # caller that reads only the return value concludes nothing happened.
    established = getattr(refusal, "established", None)
    rehearsal_facts = dict(outcome.facts())
    if isinstance(established, dict) and isinstance(
        established.get("rehearsal"), dict
    ):
        rehearsal_facts = dict(established["rehearsal"])
    # The ledger is the single source for residue. The snapshot above was taken
    # when the operation ended and the sweep happens after it, so this is a
    # re-read, never an append.
    residue_records = list(outcome.facts()["residue"])
    rehearsal_facts["residue"] = residue_records
    rehearsal_facts["residue_present"] = bool(residue_records)
    _settle_the_transient_artifacts(rehearsal_facts, swept=residue is None)

    if refusal is None and residue_records:
        # Nothing else decided this run's fate, so residue names it. When
        # something did, the reason below stays ITS reason and the residue
        # rides alongside — a late failure sets the exit status, it does not
        # get to relabel what the run was refused for.
        refusal = rollback.TurnFenceRollbackRefused(
            f"the dry run left {len(residue_records)} thing(s) behind that it "
            "created and could not remove",
            reason="residue-not-removed",
        )
    if refusal is not None:
        return _emit_refusal(
            refusal, store=store, backup=backup, dry_run=True,
            rehearsal=rehearsal_facts,
            residue=residue_records,
            would_drop=(established or {}).get("would_drop"),
            installed_triggers=(established or {}).get("installed_triggers"),
        )

    # Unreachable while nothing can establish offline authority over the
    # operator's artifact; kept honest rather than asserted away.
    _emit(
        {
            "verb": _VERB,
            "ok": True,
            "dry_run": True,
            "changed": False,
            "store": str(store),
            "backup": str(backup),
            "generation": plan["generation"],
            "installed_triggers": plan["installed_triggers"],
            "would_drop": plan["would_drop"],
            "dropped_triggers": [],
            "preflight": plan["preflight"],
            "rehearsal": rehearsal_facts,
        }
    )
    return 0


def _settle_the_transient_artifacts(facts: dict, *, swept: bool) -> None:
    """Re-answer PRESENCE for everything that lived in the swept directory.

    The rehearsal's facts are snapshotted when the operation ends, and the
    sweep happens after. Every presence claim taken before an invalidating step
    is stale by the time it is printed — the backup said ``present: true`` about
    a file the sweep had already removed, and ``copy`` was a bare path string
    making no claim at all, which is the same error one field over. Fixing them
    where they are set does not help: they are set BEFORE the thing that
    invalidates them. It has to happen here, once, after the only step that can.

    History is untouched. ``backup_created``/``verified``/``durable`` stay true
    because they happened, and under a contract where no real run proceeds that
    record is the only evidence the operation works at all. What changes is the
    claim about what an operator can go and find, and ``transient`` says why it
    is gone: this command created it in a private directory and removed it on
    purpose, which is a different thing from never having made one.

    One site, driven by the sweep result, because N per-field patches would be
    N places to get wrong and the next nested block would arrive silent again.
    """
    import os as _os

    def _settle(record: dict) -> dict:
        """Presence is MEASURED here, never inferred from the sweep's return.

        Deriving it from ``swept`` would be asserting a fact instead of taking
        one, and it gets a case wrong immediately: a withdrawal that ended in
        ``ownership-lost`` left presence UNKNOWN, and stamping ``False`` over
        that on the strength of "the sweep returned clean" is the same
        overwrite this function exists to stop, one level down. Stat the path.
        Then the claim cannot disagree with the filesystem, which is the rule
        the report is held to, and it stays correct if a later change moves one
        of these artifacts outside the swept directory.
        """
        record = dict(record)
        record["transient"] = True
        record["present"] = _os.path.lexists(record["path"])
        record["removed"] = swept and not record["present"]
        return record

    if isinstance(facts.get("backup"), dict):
        facts["backup"] = _settle(facts["backup"])
        facts["backup_present"] = facts["backup"]["present"]
    if facts.get("copy") is not None:
        # Was a bare string. A path with no presence claim is unfalsifiable,
        # so it becomes a record that states one like every other path does.
        facts["copy"] = _settle({"path": str(facts["copy"])})


def _report_rollback(store: Path, backup: Path) -> int:
    """The real run. It refuses, and the report says which refusal it is.

    NO WORKING DIRECTORY IS CREATED HERE. This used to mkdtemp one before the
    call, and the call refuses at the authority before it observes the source —
    so the directory was made for a run that cannot proceed, stayed empty, and
    the cleanup then reported "a duplicate of every conversation" about zero
    files. The library owns the directory and only creates one if it gets far
    enough to need it, which under this contract is never.
    """
    from hermes_cli import session_fence_rollback as rollback

    outcome = rollback.RollbackOutcome()
    report = None
    refusal = None
    try:
        report = rollback.rollback_turn_fence(
            store, backup_path=backup, outcome=outcome
        )
    except rollback.TurnFenceRollbackRefused as exc:
        refusal = exc
    except (sqlite3.DatabaseError, OSError) as exc:
        # A classified refusal names a next move; this does not, and saying so
        # is the point. The store is untouched either way — every raise inside
        # the boundary rolls its transaction back.
        refusal = _unexpected(exc)

    # THE FACTS THE RUN ESTABLISHED, read from the object the run wrote into
    # and not from a report it may never have returned. Whatever goes wrong
    # after a step does not get to rewrite what that step established: a run
    # whose fence came off and whose backup landed says so even when cleanup
    # then fails, because an operator told "nothing was changed" stops looking
    # for a store they now have to restore.
    established = {
        "changed": outcome.changed,
        "outcome": outcome.outcome,
        "backup_created": outcome.backup_created,
        "backup_verified": outcome.backup_verified,
        "backup_durable": outcome.backup_durable,
        "backup_present": outcome.backup_present,
        "backup_withdrawn": outcome.backup_withdrawn,
        "residue_present": outcome.residue_present,
    }
    residue_records = list(outcome.facts()["residue"])
    if residue_records:
        # A fact nobody reads is not a fact. The library records what it could
        # not remove or could not prove it removed; if that stops here the
        # operator is told about a directory that is clean and a backup that
        # exists, and neither may be true.
        established["residue"] = residue_records
    if outcome.backup is not None:
        established["backup"] = outcome.backup
    if report is not None:
        established["installed_triggers"] = report["installed_triggers"]
        established["dropped_triggers"] = report["dropped_triggers"]
    elif outcome.outcome in ("committing", "committed", "commit-unknown"):
        # The drops were issued. Naming them from the plan rather than from a
        # report that never came back is the difference between an operator who
        # knows what to look for and one who is told there is nothing to find.
        established["dropped_triggers"] = sorted(rollback.rollback_trigger_names())

    if refusal is None and residue_records:
        # Only when nothing else decided the run's fate. Residue is additive to
        # the report and never relabels a refusal that already named one.
        refusal = rollback.TurnFenceRollbackRefused(
            f"the run left {len(residue_records)} thing(s) behind that it "
            "created and could not remove",
            reason="residue-not-removed",
        )
    if refusal is not None:
        return _emit_refusal(
            refusal, store=store, backup=backup, dry_run=False,
            preflight=(report or {}).get("preflight"),
            established=established,
            residue=residue_records,
        )

    _emit(
        {
            "verb": _VERB,
            "ok": True,
            "dry_run": False,
            "changed": True,
            "store": str(store),
            "backup": report["backup"],
            "generation": report["generation"],
            "installed_triggers": report["installed_triggers"],
            "dropped_triggers": report["dropped_triggers"],
            "preflight": report["preflight"],
            "outcome": outcome.outcome,
        }
    )
    return 0


def _unexpected(exc: BaseException):
    """Wrap an unclassified failure so it still arrives with a reason."""
    from hermes_cli import session_fence_rollback as rollback

    return rollback.TurnFenceRollbackRefused(
        f"{type(exc).__name__}: {exc}", reason="unexpected-error"
    )


def _emit_refusal(
    exc,
    *,
    store: Path,
    backup: Path,
    dry_run: bool,
    preflight: dict | None = None,
    established: dict | None = None,
    rehearsal: dict | None = None,
    residue: list | None = None,
    would_drop: list | None = None,
    installed_triggers: list | None = None,
) -> int:
    """Report a failure WITHOUT retracting what already happened.

    A late failure does not undo the facts established before it. The original
    cleanup blocker was residue on disk while claiming success; flattening
    `changed`, `dropped_triggers` and the backup into their empty values on
    every error path is the same defect pointed the other way, and worse — an
    operator told "Nothing was changed." after the fence came off stops
    looking, because they have been told there is nothing to find.

    So failure precedence decides the exit status and the primary reason.
    *established* decides the outcome fields, and it is whatever the run
    actually accomplished before it went wrong.
    """
    from hermes_cli import session_fence_rollback as rollback

    import hermes_state_common

    facts = dict(established or {})
    payload = {
        "verb": _VERB,
        "ok": False,
        "dry_run": dry_run,
        "changed": facts.get("changed", False),
        "outcome": facts.get("outcome", "not-started"),
        "store": str(store),
        # NEVER the bare requested pathname. A string where an operator
        # expects a backup record reads as "your backup is at ...", and on a
        # refusal there is usually nothing there at all.
        "backup": facts.get(
            "backup",
            {"path": str(backup), "created": False, "present": False},
        ),
        "generation": hermes_state_common.TURN_FENCE_GENERATION,
        "installed_triggers": facts.get(
            "installed_triggers", installed_triggers or []
        ),
        "dropped_triggers": facts.get("dropped_triggers", []),
        "backup_present": facts.get("backup_present", False),
        "backup_withdrawn": facts.get("backup_withdrawn", False),
        "preflight": (
            preflight
            or getattr(exc, "preflight", None)
            or rollback._blank_preflight()
        ),
        "refused": {
            "reason": getattr(exc, "reason", "refused"),
            "detail": str(exc),
        },
    }
    records = residue if residue is not None else facts.get("residue")
    if records:
        payload["residue"] = records
    if rehearsal is not None:
        payload["rehearsal"] = rehearsal
    if would_drop is not None:
        payload["would_drop"] = would_drop
    _emit(payload)
    return 1


def _emit(payload: dict[str, Any]) -> None:
    """One JSON document on stdout; the sentence version on stderr."""
    print(json.dumps(payload, indent=2, sort_keys=True))
    for line in _human_lines(payload):
        print(line, file=sys.stderr)


def _human_lines(payload: dict[str, Any]) -> list:
    from hermes_cli import session_fence_rollback as rollback

    if not payload.get("ok"):
        refused = payload.get("refused", {})
        lines = [f"✗ {refused.get('reason')}: {refused.get('detail')}"]
        rehearsal = payload.get("rehearsal") or {}
        if payload.get("changed") is True:
            # The run got further than the failure suggests. Saying "nothing
            # was changed" here would send the operator away from a store whose
            # fence is already off and a backup they need to keep.
            lines.append(
                f"  THE ROLLBACK ITSELF COMPLETED: "
                f"{len(payload.get('dropped_triggers') or [])} trigger(s) were "
                "removed and the verified backup was written. The failure above "
                "is what happened AFTER that, and it still needs action."
            )
            backup = payload.get("backup")
            if isinstance(backup, dict) and backup.get("path"):
                if backup.get("present") is True:
                    lines.append(f"  Verified backup: {backup['path']}")
                elif backup.get("present") is None:
                    lines.append(
                        f"  A backup was written to {backup['path']} and this "
                        "run could not confirm removing it. Something else may "
                        "be at that path now — look before you rely on it."
                    )
                else:
                    lines.append(
                        f"  The backup at {backup['path']} was written and then "
                        "withdrawn by this run. It is NOT there."
                    )

        elif payload.get("changed") is None:
            lines.append(
                "  WHETHER THE ROLLBACK LANDED IS UNKNOWN. The COMMIT was "
                "issued and did not report back, so the fence may or may not "
                "still be installed — check the store before doing anything "
                "else, and keep the backup."
            )
        elif rehearsal.get("changed") is None:
            lines.append(
                "  The store was not modified. The rehearsal's own commit on a "
                "disposable copy did not report back, which is a fault in this "
                "run and not in your store."
            )
        elif rehearsal.get("changed") is True:
            lines.append(
                "  The store was not modified. The rehearsal performed the "
                "rollback on a disposable copy, so the refusal above is the "
                "verdict on the real run, not a failure to run one."
            )
        else:
            lines.append("  Nothing was changed.")
        for record in payload.get("residue") or []:
            # Rendered from the record's own derived fields, so the sentence
            # and the structured output cannot disagree about one incident.
            lines.append(rollback.residue_sentence(record))
        backup = payload.get("backup")
        if (
            isinstance(backup, dict)
            and backup.get("created") is False
            and payload.get("changed") is not True
        ):
            lines.append(
                f"  No backup was written. Nothing is at {backup.get('path')} "
                "because of this run."
            )
        return lines
    if payload.get("dry_run"):
        return [
            f"✓ dry run on {payload['store']}: "
            f"{len(payload['would_drop'])} turn-fence trigger(s) would be "
            "removed.",
            "  The store was not modified.",
            "  This does NOT mean a real run would succeed: this build cannot "
            "prove any artifact you can name is offline, so `--dry-run` is the "
            "only form of this command that does anything.",
        ]
    return [
        f"✓ removed {len(payload['dropped_triggers'])} turn-fence trigger(s) "
        f"from {payload['store']}.",
        f"  Verified backup: {payload['backup']['path']}",
        "  Reopening this store with a current binary reinstalls the fence.",
    ]
