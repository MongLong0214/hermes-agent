"""``hermes sessions fence-rollback`` — the operator surface, currently a refusal.

WHAT THIS VERB DOES TODAY
    It refuses. Every invocation, with or without ``--dry-run``, before it
    opens, copies or reads the store it was given. It reports which kind of
    target you named and why the rollback is not permitted, and it leaves the
    store byte-for-byte and file-for-file as it found it.

    That is the whole behaviour, and the help strings below say so. They said
    otherwise twice in this slice while the behaviour underneath them had
    already moved, which is the failure an operator meets first.

WHY THERE IS NO REHEARSAL ANY MORE
    There was one: the rollback ran against a private copy assembled by
    checking that no sidecars sat beside the store and then reading its main
    image. Check and read are two operations. A writer that commits into a
    ``-wal`` in between leaves the check's answer true and the image missing
    committed rows — and the run then reported a completed rollback, a verified
    durable backup and a full surface, every one of those derived from a
    database known to be short. Reproduced on both SQLite builds.

    A further existence check does not close that. The interval is what the
    writer uses, and observations do not remove intervals. It is not a guard
    that needs strengthening: a copy assembled by check-then-read cannot be
    proven coherent against a live-capable store, so the rehearsal was never a
    weaker form of the real operation — it was an operation on an artifact
    nobody can vouch for, and every fact downstream inherited that.

    So no plan, no backup, no rehearsal is reported. Those fields are absent
    rather than false, because absent and ``false`` are different statements:
    ``backup_created: false`` would claim the backup step was reached.

WHAT THIS DELIBERATELY IS NOT
    Not a compatibility fallback, not an automatic trigger drop, not a
    discovery pass over the machine's stores. The old binary being unable to
    write a fenced store is intentional and required, and nothing here softens
    it.

    The target is named, never defaulted. A rollback that runs against "the
    store we would have opened anyway" is one wrong host away from the wrong
    file, and the invocation that does it looks identical in the shell history
    to the one they meant.

WHAT IS WAITING BEHIND IT
    A real rollback would need exclusive destination acquisition, an
    engine-written backup, flushes of the file and its directory entry, and an
    ownership-checked withdrawal ledger. That machinery is not in this tree —
    it sat entirely behind :func:`hermes_cli.session_fence_rollback.establish_offline_authority`,
    which refuses unconditionally, so no invocation could ever reach it, and it
    was removed as unreachable rather than kept as untestable. It remains in
    this repository's history for whichever future slice establishes an
    artifact's offline authority.

OUTPUT CONTRACT
    stdout carries exactly one JSON document. Human-readable lines go to
    stderr. That split is what lets a runbook step do
    ``hermes sessions fence-rollback … | jq -r .refused.reason`` while a person
    watching the terminal still reads a sentence.
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
            "Report why a turn-fence rollback is refused; this build performs "
            "none, with or without --dry-run"
        ),
        description=(
            "Step a NAMED session store back off the turn-fence generation "
            "barrier. THIS BUILD REFUSES EVERY INVOCATION, with or without "
            "--dry-run, and does so before it opens, copies or reads the store "
            "you name. The rollback is permitted only on an artifact whose "
            "coherence and detachment its producer established, and nothing "
            "here can establish either: a private copy assembled by checking "
            "for sidecars and then reading the main image cannot be proven "
            "coherent against a live-capable store, because a writer that "
            "commits between the check and the read leaves the check true and "
            "the image short. So no plan, no backup and no rehearsal is "
            "reported — none is produced. What you get is the refusal, its "
            "reason, and a store that is byte-for-byte as you left it. Prints "
            "one JSON report on stdout; human-readable lines go to stderr."
        ),
    )
    parser.add_argument(
        "--store",
        type=Path,
        required=True,
        help=(
            "The session database this refusal is about. Required and never "
            "defaulted — name the file you mean. This build does not roll it "
            "back, and does not open, copy or read it: the refusal is decided "
            "from the pathname before anything is examined"
        ),
    )
    parser.add_argument(
        "--backup",
        type=Path,
        required=True,
        help=(
            "Where an authority-bearing run would write its verified backup. "
            "This build never reaches the backup step, so the path is not "
            "examined and nothing is reported about it — an existing file "
            "there neither changes the outcome nor is mentioned. Still "
            "required so the invocation that eventually performs one names it"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Accepted, and currently identical to a real run: both refuse "
            "before observing the store, for the same reason. No copy is made "
            "and no rollback is rehearsed, because a copy this command "
            "assembles cannot be proven to hold every committed row"
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "ACCEPTED AND UNUSED while this build fails closed. Nothing is "
            "written anywhere, so this path is neither created, examined, nor "
            "required to exist — passing a nonexistent one changes nothing. "
            "Kept so existing invocations keep parsing; removing it is a "
            "public-compatibility decision and not this change"
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
    """``--dry-run``: the same refusal the real run gives, and nothing else.

    IT NO LONGER REHEARSES, AND THAT IS A STATEMENT ABOUT THE INPUT
        The rehearsal ran on a private copy assembled by reading the source's
        main image after checking no sidecars sat beside it. A writer that
        commits into a ``-wal`` between the check and the read leaves the check
        true and the image short — and the run then reported a committed
        rollback, a verified durable backup and a full surface, every one of
        them derived from a database known to be missing rows.

        A copy assembled that way cannot be proven coherent against a
        live-capable source, so the honest report is a refusal with nothing
        attached: no plan, no backup facts, no working directory. Those fields
        are absent because they are not produced, not because they are
        withheld.

    NOTHING IS CREATED. Not a working directory, not a copy. A dry run that
    manufactures an artifact for an operation that cannot proceed leaves the
    operator an incident to clean up that this run invented.
    """
    from hermes_cli import session_fence_rollback as rollback

    # --work-dir IS NOT EXAMINED. Nothing creates a working copy any more, so
    # the flag feeds a capability that no longer exists — and it was still
    # stat'd, and could still change the verdict: a nonexistent path produced
    # `rehearsal-unwritable` where the real path gives the authority refusal,
    # so a dead input decided what the operator was told AND broke the dry
    # run's own same-refusal-as-real contract. When a boundary removes a
    # capability, every input that fed it must stop influencing the outcome.

    outcome = rollback.RollbackOutcome()
    refusal = None
    try:
        rollback.rehearse_turn_fence_rollback(
            store, backup_path=backup, work_dir=work_parent, outcome=outcome
        )
    except rollback.TurnFenceRollbackRefused as exc:
        refusal = exc
    except (sqlite3.DatabaseError, OSError) as exc:
        refusal = _unexpected(exc)

    if refusal is None:  # pragma: no cover - the rehearsal has no success path
        refusal = rollback.TurnFenceRollbackRefused(
            "the rehearsal returned without refusing, which it has no path to do",
            reason="unexpected-error",
        )
    # NO `rehearsal` BLOCK. Nothing was rehearsed, so there are no rehearsal
    # facts — not false ones. Passing the ledger here would publish
    # backup_created=false about a backup step that never ran.
    return _emit_refusal(
        refusal, store=store, backup=backup, dry_run=True,
        residue=list(outcome.facts()["residue"]) or None,
    )


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
    """Report a failure WITHOUT retracting what happened, and without inventing
    answers to questions that were never asked.

    ABSENT AND ``false`` ARE DIFFERENT STATEMENTS, and this is the finer form of
    the output-truth rule the rest of this module already follows.
    ``backup_created: false`` says the backup step was REACHED and did not
    produce one. When the run refuses before deriving anything, no backup step
    ran at all — so rendering ``false`` is a claim about an event that never had
    a chance to occur. The same goes for a ``preflight`` block of all-false
    checks that were never performed, and for an empty ``would_drop``.

    A field that is PRESENT asserts that the question was asked. So every field
    below that describes work is emitted only when that work happened, and the
    keys are absent otherwise. Nothing is withheld; there is nothing to
    withhold.

    A late failure still does not undo the facts established before it: where
    work DID happen, *established* carries it, and failure precedence decides
    only the exit status and the primary reason.
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
        "generation": hermes_state_common.TURN_FENCE_GENERATION,
        "refused": {
            "reason": getattr(exc, "reason", "refused"),
            "detail": str(exc),
        },
    }
    # EACH OF THESE ONLY WHEN THE WORK BEHIND IT HAPPENED.
    resolved_preflight = preflight or getattr(exc, "preflight", None)
    if resolved_preflight:
        payload["preflight"] = resolved_preflight
    if installed_triggers or facts.get("installed_triggers"):
        payload["installed_triggers"] = (
            facts.get("installed_triggers") or installed_triggers
        )
    if facts.get("dropped_triggers"):
        payload["dropped_triggers"] = facts["dropped_triggers"]
    if would_drop:
        payload["would_drop"] = would_drop
    # THE BACKUP DESTINATION IS NEVER EXAMINED BY A NOT-STARTED REFUSAL, so it
    # says nothing about it. `present: false` looked like a cautious default
    # and is objectively wrong the moment the operator names an existing file:
    # the report would be stating as fact something it never looked at. Absent,
    # like every other field describing work that did not happen.
    if facts.get("backup") is not None:
        payload["backup"] = facts["backup"]
    if rehearsal:
        payload["rehearsal"] = rehearsal
    records = residue if residue is not None else facts.get("residue")
    if records:
        payload["residue"] = records
    _emit(payload)
    return 1


def _emit(payload: dict[str, Any]) -> None:
    """One JSON document on stdout; the sentence version on stderr."""
    print(json.dumps(payload, indent=2, sort_keys=True))
    for line in _human_lines(payload):
        print(line, file=sys.stderr)


def _human_lines(payload: dict[str, Any]) -> list:
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
        # `rehearsal and` is the whole guard, and it is not defensive: an
        # ABSENT rehearsal is `{}`, and `{}.get("changed") is None` is true, so
        # without it this arm claims a rehearsal for every payload that reaches
        # it — which is all of them, since nothing in the tree ever publishes
        # the key. Absent and None are different statements here for the same
        # reason _emit_refusal spells out for backup_created: rendering a value
        # is a claim that the step it describes was reached.
        elif rehearsal and rehearsal.get("changed") is None:
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
        # NO RESIDUE LOOP: residue is only ever noted by the removed
        # backup/commit engine (`RollbackOutcome.note_residue`), which
        # nothing on this build's reachable path calls — `payload["residue"]`
        # is never populated, so there is nothing here to render.
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
