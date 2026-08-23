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
        help="Remove the turn-fence triggers from an IDLE store you name",
        description=(
            "Step a NAMED session store back off the turn-fence generation "
            "barrier, offline and all-or-nothing. The store and the backup "
            "path are both given explicitly — this command never discovers a "
            "target and has no default. It refuses unless the store carries "
            "exactly the fence surface this binary declares, no conversation "
            "in it is owned by a live turn, and a verified backup can be "
            "written first; on any refusal nothing is changed. Reopening the "
            "store with a current binary reinstalls the fence. Prints one JSON "
            "report on stdout (success or refusal); human-readable lines go to "
            "stderr."
        ),
    )
    parser.add_argument(
        "--store",
        type=Path,
        required=True,
        help=(
            "The session database to roll back. Required and never defaulted — "
            "name the file you mean"
        ),
    )
    parser.add_argument(
        "--backup",
        type=Path,
        required=True,
        help=(
            "Where to write the verified backup taken before any trigger is "
            "dropped. Must not already exist"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Verify and rehearse on a disposable copy, then report what the "
            "real run would remove. The store is not modified"
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Existing directory for --dry-run's disposable copy of the store "
            "(defaults to the system temporary directory). Point it at a "
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
    """``--dry-run``: rehearse on a copy, report the plan, touch nothing.

    THE COPY DOES NOT GO IN THE OPERATOR'S DIRECTORY. An earlier version put it
    beside the store, reasoning that a hidden subdirectory removed in a
    ``finally`` leaves the listing exactly as it found it. It does — until the
    process dies hard, and then a full UNFENCED duplicate of every conversation
    is sitting next to the original with nothing to draw attention to it.
    ``mkdtemp`` is 0700 and lives where the platform already expects disposable
    copies; ``--work-dir`` moves it for a store too large for that volume.
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
    plan = None
    refusal = None
    try:
        try:
            plan = rollback.rehearse_turn_fence_rollback(
                store, backup_path=backup, work_dir=work_dir
            )
        except rollback.TurnFenceRollbackRefused as exc:
            refusal = exc
        except (sqlite3.DatabaseError, OSError) as exc:
            refusal = _unexpected(exc)
    finally:
        residue = rollback.sweep_work_dir(work_dir)

    # RESIDUE OUTRANKS EVERY OTHER OUTCOME OF A DRY RUN. What is left in there
    # is a rolled-back — that is, UNFENCED — duplicate of every conversation in
    # the store, in a directory the operator was never told about. Reporting
    # "nothing changed" and exiting 0 over the top of that is false twice.
    if residue is not None:
        return _emit_refusal(
            rollback.TurnFenceRollbackRefused(
                f"the dry run could not remove its working copy of {store}: "
                f"{residue['files']} file(s) remain under "
                f"{residue['work_dir']}{residue['error']}. That copy is an "
                "UNFENCED duplicate of every conversation in the store — "
                "remove that directory",
                reason="rehearsal-residue",
            ),
            store=store, backup=backup, dry_run=True,
        )
    if refusal is not None:
        return _emit_refusal(refusal, store=store, backup=backup, dry_run=True)

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
            # The private directory the rehearsal worked in, so a pin — or an
            # operator — can check that it is gone rather than checking a
            # location this code might move next. Naming it here is what stops
            # the observation going blind when the work relocates.
            "rehearsal_work_dir": plan["rehearsal"]["work_dir"],
        }
    )
    return 0


def _report_rollback(store: Path, backup: Path) -> int:
    """The real run. Pre-flight first, and its result is part of the report."""
    import tempfile

    from hermes_cli import session_fence_rollback as rollback

    try:
        work_dir = Path(tempfile.mkdtemp(prefix="hermes-fence-preflight-"))
    except OSError as exc:
        return _emit_refusal(
            _unexpected(exc), store=store, backup=backup, dry_run=False
        )

    report = None
    refusal = None
    try:
        try:
            # ONE call. `rollback_turn_fence` runs the pre-flight itself and
            # returns its result — driving the pre-flight separately here ran
            # it twice against the same working directory, which the exclusive
            # create correctly refused as a collision with our own first copy.
            report = rollback.rollback_turn_fence(
                store, backup_path=backup, work_dir=work_dir
            )
        except rollback.TurnFenceRollbackRefused as exc:
            refusal = exc
        except (sqlite3.DatabaseError, OSError) as exc:
            # A classified refusal names a next move; this does not, and saying
            # so is the point. The store is untouched either way — every raise
            # inside the boundary rolls its transaction back.
            refusal = _unexpected(exc)
    finally:
        residue = rollback.sweep_work_dir(work_dir)

    if residue is not None:
        # Same rule as the dry run, and it applies even when the rollback
        # SUCCEEDED: a copy of the store from before the fence came off is
        # still a copy of every conversation.
        return _emit_refusal(
            rollback.TurnFenceRollbackRefused(
                f"the pre-flight copy of {store} could not be removed: "
                f"{residue['files']} file(s) remain under "
                f"{residue['work_dir']}{residue['error']}. The rollback itself "
                f"{'completed' if report is not None else 'did not complete'}. "
                "That copy is a duplicate of every conversation in the store — "
                "remove that directory",
                reason="preflight-residue",
            ),
            store=store, backup=backup, dry_run=False,
            preflight=(report or {}).get("preflight"),
        )
    if refusal is not None:
        return _emit_refusal(
            refusal, store=store, backup=backup, dry_run=False,
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
) -> int:
    from hermes_cli import session_fence_rollback as rollback

    import hermes_state_common

    _emit(
        {
            "verb": _VERB,
            "ok": False,
            "dry_run": dry_run,
            "changed": False,
            "store": str(store),
            "backup": str(backup),
            "generation": hermes_state_common.TURN_FENCE_GENERATION,
            "dropped_triggers": [],
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
    )
    return 1


def _emit(payload: dict[str, Any]) -> None:
    """One JSON document on stdout; the sentence version on stderr."""
    print(json.dumps(payload, indent=2, sort_keys=True))
    for line in _human_lines(payload):
        print(line, file=sys.stderr)


def _human_lines(payload: dict[str, Any]) -> list:
    if not payload.get("ok"):
        refused = payload.get("refused", {})
        return [
            f"✗ refused ({refused.get('reason')}): {refused.get('detail')}",
            "  Nothing was changed.",
        ]
    if payload.get("dry_run"):
        return [
            f"✓ dry run on {payload['store']}: "
            f"{len(payload['would_drop'])} turn-fence trigger(s) would be "
            "removed.",
            "  The store was not modified. Re-run without --dry-run to "
            "perform it.",
        ]
    return [
        f"✓ removed {len(payload['dropped_triggers'])} turn-fence trigger(s) "
        f"from {payload['store']}.",
        f"  Verified backup: {payload['backup']['path']}",
        "  Reopening this store with a current binary reinstalls the fence.",
    ]
