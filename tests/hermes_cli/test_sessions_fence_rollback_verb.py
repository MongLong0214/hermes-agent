"""`hermes sessions fence-rollback` — the operator's way off the turn fence.

WHY A VERB AND NOT A LIBRARY ENTRY POINT
    ``hermes_cli.session_fence_rollback.rollback_turn_fence`` is bounded,
    offline and fail-closed, and none of that reaches an operator at 3am. What
    reached them was a Python one-liner in a runbook, or the thing the runbook
    replaced: typing ``DROP TRIGGER`` twenty-four times into whatever database
    they believed was the right one. Neither is a rollback story. A one-liner
    has no dry run, no exit code a script can branch on, and no way to say
    WHICH precondition stopped it — and "it failed" is the answer that sends
    someone editing the store by hand.

    So the verb is the deliverable, and what this file pins is the verb:

    * it takes its target BY NAME. No default, no discovery, no "the store we
      would have opened anyway". The counterexample is an operator who runs it
      on the wrong host and rolls back the store they did not mean;
    * a target that is not this fence is refused, and the refusal SAYS WHICH
      kind of wrong it was — missing file, unreadable file, or a database that
      does not carry the surface this binary declares. Three wrong targets,
      three distinct machine-readable reasons;
    * a live turn refuses it, with the rows and the trigger set both asserted
      unchanged and no backup file left behind;
    * a half-installed surface refuses the WHOLE operation, with the remaining
      triggers intact — a verb that drops what it recognises and shrugs at the
      rest leaves a store fenced against some writes and not others;
    * the dry run reports the exact surface it would remove and does not change
      a single byte of the store — not the rows, not the triggers, not the
      file. It rehearses on a disposable copy precisely so that this can be
      asserted as bytes rather than as content;
    * the dry run refuses exactly what the real run refuses. Rehearsing on a
      copy buys the byte assertion and costs an identity: one branch of the
      liveness predicate is keyed by the store's PATH, so without a guard a
      conversation this very process is mid-turn on reads free on the copy and
      held on the original. That was not a hypothesis — the first version of
      the rehearsal reported "this would proceed" about a run that then
      refused, which is worse than having no rehearsal, because the operator
      types the real command on its say-so;
    * and the completed run reports, in machine-readable form, the surface it
      removed and the backup it verified.

WHAT IT DOES NOT SOFTEN
    The old binary being unable to write a fenced v27 store is intentional and
    required. This verb does not add a compatibility fallback, does not drop a
    trigger on open, and does not run unless a person names a store and a
    backup path. Every pin below is about the operator surface; the underlying
    operation's own properties live in ``tests/state/test_turn_fence_rollback``
    and are deliberately not repeated here.

    Temp databases only. No live checkout, no ``state.db``, no service.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from argparse import Namespace
from dataclasses import dataclass

import pytest

import hermes_state_common
from tests.state.lease_mutation_harness import (
    Mutation,
    assert_every_pin_has_a_killer,
    assert_mutation_kills_the_pin,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: This file, so a mutated extract can run the same checks it defines.
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)

#: The verb lives in the CLI package and drives the state layer, so the narrow
#: state-only extract the harness defaults to is not enough.
_EXTRA_EXTRACT = (".",)


# ---------------------------------------------------------------------------
# Driving the verb, with every failure carried as a VALUE.
#
# A pin that dies by an escaping exception proves the code fell over; it does
# not prove the pin could see what the guard protects, and the mutation harness
# scores exactly that distinction. So the import, the dispatch and the JSON
# parse each return a value the pin then asserts on.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VerbRun:
    """One `hermes sessions fence-rollback` invocation, reduced to values."""

    rc: object
    stdout: str
    stderr: str
    crash: str


def _import_verb():
    """The verb module, or the reason there isn't one."""
    try:
        from hermes_cli import session_fence_rollback_cmd as module
    except ImportError as exc:  # pragma: no cover - the RED state
        return None, f"{type(exc).__name__}: {exc}"
    return module, ""


def _run_verb(store, backup, *, dry_run=False, work_dir=None) -> VerbRun:
    """Drive the verb THROUGH `cmd_sessions`, so the wiring is under test too.

    Going straight at the handler would pass while ``hermes sessions
    fence-rollback`` still printed the subcommand help — the operator surface
    is the dispatch, not the function it eventually reaches.
    """
    from hermes_cli.sessions_cmd import cmd_sessions

    args = Namespace(
        sessions_action="fence-rollback",
        store=store,
        backup=backup,
        dry_run=dry_run,
        work_dir=work_dir,
    )
    out, err = io.StringIO(), io.StringIO()
    crash = ""
    rc = None
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cmd_sessions(args)
    except BaseException as exc:  # noqa: BLE001 - carried, not swallowed
        crash = f"{type(exc).__name__}: {exc}"
    return VerbRun(rc=rc, stdout=out.getvalue(), stderr=err.getvalue(), crash=crash)


def _payload(run: VerbRun):
    """The verb's structured report, or None when it did not print one."""
    try:
        return json.loads(run.stdout)
    except Exception:
        return None


@dataclass(frozen=True)
class ParseAttempt:
    errored: bool
    message: str
    namespace: object


def _parse(parser, argv) -> ParseAttempt:
    """argparse's refusal as a value — SystemExit is not an observation."""
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            namespace = parser.parse_args(argv)
    except SystemExit:
        return ParseAttempt(errored=True, message=err.getvalue(), namespace=None)
    return ParseAttempt(errored=False, message=err.getvalue(), namespace=namespace)


# ---------------------------------------------------------------------------
# Fixtures. Temp stores only.
# ---------------------------------------------------------------------------

def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _installed_triggers(path: pathlib.Path) -> list:
    conn = sqlite3.connect(str(path))
    try:
        return sorted(
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' "
                "AND name LIKE 'hermes_turn_fence_%'"
            )
        )
    finally:
        conn.close()


def _canonical_rows(path: pathlib.Path) -> dict:
    """Every row of the tables the fence covers, derived from the declaration.

    Not a typed list of three table names: the surface grew from three tables
    to eight once already, and a snapshot that does not read a table cannot
    notice the rollback moved it. That is the dead-row shape review found in
    the base-binary attempt table.
    """
    tables = sorted({table for table, _op in hermes_state_common.TURN_FENCE_SURFACE})
    conn = sqlite3.connect(str(path))
    try:
        return {
            table: sorted(
                tuple(row) for row in conn.execute(f'SELECT * FROM "{table}"')
            )
            for table in tables
        }
    finally:
        conn.close()


def _store_digest(path: pathlib.Path) -> dict:
    """The store's BYTES — the main file and every sidecar beside it."""
    digest = {}
    for candidate in sorted(path.parent.iterdir()):
        if candidate.is_file() and candidate.name.startswith(path.name):
            digest[candidate.name] = hashlib.sha256(
                candidate.read_bytes()
            ).hexdigest()
    return digest


def _fenced_store(path: pathlib.Path, *, leave_lease_live: bool):
    """A store this generation created, fenced, with rows worth preserving."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=path)
    try:
        db.create_session("keep", "test")
        grant = db.try_acquire_session_turn_lease(
            "keep", _holder("keep"), ttl_seconds=600
        )
        assert grant, "could not take the lease this fixture depends on"
        db.append_message(
            session_id="keep", role="user", content="irreplaceable",
            turn_lease_holder=grant,
        )
        if not leave_lease_live:
            db.release_session_turn_lease("keep", grant)
    finally:
        db.close()
    assert _installed_triggers(path) == sorted(
        hermes_state_common.TURN_FENCE_TRIGGERS
    ), "the fixture did not produce a fenced store"
    return grant


def _hand_the_lease_to_a_foreign_live_owner(store: pathlib.Path) -> None:
    """Re-stamp the lease row so a DIFFERENT, ALIVE process owns the turn.

    WHY NOT JUST HOLD IT IN THIS PROCESS
        Because that exercises the wrong branch.
        ``SessionDB._turn_lease_row_is_free`` frees a row whose ``owner_pid``
        is THIS process unless this process holds a grant for that exact
        ``db_path`` — so a lease held here is decided by the path-keyed branch
        and by :func:`_refuse_if_this_process_owns_a_turn`, never by the
        row-reading predicate. A pin built on it cannot see
        ``_refuse_if_live`` at all, and its mutation row would score a kill
        that a different guard delivered.

        PID 1 exists on every platform this runs on and is not going away
        mid-test, so the row reads as foreign-and-alive: the ``never free``
        branch, which is the case a real operator meets — the gateway holds
        the turn, and the operator runs the rollback from another process.

    The re-stamp goes through ``SessionDB._execute_write``, the store's own
    registered connection, because ``session_turn_leases`` is fenced and a raw
    handle is refused. That is the fixture using the sanctioned door, not a
    hole in it.
    """
    from hermes_state import SessionDB

    db = SessionDB(db_path=store)
    try:
        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE session_turn_leases SET holder = ?, owner_pid = ?, "
                "owner_pid_start = NULL WHERE conversation_id = ?",
                ("pid=1:turn=foreign:platform=test", 1, "keep"),
            )
        )
    finally:
        db.close()


def _sandbox_home(tmpdir: pathlib.Path) -> None:
    home = tmpdir / "hermes-home"
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HERMES_HOME"] = str(home)


# ---------------------------------------------------------------------------
# The pins.
# ---------------------------------------------------------------------------

def check_the_verb_is_registered_under_sessions_and_names_its_target(
    tmpdir: pathlib.Path,
) -> None:
    """No default target, no discovery, and `main()` really registers it.

    The failure this rules out is the convenient one: a verb that defaults to
    "the store we would have opened anyway". An operator who runs a rollback on
    the wrong host then rolls back a store they never named, and the command
    that did it looks identical in their shell history to the one they meant.
    """
    module, why = _import_verb()
    assert module is not None, (
        "there is no `hermes sessions fence-rollback` verb module "
        f"(hermes_cli.session_fence_rollback_cmd): {why}. The only operator "
        "path off the generation fence is a library call or a Python "
        "one-liner in a runbook, which is not a rollback story"
    )

    register = getattr(module, "add_fence_rollback_parser", None)
    assert register is not None, (
        "the verb module exposes no `add_fence_rollback_parser`, so `main()` "
        "has nothing to wire beside `sessions recover`"
    )

    parser = argparse.ArgumentParser(prog="hermes")
    top = parser.add_subparsers(dest="command")
    sessions = top.add_parser("sessions")
    sessions_sub = sessions.add_subparsers(dest="sessions_action")
    register(sessions_sub)

    store = tmpdir / "named-store.db"
    backup = tmpdir / "named-backup.db"

    no_store = _parse(parser, ["sessions", "fence-rollback", "--backup", str(backup)])
    assert no_store.errored, (
        "the verb parsed with no --store, so it carries an implicit default "
        "target: store="
        f"{getattr(no_store.namespace, 'store', None)!r}. A rollback that can "
        "run without being handed a store is a rollback that can run on the "
        "wrong one"
    )

    no_backup = _parse(parser, ["sessions", "fence-rollback", "--store", str(store)])
    assert no_backup.errored, (
        "the verb parsed with no --backup, so it would pick a backup path "
        "itself: backup="
        f"{getattr(no_backup.namespace, 'backup', None)!r}. The operator names "
        "where the copy of their data goes"
    )

    named = _parse(
        parser,
        [
            "sessions", "fence-rollback",
            "--store", str(store), "--backup", str(backup),
        ],
    )
    assert not named.errored, (
        f"the verb refused a fully named invocation: {named.message}"
    )
    assert named.namespace.sessions_action == "fence-rollback", (
        f"the verb registered under the wrong name: {named.namespace!r}"
    )
    assert pathlib.Path(named.namespace.store) == store
    assert pathlib.Path(named.namespace.backup) == backup
    assert named.namespace.dry_run is False, (
        "the verb has no --dry-run, or it defaults to on: "
        f"{named.namespace!r}"
    )

    rehearsal = _parse(
        parser,
        [
            "sessions", "fence-rollback",
            "--store", str(store), "--backup", str(backup), "--dry-run",
        ],
    )
    assert not rehearsal.errored, f"--dry-run does not parse: {rehearsal.message}"
    assert rehearsal.namespace.dry_run is True

    # And `main()` actually calls the registration. A verb nothing wires is a
    # module, not a command.
    import ast

    main_source = (REPO_ROOT / "hermes_cli" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(main_source)
    wired = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "add_fence_rollback_parser"
    ]
    assert wired, (
        "hermes_cli/main.py never calls add_fence_rollback_parser, so the "
        "verb is registered nowhere and `hermes sessions fence-rollback` "
        "prints the sessions help"
    )
    targets = [
        arg.id for call in wired for arg in call.args if isinstance(arg, ast.Name)
    ]
    assert "sessions_subparsers" in targets, (
        "add_fence_rollback_parser is called, but not on the `sessions` "
        f"subparsers it must hang off: {targets}"
    )


def check_a_target_that_is_not_this_fence_is_refused_by_a_named_reason(
    tmpdir: pathlib.Path,
) -> None:
    """Three wrong targets, three DISTINCT reasons, and nothing written.

    "It failed" is the answer that sends an operator editing the store by
    hand. Missing, unreadable, and "a real database that is not this fence"
    are three different next moves, so they must be three different reasons.

    The byte assertion on the foreign database is the load-bearing one: the
    verb must decide the target is wrong BEFORE it opens it as a store, because
    opening someone's unrelated SQLite file as a Hermes store creates the
    Hermes schema inside it.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    reasons = {}

    missing = tmpdir / "there-is-no-store-here.db"
    run = _run_verb(missing, tmpdir / "backup-missing.db", dry_run=True)
    assert not run.crash, f"the verb crashed on a missing target: {run.crash}"
    assert run.rc not in (0, None), (
        f"a missing target exited {run.rc!r}; a script cannot branch on that"
    )
    payload = _payload(run)
    assert payload is not None, (
        f"no machine-readable refusal for a missing target: {run.stdout!r}"
    )
    assert payload.get("ok") is False
    reasons["missing"] = payload["refused"]["reason"]
    assert str(missing) in payload["refused"]["detail"], (
        "the refusal does not name the target it refused: "
        f"{payload['refused']['detail']!r}"
    )

    foreign = tmpdir / "someone-elses-notes.db"
    conn = sqlite3.connect(str(foreign))
    try:
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
        conn.execute("INSERT INTO notes (body) VALUES ('do not touch this')")
        conn.commit()
    finally:
        conn.close()
    foreign_before = _store_digest(foreign)
    foreign_backup = tmpdir / "backup-foreign.db"

    run = _run_verb(foreign, foreign_backup, dry_run=True)
    assert not run.crash, f"the verb crashed on a foreign database: {run.crash}"
    assert run.rc not in (0, None), (
        f"the verb accepted a database that is not a Hermes store: rc={run.rc!r} "
        f"stdout={run.stdout!r}"
    )
    payload = _payload(run)
    assert payload is not None, (
        f"no machine-readable refusal for a foreign database: {run.stdout!r}"
    )
    reasons["foreign"] = payload["refused"]["reason"]
    assert _store_digest(foreign) == foreign_before, (
        "the verb was pointed at a database that is NOT a Hermes store and "
        "wrote it anyway. Deciding the target is wrong only after opening it "
        "as a store means the wrong-target case is the one that damages a "
        f"file: {foreign_before} -> {_store_digest(foreign)}"
    )
    assert not foreign_backup.exists(), (
        "a refused run left a backup file behind, so 'nothing was changed' is "
        "not what happened"
    )
    a_declared_trigger = hermes_state_common.turn_fence_trigger_name(
        "messages", "INSERT"
    )
    assert a_declared_trigger in payload["refused"]["detail"], (
        "the refusal does not say WHAT it expected to find and did not, so the "
        "operator cannot tell a wrong target from a damaged one: "
        f"{payload['refused']['detail']!r}"
    )

    junk = tmpdir / "not-a-database-at-all.db"
    junk.write_bytes(b"this is a text file an operator renamed by mistake\n")
    junk_before = _store_digest(junk)
    run = _run_verb(junk, tmpdir / "backup-junk.db", dry_run=True)
    assert not run.crash, f"the verb crashed on a non-database file: {run.crash}"
    assert run.rc not in (0, None), f"a non-database target exited {run.rc!r}"
    payload = _payload(run)
    assert payload is not None, (
        f"no machine-readable refusal for a non-database file: {run.stdout!r}"
    )
    reasons["junk"] = payload["refused"]["reason"]
    assert _store_digest(junk) == junk_before

    assert len(set(reasons.values())) == 3, (
        "three different wrong targets produced fewer than three distinct "
        f"refusal reasons: {reasons}. The operator can tell THAT it was "
        "refused and not WHICH refusal they hit, and those have different "
        "next moves"
    )


def check_a_partial_surface_is_refused_whole_and_writes_no_backup(
    tmpdir: pathlib.Path,
) -> None:
    """All-or-nothing, and the check runs before anything is written.

    One trigger is removed out of band. A verb that drops what it recognises
    and shrugs at the rest leaves a store fenced against some writes and not
    others, which is worse than either end state — so the WHOLE operation is
    refused, the remaining triggers stay, and no backup is written for a run
    that never ran.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)

    victim = hermes_state_common.turn_fence_trigger_name("messages", "INSERT")
    conn = sqlite3.connect(str(store), isolation_level=None)
    try:
        conn.execute(f"DROP TRIGGER {victim}")
    finally:
        conn.close()

    remaining = _installed_triggers(store)
    # Counted off the DECLARATION. A literal was right while the surface had
    # nine entries and silently wrong the moment it grew to twenty-four.
    expected_after_drop = len(hermes_state_common.TURN_FENCE_TRIGGERS) - 1
    assert victim not in remaining and len(remaining) == expected_after_drop, (
        f"dropping {victim} out of band should leave every other declared "
        f"trigger: expected {expected_after_drop}, found {len(remaining)}"
    )
    rows_before = _canonical_rows(store)
    backup = tmpdir / "backup.db"

    run = _run_verb(store, backup, dry_run=True)
    assert not run.crash, f"the verb crashed on a partial surface: {run.crash}"
    assert run.rc not in (0, None), (
        f"the verb accepted a half-fenced store: rc={run.rc!r} "
        f"stdout={run.stdout!r}"
    )
    payload = _payload(run)
    assert payload is not None, (
        f"no machine-readable refusal for a partial surface: {run.stdout!r}"
    )
    assert payload["refused"]["reason"] == "surface-mismatch", (
        "a half-installed surface was not reported as a surface mismatch, so "
        "the operator cannot tell it apart from a wrong target or a crash: "
        f"{payload['refused']!r}"
    )
    assert victim in payload["refused"]["detail"], (
        "the refusal does not name the trigger that is missing: "
        f"{payload['refused']['detail']!r}"
    )
    assert not backup.exists(), (
        "the verb refused and still wrote a backup, so the run got past its "
        "pre-flight and stopped somewhere later — the refusal is not the "
        "no-change refusal it reports"
    )
    assert _installed_triggers(store) == remaining, (
        "the verb refused and still removed triggers — it is not "
        "all-or-nothing"
    )
    assert _canonical_rows(store) == rows_before


def check_the_dry_run_reports_the_plan_and_changes_no_byte(
    tmpdir: pathlib.Path,
) -> None:
    """A rehearsal that reports the whole plan and does not touch the store.

    Asserted as BYTES, not as content. A dry run that opens the store as a
    store already rewrote it — schema init writes ``messages`` and re-creates
    every trigger — so "the rows look the same afterwards" would pass on a
    store the rehearsal had modified. The bytes are the only statement that
    cannot be satisfied by a rehearsal that quietly wrote.

    THIS PIN IS SQLITE-VERSION-SENSITIVE, AND THE DIRECTORY LISTING IS WHY
        The digest covers the main file AND every sidecar, and the listing
        assertion at the end covers the directory. That pair is not belt and
        braces: they catch different things, and for a while only one build of
        SQLite could show it. Observed on this exact source, same test, same
        arguments, only the interpreter differing::

            SQLite 3.50.4   passed   store opened journal_mode=DELETE (the
                                     WAL-reset-bug fallback), so no sidecar
                                     ever exists and nothing can be left
            SQLite 3.53.1   FAILED   WAL is enabled; a `mode=ro` probe of the
                                     store created `state.db-wal` (0 bytes,
                                     sha e3b0c442…) and `state.db-shm`, and
                                     3.53.1 leaves them behind on close

        `state.db` was byte-identical in both. Nothing was written. The
        directory still gained two files, which is a change to what the
        operator was promised would be left alone — so the fix was to stop the
        dry run opening the store at all, not to loosen this. A future reader
        who sees this pass should not conclude the property is build-independent
        without checking which SQLite they ran on.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    before = _store_digest(store)
    listing_before = sorted(entry.name for entry in tmpdir.iterdir())
    rows_before = _canonical_rows(store)
    backup = tmpdir / "backup.db"

    run = _run_verb(store, backup, dry_run=True)
    assert not run.crash, f"the dry run crashed: {run.crash}"
    payload = _payload(run)
    assert payload is not None, (
        f"the dry run printed no machine-readable plan: {run.stdout!r}"
    )
    assert payload["dry_run"] is True, (
        f"the dry run does not report itself as one: {payload!r}"
    )
    # THE PLAN AND THE VERDICT ARE DIFFERENT ANSWERS. The rehearsal really
    # performs the rollback on a copy this run made, so the plan is observed
    # rather than predicted; the real run is then refused because no target the
    # operator can name has offline authority. Reporting the first without the
    # second is how a dry run says "this would work" about a command that will
    # not run, which is the failure mode a rehearsal exists to remove.
    assert run.rc not in (0, None), (
        f"the dry run exited {run.rc!r} about a real run that refuses. An "
        "operator types the real command on a dry run's say-so"
    )
    assert payload["refused"]["reason"] == "offline-authority-unknown", (
        f"the dry run reports a different verdict than the real run: "
        f"{payload['refused']!r}"
    )
    # THE SENTENCE IS PART OF THE CONTRACT. "Re-run without --dry-run to
    # perform it" is a promise this build cannot keep, and it is the line an
    # operator bases their next action on — the same output-truth rule the
    # ledger is held to, applied where it is actually read.
    assert "Re-run without --dry-run" not in run.stderr, (
        "the dry run tells the operator to run the real command, which this "
        f"build refuses every time:\n{run.stderr}"
    )
    assert "offline-authority-unknown" in run.stderr, (
        f"the refusal an operator reads does not name itself:\n{run.stderr}"
    )
    assert payload["changed"] is False
    assert payload["dropped_triggers"] == [], (
        f"the dry run reports having dropped triggers: {payload!r}"
    )
    assert sorted(payload["would_drop"]) == sorted(
        hermes_state_common.TURN_FENCE_TRIGGERS
    ), (
        "the dry run does not report the surface it would remove, so it "
        "answers 'it would work' and not 'here is what it would do': "
        f"{payload.get('would_drop')!r}"
    )
    assert payload["preflight"]["surface_verified"] is True
    assert payload["preflight"]["offline_verified"] is True, (
        "the dry run did not read the lease table, so its plan rests on less "
        f"than the real run would: {payload['preflight']!r}"
    )
    rehearsal = payload.get("rehearsal") or {}
    assert rehearsal.get("outcome") == "committed", (
        "the rehearsal did not actually perform the rollback, so what is "
        f"reported as a plan is a prediction: {rehearsal!r}"
    )
    assert rehearsal.get("backup_durable") is True, (
        f"the rehearsal reports no durable backup: {rehearsal!r}"
    )

    assert _store_digest(store) == before, (
        "the dry run changed the store it promised only to inspect: "
        f"{before} -> {_store_digest(store)}"
    )
    assert _installed_triggers(store) == sorted(
        hermes_state_common.TURN_FENCE_TRIGGERS
    )
    assert _canonical_rows(store) == rows_before
    assert not backup.exists(), (
        "the dry run wrote the backup it was only supposed to plan"
    )
    assert sorted(entry.name for entry in tmpdir.iterdir()) == listing_before, (
        "the dry run left its working copy behind. A rehearsal on a copy of a "
        "session store leaves an unfenced duplicate of every conversation on "
        "disk if it is not cleaned up: "
        f"{sorted(entry.name for entry in tmpdir.iterdir())}"
    )


def check_the_dry_run_refuses_what_the_real_run_would_refuse(
    tmpdir: pathlib.Path,
) -> None:
    """The rehearsal is only worth anything if it predicts the real run.

    Rehearsing on a copy is what buys the byte assertion, and it costs an
    identity. ``SessionDB._turn_lease_row_is_free`` frees a row whose
    ``owner_pid`` is this process when this process holds no grant FOR THAT
    ``db_path`` — a turn that died without releasing. The copy has a different
    path, so a conversation this very process is genuinely mid-turn on reads
    free on the copy and held on the original.

    Measured, not reasoned about: the first version of this rehearsal reported
    "this would proceed" on exactly the store the real run then refused. A
    rehearsal that predicts the wrong outcome is worse than no rehearsal — the
    operator types the real command on its say-so.

    Both invocations are asserted here TOGETHER, so the claim is the agreement
    rather than either verdict on its own.

    Also SQLite-version-sensitive, for the reason spelled out in
    :func:`check_the_dry_run_reports_the_plan_and_changes_no_byte`: the digest
    below passed on SQLite 3.50.4 and failed on 3.53.1 while `state.db` itself
    was byte-identical, because a `mode=ro` probe of a WAL-mode store leaves
    `-wal` and `-shm` behind on the newer build.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=True)
    before = _store_digest(store)
    listing_before = sorted(entry.name for entry in tmpdir.iterdir())

    rehearsal = _run_verb(store, tmpdir / "backup-dry.db", dry_run=True)
    assert not rehearsal.crash, f"the dry run crashed on a live store: {rehearsal.crash}"
    dry = _payload(rehearsal)
    assert dry is not None, (
        f"the dry run printed no machine-readable report: {rehearsal.stdout!r}"
    )
    # Read BEFORE the real run below. The real run's own liveness check opens
    # the store as a store, and that writes — so a digest taken after it would
    # attribute the real run's bytes to the rehearsal.
    after_rehearsal = _store_digest(store)
    listing_after_rehearsal = sorted(entry.name for entry in tmpdir.iterdir())

    real = _run_verb(store, tmpdir / "backup-real.db")
    assert not real.crash, f"the real run crashed on a live store: {real.crash}"
    live = _payload(real)
    assert live is not None, (
        f"the real run printed no machine-readable report: {real.stdout!r}"
    )
    assert live.get("ok") is False and live["refused"]["reason"] == "live-turn", (
        "the fixture did not produce a store the real run refuses, so this "
        f"check is comparing nothing: {live!r}"
    )

    assert dry.get("ok") is False, (
        "the dry run reported that the rollback WOULD PROCEED on a store the "
        f"real run refuses ({live['refused']['reason']}). The rehearsal is "
        "run on a copy, and the copy does not carry the store's path — so an "
        "answer that depends on the path is an answer about the wrong file: "
        f"{dry!r}"
    )
    assert dry["refused"]["reason"] == live["refused"]["reason"], (
        "the dry run and the real run refused for DIFFERENT reasons "
        f"({dry['refused']['reason']} vs {live['refused']['reason']}), so the "
        "rehearsal is not a rehearsal of this run"
    )

    assert after_rehearsal == before, (
        f"the dry run changed the store while refusing it: {before} -> "
        f"{after_rehearsal}"
    )
    assert listing_after_rehearsal == listing_before, (
        f"a refused dry run left files behind: {listing_after_rehearsal}"
    )


def check_every_preflight_refusal_leaves_the_artifact_byte_identical(
    tmpdir: pathlib.Path,
) -> None:
    """`changed: false` is a claim about BYTES, and it is checked as bytes.

    A refusal decided before the store is opened for writing must leave the
    file map and the directory listing exactly as it found them. Rows and
    triggers are not enough to see this: ``SessionDB.__init__`` runs schema
    init, schema init writes, and a liveness check routed through it therefore
    initialises the store it is asking about. Measured on both SQLite builds —
    a ``live-turn`` refusal moved ``state.db``'s digest while every row and
    every trigger read back identical, and the report still said
    ``changed: false``.

    TWO REFUSALS, ON PURPOSE, AT DIFFERENT DEPTHS
        ``live-turn`` is decided in the middle of the pre-flight and
        ``backup-exists`` after surface verification has already passed, so
        the property being pinned is "no classified pre-flight refusal touches
        the artifact", not "the first check happens to be cheap". A fix that
        made only the earliest refusal inert would pass one of these and fail
        the other.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    outcomes = {}

    # (1) live-turn — a foreign, live owner holds the conversation.
    live_dir = tmpdir / "live"
    live_dir.mkdir()
    live_store = live_dir / "state.db"
    _fenced_store(live_store, leave_lease_live=True)
    _hand_the_lease_to_a_foreign_live_owner(live_store)
    outcomes["live-turn"] = (
        live_store,
        _store_digest(live_store),
        _canonical_rows(live_store),
        _installed_triggers(live_store),
        sorted(entry.name for entry in live_dir.iterdir()),
        _run_verb(live_store, live_dir / "backup.db", dry_run=True),
    )

    # (2) backup-exists — idle store, surface fine, destination occupied.
    taken_dir = tmpdir / "taken"
    taken_dir.mkdir()
    taken_store = taken_dir / "state.db"
    _fenced_store(taken_store, leave_lease_live=False)
    occupied = taken_dir / "backup.db"
    occupied.write_bytes(b"an earlier backup that must not be clobbered")
    outcomes["backup-exists"] = (
        taken_store,
        _store_digest(taken_store),
        _canonical_rows(taken_store),
        _installed_triggers(taken_store),
        sorted(entry.name for entry in taken_dir.iterdir()),
        _run_verb(taken_store, occupied, dry_run=True),
    )

    for reason, (store, digest, rows, triggers, listing, run) in outcomes.items():
        assert not run.crash, f"the verb crashed on the {reason} case: {run.crash}"
        payload = _payload(run)
        assert payload is not None, (
            f"no machine-readable refusal for {reason}: {run.stdout!r}"
        )
        assert run.rc not in (0, None), f"{reason} exited {run.rc!r}"
        assert payload["refused"]["reason"] == reason, (
            f"expected the {reason} refusal, got {payload['refused']!r}"
        )
        assert payload["changed"] is False
        assert _store_digest(store) == digest, (
            f"the {reason} refusal reported changed=false and rewrote the "
            f"artifact. The label is not the contract, the bytes are: "
            f"{digest} -> {_store_digest(store)}"
        )
        assert sorted(entry.name for entry in store.parent.iterdir()) == listing, (
            f"the {reason} refusal added or removed files in the store's "
            f"directory: {sorted(entry.name for entry in store.parent.iterdir())}"
        )
        assert _canonical_rows(store) == rows, f"{reason} moved rows"
        assert _installed_triggers(store) == triggers, f"{reason} moved triggers"

    assert occupied.read_bytes() == b"an earlier backup that must not be clobbered"


def check_a_lease_taken_after_preflight_still_refuses_the_rollback(
    tmpdir: pathlib.Path,
) -> None:
    """Check and use, bound by ONE exclusion boundary. Nothing weaker.

    Every writer in this program must check root, holder and epoch in the same
    transaction as its DML. The verb REMOVES that fence, so it does not get a
    weaker standard — and it had one: liveness was decided in the pre-flight
    and used much later, with only the trigger surface re-checked inside the
    exclusive transaction. Surface is not liveness.

    The barrier below is deterministic, not timed. A real child process takes
    a real lease on the conversation at the one moment that matters — after
    the pre-flight has returned its verdict and before the store is mutated —
    by wrapping the pre-flight itself. No sleeping, no TTL games, no holder
    string heuristics: the child acquires through the ordinary API and its
    grant is confirmed on stdout before the parent is allowed to continue.

    The child then CLOSES its connection and stays alive. That is deliberate:
    with no connection held, ``BEGIN EXCLUSIVE`` succeeds, so the rollback
    cannot be rescued by file locking and has to refuse on the LEASE ROW or
    not at all. An idle connection does not prevent ``BEGIN EXCLUSIVE`` either,
    which is the same point from the other side.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    rows_before = _canonical_rows(store)
    triggers_before = _installed_triggers(store)
    backup = tmpdir / "backup.db"

    child_script = textwrap.dedent(
        """
        import pathlib
        import sys
        from hermes_state import SessionDB

        db = SessionDB(db_path=pathlib.Path(sys.argv[1]))
        grant = db.try_acquire_session_turn_lease(
            "keep", "pid=%d:turn=late:platform=test" % __import__("os").getpid(),
            ttl_seconds=600,
        )
        db.close()
        print("GRANT" if grant else "NO-GRANT", flush=True)
        # Stay alive, holding nothing open, until the parent closes stdin.
        sys.stdin.read()
        """
    )

    started = {}

    def _acquire_the_lease_late(*args, **kwargs):
        """Wrap the pre-flight: let it answer, THEN let a real turn start."""
        verdict = started["real_preflight"](*args, **kwargs)
        if "child" not in started:
            child = subprocess.Popen(
                [sys.executable, "-c", child_script, str(store)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, env=dict(os.environ),
            )
            started["child"] = child
            # One blocking read, not a poll: the child prints exactly one line
            # once the grant is decided, so the parent resumes at a known point.
            started["grant"] = (child.stdout.readline() or "").strip()
        return verdict

    started["real_preflight"] = library.preflight_turn_fence_rollback
    library.preflight_turn_fence_rollback = _acquire_the_lease_late
    try:
        run = _run_verb(store, backup)
    finally:
        library.preflight_turn_fence_rollback = started["real_preflight"]
        child = started.get("child")
        if child is not None:
            try:
                child.stdin.close()
            except Exception:
                pass
            try:
                child.wait(timeout=30)
            except Exception:
                child.kill()

    assert started.get("grant") == "GRANT", (
        "the barrier child did not actually acquire the turn lease "
        f"({started.get('grant')!r}), so this pin measures nothing — it would "
        "pass against a verb with no in-transaction liveness check at all"
    )
    assert not run.crash, f"the verb crashed under the barrier: {run.crash}"

    payload = _payload(run)
    assert payload is not None, f"no machine-readable report: {run.stdout!r}"
    assert run.rc not in (0, None), (
        f"a turn was acquired between the pre-flight and the mutation and the "
        f"rollback went ahead anyway: rc={run.rc!r}. The fence came off "
        "underneath a live conversation, which is the exact interleave it "
        "exists to prevent"
    )
    assert payload.get("ok") is False
    assert payload["refused"]["reason"] == "live-turn", (
        "the late lease was not reported as a liveness refusal: "
        f"{payload['refused']!r}"
    )
    assert "keep" in payload["refused"]["detail"], (
        f"the refusal does not name the conversation: {payload['refused']!r}"
    )

    assert _installed_triggers(store) == triggers_before, (
        "the rollback dropped triggers out from under a live turn"
    )
    # Every protected table EXCEPT the lease table, which the barrier child
    # deliberately wrote — comparing it against a snapshot taken before the
    # child ran would be measuring the child's acquisition, not the rollback's
    # restraint.
    after = _canonical_rows(store)
    for table in sorted(rows_before):
        if table == "session_turn_leases":
            continue
        assert after[table] == rows_before[table], (
            f"the rollback moved {table} under a live turn"
        )
    leases = after["session_turn_leases"]
    assert leases and all(row[1] for row in leases), (
        "the live turn's lease row was cleared by a rollback that refused it: "
        f"{leases}"
    )
    assert not backup.exists(), (
        "a run refused inside the boundary still left a backup on disk, so the "
        "backup does not correspond to a state anything relied on"
    )


def check_a_target_swapped_for_another_valid_store_is_refused(
    tmpdir: pathlib.Path,
) -> None:
    """A path is a name. The operation has to be bound to a FILE.

    Every other counterexample in this file is check/use TIMING. This one is
    check/use SUBJECT, and no amount of re-checking fixes it: rename the named
    store away after the pre-flight, drop a DIFFERENT valid fenced idle store
    at the same path, and every consistency check passes — the surface is the
    declared one, nothing is live, the generation matches — while the backup
    describes the store that left and the twenty-four drops land on the store
    that arrived. Reproduced exactly that way, reporting success.

    So the assertions are about identity, and they are made on BOTH stores:
    the one the operator named must be untouched, and the substitute must be
    untouched too. A run that "did nothing to A" by rolling back B has not
    passed this.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    named = tmpdir / "state.db"
    _fenced_store(named, leave_lease_live=False)
    a_rows = _canonical_rows(named)
    a_triggers = _installed_triggers(named)

    substitute_home = tmpdir / "substitute"
    substitute_home.mkdir()
    substitute = substitute_home / "other.db"
    _fenced_store(substitute, leave_lease_live=False)
    # A must be distinguishable from B, or "the drops hit B" cannot be seen.
    conn = sqlite3.connect(str(substitute))
    try:
        b_sessions = sorted(r[0] for r in conn.execute("SELECT id FROM sessions"))
    finally:
        conn.close()
    b_rows = _canonical_rows(substitute)
    b_triggers = _installed_triggers(substitute)

    moved_aside = tmpdir / "moved-aside.db"
    swapped = {"done": False}
    real_preflight = library.preflight_turn_fence_rollback

    def _swap_the_target_after_the_preflight(*args, **kwargs):
        verdict = real_preflight(*args, **kwargs)
        if not swapped["done"]:
            swapped["done"] = True
            os.rename(named, moved_aside)
            for suffix in ("-wal", "-shm", "-journal"):
                sidecar = named.with_name(named.name + suffix)
                if sidecar.exists():
                    os.rename(sidecar, moved_aside.with_name(moved_aside.name + suffix))
            os.rename(substitute, named)
            for suffix in ("-wal", "-shm", "-journal"):
                sidecar = substitute.with_name(substitute.name + suffix)
                if sidecar.exists():
                    os.rename(sidecar, named.with_name(named.name + suffix))
        return verdict

    library.preflight_turn_fence_rollback = _swap_the_target_after_the_preflight
    try:
        run = _run_verb(named, tmpdir / "backup.db")
    finally:
        library.preflight_turn_fence_rollback = real_preflight

    assert swapped["done"], (
        "the swap never happened, so this pin measures nothing"
    )
    assert not run.crash, f"the verb crashed under the swap: {run.crash}"
    payload = _payload(run)
    assert payload is not None, f"no machine-readable report: {run.stdout!r}"

    assert run.rc not in (0, None), (
        f"the verb exited {run.rc!r} after its target was replaced. The backup "
        "it wrote belongs to one file and the drops landed on another"
    )
    assert payload.get("ok") is False
    assert payload["refused"]["reason"] == "target-replaced", (
        "a substituted target was not reported as one. A consistency check "
        "cannot see this — the substitute is a perfectly valid fenced store — "
        f"so only an identity check can: {payload['refused']!r}"
    )

    # Where A ended up, it must be whole.
    assert _canonical_rows(moved_aside) == a_rows, (
        "the store the operator NAMED lost rows while it was renamed aside"
    )
    assert _installed_triggers(moved_aside) == a_triggers, (
        "the store the operator NAMED lost fence triggers"
    )
    # And the substitute, now sitting at the named path, must be whole too.
    assert _installed_triggers(named) == b_triggers, (
        "the rollback dropped the fence from the SUBSTITUTE store — a file the "
        f"operator never named. Its sessions are {b_sessions}"
    )
    assert _canonical_rows(named) == b_rows, "the substitute lost rows"
    assert not (tmpdir / "backup.db").exists(), (
        "a refused run left a backup behind, and a backup taken across a "
        "target swap describes neither store"
    )


def check_a_destination_appearing_after_the_check_is_never_clobbered(
    tmpdir: pathlib.Path,
) -> None:
    """"Must not already exist" is a create, not a look.

    ``Path.exists()`` and ``shutil.copyfile`` are two operations with a window
    between them, and ``copyfile`` truncates whatever it opens. So a
    destination created in that window is destroyed silently. Reproduced at the
    real seam: a sentinel written after the check was gone afterwards, and the
    rollback reported success and dropped all twenty-four triggers.

    The barrier is deterministic and sits exactly where the defect is — the
    check returns, THEN the sentinel appears, THEN the copy runs. Only an
    acquisition the filesystem arbitrates (``O_CREAT | O_EXCL``) can survive
    it; a second, better-placed check cannot, because a check is what is
    broken.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    rows_before = _canonical_rows(store)
    triggers_before = _installed_triggers(store)
    backup = tmpdir / "backup.db"
    sentinel = b"DO NOT CLOBBER - a concurrent writer got here first"

    injected = {"fired": False, "path": None}
    real_check = library._refuse_unusable_backup_path

    def _competing_writer_after_the_check(path, *args, **kwargs):
        """A racer that re-appears in EVERY window, not just the first.

        The destination check runs more than once — pre-flight, then again at
        the acquisition — and a one-shot injection is caught by the second
        check, which proves nothing: a check saving the run is exactly what is
        not allowed to be the guarantee. So the sentinel is cleared before each
        check and re-created immediately after it, which is what a competing
        writer looks like. Only an acquisition the filesystem arbitrates can
        survive that.

        AIMED AT THE DESTINATION THAT IS ACTUALLY ACQUIRED. Under a contract
        where no target the operator can name has offline authority, the run
        that reaches a backup is the rehearsal, and the destination it acquires
        is its own — so the sentinel goes where the bytes are about to be
        written, which is the only place a clobber could happen. Keyed by the
        name the check is called with rather than by a path this pin computes,
        because the working directory is the command's and not this pin's.
        """
        target = pathlib.Path(path)
        if target.name != "rehearsal-backup.db":
            return real_check(path, *args, **kwargs)
        if target.exists():
            target.unlink()
        real_check(path, *args, **kwargs)
        injected["fired"] = True
        injected["path"] = target
        target.write_bytes(sentinel)

    library._refuse_unusable_backup_path = _competing_writer_after_the_check
    try:
        run = _run_verb(store, backup, dry_run=True)
    finally:
        library._refuse_unusable_backup_path = real_check

    assert injected["fired"], (
        "the sentinel was never written, so the verb never reached the backup "
        "destination check and this pin measures nothing"
    )
    assert not run.crash, f"the verb crashed under the race: {run.crash}"

    payload = _payload(run)
    assert payload is not None, f"no machine-readable report: {run.stdout!r}"
    landed = injected["path"]
    assert landed.read_bytes() == sentinel, (
        "the backup step overwrote a file that appeared after its own "
        "existence check. Whatever was at the destination is gone, and no "
        "check placed anywhere can fix that — the create has to be the check"
    )
    assert run.rc not in (0, None), (
        f"the verb exited {run.rc!r} after racing a destination it was told "
        "not to overwrite"
    )
    assert payload.get("ok") is False
    assert payload["refused"]["reason"] == "backup-exists", (
        f"the collision was not reported as one: {payload['refused']!r}"
    )
    assert _canonical_rows(store) == rows_before
    assert _installed_triggers(store) == triggers_before, (
        "the verb dropped triggers on a run whose backup never landed"
    )


def check_an_orphan_backup_sidecar_is_not_overwritten(
    tmpdir: pathlib.Path,
) -> None:
    """The backup is a FAMILY of files, and every member is a destination.

    An operator whose previous attempt died leaves ``backup.db-wal`` behind
    with no ``backup.db``. "The path you named does not exist" is true of the
    name they typed and useless: the very next WAL-mode store makes
    ``backup.db-wal`` a destination, and the orphan is what gets truncated.

    No race here on purpose — this is the no-race half of the same obligation,
    and it must hold identically on a build that uses WAL and one that does
    not, which is why the refusal is required rather than "the sidecar happens
    not to be written on this build".
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    rows_before = _canonical_rows(store)
    triggers_before = _installed_triggers(store)
    digest_before = _store_digest(store)

    backup = tmpdir / "backup.db"
    orphan = tmpdir / "backup.db-wal"
    orphan_bytes = b"an orphaned -wal from an attempt that died"
    orphan.write_bytes(orphan_bytes)
    assert not backup.exists(), "the fixture wants the MAIN path absent"

    run = _run_verb(store, backup, dry_run=True)

    assert not run.crash, f"the verb crashed on the orphan sidecar: {run.crash}"
    payload = _payload(run)
    assert payload is not None, f"no machine-readable report: {run.stdout!r}"
    assert orphan.read_bytes() == orphan_bytes, (
        "an orphaned backup sidecar was overwritten. The destination check "
        "looked only at the name the operator typed, and a backup is the file "
        "plus its -wal/-shm/-journal siblings"
    )
    assert run.rc not in (0, None), (
        f"the verb exited {run.rc!r} with a member of its backup destination "
        "family already on disk"
    )
    assert payload["refused"]["reason"] == "backup-exists", (
        f"the occupied family member was not reported as one: "
        f"{payload['refused']!r}"
    )
    assert str(orphan) in payload["refused"]["detail"], (
        "the refusal does not name the file that is in the way: "
        f"{payload['refused']['detail']!r}"
    )
    assert _store_digest(store) == digest_before
    assert _canonical_rows(store) == rows_before
    assert _installed_triggers(store) == triggers_before


def check_a_dry_run_that_cannot_clean_up_does_not_report_success(
    tmpdir: pathlib.Path,
) -> None:
    """Cleanup is not housekeeping — what it removes is a copy of the store.

    The rehearsal's working directory holds a full, ROLLED-BACK duplicate of
    every conversation: the fence has been dropped from it, which is the whole
    point of rehearsing. If cleanup does not happen, that duplicate stays on
    disk, unfenced, somewhere the operator was never told about — while the
    command exits 0 and says nothing changed. Data residue and a false report
    in one outcome.

    ``shutil.rmtree(..., ignore_errors=True)`` cannot tell the difference
    between "removed" and "failed and was swallowed", so this drives the seam
    the verb actually uses — the module's own ``shutil`` reference — and makes
    cleanup a no-op. That is the same thing a read-only parent directory, a
    Windows file lock, or an NFS ``EBUSY`` produces, and it is exactly what an
    independent review reproduced by suppressing ``rmtree`` alone.

    WHY THIS PIN HAD TO BE WRITTEN SEPARATELY
        ``check_the_dry_run_reports_the_plan_and_changes_no_byte`` watches the
        STORE'S directory, and the residue moved out of it — the fix for the
        WAL-sidecar finding relocated the working copy to a private temp
        directory. The observation did not fail; it went blind, and silence
        reads as pass. Every location-watching assertion in this file is now
        anchored to a path the REPORT names, so it follows the code.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    work_parent = tmpdir / "work"
    work_parent.mkdir()

    class _CleanupDoesNothing:
        """`shutil` with `rmtree` neutered; everything else passes through."""

        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def rmtree(self, *args, **kwargs):
            return None

    from hermes_cli import session_fence_rollback as library

    real_shutil = getattr(library, "shutil", None)
    assert real_shutil is not None, (
        "the module that performs the removal has no `shutil` to neuter, so "
        "this pin cannot reach the cleanup seam and would pass without "
        "measuring it"
    )
    library.shutil = _CleanupDoesNothing(real_shutil)
    try:
        run = _run_verb(
            store, tmpdir / "backup.db", dry_run=True, work_dir=work_parent
        )
    finally:
        library.shutil = real_shutil

    assert not run.crash, f"the dry run crashed with cleanup suppressed: {run.crash}"

    residue = sorted(entry for entry in work_parent.rglob("*") if entry.is_file())
    assert residue, (
        "cleanup was not actually suppressed, so this pin measures nothing — "
        f"{work_parent} is empty and the check below would pass on a verb that "
        "never cleans up at all"
    )

    payload = _payload(run)
    assert payload is not None, (
        f"the dry run printed no machine-readable report: {run.stdout!r}"
    )
    assert run.rc not in (0, None), (
        f"the dry run exited {run.rc!r} while leaving {len(residue)} file(s) "
        f"under {work_parent} — a rolled-back, UNFENCED duplicate of every "
        "conversation in the store. A script branching on the exit code is "
        "told this run was clean"
    )
    assert payload.get("ok") is False, (
        f"the report claims success with the working copy still on disk: "
        f"{payload!r}"
    )
    assert payload["refused"]["reason"] == "rehearsal-residue", (
        "residue was not reported as its own outcome, so the operator cannot "
        f"tell it apart from a pre-flight refusal: {payload['refused']!r}"
    )

    printed = run.stdout + run.stderr
    assert str(work_parent) in printed, (
        "the failure does not name the directory that has to be removed, so "
        f"the operator has nothing to act on: {printed!r}"
    )
    assert "irreplaceable" not in printed, (
        "the residue report leaked conversation content into stdout/stderr; it "
        "may name the path and count the files and nothing else"
    )


def check_a_late_failure_does_not_retract_what_already_happened(
    tmpdir: pathlib.Path,
) -> None:
    """Failure precedence sets the exit status. It does not rewrite the facts.

    Two situations that must not print the same thing:

    * the rollback COMPLETED and cleanup then failed — twenty-four triggers are
      gone and a verified backup exists;
    * the run was refused BEFORE it mutated anything, and cleanup then failed —
      nothing was dropped and no backup was written.

    Collapsing both into ``changed: false`` / ``dropped_triggers: []`` /
    "Nothing was changed." is the original residue defect pointed the other
    way, and worse: an operator who reads "nothing was changed" after the fence
    came off stops looking. They will not go and find the unfenced store or the
    backup they now depend on, because they have been told there is nothing to
    find.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    class _CleanupDoesNothing:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def rmtree(self, *args, **kwargs):
            return None

    def _run_with_cleanup_suppressed(store, backup):
        real_shutil = library.shutil
        library.shutil = _CleanupDoesNothing(real_shutil)
        try:
            return _run_verb(store, backup)
        finally:
            library.shutil = real_shutil

    # (1) COMPLETED, then cleanup failed.
    done_dir = tmpdir / "completed"
    done_dir.mkdir()
    done_store = done_dir / "state.db"
    _fenced_store(done_store, leave_lease_live=False)
    done_backup = done_dir / "backup.db"
    completed = _run_with_cleanup_suppressed(done_store, done_backup)

    assert not completed.crash, f"the verb crashed: {completed.crash}"
    done = _payload(completed)
    assert done is not None, f"no machine-readable report: {completed.stdout!r}"
    assert _installed_triggers(done_store) == [], (
        "the fixture did not actually complete a rollback, so this pin is "
        "comparing nothing"
    )
    assert done_backup.is_file(), "the fixture did not produce a backup"
    assert done["changed"] is True, (
        "the fence came off and a verified backup was written, and the report "
        f"says changed=false. A late failure retracted an earlier fact: {done!r}"
    )
    assert sorted(done["dropped_triggers"]) == sorted(
        hermes_state_common.TURN_FENCE_TRIGGERS
    ), (
        "the completed surface was flattened to empty by the cleanup failure: "
        f"{done.get('dropped_triggers')!r}"
    )
    assert isinstance(done.get("backup"), dict) and done["backup"].get("verified"), (
        f"the verified backup was dropped from the report: {done.get('backup')!r}"
    )
    assert "Nothing was changed." not in completed.stderr, (
        "the operator is told nothing changed after twenty-four triggers came "
        f"off. They will stop looking:\n{completed.stderr}"
    )
    assert done["refused"]["reason"] == "completed-with-residue", (
        "a completed run with residue is not distinguishable from a run that "
        f"never mutated anything: {done['refused']!r}"
    )
    assert completed.rc not in (0, None), "residue still needs a nonzero exit"

    # (2) REFUSED before mutating, then cleanup failed.
    live_dir = tmpdir / "refused"
    live_dir.mkdir()
    live_store = live_dir / "state.db"
    _fenced_store(live_store, leave_lease_live=True)
    _hand_the_lease_to_a_foreign_live_owner(live_store)
    live_triggers = _installed_triggers(live_store)
    live_backup = live_dir / "backup.db"
    refused = _run_with_cleanup_suppressed(live_store, live_backup)

    assert not refused.crash, f"the verb crashed: {refused.crash}"
    stopped = _payload(refused)
    assert stopped is not None, f"no machine-readable report: {refused.stdout!r}"
    assert stopped["changed"] is False, (
        "a run that never mutated the store reports changed=true: "
        f"{stopped!r}"
    )
    assert stopped["dropped_triggers"] == []
    assert not live_backup.exists(), "a pre-mutation refusal wrote a backup"
    assert _installed_triggers(live_store) == live_triggers
    assert stopped["refused"]["reason"] != done["refused"]["reason"], (
        "a run that completed and a run that never started report the SAME "
        f"outcome ({stopped['refused']['reason']}). Two different situations "
        "that print the same thing is the defect"
    )


def check_the_completed_run_reports_the_surface_it_removed(
    tmpdir: pathlib.Path,
) -> None:
    """Structured output on SUCCESS too, and it says what was removed.

    A verb that prints "done" leaves the operator to go and check by hand what
    it did, which is the state the verb exists to replace.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    rows_before = _canonical_rows(store)
    backup = tmpdir / "backup.db"

    run = _run_verb(store, backup)
    assert not run.crash, f"the verb crashed on a store it should roll back: {run.crash}"
    assert run.rc in (0, None), (
        f"the verb refused an idle, fully fenced store: rc={run.rc!r} "
        f"stdout={run.stdout!r} stderr={run.stderr!r}"
    )
    payload = _payload(run)
    assert payload is not None, (
        f"the verb printed no machine-readable report on success: {run.stdout!r}"
    )
    assert payload["ok"] is True
    assert payload["dry_run"] is False
    assert payload["changed"] is True
    assert sorted(payload["dropped_triggers"]) == sorted(
        hermes_state_common.TURN_FENCE_TRIGGERS
    ), (
        "the completed run does not report the surface it removed, so nothing "
        "in the output distinguishes a full rollback from a partial one: "
        f"{payload.get('dropped_triggers')!r}"
    )
    assert payload["backup"]["verified"] is True, (
        f"the report does not claim a verified backup: {payload.get('backup')!r}"
    )
    assert payload["backup"]["path"] == str(backup)

    assert _installed_triggers(store) == [], (
        "the verb reported success and the fence is still installed"
    )
    assert _canonical_rows(store) == rows_before, (
        "the rollback changed user rows; it is only allowed to remove triggers"
    )
    assert backup.is_file() and _canonical_rows(backup) == rows_before, (
        "the backup the report claims to have verified does not reproduce the "
        "store"
    )


# ---------------------------------------------------------------------------
# The offline-required contract.
#
# The verb's success path is permitted ONLY on an artifact whose offline
# authority was established EXTERNALLY, and this build has no capability that
# can establish it. So every in-place invocation refuses, and the pins below
# are about (a) that the refusal is classified rather than generic and (b) that
# the operation the future authority-bearing caller will drive is nonetheless
# exercised and correct — through the REHEARSAL, which really performs it on a
# copy this run made in its own private directory.
#
# Why the rehearsal is the seam and not a hand-built call: it is what
# ``--dry-run`` drives, with the same arguments, so the backup, durability and
# commit machinery is reached by a production path rather than pinned as dead
# code. A helper whose greenness comes from never running is not covered.
# ---------------------------------------------------------------------------

def _commit_a_marker_that_lives_only_in_the_wal(
    store: pathlib.Path, marker: str
) -> None:
    """Commit *marker* into an UNCHECKPOINTED WAL frame, and leave it there.

    The child exits with ``os._exit`` on purpose. Closing a SQLite connection
    normally checkpoints and removes the ``-wal``, so a marker committed and
    then closed politely ends up in the MAIN FILE and proves nothing — measured
    on both builds, and it is what the retired chimera fixture was actually
    doing while its docstring claimed otherwise. A hard exit skips the close
    path entirely, which is also exactly what an interrupted detach leaves
    behind.

    ``journal_mode=WAL`` is set here rather than inherited: this program opens
    ``journal_mode=DELETE`` on a SQLite build carrying the WAL-reset bug, and
    the property under test is about committed state living outside the main
    file, not about that fallback. A scratch table, because a raw handle
    writing a fenced table is refused by the generation trigger, correctly.
    """
    child = subprocess.run(
        [
            sys.executable, "-c",
            textwrap.dedent(
                """
                import os, sqlite3, sys
                conn = sqlite3.connect(sys.argv[1], isolation_level=None)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA wal_autocheckpoint=0")
                conn.execute("CREATE TABLE IF NOT EXISTS wal_marker (who TEXT)")
                conn.execute(
                    "INSERT INTO wal_marker (who) VALUES (?)", (sys.argv[2],)
                )
                os._exit(0)
                """
            ),
            str(store), marker,
        ],
        capture_output=True, text=True, env=dict(os.environ), timeout=120,
    )
    assert child.returncode == 0, (
        f"the WAL-marker child failed: {child.returncode} {child.stderr}"
    )


def _pragma(path: pathlib.Path, pragma: str):
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute(f"PRAGMA {pragma}").fetchone()[0]
    finally:
        conn.close()


def _family_beside(path: pathlib.Path) -> set:
    """The FILE SET this name owns: every file whose name starts with it."""
    return {
        entry.name
        for entry in path.parent.iterdir()
        if entry.is_file() and entry.name.startswith(path.name)
    }


def _rehearse(library, store, backup, work_dir) -> dict:
    """The rehearsal as VALUES — what it returned, or how it refused.

    ``rehearse_turn_fence_rollback`` is the call ``--dry-run`` makes, with the
    same arguments; the only thing not borrowed from the CLI is the private
    working directory, which the pin owns so the artifacts the rehearsal
    produced can be read as bytes instead of taken from the report.
    """
    outcome = {"plan": None, "reason": "", "detail": "", "crash": ""}
    try:
        outcome["plan"] = library.rehearse_turn_fence_rollback(
            store, backup_path=backup, work_dir=work_dir
        )
    except library.TurnFenceRollbackRefused as exc:
        outcome["reason"] = getattr(exc, "reason", "refused")
        outcome["detail"] = str(exc)
        outcome["plan"] = getattr(exc, "established", None)
    except BaseException as exc:  # noqa: BLE001 - carried, not swallowed
        outcome["crash"] = f"{type(exc).__name__}: {exc}"
    return outcome


def check_no_in_place_run_succeeds_and_each_wrong_target_names_its_own_reason(
    tmpdir: pathlib.Path,
) -> None:
    """Offline is PROVEN or the verb refuses. It is never inferred.

    The retired design tried to establish "nobody is using this store" from
    inside the store: a lease sweep, ``BEGIN EXCLUSIVE``, and a descriptor bound
    to the file. None of the three can carry it. ``BEGIN EXCLUSIVE`` in WAL mode
    excludes WRITERS only; a lease table is a snapshot of what the store was
    told, not of who holds it; and ``O_NOFOLLOW`` pins an inode, not a pathname.

    So the success path is permitted only on a detached artifact whose offline
    authority was established OUTSIDE this verb, and this build has no
    capability that can establish it — the only producer of a detached
    ``state.db`` in the tree is ``create_quick_snapshot``, whose ``manifest.json``
    records SIZE and nothing else, so a same-size replacement satisfies the
    manifest while the contents are a different database. Measured, not argued.

    That makes this pin's subject the REFUSAL, and a refusal is only useful if
    it says which kind of wrong it was — four targets, four next moves:

    * a store with another hard link is in a namespace this run cannot bound:
      a second name reaches the same inode and nothing here governs it;
    * the canonical store is the live one BY DEFINITION, whatever it looks like
      at this instant;
    * SQLite sidecars beside the artifact mean it is attached, or was
      interrupted while attached. Either way it is not quiesced;
    * and everything else is refused for the honest reason: no capability in
      this build can establish authority over it.

    Each leg also asserts the artifact is untouched as BYTES and as a FILE SET.
    "Nothing was changed" is a claim about the directory, not a label.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    home = pathlib.Path(os.environ["HERMES_HOME"])

    # Each case is a target and, where the case IS a sidecar, the sidecar to
    # plant. It is planted AFTER the rows are read, because reading a store
    # with a hot journal beside it makes SQLite roll the journal back and
    # remove it — the fixture would have deleted the condition it is for.
    cases = {}

    # (1) UNTRUSTED NAMESPACE — a second hard link on the artifact.
    shared_dir = tmpdir / "shared"
    shared_dir.mkdir()
    shared = shared_dir / "state.db"
    _fenced_store(shared, leave_lease_live=False)
    os.link(shared, shared_dir / "also-state.db")
    cases["target-untrusted-namespace"] = (shared, None)

    # (2) CANONICAL — ``state.db`` in the profile home this process is
    #     running under. Deliberately NOT pinned through DEFAULT_DB_PATH: this
    #     suite already re-points that, so the two candidates disagree here
    #     exactly as a profile override makes them disagree in production, and
    #     a classifier that consults only one of them calls the operator's real
    #     store unrecognised. Making the fixture agree with the classifier
    #     would have hidden that.
    canonical = home / "state.db"
    _fenced_store(canonical, leave_lease_live=False)
    cases["canonical-store-target"] = (canonical, None)

    # (3) NOT QUIESCED — a sidecar beside it. Written rather than produced by
    #     an open connection so the leg means the same thing on the DELETE-mode
    #     build and the WAL-mode one; an artifact with a hot journal or an
    #     uncheckpointed -wal is exactly what an interrupted detach leaves.
    attached_dir = tmpdir / "attached"
    attached_dir.mkdir()
    attached = attached_dir / "state.db"
    _fenced_store(attached, leave_lease_live=False)
    cases["target-not-quiesced"] = (
        attached, attached.with_name(attached.name + "-journal")
    )

    # (4) EVERYTHING ELSE — a perfectly ordinary, idle, fully fenced store,
    #     which is the case the retired design called a success.
    plain_dir = tmpdir / "plain"
    plain_dir.mkdir()
    plain = plain_dir / "state.db"
    _fenced_store(plain, leave_lease_live=False)
    cases["offline-authority-unknown"] = (plain, None)

    observed = {}
    for reason, (store, sidecar) in sorted(cases.items()):
        backup = store.parent / "backup.db"
        rows_before = _canonical_rows(store)
        triggers_before = _installed_triggers(store)
        if sidecar is not None:
            sidecar.write_bytes(b"a journal from a write that was interrupted")
        digest_before = _store_digest(store)
        listing_before = sorted(entry.name for entry in store.parent.iterdir())

        run = _run_verb(store, backup)
        assert not run.crash, f"the verb crashed on the {reason} case: {run.crash}"
        payload = _payload(run)
        assert payload is not None, (
            f"no machine-readable refusal for {reason}: {run.stdout!r}"
        )
        assert run.rc not in (0, None), (
            f"the verb SUCCEEDED in place on the {reason} target (rc={run.rc!r}). "
            "Nothing in this build can establish that an artifact is offline, so "
            "there is no target it may roll back"
        )
        assert payload["ok"] is False
        assert payload["changed"] is False, (
            f"the {reason} refusal claims it changed the store: {payload!r}"
        )
        assert payload["dropped_triggers"] == []
        observed[reason] = payload["refused"]["reason"]
        assert payload["refused"]["reason"] == reason, (
            f"the {reason} target was refused as {payload['refused']!r}. Four "
            "wrong targets that print the same reason leave the operator with "
            "one next move for four different situations"
        )
        assert _store_digest(store) == digest_before, (
            f"the {reason} refusal rewrote the artifact: {digest_before} -> "
            f"{_store_digest(store)}"
        )
        assert sorted(entry.name for entry in store.parent.iterdir()) == listing_before, (
            f"the {reason} refusal changed the artifact's directory: "
            f"{sorted(entry.name for entry in store.parent.iterdir())}"
        )
        assert not backup.exists(), (
            f"the {reason} refusal wrote a backup for a rollback that never ran"
        )
        # AND THE REPORT MUST NOT IMPLY OTHERWISE. A bare pathname where an
        # operator expects a backup record reads as "your backup is at ...",
        # and there is nothing at that path. Both renderings are checked,
        # because two statements of one outcome that can disagree mean one of
        # them is unpinned.
        assert isinstance(payload["backup"], dict), (
            f"the {reason} refusal reports a backup as a bare path: "
            f"{payload['backup']!r}"
        )
        assert payload["backup"]["created"] is False, (
            f"the {reason} refusal claims a backup was created: "
            f"{payload['backup']!r}"
        )
        assert payload["backup"]["present"] is False, (
            f"the {reason} refusal claims a backup is present: "
            f"{payload['backup']!r}"
        )
        assert "No backup was written." in run.stderr, (
            "the sentence an operator reads does not say there is no backup, "
            f"while the JSON does. One of the two is unpinned:\n{run.stderr}"
        )
        if sidecar is not None:
            # Removed only now, so the rows can be read without SQLite
            # rolling the journal back mid-assertion.
            sidecar.unlink()
        assert _canonical_rows(store) == rows_before, f"{reason} moved rows"
        assert _installed_triggers(store) == triggers_before, (
            f"the {reason} refusal dropped fence triggers"
        )

    assert len(set(observed.values())) == len(observed), (
        f"the four refusals are not distinguishable from each other: {observed!r}"
    )


def check_the_backup_is_a_logical_copy_that_restores_the_pre_rollback_state(
    tmpdir: pathlib.Path,
) -> None:
    """A backup is proven by RESTORING it, and it captures COMMITTED state.

    Two claims, and the second is what makes the first mean anything.

    IT IS TAKEN BY SQLITE, FROM THE SOURCE CONNECTION. Whether that is
    ``VACUUM INTO`` (this repo already uses it, ``hermes_state.py:14858``) or
    the online backup API is not this pin's business — both produce a database
    the engine assembled from the source's committed view, and the property is
    what they have in common, not the artifact one of them happens to leave.
    A copy of the main file has neither: it reproduces whatever bytes are on
    disk, and committed state does not have to be there yet. So the artifact
    carries a row that is COMMITTED and lives only in an uncheckpointed
    ``-wal`` — asserted as bytes, in the main file and in the sidecar, before
    anything is copied — and the restored backup has to contain it.

    IT RESTORES. Not "it opens" — an earlier version proved the backup by
    opening it, and a file that opens can be missing every row the operator
    needs. So it is copied to a fresh path, read as a store, and compared
    against what the artifact carried BEFORE the rollback: the rows, and the
    fence surface that was still on it then.

    The ``-shm`` leg is observed on the FILE SET before the backup is opened,
    because opening a WAL-mode database is itself enough to create one. A
    backup that arrives with a shared-memory file beside it hands its next
    reader a coordination file for a connection that no longer exists.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    rows_before = _canonical_rows(store)
    triggers_before = _installed_triggers(store)

    marker = "committed-but-not-checkpointed"
    _commit_a_marker_that_lives_only_in_the_wal(store, marker)
    wal = store.with_name(store.name + "-wal")
    assert wal.is_file() and marker.encode() in wal.read_bytes(), (
        "the fixture did not leave the row in an uncheckpointed -wal, so a "
        "copy of the main file would carry it and this pin would measure "
        f"nothing: {sorted(_family_beside(store))}"
    )
    assert marker.encode() not in store.read_bytes(), (
        "the row reached the MAIN file, so it is not committed state living "
        "outside it and the property under test is not exercised"
    )

    work_dir = tmpdir / "work"
    work_dir.mkdir()
    outcome = _rehearse(library, store, tmpdir / "backup.db", work_dir)
    assert not outcome["crash"], f"the rehearsal crashed: {outcome['crash']}"

    backup = work_dir / "rehearsal-backup.db"
    assert backup.is_file(), (
        "the rehearsal performed the operation and produced no backup at "
        f"{backup}: {sorted(p.name for p in work_dir.iterdir())} "
        f"(refusal: {outcome['reason']} {outcome['detail']})"
    )
    # THE FILE SET FIRST. Opening the backup can create the -shm this asserts
    # the absence of, so the observation has to precede the open.
    assert _family_beside(backup) == {backup.name}, (
        "the backup arrived as a FAMILY of files. What the engine hands back "
        "is one database and nothing else: "
        f"{sorted(_family_beside(backup))}"
    )

    assert _pragma(backup, "integrity_check") == "ok", (
        f"the backup does not pass integrity_check: "
        f"{_pragma(backup, 'integrity_check')!r}"
    )

    # RESTORE IT — a copy to a path nothing else knows, read as a store.
    restored = tmpdir / "restored.db"
    shutil.copyfile(backup, restored)
    conn = sqlite3.connect(str(restored))
    try:
        rows = conn.execute("SELECT who FROM wal_marker").fetchall()
        held = sorted(str(row[0]) for row in rows)
        content = sorted(
            str(row[0]) for row in conn.execute("SELECT content FROM messages")
        )
    except sqlite3.DatabaseError as exc:
        held = [f"<unreadable: {exc}>"]
        content = []
    finally:
        conn.close()
    assert held == [marker], (
        "the restored backup is missing the row that was committed into the "
        f"-wal: {held!r}. It reproduces the bytes that were in the main file, "
        "which is not the state the source connection had committed"
    )
    assert "irreplaceable" in content, (
        f"the restored store has lost the message rows: {content!r}"
    )
    assert _canonical_rows(restored) == rows_before, (
        "the backup does not restore the rows the artifact carried before the "
        "rollback, so it is not a way back"
    )
    assert _installed_triggers(restored) == triggers_before, (
        "the backup was taken AFTER the drops — it restores a store with no "
        f"fence on it: {_installed_triggers(restored)}"
    )

    # And the artifact the operator named was never the subject.
    assert _installed_triggers(store) == triggers_before
    assert _canonical_rows(store) == rows_before


def check_a_partial_destination_collision_keeps_only_what_the_run_created(
    tmpdir: pathlib.Path,
) -> None:
    """The acquisition is the check, and the cleanup owns only what it made.

    ``VACUUM INTO`` and the online backup API both take a FILENAME; neither can
    inherit a descriptor. So the destination cannot be created by the call that
    writes it, and "must not already exist" has to be re-established where it
    can be: an ``O_CREAT | O_EXCL`` acquisition of the final path, with the
    bytes then going through THAT descriptor.

    A backup destination is a FAMILY, so the acquisition is over the family, and
    the interesting case is the PARTIAL one — the main name free, a sidecar
    occupied. That is what an operator whose previous attempt died leaves
    behind, and it is where a run can both refuse AND leave its own half-built
    destination next to the file it refused to touch. Two obligations, and the
    second is the one that gets dropped: refuse, and then remove ONLY what this
    run created.

    ENTERED WITH THE PRODUCTION PREPARER'S OWN TARGET. The boundary takes a
    private copy that :func:`prepare_the_private_copy` returns, and nothing
    else can produce one — so this pin cannot mint its way in either, which is
    the point of the seal. Driving it through the rehearsal instead does not
    reach this seam on a build that falls back to ``journal_mode=DELETE``, and
    a RED that reds somewhere else goes green when that somewhere else is
    fixed.

    THE PHASE EVENT IS THE ACQUISITION ITSELF. Every exclusive create the run
    makes is recorded, so "it reached the collision after acquiring the main
    destination" is observed rather than inferred from the outcome. A run whose
    acquisitions cannot be seen does not pass this — for a check, silence is
    not evidence.

    NOTHING IS CLAIMED ABOUT A BACKUP THAT DOES NOT EXIST. The snapshot in the
    private staging directory is created and verified before the destination is
    touched, and it is not the operator's backup: reporting ``backup_created``
    when IT appears leaves the ledger saying a backup was made and verified
    while the collision path has just removed every file. So the public facts
    are asserted false here, and the directory is asserted to hold nothing this
    run put in it — including the staging directory, which lives beside the
    BACKUP and not in the work dir the residue pin watches.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)

    work_dir = tmpdir / "work"
    work_dir.mkdir()
    prepared = library.prepare_the_private_copy(store, work_dir=work_dir)
    target = pathlib.Path(prepared.path)
    target_triggers = _installed_triggers(target)
    assert target_triggers == sorted(hermes_state_common.TURN_FENCE_TRIGGERS), (
        f"the prepared copy is not a fenced store: {target_triggers}"
    )

    backup = work_dir / "backup.db"
    squatter = work_dir / "backup.db-shm"
    squatter_bytes = b"a shared-memory file from an attempt that died"

    exclusive_creates = []

    class _RecordingOs:
        """``os``, with every exclusive create recorded as a phase event."""

        def __getattr__(self, name):
            return getattr(os, name)

        def open(self, path, flags, *args, **kwargs):
            if flags & os.O_EXCL:
                exclusive_creates.append(str(path))
            return os.open(path, flags, *args, **kwargs)

    planted = {"fired": False}
    real_check = library._refuse_unusable_backup_path

    def _a_sidecar_appears_after_the_check(path, *args, **kwargs):
        """The squatter arrives AFTER the check returns, in every window.

        A collision the check catches proves the check; the obligation is that
        the acquisition catches it, so the destination is cleared before each
        check and re-occupied immediately after it.
        """
        target_path = pathlib.Path(path)
        if target_path.name == backup.name and squatter.exists():
            squatter.unlink()
        real_check(path, *args, **kwargs)
        if target_path.name == backup.name:
            planted["fired"] = True
            squatter.write_bytes(squatter_bytes)

    outcome = library.RollbackOutcome()
    # What is in the destination's directory before the run: the private copy
    # the preparer made, and nothing else. The squatter arrives during it.
    listing_before = sorted(entry.name for entry in work_dir.iterdir())

    library._refuse_unusable_backup_path = _a_sidecar_appears_after_the_check
    had_os = hasattr(library, "os")
    previous_os = getattr(library, "os", None)
    library.os = _RecordingOs()
    refusal = {"reason": "", "detail": "", "returned": None, "crash": ""}
    try:
        refusal["returned"] = library._commit_the_rollback(
            prepared, backup, sorted(hermes_state_common.TURN_FENCE_TRIGGERS),
            report_as=store, outcome=outcome,
        )
    except library.TurnFenceRollbackRefused as exc:
        refusal["reason"] = getattr(exc, "reason", "refused")
        refusal["detail"] = str(exc)
    except BaseException as exc:  # noqa: BLE001 - carried, not swallowed
        refusal["crash"] = f"{type(exc).__name__}: {exc}"
    finally:
        library._refuse_unusable_backup_path = real_check
        if had_os:
            library.os = previous_os
        else:
            del library.os

    assert planted["fired"], (
        "the collision was never introduced, so this pin measures nothing — "
        "the run never reached its own backup destination"
    )
    assert not refusal["crash"], f"the boundary crashed: {refusal['crash']}"
    assert refusal["reason"] == "backup-exists", (
        "a member of the backup destination family was already there and the "
        f"run did not refuse for that reason: {refusal['reason']!r} "
        f"{refusal['detail']!r}"
    )
    assert squatter.read_bytes() == squatter_bytes, (
        "the run removed a file it did not create. The collision artifact is "
        "the one thing at that destination that is not this run's to touch"
    )
    assert str(backup) in exclusive_creates, (
        "the run never exclusively created the final destination, so it did "
        "not reach the sidecar collision by way of acquiring the main name — "
        "whatever refused it was something else, and the property this pin is "
        f"named for was never exercised: {exclusive_creates!r}"
    )
    assert _family_beside(backup) == {squatter.name}, (
        "the refused run left its own half-built destination behind: "
        f"{sorted(_family_beside(backup))}. An operator who retries now hits a "
        "collision this run manufactured"
    )
    assert sorted(entry.name for entry in work_dir.iterdir()) == sorted(
        listing_before + [squatter.name]
    ), (
        "the refused run left something in the destination's directory that "
        "was not there before and is not the squatter — the staging directory "
        "lives HERE, beside the backup, not in the work dir the residue pin "
        f"watches: {sorted(entry.name for entry in work_dir.iterdir())}"
    )
    facts = outcome.facts()
    assert facts["backup_created"] is False, (
        "the ledger says a backup was created and there is no backup: the "
        f"private staging snapshot is not the operator's backup: {facts!r}"
    )
    assert facts["backup_verified"] is False, (
        f"the ledger verifies a backup that does not exist: {facts!r}"
    )
    assert facts["backup"] is None, (
        f"the ledger carries a backup record for no backup: {facts!r}"
    )
    assert _installed_triggers(target) == target_triggers, (
        "the run dropped triggers on a rollback whose backup never landed"
    )


def check_the_backup_is_fsynced_and_so_is_its_parent_directory(
    tmpdir: pathlib.Path,
) -> None:
    """A backup that is not on the platter is a backup that does not exist.

    The whole reason to take one before removing the fence is a machine that
    stops between the two. ``write()`` returning says the kernel has the bytes;
    it says nothing about the disk. And flushing the FILE is only half of it: an
    unflushed directory entry means the file's contents are durable and its NAME
    is not, so the crash leaves a backup nothing can find.

    So the evidence is the two ``fsync`` calls, observed at the descriptors they
    were made on and matched against the inodes they must have covered — the
    backup's own, and its parent directory's. Reporting a ``durable`` flag would
    only pin the flag.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    work_dir = tmpdir / "work"
    work_dir.mkdir()

    flushed = []

    class _RecordingOs:
        """Every attribute of ``os``, with ``fsync`` recorded as an inode."""

        def __getattr__(self, name):
            return getattr(os, name)

        def fsync(self, fd):
            try:
                info = os.fstat(fd)
                flushed.append((info.st_dev, info.st_ino))
            except OSError:
                pass
            return os.fsync(fd)

    had_os = hasattr(library, "os")
    previous = getattr(library, "os", None)
    library.os = _RecordingOs()
    try:
        outcome = _rehearse(library, store, tmpdir / "backup.db", work_dir)
    finally:
        if had_os:
            library.os = previous
        else:
            del library.os

    assert not outcome["crash"], f"the rehearsal crashed: {outcome['crash']}"
    backup = work_dir / "rehearsal-backup.db"
    assert backup.is_file(), (
        f"the rehearsal produced no backup to flush: {outcome['reason']!r} "
        f"{outcome['detail']!r}"
    )

    backup_info = os.stat(backup)
    parent_info = os.stat(backup.parent)
    assert (backup_info.st_dev, backup_info.st_ino) in flushed, (
        "the backup file was never fsynced. It exists in the page cache and the "
        f"rollback then removed the fence: flushed={flushed!r}"
    )
    assert (parent_info.st_dev, parent_info.st_ino) in flushed, (
        "the backup's parent directory was never fsynced, so the file's bytes "
        "are durable and its NAME is not. A crash here leaves a backup nothing "
        f"can find: flushed={flushed!r}"
    )


def check_a_fault_after_commit_never_reports_that_nothing_changed(
    tmpdir: pathlib.Path,
) -> None:
    """After COMMIT there is no outcome that says "nothing happened".

    ``COMMIT`` runs before the connection is closed and before the report is
    assembled, and everything between them can fail — a close that raises, a
    report that cannot be built. The shape this replaces treated a call that
    raised as a call that did nothing: no report meant ``changed: false``. An
    operator told nothing changed after the fence came off stops looking, and
    the store they now have to restore is the one they were told not to worry
    about.

    Two faults, two outcomes, and neither of them is ``changed: false``:

    * a fault AFTER a COMMIT that returned is ``committed``. The fact was
      established; a later failure sets the exit status and the reason and does
      not get to rewrite it;
    * a fault RAISED BY the COMMIT is ``commit-unknown``, because that is what
      it is. SQLite may have committed and failed afterwards. ``changed`` is
      then neither true nor false and must not be spelled as either — a
      three-state fact rendered as a two-state one always loses the same state.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    def _run_with_a_fault(store, backup, *, break_close: bool):
        """Drive ``--dry-run`` with the rehearsal's own COMMIT sabotaged."""

        class _Connection:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def execute(self, statement, *args, **kwargs):
                result = self._real.execute(statement, *args, **kwargs)
                if not break_close and statement.strip().upper() == "COMMIT":
                    # A COMMIT that really committed and then reported a
                    # failure. Whether the transaction landed is genuinely
                    # unknown to the caller, which is the point.
                    raise sqlite3.OperationalError("disk I/O error after COMMIT")
                return result

            def close(self):
                self._real.close()
                if break_close:
                    raise sqlite3.OperationalError("disk I/O error on close")

        class _Sqlite:
            def __getattr__(self, name):
                return getattr(sqlite3, name)

            def connect(self, target, *args, **kwargs):
                real = sqlite3.connect(target, *args, **kwargs)
                if str(target).endswith("preflight.db"):
                    return _Connection(real)
                return real

        previous = library.sqlite3
        library.sqlite3 = _Sqlite()
        try:
            return _run_verb(store, backup, dry_run=True)
        finally:
            library.sqlite3 = previous

    # (1) The fault lands AFTER a COMMIT that returned.
    done_dir = tmpdir / "committed"
    done_dir.mkdir()
    done_store = done_dir / "state.db"
    _fenced_store(done_store, leave_lease_live=False)
    late = _run_with_a_fault(done_store, done_dir / "backup.db", break_close=True)

    assert not late.crash, f"the verb crashed on the post-commit fault: {late.crash}"
    done = _payload(late)
    assert done is not None, f"no machine-readable report: {late.stdout!r}"
    rehearsal = done.get("rehearsal") or {}
    assert rehearsal.get("outcome") == "committed", (
        "a fault after a COMMIT that returned is reported as something other "
        f"than a completed commit: {rehearsal!r}. The full report was {done!r}"
    )
    assert rehearsal.get("changed") is True, (
        "the rollback committed against the rehearsal's copy and the report "
        f"says it changed nothing: {rehearsal!r}"
    )
    assert rehearsal.get("backup_created") is True, (
        f"the backup was written and the report does not say so: {rehearsal!r}"
    )
    assert rehearsal.get("backup_durable") is True, (
        f"the backup was flushed and the report does not say so: {rehearsal!r}"
    )
    assert late.rc not in (0, None), "a late fault still needs a nonzero exit"

    # (2) The fault IS the COMMIT.
    unknown_dir = tmpdir / "unknown"
    unknown_dir.mkdir()
    unknown_store = unknown_dir / "state.db"
    _fenced_store(unknown_store, leave_lease_live=False)
    ambiguous = _run_with_a_fault(
        unknown_store, unknown_dir / "backup.db", break_close=False
    )

    assert not ambiguous.crash, f"the verb crashed on the failed COMMIT: {ambiguous.crash}"
    unsure = _payload(ambiguous)
    assert unsure is not None, f"no machine-readable report: {ambiguous.stdout!r}"
    unsure_rehearsal = unsure.get("rehearsal") or {}
    assert unsure_rehearsal.get("outcome") == "commit-unknown", (
        "a COMMIT that raised was resolved into a certainty the caller does "
        f"not have: {unsure_rehearsal!r}. The full report was {unsure!r}"
    )
    assert unsure_rehearsal.get("changed") is None, (
        "an unknown commit was rendered as a boolean. Whichever way it is "
        "spelled it is a claim nobody is entitled to make: "
        f"{unsure_rehearsal!r}"
    )
    assert ambiguous.rc not in (0, None), "an unknown commit is not a success"
    assert "Nothing was changed." not in ambiguous.stderr, (
        "the operator is told nothing was changed about a commit whose fate is "
        f"unknown:\n{ambiguous.stderr}"
    )


def check_a_withdrawn_backup_is_never_reported_as_one_the_operator_has(
    tmpdir: pathlib.Path,
) -> None:
    """History and presence are different questions, and only one of them moves.

    ``backup_created`` is a fact about the past: it happened, and a later
    failure does not get to say it did not. ``backup_present`` is a fact about
    now, and withdrawing the backup changes it. Carrying both in one field is
    how a report tells an operator they hold a backup that this very run
    deleted — and an operator who believes they hold one acts more boldly than
    one who knows they do not, which makes that the worst direction to be wrong
    in.

    THE ABSENCE IS FLUSHED. Removing the file and not flushing the directory
    leaves a crash able to bring the entry back while the report says the
    backup was withdrawn. Creating one flushes the parent for that reason;
    removing one owes the same. So the ordering is observed: the unlink, and
    then a flush of the directory it happened in.

    AND THE WITHDRAWAL IS BY IDENTITY. A published path can be replaced between
    the moment the backup lands and the moment a failure decides to take it
    back. Unlinking by name then destroys a file that is not this run's, and
    deletion is the one operation with no way back. So the second leg puts a
    stranger's file at that path and requires the run to leave it alone, say it
    does not know whether its own backup survived, and report what it found
    rather than tidying the question away.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    def _run_until_the_ddl_fails(where: pathlib.Path, meddle=None):
        """Drive the boundary and fail it AFTER the backup is durable.

        The failure is placed in the second decision, which is where a real one
        lands: the backup has been written and flushed and the drops have not
        happened. Keyed on the backup having returned rather than on a call
        count, so the leg says which phase it is in.
        """
        store = where / "state.db"
        _fenced_store(store, leave_lease_live=False)
        work_dir = where / "work"
        work_dir.mkdir()
        prepared = library.prepare_the_private_copy(store, work_dir=work_dir)
        backup = work_dir / "backup.db"
        outcome = library.RollbackOutcome()

        events = []

        class _RecordingOs:
            def __getattr__(self, name):
                return getattr(os, name)

            def fsync(self, fd):
                try:
                    info = os.fstat(fd)
                    events.append(("fsync", info.st_dev, info.st_ino))
                except OSError:
                    pass
                return os.fsync(fd)

            def unlink(self, path, *args, **kwargs):
                events.append(("unlink", str(path)))
                return os.unlink(path, *args, **kwargs)

        state = {"backed_up": False}
        real_backup = library._make_verified_backup
        real_surface = library._refuse_unexpected_surface

        def _backup_then_remember(*args, **kwargs):
            report = real_backup(*args, **kwargs)
            state["backed_up"] = True
            if meddle is not None:
                meddle(backup)
            return report

        def _fail_the_decision_that_follows_the_backup(*args, **kwargs):
            real_surface(*args, **kwargs)
            if state["backed_up"]:
                raise library.TurnFenceRollbackRefused(
                    "the surface moved between the backup and the drops",
                    reason="surface-mismatch",
                )

        library._make_verified_backup = _backup_then_remember
        library._refuse_unexpected_surface = _fail_the_decision_that_follows_the_backup
        had_os = hasattr(library, "os")
        previous_os = getattr(library, "os", None)
        library.os = _RecordingOs()
        result = {"reason": "", "detail": "", "crash": ""}
        try:
            library._commit_the_rollback(
                prepared, backup,
                sorted(hermes_state_common.TURN_FENCE_TRIGGERS),
                report_as=store, outcome=outcome,
            )
        except library.TurnFenceRollbackRefused as exc:
            result["reason"] = getattr(exc, "reason", "refused")
            result["detail"] = str(exc)
        except BaseException as exc:  # noqa: BLE001 - carried, not swallowed
            result["crash"] = f"{type(exc).__name__}: {exc}"
        finally:
            library._make_verified_backup = real_backup
            library._refuse_unexpected_surface = real_surface
            if had_os:
                library.os = previous_os
            else:
                del library.os
        return {
            "store": store, "target": pathlib.Path(prepared.path),
            "backup": backup, "outcome": outcome, "events": events,
            "result": result, "backed_up": state["backed_up"],
        }

    # (1) THE RUN WITHDRAWS ITS OWN BACKUP.
    withdrawn_dir = tmpdir / "withdrawn"
    withdrawn_dir.mkdir()
    one = _run_until_the_ddl_fails(withdrawn_dir)
    assert one["backed_up"], (
        "the backup never landed, so the withdrawal this pin is about never "
        f"happened: {one['result']!r}"
    )
    assert not one["result"]["crash"], f"the boundary crashed: {one['result']['crash']}"
    facts = one["outcome"].facts()
    assert facts["backup_created"] is True, (
        f"a backup was written and the history says it was not: {facts!r}"
    )
    assert facts["backup_present"] is False, (
        "the run deleted its backup and still reports one as present. An "
        f"operator reads that as a file they can go and use: {facts!r}"
    )
    assert facts["backup_withdrawn"] is True, (
        f"the withdrawal is not reported at all: {facts!r}"
    )
    assert facts["changed"] is False, (
        f"nothing was dropped and the run says it changed the store: {facts!r}"
    )
    assert not one["backup"].exists(), (
        "the report says the backup was withdrawn and it is still there"
    )
    parent = os.stat(one["backup"].parent)
    unlinked = [
        index for index, event in enumerate(one["events"])
        if event[0] == "unlink" and event[1] == str(one["backup"])
    ]
    assert unlinked, (
        f"the backup was never unlinked through an observable call: "
        f"{one['events']!r}"
    )
    flushed_after = [
        index for index, event in enumerate(one["events"])
        if event[0] == "fsync"
        and (event[1], event[2]) == (parent.st_dev, parent.st_ino)
        and index > unlinked[-1]
    ]
    assert flushed_after, (
        "the directory entry was removed and never flushed, so a crash can "
        "bring the backup back while the report says it was withdrawn: "
        f"{one['events']!r}"
    )
    assert _installed_triggers(one["target"]) == sorted(
        hermes_state_common.TURN_FENCE_TRIGGERS
    ), "the drops landed on a run that refused before them"

    # (2) SOMETHING ELSE IS AT THAT PATH BY THEN.
    stranger_bytes = b"a different file that arrived at the backup path"

    def _replace_the_backup(path):
        path.unlink()
        path.write_bytes(stranger_bytes)

    swapped_dir = tmpdir / "swapped"
    swapped_dir.mkdir()
    two = _run_until_the_ddl_fails(swapped_dir, meddle=_replace_the_backup)
    assert two["backed_up"], "the backup never landed in the second leg"
    assert not two["result"]["crash"], f"the boundary crashed: {two['result']['crash']}"
    assert two["backup"].read_bytes() == stranger_bytes, (
        "the run deleted a file it did not create. A published path is a name, "
        "and deletion is the operation with no way back"
    )
    swapped_facts = two["outcome"].facts()
    assert swapped_facts["backup_created"] is True
    assert swapped_facts["backup_present"] is None, (
        "the run could not remove its backup and reports a certainty about "
        f"whether one is there: {swapped_facts!r}"
    )
    assert swapped_facts["backup_withdrawn"] is False, (
        f"nothing was withdrawn and the report says it was: {swapped_facts!r}"
    )
    assert swapped_facts["residue_present"] is True, (
        f"what was left behind is not reported: {swapped_facts!r}"
    )
    assert swapped_facts["residue"]["error"] == "ownership-lost", (
        f"the residue does not say why it was left: {swapped_facts['residue']!r}"
    )

    # (3) AND THE VERB SAYS SO, not only the library.
    cli_dir = tmpdir / "surfaced"
    cli_dir.mkdir()
    cli_store = cli_dir / "state.db"
    _fenced_store(cli_store, leave_lease_live=False)
    real_backup = library._make_verified_backup
    real_surface = library._refuse_unexpected_surface
    seen = {"backed_up": False}

    def _backup_then_swap(*args, **kwargs):
        report = real_backup(*args, **kwargs)
        seen["backed_up"] = True
        target = pathlib.Path(report["path"])
        target.unlink()
        target.write_bytes(stranger_bytes)
        return report

    def _fail_after(*args, **kwargs):
        real_surface(*args, **kwargs)
        if seen["backed_up"]:
            raise library.TurnFenceRollbackRefused(
                "the surface moved between the backup and the drops",
                reason="surface-mismatch",
            )

    library._make_verified_backup = _backup_then_swap
    library._refuse_unexpected_surface = _fail_after
    try:
        run = _run_verb(cli_store, cli_dir / "backup.db", dry_run=True)
    finally:
        library._make_verified_backup = real_backup
        library._refuse_unexpected_surface = real_surface

    assert seen["backed_up"], "the rehearsal never reached its backup"
    assert not run.crash, f"the verb crashed: {run.crash}"
    payload = _payload(run)
    assert payload is not None, f"no machine-readable report: {run.stdout!r}"
    rehearsal = payload.get("rehearsal") or {}
    assert rehearsal.get("residue_present") is True, (
        "the library recorded what it could not remove and the verb dropped "
        f"it. A fact nobody reads is not a fact: {payload!r}"
    )
    assert rehearsal.get("backup_present") is None, (
        f"the verb resolved an unknown into a certainty: {rehearsal!r}"
    )


PINS = {
    "check_the_verb_is_registered_under_sessions_and_names_its_target":
        check_the_verb_is_registered_under_sessions_and_names_its_target,
    "check_a_target_that_is_not_this_fence_is_refused_by_a_named_reason":
        check_a_target_that_is_not_this_fence_is_refused_by_a_named_reason,
    "check_a_partial_surface_is_refused_whole_and_writes_no_backup":
        check_a_partial_surface_is_refused_whole_and_writes_no_backup,
    "check_the_dry_run_reports_the_plan_and_changes_no_byte":
        check_the_dry_run_reports_the_plan_and_changes_no_byte,
    "check_the_dry_run_refuses_what_the_real_run_would_refuse":
        check_the_dry_run_refuses_what_the_real_run_would_refuse,
    "check_every_preflight_refusal_leaves_the_artifact_byte_identical":
        check_every_preflight_refusal_leaves_the_artifact_byte_identical,
    "check_a_lease_taken_after_preflight_still_refuses_the_rollback":
        check_a_lease_taken_after_preflight_still_refuses_the_rollback,
    "check_a_target_swapped_for_another_valid_store_is_refused":
        check_a_target_swapped_for_another_valid_store_is_refused,
    "check_a_destination_appearing_after_the_check_is_never_clobbered":
        check_a_destination_appearing_after_the_check_is_never_clobbered,
    "check_an_orphan_backup_sidecar_is_not_overwritten":
        check_an_orphan_backup_sidecar_is_not_overwritten,
    "check_a_dry_run_that_cannot_clean_up_does_not_report_success":
        check_a_dry_run_that_cannot_clean_up_does_not_report_success,
    "check_a_late_failure_does_not_retract_what_already_happened":
        check_a_late_failure_does_not_retract_what_already_happened,
    "check_the_completed_run_reports_the_surface_it_removed":
        check_the_completed_run_reports_the_surface_it_removed,
    "check_no_in_place_run_succeeds_and_each_wrong_target_names_its_own_reason":
        check_no_in_place_run_succeeds_and_each_wrong_target_names_its_own_reason,
    "check_the_backup_is_a_logical_copy_that_restores_the_pre_rollback_state":
        check_the_backup_is_a_logical_copy_that_restores_the_pre_rollback_state,
    "check_a_partial_destination_collision_keeps_only_what_the_run_created":
        check_a_partial_destination_collision_keeps_only_what_the_run_created,
    "check_the_backup_is_fsynced_and_so_is_its_parent_directory":
        check_the_backup_is_fsynced_and_so_is_its_parent_directory,
    "check_a_fault_after_commit_never_reports_that_nothing_changed":
        check_a_fault_after_commit_never_reports_that_nothing_changed,
    "check_a_withdrawn_backup_is_never_reported_as_one_the_operator_has":
        check_a_withdrawn_backup_is_never_reported_as_one_the_operator_has,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_sessions_fence_rollback_verb_property(name, tmp_path):
    PINS[name](tmp_path)


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_the_verb_is_registered_under_sessions_and_names_its_target",
        module="hermes_cli/session_fence_rollback_cmd.py",
        find=(
            '        "--store",\n'
            "        type=Path,\n"
            "        required=True,\n"
        ),
        replace=(
            '        "--store",\n'
            "        type=Path,\n"
            '        default=Path("~/.hermes/state.db").expanduser(),\n'
        ),
        why="giving the target a default is the convenient version of this "
            "verb and the one that rolls back a store nobody named. The "
            "invocation that damages the wrong file then looks identical in "
            "the shell history to the one the operator meant",
    ),
    Mutation(
        pin="check_a_target_that_is_not_this_fence_is_refused_by_a_named_reason",
        module="hermes_cli/session_fence_rollback.py",
        find=(
            "    if list(installed) == list(expected):\n"
            "        return\n"
        ),
        replace=(
            "    if True:\n"
            "        return\n"
        ),
        why="without the surface check the verb decides nothing before it "
            "opens the target as a store — and opening a foreign SQLite file "
            "as a Hermes store creates the Hermes schema (fence triggers "
            "included) inside it, after which the rollback 'succeeds' on a "
            "file that was never a Hermes store",
    ),
    Mutation(
        pin="check_a_partial_surface_is_refused_whole_and_writes_no_backup",
        module="hermes_cli/session_fence_rollback.py",
        find="    if list(installed) == list(expected):\n",
        replace="    if set(installed) <= set(expected):\n",
        why="tolerating a SUBSET is exactly 'drop what we recognise and shrug "
            "at the rest'. The pre-flight passes, a backup is written, and the "
            "run only discovers the missing trigger inside the transaction — "
            "so the operator gets an unclassified failure and a stray backup "
            "instead of a named, no-change refusal",
    ),
    Mutation(
        pin="check_the_dry_run_reports_the_plan_and_changes_no_byte",
        module="hermes_cli/session_fence_rollback_cmd.py",
        find=(
            "    if dry_run:\n"
            "        return _report_rehearsal(store, backup, work_parent)\n"
        ),
        replace=(
            "    if False:\n"
            "        return _report_rehearsal(store, backup, work_parent)\n"
        ),
        why="a --dry-run that falls through to the real operation is worse "
            "than no dry run at all: the operator asked what would happen and "
            "it happened",
    ),
    Mutation(
        pin="check_the_dry_run_refuses_what_the_real_run_would_refuse",
        module="hermes_cli/session_fence_rollback.py",
        find="            report_as=store_path,\n",
        replace="            report_as=None,\n",
        why="without it the rehearsal asks the liveness question of a copy, "
            "and the one branch of the predicate that is keyed by the store's "
            "PATH answers about the wrong file — so a conversation this "
            "process is mid-turn on reads free, and the dry run reports that "
            "a run which is about to be refused would proceed",
    ),
    Mutation(
        pin="check_every_preflight_refusal_leaves_the_artifact_byte_identical",
        module="hermes_cli/session_fence_rollback.py",
        find="        _refuse_if_live(copy, report_as=store_path)\n",
        replace="        _refuse_if_live(store_path)\n",
        why="pointing the liveness check at the operator's store is how the "
            "inspection constructs the thing it is inspecting: SessionDB's "
            "schema init writes, so a refusal has already rewritten the file "
            "it is about to promise was left alone",
    ),
    Mutation(
        pin="check_a_lease_taken_after_preflight_still_refuses_the_rollback",
        module="hermes_cli/session_fence_rollback.py",
        find=(
            "            owned = _live_holders_inside_the_transaction"
            "(conn, reported)\n"
        ),
        replace="            owned = []\n",
        why="without the in-transaction liveness decision the verb is back to "
            "trusting a pre-flight answer across the whole backup step, and a "
            "turn acquired in that window watches all twenty-four triggers go. "
            "Re-checking the SURFACE inside the transaction does not cover it "
            "— surface is not liveness",
    ),
    Mutation(
        pin="check_a_target_swapped_for_another_valid_store_is_refused",
        module="hermes_cli/session_fence_rollback.py",
        find="        with BoundTarget(store_path) as bound:\n",
        replace="        if True:\n            bound = None\n",
        why="with the identity comparison gone the operation is bound to a "
            "NAME again, and a different valid fenced store moved to that name "
            "passes every consistency check while the backup describes the "
            "store that left and the drops land on the one that arrived",
    ),
    Mutation(
        pin="check_a_destination_appearing_after_the_check_is_never_clobbered",
        module="hermes_cli/session_fence_rollback.py",
        find=(
            "    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY\n"
            '    flags |= getattr(os, "O_NOFOLLOW", 0)\n'
            '    flags |= getattr(os, "O_BINARY", 0)\n'
            "    handle = os.open(destination, flags, 0o600)\n"
        ),
        replace=(
            "    flags = os.O_CREAT | os.O_TRUNC | os.O_WRONLY\n"
            '    flags |= getattr(os, "O_BINARY", 0)\n'
            "    handle = os.open(destination, flags, 0o600)\n"
        ),
        why="restoring overwrite-capable creation is the defect itself: "
            "copyfile truncates whatever it opens, so anything that appeared "
            "between the existence check and the write is destroyed. No "
            "better-placed check can fix it, because a check is what is broken",
    ),
    Mutation(
        pin="check_an_orphan_backup_sidecar_is_not_overwritten",
        module="hermes_cli/session_fence_rollback.py",
        find='_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")\n',
        replace="_SIDECAR_SUFFIXES = ()\n",
        why="an orphaned backup.db-wal is a destination too, and narrowing the "
            "family to the one name the operator typed is what makes it "
            "invisible. Aimed at the DECLARATION rather than at either guard "
            "on purpose: the family pre-check is load-bearing on a "
            "journal_mode=DELETE build and redundant on a WAL one, where the "
            "exclusive create refuses first — a mutation aimed at either guard "
            "alone scores a kill on one SQLite and none on the other",
    ),
    Mutation(
        pin="check_a_dry_run_that_cannot_clean_up_does_not_report_success",
        module="hermes_cli/session_fence_rollback.py",
        find="    if not work_dir.exists():\n        return None\n",
        replace="    if True:\n        return None\n",
        why="trusting the removal call instead of looking is the defect: "
            "rmtree with ignore_errors cannot distinguish 'removed' from "
            "'failed and swallowed', so a rolled-back — unfenced — duplicate "
            "of every conversation stays on disk while the verb exits 0 and "
            "reports that nothing changed",
    ),
    Mutation(
        pin="check_a_late_failure_does_not_retract_what_already_happened",
        module="hermes_cli/session_fence_rollback_cmd.py",
        find='            "changed": bool(facts.get("changed", False)),\n',
        replace='            "changed": False,\n',
        why="hardcoding the outcome fields on the failure path is how a run "
            "that dropped twenty-four triggers and wrote a verified backup "
            "announces that nothing changed. The operator stops looking, and "
            "the store they now have to restore is the one they were told not "
            "to worry about",
    ),
    Mutation(
        pin="check_the_completed_run_reports_the_surface_it_removed",
        module="hermes_cli/session_fence_rollback_cmd.py",
        find=(
            '            "installed_triggers": report["installed_triggers"],\n'
            '            "dropped_triggers": report["dropped_triggers"],\n'
            '            "preflight": report["preflight"],\n'
        ),
        replace=(
            '            "installed_triggers": [],\n'
            '            "dropped_triggers": [],\n'
            '            "preflight": report["preflight"],\n'
        ),
        why="a success report that does not name the surface it removed "
            "leaves nothing in the output to distinguish a full rollback from "
            "a partial one, and the operator is back to checking by hand",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    assert_mutation_kills_the_pin(mutation, str(_SELF), tmp_path, *_EXTRA_EXTRACT)


def test_every_pin_has_a_mutation_that_kills_it():
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)
