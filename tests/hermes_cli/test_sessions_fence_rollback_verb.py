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
import sqlite3
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
    run = _run_verb(missing, tmpdir / "backup-missing.db")
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

    run = _run_verb(foreign, foreign_backup)
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
    run = _run_verb(junk, tmpdir / "backup-junk.db")
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


def check_a_live_turn_refuses_the_verb_and_no_row_or_trigger_moves(
    tmpdir: pathlib.Path,
) -> None:
    """Offline-only, stated as rows, as triggers, and as an exit code.

    Rolling back mid-turn hands the conversation to a binary that has never
    heard of the lease — the exact interleave the fence was installed to stop,
    arranged by the tool that removes it.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=True)
    rows_before = _canonical_rows(store)
    triggers_before = _installed_triggers(store)
    backup = tmpdir / "backup.db"

    run = _run_verb(store, backup)
    assert not run.crash, f"the verb crashed on a live store: {run.crash}"
    assert run.rc not in (0, None), (
        f"the verb rolled back a store with a live turn: rc={run.rc!r} "
        f"stdout={run.stdout!r}"
    )
    payload = _payload(run)
    assert payload is not None, (
        f"no machine-readable refusal for a live store: {run.stdout!r}"
    )
    assert payload["refused"]["reason"] == "live-turn", (
        "a live turn was not reported as the liveness refusal: "
        f"{payload['refused']!r}"
    )
    assert "keep" in payload["refused"]["detail"], (
        "the refusal does not name the conversation that is live, so the "
        f"operator has nothing to go end: {payload['refused']['detail']!r}"
    )
    assert payload["preflight"]["surface_verified"] is True, (
        "the surface check is reported as not run, yet the liveness check "
        f"refused after it: {payload['preflight']!r}"
    )
    assert payload["preflight"]["offline_verified"] is False

    assert _installed_triggers(store) == triggers_before, (
        "the verb refused a live store and removed triggers anyway"
    )
    assert _canonical_rows(store) == rows_before, (
        "the verb refused a live store and moved rows anyway"
    )
    assert not backup.exists(), (
        "a refused run left a backup file behind"
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

    run = _run_verb(store, backup)
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
    assert run.rc in (0, None), (
        f"the dry run of a rollback that would succeed exited {run.rc!r}: "
        f"{run.stdout!r}"
    )
    payload = _payload(run)
    assert payload is not None, (
        f"the dry run printed no machine-readable plan: {run.stdout!r}"
    )
    assert payload["dry_run"] is True, (
        f"the dry run does not report itself as one: {payload!r}"
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
        "the dry run did not verify liveness, so it cannot say the real run "
        f"would proceed: {payload['preflight']!r}"
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


PINS = {
    "check_the_verb_is_registered_under_sessions_and_names_its_target":
        check_the_verb_is_registered_under_sessions_and_names_its_target,
    "check_a_target_that_is_not_this_fence_is_refused_by_a_named_reason":
        check_a_target_that_is_not_this_fence_is_refused_by_a_named_reason,
    "check_a_live_turn_refuses_the_verb_and_no_row_or_trigger_moves":
        check_a_live_turn_refuses_the_verb_and_no_row_or_trigger_moves,
    "check_a_partial_surface_is_refused_whole_and_writes_no_backup":
        check_a_partial_surface_is_refused_whole_and_writes_no_backup,
    "check_the_dry_run_reports_the_plan_and_changes_no_byte":
        check_the_dry_run_reports_the_plan_and_changes_no_byte,
    "check_the_dry_run_refuses_what_the_real_run_would_refuse":
        check_the_dry_run_refuses_what_the_real_run_would_refuse,
    "check_the_completed_run_reports_the_surface_it_removed":
        check_the_completed_run_reports_the_surface_it_removed,
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
        pin="check_a_live_turn_refuses_the_verb_and_no_row_or_trigger_moves",
        module="hermes_cli/session_fence_rollback.py",
        find=(
            "    except SessionTurnLeaseLostError as exc:\n"
            "        raise TurnFenceRollbackRefused(\n"
            '            f"refusing to roll back the turn fence on {store_path}: {exc}",\n'
            '            reason="live-turn",\n'
            "        ) from exc\n"
        ),
        replace=(
            "    except SessionTurnLeaseLostError:\n"
            "        pass\n"
        ),
        why="swallowing the liveness refusal is how a rollback runs mid-turn: "
            "the triggers come off a conversation somebody owns and the next "
            "write from a pre-fence binary interleaves with it",
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
        find=(
            "        _refuse_if_this_process_owns_a_turn"
            "(copy, identity_path=store_path)\n"
        ),
        replace="        pass\n",
        why="without it the rehearsal asks the liveness question of a copy, "
            "and the one branch of the predicate that is keyed by the store's "
            "PATH answers about the wrong file — so a conversation this "
            "process is mid-turn on reads free, and the dry run reports that "
            "a run which is about to be refused would proceed",
    ),
    Mutation(
        pin="check_the_completed_run_reports_the_surface_it_removed",
        module="hermes_cli/session_fence_rollback_cmd.py",
        find='            "dropped_triggers": report["dropped_triggers"],\n',
        replace='            "dropped_triggers": [],\n',
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
