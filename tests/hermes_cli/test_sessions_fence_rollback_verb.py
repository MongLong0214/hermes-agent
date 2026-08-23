"""`hermes sessions fence-rollback` — a verb that refuses, and what is behind it.

WHAT THE VERB DOES, WHICH IS ALL THIS FILE'S BOUNDARY PINS ARE ABOUT
    It refuses. Every invocation, with or without ``--dry-run``, before it
    opens, copies or reads the store it was given, leaving that store
    byte-for-byte and file-for-file as it found it. It names which kind of
    target it was handed and why the rollback is not permitted, and it reports
    nothing else — no plan, no backup facts, no rehearsal — because it produces
    nothing else.

WHY THERE IS NO REHEARSAL TO PIN
    There was one. The rollback ran against a private copy assembled by
    checking that no sidecars sat beside the store and then reading its main
    image. Check and read are two operations, and a writer that commits into a
    ``-wal`` in between leaves the check's answer true and the image short of
    committed rows — after which the run reported a completed rollback, a
    verified durable backup and a full surface, all of it derived from a
    database known to be missing data. Reproduced on both SQLite builds.

    A further existence check does not close it; the interval is what the
    writer uses, and that move failed three times in this slice at three
    different seams. It is not a guard needing strengthening. A copy assembled
    by check-then-read cannot be proven coherent against a live-capable store,
    so the rehearsal was never a weaker form of the real operation.

TWO KINDS OF PIN IN THIS FILE, AND THEY MUST NOT BE ADDED TOGETHER
    BOUNDARY EVIDENCE — every path refuses before observing the source; the
    source's byte map and file set are invariant; the report carries exactly
    the fields the run produced and no others; a dead input cannot change the
    verdict. These are what the verb is gated on.

    RETAINED MACHINERY — the exclusive destination acquisition, the
    engine-written backup, the flushes, the withdrawal ledger, the residue
    accounting. **None of it is reachable from the verb.** These pins drive it
    directly so it does not rot before the slice that supplies a provably
    coherent artifact, and they are maintenance evidence: they say nothing
    about the boundary. Counting them as boundary coverage would be the same
    error as counting the anchor guard as coverage of the verb.

WHAT IT DOES NOT SOFTEN
    The old binary being unable to write a fenced v27 store is intentional and
    required. This verb adds no compatibility fallback, drops no trigger on
    open, and runs against nothing it was not explicitly given. The target is
    named, never defaulted: a rollback that runs against "the store we would
    have opened anyway" is one wrong host away from the wrong file.

    Temp databases only. No live checkout, no ``state.db``, no service.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import inspect
import io
import json
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import types
import threading
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

    BYTE INVARIANCE IS N/A HERE, AND THE FILE SET IS NOT. Every other boundary
    pin asserts both, and the two travelled together often enough to look like
    one exemption. They are independent: byte invariance needs a FILE, and this
    pin creates none — ``store`` and ``backup`` are argv strings handed to the
    parser, so a digest of a non-existent store asserts nothing. The file set
    needs only a DIRECTORY, and ``tmpdir`` is one. An exemption inherits the
    granularity of the thing exempted, not of the habit, so it is justified
    conjunct by conjunct.

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

    # Nothing is created: these are argv strings, and this pin exercises the
    # PARSER. That is why the file set below is a real assertion here.
    listing_before = sorted(entry.name for entry in tmpdir.iterdir())
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

    # ARGV PARSING IS INERT ON DISK. Nothing else runs the parser in isolation,
    # so nothing else can hold this: a parser that touched the filesystem while
    # reading arguments would be a defect no other pin is placed to see.
    assert sorted(entry.name for entry in tmpdir.iterdir()) == listing_before, (
        "parsing the verb's arguments created or removed files: "
        f"{sorted(entry.name for entry in tmpdir.iterdir())}"
    )


def _drive_the_boundary_and_meddle_after_the_backup(
    library, store: pathlib.Path, work_dir: pathlib.Path, meddle
) -> dict:
    """Enter the mutating boundary on the production preparer's own target.

    The backup lands, *meddle* runs, and then the second decision happens —
    which is the window every check/use property in this verb lives in. The
    boundary takes a private copy only :func:`prepare_the_private_copy` builds,
    so this is the same entry the rehearsal uses and the same one an
    authority-bearing caller would.
    """
    backup = work_dir / "backup.db"
    outcome = library.RollbackOutcome()
    state = {"backed_up": False, "copy": None}
    real_backup = library._make_verified_backup

    def _backup_then_meddle(*args, **kwargs):
        report = real_backup(*args, **kwargs)
        state["backed_up"] = True
        meddle(state["copy"], backup)
        return report

    # ARMED BEFORE THE PREPARER RUNS. A probe that begins at the boundary a fix
    # defends can only confirm that fix; the preparer has its own sequence, and
    # a defect that moves into it is invisible to an observer installed after
    # it returns. That is not hypothetical here — it is how a green suite sat
    # over a live A->B->A for a round.
    library._make_verified_backup = _backup_then_meddle
    result = {"reason": "", "detail": "", "crash": ""}
    try:
        prepared = library.prepare_the_private_copy(store, work_dir=work_dir)
        state["copy"] = copy = pathlib.Path(prepared.path)
        library._commit_the_rollback(
            prepared, backup, sorted(hermes_state_common.TURN_FENCE_TRIGGERS),
            report_as=store, outcome=outcome,
        )
    except library.TurnFenceRollbackRefused as exc:
        result["reason"] = getattr(exc, "reason", "refused")
        result["detail"] = str(exc)
    except BaseException as exc:  # noqa: BLE001 - carried, not swallowed
        result["crash"] = f"{type(exc).__name__}: {exc}"
    finally:
        library._make_verified_backup = real_backup
    return {
        "copy": state["copy"], "backup": backup, "outcome": outcome,
        "result": result, "backed_up": state["backed_up"],
    }


def check_a_target_swapped_for_another_valid_store_is_refused(
    tmpdir: pathlib.Path,
) -> None:
    """A path is a name. The operation has to be bound to a FILE.

    Every other counterexample in this file is check/use TIMING. This one is
    check/use SUBJECT, and no amount of re-checking fixes it: move the target
    away after it has been inspected, drop a DIFFERENT valid fenced idle store
    at the same path, and every consistency check passes — the surface is the
    declared one, nothing is live, the generation matches — while the backup
    describes the store that left and the drops land on the store that arrived.

    IDENTITY MOVED TO THE ARTIFACT THAT IS ACTUALLY MUTATED
        The operator's store is no longer opened for writing by anything, so
        the substitution that matters is of the private copy: it is prepared,
        inspected, backed up, and only then dropped against, and the backup
        window in the middle is a real interval during which its pathname can
        be made to mean a different file. :meth:`_PrivateCopy.verify`
        re-establishes ``(st_dev, st_ino)`` at the point of use rather than
        trusting the object it was handed, and that is what this pins.

    The assertions are about identity, and they are made on BOTH stores: the
    one that was prepared must be untouched where it ended up, and the
    substitute must be untouched too. A run that "did nothing to A" by rolling
    back B has not passed this.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    rows_before = _canonical_rows(store)
    triggers_before = _installed_triggers(store)

    substitute_home = tmpdir / "substitute"
    substitute_home.mkdir()
    substitute = substitute_home / "other.db"
    _fenced_store(substitute, leave_lease_live=False)
    conn = sqlite3.connect(str(substitute))
    try:
        b_sessions = sorted(str(r[0]) for r in conn.execute("SELECT id FROM sessions"))
    finally:
        conn.close()
    b_rows = _canonical_rows(substitute)
    b_triggers = _installed_triggers(substitute)

    work_dir = tmpdir / "work"
    work_dir.mkdir()
    moved_aside = tmpdir / "moved-aside.db"
    swapped = {"done": False}

    def _substitute_the_target_in_the_backup_window(copy, backup):
        if swapped["done"]:
            return
        swapped["done"] = True
        os.rename(copy, moved_aside)
        for suffix in ("-wal", "-shm", "-journal"):
            sidecar = copy.with_name(copy.name + suffix)
            if sidecar.exists():
                os.rename(sidecar, moved_aside.with_name(moved_aside.name + suffix))
        os.rename(substitute, copy)

    run = _drive_the_boundary_and_meddle_after_the_backup(
        library, store, work_dir, _substitute_the_target_in_the_backup_window
    )

    assert run["backed_up"], "the backup never landed, so there was no window"
    assert swapped["done"], "the swap never happened, so this pin measures nothing"
    assert not run["result"]["crash"], f"the boundary crashed: {run['result']['crash']}"
    assert run["result"]["reason"] == "target-replaced", (
        "a substituted target was not reported as one. A consistency check "
        "cannot see this — the substitute is a perfectly valid fenced store — "
        f"so only an identity check can: {run['result']!r}"
    )

    # Where the prepared copy ended up, it must be whole.
    assert _installed_triggers(moved_aside) == sorted(
        hermes_state_common.TURN_FENCE_TRIGGERS
    ), "the copy that was prepared lost fence triggers while it was renamed aside"
    # And the substitute, now sitting at the prepared path, must be whole too.
    assert _installed_triggers(run["copy"]) == b_triggers, (
        "the rollback dropped the fence from the SUBSTITUTE store — a file "
        f"nothing prepared. Its sessions are {b_sessions}"
    )
    assert _canonical_rows(run["copy"]) == b_rows, "the substitute lost rows"
    assert not run["backup"].exists(), (
        "a refused run left a backup behind, and a backup taken across a "
        "target swap describes neither store"
    )
    assert _canonical_rows(store) == rows_before
    assert _installed_triggers(store) == triggers_before


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


def _residue_errors(facts) -> list:
    """Every residue record a run accumulated, as reasons."""
    records = (facts or {}).get("residue") or []
    if isinstance(records, dict):  # pragma: no cover - the shape this replaced
        records = [records]
    return sorted(str(record.get("error")) for record in records)


def _family_beside(path: pathlib.Path) -> set:
    """The FILE SET this name owns: every file whose name starts with it."""
    return {
        entry.name
        for entry in path.parent.iterdir()
        if entry.is_file() and entry.name.startswith(path.name)
    }


def _drive_the_machinery(library, store, work_dir, *, backup=None, outcome=None):
    """Run the rollback machinery directly, on a private object this call makes.

    RETAINED MACHINERY, NOT BOUNDARY EVIDENCE. No production path reaches this
    any more: the verb refuses before deriving anything from a copy it cannot
    prove coherent. What is below the boundary is intact and still has to stay
    that way, because unreachable code that passes is unexecuted rather than
    verified — so these pins drive it explicitly and say so, and none of them
    is evidence about what the verb does.

    They keep the backup, cleanup and withdrawal machinery from rotting until
    the slice that supplies a provably coherent artifact reopens the path.
    """
    outcome = outcome if outcome is not None else library.RollbackOutcome()
    backup = backup if backup is not None else work_dir / "backup.db"
    result = {"returned": None, "reason": "", "detail": "", "crash": "",
              "outcome": outcome, "backup": backup, "copy": None, "prepared": None}
    try:
        prepared = library.prepare_the_private_copy(store, work_dir=work_dir)
        result["prepared"] = prepared
        result["copy"] = pathlib.Path(prepared.path)
        result["returned"] = library._commit_the_rollback(
            prepared, backup, sorted(library.rollback_trigger_names()),
            report_as=store, outcome=outcome,
        )
    except library.TurnFenceRollbackRefused as exc:
        result["reason"] = getattr(exc, "reason", "refused")
        result["detail"] = str(exc)
    except BaseException as exc:  # noqa: BLE001 - carried, not swallowed
        result["crash"] = f"{type(exc).__name__}: {exc}"
    return result


def _rehearse(library, store, backup, work_dir) -> dict:
    """MAINTENANCE-ONLY shape over :func:`_drive_the_machinery`.

    Keeps the destination name the pins built around. The operator's
    ``backup`` argument is deliberately ignored: no production path writes
    there any more, and a pin that watched it would be watching nothing.
    """
    run = _drive_the_machinery(
        library, store, work_dir, backup=work_dir / "rehearsal-backup.db"
    )
    return {"plan": run["returned"], "reason": run["reason"],
            "detail": run["detail"], "crash": run["crash"]}


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
        assert "dropped_triggers" not in payload, (
            f"the {reason} refusal reports a surface it never removed: {payload!r}"
        )
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
        # AND IT SAYS NOTHING ABOUT A DESTINATION IT NEVER LOOKED AT.
        assert "backup" not in payload, (
            f"the {reason} refusal reports on the backup destination without "
            f"having examined it: {payload['backup']!r}"
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

    # ARMED BEFORE THE PREPARER RUNS, then the preparer, then the boundary. A
    # probe that begins at the boundary a fix defends can only confirm that
    # fix — the preparer has its own sequence and its own seams.
    library._refuse_unusable_backup_path = _a_sidecar_appears_after_the_check
    had_os = hasattr(library, "os")
    previous_os = getattr(library, "os", None)
    library.os = _RecordingOs()
    refusal = {"reason": "", "detail": "", "returned": None, "crash": ""}
    target = None
    try:
        prepared = library.prepare_the_private_copy(store, work_dir=work_dir)
        target = pathlib.Path(prepared.path)
        target_triggers = _installed_triggers(target)
        assert target_triggers == sorted(hermes_state_common.TURN_FENCE_TRIGGERS), (
            f"the prepared copy is not a fenced store: {target_triggers}"
        )
        listing_before = sorted(entry.name for entry in work_dir.iterdir())
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


def _fsync_inodes_during_a_rehearsal(library, store, work_dir) -> dict:
    """Every ``fsync`` the rehearsal makes, recorded as the inode it covered.

    Observed at the descriptor rather than trusted from a ``durable`` flag,
    because reporting a flag would only ever pin the flag.
    """
    flushed = []

    class _RecordingOs:
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
        outcome = _rehearse(library, store, work_dir.parent / "backup.db", work_dir)
    finally:
        if had_os:
            library.os = previous
        else:
            del library.os
    return {"flushed": flushed, "outcome": outcome}


def check_the_backup_file_itself_is_flushed_to_the_platter(
    tmpdir: pathlib.Path,
) -> None:
    """``write()`` returning says the KERNEL has the bytes. Not the disk.

    The whole reason to take a backup before removing the fence is a machine
    that stops between the two, and a backup that exists only in the page cache
    is not there for exactly the failure it was taken against.

    Split from the directory-entry half on purpose: the two are different
    syscalls at different seams, either can be removed without the other, and a
    single pin asserting both can only be killed by whichever mutation the
    table happens to name — leaving the other half asserted and unproven.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    work_dir = tmpdir / "work"
    work_dir.mkdir()

    seen = _fsync_inodes_during_a_rehearsal(library, store, work_dir)
    assert not seen["outcome"]["crash"], f"the rehearsal crashed: {seen['outcome']['crash']}"
    backup = work_dir / "rehearsal-backup.db"
    assert backup.is_file(), (
        f"the rehearsal produced no backup to flush: {seen['outcome']['reason']!r}"
    )
    info = os.stat(backup)
    assert (info.st_dev, info.st_ino) in seen["flushed"], (
        "the backup file was never fsynced. It exists in the page cache and "
        f"the rollback then removed the fence: flushed={seen['flushed']!r}"
    )


def check_the_backups_directory_entry_is_flushed_too(
    tmpdir: pathlib.Path,
) -> None:
    """Flushing the file is half of it. The NAME has to survive as well.

    An unflushed directory entry means the backup's contents are durable and
    the entry pointing at them is not, so the crash this backup exists for
    leaves a file nothing can find. Same obligation, different syscall, and it
    is the half that gets forgotten.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    work_dir = tmpdir / "work"
    work_dir.mkdir()

    seen = _fsync_inodes_during_a_rehearsal(library, store, work_dir)
    assert not seen["outcome"]["crash"], f"the rehearsal crashed: {seen['outcome']['crash']}"
    backup = work_dir / "rehearsal-backup.db"
    assert backup.is_file(), (
        f"the rehearsal produced no backup: {seen['outcome']['reason']!r}"
    )
    parent = os.stat(backup.parent)
    assert (parent.st_dev, parent.st_ino) in seen["flushed"], (
        "the backup's parent directory was never fsynced, so the file's bytes "
        "are durable and its NAME is not. A crash here leaves a backup nothing "
        f"can find: flushed={seen['flushed']!r}"
    )


def check_a_fault_after_commit_never_reports_that_nothing_changed(
    tmpdir: pathlib.Path,
) -> None:
    """MAINTENANCE ONLY. After COMMIT there is no outcome saying "nothing happened".

    ``COMMIT`` runs before the connection is read back and before any report is
    assembled, and everything between can fail. The shape this replaces treated
    a call that raised as a call that did nothing.

    Two faults, two outcomes, and neither is ``changed: false``:

    * a fault AFTER a COMMIT that returned is ``committed`` — the fact was
      established and a later failure does not rewrite it;
    * a fault RAISED BY the COMMIT is ``commit-unknown``, because that is what
      it is. ``changed`` is then neither true nor false, and a three-state fact
      rendered as two always loses the same state.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    def _run_with_a_fault(where, *, after_commit: bool):
        where.mkdir()
        store = where / "state.db"
        _fenced_store(store, leave_lease_live=False)
        work = where / "work"
        work.mkdir()

        class _Connection:
            def __init__(self, real):
                self._real = real
                self._committed = False

            def __getattr__(self, name):
                return getattr(self._real, name)

            def execute(self, statement, *args, **kwargs):
                if after_commit and self._committed:
                    raise sqlite3.OperationalError("disk I/O error after COMMIT")
                result = self._real.execute(statement, *args, **kwargs)
                if statement.strip().upper() == "COMMIT":
                    self._committed = True
                    if not after_commit:
                        raise sqlite3.OperationalError("disk I/O error on COMMIT")
                return result

        real_prepare = library.prepare_the_private_copy

        def _prepare_then_sabotage(*args, **kwargs):
            prepared = real_prepare(*args, **kwargs)
            prepared.connection = _Connection(prepared.connection)
            return prepared

        library.prepare_the_private_copy = _prepare_then_sabotage
        try:
            return _drive_the_machinery(library, store, work)
        finally:
            library.prepare_the_private_copy = real_prepare

    # (1) The fault lands AFTER a COMMIT that returned. Nothing reads the
    #     connection again inside the boundary, so the fault surfaces on the
    #     ledger rather than as an exception -- and the ledger is the point.
    late = _run_with_a_fault(tmpdir / "committed", after_commit=True)
    late_facts = late["outcome"].facts()
    assert not late["crash"], f"the machinery crashed: {late['crash']}"
    assert late_facts["outcome"] == "committed", (
        f"a COMMIT that returned is not recorded as one: {late_facts!r}"
    )
    assert late_facts["changed"] is True, (
        f"the rollback committed and the ledger says nothing changed: {late_facts!r}"
    )
    assert late_facts["backup_durable"] is True, f"{late_facts!r}"

    # (2) The fault IS the COMMIT.
    unsure = _run_with_a_fault(tmpdir / "unknown", after_commit=False)
    unsure_facts = unsure["outcome"].facts()
    assert not unsure["crash"], f"the machinery crashed: {unsure['crash']}"
    assert unsure_facts["outcome"] == "commit-unknown", (
        "a COMMIT that raised was resolved into a certainty the caller does "
        f"not have: {unsure_facts!r}"
    )
    assert unsure_facts["changed"] is None, (
        "an unknown commit was rendered as a boolean; whichever way it is "
        f"spelled it is a claim nobody is entitled to make: {unsure_facts!r}"
    )
    assert unsure["reason"] == "commit-unknown", f"{unsure['reason']!r}"


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
        backup = work_dir / "backup.db"
        outcome = library.RollbackOutcome()
        held = {"copy": None}

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

        # ARMED BEFORE THE PREPARER, for the reason recorded above.
        library._make_verified_backup = _backup_then_remember
        library._refuse_unexpected_surface = _fail_the_decision_that_follows_the_backup
        had_os = hasattr(library, "os")
        previous_os = getattr(library, "os", None)
        library.os = _RecordingOs()
        result = {"reason": "", "detail": "", "crash": ""}
        try:
            prepared = library.prepare_the_private_copy(store, work_dir=work_dir)
            held["copy"] = pathlib.Path(prepared.path)
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
            "store": store, "target": held["copy"],
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
    assert "ownership-lost" in _residue_errors(swapped_facts), (
        f"the residue does not say why it was left: {swapped_facts['residue']!r}"
    )

    # (3) SOMEBODY ELSE REMOVED IT FIRST.
    #     ENOENT answers only the OBSERVATION question. This run performed no
    #     unlink, so it has no agency to report, and it performed no directory
    #     flush, so if the competitor has not flushed either then a crash can
    #     bring the entry back. Claiming withdrawal here would be claiming an
    #     act this process never carried out.
    def _somebody_else_removes_it(path):
        path.unlink()

    raced_dir = tmpdir / "raced"
    raced_dir.mkdir()
    three = _run_until_the_ddl_fails(raced_dir, meddle=_somebody_else_removes_it)
    assert three["backed_up"], "the backup never landed in the third leg"
    assert not three["result"]["crash"], f"the boundary crashed: {three['result']['crash']}"
    raced = three["outcome"].facts()
    assert raced["backup_created"] is True, (
        f"the backup was written and the history denies it: {raced!r}"
    )
    assert raced["backup_unlinked_by_this_run"] is False, (
        "this run claims it removed a file it never called unlink on. ENOENT "
        f"is not evidence of agency: {raced!r}"
    )
    assert raced["backup_withdrawn"] is False, (
        f"a withdrawal is claimed that this run did not perform: {raced!r}"
    )
    assert raced["backup_absence_durable"] is None, (
        "the absence is reported as durable, and this run issued no directory "
        f"flush — a crash can bring the entry back: {raced!r}"
    )
    assert raced["backup_present"] is None, (
        "an observation somebody else made is reported as a certainty this "
        f"run established: {raced!r}"
    )
    assert "absent-not-by-this-run" in _residue_errors(raced), (
        f"the report does not say why presence is unknown: {raced!r}"
    )


def check_a_real_invocation_refuses_before_it_observes_the_source(
    tmpdir: pathlib.Path,
) -> None:
    """The documented order and the executed order are the same order.

    ``rollback_turn_fence`` says a real invocation refuses before it observes
    the source at all, and its docstring once said the opposite — that the
    pre-flight, the identity binding and the in-transaction decisions ran on
    the operator's own store first. They have not since the refusal moved ahead
    of every side effect. A claim about CONTROL FLOW is as checkable as a claim
    about a guarantee, and left unchecked it teaches the next reader the shape
    this contract replaced.

    WHY THE ORDER IS THE PROPERTY AND NOT AN OPTIMISATION
        Opening the source is not free. A ``mode=ro`` probe of a WAL-mode store
        creates ``-wal`` and ``-shm`` beside it, and on SQLite 3.53.1 they
        survive the close — measured, on this file, with ``state.db`` itself
        byte-identical. A refusal that added two files to the operator's
        directory has changed the thing it is about to say it left alone. So
        "refuses first" is what makes "nothing was changed" true, and it is
        asserted as EVENTS, not inferred from the bytes: bytes agreeing would
        also be consistent with an open that happened to leave no trace on this
        build.

    Each step the docstring says does not run is observed separately, so a
    failure names WHICH one started running rather than that something did.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    listing_before = sorted(entry.name for entry in tmpdir.iterdir())
    # CONTENT, as well as the listing and the events. An in-place rewrite
    # changes neither the file set nor any event this pin watches, so without
    # this the source could be modified byte-for-byte under a green pin.
    digest_before = _store_digest(store)

    observed = {"preflight": 0, "bound": 0, "prepared": 0, "connected": []}
    real_preflight = library.preflight_turn_fence_rollback
    real_bound = library.BoundTarget
    real_prepare = library.prepare_the_private_copy
    real_sqlite = library.sqlite3

    def _count_preflight(*args, **kwargs):
        observed["preflight"] += 1
        return real_preflight(*args, **kwargs)

    def _count_bound(*args, **kwargs):
        observed["bound"] += 1
        return real_bound(*args, **kwargs)

    def _count_prepare(*args, **kwargs):
        observed["prepared"] += 1
        return real_prepare(*args, **kwargs)

    class _WatchingSqlite:
        def __getattr__(self, name):
            return getattr(real_sqlite, name)

        def connect(self, target, *args, **kwargs):
            observed["connected"].append(str(target))
            return real_sqlite.connect(target, *args, **kwargs)

    library.preflight_turn_fence_rollback = _count_preflight
    library.BoundTarget = _count_bound
    library.prepare_the_private_copy = _count_prepare
    library.sqlite3 = _WatchingSqlite()
    try:
        run = _run_verb(store, tmpdir / "backup.db")
    finally:
        library.preflight_turn_fence_rollback = real_preflight
        library.BoundTarget = real_bound
        library.prepare_the_private_copy = real_prepare
        library.sqlite3 = real_sqlite

    assert not run.crash, f"the verb crashed: {run.crash}"
    payload = _payload(run)
    assert payload is not None, f"no machine-readable report: {run.stdout!r}"
    assert payload["refused"]["reason"] == "offline-authority-unknown", (
        f"the fixture did not reach the authority refusal: {payload['refused']!r}"
    )

    assert observed["preflight"] == 0, (
        "a real invocation ran the pre-flight. The docstring says it refuses "
        "before it observes the source, and the pre-flight copies the store"
    )
    assert observed["bound"] == 0, (
        "a real invocation opened the store with BoundTarget before refusing"
    )
    assert observed["prepared"] == 0, (
        "a real invocation made a private copy of the store for a run that "
        "cannot proceed — an unfenced duplicate written for nothing"
    )
    touched = [
        target for target in observed["connected"]
        if str(store) in target
    ]
    assert touched == [], (
        "a real invocation opened the operator's store with SQLite before "
        f"refusing: {touched!r}. On a WAL build that leaves -wal and -shm "
        "beside the artifact, so the refusal changes the directory it is "
        "about to say it left alone"
    )
    assert sorted(entry.name for entry in tmpdir.iterdir()) == listing_before, (
        "the refusal added or removed files in the store's directory: "
        f"{sorted(entry.name for entry in tmpdir.iterdir())}"
    )
    assert _store_digest(store) == digest_before, (
        "the refusal rewrote the source IN PLACE. The file set is unchanged "
        "and no watched event fired, which is exactly why the listing and the "
        f"counters cannot carry this on their own: {digest_before} -> "
        f"{_store_digest(store)}"
    )

    # AND THE HELP SAYS THE SAME THING. Two statements of one behaviour that
    # can disagree mean one of them is unpinned, and the one an operator reads
    # is the option help.
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="sessions_action")
    registered = module.add_fence_rollback_parser(subparsers)
    options = {
        action.dest: (action.help or "") for action in registered._actions
    }
    # The banned strings are the PROMISES, not the word. "does not roll back"
    # is the correction; a bare substring check would reject the fix and pass
    # the defect, which is a check that reads its own subject backwards.
    store_help = options.get("store", "").lower()
    assert "database to roll back" not in store_help, (
        f"--store still promises a rollback of the named store: {store_help!r}"
    )
    assert "does not roll it back" in store_help, (
        f"--store does not say what this build declines to do: {store_help!r}"
    )
    assert "does not open, copy or read it" in store_help, (
        f"--store still claims this build reads the named file: {store_help!r}"
    )
    backup_help = options.get("backup", "").lower()
    assert "still validated" not in backup_help, (
        f"--backup claims a validation that no longer runs: {backup_help!r}"
    )
    assert "not examined" in backup_help, (
        f"--backup does not say the destination is untouched: {backup_help!r}"
    )


def _suppress_directory_removal(library):
    """Make the REHEARSAL working directory un-removable, and nothing else.

    Suppressing every ``rmtree`` also breaks the backup's own staging cleanup,
    which correctly refuses the run before it can commit — so the injection
    would prevent the completion some of these legs are about, and the pin
    would be measuring a different failure than the one it names.
    """

    class _TheRehearsalDirectorySurvives:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def rmtree(self, path, *args, **kwargs):
            if pathlib.Path(path).name.startswith("hermes-fence-rehearsal-"):
                return None
            return self._real.rmtree(path, *args, **kwargs)

    real = library.shutil
    library.shutil = _TheRehearsalDirectorySurvives(real)
    return real


def check_two_notices_of_one_incident_are_one_record(
    tmpdir: pathlib.Path,
) -> None:
    """The ledger merges by WHICH INCIDENT, so a second look is not a second event.

    More than one code path can observe the same failure — the sweep that
    performed it, and a reporter that re-reads the ledger afterwards. Neither is
    wrong to look, and neither should be able to make one stuck directory
    become two by looking.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    outcome = library.RollbackOutcome()
    where = str(tmpdir / "stuck")
    outcome.note_residue(
        {"work_dir": where, "files": 3, "error": ""}, incident=f"work-dir:{where}"
    )
    outcome.note_residue(
        {"work_dir": where, "files": 3, "error": ""}, incident=f"work-dir:{where}"
    )

    records = outcome.facts()["residue"]
    assert len(records) == 1, (
        "one incident observed twice became two records. The second observer "
        f"is not a second stuck directory: {records!r}"
    )
    assert outcome.facts()["residue_present"] is True


def check_two_incidents_with_identical_values_stay_two_records(
    tmpdir: pathlib.Path,
) -> None:
    """Identical VALUES are not evidence of one incident. Never dedupe by value.

    The obvious repair for a double-count is to drop records that look the
    same, and it is the wrong one. Two genuinely distinct failures can produce
    byte-identical payloads — the same file count under the same reported error
    is a plausible coincidence, not proof they are one event — and collapsing
    them loses a fact the operator needs. That trade is strictly worse than the
    double-count it fixes: a spurious extra record wastes a look, a missing one
    leaves an unfenced duplicate on disk that nobody is told about.

    So identity comes from WHICH obligation over WHICH object, supplied by the
    caller that knows, and the payloads are never compared.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    outcome = library.RollbackOutcome()
    same_payload = {"work_dir": str(tmpdir / "somewhere"), "files": 3, "error": ""}
    outcome.note_residue(dict(same_payload), incident="work-dir:/a")
    outcome.note_residue(dict(same_payload), incident="backup:/a")

    records = outcome.facts()["residue"]
    assert len(records) == 2, (
        "two distinct incidents that happened to produce identical payloads "
        "were collapsed into one. A run can fail to remove its working copy "
        "AND a backup it tried to withdraw, and those are two places to go: "
        f"{records!r}"
    )
    incidents = sorted(str(record.get("incident")) for record in records)
    assert incidents == ["backup:/a", "work-dir:/a"], (
        f"the records do not say which incident each one is: {records!r}"
    )


def check_a_run_that_creates_nothing_reports_no_residue(
    tmpdir: pathlib.Path,
) -> None:
    """A directory existing is not an incident. Something unresolved is.

    The real path refuses at the authority BEFORE it observes the source, so it
    makes no copy and needs no working directory. One was being created up
    front anyway, and the cleanup then reported an incident about it — "a
    duplicate of every conversation in the store" asserted for a directory with
    zero files in it, and that manufactured reason overwrote the authority
    refusal that had actually decided the run.

    Two failures in one line: a claim with no evidence behind it, and a late
    fact erasing an earlier one. Both are visible from outside, so both are
    asserted here.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    digest_before = _store_digest(store)
    listing_before = sorted(entry.name for entry in tmpdir.iterdir())

    real_shutil = _suppress_directory_removal(library)
    try:
        run = _run_verb(store, tmpdir / "backup.db")
    finally:
        library.shutil = real_shutil

    assert not run.crash, f"the verb crashed: {run.crash}"
    payload = _payload(run)
    assert payload is not None, f"no report: {run.stdout!r}"

    assert payload["refused"]["reason"] == "offline-authority-unknown", (
        "a cleanup failure relabelled the run. The authority decided this "
        f"one's fate and residue may not take the reason from it: "
        f"{payload['refused']!r}"
    )
    assert not (payload.get("residue") or []), (
        "the run reports something left behind, and it created nothing — it "
        f"refused before it observed the source: {payload.get('residue')!r}"
    )
    assert "duplicate" not in run.stdout + run.stderr, (
        f"a duplicate of the store is claimed and none was ever made:\n{run.stderr}"
    )
    assert _store_digest(store) == digest_before
    assert sorted(entry.name for entry in tmpdir.iterdir()) == listing_before, (
        f"the refusal changed the store's directory: "
        f"{sorted(entry.name for entry in tmpdir.iterdir())}"
    )


def _drive_the_boundary_with_unlink_failing(library, store, work_dir, sabotage):
    """Enter the boundary with ``os.unlink`` sabotaged for chosen pathnames.

    *sabotage* is called with each pathname before the real unlink and may
    raise, or may replace what is there. The backup destination sits OUTSIDE
    the private working directory so a sidecar left behind is a real artifact
    in the operator's chosen location, not something the sweep tidies away.
    """
    backup = work_dir.parent / "backup.db"
    outcome = library.RollbackOutcome()
    prepared = {"copy": None}

    class _SabotagedOs:
        def __getattr__(self, name):
            return getattr(os, name)

        def lstat(self, path, *args, **kwargs):
            sabotage(pathlib.Path(path), "lstat")
            return os.lstat(path, *args, **kwargs)

        def unlink(self, path, *args, **kwargs):
            sabotage(pathlib.Path(path), "unlink")
            return os.unlink(path, *args, **kwargs)

    had_os = hasattr(library, "os")
    previous = getattr(library, "os", None)
    # ARMED BEFORE THE PREPARER, for the reason recorded above.
    library.os = _SabotagedOs()
    result = {"returned": None, "reason": "", "detail": "", "crash": ""}
    try:
        handle = library.prepare_the_private_copy(store, work_dir=work_dir)
        prepared["copy"] = pathlib.Path(handle.path)
        result["returned"] = library._commit_the_rollback(
            handle, backup, sorted(hermes_state_common.TURN_FENCE_TRIGGERS),
            report_as=store, outcome=outcome,
        )
    except library.TurnFenceRollbackRefused as exc:
        result["reason"] = getattr(exc, "reason", "refused")
        result["detail"] = str(exc)
    except BaseException as exc:  # noqa: BLE001 - carried, not swallowed
        result["crash"] = f"{type(exc).__name__}: {exc}"
    finally:
        if had_os:
            library.os = previous
        else:
            del library.os
    return {"backup": backup, "outcome": outcome, "result": result,
            "copy": prepared["copy"]}


def check_a_sidecar_that_cannot_be_released_is_not_a_successful_backup(
    tmpdir: pathlib.Path,
) -> None:
    """Ownership is held until the release RESULT is known, not until it starts.

    The destination family is reserved with exclusive creates, and the sidecar
    reservations are handed back at the end so the backup is one file. Handing
    back is an operation that can fail — a pinned file, a permission change, a
    filesystem that says no — and the release path dropped the handle and the
    recorded identity BEFORE attempting it, then swallowed every error. By the
    time the unlink failed there was nothing left to report with and nothing to
    retry from, so the run returned a verified, durable backup while an
    unreleased reservation sat beside it on disk, unmentioned.

    THIS IS THE SHAPE ALREADY FIXED ONCE, ONE SEAM OVER. Presence was
    snapshotted before the sweep and re-emitted as current; here ownership is
    discarded before the release and the outcome is emitted as if it had
    succeeded. Both are *state that governs an operation released before that
    operation's result is known*, and the repair is the same: keep the thing
    that decides until the decision is in.

    An unreleased ``-wal`` beside a database is not litter. The next reader
    picks it up as that database's write-ahead log, which is the exact hazard
    the family reservation exists to prevent — so this cannot be a success.

    The durability claims are re-checked THROUGH THIS PATH, not merely where
    they were written: a release that failed halfway leaves the parent
    directory in a state the earlier fsync no longer describes.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    triggers_before = _installed_triggers(store)
    work_dir = tmpdir / "work"
    work_dir.mkdir()

    def _pin_the_wal_sidecar(path, op):
        if op == "unlink" and path.name.endswith("backup.db-wal"):
            raise PermissionError("pinned sidecar")

    run = _drive_the_boundary_with_unlink_failing(
        library, store, work_dir, _pin_the_wal_sidecar
    )
    facts = run["outcome"].facts()
    orphan = run["backup"].with_name(run["backup"].name + "-wal")

    assert not run["result"]["crash"], f"the boundary crashed: {run['result']['crash']}"
    assert orphan.exists(), (
        "the fixture did not actually leave a sidecar behind, so this pin "
        f"measures nothing: {sorted(_family_beside(run['backup']))}"
    )
    assert run["result"]["returned"] is None, (
        "a reservation could not be released and the boundary returned a "
        f"backup report anyway: {run['result']['returned']!r}. There is a "
        "stale -wal beside that database and the next reader will use it"
    )
    assert facts["backup_durable"] is False, (
        f"a durable backup is claimed for a destination family that was never "
        f"cleanly established: {facts!r}"
    )
    assert facts["backup_verified"] is False, (
        f"a verified backup is claimed: {facts!r}"
    )
    assert facts["outcome"] not in ("committed", "commit-unknown"), (
        f"the rollback committed on a run whose backup never completed: {facts!r}"
    )
    assert facts["changed"] is False, (
        f"the run reports it changed the store: {facts!r}"
    )
    assert facts["residue_present"] is True, (
        f"the unreleased reservation is on disk and unreported: {facts!r}"
    )
    assert any(
        str(orphan) == record.get("path") for record in facts["residue"]
    ), (
        f"the residue does not name the file that was left: {facts['residue']!r}"
    )
    assert _installed_triggers(run["copy"]) == triggers_before, (
        "triggers were dropped on a run whose backup did not complete"
    )
    assert _installed_triggers(store) == triggers_before


def check_a_foreign_file_at_a_reserved_sidecar_is_never_deleted(
    tmpdir: pathlib.Path,
) -> None:
    """A reserved name that now resolves elsewhere is not ours to remove.

    Between reserving a sidecar and handing it back, the pathname can come to
    mean a different file. Releasing by NAME then deletes something this run
    never created — the same rule the published backup already follows, and it
    binds harder here because nothing about the sidecar's contents would ever
    look wrong afterwards.

    So identity decides, the foreign file survives, and the run says it could
    not finish rather than tidying the question away.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    work_dir = tmpdir / "work"
    work_dir.mkdir()
    stranger = b"a different file that arrived at the reserved sidecar"

    def _swap_the_sidecar_for_a_strangers_file(path, op):
        # BEFORE the identity check, which is the window that matters: the
        # reservation is handed back by NAME, and the name can come to mean a
        # different file between reserving it and releasing it.
        if op == "lstat" and path.name.endswith("backup.db-wal") and path.exists():
            path.unlink()
            path.write_bytes(stranger)

    run = _drive_the_boundary_with_unlink_failing(
        library, store, work_dir, _swap_the_sidecar_for_a_strangers_file
    )
    facts = run["outcome"].facts()
    orphan = run["backup"].with_name(run["backup"].name + "-wal")

    assert not run["result"]["crash"], f"the boundary crashed: {run['result']['crash']}"
    assert orphan.exists() and orphan.read_bytes() == stranger, (
        "the run deleted a file it did not create at a name it had merely "
        f"reserved: {orphan.exists()!r}"
    )
    assert run["result"]["returned"] is None, (
        f"the run reported a completed backup: {run['result']['returned']!r}"
    )
    assert "ownership-lost" in _residue_errors(facts), (
        f"the report does not say the name stopped being ours: {facts['residue']!r}"
    )


def check_a_sidecar_that_vanished_is_not_claimed_as_our_removal(
    tmpdir: pathlib.Path,
) -> None:
    """Absence answers "is it there". It does not answer "did we remove it".

    Same rule as the withdrawal edge, at the other end of the same object: a
    reservation that is already gone when the release looks was removed by
    somebody, and this run has no basis for saying it was the one. The
    obligation is satisfied — nothing is at that name — so there is no residue
    to report, and equally no agency to claim.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    work_dir = tmpdir / "work"
    work_dir.mkdir()

    def _somebody_else_removed_it_first(path, op):
        if op == "lstat" and path.name.endswith("backup.db-wal") and path.exists():
            os.unlink(path)

    run = _drive_the_boundary_with_unlink_failing(
        library, store, work_dir, _somebody_else_removed_it_first
    )
    facts = run["outcome"].facts()
    orphan = run["backup"].with_name(run["backup"].name + "-wal")

    assert not run["result"]["crash"], f"the boundary crashed: {run['result']['crash']}"
    assert not orphan.exists(), "the fixture left the sidecar in place"
    assert run["result"]["returned"] is not None, (
        "the reservation is gone, which is the state the release wanted, and "
        f"the run refused anyway: {run['result']!r}"
    )
    assert facts["residue_present"] is False, (
        f"nothing is at that name and residue is reported: {facts['residue']!r}"
    )


def check_every_surviving_destination_member_is_reported_exactly_once(
    tmpdir: pathlib.Path,
) -> None:
    """A cleanup that RETURNS its failures, and a caller that reads them.

    The release path was taught to hold ownership until the result was known,
    and it does: it detects the member it could not remove and returns it. The
    caller then called it for effect and threw the list away, so a surviving
    ``backup.db`` sat on disk while the layer below had already worked out that
    it was there. The defect did not survive where it was fixed — it reappeared
    one level up, where the fix was invisible from outside.

    That is the second time in this slice a defect has moved up a level after
    being closed down one, so the rule is stated rather than filed: after
    fixing something, check its CALLERS. Every level that consumes the fixed
    thing's result. A return value describing a failure is a fact, and a fact
    nobody reads is not a fact.

    THE CARDINALITY RULE, ACROSS LAYERS THIS TIME. Both the sidecar release and
    the cleanup pass can see the same physical file, so the incident key is the
    MEMBER — not the layer that noticed it. One surviving destination is one
    record however many passes observe it, and two surviving members are two.

    And the refusal that decided the run stays primary: residue is additive.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    triggers_before = _installed_triggers(store)
    work_dir = tmpdir / "work"
    work_dir.mkdir()

    def _pin_both_the_sidecar_and_the_backup(path, op):
        if op == "unlink" and path.name in ("backup.db-wal", "backup.db"):
            raise PermissionError("pinned")

    run = _drive_the_boundary_with_unlink_failing(
        library, store, work_dir, _pin_both_the_sidecar_and_the_backup
    )
    facts = run["outcome"].facts()
    backup = run["backup"]
    orphan = backup.with_name(backup.name + "-wal")

    assert not run["result"]["crash"], f"the boundary crashed: {run['result']['crash']}"
    assert backup.exists() and orphan.exists(), (
        "the fixture did not leave both members behind, so this pin measures "
        f"nothing: {sorted(_family_beside(backup))}"
    )
    assert run["result"]["returned"] is None, (
        f"the run reported a backup: {run['result']['returned']!r}"
    )
    assert run["result"]["reason"] == "backup-destination-residue", (
        "residue took the reason from whatever decided the run's fate: "
        f"{run['result']!r}"
    )

    reported = sorted(record["path"] for record in facts["residue"])
    assert reported == sorted([str(backup), str(orphan)]), (
        "the surviving destination members are not each reported exactly "
        f"once: {facts['residue']!r}. Two files remain on disk and the "
        "operator is told about a different set"
    )
    assert len(facts["residue"]) == len(set(reported)), (
        f"one physical member produced more than one record: {facts['residue']!r}"
    )
    assert facts["backup_created"] is False and facts["backup_verified"] is False, (
        f"a valid backup is claimed for one that did not survive: {facts!r}"
    )
    assert facts["backup_durable"] is False, (
        f"durability is claimed for a destination never cleanly established: {facts!r}"
    )
    assert facts["changed"] is False
    assert _installed_triggers(run["copy"]) == triggers_before
    assert _installed_triggers(store) == triggers_before


def check_a_cleanup_failure_never_replaces_the_refusal_that_decided_the_run(
    tmpdir: pathlib.Path,
) -> None:
    """A ``finally`` that raises DESTROYS the exception already in flight.

    Python discards the live exception when a ``finally`` raises, so the
    refusal that actually decided the run never reaches the caller and a
    cleanup problem takes its place. That is a late failure retracting an
    established fact — the same obligation as the ledger's — delivered by
    control flow instead of by an assignment, which is why it survived the
    ledger being made monotonic. It is invisible unless two things go wrong at
    once, and this pin is the case where they do.

    An explicit ``raise`` in a ``finally`` is easy to see. A ``close()`` or an
    ``execute("ROLLBACK")`` that happens to fail is the identical trap in
    ordinary clothes, so cleanup here records and lets the original propagate,
    and only raises when nothing else is propagating.

    AND THE PROSE IS RENDERED FROM THE OUTCOME. "The backup landed" is a claim
    about the final verified-and-durable fact; on this path that fact is false
    three ways, and the sentence was being written by the branch that happened
    to be printing rather than read from the thing that decides it.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    triggers_before = _installed_triggers(store)
    work_dir = tmpdir / "work"
    work_dir.mkdir()

    class _StagingSurvives:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def rmtree(self, path, *args, **kwargs):
            if pathlib.Path(path).name.startswith(".fence-rollback-backup-"):
                return None
            return self._real.rmtree(path, *args, **kwargs)

    def _pin_the_wal_sidecar(path, op):
        if op == "unlink" and path.name.endswith("backup.db-wal"):
            raise PermissionError("pinned sidecar")

    real_shutil = library.shutil
    library.shutil = _StagingSurvives(real_shutil)
    try:
        run = _drive_the_boundary_with_unlink_failing(
            library, store, work_dir, _pin_the_wal_sidecar
        )
    finally:
        library.shutil = real_shutil

    facts = run["outcome"].facts()
    assert not run["result"]["crash"], f"the boundary crashed: {run['result']['crash']}"
    assert run["result"]["returned"] is None, (
        f"the run reported a backup: {run['result']['returned']!r}"
    )

    assert run["result"]["reason"] == "backup-destination-residue", (
        "a cleanup problem in a finally replaced the refusal that decided this "
        f"run: {run['result']['reason']!r}. The staging sweep is additive — it "
        "never becomes the reason the run failed"
    )
    assert "landed" not in run["result"]["detail"], (
        "the message says the backup landed, and the outcome says it was "
        f"never created, verified or made durable: {run['result']['detail']!r}"
    )

    incidents = sorted(record["incident"] for record in facts["residue"])
    assert len(incidents) == len(set(incidents)) and len(incidents) == 2, (
        "two physical incidents — a pinned sidecar and an un-swept staging "
        f"directory — are not reported exactly once each: {facts['residue']!r}"
    )
    assert any(i.startswith("staging:") for i in incidents), (
        f"the staging directory is not reported: {incidents!r}"
    )
    assert any(i.startswith("destination:") for i in incidents), (
        f"the pinned sidecar is not reported: {incidents!r}"
    )

    assert facts["backup_created"] is False, f"a backup is claimed: {facts!r}"
    assert facts["backup_verified"] is False, f"a valid backup is claimed: {facts!r}"
    assert facts["backup_durable"] is False, f"durability is claimed: {facts!r}"
    assert facts["changed"] is False
    assert _installed_triggers(run["copy"]) == triggers_before
    assert _installed_triggers(store) == triggers_before


def _stamp(path: pathlib.Path, version: int) -> None:
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.execute(f"PRAGMA user_version={int(version)}")
    finally:
        conn.close()


def _identify(path: pathlib.Path) -> dict:
    """Which store this is, whether it still reads, and what fence it carries."""
    try:
        conn = sqlite3.connect(str(path))
    except sqlite3.DatabaseError as exc:
        return {"readable": False, "why": f"{type(exc).__name__}: {exc}"}
    try:
        return {
            "readable": True,
            "user_version": int(conn.execute("PRAGMA user_version").fetchone()[0]),
            "triggers": len(
                conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger' "
                    "AND name LIKE 'hermes_turn_fence_%'"
                ).fetchall()
            ),
            "integrity": str(conn.execute("PRAGMA integrity_check").fetchone()[0]),
        }
    except sqlite3.DatabaseError as exc:
        return {"readable": False, "why": f"{type(exc).__name__}: {exc}"}
    finally:
        conn.close()


def check_a_target_swapped_and_restored_around_any_open_cannot_be_reached(
    tmpdir: pathlib.Path,
) -> None:
    """A→B→A is invisible to any number of point checks. Hold the object.

    Rename the prepared copy aside, put a DIFFERENT valid fenced store at its
    pathname, let an open happen, then put the original back. Every ``stat``
    before and after sees A, because the substitution existed only for the
    instant of the resolution in between. Measured: a committed rollback
    against a store nothing prepared, with the substituted one left corrupted,
    reported as a successful rehearsal.

    THE PROBE STARTS BEFORE THE CODE'S OWN SETUP, WHICH IS THE POINT
        An earlier version of this pin installed the swap only after
        ``prepare_the_private_copy`` returned. That proved the boundary does
        not reopen the pathname — true, and the thing that had just been fixed
        — while saying nothing about the preparer's own sequence, which still
        had copy → lstat → connect: three resolutions and two intervals. The
        defect had moved one function upstream and the suite was green over it.

        An attacker does not have to attack where the defence is. So this arms
        the swap FIRST and lets it fire at any resolution of the copy's
        pathname, wherever the code chooses to make one.

    A THIRD CHECK IS NOT THE FIX, BY CONSTRUCTION
        Observations do not remove intervals — the interval is where the swap
        lives. The mutable object is now deserialized from bytes this run holds
        and has no pathname at all, so the assertion is a COUNT: zero.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    _stamp(store, 111)
    a_before = _identify(store)

    substitute = tmpdir / "substitute.db"
    _fenced_store(substitute, leave_lease_live=False)
    _stamp(substitute, 222)
    b_before = _identify(substitute)
    assert b_before["user_version"] == 222 and b_before["triggers"] == 24, (
        f"the substitute fixture is not a distinguishable fenced store: {b_before!r}"
    )

    work_dir = tmpdir / "work"
    work_dir.mkdir()
    backup = work_dir / "backup.db"
    resolutions = []
    real_sqlite = library.sqlite3

    class _SwapAroundEveryResolution:
        """A→B→A for the exact instant of ANY resolution of the copy's name."""

        def __getattr__(self, name):
            return getattr(real_sqlite, name)

        def connect(self, target, *args, **kwargs):
            spelled = str(target)
            if not spelled.endswith("preflight.db"):
                return real_sqlite.connect(target, *args, **kwargs)
            resolutions.append(spelled)
            here = pathlib.Path(spelled)
            aside = here.parent / "aside.db"
            os.rename(here, aside)
            os.rename(substitute, here)
            try:
                return real_sqlite.connect(target, *args, **kwargs)
            finally:
                os.rename(here, substitute)
                os.rename(aside, here)

    outcome = library.RollbackOutcome()
    # ARMED BEFORE THE PREPARER RUNS. The preparer is inside the attack window,
    # not outside it.
    library.sqlite3 = _SwapAroundEveryResolution()
    result = {"returned": None, "reason": "", "detail": "", "crash": "", "copy": None}
    try:
        prepared = library.prepare_the_private_copy(store, work_dir=work_dir)
        result["copy"] = pathlib.Path(prepared.path)
        result["returned"] = library._commit_the_rollback(
            prepared, backup, sorted(hermes_state_common.TURN_FENCE_TRIGGERS),
            report_as=store, outcome=outcome,
        )
        result["remaining"] = len(
            prepared.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' "
                "AND name LIKE 'hermes_turn_fence_%'"
            ).fetchall()
        )
    except library.TurnFenceRollbackRefused as exc:
        result["reason"] = getattr(exc, "reason", "refused")
        result["detail"] = str(exc)
    except BaseException as exc:  # noqa: BLE001 - carried, not swallowed
        result["crash"] = f"{type(exc).__name__}: {exc}"
    finally:
        library.sqlite3 = real_sqlite

    assert not result["crash"], f"the run crashed: {result['crash']}"

    # NEITHER STORE IS EVER THIS RUN'S SUBJECT, on either branch.
    assert _identify(substitute) == b_before, (
        f"the substituted store was modified: {b_before!r} -> "
        f"{_identify(substitute)!r}"
    )
    assert _identify(store) == a_before, (
        f"the source store was modified: {a_before!r} -> {_identify(store)!r}"
    )

    # THE WINDOW ITSELF. Zero is the property; anything else is an interval a
    # rename can live in, wherever in the code it happens to be.
    assert resolutions == [], (
        "the copy's pathname was resolved by SQLite at least once "
        f"({len(resolutions)}x), so an A->B->A lands inside that interval and "
        "is invisible to a check on either side. The object has to be held, "
        "not looked at again"
    )

    if result["returned"] is not None:
        assert outcome.facts()["outcome"] == "committed", f"{outcome.facts()!r}"
        assert result["remaining"] == 0, (
            f"the rollback did not run against the prepared image: {result!r}"
        )
        landed = _identify(backup)
        assert landed["readable"] and landed["user_version"] == 111, (
            f"the backup describes a store nothing prepared: {landed!r}"
        )
        assert landed["triggers"] == 24, (
            f"the backup was not taken before the drops: {landed!r}"
        )
    else:
        assert outcome.facts()["outcome"] not in ("committed", "commit-unknown")
        assert outcome.facts()["changed"] is False
        assert not backup.exists(), "a refused run left a backup behind"


def check_an_artifact_whose_image_is_incomplete_is_refused(
    tmpdir: pathlib.Path,
) -> None:
    """MAINTENANCE ONLY. A main image is not the database when a sidecar holds rows.

    Driven at the preparer, not through the verb: the boundary's own sidecar
    disqualifier reaches the same verdict for the same artifact, so asserting
    it through the CLI passes whether or not the preparer ever checked — a
    redundant guarantee reporting coverage it does not have. What the preparer
    owns is refusing before it deserializes an image whose committed rows live
    in an uncheckpointed ``-wal``: such an image opens cleanly, passes
    ``integrity_check`` and is quietly short of data.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    marker = "committed-but-not-checkpointed"
    _commit_a_marker_that_lives_only_in_the_wal(store, marker)
    wal = store.with_name(store.name + "-wal")
    assert wal.is_file() and marker.encode() in wal.read_bytes(), (
        f"the fixture left no uncheckpointed -wal: {sorted(_family_beside(store))}"
    )
    assert marker.encode() not in store.read_bytes(), (
        "the row reached the MAIN file, so the image is complete"
    )
    work_dir = tmpdir / "work"
    work_dir.mkdir()

    prepared = None
    reason = ""
    try:
        prepared = library.prepare_the_private_copy(store, work_dir=work_dir)
    except library.TurnFenceRollbackRefused as exc:
        reason = exc.reason

    assert prepared is None, (
        "the preparer built a working object from an image that is missing "
        "committed rows; every later claim is about a different database"
    )
    assert reason == "target-not-quiesced", f"{reason!r}"
    assert not any(work_dir.iterdir()), (
        f"a private copy was created anyway: {sorted(p.name for p in work_dir.iterdir())}"
    )


def check_the_backup_describes_the_prepared_image_not_the_source_path(
    tmpdir: pathlib.Path,
) -> None:
    """The backup comes from the object being rolled back, not from a pathname.

    The engine writes it from the connection the DDL runs on, so it describes
    the artifact this run prepared. A copy taken from the source PATH describes
    whatever is at that path when the copy happens — a different object, and on
    a live-ish store a moving one.

    The two are told apart by changing the source after preparation: the backup
    must show the prepared state and not the later one. That is a property of
    both permitted implementations — ``VACUUM INTO`` and the online backup API
    — and false of any file copy, which is the distinction that matters rather
    than which call was used.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library
    from hermes_state import SessionDB

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    rows_before = _canonical_rows(store)
    triggers_before = _installed_triggers(store)

    work_dir = tmpdir / "work"
    work_dir.mkdir()
    added = "written-after-the-image-was-taken"
    real_backup = library._make_verified_backup
    moved = {"done": False}

    def _change_the_source_then_back_up(*args, **kwargs):
        if not moved["done"]:
            moved["done"] = True
            db = SessionDB(db_path=store)
            try:
                grant = db.try_acquire_session_turn_lease(
                    "keep", _holder("later"), ttl_seconds=600
                )
                db.append_message(
                    session_id="keep", role="user", content=added,
                    turn_lease_holder=grant,
                )
                db.release_session_turn_lease("keep", grant)
            finally:
                db.close()
        return real_backup(*args, **kwargs)

    library._make_verified_backup = _change_the_source_then_back_up
    try:
        outcome = _rehearse(library, store, tmpdir / "backup.db", work_dir)
    finally:
        library._make_verified_backup = real_backup

    assert moved["done"], "the source was never changed, so this pin compares nothing"
    assert not outcome["crash"], f"the rehearsal crashed: {outcome['crash']}"
    backup = work_dir / "rehearsal-backup.db"
    assert backup.is_file(), (
        f"the rehearsal produced no backup: {outcome['reason']!r} {outcome['detail']!r}"
    )
    assert _family_beside(backup) == {backup.name}, (
        f"the backup arrived as a family of files: {sorted(_family_beside(backup))}"
    )

    conn = sqlite3.connect(str(backup))
    try:
        content = sorted(str(r[0]) for r in conn.execute("SELECT content FROM messages"))
    finally:
        conn.close()
    assert added not in content, (
        "the backup carries a row written to the source AFTER the image was "
        "taken, so it was copied from the pathname rather than written from "
        f"the object being rolled back: {content!r}"
    )
    assert "irreplaceable" in content, (
        f"the backup lost the rows the prepared image carried: {content!r}"
    )
    assert _installed_triggers(backup) == triggers_before, (
        "the backup was taken after the drops — it restores an unfenced store: "
        f"{_installed_triggers(backup)}"
    )
    assert _pragma(backup, "integrity_check") == "ok"

    restored = tmpdir / "restored.db"
    shutil.copyfile(backup, restored)
    assert _canonical_rows(restored) == rows_before, (
        "the backup does not restore the state the artifact was prepared from"
    )


def check_a_late_sidecar_never_yields_committed_or_backup_facts(
    tmpdir: pathlib.Path,
) -> None:
    """Check-then-read cannot vouch for an image, so nothing is derived from one.

    The source-sidecar guard decided ``beside == []`` and the main image was
    read afterwards. Those are two operations, and a writer that commits into a
    ``-wal`` in between leaves the check's answer true and the image short of
    committed rows. Injected at exactly that point — inside
    ``BoundTarget.open_for_reading``, after the check and before the bytes —
    the run reported ``outcome=committed``, ``changed=true``, a verified and
    durable backup and a full ``would_drop``, every one of them derived from a
    database missing the row. The marker was in ``state.db-wal`` and absent
    from the main file. Both runtimes.

    A THIRD EXISTENCE CHECK IS NOT THE ANSWER. The interval is what the writer
    uses, and observations do not remove intervals; that move failed three
    times in this slice at three different seams. This is not a guard needing
    strengthening — it is what the input is. A private copy assembled by
    check-then-read cannot be proven coherent against a live-capable source,
    so the rehearsal was never a weaker version of the real operation. It was
    an operation on an artifact nobody can vouch for, and every fact
    downstream inherited that.

    So the assertion is that the injector NEVER FIRES: nothing reads an image,
    because nothing derives anything. A count of zero, in the same shape as the
    A→B→A pin, for the same reason — the property has to hold by construction
    rather than by looking more often.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    digest_before = _store_digest(store)
    listing_before = sorted(entry.name for entry in tmpdir.iterdir())
    rows_before = _canonical_rows(store)

    marker = "committed-after-the-sidecar-check"
    fired = {"reads": 0}
    real_bound = library.BoundTarget

    class _AWriterCommitsAfterTheCheck(real_bound):
        """Commit into a ``-wal`` between the sidecar check and the image read."""

        def open_for_reading(self):
            fired["reads"] += 1
            _commit_a_marker_that_lives_only_in_the_wal(store, marker)
            return super().open_for_reading()

    library.BoundTarget = _AWriterCommitsAfterTheCheck
    try:
        run = _run_verb(store, tmpdir / "backup.db", dry_run=True)
    finally:
        library.BoundTarget = real_bound

    assert not run.crash, f"the verb crashed: {run.crash}"
    payload = _payload(run)
    assert payload is not None, f"no machine-readable report: {run.stdout!r}"

    assert fired["reads"] == 0, (
        "the run read a source image, so there is an interval between deciding "
        "the source is quiesced and reading it — and a writer that commits in "
        f"that interval is invisible to both ends of it ({fired['reads']} read(s))"
    )

    # ABSENT, not false. Those facts were never produced, and a `false` would
    # claim the step was reached and did not deliver.
    published = sorted(_every_key(payload))
    for fact in ("rehearsal", "backup_created", "backup_verified",
                 "backup_durable", "would_drop", "dropped_triggers"):
        assert not any(k.rsplit(".", 1)[-1] == fact for k in published), (
            f"{fact} was published for work that never ran: {published}"
        )
    assert payload["outcome"] == "not-started", (
        f"the run derived a rollback outcome from that image: {payload!r}"
    )
    assert payload["changed"] is False
    assert payload["refused"]["reason"] == "offline-authority-unknown", (
        f"the run refused for another reason: {payload['refused']!r}"
    )

    # THE SOURCE IS AS IT WAS FOUND — bytes and file set, not just rows.
    assert _store_digest(store) == digest_before, (
        f"the refusal rewrote the source: {digest_before} -> {_store_digest(store)}"
    )
    assert sorted(entry.name for entry in tmpdir.iterdir()) == listing_before, (
        "the refusal added or removed files beside the source: "
        f"{sorted(entry.name for entry in tmpdir.iterdir())}"
    )
    assert _canonical_rows(store) == rows_before
    assert not (tmpdir / "backup.db").exists()


#: Everything a refusal that derived nothing is entitled to say. Asserted as an
#: EXACT set, not as a list of absences: a pin that only checks
#: ``"rehearsal" not in payload`` goes on passing when the block is re-added
#: under another name, which is the failure mode of every absence assertion.
_REFUSAL_KEYS_WHEN_NOTHING_RAN = {
    "verb", "ok", "dry_run", "changed", "outcome", "store", "generation",
    "refused",
}


def _every_key(node, trail=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield f"{trail}.{key}"
            yield from _every_key(value, f"{trail}.{key}")
    elif isinstance(node, list):
        for item in node:
            yield from _every_key(item, trail)


def check_a_refusal_publishes_no_fact_it_never_produced(
    tmpdir: pathlib.Path,
) -> None:
    """Absent and ``false`` are different statements, and only one is true here.

    ``backup_created: false`` says the backup step was REACHED and produced
    nothing. When the run refuses before deriving anything, no backup step ran
    at all — so ``false`` is a claim about an event that never had a chance to
    occur, and a ``preflight`` block of false checks claims each one was
    performed and failed. A field that is PRESENT asserts the question was
    asked.

    So the report is required to carry exactly the fields it can stand behind
    and no others. Asserted as an EXACT KEY SET rather than as a handful of
    ``not in`` checks: absence assertions pass happily when the thing comes
    back under a different name, and a key set catches that because the new
    name is also not in the set.

    Both paths, because they are the same refusal — there is no rehearsal to
    distinguish them any more.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    forbidden = {
        "rehearsal", "would_drop", "installed_triggers", "dropped_triggers",
        "preflight", "backup_created", "backup_verified", "backup_durable",
        "backup_present", "backup_withdrawn", "residue",
    }
    for dry_run in (True, False):
        where = tmpdir / ("dry" if dry_run else "real")
        where.mkdir()
        store = where / "state.db"
        _fenced_store(store, leave_lease_live=False)
        digest_before = _store_digest(store)
        listing_before = sorted(entry.name for entry in where.iterdir())
        rows_before = _canonical_rows(store)
        backup = where / "backup.db"

        run = _run_verb(store, backup, dry_run=dry_run)
        assert not run.crash, f"the verb crashed: {run.crash}"
        payload = _payload(run)
        assert payload is not None, f"no machine-readable report: {run.stdout!r}"

        assert set(payload) == _REFUSAL_KEYS_WHEN_NOTHING_RAN, (
            f"the {'dry' if dry_run else 'real'} refusal publishes a different "
            f"set of fields than it produced. extra="
            f"{sorted(set(payload) - _REFUSAL_KEYS_WHEN_NOTHING_RAN)} missing="
            f"{sorted(_REFUSAL_KEYS_WHEN_NOTHING_RAN - set(payload))}"
        )
        leaked = sorted(
            key for key in _every_key(payload)
            if key.rsplit(".", 1)[-1] in forbidden
        )
        assert leaked == [], (
            "facts about work that never ran appear somewhere in the document: "
            f"{leaked}"
        )
        assert payload["ok"] is False
        assert payload["changed"] is False
        assert payload["outcome"] == "not-started"
        assert payload["refused"]["reason"] == "offline-authority-unknown"
        assert run.rc not in (0, None)

        assert _store_digest(store) == digest_before, (
            f"the refusal rewrote the source: {digest_before} -> "
            f"{_store_digest(store)}"
        )
        assert sorted(entry.name for entry in where.iterdir()) == listing_before, (
            f"the refusal changed the source's directory: "
            f"{sorted(entry.name for entry in where.iterdir())}"
        )
        assert _canonical_rows(store) == rows_before
        assert not backup.exists()


def check_an_unused_work_dir_cannot_change_the_verdict(
    tmpdir: pathlib.Path,
) -> None:
    """A dead input may not decide what the operator is told.

    ``--work-dir`` fed the working copy, and there is no working copy. It was
    still ``stat``-ed, and a nonexistent path produced ``rehearsal-unwritable``
    where the real path gives the authority refusal — so an input that does
    nothing changed the verdict, and broke the dry run's own
    same-refusal-as-the-real-run contract at the same time.

    The general form, which is why this is pinned rather than just fixed: when
    a boundary removes a capability, every input that fed that capability must
    stop influencing the outcome. Otherwise dead inputs quietly become live
    verdict-changers. This is what stops one being re-attached later.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    digest_before = _store_digest(store)
    listing_before = sorted(entry.name for entry in tmpdir.iterdir())

    absent = tmpdir / "no-such-directory"
    assert not absent.exists()

    baseline = _payload(_run_verb(store, tmpdir / "backup.db", dry_run=True))
    with_flag = _payload(
        _run_verb(store, tmpdir / "backup.db", dry_run=True, work_dir=absent)
    )
    real = _payload(_run_verb(store, tmpdir / "backup.db", dry_run=False))
    for name, payload in (("dry", baseline), ("dry+work-dir", with_flag), ("real", real)):
        assert payload is not None, f"{name}: no machine-readable report"
        assert payload["refused"]["reason"] == "offline-authority-unknown", (
            f"{name} refused for a different reason than the others: "
            f"{payload['refused']!r}"
        )
    assert with_flag["refused"]["reason"] == real["refused"]["reason"], (
        "a nonexistent --work-dir made the dry run refuse for a reason the "
        f"real run does not give: {with_flag['refused']!r} vs {real['refused']!r}"
    )

    assert not absent.exists(), "the unused --work-dir was created"
    assert _store_digest(store) == digest_before
    assert sorted(entry.name for entry in tmpdir.iterdir()) == listing_before, (
        "a work artifact was left beside the source: "
        f"{sorted(entry.name for entry in tmpdir.iterdir())}"
    )


def check_a_refusal_says_nothing_about_a_destination_it_never_examined(
    tmpdir: pathlib.Path,
) -> None:
    """An occupied destination is where ``present: false`` is provably wrong.

    The refusal used to publish ``backup: {created: false, present: false}`` on
    every path. That reads as a cautious default and is not one: the boundary
    never examines the destination, so on an operator who names an existing
    file the report states as fact something it never looked at. ``created``
    happens to be true; ``present`` is simply false.

    So the artifact facts are absent, like every other field describing work
    that did not happen — and the file is proved untouched by INODE as well as
    by bytes, because "same content" would also hold for a file this run had
    replaced with an identical one.

    ASSERTED AS KEY ABSENCE, EXACTLY. ``assert not payload.get("backup")``
    passes against the very defect being fixed, because the old value was a
    dict whose fields were falsy in the ways that mattered. The key must not be
    there at all.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    for dry_run in (True, False):
        where = tmpdir / ("dry" if dry_run else "real")
        where.mkdir()
        store = where / "state.db"
        _fenced_store(store, leave_lease_live=False)

        occupied = where / "backup.db"
        occupied.write_bytes(b"an earlier backup that this run never looked at")
        before_bytes = occupied.read_bytes()
        before_stat = os.stat(occupied)
        digest_before = _store_digest(store)
        listing_before = sorted(entry.name for entry in where.iterdir())

        run = _run_verb(store, occupied, dry_run=dry_run)
        assert not run.crash, f"the verb crashed: {run.crash}"
        payload = _payload(run)
        assert payload is not None, f"no machine-readable report: {run.stdout!r}"

        published = sorted(_every_key(payload))
        for fact in ("backup", "created", "present", "backup_created",
                     "backup_present", "backup_withdrawn", "backup_verified",
                     "backup_durable"):
            assert not any(k.rsplit(".", 1)[-1] == fact for k in published), (
                f"the {'dry' if dry_run else 'real'} refusal reports "
                f"{fact!r} about a destination it never examined: {published}"
            )
        assert payload["refused"]["reason"] == "offline-authority-unknown"

        after_stat = os.stat(occupied)
        assert occupied.read_bytes() == before_bytes, (
            "the run altered a destination it never opened"
        )
        assert (after_stat.st_dev, after_stat.st_ino) == (
            before_stat.st_dev, before_stat.st_ino
        ), (
            "the destination is a different file than before — same bytes "
            "would not have caught a replacement"
        )
        assert _store_digest(store) == digest_before
        assert sorted(entry.name for entry in where.iterdir()) == listing_before


def check_a_target_that_is_not_this_fence_is_refused_by_a_named_reason(
    tmpdir: pathlib.Path,
) -> None:
    """MAINTENANCE ONLY. Three wrong targets, three distinct reasons.

    Not boundary evidence: telling a foreign database apart from a
    half-installed one requires OPENING it, and the verb refuses before it
    opens anything. Preserving that granularity at the boundary would mean the
    boundary peeking, which is the check-then-read defect this contract exists
    to remove. So the property moves behind the pre-flight, which no invocation
    reaches, and it is kept so it does not rot.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    junk = tmpdir / "someone-elses-notes.db"
    junk.write_bytes(b"this is not a database at all, it is a text file")
    foreign_dir = tmpdir / "foreign"
    foreign_dir.mkdir()
    foreign = foreign_dir / "other.db"
    conn = sqlite3.connect(str(foreign))
    try:
        conn.execute("CREATE TABLE unrelated (x TEXT)")
    finally:
        conn.close()

    outcomes = {}
    for name, target in (
        ("store-missing", tmpdir / "no-such-store.db"),
        ("store-unreadable", junk),
        ("surface-mismatch", foreign),
    ):
        work = tmpdir / f"work-{name}"
        work.mkdir()
        try:
            library.preflight_turn_fence_rollback(
                target, backup_path=tmpdir / f"backup-{name}.db", work_dir=work
            )
            outcomes[name] = "<no refusal>"
        except library.TurnFenceRollbackRefused as exc:
            outcomes[name] = exc.reason
        except BaseException as exc:  # noqa: BLE001 - carried, not swallowed
            # A row that removes the surface check makes the pre-flight run on
            # past it; whatever it hits next must reach this pin as a VALUE, or
            # the pin dies by falling over instead of by observing.
            outcomes[name] = f"<{type(exc).__name__}>"

    assert outcomes["store-missing"] == "store-missing", (
        "a target that is not there was not refused as store-missing: "
        f"{outcomes!r}"
    )
    assert outcomes["store-unreadable"] == "store-unreadable", (
        "a target that is not a database was not refused as store-unreadable: "
        f"{outcomes!r}"
    )
    assert outcomes["surface-mismatch"] == "surface-mismatch", (
        "a target carrying a different fence surface was not refused as "
        f"surface-mismatch: {outcomes!r}"
    )
    assert len(set(outcomes.values())) == 3, (
        f"three kinds of wrong target are not distinguishable: {outcomes!r}"
    )
    assert junk.read_bytes() == b"this is not a database at all, it is a text file"


def check_a_partial_surface_is_refused_whole_and_writes_no_backup(
    tmpdir: pathlib.Path,
) -> None:
    """MAINTENANCE ONLY. All-or-nothing, and the check runs before the drops.

    A verb that drops what it recognises and shrugs at the rest leaves a store
    fenced against some writes and not others. Behind the boundary now, because
    reading the installed surface means opening the artifact.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    victim = hermes_state_common.turn_fence_trigger_name("messages", "INSERT")
    conn = sqlite3.connect(str(store), isolation_level=None)
    try:
        conn.execute(f"DROP TRIGGER {victim}")
    finally:
        conn.close()
    remaining = _installed_triggers(store)
    rows_before = _canonical_rows(store)

    work_dir = tmpdir / "work"
    work_dir.mkdir()
    run = _drive_the_machinery(library, store, work_dir)

    assert not run["crash"], f"the machinery crashed: {run['crash']}"
    assert run["reason"] == "surface-mismatch", (
        f"a half-installed surface was not reported as one: {run['reason']!r}"
    )
    assert victim in run["detail"], (
        f"the refusal does not name what it expected: {run['detail']!r}"
    )
    assert not run["backup"].exists(), "a refused run wrote a backup"
    assert _installed_triggers(store) == remaining, (
        "the run refused and still removed triggers — it is not all-or-nothing"
    )
    assert _canonical_rows(store) == rows_before


def check_a_completed_rollback_reports_the_surface_it_removed(
    tmpdir: pathlib.Path,
) -> None:
    """MAINTENANCE ONLY. The completion report names what it did.

    A rollback that prints "done" leaves the operator to check by hand, which
    is the state this tool exists to replace. No invocation reaches a
    completion today; this keeps the report honest for the one that will.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    rows_before = _canonical_rows(store)
    triggers_before = _installed_triggers(store)

    work_dir = tmpdir / "work"
    work_dir.mkdir()
    run = _drive_the_machinery(library, store, work_dir)
    facts = run["outcome"].facts()

    assert not run["crash"], f"the machinery crashed: {run['crash']}"
    assert run["returned"] is not None, f"the rollback did not complete: {run['reason']!r}"
    assert facts["outcome"] == "committed", f"{facts!r}"
    assert sorted(run["returned"]["rows"]) if False else True
    report = run["returned"]
    assert report["verified"] is True, f"no verified backup claimed: {report!r}"
    assert report["durable"] is True, f"no durable backup claimed: {report!r}"
    assert report["rows"], f"the report does not say what was preserved: {report!r}"
    assert run["backup"].is_file()
    assert _canonical_rows(run["backup"]) == rows_before, (
        "the backup does not reproduce the prepared state"
    )
    assert _installed_triggers(run["backup"]) == triggers_before, (
        "the backup was taken after the drops"
    )
    # The source was never the subject.
    assert _installed_triggers(store) == triggers_before
    assert _canonical_rows(store) == rows_before


def check_a_destination_appearing_after_the_check_is_never_clobbered(
    tmpdir: pathlib.Path,
) -> None:
    """MAINTENANCE ONLY. "Must not already exist" is a create, not a look."""
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    work_dir = tmpdir / "work"
    work_dir.mkdir()
    backup = work_dir / "backup.db"
    sentinel = b"DO NOT CLOBBER - a concurrent writer got here first"
    injected = {"fired": False}
    real_check = library._refuse_unusable_backup_path

    def _competing_writer_after_the_check(path, *args, **kwargs):
        target = pathlib.Path(path)
        if target.name != backup.name:
            return real_check(path, *args, **kwargs)
        if target.exists():
            target.unlink()
        real_check(path, *args, **kwargs)
        injected["fired"] = True
        target.write_bytes(sentinel)

    library._refuse_unusable_backup_path = _competing_writer_after_the_check
    try:
        run = _drive_the_machinery(library, store, work_dir, backup=backup)
    finally:
        library._refuse_unusable_backup_path = real_check

    assert injected["fired"], "the sentinel was never written; this measures nothing"
    assert not run["crash"], f"the machinery crashed: {run['crash']}"
    assert backup.read_bytes() == sentinel, (
        "the backup step overwrote a file that appeared after its own check"
    )
    assert run["reason"] == "backup-exists", f"{run['reason']!r} {run['detail']!r}"
    assert _installed_triggers(store) == sorted(hermes_state_common.TURN_FENCE_TRIGGERS)


def check_an_orphan_backup_sidecar_is_not_overwritten(
    tmpdir: pathlib.Path,
) -> None:
    """MAINTENANCE ONLY. A backup destination is a FAMILY, every member of it."""
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    digest_before = _store_digest(store)
    work_dir = tmpdir / "work"
    work_dir.mkdir()
    backup = work_dir / "backup.db"
    orphan = work_dir / "backup.db-wal"
    orphan_bytes = b"an orphaned -wal from an attempt that died"
    orphan.write_bytes(orphan_bytes)

    run = _drive_the_machinery(library, store, work_dir, backup=backup)

    assert not run["crash"], f"the machinery crashed: {run['crash']}"
    assert orphan.read_bytes() == orphan_bytes, (
        "an orphaned backup sidecar was overwritten"
    )
    assert run["reason"] == "backup-exists", (
        "an orphan sidecar at the backup destination was not reported as an "
        f"existing backup: {run['reason']!r}"
    )
    assert str(orphan) in run["detail"], (
        f"the refusal does not name the file in the way: {run['detail']!r}"
    )
    assert _store_digest(store) == digest_before


def check_a_run_that_cannot_clean_up_does_not_report_success(
    tmpdir: pathlib.Path,
) -> None:
    """MAINTENANCE ONLY. Looking is the proof; rmtree returning is not.

    ``rmtree(..., ignore_errors=True)`` cannot tell "removed" from "failed and
    swallowed", and what is in a staging directory is a copy of every
    conversation in the store.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    work_dir = tmpdir / "work"
    work_dir.mkdir()

    class _StagingSurvives:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def rmtree(self, path, *args, **kwargs):
            if pathlib.Path(path).name.startswith(".fence-rollback-backup-"):
                return None
            return self._real.rmtree(path, *args, **kwargs)

    real_shutil = library.shutil
    library.shutil = _StagingSurvives(real_shutil)
    try:
        run = _drive_the_machinery(library, store, work_dir)
    finally:
        library.shutil = real_shutil

    facts = run["outcome"].facts()
    assert not run["crash"], f"the machinery crashed: {run['crash']}"
    assert run["returned"] is None, (
        f"a run that could not clean up reported a completed backup: {run!r}"
    )
    assert run["reason"] == "backup-staging-residue", f"{run['reason']!r}"
    assert facts["residue_present"] is True, (
        f"the un-removable staging copy is unreported: {facts!r}"
    )
    assert any(
        r.get("obligation") == "the backup staging directory"
        for r in facts["residue"]
    ), f"the residue does not name what it is: {facts['residue']!r}"


def check_a_residue_claim_names_the_fence_state_that_was_established(
    tmpdir: pathlib.Path,
) -> None:
    """MAINTENANCE ONLY. A claim is bound to what determines it.

    The message was generated where cleanup fails and inherited that site's
    subject, announcing every left-behind copy as an UNFENCED duplicate. Fence
    state is determined by whether the DDL committed against THAT artifact —
    and the working object is now an in-memory image, so nothing left on disk
    is ever unfenced. The claim must therefore never be ``unfenced``, and a
    record that says so is describing a file that does not exist.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    work_dir = tmpdir / "work"
    work_dir.mkdir()

    class _StagingSurvives:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def rmtree(self, path, *args, **kwargs):
            if pathlib.Path(path).name.startswith(".fence-rollback-backup-"):
                return None
            return self._real.rmtree(path, *args, **kwargs)

    real_shutil = library.shutil
    library.shutil = _StagingSurvives(real_shutil)
    try:
        run = _drive_the_machinery(library, store, work_dir)
    finally:
        library.shutil = real_shutil

    records = run["outcome"].facts()["residue"]
    assert records, "nothing was left behind, so this pin measures nothing"
    for record in records:
        assert record["fence_state"] != "unfenced", (
            "a left-behind artifact is announced as an UNFENCED duplicate of "
            "every conversation, and no on-disk artifact is ever unfenced — "
            f"the rollback runs against an in-memory image: {record!r}"
        )
        on_disk = pathlib.Path(record["path"])
        if record.get("holds") == "store-copy" and on_disk.is_file():
            assert _installed_triggers(on_disk) != [], (
                f"the record and the file disagree about the fence: {record!r}"
            )


def check_the_boundary_decides_liveness_again_keyed_by_the_operators_store(
    tmpdir: pathlib.Path,
) -> None:
    """MAINTENANCE ONLY. Two conjuncts: it decides AGAIN, keyed by the STORE.

    The lock must be released for the backup, so everything decided before it
    is a snapshot by the time the drops run. And the identity the liveness
    predicate is keyed by has to be the store the operator named, not the
    working object — a conversation this process is mid-turn on reads FREE
    against any other key.

    The fixture separates the two decisions deliberately: the lease row is
    owned by this process with NO grant, so both decisions would read it free,
    and the grant is taken in the backup window — the only interval between
    them. Free before, live after. Without that, the first decision refuses and
    a row deleting the second scores a kill it has not earned.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library
    from hermes_state import SessionDB

    store = tmpdir / "state.db"
    _fenced_store(store, leave_lease_live=False)
    stale = SessionDB(db_path=store)
    try:
        stale._execute_write(
            lambda conn: conn.execute(
                "UPDATE session_turn_leases SET holder = ?, owner_pid = ?, "
                "owner_pid_start = NULL WHERE conversation_id = ?",
                (_holder("stale"), os.getpid(), "keep"),
            )
        )
    finally:
        stale.close()
    triggers_before = _installed_triggers(store)

    work_dir = tmpdir / "work"
    work_dir.mkdir()
    barrier = {"granted": None, "db": None}

    def _a_turn_starts_in_the_backup_window(copy, backup):
        if barrier["granted"] is None:
            db = SessionDB(db_path=store)
            barrier["db"] = db
            barrier["granted"] = db.try_acquire_session_turn_lease(
                "keep", _holder("late"), ttl_seconds=600
            )

    try:
        run = _drive_the_boundary_and_meddle_after_the_backup(
            library, store, work_dir, _a_turn_starts_in_the_backup_window
        )
    finally:
        if barrier["db"] is not None:
            barrier["db"].close()

    assert run["backed_up"], "the backup never landed, so there was no window"
    assert barrier["granted"], (
        f"the barrier never took the turn: {barrier['granted']!r}"
    )
    assert not run["result"]["crash"], f"the boundary crashed: {run['result']['crash']}"
    assert run["result"]["reason"] == "live-turn", (
        "a turn taken between the backup and the drops was not refused. Either "
        "the boundary trusted its earlier snapshot, or it asked about the "
        f"working object rather than the store named: {run['result']!r}"
    )
    assert "keep" in run["result"]["detail"], f"{run['result']!r}"
    assert _installed_triggers(store) == triggers_before


def check_a_late_failure_does_not_retract_what_already_happened(
    tmpdir: pathlib.Path,
) -> None:
    """MAINTENANCE ONLY. Failure precedence sets the status, not the facts.

    Two situations that must not print the same thing: a rollback that
    COMPLETED and whose cleanup then failed, and one refused before it mutated
    anything whose cleanup then failed. Collapsing them into one outcome is the
    residue defect pointed the other way — an operator told "nothing happened"
    stops looking for the backup they now depend on.
    """
    _sandbox_home(tmpdir)
    module, why = _import_verb()
    assert module is not None, f"there is no fence-rollback verb: {why}"

    from hermes_cli import session_fence_rollback as library

    class _StagingSurvives:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def rmtree(self, path, *args, **kwargs):
            if pathlib.Path(path).name.startswith(".fence-rollback-backup-"):
                return None
            return self._real.rmtree(path, *args, **kwargs)

    def _run(where, *, break_surface):
        where.mkdir()
        store = where / "state.db"
        _fenced_store(store, leave_lease_live=False)
        if break_surface:
            victim = hermes_state_common.turn_fence_trigger_name("messages", "INSERT")
            conn = sqlite3.connect(str(store), isolation_level=None)
            try:
                conn.execute(f"DROP TRIGGER {victim}")
            finally:
                conn.close()
        work = where / "work"
        work.mkdir()
        real_shutil = library.shutil
        library.shutil = _StagingSurvives(real_shutil)
        try:
            return _drive_the_machinery(library, store, work)
        finally:
            library.shutil = real_shutil

    done = _run(tmpdir / "completed", break_surface=False)
    done_facts = done["outcome"].facts()
    assert not done["crash"], f"the machinery crashed: {done['crash']}"
    assert done_facts["backup_created"] is True, (
        f"a backup landed and the report denies it: {done_facts!r}"
    )
    assert done_facts["backup_durable"] is True, f"{done_facts!r}"
    assert done_facts["residue_present"] is True, (
        f"the un-removable staging copy is unreported: {done_facts!r}"
    )
    assert done["reason"] == "backup-staging-residue", f"{done['reason']!r}"

    stopped = _run(tmpdir / "refused", break_surface=True)
    stopped_facts = stopped["outcome"].facts()
    assert not stopped["crash"], f"the machinery crashed: {stopped['crash']}"
    assert stopped_facts["backup_created"] is False, (
        f"a backup is claimed for a run refused before it: {stopped_facts!r}"
    )
    assert stopped_facts["outcome"] == "not-started", f"{stopped_facts!r}"
    assert stopped["reason"] == "surface-mismatch", f"{stopped['reason']!r}"
    assert stopped["reason"] != done["reason"], (
        "a run that completed and one that never started report the SAME "
        f"outcome ({stopped['reason']})"
    )


PINS = {
    "check_the_verb_is_registered_under_sessions_and_names_its_target":
        check_the_verb_is_registered_under_sessions_and_names_its_target,
    "check_a_target_that_is_not_this_fence_is_refused_by_a_named_reason":
        check_a_target_that_is_not_this_fence_is_refused_by_a_named_reason,
    "check_a_partial_surface_is_refused_whole_and_writes_no_backup":
        check_a_partial_surface_is_refused_whole_and_writes_no_backup,
    "check_the_boundary_decides_liveness_again_keyed_by_the_operators_store":
        check_the_boundary_decides_liveness_again_keyed_by_the_operators_store,
    "check_a_target_swapped_for_another_valid_store_is_refused":
        check_a_target_swapped_for_another_valid_store_is_refused,
    "check_a_target_swapped_and_restored_around_any_open_cannot_be_reached":
        check_a_target_swapped_and_restored_around_any_open_cannot_be_reached,
    "check_a_destination_appearing_after_the_check_is_never_clobbered":
        check_a_destination_appearing_after_the_check_is_never_clobbered,
    "check_an_orphan_backup_sidecar_is_not_overwritten":
        check_an_orphan_backup_sidecar_is_not_overwritten,
    "check_a_run_that_cannot_clean_up_does_not_report_success":
        check_a_run_that_cannot_clean_up_does_not_report_success,
    "check_two_notices_of_one_incident_are_one_record":
        check_two_notices_of_one_incident_are_one_record,
    "check_two_incidents_with_identical_values_stay_two_records":
        check_two_incidents_with_identical_values_stay_two_records,
    "check_a_residue_claim_names_the_fence_state_that_was_established":
        check_a_residue_claim_names_the_fence_state_that_was_established,
    "check_a_run_that_creates_nothing_reports_no_residue":
        check_a_run_that_creates_nothing_reports_no_residue,
    "check_a_sidecar_that_cannot_be_released_is_not_a_successful_backup":
        check_a_sidecar_that_cannot_be_released_is_not_a_successful_backup,
    "check_a_foreign_file_at_a_reserved_sidecar_is_never_deleted":
        check_a_foreign_file_at_a_reserved_sidecar_is_never_deleted,
    "check_a_sidecar_that_vanished_is_not_claimed_as_our_removal":
        check_a_sidecar_that_vanished_is_not_claimed_as_our_removal,
    "check_every_surviving_destination_member_is_reported_exactly_once":
        check_every_surviving_destination_member_is_reported_exactly_once,
    "check_a_cleanup_failure_never_replaces_the_refusal_that_decided_the_run":
        check_a_cleanup_failure_never_replaces_the_refusal_that_decided_the_run,
    "check_a_late_failure_does_not_retract_what_already_happened":
        check_a_late_failure_does_not_retract_what_already_happened,
    "check_a_completed_rollback_reports_the_surface_it_removed":
        check_a_completed_rollback_reports_the_surface_it_removed,
    "check_no_in_place_run_succeeds_and_each_wrong_target_names_its_own_reason":
        check_no_in_place_run_succeeds_and_each_wrong_target_names_its_own_reason,
    "check_a_real_invocation_refuses_before_it_observes_the_source":
        check_a_real_invocation_refuses_before_it_observes_the_source,
    "check_a_late_sidecar_never_yields_committed_or_backup_facts":
        check_a_late_sidecar_never_yields_committed_or_backup_facts,
    "check_a_refusal_publishes_no_fact_it_never_produced":
        check_a_refusal_publishes_no_fact_it_never_produced,
    "check_an_unused_work_dir_cannot_change_the_verdict":
        check_an_unused_work_dir_cannot_change_the_verdict,
    "check_a_refusal_says_nothing_about_a_destination_it_never_examined":
        check_a_refusal_says_nothing_about_a_destination_it_never_examined,
    "check_an_artifact_whose_image_is_incomplete_is_refused":
        check_an_artifact_whose_image_is_incomplete_is_refused,
    "check_the_backup_describes_the_prepared_image_not_the_source_path":
        check_the_backup_describes_the_prepared_image_not_the_source_path,
    "check_a_partial_destination_collision_keeps_only_what_the_run_created":
        check_a_partial_destination_collision_keeps_only_what_the_run_created,
    "check_the_backup_file_itself_is_flushed_to_the_platter":
        check_the_backup_file_itself_is_flushed_to_the_platter,
    "check_the_backups_directory_entry_is_flushed_too":
        check_the_backups_directory_entry_is_flushed_too,
    "check_a_fault_after_commit_never_reports_that_nothing_changed":
        check_a_fault_after_commit_never_reports_that_nothing_changed,
    "check_a_withdrawn_backup_is_never_reported_as_one_the_operator_has":
        check_a_withdrawn_backup_is_never_reported_as_one_the_operator_has,
}



#: WHAT C5 IS GATED ON. Every pin here drives the verb through its public
#: entry points and asserts what the boundary guarantees: that a refusal is
#: reached from pathname metadata alone, before the store is opened, copied or
#: read; that the source's bytes and file set are invariant across it; and that
#: the report carries exactly the fields the run produced.
#:
#: A pin belongs here only if it can reach its verdict WITHOUT touching the
#: machinery. That is not a stylistic rule — the two buckets are counted
#: differently, so a boundary pin that quietly acquires machinery reach would
#: inflate the number that decides whether this slice is evidenced.
BOUNDARY_EVIDENCE = frozenset({
    "check_the_verb_is_registered_under_sessions_and_names_its_target",
    "check_no_in_place_run_succeeds_and_each_wrong_target_names_its_own_reason",
    "check_a_real_invocation_refuses_before_it_observes_the_source",
    "check_a_late_sidecar_never_yields_committed_or_backup_facts",
    "check_a_refusal_publishes_no_fact_it_never_produced",
    "check_a_refusal_says_nothing_about_a_destination_it_never_examined",
    "check_an_unused_work_dir_cannot_change_the_verdict",
    "check_a_run_that_creates_nothing_reports_no_residue",
})

#: MACHINERY NO INVOCATION REACHES. The exclusive destination acquisition, the
#: engine-written backup, the flushes, the withdrawal ledger, the residue
#: accounting, the private-copy preparation. These pins drive it directly so it
#: does not rot before the slice that supplies a provably coherent artifact.
#:
#: They are maintenance evidence and say NOTHING about what the verb does.
#: Counting them as boundary coverage is the same error as counting the anchor
#: guard as coverage of the verb.
MAINTENANCE_ONLY = frozenset({
    "check_a_cleanup_failure_never_replaces_the_refusal_that_decided_the_run",
    "check_a_completed_rollback_reports_the_surface_it_removed",
    "check_a_destination_appearing_after_the_check_is_never_clobbered",
    "check_a_fault_after_commit_never_reports_that_nothing_changed",
    "check_a_foreign_file_at_a_reserved_sidecar_is_never_deleted",
    "check_a_late_failure_does_not_retract_what_already_happened",
    "check_a_partial_destination_collision_keeps_only_what_the_run_created",
    "check_a_partial_surface_is_refused_whole_and_writes_no_backup",
    "check_a_residue_claim_names_the_fence_state_that_was_established",
    "check_a_run_that_cannot_clean_up_does_not_report_success",
    "check_a_sidecar_that_cannot_be_released_is_not_a_successful_backup",
    "check_a_sidecar_that_vanished_is_not_claimed_as_our_removal",
    "check_a_target_swapped_and_restored_around_any_open_cannot_be_reached",
    "check_a_target_swapped_for_another_valid_store_is_refused",
    "check_a_target_that_is_not_this_fence_is_refused_by_a_named_reason",
    "check_a_withdrawn_backup_is_never_reported_as_one_the_operator_has",
    "check_an_artifact_whose_image_is_incomplete_is_refused",
    "check_an_orphan_backup_sidecar_is_not_overwritten",
    "check_every_surviving_destination_member_is_reported_exactly_once",
    "check_the_backup_describes_the_prepared_image_not_the_source_path",
    "check_the_backup_file_itself_is_flushed_to_the_platter",
    "check_the_backups_directory_entry_is_flushed_too",
    "check_the_boundary_decides_liveness_again_keyed_by_the_operators_store",
    "check_two_incidents_with_identical_values_stay_two_records",
    "check_two_notices_of_one_incident_are_one_record",
})

#: Calling any of these is machinery reach. Attribute references are not — a
#: boundary pin may wrap ``preflight_turn_fence_rollback`` to COUNT that it was
#: never called, which is the opposite of driving it.
#: THE ONLY LIBRARY CODE OBJECTS A BOUNDARY-EVIDENCE PIN MAY ENTER.
#:
#: WRITTEN BY HAND AND CONFIRMED ONE BY ONE, not derived from what the pins do
#: today. A baseline taken from current execution RATIFIES current execution: a
#: new deep call would approve itself on its first run and the gate would
#: certify the thing it exists to catch.
#:
#: Every entry below is on the refusal path -- it constructs a record, compares
#: pathnames or inodes, or is a fail-closed wrapper. None of them prepares a
#: copy, reads the store, decides under a lock, backs up, or commits.
#:
#: ``disqualify_the_target.<listcomp>`` is a COMPREHENSION inside an approved
#: function, and it is listed because the roster is a set of CODE OBJECTS: a
#: comprehension is its own code object and `sys.setprofile` reports it
#: separately. The roster IS descriptor-rooted and DOES see it, because it
#: recurses ``co_consts`` from each live root -- what a descriptor enumeration
#: would miss is nested objects, and recursing is exactly how that is closed.
#: Module-rooting is a different thing and deliberately not used: it would add
#: the six import-time bodies and break identity matching.
_APPROVED_BOUNDARY_SURFACE = frozenset({
    "RollbackOutcome.__init__",
    "RollbackOutcome.facts",
    "RollbackOutcome.residue_present",
    "TurnFenceRollbackRefused.__init__",
    "_canonical_store_paths",
    "_same_file",
    "disqualify_the_target",
    "disqualify_the_target.<listcomp>",
    "establish_offline_authority",
    "rehearse_turn_fence_rollback",
    "rollback_turn_fence",
})

#: The BOUNDARY RUNTIME denominator: live descriptor roots plus their nested
#: closures. 58 named + 21 nested. The module body and the five class bodies are
#: deliberately NOT here -- they execute at import, before any pin runs, and
#: cannot be re-entered afterwards, so they are outside the claim "this pin's
#: execution reaches nothing deep". Covering them needs a separate instrument
#: (a fresh isolated import under the profiler), not a bigger runtime roster.
_LIBRARY_CODE_OBJECT_COUNT = 79

#: Reported separately, as a STATIC fact, never as the dynamic denominator.
_IMPORT_TIME_CODE_OBJECT_COUNT = 85
_IMPORT_TIME_ONLY_BODIES = 6


def _all_library_code_objects():
    """Live code objects a boundary pin can enter, keyed by ``id()``.

    IDENTITY, NOT VALUE. CPython code objects hash and compare by CONTENT, so a
    dict keyed by code objects matches a compile-derived graph too -- the roster
    would appear to work while being built from objects no live frame ever uses,
    and the day content diverged it would report total false-CLEAN, which is
    indistinguishable from a perfect run. Keying by ``id()`` makes the match mean
    what it says; the objects are held in ``_LIVE_CODE_OBJECTS`` so the ids stay
    valid for the process lifetime.

    Rooted at the imported module's DESCRIPTORS (functions, staticmethods,
    classmethods, property accessors) and recursed through ``co_consts``, so
    comprehensions, lambdas and generators are classified rather than excluded.
    """
    import hermes_cli.session_fence_rollback as library

    where = library.__file__
    roots = []

    def collect(obj, label):
        code = getattr(obj, "__code__", None)
        if code is not None and code.co_filename == where:
            roots.append((code, label))

    for name, value in vars(library).items():
        if isinstance(value, types.FunctionType):
            collect(value, name)
        elif isinstance(value, type) and getattr(value, "__module__", None) == library.__name__:
            for attribute, member in vars(value).items():
                label = f"{name}.{attribute}"
                if isinstance(member, types.FunctionType):
                    collect(member, label)
                elif isinstance(member, (staticmethod, classmethod)):
                    collect(member.__func__, label)
                elif isinstance(member, property):
                    for kind in ("fget", "fset", "fdel"):
                        accessor = getattr(member, kind)
                        if accessor is not None:
                            collect(accessor, label)

    labels, live = {}, []
    stack = list(roots)
    while stack:
        code, path = stack.pop()
        if id(code) in labels:
            continue
        label = path
        if label in labels.values():
            label = f"{path}#{code.co_firstlineno}"
        labels[id(code)] = label
        live.append(code)
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                stack.append((constant, f"{path}.{constant.co_name}"))
    _LIVE_CODE_OBJECTS.extend(live)
    return labels


#: Holds strong references so the ids used as roster keys cannot be recycled.
_LIVE_CODE_OBJECTS = []


def _deep_code_objects():
    """Everything NOT approved. Deep by default, approved only by argument."""
    return {
        key: label for key, label in _all_library_code_objects().items()
        if label not in _APPROVED_BOUNDARY_SURFACE
    }


def _wrapper_code_objects():
    """The approved surface. Evidence that the subject RAN at all."""
    return {
        key: label for key, label in _all_library_code_objects().items()
        if label in _APPROVED_BOUNDARY_SURFACE
    }


#: `threading.setprofile` is an ordinary Python function, so disabling the thread
#: hook arrives as a `call` event carrying this code object. `sys.setprofile` is a
#: C function, so it arrives as `c_call` with itself as `arg`. Both were measured
#: on both runtimes; they are the two ways the instrument can be switched off.
_THREADING_SETPROFILE_CODE = threading.setprofile.__code__


class _DeepCallTracer:
    """Records entries into the deep roster; restores whatever was installed."""

    def __init__(self, roster):
        self.roster = roster
        self.entered = []
        self.still_installed = False
        self.sys_hook_intact = False
        self.thread_hook_intact = False
        self.tampered = None
        # SCOPING: the judgement window opens once __enter__ has installed both
        # hooks and closes the instant __exit__ begins. The harness's own install
        # happens BEFORE this hook is active, so it cannot see it; its own restore
        # happens after `_closing` is set, so it does not accuse itself. This
        # boundary is where the check would go wrong if it did.
        self._closing = False
        self._previous_sys = None
        self._previous_thread = None
        # BIND ONCE. `self._hook` builds a NEW bound-method object on every
        # attribute access, so `sys.getprofile() is self._hook` compares two
        # different objects and is never true -- the post-check would fail on a
        # perfectly installed hook.
        self._bound_hook = self._hook

    def _hook(self, frame, event, arg):
        # TAMPER IS A TRANSITION, NOT AN END STATE. Identity at exit cannot see
        #
        #     h = sys.getprofile(); sys.setprofile(None)
        #     ...a deep call happens here, unobserved...
        #     sys.setprofile(h)
        #
        # because both hooks are correct again by the time anyone looks. This is
        # the third instance of that shape in this slice -- after the A->B->A
        # rollback target and the alias map read at its final state -- and the
        # fix is the same each time: watch the transition.
        #
        # MONOTONIC: set once, never cleared, for the same reason the outcome
        # ledger is monotonic. A flag that can be turned off is an end state again.
        if not self._closing and self.tampered is None:
            if event == "c_call" and arg is sys.setprofile:
                self.tampered = "sys.setprofile"
            elif event == "call" and frame.f_code is _THREADING_SETPROFILE_CODE:
                self.tampered = "threading.setprofile"
        if event == "call" and id(frame.f_code) in self.roster:
            self.entered.append(self.roster[id(frame.f_code)])
        return None

    def __enter__(self):
        # BOTH hooks are captured. Storing only the sys hook and then calling
        # threading.setprofile(None) DESTROYS a pre-existing thread hook instead
        # of restoring it -- the class promises restoration and delivered it for
        # one of the two.
        self._previous_sys = sys.getprofile()
        self._previous_thread = threading.getprofile()
        # threading.setprofile FIRST: sys.setprofile is per-thread and does not
        # reach threads that start later, and does not apply retroactively at all.
        threading.setprofile(self._bound_hook)
        sys.setprofile(self._bound_hook)
        return self

    def __exit__(self, *exc_info):
        # THE HOOK MUST STILL BE OURS. Code under test can call
        # sys.setprofile(None) -- directly, or through a library that profiles
        # something of its own -- and a hook uninstalled midway reports clean for
        # everything after that point, which is indistinguishable from a clean run.
        # BOTH HOOKS, against the cached identity. Checking only sys.getprofile()
        # leaves threading.setprofile(None) mid-subject undetected: new threads
        # then run unobserved while still_installed stays True, and the run
        # reports CLEAN for whatever they did.
        #
        # IDENTITY, NOT EQUALITY. Bound methods compare equal when __self__ and
        # __func__ match, so `==` would also pass for a hook installed by a
        # DIFFERENT tracer instance or a nested harness -- the check would then
        # mean "something resembling my hook is installed".
        self._closing = True
        self.sys_hook_intact = sys.getprofile() is self._bound_hook
        self.thread_hook_intact = threading.getprofile() is self._bound_hook
        self.still_installed = (
            self.sys_hook_intact and self.thread_hook_intact
            and self.tampered is None
        )
        sys.setprofile(self._previous_sys)
        threading.setprofile(self._previous_thread)
        return False


#: The ONE boundary pin that spawns a child, and why. Disposition B: exempted by
#: name with the exemption CHECKED, because an exemption whose precondition is
#: unverified is the defect this slice keeps finding. A profile hook does not
#: cross a process boundary, so whatever the child runs is invisible; the argv
#: assertion below is what stops that invisibility from widening silently.
_PIN_THAT_SPAWNS_A_CHILD = "check_a_late_sidecar_never_yields_committed_or_backup_facts"
_WHY_IT_SPAWNS = (
    "a committed-but-uncheckpointed WAL row needs a separate connection from a "
    "separate process, which is the whole subject of this pin"
)


def _validate_surface_partition(approved):
    """The partition itself, so a planted mutation can be required to fail."""
    everything = _all_library_code_objects()
    labels = set(everything.values())
    approved_labels = labels & set(approved)
    deep_labels = labels - set(approved)
    phantom = set(approved) - labels
    assert not phantom, (
        f"approved names that are not library code objects: {sorted(phantom)}. "
        f"A rename left the allowlist behind, and a stale entry approves nothing "
        f"while looking like it approves something"
    )
    assert len(approved_labels) + len(deep_labels) == len(labels), (
        "the partition does not sum to the surface"
    )
    assert approved_labels | deep_labels == labels
    assert not (approved_labels & deep_labels)
    return labels, approved_labels, deep_labels


def _assert_surface_literals(approved):
    """THE GATE ITSELF, callable with a mutated allowlist.

    Factored so a planted mutation can be required to kill this exact assertion.
    A planted test that only observes "the counts moved" proves the arithmetic
    works, not that the gate fires -- the same distinction as a mutation row that
    kills by crashing instead of by violating the property it names.
    """
    labels, approved_labels, deep = _validate_surface_partition(approved)
    assert (len(labels), len(approved_labels), len(deep)) == (
        _LIBRARY_CODE_OBJECT_COUNT, 11, _LIBRARY_CODE_OBJECT_COUNT - 11,
    ), (
        f"library runtime surface: {len(labels)} code objects, "
        f"{len(approved_labels)} approved, {len(deep)} deep. Expected "
        f"{_LIBRARY_CODE_OBJECT_COUNT}/11/{_LIBRARY_CODE_OBJECT_COUNT - 11}. A "
        f"code object appeared, left, or moved between the sets: classify it "
        f"deliberately. Do NOT edit the literal to make this green"
    )
    assert approved_labels == set(_APPROVED_BOUNDARY_SURFACE), (
        f"the approved set drifted: "
        f"{sorted(approved_labels ^ set(_APPROVED_BOUNDARY_SURFACE))}"
    )
    return labels, approved_labels, deep


def test_the_library_surface_partition_is_exact():
    """79 total, 11 approved, 68 deep -- as literals, and as exact sets.

    IF THIS FAILS BECAUSE THE COUNT MOVED: a code object appeared in or left the
    library and is unclassified. Add it to the approved list WITH A REASON it is
    safe for a boundary-evidence pin to enter, or leave it deep. Do NOT edit the
    number to make the gate green -- that is the one edit that kills this guard
    completely, and it looks like maintenance.

    Counts alone are not enough: a new object replacing a removed one keeps the
    total identical, so the sets are compared both ways.
    """
    labels, approved, deep = _assert_surface_literals(_APPROVED_BOUNDARY_SURFACE)

    # NESTED code objects ARE inside this count: the enumeration is rooted at
    # the imported module's LIVE DESCRIPTORS and recurses co_consts, so comprehensions, lambdas and
    # generators are classified rather than excluded with an argument.
    nested = sorted(label for label in labels if "<" in label.rsplit(".", 1)[-1])
    assert nested, (
        "no nested code objects were enumerated at all, so the recursion is not "
        "reaching them and the roster is descriptor-shaped again"
    )


def test_the_roster_holds_live_identities_not_compiled_copies():
    """PLANTED CONTROL for the most dangerous possible failure.

    A compile-derived graph is NOT the imported module's objects. Because code
    objects compare and hash by CONTENT, a roster built from one would appear to
    work -- and the day the content diverged it would match nothing and report
    total false-CLEAN, which looks exactly like a perfect run. So the roster is
    keyed by ``id()``, and this pins the distinction that makes that safe.
    """
    import hermes_cli.session_fence_rollback as library

    roster = _all_library_code_objects()
    live = library.rollback_turn_fence.__code__
    assert id(live) in roster, (
        "a production function's live code object is not in the roster by "
        "identity, so the tracer is watching objects no frame will ever carry"
    )

    compiled = library.__loader__.get_code(library.__name__)
    compiled_objects = []
    stack = [compiled]
    while stack:
        code = stack.pop()
        compiled_objects.append(code)
        stack.extend(c for c in code.co_consts if isinstance(c, types.CodeType))

    twin = next((c for c in compiled_objects if c == live), None)
    assert twin is not None, "expected a content-equal compiled twin to exist"
    assert twin is not live, "the compiled graph unexpectedly shares identity"
    assert id(twin) not in roster, (
        "a compile-derived code object is present in the roster by identity, so "
        "the roster is not built from the imported module"
    )


def test_a_nested_deep_code_object_fires_the_tracer():
    """POSITIVE CONTROL for a NESTED object, called directly.

    The named-function control cannot show this: comprehensions, lambdas and
    generators are separate code objects, and a roster that reached only named
    descriptors would miss deep logic moved one level in.
    """
    import hermes_cli.session_fence_rollback as library

    deep = _deep_code_objects()
    by_id = {id(code): code for code in _LIVE_CODE_OBJECTS}
    candidates = [
        (key, label) for key, label in deep.items()
        if label.rsplit(".", 1)[-1].startswith("<")
        and by_id[key].co_argcount == 1
    ]
    assert candidates, "no nested deep code object is callable as a probe"
    key, label = sorted(candidates, key=lambda item: item[1])[0]
    probe = types.FunctionType(by_id[key], vars(library))
    # NOT a faulting probe: a comprehension over an empty iterator returns
    # cleanly, so any exception here is a real failure and must propagate.
    entered = _deep_calls_of(lambda: probe(iter(())))
    assert label in entered, (
        f"a nested deep code object ({label}) ran and the tracer did not record "
        f"it, so nested objects are unwatched"
    )


#: sha256 of the child script AS IT EXECUTES -- after textwrap.dedent, which is
#: what the fixture actually hands to the interpreter. Hashing the raw literal
#: pins a string the child process never sees.
#:
#: CHECK THE ARTIFACT THAT EXECUTES, NEVER THE ARTIFACT THAT IS WRITTEN. Same
#: rule as the roster: a compile-derived code graph is not the imported module's
#: objects, and a source literal is not the dedented bytes. Wherever a value is
#: transformed on the way to use -- dedent, format, encode, compile, strip -- the
#: transformed value is the subject. The rest of this file was scanned for the
#: same shape: `marker.encode()` at the WAL assertions searches the artifact's own
#: bytes (correct direction), and the two `.strip()` uses normalise a command
#: result and an SQL statement. This was the only instance.
_WAL_CHILD_SCRIPT_SHA256 = (
    "e34a7e84f8965060539312e00b1e8b6be2b634b6209626e94264b32627d19ff9"
)


#: WHAT THE CHILD MUST DO, written from the fixture's PURPOSE, not read back
#: from its source. The digest alone is a baseline derived from the subject: it
#: changes whenever the subject changes and therefore approves whatever the
#: subject becomes. These are the independent expectations the digest anchors.
_WAL_CHILD_MUST_CONTAIN = (
    "sqlite3.connect(sys.argv[1]",        # the store this test names, not another
    "PRAGMA journal_mode=WAL",            # the marker must live in a WAL
    "PRAGMA wal_autocheckpoint=0",        # ...and must NOT be checkpointed out
    "INSERT INTO wal_marker",             # one row, in the fixture's own table
    "(sys.argv[2],)",                     # the marker value this test passes
    "os._exit(0)",                        # hard exit: a polite close checkpoints
)
#: A child on a boundary pin's path runs where the tracer cannot see. These are
#: the things it must never grow into.
_WAL_CHILD_MUST_NOT_CONTAIN = (
    "DROP", "DELETE", "ATTACH", "subprocess", "os.system", "eval(", "exec(",
    "unlink", "rmtree", "import hermes",
)


def test_the_child_the_wal_fixture_runs_is_pinned_by_content():
    """The one shell-out on a boundary pin's path, pinned as independent facts.

    A profile hook does not cross a process boundary, so whatever this child runs
    is invisible to the tracer. An argv PREFIX check approves any script that
    starts `python -c`; a digest ALONE is an expectation derived from the subject
    and moves with it. So the interpreter, the required behaviours, the forbidden
    behaviours, the target arguments and the digest are asserted separately --
    widening the script or adding a command fails several of them at once.
    """
    tree = ast.parse((REPO_ROOT / _SELF).read_text(encoding="utf-8"))
    owner = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_commit_a_marker_that_lives_only_in_the_wal"
    )
    calls = [
        node for node in ast.walk(owner)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("run", "Popen", "call", "check_output")
        and getattr(node.func.value, "id", "") == "subprocess"
    ]
    assert len(calls) == 1, f"expected exactly one shell-out, found {len(calls)}"
    argv = calls[0].args[0]
    assert isinstance(argv, ast.List) and len(argv.elts) == 5, (
        f"the child's argv shape changed: {len(getattr(argv, 'elts', []))} elements"
    )

    interpreter = argv.elts[0]
    assert isinstance(interpreter, ast.Attribute) \
        and interpreter.attr == "executable" \
        and getattr(interpreter.value, "id", "") == "sys", (
        "the child no longer runs THIS interpreter"
    )
    assert isinstance(argv.elts[1], ast.Constant) and argv.elts[1].value == "-c"

    script_call = argv.elts[2]
    assert isinstance(script_call, ast.Call) \
        and isinstance(script_call.func, ast.Attribute) \
        and script_call.func.attr == "dedent" \
        and getattr(script_call.func.value, "id", "") == "textwrap", (
        "the child's script is no longer textwrap.dedent(<literal>). The exact "
        "transform matters: every assertion below is computed by applying it, so "
        "a different one would leave them checking bytes that never run"
    )
    literal = script_call.args[0]
    assert isinstance(literal, ast.Constant) and isinstance(literal.value, str), (
        "the child's script is computed rather than literal; what it runs is no "
        "longer decidable from the source"
    )
    # THE EXECUTED BYTES, not the written ones.
    body = textwrap.dedent(literal.value)

    missing = [need for need in _WAL_CHILD_MUST_CONTAIN if need not in body]
    assert not missing, (
        f"the child no longer does what this fixture exists to do: {missing}. "
        f"Without an uncheckpointed WAL row the pin is testing nothing"
    )
    forbidden = [banned for banned in _WAL_CHILD_MUST_NOT_CONTAIN if banned in body]
    assert not forbidden, (
        f"the child gained {forbidden}. It runs where the tracer cannot follow, "
        f"so its reach is bounded here or nowhere"
    )
    assert body.count("sqlite3.connect") == 1, (
        f"the child opens {body.count('sqlite3.connect')} connections; one store, "
        f"one connection is the contract"
    )

    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert digest == _WAL_CHILD_SCRIPT_SHA256, (
        f"the child script changed (sha256 {digest}). The behavioural assertions "
        f"above still pass, so this is a REVIEW prompt, not a defect: re-read the "
        f"script, satisfy yourself it touches nothing in the deep roster, then "
        f"update this digest"
    )

    store_argument, marker_argument = argv.elts[3], argv.elts[4]
    assert isinstance(store_argument, ast.Call) \
        and getattr(store_argument.func, "id", "") == "str" \
        and getattr(store_argument.args[0], "id", "") == "store", (
        "the child is no longer pointed at this test's own `store`"
    )
    assert isinstance(marker_argument, ast.Name) and marker_argument.id == "marker", (
        "the child no longer receives this test's own `marker` value"
    )


def test_the_import_time_census_is_a_separate_fact():
    """85 code objects exist; 79 are reachable by a pin. Both, never conflated.

    The six extra are the module body and five class bodies. They execute at
    IMPORT, before any boundary pin runs, so a pin cannot re-enter them -- they
    are outside the claim this gate makes, not inside it and unwatched. Verifying
    them dynamically would need a fresh isolated import under the profiler, which
    is a different instrument; folding them into this roster would break identity
    matching and silently disarm the gate.
    """
    import hermes_cli.session_fence_rollback as library

    compiled = library.__loader__.get_code(library.__name__)
    total, stack = 0, [compiled]
    while stack:
        code = stack.pop()
        total += 1
        stack.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
    assert total == _IMPORT_TIME_CODE_OBJECT_COUNT, (
        f"the import-time census moved to {total}. This is a STATIC fact about "
        f"the module, reported beside the runtime roster and never as its "
        f"denominator"
    )
    assert total - _LIBRARY_CODE_OBJECT_COUNT == _IMPORT_TIME_ONLY_BODIES, (
        f"the import-time-only bodies moved to "
        f"{total - _LIBRARY_CODE_OBJECT_COUNT}: a new class body or module-level "
        f"executable block appeared, and it runs where this gate cannot see"
    )


def test_moving_a_deep_object_into_the_allowlist_is_caught():
    """PLANTED: approving something that is not in the library must fail."""
    with pytest.raises(AssertionError, match="not library code objects"):
        _validate_surface_partition(
            set(_APPROVED_BOUNDARY_SURFACE) | {"a_function_that_does_not_exist"}
        )


def test_a_real_deep_label_in_the_allowlist_kills_the_gate():
    """PLANTED: approve something genuinely deep; the exact gate must die.

    Not a phantom name, and not an observation that the arithmetic moved: a real
    deep code object is moved into the allowlist and the SAME assertion the real
    test relies on is required to raise.
    """
    _, _, deep = _validate_surface_partition(_APPROVED_BOUNDARY_SURFACE)
    victim = sorted(deep)[0]
    assert victim not in _APPROVED_BOUNDARY_SURFACE
    with pytest.raises(AssertionError) as caught:
        _assert_surface_literals(set(_APPROVED_BOUNDARY_SURFACE) | {victim})
    message = str(caught.value)
    assert "approved" in message and ("12" in message or "drifted" in message), (
        f"the gate raised, but not about the allowlist widening: {message[:200]}"
    )


def test_widening_the_allowlist_changes_the_deep_roster():
    """PLANTED: a real deep object moved into the allowlist stops being watched.

    This is the edit the gate exists to make expensive, so it is shown working:
    the object disappears from the deep roster, and the pinned literal catches it.
    """
    _, _, deep = _validate_surface_partition(_APPROVED_BOUNDARY_SURFACE)
    victim = sorted(deep)[0]
    _, widened, narrowed = _validate_surface_partition(
        set(_APPROVED_BOUNDARY_SURFACE) | {victim}
    )
    assert victim in widened and victim not in narrowed, (
        f"{victim} was moved into the allowlist and the partition did not move"
    )
    assert len(narrowed) == len(deep) - 1
    assert len(widened) != len(_APPROVED_BOUNDARY_SURFACE), (
        "the approved count did not change, so the pinned literal would not catch it"
    )


def test_an_arbitrary_deep_code_object_fires_the_tracer():
    """POSITIVE CONTROL on a deep object OTHER than the one the shapes use.

    Otherwise "the tracer fires" could be true only of one specially-handled
    function rather than of the roster.
    """
    import hermes_cli.session_fence_rollback as library

    deep = _deep_code_objects()
    chosen = "_make_verified_backup"
    assert chosen in set(deep.values()), f"{chosen} is not in the deep roster"
    # BUILT FROM THE SIGNATURE. A profile `call` event fires when the frame is
    # pushed, and argument binding happens first -- a hand-written arity that
    # drifts raises TypeError with no frame, and the probe would report "no deep
    # call" while looking like a clean result.
    target = getattr(library, chosen)
    parameters = inspect.signature(target).parameters
    positional, keywords = [], {}
    for parameter in parameters.values():
        if parameter.default is not inspect.Parameter.empty:
            continue
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            keywords[parameter.name] = None
        elif parameter.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional.append(None)
    entered = _deep_calls_of_a_faulting_probe(
        lambda: target(*positional, **keywords), chosen,
        fault=AttributeError, because="has no attribute 'with_name'",
    )
    assert chosen in entered


def test_the_tracer_restores_both_hooks_it_found():
    """Exact identity restoration of BOTH hooks, including the raising path.

    `finally` blocks are the least-exercised code in any harness, so the path
    where the subject raises is the one that rots silently.
    """
    def planted_sys(frame, event, arg):
        return None

    def planted_thread(frame, event, arg):
        return None

    sys.setprofile(planted_sys)
    threading.setprofile(planted_thread)
    try:
        _deep_calls_of(lambda: None)
        assert sys.getprofile() is planted_sys, "the sys hook was not restored"
        assert threading.getprofile() is planted_thread, (
            "the thread hook was destroyed rather than restored"
        )
        with pytest.raises(RuntimeError):
            _deep_calls_of(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert sys.getprofile() is planted_sys, (
            "the sys hook was not restored on the raising path"
        )
        assert threading.getprofile() is planted_thread, (
            "the thread hook was not restored on the raising path"
        )
    finally:
        sys.setprofile(None)
        threading.setprofile(None)


def test_the_post_check_detects_a_hook_uninstalled_mid_run():
    """PLANTED INSTANCE for the post-check itself.

    A guard I was asked for is not exempt from the standard asked of everything
    else: its first implementation compared bound-method objects that are rebuilt
    on every attribute access, so it was False while the hook was correctly
    installed. Untested, this check is exactly the kind this slice removes.
    """
    with pytest.raises(AssertionError, match="sys intact=False"):
        _deep_calls_of(lambda: sys.setprofile(None))


def _a_deep_call_that_would_be_missed():
    """One real deep entry, used inside a blinded window by the ABA controls."""
    import hermes_cli.session_fence_rollback as library

    try:
        library.prepare_the_private_copy(None, work_dir=None)
    except TypeError:
        pass


def test_the_post_check_detects_a_sys_hook_uninstalled_and_RESTORED():
    """PLANTED ABA: blind the sys hook, make a deep call, put the hook back.

    This is the case that passes an end-state identity check: at exit both hooks
    are the original objects and nothing looks wrong. The deep call in the middle
    was never observed, so the run would report CLEAN -- the same defect as an
    A->B->A rollback target that is correct whenever anyone stats it.
    """
    def aba():
        original = sys.getprofile()
        sys.setprofile(None)
        _a_deep_call_that_would_be_missed()
        sys.setprofile(original)

    with pytest.raises(AssertionError, match="tamper='sys.setprofile'"):
        _deep_calls_of(aba)


def test_the_post_check_detects_a_THREAD_hook_uninstalled_and_RESTORED():
    """PLANTED ABA on the thread hook, same shape, same requirement."""
    def aba():
        original = threading.getprofile()
        threading.setprofile(None)
        _a_deep_call_that_would_be_missed()
        threading.setprofile(original)

    with pytest.raises(AssertionError, match="tamper='threading.setprofile'"):
        _deep_calls_of(aba)


def test_the_tamper_flag_is_monotonic():
    """Set once, never cleared. A flag that can be turned off is an end state.

    The subject tampers and then does ordinary work; the flag must survive it,
    or a run could blind the instrument early and clear the evidence later.
    """
    def tamper_then_behave():
        original = sys.getprofile()
        sys.setprofile(None)
        sys.setprofile(original)
        for _ in range(50):
            sorted(range(10))

    with pytest.raises(AssertionError, match="tamper='sys.setprofile'"):
        _deep_calls_of(tamper_then_behave)


def test_the_tracer_does_not_accuse_its_own_install_or_restore():
    """SCOPING CONTROL: the harness's own hook management is outside the window.

    Without this the tracer trips on itself and every result is a failure --
    which is the failure mode a too-wide guard has, and it is as useless as one
    that never fires.
    """
    assert _deep_calls_of(lambda: None) == [], (
        "a clean subject reported deep calls"
    )
    entered = _deep_calls_of(lambda: sorted(range(5)))
    assert entered == [], "the tracer accused its own install or restore"


def test_the_post_check_detects_the_THREAD_hook_being_removed():
    """PLANTED for the half the post-check used to miss.

    Checking only `sys.getprofile()` leaves `threading.setprofile(None)`
    undetected: threads started after that point run unobserved while the run
    still reports CLEAN. Same shape as asserting one conjunct of two.
    """
    with pytest.raises(AssertionError, match="thread intact=False"):
        _deep_calls_of(lambda: threading.setprofile(None))


def test_a_boundary_subject_that_fails_is_not_reported_as_clean():
    """A pin that dies must be a FAILURE, never a zero.

    An aborted run and a clean run produce the same silence -- zero deep calls --
    and the silence gets read as the good outcome. Same shape as a mutation row
    that kills by crashing instead of by violating its property.
    """
    with pytest.raises(AssertionError, match="PIN_FAILED"):
        _deep_calls_of(lambda: (_ for _ in ()).throw(AssertionError("PIN_FAILED")))


def test_the_subprocess_census_of_the_boundary_pins(tmp_path):
    """TWO censuses, because they disagree and both are load-bearing.

    STATIC (source reachability): 1 of 8. The late-sidecar pin calls
    ``_commit_a_marker_that_lives_only_in_the_wal``, which shells out -- a
    committed-but-uncheckpointed WAL row needs a separate connection from a
    separate process, which is the whole subject of that pin.

    DYNAMIC (execution): 0 of 8, measured on BOTH runtimes. The shell-out sits
    behind a guard that is not taken on either, so the tracer's blind spot is
    real but currently unexercised.

    Reporting only one of these would be wrong in opposite directions: "1 of 8"
    overstates today's blind spot, and "0 of 8" hides a path one guard change
    away from reopening it. So both are asserted -- any child appearing at
    runtime is loud, and a new shell-out anywhere in the closure is loud.
    """
    # DYNAMIC
    seen = []
    real_run = subprocess.run

    def counting_run(argv, *args, **kwargs):
        seen.append(argv)
        return real_run(argv, *args, **kwargs)

    spawning = {}
    for name in sorted(BOUNDARY_EVIDENCE):
        seen.clear()
        where = tmp_path / name[:40]
        where.mkdir(parents=True, exist_ok=True)
        subprocess.run = counting_run
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                PINS[name](where)
        finally:
            subprocess.run = real_run
        if seen:
            spawning[name] = [list(argv[:2]) for argv in seen]

    assert not spawning, (
        f"a boundary pin spawned a child at runtime: {spawning}. A child is "
        f"invisible to the profile hook, so its deep calls are unobserved and "
        f"this gate's silence would no longer mean anything for that pin"
    )

    # STATIC
    tree = ast.parse((REPO_ROOT / _SELF).read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    closure, frontier = set(BOUNDARY_EVIDENCE), set(BOUNDARY_EVIDENCE)
    while frontier:
        nxt = set()
        for name in frontier:
            for node in ast.walk(functions[name]):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id in functions and node.func.id not in closure:
                    nxt.add(node.func.id)
        closure |= nxt
        frontier = nxt
    shelling = sorted(
        name for name in closure
        if any(
            isinstance(node, ast.Attribute) and node.attr in ("run", "Popen",
                                                              "check_output")
            and isinstance(node.value, ast.Name) and node.value.id == "subprocess"
            for node in ast.walk(functions[name])
        )
    )
    assert shelling == ["_commit_a_marker_that_lives_only_in_the_wal"], (
        f"the set of boundary-reachable functions that shell out changed: "
        f"{shelling}. Each one is a region the tracer cannot see into, so a new "
        f"one is a deliberate decision, not an edit"
    )


#: Measured: the registration pin exercises argparse only and never reaches the
#: library, so it is the one boundary pin that enters no wrapper.
_PIN_THAT_ONLY_PARSES = "check_the_verb_is_registered_under_sessions_and_names_its_target"


def test_the_boundary_pins_actually_execute_the_verb(tmp_path):
    """THE SECOND PROOF. "Zero deep calls" needs both halves, separately.

        was the INSTRUMENT live?   the dispatch-shape positive controls
        did the SUBJECT run?       this

    Both can be false independently and both look like zero. A pin that returned
    immediately, or one whose fixture silently skipped the verb, would report a
    clean gate forever.
    """
    roster = _wrapper_code_objects()
    entered_by = {}
    for name in sorted(BOUNDARY_EVIDENCE):
        where = tmp_path / name[:40]
        where.mkdir(parents=True, exist_ok=True)
        tracer = _DeepCallTracer(roster)
        with tracer:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                PINS[name](where)
        assert tracer.still_installed, f"the hook did not survive {name}"
        entered_by[name] = sorted(set(tracer.entered))

    silent = sorted(n for n, entered in entered_by.items() if not entered)
    assert silent == [_PIN_THAT_ONLY_PARSES], (
        f"boundary pins that never entered the library at all: {silent}. Their "
        f"zero-deep-calls result would be true of a pin that ran nothing, so it "
        f"proves nothing about reach"
    )
    assert "rollback_turn_fence" in set().union(*entered_by.values()), (
        "no boundary pin entered the public entry point, so the verb under test "
        "was never actually driven"
    )


@pytest.mark.parametrize("pin", sorted(BOUNDARY_EVIDENCE), ids=sorted(BOUNDARY_EVIDENCE))
def test_no_boundary_pin_enters_deep_maintenance(pin, tmp_path):
    """The gate, executed rather than read.

    Replaces a static classifier that was corrected in nineteen rounds and was
    still returning CLEAN for ordinary method and attribute dispatch. This
    watches code objects execute, so alias, helper, decorator, constructor,
    method and runtime attribute assignment are all one check.
    """
    entered = _deep_calls_of(lambda: PINS[pin](tmp_path))
    assert not entered, (
        f"{pin} is counted as boundary evidence and it ENTERED deep maintenance "
        f"code: {sorted(set(entered))}. Either the pin is maintenance and the "
        f"registry is wrong, or it acquired reach it should not have"
    )


def _the_deep_target():
    """One deep callable, invoked with a signature-correct call.

    ARITY MATTERS HERE. A profile ``call`` event fires when the frame is pushed,
    and argument binding happens first -- so calling a deep function with the
    wrong arity raises TypeError with no frame and no event, and the tracer
    would report nothing while looking like it had observed a clean run.
    """
    import hermes_cli.session_fence_rollback as library

    def enter_it():
        library.prepare_the_private_copy(None, work_dir=None)

    return enter_it


def _shape_direct(deep):
    deep()


def _shape_same_scope_alias(deep):
    alias = deep
    alias()


def _shape_nested_helper(deep):
    def helper():
        deep()
    helper()


def _shape_decorated_binding(deep):
    def swap(_fn):
        return deep
    @swap
    def helper():
        return 1
    helper()


def _shape_class_method(deep):
    class Holder:
        def go(self):
            deep()
    Holder().go()


def _shape_attribute_assignment(deep):
    class Holder:
        marker = 1
    holder = Holder()
    holder.run = deep
    holder.run()


def _shape_clean_method(deep):
    class Holder:
        def go(self):
            return 1
    Holder().go()


#: Every dispatch shape that defeated the static classifier, and one that must
#: stay clean. All six reaching shapes are the SAME code object under different
#: spellings, which is the point of tracing rather than reading.
_DISPATCH_SHAPES = {
    "direct": (_shape_direct, True),
    "same-scope-alias": (_shape_same_scope_alias, True),
    "nested-helper": (_shape_nested_helper, True),
    "decorated-binding": (_shape_decorated_binding, True),
    "class-method": (_shape_class_method, True),
    "attribute-assignment": (_shape_attribute_assignment, True),
    "clean-method": (_shape_clean_method, False),
}


@pytest.mark.parametrize("shape", sorted(_DISPATCH_SHAPES), ids=sorted(_DISPATCH_SHAPES))
def test_the_tracer_sees_every_dispatch_shape(shape):
    """POSITIVE CONTROLS. A tracer that never fires looks exactly like a clean run.

    Six of these defeated the static classifier one round at a time -- alias,
    nested helper, decorated rebinding, class method, runtime attribute
    assignment. Under a code-object roster they are one case. The seventh must
    stay clean, or "detects everything" would pass by flagging everything.
    """
    subject, must_fire = _DISPATCH_SHAPES[shape]
    deep = _the_deep_target()
    if must_fire:
        entered = _deep_calls_of_a_faulting_probe(
            lambda: subject(deep), "prepare_the_private_copy",
            fault=TypeError, because="expected str, bytes or os.PathLike object",
        )
        assert entered, (
            f"the {shape} shape reached a deep maintenance callable and the "
            f"tracer saw nothing. Either the hook was not installed or the "
            f"roster is not keyed by the code object that ran"
        )
    else:
        entered = _deep_calls_of(lambda: subject(deep))
        assert not entered, (
            f"the {shape} shape entered {entered}, but it calls nothing deep"
        )


def _deep_calls_of(subject):
    """Execute *subject* under the tracer. EXCEPTIONS PROPAGATE.

    Swallowing here would make a pin that crashed -- or that failed its own
    behavioural assertion -- produce the same observable as a clean run: zero
    deep calls. The gate would then manufacture CLEAN out of a broken execution.

    "Nothing found" owes two separate proofs, and neither substitutes for the
    other: that the INSTRUMENT was live (the positive control) and that the
    SUBJECT actually ran (this). Both can be false independently and both look
    like zero.
    """
    tracer = _DeepCallTracer(_deep_code_objects())
    with tracer:
        subject()
    assert tracer.still_installed, (
        f"the instrument did not hold for the whole subject, so an empty result "
        f"proves nothing: sys intact={tracer.sys_hook_intact}, thread "
        f"intact={tracer.thread_hook_intact}, tamper={tracer.tampered!r}. A "
        f"tamper with both identities restored is the ABA case: correct at the "
        f"end, blind in the middle"
    )
    return tracer.entered


def _deep_calls_of_a_faulting_probe(subject, expected, *, fault, because):
    """For SYNTHETIC probes that fault on purpose after entering a deep callable.

    THE FAULT CONTRACT IS EXACT. *fault* is the only exception type tolerated
    and *because* must appear in its message; anything else is re-raised. A
    control that passes on ANY exception is not proving the detector fired, only
    that something went wrong -- which is the same defect as a subject that
    swallows, one level in: the instrument that proves the detector works has to
    fail for the right reason too.
    """
    tracer = _DeepCallTracer(_deep_code_objects())
    with tracer:
        try:
            subject()
        except fault as exc:
            assert because in str(exc), (
                f"the probe raised {type(exc).__name__} but not for the expected "
                f"reason: wanted {because!r} in the message, got {str(exc)!r}. A "
                f"different failure at the same point would otherwise pass"
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised deliberately
            raise AssertionError(
                f"the probe raised {type(exc).__name__}: {exc}. Only {fault.__name__} "
                f"({because!r}) is the expected fault; anything else is a real bug "
                f"the control would have hidden"
            ) from exc
    assert tracer.still_installed, (
        f"the instrument did not hold for the probe: sys={tracer.sys_hook_intact}, "
        f"thread={tracer.thread_hook_intact}, tamper={tracer.tampered!r}"
    )
    assert expected in tracer.entered, (
        f"the probe was supposed to enter {expected!r} and the tracer recorded "
        f"{tracer.entered}. Catching its fault would hide a dead instrument"
    )
    return tracer.entered


"""Names a handler treats as known-safe. An explicit roster, not a fallback."""


#: EVERY PARAMETRIZED NODE AND THE EXACT EXPRESSION IT MUST FAN OUT OVER.
#:
#: A TOTAL IS NOT A COMPOSITION. `126 collected` looked healthy while
#: `test_sessions_fence_rollback_verb_property` had lost its parametrize and 33
#: pins had collapsed into one unrunnable item -- one number that many different
#: compositions can produce, so it cannot say WHICH one it is.
#:
#: The decorator was deleted by a bulk edit whose span ended at the next node's
#: `lineno`, which for a decorated function points at the `def` -- so the
#: FOLLOWING node's decorators fell inside the deleted range.
#:
#: EXACT EXPRESSIONS, COMPARED STRUCTURALLY. The first version of this guard used
#: `collection in ast.unparse(second_arg)` and accepted any callee whose name
#: ended in `parametrize`; `sorted(NOT_PINS)` satisfies a substring test for
#: "PINS", so a guard against a collapsing collection could be satisfied by the
#: WRONG collection. That is the fourth time in this slice a weaker relation
#: stood in for identity -- after `"AssertionError" in trace`, a decorator matched
#: by NAME, and a hook compared by EQUALITY. State the relation the claim needs,
#: then check that the code uses that relation: here the claim is "this exact
#: expression", so it is structural AST equality, never containment.
_PARAMETRIZED_NODES = {
    "test_sessions_fence_rollback_verb_property": (
        "name", "sorted(PINS)", "sorted(PINS)", "PINS",
    ),
    "test_each_pin_dies_when_its_own_guard_is_removed": (
        "mutation", "SOURCE_MUTATIONS", "[m.pin for m in SOURCE_MUTATIONS]",
        "SOURCE_MUTATIONS",
    ),
    "test_each_source_mutation_produces_the_behaviour_it_claims": (
        "mutation", "SOURCE_MUTATIONS", "[m.pin for m in SOURCE_MUTATIONS]",
        "SOURCE_MUTATIONS",
    ),
    "test_no_boundary_pin_enters_deep_maintenance": (
        "pin", "sorted(BOUNDARY_EVIDENCE)", "sorted(BOUNDARY_EVIDENCE)",
        "BOUNDARY_EVIDENCE",
    ),
    "test_the_tracer_sees_every_dispatch_shape": (
        "shape", "sorted(_DISPATCH_SHAPES)", "sorted(_DISPATCH_SHAPES)",
        "_DISPATCH_SHAPES",
    ),
}


def _is_pytest_parametrize(decorator) -> bool:
    """Exactly ``pytest.mark.parametrize``. Not "a callee ending in parametrize"."""
    if not isinstance(decorator, ast.Call):
        return False
    func = decorator.func
    return (
        isinstance(func, ast.Attribute) and func.attr == "parametrize"
        and isinstance(func.value, ast.Attribute) and func.value.attr == "mark"
        and isinstance(func.value.value, ast.Name) and func.value.value.id == "pytest"
    )


def _same_expression(node, expected: str) -> bool:
    """Structural equality against *expected* parsed as an expression."""
    return ast.dump(node) == ast.dump(ast.parse(expected, mode="eval").body)


def _check_parametrized_nodes(source: str) -> None:
    """The guard, over arbitrary source, so a planted mutation can kill it."""
    functions = {
        node.name: node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    for name, (argument, over, ids, _collection) in sorted(_PARAMETRIZED_NODES.items()):
        node = functions.get(name)
        assert node is not None, f"{name} no longer exists"
        marks = [d for d in node.decorator_list if _is_pytest_parametrize(d)]
        assert len(marks) == 1, (
            f"{name} carries {len(marks)} pytest.mark.parametrize decorators, "
            f"expected exactly 1. With none it runs as a single item -- or errors "
            f"on a missing fixture, taking every case with it"
        )
        mark = marks[0]
        assert len(mark.args) == 2, (
            f"{name}: parametrize takes ({argument!r}, <collection>); found "
            f"{len(mark.args)} positional arguments"
        )
        assert isinstance(mark.args[0], ast.Constant) \
            and mark.args[0].value == argument, (
            f"{name} is parametrized over "
            f"{ast.unparse(mark.args[0])}, expected {argument!r} exactly"
        )
        assert _same_expression(mark.args[1], over), (
            f"{name} fans out over `{ast.unparse(mark.args[1])}`, which does not "
            f"match `{over}`. Containment would accept `sorted(NOT_{over})`; this "
            f"compares the expression structurally"
        )
        keywords = {k.arg: k.value for k in mark.keywords}
        assert "ids" in keywords, f"{name}: parametrize has no ids= expression"
        assert _same_expression(keywords["ids"], ids), (
            f"{name} labels its cases with `{ast.unparse(keywords['ids'])}`, "
            f"which does not match `{ids}`. Wrong ids make a failure name the "
            f"wrong case"
        )


def test_every_parametrized_node_still_fans_out():
    """Per-node composition, so collect-only says WHAT it collected, not how many."""
    _check_parametrized_nodes((REPO_ROOT / _SELF).read_text(encoding="utf-8"))
    sizes = {
        "PINS": len(PINS),
        "SOURCE_MUTATIONS": len(SOURCE_MUTATIONS),
        "BOUNDARY_EVIDENCE": len(BOUNDARY_EVIDENCE),
        "_DISPATCH_SHAPES": len(_DISPATCH_SHAPES),
    }
    for name, (_argument, _over, _ids, collection) in sorted(_PARAMETRIZED_NODES.items()):
        assert sizes[collection] > 1, (
            f"{name} fans out over {collection}, which has {sizes[collection]} "
            f"entries -- parametrizing over it proves nothing about fan-out"
        )


def _the_property_decorator():
    """(source, target) where *target* names the REAL decorator, uniquely.

    ANCHORED ON THE `def` LINE ON PURPOSE. The bare decorator text also appears
    inside these planted tests as a string literal, and it appears EARLIER in the
    file -- so `source.replace(bare, ..., 1)` edited this test's own body and left
    the subject untouched. The guard then passed, and the planted mutation
    reported DID NOT RAISE while proving nothing.

    Same family as the child-script digest and the compile-derived roster: the
    edit was applied to a different artifact than the one under test. Uniqueness
    is asserted rather than assumed.
    """
    source = (REPO_ROOT / _SELF).read_text(encoding="utf-8")
    target = (
        '@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))\n'
        "def test_sessions_fence_rollback_verb_property("
    )
    assert source.count(target) == 1, (
        f"the planted-mutation anchor matches {source.count(target)} places; it "
        f"must name exactly one, or the mutation lands somewhere unintended"
    )
    return source, target


def test_the_fan_out_guard_dies_on_a_wrong_collection():
    """PLANTED: the guard must FAIL when a node fans out over the wrong thing.

    `NOT_PINS` CONTAINS `PINS`, which is exactly what the first version of this
    guard accepted. A composition guard with no killing mutation is the same
    object as a pin with no killing mutation: it exists, and whether it can fail
    is a separate question with its own answer.
    """
    source, target = _the_property_decorator()
    mutated = source.replace(
        target, target.replace("sorted(PINS),", "sorted(NOT_PINS),", 1), 1
    )
    assert mutated != source, "the planted mutation matched nothing"
    with pytest.raises(AssertionError, match="does not match"):
        _check_parametrized_nodes(mutated)


def test_the_fan_out_guard_dies_on_a_foreign_parametrize():
    """PLANTED: a callee that merely ENDS IN parametrize must not satisfy it."""
    source, target = _the_property_decorator()
    mutated = source.replace(target, "@not" + target.lstrip("@"), 1)
    assert mutated != source, "the planted mutation matched nothing"
    with pytest.raises(AssertionError, match="expected exactly 1"):
        _check_parametrized_nodes(mutated)


def test_the_fan_out_guard_dies_when_the_decorator_is_deleted():
    """PLANTED: the ORIGINAL defect -- the decorator simply gone."""
    source, target = _the_property_decorator()
    mutated = source.replace(target, target.split("\n", 1)[1], 1)
    assert mutated != source, "the planted mutation matched nothing"
    with pytest.raises(AssertionError, match="expected exactly 1"):
        _check_parametrized_nodes(mutated)


def test_every_pin_is_in_exactly_one_evidence_bucket():
    """The census is a fact of the FILE, and it is a partition. Not coverage.

    The split decides what counts as this slice's evidence, and it lived only
    in a chat message: nobody reading the repository could derive it, disagree
    with it, or notice it drifting. That is the defect this whole slice has
    been about — a property named in one place and enforced in none — applied
    to the classification itself.

    An EXACT partition, not two subset checks. A new pin that lands in neither
    bucket would otherwise vanish from the census silently, which is precisely
    how a number that only goes down becomes indistinguishable from a number
    that lost something.
    """
    pins = set(PINS)
    both = BOUNDARY_EVIDENCE & MAINTENANCE_ONLY
    assert not both, f"pins claimed by both buckets: {sorted(both)}"
    unclassified = pins - BOUNDARY_EVIDENCE - MAINTENANCE_ONLY
    assert not unclassified, (
        f"pins in neither bucket, so they are in no census: {sorted(unclassified)}"
    )
    phantom = (BOUNDARY_EVIDENCE | MAINTENANCE_ONLY) - pins
    assert not phantom, (
        f"bucketed names that are not pins — a rename left the registry behind: "
        f"{sorted(phantom)}"
    )
    assert len(BOUNDARY_EVIDENCE) + len(MAINTENANCE_ONLY) == len(pins), (
        f"the split does not sum to the file: {len(BOUNDARY_EVIDENCE)} + "
        f"{len(MAINTENANCE_ONLY)} != {len(pins)}"
    )
    # THE CENSUS AS LITERALS, so it cannot drift silently. The three checks
    # above are all relative — they stay green while both buckets grow, which
    # is exactly how a reported number stops matching the file without anyone
    # editing the number. A pin added to either bucket must move a digit here,
    # in the same commit, and that digit is what the report quotes.
    assert (len(BOUNDARY_EVIDENCE), len(MAINTENANCE_ONLY)) == (8, 25), (
        "the evidence census moved and the literals did not: boundary="
        f"{len(BOUNDARY_EVIDENCE)} maintenance={len(MAINTENANCE_ONLY)}. If the "
        "change is intended, update these numbers AND every place the split is "
        "reported; if it is not, a pin has landed in the wrong bucket"
    )


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_sessions_fence_rollback_verb_property(name, tmp_path):
    PINS[name](tmp_path)


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_a_real_invocation_refuses_before_it_observes_the_source",
        module="hermes_cli/session_fence_rollback.py",
        find="    # governs has not been made in time.\n"
             "    disqualify_the_target(store_path)\n"
             "    establish_offline_authority(store_path)\n",
        replace="    # governs has not been made in time.\n"
                "    disqualify_the_target(store_path)\n"
                "    with open(store_path, \"r+b\") as _h:\n"
                "        _h.seek(24)\n"
                "        _was = _h.read(1)\n"
                "        _h.seek(24)\n"
                "        _h.write(bytes([_was[0] ^ 1]))\n"
                "    establish_offline_authority(store_path)\n",
        kills_by='the refusal rewrote the source IN PLACE. The file set is unchanged and no watched event fired, which is exactly why the listing and the counters cannot carry this on their own:',
        why="SECOND CONJUNCT: byte invariance. The other row moves the refusal "
            "after the source is OBSERVED; this one rewrites the source IN "
            "PLACE, which changes neither the directory listing nor any event "
            "counter the pin watches. Found by mutation, not review: a correct "
            "argument for asserting events had silently become a reason for "
            "omitting content, and the pin stayed green while the operator's "
            "store was modified",
    ),
    Mutation(
        pin="check_a_late_sidecar_never_yields_committed_or_backup_facts",
        module="hermes_cli/session_fence_rollback.py",
        find="    # deciding after it would be deciding on the strength of what is in doubt.\n"
             "    disqualify_the_target(store_path)\n",
        replace="    # deciding after it would be deciding on the strength of what is in doubt.\n"
                "    with BoundTarget(store_path) as _restored:\n"
                "        prepare_the_private_copy(\n"
                "            store_path, work_dir=Path(tempfile.mkdtemp()), bound=_restored\n"
                "        )\n"
                "    disqualify_the_target(store_path)\n",
        kills_by='the run read a source image, so there is an interval between deciding the source is quiesced and reading it — and a writer that commits in that interval is invisible to both ends of it (',
        why="restoring a read of the source image reopens the interval between "
            "the sidecar check and the read. A writer committing there leaves "
            "the check true and the image short, and every fact derived after "
            "it describes a database missing rows",
    ),
    Mutation(
        pin="check_a_refusal_publishes_no_fact_it_never_produced",
        module="hermes_cli/session_fence_rollback_cmd.py",
        find="    resolved_preflight = preflight or getattr(exc, \"preflight\", None)\n",
        replace="    resolved_preflight = preflight or getattr(exc, \"preflight\", None) or {\n"
                "        \"target_present\": False,\n    }\n",
        kills_by='refusal publishes a different set of fields than it produced. extra=',
        why="a preflight block of false checks claims each was performed and "
            "failed. None ran. Absent and false are different statements, and "
            "the false one is the report answering a question nobody asked",
    ),
    Mutation(
        pin="check_a_refusal_says_nothing_about_a_destination_it_never_examined",
        module="hermes_cli/session_fence_rollback_cmd.py",
        find="    if facts.get(\"backup\") is not None:\n        payload[\"backup\"] = facts[\"backup\"]\n",
        replace="    payload[\"backup\"] = facts.get(\"backup\") or {\n"
                "        \"path\": str(backup), \"created\": False, \"present\": False,\n    }\n",
        kills_by='about a destination it never examined:',
        why="the boundary never examines the destination, so `present: false` "
            "is not a cautious default -- on an operator who names an existing "
            "file the report states as fact something it never looked at",
    ),
    Mutation(
        pin="check_an_unused_work_dir_cannot_change_the_verdict",
        module="hermes_cli/session_fence_rollback_cmd.py",
        find="    # --work-dir IS NOT EXAMINED.",
        replace="    if work_parent is not None and not work_parent.is_dir():\n"
                "        return _emit_refusal(\n"
                "            rollback.TurnFenceRollbackRefused(\n"
                "                \"no such --work-dir\", reason=\"rehearsal-unwritable\",\n"
                "            ),\n"
                "            store=store, backup=backup, dry_run=True,\n"
                "        )\n"
                "    # --work-dir IS NOT EXAMINED.",
        kills_by='refused for a different reason than the others:',
        why="an input that feeds a capability the boundary removed must stop "
            "influencing the outcome. Stat'ing it lets a nonexistent path give "
            "a different reason than the real run gives, so a dead flag decides "
            "what the operator is told",
    ),
    Mutation(
        pin="check_the_verb_is_registered_under_sessions_and_names_its_target",
        module="hermes_cli/session_fence_rollback_cmd.py",
        find='        "fence-rollback",\n        help=(\n',
        replace='        "fence-rollback-disabled",\n        help=(\n',
        kills_by='the verb refused a fully named invocation:',
        why="SECOND CONJUNCT: registered UNDER SESSIONS. The other row attacks "
            "the target being named; nothing attacked the verb existing at the "
            "name an operator types, so a rename to anything at all would have "
            "left the table green",
    ),
    Mutation(
        pin="check_no_in_place_run_succeeds_and_each_wrong_target_names_its_own_reason",
        module="hermes_cli/session_fence_rollback.py",
        find='    raise TurnFenceRollbackRefused(\n'
             '        f"no capability in this build can establish that {artifact} is offline, "\n'
             '        "so the rollback is refused. The only producer of a detached state.db "\n'
             '        "here is the quick snapshot, whose manifest records file SIZE only — a "\n'
             '        "same-size replacement satisfies it while the contents are a different "\n'
             '        "database — so nothing available proves which store an artifact is or "\n'
             '        "that it is detached. Nothing was changed",\n'
             '        reason=DISQUALIFICATION_REASONS["unknown"],\n'
             '    )\n',
        replace="    return None\n",
        kills_by="the verb SUCCEEDED in place on the",
        why="SECOND CONJUNCT: NO in-place run succeeds. The other row collapses "
            "the four reasons into one, which leaves every target still "
            "refused; this one lets an ordinary idle store through, which is "
            "the fail-closed half and the whole contract. REPLACES THE WHOLE "
            "RAISE STATEMENT rather than disabling it in place: an earlier "
            "form composed to `raise None`, which is a TypeError, so the pin "
            "died at its crash detector and the row scored a kill while "
            "measuring nothing",
    ),
    Mutation(
        pin="check_the_boundary_decides_liveness_again_keyed_by_the_operators_store",
        module="hermes_cli/session_fence_rollback.py",
        find="        private_copy.verify()\n"
             "        _decide_under_the_lock(conn, reported, expected, bound=None)\n",
        replace="        private_copy.verify()\n",
        kills_by='a turn taken between the backup and the drops was not refused. Either the boundary trusted its earlier snapshot, or it asked about the working object rather than the store named:',
        why="SECOND CONJUNCT: decides AGAIN. The other row attacks the keying "
            "and leaves the re-decision standing; this one deletes the "
            "re-decision and leaves the keying intact, so the boundary trusts "
            "the pre-flight's snapshot. That is the TOCTOU half, and it is the "
            "property the merged-away pin used to own",
    ),
    Mutation(
        pin="check_the_verb_is_registered_under_sessions_and_names_its_target",
        module="hermes_cli/session_fence_rollback_cmd.py",
        find='        "--store",\n        type=Path,\n        required=True,\n',
        replace='        "--store",\n        type=Path,\n'
                '        default=Path("~/.hermes/state.db").expanduser(),\n',
        kills_by='. A rollback that can run without being handed a store is a rollback that can run on the wrong one',
        why="giving the target a default is the convenient version of this "
            "verb and the one that acts on a store nobody named. The "
            "invocation that touches the wrong file then looks identical in "
            "the shell history to the one the operator meant",
    ),
    Mutation(
        pin="check_the_verb_is_registered_under_sessions_and_names_its_target",
        module="hermes_cli/session_fence_rollback_cmd.py",
        find='        "--store",\n        type=Path,\n        required=True,\n',
        replace='        "--store",\n'
                "        type=lambda raw: (\n"
                "            Path(raw).parent.is_dir()\n"
                '            and Path(raw).parent.joinpath(".probe").touch(),\n'
                "            Path(raw),\n"
                "        )[1],\n"
                "        required=True,\n",
        kills_by="parsing the verb's arguments created or removed files:",
        why="THIRD CONJUNCT: argv parsing is INERT ON DISK. The other two rows "
            "attack the verb's name and its required target, and both leave a "
            "parser that writes while it converts entirely unseen. This mutant "
            "is deliberately invisible to everything except the conjunct it "
            "targets: it returns the SAME Path, so registration and "
            "required-target stay green, and it creates exactly one file under "
            "the parsed --store parent, so the pin dies at the file-set "
            "before/after and nowhere else. A looser mutant that also broke "
            "parsing would kill by crashing, and the other two conjuncts would "
            "mask which property actually failed",
    ),
    Mutation(
        pin="check_a_target_that_is_not_this_fence_is_refused_by_a_named_reason",
        module="hermes_cli/session_fence_rollback.py",
        find="    if list(installed) == list(expected):\n        return\n",
        replace="    if True:\n        return\n",
        kills_by='a target carrying a different fence surface was not refused as surface-mismatch:',
        why="without the surface check nothing decides what the target IS "
            "before it is treated as a store, and three different wrong "
            "targets stop being distinguishable from each other",
    ),
    Mutation(
        pin="check_a_partial_surface_is_refused_whole_and_writes_no_backup",
        module="hermes_cli/session_fence_rollback.py",
        find="    if list(installed) == list(expected):\n",
        replace="    if set(installed) <= set(expected):\n",
        kills_by='a half-installed surface was not reported as one:',
        why="tolerating a SUBSET is exactly 'drop what we recognise and shrug "
            "at the rest'. A store left fenced against some writes and not "
            "others is worse than either end state",
    ),
    Mutation(
        pin="check_the_boundary_decides_liveness_again_keyed_by_the_operators_store",
        module="hermes_cli/session_fence_rollback.py",
        find="    reported = Path(report_as) if report_as is not None else target_path\n",
        replace="    reported = target_path\n",
        kills_by='a turn taken between the backup and the drops was not refused. Either the boundary trusted its earlier snapshot, or it asked about the working object rather than the store named:',
        why="the liveness predicate has a branch keyed by the store's PATH, "
            "so answering it about the copy frees a conversation this process "
            "is genuinely mid-turn on. The rehearsal then reports that a run "
            "about to be refused would proceed",
    ),
    Mutation(
        pin="check_a_target_swapped_and_restored_around_any_open_cannot_be_reached",
        module="hermes_cli/session_fence_rollback.py",
        find='    connection = sqlite3.connect(":memory:", isolation_level=None)\n'
             "    try:\n"
             "        connection.deserialize(_as_a_rollback_journal_image(image))\n",
        replace='    probe = sqlite3.connect(str(copy), isolation_level=None)\n'
                "    probe.close()\n"
                '    connection = sqlite3.connect(":memory:", isolation_level=None)\n'
                "    try:\n"
                "        connection.deserialize(_as_a_rollback_journal_image(image))\n",
        kills_by='x), so an A->B->A lands inside that interval and is invisible to a check on either side. The object has to be held, not looked at again',
        why="re-opening by pathname restores the interval between the last "
            "identity check and the open. A->B->A lands inside it and is "
            "invisible to a stat on either side, so the rollback commits "
            "against a store nothing prepared and leaves the substituted one "
            "corrupted, reporting success. Every identity check stays in "
            "place, which is the point — checks do not close intervals",
    ),
    Mutation(
        pin="check_a_target_swapped_for_another_valid_store_is_refused",
        module="hermes_cli/session_fence_rollback.py",
        find="        private_copy.verify()\n"
             "        _decide_under_the_lock(conn, reported, expected, bound=None)\n",
        replace="        _decide_under_the_lock(conn, reported, expected, bound=None)\n",
        kills_by='a substituted target was not reported as one. A consistency check cannot see this — the substitute is a perfectly valid fenced store — so only an identity check can:',
        why="without re-establishing identity at the point of use the target "
            "is bound to a NAME again, and a different valid fenced store "
            "moved to that name passes every consistency check while the "
            "backup describes the file that left. The liveness decision left "
            "in place cannot see it — the substitute is perfectly consistent",
    ),
    Mutation(
        pin="check_a_destination_appearing_after_the_check_is_never_clobbered",
        module="hermes_cli/session_fence_rollback.py",
        find="        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY\n",
        replace="        flags = os.O_CREAT | os.O_TRUNC | os.O_WRONLY\n",
        kills_by='the backup step overwrote a file that appeared after its own check',
        why="restoring overwrite-capable creation is the defect itself: "
            "whatever appeared between the existence check and the write is "
            "truncated. No better-placed check can fix it, because a check is "
            "what is broken",
    ),
    Mutation(
        pin="check_an_orphan_backup_sidecar_is_not_overwritten",
        module="hermes_cli/session_fence_rollback.py",
        find='_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")\n',
        replace="_SIDECAR_SUFFIXES = ()\n",
        kills_by='an orphan sidecar at the backup destination was not reported as an existing backup:',
        why="an orphaned backup.db-wal is a destination too, and narrowing "
            "the family to the one name the operator typed is what makes it "
            "invisible. Aimed at the DECLARATION because the family is "
            "load-bearing on a journal_mode=DELETE build and redundant on a "
            "WAL one — a mutation at either guard alone scores a kill on one "
            "SQLite and none on the other",
    ),
    Mutation(
        pin="check_a_run_that_cannot_clean_up_does_not_report_success",
        module="hermes_cli/session_fence_rollback.py",
        find="    if not work_dir.exists():\n        return None\n",
        replace="    if True:\n        return None\n",
        kills_by='a run that could not clean up reported a completed backup:',
        why="trusting the removal call instead of looking is the defect: "
            "rmtree cannot distinguish 'removed' from 'failed and swallowed', "
            "so a rolled-back — unfenced — duplicate of every conversation "
            "stays on disk while the verb reports a clean run",
    ),
    Mutation(
        pin="check_a_late_failure_does_not_retract_what_already_happened",
        module="hermes_cli/session_fence_rollback.py",
        find='        if outcome is not None:\n'
             '            outcome.advance("backup-created")\n'
             '            outcome.advance("backup-verified")\n'
             '            outcome.advance("backup-durable")\n'
             "            outcome.backup = report\n",
        replace="        if outcome is None:\n            pass\n",
        kills_by='a backup landed and the report denies it:',
        why="collapsing the two situations into one reason makes a directory "
            "holding an UNFENCED duplicate indistinguishable from one holding "
            "a copy that still carries the fence. Same words, opposite "
            "urgency, and the operator has no way to tell which they have",
    ),
    Mutation(
        pin="check_two_notices_of_one_incident_are_one_record",
        module="hermes_cli/session_fence_rollback.py",
        find="        for position, existing in enumerate(self.residue):\n"
             '            if existing.get("incident") == incident:\n'
             "                self.residue[position] = entry\n"
             "                return\n",
        replace="",
        kills_by='one incident observed twice became two records. The second observer is not a second stuck directory:',
        why="append-only without an identity is the mirror image of the "
            "erasure it replaced: every observer of one incident adds a "
            "record, so looking at a stuck directory twice makes two of them",
    ),
    Mutation(
        pin="check_two_incidents_with_identical_values_stay_two_records",
        module="hermes_cli/session_fence_rollback.py",
        find='            if existing.get("incident") == incident:\n',
        replace="            if {k: v for k, v in existing.items() if k != 'incident'} == record:\n",
        kills_by='two distinct incidents that happened to produce identical payloads were collapsed into one. A run can fail to remove its working copy AND a backup it tried to withdraw, and those are two places to go:',
        why="deduplicating by VALUE is the obvious repair for a double count "
            "and the wrong one. Two distinct failures can produce identical "
            "payloads, and collapsing them loses a place the operator has to "
            "go — a spurious record costs a look, a missing one leaves an "
            "unfenced duplicate nobody is told about",
    ),
    Mutation(
        pin="check_a_residue_claim_names_the_fence_state_that_was_established",
        module="hermes_cli/session_fence_rollback.py",
        find='    if holds != "store-copy":\n        fence_state = "not-applicable"\n',
        replace='    if holds != "store-copy":\n        fence_state = "unfenced"\n',
        kills_by='a left-behind artifact is announced as an UNFENCED duplicate of every conversation, and no on-disk artifact is ever unfenced — the rollback runs against an in-memory image:',
        why="the message was generated where cleanup fails and inherited that "
            "site's subject, announcing every left-behind copy as UNFENCED. "
            "Fence state is determined by whether the DDL committed, and a "
            "copy left by a refusal still carries its fence — an unfenced "
            "duplicate is a live exposure and a fenced one is litter",
    ),
    Mutation(
        pin="check_a_run_that_creates_nothing_reports_no_residue",
        module="hermes_cli/session_fence_rollback_cmd.py",
        find="    outcome = rollback.RollbackOutcome()\n    report = None\n",
        replace="    import tempfile\n\n"
                "    work_dir = Path(tempfile.mkdtemp(prefix=\"hermes-fence-preflight-\"))\n"
                "    outcome = rollback.RollbackOutcome()\n"
                "    outcome.note_residue(\n"
                "        rollback.describe_residue(\n"
                '            {"work_dir": str(work_dir), "files": 0, "error": ""},\n'
                '            obligation="the pre-flight working directory",\n'
                "            outcome=outcome,\n"
                "        ),\n"
                '        incident=f"work-dir:{work_dir}",\n'
                "    )\n    report = None\n",
        kills_by='the run reports something left behind, and it created nothing — it refused before it observed the source:',
        why="creating a working directory for a run that refuses before it "
            "observes the source manufactures an incident out of nothing, and "
            "the residue reason then overwrote the authority refusal that "
            "actually decided the run — a claim with no evidence behind it and "
            "a late fact erasing an earlier one, in one line",
    ),
    Mutation(
        pin="check_a_sidecar_that_cannot_be_released_is_not_a_successful_backup",
        module="hermes_cli/session_fence_rollback.py",
        find="        try:\n            os.unlink(member)\n"
             "        except OSError as exc:\n"
             '            return {"path": str(member), "files": 1,\n'
             '                    "error": f"{type(exc).__name__}: {exc}"}\n',
        replace="        try:\n            os.unlink(member)\n"
                "        except OSError:\n            pass\n",
        kills_by='a reservation could not be released and the boundary returned a backup report anyway:',
        why="swallowing the unlink failure is the defect: the reservation is "
            "still on disk and the run reports a verified, durable backup with "
            "a stale -wal beside it, which the next reader takes for that "
            "database's write-ahead log",
    ),
    Mutation(
        pin="check_a_foreign_file_at_a_reserved_sidecar_is_never_deleted",
        module="hermes_cli/session_fence_rollback.py",
        find="        if identity is not None and (info.st_dev, info.st_ino) != identity:\n"
             '            return {"path": str(member), "files": 1, "error": "ownership-lost"}\n',
        replace="",
        kills_by='the run deleted a file it did not create at a name it had merely reserved:',
        why="releasing a reserved name without checking what it resolves to "
            "now deletes a file this run never created. Nothing about the "
            "result would look wrong afterwards, and deletion has no way back",
    ),
    Mutation(
        pin="check_a_cleanup_failure_never_replaces_the_refusal_that_decided_the_run",
        module="hermes_cli/session_fence_rollback.py",
        find='                    incident=f"staging:{staging}",\n                )\n',
        replace='                    incident=f"staging:{staging}",\n                )\n'
                "                raise TurnFenceRollbackRefused(\n"
                '                    "the backup staging copy could not be removed",\n'
                '                    reason="backup-staging-residue",\n'
                "                )\n",
        kills_by='. The staging sweep is additive — it never becomes the reason the run failed',
        why="raising inside a finally discards the exception already in "
            "flight, so the refusal that decided the run is destroyed and a "
            "cleanup problem takes its place. The records stay correct and the "
            "primary reason contradicts them, which is the hardest kind of "
            "wrong report to notice",
    ),
    Mutation(
        pin="check_every_surviving_destination_member_is_reported_exactly_once",
        module="hermes_cli/session_fence_rollback.py",
        find="            for problem in reservation.remove_only_what_we_created():\n",
        replace="            reservation.remove_only_what_we_created()\n"
                "            for problem in []:\n",
        kills_by='. Two files remain on disk and the operator is told about a different set',
        why="the cleanup pass correctly works out which destination it could "
            "not remove and returns it; calling it for effect throws that away, "
            "so a surviving backup.db sits on disk unreported while the layer "
            "below has already established it is there. A return value "
            "describing a failure is a fact, and a fact nobody reads is not one",
    ),
    Mutation(
        pin="check_a_sidecar_that_vanished_is_not_claimed_as_our_removal",
        module="hermes_cli/session_fence_rollback.py",
        find="        except FileNotFoundError:\n"
             "            self.identities.pop(suffix, None)\n"
             "            return None\n",
        replace="        except FileNotFoundError:\n"
                '            return {"path": str(member), "error": "vanished"}\n',
        kills_by='the reservation is gone, which is the state the release wanted, and the run refused anyway:',
        why="a reservation that is already gone is the state the release "
            "wanted. Reporting it as an unresolved incident sends the operator "
            "to a directory to remove a file that is not there",
    ),
    Mutation(
        pin="check_a_completed_rollback_reports_the_surface_it_removed",
        module="hermes_cli/session_fence_rollback.py",
        find='            "verified": True,\n            "durable": True,\n',
        replace='            "verified": False,\n            "durable": False,\n',
        kills_by='no verified backup claimed:',
        why="a completion report that does not name the surface it removed "
            "leaves nothing in the output to distinguish a full rollback from "
            "a partial one, and the operator is back to checking by hand — "
            "which is the state this verb exists to replace",
    ),
    Mutation(
        pin="check_no_in_place_run_succeeds_and_each_wrong_target_names_its_own_reason",
        module="hermes_cli/session_fence_rollback.py",
        find='DISQUALIFICATION_REASONS = {\n'
             '    "namespace": "target-untrusted-namespace",\n'
             '    "canonical": "canonical-store-target",\n'
             '    "not-quiesced": "target-not-quiesced",\n'
             '    "unknown": "offline-authority-unknown",\n'
             '}',
        replace='DISQUALIFICATION_REASONS = {\n'
                '    "namespace": "offline-authority-unknown",\n'
                '    "canonical": "offline-authority-unknown",\n'
                '    "not-quiesced": "offline-authority-unknown",\n'
                '    "unknown": "offline-authority-unknown",\n'
                '}',

        kills_by='. Four wrong targets that print the same reason leave the operator with one next move for four different situations',
        why="four wrong targets that print the same reason leave the operator "
            "with one next move for four different situations. Every target "
            "is still refused, so nothing becomes unsafe — what is lost is "
            "the operator's ability to tell a hard-linked artifact from the "
            "live store from an interrupted detach",
    ),
    Mutation(
        pin="check_a_real_invocation_refuses_before_it_observes_the_source",
        module="hermes_cli/session_fence_rollback.py",
        find="    # governs has not been made in time.\n"
             "    disqualify_the_target(store_path)\n"
             "    establish_offline_authority(store_path)\n",
        replace="    # governs has not been made in time.\n"
                "    disqualify_the_target(store_path)\n"
                "    with BoundTarget(store_path):\n"
                "        establish_offline_authority(store_path)\n",
        kills_by='a real invocation opened the store with BoundTarget before refusing',
        why="the refusal still happens and still names itself — only the "
            "ORDER moves, so the store is opened before the decision that "
            "makes opening it unnecessary. On a WAL build that leaves -wal "
            "and -shm beside the artifact, and a refusal that changed the "
            "directory it is about to say it left alone is not a refusal",
    ),
    Mutation(
        pin="check_an_artifact_whose_image_is_incomplete_is_refused",
        module="hermes_cli/session_fence_rollback.py",
        find="    if beside:\n        raise TurnFenceRollbackRefused(\n",
        replace="    if False:\n        raise TurnFenceRollbackRefused(\n",
        kills_by='the preparer built a working object from an image that is missing committed rows; every later claim is about a different database',
        why="deserializing an image whose committed rows live in an "
            "uncheckpointed -wal produces a database that opens cleanly, "
            "passes integrity_check and is quietly short of data. Every later "
            "claim is then true of something nobody asked about",
    ),
    Mutation(
        pin="check_the_backup_describes_the_prepared_image_not_the_source_path",
        module="hermes_cli/session_fence_rollback.py",
        find='            conn.execute("VACUUM INTO ?", (str(snapshot),))\n',
        replace="            shutil.copyfile(store_path, snapshot)\n",
        kills_by='the rehearsal produced no backup:',
        why="a copy of the main file reproduces the bytes on disk, and "
            "committed state does not have to be there — a row committed into "
            "an uncheckpointed -wal is in the store and not in that file. The "
            "backup opens, passes integrity_check and restores a database "
            "that is missing it",
    ),
    Mutation(
        pin="check_a_partial_destination_collision_keeps_only_what_the_run_created",
        module="hermes_cli/session_fence_rollback.py",
        find="            for problem in reservation.remove_only_what_we_created():\n",
        replace="            for problem in []:\n",
        kills_by='. An operator who retries now hits a collision this run manufactured',
        why="the run still refuses, and still refuses for the right reason — "
            "what it stops doing is removing the half-built destination it "
            "created before hitting the occupied sibling. The operator who "
            "retries then collides with a file this run manufactured",
    ),
    Mutation(
        pin="check_the_backup_file_itself_is_flushed_to_the_platter",
        module="hermes_cli/session_fence_rollback.py",
        find="            writer.flush()\n        os.fsync(handle)\n",
        replace="            writer.flush()\n",
        kills_by='the backup file was never fsynced. It exists in the page cache and the rollback then removed the fence: flushed=',
        why="flush() pushes the bytes to the kernel and stops there. The "
            "backup then exists in the page cache while the fence comes off, "
            "which is precisely the crash it was taken against",
    ),
    Mutation(
        pin="check_the_backups_directory_entry_is_flushed_too",
        module="hermes_cli/session_fence_rollback.py",
        find="        reservation.release_the_sidecars(outcome=outcome)\n"
             "        _fsync_the_directory(backup_path.parent)\n",
        replace="        reservation.release_the_sidecars(outcome=outcome)\n",
        kills_by="the backup's parent directory was never fsynced, so the file's bytes are durable and its NAME is not. A crash here leaves a backup nothing can find: flushed=",
        why="the file's contents are durable and its NAME is not, so the "
            "crash leaves a backup nothing can find. The file fsync left in "
            "place does not cover it — they are different objects",
    ),
    Mutation(
        pin="check_a_fault_after_commit_never_reports_that_nothing_changed",
        module="hermes_cli/session_fence_rollback.py",
        find='            outcome.advance("commit-unknown")\n',
        replace="            pass\n",
        kills_by='a COMMIT that raised was resolved into a certainty the caller does not have:',
        why="a COMMIT that raised may have landed. Leaving the ledger at "
            "'committing' resolves that into whatever the reader assumes, and "
            "the assumption an operator makes about a run that reported a "
            "failure is that nothing happened",
    ),
    Mutation(
        pin="check_a_withdrawn_backup_is_never_reported_as_one_the_operator_has",
        module="hermes_cli/session_fence_rollback.py",
        find="        return _mark_the_backup_absent_but_not_by_us(outcome, backup_path)\n",
        replace="        return _mark_the_backup_withdrawn(outcome, backup_path)\n",
        kills_by='this run claims it removed a file it never called unlink on. ENOENT is not evidence of agency:',
        why="ENOENT answers only whether the file is there. Reading it as "
            "this run's own completed withdrawal claims an unlink and a "
            "directory flush that never happened, so a competitor's "
            "un-flushed removal is reported as durably ours",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    assert_mutation_kills_the_pin(mutation, str(_SELF), tmp_path, *_EXTRA_EXTRACT)


def _evaluates_to_none(expr) -> bool:
    """Can this expression evaluate to None? ``raise <None>`` is a TypeError."""
    if isinstance(expr, ast.Constant) and expr.value is None:
        return True
    if isinstance(expr, ast.IfExp):
        if isinstance(expr.test, ast.Constant):
            taken = expr.body if expr.test.value else expr.orelse
            return _evaluates_to_none(taken)
        return _evaluates_to_none(expr.body) or _evaluates_to_none(expr.orelse)
    return False


def _unreachable_statements(tree) -> list:
    """Statements following a terminator in the same block."""
    out = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            terminated = False
            for stmt in block:
                if terminated:
                    out.append(stmt.lineno)
                if isinstance(stmt, (ast.Raise, ast.Return, ast.Continue, ast.Break)):
                    terminated = True
    return out


def test_every_mutation_row_names_the_one_assertion_it_must_die_at():
    """STATIC half: every row has an anchor, and each names exactly one assertion.

    Presence, not behaviour. The runtime half is the row table itself, which
    runs each mutation and holds it to this anchor. Both are required and
    neither substitutes for the other: this one cannot tell you the row still
    kills, and the row table cannot tell you an anchor has gone ambiguous
    because a second assertion grew the same prose.
    """
    tree = ast.parse((REPO_ROOT / _SELF).read_text(encoding="utf-8"))
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    missing, ambiguous = [], []
    for mutation in SOURCE_MUTATIONS:
        if not mutation.kills_by:
            missing.append(mutation.pin)
            continue
        owners = [
            stmt for stmt in ast.walk(funcs[mutation.pin])
            if isinstance(stmt, ast.Assert) and stmt.msg is not None
            and any(
                isinstance(const, ast.Constant)
                and isinstance(const.value, str)
                and mutation.kills_by in const.value
                for const in ast.walk(stmt.msg)
            )
        ]
        if len(owners) != 1:
            ambiguous.append((mutation.pin, mutation.kills_by, len(owners)))
    assert not missing, (
        "rows with no kills_by, so they fall back to 'any AssertionError' and a "
        f"crash would score as a kill: {missing}"
    )
    assert not ambiguous, (
        "kills_by anchors that do not name exactly one assertion in their pin. "
        "An anchor matching two assertions cannot say which one fired, which is "
        f"the defect it exists to close: {ambiguous}"
    )


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_source_mutation_produces_the_behaviour_it_claims(mutation):
    """Apply the row and read the RESULT. Reading find/replace cannot see this.

    Two rows in this table were vacuous and both were found this way, not by
    review: one composed to ``raise None`` (a TypeError, so the pin died at its
    crash detector) and one reopened a read-only copy. A row is a TRANSFORMATION
    -- what it produces is a property of the composition, not of either half.
    """
    source = (REPO_ROOT / mutation.module).read_text(encoding="utf-8")
    occurrences = source.count(mutation.find)
    assert occurrences == 1, (
        f"{mutation.pin}: anchor matches {occurrences} places in "
        f"{mutation.module}, so it no longer names one guard"
    )
    mutated = source.replace(mutation.find, mutation.replace, 1)
    try:
        tree = ast.parse(mutated)
    except SyntaxError as exc:
        raise AssertionError(
            f"{mutation.pin}: the mutated module does not parse: {exc}"
        ) from exc

    bad_raises = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Raise) and node.exc is not None
        and _evaluates_to_none(node.exc)
    ]
    assert not bad_raises, (
        f"{mutation.pin}: the mutant raises an expression that evaluates to None "
        f"at line(s) {bad_raises}. `raise None` is a TypeError, so the pin dies "
        f"at its crash detector and the row measures nothing"
    )
    base_dead = len(_unreachable_statements(ast.parse(source)))
    dead = len(_unreachable_statements(tree))
    assert dead <= base_dead, (
        f"{mutation.pin}: the mutant introduces {dead - base_dead} unreachable "
        f"statement(s), so the behaviour its `why` describes may never execute"
    )


def test_a_crashing_mutation_cannot_score_as_a_kill(tmp_path):
    """NEGATIVE CONTROL: prove the discriminator REJECTS a crash.

    Deliberately NOT in SOURCE_MUTATIONS -- a row that must fail does not belong
    in the table the sweep validates. This is the historical vacuous form of
    row 6, kept because the defect is the permanent proof that the check works:
    it composes to `raise None`, the pin dies at `assert not run.crash`, and the
    old `"AssertionError" in trace` check scored that as a kill.
    """
    vacuous = Mutation(
        pin="check_no_in_place_run_succeeds_and_each_wrong_target_names_its_own_reason",
        module="hermes_cli/session_fence_rollback.py",
        find='        "database — so nothing available proves which store an artifact is or "\n'
             '        "that it is detached. Nothing was changed",\n'
             '        reason=DISQUALIFICATION_REASONS["unknown"],\n',
        replace='        "database", reason=DISQUALIFICATION_REASONS["unknown"],\n'
                "    ) if False else None\n    return None\n    raise TurnFenceRollbackRefused(\n"
                '        "unreachable",\n        reason=DISQUALIFICATION_REASONS["unknown"],\n',
        kills_by="the verb SUCCEEDED in place on the",
        why="negative control; must be rejected, never adopted",
    )
    with pytest.raises(AssertionError) as caught:
        assert_mutation_kills_the_pin(vacuous, str(_SELF), tmp_path, *_EXTRA_EXTRACT)
    assert "CRASH DETECTOR" in str(caught.value), (
        "the harness rejected the vacuous mutant, but not for being a crash. "
        f"The discriminator is not doing what this control exists to prove: "
        f"{caught.value}"
    )


def _mutation_rows_as_written(source: str) -> list:
    """Every row in ``SOURCE_MUTATIONS``, IN ORDER, as identity tuples.

    A LIST, not a dict keyed by pin. Five rows in this table are duplicates on
    four pins, so a pin key collapses 38 rows into 33 and the last one silently
    wins. That failure is silent by construction: no exception, no warning, just
    a collection smaller than the thing it represents, with every downstream
    number taken from the smaller set.

    Parsed from the tuple rather than diffed line by line: a unified diff shows
    changed LINES, so a row whose ``find`` moved by one character reports a hunk
    that may not contain the ``pin=`` line at all.
    """
    rows = []
    tree = ast.parse(source)
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "SOURCE_MUTATIONS" for t in node.targets)):
            continue
        for element in node.value.elts:
            fields = {}
            for keyword in element.keywords:
                try:
                    fields[keyword.arg] = ast.literal_eval(keyword.value)
                except (ValueError, SyntaxError):  # pragma: no cover
                    fields[keyword.arg] = "<unevaluated>"
            rows.append((
                fields.get("pin"),
                fields.get("module"),
                fields.get("find"),
                fields.get("replace"),
                # kills_by IS part of the row's meaning: it names the assertion
                # the row must die at. Omit it and a row whose only change is
                # its anchor is invisible here -- a blind spot exactly where the
                # new field lives.
                fields.get("kills_by"),
            ))
    return rows


def _changed_row_indices(before_source: str, after_source: str) -> set:
    """Indices into the AFTER table whose rows are new or changed.

    Multiset difference, so two identical-looking rows on one pin are two rows,
    and changing the FIRST of them selects the first -- not whichever happens to
    come last.
    """
    remaining = list(_mutation_rows_as_written(before_source))
    changed = set()
    for index, row in enumerate(_mutation_rows_as_written(after_source)):
        if row in remaining:
            remaining.remove(row)
        else:
            changed.add(index)
    return changed


def _rows_this_working_tree_added_or_changed():
    """Pins whose rows differ from HEAD's. ``None`` when git cannot say.

    Derived from the artifact, never from a name typed by hand. A pin name
    passed in by a person is one more thing to get wrong, and its failure mode
    is silent — you verify the wrong row and it passes.
    """
    from tests.state.test_turn_lease_generation_trigger import _git_dir

    git_dir = _git_dir()
    if git_dir is None:
        return None, None
    committed = subprocess.run(
        ["git", "-C", git_dir, "show", f"HEAD:{_SELF}"],
        capture_output=True, text=True,
    )
    if committed.returncode != 0:
        return None, None
    changed = _changed_row_indices(
        committed.stdout, (REPO_ROOT / _SELF).read_text(encoding="utf-8")
    )
    if not changed:
        # THE COMPARISON IS CHEAP AND THE REF IS NOT. Building a commit object
        # from a working tree this size costs over a minute, and paying it on
        # every clean run is how a check that must run before every commit stops
        # being run before every commit. Nothing changed, nothing to prove.
        return changed, None

    working = subprocess.run(
        ["git", "-C", git_dir, "stash", "create"], capture_output=True, text=True
    )
    # `stash create` builds a commit object from the working tree and prints
    # its sha. It pushes nothing onto the stash stack and moves no ref, so it
    # cannot disturb anyone else's work.
    return changed, (working.stdout or "").strip() or "HEAD"


def test_every_mutation_row_this_tree_adds_or_changes_actually_kills(tmp_path):
    """A row you just wrote is EARNED before it is committed. Not coverage.

    It proves nothing about the verb, and it does not say the property matters
    or that the pin observes the right thing — a row can kill for a reason that
    is not its own, and only walking what assertion fired in the mutated tree
    catches that.

    WHY IT EXISTS: my own verification of a NEW row was twice "the suite is
    green" rather than "this specific row kills", and twice a row that killed
    nothing survived into a head I reported. Running one row is a single tree
    extraction — seconds — so cost was never the obstacle. REMEMBERING was,
    which is the shape that should always have been a check instead.

    So nothing is asked of the author: the diff against HEAD decides which rows
    run. A clean tree runs none, which is correct — there is nothing new to
    earn. It measures against the WORKING TREE, because a row whose guard is
    also uncommitted cannot be measured against HEAD; it would report a stale
    anchor rather than an unproven row, which is a different complaint and a
    misleading one.
    """
    changed, ref = _rows_this_working_tree_added_or_changed()
    if changed is None:
        pytest.skip("no git repository to compare the mutation table against")
    if not changed:
        return

    # THE KEY MUST BE AT LEAST AS FINE AS THE UNIT BEING COUNTED, and the count
    # of the collection must be asserted against the count of the source. This
    # one line fails immediately on 33 vs 38.
    written = _mutation_rows_as_written((REPO_ROOT / _SELF).read_text(encoding="utf-8"))
    assert len(written) == len(SOURCE_MUTATIONS), (
        f"the parsed table has {len(written)} rows and SOURCE_MUTATIONS has "
        f"{len(SOURCE_MUTATIONS)}. Some rows are not being seen by the change "
        f"detector, so changing them would run nothing"
    )
    for position, index in enumerate(sorted(changed)):
        assert index < len(SOURCE_MUTATIONS), (
            f"changed row {index} is past the end of SOURCE_MUTATIONS"
        )
        into = tmp_path / f"new{position}"
        into.mkdir(parents=True, exist_ok=True)
        assert_mutation_kills_the_pin(
            SOURCE_MUTATIONS[index], str(_SELF), into, *_EXTRA_EXTRACT, ref=ref,
        )


def test_a_changed_row_is_selected_by_position_not_by_pin():
    """Change the FIRST of two same-pin rows and the FIRST must be selected.

    With a pin-keyed detector the LAST row per pin wins, and in most edits that
    happens to be a changed row too -- so the bug is invisible unless the changed
    row is deliberately not the last. That is what this constructs.
    """
    before = (
        'SOURCE_MUTATIONS = (\n'
        '    Mutation(pin="p", module="m", find="A", replace="x", kills_by="k"),\n'
        '    Mutation(pin="p", module="m", find="B", replace="y", kills_by="k"),\n'
        ')\n'
    )
    first_changed = before.replace('replace="x"', 'replace="CHANGED"')
    last_changed = before.replace('replace="y"', 'replace="CHANGED"')
    assert _changed_row_indices(before, first_changed) == {0}, (
        "changing the FIRST of two rows on one pin did not select row 0; the "
        "detector is still collapsing rows that share a pin"
    )
    assert _changed_row_indices(before, last_changed) == {1}
    assert _changed_row_indices(before, before) == set()


def test_every_mutation_anchor_still_names_one_place():
    """Every anchor matches exactly one place in the CURRENT source. Fast.

    NOT COVERAGE — this proves nothing about the verb. It is a tooling guard,
    and it exists because of a loop that cost three rounds: a commit moves a
    production guard, the row naming that guard silently stops matching, and
    nothing says so until the next full mutation matrix runs. By then two more
    commits have moved more code, so the head is never green and complete at
    the same instant.

    The matrix already asserts this, once per row, behind three subprocess
    tree-extractions each — minutes. Here it is a string count over two files,
    which makes it runnable before every commit, and that is the whole point:
    the delay was the defect, not the missing check.

    A commit that moves a guard and leaves its row stale is incomplete by
    construction. This is what says so at the time rather than a round later.
    """
    import collections

    sources = {}
    counts = collections.Counter()
    stale = []
    for mutation in SOURCE_MUTATIONS:
        path = REPO_ROOT / mutation.module
        text = sources.setdefault(mutation.module, path.read_text(encoding="utf-8"))
        found = text.count(mutation.find)
        counts[found] += 1
        if found != 1:
            stale.append((mutation.pin, mutation.module, found, mutation.find))
    assert not stale, (
        "mutation anchors that no longer name exactly one guard:\n"
        + "\n".join(
            f"  {pin} -> {module}: {found} match(es)\n    {find!r}"
            for pin, module, found, find in stale
        )
        + "\n\nRe-derive each in the SAME commit that moved the guard. A row "
        "pointing at replaced code scores no kill and still reports as a row."
    )


def test_every_pin_has_a_mutation_that_kills_it():
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)
