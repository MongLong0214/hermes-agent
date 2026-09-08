"""``hermes sessions fence-rollback`` — runtime tests for its reachable contract.

WHAT THIS VERB DOES TODAY
    It refuses. Every invocation, with or without ``--dry-run``, before it
    opens, copies, or writes to the store it was given. See
    ``hermes_cli/session_fence_rollback.py`` and
    ``hermes_cli/session_fence_rollback_cmd.py`` for the full contract.

WHY THIS FILE IS SMALL
    An earlier revision pinned a ~2,600-line backup/commit engine that sat
    entirely behind ``establish_offline_authority`` — unreachable from this
    verb, since that function refuses unconditionally. That pin apparatus
    (``PINS``, ``SOURCE_MUTATIONS``, the source-mutation harness) is gone;
    the properties it guarded are asserted here directly, as ordinary
    runtime behaviour, table-driven over the shapes Sol specified:

    1. parser/dispatch, including explicit ``--store`` and ``--backup``;
    2. refusal happens before the source is opened, read, or written, in
       both real and ``--dry-run`` modes;
    3. source bytes and the sidecar file set are unchanged; no backup or
       work directory is created;
    4. the JSON carries exact produced-facts and a stable per-target
       refusal reason.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import pathlib
import sqlite3
from argparse import Namespace
from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class VerbRun:
    """One ``hermes sessions fence-rollback`` invocation, reduced to values."""

    rc: object
    stdout: str
    stderr: str
    crash: str


def _run_verb(store, backup, *, dry_run=False, work_dir=None) -> VerbRun:
    """Drive the verb THROUGH ``cmd_sessions``, so the dispatch wiring is
    under test too — not just the handler it eventually reaches.
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


def _build_parser() -> argparse.ArgumentParser:
    from hermes_cli.session_fence_rollback_cmd import add_fence_rollback_parser

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="sessions_action")
    add_fence_rollback_parser(subparsers)
    return parser


def _parse(argv):
    parser = _build_parser()
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            return parser.parse_args(argv), None
    except SystemExit:
        return None, err.getvalue()


def _make_store(tmp_path) -> pathlib.Path:
    """A plain, on-disk SQLite file with no sidecars beside it."""
    path = tmp_path / "store" / "state.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO sessions (id) VALUES ('seed')")
        conn.commit()
    finally:
        conn.close()
    return path


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sidecar_names(path):
    return sorted(
        p.name
        for p in path.parent.iterdir()
        if p.name != path.name
    )


# ---------------------------------------------------------------------------
# 1. Parser/dispatch, including explicit --store and --backup.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv,should_error",
    [
        ([], True),  # neither --store nor --backup
        (["--store", "/tmp/x.db"], True),  # --backup missing
        (["--backup", "/tmp/x.bak"], True),  # --store missing
        (["--store", "/tmp/x.db", "--backup", "/tmp/x.bak"], False),
        (
            ["--store", "/tmp/x.db", "--backup", "/tmp/x.bak", "--dry-run"],
            False,
        ),
        (
            [
                "--store", "/tmp/x.db", "--backup", "/tmp/x.bak",
                "--work-dir", "/tmp/work",
            ],
            False,
        ),
    ],
)
def test_parser_requires_store_and_backup_explicitly(argv, should_error):
    full_argv = ["fence-rollback", *argv]
    namespace, error_output = _parse(full_argv)
    if should_error:
        assert namespace is None, f"expected a parse error for {argv!r}"
        assert error_output
    else:
        assert namespace is not None, f"expected {argv!r} to parse: {error_output}"
        assert namespace.store == pathlib.Path("/tmp/x.db")
        assert namespace.backup == pathlib.Path("/tmp/x.bak")


def test_parser_dry_run_defaults_false_and_work_dir_defaults_none():
    namespace, error_output = _parse(
        ["fence-rollback", "--store", "/tmp/x.db", "--backup", "/tmp/x.bak"]
    )
    assert namespace is not None, error_output
    assert namespace.dry_run is False
    assert namespace.work_dir is None


def test_dispatch_reaches_the_verb_through_cmd_sessions(tmp_path):
    """`cmd_sessions` with action="fence-rollback" reaches the verb, not a
    subcommand-help fallthrough — the operator surface is the dispatch.
    """
    store = _make_store(tmp_path)
    run = _run_verb(str(store), str(tmp_path / "backup.db"))
    assert not run.crash, run.crash
    payload = _payload(run)
    assert payload is not None, f"no JSON on stdout: {run.stdout!r}"
    assert payload["verb"] == "sessions fence-rollback"


# ---------------------------------------------------------------------------
# 2 & 3. Refusal happens before the source is touched, in both real and
#         dry-run modes; source bytes and the sidecar file set are
#         unchanged; no backup or work directory is created.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dry_run", [False, True], ids=["real", "dry-run"])
def test_refusal_leaves_the_store_byte_identical_and_creates_nothing(
    tmp_path, dry_run
):
    store = _make_store(tmp_path)
    backup = tmp_path / "backup.db"
    work_dir = tmp_path / "work"

    before_digest = _digest(store)
    before_sidecars = _sidecar_names(store)
    before_mtime_ns = store.stat().st_mtime_ns

    run = _run_verb(
        str(store), str(backup), dry_run=dry_run, work_dir=str(work_dir)
    )

    assert not run.crash, run.crash
    payload = _payload(run)
    assert payload is not None
    assert payload["ok"] is False
    assert run.rc == 1

    # THE SOURCE WAS NEVER OPENED, LET ALONE WRITTEN TO.
    assert _digest(store) == before_digest
    assert store.stat().st_mtime_ns == before_mtime_ns
    assert _sidecar_names(store) == before_sidecars

    # NO BACKUP AND NO WORK DIRECTORY.
    assert not backup.exists(), "a backup was created on a refused run"
    assert not work_dir.exists(), "a working directory was created on a refused run"


@pytest.mark.parametrize("dry_run", [False, True], ids=["real", "dry-run"])
def test_refusal_on_a_missing_store_never_creates_it(tmp_path, dry_run):
    """Naming a store that does not exist must not bring one into being —
    the refusal is decided from the pathname before anything is opened.
    """
    store = tmp_path / "does-not-exist" / "state.db"
    backup = tmp_path / "backup.db"

    run = _run_verb(str(store), str(backup), dry_run=dry_run)

    assert not run.crash, run.crash
    payload = _payload(run)
    assert payload["ok"] is False
    assert not store.exists()
    assert not store.parent.exists()
    assert not backup.exists()


# ---------------------------------------------------------------------------
# 4. The JSON carries exact produced-facts and a stable per-target refusal
#    reason.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dry_run", [False, True], ids=["real", "dry-run"])
def test_refusal_reason_is_stable_and_matches_the_target_shape(tmp_path, dry_run):
    """Different target shapes refuse for different, STABLE reasons — the
    reason names which precondition stopped the run, not just that one did.
    """
    missing = tmp_path / "missing.db"
    run_missing = _run_verb(str(missing), str(tmp_path / "b1.db"), dry_run=dry_run)
    assert _payload(run_missing)["refused"]["reason"] == "store-missing"

    present = _make_store(tmp_path)
    run_present = _run_verb(str(present), str(tmp_path / "b2.db"), dry_run=dry_run)
    assert (
        _payload(run_present)["refused"]["reason"] == "offline-authority-unknown"
    )

    sidecar_store = _make_store(tmp_path / "sidecar-case")
    (sidecar_store.parent / (sidecar_store.name + "-wal")).write_bytes(b"")
    run_sidecar = _run_verb(
        str(sidecar_store), str(tmp_path / "b3.db"), dry_run=dry_run
    )
    assert _payload(run_sidecar)["refused"]["reason"] == "target-not-quiesced"


def test_refused_payload_carries_only_the_fields_the_run_actually_produced(
    tmp_path,
):
    """A refusal that never reached a pre-flight, a backup step, or a
    commit must not render those fields at all — absent and ``false`` are
    different statements, and this checks the exact key set on the payload.
    """
    store = _make_store(tmp_path)
    run = _run_verb(str(store), str(tmp_path / "backup.db"))
    payload = _payload(run)

    assert set(payload) == {
        "verb", "ok", "dry_run", "changed", "outcome", "store", "generation",
        "refused",
    }
    assert payload["changed"] is False
    assert payload["outcome"] == "not-started"
    assert payload["dry_run"] is False
    assert set(payload["refused"]) == {"reason", "detail"}
    assert payload["refused"]["reason"] == "offline-authority-unknown"
    assert payload["store"] == str(store)
