"""A store whose lineage another build owns is migrated only while nothing holds its profile's
gateway runtime lock.

The migration swaps the fences and restamps the lineage in place, which locks the build that wrote
the store out of every governed table. A gateway of that build may still be running on the profile,
so while any other process holds ``<HERMES_HOME>/gateway.lock`` the open is refused and the store
must come out byte-identical. A store this build already fenced (settled, or parked at a gate stamp
while its FTS step defers) is not another build's, so a live lock holder must not refuse it; nor
is the process that holds the lock itself (the gateway that owns it, or a sibling open mid-migration).
A lease whose lock file is replaced mid-open (a stale-pid cleanup unlinks it, a gateway claims a
new one) no longer excludes that gateway, so it must not admit the migration either.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from hermes_state_fence import FENCE_LINEAGE_BASE, FORK_BASE_UPSTREAM_GATE, STORED_SCHEMA_VERSION
from tests.hermes_state.fence_store_probe import expected_fences, fence_triggers, stored
from tests.hermes_state.fork_store_fixture import (
    _connect_as, build_store, file_fingerprint, isolate_home, refence,
)

_REPO = Path(__file__).resolve().parents[2]
_OLDER_FENCED_STAMP = STORED_SCHEMA_VERSION - 1
# A real gateway process of any build: it takes the same OS lock on the same file.
_HOLDER = (
    "import sys\n"
    "from gateway.status import acquire_gateway_runtime_lock\n"
    "assert acquire_gateway_runtime_lock()\n"
    "print('held', flush=True)\n"
    "sys.stdin.read()\n"
)


@contextlib.contextmanager
def _gateway_holding_runtime_lock(hermes_home: Path):
    child = subprocess.Popen(
        [sys.executable, "-c", _HOLDER], cwd=_REPO, env={**os.environ, "HERMES_HOME": str(hermes_home)},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout.readline().strip() == "held"
        yield child
    finally:
        child.stdin.close()  # EOF ends the child; the kernel drops its lock with it
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


@contextlib.contextmanager
def _open_paused_at(monkeypatch, db_path: Path, step: str):
    """A SessionDB open of *db_path* in a thread, held at ``SessionSchemaMixin.<step>`` (so inside
    its admission lease); yields ``finish()``, which lets it go and returns its error, or None."""
    from hermes_state import SessionDB
    from hermes_state_schema import SessionSchemaMixin

    paused, resume = threading.Event(), threading.Event()
    real_step = getattr(SessionSchemaMixin, step)

    def pause(self, *args):
        if not paused.is_set():
            paused.set()
            assert resume.wait(30)
        return real_step(self, *args)

    outcome = []

    def open_store():
        try:
            SessionDB(db_path=db_path).close()
            outcome.append(None)
        except BaseException as exc:
            outcome.append(exc)

    def finish():
        resume.set()
        opener.join(30)
        return outcome[0]

    monkeypatch.setattr(SessionSchemaMixin, step, pause)
    opener = threading.Thread(target=open_store)
    opener.start()
    try:
        assert paused.wait(30)
        yield finish
    finally:
        resume.set()
        opener.join(30)
        monkeypatch.setattr(SessionSchemaMixin, step, real_step)


def _foreign_store(db_path: Path, kind: str) -> Path:
    if kind != "fenced_older":
        return build_store(db_path, kind)
    build_store(db_path, "fork29")
    conn = _connect_as(db_path, 29)
    try:
        refence(conn, _OLDER_FENCED_STAMP)
        conn.execute("UPDATE schema_version SET version = ?", (_OLDER_FENCED_STAMP,))
    finally:
        conn.close()
    return db_path


@pytest.mark.parametrize("kind", ["fork29", "upstream29", "fenced_older"])
def test_a_store_another_build_owns_is_left_untouched_while_its_gateway_holds_the_runtime_lock(
        tmp_path, monkeypatch, kind):
    from hermes_state import SessionDB, classify_persistence_error

    hermes_home = isolate_home(tmp_path, monkeypatch)
    db_path = _foreign_store(hermes_home / "state.db", kind)
    before = file_fingerprint(db_path)

    with _gateway_holding_runtime_lock(hermes_home):
        with pytest.raises(RuntimeError) as refused:
            SessionDB(db_path=db_path)
        assert getattr(refused.value, "code", None) == "STATE_DB_FORWARD_MIGRATION_ADMISSION_BLOCKED"
        assert file_fingerprint(db_path) == before
        # The owner's remedy is to stop that gateway, whether the refusal is live or read back as text.
        assert classify_persistence_error(refused.value) == "locked"
        assert classify_persistence_error(f"{type(refused.value).__name__}: {refused.value}") == "locked"

    SessionDB(db_path=db_path).close()
    assert stored(db_path) == STORED_SCHEMA_VERSION
    assert fence_triggers(db_path) == expected_fences(db_path, STORED_SCHEMA_VERSION)


@pytest.mark.parametrize("stamp", ["gate", "settled"])
def test_a_store_this_build_already_fenced_opens_while_a_gateway_holds_the_runtime_lock(
        tmp_path, monkeypatch, stamp):
    from hermes_state import SessionDB
    from hermes_state_schema import SessionSchemaMixin

    hermes_home = isolate_home(tmp_path, monkeypatch)
    db_path = build_store(hermes_home / "state.db", "fork29")
    with monkeypatch.context() as patch:
        if stamp == "gate":
            # The trigram step's own "defer" answer parks the stamp at the fork's upstream gate.
            patch.setattr(SessionSchemaMixin, "_migrate_trigram_cron_exclusion", lambda self, cursor: False)
        SessionDB(db_path=db_path).close()
    parked = FENCE_LINEAGE_BASE + FORK_BASE_UPSTREAM_GATE if stamp == "gate" else STORED_SCHEMA_VERSION
    assert stored(db_path) == parked

    with _gateway_holding_runtime_lock(hermes_home):
        SessionDB(db_path=db_path).close()
        assert stored(db_path) == STORED_SCHEMA_VERSION
    assert fence_triggers(db_path) == expected_fences(db_path, STORED_SCHEMA_VERSION)


def test_two_opens_in_one_process_do_not_refuse_each_other_mid_migration(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from hermes_state_schema import SessionSchemaMixin

    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fork29")
    paused, resume = threading.Event(), threading.Event()
    real_reconcile = SessionSchemaMixin._reconcile_columns

    def first_open_pauses(self, cursor):
        if not paused.is_set():
            paused.set()
            assert resume.wait(30)
        return real_reconcile(self, cursor)

    monkeypatch.setattr(SessionSchemaMixin, "_reconcile_columns", first_open_pauses)
    first_failure = []

    def first_open():
        try:
            SessionDB(db_path=db_path).close()
        except BaseException as exc:
            first_failure.append(exc)

    first = threading.Thread(target=first_open)
    first.start()
    try:
        assert paused.wait(30)
        # The first open holds this store's admission while it migrates; a sibling open in the
        # same process is not another build's gateway.
        SessionDB(db_path=db_path).close()
    finally:
        resume.set()
        first.join(30)
    assert first_failure == []
    assert stored(db_path) == STORED_SCHEMA_VERSION
    from gateway.status import is_gateway_runtime_lock_active

    assert not is_gateway_runtime_lock_active(db_path.parent / "gateway.lock")  # no lease outlives its open


@pytest.mark.parametrize("lost", ["before_first_ddl", "under_write_lock"])
def test_a_lease_whose_lock_file_was_replaced_does_not_admit_the_migration(tmp_path, monkeypatch, lost):
    from hermes_state import SessionDB

    hermes_home = isolate_home(tmp_path, monkeypatch)
    # Lost under the write lock, the open's own column DDL has already run: a store in this
    # build's schema (another build's lineage) is one that DDL leaves byte-identical.
    kind, step = ("fork29", "_init_schema") if lost == "before_first_ddl" else ("upstream29", "_reconcile_columns")
    db_path = build_store(hermes_home / "state.db", kind)
    before = file_fingerprint(db_path)

    with _open_paused_at(monkeypatch, db_path, step) as finish:
        # A stale-pid cleanup unlinks the lock file the lease holds (it carries no gateway record),
        # and a gateway of the build that owns the store claims a new one.
        (hermes_home / "gateway.lock").unlink()
        with _gateway_holding_runtime_lock(hermes_home):
            refused = finish()
    assert getattr(refused, "code", None) == "STATE_DB_FORWARD_MIGRATION_ADMISSION_BLOCKED"
    assert file_fingerprint(db_path) == before
    SessionDB(db_path=db_path).close()
    assert stored(db_path) == STORED_SCHEMA_VERSION


def test_a_status_poll_during_the_migration_does_not_refuse_it(tmp_path, monkeypatch):
    from gateway.status import get_running_pid, is_gateway_runtime_lock_active

    hermes_home = isolate_home(tmp_path, monkeypatch)
    db_path = build_store(hermes_home / "state.db", "fork29")

    with _open_paused_at(monkeypatch, db_path, "_reconcile_columns") as finish:
        # `hermes status` reads the lease's held lock with no gateway record as stale and unlinks
        # it; with no gateway claiming the path, the migration goes on under a lease on a new file.
        assert get_running_pid() is None
        assert finish() is None
    assert stored(db_path) == STORED_SCHEMA_VERSION
    assert not is_gateway_runtime_lock_active(hermes_home / "gateway.lock")  # no lease outlives its open


def test_the_gateway_that_owns_the_runtime_lock_migrates_its_own_store(tmp_path, monkeypatch):
    import hermes_state_registry
    from gateway import status

    real_home = isolate_home(tmp_path, monkeypatch)
    # The gateway may name its home through a symlink while the shared registry opens the store by
    # its resolved path; the lock the gateway owns is still this store's.
    alias = tmp_path / "hermes_home_alias"
    alias.symlink_to(real_home, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(alias))
    db_path = build_store(alias / "state.db", "fork29")

    assert status.acquire_gateway_runtime_lock()
    try:
        record = (real_home / "gateway.lock").read_bytes()
        db = hermes_state_registry.acquire(db_path)
        assert db.db_path.parent != alias  # two spellings of one lock file
        hermes_state_registry.release(db)
        assert status.owns_gateway_runtime_lock()
        assert (real_home / "gateway.lock").read_bytes() == record
    finally:
        status.release_gateway_runtime_lock()
    assert stored(db_path) == STORED_SCHEMA_VERSION
    assert fence_triggers(db_path) == expected_fences(db_path, STORED_SCHEMA_VERSION)
