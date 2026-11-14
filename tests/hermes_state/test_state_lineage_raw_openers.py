"""Raw state.db openers: lineage admission before they connect, migrate-first for writers.

``gateway.delivery_ledger._connect``, ``tools.async_delegation._connect``, the hosted-room stores and
``gateway.hosted_room_attachments`` open the shared state.db through ``hermes_cli.sqlite_util.open_db``,
whose ``initialize`` creates their tables. A store this build cannot migrate is refused before any of
that DDL, with its bytes and table set unchanged. The fork's 1030 store is handed to a writable
SessionDB open first, which migrates it once under the profile's ``gateway.lock`` lease; the raw writer
then writes through the 1033 fences with the fence UDF on a serialized connection. A raw reader never
migrates: it refuses the 1030 store and leaves it byte-identical.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

import pytest

import hermes_state_fence as fence
from hermes_cli import sqlite_util
from tests.hermes_state.test_state_fence_migration import _fence_sql, _rows, fork_1030_store
from tests.hermes_state.test_state_lineage_admission import REFUSED, assert_untouched, snapshot

_JOIN_S = 60.0


def _tables(path: Path) -> set:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        conn.close()


def _readback(path: Path):
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return conn, _fence_sql(conn)
    except BaseException:
        conn.close()
        raise


def _count_migrations(monkeypatch) -> list:
    """Every migration body that ran, as its result (True: this opener migrated the store)."""
    results, real = [], fence._migrate_locked

    def counting(db, cursor):
        migrated = real(db, cursor)
        results.append(migrated)
        return migrated

    monkeypatch.setattr(fence, "_migrate_locked", counting)
    return results


# ── Refused before connect: shapes no open of this build migrates ──

UNMIGRATABLE = ("fenced_1031", "fenced_1030_fence_missing")


@pytest.fixture(params=UNMIGRATABLE)
def unmigratable(request, tmp_path):
    return REFUSED[request.param][0](tmp_path / "store" / "state.db")


def _assert_refused(caught) -> None:
    assert type(caught.value).__name__ == "IncompatibleSchemaError"
    assert caught.value.cause == "FENCE_GENERATION_MISMATCH"
    assert not isinstance(caught.value, sqlite3.DatabaseError)


def test_delivery_ledger_connect_refuses_an_unmigratable_store(monkeypatch, unmigratable):
    from gateway import delivery_ledger

    monkeypatch.setattr(delivery_ledger, "_db_path", lambda: unmigratable)
    tables, before = _tables(unmigratable), snapshot(unmigratable)

    with pytest.raises(RuntimeError) as caught:
        delivery_ledger._connect()

    _assert_refused(caught)
    assert_untouched(unmigratable, before)
    assert "delivery_obligations" not in tables and _tables(unmigratable) == tables


def test_async_delegation_connect_refuses_an_unmigratable_store(monkeypatch, unmigratable):
    from tools import async_delegation

    monkeypatch.setattr(async_delegation, "_db_path", lambda: unmigratable)
    # async_delegation tightens the main file to 0600 before it reaches open_db (its own hardening,
    # untouched here), so the store starts at 0600 and only the refusal is under test.
    os.chmod(unmigratable, 0o600)
    tables, before = _tables(unmigratable), snapshot(unmigratable)

    with pytest.raises(RuntimeError) as caught:
        async_delegation._connect()

    _assert_refused(caught)
    assert_untouched(unmigratable, before)
    assert _tables(unmigratable) == tables


def test_hosted_room_attachments_refuse_an_unmigratable_store(unmigratable):
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    from tests.gateway.test_hosted_room_attachments import _put

    tables, before = _tables(unmigratable), snapshot(unmigratable)

    with pytest.raises(RuntimeError) as caught:
        _put(HostedRoomAttachmentStore(unmigratable))

    _assert_refused(caught)
    # The store prepares its own private blob root beside state.db before it connects; state.db
    # itself and its sidecars are untouched.
    before["listing"] = sorted(before["listing"] + ["hosted-room-attachments"])
    assert_untouched(unmigratable, before)
    assert "hosted_room_attachments" not in tables and _tables(unmigratable) == tables


def test_open_db_refuses_an_unmigratable_state_db_and_leaves_other_names_alone(tmp_path, unmigratable):
    created = []

    def initialize(conn):
        conn.execute("CREATE TABLE IF NOT EXISTS opener_table (id INTEGER PRIMARY KEY)")
        created.append(True)

    before = snapshot(unmigratable)
    with pytest.raises(RuntimeError) as caught:
        sqlite_util.open_db(unmigratable, db_label="state.db", initialize=initialize)
    _assert_refused(caught)
    assert not created
    assert_untouched(unmigratable, before)

    # The gate keys on the shared store's name only: other databases open as before.
    other = REFUSED["fenced_1030"][0](tmp_path / "other" / "kanban.db")
    conn = sqlite_util.open_db(other, db_label="kanban.db", wal=False, initialize=initialize)
    conn.close()
    assert created and "opener_table" in _tables(other)


# ── The raw writers ──

def _ledger_write(path, monkeypatch, n):
    from gateway import delivery_ledger

    monkeypatch.setattr(delivery_ledger, "_db_path", lambda: path)
    delivery_ledger.record_obligation(obligation_id=f"ob-{n}", session_key="k", platform="p", chat_id="c",
                                      thread_id=None, content="hello")
    return "SELECT COUNT(*) FROM delivery_obligations"


def _delegation_write(path, monkeypatch, n):
    from tools import async_delegation

    monkeypatch.setattr(async_delegation, "_db_path", lambda: path)
    # async_delegations is governed: this INSERT runs the fence trigger at 1033.
    async_delegation._persist_dispatch({"delegation_id": f"d{n}", "session_key": "k", "dispatched_at": 1.0})
    return "SELECT COUNT(*) FROM async_delegations"


def _attachment_write(path, monkeypatch, n):
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    from tests.gateway.test_hosted_room_attachments import _put

    _put(HostedRoomAttachmentStore(path), upload_id=f"upload-{n}")
    return "SELECT COUNT(*) FROM hosted_room_attachments"


def _open_db_governed_write(path, monkeypatch, n):
    conn = sqlite_util.open_db(path, db_label="state.db")
    try:
        with conn:  # sessions is governed: the raw connection's write runs the fence trigger
            conn.execute("INSERT INTO sessions (id, source, started_at) VALUES (?, 'raw', 1.0)", (f"raw-{n}",))
    finally:
        conn.close()
    return "SELECT COUNT(*) FROM sessions WHERE source = 'raw'"


WRITERS = {
    "delivery_ledger": _ledger_write,
    "async_delegation": _delegation_write,
    "hosted_room_attachments": _attachment_write,
    "open_db": _open_db_governed_write,
}


def _assert_kept(conn, rows) -> None:
    now = _rows(conn)
    assert now["messages"] == rows["messages"]
    assert set(rows["sessions"]) <= set(now["sessions"])


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_each_raw_writer_writes_to_a_migrated_store_and_keeps_its_fences(monkeypatch, tmp_path, writer):
    from hermes_state import SessionDB

    path = tmp_path / "home" / "state.db"
    rows = fork_1030_store(path)
    SessionDB(db_path=path).close()  # migrated by the writable SessionDB open
    conn, fences = _readback(path)
    conn.close()
    migrations = _count_migrations(monkeypatch)

    count_sql = WRITERS[writer](path, monkeypatch, 1)

    assert migrations == []
    assert fence.probe_store_lineage(path).settled
    conn, after = _readback(path)
    try:
        assert after == fences and len(after) == 3 * len(fence.ALL_GOVERNED_TABLES) - 3  # no reservations table
        assert conn.execute(count_sql).fetchone()[0] == 1
        _assert_kept(conn, rows)
    finally:
        conn.close()
    raw = sqlite_util.open_db(path, db_label="state.db")
    try:
        assert isinstance(raw, fence.SerializedConnectionMixin)
        assert raw.execute(f"SELECT {fence.TURN_FENCE_FUNCTION}()").fetchone()[0] == fence.TURN_FENCE_GENERATION
    finally:
        raw.close()


# ── Migrate-first: a writable raw open of the fork's 1030 store ──

@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_a_raw_writer_on_the_fork_store_migrates_it_exactly_once_then_writes(monkeypatch, tmp_path, writer):
    path = tmp_path / "home" / "state.db"
    rows = fork_1030_store(path)
    migrations = _count_migrations(monkeypatch)

    count_sql = WRITERS[writer](path, monkeypatch, 1)
    WRITERS[writer](path, monkeypatch, 2)  # the store is settled now: no second migration

    assert migrations == [True]
    assert fence.probe_store_lineage(path).settled
    conn, fences = _readback(path)
    try:
        assert set(fences.values()) == set(fence.turn_fence_trigger_definitions(
            fence.CORE_GOVERNED_TABLES + ("session_process_authorities",)
            + fence.TARGET_GOVERNED_TABLES).values())
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == fence.STORED_SCHEMA_VERSION
        assert conn.execute(count_sql).fetchone()[0] == 2
        _assert_kept(conn, rows)
    finally:
        conn.close()


def test_a_raw_writer_refuses_the_fork_store_when_the_writable_open_did_not_migrate_it(monkeypatch, tmp_path):
    import hermes_state

    class NoMigration:  # a writable open that returned without migrating
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    path = tmp_path / "home" / "state.db"
    fork_1030_store(path)
    monkeypatch.setattr(hermes_state, "SessionDB", NoMigration)
    created, before = [], snapshot(path)

    with pytest.raises(RuntimeError) as caught:
        sqlite_util.open_db(path, db_label="state.db", initialize=lambda conn: created.append(True))

    _assert_refused(caught)
    assert fence.MIGRATION_NEEDS_WRITABLE_OPEN in str(caught.value)
    assert not created
    assert_untouched(path, before)


# ── Readers never migrate ──

def test_a_raw_read_only_open_of_the_fork_store_is_refused_byte_identical(monkeypatch, tmp_path):
    from gateway.hosted_rooms_common import open_sqlite
    from hermes_state import SessionDB

    path = tmp_path / "home" / "state.db"
    fork_1030_store(path)
    migrations = _count_migrations(monkeypatch)
    before = snapshot(path)

    # The hosted-room steady-state reader opens state.db through open_db, without schema work.
    with pytest.raises(RuntimeError) as caught:
        open_sqlite(path)
    _assert_refused(caught)
    assert fence.MIGRATION_NEEDS_WRITABLE_OPEN in str(caught.value)
    with pytest.raises(RuntimeError) as caught:
        SessionDB(db_path=path, read_only=True)
    _assert_refused(caught)

    assert migrations == []
    assert_untouched(path, before)
    assert fence.probe_store_lineage(path).migratable


# ── Concurrency ──

def test_a_raw_writer_racing_a_session_db_open_migrates_the_fork_store_once(monkeypatch, tmp_path):
    from gateway import delivery_ledger, runtime_ownership
    from hermes_state import SessionDB

    path = tmp_path / "home" / "state.db"
    rows = fork_1030_store(path)
    monkeypatch.setattr(delivery_ledger, "_db_path", lambda: path)
    results, errors = [], []
    contended = threading.Event()
    real_migrate, real_owner = fence._migrate_locked, runtime_ownership.live_runtime_owner

    def migrate_after_contention(db, cursor):
        contended.wait(timeout=10)  # the other opener has met this one's lease
        migrated = real_migrate(db, cursor)
        results.append(migrated)
        return migrated

    def noting_owner(home):
        contended.set()
        return real_owner(home)

    monkeypatch.setattr(fence, "_migrate_locked", migrate_after_contention)
    monkeypatch.setattr(runtime_ownership, "live_runtime_owner", noting_owner)
    start = threading.Barrier(2)

    def raw_writer():
        start.wait(timeout=_JOIN_S)
        delivery_ledger.record_obligation(obligation_id="ob-race", session_key="k", platform="p", chat_id="c",
                                          thread_id=None, content="hello")

    def session_db():
        start.wait(timeout=_JOIN_S)
        SessionDB(db_path=path).close()

    def guarded(body):
        def run():
            try:
                body()
            except BaseException as exc:  # reported below
                errors.append(exc)
        return run

    threads = [threading.Thread(target=guarded(body), daemon=True) for body in (raw_writer, session_db)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=_JOIN_S)
    assert not [t for t in threads if t.is_alive()], "an opener hung"
    assert errors == []
    # The loser met the winner's lease, then found the store settled and never entered the migration.
    assert contended.is_set() and results == [True]
    assert fence.probe_store_lineage(path).settled
    conn, _ = _readback(path)
    try:
        assert conn.execute("SELECT obligation_id FROM delivery_obligations").fetchall() == [("ob-race",)]
        _assert_kept(conn, rows)
    finally:
        conn.close()
