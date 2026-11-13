"""Raw state.db openers refuse an unadmitted lineage before they connect.

``gateway.delivery_ledger._connect`` and ``tools.async_delegation._connect`` open the shared state.db
through ``hermes_cli.sqlite_util.open_db``, whose ``initialize`` creates their tables. On a fenced
store that DDL would land unfenced (DDL is never fenced), so ``open_db`` decodes the lineage first
and a refused store keeps its bytes and its table set. A raw opener never migrates: the fork's 1030
store stays refused until a writable SessionDB open migrates it, and on the migrated store the raw
writers' connections carry the fence UDF and SessionDB's per-connection serialization.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import sqlite_util
from tests.hermes_state.test_state_lineage_admission import assert_untouched, make_store, snapshot


def _tables(path: Path) -> set:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        conn.close()


@pytest.fixture
def fenced_1030(tmp_path):
    return make_store(tmp_path / "store" / "state.db")


def _assert_refused(caught) -> None:
    assert type(caught.value).__name__ == "IncompatibleSchemaError"
    assert caught.value.cause == "FENCE_GENERATION_MISMATCH"
    assert not isinstance(caught.value, sqlite3.DatabaseError)


def test_delivery_ledger_connect_refuses_a_fenced_store(monkeypatch, fenced_1030):
    from gateway import delivery_ledger

    monkeypatch.setattr(delivery_ledger, "_db_path", lambda: fenced_1030)
    tables, before = _tables(fenced_1030), snapshot(fenced_1030)

    with pytest.raises(RuntimeError) as caught:
        delivery_ledger._connect()

    _assert_refused(caught)
    assert_untouched(fenced_1030, before)
    assert "delivery_obligations" not in tables and _tables(fenced_1030) == tables


def test_async_delegation_connect_refuses_a_fenced_store(monkeypatch, fenced_1030):
    from tools import async_delegation

    monkeypatch.setattr(async_delegation, "_db_path", lambda: fenced_1030)
    # async_delegation tightens the main file to 0600 before it reaches open_db (its own hardening,
    # untouched here), so the store starts at 0600 and only the refusal is under test.
    os.chmod(fenced_1030, 0o600)
    tables, before = _tables(fenced_1030), snapshot(fenced_1030)

    with pytest.raises(RuntimeError) as caught:
        async_delegation._connect()

    _assert_refused(caught)
    assert_untouched(fenced_1030, before)
    assert _tables(fenced_1030) == tables


def test_open_db_refuses_a_fenced_state_db_and_leaves_other_names_alone(tmp_path, fenced_1030):
    created = []

    def initialize(conn):
        conn.execute("CREATE TABLE IF NOT EXISTS opener_table (id INTEGER PRIMARY KEY)")
        created.append(True)

    before = snapshot(fenced_1030)
    with pytest.raises(RuntimeError) as caught:
        sqlite_util.open_db(fenced_1030, db_label="state.db", initialize=initialize)
    _assert_refused(caught)
    assert not created
    assert_untouched(fenced_1030, before)

    # The gate keys on the shared store's name only: other databases open as before.
    other = make_store(tmp_path / "other" / "kanban.db")
    conn = sqlite_util.open_db(other, db_label="kanban.db", wal=False, initialize=initialize)
    conn.close()
    assert created and "opener_table" in _tables(other)


# ── A store this build migrated: raw writers register the fence UDF and write through the fences ──

def test_both_raw_writers_write_to_a_migrated_store(monkeypatch, tmp_path):
    from gateway import delivery_ledger
    from hermes_state import SessionDB
    import hermes_state_fence as fence
    from tests.hermes_state.test_state_fence_migration import fork_1030_store
    from tools import async_delegation

    path = tmp_path / "home" / "state.db"
    fork_1030_store(path)
    SessionDB(db_path=path).close()  # the writable open migrates it; raw openers never do
    monkeypatch.setattr(delivery_ledger, "_db_path", lambda: path)
    monkeypatch.setattr(async_delegation, "_db_path", lambda: path)

    delivery_ledger.record_obligation(obligation_id="ob-1", session_key="k", platform="p", chat_id="c",
                                      thread_id=None, content="hello")
    # async_delegations is governed: this INSERT runs the fence trigger at 1033.
    async_delegation._persist_dispatch({"delegation_id": "d1", "session_key": "k", "dispatched_at": 1.0})

    conn = sqlite_util.open_db(path, db_label="state.db")
    try:
        assert isinstance(conn, fence.SerializedConnectionMixin)
        assert conn.execute("SELECT obligation_id FROM delivery_obligations").fetchall()[0][0] == "ob-1"
        assert conn.execute("SELECT delegation_id FROM async_delegations").fetchall()[0][0] == "d1"
    finally:
        conn.close()
    assert fence.probe_store_lineage(path).settled
