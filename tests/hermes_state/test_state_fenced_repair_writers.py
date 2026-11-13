"""The repair-path writers of state.db on a store this build fenced at 1033.

``hermes_state_repair._connect_repair_durable`` is the opener behind ``hermes doctor``'s write-health
probe (``_db_opens_cleanly``: a rolled-back INSERT into ``sessions`` and ``messages``), the schema
repair and ``hermes sessions repair``'s declarative reconcile. Those writes hit governed tables, so the
connection carries the fence UDF behind SessionDB's per-connection serialization; without it a healthy
fenced store reads as FTS corruption and every repair fails. The browse catch-up behind
``web_server_sessions`` is DDL only and passes the fences unchanged.
"""

from __future__ import annotations

import sqlite3

import pytest

import hermes_state_fence as fence
from hermes_state import SessionDB
from tests.hermes_state.test_state_fence_migration import _fence_sql, _rows, fork_1030_store


@pytest.fixture
def migrated(tmp_path):
    path = tmp_path / "home" / "state.db"
    rows = fork_1030_store(path)
    SessionDB(db_path=path).close()
    assert fence.probe_store_lineage(path).settled
    return path, rows, _state(path)


def _state(path):
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return _fence_sql(conn), _rows(conn)
    finally:
        conn.close()


def _assert_fences_and_rows_kept(path, rows, before) -> None:
    fences, now = _state(path)
    assert fences == before[0] and now == rows == before[1]
    assert fence.probe_store_lineage(path).settled


def test_repair_connections_are_serialized_and_carry_the_fence_udf(migrated):
    from hermes_state_repair import _connect_repair_durable

    path, _, _ = migrated
    conn = _connect_repair_durable(path)
    try:
        assert isinstance(conn, fence.SerializedConnectionMixin)
        assert conn.execute(f"SELECT {fence.TURN_FENCE_FUNCTION}()").fetchone()[0] == fence.TURN_FENCE_GENERATION
    finally:
        conn.close()


def test_the_write_health_probe_passes_a_healthy_fenced_store(migrated):
    from hermes_state_repair import _db_opens_cleanly

    path, rows, before = migrated
    assert _db_opens_cleanly(path) is None
    _assert_fences_and_rows_kept(path, rows, before)


@pytest.mark.parametrize("held", [False, True], ids=["in_place", "snapshot"])
def test_doctor_write_health_passes_a_healthy_fenced_store(migrated, monkeypatch, held):
    import hermes_state_repair
    from hermes_cli.doctor_state import _write_health_reason

    path, rows, before = migrated
    # A store a live writer holds is probed on a read-only snapshot copy, through the same opener.
    monkeypatch.setattr(hermes_state_repair, "_live_writer_holds_db", lambda db_path: held)
    assert _write_health_reason(path, should_fix=False) is None
    _assert_fences_and_rows_kept(path, rows, before)


def test_schema_repair_finds_a_fenced_store_healthy_without_surgery(migrated):
    from hermes_state_repair import repair_state_db_schema

    path, rows, before = migrated
    report = repair_state_db_schema(path)
    assert (report["repaired"], report["strategy"], report["error"], report["backup_path"]) == (
        True, "already_healthy", None, None)
    _assert_fences_and_rows_kept(path, rows, before)


def _drop_declared_table(path) -> None:
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.execute("DROP TABLE gateway_heartbeats")  # not governed: no fence goes with it
    finally:
        conn.close()


def test_sessions_repair_reconciles_drift_on_a_fenced_store(migrated):
    from hermes_state_reconcile import repair_declarative_schema, schema_drift_reason

    path, rows, before = migrated
    _drop_declared_table(path)
    assert schema_drift_reason(path) is not None

    report = repair_declarative_schema(path)

    assert (report["repaired"], report["strategy"], report["error"]) == (True, "reconcile_declarative_schema", None)
    assert schema_drift_reason(path) is None
    _assert_fences_and_rows_kept(path, rows, before)


def test_the_browse_catch_up_is_ddl_only_and_keeps_the_fences(migrated):
    from hermes_state_reconcile import reconcile_drift_for_browse, schema_drift_reason

    path, rows, before = migrated
    _drop_declared_table(path)

    assert reconcile_drift_for_browse(path) is True
    assert schema_drift_reason(path) is None
    _assert_fences_and_rows_kept(path, rows, before)
