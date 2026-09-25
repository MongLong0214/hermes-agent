"""The lineage stamp only moves forward, and a settled fenced store opens read-first.

A settled open that took the write lock or ran DDL would block behind every sibling's write
transaction and bump the schema cookie each open; a stamp that moved backward would hand the
store back to a build this one has fenced out.
"""

from __future__ import annotations

import contextlib
import re
import sqlite3
import threading
import time

from hermes_state_fence import FENCE_LINEAGE_BASE, FORK_BASE_UPSTREAM_GATE, STORED_SCHEMA_VERSION
from tests.hermes_state.fence_store_probe import expected_fences, fence_triggers, schema_cookie, stored
from tests.hermes_state.fork_store_fixture import build_store, isolate_home

_WRITE_STATEMENT = re.compile(
    r"\s*(BEGIN\s+IMMEDIATE|INSERT|UPDATE|DELETE|REPLACE|ALTER|DROP|CREATE\s+(?!.*\bIF\s+NOT\s+EXISTS\b))",
    re.I | re.S,
)
_FORK_GATE_STAMP = FENCE_LINEAGE_BASE + FORK_BASE_UPSTREAM_GATE


def _migrated_fork_store(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fork29")
    SessionDB(db_path=db_path).close()
    assert stored(db_path) == STORED_SCHEMA_VERSION
    return db_path


def _defer_the_fts_step(patch) -> None:
    from hermes_state_schema import SessionSchemaMixin

    # The step's own "defer" answer (a trigram layout this runtime cannot inspect yet).
    patch.setattr(SessionSchemaMixin, "_migrate_trigram_cron_exclusion", lambda self, cursor: False)


def test_a_settled_open_runs_no_ddl_and_no_write(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    db_path = _migrated_fork_store(tmp_path, monkeypatch)
    cookie = schema_cookie(db_path)
    statements = []
    real_connect = sqlite3.connect

    def tracing_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", tracing_connect)
        SessionDB(db_path=db_path).close()

    assert schema_cookie(db_path) == cookie
    # The FTS5 capability probe on ``temp`` is the one exempt write.
    assert [s for s in statements if _WRITE_STATEMENT.match(s) and "temp." not in s] == []


def test_a_settled_open_does_not_wait_on_a_sibling_write_lock(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    db_path = _migrated_fork_store(tmp_path, monkeypatch)
    holder = sqlite3.connect(db_path, timeout=60)
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("UPDATE state_meta SET value = value WHERE key = 'nonexistent'")
    release = threading.Timer(4.0, holder.rollback)
    release.start()
    try:
        started = time.perf_counter()
        SessionDB(db_path=db_path).close()
        elapsed = time.perf_counter() - started
    finally:
        release.cancel()
        with contextlib.suppress(sqlite3.Error):
            holder.rollback()
        holder.close()
    assert elapsed < 2.0, f"settled open blocked on the write lock for {elapsed:.3f}s"


def test_optimize_storage_leaves_a_fenced_stamp_where_it_is(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    db_path = _migrated_fork_store(tmp_path, monkeypatch)
    db = SessionDB(db_path=db_path)
    try:
        result = db.optimize_fts_storage(vacuum=False)
    finally:
        db.close()

    assert result["ok"] is True, result
    assert stored(db_path) == STORED_SCHEMA_VERSION


def test_a_deferred_fts_step_holds_the_stamp_until_the_open_that_completes_it(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fork29")
    with monkeypatch.context() as patch:
        _defer_the_fts_step(patch)
        SessionDB(db_path=db_path).close()
        assert stored(db_path) == _FORK_GATE_STAMP
        SessionDB(db_path=db_path).close()
        assert stored(db_path) == _FORK_GATE_STAMP
    # The gate stamp already fences the store out of the fork.
    assert fence_triggers(db_path) == expected_fences(db_path, STORED_SCHEMA_VERSION)

    SessionDB(db_path=db_path).close()
    assert stored(db_path) == STORED_SCHEMA_VERSION


def test_optimize_storage_advances_a_gate_stamp_once_the_fts_work_is_done(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fork29")
    with monkeypatch.context() as patch:
        _defer_the_fts_step(patch)
        db = SessionDB(db_path=db_path)
        try:
            assert stored(db_path) == _FORK_GATE_STAMP
            result = db.optimize_fts_storage(vacuum=False)
        finally:
            db.close()

    assert result["ok"] is True, result
    assert stored(db_path) == STORED_SCHEMA_VERSION
