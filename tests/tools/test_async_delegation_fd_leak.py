"""Regression: the async-delegation ledger must close every SQLite connection.

Sibling of the cron execution-ledger leak (#69567 / PR #69594). The durable
delegation ledger used ``with _connect() as conn:`` where the connection
context manager commits/rolls back but never closes, leaking the db/-wal/-shm
file descriptors on every dispatch, completion, and delivery-claim. These tests
fail if the deterministic ``close()`` is ever removed again.
"""

import queue
import sqlite3

import pytest

from hermes_state import SessionDB
from tools import async_delegation as ad


def _recording_factory(factory, closed_ids):
    """The caller's connection factory, subclassed to record close() calls.

    A subclass, not a delegating wrapper: the tracked read-only lineage probe that
    runs before every state.db open retags the connection's class, which only a
    real sqlite3.Connection instance supports.
    """

    class _RecordingConnection(factory):
        def close(self):
            closed_ids.append(id(self))
            super().close()

    return _RecordingConnection


def _point_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    return ad


def _track_connections(monkeypatch):
    opened, closed = [], []
    real_connect = sqlite3.connect

    def tracking_connect(*args, factory=sqlite3.Connection, **kwargs):
        conn = real_connect(*args, factory=_recording_factory(factory, closed), **kwargs)
        opened.append(id(conn))
        return conn

    monkeypatch.setattr(ad.sqlite3, "connect", tracking_connect)
    return opened, closed


def test_ledger_operations_close_every_connection(monkeypatch, tmp_path):
    """Public durable-ledger reads/writes must close every connection opened."""
    _point_ledger(monkeypatch, tmp_path)
    opened, closed = _track_connections(monkeypatch)

    ad.get_durable_delegation("nope")
    ad.recover_abandoned_delegations()
    ad.restore_undelivered_completions(queue.Queue())
    ad.mark_completion_delivered("nope")
    ad.claim_completion_delivery("nope", "claim-1")

    assert opened, "expected at least one connection to be opened"
    assert len(opened) == len(closed)
    assert set(opened) == set(closed)


def test_schema_init_failure_still_closes_connection(monkeypatch, tmp_path):
    """A PRAGMA/DDL failure after connect() must still close the connection."""
    _point_ledger(monkeypatch, tmp_path)
    # An existing store, so the ledger's own connection replays the schema rather
    # than the SessionDB bootstrap a fresh store gets.
    SessionDB(db_path=tmp_path / "state.db").close()
    opened, closed = [], []
    real_connect = sqlite3.connect

    def tracking_connect(*args, factory=sqlite3.Connection, **kwargs):
        class _FailingSchemaConnection(_recording_factory(factory, closed)):
            def execute(self, sql, *args, **kwargs):
                if "CREATE TABLE" in sql:
                    raise sqlite3.OperationalError("simulated schema init failure")
                return super().execute(sql, *args, **kwargs)

            def executescript(self, script):
                # The schema initializer replays the full canonical schema as one
                # large multi-statement script (single durable-shape authority, #94691).
                if len(script) > 1000:
                    raise sqlite3.OperationalError("simulated schema init failure")
                return super().executescript(script)

        conn = real_connect(*args, factory=_FailingSchemaConnection, **kwargs)
        opened.append(id(conn))
        return conn

    monkeypatch.setattr(ad.sqlite3, "connect", tracking_connect)

    with pytest.raises(sqlite3.OperationalError, match="simulated schema init failure"):
        with ad._transaction():
            pass

    assert opened
    assert len(opened) == len(closed)
    assert set(opened) == set(closed)
