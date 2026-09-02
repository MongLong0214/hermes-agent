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

from tools import async_delegation as ad


def _point_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    return ad


def _track_connections(monkeypatch):
    """Observe closes without discarding connect_tracked's admitted factory."""
    opened, closed = [], []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        factory = kwargs.pop("factory", sqlite3.Connection)

        class TrackingConnection(factory):
            def close(self):
                closed.append(id(self))
                return super().close()

        conn = real_connect(*args, factory=TrackingConnection, **kwargs)
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
    opened, closed = _track_connections(monkeypatch)

    def fail_schema_init(_conn):
        raise sqlite3.OperationalError("simulated schema init failure")

    monkeypatch.setattr(ad, "_initialize_schema", fail_schema_init)

    with pytest.raises(sqlite3.OperationalError):
        ad._connect()

    assert opened, "expected bootstrap and ledger connections"
    assert len(opened) == len(closed)
    assert set(opened) == set(closed)
