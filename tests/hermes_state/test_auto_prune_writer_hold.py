"""Automatic prune must not hold the shared writer for the whole sweep.

The gateway runs ``maybe_auto_prune_and_vacuum`` on its housekeeping thread against the SAME shared
SessionDB its Telegram turns write through. A prune that deletes every expired session (and its
messages, through the FTS triggers) in one ``BEGIN IMMEDIATE`` keeps ``SessionDB._lock`` and the
SQLite write lock for the whole sweep; a reply's transcript write waits for all of it. On a
multi-GB store that is a reply stall measured in minutes.

Real store, real SQLite: a temp trigger makes each session delete slow, the way a large message
fan-out through the FTS triggers is, and a transcript write from another thread must land while the
sweep is still deleting — not after its last row.
"""

from __future__ import annotations

import threading
import time

import hermes_state_maintenance

_EXPIRED = 200
_PER_DELETE_S = 0.002
_WRITE_BOUND_S = 2.0


def _seed_expired(db, count: int) -> None:
    old = time.time() - 400 * 86400

    def _do(conn):
        conn.executemany(
            "INSERT INTO sessions (id, source, started_at, ended_at, end_reason) VALUES (?, 'cli', ?, ?, 'done')",
            [(f"old-{i}", old, old + 1) for i in range(count)])
    db._execute_write(_do)


def test_transcript_write_lands_while_auto_prune_is_still_deleting(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    # A small per-transaction budget so 200 empty sessions span several transactions; the production
    # budget is sized for real message fan-out. Absent on base, where the prune is one transaction.
    monkeypatch.setattr(hermes_state_maintenance, "_AUTO_PRUNE_BATCH_ROWS", 20, raising=False)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _seed_expired(db, _EXPIRED)
        db.create_session("live", "telegram")

        deleting = threading.Event()
        deleted: list = []

        def _slow_delete():
            deleted.append(1)
            deleting.set()
            time.sleep(_PER_DELETE_S)
            return 0

        # Registered on the writer connection the prune runs on; TEMP so the store is untouched.
        db._conn.create_function("_test_slow_delete", 0, _slow_delete)
        db._conn.execute(
            "CREATE TEMP TRIGGER _test_slow_session_delete BEFORE DELETE ON main.sessions "
            "BEGIN SELECT _test_slow_delete(); END")

        result: dict = {}
        maintenance = threading.Thread(target=lambda: result.update(db.maybe_auto_prune_and_vacuum(
            retention_days=90, min_interval_hours=0, vacuum=False)))
        maintenance.start()
        assert deleting.wait(10.0), "the prune never reached its delete"

        started = time.monotonic()
        db.append_message("live", "user", "reply-path write during maintenance")
        waited = time.monotonic() - started
        deleted_before_write = len(deleted)
        maintenance.join(60.0)

        assert result.get("pruned") == _EXPIRED, result
        assert deleted_before_write < _EXPIRED, (
            "the transcript write waited until the prune had deleted every expired session")
        assert waited < _WRITE_BOUND_S, (
            f"transcript write waited {waited:.2f}s behind automatic prune (bound {_WRITE_BOUND_S}s)")
        assert [m["content"] for m in db.get_messages("live")] == ["reply-path write during maintenance"]
    finally:
        db.close()
