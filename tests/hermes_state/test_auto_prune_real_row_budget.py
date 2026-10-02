"""Automatic prune's per-batch budget must count REAL stored message rows, not ``message_count``,
and must sub-batch a single session that alone exceeds the budget.

``message_count`` is maintained as the ACTIVE count only: compaction and rewind
(``replace_messages(..., archive_dropped=True)``) soft-archive dropped turns (``active = 0``) and
reset ``message_count`` to the kept/active count, but the archived rows stay in ``messages`` and
``prune_sessions`` deletes them too (no ``active = 1`` filter on the DELETE). Budgeting batches
from ``message_count`` therefore undercounts a session with real archived history — three sessions
with 3,000 stored messages and ``message_count == 0`` all landed in one nominally-small batch, the
whole sweep running under ``SessionDB._lock`` (the no-timeout writer mutex a reply's transcript
write also needs) in a single transaction. The same undercount let one oversized session's own
delete run un-split regardless of budget.

Real store, real SQLite, real ``replace_messages``: a slow-delete trigger on ``messages`` (not a
mock of prune logic) makes the proof about lock duration, not about row counts in isolation — a
transcript write from another thread must land before the whole sweep's deletes finish, not after
the last one (mirrors ``test_auto_prune_writer_hold.py``).
"""

from __future__ import annotations

import threading
import time

import hermes_state_maintenance

_PER_DELETE_S = 0.005
_WRITE_BOUND_S = 2.0
# Batch sizes below are intentionally small: this host's per-row trigger callback (Python-to-
# SQLite round trip, not just the explicit sleep) costs tens of ms, so a single BATCH must stay
# well under _WRITE_BOUND_S on its own for the bound to mean anything on a shared/loaded runner.


def _mark_old_and_ended(db, session_id: str, old: float) -> None:
    # Recency is freshest-of(last_activity_at, MAX(messages.timestamp), started_at) — the message
    # rows this test seeds carry real (now) timestamps, so they must be backdated too or the
    # session reads as freshly active regardless of started_at/ended_at.
    db._execute_write(lambda conn: (
        conn.execute("UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'done', "
                     "last_activity_at = ? WHERE id = ?", (old, old + 1, old, session_id)),
        conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = ?", (old, session_id))))


def _install_slow_message_delete(db) -> tuple:
    """TEMP trigger + Python callback: every row DELETEd from ``messages`` costs _PER_DELETE_S,
    the way a large message fan-out through the FTS triggers costs real time in production.
    Returns (deleted_count_list, deleting_event) like ``test_auto_prune_writer_hold.py``."""
    deleted: list = []
    deleting = threading.Event()

    def _slow_delete():
        deleted.append(1)
        deleting.set()
        time.sleep(_PER_DELETE_S)
        return 0

    db._conn.create_function("_test_slow_msg_delete", 0, _slow_delete)
    db._conn.execute(
        "CREATE TEMP TRIGGER _test_slow_msg_delete BEFORE DELETE ON main.messages "
        "BEGIN SELECT _test_slow_msg_delete(); END")
    return deleted, deleting


def _run_prune_and_time_concurrent_write(db, deleted: list, deleting: threading.Event):
    """Runs maybe_auto_prune_and_vacuum in a thread; returns (result, deleted_before_write, waited)."""
    result: dict = {}
    maintenance = threading.Thread(target=lambda: result.update(
        db.maybe_auto_prune_and_vacuum(retention_days=90, min_interval_hours=0, vacuum=False)))
    maintenance.start()
    assert deleting.wait(10.0), "the prune never reached its first delete"

    started = time.monotonic()
    db.append_message("live", "user", "reply-path write during archived-history prune")
    waited = time.monotonic() - started
    deleted_before_write = len(deleted)
    maintenance.join(60.0)
    return result, deleted_before_write, waited


def test_prune_batches_by_real_row_count_not_active_message_count(tmp_path, monkeypatch):
    """Three sessions store 3,000 real archived message rows but ``message_count == 0`` each
    (post-compaction shape, via ``replace_messages(archive_dropped=True)``). A small per-batch
    budget must still split the sweep across several transactions."""
    from hermes_state import SessionDB

    monkeypatch.setattr(hermes_state_maintenance, "_AUTO_PRUNE_BATCH_ROWS", 20, raising=False)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        old = time.time() - 400 * 86400
        per_session = 12
        for i in range(3):
            sid = f"old-{i}"
            db.create_session(sid, source="cli")
            db.append_messages_batch(sid, [{"role": "user", "content": f"m{j}"} for j in range(per_session)])
            # compaction/rewind shape: archive the live rows, message_count resets to the (now empty) active set.
            db.replace_messages(sid, [], archive_dropped=True)
            assert db.get_session(sid)["message_count"] == 0
            _mark_old_and_ended(db, sid, old)
        total_rows = 3 * per_session
        db.create_session("live", "telegram")

        deleted, deleting = _install_slow_message_delete(db)
        result, deleted_before_write, waited = _run_prune_and_time_concurrent_write(db, deleted, deleting)

        assert result.get("pruned") == 3, result
        assert deleted_before_write < total_rows, (
            "the transcript write waited until every archived row across all 3 sessions was deleted "
            f"({deleted_before_write}/{total_rows}) — message_count undercounted the real row budget")
        assert waited < _WRITE_BOUND_S, f"transcript write waited {waited:.2f}s (bound {_WRITE_BOUND_S}s)"
    finally:
        db.close()


def test_prune_splits_one_oversized_session_across_sub_batches(tmp_path, monkeypatch):
    """A single expired session whose own messages alone exceed the batch budget must still be
    deleted in sub-batches, releasing the writer lock between them — not as one un-split delete."""
    from hermes_state import SessionDB

    monkeypatch.setattr(hermes_state_maintenance, "_AUTO_PRUNE_BATCH_ROWS", 15, raising=False)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        old = time.time() - 400 * 86400
        total_rows = 40
        db.create_session("huge", source="cli")
        db.append_messages_batch("huge", [{"role": "user", "content": f"m{j}"} for j in range(total_rows)])
        _mark_old_and_ended(db, "huge", old)
        db.create_session("live", "telegram")

        deleted, deleting = _install_slow_message_delete(db)
        result, deleted_before_write, waited = _run_prune_and_time_concurrent_write(db, deleted, deleting)

        assert result.get("pruned") == 1, result
        assert deleted_before_write < total_rows, (
            "the transcript write waited until the whole oversized session's "
            f"{total_rows} messages were deleted ({deleted_before_write}/{total_rows}) in one un-split delete")
        assert waited < _WRITE_BOUND_S, f"transcript write waited {waited:.2f}s (bound {_WRITE_BOUND_S}s)"
        assert db.get_session("huge") is None
    finally:
        db.close()


def test_an_oversized_session_after_a_small_one_never_shares_its_transaction(tmp_path, monkeypatch):
    """Batching must look at a session's OWN stored rows before adding it to a batch in progress: a
    small expired session followed by an oversized one used to land both in one batch, so the
    oversized session's whole history went out in a single un-split transaction (one small + one
    3,000-message session = one 3,001-message DELETE). No write transaction may delete more than the
    budget's worth of messages, however the candidates are ordered."""
    from hermes_state import SessionDB

    budget = 15
    monkeypatch.setattr(hermes_state_maintenance, "_AUTO_PRUNE_BATCH_ROWS", budget, raising=False)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        old = time.time() - 400 * 86400
        sizes = {"small-0": 1, "huge-0": 40, "small-1": 2, "huge-1": 40, "small-2": 1}
        for sid, count in sizes.items():
            db.create_session(sid, source="cli")
            db.append_messages_batch(sid, [{"role": "user", "content": f"m{j}"} for j in range(count)])
            _mark_old_and_ended(db, sid, old)

        deleted: list = []
        db._conn.create_function("_test_count_msg_delete", 0, lambda: deleted.append(1) or 0)
        db._conn.execute(
            "CREATE TEMP TRIGGER _test_count_msg_delete BEFORE DELETE ON main.messages "
            "BEGIN SELECT _test_count_msg_delete(); END")
        per_transaction: list = []
        real_write = db._execute_write

        def _write(fn, *args, **kwargs):
            before = len(deleted)
            try:
                return real_write(fn, *args, **kwargs)
            finally:
                per_transaction.append(len(deleted) - before)

        monkeypatch.setattr(db, "_execute_write", _write)
        result = db.maybe_auto_prune_and_vacuum(retention_days=90, min_interval_hours=0, vacuum=False)

        assert result.get("pruned") == len(sizes), result
        assert len(deleted) == sum(sizes.values())
        assert max(per_transaction) <= budget, (
            f"one write transaction deleted {max(per_transaction)} messages (budget {budget}): "
            f"an oversized session shared a batch with a smaller one — {per_transaction}")
    finally:
        db.close()
