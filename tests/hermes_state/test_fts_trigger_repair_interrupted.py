"""An open-time FTS trigger repair survives being cut off in its rebuild.

When the sync triggers are missing (for example after a no-FTS5 runtime dropped them), the open
recreates them and refills the indexes with FTS5 ``'rebuild'``. If the recreated triggers committed
before the rebuild did, a process killed during the rebuild left them in place over an index missing
every row written while they were gone. The next open saw the triggers and repaired nothing. A writer
running during the repair also committed rows into that half-built family.

These tests check behaviour, not how the repair is written:
- a write attempted while the repair's rebuild runs does not commit;
- after the next open, every row carrying a token is found by both indexes (MATCH count == LIKE count);
- a second opener that arrives while that open repairs does not run the repair again;
- that opener neither waits for the repair nor fails: it opens, its write commits, and every index
  ends up holding that write's row;
- an opener that leaves the repair to another process still finds every row, and once that process
  lets go its housekeeping retry repairs the indexes and search uses them again.
"""

import contextlib
import sqlite3
import threading

import pytest

import hermes_state_schema
from hermes_state import SessionDB
from hermes_state_common import fts_rebuild_admission
from hermes_state_fence import register_turn_fence_generation
from tests.hermes_state.fork_store_fixture import isolate_home

NEEDLE = "zephyrquill"
INDEXED_ROWS = 60
DETACHED_ROWS = 7
# Progress callbacks (one per VM step) the rebuild runs before it is cut off: it has started, not finished.
_STEPS_BEFORE_ABORT = 50
_BASE_TRIGGERS = ("messages_fts_insert", "messages_fts_delete", "messages_fts_update")
_TRIGRAM_TRIGGERS = ("messages_fts_trigram_insert", "messages_fts_trigram_delete", "messages_fts_trigram_update")


def _store_written_while_detached(tmp_path, monkeypatch, dropped, backfill_rows):
    db_path = isolate_home(tmp_path, monkeypatch) / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("needles", source="cli")
        for i in range(INDEXED_ROWS):
            db.append_message("needles", role="user", content=f"{NEEDLE} indexed row {i}")
        if backfill_rows:
            # An optimize-storage backfill short of its last rows: the sync triggers leave ids in
            # (fts_rebuild_progress, fts_rebuild_high_water] to it.
            last = db._conn.execute("SELECT MAX(id) FROM messages").fetchone()[0]
            db._conn.executemany(
                "INSERT OR REPLACE INTO state_meta (key, value) VALUES (?, ?)",
                [("fts_rebuild_progress", str(last)), ("fts_rebuild_high_water", str(last + backfill_rows))],
            )
            for i in range(backfill_rows):
                db.append_message("needles", role="user", content=f"{NEEDLE} left to the backfill {i}")
        for trigger in dropped:
            db._conn.execute(f"DROP TRIGGER {trigger}")
        for i in range(DETACHED_ROWS):
            db.append_message("needles", role="user", content=f"{NEEDLE} written while detached {i}")
    finally:
        db.close()
    return db_path


def _counts(db):
    """(rows messages_fts finds, rows messages_fts_trigram finds, rows that hold the token)."""
    base = db._conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?", (NEEDLE,)).fetchone()[0]
    trigram = db._conn.execute(
        "SELECT COUNT(*) FROM messages_fts_trigram WHERE messages_fts_trigram MATCH ?", (f'"{NEEDLE}"',),
    ).fetchone()[0]
    stored = db._conn.execute("SELECT COUNT(*) FROM messages WHERE content LIKE ?", (f"%{NEEDLE}%",)).fetchone()[0]
    return base, trigram, stored


def _connect_cutting_off_the_repair_rebuild(real_connect, db_path, concurrent_writes, table):
    """``sqlite3.connect`` whose connections abort *table*'s 'rebuild' once it is under way
    (SQLITE_INTERRUPT from the progress handler), as a stand-in for a kill. As that rebuild starts,
    another connection tries to write a message without waiting; ``concurrent_writes`` records whether it committed."""

    def write_from_another_connection():
        other = real_connect(str(db_path), timeout=0, isolation_level=None)
        try:
            register_turn_fence_generation(other)
            other.execute("BEGIN IMMEDIATE")
            other.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES ('needles', 'user', ?, 1.0)",
                (f"{NEEDLE} written during the repair",),
            )
            other.execute("COMMIT")
            concurrent_writes.append("committed")
        except sqlite3.OperationalError as exc:
            concurrent_writes.append(f"refused: {exc}")
        finally:
            other.close()

    def connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        budget = []

        def trace(sql):
            if not budget and f"INTO {table}({table}) VALUES('rebuild')" in sql:
                budget.append(_STEPS_BEFORE_ABORT)
                write_from_another_connection()

        def progress():
            if not budget or budget[0] <= 0:
                return 0
            budget[0] -= 1
            return budget[0] == 0

        conn.set_trace_callback(trace)
        conn.set_progress_handler(progress, 1)
        return conn

    return connect


def _open_while_a_second_opener_arrives(db_path, monkeypatch):
    """Open the store. As the open's first 'rebuild' starts, a second SessionDB opens the same store
    from another thread, and the rebuild goes on once that opener waits for the rebuild authority
    (or has opened). Returns the first handle, the second opener's outcome and the 'rebuild'
    statements each opener ran."""
    rebuilds = {"first": [], "second": []}
    second = {}
    second_waits = threading.Event()

    def open_second():
        try:
            second["db"] = SessionDB(db_path=db_path)
        except Exception as exc:  # reported by the caller's assertion
            second["error"] = exc
        finally:
            second_waits.set()

    thread = threading.Thread(target=open_second, name="second opener")
    real_connect, real_admission = sqlite3.connect, hermes_state_schema.fts_rebuild_admission

    def connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opener = "second" if threading.current_thread() is thread else "first"

        def trace(sql):
            if "VALUES('rebuild')" not in "".join(sql.split()):
                return
            rebuilds[opener].append(sql)
            if opener == "first" and not thread.is_alive() and not second_waits.is_set():
                thread.start()
                second_waits.wait(timeout=60)

        conn.set_trace_callback(trace)
        return conn

    @contextlib.contextmanager
    def admission(*args, **kwargs):
        if threading.current_thread() is thread:
            second_waits.set()
        with real_admission(*args, **kwargs) as admitted:
            yield admitted

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", connect)
        patch.setattr(hermes_state_schema, "fts_rebuild_admission", admission)
        first = SessionDB(db_path=db_path)
        thread.join(timeout=150)
    return first, second, rebuilds


@pytest.mark.parametrize(
    ("dropped", "backfill_rows", "cut_off"),
    [
        (_BASE_TRIGGERS + _TRIGRAM_TRIGGERS, 0, "messages_fts"),
        (_TRIGRAM_TRIGGERS, 0, "messages_fts"),
        (_BASE_TRIGGERS, 5, "messages_fts_trigram"),
    ],
    ids=["all sync triggers", "trigram only", "base only, backfill pending, cut off in the trigram rebuild"],
)
def test_a_trigger_repair_cut_off_in_its_rebuild_admits_no_write_and_the_next_open_completes_it_once(
    tmp_path, monkeypatch, dropped, backfill_rows, cut_off,
):
    db_path = _store_written_while_detached(tmp_path, monkeypatch, dropped, backfill_rows)
    concurrent_writes = []
    with monkeypatch.context() as patch:
        patch.setattr(
            sqlite3, "connect",
            _connect_cutting_off_the_repair_rebuild(sqlite3.connect, db_path, concurrent_writes, cut_off),
        )
        with pytest.raises(sqlite3.OperationalError, match="interrupted"):
            SessionDB(db_path=db_path)

    assert len(concurrent_writes) == 1 and concurrent_writes[0].startswith("refused"), concurrent_writes
    db, second, rebuilds = _open_while_a_second_opener_arrives(db_path, monkeypatch)
    try:
        assert rebuilds["first"], "the next open did not repair"
        assert "db" in second, second
        assert rebuilds["second"] == []
        stored = INDEXED_ROWS + backfill_rows + DETACHED_ROWS
        assert _counts(db) == _counts(second["db"]) == (stored, stored, stored)
        db.append_message("needles", role="user", content=f"{NEEDLE} written after the next open")
        assert _counts(second["db"]) == (stored + 1, stored + 1, stored + 1)
    finally:
        db.close()
        if "db" in second:
            second["db"].close()


_WHILE_ADMITTED = "arrives while the repairer holds admission, before its write transaction"
_WHILE_WRITING = "arrives while the repairer is inside its write transaction"
_ADMITTED_AFTER = "measures before the repair commits, takes admission after it"


@pytest.mark.parametrize("arrival", [_WHILE_ADMITTED, _WHILE_WRITING, _ADMITTED_AFTER])
@pytest.mark.parametrize(
    "dropped", [_BASE_TRIGGERS + _TRIGRAM_TRIGGERS, _TRIGRAM_TRIGGERS], ids=["all sync triggers", "trigram only"],
)
def test_an_opener_arriving_during_another_process_repair_opens_and_writes_without_waiting_for_it(
    tmp_path, monkeypatch, dropped, arrival,
):
    db_path = _store_written_while_detached(tmp_path, monkeypatch, dropped, 0)
    held, release, repairer_done, late_at_admission, late_opened, late_wrote = (threading.Event() for _ in range(6))
    repairer, late = {}, {}
    rebuilds = {"repairer": [], "late": []}

    def open_repairer():
        try:
            repairer["db"] = SessionDB(db_path=db_path)
        except Exception as exc:  # reported by the assertions below
            repairer["error"] = exc
        finally:
            held.set()
            repairer_done.set()

    def open_late_and_write():
        try:
            late["db"] = SessionDB(db_path=db_path)
            late_opened.set()
            late["db"].append_message("needles", role="user", content=f"{NEEDLE} written by the late opener")
            late_wrote.set()
        except Exception as exc:  # reported by the assertions below
            late["error"] = exc
        finally:
            late_opened.set()

    repairer_thread = threading.Thread(target=open_repairer, name="repairer")
    late_thread = threading.Thread(target=open_late_and_write, name="late opener")
    real_connect, real_admission = sqlite3.connect, hermes_state_schema.fts_rebuild_admission

    def hold_the_repair():
        if not held.is_set():
            held.set()
            release.wait(timeout=150)

    def connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opener = "repairer" if threading.current_thread() is repairer_thread else "late"

        def trace(sql):
            if "VALUES('rebuild')" not in "".join(sql.split()):
                return
            rebuilds[opener].append(sql)
            if opener == "repairer" and arrival != _WHILE_ADMITTED:
                hold_the_repair()

        conn.set_trace_callback(trace)
        return conn

    @contextlib.contextmanager
    def admission(*args, **kwargs):
        if threading.current_thread() is late_thread and arrival == _ADMITTED_AFTER:
            # It measured the triggers missing; the repairer commits and lets go before this opener asks.
            late_at_admission.set()
            repairer_done.wait(timeout=150)
        with real_admission(*args, **kwargs) as admitted:
            if admitted and threading.current_thread() is repairer_thread and arrival == _WHILE_ADMITTED:
                hold_the_repair()
            yield admitted

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", connect)
        patch.setattr(hermes_state_schema, "fts_rebuild_admission", admission)
        repairer_thread.start()
        try:
            assert held.wait(timeout=60) and "error" not in repairer, repairer
            late_thread.start()
            if arrival == _ADMITTED_AFTER:
                assert late_at_admission.wait(timeout=30), late
                release.set()
            assert late_opened.wait(timeout=30), f"the late opener waited for the other process's repair: {late}"
            if arrival == _WHILE_ADMITTED:
                # Nobody holds the write lock yet: the write commits while its row is still owed to the indexes.
                assert late_wrote.wait(timeout=30), f"the late opener's write waited for the repair: {late}"
        finally:
            release.set()
            repairer_thread.join(timeout=150)
            if late_thread.ident is not None:
                late_thread.join(timeout=150)
    try:
        assert "error" not in repairer and "error" not in late, (repairer, late)
        assert rebuilds["repairer"], "the repairer did not repair"
        assert rebuilds["late"] == []
        stored = INDEXED_ROWS + DETACHED_ROWS + 1
        assert _counts(repairer["db"]) == _counts(late["db"]) == (stored, stored, stored)
    finally:
        for handle in (repairer.get("db"), late.get("db")):
            if handle is not None:
                handle.close()


def test_an_opener_that_leaves_the_repair_to_another_process_finds_every_row(tmp_path, monkeypatch):
    db_path = _store_written_while_detached(tmp_path, monkeypatch, _BASE_TRIGGERS + _TRIGRAM_TRIGGERS, 0)
    stored = INDEXED_ROWS + DETACHED_ROWS
    # The holder need not be a trigger repair (optimize-storage, a stale-index recovery): the gap stays while it runs.
    with fts_rebuild_admission(db_path, timeout_seconds=0) as held:
        assert held
        db = SessionDB(db_path=db_path)
        try:
            assert len(db.search_messages(NEEDLE, limit=stored + 10)) == stored
        finally:
            db.close()


def test_once_the_other_process_lets_go_the_housekeeping_retry_repairs_and_search_uses_the_indexes(
    tmp_path, monkeypatch,
):
    db_path = _store_written_while_detached(tmp_path, monkeypatch, _BASE_TRIGGERS + _TRIGRAM_TRIGGERS, 0)
    stored = INDEXED_ROWS + DETACHED_ROWS + 1
    db = None
    try:
        with fts_rebuild_admission(db_path, timeout_seconds=0) as held:
            assert held
            db = SessionDB(db_path=db_path)
            assert db.retry_deferred_fts_recovery() is False  # still held
        db.append_message("needles", role="user", content=f"{NEEDLE} written while the repair was left elsewhere")
        db._fts_stale_retry_after = 0.0  # the backoff the held attempt earned; not what this test is about
        assert db.retry_deferred_fts_recovery() is True
        assert _counts(db) == (stored, stored, stored)
        assert db._describe_search_path(NEEDLE) == "fts5"
        assert len(db.search_messages(NEEDLE, limit=stored + 10)) == stored
    finally:
        if db is not None:
            db.close()
