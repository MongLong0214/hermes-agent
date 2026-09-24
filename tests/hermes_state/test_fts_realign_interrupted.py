"""The ``messages_fts`` realign survives being cut off in its rebuild.

Realigning a v1/v2 index onto ``messages_fts_src`` drops the old index, creates the aligned
one and fills it with FTS5 ``'rebuild'``. On the autocommit writer the DDL used to commit on
its own, so a process killed during the rebuild (minutes on a multi-GB store) left the aligned
shape over an empty index: the shape probe that schedules the realign no longer fired, and
search missed every older message from then on.

Behaviour contracts on the index a reopen serves, not on how the migration is written: every
row carrying a token is found by the index (MATCH count == LIKE count), and the layout marker
is current. The cut-off realign runs on a layout-2 store, whose marker is already current when
the rebuild starts; the already-damaged stores carry each marker such a realign left behind.
"""

import sqlite3

import pytest

from hermes_state import SessionDB, StateDbCorruptError
from hermes_state_common import FTS_SQL, FTS_STORAGE_VERSION
from tests.hermes_state.fork_store_fixture import build_store, isolate_home

NEEDLE = "zephyrquill"
NEEDLE_ROWS = 120
# Progress callbacks (one per VM step) the rebuild runs before it is cut off: it has started, not finished.
_STEPS_BEFORE_ABORT = 50


def _fork_store_with_needles(tmp_path, monkeypatch):
    """The fork's generation-29 store (layout 1: ``messages_fts`` reads raw ``messages``) after its
    own writer indexed NEEDLE_ROWS more rows carrying NEEDLE, tool rows among them."""
    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fork29")
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.create_function("hermes_turn_fence_generation", 0, lambda: 29)
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            "INSERT INTO messages (session_id, role, content, tool_name, timestamp) "
            "VALUES ('fx-alpha', ?, ?, ?, 1.0)",
            [
                ("tool", f"{NEEDLE} tool row {i}", "read_file") if i % 3 == 0
                else ("user", f"{NEEDLE} user row {i}", None)
                for i in range(NEEDLE_ROWS)
            ],
        )
        conn.execute("COMMIT")
    finally:
        conn.close()
    return db_path


def _layout2_store_with_needles(tmp_path, monkeypatch):
    """A layout-2 store: ``messages_fts`` reads raw ``messages`` over a filled index, and the
    trigram is already current, so the open stamps the new layout BEFORE it realigns. The marker
    then cannot tell a realign that died from one that finished; only the index can."""
    db_path = isolate_home(tmp_path, monkeypatch) / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("needles", source="cli")
        for i in range(NEEDLE_ROWS):
            db.append_message("needles", role="user", content=f"{NEEDLE} user row {i}")
        db._conn.execute("DROP TABLE messages_fts")
        db._conn.execute(
            "CREATE VIRTUAL TABLE messages_fts USING fts5("
            "content, tool_name, tool_calls, content='messages', content_rowid='id')"
        )
        db._conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        db.set_meta("fts_storage_version", "2")
        assert _needle_counts(db) == (NEEDLE_ROWS, NEEDLE_ROWS)
    finally:
        db.close()
    return db_path


def _needle_counts(db):
    """(rows the index finds, rows that hold the token)."""
    indexed = db._conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?", (NEEDLE,)
    ).fetchone()[0]
    stored = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE content LIKE ?", (f"%{NEEDLE}%",)
    ).fetchone()[0]
    return indexed, stored


def _connect_cutting_off_the_realign_rebuild(real_connect):
    """``sqlite3.connect`` whose connections abort the base index's 'rebuild' once it is under
    way (SQLITE_INTERRUPT from the progress handler): an in-process stand-in for the kill."""

    def connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        budget = []

        def trace(sql):
            if not budget and "INTO messages_fts(messages_fts) VALUES('rebuild')" in sql:
                budget.append(_STEPS_BEFORE_ABORT)

        def progress():
            if not budget or budget[0] <= 0:
                return 0
            budget[0] -= 1
            return budget[0] == 0

        conn.set_trace_callback(trace)
        conn.set_progress_handler(progress, 1)
        return conn

    return connect


def test_a_realign_cut_off_in_its_rebuild_completes_on_the_next_open(tmp_path, monkeypatch):
    db_path = _layout2_store_with_needles(tmp_path, monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", _connect_cutting_off_the_realign_rebuild(sqlite3.connect))
        with pytest.raises(sqlite3.OperationalError, match="interrupted"):
            SessionDB(db_path=db_path)

    db = SessionDB(db_path=db_path)
    try:
        assert _needle_counts(db) == (NEEDLE_ROWS, NEEDLE_ROWS)
        assert db.get_meta("fts_storage_version") == str(FTS_STORAGE_VERSION)
    finally:
        db.close()


# The layout marker a cut-off realign left: the fork's layout 1; none, on a layout-1 store its creating
# build opened only once (older builds stamp only on reopen); current, on a layout-2 store (this build
# stamps it before realigning).
_MARKER_LEFT_BEHIND = {"layout 1": "1", "absent": None, "layout 2, stamped first": str(FTS_STORAGE_VERSION)}


@pytest.mark.parametrize("marker", list(_MARKER_LEFT_BEHIND.values()), ids=list(_MARKER_LEFT_BEHIND))
def test_an_aligned_index_a_cut_off_realign_left_unfilled_is_rebuilt_on_open(tmp_path, monkeypatch, marker):
    """Stores an older build already left behind: the aligned shape and an index holding only what
    was written since, whatever the layout marker says."""
    db_path = _fork_store_with_needles(tmp_path, monkeypatch)
    db = SessionDB(db_path=db_path)
    try:
        # What the autocommit realign had committed when its rebuild died.
        for trigger in ("messages_fts_insert", "messages_fts_delete", "messages_fts_update"):
            db._conn.execute(f"DROP TRIGGER {trigger}")
        db._conn.execute("DROP TABLE messages_fts")
        db._conn.executescript(FTS_SQL)
        db._conn.execute("DELETE FROM state_meta WHERE key = 'fts_storage_version'")
        if marker is not None:
            db.set_meta("fts_storage_version", marker)
        # One message written since makes the index non-empty, so emptiness no longer shows the loss.
        db.create_session("after-the-kill", source="cli")
        db.append_message("after-the-kill", role="user", content=f"{NEEDLE} written after the kill")
        assert _needle_counts(db) == (1, NEEDLE_ROWS + 1)
    finally:
        db.close()

    reopened = SessionDB(db_path=db_path)
    try:
        assert _needle_counts(reopened) == (NEEDLE_ROWS + 1, NEEDLE_ROWS + 1)
        assert reopened.get_meta("fts_storage_version") == str(FTS_STORAGE_VERSION)
    finally:
        reopened.close()


def test_an_index_missing_its_docsize_table_still_opens_and_its_write_is_quarantined(tmp_path, monkeypatch):
    """The unfilled-index probe reads ``messages_fts_docsize``. A store that lost that shadow table
    but kept the vtable row must open as it did before the probe, so its first write reports the
    damage and quarantines the handle (#97940), rather than every open failing."""
    db_path = isolate_home(tmp_path, monkeypatch) / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("s1", source="cli")
        db.append_message("s1", "user", "indexed before the shadow table went")
        db._conn.execute("DROP TABLE messages_fts_docsize")
    finally:
        db.close()

    db = SessionDB(db_path=db_path)
    try:
        with pytest.raises(StateDbCorruptError):
            db.append_message("s1", "user", "written after")
        assert db._db_corrupt is True
        with pytest.raises(StateDbCorruptError):
            db.append_message("s1", "user", "second write")
    finally:
        db.close()
