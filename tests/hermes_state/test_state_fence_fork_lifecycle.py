"""The fork's session-process authority triggers keep working under this build's SessionDB once
the store's fences carry this build's generation: every lifecycle call succeeds and the fork's
authority rows follow it (issued on create, closed on end, re-issued at the next generation on
reopen). A fork trigger that aborted an R write would make the migrated live store unusable."""

from __future__ import annotations

import sqlite3

import pytest

from tests.hermes_state.fork_store_fixture import CURRENT_STAMP, build_store, isolate_home, refence


def _authorities(db, session_id):
    conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return conn.execute("SELECT session_generation, status FROM session_process_authorities "
                            "WHERE session_id = ? ORDER BY session_generation", (session_id,)).fetchall()
    finally:
        conn.close()


def _fenced_at_29_stamp(db):
    # The live shape before S2's stamp: fork stamp 29, fences already at this build's literal.
    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.create_function("hermes_turn_fence_generation", 0, lambda: 29)
    try:
        refence(conn, CURRENT_STAMP)
    finally:
        conn.close()


@pytest.mark.parametrize("kind", ["fenced1030", "fork29_touched"])
def test_sessiondb_lifecycle_on_a_fork_store(tmp_path, monkeypatch, kind):
    db = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", kind)
    if kind == "fork29_touched":
        _fenced_at_29_stamp(db)
    from hermes_state import SessionDB

    session_db = SessionDB(db_path=db)
    try:
        session_db.create_session("life-1", "cli")
        session_db.append_message("life-1", "user", "lifecycle probe")
        session_db.append_message("life-1", "assistant", None, tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}])
        session_db.append_message("life-1", "tool", "tool body", tool_name="read_file", tool_call_id="c1")
        assert _authorities(db, "life-1") == [(1, "ISSUED")]
        session_db.end_session("life-1", "user_exit")
        session_db.reopen_session("life-1")
        assert _authorities(db, "life-1") == [(1, "CLOSED"), (2, "ISSUED")]
        session_db.set_session_title("life-1", "life title")
        session_db.save_gateway_routing_entry("life:key", '{"platform": "cli"}')
        session_db.update_token_counts("life-1", 3, 4, model="m")
        assert session_db.search_messages("lifecycle")
        session_db.end_session("life-1", "user_exit")
        assert session_db.delete_session("fx-alpha")
        assert session_db.prune_sessions(older_than_days=None, started_before=float("inf")) >= 1
    finally:
        session_db.close()

    assert _authorities(db, "life-1") == []
