"""Every state.db opener that writes governed tables can write a store fenced at this build's
generation: the fence triggers abort any governed write from a connection that does not report
the generation, so an opener that skips registration silently loses its writes (or raises)."""

from __future__ import annotations

import sqlite3
import time

from tests.hermes_state.fork_store_fixture import build_store, isolate_home


def _read(db, sql, params=()):
    conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def test_sessiondb_writer_writes_a_fenced_store(tmp_path, monkeypatch):
    db = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fenced1030")
    from hermes_state import SessionDB

    session_db = SessionDB(db_path=db)
    try:
        session_db.create_session("opener-writer", "cli")
        session_db.append_message("opener-writer", "user", "fenced write")
    finally:
        session_db.close()

    assert _read(db, "SELECT content FROM messages WHERE session_id = ?", ("opener-writer",)) == [("fenced write",)]


def test_repair_probe_reports_a_fenced_store_healthy(tmp_path, monkeypatch):
    db = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fenced1030")
    from hermes_state_repair import _db_opens_cleanly

    assert _db_opens_cleanly(db) is None


def test_async_delegation_ledger_writes_a_fenced_store(tmp_path, monkeypatch):
    db = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fenced1030")
    from tools import async_delegation

    async_delegation._persist_dispatch({
        "delegation_id": "opener-deleg", "dispatched_at": time.time(), "session_key": "k",
        "origin_ui_session_id": "", "parent_session_id": "fx-beta", "origin_session_id": "fx-beta",
    })

    assert _read(db, "SELECT state FROM async_delegations WHERE delegation_id = 'opener-deleg'") == [("running",)]


def test_a2a_forward_titles_a_session_in_a_fenced_store(tmp_path, monkeypatch):
    isolate_home(tmp_path, monkeypatch)
    db = build_store(tmp_path / "peer" / "state.db", "fenced1030")
    from plugins.platforms.a2a import adapter
    monkeypatch.setattr(adapter, "_profile_home", lambda profile: str(db.parent))

    adapter._state_db("peer", "UPDATE sessions SET title = ? WHERE id = ?", ("a2a-peer-ctx", "fx-beta"),
                      "title", commit=True)

    assert _read(db, "SELECT title FROM sessions WHERE id = 'fx-beta'") == [("a2a-peer-ctx",)]
