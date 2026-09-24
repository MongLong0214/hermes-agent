"""A governed table is never committed without its fences.

SQLite's ALTER TABLE RENAME carries a table's triggers onto the legacy copy and the DROP of
that copy deletes them, so a table rebuild that did not re-declare them would leave the new
table writable by any build. DDL from a build that does not know the fences can also remove
them; the next open restores them and says so.
"""

from __future__ import annotations

import logging
import sqlite3

from hermes_state_fence import STORED_SCHEMA_VERSION
from tests.hermes_state.fence_store_probe import expected_fences, fence_triggers, read_ro
from tests.hermes_state.fork_store_fixture import OPERATIONS, _connect_as, build_store, fence_sql, isolate_home


def _give_routing_its_legacy_primary_key(db_path) -> None:
    conn = _connect_as(db_path, 29)
    try:
        rows = conn.execute("SELECT session_key, entry_json, updated_at FROM gateway_routing").fetchall()
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DROP TABLE gateway_routing")
        conn.execute("CREATE TABLE gateway_routing (session_key TEXT PRIMARY KEY, entry_json TEXT NOT NULL, "
                     "updated_at REAL NOT NULL)")
        for op in OPERATIONS:
            conn.execute(fence_sql("gateway_routing", op, 29))
        conn.executemany("INSERT INTO gateway_routing VALUES (?, ?, ?)", rows)
        conn.execute("COMMIT")
    finally:
        conn.close()


def test_the_routing_heal_commits_the_rebuilt_table_with_its_fences(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from hermes_state_schema import SessionSchemaMixin

    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fork29")
    _give_routing_its_legacy_primary_key(db_path)
    committed = []
    real_rebuild = SessionSchemaMixin._rebuild_table

    def observing_rebuild(cursor, table, *args, **kwargs):
        real_rebuild(cursor, table, *args, **kwargs)
        # Another connection's view the moment the rebuild returns, long before _init_schema ends.
        committed.append((table, {name: sql for name, sql in fence_triggers(db_path).items()
                                  if name.startswith(f"turn_fence_{table}_")}))

    monkeypatch.setattr(SessionSchemaMixin, "_rebuild_table", staticmethod(observing_rebuild))
    SessionDB(db_path=db_path).close()

    assert committed == [
        ("gateway_routing", expected_fences(db_path, STORED_SCHEMA_VERSION, ["gateway_routing"])),
    ]
    pk = [row[1] for row in sorted(
        (r for r in read_ro(db_path, "PRAGMA table_info(gateway_routing)") if r[5]), key=lambda r: r[5])]
    assert pk == ["scope", "session_key"]
    assert read_ro(db_path, "SELECT COUNT(*) FROM gateway_routing")[0][0] == 1


def test_an_open_restores_fences_another_build_removed(tmp_path, monkeypatch, caplog):
    from hermes_state import SessionDB

    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fork29")
    SessionDB(db_path=db_path).close()
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        for op in OPERATIONS:
            conn.execute(f"DROP TRIGGER turn_fence_messages_{op.lower()}")
    finally:
        conn.close()

    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        SessionDB(db_path=db_path).close()

    assert fence_triggers(db_path) == expected_fences(db_path, STORED_SCHEMA_VERSION)
    warned = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert all(f"turn_fence_messages_{op.lower()}" in warned for op in OPERATIONS), warned


def test_a_heal_that_drops_a_fence_mid_open_is_restored_before_the_commit(tmp_path, monkeypatch, caplog):
    """The fence delta has already found the fences exact; hand-written DDL later in the same
    open (a heal or migration that rebuilds a governed table itself) must not commit a gap."""
    from hermes_state import SessionDB
    from hermes_state_schema import SessionSchemaMixin

    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fork29")
    SessionDB(db_path=db_path).close()
    real_heal = SessionSchemaMixin._heal_session_model_usage_pk

    def fence_dropping_heal(self, cursor):
        real_heal(self, cursor)
        cursor.execute("DROP TRIGGER turn_fence_messages_insert")

    monkeypatch.setattr(SessionSchemaMixin, "_heal_session_model_usage_pk", fence_dropping_heal)
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        SessionDB(db_path=db_path).close()

    assert fence_triggers(db_path) == expected_fences(db_path, STORED_SCHEMA_VERSION)
    warned = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert "turn_fence_messages_insert" in warned, warned
