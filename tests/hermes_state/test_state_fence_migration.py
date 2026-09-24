"""In-place migration to the fenced lineage.

A fork-29 store (plain or already touched by R's reconcile), an upstream store and a fresh
store all end at ``STORED_SCHEMA_VERSION`` with exactly this build's fences on the governed
tables present. The fork's own authority triggers keep firing, and from then on only a
connection reporting this build's generation can write a governed table.
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from hermes_state_common import FTS_STORAGE_VERSION
from hermes_state_fence import STORED_SCHEMA_VERSION
from tests.hermes_state.fence_store_probe import expected_fences, fence_triggers, governed_present, read_ro, stored
from tests.hermes_state.fork_store_fixture import (
    AUTHORITY_GOVERNED, CORE_GOVERNED, _connect_as, build_store, isolate_home,
)

# Governed tables the fork fixture leaves empty: seeded at this build's generation so the
# UPDATE and DELETE fences have a row to fire on.
_SEED_EMPTY_GOVERNED = (
    "INSERT OR IGNORE INTO system_prompts (hash, prompt) VALUES ('seed-hash', 'seed prompt')",
    "INSERT OR IGNORE INTO session_turn_leases (conversation_id, holder, acquired_at, expires_at) "
    "VALUES ('seed-conversation', 'seed', 1.0, 2.0)",
    "INSERT OR IGNORE INTO session_process_reservations (reservation_id, reservation_token_sha256, session_id, "
    "session_generation, state_db_id, state_family, status, reserved_at, expires_at) "
    "SELECT 'seed-reservation', authority_token, session_id, session_generation, state_db_id, state_family, "
    "'RESERVED', 1.0, 2.0 FROM session_process_authorities LIMIT 1",
)


def _lifecycle(db_path) -> dict:
    from hermes_state import SessionDB
    from tools import async_delegation

    db = SessionDB(db_path=db_path)
    try:
        db.create_session("life", "cli")
        db.append_message("life", "user", "quokka lifecycle question")
        db.append_message("life", "assistant", "quokka lifecycle answer")
        db.append_message("life", "tool", "quokka tool output", tool_name="terminal", tool_call_id="call-life")
        db.end_session("life", "user_exit")
        db.reopen_session("life")
        db.set_session_title("life", "Lifecycle title")
        db.save_gateway_routing_entry("k-life", json.dumps({"session_id": "life"}))
        db.update_token_counts("life", input_tokens=3, output_tokens=2, model="m-life")
        hits = {hit["session_id"] for hit in db.search_messages("quokka")}
    finally:
        db.close()
    async_delegation._persist_dispatch({
        "delegation_id": "deleg-life", "dispatched_at": time.time(), "session_key": "k-life",
        "origin_ui_session_id": "", "parent_session_id": "life", "origin_session_id": "life",
    })
    observed = {
        "hits": hits,
        "title": read_ro(db_path, "SELECT title FROM sessions WHERE id = 'life'"),
        "routing": read_ro(db_path, "SELECT session_key FROM gateway_routing WHERE session_key = 'k-life'"),
        "usage": read_ro(db_path, "SELECT model FROM session_model_usage WHERE session_id = 'life'"),
        "delegation": read_ro(db_path, "SELECT state FROM async_delegations WHERE delegation_id = 'deleg-life'"),
        # The fork's issue / close / reopen triggers derive one authority row per generation.
        "authorities": read_ro(db_path, "SELECT session_generation, status FROM session_process_authorities "
                                        "WHERE session_id = 'life' ORDER BY session_generation"),
    }
    db = SessionDB(db_path=db_path)
    try:
        observed["deleted"] = db.delete_session("life")
    finally:
        db.close()
    observed["remaining"] = read_ro(db_path, "SELECT COUNT(*) FROM messages WHERE session_id = 'life'")[0][0]
    return observed


@pytest.mark.parametrize("kind", ["fork29", "fork29_touched"])
def test_a_fork_store_serves_a_full_lifecycle_after_migration(tmp_path, monkeypatch, kind):
    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", kind)

    observed = _lifecycle(db_path)

    assert observed["hits"] == {"life"}
    assert observed["title"] == [("Lifecycle title",)]
    assert observed["routing"] == [("k-life",)]
    assert observed["usage"] == [("m-life",)]
    assert observed["delegation"] == [("running",)]
    assert [status for _, status in observed["authorities"]] == ["CLOSED", "ISSUED"]
    assert observed["deleted"] is True and observed["remaining"] == 0
    assert stored(db_path) == STORED_SCHEMA_VERSION


def _first_row_update(conn, table: str) -> str:
    column = conn.execute(f'PRAGMA table_info("{table}")').fetchone()[1]
    return f'UPDATE "{table}" SET "{column}" = "{column}" WHERE rowid IN (SELECT rowid FROM "{table}" LIMIT 1)'


@pytest.mark.parametrize("generation", [29, None], ids=["fork-generation", "no-udf"])
def test_after_migration_only_this_generation_writes_a_governed_table(tmp_path, monkeypatch, generation):
    from hermes_state import SessionDB

    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fork29")
    SessionDB(db_path=db_path).close()
    seeder = _connect_as(db_path, STORED_SCHEMA_VERSION)
    try:
        for sql in _SEED_EMPTY_GOVERNED:
            seeder.execute(sql)
    finally:
        seeder.close()

    assert stored(db_path) == STORED_SCHEMA_VERSION
    assert governed_present(db_path) == list(CORE_GOVERNED + AUTHORITY_GOVERNED)
    assert fence_triggers(db_path) == expected_fences(db_path, STORED_SCHEMA_VERSION)

    expected_error = sqlite3.IntegrityError if generation is not None else sqlite3.OperationalError
    expected_text = "state DB generation incompatible" if generation is not None else "no such function"
    foreign = _connect_as(db_path, generation)
    try:
        refused = []
        for table in CORE_GOVERNED + AUTHORITY_GOVERNED:
            assert foreign.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] > 0, table
            # BEFORE INSERT fires ahead of NOT NULL checks, so DEFAULT VALUES reaches the fence.
            for sql in (f'INSERT INTO "{table}" DEFAULT VALUES', _first_row_update(foreign, table),
                        f'DELETE FROM "{table}" WHERE rowid IN (SELECT rowid FROM "{table}" LIMIT 1)'):
                with pytest.raises(expected_error, match=expected_text):
                    foreign.execute(sql)
                refused.append(sql.split()[0])
    finally:
        foreign.close()
    assert len(refused) == 3 * len(CORE_GOVERNED + AUTHORITY_GOVERNED)


@pytest.mark.parametrize("kind", ["upstream29", "upstream30", "fresh"])
def test_upstream_and_fresh_stores_reach_the_fenced_stamp(tmp_path, monkeypatch, kind):
    from hermes_state import SessionDB
    from hermes_state_schema import SessionSchemaMixin

    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", kind)
    trigram_steps = []
    real_step = SessionSchemaMixin._migrate_trigram_cron_exclusion

    def counting_step(self, cursor):
        trigram_steps.append(kind)
        return real_step(self, cursor)

    monkeypatch.setattr(SessionSchemaMixin, "_migrate_trigram_cron_exclusion", counting_step)
    SessionDB(db_path=db_path).close()
    SessionDB(db_path=db_path).close()

    assert stored(db_path) == STORED_SCHEMA_VERSION
    # R stores carry no authority tables, so the fences cover exactly the core set.
    assert fence_triggers(db_path) == expected_fences(db_path, STORED_SCHEMA_VERSION, CORE_GOVERNED)
    # The < 30 trigram step runs on the one lineage below its gate, once, and settles the layout.
    assert len(trigram_steps) == (1 if kind == "upstream29" else 0)
    assert read_ro(db_path, "SELECT value FROM state_meta WHERE key = 'fts_storage_version'") == [
        (str(FTS_STORAGE_VERSION),)]
