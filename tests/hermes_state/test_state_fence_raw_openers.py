"""A raw state.db opener that arrives before any SessionDB open after cutover.

An opener of its own profile's store that writes governed tables (the async-delegation ledger)
delegates to the one SessionDB migration first: same lock, same single transaction, so a
failure leaves the fork store exactly as it was. The cross-profile A2A forwarder never migrates
a peer's store; its governed write is refused by the fence and nothing is lost. The delivery
ledger writes no governed table and is unaffected either way.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from hermes_state_fence import STORED_SCHEMA_VERSION
from tests.hermes_state.fence_store_probe import (
    expected_fences, fence_master_rows, fence_triggers, read_ro, stamp_rows, stored,
)
from tests.hermes_state.fork_store_fixture import build_store, isolate_home


def _dispatch(delegation_id: str) -> None:
    from tools import async_delegation

    async_delegation._persist_dispatch({
        "delegation_id": delegation_id, "dispatched_at": time.time(), "session_key": "k",
        "origin_ui_session_id": "", "parent_session_id": "fx-beta", "origin_session_id": "fx-beta",
    })


def _delegations(db_path, delegation_id: str) -> list:
    return read_ro(db_path, "SELECT state FROM async_delegations WHERE delegation_id = ?", (delegation_id,))


def test_the_delegation_ledger_migrates_an_unmigrated_fork_store_before_writing(tmp_path, monkeypatch):
    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fork29")

    _dispatch("raw-first")

    assert _delegations(db_path, "raw-first") == [("running",)]
    assert stored(db_path) == STORED_SCHEMA_VERSION
    assert fence_triggers(db_path) == expected_fences(db_path, STORED_SCHEMA_VERSION)


def test_a_raw_opener_that_fails_mid_migration_leaves_the_fork_store_intact(tmp_path, monkeypatch):
    import hermes_state_fence

    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fork29")
    fences_before, stamp_before = fence_master_rows(db_path), stamp_rows(db_path)

    def failing_stamp(*_args, **_kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    with monkeypatch.context() as patch:
        patch.setattr(hermes_state_fence, "_write_lineage_stamp", failing_stamp)
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            _dispatch("raw-failed")

    assert (fence_master_rows(db_path), stamp_rows(db_path)) == (fences_before, stamp_before)
    assert _delegations(db_path, "raw-failed") == []
    _dispatch("raw-retried")
    assert _delegations(db_path, "raw-retried") == [("running",)]
    assert stored(db_path) == STORED_SCHEMA_VERSION


def test_the_a2a_forwarder_leaves_an_unmigrated_peer_store_alone(tmp_path, monkeypatch):
    from plugins.platforms.a2a import adapter

    isolate_home(tmp_path, monkeypatch)
    peer = build_store(tmp_path / "peer" / "state.db", "fork29")
    monkeypatch.setattr(adapter, "_profile_home", lambda profile: str(peer.parent))
    fences_before, stamp_before = fence_master_rows(peer), stamp_rows(peer)
    title_before = read_ro(peer, "SELECT title FROM sessions WHERE id = 'fx-beta'")

    written = adapter._state_db("peer", "UPDATE sessions SET title = ? WHERE id = ?", ("a2a-ctx", "fx-beta"),
                                "title", commit=True)
    found = adapter._state_db("peer", "SELECT id FROM sessions WHERE id = ?", ("fx-beta",), "lookup")

    assert written == "" and found == "fx-beta"
    assert read_ro(peer, "SELECT title FROM sessions WHERE id = 'fx-beta'") == title_before
    assert (fence_master_rows(peer), stamp_rows(peer)) == (fences_before, stamp_before)


def test_the_delivery_ledger_writes_an_unmigrated_fork_store(tmp_path, monkeypatch):
    from gateway import delivery_ledger

    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fork29")

    delivery_ledger.record_obligation(obligation_id="ob-raw", session_key="k", platform="telegram",
                                      chat_id="c", thread_id=None, content="owed")

    assert read_ro(db_path, "SELECT state FROM delivery_obligations WHERE obligation_id = 'ob-raw'") == [("pending",)]
