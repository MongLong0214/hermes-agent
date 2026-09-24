"""The fence swap and the lineage stamp commit together or not at all, and a store whose
fence declarations this build does not own is refused without the migration touching it."""

from __future__ import annotations

import sqlite3

import pytest

from hermes_state_errors import (
    SCHEMA_CAUSE_BUILD_TOO_OLD, SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, IncompatibleSchemaError,
)
from hermes_state_fence import (
    FENCE_LINEAGE_BASE, FORK_BASE_UPSTREAM_GATE, STORED_SCHEMA_VERSION, register_turn_fence_generation,
)
from tests.hermes_state.fence_store_probe import (
    expected_fences, fence_master_rows, fence_triggers, read_ro, stamp_rows, stored,
)
from tests.hermes_state.fork_store_fixture import _connect_as, build_store, fence_sql, isolate_home

_UNOWNED_BODIES = {
    # A gate stamp is never a fence literal: the boundary just below the older-fenced window.
    "gate-stamp-literal": fence_sql("messages", "INSERT", FENCE_LINEAGE_BASE + FORK_BASE_UPSTREAM_GATE),
    "same-name-non-fence": "CREATE TRIGGER turn_fence_messages_insert BEFORE INSERT ON messages BEGIN SELECT 1; END",
}


def test_a_failed_stamp_rolls_back_the_fence_swap(tmp_path, monkeypatch):
    import hermes_state_fence
    from hermes_state import SessionDB

    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fork29")
    fences_before, stamp_before = fence_master_rows(db_path), stamp_rows(db_path)
    assert stamp_before == [(29, "integer")]

    def failing_stamp(*_args, **_kwargs):
        # Not a malformed-schema error, so the open does not route into the repair ladder.
        raise sqlite3.OperationalError("disk I/O error")

    with monkeypatch.context() as patch:
        patch.setattr(hermes_state_fence, "_write_lineage_stamp", failing_stamp)
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            SessionDB(db_path=db_path)

    assert fence_master_rows(db_path) == fences_before
    assert stamp_rows(db_path) == stamp_before
    SessionDB(db_path=db_path).close()
    assert stored(db_path) == STORED_SCHEMA_VERSION
    assert fence_triggers(db_path) == expected_fences(db_path, STORED_SCHEMA_VERSION)


def _replace_messages_insert_fence(db_path, sql: str) -> None:
    conn = _connect_as(db_path, 29)
    try:
        conn.execute("DROP TRIGGER turn_fence_messages_insert")
        conn.execute(sql)
    finally:
        conn.close()


def _schema(db_path) -> list:
    return read_ro(db_path, "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY rowid")


@pytest.mark.parametrize("body", sorted(_UNOWNED_BODIES))
def test_an_unowned_fence_declaration_is_refused_without_mutation(tmp_path, monkeypatch, body):
    from hermes_state import SessionDB

    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fork29")
    _replace_messages_insert_fence(db_path, _UNOWNED_BODIES[body])
    before = _schema(db_path)

    with pytest.raises(IncompatibleSchemaError) as refused:
        SessionDB(db_path=db_path)

    assert refused.value.cause == SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH
    assert _schema(db_path) == before


def _future_stamp(db_path) -> None:
    conn = _connect_as(db_path, 29)
    try:
        conn.execute("UPDATE schema_version SET version = ?", (STORED_SCHEMA_VERSION + 1,))
    finally:
        conn.close()


def _unowned_racer(sql: str):
    return lambda db_path: _replace_messages_insert_fence(db_path, sql)


_RACES = {
    **{body: (_unowned_racer(sql), SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH) for body, sql in _UNOWNED_BODIES.items()},
    "future-stamp": (_future_stamp, SCHEMA_CAUSE_BUILD_TOO_OLD),
}


@pytest.mark.parametrize("race", sorted(_RACES))
def test_the_fence_delta_re_decodes_under_its_lock(tmp_path, monkeypatch, race):
    """A declaration or stamp that appears between the read-first decode and the write lock is
    refused inside the transaction, and the rollback leaves the store exactly as the racer left it."""
    import hermes_state_fence

    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fork29")
    real_transaction = hermes_state_fence._in_write_transaction
    racer, cause = _RACES[race]
    raced = []

    def racing_transaction(cursor, body_fn):
        racer(db_path)
        raced.append((_schema(db_path), stamp_rows(db_path)))
        return real_transaction(cursor, body_fn)

    monkeypatch.setattr(hermes_state_fence, "_in_write_transaction", racing_transaction)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        register_turn_fence_generation(conn)
        with pytest.raises(IncompatibleSchemaError) as refused:
            hermes_state_fence.apply_fence_delta(conn.cursor())
        assert not conn.in_transaction
    finally:
        conn.close()

    assert refused.value.cause == cause
    assert raced == [(_schema(db_path), stamp_rows(db_path))]
