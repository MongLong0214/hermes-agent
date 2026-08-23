"""Slice 1 — a turn-lease grant carries a generation that only ever increases.

Property under test: two acquisitions of the same conversation are
distinguishable from each other, even when the holder STRING is identical.
Without that, a holder alone is a bearer credential with no expiry that any
later writer can present.

Nothing here imports a symbol that may not exist yet — the whole file must
IMPORT at any commit so that a failure is behavioural (rc 1) rather than a
collection error (rc 2), which proves only that a name is missing.
"""

from __future__ import annotations

import os
import sqlite3
import time

import pytest

from hermes_state import SessionDB


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _seed(db: SessionDB, session_id: str = "s") -> str:
    db.create_session(session_id, source="test")
    return session_id


def _epoch_of(grant) -> object:
    """The generation a grant carries, or None when it carries none."""
    return getattr(grant, "epoch", None)


def _lease_row(db: SessionDB, conversation_id: str = "s"):
    with db._read_ctx() as conn:
        return conn.execute(
            "SELECT * FROM session_turn_leases WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()


def test_acquire_returns_a_grant_carrying_an_epoch(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    _seed(db)
    grant = db.try_acquire_session_turn_lease("s", _holder("a"), ttl_seconds=5)
    assert grant, "acquisition failed outright"
    assert isinstance(_epoch_of(grant), int), (
        f"acquire returned {grant!r}, which carries no epoch — a holder string "
        f"alone cannot be told apart from a replay of an earlier grant"
    )
    assert str(grant) == _holder("a"), (
        "the grant must still be usable everywhere the holder string was"
    )


def test_the_epoch_increases_across_release_and_reacquire(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    _seed(db)
    first = db.try_acquire_session_turn_lease("s", _holder("a"), ttl_seconds=5)
    assert first
    db.release_session_turn_lease("s", first)
    second = db.try_acquire_session_turn_lease("s", _holder("b"), ttl_seconds=5)
    assert second
    assert isinstance(_epoch_of(first), int) and isinstance(_epoch_of(second), int), (
        f"grants carry no epoch: {first!r} -> {second!r}"
    )
    assert _epoch_of(second) > _epoch_of(first)


def test_the_epoch_never_restarts_even_under_one_holder_string(tmp_path):
    """Release must not reset the counter — that is the whole replay window."""
    db = SessionDB(tmp_path / "state.db")
    _seed(db)
    holder = _holder("same-every-time")
    seen = []
    for _ in range(4):
        grant = db.try_acquire_session_turn_lease("s", holder, ttl_seconds=5)
        assert grant, f"could not re-acquire on cycle {len(seen)}"
        seen.append(_epoch_of(grant))
        db.release_session_turn_lease("s", grant)
    assert all(isinstance(e, int) for e in seen), f"grants carry no epoch: {seen}"
    assert seen == sorted(seen) and len(set(seen)) == 4, seen


def test_refresh_and_release_require_the_granted_epoch(tmp_path):
    """A superseded grant must not extend or free the current owner's lease."""
    db = SessionDB(tmp_path / "state.db")
    _seed(db)
    holder = _holder("owner")
    first = db.try_acquire_session_turn_lease("s", holder, ttl_seconds=5)
    assert first
    db.release_session_turn_lease("s", first)
    second = db.try_acquire_session_turn_lease("s", holder, ttl_seconds=5)
    assert second
    assert _epoch_of(first) != _epoch_of(second), (
        "re-acquiring under the same holder string produced the same grant"
    )

    assert db.refresh_session_turn_lease("s", first, ttl_seconds=5) is False, (
        "a superseded grant extended the live owner's deadline"
    )
    db.release_session_turn_lease("s", first)
    row = _lease_row(db)
    assert row is not None and row["holder"] == holder, (
        "a superseded grant released the live owner's lease"
    )
    assert db.refresh_session_turn_lease("s", second, ttl_seconds=5) is True


def test_an_unversioned_holder_string_cannot_refresh_or_release(tmp_path):
    """A bare str is indistinguishable from a replay, so it must not work."""
    db = SessionDB(tmp_path / "state.db")
    _seed(db)
    holder = _holder("owner")
    grant = db.try_acquire_session_turn_lease("s", holder, ttl_seconds=5)
    assert grant

    assert db.refresh_session_turn_lease("s", str(holder), ttl_seconds=5) is False
    db.release_session_turn_lease("s", str(holder))
    row = _lease_row(db)
    assert row is not None and row["holder"] == holder


def test_the_lease_row_records_the_epoch_and_the_owner_process(tmp_path):
    """Schema: the generation and the owner's identity are durable columns."""
    db = SessionDB(tmp_path / "state.db")
    _seed(db)
    grant = db.try_acquire_session_turn_lease("s", _holder("a"), ttl_seconds=5)
    assert grant
    row = _lease_row(db)
    assert row is not None
    keys = set(row.keys())
    assert {"epoch", "owner_pid", "owner_pid_start"} <= keys, (
        f"session_turn_leases has no generation/identity columns: {sorted(keys)}"
    )
    assert int(row["epoch"]) >= 1
    assert int(row["owner_pid"]) == os.getpid()


def test_a_writer_that_predates_the_epoch_cannot_create_a_lease_row(tmp_path):
    """`epoch` is NOT NULL with no DEFAULT — that is the cutover fence.

    An older build's four-column INSERT must fail rather than land a row that
    no current writer is able to validate.
    """
    db = SessionDB(tmp_path / "state.db")
    _seed(db)
    now = time.time()
    with pytest.raises(sqlite3.IntegrityError):
        db._conn.execute(
            "INSERT INTO session_turn_leases "
            "(conversation_id, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
            ("s", _holder("pre-epoch"), now, now + 300),
        )
