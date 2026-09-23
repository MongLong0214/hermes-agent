"""The receipt/meta claim primitive must be one atomic existing-handle write."""

from __future__ import annotations

import os
import sqlite3

import pytest

import hermes_state
from hermes_state import SessionDB, StateDbCorruptError, StateDbReplacedError


def _unexpected_lifecycle_work(*_args, **_kwargs):
    raise AssertionError("claim_meta_once must use the already-open SessionDB handle")


def _proven_handle_kwargs(db: SessionDB):
    identity = db._db_file_identity
    if identity is None:
        pytest.skip("filesystem does not expose st_dev/st_ino for identity checks")
    return {"proven_db_path": db.db_path, "proven_db_identity": identity}


def test_claim_meta_once_inserts_only_the_first_receipt_without_lifecycle_work(tmp_path, monkeypatch):
    """The insert-or-ignore claim decides the winner; a later receipt cannot replace it."""
    db = SessionDB(tmp_path / "state.db")
    try:
        # Arm these only after SessionDB's normal construction has established
        # the writable schema/connection.  The primitive itself owns none of
        # those lifecycle responsibilities.
        monkeypatch.setattr(hermes_state.sqlite3, "connect", _unexpected_lifecycle_work)
        monkeypatch.setattr(hermes_state.Path, "mkdir", _unexpected_lifecycle_work)
        monkeypatch.setattr(db, "_execute_write", _unexpected_lifecycle_work)
        monkeypatch.setattr(db, "_reopen_after_close_locked", _unexpected_lifecycle_work)
        monkeypatch.setattr(db, "_init_schema", _unexpected_lifecycle_work)

        statements: list[str] = []
        assert db._conn is not None
        db._conn.set_trace_callback(statements.append)
        assert db.claim_meta_once(
            "receipt:run-42", "first-receipt", **_proven_handle_kwargs(db)
        ) is True
        assert db.claim_meta_once(
            "receipt:run-42", "later-receipt", **_proven_handle_kwargs(db)
        ) is False
        db._conn.set_trace_callback(None)

        writes = [statement for statement in statements if "state_meta" in statement]
        assert len(writes) == 2
        assert all("INSERT OR IGNORE INTO state_meta" in statement for statement in writes)
        assert not any("SELECT" in statement.upper() for statement in statements)
        assert db.get_meta("receipt:run-42") == "first-receipt"
    finally:
        db.close()


def test_claim_meta_once_is_atomic_across_already_open_handles(tmp_path):
    """Two existing writable connections racing one key produce exactly one winner."""
    db_path = tmp_path / "state.db"
    first, second = SessionDB(db_path), SessionDB(db_path)
    try:
        # SQLite's INSERT OR IGNORE is the compare-and-set: neither caller
        # needs a preceding read that could race the other connection.
        first_won = first.claim_meta_once("receipt:race", "first", **_proven_handle_kwargs(first))
        second_won = second.claim_meta_once("receipt:race", "second", **_proven_handle_kwargs(second))

        assert (first_won, second_won) in ((True, False), (False, True))
        assert first.get_meta("receipt:race") in {"first", "second"}
    finally:
        first.close()
        second.close()


def test_claim_meta_once_requires_matching_caller_path_and_identity_before_sqlite_work(tmp_path):
    """A receipt claim is refused unless its caller binds it to this opened file."""
    db = SessionDB(tmp_path / "state.db")
    try:
        identity = db._db_file_identity
        if identity is None:
            pytest.skip("filesystem does not expose st_dev/st_ino for identity checks")

        statements: list[str] = []
        assert db._conn is not None
        db._conn.set_trace_callback(statements.append)
        for path, claimed_identity in (
            (db.db_path, None),
            (tmp_path / "other.db", identity),
            (db.db_path, (identity[0], identity[1] + 1)),
        ):
            with pytest.raises(sqlite3.ProgrammingError, match="matching.*path.*identity"):
                db.claim_meta_once(
                    "receipt:proof",
                    "receipt",
                    proven_db_path=path,
                    proven_db_identity=claimed_identity,
                )
        assert statements == []
    finally:
        db._conn.set_trace_callback(None)
        db.close()


def test_claim_meta_once_fences_a_stale_handle_after_its_path_is_replaced(tmp_path):
    """Cached caller proof is not authority to commit through a stale handle."""
    live_path = tmp_path / "state.db"
    replacement_path = tmp_path / "replacement.db"
    db = SessionDB(live_path)
    replacement = SessionDB(replacement_path)
    try:
        proof = _proven_handle_kwargs(db)
        replacement.close()
        os.replace(replacement_path, live_path)

        statements: list[str] = []
        assert db._conn is not None
        db._conn.set_trace_callback(statements.append)
        with pytest.raises(StateDbReplacedError, match="replaced underneath"):
            db.claim_meta_once("receipt:stale", "must-not-land", **proof)
        assert statements == []
    finally:
        if db._conn is not None:
            db._conn.set_trace_callback(None)
        db.close()


def test_compare_and_set_meta_only_promotes_the_matching_pending_receipt(tmp_path, monkeypatch):
    """A terminal receipt can replace only its own pending receipt, atomically."""
    db = SessionDB(tmp_path / "state.db")
    try:
        monkeypatch.setattr(db, "_execute_write", _unexpected_lifecycle_work)
        monkeypatch.setattr(db, "_reopen_after_close_locked", _unexpected_lifecycle_work)
        monkeypatch.setattr(db, "_init_schema", _unexpected_lifecycle_work)

        proof = _proven_handle_kwargs(db)
        assert db.claim_meta_once("receipt:transition", "pending", **proof) is True
        with pytest.raises(sqlite3.ProgrammingError, match="matching.*path.*identity"):
            db.compare_and_set_meta(
                "receipt:transition",
                "pending",
                "wrong-terminal",
                proven_db_path=db.db_path,
                proven_db_identity=(proof["proven_db_identity"][0], proof["proven_db_identity"][1] + 1),
            )

        statements: list[str] = []
        assert db._conn is not None
        db._conn.set_trace_callback(statements.append)
        assert db.compare_and_set_meta(
            "receipt:transition", "pending", "terminal", **proof
        ) is True
        assert db.compare_and_set_meta(
            "receipt:transition", "pending", "later-terminal", **proof
        ) is False
        db._conn.set_trace_callback(None)

        assert statements and all("UPDATE state_meta" in statement for statement in statements)
        assert not any("SELECT" in statement.upper() for statement in statements)
        assert db.get_meta("receipt:transition") == "terminal"
    finally:
        if db._conn is not None:
            db._conn.set_trace_callback(None)
        db.close()


def test_compare_and_set_meta_requires_matching_caller_path_and_identity_before_sqlite_work(tmp_path):
    """A terminal receipt transition is refused unless it is bound to this opened file."""
    db = SessionDB(tmp_path / "state.db")
    try:
        identity = db._db_file_identity
        if identity is None:
            pytest.skip("filesystem does not expose st_dev/st_ino for identity checks")

        statements: list[str] = []
        assert db._conn is not None
        db._conn.set_trace_callback(statements.append)
        for path, claimed_identity in (
            (db.db_path, None),
            (tmp_path / "other.db", identity),
            (db.db_path, (identity[0], identity[1] + 1)),
        ):
            with pytest.raises(sqlite3.ProgrammingError, match="matching.*path.*identity"):
                db.compare_and_set_meta(
                    "receipt:proof",
                    "pending",
                    "terminal",
                    proven_db_path=path,
                    proven_db_identity=claimed_identity,
                )
        assert statements == []
    finally:
        if db._conn is not None:
            db._conn.set_trace_callback(None)
        db.close()


def test_claim_meta_once_refuses_a_quarantined_handle_before_sqlite_dml(tmp_path):
    """A sticky structural-corruption quarantine blocks a new receipt before SQLite work."""
    db = SessionDB(tmp_path / "state.db")
    try:
        statements: list[str] = []
        assert db._conn is not None
        db._conn.set_trace_callback(statements.append)
        db._db_corrupt = True
        db._db_corrupt_reason = "forced for test"

        with pytest.raises(StateDbCorruptError, match="structural corruption"):
            db.claim_meta_once("receipt:corrupt", "must-not-land", **_proven_handle_kwargs(db))

        assert statements == []
        db._conn.set_trace_callback(None)
        assert db.get_meta("receipt:corrupt") is None
    finally:
        db._db_corrupt = False
        db._db_corrupt_reason = ""
        if db._conn is not None:
            db._conn.set_trace_callback(None)
        db.close()


def test_compare_and_set_meta_refuses_a_quarantined_handle_before_sqlite_dml(tmp_path):
    """A sticky structural-corruption quarantine blocks a receipt promotion before SQLite work."""
    db = SessionDB(tmp_path / "state.db")
    try:
        proof = _proven_handle_kwargs(db)
        assert db.claim_meta_once("receipt:corrupt-transition", "pending", **proof) is True

        statements: list[str] = []
        assert db._conn is not None
        db._conn.set_trace_callback(statements.append)
        db._db_corrupt = True
        db._db_corrupt_reason = "forced for test"

        with pytest.raises(StateDbCorruptError, match="structural corruption"):
            db.compare_and_set_meta(
                "receipt:corrupt-transition", "pending", "must-not-land", **proof
            )

        assert statements == []
        db._conn.set_trace_callback(None)
        assert db.get_meta("receipt:corrupt-transition") == "pending"
    finally:
        db._db_corrupt = False
        db._db_corrupt_reason = ""
        if db._conn is not None:
            db._conn.set_trace_callback(None)
        db.close()


def test_compare_and_set_meta_fences_a_stale_handle_after_its_path_is_replaced(tmp_path):
    """A stale descriptor cannot promote its pending receipt after a path replacement."""
    live_path = tmp_path / "state.db"
    replacement_path = tmp_path / "replacement.db"
    db = SessionDB(live_path)
    replacement = SessionDB(replacement_path)
    try:
        proof = _proven_handle_kwargs(db)
        assert db.claim_meta_once("receipt:stale-transition", "pending", **proof) is True
        replacement.close()
        os.replace(replacement_path, live_path)

        statements: list[str] = []
        assert db._conn is not None
        db._conn.set_trace_callback(statements.append)
        with pytest.raises(StateDbReplacedError, match="replaced underneath"):
            db.compare_and_set_meta(
                "receipt:stale-transition", "pending", "terminal", **proof
            )
        assert statements == []
        assert db.get_meta("receipt:stale-transition") == "pending"
    finally:
        if db._conn is not None:
            db._conn.set_trace_callback(None)
        db.close()
