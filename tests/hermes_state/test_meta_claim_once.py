"""The canonical receipt's ``state_meta`` writes: one atomic statement on a caller-proven open handle.

``claim_meta_once`` inserts a key at most once and never overwrites; ``compare_and_set_meta`` replaces
a value only while it still equals the caller's expected one. Both refuse, before any SQL, a caller
whose proof does not name this handle's path and file identity, and a handle that is quarantined or
whose path now names another file.
"""

from __future__ import annotations

import os
import sqlite3
import threading

import pytest

import hermes_state
from hermes_state import SessionDB, StateDbCorruptError, StateDbReplacedError


def _proof(db: SessionDB) -> dict:
    identity = db._db_file_identity
    if identity is None:
        pytest.skip("filesystem does not expose st_dev/st_ino for identity checks")
    return {"proven_db_path": db.db_path, "proven_db_identity": identity}


def _no_lifecycle_work(*_args, **_kwargs):
    raise AssertionError("the meta claim must use the already-open handle, nothing else")


@pytest.fixture
def db(tmp_path):
    handle = SessionDB(tmp_path / "state.db")
    try:
        yield handle
    finally:
        if handle._conn is not None:
            handle._conn.set_trace_callback(None)
        handle._db_corrupt = False
        handle.close()


def test_claim_meta_once_claims_exactly_once_under_concurrency(tmp_path):
    """Eight writers on eight open handles race one key: exactly one claim wins, and the stored
    receipt is the winner's, never a later writer's."""
    path = tmp_path / "state.db"
    handles = [SessionDB(path) for _ in range(8)]
    start = threading.Barrier(len(handles))
    won: dict[int, bool] = {}
    errors: list[BaseException] = []

    def claim(index: int, handle: SessionDB) -> None:
        proof = _proof(handle)
        try:
            start.wait(10)
            won[index] = handle.claim_meta_once("receipt:race", f"writer-{index}", **proof)
        except BaseException as exc:  # surfaced below, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=claim, args=(i, h)) for i, h in enumerate(handles)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert errors == []
        winners = [index for index, ok in won.items() if ok]
        assert len(won) == len(handles) and len(winners) == 1
        assert {h.get_meta("receipt:race") for h in handles} == {f"writer-{winners[0]}"}
        # A claim after the race loses too, and leaves the winner's receipt in place.
        assert handles[0].claim_meta_once("receipt:race", "late", **_proof(handles[0])) is False
        assert handles[0].get_meta("receipt:race") == f"writer-{winners[0]}"
    finally:
        for handle in handles:
            handle.close()


def test_claim_meta_once_is_one_insert_or_ignore_on_the_open_handle(db, monkeypatch):
    monkeypatch.setattr(hermes_state.sqlite3, "connect", _no_lifecycle_work)
    monkeypatch.setattr(db, "_execute_write", _no_lifecycle_work)
    monkeypatch.setattr(db, "_init_schema", _no_lifecycle_work)
    statements: list[str] = []
    db._conn.set_trace_callback(statements.append)
    assert db.claim_meta_once("receipt:one", "first", **_proof(db)) is True
    assert db.claim_meta_once("receipt:one", "second", **_proof(db)) is False
    db._conn.set_trace_callback(None)
    writes = [s for s in statements if "state_meta" in s]
    assert len(writes) == 2 and all("INSERT OR IGNORE INTO state_meta" in s for s in writes)
    assert not any("SELECT" in s.upper() for s in statements)
    assert db.get_meta("receipt:one") == "first"


def test_compare_and_set_meta_loses_to_a_stale_expected_value(db):
    """A writer that read ``pending`` before another moved it to ``terminal-a`` must not overwrite
    ``terminal-a``: its expected value is stale, so it loses and the winner stands."""
    proof = _proof(db)
    assert db.claim_meta_once("receipt:cas", "pending", **proof) is True
    assert db.compare_and_set_meta("receipt:cas", "pending", "terminal-a", **proof) is True
    assert db.compare_and_set_meta("receipt:cas", "pending", "terminal-b", **proof) is False
    assert db.get_meta("receipt:cas") == "terminal-a"
    # An absent key is never created by a compare-and-set.
    assert db.compare_and_set_meta("receipt:absent", "pending", "terminal", **proof) is False
    assert db.get_meta("receipt:absent") is None


@pytest.mark.parametrize("write", ["claim", "cas"])
def test_a_proof_that_does_not_name_this_handle_is_refused_before_any_sql(db, tmp_path, write):
    identity = _proof(db)["proven_db_identity"]
    statements: list[str] = []
    db._conn.set_trace_callback(statements.append)
    for path, claimed in ((db.db_path, None), (tmp_path / "other.db", identity),
                          (db.db_path, (identity[0], identity[1] + 1))):
        kwargs = {"proven_db_path": path, "proven_db_identity": claimed}
        with pytest.raises(sqlite3.ProgrammingError, match="matching.*path.*identity"):
            if write == "claim":
                db.claim_meta_once("receipt:proof", "v", **kwargs)
            else:
                db.compare_and_set_meta("receipt:proof", "v", "w", **kwargs)
    assert statements == []


@pytest.mark.parametrize("write", ["claim", "cas"])
def test_a_quarantined_handle_is_refused_before_any_sql(db, write):
    proof = _proof(db)
    assert db.claim_meta_once("receipt:q", "pending", **proof) is True
    statements: list[str] = []
    db._conn.set_trace_callback(statements.append)
    db._db_corrupt, db._db_corrupt_reason = True, "forced for test"
    with pytest.raises(StateDbCorruptError):
        if write == "claim":
            db.claim_meta_once("receipt:q2", "v", **proof)
        else:
            db.compare_and_set_meta("receipt:q", "pending", "terminal", **proof)
    assert statements == []
    db._db_corrupt = False
    assert db.get_meta("receipt:q") == "pending" and db.get_meta("receipt:q2") is None


@pytest.mark.parametrize("write", ["claim", "cas"])
def test_a_handle_whose_path_was_replaced_is_fenced(tmp_path, write):
    live, replacement_path = tmp_path / "state.db", tmp_path / "replacement.db"
    db, replacement = SessionDB(live), SessionDB(replacement_path)
    try:
        proof = _proof(db)
        assert db.claim_meta_once("receipt:stale", "pending", **proof) is True
        replacement.close()
        os.replace(replacement_path, live)
        statements: list[str] = []
        db._conn.set_trace_callback(statements.append)
        with pytest.raises(StateDbReplacedError):
            if write == "claim":
                db.claim_meta_once("receipt:stale2", "v", **proof)
            else:
                db.compare_and_set_meta("receipt:stale", "pending", "terminal", **proof)
        assert statements == []
    finally:
        if db._conn is not None:
            db._conn.set_trace_callback(None)
        db.close()


def test_a_read_only_handle_refuses_both_writes(tmp_path):
    SessionDB(tmp_path / "state.db").close()
    ro = SessionDB(tmp_path / "state.db", read_only=True)
    try:
        proof = _proof(ro)
        with pytest.raises(sqlite3.OperationalError, match="read-only"):
            ro.claim_meta_once("receipt:ro", "v", **proof)
        with pytest.raises(sqlite3.OperationalError, match="read-only"):
            ro.compare_and_set_meta("receipt:ro", "v", "w", **proof)
    finally:
        ro.close()
