"""Regression: the async-delegation ledger must not leak a SQLite connection.

Sibling of the cron execution-ledger leak (#69567 / PR #69594). The durable
delegation ledger used ``with _connect() as conn:`` where the connection
context manager commits/rolls back but never closes, leaking the db/-wal/-shm
file descriptors on every dispatch, completion, and delivery-claim.

WHY THIS FILE CHANGED SHAPE
    The leak was fixed by closing every connection the module opened. The
    module now opens NONE: ``async_delegations`` joined ``TURN_FENCE_SURFACE``,
    and a private ``sqlite3.connect`` on the store cannot write a fenced table —
    it has not registered the generation marker, so the write is refused before
    a row is touched. The closure taken was to move the connection onto
    :meth:`SessionDB.write_transaction`; the closure NOT taken was to register
    the marker on the private handle, which would have minted a second admitted
    door around the token validator.

    So "closes every connection it opens" is now vacuously true, and a test
    that only asserted that would pass while proving nothing. The property that
    still has teeth is the one the leak actually violated — REPEATED LEDGER
    OPERATIONS MUST NOT ACCUMULATE HANDLES — and it is restated here against the
    new design, in two directions that can each fail on their own:

    1. the module opens no connection of its own (structural, read off the AST);
    2. running the public ledger operations opens no NEW connection at all once
       the store exists, so a ``SessionDB`` per call — which is the shape that
       would leak now — fails here.

    And one the old file could not have: a failure inside the transaction must
    release the store's write lock. A leaked lock is this design's version of a
    leaked descriptor, and it is worse: it wedges every writer in the process.
"""

import ast
import pathlib
import queue
import sqlite3

import pytest

from tools import async_delegation as ad

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _point_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    ad._STORE.clear()
    monkeypatch.setattr(ad, "_STORE", dict(ad._STORE), raising=False)
    return ad


def _drive_the_public_ledger():
    """Every public entry point that used to open (and leak) a connection."""
    ad.get_durable_delegation("nope")
    ad.recover_abandoned_delegations()
    ad.restore_undelivered_completions(queue.Queue())
    ad.mark_completion_delivered("nope")
    ad.claim_completion_delivery("nope", "claim-1")


def test_the_ledger_opens_no_sqlite_connection_of_its_own():
    """Structural half: there is no ``sqlite3.connect`` left in the module.

    Read off the AST rather than the text, so a mention in a docstring — this
    module has several, explaining why the handle went away — cannot satisfy or
    break it. A private handle reappearing here is not only an fd-leak risk: it
    is a second writer on a fenced store, which is the defect the barrier
    exists to stop.
    """
    source = (REPO_ROOT / "tools" / "async_delegation.py").read_text(
        encoding="utf-8"
    )
    opens = [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "connect"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "sqlite3"
    ]
    assert not opens, (
        f"tools/async_delegation.py opens its own SQLite connection at "
        f"line(s) {opens}. Its writes must run on SessionDB's transaction: a "
        f"private handle both re-opens the descriptor leak this file was "
        f"written for and puts a second, unmarked writer on a fenced store."
    )


def test_repeated_ledger_operations_open_no_new_connection(
    monkeypatch, tmp_path
):
    """Runtime half: after the store exists, the ledger opens nothing.

    This is the assertion the original leak would have failed, restated for a
    module that borrows a connection instead of owning one. A ``SessionDB``
    built per call — today's version of "connect and forget to close" — opens a
    connection per call and fails here.
    """
    _point_ledger(monkeypatch, tmp_path)
    # Build the store OUTSIDE the measurement: its construction legitimately
    # connects, and counting that would assert the opposite of what this is for.
    store = ad._session_store()

    opened = []
    real_connect = sqlite3.connect

    def counting_connect(*args, **kwargs):
        opened.append(args[0] if args else kwargs.get("database"))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", counting_connect)
    for _ in range(3):
        _drive_the_public_ledger()
    monkeypatch.setattr(sqlite3, "connect", real_connect)

    assert opened == [], (
        f"the ledger opened {len(opened)} connection(s) while running its "
        f"public operations three times: {opened}. Every write runs on the "
        f"store's own transaction, so the only way to open one is to rebuild "
        f"the store per call — which leaks exactly what this file was written "
        f"about."
    )
    assert ad._session_store() is store, (
        "the admission store was rebuilt; it is a per-process singleton and "
        "rebuilding it per operation is the leak in its new form"
    )


def test_a_failure_inside_the_transaction_releases_the_stores_write_lock(
    monkeypatch, tmp_path
):
    """A raise inside the block must roll back AND hand the lock back.

    The connection is no longer this module's to leak; the STORE'S WRITE LOCK
    is. Holding it after an exception wedges every writer in the process — a
    strictly worse outcome than the descriptor leak this file started as — and
    it would not show up as an error anywhere, only as a hang.
    """
    _point_ledger(monkeypatch, tmp_path)
    store = ad._session_store()

    with pytest.raises(RuntimeError):
        with ad._transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO async_delegations
                   (delegation_id, origin_session, origin_ui_session_id,
                    parent_session_id, state, dispatched_at, updated_at,
                    delivery_state, delivery_attempts)
                   VALUES ('rolled-back', '', '', NULL, 'running', 1.0, 1.0,
                           'pending', 0)"""
            )
            raise RuntimeError("simulated failure inside the transaction")

    acquired = store._lock.acquire(blocking=False)
    if acquired:
        store._lock.release()
    assert acquired, (
        "the store's write lock was still held after the transaction raised; "
        "every subsequent write in this process would block forever"
    )
    assert ad.get_durable_delegation("rolled-back") is None, (
        "the row written before the exception was committed; the transaction "
        "did not roll back"
    )
    # And the store is still usable, which is the observable consequence of
    # both of the above being true.
    with ad._transaction() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                parent_session_id, state, dispatched_at, updated_at,
                delivery_state, delivery_attempts)
               VALUES ('after', '', '', NULL, 'running', 1.0, 1.0,
                       'pending', 0)"""
        )
    assert ad.get_durable_delegation("after") is not None
