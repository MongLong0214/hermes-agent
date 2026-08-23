"""Recovery rebuilds a store; it does not become a second door into one.

WHY THIS FILE EXISTS
    ``test_no_production_module_writes_a_fenced_table_outside_sessiondb`` names
    seven raw ``sessions`` writes in ``hermes_cli/session_recovery.py`` and
    ``hermes_cli/session_lost_and_found.py``. Every one of them runs on a
    connection the module opened with a bare ``sqlite3.connect``, so none of
    them is inside the transaction the turn-lease token validator sits in.

    That census is a source-shaped statement of a defect that is also LIVE. The
    recovery output is created with ``SessionDB``, so it carries this
    generation's fence triggers from the moment it exists; the raw handle that
    then writes it registered nothing. At the base commit of this work six
    recovery tests fail with ``no such function:
    hermes_turn_fence_generation`` — production line
    ``session_recovery.py:687`` (the salvage chunk insert),
    ``session_lost_and_found.py:352`` (the direct-table copy) and
    ``session_lost_and_found.py:535`` (the parent stub insert). Recovery is
    refused by the store it exists to rebuild.

THE CLOSURE THIS FILE PINS, AND THE ONE IT REFUSES
    There are exactly two closures for a module the census names: route the
    write through ``SessionDB`` so it runs on the canonical transaction, or
    refuse deterministically while a live lease exists.

    Calling ``register_turn_fence_function`` on the raw connection is NOT one of
    them and this file must never be satisfiable that way. The marker proves
    "current generation" and nothing else — not the canonical root, not the
    holder, not the monotonic epoch — so minting it on a production writer opens
    a second admitted door around the token validator. That is the original
    defect performed deliberately.

    Recovery takes BOTH closures at once, because recovery has a real tension
    the census cannot see: it must be able to rebuild a store while not becoming
    a bypass into one. So the destination is held open as a ``SessionDB`` and
    every write runs in its transaction (closure 1), and the rebuild is entered
    through a fail-closed gate that refuses while any conversation in that store
    is owned by a live turn (closure 2).

WHY THE ASSERTIONS ARE ON ROWS
    "It raised" is not the property. A fence that raises after the row moved is
    worse than none, because the writer sees an error and the store has already
    changed. Every refusal here reads the row back and asserts its VALUE, and
    the authorized case asserts the row changed — a gate that refuses everything
    passes the first half and fails the second.
"""

from __future__ import annotations

import os
import pathlib
import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import TURN_FENCE_FUNCTION_NAME, TURN_FENCE_GENERATION


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _this_generation_connection(path: pathlib.Path) -> sqlite3.Connection:
    """A raw handle that IS this generation, for building damaged fixtures.

    A test may mint the marker; that is the whole difference between a fixture
    and a production writer. This exists only to manufacture the damage that
    recovery is asked to repair — deleting session rows out from under their
    messages — which is not reachable through any SessionDB method, since every
    one of them cascades the transcript away with the owner.
    """
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.create_function(TURN_FENCE_FUNCTION_NAME, 0, lambda: TURN_FENCE_GENERATION)
    conn.execute("PRAGMA foreign_keys=OFF")
    return conn


def _open_rebuild(db: SessionDB, *, reason: str):
    """Enter the fail-closed offline rebuild gate, or fail with why it is absent.

    Resolved by name at call time rather than imported at module scope: a
    missing attribute must be a behavioural failure of this test, not an
    ImportError that stops the file from being collected. A collection error is
    not a RED — it proves the symbol is absent, not that the property is broken.
    """
    gate = getattr(db, "offline_rebuild", None)
    if gate is None:
        pytest.fail(
            "SessionDB has no offline_rebuild(): recovery has no canonical, "
            "fail-closed path to rebuild a store, so it writes one through a "
            "second raw connection that the store refuses"
        )
    return gate(reason=reason)


def _live_owned_store(path: pathlib.Path):
    """A store with one live-owned session whose title is ``before``.

    Returns the GRANT, not a holder string rebuilt from the same parts: the
    grant carries the epoch, and releasing with an unversioned holder is
    refused — which would have made the authorized half of this file pass for
    the wrong reason.
    """
    db = SessionDB(db_path=path)
    try:
        db.create_session("live", "test")
        grant = db.try_acquire_session_turn_lease(
            "live", _holder("live"), ttl_seconds=600
        )
        assert grant, "could not take the lease this proof depends on"
        # The title pair is fenced now (archived is read by prune as a
        # do-not-collect marker, so the flag family goes through the
        # same admission); the fixture holds the grant, so it presents it.
        db.set_session_title("live", "before", turn_lease_holder=grant)
        db.append_message(
            session_id="live", role="user", content="mine",
            turn_lease_holder=grant,
        )
    finally:
        db.close()
    return grant


def _title(path: pathlib.Path) -> str | None:
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute(
            "SELECT title FROM sessions WHERE id = 'live'"
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else row[0]


def test_a_rebuild_of_a_live_owned_store_leaves_the_row_unchanged(tmp_path):
    """The counterexample: recovery must not rebuild a conversation someone owns.

    Recovery's own tension, stated as a row. The lease is held by a live process
    (this one), the rebuild asks to retitle that session, and the assertion is
    on ``sessions.title`` — not on the exception. A gate that refuses AFTER the
    UPDATE lands is not a gate.
    """
    store = tmp_path / "state.db"
    _live_owned_store(store)
    assert _title(store) == "before"

    from hermes_state import SessionTurnLeaseLostError

    db = SessionDB(db_path=store)
    try:
        with pytest.raises(SessionTurnLeaseLostError) as caught:
            with _open_rebuild(db, reason="test rebuild") as rebuilt:
                def _retitle(conn: sqlite3.Connection) -> None:
                    conn.execute(
                        "UPDATE sessions SET title = ? WHERE id = ?",
                        ("rebuilt", "live"),
                    )

                rebuilt._execute_write(_retitle)
        assert "live" in str(caught.value)
    finally:
        db.close()

    assert _title(store) == "before", (
        "the rebuild was refused and the row moved anyway — the refusal "
        "happened after the write, which is worse than no refusal"
    )


def test_a_rebuild_of_a_free_store_changes_the_row_as_authorized(tmp_path):
    """The other half: a gate that refuses everything is not a gate.

    Same store, same rebuild, lease released. The row must move — otherwise the
    previous test passes on a store nobody can ever repair.
    """
    store = tmp_path / "state.db"
    holder = _live_owned_store(store)

    db = SessionDB(db_path=store)
    try:
        db.release_session_turn_lease("live", holder)
        with _open_rebuild(db, reason="test rebuild") as rebuilt:
            def _retitle(conn: sqlite3.Connection) -> None:
                conn.execute(
                    "UPDATE sessions SET title = ? WHERE id = ?",
                    ("rebuilt", "live"),
                )

            rebuilt._execute_write(_retitle)
    finally:
        db.close()

    assert _title(store) == "rebuilt"


def test_recovery_reconstructs_orphaned_owners_into_a_fenced_output(tmp_path):
    """End to end: the store recovery creates is a store recovery can write.

    The destination is created by ``SessionDB``, so it has this generation's
    triggers. Recovery then rebuilds it. At the base commit this raises
    ``no such function: hermes_turn_fence_generation`` out of
    ``session_recovery.py:687`` — the rebuilt rows are asserted here so a fix
    that merely stops raising, without landing the rows, still fails.
    """
    from hermes_cli.session_recovery import recover_session_database

    source = tmp_path / "damaged.db"
    output = tmp_path / "recovered.db"

    db = SessionDB(db_path=source)
    try:
        for session_id in ("orphan-a", "orphan-b"):
            db.create_session(session_id, "cli")
            for index in range(3):
                db.append_message(session_id, "user", f"{session_id} {index}")
    finally:
        db.close()

    damage = _this_generation_connection(source)
    try:
        damage.execute("DELETE FROM sessions")
    finally:
        damage.close()

    report = recover_session_database(
        source, output, work_dir=tmp_path, chunk_size=4, allow_partial=True
    )

    assert report["orphan_cleanup"]["messages_removed"] == 0
    assert report["orphan_cleanup"]["sessions_reconstructed"] == 2
    assert report["orphan_cleanup"]["messages_retained"] == 6

    verify = sqlite3.connect(str(output))
    try:
        rebuilt = verify.execute(
            "SELECT id, source, message_count FROM sessions ORDER BY id"
        ).fetchall()
        retained = verify.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    finally:
        verify.close()

    assert rebuilt == [("orphan-a", "recovered", 3), ("orphan-b", "recovered", 3)]
    assert retained == 6


def test_the_lost_and_found_mapper_writes_through_the_destination_store(tmp_path):
    """The page-level salvage lane takes a store, not a raw connection.

    ``map_lost_and_found_rows`` and ``stub_missing_parent_sessions`` are the
    two writers the census names in ``session_lost_and_found.py``. Handing them
    a ``SessionDB`` is what puts their INSERTs on the canonical transaction; a
    version that still wants a bare ``sqlite3.Connection`` fails here before it
    can fail at ``no such function``.
    """
    from hermes_cli.session_lost_and_found import (
        map_lost_and_found_rows,
        stub_missing_parent_sessions,
    )

    salvaged = tmp_path / "lost_and_found.db"
    lf = sqlite3.connect(str(salvaged), isolation_level=None)
    try:
        lf.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
            "role TEXT, content TEXT, timestamp REAL, active INTEGER)"
        )
        lf.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp, "
            "active) VALUES (1, 'gone', 'user', 'irreplaceable', 1.0, 1)"
        )
    finally:
        lf.close()

    output = tmp_path / "mapped.db"
    dest = SessionDB(db_path=output)
    lf_conn = sqlite3.connect(str(salvaged), isolation_level=None)
    try:
        # Through the gate, because salvage inserts children before it can
        # prove their parents exist — that ordering IS reconstruction, and the
        # store enforces foreign keys everywhere else.
        with _open_rebuild(dest, reason="lost_and_found salvage") as store:
            mapping = map_lost_and_found_rows(lf_conn, store)
            stubbing = stub_missing_parent_sessions(store)
    finally:
        lf_conn.close()
        dest.close()

    assert mapping["direct_table_rows"]["messages"] == 1
    assert stubbing["sessions_stubbed"] == 1
    assert stubbing["messages_retained"] == 1

    verify = sqlite3.connect(str(output))
    try:
        owners = verify.execute(
            "SELECT id, source FROM sessions"
        ).fetchall()
        kept = verify.execute(
            "SELECT content FROM messages ORDER BY id"
        ).fetchall()
    finally:
        verify.close()

    assert owners == [("gone", "recovered")]
    assert kept == [("irreplaceable",)]
