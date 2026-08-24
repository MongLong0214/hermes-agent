"""Slice 7 — deletion is the most complete way to change what a turn replays.

Slice 4 fenced every CALL SITE of ``delete_session``. That is not the same as
fencing ``delete_session``, and the difference is the whole point of putting
the rule in the write transaction rather than in the callers: a call site is a
place someone has to remember, and the next caller is written by someone who
never read this file. ``append_message`` learned that in slice 3. ``delete_*``
has not.

Three properties, none of which the current code has:

1. A holderless ``delete_session`` on a conversation a live turn owns is
   refused. Every other context-bearing mutator already refuses it; deletion
   removes the rows outright, so it is the one where being wrong is least
   recoverable.
2. The DELEGATE CASCADE is checked too. ``delete_session`` re-walks the
   delegate tree inside its transaction and deletes those children with the
   parent — they are separate conversations with separate leases, and a grant
   on the parent says nothing about them. Guarding only ``session_id`` fences
   the row the operator named and none of the rows it takes with it.
3. ``take_unseen_reactions`` is consumed exactly once under contention. It
   stamps rows seen, and the announcement it returns is injected into one
   turn's model context — so two consumers racing means one turn's
   announcement is silently delivered to the other.

Nothing here imports a symbol that may not exist yet, so the file IMPORTS at
any commit and every failure is behavioural (rc 1) rather than a collection
error (rc 2).
"""

from __future__ import annotations

import os
import threading

import pytest

from hermes_state import SessionDB, SessionTurnLeaseLostError


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _foreign_owner(db: SessionDB, conversation_id: str, *, epoch: int = 5) -> None:
    """Install a lease row owned by a LIVE process that is not us.

    os.getppid() is alive by construction (it is our parent) and is not our
    PID, so the liveness rules classify it as a foreign live owner without any
    dependence on timing or on a sleep.
    """
    def _do(conn):
        conn.execute(
            "INSERT OR REPLACE INTO session_turn_leases (conversation_id, "
            "holder, acquired_at, expires_at, epoch, owner_pid, "
            "owner_pid_start) VALUES (?, ?, 0, 4102444800, ?, ?, NULL)",
            (conversation_id, f"pid={os.getppid()}:turn=foreign:platform=test",
             epoch, os.getppid()),
        )
    db._execute_write(_do)


def _session_exists(db: SessionDB, session_id: str) -> bool:
    with db._read_ctx() as conn:
        return conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone() is not None


def test_a_holderless_delete_is_refused_on_an_owned_conversation(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="test")
    db.append_message(session_id="s", role="user", content="mine")
    _foreign_owner(db, "s")

    # The exact type, not "some exception whose text mentions a lease": a
    # TypeError from an unknown keyword also mentions turn_lease_holder, and
    # that near-miss made an earlier draft of this file pass for the wrong
    # reason.
    with pytest.raises(SessionTurnLeaseLostError):
        db.delete_session("s")
    assert _session_exists(db, "s"), "the session row is gone"


def test_the_owner_can_still_delete_its_own_conversation(tmp_path):
    """The guard must admit the owner, or /exit --delete can never run."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="test")
    grant = db.try_acquire_session_turn_lease("s", _holder("owner"), ttl_seconds=600)
    assert grant is not None

    assert db.delete_session("s", turn_lease_holder=grant) is True, (
        "the conversation's own owner was refused its delete — the fence has "
        "to admit the grant, not just reject everything"
    )
    assert not _session_exists(db, "s")


def test_a_delete_is_refused_when_a_CASCADED_DELEGATE_is_owned(tmp_path):
    """The rows a delete takes with it need the same admission as the one named.

    A delegate child is a separate conversation with its own lease. The
    operator's grant on the parent authorizes nothing about it, and the parent
    delete removes it outright.
    """
    db = SessionDB(tmp_path / "state.db")
    db.create_session("parent", source="test")
    db.create_session(
        "delegate",
        source="delegate",
        parent_session_id="parent",
        model_config={"_delegate_from": "parent"},
    )
    db.append_message(session_id="delegate", role="user", content="live work")
    parent_grant = db.try_acquire_session_turn_lease(
        "parent", _holder("operator"), ttl_seconds=600
    )
    assert parent_grant is not None
    _foreign_owner(db, "delegate", epoch=9)

    with pytest.raises(SessionTurnLeaseLostError):
        db.delete_session("parent", turn_lease_holder=parent_grant)
    assert _session_exists(db, "delegate"), "the delegate's rows are gone"
    assert _session_exists(db, "parent"), (
        "the parent went even though the cascade was refused — a partially "
        "applied destructive delete is worse than either outcome"
    )


def test_an_unowned_delegate_still_cascades(tmp_path):
    """The cascade guard must not break the ordinary case."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("parent", source="test")
    db.create_session(
        "delegate",
        source="delegate",
        parent_session_id="parent",
        model_config={"_delegate_from": "parent"},
    )

    assert db.delete_session("parent") is True
    assert not _session_exists(db, "parent")
    assert not _session_exists(db, "delegate"), (
        "the delegate cascade stopped working; an orphaned delegate resurfaces "
        "in session pickers"
    )


def test_take_unseen_reactions_is_consumed_by_exactly_one_of_two_racers(tmp_path):
    """Consume-once, proved by contention rather than by a single call.

    Both threads call it on the same conversation. The announcement is injected
    into ONE turn's model context, so if both come back with the reaction, one
    turn silently loses it — and a single-threaded call can never show that.
    """
    path = tmp_path / "state.db"
    seeder = SessionDB(path)
    seeder.create_session("s", source="test")
    seeder.append_message(session_id="s", role="user", content="hello")
    row_id = seeder.latest_message_row_id("s", role="user")
    assert row_id is not None
    seeder.set_message_reaction("s", row_id, "\U0001f44d", author="user")

    handles = [SessionDB(path), SessionDB(path)]
    start = threading.Barrier(len(handles))
    results: list = []
    lock = threading.Lock()

    def _consume(db):
        start.wait(30)
        taken = db.take_unseen_reactions("s", author="user")
        with lock:
            results.append(len(taken))

    threads = [threading.Thread(target=_consume, args=(h,)) for h in handles]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
        assert not t.is_alive(), "a consumer thread never finished"

    assert sorted(results) == [0, 1], (
        f"the announcement was consumed {sum(results)} times by {len(results)} "
        f"racers; exactly one turn may receive it: {results}"
    )
