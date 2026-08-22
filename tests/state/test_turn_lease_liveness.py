"""Slice 5 — elapsed time never establishes death; a live grant does.

The rule the current code breaks: **expiry is not evidence of anything.** A
lease whose ``expires_at`` has passed says the refresher did not run — a
starved thread, a stopped-world GC, a laptop that slept. It does not say the
owner is gone, and reclaiming on it hands a live turn's conversation to a
contender.

The case this file pins is the one a subprocess CANNOT exercise: a contender
that shares ``os.getpid()`` with the owner. ``_turn_lease_owner_is_dead``
answers "not dead" for our own PID (correctly — the process is right here), so
same-PID admission falls through to the clock, and the clock is the thing that
must not decide. Two ``SessionDB`` handles and one real thread in one
interpreter reproduce it with no sleeping race: the owner thread announces it is
inside the region on a ``threading.Event`` and stays there until the assertions
are done.

The replacement signal is a process-local registry of the grants THIS process
is holding. It answers three questions the clock cannot:

* registered here            → a live owner, never free, whatever the clock says
* row names this PID, not
  registered here            → positive evidence of abandonment (the holder died
                               without releasing), free, clock-independent
* foreign, alive or unknown  → never free; ``force_release`` is the recovery

Nothing here imports a symbol that may not exist yet — the file must IMPORT at
any commit so a failure is behavioural (rc 1), not a collection error (rc 2).
"""

from __future__ import annotations

import os
import threading

import pytest

from hermes_state import SessionDB


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _backdate(db: SessionDB, conversation_id: str = "s") -> None:
    """Make the lease look expired without waiting for it to expire.

    This is what a starved refresher looks like from the DB's side, and it is
    the whole point that the two are indistinguishable: if expiry can free a
    lease, a slow turn is indistinguishable from a dead one.
    """
    def _do(conn):
        conn.execute(
            "UPDATE session_turn_leases SET expires_at = 0 "
            "WHERE conversation_id = ?",
            (conversation_id,),
        )
    db._execute_write(_do)


def test_a_same_pid_contender_cannot_take_a_lease_this_process_holds(tmp_path):
    """The owner is demonstrably inside its turn; the clock says expired.

    Deterministic: the owner thread blocks on an Event that the assertions
    release, so "still inside the region" is a fact, not a timing hope.
    """
    path = tmp_path / "state.db"
    owner_db = SessionDB(path)
    contender_db = SessionDB(path)
    owner_db.create_session("s", source="test")

    inside = threading.Event()
    may_finish = threading.Event()
    held = {}

    def _owner_turn():
        held["token"] = owner_db.try_acquire_session_turn_lease(
            "s", _holder("owner"), ttl_seconds=300
        )
        inside.set()
        may_finish.wait(30)
        if held.get("token") is not None:
            owner_db.release_session_turn_lease("s", held["token"])

    thread = threading.Thread(target=_owner_turn, daemon=True)
    thread.start()
    try:
        assert inside.wait(10), "the owner thread never acquired"
        assert held["token"] is not None, "owner acquisition failed outright"
        _backdate(owner_db)

        stolen = contender_db.try_acquire_session_turn_lease(
            "s", _holder("contender"), ttl_seconds=300
        )
        assert stolen is None, (
            "a contender sharing os.getpid() with the owner took the lease "
            "because the deadline had passed. The owner is still inside its "
            "turn — this thread is provably blocked in it — so elapsed time "
            "just handed one conversation to two writers."
        )
    finally:
        may_finish.set()
        thread.join(30)


def test_a_holderless_write_is_refused_while_this_process_holds_the_lease(tmp_path):
    """The write guard has to reach the same conclusion as the acquirer.

    An admission rule that lives only in `try_acquire` is a rule the writers do
    not have: the contender that loses the acquire can simply write holderless.
    """
    path = tmp_path / "state.db"
    owner_db = SessionDB(path)
    writer_db = SessionDB(path)
    owner_db.create_session("s", source="test")

    token = owner_db.try_acquire_session_turn_lease(
        "s", _holder("owner"), ttl_seconds=300
    )
    assert token is not None
    _backdate(owner_db)

    with pytest.raises(Exception) as excinfo:
        writer_db.append_message(session_id="s", role="user", content="stolen")
    assert "lease" in str(excinfo.value).lower(), (
        f"a holderless append landed (or failed for an unrelated reason: "
        f"{excinfo.value!r}) while this process still holds the lease and only "
        f"the deadline had passed"
    )


def test_a_row_naming_this_process_with_nothing_holding_it_is_free(tmp_path):
    """Abandonment is proved positively, without consulting the clock.

    The counterpart to the first test, and the reason the first one does not
    wedge a conversation forever: a row that names THIS process while nothing
    in this process holds the grant is a turn that died without releasing.
    That is evidence, and it does not need a deadline to be true — so the
    takeover must work even when the lease is nowhere near expiry.
    """
    import hermes_state

    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("s", source="test")
    token = db.try_acquire_session_turn_lease(
        "s", _holder("crashed"), ttl_seconds=86400
    )
    assert token is not None

    # The process survived; the thing that held the grant did not.
    registry = getattr(hermes_state, "_LIVE_TURN_GRANTS", None)
    assert registry is not None, (
        "there is no process-local live-grant registry, so 'nothing here holds "
        "it' cannot be established and the only remaining signal is the clock"
    )
    lock = getattr(hermes_state, "_LIVE_TURN_GRANTS_LOCK", None)
    if lock is not None:
        with lock:
            registry.clear()
    else:  # pragma: no cover - defensive
        registry.clear()

    recovered = db.try_acquire_session_turn_lease(
        "s", _holder("recovery"), ttl_seconds=300
    )
    assert recovered is not None, (
        "the conversation is wedged: the row names a PID that is alive (ours), "
        "nothing in this process holds the grant, and the lease will not "
        "expire for a day — so no signal frees it and the session can never be "
        "written again"
    )
    assert getattr(recovered, "epoch", None) == getattr(token, "epoch", 0) + 1, (
        f"a takeover must advance the generation so the abandoned grant cannot "
        f"be replayed: {token!r} -> {recovered!r}"
    )


def test_a_foreign_owner_is_not_freed_by_its_deadline(tmp_path):
    """Expiry must stop granting freeness altogether, not just for our PID.

    A foreign PID we cannot prove dead is the case where the clock is most
    tempting and most wrong: the other process may be mid-flush behind a slow
    disk. 'Cannot prove alive' is not 'dead'.
    """
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("s", source="test")

    # A PID that exists and is not us: the parent of this process.
    foreign_pid = os.getppid()
    def _do(conn):
        conn.execute(
            "INSERT INTO session_turn_leases (conversation_id, holder, "
            "acquired_at, expires_at, epoch, owner_pid, owner_pid_start) "
            "VALUES ('s', 'pid=%d:turn=foreign:platform=test', 0, 0, 7, ?, NULL)"
            % foreign_pid,
            (foreign_pid,),
        )
    db._execute_write(_do)

    stolen = db.try_acquire_session_turn_lease(
        "s", _holder("contender"), ttl_seconds=300
    )
    assert stolen is None, (
        "a lease held by a live foreign process was reclaimed because its "
        "deadline had passed; the deadline says the refresher stalled, not "
        "that the owner is gone"
    )


def test_force_release_is_the_recovery_that_replaces_the_deadline(tmp_path):
    """Taking expiry away needs a deliberate, non-time-based way out.

    Otherwise a legacy row, or a foreign PID that got recycled into something
    long-lived, wedges the conversation with no operator recourse.
    """
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("s", source="test")
    foreign_pid = os.getppid()

    def _do(conn):
        conn.execute(
            "INSERT INTO session_turn_leases (conversation_id, holder, "
            "acquired_at, expires_at, epoch, owner_pid, owner_pid_start) "
            "VALUES ('s', 'stuck', 0, 0, 3, ?, NULL)",
            (foreign_pid,),
        )
    db._execute_write(_do)

    force = getattr(db, "force_release_session_turn_lease", None)
    assert callable(force), (
        "expiry no longer frees a lease, so there has to be an explicit "
        "recovery; without one an operator's only option is editing the "
        "database by hand"
    )
    assert force("s") is True
    took = db.try_acquire_session_turn_lease("s", _holder("after"), ttl_seconds=300)
    assert took is not None, "force_release did not actually free the lease"
    assert getattr(took, "epoch", None) == 4, (
        f"force_release must keep the generation monotonic so the forced-out "
        f"grant cannot come back: got {took!r}"
    )
