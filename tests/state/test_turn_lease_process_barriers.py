"""Slice 6 — cross-process exclusion, proved with real processes and barriers.

Nothing else in the suite runs the lease across an OS process boundary. Every
other cross-process test drives ``run_agent`` with monkeypatched doubles, so
what they pin is the CALLER's handling of a lease it was handed, not that two
real processes cannot hold one conversation. That is the property the whole
design rests on, and it was unpinned.

WHY BARRIERS AND NOT SLEEPS
    The tests this replaces (lost in the branch reset) established ordering by
    sleeping and hoping. A sleep long enough to be reliable is also long enough
    to hide the defect: if the contender acquires 10ms after the owner
    released, a 200ms sleep makes a broken lease look correct. So every
    ordering fact here is carried by a ``multiprocessing`` primitive that one
    side sets and the other waits on, and *the contender reports its own
    refusal* rather than the parent inferring one from the clock.

    ``Event.wait(_LIVENESS_S)`` appears, and it is not a correctness threshold:
    its expiry is a test FAILURE ("the child never reached that point"), never
    a pass condition. No assertion anywhere reads elapsed time.

THE MUTATION CHECK
    A barrier test that cannot fail proves nothing, and this one only observes
    a refusal — the easiest thing in the world to observe by accident. So
    :func:`test_the_same_session_barrier_fails_when_exclusion_is_disabled` runs
    the identical scenario in a child that has had exclusion disabled, and
    asserts the contender DOES get in. If the barrier test above ever passes
    for the wrong reason, this one passes too and the pair contradicts itself.

Spawn, not fork: fork inherits the parent's SQLite handles and its interpreter
state, which would let a bug in the process-local registry masquerade as
correct cross-process behaviour. Spawn gives each child its own everything.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pytest

# Liveness bound only. Every use is asserted to have been reached; an expiry is
# reported as "the child never got there", not as a passing observation.
_LIVENESS_S = 60.0

_DISABLE_EXCLUSION_ENV = "HERMES_TEST_DISABLE_TURN_LEASE_EXCLUSION"


def _maybe_disable_exclusion() -> None:
    """Child-side mutation: make every lease row look free.

    Applied in the CHILD, by the test, never in production code — it is the
    negation of the property under test, used once to prove the barrier can
    kill.
    """
    if os.environ.get(_DISABLE_EXCLUSION_ENV) != "1":
        return
    from hermes_state import SessionDB

    SessionDB._turn_lease_row_is_free = lambda self, conversation_id, row: True


def _owner_child(db_path, session_id, acquired, may_release, released, out):
    """Acquire, announce, block on the barrier, release, announce."""
    from hermes_state import SessionDB

    db = SessionDB(Path(db_path))
    token = db.try_acquire_session_turn_lease(
        session_id, f"pid={os.getpid()}:turn=owner:platform=test", ttl_seconds=600
    )
    out.put(("owner_acquired", session_id, None if token is None else token.epoch))
    if token is None:
        acquired.set()
        return
    acquired.set()
    may_release.wait(_LIVENESS_S)
    db.release_session_turn_lease(session_id, token)
    out.put(("owner_released", session_id, token.epoch))
    released.set()


def _contender_child(db_path, session_id, owner_acquired, attempting, out):
    """Attempt only AFTER the owner announced, and report the first outcome.

    The first attempt is the whole test: it happens while the owner is
    provably inside its region, so its result is the answer. Everything after
    it is just waiting for the handover.
    """
    from hermes_state import SessionDB

    _maybe_disable_exclusion()
    db = SessionDB(Path(db_path))
    owner_acquired.wait(_LIVENESS_S)
    attempting.set()
    holder = f"pid={os.getpid()}:turn=contender:platform=test"
    first = db.try_acquire_session_turn_lease(session_id, holder, ttl_seconds=600)
    out.put(("first_attempt", session_id, None if first is None else first.epoch))
    if first is not None:
        db.release_session_turn_lease(session_id, first)
        return
    # Refused while the owner held it, which is the assertion. Now wait for the
    # handover so the test can also prove the lease is not wedged.
    while True:
        token = db.try_acquire_session_turn_lease(
            session_id, holder, ttl_seconds=600
        )
        if token is not None:
            out.put(("handover", session_id, token.epoch))
            db.release_session_turn_lease(session_id, token)
            return


_MISSING = "<never reported>"


def _drain(out, expected: int, seen=None) -> dict:
    """Collect up to *expected* messages, keyed by tag.

    Tolerates a message that never comes rather than raising ``queue.Empty``:
    a child that took a different branch is exactly what the assertions are
    there to describe, and an Empty traceback describes it as a test-harness
    fault instead. The missing key surfaces as :data:`_MISSING` in the failure
    message.
    """
    seen = {} if seen is None else seen
    for _ in range(expected):
        try:
            tag, session_id, epoch = out.get(timeout=_LIVENESS_S)
        except Exception:
            break
        seen[(tag, session_id)] = epoch
    return seen


def _seed(tmp_path, *session_ids):
    from hermes_state import SessionDB

    path = tmp_path / "state.db"
    db = SessionDB(path)
    for sid in session_ids:
        db.create_session(sid, source="test")
    db.close()
    return str(path)


def _run_same_session(tmp_path, *, disable_exclusion: bool):
    ctx = multiprocessing.get_context("spawn")
    path = _seed(tmp_path, "s")
    acquired = ctx.Event()
    may_release = ctx.Event()
    released = ctx.Event()
    attempting = ctx.Event()
    out = ctx.Queue()

    if disable_exclusion:
        os.environ[_DISABLE_EXCLUSION_ENV] = "1"
    try:
        owner = ctx.Process(
            target=_owner_child,
            args=(path, "s", acquired, may_release, released, out),
        )
        contender = ctx.Process(
            target=_contender_child, args=(path, "s", acquired, attempting, out)
        )
        owner.start()
        contender.start()
        try:
            assert acquired.wait(_LIVENESS_S), "the owner process never acquired"
            assert attempting.wait(_LIVENESS_S), (
                "the contender process never reached its attempt"
            )
            # The owner is blocked on may_release, which nothing has set — so
            # it is inside its region, as a fact rather than as a timing hope.
            assert not released.is_set()
            seen = _drain(out, 2)  # owner_acquired + first_attempt
            may_release.set()
            assert released.wait(_LIVENESS_S), "the owner never released"
            _drain(out, 1 if disable_exclusion else 2, seen)
            return seen
        finally:
            may_release.set()
            owner.join(_LIVENESS_S)
            contender.join(_LIVENESS_S)
            for proc in (owner, contender):
                if proc.is_alive():  # pragma: no cover - liveness escape hatch
                    proc.terminate()
                    proc.join(_LIVENESS_S)
    finally:
        os.environ.pop(_DISABLE_EXCLUSION_ENV, None)


@pytest.mark.timeout(180)
def test_two_processes_cannot_hold_one_conversation(tmp_path):
    """Owner in, contender refused, owner out, contender in — in that order."""
    seen = _run_same_session(tmp_path, disable_exclusion=False)

    assert seen.get(("owner_acquired", "s"), _MISSING) == 1, (
        f"the owner did not get the first generation: {seen}"
    )
    assert seen.get(("first_attempt", "s"), _MISSING) is None, (
        f"a second OS process acquired the same conversation while the first "
        f"was provably inside its region (its release barrier was still "
        f"unset): {seen}"
    )
    assert seen.get(("owner_released", "s"), _MISSING) == 1
    assert seen.get(("handover", "s"), _MISSING) == 2, (
        f"after the owner released, the contender should get the NEXT "
        f"generation; a repeat of epoch 1 would mean the counter restarted and "
        f"the released grant became valid again: {seen}"
    )


@pytest.mark.timeout(180)
def test_the_same_session_barrier_fails_when_exclusion_is_disabled(tmp_path):
    """The mutation: with exclusion off, the contender gets in.

    This is the test that gives the one above its teeth. Observing "the
    contender was refused" is worth nothing unless something proves the
    observation could have come out the other way, in the same harness, with
    the same barriers.
    """
    seen = _run_same_session(tmp_path, disable_exclusion=True)

    assert seen.get(("first_attempt", "s"), _MISSING) is not None, (
        "exclusion was disabled in the contender and it STILL could not "
        "acquire — so the barrier test above is not observing exclusion, it is "
        "observing something else, and it cannot fail"
    )


def _cohabiting_child(db_path, session_id, acquired, may_release, out):
    from hermes_state import SessionDB

    db = SessionDB(Path(db_path))
    token = db.try_acquire_session_turn_lease(
        session_id, f"pid={os.getpid()}:turn={session_id}:platform=test",
        ttl_seconds=600,
    )
    out.put(("acquired", session_id, None if token is None else token.epoch))
    acquired.set()
    if token is None:
        return
    # Hold it. The parent's proof of simultaneity is that BOTH children are
    # parked here, holding, at the same moment.
    may_release.wait(_LIVENESS_S)
    db.release_session_turn_lease(session_id, token)


@pytest.mark.timeout(180)
def test_two_processes_hold_two_conversations_at_the_same_time(tmp_path):
    """The lease must not serialize unrelated conversations.

    The complement of the test above, and the reason it is not enough on its
    own: a lease that refuses everything passes an exclusion test perfectly.
    Simultaneity is established structurally — both children announce while
    parked on a barrier the parent has not released — with no elapsed time
    involved.
    """
    ctx = multiprocessing.get_context("spawn")
    path = _seed(tmp_path, "s1", "s2")
    a_acquired, b_acquired = ctx.Event(), ctx.Event()
    may_release = ctx.Event()
    out = ctx.Queue()

    a = ctx.Process(
        target=_cohabiting_child, args=(path, "s1", a_acquired, may_release, out)
    )
    b = ctx.Process(
        target=_cohabiting_child, args=(path, "s2", b_acquired, may_release, out)
    )
    a.start()
    b.start()
    try:
        assert a_acquired.wait(_LIVENESS_S), "process A never reached its announce"
        assert b_acquired.wait(_LIVENESS_S), "process B never reached its announce"
        seen = _drain(out, 2)
        assert seen.get(("acquired", "s1"), _MISSING) == 1, (
            f"process A did not hold s1 while B held s2: {seen}"
        )
        assert seen.get(("acquired", "s2"), _MISSING) == 1, (
            f"process B was refused s2 while A held s1 — the lease is "
            f"serializing unrelated conversations: {seen}"
        )
    finally:
        may_release.set()
        a.join(_LIVENESS_S)
        b.join(_LIVENESS_S)
        for proc in (a, b):
            if proc.is_alive():  # pragma: no cover - liveness escape hatch
                proc.terminate()
                proc.join(_LIVENESS_S)
