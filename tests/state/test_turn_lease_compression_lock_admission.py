"""Taking the compression lock is a decision about someone else's conversation.

WHAT WAS MEASURED, AND WHY THE GROUND I FIRST OFFERED DOES NOT HOLD
    ``compression_locks`` entered ``TURN_FENCE_SURFACE`` because it is written
    inside a transaction that consults the turn-lease admission. That closed it
    against an OLD BINARY. It did not close it against a process of THIS
    generation, and I first argued it did not need to be: every statement the
    three lock writers issue is either ``holder = ?``-scoped or an
    ``INSERT OR IGNORE`` that cannot overwrite a row, so — the argument went —
    the lock's own token is the mutual exclusion and a caller with no token can
    neither steal a lock nor free one.

    That argument is correct for RELEASE and REFRESH and wrong for ACQUIRE, and
    the difference is not subtle. ``INSERT OR IGNORE`` is first-writer-wins.
    Measured against a store this generation created, with a live turn lease
    held on ``s`` and a second ``SessionDB`` holding NO grant for it::

        {"bystander_acquire": true,
         "lock_holder_after_bystander": "BYSTANDER",
         "owner_acquire_after": false,
         "lock_holder_final": "BYSTANDER"}

    The bystander took the lock on a conversation it does not own, and the
    OWNER then could not compress. "Cannot steal a lock somebody holds" was
    never the question — the question is who may take one that nobody holds on
    a conversation somebody owns, and the answer was anybody.

WHY THIS IS THE SAME DECISION THE COOLDOWN COLUMNS ALREADY TAKE
    This branch already fences ``compression_failure_cooldown_until`` and its
    siblings, on the reasoning that a bystander must not decide whether the
    owner's conversation may compact. Taking the lock decides that more
    directly and more completely: the cooldown delays the owner's compression,
    the lock prevents it outright for the lock's whole TTL. Fencing the column
    and grounding the lock draws the line in a place that cannot be defended,
    so the line moves to where the capability is.

WHAT IS FENCED AND WHAT DELIBERATELY IS NOT
    acquire   ADMITTED. Root, holder and epoch for the conversation, resolved
              inside the same transaction as the INSERT, exactly as every other
              writer on the surface does it.
    refresh   token-scoped, unchanged. ``WHERE session_id = ? AND holder = ?``:
              extending a lease you already hold is not a decision about
              anybody else, and requiring a turn grant to refresh would refuse
              a compressor whose grant rotated mid-compression — a fence on the
              owner, which is a different failure.
    release   token-scoped, unchanged, for the same reason. The token IS the
              authority there, and that arm of the ground was never in doubt:
              the measurement above shows the bystander's release of its OWN
              lock leaving the table empty, touching nothing it did not hold.

    So the change is one guard on one method, and this file's pin is the
    counterexample above turned into rows.
"""

from __future__ import annotations

import json
import os
import pathlib
import sqlite3

import pytest

from tests.state.lease_mutation_harness import (
    Mutation,
    assert_every_pin_has_a_killer,
    assert_mutation_kills_the_pin,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)
_EXTRA_EXTRACT = (".",)

SESSION_ID = "s"


def _lock_rows(store):
    conn = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        return sorted(
            tuple(row) for row in conn.execute(
                "SELECT session_id, holder FROM compression_locks"
            )
        )
    finally:
        conn.close()


def _acquire(db, holder, *, grant=None, ttl_seconds=600.0):
    """Take the lock and return the OUTCOME rather than raising.

    Every failure mode becomes a value: the refusal this file is about, and
    also a ``TypeError`` from the admission parameter not existing yet. That
    matters for the RED — without it the pin dies on the signature before it
    ever reaches its own assertions, and a signature error is not the
    counterexample. With it, the RED fails saying the bystander took the lock
    and the owner then could not, which is the finding.
    """
    from hermes_state import SessionTurnLeaseLostError

    kwargs = {"ttl_seconds": ttl_seconds}
    if grant is not None:
        kwargs["turn_lease_holder"] = grant
    try:
        return db.try_acquire_compression_lock(SESSION_ID, holder, **kwargs)
    except SessionTurnLeaseLostError as exc:
        return f"refused: {exc}"
    except TypeError as exc:
        return f"no admission parameter: {exc}"


def _live_owned_store(tmp_path):
    from hermes_state import SessionDB

    tmp_path = pathlib.Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = tmp_path / "state.db"
    owner = SessionDB(store)
    owner.create_session(SESSION_ID, source="test")
    grant = owner.try_acquire_session_turn_lease(
        SESSION_ID, f"pid={os.getpid()}:turn=live:platform=test", ttl_seconds=600
    )
    assert grant, "could not take the lease this measurement depends on"
    owner.append_message(
        session_id=SESSION_ID, role="user", content="one", turn_lease_holder=grant
    )
    return owner, store, grant


def check_a_bystander_cannot_take_the_owners_compression_lock(tmp_path) -> None:
    """The measured counterexample, as rows and as a capability.

    Three assertions, because "it was refused" is satisfied by all sorts of
    things that are not this property:

      the rows        ``compression_locks`` is exactly what the owner left.
      the capability  the OWNER can still acquire afterwards. This is the one
                      that would have caught the defect even if the refusal had
                      been silent — the harm was never "a row appeared", it was
                      "the owner cannot compress".
      the outcome     the bystander's call did not return success.
    """
    from hermes_state import SessionDB

    owner, store, grant = _live_owned_store(tmp_path)
    outcome = {}
    try:
        before = _lock_rows(store)

        bystander = SessionDB(store)
        try:
            # No grant presented, because it has none — this is the whole of
            # what a bystander is.
            outcome["bystander_acquire"] = _acquire(bystander, "BYSTANDER")
        finally:
            bystander.close()

        after = _lock_rows(store)
        outcome["lock_rows_after_bystander"] = after

        # The capability arm: can the holder of the turn still compress?
        outcome["owner_acquire_after"] = _acquire(owner, "OWNER", grant=grant)
        outcome["lock_rows_final"] = _lock_rows(store)
    finally:
        owner.close()

    assert after == before, (
        f"a CURRENT-GENERATION process holding no grant for {SESSION_ID!r} "
        f"wrote compression_locks while a live turn owned the conversation.\n"
        f"  before: {before}\n  after:  {after}\n"
        f"  {json.dumps(outcome, default=str, indent=4, sort_keys=True)}"
    )
    assert outcome["owner_acquire_after"] is True, (
        f"the turn's own holder cannot take the compression lock on its own "
        f"conversation. INSERT OR IGNORE is first-writer-wins, so a bystander "
        f"that got there first denies the owner its compression for the whole "
        f"lock TTL — the harm is the lost capability, not the extra row.\n"
        f"  {json.dumps(outcome, default=str, indent=4, sort_keys=True)}"
    )
    assert outcome["bystander_acquire"] is not True, (
        f"the bystander's acquire SUCCEEDED.\n"
        f"  {json.dumps(outcome, default=str, indent=4, sort_keys=True)}"
    )
    assert outcome["lock_rows_final"] == [(SESSION_ID, "OWNER")], (
        f"the owner's lock is not the one in the table: "
        f"{outcome['lock_rows_final']}"
    )


def check_the_token_scoped_arms_stay_open_for_their_holder(tmp_path) -> None:
    """Refresh and release are NOT fenced, and this states why in rows.

    A compressor extends or drops a lease it already holds; the token it
    presents is the authority, and demanding a turn grant there would refuse a
    compressor whose grant rotated mid-compression. So the owner takes the
    lock under its grant, then refreshes and releases with the TOKEN alone —
    no grant presented — and both must work.

    Stated as a pin rather than left implicit, because "we fenced the lock
    table" reads as if all three writers moved, and only one did.
    """
    from hermes_state import SessionDB

    owner, store, grant = _live_owned_store(tmp_path)
    try:
        taken = _acquire(owner, "OWNER", grant=grant)
        assert taken is True, (
            f"the owner could not take its own compression lock: {taken!r}"
        )

        # No grant presented on either of these.
        assert owner.refresh_compression_lock(
            SESSION_ID, "OWNER", ttl_seconds=900
        ) is True, (
            "refresh was refused for the holder of the lock. It is token-scoped "
            "on purpose: a compressor whose turn grant rotated mid-compression "
            "must still be able to extend the lease it holds"
        )
        assert _lock_rows(store) == [(SESSION_ID, "OWNER")]

        owner.release_compression_lock(SESSION_ID, "OWNER")
        assert _lock_rows(store) == [], (
            "release was refused for the holder of the lock; the token is the "
            "authority on that arm and always was"
        )

        # And a WRONG token still cannot free it — the ground that did hold.
        assert _acquire(owner, "OWNER-2", grant=grant) is True
        owner.release_compression_lock(SESSION_ID, "SOMEONE-ELSE")
        assert _lock_rows(store) == [(SESSION_ID, "OWNER-2")], (
            "a release presenting the wrong token freed the lock"
        )
    finally:
        owner.close()


PINS = {
    "check_a_bystander_cannot_take_the_owners_compression_lock":
        check_a_bystander_cannot_take_the_owners_compression_lock,
    "check_the_token_scoped_arms_stay_open_for_their_holder":
        check_the_token_scoped_arms_stay_open_for_their_holder,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_compression_lock_admission_property(name, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    PINS[name](tmp_path / "work")


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_a_bystander_cannot_take_the_owners_compression_lock",
        module="hermes_state.py",
        find=(
            "            self._check_turn_lease_guard(\n"
            "                conn,\n"
            "                session_id,\n"
            "                turn_lease_holder,\n"
            "                turn_lease_ttl_seconds=turn_lease_ttl_seconds,\n"
            "            )\n"
            "            reclaimed_holder = None\n"
        ),
        replace="            reclaimed_holder = None\n",
        why="without it the acquire is first-writer-wins on any conversation: "
            "a process of this generation holding no grant takes the lock and "
            "the owner cannot compress for the lock's whole TTL. Measured "
            "before the guard existed as bystander_acquire=true, "
            "owner_acquire_after=false",
    ),
    Mutation(
        pin="check_the_token_scoped_arms_stay_open_for_their_holder",
        module="hermes_state.py",
        find=(
            '                "UPDATE compression_locks SET expires_at = ? "\n'
            '                "WHERE session_id = ? AND holder = ?",\n'
        ),
        replace=(
            '                "UPDATE compression_locks SET expires_at = ? "\n'
            '                "WHERE session_id = ? AND holder != ?",\n'
        ),
        why="refresh is token-scoped and that is the whole of its authority. "
            "Inverted, the holder's own refresh matches nothing and returns "
            "False, which is the fence-on-the-owner failure this pin exists to "
            "tell apart from a working one",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    assert_mutation_kills_the_pin(
        mutation, str(_SELF), tmp_path, *_EXTRA_EXTRACT
    )


def test_every_pin_has_a_mutation_that_kills_it():
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)
