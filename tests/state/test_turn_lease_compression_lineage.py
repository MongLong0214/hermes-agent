"""Blocker (d): the compression publish path, pinned and mutation-killed.

WHY THIS FILE EXISTS
    Blocker (d) was probed by hand and reported "already true" — a stale grant
    could not publish or compact, a foreign-root grant was refused, a holderless
    publish was refused while an owner was live, and the lease key walked to the
    conversation root even three compressions deep.

    Every one of those was correct. None of them was EVIDENCE IN THE REPO. A
    property nothing checks is a property that survives until someone edits the
    line that provides it, and the whole reason blocker (b) existed is that
    exactly this had happened to the transcript-append guard: the rule was
    there, the rule was right, and the writers that did not go through it were
    admitted for months.

WHY EACH PIN IS MUTATION-KILLED
    A pin that passes is not a pin. It is a pin only if there is a change to
    the source that makes it fail — otherwise it might be asserting something
    the code cannot do anyway, and it will keep passing after the guard is
    deleted.

    So each property here appears twice: once as a test that asserts it, and
    once as a row of :data:`SOURCE_MUTATIONS` that deletes the guard providing
    it and REQUIRES the same check to fail. The mutation table is executed, not
    remembered — the failure mode being designed out is a human meaning to
    re-run the mutation and not doing it.

    Three properties of the harness, each because the obvious version is wrong:

    * Mutations are keyed by an exact source SUBSTRING that must appear exactly
      once, never by a line number. A check that names a line number goes stale
      the moment anything above it grows, and it goes stale silently.
    * Every row extracts its OWN tree and runs with ``PYTHONDONTWRITEBYTECODE``.
      A shared directory lets row 2 import row 1's compiled module and report a
      removed guard as covered.
    * Each row runs the check THREE times — clean, mutated, restored. Clean
      proves the extract works at all; restored proves the failure came from the
      mutation rather than from a broken fixture.

WHY THE CHECKS ARE FUNCTIONS AND NOT ONLY TESTS
    The mutated run has to execute the same assertions the pin does, in a
    subprocess against a different tree. So each property is a module-level
    ``check_*`` function in :data:`PINS`, called by a thin test here and by the
    mutation harness there. Deleting a check kills both halves at once; there
    is no version of this file where the pin and the thing being mutation-tested
    can drift apart.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from tests.state.lease_mutation_harness import (
    Mutation,
    assert_every_pin_has_a_killer,
    assert_mutation_kills_the_pin,
    assert_the_owner_was_not_refused,
    owner_call,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: This file, so a mutated extract can run the same checks it defines.
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _store(tmpdir, name="state.db"):
    from hermes_state import SessionDB

    return SessionDB(db_path=pathlib.Path(tmpdir) / name)


def _publish(db, parent, child, grant, *, lock="compressor"):
    """Publish a compression child of *parent*, presenting *grant*.

    The compression lock is REUSED when this holder already has it. A refused
    publish leaves the lock held, and a second attempt that quietly failed to
    take it would make every "and the authorized case still works" half of
    these checks fail for a reason that has nothing to do with the turn lease.
    """
    if db.get_compression_lock_holder(parent) != lock:
        # The ACQUIRE is admitted now too, so it presents the conversation's
        # CURRENT grant — `current_turn_grant` is the registry the production
        # compressor reads — and NOT the grant under test.
        #
        # This is load-bearing rather than tidy. Handing the grant under test
        # to the acquire makes a refused ACQUIRE raise the same exception a
        # refused PUBLISH does, one call earlier, and every check below would
        # go on passing while never reaching the publish guard it names. That
        # is a pin that has quietly stopped measuring anything — the failure
        # mode this whole family exists to rule out, arriving through a fence
        # added in front of the write rather than through a second guard
        # beside it.
        assert db.try_acquire_compression_lock(
            parent, lock, ttl_seconds=60,
            turn_lease_holder=db.current_turn_grant(parent),
        ), (
            f"could not take the compression lock on {parent!r}; held by "
            f"{db.get_compression_lock_holder(parent)!r}"
        )
    db.publish_compression_child(
        parent_session_id=parent,
        child_session_id=child,
        source="test",
        messages=[{"role": "user", "content": f"summary of {parent}"}],
        compression_lock_holder=lock,
        turn_lease_holder=grant,
    )


def _session_snapshot(db, session_id):
    """The row values a refused publish must leave alone."""
    row = db.get_session(session_id)
    if row is None:
        return None
    return (row["id"], row["ended_at"], row["end_reason"])


# ---------------------------------------------------------------------------
# The properties. Each is callable from a test and from a mutated subprocess.
# ---------------------------------------------------------------------------

def check_a_superseded_grant_cannot_publish(tmpdir) -> None:
    """A grant from an earlier epoch cannot rotate the conversation.

    Same holder string, older epoch. The holder is not the discriminator — a
    process that re-acquires after releasing has the same identity and a new
    generation, and the old grant is a replay. The assertion is on the ROWS:
    the parent must still be live and the child must not exist.
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        db.create_session("root", "test")
        holder = _holder("compressor")
        stale = db.try_acquire_session_turn_lease("root", holder, ttl_seconds=600)
        assert stale, "could not take the first lease"
        db.release_session_turn_lease("root", stale)
        current = db.try_acquire_session_turn_lease("root", holder, ttl_seconds=600)
        assert current, "could not re-take the lease"
        assert current.epoch > stale.epoch, (
            f"the re-acquisition did not advance the epoch "
            f"({stale.epoch} -> {current.epoch}); this check would pass "
            f"vacuously"
        )

        before = _session_snapshot(db, "root")
        try:
            _publish(db, "root", "stale-child", stale)
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a superseded grant published a compression child"
            )
        assert _session_snapshot(db, "root") == before, (
            "the stale grant was refused and the parent moved anyway"
        )
        assert db.get_session("stale-child") is None, (
            "the stale grant was refused and its child exists"
        )

        # And the current grant DOES publish — otherwise the refusal above is
        # indistinguishable from a path that refuses everybody.
        _publish(db, "root", "live-child", current)
        assert db.get_session("live-child") is not None
        assert db.get_session("root")["end_reason"] == "compression"
    finally:
        db.close()


def check_a_foreign_root_grant_cannot_publish(tmpdir) -> None:
    """A perfectly live grant for ANOTHER conversation is not authority here.

    Holder and epoch order acquisitions within one conversation and say
    nothing across conversations — a first grant is epoch 1 everywhere, so
    without the root comparison the epoch discriminates nothing at all.

    BOTH GRANTS USE THE SAME HOLDER STRING, AND THAT IS THE WHOLE TEST. The
    first version of this check gave them different holders, and it survived
    its own mutation: with the root comparison deleted the write was still
    refused, by the ``row["holder"] != str(token)`` comparison one line below.
    It passed while asserting nothing about the property it names. One process
    holding the lease on two conversations under one identity is the case that
    isolates the root, and it is not exotic — a compressor working two sessions
    composes the same holder for both.
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        db.create_session("mine", "test")
        db.create_session("theirs", "test")
        one_identity = _holder("compressor")
        foreign = db.try_acquire_session_turn_lease(
            "theirs", one_identity, ttl_seconds=600
        )
        assert foreign, "could not take the foreign lease"
        mine = db.try_acquire_session_turn_lease(
            "mine", one_identity, ttl_seconds=600
        )
        assert mine, "could not take my own lease"
        assert str(foreign) == str(mine), (
            "the two grants must be the same holder string, or the holder "
            "comparison refuses this write and the ROOT comparison is never "
            "exercised"
        )
        assert foreign.epoch == mine.epoch, (
            "both first grants should be the same epoch; if they are not, this "
            "check no longer isolates the ROOT comparison"
        )
        assert foreign.conversation_id != mine.conversation_id

        before = _session_snapshot(db, "mine")
        try:
            _publish(db, "mine", "smuggled", foreign)
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a grant for another conversation published here"
            )
        assert _session_snapshot(db, "mine") == before
        assert db.get_session("smuggled") is None

        _publish(db, "mine", "legitimate", mine)
        assert db.get_session("legitimate") is not None
    finally:
        db.close()


def check_a_holderless_publish_is_refused_while_owned(tmpdir) -> None:
    """Presenting nothing is admitted on a FREE conversation and only there.

    Both halves matter. Refusing every holderless write breaks fresh sessions,
    imports and single-writer installs; admitting them unconditionally is the
    hole that let a second writer rotate a conversation somebody was mid-turn
    on. So: refused while owned, admitted once released.
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        db.create_session("owned", "test")
        grant = db.try_acquire_session_turn_lease(
            "owned", _holder("owner"), ttl_seconds=600
        )
        assert grant, "could not take the lease"

        before = _session_snapshot(db, "owned")
        try:
            _publish(db, "owned", "holderless-child", None)
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a holderless publish rotated a live-owned conversation"
            )
        assert _session_snapshot(db, "owned") == before
        assert db.get_session("holderless-child") is None

        db.release_session_turn_lease("owned", grant)
        _publish(db, "owned", "free-child", None)
        assert db.get_session("free-child") is not None, (
            "holderless writes are refused even on a FREE conversation, which "
            "breaks imports, fresh sessions and single-writer installs"
        )
    finally:
        db.close()


def check_the_lease_key_walks_to_the_root_three_deep(tmpdir) -> None:
    """One grant, taken at the root, still authorizes the fourth segment.

    Compression rotates a conversation onto a new session id. If the lease key
    were the session id, every rotation would silently free the conversation
    mid-turn and the next writer would be admitted. The key is the ROOT, and
    the walk has to survive more than one hop — a walk that stops after one
    parent passes a two-segment test and fails here.
    """
    db = _store(tmpdir)
    try:
        db.create_session("root", "test")
        grant = db.try_acquire_session_turn_lease(
            "root", _holder("compressor"), ttl_seconds=600
        )
        assert grant, "could not take the root lease"
        assert grant.conversation_id == "root"

        chain = ["root", "c1", "c2", "c3"]
        for index in range(3):
            # Each hop is the ROOT's grant publishing the next segment, and
            # each one is a place the property can fail: with the lease-key
            # walk gone, `c1` no longer resolves to `root` and the second hop
            # is refused. Stated as an assertion rather than left to raise —
            # a pin that dies building its fixture cannot be told apart from
            # a pin whose fixture is broken.
            assert_the_owner_was_not_refused(
                owner_call(lambda i=index: _publish(
                    db, chain[i], chain[i + 1], grant,
                    lock=f"compressor-{i}",
                )),
                f"publishing segment {chain[index + 1]!r} under the root's "
                f"own grant — the lease key for {chain[index]!r} must still "
                f"walk to 'root'",
            )

        # Three hops from the tail back to the root, resolved by production.
        assert db._session_turn_lease_key("c3") == "root", (
            f"the lease key for the fourth segment resolved to "
            f"{db._session_turn_lease_key('c3')!r}, not the conversation root"
        )

        # The lease row never moved off the root, so nothing was freed.
        with db._read_ctx() as conn:
            keys = sorted(
                str(row[0]) for row in conn.execute(
                    "SELECT conversation_id FROM session_turn_leases"
                )
            )
        assert keys == ["root"], (
            f"rotation created lease rows for {keys}; each rotation that mints "
            f"a new key frees the conversation mid-turn"
        )

        # And the same root grant still publishes from three deep.
        assert_the_owner_was_not_refused(
            owner_call(lambda: _publish(db, "c3", "c4", grant)),
            "publish from three segments deep under the root's own grant",
        )
        assert db.get_session("c4") is not None
    finally:
        db.close()


#: Name -> callable. The single place both harnesses read from.
PINS = {
    "check_a_superseded_grant_cannot_publish":
        check_a_superseded_grant_cannot_publish,
    "check_a_foreign_root_grant_cannot_publish":
        check_a_foreign_root_grant_cannot_publish,
    "check_a_holderless_publish_is_refused_while_owned":
        check_a_holderless_publish_is_refused_while_owned,
    "check_the_lease_key_walks_to_the_root_three_deep":
        check_the_lease_key_walks_to_the_root_three_deep,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_blocker_d_property(name, tmp_path):
    """The pin. Each property, asserted against the tree under test."""
    PINS[name](tmp_path)


# ---------------------------------------------------------------------------
# The mutation table: for each pin, the source edit that must kill it.
# ---------------------------------------------------------------------------

SOURCE_MUTATIONS = (
    Mutation(
        pin="check_a_superseded_grant_cannot_publish",
        module="hermes_state.py",
        find='            or int(row["epoch"] or 0) != epoch\n',
        replace="",
        why="the epoch comparison in _authorize_turn_lease_token is the only "
            "thing distinguishing a superseded grant from the current one, "
            "because the holder string is identical",
    ),
    Mutation(
        pin="check_a_foreign_root_grant_cannot_publish",
        module="hermes_state.py",
        find="        if granted_root != conversation_id:\n            return None\n",
        replace="        if False:\n            return None\n",
        why="the root comparison is the only cross-conversation check; holder "
            "and epoch both match trivially between two first grants",
    ),
    Mutation(
        pin="check_a_holderless_publish_is_refused_while_owned",
        module="hermes_state.py",
        find="            if lease is not None and not self._turn_lease_row_is_free(\n",
        replace="            if False and lease is not None and not self._turn_lease_row_is_free(\n",
        why="this branch IS the holderless admission rule; before it existed "
            "a writer presenting nothing was admitted unconditionally",
    ),
    Mutation(
        pin="check_the_lease_key_walks_to_the_root_three_deep",
        module="hermes_state.py",
        find="        current = _row(session_id)\n        seen = {session_id}\n",
        replace="        return session_id\n        current = _row(session_id)\n"
                "        seen = {session_id}\n",
        why="without the parent walk the lease key is the segment id, so every "
            "rotation mints a new key and frees the conversation mid-turn",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    """Clean, mutated, restored. A pin that survives its mutation is not a pin."""
    assert_mutation_kills_the_pin(mutation, str(_SELF), tmp_path)


def test_every_pin_has_a_mutation_that_kills_it():
    """No pin without a killer, and no killer without a pin."""
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)
