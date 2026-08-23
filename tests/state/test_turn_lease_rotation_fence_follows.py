"""After a compression publish, the fence follows the conversation to the tip.

WHAT C2 ALREADY PINNED, AND WHAT IT DID NOT
    ``tests/state/test_turn_lease_compression_lineage`` covers the publish
    itself, mutation-killed, in four properties:

        a superseded grant cannot publish            (the parent EPOCH)
        a foreign-root grant cannot publish          (the parent ROOT)
        a holderless publish is refused while owned
        the lease key walks to the root three deep

    All four are about the act of publishing, and the fourth is about
    RESOLUTION — ``_session_turn_lease_key("c3") == "root"`` — plus the fact
    that rotation mints no second lease row.

    The gap is one step past that. Rotation gives the conversation a NEW
    SESSION ID, and that id is the one every later caller has: the TUI resumes
    on it, the gateway routes to it, a slash command names it. Resolution being
    correct is not the same claim as ADMISSION being correct on the new id, and
    the failure mode is the attack rather than a lookup: rotate, then aim at
    the tip.

    Both halves of that are pinned here, in one property, because they are one
    hole seen from two sides. A writer that presents nothing must be refused on
    the tip, and a contender must not be able to take a lease on the tip — if
    it could, the row it created would be keyed on the child id, the parent's
    row would still say the parent is owned, and both writers would be "the
    owner" of the same conversation at the same time.

WHY THE MUTATION IS THE PARENT WALK
    Because that is the guard. Everything else in the admission path reads a
    row that is keyed by whatever the walk returned; with the walk gone, the
    key for the tip is the tip, the lookup finds no row, and BOTH halves open
    at once. The lineage file mutates the same line for its resolution pin —
    the same guard provides both properties, and the properties are different
    claims about it, which is why they are pinned separately.

WHY THIS PIN DIES ON ADMISSION AND NOT ON A CRASH
    It did not, and that was the defect in the row rather than in the code.
    The resolution snapshot ran BEFORE the two admission halves and rendered
    its failure message with ``dict(before_row)``. Remove the walk and
    ``get_session_turn_lease("child")`` returns ``None``, so composing the
    message raised ``TypeError: 'NoneType' object is not iterable`` and the
    harness recorded a non-zero exit — a dead row, not a refuted claim. The
    guard whose removal the row exists to detect is precisely the one that
    makes that value ``None``, so the single input the message had to survive
    was the only one it would ever see.

    Two changes, and both are the same point: the failure text goes through
    :func:`_row_repr` so ``None`` renders, and the resolution assertions moved
    BELOW the two admission halves. The mutation now kills this pin at half
    one — "a holderless write landed on the rotated tip of a conversation
    another writer owns" — which is the claim the file is named for, and the
    resolution claim is still asserted, after the attack rather than before it.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from tests.state.lease_mutation_harness import (
    Mutation,
    assert_every_pin_has_a_killer,
    assert_mutation_kills_the_pin,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: This file, so a mutated extract can run the same checks it defines.
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _store(tmpdir, name="state.db"):
    from hermes_state import SessionDB

    return SessionDB(db_path=pathlib.Path(tmpdir) / name)


def _row_repr(row):
    """A lease row rendered so that ``None`` is a value, not an exception.

    Every failure message in this file goes through here. ``dict(row)`` is the
    obvious spelling and it is the reason the mutation below used to kill this
    pin with ``TypeError: 'NoneType' object is not iterable`` — the walk is
    exactly the guard whose removal makes the row ``None``, so the one input
    the message had to survive was the one it could not render.
    """
    return "None" if row is None else repr(dict(row))


def _lease_keys(db):
    with db._read_ctx() as conn:
        return sorted(
            str(row[0]) for row in conn.execute(
                "SELECT conversation_id FROM session_turn_leases"
            )
        )


def check_the_fence_follows_the_conversation_onto_the_rotated_tip(
    tmpdir,
) -> None:
    """Rotation renames the conversation; it must not un-fence it.

    The turn holds the root's lease and compresses, so the conversation now
    answers to ``child``. That is the id every later caller has. Two things
    have to be true of it and neither follows from the key resolving correctly:

    * a writer presenting nothing is refused ON THE TIP, with the transcript
      unchanged — not merely "the key resolved to root";
    * a contender cannot take a lease ON THE TIP, and no second lease row
      appears. A row keyed on the child id would leave the parent's row still
      saying the parent is owned, so both writers would be the owner of one
      conversation at the same time.

    And the owner is unaffected throughout: its own grant still writes to the
    tip, which is what stops this from being a check that refuses everybody.
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        db.create_session("root", "test")
        db.append_message("root", "user", "root context")
        turn = db.try_acquire_session_turn_lease(
            "root", _holder("turn"), ttl_seconds=600
        )
        assert turn, "could not take the turn's lease"

        # The turn is what compresses, so it presents its own grant:
        # taking the compression lock is admitted now, and a compressor
        # with no grant would be denying the owner its own compression.
        assert db.try_acquire_compression_lock(
            "root", "compressor", ttl_seconds=60, turn_lease_holder=turn
        )
        db.publish_compression_child(
            parent_session_id="root",
            child_session_id="child",
            source="test",
            messages=[{"role": "user", "content": "summary"}],
            compression_lock_holder="compressor",
            turn_lease_holder=turn,
        )
        assert db.get_session("child") is not None, "the rotation did not happen"
        before_keys = _lease_keys(db)
        assert before_keys == ["root"], before_keys
        before_row = db.get_session_turn_lease("child")

        # Half one: a holderless writer aiming at the TIP.
        try:
            db.append_message("child", "assistant", "smuggled")
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a holderless write landed on the rotated tip of a "
                "conversation another writer owns"
            )
        assert [m["content"] for m in db.get_messages("child")] == ["summary"], (
            "the refused write changed the tip's transcript anyway"
        )

        # Half two: a contender acquiring on the TIP.
        stolen = db.try_acquire_session_turn_lease(
            "child", _holder("contender"), ttl_seconds=300
        )
        assert stolen is None, (
            f"a contender took a lease by aiming at the rotated tip: "
            f"{stolen!r}. The parent's row still names the owner, so the "
            f"conversation now has two owners"
        )
        assert _lease_keys(db) == before_keys, (
            f"a second lease row appeared for the same conversation: "
            f"{_lease_keys(db)!r}"
        )

        # ...and the row the tip resolves to is still the owner's, unchanged.
        # Asserted here rather than before the two halves, and rendered through
        # `_row_repr`, for the reason in WHY THIS PIN DIES ON ADMISSION: an
        # f-string that calls `dict(None)` raises TypeError while composing the
        # failure message, so the pin dies by a crash instead of by the claim
        # it was written to make.
        after_row = db.get_session_turn_lease("child")
        assert before_row is not None, (
            f"the tip resolves to no lease row at all: {_row_repr(before_row)}"
        )
        assert before_row["holder"] == str(turn), (
            f"the tip does not resolve to the owned conversation: "
            f"{_row_repr(before_row)}"
        )
        assert _row_repr(after_row) == _row_repr(before_row), (
            f"the refused attempts changed the tip's lease row: "
            f"{_row_repr(before_row)} -> {_row_repr(after_row)}"
        )

        # The owner is unaffected and still writes to the tip.
        assert db.append_message(
            "child", "assistant", "owner write", turn_lease_holder=turn
        )
        assert [m["content"] for m in db.get_messages("child")] == [
            "summary", "owner write",
        ]

        # ...and once the owner releases, the tip is writable again.
        db.release_session_turn_lease("child", turn)
        assert db.append_message("child", "assistant", "after release")
        assert [m["content"] for m in db.get_messages("child")] == [
            "summary", "owner write", "after release",
        ]
    finally:
        db.close()


PINS = {
    "check_the_fence_follows_the_conversation_onto_the_rotated_tip":
        check_the_fence_follows_the_conversation_onto_the_rotated_tip,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_rotation_fence_property(name, tmp_path):
    """The pin. Each property, asserted against the tree under test."""
    PINS[name](tmp_path)


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_the_fence_follows_the_conversation_onto_the_rotated_tip",
        module="hermes_state.py",
        find="        current = _row(session_id)\n        seen = {session_id}\n",
        replace="        return session_id\n        current = _row(session_id)\n"
                "        seen = {session_id}\n",
        why="without the parent walk the key for the tip is the tip: the "
            "admission lookup finds no row, so the holderless write is "
            "admitted AND the contender gets a lease of its own, and the "
            "conversation has two owners",
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
