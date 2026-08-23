"""Nested ownership: a writer INSIDE a turn reuses the turn's exact grant.

WHAT "NESTED" MEANS HERE AND WHY ACQUIRING IS THE WRONG ANSWER
    A tool the model called, the compressor, a post-turn display stamp — these
    run inside a turn this process is already holding the lease for. They
    cannot acquire it: the thing they would be waiting for is themselves. On
    the current wait budget that costs the whole budget and then raises, on
    every in-turn call, and the writer is refused for a reason that has nothing
    to do with the property the lease exists to provide.

    So the contract is REUSE, and the mechanism is ``current_turn_grant``:
    resolve the caller's session id to the conversation root, look that root up
    in the process-local registry of grants THIS process is holding, and hand
    back the grant itself. Two production call sites already do exactly this —
    ``tools/react_to_message_tool.py`` and ``cli.py``'s ``/compress`` — and C2
    pinned the second one specifically. The GENERAL property was unpinned, and
    a mechanism two call sites depend on with nothing checking it is the same
    shape as the transcript-append guard before blocker (b).

THE THREE THINGS THAT HAVE TO BE TRUE, AND THE THREE WAYS EACH FAILS
    * **The token is the SAME one, including after rotation.** Not an equal
      string, not a fresh grant: the same ``(root, holder, epoch)``. If the
      lookup used the caller's segment id instead of walking to the root, every
      compression rotation would hide the turn's own grant from the tool it is
      running, and the tool would fall through to acquiring — against a lease
      the turn is holding.
    * **No self-steal.** A nested writer must not be able to take the
      conversation away from the turn it is running inside, and the sharp case
      is a late unwind: a superseded grant releasing after the turn re-acquired
      must leave the current registration and the current row alone.
    * **No cross-conversation hand-out.** Holding the lease on one conversation
      must not produce a grant for a different one. This is the same hole the
      root binding closes on the write path, one layer up: if the registry were
      keyed by store alone, a nested writer on conversation B would be handed
      A's grant and would stop acquiring B's lease at all.

WHAT THE PINS ASSERT
    Row values and object identity. "It returned something truthy" is not the
    property — a fresh grant for the same conversation is truthy and is exactly
    the failure being ruled out, because it advances the epoch and invalidates
    the token the turn is still using.
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


def _lease_row(db, conversation_id):
    """The lease row as VALUES, so "untouched" is a comparison."""
    with db._read_ctx() as conn:
        row = conn.execute(
            "SELECT holder, epoch, acquired_at, expires_at "
            "FROM session_turn_leases WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    return None if row is None else (
        row["holder"], int(row["epoch"]), float(row["acquired_at"]),
    )


def _identity(token):
    """The three things that make a grant a grant."""
    return (
        str(token),
        getattr(token, "epoch", None),
        getattr(token, "conversation_id", None),
    )


# ---------------------------------------------------------------------------
# The properties.
# ---------------------------------------------------------------------------

def check_a_nested_writer_gets_the_turns_exact_grant_across_rotation(
    tmpdir,
) -> None:
    """The same ``(root, holder, epoch)``, reached from a rotated segment id.

    The tool inside the turn knows the session id it was called with, which
    after a compression is the CHILD segment. The turn's grant was issued
    against the root. So the lookup has to walk, and it has to hand back the
    grant itself rather than something equivalent-looking.

    The contrast that gives this teeth is in the same check: acquiring instead
    of reusing is refused, deterministically, with the wait budget set to zero.
    That is the deadlock the reuse path exists to avoid, stated as an outcome
    rather than as elapsed time.
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

        # In-turn, before any rotation.
        assert db.current_turn_grant("root") is turn, (
            "the registry handed back something other than the turn's own "
            "grant object"
        )

        # The turn compresses. The conversation is now addressed by 'child'.
        assert db.try_acquire_compression_lock("root", "compressor", ttl_seconds=60)
        db.publish_compression_child(
            parent_session_id="root",
            child_session_id="child",
            source="test",
            messages=[{"role": "user", "content": "summary"}],
            compression_lock_holder="compressor",
            turn_lease_holder=turn,
        )
        before = _lease_row(db, "root")

        nested = db.current_turn_grant("child")
        assert nested is turn, (
            f"a tool running inside the turn was not handed the turn's grant "
            f"after rotation: {_identity(nested)!r} vs {_identity(turn)!r}"
        )
        assert _identity(nested) == _identity(turn)

        # And the reused grant WRITES, which acquiring cannot do.
        assert db.append_message(
            "child", "assistant", "nested write", turn_lease_holder=nested
        )
        assert [m["content"] for m in db.get_messages("child")] == [
            "summary", "nested write",
        ]
        assert _lease_row(db, "root") == before, (
            f"the nested write moved the lease row: {_lease_row(db, 'root')!r} "
            f"!= {before!r}. A nested writer that advances the generation "
            f"invalidates the token the turn is still using"
        )

        # The contrast: acquiring is refused, and the row is still untouched.
        try:
            with db.session_turn_lease(
                "child", _holder("nested-acquirer"), wait_seconds=0.0
            ):
                raise AssertionError(
                    "a nested writer acquired a lease this process is holding"
                )
        except SessionTurnLeaseLostError:
            pass
        assert _lease_row(db, "root") == before
        assert db.current_turn_grant("child") is turn, (
            "the refused acquisition disturbed the turn's own registration"
        )
    finally:
        db.close()


def check_a_late_unwind_cannot_steal_the_turn_from_itself(tmpdir) -> None:
    """A superseded grant releasing late must not free the current turn.

    This is the self-steal case, and it is not exotic: a nested writer that
    kept a reference to the grant it was given, unwinding after the turn had
    already released and re-acquired, presents a token with the right holder
    string and the wrong generation. The row must refuse it AND the process's
    registration must survive it — clearing the registration would send the
    next nested writer to the acquire path, against a lease this process holds.
    """
    db = _store(tmpdir)
    try:
        db.create_session("root", "test")
        db.append_message("root", "user", "root context")
        holder = _holder("turn")
        first = db.try_acquire_session_turn_lease("root", holder, ttl_seconds=600)
        assert first
        db.release_session_turn_lease("root", first)
        second = db.try_acquire_session_turn_lease("root", holder, ttl_seconds=600)
        assert second and second.epoch > first.epoch, (
            f"the re-acquisition did not advance the generation "
            f"({first.epoch} -> {getattr(second, 'epoch', None)}); this check "
            f"would pass vacuously"
        )
        assert str(first) == str(second), (
            "both grants must share a holder string, or the epoch comparison "
            "is never the thing doing the work"
        )
        before = _lease_row(db, "root")

        # The late unwind.
        db.release_session_turn_lease("root", first)

        assert _lease_row(db, "root") == before, (
            f"a superseded grant released the current turn's lease: "
            f"{_lease_row(db, 'root')!r} != {before!r}"
        )
        assert db.current_turn_grant("root") is second, (
            "the late unwind cleared the CURRENT grant's registration; the "
            "next in-turn writer would try to acquire a lease this process "
            "is holding and be refused"
        )
        assert db.append_message(
            "root", "assistant", "still ours",
            turn_lease_holder=db.current_turn_grant("root"),
        )
        assert [m["content"] for m in db.get_messages("root")] == [
            "root context", "still ours",
        ]
    finally:
        db.close()


def check_the_turns_grant_is_not_handed_out_on_another_conversation(
    tmpdir,
) -> None:
    """Holding one conversation must not produce a grant for a different one.

    The same hole the root binding closes on the write path, one layer up. If
    the registry were keyed by store alone, a nested writer on B would be
    handed A's grant, stop acquiring B's lease entirely, and be refused by the
    write guard for a reason that reads as "the lease was lost".

    The assertion is on B's lease ROW: with reuse correctly declining, the
    nested writer acquires and B gets a row of its own.
    """
    db = _store(tmpdir)
    try:
        for sid in ("a", "b"):
            db.create_session(sid, "test")
            db.append_message(sid, "user", f"{sid} context")
        turn_a = db.try_acquire_session_turn_lease(
            "a", _holder("turn"), ttl_seconds=600
        )
        assert turn_a
        assert _lease_row(db, "b") is None, "b was leased before this check ran"

        assert db.current_turn_grant("b") is None, (
            f"the registry handed out a grant for conversation 'a' when asked "
            f"about 'b': {_identity(db.current_turn_grant('b'))!r}"
        )

        # ...so an alternate writer on B genuinely acquires B's own lease.
        with db.session_turn_lease(
            "b", _holder("alternate"), wait_seconds=0.0, reload_messages=False
        ) as scope:
            assert scope.token.conversation_id == "b"
            row = _lease_row(db, "b")
            assert row is not None and row[0] == str(scope.token), (
                f"no lease row was taken for b: {row!r}"
            )
            assert db.append_message(
                "b", "assistant", "b write", turn_lease_holder=scope.token
            )
        assert _lease_row(db, "b")[0] == "", "b's lease was not released"
        assert _lease_row(db, "a")[0] == str(turn_a), (
            "the alternate writer on b disturbed a's lease"
        )
        assert [m["content"] for m in db.get_messages("b")] == [
            "b context", "b write",
        ]
    finally:
        db.close()


PINS = {
    "check_a_nested_writer_gets_the_turns_exact_grant_across_rotation":
        check_a_nested_writer_gets_the_turns_exact_grant_across_rotation,
    "check_a_late_unwind_cannot_steal_the_turn_from_itself":
        check_a_late_unwind_cannot_steal_the_turn_from_itself,
    "check_the_turns_grant_is_not_handed_out_on_another_conversation":
        check_the_turns_grant_is_not_handed_out_on_another_conversation,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_nested_ownership_property(name, tmp_path):
    """The pin. Each property, asserted against the tree under test."""
    PINS[name](tmp_path)


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_a_nested_writer_gets_the_turns_exact_grant_across_rotation",
        module="hermes_state.py",
        find="            root = self._session_turn_lease_key(session_id)\n",
        replace="            root = session_id\n",
        why="the grant is registered against the conversation ROOT; looking it "
            "up by the caller's segment id hides the turn's own grant from "
            "every tool it runs after the first compression",
    ),
    Mutation(
        pin="check_a_late_unwind_cannot_steal_the_turn_from_itself",
        module="hermes_state.py",
        find="        if held is token:\n",
        replace="        if held is not None:\n",
        why="identity rather than presence is what stops a superseded grant "
            "unwinding late from clearing the registration of the grant that "
            "replaced it",
    ),
    Mutation(
        pin="check_the_turns_grant_is_not_handed_out_on_another_conversation",
        module="hermes_state.py",
        find='    return (str(db_path or ""), str(conversation_root or ""))\n',
        replace='    return (str(db_path or ""), "")\n',
        why="the conversation root is half the registry key; without it one "
            "process holding any conversation is handed that grant for every "
            "other conversation in the same store",
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
