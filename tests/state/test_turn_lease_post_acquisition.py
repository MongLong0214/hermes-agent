"""The post-acquisition contract of ``SessionDB.session_turn_lease``, pinned.

WHY THIS FILE EXISTS, AND WHY IT NAMES ITS OWN SCOPE HONESTLY
    The terminal gate for this work lists seven acceptance items. Item 5 is:

        Post-acquire resolve/reload failure RELEASES and PROPAGATES before
        mutation; read-modify-write callers consume the acquired root/history,
        not their pre-acquire snapshot.

    ``session_turn_lease``'s own docstring claims all of it — "Fail-closed in
    three places, none of which are logged-and-continued" — and ends with
    "Consuming these is checked separately." Nothing in the tree checks them.
    Grepping for ``resolve_resume_session_id`` across ``tests/`` finds
    ``test_hermes_state`` (a different property) and a pile of TUI-gateway
    fakes; ``SessionTurnLeaseScope`` appears in no test at all.

    So this is the same shape as blocker (d): the behaviour is right and the
    repository has never said so. A docstring is not a test, and the specific
    way this one fails is quiet — swallowing a failed resolve turns "I do not
    know where this conversation lives" into "it lives exactly where you
    thought", and swallowing a failed reload presents DELETION as a view.

WHY THE ASSERTIONS ARE ON THE LEASE ROW AND ON A FLAG
    "It raised" is half the property. The other half is that the lease was
    RELEASED on the way out — a fail-closed path that propagates and leaks the
    lease wedges the conversation until a human runs
    ``force_release_session_turn_lease``, which is exactly the outcome the
    liveness contract warns is unbounded. And "the caller never entered the
    body" is checked with a flag the body sets, because an exception raised
    after the body ran is not the same event.

    Every pin is mutation-killed through the shared harness: each row deletes
    the guard providing its property and requires the check to fail.
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
    """The lease row as VALUES, so 'released' is asserted and not inferred."""
    with db._read_ctx() as conn:
        row = conn.execute(
            "SELECT holder, epoch FROM session_turn_leases "
            "WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    return None if row is None else (row["holder"], int(row["epoch"]))


# ---------------------------------------------------------------------------
# The properties.
# ---------------------------------------------------------------------------

def check_a_failed_resolve_releases_and_propagates(tmpdir) -> None:
    """A resolve that fails must not be turned into "it lives where you thought".

    ``resolve_resume_session_id`` answers "where does this conversation live
    NOW" — after however many compression rotations happened while this writer
    was waiting for the lease. Swallowing its failure and falling back to the
    caller's id is the fail-open: the writer then appends to a session that was
    closed, and the transcript the next turn replays is missing the write.

    Three assertions, because any one of them alone can hold while the
    contract is broken: it propagates, the body never ran, and the lease came
    back free.
    """
    db = _store(tmpdir)
    try:
        db.create_session("root", "test")
        entered = []

        def _boom(_session_id):
            raise RuntimeError("resolve failed")

        db.resolve_resume_session_id = _boom
        try:
            with db.session_turn_lease("root", _holder("writer")):
                entered.append(True)
        except RuntimeError as exc:
            assert "resolve failed" in str(exc)
        else:
            raise AssertionError(
                "a failed resolve_resume_session_id did not propagate; the "
                "caller was handed its own pre-acquire id as if it were true"
            )
        assert not entered, "the body ran despite the resolve failing"

        row = _lease_row(db, "root")
        assert row is not None and row[0] == "", (
            f"the lease was not released on the way out: {row!r}. A fail-closed "
            f"path that leaks the lease wedges the conversation until someone "
            f"runs force_release_session_turn_lease by hand"
        )
    finally:
        db.close()


def check_a_failed_reload_releases_and_propagates(tmpdir) -> None:
    """An empty list is a VALID history, so a swallowed reload presents deletion.

    This is why the reload failure cannot be logged-and-continued: there is no
    value ``messages`` can take that means "I could not read it". ``[]`` means
    "this conversation has no messages", and a read-modify-write caller that
    believes it will replace a real transcript with nothing.
    """
    db = _store(tmpdir)
    try:
        db.create_session("root", "test")
        db.append_message("root", "user", "irreplaceable")
        entered = []

        def _boom(*_args, **_kwargs):
            raise RuntimeError("reload failed")

        db.get_messages = _boom
        try:
            with db.session_turn_lease("root", _holder("writer")):
                entered.append(True)
        except RuntimeError as exc:
            assert "reload failed" in str(exc)
        else:
            raise AssertionError(
                "a failed transcript reload did not propagate; an empty list "
                "is a valid history, so swallowing it presents deletion as a "
                "view"
            )
        assert not entered, "the body ran despite the reload failing"

        row = _lease_row(db, "root")
        assert row is not None and row[0] == "", (
            f"the lease was not released on the way out: {row!r}"
        )
    finally:
        db.close()


def check_the_scope_carries_the_post_acquisition_view(tmpdir) -> None:
    """What the caller gets is true AFTER acquisition, not before it.

    The conversation rotated onto a new session id while this writer was not
    holding it. A caller that keeps using the id it asked for writes into a
    closed parent. So the scope must hand back the LIVE id and the transcript
    as of acquisition — and this asserts both, because handing back the right
    id with a stale history is the same bug one field over.
    """
    db = _store(tmpdir)
    try:
        db.create_session("root", "test")
        db.append_message("root", "user", "before rotation")
        assert db.try_acquire_compression_lock("root", "compressor", ttl_seconds=60)
        db.publish_compression_child(
            parent_session_id="root",
            child_session_id="child",
            source="test",
            messages=[{"role": "user", "content": "after rotation"}],
            compression_lock_holder="compressor",
        )

        with db.session_turn_lease("root", _holder("writer")) as scope:
            assert scope.session_id == "child", (
                f"the scope handed back {scope.session_id!r}, the id the caller "
                f"asked for, not where the conversation lives now"
            )
            assert [m["content"] for m in scope.messages] == ["after rotation"], (
                f"the scope handed back {[m['content'] for m in scope.messages]!r}"
                f", which is not the transcript as of acquisition"
            )
            assert scope.token.conversation_id == "root", (
                "the grant must still be keyed on the conversation root"
            )
    finally:
        db.close()


def check_a_lost_acquisition_never_enters_the_body(tmpdir) -> None:
    """A writer that could not take the lease must not run its body at all.

    Its input is still in hand and still retryable only if it never started.
    The refusal has to happen before the ``yield``, which is not the same as
    "the writer sees an exception eventually".
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        db.create_session("root", "test")
        # A live owner: same pid, so liveness reads alive, different holder.
        owner = db.try_acquire_session_turn_lease(
            "root", _holder("owner"), ttl_seconds=600
        )
        assert owner, "could not take the lease this check depends on"
        before = _lease_row(db, "root")

        entered = []
        try:
            with db.session_turn_lease(
                "root", _holder("contender"), wait_seconds=0.0
            ):
                entered.append(True)
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a writer that never acquired the lease ran its body"
            )
        assert not entered, "the body ran without the lease"
        assert _lease_row(db, "root") == before, (
            "the failed contender changed the owner's lease row; a refusal "
            "that releases somebody else's lease is worse than no refusal"
        )
    finally:
        db.close()


PINS = {
    "check_a_failed_resolve_releases_and_propagates":
        check_a_failed_resolve_releases_and_propagates,
    "check_a_failed_reload_releases_and_propagates":
        check_a_failed_reload_releases_and_propagates,
    "check_the_scope_carries_the_post_acquisition_view":
        check_the_scope_carries_the_post_acquisition_view,
    "check_a_lost_acquisition_never_enters_the_body":
        check_a_lost_acquisition_never_enters_the_body,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_post_acquisition_property(name, tmp_path):
    """The pin. Each property, asserted against the tree under test."""
    PINS[name](tmp_path)


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_a_failed_resolve_releases_and_propagates",
        module="hermes_state.py",
        find="            resolved = self.resolve_resume_session_id(session_id)\n",
        replace="            try:\n"
                "                resolved = self.resolve_resume_session_id(session_id)\n"
                "            except Exception:\n"
                "                resolved = None\n",
        why="letting the resolve failure through is the whole guard; swallowing "
            "it falls back to the caller's own id, which is the fail-open",
    ),
    Mutation(
        pin="check_a_failed_reload_releases_and_propagates",
        module="hermes_state.py",
        find="            messages: List[Dict[str, Any]] = (\n"
             "                self.get_messages(live_session_id) if reload_messages else []\n"
             "            )\n",
        replace="            try:\n"
                "                messages = (\n"
                "                    self.get_messages(live_session_id)\n"
                "                    if reload_messages else []\n"
                "                )\n"
                "            except Exception:\n"
                "                messages = []\n",
        why="an empty list is a valid history, so a swallowed reload hands the "
            "caller deletion as a view",
    ),
    Mutation(
        pin="check_the_scope_carries_the_post_acquisition_view",
        module="hermes_state.py",
        find="                token=token, session_id=live_session_id, messages=messages\n",
        replace="                token=token, session_id=session_id, messages=messages\n",
        why="the scope exists to hand back the POST-acquisition id; handing "
            "back the caller's own id makes the whole re-resolution pointless",
    ),
    Mutation(
        pin="check_a_lost_acquisition_never_enters_the_body",
        module="hermes_state.py",
        find="        if token is None:\n            raise SessionTurnLeaseLostError(\n",
        replace="        if False:\n            raise SessionTurnLeaseLostError(\n",
        why="without this the contextmanager yields with token None and the "
            "caller runs its body having acquired nothing",
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
