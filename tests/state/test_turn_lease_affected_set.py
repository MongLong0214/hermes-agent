"""A writer must admit every canonical root it REACHES, not the one it names.

THE COUNTEREXAMPLE THIS FILE EXISTS FOR
    ``_session_turn_lease_key_on_conn`` follows ``parent_session_id`` only
    while the parent's ``end_reason`` is ``'compression'`` and the child
    carries no fork marker. A BRANCH child is therefore its own lease root —
    correctly, because a branch is a separate conversation that a separate
    process can own.

    ``delete_session`` guards the root of the id it was HANDED, refuses when a
    delegate-cascade child's conversation is owned, and then executes::

        UPDATE sessions SET parent_session_id = NULL
         WHERE parent_session_id = ?

    That statement reaches the branch child, whose root was never consulted.
    Measured on the base tree, with a FREE parent ``p`` and a branch child
    ``c`` whose conversation a live turn owns::

        root(p) = p        root(c) = c        grant on c held
        delete_session('p')  ->  ADMITTED, returned True
        c.parent_session_id  ->  'p' before, None after

    A bystander mutated a row inside a conversation another process owns, and
    was told nothing. ``delete_sessions``, ``prune_sessions`` and
    ``delete_empty_sessions`` carry the identical statement over an ``IN``
    list; ``delete_session_if_empty`` carries no admission at all.

    The same shape reaches further than lineage cosmetics. ``parent_session_id``
    is the edge every lineage reader walks — ``get_compression_lineage``,
    ``_is_compression_child_row``, the lease-key walk itself. Severing it under
    a live owner changes what the next turn of THAT conversation resolves.

THE RULE, STATED SO IT CAN BE CHECKED
    A write is admitted only when EVERY canonical root it can reach admits it:

    the named root      root + holder + monotonic epoch, via
                        ``_check_turn_lease_guard``, in the same
                        ``BEGIN IMMEDIATE`` transaction as the DML.
    every other root    FREE. A grant names one conversation; it authorizes
                        nothing about a second one, so the only tokenless
                        admission is freeness.

    "Every other root" is DERIVED from the statement's own reach — the delegate
    cascade it deletes, and the direct children whose parent reference it
    severs — by :meth:`SessionDB._affected_session_ids`, inside the
    transaction. It is not a list.

REFUSE VERSUS SKIP, AND WHY BOTH ARE HERE
    A caller that named ONE session gets a refusal: removing the parent while
    leaving a live-owned child half-applied is worse than doing nothing. A
    SWEEP skips: its victims come from a filter, nobody could have held a grant
    naming them, and refusing the whole pass because one conversation is busy
    is how sweeps end up being run with the fence off. Both are pinned, and the
    sweep pin requires the UNAFFECTED session in the same batch to still be
    deleted — a sweep that refuses everything passes a skip test perfectly.

WHAT THE PINS ASSERT
    Rows and values. The child's ``parent_session_id`` byte-identical across
    the refusal, the parent row still present, the sweep's return count, and —
    on the other side — the owner's own delete of its own lineage LANDING.
"""

from __future__ import annotations

import json
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


def _parent_of(db, session_id):
    row = db.get_session(session_id)
    return None if row is None else row["parent_session_id"]


def _exists(db, session_id) -> bool:
    return db.get_session(session_id) is not None


def _branch_child(db, parent_id: str, child_id: str) -> None:
    """A child that is its OWN lease root: an explicit branch of *parent_id*."""
    db.create_session(
        child_id,
        "test",
        parent_session_id=parent_id,
        model_config=json.dumps({"_branched_from": parent_id}),
    )
    db.append_message(child_id, "user", f"{child_id} context")
    assert db._session_turn_lease_key(child_id) == child_id, (
        "the fixture is not exercising the gap: the child resolves to its "
        "parent's root, so the parent's own guard already covers it"
    )


def _compression_child(db, parent_id: str, child_id: str) -> None:
    """A child whose lease root IS *parent_id* — a compression continuation."""
    db.end_session(parent_id, "compression")
    db.create_session(child_id, "test", parent_session_id=parent_id)
    assert db._session_turn_lease_key(child_id) == parent_id, (
        "the fixture is not exercising the compression case: the child is "
        "its own root"
    )


# ---------------------------------------------------------------------------
# The pins.
# ---------------------------------------------------------------------------

def check_a_delete_is_refused_while_a_branch_child_is_owned(tmpdir) -> None:
    """The measured counterexample, and the free path it must not break."""
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        db.create_session("p", "test")
        db.append_message("p", "user", "parent context")
        _branch_child(db, "p", "c")

        grant = db.try_acquire_session_turn_lease(
            "c", _holder("child-owner"), ttl_seconds=600
        )
        assert grant, "could not take the lease on the branch child"
        assert db._session_turn_lease_key("p") == "p"

        before = _parent_of(db, "c")
        assert before == "p", "fixture never linked the child to the parent"

        try:
            db.delete_session("p")
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "delete_session('p') was admitted while a live turn owned the "
                "branch child's conversation; the SET NULL reached a row "
                "inside a conversation this caller does not hold"
            )

        assert _parent_of(db, "c") == before, (
            f"the refused delete severed the owned child's lineage anyway: "
            f"{_parent_of(db, 'c')!r} != {before!r}"
        )
        assert _exists(db, "p"), "the refused delete removed the parent row"
        assert _exists(db, "c"), "the refused delete removed the child row"

        # The other side: released, the very same call must land. A fence that
        # refuses everybody passes the assertions above perfectly.
        db.release_session_turn_lease("c", grant)
        assert db.delete_session("p") is True, (
            "the delete is refused even on a FREE conversation, which breaks "
            "every single-writer install"
        )
        assert not _exists(db, "p")
        assert _parent_of(db, "c") is None, (
            "the admitted delete did not orphan the child, so the pin above "
            "is measuring a delete that never does anything"
        )
    finally:
        db.close()


def check_a_bulk_delete_skips_the_parent_of_an_owned_branch_child(tmpdir) -> None:
    """A sweep SKIPS the reachable-owned victim and still deletes the rest."""
    db = _store(tmpdir)
    try:
        db.create_session("p", "test")
        db.append_message("p", "user", "parent context")
        _branch_child(db, "p", "c")
        db.create_session("q", "test")
        db.append_message("q", "user", "unrelated context")

        grant = db.try_acquire_session_turn_lease(
            "c", _holder("child-owner"), ttl_seconds=600
        )
        assert grant

        deleted = db.delete_sessions(["p", "q"])

        assert _parent_of(db, "c") == "p", (
            "the bulk delete severed the owned child's lineage: the IN-list "
            "SET NULL reaches rows whose root the sweep never consulted"
        )
        assert _exists(db, "p"), "the bulk delete removed the reachable parent"
        assert not _exists(db, "q"), (
            "the sweep skipped the UNAFFECTED session too — a sweep that "
            "refuses everything satisfies the assertion above and is useless"
        )
        assert deleted == 1, f"expected exactly q to be deleted, got {deleted}"

        db.release_session_turn_lease("c", grant)
        assert db.delete_sessions(["p"]) == 1, (
            "the sweep skips a FREE conversation, which makes routine "
            "maintenance unrunnable"
        )
        assert _parent_of(db, "c") is None
    finally:
        db.close()


def check_a_prune_skips_the_parent_of_an_owned_branch_child(tmpdir) -> None:
    """``prune_sessions`` picks victims from a filter; same reach, same rule."""
    db = _store(tmpdir)
    try:
        db.create_session("p", "test")
        db.append_message("p", "user", "parent context")
        _branch_child(db, "p", "c")
        db.create_session("q", "test")
        db.append_message("q", "user", "unrelated context")
        db.end_session("p", "done")
        db.end_session("q", "done")

        grant = db.try_acquire_session_turn_lease(
            "c", _holder("child-owner"), ttl_seconds=600
        )
        assert grant

        pruned = db.prune_sessions(older_than_days=None, started_before=None)

        assert _parent_of(db, "c") == "p", (
            "prune severed the owned child's lineage"
        )
        assert _exists(db, "p"), "prune removed the reachable parent"
        assert not _exists(db, "q"), (
            "prune skipped the unaffected session too, so this pin cannot "
            "tell a working fence from a broken sweep"
        )
        assert pruned == 1, f"expected exactly q to be pruned, got {pruned}"
    finally:
        db.close()


def check_an_empty_session_delete_is_refused_on_an_owned_root(tmpdir) -> None:
    """``delete_session_if_empty`` carried no admission at all.

    The victim is a compression continuation, so its root is the PARENT and
    the owner never named the child — which is exactly the id the CLI
    rotation path hands this method.
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        db.create_session("p", "test")
        db.append_message("p", "user", "parent context")
        _compression_child(db, "p", "c")

        grant = db.try_acquire_session_turn_lease(
            "p", _holder("root-owner"), ttl_seconds=600
        )
        assert grant

        try:
            deleted = db.delete_session_if_empty("c")
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                f"delete_session_if_empty('c') returned {deleted!r} while a "
                f"live turn owned the conversation 'c' belongs to"
            )
        assert _exists(db, "c"), (
            "the refused call deleted the row inside the owned conversation"
        )

        db.release_session_turn_lease("p", grant)
        assert db.delete_session_if_empty("c") is True, (
            "an empty session cannot be reaped on a FREE conversation"
        )
        assert not _exists(db, "c")
    finally:
        db.close()


def check_the_owner_can_delete_its_own_compression_lineage(tmpdir) -> None:
    """The leg that stops the rule from becoming "refuse everything".

    The owner holds the root's grant and deletes the root. The compression
    child it severs resolves to that SAME root, so an affected-set rule that
    demanded freeness of every reached root — instead of freeness of every
    root OTHER than the one the grant authorizes — would refuse the owner's
    own delete.
    """
    db = _store(tmpdir)
    try:
        db.create_session("p", "test")
        db.append_message("p", "user", "parent context")
        _compression_child(db, "p", "c")

        grant = db.try_acquire_session_turn_lease(
            "p", _holder("root-owner"), ttl_seconds=600
        )
        assert grant

        assert db.delete_session("p", turn_lease_holder=grant) is True, (
            "the owner's own delete of its own lineage was refused; the "
            "affected-set rule must exempt the root its grant authorizes"
        )
        assert not _exists(db, "p")
        assert _parent_of(db, "c") is None
    finally:
        db.close()


def check_a_stale_epoch_cannot_delete_across_the_affected_set(tmpdir) -> None:
    """A grant from a previous generation admits nothing, cascade included."""
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        db.create_session("p", "test")
        db.append_message("p", "user", "parent context")
        _branch_child(db, "p", "c")

        stale = db.try_acquire_session_turn_lease(
            "p", _holder("first"), ttl_seconds=600
        )
        assert stale
        db.release_session_turn_lease("p", stale)
        fresh = db.try_acquire_session_turn_lease(
            "p", _holder("first"), ttl_seconds=600
        )
        assert fresh and str(fresh) == str(stale), (
            "the fixture needs the SAME holder string across generations, or "
            "this is a holder check rather than an epoch check"
        )
        assert fresh.epoch != stale.epoch, "the epoch did not advance"

        try:
            db.delete_session("p", turn_lease_holder=stale)
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a grant from a superseded epoch deleted the conversation"
            )
        assert _exists(db, "p")
        assert _parent_of(db, "c") == "p"
    finally:
        db.close()


PINS = {
    "check_a_delete_is_refused_while_a_branch_child_is_owned":
        check_a_delete_is_refused_while_a_branch_child_is_owned,
    "check_a_bulk_delete_skips_the_parent_of_an_owned_branch_child":
        check_a_bulk_delete_skips_the_parent_of_an_owned_branch_child,
    "check_a_prune_skips_the_parent_of_an_owned_branch_child":
        check_a_prune_skips_the_parent_of_an_owned_branch_child,
    "check_an_empty_session_delete_is_refused_on_an_owned_root":
        check_an_empty_session_delete_is_refused_on_an_owned_root,
    "check_the_owner_can_delete_its_own_compression_lineage":
        check_the_owner_can_delete_its_own_compression_lineage,
    "check_a_stale_epoch_cannot_delete_across_the_affected_set":
        check_a_stale_epoch_cannot_delete_across_the_affected_set,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_affected_set_property(name, tmp_path):
    PINS[name](tmp_path)


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_a_delete_is_refused_while_a_branch_child_is_owned",
        module="hermes_state.py",
        find=(
            "        reached = [\n"
            "            sid for sid in self._affected_session_ids(conn, [session_id])\n"
            "            if sid != session_id\n"
            "        ]\n"
        ),
        replace=(
            "        reached = sorted(\n"
            "            _collect_delegate_child_ids(conn, [session_id])\n"
            "        )\n"
        ),
        why="restores the pre-slice reach — the delegate cascade only — so the "
            "SET NULL on branch children is unadmitted again",
    ),
    Mutation(
        pin="check_a_bulk_delete_skips_the_parent_of_an_owned_branch_child",
        module="hermes_state.py",
        find=(
            "            reached = self._affected_session_ids(conn, [sid])\n"
            "            roots = {\n"
            "                self._session_turn_lease_key_on_conn(conn, other)\n"
            "                for other in reached\n"
            "            }\n"
        ),
        replace=(
            "            roots = {self._session_turn_lease_key_on_conn(conn, sid)}\n"
        ),
        why="restores the pre-slice sweep predicate — the victim's own root "
            "only — so a sweep stops seeing the roots its SET NULL reaches",
    ),
    Mutation(
        pin="check_a_prune_skips_the_parent_of_an_owned_branch_child",
        module="hermes_state.py",
        find=(
            "        severed = conn.execute(\n"
            '            f"SELECT id FROM sessions WHERE parent_session_id IN ({ph})",\n'
            "            sorted(doomed),\n"
            "        ).fetchall()\n"
        ),
        replace="        severed = []\n",
        why="the severed-children arm is the whole of the reach a SET NULL "
            "has; without it a sweep's admission is back to the victim's own "
            "root and the branch child is invisible again",
    ),
    Mutation(
        pin="check_an_empty_session_delete_is_refused_on_an_owned_root",
        module="hermes_state.py",
        find=(
            "            # An empty row is still a row inside somebody's conversation.\n"
            "            self._check_turn_lease_guard(\n"
            "                conn,\n"
            "                session_id,\n"
            "                turn_lease_holder,\n"
            "                turn_lease_ttl_seconds=turn_lease_ttl_seconds,\n"
            "            )\n"
        ),
        replace="",
        why="delete_session_if_empty carried no admission at all before this "
            "slice; without the guard it reaps a row out of an owned lineage",
    ),
    Mutation(
        pin="check_the_owner_can_delete_its_own_compression_lineage",
        module="hermes_state.py",
        find=(
            "            if self._session_turn_lease_key_on_conn(conn, other)\n"
            "            != own_root\n"
        ),
        replace="            if True\n",
        why="dropping the own-root exemption makes the affected-set rule "
            "refuse the owner's own delete of its own lineage",
    ),
    Mutation(
        pin="check_a_stale_epoch_cannot_delete_across_the_affected_set",
        module="hermes_state.py",
        find=(
            "            self._check_turn_lease_guard(\n"
            "                conn,\n"
            "                session_id,\n"
            "                turn_lease_holder,\n"
            "                turn_lease_ttl_seconds=turn_lease_ttl_seconds,\n"
            "            )\n"
            "            cursor = conn.execute(\n"
            '                "SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)\n'
            "            )\n"
        ),
        replace=(
            "            cursor = conn.execute(\n"
            '                "SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)\n'
            "            )\n"
        ),
        why="the epoch comparison lives in _check_turn_lease_guard; removing "
            "the call is what lets a superseded grant through",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    assert_mutation_kills_the_pin(mutation, str(_SELF), tmp_path)


def test_every_pin_has_a_mutation_that_kills_it():
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)
