"""The raw SQLite sink next door: ``tools/async_delegation.py``.

WHAT THIS MODULE IS, AND WHY IT IS IN THE DENOMINATOR
    ``async_delegations`` lives in the SAME ``state.db`` as ``sessions`` and
    ``messages``, and every one of its rows names a conversation: the
    ``parent_session_id`` that dispatched the delegation, plus the origin ids
    the completion is routed back through. The module opens its OWN
    ``sqlite3.connect``, so none of its DML has ever been inside the
    transaction the turn-lease validator sits in.

    A record here is not adjunct trivia. A pending completion is a turn the
    parent conversation has not had yet: ``restore_undelivered_completions``
    replays it as a fresh turn after a restart. So a write that DELETES such a
    row, or moves its ``delivery_state`` out of ``pending``, removes work from
    a conversation somebody may be running right now — and does it from a
    connection the fence cannot see.

THE CLOSURE, AND THE ONE IT REFUSES
    Re-implementing the lineage walk and the lease predicate on this module's
    own connection is the closure this file must never be satisfiable by: a
    second implementation of an admission rule is a second door, and the two
    drift. So the module borrows the CANONICAL one —
    ``SessionDB.admit_on_connection`` / ``skip_leased_on_connection`` run the
    very helpers ``append_message`` runs, against the connection they are
    handed. The transaction is this module's own ``BEGIN IMMEDIATE``, so root,
    holder and epoch are resolved in the same transaction as the DML, which is
    the whole requirement.

WHICH STATEMENTS TAKE ADMISSION, AND WHICH DO NOT
    The rule is what the statement does to the parent's future turn:

    removes / un-pends   DELETE, and every ``delivery_state`` move away from
                         ``pending``. These take admission. A refusal here is a
                         refusal, and the sweeps (retention prune, the stale
                         replay drop) SKIP rather than refuse, for the same
                         reason every other sweep does.
    contention book-keeping
                         the claim / release / attempt-counter writes. They are
                         the mutual exclusion BETWEEN consumers, and the
                         claimer is by construction not the turn owner —
                         fencing them on the turn lease would deadlock the
                         delivery protocol against itself. They leave the
                         payload and the ``pending`` state exactly where they
                         were, which is the property that makes this an argued
                         exemption rather than an unexamined one, and
                         :func:`test_the_claim_protocol_leaves_the_payload_pending`
                         asserts it.
"""

from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import time

import pytest

from tests.state.lease_mutation_harness import (
    Mutation,
    assert_every_pin_has_a_killer,
    assert_mutation_kills_the_pin,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: This file, so a mutated extract can run the same checks it defines.
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)

#: The extract needs the whole package: the sink is outside the state layer.
_EXTRA_EXTRACT = (".",)


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _home(tmpdir) -> pathlib.Path:
    """Point ``get_hermes_home()`` at a private directory for this check."""
    home = pathlib.Path(tmpdir) / "hermes-home"
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HERMES_HOME"] = str(home)
    import hermes_constants

    for name in ("_HERMES_HOME_CACHE", "_HERMES_HOME"):
        if hasattr(hermes_constants, name):
            setattr(hermes_constants, name, None)
    return home


def _store(tmpdir):
    from hermes_state import SessionDB

    return SessionDB(db_path=_home(tmpdir) / "state.db")


def _delegation(home, delegation_id, parent, *, state, delivery, age=0.0):
    """One durable record, written straight in — this is the fixture, not the sink."""
    now = time.time() - age
    conn = sqlite3.connect(home / "state.db", timeout=10)
    try:
        conn.execute(
            """INSERT OR REPLACE INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                parent_session_id, state, dispatched_at, updated_at,
                delivery_state, delivery_attempts, event_json)
               VALUES (?, '', '', ?, ?, ?, ?, ?, 0, ?)""",
            (delegation_id, parent, state, now, now, delivery,
             json.dumps({"delegation_id": delegation_id, "status": state})),
        )
        conn.commit()
    finally:
        conn.close()


def _records(home):
    conn = sqlite3.connect(home / "state.db", timeout=10)
    try:
        return {
            row[0]: row[1] for row in conn.execute(
                "SELECT delegation_id, delivery_state FROM async_delegations"
            )
        }
    finally:
        conn.close()


def _owned(db, session_id, *, tag="owner"):
    db.create_session(session_id, "test")
    db.append_message(session_id, "user", f"{session_id} context")
    grant = db.try_acquire_session_turn_lease(
        session_id, _holder(tag), ttl_seconds=600
    )
    assert grant, f"could not take the lease on {session_id!r}"
    return grant


# ---------------------------------------------------------------------------
# The pins.
# ---------------------------------------------------------------------------

def check_a_durable_delete_is_refused_for_an_owned_parent(tmpdir) -> None:
    """A DELETE removes a turn the parent conversation has not had yet."""
    import tools.async_delegation as ad
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    home = pathlib.Path(db.db_path).parent
    try:
        grant = _owned(db, "parent")
        _delegation(home, "d1", "parent", state="completed", delivery="pending")
        assert "d1" in _records(home)

        try:
            ad._delete_durable_delegation("d1")
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a bystander deleted a pending completion belonging to a "
                "conversation a live turn owns"
            )
        assert "d1" in _records(home), (
            "the refused delete removed the record anyway"
        )

        ad._delete_durable_delegation("d1", turn_lease_holder=grant)
        assert "d1" not in _records(home), (
            "the owner's own delete of its own delegation record was refused"
        )
    finally:
        db.close()


def check_a_retention_prune_keeps_the_owned_parents_record(tmpdir) -> None:
    """The retention sweep SKIPS an owned parent and still collects the rest."""
    import tools.async_delegation as ad

    db = _store(tmpdir)
    home = pathlib.Path(db.db_path).parent
    try:
        grant = _owned(db, "owned-parent")
        db.create_session("free-parent", "test")
        old = ad._DURABLE_RETENTION_SECONDS + 10_000
        _delegation(
            home, "owned", "owned-parent",
            state="completed", delivery="delivered", age=old,
        )
        _delegation(
            home, "free", "free-parent",
            state="completed", delivery="delivered", age=old,
        )

        ad._prune_durable_records()

        after = _records(home)
        assert "owned" in after, (
            "the retention sweep collected a record belonging to a "
            "conversation a live turn owns"
        )
        assert "free" not in after, (
            "the sweep skipped the UNOWNED record too, so retention collects "
            "nothing and the table grows without bound"
        )
    finally:
        db.close()


def check_a_stale_replay_drop_skips_the_owned_parent(tmpdir) -> None:
    """The staleness cap un-pends a row; on an owned parent it must not."""
    import tools.async_delegation as ad

    db = _store(tmpdir)
    home = pathlib.Path(db.db_path).parent
    try:
        grant = _owned(db, "owned-parent")
        db.create_session("free-parent", "test")
        old = ad._MAX_COMPLETION_REPLAY_AGE_S + 10_000
        _delegation(
            home, "owned", "owned-parent",
            state="completed", delivery="pending", age=old,
        )
        _delegation(
            home, "free", "free-parent",
            state="completed", delivery="pending", age=old,
        )

        class _Queue:
            def __init__(self):
                self.items = []

            def put(self, item):
                self.items.append(item)

        ad.restore_undelivered_completions(_Queue())

        after = _records(home)
        assert after["owned"] == "pending", (
            f"the staleness cap terminally dropped a completion belonging to "
            f"a conversation a live turn owns: {after['owned']!r}"
        )
        assert after["free"] == "dropped", (
            "the cap skipped the UNOWNED record too, so a weeks-old completion "
            "replays forever"
        )
    finally:
        db.close()


PINS = {
    "check_a_durable_delete_is_refused_for_an_owned_parent":
        check_a_durable_delete_is_refused_for_an_owned_parent,
    "check_a_retention_prune_keeps_the_owned_parents_record":
        check_a_retention_prune_keeps_the_owned_parents_record,
    "check_a_stale_replay_drop_skips_the_owned_parent":
        check_a_stale_replay_drop_skips_the_owned_parent,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_async_delegation_sink_property(name, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    PINS[name](tmp_path)


def test_the_claim_protocol_leaves_the_payload_pending(tmp_path, monkeypatch):
    """Evidence for the one argued exemption in this module.

    The claim / release pair is mutual exclusion BETWEEN consumers, and the
    claimer is by construction not the turn owner. It is left unfenced, and the
    ground for that is asserted rather than argued: a claim and its release
    leave ``delivery_state`` at ``pending`` and the payload untouched, so
    nothing the parent's next turn would replay has moved.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    import tools.async_delegation as ad

    db = _store(tmp_path)
    home = pathlib.Path(db.db_path).parent
    try:
        _owned(db, "parent")
        _delegation(home, "d1", "parent", state="completed", delivery="pending")
        before = _records(home)

        assert ad.claim_completion_delivery("d1", "claim-1") is True, (
            "the claim protocol is fenced, which deadlocks delivery against "
            "the very turn it is delivering to"
        )
        assert ad.release_completion_delivery("d1", "claim-1") is True

        assert _records(home) == before, (
            f"claim/release moved the delivery state: {_records(home)!r} != "
            f"{before!r}"
        )
    finally:
        db.close()


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_a_durable_delete_is_refused_for_an_owned_parent",
        module="tools/async_delegation.py",
        find=(
            "        _admit_delegations(\n"
            "            conn, [delegation_id], turn_lease_holder,\n"
            "            f\"refusing to delete the durable record for \"\n"
            "            f\"{delegation_id!r}: it belongs to\",\n"
            "        )\n"
        ),
        replace="",
        why="the record is a turn the parent conversation has not had yet; "
            "deleting it from an unfenced connection removes that turn",
    ),
    Mutation(
        pin="check_a_retention_prune_keeps_the_owned_parents_record",
        module="tools/async_delegation.py",
        find="        keep = _leased_delegation_ids(conn)\n",
        replace="        keep = set()\n",
        why="without the per-row sweep admission the retention DELETEs reach "
            "records belonging to conversations a live turn owns",
    ),
    Mutation(
        pin="check_a_stale_replay_drop_skips_the_owned_parent",
        module="tools/async_delegation.py",
        find="        leased = _leased_delegation_ids(conn)\n",
        replace="        leased = set()\n",
        why="the staleness cap moves delivery_state out of pending, which is "
            "what stops the parent's completion ever being replayed",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    assert_mutation_kills_the_pin(mutation, str(_SELF), tmp_path, *_EXTRA_EXTRACT)


def test_every_pin_has_a_mutation_that_kills_it():
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)
