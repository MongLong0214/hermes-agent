"""The two overwriters the last slice could not close, and the price of closing them.

WHAT THE PREVIOUS SLICE LEFT, IN ITS OWN WORDS
    ``_insert_session_row`` and ``update_token_counts`` were left open with the
    reason written down: ``update_token_counts`` calls the first on EVERY
    accounted API call, so fencing either as-is refuses the owner's own
    accounting mid-turn.

    That reason is correct and it is also not the whole shape. Accounting
    deltas travel through a background writer (``queue_token_counts`` ->
    ``_token_writer_loop`` -> ``_apply_token_batch``) which applies them on a
    thread that holds no grant, and whose failure contract is
    ``logger.warning`` — accounting loss is never raised into a turn. So an
    unconditional fence on either method does not merely refuse the owner: it
    SILENTLY DROPS the owner's tokens, and the drop is loudest exactly where
    the route matters (a bare gateway row whose ``model`` is still NULL when
    the first accounted call lands).

THE SHAPE THIS FILE PINS, AND WHY IT IS NOT A SOFTENING
    Admission is decided per STATEMENT-ARM rather than per method, using the
    distinction the replay-column derivation already makes:

        model = COALESCE(model, ?)          backfill  — cannot move a route
        input_tokens = input_tokens + ?     accumulate — no replay column
        model = ?                           OVERWRITE — fenced
        model_config = <excluded>           OVERWRITE — fenced (the ON CONFLICT
                                            CASE replaces a reset-only config,
                                            and NULLs system_prompt when a new
                                            hash arrives)

    So two arms take the full root+holder+epoch admission:

    ``update_token_counts``'s ``first_accounted_route`` branch, whose
    ``UPDATE sessions SET model = ?, billing_provider = ?`` is an outright
    overwrite; and ``_insert_session_row`` WHENEVER THE CALL CARRIES a
    replay-column value, because its ``ON CONFLICT`` arms for
    ``model_config`` and ``system_prompt`` are not COALESCE-only. With all
    three of ``model`` / ``model_config`` / ``system_prompt`` absent, every one
    of those arms degenerates to a no-op on the replay columns and there is
    nothing to admit.

    And a refusal on either arm is a decision about the ROUTE, not a failure of
    the accounting: the additive UPDATE still runs, and the row-ensure retries
    without the route. That is what keeps the fence from becoming the silent
    delta-drop described above.

WHAT IT COSTS, STATED
    A bystander can still move a session's token counters, its cost columns and
    its ``api_call_count`` while a turn owns the conversation. Those columns are
    billing and telemetry; none of them is replayed to the provider, and every
    assignment in that arm is an accumulate or a NULL backfill. The trade is
    deliberate: the failure this admits is a wrong number in an insights panel,
    and the failure the alternative admits is a lost token delta that nothing
    can reconstruct.

    :func:`test_the_additive_arm_cannot_move_any_replay_column` is the evidence
    for the first half of that sentence, and it is asserted rather than argued.
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

REPLAY_COLUMNS = ("model", "model_config", "system_prompt", "system_prompt_hash")

#: One accounted API call, big enough that landing it is unmistakable.
DELTA = {
    "input_tokens": 11,
    "output_tokens": 7,
    "api_call_count": 1,
}


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _store(tmpdir, name="state.db"):
    from hermes_state import SessionDB

    return SessionDB(db_path=pathlib.Path(tmpdir) / name)


def _row(db, session_id="s"):
    return db.get_session(session_id) or {}


def _replay(db, session_id="s"):
    row = _row(db, session_id)
    return tuple(row.get(column) for column in REPLAY_COLUMNS)


def _counters(db, session_id="s"):
    row = _row(db, session_id)
    return (
        row.get("input_tokens"),
        row.get("output_tokens"),
        row.get("api_call_count"),
    )


def _owned(db, session_id="s", *, tag="owner"):
    db.append_message(session_id, "user", f"{session_id} context")
    grant = db.try_acquire_session_turn_lease(
        session_id, _holder(tag), ttl_seconds=600
    )
    assert grant, f"could not take the lease on {session_id!r}"
    return grant


# ---------------------------------------------------------------------------
# The pins.
# ---------------------------------------------------------------------------

def check_a_bystander_cannot_set_the_first_accounted_route(tmpdir) -> None:
    """``first_accounted_route`` is a bare overwrite of ``model``."""
    db = _store(tmpdir)
    try:
        db.create_session("s", "test", model="anthropic/claude-before")
        grant = _owned(db)
        before_route = _row(db)["model"]
        before_counters = _counters(db)
        assert before_route == "anthropic/claude-before"
        assert _row(db)["api_call_count"] in (0, None), (
            "the fixture already has accounted calls, so the "
            "first_accounted_route branch cannot fire"
        )

        db.update_token_counts(
            "s", model="openai/gpt-after", billing_provider="openai", **DELTA
        )

        assert _row(db)["model"] == before_route, (
            f"a bystander moved the route the owner's next turn dispatches "
            f"under: {_row(db)['model']!r} != {before_route!r}"
        )
        assert _counters(db) != before_counters, (
            "the refusal ate the accounting delta; the background writer logs "
            "that loss and never raises it, so it would be silent"
        )
    finally:
        db.close()


def check_the_owner_can_set_its_own_first_accounted_route(tmpdir) -> None:
    """The fallback-route record is the owner's own write and must land."""
    db = _store(tmpdir)
    try:
        db.create_session("s", "test", model="anthropic/claude-before")
        grant = _owned(db)

        db.update_token_counts(
            "s",
            model="openai/gpt-after",
            billing_provider="openai",
            turn_lease_holder=grant,
            **DELTA,
        )

        assert _row(db)["model"] == "openai/gpt-after", (
            "the owner's own first-accounted-route record was refused, which "
            "loses the authoritative route after a primary-provider failover"
        )
        assert _row(db)["billing_provider"] == "openai"
    finally:
        db.close()


def check_a_bystander_cannot_fill_a_null_route_through_the_row_ensure(
    tmpdir,
) -> None:
    """``_insert_session_row``'s ON CONFLICT arms are not COALESCE-only.

    The bare row a gateway creates before the agent exists has ``model``
    NULL, and that is exactly the row the first accounted call reaches.
    """
    db = _store(tmpdir)
    try:
        db.create_session("s", "test")
        grant = _owned(db)
        assert _row(db)["model"] is None, "the fixture has no NULL route to fill"
        before_counters = _counters(db)

        db.update_token_counts("s", model="openai/gpt-bystander", **DELTA)

        assert _row(db)["model"] is None, (
            f"a bystander wrote the route through the row-ensure upsert: "
            f"{_row(db)['model']!r}"
        )
        assert _counters(db) != before_counters, (
            "the route refusal took the accounting with it — the row-ensure "
            "must retry without the route rather than fail the delta"
        )
    finally:
        db.close()


def check_the_owner_can_fill_its_own_null_route(tmpdir) -> None:
    """The row-ensure upsert, from the owner, must record the route.

    Driven through ``create_session`` rather than ``update_token_counts`` on
    purpose: the accounting path can fill a NULL route through EITHER the
    row-ensure upsert or the counter statement's ``COALESCE(model, ?)``, and a
    pin that both of them satisfy cannot be killed by removing either. This one
    names a single seam.
    """
    db = _store(tmpdir)
    try:
        db.create_session("s", "test")
        grant = _owned(db)
        assert _row(db)["model"] is None

        db.create_session(
            "s", "test", model="openai/gpt-owner", turn_lease_holder=grant
        )

        assert _row(db)["model"] == "openai/gpt-owner", (
            "the owner's own route backfill was refused, so a bare gateway "
            "row never learns which model answered"
        )
    finally:
        db.close()


def test_the_additive_arm_cannot_move_any_replay_column(tmp_path) -> None:
    """The evidence for the accepted cost.

    An accounted call from a bystander, on a conversation with every replay
    column already set: the counters move and all four replay columns are
    byte-identical. That is what makes leaving the additive arm holderless an
    argued exemption rather than an unexamined one.
    """
    tmpdir = tmp_path
    db = _store(tmpdir)
    try:
        db.create_session(
            "s",
            "test",
            model="anthropic/claude-before",
            model_config={"reasoning": "high"},
            system_prompt="THE PROMPT THE TURN IS REPLAYING",
        )
        grant = _owned(db)
        # Take the row past api_call_count == 0 so first_accounted_route can
        # no longer fire; what is left is exactly the additive arm.
        db.update_token_counts(
            "s",
            model="anthropic/claude-before",
            billing_provider="anthropic",
            turn_lease_holder=grant,
            **DELTA,
        )
        before_replay = _replay(db)
        before_counters = _counters(db)
        assert all(value is not None for value in before_replay), (
            f"the fixture left a replay column NULL, so this pin would be "
            f"measuring a backfill rather than an overwrite: {before_replay!r}"
        )

        db.update_token_counts(
            "s", model="openai/gpt-bystander", billing_provider="openai", **DELTA
        )

        assert _replay(db) == before_replay, (
            f"the additive accounting arm moved a replay column, so it is not "
            f"the exemption this file claims: {_replay(db)!r} != "
            f"{before_replay!r}"
        )
        assert _counters(db) != before_counters, (
            "the accounting did not land, so this pin proves nothing about the "
            "arm it is describing"
        )
    finally:
        db.close()


def check_a_bystander_cannot_replace_a_reset_only_model_config(tmpdir) -> None:
    """The ON CONFLICT arm that is demonstrably NOT a backfill.

    ``model_config`` is REPLACED — not COALESCEd — when the stored document is
    a bare ``{"_reset_from": …}``. That is the shape a session carries right
    after a reset boundary, and it is a replay column. The vector is
    ``create_session`` on an id that already exists, which is the upsert
    ``_insert_session_row`` is: the gateway re-creates a row it did not know
    the agent had already filled in.
    """
    db = _store(tmpdir)
    try:
        db.create_session("s", "test", model_config={"_reset_from": "earlier"})
        grant = _owned(db)
        before = _row(db)["model_config"]
        assert json.loads(before) == {"_reset_from": "earlier"}

        from hermes_state import SessionTurnLeaseLostError

        try:
            db.create_session(
                "s", "test", model_config={"reasoning": "bystander"}
            )
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "the row-ensure upsert was admitted while a live turn owned "
                "the conversation"
            )
        assert _row(db)["model_config"] == before, (
            f"a bystander replaced a reset-only model_config through the "
            f"row-ensure upsert: {_row(db)['model_config']!r}"
        )

        db.create_session(
            "s", "test", model_config={"reasoning": "owner"},
            turn_lease_holder=grant,
        )
        assert json.loads(_row(db)["model_config"])["reasoning"] == "owner", (
            "the owner's own re-create was refused"
        )
    finally:
        db.close()


PINS = {
    "check_a_bystander_cannot_set_the_first_accounted_route":
        check_a_bystander_cannot_set_the_first_accounted_route,
    "check_the_owner_can_set_its_own_first_accounted_route":
        check_the_owner_can_set_its_own_first_accounted_route,
    "check_a_bystander_cannot_fill_a_null_route_through_the_row_ensure":
        check_a_bystander_cannot_fill_a_null_route_through_the_row_ensure,
    "check_the_owner_can_fill_its_own_null_route":
        check_the_owner_can_fill_its_own_null_route,
    "check_a_bystander_cannot_replace_a_reset_only_model_config":
        check_a_bystander_cannot_replace_a_reset_only_model_config,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_accounting_route_property(name, tmp_path):
    PINS[name](tmp_path)


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_a_bystander_cannot_set_the_first_accounted_route",
        module="hermes_state.py",
        find=(
            "                try:\n"
            "                    self._check_turn_lease_guard(\n"
            "                        conn,\n"
            "                        session_id,\n"
            "                        turn_lease_holder,\n"
            "                        turn_lease_ttl_seconds=turn_lease_ttl_seconds,\n"
            "                    )\n"
            "                except SessionTurnLeaseLostError:\n"
        ),
        replace=(
            "                try:\n"
            "                    pass\n"
            "                except SessionTurnLeaseLostError:\n"
        ),
        why="the route arms of an accounted call are `SET model = ?` and a "
            "COALESCE backfill of a NULL route; with the admission gone a "
            "bystander's accounting moves what the owner's next turn "
            "dispatches under",
    ),
    Mutation(
        pin="check_the_owner_can_set_its_own_first_accounted_route",
        module="hermes_state.py",
        find=(
            "                        turn_lease_holder,\n"
            "                        turn_lease_ttl_seconds=turn_lease_ttl_seconds,\n"
            "                    )\n"
            "                except SessionTurnLeaseLostError:\n"
        ),
        replace=(
            "                        None,\n"
            "                        turn_lease_ttl_seconds=turn_lease_ttl_seconds,\n"
            "                    )\n"
            "                except SessionTurnLeaseLostError:\n"
        ),
        why="dropping the presented grant makes the accounting path read the "
            "owner's own call as a bystander's, so the authoritative route "
            "after a primary-provider failover is never recorded",
    ),
    Mutation(
        pin="check_a_bystander_cannot_fill_a_null_route_through_the_row_ensure",
        module="hermes_state.py",
        find=(
            "            # prompt when a new hash arrives. Admission is required whenever\n"
            "            # this call carries one of them.\n"
            "            if carries_replay_value:\n"
        ),
        replace=(
            "            # prompt when a new hash arrives. Admission is required whenever\n"
            "            # this call carries one of them.\n"
            "            if False:\n"
        ),
        why="_insert_session_row is reached on every accounted call, and its "
            "upsert fills the route on the bare gateway row a live turn owns",
    ),
    Mutation(
        pin="check_the_owner_can_fill_its_own_null_route",
        module="hermes_state.py",
        find=(
            "            if carries_replay_value:\n"
            "                self._check_turn_lease_guard(\n"
            "                    conn,\n"
            "                    session_id,\n"
            "                    turn_lease_holder,\n"
        ),
        replace=(
            "            if carries_replay_value:\n"
            "                self._check_turn_lease_guard(\n"
            "                    conn,\n"
            "                    session_id,\n"
            "                    None,\n"
        ),
        why="the row-ensure upsert refusing the owner's OWN grant is the "
            "over-correction: the bystander pin above still passes while the "
            "route is never recorded for anybody",
    ),
    Mutation(
        pin="check_a_bystander_cannot_replace_a_reset_only_model_config",
        module="hermes_state.py",
        find=(
            "        carries_replay_value = any(\n"
            "            value is not None\n"
            "            for value in (model, model_config, system_prompt)\n"
            "        )\n"
        ),
        replace=(
            "        carries_replay_value = any(\n"
            "            value is not None\n"
            "            for value in (model,)\n"
            "        )\n"
        ),
        why="narrowing the carried-value test to `model` alone leaves the "
            "model_config and system_prompt arms unadmitted, and neither of "
            "those arms is COALESCE-only",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    assert_mutation_kills_the_pin(mutation, str(_SELF), tmp_path)


def test_every_pin_has_a_mutation_that_kills_it():
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)
