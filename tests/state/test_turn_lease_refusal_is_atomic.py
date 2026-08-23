"""A refusal must cost nothing. Measured, not argued.

WHAT AN AUDIT FOUND THAT THE REFUSAL PINS DID NOT
    Every pin in this family so far asked the same question — did the row
    change? — and every one of them answered correctly. None of them asked
    what the refused call did on the way to being refused.

    ``update_session_model`` calls ``self.flush_token_counts()`` BEFORE
    ``self._execute_write(_do)``, and the admission check lives inside ``_do``.
    So on a conversation another writer owns:

        the model switch is refused                        (correct)
        the queued token deltas are applied to the DB      (0 -> 9)
        the in-process queue is drained                    (1 -> 0)

    A caller that catches ``SessionTurnLeaseLostError`` and retries later — or
    reports "nothing happened" to a user — is wrong on both counts. The
    accounting has already been spent, and it was spent under the route the
    switch was refused from changing. ``update_session_meta`` and
    ``update_session_billing_route`` carry the identical barrier and the
    identical hole.

    This is the same defect class as the /model divergence
    (``tests/gateway/test_model_switch_memory_db_divergence``): an operation
    that mutates one half of the system before asking whether it is allowed to
    mutate the other. That one was in the gateway; these are in the store.

WHY THE FLUSH CANNOT SIMPLY MOVE
    It is a correctness barrier, not a convenience. A delta enqueued before the
    switch carries the PRE-switch route; applying it after the UPDATE trips
    ``update_token_counts``'s ``first_accounted_route`` branch (the row sees
    ``api_call_count == 0`` plus a route mismatch) and resurrects the old
    model. It also cannot move INSIDE ``_do``: the flush drains a background
    writer thread that takes the write lock itself, so calling it inside an
    open ``BEGIN IMMEDIATE`` deadlocks.

    So the barrier stays where it is and the REFUSAL moves earlier — see
    ``SessionDB._refuse_before_side_effects``. It is advisory: it can refuse a
    write the in-transaction guard would also refuse, and it can admit nothing
    at all. The guard inside the transaction remains the only authority.

WHAT THE PINS ASSERT
    Three values per call, none of them "it raised":

    total_changes   ``sqlite3.Connection.total_changes`` across the refused
                    call. Zero, or the refusal wrote something.
    the queue       ``list(db._token_queue)``, byte-equal before and after.
                    "Deep in-memory state unchanged" is not satisfied by a
                    length comparison; a coalesced batch has the same length.
    the row         the accounting columns the flush would have moved.

    And, for every method, the counterpart: the OWNER's own call DOES flush and
    DOES land. A fix that simply stopped flushing would pass a refusal pin
    perfectly and break the ordering barrier the flush exists for.
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

#: One queued delta, big enough that landing it is unmistakable in the row.
QUEUED_DELTA = {"input_tokens": 9, "api_call_count": 1}


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _store(tmpdir, name="state.db"):
    from hermes_state import SessionDB

    return SessionDB(db_path=pathlib.Path(tmpdir) / name)


def _queue_a_delta(db, session_id: str) -> None:
    """Enqueue one delta with NOTHING running that could drain it.

    Appended under the queue's own condition variable rather than through
    ``queue_token_counts``, which starts the background writer — and a writer
    that drains on its own schedule makes "the queue is unchanged" a race
    rather than a claim. This is the exact state the audit measured: one delta
    queued, no writer, ``flush_token_counts`` about to take it.
    """
    with db._token_queue_cond:
        db._token_queue.append((session_id, dict(QUEUED_DELTA)))


def _accounting(db, session_id: str):
    """The accounting columns, read WITHOUT draining the queue.

    ``get_session`` calls ``flush_token_counts()`` first — that is the whole
    point of the queue, readers must see exact mid-turn totals. So the obvious
    spelling of this helper applies the very deltas the pin is about to claim
    were not applied, and the first version of this file measured its own probe:
    the queue was empty before the refused call ever ran. Read the row straight
    off a read connection instead.
    """
    with db._read_ctx() as conn:
        row = conn.execute(
            "SELECT input_tokens, api_call_count FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    return (row["input_tokens"], row["api_call_count"])


def _owned_session(db, session_id="s", *, tag="owner"):
    db.create_session(session_id, "test")
    db.append_message(session_id, "user", f"{session_id} context")
    db.update_session_model(session_id, "anthropic/claude-before")
    grant = db.try_acquire_session_turn_lease(
        session_id, _holder(tag), ttl_seconds=600
    )
    assert grant, f"could not take the lease on {session_id!r}"
    return grant


def _refusal_costs_nothing(tmpdir, bystander, owner_write):
    """The shape every pin below uses.

    Refused: no DB change at all, the queue byte-identical, the accounting
    columns untouched. Admitted: the barrier still runs, so the owner's own
    call lands the queued delta before its own write.
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        grant = _owned_session(db)
        accounting_before = _accounting(db, "s")

        # Queued LAST, after every observation that could drain it.
        _queue_a_delta(db, "s")
        queue_before = [(sid, dict(kw)) for sid, kw in db._token_queue]
        changes_before = db._conn.total_changes
        assert queue_before, "nothing was queued, so the pin measures nothing"

        try:
            bystander(db)
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a holderless context write was admitted while another writer "
                "held the conversation's turn lease"
            )

        changes_after = db._conn.total_changes
        queue_after = [(sid, dict(kw)) for sid, kw in db._token_queue]

        assert changes_after == changes_before, (
            f"the refused call wrote "
            f"{changes_after - changes_before} row(s) to the database before "
            f"being refused. A caller that catches the refusal is entitled to "
            f"believe nothing happened."
        )
        assert queue_after == queue_before, (
            f"the refused call drained the in-process token queue: "
            f"{queue_before!r} -> {queue_after!r}. The deltas carry the "
            f"PRE-switch route and they are now gone."
        )
        assert _accounting(db, "s") == accounting_before, (
            f"the refused call spent the queued usage: "
            f"{_accounting(db, 's')!r} != {accounting_before!r}"
        )

        # ...and the barrier still works for a call that IS admitted, or the
        # fix is "stop flushing" and the ordering guarantee is gone.
        owner_write(db, grant)
        assert not db._token_queue, (
            f"the owner's own call did not drain the queue, so the barrier "
            f"that keeps a pre-switch delta from landing after the switch is "
            f"gone: {list(db._token_queue)!r}"
        )
        assert _accounting(db, "s") == (9, 1), (
            f"the owner's call was admitted but the queued delta never "
            f"landed: {_accounting(db, 's')!r}"
        )
    finally:
        db.close()


def check_a_refused_model_switch_spends_no_queued_usage(tmpdir) -> None:
    """The audit's measurement, kept as a test rather than as a paragraph."""
    _refusal_costs_nothing(
        tmpdir,
        lambda db: db.update_session_model("s", "anthropic/claude-stolen"),
        lambda db, grant: db.update_session_model(
            "s", "anthropic/claude-owner", turn_lease_holder=grant
        ),
    )


def check_a_refused_meta_write_spends_no_queued_usage(tmpdir) -> None:
    """``update_session_meta`` carries the same barrier and the same hole."""
    import json

    _refusal_costs_nothing(
        tmpdir,
        lambda db: db.update_session_meta(
            "s", json.dumps({"gateway_runtime": "stale"}),
            model="anthropic/claude-stale",
        ),
        lambda db, grant: db.update_session_meta(
            "s", json.dumps({"gateway_runtime": "owner"}),
            model="anthropic/claude-owner", turn_lease_holder=grant,
        ),
    )


def check_a_refused_billing_route_write_spends_no_queued_usage(tmpdir) -> None:
    """And so does the billing-route write, now that it is fenced."""
    _refusal_costs_nothing(
        tmpdir,
        lambda db: db.update_session_billing_route(
            "s", provider="smuggler", base_url="https://smuggled.example",
        ),
        lambda db, grant: db.update_session_billing_route(
            "s", provider="owner", base_url="https://owner.example",
            turn_lease_holder=grant,
        ),
    )


PINS = {
    "check_a_refused_model_switch_spends_no_queued_usage":
        check_a_refused_model_switch_spends_no_queued_usage,
    "check_a_refused_meta_write_spends_no_queued_usage":
        check_a_refused_meta_write_spends_no_queued_usage,
    "check_a_refused_billing_route_write_spends_no_queued_usage":
        check_a_refused_billing_route_write_spends_no_queued_usage,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_refusal_is_atomic_property(name, tmp_path):
    """The pin. Each property, asserted against the tree under test."""
    PINS[name](tmp_path)


def _early_refusal(method: str) -> str:
    """The pre-flush refusal as it appears in one method, with its own comment.

    Keyed by the method name in the comment because the call itself is
    character-identical in all three, and an anchor that matches three places
    names none of them.
    """
    return (
        f"        # Refuse before the barrier below spends anything ({method}).\n"
        "        self._refuse_before_side_effects(session_id, turn_lease_holder)\n"
    )


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_a_refused_model_switch_spends_no_queued_usage",
        module="hermes_state.py",
        find=_early_refusal("update_session_model"),
        replace="",
        why="without the early refusal the flush runs first, so a refused "
            "switch has already applied the queued deltas and drained the "
            "queue by the time the in-transaction guard says no",
    ),
    Mutation(
        pin="check_a_refused_meta_write_spends_no_queued_usage",
        module="hermes_state.py",
        find=_early_refusal("update_session_meta"),
        replace="",
        why="same barrier, same hole, different method — the read-modify-write "
            "the gateway does",
    ),
    Mutation(
        pin="check_a_refused_billing_route_write_spends_no_queued_usage",
        module="hermes_state.py",
        find=_early_refusal("update_session_billing_route"),
        replace="",
        why="the billing route write flushes for the same ordering reason and "
            "was fenced in the same slice",
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
