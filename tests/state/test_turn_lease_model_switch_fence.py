"""The model-switch call flow: provider-visible state written with no fence.

WHAT THE CENSUS DOES NOT SEE
    ``tests/state/test_turn_lease_writer_census`` derives its denominator from
    a statement pattern over ``messages`` / ``message_reactions``. Everything it
    finds is fenced, and it is right about what it looked at. The model switch
    does not write ``messages``.

    It writes ``sessions``:

        UPDATE sessions SET model = ?, model_config = ?,
                            system_prompt = NULL, system_prompt_hash = NULL
        WHERE id = ?

    Those four columns are what the NEXT TURN REPLAYS UNDER. The fence surface
    in ``hermes_state_common`` says so in as many words — "The model, the system
    prompt, the title and the end state all live in ``sessions``, and the next
    turn replays under all four" — and puts ``("sessions", "UPDATE")`` in
    ``TURN_FENCE_SURFACE`` for exactly that reason.

    But that trigger is a GENERATION fence: it locks out a binary that never
    registered the marker. It says nothing about a second CURRENT-generation
    writer. So while one process is mid-turn on a conversation, another process
    in the same generation could change the model out from under it, drop the
    system prompt, and the turn would finish and persist under a route it never
    ran on. Nothing refused, nothing logged.

WHY THESE THREE MUTATORS AND NOT THE WHOLE ``sessions`` SURFACE
    Fifty-five ``SessionDB`` methods write ``sessions``; six of them reach the
    turn-lease guard. Fencing all fifty-five is not this slice — most of them
    write bookkeeping the provider never sees (read stamps, pin/hide flags,
    token counters, handoff state), and a fence on those would refuse routine
    maintenance for no gain.

    The three here are the ones the PRODUCTION MODEL-SWITCH FLOW calls, traced
    from the call sites rather than picked from the method list:

    * ``cli.py`` ``_persist_model_switch_to_session`` → ``update_session_model``
      + ``patch_session_model_config``
    * ``gateway/slash_commands.py`` ``/model`` → ``update_session_model``
    * ``tui_gateway/server.py`` ``_persist_live_session_runtime`` →
      ``update_session_model``
    * ``gateway/run.py`` ``_sync_session_model_from_agent`` and
      ``tui_gateway/server.py`` ``_persist_live_session_runtime`` →
      ``update_session_meta``, both as a read-modify-write: ``get_session`` for
      the current ``model_config``, merge, write back.

    That last shape is why a refusal is the right closure and not merely a
    nicety. The snapshot is read outside any fence, mutated in memory, and
    written back; a turn that lands in between is silently overwritten. There
    is no grant the flow could hold that would make the stale write correct, so
    the write has to be refused while somebody owns the conversation.

THE RESIDUE, MEASURED RATHER THAN ARGUED
    The other forty-nine are not exempted here and no list of them is kept.
    ``tests/state/test_turn_lease_session_replay_columns`` is not written; what
    exists is the measurement that produced the numbers above, and the finding
    is reported as a bounded gap rather than papered over with an allowlist.

WHAT EACH PIN ASSERTS
    Rows, not exceptions. A refusal that raises and writes anyway is the defect
    with a log line attached, so every check compares the four replay columns
    before and after, and every check also proves the AUTHORIZED write lands —
    a fence that refuses everybody passes a refusal test perfectly.
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

#: The columns the next turn replays under. Read off the statement
#: ``update_session_model`` executes, which is the method whose entire purpose
#: is "change what the model sees".
REPLAY_COLUMNS = ("model", "model_config", "system_prompt", "system_prompt_hash")


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _store(tmpdir, name="state.db"):
    from hermes_state import SessionDB

    return SessionDB(db_path=pathlib.Path(tmpdir) / name)


def _replay_state(db, session_id):
    """The provider-visible route for *session_id*, as VALUES."""
    row = db.get_session(session_id)
    if row is None:
        return None
    return tuple(row[column] for column in REPLAY_COLUMNS)


def _owned_session(db, session_id="s", *, tag="owner"):
    """A conversation with a LIVE owner: a grant this process is holding.

    Same PID on purpose. ``_turn_lease_owner_is_dead`` answers "not dead" for
    our own pid, so the row is held only because the process-local registry
    says a grant is live — which is the strongest form of "owned" available
    without a second process, and the one a subprocess cannot produce.
    """
    db.create_session(session_id, "test")
    db.append_message(session_id, "user", f"{session_id} context")
    db.update_session_model(session_id, "anthropic/claude-before")
    grant = db.try_acquire_session_turn_lease(
        session_id, _holder(tag), ttl_seconds=600
    )
    assert grant, f"could not take the lease on {session_id!r}"
    return grant


# ---------------------------------------------------------------------------
# The properties.
# ---------------------------------------------------------------------------

def _the_in_transaction_guard_still_refuses(db, session_id, write) -> None:
    """Take the lease DURING the flush, i.e. AFTER the advisory read.

    ``update_session_model`` and ``update_session_meta`` now carry two
    admission points: ``_refuse_before_side_effects`` before the token-queue
    barrier, and ``_check_turn_lease_guard`` inside ``BEGIN IMMEDIATE``. With
    both in place, "a bystander is refused" is satisfied by either, so removing
    the in-transaction guard stopped killing these pins — the harness reported
    exactly that, and it was right: the pins had become claims about the pair
    rather than about the guard.

    So each pin also drives this leg. The conversation is FREE when the call
    starts, and the lease is taken inside ``flush_token_counts``, which runs
    between the advisory read and the write transaction. The advisory check has
    already returned "no finding" and cannot be consulted again; only the guard
    inside the transaction can refuse. That is the property worth pinning
    anyway: the early refusal exists to protect side effects, it is not the
    authority.
    """
    from hermes_state import SessionTurnLeaseLostError

    db.create_session("window", "test")
    db.append_message("window", "user", "window context")
    db.update_session_model("window", "anthropic/claude-before")
    before = _replay_state(db, "window")
    assert db.get_session_turn_lease("window") is None, (
        "the conversation must start FREE or the advisory check refuses first "
        "and this leg measures the wrong guard"
    )

    taken = {}
    original_flush = db.flush_token_counts

    def _flush_and_take_the_lease(*args, **kwargs):
        db.flush_token_counts = original_flush
        taken["grant"] = db.try_acquire_session_turn_lease(
            "window", _holder("landed-mid-call"), ttl_seconds=600
        )
        return original_flush(*args, **kwargs)

    db.flush_token_counts = _flush_and_take_the_lease
    try:
        write(db, "window")
    except SessionTurnLeaseLostError:
        pass
    else:
        raise AssertionError(
            "a write was admitted although the conversation was taken between "
            "the advisory read and the write transaction. The advisory check "
            "cannot see that, which is why it is not the authority"
        )
    finally:
        db.flush_token_counts = original_flush
    assert taken.get("grant"), (
        "the lease was never taken, so the window was never opened and this "
        "leg proves nothing"
    )
    assert _replay_state(db, "window") == before, (
        f"the refused write changed the route anyway: "
        f"{_replay_state(db, 'window')!r} != {before!r}"
    )


def check_a_model_switch_is_refused_while_a_live_owner_holds_it(tmpdir) -> None:
    """/model against a conversation somebody is mid-turn on.

    The write nulls ``system_prompt`` and ``system_prompt_hash`` as well as
    setting the model, so admitting it mid-turn does not just change the route:
    it deletes the assembled prompt the running turn is replaying under.
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        grant = _owned_session(db)
        before = _replay_state(db, "s")
        assert before[0] == "anthropic/claude-before"

        try:
            db.update_session_model("s", "anthropic/claude-stolen")
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a holderless model switch was admitted while another writer "
                "held the conversation's turn lease"
            )
        assert _replay_state(db, "s") == before, (
            f"the refused switch changed the route anyway: "
            f"{_replay_state(db, 's')!r} != {before!r}"
        )

        # The owner's own switch DOES land, or the refusal above is
        # indistinguishable from a path that refuses everybody.
        db.update_session_model(
            "s", "anthropic/claude-owner", turn_lease_holder=grant
        )
        after = _replay_state(db, "s")
        assert after[0] == "anthropic/claude-owner", (
            f"the owner's own model switch was refused: {after!r}"
        )

        # And on a FREE conversation a holderless switch is still legal —
        # fresh sessions, imports and single-writer installs depend on it.
        db.release_session_turn_lease("s", grant)
        db.update_session_model("s", "anthropic/claude-free")
        assert _replay_state(db, "s")[0] == "anthropic/claude-free", (
            "holderless model switches are refused even on a free "
            "conversation, which breaks every single-writer install"
        )

        _the_in_transaction_guard_still_refuses(
            db, "s",
            lambda d, sid: d.update_session_model(sid, "anthropic/claude-window"),
        )
    finally:
        db.close()


def check_a_model_config_patch_is_refused_while_a_live_owner_holds_it(
    tmpdir,
) -> None:
    """``patch_session_model_config`` is a read-merge-write on the same column.

    It is the standalone setter for callers that change ``model_config``
    without touching the transcript — the /model commit path uses it right
    after ``update_session_model`` — so leaving it open reopens the same hole
    one method over.
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        grant = _owned_session(db)
        before = _replay_state(db, "s")

        try:
            db.patch_session_model_config("s", {"provider": "smuggled"})
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a holderless model_config patch was admitted while another "
                "writer held the conversation's turn lease"
            )
        assert _replay_state(db, "s") == before, (
            f"the refused patch changed model_config anyway: "
            f"{_replay_state(db, 's')!r} != {before!r}"
        )

        db.patch_session_model_config(
            "s", {"provider": "owner"}, turn_lease_holder=grant
        )
        merged = json.loads(_replay_state(db, "s")[1] or "{}")
        assert merged.get("provider") == "owner", (
            f"the owner's own patch was refused: {merged!r}"
        )
    finally:
        db.close()


def check_a_meta_write_is_refused_while_a_live_owner_holds_it(tmpdir) -> None:
    """``update_session_meta`` is the read-modify-write the gateway does.

    ``_sync_session_model_from_agent`` reads ``model_config`` with
    ``get_session``, merges a runtime block into the parsed dict, and writes
    the whole thing back. Nothing about that sequence is atomic against another
    process's turn, and there is no grant the flow could hold that would make
    the stale write correct — so the only correct outcome is refusal.
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        grant = _owned_session(db)
        before = _replay_state(db, "s")

        try:
            db.update_session_meta(
                "s", json.dumps({"gateway_runtime": "stale"}),
                model="anthropic/claude-stale",
            )
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a holderless meta write was admitted while another writer "
                "held the conversation's turn lease"
            )
        assert _replay_state(db, "s") == before, (
            f"the refused meta write changed the route anyway: "
            f"{_replay_state(db, 's')!r} != {before!r}"
        )

        db.update_session_meta(
            "s", json.dumps({"gateway_runtime": "owner"}),
            model="anthropic/claude-owner",
            turn_lease_holder=grant,
        )
        after = _replay_state(db, "s")
        assert after[0] == "anthropic/claude-owner", (
            f"the owner's own meta write was refused: {after!r}"
        )

        _the_in_transaction_guard_still_refuses(
            db, "s",
            lambda d, sid: d.update_session_meta(
                sid, json.dumps({"gateway_runtime": "window"}),
                model="anthropic/claude-window",
            ),
        )
    finally:
        db.close()


def check_a_foreign_root_grant_cannot_switch_the_model(tmpdir) -> None:
    """A live grant for ANOTHER conversation is not authority on this one.

    Both grants use the SAME holder string and are both the first grant in
    their conversation, so they carry the same epoch: only the conversation
    root can tell them apart. That is deliberate — a check that gave them
    different holders would be refused by the holder comparison and would pass
    while asserting nothing about the root.
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        one_identity = _holder("switcher")
        foreign = _owned_session(db, "theirs", tag="switcher")
        mine = _owned_session(db, "mine", tag="switcher")
        assert str(foreign) == str(mine) == one_identity, (
            "the two grants must share a holder string or the ROOT comparison "
            "is never exercised"
        )
        assert foreign.epoch == mine.epoch, (
            "both first grants should be the same epoch; if not, this check no "
            "longer isolates the root comparison"
        )
        assert foreign.conversation_id != mine.conversation_id

        before = _replay_state(db, "mine")
        try:
            db.update_session_model(
                "mine", "anthropic/claude-smuggled", turn_lease_holder=foreign
            )
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a grant issued for another conversation switched this one's "
                "model"
            )
        assert _replay_state(db, "mine") == before
        assert _replay_state(db, "theirs")[0] == "anthropic/claude-before", (
            "the misdirected switch also disturbed the grant's own "
            "conversation"
        )

        db.update_session_model(
            "mine", "anthropic/claude-legitimate", turn_lease_holder=mine
        )
        assert _replay_state(db, "mine")[0] == "anthropic/claude-legitimate"
    finally:
        db.close()


PINS = {
    "check_a_model_switch_is_refused_while_a_live_owner_holds_it":
        check_a_model_switch_is_refused_while_a_live_owner_holds_it,
    "check_a_model_config_patch_is_refused_while_a_live_owner_holds_it":
        check_a_model_config_patch_is_refused_while_a_live_owner_holds_it,
    "check_a_meta_write_is_refused_while_a_live_owner_holds_it":
        check_a_meta_write_is_refused_while_a_live_owner_holds_it,
    "check_a_foreign_root_grant_cannot_switch_the_model":
        check_a_foreign_root_grant_cannot_switch_the_model,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_model_switch_property(name, tmp_path):
    """The pin. Each property, asserted against the tree under test."""
    PINS[name](tmp_path)


def _guard_block(comment: str) -> str:
    """The admission call as it appears in one mutator, with its own comment.

    Keyed by the comment line so each row names ONE guard: the call itself is
    character-identical in every mutator, and an anchor that matches three
    places names none of them.
    """
    return (
        f"            # {comment}\n"
        "            self._check_turn_lease_guard(\n"
        "                conn,\n"
        "                session_id,\n"
        "                turn_lease_holder,\n"
        "                turn_lease_ttl_seconds=turn_lease_ttl_seconds,\n"
        "            )\n"
    )


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_a_model_switch_is_refused_while_a_live_owner_holds_it",
        module="hermes_state.py",
        find=_guard_block(
            "The columns below are what the next turn replays under."
        ),
        replace="",
        why="without the admission call the method is what it was before: a "
            "context-bearing write with no fence, admitted while another "
            "process is mid-turn on the conversation",
    ),
    Mutation(
        pin="check_a_model_config_patch_is_refused_while_a_live_owner_holds_it",
        module="hermes_state.py",
        find=_guard_block(
            "model_config is replayed by the next turn."
        ),
        replace="",
        why="the standalone model_config setter is the same hole one method "
            "over; leaving it unguarded reopens what update_session_model "
            "just closed",
    ),
    Mutation(
        pin="check_a_meta_write_is_refused_while_a_live_owner_holds_it",
        module="hermes_state.py",
        find=_guard_block(
            "model and model_config are replayed by the next turn."
        ),
        replace="",
        why="this is the read-modify-write the gateway does; unguarded, a "
            "snapshot read outside any fence is written back over a turn that "
            "landed in between",
    ),
    Mutation(
        pin="check_a_foreign_root_grant_cannot_switch_the_model",
        module="hermes_state.py",
        find="        if granted_root != conversation_id:\n            return None\n",
        replace="        if False:\n            return None\n",
        why="the root comparison is the only cross-conversation check; holder "
            "and epoch both match trivially between two first grants",
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
