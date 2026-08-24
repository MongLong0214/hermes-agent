"""Reactions and the display/context flags, pinned against a live owner.

WHY THE REACTION EXEMPTION WAS A FAIL
    An earlier census listed "reaction write and consume" as an entry on an
    exemption list. It is not exemptible. A reaction PRODUCES consume-once
    model context — ``take_unseen_reactions`` stamps ``seen`` and the
    announcement is delivered to exactly one later turn — so the write creates
    provider-visible state and the read DESTROYS it. Taking it against a
    conversation somebody else owns hands their turn the note and leaves this
    one with nothing, and nothing about that is recoverable: the stamp is the
    only record that it was delivered.

    The source is right today. Nothing in the tree said so, which is the same
    shape the transcript-append guard failed in — the rule was there, the rule
    was right, and the writers that did not go through it were admitted for
    months.

THE DISPLAY STAMP IS NOT PRESENTATION-ONLY, AND THAT IS THE DEFECT HERE
    ``set_latest_matching_message_display_kind`` says in its own docstring that
    "the model still receives role and content unchanged", and that reads as a
    presentation write with nothing at stake. It writes:

        UPDATE messages SET display_kind = ?, display_metadata = ? WHERE id = ?

    ``display_metadata`` is the column reactions live in — ``REACTIONS_METADATA_KEY``
    is a key inside it, deliberately, so reactions survive rewind and compaction
    with the row. The statement REPLACES the column wholesale, and its
    ``display_metadata`` parameter defaults to ``None``, which encodes to NULL.

    So a holderless display stamp on a row that carries an unconsumed reaction
    deletes that reaction. The announcement the next turn was going to make is
    gone, silently, with no error anywhere — and until this change nothing
    stopped a second process from doing it while the first was mid-turn.

    The census counted this method's one production call site as fenced, and it
    was right about the call site: ``tui_gateway/server.py`` wraps it in a lease
    scope. What was unfenced is the MUTATOR, which is what a second caller
    reaches.

WHAT THE PINS ASSERT
    Row values. For the reaction checks that means the reaction list and the
    ``seen`` stamps read back out of ``display_metadata``; for the api-content
    sidecar it means the exact bytes the provider replays. "It raised" is half
    the property and the cheap half.
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


def _owned(db, session_id="s", *, text="the user said this"):
    """A conversation with one user message and a LIVE owner holding the lease.

    Same pid on purpose: ``_turn_lease_owner_is_dead`` answers "not dead" for
    our own pid, so the row is held only because the process-local registry
    says a grant is live. That is the strongest "owned" available without a
    second process and the one a subprocess cannot produce.
    """
    db.create_session(session_id, "test")
    db.append_message(session_id, "user", text)
    row_id = db.latest_message_row_id(session_id, role="user")
    assert row_id is not None, "no user row to react to"
    grant = db.try_acquire_session_turn_lease(
        session_id, _holder("owner"), ttl_seconds=600
    )
    assert grant, "could not take the lease this check depends on"
    return grant, row_id


def _reactions(db, session_id, row_id):
    """The reaction list as VALUES: (emoji, author, seen) per entry."""
    return [
        (r.get("emoji"), r.get("author"), bool(r.get("seen")))
        for r in db.get_message_reactions(session_id, row_id)
    ]


def _api_content(db, session_id, row_id):
    with db._read_ctx() as conn:
        row = conn.execute(
            "SELECT api_content FROM messages WHERE id = ?", (row_id,)
        ).fetchone()
    return None if row is None else row[0]


# ---------------------------------------------------------------------------
# The properties.
# ---------------------------------------------------------------------------

def check_a_reaction_write_is_refused_while_a_live_owner_holds_it(tmpdir) -> None:
    """Setting a reaction is a context PRODUCTION, so it takes the fence."""
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        grant, row_id = _owned(db)
        db.set_message_reaction(
            "s", row_id, "❤️", author="user", turn_lease_holder=grant
        )
        before = _reactions(db, "s", row_id)
        assert before == [("❤️", "user", False)], before

        try:
            db.set_message_reaction("s", row_id, "\U0001f525", author="user")
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a holderless reaction write was admitted while another "
                "writer held the conversation's turn lease"
            )
        assert _reactions(db, "s", row_id) == before, (
            f"the refused reaction changed the row anyway: "
            f"{_reactions(db, 's', row_id)!r} != {before!r}"
        )

        # The owner's own write lands, or the refusal above is a path that
        # refuses everybody.
        db.set_message_reaction(
            "s", row_id, "\U0001f44d", author="agent", turn_lease_holder=grant
        )
        assert ("\U0001f44d", "agent", False) in _reactions(db, "s", row_id)

        # ...and on a FREE conversation a holderless write is still legal.
        db.release_session_turn_lease("s", grant)
        db.set_message_reaction("s", row_id, "\U0001f602", author="user")
        assert ("\U0001f602", "user", False) in _reactions(db, "s", row_id), (
            "holderless reactions are refused even on a free conversation"
        )
    finally:
        db.close()


def check_a_reaction_consume_is_refused_while_a_live_owner_holds_it(
    tmpdir,
) -> None:
    """Consuming is worse than writing: the stamp is delivered exactly once.

    The assertion that matters is not that the call raised — it is that the
    announcement is STILL THERE afterwards. A consume that raised after
    stamping would be indistinguishable from a refusal by the exception alone,
    and the announcement would be gone.
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        grant, row_id = _owned(db)
        db.set_message_reaction(
            "s", row_id, "❤️", author="user", turn_lease_holder=grant
        )
        before = _reactions(db, "s", row_id)
        assert before == [("❤️", "user", False)]

        try:
            db.take_unseen_reactions("s", author="user")
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a holderless consume was admitted while another writer held "
                "the conversation's turn lease"
            )
        assert _reactions(db, "s", row_id) == before, (
            f"the refused consume stamped the reaction seen anyway, so the "
            f"announcement is spent: {_reactions(db, 's', row_id)!r}"
        )

        # The owner takes it, exactly once.
        taken = db.take_unseen_reactions(
            "s", author="user", turn_lease_holder=grant
        )
        assert [t["emoji"] for t in taken] == ["❤️"], taken
        assert _reactions(db, "s", row_id) == [("❤️", "user", True)]
        assert db.take_unseen_reactions(
            "s", author="user", turn_lease_holder=grant
        ) == [], "the announcement was delivered twice"
    finally:
        db.close()


def check_a_display_stamp_cannot_erase_reactions_while_a_live_owner_holds_it(
    tmpdir,
) -> None:
    """The display stamp REPLACES display_metadata, and reactions live in it.

    Its docstring says the model still receives role and content unchanged, and
    that is true and beside the point: the column it overwrites is where the
    consume-once announcement is kept, and the parameter that fills it defaults
    to None. So the "presentation-only" write is a delete of model context, and
    it must not be admitted while another writer owns the conversation.
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        grant, row_id = _owned(db, text="the user said this")
        db.set_message_reaction(
            "s", row_id, "❤️", author="user", turn_lease_holder=grant
        )
        before = _reactions(db, "s", row_id)
        assert before == [("❤️", "user", False)]

        try:
            db.set_latest_matching_message_display_kind(
                "s", role="user", content="the user said this",
                display_kind="synthetic",
            )
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a holderless display stamp was admitted while another writer "
                "held the conversation's turn lease"
            )
        assert _reactions(db, "s", row_id) == before, (
            f"the refused display stamp erased the pending reaction anyway; "
            f"the announcement the next turn was going to make is gone: "
            f"{_reactions(db, 's', row_id)!r}"
        )

        # The owner's own stamp lands — and it still replaces the column, which
        # is the documented behaviour and not what is being changed here.
        assert db.set_latest_matching_message_display_kind(
            "s", role="user", content="the user said this",
            display_kind="synthetic", turn_lease_holder=grant,
        ) is True
        with db._read_ctx() as conn:
            kind = conn.execute(
                "SELECT display_kind FROM messages WHERE id = ?", (row_id,)
            ).fetchone()[0]
        assert kind == "synthetic", kind

        db.release_session_turn_lease("s", grant)
        assert db.set_latest_matching_message_display_kind(
            "s", role="user", content="the user said this",
            display_kind="free-stamp",
        ) is True, "display stamps are refused even on a free conversation"
    finally:
        db.close()


def check_the_api_content_sidecar_is_refused_while_a_live_owner_holds_it(
    tmpdir,
) -> None:
    """``api_content`` is the exact bytes the provider replays.

    Not a flag about the row: the row's substitute. Rewriting it under a
    foreign owner changes what their next turn SENDS while leaving everything
    a reader would look at identical.
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        grant, row_id = _owned(db, text="turn text")
        assert db.set_latest_user_api_content(
            "s", "turn text", "turn text\n\nCONTEXT", turn_lease_holder=grant
        ) == 1
        before = _api_content(db, "s", row_id)
        assert before == "turn text\n\nCONTEXT", before

        try:
            db.set_latest_user_api_content("s", "turn text", "SMUGGLED")
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a holderless api_content rewrite was admitted while another "
                "writer held the conversation's turn lease"
            )
        assert _api_content(db, "s", row_id) == before, (
            f"the refused rewrite changed what the provider replays: "
            f"{_api_content(db, 's', row_id)!r} != {before!r}"
        )
    finally:
        db.close()


PINS = {
    "check_a_reaction_write_is_refused_while_a_live_owner_holds_it":
        check_a_reaction_write_is_refused_while_a_live_owner_holds_it,
    "check_a_reaction_consume_is_refused_while_a_live_owner_holds_it":
        check_a_reaction_consume_is_refused_while_a_live_owner_holds_it,
    "check_a_display_stamp_cannot_erase_reactions_while_a_live_owner_holds_it":
        check_a_display_stamp_cannot_erase_reactions_while_a_live_owner_holds_it,
    "check_the_api_content_sidecar_is_refused_while_a_live_owner_holds_it":
        check_the_api_content_sidecar_is_refused_while_a_live_owner_holds_it,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_reactions_and_flags_property(name, tmp_path):
    """The pin. Each property, asserted against the tree under test."""
    PINS[name](tmp_path)


def _guard_block(*comment_lines: str) -> str:
    """One mutator's admission call, keyed by the comment that precedes it.

    The call itself is character-identical in every mutator, so an anchor made
    of the call alone matches eleven places and names none of them.
    """
    return (
        "".join(f"            # {line}\n" for line in comment_lines)
        + "            self._check_turn_lease_guard(\n"
        "                conn,\n"
        "                session_id,\n"
        "                turn_lease_holder,\n"
        "                turn_lease_ttl_seconds=turn_lease_ttl_seconds,\n"
        "            )\n"
    )


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_a_reaction_write_is_refused_while_a_live_owner_holds_it",
        module="hermes_state.py",
        find=_guard_block(
            "A reaction PRODUCES consume-once model context:",
            "take_unseen_reactions injects it into exactly one later turn. So",
            "it is fenced like any other transcript mutation — including the",
            "row lookup, which must not resolve against a transcript that",
            "moves before the update lands.",
        ),
        replace="",
        why="without the admission call a second process writes reactions into "
            "a conversation somebody else is mid-turn on, which is exactly "
            "what the abandoned 'reactions are exempt' argument permitted",
    ),
    Mutation(
        pin="check_a_reaction_consume_is_refused_while_a_live_owner_holds_it",
        module="hermes_state.py",
        find=_guard_block(
            "CONSUMPTION, not a read: the seen stamp means the announcement is",
            "delivered to exactly one turn. Taking it against a conversation",
            "somebody else owns hands their turn the note and leaves this one",
            "with nothing.",
        ),
        replace="",
        why="the seen stamp is delivered exactly once; unguarded, a bystander "
            "consumes the owner's announcement and the owner gets nothing",
    ),
    Mutation(
        pin="check_a_display_stamp_cannot_erase_reactions_while_a_live_owner_holds_it",
        module="hermes_state.py",
        find=_guard_block(
            "display_metadata is where reactions live, and this statement",
            "REPLACES the column. A holderless stamp therefore deletes a",
            "pending announcement, which is model context however the",
            "docstring describes the write.",
        ),
        replace="",
        why="this method looked presentation-only and is not: the column it "
            "overwrites carries the consume-once reaction announcement",
    ),
    Mutation(
        pin="check_the_api_content_sidecar_is_refused_while_a_live_owner_holds_it",
        module="hermes_state.py",
        find=_guard_block(
            "api_content IS the context: it is the exact bytes replayed to the",
            "provider. Rewriting it under a foreign owner silently changes",
            "what their next turn sends.",
        ),
        replace="",
        why="api_content substitutes for the row when the transcript is "
            "replayed, so an unguarded rewrite changes what a foreign owner's "
            "turn sends while every visible field stays identical",
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
