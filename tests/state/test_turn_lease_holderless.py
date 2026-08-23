"""Slice 3 — presenting NOTHING is not a way past the fence.

Property under test: a write that presents no grant is admitted only when
nobody owns the conversation.

The guard runs under ``if turn_lease_holder:``, so it has never been reachable
by a writer that presents nothing — and most writers present nothing. The fence
therefore protects exactly the one caller that opts into it and is silent about
every other, which is the opposite of what an admission check is for.

Holderless writes must stay legal on an UNOWNED conversation: fresh sessions,
imports, branch copies and single-writer installs all write without ever taking
a turn. The rule is about ownership, not about ceremony.
"""

from __future__ import annotations

import os

import pytest

from hermes_state import SessionDB, SessionTurnLeaseLostError


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _seed(db: SessionDB, session_id: str = "s") -> str:
    db.create_session(session_id, source="test")
    db.append_message(session_id, "user", "context")
    return session_id


def _contents(db, sid="s"):
    return [m["content"] for m in db.get_messages(sid)]


def test_a_holderless_append_is_refused_under_a_live_owner(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    other = SessionDB(path)
    _seed(db)
    owner = db.try_acquire_session_turn_lease("s", _holder("owner"), ttl_seconds=300)
    assert owner is not None

    with pytest.raises(SessionTurnLeaseLostError):
        other.append_message("s", "assistant", "unfenced")
    assert _contents(other) == ["context"]


def test_a_holderless_batch_append_is_refused_under_a_live_owner(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    other = SessionDB(path)
    _seed(db)
    owner = db.try_acquire_session_turn_lease("s", _holder("owner"), ttl_seconds=300)
    assert owner is not None

    with pytest.raises(SessionTurnLeaseLostError):
        other.append_messages_batch(
            "s", [{"role": "assistant", "content": "unfenced batch"}]
        )
    assert _contents(other) == ["context"]


def test_a_holderless_write_is_admitted_when_nobody_owns_the_conversation(
    tmp_path
):
    db = SessionDB(tmp_path / "state.db")
    _seed(db)
    assert db.append_message("s", "assistant", "unowned")
    assert db.append_messages_batch(
        "s", [{"role": "assistant", "content": "also unowned"}]
    ) == 1
    assert _contents(db) == ["context", "unowned", "also unowned"]


def test_a_holderless_write_is_admitted_again_after_the_owner_releases(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    _seed(db)
    owner = db.try_acquire_session_turn_lease("s", _holder("owner"), ttl_seconds=300)
    with pytest.raises(SessionTurnLeaseLostError):
        db.append_message("s", "assistant", "during")
    db.release_session_turn_lease("s", owner)
    assert db.append_message("s", "assistant", "after release")
    assert _contents(db) == ["context", "after release"]


def test_a_holderless_write_cannot_get_in_by_aiming_at_the_rotated_tip(tmp_path):
    """The bypass must not be reachable by naming a child segment."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", source="test")
    db.append_message("root", "user", "context")
    db.end_session("root", "compression")
    db.create_session("tip", source="test", parent_session_id="root")

    owner = db.try_acquire_session_turn_lease("root", _holder("owner"), ttl_seconds=300)
    assert owner is not None
    with pytest.raises(SessionTurnLeaseLostError):
        db.append_message("tip", "assistant", "unfenced on tip")
    assert db.get_messages("tip") == []


def test_a_holderless_write_from_a_different_conversation_is_unaffected(tmp_path):
    """Ownership is per conversation; an owned A must not fence an unowned B."""
    db = SessionDB(tmp_path / "state.db")
    _seed(db, "a")
    _seed(db, "b")
    owner = db.try_acquire_session_turn_lease("a", _holder("owner"), ttl_seconds=300)
    assert owner is not None
    assert db.append_message("b", "assistant", "b is free")
    assert _contents(db, "b") == ["context", "b is free"]


def test_a_post_turn_holderless_flush_is_refused_once_another_turn_takes_over(
    tmp_path
):
    """The agent clears its grant in a finally, so a late persist is holderless.

    That is not a bug in the agent — the turn really is over. It becomes one
    only if the conversation has since been taken over, and the holderless rule
    is what tells those two cases apart without changing the finally.
    """
    path = tmp_path / "state.db"
    first = SessionDB(path)
    second = SessionDB(path)
    _seed(first)

    token = first.try_acquire_session_turn_lease(
        "s", _holder("turn-1"), ttl_seconds=300
    )
    assert first.append_message(
        "s", "assistant", "owned write", turn_lease_holder=token
    )
    first.release_session_turn_lease("s", token)          # ← the finally

    # Nobody owns it: the late flush still lands.
    assert first.append_message("s", "assistant", "late but unowned")

    successor = second.try_acquire_session_turn_lease(
        "s", _holder("turn-2"), ttl_seconds=300
    )
    assert successor is not None
    with pytest.raises(SessionTurnLeaseLostError):
        first.append_message("s", "assistant", "interleaved into turn 2")
    assert _contents(second) == ["context", "owned write", "late but unowned"]


def test_a_superseded_grant_under_the_same_holder_string_cannot_write(tmp_path):
    """Regression pin, not a new property.

    Same-holder ABA was closed by slice 2: the guard now runs through
    _authorize_turn_lease_token, which compares the epoch. Pinned here so a
    later simplification of the guard cannot quietly reopen it.
    """
    db = SessionDB(tmp_path / "state.db")
    _seed(db)
    holder = _holder("same-string")
    first = db.try_acquire_session_turn_lease("s", holder, ttl_seconds=300)
    db.release_session_turn_lease("s", first)
    second = db.try_acquire_session_turn_lease("s", holder, ttl_seconds=300)
    assert str(first) == str(second) and first.epoch != second.epoch

    with pytest.raises(SessionTurnLeaseLostError):
        db.append_message("s", "assistant", "replayed", turn_lease_holder=first)
    assert db.append_message(
        "s", "assistant", "current", turn_lease_holder=second
    )
    assert _contents(db) == ["context", "current"]
