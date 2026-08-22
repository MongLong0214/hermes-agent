"""Slice 2 — a grant is bound to ONE conversation root and only that one.

Property under test: authority is scoped to the conversation the grant was
issued for. The generation from slice 1 orders acquisitions *within* a
conversation; on its own it says nothing *across* conversations — and a first
grant is epoch 1 in every conversation, so for the common case the epoch
discriminates nothing at all. Holder+epoch alone therefore makes the grant
holder-scoped, which is not the property it exists to provide.

Every check here uses the SAME holder string and the SAME epoch in both
conversations, so only the root can distinguish them.
"""

from __future__ import annotations

import os

import pytest

from hermes_state import SessionDB, SessionTurnLeaseLostError


def _holder(tag: str = "shared-holder") -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _two_conversations(tmp_path):
    """Two roots, one holder string, one epoch each — nothing but the root."""
    db = SessionDB(tmp_path / "state.db")
    for sid in ("a", "b"):
        db.create_session(sid, source="test")
        db.append_message(sid, "user", f"{sid} context")
    holder = _holder()
    token_a = db.try_acquire_session_turn_lease("a", holder, ttl_seconds=300)
    token_b = db.try_acquire_session_turn_lease("b", holder, ttl_seconds=300)
    assert token_a is not None and token_b is not None
    assert str(token_a) == str(token_b), "holder strings must be identical here"
    assert token_a.epoch == token_b.epoch == 1, (
        "a first grant is epoch 1 in every conversation — that is the point"
    )
    return db, token_a, token_b


def _contents(db, sid):
    return [m["content"] for m in db.get_messages(sid)]


def test_a_grant_records_the_conversation_root_it_was_issued_for(tmp_path):
    db, token_a, token_b = _two_conversations(tmp_path)
    assert getattr(token_a, "conversation_id", None) == "a"
    assert getattr(token_b, "conversation_id", None) == "b"


def test_a_grant_for_another_root_cannot_append(tmp_path):
    db, token_a, token_b = _two_conversations(tmp_path)
    with pytest.raises(SessionTurnLeaseLostError):
        db.append_message("b", "assistant", "cross-root", turn_lease_holder=token_a)
    assert _contents(db, "a") == ["a context"]
    assert _contents(db, "b") == ["b context"]


def test_a_grant_for_another_root_cannot_batch_append(tmp_path):
    db, token_a, token_b = _two_conversations(tmp_path)
    with pytest.raises(SessionTurnLeaseLostError):
        db.append_messages_batch(
            "b", [{"role": "assistant", "content": "cross-root"}],
            turn_lease_holder=token_a,
        )
    assert _contents(db, "a") == ["a context"]
    assert _contents(db, "b") == ["b context"]


def test_a_grant_for_another_root_cannot_refresh(tmp_path):
    db, token_a, token_b = _two_conversations(tmp_path)
    assert db.refresh_session_turn_lease("b", token_a, ttl_seconds=600) is False
    assert db.refresh_session_turn_lease("a", token_a, ttl_seconds=600) is True


def test_a_grant_for_another_root_cannot_release(tmp_path):
    """Release authority must not come from the caller-supplied session id."""
    db, token_a, token_b = _two_conversations(tmp_path)
    db.release_session_turn_lease("b", token_a)
    assert db.get_session_turn_lease("b")["holder"] == str(token_b), (
        "a grant for another conversation cleared this one's lease"
    )
    assert db.get_session_turn_lease("a")["holder"] == str(token_a), (
        "the misdirected release also disturbed the grant's own conversation"
    )
    # ...and the real owner of B is still the only writer B accepts.
    assert db.append_message("b", "assistant", "b owner", turn_lease_holder=token_b)
    assert _contents(db, "b") == ["b context", "b owner"]


def test_a_grant_for_another_root_cannot_release_by_aiming_at_a_rotated_tip(
    tmp_path
):
    """The caller-supplied id is walked to a root; that walk is not authority."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", source="test")
    db.end_session("root", "compression")
    db.create_session("tip", source="test", parent_session_id="root")
    db.create_session("other", source="test")
    holder = _holder()
    token_other = db.try_acquire_session_turn_lease("other", holder, ttl_seconds=300)
    token_root = db.try_acquire_session_turn_lease("root", holder, ttl_seconds=300)
    assert token_other is not None and token_root is not None

    db.release_session_turn_lease("tip", token_other)
    assert db.get_session_turn_lease("tip")["holder"] == str(token_root)
    assert db.get_session_turn_lease("other")["holder"] == str(token_other)


def test_a_grant_carrying_no_root_is_refused(tmp_path):
    """A hand-built holder+epoch pair is not a grant."""
    from hermes_state import SessionTurnLeaseToken

    db, token_a, token_b = _two_conversations(tmp_path)
    rootless = SessionTurnLeaseToken(str(token_b), token_b.epoch)
    assert getattr(rootless, "conversation_id", None) is None
    with pytest.raises(SessionTurnLeaseLostError):
        db.append_message("b", "assistant", "rootless", turn_lease_holder=rootless)
    assert db.refresh_session_turn_lease("b", rootless, ttl_seconds=600) is False
    db.release_session_turn_lease("b", rootless)
    assert db.get_session_turn_lease("b")["holder"] == str(token_b)
    assert _contents(db, "b") == ["b context"]


def test_the_owner_of_each_root_is_unaffected_by_the_binding(tmp_path):
    """Root binding must not cost the real owner anything."""
    db, token_a, token_b = _two_conversations(tmp_path)
    assert db.append_message("a", "assistant", "a owner", turn_lease_holder=token_a)
    assert db.append_message("b", "assistant", "b owner", turn_lease_holder=token_b)
    assert _contents(db, "a") == ["a context", "a owner"]
    assert _contents(db, "b") == ["b context", "b owner"]
    db.release_session_turn_lease("a", token_a)
    db.release_session_turn_lease("b", token_b)
    assert db.get_session_turn_lease("a")["holder"] == ""
    assert db.get_session_turn_lease("b")["holder"] == ""


def test_a_grant_still_works_across_its_own_lineage_after_rotation(tmp_path):
    """Binding is to the ROOT, not to the segment id the caller happens to use."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", source="test")
    db.append_message("root", "user", "root context")
    token = db.try_acquire_session_turn_lease("root", _holder(), ttl_seconds=300)
    assert token is not None
    db.end_session("root", "compression")
    db.create_session("tip", source="test", parent_session_id="root")

    assert db.append_message(
        "tip", "assistant", "post-rotation", turn_lease_holder=token
    )
    assert db.refresh_session_turn_lease("tip", token, ttl_seconds=300) is True
    db.release_session_turn_lease("tip", token)
    assert db.get_session_turn_lease("root")["holder"] == ""
