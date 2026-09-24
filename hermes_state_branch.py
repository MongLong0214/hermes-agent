"""The durable half of a branch — the ONE child-row create and history seed that every surface
(gateway /branch, CLI /branch, TUI session.branch, API fork) shares, so a branch never reports a
title, a row or a message count it did not commit.

Plain functions over a duck-typed ``SessionDB`` (stdlib + ``hermes_state_errors`` only), so the
gateway, CLI and TUI call it without pulling each other's import graphs in.
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

from hermes_state_errors import SessionTurnLeaseLostError

# A branch's leases outlive no more than one copy; a crashed holder is reclaimed by TTL or dead pid.
BRANCH_LEASE_TTL_S = 30.0


def branch_lease_holder(stage: str, session_id: str) -> str:
    """A holder no other call can mint: a re-acquire by the SAME holder is a re-entrant success,
    so two branches sharing one would let the first release free the second's lease."""
    return f"pid={os.getpid()}:turn=branch-{stage}-{uuid.uuid4().hex}:session={session_id}"


def create_branch_session(
    db, session_id: str, source: str, *, title: Optional[str], **row,
) -> tuple[bool, Optional[str]]:
    """Strict-create the child with its title in the same transaction; a title the row cannot take
    leaves it untitled instead. ``(created, title_error)``: ``created`` False means the id is taken
    and nothing was written; ``title_error`` says why a created child is untitled."""
    try:
        return db.create_session_strict(session_id, source, title=title, **row), None
    except ValueError as exc:
        return db.create_session_strict(session_id, source, **row), str(exc)


def seed_branch_session(db, session_id: str, rows: list[dict]) -> None:
    """Copy ``rows`` into a child this caller just created, under the child's own turn lease (an
    explicit fork is its own lease root), so a writer that learns the id first cannot interleave.
    SessionTurnLeaseLostError before any write when another holder owns the child; a copy error
    propagates with the chunks already committed in place (bounded-chunk transactions, #23254)."""
    holder = branch_lease_holder("seed", session_id)
    if not db.try_acquire_session_turn_lease(session_id, holder, ttl_seconds=BRANCH_LEASE_TTL_S):
        raise SessionTurnLeaseLostError(f"branch seed lease for {session_id!r} is held")
    try:
        db.append_messages_batch(session_id, rows, turn_lease_holder=holder, chunk_rows=500)
    finally:
        db.release_session_turn_lease(session_id, holder)
