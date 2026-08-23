"""Closing, reopening and retiring a session is a context-bearing write.

THE HARM, MEASURED RATHER THAN ARGUED
    ``_check_transcript_write_guards`` raises ``CompressionSessionClosedError``
    for any append to a row whose ``end_reason`` is ``'compression'``. It is a
    correct rule — a compressed segment is closed and its continuation is
    elsewhere — and it is enforced against the appender, not against whoever
    wrote the ``end_reason``.

    ``end_session`` had no admission at all. So, on this tree::

        owner holds a valid grant on 's'
        bystander: end_session('s', 'compression')       -> admitted
        owner:     append_message('s', …, grant)         -> CompressionSessionClosedError

    The owner presented a grant that was still good and lost its turn anyway.
    Nothing in production passes ``'compression'`` from a bystander today; that
    is a mitigation, not a proof, and it is one edit away from being false.

    The same shape runs through the rest of the lifecycle column pair
    (``ended_at`` / ``end_reason``) and the routing identity that rides with
    it: ``reopen_session`` (which also stamps ``_reset_from`` into its
    children's ``model_config`` — a replay column),
    ``promote_to_session_reset``, ``set_expiry_finalized``,
    ``reopen_orphaned_compression_session``,
    ``adopt_orphaned_gateway_session`` (which RETIRES the donor row under
    ``superseded_by_repair``), and the orphan sweep
    ``finalize_orphaned_compression_sessions``.

WHAT THE PINS ASSERT
    Rows and values, and — for the compression case — the consequence rather
    than the column: the owner's own append must still LAND after the refusal.
    A pin that only checked ``end_reason`` would pass against a fence that
    stored the value somewhere else.

    Every refusal pin is paired with the write LANDING: for the owner
    presenting its grant (single-target writers) or for an unaffected session
    in the same pass (the sweep). A fence that refuses everybody satisfies half
    of this file perfectly.
"""

from __future__ import annotations

import os
import pathlib
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


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _store(tmpdir, name="state.db"):
    from hermes_state import SessionDB

    return SessionDB(db_path=pathlib.Path(tmpdir) / name)


def _lifecycle(db, session_id):
    row = db.get_session(session_id)
    return None if row is None else (row["ended_at"], row["end_reason"])


def _owned(db, session_id="s", *, tag="owner"):
    db.create_session(session_id, "test")
    db.append_message(session_id, "user", f"{session_id} context")
    grant = db.try_acquire_session_turn_lease(
        session_id, _holder(tag), ttl_seconds=600
    )
    assert grant, f"could not take the lease on {session_id!r}"
    return grant


def _backdate(db, session_ids) -> None:
    """Age *session_ids* past a sweep cutoff.

    Straight down ``db._conn`` — the idiom every other prune fixture in this
    repository uses — rather than through a foreign handle: the sessions table
    carries a generation trigger that a connection outside this process's
    ``SessionDB`` cannot satisfy, and rather than through a public writer,
    which would put the fixture inside the fence it is setting up for.
    """
    placeholders = ",".join("?" * len(session_ids))
    db._conn.execute(
        f"UPDATE sessions SET started_at = ? WHERE id IN ({placeholders})",
        (time.time() - 1_000_000, *session_ids),
    )
    db._conn.commit()


def _refused(call):
    from hermes_state import SessionTurnLeaseLostError

    try:
        result = call()
    except SessionTurnLeaseLostError:
        return
    raise AssertionError(
        f"the write was admitted while another writer held the "
        f"conversation's turn lease (returned {result!r})"
    )


# ---------------------------------------------------------------------------
# The pins.
# ---------------------------------------------------------------------------

def check_a_bystander_cannot_close_the_owners_session_as_compression(
    tmpdir,
) -> None:
    """The named hole, pinned by its CONSEQUENCE.

    The assertion that matters is not "end_reason is unchanged" — it is that
    the owner can still append with the grant it was holding all along.
    """
    from hermes_state import CompressionSessionClosedError

    db = _store(tmpdir)
    try:
        grant = _owned(db)
        before = _lifecycle(db, "s")
        assert before == (None, None), "the fixture starts already ended"

        _refused(lambda: db.end_session("s", "compression"))

        assert _lifecycle(db, "s") == before, (
            f"the refused close landed anyway: {_lifecycle(db, 's')!r}"
        )
        try:
            db.append_message("s", "assistant", "the reply", turn_lease_holder=grant)
        except CompressionSessionClosedError as exc:  # pragma: no cover
            raise AssertionError(
                f"a bystander closed the owner's conversation out from under a "
                f"valid grant: {exc}"
            )
        assert db.message_count("s") == 2, (
            "the owner's append did not land, so this pin cannot tell a fence "
            "from a broken transcript path"
        )
    finally:
        db.close()


def check_the_owner_can_close_its_own_session_as_compression(tmpdir) -> None:
    """Rotation ends the segment it owns; the fence must not break that."""
    db = _store(tmpdir)
    try:
        grant = _owned(db)
        db.end_session("s", "compression", turn_lease_holder=grant)
        ended_at, end_reason = _lifecycle(db, "s")
        assert end_reason == "compression", (
            f"the owner's own close was refused: end_reason={end_reason!r}"
        )
        assert ended_at is not None
    finally:
        db.close()


def check_a_bystander_cannot_reopen_the_owners_ended_session(tmpdir) -> None:
    """``reopen_session`` clears the boundary AND stamps the children."""
    db = _store(tmpdir)
    try:
        grant = _owned(db)
        db.end_session("s", "compression", turn_lease_holder=grant)
        before = _lifecycle(db, "s")
        assert before[1] == "compression"

        _refused(lambda: db.reopen_session("s"))
        assert _lifecycle(db, "s") == before, (
            f"the refused reopen cleared the boundary anyway: "
            f"{_lifecycle(db, 's')!r} != {before!r}"
        )

        db.reopen_session("s", turn_lease_holder=grant)
        assert _lifecycle(db, "s") == (None, None), (
            "the owner's own reopen was refused"
        )
    finally:
        db.close()


def check_a_bystander_cannot_promote_the_owners_session_to_reset(tmpdir) -> None:
    """``promote_to_session_reset`` overwrites a LIVE row's boundary."""
    db = _store(tmpdir)
    try:
        grant = _owned(db)
        _refused(lambda: db.promote_to_session_reset("s", "idle"))
        assert _lifecycle(db, "s") == (None, None), (
            f"the refused promotion landed anyway: {_lifecycle(db, 's')!r}"
        )

        assert db.promote_to_session_reset(
            "s", "idle", turn_lease_holder=grant
        ) is True, "the owner's own reset boundary was refused"
        assert _lifecycle(db, "s")[1] == "idle"
    finally:
        db.close()


def check_a_bystander_cannot_finalize_the_owners_expiry(tmpdir) -> None:
    """``expiry_finalized`` is what stops recovery resurrecting the row."""
    db = _store(tmpdir)
    try:
        grant = _owned(db)
        before = (db.get_session("s") or {})["expiry_finalized"]

        _refused(lambda: db.set_expiry_finalized("s", True))
        assert (db.get_session("s") or {})["expiry_finalized"] == before, (
            "the refused finalization landed anyway"
        )

        db.set_expiry_finalized("s", True, turn_lease_holder=grant)
        assert (db.get_session("s") or {})["expiry_finalized"] != before, (
            "the owner's own finalization was refused"
        )
    finally:
        db.close()


def check_a_bystander_cannot_retire_the_owners_session_by_adoption(
    tmpdir,
) -> None:
    """Adoption RETIRES the donor — a live-owned row, under a foreign grant.

    The orphan and the donor are two different conversations, so the caller's
    grant on one authorizes nothing about the other: the donor's root has to
    be free, which is the affected-set rule applied to a two-target write.
    """
    db = _store(tmpdir)
    try:
        db.create_session(
            "donor", "telegram", session_key="chat:42", chat_id="42"
        )
        db.append_message("donor", "user", "donor context")
        db.create_session("orphan", "telegram")
        donor_grant = db.try_acquire_session_turn_lease(
            "donor", _holder("donor-owner"), ttl_seconds=600
        )
        assert donor_grant

        before = _lifecycle(db, "donor")
        _refused(lambda: db.adopt_orphaned_gateway_session("orphan", "donor"))
        assert _lifecycle(db, "donor") == before, (
            f"the refused adoption retired the owned donor anyway: "
            f"{_lifecycle(db, 'donor')!r}"
        )
        assert (db.get_session("orphan") or {})["session_key"] is None, (
            "the refused adoption stamped the orphan anyway"
        )

        db.release_session_turn_lease("donor", donor_grant)
        assert db.adopt_orphaned_gateway_session("orphan", "donor") is True, (
            "adoption is refused even when both conversations are FREE"
        )
        assert (db.get_session("orphan") or {})["session_key"] == "chat:42"
    finally:
        db.close()


def check_a_bystander_cannot_reopen_the_owners_compression_orphan(
    tmpdir,
) -> None:
    """``reopen_orphaned_compression_session`` un-ends a closed segment."""
    db = _store(tmpdir)
    try:
        grant = _owned(db)
        db.end_session("s", "compression", turn_lease_holder=grant)
        before = _lifecycle(db, "s")

        _refused(lambda: db.reopen_orphaned_compression_session("s"))
        assert _lifecycle(db, "s") == before, (
            f"the refused reopen landed anyway: {_lifecycle(db, 's')!r}"
        )

        assert db.reopen_orphaned_compression_session(
            "s", turn_lease_holder=grant
        ) is True, "the owner's own orphan recovery was refused"
        assert _lifecycle(db, "s") == (None, None)
    finally:
        db.close()


def check_the_orphan_finalize_sweep_skips_the_owned_conversation(
    tmpdir,
) -> None:
    """A sweep skips the owned row and still finalizes the unowned one."""
    db = _store(tmpdir)
    try:
        for parent, child in (("p1", "c1"), ("p2", "c2")):
            db.create_session(parent, "test")
            db.end_session(parent, "compression")
            db.create_session(child, "test", parent_session_id=parent)
            db.append_message(child, "user", f"{child} context")
        # The sweep only reaches rows older than its 7-day cutoff. Backdated
        # from OUTSIDE the store on purpose: a fixture that reached in through
        # SessionDB's own generic write path would be exercising the very sink
        # this family is closing.
        _backdate(db, ("c1", "c2"))

        grant = db.try_acquire_session_turn_lease(
            "c1", _holder("owner"), ttl_seconds=600
        )
        assert grant, "could not take the lease on the first continuation"

        finalized = db.finalize_orphaned_compression_sessions()

        assert _lifecycle(db, "c1") == (None, None), (
            f"the sweep finalized the conversation a live turn owns: "
            f"{_lifecycle(db, 'c1')!r}"
        )
        assert _lifecycle(db, "c2")[1] == "orphaned_compression", (
            "the sweep skipped the UNOWNED orphan too — a sweep that finalizes "
            "nothing satisfies the assertion above and repairs nothing"
        )
        assert finalized == 1, f"expected exactly c2 to be finalized, got {finalized}"
    finally:
        db.close()


PINS = {
    "check_a_bystander_cannot_close_the_owners_session_as_compression":
        check_a_bystander_cannot_close_the_owners_session_as_compression,
    "check_the_owner_can_close_its_own_session_as_compression":
        check_the_owner_can_close_its_own_session_as_compression,
    "check_a_bystander_cannot_reopen_the_owners_ended_session":
        check_a_bystander_cannot_reopen_the_owners_ended_session,
    "check_a_bystander_cannot_promote_the_owners_session_to_reset":
        check_a_bystander_cannot_promote_the_owners_session_to_reset,
    "check_a_bystander_cannot_finalize_the_owners_expiry":
        check_a_bystander_cannot_finalize_the_owners_expiry,
    "check_a_bystander_cannot_retire_the_owners_session_by_adoption":
        check_a_bystander_cannot_retire_the_owners_session_by_adoption,
    "check_a_bystander_cannot_reopen_the_owners_compression_orphan":
        check_a_bystander_cannot_reopen_the_owners_compression_orphan,
    "check_the_orphan_finalize_sweep_skips_the_owned_conversation":
        check_the_orphan_finalize_sweep_skips_the_owned_conversation,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_session_lifecycle_property(name, tmp_path):
    PINS[name](tmp_path)


def _guard_block(comment: str) -> str:
    """The admission call as it appears in ONE mutator, keyed by its comment.

    The call itself is character-identical in every mutator, so an anchor
    without the comment names none of them.
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
        pin="check_a_bystander_cannot_close_the_owners_session_as_compression",
        module="hermes_state.py",
        find=_guard_block(
            "end_reason='compression' closes the transcript against its own owner."
        ),
        replace="",
        why="without it a bystander writes the end_reason that "
            "_check_transcript_write_guards then enforces against the holder "
            "of a still-valid grant",
    ),
    Mutation(
        pin="check_the_owner_can_close_its_own_session_as_compression",
        module="hermes_state.py",
        find=(
            "                turn_lease_holder,\n"
            "                turn_lease_ttl_seconds=turn_lease_ttl_seconds,\n"
            "            )\n"
            "            conn.execute(\n"
            '                "UPDATE sessions SET ended_at = ?, end_reason = ? "\n'
            '                "WHERE id = ? AND ended_at IS NULL",\n'
        ),
        replace=(
            "                None,\n"
            "                turn_lease_ttl_seconds=turn_lease_ttl_seconds,\n"
            "            )\n"
            "            conn.execute(\n"
            '                "UPDATE sessions SET ended_at = ?, end_reason = ? "\n'
            '                "WHERE id = ? AND ended_at IS NULL",\n'
        ),
        why="dropping the presented grant on the floor makes the guard read "
            "every close as holderless, so the fence refuses the owner's own "
            "rotation — a fence that refuses everybody passes the bystander "
            "pin above perfectly",
    ),
    Mutation(
        pin="check_a_bystander_cannot_reopen_the_owners_ended_session",
        module="hermes_state.py",
        find=_guard_block(
            "Reopening clears the boundary and stamps the children's model_config."
        ),
        replace="",
        why="reopen_session rewrites both lifecycle columns and json_sets "
            "_reset_from into every legacy child's model_config",
    ),
    Mutation(
        pin="check_a_bystander_cannot_promote_the_owners_session_to_reset",
        module="hermes_state.py",
        find=_guard_block(
            "A reset boundary ends a LIVE row, which is the owner's own row."
        ),
        replace="",
        why="promote_to_session_reset targets `ended_at IS NULL` rows, i.e. "
            "precisely the row a live turn is writing",
    ),
    Mutation(
        pin="check_a_bystander_cannot_finalize_the_owners_expiry",
        module="hermes_state.py",
        find=_guard_block(
            "expiry_finalized is what stops recovery resurrecting this row."
        ),
        replace="",
        why="the flag decides whether stale-route recovery may resurrect the "
            "session with its full history",
    ),
    Mutation(
        pin="check_a_bystander_cannot_retire_the_owners_session_by_adoption",
        module="hermes_state.py",
        find=(
            "            self._refuse_if_any_conversation_is_owned(\n"
            "                conn, [donor_id],\n"
            "                f\"refusing to adopt {orphan_id!r} from {donor_id!r}: \"\n"
            "                f\"the donor is\",\n"
            "            )\n"
        ),
        replace="",
        why="the donor is a SECOND conversation the caller's grant says "
            "nothing about, and adoption retires it",
    ),
    Mutation(
        pin="check_a_bystander_cannot_reopen_the_owners_compression_orphan",
        module="hermes_state.py",
        find=_guard_block(
            "Un-ending a segment changes what its conversation replays."
        ),
        replace="",
        why="the recovery path un-ends a row a live turn may be appending to",
    ),
    Mutation(
        pin="check_the_orphan_finalize_sweep_skips_the_owned_conversation",
        module="hermes_state.py",
        find=(
            "            orphans = self._skip_leased_conversations(conn, orphans)[0]\n"
        ),
        replace="",
        why="without the per-row sweep admission the set-based UPDATE closes "
            "every matching row, owned conversations included",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    assert_mutation_kills_the_pin(mutation, str(_SELF), tmp_path)


def test_every_pin_has_a_mutation_that_kills_it():
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)
