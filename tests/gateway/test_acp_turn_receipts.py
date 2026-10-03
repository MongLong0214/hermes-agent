"""U4 H1/H3 — the Telegram-turn receipt store and its state machine.

Writing (claim/settle) and the dead-owner sweep are exercised directly against a real SessionDB;
the GET route (``handle_telegram_receipt_lookup``) is covered separately in
``test_canonical_surface_telegram_receipt_route.py``. The managed-ingress seam that will call
``claim_pending``/``settle_completed`` from a live turn (H2) is not wired yet — see
``is_acp_managed_message``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway import acp_turn_receipts as receipts
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path: Path):
    path = tmp_path / "state.db"
    database = SessionDB(path)
    yield database
    database.close()


def _proof(database: SessionDB) -> dict:
    return {"proven_db_path": database.db_path, "proven_db_identity": database._db_file_identity}


def test_a_never_claimed_update_reads_not_found(db):
    assert receipts.lookup(db, "123").status == "NEVER_FOUND"


def test_claim_then_lookup_reads_pending_with_its_identity(db):
    claimed, owner = receipts.claim_pending(
        db, "123", message_id="m1", turn_request_id="t1", receipt_identity={"a": 1}, **_proof(db),
    )
    assert claimed is True
    receipt = receipts.lookup(db, "123")
    assert receipt.status == "PENDING"
    assert receipt.message_id == "m1" and receipt.turn_request_id == "t1"
    assert receipt.receipt_identity == {"a": 1}
    # The response never reads as COMPLETED/ABORTED while pending.
    assert receipt.to_response()["status"] == "PENDING"


def test_a_second_claim_for_the_same_update_loses_and_learns_the_winners_owner(db):
    _, owner_a = receipts.claim_pending(
        db, "123", message_id="m1", turn_request_id="t1", receipt_identity={}, owner="owner-a", **_proof(db),
    )
    claimed_b, learned_owner = receipts.claim_pending(
        db, "123", message_id="m1", turn_request_id="t2", receipt_identity={}, owner="owner-b", **_proof(db),
    )
    assert claimed_b is False
    assert learned_owner == owner_a == "owner-a"
    # The loser's turn_request_id never overwrote the winner's claim.
    assert receipts.lookup(db, "123").turn_request_id == "t1"


def test_settle_completed_replaces_pending_with_the_terminal_and_digest(db):
    receipts.claim_pending(
        db, "123", message_id="m1", turn_request_id="t1", receipt_identity={}, **_proof(db),
    )
    ok = receipts.settle_completed(
        db, "123", receipt_id="hermes-tg:ob1", evidence_digest="sha256:abc",
        content="final text", delivery={"chat_id": "c1", "state": "delivered"}, **_proof(db),
    )
    assert ok is True
    receipt = receipts.lookup(db, "123")
    assert receipt.status == "COMPLETED"
    assert receipt.receipt_id == "hermes-tg:ob1" and receipt.evidence_digest == "sha256:abc"
    assert receipt.content == "final text" and receipt.delivery == {"chat_id": "c1", "state": "delivered"}
    assert receipt.completed_at is not None
    # PENDING's bytes are gone: a second settle on the same (now-stale) pending value is refused.
    assert receipts.settle_completed(
        db, "123", receipt_id="hermes-tg:ob2", evidence_digest="sha256:xyz", **_proof(db),
    ) is False
    assert receipts.lookup(db, "123").receipt_id == "hermes-tg:ob1"  # the first settlement stands


def test_settle_aborted_is_terminal_and_sticky(db):
    receipts.claim_pending(
        db, "1", message_id="m", turn_request_id="t", receipt_identity={}, **_proof(db),
    )
    assert receipts.settle_aborted(db, "1", reason_code="SOMETHING_FAILED", **_proof(db)) is True
    receipt = receipts.lookup(db, "1")
    assert receipt.status == "ABORTED" and receipt.reason_code == "SOMETHING_FAILED"
    # A later COMPLETED attempt on the same key does not overwrite an already-terminal ABORTED —
    # this is CEO 3c058be8's "IN_DOUBT/terminal preserved on failed settlement" invariant in
    # miniature: once ABORTED is recorded, nothing silently promotes it.
    assert receipts.settle_completed(db, "1", receipt_id="x", evidence_digest="y", **_proof(db)) is False
    assert receipts.lookup(db, "1").status == "ABORTED"


def test_refuse_before_run_claims_and_settles_in_one_call(db):
    receipts.refuse_before_run(
        db, "1", message_id="m", turn_request_id="t", receipt_identity={}, **_proof(db),
    )
    receipt = receipts.lookup(db, "1")
    assert receipt.status == "ABORTED" and receipt.reason_code == "REFUSED_BEFORE_RUN"


def test_refuse_before_run_does_not_clobber_an_already_claimed_receipt(db):
    """A concurrent attempt already owns this update id; giving up must not race its settlement."""
    receipts.claim_pending(
        db, "1", message_id="m", turn_request_id="t", receipt_identity={}, owner="live-owner", **_proof(db),
    )
    receipts.refuse_before_run(
        db, "1", message_id="m", turn_request_id="t2", receipt_identity={}, **_proof(db),
    )
    assert receipts.lookup(db, "1").status == "PENDING"  # untouched — still the live owner's claim


class TestLateCommitAfterAdmissionTimeout:
    """The focused reproduction CEO 3c058be8 asked for: an admission that timed out and was
    tombstoned REFUSED_BEFORE_RUN, followed by a late commit from the turn that actually ran.
    Zero duplicate execution means the late commit's settlement is refused, not re-applied."""

    def test_a_late_completion_after_the_timeout_tombstone_is_refused(self, db):
        receipts.claim_pending(
            db, "1", message_id="m", turn_request_id="t", receipt_identity={}, owner="o1", **_proof(db),
        )
        # Admission gave up waiting and tombstoned REFUSED_BEFORE_RUN before the real run answered.
        assert receipts.settle_aborted(db, "1", reason_code="ADMISSION_TIMEOUT", **_proof(db))
        # The turn that was actually running finishes late and tries to settle COMPLETED.
        late_commit_ok = receipts.settle_completed(
            db, "1", receipt_id="hermes-tg:ob1", evidence_digest="sha256:late", **_proof(db),
        )
        assert late_commit_ok is False
        assert receipts.lookup(db, "1").status == "ABORTED"  # never silently promoted, never re-run


class TestDeadOwnerSweep:
    def test_a_pending_receipt_with_no_resuming_owner_becomes_aborted_process_died(self, db):
        receipts.claim_pending(
            db, "1", message_id="m", turn_request_id="t", receipt_identity={}, owner="gone-owner", **_proof(db),
        )
        aborted = receipts.sweep_dead_owner_receipts(db, resuming_owners=(), **_proof(db))
        assert aborted == ["1"]
        receipt = receipts.lookup(db, "1")
        assert receipt.status == "ABORTED" and receipt.reason_code == "HERMES_PROCESS_DIED_BEFORE_ANSWER"

    def test_a_pending_receipt_whose_owner_is_being_resumed_is_left_pending(self, db):
        receipts.claim_pending(
            db, "1", message_id="m", turn_request_id="t", receipt_identity={}, owner="resuming-owner", **_proof(db),
        )
        aborted = receipts.sweep_dead_owner_receipts(db, resuming_owners=("resuming-owner",), **_proof(db))
        assert aborted == []
        assert receipts.lookup(db, "1").status == "PENDING"

    def test_an_already_terminal_receipt_is_never_touched_by_the_sweep(self, db):
        receipts.claim_pending(
            db, "1", message_id="m", turn_request_id="t", receipt_identity={}, owner="o", **_proof(db),
        )
        receipts.settle_completed(db, "1", receipt_id="hermes-tg:ob1", evidence_digest="sha256:a", **_proof(db))
        aborted = receipts.sweep_dead_owner_receipts(db, resuming_owners=(), **_proof(db))
        assert aborted == []
        assert receipts.lookup(db, "1").status == "COMPLETED"


def test_is_acp_managed_message_is_always_false_until_the_boundary_is_decided():
    # H2 seam predicate: inert by construction until CEO 3c058be8's boundary is decided.
    assert receipts.is_acp_managed_message(object(), object()) is False
