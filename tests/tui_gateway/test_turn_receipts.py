"""Focused tests for the dormant durable terminal-receipt adapter."""

from __future__ import annotations

import json
import sqlite3

from hermes_state import SessionDB
from tui_gateway.turn_receipts import TurnReceiptAdapter


def _opened_adapter(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session", source="test")
    return db, TurnReceiptAdapter(db)


def test_prepare_get_claim_and_finish_terminal_receipt(tmp_path):
    db, receipts = _opened_adapter(tmp_path)
    try:
        prepared = receipts.prepare("session", "request-1")
        assert prepared["status"] == "PREPARED"
        assert receipts.get("session", "request-1") == prepared
        assert receipts.status("other-session", "request-1") is None

        token, claimed = receipts.claim(
            "session", "request-1", claim_token="claim-1"
        )
        assert token == "claim-1"
        assert claimed is not None
        assert claimed["status"] == "CLAIMED"

        completed = receipts.finish(
            "session",
            "request-1",
            token,
            assistant_content="terminal reply",
            response_digest="sha256:" + "a" * 64,
        )
        assert completed["status"] == "COMPLETED"
        assert completed["turnRequestId"] == "request-1"
        assert completed["responseDigest"] == "sha256:" + "a" * 64
        assert completed["terminalMessageId"] is not None
        assert [message["content"] for message in db.get_messages("session")] == [
            "terminal reply"
        ]
    finally:
        db.close()


def test_completed_receipt_is_byte_equivalent_after_restart(tmp_path):
    db, receipts = _opened_adapter(tmp_path)
    try:
        receipts.prepare("session", "request-1")
        token, _ = receipts.claim("session", "request-1", claim_token="claim-1")
        completed = receipts.finish(
            "session",
            "request-1",
            token,
            assistant_content="terminal reply",
            response_digest="sha256:" + "b" * 64,
        )
        before_restart = json.dumps(completed, separators=(",", ":"))
    finally:
        db.close()

    reopened = SessionDB(tmp_path / "state.db")
    try:
        recovered = TurnReceiptAdapter(reopened).get("session", "request-1")
        assert recovered is not None
        assert json.dumps(recovered, separators=(",", ":")) == before_restart
    finally:
        reopened.close()


def test_finish_is_idempotent_for_a_completed_receipt(tmp_path):
    db, receipts = _opened_adapter(tmp_path)
    try:
        receipts.prepare("session", "request-1")
        token, _ = receipts.claim("session", "request-1", claim_token="claim-1")
        first = receipts.finish(
            "session",
            "request-1",
            token,
            assistant_content="terminal reply",
            response_digest="sha256:" + "c" * 64,
        )
        second = receipts.finish(
            "session",
            "request-1",
            token,
            assistant_content="terminal reply",
            response_digest="sha256:" + "c" * 64,
        )

        assert second == first
        assert db.message_count("session") == 1
    finally:
        db.close()


def test_legacy_schema_migration_and_completed_receipt_pruning(tmp_path):
    """A state DB without the new table converges without a sidecar store."""
    path = tmp_path / "state.db"
    initial = SessionDB(path)
    initial.create_session("session", source="test")
    initial.close()
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE turn_receipts")

    db = SessionDB(path)
    receipts = TurnReceiptAdapter(db)
    try:
        assert receipts.migration() == 0
        assert db._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'turn_receipts'"
        ).fetchone()

        receipts.prepare("session", "settled")
        token, _ = receipts.claim("session", "settled", claim_token="claim-settled")
        receipts.finish(
            "session",
            "settled",
            token,
            assistant_content="old reply",
            response_digest="sha256:" + "d" * 64,
        )
        receipts.prepare("session", "pending")
        pending_token, _ = receipts.claim(
            "session", "pending", claim_token="claim-pending"
        )
        assert pending_token == "claim-pending"
        db._conn.execute(
            "UPDATE turn_receipts SET completed_at = 1 WHERE turn_request_id = 'settled'"
        )

        assert receipts.prune("session", completed_before=2) == 1
        assert receipts.get("session", "settled") is None
        pending = receipts.get("session", "pending")
        assert pending is not None
        assert pending["status"] == "CLAIMED"
    finally:
        db.close()
