"""Focused tests for the dormant durable terminal-receipt adapter."""

from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from hermes_state import (
    SessionDB,
    TurnReceiptConflictError,
    TurnReceiptFenceError,
)
from tui_gateway.turn_receipts import ReceiptRequest, TurnReceiptAdapter


def _opened_adapter(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session", source="test")
    return db, TurnReceiptAdapter(db)


def test_prepare_rejects_a_different_binding_digest_for_the_same_request(tmp_path):
    """A request id is not reusable for a differently bound turn."""
    db, receipts = _opened_adapter(tmp_path)
    try:
        receipts.prepare("session", "request-1", "binding-a")

        with pytest.raises(TurnReceiptConflictError):
            receipts.prepare("session", "request-1", "binding-b")
    finally:
        db.close()


def test_prepare_get_claim_and_finish_terminal_receipt(tmp_path):
    db, receipts = _opened_adapter(tmp_path)
    try:
        binding_digest = "binding:request-1"
        prepared = receipts.prepare("session", "request-1", binding_digest)
        assert prepared["status"] == "PREPARED"
        assert receipts.prepare("session", "request-1", binding_digest) == prepared
        assert receipts.get("session", "request-1", binding_digest) == prepared
        assert receipts.status("session", "request-1", binding_digest) == prepared
        assert {"claimToken", "bindingDigest"}.isdisjoint(prepared)

        with pytest.raises(TurnReceiptConflictError):
            receipts.get("other-session", "request-1", binding_digest)
        with pytest.raises(TurnReceiptConflictError):
            receipts.status("session", "request-1", "binding:other")
        with pytest.raises(TurnReceiptConflictError):
            receipts.claim(
                "session",
                "request-1",
                "binding:other",
                claim_token="claim-wrong-binding",
            )

        token, claimed = receipts.claim(
            "session", "request-1", binding_digest, claim_token="claim-1"
        )
        assert token == "claim-1"
        assert claimed is not None
        assert claimed["status"] == "CLAIMED"

        completed = receipts.finish(
            "session",
            "request-1",
            binding_digest,
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
        binding_digest = "binding:restart"
        receipts.prepare("session", "request-1", binding_digest)
        token, _ = receipts.claim(
            "session", "request-1", binding_digest, claim_token="claim-1"
        )
        completed = receipts.finish(
            "session",
            "request-1",
            binding_digest,
            token,
            assistant_content="terminal reply",
            response_digest="sha256:" + "b" * 64,
        )
        before_restart = json.dumps(completed, separators=(",", ":"))
    finally:
        db.close()

    reopened = SessionDB(tmp_path / "state.db")
    try:
        receipts = TurnReceiptAdapter(reopened)
        recovered = receipts.get("session", "request-1", binding_digest)
        assert recovered is not None
        assert json.dumps(recovered, separators=(",", ":")) == before_restart
        assert receipts.prepare("session", "request-1", binding_digest) == recovered
        with pytest.raises(TurnReceiptConflictError):
            receipts.claim(
                "session",
                "request-1",
                "binding:restart-other",
                claim_token="claim-replay",
            )
    finally:
        reopened.close()


def test_finish_is_idempotent_for_a_completed_receipt(tmp_path):
    db, receipts = _opened_adapter(tmp_path)
    try:
        binding_digest = "binding:completed"
        receipts.prepare("session", "request-1", binding_digest)
        token, _ = receipts.claim(
            "session", "request-1", binding_digest, claim_token="claim-1"
        )
        first = receipts.finish(
            "session",
            "request-1",
            binding_digest,
            token,
            assistant_content="terminal reply",
            response_digest="sha256:" + "c" * 64,
        )
        second = receipts.finish(
            "session",
            "request-1",
            binding_digest,
            token,
            assistant_content="terminal reply",
            response_digest="sha256:" + "c" * 64,
        )

        assert second == first
        assert db.message_count("session") == 1
        with pytest.raises(TurnReceiptConflictError):
            receipts.prepare("session", "request-1", "binding:completed-other")
    finally:
        db.close()


def test_completed_replay_finds_the_exact_soft_archived_terminal_row(tmp_path):
    db, receipts = _opened_adapter(tmp_path)
    try:
        request = ReceiptRequest("session", "archived-request", "binding:archived")
        receipts.prepare_or_replay(request)
        claimed = receipts.claim_after_lease(request)
        assistant_bytes = "transformed terminal � bytes 😀"
        response_digest = "sha256:" + hashlib.sha256(
            assistant_bytes.encode("utf-8")
        ).hexdigest()
        completed = receipts.finish(
            request.session_id,
            request.turn_request_id,
            request.binding_digest,
            claimed.claim_token,
            assistant_content=assistant_bytes,
            response_digest=response_digest,
        )
        db.replace_messages(
            "session", [], active_only=True, archive_dropped=True
        )
        db.append_message("session", "assistant", content="different active assistant")

        replay = receipts.completed_replay(request)

        assert db.get_messages("session")[0]["content"] == "different active assistant"
        assert replay == {
            **completed,
            "assistantContent": assistant_bytes,
        }
        assert replay["terminalMessageId"] != db.get_messages("session")[0]["id"]
        assert replay["responseDigest"] == response_digest
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

        receipts.prepare("session", "settled", "binding:settled")
        token, _ = receipts.claim(
            "session", "settled", "binding:settled", claim_token="claim-settled"
        )
        receipts.finish(
            "session",
            "settled",
            "binding:settled",
            token,
            assistant_content="old reply",
            response_digest="sha256:" + "d" * 64,
        )
        receipts.prepare("session", "pending", "binding:pending")
        pending_token, _ = receipts.claim(
            "session",
            "pending",
            "binding:pending",
            claim_token="claim-pending",
        )
        assert pending_token == "claim-pending"
        db._conn.execute(
            "UPDATE turn_receipts SET completed_at = 1 WHERE turn_request_id = 'settled'"
        )

        assert receipts.prune("session", completed_before=2) == 1
        assert receipts.get("session", "settled", "binding:settled") is None
        pending = receipts.get("session", "pending", "binding:pending")
        assert pending is not None
        assert pending["status"] == "CLAIMED"
    finally:
        db.close()


def test_fresh_receipt_schema_requires_a_binding_digest(tmp_path):
    db, _ = _opened_adapter(tmp_path)
    try:
        columns = {
            row["name"]: row
            for row in db._conn.execute("PRAGMA table_info(turn_receipts)").fetchall()
        }
        assert columns["binding_digest"]["notnull"] == 1
        with pytest.raises(sqlite3.IntegrityError):
            db._conn.execute(
                "INSERT INTO turn_receipts "
                "(turn_request_id, session_id, status, created_at) "
                "VALUES ('unbound', 'session', 'PREPARED', 1)"
            )
    finally:
        db.close()


def test_c3a_unbound_receipt_migration_preserves_but_fences_legacy_row(tmp_path):
    path = tmp_path / "state.db"
    initial = SessionDB(path)
    initial.create_session("session", source="test")
    initial.close()
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE turn_receipts")
        conn.execute(
            """CREATE TABLE turn_receipts (
                turn_request_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                status TEXT NOT NULL,
                claim_token TEXT,
                terminal_message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
                response_digest TEXT,
                created_at REAL NOT NULL,
                claimed_at REAL,
                completed_at REAL
            )"""
        )
        conn.execute(
            "INSERT INTO turn_receipts "
            "(turn_request_id, session_id, status, created_at) "
            "VALUES ('legacy-request', 'session', 'PREPARED', 7)"
        )

    db = SessionDB(path)
    receipts = TurnReceiptAdapter(db)
    try:
        migrated = db._conn.execute(
            "SELECT turn_request_id, session_id, status, created_at, binding_digest "
            "FROM turn_receipts WHERE turn_request_id = 'legacy-request'"
        ).fetchone()
        assert tuple(migrated) == (
            "legacy-request",
            "session",
            "PREPARED",
            7.0,
            None,
        )

        with pytest.raises(TurnReceiptFenceError):
            receipts.prepare("session", "legacy-request", "binding:legacy")
        with pytest.raises(TurnReceiptFenceError):
            receipts.get("session", "legacy-request", "binding:legacy")
        with pytest.raises(TurnReceiptFenceError):
            receipts.claim(
                "session",
                "legacy-request",
                "binding:legacy",
                claim_token="claim-legacy",
            )

        assert db._conn.execute(
            "SELECT binding_digest FROM turn_receipts "
            "WHERE turn_request_id = 'legacy-request'"
        ).fetchone()["binding_digest"] is None
    finally:
        db.close()
