"""Tests for SessionDB.append_messages_batch (#23254 salvage).

The batch writer reuses _insert_message_rows (the same row-serialization
path as replace/compact/import), runs the same admission guards as
append_message, is atomic (all rows or none), and aggregates the session
counters in one UPDATE.
"""

import json
import sqlite3
from dataclasses import replace

import pytest

from hermes_state import (
    CompressionSessionClosedError,
    SessionDB,
    TerminalTurnReceipt,
    TurnReceiptConflictError,
    TurnReceiptFenceError,
)


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("sess-batch", source="cli")
    yield d
    d.close()


def _turn_messages():
    return [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "let me check",
            "tool_calls": [{"name": "terminal", "arguments": "{}"}],
            "reasoning_content": "thinking...",
            "finish_reason": "tool_calls",
        },
        {
            "role": "tool",
            "content": "tool output",
            "tool_name": "terminal",
            "tool_call_id": "call_1",
        },
        {"role": "assistant", "content": "answer", "finish_reason": "stop"},
    ]


def _claimed_terminal_receipt(
    db: SessionDB, request_id: str, *, terminal_message_index: int
) -> TerminalTurnReceipt:
    binding_digest = f"binding:{request_id}"
    claim_token = f"claim:{request_id}"
    db.prepare_turn_receipt("sess-batch", request_id, binding_digest)
    assert db.claim_turn_receipt(
        "sess-batch", request_id, binding_digest, claim_token
    )
    return TerminalTurnReceipt(
        session_id="sess-batch",
        turn_request_id=request_id,
        binding_digest=binding_digest,
        claim_token=claim_token,
        response_digest="sha256:" + "a" * 64,
        terminal_message_index=terminal_message_index,
    )


class TestAppendMessagesBatch:
    def test_terminal_receipt_binds_its_explicit_batch_row_not_the_tail(self, db):
        """A later assistant does not replace the explicitly held terminal row."""
        receipt = _claimed_terminal_receipt(
            db, "explicit-terminal-index", terminal_message_index=1
        )
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "held terminal"},
            {"role": "assistant", "content": "later assistant"},
        ]

        assert db.append_messages_batch(
            "sess-batch", messages, terminal_turn_receipt=receipt
        ) == 3

        completed = db.get_turn_receipt(
            "sess-batch", receipt.turn_request_id, receipt.binding_digest
        )
        assert completed["status"] == "COMPLETED"
        rows = db.get_messages("sess-batch")
        held_row = rows[1]
        assert held_row["content"] == "held terminal"
        assert completed["terminalMessageId"] == held_row["id"]
        assert completed["terminalMessageId"] != rows[2]["id"]

    def test_receipted_batch_rejects_invalid_mapping_before_writing(self, db):
        """Every invalid receipt mapping leaves its claimed receipt and rows intact."""
        cases = [
            (
                "out-of-range-index",
                [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "held terminal"},
                ],
                2,
                {},
                ValueError,
            ),
            (
                "moved-to-user-row",
                [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "held terminal"},
                ],
                0,
                {},
                ValueError,
            ),
            (
                "hidden-terminal-row",
                [
                    {"role": "user", "content": "question"},
                    {
                        "role": "assistant",
                        "content": "held terminal",
                        "display_kind": "hidden",
                    },
                ],
                1,
                {},
                ValueError,
            ),
            (
                "duplicate-held-object",
                (lambda held: [
                    {"role": "user", "content": "question"}, held, held
                ])({"role": "assistant", "content": "held terminal"}),
                1,
                {},
                ValueError,
            ),
            (
                "invalid-digest",
                [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "held terminal"},
                ],
                1,
                {"response_digest": "not-a-sha256-digest"},
                ValueError,
            ),
            (
                "wrong-claim-token",
                [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "held terminal"},
                ],
                1,
                {"claim_token": "wrong-claim-token"},
                TurnReceiptFenceError,
            ),
            (
                "wrong-binding",
                [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "held terminal"},
                ],
                1,
                {"binding_digest": "wrong-binding"},
                TurnReceiptConflictError,
            ),
        ]

        for name, messages, index, overrides, error_type in cases:
            receipt = _claimed_terminal_receipt(
                db, name, terminal_message_index=index
            )
            receipt = replace(receipt, **overrides)

            with pytest.raises(error_type):
                db.append_messages_batch(
                    "sess-batch", messages, terminal_turn_receipt=receipt
                )

            assert db.get_messages("sess-batch") == [], name
            actual = db.get_turn_receipt(
                "sess-batch", name, f"binding:{name}"
            )
            assert actual["status"] == "CLAIMED", name
            assert all("_row_id" not in message for message in messages), name

    def test_batch_rows_identical_to_single_appends(self, db, tmp_path):
        """The batch writer stores the same bytes append_message would."""
        db2 = SessionDB(db_path=tmp_path / "state2.db")
        db2.create_session("sess-batch", source="cli")
        try:
            msgs = _turn_messages()
            db.append_messages_batch("sess-batch", msgs)
            for m in msgs:
                role = m["role"]
                db2.append_message(
                    session_id="sess-batch",
                    role=role,
                    content=m.get("content"),
                    tool_name=m.get("tool_name"),
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                    finish_reason=m.get("finish_reason"),
                    reasoning_content=(
                        m.get("reasoning_content") if role == "assistant" else None
                    ),
                )
            cols = (
                "role, content, tool_call_id, tool_calls, tool_name, "
                "finish_reason, reasoning_content, observed, active"
            )
            rows_a = db._conn.execute(
                f"SELECT {cols} FROM messages ORDER BY id"
            ).fetchall()
            rows_b = db2._conn.execute(
                f"SELECT {cols} FROM messages ORDER BY id"
            ).fetchall()
            assert [tuple(r) for r in rows_a] == [tuple(r) for r in rows_b]
        finally:
            db2.close()

    def test_reasoning_gated_to_assistant_rows(self, db):
        """_insert_message_rows role-gates reasoning fields; a tool row
        carrying reasoning keys must not persist them."""
        db.append_messages_batch(
            "sess-batch",
            [
                {
                    "role": "tool",
                    "content": "out",
                    "tool_name": "t",
                    "tool_call_id": "c1",
                    "reasoning_content": "should not persist",
                }
            ],
        )
        row = db._conn.execute(
            "SELECT reasoning_content FROM messages"
        ).fetchone()
        assert row[0] is None

    def test_counters_aggregate_once(self, db):
        db.append_messages_batch("sess-batch", _turn_messages())
        row = db._conn.execute(
            "SELECT message_count, tool_call_count FROM sessions WHERE id = ?",
            ("sess-batch",),
        ).fetchone()
        assert row["message_count"] == 4
        assert row["tool_call_count"] == 1

    def test_returns_inserted_count(self, db):
        assert db.append_messages_batch("sess-batch", _turn_messages()) == 4

    def test_empty_batch_is_noop(self, db):
        assert db.append_messages_batch("sess-batch", []) == 0
        row = db._conn.execute(
            "SELECT message_count FROM sessions WHERE id = ?", ("sess-batch",)
        ).fetchone()
        assert row["message_count"] == 0

    def test_atomicity_all_or_nothing(self, db, monkeypatch):
        """A failure mid-batch leaves ZERO rows and untouched counters."""
        real_insert = SessionDB._insert_message_rows

        def failing_insert(self_db, conn, session_id, messages):
            real_conn_execute = conn.execute
            calls = {"n": 0}

            def exec_counting(sql, *args):
                if sql.lstrip().startswith("INSERT INTO messages"):
                    calls["n"] += 1
                    if calls["n"] == 3:
                        raise sqlite3.OperationalError("boom mid-batch")
                return real_conn_execute(sql, *args)

            conn.execute = exec_counting
            try:
                return real_insert(self_db, conn, session_id, messages)
            finally:
                conn.execute = real_conn_execute

        monkeypatch.setattr(SessionDB, "_insert_message_rows", failing_insert)
        with pytest.raises(sqlite3.OperationalError):
            db.append_messages_batch("sess-batch", _turn_messages())
        monkeypatch.undo()

        count = db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert count == 0
        row = db._conn.execute(
            "SELECT message_count, tool_call_count FROM sessions WHERE id = ?",
            ("sess-batch",),
        ).fetchone()
        assert row["message_count"] == 0
        assert row["tool_call_count"] == 0

    def test_compression_closed_session_rejected(self, db):
        db._conn.execute(
            "UPDATE sessions SET ended_at = 1.0, end_reason = 'compression' "
            "WHERE id = ?",
            ("sess-batch",),
        )
        db._conn.commit()
        with pytest.raises(CompressionSessionClosedError):
            db.append_messages_batch("sess-batch", _turn_messages())

    def test_multimodal_content_encoded(self, db):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:x"}},
                ],
            }
        ]
        db.append_messages_batch("sess-batch", msgs)
        raw = db._conn.execute("SELECT content FROM messages").fetchone()[0]
        # encoded via _encode_content — same sentinel prefix as append_message
        loaded = db.get_messages("sess-batch")
        assert loaded, raw

    def test_tool_calls_json_string_not_double_encoded(self, db):
        msgs = [
            {
                "role": "assistant",
                "content": "x",
                "tool_calls": json.dumps([{"name": "t", "arguments": "{}"}]),
            }
        ]
        db.append_messages_batch("sess-batch", msgs)
        raw = db._conn.execute("SELECT tool_calls FROM messages").fetchone()[0]
        assert json.loads(raw) == [{"name": "t", "arguments": "{}"}]
