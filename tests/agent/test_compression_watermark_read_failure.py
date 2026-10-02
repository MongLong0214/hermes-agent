"""L2-2: a compression that cannot read its watermark must not commit.

Ordinary appends are allowed while a slow summary holds the compression lease; the start watermark (and,
on rotation, the foreign-tail ceiling) is what keeps those rows live at commit. When the read raised, the
code logged, carried on with ``watermark=None`` and committed: the in-place archive treated ``None`` as
"archive every active row" and the rotation skipped tail preservation, so a same-session row appended after
the snapshot left the live transcript without ever reaching the summary.

Real ``SessionDB``; only the watermark read is made to fail. The "concurrent" row is appended by another
writer after the transcript snapshot: at the moment the start-watermark read fails (in-place), or during
the summary step (rotation, whose ceiling read fails afterwards).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB

LATE_ROW = "LATE APPEND: also update the changelog"


def _build_agent(db: SessionDB, session_id: str, *, in_place: bool, append_during_summary: bool):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
            session_db=db, session_id=session_id, skip_context_files=True, skip_memory=True,
        )

    compressor = MagicMock()

    def _compress(*_a, **_kw):
        if append_during_summary:  # another writer on the same session appends while the summary is in flight
            db.append_message(session_id, "user", LATE_ROW)
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    compressor.compress.side_effect = _compress
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    agent._compression_feasibility_checked = True
    agent.compression_in_place = in_place
    return agent


def _seed(db: SessionDB, session_id: str):
    db.create_session(session_id, source="cli")
    for i in range(4):
        db.append_message(session_id, "user", f"question {i}")
        db.append_message(session_id, "assistant", f"answer {i}")
    return db.get_messages_as_conversation(session_id)


def _fail_watermark_read(db: SessionDB, *, on_call: int, append_first: bool = False):
    real = db.get_active_message_watermark
    calls = {"n": 0}

    def _read(session_id):
        calls["n"] += 1
        if calls["n"] == on_call:
            if append_first:  # a writer commits right after the snapshot, as this read fails
                db.append_message(session_id, "user", LATE_ROW)
            raise RuntimeError("database is locked")
        return real(session_id)
    return _read


def _live_contents(db: SessionDB, session_id: str):
    return [row.get("content") for row in db.get_messages_as_conversation(session_id)]


def test_in_place_commit_does_not_archive_a_row_appended_after_a_failed_watermark_read(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "WATERMARK_READ_FAILS_IN_PLACE"
    messages = _seed(db, sid)
    agent = _build_agent(db, sid, in_place=True, append_during_summary=False)

    with patch.object(db, "get_active_message_watermark",
                      side_effect=_fail_watermark_read(db, on_call=1, append_first=True)):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert LATE_ROW in _live_contents(db, agent.session_id)
    # The lease is released, so the next attempt can compress right away.
    assert db.get_compression_lock_holder(sid) is None


def test_rotation_does_not_drop_a_row_appended_after_a_failed_ceiling_read(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "WATERMARK_CEILING_READ_FAILS_ROTATION"
    messages = _seed(db, sid)
    agent = _build_agent(db, sid, in_place=False, append_during_summary=True)

    # Call 1 is the start watermark (succeeds); call 2 is the rotation's foreign-tail ceiling.
    with patch.object(db, "get_active_message_watermark", side_effect=_fail_watermark_read(db, on_call=2)):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert LATE_ROW in _live_contents(db, agent.session_id)
    assert db.get_compression_lock_holder(sid) is None


def test_watermark_unreadable_abort_publishes_a_transient_defer_signal(tmp_path: Path):
    """R1-3: the watermark-unreadable abort (fail-closed: sit out, commit nothing) must also publish the
    existing transient-defer signal. Without it, an unchanged transcript with no signal set is
    indistinguishable from a genuinely incompressible session to every caller that only checks the
    lock-skip signal (turn-start preflight, post-tool) or request_exceeds_model_window (the over-window
    guard) — each would treat a one-off, already-recovered read failure as proof compression is exhausted."""
    from agent.conversation_compression import compression_blocked_transiently

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "WATERMARK_READ_FAILS_TRANSIENT_SIGNAL"
    messages = _seed(db, sid)
    agent = _build_agent(db, sid, in_place=True, append_during_summary=False)

    with patch.object(db, "get_active_message_watermark", side_effect=_fail_watermark_read(db, on_call=1)):
        out_messages, _out_prompt = agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert out_messages is messages  # aborted: sat out, transcript unchanged
    assert compression_blocked_transiently(agent) is True


def test_watermark_unreadable_abort_does_not_trip_the_incompressible_session_error(tmp_path: Path):
    """R1-3 repro: a request over the model's context window, compressed on a session whose watermark read
    failed (then recovered), must not raise "Start a new session with /new; this session is too large to
    compress further" — the transient signal from the abort must reach this exact guard."""
    from agent.turn_context import _fail_closed_on_insufficient_progress

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "WATERMARK_READ_FAILS_OVER_WINDOW"
    messages = _seed(db, sid)
    agent = _build_agent(db, sid, in_place=True, append_during_summary=False)
    agent.context_compressor.context_length = 272_000

    with patch.object(db, "get_active_message_watermark", side_effect=_fail_watermark_read(db, on_call=1)):
        agent._compress_context(messages, "sys", approx_tokens=300_000)

    # Must not raise: a transient defer, not proof a 300k-token request can never fit a 272k window.
    _fail_closed_on_insufficient_progress(agent, 300_000)
