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

import pytest

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


# ── Round 2: the transient defer must reach the CALLERS, on rotation too and through the forced preflight ──
# Callers see an unchanged transcript; only the transient signal tells them a store read failed once (defer)
# rather than that the session cannot shrink (exhaust / "/new").

_WINDOW = 272_000
_OVERSIZED_REQUEST = 300_000


def _over_window_agent(tmp_path: Path, sid: str, *, in_place: bool):
    db = SessionDB(db_path=tmp_path / "state.db")
    messages = _seed(db, sid)
    agent = _build_agent(db, sid, in_place=in_place, append_during_summary=not in_place)
    compressor = agent.context_compressor
    compressor.context_length = _WINDOW
    compressor.threshold_tokens = _WINDOW // 2
    compressor.should_compress.return_value = True
    compressor.get_active_compression_failure_cooldown.return_value = None
    return db, agent, messages


# In-place fails the start watermark (read 1); rotation reads it fine and fails the foreign-tail ceiling (read 2).
_WATERMARK_FAILURES = [
    pytest.param(True, 1, id="in_place_start_watermark"),
    pytest.param(False, 2, id="rotation_ceiling"),
]


def test_rotation_ceiling_read_failure_does_not_trip_the_incompressible_session_error(tmp_path: Path):
    """R1-3 (rotation sibling): the turn-start guard must see the rotation's ceiling-read rollback as a defer."""
    from agent.conversation_compression import compression_blocked_transiently
    from agent.turn_context import _fail_closed_on_insufficient_progress

    db, agent, messages = _over_window_agent(tmp_path, "CEILING_READ_FAILS_OVER_WINDOW", in_place=False)
    with patch.object(db, "get_active_message_watermark", side_effect=_fail_watermark_read(db, on_call=2)):
        out, _prompt = agent._compress_context(messages, "sys", approx_tokens=_OVERSIZED_REQUEST)

    assert out is messages  # rolled back to the parent, nothing committed
    assert compression_blocked_transiently(agent) is True
    _fail_closed_on_insufficient_progress(agent, _OVERSIZED_REQUEST)  # must not raise "/new"


@pytest.mark.parametrize("in_place,failing_read", _WATERMARK_FAILURES)
def test_overflow_recovery_defers_instead_of_exhausting_on_a_watermark_read_failure(
    tmp_path: Path, in_place, failing_read
):
    """R1-3: provider 413 / context-exceeded recovery must end the turn deferred, not compression_exhausted."""
    from agent.turn_overflow import _Recovery

    db, agent, messages = _over_window_agent(tmp_path, f"OVERFLOW_WM_FAIL_{failing_read}", in_place=in_place)
    rec = _Recovery(
        agent=agent, api_messages=list(messages), system_message="sys", effective_task_id=None,
        api_call_count=1, max_compression_attempts=3, messages=messages, active_system_prompt="sys",
        conversation_history=list(messages), approx_tokens=_OVERSIZED_REQUEST, compression_attempts=1,
    )
    with patch.object(db, "get_active_message_watermark",
                      side_effect=_fail_watermark_read(db, on_call=failing_read)):
        verdict = rec.compress(_OVERSIZED_REQUEST)

    assert verdict is not None and verdict.action == "return"
    assert verdict.result.get("compression_deferred") is True
    assert not verdict.result.get("compression_exhausted")
    assert verdict.compression_attempts == 0  # the attempt was refunded


@pytest.mark.parametrize("in_place,failing_read", _WATERMARK_FAILURES)
def test_forced_pre_api_preflight_defers_instead_of_exhausting_on_a_watermark_read_failure(
    tmp_path: Path, in_place, failing_read
):
    """R2-1: after a provider overflow the next pre-API preflight is forced; its transient branch refunds the
    attempt, so it must return the deferred result — not fall through to the forced-preflight exhaustion with
    zero attempts spent — and must still not send the oversized request."""
    from agent.turn_preflight import PreflightGateVerdict, run_preflight_compression

    db, agent, messages = _over_window_agent(tmp_path, f"FORCED_PREFLIGHT_WM_FAIL_{failing_read}", in_place=in_place)
    v = PreflightGateVerdict(
        action="fallthrough", pending_moa_prepared_request=None, messages=messages, active_system_prompt="sys",
        conversation_history=list(messages), api_call_count=1, compression_attempts=0, final_response=None,
        failed=False, _turn_exit_reason=None, _compression_timeout_exhausted=False,
        _preflight_compression_blocked=False, _provider_overflow_recovery_pending=True,
        _last_preflight_pressure=None,
    )
    reads = _fail_watermark_read(db, on_call=failing_read)
    with patch.object(db, "get_active_message_watermark", side_effect=reads) as watermark:
        v = run_preflight_compression(
            agent, v, compressor=agent.context_compressor, request_pressure_tokens=_OVERSIZED_REQUEST,
            provider_overflow_preflight=True, defer_preflight=lambda _t: False, moa_prepared_request=None,
            system_message="sys", user_message="question", max_compression_attempts=3, effective_task_id=None,
        )

    assert watermark.call_count == failing_read  # the pass really ran and hit the injected failure
    assert v.action == "return"  # the oversized request is not sent
    assert v.result.get("compression_deferred") is True
    assert not v.result.get("compression_exhausted")
    assert v.compression_attempts == 0
