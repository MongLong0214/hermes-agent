"""A failed compression-lock DELETE must be retried, not treated as a completed release.

Cleanup marks the lease released and stops its refresher before the holder-qualified DELETE. The DB helper
logged and swallowed ``sqlite3.Error``, so one transient failure left a live 300 s lease nobody refreshed or
deleted: every later attempt reported "another path is compressing" until the TTL expired.

Real ``SessionDB``; DELETEs on ``compression_locks`` fail at the SQL write layer.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch):
    lock = _lock_module()
    if lock is not None:
        monkeypatch.setattr(lock, "_LOCK_RELEASE_RETRY_DELAY_S", 0.0)


def _lock_module():
    """The lock-release module, or None on a tree without it (there a release is a single attempt)."""
    try:
        import agent.conversation_compression_lock as lock
    except ImportError:
        return None
    return lock


def _build_agent(db: SessionDB, session_id: str):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
            session_db=db, session_id=session_id, skip_context_files=True, skip_memory=True,
        )
    compressor = MagicMock()
    compressor.compress.side_effect = lambda *_a, **_kw: [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    agent._compression_feasibility_checked = True
    agent.compression_in_place = True
    return agent


def _seed_session(db: SessionDB, sid: str):
    db.create_session(sid, source="cli")
    for i in range(4):
        db.append_message(sid, "user", f"question {i}")
        db.append_message(sid, "assistant", f"answer {i}")
    return db.get_messages_as_conversation(sid)


def _fail_n_lock_deletes(db: SessionDB, n: int):
    """Fail the first ``n`` DELETEs on ``compression_locks`` at the SQL write layer."""
    real = db._write_sql
    failed = {"n": 0}

    def _write_sql(sql, params=(), **kwargs):
        if sql.lstrip().upper().startswith("DELETE FROM COMPRESSION_LOCKS") and failed["n"] < n:
            failed["n"] += 1
            raise sqlite3.OperationalError("disk I/O error")
        return real(sql, params, **kwargs)
    return _write_sql, failed


def _release_batch() -> int:
    """One cleanup call's DELETE retry batch (one attempt where releases are not retried)."""
    lock = _lock_module()
    return 1 if lock is None else lock._LOCK_RELEASE_ATTEMPTS


def _assert_next_compression_runs_and_releases(agent, db: SessionDB, sid: str):
    agent.context_compressor.compress.reset_mock()
    agent._compress_context(db.get_messages_as_conversation(sid), "sys", approx_tokens=120_000)
    assert not agent._compression_skipped_due_to_lock  # not "another path is compressing" against itself
    assert agent.context_compressor.compress.call_count == 1
    assert db.get_compression_lock_holder(sid) is None


# ── counterexample 1: one failed release write is retried and the row is gone ─────────────────────────────


def test_failed_lock_delete_is_retried_so_the_session_can_compress_again_without_waiting_for_ttl(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "LOCK_RELEASE_DELETE_FAILS"
    messages = _seed_session(db, sid)
    agent = _build_agent(db, sid)

    write_sql, failed = _fail_n_lock_deletes(db, 1)
    with patch.object(db, "_write_sql", side_effect=write_sql):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert failed["n"] == 1  # the injected DELETE failure actually happened
    assert agent.context_compressor.compress.call_count == 1  # the compression itself ran
    assert db.get_compression_lock_holder(sid) is None
    # A different path can take the lock right away instead of sitting out until the 300 s TTL.
    assert db.try_acquire_compression_lock(sid, "next-attempt", ttl_seconds=300) is True


def test_release_compression_lock_reports_whether_the_delete_committed(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("S", source="cli")
    assert db.try_acquire_compression_lock("S", "holder-a", ttl_seconds=300) is True

    write_sql, _failed = _fail_n_lock_deletes(db, 1)
    with patch.object(db, "_write_sql", side_effect=write_sql):
        assert db.release_compression_lock("S", "holder-a") is False
        assert db.get_compression_lock_holder("S") == "holder-a"
        assert db.release_compression_lock("S", "holder-a") is True
    assert db.get_compression_lock_holder("S") is None
    # Idempotent: a holder whose row is already gone still reports a completed release.
    assert db.release_compression_lock("S", "holder-a") is True
    assert db.release_compression_lock("", "holder-a") is True


# ── counterexample 2: persistent failure parks the lock; the next attempt releases it ─────────────────────


def test_release_holder_only_retries_after_an_exhausted_batch_instead_of_reporting_done(tmp_path: Path):
    """``release_holder_only()`` must not mark the lease released (short-circuiting every later call with no
    DELETE at all) until the release actually committed. The first cleanup call exhausts the whole retry
    batch; a second cleanup call on the same lease, after the database recovered, must retry the DELETE."""
    from types import SimpleNamespace

    from agent.conversation_compression import _CompactionLifecycle, _CompressionLease

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "LOCK_RELEASE_EXHAUSTED_BATCH"
    db.create_session(sid, source="cli")
    assert db.try_acquire_compression_lock(sid, "holder-a", ttl_seconds=300) is True

    lifecycle = _CompactionLifecycle(SimpleNamespace(), status_emitted=False)
    lease = _CompressionLease(
        SimpleNamespace(), db=db, sid=sid, ttl=300.0, refresh_interval=None, commit_fence=None,
        lifecycle=lifecycle,
    )
    lease.holder = "holder-a"

    write_sql, failed = _fail_n_lock_deletes(db, _release_batch())
    with patch.object(db, "_write_sql", side_effect=write_sql):
        first = lease.release_holder_only()
    assert first is False
    assert failed["n"] == _release_batch()  # the whole retry batch was exhausted, not short-circuited
    assert db.get_compression_lock_holder(sid) == "holder-a"  # row survives a fully-failed cleanup
    assert db.try_acquire_compression_lock(sid, "holder-b", ttl_seconds=300) is False

    second = lease.release_holder_only()
    assert second is True
    assert db.get_compression_lock_holder(sid) is None
    assert db.try_acquire_compression_lock(sid, "holder-b", ttl_seconds=300) is True


def test_commit_with_an_exhausted_release_batch_is_released_by_the_next_compression(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "COMMIT_RELEASE_EXHAUSTED"
    messages = _seed_session(db, sid)
    agent = _build_agent(db, sid)

    write_sql, failed = _fail_n_lock_deletes(db, _release_batch())
    with patch.object(db, "_write_sql", side_effect=write_sql):
        agent._compress_context(messages, "sys", approx_tokens=120_000)
    assert failed["n"] == _release_batch()
    assert db.get_compression_lock_holder(sid) is not None  # the failed batch left this agent's row behind

    # Database recovered: the agent's next compression must not lose to its own abandoned holder.
    _assert_next_compression_runs_and_releases(agent, db, sid)


def test_watermark_abort_with_an_exhausted_release_batch_is_retried_by_the_next_compression(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "WATERMARK_ABORT_RELEASE_EXHAUSTED"
    messages = _seed_session(db, sid)
    agent = _build_agent(db, sid)

    def _fail_watermark(_sid):
        raise RuntimeError("database is locked")

    write_sql, failed = _fail_n_lock_deletes(db, _release_batch())
    with patch.object(db, "_write_sql", side_effect=write_sql), \
            patch.object(db, "get_active_message_watermark", side_effect=_fail_watermark):
        out, _prompt = agent._compress_context(messages, "sys", approx_tokens=120_000)
    assert out is messages  # sat out
    assert failed["n"] == _release_batch()
    assert db.get_compression_lock_holder(sid) is not None

    _assert_next_compression_runs_and_releases(agent, db, sid)


def test_acquire_exception_with_an_exhausted_release_batch_is_retried_by_the_next_compression(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "ACQUIRE_RAISES_RELEASE_EXHAUSTED"
    messages = _seed_session(db, sid)
    agent = _build_agent(db, sid)
    real_acquire = db.try_acquire_compression_lock

    def _acquire_then_raise(session_id, holder, ttl_seconds=300.0):
        real_acquire(session_id, holder, ttl_seconds=ttl_seconds)  # the row committed ...
        raise RuntimeError("connection reset after commit")  # ... but the caller only sees the raise

    write_sql, failed = _fail_n_lock_deletes(db, _release_batch())
    with patch.object(db, "_write_sql", side_effect=write_sql), \
            patch.object(db, "try_acquire_compression_lock", side_effect=_acquire_then_raise):
        out, _prompt = agent._compress_context(messages, "sys", approx_tokens=120_000)
    assert out is messages
    assert failed["n"] == _release_batch()
    assert db.get_compression_lock_holder(sid) is not None

    _assert_next_compression_runs_and_releases(agent, db, sid)


def test_cancelled_during_lock_setup_with_failing_releases_is_retried_by_the_next_compression(tmp_path: Path):
    """The host's timeout requests the cancelled-lock release while the worker is still acquiring; the hook
    publication runs it synchronously and the abort runs ``lease.release()`` again — both batches fail."""
    from agent.conversation_compression import (
        CompressionCommitFence, _acquire_compression_lease, _Attempt, _CompactionLifecycle,
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "CANCELLED_DURING_SETUP_RELEASE_EXHAUSTED"
    _seed_session(db, sid)
    agent = _build_agent(db, sid)
    fence = CompressionCommitFence()
    real_acquire = db.try_acquire_compression_lock

    def _acquire_then_host_times_out(session_id, holder, ttl_seconds=300.0):
        ok = real_acquire(session_id, holder, ttl_seconds=ttl_seconds)
        fence.release_cancelled_compression_lock()  # host timeout fires mid-setup: no hook yet, request kept
        return ok

    write_sql, failed = _fail_n_lock_deletes(db, 2 * _release_batch())
    with patch.object(db, "_write_sql", side_effect=write_sql), \
            patch.object(db, "try_acquire_compression_lock", side_effect=_acquire_then_host_times_out):
        lease, _prompt = _acquire_compression_lease(
            agent, commit_fence=fence, lifecycle=_CompactionLifecycle(agent, status_emitted=False),
            system_message="sys", approx_tokens=120_000,
            attempt=_Attempt(snapshot={}, generation=0, started_at=0.0, seed={}),
        )
    assert lease is None  # aborted before any summary work
    assert failed["n"] == 2 * _release_batch()
    assert db.get_compression_lock_holder(sid) is not None

    _assert_next_compression_runs_and_releases(agent, db, sid)
