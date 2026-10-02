"""L2-4: a failed compression-lock DELETE must be retried, not treated as a completed release.

Cleanup marks the lease released and stops its refresher before the holder-qualified DELETE. The DB helper
logged and swallowed ``sqlite3.Error``, so one transient failure left a live 300 s lease nobody refreshed or
deleted: every later attempt reported "another path is compressing" until the TTL expired.

Real ``SessionDB``; the first DELETE on ``compression_locks`` fails at the SQL write layer.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


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


def _fail_first_lock_delete(db: SessionDB):
    real = db._write_sql
    failed = {"n": 0}

    def _write_sql(sql, params=(), **kwargs):
        if sql.lstrip().upper().startswith("DELETE FROM COMPRESSION_LOCKS") and not failed["n"]:
            failed["n"] += 1
            raise sqlite3.OperationalError("disk I/O error")
        return real(sql, params, **kwargs)
    return _write_sql, failed


def test_failed_lock_delete_is_retried_so_the_session_can_compress_again_without_waiting_for_ttl(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "LOCK_RELEASE_DELETE_FAILS"
    db.create_session(sid, source="cli")
    for i in range(4):
        db.append_message(sid, "user", f"question {i}")
        db.append_message(sid, "assistant", f"answer {i}")
    agent = _build_agent(db, sid)

    write_sql, failed = _fail_first_lock_delete(db)
    with patch.object(db, "_write_sql", side_effect=write_sql):
        agent._compress_context(db.get_messages_as_conversation(sid), "sys", approx_tokens=120_000)

    assert failed["n"] == 1  # the injected DELETE failure actually happened
    assert db.get_compression_lock_holder(sid) is None
    # A different path can take the lock right away instead of sitting out until the 300 s TTL.
    assert db.try_acquire_compression_lock(sid, "next-attempt", ttl_seconds=300) is True


def test_release_compression_lock_reports_whether_the_delete_committed(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("S", source="cli")
    assert db.try_acquire_compression_lock("S", "holder-a", ttl_seconds=300) is True

    write_sql, _failed = _fail_first_lock_delete(db)
    with patch.object(db, "_write_sql", side_effect=write_sql):
        assert db.release_compression_lock("S", "holder-a") is False
        assert db.get_compression_lock_holder("S") == "holder-a"
        assert db.release_compression_lock("S", "holder-a") is True
    assert db.get_compression_lock_holder("S") is None
