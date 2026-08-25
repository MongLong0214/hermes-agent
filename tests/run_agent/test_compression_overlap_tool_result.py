"""Regression for a terminal tool-result flush overlapping compression."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.conversation_compression import compress_context
from agent.tool_executor import _flush_session_db_after_tool_progress
from hermes_state import SessionDB


SESSION_ID = "compression-overlap-tool-result"
SUMMARY_TEXT = "[CONTEXT COMPACTION] exact overlap summary"
TOOL_CALL_ID = "call_terminal_overlap"
TOOL_NAME = "terminal"


def _build_agent(db: SessionDB, session_id: str, hermes_home: Path):
    with (
        patch.dict(
            os.environ,
            {
                "HERMES_HOME": str(hermes_home),
                "OPENROUTER_API_KEY": "test-key",
            },
        ),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    agent._session_db_created = True
    agent._compression_feasibility_checked = True
    agent._cached_system_prompt = "system"
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._db_flush_scan_prefix = None
    agent._persist_disabled = False
    agent._incremental_persistence_failed = False
    agent._last_persistence_error_cause = None
    return agent


def test_terminal_tool_result_survives_compression_overlap(
    tmp_path: Path, monkeypatch
) -> None:
    """A terminal result appended during summary work remains canonical."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(SESSION_ID, source="test", model="test/model")
    seeded_messages = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"seed message {index}: " + "x" * 200,
        }
        for index in range(20)
    ]
    db.append_messages_batch(SESSION_ID, seeded_messages)
    compression_input = db.get_messages_as_conversation(SESSION_ID)

    hermes_home = tmp_path / "hermes-home"
    (hermes_home / "logs").mkdir(parents=True)
    compressor_agent = _build_agent(db, SESSION_ID, hermes_home)
    compressor_agent.compression_in_place = True
    flush_agent = _build_agent(db, SESSION_ID, hermes_home)

    lock_acquired = threading.Event()
    summary_entered = threading.Event()
    release_summary = threading.Event()
    flush_finished = threading.Event()
    original_try_acquire = db.try_acquire_compression_lock

    def observed_try_acquire(*args, **kwargs):
        acquired = original_try_acquire(*args, **kwargs)
        if acquired:
            lock_acquired.set()
        return acquired

    monkeypatch.setattr(db, "try_acquire_compression_lock", observed_try_acquire)
    monkeypatch.setattr(SessionDB, "_COMPRESSION_BUSY_WAIT_S", 0.05)

    summary_provider = MagicMock(name="event_blocked_aux_summary_provider")

    def blocked_summary(*_args, **_kwargs):
        summary_entered.set()
        if not release_summary.wait(timeout=5):
            raise AssertionError("test cleanup did not release the summary provider")
        return [
            {"role": "user", "content": SUMMARY_TEXT},
            {"role": "user", "content": "retained compression tail"},
        ]

    summary_provider.compress.side_effect = blocked_summary
    summary_provider.compression_count = 1
    summary_provider.last_prompt_tokens = 0
    summary_provider.last_completion_tokens = 0
    summary_provider._last_summary_error = None
    summary_provider._last_compress_aborted = False
    summary_provider._last_compression_made_progress = True
    summary_provider._last_summary_fallback_used = False
    summary_provider._last_aux_model_failure_model = None
    summary_provider._last_aux_model_failure_error = None
    summary_provider._proactive_prune_rearm_tokens = 0
    summary_provider.awaiting_real_usage_after_compression = False
    compressor_agent.context_compressor = summary_provider

    tool_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": TOOL_CALL_ID,
                    "type": "function",
                    "function": {
                        "name": TOOL_NAME,
                        "arguments": '{"command":"printf overlap"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "overlap terminal output",
            "tool_call_id": TOOL_CALL_ID,
            "tool_name": TOOL_NAME,
        },
    ]
    compression_outcome: dict[str, object] = {}
    flush_outcome: dict[str, object] = {}
    append_errors: list[str] = []
    original_append_batch = db.append_messages_batch

    def observed_append_batch(*args, **kwargs):
        try:
            return original_append_batch(*args, **kwargs)
        except Exception as exc:
            append_errors.append(type(exc).__name__)
            raise

    monkeypatch.setattr(db, "append_messages_batch", observed_append_batch)

    def run_compression() -> None:
        try:
            compression_outcome["value"] = compress_context(
                compressor_agent,
                compression_input,
                "system",
                approx_tokens=120_000,
                force=True,
            )
        except BaseException as exc:  # surfaced after bounded join
            compression_outcome["error"] = exc

    def run_flush() -> None:
        try:
            flush_outcome["persisted"] = _flush_session_db_after_tool_progress(
                flush_agent,
                tool_messages,
                stage="terminal result",
            )
            flush_outcome["incremental_persistence_failed"] = (
                flush_agent._incremental_persistence_failed
            )
            cause = flush_agent._last_persistence_error_cause
            flush_outcome["failure_reason"] = (
                f"session_persistence_failed:{cause or 'unknown'}"
                if flush_agent._incremental_persistence_failed
                else None
            )
        except BaseException as exc:  # surfaced after bounded join
            flush_outcome["error"] = exc
        finally:
            flush_finished.set()

    compression_thread = threading.Thread(
        target=run_compression, name="compression-overlap-compressor"
    )
    flush_thread = threading.Thread(
        target=run_flush, name="compression-overlap-tool-flush"
    )
    active_transcript: list[dict] = []
    try:
        compression_thread.start()
        assert lock_acquired.wait(timeout=3), "compression lock was not acquired"
        assert summary_entered.wait(timeout=3), "summary provider was not entered"

        flush_thread.start()
        try:
            assert flush_finished.wait(timeout=3), (
                "terminal tool-result flush did not finish while the summary "
                "provider remained blocked"
            )
            assert flush_outcome == {
                "persisted": True,
                "incremental_persistence_failed": False,
                "failure_reason": None,
            }, (
                "terminal tool-result flush failed before summary release: "
                f"append_errors={append_errors!r}, outcome={flush_outcome!r}; "
                "the historical failure is SessionCompressionInProgressError "
                "producing session_persistence_failed"
            )
        finally:
            release_summary.set()
            flush_thread.join(timeout=5)
            compression_thread.join(timeout=5)

        assert not flush_thread.is_alive(), "tool-result flush thread did not join"
        assert not compression_thread.is_alive(), "compressor thread did not join"
        assert "error" not in compression_outcome, compression_outcome.get("error")
        assert "error" not in flush_outcome, flush_outcome.get("error")

        active_transcript = db.get_messages_as_conversation(SESSION_ID)
    finally:
        release_summary.set()
        if flush_thread.is_alive():
            flush_thread.join(timeout=5)
        if compression_thread.is_alive():
            compression_thread.join(timeout=5)
        db.close()

    assert db._conn is None, "temporary SessionDB did not close cleanly"
    summary_index = next(
        index
        for index, message in enumerate(active_transcript)
        if message.get("content") == SUMMARY_TEXT
    )
    tool_result_index = next(
        index
        for index, message in enumerate(active_transcript)
        if message.get("role") == "tool"
        and message.get("content") == "overlap terminal output"
    )
    tool_result = active_transcript[tool_result_index]
    assert tool_result_index > summary_index
    assert tool_result["tool_call_id"] == TOOL_CALL_ID
    assert tool_result["tool_name"] == TOOL_NAME
