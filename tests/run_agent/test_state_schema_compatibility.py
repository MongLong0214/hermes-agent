"""Turn-entry safety contracts for state-store schema compatibility."""

import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import hermes_state
from hermes_state import SCHEMA_VERSION, SessionDB
from run_agent import AIAgent


def _make_agent(session_db: SessionDB, session_id: str) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=session_db,
            session_id=session_id,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._session_db_created = True
    return agent


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
                finish_reason="stop",
            )
        ],
        model="test/model",
        usage=None,
    )


def test_turn_refuses_future_schema_before_model_tool_or_transcript_work(
    tmp_path, monkeypatch
):
    """An already-open store detects a sibling upgrade at the next turn."""
    db_path = tmp_path / "state.db"
    session_id = "schema-guard-session"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session(session_id, source="cli")
        before_rows = tuple(
            db._conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        )
        before_changes = db._conn.total_changes

        with sqlite3.connect(db_path) as sibling:
            sibling.execute(
                "UPDATE schema_version SET version = ?", (SCHEMA_VERSION + 1,)
            )

        agent = _make_agent(db, session_id)
        agent.client.chat.completions.create.return_value = _response("must not run")
        agent._persist_session = MagicMock()
        agent._execute_tool_calls = MagicMock()

        def unexpected_connection_or_marker(*_args, **_kwargs):
            raise AssertionError("turn schema guard must use the injected SessionDB")

        # The prologue may only query its existing SessionDB connection. It
        # must neither open a raw connection nor mint a turn-fence marker.
        monkeypatch.setattr(sqlite3, "connect", unexpected_connection_or_marker)
        monkeypatch.setattr(
            hermes_state,
            "register_turn_fence_function",
            unexpected_connection_or_marker,
        )

        history = [{"role": "assistant", "content": "prior reply"}]
        result = agent.run_conversation("new request", conversation_history=history)

        assert result == {
            "final_response": (
                "⚠️ State database is newer than this running Hermes process. "
                "Restart the process, then send your message again."
            ),
            "messages": history,
            "api_calls": 0,
            "completed": False,
            "failed": True,
            "error": "state_store_schema_incompatible",
            "failure_reason": "state_store_schema_incompatible",
            "turn_exit_reason": "state_store_schema_incompatible",
            "restart_required": True,
        }
        agent.client.chat.completions.create.assert_not_called()
        agent._execute_tool_calls.assert_not_called()
        agent._persist_session.assert_not_called()
        assert db._conn.total_changes == before_changes
        assert tuple(
            db._conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        ) == before_rows
    finally:
        db.close()


def test_current_schema_allows_normal_turn_and_transcript_write(tmp_path):
    """The compatibility prologue is transparent for a current store."""
    db_path = tmp_path / "state.db"
    session_id = "current-schema-session"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session(session_id, source="cli")
        agent = _make_agent(db, session_id)
        agent.client.chat.completions.create.return_value = _response("all clear")

        result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["final_response"] == "all clear"
        assert result["api_calls"] == 1
        assert [message["role"] for message in db.get_messages_as_conversation(session_id)] == [
            "user",
            "assistant",
        ]
    finally:
        db.close()
