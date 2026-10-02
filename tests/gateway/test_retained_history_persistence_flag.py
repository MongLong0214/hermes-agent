"""L4-2/R67-5 — a retained live transcript substituted for the reloaded persisted history
(``gateway/run.py::_select_cached_agent_history``, the FTS write-corruption guard, #50502) must not
let ``agent/session_persistence.py``'s "already in conversation_history => durable" shortcut
falsely certify its unflushed rows. Before this fix, the reviewer's probe showed the recovery
turn's flush wrote only the NEW message and skipped the retained backlog entirely, while
``_last_flushed_db_idx`` still advanced past it — ``transcript_persistence_caught_up`` then
reported the session durable and eviction became eligible, permanently losing the backlog.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from gateway.run import _select_cached_agent_history
from hermes_state import SessionDB
from run_agent import AIAgent


def test_recovery_turn_writes_the_retained_backlog_not_just_the_new_message():
    tmp = tempfile.mkdtemp(prefix="retained_history_")
    db = None
    try:
        db = SessionDB(Path(tmp) / "state.db")
        sid = "sess-retained"
        db.create_session(session_id=sid, source="telegram")

        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            session_db=db, session_id=sid,
        )
        agent._session_db_created = True

        # A prior turn's flush never reached state.db: both rows are still unflushed in memory.
        user1 = {"role": "user", "content": "first question"}
        assistant1 = {"role": "assistant", "content": "first answer"}
        agent._session_messages = [user1, assistant1]

        # The FTS write-corruption guard substitutes this retained live transcript for the
        # persisted reload (empty here, since nothing ever reached disk).
        selected = _select_cached_agent_history([], agent._session_messages)
        assert selected == [user1, assistant1]
        assert selected[0].get("_retained_unflushed_history") is True
        assert selected[1].get("_retained_unflushed_history") is True

        # The recovery turn appends its own new user message and flushes with `selected` as
        # conversation_history — exactly gateway/run_turn_runner.py::_run_conversation_with_approval's
        # shape (``conversation_history=agent_history``, the value this function returned).
        user2 = {"role": "user", "content": "second question"}
        messages = selected + [user2]
        agent._session_messages = messages

        assert agent._flush_messages_to_session_db(messages, conversation_history=selected) is True

        rows = db.get_messages(sid, include_inactive=True)
        contents = [r["content"] for r in rows]
        assert "first question" in contents, contents
        assert "first answer" in contents, contents
        assert "second question" in contents, contents
        assert len(contents) == 3, (
            "the retained backlog must be written exactly once, not skipped as already-durable"
        )

        from gateway.agent_cache_pressure import transcript_persistence_caught_up
        assert transcript_persistence_caught_up(agent) is True
    finally:
        if db is not None:
            db.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_select_cached_history_marks_only_the_genuinely_unpersisted_rows():
    """Guardrail: a row the agent already flushed (``_db_persisted``) must not be re-marked or
    re-written — only the rows the finding is actually about."""
    persisted = [{"role": "user", "content": "hello"}]
    live = [
        {"role": "user", "content": "hello", "_db_persisted": True},
        {"role": "assistant", "content": "not written"},
    ]

    selected = _select_cached_agent_history(persisted, live)

    assert selected[0].get("_retained_unflushed_history") is None
    assert selected[1].get("_retained_unflushed_history") is True
