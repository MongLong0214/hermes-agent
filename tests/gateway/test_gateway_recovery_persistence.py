"""Gateway-side recovery writes in ``_hmwa_persist_turn_transcript`` (R67-2).

The gateway falls back to its own transcript write whenever the agent reports
``agent_persisted=False`` (a flush failed or the runtime doesn't self-persist). The round-3
production review found that fallback still inserted through a raw single-row append:

- it never stamped the durable marker back onto the live message objects, so the agent's own
  next flush saw them as un-persisted and duplicated them;
- it never applied ``_row_id`` repair, so a message that already had a blank placeholder row
  (e.g. from an earlier partial flush) got a brand-new row instead of filling the existing one,
  orphaning the blank;
- the failed-turn branch only deduped on the platform ``message_id``, so a retried input with no
  platform id (internal events, some adapters) could duplicate the user row even though the
  gateway already has an ownership-based dedup check for exactly this case
  (``has_input_owner``, used by ``_hmwa_agent_error_reply``).

Each test below reproduces one of the three scenarios against a real, file-backed SessionDB.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent.context_compressor import _DB_PERSISTED_MARKER
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from run_agent import AIAgent


async def _noop_refresh(self, session_key, session_id):
    return None


def _make_runner(tmp_path: Path):
    from gateway.session import AsyncSessionStore

    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._refresh_agent_cache_message_count = _noop_refresh.__get__(runner, GatewayRunner)
    runner._session_db = store._db
    return runner, store


def _flush_agent(db, session_id, messages):
    """Minimal AIAgent stand-in bound to the REAL flush methods (same shape as
    tests/agent/test_session_persistence_flush_failure.py), used to simulate "the agent's own
    next flush" over messages the gateway recovery path already touched."""
    from types import SimpleNamespace

    agent = SimpleNamespace(
        _session_db=db,
        _session_db_created=True,
        _persist_disabled=False,
        session_id=session_id,
        _session_persist_lock=None,
        _flushed_db_message_ids=set(),
        _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0,
        _db_flush_scan_prefix=None,
        _persist_user_message_idx=None,
        _persist_user_message_override=None,
        _persist_user_message_timestamp=None,
        _pending_cli_user_message=None,
        _active_session_turn_lease_holder=None,
        _last_persistence_error_cause=None,
        _compression_adoption_failed=False,
    )
    agent._ensure_db_session = lambda: None
    agent._flush_messages_to_session_db = (
        AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
    )
    agent._flush_messages_to_session_db_unlocked = (
        AIAgent._flush_messages_to_session_db_unlocked.__get__(agent, AIAgent)
    )
    return agent._flush_messages_to_session_db(messages, None)


def test_recovered_rows_do_not_duplicate_on_the_agents_next_flush(tmp_path: Path):
    """Scenario 1: fallback writes [question, answer]; the agent's own subsequent flush over the
    SAME live message objects must not re-insert them as [question, answer, answer]."""
    runner, store = _make_runner(tmp_path)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="r67-2-dup", user_id="u")
    entry = store.get_or_create_session(source)
    sid = entry.session_id
    db = store._db_for_session_id(sid)

    user_msg = {"role": "user", "content": "question"}
    assistant_msg = {"role": "assistant", "content": "answer"}
    agent_messages = [user_msg, assistant_msg]
    prepared = runner._PreparedTurn(
        [{"role": "system", "content": "sys"}], "", "question", "question", 1700000000.0, None, sid, None,
    )
    agent_result = {"agent_persisted": False, "history_offset": 0, "tools": []}

    async def run():
        await runner._hmwa_persist_turn_transcript(
            event=MessageEvent(text="question", source=source, message_id=None),
            source=source, session_entry=entry, session_key=entry.session_key,
            agent_result=agent_result, agent_messages=agent_messages, prepared=prepared,
            response="answer", agent_failed_early=False, hidden_reasoning_incomplete=False,
            is_context_overflow_failure=False,
        )

    asyncio.run(run())

    rows_after_fallback = db.get_messages(sid)
    assert [r["content"] for r in rows_after_fallback] == ["question", "answer"], rows_after_fallback

    # The fix must stamp both live message dicts durable so a later flush recognizes them.
    assert user_msg.get(_DB_PERSISTED_MARKER) is True, user_msg
    assert assistant_msg.get(_DB_PERSISTED_MARKER) is True, assistant_msg

    # Simulate "the agent's own subsequent flush" over the identical live objects.
    flushed = _flush_agent(db, sid, agent_messages)
    assert flushed is True
    rows_after_agent_flush = db.get_messages(sid)
    assert [r["content"] for r in rows_after_agent_flush] == ["question", "answer"], rows_after_agent_flush


def test_recovered_assistant_row_fills_an_existing_blank_placeholder(tmp_path: Path):
    """Scenario 2: an assistant row was already inserted blank (e.g. a streaming placeholder) and
    carries ``_row_id``. Recovery must fill THAT row, not leave it blank and insert a second one."""
    runner, store = _make_runner(tmp_path)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="r67-2-blank", user_id="u")
    entry = store.get_or_create_session(source)
    sid = entry.session_id
    db = store._db_for_session_id(sid)

    # Turn-start already persisted the user row; a mid-turn streaming placeholder then inserted a
    # blank assistant row, matching the review's observed order (question, blank assistant).
    db.append_message(sid, "user", content="question")
    blank_row_id = db.append_message(sid, "assistant", content="")

    user_msg = {"role": "user", "content": "question", _DB_PERSISTED_MARKER: True}
    assistant_msg = {"role": "assistant", "content": "the real answer", "_row_id": blank_row_id}
    agent_messages = [user_msg, assistant_msg]
    prepared = runner._PreparedTurn(
        [{"role": "system", "content": "sys"}], "", "question", "question", 1700000000.0, None, sid, None,
    )
    agent_result = {"agent_persisted": False, "history_offset": 0, "tools": []}

    async def run():
        await runner._hmwa_persist_turn_transcript(
            event=MessageEvent(text="question", source=source, message_id=None),
            source=source, session_entry=entry, session_key=entry.session_key,
            agent_result=agent_result, agent_messages=agent_messages, prepared=prepared,
            response="the real answer", agent_failed_early=False, hidden_reasoning_incomplete=False,
            is_context_overflow_failure=False,
        )

    asyncio.run(run())

    rows = db.get_messages(sid)
    # Exactly one user row and one assistant row — the blank placeholder was filled in place,
    # not left orphaned beside a freshly inserted third row.
    assert [r["content"] for r in rows] == ["question", "the real answer"], rows
    assert rows[-1]["id"] == blank_row_id, rows


def test_failed_turn_without_platform_message_id_dedupes_on_input_owner(tmp_path: Path):
    """Scenario 3: a failed turn with no platform message_id must still dedupe against an
    already-persisted copy of the SAME accepted input, via the ownership marker the gateway
    already uses for this exact purpose on the exception path (has_input_owner)."""
    runner, store = _make_runner(tmp_path)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="r67-2-owner", user_id="u")
    entry = store.get_or_create_session(source)
    sid = entry.session_id
    db = store._db_for_session_id(sid)

    owner = "accepted-owner-1"
    db.append_message(sid, "user", "question", display_metadata={"gateway_input_owner": owner})
    before = db.message_count()

    prepared = runner._PreparedTurn(
        [{"role": "system", "content": "sys"}], "", "question", "question", 1700000000.0, None, sid, owner,
    )
    agent_result = {"agent_persisted": False, "failed": True}

    async def run():
        await runner._hmwa_persist_turn_transcript(
            event=MessageEvent(text="question", source=source, message_id=None),
            source=source, session_entry=entry, session_key=entry.session_key,
            agent_result=agent_result, agent_messages=[], prepared=prepared,
            response=None, agent_failed_early=True, hidden_reasoning_incomplete=False,
            is_context_overflow_failure=False,
        )

    asyncio.run(run())

    # No duplicate "question" user row — only the failed-turn boundary (if any) may have landed.
    rows = db.get_messages(sid)
    assert [r["content"] for r in rows if r["role"] == "user"] == ["question"], rows
    assert db.message_count() <= before + 1, rows
