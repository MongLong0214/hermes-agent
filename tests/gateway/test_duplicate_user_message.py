"""Tests for #42039 — user messages stored twice in state.db.

When the agent has its own SessionDB reference (``_session_db is not None``),
``_flush_messages_to_session_db()`` persists messages to SQLite during the
agent run.  The gateway's ``append_to_transcript()`` must then use
``skip_db=True`` on all fallback paths to prevent writing a second copy
to the same SQLite file.

This test covers the two fallback paths that previously lacked
``skip_db=agent_persisted``:

1. ``agent_failed_early`` path — transient 429/timeout failures
2. ``not new_messages`` path — edge case where ``history_offset`` exceeds
   the actual message count
"""

import sys
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.turn_failure_copy import FAILED_TURN_NOTICE, PARTIAL_FAILED_TURN_NOTICE
import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource
from gateway.session_transcript import TranscriptReadError


def _bootstrap(monkeypatch, tmp_path):
    """Minimal GatewayRunner setup shared by all tests in this module."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    config = GatewayConfig()
    runner = gateway_run.GatewayRunner(config)
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._begin_session_run_generation = lambda _key: 1
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001:12345",
        session_id="sess-dedup",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    # Mock has_platform_message_id to return False so the dedupe guard
    # (#47237) in gateway/run.py does not skip the append_to_transcript call.
    runner.session_store.has_platform_message_id.return_value = False
    # The durable tail after the user row landed (gateway write or agent flush) is that user row.
    runner.session_store.transcript_tail_role.return_value = "user"
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


def _event():
    return MessageEvent(
        text="hello world",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001",
            chat_type="group",
            user_id="12345",
        ),
        message_id="msg-42",
    )


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
    )


def _assert_user_call_has_skip_db(calls, expected_skip_db: bool):
    """Find append_to_transcript calls with role='user' and check skip_db."""
    user_calls = []
    for call in calls:
        args = call.args
        if len(args) >= 2 and isinstance(args[1], dict):
            if args[1].get("role") == "user":
                user_calls.append(call)
    assert len(user_calls) >= 1, (
        f"Expected at least one user-role append_to_transcript call, "
        f"got calls: {[c.args for c in calls if len(c.args)>=2]}"
    )
    for call in user_calls:
        actual = call.kwargs.get("skip_db", False)
        assert actual == expected_skip_db, (
            f"Expected skip_db={expected_skip_db} for user-role call, "
            f"got skip_db={actual}. kwargs={call.kwargs}"
        )


def _batch_rows(runner, role: str | None = None):
    """Rows the gateway recovery path (#R67-2) batched through ``append_transcript_batch``
    (row-id-repair-aware write replacing the old per-row ``append_to_transcript(skip_db=...)``
    fallback for every write that is NOT skipped outright)."""
    rows = []
    for call in runner.session_store.append_transcript_batch.call_args_list:
        if len(call.args) >= 2 and isinstance(call.args[1], list):
            rows.extend(r for r in call.args[1] if isinstance(r, dict))
    return [r for r in rows if role is None or r.get("role") == role]


# ── Test 1: agent_failed_early path uses skip_db=True ─────────────────


@pytest.mark.asyncio
async def test_agent_failed_early_skip_db_when_agent_has_session_db(
    monkeypatch, tmp_path
):
    runner = _bootstrap(monkeypatch, tmp_path)

    # Agent fails with transient 429
    runner._run_agent = AsyncMock(
        return_value={
            "failed": True,
            "final_response": "API call failed after 3 retries: 429 Too Many Requests",
            "error": "429 Too Many Requests — rate limit exceeded",
            "messages": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    # agent_persisted=True (the agent has its own _session_db): the gateway's recovery write
    # must be skipped outright — no call to either write path for the user row, not a
    # ``skip_db=True`` no-op call (#R67-2 batches recovery writes through
    # ``append_transcript_batch`` instead of a per-row ``append_to_transcript``).
    assert _batch_rows(runner, role="user") == []
    assert not any(
        len(call.args) >= 2 and call.args[1].get("role") == "user"
        for call in runner.session_store.append_to_transcript.call_args_list
    )
    assert FAILED_TURN_NOTICE in response

    # The failed-turn boundary (closed via _hmwa_close_failed_turn, unchanged by #R67-2) is the
    # only append_to_transcript call left.
    transcript_rows = [
        call.args[1]
        for call in runner.session_store.append_to_transcript.call_args_list
        if len(call.args) >= 2 and call.args[1].get("role") in {"user", "assistant"}
    ]
    assert [row["role"] for row in transcript_rows] == ["assistant"]
    assert transcript_rows[-1]["content"] == FAILED_TURN_NOTICE

    # The next unrelated input remains its own turn instead of alternation repair merging the
    # failed mutating request into it. The user row itself landed via the agent's own
    # persistence (skip_db case), not through this mock, so it's reconstructed here to match
    # what the durable transcript actually holds.
    from agent.agent_runtime_helpers import repair_message_sequence

    replay = [{"role": "user", "content": "hello world"}, *transcript_rows,
              {"role": "user", "content": "unrelated question"}]
    assert repair_message_sequence(None, replay) == 0
    assert replay[-1]["content"] == "unrelated question"



@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tail_role, expected_roles",
    [("user", ["assistant"]), ("assistant", [])],
    ids=["agent-flushed-user-row-still-closed", "redelivery-of-closed-turn-adds-nothing"],
)
async def test_boundary_keyed_on_durable_tail_when_user_row_is_deduped(
    monkeypatch, tmp_path, tail_role, expected_roles
):
    """The platform-id dedupe skips the gateway's user write in two production shapes: the agent's
    own turn-start flush already persisted THIS turn's row (tail = user → boundary must still land),
    and a platform redelivery of an already-closed turn (tail = boundary → nothing may stack)."""
    runner = _bootstrap(monkeypatch, tmp_path)
    runner.session_store.has_platform_message_id.return_value = True
    runner.session_store.transcript_tail_role.return_value = tail_role
    runner._run_agent = AsyncMock(
        return_value={
            "failed": True,
            "final_response": "API call failed after 3 retries: 429 Too Many Requests",
            "error": "429 Too Many Requests — rate limit exceeded",
            "messages": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(_event(), _source(), "agent:main:telegram:group:-1001:12345", 1)

    rows = [
        call.args[1] for call in runner.session_store.append_to_transcript.call_args_list
        if len(call.args) >= 2 and call.args[1].get("role") in {"user", "assistant"}
    ]
    assert [row["role"] for row in rows] == expected_roles
    assert all(row["content"] == FAILED_TURN_NOTICE for row in rows)


@pytest.mark.asyncio
async def test_failed_turn_with_tool_activity_does_not_recommend_blind_retry(
    monkeypatch, tmp_path
):
    runner = _bootstrap(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "failed": True,
            "final_response": "API call failed after 3 retries: 500 Internal Server Error",
            "error": "500 Internal Server Error",
            "messages": [
                {"role": "user", "content": "reset the password"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "reset_password", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "Password reset"},
            ],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assistant_rows = [
        call.args[1]
        for call in runner.session_store.append_to_transcript.call_args_list
        if len(call.args) >= 2 and call.args[1].get("role") == "assistant"
    ]
    assert len(assistant_rows) == 1
    assert assistant_rows[0]["content"] == PARTIAL_FAILED_TURN_NOTICE
    assert PARTIAL_FAILED_TURN_NOTICE in response
    assert "not processed" not in response
    assert "Send it again" not in response


# ── Test 2: agent_failed_early with no _session_db → skip_db not True ─


# ── Test 3: not-new-messages path uses skip_db=True ───────────────────


@pytest.mark.asyncio
async def test_not_new_messages_skip_db_when_agent_has_session_db(
    monkeypatch, tmp_path
):
    runner = _bootstrap(monkeypatch, tmp_path)

    # Agent succeeds but history_offset equals messages length → no new messages
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Hello!",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
            "history_offset": 1,  # equals len(messages) → new_messages=[]
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    # agent_persisted=True → the gateway recovery write is skipped outright, not called with
    # skip_db=True (#R67-2 moved this write to append_transcript_batch).
    assert _batch_rows(runner, role="user") == []
    assert not any(
        len(call.args) >= 2 and call.args[1].get("role") == "user"
        for call in runner.session_store.append_to_transcript.call_args_list
    )


@pytest.mark.asyncio
async def test_transcript_read_failure_stops_turn_before_agent_or_append(
    monkeypatch, tmp_path
):
    runner = _bootstrap(monkeypatch, tmp_path)
    runner.session_store.load_transcript.side_effect = TranscriptReadError("sess-dedup")
    runner._run_agent = AsyncMock()

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert "history is temporarily unavailable" in response
    assert "not processed" in response
    runner._run_agent.assert_not_awaited()
    runner.session_store.append_to_transcript.assert_not_called()


# ── Post-stream MEDIA delivery keeps prior-turn deduplication ──────────


# ── Test 4: normal path (new_messages found) uses skip_db=True ────────


# ── R67-2 (L4-2): agent_persisted=False must not re-insert already-committed rows ──────


@pytest.mark.asyncio
async def test_agent_persisted_false_only_writes_the_unpersisted_tail(monkeypatch, tmp_path):
    """Turn-start persistence (and any mid-turn incremental flush) can already have committed
    the user row and earlier tool exchanges before the FINAL flush fails and the turn reports
    ``agent_persisted=False`` (#L4-2/R67-1 now makes this a real, reachable signal). The gateway's
    new_messages loop used to apply one blanket ``skip_db=agent_persisted`` to the entire
    current-turn suffix, so a partial failure re-inserted rows the agent already wrote — a
    probe through the real append path produced ``old user, old assistant, new user, new user,
    new assistant``. Only rows the agent never persisted (no ``_db_persisted`` marker — including
    ones the finalizer's blank-row fill cleared the marker on for a rewrite) may reach the DB."""
    runner = _bootstrap(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "final answer",
            "messages": [
                {"role": "user", "content": "hi", "_db_persisted": True},
                {
                    "role": "assistant", "content": "", "_db_persisted": True,
                    "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "ok", "_db_persisted": True},
                {"role": "assistant", "content": "final answer"},  # the flush that FAILED
            ],
            "tools": [],
            "history_offset": 0,
            "agent_persisted": False,
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    # #R67-2 batches the unpersisted suffix through append_transcript_batch (row-id-repair-aware)
    # instead of looping per-row append_to_transcript(skip_db=...) calls; a row carrying
    # ``_db_persisted`` is now excluded from the write entirely rather than passed with
    # skip_db=True.
    single_calls = runner.session_store.append_to_transcript.call_args_list
    written_single = {
        call.args[1]["content"]
        for call in single_calls
        if len(call.args) >= 2 and isinstance(call.args[1], dict)
        and call.args[1].get("role") in {"user", "assistant", "tool"}
    }
    batched = _batch_rows(runner)
    written_batched = {row["content"] for row in batched}
    assert "hi" not in written_single and "hi" not in written_batched
    assert "ok" not in written_single and "ok" not in written_batched
    assert "final answer" in written_batched, (written_single, written_batched)
    assert "final answer" not in written_single


@pytest.mark.asyncio
async def test_agent_persisted_false_still_writes_everything_without_markers(monkeypatch, tmp_path):
    """Guardrail: a runtime that never flushed anything itself (every message lacks
    ``_db_persisted``) must keep writing the whole new-message tail, same as before this fix."""
    runner = _bootstrap(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "final answer",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "final answer"},
            ],
            "tools": [],
            "history_offset": 0,
            "agent_persisted": False,
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    # Neither row carries ``_db_persisted``, so both go through append_transcript_batch
    # (#R67-2's row-id-repair-aware recovery write), not the old skip_db=False per-row call.
    user_rows = _batch_rows(runner, role="user")
    assert len(user_rows) == 1 and user_rows[0]["content"] == "hi"


