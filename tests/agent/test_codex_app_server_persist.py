"""Regression for #49225 — codex app-server turns must reach the session DB
exactly once, and say honestly whether they did.

The codex app-server runtime (``run_codex_app_server_turn``) is an early-return
path that bypasses ``conversation_loop`` and therefore never runs the loop's
per-step ``_persist_session()`` flushes. Before the fix, the projected
assistant/tool messages were persisted *nowhere* (state.db got only
session_meta rows), leaving ``session_search`` (FTS) and conversation-distill
blind to real gateway conversations.

The fix has the codex runtime flush its own projected messages via
``_flush_messages_to_session_db()`` (idempotent through the intrinsic
``_DB_PERSISTED_MARKER``). The inbound user turn is already flushed at turn
start (``turn_context._persist_session``), and ``append_message`` is a raw
INSERT with no dedup — a gateway re-write would duplicate the user turn
(#860 / #42039). So the gateway skips its own write only on
``agent_persisted=True``, and that value is this turn's real outcome: the
flush returned True and every row of the turn carries the marker. A failed,
raising or unclear flush, or (with nothing projected) a failed turn-start
flush, reports False so the gateway keeps its recovery copy. The same honest
value rides on the generic loop's early ``partial_result`` returns.
"""

import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.codex_runtime import run_codex_app_server_turn
from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.turn_response_intake import normalize_model_response
from agent.turn_truncation import _Trunc, continue_codex_incomplete, turn_rows_persisted
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.agent.test_run_agent import _mock_response

def _make_turn(projected=None):
    return SimpleNamespace(
        interrupted=False,
        error=None,
        thread_id="thread-1",
        turn_id="turn-1",
        projected_messages=[{"role": "assistant", "content": "CODEX_ASSISTANT"}] if projected is None else projected,
        tool_iterations=0,
        final_text="CODEX_ASSISTANT",
        should_retire=False,
    )

def _make_agent(session_db=None, session_id="sess-codex"):
    agent = MagicMock()
    # Pre-seed the session so run_codex_app_server_turn skips the spawn block.
    agent._codex_session = MagicMock()
    agent._codex_session.run_turn.return_value = _make_turn()
    agent._codex_session_prompt = None  # seeded session: no recorded composition to compare
    agent.tool_progress_callback = None
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._session_db = session_db
    agent._session_db_created = True
    agent.session_id = session_id
    return agent

@pytest.fixture
def real_agent(tmp_path):
    """A real AIAgent on a real SessionDB, so the flush and its markers are the shipped ones."""
    db = SessionDB(tmp_path / "state.db")
    sid = "sess-codex-once"
    db.create_session(session_id=sid, source="telegram", model="codex")
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_db=db,
        session_id=sid,
    )
    agent._session_db_created = True
    agent._codex_session = MagicMock()
    agent._codex_session.run_turn.return_value = _make_turn()
    agent.tool_progress_callback = None
    yield agent
    db.close()

def _break_writes(agent, after=0):
    """Every session-DB append after the next ``after`` raises, as a locked or full disk does."""
    real, calls = agent._session_db.append_messages_batch, []

    def append(*args, **kwargs):
        calls.append(None)
        if len(calls) > after:
            raise sqlite3.OperationalError("database is locked")
        return real(*args, **kwargs)

    agent._session_db.append_messages_batch = append

def _run(agent, messages, text="hello"):
    return run_codex_app_server_turn(
        agent, user_message=text, original_user_message=text, messages=messages, effective_task_id="task-1",
    )

def test_codex_success_flushes_and_reports_persisted():
    """No session DB configured: nothing was owed, so the gateway may skip."""
    agent = _make_agent(session_db=None)
    result = _run(agent, [{"role": "user", "content": "hello"}])
    assert result["completed"] is True
    assert isinstance(result["messages"][-1]["timestamp"], float)
    assert result["agent_persisted"] is True

def test_codex_user_interrupt_is_reported_and_cleared():
    agent = _make_agent(session_db=None)
    turn = _make_turn()
    turn.interrupted = True
    turn.final_text = ""
    agent._codex_session.run_turn.return_value = turn
    agent._interrupt_requested = True
    agent._interrupt_message = "new correction"

    def clear_interrupt():
        agent._interrupt_requested = False
        agent._interrupt_message = None

    agent.clear_interrupt.side_effect = clear_interrupt
    result = _run(agent, [{"role": "user", "content": "hello"}])

    assert result["interrupted"] is True
    assert result["interrupt_message"] == "new correction"
    agent.clear_interrupt.assert_called_once_with()
    assert agent._interrupt_requested is False

def test_codex_turn_persists_each_message_exactly_once(real_agent):
    """The normal path: the user turn (flushed at turn start) is not duplicated, the projected
    assistant message lands once, and the turn reports agent_persisted=True."""
    db, sid = real_agent._session_db, real_agent.session_id
    # Model the real flow: the turn-start flush stamps _DB_PERSISTED_MARKER on the user dict of
    # the SAME `messages` list the codex path later reuses, so the codex flush skips it.
    messages = [{"role": "user", "content": "USER_TURN"}]
    real_agent._flush_messages_to_session_db(messages)  # turn-start flush

    result = _run(real_agent, messages, "USER_TURN")
    assert result["agent_persisted"] is True

    rows = db.get_messages(sid, include_inactive=True)
    contents = [r["content"] for r in rows]
    assert contents.count("USER_TURN") == 1, contents
    assert contents.count("CODEX_ASSISTANT") == 1, contents
    assistant_row = next(row for row in rows if row["content"] == "CODEX_ASSISTANT")
    assert isinstance(assistant_row["timestamp"], float)
    assert sid in {r["session_id"] for r in db.search_messages("CODEX_ASSISTANT")}

def test_codex_flush_failure_reports_not_persisted(real_agent):
    """Counterexample 1: the projected rows' flush fails, so the gateway must keep its own copy."""
    messages = [{"role": "user", "content": "USER_TURN"}]
    real_agent._flush_messages_to_session_db(messages)  # turn-start flush lands
    _break_writes(real_agent)

    result = _run(real_agent, messages, "USER_TURN")

    assert result["completed"] is True
    assert result["agent_persisted"] is False
    contents = [r["content"] for r in real_agent._session_db.get_messages(real_agent.session_id)]
    assert "CODEX_ASSISTANT" not in contents

@pytest.mark.parametrize("outcome", [False, None, "raise"])
def test_codex_flush_without_a_true_result_is_not_persisted(outcome):
    """A flush that failed, raised or answered unclearly is not success, even with every row stamped."""
    agent = _make_agent(session_db=MagicMock())

    def flush(messages):
        for message in messages:
            message[_DB_PERSISTED_MARKER] = True
        if outcome == "raise":
            raise sqlite3.OperationalError("disk I/O error")
        return outcome

    agent._flush_messages_to_session_db = MagicMock(side_effect=flush)
    assert _run(agent, [{"role": "user", "content": "hello"}])["agent_persisted"] is False

def test_codex_flush_true_without_durable_rows_is_not_persisted():
    """A flush call returning True does not prove the rows exist: the markers must."""
    agent = _make_agent(session_db=MagicMock())
    agent._flush_messages_to_session_db = MagicMock(return_value=True)
    assert _run(agent, [{"role": "user", "content": "hello"}])["agent_persisted"] is False

def test_codex_no_projected_rows_keeps_turn_start_failure(real_agent):
    """Counterexample 2: nothing projected means no codex flush, so the failed turn-start flush stands."""
    _break_writes(real_agent)
    messages = [{"role": "user", "content": "USER_TURN"}]
    assert real_agent._flush_messages_to_session_db(messages) is False  # turn-start flush fails
    real_agent._codex_session.run_turn.return_value = _make_turn(projected=[])

    result = _run(real_agent, messages, "USER_TURN")

    assert result["completed"] is True
    assert result["agent_persisted"] is False

def test_codex_persisted_is_this_turns_outcome(real_agent):
    """An earlier turn's success never answers for a later turn whose turn-start flush failed."""
    real_agent._codex_session.run_turn.return_value = _make_turn(projected=[])
    first = [{"role": "user", "content": "FIRST"}]
    real_agent._flush_messages_to_session_db(first)
    assert _run(real_agent, first, "FIRST")["agent_persisted"] is True

    _break_writes(real_agent)
    second = [*first, {"role": "user", "content": "SECOND"}]
    assert real_agent._flush_messages_to_session_db(second) is False
    assert _run(real_agent, second, "SECOND")["agent_persisted"] is False

def test_turn_rows_persisted_reads_the_rows():
    """Rows the flush never writes are not required; no rows, or no list, is no proof."""
    with_db, no_db = SimpleNamespace(_session_db=object()), SimpleNamespace(_session_db=None)
    scaffold = {"role": "user", "content": "retry", "_empty_recovery_synthetic": True}
    assert turn_rows_persisted(with_db, [{"role": "user", _DB_PERSISTED_MARKER: True}, "junk", scaffold]) is True
    assert turn_rows_persisted(with_db, [{"role": "user", _DB_PERSISTED_MARKER: True}, {"role": "assistant"}]) is False
    assert turn_rows_persisted(with_db, [scaffold]) is False
    assert turn_rows_persisted(with_db, []) is False
    assert turn_rows_persisted(with_db, None) is False
    assert turn_rows_persisted(no_db, []) is True

# --- The generic loop's early returns: _persist_session() then partial_result(agent=) -------------

def _end_turn(agent, messages):
    st = _Trunc(
        agent=agent, response=None, finish_reason="length", conversation_history=[], api_call_count=1,
        effective_task_id="t", current_turn_user_idx=0, messages=messages, length_continue_retries=0,
        truncated_response_parts=[], truncated_tool_call_retries=0, retry_count=0, compression_attempts=0,
    )
    return st.end_turn("final text", "error text", cleanup=False).result

def _codex_incomplete_exhausted(agent, messages):
    agent._codex_incomplete_retries = 2  # this call makes it 3: the terminal partial result
    agent._codex_reasoning_only_streak = 0
    agent._build_assistant_message = lambda msg, fr: {"role": "assistant", "content": msg.content, "finish_reason": fr}
    message = SimpleNamespace(content="partial answer", tool_calls=None)
    return continue_codex_incomplete(
        agent, message, "incomplete", messages=messages, conversation_history=None, api_call_count=3,
    )

def _scratchpad_exhausted(agent, messages):
    agent._incomplete_scratchpad_retries = 2  # this call makes it 3: past the retry cap
    assistant = SimpleNamespace(content="<REASONING_SCRATCHPAD>thinking forever", finish_reason="length", tool_calls=None)
    agent._get_transport = lambda: SimpleNamespace(normalize_response=lambda response: assistant)
    return normalize_model_response(
        agent, response=SimpleNamespace(), messages=messages, api_messages=[], conversation_history=None,
        api_call_count=1, api_duration=0.1, api_start_time=0.0, api_request_id="req-1",
        effective_task_id="task-1", turn_id="turn-1",
    ).result

@pytest.mark.parametrize("writes_ok", [True, False])
@pytest.mark.parametrize("site", [_end_turn, _codex_incomplete_exhausted, _scratchpad_exhausted])
def test_partial_result_reports_its_own_flush(real_agent, site, writes_ok):
    """The normal _persist_session() path: each early return reports whether its flush landed."""
    if not writes_ok:
        _break_writes(real_agent)
    result = site(real_agent, [{"role": "user", "content": "write an essay"}])
    assert result["agent_persisted"] is writes_ok


# --- The main path: finalize_turn after its own _persist_session() ------------------------------------

@pytest.mark.parametrize(("appends_that_land", "persisted", "stored"), [
    (None, True, ["write an essay", "An essay."]), (1, False, ["write an essay"]), (0, False, [])])
def test_finalize_turn_reports_this_turns_flush(real_agent, appends_that_land, persisted, stored):
    """A plain text turn: the turn-start flush writes the ask, finalize_turn's flush the answer. Either
    failing reports False; a turn whose writes all land reports True."""
    agent = real_agent
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [_mock_response(content="An essay.", finish_reason="stop")]
    agent._cached_system_prompt, agent._use_prompt_caching, agent._skip_mcp_refresh = "You are helpful.", False, True
    agent.compression_enabled = agent.save_trajectories = False
    if appends_that_land is not None:
        _break_writes(agent, after=appends_that_land)
    with patch.object(agent, "_cleanup_task_resources"):
        result = agent.run_conversation("write an essay")

    assert result["final_response"] == "An essay." and result["completed"] is True
    assert [r["content"] for r in agent._session_db.get_messages(agent.session_id)] == stored  # reach
    assert result["agent_persisted"] is persisted
