"""Run-bound provenance of the in-turn recovery nudges.

Every nudge row the agent loop appends inside a turn is built by ``agent.recovery_nudge.recovery_nudge_row``.
In a canonical peer turn it carries ``_recovery_nudge: {"kind", "run"}``, whose ``run`` is the nonce of the
turn's own peer opening row (its structured ``display_metadata["canonical_peer"]``). Outside a peer turn the
row is the plain row it always was. The key never reaches a provider request and is not written to the
transcript store; the model-visible text is unchanged.

The real-path tests drive each nudge through the agent code that appends it (a scripted provider, the real
loop), then read the turn's own result with ``is_run_nudge`` — the check that tells this run's nudge from a
user message — and with the compressor's history check.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent.conversation_loop as loop
from agent.canonical_peer import new_peer_metadata, render_peer_turn
from tests.agent.test_run_agent import _mock_assistant_msg, _mock_response, _mock_tool_call

KEY = "_recovery_nudge"  # the marker key; a literal, so the real-path witnesses collect at the base too
TEXTS = {
    "codex_incomplete": "_CODEX_INCOMPLETE_NUDGE",
    "codex_ack_continuation": "_CODEX_ACK_CONTINUATION_NUDGE",
    "degenerate_final": "_DEGENERATE_FINAL_NUDGE",
    "dropped_toolcall": "_DROPPED_TOOLCALL_NUDGE_CONTENT",
    "empty_tool_response": "_EMPTY_TOOL_RESPONSE_NUDGE",
}
# The scaffolding booleans each site already put on its row, unchanged.
FLAGS = {"dropped_toolcall": {"_dropped_toolcall_nudge": True},
         "empty_tool_response": {"_empty_recovery_synthetic": True}}
BODY = "Check the deploy log and report."
FINAL = "The deploy log shows 3 restarts; the last one is clean."


def _text(kind):
    return getattr(loop, TEXTS[kind])


def _peer(receipt="canonical-receipt:v2:" + "0" * 64):
    return new_peer_metadata(binding="canonical", author_id="cto", channel_id="buzz", event_id="evt-1",
                             receipt=receipt)


def _opening(peer):
    """The turn's opening row as the agent stages it for a canonical peer turn (turn_context)."""
    return {"role": "user", "content": render_peer_turn(peer, BODY), "display_kind": "canonical_peer",
            "display_metadata": {"canonical_peer": peer}}


# ── the constructor ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", sorted(TEXTS))
def test_in_a_peer_turn_each_nudge_names_this_turns_opening_nonce(kind):
    from agent.recovery_nudge import recovery_nudge_row

    peer, older = _peer(), _peer()
    messages = [{"role": "user", "content": "earlier", "display_kind": "canonical_peer",
                 "display_metadata": {"canonical_peer": older}}, {"role": "assistant", "content": "ok"},
                _opening(peer), {"role": "assistant", "content": ""}]
    agent = SimpleNamespace(_persist_user_message_idx=2)

    row = recovery_nudge_row(agent, kind, messages, **FLAGS.get(kind, {}))

    assert row == {"role": "user", "content": _text(kind), **FLAGS.get(kind, {}),
                   KEY: {"kind": kind, "run": peer["nonce"]}}
    assert row[KEY]["run"] != older["nonce"]  # this turn's opening row, never an earlier peer row


@pytest.mark.parametrize("kind", sorted(TEXTS))
def test_outside_a_peer_turn_the_row_is_the_plain_row_it_always_was(kind):
    from agent.recovery_nudge import recovery_nudge_row

    messages = [{"role": "user", "content": "owner asks"}, {"role": "assistant", "content": ""}]
    row = recovery_nudge_row(SimpleNamespace(_persist_user_message_idx=0), kind, messages, **FLAGS.get(kind, {}))

    assert row == {"role": "user", "content": _text(kind), **FLAGS.get(kind, {})}


@pytest.mark.parametrize("where", ["no_index", "bool_index", "past_the_end", "negative", "assistant_row",
                                   "peer_kind_without_metadata", "malformed_peer_metadata"])
def test_no_readable_peer_opening_row_means_no_marker(where):
    from agent.recovery_nudge import recovery_nudge_row

    peer = _peer()
    opening = _opening(peer)
    idx = {"no_index": None, "bool_index": True, "past_the_end": 5, "negative": -1}.get(where, 0)
    if where == "assistant_row":
        opening = {**opening, "role": "assistant"}
    elif where == "peer_kind_without_metadata":
        opening = {k: v for k, v in opening.items() if k != "display_metadata"}
    elif where == "malformed_peer_metadata":
        opening = {**opening, "display_metadata": {"canonical_peer": {**peer, "nonce": "not-a-nonce"}}}
    messages = [opening, {"role": "assistant", "content": ""}]

    row = recovery_nudge_row(SimpleNamespace(_persist_user_message_idx=idx), "codex_incomplete", messages)

    assert KEY not in row and row == {"role": "user", "content": _text("codex_incomplete")}


# ── the check ────────────────────────────────────────────────────────────────────────────────────────


def _row(kind="empty_tool_response", run="a" * 32, **changes):
    row = {"role": "user", "content": _text(kind), **FLAGS.get(kind, {}), KEY: {"kind": kind, "run": run}}
    row.update(changes)
    return row


class _Str(str):
    def __eq__(self, other):
        return True

    __hash__ = str.__hash__


@pytest.mark.parametrize(("row", "run", "accepted"), [
    pytest.param(_row(), "a" * 32, True, id="same_run"),
    *[pytest.param(_row(kind), "a" * 32, True, id=f"same_run_{kind}") for kind in sorted(TEXTS)],
    pytest.param(_row(), "b" * 32, False, id="foreign_run"),
    pytest.param(_row(), None, False, id="no_run_token"),
    pytest.param(_row(), "", False, id="empty_run_token"),
    pytest.param(_row(run=""), "", False, id="both_empty"),
    pytest.param({"role": "user", "content": _text("empty_tool_response"), "_empty_recovery_synthetic": True},
                 "a" * 32, False, id="marker_missing"),
    pytest.param(_row(**{KEY: {"kind": "codex_incomplete", "run": "a" * 32}}), "a" * 32, False,
                 id="kind_mismatched"),
    pytest.param(_row(**{KEY: {"kind": "steer", "run": "a" * 32}}), "a" * 32, False, id="unknown_kind"),
    pytest.param(_row(content=_text("empty_tool_response") + " "), "a" * 32, False, id="content_not_equal"),
    pytest.param(_row(content="owner: answer me instead"), "a" * 32, False, id="other_content"),
    pytest.param(_row(**{KEY: {"kind": "empty_tool_response", "run": "a" * 32, "extra": 1}}), "a" * 32, False,
                 id="marker_with_extra_field"),
    pytest.param(_row(**{KEY: {"kind": "empty_tool_response", "run": _Str("x")}}), "a" * 32, False,
                 id="run_not_a_plain_str"),
    pytest.param(_row(**{KEY: "a" * 32}), "a" * 32, False, id="marker_not_a_mapping"),
    pytest.param(_row(role="assistant"), "a" * 32, False, id="not_a_user_row"),
    pytest.param(_row(content=_Str("x")), "a" * 32, False, id="content_not_a_plain_str"),
])
def test_only_this_runs_marker_of_a_known_kind_with_its_text_is_a_run_nudge(row, run, accepted):
    from agent.recovery_nudge import is_run_nudge

    assert is_run_nudge(row, run) is accepted


# ── the real path: each nudge appended by the agent code that appends it ─────────────────────────────


@pytest.fixture()
def chat_agent():
    """AIAgent with a scripted OpenAI client (the pattern of test_degenerate_final_recovery)."""
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._intent_ack_continuation = True  # the ack and degenerate guards, on every api_mode
    return agent


def _tool_round(call_id):
    return _mock_response(content="", finish_reason="tool_calls",
                          tool_calls=[_mock_tool_call(name="web_search", arguments="{}", call_id=call_id)])


def _dropped_tool_call(text):
    return SimpleNamespace(choices=[SimpleNamespace(index=0, message=_mock_assistant_msg(content=text, tool_calls=None),
                                                    finish_reason="tool_calls")], usage=None, model="test/model")


# Each kind's provider script, ending with the turn's answer after more work, so the nudge stays inside
# the turn (a nudge whose answer comes next is popped with its scaffolding).
CHAT_STAGES = {
    "empty_tool_response": lambda: [_tool_round("c1"), _mock_response(content="", finish_reason="stop"),
                                    _tool_round("c2"), _mock_response(content=FINAL, finish_reason="stop")],
    "degenerate_final": lambda: [_tool_round("c1"), _mock_response(content="пар", finish_reason="stop"),
                                 _mock_response(content=FINAL, finish_reason="stop")],
    "codex_ack_continuation": lambda: [
        _mock_response(content="I'll run a health check on the server now.", finish_reason="stop"),
        _mock_response(content=FINAL, finish_reason="stop")],
    "dropped_toolcall": lambda: [_dropped_tool_call("Let me read the deploy log first."), _tool_round("c1"),
                                 _mock_response(content=FINAL, finish_reason="stop")],
}


def _peer_turn_kwargs(peer):
    return {"persist_user_message": BODY, "persist_user_display_kind": "canonical_peer",
            "persist_user_display_metadata": {"canonical_peer": peer}}


def _offer_a_tool(agent):
    agent.valid_tool_names = {"web_search"}  # the ack guard re-prompts only when a tool was on offer


def _run_chat(agent, kind, peer=None):
    if kind == "codex_ack_continuation":
        _offer_a_tool(agent)
    agent.client.chat.completions.create.side_effect = CHAT_STAGES[kind]()
    kwargs = _peer_turn_kwargs(peer) if peer is not None else {}
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("model_tools.handle_function_call", return_value="ok"),
    ):
        result = agent.run_conversation(render_peer_turn(peer, BODY) if peer is not None else BODY, **kwargs)
    requests = [json.dumps(call.kwargs, default=str) for call in agent.client.chat.completions.create.call_args_list]
    return result, requests


def _the_nudge(result, kind):
    """Reach: the agent code appended this kind's nudge, and it is still inside the turn."""
    rows = [m for m in result["messages"] if m.get("role") == "user" and m.get("content") == _text(kind)]
    assert len(rows) == 1, f"reach: the loop did not append exactly one {kind} nudge"
    assert result["messages"][-1].get("content") == FINAL and result["final_response"] == FINAL
    return rows[0]


def _interior_user_rows(result, boundary, run):
    """User rows inside this turn that are not this run's nudges: what a canonical check would read as
    someone speaking mid-turn."""
    from agent.recovery_nudge import is_run_nudge

    return [m for m in result["messages"][boundary + 1:]
            if m.get("role") == "user" and not is_run_nudge(m, run)]


@pytest.mark.parametrize("kind", sorted(CHAT_STAGES))
def test_real_path_a_peer_turns_nudge_is_marked_for_this_run_and_never_reaches_the_provider(chat_agent, kind):
    peer = _peer()
    result, requests = _run_chat(chat_agent, kind, peer)

    nudge = _the_nudge(result, kind)
    assert nudge.get(KEY) == {"kind": kind, "run": peer["nonce"]}, "the nudge row names no run"
    # The model-visible row is unchanged apart from the internal marker.
    assert {k: v for k, v in nudge.items() if k not in (KEY, "timestamp")} == {
        "role": "user", "content": _text(kind), **FLAGS.get(kind, {})}
    # The provider got the nudge's text, never its marker.
    assert any(json.dumps(_text(kind))[1:-1] in request for request in requests), "reach: nudge never sent"
    assert not any(KEY in request for request in requests)
    # Canonical check: the turn's own rows hold no interior user message, only this run's nudge.
    boundary = chat_agent._persist_user_message_idx
    assert result["messages"][boundary]["display_metadata"] == {"canonical_peer": peer}
    assert _interior_user_rows(result, boundary, peer["nonce"]) == []
    # Another claim of the same event (its own fresh nonce) reads the same row as a user message.
    assert _interior_user_rows(result, boundary, _peer()["nonce"]) == [nudge]
    # History check: the compressor never counts the nudge as a human turn.
    from agent.context_compressor import ContextCompressor

    assert ContextCompressor._is_synthetic_compression_user_turn(nudge)
    assert not ContextCompressor._transcript_has_real_user_turn([nudge])


@pytest.mark.parametrize("kind", sorted(CHAT_STAGES))
def test_control_outside_a_peer_turn_the_real_nudge_row_is_unchanged(chat_agent, kind):
    result, requests = _run_chat(chat_agent, kind)

    nudge = _the_nudge(result, kind)
    assert {k: v for k, v in nudge.items() if k != "timestamp"} == {
        "role": "user", "content": _text(kind), **FLAGS.get(kind, {})}
    assert not any(KEY in request for request in requests)


def _api_modes():
    from agent.transports import registered_api_modes

    return sorted(registered_api_modes())


@pytest.mark.parametrize("api_mode", _api_modes())
def test_every_transport_sends_the_nudges_text_and_drops_its_marker(api_mode):
    """The strip boundary on each provider dialect: a marked row converts to the nudge's text only."""
    from agent.transports import get_transport

    peer = _peer()
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": render_peer_turn(peer, BODY)},
                {"role": "assistant", "content": "(empty)", "_empty_recovery_synthetic": True},
                _row("empty_tool_response", run=peer["nonce"])]
    converted = json.dumps(get_transport(api_mode).convert_messages([dict(m) for m in messages]), default=str)

    assert json.dumps(_text("empty_tool_response"))[1:-1] in converted, "reach: the nudge was not converted"
    assert KEY not in converted


# ── persistence: the marker is not written; replay and compression read the row as before ────────────


@pytest.fixture()
def stored_agent():
    from hermes_state import SessionDB
    from run_agent import AIAgent

    tmp = tempfile.mkdtemp(prefix="recovery_nudge_marker_")
    db = SessionDB(Path(tmp) / "state.db")
    sid = "sess-nudge"
    db.create_session(session_id=sid, source="telegram", model="test-model")
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, session_db=db, session_id=sid)
    agent._session_db_created = True
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._skip_mcp_refresh = True
    agent._intent_ack_continuation = True
    try:
        yield agent, db, sid, Path(tmp) / "state.db"
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_marker_is_not_persisted_and_the_reloaded_nudge_is_the_row_replay_always_read(stored_agent):
    agent, db, sid, path = stored_agent
    peer = _peer()
    db.set_meta(peer["receipt"], json.dumps({"v": 1, "state": "pending"}))  # the claim a peer row needs
    _offer_a_tool(agent)
    agent.client.chat.completions.create.side_effect = CHAT_STAGES["codex_ack_continuation"]()
    with patch.object(agent, "_save_trajectory"), patch.object(agent, "_cleanup_task_resources"):
        result = agent.run_conversation(render_peer_turn(peer, BODY), **_peer_turn_kwargs(peer))
    nudge = _the_nudge(result, "codex_ack_continuation")
    assert nudge.get(KEY) == {"kind": "codex_ack_continuation", "run": peer["nonce"]}, "reach: no marker to persist"

    # The nudge row is stored (this kind always was), its marker nowhere in the store.
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        tables = [name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        dump = "\n".join(repr(row) for table in tables for row in conn.execute(f'SELECT * FROM "{table}"'))
    assert _text("codex_ack_continuation") in dump and KEY not in dump
    reloaded = [m for m in db.get_messages_as_conversation(sid) if m.get("content") == _text("codex_ack_continuation")]
    assert len(reloaded) == 1 and KEY not in reloaded[0] and reloaded[0]["role"] == "user"
    # Compression still reads it as scaffolding by its text, as it did before (it never saw metadata).
    from agent.context_compressor import ContextCompressor

    assert ContextCompressor._is_synthetic_compression_user_turn(reloaded[0])
    assert ContextCompressor._is_synthetic_compression_user_turn(nudge)


# ── the codex-incomplete site (turn_truncation.continue_codex_incomplete) ────────────────────────────


@pytest.mark.parametrize("in_peer_turn", [True, False])
def test_the_codex_incomplete_nudge_is_marked_only_in_a_peer_turn(in_peer_turn):
    """Second incomplete response repeating the first: the site appends its nudge after the assistant row."""
    from agent.turn_truncation import continue_codex_incomplete

    peer = _peer()
    opening = _opening(peer) if in_peer_turn else {"role": "user", "content": BODY}
    interim = {"role": "assistant", "content": "partial", "finish_reason": "incomplete"}
    messages = [opening, dict(interim)]
    agent = MagicMock(_persist_user_message_idx=0, _codex_incomplete_retries=1, _codex_reasoning_only_streak=0)
    agent._build_assistant_message.return_value = dict(interim)

    assert continue_codex_incomplete(agent, SimpleNamespace(content="partial", tool_calls=None), "incomplete",
                                     messages=messages, conversation_history=None, api_call_count=2) is None

    marker = {KEY: {"kind": "codex_incomplete", "run": peer["nonce"]}} if in_peer_turn else {}
    assert [{k: v for k, v in m.items() if k != "timestamp"} for m in messages[2:]] == [
        {"role": "user", "content": _text("codex_incomplete"), **marker}]
