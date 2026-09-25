"""skill_view repeat-view dedup across a transcript rewrite (compaction, committed proactive prune,
micro-compaction) and across a native Responses compaction checkpoint.

The "unchanged" stub is honest only while the earlier skill_view result is still in what the next request
carries. After a rewrite the stub must survive for a body the rewrite kept verbatim, and must be gone for a
body it cut so that skill is served in full again (ghost-skill defense, #32106). The cut body here keeps its
row in the inline-truncation shape: same opening bytes, longer than the served text, not the served text. A
native checkpoint takes every earlier item off the wire while the local transcript keeps it verbatim.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.native_compaction import has_compaction_checkpoint
from agent.tool_dispatch_helpers import make_tool_result_message
from tools.skills_tool import _skill_view_with_bump
from tools.skills_tool_dedup import reset_skill_view_dedup
from tools.tool_result_storage import generate_preview

TASK = "rewrite-task"
CHECKPOINT = {"type": "compaction", "encrypted_content": "opaque-checkpoint"}


@pytest.fixture
def served(tmp_path, monkeypatch):
    """Serve two real skills to TASK once each; return each one's (assistant call, tool result) rows."""
    home = tmp_path / ".hermes"
    for name in ("kept-skill", "lost-skill"):
        skill_dir = home / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Demo.\n---\n# {name}\n\nDo the {name} steps.\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    reset_skill_view_dedup()
    rows = {}
    for i, name in enumerate(("kept-skill", "lost-skill")):
        call_id = f"call_skill_{i}"
        call = {"id": call_id, "type": "function",
                "function": {"name": "skill_view", "arguments": json.dumps({"name": name})}}
        result = _skill_view_with_bump({"name": name}, task_id=TASK)
        rows[name] = [{"role": "assistant", "content": "", "tool_calls": [call]},
                      make_tool_result_message("skill_view", result, call_id)]
    yield rows
    reset_skill_view_dedup()


def _history(served):
    return [{"role": "user", "content": "load both skills"}, *served["kept-skill"], *served["lost-skill"],
            {"role": "assistant", "content": "loaded"}]


def _cut(content: str) -> str:
    """tool_result_storage's inline-truncation shape, its preview stopping just short of the end."""
    preview, _ = generate_preview(content, max_chars=len(content) - 20)
    return (f"{preview}\n\n[Truncated: tool response was {len(content):,} chars. "
            "Full output could not be saved to sandbox.]")


def _cut_lost(messages: list, served) -> list:
    lost_id = served["lost-skill"][1]["tool_call_id"]
    return [dict(m, content=_cut(m["content"])) if m.get("tool_call_id") == lost_id else m for m in messages]


def _dispatch(name, args, task_id=None, **_kw):
    """The real skill_view for the calling task; any other tool is a stand-in."""
    return _skill_view_with_bump(args, task_id=task_id) if name == "skill_view" else json.dumps({"ok": True})


def _assert_only_the_surviving_body_is_stubbed():
    kept = json.loads(_skill_view_with_bump({"name": "kept-skill"}, task_id=TASK))
    lost = json.loads(_skill_view_with_bump({"name": "lost-skill"}, task_id=TASK))
    assert kept.get("dedup") is True and "content" not in kept
    assert lost.get("dedup") is None and "Do the lost-skill steps." in lost["content"]


@pytest.mark.parametrize("loss", ["truncated", "behind_native_checkpoint"])
def test_compaction_keeps_the_stub_only_for_a_body_still_on_the_wire(served, loss):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                        quiet_mode=True, session_id="rewrite-session", skip_context_files=True, skip_memory=True)
    compressor = MagicMock()
    kept_call, kept_result = served["kept-skill"]
    if loss == "truncated":  # both in the protected tail; only lost-skill's result comes back cut
        tail = _cut_lost([kept_call, kept_result, *served["lost-skill"]], served)
    else:  # lost-skill's row stays verbatim, after an older checkpoint but ahead of the newest one
        older = {"role": "assistant", "content": "earlier", "codex_reasoning_items": [CHECKPOINT]}
        tail = [older, {"role": "user", "content": "continue"}, *served["lost-skill"],
                dict(kept_call, codex_reasoning_items=[CHECKPOINT]), kept_result]
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"}, *tail, {"role": "user", "content": "tail"}]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    agent.context_compressor = compressor

    agent._compress_context([*_history(served), {"role": "user", "content": "go on"}], "sys",
                            approx_tokens=120_000, task_id=TASK)

    _assert_only_the_surviving_body_is_stubbed()


def _native_checkpoint_turn(served, monkeypatch):
    """Codex route with native compaction on: the first response carries a server checkpoint and re-views
    kept-skill. Every item before the checkpoint leaves the wire; the local transcript keeps them."""
    from run_agent import AIAgent
    monkeypatch.setattr("agent.model_metadata._fetch_codex_oauth_context_lengths_with_source", lambda _t: ({}, False))
    agent = AIAgent(model="gpt-5.6-sol", base_url="https://chatgpt.com/backend-api/codex", api_key="codex-token",
                    quiet_mode=True, skip_context_files=True, skip_memory=True, max_iterations=10)
    agent.codex_responses_native_compaction = True
    agent.runtime_capabilities = {"native_compaction": True}
    usage = SimpleNamespace(input_tokens=9, output_tokens=1, total_tokens=10)
    responses = [
        SimpleNamespace(output=[
            SimpleNamespace(type="compaction", encrypted_content="opaque-checkpoint"),
            SimpleNamespace(type="function_call", id="fc_kept", call_id="call_kept", name="skill_view",
                            arguments=json.dumps({"name": "kept-skill"}))],
            usage=usage, status="completed", model="gpt-5.6-sol"),
        SimpleNamespace(output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="done")])],
                        usage=usage, status="completed", model="gpt-5.6-sol"),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))
    return agent


def _chat_turn(served, rewrite):
    """Chat Completions route whose post-tool prune or post-turn micro-compaction cuts lost-skill's row."""
    with patch("agent.process_bootstrap.OpenAI"):
        from run_agent import AIAgent
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=10)
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.compression_enabled = True
    compressor = MagicMock()  # never demands full compaction, so the post-tool prune arm runs
    compressor.protect_first_n, compressor.protect_last_n = 3, 20
    compressor.threshold_tokens, compressor.context_length = 500_000, 1_000_000
    compressor.last_prompt_tokens = 120_000
    compressor.awaiting_real_usage_after_compression = False
    compressor.should_compress.return_value = False
    compressor.should_compress_info.return_value = (False, None)
    compressor.should_defer_preflight_to_real_usage.return_value = True
    compressor.get_active_compression_failure_cooldown.return_value = None
    cut_row = _cut(served["lost-skill"][1]["content"])

    def _prune(messages, current_tokens=None):
        if rewrite != "proactive_prune" or any(m.get("content") == cut_row for m in messages):
            return messages, 0
        return _cut_lost(messages, served), 1

    compressor.prune_tool_results_only = _prune
    # Post-turn micro-compaction (finalizer) returns a new list when it splices.
    compressor._micro_compact_enabled = rewrite == "micro_compaction"
    compressor._micro_compact = lambda messages: _cut_lost(messages, served)
    compressor._flush_scan_cursor_invalidated = False
    agent.context_compressor = compressor
    tool_call = SimpleNamespace(id="call_web", type="function",
                                function=SimpleNamespace(name="web_search", arguments='{"query": "x"}'))
    agent.client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=None, reasoning_content=None, reasoning=None, tool_calls=[tool_call]),
            finish_reason="tool_calls")], model="test/model", usage=None),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="done", reasoning_content=None, reasoning=None, tool_calls=None),
            finish_reason="stop")], model="test/model", usage=None),
    ]
    return agent


@pytest.mark.parametrize("rewrite", ["proactive_prune", "micro_compaction", "native_checkpoint"])
def test_in_turn_rewrites_keep_the_stub_only_for_a_body_they_kept(served, rewrite, monkeypatch):
    tool_defs = [{"type": "function", "function": {"name": n, "description": n,
                                                   "parameters": {"type": "object", "properties": {}}}}
                 for n in ("web_search", "skill_view")]
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **kw: tool_defs)
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *a, **kw: {})
    native = rewrite == "native_checkpoint"
    agent = _native_checkpoint_turn(served, monkeypatch) if native else _chat_turn(served, rewrite)
    agent.tool_delay = 0
    agent.save_trajectories = False
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("model_tools.handle_function_call", _dispatch),
    ):
        result = agent.run_conversation("go on", conversation_history=_history(served), task_id=TASK)

    assert result["completed"] is True
    if native:  # the re-view right after the checkpoint already had to reload
        rows = [m for m in result["messages"] if m.get("role") == "tool" and "kept-skill" in m.get("content", "")]
        assert any(has_compaction_checkpoint(m.get("codex_reasoning_items")) for m in result["messages"])
        assert "Do the kept-skill steps." in rows[-1]["content"]
    else:
        cut_row = _cut(served["lost-skill"][1]["content"])
        assert any(m.get("content") == cut_row for m in result["messages"]), f"the {rewrite} was never committed"
    _assert_only_the_surviving_body_is_stubbed()
