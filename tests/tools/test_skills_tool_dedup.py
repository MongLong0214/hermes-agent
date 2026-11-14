"""skill_view dedup survives a transcript rewrite only while the served text is still on the wire.

Post-turn micro-compaction and a native Responses compaction checkpoint can take a served skill body off
the next request. After either, a repeat view of that skill must serve the full text again, not a "see
above" stub. A skill whose tool row the next request still carries keeps its dedup; only the bodies that
left the wire are dropped. File-read dedup still starts a new generation at a micro-compaction splice.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.file_tools_read_tracking import _read_tracker, _read_tracker_lock, _task_data
from tools.skills_tool import _skill_view_with_bump
from tools.skills_tool_dedup import _skill_view_tracker, dedup_skill_view, reset_skill_view_dedup

TASK = "skill-dedup-wire-task"
FILE_KEY = ("/x/big.txt", 1, 2000)
STEPS = "Do the demo steps."


@pytest.fixture
def skills_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    for name in ("demo-skill", "other-skill"):
        d = home / "skills" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Demo.\n---\n# {name}\n\n{STEPS}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    reset_skill_view_dedup()
    with _read_tracker_lock:
        _read_tracker.pop(TASK, None)
    yield home
    reset_skill_view_dedup()
    with _read_tracker_lock:
        _read_tracker.pop(TASK, None)


def _view(name="demo-skill", task=TASK):
    return _skill_view_with_bump({"name": name}, task_id=task)


def _stubbed(text):
    parsed = json.loads(text)
    return parsed.get("dedup") is True and "content" not in parsed


def _exchange(name, call_id, body, answer):
    call = {"id": call_id, "type": "function",
            "function": {"name": "skill_view", "arguments": json.dumps({"name": name})}}
    return [{"role": "user", "content": f"load {name}"}, {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call_id, "content": body}, {"role": "assistant", "content": answer}]


@pytest.fixture
def history(skills_home):
    """Two exchanges, each serving one real skill to TASK, and one file read served this generation."""
    old = _exchange("other-skill", "call_old", _view("other-skill"), "loaded other")
    new = _exchange("demo-skill", "call_new", _view("demo-skill"), "loaded demo")
    with _read_tracker_lock:
        td = _task_data(TASK)
        td["dedup"][FILE_KEY] = 1.0
        td["dedup_generation_reads"].add(FILE_KEY)
    assert _stubbed(_view("other-skill")) and _stubbed(_view("demo-skill"))  # reach: both dedup now
    return old + new


def _file_stubbed():
    with _read_tracker_lock:
        return FILE_KEY in _read_tracker.get(TASK, {}).get("dedup_generation_reads", ())


def _agent_with_micro_pass(micro_compact):
    with patch("agent.process_bootstrap.OpenAI"):
        from run_agent import AIAgent
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=4)
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.save_trajectories = False
    compressor = MagicMock()  # never asks for full compaction; only the post-turn micro pass runs
    compressor.protect_first_n, compressor.protect_last_n = 3, 20
    compressor.threshold_tokens, compressor.context_length = 500_000, 1_000_000
    compressor.last_prompt_tokens = 1_000
    compressor.awaiting_real_usage_after_compression = False
    compressor.should_compress.return_value = False
    compressor.should_compress_info.return_value = (False, None)
    compressor.should_defer_preflight_to_real_usage.return_value = True
    compressor.get_active_compression_failure_cooldown.return_value = None
    compressor._micro_compact_enabled = True
    compressor._flush_scan_cursor_invalidated = False
    compressor._micro_compact = micro_compact
    agent.context_compressor = compressor
    agent.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="done", reasoning_content=None, reasoning=None, tool_calls=None), finish_reason="stop")],
        model="test/model", usage=None)
    return agent


def _run(agent, history, monkeypatch):
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **kw: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *a, **kw: {})
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("go on", conversation_history=history, task_id=TASK)
    assert result["completed"] is True
    return result


def _micro(rewrite):
    """A splice: a new list with ``rewrite`` applied, opened by a rolling-summary row."""
    return lambda messages, **_kw: [{"role": "user", "content": "[MICRO] earlier work"}, *rewrite(messages)]


def _drop_old_exchange(messages):
    return [m for m in messages if m.get("tool_call_id") != "call_old" and m.get("content") not in (
        "load other-skill", "loaded other") and not any(
        tc.get("id") == "call_old" for tc in m.get("tool_calls") or ())]


# ── micro-compaction (the real post-turn pass) ──────────────────────────────────────────────────────


def test_micro_compaction_that_removes_a_skills_row_serves_its_full_text_again(history, monkeypatch):
    result = _run(_agent_with_micro_pass(_micro(_drop_old_exchange)), history, monkeypatch)
    assert not any(m.get("tool_call_id") == "call_old" for m in result["messages"])  # reach: row gone

    again = _view("other-skill")
    assert not _stubbed(again) and STEPS in json.loads(again)["content"]


def test_a_skill_row_still_on_the_wire_keeps_its_dedup_after_micro_compaction(history, monkeypatch):
    result = _run(_agent_with_micro_pass(_micro(_drop_old_exchange)), history, monkeypatch)
    assert any(m.get("tool_call_id") == "call_new" for m in result["messages"])  # reach: row kept

    assert _stubbed(_view("demo-skill"))
    assert _file_stubbed() is False  # file reads still start a new generation at the splice


def test_a_kept_row_whose_body_was_rewritten_serves_the_full_text_again(history, monkeypatch):
    def demote(messages):
        return [{**m, "content": "[skill_view output demoted]"} if m.get("tool_call_id") == "call_new" else m
                for m in messages]

    _run(_agent_with_micro_pass(_micro(demote)), history, monkeypatch)

    assert not _stubbed(_view("demo-skill"))
    assert _stubbed(_view("other-skill"))  # its row is untouched


def test_control_a_no_op_micro_pass_keeps_every_dedup(history, monkeypatch):
    _run(_agent_with_micro_pass(lambda messages, **_kw: messages), history, monkeypatch)

    assert _stubbed(_view("demo-skill")) and _stubbed(_view("other-skill")) and _file_stubbed()


# ── the native compaction checkpoint ────────────────────────────────────────────────────────────────


def test_native_checkpoint_serves_the_full_text_again(history, monkeypatch):
    """The response carrying a server checkpoint takes every earlier item off the next request."""
    from run_agent import AIAgent
    monkeypatch.setattr("agent.model_metadata._fetch_codex_oauth_context_lengths_with_source",
                        lambda _t, **_kw: ({}, False))
    agent = AIAgent(model="gpt-5.6", base_url="https://chatgpt.com/backend-api/codex", api_key="codex-token",
                    quiet_mode=True, skip_context_files=True, skip_memory=True, max_iterations=4)
    agent.codex_responses_native_compaction = True
    agent.runtime_capabilities = {"native_compaction": True}
    agent.tool_delay = 0
    agent.save_trajectories = False
    usage = SimpleNamespace(input_tokens=9, output_tokens=1, total_tokens=10)
    responses = [
        SimpleNamespace(output=[SimpleNamespace(type="compaction", encrypted_content="opaque-checkpoint"),
                                SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text",
                                                                                         text="done")])],
                        usage=usage, status="completed", model="gpt-5.6"),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    result = _run(agent, history, monkeypatch)
    assert not responses and any(m.get("codex_reasoning_items") for m in result["messages"])  # reach

    for name in ("demo-skill", "other-skill"):
        again = _view(name)
        assert not _stubbed(again) and STEPS in json.loads(again)["content"]


# ── the registry ─────────────────────────────────────────────────────────────────────────────────────


def _record(task, name, body):
    return dedup_skill_view(task, name, None, {"name": name}, body)


def _tool(body):
    return {"role": "tool", "tool_call_id": "c", "content": body}


def _checkpoint_carrier():
    return {"role": "assistant", "content": "",
            "codex_reasoning_items": [{"type": "compaction", "encrypted_content": "opaque"}]}


def test_only_rows_after_the_newest_checkpoint_count_as_on_the_wire(skills_home):
    from tools.skills_tool_dedup import drop_lost_skill_views

    _record(TASK, "a", '{"success": true, "content": "# a"}')
    _record(TASK, "b", '{"success": true, "content": "# b"}')
    drop_lost_skill_views(TASK, [_tool('{"success": true, "content": "# a"}'), _checkpoint_carrier(),
                                 _tool('{"success": true, "content": "# b"} [hint appended later]')])

    assert set(_skill_view_tracker.get(TASK, {})) == {("b", "")}  # a verbatim prefix proves the body


@pytest.mark.parametrize("transcript", [[], None], ids=["empty", "unreadable"])
def test_an_unproven_transcript_drops_the_whole_bucket(skills_home, transcript):
    from tools.skills_tool_dedup import drop_lost_skill_views

    _record(TASK, "a", '{"success": true, "content": "# a"}')
    drop_lost_skill_views(TASK, transcript)

    assert TASK not in _skill_view_tracker


def test_task_buckets_are_capped_least_recently_used_first(skills_home):
    from tools.skills_tool_dedup import _SKILL_VIEW_TASK_CAP

    for i in range(_SKILL_VIEW_TASK_CAP):
        _record(f"task-{i}", "a", '{"success": true, "content": "# a"}')
    _record("task-0", "b", '{"success": true, "content": "# b"}')  # task-0 used again: now the newest
    _record("task-new", "a", '{"success": true, "content": "# a"}')

    assert len(_skill_view_tracker) == _SKILL_VIEW_TASK_CAP
    assert "task-1" not in _skill_view_tracker and "task-0" in _skill_view_tracker
    assert "task-new" in _skill_view_tracker


def test_dedup_still_compares_the_exact_served_text(skills_home):
    body = '{"success": true, "content": "# a"}'
    assert _record(TASK, "a", body) is None
    assert _record(TASK, "a", body) is not None
    assert _record(TASK, "a", body.replace("# a", "# A")) is None  # same length, different text
