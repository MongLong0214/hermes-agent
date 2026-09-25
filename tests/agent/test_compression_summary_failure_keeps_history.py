"""A failed summary keeps the transcript unless the user opted into the lossy fallback.

Under the default config (key absent or ``null``) a summary failure must not replace the middle of the
conversation with the deterministic "summary unavailable" handoff: compression aborts, nothing is dropped, and
the user is told. Writing ``compression.abort_on_summary_failure: false`` still opts into that fallback. Real
AIAgent against the temp ``HERMES_HOME`` + real ``compress_context``; only ``call_llm`` is stubbed.

A kept session must not accumulate the exhausted turns themselves: each one's turn-start user row would be
merged into the next ask and replayed on every later request (#107070).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent.context_compressor import SUMMARY_PREFIX
from hermes_state import SessionDB

SESSION_ID = "SUMMARY_FAILURE_KEEPS_HISTORY"


def _make_agent(tmp_path):
    db = SessionDB(db_path=Path(tmp_path) / "state.db")
    db.create_session(SESSION_ID, source="cli")
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
            session_db=db, session_id=SESSION_ID, skip_context_files=True, skip_memory=True,
        )
    agent._compression_feasibility_checked = True
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    agent.context_compressor.threshold_tokens = 1_000
    return agent


def _transcript():
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + ("lorem ipsum " * 300)}
        for i in range(40)
    ]


@pytest.mark.parametrize(
    "user_compression_cfg, fallback_opted_in",
    [(None, False), ({"abort_on_summary_failure": None}, False), ({"abort_on_summary_failure": False}, True)],
    ids=["key-absent", "explicit-null", "explicit-opt-in"],
)
def test_summary_failure_drops_history_only_when_the_user_opted_in(
    tmp_path, monkeypatch, user_compression_cfg, fallback_opted_in,
):
    if user_compression_cfg is not None:
        config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
        config_path.write_text(yaml.safe_dump({"compression": user_compression_cfg}), encoding="utf-8")
    agent = _make_agent(tmp_path)
    warnings = []
    monkeypatch.setattr(agent, "_emit_warning", warnings.append)
    live = _transcript()
    before = copy.deepcopy(live)

    with patch("agent.context_compressor.call_llm", side_effect=Exception("summary backend returned HTTP 500")), \
            patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": []}):
        out, _ = agent._compress_context(live, "sys", approx_tokens=50_000)

    summary_rows = [m for m in out if isinstance(m.get("content"), str) and m["content"].startswith(SUMMARY_PREFIX)]
    if fallback_opted_in:
        assert len(out) < len(before) and len(summary_rows) == 1, "the explicit opt-in keeps the lossy fallback"
        return
    assert out == before and live == before, "a failed summary must leave the conversation exactly as it was"
    assert summary_rows == []
    assert any("/compress" in w and "/new" in w for w in warnings), "the user is told compression failed"


def _ok_response(**_kwargs):
    msg = SimpleNamespace(content="normal recovery", tool_calls=None, reasoning_content=None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model="test/model", usage=None)


def test_exhausted_turns_leave_no_user_row_for_the_next_request(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path)
    db = agent._session_db
    for message in _transcript()[:10]:
        db.append_message(SESSION_ID, message["role"], message["content"])
    agent.context_compressor.abort_on_summary_failure = True  # the default here; pinned so the premise holds anywhere
    agent.context_compressor.context_length, agent.context_compressor.threshold_tokens = 200_000, 100_000
    agent.client = MagicMock()
    agent._use_prompt_caching, agent._disable_streaming, agent.save_trajectories = False, True, False
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 0.0)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(agent, "_save_trajectory", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_cleanup_task_resources", lambda *a, **k: None)
    overflow = Exception("Error code: 400 - prompt is too long: 233153 tokens > 200000 maximum")
    overflow.status_code = 400

    def _run(text, tokens):
        with patch("agent.turn_context.estimate_request_tokens_rough", return_value=tokens), \
                patch("agent.conversation_loop.estimate_request_tokens_rough", return_value=tokens), \
                patch("agent.conversation_loop.estimate_messages_tokens_rough", return_value=tokens):
            return agent.run_conversation(text, conversation_history=db.get_messages_as_conversation(SESSION_ID))

    agent.client.chat.completions.create.side_effect = overflow
    with patch("agent.context_compressor.call_llm", side_effect=RuntimeError("summary backend returned HTTP 500")), \
            patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": []}):
        for n in range(3):
            assert _run(f"oversized request {n}", 250_000).get("compression_exhausted"), "premise: the turn exhausts"
    agent.client.chat.completions.create.side_effect = _ok_response
    assert _run("normal request", 100).get("completed") is True

    sent = [m for m in agent.client.chat.completions.create.call_args.kwargs["messages"] if m.get("role") != "system"]
    roles = [m["role"] for m in sent]
    assert all(a != b for a, b in zip(roles, roles[1:])), f"same-role run in the next request: {roles}"
    assert not [m for m in sent if "oversized request" in str(m.get("content"))], "a failed turn's text is replayed"
    assert sent[-1]["content"] == "normal request"
