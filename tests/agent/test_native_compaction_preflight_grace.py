"""Native compaction gets the first attempt when a request crosses the local trigger.

Ported from the fork's native preflight grace (949ce03117, request-local failures 7e68291e9a).
Real ``AIAgent`` on a native-compaction route (gpt-5.6 @ api.openai.com, codex_responses) with the
real loop, preflight gates, transport and ``run_codex_stream``; only the Responses wire, the
summary LLM, tool execution and the retry backoff length are stubbed. The wire-boundary and
reconnect halves of the grace are covered in ``test_native_compaction.py`` and
``test_native_compaction_summary_retention.py``, which reuse this harness.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from agent import native_compaction_grace as grace
from agent.turn_preflight import (
    PreflightGateVerdict, compress_after_tool_results, run_preflight_compression,
)

CONTEXT, TRIGGER = 100_000, 50_000
CHARS_PER_TOKEN = 4
_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/responses")


def _tokens(obj) -> int:
    return len(json.dumps(obj, default=str)) // CHARS_PER_TOKEN


class _Server:
    """Honours ``context_management``: compacts a checkpoint-free input over the threshold and
    bills a replayed checkpoint as its tail only. ``plan``: "tool" | "text" | "zero_event" | "429"."""

    def __init__(self, plan):
        self.plan, self.wire = list(plan), []

    def create(self, **kw):
        kw = {**kw, **(kw.get("extra_body") or {})}
        items = kw.get("input") or []
        compact_at = next((c["compact_threshold"] for c in kw.get("context_management") or []
                           if c.get("type") == "compaction"), None)
        cps = [i for i, it in enumerate(items) if isinstance(it, dict) and it.get("type") == "compaction"]
        billed = 3_000 + _tokens(items[cps[-1] + 1:]) if cps else _tokens(items) + _tokens(kw.get("instructions"))
        action = self.plan.pop(0) if self.plan else "text"
        compacts = compact_at is not None and not cps and billed >= compact_at and action in ("tool", "text")
        self.wire.append({
            "action": action, "billed": billed, "native": compact_at is not None, "compacts": compacts,
            "replays": bool(cps),
            "payload": hashlib.sha256(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest(),
        })
        if action == "429":
            raise openai.RateLimitError("Rate limit reached for gpt-5.6", response=httpx.Response(429, request=_REQUEST),
                                        body={"error": {"type": "rate_limit_exceeded"}})
        if action == "zero_event":
            def _dead():
                raise httpx.RemoteProtocolError("peer closed connection without response", request=_REQUEST)
                yield  # pragma: no cover

            return _dead()
        n = len(self.wire)
        events = []
        if compacts:
            events.append(SimpleNamespace(type="response.output_item.done", item=SimpleNamespace(
                type="compaction", id=f"cmp_{n}", encrypted_content=f"sealed-{n}")))
        if action == "tool":
            events.append(SimpleNamespace(type="response.output_item.done", item=SimpleNamespace(
                type="function_call", id=f"fc_{n}", call_id=f"call_{n}", name="terminal",
                arguments='{"command": "cat big.log"}', status="completed")))
        else:
            events.append(SimpleNamespace(type="response.output_item.done", item=SimpleNamespace(
                type="message", role="assistant", status="completed", id=f"msg_{n}",
                content=[SimpleNamespace(type="output_text", text=f"answer {n}")])))
        events.append(SimpleNamespace(type="response.completed", response=SimpleNamespace(
            id=f"resp_{n}", status="completed",
            usage=SimpleNamespace(input_tokens=billed, output_tokens=20, total_tokens=billed + 20))))
        return iter(events)


def _native_agent(home, monkeypatch, server, summaries, *, summary_fails=False):
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / ".env").write_text("", encoding="utf-8")
    (home / "config.yaml").write_text("compression:\n  codex_responses_native: true\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model="gpt-5.6", provider="openai", api_key="sk-dummy", base_url="https://api.openai.com/v1",
        api_mode="codex_responses", quiet_mode=True, skip_context_files=True, skip_memory=True,
        platform="cli",
    )
    assert agent.codex_responses_native_compaction is True
    agent.context_compressor.context_length = CONTEXT
    agent.context_compressor.threshold_tokens = TRIGGER
    client = SimpleNamespace(responses=SimpleNamespace(create=server.create))
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **_k: client)
    monkeypatch.setattr(agent, "_ensure_primary_openai_client", lambda **_k: client)
    monkeypatch.setattr(agent, "_emit_status", lambda *a, **k: None)
    monkeypatch.setattr("agent.turn_api_error.compute_error_backoff", lambda *a, **k: 0.01)

    def _summary(**kwargs):
        summaries.append(kwargs)
        if summary_fails:
            raise RuntimeError("summary provider unavailable")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="## Goal\nstub summary", reasoning_content=None, reasoning=None, tool_calls=None),
            finish_reason="stop")], usage=None, model="stub-summary")

    monkeypatch.setattr("agent.context_compressor.call_llm", _summary)

    def _run_tools(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls:  # one 25K-token result pushes the next request over
            messages.append({"role": "tool", "tool_call_id": call.id, "name": "terminal",
                             "content": "L" * (25_000 * CHARS_PER_TOKEN)})

    monkeypatch.setattr(agent, "_execute_tool_calls", _run_tools)
    return agent


def _history(tokens):
    rows = []
    for i in range(tokens // 2_000):
        rows += [{"role": "user", "content": f"q{i} " + "u" * 4_000},
                 {"role": "assistant", "content": f"a{i} " + "a" * 4_000}]
    return rows


def test_trigger_crossing_is_compacted_natively_before_any_local_summary(tmp_path, monkeypatch):
    server, summaries = _Server(["tool", "text", "text"]), []
    agent = _native_agent(tmp_path / "crossing", monkeypatch, server, summaries)

    first = agent.run_conversation("run the log tool", conversation_history=_history(30_000))
    agent.run_conversation("and now?", conversation_history=first["messages"])

    crossing = [w for w in server.wire if w["billed"] >= TRIGGER]
    assert summaries == []
    assert crossing and all(w["native"] and w["compacts"] for w in crossing)
    replay = server.wire[-1]
    assert replay["replays"] and replay["billed"] < min(w["billed"] for w in crossing)

    # A crossing made by the user's own ask belongs to the turn-start guard: when local
    # compaction cannot fit it, the turn is refused before any wire and the history is kept.
    server = _Server(["text", "text"])
    agent = _native_agent(tmp_path / "big_ask", monkeypatch, server, [])
    first = agent.run_conversation("hi", conversation_history=_history(30_000))
    sent = len(server.wire)
    refused = agent.run_conversation("Q " + "x" * (46_000 * CHARS_PER_TOKEN), conversation_history=first["messages"])

    assert len(server.wire) == sent and refused["compression_exhausted"]
    assert len(refused["messages"]) == len(first["messages"])


def test_a_stop_before_the_attempt_leaves_the_next_turn_local(tmp_path, monkeypatch):
    # A /stop between the tool round and the wire: the attempt had taken the crossing (no
    # summary yet), and the next turn compacts locally instead of inheriting it.
    server, summaries = _Server(["tool", "text", "text"]), []
    agent = _native_agent(tmp_path / "interrupted", monkeypatch, server, summaries)
    real_call, stopped = agent._interruptible_streaming_api_call, []

    def _stop_before_second_wire(api_kwargs, **kw):
        if not stopped and len(server.wire) == 1:
            stopped.append(True)
            agent.interrupt("/stop")
            raise InterruptedError()
        return real_call(api_kwargs, **kw)

    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", _stop_before_second_wire)
    first = agent.run_conversation("run the log tool", conversation_history=_history(30_000))
    assert stopped and summaries == []
    agent.clear_interrupt()
    agent.run_conversation("again", conversation_history=first["messages"])

    assert len(server.wire) >= 2 and all(w["billed"] < TRIGGER for w in server.wire)


def test_a_spent_grace_is_that_turns_alone(tmp_path, monkeypatch):
    # A turn whose attempt fell back keeps the grace spent; the next turn's crossing gets its
    # native attempt again, with no local summary.
    server, summaries = _Server(["text", "tool", "text"]), []
    agent = _native_agent(tmp_path / "spent", monkeypatch, server, summaries)
    first = agent.run_conversation("hi", conversation_history=_history(30_000))
    setattr(agent, grace._STATE_ATTR, {"phase": grace._SPENT, "identity": grace._identity(agent)})
    sent = len(server.wire)

    agent.run_conversation("run the log tool", conversation_history=first["messages"])

    crossing = [w for w in server.wire[sent:] if w["billed"] >= TRIGGER]
    assert summaries == []
    assert crossing and all(w["native"] and w["compacts"] for w in crossing)


def _armed(home, monkeypatch):
    agent = _native_agent(home, monkeypatch, _Server([]), [])
    agent.context_compressor.last_real_prompt_tokens = TRIGGER - 10_000
    assert grace.defer_local_preflight_for_native_compaction(agent, TRIGGER + 8_000, messages=[])
    return agent


def test_the_grace_arms_only_on_the_crossing_itself(tmp_path, monkeypatch):
    agent = _native_agent(tmp_path / "unit", monkeypatch, _Server([]), [])
    compressor = agent.context_compressor
    compressor.last_real_prompt_tokens = TRIGGER - 10_000
    # Under the trigger: nothing to defer.
    assert not grace.defer_local_preflight_for_native_compaction(agent, TRIGGER - 1, messages=[])
    # The last real prompt was already over: not a crossing.
    compressor.last_real_prompt_tokens = TRIGGER + 1
    assert not grace.defer_local_preflight_for_native_compaction(agent, TRIGGER + 8_000, messages=[])
    # No headroom for the output budget: the server could not take the request either.
    compressor.last_real_prompt_tokens = TRIGGER - 10_000
    assert not grace.defer_local_preflight_for_native_compaction(agent, CONTEXT, messages=[])
    # A history that already carries a checkpoint replays it instead.
    checkpoint = [{"role": "assistant", "content": "", "codex_reasoning_items": [
        {"type": "compaction", "encrypted_content": "sealed"}]}]
    assert not grace.defer_local_preflight_for_native_compaction(agent, TRIGGER + 8_000, messages=checkpoint)
    # A route without the native field never arms.
    agent.codex_responses_native_compaction = False
    assert not grace.defer_local_preflight_for_native_compaction(agent, TRIGGER + 8_000, messages=[])


@pytest.mark.parametrize("change", ["session_id", "model", "base_url"])
def test_a_session_or_route_change_clears_the_grace(tmp_path, monkeypatch, change):
    agent = _armed(tmp_path / "unit", monkeypatch)
    setattr(agent, change, {"session_id": "s-rotated", "model": "gpt-5.5",
                            "base_url": "https://proxy.example/v1"}[change])

    assert not grace.defer_local_preflight_for_native_compaction(agent, TRIGGER + 8_000, messages=[])
    # The old chance falls back once and stays spent for the turn.
    assert grace.consume_native_compaction_preflight_fallback(agent)
    assert not grace.consume_native_compaction_preflight_fallback(agent)
    assert not grace.defer_local_preflight_for_native_compaction(agent, TRIGGER + 8_000, messages=[])
    # A new turn starts clean.
    grace.begin_native_compaction_turn(agent)
    assert getattr(agent, grace._STATE_ATTR) is None


def test_turn_start_never_arms_and_keeps_only_a_pending_replay(tmp_path, monkeypatch):
    agent = _native_agent(tmp_path / "unit", monkeypatch, _Server([]), [])
    agent.context_compressor.last_real_prompt_tokens = TRIGGER - 10_000
    assert not grace.defer_turn_start_preflight_for_native_replay(agent, TRIGGER + 8_000, messages=[])
    assert getattr(agent, grace._STATE_ATTR, None) is None

    agent = _armed(tmp_path / "armed", monkeypatch)
    grace.begin_native_compaction_turn(agent)
    assert getattr(agent, grace._STATE_ATTR) is None

    agent = _armed(tmp_path / "replay", monkeypatch)
    getattr(agent, grace._STATE_ATTR)["phase"] = grace._AWAITING_REPLAY
    grace.begin_native_compaction_turn(agent)
    assert getattr(agent, grace._STATE_ATTR)["phase"] == grace._AWAITING_REPLAY
    # The replay cannot go out (no checkpoint on this route's wire): dropped, turn start runs as before.
    assert not grace.defer_turn_start_preflight_for_native_replay(agent, TRIGGER + 8_000, messages=[])
    assert getattr(agent, grace._STATE_ATTR) is None


def _messages() -> list:
    return [
        {"role": "user", "content": "x" * 8_000},
        {"role": "assistant", "content": "y" * 8_000},
        {"role": "user", "content": "current ask"},
    ]


def _double(*, lock: bool):
    """A minimal agent whose compression pass no-ops on the lock or a transient guard."""
    def _noop(messages, system, **_kwargs):
        if lock:
            agent._compression_skipped_due_to_lock = "other-path"
        else:
            agent._compression_blocked_transient = "cooldown:42"
        return messages, system

    compressor = SimpleNamespace(
        threshold_tokens=1_000, context_length=200_000, summary_target_ratio=0.2,
        protect_first_n=1, protect_last_n=1, last_prompt_tokens=5_000,
        last_compression_rough_tokens=0, awaiting_real_usage_after_compression=False,
        should_compress=lambda _tokens: True,
    )
    persisted, refunds = [], []
    agent = SimpleNamespace(
        context_compressor=compressor, compression_enabled=True, model="m", session_id="s1",
        max_compression_attempts=3, _compression_feasibility_checked=True, log_prefix="",
        _emit_status=lambda _msg: None, _safe_print=lambda *_a: None, _compress_context=_noop,
        _compression_skipped_due_to_lock=None, _compression_blocked_transient=None,
        _persist_session=lambda *a: persisted.append(a), _flush_status_buffer=lambda: None,
        iteration_budget=SimpleNamespace(refund=lambda: refunds.append(1)),
        persisted=persisted, refunds=refunds,
    )
    return agent


@pytest.mark.parametrize("lock", [True, False], ids=["lock", "transient"])
def test_a_blocked_forced_preflight_ends_deferred_not_exhausted(lock):
    agent = _double(lock=lock)
    verdict = PreflightGateVerdict(
        action="fallthrough", pending_moa_prepared_request=None, messages=_messages(),
        active_system_prompt="sys", conversation_history=None, api_call_count=2,
        compression_attempts=0, final_response=None, failed=False, _turn_exit_reason=None,
        _compression_timeout_exhausted=False, _preflight_compression_blocked=False,
        _provider_overflow_recovery_pending=True, _last_preflight_pressure=None,
    )
    out = run_preflight_compression(
        agent, verdict, compressor=agent.context_compressor, request_pressure_tokens=5_000,
        provider_overflow_preflight=True, defer_preflight=lambda _tokens: False,
        moa_prepared_request=None, system_message="sys", user_message="current ask",
        max_compression_attempts=3, effective_task_id="task",
    )

    assert out.action == "return"
    assert out.result["compression_deferred"] is True and not out.result.get("compression_exhausted")
    assert out.result["api_calls"] == 1 and agent.refunds == [1] and agent.persisted
    assert out.compression_attempts == 0


@pytest.mark.parametrize("lock", [True, False], ids=["lock", "transient"])
def test_a_blocked_post_tool_pass_refunds_its_attempt(lock):
    agent = _double(lock=lock)
    messages = _messages()
    out = compress_after_tool_results(
        agent, messages=messages, system_message="sys", user_message="current ask",
        active_system_prompt="sys", conversation_history=None, compression_attempts=0,
        max_compression_attempts=3, effective_task_id="task", final_response=None,
        turn_exit_reason=None, current_turn_user_idx=len(messages) - 1,
    )

    assert out.compression_attempts == 0 and out.end_turn is False
