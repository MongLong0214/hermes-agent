"""Native compaction gets the first attempt when a request crosses the local trigger.

Ported from the fork's native preflight grace (949ce03117, request-local failures 7e68291e9a).
Real ``AIAgent`` on a native-compaction route (gpt-5.6 @ api.openai.com, codex_responses) with the
real loop, preflight gates, transport and ``run_codex_stream``; only the Responses wire, the
summary LLM, tool execution and the retry backoff length are stubbed.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import httpx
import openai

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


def test_failed_or_interrupted_native_attempt_never_resends_its_payload(tmp_path, monkeypatch):
    # The request after the tool round dies before any stream event (nothing billed): what goes
    # out next must be a rebuilt request under the trigger, never that payload again.
    server = _Server(["tool", "zero_event", "text", "tool", "text"])
    agent = _native_agent(tmp_path / "failed", monkeypatch, server, [])
    first = agent.run_conversation("run the log tool", conversation_history=_history(30_000))

    failed, later = server.wire[1], server.wire[2:]
    assert later and failed["payload"] not in {w["payload"] for w in later}
    assert all(w["billed"] < TRIGGER for w in later)

    # The fallback is that turn's alone: the next turn's crossing gets its native attempt.
    sent = len(server.wire)
    agent.run_conversation("run it again", conversation_history=first["messages"])
    crossing = [w for w in server.wire[sent:] if w["billed"] >= TRIGGER]
    assert crossing and all(w["native"] and w["compacts"] for w in crossing)

    # Rate-limited with the summary provider down: the attempt is retried under the turn's
    # own retry budget, never re-armed around it.
    server = _Server(["tool"] + ["429"] * 8)
    agent = _native_agent(tmp_path / "limited", monkeypatch, server, [], summary_fails=True)
    agent.run_conversation("run the log tool", conversation_history=_history(30_000))

    limited = [w for w in server.wire if w["action"] == "429"]
    assert limited and len(limited) <= agent._api_max_retries

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
