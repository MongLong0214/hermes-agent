"""One tool-calling round of the conversation turn loop: validate/cap/dedupe the model's
tool calls, persist the tool-call turn BEFORE any side effect, execute the tools, honour
guardrail halts / persistence failures, then compress after tool results. Nothing here
imports ``agent.conversation_loop`` at module level (cycle) — loop-internal helpers resolve
lazily.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple

from agent.message_metadata import append_message
from agent.message_sanitization import coalesce_tool_call_id, normalize_provider_tool_call_ids
from agent.turn_preflight import compress_after_tool_results
from agent.turn_tool_validation import validate_tool_calls

logger = logging.getLogger("agent.conversation_loop")

# Post-response housekeeping tools: a round made only of these mutes tool progress.
_HOUSEKEEPING_TOOLS = frozenset({"memory", "todo_list", "skill_manage", "session_search"})
# Rounds whose visible line is the turn's reply if the follow-up is empty (no empty-response nudge):
# housekeeping, and setup_beat, whose card already blocked for the user's answer.
_LINE_IS_REPLY_TOOLS = _HOUSEKEEPING_TOOLS | {"setup_beat"}


@dataclass
class ToolRoundVerdict:
    """``action``: ``"continue"`` (tools ran, next API call), ``"break"`` (turn ends:
    persistence failure, guardrail halt, post-tool compression end) or ``"return"``
    (``result`` is the turn's result dict). The other fields are the loop locals the round
    rebinds."""

    action: str
    messages: Any
    conversation_history: Any
    active_system_prompt: Any
    compression_attempts: Any
    final_response: Any
    failed: Any
    _turn_exit_reason: Any
    truncated_tool_call_retries: Any
    current_turn_user_idx: Any
    result: Optional[dict[str, Any]] = None


def _peer_refused_or_normalized(agent, assistant_message, emitted_names: list) -> bool:
    """True when a canonical peer turn may not run the batch. That is decided on the COMPLETE
    emitted batch, every name as emitted, before any call is expanded, deduped, capped or dropped as
    unknown: one unknown, repaired or disallowed call runs none of them, and every call keeps a
    matching refusal result. Otherwise the batch is normalized here."""
    if peer_batch_refused(agent, assistant_message.tool_calls, emitted_names):
        return True
    # Normalize only this unpersisted turn. Each local bridge entry must pass
    # through the same scope, approvals, state and scheduling as a singleton.
    from agent.tool_call_batches import expand_local_tool_batches
    assistant_message.tool_calls = expand_local_tool_batches(
        assistant_message.tool_calls, provider_data=getattr(assistant_message, "provider_data", None))

    # Post-call guardrails.
    assistant_message.tool_calls = agent._deduplicate_tool_calls(
        agent._cap_delegate_task_calls(assistant_message.tool_calls)
    )
    return False


def run_tool_round(
    agent: Any, *, assistant_message: Any, finish_reason: Any, messages: Any,
    conversation_history: Any, api_call_count: Any, effective_task_id: Any, user_message: Any,
    system_message: Any, active_system_prompt: Any, compression_attempts: Any,
    max_compression_attempts: Any, final_response: Any, failed: Any, _turn_exit_reason: Any,
    truncated_tool_call_retries: Any, current_turn_user_idx: Any,
) -> ToolRoundVerdict:
    """Execute one tool round in the exact original order. Persist-before-execute is a
    durability invariant: resume must see the executed block if a destructive tool restarts
    Hermes; a failed canonical append ends the turn rather than running tools from
    process-only state."""
    from agent.conversation_loop import _invalid_tool_name_error_content

    def _verdict(action: str, result: Optional[dict[str, Any]] = None) -> ToolRoundVerdict:
        return ToolRoundVerdict(
            action=action, messages=messages, conversation_history=conversation_history,
            active_system_prompt=active_system_prompt, compression_attempts=compression_attempts,
            final_response=final_response, failed=failed, _turn_exit_reason=_turn_exit_reason,
            truncated_tool_call_retries=truncated_tool_call_retries,
            current_turn_user_idx=current_turn_user_idx, result=result,
        )

    if not agent.quiet_mode:
        agent._vprint(f"{agent.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")

    if agent.verbose_logging:
        for tc in assistant_message.tool_calls:
            raw_args = tc.function.arguments
            args_preview = raw_args[:200] if isinstance(raw_args, str) else repr(raw_args)[:200]
            logging.debug("Tool call: %s with args: %s...", tc.function.name, args_preview)

    emitted_names = [tc.function.name for tc in assistant_message.tool_calls]
    _tvv = validate_tool_calls(
        agent, assistant_message, finish_reason, messages=messages,
        conversation_history=conversation_history, api_call_count=api_call_count,
        effective_task_id=effective_task_id,
    )
    if _tvv.action == "return":
        return _verdict("return", _tvv.result)
    if _tvv.action == "continue":
        return _verdict("continue")

    peer_refused = _peer_refused_or_normalized(agent, assistant_message, emitted_names)
    # Filtering can turn a mixed batch into an all-provider one; re-check the final batch
    # before it is staged (idempotent for already-normalized ids).
    normalize_provider_tool_call_ids(assistant_message.tool_calls)

    # Mixed batch: the assistant message keeps EVERY emitted call (each tool_call needs a
    # matching result) while only valid ones dispatch.
    _invalid_batch_calls = [
        tc for tc in assistant_message.tool_calls if tc.function.name not in agent.valid_tool_names
    ] if _tvv.mixed_invalid_batch and not peer_refused else []

    assistant_msg, duplicate_previous_interim = stage_tool_call_message(
        agent, assistant_message=assistant_message, finish_reason=finish_reason, messages=messages
    )
    append_message(messages, assistant_msg)

    # Mixed batch: error-result invalid calls and drop them from execution.
    if _invalid_batch_calls:
        for tc in _invalid_batch_calls:
            append_message(messages, {
                "role": "tool",
                "name": tc.function.name,
                "tool_call_id": coalesce_tool_call_id(tc),
                "content": _invalid_tool_name_error_content(
                    tc.function.name, agent.valid_tool_names
                ),
            })
        assistant_message.tool_calls = [
            tc for tc in assistant_message.tool_calls if tc.function.name in agent.valid_tool_names
        ]

    # Persist the tool-call turn before any tool side effects so resume sees the executed
    # block if a destructive tool restarts Hermes.
    try:
        _tool_turn_persisted = agent._flush_messages_to_session_db(messages, conversation_history)
    except Exception as exc:
        _tool_turn_persisted = False
        from hermes_state import classify_persistence_error
        agent._last_persistence_error_cause = classify_persistence_error(exc)
        logger.warning(
            "Incremental tool-call persistence failed before execution "
            "(session=%s): %s",
            agent.session_id or "none",
            exc,
        )

    if _tool_turn_persisted is False:
        # Canonical append failed: never project the row or run tools from process-only
        # state; break rather than retry. No recorded cause means genuinely unknown.
        if getattr(agent, "_last_persistence_error_cause", None) is None:
            agent._last_persistence_error_cause = "unknown"
        _turn_exit_reason = "session_persistence_failed"
        final_response = ""
        failed = True
        return _verdict("break")

    # A UI must never observe an assistant/tool-call row that is only an in-memory
    # projection: emit interim commentary after the DB append.
    if not duplicate_previous_interim:
        agent._emit_interim_assistant_message(assistant_msg)

    # Flush open streaming boxes before tools so early content doesn't wrap tool feed
    # lines. Display callback only — TTS (_stream_callback) must NOT receive None (EOS).
    if agent.stream_delta_callback:
        with suppress(Exception):
            agent.stream_delta_callback(None)

    # Opt-in (compression.prepare_ahead): when this response put the session near the trigger,
    # write the next compaction summary while the tools run. No-op otherwise; never raises.
    from agent.prepared_compaction import maybe_prepare
    maybe_prepare(agent, messages, origin="tool-batch")

    if peer_refused:
        append_peer_refusals(agent, assistant_message.tool_calls, messages)
    else:
        agent._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)
    from hermes_cli.observability.shared_metrics_harness import finish_tool_round

    finish_tool_round(agent)

    if getattr(agent, "_incremental_persistence_failed", False):
        # Tool result could not be made canonical: never send the in-memory result to
        # the model or project later events from this turn.
        _turn_exit_reason = "session_persistence_failed"
        final_response = ""
        failed = True
        return _verdict("break")

    if agent._tool_guardrail_halt_decision is not None:
        decision = agent._tool_guardrail_halt_decision
        _turn_exit_reason = "guardrail_halt"
        final_response = agent._toolguard_controlled_halt_response(decision)
        agent._emit_diagnostic_status(f"⚠️ Tool guardrail halted {decision.tool_name}: {decision.code}")
        append_message(messages, {"role": "assistant", "content": final_response})
        # Emit the halt so it isn't mistaken for a crash; the stream callback is still
        # alive, so SSE/TUI clients see the explanation. A held turn shows it as the final only.
        if final_response:
            agent._safe_print(f"\n{final_response}\n")
            if agent.stream_delta_callback and not getattr(agent, "_hold_stream", False):
                with suppress(Exception):
                    agent.stream_delta_callback(final_response)
                    agent.stream_delta_callback(None)
        return _verdict("break")

    # Reset per-turn retry counters so one truncation can't poison the turn.
    truncated_tool_call_retries = 0
    # Defer the paragraph break: _fire_stream_delta() prepends one "\n\n" when real
    # text arrives, so tool iterations don't stack blank lines.
    agent._stream_needs_break = True
    # Refund the iteration when the ONLY tool was execute_code (programmatic tool
    # calling) — cheap RPC-style calls shouldn't eat the budget.
    if {tc.function.name for tc in assistant_message.tool_calls} == {"execute_code"}:
        agent.iteration_budget.refund()

    _ptc = compress_after_tool_results(
        agent, messages=messages, system_message=system_message, user_message=user_message,
        active_system_prompt=active_system_prompt, conversation_history=conversation_history,
        compression_attempts=compression_attempts,
        max_compression_attempts=max_compression_attempts, effective_task_id=effective_task_id,
        final_response=final_response, turn_exit_reason=_turn_exit_reason,
        current_turn_user_idx=current_turn_user_idx,
    )
    messages = _ptc.messages
    active_system_prompt = _ptc.active_system_prompt
    conversation_history = _ptc.conversation_history
    compression_attempts = _ptc.compression_attempts
    final_response = _ptc.final_response
    _turn_exit_reason = _ptc.turn_exit_reason
    current_turn_user_idx = _ptc.current_turn_user_idx
    if _ptc.end_turn:
        return _verdict("break")

    # Save session log incrementally (so progress is visible even if interrupted)
    agent._session_messages = messages
    # Touch activity so slow post-tool work plus a slow follow-up API call can't exceed
    # the gateway inactivity timeout (HERMES_AGENT_TIMEOUT).
    # Touch activity before continuing so the gateway's inactivity monitor never sees a stale timestamp
    # between tool completion and the start of the next API call. Without this, a tool-call result (which
    # takes ~0s to process) followed by slow post-tool processing (compression, persist) and a slow
    # follow-up API call can exceed the gateway inactivity timeout (HERMES_AGENT_TIMEOUT, default 1800s) and
    # the gateway kills the session before the next activity touch fires (#69559, #69131).
    agent._touch_activity(f"tool results posted, continuing iteration #{api_call_count}")
    return _verdict("continue")


def stage_tool_call_message(
    agent: Any, *, assistant_message: Any, finish_reason: Any, messages: Any
) -> tuple[dict[str, Any], bool]:
    """Build the assistant tool-call row and update the per-turn fallback/mute state.

    Drops a bare bracketed marker beside a call (#78148), classifies housekeeping-only
    rounds, keeps visible content as the empty-follow-up fallback, pops thinking-only
    prefills (resetting their counters), re-arms the post-tool nudge and the
    dropped-tool-call stall budget. Returns ``(assistant_msg, duplicate_previous_interim)``;
    the flag suppresses re-emitting interim commentary the previous ``incomplete`` row
    already showed."""
    from agent.conversation_loop import _STALE_MARKER_RE

    assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)

    turn_content = assistant_message.content or ""

    # A bare bracketed token (e.g. ``[memory]``) beside a function call is protocol
    # scaffolding; persisting it lets the post-tool fallback replay it forever (#78148).
    if assistant_message.tool_calls and _STALE_MARKER_RE.fullmatch(turn_content.strip()):
        logger.warning(
            "Discarding bare tool-call marker from assistant content: %s", turn_content
        )
        turn_content = ""
        assistant_msg["content"] = ""

    # Classify tools regardless of visible content: a substantive tool-only turn must
    # invalidate any older housekeeping fallback (so a two-turn-old housekeeping
    # narration isn't attributed to the preceding tool turn), and clear the mute flag a
    # prior housekeeping turn set, else _vprint suppresses this turn's tool progress.
    _all_housekeeping = all(
        tc.function.name in _HOUSEKEEPING_TOOLS for tc in assistant_message.tool_calls
    )
    # The line written with a blocking card is the reply when the follow-up comes back empty.
    _line_is_reply = all(tc.function.name in _LINE_IS_REPLY_TOOLS for tc in assistant_message.tool_calls)
    if assistant_message.tool_calls and not _all_housekeeping:
        agent._last_content_with_tools = None
        agent._last_content_tools_all_housekeeping = False
        agent._mute_post_response = False

    # Content + tool_calls in one turn: keep the content as a fallback final response in
    # case the follow-up turn after tools is empty. Mute only when EVERY tool call is
    # post-response housekeeping; substantive tools keep output on.
    if turn_content and agent._has_content_after_think_block(turn_content):
        agent._last_content_with_tools = turn_content
        agent._last_content_tools_all_housekeeping = _line_is_reply
        if _all_housekeeping and agent._has_stream_consumers():
            agent._mute_post_response = True
        elif agent._should_emit_quiet_tool_messages() and not getattr(agent, "_hold_stream", False):
            clean = agent._strip_think_blocks(turn_content).strip()
            if clean:
                agent._vprint(f"  ┊ 💬 {clean}")

    # Pop thinking-only prefill message(s) before appending (same rationale as the
    # final-response path). Tool calls after a prefill recovery reset the prefill
    # counter, so each tool-call success is a fresh start, not a cumulative burn.
    _had_prefill = False
    while messages and isinstance(messages[-1], dict) and messages[-1].get("_thinking_prefill"):
        messages.pop()
        _had_prefill = True
    if _had_prefill:
        agent._thinking_prefill_retries = 0
        agent._empty_content_retries = 0
    # Re-arm the post-tool nudge so it can fire on a LATER tool round; a landed tool call
    # recovers any dropped-tool-call stall, so refresh that budget per stall.
    agent._post_tool_empty_retried = False
    agent._dropped_toolcall_retries = 0

    previous_msg = messages[-1] if messages else None
    current_interim_visible = agent._interim_assistant_visible_text(assistant_msg)
    duplicate_previous_interim = (
        bool(current_interim_visible)
        and isinstance(previous_msg, dict)
        and previous_msg.get("role") == "assistant"
        and previous_msg.get("finish_reason") == "incomplete"
        and agent._interim_assistant_visible_text(previous_msg) == current_interim_visible
    )
    return assistant_msg, duplicate_previous_interim


# ── Canonical peer turns: whole-batch admission ─────────────────────────────────────────────
# A canonical peer turn runs a tool batch only as a whole, under the grant its verified ingress
# bound (agent.canonical_peer_tools). Authority follows the turn principal and grant, never message
# text. The executors in agent.tool_executor call admit_peer_batch before their first call.

PEER_TOOL_REFUSAL = ("[Tool call not executed — {name} was requested during a canonical peer turn. Only the "
                     "configured ACP CEO tools, with in-scope arguments, may run in a canonical peer turn, and a "
                     "batch holding any other call runs nothing. A peer has no approval authority.]")


def peer_batch_refused(agent, tool_calls, emitted_names=None) -> bool:
    """True in a canonical peer turn whose batch may not run: it runs only when EVERY call is an
    allowed call under the grant its verified ingress bound and, given ``emitted_names``, every name
    is still exactly as the model emitted it."""
    from agent.tool_executor import _canonical_tool_name, _parse_tool_arguments, _peer_turn, _tc_name

    if not _peer_turn(agent):
        return False
    if emitted_names is not None and [_tc_name(tc) for tc in tool_calls] != list(emitted_names):
        return True
    from agent.canonical_peer_tools import batch_allowed, batch_resolved

    if batch_resolved(agent, tool_calls, coalesce_tool_call_id):
        return False  # part of a batch already prepared and pinned as a whole (a segment)
    calls = []
    for tc in tool_calls:
        raw_name = _tc_name(tc)
        args, parse_error = _parse_tool_arguments(getattr(getattr(tc, "function", None), "arguments", None))
        # An alias would dispatch as a different tool than the one the policy names.
        calls.append((raw_name if _canonical_tool_name(raw_name) == raw_name else "", None if parse_error else args))
    return not batch_allowed(agent, calls)


def admit_peer_batch(agent, tool_calls, messages: list, effective_task_id: str) -> bool:
    """An executor's entry check; always True outside a canonical peer turn. In one, the WHOLE batch
    is prepared before its first call runs (:func:`_prepare_peer_batch`) and every executor then
    dispatches exactly the prepared name, handler and arguments. False, with a refusal per call and
    nothing run, when any call is refused or an emitted call changed while it was prepared."""
    from agent.canonical_peer_tools import batch_resolved
    from agent.tool_executor import _peer_turn

    if not _peer_turn(agent) or batch_resolved(agent, tool_calls, coalesce_tool_call_id):
        return True
    if not peer_batch_refused(agent, tool_calls) and _prepare_peer_batch(agent, tool_calls, effective_task_id):
        return True
    append_peer_refusals(agent, tool_calls, messages)
    return False


def _prepare_peer_batch(agent, tool_calls, effective_task_id: str) -> bool:
    """Each call's unwrap, request middleware, ``pre_tool_call`` hooks, coercion and guardrail verdict
    first, then (after any registry refresh they caused) each native handler and the policy on the
    final arguments, pinned as one batch. False (everything cleared) when any call is refused or an
    emitted call changed while it was prepared."""
    from agent.canonical_peer_tools import batch_allowed
    from agent.tool_executor import _canonical_tool_name, _parse_tool_call, _tc_name
    from tools.arg_coercion import coerce_tool_args
    from tools.tool_search import TOOL_CALL_NAME

    def as_emitted() -> list:
        return [(_tc_name(tc), getattr(getattr(tc, "function", None), "arguments", None)) for tc in tool_calls]

    before, finals = as_emitted(), []
    for tc in tool_calls:
        pc = _parse_tool_call(agent, tc, as_emitted=True)
        final = None
        if (pc.parse_error is None and pc.scope_block is None and pc.name != TOOL_CALL_NAME
                and _canonical_tool_name(_tc_name(tc)) == _tc_name(tc)):
            block, final = _peer_final_args(agent, pc.ref(effective_task_id))
            final = coerce_tool_args(pc.name, final) if block is None else None
        finals.append((pc.name, final))
    if before == as_emitted():
        return batch_allowed(agent, finals, emitted=[(coalesce_tool_call_id(tc), tc) for tc in tool_calls])
    batch_allowed(agent, [])  # an emitted call was changed while it was prepared: clear everything
    return False


def _peer_final_args(agent, ref) -> tuple:
    """``(block, final arguments)`` for one call of a peer batch, through the same request
    middleware, ``pre_tool_call`` hooks and guardrail check its dispatch would run."""
    from agent.inline_tool_executors import tool_hook_ids
    from agent.tool_executor import _pre_tool_block, _ToolCallRef
    from hermes_cli.middleware import apply_tool_request_middleware

    request = apply_tool_request_middleware(ref.name, ref.args, skip_relay=True,
                                            **tool_hook_ids(agent, ref.task_id, ref.call_id))
    args = request.payload if isinstance(request.payload, dict) else ref.args
    block, final = _pre_tool_block(agent, _ToolCallRef(ref.name, args, ref.task_id, ref.call_id, ref.trace))
    guardrails = getattr(agent, "_tool_guardrails", None)
    if block is None and guardrails is not None and not guardrails.before_call(ref.name, final).allows_execution:
        block = "guardrail"
    return block, final


def append_peer_refusals(agent, tool_calls, messages: list) -> None:
    """One refusal result per call, so the assistant turn keeps its matching results."""
    from agent.tool_executor import _append_skipped_tool_results

    _append_skipped_tool_results(agent, messages, tool_calls, "", content=PEER_TOOL_REFUSAL)
