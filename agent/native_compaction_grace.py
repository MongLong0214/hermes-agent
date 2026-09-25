"""One native-compaction attempt before local compression when a request crosses the trigger.

Local preflight runs on usage-anchored pressure (real prompt count + delta) and an anchored
figure is never deferred, so on a native-compaction route the request that first crosses the
local trigger was summarized locally before the server could compact it. This grace gives that
request one native capture and the next request one checkpoint replay, which must bill below
both the pre-crossing and the capture prompt count.

A failed grace request is retried by the loop's ordinary error handling, under its backoff and
retry budget, like any other request. A one-shot ``fallback`` (the next preflight check runs
local compression) follows only a capture whose stream died before any event (its in-stream
reconnect would resend the over-trigger payload), a wire that lost its native owner, a capture
without a checkpoint, a replay without a proven reduction, a session/route change and a request
whose response the loop never observed. A consumed fallback leaves the grace ``spent`` until the
turn ends, so no turn re-arms after an attempt fell back.

Only the loop's pre-API and post-tool checks arm an attempt. A new turn keeps nothing but a
pending replay, so the turn-start preflight otherwise runs exactly as without the grace.

State lives on the agent in memory only (a resumed session starts local). No module-level
transport imports: ``codex_runtime`` asks before it reconnects.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agent.native_compaction import has_compaction_checkpoint, native_compaction_context_management

_STATE_ATTR = "_native_compaction_preflight_grace"
_AWAITING_CAPTURE, _AWAITING_REPLAY = "awaiting_capture", "awaiting_replay"
_IN_FLIGHT, _FALLBACK, _SPENT = "in_flight", "fallback", "spent"
_AWAITING = (_AWAITING_CAPTURE, _AWAITING_REPLAY)


class NativeCompactionPreflightRefused(RuntimeError):
    """The armed grace found no native owner at the wire boundary; the loop rebuilds the
    request through local preflight instead of sending it."""


def _count(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _state(agent: Any) -> Optional[Dict[str, Any]]:
    state = getattr(agent, _STATE_ATTR, None)
    return state if isinstance(state, dict) else None


def _clear(agent: Any) -> None:
    setattr(agent, _STATE_ATTR, None)


def _identity(agent: Any) -> tuple:
    return tuple(str(getattr(agent, name, "") or "") for name in (
        "session_id", "api_mode", "model", "provider", "base_url"))


def _eligible(agent: Any) -> bool:
    """Same gate the payload builder uses, so the grace never outlives the native field."""
    if getattr(agent, "api_mode", None) != "codex_responses":
        return False
    from agent.codex_responses_adapter import classify_responses_route

    route = classify_responses_route(agent)._asdict()
    return native_compaction_context_management(agent, **route) is not None


def _has_headroom(agent: Any, pressure: int) -> bool:
    """The server can only compact a request the window accepts with its output budget."""
    compressor = getattr(agent, "context_compressor", None)
    context_length = _count(getattr(compressor, "context_length", None))
    max_output = _count(getattr(compressor, "max_tokens", None)) or _count(getattr(agent, "max_tokens", None))
    return context_length - max_output > pressure


def _messages_have_checkpoint(messages: Any) -> bool:
    return isinstance(messages, list) and any(
        isinstance(message, dict) and has_compaction_checkpoint(message.get("codex_reasoning_items"))
        for message in messages
    )


def _route_replays_checkpoint(agent: Any, messages: Any) -> bool:
    """The adapter's own filter: whether this route's wire input will carry the checkpoint."""
    from agent.codex_responses_adapter import has_replayable_native_compaction_checkpoint

    return isinstance(messages, list) and has_replayable_native_compaction_checkpoint(agent, messages)


def _carries_native_field(api_kwargs: Any) -> bool:
    field = api_kwargs.get("context_management") if isinstance(api_kwargs, dict) else None
    return isinstance(field, list) and any(
        isinstance(item, dict) and item.get("type") == "compaction" for item in field
    )


def _response_has_checkpoint(response: Any) -> bool:
    output = getattr(response, "output", None)
    if not isinstance(output, list):
        return False
    items = [item if isinstance(item, dict) else {
        "type": getattr(item, "type", None), "encrypted_content": getattr(item, "encrypted_content", None),
    } for item in output]
    return has_compaction_checkpoint(items)


def begin_native_compaction_turn(agent: Any) -> None:
    """A new turn keeps only a proven capture's pending replay: an unresolved, failed or spent
    attempt never reaches the turn-start preflight."""
    state = _state(agent)
    if state is not None and state["phase"] != _AWAITING_REPLAY:
        _clear(agent)


def defer_turn_start_preflight_for_native_replay(agent: Any, pressure_tokens: Any, *, messages: Any) -> bool:
    """True when turn-start compression must wait for the replay a capture is owed. Turn start
    never arms an attempt: one that failed there would bypass the turn-start fail-closed passes.
    A replay that cannot go out is dropped, leaving turn start exactly as without the grace."""
    state = _state(agent)
    if state is None or state["phase"] != _AWAITING_REPLAY:
        return False
    if defer_local_preflight_for_native_compaction(agent, pressure_tokens, messages=messages):
        return True
    _clear(agent)
    return False


def defer_local_preflight_for_native_compaction(agent: Any, pressure_tokens: Any, *, messages: Any) -> bool:
    """True when local compression must wait for this request's native attempt.

    Arms only on the crossing itself: the last real prompt was under the trigger, the
    history holds no checkpoint yet, the route sends the native field and the request fits
    the window with its output budget."""
    state = _state(agent)
    if state is not None and state["phase"] == _SPENT:
        return False
    if state is not None and (state["phase"] == _IN_FLIGHT or state["identity"] != _identity(agent)):
        # A preflight check while the grace request is unresolved means its response was
        # discarded (redirect, restart); a rebuilt route/session cannot spend the old chance.
        state["phase"] = _FALLBACK
    if state is not None and state["phase"] == _FALLBACK:
        return False
    compressor = getattr(agent, "context_compressor", None)
    threshold = _count(getattr(compressor, "threshold_tokens", None))
    pressure = _count(pressure_tokens)
    if not threshold or pressure < threshold or not _eligible(agent) or not _has_headroom(agent, pressure):
        _clear(agent)
        return False
    if state is not None:
        if state["phase"] == _AWAITING_REPLAY and not _route_replays_checkpoint(agent, messages):
            state["phase"] = _FALLBACK
            return False
        return True
    last_real = _count(getattr(compressor, "last_real_prompt_tokens", None))
    if _messages_have_checkpoint(messages) or not 0 < last_real < threshold:
        return False
    setattr(agent, _STATE_ATTR, {
        "phase": _AWAITING_CAPTURE, "identity": _identity(agent), "baseline_prompt_tokens": last_real,
    })
    return True


def consume_native_compaction_preflight_fallback(agent: Any) -> bool:
    """Spend a pending fallback: the caller runs local preflight now, overriding the
    real-usage deferral (the over-threshold request has no native owner left). The grace
    stays spent for the rest of the turn, so a failed local pass cannot re-arm an attempt."""
    state = _state(agent)
    if state is None or state["phase"] != _FALLBACK:
        return False
    state["phase"] = _SPENT
    return True


def start_native_compaction_preflight_request(agent: Any, api_kwargs: Any) -> bool:
    """Spend an armed grace at the outbound boundary. False: this request has no native
    owner (a capture without the field, a replay without the checkpoint, a changed route)
    and must not be sent."""
    state = _state(agent)
    if state is None or state["phase"] not in _AWAITING:
        return True
    owned = (
        has_compaction_checkpoint(api_kwargs.get("input")) if state["phase"] == _AWAITING_REPLAY
        else _carries_native_field(api_kwargs)
    )
    if state["identity"] != _identity(agent) or not owned:
        state["phase"] = _FALLBACK
        return False
    state["request_phase"], state["phase"] = state["phase"], _IN_FLIGHT
    return True


def observe_native_compaction_preflight_response(agent: Any, response: Any, *, prompt_tokens: Any) -> None:
    """Resolve the in-flight grace from its provider response, exactly once."""
    state = _state(agent)
    if state is None or state["phase"] != _IN_FLIGHT:
        return
    prompt = _count(prompt_tokens)
    if state["request_phase"] == _AWAITING_CAPTURE and _response_has_checkpoint(response):
        state.update(phase=_AWAITING_REPLAY, capture_prompt_tokens=prompt)
        return
    if state["request_phase"] == _AWAITING_REPLAY:
        bounds = [n for n in (state["baseline_prompt_tokens"], state.get("capture_prompt_tokens", 0)) if n]
        if prompt and bounds and prompt < min(bounds):
            _clear(agent)
            return
    state["phase"] = _FALLBACK


def withhold_native_capture_reconnect(agent: Any) -> bool:
    """True when the stream that died before any event carried the native capture: its
    in-stream reconnect would resend the over-trigger payload, so the caller raises and the
    loop rebuilds the request through local preflight. A replay reconnects like any request."""
    state = _state(agent)
    if state is None or state["phase"] != _IN_FLIGHT or state["request_phase"] != _AWAITING_CAPTURE:
        return False
    state["reconnect_withheld"] = True
    return True


def release_failed_native_preflight(agent: Any, error: BaseException) -> bool:
    """Settle the grace request ``error`` ended. True when the caller must rebuild it through
    local preflight: the wire-boundary refusal, or a capture whose reconnect was withheld — two
    failures base never brings to the error handler. Any other failure puts the grace back to
    its awaiting phase and returns False, so the ordinary handling retries the same request."""
    if isinstance(error, NativeCompactionPreflightRefused):
        return True
    state = _state(agent)
    if state is None or state["phase"] != _IN_FLIGHT:
        return False
    if state.pop("reconnect_withheld", False):
        state["phase"] = _FALLBACK
        return True
    state["phase"] = state["request_phase"]
    return False
