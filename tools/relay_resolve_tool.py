"""``relay_resolve``: the structured declaration a relayed owner turn closes its instruction with.

A relayed owner instruction (gateway/owner_relay.py) is owed a reply the thread shows, and it is complete only
when the turn that answered it declares it resolved. The declaration is this tool call, never a word in an
answer: ``relay_resolve(event_id, outcome, evidence)`` with an ``outcome`` of done, answered, declined or
deferred_with_reason and non-empty ``evidence``, each item either ``{"tool_call_id": ...}`` (a tool result the
event's turns produced) or ``{"answer": ...}`` (an explicit answer to what was asked).

The tool is registered so a relayed owner turn can dispatch it, and is in no toolset: the gateway adds it to
that turn alone. Here it only refuses what cannot count: outside a relayed owner turn, or for another event,
or without an outcome and well-formed evidence. Whether the evidence cites something the event's turns really
did is judged by the gateway from the turn's own rows (``gateway.owner_relay.accepted_outcome``), the only place
a declaration is ever accepted.

A recovery turn for an owed event is reply-only (:func:`set_reply_only`): it may answer and declare, and the
executor refuses every other call, so recovering a reply never repeats the work.
"""

from __future__ import annotations

import contextvars
from typing import Any, Iterable, Optional

from tools.registry import registry, tool_error, tool_result

TOOL_NAME = "relay_resolve"
OUTCOMES = ("done", "answered", "declined", "deferred_with_reason")
# What the peer turn of a post a bound CTO declared a question, ruling request or conflict may declare: no work to
# report, so never ``done``; the explicit answer, decision or resolution is the evidence.
PEER_OUTCOMES = ("answered", "declined", "deferred_with_reason")
# Tools the gateway adds to one turn's surface and no other's: a session's tools freeze never carries or pins them.
TURN_SCOPED_TOOLS = frozenset({TOOL_NAME})
_MAX_EVIDENCE = 64

# The event id of the relayed owner turn running in this context; None for every other turn.
_scope: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("relay_resolve_scope", default=None)
_reply_only: contextvars.ContextVar[bool] = contextvars.ContextVar("relay_reply_only", default=False)


def bind_scope(event_id: str) -> contextvars.Token:
    return _scope.set(event_id)


def reset_scope(token: contextvars.Token) -> None:
    _scope.reset(token)


def current_scope() -> Optional[str]:
    return _scope.get()


def set_reply_only(flag: bool) -> contextvars.Token:
    return _reply_only.set(bool(flag))


def reset_reply_only(token: contextvars.Token) -> None:
    _reply_only.reset(token)


def is_reply_only_turn() -> bool:
    return _reply_only.get()


def reply_only_refuses(names: Iterable[str]) -> bool:
    """True in a reply-only turn when any of *names* is a tool other than the declaration."""
    return is_reply_only_turn() and any(name != TOOL_NAME for name in names)


def evidence_items(evidence: Any) -> Optional[list]:
    """``[("tool", id) | ("answer", text), ...]`` for well-formed evidence, else None: a non-empty list of at
    most 64 objects, each holding exactly one non-blank ``tool_call_id`` or ``answer`` string."""
    if not isinstance(evidence, list) or not 0 < len(evidence) <= _MAX_EVIDENCE:
        return None
    items = []
    for item in evidence:
        if not isinstance(item, dict) or len(item) != 1:
            return None
        (key, value), = item.items()
        if key not in ("tool_call_id", "answer") or not isinstance(value, str) or not value.strip():
            return None
        items.append(("tool" if key == "tool_call_id" else "answer", value.strip()))
    return items


SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Declare the relayed owner instruction of this turn handled. Call it once, after the work it cites, "
        "only when the instruction really is resolved; your final answer is the reply the thread receives. "
        "outcome: done (the work is finished: cite its tool results), answered (the instruction asked a "
        "question: give the explicit answer), declined (you will not do it: give the reason) or "
        "deferred_with_reason (it waits on something: say what). evidence is required: tool_call_id items "
        "name tool results of this instruction's turns, answer items state the explicit answer or reason. "
        "A declaration without real evidence leaves the instruction open."),
    "parameters": {
        "type": "object",
        "properties": {
            "event_id": {"type": "string", "description": "The Buzz event id of the relayed instruction."},
            "outcome": {"type": "string", "enum": list(OUTCOMES)},
            "evidence": {
                "type": "array", "minItems": 1, "maxItems": _MAX_EVIDENCE,
                "items": {"type": "object", "minProperties": 1, "maxProperties": 1, "properties": {
                    "tool_call_id": {"type": "string", "description": "A tool call id whose result you cite."},
                    "answer": {"type": "string", "description": "The explicit answer or reason."}}}},
        },
        "required": ["event_id", "outcome", "evidence"],
    },
}


def _handle(args: dict, **_kwargs: Any) -> str:
    scope = _scope.get()
    if scope is None:
        return tool_error("relay_resolve is available only in a relayed owner turn.")
    if args.get("event_id") != scope:
        return tool_error("event_id is not the event this turn answers; the declaration is refused.")
    outcome = args.get("outcome")
    if outcome not in OUTCOMES:
        return tool_error(f"outcome must be one of {', '.join(OUTCOMES)}.")
    if evidence_items(args.get("evidence")) is None:
        return tool_error("evidence must be a non-empty list of {tool_call_id} or {answer} items.")
    return tool_result(declared=True, event_id=scope, outcome=outcome)


registry.register(
    name=TOOL_NAME, toolset="relay_resolve", schema=SCHEMA, handler=_handle, check_fn=lambda: False,
    description="Declare a relayed owner instruction resolved, with evidence.", emoji="✅",
)
