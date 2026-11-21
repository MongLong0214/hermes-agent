"""``relay_resolve``: the structured declaration a relayed owner turn closes its instruction with.

A relayed owner instruction (gateway/owner_relay.py) is owed a reply the thread shows, and it is complete only
when the turn that answered it declares it resolved. The declaration is this tool call, never a word in an
answer: ``relay_resolve(event_id, outcome, public_reply, evidence)`` with an ``outcome`` of done, answered,
declined or deferred_with_reason. ``public_reply`` is the one thing the thread is told (``<label>: <public_reply>``),
so for answered, declined and deferred it IS the declared answer; ``evidence`` is internal and cites tool calls
only (``{"tool_call_id": ...}``, a tool result the event's turns produced), required for ``done`` and never
published.

The tool is registered so a relayed owner turn can dispatch it, and is in no toolset: the gateway adds it to
that turn alone. Here it only refuses what cannot count: outside a relayed owner turn, or for another event,
or without an outcome and well-formed evidence. Whether the evidence cites something the event's turns really
did is judged by the gateway from the turn's own rows (``gateway.owner_relay.accepted_outcome``), the only place
a declaration is ever accepted.

A recovery turn for an owed event is reply-only (:func:`set_reply_only`): it may answer and declare, and the
executor refuses every other call, so recovering a reply never repeats the work.

What a declaration proves, and what it does not. An owed event closes on the structured declaration plus the
thread's read-back of the published reply, and on nothing else: a bare acknowledgement, the silence marker or an
answer with no declaration never closes it. The closure is a *reply* closure only. Whether the public_reply
actually answers the instruction is NOT guaranteed: no length, sentence or synonym check judges it, so a thin or
off-topic answer closes the event as readily as a good one. A correction finds the reply by the instruction's
event id (the thread's reply is tagged to it, and the shim's journal keeps that id and the reply's digest), and
nothing runs again. Nothing here marks the work done, an acceptance passed or the owner's approval given.
"""

from __future__ import annotations

import contextvars
import re
from typing import Any, Iterable, Optional

from tools.registry import registry, tool_error, tool_result

TOOL_NAME = "relay_resolve"
OUTCOMES = ("done", "answered", "declined", "deferred_with_reason")
# What the peer turn of a post a bound CTO declared a question, ruling request or conflict may declare: no work to
# report, so never ``done``; the declared answer, decision or resolution is the public_reply.
PEER_OUTCOMES = ("answered", "declined", "deferred_with_reason")
# Tools the gateway adds to one turn's surface and no other's: a session's tools freeze never carries or pins them.
TURN_SCOPED_TOOLS = frozenset({TOOL_NAME})
_MAX_EVIDENCE = 64
# What the thread is told is only ever ``<label>: <public_reply>``, composed by the gateway; the evidence is
# internal, kept in the receipt and never published.
PUBLIC_LABELS = {"done": "done", "answered": "answered", "declined": "declined", "deferred_with_reason": "deferred"}
_MAX_PUBLIC_REPLY = 2_000

# A public_reply may not carry what the codebase keeps private. One strict rule for paths, no list of roots and no
# exception: strip the http(s) URLs, then refuse if any "/", any backslash or ``file:`` remains. ``/stop``, ``/root``,
# ``a/b``, ``and/or``, ``TCP/IP``, ``C:/x`` and ``~/x`` are all refused. A URL is exactly ``https?://``, a host of
# letters, digits, dots and hyphens, an optional ``:port`` and an optional path, query or fragment made only of the
# characters below: no ``<>()[]``, no quote or backtick and no backslash, so closing markup ends it and whatever
# follows it, a path or ``file:`` included, is judged like any other text.
_URL = re.compile(r"\bhttps?://[A-Za-z0-9.-]+(?::[0-9]{1,5})?(?:[/?#][A-Za-z0-9._~%!$&*+,;=:@?#/-]*)?",
                  re.IGNORECASE)
_FILE_URI = re.compile(r"\bfile:", re.IGNORECASE)


def _holds_path(text: str) -> bool:
    rest = _URL.sub(" ", text)  # only a URL's own characters are exempt; the rest is judged
    return "/" in rest or "\\" in rest or bool(_FILE_URI.search(rest))


_ID = re.compile(
    r"\b(?:call|fc|toolu)_[A-Za-z0-9]{8,}\b|\bacp_call_\d+\b"            # tool call ids (generated or a provider's)
    r"|\b\d{8}_\d{6}_[0-9a-f]{6,12}\b"                                 # hermes_state_ids.new_session_id
    r"|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"  # relay session and record uuids
    r"|\b(?:canonical-(?:receipt|claim|answer-kept):v\d+|owner-relay-record:v\d+):\S*"   # receipt, claim, ledger keys
    r"|\bsha256:[0-9a-f]{64}\b|\b(?:[0-9a-f]{64}|[0-9a-f]{32})\b",         # digests, event ids, claim owners, nonces
    re.IGNORECASE)

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


def public_reply_problem(text: Any, known_ids: Iterable[str] = ()) -> Optional[str]:
    """Why *text* may not be published, or None. Refused: nothing, over 2,000 characters, a path (any "/" outside
    an http(s) URL, any backslash, or ``file:``), an id of a shape this codebase issues (a tool call, a session, a
    receipt, a claim or a ledger key, a claim owner or event id), or any of *known_ids* (the tool call ids of the
    turns concerned) as a whole id. It judges the shape of what is published, never whether it answers anything."""
    if not isinstance(text, str) or not text.strip():
        return "it is empty"
    if len(text) > _MAX_PUBLIC_REPLY:
        return f"it is over {_MAX_PUBLIC_REPLY} characters"
    if _holds_path(text):
        return "it holds a path (a slash outside an http(s) URL, a backslash or file:)"
    if _ID.search(text):
        return "it holds an id (tool call, session, receipt or claim)"
    for known in known_ids:
        if isinstance(known, str) and known and re.search(rf"(?<![\w-]){re.escape(known)}(?![\w-])", text):
            return "it holds a tool call id of this instruction"
    return None


def evidence_items(evidence: Any) -> Optional[list]:
    """``[("tool", id), ...]`` for well-formed evidence, else None: absent or empty (only ``done`` needs a call to
    cite) or at most 64 objects, each holding exactly one non-blank ``tool_call_id``. Evidence cites tool calls
    only: prose in it is never an answer (the declared answer is the ``public_reply``), so an ``answer`` item is
    malformed."""
    if evidence is None:
        return []
    if not isinstance(evidence, list) or len(evidence) > _MAX_EVIDENCE:
        return None
    items = []
    for item in evidence:
        value = item.get("tool_call_id") if isinstance(item, dict) and set(item) == {"tool_call_id"} else None
        if not isinstance(value, str) or not value.strip():
            return None
        items.append(("tool", value.strip()))
    return items


SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Declare the relayed owner instruction of this turn handled. Call it once, after the work it cites, "
        "only when the instruction really is resolved. outcome: done (the work is finished: cite its tool "
        "results), answered (the instruction asked a question), declined (you will not do it) or "
        "deferred_with_reason (it waits on something). What the thread is told is exactly '<outcome>: "
        "<public_reply>' (outcome as done, answered, declined or deferred): public_reply IS the answer, the "
        "decision or the reason, one plain statement for the CTO, with no slash except in an http(s) URL, no "
        "backslash, no file: and no tool call, session, claim or receipt id; one that has any is refused and the "
        "instruction stays open. evidence is internal, never published, and cites tool calls only: tool_call_id items naming "
        "tool results of this instruction's turns, required for done and optional otherwise; prose in evidence "
        "is never the answer."),
    "parameters": {
        "type": "object",
        "properties": {
            "event_id": {"type": "string", "description": "The Buzz event id of the relayed instruction."},
            "outcome": {"type": "string", "enum": list(OUTCOMES)},
            "evidence": {
                "type": "array", "maxItems": _MAX_EVIDENCE,
                "items": {"type": "object", "required": ["tool_call_id"], "additionalProperties": False,
                          "properties": {"tool_call_id": {
                              "type": "string", "description": "A tool call id whose result you cite."}}}},
            "public_reply": {"type": "string", "maxLength": _MAX_PUBLIC_REPLY,
                             "description": "What the thread is told after the outcome label, in plain words."},
        },
        "required": ["event_id", "outcome", "public_reply"],
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
    evidence = evidence_items(args.get("evidence"))
    if evidence is None:
        return tool_error("evidence must be a list of {tool_call_id} items; prose in it is never the answer.")
    if outcome == "done" and not evidence:
        return tool_error("done cites at least one tool result of this instruction's turns in evidence.")
    problem = public_reply_problem(args.get("public_reply"))
    if problem is not None:
        return tool_error(f"public_reply refused: {problem}. Say it in plain words, with no path and no id.")
    return tool_result(declared=True, event_id=scope, outcome=outcome)


registry.register(
    name=TOOL_NAME, toolset="relay_resolve", schema=SCHEMA, handler=_handle, check_fn=lambda: False,
    description="Declare a relayed owner instruction resolved, with evidence.", emoji="✅",
)
