"""Which tool calls a canonical peer turn may run: the configured CEO tools, under a verified grant.

A canonical peer turn runs no tool (``agent.tool_executor`` refuses the whole batch) unless the
verified canonical ingress (``GatewayRunner.run_bound_existing_turn``) bound a
:class:`PeerToolGrant` for it after validating the peer metadata against its binding and event.
Under a grant a batch runs only when ALL of these hold, decided per batch from
``gateway.canonical_peer_tools`` read fresh from disk (the ``/reload-mcp`` confirm-gate pattern,
so a later id needs no restart):

* the grant's binding is the configured ``binding``, the grant was issued to this agent, and its
  identity fence still holds (``PeerToolGrant.verify``);
* every call is allowed: a ``read`` name (any arguments) or a ``mutate`` name whose every
  constrained argument is present as one of its exact string values (``mutate`` wins over
  ``read``). The tool must be a native tool of the configured ``mcp_server`` in this profile and
  must run without a Hermes approval, since a peer turn never asks the owner anything;
* ``tool_search``/``tool_describe`` only discover allowed tools (:func:`scope_discovery_defs`);
  ``tool_call`` is allowed only when every call it wraps is.

Anything else refuses the whole batch. An allowed batch PINS each MCP call it checked: the exact
tool, its current native handler and final arguments, with ``peer_grant``, the digest of the policy
entry this decision read. The MCP handler sends a peer-turn call only by claiming its pin
(:func:`claim_pinned_call`), which re-proves the identity fence, so a hook or wrapper that changes
the call after the check, a handler replaced by a refresh, a policy edited since or a binding that
went stale sends nothing. Message text never grants anything.
"""

from __future__ import annotations

import contextvars
import copy
import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

CONFIG_KEY = "canonical_peer_tools"


@dataclass(frozen=True)
class PeerToolGrant:
    """What the verified ingress proved for one peer turn. ``verify`` raises once the binding's
    session is no longer the one the turn was admitted on. ``pins`` holds the calls the latest
    allowed batch checked: ``(tool, canonical arguments, handler) -> [peer_grant, ...]``."""

    binding: str
    agent: Any = field(repr=False, compare=False)
    verify: Callable[[], None] = field(repr=False, compare=False)
    pins: dict = field(default_factory=dict, repr=False, compare=False)


_grant: contextvars.ContextVar[Optional[PeerToolGrant]] = contextvars.ContextVar(
    "canonical_peer_tool_grant", default=None)
_pins_lock = threading.Lock()  # concurrent batches claim pins from worker threads


def bind_grant(grant: PeerToolGrant) -> contextvars.Token:
    """Bind the current peer turn's grant; pair with :func:`reset_grant`."""
    return _grant.set(grant)


def reset_grant(token: contextvars.Token) -> None:
    _grant.reset(token)


@dataclass(frozen=True)
class _Policy:
    binding: str
    server: str
    read: frozenset
    # tool -> {argument: allowed values}; None is a malformed entry, which allows nothing.
    mutate: Mapping[str, Optional[Mapping[str, frozenset]]]
    # tool -> its mutate entry exactly as read, for the peer_grant digest.
    mutate_raw: Mapping[str, Any]

    def names(self) -> frozenset:
        return self.read | frozenset(self.mutate)


def _strings(raw: Any) -> Optional[frozenset]:
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        return None
    return frozenset(raw)


def _constraints(raw: Any) -> Optional[dict]:
    if not isinstance(raw, Mapping):
        return None
    parsed = {argument: _strings(values) for argument, values in raw.items()}
    if any(not isinstance(argument, str) or values is None for argument, values in parsed.items()):
        return None
    return parsed


def _load_policy() -> Optional[_Policy]:
    """The configured policy, read fresh; None when it allows nothing."""
    try:
        from hermes_cli.config import load_config

        raw = (load_config().get("gateway") or {}).get(CONFIG_KEY)
    except Exception:
        return None
    if not isinstance(raw, Mapping):
        return None
    binding, server, mutate = raw.get("binding"), raw.get("mcp_server"), raw.get("mutate")
    if not (isinstance(binding, str) and binding and isinstance(server, str) and server):
        return None
    mutate = {name: c for name, c in mutate.items() if isinstance(name, str)} if isinstance(mutate, Mapping) else {}
    return _Policy(binding, server, _strings(raw.get("read")) or frozenset(),
                   {name: _constraints(c) for name, c in mutate.items()}, copy.deepcopy(mutate))


def _canonical(value: Any) -> Optional[str]:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return None


def peer_grant_digest(entry: Mapping[str, Any]) -> str:
    """``sha256:<hex>`` of the canonical JSON (sorted keys, no spaces, UTF-8) of one policy entry:
    ``{"read": tool}`` or ``{"mutate": {tool: {argument: [values as configured]}}}``."""
    return "sha256:" + hashlib.sha256(_canonical(entry).encode("utf-8")).hexdigest()


def _active_policy(agent: Any = None) -> Optional[_Policy]:
    """The policy, only while this context holds a grant for the configured binding whose fence
    holds (and, given an agent, that was issued to it)."""
    grant = _grant.get()
    if grant is None or (agent is not None and agent is not grant.agent):
        return None
    policy = _load_policy()
    return policy if policy is not None and grant.binding == policy.binding and _fence_holds(grant) else None


def _native_handler(policy: _Policy, name: str) -> Any:
    """The handler registered for ``name`` now, when it is a native tool handler of the configured
    MCP server that runs without approval; else None."""
    from tools.mcp_tool_handlers import mcp_tool_origin

    origin = mcp_tool_origin(name)
    return origin[2] if origin is not None and origin[0] == policy.server and not origin[1] else None


def _fence_holds(grant: PeerToolGrant) -> bool:
    try:
        grant.verify()
    except Exception:
        return False
    return True


def _check(policy: _Policy, name: str, args: Any) -> Optional[list]:
    """The pins an allowed call needs (``[]`` for discovery), or None when it is not allowed."""
    from tools.arg_coercion import coerce_tool_args
    from tools.tool_search import TOOL_CALL_NAME, TOOL_DESCRIBE_NAME, TOOL_SEARCH_NAME, normalize_tool_call_entries

    if not isinstance(args, Mapping):
        return None
    if name == TOOL_SEARCH_NAME:
        return []  # read-only; its results are scoped by scope_discovery_defs
    if name == TOOL_DESCRIBE_NAME:
        raw = args.get("names")
        names = [str(n or "").strip() for n in ([raw] if isinstance(raw, str) else raw)] if isinstance(
            raw, (str, list)) else []
        names = [n for n in names if n]
        allowed = bool(names) and all(n in policy.names() and _native_handler(policy, n) is not None for n in names)
        return [] if allowed else None
    if name == TOOL_CALL_NAME:
        entries, error = normalize_tool_call_entries(dict(args))
        if error is not None:
            return None
        pins: list = []
        for entry in entries:
            inner = _check(policy, entry["name"], entry["arguments"])
            if inner is None:
                return None
            pins.extend(inner)
        return pins
    handler = _native_handler(policy, name)
    if handler is None:
        return None
    # Checked and pinned as dispatch will send them (handle_function_call coerces first).
    final = coerce_tool_args(name, copy.deepcopy(dict(args)))
    if name in policy.mutate:
        constraints = policy.mutate[name]
        if constraints is None or any(not isinstance(final.get(argument), str) or final[argument] not in allowed
                                      for argument, allowed in constraints.items()):
            return None
        entry = {"mutate": {name: policy.mutate_raw[name]}}
    elif name in policy.read:
        entry = {"read": name}
    else:
        return None
    canonical = _canonical(final)
    return None if canonical is None else [(name, canonical, handler, peer_grant_digest(entry))]


def batch_allowed(agent: Any, calls: Sequence[tuple[str, Any]]) -> bool:
    """Whether EVERY ``(tool name, parsed arguments)`` of a peer turn's batch may run (arguments
    that did not parse are None and refuse it). An allowed batch replaces the grant's pins with
    exactly its own calls; a refused one clears them."""
    policy = _active_policy(agent)
    pins: Optional[list] = [] if policy is not None and calls else None
    for name, args in calls if pins is not None else ():
        checked = _check(policy, name, args)
        if checked is None:
            pins = None
            break
        pins.extend(checked)
    grant = _grant.get()
    if grant is not None:
        with _pins_lock:
            grant.pins.clear()
            for tool, canonical, handler, digest in pins or ():
                grant.pins.setdefault((tool, canonical, handler), []).append(digest)
    return pins is not None


def claim_pinned_call(tool: str, args: Any, handler: Any) -> Optional[tuple[str, Callable[[], bool]]]:
    """Consume the pin for exactly this call through exactly this handler and return its
    ``peer_grant`` with a re-check of the identity fence for each send attempt; None when the
    current batch's check did not pin it or the fence no longer holds (it must not be sent)."""
    grant, canonical = _grant.get(), _canonical(args)
    if grant is None or canonical is None:
        return None
    with _pins_lock:
        digests = grant.pins.get((tool, canonical, handler))
        digest = digests.pop(0) if digests else None
    if digest is None or not _fence_holds(grant):
        return None
    return digest, lambda: _fence_holds(grant)


def scope_discovery_defs(tool_defs: Iterable[dict]) -> list:
    """The tool defs the bridge may search, describe or call: unchanged outside a peer turn; in one,
    only the allowed tools (none without an active grant)."""
    from tools.approval_context import is_peer_turn

    if not is_peer_turn():
        return list(tool_defs)
    policy = _active_policy()
    if policy is None:
        return []
    allowed = policy.names()

    def name(tool_def: Any) -> Any:
        return ((tool_def or {}).get("function") or {}).get("name") if isinstance(tool_def, dict) else None

    return [d for d in tool_defs if name(d) in allowed and _native_handler(policy, name(d)) is not None]
