"""Request-scoped provenance attached to every outbound MCP ``tools/call``.

A Gateway profile's MCP server process is shared by the CEO conversation, other chats, cron, and
subagents (see ``tools/mcp_tool.py``, ``mcp_tool_scope.py``). Without per-call provenance, a server
cannot tell those apart, so a server-side guard protecting a sensitive mutation (e.g. an adopted
CEO's tool) can only choose between trusting every caller or refusing all of them. This module
builds a small, host-derived ``_meta`` block for every call so a server CAN tell them apart.

Trust boundary: every value here comes from the Gateway's own call-time context (session
ContextVars, the approval context, lineage lookup) -- NEVER from caller- or model-supplied
``arguments``. :func:`strip_caller_provenance` must be applied to ``args`` before this module's
output is attached, so a malicious or confused caller cannot plant a forged provenance block under
the same key and have it survive merge ordering.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# The _meta key a request carries its provenance under. Namespaced per MCP's own convention for
# third-party extension metadata (reverse-DNS-ish prefix) to avoid colliding with keys the MCP
# spec or a server's own tooling might use.
PROVENANCE_META_KEY = "agent-control-plane/provenance"


def strip_caller_provenance(args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a copy of *args* with any caller-supplied ``_meta`` (or our provenance key directly)
    removed. A model or a compromised/confused caller must never be able to plant its own
    provenance and have it reach a server as if the Gateway had vouched for it."""
    if not isinstance(args, dict):
        return {}
    cleaned = dict(args)
    if "_meta" in cleaned:
        logger.warning("Stripped a caller-supplied '_meta' from MCP tool arguments before dispatch "
                       "(provenance is host-derived only, never caller-supplied)")
        cleaned.pop("_meta", None)
    cleaned.pop(PROVENANCE_META_KEY, None)
    return cleaned


def _lineage_root_digest(session_id: str) -> Optional[str]:
    """The same digest hermes_state_target_bind.py commits to: a lineage root's identity without
    disclosing the raw session id. None if the lookup fails or there is no bound session."""
    if not session_id:
        return None
    try:
        from hermes_state import SessionDB
        from hermes_state_target_bind import _lineage_root_digest as _digest

        with SessionDB(read_only=True) as db:
            root_id = db._session_lineage_root_to_tip(session_id)
        if not root_id:
            return None
        # _session_lineage_root_to_tip returns root..tip; the root is the first element.
        return _digest(root_id[0])
    except Exception:
        logger.debug("MCP call provenance: lineage root lookup failed for %r", session_id, exc_info=True)
        return None


def build_call_provenance() -> Dict[str, Any]:
    """The current call-time provenance block, derived only from Gateway context. Every key is
    always present (fail-closed consumers, e.g. ACP's guard, must not have to treat a missing key
    as "trust it anyway")."""
    from gateway.session_context import get_session_env
    from tools.approval_context import get_delegation_depth, get_turn_principal

    session_id = get_session_env("HERMES_SESSION_ID", "")
    provenance: Dict[str, Any] = {
        "session_id": session_id,
        "session_key": get_session_env("HERMES_SESSION_KEY", ""),
        "platform": get_session_env("HERMES_SESSION_PLATFORM", ""),
        "chat_id": get_session_env("HERMES_SESSION_CHAT_ID", ""),
        "cron": get_session_env("HERMES_CRON_SESSION", "") == "1",
        "parent_chat_id": get_session_env("HERMES_SESSION_PARENT_CHAT_ID", ""),
        "principal": get_turn_principal(),
        "delegation_depth": get_delegation_depth(),
        "lineage_root_digest": _lineage_root_digest(session_id),
    }
    return provenance
