"""Request-scoped provenance attached to every outbound MCP ``tools/call``.

One profile's MCP server process is shared by the owner's conversations, canonical peer turns, cron and
subagents. Without per-call provenance a server cannot tell them apart, so a server-side guard protecting a
sensitive mutation can only trust every caller or refuse all of them. This module builds a small host-derived
``_meta`` block for each call so a server can tell them apart.

Trust boundary: every value comes from the host's call-time context (session ContextVars and the approval
context), never from caller- or model-supplied ``arguments``. Apply :func:`strip_caller_provenance` to the
arguments before dispatch, so a caller cannot plant a forged block under the same key. Build the block in the
tool handler, on the turn's own thread, so it never depends on which context the MCP loop runs the RPC in.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# The ``_meta`` key a request carries its provenance under (reverse-DNS-style prefix, per MCP's convention for
# third-party extension metadata, so it cannot collide with spec or server keys).
PROVENANCE_META_KEY = "agent-control-plane/provenance"


def strip_caller_provenance(args: Optional[dict[str, Any]]) -> dict[str, Any]:
    """A copy of *args* without any caller-supplied ``_meta`` or provenance key."""
    if not isinstance(args, dict):
        return {}
    cleaned = dict(args)
    if cleaned.pop("_meta", None) is not None:
        logger.warning("Stripped a caller-supplied '_meta' from MCP tool arguments "
                       "(provenance is host-derived only)")
    cleaned.pop(PROVENANCE_META_KEY, None)
    return cleaned


def _turn_principal() -> str:
    """The turn's principal; ``"peer"`` when the approval context cannot be read (fail closed)."""
    try:
        from tools.approval_context import get_turn_principal
        return get_turn_principal()
    except Exception:
        logger.warning("MCP call provenance: turn principal unreadable; reporting peer", exc_info=True)
        return "peer"


def build_call_provenance() -> dict[str, Any]:
    """The call-time provenance block, derived only from host context. Every key is always present, so a
    fail-closed consumer never has to read a missing key as "trust it". ``lineage_root_digest`` stays None
    until this tree has a target-binding store to commit the digest to."""
    from gateway.session_context import get_session_env
    from tools.approval_context import get_delegation_depth

    return {
        "session_id": get_session_env("HERMES_SESSION_ID", ""),
        "session_key": get_session_env("HERMES_SESSION_KEY", ""),
        "platform": get_session_env("HERMES_SESSION_PLATFORM", ""),
        "chat_id": get_session_env("HERMES_SESSION_CHAT_ID", ""),
        "cron": get_session_env("HERMES_CRON_SESSION", "") == "1",
        "parent_chat_id": get_session_env("HERMES_SESSION_PARENT_CHAT_ID", ""),
        "principal": _turn_principal(),
        "delegation_depth": get_delegation_depth(),
        "lineage_root_digest": None,
    }
