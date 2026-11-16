"""Rotate an aged gateway session by itself, the way ``/new`` does, and carry its open requests.

Production 2026-11-16: the CEO session had been compacted three times and every turn sent ~193K
prompt tokens; the warning said "Consider /new", and only the owner could type it. Here the gateway
rotates the session itself once it is idle (nothing running, nothing queued) and its last real
prompt reached the compaction threshold: compaction no longer keeps it under budget at a turn
boundary, so a fresh session replaces it. The trigger is the provider-reported prompt size, not the
compression count: that count lives on the cached agent's compressor and restarts at 0 whenever the
gateway rebuilds the agent (every cross-process write evicts it), so it never sees the compactions
a long-lived session has accumulated.

The new session opens with one system notice (:data:`HANDOFF_DISPLAY_KIND`) naming the previous
session and the last principal requests it never answered. An owner request carries its text; a
canonical peer request carries only its event, author and channel, never its body, which is the
peer's untrusted text and is only ever shown to the model quoted inside a peer turn.
"""

from __future__ import annotations

from typing import Any, Optional

from hermes_state_messages import ROTATION_HANDOFF_DISPLAY_KIND as HANDOFF_DISPLAY_KIND
from hermes_state_messages import ROTATION_HANDOFF_METADATA as HANDOFF_METADATA

_HANDOFF_ITEMS = 5
_ITEM_CHARS = 400
_SCANNED_ROWS = 200
_OWNER_KINDS = (None, "", "owner_relay")  # Telegram owner rows; gateway.owner_relay.DISPLAY_KIND


def rotation_due(runner: Any, session_key: str) -> Optional[tuple[int, int]]:
    """``(prompt_tokens, threshold)`` when the session's last real prompt reached the compaction
    threshold, else None. The cached agent's compressor gives both when it is there (the threshold
    it compacts at, the prompt its last call sent); the session entry keeps the last prompt across
    agent rebuilds, and ``compression.threshold_tokens`` is the threshold without an agent."""
    entry = getattr(runner.session_store, "_entries", {}).get(session_key)
    if entry is None:
        return None
    from gateway.canonical_surface import cached_actor
    agent, cached_sid = cached_actor(runner, session_key)
    # Only the agent of the session the key routes to now: one left cached across a rotation (its
    # transcript not yet flushed, so /new kept it) speaks for the session before.
    current = agent is not None and cached_sid == getattr(entry, "session_id", None)
    compressor = getattr(agent, "context_compressor", None) if current else None
    tokens = max(_int(getattr(entry, "last_prompt_tokens", 0)),
                 _int(getattr(compressor, "last_prompt_tokens", 0)))
    threshold = _int(getattr(compressor, "threshold_tokens", 0))
    if threshold <= 0:
        compression = (runner._read_user_config() or {}).get("compression")
        threshold = _int(compression.get("threshold_tokens")) if isinstance(compression, dict) else 0
    if threshold <= 0 or tokens < threshold:
        return None
    return tokens, threshold


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal parts: the text ones
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def _request_of(row: dict, nudges: set) -> Optional[str]:
    """The handoff line for a principal's request row, or None when the row is no principal's."""
    if row.get("role") != "user" or row.get("_compressed_summary"):
        return None
    kind = row.get("display_kind")
    if kind == "canonical_peer":
        meta = row.get("display_metadata")  # decoded by SessionDB.get_messages
        peer = meta.get("canonical_peer") if isinstance(meta, dict) else None
        peer = peer if isinstance(peer, dict) else {}
        return ("canonical peer event {event} from {author} in channel {channel} (body not carried; "
                "re-read it from that channel)").format(
                    event=str(peer.get("event_id") or "?")[:12], author=str(peer.get("author_id") or "?")[:12],
                    channel=str(peer.get("channel_id") or "?"))
    if kind not in _OWNER_KINDS:
        return None  # notices, delegation deliveries, hidden prompts
    text = _text(row.get("content")).strip()
    if not text or text in nudges:
        return None
    if len(text) > _ITEM_CHARS:
        text = text[:_ITEM_CHARS] + "…"
    return "owner: " + text


def build_handoff(db: Any, old_session_id: str, tokens: int, threshold: int) -> dict:
    """The notice the new session opens with: where it came from, and the last principal requests
    of the old one that no assistant text answered (oldest first, at most :data:`_HANDOFF_ITEMS`)."""
    from agent.recovery_nudge import nudge_texts
    nudges = set(nudge_texts().values())
    rows = db.get_messages(old_session_id, latest=True, limit=_SCANNED_ROWS) or []
    open_requests: list[str] = []
    current: Optional[str] = None
    for row in rows:
        line = _request_of(row, nudges)
        if line is not None:
            if current is not None:
                open_requests.append(current)
            current = line
        elif (current is not None and row.get("role") == "assistant"
              and _text(row.get("content")).strip()):
            current = None  # answered
    if current is not None:
        open_requests.append(current)
    open_requests = open_requests[-_HANDOFF_ITEMS:]
    lines = [f"[IMPORTANT: This session was started automatically: the previous session "
             f"{old_session_id} reached {tokens} prompt tokens (compaction threshold {threshold}). "
             "Search it with session_search for earlier context."]
    if open_requests:
        lines.append("Requests it never answered, oldest first:")
        lines.extend(f"- {line}" for line in open_requests)
    else:
        lines.append("It had no unanswered requests.")
    lines[-1] += "]"
    return {"role": "user", "content": "\n".join(lines), "display_kind": HANDOFF_DISPLAY_KIND,
            "display_metadata": dict(HANDOFF_METADATA)}
