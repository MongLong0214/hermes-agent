"""The one constructor of the user-role recovery nudges the agent loop appends inside a turn, and their
run-bound provenance.

Five nudges re-prompt the model from inside a turn (``agent/conversation_loop.py`` holds their text):
a reasoning-only Codex response, an acknowledgment-only reply, a degenerate final, a dropped tool call
and an empty response after tool calls. Their fixed text is never provenance: anyone can write it.

In a canonical peer turn every nudge row is built here and carries ``_recovery_nudge``:
``{"kind": <kind>, "run": <run token>}``. The run token has ONE source: the ``nonce`` of this turn's own
canonical peer opening row, read from its structured ``display_metadata["canonical_peer"]`` (validated
by ``peer_metadata``) at ``messages[agent._persist_user_message_idx]``. The receipt coordinator mints that
nonce fresh for each claim (``new_peer_metadata``: 128 random bits) and the gateway hands that very
metadata to the agent for the one turn it runs under the claim, so it names this run of this claim and
no other. It is never read from any row's text. With no readable peer opening row (every non-canonical
turn, or an index that no longer points at it) no token exists and the row is the plain row it always
was; the refused-answer keep then refuses the turn, as before.

The key is internal: every provider path drops ``_``-prefixed message keys before a request is sent, and
the transcript store writes named columns only, so the marker is neither sent nor persisted. It lives on
the live row and in the turn's returned ``messages``, which is what ``gateway/canonical_surface.py``
``turn_provenance`` reads through :func:`is_run_nudge`.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

RECOVERY_NUDGE_KEY = "_recovery_nudge"


def nudge_texts() -> dict[str, str]:
    """Each nudge kind and the exact text its row carries."""
    # Lazy: agent.conversation_loop imports the turn phases that import this module.
    from agent.conversation_loop import (
        _CODEX_ACK_CONTINUATION_NUDGE,
        _CODEX_INCOMPLETE_NUDGE,
        _DEGENERATE_FINAL_NUDGE,
        _DROPPED_TOOLCALL_NUDGE_CONTENT,
        _EMPTY_TOOL_RESPONSE_NUDGE,
    )

    return {
        "codex_incomplete": _CODEX_INCOMPLETE_NUDGE,
        "codex_ack_continuation": _CODEX_ACK_CONTINUATION_NUDGE,
        "degenerate_final": _DEGENERATE_FINAL_NUDGE,
        "dropped_toolcall": _DROPPED_TOOLCALL_NUDGE_CONTENT,
        "empty_tool_response": _EMPTY_TOOL_RESPONSE_NUDGE,
    }


def _run_token(agent: Any, messages: Any) -> Optional[str]:
    """The nonce of this turn's own canonical peer opening row, or None when there is none."""
    from agent.canonical_peer import PeerProvenanceError, peer_metadata

    idx = getattr(agent, "_persist_user_message_idx", None)
    if isinstance(idx, bool) or not isinstance(idx, int) or not isinstance(messages, list):
        return None
    if not 0 <= idx < len(messages):
        return None
    try:
        peer = peer_metadata(messages[idx])
    except PeerProvenanceError:
        return None
    return peer["nonce"] if peer is not None else None


def recovery_nudge_row(agent: Any, kind: str, messages: Any, **flags: Any) -> dict[str, Any]:
    """The user row for nudge ``kind``, with ``flags`` (the site's own scaffolding booleans) and, in a
    canonical peer turn, the marker naming this run."""
    row: dict[str, Any] = {"role": "user", "content": nudge_texts()[kind], **flags}
    run = _run_token(agent, messages)
    if run is not None:
        row[RECOVERY_NUDGE_KEY] = {"kind": kind, "run": run}
    return row


def is_run_nudge(row: Any, run: Any) -> bool:
    """Whether ``row`` is a nudge the loop appended in the run named by ``run``: a user row whose marker
    names exactly that run and one of the five kinds, with that kind's text as its content. The kind and
    text are checked only after the run matches; text alone never qualifies a row."""
    if type(run) is not str or not run or not isinstance(row, Mapping) or row.get("role") != "user":
        return False
    marker = row.get(RECOVERY_NUDGE_KEY)
    if not isinstance(marker, Mapping) or set(marker) != {"kind", "run"}:
        return False
    if type(marker["run"]) is not str or marker["run"] != run or type(marker["kind"]) is not str:
        return False
    text, content = nudge_texts().get(marker["kind"]), row.get("content")
    return text is not None and type(content) is str and content == text
