"""skill_view repeat-view dedup registry: per-task cache of (resolved skill name, file_path) ->
fingerprint of the skill_view result served. A repeat view whose result is byte-identical returns a short
stub — the earlier tool result already carries the content verbatim. Content fingerprints need no file
timestamps, so a same-size rewrite in one coarse mtime tick (or on a host whose clock disagrees with
the filesystem's) is never mistaken for "unchanged". Cleared via ``reset_skill_view_dedup()`` on
context compression AND on a committed proactive tool-result prune, because both replace the
original content with a one-line marker. Micro-compaction and a native Responses compaction checkpoint
take only some bodies off the wire: ``drop_lost_skill_views()`` then keeps only the entries whose served
text still opens a tool message the next request carries (#32106).
"""

import hashlib
import json
import threading
from collections.abc import Iterable

_skill_view_tracker: dict[str, dict[tuple, tuple]] = {}
_skill_view_tracker_lock = threading.Lock()
_SKILL_VIEW_DEDUP_CAP = 200
# Task buckets, least recently used evicted first. Only a rewrite that loses every entry frees a bucket,
# so one left by an ended or rotated session would otherwise live for the process; an evicted bucket
# costs one full reload per skill, never a stub.
_SKILL_VIEW_TASK_CAP = 256
# Cheap prefilter before hashing: a served skill_view result opens with its name.
_SERVED_HEAD_CHARS = 64

_SKILL_VIEW_DEDUP_MESSAGE = (
    "Skill content unchanged since it was loaded earlier in this "
    "conversation — refer to the earlier skill_view result; it is still "
    "current and complete. (Re-issued after context compression, this "
    "returns the full content again.)")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _served_mark(served: str) -> tuple:
    """(length, head, SHA-256) of the exact text a skill_view call returned."""
    return (len(served), served[:_SERVED_HEAD_CHARS], _sha256(served))


def dedup_skill_view(task_id, name, file_path, payload: dict, result: str) -> str | None:
    """The "unchanged" stub when this task was already served exactly ``result`` for this skill
    file; otherwise record ``result``'s fingerprint and return None (serve the full result)."""
    # Never dedup setup-needed views: readiness depends on config/env state that
    # changes without the file changing; the model must see the refreshed status.
    if (not task_id or payload.get("setup_needed")
            or payload.get("readiness_status") == "setup_needed"):
        return None
    resolved = str(payload.get("name") or name)
    key = (resolved, file_path or "")
    mark = _served_mark(result)
    with _skill_view_tracker_lock:
        cache = _skill_view_tracker.pop(str(task_id), None) or {}
        _skill_view_tracker[str(task_id)] = cache  # re-inserted as the most recently used bucket
        if cache.get(key) == mark:
            return json.dumps({
                "success": True, "status": "unchanged", "name": resolved,
                "file": file_path or "SKILL.md", "dedup": True, "content_returned": False,
                "message": _SKILL_VIEW_DEDUP_MESSAGE}, ensure_ascii=False)
        cache.pop(key, None)  # re-insert at the end: FIFO eviction keeps the recently served
        cache[key] = mark
        while len(cache) > _SKILL_VIEW_DEDUP_CAP:  # FIFO eviction
            del cache[next(iter(cache))]
        while len(_skill_view_tracker) > _SKILL_VIEW_TASK_CAP:
            del _skill_view_tracker[next(iter(_skill_view_tracker))]
    return None


def _opens_a_tool_text(mark: tuple, tool_texts: Iterable[str]) -> bool:
    """True when some tool message starts with the served text byte for byte. Later pipeline
    stages only append (guardrail guidance, subdirectory hints), so a verbatim prefix is the
    full body; a replaced, truncated or demoted row never matches."""
    length, head, digest = mark
    return any(len(t) >= length and t.startswith(head) and _sha256(t[:length]) == digest
               for t in tool_texts)


def _wire_tool_texts(messages: Iterable[dict]) -> list:
    """Tool texts the next request still carries. A native Responses compaction checkpoint takes
    every item before it off the wire while the local transcript keeps them, so only tool rows
    after the newest checkpoint carrier count."""
    from agent.native_compaction import has_compaction_checkpoint

    rows = [m for m in messages if isinstance(m, dict)]
    start = next((i + 1 for i in range(len(rows) - 1, -1, -1) if rows[i].get("role") == "assistant"
                  and has_compaction_checkpoint(rows[i].get("codex_reasoning_items"))), 0)
    return [m["content"] for m in rows[start:] if m.get("role") == "tool" and isinstance(m.get("content"), str)]


def drop_lost_skill_views(task_id: str, messages: Iterable[dict]) -> None:
    """After a transcript rewrite or a new native compaction checkpoint, drop *task_id*'s entries
    whose served text no longer opens a tool message of *messages* (the transcript as it stands)
    that the next request carries. Survival must be proven; an unproven entry is dropped, so the
    next view reloads rather than stubbing a lost body. An empty transcript proves nothing and
    drops the whole bucket."""
    with _skill_view_tracker_lock:
        entries = list((_skill_view_tracker.get(str(task_id)) or {}).items())
    # Hash outside the process-wide lock. Only the entry objects checked here are dropped, so one
    # re-recorded in the meantime (a fresh serve, in context by construction) stays.
    try:
        tool_texts = _wire_tool_texts(messages)
        lost = [(key, mark) for key, mark in entries if not _opens_a_tool_text(mark, tool_texts)]
    except (TypeError, AttributeError):  # a proof that raised proved nothing: reload rather than stub
        lost = entries
    with _skill_view_tracker_lock:
        if (cache := _skill_view_tracker.get(str(task_id))) is None:
            return
        for key, mark in lost:
            if cache.get(key) is mark:
                del cache[key]
        if not cache:
            del _skill_view_tracker[str(task_id)]


def reset_skill_view_dedup(task_id: str | None = None) -> None:
    """Clear the dedup cache (all tasks when task_id is None); called on context compression."""
    with _skill_view_tracker_lock:
        if task_id is None:
            _skill_view_tracker.clear()
        else:
            _skill_view_tracker.pop(str(task_id), None)
