"""skill_view repeat-view dedup registry: per-task cache of (skill name, file_path) ->
(skill file mtime+size, fingerprint of the text served). A repeat view of an UNCHANGED file returns
a short stub — the earlier tool result already carries the content verbatim. Context compression,
a committed proactive tool-result prune and micro-compaction rewrite the transcript, and a native
Responses compaction checkpoint takes everything before it off the wire; ``drop_lost_skill_views()``
then keeps only the entries whose served text still opens a tool message the next request carries, so
a body demoted, truncated, summarized or compacted away is served in full on its next view (#32106).
"""

import hashlib
import json
import os
import threading
from typing import Dict, Iterable

_skill_view_tracker: Dict[str, Dict[tuple, tuple]] = {}
_skill_view_tracker_lock = threading.Lock()
_SKILL_VIEW_DEDUP_CAP = 200
# Task buckets, least recently recorded evicted first. Only a rewrite that loses every entry frees a
# bucket, so one left by an ended or rotated session would otherwise live for the process; an evicted
# bucket costs one full reload per skill, never a stub.
_SKILL_VIEW_TASK_CAP = 256
# Cheap prefilter before hashing: a served skill_view result opens with its name.
_SERVED_HEAD_CHARS = 64

_SKILL_VIEW_DEDUP_MESSAGE = (
    "Skill content unchanged since it was loaded earlier in this "
    "conversation — refer to the earlier skill_view result; it is still "
    "current and complete. (If context compression removed that result, "
    "re-issuing this returns the full content again.)")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _served_mark(served: str) -> tuple:
    """(length, head, SHA-256) of the exact text a skill_view call returned."""
    return (len(served), served[:_SERVED_HEAD_CHARS], _sha256(served))


def _opens_a_tool_text(mark: tuple, tool_texts: Iterable[str]) -> bool:
    """True when some tool message starts with the served text byte for byte. Later pipeline
    stages only append (guardrail guidance, subdirectory hints), so a verbatim prefix is the
    full body; a replaced, truncated or demoted row never matches."""
    length, head, digest = mark
    return any(len(t) >= length and t.startswith(head) and _sha256(t[:length]) == digest
               for t in tool_texts)


def _skill_view_fingerprint(payload: dict) -> tuple | None:
    """Stat the skill file a successful skill_view served, for change detection."""
    if not (src := payload.get("_source_path")):
        return None
    try:
        st = os.stat(src)
        return (src, st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _record_skill_view(task_id, name, file_path, payload: dict, served: str) -> None:
    """Record a served skill_view (``served`` = the exact result text) so an identical repeat
    can be deduped."""
    # Never dedup setup-needed views: readiness depends on config/env state that
    # changes without the file changing; the model must see the refreshed status.
    if (not task_id or payload.get("setup_needed")
            or payload.get("readiness_status") == "setup_needed"):
        return
    if (fp := _skill_view_fingerprint(payload)) is None:
        return
    key = (str(payload.get("name") or name), file_path or "")
    entry = (*fp, _served_mark(served))
    with _skill_view_tracker_lock:
        cache = _skill_view_tracker.pop(str(task_id), None) or {}
        _skill_view_tracker[str(task_id)] = cache  # re-inserted as the most recently used bucket
        cache[key] = entry
        while len(cache) > _SKILL_VIEW_DEDUP_CAP:  # FIFO eviction
            del cache[next(iter(cache))]
        while len(_skill_view_tracker) > _SKILL_VIEW_TASK_CAP:
            del _skill_view_tracker[next(iter(_skill_view_tracker))]


def _check_skill_view_dedup(task_id, name, file_path) -> str | None:
    """Dedup stub when this exact skill file was already served to this task and
    is unchanged on disk; None otherwise."""
    if not task_id:
        return None
    n = str(name)
    with _skill_view_tracker_lock:
        if not (cache := _skill_view_tracker.get(str(task_id))):
            return None
        # Record key is the RESOLVED name; match raw and resolved forms so
        # 'category/skill' and bare-name views coalesce.
        for key, (src, mtime_ns, size, _served) in list(cache.items()):
            rec_name, rec_fp = key
            if rec_fp != (file_path or "") or (
                    rec_name != n and not n.endswith("/" + rec_name)
                    and not rec_name.endswith("/" + n) and n.split(":")[-1] != rec_name):
                continue
            try:
                st = os.stat(src)
                changed = (st.st_mtime_ns, st.st_size) != (mtime_ns, size)
            except OSError:
                changed = True
            if changed:
                cache.pop(key, None)
                return None
            return json.dumps({
                "success": True, "status": "unchanged", "name": rec_name,
                "file": file_path or "SKILL.md", "dedup": True, "content_returned": False,
                "message": _SKILL_VIEW_DEDUP_MESSAGE}, ensure_ascii=False)
    return None


def _wire_tool_texts(messages: Iterable[dict]) -> list:
    """Tool texts the next request still carries. A native Responses compaction checkpoint takes
    every item before it off the wire (``prune_pre_checkpoint_items``) while the local transcript
    keeps them, so only tool rows after the newest checkpoint carrier count."""
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
        lost = [(key, entry) for key, entry in entries if not _opens_a_tool_text(entry[3], tool_texts)]
    except Exception:  # a proof that raised proved nothing: reload rather than stub
        lost = entries
    with _skill_view_tracker_lock:
        if (cache := _skill_view_tracker.get(str(task_id))) is None:
            return
        for key, entry in lost:
            if cache.get(key) is entry:
                del cache[key]
        if not cache:
            del _skill_view_tracker[str(task_id)]


def reset_skill_view_dedup(task_id: str | None = None) -> None:
    """Clear the dedup cache (all tasks when task_id is None)."""
    with _skill_view_tracker_lock:
        if task_id is None:
            _skill_view_tracker.clear()
        else:
            _skill_view_tracker.pop(str(task_id), None)
