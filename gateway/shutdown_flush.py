"""Flush pending messages and agent transcripts to disk before shutdown to prevent data loss.

When FTS5 corruption blocks ``INSERT INTO messages``, ``_pending_messages`` and the live
``agent._session_messages`` are the only surviving copies; shutdown ``.clear()`` would drop them.
All hooks write atomic JSON payloads under ``<hermes_home>/pending_messages/``:
``flush_pending_to_file`` / ``flush_overflow_to_file`` (queue head / FIFO tail, before clear),
``recover_pending_to_db`` (after ``runner.start()``; replays via ``SessionDB.append_message``,
deletes each file on success), ``flush_agent_history_to_file`` (DB flush raised),
``spool_dropped_transcript_message`` / ``drain_transcript_spool``, and the inbound turn records
(``store_inbound_record`` ... ``inbound_event_from_record``) that keep an owner message until its turn ends.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Reason tag for transcript messages dropped by the in-memory pending cap during live
# operation. Payloads carry the full transcript message dict for verbatim replay.
# See #78182.
TRANSCRIPT_CAP_DROP_REASON = "transcript_cap_drop"
# Monotonic tiebreaker so same-second spool files replay in drop order.
_TRANSCRIPT_SPOOL_SEQ = itertools.count()
# Monotonic tiebreaker shared by every pending/overflow payload written in one flush pass
# (head and overflow write their payloads back-to-back from the same shutdown call); recovery
# sorts on this instead of the random UUID filename so FIFO arrival order survives (ROUND1-ESCAPE-1).
# The counter alone governs order WITHIN this process's lifetime -- it only ever increases, so a
# clock step backward mid-flush (round-3 regression) cannot reorder two payloads this process
# wrote. ``_FLUSH_EPOCH_NS`` is read ONCE, separating this process's generation of payloads from
# an earlier process's leftover files (if recovery never got to run before another shutdown):
# a later epoch's payloads still sort after an earlier epoch's, without resampling the clock on
# every write (which is what let it drift mid-batch in the first place).
_FLUSH_ORDER_SEQ = itertools.count()
_FLUSH_EPOCH_NS = time.time_ns()


def _get_flush_dir():
    """Return the pending-messages flush directory under the active HERMES_HOME."""
    from hermes_constants import get_hermes_home
    flush_dir = get_hermes_home() / "pending_messages"
    from hermes_constants import assert_named_profile_home_live
    assert_named_profile_home_live(flush_dir)
    flush_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        os.chmod(flush_dir, 0o700)
    return flush_dir


def _write_payload(flush_dir: Path, payload: Dict[str, Any]) -> Path:
    """Atomically write one private, uniquely named recovery payload; return its path.

    Stamps ``flush_order`` with this process's next flush-sequence value: the filename itself
    (a random UUID) carries no ordering information, so recovery cannot rely on it (ROUND1-ESCAPE-1).
    """
    from utils import atomic_json_write
    payload = {**payload, "flush_order": [_FLUSH_EPOCH_NS, next(_FLUSH_ORDER_SEQ)]}
    final_path = flush_dir / f"pending-{uuid.uuid4().hex}.json"
    atomic_json_write(final_path, payload, mode=0o600, default=str)
    if os.name == "posix":
        # Persist the directory entry too; keep the published file (the only recovery copy) even if
        # fsync fails.
        try:
            directory_fd = os.open(flush_dir, os.O_RDONLY)
        except OSError as exc:
            logger.debug("Failed to fsync pending-message directory: %s", exc)
        else:
            try:
                os.fsync(directory_fd)
            except OSError as exc:
                logger.debug("Failed to fsync pending-message directory: %s", exc)
            finally:
                os.close(directory_fd)
    return final_path


def _flush_value(flush_dir: Path, kind: str, session_key: str, value: Any, **extra: Any) -> bool:
    """Serialise and write one pending value; return True when a payload was written."""
    try:
        if inbound_record_exists(getattr(value, "_inbound_record_id", None)):
            return False  # its inbound record already holds it; a copy here would run it twice
        serialised = _serialise_value(value)
        if serialised is None:
            return False
        _write_payload(flush_dir, {"session_key": session_key, **extra, "data": serialised})
        return True
    except Exception as exc:
        logger.debug("Failed to flush %s message for %s: %s", kind, session_key, exc)
        return False


def flush_pending_to_file(pending: Dict[str, Any], *, reason: str = "shutdown") -> int:
    """Serialise non-empty ``_pending_messages`` slots (``MessageEvent`` or str); return count."""
    if not pending:
        return 0
    flush_dir, ts, flushed = _get_flush_dir(), int(time.time()), 0
    for session_key, value in list(pending.items()):
        if value is not None:
            flushed += _flush_value(flush_dir, "pending", session_key, value, reason=reason, ts=ts)
    if flushed:
        logger.info("Flushed %d pending message(s) to %s (reason=%s)", flushed, flush_dir, reason)
    return flushed


def flush_overflow_to_file(overflow_by_session: Dict[str, Any], *, reason: str = "shutdown") -> int:
    """Serialise the FIFO overflow tails (``queued_events``) to disk; return events flushed.

    The adapter slot holds the queue head and ``SessionState.conversation.queued_events`` the
    tail; both must survive restart. Each event is its own payload in the slot-flush shape so
    ``recover_pending_to_db`` replays them unchanged; ``seq`` preserves arrival order per session.
    """
    if not overflow_by_session:
        return 0
    flush_dir, ts, flushed = _get_flush_dir(), int(time.time()), 0
    for session_key, events in list(overflow_by_session.items()):
        if not session_key or not events:
            continue
        for seq, value in enumerate(list(events)):
            if value is not None:
                flushed += _flush_value(flush_dir, "overflow", session_key, value, reason=reason,
                                        ts=ts, seq=seq)
    if flushed:
        logger.info("Flushed %d queued overflow message(s) to %s (reason=%s)", flushed, flush_dir,
                    reason)
    return flushed


def spool_dropped_transcript_message(session_id: str, message: Dict[str, Any]) -> Optional[Path]:
    """Spool a cap-evicted transcript message; ``None`` on failure (callers degrade to drop+log).

    Uses the same on-disk pending spool as :func:`flush_pending_to_file` (one atomic JSON payload per
    message under ``<hermes_home>/pending_messages/``), so a runtime cap rotation no longer silently
    discards user data while the process stays up (#78182).
    """
    try:
        return _write_payload(_get_flush_dir(), {
            "session_key": session_id, "reason": TRANSCRIPT_CAP_DROP_REASON, "ts": int(time.time()),
            "seq": next(_TRANSCRIPT_SPOOL_SEQ),
            "data": {"session_id": session_id, "message": message},
        })
    except Exception as exc:
        logger.debug("Failed to spool cap-dropped transcript message for %s: %s", session_id, exc)
        return None


def drain_transcript_spool(session_id: str, replay, *, db_known_failing: bool = False) -> tuple[int, int]:
    """Replay cap-dropped transcript messages spooled for *session_id*; return ``(replayed,
    remaining)``. ``replay(message_dict)`` runs per message in drop order; a spool file is deleted
    only after its replay succeeds. The first failure stops the drain (the DB is likely still
    unhealthy) and keeps the rest for retry. With ``db_known_failing`` (the caller's last write
    already failed and is being logged/escalated) a replay failure is expected and logs at DEBUG,
    so a stalled session does not add one WARNING per append on top of its ERROR (#114266).
    """
    try:
        candidates = list(_get_flush_dir().glob("pending-*.json"))
    except Exception as exc:
        logger.debug("Cannot scan transcript spool: %s", exc)
        return 0, 0
    entries = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # A parseable non-object file (scalar/list) cannot be attributed to any session: skip it
        # like unparseable JSON instead of letting ``.get`` abort the whole drain.
        if (not isinstance(payload, dict)
                or payload.get("reason") != TRANSCRIPT_CAP_DROP_REASON
                or payload.get("session_key") != session_id):
            continue
        message = (payload.get("data") or {}).get("message")
        if not isinstance(message, dict):
            logger.warning("Removing structurally invalid transcript spool file %s", path)
            path.unlink(missing_ok=True)
            continue
        entries.append((payload.get("ts", 0), payload.get("seq", 0), path.name, path, message))
    ordered, replayed, remaining = sorted(entries, key=lambda e: e[:3]), 0, 0
    for idx, (_ts, _seq, _name, path, message) in enumerate(ordered):
        try:
            replay(message)
        except Exception as exc:
            (logger.debug if db_known_failing else logger.warning)(
                "Replay of spooled transcript message %s for %s failed; "
                "keeping spool file for retry: %s", path, session_id, exc)
            remaining = len(ordered) - idx
            break
        path.unlink(missing_ok=True)
        replayed += 1
    if replayed:
        logger.info("Replayed %d spooled transcript message(s) for %s after DB recovery", replayed,
                    session_id)
    return replayed, remaining


# Inbound turn records: one durable store with the pending-message spool above. An owner message's
# turn writes its record (``inbound-<recovery id>.json``) before it takes the message from the queue and
# claims the session, and deletes it once the turn has ended; a restart that cut the turn short leaves
# it, and the next process re-admits the message through its adapter (gateway/run_startup.py). The
# record holds the whole event, so it runs again as the message it was: same source, receiving bot
# (``transport_profile``), chat, message id and flags. ``recover_pending_to_db`` never replays one, and
# the flushes above skip an event whose record is on disk (it would otherwise exist twice).
INBOUND_TURN_REASON = "inbound_turn"
_INBOUND_PREFIX = "inbound-"
_INBOUND_EVENT_FIELDS = (
    "text", "message_id", "ledger_message_id", "reply_anchor_override", "platform_update_id", "media_urls",
    "media_types", "media_text_inlined", "reply_to_message_id", "reply_to_text", "reply_to_author_id",
    "reply_to_author_name", "reply_to_is_own_message", "prompt_response", "auto_skill", "channel_prompt",
    "channel_context", "internal", "metadata", "allow_gateway_control", "user_id", "user_name",
)
# Trust signals SessionSource.to_dict() leaves off the wire on purpose; a local record keeps them.
_INBOUND_SOURCE_FLAGS = ("is_bot", "role_authorized", "delivered_via_upstream_relay")


def _inbound_record_path(recovery_id: str) -> Path:
    # The launch home, like the delivery ledger: a served profile's turn runs under that profile's home
    # override, but the startup that re-admits the record reads from the launch context.
    from hermes_constants import get_process_hermes_home
    spool = get_process_hermes_home() / "pending_messages"
    spool.mkdir(parents=True, exist_ok=True, mode=0o700)
    return spool / f"{_INBOUND_PREFIX}{recovery_id}.json"


def next_inbound_order() -> list:
    """Arrival order of a new record, on the flush payloads' own sequence (see ``_FLUSH_ORDER_SEQ``)."""
    return [_FLUSH_EPOCH_NS, next(_FLUSH_ORDER_SEQ)]


def store_inbound_record(event: Any, *, session_key: str, transport_profile: Optional[str],
                         recovery_id: str, order, run: int = 0) -> None:
    """Write *event*'s inbound record atomically (temp file, fsync, rename, directory fsync). Raises
    when the record is not durably on disk; rewriting a record keeps its file and recovery id. *run* is
    the run of the message the record admits next (0 on arrival; gateway/run_startup.py,
    _allocate_inbound_run, writes each rerun's before it is admitted)."""
    from datetime import datetime

    from utils import atomic_json_write
    source = event.source
    data: Dict[str, Any] = {name: getattr(event, name, None) for name in _INBOUND_EVENT_FIELDS}
    data["message_type"] = getattr(event.message_type, "value", event.message_type)
    stamp = getattr(event, "timestamp", None)
    data["timestamp"] = stamp.isoformat() if isinstance(stamp, datetime) else None
    stt_text = getattr(event, "_gateway_pending_stt_text", None)
    if isinstance(stt_text, str) and stt_text:
        data["pending_stt_text"] = stt_text
    data["source"] = {**source.to_dict(),
                      **{flag: bool(getattr(source, flag, False)) for flag in _INBOUND_SOURCE_FLAGS}}
    atomic_json_write(_inbound_record_path(recovery_id), {
        "reason": INBOUND_TURN_REASON, "recovery_id": recovery_id, "session_key": session_key,
        "transport_profile": transport_profile, "order": list(order), "run": int(run), "event": data,
    }, mode=0o600, default=str, fsync_dir=True)


def set_inbound_record_run(record: Dict[str, Any], run: int) -> None:
    """Rewrite a stored record with the run it admits next, atomically, everything else as read (raises
    when it is not durably on disk; the caller then admits nothing)."""
    from utils import atomic_json_write
    atomic_json_write(_inbound_record_path(record["recovery_id"]), {**record, "run": int(run)},
                      mode=0o600, default=str, fsync_dir=True)


def inbound_record_run(record: Dict[str, Any]) -> int:
    """The run a stored record last admitted (``store_inbound_record``); 0 for its first run."""
    run = record.get("run", 0)
    return run if isinstance(run, int) and not isinstance(run, bool) and run >= 0 else 0


def inbound_record_exists(recovery_id: Any) -> bool:
    return isinstance(recovery_id, str) and bool(recovery_id) and _inbound_record_path(recovery_id).exists()


def discard_inbound_record(recovery_id: str) -> bool:
    """Delete a record. Only the settlement boundary calls this (gateway/run_inbound.py,
    _settle_inbound). False when the delete failed (logged; the next start then judges the record)."""
    try:
        _inbound_record_path(recovery_id).unlink(missing_ok=True)
        return True
    except OSError as exc:
        logger.warning("Could not delete inbound record %s: %s", recovery_id, exc)
        return False


def load_inbound_records() -> list:
    """Every readable record, in arrival order. An unreadable file is kept and logged, never deleted."""
    from hermes_constants import get_process_hermes_home
    spool = get_process_hermes_home() / "pending_messages"
    records = []
    for path in spool.glob(f"{_INBOUND_PREFIX}*.json") if spool.is_dir() else ():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            valid = (isinstance(record, dict) and record.get("reason") == INBOUND_TURN_REASON
                     and isinstance(record.get("recovery_id"), str) and isinstance(record.get("session_key"), str)
                     and isinstance(record.get("event"), dict) and isinstance(record["event"].get("source"), dict))
        except Exception:
            valid = False
        if not valid:
            logger.warning("Unreadable inbound record %s kept for manual inspection", path)
            continue
        records.append(record)
    return sorted(records, key=lambda r: (list(r.get("order") or []), r["recovery_id"]))


def inbound_record_ids(event: Any) -> list:
    """Every record an event answers for: its own and those of the messages folded into it."""
    own = getattr(event, "_inbound_record_id", None)
    return [*([own] if own else []), *getattr(event, "_inbound_carried", ())]


def carry_inbound_records(into: Any, absorbed: Any) -> None:
    """*absorbed* was merged into *into*, replaced by it, or steered into its running turn: it has no
    turn of its own now, so its records stay on disk until *into*'s turn settles them (they are never
    deleted at the merge). *absorbed* is marked so nothing settles them separately."""
    if into is None or absorbed is None or into is absorbed:
        return
    ids = inbound_record_ids(absorbed)
    if not ids:
        return
    into._inbound_carried = [*getattr(into, "_inbound_carried", ()), *ids]
    absorbed._inbound_absorbed = True

def inbound_event_from_record(record: Dict[str, Any]) -> Any:
    """The ``MessageEvent`` a record holds, flags included (``internal`` stays as stored)."""
    from datetime import datetime

    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session import SessionSource
    data = record["event"]
    source = SessionSource.from_dict(data["source"])
    for flag in _INBOUND_SOURCE_FLAGS:
        setattr(source, flag, bool(data["source"].get(flag, False)))
    fields = {name: data[name] for name in _INBOUND_EVENT_FIELDS if name in data}
    for name in ("media_urls", "media_types", "media_text_inlined"):
        fields[name] = list(fields.get(name) or [])
    fields["metadata"] = dict(fields.get("metadata") or {})
    for name in ("internal", "reply_to_is_own_message"):
        fields[name] = bool(fields.get(name, False))
    fields["allow_gateway_control"] = bool(fields.get("allow_gateway_control", True))
    stamp = data.get("timestamp")
    if isinstance(stamp, str) and stamp:
        fields["timestamp"] = datetime.fromisoformat(stamp)
    event = MessageEvent(message_type=MessageType(data.get("message_type") or MessageType.TEXT.value),
                         source=source, **fields)
    if isinstance(data.get("pending_stt_text"), str):
        event._gateway_pending_stt_text = data["pending_stt_text"]
    return event


def _json_safe(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _event_recovery_text(event: Any) -> str:
    """Text that stands for a queued ``MessageEvent`` once replayed as a user row.

    A voice note transcribed while its turn was being drained keeps the transcript only in the STT
    cache (``_gateway_pending_stt_text``, its ``text`` stays empty), and ``media_urls`` is not part
    of ``text`` at all — so a media-only event serialised to empty text, which recovery rejects.
    Prefer the cached transcript, and name every attachment by the placeholder the live drain uses.
    """
    stt_text = getattr(event, "_gateway_pending_stt_text", None)
    text = stt_text if isinstance(stt_text, str) and stt_text else (getattr(event, "text", "") or "")
    media_urls = getattr(event, "media_urls", None)
    if not isinstance(media_urls, (list, tuple)) or not media_urls:
        return text
    from gateway.run import _build_media_placeholder
    placeholder = _build_media_placeholder(event)
    return f"{text}\n{placeholder}" if text else placeholder


def _serialise_value(value: Any) -> Optional[dict]:
    """Convert a pending message value to a JSON-serialisable dict."""
    if hasattr(value, "text"):  # MessageEvent-like object
        result: Dict[str, Any] = {"text": _event_recovery_text(value)}
        for attr in ("session_id", "platform", "sender_id", "sender_name", "reply_to", "media",
                     "media_urls", "media_types", "raw_event"):
            val = getattr(value, attr, None)
            if val is not None:
                result[attr] = val if _json_safe(val) else str(val)
        return result
    if isinstance(value, str):  # runner-level _pending_messages
        return {"text": value}
    if isinstance(value, dict) and _json_safe(value):
        return value
    return {"text": str(value)}


def recover_pending_to_db(session_db=None, *, session_resolver=None) -> int:
    """Replay flush-dir ``*.json`` files via ``SessionDB.append_message``, deleting each on success.

    ``session_db=None`` opens (and afterwards releases) the shared default ``state.db``.
    ``session_resolver`` (optional ``(session_key, not_after=ts) -> (session_id, db) | None``, e.g.
    ``SessionStore.resolve_session_id_for_key``) is required for real flush files: adapter
    ``MessageEvent`` objects carry no ``session_id``, so without it every recovery lands in the skip
    branch. A returned ``db`` routes the append to the profile store owning the key (multiplexed
    gateways); ``None`` falls back to ``session_db``. Returns the number of messages recovered.
    """
    raw_paths = list(_get_flush_dir().glob("*.json"))
    if not raw_paths:
        return 0
    # Pre-read every payload so files can be recovered in flush order (ROUND1-ESCAPE-1): the
    # filename is a random UUID and carries no ordering, but each payload written by
    # ``_write_payload`` since the ROUND1-ESCAPE-1 fix carries a monotonic ``flush_order``. A file
    # that fails to parse here is kept (payload=None) and re-attempted — and re-logged — in the
    # recovery loop below, unchanged from the previous one-file-isolation behaviour.
    entries = []
    for path in raw_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        entries.append((path, payload))
    def _order_key(entry):
        path, payload = entry
        order = payload.get("flush_order") if isinstance(payload, dict) else None
        # A legacy file (written before this flush_order field existed) predates every sequenced
        # one -- it sorts FIRST (ROUND1-ESCAPE-1 round 3: the previous key sorted it last, an
        # inverted chronology for exactly the case this sequencing exists to protect: state that
        # outlived an upgrade). Among themselves, legacy files keep filename order (no other
        # ordering signal exists for them); sequenced files sort by their flush order.
        if isinstance(order, int):  # an earlier build of this fix wrote a bare counter
            order = [0, order]
        valid = (isinstance(order, list) and len(order) == 2
                 and all(isinstance(v, int) and not isinstance(v, bool) for v in order))
        return (1, tuple(order), path.name) if valid else (0, (0, 0), path.name)
    flush_files = sorted(entries, key=_order_key)
    own_db = session_db is None
    if own_db:
        from hermes_state_registry import acquire
        session_db = acquire()
    recovered = 0
    try:
        for path, payload in flush_files:
            # One unparseable payload or rejected append must only skip THIS file: the file is
            # never unlinked, so aborting the pass would re-poison every later boot.
            try:
                if payload is None:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                # Agent-history snapshots are for manual operator recovery, not automatic DB
                # insertion.
                if payload.get("reason") == "shutdown-with-unpersisted-agent-history":
                    continue
                if payload.get("reason") == INBOUND_TURN_REASON:
                    continue  # re-admitted as the owner's message at startup, never replayed as a row
                if _recover_one_payload(session_db, path, payload,
                                        session_resolver=session_resolver):
                    recovered += 1
                    path.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Failed to recover pending message from %s: %s", path, exc)
    finally:
        if own_db:  # shutdown cancellation/interrupt must not strand an owned DB
            with contextlib.suppress(Exception):
                from hermes_state_registry import release_or_close
                release_or_close(session_db)
    if recovered:
        logger.info("Recovered %d pending message(s) from shutdown flush", recovered)
    return recovered


def _recover_one_payload(session_db, path: Path, payload: Dict[str, Any], *,
                         session_resolver=None) -> bool:
    """Append one flush payload to ``session_db``; False (file kept) when structurally invalid."""
    # Cap-dropped transcript payloads carry the full message dict keyed by session_id — replay directly
    # (#78182). This handles spool files that were never drained before a restart.
    if payload.get("reason") == TRANSCRIPT_CAP_DROP_REASON:
        # Cap-dropped payloads carry the full message dict keyed by session_id — replay directly.
        data = payload.get("data", {}) or {}
        spooled_sid, message = data.get("session_id", ""), data.get("message")
        if not spooled_sid or not isinstance(message, dict):
            logger.warning("Cannot recover structurally invalid transcript spool "
                           "file %s; preserved for manual inspection", path)
            return False
        session_db.append_message(session_id=spooled_sid, role=message.get("role", "unknown"),
                                  content=message.get("content") or "",
                                  timestamp=message.get("timestamp") or payload.get("ts"))
        return True
    session_key, data = payload.get("session_key", ""), payload.get("data", {})
    text = data.get("text", "")
    if not text or not session_key:
        logger.warning("Cannot recover structurally invalid pending message from %s; "
                       "the flush file has been preserved", path)
        return False
    # session_key is a gateway routing key (e.g. "agent:main:telegram:..."); appending a row
    # needs the real session_id, which real payloads lack — the resolver supplies it together with
    # the store owning the key. ``session_db`` (the owned default) serves only payloads that already
    # carry a session_id; a resolver-resolved payload goes to the resolver's db alone, never the
    # ambient root store (a None db from the resolver is not a fallback signal — it is "preserve").
    session_id, target_db = data.get("session_id", ""), session_db
    if not session_id and session_resolver is not None:
        try:
            resolved = session_resolver(session_key, not_after=payload.get("ts"))
        except Exception as exc:
            logger.debug("Session key->id resolution failed for %s: %s", session_key, exc)
            resolved = None
        if resolved and resolved[1] is not None:
            session_id, target_db = resolved
    if not session_id:
        logger.warning("Cannot recover pending message for %s: no session_id in flush file and "
                       "session_key-to-id resolution failed. "
                       "The message text is preserved in %s", session_key, path)
        return False
    target_db.append_message(session_id=session_id, role="user", content=text,
                             timestamp=payload.get("ts", int(time.time())))
    return True


def flush_agent_history_to_file(session_id: Optional[str], history: list) -> None:
    """Best-effort dump of an agent's in-memory transcript before teardown. Used when
    ``_flush_messages_to_session_db`` raises (e.g. FTS/SQLite corruption): the transcript is written
    outside the broken DB so an operator can salvage it after repairing state.db. Failures are
    swallowed — shutdown must never block on a best-effort backup."""
    if not history:
        return
    try:
        flush_dir = _get_flush_dir()
        snapshot = []
        for _m in history:
            try:
                plain = isinstance(_m, (dict, list, str, int, float, bool, type(None)))
                snapshot.append(_m if plain else str(_m))
            except Exception:
                continue
        _write_payload(flush_dir, {
            "reason": "shutdown-with-unpersisted-agent-history", "issue": "#72680",
            "session_id": session_id, "count": len(snapshot), "messages": snapshot,
        })
        logger.warning("Preserved %d in-memory message(s) for session %s "
                       "(possible FTS corruption — recover after repairing state.db)",
                       len(snapshot), session_id)
    except Exception as _e:
        logger.warning("Agent-history shutdown preservation failed for session %s: %s", session_id,
                       _e)
