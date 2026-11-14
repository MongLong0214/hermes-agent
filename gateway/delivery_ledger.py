"""Durable delivery-obligation ledger for gateway final responses (rows in the shared ``state.db``;
WAL, owner pid + process-start liveness, bounded retention) so a crash between finalize and
platform ACK cannot lose a response silently. Checkpoints: record_obligation() 'pending' before
any send | mark_attempting() 'attempting' right before the await | mark_delivered() 'delivered'
only on SendResult.success | mark_failed() 'failed' on a definitive rejection. Crash semantics
(never silently resend an ambiguous send): pending = never started, redeliver plainly; attempting
= crashed mid-await, platform MAY have it, redeliver WITH a visible recovered marker; failed =
rejected once, restart is a retry boundary, also marked; delivered = prune. Attempts are capped
and stale rows expire, both -> 'abandoned' (kept briefly, then pruned). Everything is
best-effort: ledger failures must never block a send; callers wrap every call in try/except. The
one exception is an owner message's reply: it is sent only once what it requires is stored
(record_components, ReplyNotStored), and its delivery is decided by inbound_completion.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import re
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from gateway.dead_targets import classify_dead_error
from hermes_cli.sqlite_util import add_column_if_missing
from hermes_constants import get_process_hermes_home

logger = logging.getLogger(__name__)
_DB_LOCK = threading.Lock()

# Redelivery policy knobs (deliberately not config — the ledger is gated by
# ``gateway.delivery_ledger`` and these only matter in the rare recovery path).
MAX_ATTEMPTS = 3
STALE_AFTER_SECONDS = 24 * 60 * 60
_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_ROWS = 500
_ACP_RETENTION_SECONDS = 90 * 24 * 60 * 60

# Visible prefixes for redeliveries that might duplicate an already-received message (crash mid-send /
# post-rejection retry) — honest at-least-once. Runtime recovery uses a distinct marker: no restart
# occurred, but a network rejection's acknowledgement can still have been lost independently.
RECOVERED_MARKER = "♻️ Recovered reply — the gateway restarted during delivery, so this may be a duplicate:\n\n"
RECONNECTED_MARKER = ("♻️ Recovered reply — the messaging platform reconnected after the original "
                      "delivery failed, so this may be a duplicate:\n\n")
# A reply refused by flood control may have gone out as several requests (the adapter chunks long replies,
# and MarkdownV2 escaping alone can push a reply that fits one message into two), and the platform may have
# accepted the first chunk(s) before refusing the rest; the send result does not say which landed. The raw
# length of the stored text says nothing about that, so every flood redelivery carries a marker, and neither
# of the markers above tells the truth here (no restart, no reconnect): the rate limit gets its own.
FLOOD_MARKER = ("♻️ Recovered reply — the messaging platform's rate limit refused the original, so part of "
                "it may already have arrived above:\n\n")

# Errors whose send contract proves the platform never saw the request: retried as soon as the adapter
# is back, no backoff. Every other rejection is retried too (#91653: a 5xx or a transient parse error
# used to strand the reply in ``failed`` until the next restart), but only after a backoff that grows
# with the attempts already spent, so a platform-side outage is not hammered by the redelivery timer.
# A whole-chat death (blocked bot, deleted group, deactivated user) is never retried: the target is gone.
_RUNTIME_RETRYABLE_ERRORS = frozenset({"send_path_degraded"})
# One tier per in-process retry; the last budgeted attempt is left to the boot sweep (retry_not_before).
_RETRY_BACKOFF_SECONDS = (30.0, 120.0)
assert len(_RETRY_BACKOFF_SECONDS) == MAX_ATTEMPTS - 1

# A final send the platform refused with flood control is the other transient case: a 429 means the
# refused request was never accepted, and the platform said how long to wait. Adapters fail such sends
# closed as ``flood_control:<seconds>`` on purpose (#91969) so that this ledger owns the wait instead of
# the send coroutine sleeping through it. Before this the row simply sat in ``failed`` until the next
# restart's sweep, which then redelivered it hours late under the "gateway restarted during delivery"
# marker.
FLOOD_ERROR_PREFIX = "flood_control:"
FLOOD_RETRY_DEFAULT_SECONDS = 60.0
FLOOD_RETRY_CAP_SECONDS = 15 * 60.0
FLOOD_RETRY_SLACK_SECONDS = 2.0

# The canonical prefix above is what the adapters produce for a flood they decide not to sleep. It is
# not the only shape that reaches this ledger. PTB raises ``RetryAfter``, whose own text reads
# "Flood control exceeded. Retry in 185 seconds", and that text is what lands in ``last_error``
# whenever a send fails on a path that has not been normalized, or was written by an older build
# before it was. Such a row has to be recognised too: unrecognised, it is treated as an ordinary
# failure, so no redelivery timer is armed for it and a boot sweep claims it immediately instead of
# adopting it until its deadline, spending the one attempt inside the penalty that caused it.
# Matching requires the "flood control" wording as well as the delay, so an unrelated error that
# merely suggests retrying is never mistaken for a flood.
_RAW_FLOOD_RE = re.compile(r"flood control exceeded.*?retry in\s+(\d+(?:\.\d+)?)", re.IGNORECASE)


def _raw_flood_wait(text: str) -> Optional[float]:
    """Seconds asked for by a flood error still carrying the platform's own wording, else ``None``."""
    match = _RAW_FLOOD_RE.search(text or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def is_reconnect_only(error: Any) -> bool:
    """True for a row that only an adapter reconnect may retry (no timer, no backoff)."""
    return str(error or "").strip().lower() in _RUNTIME_RETRYABLE_ERRORS


def is_flood_error(error: Any) -> bool:
    """True for a flood refusal: the adapters' fail-closed ``flood_control:<seconds>`` result, or a
    row still carrying the platform's own flood wording (see ``_RAW_FLOOD_RE``)."""
    text = str(error or "").strip().lower()
    return text.startswith(FLOOD_ERROR_PREFIX) or _raw_flood_wait(text) is not None


def flood_wait_seconds(error: Any, default: float = FLOOD_RETRY_DEFAULT_SECONDS) -> float:
    """The wait the platform asked for, read from ``flood_control:<seconds>``; ``default`` when unreadable."""
    text = str(error or "").strip().lower()
    wait = default
    if text.startswith(FLOOD_ERROR_PREFIX):
        try:
            wait = float(text[len(FLOOD_ERROR_PREFIX):].strip())
        except ValueError:
            wait = default
    else:
        # A row still carrying the platform's own flood wording states its delay just as precisely,
        # and the deadline must use it rather than the generic default.
        raw = _raw_flood_wait(text)
        if raw is not None:
            wait = raw
    return wait if wait > 0 else default


def flood_retry_delay(seconds: Any) -> float:
    """How long a redelivery timer sleeps for a wait of ``seconds``: capped so a huge or bogus value cannot
    park the timer for hours (the row's own deadline, not the timer, decides eligibility when it fires),
    plus a little slack so the timer lands after the deadline rather than on it."""
    try:
        wait = float(seconds)
    except (TypeError, ValueError):
        wait = FLOOD_RETRY_DEFAULT_SECONDS
    return min(max(wait, 0.0), FLOOD_RETRY_CAP_SECONDS) + FLOOD_RETRY_SLACK_SECONDS


def _failed_stamp(updated_at: Any) -> float:
    try:
        return float(updated_at or 0.0)
    except (TypeError, ValueError):
        return 0.0


def flood_not_before(updated_at: Any, last_error: Any) -> float:
    """Earliest moment a flood-refused row may be resent: the refusal's timestamp (``mark_failed`` sets
    ``updated_at``) plus the platform's wait. Enforced by the sweeps so neither an early timer nor a
    reconnect sweep spends a redelivery attempt inside the penalty window."""
    return _failed_stamp(updated_at) + flood_wait_seconds(last_error)


def retry_not_before(updated_at: Any, last_error: Any, attempts: Any) -> Optional[float]:
    """Earliest moment a failed row may be resent, or ``None`` for a row the runtime must leave alone:
    a flood refusal keeps the platform's own wait, an allowlisted reconnect error is due at once, a
    whole-chat death is final, and any other rejection backs off by the attempts already spent — but
    never spends the LAST budgeted attempt. An unclassified outage can outlast any timer, and a row
    the timer abandoned would be lost for good; leaving one attempt keeps it recoverable by the boot
    sweep after a restart, which is a real recovery signal."""
    if is_flood_error(last_error):
        return flood_not_before(updated_at, last_error)
    text = str(last_error or "").strip().lower()
    if is_reconnect_only(text):
        return _failed_stamp(updated_at)
    if classify_dead_error(text):
        return None
    spent = int(attempts or 0)
    if spent >= MAX_ATTEMPTS - 1:
        return None
    return _failed_stamp(updated_at) + _RETRY_BACKOFF_SECONDS[spent]


def _db_path():
    # Launch home, not get_hermes_home(): a multiplexed gateway records a served profile's replies
    # under that profile's home override, but the boot sweep reads from the launch context, so both
    # must open the one shared store (adapter_profile tells the bots apart). No get_hermes_home()
    # fallback for an unset HERMES_HOME: a default gateway run in the foreground has none.
    return get_process_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    from hermes_cli.sqlite_util import open_db
    from hermes_state_fence import open_fenced_state_connection

    path = _db_path()
    # Shared state.db: SessionDB owns the durable PRAGMA set; this opener keeps the plain-tuple rows
    # and the 10 s busy timeout it always had. The lineage probe runs before open_db so a refused
    # store gets neither the WAL switch nor the DDL below.
    return open_fenced_state_connection(
        path, bootstrap=False, initialize=_initialize_schema,
        connect=lambda: open_db(path, db_label="state.db (delivery_ledger)", busy_timeout_ms=10_000,
                                row_factory=None),
    )


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_obligations (
            obligation_id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            platform TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT,
            content TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            last_error TEXT,
            adapter_profile TEXT
        )"""
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(delivery_obligations)")}
    if "adapter_profile" not in columns:
        add_column_if_missing(conn, "delivery_obligations", "adapter_profile", "adapter_profile TEXT")
    if "acp_update_id" not in columns:
        # U4: the Telegram update an /acp task answered, written in the same INSERT as the obligation so
        # a ledgered answer is never without its receipt correlation (gateway.acp_managed_ingress).
        add_column_if_missing(conn, "delivery_obligations", "acp_update_id", "acp_update_id TEXT")
    if "delivered_message_ids" not in columns:
        # JSON list of the platform message ids the delivered answer occupies (send order), written by
        # the same UPDATE that marks the row delivered: an /acp receipt's delivery evidence.
        add_column_if_missing(conn, "delivery_obligations", "delivered_message_ids",
                              "delivered_message_ids TEXT")
    if "reactivations" not in columns:
        # How often a later start gave an owner message's reply another send after its retry budget was
        # spent (reactivate_expired); bounded by MAX_REACTIVATIONS.
        add_column_if_missing(conn, "delivery_obligations", "reactivations",
                              "reactivations INTEGER NOT NULL DEFAULT 0")
    # The components an owner message's reply requires (record_components): its text, whose delivery is
    # the delivery_obligations row of the same id (mirrored here), and each attachment, which no row
    # carries. A reply is delivered only when every one of them is (inbound_completion).
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_components (
            obligation_id TEXT NOT NULL,
            component_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            state TEXT NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (obligation_id, component_id)
        )"""
    )
    if "redeliveries" not in {row[1] for row in conn.execute("PRAGMA table_info(delivery_components)")}:
        # How many later starts tried an undelivered attachment again (claim_attachment_redelivery).
        add_column_if_missing(conn, "delivery_components", "redeliveries",
                              "redeliveries INTEGER NOT NULL DEFAULT 0")


def _transaction():
    from hermes_cli.sqlite_util import transaction

    return transaction(_connect())


def _start_time(pid: int) -> Optional[int]:
    try:
        from gateway.status import get_process_start_time  # lazy: tests monkeypatch gateway.status
        return get_process_start_time(pid)
    except Exception:
        return None


def _owner_stamp() -> tuple[int, Optional[int]]:
    pid = os.getpid()
    return pid, _start_time(pid)


def _owner_alive(pid: Any, started_at: Any) -> bool:
    """True when the recorded owning process still exists (pid + start time)."""
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    current_start = _start_time(pid)
    if current_start is None:
        # Start time unreadable: alive iff the pid exists. Route through the cross-platform probe — on Windows
        # ``os.kill(pid, 0)`` is NOT a no-op (bpo-14484: it maps to ``GenerateConsoleCtrlEvent(0, pid)`` and could
        # Ctrl+C the gateway's own console group). ``_pid_exists`` keeps EPERM-means-alive (pid owned by another user).
        try:
            from gateway.status import _pid_exists
        except Exception:
            if os.name == "nt":
                return False  # never fall back to a raw sig-0 probe on Windows
            try:
                os.kill(pid, 0)  # windows-footgun: ok — POSIX-only fallback branch
                return True
            except OSError as exc:  # incl. ProcessLookupError; EPERM means the pid exists
                return isinstance(exc, PermissionError)
        try:
            return bool(_pid_exists(pid))
        except Exception:
            return False
    try:
        from gateway.status import start_time_fingerprints_match
        return started_at is None or start_time_fingerprints_match(started_at, current_start)
    except (TypeError, ValueError):
        return True


def compute_obligation_id(session_key: str, message_ref: str, content: str) -> str:
    """Stable id: same turn + same content re-records idempotently, while distinct threads/topics on one
    chat never collide (session_key carries platform/chat/thread; ``message_ref`` = inbound message id)."""
    return hashlib.sha256(f"{session_key}|{message_ref}|{content}".encode("utf-8", "replace")).hexdigest()[:24]


def inbound_answer_ref(recovery_id: str, run: Any = 0) -> str:
    """The ledger reference of the answer one run of an inbound record's message gave: the record id,
    with the run's ordinal when the message ran again after a restart. The ordinal is written into the
    record before the run is admitted (gateway/run_startup.py, _allocate_inbound_run) and stamped on the
    run's user row (``inbound_run``), so it is the same at delivery and at a later recovery, and a
    rerun's reply never takes over the obligation an earlier run's reply still holds."""
    return f"{recovery_id}#{run}" if isinstance(run, int) and run > 0 else str(recovery_id)


# What a reply is to the owner message it goes to (``REPLY`` on each of its stored components). An answer
# is the run's answer. A notice is what a run sends instead of one (an error or interruption notice): it has
# an obligation of its own, so it never stands in for the answer of the same message and run. A notice
# *terminates* the message when the run that sent it ended (delivered, the message owes nothing more); one
# whose run a restart cut short (``CUT_SHORT``) is *superseded*: the message stays owed until the answer
# its transcript holds, or a rerun's reply, is delivered, and the notice's obligation is kept as history.
REPLY = "reply"
REPLY_ANSWER = "answer"
REPLY_NOTICE = "notice"


def inbound_obligation_id(session_key: str, answer_ref: str, *, platform: str, chat_id: Any,
                          thread_id: Any, kind: str = REPLY_ANSWER) -> str:
    """Id of the reply owed to the owner message whose inbound record (gateway/shutdown_flush.py) and
    run *answer_ref* names (``inbound_answer_ref``): that, where the reply goes and what it is (*kind*:
    the answer, or a notice sent instead of it), never the reply's text. Normal delivery (platforms/
    base.py, _record_delivery_obligation) and startup recovery of an answered record (gateway/
    run_startup.py, _owe_stored_answer) agree on it even when the delivered text is not the transcript's
    (a runtime footer added at delivery), and two records answered alike stay two."""
    ref = f"inbound:{answer_ref}|{platform}|{chat_id}|{thread_id or ''}"
    return compute_obligation_id(session_key, ref if kind == REPLY_ANSWER else f"{ref}|{kind}", "")


@dataclasses.dataclass(frozen=True)
class InboundObligation:
    """What one run of one owner message owes one destination: its answer, or a notice the run sent
    instead (each under its own ``inbound_obligation_id``). The settlement boundary (gateway/
    run_inbound.py, _settle_inbound_records) and startup recovery read both through one decision,
    ``inbound_completion``. ``sent``: every component of the reply was sent successfully here, the only
    proof there is with the ledger off."""
    session_key: str
    answer_ref: str
    platform: str
    chat_id: Any
    thread_id: Any = None
    sent: bool = False

    @classmethod
    def to(cls, session_key: str, answer_ref: str, source: Any, **outcome: Any) -> "InboundObligation":
        return cls(session_key, str(answer_ref), str(getattr(source.platform, "value", source.platform)),
                   source.chat_id, getattr(source, "thread_id", None), **outcome)

    def reply_id(self, kind: Optional[str] = REPLY_ANSWER) -> str:
        """The obligation id of this run's reply of *kind* (``REPLY_ANSWER`` or ``REPLY_NOTICE``)."""
        return inbound_obligation_id(self.session_key, self.answer_ref, platform=self.platform,
                                     chat_id=self.chat_id, thread_id=self.thread_id, kind=kind or REPLY_ANSWER)

    @property
    def obligation_id(self) -> str:
        """The answer's obligation id."""
        return self.reply_id(REPLY_ANSWER)


def obligation_state(obligation_id: str) -> Optional[str]:
    """The obligation's state read back from the ledger (``pending``, ``attempting``, ``failed``,
    ``delivered``, ``abandoned``), or None when there is no such row. Raises when the ledger cannot be
    read: a failed read is never "no row"."""
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute("SELECT state FROM delivery_obligations WHERE obligation_id=?",
                           (obligation_id,)).fetchone()
    return None if row is None else str(row[0])


# A component's states: ``pending`` (not sent yet), ``failed`` (a send was refused), ``missing`` (its file was
# gone when a send was due), ``blocked`` (the media delivery policy refused its file when a send was due; it is
# never uploaded), ``delivered``. Only ``delivered`` is ever final: the check before each send (platforms/base.py,
# attachment_verdict) can move a component between the others.
COMPONENT_DELIVERED = "delivered"
COMPONENT_MISSING = "missing"
COMPONENT_BLOCKED = "blocked"
# Payload flag of every component of a reply sent by a run that did not end its message (a restart cut it
# short and its record stays for the next process): such a reply never settles the message by itself
# (inbound_completion). A notice of such a run is superseded; an answer is provisional until the next start
# finds the run's completed answer in its transcript (confirm_answer).
CUT_SHORT = "run_cut_short"
# Payload flag of a notice sent by a run that had already produced its answer (the agent stores it in the
# transcript) before the turn failed: that notice is superseded too, never completion evidence for the
# stored answer, which stays owed under the answer's own obligation.
ANSWER_STORED = "answer_stored"

# The answers of inbound_completion, the one decision on whether an owner message's reply is delivered.
COMPLETE = "complete"    # every component of the reply delivered: the record and marker may go
OWED = "owed"            # stored and not yet delivered: a component pending or refused, within its retries
MISSING = "missing"      # a required attachment's source is gone: blocks completion and deletion
BLOCKED = "blocked"      # the media policy refuses a required attachment: never uploaded; blocks completion, deletion
EXPIRED = "expired"      # the text's retry budget is spent (abandoned, attempts cap, stale) undelivered
OPEN = "open"            # only a reply of a run a restart cut short: the transcript decides what is owed
UNTRACKED = "untracked"  # a text row without components (written before components were): never complete
# How many later starts may each give an expired reply's text one more send (reactivate_expired), and each
# undelivered attachment of a reply one more try (claim_attachment_redelivery).
MAX_REACTIVATIONS = 3


def _components_unlocked(conn, obligation_id: str) -> List[Dict[str, Any]]:
    import json as _json
    rows = conn.execute(
        """SELECT component_id, kind, payload, state FROM delivery_components
           WHERE obligation_id=? ORDER BY rowid""", (obligation_id,)).fetchall()
    return [{**_json.loads(payload), "id": cid, "kind": kind, "state": state} for cid, kind, payload, state in rows]


def record_components(obligation_id: str, components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Store what the reply under *obligation_id* requires, before any of it is sent: each component
    (``id`` stable within the reply, ``kind`` text / image / media / file, and what its send needs: the
    text, the image URL, the file path), state ``pending``. Insert-only: once a reply's components are
    stored they are what the obligation requires, and a later call stores nothing. A text component
    starts ``delivered`` when the text row of the same id already is. Returns the components the
    obligation holds, with their states. Raises when the ledger cannot be written."""
    import json as _json
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        held = _components_unlocked(conn, obligation_id)
        if held or not components:
            return held
        text_delivered = conn.execute(
            "SELECT 1 FROM delivery_obligations WHERE obligation_id=? AND state='delivered'",
            (obligation_id,)).fetchone() is not None
        for part in components:
            payload = {k: v for k, v in part.items() if k not in ("id", "kind", "state")}
            conn.execute(
                """INSERT INTO delivery_components (obligation_id, component_id, kind, payload, state, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (obligation_id, str(part["id"]), str(part["kind"]), _json.dumps(payload, sort_keys=True),
                 COMPONENT_DELIVERED if part["kind"] == "text" and text_delivered else "pending", now))
        return _components_unlocked(conn, obligation_id)


class ReplyNotStored(RuntimeError):
    """What an owner message's reply requires could not be stored (``record_components`` failed). Nothing
    of the reply may then be sent: an untracked send can neither prove the reply delivered nor stop a
    later start sending it again. The message's record and the reply's source (its transcript) stay."""


def claim_attachment_redelivery(obligation_id: str) -> "tuple[List[Dict[str, Any]], List[Dict[str, Any]]]":
    """A later start's try at the attachments the reply under *obligation_id* still owes (pending, refused
    or ``missing``): each gets at most ``MAX_REACTIVATIONS`` such tries, the same bound as an expired
    text's reactivation, counted here before the try. Returns ``(claimed, spent)``: the components to try
    now, and those whose tries are spent; a spent one keeps its state (it still blocks completion, and
    the record stays) and is not tried again. Raises when the ledger cannot be written."""
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT component_id, redeliveries FROM delivery_components
               WHERE obligation_id=? AND kind != 'text' AND state != ?""",
            (obligation_id, COMPONENT_DELIVERED)).fetchall()
        claimed = {cid for cid, tries in rows if int(tries or 0) < MAX_REACTIVATIONS}
        for cid in claimed:
            conn.execute("""UPDATE delivery_components SET redeliveries=redeliveries+1
                            WHERE obligation_id=? AND component_id=?""", (obligation_id, cid))
        owed = {cid for cid, _tries in rows}
        parts = _components_unlocked(conn, obligation_id)
    return ([p for p in parts if p["id"] in claimed], [p for p in parts if p["id"] in owed - claimed])


def obligation_components(obligation_id: str) -> List[Dict[str, Any]]:
    """The components stored for *obligation_id* (``record_components``) with their states; ``[]`` when
    none were. Raises when the ledger cannot be read."""
    with _DB_LOCK, _transaction() as conn:
        return _components_unlocked(conn, obligation_id)


def mark_components(obligation_id: str, outcomes: Dict[str, str]) -> List[str]:
    """Record attachment outcomes (component id -> ``delivered`` / ``failed`` / ``missing`` / ``blocked``). A
    delivered component stays delivered, and the text component follows its row (``_update_state``), so
    neither is written here. Returns the ids whose state changed."""
    changed: List[str] = []
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        for component_id, state in outcomes.items():
            cursor = conn.execute(
                """UPDATE delivery_components SET state=?, updated_at=?
                   WHERE obligation_id=? AND component_id=? AND kind != 'text' AND state NOT IN (?, ?)""",
                (state, now, obligation_id, component_id, COMPONENT_DELIVERED, state))
            if cursor.rowcount:
                changed.append(component_id)
    return changed


def _retry_spent(attempts: Any, created_at: Any, now: float) -> bool:
    """A row's redelivery budget is spent: the attempts cap, or the stale cutoff. The sweeps abandon such a
    row; for an owner message's reply that is not the end (inbound_completion, reactivate_expired)."""
    return int(attempts or 0) >= MAX_ATTEMPTS or (now - float(created_at or 0.0)) > STALE_AFTER_SECONDS


def _text_expired_unlocked(conn, obligation_id: str, now: float) -> bool:
    row = conn.execute("SELECT state, attempts, created_at FROM delivery_obligations WHERE obligation_id=?",
                       (obligation_id,)).fetchone()
    return row is not None and (row[0] == "abandoned" or (row[0] != "delivered" and _retry_spent(row[1], row[2], now)))


def inbound_completion(obligation: InboundObligation, *, now: Optional[float] = None
                       ) -> "tuple[Optional[str], Optional[str]]":
    """The one decision on an owner message's reply, for one run and destination: ``(state, id)``, where
    *id* is the obligation the reply was decided on, its answer's when one is stored, else its notice's.
    Every caller reads it: the settlement boundary (gateway/run_inbound.py, _inbound_release_proven),
    which releases only on ``COMPLETE``, and startup recovery (gateway/run_startup.py, _owe_stored_answer).

    It weighs what the reply is (``REPLY``), what each component's state is, and the text's retry state:
    ``OPEN`` when the reply was sent by a run a restart cut short (``CUT_SHORT``: a superseded notice or a
    provisional answer, never completion by itself), or is a notice sent after the run had stored its
    answer (``ANSWER_STORED``: superseded by that answer); ``COMPLETE`` when every component is delivered
    (an answer, or with no answer stored a notice that terminated the message); else ``EXPIRED`` when the
    undelivered text's retry budget is spent, ``BLOCKED`` when the media policy refused an attachment,
    ``MISSING`` when an attachment's source is gone, ``OWED`` otherwise. ``UNTRACKED`` for a text row
    without components, ``(None, None)`` for nothing. Raises when the ledger cannot be read: a failed read
    is never an answer."""
    now = time.time() if now is None else now
    with _DB_LOCK, _transaction() as conn:
        for reply_id in (obligation.reply_id(REPLY_ANSWER), obligation.reply_id(REPLY_NOTICE)):
            parts = _components_unlocked(conn, reply_id)
            if parts:
                break
        else:
            answer_id = obligation.reply_id(REPLY_ANSWER)
            row = conn.execute("SELECT 1 FROM delivery_obligations WHERE obligation_id=?", (answer_id,)).fetchone()
            return (UNTRACKED, answer_id) if row is not None else (None, None)
        if any(part.get(CUT_SHORT) or (part.get(REPLY) == REPLY_NOTICE and part.get(ANSWER_STORED))
               for part in parts):
            return OPEN, reply_id
        owed = [part for part in parts if part["state"] != COMPONENT_DELIVERED]
        if not owed:
            return COMPLETE, reply_id
        if any(part["kind"] == "text" for part in owed) and _text_expired_unlocked(conn, reply_id, now):
            return EXPIRED, reply_id
        if any(part["state"] == COMPONENT_BLOCKED for part in owed):
            return BLOCKED, reply_id
        if any(part["state"] == COMPONENT_MISSING for part in owed):
            return MISSING, reply_id
        return OWED, reply_id


def confirm_answer(obligation_id: str) -> None:
    """The transcript holds the completed answer of the run whose provisional answer (a reply a restart
    cut short, ``CUT_SHORT``) is stored under *obligation_id*: that reply carried it, so it is the run's
    answer from now on and settles the message once delivered. A notice is never confirmed."""
    import json as _json
    with _DB_LOCK, _transaction() as conn:
        for part in _components_unlocked(conn, obligation_id):
            if part.get(CUT_SHORT) and part.get(REPLY, REPLY_ANSWER) == REPLY_ANSWER:
                payload = {k: v for k, v in part.items() if k not in ("id", "kind", "state", CUT_SHORT)}
                conn.execute("UPDATE delivery_components SET payload=? WHERE obligation_id=? AND component_id=?",
                             (_json.dumps(payload, sort_keys=True), obligation_id, part["id"]))


def reactivate_expired(obligation_id: str, *, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Bounded reactivation of an owner message's reply whose text spent its retry budget undelivered
    (``EXPIRED``): called by a later start, the signal that the fault may have cleared, while the message's
    record still owes the reply. The row is claimed for exactly one send by this process (``attempting``,
    attempts + 1, the duplicate marker) and its ``reactivations`` count goes up; the caller sends it like a
    boot claim. Never for a row that is not expired, belongs to a managed /acp answer, is held by a live
    process, whose target refused it for good (a dead chat: send-forbidden), or that was reactivated
    ``MAX_REACTIVATIONS`` times already: those stay as they are, with the record. Returns the claimed row,
    or None."""
    now = time.time() if now is None else now
    pid, started = _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            """SELECT session_key, platform, chat_id, thread_id, content, state, attempts, created_at,
                      owner_pid, owner_started_at, adapter_profile, last_error, reactivations
               FROM delivery_obligations WHERE obligation_id=? AND acp_update_id IS NULL""",
            (obligation_id,)).fetchone()
        if row is None or not _text_expired_unlocked(conn, obligation_id, now):
            return None
        (session_key, platform, chat_id, thread_id, content, state, attempts, _created, owner_pid, owner_started,
         profile, last_error, reactivations) = row
        if (int(reactivations or 0) >= MAX_REACTIVATIONS or classify_dead_error(str(last_error or ""))
                or (state != "abandoned" and _owner_alive(owner_pid, owner_started))):
            return None
        cursor = conn.execute(
            """UPDATE delivery_obligations
               SET state='attempting', attempts=attempts+1, updated_at=?, owner_pid=?, owner_started_at=?,
                   adapter_profile=COALESCE(adapter_profile, 'default'), last_error=NULL,
                   reactivations=reactivations+1
               WHERE obligation_id=? AND state=? AND attempts=?""",
            (now, pid, started, obligation_id, state, attempts))
        if not cursor.rowcount:
            return None
    logger.warning("delivery obligation %s to %s:%s spent its retry budget undelivered; reactivated for one more "
                   "send (%d of %d)", obligation_id, platform, chat_id, int(reactivations or 0) + 1, MAX_REACTIVATIONS)
    return _claimed_row(obligation_id, session_key, platform, chat_id, thread_id, content, attempts,
                        profile or "default", needs_marker=True, last_error=last_error)


def _mirror_text_delivered(conn, obligation_id: str, now: float) -> None:
    """The text component of *obligation_id* is delivered when its row is: same transaction."""
    conn.execute(
        """UPDATE delivery_components SET state=?, updated_at=?
           WHERE obligation_id=? AND component_id='text' AND state != ?
             AND EXISTS (SELECT 1 FROM delivery_obligations WHERE obligation_id=? AND state='delivered')""",
        (COMPONENT_DELIVERED, now, obligation_id, COMPONENT_DELIVERED, obligation_id))


class ObligationTaken(RuntimeError):
    """The obligation id already holds another reply (``record_obligation(replace=False)``)."""


def record_obligation(*, obligation_id: str, session_key: str, platform: str, chat_id: str,
                      thread_id: Optional[str], content: str, adapter_profile: Optional[str] = None,
                      acp_update_id: Optional[str] = None, replace: bool = True) -> None:
    """Record a final response as owed to the platform (state='pending'). ``replace=False`` is for an
    id that names the input, run and destination rather than the text (``inbound_obligation_id``): an
    existing row is never replaced, so a later reply (an error notice, a retry) cannot take over an
    earlier answer's row, delivered or not. The same text again leaves the row as it is; another text
    raises ``ObligationTaken`` and records nothing."""
    now, (pid, started) = time.time(), _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            f"""INSERT OR {'REPLACE' if replace else 'IGNORE'} INTO delivery_obligations
               (obligation_id, session_key, platform, chat_id, thread_id,
                content, state, attempts, created_at, updated_at,
                owner_pid, owner_started_at, adapter_profile, acp_update_id)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?, ?, ?)""",
            (obligation_id, session_key, platform, str(chat_id), str(thread_id) if thread_id else None,
             content, now, now, pid, started, str(adapter_profile).strip() if adapter_profile else "default",
             str(acp_update_id) if acp_update_id is not None else None))
        if not replace:
            held = conn.execute("SELECT content FROM delivery_obligations WHERE obligation_id=?",
                                (obligation_id,)).fetchone()
            if held is None or held[0] != content:
                raise ObligationTaken(f"obligation {obligation_id} already holds another reply")
        # Same transaction, same connection: the cron ledgers prune this way too
        # (cron/delivery_queue._prune_terminal_unlocked, cron/executions._prune_unlocked).
        _prune_unlocked(conn, now)


def record_delivered(*, obligation_id: str, session_key: str, platform: str, chat_id: str,
                     thread_id: Optional[str], content: str, adapter_profile: Optional[str] = None) -> None:
    """Record a reply that reached the chat without a ledgered send (a stream whose final delivery
    the consumer confirmed) as 'delivered' under *obligation_id*. An existing row is left as it is;
    the text component follows whatever the row then reads."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO delivery_obligations
               (obligation_id, session_key, platform, chat_id, thread_id,
                content, state, attempts, created_at, updated_at, adapter_profile)
               VALUES (?, ?, ?, ?, ?, ?, 'delivered', 1, ?, ?, ?)""",
            (obligation_id, session_key, platform, str(chat_id), str(thread_id) if thread_id else None,
             content, now, now, str(adapter_profile).strip() if adapter_profile else "default"))
        _mirror_text_delivered(conn, obligation_id, now)
        _prune_unlocked(conn, now)


def record_crash_left_reply(*, obligation_id: str, session_key: str, platform: str, chat_id: str,
                            thread_id: Optional[str], content: str, since: float,
                            adapter_profile: Optional[str] = None, same_reply_dedupe: bool = True) -> None:
    """Adopt a reply a killed process persisted but never ledgered. Unowned, so this boot's sweep
    claims it, and 'attempting', because a streamed reply may already be on screen: it is
    redelivered once, with the recovered marker. A no-op when the same reply was already ledgered
    since *since* (the turn start), and idempotent across boots that die before their sweep. A managed
    /acp answer's row is never "the same reply": an ordinary reply with identical text is its own
    delivery. ``same_reply_dedupe=False`` is for a caller whose *obligation_id* already names the one
    message the reply answers (an inbound record's recovery id): only that id is deduplicated, so two
    messages answered with the same text stay two deliveries."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO delivery_obligations
               (obligation_id, session_key, platform, chat_id, thread_id,
                content, state, attempts, created_at, updated_at,
                owner_pid, owner_started_at, adapter_profile)
               SELECT ?, ?, ?, ?, ?, ?, 'attempting', 0, ?, ?, NULL, NULL, ?
               WHERE ? OR NOT EXISTS (SELECT 1 FROM delivery_obligations
                                      WHERE session_key = ? AND content = ? AND created_at >= ?
                                        AND acp_update_id IS NULL)""",
            (obligation_id, session_key, platform, str(chat_id), str(thread_id) if thread_id else None,
             content, now, now, str(adapter_profile).strip() if adapter_profile else "default",
             0 if same_reply_dedupe else 1, session_key, content, since))


def mark_attempting(obligation_id: str) -> None:
    _update_state(obligation_id, "attempting")


def mark_delivered(obligation_id: str, *, attempt: Optional[int] = None,
                   message_ids: Optional[List[str]] = None) -> bool:
    return _update_state(obligation_id, "delivered", attempt=attempt,
                         delivered_message_ids=json.dumps([str(m) for m in message_ids]) if message_ids else None)


def mark_failed(obligation_id: str, error: str = "", *, attempt: Optional[int] = None) -> bool:
    return _update_state(obligation_id, "failed", error=error, attempt=attempt)


def release_runtime_claim(obligation_id: str, error: str = "", *, attempt: Optional[int] = None,
                          pending: bool = False) -> bool:
    """Return an unsent claim (runtime or boot) to ``failed`` without spending an attempt.

    Runtime recovery claims before clearing ``resume_pending`` so two reconnect paths cannot send the
    same row; if the flag cannot be cleared no send was attempted and the claim must not consume the
    redelivery budget. Fail-closed to the exact current process instance and ``attempting`` state;
    ``attempt`` (the claimed row's ``attempts``) narrows it to that one claim, as for settlement.
    ``pending`` returns a boot claim taken from ``pending`` as it was, so the next boot sends it without
    the duplicate marker; only for a process that is exiting, as no sweep of a live one claims ``pending``."""
    pid, started = _owner_stamp()
    if started is None:
        return False
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """UPDATE delivery_obligations
               SET state=?, attempts=CASE
                       WHEN attempts > 0 THEN attempts - 1 ELSE 0 END,
                   updated_at=?, last_error=?
               WHERE obligation_id=? AND state='attempting'
                 AND owner_pid IS ? AND owner_started_at IS ? AND (? IS NULL OR attempts=?)""",
            ("pending" if pending else "failed", time.time(), error[:500] if error and not pending else None,
             obligation_id, pid, started, attempt, attempt))
    return bool(cursor.rowcount)


def _update_state(obligation_id: str, state: str, error: str = "", *, attempt: Optional[int] = None,
                  delivered_message_ids: Optional[str] = None) -> bool:
    """Write a checkpoint; False when nothing matched. With ``attempt`` it settles ONE claim: the row must
    still be ``attempting`` under this pid with that ``attempts`` count (every claim bumps it and re-stamps
    the owner). A claimant whose row was claimed since, by another process or by a later claim here,
    matches nothing, so its stale outcome cannot overwrite the newer claim's (a delivered row turned back
    to failed is sent again). The pid alone is enough: the settler is the live claimant, and its claim
    re-stamped the row with its own pid and that count, so a start-time compare would add nothing."""
    pid = _owner_stamp()[0] if attempt is not None else None
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """UPDATE delivery_obligations
               SET state=?, updated_at=?, last_error=?,
                   delivered_message_ids=COALESCE(?, delivered_message_ids)
               WHERE obligation_id=?
                 AND (? IS NULL OR (state='attempting' AND owner_pid IS ? AND attempts=?))""",
            (state, now, error[:500] if error else None, delivered_message_ids,
             obligation_id, attempt, pid, attempt))
        if cursor.rowcount and state == "delivered":
            _mirror_text_delivered(conn, obligation_id, now)
    return bool(cursor.rowcount)


def _log_abandoned(oid, platform, chat_id, state, attempts, created_at, now) -> None:
    """A reply given up on is logged at ERROR: nothing sends it again and nobody else is told."""
    logger.error("delivery obligation %s to %s:%s abandoned in state %s after %d attempt(s), %.1f h after it "
                 "was recorded: the reply was never confirmed delivered and is not sent again",
                 oid, platform, chat_id, state, attempts, (now - created_at) / 3600)


def _claimed_row(oid, session_key, platform, chat_id, thread_id, content, attempts, profile, *,
                 needs_marker: bool, runtime: bool = False, flood: bool = False,
                 last_error: Optional[str] = None) -> Dict[str, Any]:
    """Claimed-row dict handed back for redelivery. A marked row names its own cause: ``flood`` (a reply
    the rate limit refused, possibly after accepting part of it) gets FLOOD_MARKER at boot or at runtime, a
    ``runtime`` reconnect replay gets RECONNECTED_MARKER, and a boot-recovered crash keeps the runner's
    restart marker default. ``last_error`` is the row's pre-claim error, carried so a claim that is
    released unsent goes back to ``failed`` with the same error and keeps its retry eligibility.
    ``attempts`` is the post-claim count: with the owner pid it names this claim when it is settled."""
    marker = FLOOD_MARKER if flood else (RECONNECTED_MARKER if runtime else None)
    return {"obligation_id": oid, "session_key": session_key, "platform": platform, "chat_id": chat_id,
            "thread_id": thread_id, "content": content, "needs_marker": needs_marker,
            **({"marker": marker} if needs_marker and marker else {}), "profile": profile,
            **({"runtime_recovery": True} if runtime else {}),
            **({"last_error": last_error} if last_error else {}), "attempts": attempts + 1}


def sweep_recoverable(now: Optional[float] = None, *, deliverable_platforms: Optional[set] = None,
                      deliverable_targets: Optional[set] = None) -> List[Dict[str, Any]]:
    """Claim undelivered rows owned by dead processes; return them for redelivery.

    Claiming atomically re-stamps the owner to THIS process, moves the row to 'attempting' and increments
    ``attempts`` (the UPDATE is guarded on the previous owner stamp, so a second gateway racing the same
    sweep cannot double-claim).
    Rows over the attempts cap or stale cutoff become 'abandoned'. ``deliverable_platforms`` restricts
    claiming to platforms the caller can send on this boot: ``attempts`` is the redelivery budget and
    must only be spent on a real send, else a platform that failed to connect burns one attempt per boot
    and hits the cap having never been sent once (the stale cutoff still bounds untouched rows).
    ``deliverable_targets`` further scopes multiplexed gateways by exact ``(platform, adapter_profile)``
    so one connected bot cannot spend another disconnected bot's retry budget.

    A flood-refused row still inside its wait is adopted (owner re-stamped, no attempt spent) and
    returned flagged ``adopted`` with its ``not_before``: the caller clears its session's resume flag
    like any other claimed row, since the answer is in the ledger, but must not send it; the flood
    timer does once the wait has passed. A legacy row without ``adapter_profile`` is normalised to
    ``'default'`` on claim or adoption (the caller only accepts such rows when it is not multiplexed),
    because the runtime sweep matches profiles exactly and could otherwise never claim it."""
    now, (pid, started) = now if now is not None else time.time(), _owner_stamp()
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, state, attempts, created_at,
                      owner_pid, owner_started_at, adapter_profile, last_error, updated_at
               FROM delivery_obligations
               WHERE state IN ('pending', 'attempting', 'failed') AND acp_update_id IS NULL"""
        ).fetchall()
        for (oid, session_key, platform, chat_id, thread_id, content, state, attempts, created_at,
             owner_pid, owner_started_at, adapter_profile, last_error, updated_at) in rows:
            if _owner_alive(owner_pid, owner_started_at):
                continue  # a live gateway still owns this row
            if _retry_spent(attempts, created_at, now):  # exhausted -> abandoned
                conn.execute(
                    """UPDATE delivery_obligations
                       SET state='abandoned', updated_at=? WHERE obligation_id=?""", (now, oid))
                _log_abandoned(oid, platform, chat_id, state, attempts, created_at, now)
                continue
            if ((deliverable_platforms is not None and platform not in deliverable_platforms)
                    or (deliverable_targets is not None and (platform, adapter_profile) not in deliverable_targets)):
                continue  # no adapter this boot — claiming would spend an attempt on a no-op
            flood_row = state == "failed" and is_flood_error(last_error)
            if flood_row and now < flood_not_before(updated_at, last_error):
                # Still inside the platform's wait: adopt the dead owner's row without spending an attempt
                # (state and error kept) so this process's flood timer can claim it once the wait passes.
                cursor = conn.execute(
                    """UPDATE delivery_obligations
                       SET owner_pid=?, owner_started_at=?,
                           adapter_profile=COALESCE(adapter_profile, 'default')
                       WHERE obligation_id=? AND (owner_pid IS ? OR owner_pid=?)""",
                    (pid, started, oid, owner_pid, owner_pid))
                if cursor.rowcount:
                    claimed.append({
                        "obligation_id": oid, "session_key": session_key, "platform": platform,
                        "chat_id": chat_id, "thread_id": thread_id, "content": content,
                        "profile": adapter_profile or "default", "attempts": attempts,
                        "adopted": True, "not_before": flood_not_before(updated_at, last_error)})
                continue
            # Every claim starts a send, so the row leaves 'pending'/'failed' for 'attempting' in the same
            # CAS, as a fresh attempt held by this process with no error: a boot killed inside the
            # redelivery then leaves proof the platform may have it (next boot marks it), and the runtime
            # sweep, which only takes 'failed', cannot re-claim it while the boot send is in flight.
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET owner_pid=?, owner_started_at=?, attempts=attempts+1, updated_at=?,
                       adapter_profile=COALESCE(adapter_profile, 'default'),
                       state='attempting', last_error=NULL
                   WHERE obligation_id=? AND (owner_pid IS ? OR owner_pid=?)""",
                (pid, started, now, oid, owner_pid, owner_pid))
            if cursor.rowcount:
                # A never-claimed pending row was never sent: redeliver plainly. Anything else (crashed
                # mid-await, other rejection, a flood refusal whose earlier chunks the platform may have
                # accepted, or a pending row an older build already claimed and may have sent) carries
                # the marker. The pre-claim error rides along for a claim released unsent.
                claimed.append(_claimed_row(oid, session_key, platform, chat_id, thread_id, content, attempts,
                                            adapter_profile or "default",
                                            needs_marker=state != "pending" or attempts > 0,
                                            flood=flood_row, last_error=last_error))
    return claimed


def sweep_failed_for_runtime(platform: str, now: Optional[float] = None, *,
                             profile: Optional[str] = None) -> List[Dict[str, Any]]:
    """Claim this process's failed rows that are due for another send, for one adapter.

    ``profile`` scopes multiplexed gateways to the bot identity that owned the failed send (``None`` =
    primary/default adapter); unowned rows and rows owned by another process are left for the
    startup/dead-owner sweep. Startup recovery ignores rows owned by a live gateway, so a response
    rejected with ``send_path_degraded`` would stay stranded when only the adapter reconnects; this closes
    that gap without weakening ownership: only rows stamped to this exact process instance, only rows
    past their ``retry_not_before`` deadline, same attempts/staleness bounds, every update guarded by the
    prior owner stamp and ``failed`` state. Claimed rows always carry a marker (the failed send's ack is not safe to
    infer): the reconnect one, or the rate-limit one for a flood-refused row."""
    now, (pid, started) = now if now is not None else time.time(), _owner_stamp()
    if started is None:  # PID alone cannot distinguish this process from a stale row left after PID
        return []        # reuse; runtime replay is optional, so fail closed (startup recovery remains).
    expected_profile = "default" if not profile or profile == "default" else str(profile)
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, attempts, created_at, owner_pid,
                      owner_started_at, last_error, adapter_profile, updated_at
               FROM delivery_obligations
               WHERE state='failed' AND platform=? AND acp_update_id IS NULL""", (platform,)).fetchall()
        for (oid, session_key, row_platform, chat_id, thread_id, content, attempts, created_at,
             owner_pid, owner_started_at, last_error, adapter_profile, updated_at) in rows:
            # Exact process-start matching prevents PID reuse from stealing work.
            if adapter_profile != expected_profile or owner_pid != pid or owner_started_at != started:
                continue
            due = retry_not_before(updated_at, last_error, attempts)
            if due is None:
                continue
            owner_guard = (now, oid, owner_pid, owner_started_at)
            if _retry_spent(attempts, created_at, now):  # exhausted -> abandoned
                if conn.execute(
                        """UPDATE delivery_obligations
                           SET state='abandoned', updated_at=?
                           WHERE obligation_id=? AND state='failed'
                             AND owner_pid IS ? AND owner_started_at IS ?""", owner_guard).rowcount:
                    _log_abandoned(oid, row_platform, chat_id, "failed", attempts, created_at, now)
                continue
            if now < due:
                continue  # the platform's wait or the backoff has not passed; the timer comes back for it
            # The claim clears the stale error: this is a fresh attempt, and if it is interrupted the next
            # boot must see 'attempting' with no proof of non-delivery, hence the marker.
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET state='attempting', attempts=attempts+1, updated_at=?, last_error=NULL
                   WHERE obligation_id=? AND state='failed'
                     AND owner_pid IS ? AND owner_started_at IS ?""", owner_guard)
            if cursor.rowcount:
                # The failed send's ack may have been lost (reconnect) or its earlier chunks accepted
                # (flood): every runtime redelivery carries a marker. The pre-claim error rides along so a
                # claim released unsent keeps its flood retry eligibility.
                claimed.append(_claimed_row(oid, session_key, row_platform, chat_id, thread_id, content,
                                            attempts, adapter_profile, needs_marker=True, runtime=True,
                                            flood=is_flood_error(last_error), last_error=last_error))
    return claimed


def pending_retries(now: Optional[float] = None) -> List[Dict[str, Any]]:
    """This process's failed rows that still await redelivery, one entry per adapter identity with the
    earliest deadline (``not_before``). The runner arms one redelivery timer per entry, so a row adopted
    at boot, skipped because its wait had not passed, or rejected again is never stranded. Rows past the
    attempts cap or stale cutoff are left for the sweeps to abandon."""
    now, (pid, started) = now if now is not None else time.time(), _owner_stamp()
    if started is None:
        return []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT platform, adapter_profile, updated_at, last_error, attempts, created_at
               FROM delivery_obligations
               WHERE state='failed' AND owner_pid IS ? AND owner_started_at IS ? AND acp_update_id IS NULL""",
            (pid, started)).fetchall()
    earliest: Dict[tuple, float] = {}
    for platform, adapter_profile, updated_at, last_error, attempts, created_at in rows:
        # Reconnect-only rows (a claim released because the adapter was gone) are re-claimed by the
        # reconnect sweep; a timer would claim and release them every tick until the adapter is back.
        if is_reconnect_only(last_error):
            continue
        due = retry_not_before(updated_at, last_error, attempts)
        if due is None or _retry_spent(attempts, created_at, now):
            continue
        key = (platform, adapter_profile or "default")
        if key not in earliest or due < earliest[key]:
            earliest[key] = due
    return [{"platform": platform, "profile": profile, "not_before": due}
            for (platform, profile), due in sorted(earliest.items())]


_NOT_OWED = """NOT EXISTS (SELECT 1 FROM delivery_components c
                        WHERE c.obligation_id = delivery_obligations.obligation_id AND c.state != 'delivered')"""


def _prune_unlocked(conn, now: float) -> None:
    """Retention DELETEs on the caller's open connection — must run inside the caller's transaction."""
    # An /acp answer row is the evidence its receipt settles from (and the startup sweep reads before
    # declaring a death before answer), so it outlives the ordinary retention and the row cap. The row of a
    # reply whose components are still owed is kept too (abandoned or not): it is that reply's retry state
    # and its source, which a later start reactivates (reactivate_expired).
    conn.execute(
        f"""DELETE FROM delivery_obligations
           WHERE state IN ('delivered', 'abandoned') AND updated_at < ?
             AND (acp_update_id IS NULL OR updated_at < ?) AND {_NOT_OWED}""",
        (now - _RETENTION_SECONDS, now - _ACP_RETENTION_SECONDS))
    # A reply's components go together, once every one has been delivered for the retention period; one
    # still owed keeps them all.
    conn.execute(
        """DELETE FROM delivery_components WHERE obligation_id IN (
             SELECT obligation_id FROM delivery_components GROUP BY obligation_id
             HAVING MIN(state = 'delivered') = 1 AND MAX(updated_at) < ?)""", (now - _RETENTION_SECONDS,))
    total = conn.execute("SELECT COUNT(*) FROM delivery_obligations WHERE acp_update_id IS NULL").fetchone()[0]
    if total > _MAX_ROWS:
        conn.execute(
            f"""DELETE FROM delivery_obligations WHERE obligation_id IN (
                 SELECT obligation_id FROM delivery_obligations WHERE acp_update_id IS NULL AND {_NOT_OWED}
                 ORDER BY CASE state
                            WHEN 'delivered' THEN 0
                            WHEN 'abandoned' THEN 1
                            ELSE 2
                          END, updated_at ASC
                 LIMIT ?)""", (total - _MAX_ROWS,))


def ledger_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Read the ``gateway.delivery_ledger`` config gate (default on)."""
    try:
        if config is None:
            from hermes_cli.config import load_config
            config = load_config()
        value = (config.get("gateway") or {}).get("delivery_ledger", True)
        return value.strip().lower() not in {"false", "0", "no", "off"} if isinstance(value, str) else bool(value)
    except Exception:
        return True


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
import json  # noqa: F401,E402

def debug_rows(limit: int = 20) -> str:
    """Human-readable dump for ad-hoc inspection (sqlite3-free path)."""
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, state, attempts,
                      created_at, updated_at, last_error
               FROM delivery_obligations
               ORDER BY updated_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return json.dumps(
        [
            {
                "id": r[0], "session": r[1], "state": r[2], "attempts": r[3],
                "created_at": r[4], "updated_at": r[5], "last_error": r[6],
            }
            for r in rows
        ],
        indent=2,
    )
# ---- END PLUGIN-COMPAT ----
