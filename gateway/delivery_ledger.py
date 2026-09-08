"""Durable delivery-obligation ledger for gateway final responses.

A final agent response that was generated but not yet confirmed-delivered
to the messaging platform is the one artifact the gateway can lose without
a trace: the turn already burned its tokens, the text exists only in a
Python local, and a crash / planned restart between finalize and platform
ACK drops it silently (#58818, #41696, #63695).

This module records a small durable row per outbound final response in the
shared ``state.db`` (same file and conventions as
``tools.async_delegation`` — WAL, owner pid + process-start-time liveness,
bounded retention). The gateway writes three checkpoints around the send:

    record_obligation()   state='pending'     before any send attempt
    mark_attempting()     state='attempting'  immediately before the await
    mark_delivered() /    state='delivered'   only on SendResult.success
    mark_failed()         state='failed'      on a definitive rejection

On startup, ``sweep_recoverable()`` claims rows whose owning process is
dead and hands them to the gateway for redelivery. Crash semantics are
explicit about ambiguity (the contract review of the earlier
delivery-outbox attempt, #61790, closed it for silently resending
ambiguous sends):

- ``pending``     — the send never started: redeliver plainly, no dup risk.
- ``attempting``  — crashed mid-await: the platform MAY already have the
  message. Redelivered WITH a visible recovered-reply marker so the
  contract is honest at-least-once, never a silent duplicate.
- ``failed``      — definitively rejected once; the restart is a natural
  retry boundary. Also carries the marker.
- ``delivered``   — nothing to do; retention prunes.

Poison rows cannot spin: attempts are capped, stale rows expire, and both
transition to ``abandoned`` (kept briefly for inspection, then pruned).

Everything here is best-effort by design: ledger failures must never block
or delay an actual send. Callers wrap every call in try/except.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DB_LOCK = threading.Lock()

# Redelivery policy knobs (module constants; deliberately not config — the
# ledger itself is gated by ``gateway.delivery_ledger`` and these bounds
# only matter in the rare recovery path).
MAX_ATTEMPTS = 3
STALE_AFTER_SECONDS = 24 * 60 * 60
_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_ROWS = 500

# Visible prefix for redeliveries that might duplicate an already-received
# message (crash mid-send / post-rejection retry). Honest at-least-once.
RECOVERED_MARKER = (
    "♻️ Recovered reply — the gateway restarted during delivery, "
    "so this may be a duplicate:\n\n"
)


# Runtime replay is deliberately fail-closed. Only errors whose send contract
# proves they are transient reconnect failures belong here; permanent rejects
# must not be retried merely because an adapter reconnected.
_RUNTIME_RETRYABLE_ERRORS = frozenset({"send_path_degraded"})

# Runtime recovery uses a distinct marker because no gateway restart occurred.
RECONNECTED_MARKER = (
    "♻️ Recovered reply — the messaging platform reconnected after the original "
    "delivery failed, so this may be a duplicate:\n\n"
)


def _runtime_adapter_profile(adapter_profile: Any) -> str:
    """Canonicalize only legacy blank profile values at runtime row handoff."""
    if adapter_profile is None:
        return "default"
    profile = str(adapter_profile)
    return profile if profile.strip() else "default"


def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        _initialize_schema(conn)
    except Exception:
        # A PRAGMA/DDL failure after a successful connect() must not leak the
        # just-opened connection back to the caller.
        conn.close()
        raise
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import (
        _assert_offline_rebuild_maintenance_authority,
        _assert_offline_rebuild_write_authority,
        apply_wal_with_fallback,
    )

    apply_wal_with_fallback(
        conn,
        db_label="state.db (delivery_ledger)",
        before_journal_mode_change=lambda: _assert_offline_rebuild_maintenance_authority(
            conn, local_marker=None
        ),
    )
    conn.execute("BEGIN IMMEDIATE")
    try:
        _assert_offline_rebuild_write_authority(conn, local_marker=None)
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
                adapter_profile TEXT NOT NULL DEFAULT 'default',
                runtime_claim_token TEXT
            )"""
        )
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(delivery_obligations)")
        }
        if "adapter_profile" not in columns:
            try:
                # Legacy rows predate multiplexing, so their only deterministic
                # transport owner is the default adapter profile.  Do not infer
                # ownership from a routed session profile.
                conn.execute(
                    "ALTER TABLE delivery_obligations ADD COLUMN "
                    "adapter_profile TEXT NOT NULL DEFAULT 'default'"
                )
            except sqlite3.OperationalError as exc:
                # Concurrent first-use connections can both see the old schema.
                if "duplicate column" not in str(exc).lower():
                    raise
        if "runtime_claim_token" not in columns:
            try:
                conn.execute(
                    "ALTER TABLE delivery_obligations ADD COLUMN "
                    "runtime_claim_token TEXT"
                )
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise
        # Some deployments acquired the column before its default/non-null
        # contract existed. Treat their NULL and blank values as the legacy
        # default profile without changing an explicit multiplex identity.
        conn.execute(
            """UPDATE delivery_obligations
               SET adapter_profile='default'
               WHERE adapter_profile IS NULL OR trim(adapter_profile)=''"""
        )
        conn.execute(
            """UPDATE delivery_obligations
               SET runtime_claim_token=NULL
               WHERE runtime_claim_token IS NOT NULL
                 AND (state != 'attempting' OR trim(runtime_claim_token)='')"""
        )
        conn.execute("COMMIT")
    except BaseException as exc:
        if conn.in_transaction:
            try:
                conn.execute("ROLLBACK")
            except BaseException as rollback_exc:
                try:
                    exc.add_note(
                        f"delivery ledger schema rollback failed: {rollback_exc}"
                    )
                except Exception:
                    pass
        raise


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, and ALWAYS close it.

    ``sqlite3.Connection.__enter__``/``__exit__`` only commit or roll back the
    transaction; they do not close the connection. Using ``with _connect()``
    alone therefore leaks a connection — and its WAL/SHM file descriptors — on
    every call, deferring the close to the garbage collector. On a long-running
    gateway that exhausts ``RLIMIT_NOFILE`` (the cron-ledger sibling of this
    bug was #69567 / PR #69594). ``record_obligation`` runs on every outbound
    final response, so this ledger is the highest-frequency leaker.
    """
    conn = _connect()
    try:
        with conn:
            # A reserved write transaction makes the authority read and all
            # following ledger DML one ownership boundary: another owner
            # cannot install the durable rebuild marker between them.
            conn.execute("BEGIN IMMEDIATE")
            from hermes_state import _assert_offline_rebuild_write_authority

            _assert_offline_rebuild_write_authority(conn, local_marker=None)
            yield conn
    finally:
        conn.close()


def _owner_stamp() -> tuple[int, Optional[int]]:
    pid = os.getpid()
    try:
        from gateway.status import get_process_start_time

        return pid, get_process_start_time(pid)
    except Exception:
        return pid, None


def _owner_alive(pid: Any, started_at: Any) -> bool:
    """True when the recorded owning process still exists (pid + start time)."""
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        from gateway.status import get_process_start_time

        current_start = get_process_start_time(pid)
    except Exception:
        current_start = None
    if current_start is None:
        # No such process (or unreadable) — treat unreadable-but-extant
        # processes as alive only if the pid exists. Route through the
        # cross-platform probe: ``os.kill(pid, 0)`` on Windows is NOT a
        # no-op (bpo-14484 — CPython maps sig=0 to
        # ``GenerateConsoleCtrlEvent(0, pid)``), so a raw probe here could
        # Ctrl+C the gateway's own console group whenever psutil failed to
        # read the start time of a live pid. ``_pid_exists`` keeps the
        # EPERM-means-alive semantics (exists but owned by another user).
        try:
            from gateway.status import _pid_exists
        except Exception:
            if os.name == "nt":
                # Never fall back to a raw sig-0 probe on Windows.
                return False
            try:
                os.kill(pid, 0)  # windows-footgun: ok — POSIX-only fallback branch
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            except OSError:
                return False
            return True
        try:
            return bool(_pid_exists(pid))
        except Exception:
            return False
    if started_at is None:
        return True
    try:
        return int(current_start) == int(started_at)
    except (TypeError, ValueError):
        return True


def compute_obligation_id(session_key: str, message_ref: str, content: str) -> str:
    """Stable id: same turn + same content re-records idempotently, while
    distinct threads/topics on the same chat can never collide (the
    session_key carries platform, chat and thread; ``message_ref`` is the
    triggering inbound message id, distinguishing turns in one session)."""
    payload = f"{session_key}|{message_ref}|{content}"
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:24]


def record_obligation(
    *,
    obligation_id: str,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    content: str,
    adapter_profile: Optional[str] = None,
) -> None:
    """Record a final response as owed to the owning adapter profile."""
    now = time.time()
    stored_profile = str(adapter_profile or "").strip() or "default"
    pid, started = _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO delivery_obligations
               (obligation_id, session_key, platform, chat_id, thread_id,
                content, state, attempts, created_at, updated_at,
                owner_pid, owner_started_at, adapter_profile)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?, ?)""",
            (obligation_id, session_key, platform, str(chat_id),
             str(thread_id) if thread_id else None, content, now, now,
             pid, started, stored_profile),
        )
    _prune()


def mark_attempting(obligation_id: str) -> None:
    _update_state(obligation_id, "attempting")


def mark_delivered(obligation_id: str) -> None:
    _update_state(obligation_id, "delivered")


def mark_failed(obligation_id: str, error: str = "") -> None:
    _update_state(obligation_id, "failed", error=error)


def _valid_runtime_claim_token(claim_token: Any) -> bool:
    return (
        isinstance(claim_token, str)
        and len(claim_token) >= 32
        and claim_token.isascii()
        and claim_token.strip() == claim_token
    )


def _mint_runtime_claim_token() -> str:
    """Return a fresh opaque authority for exactly one runtime claim."""
    return secrets.token_urlsafe(32)


def release_runtime_claim(
    obligation_id: str, *, claim_token: str, error: str = ""
) -> bool:
    """Return this exact unsent runtime claim to ``failed`` atomically."""
    if not _valid_runtime_claim_token(claim_token):
        return False
    pid, started = _owner_stamp()
    if started is None:
        return False
    retryable_errors = tuple(sorted(_RUNTIME_RETRYABLE_ERRORS))
    placeholders = ", ".join("?" for _ in retryable_errors)
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            f"""UPDATE delivery_obligations
                SET state='failed',
                    attempts=CASE WHEN attempts > 0 THEN attempts - 1 ELSE 0 END,
                    updated_at=?, last_error=COALESCE(?, last_error),
                    runtime_claim_token=NULL
                WHERE obligation_id=? AND state='attempting'
                  AND owner_pid IS ? AND owner_started_at IS ?
                  AND runtime_claim_token=?
                  AND lower(trim(COALESCE(last_error, ''))) IN ({placeholders})""",
            (
                time.time(),
                error[:500] if error else None,
                obligation_id,
                pid,
                started,
                claim_token,
                *retryable_errors,
            ),
        )
    return bool(cursor.rowcount)


def _settle_runtime_claim(
    obligation_id: str, *, claim_token: str, state: str, error: Optional[str]
) -> bool:
    """Settle only this process's exact still-current runtime claim."""
    if not _valid_runtime_claim_token(claim_token):
        return False
    pid, started = _owner_stamp()
    if started is None:
        return False
    retryable_errors = tuple(sorted(_RUNTIME_RETRYABLE_ERRORS))
    placeholders = ", ".join("?" for _ in retryable_errors)
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            f"""UPDATE delivery_obligations
                SET state=?, updated_at=?, last_error=?, runtime_claim_token=NULL
                WHERE obligation_id=? AND state='attempting'
                  AND owner_pid IS ? AND owner_started_at IS ?
                  AND runtime_claim_token=?
                  AND lower(trim(COALESCE(last_error, ''))) IN ({placeholders})""",
            (
                state,
                time.time(),
                error[:500] if error else None,
                obligation_id,
                pid,
                started,
                claim_token,
                *retryable_errors,
            ),
        )
    return bool(cursor.rowcount)


def settle_runtime_claim(
    obligation_id: str, *, claim_token: str, delivered: bool, error: str = ""
) -> bool:
    """Atomically settle this exact runtime claim's normal send result."""
    return _settle_runtime_claim(
        obligation_id,
        claim_token=claim_token,
        state="delivered" if delivered else "failed",
        error=None if delivered else error,
    )


def settle_runtime_claim_after_send_started(
    obligation_id: str, *, claim_token: str
) -> bool:
    """Fence an ambiguous runtime send back to failed without refunding it."""
    return _settle_runtime_claim(
        obligation_id,
        claim_token=claim_token,
        state="failed",
        error="send_path_degraded",
    )


def _update_state(obligation_id: str, state: str, error: str = "") -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE delivery_obligations
               SET state=?, updated_at=?, last_error=?,
                   runtime_claim_token=CASE
                       WHEN ?='attempting' THEN runtime_claim_token ELSE NULL END
               WHERE obligation_id=?""",
            (
                state,
                time.time(),
                error[:500] if error else None,
                state,
                obligation_id,
            ),
        )


def sweep_recoverable(
    now: Optional[float] = None,
    *,
    deliverable_platforms: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Claim undelivered rows owned by dead processes; return them for
    redelivery.

    Claiming atomically re-stamps the owner to THIS process and increments
    ``attempts``, so a second gateway racing the same sweep cannot
    double-claim (the UPDATE is guarded on the previous owner stamp).
    Rows over the attempts cap or older than the stale cutoff transition to
    'abandoned' instead of being returned.

    ``deliverable_platforms`` (platform value strings) restricts claiming to
    platforms the caller can actually send on this boot.  ``attempts`` is the
    redelivery budget, so it must only be spent on a real send: a platform
    that failed to connect would otherwise burn one attempt per boot and hit
    the cap having never been sent once.  Rows for absent platforms are left
    untouched for a later boot; the stale cutoff still bounds them.
    """
    now = now if now is not None else time.time()
    pid, started = _owner_stamp()
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, state, attempts, created_at,
                      owner_pid, owner_started_at
               FROM delivery_obligations
               WHERE state IN ('pending', 'attempting', 'failed')"""
        ).fetchall()
        for (oid, session_key, platform, chat_id, thread_id, content, state,
             attempts, created_at, owner_pid, owner_started_at) in rows:
            if _owner_alive(owner_pid, owner_started_at):
                continue  # a live gateway still owns this row
            if attempts >= MAX_ATTEMPTS or (now - created_at) > STALE_AFTER_SECONDS:
                conn.execute(
                    """UPDATE delivery_obligations
                       SET state='abandoned', updated_at=?, runtime_claim_token=NULL
                       WHERE obligation_id=?""",
                    (now, oid),
                )
                continue
            if (
                deliverable_platforms is not None
                and platform not in deliverable_platforms
            ):
                # No adapter for this platform this boot — the caller cannot
                # send, so claiming would spend an attempt on a no-op.
                continue
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET owner_pid=?, owner_started_at=?, attempts=attempts+1,
                       updated_at=?, runtime_claim_token=NULL
                   WHERE obligation_id=? AND (owner_pid IS ? OR owner_pid=?)""",
                (pid, started, now, oid, owner_pid, owner_pid),
            )
            if cursor.rowcount:
                claimed.append({
                    "obligation_id": oid,
                    "session_key": session_key,
                    "platform": platform,
                    "chat_id": chat_id,
                    "thread_id": thread_id,
                    "content": content,
                    # pending = send never started, redeliver plainly;
                    # attempting/failed = ambiguous or rejected, carry marker.
                    "needs_marker": state != "pending",
                    "attempts": attempts + 1,
                })
    return claimed


def peek_failed_for_runtime(
    platform: str,
    now: Optional[float] = None,
    *,
    profile: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Return a non-mutating snapshot of this owner's eligible runtime rows.

    This is deliberately advisory: callers must use
    :func:`claim_failed_for_runtime` after any async precondition.  The
    claim repeats every predicate atomically, so a reconnect race cannot
    convert a snapshot into a duplicate send.
    """
    expected_profile = str(profile or "").strip() or "default"
    now = now if now is not None else time.time()
    pid, started = _owner_stamp()
    if started is None:
        return []
    retryable_errors = tuple(sorted(_RUNTIME_RETRYABLE_ERRORS))
    placeholders = ", ".join("?" for _ in retryable_errors)
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            f"""SELECT obligation_id, session_key, adapter_profile
                  FROM delivery_obligations
                 WHERE state='failed' AND platform=?
                   AND COALESCE(NULLIF(trim(adapter_profile), ''), 'default')=?
                   AND owner_pid IS ? AND owner_started_at IS ?
                   AND lower(trim(COALESCE(last_error, ''))) IN ({placeholders})
                   AND attempts < ? AND created_at >= ?""",
            (
                platform,
                expected_profile,
                pid,
                started,
                *retryable_errors,
                MAX_ATTEMPTS,
                now - STALE_AFTER_SECONDS,
            ),
        ).fetchall()
    return [
        {
            "obligation_id": oid,
            "session_key": session_key,
            "profile": _runtime_adapter_profile(adapter_profile),
        }
        for oid, session_key, adapter_profile in rows
    ]


def claim_failed_for_runtime(
    obligation_id: str,
    platform: str,
    now: Optional[float] = None,
    *,
    profile: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Atomically claim one eligible runtime row after its preconditions hold."""
    expected_profile = str(profile or "").strip() or "default"
    now = now if now is not None else time.time()
    pid, started = _owner_stamp()
    if started is None:
        return None
    retryable_errors = tuple(sorted(_RUNTIME_RETRYABLE_ERRORS))
    placeholders = ", ".join("?" for _ in retryable_errors)
    claim_token = _mint_runtime_claim_token()
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            f"""UPDATE delivery_obligations
                   SET state='attempting', attempts=attempts+1, updated_at=?,
                       runtime_claim_token=?
                 WHERE obligation_id=? AND state='failed' AND platform=?
                   AND COALESCE(NULLIF(trim(adapter_profile), ''), 'default')=?
                   AND owner_pid IS ? AND owner_started_at IS ?
                   AND lower(trim(COALESCE(last_error, ''))) IN ({placeholders})
                   AND attempts < ? AND created_at >= ?
             RETURNING obligation_id, session_key, platform, chat_id, thread_id,
                       content, adapter_profile, attempts""",
            (
                now,
                claim_token,
                obligation_id,
                platform,
                expected_profile,
                pid,
                started,
                *retryable_errors,
                MAX_ATTEMPTS,
                now - STALE_AFTER_SECONDS,
            ),
        ).fetchone()
    if row is None:
        return None
    (
        oid,
        session_key,
        row_platform,
        chat_id,
        thread_id,
        content,
        adapter_profile,
        attempts,
    ) = row
    return {
        "obligation_id": oid,
        "session_key": session_key,
        "platform": row_platform,
        "chat_id": chat_id,
        "thread_id": thread_id,
        "content": content,
        "needs_marker": True,
        "marker": RECONNECTED_MARKER,
        "profile": _runtime_adapter_profile(adapter_profile),
        "runtime_recovery": True,
        "attempts": attempts,
        "runtime_claim_token": claim_token,
    }


def sweep_failed_for_runtime(
    platform: str,
    now: Optional[float] = None,
    *,
    profile: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Atomically claim this process's transient failed rows after reconnect."""
    expected_profile = str(profile or "").strip() or "default"
    now = now if now is not None else time.time()
    pid, started = _owner_stamp()
    if started is None:
        return []
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, attempts, created_at, owner_pid,
                      owner_started_at, last_error, adapter_profile
               FROM delivery_obligations
               WHERE state='failed' AND platform=?
                 AND COALESCE(NULLIF(trim(adapter_profile), ''), 'default')=?""",
            (platform, expected_profile),
        ).fetchall()
        for (oid, session_key, row_platform, chat_id, thread_id, content,
             attempts, created_at, owner_pid, owner_started_at, last_error,
             adapter_profile) in rows:
            if owner_pid != pid or owner_started_at != started:
                continue
            if str(last_error or "").strip().lower() not in _RUNTIME_RETRYABLE_ERRORS:
                continue
            if attempts >= MAX_ATTEMPTS or (now - created_at) > STALE_AFTER_SECONDS:
                conn.execute(
                    """UPDATE delivery_obligations
                       SET state='abandoned', updated_at=?, runtime_claim_token=NULL
                       WHERE obligation_id=? AND state='failed'
                         AND COALESCE(NULLIF(trim(adapter_profile), ''), 'default')=?
                         AND owner_pid IS ? AND owner_started_at IS ?""",
                    (now, oid, expected_profile, owner_pid, owner_started_at),
                )
                continue
            claim_token = _mint_runtime_claim_token()
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET state='attempting', attempts=attempts+1, updated_at=?,
                       runtime_claim_token=?
                   WHERE obligation_id=? AND state='failed'
                     AND COALESCE(NULLIF(trim(adapter_profile), ''), 'default')=?
                     AND owner_pid IS ? AND owner_started_at IS ?""",
                (
                    now,
                    claim_token,
                    oid,
                    expected_profile,
                    owner_pid,
                    owner_started_at,
                ),
            )
            if cursor.rowcount:
                claimed.append({
                    "obligation_id": oid,
                    "session_key": session_key,
                    "platform": row_platform,
                    "chat_id": chat_id,
                    "thread_id": thread_id,
                    "content": content,
                    "needs_marker": True,
                    "marker": RECONNECTED_MARKER,
                    "profile": _runtime_adapter_profile(adapter_profile),
                    "runtime_recovery": True,
                    "attempts": attempts + 1,
                    "runtime_claim_token": claim_token,
                })
    return claimed


def _prune(now: Optional[float] = None) -> None:
    now = now if now is not None else time.time()
    cutoff = now - _RETENTION_SECONDS
    try:
        with _transaction() as conn:
            conn.execute(
                """DELETE FROM delivery_obligations
                   WHERE state IN ('delivered', 'abandoned') AND updated_at < ?""",
                (cutoff,),
            )
            total = conn.execute(
                "SELECT COUNT(*) FROM delivery_obligations"
            ).fetchone()[0]
            excess = max(0, total - _MAX_ROWS)
            if excess:
                conn.execute(
                    """DELETE FROM delivery_obligations WHERE obligation_id IN (
                         SELECT obligation_id FROM delivery_obligations
                         ORDER BY CASE state
                                    WHEN 'delivered' THEN 0
                                    WHEN 'abandoned' THEN 1
                                    ELSE 2
                                  END, updated_at ASC
                         LIMIT ?)""",
                    (excess,),
                )
    except Exception:
        logger.debug("delivery ledger prune failed", exc_info=True)


def ledger_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Read the ``gateway.delivery_ledger`` config gate (default on)."""
    try:
        if config is None:
            from hermes_cli.config import load_config

            config = load_config()
        gw = config.get("gateway") or {}
        value = gw.get("delivery_ledger", True)
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no", "off"}
        return bool(value)
    except Exception:
        return True


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
