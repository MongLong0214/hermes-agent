"""The durable compression lock's API resolution and release.

A release is retried, and a holder whose DELETE kept failing is parked on its agent: nobody refreshes or
deletes a released holder's row, so one failed DELETE would otherwise leave a live lease that every later
attempt sits out behind until its TTL. Holder ids are unique per acquisition, so a late retry only ever
deletes the holder that failed.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _lock_api_is_absent_on_session_db(lock_db: Any) -> bool:
    """Whether the live in-memory SessionDB class structurally predates locks.
    Only the exact old ``hermes_state.SessionDB`` class (hot-reload skew) may fail open; proxies, lookalikes,
    non-callables and descriptor failures fail closed."""
    try:
        from hermes_state import SessionDB
        missing = object()
        return (
            type(lock_db) is SessionDB
            and inspect.getattr_static(SessionDB, "try_acquire_compression_lock", missing) is missing
        )
    except Exception:
        return False


def _resolve_lock_api(lock_db: Any) -> tuple[Any, Optional[Exception]]:
    """Return ``(try_acquire_compression_lock, lookup_error)`` for ``lock_db``.
    ``(None, None)`` = no db or legacy SessionDB without the lock API (fail open); ``(None, exc)`` = lookup
    itself failed (caller fails closed)."""
    if lock_db is None:
        return None, None
    try:
        if _lock_api_is_absent_on_session_db(lock_db):
            return None, None
        try_acquire = lock_db.try_acquire_compression_lock
    except Exception as exc:
        return None, exc
    if not callable(try_acquire):
        return None, TypeError("compression lock API is present but not callable")
    return try_acquire, None


_LOCK_RELEASE_ATTEMPTS = 3
_LOCK_RELEASE_RETRY_DELAY_S = 0.25
_unreleased_locks_guard = threading.Lock()


def _release_lock_row(db: Any, sid: str, holder: str) -> bool:
    """Holder-qualified DELETE of the durable lock, retried: nobody refreshes or deletes a released holder's row,
    so one transient failure would leave every later attempt sitting out until the TTL. ``None`` (a store
    without a release result) counts as released."""
    for attempt in range(1, _LOCK_RELEASE_ATTEMPTS + 1):
        try:
            if db.release_compression_lock(sid, holder) is not False:
                return True
        except Exception as exc:  # health: allow BLE001 -- any store error is retried, then logged below
            logger.debug("compression lock release attempt %d failed: %s", attempt, exc)
        if attempt < _LOCK_RELEASE_ATTEMPTS:
            time.sleep(_LOCK_RELEASE_RETRY_DELAY_S * attempt)
    logger.warning(
        "compression lock release failed %d times for session=%s holder=%s — parked for the next attempt",
        _LOCK_RELEASE_ATTEMPTS, sid, holder,
    )
    return False


def _park_unreleased_lock(agent: Any, db: Any, sid: str, holder: str) -> None:
    """Keep a holder whose DELETE batch failed reachable from its agent: acquisition aborts drop the lease right
    after a failed release, and the agent's next attempt would otherwise sit out behind its own abandoned holder
    until the TTL. Holder ids are unique per acquisition, so a late retry only ever deletes this holder's row."""
    with _unreleased_locks_guard:
        parked = getattr(agent, "_unreleased_compression_locks", None)
        if not isinstance(parked, list):
            parked = []
            with contextlib.suppress(Exception):
                agent._unreleased_compression_locks = parked
        if not any(entry[1:] == (sid, holder) for entry in parked):
            parked.append((db, sid, holder))


def _retry_unreleased_locks(agent: Any) -> None:
    """Retry every parked holder's DELETE before taking a new lock; one that still fails stays parked."""
    with _unreleased_locks_guard:
        parked = getattr(agent, "_unreleased_compression_locks", None)
        if not isinstance(parked, list) or not parked:
            return
        pending = list(parked)
        parked.clear()
    for db, sid, holder in pending:
        if not _release_lock_row(db, sid, holder):
            _park_unreleased_lock(agent, db, sid, holder)
