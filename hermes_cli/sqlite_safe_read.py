"""Fail-closed raw SQLite inspection and connection-lifecycle accounting.

The registry protects only registry state.  It is deliberately never retained
while a connection opener, SQLite call, serial-lock wait, close hook, or raw
file descriptor operation runs: reservations make those slow operations safe
without creating a lifecycle/connection-lock ABBA cycle.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlsplit

logger = logging.getLogger(__name__)

SQLITE_HEADER_MAGIC = b"SQLite format 3\x00"

_live_lock = threading.RLock()
_live_connections: dict[str, int] = {}
_pending_unresolved_opens = 0
_raw_file_reservations: dict[str, int] = {}
_offline_file_access_state = threading.local()

_FILE = "file"
_MEMORY = "memory"
_UNRESOLVED = "unresolved"


class UntrackableConnectionError(RuntimeError):
    """A file-backed connection cannot retain the raw-probe safety contract."""


class LiveConnectionError(RuntimeError):
    """Raw access conflicts with an open, pending, or reserved database path."""


@dataclass(frozen=True)
class _DatabaseIdentity:
    kind: str
    path: str | None = None


def _key(path: Path | str) -> str:
    try:
        return str(Path(path).resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return str(path)


def _identity(path: Path | str, *, uri: bool = False) -> _DatabaseIdentity:
    """Classify a requested location without opening a file descriptor."""
    if not uri or not isinstance(path, str) or not path.startswith("file:"):
        return _DatabaseIdentity(_FILE, _key(path))
    try:
        parsed = urlsplit(path)
    except ValueError:
        return _DatabaseIdentity(_UNRESOLVED)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        return _DatabaseIdentity(_UNRESOLVED)
    query = parse_qs(parsed.query, keep_blank_values=True)
    candidate = unquote(parsed.path)
    if candidate in ("", ":memory:") or query.get("mode") == ["memory"]:
        return _DatabaseIdentity(_MEMORY)
    return _DatabaseIdentity(_FILE, _key(candidate))


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _decrement(counts: dict[str, int], key: str, *, label: str) -> None:
    count = counts.get(key, 0)
    if count <= 0:
        raise RuntimeError(f"SQLite {label} accounting underflow for {key}")
    if count == 1:
        del counts[key]
    else:
        counts[key] = count - 1


def _begin_pending_open(requested: _DatabaseIdentity) -> None:
    """Reserve the unknown actual identity before invoking arbitrary code."""
    global _pending_unresolved_opens
    with _live_lock:
        if requested.kind == _FILE and requested.path and _raw_file_reservations.get(requested.path):
            raise LiveConnectionError(
                "refusing SQLite open while raw file access owns the database path"
            )
        _pending_unresolved_opens += 1


def _finish_pending_open(actual: _DatabaseIdentity | None) -> None:
    global _pending_unresolved_opens
    with _live_lock:
        if _pending_unresolved_opens <= 0:
            raise RuntimeError("SQLite unresolved-open accounting underflow")
        _pending_unresolved_opens -= 1
        if actual is not None and actual.kind == _FILE and actual.path:
            _increment(_live_connections, actual.path)


def _post_open_identity(connection: sqlite3.Connection) -> _DatabaseIdentity:
    """Ask SQLite for the main DB identity outside the registry lock."""
    try:
        cursor = sqlite3.Connection.execute(connection, "PRAGMA database_list")
        row = sqlite3.Cursor.fetchone(cursor)
    except sqlite3.Error:
        return _DatabaseIdentity(_UNRESOLVED)
    if not row or len(row) < 3 or not isinstance(row[2], str):
        return _DatabaseIdentity(_UNRESOLVED)
    if not row[2]:
        return _DatabaseIdentity(_MEMORY)
    return _DatabaseIdentity(_FILE, _key(row[2]))


def _canonical_db_path(connection: sqlite3.Connection) -> Optional[str]:
    identity = _post_open_identity(connection)
    return identity.path if identity.kind == _FILE else None


def track_connection(path: Path | str) -> None:
    key = _key(path)
    with _live_lock:
        _increment(_live_connections, key)


def untrack_connection(path: Path | str) -> None:
    key = _key(path)
    with _live_lock:
        _decrement(_live_connections, key, label="live-connection")


def has_live_connection(path: Path | str) -> bool:
    key = _key(path)
    with _live_lock:
        return bool(_pending_unresolved_opens or _live_connections.get(key, 0))


def _is_physically_closed(connection: sqlite3.Connection) -> bool:
    try:
        cursor = sqlite3.Connection.execute(connection, "SELECT 1")
        sqlite3.Cursor.fetchone(cursor)
    except sqlite3.ProgrammingError as exc:
        # Only SQLite's closed-handle result proves physical retirement.
        # Thread-affinity and other ProgrammingError cases leave the owner live.
        return str(exc) == "Cannot operate on a closed database."
    except sqlite3.Error:
        return False
    return False


def _attach_cleanup_error(primary: BaseException, cleanup: BaseException) -> None:
    try:
        setattr(primary, "_hermes_cleanup_error", cleanup)
    except Exception:
        logger.debug("unable to attach rejected SQLite cleanup failure", exc_info=True)


def close_rejected_handle_once(handle) -> BaseException | None:
    """Close an unexposed rejected handle without redispatching a hook twice."""
    if not isinstance(handle, sqlite3.Connection):
        return None
    primary = None
    try:
        handle.close()
    except BaseException as exc:
        primary = exc
    if not _is_physically_closed(handle):
        try:
            sqlite3.Connection.close(handle)
        except BaseException as exc:
            return primary or exc
    return primary


class _TrackingMixin:
    """Physical-close accounting composed ahead of arbitrary connection classes."""

    _hermes_tracked_path: str | None = None
    _hermes_close_hook_invoked = False
    _hermes_native_close_complete = False
    _hermes_tracking_close_active = False

    def close(self) -> None:  # type: ignore[misc]
        if getattr(self, "_hermes_native_close_complete", False):
            return
        if getattr(self, "_hermes_tracking_close_active", False):
            return

        serial_lock = None
        serial_mixin = getattr(sys.modules.get("hermes_state"), "_SerializedConnectionMixin", None)
        if getattr(self, "_hermes_serial_lock", None) is not None or (
            serial_mixin is not None and serial_mixin in type(self).__mro__
        ):
            from hermes_state import _serial_lock_for_connection

            serial_lock = _serial_lock_for_connection(self)
            boundary = serial_lock
        else:
            boundary = contextlib.nullcontext()

        primary: BaseException | None = None
        cleanup: BaseException | None = None
        with boundary:
            if serial_lock is not None:
                from hermes_state import _serial_lock_for_connection

                if _serial_lock_for_connection(self) is not serial_lock:
                    raise UntrackableConnectionError("connection serial owner changed during close")
            self._hermes_tracking_close_active = True
            try:
                if not getattr(self, "_hermes_close_hook_invoked", False):
                    self._hermes_close_hook_invoked = True
                    try:
                        super().close()  # type: ignore[misc]
                    except BaseException as exc:
                        primary = exc
                # Custom dispatch is allowed one primary failure, but physical
                # ownership stays here: direct native cleanup must still run
                # exactly once without redispatching that hook.
                if not _is_physically_closed(self):
                    try:
                        sqlite3.Connection.close(self)
                    except BaseException as exc:
                        cleanup = exc
                if _is_physically_closed(self):
                    self._hermes_native_close_complete = True
                    path = getattr(self, "_hermes_tracked_path", None)
                    if path is not None:
                        with _live_lock:
                            _decrement(_live_connections, path, label="live-connection")
                        self._hermes_tracked_path = None
            finally:
                self._hermes_tracking_close_active = False

        if primary is not None:
            if cleanup is not None:
                _attach_cleanup_error(primary, cleanup)
            raise primary
        if cleanup is not None:
            raise cleanup


class TrackedConnection(_TrackingMixin, sqlite3.Connection):
    pass


_tracking_factories: dict[type, type] = {}


def _tracking_factory(factory: type) -> type:
    if factory is sqlite3.Connection:
        return TrackedConnection
    if not isinstance(factory, type) or not issubclass(factory, sqlite3.Connection):
        return factory
    if issubclass(factory, _TrackingMixin):
        return factory
    mixed = _tracking_factories.get(factory)
    if mixed is None:
        mixed = type(f"Tracked{factory.__name__}", (_TrackingMixin, factory), {})
        _tracking_factories[factory] = mixed
    return mixed


def _retrofit_tracking(connection: sqlite3.Connection) -> sqlite3.Connection:
    if isinstance(connection, _TrackingMixin):
        return connection
    try:
        connection.__class__ = _tracking_factory(type(connection))
    except (AttributeError, TypeError) as exc:
        raise UntrackableConnectionError(
            "SQLite connection cannot retain physical-close tracking"
        ) from exc
    return connection


def connect_tracked(
    path: Path | str,
    *,
    tracking_path: Path | str | None = None,
    connect_fn=None,
    **kwargs,
) -> sqlite3.Connection:
    """Open a tracked connection without holding ``_live_lock`` across SQLite."""
    opener = connect_fn if connect_fn is not None else sqlite3.connect
    # Every tracked cohort opener—direct or callable transfer—must compose the
    # physical serialization owner before tracking adds its close accounting.
    # The import is intentionally lazy: ``hermes_state`` calls this helper only
    # after defining the serialization types, while Kanban reaches it directly.
    from hermes_state import _ensure_serialized_connection, _serialized_connection_factory

    requested = _identity(path, uri=bool(kwargs.get("uri")))
    expected = _identity(tracking_path) if tracking_path is not None else requested
    kwargs["factory"] = _tracking_factory(
        _serialized_connection_factory(kwargs.get("factory", sqlite3.Connection))
    )
    _begin_pending_open(requested)
    published = False
    connection = None
    try:
        connection = opener(str(path), **kwargs)
        if not isinstance(connection, sqlite3.Connection):
            raise UntrackableConnectionError("SQLite opener returned a non-connection")
        connection = _ensure_serialized_connection(connection)
        actual = _post_open_identity(connection)
        if actual.kind == _UNRESOLVED:
            raise UntrackableConnectionError("SQLite database identity is unresolved after open")
        if expected.kind == _FILE and (actual.kind != _FILE or actual.path != expected.path):
            raise UntrackableConnectionError("SQLite opener resolved a different database path")
        if actual.kind == _FILE:
            connection = _retrofit_tracking(connection)
            connection._hermes_tracked_path = actual.path
        _finish_pending_open(actual)
        published = True
        return connection
    except BaseException as exc:
        if connection is not None:
            cleanup_error = close_rejected_handle_once(connection)
            if cleanup_error is not None:
                _attach_cleanup_error(exc, cleanup_error)
        raise
    finally:
        if not published:
            _finish_pending_open(None)


def page_count_bytes(connection: sqlite3.Connection) -> Optional[int]:
    try:
        page_count = connection.execute("PRAGMA page_count").fetchone()[0]
        page_size = connection.execute("PRAGMA page_size").fetchone()[0]
        return int(page_count) * int(page_size)
    except (sqlite3.Error, TypeError, IndexError, ValueError):
        return None


def file_length_matches_header(connection: sqlite3.Connection) -> Optional[bool]:
    path = _canonical_db_path(connection)
    logical = page_count_bytes(connection)
    if path is None or not logical:
        return None
    try:
        return os.path.getsize(path) >= logical
    except OSError:
        return None


def _reserve_raw_access(path: Path | str, *, force: bool) -> str | None:
    key = _key(path)
    with _live_lock:
        unsafe = _pending_unresolved_opens or _live_connections.get(key, 0)
        if unsafe and not force:
            return None
        _increment(_raw_file_reservations, key)
    return key


def _release_raw_access(key: str) -> None:
    with _live_lock:
        _decrement(_raw_file_reservations, key, label="raw-file reservation")


def read_header_bytes_preopen(
    path: Path | str, *, length: int = 100, force: bool = False
) -> Optional[bytes]:
    """Read a header only while a reservation excludes new tracked opens."""
    reservation = _reserve_raw_access(path, force=force)
    if reservation is None:
        return None
    try:
        try:
            with open(path, "rb") as handle:
                return handle.read(length)
        except OSError:
            return None
    finally:
        _release_raw_access(reservation)


@contextlib.contextmanager
def offline_file_access(path: Path | str, *, what: str = "read"):
    """Reserve raw descriptor access without retaining the lifecycle mutex."""
    reservation = _reserve_raw_access(path, force=False)
    if reservation is None:
        raise LiveConnectionError(
            f"Refusing to {what} {path}: a connection is live or opening in this process"
        )
    try:
        yield
    finally:
        _release_raw_access(reservation)
