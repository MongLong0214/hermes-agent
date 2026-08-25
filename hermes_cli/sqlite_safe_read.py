"""Lock-safe inspection of SQLite database files.

Why this module exists
----------------------
POSIX advisory locks are cancelled **process-wide** by ``close()`` on *any*
file descriptor for that file::

    the close() system call will cancel all POSIX advisory locks on the
    same file for all threads and all file descriptors in the process
    -- https://sqlite.org/howtocorrupt.html#_posix_advisory_locks_canceled_by_a_separate_thread_doing_close_

So a bare ``open(db_path, "rb") ... close()`` on a **live** database silently
drops every lock SQLite holds on it from this process -- including the
EXCLUSIVE lock a ``VACUUM`` is holding while it rewrites the whole file, and
the RESERVED lock an in-flight ``BEGIN IMMEDIATE`` is holding. Other processes
are then free to write into a file that a writer still believes it owns, which
is the documented route to "database disk image is malformed".

Hermes is exactly the topology this hits: gateway, dispatcher, dashboard,
TUI, CLI, cron and kanban workers all open the same ``state.db`` /
``kanban.db``, and several code paths used to byte-probe those files while
connections were live.

The rules
---------
1. **Never** ``open()`` a database file that may have live connections in this
   process. Ask SQLite instead -- :func:`page_count_bytes` reads the same
   header field via ``PRAGMA``, over the existing connection, taking no new
   descriptor.
2. Byte-level probes are only safe **before any connection exists** for that
   path (first-open validation). Route those through
   :func:`read_header_bytes_preopen`, which refuses once a connection has been
   registered for the path.

Concurrency contract
--------------------
The registry is not advisory bookkeeping -- it is the guard.  Every tracked
open first reserves one global unresolved-open count under ``_live_lock``;
the arbitrary opener and SQLite identity lookup run outside that mutex.  The
actual file identity is published atomically with consumption of that count.
While any count remains, raw descriptor access refuses globally.  The raw
check and ``open``/``read``/``close`` stay together under ``_live_lock``.

Successful close unregisters only afterwards. This is conservatively safe:
the registry can briefly report a closed descriptor as live, but can never
report a live descriptor as absent. It must not hold ``_live_lock`` while a
serialized connection waits for its own lock.

Without that, a thread could pass the "no live connection" check, a second
thread could open a connection and take a write lock, and the first thread's
``close()`` would then cancel it -- reintroducing the exact bug this module
exists to prevent. The lock is never held while a caller *uses* a connection
or while an arbitrary opener runs, so it does not serialise database work.

Path identity
-------------
Connections are keyed by the **canonical database path**, resolved from
``PRAGMA database_list`` on the opened connection. The caller's spelling is
not trustworthy: ``SessionDB``'s read-only path opens
``file:/…/state.db?mode=ro`` with ``uri=True``, and treating that string as a
filesystem path yields a key like ``<cwd>/file:/…/state.db?mode=ro`` which no
later probe of the real ``Path`` can ever match.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlsplit

logger = logging.getLogger(__name__)

SQLITE_HEADER_MAGIC = b"SQLite format 3\x00"

# Offset of the 4-byte big-endian page-count field in the SQLite header.
_HEADER_PAGE_COUNT_OFFSET = 28

# Guards BOTH the registry and the lifecycle syscalls it describes.
_live_lock = threading.RLock()
# ``offline_file_access`` is caller-controlled raw I/O.  Its owner must not
# re-enter ``connect_tracked`` through this re-entrant lifecycle lock: that
# would let an opener reach SQLite before the raw descriptor is closed.
_offline_file_access_state = threading.local()
# canonical path -> number of live connections opened by this process
_live_connections: dict[str, int] = {}
# Every open owns one pending count before it calls an opener.  Until SQLite
# tells us the actual database identity, a path reservation is not safe: a
# custom opener may redirect it, and a symlink may change beneath it.  Raw
# descriptor access therefore refuses globally while this is nonzero.
_pending_unresolved_opens = 0


class UntrackableConnectionError(RuntimeError):
    """A connection to a probe-able database could not be tracked.

    Raised rather than silently returning an untracked connection: on these
    paths tracking is part of the correctness contract, not an optimisation.
    """


class LiveConnectionError(RuntimeError):
    """A raw file operation conflicts with a live or pending connection."""


@dataclass(frozen=True)
class _DatabaseIdentity:
    """The only three useful outcomes of identifying SQLite's ``main`` DB."""

    kind: str
    path: str | None = None


_FILE_IDENTITY = "file"
_MEMORY_IDENTITY = "memory/unnamed"
_UNRESOLVED_IDENTITY = "unresolved"
_FILESYSTEM_IDENTITY_MODE = "filesystem"
_SQLITE_URI_IDENTITY_MODE = "sqlite_uri"


def _key(path: Path | str) -> str:
    """Canonicalise a *filesystem* path for use as a registry key."""
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


def _file_identity(path: Path | str) -> _DatabaseIdentity:
    """Resolve one filesystem spelling, or explicitly report no authority."""
    try:
        return _DatabaseIdentity(_FILE_IDENTITY, str(Path(path).resolve()))
    except (OSError, RuntimeError, TypeError, ValueError):
        return _DatabaseIdentity(_UNRESOLVED_IDENTITY)


def _database_identity(
    path: Path | str,
    *,
    mode: str,
) -> _DatabaseIdentity:
    """Classify a target without opening it in its explicit interpretation mode.

    Filesystem callers never infer URI syntax from a ``file:`` prefix: that is
    a legal POSIX filename.  ``sqlite_uri`` is reserved for a real
    ``sqlite3.connect`` request whose caller supplied a string URI with
    ``uri=True``.  URI requests retain their SQLite query/memory semantics,
    while every other spelling is a literal filesystem path.
    """
    if mode == _FILESYSTEM_IDENTITY_MODE:
        return _file_identity(path)
    if mode != _SQLITE_URI_IDENTITY_MODE:  # pragma: no cover - internal invariant
        raise ValueError(f"unknown SQLite identity mode: {mode}")
    if not isinstance(path, str) or not path.startswith("file:"):
        return _file_identity(path)
    try:
        parsed = urlsplit(path)
    except ValueError:
        return _DatabaseIdentity(_UNRESOLVED_IDENTITY)
    if parsed.scheme != "file":
        return _DatabaseIdentity(_UNRESOLVED_IDENTITY)
    # SQLite file URIs with a non-local authority have no filesystem spelling
    # this layer can authoritatively guard.
    if parsed.netloc not in ("", "localhost"):
        return _DatabaseIdentity(_UNRESOLVED_IDENTITY)
    query = parse_qs(parsed.query, keep_blank_values=True)
    candidate = unquote(parsed.path)
    if candidate in ("", ":memory:") or query.get("mode") == ["memory"]:
        return _DatabaseIdentity(_MEMORY_IDENTITY)
    return _file_identity(candidate)


def _offline_file_access_depth() -> int:
    """The nesting depth of caller-controlled raw access in this thread."""
    return getattr(_offline_file_access_state, "depth", 0)


def _increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _decrement_count(counts: dict[str, int], key: str) -> None:
    remaining = counts.get(key, 0) - 1
    if remaining > 0:
        counts[key] = remaining
    else:
        counts.pop(key, None)


def _begin_pending_open() -> None:
    """Reserve exactly one unresolved-open slot before an arbitrary opener."""
    global _pending_unresolved_opens
    with _live_lock:
        _pending_unresolved_opens += 1


def _publish_or_finish_pending_open(identity: _DatabaseIdentity) -> None:
    """Publish a file identity, if any, and consume this call's pending slot."""
    global _pending_unresolved_opens
    with _live_lock:
        if identity.kind == _FILE_IDENTITY and identity.path is not None:
            _increment_count(_live_connections, identity.path)
        if _pending_unresolved_opens <= 0:  # pragma: no cover - internal invariant
            raise RuntimeError("SQLite unresolved-open accounting underflow")
        _pending_unresolved_opens -= 1


def _post_open_database_identity(
    conn: sqlite3.Connection,
    *,
    mode: str,
) -> _DatabaseIdentity:
    """Return SQLite's authoritative post-open identity, never an inference."""
    try:
        # This bypasses an optional connection wrapper.  It runs before the
        # handle is registered and never under _live_lock, preserving the
        # serial -> live callback order.
        cursor = sqlite3.Connection.execute(conn, "PRAGMA database_list")
        row = sqlite3.Cursor.fetchone(cursor)
    except sqlite3.Error:
        # An authorizer denial is unresolved, not evidence of an in-memory DB.
        return _DatabaseIdentity(_UNRESOLVED_IDENTITY)
    if not row or len(row) < 3:
        return _DatabaseIdentity(_UNRESOLVED_IDENTITY)
    path_str = row[2]
    if not path_str:
        return _DatabaseIdentity(_MEMORY_IDENTITY)
    if not isinstance(path_str, str):
        return _DatabaseIdentity(_UNRESOLVED_IDENTITY)
    return _database_identity(path_str, mode=mode)


def _canonical_db_path(conn: sqlite3.Connection) -> Optional[str]:
    """Compatibility helper for callers that only need a known file path."""
    identity = _post_open_database_identity(conn, mode=_FILESYSTEM_IDENTITY_MODE)
    return identity.path if identity.kind == _FILE_IDENTITY else None


def _require_matching_identity(
    requested: _DatabaseIdentity,
    tracking: _DatabaseIdentity | None,
    actual: _DatabaseIdentity,
) -> None:
    """Reject anything that cannot prove the opened file is the requested one."""
    if actual.kind == _UNRESOLVED_IDENTITY:
        raise UntrackableConnectionError(
            "SQLite database identity is unresolved after open; refusing to expose "
            "a possibly file-backed connection without byte-probe protection"
        )
    for label, expected in (("requested", requested), ("tracking", tracking)):
        if expected is None:
            continue
        if expected.kind == _UNRESOLVED_IDENTITY:
            raise UntrackableConnectionError(
                f"{label} SQLite database identity cannot be resolved before open"
            )
        if expected.kind != actual.kind:
            raise UntrackableConnectionError(
                f"{label} SQLite database identity does not match the opened database"
            )
        if (
            expected.kind == _FILE_IDENTITY
            and expected.path != actual.path
        ):
            raise UntrackableConnectionError(
                f"{label} SQLite database identity does not match the opened database"
            )


def track_connection(path: Path | str) -> None:
    """Record that this process now holds a connection to *path*.

    Prefer :func:`connect_tracked`; this exists for callers that manage their
    own connection objects, and for tests.
    """
    key = _key(path)
    with _live_lock:
        _live_connections[key] = _live_connections.get(key, 0) + 1


def untrack_connection(path: Path | str) -> None:
    """Record that one connection to *path* has been closed."""
    key = _key(path)
    with _live_lock:
        remaining = _live_connections.get(key, 0) - 1
        if remaining > 0:
            _live_connections[key] = remaining
        else:
            _live_connections.pop(key, None)


def has_live_connection(path: Path | str) -> bool:
    """Whether *path* has a live handle or any opener lacks an identity."""
    identity = _database_identity(path, mode=_FILESYSTEM_IDENTITY_MODE)
    with _live_lock:
        return _pending_unresolved_opens > 0 or (
            identity.kind == _FILE_IDENTITY
            and identity.path is not None
            and identity.path in _live_connections
        )


class _TrackingMixin:
    """Untrack-on-close behaviour, mixable into any Connection subclass."""

    _hermes_tracked_path: str | None = None

    def close(self) -> None:  # type: ignore[misc]
        path = getattr(self, "_hermes_tracked_path", None)
        # Do not take _live_lock before this call.  A serialized Connection
        # may wait for its per-connection RLock here, while Python callbacks
        # from an in-flight SQLite operation legitimately ask the registry a
        # serial -> live question.  Holding live -> serial would deadlock.
        #
        # The entry remains tracked until after close succeeds, so the guard
        # is conservative during the small post-close window: a raw probe can
        # be refused too long, but it can never be allowed while the FD lives.
        # A failure deliberately leaves the attribute and count untouched.
        super().close()  # type: ignore[misc]
        if path is None:
            return
        with _live_lock:
            # Two concurrent/double close calls can both observe ``path``
            # before SQLite closes.  Only the first successful closer gets to
            # consume this connection's tracking count.
            if getattr(self, "_hermes_tracked_path", None) != path:
                return
            self._hermes_tracked_path = None
            remaining = _live_connections.get(path, 0) - 1
            if remaining > 0:
                _live_connections[path] = remaining
            else:
                _live_connections.pop(path, None)


class TrackedConnection(_TrackingMixin, sqlite3.Connection):
    """A ``sqlite3.Connection`` that untracks its path exactly once on close.

    Counting opens is easy; counting closes reliably is not, because callers
    close connections in many places (and some hand them to
    ``contextlib.closing``). Putting the decrement on ``close()`` — the one
    method every close path must go through — keeps the registry from
    drifting upward and permanently disabling byte-probes.

    Unregister runs only after ``close()`` succeeds; a raising close leaves
    the connection tracked so the byte-probe guard keeps refusing. There is
    deliberately no ``_live_lock`` around the SQLite close: a serialized
    close may wait for the per-connection lock, and callback code acquires
    those locks in the opposite (serial -> live) order. The brief window after
    a successful close remains conservatively tracked, so a probe can never
    observe "no live connection" while the descriptor is still open.

    Note ``with conn:`` does NOT close a sqlite3 connection (it only commits or
    rolls back), so this hook is not fired spuriously by transaction scopes.
    """


_tracked_factory_cache: dict[type, type] = {}


def _tracking_factory(factory: type) -> type:
    """Return *factory* augmented with untrack-on-close.

    Callers legitimately supply their own ``Connection`` subclasses (the test
    suite uses them to simulate FTS5-less or pragma-failing runtimes). Rather
    than refusing those — or silently leaving them untracked, which would
    quietly unguard the database — we mix the tracking ``close()`` into the
    caller's class so tracking is preserved either way.
    """
    if factory is sqlite3.Connection:
        return TrackedConnection
    if issubclass(factory, _TrackingMixin):
        return factory
    cached = _tracked_factory_cache.get(factory)
    if cached is None:
        cached = type(f"Tracked{factory.__name__}", (_TrackingMixin, factory), {})
        _tracked_factory_cache[factory] = cached
    return cached


def connect_tracked(
    path: Path | str,
    *,
    tracking_path: Path | str | None = None,
    connect_fn=None,
    **kwargs,
) -> sqlite3.Connection:
    """``sqlite3.connect`` that registers the connection for the lifetime of the fd.

    Use for any connection to a database whose file might otherwise be
    byte-probed (``state.db``, ``kanban.db``). The registration is released
    automatically on ``close()``.

    An unresolved-open count is created under ``_live_lock`` before the opener
    runs.  The opener, SQLite validation, retrofit, and rejected-handle cleanup
    run outside that mutex.  Raw probes refuse globally until SQLite reports
    the actual identity, which is then published atomically with consumption of
    this call's pending count.

    The registry key is the canonical path reported by ``PRAGMA
    database_list`` -- not *path*, which may be a ``file:`` URI.
    ``tracking_path`` is an identity assertion: it and the normal request must
    both agree with SQLite's post-open path.

    ``connect_fn`` lets a caller supply its own opener (defaults to
    :func:`sqlite3.connect`), so a module that owns the connection — and any
    test that patches that module's ``sqlite3.connect`` — keeps control of how
    the connection is created while this helper owns tracking.

    A caller-supplied ``factory`` is honoured but is transparently augmented
    with untrack-on-close, so tracking is never silently skipped. If a
    file-backed connection still cannot be tracked,
    :class:`UntrackableConnectionError` is raised rather than handing back a
    connection whose database has quietly lost byte-probe protection.
    """
    if _offline_file_access_depth():
        raise LiveConnectionError(
            "Refusing to open a SQLite connection while offline file access is "
            "active in this thread."
        )
    opener = connect_fn if connect_fn is not None else sqlite3.connect
    kwargs["factory"] = _tracking_factory(kwargs.get("factory", sqlite3.Connection))
    identity_mode = (
        _SQLITE_URI_IDENTITY_MODE
        if isinstance(path, str) and path.startswith("file:") and bool(kwargs.get("uri"))
        else _FILESYSTEM_IDENTITY_MODE
    )
    requested_identity = (
        _DatabaseIdentity(_MEMORY_IDENTITY)
        if isinstance(path, str) and path in ("", ":memory:")
        else _database_identity(path, mode=identity_mode)
    )
    tracking_identity = (
        _database_identity(tracking_path, mode=identity_mode)
        if tracking_path is not None
        else None
    )
    open_path = str(path)
    if (
        connect_fn is None
        and requested_identity.kind == _FILE_IDENTITY
        and requested_identity.path is not None
        and identity_mode == _FILESYSTEM_IDENTITY_MODE
    ):
        # Resolve once before default opening.  This spelling is also what a
        # later raw probe uses, so an alias retarget cannot change one side.
        open_path = requested_identity.path

    _begin_pending_open()
    pending_open = True
    conn = None
    try:
        # Never hold _live_lock while calling arbitrary opener code or SQLite.
        conn = opener(open_path, **kwargs)
        if not isinstance(conn, sqlite3.Connection):
            raise UntrackableConnectionError(
                "SQLite opener returned "
                f"{type(conn).__name__}, not a sqlite3.Connection; "
                "byte-probe safety cannot be tracked"
            )
        actual_identity = _post_open_database_identity(conn, mode=identity_mode)
        _require_matching_identity(
            requested_identity,
            tracking_identity,
            actual_identity,
        )
        if actual_identity.kind == _FILE_IDENTITY:
            assert actual_identity.path is not None  # narrowed by the identity tag
            if not isinstance(conn, _TrackingMixin):
                # The opener substituted its own factory and discarded ours
                # (test doubles simulating FTS5-less runtimes do this). Retag
                # outside _live_lock so no path can wait on custom class
                # machinery there.
                conn = _retrofit_tracking(conn, actual_identity.path)
            conn._hermes_tracked_path = actual_identity.path
        # The success transition publishes a file identity (or only consumes a
        # memory/unnamed open) under the lifecycle mutex.
        _publish_or_finish_pending_open(actual_identity)
        pending_open = False
        return conn
    except BaseException as exc:
        if conn is not None:
            cleanup_error = close_rejected_handle_once(conn)
            if cleanup_error is not None:
                # Preserve the validation/open failure as primary while keeping
                # the cleanup failure available for diagnostics.
                try:
                    setattr(exc, "_hermes_cleanup_error", cleanup_error)
                except Exception:
                    logger.debug("rejected SQLite opener cleanup also failed", exc_info=True)
        raise
    finally:
        if pending_open:
            # Cleanup above happens before this transition.  Each nested or
            # concurrent call owns one slot, so a failure cannot erase another
            # opener's protection.
            _publish_or_finish_pending_open(_DatabaseIdentity(_MEMORY_IDENTITY))


def close_rejected_handle_once(handle) -> Optional[BaseException]:
    """Directly close one rejected actual resource without trusting overrides."""
    try:
        if isinstance(handle, sqlite3.Connection):
            sqlite3.Connection.close(handle)
        elif isinstance(handle, sqlite3.Cursor):
            sqlite3.Cursor.close(handle)
        elif hasattr(sqlite3, "Blob") and isinstance(handle, sqlite3.Blob):
            sqlite3.Blob.close(handle)
        else:
            handle.close()
    except BaseException as exc:
        return exc
    return None


def _retrofit_tracking(conn: sqlite3.Connection, resolved: str) -> sqlite3.Connection:
    """Give an already-open connection untrack-on-close semantics.

    ``sqlite3.Connection`` subclasses are ordinary Python classes, so the
    instance's ``__class__`` can be swapped for one that mixes in the tracking
    ``close()``. Used when an opener ignored the factory we asked for.
    """
    cls = type(conn)
    if issubclass(cls, _TrackingMixin):
        return conn
    try:
        conn.__class__ = _tracking_factory(cls)  # type: ignore[assignment]
        return conn
    except TypeError as exc:
        raise UntrackableConnectionError(
            f"connection to {resolved} uses factory {cls.__name__}, which "
            "cannot release its tracking entry on close; byte-probe safety "
            "for this database would be silently lost"
        ) from exc


def page_count_bytes(conn: sqlite3.Connection) -> Optional[int]:
    """Logical database size in bytes, read through *conn*.

    ``page_count * page_size`` is the same quantity the 4-byte header field at
    offset 28 carries, but reading it via ``PRAGMA`` opens no new file
    descriptor and therefore cannot cancel this process's POSIX locks.

    Returns ``None`` when the pragmas cannot be read.
    """
    try:
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    except (sqlite3.Error, TypeError, IndexError) as exc:
        logger.debug("page_count/page_size unavailable: %s", exc)
        return None
    try:
        return int(page_count) * int(page_size)
    except (TypeError, ValueError):
        return None


def file_length_matches_header(conn: sqlite3.Connection) -> Optional[bool]:
    """Whether the file on disk is at least as long as the header claims.

    Detects the "torn extend" shape (file shorter than its own page count)
    without ever opening the database file: the header side comes from
    ``PRAGMA page_count`` over *conn*, and the on-disk side from ``stat()``,
    which takes no descriptor and cannot break locks.

    Returns ``None`` when the check is not applicable (in-memory database,
    unreadable pragmas, or a stat failure).

    Note: in WAL mode a freshly committed page may still live in the ``-wal``
    file, so the main file legitimately lags. Callers must treat this as
    advisory unless the database is in a rollback journal mode.
    """
    path_str = _canonical_db_path(conn)
    if path_str is None:
        return None

    logical = page_count_bytes(conn)
    if not logical:
        return None
    try:
        actual = os.path.getsize(path_str)
    except OSError:
        return None
    return actual >= logical


def read_header_bytes_preopen(
    path: Path | str,
    *,
    length: int = 100,
    force: bool = False,
) -> Optional[bytes]:
    """Read the first *length* bytes of *path* -- only when no connection is live.

    This is the ONLY sanctioned byte-level read of a database file, and it is
    restricted to first-open validation (is this file a real SQLite database,
    is it zeroed, has it been overwritten by something else). Once any
    connection to *path* exists in this process, the read is refused and
    ``None`` is returned, because the ``close()`` would cancel that
    connection's POSIX locks.

    The registry check and the ``open``/``read``/``close`` are performed
    together under ``_live_lock``, so a connection cannot be opened in the
    window between deciding "nothing is live" and closing this descriptor.

    Set ``force=True`` only for genuinely offline files (quarantined copies,
    snapshot artifacts, archives) that no live connection can reference.  It
    never bypasses a global unresolved opener: that opener may turn out to own
    this file under a spelling no path registry can predict.
    """
    identity = _database_identity(path, mode=_FILESYSTEM_IDENTITY_MODE)
    if identity.kind != _FILE_IDENTITY or identity.path is None:
        return None
    with _live_lock:
        if _pending_unresolved_opens > 0 or (
            not force and identity.path in _live_connections
        ):
            logger.debug(
                "refusing byte-level read of %s: a live connection or unresolved "
                "opener exists in this process and close() would cancel POSIX locks",
                path,
            )
            return None
        try:
            # Open the exact resolved spelling checked above.  Do not reopen a
            # caller alias after a symlink retarget between check and use.
            with open(identity.path, "rb") as handle:
                return handle.read(length)
        except OSError:
            return None


@contextlib.contextmanager
def offline_file_access(path: Path | str, *, what: str = "read"):
    """Hold the connection-lifecycle lock across a raw read of a database file.

    Checking :func:`has_live_connection` and *then* doing the raw I/O is a
    check/use race: a connection can be opened in the window between the two,
    and the raw ``close()`` will cancel its POSIX advisory locks — the exact
    failure class the registry exists to prevent. Any multi-step raw access
    (copying a database plus its ``-wal``/``-shm``/``-journal`` sidecars,
    hashing a file, moving a bundle aside) must therefore run *inside* this
    context manager rather than after a bare check.

    While held, :func:`connect_tracked` blocks, so no new connection can
    appear mid-copy. Raises :class:`LiveConnectionError` if a connection is
    already live when the guard is entered.

    The lock is only held for the duration of the raw I/O; it never spans
    caller work on an open connection, so it does not serialise database use.
    """
    identity = _database_identity(path, mode=_FILESYSTEM_IDENTITY_MODE)
    if identity.kind != _FILE_IDENTITY or identity.path is None:
        raise LiveConnectionError(
            f"Refusing to {what} {path}: its filesystem identity cannot be "
            "authoritatively resolved for safe raw access."
        )
    with _live_lock:
        if _pending_unresolved_opens > 0 or identity.path in _live_connections:
            raise LiveConnectionError(
                f"Refusing to {what} {path}: a connection is open or its identity "
                "is being resolved in this process, and raw file access would cancel "
                "connection's POSIX advisory locks. Close all database "
                "handles (stop the gateway/dashboard) and retry."
            )
        previous_depth = _offline_file_access_depth()
        _offline_file_access_state.depth = previous_depth + 1
        try:
            yield
        finally:
            if previous_depth:
                _offline_file_access_state.depth = previous_depth
            else:
                del _offline_file_access_state.depth
