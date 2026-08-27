"""POSIX advisory locks must survive Hermes' own database inspection.

close() on ANY file descriptor for a SQLite database cancels every POSIX
advisory lock the process holds on that file -- including a running VACUUM's
EXCLUSIVE lock and an in-flight BEGIN IMMEDIATE's RESERVED lock:

    https://sqlite.org/howtocorrupt.html#_posix_advisory_locks_canceled_by_a_separate_thread_doing_close_

Hermes used to byte-probe live databases in several places (kanban's
post-commit page-count check, the zeroed-state.db detector run on every
SessionDB construction, backup header verification). Under `hermes sessions
optimize` this let an external process write into a database while VACUUM was
rewriting it, producing "database disk image is malformed".

These tests pin the behavioural contract: an external process must stay locked
out across Hermes' inspection calls.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
import threading

import pytest

from hermes_cli.sqlite_safe_read import (
    file_length_matches_header,
    has_live_connection,
    page_count_bytes,
    read_header_bytes_preopen,
    track_connection,
    untrack_connection,
)


_INTRUDER = textwrap.dedent(
    """
    import sqlite3, sys
    conn = sqlite3.connect(sys.argv[1], isolation_level=None, timeout=0)
    try:
        conn.execute("PRAGMA busy_timeout=0")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO t(v) VALUES ('intruder')")
        conn.execute("COMMIT")
        print("ACQUIRED")
    except sqlite3.OperationalError:
        print("BLOCKED")
    """
)


def _external_writer_can_break_in(db_path) -> bool:
    """True when a separate process managed to write to a locked database."""
    result = subprocess.run(
        [sys.executable, "-c", _INTRUDER, str(db_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return "ACQUIRED" in result.stdout


def _make_db(path, journal_mode: str) -> None:
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute(f"PRAGMA journal_mode={journal_mode}")
    conn.execute("CREATE TABLE t(v TEXT)")
    conn.executemany("INSERT INTO t(v) VALUES (?)", [(f"row{i}",) for i in range(200)])
    conn.close()


@pytest.fixture
def clean_registry():
    yield
    # Keep the module-level registry from leaking across tests.
    import hermes_cli.sqlite_safe_read as mod

    with mod._live_lock:
        mod._live_connections.clear()






def test_preopen_read_refused_while_connection_is_live(tmp_path, clean_registry):
    """The byte-level probe is allowed pre-open and refused once connected."""
    db = tmp_path / "state.db"
    _make_db(db, "WAL")

    assert not has_live_connection(db)
    head = read_header_bytes_preopen(db, length=16)
    assert head == b"SQLite format 3\x00"

    track_connection(db)
    try:
        assert has_live_connection(db)
        assert read_header_bytes_preopen(db, length=16) is None
        # An explicit override stays available for offline artifacts.
        assert read_header_bytes_preopen(db, length=16, force=True) is not None
    finally:
        untrack_connection(db)

    assert not has_live_connection(db)
    assert read_header_bytes_preopen(db, length=16) is not None


def test_tracking_registry_does_not_leak_across_close_paths(tmp_path, clean_registry):
    """A drifting counter would silently disable the probe guard forever.

    Opens are easy to count; closes happen in many places. If the registry
    ever over-counts, ``has_live_connection`` stays true for a path with no
    live connection and every later byte-probe is refused — turning the
    safety guard into a permanent outage of zeroed-file / header detection.
    """
    import contextlib

    from hermes_cli.sqlite_safe_read import connect_tracked

    db = tmp_path / "state.db"
    boot = connect_tracked(db, isolation_level=None)
    boot.execute("CREATE TABLE t(v TEXT)")
    boot.close()
    assert not has_live_connection(db)

    # plain close
    connect_tracked(db).close()
    assert not has_live_connection(db)

    # contextlib.closing
    with contextlib.closing(connect_tracked(db)):
        assert has_live_connection(db)
    assert not has_live_connection(db)

    # `with conn:` is a TRANSACTION scope, not a close — must stay tracked
    conn = connect_tracked(db, isolation_level=None)
    with conn:
        conn.execute("INSERT INTO t(v) VALUES ('x')")
    assert has_live_connection(db), "transaction scope must not untrack"
    conn.close()
    assert not has_live_connection(db)

    # double close is idempotent (must not under-count into negatives)
    dup = connect_tracked(db)
    dup.close()
    dup.close()
    assert not has_live_connection(db)

    # nested lifetimes: still live until the last one closes
    first = connect_tracked(db)
    second = connect_tracked(db)
    first.close()
    assert has_live_connection(db)
    second.close()
    assert not has_live_connection(db)

    # churn must not drift
    for _ in range(100):
        connect_tracked(db).close()
    assert not has_live_connection(db)


def test_custom_primary_close_with_native_success_consumes_tracking_once(tmp_path, clean_registry):
    """A custom-primary error survives direct native retirement exactly once.

    A raising custom hook does not by itself prove that the physical SQLite
    handle remains live. When direct native cleanup succeeds, v3 C03 requires
    the original error to survive while tracking is consumed, raw inspection is
    safe again, and a repeat close cannot redispatch or underflow.
    """
    import hermes_cli.sqlite_safe_read as safe_read
    from hermes_cli.sqlite_safe_read import connect_tracked

    class ControllableConnection(sqlite3.Connection):
        custom_close_calls = 0

        def close(self):
            type(self).custom_close_calls += 1
            if getattr(self, "_hermes_fail_close", False):
                raise sqlite3.ProgrammingError(
                    "SQLite objects created in a thread can only be used in "
                    "that same thread"
                )
            return super().close()

    db = tmp_path / "state.db"
    _make_db(db, "WAL")

    conn = connect_tracked(db, factory=ControllableConnection)
    assert has_live_connection(db)
    assert read_header_bytes_preopen(db, length=16) is None

    conn._hermes_fail_close = True
    with pytest.raises(sqlite3.ProgrammingError, match="that same thread"):
        conn.close()

    assert getattr(type(conn), "custom_close_calls") == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        sqlite3.Connection.execute(conn, "SELECT 1")
    assert not has_live_connection(db)
    assert safe_read._live_connections == {}
    assert read_header_bytes_preopen(db, length=16) is not None

    conn.close()
    assert getattr(type(conn), "custom_close_calls") == 1
    assert not has_live_connection(db)
    assert safe_read._live_connections == {}
    assert read_header_bytes_preopen(db, length=16) is not None


def test_custom_close_failure_runs_direct_native_cleanup_once(tmp_path, clean_registry):
    """A custom-close primary error cannot retain a physically closed handle."""
    from hermes_cli.sqlite_safe_read import connect_tracked

    class RaisingClose(sqlite3.Connection):
        custom_close_calls = 0

        def close(self):
            type(self).custom_close_calls += 1
            raise RuntimeError("custom close failed")

    db = tmp_path / "custom-close.db"
    conn = connect_tracked(db, factory=RaisingClose)
    try:
        with pytest.raises(RuntimeError, match="custom close failed"):
            conn.close()
        assert conn.custom_close_calls == 1
        assert not has_live_connection(db)
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            sqlite3.Connection.execute(conn, "SELECT 1")
    finally:
        if has_live_connection(db):
            conn.close()


def test_wrong_thread_programming_error_does_not_untrack_live_connection(tmp_path, clean_registry):
    """Thread-affinity errors are not the authoritative native closed-handle result."""
    from hermes_cli.sqlite_safe_read import connect_tracked

    db = tmp_path / "wrong-thread-close.db"
    conn = connect_tracked(db)
    errors: list[BaseException] = []

    def wrong_thread_close() -> None:
        try:
            conn.close()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=wrong_thread_close)
    worker.start()
    worker.join(2)
    try:
        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], sqlite3.ProgrammingError)
        assert has_live_connection(db)
        assert conn.execute("SELECT 1").fetchone() == (1,)
    finally:
        if has_live_connection(db):
            conn.close()
        else:
            sqlite3.Connection.close(conn)


def test_probe_reservation_refuses_racing_connect(tmp_path, clean_registry, monkeypatch):
    """A raw reservation deterministically refuses a racing tracked open.

    The probe owns a raw descriptor while the writer reaches ``connect_tracked``.
    A6 is fail-closed: the writer must receive ``LiveConnectionError`` before it
    creates a SQLite descriptor, then release the probe through an explicit
    event.  No scheduler delay or retry decides this result.
    """
    import hermes_cli.sqlite_safe_read as ssr

    db = tmp_path / "state.db"
    _make_db(db, "DELETE")

    inside_read = threading.Event()
    may_close = threading.Event()
    writer_done = threading.Event()
    refusals: list[BaseException] = []
    failures: list[BaseException] = []
    real_open = open

    def slow_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        inside_read.set()
        assert may_close.wait(5), "writer did not release the reserved probe"
        return handle

    def writer():
        try:
            if not inside_read.wait(5):
                failures.append(AssertionError("probe did not open its descriptor"))
                return
            try:
                ssr.connect_tracked(db, isolation_level=None, timeout=0.5)
            except ssr.LiveConnectionError as exc:
                refusals.append(exc)
            except BaseException as exc:
                failures.append(exc)
        finally:
            may_close.set()
            writer_done.set()

    monkeypatch.setattr(ssr, "open", slow_open, raising=False)
    thread = threading.Thread(target=writer)
    thread.start()
    try:
        header = read_header_bytes_preopen(db, length=16)
    finally:
        may_close.set()  # never wedge the probe if the writer failed
    assert writer_done.wait(5), "writer did not finish"
    thread.join(5)

    assert not thread.is_alive(), "writer thread did not terminate"
    assert not failures, repr(failures)
    assert len(refusals) == 1
    assert header == b"SQLite format 3\x00"




def test_session_db_read_only_is_tracked(tmp_path, clean_registry, monkeypatch):
    """End-to-end: a real read-only SessionDB blocks byte-probes."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.create_session("s1", source="cli")
    seed.close()

    ro = SessionDB(db_path=db_path, read_only=True)
    try:
        assert has_live_connection(db_path)
        assert read_header_bytes_preopen(db_path, length=16) is None
    finally:
        ro.close()

    assert not has_live_connection(db_path)
    assert read_header_bytes_preopen(db_path, length=16) is not None








def test_page_count_bytes_matches_on_disk_size(tmp_path):
    """The PRAGMA route reports the same size the header field encodes."""
    db = tmp_path / "state.db"
    _make_db(db, "DELETE")

    conn = sqlite3.connect(str(db))
    try:
        logical = page_count_bytes(conn)
        assert logical is not None
        assert logical == db.stat().st_size
        assert file_length_matches_header(conn) is True
    finally:
        conn.close()


def test_file_length_check_never_reports_truncated_db_as_healthy(tmp_path):
    """A short file must not come back as a clean 'file length matches'.

    On a truncated database SQLite refuses the pragma outright, so the helper
    returns None (inconclusive) rather than False. Either way the contract that
    matters is the same: a torn file is never reported as healthy.
    """
    db = tmp_path / "state.db"
    _make_db(db, "DELETE")

    conn = sqlite3.connect(str(db))
    try:
        logical = page_count_bytes(conn)
        assert logical is not None
        assert file_length_matches_header(conn) is True

        # Truncate behind SQLite's back to simulate a torn extend.
        with open(db, "r+b") as handle:
            handle.truncate(logical // 2)

        assert file_length_matches_header(conn) is not True
    finally:
        conn.close()


def test_pending_open_refuses_raw_probe_without_holding_registry_lock(tmp_path, clean_registry):
    """An arbitrary opener is outside the registry lock but still fail-closed."""
    from hermes_cli import sqlite_safe_read as safe_read

    db = tmp_path / "pending-open.db"
    sqlite3.connect(db).close()
    entered_opener = threading.Event()
    release_opener = threading.Event()
    opened = []
    open_done = threading.Event()
    probe_done = threading.Event()
    probe_result = []

    def blocking_opener(path, **kwargs):
        entered_opener.set()
        assert release_opener.wait(5), "test opener was not released"
        return sqlite3.connect(path, **kwargs)

    def open_connection():
        try:
            opened.append(
                safe_read.connect_tracked(
                    db, connect_fn=blocking_opener, check_same_thread=False
                )
            )
        finally:
            open_done.set()

    def raw_probe():
        probe_result.append(safe_read.read_header_bytes_preopen(db))
        probe_done.set()

    opener_thread = threading.Thread(target=open_connection)
    opener_thread.start()
    assert entered_opener.wait(5), "test opener did not start"
    probe_thread = threading.Thread(target=raw_probe)
    probe_thread.start()
    try:
        assert probe_done.wait(1), "raw probe waited behind arbitrary opener"
        assert probe_result == [None]
    finally:
        release_opener.set()
        assert open_done.wait(5), "open did not complete after release"
        opener_thread.join()
        probe_thread.join()
        for connection in opened:
            connection.close()


def test_baseexception_open_failure_releases_raw_probe_reservation(tmp_path, clean_registry):
    """A rejected callable opener cannot strand its pending-open reservation."""
    from hermes_cli.sqlite_safe_read import connect_tracked

    db = tmp_path / "baseexception-open.db"
    # A bare sqlite3.connect()/close() can leave a zero-byte file; write the
    # minimum durable SQLite header before this post-failure raw-probe check.
    bootstrap = sqlite3.connect(db)
    try:
        bootstrap.execute("CREATE TABLE bootstrap(value INTEGER)")
    finally:
        bootstrap.close()

    def rejected_opener(*args, **kwargs):
        raise KeyboardInterrupt("intentional opener cancellation")

    with pytest.raises(KeyboardInterrupt, match="intentional opener cancellation"):
        connect_tracked(db, connect_fn=rejected_opener)

    assert not has_live_connection(db)
    assert read_header_bytes_preopen(db, length=16) == b"SQLite format 3\x00"


def test_raw_probe_reservation_releases_after_every_file_open_outcome(
    tmp_path, clean_registry, monkeypatch
):
    """Raw file I/O is outside the registry lock and never strands a reservation."""
    import hermes_cli.sqlite_safe_read as safe_read

    class GuardedLiveLock:
        def __init__(self):
            self._lock = threading.RLock()

        def __enter__(self):
            self._lock.acquire()
            return self

        def __exit__(self, *exc_info):
            self._lock.release()

        def held_by_current_thread(self):
            return bool(getattr(self._lock, "_is_owned")())

    db = tmp_path / "reservation-cleanup.db"
    _make_db(db, "DELETE")
    live_lock = GuardedLiveLock()
    real_open = open
    monkeypatch.setattr(safe_read, "_live_lock", live_lock)

    def checked_open(*args, **kwargs):
        assert not live_lock.held_by_current_thread(), "raw path I/O held _live_lock"
        return real_open(*args, **kwargs)

    monkeypatch.setattr(safe_read, "open", checked_open, raising=False)
    assert safe_read.read_header_bytes_preopen(db, length=16) == b"SQLite format 3\x00"
    assert safe_read._raw_file_reservations == {}

    for failure in (OSError("open failed"), RuntimeError("open failed"), KeyboardInterrupt("cancelled")):
        def failing_open(*args, _failure=failure, **kwargs):
            assert not live_lock.held_by_current_thread(), "failing path I/O held _live_lock"
            raise _failure

        with monkeypatch.context() as patch:
            patch.setattr(safe_read, "open", failing_open, raising=False)
            if isinstance(failure, OSError):
                assert safe_read.read_header_bytes_preopen(db, length=16) is None
            else:
                with pytest.raises(type(failure), match=str(failure)):
                    safe_read.read_header_bytes_preopen(db, length=16)
        assert safe_read._raw_file_reservations == {}
        assert not has_live_connection(db)


def test_connect_rejection_cleans_pending_reservations_outside_registry_lock(
    tmp_path, clean_registry, monkeypatch
):
    """Open, identity, and rejected-handle cleanup never retain ``_live_lock``."""
    import hermes_cli.sqlite_safe_read as safe_read

    class GuardedLiveLock:
        def __init__(self):
            self._lock = threading.RLock()

        def __enter__(self):
            self._lock.acquire()
            return self

        def __exit__(self, *exc_info):
            self._lock.release()

        def held_by_current_thread(self):
            return bool(getattr(self._lock, "_is_owned")())

    db = tmp_path / "requested.db"
    other = tmp_path / "rejected.db"
    _make_db(db, "DELETE")
    _make_db(other, "DELETE")
    live_lock = GuardedLiveLock()
    monkeypatch.setattr(safe_read, "_live_lock", live_lock)
    real_identity = safe_read._post_open_identity
    real_rejected_close = safe_read.close_rejected_handle_once

    def checked_identity(connection):
        assert not live_lock.held_by_current_thread(), "SQLite identity query held _live_lock"
        return real_identity(connection)

    def checked_rejected_close(connection):
        assert not live_lock.held_by_current_thread(), "rejected-handle close held _live_lock"
        return real_rejected_close(connection)

    monkeypatch.setattr(safe_read, "_post_open_identity", checked_identity)
    monkeypatch.setattr(safe_read, "close_rejected_handle_once", checked_rejected_close)

    def checked_opener(path, **kwargs):
        assert not live_lock.held_by_current_thread(), "SQLite opener held _live_lock"
        return sqlite3.connect(path, **kwargs)

    connection = safe_read.connect_tracked(db, connect_fn=checked_opener, isolation_level=None)
    connection.close()
    assert not has_live_connection(db)

    for failure in (RuntimeError("open failed"), KeyboardInterrupt("open cancelled")):
        def failing_opener(*args, _failure=failure, **kwargs):
            assert not live_lock.held_by_current_thread(), "failing opener held _live_lock"
            raise _failure

        with pytest.raises(type(failure), match=str(failure)):
            safe_read.connect_tracked(db, connect_fn=failing_opener)
        assert safe_read._pending_unresolved_opens == 0
        assert safe_read._raw_file_reservations == {}
        assert not has_live_connection(db)

    rejected = []

    def wrong_database_opener(*args, **kwargs):
        assert not live_lock.held_by_current_thread(), "wrong-identity opener held _live_lock"
        connection = sqlite3.connect(other, **kwargs)
        rejected.append(connection)
        return connection

    with pytest.raises(safe_read.UntrackableConnectionError, match="different database path"):
        safe_read.connect_tracked(db, connect_fn=wrong_database_opener, isolation_level=None)

    assert len(rejected) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        sqlite3.Connection.execute(rejected[0], "SELECT 1")
    assert safe_read._pending_unresolved_opens == 0
    assert safe_read._raw_file_reservations == {}
    assert not has_live_connection(db)
    assert not has_live_connection(other)
