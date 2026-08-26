"""Focused regressions for the serialized SQLite connection boundary."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import textwrap
import threading

import pytest

import hermes_state
from hermes_state import SessionDB, _connect_tracked_db
from hermes_cli.sqlite_safe_read import (
    LiveConnectionError,
    UntrackableConnectionError,
    connect_tracked,
    has_live_connection,
    offline_file_access,
    read_header_bytes_preopen,
)

SQLiteSerializationError = getattr(
    hermes_state, "SQLiteSerializationError", RuntimeError
)


class _ForeignAcquireProbe:
    """Observe one foreign attempt against the real connection serial lock."""

    def __init__(self, monkeypatch, connection):
        self._lock = connection._hermes_serial_lock
        self._foreign_threads: set[int] = set()
        self._status: list[str] = []
        self._observed = threading.Event()
        self._release = threading.Event()
        original_acquire = type(self._lock).acquire

        def observed_acquire(lock, *args, **kwargs):
            is_foreign_probe = (
                lock is self._lock
                and threading.get_ident() in self._foreign_threads
                and not self._status
            )
            if not is_foreign_probe:
                return original_acquire(lock, *args, **kwargs)

            # A non-blocking attempt gives a deterministic state result: on the
            # broken boundary the foreign thread owns the real lock immediately;
            # on the corrected boundary it must take the normal blocking path.
            acquired = original_acquire(lock, blocking=False)
            self._status.append("acquired" if acquired else "blocked")
            self._observed.set()
            if acquired:
                if not self._release.wait(2):
                    raise AssertionError("owner did not release the foreign probe")
                return True
            return original_acquire(lock, *args, **kwargs)

        monkeypatch.setattr(type(self._lock), "acquire", observed_acquire)

    def register_current_thread(self) -> None:
        self._foreign_threads.add(threading.get_ident())

    def await_status(self) -> str:
        assert self._observed.wait(2), "foreign thread never reached serial acquire"
        assert len(self._status) == 1
        return self._status[0]

    def release(self) -> None:
        self._release.set()


def _assert_waits_for_serial_lock(conn, call, expected_exception=None) -> None:
    """Prove ``call`` cannot enter SQLite while another thread owns the lock."""
    started = threading.Event()
    finished = threading.Event()
    failures: list[BaseException] = []

    def invoke() -> None:
        started.set()
        try:
            call()
        except BaseException as exc:  # surfaced after the deterministic join
            failures.append(exc)
        finally:
            finished.set()

    conn._hermes_serial_lock.acquire()
    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    try:
        assert started.wait(2)
        assert not finished.wait(0.2), "SQLite entry bypassed the serial lock"
    finally:
        conn._hermes_serial_lock.release()
    assert finished.wait(2)
    worker.join(2)
    assert not worker.is_alive()
    if expected_exception is None:
        assert not failures
    else:
        assert len(failures) == 1
        assert isinstance(failures[0], expected_exception)


def test_authorizer_registration_waits_for_connection_serial_lock(tmp_path):
    """Callback-registration APIs are SQLite entries, not Python-only setters."""
    conn = _connect_tracked_db(tmp_path / "state.db", check_same_thread=False)
    try:
        _assert_waits_for_serial_lock(
            conn,
            lambda: conn.set_authorizer(lambda *_args: 0),
        )
    finally:
        conn.close()


def test_connection_context_entry_waits_for_connection_serial_lock(tmp_path):
    """Both halves of ``with conn:`` stay inside the serialization boundary."""
    conn = _connect_tracked_db(tmp_path / "state.db", check_same_thread=False)
    try:
        _assert_waits_for_serial_lock(conn, conn.__enter__)
        _assert_waits_for_serial_lock(
            conn,
            lambda: conn.__exit__(None, None, None),
        )
    finally:
        conn.close()


def test_administrative_connection_entries_wait_for_connection_serial_lock(tmp_path):
    """Every exposed administrative SQLite API shares the same fence."""
    conn = _connect_tracked_db(tmp_path / "state.db", check_same_thread=False)
    try:
        conn.execute("CREATE TABLE t(value TEXT)")
        image = conn.serialize()

        class Window:
            def step(self, value):
                pass

            def value(self):
                return 0

            def inverse(self, value):
                pass

            def finalize(self):
                return 0

        calls = [
            ("create_window_function", lambda: conn.create_window_function("w", 1, Window)),
            ("set_progress_handler", lambda: conn.set_progress_handler(None, 0)),
            ("set_trace_callback", lambda: conn.set_trace_callback(None)),
            ("enable_load_extension", lambda: conn.enable_load_extension(False)),
            ("serialize", conn.serialize),
            ("deserialize", lambda: conn.deserialize(image)),
            ("in_transaction", lambda: conn.in_transaction),
            ("total_changes", lambda: conn.total_changes),
            ("isolation_level get", lambda: conn.isolation_level),
            (
                "isolation_level set",
                lambda: setattr(conn, "isolation_level", conn.isolation_level),
            ),
            (
                "getlimit",
                lambda: conn.getlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH),
            ),
            (
                "setlimit",
                lambda: conn.setlimit(
                    sqlite3.SQLITE_LIMIT_SQL_LENGTH,
                    conn.getlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH),
                ),
            ),
        ]
        if hasattr(conn, "getconfig"):
            category = sqlite3.SQLITE_DBCONFIG_ENABLE_FKEY
            calls.extend(
                [
                    ("getconfig", lambda: conn.getconfig(category)),
                    ("setconfig", lambda: conn.setconfig(category, conn.getconfig(category))),
                ]
            )
        if hasattr(conn, "autocommit"):
            autocommit_target = conn.autocommit
            calls.append(
                (
                    "autocommit set",
                    lambda: setattr(conn, "autocommit", autocommit_target),
                )
            )

        for name, call in calls:
            _assert_waits_for_serial_lock(conn, call), name

        # ``load_extension`` reaches SQLite even when the module cannot exist.
        conn.enable_load_extension(True)
        try:
            _assert_waits_for_serial_lock(
                conn,
                lambda: conn.load_extension("__hermes_missing_extension__"),
                sqlite3.OperationalError,
            )
        finally:
            conn.enable_load_extension(False)

        dump = conn.iterdump()
        _assert_waits_for_serial_lock(conn, lambda: next(dump))
    finally:
        conn.close()


def test_tracking_close_never_holds_live_lock_while_waiting_for_serial_lock(
    tmp_path, monkeypatch
):
    """A re-entrant callback can query tracking while another thread closes."""
    path = tmp_path / "state.db"
    conn = _connect_tracked_db(path, check_same_thread=False)
    serial_close_entered = threading.Event()
    close_attempting = threading.Event()
    close_finished = threading.Event()
    query_finished = threading.Event()
    errors: list[BaseException] = []
    serial_lock = conn._hermes_serial_lock
    original_acquire = type(serial_lock).acquire

    def observed_acquire(self, *args, **kwargs):
        # The tracking boundary itself now waits for this lock, before either
        # the polymorphic close hook or direct native close can run.
        if self is serial_lock and close_attempting.is_set():
            serial_close_entered.set()
        return original_acquire(self, *args, **kwargs)

    monkeypatch.setattr(type(serial_lock), "acquire", observed_acquire)

    def close_in_worker() -> None:
        try:
            close_attempting.set()
            conn.close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            close_finished.set()

    def query_live_registry() -> None:
        try:
            assert has_live_connection(path)
        except BaseException as exc:
            errors.append(exc)
        finally:
            query_finished.set()

    conn._hermes_serial_lock.acquire()
    closer = threading.Thread(target=close_in_worker, daemon=True)
    query = None
    closer.start()
    try:
        assert serial_close_entered.wait(2)
        query = threading.Thread(target=query_live_registry, daemon=True)
        query.start()
        assert query_finished.wait(0.2), (
            "tracking close held _live_lock while waiting for the serial lock"
        )
    finally:
        conn._hermes_serial_lock.release()
    assert close_finished.wait(2)
    closer.join(2)
    assert query is not None
    query.join(2)
    assert not closer.is_alive()
    assert not query.is_alive()
    assert not errors
    assert not has_live_connection(path)


def test_serialized_custom_close_hook_runs_under_its_owner_lock(tmp_path):
    """The compatibility hook shares the serial boundary with native close."""
    lock_ownership: list[bool] = []

    class NoOpClose(sqlite3.Connection):
        def close(self):
            lock_ownership.append(self._hermes_serial_lock._is_owned())

    connection = _connect_tracked_db(
        tmp_path / "serialized-custom-close.db",
        factory=NoOpClose,
        isolation_level=None,
        check_same_thread=False,
    )
    connection.execute("BEGIN IMMEDIATE")
    connection.close()

    assert lock_ownership == [True]
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        sqlite3.Connection.execute(connection, "SELECT 1")


def test_custom_noop_close_keeps_tracking_until_native_handle_is_closed(tmp_path):
    """A custom close hook cannot release raw access while its handle is live."""
    import hermes_cli.sqlite_safe_read as sqlite_safe_read

    database = tmp_path / "custom-noop-close.db"
    close_calls: list[sqlite3.Connection] = []

    class NoOpClose(sqlite3.Connection):
        def close(self):
            close_calls.append(self)

    with sqlite_safe_read._live_lock:
        baseline_counts = dict(sqlite_safe_read._live_connections)
        baseline_pending = sqlite_safe_read._pending_unresolved_opens

    connection = connect_tracked(
        database,
        factory=NoOpClose,
        isolation_level=None,
    )
    try:
        connection.execute("BEGIN IMMEDIATE")
        assert has_live_connection(database)
        assert read_header_bytes_preopen(database, length=16) is None

        connection.close()

        # The factory contract remains polymorphic, but a non-raising no-op
        # must not make this live SQLite descriptor disappear from the guard.
        assert close_calls == [connection]
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            sqlite3.Connection.execute(connection, "SELECT 1")
        assert not has_live_connection(database)
        assert read_header_bytes_preopen(database, length=16) is not None
        with offline_file_access(database, what="post-close regression"):
            pass

        # A completed close is idempotent: neither the custom hook nor the
        # registry count is consumed again.
        connection.close()
        assert close_calls == [connection]
        with sqlite_safe_read._live_lock:
            assert sqlite_safe_read._live_connections == baseline_counts
            assert sqlite_safe_read._pending_unresolved_opens == baseline_pending
    finally:
        try:
            sqlite3.Connection.close(connection)
        except sqlite3.Error:
            pass


def test_close_rejects_owner_spoof_before_any_close_boundary(tmp_path):
    """A post-open owner spoof cannot authorize custom or native close."""
    _run_sqlite_child(
        """
        import pathlib
        import sqlite3
        import sys
        import threading

        import hermes_cli.sqlite_safe_read as sqlite_safe_read
        from hermes_cli.sqlite_safe_read import (
            UntrackableConnectionError,
            has_live_connection,
            read_header_bytes_preopen,
        )
        from hermes_state import SQLiteSerializationError, _connect_tracked_db

        database = pathlib.Path(sys.argv[1]) / "owner-spoof-close.db"
        close_calls = []
        spoof_entered = threading.Event()
        failures = []

        class NoOpClose(sqlite3.Connection):
            def close(self):
                close_calls.append(self)

        class OwnerLookalike:
            def __init__(self, owner):
                self.owner = owner

            def __enter__(self):
                spoof_entered.set()
                return self

            def __exit__(self, *_exc_info):
                return False

        with sqlite_safe_read._live_lock:
            baseline_counts = dict(sqlite_safe_read._live_connections)
            baseline_pending = sqlite_safe_read._pending_unresolved_opens

        connection = _connect_tracked_db(
            database,
            factory=NoOpClose,
            check_same_thread=False,
        )
        genuine_lock = connection._hermes_serial_lock
        genuine_lock.acquire()
        genuine_held = True
        worker = None
        try:
            spoof = OwnerLookalike(connection)
            assert spoof.owner is connection
            assert type(spoof) is not type(genuine_lock)
            connection._hermes_serial_lock = spoof

            def close_worker():
                try:
                    connection.close()
                except BaseException as exc:
                    failures.append(exc)

            worker = threading.Thread(target=close_worker, daemon=True)
            worker.start()
            worker.join(1)
            assert not worker.is_alive(), "spoofed close did not fail promptly"
            assert len(failures) == 1
            assert isinstance(failures[0], (SQLiteSerializationError, UntrackableConnectionError))
            assert not spoof_entered.is_set(), "close entered the owner-spoofed lock"
            assert close_calls == [], "close reached the custom hook under a spoof"

            assert sqlite3.Connection.execute(connection, "SELECT 1").fetchone() == (1,)
            assert has_live_connection(database)
            assert read_header_bytes_preopen(database, length=16) is None

            connection._hermes_serial_lock = genuine_lock
            assert connection._hermes_serial_lock is genuine_lock
            genuine_lock.release()
            genuine_held = False
            connection.close()
            assert close_calls == [connection]

            try:
                sqlite3.Connection.execute(connection, "SELECT 1")
            except sqlite3.ProgrammingError:
                pass
            else:
                raise AssertionError("native SQLite handle remained usable after close")
            assert not has_live_connection(database)
            assert read_header_bytes_preopen(database, length=16) is not None

            connection.close()
            assert close_calls == [connection]
            with sqlite_safe_read._live_lock:
                assert sqlite_safe_read._live_connections == baseline_counts
                assert sqlite_safe_read._pending_unresolved_opens == baseline_pending
        finally:
            if connection._hermes_serial_lock is not genuine_lock:
                connection._hermes_serial_lock = genuine_lock
            if genuine_held:
                genuine_lock.release()
            if worker is not None:
                worker.join(2)
            try:
                sqlite3.Connection.close(connection)
            except sqlite3.Error:
                pass
        """,
        tmp_path,
        timeout=8,
    )


def _run_missing_serial_lock_close_regression(tmp_path, mutation):
    """A lost serialized lock must not authorize custom or native close."""
    _run_sqlite_child(
        f"""
        import pathlib
        import sqlite3
        import sys
        import threading

        import hermes_cli.sqlite_safe_read as sqlite_safe_read
        from hermes_cli.sqlite_safe_read import (
            UntrackableConnectionError,
            has_live_connection,
            read_header_bytes_preopen,
        )
        from hermes_state import SQLiteSerializationError, _connect_tracked_db

        mutation = {mutation!r}
        database = pathlib.Path(sys.argv[1]) / (mutation + "-close.db")
        close_calls = []
        failures = []

        class NoOpClose(sqlite3.Connection):
            def close(self):
                close_calls.append(self)

        with sqlite_safe_read._live_lock:
            baseline_counts = dict(sqlite_safe_read._live_connections)
            baseline_pending = sqlite_safe_read._pending_unresolved_opens

        connection = _connect_tracked_db(
            database,
            factory=NoOpClose,
            check_same_thread=False,
        )
        genuine_lock = connection._hermes_serial_lock
        genuine_lock.acquire()
        genuine_held = True
        worker = None
        try:
            if mutation == "none":
                connection._hermes_serial_lock = None
                assert connection._hermes_serial_lock is None
            else:
                del connection._hermes_serial_lock
                assert not hasattr(connection, "_hermes_serial_lock")

            def close_worker():
                try:
                    connection.close()
                except BaseException as exc:
                    failures.append(exc)

            worker = threading.Thread(target=close_worker, daemon=True)
            worker.start()
            worker.join(1)
            assert not worker.is_alive(), mutation + " close did not fail promptly"
            assert len(failures) == 1
            assert isinstance(
                failures[0], (SQLiteSerializationError, UntrackableConnectionError)
            )
            assert close_calls == [], "close reached the custom hook"

            # The held genuine lock and live descriptor must survive rejection.
            assert sqlite3.Connection.execute(connection, "SELECT 1").fetchone() == (1,)
            assert has_live_connection(database)
            assert read_header_bytes_preopen(database, length=16) is None

            connection._hermes_serial_lock = genuine_lock
            assert connection._hermes_serial_lock is genuine_lock
            genuine_lock.release()
            genuine_held = False
            connection.close()
            assert close_calls == [connection]

            try:
                sqlite3.Connection.execute(connection, "SELECT 1")
            except sqlite3.ProgrammingError:
                pass
            else:
                raise AssertionError("native SQLite handle remained usable after close")
            assert not has_live_connection(database)
            assert read_header_bytes_preopen(database, length=16) is not None

            connection.close()
            assert close_calls == [connection]
            with sqlite_safe_read._live_lock:
                assert sqlite_safe_read._live_connections == baseline_counts
                assert sqlite_safe_read._pending_unresolved_opens == baseline_pending
        finally:
            if getattr(connection, "_hermes_serial_lock", None) is not genuine_lock:
                connection._hermes_serial_lock = genuine_lock
            if genuine_held:
                genuine_lock.release()
            if worker is not None:
                worker.join(2)
            try:
                sqlite3.Connection.close(connection)
            except sqlite3.Error:
                pass
        """,
        tmp_path,
        timeout=8,
    )


def test_close_rejects_none_serial_lock_before_any_close_boundary(tmp_path):
    _run_missing_serial_lock_close_regression(tmp_path, "none")


def test_close_rejects_missing_serial_lock_before_any_close_boundary(tmp_path):
    _run_missing_serial_lock_close_regression(tmp_path, "missing")


def test_custom_close_error_stays_primary_after_native_cleanup(tmp_path):
    """A raising compatibility hook cannot leave a closed handle registered."""
    import hermes_cli.sqlite_safe_read as sqlite_safe_read

    database = tmp_path / "custom-raising-close.db"
    close_calls: list[sqlite3.Connection] = []

    class CustomCloseError(RuntimeError):
        pass

    class RaisingClose(sqlite3.Connection):
        def close(self):
            close_calls.append(self)
            raise CustomCloseError("custom close failed")

    with sqlite_safe_read._live_lock:
        baseline_counts = dict(sqlite_safe_read._live_connections)
        baseline_pending = sqlite_safe_read._pending_unresolved_opens

    connection = connect_tracked(database, factory=RaisingClose, isolation_level=None)
    try:
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(CustomCloseError, match="custom close failed") as raised:
            connection.close()

        assert close_calls == [connection]
        assert getattr(raised.value, "_hermes_cleanup_error", None) is None
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            sqlite3.Connection.execute(connection, "SELECT 1")
        assert not has_live_connection(database)

        connection.close()
        assert close_calls == [connection]
        with sqlite_safe_read._live_lock:
            assert sqlite_safe_read._live_connections == baseline_counts
            assert sqlite_safe_read._pending_unresolved_opens == baseline_pending
    finally:
        try:
            sqlite3.Connection.close(connection)
        except sqlite3.Error:
            pass


def test_execute_shortcut_and_cursor_iteration_remain_serialized(tmp_path):
    """CPython 3.11's C shortcut cannot manufacture a plain cursor for Hermes."""
    class ProbeConnection(sqlite3.Connection):
        cursor_called = False

        def cursor(self, factory=None):
            self.cursor_called = True
            return super().cursor(factory)

    raw = sqlite3.connect(":memory:", factory=ProbeConnection)
    try:
        raw.execute("SELECT 1").fetchone()
        assert not raw.cursor_called, "this interpreter no longer has the C shortcut"
    finally:
        raw.close()

    conn = _connect_tracked_db(tmp_path / "state.db", check_same_thread=False)
    try:
        conn.execute("CREATE TABLE t(value INTEGER)")
        conn.executemany("INSERT INTO t(value) VALUES (?)", [(1,), (2,)])
        cursor = conn.execute("SELECT value FROM t ORDER BY value")
        assert isinstance(cursor, sqlite3.Cursor)
        _assert_waits_for_serial_lock(conn, cursor.fetchone)
        _assert_waits_for_serial_lock(conn, lambda: next(cursor))
    finally:
        conn.close()


def test_custom_connection_and_cursor_factories_stay_serialized(tmp_path, monkeypatch):
    """Both requested and opener-substituted Python factories are validated."""
    import hermes_state

    class CustomCursor(sqlite3.Cursor):
        execute_calls = 0

        def execute(self, *args, **kwargs):
            self.execute_calls += 1
            return super().execute(*args, **kwargs)

    class CustomConnection(sqlite3.Connection):
        seen_cursor_factory = None

        def cursor(self, factory=None):
            self.seen_cursor_factory = factory
            return super().cursor(factory)

    direct = _connect_tracked_db(
        tmp_path / "direct.db",
        factory=CustomConnection,
        check_same_thread=False,
    )
    try:
        assert isinstance(direct, CustomConnection)
        cursor = direct.cursor(factory=CustomCursor)
        assert isinstance(cursor, CustomCursor)
        _assert_waits_for_serial_lock(cursor.connection, lambda: cursor.execute("SELECT 1"))
        assert cursor.execute_calls == 1
        assert direct.seen_cursor_factory is not None
    finally:
        direct.close()

    class RequestedConnection(sqlite3.Connection):
        pass

    class SubstituteConnection(sqlite3.Connection):
        pass

    raw_connect = sqlite3.connect
    received = {}

    def substitute_opener(*args, **kwargs):
        received["factory"] = kwargs.get("factory")
        kwargs["factory"] = SubstituteConnection
        return raw_connect(*args, **kwargs)

    monkeypatch.setattr(hermes_state.sqlite3, "connect", substitute_opener)
    substituted = hermes_state._connect_tracked_db(
        tmp_path / "substituted.db",
        factory=RequestedConnection,
        check_same_thread=False,
    )
    try:
        assert received["factory"] is not RequestedConnection
        assert isinstance(substituted, SubstituteConnection)
        _assert_waits_for_serial_lock(substituted, lambda: substituted.execute("SELECT 1"))
    finally:
        substituted.close()


def test_unserializable_cursor_factory_fails_closed(tmp_path):
    """A custom connection that returns a base Cursor cannot bypass the lock."""
    class RawCursorConnection(sqlite3.Connection):
        def cursor(self, factory=None):
            return super().cursor()

    conn = _connect_tracked_db(
        tmp_path / "unsafe-cursor.db",
        factory=RawCursorConnection,
        check_same_thread=False,
    )
    try:
        with pytest.raises(SQLiteSerializationError, match="cannot be serialized"):
            conn.cursor()
    finally:
        conn.close()


def test_non_connection_opener_result_fails_closed(tmp_path):
    """An opener cannot smuggle an arbitrary object past either handle guard."""
    class NotAConnection:
        def close(self):
            pass

    with pytest.raises(UntrackableConnectionError, match="not a sqlite3.Connection"):
        connect_tracked(
            tmp_path / "not-a-connection.db",
            connect_fn=lambda *_args, **_kwargs: NotAConnection(),
        )


def test_opener_substitution_to_nonretrofittable_base_connection_fails_closed(
    tmp_path, monkeypatch
):
    """A factory replacement must never return a base Connection unwrapped."""
    raw_connect = sqlite3.connect

    def substitute_base_connection(*args, **kwargs):
        kwargs["factory"] = sqlite3.Connection
        return raw_connect(*args, **kwargs)

    monkeypatch.setattr(hermes_state.sqlite3, "connect", substitute_base_connection)
    with pytest.raises(UntrackableConnectionError, match="cannot release its tracking"):
        hermes_state._connect_tracked_db(
            tmp_path / "unsafe-base-connection.db", check_same_thread=False
        )


def test_blob_surface_waits_for_connection_serial_lock(tmp_path):
    """The non-subclassable Blob object is safely proxied across its full API."""
    conn = _connect_tracked_db(tmp_path / "blob.db", check_same_thread=False)
    try:
        conn.execute("CREATE TABLE blobs(value BLOB)")
        conn.execute("INSERT INTO blobs(value) VALUES (zeroblob(8))")
        rowid = conn.execute("SELECT rowid FROM blobs").fetchone()[0]

        def fresh_blob():
            return conn.blobopen("blobs", "value", rowid)

        calls = [
            ("read", lambda blob: blob.read(1), True),
            ("write", lambda blob: blob.write(b"x"), True),
            ("seek", lambda blob: blob.seek(0), True),
            ("tell", lambda blob: blob.tell(), True),
            ("length", lambda blob: len(blob), True),
            ("index", lambda blob: blob[0], True),
            ("index write", lambda blob: blob.__setitem__(0, 1), True),
            ("context enter", lambda blob: blob.__enter__(), True),
            ("context exit", lambda blob: blob.__exit__(None, None, None), False),
            ("close", lambda blob: blob.close(), False),
        ]
        for name, call, close_after in calls:
            blob = fresh_blob()
            _assert_waits_for_serial_lock(conn, lambda call=call, blob=blob: call(blob)), name
            if close_after:
                blob.close()

        _assert_waits_for_serial_lock(conn, fresh_blob)
    finally:
        conn.close()


def test_backup_serializes_both_connections_in_one_total_order(tmp_path):
    """A serial target is required even when native backup is overridden."""
    class BackupConnection(sqlite3.Connection):
        def backup(self, *args, **kwargs):
            return ("native-backup-result", self._hermes_serial_lock._recursion_count())

    source = _connect_tracked_db(
        tmp_path / "source.db", factory=BackupConnection, check_same_thread=False
    )
    target = _connect_tracked_db(
        tmp_path / "target.db", factory=BackupConnection, check_same_thread=False
    )

    try:
        assert source.backup(target) == ("native-backup-result", 1)
        assert target.backup(source) == ("native-backup-result", 1)
        # Identical locks are acquired exactly once, before native dispatch.
        assert source.backup(source) == ("native-backup-result", 1)

        unsafe = sqlite3.connect(tmp_path / "unsafe-backup.db")
        try:
            with pytest.raises(SQLiteSerializationError, match="destination"):
                source.backup(unsafe)
        finally:
            unsafe.close()
    finally:
        source.close()
        target.close()


def _exception_shape(call):
    with pytest.raises(BaseException) as raised:
        call()
    return type(raised.value), str(raised.value)


def _run_sqlite_child(program: str, tmp_path, *, timeout: float = 12) -> None:
    """Run a potentially GIL-wide lock regression without risking pytest."""
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(program), str(tmp_path)],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "serialized SQLite regression child timed out and was killed: "
            f"stdout={exc.stdout!r} stderr={exc.stderr!r}"
        )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_backup_keyword_target_preserves_native_argument_errors(tmp_path):
    """Keyword targets lock both real handles without rewriting bad calls."""
    source = _connect_tracked_db(tmp_path / "source.db", check_same_thread=False)
    destination = _connect_tracked_db(
        tmp_path / "destination.db", check_same_thread=False
    )
    raw_source = sqlite3.connect(":memory:")
    raw_destination = sqlite3.connect(":memory:")
    try:
        source.execute("CREATE TABLE copied(value TEXT)")
        source.execute("INSERT INTO copied(value) VALUES ('native target')")
        source.commit()
        source.backup(target=destination)
        assert destination.execute("SELECT value FROM copied").fetchall() == [
            ("native target",)
        ]
        _assert_waits_for_serial_lock(
            destination, lambda: source.backup(target=destination)
        )

        malformed = [
            (lambda conn, target: conn.backup(), lambda conn, target: conn.backup()),
            (
                lambda conn, target: conn.backup(target, target=target),
                lambda conn, target: conn.backup(target, target=target),
            ),
            (
                lambda conn, target: conn.backup(target=None),
                lambda conn, target: conn.backup(target=None),
            ),
            (
                lambda conn, target: conn.backup(target, unexpected=True),
                lambda conn, target: conn.backup(target, unexpected=True),
            ),
        ]
        for raw_call, serialized_call in malformed:
            assert _exception_shape(lambda: raw_call(raw_source, raw_destination)) == _exception_shape(
                lambda: serialized_call(source, destination)
            )

    finally:
        raw_source.close()
        raw_destination.close()
        source.close()
        destination.close()


def test_reverse_native_backups_are_timeout_bounded_and_ordered(tmp_path):
    """Real reverse backups must finish or fail closed, never lock-order hang."""
    _run_sqlite_child(
        """
        import pathlib
        import sys
        import sqlite3
        import threading

        from hermes_state import _connect_tracked_db

        root = pathlib.Path(sys.argv[1])
        first = _connect_tracked_db(root / "first.db", check_same_thread=False)
        second = _connect_tracked_db(root / "second.db", check_same_thread=False)
        first.execute("CREATE TABLE data(value)")
        first.execute("INSERT INTO data VALUES ('first')")
        second.execute("CREATE TABLE data(value)")
        second.execute("INSERT INTO data VALUES ('second')")
        first.commit()
        second.commit()
        barrier = threading.Barrier(2)
        errors = []

        def copy(source, destination):
            try:
                barrier.wait(4)
                for _ in range(20):
                    source.backup(target=destination)
            except sqlite3.Error as exc:
                errors.append(exc)

        left = threading.Thread(target=copy, args=(first, second), daemon=True)
        right = threading.Thread(target=copy, args=(second, first), daemon=True)
        left.start()
        right.start()
        left.join(6)
        right.join(6)
        assert not left.is_alive() and not right.is_alive()
        assert all(isinstance(exc, sqlite3.Error) for exc in errors)
        first.close()
        second.close()
        print("NATIVE-BACKUP-ORDER-OK")
        """,
        tmp_path,
    )


def test_reentrant_backup_refuses_lower_ranked_destination_before_acquire(tmp_path):
    """A UDF cannot invert the total order by already owning its source lock."""
    _run_sqlite_child(
        """
        import pathlib
        import sqlite3
        import sys

        from hermes_state import _connect_tracked_db

        root = pathlib.Path(sys.argv[1])
        first = _connect_tracked_db(root / "first.db", check_same_thread=False)
        second = _connect_tracked_db(root / "second.db", check_same_thread=False)
        source, destination = sorted(
            (first, second),
            key=lambda conn: conn._hermes_serial_lock.rank,
            reverse=True,
        )
        seen = []

        def reentrant_backup():
            try:
                source.backup(target=destination)
            except BaseException as exc:
                seen.append(exc)
                return 1
            return 0

        source.create_function("reentrant_backup", 0, reentrant_backup)
        assert source.execute("SELECT reentrant_backup()").fetchone() == (1,)
        assert len(seen) == 1
        assert isinstance(seen[0], sqlite3.ProgrammingError)
        assert "lower-ranked destination lock" in str(seen[0])
        first.close()
        second.close()
        print("REENTRANT-BACKUP-ORDER-OK")
        """,
        tmp_path,
    )


def test_custom_shortcuts_and_every_cursor_result_stay_serialized(tmp_path):
    """Real custom overrides run once under lock and cannot leak cursors."""
    class CustomCursor(sqlite3.Cursor):
        pass

    class CustomConnection(sqlite3.Connection):
        shortcut_calls: dict[str, int]
        cursor_calls: list[tuple[tuple, dict]]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.shortcut_calls = {"execute": 0, "executemany": 0, "executescript": 0}
            self.cursor_calls = []

        def cursor(self, *args, **kwargs):
            self.cursor_calls.append((args, kwargs))
            # Deliberately accepts, ignores, and substitutes any factory.
            return super().cursor(CustomCursor)

        def execute(self, *args, **kwargs):
            assert self._hermes_serial_lock._is_owned()
            self.shortcut_calls["execute"] += 1
            return self.cursor().execute(*args, **kwargs)

        def executemany(self, *args, **kwargs):
            assert self._hermes_serial_lock._is_owned()
            self.shortcut_calls["executemany"] += 1
            return self.cursor().executemany(*args, **kwargs)

        def executescript(self, *args, **kwargs):
            assert self._hermes_serial_lock._is_owned()
            self.shortcut_calls["executescript"] += 1
            return self.cursor().executescript(*args, **kwargs)

    conn = _connect_tracked_db(
        tmp_path / "custom.db", factory=CustomConnection, check_same_thread=False
    )
    try:
        conn.execute("CREATE TABLE data(value)")
        conn.executemany("INSERT INTO data VALUES (?)", [(1,), (2,)])
        conn.executescript("INSERT INTO data VALUES (3);")
        assert conn.shortcut_calls == {"execute": 1, "executemany": 1, "executescript": 1}

        for cursor in (
            conn.cursor(),
            conn.cursor(CustomCursor),
            conn.cursor(factory=CustomCursor),
            conn.cursor(factory=lambda: None),
        ):
            assert isinstance(cursor, hermes_state._SerializedCursorMixin)
            _assert_waits_for_serial_lock(conn, lambda cursor=cursor: cursor.execute("SELECT 1"))
            cursor.close()
        assert len(conn.cursor_calls) >= 7
    finally:
        conn.close()


def test_pre_mixed_handles_and_untrusted_actual_locks_fail_closed(tmp_path):
    """Mere mixin inheritance or a lookalike lock is never sufficient."""
    seen_connections = []

    class OverrideConnection(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen_connections.append(self)

        def execute(self, *args, **kwargs):
            return super().execute(*args, **kwargs)

    class PreMixedConnection(OverrideConnection, hermes_state._SerializedConnectionMixin):
        pass

    with pytest.raises((UntrackableConnectionError, SQLiteSerializationError)):
        _connect_tracked_db(tmp_path / "premixed.db", factory=PreMixedConnection)
    assert len(seen_connections) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        sqlite3.Connection.execute(seen_connections[0], "SELECT 1")

    class OverrideCursor(sqlite3.Cursor):
        close_calls = 0

        def execute(self, *args, **kwargs):
            return super().execute(*args, **kwargs)

        def close(self):
            type(self).close_calls += 1
            return super().close()

    class PreMixedCursor(OverrideCursor, hermes_state._SerializedCursorMixin):
        pass

    class CursorFactoryConnection(sqlite3.Connection):
        def cursor(self, *args, **kwargs):
            return super().cursor(PreMixedCursor)

    conn = _connect_tracked_db(tmp_path / "premixed-cursor.db", factory=CursorFactoryConnection)
    try:
        with pytest.raises(SQLiteSerializationError, match="precedes"):
            conn.cursor()
        assert PreMixedCursor.close_calls == 0
    finally:
        conn.close()

    for label, supplied_lock in (("fake", object()), ("foreign", threading.RLock())):
        opened = []

        class SuppliedLockConnection(sqlite3.Connection):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._hermes_serial_lock = supplied_lock
                opened.append(self)

        with pytest.raises((UntrackableConnectionError, SQLiteSerializationError), match="owner-bound"):
            _connect_tracked_db(tmp_path / f"{label}-lock.db", factory=SuppliedLockConnection)
        assert len(opened) == 1
        with pytest.raises(sqlite3.ProgrammingError):
            sqlite3.Connection.execute(opened[0], "SELECT 1")


def test_rejected_arbitrary_opener_result_closes_its_own_resource_once(tmp_path):
    """Rejecting a non-Connection must close its actual resource, not a cast."""
    import io
    import hermes_cli.sqlite_safe_read as sqlite_safe_read

    class ResourceOwner:
        def __init__(self):
            self.resource = io.BytesIO(b"owned")
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            self.resource.close()

    rejected = ResourceOwner()
    with pytest.raises(UntrackableConnectionError, match="not a sqlite3.Connection"):
        connect_tracked(
            tmp_path / "rejected.db", connect_fn=lambda *_args, **_kwargs: rejected
        )
    assert rejected.close_calls == 1
    assert rejected.resource.closed
    key = sqlite_safe_read._key(tmp_path / "rejected.db")
    assert key not in sqlite_safe_read._live_connections
    assert sqlite_safe_read._pending_unresolved_opens == 0

    class CleanupFailure:
        def close(self):
            raise OSError("cleanup failed")

    with pytest.raises(UntrackableConnectionError, match="not a sqlite3.Connection") as raised:
        connect_tracked(
            tmp_path / "cleanup-failure.db",
            connect_fn=lambda *_args, **_kwargs: CleanupFailure(),
        )
    assert isinstance(getattr(raised.value, "_hermes_cleanup_error", None), OSError)


def test_opener_reservations_prevent_probe_races_and_live_lock_abba(tmp_path):
    """Custom openers run outside live-lock while their path stays busy."""
    _run_sqlite_child(
        """
        import pathlib
        import sqlite3
        import sys
        import threading

        import hermes_cli.sqlite_safe_read as ssr
        from hermes_state import _connect_tracked_db

        root = pathlib.Path(sys.argv[1])
        reserved = root / "reserved.db"
        sqlite3.connect(reserved).close()
        entered = threading.Event()
        release = threading.Event()
        opened = []
        failures = []

        def paused_opener(path, **kwargs):
            entered.set()
            assert release.wait(5)
            return sqlite3.connect(path, **kwargs)

        def open_paused():
            try:
                opened.append(
                    ssr.connect_tracked(
                        reserved, connect_fn=paused_opener, check_same_thread=False
                    )
                )
            except BaseException as exc:
                failures.append(exc)

        worker = threading.Thread(target=open_paused, daemon=True)
        worker.start()
        assert entered.wait(3)
        assert ssr.has_live_connection(reserved)
        assert ssr.read_header_bytes_preopen(reserved, length=16) is None
        release.set()
        worker.join(5)
        assert not worker.is_alive() and not failures
        opened.pop().close()
        assert not ssr.has_live_connection(reserved)

        first_entered = threading.Event()
        second_entered = threading.Event()
        two_release = threading.Event()
        two_opened = []

        def concurrent_opener(path, **kwargs):
            (first_entered if not first_entered.is_set() else second_entered).set()
            assert two_release.wait(5)
            return sqlite3.connect(path, **kwargs)

        workers = [
            threading.Thread(
                target=lambda: two_opened.append(
                    ssr.connect_tracked(
                        reserved,
                        connect_fn=concurrent_opener,
                        check_same_thread=False,
                    )
                ),
                daemon=True,
            )
            for _ in range(2)
        ]
        for child in workers:
            child.start()
        assert first_entered.wait(3) and second_entered.wait(3)
        assert ssr.has_live_connection(reserved)
        two_release.set()
        for child in workers:
            child.join(5)
            assert not child.is_alive()
        while two_opened:
            two_opened.pop().close()
        assert not ssr.has_live_connection(reserved)

        existing = _connect_tracked_db(root / "existing.db", check_same_thread=False)
        callback_entered = threading.Event()
        opener_entered = threading.Event()
        errors = []

        def callback():
            callback_entered.set()
            assert opener_entered.wait(4)
            assert ssr.has_live_connection(reserved)
            return 1

        def consulting_opener(path, **kwargs):
            opener_entered.set()
            assert existing.execute("SELECT 7").fetchone() == (7,)
            return sqlite3.connect(path, **kwargs)

        def run_callback():
            try:
                assert existing.execute("SELECT callback()").fetchone() == (1,)
            except BaseException as exc:
                errors.append(exc)

        def open_consulting():
            try:
                two_opened.append(
                    ssr.connect_tracked(
                        reserved,
                        connect_fn=consulting_opener,
                        check_same_thread=False,
                    )
                )
            except BaseException as exc:
                errors.append(exc)

        existing.create_function("callback", 0, callback)
        query = threading.Thread(target=run_callback, daemon=True)
        query.start()
        assert callback_entered.wait(3)
        opening = threading.Thread(target=open_consulting, daemon=True)
        opening.start()
        query.join(6)
        opening.join(6)
        assert not query.is_alive() and not opening.is_alive(), "live-lock/serial ABBA"
        assert not errors, errors
        two_opened.pop().close()
        existing.close()
        print("OPENER-RESERVATION-OK")
        """,
        tmp_path,
    )


def test_pending_open_globally_refuses_forced_raw_probes_and_cleans_nested_failures(
    tmp_path,
):
    """A not-yet-identified opener blocks every raw probe, not just its request path."""
    import hermes_cli.sqlite_safe_read as sqlite_safe_read

    requested = tmp_path / "requested-a.db"
    opened_elsewhere = tmp_path / "opened-b.db"
    nested_outer = tmp_path / "nested-outer.db"
    nested_inner = tmp_path / "nested-inner.db"
    for database in (requested, opened_elsewhere, nested_outer, nested_inner):
        sqlite3.connect(database).close()

    entered = threading.Event()
    release = threading.Event()
    opened_actual: list[sqlite3.Connection] = []
    returned: list[sqlite3.Connection] = []
    failures: list[BaseException] = []

    def redirected_and_paused(_path, **kwargs):
        connection = sqlite3.connect(opened_elsewhere, **kwargs)
        opened_actual.append(connection)
        entered.set()
        assert release.wait(4)
        return connection

    def open_in_worker() -> None:
        try:
            returned.append(
                connect_tracked(
                    requested,
                    connect_fn=redirected_and_paused,
                    check_same_thread=False,
                )
            )
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=open_in_worker, daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        # Even an explicit force request has no authority to race an opener
        # whose actual file identity is not known yet.
        assert (
            sqlite_safe_read.read_header_bytes_preopen(
                opened_elsewhere, length=16, force=True
            )
            is None
        )
    finally:
        release.set()
        worker.join(4)
        for connection in returned:
            connection.close()
    assert not worker.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], UntrackableConnectionError)
    with pytest.raises(sqlite3.ProgrammingError):
        sqlite3.Connection.execute(opened_actual[0], "SELECT 1")
    assert not has_live_connection(requested)
    assert not has_live_connection(opened_elsewhere)

    def rejected_inner(*_args, **_kwargs):
        return object()

    def nested_then_fail(*_args, **_kwargs):
        with pytest.raises(UntrackableConnectionError):
            connect_tracked(nested_inner, connect_fn=rejected_inner)
        raise sqlite3.OperationalError("outer opener failed after nested open")

    with pytest.raises(sqlite3.OperationalError, match="outer opener failed"):
        connect_tracked(nested_outer, connect_fn=nested_then_fail)
    # A fresh raw read proves both pending counts were consumed; a stale count
    # must remain conservative and refuse it.
    assert sqlite_safe_read.read_header_bytes_preopen(nested_outer, length=16) is not None


def test_file_prefix_is_literal_except_for_explicit_string_sqlite_uris(
    tmp_path, monkeypatch
):
    """Filesystem access never mistakes a legal ``file:`` name for a URI."""
    import hermes_cli.sqlite_safe_read as sqlite_safe_read

    monkeypatch.chdir(tmp_path)
    raw_literal = Path("file:raw-literal.db")
    raw_uri_target = Path("raw-literal.db")
    raw_literal.write_bytes(b"literal raw file")
    raw_uri_target.write_bytes(b"wrong URI target")

    # Every raw public seam treats either spelling as the literal POSIX name.
    assert read_header_bytes_preopen(raw_literal, length=32) == b"literal raw file"
    assert read_header_bytes_preopen(str(raw_literal), length=32) == b"literal raw file"
    sqlite_safe_read.track_connection(raw_literal)
    try:
        assert has_live_connection(raw_literal)
        assert has_live_connection(str(raw_literal))
        for spelling in (raw_literal, str(raw_literal)):
            assert read_header_bytes_preopen(spelling, length=32) is None
            with pytest.raises(LiveConnectionError, match="connection is open"):
                with offline_file_access(spelling):
                    pass
    finally:
        sqlite_safe_read.untrack_connection(raw_literal)

    # The default and explicit uri=False requests open and track the literal
    # name; a Path stays literal even when SQLite URI processing is enabled.
    for name, spelling, kwargs in (
        ("file:literal-default.db", "file:literal-default.db", {}),
        ("file:literal-false.db", "file:literal-false.db", {"uri": False}),
        ("file:literal-path.db", Path("file:literal-path.db"), {"uri": True}),
    ):
        literal = Path(name)
        wrong_target = Path(name.removeprefix("file:"))
        wrong_connection = sqlite3.connect(wrong_target)
        wrong_connection.execute("CREATE TABLE marker(value)")
        wrong_connection.close()
        connection = connect_tracked(spelling, **kwargs)
        try:
            actual = sqlite3.Connection.execute(
                connection, "PRAGMA database_list"
            ).fetchone()[2]
            assert actual == str(literal.resolve())
            connection.execute("CREATE TABLE literal_marker(value)")
            assert has_live_connection(literal)
            assert has_live_connection(str(literal))
            assert read_header_bytes_preopen(literal, length=16) is None
            assert read_header_bytes_preopen(wrong_target, length=16) == b"SQLite format 3\x00"
        finally:
            connection.close()
        assert read_header_bytes_preopen(literal, length=16) == b"SQLite format 3\x00"

    # In contrast, a string request with uri=True retains its SQLite URI
    # query/path semantics and is guarded under the real on-disk path.
    uri_target = tmp_path / "uri-target.db"
    uri = f"file:{uri_target}?mode=rwc"
    connection = connect_tracked(uri, tracking_path=uri_target, uri=True)
    try:
        assert sqlite3.Connection.execute(
            connection, "PRAGMA database_list"
        ).fetchone()[2] == str(uri_target.resolve())
        assert has_live_connection(uri_target)
        assert read_header_bytes_preopen(uri_target, length=16) is None
    finally:
        connection.close()

    memory_target = tmp_path / "shared-memory-target"
    memory_uri = f"file:{memory_target}?mode=memory&cache=shared"
    memory_connection = connect_tracked(memory_uri, uri=True)
    try:
        assert sqlite3.Connection.execute(
            memory_connection, "PRAGMA database_list"
        ).fetchone()[2] == ""
        assert not memory_target.exists()
    finally:
        memory_connection.close()

    plain_memory_connection = connect_tracked(":memory:")
    try:
        assert sqlite3.Connection.execute(
            plain_memory_connection, "PRAGMA database_list"
        ).fetchone()[2] == ""
    finally:
        plain_memory_connection.close()


def test_offline_file_access_refuses_same_thread_connect_before_opener(
    tmp_path,
):
    """The raw-access guard fails closed without re-entering SQLite opening."""
    import hermes_cli.sqlite_safe_read as sqlite_safe_read

    database = tmp_path / "offline-reentry.db"
    seed = sqlite3.connect(database)
    seed.execute("CREATE TABLE marker(value)")
    seed.close()
    opener_calls: list[tuple[object, dict]] = []

    def forbidden_opener(*args, **kwargs):
        opener_calls.append((args, kwargs))
        raise AssertionError("offline guard must reject before invoking an opener")

    with offline_file_access(database, what="same-thread regression"):
        with sqlite_safe_read._live_lock:
            before_counts = dict(sqlite_safe_read._live_connections)
            before_pending = sqlite_safe_read._pending_unresolved_opens
        with pytest.raises(LiveConnectionError, match="offline file access"):
            connect_tracked(database, connect_fn=forbidden_opener)
        assert not opener_calls
        with sqlite_safe_read._live_lock:
            assert sqlite_safe_read._live_connections == before_counts
            assert sqlite_safe_read._pending_unresolved_opens == before_pending
        assert not has_live_connection(database)
        with open(database, "rb") as handle:
            assert handle.read(16) == b"SQLite format 3\x00"

        # Nested raw guards remain honest, and a different thread waits on
        # the existing lifecycle lock instead of entering SQLite mid-access.
        with offline_file_access(database, what="nested same-thread regression"):
            with pytest.raises(LiveConnectionError, match="offline file access"):
                connect_tracked(database, connect_fn=forbidden_opener)

        other_started = threading.Event()
        other_opened = threading.Event()
        other_failures: list[BaseException] = []
        opened: list[sqlite3.Connection] = []

        def open_elsewhere() -> None:
            other_started.set()
            try:
                opened.append(connect_tracked(database, check_same_thread=False))
                other_opened.set()
            except BaseException as exc:
                other_failures.append(exc)

        worker = threading.Thread(target=open_elsewhere, daemon=True)
        worker.start()
        assert other_started.wait(2)
        assert not other_opened.wait(0.2), "other thread entered SQLite during raw access"

    assert other_opened.wait(2)
    worker.join(2)
    try:
        assert not worker.is_alive()
        assert not other_failures
        assert len(opened) == 1
    finally:
        for connection in opened:
            connection.close()

    # Exception cleanup and nested depth restoration leave a later open live.
    with pytest.raises(RuntimeError, match="nested cleanup"):
        with offline_file_access(database, what="exception cleanup"):
            with offline_file_access(database, what="nested exception cleanup"):
                raise RuntimeError("nested cleanup")
    connection = connect_tracked(database)
    connection.close()
    assert not has_live_connection(database)


def test_raw_probe_opens_the_single_resolved_spelling_after_symlink_retarget(
    tmp_path, monkeypatch
):
    """The raw read uses the identity checked before an alias can be retargeted."""
    import builtins

    import hermes_cli.sqlite_safe_read as sqlite_safe_read

    target_b = tmp_path / "target-b.db"
    target_c = tmp_path / "target-c.db"
    alias = tmp_path / "requested-alias.db"
    target_b.write_bytes(b"B identity must be read")
    target_c.write_bytes(b"C retarget must not be read")
    alias.symlink_to(target_b)

    real_open = builtins.open
    retargeted = False

    def retarget_before_open(opened_path, *args, **kwargs):
        nonlocal retargeted
        if not retargeted:
            alias.unlink()
            alias.symlink_to(target_c)
            retargeted = True
        return real_open(opened_path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", retarget_before_open)
    assert sqlite_safe_read.read_header_bytes_preopen(alias, length=20) == b"B identity must be r"
    assert retargeted


def test_unresolved_file_backed_identity_fails_closed_with_direct_native_close(tmp_path):
    """An authorizer-denied database_list pragma is unresolved, never memory."""
    import hermes_cli.sqlite_safe_read as sqlite_safe_read

    captured: list[sqlite3.Connection] = []

    class DenyDatabaseListConnection(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.append(self)

            def deny_database_list(action, first_argument, *_args):
                if action == sqlite3.SQLITE_PRAGMA and first_argument == "database_list":
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            sqlite3.Connection.set_authorizer(self, deny_database_list)

    database = tmp_path / "authorizer-denied.db"
    try:
        with pytest.raises(UntrackableConnectionError, match="identity"):
            connect_tracked(database, factory=DenyDatabaseListConnection)
        assert len(captured) == 1
        with pytest.raises(sqlite3.ProgrammingError):
            sqlite3.Connection.execute(captured[0], "SELECT 1")
        assert not has_live_connection(database)
        assert sqlite_safe_read.read_header_bytes_preopen(database, length=16) is not None
    finally:
        # The parent RED path returns this handle; leave no descriptor behind
        # when its expected assertion aborts the test.
        for connection in captured:
            try:
                sqlite3.Connection.close(connection)
            except sqlite3.Error:
                pass


def test_old_autocommit_helper_shape_can_mask_an_unwrapped_setter():
    """A serialized getter can make the former setter assertion pass alone."""
    setter_called = threading.Event()

    class GetterOnlySerialized:
        _hermes_serial_lock = threading.RLock()

        @property
        def autocommit(self):
            with self._hermes_serial_lock:
                return True

        @autocommit.setter
        def autocommit(self, _value):
            # Deliberately unwrapped: this is the defect the old callable hid.
            setter_called.set()

    connection = GetterOnlySerialized()
    _assert_waits_for_serial_lock(
        connection,
        lambda: setattr(connection, "autocommit", connection.autocommit),
    )
    assert setter_called.is_set()


def test_autocommit_surface_is_always_mro_dominant_and_native_setter_waits(
    tmp_path,
):
    """Autocommit is in the structural denominator even before CPython exposes it."""
    class AutocommitOnlyOverride:
        @property
        def autocommit(self):
            return "unserialized override"

        @autocommit.setter
        def autocommit(self, _value):
            pass

    class BadAutocommit(
        AutocommitOnlyOverride,
        hermes_state._SerializedConnectionMixin,
        sqlite3.Connection,
    ):
        pass

    raw = sqlite3.connect(tmp_path / "bad-autocommit.db", factory=BadAutocommit)
    try:
        with pytest.raises(SQLiteSerializationError, match="precedes"):
            hermes_state._ensure_serialized_connection(raw)
    finally:
        sqlite3.Connection.close(raw)

    if hasattr(sqlite3.Connection, "autocommit"):
        connection = _connect_tracked_db(
            tmp_path / "native-autocommit.db", check_same_thread=False
        )
        try:
            autocommit_target = connection.autocommit
            _assert_waits_for_serial_lock(
                connection,
                lambda: setattr(connection, "autocommit", autocommit_target),
            )
        finally:
            connection.close()


def test_foreign_layer_lock_swap_fails_before_waiting_on_that_lock(tmp_path):
    """A connection cannot borrow another serialized connection's lock during a UDF."""
    _run_sqlite_child(
        """
        import pathlib
        import sqlite3
        import sys
        import threading

        from hermes_state import SQLiteSerializationError, _connect_tracked_db

        root = pathlib.Path(sys.argv[1])
        first = _connect_tracked_db(root / "first.db", check_same_thread=False)
        second = _connect_tracked_db(root / "second.db", check_same_thread=False)
        first_entered = threading.Event()
        release_first = threading.Event()
        first_done = threading.Event()
        second_done = threading.Event()
        failures = []

        def swap_during_real_udf():
            # Event.wait releases the GIL while SQLite still owns first's native
            # mutex.  The second entry must reject the borrowed lock before it
            # tries to acquire the deliberately-held second lock.
            first._hermes_serial_lock = second._hermes_serial_lock
            first_entered.set()
            assert release_first.wait(5)
            return 1

        def run_first():
            try:
                assert first.execute("SELECT swap_during_real_udf()").fetchone() == (1,)
            except BaseException as exc:
                failures.append(exc)
            finally:
                first_done.set()

        def run_second():
            try:
                first.execute("SELECT 2").fetchone()
            except BaseException as exc:
                failures.append(exc)
            finally:
                second_done.set()

        first.create_function("swap_during_real_udf", 0, swap_during_real_udf)
        second._hermes_serial_lock.acquire()
        one = threading.Thread(target=run_first, daemon=True)
        two = threading.Thread(target=run_second, daemon=True)
        one.start()
        assert first_entered.wait(4)
        two.start()
        try:
            assert second_done.wait(1.5), (
                "competing entry accepted the foreign lock and waited for it"
            )
            assert len(failures) == 1
            assert isinstance(failures[0], SQLiteSerializationError)
        finally:
            release_first.set()
            second._hermes_serial_lock.release()
            one.join(4)
            two.join(4)
            sqlite3.Connection.close(first)
            second.close()
        assert first_done.is_set() and second_done.is_set()
        assert not one.is_alive() and not two.is_alive()
        print("OWNER-BOUND-SERIAL-LOCK-OK")
        """,
        tmp_path,
    )


def test_rejected_premixed_cursor_uses_native_close_without_override(tmp_path):
    """Rejected Cursor cleanup bypasses a hostile close override exactly once."""
    retained: list[sqlite3.Cursor] = []
    override_calls: list[sqlite3.Cursor] = []

    class NoOpCloseCursor(sqlite3.Cursor):
        def close(self):
            override_calls.append(self)

    class PreMixedNoOpCursor(NoOpCloseCursor, hermes_state._SerializedCursorMixin):
        pass

    class RetainingCursorConnection(sqlite3.Connection):
        def cursor(self, *args, **kwargs):
            cursor = super().cursor(PreMixedNoOpCursor)
            retained.append(cursor)
            return cursor

    connection = _connect_tracked_db(
        tmp_path / "no-op-close-cursor.db",
        factory=RetainingCursorConnection,
        check_same_thread=False,
    )
    try:
        with pytest.raises(SQLiteSerializationError, match="precedes"):
            connection.cursor()
        assert not override_calls
        assert len(retained) == 1
        with pytest.raises(sqlite3.ProgrammingError):
            sqlite3.Cursor.execute(retained[0], "SELECT 1")
    finally:
        connection.close()


def test_interrupt_remains_concurrently_callable(tmp_path):
    """``interrupt()`` must not queue behind the operation it needs to stop."""
    conn = _connect_tracked_db(tmp_path / "interrupt.db", check_same_thread=False)
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    errors: list[BaseException] = []

    def wait_in_udf():
        entered.set()
        assert release.wait(2)
        return 1

    def run_query():
        try:
            conn.execute("SELECT wait_in_udf()").fetchone()
        except BaseException as exc:
            errors.append(exc)
        finally:
            completed.set()

    try:
        conn.create_function("wait_in_udf", 0, wait_in_udf)
        worker = threading.Thread(target=run_query, daemon=True)
        worker.start()
        assert entered.wait(2)
        interrupt_done = threading.Event()
        interrupter = threading.Thread(
            target=lambda: (conn.interrupt(), interrupt_done.set()), daemon=True
        )
        interrupter.start()
        assert interrupt_done.wait(0.2), "interrupt() waited for the serial lock"
        release.set()
        assert completed.wait(2)
        worker.join(2)
        interrupter.join(2)
        assert not worker.is_alive()
        assert not interrupter.is_alive()
        assert errors and isinstance(errors[0], sqlite3.OperationalError)
    finally:
        release.set()
        conn.close()


def test_foreign_writer_refusal_stays_atomic_under_the_serial_wrapper(tmp_path):
    """Serialization does not weaken the generation trigger's refusal path."""
    path = tmp_path / "state.db"
    owner = SessionDB(db_path=path)
    foreign = sqlite3.connect(path, isolation_level=None)
    try:
        owner.create_session("owned", source="test")
        before = owner._conn.execute(
            "SELECT title FROM sessions WHERE id = 'owned'"
        ).fetchone()[0]
        with pytest.raises(sqlite3.OperationalError, match="hermes_turn_fence_generation"):
            foreign.execute("UPDATE sessions SET title = 'foreign' WHERE id = 'owned'")
        after = owner._conn.execute(
            "SELECT title FROM sessions WHERE id = 'owned'"
        ).fetchone()[0]
        assert after == before
    finally:
        foreign.close()
        owner.close()


def test_every_connection_and_cursor_shortcut_waits_for_the_serial_lock(tmp_path):
    """DB-API shortcuts, cursor fetches, iteration, and close share one lock."""
    conn = _connect_tracked_db(tmp_path / "all-methods.db", check_same_thread=False)
    try:
        _assert_waits_for_serial_lock(conn, lambda: conn.execute("CREATE TABLE t(v)"))
        _assert_waits_for_serial_lock(
            conn,
            lambda: conn.executemany("INSERT INTO t(v) VALUES (?)", [(1,), (2,), (3,)]),
        )
        _assert_waits_for_serial_lock(
            conn,
            lambda: conn.executescript("CREATE TABLE script_table(v);"),
        )

        cursor = conn.cursor()
        _assert_waits_for_serial_lock(cursor.connection, lambda: cursor.execute("SELECT v FROM t"))
        _assert_waits_for_serial_lock(cursor.connection, cursor.fetchone)
        _assert_waits_for_serial_lock(cursor.connection, cursor.fetchmany)
        _assert_waits_for_serial_lock(cursor.connection, cursor.fetchall)
        _assert_waits_for_serial_lock(cursor.connection, lambda: cursor.execute("SELECT v FROM t"))
        _assert_waits_for_serial_lock(cursor.connection, cursor.__iter__)
        _assert_waits_for_serial_lock(cursor.connection, lambda: next(cursor))
        _assert_waits_for_serial_lock(cursor.connection, cursor.close)

        another = conn.cursor()
        _assert_waits_for_serial_lock(
            another.connection,
            lambda: another.executemany("INSERT INTO t(v) VALUES (?)", [(4,), (5,)]),
        )
        _assert_waits_for_serial_lock(
            another.connection,
            lambda: another.executescript("INSERT INTO script_table(v) VALUES (1);"),
        )
        another.close()
    finally:
        conn.close()


class TestSQLiteTransactionInterference:
    def test_execute_write_excludes_foreign_connection_execute_until_commit(
        self, tmp_path, monkeypatch
    ):
        db = SessionDB(db_path=tmp_path / "r1.db")
        probe = _ForeignAcquireProbe(monkeypatch, db._conn)
        start_foreign = threading.Event()
        foreign_done = threading.Event()
        foreign_values: list[str] = []
        failures: list[BaseException] = []
        statuses: list[str] = []
        trace: list[str] = []

        db._execute_write(
            lambda conn: conn.execute(
                "INSERT INTO state_meta(key, value) VALUES ('r1', 'before')"
            )
        )
        db._conn.set_trace_callback(lambda sql: trace.append(" ".join(sql.split())))

        def foreign_read() -> None:
            probe.register_current_thread()
            assert start_foreign.wait(2), "owner never opened R1 transaction"
            try:
                row = db._conn.execute(
                    "SELECT value FROM state_meta WHERE key = 'r1'"
                ).fetchone()
                foreign_values.append(row[0])
            except BaseException as exc:
                failures.append(exc)
            finally:
                foreign_done.set()

        worker = threading.Thread(target=foreign_read, daemon=True)
        worker.start()
        try:
            def owner_write(conn):
                conn.execute("UPDATE state_meta SET value = 'committed' WHERE key = 'r1'")
                start_foreign.set()
                statuses.append(probe.await_status())
                probe.release()

            db._execute_write(owner_write)
            assert foreign_done.wait(2), "foreign SELECT did not finish after owner commit"
            worker.join(2)
            assert not worker.is_alive()
            assert not failures
            assert statuses == ["blocked"]
            assert foreign_values == ["committed"]
            commit_index = next(i for i, sql in enumerate(trace) if sql == "COMMIT")
            select_index = next(
                i for i, sql in enumerate(trace)
                if sql == "SELECT value FROM state_meta WHERE key = 'r1'"
            )
            assert commit_index < select_index
        finally:
            probe.release()
            start_foreign.set()
            worker.join(2)
            db._conn.set_trace_callback(None)
            db.close()

    def test_execute_write_does_not_commit_foreign_dml(self, tmp_path, monkeypatch):
        db = SessionDB(db_path=tmp_path / "r3.db")
        probe = _ForeignAcquireProbe(monkeypatch, db._conn)
        start_foreign = threading.Event()
        foreign_done = threading.Event()
        failures: list[BaseException] = []
        statuses: list[str] = []
        trace: list[str] = []
        db._conn.set_trace_callback(lambda sql: trace.append(" ".join(sql.split())))

        def foreign_write() -> None:
            probe.register_current_thread()
            assert start_foreign.wait(2), "owner never opened R3 transaction"
            try:
                db._conn.execute(
                    "INSERT INTO state_meta(key, value) VALUES ('r3-foreign', 'B')"
                )
            except BaseException as exc:
                failures.append(exc)
            finally:
                foreign_done.set()

        worker = threading.Thread(target=foreign_write, daemon=True)
        worker.start()
        try:
            def owner_write(conn):
                conn.execute(
                    "INSERT INTO state_meta(key, value) VALUES ('r3-owner', 'A')"
                )
                start_foreign.set()
                statuses.append(probe.await_status())
                probe.release()

            db._execute_write(owner_write)
            assert foreign_done.wait(2), "foreign DML did not finish after owner commit"
            worker.join(2)
            assert not worker.is_alive()
            assert not failures
            assert statuses == ["blocked"]
            commit_index = next(i for i, sql in enumerate(trace) if sql == "COMMIT")
            foreign_index = next(
                i for i, sql in enumerate(trace)
                if "VALUES ('r3-foreign', 'B')" in sql
            )
            assert commit_index < foreign_index
            assert db._conn.execute(
                "SELECT value FROM state_meta WHERE key = 'r3-foreign'"
            ).fetchone()[0] == "B"
        finally:
            probe.release()
            start_foreign.set()
            worker.join(2)
            db._conn.set_trace_callback(None)
            db.close()

    def test_write_transaction_rejects_foreign_rollback_theft(
        self, tmp_path, monkeypatch
    ):
        db = SessionDB(db_path=tmp_path / "r2.db")
        probe = _ForeignAcquireProbe(monkeypatch, db._conn)
        start_foreign = threading.Event()
        foreign_done = threading.Event()
        failures: list[BaseException] = []
        statuses: list[str] = []

        def foreign_rollback() -> None:
            probe.register_current_thread()
            assert start_foreign.wait(2), "owner never opened R2 transaction"
            try:
                db._conn.rollback()
            except BaseException as exc:
                failures.append(exc)
            finally:
                foreign_done.set()

        worker = threading.Thread(target=foreign_rollback, daemon=True)
        worker.start()
        try:
            with db.write_transaction() as conn:
                conn.execute(
                    "INSERT INTO state_meta(key, value) VALUES ('r2-owner', 'A')"
                )
                start_foreign.set()
                statuses.append(probe.await_status())
                probe.release()

            assert foreign_done.wait(2), "foreign rollback did not finish"
            worker.join(2)
            assert not worker.is_alive()
            assert not failures
            assert statuses == ["blocked"]
            assert db._conn.execute(
                "SELECT value FROM state_meta WHERE key = 'r2-owner'"
            ).fetchone()[0] == "A"
        finally:
            probe.release()
            start_foreign.set()
            worker.join(2)
            db.close()

    def test_write_transaction_busy_retry_does_not_retain_python_locks(
        self, tmp_path, monkeypatch
    ):
        db = SessionDB(db_path=tmp_path / "r6.db")
        blocker_held = threading.Event()
        release_blocker = threading.Event()
        blocker_released = threading.Event()
        blocker_failures: list[BaseException] = []
        availability: list[tuple[bool, bool]] = []
        owned_in_body: list[bool] = []

        def hold_external_write_lock() -> None:
            connection = sqlite3.connect(
                db.db_path,
                isolation_level=None,
                check_same_thread=False,
            )
            try:
                connection.execute("BEGIN IMMEDIATE")
                blocker_held.set()
                assert release_blocker.wait(5), "retry path did not release blocker"
                connection.rollback()
            except BaseException as exc:
                blocker_failures.append(exc)
            finally:
                connection.close()
                blocker_released.set()

        blocker = threading.Thread(target=hold_external_write_lock, daemon=True)
        blocker.start()
        assert blocker_held.wait(2), "external SQLite writer never acquired file lock"

        def release_during_retry(_deadline, _patience_s):
            db_lock_available = db._lock.acquire(blocking=False)
            if db_lock_available:
                db._lock.release()
            serial_lock_available = db._conn._hermes_serial_lock.acquire(blocking=False)
            if serial_lock_available:
                db._conn._hermes_serial_lock.release()
            availability.append((db_lock_available, serial_lock_available))
            release_blocker.set()
            assert blocker_released.wait(2), "external SQLite writer did not release"
            return True

        monkeypatch.setattr(db, "_sleep_before_write_retry", release_during_retry)
        try:
            with db.write_transaction(patience_s=2) as conn:
                owned_in_body.append(conn._hermes_serial_lock._is_owned())
                conn.execute(
                    "INSERT INTO state_meta(key, value) VALUES ('r6', 'committed')"
                )

            blocker.join(2)
            assert not blocker.is_alive()
            assert not blocker_failures
            assert availability == [(True, True)]
            assert owned_in_body == [True]
            assert db.get_meta("r6") == "committed"
        finally:
            release_blocker.set()
            blocker.join(2)
            db.close()

    def test_transaction_acquisition_busy_timeout_is_bounded_and_restored_before_retry(
        self, tmp_path, monkeypatch
    ):
        db = SessionDB(db_path=tmp_path / "busy-deadline.db")
        conn = db._conn
        assert conn is not None
        conn.execute("PRAGMA busy_timeout=1000")
        original_execute = type(conn).execute
        events: list[tuple[str, int, bool, bool]] = []

        def busy_begin(connection, sql, *args, **kwargs):
            if " ".join(sql.split()).upper() == "BEGIN IMMEDIATE":
                timeout_ms = original_execute(
                    connection, "PRAGMA busy_timeout"
                ).fetchone()[0]
                events.append(
                    (
                        "begin",
                        timeout_ms,
                        db._lock._is_owned(),
                        conn._hermes_serial_lock._is_owned(),
                    )
                )
                raise sqlite3.OperationalError("database is locked")
            return original_execute(connection, sql, *args, **kwargs)

        def refuse_retry(_deadline, _patience_s):
            timeout_ms = original_execute(conn, "PRAGMA busy_timeout").fetchone()[0]
            events.append(
                (
                    "retry",
                    timeout_ms,
                    db._lock._is_owned(),
                    conn._hermes_serial_lock._is_owned(),
                )
            )
            return False

        monkeypatch.setattr(hermes_state.time, "monotonic", lambda: 100.0)
        monkeypatch.setattr(type(conn), "execute", busy_begin)
        monkeypatch.setattr(db, "_sleep_before_write_retry", refuse_retry)

        try:
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                db._execute_write(lambda _conn: pytest.fail("write body ran"), patience_s=0.5)
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                with db.write_transaction(patience_s=0.5):
                    pytest.fail("transaction body ran")

            assert events == [
                ("begin", 500, True, True),
                ("retry", 1000, False, False),
                ("begin", 500, True, True),
                ("retry", 1000, False, False),
            ]
            assert original_execute(conn, "PRAGMA busy_timeout").fetchone()[0] == 1000
        finally:
            db.close()

    def test_manual_fail_open_root_holds_one_connection_boundary(
        self, tmp_path, monkeypatch
    ):
        db = SessionDB(db_path=tmp_path / "r7.db")
        probe = _ForeignAcquireProbe(monkeypatch, db._conn)
        start_foreign = threading.Event()
        foreign_done = threading.Event()
        failures: list[BaseException] = []
        statuses: list[str] = []
        original_drop = db._drop_all_fts_triggers

        def foreign_rollback() -> None:
            probe.register_current_thread()
            assert start_foreign.wait(2), "T3 never wrote its stale marker"
            try:
                db._conn.rollback()
            except BaseException as exc:
                failures.append(exc)
            finally:
                foreign_done.set()

        def observed_drop(cursor) -> None:
            start_foreign.set()
            statuses.append(probe.await_status())
            probe.release()
            original_drop(cursor)

        monkeypatch.setattr(db, "_drop_all_fts_triggers", observed_drop)
        worker = threading.Thread(target=foreign_rollback, daemon=True)
        worker.start()
        try:
            db._fts_enabled = True
            assert db._enter_fts_fail_open(
                sqlite3.DatabaseError("database disk image is malformed")
            ) is True
            assert foreign_done.wait(2), "foreign T3 rollback did not finish"
            worker.join(2)
            assert not worker.is_alive()
            assert not failures
            assert statuses == ["blocked"]
            assert db.get_meta(hermes_state.FTS_STALE_KEY) == "1"
        finally:
            probe.release()
            start_foreign.set()
            worker.join(2)
            db.close()

    def test_stale_fts_script_root_uses_same_boundary(self, tmp_path, monkeypatch):
        db = SessionDB(db_path=tmp_path / "r8.db")
        probe = _ForeignAcquireProbe(monkeypatch, db._conn)
        start_foreign = threading.Event()
        foreign_done = threading.Event()
        failures: list[BaseException] = []
        statuses: list[str] = []
        foreign_values: list[str | None] = []
        real_cursor = db._conn.cursor()

        class ControlledRealCursor:
            def execute(self, *args, **kwargs):
                return real_cursor.execute(*args, **kwargs)

            def executescript(self, _recovery_sql):
                db._conn.execute("BEGIN IMMEDIATE")
                db._conn.execute(
                    "INSERT INTO state_meta(key, value) VALUES ('r8-probe', 'pending')"
                )
                start_foreign.set()
                statuses.append(probe.await_status())
                probe.release()
                raise sqlite3.DatabaseError("controlled stale FTS script failure")

        def foreign_read() -> None:
            probe.register_current_thread()
            assert start_foreign.wait(2), "T4 never opened its script transaction"
            try:
                row = db._conn.execute(
                    "SELECT value FROM state_meta WHERE key = 'r8-probe'"
                ).fetchone()
                foreign_values.append(None if row is None else row[0])
            except BaseException as exc:
                failures.append(exc)
            finally:
                foreign_done.set()

        worker = threading.Thread(target=foreign_read, daemon=True)
        worker.start()
        try:
            assert db._recover_stale_fts(ControlledRealCursor(), legacy=False) is False
            assert foreign_done.wait(2), "foreign T4 read did not finish"
            worker.join(2)
            assert not worker.is_alive()
            assert not failures
            assert statuses == ["blocked"]
            assert foreign_values == [None]
        finally:
            probe.release()
            start_foreign.set()
            worker.join(2)
            real_cursor.close()
            db.close()

    def test_offline_rebuild_hides_foreign_keys_epoch_from_same_connection(
        self, tmp_path, monkeypatch
    ):
        db = SessionDB(db_path=tmp_path / "r9.db")
        probe = _ForeignAcquireProbe(monkeypatch, db._conn)
        start_foreign = threading.Event()
        foreign_done = threading.Event()
        failures: list[BaseException] = []
        statuses: list[str] = []
        foreign_keys: list[int] = []

        def foreign_read() -> None:
            probe.register_current_thread()
            assert start_foreign.wait(2), "offline epoch never disabled foreign keys"
            try:
                foreign_keys.append(
                    db._conn.execute("PRAGMA foreign_keys").fetchone()[0]
                )
            except BaseException as exc:
                failures.append(exc)
            finally:
                foreign_done.set()

        worker = threading.Thread(target=foreign_read, daemon=True)
        worker.start()
        try:
            with db.offline_rebuild(reason="R9 same-connection epoch"):
                db._execute_write(
                    lambda conn: conn.execute(
                        "INSERT INTO state_meta(key, value) VALUES ('r9', 'nested')"
                    )
                )
                start_foreign.set()
                statuses.append(probe.await_status())
                probe.release()

            assert foreign_done.wait(2), "foreign PRAGMA did not finish after epoch"
            worker.join(2)
            assert not worker.is_alive()
            assert not failures
            assert statuses == ["blocked"]
            assert foreign_keys == [1]
            assert db.get_meta("r9") == "nested"
        finally:
            probe.release()
            start_foreign.set()
            worker.join(2)
            db.close()

    def test_separate_wal_read_connection_keeps_committed_read_concurrency(
        self, tmp_path
    ):
        db = SessionDB(db_path=tmp_path / "r12.db")
        owner_ready = threading.Event()
        release_owner = threading.Event()
        owner_done = threading.Event()
        reader_done = threading.Event()
        owner_failures: list[BaseException] = []
        reader_failures: list[BaseException] = []
        borrowed_is_distinct: list[bool] = []
        observed_values: list[str] = []

        class RollBackOwner(Exception):
            pass

        # Exercise the real WAL path even when the test interpreter is on a
        # SQLite patch that production correctly degrades to DELETE mode. This
        # database is disposable; prove WAL actually took effect before opening
        # the independent read connection.
        writer_conn = db._conn
        assert writer_conn is not None
        journal_mode = writer_conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        assert journal_mode.lower() == "wal"
        db._wal_active = True
        db._execute_write(
            lambda conn: conn.execute(
                "INSERT INTO state_meta(key, value) VALUES ('r12', 'committed')"
            )
        )

        def hold_uncommitted_writer() -> None:
            try:
                def temporary_update(conn):
                    conn.execute(
                        "UPDATE state_meta SET value = 'uncommitted' "
                        "WHERE key = 'r12'"
                    )
                    owner_ready.set()
                    assert release_owner.wait(5), "R12 reader never finished"
                    raise RollBackOwner()

                with pytest.raises(RollBackOwner):
                    db._execute_write(temporary_update)
            except BaseException as exc:
                owner_failures.append(exc)
            finally:
                owner_done.set()

        def read_committed_value() -> None:
            assert owner_ready.wait(2), "R12 owner transaction never opened"
            try:
                with db._read_ctx() as conn:
                    assert conn is not None
                    borrowed_is_distinct.append(conn is not db._conn)
                    observed_values.append(
                        conn.execute(
                            "SELECT value FROM state_meta WHERE key = 'r12'"
                        ).fetchone()[0]
                    )
            except BaseException as exc:
                reader_failures.append(exc)
            finally:
                reader_done.set()

        owner = threading.Thread(target=hold_uncommitted_writer, daemon=True)
        reader = threading.Thread(target=read_committed_value, daemon=True)
        owner.start()
        try:
            assert owner_ready.wait(2), "R12 owner transaction never opened"
            assert db._wal_active is True
            reader.start()
            read_finished_while_writer_open = reader_done.wait(2)
            release_owner.set()
            assert owner_done.wait(2), "R12 owner transaction did not roll back"
            owner.join(2)
            reader.join(2)
            assert read_finished_while_writer_open, (
                "separate WAL read waited for the writer transaction"
            )
            assert not owner.is_alive()
            assert not reader.is_alive()
            assert not owner_failures
            assert not reader_failures
            assert borrowed_is_distinct == [True]
            assert observed_values == ["committed"]
            assert db.get_meta("r12") == "committed"
        finally:
            release_owner.set()
            owner.join(2)
            if reader.ident is not None:
                reader.join(2)
            db.close()

    def test_connection_transaction_lock_order_is_single_and_stable(
        self, tmp_path, monkeypatch
    ):
        db = SessionDB(db_path=tmp_path / "r13.db")
        serial_lock = db._conn._hermes_serial_lock
        events: list[str] = []
        original_session_lock = db._lock
        original_serial_acquire = type(serial_lock).acquire
        original_serial_release = type(serial_lock).release

        class ObservedRLock:
            def acquire(self, *args, **kwargs):
                outermost = not original_session_lock._is_owned()
                acquired = original_session_lock.acquire(*args, **kwargs)
                if acquired and outermost:
                    events.append("session.acquire")
                return acquired

            def release(self):
                outermost = original_session_lock._recursion_count() == 1
                if outermost:
                    events.append("session.release")
                return original_session_lock.release()

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, *_exc_info):
                self.release()

        def observed_serial_acquire(lock, *args, **kwargs):
            outermost = lock is serial_lock and not lock._is_owned()
            acquired = original_serial_acquire(lock, *args, **kwargs)
            if acquired and outermost:
                events.append("connection.acquire")
            return acquired

        def observed_serial_release(lock):
            outermost = lock is serial_lock and lock._recursion_count() == 1
            if outermost:
                events.append("connection.release")
            return original_serial_release(lock)

        db._lock = ObservedRLock()
        monkeypatch.setattr(type(serial_lock), "acquire", observed_serial_acquire)
        monkeypatch.setattr(type(serial_lock), "release", observed_serial_release)

        def assert_root_order(call, *, expected_roots=1):
            events.clear()
            call()
            assert events.count("session.acquire") == expected_roots
            assert events.count("session.release") == expected_roots
            roots = 0
            session_owned = False
            connection_owned = False
            for event in events:
                if event == "session.acquire":
                    assert not connection_owned, events
                    session_owned = True
                elif event == "connection.acquire":
                    if session_owned:
                        roots += 1
                    connection_owned = True
                elif event == "connection.release":
                    connection_owned = False
                elif event == "session.release":
                    assert not connection_owned, events
                    session_owned = False
            assert roots == expected_roots, events
            assert not session_owned and not connection_owned

        try:
            def run_t1():
                with db.write_transaction() as conn:
                    conn.execute(
                        "INSERT INTO state_meta(key, value) VALUES ('r13-t1', 'ok')"
                    )

            def run_t2():
                db._execute_write(
                    lambda conn: conn.execute(
                        "INSERT INTO state_meta(key, value) VALUES ('r13-t2', 'ok')"
                    )
                )

            def run_t3():
                db._fts_enabled = True
                assert db._enter_fts_fail_open(
                    sqlite3.DatabaseError("database disk image is malformed")
                ) is True

            def run_e1():
                with db.offline_rebuild(reason="R13 lock order"):
                    db._execute_write(
                        lambda conn: conn.execute(
                            "INSERT INTO state_meta(key, value) "
                            "VALUES ('r13-e1-t2', 'ok')"
                        )
                    )

            assert_root_order(run_t1)
            assert_root_order(run_t2)
            t4_cursor = db._conn.cursor()
            try:
                assert_root_order(
                    lambda: db._recover_stale_fts(t4_cursor, legacy=False)
                )
            finally:
                t4_cursor.close()
            assert_root_order(run_t3)
            # One outer T2 preflight plus one E1 epoch. The nested E1→T2 call
            # re-enters both locks and therefore adds no second outer pair.
            assert_root_order(run_e1, expected_roots=2)
        finally:
            db._lock = original_session_lock
            db.close()

    def test_transaction_boundary_is_reentrant_for_owner_cursor_and_callback(
        self, tmp_path
    ):
        db = SessionDB(db_path=tmp_path / "r4.db")
        owned_during_callback: list[bool] = []

        try:
            def owner_write(conn):
                serial_owned = conn._hermes_serial_lock._is_owned()
                owned_during_callback.append(serial_owned)
                assert serial_owned, "transaction root did not retain the connection lock"
                assert conn.execute("SELECT 1").fetchone()[0] == 1
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT 2")
                    assert cursor.fetchone()[0] == 2
                finally:
                    cursor.close()
                assert db.get_meta("r4-missing") is None
                conn.execute(
                    "INSERT INTO state_meta(key, value) VALUES ('r4', 'owner')"
                )

            db._execute_write(owner_write)
            assert owned_during_callback == [True]
            assert db.get_meta("r4") == "owner"
        finally:
            db.close()

    def test_transaction_boundary_releases_both_locks_on_baseexception(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "r5.db")
        owned_during_callback: list[bool] = []
        worker_done = threading.Event()
        failures: list[BaseException] = []

        class FatalBoundaryExit(BaseException):
            pass

        try:
            def fail_owner(conn):
                owned_during_callback.append(conn._hermes_serial_lock._is_owned())
                conn.execute(
                    "INSERT INTO state_meta(key, value) VALUES ('r5-rolled-back', 'x')"
                )
                raise FatalBoundaryExit()

            with pytest.raises(FatalBoundaryExit):
                db._execute_write(fail_owner)

            def second_owner() -> None:
                try:
                    db._execute_write(
                        lambda conn: conn.execute(
                            "INSERT INTO state_meta(key, value) "
                            "VALUES ('r5-committed', 'ok')"
                        )
                    )
                except BaseException as exc:
                    failures.append(exc)
                finally:
                    worker_done.set()

            worker = threading.Thread(target=second_owner, daemon=True)
            worker.start()
            assert worker_done.wait(2), "both Python locks were stranded"
            worker.join(2)
            assert not worker.is_alive()
            assert not failures
            assert owned_during_callback == [True]
            assert db._conn.execute(
                "SELECT value FROM state_meta WHERE key = 'r5-rolled-back'"
            ).fetchone() is None
            assert db._conn.execute(
                "SELECT value FROM state_meta WHERE key = 'r5-committed'"
            ).fetchone()[0] == "ok"
        finally:
            db.close()


def test_udf_callback_stress_is_timeout_bounded_in_a_subprocess(tmp_path):
    """A hung UDF/GIL regression kills only its child, never the test runner."""
    root = Path(__file__).resolve().parents[2]
    program = textwrap.dedent(
        """
        import pathlib
        import sys
        import threading

        from hermes_state import _connect_tracked_db

        path = pathlib.Path(sys.argv[1]) / "stress.db"
        conn = _connect_tracked_db(path, check_same_thread=False)
        entered = threading.Event()
        proceed = threading.Event()
        failures = []

        def callback():
            entered.set()
            if not proceed.wait(4):
                raise RuntimeError("callback release timed out")
            # The same thread may re-enter through the RLock while SQLite is
            # executing the UDF; a second thread must wait without holding GIL.
            return conn.execute("SELECT 42").fetchone()[0]

        def first():
            try:
                assert conn.execute("SELECT callback()").fetchone()[0] == 42
            except BaseException as exc:
                failures.append(exc)

        def second():
            try:
                assert conn.execute("SELECT 7").fetchone()[0] == 7
            except BaseException as exc:
                failures.append(exc)

        conn.create_function("callback", 0, callback)
        one = threading.Thread(target=first, daemon=True)
        one.start()
        assert entered.wait(4)
        two = threading.Thread(target=second, daemon=True)
        two.start()
        proceed.set()
        one.join(4)
        two.join(4)
        assert not one.is_alive() and not two.is_alive()
        assert not failures, failures

        def stress():
            try:
                for _ in range(30):
                    cursor = conn.execute("SELECT hermes_turn_fence_generation()")
                    assert list(cursor) == [(1,)]
            except BaseException as exc:
                failures.append(exc)

        workers = [threading.Thread(target=stress, daemon=True) for _ in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(4)
        assert not any(worker.is_alive() for worker in workers)
        assert not failures, failures
        conn.close()
        print("SQLITE-SERIAL-STRESS-OK")
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-c", program, str(tmp_path)],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "serialized SQLite UDF stress child timed out and was killed: "
            f"stdout={exc.stdout!r} stderr={exc.stderr!r}"
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SQLITE-SERIAL-STRESS-OK" in result.stdout
