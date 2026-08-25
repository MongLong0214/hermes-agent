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
    UntrackableConnectionError,
    connect_tracked,
    has_live_connection,
)

SQLiteSerializationError = getattr(
    hermes_state, "SQLiteSerializationError", RuntimeError
)


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
            calls.append(
                (
                    "autocommit set",
                    lambda: setattr(conn, "autocommit", conn.autocommit),
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
    import hermes_state

    path = tmp_path / "state.db"
    conn = _connect_tracked_db(path, check_same_thread=False)
    serial_close_entered = threading.Event()
    close_finished = threading.Event()
    query_finished = threading.Event()
    errors: list[BaseException] = []
    original_close = hermes_state._SerializedConnectionMixin.close

    def observed_close(self):
        # Tracking.close() reaches this only after its own implementation has
        # made its lock-order decision.
        serial_close_entered.set()
        return original_close(self)

    monkeypatch.setattr(hermes_state._SerializedConnectionMixin, "close", observed_close)

    def close_in_worker() -> None:
        try:
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
    """Reverse backups acquire the same two locks in the same order."""
    class BackupConnection(sqlite3.Connection):
        def backup(self, *args, **kwargs):
            return "native-backup-result"

    source = _connect_tracked_db(
        tmp_path / "source.db", factory=BackupConnection, check_same_thread=False
    )
    target = _connect_tracked_db(
        tmp_path / "target.db", factory=BackupConnection, check_same_thread=False
    )

    class RecordingLock:
        def __init__(self, name, calls):
            self.name = name
            self.calls = calls
            self.lock = threading.RLock()

        def __enter__(self):
            self.calls.append(self.name)
            self.lock.acquire()
            return self

        def __exit__(self, *exc_info):
            self.lock.release()

    try:
        calls: list[str] = []
        source._hermes_serial_lock = RecordingLock("source", calls)
        target._hermes_serial_lock = RecordingLock("target", calls)

        assert source.backup(target) == "native-backup-result"
        source_to_target = calls[:]
        calls.clear()
        assert target.backup(source) == "native-backup-result"
        target_to_source = calls[:]
        assert source_to_target == target_to_source
        assert set(source_to_target) == {"source", "target"}
        calls.clear()
        assert source.backup(source) == "native-backup-result"
        assert calls == ["source"]

        unsafe = sqlite3.connect(tmp_path / "unsafe-backup.db")
        try:
            with pytest.raises(SQLiteSerializationError, match="destination"):
                source.backup(unsafe)
        finally:
            unsafe.close()
    finally:
        source.close()
        target.close()


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
