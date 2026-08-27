"""Behavioral regressions for the canonical SQLite serialization boundary."""

from __future__ import annotations

import threading

import pytest


def test_tracked_connection_serializes_a_real_cursor_operation(tmp_path):
    """A second caller waits at the physical connection's serialization gate."""
    from hermes_state import _connect_tracked_db

    conn = _connect_tracked_db(
        tmp_path / "state.db", check_same_thread=False, isolation_level=None
    )
    try:
        conn.execute("CREATE TABLE entries(value INTEGER)")
        serial_lock = conn._hermes_serial_lock
        serial_lock.acquire()
        attempted = threading.Event()
        finished = threading.Event()
        result: list[object] = []
        errors: list[BaseException] = []

        def read_through_the_real_cursor() -> None:
            attempted.set()
            try:
                result.append(conn.execute("SELECT COUNT(*) FROM entries").fetchone())
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        worker = threading.Thread(target=read_through_the_real_cursor)
        worker.start()
        assert attempted.wait(2)
        assert not finished.wait(0.1), "cursor operation bypassed the serial gate"
        serial_lock.release()
        assert finished.wait(2)
        worker.join(2)
        assert not worker.is_alive()
        assert errors == []
        assert result == [(0,)]
    finally:
        conn.close()


def test_kanban_connect_fn_returns_serialized_tracked_owner(tmp_path):
    """The immutable Kanban ``connect_fn=sqlite3.connect`` route composes both owners."""
    from hermes_cli.kanban_db import _sqlite_connect
    from hermes_cli.sqlite_safe_read import _TrackingMixin, has_live_connection
    from hermes_state import _SerializedConnectionMixin, _serial_lock_for_connection

    db_path = tmp_path / "kanban.db"
    conn = _sqlite_connect(db_path)
    try:
        assert isinstance(conn, _TrackingMixin)
        assert isinstance(conn, _SerializedConnectionMixin)
        assert _serial_lock_for_connection(conn).owner is conn
        assert has_live_connection(db_path)
        conn.execute("CREATE TABLE proof(value INTEGER)")
        assert conn.execute("SELECT COUNT(*) FROM proof").fetchone() == (0,)
    finally:
        conn.close()
    assert not has_live_connection(db_path)


def test_cursor_row_factory_get_waits_for_physical_owner_lock(tmp_path):
    """Cursor mutable state reads use the connection's physical owner lock."""
    from hermes_state import _connect_tracked_db

    conn = _connect_tracked_db(tmp_path / "cursor-row-factory.db", check_same_thread=False)
    cursor = conn.cursor()
    serial_lock = conn._hermes_serial_lock
    attempted = threading.Event()
    finished = threading.Event()
    observed: list[object] = []
    errors: list[BaseException] = []

    def read_row_factory() -> None:
        attempted.set()
        try:
            observed.append(cursor.row_factory)
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    serial_lock.acquire()
    worker = threading.Thread(target=read_row_factory)
    worker.start()
    try:
        assert attempted.wait(2)
        assert not finished.wait(0.2), "Cursor.row_factory getter bypassed the owner lock"
    finally:
        serial_lock.release()
    assert finished.wait(2)
    worker.join(2)
    assert not worker.is_alive()
    assert errors == []
    assert observed == [None]
    conn.close()


def test_cursor_row_factory_set_waits_for_physical_owner_lock(tmp_path):
    """Cursor mutable state writes use the connection's physical owner lock."""
    from hermes_state import _connect_tracked_db

    conn = _connect_tracked_db(tmp_path / "cursor-row-factory-set.db", check_same_thread=False)
    cursor = conn.cursor()
    serial_lock = conn._hermes_serial_lock
    attempted = threading.Event()
    finished = threading.Event()
    errors: list[BaseException] = []

    def set_row_factory() -> None:
        attempted.set()
        try:
            cursor.row_factory = lambda _cur, row: tuple(row)
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    serial_lock.acquire()
    worker = threading.Thread(target=set_row_factory)
    worker.start()
    try:
        assert attempted.wait(2)
        assert not finished.wait(0.2), "Cursor.row_factory setter bypassed the owner lock"
    finally:
        serial_lock.release()
    assert finished.wait(2)
    worker.join(2)
    assert not worker.is_alive()
    assert errors == []
    assert cursor.row_factory is not None
    conn.close()


def test_close_waits_for_callback_held_connection_lock(tmp_path, monkeypatch):
    """Close reaches the serial gate before a live SQLite callback can release it."""
    from hermes_state import _connect_tracked_db

    conn = _connect_tracked_db(tmp_path / "callback-close.db", check_same_thread=False)
    serial_lock = conn._hermes_serial_lock
    callback_entered = threading.Event()
    close_waiting = threading.Event()
    callback_observed = threading.Event()
    release_callback = threading.Event()
    query_done = threading.Event()
    close_done = threading.Event()
    errors: list[BaseException] = []
    close_thread: list[threading.Thread] = []
    original_acquire = type(serial_lock).acquire

    def observing_acquire(lock, *args, **kwargs):
        if lock is serial_lock and close_thread and threading.current_thread() is close_thread[0]:
            close_waiting.set()
        return original_acquire(lock, *args, **kwargs)

    monkeypatch.setattr(type(serial_lock), "acquire", observing_acquire)

    def callback():
        callback_entered.set()
        try:
            assert close_waiting.wait(2), "close bypassed the connection serial gate"
            assert not close_done.is_set(), "close completed while the callback held SQLite"
            callback_observed.set()
            assert release_callback.wait(2), "callback release was not signalled"
            return 1
        finally:
            callback_observed.set()

    conn.create_function("hold_callback", 0, callback)

    def query() -> None:
        try:
            assert conn.execute("SELECT hold_callback()").fetchone() == (1,)
        except BaseException as exc:
            errors.append(exc)
        finally:
            query_done.set()

    def close() -> None:
        try:
            conn.close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            close_done.set()

    query_thread = threading.Thread(target=query)
    query_thread.start()
    assert callback_entered.wait(2)
    worker = threading.Thread(target=close)
    close_thread.append(worker)
    worker.start()
    try:
        assert callback_observed.wait(2)
        release_callback.set()
        assert query_done.wait(2)
        assert close_done.wait(2)
    finally:
        release_callback.set()
        query_thread.join(2)
        worker.join(2)
    assert not query_thread.is_alive()
    assert not worker.is_alive()
    assert errors == []


def test_callback_close_failure_waits_for_serial_owner_outside_registry_lock(tmp_path, monkeypatch):
    """A callback holds only the serial owner while a failing close still retires natively."""
    import sqlite3

    import hermes_cli.sqlite_safe_read as safe_read
    from hermes_state import _connect_tracked_db

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

    class RaisingClose(sqlite3.Connection):
        def close(self):
            assert not live_lock.held_by_current_thread(), "custom close held _live_lock"
            close_hook_entered.set()
            raise RuntimeError("custom close failed")

    live_lock = GuardedLiveLock()
    monkeypatch.setattr(safe_read, "_live_lock", live_lock)
    conn = _connect_tracked_db(
        tmp_path / "callback-close-failure.db",
        factory=RaisingClose,
        check_same_thread=False,
    )
    serial_lock = getattr(conn, "_hermes_serial_lock")
    callback_entered = threading.Event()
    close_waiting = threading.Event()
    release_callback = threading.Event()
    callback_done = threading.Event()
    close_hook_entered = threading.Event()
    query_done = threading.Event()
    close_done = threading.Event()
    errors: list[BaseException] = []
    close_threads: list[threading.Thread] = []
    original_acquire = type(serial_lock).acquire

    def observing_acquire(lock, *args, **kwargs):
        if lock is serial_lock and close_threads and threading.current_thread() is close_threads[0]:
            close_waiting.set()
        return original_acquire(lock, *args, **kwargs)

    monkeypatch.setattr(type(serial_lock), "acquire", observing_acquire)

    def callback():
        assert not live_lock.held_by_current_thread(), "SQLite callback held _live_lock"
        callback_entered.set()
        try:
            assert close_waiting.wait(2), "close bypassed the connection serial gate"
            assert not close_hook_entered.is_set(), "close hook ran while callback owned SQLite"
            callback_done.set()
            assert release_callback.wait(2), "callback release was not signalled"
            return 1
        finally:
            callback_done.set()

    conn.create_function("hold_callback", 0, callback)

    def query() -> None:
        try:
            assert conn.execute("SELECT hold_callback()").fetchone() == (1,)
        except BaseException as exc:
            errors.append(exc)
        finally:
            query_done.set()

    def close() -> None:
        try:
            conn.close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            close_done.set()

    query_thread = threading.Thread(target=query)
    query_thread.start()
    assert callback_entered.wait(2), "statement did not reach the callback"
    close_thread = threading.Thread(target=close)
    close_threads.append(close_thread)
    close_thread.start()
    try:
        assert callback_done.wait(2), "callback did not observe the close waiter"
        release_callback.set()
        assert query_done.wait(2), "query did not finish"
        assert close_done.wait(2), "close did not finish"
    finally:
        release_callback.set()
        query_thread.join(2)
        close_thread.join(2)

    assert not query_thread.is_alive()
    assert not close_thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "custom close failed"
    assert close_hook_entered.is_set()
    assert not safe_read.has_live_connection(tmp_path / "callback-close-failure.db")
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        sqlite3.Connection.execute(conn, "SELECT 1")


def test_backup_orders_serialized_pairs_and_releases_every_failure_path(tmp_path, monkeypatch):
    """Real backup orders locks consistently and leaves every connection usable."""
    import sqlite3

    from hermes_state import (
        SQLiteSerializationError,
        _connect_tracked_db,
        _serial_lock_for_connection,
    )

    def open_connection(name, **kwargs):
        return _connect_tracked_db(
            tmp_path / name,
            check_same_thread=False,
            isolation_level=None,
            **kwargs,
        )

    first = open_connection("first.db")
    second = open_connection("second.db")
    raw = sqlite3.connect(tmp_path / "raw.db", isolation_level=None)
    failure_source = None
    failure_target = None
    try:
        first_lock = _serial_lock_for_connection(first)
        second_lock = _serial_lock_for_connection(second)
        assert first_lock.rank < second_lock.rank
        acquired: list[int] = []
        original_acquire = type(first_lock).acquire

        def observing_acquire(lock, *args, **kwargs):
            if lock in {first_lock, second_lock}:
                acquired.append(lock.rank)
            return original_acquire(lock, *args, **kwargs)

        monkeypatch.setattr(type(first_lock), "acquire", observing_acquire)

        first.execute("CREATE TABLE forward(value INTEGER)")
        first.execute("INSERT INTO forward(value) VALUES (7)")
        acquired.clear()
        first.backup(second)
        assert acquired == [first_lock.rank, second_lock.rank]
        assert second.execute("SELECT value FROM forward").fetchall() == [(7,)]

        second.execute("CREATE TABLE reverse(value INTEGER)")
        second.execute("INSERT INTO reverse(value) VALUES (9)")
        acquired.clear()
        second.backup(first)
        assert acquired == [first_lock.rank, second_lock.rank]
        assert first.execute("SELECT value FROM reverse").fetchall() == [(9,)]

        def assert_released_and_usable(*connections):
            for connection in connections:
                assert not _serial_lock_for_connection(connection).is_owned_by_current_thread()
                assert connection.execute("SELECT 1").fetchone() == (1,)

        with pytest.raises(ValueError, match="same connection"):
            first.backup(first)
        assert_released_and_usable(first)

        with pytest.raises(SQLiteSerializationError, match="no matching Hermes serialization owner"):
            first.backup(raw)
        assert_released_and_usable(first)
        assert raw.execute("SELECT 1").fetchone() == (1,)

        with pytest.raises(SQLiteSerializationError, match="not a serialized SQLite connection"):
            getattr(first, "backup")(object())
        assert_released_and_usable(first)

        with second_lock:
            with pytest.raises(SQLiteSerializationError, match="invert connection lock order"):
                second.backup(first)
            assert not first_lock.is_owned_by_current_thread()
        assert_released_and_usable(first, second)

        native_calls: list[sqlite3.Connection] = []

        class NativeFailingBackup(sqlite3.Connection):
            def backup(self, target, *args, **kwargs):
                native_calls.append(self)
                assert _serial_lock_for_connection(self).is_owned_by_current_thread()
                assert _serial_lock_for_connection(target).is_owned_by_current_thread()
                raise sqlite3.OperationalError("native backup failed")

        failure_source = open_connection("failure-source.db", factory=NativeFailingBackup)
        failure_target = open_connection("failure-target.db")
        with pytest.raises(sqlite3.OperationalError, match="native backup failed"):
            failure_source.backup(failure_target)
        assert native_calls == [failure_source]
        assert_released_and_usable(failure_source, failure_target)
    finally:
        for connection in (failure_target, failure_source, raw, second, first):
            if connection is not None:
                connection.close()


def test_interrupt_cancels_an_inflight_serialized_statement_without_waiting(tmp_path):
    """Native cancellation reaches SQLite before the statement owner is released."""
    import sqlite3

    from hermes_state import _connect_tracked_db

    conn = _connect_tracked_db(tmp_path / "interrupt.db", check_same_thread=False)
    entered = threading.Event()
    let_query_continue = threading.Event()
    query_done = threading.Event()
    cancel_done = threading.Event()
    errors: list[BaseException] = []
    cancel_errors: list[BaseException] = []

    def wait_in_sqlite_callback():
        entered.set()
        assert let_query_continue.wait(5), "test barrier was not released"
        return 1

    conn.create_function("wait_in_sqlite_callback", 0, wait_in_sqlite_callback)

    def run_query():
        try:
            conn.execute("SELECT wait_in_sqlite_callback()").fetchone()
        except Exception as exc:  # sqlite raises OperationalError after interrupt
            errors.append(exc)
        finally:
            query_done.set()

    query_thread = threading.Thread(target=run_query)
    query_thread.start()
    assert entered.wait(5), "the statement never reached its SQLite callback"

    def cancel() -> None:
        try:
            conn.interrupt()
        except BaseException as exc:
            cancel_errors.append(exc)
        finally:
            cancel_done.set()

    cancel_thread = threading.Thread(target=cancel)
    cancel_thread.start()
    assert cancel_done.wait(5), "interrupt waited on the in-flight serial lock"
    assert cancel_errors == []
    assert not query_done.is_set(), "statement owner released before the callback barrier"
    let_query_continue.set()
    assert query_done.wait(5), "interrupted statement did not finish"
    query_thread.join(2)
    cancel_thread.join(2)
    try:
        assert not query_thread.is_alive()
        assert not cancel_thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], sqlite3.OperationalError)
        assert str(errors[0]) == "interrupted"
        assert conn.execute("SELECT 1").fetchone() == (1,)
    finally:
        conn.close()


def test_supported_sqlite_surface_has_one_serialization_mapping(tmp_path):
    """The public runtime surface is either gated once or is ``interrupt``."""
    import sqlite3

    from hermes_state import _CONNECTION_SERIALIZATION_SURFACE, _connect_tracked_db

    conn = _connect_tracked_db(tmp_path / "surface.db", isolation_level=None)
    try:
        supported = {
            name
            for name in (
                "cursor", "execute", "executemany", "executescript", "commit",
                "rollback", "close", "blobopen", "iterdump", "backup", "serialize",
                "deserialize", "create_function", "create_aggregate", "create_collation",
                "create_window_function", "set_authorizer", "set_progress_handler",
                "set_trace_callback", "enable_load_extension", "load_extension", "getlimit",
                "setlimit", "getconfig", "setconfig", "in_transaction", "row_factory",
                "text_factory", "total_changes", "isolation_level", "autocommit", "interrupt",
            )
            if name in {"interrupt", "row_factory", "text_factory"} or hasattr(sqlite3.Connection, name)
        }
        assert set(_CONNECTION_SERIALIZATION_SURFACE) == supported
        assert _CONNECTION_SERIALIZATION_SURFACE["interrupt"] == "nonblocking"
        assert all(
            _CONNECTION_SERIALIZATION_SURFACE[name] == "locked"
            for name in supported - {"interrupt"}
        )

        conn.execute("CREATE TABLE data(id INTEGER PRIMARY KEY, payload BLOB)")
        conn.execute("INSERT INTO data(payload) VALUES (zeroblob(4))")
        cursor = conn.cursor()
        assert cursor.execute("SELECT payload FROM data").fetchone() == (b"\x00" * 4,)
        blob = conn.blobopen("data", "payload", 1)
        try:
            blob.write(b"lock")
            blob.seek(0)
            assert blob.read() == b"lock"
            assert len(blob) == 4
        finally:
            blob.close()
        assert next(conn.iterdump()).startswith("BEGIN TRANSACTION")
    finally:
        conn.close()
