"""Per-connection serialization for state.db connections that run Python UDFs.

A fenced write runs ``hermes_turn_fence_generation()`` inside ``sqlite3_step``:
thread A holds the connection mutex with the GIL released and waits for the
GIL to run the callback, while thread B holds the GIL and waits for that mutex
(bind, cursor description and fetch paths keep the GIL while taking it). The
process then freezes; the fork gateway sat in exactly this ABBA deadlock for
80+ minutes, and the same unsynchronized sharing also segfaults under load.

At most one thread enters SQLite per connection, enforced at the connection
factory rather than at 70+ call sites. Waiting on this RLock releases the GIL,
so the callback thread can always finish and the cycle cannot form.
``interrupt()`` is deliberately not wrapped: it exists to cancel another
thread's in-flight statement, and sqlite3_interrupt is safe without the mutex.
"""

from __future__ import annotations

import sqlite3
import threading


class _SerializedCursor(sqlite3.Cursor):
    """Cursor whose SQLite entries run under the connection's RLock. Fetches are
    wrapped too: they step the VM while holding the GIL, which is the
    GIL-held mutex-wait leg of the deadlock."""

    def _serial(self):
        return self.connection._hermes_serial_lock

    def execute(self, *args, **kwargs):
        with self._serial():
            return super().execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        with self._serial():
            return super().executemany(*args, **kwargs)

    def executescript(self, *args, **kwargs):
        with self._serial():
            return super().executescript(*args, **kwargs)

    def fetchone(self):
        with self._serial():
            return super().fetchone()

    def fetchmany(self, *args, **kwargs):
        with self._serial():
            return super().fetchmany(*args, **kwargs)

    def fetchall(self):
        with self._serial():
            return super().fetchall()

    def __next__(self):
        with self._serial():
            return super().__next__()

    def close(self):
        with self._serial():
            return super().close()


class SerializedConnectionMixin:
    """Serialize every SQLite entry on one connection behind an RLock."""

    def __init__(self, *args, **kwargs):
        self._hermes_serial_lock = threading.RLock()
        super().__init__(*args, **kwargs)

    def cursor(self, factory=None):
        with self._hermes_serial_lock:
            return super().cursor(factory or _SerializedCursor)

    # Routed through self.cursor() instead of the C shortcuts: on Python 3.11
    # those build a plain Cursor without calling the overridden cursor(), so
    # ``conn.execute(...).fetchall()`` would fetch unserialized.
    def execute(self, *args, **kwargs):
        with self._hermes_serial_lock:
            return self.cursor().execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        with self._hermes_serial_lock:
            return self.cursor().executemany(*args, **kwargs)

    def executescript(self, *args, **kwargs):
        with self._hermes_serial_lock:
            return self.cursor().executescript(*args, **kwargs)

    def commit(self):
        with self._hermes_serial_lock:
            return super().commit()

    def rollback(self):
        with self._hermes_serial_lock:
            return super().rollback()

    def close(self):
        with self._hermes_serial_lock:
            return super().close()

    def __exit__(self, *exc_info):
        with self._hermes_serial_lock:
            return super().__exit__(*exc_info)

    def create_function(self, *args, **kwargs):
        with self._hermes_serial_lock:
            return super().create_function(*args, **kwargs)

    def create_collation(self, *args, **kwargs):
        with self._hermes_serial_lock:
            return super().create_collation(*args, **kwargs)

    def create_aggregate(self, *args, **kwargs):
        with self._hermes_serial_lock:
            return super().create_aggregate(*args, **kwargs)

    def backup(self, *args, **kwargs):
        with self._hermes_serial_lock:
            return super().backup(*args, **kwargs)


class SerializedConnection(SerializedConnectionMixin, sqlite3.Connection):
    pass


_serialized_factory_cache: dict = {}


def serialized_connection_factory(factory: type = sqlite3.Connection) -> type:
    """Mix serialization into *factory* (mirrors ``sqlite_safe_read._tracking_factory``)."""
    if factory is sqlite3.Connection:
        return SerializedConnection
    if issubclass(factory, SerializedConnectionMixin):
        return factory
    cached = _serialized_factory_cache.get(factory)
    if cached is None:
        cached = type(f"Serialized{factory.__name__}", (SerializedConnectionMixin, factory), {})
        _serialized_factory_cache[factory] = cached
    return cached
