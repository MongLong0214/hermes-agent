"""Standalone v27-compatible canonical SQLite writer fixture."""

import os
import sqlite3
import sys
from pathlib import Path


_WINDOWS_LOCK_OFFSET = 1024 * 1024
_SENTINEL_SESSION_KEY = "legacy-writer"
_SENTINEL_ENTRY_JSON = "{}"


def _try_acquire_gateway_lock(handle) -> bool:
    if sys.platform == "win32":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write("\n")
            handle.flush()
        handle.seek(_WINDOWS_LOCK_OFFSET)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _release_gateway_lock(handle) -> None:
    if sys.platform == "win32":
        import msvcrt

        handle.seek(_WINDOWS_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> int:
    home = Path(os.environ["HERMES_HOME"])
    state_db_path = Path(os.environ["STATE_DB_PATH"])
    legacy_generation = int(os.environ["LEGACY_SCHEMA_GENERATION"])
    lock_handle = None
    connection = None
    lock_acquired = False
    try:
        home.mkdir(parents=True, exist_ok=True)
        lock_handle = (home / "gateway.lock").open("a+", encoding="utf-8")
        lock_acquired = _try_acquire_gateway_lock(lock_handle)
        if not lock_acquired:
            print("REFUSED", flush=True)
            return 0

        connection = sqlite3.connect(str(state_db_path))
        connection.create_function(
            "hermes_turn_fence_generation", 0, lambda: legacy_generation
        )
        connection.execute(
            "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) "
            "VALUES ('', ?, ?, 1)",
            (_SENTINEL_SESSION_KEY, _SENTINEL_ENTRY_JSON),
        )
        connection.commit()
        readback = connection.execute(
            "SELECT entry_json FROM gateway_routing WHERE session_key = ?",
            (_SENTINEL_SESSION_KEY,),
        ).fetchone()
        if readback != (_SENTINEL_ENTRY_JSON,):
            raise RuntimeError("legacy gateway routing readback failed")
        print("READY:READBACK", flush=True)
        sys.stdin.read()
        return 0
    finally:
        if connection is not None:
            connection.close()
        if lock_handle is not None:
            try:
                if lock_acquired:
                    _release_gateway_lock(lock_handle)
            finally:
                lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
