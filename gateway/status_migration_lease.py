"""Migration leases on a Hermes home's gateway runtime lock (``gateway.lock``).

A state.db migration that would lock another build out of its own store runs only under a lease:
a private nonblocking OS lock on the same file, held for the migration, so a gateway started
meanwhile fails its own claim and the kernel drops the lease with the process. The lease never
rewrites or unlinks the file, which carries the owning gateway's identity record.

The OS lock belongs to one open file description, so a second handle in this process would fail
against a lock this process already holds. The gateway that owns the lock therefore borrows it
(releasing the borrow leaves the gateway's lock alone), and concurrent migrations in one process
share one private lock by reference count. Every check-and-take runs under
``gateway.status._gateway_lock_guard``, the guard of the gateway's own handle.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class _HeldLock:
    handle: Any
    refs: int = 1


# This process's private leases; read and changed only under gateway.status._gateway_lock_guard.
_held_locks: list[_HeldLock] = []


@dataclass
class GatewayRuntimeMigrationLease:
    lock_path: Path
    _held: Optional[_HeldLock] = None  # None: borrowed from this process's own gateway lock


def gateway_runtime_lock_path_for_home(home) -> Path:
    from gateway import status

    return Path(home) / status._GATEWAY_LOCK_FILENAME


def _is_lock_file(handle, lock_path: Path) -> bool:
    # File identity, not path text: a home named through a symlink is the same lock.
    try:
        return os.path.samestat(os.fstat(handle.fileno()), os.stat(lock_path))
    except (OSError, ValueError):
        return False


def _open_lock_file(lock_path: Path):
    try:
        return open(lock_path, "a+", encoding="utf-8")
    except PermissionError:
        # A lock file another user left (a root launchd session): the OS lock works on a read handle.
        with contextlib.suppress(OSError):
            return open(lock_path, "r", encoding="utf-8")
    except OSError:
        pass
    return None


def acquire_gateway_runtime_migration_lease(lock_path) -> Optional[GatewayRuntimeMigrationLease]:
    """A lease on *lock_path*, or None while another process holds that lock (or it cannot be opened)."""
    from gateway import status

    lock_path = Path(lock_path)
    with status._gateway_lock_guard:
        owned = status._gateway_lock_handle
        if owned is not None and _is_lock_file(owned, lock_path):
            return GatewayRuntimeMigrationLease(lock_path)
        shared = next((held for held in _held_locks if _is_lock_file(held.handle, lock_path)), None)
        if shared is not None:
            shared.refs += 1
            return GatewayRuntimeMigrationLease(lock_path, shared)
        # Twice: an unheld lock file is unlinked by every unscoped get_running_pid(), so one can go
        # between the open and the lock, and a lock on it would exclude no gateway.
        for _attempt in range(2):
            handle = _open_lock_file(lock_path)
            if handle is None:
                return None
            if not status._try_acquire_file_lock(handle):
                with contextlib.suppress(OSError):
                    handle.close()
                return None
            if _is_lock_file(handle, lock_path):
                held = _HeldLock(handle)
                _held_locks.append(held)
                return GatewayRuntimeMigrationLease(lock_path, held)
            status._release_file_lock(handle)
            with contextlib.suppress(OSError):
                handle.close()
        return None


def lease_holds_lock_file(lease: GatewayRuntimeMigrationLease) -> bool:
    """True while a held *lease*'s OS lock is on the file now at its lock path.

    The lease writes no gateway record, so an unscoped ``get_running_pid()`` of any build reads
    its held lock as stale and unlinks the file; the next gateway then claims a new file that
    this lease does not exclude."""
    from gateway import status

    with status._gateway_lock_guard:
        handle = status._gateway_lock_handle if lease._held is None else lease._held.handle
        return handle is not None and _is_lock_file(handle, lease.lock_path)


def release_gateway_runtime_migration_lease(lease: Optional[GatewayRuntimeMigrationLease]) -> None:
    """Drop a lease (idempotent); a borrowed one leaves the gateway's lock where it is."""
    if lease is None:
        return
    from gateway import status

    with status._gateway_lock_guard:
        held, lease._held = lease._held, None
        if held is None:
            return
        held.refs -= 1
        if held.refs:
            return
        _held_locks.remove(held)
        status._release_file_lock(held.handle)
        with contextlib.suppress(OSError):
            held.handle.close()
