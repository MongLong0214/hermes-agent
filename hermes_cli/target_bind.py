"""``hermes target bind --json``: the local preflight an external controller runs to get a receipt.

The controller writes one 7-key JSON request to stdin. On success this prints the 8-key public receipt
and exits 0. On any failure it prints exactly one closed error object and exits 1. Stdout carries
nothing else, because the controller compares error replies byte for byte.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

_REQUEST_KEYS = frozenset(
    {
        "domain",
        "version",
        "session_id",
        "expected_lineage_root_digest",
        "actor_id",
        "binding_generation",
        "executor_runtime_identity",
    }
)
_DOMAIN = "hermes.target-bind"
_VERSION = 1
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
# The controller holds the generation as a JS number; a larger integer would reach the receipt digest
# with digits it cannot reproduce.
_MAX_GENERATION = 2**53 - 1
_PUBLIC_RECEIPT_KEYS = (
    "domain",
    "version",
    "actor_id",
    "binding_generation",
    "executor_runtime_identity",
    "requested_session_id",
    "lineage_root_digest",
    "receipt_digest",
)
_INVALID = "target_bind_preflight_invalid"
_CONFLICT = "target_bind_preflight_conflict"
_UNAVAILABLE = "target_bind_preflight_unavailable"
# The controller kills the command after 5 s, cold start included. Lock waits give up once this much
# has passed since stdout was reserved (early in startup); interpreter start precedes the reservation.
_REPLY_BUDGET_S = 1.5
# SQLite's own busy wait overruns its timeout (a 1 s timeout blocked for 2.5 s on 3.50.4, macOS), so
# bind's connections wait only briefly and leave the rest to the store's retry loop, which stops at its
# deadline.
_BUSY_TIMEOUT_S = 0.05

_reply_fd: int | None = None
_reply_deadline: float | None = None


def reserve_stdout_for_reply() -> None:
    """Keep the real stdout for the one reply and send every other stdout write to stderr.

    Covers Python prints, C-level writes to fd 1 and children that inherit it (installers, entry-point
    plugins printing at import). The kept fd is non-inheritable. Idempotent.
    """
    global _reply_fd, _reply_deadline
    if _reply_fd is not None:
        return
    sys.stdout.flush()
    try:
        reply_fd = os.dup(1)
    except OSError:
        return  # no stdout to protect
    try:
        os.dup2(2, 1)
    except OSError:
        os.close(reply_fd)
        return
    _reply_fd = reply_fd
    _reply_deadline = time.monotonic() + _REPLY_BUDGET_S
    sys.stdout = sys.stderr


def _write_json(payload: dict[str, Any]) -> None:
    # Bytes, not text: the reply must be UTF-8 whatever locale the controller's bare env implies.
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if _reply_fd is None:
        sys.stdout.buffer.write(data)
        sys.stdout.flush()
        return
    view = memoryview(data)
    while view:
        view = view[os.write(_reply_fd, view):]


def _closed_error(kind: str) -> int:
    _write_json({"error": kind})
    return 1


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _read_request() -> dict[str, Any]:
    raw = sys.stdin.buffer.read()
    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, TypeError, ValueError) as exc:
        raise ValueError("invalid JSON request") from exc
    if not isinstance(parsed, dict):
        raise ValueError("target bind request must be an object")
    return parsed


def _require_nonempty_text(request: dict[str, Any], key: str) -> str:
    value = request[key]
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f"{key} is invalid")
    return value


def _validated_request(request: dict[str, Any]) -> tuple[str, str, str, int, str]:
    if set(request) != _REQUEST_KEYS:
        raise ValueError("target bind request has an invalid schema")
    if (
        request["domain"] != _DOMAIN
        or type(request["version"]) is not int
        or request["version"] != _VERSION
    ):
        raise ValueError("target bind request has an invalid domain or version")
    session_id = _require_nonempty_text(request, "session_id")
    expected_digest = _require_nonempty_text(request, "expected_lineage_root_digest")
    actor_id = _require_nonempty_text(request, "actor_id")
    runtime_identity = _require_nonempty_text(request, "executor_runtime_identity")
    generation = request["binding_generation"]
    if type(generation) is not int or not 0 < generation <= _MAX_GENERATION:
        raise ValueError("binding_generation is invalid")
    if _DIGEST_RE.fullmatch(expected_digest) is None:
        raise ValueError("expected lineage root digest is invalid")
    return session_id, expected_digest, actor_id, generation, runtime_identity


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _is_settled_store(path: Path) -> bool:
    """True only for a store this build has already migrated, read without writing a byte. The writer
    open creates a missing file, quarantines a headerless one, and migrates an unsettled one (fence swap,
    lineage stamp, data migrations, FTS rebuild) irreversibly and with no bound; bind does none of that.
    A store it cannot read raises, which the caller answers as unavailable."""
    from hermes_cli.sqlite_safe_read import connect_tracked
    from hermes_state_common import FTS_STALE_KEY, FTS_STORAGE_VERSION
    from hermes_state_dbfile import has_invalid_sqlite_header_preopen
    from hermes_state_fence import LINEAGE_FENCED, STORED_SCHEMA_VERSION, decode_store_lineage, fences_exact
    from hermes_state_holders import read_only_db_uri

    if not path.is_file() or has_invalid_sqlite_header_preopen(path):
        return False
    uri = read_only_db_uri(path)
    # probe_store_lineage's rule: a mode=ro open of a WAL store with no sidecars leaves -wal/-shm behind,
    # so with none present read the main file alone.
    if not Path(f"{path}-wal").exists() and not Path(f"{path}-journal").exists():
        uri += "&immutable=1"
    # Tracked, like every connection to state.db: an untracked close would cancel this process's POSIX locks.
    conn = connect_tracked(uri, tracking_path=path, uri=True, timeout=_BUSY_TIMEOUT_S)
    try:
        cursor = conn.cursor()
        # apply_fence_delta's settled test, plus the stamp that advance_lineage_stamp writes only once the
        # open's migrations have completed.
        lineage = decode_store_lineage(cursor)
        if lineage.lineage != LINEAGE_FENCED or lineage.stored != STORED_SCHEMA_VERSION or not fences_exact(cursor):
            return False
        meta = dict(
            cursor.execute(
                "SELECT key, value FROM state_meta WHERE key IN (?, 'fts_storage_version')", (FTS_STALE_KEY,)
            ).fetchall()
        )
    finally:
        conn.close()
    # A stale breadcrumb makes the open rebuild the index. The storage version is absent until a store
    # this build created is opened a second time, and that open only stamps it.
    return FTS_STALE_KEY not in meta and meta.get("fts_storage_version", str(FTS_STORAGE_VERSION)) == str(
        FTS_STORAGE_VERSION
    )


def _open_store(path: Path, patience_s: float):
    from hermes_state import SessionDB

    class _ReplyBoundSessionDB(SessionDB):
        # The open waits out a foreign write lock with the store's patience (20 s); bind has less.
        _WRITE_PATIENCE_S = patience_s

        def _open_writer_conn(self):
            conn = super()._open_writer_conn()
            conn.execute(f"PRAGMA busy_timeout={int(_BUSY_TIMEOUT_S * 1000)}")
            return conn

    return _ReplyBoundSessionDB(path)


def cmd_target_bind(args: Any) -> int:
    """Run the stdin-only preflight without exposing state implementation details."""
    if not getattr(args, "json", False):
        return _closed_error(_INVALID)
    try:
        session_id, expected_digest, actor_id, generation, runtime_identity = _validated_request(
            _read_request()
        )
    except Exception:
        return _closed_error(_INVALID)

    deadline = _reply_deadline if _reply_deadline is not None else time.monotonic() + _REPLY_BUDGET_S
    try:
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "state.db"
        if not _is_settled_store(path):
            return _closed_error(_UNAVAILABLE)
        db = _open_store(path, _remaining(deadline))
    except Exception:
        return _closed_error(_UNAVAILABLE)

    from hermes_state_target_bind import (
        TargetBindReceiptConflictError,
        TargetBindReceiptFenceError,
    )

    try:
        with db:
            receipt = db.prepare_target_bind_receipt(
                session_id,
                actor_id,
                generation,
                runtime_identity,
                expected_lineage_root_digest=expected_digest,
                patience_s=_remaining(deadline),
            )
    except TargetBindReceiptConflictError:
        return _closed_error(_CONFLICT)
    except (TargetBindReceiptFenceError, ValueError):
        return _closed_error(_INVALID)
    except Exception:
        return _closed_error(_UNAVAILABLE)

    _write_json({key: receipt[key] for key in _PUBLIC_RECEIPT_KEYS})
    return 0
