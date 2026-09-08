"""Local JSON preflight for target-bind receipt creation."""

from __future__ import annotations

import json
import re
import sys
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


def _write_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.flush()


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
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
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
    if type(generation) is not int or generation <= 0:
        raise ValueError("binding_generation is invalid")
    if _DIGEST_RE.fullmatch(expected_digest) is None:
        raise ValueError("expected lineage root digest is invalid")
    return session_id, expected_digest, actor_id, generation, runtime_identity


def cmd_target_bind(args: Any) -> int:
    """Run the stdin-only preflight without exposing state implementation details."""
    if not getattr(args, "json", False):
        return _closed_error("target_bind_preflight_invalid")
    try:
        request = _read_request()
        (
            session_id,
            expected_digest,
            actor_id,
            binding_generation,
            runtime_identity,
        ) = _validated_request(request)
    except Exception:
        return _closed_error("target_bind_preflight_invalid")

    try:
        from hermes_constants import get_hermes_home
        from hermes_state import (
            SessionDB,
            TargetBindReceiptConflictError,
            TargetBindReceiptFenceError,
        )

        with SessionDB(get_hermes_home() / "state.db") as db:
            receipt = db.prepare_target_bind_receipt(
                session_id,
                actor_id,
                binding_generation,
                runtime_identity,
                expected_lineage_root_digest=expected_digest,
            )
    except TargetBindReceiptConflictError:
        return _closed_error("target_bind_preflight_conflict")
    except (TargetBindReceiptFenceError, ValueError):
        return _closed_error("target_bind_preflight_invalid")
    except Exception:
        return _closed_error("target_bind_preflight_unavailable")

    _write_json({key: receipt[key] for key in _PUBLIC_RECEIPT_KEYS})
    return 0
