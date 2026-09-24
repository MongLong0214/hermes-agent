"""Target-bind receipt mixin for SessionDB.

An external controller binds one caller identity (actor, binding generation, executor runtime) to one
session lineage and gets back an immutable receipt it can verify on its own. The receipt commits to the
lineage root through a domain-separated digest, so a verifier can check the binding without learning
the private root id.

Receipts live in ``state_meta`` under ``target_bind_receipt:<hex>``, where ``<hex>`` is the digest of
the binding identity. There is no schema change and no version stamp, because the family reads neither.
Writes go through ``SessionDB._execute_write``, so the structural-corruption quarantine refuses them the
same way it refuses every other write.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Optional, Tuple


class TargetBindReceiptFenceError(RuntimeError):
    """A target bind receipt cannot be safely resolved from durable state."""


class TargetBindReceiptConflictError(RuntimeError):
    """An existing target bind identity conflicts with immutable evidence."""


_TARGET_BIND_RECEIPT_META_PREFIX = "target_bind_receipt:"
_TARGET_BIND_RECEIPT_SCHEMA = "hermes.target-bind-receipt"
_TARGET_BIND_RECEIPT_DOMAIN = "hermes.target-bind"
_TARGET_BIND_RECEIPT_VERSION = 1
_TARGET_BIND_LINEAGE_ROOT_DIGEST_DOMAIN = b"hermes.target-bind:lineage-root\0"
# A lineage deeper than this is treated as malformed rather than walked forever.
_TARGET_BIND_MAX_LINEAGE_DEPTH = 100
_TARGET_BIND_MAX_IDENTIFIER_LENGTH = 512
# The controller holds the generation as a JS number; a larger integer would reach the receipt digest
# with digits it cannot reproduce.
_TARGET_BIND_MAX_GENERATION = 2**53 - 1
_TARGET_BIND_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
# Caller-safe evidence: the wire receipt plus "schema". Never the private lineage_root_id or the
# binding_identity key digest.
_TARGET_BIND_EVIDENCE_FIELDS = (
    "schema",
    "domain",
    "version",
    "actor_id",
    "binding_generation",
    "executor_runtime_identity",
    "requested_session_id",
    "lineage_root_digest",
    "receipt_digest",
)


def _canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
    # Byte-compatible with the controller's canonical JSON: sorted keys, no whitespace, raw UTF-8.
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_digest(payload: Dict[str, Any]) -> str:
    """Return the canonical SHA-256 commitment for a public receipt payload."""
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _lineage_root_digest(lineage_root_id: str) -> str:
    """Commit to the resolved root bytes without disclosing the private root."""
    return "sha256:" + hashlib.sha256(
        _TARGET_BIND_LINEAGE_ROOT_DIGEST_DOMAIN + lineage_root_id.encode("utf-8")
    ).hexdigest()


def _require_generation(value: Any, error: type[Exception], message: str) -> int:
    if type(value) is not int or not 1 <= value <= _TARGET_BIND_MAX_GENERATION:
        raise error(message)
    return value


def _resolve_lineage_root(conn, session_id: str) -> str:
    """Resolve one exact durable lineage root, or fence malformed ancestry."""
    current = session_id
    seen = set()
    for _ in range(_TARGET_BIND_MAX_LINEAGE_DEPTH):
        if current in seen:
            raise TargetBindReceiptFenceError("target bind lineage is cyclic")
        seen.add(current)
        row = conn.execute(
            "SELECT id, parent_session_id FROM sessions WHERE id = ?", (current,)
        ).fetchone()
        if row is None:
            raise TargetBindReceiptFenceError("target bind session is unavailable")
        parent_id = row["parent_session_id"]
        if parent_id is None:
            return str(row["id"])
        if not isinstance(parent_id, str) or not parent_id:
            raise TargetBindReceiptFenceError("target bind lineage is invalid")
        current = parent_id
    raise TargetBindReceiptFenceError("target bind lineage is ambiguous")


def _receipt_record(
    *,
    session_id: str,
    lineage_root_id: str,
    actor_id: str,
    binding_generation: int,
    executor_runtime_identity: str,
) -> Tuple[str, Dict[str, Any]]:
    """Build the stable identity key and the immutable receipt evidence."""
    binding_identity = {
        "domain": _TARGET_BIND_RECEIPT_DOMAIN,
        "version": _TARGET_BIND_RECEIPT_VERSION,
        "actor_id": actor_id,
        "binding_generation": binding_generation,
        "executor_runtime_identity": executor_runtime_identity,
    }
    identity_digest = _canonical_digest(binding_identity)
    public_receipt = {
        **binding_identity,
        "requested_session_id": session_id,
        "lineage_root_digest": _lineage_root_digest(lineage_root_id),
    }
    record = {
        "schema": _TARGET_BIND_RECEIPT_SCHEMA,
        **public_receipt,
        "lineage_root_id": lineage_root_id,
        "binding_identity": identity_digest,
        "receipt_digest": _canonical_digest(public_receipt),
    }
    return _TARGET_BIND_RECEIPT_META_PREFIX + identity_digest.removeprefix("sha256:"), record


def _load_stored_record(row) -> Dict[str, Any]:
    try:
        stored = json.loads(row["value"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise TargetBindReceiptFenceError("target bind receipt is malformed") from exc
    if not isinstance(stored, dict):
        raise TargetBindReceiptFenceError("target bind receipt is malformed")
    return stored


class SessionTargetBindMixin:
    """Persist, replay and validate immutable target-bind receipts in ``state_meta``."""

    @staticmethod
    def _require_target_bind_identifier(value: Any, name: str) -> str:
        """Accept only a canonical opaque identifier for a bind receipt."""
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or len(value) > _TARGET_BIND_MAX_IDENTIFIER_LENGTH
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError(f"{name} must be a non-empty opaque identifier")
        return value

    @staticmethod
    def _target_bind_public_receipt(record: Dict[str, Any]) -> Dict[str, Any]:
        """Return only caller-safe, durable target-bind authorization evidence."""
        return {field: record[field] for field in _TARGET_BIND_EVIDENCE_FIELDS}

    def prepare_target_bind_receipt(
        self,
        session_id: str,
        actor_id: str,
        binding_generation: int,
        executor_runtime_identity: str,
        *,
        expected_lineage_root_digest: Optional[str] = None,
        patience_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Persist or replay one immutable target-authenticated bind receipt.

        ``expected_lineage_root_digest`` is a preflight fence for local callers: the bind is refused
        unless the session's durable lineage root hashes to it. Callers that omit it accept the
        server-resolved root. ``patience_s`` bounds the wait for the write lock (default: the store's).
        """
        session_id = self._require_target_bind_identifier(session_id, "session_id")
        actor_id = self._require_target_bind_identifier(actor_id, "actor_id")
        executor_runtime_identity = self._require_target_bind_identifier(
            executor_runtime_identity, "executor_runtime_identity"
        )
        binding_generation = _require_generation(
            binding_generation, ValueError, "binding_generation must be an integer in 1..2**53-1"
        )
        if expected_lineage_root_digest is not None and (
            not isinstance(expected_lineage_root_digest, str)
            or _TARGET_BIND_DIGEST_RE.fullmatch(expected_lineage_root_digest) is None
        ):
            raise ValueError("expected_lineage_root_digest must be canonical")

        def _do(conn):
            lineage_root_id = _resolve_lineage_root(conn, session_id)
            if (
                expected_lineage_root_digest is not None
                and expected_lineage_root_digest != _lineage_root_digest(lineage_root_id)
            ):
                raise TargetBindReceiptFenceError("expected target bind lineage root does not match")
            key, expected = _receipt_record(
                session_id=session_id,
                lineage_root_id=lineage_root_id,
                actor_id=actor_id,
                binding_generation=binding_generation,
                executor_runtime_identity=executor_runtime_identity,
            )
            row = conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
            if row is not None:
                stored = _load_stored_record(row)
                if stored != expected:
                    raise TargetBindReceiptConflictError(
                        "target bind identity is already bound to different evidence"
                    )
                return stored
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?)",
                (key, _canonical_json_bytes(expected).decode("utf-8")),
            )
            return expected

        return self._execute_write(_do, patience_s=patience_s)

    def _validate_target_bind_receipt_on_conn(
        self, conn, session_id: str, value: Any
    ) -> Dict[str, Any]:
        """Resolve caller evidence against exact current local durable state.

        ``value`` is the internal form (the public wire receipt plus ``schema``). Runs on the
        caller's connection so a writer can validate inside the transaction that consumes the receipt.
        """
        if not isinstance(value, dict) or set(value) != set(_TARGET_BIND_EVIDENCE_FIELDS):
            raise TargetBindReceiptFenceError("target bind receipt is not closed")
        if (
            value.get("schema") != _TARGET_BIND_RECEIPT_SCHEMA
            or value.get("domain") != _TARGET_BIND_RECEIPT_DOMAIN
            # ``True == 1`` and ``1.0 == 1``: only the int itself is the version the receipt committed to.
            or type(value.get("version")) is not int
            or value.get("version") != _TARGET_BIND_RECEIPT_VERSION
            or value.get("requested_session_id") != session_id
        ):
            raise TargetBindReceiptFenceError("target bind receipt is not current")
        actor_id = self._require_target_bind_identifier(value.get("actor_id"), "actor_id")
        runtime_identity = self._require_target_bind_identifier(
            value.get("executor_runtime_identity"), "executor_runtime_identity"
        )
        generation = _require_generation(
            value.get("binding_generation"),
            TargetBindReceiptFenceError,
            "target bind receipt generation is invalid",
        )
        key, expected = _receipt_record(
            session_id=session_id,
            lineage_root_id=_resolve_lineage_root(conn, session_id),
            actor_id=actor_id,
            binding_generation=generation,
            executor_runtime_identity=runtime_identity,
        )
        row = conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise TargetBindReceiptFenceError("target bind receipt is unavailable")
        if _load_stored_record(row) != expected:
            raise TargetBindReceiptFenceError("target bind receipt is stale")
        public = self._target_bind_public_receipt(expected)
        if value != public:
            raise TargetBindReceiptFenceError("target bind receipt does not match durable state")
        return public
