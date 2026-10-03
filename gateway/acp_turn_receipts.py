"""U4 — the durable receipt for one Telegram update admitted through ACP claim/dispatch.

Hermes stays the single Telegram consumer (CEO 23e4a944: no second ``getUpdates``, no new bot).
ACP's ``HermesGatewayReceiptPort`` learns a managed turn's outcome only through
``GET /v1/canonical-surface/receipts/telegram/{update_id}`` (``api_server_canonical.py``), answered
from the record this module reads and writes — never from a live turn, so a slow or crashed
gateway cannot make the port wait on one.

This module owns the receipt's shape and state machine. Writing it (claiming ``PENDING``,
settling ``COMPLETED``/``ABORTED``) is the managed-ingress seam's job (H2): a single predicate at
that seam decides which messages are managed, and the seam is not wired to call it yet. Today
nothing claims a receipt, so every lookup answers ``NEVER_FOUND`` and no behavior changes; the
contract is in place so the seam and ACP's A1/A2 can land in a later slice without moving it.

State machine, one ``state_meta`` row per update id, on the Telegram-bound canonical binding's own
SessionDB (claimed and settled with the same proven-handle primitives the canonical POST ingress
uses — ``claim_meta_once`` / ``compare_and_set_meta`` — so a receipt is never read or written on a
stale or reopened database file):

    (absent) --claim_pending()--> PENDING --settle_completed()--> COMPLETED
                                      |
                                      +-------settle_aborted()--> ABORTED

``PENDING`` is sticky by design (mirrors ``CanonicalReceiptCoordinator``): an uncertain outcome is
never silently retried. The one scheduled exception is ``sweep_dead_owner_receipts``, called at
gateway startup next to the other crash-recovery sweeps in ``run_startup.py`` — a ``PENDING``
receipt whose owner token names no turn this process is resuming did not survive a crash with its
answer; it is marked ``ABORTED HERMES_PROCESS_DIED_BEFORE_ANSWER`` so a stuck ``IN_DOUBT`` on the
ACP side does not wait forever for a process that is gone. A receipt whose owner *is* being
resumed is left ``PENDING``: the turn may still answer it.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

_RECEIPT_NAMESPACE = "acp-tg-receipt:v1"
_SCHEMA = "hermes.gateway-turn-receipt/v1"
_MAX_RECEIPT_CHARS = 32_768
_ABORT_PROCESS_DIED = "HERMES_PROCESS_DIED_BEFORE_ANSWER"
_ABORT_REFUSED_BEFORE_RUN = "REFUSED_BEFORE_RUN"


_MANAGED_PREFIX = "/acp "
_MANAGED_PREFIX_BARE = "/acp"  # the bare command with no task text is not managed — nothing to admit


def acp_managed_task_text(event_text: Optional[str]) -> Optional[str]:
    """The task text after an explicit ``/acp `` prefix, or None when this message is not a
    managed-ingress candidate. CEO 3c058be8's ruling (2026-10-03, Buzz event 36ade192): only this
    explicit prefix is the admission boundary — never a config opt-in flag or body classification,
    which would risk pulling ordinary DM traffic into ACP admission. Whitespace around the prefix
    is not trimmed first: a message must *start* with it, so quoting or discussing ``/acp`` text
    elsewhere in a sentence is not managed."""
    text = event_text or ""
    if text.startswith(_MANAGED_PREFIX):
        task = text[len(_MANAGED_PREFIX):].strip()
        return task or None
    return None


def is_acp_managed_message(event: Any, source: Any) -> bool:
    """H2's single admission-boundary predicate: is this inbound message a candidate for ACP
    claim/dispatch at all? True only for an explicit ``/acp <task>`` prefix on the Telegram
    origin a canonical binding is bound to — ordinary chat and status/recovery requests always
    take today's path, never routed through ACP by body classification or a config flag.

    The seam call site in ``run_inbound.py`` is not wired to the actual ACP admission call in
    this slice: even though this predicate can now return True, nothing yet dispatches the
    ``telegram-update.ingress.sock`` call, writes the pending receipt, or runs the turn through
    ACP — that integration needs ACP's A1/A2 contract (PR #1062) settled first; wiring a call
    against a still-repairing contract would be untested. Until then a ``/acp`` message takes
    the ordinary path like any other text, unrouted."""
    text = getattr(event, "text", None)
    return acp_managed_task_text(text) is not None


def receipt_key(update_id: Any) -> str:
    """The ``state_meta`` key one Telegram update's receipt lives under."""
    return f"{_RECEIPT_NAMESPACE}:{update_id}"


@dataclass(frozen=True)
class TelegramTurnReceipt:
    """The answer ``GET .../receipts/telegram/{update_id}`` serializes.

    ``status`` is always one of ``PENDING`` / ``COMPLETED`` / ``ABORTED`` / ``NEVER_FOUND``.
    ``NEVER_FOUND`` means this key was never claimed — a resend under it is safe; it is never
    returned for a key this process cannot currently read (see ``RECEIPT_UNREADABLE``, which reads
    as ``ABORTED`` so a caller never treats "cannot tell" as "safe to resend").
    """

    status: str
    update_id: str
    message_id: Optional[str] = None
    turn_request_id: Optional[str] = None
    receipt_identity: Optional[dict] = None
    receipt_id: Optional[str] = None
    evidence_digest: Optional[str] = None
    reason_code: Optional[str] = None
    content: Optional[str] = None
    delivery: Optional[dict] = None
    completed_at: Optional[float] = None

    def to_response(self) -> dict:
        out = {
            "schema": _SCHEMA, "update_id": self.update_id, "message_id": self.message_id,
            "status": self.status, "turnRequestId": self.turn_request_id,
            "receiptIdentity": self.receipt_identity, "receiptId": self.receipt_id,
            "evidenceDigest": self.evidence_digest, "reasonCode": self.reason_code,
            "delivery": self.delivery,
        }
        if self.content is not None:
            out["content"] = self.content
        return out


def not_found(update_id: Any) -> TelegramTurnReceipt:
    return TelegramTurnReceipt(status="NEVER_FOUND", update_id=str(update_id))


def _unreadable(update_id: Any, *, reason: str = "RECEIPT_UNREADABLE") -> TelegramTurnReceipt:
    # Unreadable reads as ABORTED, not NEVER_FOUND: the key WAS claimed, so a caller must not
    # treat this as license to resend under it while this process cannot tell what it holds.
    return TelegramTurnReceipt(status="ABORTED", update_id=str(update_id), reason_code=reason)


def _encode(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if len(encoded) > _MAX_RECEIPT_CHARS:
        raise ValueError("acp_turn_receipt_invalid")
    return encoded


def decode(raw: Optional[str], update_id: Any) -> TelegramTurnReceipt:
    """Parse one stored ``state_meta`` value; never raises — any malformed byte reads as the
    unreadable/aborted case rather than propagating a parse error into an HTTP 500."""
    if raw is None:
        return not_found(update_id)
    try:
        parsed = json.loads(raw)
    except Exception:
        return _unreadable(update_id)
    if not isinstance(parsed, dict) or parsed.get("v") != 1:
        return _unreadable(update_id)
    state = parsed.get("state")
    if state == "pending":
        return TelegramTurnReceipt(
            status="PENDING", update_id=str(update_id), message_id=parsed.get("message_id"),
            turn_request_id=parsed.get("turn_request_id"),
            receipt_identity=parsed.get("receipt_identity"),
        )
    if state == "terminal":
        terminal = parsed.get("terminal")
        if not isinstance(terminal, dict) or terminal.get("status") not in ("COMPLETED", "ABORTED"):
            return _unreadable(update_id)
        return TelegramTurnReceipt(
            status=terminal["status"], update_id=str(update_id), message_id=parsed.get("message_id"),
            turn_request_id=parsed.get("turn_request_id"), receipt_identity=parsed.get("receipt_identity"),
            receipt_id=terminal.get("receipt_id"), evidence_digest=terminal.get("evidence_digest"),
            reason_code=terminal.get("reason_code"), content=terminal.get("content"),
            delivery=terminal.get("delivery"), completed_at=terminal.get("completed_at"),
        )
    return _unreadable(update_id)


def lookup(db: Any, update_id: Any) -> TelegramTurnReceipt:
    """Read-only answer for the GET route: one ``get_meta`` call, no write, no lease, no turn."""
    return decode(db.get_meta(receipt_key(update_id)), update_id)


def claim_pending(
    db: Any, update_id: Any, *, message_id: str, turn_request_id: str, receipt_identity: dict,
    owner: Optional[str] = None, proven_db_path: Path, proven_db_identity: Optional[tuple],
) -> tuple[bool, str]:
    """Claim the receipt once, before the turn runs. Returns ``(claimed, owner)`` — on a losing
    race ``claimed`` is False and ``owner`` is the winner's, so a caller never runs the turn
    twice for one update id. ``owner`` is this process's token for ``sweep_dead_owner_receipts``;
    the H2 seam must persist it (e.g. alongside its in-flight turn table) before awaiting the run.
    """
    owner = owner or secrets.token_hex(16)
    pending = _encode({
        "v": 1, "state": "pending", "message_id": message_id, "turn_request_id": turn_request_id,
        "receipt_identity": receipt_identity, "owner": owner,
    })
    key = receipt_key(update_id)
    if db.claim_meta_once(key, pending, proven_db_path=proven_db_path, proven_db_identity=proven_db_identity):
        return True, owner
    existing = db.get_meta(key)
    try:
        won_owner = json.loads(existing).get("owner") if existing else None
    except Exception:
        won_owner = None
    return False, won_owner or owner


def _settle(
    db: Any, update_id: Any, *, terminal: dict, proven_db_path: Path,
    proven_db_identity: Optional[tuple],
) -> bool:
    key = receipt_key(update_id)
    pending_raw = db.get_meta(key)
    if pending_raw is None:
        return False
    try:
        parsed = json.loads(pending_raw)
    except Exception:
        return False
    if not isinstance(parsed, dict) or parsed.get("state") != "pending":
        return False  # already settled (or malformed) — compare_and_set below would refuse anyway
    value = _encode({**parsed, "state": "terminal", "terminal": terminal})
    return db.compare_and_set_meta(
        key, pending_raw, value, proven_db_path=proven_db_path, proven_db_identity=proven_db_identity,
    )


def settle_completed(
    db: Any, update_id: Any, *, receipt_id: str, evidence_digest: str, content: Optional[str] = None,
    delivery: Optional[dict] = None, proven_db_path: Path, proven_db_identity: Optional[tuple],
) -> bool:
    """Replace a PENDING receipt with its terminal COMPLETED outcome. False only when the
    receipt was not PENDING with these exact bytes (already settled by a concurrent caller, or
    the row is gone) — the caller's own answer is unaffected either way, since the turn already
    produced it; this only decides whether THIS settlement is the one that gets recorded."""
    return _settle(
        db, update_id, proven_db_path=proven_db_path, proven_db_identity=proven_db_identity,
        terminal={
            "status": "COMPLETED", "receipt_id": receipt_id, "evidence_digest": evidence_digest,
            "content": content, "delivery": delivery, "completed_at": time.time(),
        },
    )


def settle_aborted(
    db: Any, update_id: Any, *, reason_code: str, proven_db_path: Path,
    proven_db_identity: Optional[tuple],
) -> bool:
    return _settle(
        db, update_id, proven_db_path=proven_db_path, proven_db_identity=proven_db_identity,
        terminal={"status": "ABORTED", "reason_code": reason_code, "completed_at": time.time()},
    )


def refuse_before_run(
    db: Any, update_id: Any, *, message_id: str, turn_request_id: str, receipt_identity: dict,
    proven_db_path: Path, proven_db_identity: Optional[tuple],
) -> None:
    """H3: admission was given up on before any turn ran. Claims and immediately settles a
    REFUSED_BEFORE_RUN tombstone in one call so the GET route answers ABORTED, never PENDING, for
    an update the seam decided never to attempt. Best-effort: a receipt already claimed by a
    concurrent attempt is left exactly as that attempt is settling it."""
    claimed, owner = claim_pending(
        db, update_id, message_id=message_id, turn_request_id=turn_request_id,
        receipt_identity=receipt_identity, proven_db_path=proven_db_path,
        proven_db_identity=proven_db_identity,
    )
    if claimed:
        settle_aborted(
            db, update_id, reason_code=_ABORT_REFUSED_BEFORE_RUN,
            proven_db_path=proven_db_path, proven_db_identity=proven_db_identity,
        )


def sweep_dead_owner_receipts(
    db: Any, *, resuming_owners: Iterable[str], proven_db_path: Path,
    proven_db_identity: Optional[tuple],
) -> list[str]:
    """H3 crash recovery: called once at gateway startup, next to the other interrupted-delivery
    sweeps in ``run_startup.py``. Every PENDING receipt whose ``owner`` is not one this process is
    resuming did not survive whatever process claimed it; it is settled ABORTED
    HERMES_PROCESS_DIED_BEFORE_ANSWER so a port polling it is not left waiting on a turn that no
    longer exists. A receipt whose owner IS being resumed is left untouched — its turn may still
    answer it. Returns the update ids this call aborted."""
    resuming = set(resuming_owners)
    aborted: list[str] = []
    for key, raw in db.list_meta_prefix(f"{_RECEIPT_NAMESPACE}:"):
        update_id = key[len(f"{_RECEIPT_NAMESPACE}:"):]
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if not isinstance(parsed, dict) or parsed.get("state") != "pending":
            continue
        if parsed.get("owner") in resuming:
            continue
        if settle_aborted(
            db, update_id, reason_code=_ABORT_PROCESS_DIED,
            proven_db_path=proven_db_path, proven_db_identity=proven_db_identity,
        ):
            aborted.append(update_id)
    return aborted
