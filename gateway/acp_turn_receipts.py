"""U4 — the durable receipt for one Telegram update admitted through ACP claim/dispatch.

Hermes stays the single Telegram consumer (CEO 23e4a944: no second ``getUpdates``, no new bot).
ACP's ``HermesGatewayReceiptPort`` learns a managed turn's outcome only through
``GET /v1/canonical-surface/receipts/telegram/{update_id}`` (``api_server_canonical.py``), answered
from the record this module reads and writes — never from a live turn, so a slow or crashed
gateway cannot make the port wait on one.

This module owns the receipt's shape and state machine. Writing it is the managed-ingress seam's
job (``gateway.acp_managed_ingress``, H2): an explicit ``/acp <task>`` on the bound chat is admitted
by ACP over its ingress lane, and only an allowed admission claims ``PENDING`` before the turn runs.
A dispatched turn refused before it runs, or one that fails with no answer recorded, settles
``ABORTED``. The answer goes out only through the ledgered final send, whose delivery-obligation
row carries the update id; the receipt settles ``COMPLETED`` from that delivered row
(``settle_from_ledger``), and ``lookup`` reads the row too. An update that was never admitted here has no receipt and answers
``NEVER_FOUND``.

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
import re
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
# The answer was ledgered but the ledger abandoned it undelivered.
_ABORT_UNDELIVERABLE = "HERMES_ANSWER_UNDELIVERABLE"


_MANAGED_COMMAND = "/acp"
_COMMAND_MENTION = re.compile(r"@([A-Za-z0-9_]+)")


def acp_command_task(event_text: Optional[str], bot_username: Optional[str] = None) -> Optional[str]:
    """The task text of a message whose command token is ``/acp`` (``""`` when it carries none), or
    None when it is not one. CEO 3c058be8's ruling (2026-10-03, Buzz event 36ade192): only this
    explicit command is the admission boundary — never a config opt-in flag or body classification,
    which would risk pulling ordinary DM traffic into ACP admission.

    The token is ``/acp`` at the very start of the text (nothing is trimmed first, so quoting or
    discussing ``/acp`` elsewhere in a sentence is not managed), followed by whitespace (space, tab,
    newline) or the end of the text, or by ``@<bot_username>`` — this bot's own handle, compared
    case-insensitively — and then whitespace or the end. ``/acpx``, ``/acp@other_bot`` and an
    ``@`` form with no known own handle are not the token. The task is everything after the token,
    stripped."""
    text = event_text or ""
    if not text.startswith(_MANAGED_COMMAND):
        return None
    rest = text[len(_MANAGED_COMMAND):]
    if rest.startswith("@"):
        mention = _COMMAND_MENTION.match(rest)
        own = (bot_username or "").lstrip("@").lower()
        if mention is None or not own or mention.group(1).lower() != own:
            return None
        rest = rest[mention.end():]
    if rest and not rest[0].isspace():
        return None
    return rest.strip()


def acp_managed_task_text(event_text: Optional[str], bot_username: Optional[str] = None) -> Optional[str]:
    """The non-empty task text of an ``/acp`` command (``acp_command_task``), or None when the
    message is not one or carries no task."""
    return acp_command_task(event_text, bot_username) or None


def is_acp_managed_message(event: Any, source: Any, bot_username: Optional[str] = None) -> bool:
    """H2's single admission-boundary predicate: is this inbound message a candidate for ACP
    claim/dispatch at all? True for any message whose command token is ``/acp``
    (``acp_command_task``), including one with no task — on the bound chat that message is the
    managed path's to refuse, never the generic slash dispatch's. Ordinary chat and status/recovery
    requests always take today's path, never routed through ACP by body classification or a config
    flag; the bound-chat check is ``gateway.acp_managed_ingress.is_managed``'s."""
    text = getattr(event, "text", None)
    return acp_command_task(text, bot_username) is not None


def receipt_key(update_id: Any) -> str:
    """The ``state_meta`` key one Telegram update's receipt lives under."""
    return f"{_RECEIPT_NAMESPACE}:{update_id}"


_COMPLETED_REASON = "OK"
_MAX_SAFE_INTEGER = 2**53 - 1  # JavaScript's Number.MAX_SAFE_INTEGER
_CANONICAL_ID = re.compile(r"[1-9][0-9]*")


def _wire_id(value: Any) -> Any:
    """A Telegram id as the JSON integer ACP reads when it is a positive safe integer, else the
    stored value unchanged (see ``TelegramTurnReceipt.to_response``)."""
    if isinstance(value, int) and not isinstance(value, bool):
        number = value
    elif isinstance(value, str) and _CANONICAL_ID.fullmatch(value):
        number = int(value)
    else:
        return value
    return number if 0 < number <= _MAX_SAFE_INTEGER else value


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
        # Wire contract read by ACP's HermesGatewayReceiptPort (terminalReceipt): ``update_id`` is the
        # JSON number of the update asked about and ``message_id`` a positive safe integer, so each is
        # sent as an integer when it is one (canonical decimal, 1..2**53-1). Any other stored value is
        # sent exactly as stored, never coerced, and ACP refuses that body rather than read it under an
        # id Hermes cannot vouch for. ``reasonCode`` is a non-empty string on every terminal receipt:
        # "OK" for COMPLETED, the recorded failure code for ABORTED. Stored values are not rewritten.
        reason_code = self.reason_code
        if self.status == "COMPLETED" and not reason_code:
            reason_code = _COMPLETED_REASON
        out = {
            "schema": _SCHEMA, "update_id": _wire_id(self.update_id), "message_id": _wire_id(self.message_id),
            "status": self.status, "turnRequestId": self.turn_request_id,
            "receiptIdentity": self.receipt_identity, "receiptId": self.receipt_id,
            "evidenceDigest": self.evidence_digest, "reasonCode": reason_code,
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


def ledger_answers(db: Any, update_id: Any) -> list[dict]:
    """The delivery-ledger rows answering one /acp update, read on the receipt's own store handle:
    the ledger lives in the same state.db, and each row carries ``acp_update_id`` from the INSERT
    that recorded the answer. A store without the table or the column has no answers."""
    import sqlite3

    try:
        rows = db._read_all(
            "SELECT obligation_id, content, state, chat_id, delivered_message_ids FROM delivery_obligations "
            "WHERE acp_update_id = ? ORDER BY created_at", (str(update_id),))
    except sqlite3.OperationalError:
        return []
    return [{"obligation_id": r[0], "content": r[1], "state": r[2], "chat_id": r[3],
             "message_ids": r[4]} for r in rows]


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _completed_terminal(answer: dict, reply_to_message_id: Any) -> Optional[dict]:
    """COMPLETED evidence from the ledgered answer that was actually delivered, in the delivery shape
    ACP's receipt port parses: exactly {obligation_id, state, content_digest, chat_id,
    reply_to_message_id, message_ids}, with integer ids and at least one sent message. None when the
    row cannot prove that (no recorded message ids, a non-numeric id): the receipt then stays in doubt
    rather than claiming a delivery ACP would reject."""
    import hashlib

    try:
        message_ids = [int(m) for m in json.loads(answer.get("message_ids") or "[]")]
    except (TypeError, ValueError):
        return None
    chat_id, reply_to = _int_or_none(answer.get("chat_id")), _int_or_none(reply_to_message_id)
    if not message_ids or chat_id is None or reply_to is None:
        return None
    digest = "sha256:" + hashlib.sha256(answer["content"].encode("utf-8")).hexdigest()
    return {"status": "COMPLETED", "receipt_id": f"hermes-tg:{answer['obligation_id']}",
            "evidence_digest": digest, "content": answer["content"],
            "delivery": {"obligation_id": answer["obligation_id"], "state": "delivered",
                         "content_digest": digest, "chat_id": chat_id,
                         "reply_to_message_id": reply_to, "message_ids": message_ids},
            "completed_at": time.time()}


def _delivered_terminal(db: Any, update_id: Any, reply_to_message_id: Any,
                        obligation_id: Optional[str] = None) -> Optional[dict]:
    for answer in ledger_answers(db, update_id):
        if answer["state"] == "delivered" and obligation_id in (None, answer["obligation_id"]):
            terminal = _completed_terminal(answer, reply_to_message_id)
            if terminal is not None:
                return terminal
    return None


def lookup(db: Any, update_id: Any) -> TelegramTurnReceipt:
    """Read-only answer for the GET route: no write, no lease, no turn. A PENDING receipt whose
    answer the ledger records as delivered reads COMPLETED from that row, so a settlement write that
    failed after delivery never leaves ACP waiting on an answer the owner already has."""
    receipt = decode(db.get_meta(receipt_key(update_id)), update_id)
    if receipt.status != "PENDING":
        return receipt
    terminal = _delivered_terminal(db, update_id, receipt.message_id)
    if terminal is None:
        return receipt
    return TelegramTurnReceipt(
        status="COMPLETED", update_id=receipt.update_id, message_id=receipt.message_id,
        turn_request_id=receipt.turn_request_id, receipt_identity=receipt.receipt_identity,
        receipt_id=terminal["receipt_id"], evidence_digest=terminal["evidence_digest"],
        content=terminal["content"], delivery=terminal["delivery"])


class ReceiptContractError(Exception):
    """A receipt this store holds that cannot be stated in ACP's receipt contract without inventing a
    value. The GET answers an explicit error for it, so ACP keeps the turn in doubt."""


_ABORTED_EVIDENCE_SCHEMA = "hermes.gateway-turn-receipt.aborted-evidence/v1"
_ABORTED_RECEIPT_PREFIX = "hermes-tg:aborted:"


def served(db: Any, update_id: Any) -> TelegramTurnReceipt:
    """The receipt the GET route serves: ``lookup``, with an ABORTED receipt given its receipt id and
    evidence digest (``aborted_on_the_wire``). Raises ``ReceiptContractError`` instead of serving one
    it cannot state. Read-only, like ``lookup``."""
    receipt = lookup(db, update_id)
    return aborted_on_the_wire(db, receipt) if receipt.status == "ABORTED" else receipt


def aborted_on_the_wire(db: Any, receipt: TelegramTurnReceipt) -> TelegramTurnReceipt:
    """An ABORTED receipt with the ``receiptId`` and ``evidenceDigest`` ACP requires, computed on every
    read from what the store preserves, so receipts stored before this existed are served unchanged
    on disk and every read of one answers the same two values (no clock, no randomness).

    receiptId = ``hermes-tg:aborted:<update_id>``. The store holds exactly one receipt per update id
    (claimed once), ACP admits at most one turn per update (nonce ``update:<id>``) and asks by that id,
    so it names exactly this receipt; it cannot collide with a COMPLETED id (``hermes-tg:<obligation
    hex>``), and it stays short and printable whatever the turn id holds.

    evidenceDigest = ``sha256:`` + hex SHA-256 of the UTF-8, sorted-key, compact JSON of exactly
    {"schema": "hermes.gateway-turn-receipt.aborted-evidence/v1", "update_id": <int>, "message_id":
    <as stored>, "turnRequestId": <as stored>, "receiptIdentity": <as stored>, "reasonCode": <as
    stored>, "deliveredAnswer": false}. ``deliveredAnswer`` is the ledger part: re-checked on every
    read, and a delivered row for the update is a contradiction refused below, never served. The
    undelivered rows themselves are not bound: their story is the reason code, and the ledger prunes
    them, which would change the digest on a later read.

    Raises ``ReceiptContractError`` — never a guessed value — when the update id is not a positive
    safe integer, the turn identity or the reason code was not preserved (``RECEIPT_UNREADABLE`` has
    neither), a delivery is recorded on the receipt, or the ledger cannot say no answer was delivered."""
    import dataclasses
    import hashlib

    update_id = _wire_id(receipt.update_id)
    if not isinstance(update_id, int) or isinstance(update_id, bool):
        raise ReceiptContractError("the update id is not a positive safe integer")
    identity, turn, reason = receipt.receipt_identity, receipt.turn_request_id, receipt.reason_code
    if (not isinstance(identity, dict) or not isinstance(turn, str) or not turn
            or not isinstance(reason, str) or not reason):
        raise ReceiptContractError("the aborted receipt did not preserve its turn identity and reason code")
    if receipt.delivery is not None:
        raise ReceiptContractError("an aborted receipt records a delivery")
    delivered = _delivered_answer_recorded(db, receipt.update_id)
    if delivered is None:
        raise ReceiptContractError("the delivery ledger cannot be read")
    if delivered:
        raise ReceiptContractError("an answer to this update was delivered")
    evidence = {"schema": _ABORTED_EVIDENCE_SCHEMA, "update_id": update_id, "message_id": receipt.message_id,
                "turnRequestId": turn, "receiptIdentity": identity, "reasonCode": reason,
                "deliveredAnswer": False}
    encoded = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return dataclasses.replace(receipt, receipt_id=f"{_ABORTED_RECEIPT_PREFIX}{update_id}",
                               evidence_digest="sha256:" + hashlib.sha256(encoded).hexdigest())


def _delivered_answer_recorded(db: Any, update_id: Any) -> Optional[bool]:
    """Whether the ledger records a delivered answer to this update; None when it cannot be read. A
    store without the ledger table (or an older one without ``acp_update_id``) has recorded none."""
    import sqlite3

    try:
        rows = db._read_all(
            "SELECT 1 FROM delivery_obligations WHERE acp_update_id = ? AND state = 'delivered' LIMIT 1",
            (str(update_id),))
    except sqlite3.OperationalError as exc:
        return False if ("no such table" in str(exc) or "no such column" in str(exc)) else None
    except Exception:
        return None
    return bool(rows)


def settle_from_ledger(
    db: Any, update_id: Any, *, obligation_id: Optional[str] = None, proven_db_path: Path,
    proven_db_identity: Optional[tuple],
) -> bool:
    """Settle COMPLETED from the delivered ledger row (``obligation_id`` when given)."""
    pending = decode(db.get_meta(receipt_key(update_id)), update_id)
    if pending.status != "PENDING":
        return False
    terminal = _delivered_terminal(db, update_id, pending.message_id, obligation_id)
    if terminal is None:
        return False
    return _settle(db, update_id, terminal=terminal,
                   proven_db_path=proven_db_path, proven_db_identity=proven_db_identity)


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
    proof = {"proven_db_path": proven_db_path, "proven_db_identity": proven_db_identity}
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
        # The ledger decides before any abort: a delivered answer settles COMPLETED; an answer
        # still owed (pending/attempting/failed) is the redelivery's to settle; a correlation an
        # older build recorded on the receipt is never treated as a missing answer.
        answers = ledger_answers(db, update_id)
        if settle_from_ledger(db, update_id, **proof):
            continue
        # "delivered" here is a delivered row without provable message ids: answered, so never a
        # death before answer, but not COMPLETED evidence either — it stays in doubt.
        if parsed.get("obligation_id") or any(
                a["state"] in ("pending", "attempting", "failed", "delivered") for a in answers):
            continue
        reason = _ABORT_UNDELIVERABLE if answers else _ABORT_PROCESS_DIED
        if settle_aborted(db, update_id, reason_code=reason, **proof):
            aborted.append(update_id)
    return aborted
