"""U4 H2 — ACP admission for an explicit ``/acp <task>`` message on the bound Telegram chat.

CEO 36ade192 (2026-10-03): only the explicit ``/acp`` prefix is the managed boundary. Ordinary
conversation and recovery requests never reach this module. A managed message runs only after
ACP answers ``allowed`` on its ``telegram-update.ingress.sock`` lane (admission, claim and dispatch
committed in one ACP transaction, ACP #1062). A deny, a timeout or an unreadable answer means the
turn does not run here and is not retried: the outcome is reported to the owner and left to ACP,
which sees the claimed turn as in doubt until it reads this gateway's receipt.

The receipt (``gateway.acp_turn_receipts``) is claimed before the turn runs and settled COMPLETED
by ``send_final_ledgered`` once the final reply has an obligation id. Nothing here writes ABORTED
on a timeout: an admission whose answer never arrived may still have committed on ACP's side, so
the receipt stays absent (NEVER_FOUND, i.e. in doubt for ACP) rather than claiming a non-run this
process cannot prove.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from gateway import acp_turn_receipts as receipts

logger = logging.getLogger(__name__)

_SCHEMA = "acp.telegram-external-update/v1"
_BINDING = "acp-canonical-ceo"
_SOCKET_NAME = "telegram-update.ingress.sock"
_KEYCHAIN_SERVICE = "com.agentcontrolplane.agentcpd"
_KEYCHAIN_ACCOUNT = "ACP_TELEGRAM_EXTERNAL_SECRET"
_ADMISSION_TIMEOUT_S = 15.0
_MAX_ANSWER_BYTES = 16 * 1024
_TURN_FIELDS = (
    "turnRequestId", "targetActorId", "promptDigest", "bindingGeneration", "targetBindingId",
    "targetAttestationId", "executorSessionId", "executorSessionIncarnation",
)

REFUSED = "🛑 ACP did not admit this /acp task ({reason}); it was not run."
UNKNOWN = ("⚠️ The ACP admission outcome for this /acp task is unknown. It was not run here and "
           "will not be retried automatically; check its state in ACP and resend if needed.")
BUSY = "⏳ Another turn is running, so this /acp task was not admitted. Resend it when the turn ends."
DUPLICATE = "This /acp update was already handled and was not run again (receipt: {status})."


def socket_path() -> Path:
    """ACP's owner-only state directory (the daemon's database dir) holds the lane's socket."""
    return Path.home() / ".agent-control-plane" / _SOCKET_NAME


def read_secret() -> Optional[str]:
    """The shared lane secret from the login Keychain, never logged or returned to a caller's reply."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-w", "-s", _KEYCHAIN_SERVICE, "-a", _KEYCHAIN_ACCOUNT],
            capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    secret = result.stdout.strip() if result.returncode == 0 else ""
    return secret or None


def bound_binding(runner: Any, source: Any) -> Any:
    """The ACP CEO canonical binding whose Telegram origin this source is, or None."""
    from gateway.config import Platform

    if getattr(source, "platform", None) != Platform.TELEGRAM:
        return None
    binding = (getattr(runner.config, "canonical_surface_bindings", None) or {}).get(_BINDING)
    if binding is None or str(source.chat_id) != str(binding.telegram_chat_id):
        return None
    if binding.telegram_user_id is not None and str(source.user_id) != str(binding.telegram_user_id):
        return None
    return binding


def is_managed(runner: Any, event: Any, source: Any) -> bool:
    """The single seam predicate: an explicit ``/acp <task>`` on the bound chat."""
    return receipts.is_acp_managed_message(event, source) and bound_binding(runner, source) is not None


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None and str(value).lstrip("-").isdigit() else None
    except (TypeError, ValueError):
        return None


def build_envelope(event: Any, source: Any, secret: str) -> Optional[dict]:
    """The closed request ACP #1062 parses, or None when the event lacks a numeric id it needs."""
    update_id, message_id = _as_int(event.platform_update_id), _as_int(event.message_id)
    from_id, chat_id = _as_int(source.user_id), _as_int(source.chat_id)
    if None in (update_id, message_id, from_id, chat_id):
        return None
    message: dict = {"message_id": message_id, "from": {"id": from_id},
                     "chat": {"id": chat_id, "type": "private"}, "text": event.text}
    raw = getattr(event, "raw_message", None)
    if getattr(raw, "forward_origin", None) is not None:
        # ACP refuses forwarded messages as data; the marker's presence is what it reads.
        message["forward_origin"] = "forwarded"
    return {"schema": _SCHEMA, "binding": _BINDING, "secret": secret,
            "update": {"update_id": update_id, "message": message}}


async def request_admission(envelope: dict, *, path: Optional[Path] = None,
                            timeout: float = _ADMISSION_TIMEOUT_S) -> Optional[dict]:
    """One NDJSON envelope out, one answer line back. None = the outcome is unknown."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(path or socket_path())), timeout)
    except (OSError, asyncio.TimeoutError):
        return None
    try:
        writer.write(json.dumps(envelope, ensure_ascii=False).encode("utf-8") + b"\n")
        await asyncio.wait_for(writer.drain(), timeout)
        line = await asyncio.wait_for(reader.readline(), timeout)
    except (OSError, asyncio.TimeoutError, ValueError):
        return None
    finally:
        writer.close()
    if not line or len(line) > _MAX_ANSWER_BYTES:
        return None
    try:
        answer = json.loads(line)
    except ValueError:
        return None
    return answer if isinstance(answer, dict) and isinstance(answer.get("allowed"), bool) else None


def _turn_identity(answer: dict) -> Optional[dict]:
    turn = answer.get("turn")
    if not isinstance(turn, dict) or any(field not in turn for field in _TURN_FIELDS):
        return None
    return {field: turn[field] for field in _TURN_FIELDS}


def _binding_lineage_digest(db: Any, session_id: str) -> Optional[str]:
    from hermes_state_target_bind import _lineage_root_digest

    try:
        chain = db._session_lineage_root_to_tip(session_id)
    except Exception:
        return None
    return _lineage_root_digest(chain[0]) if chain else None


@dataclass
class Admission:
    update_id: int
    db: Any
    proof: dict
    task_text: str


@dataclass
class Outcome:
    admission: Optional[Admission] = None
    reply: Optional[str] = None  # set when the turn must not run


async def admit(runner: Any, event: Any, source: Any, session_key: str, *, path: Optional[Path] = None,
                secret: Optional[str] = None, timeout: float = _ADMISSION_TIMEOUT_S) -> Outcome:
    """Ask ACP to admit this managed message and claim its receipt; never runs the turn itself."""
    binding = bound_binding(runner, source)
    task_text = receipts.acp_managed_task_text(event.text)
    secret = secret if secret is not None else read_secret()
    envelope = build_envelope(event, source, secret) if secret and task_text else None
    if envelope is None:
        logger.warning("ACP managed message not admitted: lane secret or Telegram ids unavailable")
        return Outcome(reply=UNKNOWN)
    answer = await request_admission(envelope, path=path, timeout=timeout)
    if answer is None:
        # The answer never arrived: ACP may still have committed the claim, so nothing is written
        # here — the receipt stays absent and ACP keeps the turn in doubt (CEO 3c058be8).
        logger.warning("ACP admission outcome unknown for update %s; not running, not retrying",
                       envelope["update"]["update_id"])
        return Outcome(reply=UNKNOWN)
    if not answer["allowed"]:
        return Outcome(reply=REFUSED.format(reason=str(answer.get("reasonCode") or "denied")[:64]))

    update_id = envelope["update"]["update_id"]
    turn = _turn_identity(answer)
    source_info = answer.get("source") or {}
    db = runner.session_store._db_for_key(binding.session_key)
    proof = {"proven_db_path": db.db_path, "proven_db_identity": db._db_file_identity}
    entry = runner.session_store.lookup_by_session_key_existing(binding.session_key)
    expected_lineage = _binding_lineage_digest(db, entry.session_id) if entry is not None else None
    target_bind = answer.get("targetBind") or {}
    if (turn is None or source_info.get("nonce") != f"update:{update_id}"
            or expected_lineage is None or target_bind.get("lineage_root_digest") != expected_lineage):
        if turn is not None:
            # ACP dispatched a turn this gateway refuses to run: record the definite non-run so
            # ACP settles it ABORTED instead of waiting on it.
            receipts.refuse_before_run(db, update_id, message_id=str(event.message_id),
                                       turn_request_id=str(turn["turnRequestId"]), receipt_identity=turn, **proof)
        return Outcome(reply=REFUSED.format(reason="target mismatch"))
    if runner._is_session_running(session_key):
        receipts.refuse_before_run(db, update_id, message_id=str(event.message_id),
                                   turn_request_id=str(turn["turnRequestId"]), receipt_identity=turn, **proof)
        return Outcome(reply=BUSY)
    claimed, _owner = receipts.claim_pending(
        db, update_id, message_id=str(event.message_id), turn_request_id=str(turn["turnRequestId"]),
        receipt_identity=turn, owner=process_owner(), **proof)
    if not claimed:
        # An earlier delivery of this update already claimed (and ran) it: zero duplicate execution.
        return Outcome(reply=DUPLICATE.format(status=receipts.lookup(db, update_id).status))
    return Outcome(admission=Admission(update_id=update_id, db=db, proof=proof, task_text=task_text))


def abort_claimed(admission: Optional[Admission]) -> None:
    """The receipt was claimed but the turn did not start (a gate after admission refused it)."""
    if admission is not None:
        receipts.settle_aborted(admission.db, admission.update_id,
                                reason_code="REFUSED_BEFORE_RUN", **admission.proof)


_PROCESS_OWNER: Optional[str] = None


def process_owner() -> str:
    """This gateway process's owner token for claimed receipts; the startup sweep aborts PENDING
    receipts whose owner is any other value (a process that died before answering)."""
    global _PROCESS_OWNER
    if _PROCESS_OWNER is None:
        import os
        import secrets as _secrets

        _PROCESS_OWNER = f"pid{os.getpid()}-{_secrets.token_hex(8)}"
    return _PROCESS_OWNER


def settle_after_delivery(event: Any, *, obligation_id: Optional[str], text: str, result: Any) -> None:
    """Called by ``send_final_ledgered`` after the final reply's obligation is finalized. A failed
    settlement leaves the receipt PENDING — in doubt, never promoted (CEO 3c058be8)."""
    admission = getattr(event, "_acp_admission", None)
    if admission is None or obligation_id is None:
        return
    import hashlib

    digest = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    delivery = {
        "chat_id": str(event.source.chat_id), "reply_to_message_id": str(event.message_id),
        "message_ids": [str(result.message_id)] if getattr(result, "message_id", None) else [],
        "content_digest": digest, "state": "sent" if getattr(result, "success", False) else "failed",
    }
    try:
        settled = receipts.settle_completed(
            admission.db, admission.update_id, receipt_id=f"hermes-tg:{obligation_id}",
            evidence_digest=digest, content=text, delivery=delivery, **admission.proof)
    except Exception:
        logger.exception("ACP receipt settlement failed for update %s; left in doubt", admission.update_id)
        return
    if not settled:
        logger.warning("ACP receipt for update %s was not PENDING at settlement; left as recorded",
                       admission.update_id)


def _binding_db(runner: Any) -> Any:
    config = getattr(runner, "config", None)
    binding = (getattr(config, "canonical_surface_bindings", None) or {}).get(_BINDING)
    return runner.session_store._db_for_key(binding.session_key) if binding is not None else None


def attach_obligation(event: Any, obligation_id: Optional[str]) -> None:
    """Called by ``send_final_ledgered`` once the final reply has a ledger row, before the send."""
    admission = getattr(event, "_acp_admission", None)
    if admission is None or obligation_id is None:
        return
    try:
        receipts.attach_obligation(admission.db, admission.update_id, obligation_id, **admission.proof)
    except Exception:
        logger.exception("ACP receipt obligation attach failed for update %s", admission.update_id)


def settle_redelivered(runner: Any, obligation_id: str, content: str) -> list:
    """The boot redelivery sent a ledgered final: settle the receipt that recorded it."""
    db = _binding_db(runner)
    if db is None:
        return []
    return receipts.settle_by_obligation(
        db, obligation_id, content, proven_db_path=db.db_path, proven_db_identity=db._db_file_identity)


def sweep_at_startup(runner: Any) -> list:
    """H3: abort PENDING receipts claimed by a process that is no longer this one."""
    db = _binding_db(runner)
    if db is None:
        return []
    return receipts.sweep_dead_owner_receipts(
        db, resuming_owners=(process_owner(),),
        proven_db_path=db.db_path, proven_db_identity=db._db_file_identity)
