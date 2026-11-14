"""U4 H2 — ACP admission for an explicit ``/acp <task>`` message on the bound Telegram chat.

CEO 36ade192 (2026-10-03): only the explicit ``/acp`` prefix is the managed boundary. Ordinary
conversation and recovery requests never reach this module. A managed message runs only after
ACP answers ``allowed`` on its ``telegram-update.ingress.sock`` lane (admission, claim and dispatch
committed in one ACP transaction, ACP #1062). A deny, a timeout or an unreadable answer means the
turn does not run here and is not retried: the outcome is reported to the owner and left to ACP,
which sees the claimed turn as in doubt until it reads this gateway's receipt.

The receipt (``gateway.acp_turn_receipts``) is claimed before the turn runs. The managed turn is not
streamed: its answer goes out only through the ledgered final send, whose obligation row carries the
update id in the same INSERT. The receipt settles COMPLETED from that row once it is delivered, live or
by redelivery; GET and the startup sweep read the row too, so no settlement write can strand an answer
the owner already has. Nothing here writes ABORTED
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
# Telegram splits a longer paste client-side into messages Hermes cannot tell from separate ones, so a
# managed task must fit in one (the adapter's own split threshold).
_MAX_TASK_MESSAGE_CHARS = 4000
_MAX_ANSWER_BYTES = 16 * 1024
_TURN_FIELDS = (
    "turnRequestId", "targetActorId", "promptDigest", "bindingGeneration", "targetBindingId",
    "targetAttestationId", "executorSessionId", "executorSessionIncarnation",
)

REFUSED = "🛑 ACP did not admit this /acp task ({reason}); it was not run."
UNKNOWN = ("⚠️ The ACP admission outcome for this /acp task is unknown. It was not run here and "
           "will not be retried automatically; check its state in ACP and resend if needed.")
BUSY = "⏳ Another turn is running, so this /acp task was not admitted. Resend it when the turn ends."
TOO_LONG = ("🛑 This /acp task is too long for one Telegram message, so it was not run. Send it under "
            "4000 characters (or as a file and refer to it).")
UNRECORDED = ("⚠️ The answer to this /acp task could not be recorded for delivery, so it was not sent. "
              "It will not be retried; check the task in ACP and resend if needed.")
DUPLICATE = "This /acp update was already handled and was not run again (receipt: {status})."
PAUSED = REFUSED.format(reason="managed admission paused")
EMPTY_TASK = REFUSED.format(reason="empty task")
# Given to the model, API-only, on a turn that runs under a server-confirmed ACP admission.
MANAGED_TURN_NOTE = ("[System note: The owner sent this with the /acp command on the bound Telegram chat, and "
                     "ACP admitted it as a managed task. The server removed the /acp command before this turn; "
                     "the text below is the admitted task.]")


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
    named = getattr(runner.session_store, "_named_profile_for_key", None)
    if callable(named) and named(binding.session_key) is not None:
        # Managed /acp is supported only for a launch-home (agent:main) binding, whose receipts and
        # delivery ledger share one state.db. A named-profile binding keeps today's ordinary path.
        return None
    return binding


def launch_home_store(runner: Any) -> Any:
    """The one store receipts and the delivery ledger both live in: the binding's DB, accepted only
    when it is the launch home's state.db (the ledger's file). None when they differ."""
    from gateway.delivery_ledger import _db_path

    binding = (getattr(getattr(runner, "config", None), "canonical_surface_bindings", None) or {}).get(_BINDING)
    if binding is None:
        return None
    db = runner.session_store._db_for_key(binding.session_key)
    try:
        same = Path(db.db_path).resolve() == Path(_db_path()).resolve()
    except Exception:
        return None
    return db if same else None


def bot_username(runner: Any, source: Any, adapter: Any = None) -> Optional[str]:
    """The live @username of the bot that received *source*'s message (the Telegram adapter's
    ``_current_bot_username``), or None when it is not known — then no ``/acp@handle`` form is ours."""
    try:
        if adapter is None:
            adapter = runner._intake_adapter_for(source)
        current = getattr(adapter, "_current_bot_username", None)
        return (current() or None) if callable(current) else None
    except Exception:
        logger.debug("ACP managed check: bot username unavailable", exc_info=True)
        return None


def _username_for(runner: Any, event: Any, source: Any, adapter: Any = None) -> Optional[str]:
    # Only the ``/acp@handle`` form needs the bot's own handle.
    return bot_username(runner, source, adapter) if (getattr(event, "text", None) or "").startswith("/acp@") else None


def is_managed(runner: Any, event: Any, source: Any, adapter: Any = None) -> bool:
    """The single seam predicate: a message whose command token is ``/acp`` (with or without a task)
    from the bound owner on the bound chat. Such a message is the managed path's alone — admitted,
    or refused as an ordinary reply — and never reaches the generic slash dispatch."""
    return (receipts.is_acp_managed_message(event, source, _username_for(runner, event, source, adapter))
            and bound_binding(runner, source) is not None)


def managed_task_text(runner: Any, event: Any, source: Any, adapter: Any = None) -> Optional[str]:
    """The managed message's task text (``""`` when it carries none), by the same rule as is_managed."""
    return receipts.acp_command_task(getattr(event, "text", None), _username_for(runner, event, source, adapter))


def admission_enabled(config: Optional[dict] = None) -> bool:
    """Read the ``gateway.acp_managed_admission`` gate (default on), re-read on every call like
    ``gateway.delivery_ledger``, so an operator can pause managed admission without a restart."""
    try:
        if config is None:
            from hermes_cli.config import load_config
            config = load_config()
        value = (config.get("gateway") or {}).get("acp_managed_admission", True)
        return value.strip().lower() not in {"false", "0", "no", "off"} if isinstance(value, str) else bool(value)
    except Exception:
        return True


def paused_reply() -> Optional[str]:
    """The ordinary refusal for a managed message while the gate is closed, else None. Checked right
    after classification, before any ACP request, receipt, turn marker or ACP-correlated ledger row."""
    return None if admission_enabled() else PAUSED


def refusal_before_admission(runner: Any, event: Any, source: Any, adapter: Any = None) -> Optional[str]:
    """The ordinary refusal owed to a managed message before anything else happens to it: the closed
    gate first, then an ``/acp`` with no task. None when it may go on to admission."""
    paused = paused_reply()
    if paused is not None:
        return paused
    return EMPTY_TASK if not managed_task_text(runner, event, source, adapter) else None


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
    turn: dict
    session_key: str
    lineage_root_digest: str


@dataclass
class Outcome:
    admission: Optional[Admission] = None
    reply: Optional[str] = None  # set when the turn must not run


async def admit(runner: Any, event: Any, source: Any, session_key: str, *, path: Optional[Path] = None,
                secret: Optional[str] = None, timeout: float = _ADMISSION_TIMEOUT_S) -> Outcome:
    """Ask ACP to admit this managed message and claim its receipt; never runs the turn itself."""
    refusal = refusal_before_admission(runner, event, source)
    if refusal is not None:
        return Outcome(reply=refusal)
    binding = bound_binding(runner, source)
    if session_key != binding.session_key:
        # The bound chat routed to another session (a profile route, a topic): the lineage ACP
        # would approve is not the session that would run, so the task is refused unasked.
        return Outcome(reply=REFUSED.format(reason="target mismatch"))
    task = managed_task_text(runner, event, source)
    if len(event.text or "") >= _MAX_TASK_MESSAGE_CHARS:
        return Outcome(reply=TOO_LONG)
    if launch_home_store(runner) is None:
        return Outcome(reply=REFUSED.format(reason="receipt store is not the launch home"))
    from gateway.delivery_ledger import ledger_enabled
    if not ledger_enabled():
        # The ledger row is the receipt's only evidence of an answer: without it, refuse unasked.
        return Outcome(reply=REFUSED.format(reason="delivery ledger disabled"))
    secret = secret if secret is not None else read_secret()
    envelope = build_envelope(event, source, secret) if secret and task else None
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
    return Outcome(admission=Admission(update_id=update_id, db=db, proof=proof, task_text=task,
                                       turn=turn, session_key=binding.session_key,
                                       lineage_root_digest=expected_lineage))


def abort_claimed(admission: Optional[Admission]) -> None:
    """The receipt was claimed but the turn did not start (a gate after admission refused it)."""
    if admission is not None:
        receipts.settle_aborted(admission.db, admission.update_id,
                                reason_code="REFUSED_BEFORE_RUN", **admission.proof)


def abort_unrecorded(admission: Optional[Admission]) -> bool:
    """The answer was withheld because its obligation could not be recorded. That is a definite
    non-delivery only when no earlier send of this update was ever recorded: an earlier attempt that
    failed after submission may have reached the chat, so its receipt stays PENDING (in doubt), and so
    does one whose ledger cannot be read. Returns True when the receipt was aborted."""
    if admission is None:
        return False
    import sqlite3

    try:
        rows = admission.db._read_all(
            "SELECT 1 FROM delivery_obligations WHERE acp_update_id = ? LIMIT 1",
            (str(admission.update_id),))
    except sqlite3.OperationalError as exc:
        # The binding store is the ledger's own file (launch_home_store), so a ledger table that does
        # not exist yet has recorded nothing; any other failure leaves the receipt in doubt.
        if "no such table" not in str(exc):
            logger.warning("ACP update %s: ledger unreadable, receipt left in doubt", admission.update_id)
            return False
        rows = []
    except Exception:
        logger.warning("ACP update %s: ledger unreadable, receipt left in doubt", admission.update_id)
        return False
    if rows:
        return False
    return bool(receipts.settle_aborted(admission.db, admission.update_id,
                                        reason_code="HERMES_ANSWER_NOT_RECORDED", **admission.proof))


# A crash-left managed turn is recognised by its durable active-turn marker rather than message time:
# the persisted user row carries Telegram's authored time, which precedes the marker (round-2
# ROUND1-ESCAPE-01). The marker token itself stays out of stored state; only its digest is kept, in the
# launch-home store beside the receipt, written before the turn runs.
def abort_failed_turn(admission: Optional[Admission]) -> bool:
    """The turn raised: abort the receipt HERMES_TURN_FAILED only when no answer of this update was
    ever recorded (an earlier recorded send keeps it in doubt). True when aborted."""
    if admission is None or _recorded_answers(admission) != 0:
        return False
    return bool(receipts.settle_aborted(admission.db, admission.update_id,
                                        reason_code="HERMES_TURN_FAILED", **admission.proof))


def _recorded_answers(admission: Admission) -> Optional[int]:
    """How many ledger rows answer this update; None when the ledger cannot be read."""
    import sqlite3

    try:
        return len(admission.db._read_all(
            "SELECT 1 FROM delivery_obligations WHERE acp_update_id = ? LIMIT 1", (str(admission.update_id),)))
    except sqlite3.OperationalError as exc:
        return 0 if "no such table" in str(exc) else None
    except Exception:
        return None


_TURN_MARKER_PREFIX = "acp_turn_marker:"


def _turn_marker_key(token: str) -> str:
    import hashlib

    return _TURN_MARKER_PREFIX + hashlib.sha256(token.encode("utf-8")).hexdigest()


def bind_turn_marker(admission: Admission, token: Optional[str]) -> bool:
    """Record that the active-turn marker *token* runs this admitted task. False means the binding is
    not durable, and the caller must not run the turn."""
    if not token:
        return False
    try:
        return bool(admission.db.claim_meta_once(_turn_marker_key(token), str(admission.update_id),
                                                 **admission.proof))
    except Exception:
        logger.warning("ACP update %s: turn marker not recorded", admission.update_id, exc_info=True)
        return False


_FOLLOWUP_PREFIX = "followup:"


def mark_followup_start(admission: Admission, token: Optional[str], started_at: float) -> bool:
    """Record, under the managed marker, the time an ordinary follow-up starts in this chain."""
    if not token:
        return False
    try:
        key = _turn_marker_key(token)
        current = admission.db.get_meta(key)
        if current is None:
            return False
        return bool(admission.db.compare_and_set_meta(key, current, f"{_FOLLOWUP_PREFIX}{float(started_at)}",
                                                      **admission.proof))
    except Exception:
        logger.warning("ACP update %s: follow-up start not recorded", admission.update_id, exc_info=True)
        return False


def followup_start(runner: Any, session_key: str, token: str) -> Optional[float]:
    """The follow-up start recorded under a managed marker, or None when the chain ran none."""
    db = runner.session_store._db_for_key(session_key)
    value = db.get_meta(_turn_marker_key(token)) if db is not None else None
    if value and value.startswith(_FOLLOWUP_PREFIX):
        return float(value[len(_FOLLOWUP_PREFIX):])
    return None


def crash_left_turn_is_managed(runner: Any, session_key: str, token: str) -> Optional[bool]:
    """Whether the turn the dead process left marked on *session_key* ran an admitted /acp task:
    True when its marker was bound, False when it was not, None when that cannot be read. Read from the session's own store, independent of the current binding configuration: the
    marker was written there before the turn ran, and an edited or removed binding must not turn an
    admitted task back into an ordinary one."""
    if not token:
        return False
    try:
        db = runner.session_store._db_for_key(session_key)
        if db is None:
            return None
        value = db.get_meta(_turn_marker_key(token))
    except Exception:
        logger.warning("Crash-left turn on %s: marker store unreadable", session_key, exc_info=True)
        return None
    return value is not None


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


def target_still_matches(admission: Admission, session_key: Optional[str], session_db: Any,
                         session_id: Optional[str]) -> bool:
    """Recheck under the turn's slot and lease, with the executing agent resolved: the session that
    will run is the binding's, and its lineage root is the one ACP approved."""
    if session_key != admission.session_key or not session_id or session_db is None:
        return False
    return _binding_lineage_digest(session_db, session_id) == admission.lineage_root_digest


def settle_delivered(runner: Any, obligation_id: str) -> bool:
    """A ledgered answer was delivered (live send or redelivery): settle the receipt of the /acp
    update its ledger row names, from that row's content. Not an /acp answer: nothing to do."""
    db = _binding_db(runner)
    if db is None:
        return False
    rows = db._read_all("SELECT acp_update_id FROM delivery_obligations WHERE obligation_id = ?",
                        (obligation_id,))
    update_id = rows[0][0] if rows else None
    if update_id is None:
        return False
    return receipts.settle_from_ledger(
        db, update_id, obligation_id=obligation_id,
        proven_db_path=db.db_path, proven_db_identity=db._db_file_identity)


def _binding_db(runner: Any) -> Any:
    return launch_home_store(runner)


def sweep_at_startup(runner: Any) -> list:
    """H3: abort PENDING receipts claimed by a process that is no longer this one."""
    db = _binding_db(runner)
    if db is None:
        return []
    return receipts.sweep_dead_owner_receipts(
        db, resuming_owners=(process_owner(),),
        proven_db_path=db.db_path, proven_db_identity=db._db_file_identity)
