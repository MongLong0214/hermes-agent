"""ACP terminal receipts: the closed ``_meta.hermes`` request shape, and this build's answer to it.

A controller asks for an idempotent turn by attaching ``_meta.hermes.acpTerminalReceipt`` to
``session/prompt``. Answering needs a durable receipt store, and this build has none: it cannot tell
a retried ``execute`` from a new one, nor report a turn's status. So every prompt that carries
``_meta.hermes`` is refused before the session is looked up: no restore, no agent build, no model
call, no transcript write. An ordinary prompt without ``_meta.hermes`` is untouched.

The parser is the closed wire shape the store will admit. Until then it only names the refusal.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from acp.schema import PromptResponse

logger = logging.getLogger(__name__)

_METADATA_KEYS = frozenset({"operation", "receiptIdentity", "targetBindReceipt"})
_OPERATIONS = frozenset({"execute", "status"})
_TARGET_BIND_WIRE_KEYS = frozenset(
    {
        "domain",
        "version",
        "actor_id",
        "binding_generation",
        "executor_runtime_identity",
        "requested_session_id",
        "lineage_root_digest",
        "receipt_digest",
    }
)
_TARGET_BIND_DOMAIN = "hermes.target-bind"
_TARGET_BIND_VERSION = 1
_TARGET_BIND_RECEIPT_SCHEMA = "hermes.target-bind-receipt"
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
# The controller holds integers as JS numbers.
_MAX_SAFE_INTEGER = 2**53 - 1


@dataclass(frozen=True)
class TerminalReceiptRequest:
    """One closed ``acpTerminalReceipt`` request; ``target_bind_receipt`` is in the internal form."""

    operation: str
    receipt_identity: dict[str, Any]
    target_bind_receipt: dict[str, Any]


def terminal_receipt_refusal() -> PromptResponse:
    """The one reply a controller reads as REFUSED: ``stopReason`` plus a ``{status}``-only receipt."""
    return PromptResponse(
        stop_reason="refusal",
        field_meta={"hermes": {"acpTerminalReceipt": {"status": "REFUSED"}}},
    )


def _is_text(value: Any) -> bool:
    return type(value) is str and bool(value)


def _is_digest(value: Any) -> bool:
    return type(value) is str and _DIGEST_RE.fullmatch(value) is not None


def target_bind_receipt_from_wire(receipt: Any, session_id: str) -> dict[str, Any] | None:
    """Adapt one closed 8-key wire receipt for ``session_id`` to the internal form (plus ``schema``).

    Types are exact (``type() is``): ``True`` is an ``int`` to ``isinstance``, never a version or a
    generation. Whether the receipt is DURABLE is the store's question, not this shape check's.
    """
    if type(receipt) is not dict or set(receipt) != _TARGET_BIND_WIRE_KEYS:
        return None
    generation = receipt["binding_generation"]
    if (
        receipt["domain"] != _TARGET_BIND_DOMAIN
        or type(receipt["version"]) is not int
        or receipt["version"] != _TARGET_BIND_VERSION
        or type(generation) is not int
        or not 0 < generation <= _MAX_SAFE_INTEGER
        or not _is_text(receipt["actor_id"])
        or not _is_text(receipt["executor_runtime_identity"])
        or type(receipt["requested_session_id"]) is not str
        or receipt["requested_session_id"] != session_id
        or not _is_digest(receipt["lineage_root_digest"])
        or not _is_digest(receipt["receipt_digest"])
    ):
        return None
    return {"schema": _TARGET_BIND_RECEIPT_SCHEMA, **receipt}


def parse_terminal_receipt_request(hermes_metadata: Any, session_id: str) -> TerminalReceiptRequest | None:
    """Parse ``_meta.hermes`` as exactly ``{"acpTerminalReceipt": {operation, receiptIdentity,
    targetBindReceipt}}``; anything else, extra or missing keys included, is ``None``."""
    if type(hermes_metadata) is not dict or set(hermes_metadata) != {"acpTerminalReceipt"}:
        return None
    metadata = hermes_metadata["acpTerminalReceipt"]
    if type(metadata) is not dict or set(metadata) != _METADATA_KEYS:
        return None
    operation = metadata["operation"]
    receipt_identity = metadata["receiptIdentity"]
    target_bind_receipt = target_bind_receipt_from_wire(metadata["targetBindReceipt"], session_id)
    if (
        type(operation) is not str
        or operation not in _OPERATIONS
        or type(receipt_identity) is not dict
        or target_bind_receipt is None
    ):
        return None
    return TerminalReceiptRequest(operation, receipt_identity, target_bind_receipt)


def refuse_terminal_receipt_prompt(prompt_meta: dict[str, Any], session_id: str) -> PromptResponse | None:
    """The refusal for a prompt whose ``_meta`` carries a ``hermes`` key, else ``None``.

    ``prompt_meta`` is the handler's ``**kwargs``: the ACP router merges the request's ``_meta``
    entries into them. Presence of the key is the whole test, so a null, non-object or malformed
    value is refused exactly like a well-formed one.
    """
    if "hermes" not in prompt_meta:
        return None
    request = parse_terminal_receipt_request(prompt_meta["hermes"], session_id)
    logger.info(
        "Refused ACP prompt for session %s: %s",
        session_id,
        f"{request.operation} needs a terminal-receipt store" if request else "malformed _meta.hermes",
    )
    return terminal_receipt_refusal()
