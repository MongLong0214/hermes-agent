"""Dormant adapter for durable terminal turn receipts.

The adapter deliberately receives an existing :class:`hermes_state.SessionDB`.
It does not open another database, run transaction statements, or connect any
prompt/model/finalizer path.  SessionDB remains the one state authority and
its canonical transcript write is the only completion boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import uuid
from typing import Any, Optional

from hermes_state import SessionDB


@dataclass(frozen=True)
class ReceiptRequest:
    """Immutable, invocation-local identity for an opt-in turn."""

    session_id: str
    turn_request_id: str
    binding_digest: str


@dataclass(frozen=True)
class ClaimedReceipt:
    request: ReceiptRequest
    claim_token: str


@dataclass(frozen=True)
class TerminalReceiptHold:
    """The exact assistant object selected after every continuation gate."""

    claimed: ClaimedReceipt
    terminal_message: dict[str, Any]
    terminal_message_index: int
    response_digest: str | None = None


def request_binding(
    *,
    session_id: str,
    turn_request_id: str,
    text: Any,
    display_kind: str | None,
    attachments: list[str] | None,
    truncation: dict[str, Any] | None,
) -> ReceiptRequest:
    """Hash only server-derived, canonical request facts.

    Attachment paths are gateway-owned session files.  Their current content
    is folded into the binding as well, so a path re-used with different bytes
    cannot replay a previous turn.
    """
    attachment_binding = []
    for path in attachments or []:
        path_text = str(path)
        try:
            with open(path_text, "rb") as attachment_file:
                digest = "sha256:" + hashlib.sha256(attachment_file.read()).hexdigest()
        except OSError:
            # The effective attachment is an unavailable path, not a client
            # supplied claim about what used to live there.
            digest = "unavailable"
        attachment_binding.append({"path": path_text, "content_digest": digest})
    binding = {
        "version": 1,
        "session_id": str(session_id),
        "turn_request_id": str(turn_request_id),
        "text": text if isinstance(text, str) else None,
        "display_kind": display_kind or None,
        "attachments": attachment_binding,
        "truncation": truncation or None,
    }
    encoded = json.dumps(
        binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return ReceiptRequest(
        session_id=str(session_id),
        turn_request_id=str(turn_request_id),
        binding_digest="sha256:" + hashlib.sha256(encoded).hexdigest(),
    )


class TurnReceiptAdapter:
    """The sole gateway-facing prepare/get/status/claim/finish/prune surface."""

    def __init__(self, db: SessionDB):
        self._db = db

    def migration(self) -> int:
        """Converge a pre-receipt canonical state DB through SessionDB."""
        return self._db.migrate_turn_receipts()

    def prepare(
        self, session_id: str, turn_request_id: str, binding_digest: str
    ) -> dict[str, Any]:
        """Create (or idempotently retrieve) a PREPARED receipt."""
        return self._db.prepare_turn_receipt(
            session_id, turn_request_id, binding_digest
        )

    def get(
        self, session_id: str, turn_request_id: str, binding_digest: str
    ) -> Optional[dict[str, Any]]:
        """Get a receipt scoped to its immutable durable request binding."""
        return self._db.get_turn_receipt(
            session_id, turn_request_id, binding_digest
        )

    def status(
        self, session_id: str, turn_request_id: str, binding_digest: str
    ) -> Optional[dict[str, Any]]:
        """Alias for the stable status lookup used by later protocol wiring."""
        return self.get(session_id, turn_request_id, binding_digest)

    def prepare_or_replay(self, request: ReceiptRequest) -> dict[str, Any]:
        return self.prepare(
            request.session_id, request.turn_request_id, request.binding_digest
        )

    def status_for(self, request: ReceiptRequest) -> Optional[dict[str, Any]]:
        return self.status(
            request.session_id, request.turn_request_id, request.binding_digest
        )

    def completed_replay(self, request: ReceiptRequest) -> Optional[dict[str, Any]]:
        """Return the public receipt plus exact stored assistant bytes."""
        receipt = self.status_for(request)
        if not receipt or receipt.get("status") != "COMPLETED":
            return None
        terminal_id = receipt.get("terminalMessageId")
        for message in self._db.get_messages(request.session_id):
            if message.get("id") == terminal_id:
                content = message.get("content")
                if isinstance(content, str):
                    return {**receipt, "assistantContent": content}
        # A completed receipt without its committed terminal row is corrupt;
        # never substitute a tail assistant from the current transcript.
        raise RuntimeError("completed turn receipt terminal message is missing")

    def claim(
        self,
        session_id: str,
        turn_request_id: str,
        binding_digest: str,
        *,
        claim_token: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        """Claim a prepared receipt once and return the token on success."""
        token = claim_token or uuid.uuid4().hex
        if not self._db.claim_turn_receipt(
            session_id, turn_request_id, binding_digest, token
        ):
            return None, self.get(session_id, turn_request_id, binding_digest)
        return token, self.get(session_id, turn_request_id, binding_digest)

    def claim_after_lease(
        self, request: ReceiptRequest
    ) -> ClaimedReceipt | dict[str, Any] | None:
        """Claim only from PREPARED; callers branch on replay/in-progress."""
        token, receipt = self.claim(
            request.session_id, request.turn_request_id, request.binding_digest
        )
        if token:
            return ClaimedReceipt(request, token)
        return receipt

    def finish(
        self,
        session_id: str,
        turn_request_id: str,
        binding_digest: str,
        claim_token: str,
        *,
        assistant_content: str,
        response_digest: str,
    ) -> dict[str, Any]:
        """Commit the final assistant row and COMPLETED receipt as one unit."""
        return self._db.finish_turn_receipt(
            session_id,
            turn_request_id,
            binding_digest,
            claim_token,
            assistant_content=assistant_content,
            response_digest=response_digest,
        )

    def prune(self, session_id: str, *, completed_before: float) -> int:
        """Retire old completed receipts without deleting unresolved evidence."""
        return self._db.prune_turn_receipts(session_id, completed_before)
