"""Dormant adapter for durable terminal turn receipts.

The adapter deliberately receives an existing :class:`hermes_state.SessionDB`.
It does not open another database, run transaction statements, or connect any
prompt/model/finalizer path.  SessionDB remains the one state authority and
its canonical transcript write is the only completion boundary.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from hermes_state import SessionDB


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
