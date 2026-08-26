"""Existing-only canonical HTTP ingress contracts for one bound agent turn."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.session import SessionEntry, SessionStore


_CANONICAL_EVENT_FIELDS = frozenset(
    {"binding", "event_id", "author_id", "channel_id", "text"}
)
_CANONICAL_MAX_ID_CHARS = 256
_CANONICAL_MAX_TEXT_CHARS = 64 * 1024


def _bounded_string(value: Any, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError("canonical_invalid_request")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum:
        raise ValueError("canonical_invalid_request")
    return cleaned


@dataclass(frozen=True)
class CanonicalSurfaceBinding:
    """Server-owned route generation and allowed Buzz principals."""

    name: str
    session_key: str
    session_id: str
    telegram_chat_id: str
    telegram_chat_type: str
    telegram_user_id: str | None
    telegram_thread_id: str | None
    allowed_author_ids: tuple[str, ...]
    allowed_channel_ids: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalIngressEvent:
    """Closed Buzz-shaped request body with no caller routing targets."""

    binding: str
    event_id: str
    author_id: str
    channel_id: str
    text: str

    @classmethod
    def from_payload(cls, payload: Any) -> "CanonicalIngressEvent":
        if not isinstance(payload, dict) or set(payload) != _CANONICAL_EVENT_FIELDS:
            raise ValueError("canonical_invalid_request")
        return cls(
            binding=_bounded_string(
                payload.get("binding"), maximum=_CANONICAL_MAX_ID_CHARS
            ),
            event_id=_bounded_string(
                payload.get("event_id"), maximum=_CANONICAL_MAX_ID_CHARS
            ),
            author_id=_bounded_string(
                payload.get("author_id"), maximum=_CANONICAL_MAX_ID_CHARS
            ),
            channel_id=_bounded_string(
                payload.get("channel_id"), maximum=_CANONICAL_MAX_ID_CHARS
            ),
            text=_bounded_string(
                payload.get("text"), maximum=_CANONICAL_MAX_TEXT_CHARS
            ),
        )

    @classmethod
    def from_json_bytes(cls, raw: bytes) -> "CanonicalIngressEvent":
        duplicate = False

        def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            nonlocal duplicate
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    duplicate = True
                result[key] = value
            return result

        try:
            payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_closed_object)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("canonical_invalid_request") from None
        if duplicate:
            raise ValueError("canonical_invalid_request")
        return cls.from_payload(payload)


class ExistingCanonicalBindingResolver:
    """Resolve one pinned Telegram routing generation without healing it."""

    def __init__(self, session_store: "SessionStore") -> None:
        self._session_store = session_store

    def resolve(
        self,
        binding: CanonicalSurfaceBinding,
        event: CanonicalIngressEvent,
    ) -> "SessionEntry":
        from gateway.config import Platform

        if event.author_id not in binding.allowed_author_ids:
            raise ValueError("canonical_principal_rejected")
        if event.channel_id not in binding.allowed_channel_ids:
            raise ValueError("canonical_principal_rejected")

        entry = self._session_store.lookup_by_session_key_existing(
            binding.session_key
        )
        if entry is None or entry.session_id != binding.session_id:
            raise ValueError("canonical_binding_stale")
        if entry.platform != Platform.TELEGRAM or entry.origin is None:
            raise ValueError("canonical_binding_stale")

        origin = entry.origin
        if (
            origin.platform != Platform.TELEGRAM
            or str(origin.chat_id) != binding.telegram_chat_id
            or str(origin.chat_type) != binding.telegram_chat_type
            or (str(origin.user_id) if origin.user_id is not None else None)
            != binding.telegram_user_id
            or (str(origin.thread_id) if origin.thread_id is not None else None)
            != binding.telegram_thread_id
        ):
            raise ValueError("canonical_binding_stale")

        try:
            reset_required = (
                self._session_store._should_reset(entry, origin) is not None
            )
        except Exception:
            raise ValueError("canonical_binding_stale") from None
        if reset_required:
            raise ValueError("canonical_binding_stale")

        db = self._session_store._db
        if db is None:
            raise ValueError("canonical_binding_stale")
        try:
            row: Mapping[str, Any] | None = db.get_session(binding.session_id)
            tip = db.get_compression_tip(binding.session_id)
        except Exception:
            raise ValueError("canonical_binding_stale") from None
        if row is None:
            raise ValueError("canonical_binding_stale")
        if row.get("ended_at") is not None or row.get("end_reason") is not None:
            raise ValueError("canonical_binding_stale")
        if (
            str(row.get("source") or "") != "telegram"
            or str(row.get("session_key") or "") != binding.session_key
            or str(row.get("chat_id") or "") != binding.telegram_chat_id
            or str(row.get("chat_type") or "") != binding.telegram_chat_type
            or (str(row.get("user_id")) if row.get("user_id") is not None else None)
            != binding.telegram_user_id
            or (str(row.get("thread_id")) if row.get("thread_id") is not None else None)
            != binding.telegram_thread_id
            or tip != binding.session_id
        ):
            raise ValueError("canonical_binding_stale")
        return entry
