"""Closed, existing-only values for the canonical HTTP surface."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping, Protocol

if TYPE_CHECKING:
    from gateway.session import SessionEntry, SessionStore


_EVENT_FIELDS = frozenset({"binding", "event_id", "author_id", "channel_id", "text"})
_MAX_ID_CHARS = 256
_MAX_TEXT_CHARS = 16_384


def _required_text(value: Any, *, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError("canonical_invalid_request")
    value = value.strip()
    if not value or len(value) > limit:
        raise ValueError("canonical_invalid_request")
    return value


@dataclass(frozen=True)
class CanonicalTurnResult:
    """The non-routable terminal selected from one admitted turn."""

    binding_name: str
    terminal_text: str


class RequestLocalReplySink(Protocol):
    """Opaque capability for returning a terminal only to its own request."""

    async def publish(self, result: CanonicalTurnResult) -> None: ...


class _RequestReplySink:
    """One-use destination capability retained only by the active request."""

    def __init__(
        self, publisher: Callable[[CanonicalTurnResult], Awaitable[None]]
    ) -> None:
        self._publisher = publisher
        self._published = False
        self._lock = asyncio.Lock()

    async def publish(self, result: CanonicalTurnResult) -> None:
        async with self._lock:
            if self._published:
                raise ValueError("canonical_reply_already_published")
            self._published = True
            try:
                await self._publisher(result)
            except Exception:
                raise ValueError("canonical_reply_publish_failed") from None


def request_local_reply_sink(
    publisher: Callable[[CanonicalTurnResult], Awaitable[None]],
) -> RequestLocalReplySink:
    return _RequestReplySink(publisher)


def require_request_local_reply_sink(value: Any) -> RequestLocalReplySink:
    if not isinstance(value, _RequestReplySink):
        raise ValueError("canonical_reply_sink_missing")
    return value


@dataclass(frozen=True)
class CanonicalSurfaceBinding:
    """A server-owned existing gateway identity and admitted principals."""

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
    """Closed source-admission payload with no caller destination fields."""

    binding: str
    event_id: str
    author_id: str
    channel_id: str
    text: str

    @classmethod
    def from_json_bytes(cls, raw: bytes) -> "CanonicalIngressEvent":
        duplicate = False

        def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            nonlocal duplicate
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    duplicate = True
                result[key] = value
            return result

        try:
            payload = json.loads(raw.decode("utf-8"), object_pairs_hook=closed_object)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("canonical_invalid_request") from None
        if duplicate or not isinstance(payload, dict) or set(payload) != _EVENT_FIELDS:
            raise ValueError("canonical_invalid_request")
        return cls(
            binding=_required_text(payload["binding"], limit=_MAX_ID_CHARS),
            event_id=_required_text(payload["event_id"], limit=_MAX_ID_CHARS),
            author_id=_required_text(payload["author_id"], limit=_MAX_ID_CHARS),
            channel_id=_required_text(payload["channel_id"], limit=_MAX_ID_CHARS),
            text=_required_text(payload["text"], limit=_MAX_TEXT_CHARS),
        )


class ExistingCanonicalBindingResolver:
    """Read an exact configured binding without creating, healing, or rotating it."""

    def __init__(self, session_store: "SessionStore") -> None:
        self._session_store = session_store

    def resolve(
        self, binding: CanonicalSurfaceBinding, event: CanonicalIngressEvent
    ) -> "SessionEntry":
        from gateway.config import Platform

        if (
            event.author_id not in binding.allowed_author_ids
            or event.channel_id not in binding.allowed_channel_ids
        ):
            raise ValueError("canonical_principal_rejected")
        entry = self._session_store.lookup_by_session_key_existing(binding.session_key)
        if entry is None or entry.session_id != binding.session_id:
            raise ValueError("canonical_binding_stale")
        if entry.platform != Platform.TELEGRAM or entry.origin is None:
            raise ValueError("canonical_binding_stale")
        origin = entry.origin
        if (
            str(origin.chat_id) != binding.telegram_chat_id
            or str(origin.chat_type) != binding.telegram_chat_type
            or (str(origin.user_id) if origin.user_id is not None else None)
            != binding.telegram_user_id
            or (str(origin.thread_id) if origin.thread_id is not None else None)
            != binding.telegram_thread_id
        ):
            raise ValueError("canonical_binding_stale")
        try:
            if self._session_store._should_reset(entry, origin) is not None:
                raise ValueError("canonical_binding_stale")
            db = self._session_store._db
            if db is None:
                raise ValueError("canonical_binding_stale")
            row: Mapping[str, Any] | None = db.get_session(binding.session_id)
            if row is None or row.get("ended_at") is not None:
                raise ValueError("canonical_binding_stale")
            if db.get_compression_tip(binding.session_id) != binding.session_id:
                raise ValueError("canonical_binding_stale")
        except ValueError:
            raise
        except Exception:
            raise ValueError("canonical_binding_stale") from None
        return entry
