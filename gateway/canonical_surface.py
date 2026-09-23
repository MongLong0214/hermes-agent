"""Pure admission checks for configured canonical bindings.

This module deliberately does not receive ingress, create sessions, or deliver replies.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from gateway.config import Platform
from gateway.session_persistence import _DB_UNPINNED


_EVENT_FIELDS = frozenset({"binding", "event_id", "author_id", "channel_id", "text"})
_MAX_ID_CHARS = 256
_MAX_TEXT_CHARS = 16_384


def _required_text(value: Any, *, limit: int) -> str:
    """Validate text without normalizing caller-provided event identity."""

    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("canonical_invalid_request")
    return value


@dataclass(frozen=True)
class CanonicalTurnResult:
    """The terminal selected from one canonical turn, before request readback."""

    binding_name: str
    terminal_text: str


class RequestLocalReplySink(Protocol):
    """Opaque request-scoped capability for accepting exactly one terminal."""

    async def publish(self, result: CanonicalTurnResult) -> None: ...


class _RequestReplySink:
    """One-use sink whose publisher remains private to the creating request."""

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
    """Create a reply capability that cannot select or retain a destination."""

    return _RequestReplySink(publisher)


def require_request_local_reply_sink(value: Any) -> RequestLocalReplySink:
    """Reject caller-supplied lookalikes in the existing-actor turn path."""

    if not isinstance(value, _RequestReplySink):
        raise ValueError("canonical_reply_sink_missing")
    return value


@dataclass(frozen=True)
class CanonicalSurfaceBinding:
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
class CanonicalBindingProof:
    """The exact live head and physical DB generation a resolver just validated.

    This is evidence for a later existing-actor claimant to re-check. It does
    not reserve the route or make a subsequent turn safe by itself.
    """

    entry: Any
    session_key: str
    session_id: str
    db_path: Path
    db_identity: tuple[int, int]


@dataclass(frozen=True)
class CanonicalIngressEvent:
    """Closed canonical ingress payload with no caller-controlled destination."""

    binding: str
    event_id: str
    author_id: str
    channel_id: str
    text: str

    @classmethod
    def from_json_bytes(cls, raw: bytes) -> "CanonicalIngressEvent":
        """Parse the exact canonical event object, rejecting ambiguity at ingress."""

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
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
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
    """Resolve a binding only to its current, already-live session-key head."""

    def __init__(self, session_store: Any) -> None:
        self._session_store = session_store

    def _existing_db_path(self, session_key: str) -> Path | None:
        """Prove the owning state.db exists without opening a writable store handle."""
        store = self._session_store
        try:
            pinned_db_for_store = getattr(store, "_pinned_db", None)
            if callable(pinned_db_for_store):
                pinned_db = pinned_db_for_store()
                if pinned_db is not _DB_UNPINNED:
                    pinned_path = getattr(pinned_db, "db_path", None)
                    if pinned_path is None:
                        return None
                    path = Path(pinned_path)
                    return path if path.is_file() else None

            named_profile = getattr(store, "_named_profile_for_key", None)
            if callable(named_profile) and named_profile(session_key) is not None:
                profile_home = getattr(store, "_profile_home_for_key", lambda _key: None)(session_key)
                if profile_home is None:
                    return None
                path = Path(profile_home) / "state.db"
                return path if path.is_file() else None

            routing_home = getattr(store, "_routing_home", None)
            if routing_home is not None:
                path = Path(routing_home) / "state.db"
                return path if path.is_file() else None
        except Exception:
            return None
        return None

    def _read_db_for_existing_path(
        self, session_key: str, *, existing_path: Path | None = None
    ) -> tuple[Any, bool] | tuple[None, bool]:
        """Return a separately-owned read-only DB, or the narrow fake-store compatibility DB."""
        path = existing_path if existing_path is not None else self._existing_db_path(session_key)
        if path is not None:
            from hermes_state import SessionDB

            return SessionDB(path, read_only=True), True
        # Production SessionStore instances provide this writer helper. Do not fall back to _db for
        # them: accessing it can create/open a writable database. Synthetic stores retain _db=fake.
        if callable(getattr(self._session_store, "_db_for_key", None)):
            return None, False
        return getattr(self._session_store, "_db", None), False

    def resolve(self, binding: CanonicalSurfaceBinding, event: Any) -> Any:
        """Return the existing entry exactly as before, without exposing a proof."""
        entry, _ = self._resolve(binding, event, require_proof=False)
        return entry

    def resolve_with_proof(
        self, binding: CanonicalSurfaceBinding, event: Any
    ) -> tuple[Any, CanonicalBindingProof]:
        """Return an existing live entry together with its frozen validation evidence."""
        entry, proof = self._resolve(binding, event, require_proof=True)
        assert proof is not None
        return entry, proof

    @staticmethod
    def _db_identity(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return stat.st_dev, stat.st_ino

    def _resolve(
        self, binding: CanonicalSurfaceBinding, event: Any, *, require_proof: bool
    ) -> tuple[Any, CanonicalBindingProof | None]:
        if (
            getattr(event, "author_id", None) not in binding.allowed_author_ids
            or getattr(event, "channel_id", None) not in binding.allowed_channel_ids
        ):
            raise ValueError("canonical_principal_rejected")

        entry = self._session_store.lookup_by_session_key_existing(binding.session_key)
        if entry is None or getattr(entry, "platform", None) != Platform.TELEGRAM:
            raise ValueError("canonical_binding_stale")
        origin = getattr(entry, "origin", None)
        if origin is None or (
            getattr(origin, "chat_id", None) != binding.telegram_chat_id
            or getattr(origin, "chat_type", None) != binding.telegram_chat_type
            or getattr(origin, "user_id", None) != binding.telegram_user_id
            or getattr(origin, "thread_id", None) != binding.telegram_thread_id
        ):
            raise ValueError("canonical_binding_stale")

        db = None
        close_db = False
        db_path = self._existing_db_path(binding.session_key) if require_proof else None
        db_identity = self._db_identity(db_path) if db_path is not None else None
        try:
            if self._session_store._should_reset(entry, origin) is not None:
                raise ValueError("canonical_binding_stale")
            db, close_db = self._read_db_for_existing_path(
                binding.session_key, existing_path=db_path
            )
            session_id = getattr(entry, "session_id", None)
            if db is None or not session_id:
                raise ValueError("canonical_binding_stale")
            row = db.get_session(session_id)
            if not row or row.get("ended_at") is not None or db.get_compression_tip(session_id) != session_id:
                raise ValueError("canonical_binding_stale")
            current_entry = self._session_store.lookup_by_session_key_existing(binding.session_key)
            if current_entry is not entry or getattr(current_entry, "session_id", None) != session_id:
                raise ValueError("canonical_binding_stale")
            if require_proof:
                if (
                    db_path is None
                    or db_identity is None
                    or self._db_identity(db_path) != db_identity
                ):
                    raise ValueError("canonical_binding_stale")
                proof: CanonicalBindingProof | None = CanonicalBindingProof(
                    entry=entry,
                    session_key=binding.session_key,
                    session_id=session_id,
                    db_path=db_path,
                    db_identity=db_identity,
                )
            else:
                proof = None
        except ValueError:
            raise ValueError("canonical_binding_stale")
        except Exception:
            raise ValueError("canonical_binding_stale") from None
        finally:
            if close_db and db is not None:
                try:
                    db.close()
                except Exception:
                    pass
        return entry, proof
