"""Pure admission checks for configured canonical bindings.

This module deliberately does not receive ingress, create sessions, or deliver replies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gateway.config import Platform
from gateway.session_persistence import _DB_UNPINNED


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

    def _read_db_for_existing_path(self, session_key: str) -> tuple[Any, bool] | tuple[None, bool]:
        """Return a separately-owned read-only DB, or the narrow fake-store compatibility DB."""
        path = self._existing_db_path(session_key)
        if path is not None:
            from hermes_state import SessionDB

            return SessionDB(path, read_only=True), True
        # Production SessionStore instances provide this writer helper. Do not fall back to _db for
        # them: accessing it can create/open a writable database. Synthetic stores retain _db=fake.
        if callable(getattr(self._session_store, "_db_for_key", None)):
            return None, False
        return getattr(self._session_store, "_db", None), False

    def resolve(self, binding: CanonicalSurfaceBinding, event: Any) -> Any:
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
        try:
            if self._session_store._should_reset(entry, origin) is not None:
                raise ValueError("canonical_binding_stale")
            db, close_db = self._read_db_for_existing_path(binding.session_key)
            session_id = getattr(entry, "session_id", None)
            if db is None or not session_id:
                raise ValueError("canonical_binding_stale")
            row = db.get_session(session_id)
            if not row or row.get("ended_at") is not None or db.get_compression_tip(session_id) != session_id:
                raise ValueError("canonical_binding_stale")
            current_entry = self._session_store.lookup_by_session_key_existing(binding.session_key)
            if current_entry is not entry or getattr(current_entry, "session_id", None) != session_id:
                raise ValueError("canonical_binding_stale")
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
        return entry
