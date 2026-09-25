"""Canonical resolution stays on an existing live Telegram session only."""

from datetime import datetime, timezone
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any

import pytest

from gateway.config import Platform
from gateway.session import SessionEntry, SessionSource, SessionStore


def _binding():
    from gateway.canonical_surface import CanonicalSurfaceBinding

    return CanonicalSurfaceBinding(
        name="synthetic-binding",
        session_key="agent:main:telegram:dm:synthetic-chat",
        session_id="synthetic-ended-predecessor",
        telegram_chat_id="synthetic-chat",
        telegram_chat_type="dm",
        telegram_user_id="synthetic-user",
        telegram_thread_id=None,
        allowed_author_ids=("synthetic-author",),
        allowed_channel_ids=("synthetic-channel",),
    )


def _entry(*, platform=Platform.TELEGRAM, chat_id="synthetic-chat"):
    return SimpleNamespace(
        session_id="synthetic-live-head",
        platform=platform,
        origin=SimpleNamespace(
            chat_id=chat_id,
            chat_type="dm",
            user_id="synthetic-user",
            thread_id=None,
        ),
    )


class _Store:
    def __init__(self, entry, row, tip):
        self.entry, self.row, self.tip = entry, row, tip
        self.looked_up = []
        self._db = self

    def lookup_by_session_key(self, key):
        self.looked_up.append(key)
        return self.entry

    def lookup_by_session_key_existing(self, key):
        self.looked_up.append(key)
        return self.entry

    def _should_reset(self, entry, origin):
        return None

    def get_session(self, session_id):
        assert session_id == self.entry.session_id
        return self.row

    def get_compression_tip(self, session_id):
        assert session_id == self.entry.session_id
        return self.tip


def test_resolver_follows_live_key_head_without_creating_healing_or_rotating():
    from gateway.canonical_surface import ExistingCanonicalBindingResolver

    binding = _binding()
    store = _Store(_entry(), {"ended_at": None}, "synthetic-live-head")
    event = SimpleNamespace(author_id="synthetic-author", channel_id="synthetic-channel")

    assert ExistingCanonicalBindingResolver(store).resolve(binding, event) is store.entry
    assert store.looked_up == [binding.session_key, binding.session_key]


def test_resolver_rejects_old_live_head_replaced_during_db_validation():
    """The final existing-only read fences a route replacement after old-row validation."""
    from gateway.canonical_surface import ExistingCanonicalBindingResolver

    binding = _binding()
    old_entry = _entry()
    replacement = _entry()
    replacement.session_id = "synthetic-replacement-head"
    store = _Store(old_entry, {"ended_at": None}, old_entry.session_id)

    def validate_old_tip_then_replace(session_id):
        assert session_id == old_entry.session_id
        store.entry = replacement
        return old_entry.session_id

    store.get_compression_tip = validate_old_tip_then_replace
    event = SimpleNamespace(author_id="synthetic-author", channel_id="synthetic-channel")

    with pytest.raises(ValueError, match="canonical_binding_stale"):
        ExistingCanonicalBindingResolver(store).resolve(binding, event)

    assert store.entry is replacement
    assert store.looked_up == [binding.session_key, binding.session_key]


@pytest.mark.parametrize(
    ("entry", "row", "tip", "author", "channel"),
    [
        (_entry(chat_id="foreign-chat"), {"ended_at": None}, "synthetic-live-head", "synthetic-author", "synthetic-channel"),
        (_entry(), {"ended_at": "synthetic-ended"}, "synthetic-live-head", "synthetic-author", "synthetic-channel"),
        (_entry(), {"ended_at": None}, "synthetic-compression-parent", "synthetic-author", "synthetic-channel"),
        (_entry(), {"ended_at": None}, "synthetic-live-head", "foreign-author", "synthetic-channel"),
        (_entry(), {"ended_at": None}, "synthetic-live-head", "synthetic-author", "foreign-channel"),
    ],
    ids=["foreign-origin", "ended", "invalid-compression-tip", "foreign-author", "foreign-channel"],
)
def test_resolver_refuses_foreign_or_non_live_binding(entry, row, tip, author, channel):
    from gateway.canonical_surface import ExistingCanonicalBindingResolver

    binding = _binding()
    store = _Store(entry, row, tip)
    event = SimpleNamespace(author_id=author, channel_id=channel)

    with pytest.raises(ValueError, match="canonical_(principal_rejected|binding_stale)"):
        ExistingCanonicalBindingResolver(store).resolve(binding, event)
    assert store.looked_up in ([], [binding.session_key])


@pytest.mark.parametrize("loaded", [False, True], ids=["not-loaded", "stale-head"])
def test_resolver_never_loads_or_repairs_unavailable_canonical_head(tmp_path, monkeypatch, loaded):
    """Canonical lookup is a pure, existing-entry read even before store initialization."""
    from gateway.canonical_surface import ExistingCanonicalBindingResolver

    store = object.__new__(SessionStore)
    store._lock = Lock()
    store._loaded = loaded
    store.__dict__["_entries"] = (
        {} if not loaded else {_binding().session_key: _entry(chat_id="foreign-chat")}
    )
    store.sessions_dir = tmp_path / "sessions"

    def unexpected(*_args, **_kwargs):
        raise AssertionError("canonical resolution must not load, recover, mkdir, or persist routing")

    for name in (
        "_ensure_loaded_locked",
        "_prune_stale_sessions_locked",
        "_route_recover",
        "_save_entries",
        "_save_entry",
    ):
        monkeypatch.setattr(store, name, unexpected, raising=False)
    monkeypatch.setattr(type(store.sessions_dir), "mkdir", unexpected)

    with pytest.raises(ValueError, match="canonical_binding_stale"):
        ExistingCanonicalBindingResolver(store).resolve(
            _binding(), SimpleNamespace(author_id="synthetic-author", channel_id="synthetic-channel")
        )
    assert not store.sessions_dir.exists()


@pytest.mark.parametrize("failure", ["reset", "session-row", "compression-tip"])
def test_resolver_rejects_reset_ineligible_or_exceptional_live_head(failure):
    from gateway.canonical_surface import ExistingCanonicalBindingResolver

    store = _Store(_entry(), {"ended_at": None}, "synthetic-live-head")

    def legacy_lookup(*_args, **_kwargs):
        raise AssertionError("canonical resolution must use the non-loading accessor")

    store.lookup_by_session_key = legacy_lookup
    if failure == "reset":
        setattr(store, "_should_reset", lambda *_args: "expired")
    elif failure == "session-row":
        setattr(store, "get_session", lambda *_args: (_ for _ in ()).throw(RuntimeError("db unavailable")))
    else:
        setattr(store, "get_compression_tip", lambda *_args: (_ for _ in ()).throw(RuntimeError("db unavailable")))

    with pytest.raises(ValueError, match="canonical_binding_stale"):
        ExistingCanonicalBindingResolver(store).resolve(
            _binding(), SimpleNamespace(author_id="synthetic-author", channel_id="synthetic-channel")
        )


def test_resolver_does_not_create_missing_named_profile_store(tmp_path, monkeypatch):
    """A configured but absent profile DB is not proof of a canonical live head."""
    from gateway.canonical_surface import ExistingCanonicalBindingResolver
    from gateway.config import GatewayConfig

    root, profile_home = tmp_path / "root", tmp_path / "profiles" / "code"
    root.mkdir()
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    store = SessionStore(root / "sessions", GatewayConfig(multiplex_profiles=True))
    key = "agent:code:telegram:dm:synthetic-chat"
    now = datetime.now(timezone.utc)
    store._loaded = True
    store._entries[key] = SessionEntry(
        session_key=key,
        session_id="synthetic-live-head",
        created_at=now,
        updated_at=now,
        platform=Platform.TELEGRAM,
        origin=SessionSource(
            platform=Platform.TELEGRAM, chat_id="synthetic-chat", chat_type="dm", user_id="synthetic-user"
        ),
    )
    monkeypatch.setattr(store, "_profile_home_for_key", lambda _key: profile_home)
    binding = _binding().__class__(**{**_binding().__dict__, "session_key": key})
    target = profile_home / "state.db"
    try:
        assert not target.exists()
        with pytest.raises(ValueError, match="canonical_binding_stale"):
            ExistingCanonicalBindingResolver(store).resolve(
                binding, SimpleNamespace(author_id="synthetic-author", channel_id="synthetic-channel")
            )
        assert not target.exists()
        assert not (profile_home / "state.db-wal").exists()
        assert not (profile_home / "state.db-shm").exists()
    finally:
        store.close_all_db_handles()


def test_resolver_reads_real_head_without_flushing_queued_token_writer(tmp_path, monkeypatch):
    """Canonical validation uses its own read-only DB beside a queued live writer."""
    from gateway.canonical_surface import ExistingCanonicalBindingResolver
    from gateway.config import GatewayConfig

    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    # conftest pins this import-time default to its own sandbox; keep the real
    # queued writer and resolver read handle together under this test's root.
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", root / "state.db")
    store = SessionStore(root / "sessions", GatewayConfig())
    db = store._db
    entry = _entry()
    now = datetime.now(timezone.utc)
    store._loaded = True
    store._entries[_binding().session_key] = SessionEntry(
        session_key=_binding().session_key,
        session_id=entry.session_id,
        created_at=now,
        updated_at=now,
        platform=entry.platform,
        origin=SessionSource(
            platform=Platform.TELEGRAM, chat_id="synthetic-chat", chat_type="dm", user_id="synthetic-user"
        ),
    )
    db.create_session(entry.session_id, "telegram")
    started, release = Event(), Event()
    original_update, original_flush = db.update_token_counts, db.flush_token_counts

    def blocked_update(*args, **kwargs):
        started.set()
        assert release.wait(timeout=10)
        return original_update(*args, **kwargs)

    def unexpected_flush(*_args, **_kwargs):
        raise AssertionError("canonical resolution must not flush the live writer token queue")

    db.update_token_counts = blocked_update
    try:
        db.queue_token_counts(entry.session_id, input_tokens=1, api_call_count=1)
        assert started.wait(timeout=10)
        db.flush_token_counts = unexpected_flush
        monkeypatch.setattr(
            store, "_db_for_key", lambda *_args: (_ for _ in ()).throw(AssertionError("must not be called"))
        )

        assert ExistingCanonicalBindingResolver(store).resolve(
            _binding(), SimpleNamespace(author_id="synthetic-author", channel_id="synthetic-channel")
        ) is store._entries[_binding().session_key]
    finally:
        db.flush_token_counts = original_flush
        db.update_token_counts = original_update
        release.set()
        store.close_all_db_handles()


def test_resolver_proof_captures_the_validated_live_head_and_db_identity(tmp_path, monkeypatch):
    """A future claimant can re-check the exact head and physical DB the resolver validated."""
    from gateway.canonical_surface import ExistingCanonicalBindingResolver
    from gateway.config import GatewayConfig

    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", root / "state.db")
    store = SessionStore(root / "sessions", GatewayConfig())
    now = datetime.now(timezone.utc)
    binding = _binding()
    entry = SessionEntry(
        session_key=binding.session_key,
        session_id="synthetic-live-head",
        created_at=now,
        updated_at=now,
        platform=Platform.TELEGRAM,
        origin=SessionSource(
            platform=Platform.TELEGRAM, chat_id="synthetic-chat", chat_type="dm", user_id="synthetic-user"
        ),
    )
    store._loaded = True
    store._entries[binding.session_key] = entry
    db: Any = store._db
    db.create_session(entry.session_id, "telegram")
    event = SimpleNamespace(author_id="synthetic-author", channel_id="synthetic-channel")
    expected_identity = (db.db_path.stat().st_dev, db.db_path.stat().st_ino)

    try:
        resolver = ExistingCanonicalBindingResolver(store)
        # The established entry-only API stays compatible for existing callers.
        assert resolver.resolve(binding, event) is entry

        resolved, proof = resolver.resolve_with_proof(binding, event)

        assert resolved is entry
        assert proof.entry is entry
        assert proof.session_key == binding.session_key
        assert proof.session_id == entry.session_id
        assert proof.db_path == db.db_path
        assert proof.db_identity == expected_identity
    finally:
        store.close_all_db_handles()


@pytest.mark.parametrize("pinned_db", [None, object()], ids=["disabled", "pathless"])
def test_resolver_rejects_explicit_non_path_pins_without_reading_existing_root_db(tmp_path, monkeypatch, pinned_db):
    """An explicit pin never authorizes an ambient root DB fallback."""
    from gateway.canonical_surface import ExistingCanonicalBindingResolver
    from gateway.config import GatewayConfig

    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", root / "state.db")
    store = SessionStore(root / "sessions", GatewayConfig())
    entry = _entry()
    now = datetime.now(timezone.utc)
    store._loaded = True
    store._entries[_binding().session_key] = SessionEntry(
        session_key=_binding().session_key,
        session_id=entry.session_id,
        created_at=now,
        updated_at=now,
        platform=entry.platform,
        origin=SessionSource(
            platform=Platform.TELEGRAM, chat_id="synthetic-chat", chat_type="dm", user_id="synthetic-user"
        ),
    )
    root_db_handle = store._db
    getattr(root_db_handle, "create_session")(entry.session_id, "telegram")
    root_db = root / "state.db"
    assert root_db.is_file()
    store._db = pinned_db
    opened_paths = []

    def unexpected_root_read(path, *args, **kwargs):
        opened_paths.append(path)
        raise AssertionError("explicit pinned stores must not read the ambient root DB")

    monkeypatch.setattr("hermes_state.SessionDB", unexpected_root_read)
    try:
        with pytest.raises(ValueError, match="canonical_binding_stale"):
            ExistingCanonicalBindingResolver(store).resolve(
                _binding(), SimpleNamespace(author_id="synthetic-author", channel_id="synthetic-channel")
            )
        assert opened_paths == []
        assert set(store._entries) == {_binding().session_key}
        assert root_db.is_file()
    finally:
        store.close_all_db_handles()
