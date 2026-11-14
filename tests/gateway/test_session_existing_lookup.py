"""``SessionStore.lookup_by_session_key_existing``: the existing route for a key, as it stands.

It never creates, prunes, reopens, repoints or saves anything. A store nothing has loaded yet reads its
routing index first without recovery, so the first canonical event after a clean restart still finds
its binding; the startup prune (and the mirror rewrite a first load owes) then runs on the first
ordinary load, exactly once. The ordinary first load is unchanged: it heals missing origins and
rewrites the mirror as before.
"""

from __future__ import annotations

import json

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


def _store(tmp_path, db) -> SessionStore:
    store = SessionStore(config=GatewayConfig(), sessions_dir=tmp_path / "sessions")
    store._db = db
    return store


def _session_rows(db):
    with db._lock:
        return db._conn.execute("SELECT id, ended_at FROM sessions ORDER BY id").fetchall()


def _spy(monkeypatch, store, name):
    calls = []
    real = getattr(type(store), name)
    monkeypatch.setattr(type(store), name, lambda self, *a, **k: calls.append(name) or real(self, *a, **k))
    return calls


@pytest.fixture
def routed(tmp_path):
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "state.db")
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user")
    entry = _store(tmp_path, db).get_or_create_session(source)
    try:
        yield db, source, entry
    finally:
        db.close()


def test_lookup_by_session_key_existing_never_creates_a_session(tmp_path, routed, monkeypatch):
    db, _source, entry = routed
    rows = _session_rows(db)
    for loaded in (False, True):
        store = _store(tmp_path, db)
        if loaded:
            store._ensure_loaded()
        created = []
        monkeypatch.setattr(store, "get_or_create_session", lambda *a, **k: created.append(a))
        saves = _spy(monkeypatch, store, "_save")
        before = dict(store._entries)
        for key in (entry.session_key + ":ghost", "", None):
            assert store.lookup_by_session_key_existing(key) is None
        assert created == [] and saves == []
        assert set(store._entries) == (set(before) if loaded else {entry.session_key})
        assert _session_rows(db) == rows


def test_an_unreadable_index_answers_none_and_creates_nothing(tmp_path, routed, monkeypatch):
    db, _source, entry = routed
    rows = _session_rows(db)
    store = _store(tmp_path, db)

    def broken(self):
        raise RuntimeError("index unreadable")
    monkeypatch.setattr(type(store), "_read_routing_index_locked", broken)
    assert store.lookup_by_session_key_existing(entry.session_key) is None
    assert store._entries == {} and _session_rows(db) == rows


def test_a_fresh_store_answers_the_persisted_route_and_the_same_object_again(tmp_path, routed, monkeypatch):
    db, source, entry = routed
    store = _store(tmp_path, db)
    assert store._loaded is False
    found = store.lookup_by_session_key_existing(entry.session_key)
    assert found is not None and found.session_id == entry.session_id
    assert found.origin is not None and found.origin.chat_id == source.chat_id
    assert store.lookup_by_session_key_existing(entry.session_key) is found
    assert store.lookup_by_session_key(entry.session_key) is found  # the ordinary load keeps the object


def test_a_route_to_an_ended_session_is_read_as_it_stands_and_the_prune_stays_owed(tmp_path, routed, monkeypatch):
    db, _source, entry = routed
    db.end_session(entry.session_id, "agent_close")
    rows = _session_rows(db)
    store = _store(tmp_path, db)
    pruned = _spy(monkeypatch, store, "_prune_stale_sessions_locked")
    saves = _spy(monkeypatch, store, "_save")
    recovered = _spy(monkeypatch, store, "_recover_session_from_db")
    found = store.lookup_by_session_key_existing(entry.session_key)
    assert found is not None and found.session_id == entry.session_id
    assert (pruned, saves, recovered) == ([], [], [])
    assert _session_rows(db) == rows  # still ended: nothing reopened it
    store._ensure_loaded()  # the first ordinary load owes the prune and runs it
    store._ensure_loaded()
    assert pruned == ["_prune_stale_sessions_locked"]
    assert store._startup_prune_pending is False


def test_the_ordinary_first_load_still_prunes_once_and_then_reconciles(tmp_path, routed, monkeypatch):
    db, _source, _entry = routed
    store = _store(tmp_path, db)
    pruned = _spy(monkeypatch, store, "_prune_stale_sessions_locked")
    reconciled = _spy(monkeypatch, store, "_reconcile_recovered_routing_locked")
    store._ensure_loaded()
    store._ensure_loaded()
    assert pruned == ["_prune_stale_sessions_locked"]
    assert reconciled == ["_reconcile_recovered_routing_locked"]


def _strip_origin(db, store, key):
    data = store._entries[key].to_dict()
    data["origin"] = None
    data.pop("platform")
    db.save_gateway_routing_entry(key, json.dumps(data), scope=store._routing_scope())


@pytest.mark.parametrize("first", ["ordinary", "existing_only"])
def test_origin_healing_is_the_same_whichever_load_comes_first(tmp_path, routed, first):
    db, source, entry = routed
    writer = _store(tmp_path, db)
    writer._ensure_loaded()
    _strip_origin(db, writer, entry.session_key)
    store = _store(tmp_path, db)
    if first == "ordinary":
        healed = store.lookup_by_session_key(entry.session_key)
    else:
        healed = store.lookup_by_session_key_existing(entry.session_key)
        assert store.lookup_by_session_key(entry.session_key) is healed
    assert healed.origin is not None and healed.origin.to_dict() == source.to_dict()
    assert store._routing_fallback_baseline is None


def _mirror_with_a_retired_route(tmp_path, monkeypatch, entry):
    import hermes_state_mutation_retirement as retirement
    retired_key = entry.session_key + ":retired"
    mirror = {retired_key: {**entry.to_dict(), "session_key": retired_key, "session_id": "sid-retired"}}
    (tmp_path / "sessions").mkdir(exist_ok=True)
    (tmp_path / "sessions" / "sessions.json").write_text(json.dumps(mirror))
    monkeypatch.setattr(retirement, "route_retired", lambda _db, key, _json: key == retired_key)
    return retired_key


@pytest.mark.parametrize("first", ["ordinary", "existing_only"])
def test_the_mirror_rewrite_a_first_load_owes_runs_once_whichever_load_comes_first(
        tmp_path, routed, monkeypatch, first):
    db, _source, entry = routed
    retired_key = _mirror_with_a_retired_route(tmp_path, monkeypatch, entry)
    store = _store(tmp_path, db)
    saves = _spy(monkeypatch, store, "_save")
    if first == "existing_only":
        assert store.lookup_by_session_key_existing(retired_key) is None
        assert store.lookup_by_session_key_existing(entry.session_key) is not None
        assert saves == []  # the reader writes nothing; the rewrite stays owed
    store._ensure_loaded()
    store._ensure_loaded()
    assert saves == ["_save"]
    assert retired_key not in store._entries and entry.session_key in store._entries


def test_should_reset_reports_only_an_explicit_suspension(tmp_path, routed):
    db, source, entry = routed
    store = _store(tmp_path, db)
    found = store.lookup_by_session_key_existing(entry.session_key)
    assert store._should_reset(found, source) is None
    found.suspended = True
    assert store._should_reset(found, source) == "suspended"
