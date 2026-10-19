"""The canonical Buzz binding admits each peer only in its own room (CEO choice A).

The CommitLore CTO joins the canonical CEO as a peer beside the repo-factory CTO, each in its own room.
Two authors and two channels kept as independent lists would also admit the cross pairs: each CTO in
the other's room. A binding of more than one author or channel therefore declares its author-channel
pairs, admission is by declared pair, and the identity route resolves the binding's primary pair. A
binding of one author in one channel loads, admits, serializes and identifies exactly as before.

Admission and identity run through the real config parser, resolver, receipt coordinator and HTTP
handlers of an in-process gateway under a tmp home. No model, relay or live gateway is reached.
"""

import asyncio
import contextlib
import json
import os
import secrets
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.hermes_home_pin import import_home, pinned_hermes_home, tmp_root

try:
    import pwd  # the account database names the real home even when HOME is redirected
except ImportError:  # not POSIX
    pwd = None

_REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir) if pwd is not None else Path.home()
_LIVE_ROOT = (_REAL_HOME / ".hermes").resolve()
_PINNED_ROOT = tmp_root("hermes-peer-pairs-")  # removed at exit by this process only
if _PINNED_ROOT.is_relative_to(_LIVE_ROOT) or _LIVE_ROOT.is_relative_to(_PINNED_ROOT):
    raise RuntimeError(f"the pinned test home {_PINNED_ROOT} overlaps the real home")

with pinned_hermes_home(import_home(), importer=__name__):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    from gateway import canonical_surface
    from gateway.canonical_surface import CanonicalSurfaceBinding, ExistingCanonicalBindingResolver
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource


@pytest.fixture(autouse=True, scope="module")
def _pin_the_module_home():
    with pinned_hermes_home(_PINNED_ROOT):
        yield


def _is_live(path) -> bool:
    return Path(path).expanduser().resolve().is_relative_to(_LIVE_ROOT)


def _author(prefix):
    """A synthetic 64-hex signer named by the live signer's 8-hex prefix (never the live key)."""
    return (prefix * 8)[:64]


def _room(prefix):
    """A synthetic room UUID named by the live room's 8-hex prefix (never the live room id)."""
    return "-".join((prefix, "0000", "4000", "8000", prefix + "0000"))


RF_CTO, CL_CTO, LPM_CTO = _author("ed04f02d"), _author("c42d0021"), _author("4afeff6d")
ACP_ROOM, CL_ROOM, LPM_ROOM = _room("c37e88d0"), _room("36536cb2"), _room("6dcb2a67")
PEERS = ((RF_CTO, ACP_ROOM), (CL_CTO, CL_ROOM))
CROSS = ((CL_CTO, ACP_ROOM), (RF_CTO, CL_ROOM))
FOREIGN = ((LPM_CTO, ACP_ROOM), (LPM_CTO, CL_ROOM), (RF_CTO, LPM_ROOM), (CL_CTO, LPM_ROOM))
_ROUTE = "/v1/canonical-surface/events"
_IDENTITY_ROUTE = "/v1/canonical-surface/identity"


def _buzz(authors, channels, pairs=None):
    buzz = {"author_ids": list(authors), "channel_ids": list(channels)}
    if pairs is not None:
        buzz["pairs"] = [{"author_id": author, "channel_id": channel} for author, channel in pairs]
    return buzz


TODAY = _buzz((RF_CTO,), (ACP_ROOM,))
PEER_BINDING = _buzz((RF_CTO, CL_CTO), (ACP_ROOM, CL_ROOM), PEERS)


def _document(buzz, *, session_key="agent:main:telegram:dm:user", session_id="synthetic-session"):
    return {"canonical_surface_bindings": {"canonical": {
        "session_key": session_key, "session_id": session_id,
        "telegram": {"chat_id": "chat", "chat_type": "dm", "user_id": "user", "thread_id": None},
        "buzz": buzz}}}


def _load(buzz, **identity):
    return GatewayConfig.from_dict(_document(buzz, **identity)).canonical_surface_bindings["canonical"]


# ── Config validation ───────────────────────────────────────────────────────────────────────────


def test_todays_single_pair_binding_loads_and_serializes_unchanged():
    """Control (holds before and after): the live shape, one author in one room, with no pairs."""
    config = GatewayConfig.from_dict(_document(TODAY))
    binding = config.canonical_surface_bindings["canonical"]

    assert (binding.allowed_author_ids, binding.allowed_channel_ids) == ((RF_CTO,), (ACP_ROOM,))
    assert getattr(binding, "pairs", ()) == ()
    serialized = config.to_dict()["canonical_surface_bindings"]
    assert serialized == _document(TODAY)["canonical_surface_bindings"]
    assert "pairs" not in serialized["canonical"]["buzz"]
    assert GatewayConfig.from_dict({"canonical_surface_bindings": serialized}).canonical_surface_bindings == {
        "canonical": binding}


def test_a_peer_binding_loads_its_declared_pairs_in_order_and_round_trips():
    config = GatewayConfig.from_dict(_document(PEER_BINDING))
    binding = config.canonical_surface_bindings["canonical"]

    assert binding.pairs == PEERS
    assert binding.primary_pair() == (RF_CTO, ACP_ROOM)
    serialized = config.to_dict()["canonical_surface_bindings"]
    assert serialized == _document(PEER_BINDING)["canonical_surface_bindings"]
    assert GatewayConfig.from_dict({"canonical_surface_bindings": serialized}).canonical_surface_bindings == {
        "canonical": binding}


def test_one_declared_pair_for_one_author_in_one_room_loads():
    binding = _load(_buzz((RF_CTO,), (ACP_ROOM,), [(RF_CTO, ACP_ROOM)]))
    assert binding.pairs == ((RF_CTO, ACP_ROOM),)


@pytest.mark.parametrize("authors, channels", [
    ((RF_CTO, CL_CTO), (ACP_ROOM, CL_ROOM)),  # both lists extended: the cross pairs would be admitted
    ((RF_CTO, CL_CTO), (ACP_ROOM,)),
    ((RF_CTO,), (ACP_ROOM, CL_ROOM)),
], ids=["two-authors-two-rooms", "two-authors", "two-rooms"])
def test_more_than_one_author_or_channel_without_pairs_refuses_to_load(authors, channels):
    with pytest.raises(ValueError, match="canonical Buzz binding must declare its author.channel pairs"):
        _load(_buzz(authors, channels))


@pytest.mark.parametrize("authors, channels, pairs", [
    ((RF_CTO, CL_CTO), (ACP_ROOM, CL_ROOM), [(RF_CTO, ACP_ROOM)]),  # the CommitLore pair left out
    ((RF_CTO, CL_CTO), (ACP_ROOM,), [(RF_CTO, ACP_ROOM)]),  # a listed author in no pair
    ((RF_CTO,), (ACP_ROOM, CL_ROOM), [(RF_CTO, ACP_ROOM)]),  # a listed room in no pair
    ((RF_CTO,), (ACP_ROOM,), [(RF_CTO, ACP_ROOM), (CL_CTO, ACP_ROOM)]),  # a pair's author not listed
    ((RF_CTO,), (ACP_ROOM,), [(RF_CTO, ACP_ROOM), (RF_CTO, CL_ROOM)]),  # a pair's room not listed
    ((RF_CTO, CL_CTO), (ACP_ROOM, CL_ROOM), [(RF_CTO, ACP_ROOM), (CL_CTO, LPM_ROOM)]),
], ids=["pair-missing", "author-uncovered", "room-uncovered", "pair-author-unlisted", "pair-room-unlisted",
        "pair-room-foreign"])
def test_pairs_that_do_not_cover_exactly_the_declared_lists_refuse_to_load(authors, channels, pairs):
    with pytest.raises(ValueError, match="pairs must cover exactly its author_ids and channel_ids"):
        _load(_buzz(authors, channels, pairs))


@pytest.mark.parametrize("pairs", [
    [],
    None,
    "pairs",
    {"author_id": RF_CTO, "channel_id": ACP_ROOM},
    [{"author_id": RF_CTO}],
    [{"author_id": RF_CTO, "channel_id": ACP_ROOM, "owner": True}],
    [{"author_id": RF_CTO, "channel_id": ACP_ROOM}, {"author_id": RF_CTO, "channel_id": ACP_ROOM}],
    [{"author_id": 7, "channel_id": ACP_ROOM}],
    [{"author_id": " ", "channel_id": ACP_ROOM}],
    [[RF_CTO, ACP_ROOM]],
], ids=["empty", "null", "string", "mapping", "missing-channel", "extra-key", "duplicate", "not-text",
        "blank", "list-pair"])
def test_a_malformed_pairs_value_refuses_to_load(pairs):
    """Strictness guard (refused before and after: before, any ``pairs`` key was refused)."""
    buzz = _buzz((RF_CTO,), (ACP_ROOM,))
    buzz["pairs"] = pairs
    with pytest.raises(ValueError):
        _load(buzz)


@pytest.mark.parametrize("extra", [{"owner": True}, {"pairs": [{"author_id": RF_CTO, "channel_id": ACP_ROOM}],
                                                     "owner": True}], ids=["other-key", "pairs-and-other-key"])
def test_the_buzz_key_set_stays_closed_beyond_pairs(extra):
    with pytest.raises(ValueError, match="canonical Buzz binding schema is invalid"):
        _load({**_buzz((RF_CTO,), (ACP_ROOM,)), **extra})


# ── Admission, at the resolver ─────────────────────────────────────────────────────────────────


class _Store:
    """A session store that records whether admission let the event reach the binding lookup."""

    def __init__(self):
        self.lookups = 0

    def lookup_by_session_key_existing(self, session_key):
        self.lookups += 1
        return None  # canonical_binding_stale: past admission, nothing else is exercised here


def _admission(binding, author, channel):
    store = _Store()
    with pytest.raises(ValueError) as refused:
        ExistingCanonicalBindingResolver(store).resolve(binding, SimpleNamespace(author_id=author, channel_id=channel))
    return str(refused.value), store.lookups


def test_the_resolver_admits_only_declared_pairs():
    binding = _load(PEER_BINDING)
    for author, channel in PEERS:
        assert _admission(binding, author, channel) == ("canonical_binding_stale", 1)
    for author, channel in CROSS + FOREIGN:
        assert _admission(binding, author, channel) == ("canonical_principal_rejected", 0), (author[:8], channel[:8])


def test_a_binding_built_with_two_authors_and_no_pairs_admits_no_one():
    """Fail closed for a binding the loader refuses but code could still build: the cross product of
    the lists is never the admission rule."""
    binding = CanonicalSurfaceBinding(
        name="canonical", session_key="k", session_id="s", telegram_chat_id="chat", telegram_chat_type="dm",
        telegram_user_id="user", telegram_thread_id=None, allowed_author_ids=(RF_CTO, CL_CTO),
        allowed_channel_ids=(ACP_ROOM, CL_ROOM))
    for author, channel in PEERS + CROSS:
        assert _admission(binding, author, channel) == ("canonical_principal_rejected", 0)


# ── Admission and identity, through the real HTTP handlers ─────────────────────────────────────


class _CachedActor:
    """The bound cached agent, with no model."""

    compression_in_place = True

    def __init__(self, session_id, db):
        self.session_id = session_id
        self._session_db = db
        self.calls = []

    def interrupt(self, text=None):
        pass

    def release_clients(self):
        pass

    def run_conversation(self, text, *, conversation_history, task_id, **kwargs):
        self.calls.append(kwargs.get("persist_user_message", text))
        self._persist_user_message_idx = len(conversation_history)
        answer = f"answer #{len(self.calls)}"
        return {"completed": True, "session_id": task_id, "final_response": answer,
                "messages": [*conversation_history, {"role": "user", "content": text},
                             {"role": "assistant", "content": answer}]}


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """``gateway(buzz)``: a real runner whose canonical binding is loaded from *buzz* by the config
    parser, its cached actor, and the POST and identity handlers of a real API server adapter."""
    home, user_home = tmp_path / "hermes-home", tmp_path / "user"
    for path in (home, user_home):
        if _is_live(path):
            pytest.fail(f"the test home resolves into the live Hermes home: {path}")
        path.mkdir()
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setattr(Path, "home", lambda: user_home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    override = set_hermes_home_override(home)
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    import hermes_state
    from hermes_cli.config import get_config_path

    for path in (get_config_path(), hermes_state._default_db_path()):
        if _is_live(path) or not path.resolve().is_relative_to(home.resolve()):
            pytest.fail(f"a resolver names a path outside the test's tmp home: {path}")
    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", lambda *a, **kw: pytest.fail(
        "canonical ingress built an agent while one was cached for the session"))
    opened = []

    def build(buzz):
        runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
        opened.append(runner)
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user")
        entry = runner.session_store.get_or_create_session(source)
        db = runner.session_store._db
        binding = _load(buzz, session_key=entry.session_key, session_id=entry.session_id)
        runner.config.canonical_surface_bindings = {binding.name: binding}
        actor = _CachedActor(entry.session_id, db)
        with runner._agent_cache_lock:
            runner._agent_cache[entry.session_key] = (actor, "exact", 0, entry.session_id)
        key = secrets.token_hex(32)
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key}))
        adapter.gateway_runner = runner
        opened.append(adapter)
        routes = {(method, path): handler for method, path, handler in adapter._http_route_table()}

        async def call(handler, request):
            token = _api_request_profile.set(None)
            try:
                response = await handler(request)
            finally:
                _api_request_profile.reset(token)
            return response.status, json.loads(response.text)

        def post(author, channel, event_id, text="hello"):
            raw = json.dumps({"binding": "canonical", "event_id": event_id, "author_id": author,
                              "channel_id": channel, "text": text}).encode()

            async def read():
                return raw
            return asyncio.run(call(routes[("POST", _ROUTE)], SimpleNamespace(
                headers={"Authorization": f"Bearer {key}"}, read=read, method="POST", path_qs=_ROUTE,
                transport=None)))

        def identity():
            return asyncio.run(call(routes[("GET", _IDENTITY_ROUTE)], SimpleNamespace(
                headers={"Authorization": f"Bearer {key}"}, method="GET", path_qs=_IDENTITY_ROUTE,
                raw_path=_IDENTITY_ROUTE, transport=None)))

        def receipts():
            with contextlib.closing(sqlite3.connect(db.db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
                return conn.execute(
                    "SELECT COUNT(*) FROM state_meta WHERE key LIKE 'canonical-receipt:%'").fetchone()[0]

        return SimpleNamespace(post=post, identity=identity, receipts=receipts, actor=actor, entry=entry,
                               binding=binding)

    try:
        yield build
    finally:
        reset_hermes_home_override(override)
        for each in opened:
            if isinstance(each, GatewayRunner):
                each.session_store.close_all_db_handles()
            else:
                each._response_store.close()


def _refusal(response):
    status, body = response
    return status, body.get("error", {}).get("code")


def test_each_intended_pair_is_admitted_and_runs_its_turn(gateway):
    gw = gateway(PEER_BINDING)
    for n, (author, channel) in enumerate(PEERS, 1):
        status, body = gw.post(author, channel, f"event-{n}", text=f"from {author[:8]} in {channel[:8]}")
        assert status == 200, body
        assert body["text"] == f"answer #{n}"
    assert len(gw.actor.calls) == 2 and gw.receipts() == 2


def test_each_cross_pair_is_refused_before_any_claim(gateway):
    gw = gateway(PEER_BINDING)
    for n, (author, channel) in enumerate(CROSS, 1):
        assert _refusal(gw.post(author, channel, f"cross-{n}")) == (403, "canonical_principal_rejected")
    assert gw.actor.calls == [] and gw.receipts() == 0


def test_a_foreign_author_and_a_foreign_room_are_refused(gateway):
    gw = gateway(PEER_BINDING)
    for n, (author, channel) in enumerate(FOREIGN, 1):
        assert _refusal(gw.post(author, channel, f"foreign-{n}")) == (403, "canonical_principal_rejected")
    assert gw.actor.calls == [] and gw.receipts() == 0


def test_todays_binding_admits_exactly_as_before(gateway):
    """Control (holds before and after): only repo-factory's CTO, only in its room."""
    gw = gateway(TODAY)
    assert gw.post(RF_CTO, ACP_ROOM, "today")[0] == 200
    for n, (author, channel) in enumerate(((CL_CTO, CL_ROOM),) + CROSS + FOREIGN, 1):
        assert _refusal(gw.post(author, channel, f"other-{n}")) == (403, "canonical_principal_rejected")
    assert len(gw.actor.calls) == 1 and gw.receipts() == 1


@pytest.fixture
def resolved(monkeypatch):
    """The (author_id, channel_id) every resolver call was asked to admit."""
    seen = []
    original = canonical_surface.ExistingCanonicalBindingResolver.resolve

    def spy(self, binding, event):
        seen.append((event.author_id, event.channel_id))
        return original(self, binding, event)

    monkeypatch.setattr(canonical_surface.ExistingCanonicalBindingResolver, "resolve", spy)
    return seen


@pytest.mark.parametrize("authors, channels, pairs", [
    # The lists' first entries form a cross pair: identifying with them would be refused.
    ((RF_CTO, CL_CTO), (CL_ROOM, ACP_ROOM), PEERS),
    ((RF_CTO, CL_CTO), (ACP_ROOM, CL_ROOM), PEERS[::-1]),
], ids=["lists-first-is-cross", "commitlore-declared-first"])
def test_the_identity_route_resolves_the_first_declared_pair(gateway, resolved, authors, channels, pairs):
    from hermes_state_target_bind import _lineage_root_digest

    gw = gateway(_buzz(authors, channels, pairs))
    status, identity = gw.identity()

    assert status == 200, identity
    assert resolved == [pairs[0]]
    assert identity["session_id"] == gw.entry.session_id
    assert identity["lineage_root_digest"] == _lineage_root_digest(gw.entry.session_id)
    assert gw.actor.calls == [] and gw.receipts() == 0


def test_the_identity_route_for_todays_binding_is_unchanged(gateway, resolved):
    """Control (holds before and after): the single author and channel, the same four fields."""
    from hermes_state_target_bind import _lineage_root_digest

    gw = gateway(TODAY)
    status, identity = gw.identity()

    assert status == 200, identity
    assert resolved == [(RF_CTO, ACP_ROOM)]
    assert set(identity) == {"session_id", "lineage_root_digest", "process_pid", "process_started_at"}
    assert identity["session_id"] == gw.entry.session_id
    assert identity["lineage_root_digest"] == _lineage_root_digest(gw.entry.session_id)
    assert gw.identity() == (status, identity)
    assert gw.actor.calls == [] and gw.receipts() == 0
