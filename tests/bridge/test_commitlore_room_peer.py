"""A mention in the CommitLore room reaches the canonical CEO and its answer goes back to that room.

CEO choice A adds the CommitLore CTO as a peer of the canonical CEO, admitted only in its own room. The
shim already reads the author and the room from the event itself and posts the reply to the event's
room with the handle the reply registry names for it; these witnesses drive the real shim
(``bridge/canonical/acp-canonical-shim.py``, or ``SHIM_PATH``) through ``handle()`` into the real
canonical handler, coordinator and SQLite of an in-process gateway whose binding is loaded by the real
config parser with declared author-room pairs. The registry mirrors the live file's shape with
synthetic keys; the Buzz wrapper is a stand-in that records what was posted.
"""

import asyncio
import importlib.util
import io
import json
import os
import secrets
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.bridge import buzz_prompt_port as bp
from tests.hermes_home_pin import import_home, pinned_hermes_home, tmp_root

try:
    import pwd  # the account database names the real home even when HOME is redirected
except ImportError:  # not POSIX
    pwd = None

_REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir) if pwd is not None else Path.home()
_LIVE_ROOT = (_REAL_HOME / ".hermes").resolve()
_PINNED_ROOT = tmp_root("hermes-shim-peer-room-")  # removed at exit by this process only
if _PINNED_ROOT.is_relative_to(_LIVE_ROOT) or _LIVE_ROOT.is_relative_to(_PINNED_ROOT):
    raise RuntimeError(f"the pinned test home {_PINNED_ROOT} overlaps the real home")

with pinned_hermes_home(import_home(), importer=__name__):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

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


def _hex(prefix):
    """A synthetic 64-hex value named by an 8-hex prefix (never a live key or event id)."""
    return (prefix * 8)[:64]


def _room(prefix):
    """A synthetic room UUID named by the live room's 8-hex prefix (never the live room id)."""
    return "-".join((prefix, "0000", "4000", "8000", prefix + "0000"))


_SHIM = Path(os.environ.get("SHIM_PATH") or
             Path(__file__).resolve().parents[2] / "bridge" / "canonical" / "acp-canonical-shim.py")
_ROUTE = "/v1/canonical-surface/events"
_URL = "http://127.0.0.1:9"  # nothing listens; urlopen is routed in-process
_PATH_VARS = ("CANONICAL_SHIM_LOG", "CANONICAL_QUARANTINE_LOG", "CANONICAL_CANARY_FILE",
              "CANONICAL_PUBLISHED_DIR", "CANONICAL_MENTIONS_FILE", "CANONICAL_BUZZ_CLI")

CEO = _hex("2adaf98f")  # this bridge's signer, every room's registered ``ceo``
RF_CTO, CL_CTO, LPM_CTO = _hex("ed04f02d"), _hex("c42d0021"), _hex("4afeff6d")
ACP_ROOM, CL_ROOM, LPM_ROOM = _room("c37e88d0"), _room("36536cb2"), _room("6dcb2a67")
ROOMS = {ACP_ROOM: ("repo-factory", RF_CTO, "cto-repo-factory"), CL_ROOM: ("commitlore", CL_CTO, "cto-clore"),
         LPM_ROOM: ("logic-pro-mcp", LPM_CTO, "cto-lpm")}
LABELS = {CEO: "ceo", RF_CTO: "cto-repo-factory", CL_CTO: "cto-clore", LPM_CTO: "cto-lpm"}


def _registry():
    """The live ``required-mentions.json`` shape: one entry per room, the one ``ceo`` key, the room's peer."""
    return {"version": 1, "channels": {
        room: {"relay_name": name, "ceo": {"pubkey": CEO, "handle": "ceo"},
               "peer": {"pubkey": peer, "handle": handle}, "classification": "current project"}
        for room, (name, peer, handle) in ROOMS.items()},
        "known_profiles": {handle: peer for _, peer, handle in ROOMS.values()} | {"ceo": CEO}}


def _binding_document(session_key, session_id):
    """The canonical CEO binding of choice A: each CTO admitted in its own room only."""
    return {"canonical_surface_bindings": {"acp-canonical-ceo": {
        "session_key": session_key, "session_id": session_id,
        "telegram": {"chat_id": "chat", "chat_type": "dm", "user_id": "user", "thread_id": None},
        "buzz": {"author_ids": [RF_CTO, CL_CTO], "channel_ids": [ACP_ROOM, CL_ROOM],
                 "pairs": [{"author_id": RF_CTO, "channel_id": ACP_ROOM},
                           {"author_id": CL_CTO, "channel_id": CL_ROOM}]}}}}


def mention(event_id, author, room, body):
    """buzz-acp's own single-event prompt for a top-level @mention (queue.rs ``format_prompt``)."""
    return bp.format_prompt(room, ROOMS[room][0], [bp.Ev(event_id, author, 1_792_324_251, body, [["h", room]])],
                            LABELS)


class _Actor:
    """The bound cached agent, with no model: run n answers ``answer #<n>``."""

    compression_in_place = True

    def __init__(self, session_id, db, runs):
        self.session_id = session_id
        self._session_db = db
        self.runs = runs

    def interrupt(self, text=None):
        pass

    def release_clients(self):
        pass

    def run_conversation(self, text, *, conversation_history, task_id, **kwargs):
        self.runs.append(text)
        answer = f"answer #{len(self.runs)}"
        self._persist_user_message_idx = len(conversation_history)
        return {"completed": True, "session_id": task_id, "final_response": answer,
                "messages": [*conversation_history, {"role": "user", "content": text},
                             {"role": "assistant", "content": answer}]}


class _Response:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def gw(tmp_path, monkeypatch):
    """A real runner, coordinator and canonical HTTP handler under a tmp home, its binding loaded from
    the choice-A document by the config parser. The shim's ``urlopen`` is routed to that handler."""
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
    runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user")
    entry = runner.session_store.get_or_create_session(source)
    db = runner.session_store._db
    runner.config.canonical_surface_bindings = GatewayConfig.from_dict(
        _binding_document(entry.session_key, entry.session_id)).canonical_surface_bindings
    runs = []
    with runner._agent_cache_lock:
        runner._agent_cache[entry.session_key] = (_Actor(entry.session_id, db, runs), "exact", 0, entry.session_id)
    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", lambda *a, **kw: pytest.fail(
        "canonical ingress built an agent while one was cached for the session"))
    key = secrets.token_hex(32)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key}))
    adapter.gateway_runner = runner
    [handler] = [h for method, path, h in adapter._http_route_table() if (method, path) == ("POST", _ROUTE)]
    received, answers = [], []

    async def post(raw, authorization):
        async def read():
            return raw
        request = SimpleNamespace(headers={"Authorization": authorization}, read=read, method="POST",
                                  path_qs=_ROUTE, transport=None)
        token = _api_request_profile.set(None)
        try:
            response = await handler(request)
        finally:
            _api_request_profile.reset(token)
        return response.status, response.text.encode()

    def urlopen(request, timeout=None):
        if request.full_url != _URL + _ROUTE:
            pytest.fail(f"the shim posted somewhere else: {request.full_url}")
        received.append(json.loads(request.data))
        status, body = asyncio.run(post(request.data, request.get_header("Authorization")))
        answers.append((status, json.loads(body)))
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "refused", {}, io.BytesIO(body))
        return _Response(body)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    try:
        yield SimpleNamespace(key=key, runs=runs, received=received, answers=answers)
    finally:
        reset_hermes_home_override(override)
        runner.session_store.close_all_db_handles()
        adapter._response_store.close()


@pytest.fixture
def shim(gw, tmp_path, monkeypatch):
    root = tmp_path / "bridge"
    root.mkdir()
    calls = root / "buzz-calls.jsonl"
    (root / "buzz").write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"open({str(calls)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], 'stdin': sys.stdin.read()}}) + '\\n')\n")
    (root / "buzz").chmod(0o755)
    (root / "required-mentions.json").write_text(json.dumps(_registry()))
    env = {
        "CANONICAL_SURFACE_URL": _URL, "API_SERVER_KEY": gw.key, "CANONICAL_BINDING": "acp-canonical-ceo",
        "CANONICAL_AUTHOR_ID": RF_CTO, "CANONICAL_CHANNEL_ID": ACP_ROOM, "CANONICAL_FORWARD": "live",
        "CANONICAL_BUSY_RETRY_SECS": "0", "CANONICAL_BUSY_DEADLINE_SECS": "5",
        "CANONICAL_SHIM_LOG": str(root / "shim.log"), "CANONICAL_QUARANTINE_LOG": str(root / "q.jsonl"),
        "CANONICAL_CANARY_FILE": str(root / "canary.json"), "CANONICAL_PUBLISHED_DIR": str(root / "published"),
        "CANONICAL_MENTIONS_FILE": str(root / "required-mentions.json"), "CANONICAL_BUZZ_CLI": str(root / "buzz"),
    }
    for name in _PATH_VARS:
        if _is_live(env[name]):
            pytest.fail(f"{name} names a live path: {env[name]}")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location("shim_peer_room_under_test", str(_SHIM))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "write", lambda payload: None)  # buzz-acp's stdout, not a post
    module.posts = lambda: [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    return module


def drive(shim, blocks, rpc_id=1):
    shim.handle({"jsonrpc": "2.0", "id": rpc_id, "method": "session/prompt",
                 "params": bp.prompt_params("canonical:acp-canonical-ceo", blocks)})


def _reply(post):
    argv = post["argv"]
    return argv[argv.index("--channel") + 1], argv[argv.index("--reply-to") + 1], post["stdin"]


# ── Witnesses ────────────────────────────────────────────────────────────────────────────────────


def test_a_commitlore_room_mention_is_answered_in_that_room_with_the_cto_clore_handle(gw, shim):
    event = _hex("c1000001")
    drive(shim, mention(event, CL_CTO, CL_ROOM, "@ceo [CTO→CEO] status from the CommitLore room"))

    [sent] = gw.received
    assert (sent["binding"], sent["event_id"], sent["author_id"], sent["channel_id"]) == (
        "acp-canonical-ceo", event, CL_CTO, CL_ROOM)
    [(status, body)] = gw.answers
    assert (status, body["text"]) == (200, "answer #1")
    assert len(gw.runs) == 1
    [post] = shim.posts()
    assert _reply(post) == (CL_ROOM, event, "@cto-clore answer #1")


def test_a_repo_factory_room_mention_is_still_answered_in_its_own_room(gw, shim):
    event = _hex("a1000001")
    drive(shim, mention(event, RF_CTO, ACP_ROOM, "@ceo [CTO→CEO] status from the repo-factory room"))

    [sent] = gw.received
    assert (sent["author_id"], sent["channel_id"]) == (RF_CTO, ACP_ROOM)
    assert len(gw.runs) == 1
    [post] = shim.posts()
    assert _reply(post) == (ACP_ROOM, event, "@cto-repo-factory answer #1")


def test_both_rooms_in_turn_each_answer_goes_to_its_own_room(gw, shim):
    first, second = _hex("a1000002"), _hex("c1000002")
    drive(shim, mention(first, RF_CTO, ACP_ROOM, "@ceo one"), rpc_id=1)
    drive(shim, mention(second, CL_CTO, CL_ROOM, "@ceo two"), rpc_id=2)

    assert [(s["author_id"], s["channel_id"]) for s in gw.received] == [(RF_CTO, ACP_ROOM), (CL_CTO, CL_ROOM)]
    assert [_reply(p) for p in shim.posts()] == [(ACP_ROOM, first, "@cto-repo-factory answer #1"),
                                                 (CL_ROOM, second, "@cto-clore answer #2")]


@pytest.mark.parametrize("author, room", [
    (RF_CTO, CL_ROOM),  # repo-factory's CTO in the CommitLore room
    (CL_CTO, ACP_ROOM),  # CommitLore's CTO in the repo-factory room
    (LPM_CTO, CL_ROOM),  # a foreign author in an admitted room
    (LPM_CTO, LPM_ROOM),  # a registered room the binding does not admit (Logic, until its CTO is live)
], ids=["rf-cto-in-commitlore", "clore-cto-in-repo-factory", "foreign-author", "foreign-room"])
def test_a_cross_or_foreign_pair_is_refused_at_the_gateway_and_nothing_is_posted(gw, shim, author, room):
    event = _hex("d1000001")
    drive(shim, mention(event, author, room, "@ceo should not run"))

    [sent] = gw.received
    assert (sent["author_id"], sent["channel_id"]) == (author, room)  # the event's own, nothing stands in
    [(status, body)] = gw.answers
    assert (status, body["error"]["code"]) == (403, "canonical_principal_rejected")
    assert gw.runs == []
    assert shim.posts() == []  # a refusal before the claim keeps its silence
