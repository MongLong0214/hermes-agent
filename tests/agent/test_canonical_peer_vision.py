"""A canonical peer turn may run the built-in ``vision_analyze``, only inside its ``local`` media scope.

The policy (``gateway.canonical_peer_tools.local``) admits an image file whose realpath sits under a
configured root, or a URL of the configured Blossom media base (exact scheme, host, port, path prefix).
Everything else refuses the whole batch before anything is read or fetched, and the call then runs
under the same scope, so a redirect hop outside it is refused before it is requested. The config is
patched in memory; nothing here reads or writes a real Hermes config."""

from __future__ import annotations

import asyncio
import base64

import httpx
import pytest

BINDING = "acp-canonical-ceo"
BASE = "https://isaac-macmini.tailb41ff5.ts.net:3000/media"  # the relay's BUZZ_MEDIA_BASE_URL
BLOB = f"{BASE}/0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0.png"
EVIL = "https://evil.example/media/x.png"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
AGENT = object()


@pytest.fixture
def media(tmp_path):
    work = tmp_path / "work"
    for d in ("episodes", "upload", "episodes-x"):
        (work / d).mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    for f in (work / "episodes" / "frame.png", work / "upload" / "thumb.png", work / "episodes-x" / "frame.png",
              tmp_path / "outside" / "secret.png"):
        f.write_bytes(PNG)
    (work / "episodes" / "escape.png").symlink_to(tmp_path / "outside" / "secret.png")
    return work


def _policy(media, **local) -> dict:
    entry = {"urls": [BASE], "roots": [str(media / "episodes"), str(media / "upload")]}
    return {"binding": BINDING, "mcp_server": "acp-ceo", "read": ["read_file", "terminal", "vision_analyze"],
            "mutate": {}, "local": local or {"vision_analyze": entry, "read_file": entry}}


@pytest.fixture
def peer(monkeypatch, media):
    """A bound grant for the configured binding, the policy patched in memory, and a record of every
    fetch (``fetched``) and every read or fetch that reached the resolver's chokepoint (``read``)."""
    import tools.image_source as image_source
    import tools.vision_tools as vision_tools
    from agent.canonical_peer_tools import PeerToolGrant, bind_grant, reset_grant
    from tools.approval_context import reset_turn_principal, set_turn_principal

    state = {"policy": _policy(media), "routes": {BLOB: httpx.Response(200, content=PNG)}}
    fetched, read = [], []
    monkeypatch.setattr("hermes_cli.config.load_config",
                        lambda *a, **k: {"gateway": {"canonical_peer_tools": state["policy"]}})

    def transport(request):
        fetched.append(str(request.url))
        return state["routes"].get(str(request.url), httpx.Response(404))

    monkeypatch.setattr(image_source, "_scoped_http_client", lambda timeout: httpx.AsyncClient(
        transport=httpx.MockTransport(transport), follow_redirects=False), raising=False)
    real_finalize = image_source._finalize
    monkeypatch.setattr(image_source, "_finalize",
                        lambda data, mime, origin, src, permitted=("image",): read.append((origin, src))
                        or real_finalize(data, mime, origin, src, permitted))
    monkeypatch.setattr(vision_tools, "_should_use_native_vision_fast_path", lambda: True)
    principal = set_turn_principal("peer")
    token = bind_grant(PeerToolGrant(binding=BINDING, agent=AGENT, verify=lambda: None))
    try:
        yield state, fetched, read
    finally:
        reset_grant(token)
        reset_turn_principal(principal)


def _allowed(*image_urls: str) -> bool:
    from agent.canonical_peer_tools import batch_allowed

    return batch_allowed(AGENT, [("vision_analyze", {"image_url": u, "question": "what is shown?"}) for u in image_urls])


def _run(image_url: str, args: dict | None = None):
    """Dispatch the pinned call as ``model_tools._execute_tool`` does; the parsed result."""
    import json

    from agent.canonical_peer_tools import REFUSED, dispatch_handler
    from tools.registry import registry

    checked = {"image_url": image_url, "question": "what is shown?"}
    handler = dispatch_handler("vision_analyze", checked, "")
    assert handler is not None and handler is not REFUSED
    result = registry.dispatch("vision_analyze", args or checked, handler=handler, task_id="t")
    return result if isinstance(result, dict) else json.loads(result)


def _loaded(result) -> bool:
    return isinstance(result, dict) and result.get("_multimodal") is True


# --- admitted ----------------------------------------------------------------------------------


def test_a_png_under_an_allowed_root_is_admitted_and_read(peer, media):
    _, fetched, read = peer
    path = str(media / "episodes" / "frame.png")
    assert _allowed(path, str(media / "upload" / "thumb.png"), f"file://{path}")
    assert _allowed(path)
    assert _loaded(_run(path))
    assert read == [("file", path)] and fetched == []


def test_an_allowed_blossom_url_is_admitted_and_fetched(peer):
    _, fetched, read = peer
    assert _allowed(BLOB)
    assert _loaded(_run(BLOB))
    assert fetched == [BLOB] and read == [("http", BLOB)]


def test_a_redirect_hop_inside_the_base_is_followed(peer):
    state, fetched, _ = peer
    hop = f"{BASE}/moved.png"
    state["routes"] = {BLOB: httpx.Response(302, headers={"location": "/media/moved.png"}),
                       hop: httpx.Response(200, content=PNG)}
    assert _allowed(BLOB)
    assert _loaded(_run(BLOB))
    assert fetched == [BLOB, hop]


# --- refused before any read or fetch ----------------------------------------------------------

HOST = "isaac-macmini.tailb41ff5.ts.net"
OUT_OF_SCOPE_URLS = [
    pytest.param(f"https://{HOST}:3001/media/x.png", id="same-host-other-port"),
    pytest.param(f"https://{HOST}/media/x.png", id="same-host-default-port"),
    pytest.param(f"http://{HOST}:3000/media/x.png", id="same-host-other-scheme"),
    pytest.param(f"https://{HOST}:3000/admin/x.png", id="same-host-disallowed-path"),
    pytest.param(f"https://{HOST}:3000/mediax/x.png", id="same-host-prefix-sibling-path"),
    pytest.param(f"https://{HOST}:3000/media/../admin/x.png", id="url-dot-dot"),
    pytest.param(f"https://{HOST}:3000/media/%2e%2e/admin/x.png", id="url-encoded-dot-dot"),
    pytest.param(EVIL, id="other-host"),
    pytest.param(f"https://{HOST}:3000@evil.example/media/x.png", id="userinfo-allowed-host-at-evil"),
    pytest.param(f"https://{HOST}@evil.example/media/x.png", id="userinfo-allowed-at-evil"),
    pytest.param(f"https://user@{HOST}:3000/media/x.png", id="userinfo-on-allowed-host"),
    pytest.param(f"https://evil.example\\@{HOST}:3000/media/x.png", id="backslash-userinfo"),
    pytest.param(f"https://{HOST}.:3000/media/x.png", id="host-trailing-dot"),
    pytest.param("data:image/png;base64," + base64.b64encode(PNG).decode(), id="data-url"),
    pytest.param("ftp://evil.example/media/x.png", id="other-scheme"),
]


def _out_of_scope_paths(media) -> dict:
    outside = media.parent / "outside" / "secret.png"
    return {
        "sibling-dir": str(media / "episodes-x" / "frame.png"),
        "symlink-escape": str(media / "episodes" / "escape.png"),
        "dot-dot": f"{media}/episodes/../../outside/secret.png",
        "file-url-outside": f"file://{outside}",
        "file-url-with-host": f"file://evil.example{media}/episodes/frame.png",
        "relative": "frame.png",
    }


PATH_CASES = ["sibling-dir", "symlink-escape", "dot-dot", "file-url-outside", "file-url-with-host", "relative"]


@pytest.mark.parametrize("image_url", OUT_OF_SCOPE_URLS)
def test_an_out_of_scope_url_refuses_the_batch_before_any_fetch(peer, image_url):
    _, fetched, read = peer
    assert not _allowed(image_url)
    assert not _allowed(BLOB, image_url)  # one refusal refuses the whole batch
    assert _allowed(BLOB)
    assert fetched == [] and read == []


@pytest.mark.parametrize("case", PATH_CASES)
def test_an_out_of_scope_path_refuses_the_batch_before_any_read(peer, media, case):
    _, fetched, read = peer
    image_url = _out_of_scope_paths(media)[case]
    assert not _allowed(image_url)
    assert not _allowed(str(media / "episodes" / "frame.png"), image_url)
    assert _allowed(str(media / "episodes" / "frame.png"))
    assert fetched == [] and read == []


def _scope():
    """The media scope the configured policy holds for vision_analyze."""
    from agent.canonical_peer_tools import _load_policy

    policy = _load_policy()
    assert policy is not None and policy.local["vision_analyze"] is not None
    return policy.local["vision_analyze"]


@pytest.mark.parametrize("image_url", OUT_OF_SCOPE_URLS)
def test_the_scoped_resolver_refuses_an_out_of_scope_url_without_fetching(peer, image_url):
    """Defense in depth: the same scope is enforced again where the bytes are read."""
    from tools.image_source import ResolveContext, SourceUnsafe, resolve_image_source, scoped_media

    _, fetched, read = peer
    with scoped_media(_scope()), pytest.raises(SourceUnsafe):
        asyncio.run(resolve_image_source(image_url, ResolveContext()))
    assert fetched == [] and read == []


@pytest.mark.parametrize("case", PATH_CASES)
def test_the_scoped_resolver_refuses_an_out_of_scope_path_without_reading(peer, media, case):
    from tools.image_source import ResolveContext, SourceUnsafe, resolve_image_source, scoped_media

    _, fetched, read = peer
    with scoped_media(_scope()), pytest.raises(SourceUnsafe):
        asyncio.run(resolve_image_source(_out_of_scope_paths(media)[case], ResolveContext()))
    assert fetched == [] and read == []


def test_a_redirect_to_another_host_is_refused_before_it_is_requested(peer):
    state, fetched, read = peer
    state["routes"] = {BLOB: httpx.Response(302, headers={"location": EVIL}), EVIL: httpx.Response(200, content=PNG)}
    assert _allowed(BLOB)
    result = _run(BLOB)
    assert not _loaded(result) and "outside the media base" in json_text(result)
    assert fetched == [BLOB] and read == []


def test_a_redirect_chain_leaving_on_its_second_hop_is_refused_there(peer):
    state, fetched, read = peer
    hop = f"{BASE}/hop.png"
    state["routes"] = {BLOB: httpx.Response(302, headers={"location": hop}),
                       hop: httpx.Response(307, headers={"location": EVIL}), EVIL: httpx.Response(200, content=PNG)}
    assert _allowed(BLOB)
    result = _run(BLOB)
    assert not _loaded(result) and "outside the media base" in json_text(result)
    assert fetched == [BLOB, hop] and read == []


def test_dispatch_with_arguments_other_than_the_checked_ones_runs_nothing(peer, media):
    _, fetched, read = peer
    allowed = str(media / "episodes" / "frame.png")
    assert _allowed(allowed)
    result = _run(allowed, {"image_url": str(media.parent / "outside" / "secret.png"), "question": "what is shown?"})
    assert not _loaded(result) and "Not run" in json_text(result)
    assert fetched == [] and read == []


def json_text(result) -> str:
    import json

    return json.dumps(result)


# --- the rest of the policy is unchanged -------------------------------------------------------


def test_read_file_and_other_non_native_tools_are_still_refused(peer, media):
    from agent.canonical_peer_tools import batch_allowed

    path = str(media / "episodes" / "frame.png")
    assert _allowed(path)
    # read_file is listed under ``read`` and under ``local``; terminal under ``read``: none is native to
    # the CEO MCP server and ``local`` names vision_analyze only.
    for name, args in (("read_file", {"path": path}), ("terminal", {"command": "true"})):
        assert not batch_allowed(AGENT, [(name, args)])
        assert not batch_allowed(AGENT, [("vision_analyze", {"image_url": path}), (name, args)])


def test_vision_analyze_needs_a_well_formed_local_entry(peer, media):
    state, _, _ = peer
    path = str(media / "episodes" / "frame.png")
    roots = [str(media / "episodes")]
    for local in ({}, {"vision_analyze": {"roots": roots, "extra": True}}, {"vision_analyze": {"roots": roots[0]}},
                  {"vision_analyze": {"roots": ["relative/dir"]}}, {"vision_analyze": {"urls": [f"{BASE}?q=1"]}},
                  {"vision_analyze": None}):
        state["policy"] = {**_policy(media), "local": local}  # vision_analyze stays listed under ``read``
        assert not _allowed(path), local
    state["policy"] = _policy(media, vision_analyze={"roots": roots})
    assert _allowed(path)


@pytest.mark.parametrize("dispatcher", ["execute_tool_calls_sequential", "execute_tool_calls_concurrent",
                                        "execute_tool_calls_segmented"])
def test_the_executors_run_an_admitted_call_and_nothing_of_a_batch_with_an_out_of_scope_one(peer, media, dispatcher):
    import json
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    import agent.tool_executor as executor
    from agent.canonical_peer_tools import PeerToolGrant, bind_grant, reset_grant
    from run_agent import AIAgent

    _, fetched, read = peer
    defs = [{"type": "function", "function": {"name": n, "description": n, "parameters": {"type": "object"}}}
            for n in ("vision_analyze", "read_file")]
    with (patch("model_tools.get_tool_definitions", return_value=defs),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("hermes_cli.config.load_config_readonly", return_value={}),
          patch("agent.process_bootstrap.OpenAI")):
        built = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, platform="telegram")
    for attr, value in (("client", MagicMock()), ("compression_enabled", False), ("save_trajectories", False),
                        ("_turn_principal", "peer")):
        setattr(built, attr, value)

    def call(name, args, call_id):
        return SimpleNamespace(id=call_id, type="function",
                               function=SimpleNamespace(name=name, arguments=json.dumps(args)))

    def run(calls):
        messages: list = []
        getattr(executor, dispatcher)(built, SimpleNamespace(tool_calls=calls), messages, "task")
        return messages

    allowed = call("vision_analyze", {"image_url": str(media / "episodes" / "frame.png"), "question": "q"}, "c1")
    token = bind_grant(PeerToolGrant(binding=BINDING, agent=built, verify=lambda: None))
    try:
        for other in (call("vision_analyze", {"image_url": str(media / "episodes-x" / "frame.png")}, "c2"),
                      call("vision_analyze", {"image_url": BLOB.replace(":3000", ":3001")}, "c2"),
                      call("read_file", {"path": str(media / "episodes" / "frame.png")}, "c2")):
            messages = run([allowed, other])
            assert len(messages) == 2 and all("not executed" in str(m["content"]) for m in messages)
            assert fetched == [] and read == []
        messages = run([allowed])
    finally:
        reset_grant(token)
        setattr(built, "_turn_principal", None)
    assert len(messages) == 1 and "not executed" not in str(messages[0]["content"])
    assert read == [("file", str(media / "episodes" / "frame.png"))]


def test_discovery_offers_vision_analyze_only_through_local(peer, media):
    from agent.canonical_peer_tools import scope_discovery_defs

    state, _, _ = peer
    defs = [{"type": "function", "function": {"name": n}} for n in ("vision_analyze", "read_file", "terminal")]
    assert [d["function"]["name"] for d in scope_discovery_defs(defs)] == ["vision_analyze"]
    state["policy"] = {**_policy(media), "local": {}}
    assert scope_discovery_defs(defs) == []
