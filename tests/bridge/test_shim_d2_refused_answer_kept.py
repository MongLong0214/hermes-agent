"""Canonical reply delivery and refused-answer storage through the real shim and gateway.

Completed turns deliver through empty assistant rows and recovery nudges. A tool-pairing error
still refuses; its bound answer exercises kept-answer storage and sender notices without relying
on the obsolete rule that refused recovery rows.
"""

import asyncio
import contextlib
import importlib.util
import io
import json
import os
import secrets
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
import buzz_prompt_port as bp

from tests.hermes_home_pin import import_home, pinned_hermes_home, tmp_root

try:
    import pwd  # the account database names the real home even when HOME is redirected
except ImportError:  # not POSIX
    pwd = None

_REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir) if pwd is not None else Path.home()
_LIVE_ROOT = (_REAL_HOME / ".hermes").resolve()
_PINNED_ROOT = tmp_root("hermes-shim-d2-o4-")
if _PINNED_ROOT.is_relative_to(_LIVE_ROOT) or _LIVE_ROOT.is_relative_to(_PINNED_ROOT):
    raise RuntimeError(f"the pinned test home {_PINNED_ROOT} overlaps the real home")

with pinned_hermes_home(import_home(), importer=__name__):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    from agent.canonical_peer import new_peer_metadata, render_peer_turn
    from agent.conversation_loop import _CODEX_INCOMPLETE_NUDGE
    from gateway import canonical_answer_fallback, canonical_surface
    from gateway.canonical_surface import CanonicalReceiptCoordinator, CanonicalSurfaceBinding
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


_SHIM = Path(os.environ.get("SHIM_PATH") or
             Path(__file__).resolve().parents[2] / "bridge" / "canonical" / "acp-canonical-shim.py")
_ROUTE = "/v1/canonical-surface/events"
_URL = "http://127.0.0.1:9"  # nothing listens; urlopen is routed in-process
SIGNER = "73bd7a526adb00a06ed2ec820701e2f761d00cc0dc0ed1fb051f8923d6fdc75a"
OTHER = "e79dda8ddd8f994b5758a2f8eed4bc76b6b9c418d16ed016f9e437ee7c35ec4d"
CHANNEL = "b9cda259-f165-45ce-afaf-3e8f078a144a"
EVENT = "813415c3" + "3" * 56
FINAL = "`capfit` context message: no Buzz reply needed; receipt alone does not verify the cap."


def channel_prompt(body):
    """buzz-acp's own single-event prompt for a top-level message (queue.rs ``format_prompt``, ported in
    buzz_prompt_port): ``[Context]`` and the event as elements of their own, the event's Tags line naming
    the [Context] channel, and the anchor its own id, as #91 r1 requires."""
    # Receipt tests must reach a turn: non-owner broadcasts are now intentionally dropped.
    return bp.format_prompt(CHANNEL, "repo-factory", [bp.Ev(EVENT, SIGNER, 1_792_324_251, body,
                                                            [["h", CHANNEL], ["p", OTHER]],
                                                            prompt_tag="message")], {SIGNER: "cto-repo-factory"})


class _ShapedActor:
    """Return completed recovery shapes, foreign-event rows, or a tool-pairing error."""

    compression_in_place = True

    def __init__(self, session_id, db, runs):
        self.session_id = session_id
        self._session_db = db
        self.runs = runs
        self.shape = "unanswered_tool"  # pairing error, provably this turn's

    def interrupt(self, text=None):
        pass

    def release_clients(self):
        pass

    def run_conversation(self, text, *, conversation_history, task_id, **kwargs):
        self.runs.append(text)
        self._persist_user_message_idx = len(conversation_history)
        user = {"role": "user", "content": text}
        # The row as the real agent returns it: the clean body, marked with this claim's peer metadata.
        marked = {"role": "user", "content": kwargs["persist_user_message"],
                  "display_kind": kwargs["persist_user_display_kind"],
                  "display_metadata": kwargs["persist_user_display_metadata"]}
        foreign = {**marked, "display_metadata": {"canonical_peer": {
            **kwargs["persist_user_display_metadata"]["canonical_peer"], "nonce": "0" * 32}}}
        final = {"role": "assistant", "content": FINAL, "finish_reason": "stop"}
        empty = {"role": "assistant", "content": "", "finish_reason": "incomplete"}
        nudge = {"role": "user", "content": _CODEX_INCOMPLETE_NUDGE}
        suffix = {
            "plain": [user, final],
            "unanswered_tool": [user, {**empty, "tool_calls": [{"id": "missing-result"}]}, final],
            "unanswered_tool_marked": [marked, {**empty, "tool_calls": [{"id": "missing-result"}]}, final],
            "unnudged_retry": [user, empty, final],
            "unnudged_retry_marked": [marked, empty, final],
            "nudged": [user, empty, nudge, final],
            "nudged_marked": [marked, empty, nudge, final],
            "foreign_marked": [foreign, empty, nudge, final],
            # an owner (or any non-agent) row inside the turn: its answer may address that row
            "other_user_row": [user, empty, {"role": "user", "content": "owner: answer me instead"}, final],
            # the turn's own row rendered under another claim's nonce
            "foreign_peer_row": [{"role": "user", "content": render_peer_turn(new_peer_metadata(
                binding="canonical", author_id=SIGNER, channel_id=CHANNEL, event_id=EVENT,
                receipt="canonical-receipt:v2:" + "0" * 64), "the same event, another claim")},
                                 empty, nudge, final],
            # the last row is not the returned final answer
            "final_mismatch": [user, empty, nudge, {"role": "assistant", "content": "another text"}],
        }[self.shape]
        return {"completed": True, "session_id": task_id, "final_response": FINAL,
                "messages": [*conversation_history, *suffix]}


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
    entry_ = runner.session_store.get_or_create_session(source)
    db = runner.session_store._db
    binding = CanonicalSurfaceBinding(
        name="canonical", session_key=entry_.session_key, session_id=entry_.session_id,
        telegram_chat_id="chat", telegram_chat_type="dm", telegram_user_id="user",
        telegram_thread_id=None, allowed_author_ids=(SIGNER,), allowed_channel_ids=(CHANNEL,),
    )
    runner.config.canonical_surface_bindings = {binding.name: binding}
    runs = []
    actor = _ShapedActor(entry_.session_id, db, runs)
    with runner._agent_cache_lock:
        runner._agent_cache[entry_.session_key] = (actor, "exact", 0, entry_.session_id)
    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", lambda *a, **kw: pytest.fail("an agent was built"))
    key = secrets.token_hex(32)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key}))
    adapter.gateway_runner = runner
    [handler] = [h for method, path, h in adapter._http_route_table() if (method, path) == ("POST", _ROUTE)]
    responses = []
    # Reach: where D2's answer was lost. The selector refuses the turn's shape with the actor's answer in
    # hand; with O4 the binding check then decides whether that answer is kept.
    refusals, bound = [], []
    select = GatewayRunner._select_canonical_turn_result

    def spy_select(result, **kwargs):
        try:
            return select(result, **kwargs)
        except ValueError as exc:
            refusals.append((str(exc), result.get("final_response") if isinstance(result, dict) else None))
            raise

    monkeypatch.setattr(GatewayRunner, "_select_canonical_turn_result", staticmethod(spy_select))
    check = getattr(canonical_surface, "bound_refused_answer", None)
    if check is not None:
        def spy_bound(*args, **kwargs):
            bound.append(check(*args, **kwargs))
            return bound[-1]

        monkeypatch.setattr(canonical_surface, "bound_refused_answer", spy_bound)

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
        status, body = asyncio.run(post(request.data, request.get_header("Authorization")))
        responses.append((status, json.loads(body)))
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "refused", {}, io.BytesIO(body))
        return _Response(body)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    def rows(prefix):
        with contextlib.closing(sqlite3.connect((home / "state.db").resolve().as_uri() + "?mode=ro",
                                                uri=True)) as conn:
            return dict(conn.execute("SELECT key, value FROM state_meta WHERE key LIKE ?", (prefix + "%",)))

    try:
        yield SimpleNamespace(key=key, runs=runs, rows=rows, responses=responses, home=home, actor=actor, db=db,
                              refusals=refusals, bound=bound)
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
    (root / "required-mentions.json").write_text(json.dumps({"channels": {CHANNEL: {
        "ceo": {"handle": "ceo", "pubkey": OTHER}, "peer": {"handle": "cto-repo-factory", "pubkey": SIGNER}}}}))
    env = {
        "CANONICAL_SURFACE_URL": _URL, "API_SERVER_KEY": gw.key, "CANONICAL_BINDING": "canonical",
        "CANONICAL_AUTHOR_ID": SIGNER, "CANONICAL_CHANNEL_ID": CHANNEL, "CANONICAL_FORWARD": "live",
        "CANONICAL_BUSY_RETRY_SECS": "0", "CANONICAL_BUSY_DEADLINE_SECS": "5",
        "CANONICAL_SHIM_LOG": str(root / "shim.log"), "CANONICAL_QUARANTINE_LOG": str(root / "q.jsonl"),
        "CANONICAL_CANARY_FILE": str(root / "canary.json"), "CANONICAL_PUBLISHED_DIR": str(root / "published"),
        "CANONICAL_MENTIONS_FILE": str(root / "required-mentions.json"), "CANONICAL_BUZZ_CLI": str(root / "buzz"),
    }
    for name, value in env.items():
        if "/" in value and name != "CANONICAL_SURFACE_URL" and _is_live(value):
            pytest.fail(f"{name} names a live path: {value}")
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location("shim_d2_o4_under_test", str(_SHIM))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "write", lambda payload: None)
    module.posts = lambda: [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    module.root = root
    return module


def _drive(shim, rpc_id=4):
    shim.handle({"jsonrpc": "2.0", "id": rpc_id, "method": "session/prompt",
                 "params": bp.prompt_params("s", channel_prompt("x" * 300))})


def _held(gw):
    """Every copy of the answer text the receipt store holds for the event."""
    texts = []
    for value in gw.rows("canonical-receipt:").values():
        response = json.loads(value).get("response") or {}
        texts.append(response.get("terminal_text"))
    for value in gw.rows("canonical-answer-kept:").values():
        texts.append(((json.loads(value).get("answer") or {}).get("response") or {}).get("terminal_text"))
    fallback = gw.home / "canonical-answer-kept"
    if fallback.exists():
        for path in fallback.iterdir():
            parsed = json.loads(path.read_bytes())
            texts.append(((parsed.get("answer") or {}).get("response") or {}).get("terminal_text"))
    return [text for text in texts if text]


def _receipt(gw):
    [value] = gw.rows("canonical-receipt:").values()
    return json.loads(value)


def _codes(gw):
    return [(status, body["error"]["code"] if status != 200 else "text") for status, body in gw.responses]


def _attributed_and_submitted(gw, shim):
    """Reach, asserted: the shim's attribution guard (#91 r1) took the prompt as one event and the
    gateway received it. A prompt the guard refuses first never reaches D2's failure point."""
    assert f"base={EVENT[:12]}" in (shim.root / "shim.log").read_text(), "the attribution guard refused it first"
    assert gw.responses, "nothing was submitted to the gateway"


def _reached_the_refusal(gw, shim):
    """Reach, asserted: D2's failure point. The actor ran once and the answer it returned reached the
    selector, which refused the turn's shape with that answer in hand."""
    _attributed_and_submitted(gw, shim)
    assert len(gw.runs) == 1 and gw.refusals == [("canonical_turn_refused", FINAL)]


def test_the_witness_prompt_is_valid_under_the_shims_own_rules(shim):
    """Reach precondition, asserted against the shim under test (#91 r1-r3): the prompt is attributed by
    ``envelope`` to its own event, and its [Context] is in exact form and bound to the event's Tags."""
    prompt = bp.prompt_params("s", channel_prompt("x" * 300))["prompt"]
    assert shim.envelope(prompt)[:3] == (SIGNER, CHANNEL, EVENT)
    assert shim.notice_destination(shim.prompt_blocks(prompt)) == (CHANNEL, EVENT, None)


# ── W2 (ported): the live 813415c3 shape ─────────────────────────────────────────────────────────────


def test_w2_the_live_nudged_turn_delivers_the_completed_answer(gw, shim):
    gw.actor.shape = "nudged"
    _drive(shim)
    _delivered_once(gw, shim)


@pytest.mark.parametrize("shape", ["unanswered_tool", "unanswered_tool_marked"])
def test_a_proven_answer_is_kept_not_delivered_and_the_refusal_stays_the_events_terminal(gw, shim, shape):
    gw.actor.shape = shape
    _drive(shim)

    _reached_the_refusal(gw, shim)
    # Kept beside the receipt, by this claim, whole; never the receipt's (delivered) answer.
    [kept] = [json.loads(v) for v in gw.rows("canonical-answer-kept:").values()]
    [claim] = [json.loads(v) for v in gw.rows("canonical-claim:").values()]
    assert kept["owner"] == claim["owner"] and kept["answer"]["response"]["terminal_text"] == FINAL
    assert kept["unrecorded"]["stage"] == "turn.refused_bound_answer"
    receipt = _receipt(gw)
    assert (receipt["state"], receipt["failure"]["code"], receipt["owner"]) == (
        "terminal", "canonical_turn_refused", claim["owner"])
    assert "response" not in receipt
    # The request answers the kept state, never success; the sender is told once, without the answer.
    assert _codes(gw) == [(409, "canonical_answer_unrecorded")]
    assert gw.responses[0][1]["error"]["answer_kept"] is True
    [post] = shim.posts()
    assert "canonical_answer_unrecorded" in post["stdin"] and "answer_kept=true" in post["stdin"]
    assert FINAL not in post["stdin"] and post["argv"][post["argv"].index("--reply-to") + 1] == EVENT

    # A replay never runs the turn again, never delivers the kept answer and posts nothing more.
    _drive(shim, rpc_id=5)
    assert len(gw.runs) == 1 and len(shim.posts()) == 1
    assert _codes(gw) == [(409, "canonical_answer_unrecorded")] * 2
    assert gw.bound == [FINAL]  # the binding check proved it, once; the replay never reached the actor


@pytest.mark.parametrize("shape", ["foreign_peer_row", "foreign_marked"])
def test_an_answer_not_provably_this_turns_is_not_kept_and_the_failure_is_noticed(gw, shim, shape):
    gw.actor.shape = shape
    _drive(shim)

    _attributed_and_submitted(gw, shim)
    assert len(gw.runs) == 1
    assert gw.rows("canonical-answer-kept:") == {} and _held(gw) == []
    assert not (gw.home / "canonical-answer-kept").exists()
    receipt = _receipt(gw)
    assert (receipt["state"], receipt["failure"]["code"]) == ("terminal", "canonical_turn_refused")
    assert _codes(gw) == [(409, "canonical_event_failed")]
    # canonical_event_failed no longer ends in silence: one notice, stating the gateway's own scope.
    [post] = shim.posts()
    assert "canonical_event_failed" in post["stdin"] and "side_effects=none" in post["stdin"]
    assert FINAL not in post["stdin"]
    assert json.loads((shim.root / "published" / EVENT).read_text())["kind"] == "notice"

    _drive(shim, rpc_id=5)  # buzz-acp redelivers: the same failure, never a second post or run
    assert len(gw.runs) == 1 and len(shim.posts()) == 1
    assert gw.bound == []  # the shared provenance check refused it before recording


def _delivered_once(gw, shim):
    _attributed_and_submitted(gw, shim)
    assert len(gw.runs) == 1 and gw.refusals == [] and gw.bound == []
    assert _codes(gw) == [(200, "text")]
    assert _held(gw) == [FINAL] and gw.rows("canonical-answer-kept:") == {}
    assert [post["stdin"] for post in shim.posts()] == [f"@cto-repo-factory {FINAL}"]
    _drive(shim, rpc_id=5)
    assert len(gw.runs) == 1 and len(shim.posts()) == 1
    assert _codes(gw) == [(200, "text")] * 2


@pytest.mark.parametrize("shape", ["plain", "unnudged_retry", "unnudged_retry_marked", "nudged_marked",
                                   "other_user_row", "final_mismatch"])
def test_completed_answer_is_recorded_and_posted_once(gw, shim, shape):
    gw.actor.shape = shape
    _drive(shim)
    _delivered_once(gw, shim)


# ── answer_kept only once the answer read back ───────────────────────────────────────────────────────


def _no_fallback(*_args, **_kwargs):
    raise OSError(28, "No space left on device")


@pytest.mark.parametrize("how", ["store_refuses", "store_reads_back_other_bytes", "fallback_reads_back_other"])
def test_a_bound_answer_that_did_not_read_back_is_reported_not_kept(gw, shim, monkeypatch, how):
    claim, get = gw.db.claim_meta_once, gw.db.get_meta

    def claim_meta_once(key, *args, **kwargs):
        if key.startswith("canonical-answer-kept:") and how in ("store_refuses", "fallback_reads_back_other"):
            raise sqlite3.OperationalError("database is locked")
        return claim(key, *args, **kwargs)

    def get_meta(key, *args, **kwargs):
        value = get(key, *args, **kwargs)
        if key.startswith("canonical-answer-kept:") and how == "store_reads_back_other_bytes" and value:
            return value.replace(FINAL, "another answer")
        return value

    monkeypatch.setattr(gw.db, "claim_meta_once", claim_meta_once)
    monkeypatch.setattr(gw.db, "get_meta", get_meta)
    if how == "fallback_reads_back_other":
        # The file is written, but what a replay reads first for the event is another claim's record.
        monkeypatch.setattr(canonical_answer_fallback, "kept_for", lambda home, digest: [("0" * 32, b"{}")])
    else:
        monkeypatch.setattr(canonical_answer_fallback, "keep", _no_fallback)
    _drive(shim)

    _reached_the_refusal(gw, shim)
    assert _codes(gw) == [(409, "canonical_answer_unrecorded")]
    assert gw.responses[0][1]["error"]["answer_kept"] is False
    assert _receipt(gw)["failure"]["code"] == "canonical_turn_refused"
    [post] = shim.posts()
    assert "answer_kept=false" in post["stdin"]
    assert gw.bound == [FINAL]  # proven, yet not reported kept: it did not read back


def test_a_kept_failure_without_answer_text_never_reads_as_a_kept_answer():
    fingerprint = "f" * 64
    failure = {"code": "canonical_answer_unrecorded", "side_effects": "none", "stage": "record.compare_and_set"}
    record = json.dumps({"v": 1, "state": "kept", "fingerprint": fingerprint, "owner": "o", "receipt": "r",
                         "unrecorded": failure, "answer": {"failure": {"code": "canonical_turn_failed",
                                                                       "side_effects": "none"}}})
    decoded = CanonicalReceiptCoordinator._decode_kept(record, fingerprint)
    assert (decoded.status, decoded.failure["answer_kept"]) == ("unrecorded", False)
    assert CanonicalReceiptCoordinator._kept_text(record, fingerprint) is None


# ── the shim's notice for terminal codes ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("code", "error", "scope"), [
    ("canonical_event_failed", {"side_effects": "none"}, "none"),
    ("canonical_event_failed", {"side_effects": "attempted"}, "attempted"),
    ("canonical_event_failed", {}, "unknown"),  # never assumed
    ("canonical_event_uncertain", {}, "unknown"),
    ("canonical_event_conflict", {"side_effects": "none; ok"}, "unknown"),  # not a stated scope
])
def test_each_post_claim_terminal_code_has_a_notice_that_assumes_nothing(shim, code, error, scope):
    notice = shim.refusal_notice(409, code, {"code": code, **error}, EVENT, 300)

    assert notice is not None and code in notice and f"side_effects={scope}" in notice
    assert f"(event {EVENT})" in notice and "다시 실행되지 않으며" in notice


@pytest.mark.parametrize(("status", "code"), [(409, "canonical_turn_busy"), (409, "canonical_binding_stale"),
                                              (403, "canonical_binding_unknown"), (401, None),
                                              (503, "canonical_unavailable")])
def test_control_a_refusal_raised_before_the_claim_keeps_its_existing_silence(shim, status, code):
    assert shim.refusal_notice(status, code, {"code": code} if code else {}, EVENT, 300) is None


_PRINCIPAL_NOTICE = (
    "이 메시지는 CEO 에게 전달되지 않았습니다. 이 방에서 이 발신자의 메시지는 CEO 가 받지 않도록 "
    "설정되어 있습니다. 턴은 실행되지 않았고 자동 재전송도 없습니다. (event {event_id})")


def test_a_403_principal_rejection_posts_one_fixed_notice_and_a_redelivery_posts_nothing_more(
        gw, shim, monkeypatch):
    """HTTP 403 canonical_principal_rejected tells the sender once, in fixed text. The gateway body
    (its prose, a binding name, a key) is not copied, the refusal is not retried, and a second
    delivery of the same event does not post again."""
    expected = _PRINCIPAL_NOTICE.format(event_id=EVENT)
    secret = "sk-live-" + gw.key
    binding = "acp-canonical-ceo"
    poisoned = {"code": "canonical_principal_rejected", "message": "Canonical request rejected.",
                "binding": binding, "key": secret}
    notice = shim.refusal_notice(403, "canonical_principal_rejected", poisoned, EVENT, 300)
    assert notice == expected
    assert secret not in notice and gw.key not in notice and binding not in notice
    assert "Canonical request rejected" not in notice

    calls = []

    def urlopen(request, timeout=None):
        calls.append(1)
        raise urllib.error.HTTPError(request.full_url, 403, "refused", {}, io.BytesIO(json.dumps(
            {"error": poisoned}).encode()))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    _drive(shim)

    assert calls == [1]  # 403 leaves the submit loop; it is not the busy retry
    [post] = shim.posts()
    assert post["stdin"] == f"@cto-repo-factory {expected}"
    assert secret not in json.dumps(post) and gw.key not in json.dumps(post)
    assert binding not in post["stdin"] and "Canonical request rejected" not in post["stdin"]
    journal = json.loads((shim.root / "published" / EVENT).read_text())
    assert (journal["kind"], journal["sent"], journal["event_id"]) == ("notice", True, EVENT)

    _drive(shim, rpc_id=5)
    assert shim.posts() == [post]


def test_the_bridges_own_signer_is_refused_locally_and_never_submitted(gw, shim):
    """A notice this bridge signed comes back as its own signer. That is refused before submit, so
    it cannot be refused again at the gateway and post another notice."""
    own = bp.format_prompt(CHANNEL, "repo-factory", [bp.Ev(
        EVENT, OTHER, 1_792_324_251, "the bridge's own notice", [["h", CHANNEL], ["p", OTHER]],
        prompt_tag="message")],
        {OTHER: "ceo", SIGNER: "cto-repo-factory"})
    blocks = bp.prompt_params("s", own)["prompt"]
    assert shim.envelope(blocks)[:3] == (OTHER, CHANNEL, EVENT)

    shim.handle({"jsonrpc": "2.0", "id": 9, "method": "session/prompt", "params": bp.prompt_params("s", own)})

    assert gw.responses == [] and gw.runs == [] and shim.posts() == []
    assert "refused locally: authored by this bridge's own signer" in (shim.root / "shim.log").read_text()


def test_a_submission_the_gateway_never_answered_ends_in_a_notice_that_states_no_outcome(gw, shim, monkeypatch):
    def unanswered(request, timeout=None):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(urllib.request, "urlopen", unanswered)
    _drive(shim)

    log = (shim.root / "shim.log").read_text()  # reach: attributed, and the submission was attempted
    assert f"base={EVENT[:12]}" in log and f"prompt {EVENT} failed: URLError" in log
    [post] = shim.posts()
    assert "URLError" in post["stdin"] and "알 수 없고" in post["stdin"] and f"(event {EVENT})" in post["stdin"]
    assert json.loads((shim.root / "published" / EVENT).read_text())["kind"] == "notice"
