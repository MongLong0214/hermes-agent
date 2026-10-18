"""The bridge tells the sender when the canonical surface will not take or deliver an event.

Regression for agent-control-plane #1111 (a thread reply over the gateway's 16,384-char text cap was
refused and the sender was told nothing) and for the sender notice #1110 needs (a turn that ran but
whose answer could not be recorded ends in ``canonical_answer_unrecorded``).

Each case drives the real shim (``bridge/canonical/acp-canonical-shim.py``, or ``SHIM_PATH``) through
``handle()``; its HTTP request goes to the real canonical handler, coordinator and SQLite of an
in-process gateway under a tmp home. The Buzz wrapper is a stand-in that records what was posted.
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

from tests.hermes_home_pin import import_home, pinned_hermes_home, tmp_root

try:
    import pwd  # the account database names the real home even when HOME is redirected
except ImportError:  # not POSIX
    pwd = None

# HOME, HERMES_HOME and the context-local home override name a fresh tmp dir while this module
# imports Hermes and while its tests run. A path that still resolves into the real Hermes home fails
# before anything is written.
_REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir) if pwd is not None else Path.home()
_LIVE_ROOT = (_REAL_HOME / ".hermes").resolve()
_PINNED_ROOT = tmp_root("hermes-shim-cap-")  # removed at exit by this process only
if _PINNED_ROOT.is_relative_to(_LIVE_ROOT) or _LIVE_ROOT.is_relative_to(_PINNED_ROOT):
    raise RuntimeError(f"the pinned test home {_PINNED_ROOT} overlaps the real home")

# The imports run pinned to the tmp home all pinned modules import under, so a path they cache names
# no module's test home; the pin is put back (in ``finally``) as soon as they have run: collecting
# this module leaves no home, override or home cache behind for the modules collected after it.
with pinned_hermes_home(import_home(), importer=__name__):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    from gateway import canonical_surface
    from gateway.canonical_surface import CanonicalSurfaceBinding
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource


@pytest.fixture(autouse=True, scope="module")
def _pin_the_module_home():
    """The same pin while this module's tests run, owned by this fixture and put back on teardown."""
    with pinned_hermes_home(_PINNED_ROOT):
        yield


def _is_live(path) -> bool:
    return Path(path).expanduser().resolve().is_relative_to(_LIVE_ROOT)


_SHIM = Path(os.environ.get("SHIM_PATH") or
             Path(__file__).resolve().parents[2] / "bridge" / "canonical" / "acp-canonical-shim.py")
_ROUTE = "/v1/canonical-surface/events"
# Nothing listens on the discard port; a request the test does not route never reaches a gateway.
_URL = "http://127.0.0.1:9"
_PATH_VARS = ("CANONICAL_SHIM_LOG", "CANONICAL_QUARANTINE_LOG", "CANONICAL_CANARY_FILE",
              "CANONICAL_PUBLISHED_DIR", "CANONICAL_MENTIONS_FILE", "CANONICAL_BUZZ_CLI")
LIMIT = 16_384

SIGNER = "73bd7a526adb00a06ed2ec820701e2f761d00cc0dc0ed1fb051f8923d6fdc75a"  # the sender (CTO)
OTHER = "e79dda8ddd8f994b5758a2f8eed4bc76b6b9c418d16ed016f9e437ee7c35ec4d"  # this bridge's signer
CHANNEL = "b9cda259-f165-45ce-afaf-3e8f078a144a"
EVENT = "e" * 64
ROOT = "a" * 64
PARENT = "d" * 64


def thread_prompt(entries, body, *, sender=SIGNER):
    """buzz-acp's prompt for a reply inside a thread: context (target, root, reply correlation), the
    quoted history oldest first, then the event section with the current message."""
    lines = [
        "[Context]", "Scope: thread", f"Channel: repo-factory (#{CHANNEL})", f"Thread root: {ROOT}",
        f"Parent: {PARENT}",
        f"IMPORTANT: For ordinary replies in this turn, use `--reply-to {ROOT}` on `buzz messages send`.",
        f"[Thread Context ({len(entries)} of {len(entries)} messages)]",
    ]
    for n, text in enumerate(entries, 1):
        who, hexid = ("ceo", OTHER) if n % 2 else ("cto-repo-factory", SIGNER)
        lines.append(f"[{n}] {who} ({hexid}) (2026-10-16T{n:02d}:00:00+00:00): {text}")
    lines += [
        "", "[Buzz event: @mention]", f"Event ID: {EVENT}", f"Channel: repo-factory (#{CHANNEL})", "Kind: 9",
        f"From: cto-repo-factory (npub: npub1x, hex: {sender})", "Time: 2026-10-16T00:51:51+00:00",
        f"Content: {body}", "",
        f'Tags: [["h","{CHANNEL}"],["e","{ROOT}","","root"],["e","{PARENT}","","reply"]]',
    ]
    return "\n".join(lines)


def entry(n, size):
    return f"@cto-repo-factory message {n}\nhistory-{n:02d} " + "x" * size


class _Actor:
    """The bound cached agent, with no model: run n answers ``answer #<n>``."""

    compression_in_place = True

    def __init__(self, session_id, db, runs):
        self.session_id = session_id
        self._session_db = db
        self.runs = runs
        self.during_turn = None

    def interrupt(self, text=None):
        pass

    def release_clients(self):
        pass

    def run_conversation(self, text, *, conversation_history, task_id, **kwargs):
        self.runs.append(text)
        answer = f"answer #{len(self.runs)}"
        if self.during_turn is not None:
            self.during_turn()
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
    """A real runner, coordinator and canonical HTTP handler under a tmp home. The shim's ``urlopen``
    is routed to that handler, carrying the shim's own body and Authorization header."""
    home, user_home = tmp_path / "hermes-home", tmp_path / "user"
    for path in (home, user_home):
        if _is_live(path):  # before anything is created or opened
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
    actor = _Actor(entry_.session_id, db, runs)
    with runner._agent_cache_lock:
        runner._agent_cache[entry_.session_key] = (actor, "exact", 0, entry_.session_id)
    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", lambda *a, **kw: pytest.fail(
        "canonical ingress built an agent while one was cached for the session"))
    key = secrets.token_hex(32)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key}))
    adapter.gateway_runner = runner
    [handler] = [h for method, path, h in adapter._http_route_table() if (method, path) == ("POST", _ROUTE)]
    received = []

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
        received.append(request.data)
        status, body = asyncio.run(post(request.data, request.get_header("Authorization")))
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "refused", {}, io.BytesIO(body))
        return _Response(body)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    def rows(prefix):
        with contextlib.closing(sqlite3.connect((home / "state.db").resolve().as_uri() + "?mode=ro",
                                                uri=True)) as conn:
            return dict(conn.execute("SELECT key, value FROM state_meta WHERE key LIKE ?", (prefix + "%",)))

    def take_claim():
        """Another holder takes the event's receipt row mid-turn: S14b's record compare-and-set miss."""
        [(row, value)] = rows("canonical-receipt:").items()
        with contextlib.closing(sqlite3.connect(home / "state.db", timeout=10)) as conn:
            conn.execute("UPDATE state_meta SET value = ? WHERE key = ?",
                         (json.dumps({**json.loads(value), "owner": "another-holder"}), row))
            conn.commit()

    def kept():
        values = list(rows("canonical-answer-kept:").values())
        return json.loads(values[0])["answer"]["response"]["terminal_text"] if values else None

    state = SimpleNamespace(key=key, runs=runs, actor=actor, received=received, take_claim=take_claim,
                            kept=kept, sent=lambda: [json.loads(raw) for raw in received], db=db, home=home,
                            runner=runner, entry=entry_, rows=rows)
    try:
        yield state
    finally:
        reset_hermes_home_override(override)
        runner.session_store.close_all_db_handles()
        adapter._response_store.close()


def _shim_env(root, key):
    return {
        "CANONICAL_SURFACE_URL": _URL, "API_SERVER_KEY": key, "CANONICAL_BINDING": "canonical",
        "CANONICAL_AUTHOR_ID": SIGNER, "CANONICAL_CHANNEL_ID": CHANNEL, "CANONICAL_FORWARD": "live",
        "CANONICAL_BUSY_RETRY_SECS": "0", "CANONICAL_BUSY_DEADLINE_SECS": "5",
        "CANONICAL_SHIM_LOG": str(root / "shim.log"), "CANONICAL_QUARANTINE_LOG": str(root / "q.jsonl"),
        "CANONICAL_CANARY_FILE": str(root / "canary.json"), "CANONICAL_PUBLISHED_DIR": str(root / "published"),
        "CANONICAL_MENTIONS_FILE": str(root / "required-mentions.json"), "CANONICAL_BUZZ_CLI": str(root / "buzz"),
    }


def _load_shim(monkeypatch, env):
    """The shim module under *env*, refused before anything is set, loaded or written when a path it
    writes or runs is in the live Hermes home, or when it would post anywhere but the routed URL."""
    for name in _PATH_VARS:
        if not env.get(name) or _is_live(env[name]):
            pytest.fail(f"{name} names a live path or nothing: {env.get(name)}")
    if env.get("CANONICAL_SURFACE_URL") != _URL:
        pytest.fail(f"the shim would post to {env.get('CANONICAL_SURFACE_URL')}")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location("shim_text_cap_under_test", str(_SHIM))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "write", lambda payload: None)  # buzz-acp's stdout, not a post
    return module


@pytest.fixture
def shim(gw, tmp_path, monkeypatch):
    root = tmp_path / "bridge"
    root.mkdir()
    calls = root / "buzz-calls.jsonl"
    (root / "buzz").write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        f"open({str(calls)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], 'stdin': sys.stdin.read()}}) + '\\n')\n"
        "code = int(os.environ.get('FAKE_BUZZ_EXIT', '0'))\n"
        "if code: sys.stderr.write('relay unreachable')\n"
        "sys.exit(code)\n")
    (root / "buzz").chmod(0o755)
    (root / "required-mentions.json").write_text(json.dumps({"channels": {CHANNEL: {
        "ceo": {"handle": "ceo", "pubkey": OTHER}, "peer": {"handle": "cto-repo-factory", "pubkey": SIGNER}}}}))
    module = _load_shim(monkeypatch, _shim_env(root, gw.key))
    module.posts = lambda: [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    module.journal = lambda: (json.loads((root / "published" / EVENT).read_text())
                              if (root / "published" / EVENT).exists() else None)
    module.log_text = lambda: (root / "shim.log").read_text() if (root / "shim.log").exists() else ""
    return module


def drive(shim, text, rpc_id=1):
    shim.handle({"jsonrpc": "2.0", "id": rpc_id, "method": "session/prompt",
                 "params": {"sessionId": "s", "prompt": [{"type": "text", "text": text}]}})


def _reply(post):
    argv = post["argv"]
    return argv[argv.index("--channel") + 1], argv[argv.index("--reply-to") + 1], post["stdin"]


# ── Witnesses ────────────────────────────────────────────────────────────────────────────────────


def test_a_thread_reply_over_the_cap_is_delivered_with_its_oldest_context_trimmed(gw, shim):
    body = "@ceo [CTO→CEO] blocker: the decision text stays whole\n" + "- evidence line\n" * 40
    text = thread_prompt([entry(n, 1_065) for n in range(1, 14)], body)
    assert LIMIT < len(text) < 18_000  # the 17,475-char shape refused on 2026-10-16

    drive(shim, text)

    [sent] = gw.sent()
    assert len(sent["text"]) <= LIMIT
    # The current message, its target and its reply correlation arrive whole and unchanged.
    assert sent["text"].endswith(text[text.index("[Buzz event:"):])
    assert sent["text"].startswith(text[:text.index("[Thread Context (")])
    assert (sent["event_id"], sent["author_id"], sent["channel_id"]) == (EVENT, SIGNER, CHANNEL)
    # Oldest history went first.
    assert "history-01 " not in sent["text"] and "history-13 " in sent["text"]
    assert len(gw.runs) == 1
    [post] = shim.posts()
    assert _reply(post) == (CHANNEL, EVENT, "@cto-repo-factory answer #1")


def test_a_current_message_over_the_cap_on_its_own_is_refused_with_a_visible_notice(gw, shim):
    text = thread_prompt([entry(n, 200) for n in range(1, 4)], "@ceo " + "y" * LIMIT + " BODY-MARK")

    drive(shim, text)

    assert gw.received == [] and gw.runs == []  # refused, never truncated and sent
    [post] = shim.posts()
    channel, reply_to, notice = _reply(post)
    assert (channel, reply_to) == (CHANNEL, EVENT)
    assert EVENT in notice and str(LIMIT) in notice and "전달되지 않았습니다" in notice
    assert "BODY-MARK" not in notice and "y" * 32 not in notice and len(notice) < 1_000


def test_a_gateway_400_over_its_cap_tells_the_sender_once_and_nothing_is_resent(gw, shim, monkeypatch):
    # The gateway's cap is below the one the bridge fits to: the gateway, not the bridge, refuses.
    monkeypatch.setattr(canonical_surface, "_MAX_TEXT_CHARS", 8_192)
    text = thread_prompt([entry(1, 100)], "@ceo " + "z" * 9_000 + " BODY-MARK")

    drive(shim, text)

    assert len(gw.received) == 1 and gw.runs == []
    [post] = shim.posts()
    channel, reply_to, notice = _reply(post)
    assert (channel, reply_to) == (CHANNEL, EVENT)
    assert "HTTP 400 canonical_invalid_request" in notice and f"제출 {len(text)}자" in notice
    assert EVENT in notice and "BODY-MARK" not in notice and "z" * 32 not in notice
    assert len(notice) < 1_000 and gw.key not in notice
    assert shim.journal()["sent"] is True and shim.journal()["kind"] == "notice"

    drive(shim, text, rpc_id=2)  # buzz-acp redelivers the event
    assert len(shim.posts()) == 1  # told once


def test_an_unrecorded_answer_tells_the_sender_its_state_never_the_answer_and_never_reruns(gw, shim):
    """#1110 with S14b: the turn ran, another holder took its claim, and the gateway kept the answer
    and answered 409 canonical_answer_unrecorded. The sender is told, once; nothing is resent."""
    gw.actor.during_turn = gw.take_claim

    drive(shim, thread_prompt([entry(1, 100)], "@ceo please decide"))
    gw.actor.during_turn = None

    assert len(gw.received) == 1 and len(gw.runs) == 1  # no busy retry, no resend
    assert gw.kept() == "answer #1"  # the answer stays held at the gateway
    [post] = shim.posts()
    channel, reply_to, notice = _reply(post)
    assert (channel, reply_to) == (CHANNEL, EVENT)
    assert EVENT in notice and "canonical_answer_unrecorded" in notice and "delivery=not_delivered" in notice
    assert "answer #" not in notice and gw.key not in json.dumps(post)
    assert "answer #" not in shim.log_text()

    drive(shim, thread_prompt([entry(1, 100)], "@ceo please decide"), rpc_id=2)  # buzz-acp redelivers
    assert len(gw.runs) == 1 and len(shim.posts()) == 1


@pytest.mark.parametrize("case", ["invalid_request", "answer_unrecorded"])
def test_a_failed_notice_is_journaled_as_a_failure_and_never_retried_or_resent(gw, shim, monkeypatch, case):
    monkeypatch.setenv("FAKE_BUZZ_EXIT", "1")
    if case == "invalid_request":
        monkeypatch.setattr(canonical_surface, "_MAX_TEXT_CHARS", 8_192)
        text = thread_prompt([entry(1, 100)], "@ceo " + "z" * 9_000)
    else:
        gw.actor.during_turn = gw.take_claim
        text = thread_prompt([entry(1, 100)], "@ceo please decide")

    drive(shim, text)
    gw.actor.during_turn = None

    assert len(gw.received) == 1  # the event was not resent because its notice failed
    assert len(shim.posts()) == 1  # one notice attempt
    journal = shim.journal()
    assert journal is not None and (journal["sent"], journal["state"], journal["kind"]) == (
        False, "send_failed", "notice")
    assert "notice NOT posted (no retry)" in shim.log_text() and "POSTED" not in shim.log_text()

    drive(shim, text, rpc_id=2)
    assert len(shim.posts()) == 1 and len(gw.runs) == (1 if case == "answer_unrecorded" else 0)


def test_a_refusal_is_read_by_its_code_so_an_unrecorded_answer_is_never_retried_as_busy(shim, monkeypatch):
    """The busy retry keys on the gateway's code, not on a substring of the body."""
    posts = []

    def urlopen(request, timeout=None):
        posts.append(request.full_url)
        if len(posts) > 1:
            raise OSError("resent")
        raise urllib.error.HTTPError(request.full_url, 409, "refused", {}, io.BytesIO(json.dumps({"error": {
            "code": "canonical_answer_unrecorded", "message": "not canonical_turn_busy",
            "delivery": "not_delivered", "answer_kept": True}}).encode()))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    drive(shim, thread_prompt([entry(1, 100)], "@ceo please decide"))
    assert len(posts) == 1
    [post] = shim.posts()
    assert "canonical_answer_unrecorded" in post["stdin"] and "answer_kept=true" in post["stdin"]


def _identical_answer_by_another_holder(gw):
    """Another holder takes the claim mid-turn and records the very answer this turn produces."""
    def finish():
        [(row, value)] = gw.rows("canonical-receipt:").items()
        pending = json.loads(value)
        # Byte for byte the terminal this turn would write without its claim binding.
        terminal = json.dumps({"v": 1, "state": "terminal", "fingerprint": pending["fingerprint"],
                               "response": {"binding_name": "canonical", "terminal_text": "answer #1"}},
                              ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        with contextlib.closing(sqlite3.connect(gw.home / "state.db", timeout=10)) as conn:
            conn.execute("UPDATE state_meta SET value = ? WHERE key = ?",
                         (json.dumps({**pending, "owner": "another-holder"}), row))
            conn.execute("UPDATE state_meta SET value = ? WHERE key = ?", (terminal, row))
            conn.commit()
    return finish


def _refuse_every_store(gw, monkeypatch):
    """The state.db refuses the kept answer, and the local fallback has no space for it."""
    from gateway import canonical_answer_fallback

    original = gw.db.claim_meta_once

    def claim(key, *args, **kwargs):
        if key.startswith("canonical-answer-kept:"):
            raise sqlite3.OperationalError("database is locked")
        return original(key, *args, **kwargs)
    monkeypatch.setattr(gw.db, "claim_meta_once", claim)

    def no_space(*_args, **_kwargs):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(canonical_answer_fallback, "keep", no_space)


def _replaced_after_the_turn(gw):
    def replace():
        replacement = _Actor(gw.entry.session_id, gw.db, gw.runs)
        with gw.runner._agent_cache_lock:
            gw.runner._agent_cache[gw.entry.session_key] = (replacement, "exact", 0, gw.entry.session_id)
    return replace


@pytest.mark.parametrize("state", ["another_holders_identical_answer", "rejected_after_the_actor_returned",
                                   "kept_by_no_store"])
def test_each_unrecorded_state_tells_the_sender_once_never_the_answer_and_nothing_reruns(gw, shim, monkeypatch, state):
    """#1110 r1: an answer another holder's identical bytes would have passed for, one the turn was
    rejected after returning, and one no store could keep each end in 409 canonical_answer_unrecorded.
    The sender is told which (kept or not preserved), once; no answer is posted; nothing is resent."""
    if state == "kept_by_no_store":
        _refuse_every_store(gw, monkeypatch)
        gw.actor.during_turn = gw.take_claim
    elif state == "another_holders_identical_answer":
        gw.actor.during_turn = _identical_answer_by_another_holder(gw)
    else:
        gw.actor.during_turn = _replaced_after_the_turn(gw)
    text = thread_prompt([entry(1, 100)], "@ceo please decide")

    drive(shim, text)
    gw.actor.during_turn = None

    assert len(gw.received) == 1 and len(gw.runs) == 1  # no busy retry, no resend
    [post] = shim.posts()
    channel, reply_to, notice = _reply(post)
    assert (channel, reply_to) == (CHANNEL, EVENT)
    assert EVENT in notice and "canonical_answer_unrecorded" in notice and "delivery=not_delivered" in notice
    if state == "kept_by_no_store":
        assert "answer_kept=false" in notice and "preservation=failed" in notice and "보존하지도 못했습니다" in notice
        assert "answer NOT preserved" in shim.log_text() and gw.kept() is None
    else:
        assert "answer_kept=true" in notice and "preservation=failed" not in notice
    assert "answer #" not in notice and gw.key not in json.dumps(post) and "answer #" not in shim.log_text()
    assert shim.journal()["kind"] == "notice"

    drive(shim, text, rpc_id=2)  # buzz-acp redelivers
    assert len(gw.runs) == 1 and len(shim.posts()) == 1


# ── Controls ─────────────────────────────────────────────────────────────────────────────────────


def test_control_a_short_thread_reply_reaches_the_gateway_byte_for_byte(gw, shim):
    text = thread_prompt([entry(1, 100), entry(2, 100)], "@ceo short reply")

    drive(shim, text)

    assert gw.received == [json.dumps({"binding": "canonical", "event_id": EVENT, "author_id": SIGNER,
                                       "channel_id": CHANNEL, "text": text}).encode()]
    [post] = shim.posts()
    assert _reply(post) == (CHANNEL, EVENT, "@cto-repo-factory answer #1")


def test_control_any_other_refusal_still_posts_nothing(gw, shim):
    drive(shim, thread_prompt([entry(1, 100)], "@ceo hello", sender="f" * 64))  # not an allowed author

    assert len(gw.received) == 1 and gw.runs == []
    assert shim.posts() == [] and shim.journal() is None


def test_canary_a_live_path_or_surface_fails_before_anything_is_loaded_or_written(tmp_path, monkeypatch):
    probe = _LIVE_ROOT / f".shim-canary-{secrets.token_hex(8)}"
    for name, value in (("CANONICAL_SHIM_LOG", str(probe / "shim.log")),
                        ("CANONICAL_PUBLISHED_DIR", str(probe / "published")),
                        ("CANONICAL_SURFACE_URL", "http://127.0.0.1:8642")):
        env = {**_shim_env(tmp_path, "test-key"), name: value}
        with pytest.raises(pytest.fail.Exception):
            _load_shim(monkeypatch, env)
        assert os.environ.get(name) != value
    assert not probe.exists()
