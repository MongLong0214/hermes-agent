"""A canonical turn whose answer exists but whose terminal record does not land (#1110).

The coordinator claims an event's receipt row, runs the turn once, and records the terminal with a
compare-and-set on the pending row it wrote. When that compare-and-set changes no row, raises, or the
row moved on, the turn already produced its answer. These counterexamples pin what happens to that
answer, to its delivery (the HTTP answer the bridge posts) and to the receipt:

- the claim taken by another holder mid-turn, the row deleted (and restored), an answer already
  recorded (by someone else, or our own write that reported a miss), a write that raised;
- a concurrent retry of the same event, and a gateway restart between the turn and the record.

Never: a second run of the turn, an overwrite of another holder's row or answer, a success answer
the receipt does not bind, or an answer that is neither delivered nor reported as a failure.
"""

import asyncio
import atexit
import contextlib
import hashlib
import json
import logging
import os
import secrets
import shutil
import sqlite3
import stat
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.hermes_home_pin import import_home, pinned_hermes_home

try:
    import pwd  # the account database names the real home even when HOME is redirected
except ImportError:  # not POSIX
    pwd = None

# Pinned for every Hermes import of this module (its import-time paths and caches) and, by a
# module fixture, for every resolver and subprocess of its tests: HOME and HERMES_HOME name a fresh
# tmp dir and the context-local home override (which wins over both) names that same tmp home. A
# path that still resolves into the real Hermes home fails the test before anything is written (see
# ``_never_a_live_path``).
_REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir) if pwd is not None else Path.home()
_LIVE_ROOT = (_REAL_HOME / ".hermes").resolve()
_LIVE_CONFIG = _LIVE_ROOT / "config.yaml"
_PINNED_ROOT = Path(tempfile.mkdtemp(prefix="hermes-record-cas-")).resolve()
if _PINNED_ROOT.is_relative_to(_LIVE_ROOT) or _LIVE_ROOT.is_relative_to(_PINNED_ROOT):
    raise RuntimeError(f"the pinned test home {_PINNED_ROOT} overlaps the real home")
atexit.register(shutil.rmtree, _PINNED_ROOT, True)

# The imports run pinned to the tmp home all pinned modules import under, so a path they cache names
# no module's test home; the pin is put back (in ``finally``) as soon as they have run: collecting
# this module leaves no home, override or home cache behind for the modules collected after it.
with pinned_hermes_home(import_home(), importer=__name__):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    from gateway import canonical_answer_fallback
    from gateway.canonical_surface import (
        CanonicalIngressEvent,
        CanonicalReceiptCoordinator,
        CanonicalSurfaceBinding,
    )
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from run_agent import AIAgent as _PersistenceAgent


@pytest.fixture(autouse=True, scope="module")
def _pin_the_module_home():
    """The same pin while this module's tests run, owned by this fixture and put back on teardown."""
    with pinned_hermes_home(_PINNED_ROOT):
        yield


def _is_live(path) -> bool:
    return Path(path).expanduser().resolve().is_relative_to(_LIVE_ROOT)


@pytest.fixture(autouse=True)
def _never_a_live_path():
    """Fail before the test body when the config or state.db resolver names the real Hermes home."""
    import hermes_state
    from hermes_cli.config import get_config_path

    for path in (get_config_path(), hermes_state._default_db_path()):
        if _is_live(path):
            pytest.fail(f"a resolver names the live Hermes home: {path}")


_ROUTE = "/v1/canonical-surface/events"
_RECEIPT = "canonical-receipt:"
_KEPT = "canonical-answer-kept:"
# The local fallback beside the state.db, when the state.db refuses the kept answer too.
_FALLBACK_DIR = "canonical-answer-kept"


class _Actor:
    """Stands in for the bound cached agent: no model call; every run answers ``answer #<n>``, where
    n counts runs across every actor of the test, so a second run of one event is visible."""

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
        self.runs.append(kwargs.get("persist_user_message", text))
        answer = f"answer #{len(self.runs)}"
        if self.during_turn is not None:
            self.during_turn()
        user = {"role": "user", "content": text}
        for key, name in (("display_kind", "persist_user_display_kind"),
                          ("display_metadata", "persist_user_display_metadata")):
            if kwargs.get(name):
                user[key] = kwargs[name]
        self._persist_user_message_idx = len(conversation_history)
        return {"completed": True, "session_id": task_id, "final_response": answer,
                "messages": [*conversation_history, user, {"role": "assistant", "content": answer}]}


def _homes(tmp_path):
    """The test's HERMES_HOME and HOME."""
    return tmp_path / "hermes-home", tmp_path / "user"


@pytest.fixture
def gw(tmp_path, monkeypatch):
    """Real runner, resolver, coordinator, HTTP handler and SQLite under a tmp home; no model."""
    home, user_home = _homes(tmp_path)
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
    entry = runner.session_store.get_or_create_session(source)
    db = runner.session_store._db
    assert not _is_live(db.db_path)
    binding = CanonicalSurfaceBinding(
        name="canonical", session_key=entry.session_key, session_id=entry.session_id,
        telegram_chat_id="chat", telegram_chat_type="dm", telegram_user_id="user",
        telegram_thread_id=None, allowed_author_ids=("author",), allowed_channel_ids=("channel",),
    )
    runner.config.canonical_surface_bindings = {binding.name: binding}
    runs = []
    actor = _Actor(entry.session_id, db, runs)
    with runner._agent_cache_lock:
        runner._agent_cache[entry.session_key] = (actor, "exact", 0, entry.session_id)
    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", lambda *a, **kw: pytest.fail(
        "canonical ingress built an agent while one was cached for the session"))
    key = secrets.token_hex(32)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key}))
    adapter.gateway_runner = runner
    [handler] = [h for method, path, h in adapter._http_route_table() if (method, path) == ("POST", _ROUTE)]

    async def post(**changes):
        raw = json.dumps({"binding": "canonical", "event_id": "event", "author_id": "author",
                          "channel_id": "channel", "text": "hello", **changes}).encode()

        async def read():
            return raw
        request = SimpleNamespace(headers={"Authorization": f"Bearer {key}"}, read=read, method="POST",
                                  path_qs=_ROUTE, transport=None)
        token = _api_request_profile.set(None)
        try:
            response = await handler(request)
        finally:
            _api_request_profile.reset(token)
        return response.status, json.loads(response.text)

    async def send(**changes):
        status, body = await post(**changes)
        return status, body["text"] if status == 200 else body["error"]["code"]

    def rows(prefix):
        with contextlib.closing(sqlite3.connect((home / "state.db").resolve().as_uri() + "?mode=ro",
                                                uri=True)) as conn:
            return dict(conn.execute("SELECT key, value FROM state_meta WHERE key LIKE ?", (prefix + "%",)))

    def write(sql, params=()):
        """Another holder's write: its own connection to the same state.db, not the actor's handle."""
        with contextlib.closing(sqlite3.connect(home / "state.db", timeout=10)) as conn:
            conn.execute(sql, params)
            conn.commit()

    def receipt():
        """The event's receipt row as (state, text-or-owner), or None when there is no row."""
        [(_key, value)] = rows(_RECEIPT).items() or [(None, None)]
        if value is None:
            return None
        parsed = json.loads(value)
        if parsed["state"] == "pending":
            return "pending", parsed["owner"]
        if "failure" in parsed:
            return "terminal", parsed["failure"]["code"]
        return "terminal", parsed["response"]["terminal_text"]

    def fallback():
        """The local fallback's kept files beside the state.db, by name (never through a symlink)."""
        found = home / _FALLBACK_DIR
        return ({path.name: path for path in sorted(found.iterdir())}
                if found.is_dir() and not found.is_symlink() else {})

    def kept():
        """The answer text the event kept, beside its receipt or in the local fallback, or None."""
        values = [*rows(_KEPT).values(), *(path.read_text() for path in fallback().values())]
        answers = {json.loads(value)["answer"]["response"]["terminal_text"] for value in values}
        assert len(answers) <= 1
        return answers.pop() if answers else None

    def restart():
        """A clean restart: a new runner over the same home, nothing cached, on the real build path."""
        runner_now = adapter.gateway_runner
        runner_now.session_store.close_all_db_handles()
        restarted = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
        restarted.config.canonical_surface_bindings = {binding.name: binding}
        adapter.gateway_runner = restarted
        state.runners.append(restarted)
        built = []

        def build(**kwargs):
            built.append(kwargs["session_id"])
            return _Actor(kwargs["session_id"], kwargs["session_db"], runs)
        monkeypatch.setattr(run_agent, "AIAgent", build)
        monkeypatch.setattr(restarted, "_resolve_session_agent_runtime", lambda **kw: (
            "test-model", {"provider": "openrouter", "api_key": "test-key", "base_url": "http://127.0.0.1:9"}))
        return built

    state = SimpleNamespace(runner=runner, actor=actor, db=db, entry=entry, binding=binding, home=home,
                            runs=runs, send=send, post=post, rows=rows, write=write, receipt=receipt, kept=kept,
                            fallback=fallback, restart=restart, runners=[runner])
    try:
        yield state
    finally:
        reset_hermes_home_override(override)
        for each in state.runners:
            each.session_store.close_all_db_handles()
        adapter._response_store.close()


def _unrecorded_logs(caplog):
    """(stage, observed, delivery) of each 'terminal not recorded' log line, which must also name the
    event and its receipt (the correlation) and never the answer."""
    found = []
    for record in caplog.records:
        message = record.getMessage()
        if "terminal not recorded" not in message:
            continue
        assert "event_id='event'" in message and "receipt=canonical-receipt:v2:" in message, message
        assert "answer #" not in message, message
        fields = dict(part.split("=", 1) for part in message.split() if "=" in part)
        found.append((fields["stage"], fields["observed"], fields["delivery"]))
    return found


def _replace_receipt(gw, change):
    [(key, value)] = gw.rows(_RECEIPT).items()
    gw.write("UPDATE state_meta SET value = ? WHERE key = ?", (json.dumps(change(json.loads(value))), key))


def _other_holder(record):
    return {**record, "owner": "another-holder"}


def _recorded_elsewhere(record):
    return {"v": 1, "state": "terminal", "fingerprint": record["fingerprint"],
            "response": {"binding_name": "canonical", "terminal_text": "recorded elsewhere"}}


# ── Counterexamples: the turn produced "answer #1" and its terminal record did not land ──────────


def test_claim_taken_by_another_holder_mid_turn_is_never_overwritten_and_fails_explicitly(gw, caplog):
    caplog.set_level(logging.INFO)
    gw.actor.during_turn = lambda: _replace_receipt(gw, _other_holder)

    async def exercise():
        status, body = await gw.post()
        gw.actor.during_turn = None
        first = (status, body["error"]["code"])
        return {"first": first, "replay": await gw.send(), "runs": len(gw.runs), "receipt": gw.receipt(),
                "kept": gw.kept(), "log": _unrecorded_logs(caplog), "body": body}

    observed = asyncio.run(exercise())
    body = observed.pop("body")
    assert observed == {
        "first": (409, "canonical_answer_unrecorded"), "replay": (409, "canonical_answer_unrecorded"),
        "runs": 1, "receipt": ("pending", "another-holder"), "kept": "answer #1",
        "log": [("record.compare_and_set", "other_pending", "not_delivered")]}
    # An explicit failure that says what ran and what became of the answer, never the answer itself.
    assert {k: v for k, v in body["error"].items() if k != "message"} == {
        "code": "canonical_answer_unrecorded", "side_effects": "none", "delivery": "not_delivered",
        "answer_kept": True}
    assert "answer #1" not in json.dumps(body)


def test_row_deleted_mid_turn_is_not_owned_and_neither_reentry_nor_restoring_it_reruns(gw, caplog):
    caplog.set_level(logging.INFO)
    deleted = {}

    def delete():
        [(key, value)] = gw.rows(_RECEIPT).items()
        deleted.update(key=key, value=value)
        gw.write("DELETE FROM state_meta WHERE key = ?", (key,))
    gw.actor.during_turn = delete

    async def exercise():
        first = await gw.send()
        gw.actor.during_turn = None
        replay = await gw.send()
        after_reentry = {"runs": len(gw.runs), "receipt": gw.receipt()}
        # A restore from an earlier copy brings the deleted pending row back: still never run again.
        gw.write("INSERT OR IGNORE INTO state_meta (key, value) VALUES (?, ?)", (deleted["key"], deleted["value"]))
        restored = await gw.send()
        return {"first": first, "replay": replay, "after_reentry": after_reentry, "restored": restored,
                "runs": len(gw.runs), "kept": gw.kept(), "log": _unrecorded_logs(caplog)}

    assert asyncio.run(exercise()) == {
        "first": (409, "canonical_answer_unrecorded"), "replay": (409, "canonical_answer_unrecorded"),
        "after_reentry": {"runs": 1, "receipt": None},  # the missing row is not recreated as ours
        "restored": (409, "canonical_answer_unrecorded"), "runs": 1, "kept": "answer #1",
        "log": [("record.compare_and_set", "absent", "not_delivered")]}


def test_an_answer_recorded_by_someone_else_is_never_overwritten_and_ours_fails_explicitly(gw, caplog):
    caplog.set_level(logging.INFO)
    gw.actor.during_turn = lambda: _replace_receipt(gw, _recorded_elsewhere)

    async def exercise():
        first = await gw.send()
        gw.actor.during_turn = None
        return {"first": first, "replay": await gw.send(), "runs": len(gw.runs), "receipt": gw.receipt(),
                "kept": gw.kept(), "log": _unrecorded_logs(caplog)}

    assert asyncio.run(exercise()) == {
        "first": (409, "canonical_answer_unrecorded"), "replay": (409, "canonical_answer_unrecorded"),
        "runs": 1, "receipt": ("terminal", "recorded elsewhere"), "kept": "answer #1",
        "log": [("record.compare_and_set", "recorded_other", "not_delivered")]}


@pytest.mark.parametrize("miss", ["reports_false", "raises"])
def test_our_own_terminal_that_landed_but_reported_a_miss_is_delivered_from_the_receipt(gw, caplog, monkeypatch, miss):
    """The compare-and-set wrote the terminal and then answered False or raised: the receipt binds
    exactly this answer, so it is the answer, now and on every replay; nothing else is kept."""
    caplog.set_level(logging.INFO)
    real = gw.db.compare_and_set_meta

    def landed_then_missed(*args, **kwargs):
        assert real(*args, **kwargs)
        if miss == "raises":
            raise sqlite3.OperationalError("disk I/O error")
        return False
    monkeypatch.setattr(gw.db, "compare_and_set_meta", landed_then_missed)

    async def exercise():
        first = await gw.send()
        monkeypatch.setattr(gw.db, "compare_and_set_meta", real)
        return {"first": first, "replay": await gw.send(), "runs": len(gw.runs), "receipt": gw.receipt(),
                "kept": gw.kept(), "log": _unrecorded_logs(caplog)}

    assert asyncio.run(exercise()) == {
        "first": (200, "answer #1"), "replay": (200, "answer #1"), "runs": 1,
        "receipt": ("terminal", "answer #1"), "kept": None,
        "log": [("record.compare_and_set", "recorded_same", "answered_from_receipt")]}


def test_a_terminal_write_that_raises_before_landing_keeps_the_answer_and_fails_explicitly(gw, caplog, monkeypatch):
    caplog.set_level(logging.INFO)

    def locked(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(gw.db, "compare_and_set_meta", locked)

    async def exercise():
        first = await gw.send()
        return {"first": first, "replay": await gw.send(), "runs": len(gw.runs),
                "receipt": gw.receipt()[0], "kept": gw.kept(), "log": _unrecorded_logs(caplog)}

    assert asyncio.run(exercise()) == {
        "first": (409, "canonical_answer_unrecorded"), "replay": (409, "canonical_answer_unrecorded"),
        "runs": 1, "receipt": "pending", "kept": "answer #1",
        "log": [("record.compare_and_set", "own_pending", "not_delivered")]}


def test_a_concurrent_retry_while_the_row_is_gone_never_runs_the_turn_again(gw):
    """The bridge resends the same event while the first request is still in its turn (busy, so it
    keeps retrying) and the receipt row disappears meanwhile; once the first request ends, the retry
    must neither claim the event as new nor deliver a second answer."""
    async def exercise():
        loop = asyncio.get_running_loop()
        during = []

        def delete_then_retry():
            [key] = gw.rows(_RECEIPT)
            gw.write("DELETE FROM state_meta WHERE key = ?", (key,))
            during.append(asyncio.run_coroutine_threadsafe(gw.send(), loop).result(timeout=10))
        gw.actor.during_turn = delete_then_retry
        first = await gw.send()
        gw.actor.during_turn = None
        after = await gw.send()
        delivered = [text for status, text in (*during, first, after) if status == 200]
        return {"first": first, "retry_after": after, "runs": len(gw.runs), "delivered": delivered}

    assert asyncio.run(exercise()) == {
        "first": (409, "canonical_answer_unrecorded"), "retry_after": (409, "canonical_answer_unrecorded"),
        "runs": 1, "delivered": []}


class _Crash(BaseException):
    """The gateway process dying: nothing after this point runs, and no handler may catch it."""


@pytest.mark.parametrize("restore", ["receipt_intact", "receipt_lost"])
def test_a_restart_between_the_turn_and_the_record_never_reruns_or_records_success(gw, monkeypatch, restore):
    def crash(*_args, **_kwargs):
        raise _Crash()
    monkeypatch.setattr(gw.db, "compare_and_set_meta", crash)

    async def exercise():
        with pytest.raises(_Crash):
            await gw.send()
        if restore == "receipt_lost":  # the restore that came back with the restart lost the row
            [key] = gw.rows(_RECEIPT)
            gw.write("DELETE FROM state_meta WHERE key = ?", (key,))
        built = gw.restart()
        after = await gw.send()
        return {"after_restart": after, "runs": len(gw.runs), "built": len(built),
                "receipt": (gw.receipt() or (None,))[0]}

    assert asyncio.run(exercise()) == {
        "after_restart": (409, "canonical_event_uncertain"), "runs": 1, "built": 0,
        "receipt": {"receipt_intact": "pending", "receipt_lost": None}[restore]}


def test_a_restart_after_the_answer_was_kept_answers_the_same_failure_and_never_reruns(gw):
    gw.actor.during_turn = lambda: gw.write("DELETE FROM state_meta WHERE key LIKE ?", (_RECEIPT + "%",))

    async def exercise():
        first = await gw.send()
        built = gw.restart()
        after = await gw.send()
        return {"first": first, "after_restart": after, "runs": len(gw.runs), "built": len(built),
                "kept": gw.kept()}

    assert asyncio.run(exercise()) == {
        "first": (409, "canonical_answer_unrecorded"), "after_restart": (409, "canonical_answer_unrecorded"),
        "runs": 1, "built": 0, "kept": "answer #1"}


def test_a_recorded_events_row_deleted_later_is_never_reentered_as_new(gw):
    async def exercise():
        first = await gw.send()
        gw.write("DELETE FROM state_meta WHERE key LIKE ?", (_RECEIPT + "%",))
        return {"first": first, "reentry": await gw.send(), "runs": len(gw.runs)}

    assert asyncio.run(exercise()) == {
        "first": (200, "answer #1"), "reentry": (409, "canonical_event_uncertain"), "runs": 1}


# ── #1110 r1 counterexamples from the review evidence (test_counterexamples.py), byte for byte ──────


def test_another_holder_writing_identical_text_does_not_authorize_our_delivery(gw, caplog):
    caplog.set_level(logging.INFO)
    other = {}
    def other_holder_finishes():
        [(key, value)] = gw.rows(_RECEIPT).items()
        pending = json.loads(value)
        # Establish another holder explicitly before it records its own identical answer.
        other_pending = json.dumps({**pending, "owner": "another-holder"})
        gw.write("UPDATE state_meta SET value = ? WHERE key = ?", (other_pending, key))
        assert json.loads(gw.rows(_RECEIPT)[key])["owner"] == "another-holder"
        terminal = {"v": 1, "state": "terminal", "fingerprint": pending["fingerprint"],
                    "response": {"binding_name": "canonical", "terminal_text": "answer #1"}}
        encoded = json.dumps(terminal, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        other.update(key=key, terminal=encoded)
        gw.write("UPDATE state_meta SET value = ? WHERE key = ?", (encoded, key))
    gw.actor.during_turn = other_holder_finishes
    status, body = asyncio.run(gw.post())
    observed = {"status": status, "body": body, "runs": len(gw.runs),
                "kept": gw.kept(), "receipt_unchanged": gw.rows(_RECEIPT)[other["key"]] == other["terminal"]}
    print("OTHER_HOLDER_SAME_ANSWER", json.dumps(observed, sort_keys=True))
    assert status == 409, observed
    assert "text" not in body, observed


def _exercise_real_flush(gw, monkeypatch, *, inject_locked=True):
    # Use the production flush implementation. A locked transcript write is allowed to return
    # a completed model answer with agent_persisted=False; it does not create a JSONL diversion.
    writer = SimpleNamespace(
        _session_db=gw.db, _session_db_created=True, _persist_disabled=False,
        session_id=gw.entry.session_id, _session_persist_lock=None, _flushed_db_message_ids=set(),
        _flushed_db_message_session_id=None, _last_flushed_db_idx=0, _db_flush_scan_prefix=None,
        _persist_user_message_idx=None, _persist_user_message_override=None,
        _persist_user_message_timestamp=None, _pending_cli_user_message=None,
        _active_session_turn_lease_holder=None, _last_persistence_error_cause=None,
        _compression_adoption_failed=False)
    writer._ensure_db_session = lambda: None
    for method in ("_flush_messages_to_session_db", "_flush_messages_to_session_db_unlocked",
                   "_drop_trailing_empty_response_scaffolding", "_persist_session"):
        setattr(writer, method, getattr(_PersistenceAgent, method).__get__(writer, _PersistenceAgent))
    flush_calls = []
    def locked(*a, **kw):
        flush_calls.append(kw)
        raise sqlite3.OperationalError("database is locked")
    if inject_locked:
        monkeypatch.setattr(gw.db, "append_messages_batch", locked)
    run = gw.actor.run_conversation
    def run_and_flush(*a, **kw):
        result = run(*a, **kw)
        writer._persist_session(result["messages"], kw["conversation_history"])
        assert writer._last_persist_succeeded is False
        assert writer._last_persistence_error_cause == "locked"
        assert not inject_locked or flush_calls
        result["agent_persisted"] = writer._last_persist_succeeded
        return result
    monkeypatch.setattr(gw.actor, "run_conversation", run_and_flush)


def test_failed_keeper_does_not_discard_the_only_returned_answer(gw, monkeypatch):
    _exercise_real_flush(gw, monkeypatch)
    original_claim = gw.db.claim_meta_once
    def claim(key, *args, **kwargs):
        if key.startswith(_KEPT):
            raise sqlite3.OperationalError("database is locked")
        return original_claim(key, *args, **kwargs)
    monkeypatch.setattr(gw.db, "claim_meta_once", claim)
    monkeypatch.setattr(gw.db, "compare_and_set_meta", lambda *a, **kw: False)
    async def exercise():
        first = await gw.post()
        built = gw.restart()
        again = await gw.post()
        transcript = gw.runners[-1].session_store._db.get_messages(gw.entry.session_id)
        return {"first": first, "again": again, "runs": len(gw.runs), "kept": gw.kept(),
                "built": len(built), "transcript": transcript}
    observed = asyncio.run(exercise())
    print("FAILED_KEEP", json.dumps(observed, sort_keys=True, default=str))
    assert gw.kept() == "answer #1", observed


def test_generated_answer_survives_actor_replacement(gw, monkeypatch):
    _exercise_real_flush(gw, monkeypatch)
    def replace():
        replacement = _Actor(gw.entry.session_id, gw.db, gw.runs)
        with gw.runner._agent_cache_lock:
            gw.runner._agent_cache[gw.entry.session_key] = (replacement, "exact", 0, gw.entry.session_id)
    gw.actor.during_turn = replace
    observed = {"response": asyncio.run(gw.post()), "kept": gw.kept(), "runs": len(gw.runs)}
    print("POST_TURN_REPLACEMENT", json.dumps(observed, sort_keys=True))
    assert gw.kept() == "answer #1", observed



def test_real_sqlite_contention_keeps_the_completed_answer_before_returning_failure(gw, monkeypatch):
    _exercise_real_flush(gw, monkeypatch, inject_locked=False)
    # Shorten budgets only. The second connection really holds SQLite's writer lock, and
    # transcript append, terminal CAS and answer keeper use their production implementations.
    monkeypatch.setattr(gw.db, "_TRANSCRIPT_WRITE_PATIENCE_S", 0)
    monkeypatch.setattr(gw.db, "_WRITE_PATIENCE_S", 0)
    gw.db._conn.execute("PRAGMA busy_timeout=10")
    blockers = []
    def hold_write_lock():
        conn = sqlite3.connect(gw.home / "state.db", timeout=1, check_same_thread=False)
        conn.execute("BEGIN IMMEDIATE")
        blockers.append(conn)
    gw.actor.during_turn = hold_write_lock
    try:
        first = asyncio.run(gw.post())
    finally:
        for blocker in blockers:
            blocker.rollback()
            blocker.close()
    assert blockers, "the real competing connection must have acquired the writer lock"
    built = gw.restart()
    again = asyncio.run(gw.post())
    transcript = gw.runners[-1].session_store._db.get_messages(gw.entry.session_id)
    observed = {"first": first, "again": again, "runs": len(gw.runs), "kept": gw.kept(),
                "built": len(built), "transcript": transcript}
    print("REAL_SQLITE_CONTENTION", json.dumps(observed, sort_keys=True, default=str))
    assert gw.kept() == "answer #1", observed


# ── The preservation boundary, its local fallback, and the claim-bound readback (#1110 r1) ─────────


def _claim_of_pending(gw):
    """(receipt key, the claim owner its pending row names) for the event in flight."""
    [(key, value)] = gw.rows(_RECEIPT).items()
    return key, json.loads(value)["owner"]


def _refuse_the_kept_row(gw, monkeypatch):
    """The state.db refuses the kept answer as it refused the terminal record."""
    original = gw.db.claim_meta_once

    def claim(key, *args, **kwargs):
        if key.startswith(_KEPT):
            raise sqlite3.OperationalError("database is locked")
        return original(key, *args, **kwargs)
    monkeypatch.setattr(gw.db, "claim_meta_once", claim)


def _no_space(*_args, **_kwargs):
    raise OSError(28, "No space left on device")


def _error(body):
    return {name: value for name, value in body["error"].items() if name != "message"}


_UNRECORDED_KEPT = {"code": "canonical_answer_unrecorded", "side_effects": "none", "delivery": "not_delivered",
                    "answer_kept": True}


def test_an_answer_the_state_db_refuses_to_keep_is_kept_privately_beside_it_and_read_after_a_restart(
        gw, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    _refuse_the_kept_row(gw, monkeypatch)
    claim = {}

    def other_holder():
        claim["key"], claim["owner"] = _claim_of_pending(gw)
        _replace_receipt(gw, _other_holder)
    gw.actor.during_turn = other_holder

    async def exercise():
        first = await gw.post()
        gw.actor.during_turn = None
        built = gw.restart()
        return first, await gw.post(), built

    first, after_restart, built = asyncio.run(exercise())
    # Keyed by the event and the claim that ran it; private, and holding exactly what was kept.
    name = f"{claim['key'].rsplit(':', 1)[1]}.{claim['owner']}.json"
    files = gw.fallback()
    assert list(files) == [name]
    record = json.loads(files[name].read_text())
    assert (stat.S_IMODE(files[name].stat().st_mode), stat.S_IMODE(files[name].parent.stat().st_mode)) == (0o600, 0o700)
    assert (record["receipt"], record["owner"], record["answer"]["response"]["terminal_text"]) == (
        claim["key"], claim["owner"], "answer #1")
    assert gw.rows(_KEPT) == {}
    # Kept is not delivered: the first answer and the one after the restart are the same explicit failure.
    assert (first[0], _error(first[1]), after_restart[0], _error(after_restart[1])) == (
        409, _UNRECORDED_KEPT, 409, _UNRECORDED_KEPT)
    assert (len(gw.runs), len(built), gw.receipt()) == (1, 0, ("pending", "another-holder"))
    [line] = [r.getMessage() for r in caplog.records if "terminal not recorded" in r.getMessage()]
    assert "preservation=local_fallback" in line and "keep_error=OperationalError" in line
    assert "answer #" not in line


@pytest.mark.parametrize("fails", ["symlinked_directory", "no_space"])
def test_when_no_store_keeps_the_answer_the_failure_says_so_and_nothing_reruns_or_is_delivered(
        gw, monkeypatch, caplog, tmp_path, fails):
    caplog.set_level(logging.INFO)
    _refuse_the_kept_row(gw, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    if fails == "symlinked_directory":
        (gw.home / _FALLBACK_DIR).symlink_to(outside, target_is_directory=True)
    else:
        monkeypatch.setattr(canonical_answer_fallback, "keep", _no_space)
    gw.actor.during_turn = lambda: _replace_receipt(gw, _other_holder)

    async def exercise():
        first = await gw.post()
        gw.actor.during_turn = None
        replay = await gw.send()
        built = gw.restart()
        return first, replay, await gw.send(), built

    first, replay, after_restart, built = asyncio.run(exercise())
    assert (first[0], _error(first[1])) == (409, {**_UNRECORDED_KEPT, "answer_kept": False})
    assert "neither recorded nor preserved" in first[1]["error"]["message"]
    # Nothing claims it was kept, nothing answers with it, and the turn never runs again.
    assert (replay, after_restart) == ((409, "canonical_event_uncertain"),) * 2
    assert (len(gw.runs), len(built), gw.kept(), sorted(outside.iterdir())) == (1, 0, None, [])
    [line] = [r.getMessage() for r in caplog.records if "terminal not recorded" in r.getMessage()]
    assert "NOT preserved" in line and "preservation=failed" in line and "answer_kept=False" in line
    # A symlink is refused, never followed (ELOOP or ENOTDIR, by platform); a full disk is ENOSPC.
    assert ("fallback_error=OSError" in line or (fails == "symlinked_directory" and "fallback_error=NotADirectoryError" in line))
    assert "answer #" not in line
    assert [r.levelname for r in caplog.records if "NOT preserved" in r.getMessage()] == ["CRITICAL"]


@pytest.mark.parametrize("existing", ["another_claims_file", "a_file_at_this_claims_name"])
def test_the_fallback_never_writes_over_a_file_it_did_not_create(gw, monkeypatch, existing):
    _refuse_the_kept_row(gw, monkeypatch)
    folder = gw.home / _FALLBACK_DIR
    folder.mkdir(mode=0o700)
    planted, claim = {}, {}

    def plant():
        claim["key"], claim["owner"] = _claim_of_pending(gw)
        owner = claim["owner"] if existing == "a_file_at_this_claims_name" else "f" * 32
        path = folder / f"{claim['key'].rsplit(':', 1)[1]}.{owner}.json"
        path.write_bytes(b"another writer's bytes")
        path.chmod(0o600)
        planted[path.name] = path.read_bytes()
        _replace_receipt(gw, _other_holder)
    gw.actor.during_turn = plant

    status, body = asyncio.run(gw.post())
    assert {name: (folder / name).read_bytes() for name in planted} == planted  # untouched
    ours = f"{claim['key'].rsplit(':', 1)[1]}.{claim['owner']}.json"
    if existing == "another_claims_file":
        assert (status, _error(body)) == (409, _UNRECORDED_KEPT)
        assert sorted(path.name for path in folder.iterdir()) == sorted([*planted, ours])
        assert json.loads((folder / ours).read_text())["answer"]["response"]["terminal_text"] == "answer #1"
    else:
        assert (status, _error(body)) == (409, {**_UNRECORDED_KEPT, "answer_kept": False})
        assert sorted(path.name for path in folder.iterdir()) == [ours]  # no temporary left behind
    assert len(gw.runs) == 1


def test_an_answer_rejected_after_the_actor_returned_passes_the_same_boundary_into_the_fallback(gw, monkeypatch):
    _refuse_the_kept_row(gw, monkeypatch)
    cas = []
    real_cas = gw.db.compare_and_set_meta
    monkeypatch.setattr(gw.db, "compare_and_set_meta", lambda *a, **kw: cas.append(a) or real_cas(*a, **kw))

    def replace():
        replacement = _Actor(gw.entry.session_id, gw.db, gw.runs)
        with gw.runner._agent_cache_lock:
            gw.runner._agent_cache[gw.entry.session_key] = (replacement, "exact", 0, gw.entry.session_id)
    gw.actor.during_turn = replace

    async def exercise():
        first = await gw.post()
        built = gw.restart()
        return first, await gw.post(), built

    first, after_restart, built = asyncio.run(exercise())
    assert (first[0], _error(first[1]), after_restart[0], _error(after_restart[1])) == (
        409, _UNRECORDED_KEPT, 409, _UNRECORDED_KEPT)
    assert (gw.kept(), len(gw.fallback()), gw.rows(_KEPT), cas) == ("answer #1", 1, {}, [])
    assert (len(gw.runs), len(built), gw.receipt()[0]) == (1, 0, "pending")


@pytest.mark.parametrize("kept_in", ["state_db", "local_fallback", "nowhere"])
def test_another_holders_identical_answer_is_never_delivered_concurrently_or_after_a_restart(gw, monkeypatch, kept_in):
    """Another holder takes the claim and records the very answer this turn produced while the bridge
    resends the event: its bytes bind no delivery here, before or after a restart, wherever the answer
    is kept, and even when nothing could keep it."""
    if kept_in != "state_db":
        _refuse_the_kept_row(gw, monkeypatch)
    if kept_in == "nowhere":
        monkeypatch.setattr(canonical_answer_fallback, "keep", _no_space)
    other = {}

    async def exercise():
        loop = asyncio.get_running_loop()
        during = []

        def other_holder_records_the_same_answer_while_the_bridge_resends():
            [(key, value)] = gw.rows(_RECEIPT).items()
            pending = json.loads(value)
            gw.write("UPDATE state_meta SET value = ? WHERE key = ?",
                     (json.dumps({**pending, "owner": "another-holder"}), key))
            other.update(key=key, terminal=json.dumps(
                {"v": 1, "state": "terminal", "fingerprint": pending["fingerprint"],
                 "response": {"binding_name": "canonical", "terminal_text": "answer #1"}},
                ensure_ascii=False, separators=(",", ":"), sort_keys=True))
            gw.write("UPDATE state_meta SET value = ? WHERE key = ?", (other["terminal"], key))
            during.append(asyncio.run_coroutine_threadsafe(gw.send(), loop).result(timeout=10))
        gw.actor.during_turn = other_holder_records_the_same_answer_while_the_bridge_resends
        first = await gw.send()
        gw.actor.during_turn = None
        replay = await gw.send()
        built = gw.restart()
        after_restart = await gw.send()
        return {"during": during, "first": first, "replay": replay, "after_restart": after_restart,
                "built": len(built), "runs": len(gw.runs),
                "delivered": [text for status, text in (*during, first, replay, after_restart) if status == 200]}

    replayed = {"nowhere": (409, "canonical_event_conflict")}.get(kept_in, (409, "canonical_answer_unrecorded"))
    assert asyncio.run(exercise()) == {
        "during": [(409, "canonical_event_conflict")], "first": (409, "canonical_answer_unrecorded"),
        "replay": replayed, "after_restart": replayed, "built": 0, "runs": 1, "delivered": []}
    assert gw.rows(_RECEIPT)[other["key"]] == other["terminal"]  # never written over
    assert gw.kept() == (None if kept_in == "nowhere" else "answer #1")


# ── #1110 r3 counterexamples from the review evidence (test_review_regressions.py), byte for byte ──────


@pytest.mark.parametrize("path", ["cas_miss", "actor_replaced"])
def test_full_generated_answer_remains_retrievable_when_it_exceeds_the_wire_cap(gw, monkeypatch, path):
    answer = "FULL-ANSWER-" + "q" * 17000
    real = gw.actor.run_conversation

    def long_answer(*args, **kwargs):
        result = real(*args, **kwargs)
        result["final_response"] = answer
        result["messages"][-1]["content"] = answer
        return result

    monkeypatch.setattr(gw.actor, "run_conversation", long_answer)
    _exercise_real_flush(gw, monkeypatch)
    _refuse_the_kept_row(gw, monkeypatch)
    if path == "cas_miss":
        monkeypatch.setattr(gw.db, "compare_and_set_meta", lambda *a, **kw: False)
    else:
        def replace():
            other = _Actor(gw.entry.session_id, gw.db, gw.runs)
            with gw.runner._agent_cache_lock:
                gw.runner._agent_cache[gw.entry.session_key] = (other, "exact", 0, gw.entry.session_id)
        gw.actor.during_turn = replace
    first = asyncio.run(gw.post())
    built = gw.restart()
    replay = asyncio.run(gw.post())
    records = [json.loads(p.read_text()) for p in gw.fallback().values()]
    transcript = gw.runners[-1].session_store._db.get_messages(gw.entry.session_id)
    found = any(answer in json.dumps(record) for record in [*records, *transcript])
    evidence = {"path": path, "first": first, "replay": replay, "runs": len(gw.runs),
                "builds_after_restart": len(built), "transcript": transcript, "records": records,
                "answer_chars": len(answer), "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                "full_answer_retrievable": found}
    print("OVERCAP_PRESERVATION", json.dumps(evidence, sort_keys=True))
    assert found, evidence


@pytest.mark.parametrize("claim_state", ["legacy_without_owner", "claim_row_lost"])
def test_unknown_terminal_ownership_is_a_conflict_before_and_after_restart(gw, claim_state):
    """The real round-1 receipt representation cannot identify its writer after an upgrade."""
    event = CanonicalIngressEvent(binding="canonical", event_id="event", author_id="author",
                                  channel_id="channel", text="hello")
    key = CanonicalReceiptCoordinator._key(gw.binding, event)
    fingerprint = CanonicalReceiptCoordinator._fingerprint(gw.binding, event)
    if claim_state == "legacy_without_owner":
        claim = {"v": 1, "state": "claimed", "fingerprint": fingerprint}
        gw.write("INSERT INTO state_meta(key,value) VALUES(?,?)",
                 (CanonicalReceiptCoordinator._beside("canonical-claim:v1", key), json.dumps(claim)))
    terminal = {"v": 1, "state": "terminal", "fingerprint": fingerprint,
                "response": {"binding_name": "canonical", "terminal_text": "answer by unknown holder"}}
    if claim_state == "claim_row_lost":
        terminal.update(owner="f" * 32, write="a" * 32)
    gw.write("INSERT INTO state_meta(key,value) VALUES(?,?)", (key, json.dumps(terminal)))
    if claim_state == "claim_row_lost":
        # A gateway restart must not turn the absence of the original claim into proof of its writer.
        gw.restart()
    first = asyncio.run(gw.post())
    built = gw.restart()
    replay = asyncio.run(gw.post())
    observed = {"claim": claim_state, "first": first, "after_restart": replay,
                "runs": len(gw.runs), "rebuilt": len(built)}
    print("UNKNOWN_OWNERSHIP", json.dumps(observed, sort_keys=True))
    assert first[0] == replay[0] == 409, observed
    assert "text" not in first[1] and "text" not in replay[1], observed


# ── The whole answer is what is kept; an unknown owner is never proof (#1110 r3) ────────────────────


@pytest.mark.parametrize("kept_in", ["state_db", "local_fallback", "past_the_kept_bound"])
def test_an_over_cap_answer_is_kept_whole_and_byte_identical_after_a_restart_or_reported_not_kept(
        gw, monkeypatch, caplog, kept_in):
    """Only the receipt carries the length-and-digest summary of an answer over the wire cap. What is
    kept is the answer itself, exactly as the actor returned it (escapes, non-ASCII and astral
    characters included), and a restart reads back those very bytes. A record that cannot hold it
    whole says the answer is not kept; it never says kept for a summary, and nothing runs again."""
    from gateway import canonical_surface

    caplog.set_level(logging.INFO)
    answer = "WHOLE é中\"\\\n\x01 " + "q" * 17_000 + " \U0001f600"
    real = gw.actor.run_conversation

    def long_answer(*args, **kwargs):
        result = real(*args, **kwargs)
        result["final_response"] = answer
        result["messages"][-1]["content"] = answer
        return result
    monkeypatch.setattr(gw.actor, "run_conversation", long_answer)
    _exercise_real_flush(gw, monkeypatch)  # the transcript holds no copy of it either
    monkeypatch.setattr(gw.db, "compare_and_set_meta", lambda *a, **kw: False)
    if kept_in != "state_db":
        _refuse_the_kept_row(gw, monkeypatch)
    if kept_in == "past_the_kept_bound":
        monkeypatch.setattr(canonical_surface, "_MAX_KEPT_CHARS", len(answer))

    first = asyncio.run(gw.post())
    built = gw.restart()
    after_restart = asyncio.run(gw.post())
    kept = [*gw.rows(_KEPT).values(), *(path.read_bytes().decode("utf-8") for path in gw.fallback().values())]
    assert (len(gw.runs), len(built)) == (1, 0)
    if kept_in == "past_the_kept_bound":
        assert kept == [] and (first[0], _error(first[1])) == (409, {**_UNRECORDED_KEPT, "answer_kept": False})
        assert (after_restart[0], after_restart[1]["error"]["code"]) == (409, "canonical_event_uncertain")
        assert [r.levelname for r in caplog.records if "NOT preserved" in r.getMessage()] == ["CRITICAL"]
        return
    [record] = [json.loads(value) for value in kept]
    whole = record["answer"]["response"]["terminal_text"]
    assert whole.encode("utf-8", "surrogatepass") == answer.encode("utf-8", "surrogatepass")
    assert record["answer_sha256"] == hashlib.sha256(answer.encode("utf-8")).hexdigest()
    assert (first[0], _error(first[1]), after_restart[0], _error(after_restart[1])) == (
        409, _UNRECORDED_KEPT, 409, _UNRECORDED_KEPT)
    assert answer not in json.dumps([first, after_restart])  # kept is not delivered
    # A record that holds only a summary of the answer never reads as kept, before or after a restart.
    summary_only = {**record, "answer": {"failure": {"code": "canonical_result_too_large", "side_effects": "none",
                                                     "result_chars": len(answer), "result_sha256": record["answer_sha256"]}}}
    decoded = CanonicalReceiptCoordinator._decode_kept(json.dumps(summary_only), record["fingerprint"])
    assert (decoded.status, decoded.failure["answer_kept"]) == ("unrecorded", False)


def test_control_a_proven_owner_answers_its_text_before_and_after_a_restart(gw):
    """The claim names its owner and the terminal names the same one: the normal path still answers
    its recorded text on a replay and, cold, after a restart, without running the turn again."""
    async def exercise():
        first = await gw.send()
        replay = await gw.send()
        built = gw.restart()
        return {"first": first, "replay": replay, "after_restart": await gw.send(), "runs": len(gw.runs),
                "built": len(built)}

    assert asyncio.run(exercise()) == {
        "first": (200, "answer #1"), "replay": (200, "answer #1"), "after_restart": (200, "answer #1"),
        "runs": 1, "built": 0}


# ── Controls ─────────────────────────────────────────────────────────────────────────────────────


def test_control_a_normal_record_records_once_and_replays_it(gw, caplog):
    caplog.set_level(logging.INFO)

    async def exercise():
        first = await gw.send()
        return {"first": first, "replay": await gw.send(), "runs": len(gw.runs), "receipt": gw.receipt(),
                "kept": gw.kept(), "log": _unrecorded_logs(caplog)}

    assert asyncio.run(exercise()) == {
        "first": (200, "answer #1"), "replay": (200, "answer #1"), "runs": 1,
        "receipt": ("terminal", "answer #1"), "kept": None, "log": []}


def test_control_concurrent_duplicates_run_once_and_answer_only_the_recorded_text(gw):
    async def exercise():
        loop = asyncio.get_running_loop()
        during = []
        gw.actor.during_turn = lambda: during.append(
            asyncio.run_coroutine_threadsafe(gw.send(), loop).result(timeout=10))
        first = await gw.send()
        gw.actor.during_turn = None
        replay = await gw.send()
        return {"during": during, "first": first, "replay": replay, "runs": len(gw.runs)}

    assert asyncio.run(exercise()) == {
        "during": [(409, "canonical_event_uncertain")], "first": (200, "answer #1"),
        "replay": (200, "answer #1"), "runs": 1}


def test_control_existing_refusals_keep_their_codes(gw):
    async def exercise():
        refused = {"principal": await gw.send(author_id="stranger"),
                   "binding": await gw.send(binding="unknown")}
        gw.runner._session_state(gw.entry.session_key).turn.agent = object()  # an ordinary turn runs
        refused["busy"] = await gw.send()
        gw.runner._session_state(gw.entry.session_key).turn.agent = None
        refused["first"] = await gw.send()
        refused["changed"] = await gw.send(text="changed")
        return refused, len(gw.runs)

    assert asyncio.run(exercise()) == ({
        "principal": (403, "canonical_principal_rejected"), "binding": (404, "canonical_binding_unknown"),
        "busy": (409, "canonical_turn_busy"), "first": (200, "answer #1"),
        "changed": (409, "canonical_event_conflict")}, 1)


def test_canary_a_test_home_inside_the_live_root_fails_before_anything_is_written(tmp_path, monkeypatch, request):
    """The pre-write guard, aimed at a stand-in live root: the fixture fails before it creates a
    directory, opens a database or writes a row there."""
    import sys

    live = tmp_path / "account" / ".hermes"
    live.mkdir(parents=True)
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_LIVE_ROOT", live.resolve())
    monkeypatch.setattr(module, "_homes", lambda _tmp: (live / "hermes-home", live / "user"))
    with pytest.raises(pytest.fail.Exception, match="live Hermes home"):
        request.getfixturevalue("gw")
    assert sorted(live.rglob("*")) == []
