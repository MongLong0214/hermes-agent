"""Gateway ``/branch`` replies only with what it committed, and forks only what the parent committed.

Drives the REAL ``_handle_branch_command`` against a REAL SessionStore + SessionDB (SQLite in
tmp_path). A second ``SessionDB`` handle on the same file plays another process (CLI, Desktop, a
second routing key) that owns the parent's durable turn lease. Ported from the fork's
tests/gateway/test_branch_commit_truth.py, cut to the untitled reply and the busy-parent refusal.
"""

from __future__ import annotations

import asyncio
import os
import re
import sqlite3
import uuid

import pytest

from agent.i18n import t
from gateway import slash_commands_session
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, SessionStore
from hermes_state import AsyncSessionDB, SessionDB


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


@pytest.fixture()
def short_parent_wait(monkeypatch):
    monkeypatch.setattr(slash_commands_session, "_BRANCH_PARENT_WAIT_S", 0.6, raising=False)
    monkeypatch.setattr(slash_commands_session, "_BRANCH_POLL_S", 0.05, raising=False)


def _runner(store):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._profile_adapters = {}
    runner.config = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._agent_cache_lock = None
    runner.session_store = store
    runner._session_db = AsyncSessionDB(store._db)
    runner._pending_skills_reload_notes = {}
    return runner


def _source():
    return SessionSource(platform=Platform.DISCORD, chat_id="123", chat_type="group", user_id="u1",
                         user_name="ann", scope_id="g9")


def _seed(store, source):
    entry = store.get_or_create_session(source)
    store._db.append_message(entry.session_id, role="user", content="hello")
    store._db.append_message(entry.session_id, role="assistant", content="world")
    return entry


def _children(store, parent_id):
    with sqlite3.connect(store._db.db_path) as conn:
        return [r[0] for r in conn.execute(
            "SELECT id FROM sessions WHERE parent_session_id = ?", (parent_id,))]


def _hold_parent_turn(store, parent_id):
    other_process = SessionDB(store._db.db_path)
    holder = f"pid={os.getpid()}:turn=in-flight-{uuid.uuid4().hex}"
    assert other_process.try_acquire_session_turn_lease(parent_id, holder, ttl_seconds=120)
    return other_process, holder


@pytest.mark.asyncio
async def test_a_busy_parent_is_refused_and_no_fork_row_is_written(store, short_parent_wait):
    source = _source()
    parent = _seed(store, source)
    other_process, holder = _hold_parent_turn(store, parent.session_id)
    try:
        reply = await _runner(store)._handle_branch_command(MessageEvent(text="/branch --here", source=source))
    finally:
        other_process.release_session_turn_lease(parent.session_id, holder)
        other_process.close()

    assert reply == t("gateway.branch.parent_busy")
    assert _children(store, parent.session_id) == []
    assert store.get_or_create_session(source).session_id == parent.session_id


@pytest.mark.asyncio
async def test_a_parent_turn_committed_during_branch_is_in_the_branch(store, short_parent_wait):
    source = _source()
    parent = _seed(store, source)
    other_process, holder = _hold_parent_turn(store, parent.session_id)

    branch = asyncio.create_task(
        _runner(store)._handle_branch_command(MessageEvent(text="/branch --here", source=source)))
    # The in-flight turn finishes on its own clock, inside /branch's wait.
    await asyncio.wait({branch}, timeout=0.2)
    other_process.append_messages_batch(
        parent.session_id,
        [{"role": "user", "content": "late question"}, {"role": "assistant", "content": "late answer"}],
        turn_lease_holder=holder)
    other_process.release_session_turn_lease(parent.session_id, holder)
    await asyncio.wait_for(branch, timeout=30)
    other_process.close()

    children = _children(store, parent.session_id)
    assert len(children) == 1
    assert [m["content"] for m in store._db.get_messages(children[0])] == [
        "hello", "world", "late question", "late answer"]


@pytest.mark.asyncio
async def test_an_untitled_branch_reply_never_names_a_title(store):
    source = _source()
    parent = _seed(store, source)
    store._db.create_session("holder", source="cli")
    store._db.set_session_title("holder", "taken")

    reply = await _runner(store)._handle_branch_command(MessageEvent(text="/branch --here taken", source=source))

    (child,) = _children(store, parent.session_id)
    assert store._db.get_session(child)["title"] is None
    assert "**taken**" not in reply and reply.startswith("⑂ Branched (")
    assert "already in use" in reply  # says why it is untitled


@pytest.mark.asyncio
async def test_a_titled_branch_keeps_the_target_copy(store):
    source = _source()
    parent = _seed(store, source)

    reply = await _runner(store)._handle_branch_command(MessageEvent(text="/branch --here ozon", source=source))

    (child,) = _children(store, parent.session_id)
    assert store._db.get_session(child)["title"] == "ozon"
    assert reply == t("gateway.branch.branched_one", title="ozon", count=1, parent=parent.session_id, new=child)


def test_no_untitled_reply_names_a_title():
    keys = slash_commands_session._BRANCH_REPLY_KEYS
    assert len(keys) == 8
    for (thread, one, titled), key in keys.items():
        template = t(key, lang="en")  # no format kwargs: the raw English template
        assert ("{title}" in template) is titled
        assert ("{thread}" in template) is thread
        assert bool(re.search(r"\{count\} messages? copied", template))
        assert ("message copied" in template) is one
