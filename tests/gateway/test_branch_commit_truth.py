"""Gateway ``/branch`` forks what the parent durably committed and never writes a row it did not create.

Drives the REAL ``_handle_branch_command`` against a REAL SessionStore + SessionDB (SQLite in tmp_path).
A second ``SessionDB`` handle on the same file plays another process (CLI, Desktop, a second routing
key) that owns the parent's durable turn lease.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import uuid
from datetime import datetime, timedelta

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, SessionStore
from hermes_state import AsyncSessionDB, SessionDB


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


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
    runner._update_prompt_pending = {}
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


@pytest.mark.asyncio
async def test_parent_turn_committed_during_branch_is_in_the_branch_or_the_branch_refuses(store):
    source = _source()
    parent = _seed(store, source)
    other_process = SessionDB(store._db.db_path)
    holder = f"pid={os.getpid()}:turn=in-flight-{uuid.uuid4().hex}"
    assert other_process.try_acquire_session_turn_lease(parent.session_id, holder, ttl_seconds=120)

    branch = asyncio.create_task(
        _runner(store)._handle_branch_command(MessageEvent(text="/branch --here", source=source)))
    # The in-flight turn finishes on its own clock, whether or not /branch waited for it.
    await asyncio.wait({branch}, timeout=2.0)
    other_process.append_messages_batch(
        parent.session_id,
        [{"role": "user", "content": "late question"}, {"role": "assistant", "content": "late answer"}],
        turn_lease_holder=holder)
    other_process.release_session_turn_lease(parent.session_id, holder)
    await asyncio.wait_for(branch, timeout=90)
    other_process.close()

    # Refusing is allowed; a branch row that lacks the parent's commit is not.
    with sqlite3.connect(store._db.db_path) as conn:
        children = [r[0] for r in conn.execute(
            "SELECT id FROM sessions WHERE parent_session_id = ?", (parent.session_id,))]
    for child in children:
        assert [m["content"] for m in store._db.get_messages(child)] == [
            "hello", "world", "late question", "late answer"]
    current = store.get_or_create_session(source).session_id
    assert current == parent.session_id or current in children


@pytest.mark.asyncio
async def test_taken_branch_id_is_refused_without_writing_anything(store, monkeypatch):
    source = _source()
    parent = _seed(store, source)
    fixed = uuid.UUID(hex="abcdef12" * 4)
    start = datetime.now()
    # A foreign row on every id /branch can mint in the next seconds (same shape, same random part).
    foreign = sorted({f"{(start + timedelta(seconds=s)).strftime('%Y%m%d_%H%M%S')}_{fixed.hex[:6]}"
                      for s in range(-1, 15)})
    for fid in foreign:
        store._db.create_session(fid, source="cli")

    def _snapshot():
        with sqlite3.connect(store._db.db_path) as conn:
            marks = ",".join("?" * len(foreign))
            return (conn.execute(f"SELECT * FROM sessions WHERE id IN ({marks}) ORDER BY id", foreign).fetchall(),
                    conn.execute(f"SELECT COUNT(*) FROM messages WHERE session_id IN ({marks})", foreign).fetchone(),
                    conn.execute("SELECT COUNT(*) FROM sessions").fetchone(),
                    conn.execute("SELECT * FROM system_prompts ORDER BY hash").fetchall())

    before = _snapshot()
    monkeypatch.setattr(uuid, "uuid4", lambda: fixed)
    reply = await _runner(store)._handle_branch_command(MessageEvent(text="/branch --here", source=source))

    assert _snapshot() == before
    assert store.get_or_create_session(source).session_id == parent.session_id
    assert not any(fid in reply for fid in foreign)
    # The same contract at the store: a taken id is refused before the prompt blob is stored.
    assert store._db.create_session_strict(foreign[0], source="telegram", system_prompt="fresh prompt",
                                           parent_session_id=parent.session_id, chat_id="123") is False
    assert _snapshot() == before
