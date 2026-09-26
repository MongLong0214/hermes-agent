"""Transactional result-truth regressions for gateway ``/branch``."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.i18n import t
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, SessionStore, build_session_key
from hermes_state import (
    AsyncSessionDB,
    SessionDB,
    SessionTurnLeaseLostError,
)
from hermes_state_common import TURN_FENCE_GENERATION


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A real ``SessionStore`` backed by temporary SQLite state."""
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="truth-user",
        chat_id="truth-chat",
        chat_type="dm",
        thread_id="truth-thread",
    )


def _event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_source(), message_id="truth-message")


def _runner(store: SessionStore):
    """Wire the real branch persistence/routing seam into message dispatch."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="test-token")
        }
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner.session_store = store
    runner._session_db = AsyncSessionDB(store._db)
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._pending_skills_reload_notes = {}
    runner._agent_cache_lock = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *_args, **_kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


def _branch_marker(row: dict) -> str | None:
    config = json.loads(row["model_config"] or "{}")
    return config.get("_branched_from")


def test_create_session_strict_propagates_fk_violation_not_a_collision(store):
    """An unknown ``parent_session_id`` trips the real
    ``FOREIGN KEY (parent_session_id) REFERENCES sessions(id)`` constraint
    (schema in hermes_state_common.py), which raises
    ``sqlite3.IntegrityError`` — the SAME exception class a PRIMARY KEY
    collision on ``id`` raises. ``create_session_strict`` must not conflate
    the two: only a real collision on the row's own id returns ``False``.
    Everything else must propagate, because gateway/slash_commands.py
    reports a bare ``False`` (no exception) as "generated session ID
    collision" — a false result for a broken foreign key.
    """
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        store._db.create_session_strict(
            session_id="fk-violation-child",
            source="test",
            parent_session_id="does-not-exist-anywhere",
        )
    # Refused before a single column landed — same guarantee as a real
    # collision, just via a different, propagating failure.
    assert store._db.get_session("fk-violation-child") is None


def test_create_session_strict_propagates_turn_fence_abort_not_a_collision(store):
    """A turn-fence trigger's ``RAISE(ABORT)`` also raises
    ``sqlite3.IntegrityError`` (verified: SQLite reports it as
    ``SQLITE_CONSTRAINT_TRIGGER``, not a PRIMARY KEY/UNIQUE conflict).
    Swallowing this into the same bare ``False`` a real collision returns
    would report a fence refusal to the caller as "generated session ID
    collision" and silently lose the fence signal entirely.

    Simulated the same way
    tests/state/test_turn_fence_generation_current_main.py does: override
    the registered generation scalar on this connection so the governed
    table's BEFORE INSERT trigger aborts.
    """
    store._db._conn.create_function(
        "hermes_turn_fence_generation", 0, lambda: TURN_FENCE_GENERATION - 1
    )
    try:
        with pytest.raises(
            sqlite3.IntegrityError, match="generation incompatible"
        ):
            store._db.create_session_strict(
                session_id="fenced-child", source="test"
            )
    finally:
        store._db._conn.create_function(
            "hermes_turn_fence_generation", 0, lambda: TURN_FENCE_GENERATION
        )
    assert store._db.get_session("fenced-child") is None


def test_create_session_strict_fence_wins_over_a_pre_existing_row_with_same_id(
    store,
):
    """Compound case: a row with this id ALREADY EXISTS **and** the turn-fence
    generation is wrong.

    The id-existence check is the FIRST statement ``_do`` runs — before any
    write, before the governed ``INSERT`` is even attempted. So when a row
    with this id already exists, ``create_session_strict`` returns ``False``
    before the fence trigger ever gets a chance to fire: there is no
    governed INSERT for it to guard, so there is nothing for the fence to
    refuse and nothing to swallow. The pre-existing row must come back
    completely unchanged — not just present, but byte-for-byte identical to
    what it was before the call.
    """
    assert (
        store._db.create_session_strict(
            session_id="fenced-collision-existing", source="test"
        )
        is True
    )
    row_before = store._db.get_session("fenced-collision-existing")
    store._db._conn.create_function(
        "hermes_turn_fence_generation", 0, lambda: TURN_FENCE_GENERATION - 1
    )
    try:
        assert (
            store._db.create_session_strict(
                session_id="fenced-collision-existing", source="test"
            )
            is False
        )
    finally:
        store._db._conn.create_function(
            "hermes_turn_fence_generation", 0, lambda: TURN_FENCE_GENERATION
        )
    # The pre-existing row must be completely unchanged by the refused call
    # — the governed INSERT was never attempted, so there is nothing that
    # could have touched it.
    row_after = store._db.get_session("fenced-collision-existing")
    assert row_after is not None
    assert dict(row_after) == dict(row_before)


def test_create_session_strict_existence_check_precedes_every_write(store):
    """The id-existence check must be the FIRST statement in ``_do`` —
    before ``_store_system_prompt``, which writes to the ``system_prompts``
    table. If the existence check ran after that write (or after any other
    write), a call carrying a ``system_prompt`` for an id that already
    exists would land a ``system_prompts`` row before returning ``False``,
    breaking the contract that ``False`` means nothing of ours was written.
    """
    assert (
        store._db.create_session_strict(
            session_id="existing-id-for-ordering-check", source="test"
        )
        is True
    )
    prompts_before = store._db._conn.execute(
        "SELECT COUNT(*) FROM system_prompts"
    ).fetchone()[0]

    result = store._db.create_session_strict(
        session_id="existing-id-for-ordering-check",
        source="test",
        system_prompt="a system prompt only this call provides",
    )

    prompts_after = store._db._conn.execute(
        "SELECT COUNT(*) FROM system_prompts"
    ).fetchone()[0]
    assert result is False
    assert prompts_after == prompts_before, (
        "create_session_strict wrote a system_prompts row for an id that "
        "was already taken — the existence check did not precede every "
        "write"
    )


def test_create_session_strict_still_reports_real_pk_collision_as_false(store):
    """The classification change must leave the actual collision contract
    unchanged: a real PRIMARY KEY conflict on ``id`` still returns ``False``
    rather than raising."""
    assert (
        store._db.create_session_strict(session_id="dup-strict", source="test")
        is True
    )
    assert (
        store._db.create_session_strict(session_id="dup-strict", source="test")
        is False
    )


def test_create_session_strict_propagates_non_pk_unique_violation_not_a_collision(
    store,
):
    """A UNIQUE violation on a DIFFERENT column — NOT the row's own id — must
    also propagate rather than being misreported as an id collision.

    Real repro, same shape as the FK/fence tests above: add a genuine UNIQUE
    index on ``sessions.display_name`` (the same pattern production already
    uses for ``idx_sessions_title_unique`` in hermes_state_schema.py), seed a
    row that already holds a display_name, then call ``create_session_strict``
    with a FRESH, non-colliding ``id`` but the SAME display_name. The id
    itself never collides — only the secondary index does — so a correct
    implementation must raise, not return ``False``.

    Verified empirically (see create_session_strict's docstring): a real
    PRIMARY KEY collision on ``sessions.id`` reports
    ``sqlite_errorname == "SQLITE_CONSTRAINT_PRIMARYKEY"``, while a UNIQUE
    violation on any OTHER column reports ``"SQLITE_CONSTRAINT_UNIQUE"`` — the
    two are reliably distinguishable, so only the former may be swallowed.
    """
    store._db._conn.execute(
        "CREATE UNIQUE INDEX idx_display_name_unique_for_test "
        "ON sessions(display_name) WHERE display_name IS NOT NULL"
    )
    assert (
        store._db.create_session_strict(
            session_id="display-name-owner",
            source="test",
            display_name="shared-display-name",
        )
        is True
    )
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        store._db.create_session_strict(
            session_id="fresh-id-no-pk-collision",
            source="test",
            display_name="shared-display-name",
        )
    # The fresh, non-colliding id must not have been left half-written by the
    # failed insert.
    assert store._db.get_session("fresh-id-no-pk-collision") is None


@pytest.mark.anyio
async def test_lease_loss_after_first_durable_chunk_reports_committed_truth(
    store, monkeypatch
):
    """A durable partial child is a branch result, not a create failure."""
    source = _source()
    session_key = build_session_key(source)
    parent = store.get_or_create_session(source)
    parent_rows = [
        {"role": "user", "content": f"parent-user-{index}"}
        for index in range(501)
    ]
    assert store._db.append_messages_batch(parent.session_id, parent_rows) == 501

    # Branch copying must see all 501 durable user rows. Live replay normally
    # repairs consecutive users in-memory; this regression needs the real raw
    # SQLite transcript because copied-user truth is the behavior under test.
    monkeypatch.setattr(
        store,
        "load_transcript",
        lambda session_id: store._db.get_messages_as_conversation(session_id),
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.fire_pre_command_hook", lambda **_kwargs: None
    )

    lease_db = SessionDB(store._db.db_path)
    real_append = SessionDB.append_messages_batch
    observed: dict[str, object] = {
        "injected": False,
        "lease": None,
    }

    def append_then_refuse_second_chunk(
        db,
        session_id,
        messages,
        compression_lock_holder=None,
        turn_lease_holder=None,
        chunk_rows=None,
        turn_lease_ttl_seconds=300.0,
    ):
        try:
            inserted = real_append(
                db,
                session_id,
                messages,
                compression_lock_holder=compression_lock_holder,
                turn_lease_holder=turn_lease_holder,
                chunk_rows=chunk_rows,
                turn_lease_ttl_seconds=turn_lease_ttl_seconds,
            )
        except BaseException as exc:
            lease = observed["lease"]
            if lease is not None:
                observed["refusal_type"] = type(exc)
                held_session_id, held_holder = lease
                lease_db.release_session_turn_lease(held_session_id, held_holder)
                observed["lease"] = None
            raise

        if (
            db is store._db
            and session_id != parent.session_id
            and len(messages) == 500
            and chunk_rows is None
            and observed["injected"] is False
        ):
            # The real inner call has returned only after its transaction
            # committed. Prove that durable prefix before taking the child lease.
            durable = lease_db.get_messages(session_id)
            assert len(durable) == 500
            assert all(message["role"] == "user" for message in durable)
            observed["child_id"] = session_id
            observed["durable_before_refusal"] = len(durable)
            # Production's own copy holds a live, same-process lease on this
            # child for the whole seed, so it can never be reclaimed as a
            # dead PID (see _compression_lock_holder_process_is_dead). Force
            # it into the past instead and take it over through the normal
            # expiry path -- the same takeover an actually-different process
            # would perform after the TTL lapsed.
            competing_holder = f"pid={os.getpid()}:turn=branch-truth-regression"
            lease_db._execute_write(
                lambda conn: conn.execute(
                    "UPDATE session_turn_leases SET expires_at = 0 "
                    "WHERE conversation_id = ?",
                    (session_id,),
                )
            )
            assert lease_db.try_acquire_session_turn_lease(
                session_id, competing_holder, ttl_seconds=30.0,
            )
            observed["lease"] = (session_id, competing_holder)
            observed["injected"] = True
        return inserted

    monkeypatch.setattr(
        SessionDB, "append_messages_batch", append_then_refuse_second_chunk
    )
    runner = _runner(store)

    try:
        result = await runner._handle_message(_event("/branch committed child"))
        child_id = str(observed["child_id"])
        child = lease_db.get_session(child_id)
        durable_users = [
            message
            for message in lease_db.get_messages(child_id)
            if message["role"] == "user"
        ]

        assert observed["refusal_type"] is SessionTurnLeaseLostError
        assert child is not None
        assert child["parent_session_id"] == parent.session_id
        assert _branch_marker(child) == parent.session_id
        assert {
            "result": result,
            "durable_user_count": len(durable_users),
            "canonical_route": store.peek_session_id(session_key),
        } == {
            "result": t(
                "gateway.branch.branched_many",
                title="committed child",
                count=500,
                parent=parent.session_id,
                new=child_id,
            ),
            "durable_user_count": 500,
            "canonical_route": child_id,
        }

        descendant_result = await runner._handle_message(
            _event("/branch committed descendant")
        )
        descendant_id = store.peek_session_id(session_key)
        assert descendant_id not in {None, parent.session_id, child_id}
        descendant = lease_db.get_session(descendant_id)
        assert descendant is not None
        assert descendant["parent_session_id"] == child_id
        assert _branch_marker(descendant) == child_id
        assert descendant_result == t(
            "gateway.branch.branched_many",
            title="committed descendant",
            count=500,
            parent=child_id,
            new=descendant_id,
        )
    finally:
        lease = observed["lease"]
        if lease is not None:
            held_session_id, held_holder = lease
            lease_db.release_session_turn_lease(held_session_id, held_holder)
        lease_db.close()


@pytest.mark.anyio
async def test_wrong_target_collision_is_never_adopted_or_modified(
    store, monkeypatch
):
    """A generated-ID collision with foreign provenance is not our branch."""
    source = _source()
    session_key = build_session_key(source)
    parent = store.get_or_create_session(source)
    assert store._db.append_messages_batch(
        parent.session_id,
        [
            {"role": "user", "content": "parent-one"},
            {"role": "user", "content": "parent-two"},
        ],
    ) == 2
    foreign_parent_id = "foreign-parent"
    store._db.create_session(foreign_parent_id, source="test")

    monkeypatch.setattr(
        store,
        "load_transcript",
        lambda session_id: store._db.get_messages_as_conversation(session_id),
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.fire_pre_command_hook", lambda **_kwargs: None
    )

    real_create = store._db.create_session
    real_create_strict = store._db.create_session_strict
    collision: dict[str, str] = {}

    def create_foreign_collision_strict(*args, **kwargs):
        # Simulate a truly concurrent writer claiming the generated id a
        # moment before our own strict insert runs: by the time
        # create_session_strict executes, the id is already occupied by an
        # unrelated foreign row with its own provenance and no routing
        # identity of its own (session_key/chat_id/chat_type all NULL).
        candidate_id = kwargs["session_id"]
        collision["id"] = candidate_id
        real_create(
            session_id=candidate_id,
            source="test",
            parent_session_id=foreign_parent_id,
            model_config={"_branched_from": foreign_parent_id},
        )
        store._db.set_session_title(candidate_id, "collision sentinel")
        store._db.append_message(
            candidate_id, role="user", content="foreign sentinel"
        )
        return real_create_strict(*args, **kwargs)

    monkeypatch.setattr(
        store._db, "create_session_strict", create_foreign_collision_strict
    )

    def forbidden_recovery(*_args, **_kwargs):
        raise AssertionError("branch collision attempted broad recovery")

    for name in (
        "find_latest_gateway_session_for_peer",
        "get_session_by_title",
        "list_sessions_rich",
        "resolve_session_id",
    ):
        monkeypatch.setattr(store._db, name, forbidden_recovery)
    monkeypatch.setattr(
        store._db,
        "delete_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("branch collision attempted deletion")
        ),
    )

    runner = _runner(store)
    result = await runner._handle_message(_event("/branch collision target"))

    collision_id = collision["id"]
    row = store._db.get_session(collision_id)
    messages = store._db.get_messages(collision_id)
    assert row is not None
    assert {
        "result": result,
        "canonical_route": store.peek_session_id(session_key),
        "parent_session_id": row["parent_session_id"],
        "branch_marker": _branch_marker(row),
        "title": row["title"],
        "messages": [
            (message["role"], message["content"]) for message in messages
        ],
        # The refused branch must never have written a single column onto
        # the foreign row it collided with — not even a NULL-filling
        # enrichment of its own routing identity (session_key/chat_id/
        # chat_type), which is exactly what create_session's ON CONFLICT DO
        # UPDATE upsert used to do before the provenance check ever ran.
        "session_key": row["session_key"],
        "chat_id": row["chat_id"],
        "chat_type": row["chat_type"],
    } == {
        "result": t(
            "gateway.branch.create_failed",
            error="generated session ID collision",
        ),
        "canonical_route": parent.session_id,
        "parent_session_id": foreign_parent_id,
        "branch_marker": foreign_parent_id,
        "title": "collision sentinel",
        "messages": [("user", "foreign sentinel")],
        "session_key": None,
        "chat_id": None,
        "chat_type": None,
    }


@pytest.mark.anyio
async def test_parent_read_is_fenced_against_a_concurrent_parent_turn(
    store, monkeypatch
):
    """The parent transcript read must be fenced by the PARENT's own turn
    lease, not skipped and not fenced on the child instead.

    Without this, a legitimately in-flight parent turn can commit a new
    message between the read and the copy that the branch then silently
    never sees — a lost update, not a visible failure. This stands a
    concurrent writer in for that in-flight turn: it tries to acquire the
    IDENTICAL parent turn lease while ``/branch``'s own read is in
    progress. If the read is holder-fenced, the racer's acquire must fail;
    if the read is unfenced (the pre-fix shape), the racer succeeds.
    """
    source = _source()
    session_key = build_session_key(source)
    parent = store.get_or_create_session(source)
    store._db.append_message(parent.session_id, role="user", content="parent-one")

    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.fire_pre_command_hook", lambda **_kwargs: None
    )

    observed: dict[str, object] = {"racer_acquired": None}
    real_load_transcript = store.load_transcript

    def load_transcript_with_racer(session_id):
        # Stands in for a genuinely concurrent process's in-flight turn on
        # the SAME parent, trying to claim the identical lease while
        # /branch's own fenced read is (supposedly) holding it open.
        if session_id == parent.session_id and observed["racer_acquired"] is None:
            observed["racer_acquired"] = store._db.try_acquire_session_turn_lease(
                parent.session_id, "racer-turn-holder", ttl_seconds=5.0,
            )
            if observed["racer_acquired"]:
                store._db.release_session_turn_lease(
                    parent.session_id, "racer-turn-holder"
                )
        return real_load_transcript(session_id)

    monkeypatch.setattr(store, "load_transcript", load_transcript_with_racer)

    runner = _runner(store)
    result = await runner._handle_message(_event("/branch racer target"))

    assert observed["racer_acquired"] is False, (
        "a concurrent writer acquired the parent's turn lease while "
        "/branch's own transcript read was supposed to hold it — the read "
        "is not fenced against a concurrent parent turn"
    )
    child_id = store.peek_session_id(session_key)
    assert child_id not in {None, parent.session_id}
    assert result == t(
        "gateway.branch.branched_one",
        title="racer target",
        count=1,
        parent=parent.session_id,
        new=child_id,
    )


async def _assert_branch_claim_serializes_and_retry_succeeds(store, monkeypatch, colliding_uuids=None):
    """Hold a real parent read lease while the competing real caller is refused."""
    source = _source()
    session_key = build_session_key(source)
    parent = store.get_or_create_session(source)
    store._db.append_message(parent.session_id, role="user", content="parent-one")
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("hermes_cli.plugins.fire_pre_command_hook", lambda **_kwargs: None)

    if colliding_uuids is not None:
        # First lease, first child ID, retry lease, retry child ID. The busy
        # attempt must not consume a nonce or create a child at all.
        scripted = iter((
            colliding_uuids[0], uuid.UUID(hex="a" * 32),
            colliding_uuids[1], uuid.UUID(hex="b" * 32),
        ))
        real_uuid4 = uuid.uuid4
        minted = []

        def scripted_uuid4():
            value = next(scripted, None)
            if value is None:
                value = real_uuid4()
            minted.append(value)
            return value

        monkeypatch.setattr(uuid, "uuid4", scripted_uuid4)

    first_parked = threading.Event()
    release_first = threading.Event()
    real_load_transcript = store.load_transcript

    def gated_load(session_id):
        if session_id == parent.session_id:
            first_parked.set()
            assert release_first.wait(timeout=10.0), "first read was never released"
        return real_load_transcript(session_id)

    monkeypatch.setattr(store, "load_transcript", gated_load)
    acquisitions = []
    releases = []
    real_acquire = store._db.try_acquire_session_turn_lease
    real_release = store._db.release_session_turn_lease

    def recording_acquire(session_id, holder, **kwargs):
        result = real_acquire(session_id, holder, **kwargs)
        acquisitions.append((session_id, holder, result))
        return result

    def recording_release(session_id, holder):
        if ":turn=branch-read-" in holder:
            releases.append((session_id, holder))
        return real_release(session_id, holder)

    monkeypatch.setattr(store._db, "try_acquire_session_turn_lease", recording_acquire)
    monkeypatch.setattr(store._db, "release_session_turn_lease", recording_release)

    def session_ids():
        with sqlite3.connect(store._db.db_path) as conn:
            return {row[0] for row in conn.execute("SELECT id FROM sessions")}

    runner = _runner(store)
    first_task = asyncio.create_task(runner._handle_message(_event("/branch first")))
    try:
        assert await asyncio.to_thread(first_parked.wait, 10.0), (
            "first /branch never reached its fenced parent read"
        )
        assert len(acquisitions) == 1
        assert acquisitions[0][0] == parent.session_id
        first_holder = acquisitions[0][1]
        assert acquisitions[0][2] is True
        assert releases == []

        # The second caller overlaps both the atomic claim and the parent's
        # held DB read lease. It must be busy before touching either DB path.
        busy = await runner._handle_message(_event("/branch second"))
        assert busy == "⏳ Session is busy — wait for the current response before `/branch`."
        assert acquisitions == [(parent.session_id, first_holder, True)]
        assert releases == []
        assert session_ids() == {parent.session_id}, "busy branch created a child"
        assert store.peek_session_id(session_key) == parent.session_id
        with sqlite3.connect(store._db.db_path) as conn:
            assert conn.execute(
                "SELECT holder FROM session_turn_leases WHERE conversation_id = ?",
                (parent.session_id,),
            ).fetchone() == (first_holder,), "busy branch released/replaced first holder"
        assert not first_task.done()
    finally:
        release_first.set()
    first_result = await first_task
    first_child = store.peek_session_id(session_key)
    assert first_child not in {None, parent.session_id}
    assert store._db.get_session(first_child)["parent_session_id"] == parent.session_id
    assert first_result == t(
        "gateway.branch.branched_one", title="first", count=1,
        parent=parent.session_id, new=first_child,
    )
    assert releases == [(parent.session_id, first_holder)]

    # After the claim is released, the same second command succeeds against
    # the newly active child. No second lease attempt was ever made on the
    # first parent, so compare full holders across the two successful calls.
    second_result = await runner._handle_message(_event("/branch second"))
    second_child = store.peek_session_id(session_key)
    assert second_child not in {None, parent.session_id, first_child}
    assert store._db.get_session(second_child)["parent_session_id"] == first_child
    assert second_result == t(
        "gateway.branch.branched_one", title="second", count=1,
        parent=first_child, new=second_child,
    )
    parent_reads = [
        entry for entry in acquisitions if ":turn=branch-read-" in entry[1]
    ]
    child_seeds = [
        entry for entry in acquisitions if ":turn=branch-seed:session=" in entry[1]
    ]
    assert len(acquisitions) == len(parent_reads) + len(child_seeds)
    assert len(parent_reads) == 2
    assert parent_reads[0] == (parent.session_id, first_holder, True)
    assert parent_reads[1][0] == first_child
    assert parent_reads[1][2] is True
    second_holder = parent_reads[1][1]
    assert second_holder != first_holder, "sequential branches aliased a read holder"
    assert len(child_seeds) == 2
    assert child_seeds[0][0] == first_child
    assert child_seeds[1][0] == second_child
    assert all(entry[2] is True for entry in child_seeds)
    assert child_seeds[0][1].endswith(f":turn=branch-seed:session={first_child}")
    assert child_seeds[1][1].endswith(f":turn=branch-seed:session={second_child}")
    assert releases == [
        (parent.session_id, first_holder), (first_child, second_holder),
    ]
    assert session_ids() == {parent.session_id, first_child, second_child}
    if colliding_uuids is not None:
        assert minted == [
            colliding_uuids[0], uuid.UUID(hex="a" * 32),
            colliding_uuids[1], uuid.UUID(hex="b" * 32),
        ], "busy call consumed a branch nonce or retry used an unexpected nonce"
        assert colliding_uuids[0].hex in first_holder
        assert colliding_uuids[1].hex in second_holder, (
            "a shared 8-character prefix truncated/aliased the retry holder"
        )


@pytest.mark.anyio
async def test_concurrent_branch_calls_on_same_parent_never_alias_the_read_lease(
    store, monkeypatch
):
    """A busy concurrent branch cannot acquire/release the first read lease;
    after the first switch, a retry succeeds with a distinct full holder."""
    await _assert_branch_claim_serializes_and_retry_succeeds(store, monkeypatch)


@pytest.mark.anyio
async def test_prefix_collision_never_admits_second_branch_as_same_holder_reentry(
    store, monkeypatch
):
    """Even shared eight-character UUID prefixes cannot alias lease holders
    after a busy overlap and subsequent successful retry."""
    shared_prefix = "deadbeef"
    colliding_uuids = (
        uuid.UUID(hex=shared_prefix + "0" * 24),
        uuid.UUID(hex=shared_prefix + "1" * 24),
    )
    assert colliding_uuids[0].hex[:8] == colliding_uuids[1].hex[:8] == shared_prefix
    assert colliding_uuids[0].hex != colliding_uuids[1].hex
    await _assert_branch_claim_serializes_and_retry_succeeds(
        store, monkeypatch, colliding_uuids
    )


@pytest.mark.anyio
async def test_normal_success_reports_durable_count_and_routes_exact_child(
    store, monkeypatch
):
    source = _source()
    session_key = build_session_key(source)
    parent = store.get_or_create_session(source)
    parent_rows = [
        {"role": "user", "content": f"normal-user-{index}"}
        for index in range(501)
    ]
    assert store._db.append_messages_batch(parent.session_id, parent_rows) == 501
    monkeypatch.setattr(
        store,
        "load_transcript",
        lambda session_id: store._db.get_messages_as_conversation(session_id),
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.fire_pre_command_hook", lambda **_kwargs: None
    )

    runner = _runner(store)
    result = await runner._handle_message(_event("/branch normal child"))

    child_id = store.peek_session_id(session_key)
    assert child_id not in {None, parent.session_id}
    child = store._db.get_session(child_id)
    assert child is not None
    durable_users = [
        message
        for message in store._db.get_messages(child_id)
        if message["role"] == "user"
    ]
    assert child["parent_session_id"] == parent.session_id
    assert _branch_marker(child) == parent.session_id
    assert len(durable_users) == 501
    assert result == t(
        "gateway.branch.branched_many",
        title="normal child",
        count=501,
        parent=parent.session_id,
        new=child_id,
    )


@pytest.mark.anyio
async def test_true_prepublication_failure_preserves_parent_and_sentinel(
    store, monkeypatch
):
    source = _source()
    session_key = build_session_key(source)
    parent = store.get_or_create_session(source)
    store._db.append_message(parent.session_id, role="user", content="parent")
    sentinel_id = "unrelated-sentinel"
    store._db.create_session(sentinel_id, source="test")
    store._db.append_message(sentinel_id, role="user", content="sentinel")
    monkeypatch.setattr(
        store,
        "load_transcript",
        lambda session_id: store._db.get_messages_as_conversation(session_id),
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.fire_pre_command_hook", lambda **_kwargs: None
    )

    expected_error = RuntimeError("forced pre-publication create failure")
    attempted: dict[str, str] = {}

    def fail_before_create(*args, **kwargs):
        attempted["id"] = kwargs["session_id"]
        raise expected_error

    monkeypatch.setattr(store._db, "create_session_strict", fail_before_create)
    runner = _runner(store)
    result = await runner._handle_message(_event("/branch never published"))

    assert result == t("gateway.branch.create_failed", error=expected_error)
    assert store._db.get_session(attempted["id"]) is None
    assert store.peek_session_id(session_key) == parent.session_id
    assert [
        (message["role"], message["content"])
        for message in store._db.get_messages(sentinel_id)
    ] == [("user", "sentinel")]


@pytest.mark.anyio
async def test_exact_child_switch_failure_remains_switch_failed(store, monkeypatch):
    source = _source()
    session_key = build_session_key(source)
    parent = store.get_or_create_session(source)
    assert store._db.append_messages_batch(
        parent.session_id,
        [
            {"role": "user", "content": "switch-one"},
            {"role": "user", "content": "switch-two"},
        ],
    ) == 2
    monkeypatch.setattr(
        store,
        "load_transcript",
        lambda session_id: store._db.get_messages_as_conversation(session_id),
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.fire_pre_command_hook", lambda **_kwargs: None
    )

    attempted: dict[str, str] = {}

    def refuse_switch(received_session_key, target_session_id, *, command_claim):
        assert command_claim is not None
        assert store.command_claim_owned(parent, command_claim)
        attempted["session_key"] = received_session_key
        attempted["child_id"] = target_session_id
        return None

    monkeypatch.setattr(store, "switch_session", refuse_switch)
    runner = _runner(store)
    result = await runner._handle_message(_event("/branch unswitched child"))

    child_id = attempted["child_id"]
    child = store._db.get_session(child_id)
    assert result == t("gateway.branch.switch_failed")
    assert attempted["session_key"] == session_key
    assert store.peek_session_id(session_key) == parent.session_id
    assert child is not None
    assert child["parent_session_id"] == parent.session_id
    assert _branch_marker(child) == parent.session_id
    assert [
        (message["role"], message["content"])
        for message in store._db.get_messages(child_id)
    ] == [("user", "switch-one"), ("user", "switch-two")]


@pytest.mark.anyio
async def test_title_collision_reports_untitled_branch_not_a_titled_one(
    store, monkeypatch
):
    """A durable branch whose title write lost to a collision must not claim
    the title as applied — the branch itself still succeeds and switches."""
    source = _source()
    session_key = build_session_key(source)
    parent = store.get_or_create_session(source)
    store._db.append_message(parent.session_id, role="user", content="parent-one")

    # Some other, unrelated session already owns the name the branch will ask
    # for.
    other_id = "title-holder"
    store._db.create_session(other_id, source="test")
    assert store._db.set_session_title(other_id, "taken") is True

    monkeypatch.setattr(
        store,
        "load_transcript",
        lambda session_id: store._db.get_messages_as_conversation(session_id),
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.fire_pre_command_hook", lambda **_kwargs: None
    )

    runner = _runner(store)
    result = await runner._handle_message(_event("/branch taken"))

    child_id = store.peek_session_id(session_key)
    assert child_id not in {None, parent.session_id}
    child = store._db.get_session(child_id)

    # The branch itself is a genuine success: durable child row, copied
    # history, and the route switch all landed.
    assert child is not None
    assert child["parent_session_id"] == parent.session_id
    assert _branch_marker(child) == parent.session_id
    assert [
        (message["role"], message["content"])
        for message in store._db.get_messages(child_id)
    ] == [("user", "parent-one")]

    # The title write did NOT land: the collision means the child stays
    # untitled and the other session keeps the name.
    assert child["title"] is None
    assert store._db.get_session(other_id)["title"] == "taken"

    # The outward message must not assert the untrue "titled" fact.
    assert "taken" not in result.split("\n")[0]
    assert result != t(
        "gateway.branch.branched_one",
        title="taken",
        count=1,
        parent=parent.session_id,
        new=child_id,
    )
    assert result == t(
        "gateway.branch.branched_one_untitled",
        count=1,
        parent=parent.session_id,
        new=child_id,
    ) + t(
        "gateway.branch.title_not_set_note",
        title="taken",
        error=f"Title 'taken' is already in use by session {other_id}",
    )
