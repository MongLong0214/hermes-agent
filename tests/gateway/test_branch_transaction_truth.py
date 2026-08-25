"""Transactional result-truth regressions for gateway ``/branch``."""

from __future__ import annotations

import json
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
    make_turn_lease_holder,
)


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
                lease.__exit__(type(exc), exc, exc.__traceback__)
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
            lease = lease_db.session_turn_lease(
                session_id,
                make_turn_lease_holder("branch-truth-regression"),
                wait_seconds=0.0,
                ttl_seconds=30.0,
                reload_messages=False,
            )
            lease.__enter__()
            observed["lease"] = lease
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
            lease.__exit__(None, None, None)
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
    collision: dict[str, str] = {}

    def create_foreign_collision(*args, **kwargs):
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
        return real_create(*args, **kwargs)

    monkeypatch.setattr(store._db, "create_session", create_foreign_collision)

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
    }


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

    monkeypatch.setattr(store._db, "create_session", fail_before_create)
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

    def refuse_switch(received_session_key, target_session_id):
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
