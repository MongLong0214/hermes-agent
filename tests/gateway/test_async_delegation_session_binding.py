"""Gateway-side session binding for async delegations (#57498, #55578).

Three invariants on the messaging-gateway surface, mirroring the TUI rules:

1. Completions are pinned to the spawning session (contributor commit).
2. A dead/ended spawning session is never resurrected: the injection is
   dropped, fail-closed (never rerouted to the peer's current session).
3. /new interrupts the old conversation's in-flight async delegations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import tools.async_delegation as ad


@pytest.fixture(autouse=True)
def _reset_async_delegation():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _seed_record(delegation_id, session_key="", parent_session_id="", status="running"):
    fn = MagicMock()
    with ad._records_lock:
        ad._records[delegation_id] = {
            "delegation_id": delegation_id,
            "status": status,
            "session_key": session_key,
            "parent_session_id": parent_session_id,
            "interrupt_fn": fn,
        }
    return fn


class TestInterruptForSessionByParentId:
    def test_parent_session_id_selector(self):
        mine = _seed_record("d1", session_key="agent:main:telegram:dm:1", parent_session_id="sess_old")
        other = _seed_record("d2", session_key="agent:main:telegram:dm:2", parent_session_id="sess_other")
        n = ad.interrupt_for_session(parent_session_id="sess_old")
        assert n == 1
        mine.assert_called_once()
        other.assert_not_called()


class TestGatewayPinningFailsClosed:
    """The gateway must follow only verified compression continuations."""

    @staticmethod
    def _entry(session_id):
        from datetime import datetime

        from gateway.config import Platform
        from gateway.session import SessionEntry

        return SessionEntry(
            session_key="agent:main:telegram:group:-100:4",
            session_id=session_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            platform=Platform.TELEGRAM,
            chat_type="group",
        )

    def _make_runner(
        self,
        rows,
        *,
        compression_tip=None,
        compression_error=None,
        switched_entry=None,
    ):
        from gateway.run import GatewayRunner
        from gateway.session import AsyncSessionStore

        runner = object.__new__(GatewayRunner)
        db = MagicMock()
        db.get_session = AsyncMock(side_effect=lambda session_id: rows.get(session_id))
        db.get_compression_tip = AsyncMock(
            return_value=compression_tip,
            side_effect=compression_error,
        )
        runner._session_db = db
        runner.session_store = MagicMock()
        runner.session_store.switch_session = MagicMock(return_value=switched_entry)
        runner.session_store.advance_compression_session = MagicMock(
            return_value=switched_entry
        )
        runner._async_session_store = AsyncSessionStore(runner.session_store)
        return runner

    @staticmethod
    def _assert_no_route_change(runner):
        getattr(runner.session_store, "switch_session").assert_not_called()
        getattr(
            runner.session_store, "advance_compression_session"
        ).assert_not_called()


    @pytest.mark.asyncio
    async def test_live_spawning_session_cannot_rebind_different_route(self):
        current = self._entry("sess_current")
        pinned = self._entry("sess_live")
        runner = self._make_runner(
            {"sess_live": {"id": "sess_live", "ended_at": None}},
            switched_entry=pinned,
        )

        resolved = await runner._resolve_async_delegation_session(
            current, "sess_live"
        )

        assert resolved is None
        self._assert_no_route_change(runner)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["delegation", "process"])
    @pytest.mark.parametrize("state", ["cold", "busy", "mismatch", "idle", "same"])
    async def test_completion_handler_observes_existing_route(self, kind, state):
        from types import SimpleNamespace
        from gateway.config import Platform

        current = self._entry("sess_parent")
        pinned_id = "sess_parent" if state == "same" else "sess_child"
        row = {"id": pinned_id, "ended_at": None}
        if state == "idle":
            row.update(ended_at="2026-08-09T00:00:00", end_reason="idle")
        runner = self._make_runner({pinned_id: row}, switched_entry=self._entry(pinned_id))
        runner._recover_telegram_topic_thread_id = lambda source: None
        runner._session_key_for_source = lambda source: current.session_key
        runner.session_store.lookup_by_session_key.return_value = None if state == "cold" else current
        runner.session_store.get_or_create_session.return_value = current
        runner._running_agents = {current.session_key: object()} if state == "busy" else {}
        # Stop at the real handler's post-resolution seam, before any LLM/I/O.
        class Delivered(Exception):
            pass
        runner._cache_session_source = MagicMock(side_effect=Delivered)
        source = SimpleNamespace(platform=Platform.TELEGRAM, user_name="user",
                                 user_id="1", chat_id="1", thread_id=None)
        event = SimpleNamespace(text="completed", internal=True, metadata={
            "gateway_session_key": current.session_key,
            "gateway_session_id": pinned_id,
            "completion_kind": kind,
        })
        if state == "same":
            with pytest.raises(Delivered):
                await runner._handle_message_with_agent(event, source, current.session_key, 1)
            runner._cache_session_source.assert_called_once_with(current.session_key, source)
        else:
            await runner._handle_message_with_agent(event, source, current.session_key, 1)
            runner._cache_session_source.assert_not_called()
        runner.session_store.get_or_create_session.assert_not_called()
        self._assert_no_route_change(runner)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("state", ["cold", "mismatch", "same"])
    async def test_injection_checks_owner_before_adapter_acceptance(self, state):
        from types import SimpleNamespace
        from gateway.config import Platform

        current = self._entry("sess_parent")
        runner = self._make_runner({"sess_child": {"id": "sess_child", "ended_at": None}})
        source = SimpleNamespace(platform=Platform.TELEGRAM, chat_id="1", thread_id=None)
        runner._build_process_event_source = lambda evt: source
        runner._session_key_for_source = lambda src: current.session_key
        runner.session_store.lookup_by_session_key.return_value = None if state == "cold" else current
        runner.adapters = {Platform.TELEGRAM: SimpleNamespace(handle_message=AsyncMock())}
        parent = "sess_parent" if state == "same" else "sess_child"
        runner._session_db.get_session = AsyncMock(return_value={"id": parent, "ended_at": None})
        result = await runner._inject_watch_notification("completed", {"parent_session_id": parent})
        adapter = runner.adapters[Platform.TELEGRAM]
        if state == "same":
            assert result is True
            adapter.handle_message.assert_awaited_once()
            assert adapter.handle_message.call_args.args[0].metadata["gateway_session_id"] == parent
        else:
            assert result is None
            adapter.handle_message.assert_not_awaited()
        self._assert_no_route_change(runner)

    @pytest.mark.asyncio
    async def test_non_compression_ended_parent_drops(self):
        current = self._entry("sess_old")
        runner = self._make_runner(
            {
                "sess_old": {
                    "id": "sess_old",
                    "ended_at": "2026-07-08T00:00:00",
                    "end_reason": "session_reset",
                }
            }
        )

        resolved = await runner._resolve_async_delegation_session(
            current, "sess_old"
        )

        assert resolved is None
        self._assert_no_route_change(runner)


    @pytest.mark.asyncio
    async def test_intermediate_compression_route_advances_to_same_live_tip(self):
        current = self._entry("sess_middle")
        tip = self._entry("sess_tip")
        runner = self._make_runner(
            {
                "sess_parent": {
                    "id": "sess_parent",
                    "ended_at": "2026-07-08T00:00:00",
                    "end_reason": "compression",
                },
                "sess_middle": {
                    "id": "sess_middle",
                    "ended_at": "2026-07-08T00:01:00",
                    "end_reason": "compression",
                    "parent_session_id": "sess_parent",
                },
                "sess_tip": {
                    "id": "sess_tip",
                    "ended_at": None,
                    "parent_session_id": "sess_middle",
                },
            },
            compression_tip="sess_tip",
            switched_entry=tip,
        )

        resolved = await runner._resolve_async_delegation_session(
            current, "sess_parent"
        )

        assert resolved is tip
        getattr(
            runner.session_store, "advance_compression_session"
        ).assert_called_once_with(current.session_key, "sess_middle", "sess_tip")

    @pytest.mark.asyncio
    async def test_compression_parent_follows_real_sessiondb_lineage(self, tmp_path):
        from gateway.run import GatewayRunner
        from gateway.session import AsyncSessionStore
        from hermes_state import AsyncSessionDB, SessionDB

        session_db = SessionDB(db_path=tmp_path / "state.db")
        session_db.create_session("sess_parent", source="telegram")
        session_db.end_session("sess_parent", end_reason="compression")
        session_db.create_session(
            "sess_tip",
            source="telegram",
            parent_session_id="sess_parent",
        )

        current = self._entry("sess_parent")
        tip = self._entry("sess_tip")
        runner = object.__new__(GatewayRunner)
        runner._session_db = AsyncSessionDB(session_db)
        runner.session_store = MagicMock()
        runner.session_store.switch_session = MagicMock(return_value=tip)
        runner.session_store.advance_compression_session = MagicMock(return_value=tip)
        runner._async_session_store = AsyncSessionStore(runner.session_store)

        resolved = await runner._resolve_async_delegation_session(
            current, "sess_parent"
        )

        assert resolved is tip
        getattr(
            runner.session_store, "advance_compression_session"
        ).assert_called_once_with(current.session_key, "sess_parent", "sess_tip")


class TestResetHandlerInterruptsDelegations:
    def test_reset_command_calls_interrupt_for_session(self):
        """The /new handler must sever the old conversation's delegations."""
        import inspect
        from gateway import slash_commands

        src = inspect.getsource(slash_commands.GatewaySlashCommandsMixin._handle_reset_command)
        assert "interrupt_for_session" in src
        assert "session_reset" in src
