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
    async def test_live_spawning_session_rebinds_from_different_route(self):
        current = self._entry("sess_current")
        pinned = self._entry("sess_live")
        runner = self._make_runner(
            {"sess_live": {"id": "sess_live", "ended_at": None}},
            switched_entry=pinned,
        )

        resolved = await runner._resolve_async_delegation_session(
            current, "sess_live"
        )

        assert resolved is pinned
        getattr(runner.session_store, "switch_session").assert_called_once_with(
            current.session_key, "sess_live"
        )

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
    @pytest.mark.asyncio
    async def test_reset_interrupts_old_delegations_without_touching_other_sessions(self):
        """A /new boundary severs only the old conversation's delegations."""
        from datetime import datetime
        from types import SimpleNamespace

        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent
        from gateway.run import GatewayRunner
        from gateway.session import AsyncSessionStore, SessionEntry, SessionSource, build_session_key

        source = SessionSource(
            platform=Platform.TELEGRAM, user_id="1", chat_id="1", chat_type="dm"
        )
        key = build_session_key(source)
        old = SessionEntry(
            session_key=key, session_id="sess_old", created_at=datetime.now(),
            updated_at=datetime.now(), platform=Platform.TELEGRAM, chat_type="dm",
            origin=source,
        )
        new = SessionEntry(
            session_key=key, session_id="sess_new", created_at=datetime.now(),
            updated_at=datetime.now(), platform=Platform.TELEGRAM, chat_type="dm",
            origin=source,
        )
        mine = _seed_record("mine", session_key=key, parent_session_id="sess_old")
        other = _seed_record("other", session_key="another-route", parent_session_id="sess_other")

        runner = object.__new__(GatewayRunner)
        runner.session_store = MagicMock()
        runner.session_store._entries = {key: old}
        runner.session_store.reset_session.return_value = new
        runner._async_session_store = AsyncSessionStore(runner.session_store)
        runner._session_key_for_source = lambda _source: key
        runner._invalidate_session_run_generation = MagicMock()
        runner._release_running_agent_state = MagicMock()
        runner._evict_cached_agent = MagicMock()
        runner._clear_conversation_scope = MagicMock()
        runner._reset_notice_session_info = MagicMock(return_value="")
        runner._telegram_topic_new_header = MagicMock(return_value="")
        runner._is_telegram_topic_lane = MagicMock(return_value=False)
        runner._finalize_session_off_loop = AsyncMock()
        runner._session_db = None
        runner.hooks = SimpleNamespace(emit=AsyncMock())

        await runner._handle_reset_command(MessageEvent(text="/new", source=source))

        mine.assert_called_once()
        other.assert_not_called()
        runner.session_store.reset_session.assert_called_once_with(
            key, command_claim=runner.session_store.claim_session_command.return_value
        )
        runner.session_store.release_session_command.assert_called_once_with(
            old, runner.session_store.claim_session_command.return_value
        )
