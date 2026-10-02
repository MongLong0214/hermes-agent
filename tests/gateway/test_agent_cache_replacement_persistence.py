"""L4-2/R67-3 — ordinary cache replacement (a changed config signature building a fresh agent, or
cross-process message-count invalidation) must not silently drop an agent whose transcript hasn't
reached state.db. ``_evict_cached_agent`` already retains such an agent for explicit/LRU/idle-TTL
eviction (the base L4-2 fix); this closes the two paths the reviewer found that bypass that guard
entirely: the cross-process pop in ``_lookup_cached_agent``/``_release_evicted_agent`` and the
plain dict overwrite in ``_resolve_turn_agent`` when a signature mismatch discards the old entry
outright. /stop, the reaper and /resume are deliberate exceptions (``require_persisted=False``)
and are untouched by this fix.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest


def _make_runner():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._running_agents = {}
    return runner


def _unflushed_agent():
    agent = MagicMock()
    agent._session_messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    agent._last_flushed_db_idx = 0  # the assistant reply never reached state.db
    return agent


def _flushed_agent():
    agent = MagicMock()
    agent._session_messages = [{"role": "user", "content": "x", "_db_persisted": True}]
    agent._last_flushed_db_idx = 1
    return agent


# ── unit: the new composed helper ──────────────────────────────────────────────────────────


class TestEnsurePersistedThenReleaseSoft:
    def test_flushes_before_releasing_when_transcript_has_not_caught_up(self):
        runner = _make_runner()
        agent = _unflushed_agent()
        calls = []
        runner._flush_agent_transcript_at_shutdown = lambda a: calls.append(("flush", a))
        runner._release_evicted_agent_soft = lambda a: calls.append(("release", a))

        runner._ensure_persisted_then_release_soft(agent)

        assert calls == [("flush", agent), ("release", agent)], (
            "flush-or-spool must run BEFORE the soft release clears _session_messages"
        )

    def test_skips_the_flush_when_already_caught_up(self):
        runner = _make_runner()
        agent = _flushed_agent()
        calls = []
        runner._flush_agent_transcript_at_shutdown = lambda a: calls.append(("flush", a))
        runner._release_evicted_agent_soft = lambda a: calls.append(("release", a))

        runner._ensure_persisted_then_release_soft(agent)

        assert calls == [("release", agent)]


# ── unit: retiring an overwritten cache slot ───────────────────────────────────────────────


class TestRetireReplacedAgent:
    def test_schedules_the_safe_release_for_an_unflushed_replaced_agent(self):
        runner = _make_runner()
        old_agent = _unflushed_agent()
        scheduled = []
        runner._spawn_release_thread = lambda target, args, name, **kw: scheduled.append((target, args))

        runner._retire_replaced_agent("sess-1", (old_agent, "old-sig", 3, "sid-1"))

        assert len(scheduled) == 1
        target, args = scheduled[0]
        assert target == runner._ensure_persisted_then_release_soft
        assert args == (old_agent,)

    def test_skips_a_none_entry(self):
        runner = _make_runner()
        scheduled = []
        runner._spawn_release_thread = lambda *a, **kw: scheduled.append(a)
        runner._retire_replaced_agent("sess-1", None)
        assert scheduled == []

    def test_skips_a_currently_running_agent(self):
        """A concurrent caller still holding the slot owns its own release — never race it."""
        runner = _make_runner()
        agent = _unflushed_agent()
        runner._running_agents = {"sess-1": agent}
        scheduled = []
        runner._spawn_release_thread = lambda *a, **kw: scheduled.append(a)

        runner._retire_replaced_agent("sess-1", (agent, "sig", 1, "sid-1"))

        assert scheduled == []


# ── integration: TurnRunner wiring for the two reviewer-cited call sites ──────────────────


def _make_turn_runner(runner, *, session_key="sess-key", session_id="sid-1"):
    from gateway.config import Platform
    from gateway.run_turn_runner import TurnRunner
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext

    ctx = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1"),
        session_key=session_key, session_id=session_id, user_config={}, message="hi",
        enabled_toolsets=None, disabled_toolsets=None,
    )
    return TurnRunner(runner, ctx)


class TestResolveTurnAgentRetainsUnflushedOnReplacement:
    """Reproduces the reviewer's probe: explicit eviction retains an unflushed agent, then an
    ordinary replacement (signature change) used to overwrite the slot and lose it outright."""

    def test_signature_change_flushes_the_old_agent_instead_of_dropping_it(self, monkeypatch):
        import gateway.run_agent_cache as run_agent_cache

        runner = _make_runner()
        runner._session_db = None
        monkeypatch.setattr(run_agent_cache.GatewayAgentCacheMixin, "_agent_config_signature",
                             staticmethod(lambda *a, **k: "new-sig"))
        monkeypatch.setattr(run_agent_cache.GatewayAgentCacheMixin, "_extract_cache_busting_config",
                             classmethod(lambda cls, cfg: {}))

        old_agent = _unflushed_agent()
        runner._agent_cache["sess-key"] = (old_agent, "old-sig", 3, "sid-1")

        tr = _make_turn_runner(runner)
        monkeypatch.setattr(tr, "_cached_sid_is_dead", lambda cache_lock, cache: (None, False))
        monkeypatch.setattr(tr, "_current_message_count", lambda: None)
        monkeypatch.setattr(tr, "_skip_context_files", lambda platform_key: False)
        new_agent = MagicMock()
        monkeypatch.setattr(tr, "_build_fresh_agent", lambda *a, **k: new_agent)

        released = []
        runner._spawn_release_thread = lambda target, args, name, **kw: (released.append((target, args)), target(*args))[-1]

        agent, reused = tr._resolve_turn_agent(
            {"model": "m", "runtime": {}}, "telegram", "", 50, None, {},
        )

        assert reused is False
        assert agent is new_agent
        assert runner._agent_cache["sess-key"][0] is new_agent
        # The old entry's flush-or-spool ran (not just a bare soft release that would have wiped
        # its unflushed messages unread).
        assert len(released) == 1
        assert released[0][1] == (old_agent,)
        # The real _flush_agent_transcript_at_shutdown ran (not mocked) and attempted the flush
        # BEFORE the soft release below rebinds _session_messages to a fresh empty list.
        old_agent._flush_messages_to_session_db.assert_called_once_with(
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
        )
        assert old_agent._session_messages == []  # soft release still ran after the flush attempt
        old_agent.release_clients.assert_called_once()

    def test_reused_agent_leaves_the_cache_slot_untouched(self, monkeypatch):
        """Guardrail: an unchanged signature must keep reusing the cached agent — no replacement,
        no release at all."""
        import gateway.run_agent_cache as run_agent_cache

        runner = _make_runner()
        runner._session_db = None
        monkeypatch.setattr(run_agent_cache.GatewayAgentCacheMixin, "_agent_config_signature",
                             staticmethod(lambda *a, **k: "same-sig"))
        monkeypatch.setattr(run_agent_cache.GatewayAgentCacheMixin, "_extract_cache_busting_config",
                             classmethod(lambda cls, cfg: {}))

        cached_agent = _flushed_agent()
        runner._agent_cache["sess-key"] = (cached_agent, "same-sig", None, "sid-1")

        tr = _make_turn_runner(runner)
        monkeypatch.setattr(tr, "_cached_sid_is_dead", lambda cache_lock, cache: (None, False))
        monkeypatch.setattr(tr, "_current_message_count", lambda: None)
        monkeypatch.setattr(tr, "_skip_context_files", lambda platform_key: False)
        runner._init_cached_agent_for_turn = lambda *a, **k: None
        runner._apply_fallback_chain_to_agent = lambda *a, **k: None
        runner._refresh_fallback_model = lambda: None

        released = []
        runner._spawn_release_thread = lambda target, args, name, **kw: released.append((target, args))

        agent, reused = tr._resolve_turn_agent(
            {"model": "m", "runtime": {}}, "telegram", "", 50, None, {},
        )

        assert reused is True
        assert agent is cached_agent
        assert released == []


class TestCrossProcessInvalidationFlushesBeforeRelease:
    """The message-count-changed branch of ``_lookup_cached_agent`` pops the stale entry; the
    reviewer's probe showed the subsequent soft release cleared its unflushed messages to ``[]``
    without ever attempting a flush or spool."""

    def test_message_count_mismatch_flushes_the_popped_agent_before_release(self, monkeypatch):
        import gateway.run_agent_cache as run_agent_cache

        runner = _make_runner()
        runner._session_db = None
        monkeypatch.setattr(run_agent_cache.GatewayAgentCacheMixin, "_agent_config_signature",
                             staticmethod(lambda *a, **k: "sig-1"))
        monkeypatch.setattr(run_agent_cache.GatewayAgentCacheMixin, "_extract_cache_busting_config",
                             classmethod(lambda cls, cfg: {}))

        old_agent = _unflushed_agent()
        # cached_mc=3 (stale snapshot) vs the live count below (5) -> message_count changed branch.
        runner._agent_cache["sess-key"] = (old_agent, "sig-1", 3, "sid-1")

        tr = _make_turn_runner(runner)
        monkeypatch.setattr(tr, "_cached_sid_is_dead", lambda cache_lock, cache: (None, False))
        monkeypatch.setattr(tr, "_current_message_count", lambda: 5)
        monkeypatch.setattr(tr, "_skip_context_files", lambda platform_key: False)
        new_agent = MagicMock()
        monkeypatch.setattr(tr, "_build_fresh_agent", lambda *a, **k: new_agent)

        released = []
        runner._spawn_release_thread = lambda target, args, name, **kw: (released.append((target, args)), target(*args))[-1]

        agent, reused = tr._resolve_turn_agent(
            {"model": "m", "runtime": {}}, "telegram", "", 50, None, {},
        )

        assert reused is False
        assert agent is new_agent
        assert "sess-key" not in runner._agent_cache or runner._agent_cache["sess-key"][0] is new_agent
        assert len(released) == 1
        assert released[0][0] == runner._ensure_persisted_then_release_soft
        assert released[0][1] == (old_agent,)
        old_agent.release_clients.assert_called_once()
