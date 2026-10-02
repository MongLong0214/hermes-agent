"""Recovery-release threads must run under the OWNING profile's scope, not the default (R67-R1).

Two ordinary-replacement sites hand a release off to ``_spawn_release_thread``, which enters the
owning profile's scope via ``_run_release_in_profile_scope(target, args, session_key)`` so that a
flush-or-spool recovery write (``_ensure_persisted_then_release_soft`` ->
``_flush_agent_transcript_at_shutdown`` -> the on-disk snapshot) lands under the SESSION's home, not
whichever profile happens to be active on the calling thread:

- ``_retire_replaced_agent`` (gateway/run_agent_cache.py) — a changed config signature building a
  fresh agent, retiring the old one.
- ``_release_evicted_agent`` (gateway/run_turn_runner.py) — cross-process message-count
  invalidation.

Both omitted the ``session_key=`` kwarg, so ``_spawn_release_thread`` defaulted it to ``None`` and
the release ran under whatever profile happened to be active (the default profile home), carrying
that profile's private transcript into another profile's directory on an A -> B -> A multiplex
probe. Each test drives the REAL call site with two distinct profile homes. ``_spawn_release_thread``
itself is replaced with a synchronous call to the SAME production
``_run_release_in_profile_scope`` it would otherwise hand to a background daemon thread — real
scoping code, just off the timing-sensitive thread so the assertion isn't racing a background
worker (the recovery snapshot's on-disk write still goes through the unmocked production path).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import secret_scope as ss
from gateway.run import GatewayRunner, _SESSION_DB_UNPINNED, _profile_runtime_scope


@pytest.fixture(autouse=True)
def _multiplex_off_after():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


def _agent_with_failing_flush(session_id: str):
    """An agent whose transcript never reached SQLite, so release must flush-or-spool it."""
    return SimpleNamespace(
        _flush_messages_to_session_db=lambda msgs: False,
        _session_messages=[{"role": "user", "content": f"private turn for {session_id}"}],
        session_id=session_id,
        release_clients=lambda: None,
    )


def _make_runner(profile_homes: dict, monkeypatch):
    """A runner whose ``_spawn_release_thread`` runs the SAME production
    ``_run_release_in_profile_scope`` synchronously, instead of on a background daemon thread —
    deterministic for the test, not a stand-in for the scoping logic under test."""
    runner = object.__new__(GatewayRunner)
    runner._session_db_pinned = _SESSION_DB_UNPINNED
    runner._session_db = None
    store = MagicMock()
    store._profile_home_for_key = lambda key: profile_homes.get(key)
    runner.session_store = store
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._running_agent_ids = lambda: set()

    def _sync_spawn(self, target, args, name, *, inline_fallback=True, session_key=None):
        self._run_release_in_profile_scope(target, args, session_key)

    monkeypatch.setattr(GatewayRunner, "_spawn_release_thread", _sync_spawn)
    return runner, store


def test_retire_replaced_agent_spools_under_the_sessions_own_profile(tmp_path: Path, monkeypatch):
    default_home = tmp_path / "default"
    home_a = tmp_path / "profiles" / "a"
    home_b = tmp_path / "profiles" / "b"
    for home in (default_home, home_a, home_b):
        home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    runner, store = _make_runner({
        "agent:a:telegram:dm:1": home_a,
        "agent:b:telegram:dm:1": home_b,
    }, monkeypatch)

    ss.set_multiplex_active(True)
    # Simulate "A is the currently active scope" (an A -> B -> A multiplex probe): entering B's
    # release while A's scope is active on the calling context is exactly the reviewer's shape.
    with _profile_runtime_scope(home_a, hydrate_secrets=False):
        agent_b = _agent_with_failing_flush("session-for-b")
        runner._retire_replaced_agent("agent:b:telegram:dm:1", (agent_b,))

    b_files = list((home_b / "pending_messages").glob("*.json")) if (home_b / "pending_messages").exists() else []
    a_files = list((home_a / "pending_messages").glob("*.json")) if (home_a / "pending_messages").exists() else []
    default_files = (
        list((default_home / "pending_messages").glob("*.json"))
        if (default_home / "pending_messages").exists() else []
    )
    assert b_files, "snapshot must land under B's own home"
    assert not a_files, "must not land under the caller's active scope (A)"
    assert not default_files, "must not land under the default profile"
    payload = json.loads(b_files[0].read_text(encoding="utf-8"))
    assert payload["session_id"] == "session-for-b"


def test_release_evicted_agent_spools_under_the_sessions_own_profile(tmp_path: Path, monkeypatch):
    from gateway.run_turn_runner import TurnRunner

    default_home = tmp_path / "default"
    home_a = tmp_path / "profiles" / "a"
    home_b = tmp_path / "profiles" / "b"
    for home in (default_home, home_a, home_b):
        home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    runner, store = _make_runner({
        "agent:a:telegram:dm:1": home_a,
        "agent:b:telegram:dm:1": home_b,
    }, monkeypatch)

    ss.set_multiplex_active(True)
    ctx = SimpleNamespace(session_key="agent:b:telegram:dm:1")
    turn_runner = TurnRunner(runner, ctx)

    with _profile_runtime_scope(home_a, hydrate_secrets=False):
        agent_b = _agent_with_failing_flush("session-for-b")
        turn_runner._release_evicted_agent(agent_b)

    b_files = list((home_b / "pending_messages").glob("*.json")) if (home_b / "pending_messages").exists() else []
    a_files = list((home_a / "pending_messages").glob("*.json")) if (home_a / "pending_messages").exists() else []
    assert b_files, "snapshot must land under B's own home"
    assert not a_files, "must not land under the caller's active scope (A)"
