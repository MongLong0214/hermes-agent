"""Hygiene adopt step: an uncommitted attempt keeps the transcript, a committed in-place one is adopted (#71097).

Both tests drive the production adopt path (``_hmwa_hygiene_adopt_transcript``), the only
reader of the compressor's ``_last_compaction_in_place`` flag on the hygiene route.
"""

import logging
from types import SimpleNamespace

import pytest

from gateway.run_turn import GatewayTurnMixin

_HISTORY = [{"role": "user", "content": "x" * 400}, {"role": "assistant", "content": "y" * 400}] * 4
_COMPRESSED = [{"role": "user", "content": "summary"}, {"role": "assistant", "content": "ok"}]


def _agent(**signals):
    base = {
        "session_id": "sid-1",
        "_last_compaction_in_place": False,
        "_last_compression_attempt_recorded": True,
        "_last_compression_attempt_in_place": None,
        "_compression_skipped_due_to_lock": None,
        "_compression_blocked_transient": None,
        "_last_compression_timed_out": False,
        "_last_compression_summary_warning": None,
        "_session_db": object(),
    }
    base.update(signals)
    return SimpleNamespace(**base)


async def _adopt(agent, caplog):
    """Run the gateway adopt step for an unrotated hygiene attempt; returns (attempt, entry, result)."""
    runner = GatewayTurnMixin.__new__(GatewayTurnMixin)
    attempt = SimpleNamespace(agent=agent, history=_HISTORY)
    plan = SimpleNamespace(msg_count=len(_HISTORY), approx_tokens=4_000, warn_token_threshold=10**9)
    entry = SimpleNamespace(session_id="sid-1", last_prompt_tokens=4_000)
    with caplog.at_level(logging.WARNING, logger="gateway.run_turn"):
        result = await runner._hmwa_hygiene_adopt_transcript(
            attempt, _COMPRESSED, _HISTORY, plan, session_entry=entry, source=None, _quick_key=None, run_generation=0,
        )
    return attempt, entry, result


@pytest.mark.asyncio
async def test_aborted_attempt_on_db_backed_agent_keeps_the_transcript(caplog):
    # codex_app_server thread interrupted / summary aborted: the compressor never reached the
    # commit boundary, so `_last_compression_attempt_in_place` stays None while `_session_db` is real.
    attempt, entry, (rotated, in_place, count, tokens) = await _adopt(_agent(), caplog)
    assert (rotated, in_place, count, tokens) == (False, False, len(_HISTORY), 4_000)
    assert attempt.history is _HISTORY and entry.last_prompt_tokens == 4_000


@pytest.mark.asyncio
async def test_committed_in_place_compaction_is_adopted_not_warned_about(caplog):
    # #71097 reporter 1: split_status=in_place_committed. compress_context sets
    # `_last_compaction_in_place` from the same `compacted_in_place` variable that produces that
    # telemetry, so a committed in-place attempt reaches the adopt step flagged and is adopted.
    agent = _agent(_last_compaction_in_place=True, _last_compression_attempt_in_place=True)
    attempt, entry, (rotated, in_place, count, tokens) = await _adopt(agent, caplog)
    assert (rotated, in_place, count) == (False, True, len(_COMPRESSED))
    assert attempt.history is _COMPRESSED and entry.last_prompt_tokens == 0
    assert "did not rotate or compact in place" not in caplog.text


@pytest.mark.asyncio
async def test_rewrite_failure_during_rotation_does_not_rebind_the_live_entry(tmp_path, monkeypatch, caplog):
    """L4-1/L4-2: a rotated compression whose child-transcript write fails against the REAL
    canonical store (RecoverableHandleCache backoff, not a deliberately DB-less store) must not
    repoint the live session onto that unpersisted child. Write-before-repoint only holds if
    ``rewrite_transcript`` honestly reports failure instead of the stale "no DB == True" contract
    (#L4-1); this exercises the real ``SessionStore``/``AsyncSessionStore`` so a regression of that
    contract shows up here as an unexpected rebind/lease call, not as a mocked return value."""
    import hermes_state_registry
    from gateway.config import GatewayConfig
    from gateway.session import AsyncSessionStore, SessionStore

    store = SessionStore(sessions_dir=tmp_path / "gw", config=GatewayConfig())
    db = store._db
    assert db is not None
    db.create_session("sid-1", "telegram", session_key="telegram:1")
    db.append_message("sid-1", "user", "do not forget this conversation")

    # Break the canonical store the way an operational blip would: evict the cached handle and
    # make the next reopen fail (RecoverableHandleCache backoff) -- not a deliberately DB-less store.
    path = next(iter(store._db_handle_cache.handles))
    store._db_handle_cache.handles.pop(path)

    def _boom(*_a, **_kw):
        raise OSError("database is locked")

    monkeypatch.setattr(hermes_state_registry, "acquire", _boom)

    runner = GatewayTurnMixin.__new__(GatewayTurnMixin)
    runner.async_session_store = AsyncSessionStore(store)
    # Deliberately NOT stubbed: if the broken rewrite were wrongly reported as success, the code
    # would reach these calls and this test would fail with AttributeError (proving the rebind ran).

    agent = _agent(session_id="sid-2")  # rotated: compression minted a new child id
    attempt = SimpleNamespace(agent=agent, history=_HISTORY)
    plan = SimpleNamespace(msg_count=len(_HISTORY), approx_tokens=4_000, warn_token_threshold=10**9)
    entry = SimpleNamespace(session_id="sid-1", last_prompt_tokens=4_000)

    with caplog.at_level(logging.ERROR, logger="gateway.run_turn"):
        rotated, in_place, count, tokens = await runner._hmwa_hygiene_adopt_transcript(
            attempt, _COMPRESSED, _HISTORY, plan, session_entry=entry, source=None,
            _quick_key=None, run_generation=0,
        )

    assert rotated is False
    assert entry.session_id == "sid-1"  # NOT repointed at the unpersisted child
    assert attempt.history is _HISTORY
    assert "failed to persist compressed transcript" in caplog.text
