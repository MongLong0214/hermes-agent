"""L4-2/R67-1 — the empty-``final_response`` early return in ``TurnRunner.run_sync()`` must not
discard a real ``agent_persisted`` signal the turn loop already computed.

``run_turn.py``'s ``_hmwa_persist_turn_transcript`` trusts ``agent_result.get("agent_persisted",
self._session_db is not None)`` to decide ``skip_db``. Before this fix, the empty-response branch
of ``run_sync()`` built its return dict from ``common`` alone and never looked at
``result.get("agent_persisted")`` — so an explicit ``False`` (a real flush failure reported by
``finalize_turn``/``continue_codex_incomplete``/the codex runtime) was silently dropped, and the
gateway defaulted to "persisted" purely because a session DB exists, skipping its own write and
losing the turn.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _run_sync_with_agent_result(monkeypatch, agent_result):
    import gateway.run as gateway_run
    from gateway.config import Platform
    from gateway.run_turn_runner import TurnRunner
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext

    monkeypatch.setattr(gateway_run, "_current_max_iterations", lambda: 30)
    monkeypatch.setattr(TurnRunner, "_combined_ephemeral_prompt", lambda self: "")
    monkeypatch.setattr(TurnRunner, "_setup_stream_consumer", lambda self, platform_key: (None, None, None, False))
    monkeypatch.setattr(TurnRunner, "_resolve_turn_agent", lambda self, *a, **k: (SimpleNamespace(), False))
    monkeypatch.setattr(TurnRunner, "_wire_turn_agent_callbacks", lambda self, *a, **k: None)
    monkeypatch.setattr(TurnRunner, "_load_turn_history", lambda self, *a, **k: ([], None, []))
    monkeypatch.setattr(TurnRunner, "_prepare_turn_message", lambda self, *a, **k: (None, None))
    monkeypatch.setattr(TurnRunner, "_run_conversation_with_approval", lambda self, *a, **k: agent_result)
    monkeypatch.setattr(TurnRunner, "_finish_stream_consumer", lambda self, *a, **k: None)
    monkeypatch.setattr(TurnRunner, "_sync_session_after_run", lambda self, *a, **k: (False, "sess-1", 0))
    monkeypatch.setattr(TurnRunner, "_append_auto_media_tags", lambda self, final_response, *a, **k: final_response)

    runner = SimpleNamespace(
        _resolve_session_agent_runtime=lambda **k: ("model-x", {"provider": "openrouter"}),
        _provider_routing=None,
        _resolve_session_reasoning_config=lambda **k: None,
        _resolve_session_service_tier=lambda **k: None,
        _resolve_turn_agent_config=lambda *a, **k: None,
    )
    ctx = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1"),
        session_key="telegram:c1", user_config={}, message="hi",
    )
    return TurnRunner(runner, ctx).run_sync()


def test_empty_response_forwards_explicit_agent_persisted_false(monkeypatch):
    """A real flush failure (finalize_turn / continue_codex_incomplete / codex runtime, all now
    reporting an honest ``agent_persisted``) must survive the empty-final_response early return
    instead of being dropped so the gateway's own default (session DB exists -> True) takes over."""
    agent_result = {
        "final_response": "", "messages": [], "api_calls": 1, "failed": False,
        "agent_persisted": False,
    }
    result = _run_sync_with_agent_result(monkeypatch, agent_result)
    assert result["agent_persisted"] is False


def test_empty_response_forwards_explicit_agent_persisted_true(monkeypatch):
    """Guardrail: an explicit True (ordinary self-persisting runtime) must still come through."""
    agent_result = {
        "final_response": "", "messages": [], "api_calls": 1, "failed": False,
        "agent_persisted": True,
    }
    result = _run_sync_with_agent_result(monkeypatch, agent_result)
    assert result["agent_persisted"] is True


def test_empty_response_omits_the_key_when_the_turn_never_set_it(monkeypatch):
    """Guardrail: when nothing ever computed ``agent_persisted`` (no key in the turn result at
    all), the gateway's own session-DB-existence default must still apply — the key must stay
    absent rather than be synthesized here."""
    agent_result = {"final_response": "", "messages": [], "api_calls": 1, "failed": False}
    result = _run_sync_with_agent_result(monkeypatch, agent_result)
    assert "agent_persisted" not in result
