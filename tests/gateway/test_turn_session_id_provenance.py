"""A gateway turn's MCP calls carry the session they run in, whether or not the agent was cached.

``_set_session_env`` binds every session var but the id (it is ``""`` there), and only
``agent_init`` publishes an agent's id, so a turn served by a REUSED cached agent sent
``session_id: ""`` in MCP call provenance and ACP's owner guard refused the CEO's own
``run_create`` as "not the bound live session".
"""

from __future__ import annotations

from types import SimpleNamespace


def test_a_reused_agent_turn_reports_its_own_session_in_mcp_provenance(monkeypatch):
    from gateway.config import Platform
    from gateway.run_turn_runner import TurnRunner
    from gateway.session import SessionSource
    from gateway.session_context import clear_session_vars, get_session_env, set_session_vars
    from gateway.turn_context import TurnContext
    from tools.mcp_call_provenance import build_call_provenance

    monkeypatch.setattr(TurnRunner, "_native_image_run_message", lambda self: "hi")
    seen = []

    def run_conversation(message, **kwargs):
        seen.append(build_call_provenance()["session_id"])
        return {"final_response": "ok"}

    cached_agent = SimpleNamespace(session_id="20261001_092716_99cbd6", run_conversation=run_conversation)
    ctx = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1"),
        session_key="agent:main:telegram:dm:c1", user_config={}, message="hi",
    )
    # What _set_session_env binds for every turn: the key, never the id.
    tokens = set_session_vars(platform="telegram", chat_id="c1", session_key=ctx.session_key, cron_session="")
    try:
        TurnRunner(SimpleNamespace(), ctx)._run_conversation_with_approval(cached_agent, [], None, None, None)
        after = get_session_env("HERMES_SESSION_ID", "")
    finally:
        clear_session_vars(tokens)

    assert seen == ["20261001_092716_99cbd6"]
    assert after == ""  # bound for the turn only; the handler's context is left as it was


class _ContendedDb:
    """A session another process held (and compressed) while this turn waited for its lease."""

    def get_session(self, session_id):
        return {"id": session_id}

    def acquire_session_turn_lease(self, session_id, holder, *, on_wait, **kwargs):
        on_wait(0.5)
        return True

    def resolve_resume_session_id(self, session_id):
        return "compressed-tip"

    def get_messages_as_conversation(self, session_id, **kwargs):
        return []

    def release_session_turn_lease(self, session_id, holder):
        pass


def test_a_turn_that_waited_onto_a_compressed_tip_reports_the_tip(monkeypatch):
    """The holder compressed the session while this turn waited for its lease; admission adopts the
    continuation tip, and the turn's MCP calls must name that tip, not the id it was bound with."""
    from agent.turn_facade_lease import admit_durable_turn_lease
    from gateway.session_context import scoped_current_session_id
    from tools.mcp_call_provenance import build_call_provenance

    import threading

    monkeypatch.setenv("HERMES_SESSION_ID", "")  # the legacy rebind also writes os.environ
    monkeypatch.setattr("agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0))
    agent = SimpleNamespace(
        _session_db=_ContendedDb(), session_id="live-agent", _persist_disabled=False,
        _interrupt_requested=False, _interrupt_message=None, _execution_thread_id=None,
        _session_turn_lease_refresh_interval=60.0, _emit_status=lambda *a: None,
        _emit_warning=lambda *a: None, _touch_activity=lambda *a, **k: None,
        _liveness_activity_lock=lambda: threading.Lock())
    task_context = {"session_id": "live-agent", "task_id": "t", "platform": "telegram"}
    with scoped_current_session_id("live-agent"):
        admission = admit_durable_turn_lease(
            agent, session_id="live-agent", relay_turn_id="live-agent:t:1",
            task_context=task_context, conversation_history=[])
        try:
            reported = build_call_provenance()["session_id"]
        finally:
            admission.lease.release()

    assert agent.session_id == "compressed-tip"
    assert reported == "compressed-tip"
