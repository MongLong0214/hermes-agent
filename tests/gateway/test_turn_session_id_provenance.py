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
