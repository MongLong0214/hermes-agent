from __future__ import annotations

import json
import logging
import types
from unittest.mock import MagicMock

import pytest

from tools import delegate_tool


ROUTED_CREDS = {
    "model": "gpt-5.6-terra",
    "provider": "openai-codex",
    "base_url": "https://chatgpt.com/backend-api/codex",
    "api_key": "token",
    "api_mode": "codex_responses",
    "request_overrides": {},
    "max_output_tokens": None,
    "command": None,
    "args": [],
}


def test_delegate_roles_map_to_dynamic_router_roles():
    assert delegate_tool._routing_role_for_task({}, "leaf") == "subagent"
    assert delegate_tool._routing_role_for_task({}, "orchestrator") == "subagent"
    assert (
        delegate_tool._routing_role_for_task(
            {"routing_role": "subagent_simple"}, "leaf"
        )
        == "subagent_simple"
    )
    assert (
        delegate_tool._routing_role_for_task({"routing_role": "boomer"}, "leaf")
        == "boomer"
    )


def test_pick_spawn_record_applies_router_model_and_provider(monkeypatch):
    calls = []
    built = MagicMock(name="child")
    resolved_cfg = {}

    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: True)
    monkeypatch.setattr(
        delegate_tool,
        "_run_local_router",
        lambda *args: calls.append(args)
        or (
            {"model": "gpt-5.6-terra", "provider": "codex"}
            if args[0] == "pick"
            else {"recorded": True}
        ),
    )

    def fake_resolve(cfg, parent):
        resolved_cfg.update(cfg)
        return ROUTED_CREDS

    monkeypatch.setattr(
        delegate_tool, "_resolve_delegation_credentials", fake_resolve
    )

    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return built

    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools", fake_build)

    child = delegate_tool._build_child_with_local_routing(
        routing_role="subagent",
        delegation_cfg={},
        parent_agent=MagicMock(),
        build_kwargs={"goal": "implement"},
    )

    assert child is built
    assert captured["model"] == "gpt-5.6-terra"
    assert captured["override_provider"] == "openai-codex"
    assert resolved_cfg["model"] == "gpt-5.6-terra"
    assert resolved_cfg["provider"] == "openai-codex"
    assert calls == [("pick", "subagent"), ("record", "codex")]


def test_router_block_fails_closed_before_spawn_and_never_records(monkeypatch):
    calls = []
    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: True)

    def blocked(*args):
        calls.append(args)
        raise delegate_tool.LocalRouterError("blocked by stale probe")

    monkeypatch.setattr(delegate_tool, "_run_local_router", blocked)
    build = MagicMock()
    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools", build)

    with pytest.raises(delegate_tool.LocalRouterError, match="stale probe"):
        delegate_tool._build_child_with_local_routing(
            routing_role="boomer",
            delegation_cfg={},
            parent_agent=MagicMock(),
            build_kwargs={"goal": "verify"},
        )

    build.assert_not_called()
    assert calls == [("pick", "boomer")]


def test_spawn_failure_never_records_success(monkeypatch):
    calls = []
    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: True)
    monkeypatch.setattr(
        delegate_tool,
        "_run_local_router",
        lambda *args: calls.append(args)
        or {"model": "gpt-5.6-luna", "provider": "codex"},
    )
    monkeypatch.setattr(
        delegate_tool,
        "_resolve_delegation_credentials",
        lambda cfg, parent: ROUTED_CREDS,
    )
    monkeypatch.setattr(
        delegate_tool,
        "_build_child_preserving_parent_tools",
        MagicMock(side_effect=RuntimeError("spawn failed")),
    )

    with pytest.raises(RuntimeError, match="spawn failed"):
        delegate_tool._build_child_with_local_routing(
            routing_role="subagent_simple",
            delegation_cfg={},
            parent_agent=MagicMock(),
            build_kwargs={"goal": "mechanical change"},
        )

    assert calls == [("pick", "subagent_simple")]


def test_explicit_config_override_is_audited_and_bypasses_router(monkeypatch, caplog):
    calls = []
    built = MagicMock(name="child")
    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: True)
    monkeypatch.setattr(
        delegate_tool, "_run_local_router", lambda *args: calls.append(args)
    )
    monkeypatch.setattr(
        delegate_tool,
        "_resolve_delegation_credentials",
        lambda cfg, parent: {**ROUTED_CREDS, "model": "manual-model"},
    )
    monkeypatch.setattr(
        delegate_tool,
        "_build_child_preserving_parent_tools",
        lambda **kwargs: built,
    )

    with caplog.at_level(logging.WARNING):
        child = delegate_tool._build_child_with_local_routing(
            routing_role="subagent",
            delegation_cfg={"model": "manual-model", "provider": "openai-codex"},
            parent_agent=MagicMock(),
            build_kwargs={"goal": "manual exception"},
        )

    assert child is built
    assert calls == []
    assert "routing policy exception" in caplog.text.lower()
    assert "manual-model" in caplog.text


def test_delegate_task_real_spawn_entry_uses_routing_role(monkeypatch):
    calls = []
    child = types.SimpleNamespace(tool_progress_callback=None)
    parent = MagicMock(
        _delegate_depth=0,
        _active_children=[],
        session_id="parent-session",
        tool_progress_callback=None,
    )

    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"max_iterations": 2})
    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: True)
    monkeypatch.setattr(
        delegate_tool,
        "_run_local_router",
        lambda *args: calls.append(args)
        or (
            {"model": "gpt-5.6-sol", "provider": "codex"}
            if args[0] == "pick"
            else {"recorded": True}
        ),
    )
    monkeypatch.setattr(
        delegate_tool,
        "_resolve_delegation_credentials",
        lambda cfg, parent_agent: ROUTED_CREDS,
    )
    monkeypatch.setattr(
        delegate_tool,
        "_build_child_preserving_parent_tools",
        lambda **kwargs: child,
    )
    monkeypatch.setattr(
        delegate_tool,
        "_run_single_child",
        lambda *args, **kwargs: {
            "task_index": 0,
            "status": "completed",
            "summary": "ok",
            "duration_seconds": 0,
        },
    )
    monkeypatch.setattr(delegate_tool, "_finalize_child_results", lambda *args: None)
    monkeypatch.setattr(
        "tools.delegation_live_log.create_live_transcripts",
        lambda tasks, context: ("", [], []),
    )
    monkeypatch.setattr(
        "tools.delegation_live_log.update_manifest_statuses", lambda *args: None
    )

    result = delegate_tool.delegate_task(
        goal="independent adversarial review",
        routing_role="boomer",
        parent_agent=parent,
    )

    assert json.loads(result)["results"][0]["status"] == "completed"
    assert calls == [("pick", "boomer"), ("record", "codex")]


def test_aia_agent_dispatch_forwards_routing_role(monkeypatch):
    from run_agent import AIAgent

    captured = {}
    monkeypatch.setattr(
        delegate_tool,
        "delegate_task",
        lambda **kwargs: captured.update(kwargs) or "ok",
    )
    parent = types.SimpleNamespace(_delegate_depth=0)

    result = AIAgent._dispatch_delegate_task(
        parent,  # type: ignore[arg-type]
        {"goal": "review", "routing_role": "boomer"},
    )

    assert result == "ok"
    assert captured["routing_role"] == "boomer"


def test_router_nonzero_error_is_stable_and_hides_stdout_and_stderr(monkeypatch):
    stdout = '{"error": "private router payload", "path": "/private/stdout"}'
    stderr = "private stderr at /private/stderr"
    monkeypatch.setattr(
        delegate_tool.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(
            returncode=9, stdout=stdout, stderr=stderr
        ),
    )

    with pytest.raises(delegate_tool.LocalRouterError) as raised:
        delegate_tool._run_local_router("pick", "subagent")

    assert str(raised.value) == "local_router_command_failed: command=pick role=subagent"
    assert stdout not in str(raised.value)
    assert stderr not in str(raised.value)
    assert "/private/" not in str(raised.value)


def test_router_invalid_json_error_is_stable_and_hides_payload(monkeypatch):
    payload = "malformed router output /private/router.py Traceback"
    monkeypatch.setattr(
        delegate_tool.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(
            returncode=0, stdout=payload, stderr=""
        ),
    )

    with pytest.raises(delegate_tool.LocalRouterError) as raised:
        delegate_tool._run_local_router("pick", "subagent_simple")

    assert (
        str(raised.value)
        == "local_router_invalid_pick: command=pick role=subagent_simple"
    )
    assert payload not in str(raised.value)
    assert "Traceback" not in str(raised.value)


def test_router_unusable_pick_error_is_stable_and_hides_payload(monkeypatch):
    payload = json.dumps(
        {
            "blocked": True,
            "reason": "private router denial",
            "path": "/private/router.py",
        }
    )
    monkeypatch.setattr(
        delegate_tool.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(
            returncode=0, stdout=payload, stderr=""
        ),
    )

    with pytest.raises(delegate_tool.LocalRouterError) as raised:
        delegate_tool._run_local_router("pick", "boomer")

    assert str(raised.value) == "local_router_unusable_pick: command=pick role=boomer"
    assert payload not in str(raised.value)
    assert "private router denial" not in str(raised.value)
    assert "/private/" not in str(raised.value)


def test_router_execution_error_is_stable_and_hides_exception_text(monkeypatch):
    monkeypatch.setattr(
        delegate_tool.subprocess,
        "run",
        MagicMock(side_effect=OSError("private error from /private/router.py")),
    )

    with pytest.raises(delegate_tool.LocalRouterError) as raised:
        delegate_tool._run_local_router("pick", "subagent")

    assert str(raised.value) == "local_router_execution_failed: command=pick role=subagent"
    assert "/private/" not in str(raised.value)
    assert "private error" not in str(raised.value)


def test_record_failure_is_conspicuous_without_exception_payload_or_traceback(
    monkeypatch, caplog
):
    raw_record_failure = "private record stderr /private/router.py Traceback"
    built = MagicMock(name="child")
    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: True)
    monkeypatch.setattr(
        delegate_tool,
        "_run_local_router",
        lambda command, value: (
            {"model": "gpt-5.6-terra", "provider": "codex"}
            if command == "pick"
            else (_ for _ in ()).throw(delegate_tool.LocalRouterError(raw_record_failure))
        ),
    )
    monkeypatch.setattr(
        delegate_tool,
        "_resolve_delegation_credentials",
        lambda cfg, parent: ROUTED_CREDS,
    )
    monkeypatch.setattr(
        delegate_tool, "_build_child_preserving_parent_tools", lambda **kwargs: built
    )

    with caplog.at_level(logging.ERROR, logger="tools.delegate_tool"):
        child = delegate_tool._build_child_with_local_routing(
            routing_role="subagent",
            delegation_cfg={},
            parent_agent=MagicMock(),
            build_kwargs={"goal": "spawn"},
        )

    assert child is built
    assert "local_router_record_failed command=record provider=codex role=subagent" in caplog.text
    assert raw_record_failure not in caplog.text
    assert "/private/" not in caplog.text
    assert "Traceback" not in caplog.text
    assert not caplog.records[0].exc_info
