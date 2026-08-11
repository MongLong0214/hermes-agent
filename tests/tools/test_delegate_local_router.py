from __future__ import annotations

import json
import logging
import sys
import traceback
import types
from unittest.mock import MagicMock

import pytest

from tools import delegate_tool


SYNTHETIC_MODEL = "router-model-placeholder"
PROTOCOL_PROVIDER = "router-provider-placeholder"
RUNTIME_PROVIDER = "runtime-provider-placeholder"
ROUTER_ROLE = "standard"
MAX_ROUTER_STDOUT_BYTES = 4096

ROUTED_CREDS = {
    "model": SYNTHETIC_MODEL,
    "provider": RUNTIME_PROVIDER,
    "base_url": "https://router.invalid/v1",
    "api_key": "test-key-placeholder",
    "api_mode": "chat_completions",
    "request_overrides": {},
    "max_output_tokens": None,
    "command": None,
    "args": [],
}


def _captured_stdout(stdout: bytes, *, returncode: int = 0):
    return returncode, stdout


def _allow_synthetic_provider(monkeypatch):
    monkeypatch.setattr(
        delegate_tool,
        "_LOCAL_TO_RUNTIME_PROVIDER",
        {PROTOCOL_PROVIDER: RUNTIME_PROVIDER},
    )


def test_delegate_roles_map_to_public_neutral_router_roles():
    assert delegate_tool._routing_role_for_task({}, "leaf") == "standard"
    assert delegate_tool._routing_role_for_task({}, "orchestrator") == "standard"
    assert delegate_tool._routing_role_for_task({"routing_role": "fast"}, "leaf") == "fast"
    assert (
        delegate_tool._routing_role_for_task(
            {"routing_role": "deliberate"}, "leaf"
        )
        == "deliberate"
    )


def test_delegate_schema_exposes_only_public_neutral_router_roles():
    expected = ["fast", "standard", "deliberate"]
    properties = delegate_tool.DELEGATE_TASK_SCHEMA["parameters"]["properties"]

    assert properties["routing_role"]["enum"] == expected
    assert properties["tasks"]["items"]["properties"]["routing_role"]["enum"] == expected


def test_invalid_router_role_error_is_stable_and_hides_input():
    raw_role = "private-role /private/router.py Traceback"

    with pytest.raises(delegate_tool.LocalRouterError) as raised:
        delegate_tool._routing_role_for_task({"routing_role": raw_role}, "leaf")

    assert str(raised.value) == "local_router_invalid_role"
    assert raw_role not in str(raised.value)
    assert "/private/" not in str(raised.value)
    assert "Traceback" not in str(raised.value)


def test_pick_spawn_record_applies_allowlisted_provider_translation(monkeypatch):
    calls = []
    built = MagicMock(name="child")
    resolved_cfg = {}
    _allow_synthetic_provider(monkeypatch)

    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: True)
    monkeypatch.setattr(
        delegate_tool,
        "_run_local_router",
        lambda *args: calls.append(args)
        or (
            {"model": SYNTHETIC_MODEL, "provider": PROTOCOL_PROVIDER}
            if args[0] == "pick"
            else {}
        ),
    )

    def fake_resolve(cfg, parent):
        resolved_cfg.update(cfg)
        return ROUTED_CREDS

    monkeypatch.setattr(delegate_tool, "_resolve_delegation_credentials", fake_resolve)

    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return built

    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools", fake_build)

    child = delegate_tool._build_child_with_local_routing(
        routing_role=ROUTER_ROLE,
        delegation_cfg={},
        parent_agent=MagicMock(),
        build_kwargs={"goal": "implement"},
    )

    assert child is built
    assert captured["model"] == SYNTHETIC_MODEL
    assert captured["override_provider"] == RUNTIME_PROVIDER
    assert resolved_cfg["model"] == SYNTHETIC_MODEL
    assert resolved_cfg["provider"] == RUNTIME_PROVIDER
    assert calls == [("pick", ROUTER_ROLE), ("record", PROTOCOL_PROVIDER)]


def test_router_block_fails_closed_before_spawn_and_never_records(monkeypatch):
    calls = []
    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: True)

    def blocked(*args):
        calls.append(args)
        raise delegate_tool.LocalRouterError("local_router_command_failed")

    monkeypatch.setattr(delegate_tool, "_run_local_router", blocked)
    build = MagicMock()
    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools", build)

    with pytest.raises(
        delegate_tool.LocalRouterError, match="^local_router_command_failed$"
    ):
        delegate_tool._build_child_with_local_routing(
            routing_role="deliberate",
            delegation_cfg={},
            parent_agent=MagicMock(),
            build_kwargs={"goal": "verify"},
        )

    build.assert_not_called()
    assert calls == [("pick", "deliberate")]


def test_router_child_construction_failure_is_sanitized_before_spawn(monkeypatch):
    calls = []
    raw_provider = "private-provider"
    raw_endpoint = "https://private-router.invalid/v1"
    raw_path = "/private/router/credentials.invalid"
    raw_credential = "credential=sensitive-credential-fixture"
    raw_failure = (
        f"child init failed for provider={raw_provider} endpoint={raw_endpoint} "
        f"credential={raw_credential} path={raw_path}"
    )
    _allow_synthetic_provider(monkeypatch)
    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: True)
    monkeypatch.setattr(
        delegate_tool,
        "_run_local_router",
        lambda *args: calls.append(args)
        or {"model": SYNTHETIC_MODEL, "provider": PROTOCOL_PROVIDER},
    )
    monkeypatch.setattr(
        delegate_tool,
        "_resolve_delegation_credentials",
        lambda cfg, parent: ROUTED_CREDS,
    )
    monkeypatch.setattr(
        delegate_tool,
        "_build_child_preserving_parent_tools",
        MagicMock(side_effect=RuntimeError(raw_failure)),
    )

    with pytest.raises(delegate_tool.LocalRouterError) as raised:
        delegate_tool._build_child_with_local_routing(
            routing_role="fast",
            delegation_cfg={},
            parent_agent=MagicMock(),
            build_kwargs={"goal": "mechanical change"},
        )

    exposed = str(raised.value) + "\n" + "".join(
        traceback.format_exception(raised.value)
    )
    assert str(raised.value) == "local_router_child_construction_failed"
    assert raw_failure not in exposed
    assert raw_provider not in exposed
    assert raw_endpoint not in exposed
    assert raw_path not in exposed
    assert raw_credential not in exposed
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    assert calls == [("pick", "fast")]


def test_child_construction_without_router_preserves_friendly_value_error(
    monkeypatch,
):
    friendly_failure = "No API key configured for delegated provider."
    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: False)
    monkeypatch.setattr(
        delegate_tool,
        "_build_child_preserving_parent_tools",
        MagicMock(side_effect=ValueError(friendly_failure)),
    )

    with pytest.raises(ValueError) as raised:
        delegate_tool._build_child_with_local_routing(
            routing_role=ROUTER_ROLE,
            delegation_cfg={},
            parent_agent=MagicMock(),
            default_credentials=ROUTED_CREDS,
            build_kwargs={"goal": "spawn normally"},
        )

    assert str(raised.value) == friendly_failure


def test_explicit_config_override_logs_only_generic_event(monkeypatch, caplog):
    built = MagicMock(name="child")
    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: True)
    monkeypatch.setattr(
        delegate_tool,
        "_resolve_delegation_credentials",
        lambda cfg, parent: ROUTED_CREDS,
    )
    monkeypatch.setattr(
        delegate_tool,
        "_build_child_preserving_parent_tools",
        lambda **kwargs: built,
    )

    with caplog.at_level(logging.WARNING, logger="tools.delegate_tool"):
        child = delegate_tool._build_child_with_local_routing(
            routing_role=ROUTER_ROLE,
            delegation_cfg={
                "model": "private-model",
                "provider": "private-provider",
            },
            parent_agent=MagicMock(),
            default_credentials=ROUTED_CREDS,
            build_kwargs={"goal": "manual exception"},
        )

    assert child is built
    records = [r for r in caplog.records if r.name == "tools.delegate_tool"]
    assert len(records) == 1
    assert records[0].getMessage() == "delegate_local_router_policy_override"
    assert not records[0].exc_info
    assert "private-model" not in caplog.text
    assert "private-provider" not in caplog.text
    assert ROUTER_ROLE not in caplog.text


def test_delegate_task_real_spawn_entry_uses_neutral_routing_role(monkeypatch):
    calls = []
    child = types.SimpleNamespace(tool_progress_callback=None)
    parent = MagicMock(
        _delegate_depth=0,
        _active_children=[],
        session_id="parent-session",
        tool_progress_callback=None,
    )
    _allow_synthetic_provider(monkeypatch)

    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"max_iterations": 2})
    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: True)
    monkeypatch.setattr(
        delegate_tool,
        "_run_local_router",
        lambda *args: calls.append(args)
        or (
            {"model": SYNTHETIC_MODEL, "provider": PROTOCOL_PROVIDER}
            if args[0] == "pick"
            else {}
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
        goal="independent verification",
        routing_role="deliberate",
        parent_agent=parent,
    )

    assert json.loads(result)["results"][0]["status"] == "completed"
    assert calls == [("pick", "deliberate"), ("record", PROTOCOL_PROVIDER)]


def test_delegate_task_child_construction_failure_is_sanitized_at_caller_boundary(
    monkeypatch, caplog
):
    calls = []
    raw_provider = "private-provider"
    raw_endpoint = "https://private-router.invalid/v1"
    raw_path = "/private/router/credentials.invalid"
    raw_credential = "credential=sensitive-credential-fixture"
    raw_failure = (
        f"Traceback: child init failed for provider={raw_provider} "
        f"endpoint={raw_endpoint} credential={raw_credential} path={raw_path}"
    )
    parent = MagicMock(
        _delegate_depth=0,
        _active_children=[],
        session_id="parent-session",
        tool_progress_callback=None,
    )
    _allow_synthetic_provider(monkeypatch)

    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"max_iterations": 2})
    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: True)
    monkeypatch.setattr(
        delegate_tool,
        "_run_local_router",
        lambda *args: calls.append(args)
        or {"model": SYNTHETIC_MODEL, "provider": PROTOCOL_PROVIDER},
    )
    monkeypatch.setattr(
        delegate_tool,
        "_resolve_delegation_credentials",
        lambda cfg, parent_agent: ROUTED_CREDS,
    )
    monkeypatch.setattr(
        delegate_tool,
        "_build_child_preserving_parent_tools",
        MagicMock(side_effect=RuntimeError(raw_failure)),
    )
    monkeypatch.setattr(
        "tools.delegation_live_log.create_live_transcripts",
        lambda tasks, context: ("", [], []),
    )
    monkeypatch.setattr(
        "tools.delegation_live_log.update_manifest_statuses", lambda *args: None
    )

    with caplog.at_level(logging.DEBUG):
        result = delegate_tool.delegate_task(
            goal="construct safely",
            routing_role="fast",
            parent_agent=parent,
        )

    assert json.loads(result) == {"error": "local_router_child_construction_failed"}
    assert calls == [("pick", "fast")]
    sensitive_values = (
        raw_failure,
        raw_provider,
        raw_endpoint,
        raw_path,
        raw_credential,
        "Traceback",
    )
    for sensitive_value in sensitive_values:
        assert sensitive_value not in result
        assert sensitive_value not in caplog.text


def test_aia_agent_dispatch_forwards_neutral_routing_role(monkeypatch):
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
        {"goal": "review", "routing_role": "deliberate"},
    )

    assert result == "ok"
    assert captured["routing_role"] == "deliberate"


def test_router_nonzero_error_is_stable_and_hides_stdout(monkeypatch):
    stdout = b'{"error": "private router payload", "path": "/private/stdout"}'
    monkeypatch.setattr(
        delegate_tool,
        "_capture_local_router_stdout",
        lambda *args, **kwargs: _captured_stdout(stdout, returncode=9),
    )

    with pytest.raises(delegate_tool.LocalRouterError) as raised:
        delegate_tool._run_local_router("pick", ROUTER_ROLE)

    assert str(raised.value) == "local_router_command_failed"
    assert stdout.decode() not in str(raised.value)
    assert "/private/" not in str(raised.value)


def test_router_invalid_json_error_is_stable_and_hides_payload(monkeypatch):
    payload = b"malformed router output /private/router.py Traceback"
    monkeypatch.setattr(
        delegate_tool,
        "_capture_local_router_stdout",
        lambda *args, **kwargs: _captured_stdout(payload),
    )

    with pytest.raises(delegate_tool.LocalRouterError) as raised:
        delegate_tool._run_local_router("pick", ROUTER_ROLE)

    assert str(raised.value) == "local_router_malformed_output"
    assert payload.decode() not in str(raised.value)
    assert "Traceback" not in str(raised.value)


def test_router_invalid_utf8_fails_closed(monkeypatch):
    monkeypatch.setattr(
        delegate_tool,
        "_capture_local_router_stdout",
        lambda *args, **kwargs: _captured_stdout(b"\xff\xfe"),
    )

    with pytest.raises(delegate_tool.LocalRouterError) as raised:
        delegate_tool._run_local_router("pick", ROUTER_ROLE)

    assert str(raised.value) == "local_router_malformed_output"


@pytest.mark.parametrize(
    "captured",
    [
        (False, b"{}"),
        (0, "{}"),
        (0, bytearray(b"{}")),
    ],
)
def test_router_capture_requires_exact_return_types(monkeypatch, captured):
    monkeypatch.setattr(
        delegate_tool,
        "_capture_local_router_stdout",
        lambda *args, **kwargs: captured,
    )

    with pytest.raises(
        delegate_tool.LocalRouterError,
        match="^local_router_malformed_output$",
    ):
        delegate_tool._run_local_router("pick", ROUTER_ROLE)


def test_router_oversize_stdout_fails_before_json_parse(monkeypatch):
    payload = b"{" + (b" " * MAX_ROUTER_STDOUT_BYTES) + b"}"
    parse = MagicMock(side_effect=AssertionError("JSON parser must not run"))
    monkeypatch.setattr(
        delegate_tool,
        "_capture_local_router_stdout",
        lambda *args, **kwargs: _captured_stdout(payload),
    )
    monkeypatch.setattr(delegate_tool.json, "loads", parse)

    with pytest.raises(delegate_tool.LocalRouterError) as raised:
        delegate_tool._run_local_router("pick", ROUTER_ROLE)

    assert str(raised.value) == "local_router_output_too_large"
    parse.assert_not_called()


def test_router_real_capture_reads_limit_plus_one_and_reaps_oversize_child(
    monkeypatch, tmp_path
):
    limit = 64
    router = tmp_path / "oversize_router.py"
    router.write_text(
        "import sys, threading\n"
        f"sys.stdout.buffer.write(b'x' * {limit + 2})\n"
        "sys.stdout.buffer.flush()\n"
        "threading.Event().wait()\n",
        encoding="utf-8",
    )
    real_popen = delegate_tool.subprocess.Popen
    real_read = delegate_tool.os.read
    processes = []
    tracked_fd = None
    requested_sizes = []
    returned_sizes = []

    def recording_popen(*args, **kwargs):
        nonlocal tracked_fd
        process = real_popen(*args, **kwargs)
        processes.append(process)
        assert process.stdout is not None
        tracked_fd = process.stdout.fileno()
        return process

    def recording_read(fd, size):
        data = real_read(fd, size)
        if fd == tracked_fd:
            requested_sizes.append(size)
            returned_sizes.append(len(data))
        return data

    monkeypatch.setattr(delegate_tool, "_local_router_path", lambda: router)
    monkeypatch.setattr(delegate_tool, "_LOCAL_ROUTER_MAX_STDOUT_BYTES", limit)
    monkeypatch.setattr(delegate_tool.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(delegate_tool.os, "read", recording_read)

    try:
        with pytest.raises(
            delegate_tool.LocalRouterError,
            match="^local_router_output_too_large$",
        ):
            delegate_tool._run_local_router("pick", ROUTER_ROLE)

        assert requested_sizes
        assert sum(returned_sizes) == limit + 1
        assert max(requested_sizes) <= limit + 1
        assert len(processes) == 1
        assert processes[0].returncode is not None
        assert processes[0].stdout is not None
        assert processes[0].stdout.closed
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_router_capture_reaps_child_on_timeout(monkeypatch, tmp_path):
    router = tmp_path / "waiting_router.py"
    router.write_text(
        "import threading\nthreading.Event().wait()\n",
        encoding="utf-8",
    )
    real_popen = delegate_tool.subprocess.Popen
    processes = []

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(delegate_tool.subprocess, "Popen", recording_popen)

    try:
        with pytest.raises(
            delegate_tool.LocalRouterError,
            match="^local_router_execution_failed$",
        ):
            delegate_tool._capture_local_router_stdout(
                [sys.executable, str(router)],
                max_stdout_bytes=64,
                timeout_seconds=0.05,
            )

        assert len(processes) == 1
        assert processes[0].returncode is not None
        assert processes[0].stdout.closed
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"model": SYNTHETIC_MODEL},
        {"provider": PROTOCOL_PROVIDER},
        {
            "model": SYNTHETIC_MODEL,
            "provider": PROTOCOL_PROVIDER,
            "unexpected": True,
        },
    ],
)
def test_router_pick_requires_exact_schema(monkeypatch, payload):
    _allow_synthetic_provider(monkeypatch)
    monkeypatch.setattr(
        delegate_tool,
        "_capture_local_router_stdout",
        lambda *args, **kwargs: _captured_stdout(json.dumps(payload).encode()),
    )

    with pytest.raises(delegate_tool.LocalRouterError) as raised:
        delegate_tool._run_local_router("pick", ROUTER_ROLE)

    assert str(raised.value) == "local_router_invalid_pick"


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        (1, PROTOCOL_PROVIDER),
        (True, PROTOCOL_PROVIDER),
        (None, PROTOCOL_PROVIDER),
        ("", PROTOCOL_PROVIDER),
        ("   ", PROTOCOL_PROVIDER),
        ("m" * 257, PROTOCOL_PROVIDER),
        (SYNTHETIC_MODEL, 1),
        (SYNTHETIC_MODEL, True),
        (SYNTHETIC_MODEL, None),
        (SYNTHETIC_MODEL, ""),
        (SYNTHETIC_MODEL, "   "),
        (SYNTHETIC_MODEL, "p" * 65),
    ],
)
def test_router_pick_requires_bounded_nonempty_strings(monkeypatch, model, provider):
    _allow_synthetic_provider(monkeypatch)
    payload = json.dumps({"model": model, "provider": provider}).encode()
    monkeypatch.setattr(
        delegate_tool,
        "_capture_local_router_stdout",
        lambda *args, **kwargs: _captured_stdout(payload),
    )

    with pytest.raises(delegate_tool.LocalRouterError) as raised:
        delegate_tool._run_local_router("pick", ROUTER_ROLE)

    assert str(raised.value) == "local_router_invalid_pick"


def test_router_pick_rejects_provider_outside_allowlist(monkeypatch):
    raw_provider = "private-provider /private/router.py Traceback"
    payload = json.dumps(
        {"model": SYNTHETIC_MODEL, "provider": raw_provider}
    ).encode()
    _allow_synthetic_provider(monkeypatch)
    monkeypatch.setattr(
        delegate_tool,
        "_capture_local_router_stdout",
        lambda *args, **kwargs: _captured_stdout(payload),
    )

    with pytest.raises(delegate_tool.LocalRouterError) as raised:
        delegate_tool._run_local_router("pick", ROUTER_ROLE)

    assert str(raised.value) == "local_router_invalid_pick"
    assert raw_provider not in str(raised.value)


def test_router_pick_returns_only_validated_protocol_fields(monkeypatch):
    payload = {"model": SYNTHETIC_MODEL, "provider": PROTOCOL_PROVIDER}
    _allow_synthetic_provider(monkeypatch)
    monkeypatch.setattr(
        delegate_tool,
        "_capture_local_router_stdout",
        lambda *args, **kwargs: _captured_stdout(json.dumps(payload).encode()),
    )

    assert delegate_tool._run_local_router("pick", ROUTER_ROLE) == payload


def test_router_credential_resolution_failure_is_sanitized(monkeypatch):
    raw_failure = "private provider error /private/credentials Traceback"
    _allow_synthetic_provider(monkeypatch)
    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: True)
    monkeypatch.setattr(
        delegate_tool,
        "_run_local_router",
        lambda *args: {"model": SYNTHETIC_MODEL, "provider": PROTOCOL_PROVIDER},
    )
    monkeypatch.setattr(
        delegate_tool,
        "_resolve_delegation_credentials",
        MagicMock(side_effect=RuntimeError(raw_failure)),
    )
    build = MagicMock()
    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools", build)

    with pytest.raises(delegate_tool.LocalRouterError) as raised:
        delegate_tool._build_child_with_local_routing(
            routing_role=ROUTER_ROLE,
            delegation_cfg={},
            parent_agent=MagicMock(),
            build_kwargs={"goal": "spawn"},
        )

    assert str(raised.value) == "local_router_credential_resolution_failed"
    assert raw_failure not in str(raised.value)
    assert "/private/" not in str(raised.value)
    assert "Traceback" not in str(raised.value)
    build.assert_not_called()


def test_delegate_task_initial_router_resolver_runtime_failure_is_sanitized(
    monkeypatch,
):
    raw_failure = "private resolver error /private/credentials.py Traceback"
    parent = MagicMock(_delegate_depth=0)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"max_iterations": 2})
    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: True)
    monkeypatch.setattr(
        delegate_tool,
        "_resolve_delegation_credentials",
        MagicMock(side_effect=RuntimeError(raw_failure)),
    )

    result = json.loads(
        delegate_tool.delegate_task(goal="resolve safely", parent_agent=parent)
    )

    assert result == {"error": "local_router_credential_resolution_failed"}
    assert raw_failure not in json.dumps(result)
    assert "/private/" not in json.dumps(result)
    assert "Traceback" not in json.dumps(result)


def test_delegate_task_without_router_preserves_user_friendly_resolver_error(
    monkeypatch,
):
    friendly_failure = "No API key configured for delegated provider."
    parent = MagicMock(_delegate_depth=0)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"max_iterations": 2})
    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: False)
    monkeypatch.setattr(
        delegate_tool,
        "_resolve_delegation_credentials",
        MagicMock(side_effect=ValueError(friendly_failure)),
    )

    result = json.loads(
        delegate_tool.delegate_task(goal="resolve normally", parent_agent=parent)
    )

    assert result == {"error": friendly_failure}


@pytest.mark.parametrize("credentials", [None, {}, {"model": SYNTHETIC_MODEL}])
def test_router_malformed_resolver_result_is_sanitized(
    monkeypatch, credentials
):
    _allow_synthetic_provider(monkeypatch)
    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: True)
    monkeypatch.setattr(
        delegate_tool,
        "_run_local_router",
        lambda *args: {"model": SYNTHETIC_MODEL, "provider": PROTOCOL_PROVIDER},
    )
    monkeypatch.setattr(
        delegate_tool, "_resolve_delegation_credentials", lambda *args: credentials
    )
    build = MagicMock()
    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools", build)

    with pytest.raises(delegate_tool.LocalRouterError) as raised:
        delegate_tool._build_child_with_local_routing(
            routing_role=ROUTER_ROLE,
            delegation_cfg={},
            parent_agent=MagicMock(),
            build_kwargs={"goal": "spawn"},
        )

    assert str(raised.value) == "local_router_credential_resolution_failed"
    build.assert_not_called()


@pytest.mark.parametrize(
    "credentials",
    [None, [], {}, {"model": SYNTHETIC_MODEL}],
)
def test_explicit_router_override_malformed_credential_bundle_is_sanitized(
    monkeypatch, credentials
):
    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: True)
    monkeypatch.setattr(
        delegate_tool,
        "_resolve_delegation_credentials",
        lambda *args: credentials,
    )
    build = MagicMock()
    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools", build)

    with pytest.raises(
        delegate_tool.LocalRouterError,
        match="^local_router_credential_resolution_failed$",
    ):
        delegate_tool._build_child_with_local_routing(
            routing_role=ROUTER_ROLE,
            delegation_cfg={
                "model": "private-model",
                "provider": "private-provider",
            },
            parent_agent=MagicMock(),
            build_kwargs={"goal": "spawn"},
        )

    build.assert_not_called()


def test_custom_child_credential_pool_failure_log_is_metadata_free(
    monkeypatch, caplog
):
    from agent import credential_pool

    raw_endpoint = "https://private.invalid/credentials/Traceback"
    raw_failure = "private custom pool error /private/credential_pool.py Traceback"
    monkeypatch.setattr(
        credential_pool,
        "get_custom_provider_pool_key",
        MagicMock(side_effect=RuntimeError(raw_failure)),
    )
    parent = MagicMock(provider="custom", _credential_pool=None)

    with caplog.at_level(logging.DEBUG, logger="tools.delegate_tool"):
        result = delegate_tool._resolve_child_credential_pool(
            "custom", parent, raw_endpoint
        )

    assert result is None
    records = [r for r in caplog.records if r.name == "tools.delegate_tool"]
    assert len(records) == 1
    assert records[0].getMessage() == "delegate_child_credential_pool_resolution_failed"
    assert not records[0].exc_info
    assert raw_endpoint not in caplog.text
    assert raw_failure not in caplog.text
    assert "/private/" not in caplog.text
    assert "Traceback" not in caplog.text


def test_provider_child_credential_pool_failure_log_is_metadata_free(
    monkeypatch, caplog
):
    from agent import credential_pool

    raw_provider = "private-provider /private/provider.py Traceback"
    raw_failure = "private provider pool error /private/credential_pool.py Traceback"
    monkeypatch.setattr(
        credential_pool,
        "load_pool",
        MagicMock(side_effect=RuntimeError(raw_failure)),
    )
    parent = MagicMock(provider="parent-provider", _credential_pool=None)

    with caplog.at_level(logging.DEBUG, logger="tools.delegate_tool"):
        result = delegate_tool._resolve_child_credential_pool(raw_provider, parent)

    assert result is None
    records = [r for r in caplog.records if r.name == "tools.delegate_tool"]
    assert len(records) == 1
    assert records[0].getMessage() == "delegate_child_credential_pool_resolution_failed"
    assert not records[0].exc_info
    assert raw_provider not in caplog.text
    assert raw_failure not in caplog.text
    assert "/private/" not in caplog.text
    assert "Traceback" not in caplog.text


def test_record_failure_logs_only_generic_stable_event(monkeypatch, caplog):
    raw_record_failure = "private record stderr /private/router.py Traceback"
    built = MagicMock(name="child")
    _allow_synthetic_provider(monkeypatch)
    monkeypatch.setattr(delegate_tool, "_local_router_enabled", lambda: True)
    monkeypatch.setattr(
        delegate_tool,
        "_run_local_router",
        lambda command, value: (
            {"model": SYNTHETIC_MODEL, "provider": PROTOCOL_PROVIDER}
            if command == "pick"
            else (_ for _ in ()).throw(
                delegate_tool.LocalRouterError(raw_record_failure)
            )
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
            routing_role=ROUTER_ROLE,
            delegation_cfg={},
            parent_agent=MagicMock(),
            build_kwargs={"goal": "spawn"},
        )

    assert child is built
    records = [r for r in caplog.records if r.name == "tools.delegate_tool"]
    assert len(records) == 1
    assert records[0].getMessage() == "delegate_local_router_record_failed"
    assert not records[0].exc_info
    assert raw_record_failure not in caplog.text
    assert SYNTHETIC_MODEL not in caplog.text
    assert PROTOCOL_PROVIDER not in caplog.text
    assert ROUTER_ROLE not in caplog.text
    assert "/private/" not in caplog.text
    assert "Traceback" not in caplog.text
