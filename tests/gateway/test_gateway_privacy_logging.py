"""Privacy regressions for gateway operational logging."""

import ast
import hashlib
import inspect
import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway import run as gateway_run


def test_inbound_logs_never_expose_bodies_or_correlatable_identifiers(caplog):
    source = SimpleNamespace(
        platform=SimpleNamespace(value="telegram"),
        user_name="private_username",
        user_id="sensitive-user-id-fixture",
        chat_id="sensitive-chat-id-fixture",
    )
    event = SimpleNamespace(
        text="private message body with medical details",
        reply_to_message_id="private-reply-id-42",
        reply_to_text="private quoted body",
    )

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        gateway_run._log_inbound_message(event, source)

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "platform=telegram" in text
    assert "chars=41" in text
    for private in (
        "private_username",
        "sensitive-user-id-fixture",
        "sensitive-chat-id-fixture",
        "private message body",
        "private quoted body",
        "private-reply-id-42",
        hashlib.sha256(b"private_username").hexdigest()[:12],
        hashlib.sha256(b"sensitive-chat-id-fixture").hexdigest()[:12],
    ):
        assert private not in text
    assert "preview=" not in text
    assert "user_hash=" not in text
    assert "chat_hash=" not in text


def test_transcript_lag_uses_only_non_identifying_counts_and_classification(caplog):
    session_key = "agent:main:telegram:dm:private-chat-id"
    session_digest = hashlib.sha256(session_key.encode()).hexdigest()[:12]

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        gateway_run._log_transcript_lag(disk_count=10, memory_count=11)

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "transcript_lag disk=10 memory=11 classification=unverified"
    ]
    text = "\n".join(messages)
    assert session_key not in text
    assert session_digest not in text
    assert "session" not in text
    assert "hash" not in text
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_response_ready_log_omits_raw_and_hashed_chat_identifier(caplog):
    chat_id = "sensitive-response-chat-id-fixture"
    source = SimpleNamespace(platform=SimpleNamespace(value="telegram"), chat_id=chat_id)

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        gateway_run._log_response_ready(
            source,
            response_time=0.2,
            api_calls=1,
            response_length=7,
        )

    assert "response ready: platform=telegram" in caplog.text
    assert chat_id not in caplog.text
    assert hashlib.sha256(chat_id.encode()).hexdigest()[:12] not in caplog.text
    assert "chat=" not in caplog.text


def test_native_attachment_boundary_logs_only_skip_count_for_credential_paths(
    tmp_path, monkeypatch, caplog
):
    from agent import file_safety
    from agent.image_routing import build_native_content_parts

    private_home = (
        tmp_path
        / "private-temp-root"
        / "private-user"
        / "uid-918273"
        / ".hermes"
    )
    keys = private_home / "bridge" / "keys"
    keys.mkdir(parents=True)
    monkeypatch.setattr(file_safety, "_hermes_home_path", lambda: private_home)
    monkeypatch.setattr(file_safety, "_hermes_root_path", lambda: private_home)

    direct = keys / "direct-signing-key.priv"
    direct.write_bytes(b"synthetic-direct-key")

    outside_file = tmp_path / "outside-file-material.png"
    outside_file.write_bytes(b"synthetic-file-target")
    file_link = keys / "file-link-key.priv"

    outside_dir = tmp_path / "outside-directory-material"
    outside_dir.mkdir()
    (outside_dir / "directory-link-key.priv").write_bytes(b"synthetic-dir-target")
    directory_link = keys / "linked-directory"
    try:
        file_link.symlink_to(outside_file)
        directory_link.symlink_to(outside_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.fail("symlink setup failed")

    requested = [
        direct,
        file_link,
        directory_link / "directory-link-key.priv",
    ]
    raw_paths = [str(path) for path in requested]

    parts, skipped = build_native_content_parts("inspect these", raw_paths)
    assert skipped == raw_paths
    assert parts == [{"type": "text", "text": "inspect these"}]

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        run_message = gateway_run._build_native_run_message(
            "inspect these", raw_paths
        )

    assert run_message == "inspect these"
    gateway_records = [
        record for record in caplog.records if record.name == "gateway.run"
    ]
    assert [
        (record.levelno, record.getMessage(), record.args, record.exc_info)
        for record in gateway_records
    ] == [
        (logging.WARNING, "native_image_attachment_skipped count=3", (3,), None)
    ]
    public_logs = "\n".join(record.getMessage() for record in caplog.records)
    for private in (
        *raw_paths,
        *(path.name for path in requested),
        str(tmp_path),
        "private-temp-root",
        "private-user",
        "918273",
        *(hashlib.sha256(path.encode()).hexdigest()[:12] for path in raw_paths),
    ):
        assert private not in public_logs


def test_native_attachment_boundary_preserves_success_and_turn_runner_uses_it(
    tmp_path, caplog
):
    image_path = tmp_path / "private-success-image.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-image")

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        run_message = gateway_run._build_native_run_message(
            "inspect this", [str(image_path)]
        )

    assert isinstance(run_message, list)
    assert run_message[0] == {
        "type": "text",
        "text": f"inspect this\n\n[Image attached at: {image_path}]",
    }
    assert run_message[1]["type"] == "image_url"
    assert run_message[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert not [
        record
        for record in caplog.records
        if record.name == "gateway.run" and record.levelno >= logging.WARNING
    ]

    caller_source = inspect.getsource(gateway_run.TurnRunner.run_sync)
    assert (
        "_run_message = _build_native_run_message(ctx.message, _native_imgs)"
        in caller_source
    )


def test_native_attachment_boundary_exception_log_omits_exception_text(caplog):
    private_exception = (
        "native image failed at "
        "/Users/private-user/private-temp-root/credential-image.png"
    )

    with patch(
        "agent.image_routing.build_native_content_parts",
        side_effect=RuntimeError(private_exception),
    ):
        with caplog.at_level(logging.WARNING, logger="gateway.run"):
            run_message = gateway_run._build_native_run_message(
                "preserve this text", ["/Users/private-user/credential-image.png"]
            )

    assert run_message == "preserve this text"
    records = [record for record in caplog.records if record.name == "gateway.run"]
    assert [
        (record.levelno, record.getMessage(), record.args, record.exc_info)
        for record in records
    ] == [(logging.WARNING, "native_image_attachment_failed", (), None)]
    assert private_exception not in caplog.text
    assert "private-user" not in caplog.text
    assert "private-temp-root" not in caplog.text
    assert "credential-image.png" not in caplog.text


_PRIVATE_IMAGE_PATH = Path(
    "/Users/private-user/private-temp-root/uid-918273/credential-image.png"
)
_PRIVATE_IMAGE_ERROR = f"native decoder failed for {_PRIVATE_IMAGE_PATH}"
_PRIVATE_IMAGE_DIGEST = hashlib.sha256(str(_PRIVATE_IMAGE_PATH).encode()).hexdigest()[:12]
_PRIVATE_VISION_PROMPT = "private prompt with credential marker and medical details"
_PRIVATE_VISION_MODEL = "private-provider/private-vision-model"


class _StopAfterAttachmentFallback(Exception):
    """Bound a large caller immediately after its real fallback branch."""


def _assert_attachment_metadata_absent(text: str) -> None:
    for private in (
        str(_PRIVATE_IMAGE_PATH),
        _PRIVATE_IMAGE_PATH.name,
        "private-user",
        "private-temp-root",
        "918273",
        "native decoder failed",
        _PRIVATE_IMAGE_DIGEST,
        "Traceback",
    ):
        assert private not in text


def _gateway_record_tuples(caplog):
    return [
        (record.levelno, record.getMessage(), record.args, record.exc_info)
        for record in caplog.records
        if record.name == "gateway.run"
    ]


def _all_record_tuples(records):
    return [
        (
            record.name,
            record.levelno,
            record.getMessage(),
            record.args,
            record.exc_info,
        )
        for record in records
    ]


def _render_operational_records(records) -> str:
    formatter = logging.Formatter("%(name)s %(levelname)s %(message)s")
    return "\n".join(
        f"{formatter.format(record)} args={record.args!r} exc_info={record.exc_info!r}"
        for record in records
    )


def _assert_vision_telemetry_metadata_absent(text: str) -> None:
    for private in (
        str(_PRIVATE_IMAGE_PATH),
        _PRIVATE_IMAGE_PATH.name,
        "private-user",
        "private-temp-root",
        "918273",
        "native decoder failed",
        "credential marker",
        _PRIVATE_VISION_PROMPT,
        _PRIVATE_VISION_MODEL,
        "private-provider",
        "private-vision-model",
        _PRIVATE_IMAGE_DIGEST,
        "Traceback",
    ):
        assert private not in text


@pytest.mark.asyncio
async def test_actual_vision_tool_resolver_failure_is_private_across_all_surfaces(
    monkeypatch, caplog
):
    from tools import image_source, vision_tools

    async def _resolver_failure(*_args, **_kwargs):
        raise image_source.ImageResolutionError(
            _PRIVATE_IMAGE_ERROR,
            src=str(_PRIVATE_IMAGE_PATH),
            origin="local",
        )

    debug_calls = []
    debug_saves = []
    monkeypatch.setattr(image_source, "resolve_image_source", _resolver_failure)
    monkeypatch.setattr(
        vision_tools._debug,
        "log_call",
        lambda event, payload: debug_calls.append(
            (event, json.loads(json.dumps(payload)))
        ),
    )
    monkeypatch.setattr(
        vision_tools._debug,
        "save",
        lambda: debug_saves.append(True),
    )

    with caplog.at_level(logging.DEBUG):
        result_json = await vision_tools.vision_analyze_tool(
            str(_PRIVATE_IMAGE_PATH),
            _PRIVATE_VISION_PROMPT,
            _PRIVATE_VISION_MODEL,
        )

    _assert_vision_telemetry_metadata_absent(
        _render_operational_records(caplog.records)
    )
    _assert_vision_telemetry_metadata_absent(json.dumps(debug_calls, sort_keys=True))
    _assert_vision_telemetry_metadata_absent(result_json)
    assert _all_record_tuples(caplog.records) == [
        (
            "tools.vision_tools",
            logging.INFO,
            "vision_analysis_started",
            (),
            None,
        ),
        (
            "tools.vision_tools",
            logging.ERROR,
            "vision_analysis_failed",
            (),
            None,
        ),
    ]
    assert debug_calls == []
    assert debug_saves == []
    assert json.loads(result_json) == {
        "success": False,
        "error": "Vision analysis failed.",
        "analysis": (
            "There was a problem with the request and the image could not be "
            "analyzed."
        ),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_error", "expected_error", "expected_analysis"),
    [
        (
            f"402 billing required at {_PRIVATE_IMAGE_PATH}",
            "Vision analysis failed: billing required.",
            (
                "Insufficient credits or payment required. Please top up your "
                "API provider account and try again."
            ),
        ),
        (
            f"model does not support image input at {_PRIVATE_IMAGE_PATH}",
            "Vision analysis failed: vision unsupported.",
            (
                "The selected model does not support vision or the request was "
                "not accepted by the server. Try a vision-capable model."
            ),
        ),
        (
            f"invalid_request image_url at {_PRIVATE_IMAGE_PATH}",
            "Vision analysis failed: invalid image.",
            (
                "The vision API rejected the image. This can happen when the "
                "image is in an unsupported format, corrupted, or still too "
                "large after auto-resize. Try a smaller JPEG/PNG and retry."
            ),
        ),
        (
            _PRIVATE_IMAGE_ERROR,
            "Vision analysis failed.",
            (
                "There was a problem with the request and the image could not be "
                "analyzed."
            ),
        ),
    ],
)
async def test_actual_vision_tool_public_errors_use_stable_safe_categories(
    monkeypatch,
    tmp_path,
    caplog,
    provider_error,
    expected_error,
    expected_analysis,
):
    from tools import image_source, vision_tools

    image_bytes = b"\x89PNG\r\n\x1a\nsynthetic-image"

    async def _resolver_success(*_args, **_kwargs):
        return image_source.ResolvedImage(
            data=image_bytes,
            mime="image/png",
            origin="local",
        )

    async def _provider_failure(**_kwargs):
        raise RuntimeError(
            f"{provider_error}; model={_PRIVATE_VISION_MODEL}; "
            f"prompt={_PRIVATE_VISION_PROMPT}"
        )

    debug_calls = []
    debug_saves = []
    monkeypatch.setattr(image_source, "resolve_image_source", _resolver_success)
    monkeypatch.setattr(vision_tools, "get_hermes_dir", lambda *_args: tmp_path)
    monkeypatch.setattr(vision_tools, "async_call_llm", _provider_failure)
    monkeypatch.setattr(
        vision_tools,
        "extract_content_or_reasoning",
        lambda _response: "unused",
    )
    monkeypatch.setattr(
        vision_tools._debug,
        "log_call",
        lambda event, payload: debug_calls.append(
            (event, json.loads(json.dumps(payload)))
        ),
    )
    monkeypatch.setattr(
        vision_tools._debug,
        "save",
        lambda: debug_saves.append(True),
    )

    with caplog.at_level(logging.DEBUG):
        result_json = await vision_tools.vision_analyze_tool(
            str(_PRIVATE_IMAGE_PATH),
            _PRIVATE_VISION_PROMPT,
            _PRIVATE_VISION_MODEL,
        )

    _assert_vision_telemetry_metadata_absent(
        _render_operational_records(caplog.records)
    )
    _assert_vision_telemetry_metadata_absent(json.dumps(debug_calls, sort_keys=True))
    _assert_vision_telemetry_metadata_absent(result_json)
    assert debug_calls == []
    assert debug_saves == []
    assert json.loads(result_json) == {
        "success": False,
        "error": expected_error,
        "analysis": expected_analysis,
    }


@pytest.mark.asyncio
async def test_actual_gateway_vision_failure_logs_are_private_but_hint_keeps_path(
    monkeypatch, caplog
):
    from tools import image_source

    async def _resolver_failure(*_args, **_kwargs):
        raise image_source.ImageResolutionError(
            _PRIVATE_IMAGE_ERROR,
            src=str(_PRIVATE_IMAGE_PATH),
            origin="local",
        )

    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)
    monkeypatch.setattr(image_source, "resolve_image_source", _resolver_failure)

    with caplog.at_level(logging.DEBUG):
        result = await runner._enrich_message_with_vision(
            "preserve downstream user text",
            [str(_PRIVATE_IMAGE_PATH)],
        )

    _assert_vision_telemetry_metadata_absent(
        _render_operational_records(caplog.records)
    )
    assert _all_record_tuples(caplog.records) == [
        (
            "gateway.run",
            logging.DEBUG,
            "vision_enrichment_started",
            (),
            None,
        ),
        (
            "tools.vision_tools",
            logging.INFO,
            "vision_analysis_started",
            (),
            None,
        ),
        (
            "tools.vision_tools",
            logging.ERROR,
            "vision_analysis_failed",
            (),
            None,
        ),
    ]
    assert result == (
        "[The user sent an image but I couldn't quite see it this time (>_<) "
        "You can try looking at it yourself with vision_analyze using image_url: "
        f"{_PRIVATE_IMAGE_PATH}]\n\npreserve downstream user text"
    )


@pytest.mark.asyncio
async def test_actual_vision_tool_success_keeps_analysis_and_closes_telemetry(
    monkeypatch, tmp_path, caplog
):
    from tools import image_source, vision_tools

    image_bytes = b"\x89PNG\r\n\x1a\nsynthetic-image"
    successful_analysis = "safe successful analysis"

    async def _resolver_success(*_args, **_kwargs):
        return image_source.ResolvedImage(
            data=image_bytes,
            mime="image/png",
            origin="local",
        )

    async def _provider_success(**_kwargs):
        return SimpleNamespace()

    debug_calls = []
    debug_saves = []
    monkeypatch.setattr(image_source, "resolve_image_source", _resolver_success)
    monkeypatch.setattr(vision_tools, "get_hermes_dir", lambda *_args: tmp_path)
    monkeypatch.setattr(vision_tools, "async_call_llm", _provider_success)
    monkeypatch.setattr(
        vision_tools,
        "extract_content_or_reasoning",
        lambda _response: successful_analysis,
    )
    monkeypatch.setattr(
        vision_tools._debug,
        "log_call",
        lambda event, payload: debug_calls.append(
            (event, json.loads(json.dumps(payload)))
        ),
    )
    monkeypatch.setattr(
        vision_tools._debug,
        "save",
        lambda: debug_saves.append(True),
    )

    with caplog.at_level(logging.DEBUG):
        result_json = await vision_tools.vision_analyze_tool(
            str(_PRIVATE_IMAGE_PATH),
            _PRIVATE_VISION_PROMPT,
            _PRIVATE_VISION_MODEL,
        )

    _assert_vision_telemetry_metadata_absent(
        _render_operational_records(caplog.records)
    )
    _assert_vision_telemetry_metadata_absent(json.dumps(debug_calls, sort_keys=True))
    assert json.loads(result_json) == {
        "success": True,
        "analysis": successful_analysis,
    }
    assert debug_calls == [
        (
            "vision_analyze_tool",
            {
                "success": True,
                "analysis_length": len(successful_analysis),
                "image_size_bytes": len(image_bytes),
            },
        )
    ]
    assert debug_saves == [True]


def test_actual_vision_tool_logging_and_debug_ast_census_is_closed():
    from tools import vision_tools

    tree = ast.parse(Path(vision_tools.__file__).read_text(encoding="utf-8"))
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "vision_analyze_tool"
    ]
    assert len(functions) == 1
    function = functions[0]

    logger_calls = []
    forbidden_logger_names = {
        "cleanup_error",
        "e",
        "err_str",
        "error_msg",
        "exception",
        "image_url",
        "model",
        "path",
        "temp_image_path",
        "user_prompt",
    }
    for call in ast.walk(function):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "logger"
            and call.func.attr
            in {"debug", "info", "warning", "error", "exception"}
        ):
            continue
        assert call.args
        assert isinstance(call.args[0], ast.Constant)
        assert isinstance(call.args[0].value, str)
        assert not call.keywords
        dynamic_names = {
            child.id
            for arg in call.args[1:]
            for child in ast.walk(arg)
            if isinstance(child, ast.Name)
        }
        assert dynamic_names.isdisjoint(forbidden_logger_names)
        logger_calls.append(call)

    event_calls = {
        call.args[0].value: call
        for call in logger_calls
        if call.args[0].value
        in {"vision_analysis_started", "vision_analysis_failed"}
    }
    assert set(event_calls) == {
        "vision_analysis_started",
        "vision_analysis_failed",
    }
    assert all(len(call.args) == 1 for call in event_calls.values())

    debug_assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "debug_call_data"
            for target in node.targets
        )
    ]
    assert len(debug_assignments) == 1
    debug_dict = debug_assignments[0].value
    assert isinstance(debug_dict, ast.Dict)
    assert all(key is not None for key in debug_dict.keys)
    assert {
        ast.literal_eval(key) for key in debug_dict.keys if key is not None
    } == {
        "success",
        "analysis_length",
        "image_size_bytes",
    }

    allowed_debug_fields = {"success", "analysis_length", "image_size_bytes"}
    debug_field_writes = []
    for node in ast.walk(function):
        if not isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "debug_call_data"
            ):
                continue
            assert isinstance(target.slice, ast.Constant)
            debug_field_writes.append(target.slice.value)
    assert set(debug_field_writes) <= allowed_debug_fields

    def _is_debug_call(node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "_debug"
        )

    debug_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _is_debug_call(node)
    ]
    assert [
        call.func.attr
        for call in debug_calls
        if isinstance(call.func, ast.Attribute)
    ] == ["log_call", "save"]
    failure_debug_calls = [
        call
        for handler in ast.walk(function)
        if isinstance(handler, ast.ExceptHandler)
        for call in ast.walk(handler)
        if isinstance(call, ast.Call) and _is_debug_call(call)
    ]
    assert failure_debug_calls == []


def test_gateway_image_routing_runtime_failure_log_is_metadata_free(
    monkeypatch, caplog
):
    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)

    def _runtime_failure(**_kwargs):
        raise RuntimeError(_PRIVATE_IMAGE_ERROR)

    monkeypatch.setattr(runner, "_resolve_session_agent_runtime", _runtime_failure)
    with (
        patch("agent.auxiliary_client._read_main_provider", return_value="test-provider"),
        patch("agent.auxiliary_client._read_main_model", return_value="test-model"),
        patch("agent.image_routing.decide_image_input_mode", return_value="text"),
        caplog.at_level(logging.DEBUG, logger="gateway.run"),
    ):
        result = runner._decide_image_input_mode(
            session_key="private-session-key",
            user_config={},
        )

    assert result == "text"
    assert _gateway_record_tuples(caplog) == [
        (logging.DEBUG, "image_routing_runtime_resolution_failed", (), None)
    ]
    _assert_attachment_metadata_absent(caplog.text)


def test_gateway_image_routing_decision_failure_log_is_metadata_free(caplog):
    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)

    with (
        patch(
            "agent.image_routing.decide_image_input_mode",
            side_effect=RuntimeError(_PRIVATE_IMAGE_ERROR),
        ),
        caplog.at_level(logging.DEBUG, logger="gateway.run"),
    ):
        result = runner._decide_image_input_mode(
            provider="test-provider",
            model="test-model",
            user_config={},
        )

    assert result == "text"
    assert _gateway_record_tuples(caplog) == [
        (logging.DEBUG, "image_routing_decision_failed", (), None)
    ]
    _assert_attachment_metadata_absent(caplog.text)


@pytest.mark.asyncio
async def test_prepare_inbound_vision_runtime_failure_log_is_metadata_free(
    monkeypatch, caplog
):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")}
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="privacy-chat",
        chat_type="dm",
        user_id="privacy-user",
        user_name="Privacy User",
    )
    event = MessageEvent(
        text="preserve this text",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=[str(_PRIVATE_IMAGE_PATH)],
        media_types=["image/png"],
    )
    enrichment_calls = []

    def _runtime_failure(**_kwargs):
        raise RuntimeError(_PRIVATE_IMAGE_ERROR)

    async def _vision_fallback(text, image_paths):
        enrichment_calls.append((text, list(image_paths)))
        return f"vision fallback for {image_paths[0]}\n\n{text}"

    monkeypatch.setattr(runner, "_decide_image_input_mode", lambda **_kwargs: "text")
    monkeypatch.setattr(runner, "_resolve_session_agent_runtime", _runtime_failure)
    monkeypatch.setattr(runner, "_enrich_message_with_vision", _vision_fallback)

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
            session_key="privacy-session",
        )

    assert enrichment_calls == [
        ("preserve this text", [str(_PRIVATE_IMAGE_PATH)])
    ]
    assert result == f"vision fallback for {_PRIVATE_IMAGE_PATH}\n\npreserve this text"
    assert _gateway_record_tuples(caplog) == [
        (logging.INFO, "image_routing_text count=1", (1,), None),
        (logging.DEBUG, "vision_enrichment_runtime_resolution_failed", (), None),
    ]
    _assert_attachment_metadata_absent(
        "\n".join(
            f"{message} {args!r} {exc_info!r}"
            for _, message, args, exc_info in _gateway_record_tuples(caplog)
        )
    )


@pytest.mark.asyncio
async def test_vision_enrichment_failure_logs_are_metadata_free_and_hint_keeps_path(
    monkeypatch, caplog
):
    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)

    async def _vision_failure(**_kwargs):
        raise RuntimeError(_PRIVATE_IMAGE_ERROR)

    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _vision_failure)

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        result = await runner._enrich_message_with_vision(
            "preserve this text",
            [str(_PRIVATE_IMAGE_PATH)],
        )

    assert result == (
        "[The user sent an image but something went wrong when I "
        "tried to look at it~ You can try examining it yourself "
        f"with vision_analyze using image_url: {_PRIVATE_IMAGE_PATH}]"
        "\n\npreserve this text"
    )
    assert _gateway_record_tuples(caplog) == [
        (logging.DEBUG, "vision_enrichment_started", (), None),
        (logging.ERROR, "vision_enrichment_failed", (), None),
    ]
    _assert_attachment_metadata_absent(
        "\n".join(
            f"{message} {args!r} {exc_info!r}"
            for _, message, args, exc_info in _gateway_record_tuples(caplog)
        )
    )


@pytest.mark.asyncio
async def test_background_vision_failure_log_is_metadata_free_and_prompt_falls_back(
    monkeypatch, caplog
):
    from gateway.config import Platform
    from gateway.session import SessionSource

    class _Adapter:
        def __init__(self):
            self.sent = []

        async def send(self, *args, **kwargs):
            self.sent.append((args, kwargs))

        @staticmethod
        def extract_media(response):
            return [], response

        @staticmethod
        def extract_images(response):
            return [], response

    agent_calls = []

    class _Agent:
        def __init__(self, **_kwargs):
            pass

        def run_conversation(self, *, user_message, task_id):
            agent_calls.append((user_message, task_id))
            return {"final_response": "background result", "messages": []}

    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)
    adapter = _Adapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="privacy-chat",
        chat_type="dm",
        user_id="privacy-user",
        user_name="Privacy User",
    )
    runner._provider_routing = {}
    runner._session_db = None
    monkeypatch.setattr(runner, "_adapter_for_source", lambda source: adapter)
    monkeypatch.setattr(
        runner,
        "_thread_metadata_for_source",
        lambda source, reply_to_message_id=None: {},
    )
    monkeypatch.setattr(
        runner,
        "_resolve_session_agent_runtime",
        lambda **kwargs: (
            "test-model",
            {"api_key": "test-key", "provider": "test-provider"},
        ),
    )
    monkeypatch.setattr(
        runner, "_resolve_session_reasoning_config", lambda **kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "_resolve_session_service_tier",
        lambda source=None, session_key=None: None,
    )
    monkeypatch.setattr(
        runner,
        "_resolve_turn_agent_config",
        lambda user_message, model, runtime_kwargs: {
            "model": model,
            "runtime": runtime_kwargs,
            "request_overrides": None,
        },
    )
    monkeypatch.setattr(runner, "_refresh_fallback_model", lambda: None)
    monkeypatch.setattr(runner, "_cleanup_agent_resources", lambda agent: None)
    enrichment_calls = []

    async def _vision_failure(user_text, image_paths):
        enrichment_calls.append((user_text, list(image_paths)))
        raise RuntimeError(_PRIVATE_IMAGE_ERROR)

    async def _inline_executor(func, *args):
        return func(*args)

    monkeypatch.setattr(runner, "_enrich_message_with_vision", _vision_failure)
    monkeypatch.setattr(runner, "_run_in_executor_with_context", _inline_executor)

    with (
        patch("gateway.run._load_gateway_config", return_value={}),
        patch("hermes_cli.tools_config._get_platform_tools", return_value=set()),
        patch("run_agent.AIAgent", _Agent),
        caplog.at_level(logging.WARNING, logger="gateway.run"),
    ):
        await runner._run_background_task_inner(
            "preserve background prompt",
            source,
            "privacy-task",
            media_urls=[str(_PRIVATE_IMAGE_PATH)],
            media_types=["image/png"],
        )

    assert enrichment_calls == [
        ("preserve background prompt", [str(_PRIVATE_IMAGE_PATH)])
    ]
    assert agent_calls == [("preserve background prompt", "privacy-task")]
    assert len(adapter.sent) == 1
    assert "background result" in adapter.sent[0][1]["content"]
    assert _gateway_record_tuples(caplog) == [
        (logging.WARNING, "background_vision_enrichment_failed", (), None)
    ]
    _assert_attachment_metadata_absent(
        "\n".join(
            f"{message} {args!r} {exc_info!r}"
            for _, message, args, exc_info in _gateway_record_tuples(caplog)
        )
    )


def test_gateway_image_routing_logging_ast_census_is_metadata_free():
    tree = ast.parse(Path(gateway_run.__file__).read_text(encoding="utf-8"))

    def _function(name):
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ]
        assert len(matches) == 1
        return matches[0]

    def _image_paths_block(function):
        matches = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "image_paths"
        ]
        assert len(matches) == 1
        return matches[0]

    def _logger_contract(node):
        calls = []
        for call in ast.walk(node):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "logger"
                and call.func.attr
                in {"debug", "info", "warning", "error", "exception"}
            ):
                continue
            assert call.args
            assert isinstance(call.args[0], ast.Constant)
            assert isinstance(call.args[0].value, str)
            assert not call.keywords
            calls.append(
                (
                    call.lineno,
                    call.func.attr,
                    call.args[0].value,
                    tuple(ast.unparse(arg) for arg in call.args[1:]),
                )
            )
        return [item[1:] for item in sorted(calls)]

    scopes = {
        "native_builder": _function("_build_native_run_message"),
        "prepare_images": _image_paths_block(
            _function("_prepare_inbound_message_text")
        ),
        "routing_decision": _function("_decide_image_input_mode"),
        "vision_enrichment": _function("_enrich_message_with_vision"),
        "background_images": _image_paths_block(
            _function("_run_background_task_inner")
        ),
    }
    assert {name: _logger_contract(node) for name, node in scopes.items()} == {
        "native_builder": [
            ("warning", "native_image_attachment_skipped count=%d", ("len(skipped)",)),
            ("warning", "native_image_attachment_failed", ()),
        ],
        "prepare_images": [
            ("info", "image_routing_native count=%d", ("len(image_paths)",)),
            ("info", "image_routing_text count=%d", ("len(image_paths)",)),
            ("debug", "vision_enrichment_runtime_resolution_failed", ()),
        ],
        "routing_decision": [
            ("debug", "image_routing_runtime_resolution_failed", ()),
            ("debug", "image_routing_decision_failed", ()),
        ],
        "vision_enrichment": [
            ("debug", "vision_enrichment_started", ()),
            ("error", "vision_enrichment_failed", ()),
        ],
        "background_images": [
            ("warning", "background_vision_enrichment_failed", ()),
        ],
    }
    for node in scopes.values():
        assert all(
            handler.name is None
            for handler in ast.walk(node)
            if isinstance(handler, ast.ExceptHandler)
        )


def _configure_cli_attachment_caller(monkeypatch):
    import cli as cli_module

    cli = cli_module.HermesCLI.__new__(cli_module.HermesCLI)
    cli.agent = SimpleNamespace(_session_messages=[])
    cli.conversation_history = []
    cli.provider = "test-provider"
    cli.model = "test-model"
    cli.requested_provider = "test-provider"
    cli._active_agent_route_signature = None

    monkeypatch.setattr(cli_module, "set_secret_capture_callback", lambda _callback: None)
    monkeypatch.setattr(cli, "_ensure_runtime_credentials", lambda: True)
    monkeypatch.setattr(
        cli,
        "_resolve_turn_agent_config",
        lambda _message: {
            "signature": None,
            "model": None,
            "runtime": None,
            "request_overrides": None,
        },
    )
    monkeypatch.setattr(cli, "_init_agent", lambda **_kwargs: True)

    fallback_calls = []

    def _fallback(text, images):
        fallback_calls.append((text, list(images)))
        raise _StopAfterAttachmentFallback

    monkeypatch.setattr(cli, "_preprocess_images_with_vision", _fallback)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    return cli, fallback_calls


def _configure_tui_attachment_caller(monkeypatch, tmp_path):
    from tools.process_registry import process_registry
    from tui_gateway import server

    class _ImmediateThread:
        def __init__(self, target=None, **_kwargs):
            self._target = target

        def start(self):
            if self._target is not None:
                self._target()

    turns = []

    class _Agent:
        model = "test-model"
        provider = "test-provider"
        requested_provider = "test-provider"
        api_mode = ""

        def clear_interrupt(self):
            return None

        def run_conversation(self, prompt, **_kwargs):
            turns.append(prompt)
            return {"final_response": "", "messages": []}

    session = {
        "agent": _Agent(),
        "session_key": "privacy-session",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "attached_images": [str(_PRIVATE_IMAGE_PATH)],
        "cols": 80,
        "show_reasoning": False,
        "tool_progress_mode": "all",
    }
    fallback_calls = []

    def _fallback(text, images):
        fallback_calls.append((text, list(images)))
        return "plain-text fallback"

    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(server, "render_message", lambda _raw, _cols: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_apply_pending_model_switch", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_args: False)
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr(server, "_voice_mode_enabled", lambda: False)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_pending_reaction_notes", lambda _session: "")
    monkeypatch.setattr(server, "_hud_surface_note", lambda _session: "")
    monkeypatch.setattr(server, "_load_interim_assistant_messages", lambda: False)
    monkeypatch.setattr(
        server, "_plan_goal_compression_recovery", lambda *_args, **_kwargs: (None, None)
    )
    monkeypatch.setattr(server, "_is_successful_goal_turn", lambda *_args: False)
    monkeypatch.setattr(server, "_emit_settled_session_info", lambda *_args: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "record_turn_start", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_retire_turn_marker", lambda *_args: None)
    monkeypatch.setattr(server, "_enrich_with_attached_images", _fallback)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr("hermes_cli.mem_trim.trim_memory", lambda **_kwargs: None)
    monkeypatch.setattr(process_registry, "drain_notifications", lambda **_kwargs: [])
    monkeypatch.setattr("tools.tts_streaming.take_speech_interrupted", lambda: False)
    return server, session, turns, fallback_calls


def test_cli_chat_native_attachment_failure_log_is_metadata_free(monkeypatch, caplog):
    cli, fallback_calls = _configure_cli_attachment_caller(monkeypatch)

    with (
        patch("agent.image_routing.decide_image_input_mode", return_value="native"),
        patch(
            "agent.image_routing.build_native_content_parts",
            side_effect=RuntimeError(_PRIVATE_IMAGE_ERROR),
        ),
        caplog.at_level(logging.WARNING),
        pytest.raises(_StopAfterAttachmentFallback),
    ):
        cli.chat("preserve this text", images=[_PRIVATE_IMAGE_PATH])

    assert fallback_calls == [("preserve this text", [_PRIVATE_IMAGE_PATH])]
    operational = "\n".join(
        f"{record.getMessage()} {record.args!r} {record.exc_info!r}"
        for record in caplog.records
    )
    _assert_attachment_metadata_absent(operational)
    assert [
        (record.levelno, record.getMessage(), record.args, record.exc_info)
        for record in caplog.records
    ] == [(logging.WARNING, "native_image_attachment_failed", (), None)]


def test_cli_chat_image_routing_decision_failure_log_is_metadata_free(
    monkeypatch, caplog
):
    cli, fallback_calls = _configure_cli_attachment_caller(monkeypatch)

    with (
        patch(
            "agent.image_routing.decide_image_input_mode",
            side_effect=RuntimeError(_PRIVATE_IMAGE_ERROR),
        ),
        patch("agent.image_routing.build_native_content_parts") as build_parts,
        caplog.at_level(logging.DEBUG),
        pytest.raises(_StopAfterAttachmentFallback),
    ):
        cli.chat("preserve this text", images=[_PRIVATE_IMAGE_PATH])

    build_parts.assert_not_called()
    assert fallback_calls == [("preserve this text", [_PRIVATE_IMAGE_PATH])]
    operational = "\n".join(
        f"{record.getMessage()} {record.args!r} {record.exc_info!r}"
        for record in caplog.records
    )
    _assert_attachment_metadata_absent(operational)
    assert [
        (record.levelno, record.getMessage(), record.args, record.exc_info)
        for record in caplog.records
    ] == [(logging.DEBUG, "image_routing_decision_failed", (), None)]


def test_tui_prompt_submit_native_attachment_failure_stderr_is_metadata_free(
    monkeypatch, tmp_path, capsys
):
    from agent import image_routing

    server, session, turns, fallback_calls = _configure_tui_attachment_caller(
        monkeypatch, tmp_path
    )
    monkeypatch.setattr(image_routing, "decide_image_input_mode", lambda *_a, **_k: "native")
    monkeypatch.setattr(
        image_routing,
        "build_native_content_parts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError(_PRIVATE_IMAGE_ERROR)
        ),
    )

    server._run_prompt_submit("rid", "sid", session, "preserve this text")

    assert fallback_calls == [
        ("preserve this text", [str(_PRIVATE_IMAGE_PATH)])
    ]
    assert turns == ["plain-text fallback"]
    stderr = capsys.readouterr().err
    _assert_attachment_metadata_absent(stderr)
    assert stderr == "[tui_gateway] native_image_attachment_failed\n"


def test_tui_prompt_submit_image_routing_decision_failure_stderr_is_metadata_free(
    monkeypatch, tmp_path, capsys
):
    from agent import image_routing

    server, session, turns, fallback_calls = _configure_tui_attachment_caller(
        monkeypatch, tmp_path
    )

    def _decision_failure(*_args, **_kwargs):
        raise RuntimeError(_PRIVATE_IMAGE_ERROR)

    monkeypatch.setattr(image_routing, "decide_image_input_mode", _decision_failure)
    build_calls = []
    monkeypatch.setattr(
        image_routing,
        "build_native_content_parts",
        lambda *_args, **_kwargs: build_calls.append(True),
    )

    server._run_prompt_submit("rid", "sid", session, "preserve this text")

    assert build_calls == []
    assert fallback_calls == [
        ("preserve this text", [str(_PRIVATE_IMAGE_PATH)])
    ]
    assert turns == ["plain-text fallback"]
    stderr = capsys.readouterr().err
    _assert_attachment_metadata_absent(stderr)
    assert stderr == "[tui_gateway] image_routing_decision_failed\n"


_PRIVATE_TURN_MODEL = "private-provider/private-model"
_PRIVATE_TURN_PROVIDER = "private-provider"
_PRIVATE_TURN_SESSION = "private-session-uid-918273"
_PRIVATE_TURN_SESSION_ID = "private-session-id-uid-918273"
_PRIVATE_TURN_SIGNATURE = ("private-signature-uid-918273",)


class _StopAfterTurnRuntimeResolved(Exception):
    """Bound run_sync immediately after successful runtime resolution."""


class _StopAfterTurnCacheReuse(Exception):
    """Bound run_sync after the real cached-agent reuse branch."""


class _StopAfterTurnCacheCreated(Exception):
    """Bound run_sync after the real fresh-agent creation branch."""


def _make_turn_runner_privacy_harness(
    monkeypatch,
    *,
    runtime_failure=False,
    stop_after_runtime=False,
    cache_mode="none",
):
    """Build a real TurnContext/TurnRunner with bounded downstream seams."""
    from collections import OrderedDict

    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext

    state = SimpleNamespace(
        cache_cap_calls=[],
        constructor_calls=[],
        fallback_calls=[],
        init_calls=[],
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="private-chat-uid-918273",
        chat_type="dm",
        user_id="private-user-uid-918273",
        user_name="Private User",
    )
    owner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)
    owner._get_system_prompt_for_channel = lambda *_args, **_kwargs: None

    def _resolve_runtime(**_kwargs):
        if runtime_failure:
            raise RuntimeError(
                f"provider auth file failed at {_PRIVATE_IMAGE_PATH}; "
                f"model={_PRIVATE_TURN_MODEL}; provider={_PRIVATE_TURN_PROVIDER}; "
                f"session={_PRIVATE_TURN_SESSION}"
            )
        return (
            _PRIVATE_TURN_MODEL,
            {"provider": _PRIVATE_TURN_PROVIDER, "api_key": "private-api-key"},
        )

    def _resolve_reasoning(**_kwargs):
        if stop_after_runtime:
            raise _StopAfterTurnRuntimeResolved
        return None

    def _init_cached_agent(agent, interrupt_depth):
        state.init_calls.append((agent, interrupt_depth))

    def _apply_fallback(agent, fallback_model):
        state.fallback_calls.append((agent, fallback_model))
        raise _StopAfterTurnCacheReuse

    class _CreatedAgent:
        def __init__(self, **kwargs):
            state.constructor_calls.append(kwargs)

        def __setattr__(self, name, value):
            if name == "tool_progress_callback":
                raise _StopAfterTurnCacheCreated
            object.__setattr__(self, name, value)

    owner._resolve_session_agent_runtime = _resolve_runtime
    owner._provider_routing = {}
    owner._resolve_session_reasoning_config = _resolve_reasoning
    owner._resolve_session_service_tier = lambda **_kwargs: None
    owner._resolve_turn_agent_config = lambda message, model, runtime: {
        "model": model,
        "runtime": runtime,
        "request_overrides": None,
    }
    owner._agent_config_signature = lambda *_args, **_kwargs: _PRIVATE_TURN_SIGNATURE
    owner._extract_cache_busting_config = lambda _config: ()
    owner._session_db = None
    owner._prefill_messages = []
    owner._refresh_fallback_model = lambda: "fallback-control"
    owner._init_cached_agent_for_turn = _init_cached_agent
    owner._apply_fallback_chain_to_agent = _apply_fallback
    owner._enforce_agent_cache_cap = lambda: state.cache_cap_calls.append(True)
    owner.config = SimpleNamespace(streaming=None)
    owner._agent_cache_lock = threading.Lock()
    owner._agent_cache = OrderedDict()

    cached_agent = None
    if cache_mode == "reuse":
        cached_agent = SimpleNamespace(max_iterations=1)
        owner._agent_cache[_PRIVATE_TURN_SESSION] = (
            cached_agent,
            _PRIVATE_TURN_SIGNATURE,
            None,
            _PRIVATE_TURN_SESSION_ID,
        )
        owner._agent_cache["other-session"] = (
            SimpleNamespace(max_iterations=2),
            ("other-signature",),
            None,
            "other-session-id",
        )

    ctx = TurnContext(
        source=source,
        _run_still_current=lambda: True,
        message="preserve this user message",
        history=[],
        context_prompt="",
        channel_prompt="",
        session_id=_PRIVATE_TURN_SESSION_ID,
        session_key=_PRIVATE_TURN_SESSION,
        _interrupt_depth=4,
        user_config={},
        enabled_toolsets=[],
        disabled_toolsets=[],
        AIAgent=_CreatedAgent,
        resolve_display_setting=lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(gateway_run, "_current_max_iterations", lambda: 37)
    monkeypatch.setattr(gateway_run, "_checkpoint_agent_kwargs", lambda _config: {})
    return gateway_run.TurnRunner(owner, ctx), owner, ctx, state, cached_agent


def _assert_turn_runtime_metadata_absent(text):
    for private in (
        str(_PRIVATE_IMAGE_PATH),
        _PRIVATE_IMAGE_PATH.name,
        _PRIVATE_TURN_MODEL,
        _PRIVATE_TURN_PROVIDER,
        _PRIVATE_TURN_SESSION,
        _PRIVATE_TURN_SESSION_ID,
        _PRIVATE_TURN_SIGNATURE[0],
        "provider auth file failed",
        "private-user",
        "uid-918273",
        "Traceback",
    ):
        assert private not in text


def test_turn_runner_runtime_resolution_failure_is_fixed_and_private(
    monkeypatch, caplog
):
    turn_runner, _owner, _ctx, _state, _cached_agent = (
        _make_turn_runner_privacy_harness(monkeypatch, runtime_failure=True)
    )

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        result = turn_runner.run_sync()

    assert result == {
        "final_response": "⚠️ Provider authentication failed.",
        "messages": [],
        "api_calls": 0,
        "tools": [],
    }
    assert _gateway_record_tuples(caplog) == [
        (logging.DEBUG, "run_agent_runtime_resolution_failed", (), None)
    ]
    _assert_turn_runtime_metadata_absent(
        json.dumps(result, sort_keys=True)
        + "\n"
        + _render_operational_records(caplog.records)
    )


def test_turn_runner_runtime_resolution_success_log_is_fixed_and_private(
    monkeypatch, caplog
):
    turn_runner, _owner, _ctx, _state, _cached_agent = (
        _make_turn_runner_privacy_harness(monkeypatch, stop_after_runtime=True)
    )

    with (
        caplog.at_level(logging.DEBUG, logger="gateway.run"),
        pytest.raises(_StopAfterTurnRuntimeResolved),
    ):
        turn_runner.run_sync()

    assert _gateway_record_tuples(caplog) == [
        (logging.DEBUG, "run_agent_runtime_resolved", (), None)
    ]
    _assert_turn_runtime_metadata_absent(_render_operational_records(caplog.records))


def test_turn_runner_cached_agent_reuse_log_is_fixed_and_preserves_semantics(
    monkeypatch, caplog
):
    turn_runner, owner, _ctx, state, cached_agent = (
        _make_turn_runner_privacy_harness(monkeypatch, cache_mode="reuse")
    )

    with (
        caplog.at_level(logging.DEBUG, logger="gateway.run"),
        pytest.raises(_StopAfterTurnCacheReuse),
    ):
        turn_runner.run_sync()

    assert state.init_calls == [(cached_agent, 4)]
    assert cached_agent.max_iterations == 37
    assert list(owner._agent_cache) == ["other-session", _PRIVATE_TURN_SESSION]
    assert state.fallback_calls == [(cached_agent, "fallback-control")]
    assert state.constructor_calls == []
    assert state.cache_cap_calls == []
    assert _gateway_record_tuples(caplog) == [
        (logging.DEBUG, "run_agent_runtime_resolved", (), None),
        (logging.DEBUG, "agent_cache_reused", (), None),
    ]
    _assert_turn_runtime_metadata_absent(_render_operational_records(caplog.records))


def test_turn_runner_new_agent_creation_log_is_fixed_and_preserves_cache_tuple(
    monkeypatch, caplog
):
    turn_runner, owner, _ctx, state, _cached_agent = (
        _make_turn_runner_privacy_harness(monkeypatch)
    )

    with (
        caplog.at_level(logging.DEBUG, logger="gateway.run"),
        pytest.raises(_StopAfterTurnCacheCreated),
    ):
        turn_runner.run_sync()

    assert len(state.constructor_calls) == 1
    constructor_kwargs = state.constructor_calls[0]
    assert constructor_kwargs["model"] == _PRIVATE_TURN_MODEL
    assert constructor_kwargs["provider"] == _PRIVATE_TURN_PROVIDER
    assert constructor_kwargs["gateway_session_key"] == _PRIVATE_TURN_SESSION
    assert constructor_kwargs["session_id"] == _PRIVATE_TURN_SESSION_ID
    assert constructor_kwargs["max_iterations"] == 37
    assert constructor_kwargs["fallback_model"] == "fallback-control"
    created_agent = owner._agent_cache[_PRIVATE_TURN_SESSION][0]
    assert owner._agent_cache[_PRIVATE_TURN_SESSION] == (
        created_agent,
        _PRIVATE_TURN_SIGNATURE,
        None,
        _PRIVATE_TURN_SESSION_ID,
    )
    assert state.cache_cap_calls == [True]
    assert state.init_calls == []
    assert state.fallback_calls == []
    assert _gateway_record_tuples(caplog) == [
        (logging.DEBUG, "run_agent_runtime_resolved", (), None),
        (logging.DEBUG, "agent_cache_created", (), None),
    ]
    _assert_turn_runtime_metadata_absent(_render_operational_records(caplog.records))


def test_turn_runner_runtime_and_cache_logging_ast_census_is_closed():
    tree = ast.parse(Path(gateway_run.__file__).read_text(encoding="utf-8"))
    turn_runner_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TurnRunner"
    ]
    assert len(turn_runner_classes) == 1
    run_sync_functions = [
        node
        for node in turn_runner_classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "run_sync"
    ]
    assert len(run_sync_functions) == 1
    run_sync = run_sync_functions[0]

    def _is_named_call(node, name):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == name
        )

    def _logger_contract(statement):
        assert isinstance(statement, ast.Expr)
        call = statement.value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Attribute)
        assert isinstance(call.func.value, ast.Name)
        assert call.func.value.id == "logger"
        assert call.func.attr in {"debug", "info", "warning", "error", "exception"}
        assert len(call.args) == 1
        assert isinstance(call.args[0], ast.Constant)
        assert isinstance(call.args[0].value, str)
        assert call.keywords == []
        forbidden_names = {
            "ctx",
            "exc",
            "exception",
            "model",
            "provider",
            "runtime_kwargs",
            "session",
            "session_key",
            "_sig",
        }
        assert {
            node.id for node in ast.walk(call) if isinstance(node, ast.Name)
        }.isdisjoint(forbidden_names)
        return call.func.attr, call.args[0].value

    resolver_tries = [
        node
        for node in ast.walk(run_sync)
        if isinstance(node, ast.Try)
        and any(
            _is_named_call(call, "_resolve_session_agent_runtime")
            for call in ast.walk(node)
        )
    ]
    assert len(resolver_tries) == 1
    resolver_try = resolver_tries[0]
    assert len(resolver_try.handlers) == 1
    resolver_handler = resolver_try.handlers[0]
    assert resolver_handler.name is None
    assert _logger_contract(resolver_try.body[-1]) == (
        "debug",
        "run_agent_runtime_resolved",
    )
    assert _logger_contract(resolver_handler.body[0]) == (
        "debug",
        "run_agent_runtime_resolution_failed",
    )

    parents = {
        id(child): parent
        for parent in ast.walk(run_sync)
        for child in ast.iter_child_nodes(parent)
    }
    reuse_assignments = [
        node
        for node in ast.walk(run_sync)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "reused_cached_agent"
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
    ]
    assert len(reuse_assignments) == 1
    reuse_assignment = reuse_assignments[0]
    reuse_parent = parents[id(reuse_assignment)]
    reuse_bodies = [
        body
        for body in (
            getattr(reuse_parent, "body", []),
            getattr(reuse_parent, "orelse", []),
        )
        if reuse_assignment in body
    ]
    assert len(reuse_bodies) == 1
    reuse_body = reuse_bodies[0]
    reuse_index = reuse_body.index(reuse_assignment)
    assert _logger_contract(reuse_body[reuse_index - 1]) == (
        "debug",
        "agent_cache_reused",
    )

    creation_ifs = [
        node
        for node in ast.walk(run_sync)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "agent"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Is)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value is None
    ]
    assert len(creation_ifs) == 1
    assert _logger_contract(creation_ifs[0].body[-1]) == (
        "debug",
        "agent_cache_created",
    )
