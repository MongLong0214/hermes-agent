"""Tests for gateway /compress user-facing messaging."""

import asyncio
import threading
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str = "/compress") -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_history() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]


def _make_runner(history: list[dict[str, str]]):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = history
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store._save = MagicMock()
    runner._session_db = None
    return runner


@pytest.mark.asyncio
async def test_compress_command_works_when_auto_compaction_disabled():
    """compression.enabled: false disables *automatic* compaction only.

    The gateway /compress handler has never gated on the flag — pin that
    contract (every manual-compress surface must allow manual compression
    regardless of the auto toggle, #64438) and the force=True cooldown
    bypass that manual compression relies on."""
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "compressed summary"},
        history[-1],
    ]
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.compression_enabled = False
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")
    # Explicit non-lock-skip: MagicMock getattr would return a truthy mock.
    agent_instance._compression_skipped_due_to_lock = False

    def _estimate(messages, **_kwargs):
        return 100 if messages == history else 60

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "disabled" not in result.lower()
    assert "Compressed:" in result
    agent_instance._compress_context.assert_called_once()
    assert agent_instance._compress_context.call_args.kwargs.get("force") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("warning_notifications", [True, False])
async def test_compress_command_surfaces_aux_model_failure_even_when_recovered(tmp_path, monkeypatch, warning_notifications):
    """When the user's configured ``auxiliary.compression.model`` errors out
    but compression recovers by retrying on the main model, /compress must
    STILL inform the user.  Silent recovery hides broken config the user
    needs to fix."""
    import gateway.run as gateway_run
    (tmp_path / "config.yaml").write_text(f"display: {{warning_notifications: {str(warning_notifications).lower()}}}")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    history = _make_history()
    # Compressed transcript — normal successful compression, no placeholder.
    compressed = [
        history[0],
        {"role": "assistant", "content": "summary via main model"},
        history[-1],
    ]
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    # Fallback placeholder was NOT used — recovery succeeded.
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance.context_compressor._last_summary_fallback_used = False
    agent_instance.context_compressor._last_summary_dropped_count = 0
    agent_instance.context_compressor._last_summary_error = None
    # But the configured aux model DID fail before the retry succeeded.
    agent_instance.context_compressor._last_aux_model_failure_model = (
        "gemini-3-flash-preview"
    )
    agent_instance.context_compressor._last_aux_model_failure_error = (
        "404 model not found: gemini-3-flash-preview"
    )
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")
    agent_instance._compression_skipped_due_to_lock = False

    def _estimate(messages, **_kwargs):
        if messages == history:
            return 100
        if messages == compressed:
            return 60
        raise AssertionError(f"unexpected transcript: {messages!r}")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    # Compression succeeded
    assert "Compressed:" in result
    # The broken aux model is surfaced to the user
    assert "gemini-3-flash-preview" in result
    assert "404" in result
    agent_instance.shutdown_memory_provider.assert_called_once()
    agent_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_compress_command_aux_failure_reply_scrubs_credentials(tmp_path, monkeypatch):
    """L1-3 regression: the manual /compress reply is returned to the adapter's inline delivery path
    (``BasePlatformAdapter._dispatch_inline_reply``), which sends it verbatim — no final-response
    sanitization. A recovered aux-model failure whose provider error echoes a credential (a
    vendor-prefixed key or an opaque ``Bearer`` token) must not reach the chat."""
    import gateway.run as gateway_run
    (tmp_path / "config.yaml").write_text("display: {warning_notifications: true}")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    secret = "sk-testFAKEKEY1234567890ABCDEFGHIJ"
    opaque = "opaqueFixtureToken1234567890ABCDE"
    # R-COMPRESSION-SECRETS: a credential-bearing URL query param carries no vendor prefix and
    # is not a "Bearer <token>" shape, so only strict URL-credential redaction catches it.
    url_cred_token = "opaqueFixtureToken1234567890ABCDE"
    url_cred = f"https://service.example/api?access_token={url_cred_token}"
    # R1 witness: a pre-signed URL signature param, declared with a hyphen in
    # _SENSITIVE_QUERY_PARAMS ("x-amz-signature") — the strict matcher's canonicalization bug
    # let this survive verbatim even though access_token (above) was already masked correctly.
    sig_token = "SIGFIXTURE" + "a" * 54
    sig_url = f"https://bucket.s3.amazonaws.com/key?X-Amz-Signature={sig_token}"
    history = _make_history()
    compressed = [history[0], {"role": "assistant", "content": "summary via main model"}, history[-1]]
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance.context_compressor._last_summary_fallback_used = False
    agent_instance.context_compressor._last_summary_dropped_count = 0
    agent_instance.context_compressor._last_summary_error = None
    agent_instance.context_compressor._last_aux_model_failure_model = "fixture-aux"
    agent_instance.context_compressor._last_aux_model_failure_error = (
        f"litellm.AuthenticationError: Incorrect API key provided: {secret}; "
        f"upstream rejected Bearer {opaque}; GET {url_cred} failed; GET {sig_url} failed")
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")
    agent_instance._compression_skipped_due_to_lock = False

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough",
              side_effect=lambda messages, **_kw: 100 if messages == history else 60),
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "Compressed:" in result
    assert "fixture-aux" in result  # the notice itself still reaches the user
    for credential in (secret, opaque, url_cred_token, sig_token):
        assert credential not in result, f"raw credential in the /compress reply: {result!r}"


@pytest.mark.asyncio
async def test_compress_command_aborted_reply_scrubs_url_credential_param(tmp_path, monkeypatch):
    """R-COMPRESSION-SECRETS: the ABORTED branch of ``_manual_compression_reply_lines``
    (``_last_compress_aborted`` -> ``gateway.compress.aborted``, gateway/slash_commands_session.py
    ~82-86) must not echo a credential-bearing URL query param from the summariser's provider
    exception. Sibling of the aux-failure branch covered just above; the reviewer's exact fixture:
    ``https://service.example/api?access_token=opaqueFixtureToken1234567890ABCDE``."""
    import gateway.run as gateway_run
    (tmp_path / "config.yaml").write_text("display: {warning_notifications: true}")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    url_cred_token = "opaqueFixtureToken1234567890ABCDE"
    url_cred = f"https://service.example/api?access_token={url_cred_token}"
    # R1 witness: hyphenated sensitive param name — see the aux-failure sibling test above for
    # why this is a distinct regression from the access_token case.
    sig_token = "SIGFIXTURE" + "a" * 54
    sig_url = f"https://bucket.s3.amazonaws.com/key?X-Amz-Signature={sig_token}"
    history = _make_history()
    # Aborted: no usable summary, messages unchanged (fallback marker), but compress_now still
    # reports status "compressed" (see agent/conversation_compression_manual.py:138).
    compressed = list(history)
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.context_compressor._last_compress_aborted = True
    agent_instance.context_compressor._last_summary_fallback_used = False
    agent_instance.context_compressor._last_summary_dropped_count = 0
    agent_instance.context_compressor._last_summary_error = (
        f"litellm.APIError: upstream rejected GET {url_cred}; GET {sig_url} failed")
    agent_instance.context_compressor._last_aux_model_failure_model = None
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")
    agent_instance._compression_skipped_due_to_lock = False

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough",
              side_effect=lambda messages, **_kw: 100 if messages == history else 100),
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "Compression aborted" in result
    for credential in (url_cred_token, sig_token):
        assert credential not in result, f"raw URL credential in the /compress reply: {result!r}"


@pytest.mark.asyncio
async def test_compress_command_in_place_skips_destructive_rewrite():
    """In-place compaction (compression.in_place / #38763) persists via
    archive_and_compact() inside _compress_context — the previous active rows
    are soft-archived and the compacted set inserted. Calling
    rewrite_transcript() afterwards would invoke
    replace_messages(active_only=False), DELETEing the just-archived rows
    (silent data loss, #61145). The handler must skip the rewrite and still
    report success."""
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "compacted summary"},
        history[-1],
    ]
    runner = _make_runner(history)
    runner._session_db = object()
    session_entry = runner.session_store.get_or_create_session.return_value
    runner.session_store.rewrite_transcript = MagicMock()

    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    # In-place compaction: session_id is UNCHANGED but marked as a success.
    agent_instance._last_compaction_in_place = True
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")
    agent_instance._compression_skipped_due_to_lock = False

    def _estimate(messages, **_kwargs):
        if messages == history:
            return 100
        if messages == compressed:
            return 60
        raise AssertionError(f"unexpected transcript: {messages!r}")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "Compressed:" in result
    # The destructive rewrite must NOT run — archive_and_compact() already
    # persisted, and rewrite_transcript would wipe the archived rows.
    runner.session_store.rewrite_transcript.assert_not_called()
    assert session_entry.session_id == "sess-1"
    agent_instance.shutdown_memory_provider.assert_called_once()
    agent_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_compress_command_preserves_platform_and_gateway_session_key():
    """The temporary compression agent must carry the originating source's
    platform and stable gateway session key, matching a normal gateway turn.
    Without them ``_session_source_for_agent`` falls back to a default "cli"
    host source, so an external context engine misattributes the retained
    transcript tail and later duplicates it on resume (#50422)."""
    history = _make_history()
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (list(history), "")
    agent_instance._compression_skipped_due_to_lock = False

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance) as mock_agent,
        patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100),
    ):
        await runner._handle_compress_command(_make_event())

    assert mock_agent.call_count == 1
    _, kwargs = mock_agent.call_args
    # Platform preserved as the live turn's config key (TELEGRAM -> "telegram"),
    # not the unbound "cli"/"local" fallback.
    assert kwargs.get("platform") == "telegram"
    # Stable gateway session key preserved, identical to a normal gateway turn.
    assert kwargs.get("gateway_session_key") == runner._session_key_for_source(_make_source())
    assert kwargs["gateway_session_key"]


@pytest.mark.asyncio
async def test_compress_command_agent_receives_configured_reasoning():
    """#85153 class: the throwaway /compress agent is an ``AIAgent()`` built from gateway config, so
    ``agent.reasoning_effort: none`` must reach it like a normal gateway turn — otherwise the transport
    applies its default effort (a 400 on non-reasoning models)."""
    history = _make_history()
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (list(history), "")
    agent_instance._compression_skipped_due_to_lock = False

    with (
        patch("gateway.run._load_gateway_config", return_value={"agent": {"reasoning_effort": "none"}}),
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
        patch("gateway.run._resolve_gateway_model", return_value="gpt-4o-mini"),
        patch("run_agent.AIAgent", return_value=agent_instance) as mock_agent,
        patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100),
    ):
        await runner._handle_compress_command(_make_event())

    assert mock_agent.call_count == 1
    _, kwargs = mock_agent.call_args
    assert kwargs["reasoning_config"] == {"enabled": False}


@pytest.mark.asyncio
async def test_compress_command_passes_tool_messages_to_compressor():
    """Tool results must reach _compress_context (#3854).

    Filtering the transcript to user/assistant-only starved the
    compressor's tool-result pruning — tool messages are usually the bulk
    of the context.
    """
    history = [
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "t1", "type": "function",
                            "function": {"name": "x", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "BIG RESULT " * 50, "tool_call_id": "t1"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "np"},
    ]
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (list(history), "")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100),
    ):
        await runner._handle_compress_command(_make_event())

    args, _kwargs = agent_instance._compress_context.call_args
    passed = args[0]
    roles = [m.get("role") for m in passed]
    assert "tool" in roles, f"tool messages filtered out: {roles}"
    # Assistant tool_calls stubs (content=None) must survive too, or the
    # tool message would dangle without its call.
    assert any(m.get("tool_calls") for m in passed), "assistant tool_calls stub dropped"




@pytest.mark.asyncio
async def test_compress_command_multiplexed_runs_under_profile_secret_scope(tmp_path):
    """Manual /compress must install the source profile's secret scope.

    Multiplexed gateways resolve credentials fail-closed (Workstream A):
    ``get_secret`` raises ``UnscopedSecretError`` on any read outside a
    ``set_secret_scope`` block. The agent turn is scoped by ``_run_agent``'s
    wrapper, but slash-command dispatch is not — manual /compress reached the
    compressor's provider resolution unscoped and died with
    ``get_secret('OPENROUTER_BASE_URL') called with no profile secret scope
    active``. The credential read happens inside the executor hop, so this
    also pins that the handler uses the contextvar-preserving executor
    (``_run_in_executor_with_context``), not a bare ``run_in_executor``.
    """
    from agent import secret_scope as ss

    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "compressed summary"},
        history[-1],
    ]
    runner = _make_runner(history)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
        multiplex_profiles=True,
    )
    profile_home = tmp_path / "profiles" / "milo"
    profile_home.mkdir(parents=True)
    (profile_home / ".env").write_text(
        "OPENROUTER_BASE_URL=https://scoped.example/v1\n"
    )
    runner._resolve_profile_home_for_source = MagicMock(return_value=profile_home)

    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance.context_compressor._last_summary_fallback_used = False
    agent_instance.context_compressor._last_summary_dropped_count = 0
    agent_instance.context_compressor._last_summary_error = None
    agent_instance.context_compressor._last_aux_model_failure_model = None
    agent_instance.context_compressor._last_aux_model_failure_error = None
    agent_instance.session_id = "sess-1"
    agent_instance._compression_skipped_due_to_lock = False

    seen: dict[str, str | None] = {}

    def _compress(*_args, **_kwargs):
        # Runs in the executor thread — exactly where the aux client
        # resolves provider credentials. Fail-closed get_secret raises
        # here unless the profile scope survived the thread hop.
        seen["base_url"] = ss.get_secret("OPENROUTER_BASE_URL")
        return (compressed, "")

    agent_instance._compress_context.side_effect = _compress

    ss.set_multiplex_active(True)
    try:
        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch("run_agent.AIAgent", return_value=agent_instance),
            patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100),
        ):
            result = await runner._handle_compress_command(_make_event())
    finally:
        ss.set_multiplex_active(False)
        runner._shutdown_executor()

    assert "failed" not in result.lower(), result
    assert seen["base_url"] == "https://scoped.example/v1"
    runner._resolve_profile_home_for_source.assert_called_once()




@pytest.mark.asyncio
async def test_compress_command_cleanup_does_not_block_event_loop():
    """Manual /compress must not run agent teardown on the gateway event loop.

    #53175 offloaded session-expiry, hygiene, and shutdown cleanup, but the
    manual /compress finally still called ``_cleanup_agent_resources`` inline.
    A slow ``agent.close()`` there freezes the whole loop and stops the
    runtime-status heartbeat from advancing — the same wedge class as the
    original incident.

    Observation must happen from a side thread: if cleanup blocks the event
    loop, an ``await``-based waiter cannot sample ticks until close returns,
    which falsely looks healthy after the block ends.
    """
    import time

    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "compressed summary"},
        history[-1],
    ]
    runner = _make_runner(history)

    close_started = threading.Event()
    release_close = threading.Event()

    def slow_close():
        close_started.set()
        release_close.wait(timeout=5)

    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = slow_close
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance.context_compressor._last_summary_fallback_used = False
    agent_instance.context_compressor._last_summary_dropped_count = 0
    agent_instance.context_compressor._last_summary_error = None
    agent_instance.context_compressor._last_aux_model_failure_model = None
    agent_instance.context_compressor._last_aux_model_failure_error = None
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")
    agent_instance._compression_skipped_due_to_lock = False
    agent_instance._session_messages = None

    ticks = {"n": 0}
    stop = threading.Event()
    observed = {}

    async def _heartbeat():
        while not stop.is_set():
            ticks["n"] += 1
            await asyncio.sleep(0.005)

    def _observer():
        # threading.Event wait does not need the event loop. Sample ticks
        # while close() is still held so an on-loop teardown is visible.
        if not close_started.wait(timeout=5):
            observed["error"] = "close() never started"
            release_close.set()
            return
        baseline = ticks["n"]
        time.sleep(0.12)
        observed["ticks_during_block"] = ticks["n"] - baseline
        release_close.set()

    hb = asyncio.create_task(_heartbeat())
    observer = threading.Thread(target=_observer, name="compress-cleanup-observer", daemon=True)
    observer.start()

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100),
    ):
        result = await runner._handle_compress_command(_make_event())

    observer.join(timeout=5)
    stop.set()
    await hb
    runner._shutdown_executor()

    assert "Compressed:" in result
    assert "error" not in observed, observed.get("error")
    assert observed.get("ticks_during_block", 0) >= 5, (
        "event loop was blocked during manual /compress cleanup: only "
        f"{observed.get('ticks_during_block')} ticks while agent.close() was running"
    )


@pytest.mark.asyncio
async def test_compress_command_failure_reply_scrubs_credentials(monkeypatch):
    """ROUND1-ESCAPE-2: the ordinary /compress failure reply (``_handle_compress_command_inner``'s
    except branch) must not echo a raw provider exception's credential straight into the chat.

    R-COMPRESSION-SECRETS: also covers the reviewer's exact credential-bearing URL query param,
    which the ordinary egress scrub (no vendor prefix, not a ``Bearer`` shape) lets through."""
    from gateway.run import GatewayRunner

    runner = _make_runner(_make_history())
    secret = "sk-testFAKEKEY1234567890ABCDEFGHIJ"
    url_cred_token = "opaqueFixtureToken1234567890ABCDE"
    url_cred = f"https://service.example/api?access_token={url_cred_token}"
    # R1 witness: hyphenated sensitive param name (see test_every_declared_sensitive_param_is_masked).
    sig_token = "SIGFIXTURE" + "a" * 54
    sig_url = f"https://bucket.s3.amazonaws.com/key?X-Amz-Signature={sig_token}"

    async def _boom(*args, **kwargs):
        raise RuntimeError(
            f"litellm.AuthenticationError: Incorrect API key provided: {secret}; "
            f"GET {url_cred} failed; GET {sig_url} failed")

    monkeypatch.setattr(GatewayRunner, "_run_manual_compression", _boom, raising=False)
    reply = await runner._handle_compress_command_inner(_make_event())
    assert secret not in reply
    for credential in (url_cred_token, sig_token):
        assert credential not in reply, f"raw URL credential in the /compress reply: {reply!r}"


@pytest.mark.asyncio
async def test_compress_command_codex_app_server_failure_reply_scrubs_credentials(monkeypatch):
    """ROUND1-ESCAPE-2, the Codex app-server variant's except branch (direct ``t(...)`` call with
    the raw exception, no redaction).

    R-COMPRESSION-SECRETS: also covers the reviewer's exact credential-bearing URL query param,
    which the ordinary egress scrub (no vendor prefix, not a ``Bearer`` shape) lets through."""
    from gateway.run import GatewayRunner

    runner = _make_runner(_make_history())
    secret = "sk-testFAKEKEY1234567890ABCDEFGHIJ"
    url_cred_token = "opaqueFixtureToken1234567890ABCDE"
    url_cred = f"https://service.example/api?access_token={url_cred_token}"
    # R1 witness: hyphenated sensitive param name (see test_every_declared_sensitive_param_is_masked).
    sig_token = "SIGFIXTURE" + "a" * 54
    sig_url = f"https://bucket.s3.amazonaws.com/key?X-Amz-Signature={sig_token}"
    agent = MagicMock()
    agent._codex_session = object()
    agent.context_compressor.compression_count = 0

    def _compress(*args, **kwargs):
        raise RuntimeError(
            f"litellm.AuthenticationError: Incorrect API key provided: {secret}; "
            f"GET {url_cred} failed; GET {sig_url} failed")

    agent._compress_context = _compress
    monkeypatch.setattr(GatewayRunner, "_cached_agent_for", lambda self, *a, **kw: agent, raising=False)

    async def _run_sync(fn):
        return fn()

    monkeypatch.setattr(GatewayRunner, "_run_in_executor_with_context", lambda self, fn: _run_sync(fn), raising=False)
    reply = await runner._compress_codex_app_server_session(session_key="k1", session_id="sess-1")
    assert secret not in reply
    for credential in (url_cred_token, sig_token):
        assert credential not in reply, f"raw URL credential in the /compress reply: {reply!r}"
