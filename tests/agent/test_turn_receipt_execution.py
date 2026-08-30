"""Receipt execution contracts over the real agent loop and SessionDB."""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from tui_gateway.turn_receipts import TurnReceiptAdapter, request_binding


def _tool_defs():
    return [{
        "type": "function",
        "function": {
            "name": "receipt_test_tool",
            "description": "test boundary",
            "parameters": {"type": "object", "properties": {}},
        },
    }]


@pytest.fixture()
def receipt_agent(tmp_path):
    from run_agent import AIAgent

    db = SessionDB(tmp_path / "state.db")
    session_id = "receipt-agent-session"
    db.create_session(session_id, source="test")
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            skip_background_review=False,
        )
    agent.client = MagicMock()
    agent.session_id = session_id
    agent._session_db = db
    agent._session_db_created = True
    agent._cached_system_prompt = "You are helpful."
    agent.api_mode = "chat_completions"
    agent.provider = "openrouter"
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    try:
        yield agent, db, session_id
    finally:
        db.close()


def _request(session_id, request_id="request-1"):
    return request_binding(
        session_id=session_id,
        turn_request_id=request_id,
        text="question",
        display_kind=None,
        attachments=[],
        truncation=None,
    )


def _provider_user_content(agent):
    """Return the current user bytes passed to the mock provider."""
    provider_kwargs = agent.client.chat.completions.create.call_args.kwargs
    return next(
        message["content"]
        for message in provider_kwargs["messages"]
        if message.get("role") == "user"
    )


def _install_mutable_request_seams(monkeypatch, calls):
    """Install seams that would visibly rewrite provider input if invoked."""
    import hermes_cli.lifecycle as lifecycle
    import hermes_cli.middleware as middleware

    def invoke_hook(name, *_args, **_kwargs):
        calls["hooks"].append(name)
        return [{"context": "PLUGIN_CONTEXT"}] if name == "pre_llm_call" else []

    def replace_request(request, **_kwargs):
        calls["request_middleware"] += 1
        replacement = dict(request)
        replacement["messages"] = [{"role": "user", "content": "REQUEST_REWRITE"}]
        return middleware.RequestMiddlewareResult(
            payload=replacement,
            original_payload=request,
            changed=True,
            trace=[{"source": "test"}],
        )

    def replace_execution(request, next_call, **_kwargs):
        calls["execution_middleware"] += 1
        replacement = dict(request)
        replacement["messages"] = [{"role": "user", "content": "EXECUTION_REWRITE"}]
        return next_call(replacement)

    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "pre_api_request")
    monkeypatch.setattr(lifecycle, "invoke_hook", invoke_hook)
    monkeypatch.setattr(middleware, "apply_llm_request_middleware", replace_request)
    monkeypatch.setattr(middleware, "run_llm_execution_middleware", replace_execution)


def test_receipt_execution_binds_the_provider_to_raw_admitted_text(
    receipt_agent, monkeypatch
):
    """Receipt turns bypass every mutable enrichment and request seam."""
    from tests.run_agent.test_run_agent import _mock_response

    agent, db, session_id = receipt_agent
    admitted_text = "  admitted raw text  "
    request = request_binding(
        session_id=session_id,
        turn_request_id="raw-text-only-execution",
        text=admitted_text,
        display_kind=None,
        attachments=[],
        truncation=None,
    )
    TurnReceiptAdapter(db).prepare_or_replay(request)
    calls = {"hooks": [], "request_middleware": 0, "execution_middleware": 0, "memory": []}
    _install_mutable_request_seams(monkeypatch, calls)

    class Memory:
        def on_turn_start(self, *args):
            calls["memory"].append(("start", args))

        def prefetch_all(self, *args):
            calls["memory"].append(("prefetch", args))
            return "MEMORY_CONTEXT"

    agent._memory_manager = Memory()
    agent._gateway_turn_context_notes = "GATEWAY_CONTEXT"
    agent.client.chat.completions.create.return_value = _mock_response("done")

    result = agent.run_conversation(admitted_text, turn_receipt=request)

    assert result["completed"] is True
    assert agent.client.chat.completions.create.call_count == 1
    assert _provider_user_content(agent) == admitted_text
    persisted_user = next(
        message for message in result["messages"] if message["role"] == "user"
    )
    assert persisted_user["content"] == admitted_text
    assert "api_content" not in persisted_user
    assert db.get_messages(session_id)[0]["content"] == admitted_text
    assert "pre_llm_call" not in calls["hooks"]
    assert "pre_api_request" not in calls["hooks"]
    assert calls["request_middleware"] == 0
    assert calls["execution_middleware"] == 0
    assert calls["memory"] == []
    assert agent._gateway_turn_context_notes == "GATEWAY_CONTEXT"


def test_ordinary_execution_still_runs_mutable_request_seams(
    receipt_agent, monkeypatch
):
    """The receipt fence does not suppress normal turn enrichment or middleware."""
    from tests.run_agent.test_run_agent import _mock_response

    agent, _db, _session_id = receipt_agent
    calls = {"hooks": [], "request_middleware": 0, "execution_middleware": 0, "memory": []}
    _install_mutable_request_seams(monkeypatch, calls)

    class Memory:
        def on_turn_start(self, *args):
            calls["memory"].append(("start", args))

        def prefetch_all(self, *args):
            calls["memory"].append(("prefetch", args))
            return "MEMORY_CONTEXT"

    agent._memory_manager = Memory()
    agent._gateway_turn_context_notes = "GATEWAY_CONTEXT"
    agent.client.chat.completions.create.return_value = _mock_response("done")

    result = agent.run_conversation("ordinary execution")

    assert result["completed"] is True
    assert agent.client.chat.completions.create.call_count == 1
    assert _provider_user_content(agent) == "EXECUTION_REWRITE"
    assert "pre_llm_call" in calls["hooks"]
    assert "pre_api_request" in calls["hooks"]
    assert calls["request_middleware"] == 1
    assert calls["execution_middleware"] == 1
    assert [entry[0] for entry in calls["memory"]] == ["start", "prefetch"]
    assert agent._gateway_turn_context_notes == ""


def test_receipt_rebinds_raw_text_on_every_tool_loop_provider_pass(
    receipt_agent, monkeypatch
):
    """A tool continuation cannot replace the receipt turn's user bytes."""
    from tests.run_agent.test_run_agent import _mock_response, _mock_tool_call

    agent, db, session_id = receipt_agent
    admitted_text = "  receipt bytes survive every pass  "
    request = request_binding(
        session_id=session_id,
        turn_request_id="raw-text-every-provider-pass",
        text=admitted_text,
        display_kind=None,
        attachments=[],
        truncation=None,
    )
    TurnReceiptAdapter(db).prepare_or_replay(request)
    tool_call = _mock_tool_call("receipt_test_tool", call_id="receipt-loop-tool")
    agent.client.chat.completions.create.side_effect = [
        _mock_response("", finish_reason="tool_calls", tool_calls=[tool_call]),
        _mock_response("done"),
    ]
    monkeypatch.setattr(
        agent,
        "_execute_tool_calls",
        lambda _assistant, messages, *_args: messages.append(
            {
                "role": "tool",
                "name": "receipt_test_tool",
                "tool_call_id": "receipt-loop-tool",
                "content": "tool result",
            }
        ),
    )

    result = agent.run_conversation(admitted_text, turn_receipt=request)

    assert result["completed"] is True
    assert agent.client.chat.completions.create.call_count == 2
    assert [
        next(
            message["content"]
            for message in call.kwargs["messages"]
            if message.get("role") == "user"
        )
        for call in agent.client.chat.completions.create.call_args_list
    ] == [admitted_text, admitted_text]
    assert result["final_response"] == "done"
    assert all(
        "[This response was interrupted by a user correction.]"
        not in str(message.get("content"))
        for message in result["messages"]
    )


@pytest.mark.parametrize("state", ["COMPLETED", "CLAIMED"])
def test_aiagent_claims_after_durable_lease_and_races_return_before_execution(
    receipt_agent, monkeypatch, state
):
    """The actual AIAgent entry point fences races before relay/model/tool work."""
    agent, db, session_id = receipt_agent
    from agent import relay_runtime

    request = _request(session_id)
    receipts = TurnReceiptAdapter(db)
    receipts.prepare_or_replay(request)
    claimed = receipts.claim_after_lease(request)
    assistant = "already durable"
    if state == "COMPLETED":
        receipts.finish(
            session_id,
            request.turn_request_id,
            request.binding_digest,
            claimed.claim_token,
            assistant_content=assistant,
            response_digest="sha256:" + hashlib.sha256(assistant.encode()).hexdigest(),
        )

    events: list[str] = []
    real_acquire = db.acquire_session_turn_lease
    real_claim = TurnReceiptAdapter.claim_after_lease

    def acquire(*args, **kwargs):
        events.append("lease")
        return real_acquire(*args, **kwargs)

    def claim_after_lease(adapter, request_arg):
        events.append("claim")
        return real_claim(adapter, request_arg)

    monkeypatch.setattr(db, "acquire_session_turn_lease", acquire)
    monkeypatch.setattr(TurnReceiptAdapter, "claim_after_lease", claim_after_lease)
    monkeypatch.setattr(
        relay_runtime.SESSION_COORDINATOR,
        "acquire_conversation",
        lambda *_args, **_kwargs: pytest.fail("relay opened after receipt race"),
    )
    agent.client.chat.completions.create.side_effect = lambda *_a, **_kw: pytest.fail(
        "model called after receipt race"
    )

    result = agent.run_conversation("question", turn_receipt=request)

    assert events[:2] == ["lease", "claim"]
    assert agent.client.chat.completions.create.call_count == 0
    if state == "COMPLETED":
        assert result["replayed"] is True
        assert result["final_response"] == assistant
    else:
        assert result["in_progress"] is True
        assert result["final_response"] == ""


def test_receipt_codex_app_server_fails_closed_after_lease_before_claim_or_dispatch(
    receipt_agent, monkeypatch
):
    """Receipt v1 never hands a claimed turn to the app-server runtime."""
    agent, db, session_id = receipt_agent
    request = _request(session_id, request_id="app-server-unsupported")
    TurnReceiptAdapter(db).prepare_or_replay(request)
    agent.api_mode = "codex_app_server"
    events: list[str] = []
    real_acquire = db.acquire_session_turn_lease
    real_claim = TurnReceiptAdapter.claim_after_lease

    def acquire(*args, **kwargs):
        events.append("lease")
        return real_acquire(*args, **kwargs)

    def claim(adapter, request_arg):
        events.append("claim")
        return real_claim(adapter, request_arg)

    def app_server_turn(**_kwargs):
        events.append("app_server")
        return {
            "final_response": "unexpected app-server result",
            "messages": [],
            "api_calls": 0,
            "completed": True,
        }

    monkeypatch.setattr(db, "acquire_session_turn_lease", acquire)
    monkeypatch.setattr(TurnReceiptAdapter, "claim_after_lease", claim)
    monkeypatch.setattr(agent, "_run_codex_app_server_turn", app_server_turn)

    result = agent.run_conversation("question", turn_receipt=request)

    assert events == ["lease"]
    assert result["completed"] is False
    assert result["error"] == "turn_receipt_runtime_unsupported"
    assert result["failure_reason"] == "turn_receipt_runtime_unsupported"
    assert "turn_receipt" not in result
    assert TurnReceiptAdapter(db).status_for(request)["status"] == "PREPARED"
    assert db.get_messages(session_id) == []


@pytest.mark.parametrize(
    ("api_mode", "provider", "moa_config"),
    [
        ("codex_responses", "openrouter", None),
        ("anthropic_messages", "openrouter", None),
        ("chat_completions", "moa", None),
        ("chat_completions", "openrouter", {"reference_models": []}),
    ],
)
def test_receipt_closed_profile_rejects_unsupported_runtime_before_claim(
    receipt_agent, monkeypatch, api_mode, provider, moa_config
):
    """Only a direct standard chat-completions runtime may claim a receipt."""
    agent, db, session_id = receipt_agent
    request = _request(session_id, request_id=f"unsupported-{api_mode}-{provider}")
    TurnReceiptAdapter(db).prepare_or_replay(request)
    agent.api_mode = api_mode
    agent.provider = provider
    claims: list[object] = []

    def reject_claim(*_args, **_kwargs):
        claims.append("claim")
        pytest.fail("unsupported receipt runtime claimed before rejection")

    monkeypatch.setattr(TurnReceiptAdapter, "claim_after_lease", reject_claim)
    agent.client.chat.completions.create.side_effect = lambda *_a, **_kw: pytest.fail(
        "unsupported receipt runtime reached the provider"
    )

    result = agent.run_conversation(
        "question", turn_receipt=request, moa_config=moa_config
    )

    assert claims == []
    assert result["completed"] is False
    assert result["failure_reason"] == "turn_receipt_runtime_unsupported"
    assert TurnReceiptAdapter(db).status_for(request)["status"] == "PREPARED"


def test_receipt_closed_profile_skips_title_and_preflight_compression(
    receipt_agent, monkeypatch
):
    """A receipt fails before the provider when its no-rewrite profile overflows."""
    from agent import turn_context
    from tests.run_agent.test_run_agent import _mock_response

    agent, db, session_id = receipt_agent
    request = _request(session_id, request_id="closed-profile-preflight")
    TurnReceiptAdapter(db).prepare_or_replay(request)
    calls = {"title": 0, "compress": 0}
    agent.compression_enabled = True
    agent.context_compressor.threshold_tokens = 1

    def title(*_args, **_kwargs):
        calls["title"] += 1

    def compress(*_args, **_kwargs):
        calls["compress"] += 1
        pytest.fail("receipt started a compression rewrite")

    monkeypatch.setattr(turn_context, "_maybe_title_session_at_turn_start", title)
    monkeypatch.setattr(agent, "_compress_context", compress)
    agent.client.chat.completions.create.return_value = _mock_response("unexpected")

    result = agent.run_conversation("question", turn_receipt=request)

    assert calls == {"title": 0, "compress": 0}
    assert agent.client.chat.completions.create.call_count == 0
    assert result["completed"] is False
    assert TurnReceiptAdapter(db).status_for(request)["status"] == "CLAIMED"


def test_receipt_closed_profile_skips_current_turn_api_sanitizer(
    receipt_agent, monkeypatch
):
    """Receipt execution never enters the mutable API-message sanitizer."""
    from tests.run_agent.test_run_agent import _mock_response

    agent, db, session_id = receipt_agent
    request = _request(session_id, request_id="closed-profile-sanitizer")
    TurnReceiptAdapter(db).prepare_or_replay(request)
    calls = {"sanitize": 0}

    def rewrite_current_user(api_messages):
        calls["sanitize"] += 1
        for message in api_messages:
            if message.get("role") == "user":
                message["content"] = "SANITIZER_REWRITE"
        return api_messages

    monkeypatch.setattr(agent, "_sanitize_api_messages", rewrite_current_user)
    agent.client.chat.completions.create.return_value = _mock_response("done")

    result = agent.run_conversation("question", turn_receipt=request)

    assert calls == {"sanitize": 0}
    assert result["completed"] is True
    assert _provider_user_content(agent) == "question"
    assert TurnReceiptAdapter(db).status_for(request)["status"] == "COMPLETED"


def test_receipt_closed_profile_rejects_sequence_repair_without_mutating_history(
    receipt_agent, monkeypatch
):
    """A receipt refuses, rather than merging, malformed adjacent user turns."""
    from agent import agent_runtime_helpers

    agent, db, session_id = receipt_agent
    request = _request(session_id, request_id="closed-profile-sequence-repair")
    TurnReceiptAdapter(db).prepare_or_replay(request)
    monkeypatch.setattr(
        agent_runtime_helpers,
        "repair_message_sequence_with_cursor",
        lambda *_args, **_kwargs: pytest.fail("receipt mutated message sequence"),
    )
    agent.client.chat.completions.create.side_effect = lambda *_args, **_kwargs: pytest.fail(
        "receipt reached provider despite required sequence repair"
    )

    result = agent.run_conversation(
        "question",
        conversation_history=[{"role": "user", "content": "earlier question"}],
        turn_receipt=request,
    )

    assert result["completed"] is False
    assert TurnReceiptAdapter(db).status_for(request)["status"] == "CLAIMED"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "multipart", "changed"])
def test_receipt_closed_profile_rejects_a_mutated_provider_bound_current_user(
    receipt_agent, monkeypatch, mutation
):
    """The final provider boundary requires one exact, scalar current user row."""
    from tests.run_agent.test_run_agent import _mock_response

    agent, db, session_id = receipt_agent
    request = _request(session_id, request_id=f"closed-profile-wire-{mutation}")
    TurnReceiptAdapter(db).prepare_or_replay(request)
    real_build = agent._build_api_kwargs

    def build_with_mutated_current_user(messages, **kwargs):
        api_kwargs = real_build(messages, **kwargs)
        wire_messages = api_kwargs["messages"]
        current_index = next(
            index
            for index, message in enumerate(wire_messages)
            if message.get("role") == "user" and message.get("content") == "question"
        )
        if mutation == "missing":
            del wire_messages[current_index]
        elif mutation == "duplicate":
            wire_messages.insert(current_index + 1, dict(wire_messages[current_index]))
        elif mutation == "multipart":
            wire_messages[current_index]["content"] = [
                {"type": "text", "text": "question"}
            ]
        else:
            wire_messages[current_index]["content"] = "changed"
        return api_kwargs

    monkeypatch.setattr(agent, "_build_api_kwargs", build_with_mutated_current_user)
    agent.client.chat.completions.create.return_value = _mock_response("unexpected")

    result = agent.run_conversation("question", turn_receipt=request)

    assert agent.client.chat.completions.create.call_count == 0
    assert result["completed"] is False
    assert TurnReceiptAdapter(db).status_for(request)["status"] == "CLAIMED"


def test_receipt_closed_profile_never_retries_unicode_as_ascii_rewrite(
    receipt_agent,
):
    """An encoding error cannot turn a receipt into a rewritten ASCII retry."""
    from tests.run_agent.test_run_agent import _mock_response

    agent, db, session_id = receipt_agent
    request = _request(session_id, request_id="closed-profile-ascii-retry")
    TurnReceiptAdapter(db).prepare_or_replay(request)
    agent.client.chat.completions.create.side_effect = [
        UnicodeEncodeError("ascii", "question", 0, 1, "ordinal not in range"),
        _mock_response("unexpected retry"),
    ]

    result = agent.run_conversation("question", turn_receipt=request)

    assert agent.client.chat.completions.create.call_count == 1
    assert agent._force_ascii_payload is False
    assert result["completed"] is False
    assert TurnReceiptAdapter(db).status_for(request)["status"] == "CLAIMED"


def test_receipt_closed_profile_blocks_unsupported_fallback_before_second_call(
    receipt_agent, monkeypatch
):
    """A fallback mode change cannot escape the final provider-call fence."""
    from tests.run_agent.test_run_agent import _mock_response, _mock_tool_call

    agent, db, session_id = receipt_agent
    request = _request(session_id, request_id="closed-profile-fallback-mode")
    TurnReceiptAdapter(db).prepare_or_replay(request)
    tool_call = _mock_tool_call("receipt_test_tool", call_id="receipt-fallback-tool")

    def first_response(*_args, **_kwargs):
        agent.api_mode = "codex_responses"
        return _mock_response("", finish_reason="tool_calls", tool_calls=[tool_call])

    agent.client.chat.completions.create.side_effect = first_response
    monkeypatch.setattr(
        agent,
        "_execute_tool_calls",
        lambda _assistant, messages, *_args: messages.append(
            {
                "role": "tool",
                "name": "receipt_test_tool",
                "tool_call_id": "receipt-fallback-tool",
                "content": "tool result",
            }
        ),
    )

    result = agent.run_conversation("question", turn_receipt=request)

    assert agent.client.chat.completions.create.call_count == 1
    assert result["completed"] is False
    assert TurnReceiptAdapter(db).status_for(request)["status"] == "CLAIMED"


def test_ordinary_execution_retains_title_compression_and_repair_paths(
    receipt_agent, monkeypatch
):
    """The closed receipt fence leaves ordinary turn shaping untouched."""
    from agent import agent_runtime_helpers, turn_context
    from tests.run_agent.test_run_agent import _mock_response

    agent, _db, _session_id = receipt_agent
    calls = {"title": 0, "compress": 0, "repair": 0, "sanitize": 0}
    agent.compression_enabled = True
    agent.context_compressor.threshold_tokens = 1
    real_repair = agent_runtime_helpers.repair_message_sequence_with_cursor
    real_sanitize = agent._sanitize_api_messages

    monkeypatch.setattr(
        turn_context,
        "_maybe_title_session_at_turn_start",
        lambda *_args, **_kwargs: calls.__setitem__("title", calls["title"] + 1),
    )
    monkeypatch.setattr(
        agent,
        "_compress_context",
        lambda messages, *_args, **_kwargs: (
            calls.__setitem__("compress", calls["compress"] + 1) or messages,
            agent._cached_system_prompt,
        ),
    )
    monkeypatch.setattr(
        agent_runtime_helpers,
        "repair_message_sequence_with_cursor",
        lambda *args, **kwargs: (
            calls.__setitem__("repair", calls["repair"] + 1)
            or real_repair(*args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        agent,
        "_sanitize_api_messages",
        lambda api_messages: (
            calls.__setitem__("sanitize", calls["sanitize"] + 1)
            or real_sanitize(api_messages)
        ),
    )
    agent.client.chat.completions.create.return_value = _mock_response("done")

    result = agent.run_conversation("ordinary execution")

    assert result["completed"] is True
    assert calls["title"] == 1
    assert calls["compress"] >= 1
    assert calls["repair"] >= 1
    assert calls["sanitize"] >= 1


def test_ordinary_execution_retains_ascii_recovery_and_virtual_moa(
    receipt_agent, monkeypatch
):
    """Ordinary retries and virtual MoA remain available outside receipt mode."""
    from agent import moa_loop
    from tests.run_agent.test_run_agent import _mock_response

    agent, _db, _session_id = receipt_agent
    agent.client.chat.completions.create.side_effect = [
        UnicodeEncodeError("ascii", "ordinary", 0, 1, "ordinal not in range"),
        _mock_response("ascii recovered"),
    ]

    recovered = agent.run_conversation("ordinary")

    assert recovered["completed"] is True
    assert agent.client.chat.completions.create.call_count == 2
    assert agent._force_ascii_payload is True

    moa_calls: list[object] = []
    monkeypatch.setattr(
        moa_loop,
        "aggregate_moa_context",
        lambda **_kwargs: moa_calls.append("aggregate") or "",
    )
    agent._force_ascii_payload = False
    agent.client.chat.completions.create.reset_mock()
    agent.client.chat.completions.create.side_effect = None
    agent.client.chat.completions.create.return_value = _mock_response("moa done")

    virtual_moa = agent.run_conversation(
        "ordinary moa",
        moa_config={"reference_models": [], "aggregator": {}},
    )

    assert virtual_moa["completed"] is True
    assert moa_calls == ["aggregate"]
    assert agent.client.chat.completions.create.call_count == 1


def test_genuine_terminal_response_is_sanitized_and_atomically_completes(
    receipt_agent, monkeypatch
):
    """Only the later visible non-tool assistant response owns completion."""
    agent, db, session_id = receipt_agent
    from tests.run_agent.test_run_agent import _mock_response, _mock_tool_call
    import agent.verify_hooks as verify_hooks
    import agent.verification_stop as verification_stop
    import hermes_cli.lifecycle as lifecycle
    import hermes_cli.plugins as plugins

    request = _request(session_id)
    TurnReceiptAdapter(db).prepare_or_replay(request)
    tool_call = _mock_tool_call("receipt_test_tool", call_id="tool-1")
    agent.client.chat.completions.create.side_effect = [
        _mock_response("tool narration", finish_reason="tool_calls", tool_calls=[tool_call]),
        _mock_response("verify candidate", finish_reason="stop"),
        _mock_response("pre-verify candidate", finish_reason="stop"),
        _mock_response("\ud800genuine terminal", finish_reason="stop"),
    ]

    def execute_tool(_assistant_message, messages, *_args):
        agent._turn_file_mutation_paths.add("changed.py")
        messages.append({
            "role": "tool",
            "name": "receipt_test_tool",
            "tool_call_id": "tool-1",
            "content": "tool result",
        })

    monkeypatch.setattr(agent, "_execute_tool_calls", execute_tool)
    monkeypatch.setattr(verification_stop, "verify_on_stop_enabled", lambda: True)
    monkeypatch.setattr(
        verification_stop,
        "build_verify_on_stop_nudge",
        lambda **_kwargs: "verify again" if not getattr(agent, "_verify_sent", False) else None,
    )
    real_emit_interim = agent._emit_interim_assistant_message

    def emit_interim(message):
        if message.get("content") == "verify candidate":
            agent._verify_sent = True
        return real_emit_interim(message)

    monkeypatch.setattr(agent, "_emit_interim_assistant_message", emit_interim)
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "pre_verify")
    monkeypatch.setattr(verify_hooks, "max_verify_nudges", lambda: 1)
    pre_verify = iter(["verify hook again", None])
    monkeypatch.setattr(
        plugins,
        "get_pre_verify_continue_message",
        lambda **_kw: next(pre_verify),
    )
    hooks: list[str] = []
    real_invoke = lifecycle.invoke_hook

    def invoke_hook(name, *args, **kwargs):
        hooks.append(name)
        return real_invoke(name, *args, **kwargs)

    monkeypatch.setattr(lifecycle, "invoke_hook", invoke_hook)
    events: list[str] = []
    real_acquire = db.acquire_session_turn_lease
    real_claim = TurnReceiptAdapter.claim_after_lease
    monkeypatch.setattr(
        db,
        "acquire_session_turn_lease",
        lambda *args, **kwargs: events.append("lease") or real_acquire(*args, **kwargs),
    )
    monkeypatch.setattr(
        TurnReceiptAdapter,
        "claim_after_lease",
        lambda adapter, request_arg: events.append("claim") or real_claim(adapter, request_arg),
    )

    result = agent.run_conversation("question", turn_receipt=request)

    assert events[:2] == ["lease", "claim"]
    assert agent.client.chat.completions.create.call_count == 4
    assert agent._pre_verify_nudges == 1
    assert result["final_response"] == "\ufffdgenuine terminal"
    receipt = TurnReceiptAdapter(db).status_for(request)
    replay = TurnReceiptAdapter(db).completed_replay(request)
    assert receipt["status"] == "COMPLETED"
    assert replay["assistantContent"] == result["final_response"]
    assert receipt["responseDigest"] == "sha256:" + hashlib.sha256(
        result["final_response"].encode("utf-8")
    ).hexdigest()
    rows = db.get_messages(session_id)
    terminal = next(row for row in rows if row["id"] == receipt["terminalMessageId"])
    assert terminal["content"] == result["final_response"]
    assert sum(row["id"] == receipt["terminalMessageId"] for row in rows) == 1
    assert any(row.get("content") == "tool narration" and row.get("tool_calls") for row in rows)
    assert any(row.get("content") == "verify candidate" for row in rows)
    assert any(row.get("content") == "pre-verify candidate" for row in rows)
    assert all("\ud800" not in str(row.get("content")) for row in rows)
    assert "post_llm_call" in hooks


def test_claimed_receipt_interrupted_before_terminal_fails_closed(
    receipt_agent, monkeypatch
):
    """The real loop must carry a claim to finalization even without a hold."""
    import agent.conversation_loop as conversation_loop
    import hermes_cli.lifecycle as lifecycle

    agent, db, session_id = receipt_agent
    request = _request(session_id, request_id="interrupted-before-terminal")
    receipts = TurnReceiptAdapter(db)
    receipts.prepare_or_replay(request)
    real_claim = TurnReceiptAdapter.claim_after_lease

    def claim_then_interrupt(adapter, request_arg):
        claimed = real_claim(adapter, request_arg)
        agent._interrupt_requested = True
        return claimed

    monkeypatch.setattr(
        TurnReceiptAdapter, "claim_after_lease", claim_then_interrupt
    )
    hooks: list[str] = []
    monkeypatch.setattr(
        lifecycle,
        "invoke_hook",
        lambda name, *_args, **_kwargs: hooks.append(name) or [],
    )
    calls = {"memory": 0, "background": 0, "context": 0}
    monkeypatch.setattr(
        agent,
        "_sync_external_memory_for_turn",
        lambda **_kwargs: calls.__setitem__("memory", calls["memory"] + 1),
    )
    monkeypatch.setattr(
        agent,
        "_spawn_background_review",
        lambda **_kwargs: calls.__setitem__("background", calls["background"] + 1),
    )
    monkeypatch.setattr(
        conversation_loop,
        "_notify_context_engine_turn_complete",
        lambda *_args, **_kwargs: calls.__setitem__("context", calls["context"] + 1),
    )

    result = agent.run_conversation("question", turn_receipt=request)

    assert result["turn_exit_reason"] == "session_persistence_failed"
    assert result["failed"] is True
    assert TurnReceiptAdapter(db).status_for(request)["status"] == "CLAIMED"
    assert all(row.get("role") != "assistant" for row in db.get_messages(session_id))
    assert calls == {"memory": 0, "background": 0, "context": 0}
    assert "post_llm_call" not in hooks
    assert "on_session_end" not in hooks


def test_receipt_flush_binds_the_held_batch_row_not_a_later_assistant(
    receipt_agent,
):
    """The receipt index, rather than batch order, selects the terminal row."""
    from tui_gateway.turn_receipts import TerminalReceiptHold

    agent, db, session_id = receipt_agent
    request = _request(session_id, request_id="exact-held-batch-row")
    receipts = TurnReceiptAdapter(db)
    receipts.prepare_or_replay(request)
    claimed = receipts.claim_after_lease(request)
    held = {"role": "assistant", "content": "held terminal"}
    later = {"role": "assistant", "content": "later assistant"}
    messages = [{"role": "user", "content": "question"}, held, later]
    hold = TerminalReceiptHold(
        claimed,
        held,
        1,
        "sha256:" + hashlib.sha256(held["content"].encode()).hexdigest(),
    )

    assert agent._flush_messages_to_session_db(
        messages, [], terminal_receipt_hold=hold
    ) is True

    completed = receipts.status_for(request)
    rows = db.get_messages(session_id)
    held_row = rows[1]
    assert completed["status"] == "COMPLETED"
    assert completed["terminalMessageId"] == held_row["id"]
    assert held_row["content"] == held["content"]
    assert completed["terminalMessageId"] != rows[2]["id"]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "moved", "hidden", "marked"])
def test_receipt_flush_rejects_nonexact_held_mapping_before_any_write(
    receipt_agent, failure
):
    """A receipt cannot settle through a missing, stale, hidden, or duplicate hold."""
    from tui_gateway.turn_receipts import TerminalReceiptHold

    agent, db, session_id = receipt_agent
    request = _request(session_id, request_id=f"invalid-held-{failure}")
    receipts = TurnReceiptAdapter(db)
    receipts.prepare_or_replay(request)
    claimed = receipts.claim_after_lease(request)
    held = {"role": "assistant", "content": "held terminal"}
    messages = [{"role": "user", "content": "question"}, held]
    hold_index = 1
    if failure == "missing":
        hold = {"role": "assistant", "content": "missing terminal"}
    elif failure == "duplicate":
        messages.append(held)
        hold = held
    elif failure == "moved":
        messages.insert(1, {"role": "assistant", "content": "replacement"})
        hold = held
    elif failure == "hidden":
        held["display_kind"] = "hidden"
        hold = held
    else:
        held["_db_persisted"] = True
        hold = held
    terminal_hold = TerminalReceiptHold(
        claimed,
        hold,
        hold_index,
        "sha256:" + hashlib.sha256(hold["content"].encode()).hexdigest(),
    )

    with pytest.raises(RuntimeError, match="terminal turn receipt hold is not persistable"):
        agent._flush_messages_to_session_db(
            messages, [], terminal_receipt_hold=terminal_hold
        )

    assert db.get_messages(session_id) == []
    assert receipts.status_for(request)["status"] == "CLAIMED"


def test_compression_retry_preserves_receipt_context_and_never_generic_persists(
    receipt_agent, monkeypatch
):
    """A live-tip retry must carry the exact receipt or leave its held row absent."""
    from tui_gateway.turn_receipts import TerminalReceiptHold

    agent, db, session_id = receipt_agent
    request = _request(session_id, request_id="compression-held-retry")
    receipts = TurnReceiptAdapter(db)
    receipts.prepare_or_replay(request)
    claimed = receipts.claim_after_lease(request)
    held = {"role": "assistant", "content": "held after compression"}
    hold = TerminalReceiptHold(
        claimed,
        held,
        1,
        "sha256:" + hashlib.sha256(held["content"].encode()).hexdigest(),
    )
    messages = [{"role": "user", "content": "question"}, held]
    live_tip = "receipt-live-compression-tip"
    db.end_session(session_id, "compression")
    db.create_session(live_tip, source="test", parent_session_id=session_id)

    calls = []
    real_append = db.append_messages_batch

    def observed_append(*args, **kwargs):
        calls.append((kwargs["session_id"], kwargs.get("terminal_turn_receipt")))
        return real_append(*args, **kwargs)

    monkeypatch.setattr(db, "append_messages_batch", observed_append)

    assert agent._flush_messages_to_session_db(
        messages, [], terminal_receipt_hold=hold
    ) is False
    assert [session for session, _receipt in calls] == [session_id, live_tip]
    first_receipt = calls[0][1]
    second_receipt = calls[1][1]
    assert first_receipt is not None
    assert second_receipt is not None
    assert (
        first_receipt.session_id,
        first_receipt.turn_request_id,
        first_receipt.binding_digest,
        first_receipt.claim_token,
        first_receipt.response_digest,
        first_receipt.terminal_message_index,
    ) == (
        second_receipt.session_id,
        second_receipt.turn_request_id,
        second_receipt.binding_digest,
        second_receipt.claim_token,
        second_receipt.response_digest,
        second_receipt.terminal_message_index,
    )
    assert first_receipt.terminal_message_index == 1
    assert db.get_messages(session_id) == []
    assert db.get_messages(live_tip) == []
    assert receipts.status_for(request)["status"] == "CLAIMED"


@pytest.mark.parametrize(
    "failure", ["missing_hold", "flush_false", "flush_error", "replaced", "marked"]
)
def test_failed_terminal_settlement_never_completes_or_runs_post_success_work(
    receipt_agent, monkeypatch, failure
):
    """A failed terminal hold is fail-closed before memory/background/hooks."""
    from agent.turn_finalizer import finalize_turn
    from tui_gateway.turn_receipts import TerminalReceiptHold
    import hermes_cli.lifecycle as lifecycle

    agent, db, session_id = receipt_agent
    request = _request(session_id, request_id=f"{failure}-request")
    receipts = TurnReceiptAdapter(db)
    receipts.prepare_or_replay(request)
    claimed = receipts.claim_after_lease(request)
    terminal = {"role": "assistant", "content": "\ud800raw final"}
    messages = [{"role": "user", "content": "question"}, terminal]
    hold = TerminalReceiptHold(claimed, terminal, 1)
    if failure == "replaced":
        messages[1] = {"role": "assistant", "content": "replacement"}
    elif failure == "marked":
        terminal["_db_persisted"] = True
    elif failure == "flush_false":
        monkeypatch.setattr(agent, "_persist_session", lambda *_a, **_kw: False)
    elif failure == "flush_error":
        def raise_flush(*_args, **_kwargs):
            raise RuntimeError("flush exploded")

        monkeypatch.setattr(agent, "_persist_session", raise_flush)
    elif failure == "missing_hold":
        messages = [{"role": "user", "content": "question"}]
        hold = None

    hooks: list[str] = []
    monkeypatch.setattr(
        lifecycle,
        "invoke_hook",
        lambda name, *_args, **_kwargs: hooks.append(name) or [],
    )
    calls = {"memory": 0, "background": 0}
    monkeypatch.setattr(
        agent,
        "_sync_external_memory_for_turn",
        lambda **_kw: calls.__setitem__("memory", calls["memory"] + 1),
    )
    monkeypatch.setattr(
        agent,
        "_spawn_background_review",
        lambda **_kw: calls.__setitem__("background", calls["background"] + 1),
    )

    result = finalize_turn(
        agent,
        final_response="\ud800raw final",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="question",
        original_user_message="question",
        _should_review_memory=True,
        _turn_exit_reason="text_response(finish_reason=stop)",
        terminal_receipt_hold=hold,
        claimed_receipt=claimed,
    )

    assert result["turn_exit_reason"] == "session_persistence_failed"
    assert result["failed"] is True
    assert TurnReceiptAdapter(db).status_for(request)["status"] == "CLAIMED"
    assert db.get_messages(session_id) == []
    assert calls == {"memory": 0, "background": 0}
    assert "post_llm_call" not in hooks
    assert "on_session_end" not in hooks


def test_non_string_held_receipt_content_fails_before_persistence_or_completion(
    receipt_agent, monkeypatch
):
    """Receipt settlement must not replace a non-text held assistant payload."""
    from agent.turn_finalizer import finalize_turn
    from tui_gateway.turn_receipts import TerminalReceiptHold
    import hermes_cli.lifecycle as lifecycle

    agent, db, session_id = receipt_agent
    request = _request(session_id, request_id="non-string-held-content")
    receipts = TurnReceiptAdapter(db)
    receipts.prepare_or_replay(request)
    claimed = receipts.claim_after_lease(request)
    original_content = {"unexpected": "held assistant object"}
    held = {"role": "assistant", "content": original_content}
    messages = [{"role": "user", "content": "question"}, held]
    hold = TerminalReceiptHold(claimed, held, 1)
    persist_calls: list[object] = []
    hooks: list[str] = []

    def persist(*_args, **_kwargs):
        persist_calls.append("persist")
        return True

    monkeypatch.setattr(agent, "_persist_session", persist)
    monkeypatch.setattr(
        lifecycle,
        "invoke_hook",
        lambda name, *_args, **_kwargs: hooks.append(name) or [],
    )

    result = finalize_turn(
        agent,
        final_response="text response",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="question",
        original_user_message="question",
        _should_review_memory=True,
        _turn_exit_reason="text_response(finish_reason=stop)",
        terminal_receipt_hold=hold,
        claimed_receipt=claimed,
    )

    assert held["content"] is original_content
    assert persist_calls == []
    assert result["turn_exit_reason"] == "session_persistence_failed"
    assert result["failed"] is True
    assert result["completed"] is False
    assert "turn_receipt" not in result
    assert receipts.status_for(request)["status"] == "CLAIMED"
    assert db.get_messages(session_id) == []
    assert "post_llm_call" not in hooks
    assert "on_session_end" not in hooks
