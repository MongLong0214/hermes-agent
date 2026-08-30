"""Tests for acp_adapter.server — HermesACPAgent ACP server."""

import asyncio
import hashlib
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

import acp
from acp.agent.router import build_agent_router
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AuthenticateResponse,
    AvailableCommandsUpdate,
    Implementation,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    ResumeSessionResponse,
    SessionModelState,
    SessionModeState,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
    SetSessionModeResponse,
    SessionInfo,
    SessionInfoUpdate,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
    UserMessageChunk,
)
from acp_adapter.auth import TERMINAL_SETUP_AUTH_METHOD_ID
from acp_adapter.server import (
    ACP_MAX_MODELS_PER_PROVIDER,
    HermesACPAgent,
    HERMES_VERSION,
)
from acp_adapter.session import SessionManager
from hermes_state import SessionDB
from tui_gateway.turn_receipts import ClaimedReceipt, ReceiptRequest, TurnReceiptAdapter


@pytest.fixture()
def mock_manager():
    """SessionManager with a mock agent factory."""
    return SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"))


@pytest.fixture()
def agent(mock_manager):
    """HermesACPAgent backed by a mock session manager."""
    return HermesACPAgent(session_manager=mock_manager)


@pytest.mark.asyncio
async def test_new_session_exposes_edit_approvals_as_modes_not_config_options(agent):
    resp = await agent.new_session(cwd="/tmp")

    assert resp.config_options is None
    assert isinstance(resp.modes, SessionModeState)
    assert resp.modes.current_mode_id == "default"
    assert [(mode.id, mode.name) for mode in resp.modes.available_modes] == [
        ("default", "Default"),
        ("accept_edits", "Accept Edits"),
        ("dont_ask", "Don't Ask"),
    ]


@pytest.mark.asyncio
async def test_set_config_option_persists_edit_approval_policy_without_advertising_config(agent):
    resp = await agent.new_session(cwd="/tmp")
    update = await agent.set_config_option(
        "edit_approval_policy",
        resp.session_id,
        "workspace_session",
    )
    state = agent.session_manager.get_session(resp.session_id)

    assert isinstance(update, SetSessionConfigOptionResponse)
    assert update.config_options == []
    assert getattr(state, "mode", None) == "accept_edits"


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_returns_correct_protocol_version(self, agent):
        resp = await agent.initialize(protocol_version=1)
        assert isinstance(resp, InitializeResponse)
        assert resp.protocol_version == acp.PROTOCOL_VERSION




    @pytest.mark.asyncio
    async def test_initialize_advertises_provider_and_terminal_auth_methods(self, agent, monkeypatch):
        monkeypatch.setattr("acp_adapter.auth.detect_provider", lambda: "openrouter")
        monkeypatch.setattr("acp_adapter.server.detect_provider", lambda: "openrouter")

        resp = await agent.initialize(protocol_version=1)
        payloads = [method.model_dump(by_alias=True, exclude_none=True) for method in resp.auth_methods]

        assert payloads[0]["id"] == "openrouter"
        assert payloads[0]["name"] == "openrouter runtime credentials"
        terminal = next(payload for payload in payloads if payload["id"] == TERMINAL_SETUP_AUTH_METHOD_ID)
        assert terminal["type"] == "terminal"
        assert terminal["args"] == ["--setup"]



# ---------------------------------------------------------------------------
# authenticate
# ---------------------------------------------------------------------------


class TestAuthenticate:
    @pytest.mark.asyncio
    async def test_authenticate_with_matching_method_id(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="openrouter")
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_is_case_insensitive(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="OpenRouter")
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_rejects_mismatched_method_id(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="totally-invalid-method")
        assert resp is None

    @pytest.mark.asyncio
    async def test_authenticate_without_provider(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: None,
        )
        resp = await agent.authenticate(method_id="openrouter")
        assert resp is None

    @pytest.mark.asyncio
    async def test_authenticate_accepts_terminal_setup_after_provider_configured(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id=TERMINAL_SETUP_AUTH_METHOD_ID)
        assert isinstance(resp, AuthenticateResponse)



# ---------------------------------------------------------------------------
# new_session / cancel / load / resume
# ---------------------------------------------------------------------------


class TestSessionOps:

    @pytest.mark.asyncio
    async def test_new_session_returns_authenticated_cross_provider_model_state(self):
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(
                model="gpt-5.4",
                provider="openai-codex",
                base_url="https://api.openai.com/v1",
            )
        )
        acp_agent = HermesACPAgent(session_manager=manager)
        picker_context = MagicMock()
        picker_context.with_overrides.return_value = picker_context
        payload = {
            "providers": [
                {
                    "slug": "anthropic",
                    "name": "Anthropic",
                    "models": ["claude-sonnet-4-6", "claude-sonnet-4-6"],
                },
                {
                    "slug": "openai-codex",
                    "name": "OpenAI Codex",
                    "models": [
                        {"id": "gpt-5.4"},
                        "gpt-5.4-mini",
                    ],
                },
            ],
        }

        with (
            patch("hermes_cli.inventory.load_picker_context", return_value=picker_context),
            patch("hermes_cli.inventory.build_models_payload", return_value=payload) as build_payload,
        ):
            resp = await acp_agent.new_session(cwd="/tmp")

        assert isinstance(resp.models, SessionModelState)
        assert resp.models.current_model_id == "openai-codex:gpt-5.4"
        assert [model.model_id for model in resp.models.available_models] == [
            "anthropic:claude-sonnet-4-6",
            "openai-codex:gpt-5.4",
            "openai-codex:gpt-5.4-mini",
        ]
        assert [model.name for model in resp.models.available_models] == [
            "Anthropic · claude-sonnet-4-6",
            "OpenAI Codex · gpt-5.4",
            "OpenAI Codex · gpt-5.4-mini",
        ]
        assert resp.models.available_models[1].description is not None
        assert "current" in resp.models.available_models[1].description
        picker_context.with_overrides.assert_called_once_with(
            current_provider="openai-codex",
            current_model="gpt-5.4",
            current_base_url="https://api.openai.com/v1",
        )
        build_payload.assert_called_once_with(
            picker_context,
            explicit_only=True,
            include_unconfigured=False,
            picker_hints=False,
            canonical_order=True,
            pricing=False,
            capabilities=False,
            refresh=False,
            probe_custom_providers=False,
            probe_current_custom_provider=False,
            max_models=ACP_MAX_MODELS_PER_PROVIDER,
        )



    @pytest.mark.asyncio
    async def test_available_commands_include_help(self, agent):
        help_cmd = next(
            (cmd for cmd in agent._available_commands() if cmd.name == "help"),
            None,
        )

        assert help_cmd is not None
        assert help_cmd.description == "List available commands"
        assert help_cmd.input is None


    def test_build_usage_update_for_zed_context_indicator(self, agent, mock_manager):
        state = mock_manager.create_session(cwd="/tmp")
        state.history = [{"role": "user", "content": "hello"}]
        state.agent.context_compressor = MagicMock(context_length=100_000)
        state.agent._cached_system_prompt = "system"
        state.agent.tools = [{"type": "function", "function": {"name": "demo"}}]

        with patch(
            "agent.model_metadata.estimate_request_tokens_rough",
            return_value=25_000,
        ):
            update = agent._build_usage_update(state)

        assert isinstance(update, UsageUpdate)
        assert update.session_update == "usage_update"
        assert update.size == 100_000
        assert update.used == 25_000




    @pytest.mark.asyncio
    async def test_load_session_not_found_returns_none(self, agent):
        resp = await agent.load_session(cwd="/tmp", session_id="bogus")
        assert resp is None






    @pytest.mark.asyncio
    async def test_resume_session_replays_persisted_history_to_client(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [{"role": "user", "content": "So tell me the current state"}]

        mock_conn.session_update.reset_mock()
        resp = await agent.resume_session(cwd="/tmp", session_id=new_resp.session_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert isinstance(resp, ResumeSessionResponse)
        updates = [call.kwargs["update"] for call in mock_conn.session_update.await_args_list]
        assert any(
            isinstance(update, UserMessageChunk)
            and update.content.text == "So tell me the current state"
            for update in updates
        )











# ---------------------------------------------------------------------------
# list / fork
# ---------------------------------------------------------------------------


class TestListAndFork:
    @pytest.mark.asyncio
    async def test_fork_session(self, agent):
        new_resp = await agent.new_session(cwd="/original")
        fork_resp = await agent.fork_session(cwd="/forked", session_id=new_resp.session_id)
        assert fork_resp.session_id
        assert fork_resp.session_id != new_resp.session_id

    @pytest.mark.asyncio
    async def test_list_sessions_includes_title_and_updated_at(self, agent):
        with patch.object(
            agent.session_manager,
            "list_sessions",
            return_value=[
                {
                    "session_id": "session-1",
                    "cwd": "/tmp/project",
                    "title": "Fix Zed session history",
                    "updated_at": 123.0,
                }
            ],
        ):
            resp = await agent.list_sessions(cwd="/tmp/project")

        assert isinstance(resp.sessions[0], SessionInfo)
        assert resp.sessions[0].title == "Fix Zed session history"
        assert resp.sessions[0].updated_at == "123.0"






# ---------------------------------------------------------------------------
# session configuration / model routing
# ---------------------------------------------------------------------------


class TestSessionConfiguration:

    @pytest.mark.asyncio
    async def test_router_accepts_stable_session_config_methods(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")
        router = build_agent_router(agent)

        mode_result = await router(
            "session/set_mode",
            {"modeId": "accept_edits", "sessionId": new_resp.session_id},
            False,
        )
        config_result = await router(
            "session/set_config_option",
            {
                "configId": "approval_mode",
                "sessionId": new_resp.session_id,
                "value": "auto",
            },
            False,
        )

        assert mode_result == {}
        assert config_result["configOptions"] == []





# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


class TestPrompt:
    @pytest.mark.asyncio
    async def test_prompt_returns_refusal_for_unknown_session(self, agent):
        prompt = [TextContentBlock(type="text", text="hello")]
        resp = await agent.prompt(prompt=prompt, session_id="nonexistent")
        assert isinstance(resp, PromptResponse)
        assert resp.stop_reason == "refusal"

    @pytest.mark.asyncio
    async def test_prompt_binds_session_id_into_subprocess_env(self, agent, mock_manager):
        """The ACP prompt path must bridge the session id into child subprocesses.

        Regression: ``set_session_vars`` was called with ``session_key`` only,
        leaving the ``HERMES_SESSION_ID`` ContextVar bound to the explicit ""
        default. Once the session-context machinery is engaged, that empty value
        is authoritative — so ``_make_run_env`` handed child subprocesses an
        empty ``HERMES_SESSION_ID`` instead of the session's own id.
        """
        from tools.environments.local import _make_run_env

        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)

        captured: dict[str, str | None] = {}

        def _run(*args, **kwargs):
            # Runs inside the session context copy set up by prompt().
            captured["child"] = _make_run_env({}).get("HERMES_SESSION_ID")
            return {"final_response": "ok", "messages": []}

        state.agent.run_conversation = _run
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent.prompt(
            prompt=[TextContentBlock(type="text", text="hi")],
            session_id=resp.session_id,
        )

        assert captured.get("child") == resp.session_id


class TestAcpTerminalReceipt:
    @staticmethod
    def _receipt_metadata(
        session_id: str, raw_text: str, *, operation: str = "execute"
    ) -> dict:
        prompt_digest = "sha256:" + hashlib.sha256(
            json.dumps(raw_text, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "operation": operation,
            "receiptIdentity": {
                "schema": "hermes.acp-terminal-receipt-identity",
                "version": 1,
                "turnRequestId": "turn-request-1",
                "targetActorId": "target-actor",
                "promptDigest": prompt_digest,
                "bindingGeneration": 1,
                "targetBindingId": "target-binding",
                "targetAttestationId": "target-attestation",
                "executorSessionId": "executor-session",
                "executorSessionIncarnation": "executor-incarnation",
            },
            "targetBindReceipt": {"requested_session_id": session_id},
        }

    @staticmethod
    def _receipt_row(session_id: str, status: str) -> dict:
        return {
            "status": status,
            "sessionId": session_id,
            "turnRequestId": "turn-request-1",
            "receiptIdentityDigest": "sha256:" + "a" * 64,
        }

    @staticmethod
    def _admission(metadata: dict) -> dict:
        return {
            "receiptIdentity": metadata["receiptIdentity"],
            "receiptIdentityDigest": "sha256:" + "a" * 64,
            "targetBindReceipt": metadata["targetBindReceipt"],
            "targetBindReceiptDigest": "sha256:" + "c" * 64,
        }

    @classmethod
    def _durable_metadata(cls, db: SessionDB, session_id: str, raw_text: str) -> dict:
        metadata = cls._receipt_metadata(session_id, raw_text)
        record = db.prepare_target_bind_receipt(
            session_id, "target-actor", 1, "executor-runtime"
        )
        metadata["targetBindReceipt"] = {
            key: record[key]
            for key in (
                "schema", "domain", "version", "actor_id", "binding_generation",
                "executor_runtime_identity", "requested_session_id", "lineage_root_digest",
                "receipt_digest",
            )
        }
        return metadata

    @staticmethod
    def _terminal_meta(response: PromptResponse) -> dict:
        return response.field_meta["hermes"]["acpTerminalReceipt"]

    @pytest.mark.anyio
    async def test_terminal_receipt_prompt_restores_exact_persisted_non_acp_session(
        self, tmp_path
    ):
        """A validated receipt may attach only to its durable canonical target."""
        db = SessionDB(tmp_path / "canonical.db")
        session_id = "canonical-telegram-session"
        raw_text = "continue the canonical thread"
        history = [
            {"role": "user", "content": "canonical opening"},
            {"role": "assistant", "content": "canonical reply"},
        ]
        captured: dict[str, object] = {}
        created_agents: list[object] = []

        class CanonicalAgent:
            def __init__(self):
                created_agents.append(self)
                self._session_db = db
                self._session_db_created = True
                self.session_id = session_id
                self.model = "canonical-model"
                self.provider = "test"

            def run_conversation(self, **kwargs):
                captured.update(kwargs)
                receipt = kwargs["turn_receipt"]
                TurnReceiptAdapter(db).finish(
                    session_id,
                    receipt.request.turn_request_id,
                    receipt.request.binding_digest,
                    receipt.claim_token,
                    assistant_content="receipt-authorized reply",
                    response_digest="sha256:" + "b" * 64,
                )
                return {"final_response": "transient", "messages": []}

        try:
            db.create_session(
                session_id,
                source="telegram",
                model="canonical-model",
                model_config={"cwd": "/canonical"},
            )
            db.replace_messages(session_id, [dict(message) for message in history])
            metadata = self._durable_metadata(db, session_id, raw_text)
            manager = SessionManager(agent_factory=CanonicalAgent, db=db)
            acp_agent = HermesACPAgent(session_manager=manager)

            response = await acp_agent.prompt(
                prompt=[TextContentBlock(type="text", text=raw_text)],
                session_id=session_id,
                hermes={"acpTerminalReceipt": metadata},
            )

            assert response.stop_reason == "end_turn"
            assert self._terminal_meta(response)["status"] == "COMPLETED"
            assert self._terminal_meta(response)["assistantContent"] == "receipt-authorized reply"
            assert len(created_agents) == 1
            assert created_agents[0].session_id == session_id
            assert db.get_conversation_root(session_id) == session_id
            assert [
                {"role": message["role"], "content": message["content"]}
                for message in captured["conversation_history"]
            ] == history
            assert [row["id"] for row in db.list_sessions_rich(limit=10)] == [session_id]
            assert db.get_session(session_id)["source"] == "telegram"
        finally:
            db.close()

    @pytest.mark.anyio
    async def test_terminal_receipt_restore_does_not_open_non_acp_session_to_ordinary_acp(
        self, tmp_path
    ):
        """The exceptional receipt path must not make a canonical target adoptable."""
        db = SessionDB(tmp_path / "source-fence.db")
        session_id = "canonical-gateway-session"
        raw_text = "receipt-only attach"
        run_calls: list[dict] = []

        class CanonicalAgent:
            def __init__(self):
                self._session_db = db
                self._session_db_created = True
                self.session_id = session_id
                self.model = "canonical-model"
                self.provider = "test"

            def run_conversation(self, **kwargs):
                run_calls.append(kwargs)
                receipt = kwargs.get("turn_receipt")
                if receipt is not None:
                    TurnReceiptAdapter(db).finish(
                        session_id,
                        receipt.request.turn_request_id,
                        receipt.request.binding_digest,
                        receipt.claim_token,
                        assistant_content="authorized",
                        response_digest="sha256:" + "c" * 64,
                    )
                return {"final_response": "transient", "messages": []}

        try:
            db.create_session(session_id, source="gateway", model="canonical-model")
            db.replace_messages(
                session_id,
                [{"role": "user", "content": "persisted canonical history"}],
            )
            metadata = self._durable_metadata(db, session_id, raw_text)
            manager = SessionManager(agent_factory=CanonicalAgent, db=db)
            acp_agent = HermesACPAgent(session_manager=manager)

            authorized = await acp_agent.prompt(
                prompt=[TextContentBlock(type="text", text=raw_text)],
                session_id=session_id,
                hermes={"acpTerminalReceipt": metadata},
            )

            assert authorized.stop_reason == "end_turn"
            assert manager.get_session(session_id) is None
            assert await acp_agent.load_session(cwd="/tmp", session_id=session_id) is None
            resumed = await acp_agent.resume_session(cwd="/tmp", session_id=session_id)
            assert resumed.field_meta["hermes"]["sessionProvenance"]["acpSessionId"] != session_id
            refused = await acp_agent.prompt(
                prompt=[TextContentBlock(type="text", text="ordinary ACP prompt")],
                session_id=session_id,
            )
            assert refused.stop_reason == "refusal"
            assert len(run_calls) == 1
        finally:
            db.close()

    @pytest.mark.anyio
    async def test_terminal_receipt_status_does_not_materialize_non_acp_target(self, tmp_path):
        """Status is a durable receipt lookup, not a conversation restore."""
        db = SessionDB(tmp_path / "status-no-restore.db")
        session_id = "canonical-gateway-status"
        agent_builds: list[None] = []

        def fail_if_built():
            agent_builds.append(None)
            raise AssertionError("terminal status constructed an agent")

        try:
            db.create_session(session_id, source="gateway", model="canonical-model")
            db.replace_messages(session_id, [{"role": "user", "content": "canonical history"}])
            metadata = self._durable_metadata(db, session_id, "status digest")
            metadata["operation"] = "status"
            before_ids = [row["id"] for row in db.list_sessions_rich(limit=10)]
            manager = SessionManager(agent_factory=fail_if_built, db=db)
            acp_agent = HermesACPAgent(session_manager=manager)

            response = await acp_agent.prompt(
                prompt=[],
                session_id=session_id,
                hermes={"acpTerminalReceipt": metadata},
            )

            assert self._terminal_meta(response) == {
                "status": "NEVER_FOUND",
                "turnRequestId": "turn-request-1",
            }
            assert agent_builds == []
            assert manager.get_session(session_id) is None
            assert [row["id"] for row in db.list_sessions_rich(limit=10)] == before_ids
        finally:
            db.close()

    @pytest.mark.anyio
    async def test_terminal_receipt_execute_checks_exact_digest_before_non_acp_restore(self, tmp_path):
        """A digest mismatch cannot materialize or mutate a canonical target."""
        db = SessionDB(tmp_path / "digest-before-restore.db")
        session_id = "canonical-gateway-digest"
        agent_builds: list[None] = []

        def fail_if_built():
            agent_builds.append(None)
            raise AssertionError("bad terminal prompt constructed an agent")

        try:
            db.create_session(session_id, source="gateway", model="canonical-model")
            db.replace_messages(session_id, [{"role": "user", "content": "canonical history"}])
            metadata = self._durable_metadata(db, session_id, "authorized bytes")
            before_ids = [row["id"] for row in db.list_sessions_rich(limit=10)]
            manager = SessionManager(agent_factory=fail_if_built, db=db)
            acp_agent = HermesACPAgent(session_manager=manager)

            response = await acp_agent.prompt(
                prompt=[TextContentBlock(type="text", text="different bytes")],
                session_id=session_id,
                hermes={"acpTerminalReceipt": metadata},
            )

            assert response.stop_reason == "refusal"
            assert self._terminal_meta(response)["status"] == "REFUSED"
            assert agent_builds == []
            assert manager.get_session(session_id) is None
            assert [row["id"] for row in db.list_sessions_rich(limit=10)] == before_ids
            assert db.get_acp_turn_receipt(
                session_id,
                metadata["receiptIdentity"],
                metadata["targetBindReceipt"],
            ) is None
        finally:
            db.close()

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "case",
        ("tampered-target", "identity-mismatch", "requested-session-mismatch", "unknown-target"),
    )
    async def test_unvalidated_terminal_target_refuses_before_agent_build_or_turn_mutation(
        self, tmp_path, case
    ):
        """Invalid terminal target evidence cannot construct a persisted ACP agent."""
        db = SessionDB(tmp_path / f"{case}.db")
        target_session_id = "persisted-acp-target"
        raw_text = "fail closed before restore"
        agent_builds: list[None] = []

        def fail_if_built():
            agent_builds.append(None)
            raise AssertionError("invalid terminal receipt constructed an agent")

        try:
            db.create_session(target_session_id, source="acp", model="test")
            valid_metadata = self._durable_metadata(db, target_session_id, raw_text)
            metadata = valid_metadata
            requested_session_id = target_session_id
            if case == "tampered-target":
                metadata = {
                    **metadata,
                    "targetBindReceipt": {
                        **metadata["targetBindReceipt"],
                        "receipt_digest": "sha256:" + "f" * 64,
                    },
                }
            elif case == "identity-mismatch":
                metadata = {
                    **metadata,
                    "receiptIdentity": {
                        **metadata["receiptIdentity"],
                        "targetActorId": "different-target-actor",
                    },
                }
            elif case == "requested-session-mismatch":
                requested_session_id = "different-session"
            elif case == "unknown-target":
                requested_session_id = "unknown-session"

            acp_agent = HermesACPAgent(
                session_manager=SessionManager(agent_factory=fail_if_built, db=db)
            )
            response = await acp_agent.prompt(
                prompt=[TextContentBlock(type="text", text=raw_text)],
                session_id=requested_session_id,
                hermes={"acpTerminalReceipt": metadata},
            )

            assert response.stop_reason == "refusal"
            assert self._terminal_meta(response)["status"] == "REFUSED"
            assert agent_builds == []
            assert (
                db.get_acp_turn_receipt(
                    target_session_id,
                    valid_metadata["receiptIdentity"],
                    valid_metadata["targetBindReceipt"],
                )
                is None
            )
        finally:
            db.close()

    @pytest.mark.anyio
    async def test_acp_terminal_receipt_execute_forwards_exact_raw_text_and_claim(
        self, agent, mock_manager
    ):
        state = mock_manager.create_session(cwd="/tmp")
        raw_text = "  keep CRLF\r\nand spaces  "
        metadata = self._receipt_metadata(state.session_id, raw_text)
        db = MagicMock()
        prepared = self._receipt_row(state.session_id, "PREPARED")
        completed = self._receipt_row(state.session_id, "COMPLETED")
        db.prepare_acp_turn_receipt.return_value = prepared
        db.get_acp_turn_receipt.return_value = completed
        state.agent._session_db = db
        db.get_conversation_root.return_value = state.session_id
        db.validate_acp_turn_receipt_request.return_value = self._admission(metadata)
        request = ReceiptRequest(
            state.session_id, "turn-request-1", prepared["receiptIdentityDigest"]
        )
        claimed = ClaimedReceipt(request, "claim-token")
        receipt_adapter = MagicMock()
        receipt_adapter.claim_after_lease.return_value = claimed
        receipt_adapter.completed_replay.return_value = {
            **completed,
            "assistantContent": "durable exact response",
        }
        state.agent.run_conversation = MagicMock(
            return_value={"final_response": "ignored transient result", "messages": []}
        )

        with patch("acp_adapter.server.TurnReceiptAdapter", return_value=receipt_adapter):
            response = await agent.prompt(
                prompt=[TextContentBlock(type="text", text=raw_text)],
                session_id=state.session_id,
                hermes={"acpTerminalReceipt": metadata},
            )

        db.prepare_acp_turn_receipt.assert_called_once_with(
            state.session_id, metadata["receiptIdentity"], metadata["targetBindReceipt"]
        )
        receipt_adapter.claim_after_lease.assert_called_once_with(request)
        run_kwargs = state.agent.run_conversation.call_args.kwargs
        assert run_kwargs["user_message"] == raw_text
        assert run_kwargs["persist_user_message"] == raw_text
        assert run_kwargs["turn_receipt"] is claimed
        db.get_acp_turn_receipt.assert_called_once_with(
            state.session_id, metadata["receiptIdentity"], metadata["targetBindReceipt"]
        )
        assert self._terminal_meta(response) == {
            **completed,
            "assistantContent": "durable exact response",
        }

    @pytest.mark.anyio
    async def test_acp_terminal_receipt_whitespace_digest_mismatch_has_no_side_effects(
        self, agent, mock_manager
    ):
        state = mock_manager.create_session(cwd="/tmp")
        db = MagicMock()
        state.agent._session_db = db
        db.get_conversation_root.return_value = state.session_id
        state.agent.run_conversation = MagicMock()
        metadata = self._receipt_metadata(state.session_id, "without trailing whitespace")
        db.validate_acp_turn_receipt_request.return_value = self._admission(metadata)

        response = await agent.prompt(
            prompt=[TextContentBlock(type="text", text="without trailing whitespace ")],
            session_id=state.session_id,
            hermes={"acpTerminalReceipt": metadata},
        )

        assert response.stop_reason == "refusal"
        assert self._terminal_meta(response)["status"] == "REFUSED"
        db.prepare_acp_turn_receipt.assert_not_called()
        db.get_acp_turn_receipt.assert_not_called()
        state.agent.run_conversation.assert_not_called()
        assert state.history == []
        assert state.is_running is False

    @pytest.mark.anyio
    async def test_acp_terminal_receipt_digest_binds_exact_crlf_bytes(
        self, agent, mock_manager
    ):
        state = mock_manager.create_session(cwd="/tmp")
        lf_text = "line one\nline two"
        crlf_text = "line one\r\nline two"
        lf_metadata = self._receipt_metadata(state.session_id, lf_text)
        db = MagicMock()
        prepared = self._receipt_row(state.session_id, "PREPARED")
        completed = self._receipt_row(state.session_id, "COMPLETED")
        db.prepare_acp_turn_receipt.return_value = prepared
        db.get_acp_turn_receipt.return_value = completed
        db.get_conversation_root.return_value = state.session_id
        db.validate_acp_turn_receipt_request.return_value = self._admission(lf_metadata)
        state.agent._session_db = db
        state.agent.run_conversation = MagicMock(
            return_value={"final_response": "ignored transient result", "messages": []}
        )
        request = ReceiptRequest(
            state.session_id, "turn-request-1", prepared["receiptIdentityDigest"]
        )
        claimed = ClaimedReceipt(request, "claim-token")
        receipt_adapter = MagicMock()
        receipt_adapter.claim_after_lease.return_value = claimed
        receipt_adapter.completed_replay.return_value = {
            **completed,
            "assistantContent": "durable exact response",
        }

        with patch("acp_adapter.server.TurnReceiptAdapter", return_value=receipt_adapter):
            refused = await agent.prompt(
                prompt=[TextContentBlock(type="text", text=crlf_text)],
                session_id=state.session_id,
                hermes={"acpTerminalReceipt": lf_metadata},
            )

            assert refused.stop_reason == "refusal"
            assert self._terminal_meta(refused)["status"] == "REFUSED"
            db.prepare_acp_turn_receipt.assert_not_called()
            db.get_acp_turn_receipt.assert_not_called()
            state.agent.run_conversation.assert_not_called()
            assert state.history == []
            assert state.is_running is False

            exact_crlf_metadata = self._receipt_metadata(state.session_id, crlf_text)
            assert exact_crlf_metadata["receiptIdentity"]["promptDigest"] == "sha256:" + hashlib.sha256(
                json.dumps(crlf_text, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            db.reset_mock()
            db.get_conversation_root.return_value = state.session_id
            db.validate_acp_turn_receipt_request.return_value = self._admission(exact_crlf_metadata)
            state.agent.run_conversation.reset_mock()

            admitted = await agent.prompt(
                prompt=[TextContentBlock(type="text", text=crlf_text)],
                session_id=state.session_id,
                hermes={"acpTerminalReceipt": exact_crlf_metadata},
            )

        db.prepare_acp_turn_receipt.assert_called_once_with(
            state.session_id,
            exact_crlf_metadata["receiptIdentity"],
            exact_crlf_metadata["targetBindReceipt"],
        )
        run_kwargs = state.agent.run_conversation.call_args.kwargs
        assert run_kwargs["user_message"] == crlf_text
        assert run_kwargs["persist_user_message"] == crlf_text
        assert self._terminal_meta(admitted) == {
            **completed,
            "assistantContent": "durable exact response",
        }

    @pytest.mark.anyio
    async def test_acp_terminal_receipt_status_is_model_free_for_every_durable_state(
        self, agent, mock_manager
    ):
        receipt_adapter = MagicMock()
        with patch("acp_adapter.server.TurnReceiptAdapter", return_value=receipt_adapter):
            for stored_status, expected_status in (
                (None, "NEVER_FOUND"),
                ("CLAIMED", "CLAIMED"),
                ("COMPLETED", "COMPLETED"),
            ):
                state = mock_manager.create_session(cwd="/tmp")
                metadata = self._receipt_metadata(state.session_id, "status payload", operation="status")
                db = MagicMock()
                state.agent._session_db = db
                db.get_conversation_root.return_value = state.session_id
                state.agent.run_conversation = MagicMock()
                db.validate_acp_turn_receipt_request.return_value = self._admission(metadata)
                row = (
                    self._receipt_row(state.session_id, stored_status)
                    if stored_status is not None
                    else None
                )
                db.get_acp_turn_receipt.return_value = row
                receipt_adapter.reset_mock()
                receipt_adapter.completed_replay.return_value = (
                    {**row, "assistantContent": "exact completed bytes"}
                    if stored_status == "COMPLETED"
                    else None
                )

                response = await agent.prompt(
                    prompt=[],
                    session_id=state.session_id,
                    hermes={"acpTerminalReceipt": metadata},
                )

                assert self._terminal_meta(response)["status"] == expected_status
                if stored_status is None:
                    assert self._terminal_meta(response)["turnRequestId"] == "turn-request-1"
                if stored_status == "COMPLETED":
                    assert self._terminal_meta(response)["assistantContent"] == "exact completed bytes"
                    receipt_adapter.completed_replay.assert_called_once()
                else:
                    receipt_adapter.completed_replay.assert_not_called()
                db.prepare_acp_turn_receipt.assert_not_called()
                receipt_adapter.claim_after_lease.assert_not_called()
                state.agent.run_conversation.assert_not_called()

    @pytest.mark.anyio
    async def test_acp_terminal_receipt_duplicate_completed_execute_replays_exact_bytes(
        self, agent, mock_manager
    ):
        state = mock_manager.create_session(cwd="/tmp")
        metadata = self._receipt_metadata(state.session_id, "repeat this")
        db = MagicMock()
        completed = self._receipt_row(state.session_id, "COMPLETED")
        db.prepare_acp_turn_receipt.return_value = completed
        state.agent._session_db = db
        db.get_conversation_root.return_value = state.session_id
        db.validate_acp_turn_receipt_request.return_value = self._admission(metadata)
        state.agent.run_conversation = MagicMock()
        receipt_adapter = MagicMock()
        receipt_adapter.completed_replay.return_value = {
            **completed,
            "assistantContent": "replayed byte-for-byte",
        }

        with patch("acp_adapter.server.TurnReceiptAdapter", return_value=receipt_adapter):
            response = await agent.prompt(
                prompt=[TextContentBlock(type="text", text="repeat this")],
                session_id=state.session_id,
                hermes={"acpTerminalReceipt": metadata},
            )

        receipt_adapter.claim_after_lease.assert_not_called()
        state.agent.run_conversation.assert_not_called()
        assert self._terminal_meta(response)["assistantContent"] == "replayed byte-for-byte"

    @pytest.mark.anyio
    async def test_acp_terminal_receipt_refuses_selected_state_root_and_extra_metadata(
        self, agent, mock_manager
    ):
        state = mock_manager.create_session(cwd="/tmp")
        db = MagicMock()
        state.agent._session_db = db
        db.get_conversation_root.return_value = state.session_id
        state.agent.run_conversation = MagicMock()
        metadata = self._receipt_metadata(state.session_id, "wrapper failure")
        db.validate_acp_turn_receipt_request.return_value = self._admission(metadata)
        db.get_conversation_root.side_effect = ["other-conversation-root", state.session_id]
        wrong_target_response = await agent.prompt(
            prompt=[TextContentBlock(type="text", text="wrapper failure")],
            session_id=state.session_id,
            hermes={"acpTerminalReceipt": metadata},
        )
        assert wrong_target_response.stop_reason == "refusal"
        db.prepare_acp_turn_receipt.assert_not_called()
        db.get_conversation_root.side_effect = None
        db.get_conversation_root.return_value = state.session_id
        extra_metadata = {**metadata, "unexpected": True}

        extra_response = await agent.prompt(
            prompt=[TextContentBlock(type="text", text="wrapper failure")],
            session_id=state.session_id,
            hermes={"acpTerminalReceipt": extra_metadata},
        )

        assert extra_response.stop_reason == "refusal"
        db.prepare_acp_turn_receipt.assert_not_called()

        db.prepare_acp_turn_receipt.side_effect = ValueError("closed identity rejected")
        wrapper_response = await agent.prompt(
            prompt=[TextContentBlock(type="text", text="wrapper failure")],
            session_id=state.session_id,
            hermes={"acpTerminalReceipt": metadata},
        )

        assert wrapper_response.stop_reason == "refusal"
        assert self._terminal_meta(wrapper_response)["status"] == "REFUSED"
        db.prepare_acp_turn_receipt.assert_called_once_with(
            state.session_id, metadata["receiptIdentity"], metadata["targetBindReceipt"]
        )
        state.agent.run_conversation.assert_not_called()

    @pytest.mark.anyio
    async def test_acp_terminal_receipt_router_transports_nested_meta_symmetrically(self, agent):
        captured: dict = {}

        async def capture_prompt(*, prompt, session_id, **kwargs):
            captured.update(prompt=prompt, session_id=session_id, kwargs=kwargs)
            return PromptResponse(
                stop_reason="end_turn",
                field_meta={"hermes": {"acpTerminalReceipt": {"status": "NEVER_FOUND"}}},
            )

        agent.prompt = capture_prompt
        metadata = {"operation": "status", "receiptIdentity": {}, "targetBindReceipt": {}}
        router = build_agent_router(agent)

        result = await router(
            "session/prompt",
            {
                "sessionId": "session-1",
                "prompt": [{"type": "text", "text": "status"}],
                "_meta": {"hermes": {"acpTerminalReceipt": metadata}},
            },
            False,
        )

        assert captured["kwargs"]["hermes"] == {"acpTerminalReceipt": metadata}
        assert captured["kwargs"]["message_id"] is None
        assert result.field_meta["hermes"]["acpTerminalReceipt"]["status"] == "NEVER_FOUND"
        assert result.model_dump(by_alias=True)["_meta"] == {
            "hermes": {"acpTerminalReceipt": {"status": "NEVER_FOUND"}}
        }

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "payload",
        (
            lambda metadata: {
                **metadata,
                "receiptIdentity": {
                    key: value
                    for key, value in metadata["receiptIdentity"].items()
                    if key != "executorSessionId"
                },
            },
            lambda metadata: {
                **metadata,
                "receiptIdentity": {**metadata["receiptIdentity"], "unexpected": True},
            },
            lambda metadata: {
                **metadata,
                "receiptIdentity": {**metadata["receiptIdentity"], "promptDigest": "invalid"},
            },
            lambda metadata: {
                **metadata,
                "targetBindReceipt": {"requested_session_id": metadata["targetBindReceipt"]["requested_session_id"]},
            },
        ),
        ids=(
            "missing-identity-field",
            "extra-identity-field",
            "malformed-identity",
            "malformed-target-receipt",
        ),
    )
    async def test_acp_terminal_receipt_validator_refuses_before_prepare_or_model(
        self, agent, mock_manager, payload
    ):
        state = mock_manager.create_session(cwd="/tmp")
        metadata = payload(self._receipt_metadata(state.session_id, "admission must close"))
        db = MagicMock()
        db.validate_acp_turn_receipt_request.side_effect = ValueError("closed receipt refused")
        state.agent._session_db = db
        state.agent.run_conversation = MagicMock()

        response = await agent.prompt(
            prompt=[TextContentBlock(type="text", text="admission must close")],
            session_id=state.session_id,
            hermes={"acpTerminalReceipt": metadata},
        )

        assert response.stop_reason == "refusal"
        db.validate_acp_turn_receipt_request.assert_called_once_with(
            state.session_id, metadata["receiptIdentity"], metadata["targetBindReceipt"]
        )
        db.prepare_acp_turn_receipt.assert_not_called()
        state.agent.run_conversation.assert_not_called()

    @pytest.mark.anyio
    async def test_acp_terminal_receipt_requires_one_text_block_after_validation(
        self, agent, mock_manager
    ):
        state = mock_manager.create_session(cwd="/tmp")
        metadata = self._receipt_metadata(state.session_id, "one block only")
        db = MagicMock()
        db.validate_acp_turn_receipt_request.return_value = self._admission(metadata)
        db.get_conversation_root.return_value = state.session_id
        state.agent._session_db = db
        state.agent.run_conversation = MagicMock()

        response = await agent.prompt(
            prompt=[
                TextContentBlock(type="text", text="one block only"),
                TextContentBlock(type="text", text="second block"),
            ],
            session_id=state.session_id,
            hermes={"acpTerminalReceipt": metadata},
        )

        assert response.stop_reason == "refusal"
        db.validate_acp_turn_receipt_request.assert_called_once()
        db.prepare_acp_turn_receipt.assert_not_called()
        state.agent.run_conversation.assert_not_called()

    @pytest.mark.anyio
    async def test_acp_terminal_receipt_claimed_execute_is_durable_and_model_free(
        self, agent, mock_manager, tmp_path
    ):
        state = mock_manager.create_session(cwd="/tmp")
        db = SessionDB(tmp_path / "claimed.db")
        raw_text = "durable claimed"
        try:
            db.create_session(state.session_id, source="test")
            metadata = self._durable_metadata(db, state.session_id, raw_text)
            prepared = db.prepare_acp_turn_receipt(
                state.session_id, metadata["receiptIdentity"], metadata["targetBindReceipt"]
            )
            assert db.claim_turn_receipt(
                state.session_id,
                prepared["turnRequestId"],
                prepared["receiptIdentityDigest"],
                "already-claimed",
            )
            state.agent._session_db = db
            state.agent.session_id = state.session_id
            state.agent.run_conversation = MagicMock()

            response = await agent.prompt(
                prompt=[TextContentBlock(type="text", text=raw_text)],
                session_id=state.session_id,
                hermes={"acpTerminalReceipt": metadata},
            )

            assert self._terminal_meta(response)["status"] == "CLAIMED"
            state.agent.run_conversation.assert_not_called()
            receipt = db.get_acp_turn_receipt(
                state.session_id, metadata["receiptIdentity"], metadata["targetBindReceipt"]
            )
            assert receipt is not None
            assert receipt["status"] == "CLAIMED"
        finally:
            db.close()

    @pytest.mark.anyio
    async def test_acp_terminal_receipt_run_exception_keeps_claimed_without_assistant_row(
        self, agent, mock_manager, tmp_path
    ):
        state = mock_manager.create_session(cwd="/tmp")
        db = SessionDB(tmp_path / "run-exception.db")
        raw_text = "run throws after claim"
        try:
            db.create_session(state.session_id, source="test")
            metadata = self._durable_metadata(db, state.session_id, raw_text)
            state.agent._session_db = db
            state.agent.session_id = state.session_id
            state.agent.run_conversation = MagicMock(side_effect=RuntimeError("settlement failed"))

            response = await agent.prompt(
                prompt=[TextContentBlock(type="text", text=raw_text)],
                session_id=state.session_id,
                hermes={"acpTerminalReceipt": metadata},
            )

            assert self._terminal_meta(response)["status"] == "CLAIMED"
            state.agent.run_conversation.assert_called_once()
            receipt = db.get_acp_turn_receipt(
                state.session_id, metadata["receiptIdentity"], metadata["targetBindReceipt"]
            )
            assert receipt is not None
            assert receipt["status"] != "COMPLETED"
            assert db.get_messages(state.session_id) == []
        finally:
            db.close()

    @pytest.mark.anyio
    async def test_acp_terminal_receipt_completed_replays_durable_assistant_exactly(
        self, agent, mock_manager, tmp_path
    ):
        state = mock_manager.create_session(cwd="/tmp")
        db = SessionDB(tmp_path / "completed.db")
        raw_text = "duplicate completed"
        assistant_content = "durable assistant bytes\nwith exact spacing  "
        try:
            db.create_session(state.session_id, source="test")
            metadata = self._durable_metadata(db, state.session_id, raw_text)
            prepared = db.prepare_acp_turn_receipt(
                state.session_id, metadata["receiptIdentity"], metadata["targetBindReceipt"]
            )
            assert db.claim_turn_receipt(
                state.session_id,
                prepared["turnRequestId"],
                prepared["receiptIdentityDigest"],
                "completed-claim",
            )
            TurnReceiptAdapter(db).finish(
                state.session_id,
                prepared["turnRequestId"],
                prepared["receiptIdentityDigest"],
                "completed-claim",
                assistant_content=assistant_content,
                response_digest="sha256:" + "d" * 64,
            )
            state.agent._session_db = db
            state.agent.session_id = state.session_id
            state.agent.run_conversation = MagicMock()

            response = await agent.prompt(
                prompt=[TextContentBlock(type="text", text=raw_text)],
                session_id=state.session_id,
                hermes={"acpTerminalReceipt": metadata},
            )

            assert self._terminal_meta(response)["assistantContent"] == assistant_content
            state.agent.run_conversation.assert_not_called()
        finally:
            db.close()

    @pytest.mark.anyio
    async def test_acp_terminal_receipt_unrelated_metadata_keeps_legacy_prompt_path(
        self, agent, mock_manager
    ):
        state = mock_manager.create_session(cwd="/tmp")
        state.agent.run_conversation = MagicMock(
            return_value={"final_response": "legacy", "messages": []}
        )

        response = await agent.prompt(
            prompt=[TextContentBlock(type="text", text="legacy metadata")],
            session_id=state.session_id,
            hermes={"unrelated": {"value": True}},
        )

        assert response.stop_reason == "end_turn"
        state.agent.run_conversation.assert_called_once()

















# ---------------------------------------------------------------------------
# on_connect
# ---------------------------------------------------------------------------


class TestOnConnect:
    def test_on_connect_stores_client(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        agent.on_connect(mock_conn)
        assert agent._conn is mock_conn


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


class TestSlashCommands:
    """Test slash command dispatch in the ACP adapter."""

    def _make_state(self, mock_manager):
        state = mock_manager.create_session(cwd="/tmp")
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"
        state.model = "test-model"
        return state

    def test_help_lists_commands(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/help", state)
        assert result is not None
        assert "/help" in result
        assert "/model" in result
        assert "/tools" in result
        assert "/reset" in result

    def test_model_shows_current(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/model", state)
        assert "test-model" in result





    def test_reset_clears_history(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [{"role": "user", "content": "hello"}]
        result = agent._handle_slash_command("/reset", state)
        assert "cleared" in result.lower()
        assert len(state.history) == 0




    def test_compact_compresses_context(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ]
        state.agent.compression_enabled = True
        state.agent._cached_system_prompt = "system"
        state.agent.tools = None
        original_session_db = object()
        state.agent._session_db = original_session_db

        def _compress_context(messages, system_prompt, *, approx_tokens, task_id, force):
            assert state.agent._session_db is None
            assert messages == state.history
            assert system_prompt == "system"
            assert approx_tokens == 40
            assert task_id == state.session_id
            assert force is True
            return [{"role": "user", "content": "summary"}], "new-system"

        state.agent._compress_context = MagicMock(side_effect=_compress_context)

        with (
            patch.object(agent.session_manager, "save_session") as mock_save,
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                side_effect=[40, 12],
            ),
        ):
            result = agent._handle_slash_command("/compress", state)

        assert "Context compressed: 4 -> 1 messages" in result
        assert "~40 -> ~12 tokens" in result
        assert state.history == [{"role": "user", "content": "summary"}]
        assert state.agent._session_db is original_session_db
        state.agent._compress_context.assert_called_once_with(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
                {"role": "assistant", "content": "four"},
            ],
            "system",
            approx_tokens=40,
            task_id=state.session_id,
            force=True,
        )
        mock_save.assert_called_once_with(state.session_id)


    def test_unknown_command_returns_none(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/nonexistent", state)
        assert result is None


    def test_slash_handler_cwd_pin_does_not_leak(self, agent, mock_manager, tmp_path):
        """The pin is scoped to the handler's own context copy.

        Concurrent ACP sessions share the event loop, so a handler that pinned
        the ambient context would leave its workspace bound for whatever runs
        next. Asserting the ambient value is unchanged after dispatch keeps the
        fix from trading one cross-session leak for another.
        """
        from agent.runtime_cwd import resolve_agent_cwd

        workspace = tmp_path / "project"
        workspace.mkdir()
        state = mock_manager.create_session(cwd=str(workspace))
        state.cwd = str(workspace)
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"

        before = str(resolve_agent_cwd())
        agent._handle_slash_command("/help", state)
        assert str(resolve_agent_cwd()) == before





# ---------------------------------------------------------------------------
# _register_session_mcp_servers
# ---------------------------------------------------------------------------


class TestRegisterSessionMcpServers:
    """Tests for ACP MCP server registration in session lifecycle."""

    @pytest.mark.asyncio
    async def test_noop_when_no_servers(self, agent, mock_manager):
        """No-op when mcp_servers is None or empty."""
        state = mock_manager.create_session(cwd="/tmp")
        # Should not raise
        await agent._register_session_mcp_servers(state, None)
        await agent._register_session_mcp_servers(state, [])

    @pytest.mark.asyncio
    async def test_registers_stdio_servers(self, agent, mock_manager):
        """McpServerStdio servers are converted and passed to register_mcp_servers."""
        from acp.schema import McpServerStdio, EnvVariable

        state = mock_manager.create_session(cwd="/tmp")
        # Give the mock agent the attributes _register_session_mcp_servers reads
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()

        server = McpServerStdio(
            name="test-server",
            command="/usr/bin/test",
            args=["--flag"],
            env=[EnvVariable(name="KEY", value="val")],
        )

        registered_config = {}
        def capture_register(config_map):
            registered_config.update(config_map)
            return ["mcp_test_server_tool1"]

        with patch("tools.mcp_tool.register_mcp_servers", side_effect=capture_register), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await agent._register_session_mcp_servers(state, [server])

        assert "test-server" in registered_config
        cfg = registered_config["test-server"]
        assert cfg["command"] == "/usr/bin/test"
        assert cfg["args"] == ["--flag"]
        assert cfg["env"] == {"KEY": "val"}


    @pytest.mark.asyncio
    async def test_refreshes_agent_tool_surface(self, agent, mock_manager):
        """After MCP registration, agent.tools and valid_tool_names are refreshed."""
        from acp.schema import McpServerStdio

        state = mock_manager.create_session(cwd="/tmp")
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()
        state.agent._cached_system_prompt = "old prompt"
        state.agent._memory_manager = SimpleNamespace(
            get_all_tool_schemas=lambda: [
                {"name": "hindsight_recall", "description": "Recall", "parameters": {}}
            ]
        )

        server = McpServerStdio(
            name="srv",
            command="/bin/test",
            args=[],
            env=[],
        )

        fake_tools = [
            {"function": {"name": "mcp_srv_search"}},
            {"function": {"name": "memory"}},
            {"function": {"name": "terminal"}},
        ]

        with patch("tools.mcp_tool.register_mcp_servers", return_value=["mcp_srv_search"]), \
             patch("model_tools.get_tool_definitions", return_value=fake_tools) as mock_defs:
            await agent._register_session_mcp_servers(state, [server])

        mock_defs.assert_called_once_with(
            enabled_toolsets=["hermes-acp", "mcp-srv"],
            disabled_toolsets=None,
            quiet_mode=True,
        )
        assert state.agent.enabled_toolsets == ["hermes-acp", "mcp-srv"]
        assert state.agent.tools is fake_tools
        assert state.agent.tools[-1] == {
            "type": "function",
            "function": {
                "name": "hindsight_recall",
                "description": "Recall",
                "parameters": {},
            },
        }
        assert state.agent.valid_tool_names == {
            "hindsight_recall",
            "memory",
            "mcp_srv_search",
            "terminal",
        }
        # _invalidate_system_prompt should have been called
        state.agent._invalidate_system_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_failure_logs_warning(self, agent, mock_manager):
        """If register_mcp_servers raises, warning is logged but no crash."""
        from acp.schema import McpServerStdio

        state = mock_manager.create_session(cwd="/tmp")
        server = McpServerStdio(
            name="bad",
            command="/nonexistent",
            args=[],
            env=[],
        )

        with patch("tools.mcp_tool.register_mcp_servers", side_effect=RuntimeError("boom")):
            # Should not raise
            await agent._register_session_mcp_servers(state, [server])
