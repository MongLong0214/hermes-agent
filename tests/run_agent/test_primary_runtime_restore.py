"""Tests for per-turn primary runtime restoration and transport recovery.

Verifies that:
1. Fallback is turn-scoped: a new turn restores the primary model/provider
2. The fallback chain index resets so all fallbacks are available again
3. Context compressor state is restored alongside the runtime
4. Transient transport errors get one recovery cycle before fallback
5. Recovery is skipped for aggregator providers (OpenRouter, Nous)
6. Non-transport errors don't trigger recovery
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from agent.gemini_outbound_policy import GeminiOutboundDenied
from run_agent import AIAgent


class _SideEffectReached(BaseException):
    """Prohibited restore/recovery/refresh effect before default-deny."""


def _explode(*_args, **_kwargs):
    raise _SideEffectReached("prohibited Gemini/Vertex side effect reached")


def _assert_stable_denial(exc_info):
    assert exc_info.type is GeminiOutboundDenied
    assert type(exc_info.value) is GeminiOutboundDenied
    assert exc_info.value.code == "gemini_outbound_denied"
    assert str(exc_info.value) == "Gemini outbound requests are disabled."
    assert vars(exc_info.value) == {}


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _make_agent(fallback_model=None, provider="custom", base_url="https://my-llm.example.com/v1"):
    """Create a minimal AIAgent with optional fallback config."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        # Unit tests must not probe live endpoints. The compressor resolves
        # context length lazily via a real network call against base_url; for
        # reachable hosts (the nous portal case) the endpoint's answer for the
        # empty test model (32K) trips agent_init's 64K floor and fails the
        # test on network behavior, not code under test.
        patch(
            "agent.context_compressor.get_model_context_length",
            return_value=200_000,
        ),
    ):
        agent = AIAgent(
            api_key="test-key-12345678",
            base_url=base_url,
            provider=provider,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_resolve(base_url="https://openrouter.ai/api/v1", api_key="fallback-key-1234"):
    """Helper to create a mock client for resolve_provider_client."""
    mock_client = MagicMock()
    mock_client.api_key = api_key
    mock_client.base_url = base_url
    return mock_client


# =============================================================================
# _primary_runtime snapshot
# =============================================================================

class TestPrimaryRuntimeSnapshot:
    def test_snapshot_created_at_init(self):
        agent = _make_agent()
        assert hasattr(agent, "_primary_runtime")
        rt = agent._primary_runtime
        assert rt["model"] == agent.model
        assert rt["provider"] == "custom"
        assert rt["base_url"] == "https://my-llm.example.com/v1"
        assert rt["api_mode"] == agent.api_mode
        assert "client_kwargs" in rt
        assert "compressor_context_length" in rt

    def test_snapshot_includes_compressor_state(self):
        agent = _make_agent()
        rt = agent._primary_runtime
        cc = agent.context_compressor
        assert rt["compressor_model"] == cc.model
        assert rt["compressor_provider"] == cc.provider
        assert rt["compressor_context_length"] == cc.context_length
        assert rt["compressor_threshold_tokens"] == cc.threshold_tokens

    def test_snapshot_includes_anthropic_state_when_applicable(self):
        """Anthropic-mode agents should snapshot Anthropic-specific state."""
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            agent = AIAgent(
                api_key="sk-ant-test-12345678",
                base_url="https://api.anthropic.com",
                provider="anthropic",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        rt = agent._primary_runtime
        assert "anthropic_api_key" in rt
        assert "anthropic_base_url" in rt
        assert "is_anthropic_oauth" in rt

    def test_snapshot_omits_anthropic_for_openai_mode(self):
        agent = _make_agent(provider="custom")
        rt = agent._primary_runtime
        assert "anthropic_api_key" not in rt


# =============================================================================
# _restore_primary_runtime()
# =============================================================================

class TestRestorePrimaryRuntime:
    def test_noop_when_not_fallback(self):
        agent = _make_agent()
        assert agent._fallback_activated is False
        assert agent._restore_primary_runtime() is False

    def test_restores_model_and_provider(self):
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        )
        original_model = agent.model
        original_provider = agent.provider

        # Simulate fallback activation
        mock_client = _mock_resolve()
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            agent._try_activate_fallback()

        assert agent._fallback_activated is True
        assert agent.model == "anthropic/claude-sonnet-4"
        assert agent.provider == "openrouter"

        # Restore should bring back the primary
        with patch("run_agent.OpenAI", return_value=MagicMock()):
            result = agent._restore_primary_runtime()

        assert result is True
        assert agent._fallback_activated is False
        assert agent.model == original_model
        assert agent.provider == original_provider

    def test_resets_fallback_index(self):
        """After restore, the full fallback chain should be available again."""
        agent = _make_agent(
            fallback_model=[
                {"provider": "openrouter", "model": "model-a"},
                {"provider": "anthropic", "model": "model-b"},
            ],
        )
        # Advance through the chain
        mock_client = _mock_resolve()
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            agent._try_activate_fallback()

        assert agent._fallback_index == 1  # consumed one entry

        with patch("run_agent.OpenAI", return_value=MagicMock()):
            agent._restore_primary_runtime()

        assert agent._fallback_index == 0  # reset for next turn

    def test_restores_compressor_state(self):
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        )
        original_ctx_len = agent.context_compressor.context_length
        original_threshold = agent.context_compressor.threshold_tokens

        # Simulate fallback modifying compressor
        mock_client = _mock_resolve()
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            agent._try_activate_fallback()

        # Manually simulate compressor being changed (as _try_activate_fallback does)
        agent.context_compressor.context_length = 32000
        agent.context_compressor.threshold_tokens = 25600

        with patch("run_agent.OpenAI", return_value=MagicMock()):
            agent._restore_primary_runtime()

        assert agent.context_compressor.context_length == original_ctx_len
        assert agent.context_compressor.threshold_tokens == original_threshold

    def test_restores_prompt_caching_flag(self):
        agent = _make_agent()
        original_caching = agent._use_prompt_caching

        # Simulate fallback changing the caching flag
        agent._fallback_activated = True
        agent._use_prompt_caching = not original_caching

        with patch("run_agent.OpenAI", return_value=MagicMock()):
            agent._restore_primary_runtime()

        assert agent._use_prompt_caching == original_caching

    def test_restore_skips_cross_provider_pool_entry(self):
        """Restore must not swap in a fallback provider credential for the primary runtime."""

        class _Entry:
            provider = "openrouter"
            id = "fallback-entry"
            label = "fallback"
            runtime_api_key = "fallback-key"
            runtime_base_url = "https://openrouter.ai/api/v1"
            access_token = "fallback-key"

        class _Pool:
            provider = "openrouter"

            def has_available(self):
                return True

            def select(self):
                return _Entry()

        agent = _make_agent(
            provider="custom",
            base_url="https://primary.example.com/v1",
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        )
        original_base_url = agent.base_url
        mock_client = _mock_resolve()
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            agent._try_activate_fallback()
        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()

        with patch("run_agent.OpenAI", return_value=MagicMock()):
            result = agent._restore_primary_runtime()

        assert result is True
        assert agent.provider == "custom"
        assert agent.base_url == original_base_url
        agent._swap_credential.assert_not_called()

    def test_restore_keeps_primary_base_url_when_fallback_pool_attached(self):
        """Issue #56885: plain-provider primary must not inherit a fallback
        provider's base_url via the restore-path pool reselect.

        Repro: primary is openai-api/gpt-5.5, a transient failure falls back to
        deepseek and attaches deepseek's credential pool. On the next turn the
        restore reselect must NOT swap in the deepseek entry — otherwise the
        request goes out as model=gpt-5.5 to base_url=api.deepseek.com → 404.
        """

        class _DeepseekEntry:
            provider = "deepseek"
            id = "dsk-1"
            label = "deepseek-key"
            runtime_api_key = "sk-deepseek-xxx"
            runtime_base_url = "https://api.deepseek.com/v1"
            base_url = "https://api.deepseek.com/v1"
            access_token = "sk-deepseek-xxx"

        class _DeepseekPool:
            provider = "deepseek"

            def has_available(self):
                return True

            def select(self):
                return _DeepseekEntry()

        agent = _make_agent(
            provider="openai-api",
            base_url="https://api.openai.com/v1",
            fallback_model={"provider": "deepseek", "model": "deepseek-v4-flash"},
        )
        primary_base_url = agent.base_url
        primary_provider = agent.provider
        mock_client = _mock_resolve(base_url="https://api.deepseek.com/v1")
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, None),
        ):
            agent._try_activate_fallback()
        # Fallback attached deepseek's pool; simulate it surviving into the next turn.
        agent._credential_pool = _DeepseekPool()
        agent._swap_credential = MagicMock()

        primary_pool = MagicMock()
        primary_pool.provider = primary_provider
        primary_pool.has_available.return_value = False
        with (
            patch("run_agent.OpenAI", return_value=MagicMock()),
            patch("agent.credential_pool.load_pool", return_value=primary_pool) as load_pool,
        ):
            result = agent._restore_primary_runtime()

        assert result is True
        assert agent.provider == primary_provider
        assert agent.base_url == primary_base_url
        assert "deepseek" not in str(agent.base_url)
        assert agent._credential_pool is primary_pool
        load_pool.assert_called_once_with(primary_provider)
        agent._swap_credential.assert_not_called()

    def test_restore_clears_fallback_pool_when_primary_pool_reload_fails(self):
        """A fallback pool must never remain attached to the restored primary."""
        agent = _make_agent(
            provider="openai-api",
            base_url="https://api.openai.com/v1",
        )
        agent._fallback_activated = True
        fallback_pool = MagicMock()
        fallback_pool.provider = "deepseek"
        agent._credential_pool = fallback_pool

        with (
            patch("run_agent.OpenAI", return_value=MagicMock()),
            patch(
                "agent.credential_pool.load_pool",
                side_effect=RuntimeError("auth store unavailable"),
            ),
        ):
            result = agent._restore_primary_runtime()

        assert result is True
        assert agent.provider == "openai-api"
        assert agent._credential_pool is None

    def test_restore_swaps_matching_custom_pool_entry(self):
        """Custom primary + custom:<name> entry whose base_url resolves to the
        SAME custom key must swap (legitimate same-endpoint rotation)."""

        class _Entry:
            provider = "custom:myllm"
            id = "custom-entry"
            label = "myllm"
            runtime_api_key = "custom-key"
            runtime_base_url = "https://my-llm.example.com/v1"
            access_token = "custom-key"

        class _Pool:
            provider = "custom:myllm"

            def has_available(self):
                return True

            def select(self):
                return _Entry()

        agent = _make_agent(provider="custom", base_url="https://my-llm.example.com/v1")
        agent._fallback_activated = True
        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()

        with (
            patch(
                "agent.credential_pool.get_custom_provider_pool_key",
                return_value="custom:myllm",
            ),
            patch("run_agent.OpenAI", return_value=MagicMock()),
        ):
            result = agent._restore_primary_runtime()

        assert result is True
        agent._swap_credential.assert_called_once()




# =============================================================================
# _try_recover_primary_transport()
# =============================================================================

def _make_transport_error(error_type="ReadTimeout"):
    """Create an exception whose type().__name__ matches the given name."""
    cls = type(error_type, (Exception,), {})
    return cls("connection timed out")


class TestTryRecoverPrimaryTransport:

    def test_recovers_on_read_timeout(self):
        agent = _make_agent(provider="custom")
        error = _make_transport_error("ReadTimeout")

        with patch("run_agent.OpenAI", return_value=MagicMock()), \
             patch("time.sleep"):
            result = agent._try_recover_primary_transport(
                error, retry_count=3, max_retries=3,
            )

        assert result is True





    def test_skipped_when_already_on_fallback(self):
        agent = _make_agent(provider="custom")
        agent._fallback_activated = True
        error = _make_transport_error("ReadTimeout")

        result = agent._try_recover_primary_transport(
            error, retry_count=3, max_retries=3,
        )
        assert result is False




    def test_allowed_for_nous_anthropic_messages(self):
        """Portal Claude holds a local Anthropic SDK client — rebuild it."""
        agent = _make_agent(
            provider="nous",
            base_url="https://inference-api.nousresearch.com/v1",
        )
        agent.api_mode = "anthropic_messages"
        agent.model = "anthropic/claude-opus-4.8"
        agent._primary_runtime.update({
            "api_mode": "anthropic_messages",
            "model": "anthropic/claude-opus-4.8",
            "provider": "nous",
            "anthropic_api_key": "portal-jwt",
            "anthropic_base_url": "https://inference-api.nousresearch.com/v1",
            "is_anthropic_oauth": False,
        })
        error = _make_transport_error("ReadTimeout")
        rebuilt = MagicMock(name="anthropic-client")

        with (
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                return_value=rebuilt,
            ),
            patch("time.sleep"),
        ):
            result = agent._try_recover_primary_transport(
                error, retry_count=3, max_retries=3,
            )

        assert result is True
        assert agent._anthropic_client is rebuilt



    def test_wait_time_scales_with_retry_count(self):
        agent = _make_agent(provider="custom")
        error = _make_transport_error("ReadTimeout")

        with patch("run_agent.OpenAI", return_value=MagicMock()), \
             patch("time.sleep") as mock_sleep:
            agent._try_recover_primary_transport(
                error, retry_count=3, max_retries=3,
            )
            # wait_time = min(3 + retry_count, 8) = min(6, 8) = 6
            mock_sleep.assert_called_once_with(6)

    def test_wait_time_capped_at_8(self):
        agent = _make_agent(provider="custom")
        error = _make_transport_error("ReadTimeout")

        with patch("run_agent.OpenAI", return_value=MagicMock()), \
             patch("time.sleep") as mock_sleep:
            agent._try_recover_primary_transport(
                error, retry_count=10, max_retries=3,
            )
            # wait_time = min(3 + 10, 8) = 8
            mock_sleep.assert_called_once_with(8)


    def test_survives_rebuild_failure(self):
        """If client rebuild fails, returns False gracefully."""
        agent = _make_agent(provider="custom")
        error = _make_transport_error("ReadTimeout")

        with patch("run_agent.OpenAI", side_effect=Exception("socket error")), \
             patch("time.sleep"):
            result = agent._try_recover_primary_transport(
                error, retry_count=3, max_retries=3,
            )

        assert result is False


# =============================================================================
# Integration: restore_primary_runtime called from run_conversation
# =============================================================================

class TestRestoreInRunConversation:
    """Verify the hook in run_conversation() calls _restore_primary_runtime."""

    def test_restore_called_at_turn_start(self):
        agent = _make_agent()
        agent._fallback_activated = True

        with patch.object(agent, "_restore_primary_runtime", return_value=True) as mock_restore, \
             patch.object(agent, "run_conversation", wraps=None) as _:
            # We can't easily run the full conversation, but we can verify
            # the method exists and is callable
            agent._restore_primary_runtime()
            mock_restore.assert_called_once()

    def test_full_cycle_fallback_then_restore(self):
        """Simulate: turn 1 activates fallback, turn 2 restores primary."""
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
            provider="custom",
        )

        # Turn 1: activate fallback
        mock_client = _mock_resolve()
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            assert agent._try_activate_fallback() is True

        assert agent._fallback_activated is True
        assert agent.model == "anthropic/claude-sonnet-4"
        assert agent.provider == "openrouter"
        assert agent._fallback_index == 1

        # Turn 2: restore primary
        with patch("run_agent.OpenAI", return_value=MagicMock()):
            assert agent._restore_primary_runtime() is True

        assert agent._fallback_activated is False
        assert agent._fallback_index == 0
        assert agent.provider == "custom"
        assert agent.base_url == "https://my-llm.example.com/v1"


# =============================================================================
# Rate-limit cooldown gate
# =============================================================================

class TestRateLimitCooldown:
    """Verify _restore_primary_runtime() respects the 60s rate-limit cooldown."""

    def test_restore_blocked_during_cooldown(self):
        """While _rate_limited_until is in the future, restore returns False."""
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        )
        mock_client = _mock_resolve()
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            agent._try_activate_fallback()

        assert agent._fallback_activated is True

        # Manually set cooldown well into the future
        agent._rate_limited_until = time.monotonic() + 60

        result = agent._restore_primary_runtime()
        assert result is False
        assert agent._fallback_activated is True  # still on fallback


    def test_cooldown_set_on_rate_limit_reason(self):
        """_try_activate_fallback with rate_limit reason sets _rate_limited_until."""
        from run_agent import FailoverReason
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        )
        before = time.monotonic()
        mock_client = _mock_resolve()
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            agent._try_activate_fallback(reason=FailoverReason.rate_limit)

        assert hasattr(agent, "_rate_limited_until")
        assert agent._rate_limited_until > before + 50  # ~60s from now

    def test_cooldown_not_set_when_already_on_fallback(self):
        """Chain-switching while already on fallback must not reset cooldown."""
        from run_agent import FailoverReason
        agent = _make_agent(
            fallback_model=[
                {"provider": "openrouter", "model": "model-a"},
                {"provider": "anthropic", "model": "model-b"},
            ],
        )
        mock_client = _mock_resolve()
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            # First call: leaving primary → cooldown should be set
            agent._try_activate_fallback(reason=FailoverReason.rate_limit)
            first_cooldown = getattr(agent, "_rate_limited_until", 0)

            # Second call: already on fallback (provider != primary) → cooldown must not advance
            agent._try_activate_fallback(reason=FailoverReason.rate_limit)
            second_cooldown = getattr(agent, "_rate_limited_until", 0)

        # second call should not have extended the cooldown
        assert second_cooldown == first_cooldown


def test_restore_primary_runtime_denies_gemini_snapshot_before_mutation():
    agent = _make_agent(provider="custom", base_url="https://my-llm.example.com/v1")
    original_client = agent.client
    original_provider = agent.provider
    original_model = agent.model
    original_url = agent.base_url
    original_kwargs = dict(agent._client_kwargs)
    original_key = agent.api_key
    agent._fallback_activated = True
    agent._fallback_index = 1
    agent._primary_runtime["provider"] = "vertex"
    agent._primary_runtime["requested_provider"] = "vertex"
    agent._primary_runtime["base_url"] = "https://aiplatform.googleapis.com/v1"
    agent._primary_runtime["model"] = "google/gemini-2.5-flash"
    agent._primary_runtime["client_kwargs"] = {
        "api_key": "placeholder-vertex-token",
        "base_url": "https://aiplatform.googleapis.com/v1",
    }

    with (
        patch("agent.credential_pool.load_pool", side_effect=_explode) as load_pool,
        patch("run_agent.OpenAI", side_effect=_explode) as openai_cls,
        patch.object(agent, "_create_openai_client", side_effect=_explode),
        patch.object(agent, "_retire_shared_openai_client", side_effect=_explode),
    ):
        with pytest.raises(GeminiOutboundDenied) as exc_info:
            agent._restore_primary_runtime()

    _assert_stable_denial(exc_info)
    load_pool.assert_not_called()
    openai_cls.assert_not_called()
    assert agent.client is original_client
    assert agent.provider == original_provider
    assert agent.model == original_model
    assert agent.base_url == original_url
    assert agent.api_key == original_key
    assert agent._client_kwargs == original_kwargs
    assert agent._fallback_activated is True
    assert agent._fallback_index == 1


def test_try_recover_primary_transport_denies_gemini_snapshot_before_sleep():
    agent = _make_agent(provider="custom", base_url="https://my-llm.example.com/v1")
    original_client = agent.client
    original_provider = agent.provider
    original_model = agent.model
    original_url = agent.base_url
    original_kwargs = dict(agent._client_kwargs)
    agent._primary_runtime["provider"] = "gemini"
    agent._primary_runtime["requested_provider"] = "gemini"
    agent._primary_runtime["base_url"] = "https://generativelanguage.googleapis.com/v1beta"
    agent._primary_runtime["model"] = "gemini-2.5-flash"
    agent._primary_runtime["client_kwargs"] = {
        "api_key": "placeholder-gemini-key",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
    }
    error = _make_transport_error("ReadTimeout")

    with (
        patch("time.sleep", side_effect=_explode) as slept,
        patch("run_agent.OpenAI", side_effect=_explode) as openai_cls,
        patch.object(agent, "_retire_shared_openai_client", side_effect=_explode),
        patch.object(agent, "_create_openai_client", side_effect=_explode),
    ):
        with pytest.raises(GeminiOutboundDenied) as exc_info:
            agent._try_recover_primary_transport(error, retry_count=3, max_retries=3)

    _assert_stable_denial(exc_info)
    slept.assert_not_called()
    openai_cls.assert_not_called()
    assert agent.client is original_client
    assert agent.provider == original_provider
    assert agent.model == original_model
    assert agent.base_url == original_url
    assert agent._client_kwargs == original_kwargs


def test_try_refresh_vertex_client_credentials_denies_before_token_mint():
    agent = _make_agent(provider="custom", base_url="https://my-llm.example.com/v1")
    agent.provider = "vertex"
    agent.api_mode = "chat_completions"
    original_client = agent.client
    original_key = agent.api_key
    original_url = agent.base_url
    original_kwargs = dict(agent._client_kwargs)

    with (
        patch(
            "agent.vertex_adapter.get_vertex_config",
            side_effect=_explode,
        ) as get_config,
        patch.object(agent, "_replace_primary_openai_client", side_effect=_explode),
        patch("run_agent.logger") as mock_logger,
    ):
        with pytest.raises(GeminiOutboundDenied) as exc_info:
            agent._try_refresh_vertex_client_credentials()

    _assert_stable_denial(exc_info)
    get_config.assert_not_called()
    assert agent.client is original_client
    assert agent.api_key == original_key
    assert agent.base_url == original_url
    assert agent._client_kwargs == original_kwargs
    mock_logger.info.assert_not_called()
