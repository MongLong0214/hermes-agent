"""Regression tests for the ``auto`` → main-model-first policy.

Prior to this change, aggregator users (OpenRouter / Nous Portal) had aux
tasks routed through a cheap provider-side default (Gemini Flash) while
non-aggregator users got their main model.  This made behavior inconsistent
and surprising — users picked Claude but got Gemini Flash summaries.

The current policy: ``auto`` means "use my main chat model" for every user,
regardless of provider type.  Explicit per-task overrides in ``config.yaml``
(``auxiliary.<task>.provider``) still win.  The cheap fallback chain only
runs when the main provider has no working client.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch



# ── Text aux tasks — _resolve_auto_route ──────────────────────────────────────────


class TestResolveAutoMainFirst:
    """_resolve_auto_route() must prefer main provider + main model for every user."""

    def test_title_generation_auto_honors_main_model(self):
        """The default auto title route must not replace the selected main model."""
        main_model = "deepseek-v4-flash-free"
        mock_client = MagicMock()

        with patch(
            "agent.auxiliary_client._get_aux_model_for_provider",
            return_value="gemini-3-flash",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, main_model),
        ) as mock_resolve, patch(
            "agent.auxiliary_client._is_provider_unhealthy", return_value=False
        ):
            from agent.auxiliary_client import _resolve_auto_route

            client, model, _provider = _resolve_auto_route(
                main_runtime={
                    "provider": "opencode-zen",
                    "model": main_model,
                },
                task="title_generation",
            )

        assert client is mock_client
        assert model == main_model
        assert mock_resolve.call_args.args[:2] == ("opencode-zen", main_model)

    def test_title_generation_can_opt_into_provider_fast_model(self):
        """The latency optimization remains available as an explicit opt-in."""
        fast_model = "gemini-3-flash"
        mock_client = MagicMock()

        def resolve(_provider, model, **_kwargs):
            return mock_client, model

        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"prefer_fast_model": True},
        ), patch(
            "agent.auxiliary_client._get_aux_model_for_provider",
            return_value=fast_model,
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=resolve,
        ), patch(
            "agent.auxiliary_client._is_provider_unhealthy", return_value=False
        ):
            from agent.auxiliary_client import _resolve_auto_route

            client, model, _provider = _resolve_auto_route(
                main_runtime={
                    "provider": "opencode-zen",
                    "model": "deepseek-v4-flash-free",
                },
                task="title_generation",
            )

        assert client is mock_client
        assert model == fast_model


    def test_moa_main_resolves_aux_to_aggregator(self, monkeypatch, tmp_path):
        """MoA main user → aux runs on the aggregator slot, NOT the preset name.

        provider='moa'/model='opus-gpt' would otherwise send the preset name
        'opus-gpt' as the model id and 400 ("not a valid model ID"). Aux tasks
        don't need the reference fan-out — they use the aggregator (the preset's
        acting model). The virtual moa://local base_url + placeholder key must
        be dropped so the aggregator resolves via its own provider credentials.
        """
        import yaml

        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "moa": {
                        "default_preset": "opus-gpt",
                        "presets": {
                            "opus-gpt": {
                                "enabled": True,
                                "reference_models": [{"provider": "openrouter", "model": "openai/gpt-5.5"}],
                                "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
                            }
                        },
                    }
                }
            )
        )
        monkeypatch.setenv("HERMES_HOME", str(home))

        with patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve, patch(
            "agent.auxiliary_client._is_provider_unhealthy", return_value=False
        ):
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "anthropic/claude-opus-4.8")

            from agent.auxiliary_client import _resolve_auto_route

            client, model, _provider = _resolve_auto_route(
                main_runtime={
                    "provider": "moa",
                    "model": "opus-gpt",
                    "base_url": "moa://local",
                    "api_key": "moa-virtual-provider",
                    "api_mode": "chat_completions",
                },
                task="title_generation",
            )

        assert client is mock_client
        # Resolved to the aggregator's real provider+model, not the preset name.
        assert mock_resolve.call_args.args[0] == "openrouter"
        assert mock_resolve.call_args.args[1] == "anthropic/claude-opus-4.8"
        # The virtual moa://local endpoint must not be forwarded as the
        # aggregator's base_url.
        assert mock_resolve.call_args.kwargs.get("explicit_base_url") in (None, "")




    def test_main_unavailable_uses_task_fallback_chain_before_builtin_chain(self):
        """Auto aux resolution honors auxiliary.<task>.fallback_chain before built-ins."""
        task_client = MagicMock()
        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="nvidia",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="qwen/qwen3.5-122b-a10b",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(None, None),  # main provider has no client
        ), patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(task_client, "task-free-model", "fallback_chain[0](openrouter)"),
        ) as mock_task_chain, patch(
            "agent.auxiliary_client._try_main_fallback_chain",
        ) as mock_main_chain, patch(
            "agent.auxiliary_client._try_openrouter",
        ) as mock_openrouter:
            from agent.auxiliary_client import _resolve_auto_route

            client, model, _provider = _resolve_auto_route(task="title_generation")

        assert client is task_client
        assert model == "task-free-model"
        mock_task_chain.assert_called_once_with(
            "title_generation", "nvidia", reason="main provider unavailable")
        mock_main_chain.assert_not_called()
        mock_openrouter.assert_not_called()




    def test_main_fallback_skips_google_static_and_dynamic_routes_before_key_resolution(self):
        """Blocked Gemini/Vertex routes consume neither a fallback key nor client slot.

        Both provider aliases and dynamically named entries resolved to Google's inference
        URLs are rejected locally; the next benign main-fallback remains available.
        """
        benign_client = MagicMock()
        entries = [
            {"provider": "gemini", "model": "gemini-3-flash"},
            {"provider": "vertex", "model": "gemini-3-flash"},
            {
                "provider": "permitted-name", "model": "ordinary-model",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            },
            {
                "provider": "another-permitted-name", "model": "ordinary-model",
                "base_url": "https://aiplatform.googleapis.com/v1beta1/publishers/google",
            },
            {"provider": "nous", "model": "hermes-4"},
        ]
        key_reads = []
        resolve_calls = []

        def resolve_key(entry):
            key_reads.append(entry["provider"])
            return "safe-key"

        def resolve_client(provider, model, **_kwargs):
            resolve_calls.append((provider, model))
            return benign_client, model

        with patch(
            "hermes_cli.config.load_config_readonly", return_value={}
        ), patch(
            "hermes_cli.fallback_config.get_fallback_chain", return_value=entries
        ), patch(
            "hermes_cli.fallback_config.resolve_entry_api_key", side_effect=resolve_key
        ), patch(
            "agent.auxiliary_client.resolve_provider_client", side_effect=resolve_client
        ), patch(
            "agent.auxiliary_client._is_provider_unhealthy", return_value=False
        ):
            from agent.auxiliary_client import _try_main_fallback_chain

            client, model, provider = _try_main_fallback_chain("title_generation")

        assert (client, model, provider) == (benign_client, "hermes-4", "nous")
        assert key_reads == ["nous"]
        assert resolve_calls == [("nous", "hermes-4")]

    def test_main_fallback_skips_ambiguous_custom_routes_before_key_client_or_health_cache(self):
        """An unconfigured custom route cannot consume state before an explicit route wins."""
        benign_client = MagicMock()
        entries = [
            {"provider": "custom:unconfigured-route", "model": "unknown-model"},
            {"provider": "custom", "model": "unknown-model"},
            {"provider": "custom:uncertain-url", "model": "unknown-model", "base_url": "https://["},
            {"provider": "custom:invalid-port", "model": "unknown-model", "base_url": "https://relay.example.test:invalid/v1"},
            {"provider": "custom:out-of-range-port", "model": "unknown-model", "base_url": "https://relay.example.test:65536/v1"},
            {
                "provider": "custom:google-url",
                "model": "ordinary-model",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            },
            {
                "provider": "custom:relay",
                "model": "relay-model",
                "base_url": "https://relay.example.test/v1",
            },
        ]
        key_reads = []
        resolve_calls = []
        health_calls = []

        def resolve_key(entry):
            key_reads.append(entry["provider"])
            return "safe-key"

        def resolve_client(provider, model, **_kwargs):
            resolve_calls.append((provider, model))
            if provider == "custom:relay":
                return benign_client, model
            return None, None

        def health_base_url(provider, explicit_base_url=None):
            health_calls.append((provider, explicit_base_url))
            return str(explicit_base_url or "")

        with patch(
            "hermes_cli.config.load_config_readonly", return_value={}
        ), patch(
            "hermes_cli.fallback_config.get_fallback_chain", return_value=entries
        ), patch(
            "hermes_cli.fallback_config.resolve_entry_api_key", side_effect=resolve_key
        ), patch(
            "agent.auxiliary_client.resolve_provider_client", side_effect=resolve_client
        ), patch(
            "agent.auxiliary_client._custom_health_base_url", side_effect=health_base_url
        ), patch(
            "agent.auxiliary_client._is_provider_unhealthy", return_value=False
        ):
            from agent.auxiliary_client import _try_main_fallback_chain

            client, model, provider = _try_main_fallback_chain("title_generation")

        assert (client, model, provider) == (benign_client, "relay-model", "custom:relay")
        assert key_reads == ["custom:relay"]
        assert resolve_calls == [("custom:relay", "relay-model")]
        assert health_calls == [
            ("custom:relay", "https://relay.example.test/v1"),
        ]

    def test_unconfigured_named_custom_route_is_not_admitted_by_either_fallback_chain(self):
        """Absent route facts for custom aliases cannot consume health, key, or client state."""
        route_reads = []
        health_calls = []
        key_reads = []
        resolve_calls = []
        entries = [{"provider": "custom:unconfigured-route", "model": "ordinary-model"}]

        def route_fact(provider):
            route_reads.append(provider)
            return None

        def health_base_url(provider, explicit_base_url=None):
            health_calls.append((provider, explicit_base_url))
            return str(explicit_base_url or "")

        def resolve_key(entry):
            key_reads.append(entry["provider"])
            return "safe-key"

        def resolve_client(provider, model, **_kwargs):
            resolve_calls.append((provider, model))
            return MagicMock(), model

        with patch(
            "hermes_cli.config.load_config_readonly", return_value={}
        ), patch(
            "hermes_cli.fallback_config.get_fallback_chain", return_value=entries
        ), patch(
            "agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": entries}
        ), patch(
            "hermes_cli.runtime_provider_custom.peek_named_custom_provider_route", side_effect=route_fact
        ), patch(
            "agent.auxiliary_client._custom_health_base_url", side_effect=health_base_url
        ), patch(
            "hermes_cli.fallback_config.resolve_entry_api_key", side_effect=resolve_key
        ), patch(
            "agent.auxiliary_client.resolve_provider_client", side_effect=resolve_client
        ), patch(
            "agent.auxiliary_client._is_provider_unhealthy", return_value=False
        ):
            from agent.auxiliary_client import _try_configured_fallback_chain, _try_main_fallback_chain

            assert _try_main_fallback_chain("title_generation") == (None, None, "")
            assert _try_configured_fallback_chain("title_generation", "primary") == (None, None, "")

        assert route_reads == ["custom:unconfigured-route", "custom:unconfigured-route"]
        assert health_calls == []
        assert key_reads == []
        assert resolve_calls == []

    def test_main_fallback_rejects_named_custom_google_route_before_key_resolution(self):
        """A named custom alias is denied before health, key, or client resolution."""
        benign_client = MagicMock()
        key_reads = []
        resolve_calls = []
        route_reads = []
        health_calls = []

        def route_fact(provider):
            route_reads.append(provider)
            if provider == "google-relay":
                return {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai"}
            return None

        def resolve_key(entry):
            key_reads.append(entry["provider"])
            return "safe-key"

        def resolve_client(provider, model, **_kwargs):
            resolve_calls.append((provider, model))
            return benign_client, model

        def health_base_url(provider, explicit_base_url=None):
            health_calls.append((provider, explicit_base_url))
            return str(explicit_base_url or "")

        with patch(
            "hermes_cli.config.load_config_readonly", return_value={}
        ), patch(
            "hermes_cli.fallback_config.get_fallback_chain", return_value=[
                {"provider": "google-relay", "model": "ordinary-model"},
                {"provider": "nous", "model": "hermes-4"},
            ]
        ), patch(
            "hermes_cli.runtime_provider_custom.peek_named_custom_provider_route", side_effect=route_fact
        ), patch(
            "hermes_cli.fallback_config.resolve_entry_api_key", side_effect=resolve_key
        ), patch(
            "agent.auxiliary_client.resolve_provider_client", side_effect=resolve_client
        ), patch(
            "agent.auxiliary_client._custom_health_base_url", side_effect=health_base_url
        ), patch(
            "agent.auxiliary_client._is_provider_unhealthy", return_value=False
        ):
            from agent.auxiliary_client import _try_main_fallback_chain

            client, model, provider = _try_main_fallback_chain("title_generation")

        assert (client, model, provider) == (benign_client, "hermes-4", "nous")
        assert route_reads == ["google-relay", "nous"]
        assert key_reads == ["nous"]
        assert resolve_calls == [("nous", "hermes-4")]
        assert health_calls == [("nous", None)]

    def test_configured_fallback_rejects_named_custom_google_route_before_key_resolution(self):
        """Task fallback admission denies named Google routes before health lookup."""
        benign_client = MagicMock()
        key_reads = []
        resolve_calls = []
        route_reads = []
        health_calls = []

        def route_fact(provider):
            route_reads.append(provider)
            if provider == "google-relay":
                return {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai"}
            return None

        def resolve_key(entry):
            key_reads.append(entry["provider"])
            return "safe-key"

        def resolve_client(provider, model, **_kwargs):
            resolve_calls.append((provider, model))
            return benign_client, model

        def health_base_url(provider, explicit_base_url=None):
            health_calls.append((provider, explicit_base_url))
            return str(explicit_base_url or "")

        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": [
                {"provider": "google-relay", "model": "ordinary-model"},
                {"provider": "nous", "model": "hermes-4"},
            ]}
        ), patch(
            "hermes_cli.runtime_provider_custom.peek_named_custom_provider_route", side_effect=route_fact
        ), patch(
            "hermes_cli.fallback_config.resolve_entry_api_key", side_effect=resolve_key
        ), patch(
            "agent.auxiliary_client.resolve_provider_client", side_effect=resolve_client
        ), patch(
            "agent.auxiliary_client._custom_health_base_url", side_effect=health_base_url
        ), patch(
            "agent.auxiliary_client._is_provider_unhealthy", return_value=False
        ):
            from agent.auxiliary_client import _try_configured_fallback_chain

            client, model, provider = _try_configured_fallback_chain("title_generation", "primary")

        assert (client, model, provider) == (benign_client, "hermes-4", "fallback_chain[1](nous)")
        assert route_reads == ["google-relay", "nous"]
        assert key_reads == ["nous"]
        assert resolve_calls == [("nous", "hermes-4")]
        assert health_calls == [("nous", None)]

    def test_main_fallback_rejects_named_malformed_port_before_health_key_or_client(self):
        """A named endpoint fact with an invalid port is rejected before resolution."""
        health_calls = []
        key_reads = []
        resolve_calls = []

        def health_base_url(provider, explicit_base_url=None):
            health_calls.append((provider, explicit_base_url))
            return str(explicit_base_url or "")

        def resolve_key(entry):
            key_reads.append(entry["provider"])
            return "safe-key"

        def resolve_client(provider, model, **_kwargs):
            resolve_calls.append((provider, model))
            return MagicMock(), model

        with patch(
            "hermes_cli.config.load_config_readonly", return_value={}
        ), patch(
            "hermes_cli.fallback_config.get_fallback_chain", return_value=[
                {"provider": "malformed-relay", "model": "ordinary-model"},
            ]
        ), patch(
            "hermes_cli.runtime_provider_custom.peek_named_custom_provider_route",
            return_value={"base_url": "https://relay.example.test:not-a-port/v1"},
        ), patch(
            "hermes_cli.fallback_config.resolve_entry_api_key", side_effect=resolve_key
        ), patch(
            "agent.auxiliary_client.resolve_provider_client", side_effect=resolve_client
        ), patch(
            "agent.auxiliary_client._custom_health_base_url", side_effect=health_base_url
        ), patch(
            "agent.auxiliary_client._is_provider_unhealthy", return_value=False
        ):
            from agent.auxiliary_client import _try_main_fallback_chain

            assert _try_main_fallback_chain("title_generation") == (None, None, "")

        assert health_calls == []
        assert key_reads == []
        assert resolve_calls == []

    def test_configured_fallback_rejects_named_malformed_port_before_health_key_or_client(self):
        """A task fallback rejects malformed named endpoints before resolution."""
        health_calls = []
        key_reads = []
        resolve_calls = []

        def health_base_url(provider, explicit_base_url=None):
            health_calls.append((provider, explicit_base_url))
            return str(explicit_base_url or "")

        def resolve_key(entry):
            key_reads.append(entry["provider"])
            return "safe-key"

        def resolve_client(provider, model, **_kwargs):
            resolve_calls.append((provider, model))
            return MagicMock(), model

        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": [
                {"provider": "custom", "model": "no-endpoint-model"},
                {"provider": "malformed-relay", "model": "ordinary-model"},
            ]}
        ), patch(
            "hermes_cli.runtime_provider_custom.peek_named_custom_provider_route",
            return_value={"base_url": "https://relay.example.test:65536/v1"},
        ), patch(
            "hermes_cli.fallback_config.resolve_entry_api_key", side_effect=resolve_key
        ), patch(
            "agent.auxiliary_client.resolve_provider_client", side_effect=resolve_client
        ), patch(
            "agent.auxiliary_client._custom_health_base_url", side_effect=health_base_url
        ), patch(
            "agent.auxiliary_client._is_provider_unhealthy", return_value=False
        ):
            from agent.auxiliary_client import _try_configured_fallback_chain

            assert _try_configured_fallback_chain("title_generation", "primary") == (None, None, "")

        assert health_calls == []
        assert key_reads == []
        assert resolve_calls == []

    def test_resolve_provider_auto_returns_runtime_model_not_stale_config_default(self):
        """Blank auto aux requests must not pair a stale config model with live fallback provider."""
        runtime_client = MagicMock()
        with patch(
            "agent.auxiliary_client._read_main_model",
            return_value="claude-opus-4-8",
        ) as mock_read_main_model, patch(
            "agent.auxiliary_client._resolve_auto_route",
            return_value=(runtime_client, "gpt-5.5", "openai-codex"),
        ) as mock_resolve_auto:
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client(
                "auto",
                main_runtime={
                    "provider": "openai-codex",
                    "model": "gpt-5.5",
                    "base_url": "",
                    "api_key": "",
                    "api_mode": "codex_responses",
                },
            )

        assert client is runtime_client
        assert model == "gpt-5.5"
        mock_read_main_model.assert_not_called()
        mock_resolve_auto.assert_called_once()

    def test_runtime_base_url_passed_for_named_api_key_provider(self):
        """Named API-key providers inherit the live session endpoint for aux work."""
        token_plan_url = "https://token-plan-sgp.xiaomimimo.com/v1"
        with patch(
            "agent.auxiliary_client._read_main_provider",
            return_value="openrouter",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="config-model",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_resolve.return_value = (MagicMock(), "mimo-v2.5-pro")

            from agent.auxiliary_client import _resolve_auto_route

            _resolve_auto_route(main_runtime={
                "provider": "xiaomi",
                "model": "mimo-v2.5-pro",
                "base_url": token_plan_url,
                "api_key": "tp-test-key",
                "api_mode": "chat_completions",
            })

        assert mock_resolve.call_args.args[0] == "xiaomi"
        assert mock_resolve.call_args.args[1] == "mimo-v2.5-pro"
        assert mock_resolve.call_args.kwargs["explicit_base_url"] == token_plan_url
        assert mock_resolve.call_args.kwargs["explicit_api_key"] == "tp-test-key"
        assert mock_resolve.call_args.kwargs["api_mode"] == "chat_completions"


# ── Vision — resolve_vision_provider_client ─────────────────────────────────


class TestResolveVisionMainFirst:
    """Vision auto-detection prefers the main provider first."""

    def test_openrouter_main_vision_uses_main_model(self, monkeypatch):
        """OpenRouter main with vision-capable model → aux vision uses main model."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="openrouter",
        ), patch(
            "agent.auxiliary_client._read_main_model",
            return_value="anthropic/claude-sonnet-4.6",
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve, patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ):
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "anthropic/claude-sonnet-4.6")

            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "openrouter"
        assert client is mock_client
        assert model == "anthropic/claude-sonnet-4.6"
        # Verify it did NOT call the strict vision backend for OpenRouter
        # (which would have used a cheap gemini-flash-preview default)
        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.args[0] == "openrouter"
        assert mock_resolve.call_args.args[1] == "anthropic/claude-sonnet-4.6"
        assert mock_resolve.call_args.kwargs.get("is_vision") is True




    @staticmethod
    def _stub_nous_portal(seen: dict):
        """Stub the Nous network boundary, keeping the resolution chain real.

        Returns a ``_try_nous`` replacement that answers with the Portal's
        tier-aware slots: a vision model for ``vision=True``, the text chat
        default otherwise.
        """
        nous_client = MagicMock()
        nous_client.api_key = "jwt-test"
        nous_client.base_url = "https://inference-api.nousresearch.com/v1"

        def fake_try_nous(vision=False):
            seen["vision"] = vision
            return nous_client, (
                "stepfun/step-3.7-flash:free" if vision else "tencent/hy3:free"
            )

        return nous_client, fake_try_nous

    def test_nous_main_vision_uses_portal_pick_not_text_chat_model(self):
        """Nous main → vision runs the Portal's vision slot, not the chat model.

        A Nous chat default is routinely text-only (e.g. a ``:free`` chat SKU).
        Letting it reach the vision lane means the image goes to a model that
        cannot accept one and the Portal 404s. Only the Nous network boundary
        is stubbed — the strict vision backend, the provider router, and its
        missing-model pre-fill all run for real, because that pre-fill is where
        the chat model used to clobber the Portal's pick.
        """
        seen: dict = {}
        nous_client, fake_try_nous = self._stub_nous_portal(seen)

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="nous",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="tencent/hy3:free",
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client._try_nous", side_effect=fake_try_nous,
        ):
            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "nous"
        assert client is nous_client
        assert seen["vision"] is True
        assert model == "stepfun/step-3.7-flash:free"

    def test_nous_main_vision_honours_explicit_vision_model(self):
        """An explicit auxiliary.vision.model still overrides the Portal pick."""
        seen: dict = {}
        _nous_client, fake_try_nous = self._stub_nous_portal(seen)

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="nous",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="tencent/hy3:free",
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "qwen/qwen3-vl-8b-instruct", None, None, None),
        ), patch(
            "agent.auxiliary_client._try_nous", side_effect=fake_try_nous,
        ):
            from agent.auxiliary_client import resolve_vision_provider_client

            provider, _client, model = resolve_vision_provider_client()

        assert provider == "nous"
        assert model == "qwen/qwen3-vl-8b-instruct"

    def test_nous_explicit_vision_provider_also_skips_chat_model(self):
        """``auxiliary.vision.provider: nous`` takes the same Portal pick.

        The explicit-provider branch reaches the strict vision backend with no
        model too, so it has to resolve the same way the auto branch does.
        """
        seen: dict = {}
        nous_client, fake_try_nous = self._stub_nous_portal(seen)

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="nous",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="tencent/hy3:free",
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("nous", None, None, None, None),
        ), patch(
            "agent.auxiliary_client._try_nous", side_effect=fake_try_nous,
        ):
            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "nous"
        assert client is nous_client
        assert model == "stepfun/step-3.7-flash:free"

    def test_nous_text_aux_still_uses_main_chat_model(self):
        """The vision carve-out must not leak into text aux resolution.

        Text auxiliary work on a Nous main deliberately keeps the user's chat
        model rather than dropping to the Portal's cheap default.
        """
        seen: dict = {}
        _nous_client, fake_try_nous = self._stub_nous_portal(seen)

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="nous",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="tencent/hy3:free",
        ), patch(
            "agent.auxiliary_client._try_nous", side_effect=fake_try_nous,
        ):
            from agent.auxiliary_client import resolve_provider_client

            _client, model = resolve_provider_client("nous")

        assert model == "tencent/hy3:free"

    def test_copilot_vision_sets_vision_header(self, monkeypatch):
        """Copilot vision requests include the header required for vision routing."""
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghu_test-token")

        captured = {}

        def fake_headers(*, is_agent_turn=False, is_vision=False):
            captured["is_agent_turn"] = is_agent_turn
            captured["is_vision"] = is_vision
            return {"Copilot-Vision-Request": "true"} if is_vision else {}

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="copilot",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="configured-copilot-model",
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client.OpenAI",
        ) as mock_openai, patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "copilot",
                "api_key": "copilot-api-token",
                "base_url": "https://api.githubcopilot.com",
            },
        ), patch(
            "hermes_cli.copilot_auth.copilot_request_headers",
            side_effect=fake_headers,
        ):
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "copilot"
        assert client is mock_client
        assert model == "configured-copilot-model"
        assert captured == {"is_agent_turn": True, "is_vision": True}
        assert mock_openai.call_args.kwargs["default_headers"]["Copilot-Vision-Request"] == "true"

    def test_text_copilot_does_not_set_vision_header(self, monkeypatch):
        """Text Copilot requests keep the vision-only header off."""
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghu_test-token")

        captured = {}

        def fake_headers(*, is_agent_turn=False, is_vision=False):
            captured["is_agent_turn"] = is_agent_turn
            captured["is_vision"] = is_vision
            return {"Copilot-Vision-Request": "true"} if is_vision else {}

        with patch(
            "agent.auxiliary_client.OpenAI",
        ) as mock_openai, patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "copilot",
                "api_key": "copilot-api-token",
                "base_url": "https://api.githubcopilot.com",
            },
        ), patch(
            "hermes_cli.copilot_auth.copilot_request_headers",
            side_effect=fake_headers,
        ):
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client("copilot", "gpt-5-mini")

        assert client is mock_client
        assert model == "gpt-5-mini"
        assert captured == {"is_agent_turn": True, "is_vision": False}
        assert "default_headers" not in mock_openai.call_args.kwargs




# ── Vision — custom provider endpoint credential passthrough ────────────────


class TestResolveVisionCustomProvider:
    """Custom-endpoint mains must forward base_url/api_key to Step 1.

    Regression: a ``custom:<name>`` main provider resolves to the bare
    runtime provider id ``"custom"``.  ``resolve_provider_client("custom")``
    has no built-in endpoint, so without forwarding the live base_url/api_key
    it returns ``(None, None)`` and vision falls through to OpenRouter / Nous,
    which an offline / aggregator-less user has never configured — breaking
    vision entirely with ``No LLM provider configured for task=vision
    provider=auto``.  The fix recovers the live endpoint that
    ``set_runtime_main()`` recorded for the turn.
    """

    def test_custom_main_forwards_runtime_endpoint(self, monkeypatch):
        """custom main with recorded runtime endpoint → Step 1 builds a client."""
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "_RUNTIME_MAIN_BASE_URL", "https://my.endpoint.example/v1")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_KEY", "sk-runtime-key")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_MODE", "anthropic_messages")

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="custom",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="claude-opus-4-8",
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "claude-opus-4-8")

            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "custom"
        assert client is mock_client
        assert model == "claude-opus-4-8"
        # The endpoint credentials recorded for the turn MUST be forwarded,
        # otherwise resolve_provider_client("custom") returns (None, None).
        kwargs = mock_resolve.call_args.kwargs
        assert kwargs.get("explicit_base_url") == "https://my.endpoint.example/v1"
        assert kwargs.get("explicit_api_key") == "sk-runtime-key"
        assert kwargs.get("is_vision") is True

    def test_custom_prefixed_main_forwards_runtime_endpoint(self, monkeypatch):
        """A ``custom:<name>`` provider id also forwards the runtime endpoint."""
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "_RUNTIME_MAIN_BASE_URL", "https://named.example/v1")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_KEY", "sk-named")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_MODE", "")

        with patch(
            "agent.auxiliary_client._read_main_provider",
            return_value="custom:copilot-gateway",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="claude-opus-4-8",
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "claude-opus-4-8")

            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "custom:copilot-gateway"
        assert client is mock_client
        kwargs = mock_resolve.call_args.kwargs
        assert kwargs.get("explicit_base_url") == "https://named.example/v1"
        assert kwargs.get("explicit_api_key") == "sk-named"
        assert kwargs.get("is_vision") is True

    def test_custom_main_no_runtime_falls_back_to_configured_endpoint(self, monkeypatch):
        """No recorded runtime endpoint → resolve the configured custom endpoint."""
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "_RUNTIME_MAIN_BASE_URL", "")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_KEY", "")
        monkeypatch.setattr(aux, "_RUNTIME_MAIN_API_MODE", "")

        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="custom",
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="claude-opus-4-8",
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client._resolve_custom_runtime",
            return_value=("https://configured.example/v1", "sk-configured", "chat_completions"),
        ), patch(
            "agent.auxiliary_client.resolve_provider_client"
        ) as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "claude-opus-4-8")

            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert client is mock_client
        kwargs = mock_resolve.call_args.kwargs
        assert kwargs.get("explicit_base_url") == "https://configured.example/v1"
        assert kwargs.get("explicit_api_key") == "sk-configured"


# ── Constant cleanup ────────────────────────────────────────────────────────


def test_aggregator_providers_constant_removed():
    """The dead _AGGREGATOR_PROVIDERS constant should no longer live in the module.

    Removed when the main-first policy made the aggregator-skip guard obsolete.
    """
    import agent.auxiliary_client as aux_mod

    assert not hasattr(aux_mod, "_AGGREGATOR_PROVIDERS"), (
        "_AGGREGATOR_PROVIDERS was removed when _resolve_auto_route stopped "
        "treating aggregators specially. If you re-added it, the main-first "
        "policy may have regressed."
    )
