"""Tests for Vertex AI runtime-provider resolution and profile registration.

Covers: provider-profile registration + aliases, and the Google outbound
denial: resolve_runtime_provider(vertex) is refused before any credential is
read or token minted. No network calls.
"""

from __future__ import annotations

import pytest

from agent.gemini_outbound_policy import GeminiOutboundDenied


def test_resolve_runtime_provider_refuses_vertex_before_credentials(monkeypatch):
    import agent.vertex_adapter as va
    from hermes_cli import runtime_provider as rp

    touched = []
    monkeypatch.setattr(va, "get_vertex_config", lambda *a, **k: touched.append("get_vertex_config"))
    with pytest.raises(GeminiOutboundDenied):
        rp.resolve_runtime_provider(requested="vertex")
    assert touched == []


def test_vertex_registered_in_provider_registry():
    """PROVIDER_REGISTRY (hermes_cli.auth) is what agent/auxiliary_client.py's
    resolve_provider_client() looks up before dispatching on auth_type. Without
    an entry here, the ``elif pconfig.auth_type == "vertex":`` branch there is
    unreachable dead code — every auxiliary Vertex call (vision, title
    generation, MoA reference/aggregator slots, ...) fails at the
    ``pconfig is None`` guard before ever reaching it."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    cfg = PROVIDER_REGISTRY.get("vertex")
    assert cfg is not None
    assert cfg.auth_type == "vertex"


def test_vertex_registered_in_hermes_overlays():
    """hermes_cli.providers.get_provider("vertex") backs
    _preserve_provider_with_base_url() in agent/auxiliary_client.py, which
    decides whether a MoA slot's resolved Vertex (base_url, api_key) pair
    keeps its "vertex" provider identity or silently collapses to "custom" —
    losing the identity _refresh_provider_credentials() needs to re-mint an
    expired OAuth2 token on a 401."""
    from hermes_cli.providers import get_provider

    resolved = get_provider("vertex")
    assert resolved is not None
    assert resolved.auth_type == "vertex"


class _CountingCreds:
    """google-auth Credentials stand-in: every refresh() is one OAuth JWT exchange (a mint)."""

    def __init__(self, mints: list):
        self.token, self.expired, self.expiry, self._mints = None, False, None, mints

    def refresh(self, _request):
        self._mints.append(1)
        self.token = f"ya29.minted-{len(self._mints)}"


def _vertex_cli_shell(monkeypatch, tmp_path, *, loadable: bool):
    import types

    import agent.vertex_adapter as va
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    mints: list = []
    sa = tmp_path / "sa.json"
    sa.write_text('{"project_id": "sa-proj"}')
    va._creds_cache.clear()
    monkeypatch.setattr(va, "_ensure_google_auth", lambda: True)
    request_mod = types.SimpleNamespace(Request=lambda: None)
    monkeypatch.setattr(va, "google", types.SimpleNamespace(auth=types.SimpleNamespace(
        transport=types.SimpleNamespace(requests=request_mod))), raising=False)
    monkeypatch.setattr(va, "_resolve_credentials_path", lambda explicit=None: str(sa))
    monkeypatch.setattr(va, "_vertex_config", lambda: {"region": "us-central1"})
    monkeypatch.setattr(va, "_load_credentials",
                        lambda path, raw: (_CountingCreds(mints), "sa-proj") if loadable else None)

    shell = CLIAgentSetupMixin.__new__(CLIAgentSetupMixin)
    shell.model, shell.requested_provider = "google/gemini-3-flash-preview", "vertex"
    shell._explicit_api_key = shell._explicit_base_url = None
    shell._fallback_model = []
    shell.tool_progress_mode = "off"
    shell.agent = shell.api_key = shell.base_url = shell.provider = shell.acp_command = None
    shell.api_mode, shell.acp_args = "chat_completions", []
    shell._normalize_model_for_provider = lambda _provider: False
    shell._maybe_print_free_tier_available_notice = lambda: None
    return shell, mints


def test_gateway_backed_cli_refuses_vertex_route_before_minting(monkeypatch, tmp_path, capsys):
    """A Vertex route is Google outbound: the CLI refuses it with the fixed denial copy and never
    mints an OAuth token, even with loadable service-account credentials."""
    shell, mints = _vertex_cli_shell(monkeypatch, tmp_path, loadable=True)

    assert shell._ensure_runtime_credentials() is False
    assert GeminiOutboundDenied.public_message in capsys.readouterr().err
    assert mints == []
    assert shell.provider is None and shell.base_url is None
