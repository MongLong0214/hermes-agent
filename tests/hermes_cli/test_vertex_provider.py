"""Tests for Vertex AI runtime-provider resolution and profile registration.

Covers: provider-profile registration + aliases, alias canonicalization,
resolve_runtime_provider(vertex) minting an OAuth token, and the friendly
AuthError when credentials can't be resolved. No network calls.
"""

from __future__ import annotations

import pytest








def test_resolve_runtime_provider_denies_vertex_before_credential_mint(monkeypatch):
    import agent.vertex_adapter as va
    from agent.gemini_outbound_policy import GeminiOutboundDenied
    from hermes_cli import runtime_provider as rp

    credential_calls = []

    def unexpected_credential_mint():
        credential_calls.append("vertex")
        raise AssertionError("Vertex credential mint was reached")

    monkeypatch.setattr(va, "get_vertex_config", unexpected_credential_mint)
    with pytest.raises(GeminiOutboundDenied) as exc:
        rp.resolve_runtime_provider(requested="vertex")

    assert type(exc.value) is GeminiOutboundDenied
    assert vars(exc.value) == {}
    assert credential_calls == []
    assert exc.value.code == "gemini_outbound_denied"
    assert str(exc.value) == "Gemini outbound requests are disabled."


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
