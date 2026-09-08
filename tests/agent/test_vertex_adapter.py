"""Tests for the Vertex AI adapter (agent/vertex_adapter.py).

Vertex uses OAuth2 (short-lived access tokens from a service-account JSON or
ADC), NOT a static API key. These tests mock google-auth entirely — no network
calls — and cover token minting, the config.yaml→env precedence bridge, the
global vs regional base-URL shapes, and the ADC→service-account fallback.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest


def _install_fake_google_auth(monkeypatch, *, adc_ok=True, adc_project="adc-project",
                              sa_project="sa-project", token="ya29.FAKE"):
    """Register a fake google-auth tree in sys.modules and return the module set."""
    ga = types.ModuleType("google.auth")
    gt = types.ModuleType("google.auth.transport")
    gtr = types.ModuleType("google.auth.transport.requests")
    go = types.ModuleType("google.oauth2")
    gsa = types.ModuleType("google.oauth2.service_account")
    gp = types.ModuleType("google")

    gtr.Request = type("Request", (), {})

    class _Creds:
        def __init__(self):
            self.token = None
            self.expiry = None
            self.expired = False

        def refresh(self, req):
            self.token = token

    def _default(scopes=None):
        if not adc_ok:
            raise RuntimeError("Could not automatically determine credentials")
        return _Creds(), adc_project

    ga.default = _default
    ga.transport = gt
    gt.requests = gtr

    class _SA:
        @staticmethod
        def from_service_account_file(path, scopes=None):
            c = _Creds()
            c.project_id = sa_project
            return c

    gsa.Credentials = _SA
    go.service_account = gsa
    gp.auth = ga
    gp.oauth2 = go

    for name, mod in [
        ("google", gp), ("google.auth", ga), ("google.auth.transport", gt),
        ("google.auth.transport.requests", gtr), ("google.oauth2", go),
        ("google.oauth2.service_account", gsa),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return gp


@pytest.fixture
def vertex_adapter(monkeypatch):
    """Fresh vertex_adapter with a fake google-auth and clean caches/env."""
    for var in ("VERTEX_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS",
                "VERTEX_PROJECT_ID", "VERTEX_REGION", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    _install_fake_google_auth(monkeypatch)
    import agent.vertex_adapter as va
    va = importlib.reload(va)
    va._creds_cache.clear()
    # Neutralize config.yaml by default; individual tests re-patch _vertex_config.
    monkeypatch.setattr(va, "_vertex_config", lambda: {})
    return va












def test_has_vertex_credentials_via_config_project(vertex_adapter, monkeypatch):
    monkeypatch.setattr(vertex_adapter, "_vertex_config", lambda: {"project_id": "p"})
    assert vertex_adapter.has_vertex_credentials() is True


def test_has_vertex_credentials_false_when_nothing_set(vertex_adapter):
    assert vertex_adapter.has_vertex_credentials() is False




def test_multiplex_scope_takes_precedence_over_raw_environ(vertex_adapter, monkeypatch):
    """In a multiplex gateway, a profile's own secret scope must win over a
    stale value in process os.environ left behind by another profile's
    dotenv load at boot — otherwise Profile B's turn could resolve Profile
    A's Vertex project (or worse, its credentials file path)."""
    from agent import secret_scope

    monkeypatch.setenv("VERTEX_PROJECT_ID", "other-profile-project")

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"VERTEX_PROJECT_ID": "this-profile-project"})
    try:
        assert vertex_adapter._resolve_project_override() == "this-profile-project"
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


def test_multiplex_unscoped_read_fails_closed(vertex_adapter, monkeypatch):
    """A credential read with no profile scope installed while multiplexing
    is active must raise rather than silently fall back to (possibly another
    profile's) raw os.environ value."""
    from agent import secret_scope

    monkeypatch.setenv("VERTEX_PROJECT_ID", "leaked-project")
    secret_scope.set_multiplex_active(True)
    try:
        with pytest.raises(secret_scope.UnscopedSecretError):
            vertex_adapter._resolve_project_override()
    finally:
        secret_scope.set_multiplex_active(False)


def test_get_vertex_credentials_denies_before_path_cache_adc_refresh_or_config(vertex_adapter, monkeypatch):
    from agent.gemini_outbound_policy import GeminiOutboundDenied

    def explode(*args, **kwargs):
        raise AssertionError("credential resolution must not be contacted")

    class ExplodingCache:
        def get(self, *args, **kwargs):
            raise AssertionError("credential cache get must not be contacted")

        def pop(self, *args, **kwargs):
            raise AssertionError("credential cache pop must not be contacted")

        def __setitem__(self, *args, **kwargs):
            raise AssertionError("credential cache must not be contacted")

    class ExplodingRequest:
        def __init__(self, *args, **kwargs):
            raise AssertionError("credential refresh request must not be created")

    monkeypatch.setattr(vertex_adapter, "_resolve_credentials_path", explode)
    monkeypatch.setattr(vertex_adapter, "_creds_cache", ExplodingCache())
    monkeypatch.setattr(vertex_adapter.service_account.Credentials, "from_service_account_file", explode)
    monkeypatch.setattr(vertex_adapter.google.auth, "default", explode)
    monkeypatch.setattr(vertex_adapter, "_refresh_credentials", explode)
    monkeypatch.setattr(vertex_adapter.google.auth.transport.requests, "Request", ExplodingRequest)
    monkeypatch.setattr(vertex_adapter, "_vertex_config", explode)

    with pytest.raises(GeminiOutboundDenied) as excinfo:
        vertex_adapter.get_vertex_credentials("must-not-be-resolved")

    exc = excinfo.value
    assert type(exc) is GeminiOutboundDenied
    assert str(exc) == "Gemini outbound requests are disabled."
    assert exc.code == "gemini_outbound_denied"
    assert vars(exc) == {}




