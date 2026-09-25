"""Tests for acp_adapter.entry startup wiring."""

import os
import sys

import acp
import pytest

from acp_adapter import entry
from hermes_constants import get_hermes_home


def test_main_enables_unstable_protocol(monkeypatch):
    calls = {}

    async def fake_run_agent(agent, **kwargs):
        calls["kwargs"] = kwargs

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert calls["kwargs"]["use_unstable_protocol"] is True


_CANARY_KEY = "ACP_DARK_CANARY_API_KEY"
_CANARY_VALUE = "sk-acp-dark-canary-not-a-real-key"


@pytest.fixture
def secret_profile_env(monkeypatch, tmp_path):
    """Put a secret in this test's temp profile ``.env`` (never the real HERMES_HOME)."""
    home = get_hermes_home()
    assert tmp_path in home.parents
    (home / ".env").write_text(f"{_CANARY_KEY}={_CANARY_VALUE}\n", encoding="utf-8")
    # The real dotenv load writes os.environ directly; registering the key lets monkeypatch drop it after.
    monkeypatch.setenv(_CANARY_KEY, "")
    monkeypatch.delenv(_CANARY_KEY)


@pytest.mark.parametrize(
    "env_value",
    [
        pytest.param("1", id="exact"),
        pytest.param(" 1 ", id="padded"),
    ],
)
def test_main_dark_mode_skips_dotenv_and_configured_mcp_but_runs_agent(
    monkeypatch, secret_profile_env, env_value
):
    discovery_calls = []
    agent_calls = []

    async def fake_run_agent(agent, **kwargs):
        agent_calls.append((agent, kwargs))

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.start_background_mcp_discovery",
        lambda **kwargs: discovery_calls.append(kwargs),
    )
    monkeypatch.setenv("HERMES_ACP_SKIP_ENV_LOAD", env_value)
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert _CANARY_KEY not in os.environ
    assert discovery_calls == []
    assert len(agent_calls) == 1


@pytest.mark.parametrize(
    "env_value",
    [
        pytest.param(None, id="missing"),
        pytest.param("0", id="zero"),
        pytest.param("true", id="nonliteral"),
    ],
)
def test_main_loads_dotenv_unless_dark_value_is_exactly_1(monkeypatch, secret_profile_env, env_value):
    discovery_calls = []
    agent_calls = []

    async def fake_run_agent(agent, **kwargs):
        agent_calls.append((agent, kwargs))

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.start_background_mcp_discovery",
        lambda **kwargs: discovery_calls.append(kwargs),
    )
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    if env_value is None:
        monkeypatch.delenv("HERMES_ACP_SKIP_ENV_LOAD", raising=False)
    else:
        monkeypatch.setenv("HERMES_ACP_SKIP_ENV_LOAD", env_value)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert os.environ.get(_CANARY_KEY) == _CANARY_VALUE
    # The MCP marker is independent of the dotenv marker.
    assert discovery_calls == []
    assert len(agent_calls) == 1










def test_main_setup_offers_browser_install_when_tty(monkeypatch):
    """When stdin is a TTY and the user answers yes, model setup is followed
    by a browser-tools bootstrap call."""
    monkeypatch.setattr("hermes_cli.main.main", lambda: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "y")

    bootstrap_calls = []
    monkeypatch.setattr(
        entry,
        "_run_setup_browser",
        lambda assume_yes=False: bootstrap_calls.append(assume_yes) or 0,
    )

    entry.main(["--setup"])

    assert bootstrap_calls == [False]










def test_main_setup_browser_propagates_browser_failure(monkeypatch):
    """If browser install fails, exit code is 1."""
    def fake_ensure(dep, interactive=True):
        return dep != "browser"  # browser fails

    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", fake_ensure)

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--setup-browser"])
    assert excinfo.value.code == 1
