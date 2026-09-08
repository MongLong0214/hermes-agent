"""Tests for acp_adapter.entry startup wiring."""

import sys

import acp
import pytest

from acp_adapter import entry


def test_main_enables_unstable_protocol(monkeypatch):
    calls = {}

    async def fake_run_agent(agent, **kwargs):
        calls["kwargs"] = kwargs

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert calls["kwargs"]["use_unstable_protocol"] is True


def test_main_dark_mode_skips_dotenv_and_configured_mcp_but_runs_agent(monkeypatch):
    dotenv_calls = []
    discovery_calls = []
    agent_calls = []

    async def fake_run_agent(agent, **kwargs):
        agent_calls.append((agent, kwargs))

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: dotenv_calls.append(True))
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.start_background_mcp_discovery",
        lambda **kwargs: discovery_calls.append(kwargs),
    )
    monkeypatch.setenv("HERMES_ACP_SKIP_ENV_LOAD", "1")
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert dotenv_calls == []
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
def test_main_loads_dotenv_unless_dark_value_is_exactly_1(monkeypatch, env_value):
    dotenv_calls = []
    agent_calls = []

    async def fake_run_agent(agent, **kwargs):
        agent_calls.append((agent, kwargs))

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: dotenv_calls.append(True))
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    if env_value is None:
        monkeypatch.delenv("HERMES_ACP_SKIP_ENV_LOAD", raising=False)
    else:
        monkeypatch.setenv("HERMES_ACP_SKIP_ENV_LOAD", env_value)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert dotenv_calls == [True]
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
