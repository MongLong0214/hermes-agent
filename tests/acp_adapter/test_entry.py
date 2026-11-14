"""Tests for acp_adapter.entry startup wiring."""

import os
import sys
from pathlib import Path

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


def test_main_skips_configured_mcp_discovery_when_requested(monkeypatch):
    discovery_calls = []

    async def fake_run_agent(agent, **kwargs):
        pass

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    monkeypatch.setattr(
        "tools.mcp_tool_discovery.discover_mcp_tools",
        lambda: discovery_calls.append(True),
    )
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert discovery_calls == []


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


def _run_main(monkeypatch, env_value):
    discovery_calls, agent_calls = [], []

    async def fake_run_agent(agent, **kwargs):
        agent_calls.append((agent, kwargs))

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    for target in ("tools.mcp_tool_discovery.discover_mcp_tools",
                   "hermes_cli.mcp_startup.start_background_mcp_discovery"):
        monkeypatch.setattr(target, lambda *_a, **_k: discovery_calls.append(True))
    if env_value is None:
        monkeypatch.delenv("HERMES_ACP_SKIP_ENV_LOAD", raising=False)
    else:
        monkeypatch.setenv("HERMES_ACP_SKIP_ENV_LOAD", env_value)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert discovery_calls == []  # MCP belongs to the gateway daemon either way
    assert len(agent_calls) == 1
    return agent_calls


@pytest.mark.parametrize("env_value", [pytest.param("1", id="exact"), pytest.param(" 1 ", id="padded")])
def test_main_dark_mode_skips_dotenv_and_mcp_but_runs_agent(monkeypatch, secret_profile_env, env_value):
    _run_main(monkeypatch, env_value)

    assert _CANARY_KEY not in os.environ


@pytest.mark.parametrize(
    "env_value",
    [pytest.param(None, id="missing"), pytest.param("0", id="zero"), pytest.param("true", id="nonliteral")],
)
def test_main_loads_dotenv_unless_dark_value_is_exactly_1(monkeypatch, secret_profile_env, env_value):
    _run_main(monkeypatch, env_value)

    assert os.environ.get(_CANARY_KEY) == _CANARY_VALUE










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
    import pm

    def refuse(name, **kwargs):
        raise pm.InstallError(name, "download failed")

    monkeypatch.setattr(pm, "ensure", refuse)

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--setup-browser"])
    assert excinfo.value.code == 1


def test_setup_browser_is_one_explicit_package_request(monkeypatch):
    import pm

    calls = []
    monkeypatch.setattr(pm, "ensure", lambda name, **kwargs: calls.append((name, kwargs)))

    entry.main(["--setup-browser", "--yes"])

    assert calls == [("chromium", {"explicit": True})]


@pytest.mark.platforms("any")
@pytest.mark.parametrize("flags, profile", [([], "work"), (["-p", "other"], "other"), (["-p", "default"], None)],
                         ids=["sticky", "flag", "flag-default"])
def test_console_script_runs_in_the_selected_profile(tmp_path, flags, profile):
    """`hermes-acp` starts in entry.main() without hermes_cli.main, and must still serve the profile
    `hermes acp` would pick: the sticky `hermes profile use` choice, or `-p`."""
    import os
    import subprocess

    local_appdata = tmp_path / "AppData" / "Local"
    root = local_appdata / "hermes" if sys.platform == "win32" else tmp_path / ".hermes"
    for name in ("work", "other"):
        (root / "profiles" / name).mkdir(parents=True)
        (root / "profiles" / name / "config.yaml").write_text("{}\n", encoding="utf-8")
    (root / "active_profile").write_text("work\n", encoding="utf-8")
    driver = (
        "import sys, acp\n"
        "async def fake_run_agent(agent, **kw):\n"
        "    from hermes_constants import get_hermes_home\n"
        "    print('SERVED', get_hermes_home())\n"
        "acp.run_agent = fake_run_agent\n"
        f"sys.argv = ['hermes-acp', *{flags!r}]\n"
        "from acp_adapter.entry import main\n"
        "main()\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("HERMES_HOME", "HERMES_DATA_DIR_SUFFIX")}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path), LOCALAPPDATA=str(local_appdata),
               PYTHONPATH=str(Path(__file__).resolve().parents[2]), HERMES_ACP_SKIP_CONFIGURED_MCP="1")
    out = subprocess.run([sys.executable, "-c", driver], env=env, cwd=tmp_path, capture_output=True,
                         text=True, timeout=120, check=False)

    served = [line for line in out.stdout.splitlines() if line.startswith("SERVED ")]
    assert served == [f"SERVED {root / 'profiles' / profile if profile else root}"], out.stdout + out.stderr


_LAUNCHES = {
    "hermes-acp": "sys.argv = ['hermes-acp']\nfrom acp_adapter.entry import main\nmain()\n",
    "python -m acp_adapter": "import runpy\nsys.argv = ['acp_adapter']\nrunpy.run_module('acp_adapter', run_name='__main__')\n",
    "hermes acp": "sys.argv = ['hermes', 'acp']\nfrom hermes_cli.main import main\nmain()\n",
    # Another hermes command in a process that imported the adapter (the variable leaked to it).
    "hermes status after the adapter": (
        "import acp_adapter.entry\nsys.argv = ['hermes', 'status']\nimport hermes_cli.main\n"
        "print('SENTINEL', os.environ.get('ACP_DARK_SENTINEL'))\n"
    ),
}


def _sentinel_at_startup(tmp_path, launch: str, dark: bool) -> str:
    """Start ACP through a real launch path in a fresh process (scratch HOME whose profile .env
    holds a sentinel) and report what the sentinel is when the agent starts serving."""
    import subprocess

    (tmp_path / ".hermes").mkdir()
    (tmp_path / ".hermes" / ".env").write_text("ACP_DARK_SENTINEL=from-dotenv\n", encoding="utf-8")
    driver = (
        "import os, sys, acp\n"
        "async def fake_run_agent(agent, **kw):\n"
        "    print('SENTINEL', os.environ.get('ACP_DARK_SENTINEL'))\n"
        "acp.run_agent = fake_run_agent\n"
    ) + _LAUNCHES[launch]
    env = {k: v for k, v in os.environ.items()
           if k not in ("HERMES_HOME", "HERMES_DATA_DIR_SUFFIX", "ACP_DARK_SENTINEL", "HERMES_ACP_SKIP_ENV_LOAD")}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path), LOCALAPPDATA=str(tmp_path / "AppData" / "Local"),
               PYTHONPATH=str(Path(__file__).resolve().parents[2]), HERMES_ACP_SKIP_CONFIGURED_MCP="1")
    if dark:
        env["HERMES_ACP_SKIP_ENV_LOAD"] = "1"
    out = subprocess.run([sys.executable, "-c", driver], env=env, cwd=tmp_path, capture_output=True,
                         text=True, timeout=120, check=False)
    served = [line for line in out.stdout.splitlines() if line.startswith("SENTINEL ")]
    assert len(served) == 1, out.stdout + out.stderr
    return served[0].split(" ", 1)[1]


@pytest.mark.parametrize("launch", ["hermes-acp", "python -m acp_adapter"])
def test_a_dark_console_launch_never_reads_the_profile_dotenv(tmp_path, launch):
    assert _sentinel_at_startup(tmp_path, launch, dark=True) == "None"


@pytest.mark.parametrize(
    ("launch", "dark"),
    [("hermes-acp", False), ("python -m acp_adapter", False), ("hermes acp", True),
     ("hermes status after the adapter", True)],
)
def test_other_launches_still_read_the_profile_dotenv(tmp_path, launch, dark):
    # Unset, the load stands; ``hermes acp`` loads it before the adapter starts, as in the fork, and
    # a non-ACP command never goes dark.
    assert _sentinel_at_startup(tmp_path, launch, dark=dark) == "from-dotenv"
