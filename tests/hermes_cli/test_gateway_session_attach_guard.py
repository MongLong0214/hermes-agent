from argparse import Namespace
import json
import sqlite3
import subprocess
from types import SimpleNamespace

import pytest


def _chat_args(**overrides):
    values = {"continue_last": None, "in_dir": None, "resume": "routed-session"}
    values.update(overrides)
    return Namespace(**values)


def test_cmd_chat_refuses_routed_session_even_with_legacy_environment_override(
    monkeypatch, capsys
):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda _args: False)
    monkeypatch.setattr(main_mod, "_apply_safe_mode", lambda _args: None)
    monkeypatch.setattr(main_mod, "_resolve_continue_arg", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_mod, "_gateway_routed_session_owner", lambda _session_id: 4242)
    monkeypatch.setattr(
        main_mod,
        "_resolve_session_by_name_or_id",
        lambda _session_id: pytest.fail("legacy environment override bypassed refusal"),
    )
    monkeypatch.setenv("HERMES_ALLOW_GATEWAY_SESSION", "1")

    with pytest.raises(SystemExit) as excinfo:
        main_mod.cmd_chat(_chat_args())

    assert excinfo.value.code == 3
    assert capsys.readouterr().err == (
        "Refusing to attach to a session currently served by the gateway.\n"
    )


def test_gateway_session_attach_refusal_allows_when_no_live_owner(monkeypatch, capsys):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(main_mod, "_gateway_routed_session_owner", lambda _session_id: None)

    assert main_mod._refuse_gateway_routed_session_attach("routed-session") is None
    assert capsys.readouterr().err == ""


def test_gateway_routed_session_owner_reads_state_db_from_hermes_home(
    tmp_path, monkeypatch
):
    import hermes_cli.main as main_mod

    home = tmp_path / "profile"
    home.mkdir()
    with sqlite3.connect(home / "state.db") as conn:
        conn.execute("CREATE TABLE gateway_routing (entry_json TEXT)")
        conn.execute(
            "INSERT INTO gateway_routing VALUES (?)",
            (json.dumps({"session_id": "routed-session"}),),
        )

    def fake_run(command, **_kwargs):
        assert command == ["launchctl", "list", "ai.hermes.gateway"]
        return SimpleNamespace(stdout='"PID" = 4242;')

    from gateway import status as gateway_status

    pid_probe_calls = []

    def fake_pid_exists(pid):
        pid_probe_calls.append(pid)
        return True

    def fail_if_kill_called(*_args):
        pytest.fail("os.kill must not be used for gateway PID liveness")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(gateway_status, "_pid_exists", fake_pid_exists)
    monkeypatch.setattr(main_mod.os, "kill", fail_if_kill_called)

    assert main_mod._gateway_routed_session_owner("routed-session") == 4242
    assert pid_probe_calls == [4242]


def test_main_sessions_stats_help_builds_full_parser_with_rollback_registration(monkeypatch):
    import os
    import site
    import sys

    import hermes_cli.main as main_mod

    source_root = os.path.dirname(os.path.dirname(main_mod.__file__))
    script = """
import sys
import hermes_cli.main as main_mod

sys.argv = ["hermes", "sessions", "stats", "--help"]
main_mod._set_process_title = lambda: None
main_mod._advertise_agent_env = lambda: None
main_mod._cleanup_quarantined_exes = lambda: None
main_mod._sweep_stale_bytecode_if_checkout_changed = lambda: None
main_mod._recover_from_interrupted_install = lambda: None
main_mod._warn_pending_fleet_restart_on_startup = lambda: None
main_mod._try_termux_fast_tui_launch = lambda: False
main_mod._try_termux_fast_cli_launch = lambda: False
main_mod._try_fast_chat_launch = lambda: False
main_mod.main()
"""
    env = os.environ | {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join([source_root, *site.getsitepackages()]),
    }

    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage: hermes sessions stats" in result.stdout
