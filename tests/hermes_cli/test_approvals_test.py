"""Tests for ``hermes approvals test`` — dry-run approval verdict CLI.

The tester must compose the REAL runtime evaluators from ``tools.approval``
(detect_hardline_command, _match_user_deny_rule, detect_dangerous_command,
the container-skip gate, and the same ``_command_detection_variants``
normalization/de-obfuscation path) — never reimplement them. It is strictly
read-only: nothing is executed, no prompt fires, nothing is persisted.

Exit-code contract (script-friendly, documented in the CLI help):
    0 = allow, 2 = ask-approval, 3 = deny (hardline / user deny rule).
"""

import argparse
import json

import pytest

import tools.approval as A
from hermes_cli import approvals_test as at


def _args(command, env_type="local", as_json=False):
    return argparse.Namespace(
        command_words=list(command) if isinstance(command, (list, tuple)) else [command],
        env_type=env_type,
        json=as_json,
    )


@pytest.fixture
def isolated_approvals(monkeypatch):
    """Isolate the evaluators from the dev machine's real config/state."""
    monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "manual"})
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(A, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(A, "load_permanent_allowlist", lambda: set())
    saved = set(A._permanent_approved)
    A._permanent_approved.clear()
    # The tester must NEVER prompt or persist — make any attempt explode.
    def _boom(*_a, **_kw):  # pragma: no cover - failure path
        raise AssertionError("read-only tester touched a prompt/persistence path")
    monkeypatch.setattr(A, "prompt_dangerous_approval", _boom)
    monkeypatch.setattr(A, "save_permanent_allowlist", _boom)
    monkeypatch.setattr(A, "submit_pending", _boom, raising=False)
    yield A
    A._permanent_approved.clear()
    A._permanent_approved.update(saved)


class TestVerdicts:
    @pytest.mark.parametrize("command", [
        "git -C /tmp/repo push origin main",
        "G=git; $G push origin main",
        "git push https://github.com/acme/repo.git HEAD:main",
    ])
    def test_default_branch_push_denied_even_when_mode_off(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "hardline-deny"
        assert verdict["exit_code"] == 3

    @pytest.mark.parametrize("command", [
        'echo x > ~/.hermes/config.yaml',
        'python -c "import smtplib; smtplib.SMTP(\"mail.example\").sendmail(\"a\",\"b\",\"x\")"',
        'buzz messages send --channel public --message x',
    ])
    def test_security_write_and_external_send_require_approval_under_mode_off(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "ask-approval"
        assert verdict["exit_code"] == 2
    def test_benign_command_allows_with_exit_0(self, isolated_approvals, capsys):
        rc = at.approvals_test_command(_args(["ls", "-la"]))
        out = capsys.readouterr().out
        assert rc == 0
        assert "allow" in out

    def test_hardline_command_denies_with_rule_name(self, isolated_approvals, capsys):
        rc = at.approvals_test_command(_args(["sudo", "re" + "boot"]))
        out = capsys.readouterr().out
        assert rc == 3
        assert "hardline-deny" in out
        assert "system shutdown/reboot" in out

    def test_dangerous_command_asks_with_exit_2(self, isolated_approvals, capsys):
        rc = at.approvals_test_command(_args(["rm", "-rf", "~/project/build"]))
        out = capsys.readouterr().out
        assert rc == 2
        assert "ask-approval" in out
        assert "recursive delete" in out

    def test_user_deny_rule_from_config_honored(self, isolated_approvals, capsys,
                                                monkeypatch):
        # The command must match the deny glob but NOT the hardline floor:
        # a default-branch push is hardline-denied before user-deny runs
        # (see test_hardline_deny_beats_user_deny_rule), so use a
        # feature-branch push to isolate the user-deny path.
        monkeypatch.setattr(
            A, "_get_approval_config",
            lambda: {"mode": "manual", "deny": ["git push *"]})
        rc = at.approvals_test_command(
            _args(["git", "push", "origin", "feature-branch"]))
        out = capsys.readouterr().out
        assert rc == 3
        assert "user-deny" in out
        assert "git push *" in out

    def test_hardline_deny_beats_user_deny_rule(self, isolated_approvals,
                                                capsys, monkeypatch):
        # Precedence contract: the hardline floor fires BEFORE user
        # approvals.deny rules and is never bypassable. A command matching
        # BOTH the deny glob and the default-branch-push detector must
        # report hardline-deny — the user deny pattern never claims the
        # verdict.
        monkeypatch.setattr(
            A, "_get_approval_config",
            lambda: {"mode": "manual", "deny": ["git push *"]})
        rc = at.approvals_test_command(
            _args(["git", "push", "origin", "main"], as_json=True))
        payload = json.loads(capsys.readouterr().out)
        assert rc == 3
        assert payload["verdict"] == "hardline-deny"
        assert payload["rule"] == "push to protected default branch (main/master)"
        # The user deny glob must NOT be claimed as the verdict's rule.
        assert payload["rule"] != "git push *"

    def test_container_env_type_skips_guards_like_runtime(self, isolated_approvals,
                                                          capsys):
        # Mirrors check_all_command_guards: isolated docker skips BEFORE the
        # hardline floor, so even a catastrophic command reports allow.
        rc = at.approvals_test_command(_args(["rm", "-rf", "/"], env_type="docker"))
        out = capsys.readouterr().out
        assert rc == 0
        assert "allow" in out
        assert "container" in out or "isolated" in out

    def test_mode_off_bypasses_dangerous_but_not_hardline(self, isolated_approvals,
                                                          capsys, monkeypatch):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        rc = at.approvals_test_command(_args(["rm", "-rf", "~/project/build"]))
        out = capsys.readouterr().out
        assert rc == 0
        assert "off" in out
        rc = at.approvals_test_command(_args(["sudo", "re" + "boot"]))
        assert rc == 3


class TestCredentialHardlineRules:
    """Regression coverage for the credential-enumeration hardline rules.

    These three patterns block read access to material the agent never
    needs: bridge private keys, container credential env, and process
    environ. Each must hardline-deny (rc 3) regardless of mode.
    """

    @pytest.mark.parametrize("command", [
        "cat ~/.hermes/bridge/keys/" + "telegram.priv",
        "head -n 5 ~/.hermes/bridge/keys/" + "signing.priv",
        "tail ~/.hermes/bridge/keys/" + "bot.priv",
        "xxd ~/.hermes/bridge/keys/" + "signing.priv",
    ])
    def test_bridge_private_key_read_hardline_denied(self, isolated_approvals,
                                                     capsys, command):
        rc = at.approvals_test_command(_args([command], as_json=True))
        payload = json.loads(capsys.readouterr().out)
        assert rc == 3
        assert payload["verdict"] == "hardline-deny"
        assert payload["rule"] == "read Hermes bridge private key"

    def test_docker_inspect_hardline_denied(self, isolated_approvals, capsys):
        rc = at.approvals_test_command(
            _args(["docker", "inspect", "my-container"], as_json=True))
        payload = json.loads(capsys.readouterr().out)
        assert rc == 3
        assert payload["verdict"] == "hardline-deny"
        assert payload["rule"] == "inspect container credential environment"

    @pytest.mark.parametrize("command", [
        "cat /proc/self/" + "environ",
        "strings /proc/123/" + "environ",
        "head /proc/self/" + "environ",
    ])
    def test_proc_environ_read_hardline_denied(self, isolated_approvals,
                                               capsys, command):
        rc = at.approvals_test_command(_args([command], as_json=True))
        payload = json.loads(capsys.readouterr().out)
        assert rc == 3
        assert payload["verdict"] == "hardline-deny"
        assert payload["rule"] == "read process credential environment"

    @pytest.mark.parametrize("command", [
        "docker ps",
        "cat ~/.hermes/config.yaml",
    ])
    def test_benign_commands_not_hardline_denied(self, isolated_approvals,
                                                 capsys, command):
        # Negative control: non-matching commands must NOT be hardline
        # denied by these patterns — they fall through to allow.
        rc = at.approvals_test_command(_args([command], as_json=True))
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert payload["verdict"] == "allow"
        assert payload["rule"] is None


class TestNormalizationParity:
    """The tester must run the same de-obfuscation path as the runtime."""

    def test_obfuscated_command_matches_plain_verdict(self, isolated_approvals,
                                                      capsys):
        rc_plain = at.approvals_test_command(_args(["rm", "-rf", "/"]))
        out_plain = capsys.readouterr().out
        rc_obf = at.approvals_test_command(_args(["r\\m", "-rf", "/"]))
        out_obf = capsys.readouterr().out
        assert rc_plain == rc_obf == 3
        assert "recursive delete of root filesystem" in out_plain
        assert "recursive delete of root filesystem" in out_obf
        # The trace must show the de-obfuscated form the runtime evaluated.
        assert "rm -rf /" in out_obf

    def test_normalized_trace_shown_when_command_normalizes(self,
                                                            isolated_approvals,
                                                            capsys):
        rc = at.approvals_test_command(_args(['git', 'st""atus']))
        out = capsys.readouterr().out
        assert rc == 0
        assert "git status" in out

    def test_composes_real_runtime_detectors(self, isolated_approvals, capsys,
                                             monkeypatch):
        """Prove the tester calls the real evaluators, not a reimplementation."""
        calls = {}

        def _spy(name, real):
            def wrapper(c):
                calls[name] = c
                return real(c)
            return wrapper

        monkeypatch.setattr(A, "detect_hardline_command",
                            _spy("hardline", A.detect_hardline_command))
        monkeypatch.setattr(A, "detect_dangerous_command",
                            _spy("dangerous", A.detect_dangerous_command))
        monkeypatch.setattr(A, "_match_user_deny_rule",
                            _spy("deny", A._match_user_deny_rule))
        monkeypatch.setattr(A, "_command_detection_variants",
                            _spy("variants", A._command_detection_variants))
        cmd = "rm -rf ~/project/build"
        at.approvals_test_command(_args(cmd.split()))
        capsys.readouterr()
        assert calls.get("hardline") == cmd
        assert calls.get("dangerous") == cmd
        assert calls.get("deny") == cmd
        assert calls.get("variants") == cmd


class TestReadOnly:
    def test_nothing_executed(self, isolated_approvals, capsys, tmp_path):
        sentinel = tmp_path / "must_not_exist"
        rc = at.approvals_test_command(_args(["touch", str(sentinel)]))
        capsys.readouterr()
        assert rc == 0
        assert not sentinel.exists()

    def test_dangerous_command_never_prompts_or_persists(self, isolated_approvals,
                                                         capsys):
        # isolated_approvals wires prompt/persistence to AssertionError; a
        # dangerous command must complete without touching either.
        rc = at.approvals_test_command(_args(["rm", "-rf", "~/project/build"]))
        capsys.readouterr()
        assert rc == 2


class TestOutputAndWiring:
    def test_json_output_is_machine_readable(self, isolated_approvals, capsys):
        rc = at.approvals_test_command(_args(["sudo", "re" + "boot"], as_json=True))
        payload = json.loads(capsys.readouterr().out)
        assert rc == 3
        assert payload["verdict"] == "hardline-deny"
        assert payload["exit_code"] == 3
        assert payload["rule"] == "system shutdown/reboot"
        assert payload["command"] == "sudo re" + "boot"
        assert isinstance(payload["normalized_variants"], list)

    def test_empty_command_is_usage_error(self, isolated_approvals, capsys):
        rc = at.approvals_test_command(_args([]))
        assert rc == 1

    def test_dispatcher_routes_test_subcommand(self, isolated_approvals, capsys):
        from hermes_cli.approvals_suggest import approvals_command
        args = _args(["ls"])
        args.approvals_command = "test"
        rc = approvals_command(args)
        out = capsys.readouterr().out
        assert rc == 0
        assert "allow" in out

    def test_parser_wires_test_subcommand(self, isolated_approvals, capsys):
        from hermes_cli.subcommands.approvals import build_approvals_parser
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        sentinel = []
        build_approvals_parser(sub, cmd_approvals=lambda a: sentinel.append(a) or 0)
        args = parser.parse_args(
            ["approvals", "test", "--env-type", "ssh", "--", "ls", "-la"])
        assert args.approvals_command == "test"
        assert args.env_type == "ssh"
        # argparse REMAINDER keeps the leading "--"; the handler strips it.
        # dest is command_words (NOT command) so main.py's startup path can
        # keep reading args.command as the top-level subcommand name.
        assert args.command_words == ["--", "ls", "-la"]
        # The subparser must NOT claim the "command" dest — main.py's startup
        # path reads args.command as the top-level subcommand name.
        assert getattr(args, "command", None) != ["--", "ls", "-la"]
        args.func(args)
        assert sentinel

    def test_leading_separator_stripped_from_command(self, isolated_approvals,
                                                     capsys):
        rc = at.approvals_test_command(_args(["--", "ls", "-la"]))
        out = capsys.readouterr().out
        assert rc == 0
        assert "ls -la" in out
        assert "-- ls" not in out
