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


_REAL_PROMPT_DANGEROUS_APPROVAL = A.prompt_dangerous_approval


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


class TestProtectedDefaultBranchDetection:
    @pytest.mark.parametrize(
        "command",
        [
            "printf ready $(git push origin main)",
            "printf ready `git push origin master`",
            "sh -c 'git push origin main'",
            'bash -c "git push origin master"',
            'sh -c \'bash -c "git push origin main"\'',
        ],
    )
    def test_command_carried_default_branch_push_fails_closed(
        self, isolated_approvals, command
    ):
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "hardline-deny"
        assert verdict["rule"] == "push to protected default branch (main/master)"

    @pytest.mark.parametrize(
        "command",
        [
            "printf ready | git push origin main",
            "(git push origin master)",
            "{ git push origin main; }",
        ],
    )
    def test_pipeline_or_grouped_default_branch_push_fails_closed(
        self, isolated_approvals, command
    ):
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "hardline-deny"
        assert verdict["rule"] == "push to protected default branch (main/master)"

    @pytest.mark.parametrize(
        "command",
        [
            "sudo -n -u root git -C /tmp/repo push origin main",
            "env -i -- git push origin HEAD:refs/heads/master",
            "command -p git push origin main",
            "git push origin +refs/heads/main",
            "git push origin",
            "git push origin HEAD",
            "git push --all origin",
            "git push --mirror origin",
        ],
    )
    def test_wrapped_forced_or_ambiguous_default_branch_push_fails_closed(
        self, isolated_approvals, command
    ):
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "hardline-deny"
        assert verdict["rule"] == "push to protected default branch (main/master)"

    @pytest.mark.parametrize(
        "command",
        [
            "sudo -n git -C /tmp/repo push --set-upstream origin feature/topic",
            "env -i git push origin HEAD:refs/heads/feature/topic",
            "git push --delete origin old-feature",
            "git push origin +refs/heads/feature/topic",
        ],
    )
    def test_explicit_non_default_branch_push_remains_safe(
        self, isolated_approvals, command
    ):
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "allow"
        assert verdict["rule"] is None

    @pytest.mark.parametrize(
        "command",
        [
            "(git push origin feature/topic)",
            "{ git push origin feature/topic; }",
        ],
    )
    def test_grouped_explicit_non_default_branch_push_remains_safe(
        self, isolated_approvals, command
    ):
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "allow"
        assert verdict["rule"] is None

    @pytest.mark.parametrize(
        "command",
        [
            "printf '%s' 'git push origin main'",
            "printf '%s' '$(git push origin main)'",
            "printf '%s' '`git push origin master`'",
        ],
    )
    def test_nonexecuting_push_prose_remains_safe(
        self, isolated_approvals, command
    ):
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "allow"
        assert verdict["rule"] is None

    def test_feature_push_interpreter_payload_is_not_hardline_denied(
        self, isolated_approvals
    ):
        assert A.detect_hardline_command("sh -c 'git push origin feature/topic'") == (
            False,
            None,
        )

    @pytest.mark.parametrize(
        "command",
        [
            'echo "$(git push origin main)"',
            'echo "`git push origin master`"',
            'sh -c \'echo "$(git push origin main)"\'',
            'echo "$(eval \'git push origin main\')"',
        ],
    )
    def test_executable_substitution_push_fails_closed(
        self, isolated_approvals, command
    ):
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "hardline-deny"
        assert verdict["rule"] == "push to protected default branch (main/master)"

    @pytest.mark.parametrize(
        "command",
        [
            "eval 'git push origin main'",
            'printf x | xargs -0 --max-args=1 sh -c "git push origin main"',
            "find . -exec git push origin main {} \\;",
            "find . -execdir sh -c 'git push origin main' _ {} +",
        ],
    )
    def test_explicit_shell_carrier_push_fails_closed(
        self, isolated_approvals, command
    ):
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "hardline-deny"
        assert verdict["rule"] == "push to protected default branch (main/master)"

    @pytest.mark.parametrize(
        "command",
        [
            'echo "$(git push origin feature/topic)"',
            "eval 'git push origin feature/topic'",
            "printf x | xargs sh -c 'git push origin feature/topic'",
            "find . -exec git push origin feature/topic {} \\;",
        ],
    )
    def test_explicit_shell_carrier_feature_push_remains_safe(
        self, isolated_approvals, command
    ):
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "allow"
        assert verdict["rule"] is None

    @pytest.mark.parametrize(
        "command",
        [
            "echo '$(git push origin main)'",
            "echo '`git push origin master`'",
            'echo "\\$(git push origin main)"',
            'echo "\\`git push origin master\\`"',
            "printf '%s' 'eval git push origin main'",
            "printf '%s' 'xargs sh -c git push origin main'",
            "printf '%s' 'find . -exec git push origin main ;'",
        ],
    )
    def test_inert_shell_carrier_push_text_remains_safe(
        self, isolated_approvals, command
    ):
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "allow"
        assert verdict["rule"] is None

    def test_process_substitution_push_stays_hardline_denied(self, isolated_approvals):
        verdict = at.evaluate_command("cat <(git push origin main)")
        assert verdict["verdict"] == "hardline-deny"
        assert verdict["rule"] == "push to protected default branch (main/master)"


class TestCommandParserBudgets:
    @staticmethod
    def _opaque_carrier_command(count=200):
        return "printf x " + " ".join('"$(printf x)"' for _ in range(count))

    @staticmethod
    def _fail_if_variants_run(_command):
        raise AssertionError("variant generation ran before parser preflight")

    def test_hardline_preflights_size_limit_before_variant_generation(
        self, isolated_approvals, monkeypatch
    ):
        command = "x" * (A._MAX_DETECTION_COMMAND_CHARS + 1)
        monkeypatch.setattr(A, "_command_detection_variants", self._fail_if_variants_run)

        assert A.detect_hardline_command(command) == (
            True,
            "command parser limit exceeded",
        )

    def test_carrier_limit_fails_closed_before_variant_generation(
        self, isolated_approvals, monkeypatch
    ):
        command = self._opaque_carrier_command()
        assert len(command) < A._MAX_SEPARATOR_FREE_COMMAND_CHARS
        monkeypatch.setattr(A, "_command_detection_variants", self._fail_if_variants_run)

        assert A.detect_hardline_command(command) == (
            True,
            "command parser limit exceeded",
        )
        assert A.detect_dangerous_command(command) == (
            True,
            "command parser limit exceeded",
            "command parser limit exceeded",
        )

    def test_variant_budget_is_lazy_bounded_and_public_detectors_fail_closed(
        self, isolated_approvals, monkeypatch
    ):
        command = "printf parser-budget-control"

        def unique_findings(candidate):
            if candidate != command:
                return
            for index in range(10_000):
                if index == 600:
                    raise AssertionError("payload findings were eagerly materialized")
                yield ("synthetic executable payload", f"printf payload-{index}")

        monkeypatch.setattr(A, "_execution_flag_findings", unique_findings)

        variants = list(A._command_detection_variants(command))
        assert len(variants) <= 512
        assert A.detect_hardline_command(command) == (
            True,
            "command parser limit exceeded",
        )
        assert A.detect_dangerous_command(command) == (
            True,
            "command parser limit exceeded",
            "command parser limit exceeded",
        )

    def test_supported_carriers_below_limit_keep_normal_semantics(
        self, isolated_approvals
    ):
        safe = "printf x " + " ".join('"$(printf safe)"' for _ in range(8))
        protected = 'echo "$(eval \'docker inspect box\')"'

        assert A.detect_hardline_command(safe) == (False, None)
        assert A.detect_hardline_command(protected) == (
            True,
            "inspect container credential environment",
        )


class TestMandatoryHumanApproval:
    """Mandatory side effects always use one-operation human approval."""

    command = "buzz messages send --channel public --message hello"
    pattern_key = "external-message-send"

    @staticmethod
    def _configure_cli(monkeypatch, *, mode="manual"):
        import tools.tirith_security as tirith_security

        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": mode})
        monkeypatch.setattr(A, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(A, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(A, "prompt_dangerous_approval", _REAL_PROMPT_DANGEROUS_APPROVAL)
        monkeypatch.setattr(
            tirith_security,
            "check_command_security",
            lambda _command: {"action": "allow", "findings": [], "summary": ""},
        )
        monkeypatch.setattr(
            A,
            "detect_dangerous_command",
            lambda _command: (True, TestMandatoryHumanApproval.pattern_key, "external send"),
        )

    @pytest.mark.parametrize("scope", ["session", "permanent"])
    def test_existing_pattern_approval_cannot_bypass_live_human(
        self, isolated_approvals, monkeypatch, scope
    ):
        self._configure_cli(monkeypatch)
        session_key = f"mandatory-preapproved-{scope}"
        token = A.set_current_session_key(session_key)
        try:
            if scope == "session":
                A.approve_session(session_key, self.pattern_key)
            else:
                A.approve_permanent(self.pattern_key)

            prompts = []

            def approve_once(command, description, **kwargs):
                prompts.append((command, description, kwargs))
                return "once"

            result = A.check_all_command_guards(
                self.command, "local", approval_callback=approve_once
            )

            assert result["approved"] is True
            assert len(prompts) == 1
            assert prompts[0][2] == {
                "allow_permanent": False,
                "smart_denied": True,
            }
        finally:
            A.clear_session(session_key)
            A.reset_current_session_key(token)
            A._permanent_approved.discard(self.pattern_key)

    def test_smart_approval_cannot_replace_live_human(
        self, isolated_approvals, monkeypatch
    ):
        self._configure_cli(monkeypatch, mode="smart")
        session_key = "mandatory-smart"
        token = A.set_current_session_key(session_key)
        smart_calls = []
        prompts = []
        monkeypatch.setattr(
            A,
            "_smart_approve",
            lambda command, description: smart_calls.append((command, description)) or "approve",
        )
        try:
            result = A.check_all_command_guards(
                self.command,
                "local",
                approval_callback=lambda *args, **kwargs: prompts.append(
                    (args, kwargs)
                )
                or "once",
            )

            assert result["approved"] is True
            assert smart_calls == []
            assert len(prompts) == 1
            assert prompts[0][1] == {
                "allow_permanent": False,
                "smart_denied": True,
            }
        finally:
            A.clear_session(session_key)
            A.reset_current_session_key(token)

    @pytest.mark.parametrize("choice", ["session", "always"])
    def test_cli_never_offers_or_stores_reusable_mandatory_approval(
        self, isolated_approvals, monkeypatch, choice
    ):
        self._configure_cli(monkeypatch)
        session_key = f"mandatory-one-operation-{choice}"
        token = A.set_current_session_key(session_key)
        callback_kwargs = []
        try:
            result = A.check_all_command_guards(
                self.command,
                "local",
                approval_callback=lambda _command, _description, **kwargs: (
                    callback_kwargs.append(kwargs) or choice
                ),
            )

            assert result["approved"] is True
            assert callback_kwargs == [{
                "allow_permanent": False,
                "smart_denied": True,
            }]
            assert not A.is_approved(session_key, self.pattern_key)
            assert self.pattern_key not in A._permanent_approved
        finally:
            A.clear_session(session_key)
            A.reset_current_session_key(token)
            A._permanent_approved.discard(self.pattern_key)

    def test_gateway_never_offers_or_stores_reusable_mandatory_approval(
        self, isolated_approvals, monkeypatch
    ):
        import tools.tirith_security as tirith_security

        monkeypatch.setattr(A, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(A, "_is_gateway_approval_context", lambda: True)
        monkeypatch.setattr(
            tirith_security,
            "check_command_security",
            lambda _command: {"action": "allow", "findings": [], "summary": ""},
        )
        monkeypatch.setattr(
            A,
            "detect_dangerous_command",
            lambda _command: (True, self.pattern_key, "external send"),
        )
        approval_payloads = []
        monkeypatch.setattr(
            A,
            "_await_gateway_decision",
            lambda _session, _notify, data, **_kwargs: (
                approval_payloads.append(data)
                or {"resolved": True, "choice": "always", "reason": None}
            ),
        )
        session_key = "mandatory-gateway-one-operation"
        token = A.set_current_session_key(session_key)
        A.register_gateway_notify(session_key, lambda _data: None)
        try:
            result = A.check_all_command_guards(self.command, "local")

            assert result["approved"] is True
            assert len(approval_payloads) == 1
            assert approval_payloads[0]["allow_permanent"] is False
            assert approval_payloads[0]["allow_session"] is False
            assert not A.is_approved(session_key, self.pattern_key)
            assert self.pattern_key not in A._permanent_approved
        finally:
            A.unregister_gateway_notify(session_key)
            A.clear_session(session_key)
            A.reset_current_session_key(token)
            A._permanent_approved.discard(self.pattern_key)


class TestCredentialHardlineRules:
    """Regression coverage for the credential-enumeration hardline rules.

    These three patterns block read access to material the agent never
    needs: bridge private keys, container credential env, and process
    environ. Each must hardline-deny (rc 3) regardless of mode.
    """

    @pytest.mark.parametrize("command", [
        "cat ~/.hermes/bridge/keys/" + "sample-alpha.priv",
        "head -n 5 ~/.hermes/bridge/keys/" + "sample-beta.priv",
        "tail ~/.hermes/bridge/keys/" + "sample-gamma.priv",
        "xxd ~/.hermes/bridge/keys/" + "sample-delta.priv",
    ])
    def test_bridge_private_key_read_hardline_denied(self, isolated_approvals,
                                                     capsys, command):
        rc = at.approvals_test_command(_args([command], as_json=True))
        payload = json.loads(capsys.readouterr().out)
        assert rc == 3
        assert payload["verdict"] == "hardline-deny"
        assert payload["rule"] == "read Hermes bridge private key"

    @pytest.mark.parametrize("command", [
        "docker inspect example-container",
        "docker container inspect example-container",
        "docker --context example inspect example-container",
        "docker --config /tmp/docker-cli -H unix:///tmp/docker.sock container inspect example-container",
        "docker -H=unix:///tmp/docker.sock inspect example-container",
        "docker --log-level debug inspect example-container",
        "docker --tlscacert /tmp/ca.pem container inspect example-container",
    ])
    def test_docker_inspect_hardline_denied(
        self, isolated_approvals, capsys, command
    ):
        rc = at.approvals_test_command(_args([command], as_json=True))
        payload = json.loads(capsys.readouterr().out)
        assert rc == 3
        assert payload["verdict"] == "hardline-deny"
        assert payload["rule"] == "inspect container credential environment"

    @pytest.mark.parametrize("command", [
        "(docker inspect example-container)",
        "{ docker container inspect example-container; }",
        "printf ready | (docker --context example inspect example-container)",
    ])
    def test_grouped_docker_inspect_hardline_denied(
        self, isolated_approvals, capsys, command
    ):
        rc = at.approvals_test_command(_args([command], as_json=True))
        payload = json.loads(capsys.readouterr().out)
        assert rc == 3
        assert payload["verdict"] == "hardline-deny"
        assert payload["rule"] == "inspect container credential environment"

    @pytest.mark.parametrize("command", [
        "printf ready $(docker inspect example-container)",
        "printf ready `docker container inspect example-container`",
        "sh -c 'docker inspect example-container'",
        'bash -c "docker container inspect example-container"',
        'sh -c \'bash -c "docker inspect example-container"\'',
    ])
    def test_command_carried_docker_inspect_hardline_denied(
        self, isolated_approvals, capsys, command
    ):
        rc = at.approvals_test_command(_args([command], as_json=True))
        payload = json.loads(capsys.readouterr().out)
        assert rc == 3
        assert payload["verdict"] == "hardline-deny"
        assert payload["rule"] == "inspect container credential environment"

    @pytest.mark.parametrize("command", [
        "docker --context inspect",
        'docker --context "example inspect example-container',
    ])
    def test_ambiguous_docker_inspect_fails_closed(
        self, isolated_approvals, capsys, command
    ):
        rc = at.approvals_test_command(_args([command], as_json=True))
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
        "docker --context example ps",
        "docker --context example container ls",
        "docker --config /tmp/docker-cli ps",
        "docker --log-level debug container ls",
        "docker container ls",
        "(docker ps)",
        "{ docker container ls; }",
        "(docker --context example ps)",
        "{ docker --context example container ls; }",
        "printf '%s' 'docker inspect example-container'",
        "printf '%s' '$(docker inspect example-container)'",
        "printf '%s' '`docker inspect example-container`'",
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

    @pytest.mark.parametrize("command", [
        "sh -c 'docker ps'",
        "bash -c 'docker container ls'",
    ])
    def test_benign_docker_interpreter_payload_is_not_hardline_denied(
        self, isolated_approvals, command
    ):
        assert A.detect_hardline_command(command) == (False, None)

    @pytest.mark.parametrize("command", [
        'echo "$(docker inspect box)"',
        'echo "`docker inspect box`"',
        'sh -c \'echo "$(docker inspect box)"\'',
        'echo "$(eval \'docker inspect box\')"',
    ])
    def test_executable_substitution_docker_inspect_hardline_denied(
        self, isolated_approvals, command
    ):
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "hardline-deny"
        assert verdict["rule"] == "inspect container credential environment"

    @pytest.mark.parametrize("command", [
        "eval 'docker inspect box'",
        'printf x | xargs -0 --max-args=1 sh -c "docker inspect box"',
        "find . -exec docker inspect box {} \\;",
        "find . -execdir sh -c 'docker inspect box' _ {} +",
    ])
    def test_explicit_shell_carrier_docker_inspect_hardline_denied(
        self, isolated_approvals, command
    ):
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "hardline-deny"
        assert verdict["rule"] == "inspect container credential environment"

    @pytest.mark.parametrize("command", [
        "echo '$(docker inspect box)'",
        "echo '`docker inspect box`'",
        'echo "\\$(docker inspect box)"',
        'echo "\\`docker inspect box\\`"',
        "printf '%s' 'eval docker inspect box'",
        "printf '%s' 'xargs sh -c docker inspect box'",
        "printf '%s' 'find . -exec docker inspect box ;'",
        "eval 'docker ps'",
        "printf box | xargs printf '%s\\n'",
        "find . -exec docker ps {} \\;",
    ])
    def test_inert_or_safe_docker_carrier_remains_allowed(
        self, isolated_approvals, command
    ):
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "allow"
        assert verdict["rule"] is None

    def test_process_substitution_docker_inspect_stays_hardline_denied(
        self, isolated_approvals
    ):
        verdict = at.evaluate_command("cat <(docker inspect box)")
        assert verdict["verdict"] == "hardline-deny"
        assert verdict["rule"] == "inspect container credential environment"


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
