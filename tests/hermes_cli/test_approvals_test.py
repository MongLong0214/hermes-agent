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


@pytest.fixture
def copied_approval_collections(isolated_approvals, monkeypatch):
    """Keep mandatory-human state mutations off the module's real objects."""
    names = (
        "_pending",
        "_session_approved",
        "_session_yolo",
        "_permanent_approved",
        "_gateway_queues",
        "_gateway_notify_cbs",
        "_denial_tally",
        "_human_wait_states",
    )

    def copy_members(name, value):
        if name == "_session_approved":
            return {key: set(patterns) for key, patterns in value.items()}
        if name == "_gateway_queues":
            return {key: list(entries) for key, entries in value.items()}
        if isinstance(value, set):
            return set(value)
        return dict(value)

    originals = {name: getattr(A, name) for name in names}
    original_members = {
        name: copy_members(name, value) for name, value in originals.items()
    }
    for name, value in originals.items():
        monkeypatch.setattr(A, name, copy_members(name, value))

    try:
        yield A
    finally:
        for name, original in originals.items():
            assert copy_members(name, original) == original_members[name]
            setattr(A, name, original)
            assert getattr(A, name) is original
            assert copy_members(name, original) == original_members[name]


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

    def test_container_skips_only_ordinary_dangerous_guard(
        self, isolated_approvals, monkeypatch
    ):
        monkeypatch.setattr(
            A,
            "_get_approval_config",
            lambda: {"mode": "manual", "deny": ["git push origin feature/*"]},
        )

        protected = at.evaluate_command("git push origin main", env_type="docker")
        assert protected["verdict"] == "hardline-deny"

        mandatory = at.evaluate_command(
            "sh -c 'buzz messages send --channel public --message x'",
            env_type="docker",
        )
        assert mandatory["verdict"] == "ask-approval"

        configured_deny = at.evaluate_command(
            "git push origin feature/container",
            env_type="docker",
        )
        assert configured_deny["verdict"] == "user-deny"

        ordinary = at.evaluate_command("rm -rf ~/build", env_type="docker")
        assert ordinary["verdict"] == "allow"
        assert "container" in ordinary["detail"] or "isolated" in ordinary["detail"]

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


class TestMandatoryShellVariables:
    """Mandatory effects are structural, quote-aware, and non-bypassable."""

    @staticmethod
    def _assert_policy(command, expected, description=None):
        assert A.detect_mandatory_approval_command(command) == (
            expected,
            description if expected else None,
        )
        guard = A.check_all_command_guards(command, "local")
        assert guard["approved"] is not expected
        assert guard.get("mandatory_approval", False) is expected
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == ("ask-approval" if expected else "allow")
        assert verdict["exit_code"] == (2 if expected else 0)
        assert verdict["rule"] == (description if expected else None)

    @pytest.mark.parametrize(
        "command",
        [
            "buzz messages send --channel public --message x",
            "buzz-real messages send --channel public --message x",
            "B=buzz; $B messages send --channel public --message x",
            'B=buzz; "$B" messages send --channel public --message x',
            "$B messages send --channel public --message x",
            "B=printf; echo poison; $B messages send --channel public --message x",
        ],
    )
    def test_buzz_send_requires_live_approval(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, True, "send an external Buzz message")

    @pytest.mark.parametrize(
        "command",
        [
            "B=printf; $B messages send --channel public --message x",
            "$B messages list",
            "B=buzz; '$B' messages send --channel public --message x",
            r"B=buzz; \$B messages send --channel public --message x",
            "printf '%s' 'buzz messages send --channel public --message x'",
            "echo buzz messages send --channel public --message x",
        ],
    )
    def test_buzz_non_send_or_prose_remains_safe(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, False)

    @pytest.mark.parametrize(
        "command",
        [
            "echo x > ~/.hermes/config.yaml",
            "echo x >> $HERMES_HOME/config.yaml",
            "echo x > ${HERMES_HOME}/config.yaml",
            "echo x > /synthetic/.hermes/config.yaml",
            'P=/synthetic/.hermes/config.yaml; echo x > "$P"',
            'echo x > "$P/.hermes/config.yaml"',
            "echo x | tee ~/.hermes/config.yaml",
            'P=/synthetic/.hermes/config.yaml; tee "$P"',
        ],
    )
    def test_hermes_config_write_requires_live_approval(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, True, "write Hermes security configuration")

    @pytest.mark.parametrize(
        "command",
        [
            'P=/tmp/file; echo x > "$P"',
            "P=/synthetic/.hermes/config.yaml; echo x > '$P'",
            r"P=/synthetic/.hermes/config.yaml; echo x > \$P",
            "echo 'write ~/.hermes/config.yaml'",
            "printf '%s' 'tee ~/.hermes/config.yaml'",
            "cat ~/.hermes/config.yaml",
            "echo x > /tmp/config.yaml",
        ],
    )
    def test_non_config_targets_and_config_prose_remain_safe(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, False)

    @pytest.mark.parametrize(
        "command",
        [
            "echo x # > ~/.hermes/config.yaml",
            "printf ok;\n# echo x > ~/.hermes/config.yaml",
            "echo x # buzz messages send --channel public --message x",
            (
                "cat <<'EOF'\n"
                "> ~/.hermes/config.yaml\n"
                "buzz messages send --channel public --message x\n"
                "sendmail user@example.invalid\n"
                "python -c \"import smtplib\"\n"
                "EOF"
            ),
            (
                "cat <<END\n"
                "echo x > ~/.hermes/config.yaml\n"
                "buzz messages send --channel public --message x\n"
                "END"
            ),
        ],
    )
    def test_shell_comments_and_heredoc_bodies_remain_non_mandatory(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, False)

    @pytest.mark.parametrize(
        ("command", "description"),
        [
            (
                "echo x > ~/.hermes/config.yaml # explanatory prose",
                "write Hermes security configuration",
            ),
            (
                "printf x | tee ~/.hermes/config.yaml # explanatory prose",
                "write Hermes security configuration",
            ),
            (
                "echo x # comment ends here\n"
                "buzz messages send --channel public --message x",
                "send an external Buzz message",
            ),
            (
                "echo '#' > ~/.hermes/config.yaml",
                "write Hermes security configuration",
            ),
            (
                r"echo \# > ~/.hermes/config.yaml",
                "write Hermes security configuration",
            ),
        ],
    )
    def test_executable_effects_outside_comments_and_heredocs_remain_mandatory(
        self, isolated_approvals, monkeypatch, command, description
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, True, description)

    @pytest.mark.parametrize(
        "command",
        [
            'python -c "import smtplib; smtplib.SMTP(\"mail.example\").sendmail(\"a\", \"b\", \"x\")"',
            "python3 -c 'import smtplib; smtplib.SMTP().sendmail(\"a\", \"b\", \"x\")'",
            'pypy3 -c "import smtplib; smtplib.SMTP().sendmail(\"a\", \"b\", \"x\")"',
            'M=smtplib; python -c "import $M; $M.SMTP().sendmail(\"a\", \"b\", \"x\")"',
            'M=json; echo poison; python -c "import $M; print($M)"',
            'python2 -c "import $M; print($M)"',
        ],
    )
    def test_interpreter_smtp_requires_live_approval(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, True, "send email through an interpreter/SMTP")

    @pytest.mark.parametrize(
        "command",
        [
            'M=json; python -c "import $M; print($M)"',
            "M=smtplib; python -c 'import $M; print($M)'",
            r'M=smtplib; python -c "import \$M; print(\$M)"',
            "echo 'python -c \"import smtplib\"'",
            'python -c "print(\"smtplib prose\")"',
            "python -m smtplib",
            'node -e "import smtplib"',
        ],
    )
    def test_non_smtp_interpreter_code_and_prose_remain_safe(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, False)

    @pytest.mark.parametrize(
        "command",
        [
            'python3.12 -c "import smtplib; smtplib.SMTP().sendmail(1, 2, 3)"',
            'pypy3.10 -c "from smtplib import SMTP; SMTP().sendmail(1, 2, 3)"',
            "python -c'import smtplib; smtplib.SMTP().sendmail(1, 2, 3)'",
        ],
    )
    def test_versioned_and_attached_python_smtp_requires_live_approval(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, True, "send email through an interpreter/SMTP")

    @pytest.mark.parametrize(
        "command",
        [
            "python3.12 -c 'print(1)'",
            "pypy3.10 -c 'print(1)'",
            "python -c'print(1)'",
            "python-helper -c 'import smtplib'",
            "printf '%s' 'python3.12 -c import smtplib'",
        ],
    )
    def test_versioned_and_attached_interpreter_safe_controls_remain_non_mandatory(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, False)

    @pytest.mark.parametrize(
        "command",
        [
            "python -ic 'import smtplib'",
            "python3.12 -ic 'from smtplib import SMTP'",
            "pypy3.10 -ic 'import smtplib'",
            "python -ic'import smtplib'",
        ],
    )
    def test_bundled_python_c_option_owns_smtp_code(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, True, "send email through an interpreter/SMTP")

    @pytest.mark.parametrize(
        "command",
        [
            "python -W c 'import smtplib'",
            "python -Wc 'import smtplib'",
            "python -X c 'import smtplib'",
            "python -Xc 'import smtplib'",
            "node -ic 'import smtplib'",
            "python -ic 'print(\"smtplib prose\")'",
        ],
    )
    def test_python_option_values_and_non_python_bundles_do_not_own_smtp_code(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, False)

    @pytest.mark.parametrize(
        "command",
        [
            'python -c "print(\'import smtplib\')"',
            'python -c "# import smtplib\nprint(\'ok\')"',
            'python -c "print(\'sendmail(1, 2, 3)\')"',
            'python -c "# SMTP().sendmail(1, 2, 3)\nprint(\'ok\')"',
        ],
    )
    def test_python_smtp_string_and_comment_prose_remains_non_mandatory(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, False)

    @pytest.mark.parametrize(
        "command",
        [
            'python -c "import smtplib"',
            'python -c "from smtplib import SMTP"',
            'python -c "class Mailer:\n def sendmail(self, *args): pass\nMailer().sendmail(1, 2, 3)"',
        ],
    )
    def test_python_ast_smtp_effects_remain_mandatory(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, True, "send email through an interpreter/SMTP")

    def test_python_ast_overflow_is_a_hardline_floor_at_every_boundary(
        self, isolated_approvals, monkeypatch
    ):
        monkeypatch.setattr(A, "_MAX_PYTHON_SMTP_AST_NODES", 8)
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        over_limit = 'python -c "import smtplib; x = 1; y = 2"'

        assert A.detect_hardline_command(over_limit) == (
            True,
            "command parser limit exceeded",
        )
        guard = A.check_all_command_guards(over_limit, "local")
        assert guard["approved"] is False
        assert guard["hardline"] is True
        assert guard.get("mandatory_approval") is not True
        verdict = at.evaluate_command(over_limit)
        assert verdict["verdict"] == "hardline-deny"
        assert verdict["rule"] == "command parser limit exceeded"

        self._assert_policy(
            'python -c "import smtplib"',
            True,
            "send email through an interpreter/SMTP",
        )
        self._assert_policy('python -c "print(\'import smtplib\')"', False)
        self._assert_policy('python -c "# import smtplib"', False)

    @pytest.mark.parametrize(
        "command",
        [
            "sendmail user@example.invalid",
            "/usr/sbin/sendmail user@example.invalid",
            "msmtp user@example.invalid",
            "mailx -s subject user@example.invalid",
            "mutt user@example.invalid",
            "swaks --to user@example.invalid",
            "E=sendmail; $E user@example.invalid",
            "E=ms; ${E}mtp user@example.invalid",
            'E=sendmail; "$E" user@example.invalid',
        ],
    )
    def test_external_email_cli_requires_live_approval(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, True, "send external email")

    @pytest.mark.parametrize(
        "command",
        [
            "E=printf; $E user@example.invalid",
            "E=sendmail; '$E' user@example.invalid",
            r"E=sendmail; \$E user@example.invalid",
            "$E user@example.invalid",
            "echo sendmail user@example.invalid",
            "printf '%s' 'msmtp user@example.invalid'",
            "sendmail-helper user@example.invalid",
        ],
    )
    def test_non_email_executable_and_email_prose_remain_safe(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, False)

    @pytest.mark.parametrize(
        "command",
        [
            "E=sendmail sh -c '$E user@example.invalid'",
            "env E=sendmail sh -c '$E user@example.invalid'",
        ],
    )
    def test_fresh_shell_receives_only_explicit_environment_bindings(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, True, "send external email")

    def test_prior_unexported_shell_binding_does_not_enter_fresh_shell(
        self, isolated_approvals, monkeypatch
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(
            "E=sendmail; sh -c '$E user@example.invalid'",
            False,
        )

    @pytest.mark.parametrize(
        "command",
        [
            "E=sendmail; printf '%s' \"$($E user@example.invalid)\"",
            "E=sendmail; eval '$E user@example.invalid'",
        ],
    )
    def test_same_shell_payloads_receive_parent_expansion_bindings(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, True, "send external email")

    @pytest.mark.parametrize(
        "command",
        [
            "E=printf; eval '$E user@example.invalid'",
            "E=printf; printf '%s' \"$($E user@example.invalid)\"",
            "E=sendmail; printf '%s' 'eval $E user@example.invalid'",
        ],
    )
    def test_same_shell_known_safe_and_prose_controls_remain_non_mandatory(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, False)

    @pytest.mark.parametrize(
        ("command", "description"),
        [
            (
                "env buzz messages send --channel public --message x",
                "send an external Buzz message",
            ),
            (
                "sh -c 'buzz messages send --channel public --message x'",
                "send an external Buzz message",
            ),
            (
                "env sendmail user@example.invalid",
                "send external email",
            ),
            (
                "env sh -c 'echo x > ~/.hermes/config.yaml'",
                "write Hermes security configuration",
            ),
            (
                "sh -c 'python -c \"import smtplib; smtplib.SMTP().sendmail(1, 2, 3)\"'",
                "send email through an interpreter/SMTP",
            ),
            (
                "eval 'buzz messages send --channel public --message x'",
                "send an external Buzz message",
            ),
            (
                "printf x | xargs sendmail user@example.invalid",
                "send external email",
            ),
            (
                "find . -exec sh -c 'echo x > ~/.hermes/config.yaml' _ {} \\;",
                "write Hermes security configuration",
            ),
            (
                "eval 'python -c \"import smtplib; smtplib.SMTP().sendmail(1, 2, 3)\"'",
                "send email through an interpreter/SMTP",
            ),
            (
                "find . -exec sendmail user@example.invalid {} \\;",
                "send external email",
            ),
        ],
    )
    def test_wrapped_and_late_payload_effects_require_live_approval(
        self, isolated_approvals, monkeypatch, command, description
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, True, description)

    @pytest.mark.parametrize(
        "command",
        [
            "env printf '%s' ok",
            "env buzz messages list",
            "sh -c 'echo buzz messages send --channel public --message x'",
            "env sh -c 'echo x > /tmp/config.yaml'",
            "sh -c 'python -c \"print(1)\"'",
            "eval 'echo sendmail user@example.invalid'",
            "printf x | xargs printf '%s\\n'",
            "find . -exec printf '%s\\n' {} \\;",
        ],
    )
    def test_wrapped_and_late_payload_safe_controls_remain_non_mandatory(
        self, isolated_approvals, monkeypatch, command
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        self._assert_policy(command, False)

    def test_hardline_protected_push_beats_mandatory_buzz(
        self, isolated_approvals
    ):
        command = (
            "B=buzz; $B messages send --channel public --message x; "
            "git push origin main"
        )
        assert A.detect_mandatory_approval_command(command) == (
            True,
            "send an external Buzz message",
        )
        guard = A.check_all_command_guards(command, "local")
        assert guard["approved"] is False
        assert guard["hardline"] is True
        assert guard.get("mandatory_approval") is not True
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "hardline-deny"
        assert verdict["rule"] == "push to protected default branch (main/master)"

    def test_user_deny_beats_mandatory_buzz(
        self, isolated_approvals, monkeypatch
    ):
        command = "B=buzz; $B messages send --channel public --message x"
        deny_rule = "*messages send*"
        monkeypatch.setattr(
            A,
            "_get_approval_config",
            lambda: {"mode": "off", "deny": [deny_rule]},
        )
        assert A.detect_mandatory_approval_command(command)[0] is True
        guard = A.check_all_command_guards(command, "local")
        assert guard["approved"] is False
        assert guard["user_deny"] is True
        assert guard.get("mandatory_approval") is not True
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "user-deny"
        assert verdict["rule"] == deny_rule

    @pytest.mark.parametrize("bypass", ["mode-off", "process-yolo", "session-yolo"])
    def test_mandatory_buzz_beats_every_approval_bypass(
        self, isolated_approvals, monkeypatch, bypass
    ):
        command = "B=buzz; $B messages send --channel public --message x"
        if bypass == "mode-off":
            monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        elif bypass == "process-yolo":
            monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", True)
        else:
            monkeypatch.setattr(A, "is_current_session_yolo_enabled", lambda: True)

        guard = A.check_all_command_guards(command, "local")
        assert guard["approved"] is False
        assert guard["mandatory_approval"] is True
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "ask-approval"
        assert verdict["exit_code"] == 2

    def test_wrapped_mandatory_effect_keeps_hardline_precedence(
        self, isolated_approvals
    ):
        command = (
            "git push origin main; "
            "sh -c 'buzz messages send --channel public --message x'"
        )
        assert A.detect_mandatory_approval_command(command) == (
            True,
            "send an external Buzz message",
        )
        guard = A.check_all_command_guards(command, "local")
        assert guard["approved"] is False
        assert guard["hardline"] is True
        assert guard.get("mandatory_approval") is not True
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "hardline-deny"
        assert verdict["rule"] == "push to protected default branch (main/master)"

    def test_exact_user_deny_keeps_wrapped_mandatory_precedence(
        self, isolated_approvals, monkeypatch
    ):
        command = "sh -c 'buzz messages send --channel public --message x'"
        monkeypatch.setattr(
            A,
            "_get_approval_config",
            lambda: {"mode": "off", "deny": [command]},
        )
        assert A.detect_mandatory_approval_command(command) == (
            True,
            "send an external Buzz message",
        )
        guard = A.check_all_command_guards(command, "local")
        assert guard["approved"] is False
        assert guard["user_deny"] is True
        assert guard.get("mandatory_approval") is not True
        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "user-deny"
        assert verdict["rule"] == command

    @pytest.mark.parametrize("bypass", ["mode-off", "process-yolo", "session-yolo"])
    def test_wrapped_mandatory_effect_beats_every_approval_bypass(
        self, isolated_approvals, monkeypatch, bypass
    ):
        # Bind these controls to A's historical wrapper/late-payload RED:
        # before that correction this command was allowed and not mandatory.
        command = "sh -c 'buzz messages send --channel public --message x'"
        if bypass == "mode-off":
            monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "off"})
        elif bypass == "process-yolo":
            monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", True)
        else:
            monkeypatch.setattr(A, "is_current_session_yolo_enabled", lambda: True)

        verdict = at.evaluate_command(command)
        assert verdict["verdict"] == "ask-approval"
        assert verdict["exit_code"] == 2
        assert verdict["rule"] == "send an external Buzz message"
        guard = A.check_all_command_guards(command, "local")
        assert guard["approved"] is False
        assert guard["mandatory_approval"] is True

    def test_shell_variable_resolution_has_no_ambient_lookup(
        self, isolated_approvals, monkeypatch
    ):
        class NoAmbientEnvironment(dict):
            def _deny(self, *_args, **_kwargs):
                raise AssertionError("shell variable resolution touched ambient state")

            __contains__ = _deny
            __getitem__ = _deny
            __iter__ = _deny
            get = _deny
            items = _deny
            keys = _deny
            values = _deny

        def deny_lookup(*_args, **_kwargs):
            raise AssertionError("shell variable resolution performed ambient lookup")

        command = "B=buzz; $B messages send --channel public --message x"
        with monkeypatch.context() as lookup_guard:
            lookup_guard.setattr(A.os, "environ", NoAmbientEnvironment())
            lookup_guard.setattr(A.os, "getenv", deny_lookup)
            lookup_guard.setattr(A.os, "system", deny_lookup)
            lookup_guard.setattr(A.os, "popen", deny_lookup)
            lookup_guard.setattr(A.os.path, "exists", deny_lookup)
            lookup_guard.setattr(A.os.path, "isfile", deny_lookup)
            lookup_guard.setattr(A.os.path, "isdir", deny_lookup)
            lookup_guard.setattr(A, "_get_approval_config", deny_lookup)

            program = A._resolve_shell_program(command)
            assert program.status == A._ShellVariableStatus.RESOLVED
            assert A.detect_mandatory_approval_command(command) == (
                True,
                "send an external Buzz message",
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


class TestMandatoryWrappedHumanApproval:
    """Wrapped effects cannot inherit or create reusable approval scope."""

    command = "sh -c 'buzz messages send --channel public --message x'"
    pattern_key = TestMandatoryHumanApproval.pattern_key

    @staticmethod
    def _configure_cli(monkeypatch, *, mode="manual"):
        TestMandatoryHumanApproval._configure_cli(monkeypatch, mode=mode)

    @pytest.mark.parametrize("scope", ["session", "permanent"])
    def test_cached_pattern_cannot_bypass_wrapped_live_human(
        self, copied_approval_collections, monkeypatch, scope
    ):
        self._configure_cli(monkeypatch)
        session_key = f"mandatory-wrapped-preapproved-{scope}"
        token = A.set_current_session_key(session_key)
        prompts = []
        try:
            if scope == "session":
                A.approve_session(session_key, self.pattern_key)
            else:
                A.approve_permanent(self.pattern_key)
            assert A.is_approved(session_key, self.pattern_key) is True

            result = A.check_all_command_guards(
                self.command,
                "local",
                approval_callback=lambda command, description, **kwargs: (
                    prompts.append((command, description, kwargs)) or "once"
                ),
            )

            assert result["approved"] is True
            assert result["mandatory_approval"] is True
            assert result["description"] == "send an external Buzz message"
            assert prompts == [
                (
                    self.command,
                    "send an external Buzz message",
                    {"allow_permanent": False, "smart_denied": True},
                )
            ]
        finally:
            A.clear_session(session_key)
            A.reset_current_session_key(token)
            A._permanent_approved.discard(self.pattern_key)

    def test_smart_autoapproval_cannot_replace_wrapped_live_human(
        self, copied_approval_collections, monkeypatch
    ):
        self._configure_cli(monkeypatch, mode="smart")
        session_key = "mandatory-wrapped-smart"
        token = A.set_current_session_key(session_key)
        smart_calls = []
        prompts = []
        monkeypatch.setattr(
            A,
            "_smart_approve",
            lambda command, description: smart_calls.append((command, description))
            or "approve",
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
            assert result["mandatory_approval"] is True
            assert result["description"] == "send an external Buzz message"
            assert smart_calls == []
            assert len(prompts) == 1
            assert prompts[0][0][0] == self.command
            assert prompts[0][1] == {
                "allow_permanent": False,
                "smart_denied": True,
            }
        finally:
            A.clear_session(session_key)
            A.reset_current_session_key(token)

    @pytest.mark.parametrize("choice", ["session", "always"])
    def test_cli_reusable_choice_stays_one_operation_for_wrapped_effect(
        self, copied_approval_collections, monkeypatch, choice
    ):
        self._configure_cli(monkeypatch)
        session_key = f"mandatory-wrapped-cli-{choice}"
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
            assert result["mandatory_approval"] is True
            assert callback_kwargs == [
                {"allow_permanent": False, "smart_denied": True}
            ]
            assert not A.is_approved(session_key, self.pattern_key)
            assert self.pattern_key not in A._permanent_approved
        finally:
            A.clear_session(session_key)
            A.reset_current_session_key(token)
            A._permanent_approved.discard(self.pattern_key)

    @pytest.mark.parametrize("choice", ["session", "always"])
    def test_gateway_reusable_choice_stays_one_operation_for_wrapped_effect(
        self, copied_approval_collections, monkeypatch, choice
    ):
        monkeypatch.setattr(A, "_get_approval_config", lambda: {"mode": "manual"})
        monkeypatch.setattr(A, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(A, "_is_gateway_approval_context", lambda: True)
        approval_payloads = []
        monkeypatch.setattr(
            A,
            "_await_gateway_decision",
            lambda _session, _notify, data, **_kwargs: (
                approval_payloads.append(data)
                or {"resolved": True, "choice": choice, "reason": None}
            ),
        )
        session_key = f"mandatory-wrapped-gateway-{choice}"
        token = A.set_current_session_key(session_key)
        A.register_gateway_notify(session_key, lambda _data: None)
        try:
            result = A.check_all_command_guards(self.command, "local")

            assert result["approved"] is True
            assert result["mandatory_approval"] is True
            assert result["description"] == "send an external Buzz message"
            assert len(approval_payloads) == 1
            assert approval_payloads[0]["command"] == self.command
            assert approval_payloads[0]["mandatory_approval"] is True
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

        real_hardline = A.detect_hardline_command

        def hardline_spy(c, *, _mandatory_analysis=None):
            calls["hardline"] = c
            calls["hardline_mandatory_analysis"] = _mandatory_analysis
            return real_hardline(c, _mandatory_analysis=_mandatory_analysis)

        monkeypatch.setattr(A, "detect_hardline_command", hardline_spy)
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
        assert calls.get("hardline_mandatory_analysis") is not None
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
