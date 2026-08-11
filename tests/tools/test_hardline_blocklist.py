"""Tests for the unconditional hardline command blocklist.

The hardline list is a floor below yolo: a small set of commands so
catastrophic they should never run via the agent, regardless of --yolo,
gateway /yolo, approvals.mode=off, or cron approve mode.

Inspired by Mercury Agent's permission-hardened blocklist.
"""

import logging

import pytest

from tools.approval import (
    HARDLINE_PATTERNS,
    check_all_command_guards,
    check_dangerous_command,
    detect_dangerous_command,
    detect_hardline_command,
    disable_session_yolo,
    enable_session_yolo,
    reset_current_session_key,
    set_current_session_key,
)


# -------------------------------------------------------------------------
# Pattern detection
# -------------------------------------------------------------------------

# Commands that MUST be hardline-blocked.
_HARDLINE_BLOCK = [
    # rm -rf targeting root / system dirs / home
    "rm -rf /",
    "rm -rf /*",
    # Shell-equivalent spellings of "rm -rf /": repeated slashes and
    # current/parent-dir segments all collapse back to root, so they must
    # hit the hardline floor too (regression: these used to slip through the
    # root pattern's target group and fall to the softer DANGEROUS_PATTERNS
    # rule, which --yolo / approvals.mode=off / cron approve-mode bypass).
    "rm -rf //",
    "rm -rf /.",
    "rm -rf /./",
    "rm -rf /..",
    "rm -rf //*",
    "rm -fr /./",
    "ls && rm -rf //",
    "rm -rf /home",
    "rm -rf /home/*",
    "rm -rf /etc",
    "rm -rf /usr",
    "rm -rf /var",
    "rm -rf /boot",
    "rm -rf /bin",
    "rm --recursive --force /",
    "rm -fr /",
    "sudo rm -rf /",
    "rm -rf ~",
    "rm -rf ~/",
    "rm -rf ~/*",
    "rm -rf $HOME",
    # Quoted path idioms — the recommended shell form for paths with special
    # chars. These previously slipped past the floor because the surrounding
    # quote broke both the flag group and the (\s|$) terminator (regression
    # guard: catastrophic disk/home wipe under --yolo / approvals.mode=off).
    'rm -rf "/"',
    "rm -rf '/'",
    'rm -rf "/*"',
    'rm -rf "/etc"',
    "rm -rf '/etc'",
    'rm -rf "/home"',
    'rm -rf "/usr"',
    'rm -rf "$HOME"',
    "rm -rf '$HOME'",
    'rm -rf "$HOME/"',
    'rm -rf "~"',
    'sudo rm -rf "/"',
    'rm -rf "/" && echo done',
    # ${HOME} brace form (universally common, previously unmatched).
    "rm -rf ${HOME}",
    'rm -rf "${HOME}"',
    "rm -fr ${HOME}",
    # Filesystem format
    "mkfs.ext4 /dev/sda1",
    "mkfs /dev/sdb",
    "mkfs.xfs /dev/nvme0n1",
    # Raw block device overwrites
    "dd if=/dev/zero of=/dev/sda bs=1M",
    "dd if=/dev/urandom of=/dev/nvme0n1",
    "dd if=anything of=/dev/hda",
    "echo bad > /dev/sda",
    "cat /dev/urandom > /dev/sdb",
    # Fork bomb
    ":(){ :|:& };:",
    # System-wide kill
    "kill -9 -1",
    "kill -1",
    # Shutdown / reboot / halt
    "shutdown -h now",
    "shutdown -r now",
    "sudo shutdown now",
    "reboot",
    "sudo reboot",
    "halt",
    "poweroff",
    "init 0",
    "init 6",
    "telinit 0",
    "systemctl poweroff",
    "systemctl reboot",
    "systemctl halt",
    # Compound / subshell variants
    "ls; reboot",
    "echo done && shutdown -h now",
    "false || halt",
    "$(reboot)",
    "`shutdown now`",
    "sudo -E shutdown now",
    "env FOO=1 reboot",
    "exec shutdown",
    "nohup reboot",
    "setsid poweroff",
    # Bare subshell `(cmd)` and brace-group `{ cmd; }` openers put the trigger
    # at a real command position, so they must hit the floor just like `$(…)`.
    # These slipped through before the quote-aware command-start tokenizer
    # learned to recognize `(` / `{` (issue: (reboot) walked past --yolo).
    "(reboot)",
    "( reboot )",
    "(shutdown -h now)",
    "(poweroff)",
    "(halt)",
    "(init 0)",
    "(systemctl reboot)",
    "(sudo reboot)",
    "{ reboot; }",
    "{ shutdown -h now; }",
    "{ poweroff; }",
    "true && (reboot)",
    "echo hi; { reboot; }",
]


# Commands that look superficially similar but must NOT be hardline-blocked.
_HARDLINE_ALLOW = [
    # rm on non-protected paths
    "rm -rf /tmp/foo",
    "rm -rf /tmp/*",
    "rm -rf ./build",
    "rm -rf node_modules",
    "rm -rf /home/user/scratch",  # subpath of /home, not /home itself
    "rm -rf ~/Downloads/old",
    "rm -rf $HOME/tmp",
    "rm foo.txt",
    "rm -rf some/path",
    # Literal root-level directories that only LOOK like root-collapse
    # spellings. Each inter-slash segment must be exactly "." or ".." to
    # count as a collapse back to "/" — "/..." is a dir literally named
    # "..." and "/.foo" is an ordinary root dotfile. These must NOT be
    # swept into the "recursive delete of root filesystem" hardline rule
    # (regression guard for the collapse-spelling tightening).
    "rm -rf /...",
    "rm -rf /....",
    "rm -rf /.foo",
    "rm -rf /.config/foo",
    # A dangerous-looking command embedded as a quoted *argument* to another
    # command must not trip the floor: the path is immediately followed by a
    # closing quote with no matching opening quote of its own, so the
    # quote-tolerant matcher must still ignore it (no new false positives).
    'git commit -m "rm -rf /"',
    'git commit -m "wipe with rm -rf /etc"',
    # dd to regular files
    "dd if=/dev/zero of=./image.bin",
    "dd if=./data of=./backup.bin",
    # Redirect to regular files / non-block devices
    "echo done > /tmp/flag",
    "echo test > /dev/null",
    # Reading devices is fine
    "ls /dev/sda",
    "cat /dev/urandom | head -c 10",
    # Unrelated commands that happen to contain the trigger word
    "grep 'shutdown' logs.txt",
    "echo reboot",
    "echo '# init 0 in comment'",
    "cat rebooting.log",
    "echo 'halt and catch fire'",
    "python3 -c 'print(\"shutdown\")'",
    "find . -name '*reboot*'",
    # Word-boundary protection
    "mkfs_helper --version",
    # systemctl non-destructive verbs
    "systemctl status nginx",
    "systemctl restart nginx",
    "systemctl stop nginx",
    "systemctl start nginx",
    # targeted kill
    "kill -9 12345",
    "kill -HUP 1234",
    "pkill python",
    # Ordinary ops
    "git status",
    "npm run build",
    "sudo apt update",
    "curl https://example.com | head",
]


@pytest.mark.parametrize("command", _HARDLINE_BLOCK)
def test_hardline_detection_blocks(command):
    is_hl, desc = detect_hardline_command(command)
    assert is_hl, f"expected hardline to match {command!r}"
    assert desc, "hardline match must provide a description"


@pytest.mark.parametrize("command", _HARDLINE_ALLOW)
def test_hardline_detection_allows(command):
    is_hl, desc = detect_hardline_command(command)
    assert not is_hl, f"expected hardline NOT to match {command!r} (got: {desc})"
    assert desc is None


# Commands written with the ordinary quoting / brace shell idioms that
# previously slipped past the floor. Kept as an explicit regression set so
# the intent (quoting `rm -rf "/"` must not be a disk-wipe bypass) survives
# any future refactor of the rm patterns.
_QUOTED_BRACE_BYPASS = [
    'rm -rf "/"',
    "rm -rf '/'",
    'rm -rf "/etc"',
    'rm -rf "/home"',
    'rm -rf "$HOME"',
    "rm -rf ${HOME}",
    'rm -rf "${HOME}"',
]


@pytest.mark.parametrize("command", _QUOTED_BRACE_BYPASS)
def test_quoted_and_brace_paths_are_hardline_blocked(command):
    """Quoted paths and ${HOME} must hit the floor (was a silent bypass)."""
    is_hl, desc = detect_hardline_command(command)
    assert is_hl, f"quoting/brace bypass leaked through hardline floor: {command!r}"
    assert desc


# Multi-line QUOTED arguments are data, not command sequences: a newline
# inside quotes is part of the argument the shell passes to the program.
# These previously tripped the hardline floor because the flat command-start
# class treated every raw newline — even inside quotes — as a command
# boundary, blocking `hermes send` message bodies, multi-line
# `git commit -m` messages, and heredoc text that merely MENTION
# shutdown/reboot commands.
_QUOTED_NEWLINE_DATA_ALLOW = [
    # hermes send with a multi-line message body (the reported symptom)
    'hermes send -t telegram -s "spark1" "console output:\nsudo reboot\ndone"',
    'hermes send -t telegram "line1\nshutdown -h now\nline3"',
    # git commit -m with a multi-line message
    "git commit -m 'ops notes:\nreboot the box after the deploy'",
    'git commit -m "fix startup\nsystemctl reboot was flaky here"',
    # heredoc bodies quoting dangerous strings as data
    "python3 - <<'EOF'\nmsg = 'run sudo reboot later'\nprint(msg)\nEOF",
    "cat > /tmp/notes.txt <<'EOF'\nremember: shutdown -h now\nEOF",
    # rm hardline floor is anchored to the same class — quoted prose about it
    # across a line break must stay data too
    'git commit -m "docs:\nwarn about rm -rf / in the guide"',
]

# The masking must be strictly scoped to quoted data: real command
# boundaries around/inside those same shapes still hit the floor.
_QUOTED_NEWLINE_THREATS_BLOCK = [
    # unquoted newline is a real command separator
    "echo hi\nsudo reboot",
    'echo "a"\nsudo reboot',
    'git commit -m "safe message"\nshutdown -h now',
    # command substitution inside double quotes really executes
    'hermes send -t telegram "$(sudo reboot)"',
    'echo "`shutdown -h now`"',
    # multi-line quoted data followed by a REAL chained command
    'hermes send "line1\nline2" && sudo reboot',
    # a heredoc whose body is data, but the delivery command itself is hardline
    "sudo reboot <<'EOF'\nignored\nEOF",
]


@pytest.mark.parametrize("command", _QUOTED_NEWLINE_DATA_ALLOW)
def test_quoted_newline_data_not_blocked(command):
    """Newlines inside quoted arguments are data, not command starts."""
    is_hl, desc = detect_hardline_command(command)
    assert not is_hl, (
        f"multi-line quoted data false-positived the hardline floor: "
        f"{command!r} (got: {desc})"
    )


@pytest.mark.parametrize("command", _QUOTED_NEWLINE_THREATS_BLOCK)
def test_real_newline_separated_threats_still_blocked(command):
    """Unquoted newlines / $() / backticks remain real command boundaries."""
    is_hl, desc = detect_hardline_command(command)
    assert is_hl, f"real threat leaked through hardline floor: {command!r}"
    assert desc


def test_quoted_newline_data_not_blocked_by_full_guard_chain(clean_session):
    """End-to-end: the guard chain must not hardline-block a multi-line
    quoted message (yolo on, so only the unconditional floor can block)."""
    enable_session_yolo("hardline_test")
    command = 'hermes send -t telegram "status:\nsudo reboot happened at 3am"'
    result = check_all_command_guards(command, "local")
    assert result["approved"], (
        f"guard chain blocked multi-line quoted data: {result.get('message')}"
    )


# Commands that carry the literal string "rm -rf /" (or a sibling) as DATA in
# another command's quoted argument — a PR title, a commit message, an echo /
# printf argument. The shell never executes that text as an rm command, so the
# hardline floor must NOT fire; otherwise the command cannot run at all (this
# blocked `gh pr create --title "…rm -rf /…"` outright). Regression guard for
# the command-position anchor on the rm rules.
_DATA_ARG_NOT_A_COMMAND = [
    'gh pr create --title "block rm -rf / spellings"',
    'git commit -m "fixes rm -rf / bypass"',
    'echo "run rm -rf / now"',
    'echo "rm -rf /"',
    'printf "%s" "rm -rf /"',
    'gh issue comment 1 --body "the fix blocks rm -rf //"',
    # A `(` or `{` INSIDE a quoted argument is prose, not a subshell/brace
    # opener — the trigger word after it is data. Naively adding `(` / `{` to
    # the flat command-position class blocked these (it broke our own
    # `gh pr create --title "…(reboot)…"` workflow); the quote-aware tokenizer
    # must leave them alone.
    'gh pr create --title "block (reboot) spellings"',
    'git commit -m "(rm -rf /) note"',
    'echo "(reboot)"',
    'echo "{ reboot; }"',
    "echo '(poweroff)'",
    "echo '{ rm -rf /; }'",
    'find . -name "*(reboot)*"',
]


# Real root wipes at every command position — bare, chained after a separator,
# inside a command substitution ($()/backtick), or after sudo/env wrappers.
# The command-position anchor must keep catching all of these; the substitution
# forms exercise the shell-metacharacter terminator on the bare path branch.
_COMMAND_POSITION_ROOT_WIPES = [
    "rm -rf /",
    "ls && rm -rf /",
    "ls; rm -rf /",
    "echo x | rm -rf /",
    "sudo rm -rf /",
    "env X=1 rm -rf /",
    "$(rm -rf /)",
    "`rm -rf /`",
    'echo "$(rm -rf /)"',
    # Bare subshell / brace-group openers are real command positions too.
    "(rm -rf /)",
    "{ rm -rf /; }",
    "(rm -rf ~)",
    "(sudo rm -rf /)",
]


@pytest.mark.parametrize("command", _COMMAND_POSITION_ROOT_WIPES)
def test_root_wipe_at_command_position_is_hardline(command):
    """A real `rm -rf /` at any command position stays hardline-blocked."""
    is_hl, desc = detect_hardline_command(command)
    assert is_hl, f"real root wipe leaked past the floor: {command!r}"
    assert desc


# -------------------------------------------------------------------------
# Shell line-continuation bypass
# -------------------------------------------------------------------------
#
# A backslash immediately followed by a newline is a POSIX line
# continuation: the shell removes BOTH characters and joins the tokens, so
# `rm -rf \<newline>/` executes as `rm -rf /`. The normalizer used to strip
# only backslash-escapes of NON-newline characters (`\\([^\n])`), leaving the
# dangling backslash wedged between tokens — which broke the structured
# rm/dd/mkfs patterns and let a root wipe slip past the hardline floor.

# (command_with_continuation, description_substring) — each is the
# line-continuation form of a command already in _HARDLINE_BLOCK.
_HARDLINE_LINE_CONTINUATION = [
    ("rm -rf \\\n/", "root"),            # split before the path
    ("rm -r\\\nf /", "root"),            # split inside the flag bundle
    ("rm -rf \\\n~", "home"),            # home-directory wipe
    ("rm -rf \\\r\n/", "root"),          # CRLF line ending
    ("mkfs.ext4 \\\n/dev/sda1", "mkfs"),  # filesystem format
]


@pytest.mark.parametrize("command,desc_substr", _HARDLINE_LINE_CONTINUATION)
def test_hardline_blocks_line_continuation(command, desc_substr):
    is_hl, desc = detect_hardline_command(command)
    assert is_hl, f"line-continuation bypassed hardline detection: {command!r}"
    assert desc and desc_substr in desc.lower(), (
        f"unexpected description {desc!r} for {command!r}"
    )


# -------------------------------------------------------------------------
# Integration with the approval flow
# -------------------------------------------------------------------------

@pytest.fixture
def clean_session(monkeypatch):
    """Reset session-scoped approval state around each test."""
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    token = set_current_session_key("hardline_test")
    try:
        disable_session_yolo("hardline_test")
        yield
    finally:
        disable_session_yolo("hardline_test")
        reset_current_session_key(token)


def test_check_dangerous_command_blocks_hardline(clean_session):
    result = check_dangerous_command("rm -rf /", "local")
    assert result["approved"] is False
    assert result.get("hardline") is True
    assert "BLOCKED (hardline)" in result["message"]


def test_check_all_command_guards_blocks_hardline(clean_session):
    result = check_all_command_guards("rm -rf /", "local")
    assert result["approved"] is False
    assert result.get("hardline") is True
    assert "BLOCKED (hardline)" in result["message"]


def test_yolo_env_var_cannot_bypass_hardline(clean_session, monkeypatch):
    """HERMES_YOLO_MODE=1 must not bypass the hardline floor."""
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    for cmd in ['rm -rf /', 'rm -rf "/"', 'rm -rf "$HOME"', "rm -rf ${HOME}",
                "shutdown -h now", "mkfs.ext4 /dev/sda", "reboot"]:
        r1 = check_dangerous_command(cmd, "local")
        assert r1["approved"] is False, f"yolo leaked hardline on {cmd!r} (check_dangerous_command)"
        assert r1.get("hardline") is True

        r2 = check_all_command_guards(cmd, "local")
        assert r2["approved"] is False, f"yolo leaked hardline on {cmd!r} (check_all_command_guards)"
        assert r2.get("hardline") is True


def test_root_collapse_forms_cannot_bypass_hardline(clean_session, monkeypatch):
    """Shell-equivalent spellings of "rm -rf /" stay blocked under yolo.

    "//", "/.", "/./", "/..", "//*" all collapse to the root filesystem in
    the shell. They previously matched only the softer DANGEROUS_PATTERNS
    rule, which yolo bypasses — leaving the hardline floor open to a full
    root wipe under --yolo / approvals.mode=off / cron approve-mode.
    """
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    for cmd in ["rm -rf //", "rm -rf /.", "rm -rf /./", "rm -rf /..", "rm -rf //*"]:
        is_hl, _ = detect_hardline_command(cmd)
        assert is_hl, f"{cmd!r} should be hardline-blocked"
        result = check_all_command_guards(cmd, "local")
        assert result["approved"] is False, f"yolo leaked hardline on {cmd!r}"
        assert result.get("hardline") is True


def test_root_collapse_pattern_leaves_real_paths_alone(clean_session):
    """The broadened root token must not over-match real trailing segments.

    A path with a real component after the root-collapse prefix (/tmp,
    /home/user/x, /.ssh, ./build) is recoverable-or-legitimate and must NOT
    be pulled onto the hardline floor by the "collapse to /" broadening.
    """
    for cmd in ["rm -rf /tmp", "rm -rf /home/user/x", "rm -rf /.ssh",
                "rm -rf /.config", "rm -rf ./build", "rm -rf /opt/foo",
                "rm -rf /...", "rm -rf /....", "rm -rf /.foo"]:
        is_hl, _ = detect_hardline_command(cmd)
        assert not is_hl, f"{cmd!r} must not be hardline-blocked (over-match)"


def test_subshell_brace_group_cannot_bypass_hardline(clean_session, monkeypatch):
    """Wrapping a catastrophic command in `(…)` or `{ …; }` must not bypass
    the floor, even under yolo. `(reboot)` / `{ shutdown -h now; }` walked
    straight past the guard before the command-start tokenizer recognized the
    subshell and brace-group openers.
    """
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    for cmd in ["(reboot)", "( reboot )", "(shutdown -h now)", "(poweroff)",
                "(systemctl reboot)", "(init 0)", "(sudo reboot)",
                "{ reboot; }", "{ shutdown -h now; }", "{ poweroff; }",
                "(rm -rf /)", "{ rm -rf /; }", "(rm -rf ~)",
                "true && (reboot)", "echo hi; { reboot; }"]:
        r1 = check_dangerous_command(cmd, "local")
        assert r1["approved"] is False, f"yolo leaked hardline on {cmd!r} (check_dangerous_command)"
        assert r1.get("hardline") is True

        r2 = check_all_command_guards(cmd, "local")
        assert r2["approved"] is False, f"yolo leaked hardline on {cmd!r} (check_all_command_guards)"
        assert r2.get("hardline") is True


def test_quoted_paren_brace_prose_not_blocked_under_yolo(clean_session, monkeypatch):
    """A `(` / `{` inside a quoted argument is prose, not a command opener.

    Regression guard: naively adding `(` / `{` to the flat command-position
    class blocked ordinary quoted arguments — including our own
    `gh pr create --title "…(reboot)…"` workflow. The quote-aware tokenizer
    must leave quoted text untouched, so these stay runnable.
    """
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    for cmd in ['gh pr create --title "block (reboot) spellings"',
                'git commit -m "(rm -rf /) note"',
                'echo "(reboot)"', 'echo "{ reboot; }"',
                "echo '(poweroff)'", 'find . -name "*(reboot)*"']:
        assert detect_hardline_command(cmd)[0] is False, (
            f"quoted prose false-positived on the hardline floor: {cmd!r}"
        )


def test_line_continuation_root_wipe_cannot_bypass_hardline(clean_session, monkeypatch):
    """A line-continuation root wipe must stay blocked even under yolo.

    `rm -rf \\<newline>/` runs as `rm -rf /`. Yolo bypasses the regular
    dangerous-command layer, so the hardline floor is the only thing left to
    catch it — it must hold.
    """
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    result = check_all_command_guards("rm -rf \\\n/", "local")
    assert result["approved"] is False, "yolo leaked a line-continuation root wipe"
    assert result.get("hardline") is True
    assert "BLOCKED (hardline)" in result["message"]


def test_session_yolo_cannot_bypass_hardline(clean_session):
    """Gateway /yolo (session-scoped) must not bypass the hardline floor."""
    enable_session_yolo("hardline_test")

    result = check_dangerous_command("rm -rf /", "local")
    assert result["approved"] is False
    assert result.get("hardline") is True

    result = check_all_command_guards("rm -rf /", "local")
    assert result["approved"] is False
    assert result.get("hardline") is True


def test_approvals_mode_off_cannot_bypass_hardline(clean_session, monkeypatch, tmp_path):
    """config approvals.mode=off (yolo-equivalent) must not bypass hardline."""
    # _get_approval_mode() reads from hermes config; simplest path: monkeypatch the helper.
    import tools.approval as approval_mod
    monkeypatch.setattr(approval_mod, "_get_approval_mode", lambda: "off")

    result = check_all_command_guards("rm -rf /", "local")
    assert result["approved"] is False
    assert result.get("hardline") is True


def test_cron_approve_mode_cannot_bypass_hardline(clean_session, monkeypatch):
    """Cron sessions with cron_mode=approve must not bypass hardline."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    import tools.approval as approval_mod
    monkeypatch.setattr(approval_mod, "_get_cron_approval_mode", lambda: "approve")

    result = check_all_command_guards("rm -rf /", "local")
    assert result["approved"] is False
    assert result.get("hardline") is True


def test_container_backends_bypass_dangerous_prompt_but_not_hardline_floor(
    clean_session,
):
    """Containers may bypass reusable dangerous prompts, never the hardline floor."""
    for env in ("docker", "singularity", "modal", "daytona", "vercel_sandbox"):
        r1 = check_dangerous_command("rm -rf /", env)
        assert r1["approved"] is True, f"container {env} should still bypass"
        r2 = check_all_command_guards("rm -rf /", env)
        assert r2["approved"] is False, f"container {env} must not bypass hardline"
        assert r2.get("hardline") is True


def test_hardline_runs_before_dangerous_detection(clean_session):
    """Hardline command should return hardline block, not dangerous approval prompt."""
    # `rm -rf /` is both hardline AND matches DANGEROUS_PATTERNS. Hardline must win.
    is_dangerous, _, _ = detect_dangerous_command("rm -rf /")
    assert is_dangerous, "precondition: rm -rf / is also in DANGEROUS_PATTERNS"

    result = check_dangerous_command("rm -rf /", "local")
    assert result.get("hardline") is True


def test_recoverable_dangerous_commands_still_pass_yolo(clean_session, monkeypatch):
    """Yolo still bypasses the regular DANGEROUS_PATTERNS list.

    This confirms we haven't broken the yolo escape hatch — only narrowed it.
    """
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    # These are dangerous but NOT hardline — yolo should still pass them.
    for cmd in [
        "rm -rf /tmp/x",
        "chmod -R 777 .",
        "git reset --hard",
        "git push --force origin feature/topic",
    ]:
        # Sanity: still flagged as dangerous
        is_dangerous, _, _ = detect_dangerous_command(cmd)
        assert is_dangerous, f"precondition: {cmd!r} should be in DANGEROUS_PATTERNS"
        # But NOT hardline
        is_hl, _ = detect_hardline_command(cmd)
        assert not is_hl, f"{cmd!r} should not be hardline"
        # And yolo bypasses the dangerous check
        result = check_dangerous_command(cmd, "local")
        assert result["approved"] is True, f"yolo should have bypassed {cmd!r}"


def test_hardline_list_is_small():
    """Hardline list stays focused on unrecoverable commands only.

    If you're adding a 20th+ pattern, reconsider — it probably belongs in
    DANGEROUS_PATTERNS where yolo can still bypass it.
    """
    assert len(HARDLINE_PATTERNS) <= 20, (
        f"HARDLINE_PATTERNS has grown to {len(HARDLINE_PATTERNS)} entries; "
        "only truly unrecoverable commands belong here."
    )


# =========================================================================
# Sudo stdin guard — blocks "sudo -S" without SUDO_PASSWORD
# =========================================================================

_SUDO_STDIN_BLOCK = [
    "sudo -S whoami",
    "echo hunter2 | sudo -S whoami",
    "sudo -S -u root whoami",
    "sudo -S apt-get install foo",
    "echo password | sudo -S systemctl restart nginx",
    "sudo -k && sudo -S whoami",
]

_SUDO_STDIN_ALLOW = [
    # Plain sudo without -S — goes through normal approval
    "sudo whoami",
    "sudo apt-get update",
    "sudo -u root whoami",
    # -S flag not attached to sudo
    "echo -S hello",
    "some_tool -S thing",
    # Literal text mention of sudo
    "echo 'use sudo -S to pipe passwords'",
]

_SUDO_STDIN_BLOCK_YOLO = [
    "sudo -S whoami",
    "echo hunter2 | sudo -S apt-get install",
]


def test_sudo_stdin_guard_detects_without_password():
    """sudo -S is dangerous when SUDO_PASSWORD is not configured."""
    import tools.approval as approval_mod

    for cmd in _SUDO_STDIN_BLOCK:
        is_blocked, desc = approval_mod._check_sudo_stdin_guard(cmd)
        assert is_blocked, f"expected sudo stdin guard to block {cmd!r}"
        assert "sudo" in desc.lower()


def test_sudo_stdin_guard_container_bypass(clean_session):
    """Containerized backends still bypass — they can't touch the host."""
    for env in ("docker", "singularity", "modal", "daytona", "vercel_sandbox"):
        for cmd in _SUDO_STDIN_BLOCK:
            result = check_all_command_guards(cmd, env)
            assert result["approved"] is True, f"container {env} should bypass sudo guard on {cmd!r}"


class TestProtectedPushGitAliases:
    def test_indexed_config_alias_push_is_hardline_at_public_and_container_boundaries(
        self, clean_session
    ):
        command = (
            "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=alias.p "
            "GIT_CONFIG_VALUE_0=push git p origin main"
        )

        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    def test_literal_git_config_parameters_bare_push_alias_is_hardline_at_boundaries(
        self, clean_session
    ):
        command = "GIT_CONFIG_PARAMETERS=\"'alias.p=push'\" git p"

        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            "GIT_CONFIG_PARAMETERS=\"'alias.p=q' 'alias.q=push'\" git p",
            "GIT_CONFIG_PARAMETERS=\"'alias.p'='push'\" git p",
            (
                r'''GIT_CONFIG_PARAMETERS="'alias.p'='\!'echo '''
                r'''\'\''ready'\''; git push origin main'" git p'''
            ),
            "GIT_CONFIG_PARAMETERS=\"'include.path=/synthetic/config'\" git p",
            (
                "GIT_CONFIG_PARAMETERS=\""
                "'includeIf.onbranch:main.path=/synthetic/config'\" git p"
            ),
            "GIT_CONFIG_PARAMETERS=\"'alias.p=push\" git p",
            "GIT_CONFIG_PARAMETERS=$PARAMETERS git p",
            (
                "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=alias.p "
                "GIT_CONFIG_VALUE_0=status "
                "GIT_CONFIG_PARAMETERS=\"'alias.p=push'\" git p"
            ),
            (
                "GIT_CONFIG_PARAMETERS=\"'alias.p=status'\" "
                "git -c alias.p=push p"
            ),
            (
                "GIT_CONFIG_PARAMETERS=\"'alias.p=status' "
                "'alias.p=push'\" git p"
            ),
        ],
    )
    def test_git_config_parameters_alias_grammar_is_hardline_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    def test_git_config_parameters_entry_overflow_fails_closed_at_boundaries(
        self, clean_session
    ):
        parameter_source = " ".join(
            f"'alias.local{index}=status'" for index in range(65)
        )
        command = f'GIT_CONFIG_PARAMETERS="{parameter_source}" git p'

        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            "GIT_CONFIG_PARAMETERS=\"'alias.s=status'\" git s --short",
            (
                "GIT_CONFIG_PARAMETERS=\"'alias.p=push origin feature/topic'\" "
                "git p"
            ),
            (
                "GIT_CONFIG_PARAMETERS=\"'include.path=/synthetic/config'\" "
                "git status main"
            ),
            "GIT_CONFIG_PARAMETERS=\"'not-an-operation prose'\" git status main",
            "GIT_CONFIG_PARAMETERS=$PARAMETERS git status main",
            "GIT_CONFIG_PARAMETERS=$PARAMETERS git p origin feature/topic",
            (
                "GIT_CONFIG_PARAMETERS=\"'alias.p=push' 'alias.p=status'\" "
                "git p"
            ),
            (
                "GIT_CONFIG_PARAMETERS=\"'alias.p=push'\" "
                "git -c alias.p=status p"
            ),
            (
                "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=alias.p "
                "GIT_CONFIG_VALUE_0=push "
                "GIT_CONFIG_PARAMETERS=\"'alias.p=status'\" git p"
            ),
            "GIT_CONFIG_PARAMETERS=\"'alias.s'='status'\" git s --short",
        ],
    )
    def test_git_config_parameters_preserves_status_feature_and_prose_controls(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (False, None)
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is True
        assert result.get("hardline") is not True

    def test_git_config_parameters_is_not_read_from_ambient_environment(
        self, monkeypatch, clean_session
    ):
        monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'alias.p=push'")

        assert detect_hardline_command("git p") == (False, None)
        result = check_all_command_guards("git p", "docker")
        assert result["approved"] is True
        assert result.get("hardline") is not True

    def test_dynamic_include_alias_push_is_hardline_at_public_and_container_boundaries(
        self, clean_session
    ):
        command = "git -c include.path=/synthetic/config p origin main"

        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            (
                "env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=alias.p "
                "GIT_CONFIG_VALUE_0=push git p origin master"
            ),
            "git -cinclude.path=/synthetic/config p origin main",
            "git -c includeIf.onbranch:main.path=/synthetic/config p origin main",
            "git --config-env=include.path=INCLUDE_FILE p origin main",
            (
                "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=include.path "
                "GIT_CONFIG_VALUE_0=/synthetic/config git p origin main"
            ),
            (
                "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=alias.p "
                "git p origin main"
            ),
            (
                "GIT_CONFIG_COUNT=65 GIT_CONFIG_KEY_0=alias.p "
                "GIT_CONFIG_VALUE_0=push git p origin main"
            ),
        ],
    )
    def test_dynamic_or_malformed_command_config_push_shape_fails_closed(
        self, command
    ):
        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )

    @pytest.mark.parametrize(
        "command",
        [
            "git -c alias.p=push p origin main",
            "git -calias.p=push p origin main",
            "git --config-env=alias.p=PUSH_ALIAS p origin main",
            "git --config-env alias.p=PUSH_ALIAS p origin main",
            "git -c 'alias.p=!git push' p origin main",
        ],
    )
    def test_alias_push_to_default_branch_is_hardline_denied(self, command):
        is_hardline, description = detect_hardline_command(command)

        assert is_hardline is True
        assert description == "push to protected default branch (main/master)"

    @pytest.mark.parametrize(
        "command",
        [
            "git -c alias.s=status s",
            "git -calias.p=push p origin feature/topic",
            "git --config-env=alias.p=PUSH_ALIAS p origin feature/topic",
        ],
    )
    def test_non_push_or_feature_branch_alias_is_not_hardline_denied(self, command):
        assert detect_hardline_command(command) == (False, None)

    @pytest.mark.parametrize(
        "command",
        [
            (
                "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=alias.s "
                "GIT_CONFIG_VALUE_0=status git s main"
            ),
            (
                "env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=alias.s "
                "GIT_CONFIG_VALUE_0=status git s"
            ),
            (
                "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=alias.p "
                "GIT_CONFIG_VALUE_0=push git p origin feature/topic"
            ),
            "git -c include.path=/synthetic/config status main",
            "git -c include.path=/synthetic/config p origin feature/topic",
            "git -c include.path=/synthetic/config p status",
            "git -c include.path=/synthetic/config show origin main",
            "git -c include.path=/synthetic/config branch main",
            "git -c include.path=/synthetic/config",
        ],
    )
    def test_command_config_controls_are_not_hardline_denied(self, command):
        assert detect_hardline_command(command) == (False, None)

    @pytest.mark.parametrize(
        "command",
        [
            "git p origin main",
            "git p origin master",
            "GIT_CONFIG_GLOBAL=/synthetic/global git p origin main",
            "GIT_CONFIG_SYSTEM=/synthetic/system git p origin main",
            "GIT_CONFIG_PARAMETERS='alias.p=push' git p origin main",
            "HOME=/synthetic/home git p origin main",
            "XDG_CONFIG_HOME=/synthetic/xdg git p origin main",
            "GIT_DIR=/synthetic/repo/.git git p origin main",
            "GIT_COMMON_DIR=/synthetic/common git p origin main",
            "git --git-dir=/synthetic/repo/.git p origin main",
        ],
    )
    def test_unknown_alias_or_external_helper_push_shape_is_hardline_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            "git p origin feature/topic",
            "GIT_CONFIG_GLOBAL=/synthetic/global git p origin feature/topic",
            "GIT_CONFIG_SYSTEM=/synthetic/system git status main",
            "GIT_CONFIG_PARAMETERS='alias.p=push' git show origin main",
            "HOME=/synthetic/home git branch main",
        ],
    )
    def test_unknown_alias_floor_preserves_feature_and_builtin_controls(self, command):
        assert detect_hardline_command(command) == (False, None)

    @pytest.mark.parametrize(
        "command",
        [
            "git -c alias.p=q -c alias.q=push p origin main",
            "git -c alias.p=q -c 'alias.q=push origin main' p",
            "git -c alias.p=q p origin main",
            "git -c alias.p=q -c alias.q=p p origin master",
            (
                "git -c alias.a=b -c alias.b=c -c alias.c=d -c alias.d=e "
                "-c alias.e=f -c alias.f=g -c alias.g=h -c alias.h=i "
                "-c alias.i=j -c alias.j=push a origin main"
            ),
        ],
    )
    def test_bounded_alias_chains_and_ambiguity_are_hardline_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            "git -c alias.p=q -c alias.q=push p origin feature/topic",
            "git -c alias.p=q -c 'alias.q=push origin feature/topic' p",
            "git -c alias.s=t -c alias.t=status s main",
        ],
    )
    def test_bounded_alias_chains_preserve_feature_and_builtin_controls(self, command):
        assert detect_hardline_command(command) == (False, None)

    @pytest.mark.parametrize(
        "command",
        [
            "PUSH_ALIAS='push origin main' git --config-env=alias.p=PUSH_ALIAS p",
            (
                "env PUSH_ALIAS='push origin master' "
                "git --config-env alias.p=PUSH_ALIAS p"
            ),
            "PUSH_ALIAS='push \"' git --config-env=alias.p=PUSH_ALIAS p origin main",
            "git --config-env=alias.p=ABSENT_ALIAS p origin master",
        ],
    )
    def test_literal_config_env_aliases_and_ambiguity_are_hardline_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            "PUSH_ALIAS=status git --config-env=alias.p=PUSH_ALIAS p",
            "env PUSH_ALIAS=status git --config-env alias.p=PUSH_ALIAS p main",
            (
                "PUSH_ALIAS='push origin feature/topic' "
                "git --config-env=alias.p=PUSH_ALIAS p"
            ),
        ],
    )
    def test_literal_config_env_preserves_status_and_feature_controls(self, command):
        assert detect_hardline_command(command) == (False, None)

    @pytest.mark.parametrize(
        "command",
        [
            "git -c 'alias.p=!git push origin main' p",
            "git -c 'alias.p=!sh -c \"git push origin master\"' p",
            "git -c 'alias.p=!f() { git push origin main; }; f' p",
            "git -c 'alias.p=!echo ready && git push origin master' p",
            "git -c 'alias.p=!custom-helper' p origin main",
        ],
    )
    def test_shell_alias_pushes_and_opaque_push_shapes_are_hardline_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            'git -c \'alias.p=!g=git; echo "$($g push origin main)"\' p',
            (
                'git -c \'alias.p=!g=git; case "$($g push origin main)" '
                "in *) :;; esac' p"
            ),
        ],
    )
    def test_shell_alias_executable_substitutions_are_hardline_at_boundaries(
        self, command, clean_session
    ):
        detector_result = detect_hardline_command(command)
        guard_result = check_all_command_guards(command, "docker")

        assert detector_result == (
            True,
            "push to protected default branch (main/master)",
        )
        assert guard_result["approved"] is False
        assert guard_result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            'git -c \'alias.p=!g=git; echo "`$g push origin main`"\' p',
            'git -c \'alias.p=!g=git; echo "$($g push origin main"\' p',
            (
                "git -c 'alias.p=!g=git; echo "
                "$(echo $(echo $(echo $($g push origin main))))' p"
            ),
        ],
    )
    def test_shell_alias_substitution_edges_fail_closed_at_boundaries(
        self, command, clean_session
    ):
        detector_result = detect_hardline_command(command)
        guard_result = check_all_command_guards(command, "docker")

        assert detector_result == (
            True,
            "push to protected default branch (main/master)",
        )
        assert guard_result["approved"] is False
        assert guard_result["hardline"] is True

    def test_shell_alias_substitution_work_overflow_with_push_fails_closed(
        self, clean_session, monkeypatch
    ):
        import tools.approval as approval_mod

        monkeypatch.setattr(approval_mod, "_MAX_DETECTION_WORK_ITEMS", 1)
        command = (
            'git -c \'alias.p=!g=git; echo "$($g status --short)" '
            '"$($g push origin main)"\' p'
        )

        detector_result = detect_hardline_command(command)
        guard_result = check_all_command_guards(command, "docker")

        assert detector_result == (
            True,
            "push to protected default branch (main/master)",
        )
        assert guard_result["approved"] is False
        assert guard_result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            'git -c \'alias.s=!g=git; echo "$($g status --short)"\' s',
            (
                'git -c \'alias.s=!g=git; case "$($g status --short)" '
                "in *) :;; esac' s"
            ),
            'git -c "alias.s=!g=git; echo \'$($g push origin main)\'" s',
            'git -c \'alias.s=!g=git; echo "\\$($g push origin main)"\' s',
            'git -c \'alias.s=!g=git; echo "`$g status --short`"\' s',
            'git -c "alias.s=!g=git; echo \'`$g push origin main`\'" s',
            'git -c \'alias.s=!g=git; echo "\\`$g push origin main\\`"\' s',
            "git -c 'alias.s=!echo harmless' s",
            "git -c 'alias.s=!printf \"%s\" harmless' s",
        ],
    )
    def test_shell_alias_substitution_controls_remain_safe(self, command):
        assert detect_hardline_command(command) == (False, None)

    @pytest.mark.parametrize(
        "command",
        [
            "git -c 'alias.p=!git status' p origin main",
            "git -c 'alias.p=!echo harmless' p origin master",
            "git -c 'alias.p=!git push origin feature/topic' p",
            "git -c 'alias.p=!echo \"git push origin main\"' p",
            "git -c 'alias.p=!printf \"%s\" \"git push origin main\"' p origin main",
            "git -c 'alias.p=!custom-helper' p origin feature/topic",
        ],
    )
    def test_shell_aliases_preserve_harmless_prose_and_feature_controls(self, command):
        assert detect_hardline_command(command) == (False, None)

    @pytest.mark.parametrize(
        "command",
        [
            "git -c \"alias.p=!eval 'git push origin main'\" p",
            "git -c \"alias.p=!eval 'git push origin main\" p",
        ],
    )
    def test_shell_alias_eval_push_is_hardline_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    def test_shell_alias_grouped_eval_push_is_hardline_at_boundaries(
        self, clean_session
    ):
        command = "git -c \"alias.p=!( eval 'git push origin main' )\" p"

        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            "git -c \"alias.p=!( eval 'git status --short' )\" p",
            (
                "git -c \"alias.p=!( eval "
                "'git push origin feature/topic' )\" p"
            ),
            (
                "git -c 'alias.s=!printf \"%s\" "
                "\"( eval git push origin main )\"' s"
            ),
            (
                "git -c 'alias.s=!printf \"%s\" "
                "\\(\\ eval\\ git\\ push\\ origin\\ main\\ \\)' s"
            ),
        ],
    )
    def test_shell_alias_grouped_eval_controls_remain_safe_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (False, None)
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is True
        assert result.get("hardline") is not True

    def test_shell_alias_eval_work_overflow_with_push_fails_closed(
        self, monkeypatch, clean_session
    ):
        import tools.approval as approval_mod

        monkeypatch.setattr(approval_mod, "_MAX_DETECTION_WORK_ITEMS", 1)
        command = (
            "git -c \"alias.p=!eval 'git status --short'; "
            "eval 'git push origin main'\" p"
        )

        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            "git -c \"alias.s=!eval 'git status --short'\" s",
            "git -c \"alias.s=!eval 'git status --short\" s",
            "git -c \"alias.p=!eval 'git push origin feature/topic'\" p",
            "git -c 'alias.s=!printf \"%s\" \"eval git push origin main\"' s",
            "git -c 'alias.s=!printf \"%s\" eval\\ git\\ push\\ origin\\ main' s",
        ],
    )
    def test_shell_alias_eval_controls_remain_safe_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (False, None)
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is True
        assert result.get("hardline") is not True

    def test_nested_xargs_dash_push_is_hardline_at_boundaries(
        self, clean_session
    ):
        command = (
            r'''git -c 'alias.p=!printf x | xargs -0 dash -c "git push origin master"' p'''
        )

        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    def test_direct_dash_shell_alias_push_is_hardline_at_boundaries(
        self, clean_session
    ):
        command = r'''git -c 'alias.p=!/bin/dash -c "git push origin main"' p'''

        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            r'''git -c 'alias.s=!printf x | xargs -0 dash -c "git status --short"' s''',
            r'''git -c 'alias.p=!printf x | xargs -0 dash -c "git push origin feature/topic"' p''',
            r'''git -c 'alias.s=!/bin/dash -c "git status --short"' s''',
            r'''git -c 'alias.s=!printf "%s" "dash -c git push origin main"' s''',
        ],
    )
    def test_dash_shell_alias_controls_remain_safe_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (False, None)
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is True
        assert result.get("hardline") is not True

    def test_nested_xargs_csh_push_is_hardline_at_boundaries(
        self, clean_session
    ):
        command = (
            r'''git -c 'alias.p=!printf x | xargs -0 csh -c "git push origin master"' p'''
        )

        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize("shell", ["csh", "/bin/csh"])
    def test_direct_csh_shell_alias_push_is_hardline_at_boundaries(
        self, shell, clean_session
    ):
        command = rf'''git -c 'alias.p=!{shell} -c "git push origin main"' p'''

        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            r'''git -c 'alias.s=!printf x | xargs -0 csh -c "git status --short"' s''',
            r'''git -c 'alias.p=!printf x | xargs -0 csh -c "git push origin feature/topic"' p''',
            r'''git -c 'alias.s=!csh -c "git status --short"' s''',
            r'''git -c 'alias.s=!/bin/csh -c "git status --short"' s''',
            r'''git -c 'alias.s=!printf "%s" "csh -c git push origin main"' s''',
        ],
    )
    def test_csh_shell_alias_controls_remain_safe_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (False, None)
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is True
        assert result.get("hardline") is not True

    def test_nested_xargs_tcsh_push_is_hardline_at_boundaries(
        self, clean_session
    ):
        command = (
            r'''git -c 'alias.p=!printf x | xargs -0 tcsh -c "git push origin master"' p'''
        )

        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize("shell", ["tcsh", "/bin/tcsh"])
    def test_direct_tcsh_shell_alias_push_is_hardline_at_boundaries(
        self, shell, clean_session
    ):
        command = rf'''git -c 'alias.p=!{shell} -c "git push origin main"' p'''

        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            r'''git -c 'alias.s=!printf x | xargs -0 tcsh -c "git status --short"' s''',
            r'''git -c 'alias.p=!printf x | xargs -0 tcsh -c "git push origin feature/topic"' p''',
            r'''git -c 'alias.s=!tcsh -c "git status --short"' s''',
            r'''git -c 'alias.s=!/bin/tcsh -c "git status --short"' s''',
            r'''git -c 'alias.s=!printf "%s" "tcsh -c git push origin main"' s''',
        ],
    )
    def test_tcsh_shell_alias_controls_remain_safe_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (False, None)
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is True
        assert result.get("hardline") is not True

    def test_nested_xargs_shell_eval_push_is_hardline_at_boundaries(
        self, clean_session
    ):
        command = (
            r'''git -c 'alias.p=!printf x | xargs -0 sh -c "eval \"git push origin main\""' p'''
        )

        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            r'''git -c 'alias.s=!echo '\''$(git push origin main)'\''' s''',
            r'''git -c 'alias.s=!echo '\''`git push origin main`'\''' s''',
        ],
    )
    def test_single_quoted_substitution_prose_remains_safe_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (False, None)
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is True
        assert result.get("hardline") is not True

    @pytest.mark.parametrize(
        "command",
        [
            r'''git -c 'alias.s=!printf x | xargs -0 sh -c "eval \"git status --short\""' s''',
            r'''git -c 'alias.p=!printf x | xargs -0 sh -c "eval \"git push origin feature/topic\""' p''',
        ],
    )
    def test_nested_xargs_shell_eval_controls_remain_safe_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (False, None)
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is True
        assert result.get("hardline") is not True

    @pytest.mark.parametrize(
        "command",
        [
            r'''git -c 'alias.p=!echo "$(git push origin main)"' p''',
            r'''git -c 'alias.p=!echo "`git push origin main`"' p''',
        ],
    )
    def test_double_quoted_substitutions_remain_hardline_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            r'''git -c 'alias.s=!echo "\$(git push origin main)"' s''',
            r'''git -c 'alias.s=!echo "\`git push origin main\`"' s''',
        ],
    )
    def test_escaped_substitution_prose_remains_safe_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (False, None)
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is True
        assert result.get("hardline") is not True

    @pytest.mark.parametrize(
        "command",
        [
            (
                "git -c 'alias.p=!printf x | xargs -0 --max-args=1 "
                "sh -c \"git push origin main\"' p"
            ),
            (
                "git -c 'alias.p=!printf x | xargs -0 "
                "sh -c \"git push origin main' p"
            ),
        ],
    )
    def test_shell_alias_xargs_push_is_hardline_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    def test_shell_alias_xargs_work_overflow_with_push_fails_closed(
        self, monkeypatch, clean_session
    ):
        import tools.approval as approval_mod

        monkeypatch.setattr(approval_mod, "_MAX_DETECTION_WORK_ITEMS", 1)
        command = (
            "git -c 'alias.p=!printf x | xargs sh -c \"git status --short\"; "
            "printf x | xargs sh -c \"git push origin main\"' p"
        )

        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            (
                "git -c 'alias.s=!printf x | xargs -0 --max-args=1 "
                "sh -c \"git status --short\"' s"
            ),
            (
                "git -c 'alias.s=!printf x | xargs -0 "
                "sh -c \"git status --short' s"
            ),
            (
                "git -c 'alias.p=!printf x | xargs sh -c "
                "\"git push origin feature/topic\"' p"
            ),
            "git -c 'alias.s=!printf \"%s\" \"xargs sh -c git push origin main\"' s",
        ],
    )
    def test_shell_alias_xargs_controls_remain_safe_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (False, None)
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is True
        assert result.get("hardline") is not True

    @pytest.mark.parametrize(
        "command",
        [
            "git -c 'alias.p=!find . -exec git push origin main {} \\;' p",
            "git -c 'alias.p=!find . -exec git push origin main {}' p",
        ],
    )
    def test_shell_alias_find_exec_push_is_hardline_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    def test_shell_alias_nested_group_find_exec_push_is_hardline_at_boundaries(
        self, clean_session
    ):
        command = (
            "git -c \"alias.p=!{ ( find . -exec git push origin main {} "
            "\\\\; ); }\" p"
        )

        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            (
                "git -c \"alias.p=!{ ( find . -exec git status --short {} "
                "\\\\; ); }\" p"
            ),
            (
                "git -c \"alias.p=!{ ( find . -exec git push origin "
                "feature/topic {} \\\\; ); }\" p"
            ),
            (
                "git -c 'alias.s=!printf \"%s\" "
                "\"{ ( find . -exec git push origin main {} \\; ); }\"' s"
            ),
            (
                "git -c 'alias.s=!printf \"%s\" "
                "\\{\\ \\(\\ find\\ .\\ -exec\\ git\\ push\\ origin\\ main"
                "\\ \\{\\}\\ \\\\;\\ \\)\\;\\ \\}' s"
            ),
        ],
    )
    def test_shell_alias_nested_group_find_controls_remain_safe_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (False, None)
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is True
        assert result.get("hardline") is not True

    def test_shell_alias_find_execdir_shell_push_is_hardline_at_boundaries(
        self, clean_session
    ):
        command = (
            "git -c \"alias.p=!find . -execdir sh -c "
            "'git push origin main' _ {} +\" p"
        )

        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            (
                "git -c \"alias.p=!find . -execdir sh -c "
                "'git status --short' _ {} +\" p"
            ),
            (
                "git -c \"alias.p=!find . -execdir sh -c "
                "'git push origin feature/topic' _ {} +\" p"
            ),
            (
                "git -c 'alias.s=!printf \"%s\" "
                "\"find . -execdir sh -c git push origin main\"' s"
            ),
            (
                "git -c 'alias.s=!printf \"%s\" "
                "find\\ .\\ -execdir\\ sh\\ -c\\ git\\ push\\ origin\\ main' s"
            ),
        ],
    )
    def test_shell_alias_find_execdir_controls_remain_safe_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (False, None)
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is True
        assert result.get("hardline") is not True

    def test_shell_alias_find_exec_work_overflow_with_push_fails_closed(
        self, monkeypatch, clean_session
    ):
        import tools.approval as approval_mod

        monkeypatch.setattr(approval_mod, "_MAX_DETECTION_WORK_ITEMS", 1)
        command = (
            "git -c 'alias.p=!find . -exec git status --short {} \\;; "
            "find . -exec git push origin main {} \\;' p"
        )

        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            "git -c 'alias.s=!find . -exec git status --short {} \\;' s",
            "git -c 'alias.s=!find . -exec git status --short {}' s",
            (
                "git -c 'alias.p=!find . -exec git push origin "
                "feature/topic {} \\;' p"
            ),
            "git -c 'alias.s=!printf \"%s\" \"find . -exec git push origin main\"' s",
        ],
    )
    def test_shell_alias_find_exec_controls_remain_safe_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (False, None)
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is True
        assert result.get("hardline") is not True

    @pytest.mark.parametrize(
        "command",
        [
            "git -c 'alias.p=!if true; then git push origin main; fi' p",
            "git -c 'alias.p=!case x in x) git push origin main;; esac' p",
            "git -c 'alias.p=!for branch in x; do git push origin main; done' p",
            "git -c 'alias.p=!! git push origin main' p",
            "git -c 'alias.p=!if true; then git push origin main' p",
        ],
    )
    def test_shell_control_alias_pushes_are_hardline_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            "git -c 'alias.s=!if true; then git status --short; fi' s",
            "git -c 'alias.s=!case x in x) git status --short;; esac' s",
            "git -c 'alias.s=!for branch in x; do git status --short; done' s",
            "git -c 'alias.s=!! git status --short' s",
            (
                "git -c 'alias.s=!if true; then "
                "echo \"git push origin main\"; fi' s"
            ),
            (
                "git -c 'alias.s=!case x in x) "
                "printf \"%s\" \"git push origin main\";; esac' s"
            ),
        ],
    )
    def test_shell_control_aliases_preserve_status_and_quoted_prose(self, command):
        assert detect_hardline_command(command) == (False, None)

    @pytest.mark.parametrize(
        "command",
        [
            "git -c 'alias.p=-c color.ui=false push' p",
            "git -c 'alias.p=-ccolor.ui=false push origin main' p",
            (
                "git -c 'alias.p=--git-dir=/synthetic/repo "
                "push origin master' p"
            ),
            "git -c 'alias.p=--work-tree /synthetic/tree push' p",
            "git -c 'alias.p=-c color.ui=false' p origin main",
        ],
    )
    def test_global_option_prefixed_alias_pushes_are_hardline_at_boundaries(
        self, command, clean_session
    ):
        assert detect_hardline_command(command) == (
            True,
            "push to protected default branch (main/master)",
        )
        result = check_all_command_guards(command, "docker")
        assert result["approved"] is False
        assert result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            "git -c 'alias.p=-c alias.q=push q' p",
            "git -c 'alias.p=-calias.q=push q' p",
            "git -c 'alias.p=-c alias.q=r -calias.r=push q' p",
        ],
    )
    def test_alias_local_config_recursion_is_hardline_at_boundaries(
        self, command, clean_session
    ):
        detector_result = detect_hardline_command(command)
        guard_result = check_all_command_guards(command, "docker")

        assert detector_result == (
            True,
            "push to protected default branch (main/master)",
        )
        assert guard_result["approved"] is False
        assert guard_result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            "git -c 'alias.p=-c alias.q=r -c alias.r=q q' p",
            "git -c 'alias.p=--config-env=alias.q=PUSH_ALIAS q' p",
            "git -c 'alias.p=--config-env alias.q=PUSH_ALIAS q' p",
            (
                "git -c 'alias.p=-c alias.q=a -c alias.a=b -c alias.b=c "
                "-c alias.c=d -c alias.d=e -c alias.e=f -c alias.f=g "
                "-c alias.g=h -c alias.h=i -c alias.i=push q' p"
            ),
        ],
    )
    def test_alias_local_config_ambiguity_fails_closed_at_boundaries(
        self, command, clean_session
    ):
        detector_result = detect_hardline_command(command)
        guard_result = check_all_command_guards(command, "docker")

        assert detector_result == (
            True,
            "push to protected default branch (main/master)",
        )
        assert guard_result["approved"] is False
        assert guard_result["hardline"] is True

    def test_alias_local_config_entry_overflow_fails_closed_at_boundaries(
        self, clean_session
    ):
        local_entries = " ".join(
            f"-c alias.local{index}=status" for index in range(65)
        )
        command = f"git -c 'alias.p={local_entries} q' p"

        detector_result = detect_hardline_command(command)
        guard_result = check_all_command_guards(command, "docker")

        assert detector_result == (
            True,
            "push to protected default branch (main/master)",
        )
        assert guard_result["approved"] is False
        assert guard_result["hardline"] is True

    @pytest.mark.parametrize(
        "command",
        [
            "git -c 'alias.s=-c alias.t=status t' s --short",
            "git -c 'alias.s=-calias.t=status t' s --short",
            "git -c 'alias.s=-c alias.t=u -calias.u=status t' s --short",
            "git -c 'alias.p=-c alias.q=push q origin feature/topic' p",
            "git -c 'alias.s=--config-env=alias.t=STATUS_ALIAS status' s",
            (
                "git -c 'alias.p=-c alias.q=r -c alias.r=q q' "
                "p origin feature/topic"
            ),
            (
                "git -c 'alias.p=--config-env=alias.q=PUSH_ALIAS q' "
                "p origin feature/topic"
            ),
        ],
    )
    def test_alias_local_config_preserves_status_and_feature_controls(self, command):
        assert detect_hardline_command(command) == (False, None)

    def test_alias_local_config_entry_overflow_preserves_feature_control(self):
        local_entries = " ".join(
            f"-c alias.local{index}=status" for index in range(65)
        )
        command = f"git -c 'alias.p={local_entries} q' p origin feature/topic"

        assert detect_hardline_command(command) == (False, None)

    @pytest.mark.parametrize(
        "command",
        [
            "git -c 'alias.p=-c color.ui=false push origin feature/topic' p",
            "git -c 'alias.p=-ccolor.ui=false push origin feature/topic' p",
            "git -c 'alias.s=-c color.ui=false status' s",
            "git -c 'alias.s=--git-dir=/synthetic/repo status' s",
            "git -c 'alias.p=-c color.ui=false' p",
        ],
    )
    def test_global_option_prefixed_aliases_preserve_safe_controls(self, command):
        assert detect_hardline_command(command) == (False, None)


class TestTopLevelProtectedPushShellVariables:
    protected_rule = "push to protected default branch (main/master)"

    @staticmethod
    def _assert_boundaries(command, *, blocked, clean_session, monkeypatch):
        import tools.approval as approval_mod
        from hermes_cli import approvals_test as approvals_test_mod

        monkeypatch.setattr(
            approval_mod, "_get_approval_config", lambda: {"mode": "off"}
        )
        monkeypatch.setattr(approval_mod, "_YOLO_MODE_FROZEN", False)
        monkeypatch.setattr(
            approval_mod, "is_current_session_yolo_enabled", lambda: False
        )
        monkeypatch.setattr(approval_mod, "load_permanent_allowlist", lambda: set())

        detector = detect_hardline_command(command)
        guard = check_all_command_guards(command, "local")
        verdict = approvals_test_mod.evaluate_command(command)
        if blocked:
            assert detector == (
                True,
                TestTopLevelProtectedPushShellVariables.protected_rule,
            )
            assert guard["approved"] is False
            assert guard["hardline"] is True
            assert verdict["verdict"] == "hardline-deny"
            assert verdict["exit_code"] == 3
        else:
            assert detector == (False, None)
            assert guard["approved"] is True
            assert guard.get("hardline") is not True
            assert verdict["verdict"] == "allow"
            assert verdict["exit_code"] == 0

    @pytest.mark.parametrize(
        "command",
        [
            "G=git; $G push origin main",
            "G='git'; ${G} push origin master",
            'G=git; "$G" push origin main',
            'G=git; "${G}" push origin master',
            "G=git; $G'' push origin main",
            "G=git; /usr/bin/$G push origin main",
            "G=g; ${G}it push origin master",
            "A=x G=/usr/local/bin/git; $G push origin main",
            "A=x; G=git; ${G} push origin master",
        ],
    )
    def test_b1_raw_executable_composition_blocks_protected_pushes(
        self, command, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            command, blocked=True, clean_session=clean_session, monkeypatch=monkeypatch
        )

    @pytest.mark.parametrize(
        "command",
        [
            "G=printf; $G push origin main",
            "G=git; '$G' push origin main",
            r"G=git; \$G push origin main",
            "G=git; echo '$G push origin main'",
            r'G=git; printf "%s" "\$G push origin main"',
            "G=git; $G status",
            "G=git; $G push origin feature/topic",
            "G=git; ${G} push origin HEAD:refs/heads/feature/topic",
        ],
    )
    def test_b1_quote_provenance_and_nonprotected_controls_remain_safe(
        self, command, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            command, blocked=False, clean_session=clean_session, monkeypatch=monkeypatch
        )

    @pytest.mark.parametrize(
        "command",
        [
            "G=git; env $G push origin main",
            "G=git; env -u UNUSED $G push origin master",
            "G=git; nice -n 5 $G push origin main",
        ],
    )
    def test_core_wrapper_argv_blocks_variable_backed_protected_pushes(
        self, command, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            command, blocked=True, clean_session=clean_session, monkeypatch=monkeypatch
        )

    @pytest.mark.parametrize(
        "command",
        [
            "G=printf; env $G push origin main",
            "G=git; env $G status",
            "G=git; env -u UNUSED $G push origin feature/topic",
            "G=git; nice -n 5 $G status",
            'G=git; printf "%s" "env $G push origin main"',
        ],
    )
    def test_core_wrapper_argv_preserves_status_feature_and_prose_controls(
        self, command, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            command, blocked=False, clean_session=clean_session, monkeypatch=monkeypatch
        )

    @pytest.mark.parametrize(
        "command",
        [
            "env -S 'git push origin main'",
            "env --split-string='git push origin master'",
            "env -S 'git push origin main",
            'env -S "git \'push origin master"',
            r"env -S 'git push origin main \q'",
        ],
    )
    def test_core_env_split_string_payload_blocks_protected_pushes(
        self, command, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            command, blocked=True, clean_session=clean_session, monkeypatch=monkeypatch
        )

    @pytest.mark.parametrize(
        "command",
        [
            "env -S 'git status --short'",
            "env --split-string='git push origin feature/topic'",
            "env -S 'printf \"%s\" \"git push origin main\"'",
            "env -S 'git status",
            'env -S "printf \'%s\' \'git push origin main"',
            r"env -S 'git status\q'",
            "printf '%s' \"env -S 'git push origin main'\"",
        ],
    )
    def test_core_env_split_string_preserves_status_feature_and_prose_controls(
        self, command, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            command, blocked=False, clean_session=clean_session, monkeypatch=monkeypatch
        )

    def test_core_alias_definition_does_not_hide_sibling_protected_substitution(
        self, clean_session, monkeypatch
    ):
        command = 'git -c alias.s=status "$(G=git; $G push origin main)"'
        self._assert_boundaries(
            command, blocked=True, clean_session=clean_session, monkeypatch=monkeypatch
        )

    @pytest.mark.parametrize(
        "command",
        [
            'git -c "alias.s=status$(G=git; $G push origin main)" status',
            "git -calias.s=status$(G=git; $G push origin master) status",
        ],
    )
    def test_core_alias_same_word_outer_substitution_blocks_protected_pushes(
        self, command, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            command, blocked=True, clean_session=clean_session, monkeypatch=monkeypatch
        )

    @pytest.mark.parametrize(
        "command",
        [
            'git -c "alias.s=status$(G=git; $G status --short)" status',
            (
                'git -c "alias.s=status$(G=git; '
                '$G push origin feature/topic)" status'
            ),
            "git -c 'alias.s=status$(G=git; $G push origin main)' status",
            "git -calias.s=status'$(G=git; $G push origin main)' status",
            (
                'git -c "alias.s=status$(printf \'%s\' '
                "'git push origin main')\" status"
            ),
        ],
    )
    def test_core_alias_same_word_substitution_controls_remain_safe(
        self, command, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            command, blocked=False, clean_session=clean_session, monkeypatch=monkeypatch
        )

    @pytest.mark.parametrize(
        "command",
        [
            "git -c alias.s=status",
            "git -c alias.s=status s --short",
            'git -c alias.s=status "$(G=git; $G status)"',
            'git -c alias.s=status "$(G=git; $G push origin feature/topic)"',
            "git -c alias.s=status '$(G=git; $G push origin main)'",
            (
                "git -c alias.s=status "
                '"$(printf \'%s\' \'G=git; $G push origin main\')"'
            ),
        ],
    )
    def test_core_alias_definition_preserves_status_feature_and_prose_controls(
        self, command, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            command, blocked=False, clean_session=clean_session, monkeypatch=monkeypatch
        )

    @pytest.mark.parametrize(
        "command",
        [
            "cat <<'EOF'\nG=git; $G push origin main\nEOF",
            "cat <<'EOF'\n$(G=git; $G push origin main)\nEOF",
            "cat <<EOF\ngit push origin master\nEOF",
            "cat <<-'EOF'\n\tG=git; $G push origin main\n\tEOF",
            (
                "cat <<A <<B\n"
                "G=git; $G push origin main\nA\n"
                "rm -rf /\nB"
            ),
            "printf ok # git push origin main",
            "printf ok # rm -rf /",
            "printf '%s' '# git push origin main'",
            r"printf '%s' \# git push origin main",
            "printf ok; " + "\\" + "\n# rm -rf /\nprintf done",
        ],
    )
    def test_core_hardline_semantic_text_ownership_preserves_safe_data(
        self, command, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            command, blocked=False, clean_session=clean_session, monkeypatch=monkeypatch
        )

    @pytest.mark.parametrize(
        "command",
        [
            (
                "cat <<'EOF'\n"
                "G=git; $G status --short\nEOF\n"
                "G=git; $G push origin main"
            ),
            "printf ok # comment ends here\ngit push origin master",
            "printf '%s' '#' ; G=git; $G push origin main",
            r"printf '%s' \#; git push origin master",
            "cat <<EOF\n$(G=git; $G push origin main)\nEOF",
        ],
    )
    def test_core_hardline_semantic_text_ownership_preserves_real_commands(
        self, command, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            command, blocked=True, clean_session=clean_session, monkeypatch=monkeypatch
        )

    @pytest.mark.parametrize(
        "case",
        ["delimiter", "too-many", "premature-eof", "malformed-delimiter"],
    )
    def test_core_hardline_semantic_parser_limits_fail_closed_at_boundaries(
        self, case, clean_session, monkeypatch
    ):
        import tools.approval as approval_mod
        from hermes_cli import approvals_test as approvals_test_mod

        monkeypatch.setattr(
            approval_mod, "_get_approval_config", lambda: {"mode": "off"}
        )
        monkeypatch.setattr(approval_mod, "_save_blocked_payload", lambda _cmd: None)
        if case == "delimiter":
            delimiter = "D" * (approval_mod._MAX_HEREDOC_DELIMITER_CHARS + 1)
            command = f"cat <<{delimiter}\ngit push origin main\n{delimiter}"
        elif case == "too-many":
            monkeypatch.setattr(
                approval_mod, "_MAX_PENDING_HEREDOCS", 1, raising=False
            )
            command = "cat <<A <<B\nprintf ok\nA\ngit push origin main\nB"
        elif case == "premature-eof":
            command = "cat <<EOF\nG=git; $G push origin main"
        else:
            command = "cat <<'EOF\nG=git; $G push origin main\nEOF"

        assert detect_hardline_command(command) == (
            True,
            approval_mod._PARSER_LIMIT_DESCRIPTION,
        )
        guard = check_all_command_guards(command, "local")
        assert guard["approved"] is False
        assert guard["hardline"] is True
        assert approval_mod._PARSER_LIMIT_DESCRIPTION in guard["message"]
        verdict = approvals_test_mod.evaluate_command(command)
        assert verdict["verdict"] == "hardline-deny"
        assert verdict["exit_code"] == 3
        assert verdict["rule"] == approval_mod._PARSER_LIMIT_DESCRIPTION

    _LINE_CONTINUATION = "\\" + "\n"

    @pytest.mark.parametrize(
        "command",
        [
            "G=g; ${G}" + _LINE_CONTINUATION + "it push origin main",
            'G=g; "${G}' + _LINE_CONTINUATION + 'it" push origin master',
        ],
    )
    def test_core_line_continuation_composes_protected_git_executable(
        self, command, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            command, blocked=True, clean_session=clean_session, monkeypatch=monkeypatch
        )

    @pytest.mark.parametrize(
        "command",
        [
            r'G="g\it"; $G push origin main',
            r'G="g\qt"; $G push origin master',
            r"G='g\it'; $G push origin main",
            r"G=git; \$G push origin main",
            "G=g; ${G}" + _LINE_CONTINUATION + "it status",
            (
                'G=g; "${G}'
                + _LINE_CONTINUATION
                + 'it" push origin feature/topic'
            ),
            "G=g; '${G}" + _LINE_CONTINUATION + "it' push origin main",
        ],
    )
    def test_core_backslash_quote_feature_and_status_controls_remain_safe(
        self, command, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            command, blocked=False, clean_session=clean_session, monkeypatch=monkeypatch
        )

    @pytest.mark.parametrize(
        "command",
        [
            "G=git; G=printf $G push origin main",
            "G=git; $G push origin main; G=printf",
        ],
    )
    def test_b2_same_command_and_source_order_use_prior_state(
        self, command, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            command, blocked=True, clean_session=clean_session, monkeypatch=monkeypatch
        )

    @pytest.mark.parametrize(
        "command",
        [
            "G=printf; G=git $G push origin main",
            "G=printf; $G push origin main; G=git",
        ],
    )
    def test_b2_prefix_and_later_assignments_never_rewrite_the_command(
        self, command, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            command, blocked=False, clean_session=clean_session, monkeypatch=monkeypatch
        )

    _B3_POISONS = [
        "echo ready",
        "printf '%s' ready",
        "true",
        "false",
        "printf -v G git",
        "X=$OTHER",
        "X=$(printf git)",
        "unset G",
        "export G=git",
        "readonly G=git",
        "declare G=git",
        "read G",
        "eval 'G=git'",
        "source /synthetic/script",
        ". /synthetic/script",
        "alias G=git",
        "f(){ G=git; }; f",
        "trap 'G=git' DEBUG",
        "if true; then G=git; fi",
    ]

    @pytest.mark.parametrize("poison", _B3_POISONS)
    def test_b3_every_executable_dynamic_or_control_event_sticky_poisons_state(
        self, poison, clean_session, monkeypatch
    ):
        command = f"G=printf; {poison}; $G push origin main"
        self._assert_boundaries(
            command, blocked=True, clean_session=clean_session, monkeypatch=monkeypatch
        )

    @pytest.mark.parametrize("poison", _B3_POISONS)
    @pytest.mark.parametrize(
        "suffix",
        [
            "$G status",
            "$G push origin feature/topic",
            "echo '$G push origin main'",
            r"echo \$G push origin main",
        ],
    )
    def test_b3_poisoned_status_feature_and_prose_controls_remain_safe(
        self, poison, suffix, clean_session, monkeypatch
    ):
        command = f"G=printf; {poison}; {suffix}"
        self._assert_boundaries(
            command, blocked=False, clean_session=clean_session, monkeypatch=monkeypatch
        )

    _B4_BARRIERS = [
        "G=printf | G=printf;",
        "G=printf && G=printf;",
        "G=printf || G=printf;",
        "G=printf & G=printf;",
        "(G=printf);",
        "{ G=printf; };",
        "( { G=printf; }; );",
        "f(){ G=printf; }; f;",
        "case x in x) G=printf ;; esac;",
        "for x in one; do G=printf; done;",
        "while false; do G=printf; done;",
    ]

    @pytest.mark.parametrize("prefix", _B4_BARRIERS)
    def test_b4_connectors_groups_and_control_constructs_are_barriers(
        self, prefix, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            f"{prefix} $G push origin main",
            blocked=True,
            clean_session=clean_session,
            monkeypatch=monkeypatch,
        )

    @pytest.mark.parametrize("prefix", _B4_BARRIERS)
    @pytest.mark.parametrize(
        "suffix",
        [
            "$G status",
            "$G push origin feature/topic",
            "echo '$G push origin main'",
            r"echo \$G push origin main",
        ],
    )
    def test_b4_barrier_status_feature_and_prose_controls_remain_safe(
        self, prefix, suffix, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            f"{prefix} {suffix}",
            blocked=False,
            clean_session=clean_session,
            monkeypatch=monkeypatch,
        )

    @pytest.mark.parametrize(
        "command",
        [
            "git -c 'alias.s=!G=git; $G push origin main; G=printf' s",
            "sh -c 'G=git; $G push origin main; G=printf'",
            "echo \"$(G=git; $G push origin main; G=printf)\"",
        ],
    )
    def test_b5_alias_and_raw_payload_source_order_blocks_protected_pushes(
        self, command, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            command,
            blocked=True,
            clean_session=clean_session,
            monkeypatch=monkeypatch,
        )

    @pytest.mark.parametrize(
        "command",
        [
            "git -c 'alias.s=!G=printf; $G push origin main; G=git' s",
            "sh -c 'printf \"%s\" \"G=git; $G push origin main\"'",
            "echo \"$(printf '%s' 'G=git; $G push origin main')\"",
        ],
    )
    def test_b5_later_bindings_and_quoted_payload_prose_remain_safe(
        self, command, clean_session, monkeypatch
    ):
        self._assert_boundaries(
            command,
            blocked=False,
            clean_session=clean_session,
            monkeypatch=monkeypatch,
        )

    def test_b7_program_resolution_records_assignment_and_command_events(self):
        import tools.approval as approval_mod

        program = approval_mod._resolve_shell_program(
            "G=git; G=printf $G push origin main"
        )

        assert program.status == approval_mod._ShellVariableStatus.RESOLVED
        assert [event.kind for event in program.events] == [
            "assignment",
            "executable",
        ]
        assignment, command = program.events
        assert assignment.prefix_assignments == (("G", "git"),)
        assert command.prefix_assignments == (("G", "printf"),)
        assert command.executable is not None
        assert command.executable.value == "git"
        assert command.resolved_basename == "git"
        assert command.trusted_bindings == (("G", "git"),)

    def test_b7_word_resolution_accounts_fragments_replacements_and_bytes(self):
        import tools.approval as approval_mod

        budget = approval_mod._ShellVariableBudget()
        resolution = approval_mod._resolve_shell_word(
            "pre${G}post", {"G": "git"}, budget
        )

        assert resolution.status == approval_mod._ShellVariableStatus.RESOLVED
        assert resolution.value == "pregitpost"
        assert [fragment.kind for fragment in resolution.fragments] == [
            "literal",
            "parameter",
            "literal",
        ]
        assert budget.fragments == 3
        assert budget.replacements == 1
        assert budget.emitted_bytes == len("pregitpost".encode())

    @pytest.mark.parametrize(
        ("constant", "resolver_kind", "source"),
        [
            ("_MAX_GIT_LEADING_ASSIGNMENTS", "program", "G=git"),
            ("_MAX_DETECTION_WORK_ITEMS", "program", "true"),
            ("_MAX_SHELL_VARIABLE_FRAGMENTS", "word", "literal"),
            ("_MAX_SHELL_VARIABLE_REPLACEMENTS", "word", "$G"),
        ],
    )
    def test_b7_direct_budget_caps_return_overflow(
        self, constant, resolver_kind, source, monkeypatch
    ):
        import tools.approval as approval_mod

        monkeypatch.setattr(approval_mod, constant, 0)
        budget = approval_mod._ShellVariableBudget()
        if resolver_kind == "program":
            resolution = approval_mod._resolve_shell_program(source, budget=budget)
        else:
            resolution = approval_mod._resolve_shell_word(source, {}, budget)

        assert resolution.status == approval_mod._ShellVariableStatus.OVERFLOW
        assert budget.overflowed is True

    @pytest.mark.parametrize(
        ("constant", "command"),
        [
            (
                "_MAX_GIT_LEADING_ASSIGNMENTS",
                "G=git; $G push origin main",
            ),
            (
                "_MAX_DETECTION_WORK_ITEMS",
                "G=git; $G push origin main",
            ),
            (
                "_MAX_SHELL_VARIABLE_FRAGMENTS",
                "G=git; $G push origin main",
            ),
            (
                "_MAX_SHELL_VARIABLE_REPLACEMENTS",
                "$G push origin main",
            ),
        ],
    )
    def test_core_caller_budget_caps_fail_closed_without_materializing_overflow(
        self, constant, command, clean_session, monkeypatch
    ):
        import tools.approval as approval_mod
        from hermes_cli import approvals_test as approvals_test_mod

        materialized = []

        def record_materialization(parts):
            materialized.append(tuple(parts))
            return "".join(parts)

        monkeypatch.setattr(approval_mod, constant, 0)
        monkeypatch.setattr(
            approval_mod, "_materialize_shell_word", record_materialization
        )
        monkeypatch.setattr(approval_mod, "_save_blocked_payload", lambda _cmd: None)
        monkeypatch.setattr(
            approval_mod, "_get_approval_config", lambda: {"mode": "off"}
        )
        monkeypatch.setattr(approval_mod, "_YOLO_MODE_FROZEN", False)
        monkeypatch.setattr(
            approval_mod, "is_current_session_yolo_enabled", lambda: False
        )
        monkeypatch.setattr(approval_mod, "load_permanent_allowlist", lambda: set())

        assert detect_hardline_command(command) == (
            True,
            approval_mod._PARSER_LIMIT_DESCRIPTION,
        )
        assert materialized == []

        materialized.clear()
        guard = check_all_command_guards(command, "local")
        assert guard["approved"] is False
        assert guard["hardline"] is True
        assert approval_mod._PARSER_LIMIT_DESCRIPTION in guard["message"]
        assert materialized == []

        materialized.clear()
        verdict = approvals_test_mod.evaluate_command(command)
        assert verdict["verdict"] == "hardline-deny"
        assert verdict["exit_code"] == 3
        assert materialized == []

    def test_b7_unsupported_parameter_syntax_is_unresolved_not_overflow(self):
        import tools.approval as approval_mod

        budget = approval_mod._ShellVariableBudget()
        resolution = approval_mod._resolve_shell_word("${G:-git}", {}, budget)

        assert resolution.status == approval_mod._ShellVariableStatus.UNRESOLVED
        assert resolution.value is None
        assert budget.overflowed is False

    def test_b7_emitted_byte_overflow_never_materializes_output(
        self, monkeypatch
    ):
        import tools.approval as approval_mod

        materialized = []
        monkeypatch.setattr(approval_mod, "_MAX_SHELL_VARIABLE_EMITTED_BYTES", 3)
        monkeypatch.setattr(
            approval_mod,
            "_materialize_shell_word",
            lambda parts: materialized.append(parts) or "unexpected",
        )
        budget = approval_mod._ShellVariableBudget()

        resolution = approval_mod._resolve_shell_word("abcd", {}, budget)

        assert resolution.status == approval_mod._ShellVariableStatus.OVERFLOW
        assert resolution.value is None
        assert budget.overflowed is True
        assert materialized == []

    def test_b7_recursive_payload_uses_containing_source_order_state(self):
        import tools.approval as approval_mod

        budget = approval_mod._ShellVariableBudget()
        protected, status = approval_mod._shell_variable_default_branch_push(
            'G=git; echo "$($G push origin main)"', budget=budget
        )

        assert protected is True
        assert status == approval_mod._ShellVariableStatus.RESOLVED
        assert budget.payload_work == 1
        assert budget.depth == 1

    @pytest.mark.parametrize(
        ("constant", "limit", "initial_payload_work"),
        [
            ("_MAX_DETECTION_WORK_ITEMS", 2, 2),
            ("_MAX_GIT_SHELL_RECURSION", 0, 0),
        ],
    )
    def test_b7_recursive_payload_caps_return_overflow(
        self, constant, limit, initial_payload_work, monkeypatch
    ):
        import tools.approval as approval_mod

        monkeypatch.setattr(approval_mod, constant, limit)
        budget = approval_mod._ShellVariableBudget(
            payload_work=initial_payload_work
        )
        protected, status = approval_mod._shell_variable_default_branch_push(
            'echo "$(printf ready)"', budget=budget
        )

        assert protected is False
        assert status == approval_mod._ShellVariableStatus.OVERFLOW
        assert budget.overflowed is True

    def test_b7_payload_findings_overflow_surfaces_parser_limit_at_boundaries(
        self, clean_session, monkeypatch
    ):
        import tools.approval as approval_mod
        from hermes_cli import approvals_test as approvals_test_mod

        monkeypatch.setattr(approval_mod, "_MAX_PAYLOAD_FINDINGS_PER_VARIANT", 1)
        monkeypatch.setattr(
            approval_mod, "_get_approval_config", lambda: {"mode": "off"}
        )
        command = 'sh -c "printf ready" "$(printf one)"'

        protected, status = approval_mod._shell_variable_default_branch_push(command)
        assert protected is False
        assert status == approval_mod._ShellVariableStatus.OVERFLOW
        assert detect_hardline_command(command) == (
            True,
            approval_mod._PARSER_LIMIT_DESCRIPTION,
        )
        guard = check_all_command_guards(command, "local")
        assert guard["approved"] is False
        assert guard["hardline"] is True
        verdict = approvals_test_mod.evaluate_command(command)
        assert verdict["verdict"] == "hardline-deny"
        assert verdict["exit_code"] == 3

    def test_b7_synthesized_git_event_disables_variable_recursion(
        self, monkeypatch
    ):
        import tools.approval as approval_mod

        program = approval_mod._resolve_shell_program(
            "G=git; $G push origin main"
        )
        command_event = program.events[-1]

        def unexpected_recursion(*_args, **_kwargs):
            raise AssertionError("synthesized Git recursively resolved variables")

        monkeypatch.setattr(
            approval_mod,
            "_shell_variable_default_branch_push",
            unexpected_recursion,
        )
        assert approval_mod._shell_event_default_branch_push(command_event, 0) is True


class TestHardlinePublicDiagnostics:
    @pytest.mark.parametrize(
        "caller_name", ["check_dangerous_command", "check_all_command_guards"]
    )
    def test_credential_bearing_protected_push_has_fixed_warning_record(
        self, caller_name, caplog
    ):
        import tools.approval as approval_mod

        command = (
            "git push "
            "https://private-user:sensitive-credential-fixture@"
            "example.invalid/Users/private-user/private-repo main"
        )
        caller = getattr(approval_mod, caller_name)

        with caplog.at_level(logging.WARNING, logger="tools.approval"):
            result = caller(command, "local")

        assert result["approved"] is False
        assert result.get("hardline") is True
        assert [
            (record.levelno, record.getMessage(), record.args, record.exc_info)
            for record in caplog.records
        ] == [(logging.WARNING, "command_hardline_blocked", (), None)]
        public_text = result["message"] + caplog.text
        for private_text in (
            command,
            "sensitive-credential-fixture",
            "private-user",
            "/Users/private-user",
        ):
            assert private_text not in public_text

    @pytest.mark.parametrize(
        "caller_name", ["check_dangerous_command", "check_all_command_guards"]
    )
    def test_user_deny_has_fixed_warning_record(
        self, caller_name, monkeypatch, caplog
    ):
        import tools.approval as approval_mod

        command = "deploy /Users/private-user/private-directory"
        monkeypatch.setattr(
            approval_mod,
            "_get_approval_config",
            lambda: {"mode": "manual", "deny": ["deploy *"]},
        )
        caller = getattr(approval_mod, caller_name)

        with caplog.at_level(logging.WARNING, logger="tools.approval"):
            result = caller(command, "local")

        assert result["approved"] is False
        assert result.get("user_deny") is True
        assert [
            (record.levelno, record.getMessage(), record.args, record.exc_info)
            for record in caplog.records
        ] == [(logging.WARNING, "command_user_deny_blocked", (), None)]
        assert "/Users/private-user" not in result["message"] + caplog.text

    def test_parser_limit_recovery_uses_neutral_validated_script_reference(
        self, monkeypatch
    ):
        import tools.approval as approval_mod

        saved = (
            "/Users/private-user/.hermes/cache/blocked-scripts/"
            "blocked-1234567890-deadbeef.sh"
        )
        monkeypatch.setattr(approval_mod, "_save_blocked_payload", lambda _cmd: saved)

        result = approval_mod._hardline_block_result(
            approval_mod._PARSER_LIMIT_DESCRIPTION, "echo oversized"
        )

        neutral = (
            "$HERMES_HOME/cache/blocked-scripts/"
            "blocked-1234567890-deadbeef.sh"
        )
        assert neutral in result["message"]
        assert saved not in result["message"]
        assert "/Users/" not in result["message"]
        assert "private-user" not in result["message"]

    def test_parser_limit_recovery_rejects_unvalidated_script_basename(
        self, monkeypatch
    ):
        import tools.approval as approval_mod

        saved = "/Users/private-user/.hermes/cache/blocked-scripts/private.txt"
        monkeypatch.setattr(approval_mod, "_save_blocked_payload", lambda _cmd: saved)

        result = approval_mod._hardline_block_result(
            approval_mod._PARSER_LIMIT_DESCRIPTION, "echo oversized"
        )

        assert saved not in result["message"]
        assert "/Users/" not in result["message"]
        assert "$HERMES_HOME/cache/blocked-scripts/" not in result["message"]

    def test_blocked_payload_save_failure_has_fixed_metadata_free_record(
        self, monkeypatch, caplog
    ):
        import hermes_constants
        import tools.approval as approval_mod

        command = "private command at /Users/private-user/private-directory"

        def fail_home_lookup():
            raise RuntimeError("private failure at /Users/private-user/.hermes")

        monkeypatch.setattr(hermes_constants, "get_hermes_home", fail_home_lookup)

        with caplog.at_level(logging.DEBUG, logger="tools.approval"):
            assert approval_mod._save_blocked_payload(command) is None

        assert [
            (record.levelno, record.getMessage(), record.args, record.exc_info)
            for record in caplog.records
        ] == [(logging.DEBUG, "blocked_payload_save_failed", (), None)]
        assert command not in caplog.text
        assert "private failure" not in caplog.text
        assert "/Users/private-user" not in caplog.text
