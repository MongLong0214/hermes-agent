"""hermes -z must not exit 0 when the turn was discarded/failed (#lease-timeout-rc).

The turn-lease-timeout path in run_agent.py returns a result shaped like:
    {"final_response": <friendly non-empty text>, "completed": False,
     "failed": True, "error": "session_turn_lease_timeout:..."}

The user's original message was never run. Because the friendly text is
non-empty, the old rc computation in oneshot.py:
    if (result.get("failed") or result.get("partial")) and not response.strip():
        return 2
    if not response.strip():
        return 1
    return 0
skipped both guards and returned 0 -- a caller piping `hermes -z` in a script
cannot tell the message was dropped.

This module invokes the real ``run_oneshot`` (not a mock of it) in a
subprocess, monkeypatching only its internal ``_run_agent`` collaborator --
mirrors the established pattern in test_oneshot_surrogate.py.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_oneshot_exits_nonzero_on_failed_result_with_nonempty_response():
    """A lease-timeout-shaped result (failed=True, non-empty text) must not exit 0."""
    program = textwrap.dedent(
        """
        import hermes_cli.oneshot as oneshot

        friendly = (
            "\\u23f3 Another Hermes process kept this session busy too long. "
            "Your message was not processed - wait for the other process to "
            "finish, then send it again."
        )
        oneshot._run_agent = lambda *args, **kwargs: (
            friendly,
            {
                "final_response": friendly,
                "messages": [],
                "api_calls": 0,
                "completed": False,
                "failed": True,
                "error": "session_turn_lease_timeout:some-session-id",
            },
        )
        raise SystemExit(oneshot.run_oneshot("hello"))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0, (
        "hermes -z exited 0 after discarding the user's message "
        f"(stdout={result.stdout!r}, stderr={result.stderr!r})"
    )


def test_oneshot_exits_zero_on_normal_completed_turn_with_text():
    """A normal completed turn with text must still exit 0 (no regression)."""
    program = textwrap.dedent(
        """
        import hermes_cli.oneshot as oneshot

        oneshot._run_agent = lambda *args, **kwargs: (
            "Paris.",
            {
                "final_response": "Paris.",
                "messages": [],
                "api_calls": 1,
                "completed": True,
                "failed": False,
                "partial": False,
            },
        )
        raise SystemExit(oneshot.run_oneshot("What's the capital of France?"))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert result.stdout.strip() == b"Paris."

