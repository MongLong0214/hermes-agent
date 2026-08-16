"""Re-installing the routing proxy must not open another devnull handle.

The install is idempotent only while ``sys.<attr>`` is still the proxy we put
there. Anything that rebinds it makes the next call re-install, and before this
was fixed each re-install opened a fresh ``/dev/null`` that nobody closed and
chained the previous proxy in as its passthrough, so it could not be collected
either. A gateway accumulated 654 such handles over three days and then failed
with EMFILE while reading an unrelated JSON file.

The count is what these assert. A test that only checked "silencing still works"
passed throughout the leak.
"""

from __future__ import annotations

import sys

from agent import thread_scoped_output as tso


def _open_devnull_writers() -> int:
    """Write handles this process holds on /dev/null, counted from the OS."""
    import subprocess

    out = subprocess.run(
        ["lsof", "-p", str(__import__("os").getpid())],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return sum(
        1
        for line in out.splitlines()
        if line.rstrip().endswith("/dev/null") and line.split()[3].endswith("w")
    )


def test_reinstall_after_rebind_opens_no_further_handles() -> None:
    with tso.thread_scoped_silence():
        pass
    before = _open_devnull_writers()

    # Exactly the condition that defeated the idempotence guard in production:
    # something else owns sys.stdout/sys.stderr when silencing is next entered.
    for _ in range(25):
        real_out, real_err = sys.stdout, sys.stderr
        try:
            sys.stdout = open(__import__("os").devnull, "w", encoding="utf-8")
            sys.stderr = open(__import__("os").devnull, "w", encoding="utf-8")
            impostors = (sys.stdout, sys.stderr)
            with tso.thread_scoped_silence():
                pass
        finally:
            for fh in impostors:
                fh.close()
            sys.stdout, sys.stderr = real_out, real_err

    after = _open_devnull_writers()
    assert after == before, (
        f"re-installing opened {after - before} devnull handle(s) that nobody closes"
    )


def test_passthrough_never_chains_onto_our_own_proxy() -> None:
    """The leak's other half: each proxy kept the previous one alive.

    The order matters, and an earlier version of this test got it wrong and
    passed against the unfixed code. Chaining does not happen while an external
    redirect is active — ``sys.stdout`` is that redirect's target then, and
    routing through it is intended. It happens when the redirect **exits**:
    ``redirect_stdout`` restores whatever it saved, which is the proxy installed
    before it, while ``_installed`` has since moved on to a newer one. The next
    silence sees a proxy that is not the recorded one and wraps it.
    """
    import contextlib
    import io

    with tso.thread_scoped_silence():
        pass
    first = tso._installed["stdout"]
    assert sys.stdout is first

    # Enter and leave an external redirect. On exit sys.stdout is `first` again,
    # but `_installed` now records the proxy built inside the redirect.
    with contextlib.redirect_stdout(io.StringIO()):
        with tso.thread_scoped_silence():
            pass
    assert tso._installed["stdout"] is not first, "no re-install happened; nothing is tested"
    assert sys.stdout is first, "redirect_stdout did not restore the earlier proxy"

    with tso.thread_scoped_silence():
        pass

    hop = tso._installed["stdout"]._passthrough
    assert not isinstance(hop, tso._ThreadRoutingStream), (
        "the new proxy routes through an older proxy, which keeps its sink "
        "reachable and makes every re-install grow the chain by one"
    )
