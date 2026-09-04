#!/usr/bin/env python3
"""Reject whitespace defects that a change range *introduces*.

``git diff --check`` reports trailing whitespace, blank lines at end of file,
space-before-tab, and conflict markers — but only on lines the range **adds**.
That range scope is the whole point: a tree-scoped linter rule for the same
property is blocked behind reformatting most of this repository, while the
range-scoped check can run at full blocking severity today without moving a
single existing file. Pre-existing dirt in a file stays invisible even when the
change edits that very file; the check fires only on newly added bad lines.

CI runs this same script from ``.github/workflows/whitespace-check.yml``, so
the range and the verdict are identical either side of a push. Run it before
you push and the finding costs you a second instead of a CI cycle.

Usage:
    # HEAD against its merge base with the integration branch (auto-detected)
    python scripts/check-whitespace.py

    # Name the integration branch explicitly (what CI passes, using the PR base)
    python scripts/check-whitespace.py --base-ref origin/main

    # Explicit endpoints, skipping merge-base resolution entirely
    python scripts/check-whitespace.py --base <sha> --head <sha>

Exit status:
    0 — the range adds no whitespace defects
    2 — at least one defect. Git's ``path:line: reason`` diagnostics are printed
        verbatim, so the fix needs no local reproduction. This is
        ``git diff --check``'s own exit code, propagated rather than swallowed.
    3 — the range could not be resolved: a shallow clone with no merge base, a
        missing base ref, or not a git checkout at all. Deliberately distinct
        from 0: a comparison that never happened is not a pass, and the most
        likely way this check rots is by silently degrading into one.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Sequence

EXIT_CLEAN = 0
EXIT_DEFECTS = 2
EXIT_UNRESOLVED = 3

#: Tried in order when neither --base nor --base-ref is given.
DEFAULT_BASE_REFS = ("origin/main", "main", "origin/master", "master")


def _git(args: Sequence[str]) -> subprocess.CompletedProcess:
    """Run git, capturing text output under an explicit encoding.

    ``text=True`` alone would decode with the locale codepage, which is not
    UTF-8 on most Windows installs — path names outside ASCII would raise
    UnicodeDecodeError in the reader thread rather than being reported.
    """
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _is_shallow() -> bool:
    result = _git(["rev-parse", "--is-shallow-repository"])
    return result.returncode == 0 and result.stdout.strip() == "true"


def _unresolved(message: str) -> int:
    """Report a range we could not compare, and say so loudly."""
    print("✗ whitespace check could not run.", file=sys.stderr)
    print(f"  {message}", file=sys.stderr)
    if _is_shallow():
        print(
            "  This clone is shallow. In CI, set `fetch-depth: 0` on "
            "actions/checkout; locally, run `git fetch --unshallow`.",
            file=sys.stderr,
        )
    print(
        "  Nothing was compared, so this is NOT a pass — exiting "
        f"{EXIT_UNRESOLVED}.",
        file=sys.stderr,
    )
    return EXIT_UNRESOLVED


def _rev_parse(ref: str) -> str | None:
    result = _git(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"])
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _merge_base(base_ref: str, head: str) -> str | None:
    """merge-base(base_ref, head), or None if the two share no ancestor.

    ``git merge-base`` exits non-zero *and* prints nothing for unrelated
    histories, so both signals are checked — the same defensive shape
    history-check.yml uses for the identical question.
    """
    result = _git(["merge-base", base_ref, head])
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _resolve_base(
    base: str | None, base_ref: str | None, head: str,
) -> tuple[str | None, str]:
    """Return (base_sha, message). base_sha is None when unresolvable."""
    if base is not None:
        sha = _rev_parse(base)
        if sha is None:
            return None, f"--base {base!r} does not name a commit in this clone."
        return sha, f"--base {base}"

    candidates = (base_ref,) if base_ref is not None else DEFAULT_BASE_REFS
    tried = []
    for candidate in candidates:
        tried.append(candidate)
        if _rev_parse(candidate) is None:
            continue
        sha = _merge_base(candidate, head)
        if sha is None:
            return None, (
                f"{candidate!r} and {head!r} share no common ancestor, so "
                "there is no range to check."
            )
        return sha, f"merge-base({candidate}, {head})"

    return None, (
        "no integration branch found to compare against (tried: "
        + ", ".join(repr(t) for t in tried)
        + "). Pass --base-ref or --base explicitly."
    )


def _changed_file_count(base: str, head: str) -> int:
    result = _git(["diff", "--name-only", base, head])
    if result.returncode != 0:
        return -1
    return len([line for line in result.stdout.splitlines() if line.strip()])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if the change range introduces whitespace defects "
            "(git diff --check)."
        ),
    )
    parser.add_argument(
        "--base-ref",
        default=None,
        help=(
            "Integration branch to take the merge base against. "
            f"Default: first of {', '.join(DEFAULT_BASE_REFS)} that exists."
        ),
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Explicit base commit. Skips merge-base resolution entirely.",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="Head of the range. Default: HEAD.",
    )
    args = parser.parse_args(argv)

    if _git(["rev-parse", "--git-dir"]).returncode != 0:
        return _unresolved("not inside a git checkout.")

    head_sha = _rev_parse(args.head)
    if head_sha is None:
        return _unresolved(f"--head {args.head!r} does not name a commit.")

    base_sha, how = _resolve_base(args.base, args.base_ref, args.head)
    if base_sha is None:
        return _unresolved(how)

    # The range, not the working tree. Bare `git diff --check` inspects
    # unstaged changes, which are empty on any CI checkout — it would report
    # clean on a PR that adds a hundred bad lines.
    result = _git(["diff", "--check", base_sha, args.head])

    if result.returncode == EXIT_CLEAN:
        count = _changed_file_count(base_sha, args.head)
        scope = f"{count} changed file(s)" if count >= 0 else "the range"
        print(
            f"✓ No whitespace defects added in {base_sha[:10]}..{head_sha[:10]} "
            f"({how}, {scope} compared)."
        )
        return EXIT_CLEAN

    if result.returncode != EXIT_DEFECTS:
        detail = (result.stderr or result.stdout).strip() or "(no output)"
        return _unresolved(
            f"`git diff --check` exited {result.returncode}: {detail}"
        )

    print(result.stdout, end="")
    if result.stderr.strip():
        print(result.stderr, end="", file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "✗ whitespace check failed — this is not a test failure.",
        file=sys.stderr,
    )
    print(
        f"  The lines above were ADDED in {base_sha[:10]}..{head_sha[:10]} "
        f"({how}) and carry whitespace git rejects.",
        file=sys.stderr,
    )
    print(
        "  Fix the named lines (strip trailing spaces, drop the blank line at "
        "end of file), then re-run:",
        file=sys.stderr,
    )
    print("      python scripts/check-whitespace.py", file=sys.stderr)
    return EXIT_DEFECTS


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
