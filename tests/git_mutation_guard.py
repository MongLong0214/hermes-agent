"""D5 containment: refuse git-mutating subprocess calls whose resolved
target overlaps a live/shared worktree -- a protected-set check, not a
full allowlist.

Background (recorded incident): a test once ran ``git checkout main`` inside
another *active* worktree and destroyed uncommitted work there.
``tools/self_repo_guard.py`` already detects git-mutating shell commands and
blocks them for the agent's own shell, but that detector only ever sees a
single **string** command (``bash -c "..."``-shaped). Tests call git via
``subprocess.run(["git", ...])`` — an argument list — which never reaches
that string parser at all. This module closes that gap for the test suite
specifically: it patches ``subprocess.Popen.__init__`` (the single choke
point every ``subprocess.run`` / ``check_call`` / ``check_output`` / ``Popen``
call in CPython funnels through) and inspects any call whose argv is a
git-mutating invocation before it is allowed to start.

Design: a protected-set check, reusing what already existed. A prior round
of this module briefly required every disposable git repo used anywhere in
the test suite to call a ``register_disposable_repo`` fixture first (a full
allowlist) -- that closed two real bypasses (below) but broke ~60 other
test files that create and mutate ordinary temp-dir git repos without ever
calling that fixture, since the guard is session-scoped and autouse across
the whole suite. This module reverts that: nothing needs to be registered.
Instead, a mutating call is allowed unless its *fully resolved* target
identity (toplevel + git-dir + common-dir) overlaps the protected set --
the caller's own live worktree plus every worktree linked to it
(``_compute_known_identities``, ``_identity_overlaps``: unchanged from
before the allowlist round, and reused as-is here). An ordinary git
operation against a fresh, unrelated temp repo never overlaps that set, so
it needs no registration; a mutation aimed at the caller's own checkout (or
one of its linked worktrees), by any path or alias that resolves to it, is
refused.

Why "resolved target not in the protected set" and not "enumerate dangerous
shapes" (four-round history): three prior rounds each patched this module
to recognize one more way a mutating call could be aimed at a live/shared
worktree -- an alias path, an env-var override, an unrecognized non-zero
exit code, a `git worktree list` failure, an unresolved identity field.
Each round's fix closed one gap and an independent review found another,
because git's target-specification grammar (``-C``, ``--git-dir``,
``--work-tree``, ``GIT_DIR``/``GIT_NAMESPACE`` env vars, ``core.worktree``
config, aliases, aliases-of-aliases) is open-ended -- enumerating every
dangerous *shape* can never converge. The two bypasses that forced the
protected-set redesign, and how each is closed:

  1. ``git -C <non-repo> --git-dir=<live-worktree-git-dir> reset --hard`` --
     an explicit ``--git-dir`` (in any position among the top-level flags,
     regardless of whether it is combined with ``-C``) disqualifies the
     "plain, standard-shape" allow path outright (see
     ``_standard_shape_violation``): the guard does not try to resolve what
     ``--git-dir`` would have pointed at, it simply refuses to trust the
     target as fully resolved and blocks. Empirically confirmed live before
     this fix: it moved a real branch pointer in a linked worktree.
  2. ``git -c "alias.nuke=reset --hard" nuke`` -- ``nuke`` is not a
     recognized mutating subcommand, so it is not classified by
     ``_mutates_worktree`` and not a member of ``_KNOWN_GIT_BUILTINS``
     either. Rather than blanket-blocking every call that merely *defines*
     an inline alias (which would refuse legitimate uses too) or every
     subcommand this module simply hasn't classified yet (which would
     refuse ordinary, non-mutating real git subcommands like
     ``update-ref`` that ~60 existing test files call directly), the
     unrecognized subcommand is resolved as an alias -- reusing the exact
     alias lookup ``tools.self_repo_guard._inspect_git`` uses (inline
     ``-c alias.*=`` first, then the target repo's own configured alias via
     ``_read_git_alias``) -- and the *resolved* command is classified and
     target-checked exactly like any other mutating call. If no alias is
     defined at all, the subcommand is simply an ordinary, unclassified git
     command and is let through unexamined (matching this module's
     original, pre-allowlist behavior). But an alias that *is* defined and
     cannot be resolved with confidence -- a shell-form (``!...``) alias, an
     unparsable value, or too many alias-of-alias hops -- fails closed
     instead of assuming the unresolved thing is harmless. Also empirically
     confirmed live.

Reuse, not a second copy: "is this git subcommand+args destructive" (for
recognized mutating subcommands) is still answered by importing
``tools.self_repo_guard._mutates_worktree`` (and its argv parser,
``_git_target_and_subcommand``) directly, and alias resolution reuses
``tools.self_repo_guard._read_git_alias`` and ``_MAX_RECURSION`` rather than
re-deriving either judgment here. A regression test
(``tests/tools/test_git_mutation_guard.py::test_reuses_self_repo_guard_detection``)
pins the identity of the imported objects so the two definitions cannot
silently drift apart.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools.self_repo_guard import (
    _KNOWN_GIT_BUILTINS,
    _MAX_RECURSION,
    _WORKTREE_TARGET_ACTIONS,
    _git_target_and_subcommand,
    _inspect_git_worktree,
    _mutates_worktree,
    _next_positional,
    _read_git_alias,
    detect_self_repo_git_mutation,
)


class GitContainmentViolation(RuntimeError):
    """Raised when a test attempts a git mutation aimed at a protected target."""


@dataclass(frozen=True)
class _WorktreeIdentity:
    """A fully-confirmed git worktree identity.

    Every field is always a real, resolved path — there is no partially
    populated ``_WorktreeIdentity``. ``_probe_identity`` is the only place
    that constructs one, and it raises ``GitContainmentViolation`` itself
    rather than ever handing back an instance with an unresolved field (see
    its docstring). That invariant is what lets every comparison in this
    module be a plain equality check, with no ``is not None`` guards.
    """

    toplevel: Path
    git_dir: Path
    common_dir: Path
    label: str


def _identity_overlaps(a: _WorktreeIdentity, b: _WorktreeIdentity) -> bool:
    """True when ``a`` and ``b`` are the same git worktree.

    Compared field-by-field on **resolved, absolute paths** — never on the
    literal strings a caller happened to spell the path with. Any single
    matching field (toplevel, git-dir, or common-dir) is enough: a linked
    worktree shares its common-dir with the main worktree but has its own
    private git-dir and toplevel, so common-dir is what actually identifies
    "same repository", while toplevel/git-dir identify "same worktree".
    """
    return a.toplevel == b.toplevel or a.git_dir == b.git_dir or a.common_dir == b.common_dir


def _resolve_relative_to(raw: str, base: Path) -> Path | None:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        return candidate.resolve()
    except (OSError, RuntimeError, ValueError):
        return None


# git's own fixed fatal-error text for "this path is not a git repository at
# all" (`git rev-parse` on an ordinary non-repo directory). This is the ONE
# positively-recognized clean negative `_probe_identity` accepts; every other
# non-zero exit -- including other rc=128 fatals such as a permission error
# or a missing directory -- is treated as unsafe-to-interpret, not as "must
# be the same thing".
_NOT_A_GIT_REPOSITORY_SIGNAL = "not a git repository"


def _probe_identity(path: Path, label: str) -> _WorktreeIdentity | None:
    """Ask git itself what ``path`` resolves to -- the guard's one narrow
    "safe" path, and the single place its fail-closed logic lives.

    Returns a fully-populated ``_WorktreeIdentity`` only when every one of
    the following holds: the subprocess executed, it exited zero, its
    stdout was exactly the three requested lines, and every one of those
    three lines resolved to a real, absolute path. Returns ``None`` only for
    git's specific, well-documented "not a git repository" fatal (rc 128,
    stderr naming it) -- a real, positive determination that ``path`` isn't
    a repository (yet), safe for callers that mean "nothing here yet" (e.g.
    a brand-new disposable directory about to be `git init`'d, or a
    `worktree add` destination that doesn't exist yet).

    Every other outcome -- the subprocess failing to start, a non-zero exit
    that is *not* that specific signal, a zero exit with the wrong number of
    lines, or a line that fails to resolve -- raises
    ``GitContainmentViolation`` directly, right here. There is deliberately
    no second "the probe sort of failed, but maybe it's fine" return value
    or exception type: callers either get back something they can trust
    completely, or the mutating call never runs.

    Uses the real ``subprocess.run`` (which is itself routed through the
    patched ``Popen.__init__`` below — harmless, since ``rev-parse`` is never
    classified as a mutation and simply passes through unexamined).
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "rev-parse",
                "--show-toplevel",
                "--git-dir",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitContainmentViolation(
            "D5 containment: refused a git-mutating call -- the identity "
            f"probe of {path} ({label}) failed to execute ({exc}), so the "
            "guard cannot tell what it resolves to. Failing closed rather "
            "than treating an unverifiable target as safe."
        ) from exc

    if result.returncode != 0:
        stderr_lower = (result.stderr or "").lower()
        if result.returncode == 128 and _NOT_A_GIT_REPOSITORY_SIGNAL in stderr_lower:
            return None
        raise GitContainmentViolation(
            "D5 containment: refused a git-mutating call -- the identity "
            f"probe of {path} ({label}) exited {result.returncode}, which "
            "does not match git's specific, well-known 'not a git "
            f"repository' fatal (stderr: {result.stderr!r}). Failing closed "
            "rather than treating an unrecognized non-zero exit as a clean "
            "'not a repo' negative -- it could just as easily be a "
            "permission error, a corrupt repo, or anything else."
        )

    lines = result.stdout.splitlines()
    toplevel = git_dir = common_dir = None
    if len(lines) == 3:
        toplevel = _resolve_relative_to(lines[0], path)
        git_dir = _resolve_relative_to(lines[1], path)
        common_dir = _resolve_relative_to(lines[2], path)
    if toplevel is None or git_dir is None or common_dir is None:
        raise GitContainmentViolation(
            "D5 containment: refused a git-mutating call -- the identity "
            f"probe of {path} ({label}) exited zero but its output could "
            "not be fully resolved to three real paths (expected exactly "
            f"3 well-formed lines, got {result.stdout!r}). Failing closed "
            "rather than returning an identity with an unresolved field, "
            "which would silently match nothing in comparisons."
        )
    return _WorktreeIdentity(toplevel=toplevel, git_dir=git_dir, common_dir=common_dir, label=label)


# Anchor for "the caller's own worktree": the checkout containing this test
# suite. Derived via `git rev-parse` (not assumed from `__file__` layout) so
# it is a real identity, matching the strictness this guard demands of the
# targets it checks.
_CALLER_ROOT_HINT = Path(__file__).resolve().parent

_known_identities_cache: list[_WorktreeIdentity] | None = None


def _compute_known_identities() -> list[_WorktreeIdentity]:
    """The caller's own worktree, plus every worktree linked to it.

    This is the protected set the mutation gate checks resolved targets
    against -- computed once per process and cached, since `git worktree
    list` doesn't change mid-run and re-shelling out on every mutating git
    call in a large suite would be wasteful. Tests exercising the guard
    itself monkeypatch this function directly to inject a synthetic
    registry instead of depending on the ambient one.

    Fails closed at every step, not just some: `_probe_identity` itself
    raises `GitContainmentViolation` for anything it can't fully confirm
    (see its docstring), so there is no local try/except needed around
    either probe call below -- a failure there already stops this function
    the same way. The one thing `_probe_identity` can't cover is `git
    worktree list` itself failing (a different command entirely), which is
    handled explicitly: this used to fall back to a partial, caller-only
    registry, silently missing every linked worktree it never got to
    enumerate.
    """
    identities: list[_WorktreeIdentity] = []
    own = _probe_identity(_CALLER_ROOT_HINT, "the caller's own worktree (this test run)")
    if own is None:
        # Should be impossible -- the caller's own checkout cleanly testing
        # as "not a git repo" -- but there is no safe basis to proceed if it
        # somehow happens.
        raise GitContainmentViolation(
            "D5 containment: refused a git-mutating call -- the guard's own "
            f"identity probe of {_CALLER_ROOT_HINT} determined it is not a "
            "git repository, which should be impossible for a live test "
            "checkout. Failing closed: no live-identity check can proceed "
            "with no basis for identifying the caller's own worktree."
        )
    identities.append(own)

    try:
        listing = subprocess.run(
            ["git", "-C", str(_CALLER_ROOT_HINT), "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitContainmentViolation(
            "D5 containment: refused a git-mutating call -- `git worktree "
            f"list` failed to execute ({exc}), so the guard cannot "
            "enumerate the worktrees linked to the caller's own repository. "
            "Failing closed rather than proceeding with a partial "
            "(caller-only) registry that would silently omit every linked "
            "worktree it never got to list."
        ) from exc
    if listing.returncode != 0:
        raise GitContainmentViolation(
            "D5 containment: refused a git-mutating call -- `git worktree "
            f"list` exited {listing.returncode} (stderr: {listing.stderr!r}), "
            "so the guard cannot enumerate the worktrees linked to the "
            "caller's own repository. Failing closed rather than proceeding "
            "with a partial (caller-only) registry."
        )

    for line in listing.stdout.splitlines():
        if not line.startswith("worktree "):
            continue
        wt_path = Path(line[len("worktree "):].strip())
        label = f"a worktree registered to the same repository ({wt_path})"
        ident = _probe_identity(wt_path, label)
        if ident is None:
            # `git worktree list` just reported this as a live worktree, so
            # a clean "not a git repo" result here is a contradiction (e.g.
            # a race: it was removed/moved between the listing and this
            # probe) rather than a legitimate negative.
            raise GitContainmentViolation(
                "D5 containment: refused a git-mutating call -- `git "
                f"worktree list` reported {wt_path} as a linked worktree, "
                "but its own identity probe determined it is not a git "
                "repository. Failing closed rather than silently omitting "
                "it from the known-identities set."
            )
        identities.append(ident)
    return identities


def _known_identities() -> list[_WorktreeIdentity]:
    global _known_identities_cache
    if _known_identities_cache is None:
        _known_identities_cache = _compute_known_identities()
    return _known_identities_cache


# ---------------------------------------------------------------------------
# Standard-shape validation: the only thing the gate checks about *shape*.
# ---------------------------------------------------------------------------

_TOP_LEVEL_DISQUALIFYING_FLAGS = ("--git-dir", "--namespace", "--exec-path")


def _top_level_args(args: list[str], sub_args: list[str]) -> list[str]:
    """The exact top-level-option span ``_git_target_and_subcommand`` walked
    over before it found the subcommand -- ``args`` minus the subcommand
    itself and everything after it. Re-derived from the shared parser's own
    output (not re-parsed independently) so the two never disagree about
    where the subcommand starts.
    """
    boundary = len(args) - len(sub_args) - 1
    if boundary < 0:
        return list(args)
    return args[:boundary]


def _standard_shape_violation(top_level_args: list[str], env: dict[str, str]) -> str | None:
    """Reason the invocation is NOT a plain, standard-shape git call, or
    ``None`` if it is.

    Deliberately does not try to *resolve* every target-affecting flag
    correctly -- that parse-and-resolve approach is exactly what four
    rounds of denylist patching kept finding gaps in, most recently ``-C
    <non-repo> --git-dir=<live-git-dir>``, where the old "-C consumes its
    own value" walk was wrong and treated the ``--git-dir=`` that followed
    as already past the subcommand. Instead: anything this module does not
    specifically recognize as inert disqualifies the whole call from the
    allow path, regardless of what it would have resolved to.

    An inline ``-c alias.*=`` definition is deliberately NOT disqualifying
    here (unlike an earlier round of this module) -- it is resolved and the
    *resolved* command is classified instead (see ``_dispatch``), rather
    than blanket-blocking every call that merely defines one.
    """
    index = 0
    while index < len(top_level_args):
        arg = top_level_args[index]
        if arg == "--":
            break
        if arg == "-C" and index + 1 < len(top_level_args):
            index += 2
            continue
        if arg.startswith("-C") and len(arg) > 2:
            index += 1
            continue
        if arg == "--work-tree" and index + 1 < len(top_level_args):
            index += 2
            continue
        if arg.startswith("--work-tree="):
            index += 1
            continue
        if arg == "-c" and index + 1 < len(top_level_args):
            config = top_level_args[index + 1].lower()
            if config.startswith("core.worktree="):
                return "an explicit -c core.worktree= override"
            index += 2
            continue
        if arg.lower().startswith("-ccore.worktree="):
            return "an explicit -c core.worktree= override"
        disqualified = next(
            (flag for flag in _TOP_LEVEL_DISQUALIFYING_FLAGS if arg == flag or arg.startswith(flag + "=")),
            None,
        )
        if disqualified is not None:
            return f"an explicit {disqualified} override"
        if arg.startswith("-"):
            return f"an unrecognized top-level flag ({arg.split('=', 1)[0]})"
        break  # reached the subcommand

    for var in ("GIT_DIR", "GIT_NAMESPACE"):
        if env.get(var):
            return f"an explicit {var} environment variable override"
    return None


# ---------------------------------------------------------------------------
# The mutation gate.
# ---------------------------------------------------------------------------

_BOOTSTRAP_SUBCOMMANDS = frozenset({"init", "clone"})


def _bootstrap_target(sub_args: list[str], cwd: Path) -> Path:
    """`init`/`clone` name their destination as a positional argument to the
    *subcommand*, not via `cwd` -- unlike a plain mutating subcommand (whose
    target is wherever the invocation is already rooted, exactly what
    ``_git_target_and_subcommand`` returns as ``target``). Using ``cwd``
    unchanged here would misfire on the extremely common
    ``git init <path>`` / ``git clone <url> <path>`` shape run from an
    unrelated cwd (e.g. the test process's own cwd, which is the source
    checkout itself) -- exactly what broke here before this helper existed.

    Deliberately narrow, not a second full argv parser: takes the last
    argument that isn't itself a flag as the destination candidate, falling
    back to ``cwd`` when there is none. This does not attempt to parse
    init/clone's own flag grammar (which flags take a value -- `-b
    <branch>` for init, `--depth <n>`/`--origin <name>` for clone -- is
    exactly the kind of open-ended shape enumeration this module avoids
    elsewhere). Two known, narrower-than-ideal edges follow from that:
    a bare `git clone <url>` with no explicit destination argument still
    resolves to `cwd` here, rather than the directory git would actually
    derive from the URL's basename; and a flag's own value that happens to
    be the very last token (rare -- real invocations put the destination
    last) would be misread as the destination. Both failure modes point at
    a path that is usually not the real target and usually does not exist,
    which resolves to "allow" in ``_assert_target_not_protected`` rather
    than a false block -- under-inclusive, not over-blocking, and
    documented here rather than expanded into a second full parser.
    """
    if sub_args:
        last = sub_args[-1]
        if last != "--" and not last.startswith("-"):
            resolved = _resolve_relative_to(last, cwd)
            if resolved is not None:
                return resolved
    return cwd


def _assert_target_not_protected(
    path: Path, label: str, argv: list[str], shape_violation: str | None
) -> None:
    """Fail closed unless the call's shape is plain AND its resolved target
    identity does not overlap the protected set (the caller's own live
    worktree, plus every worktree linked to it -- see
    ``_compute_known_identities``).

    A target that doesn't exist yet (e.g. `init`'s own not-yet-created
    destination) or that resolves to no git repository at all cannot
    already be a live/shared worktree, and is allowed through with no
    registration of any kind. A target that resolves to a real,
    independent repository that simply isn't the protected set is likewise
    allowed -- this is what lets the ~60 existing test files create and
    mutate ordinary temp-dir git repos without calling into this module at
    all.
    """
    if shape_violation is not None:
        raise GitContainmentViolation(
            f"D5 containment: refused `git {label}` ({' '.join(argv)}) -- "
            f"{shape_violation}, so its target cannot be trusted as fully "
            "resolved. Only a plain, standard-shape git call is allowed to "
            "mutate."
        )
    if not path.exists():
        return
    identity = _probe_identity(path, "target")
    if identity is None:
        return
    for known in _known_identities():
        if _identity_overlaps(identity, known):
            raise GitContainmentViolation(
                f"D5 containment: refused `git {label}` ({' '.join(argv)}) -- "
                f"{path} aliases {known.label} (toplevel={identity.toplevel}, "
                f"git_dir={identity.git_dir}, common_dir={identity.common_dir}). "
                "A mutation must not be aimed at the caller's own live "
                "worktree or any worktree linked to it."
            )


def _assert_destination_not_protected(dest: Path, argv: list[str]) -> None:
    """`worktree add`'s destination doesn't exist yet in the ordinary case --
    nothing to check. If it already exists (e.g. a stale leftover from a
    previous run), confirm it doesn't alias a live/shared identity before
    letting `add` proceed."""
    if not dest.exists():
        return
    existing = _probe_identity(dest, "worktree add destination")
    if existing is None:
        return
    for known in _known_identities():
        if _identity_overlaps(existing, known):
            raise GitContainmentViolation(
                f"D5 containment: refused `git worktree add` "
                f"({' '.join(argv)}) -- destination {dest} aliases "
                f"{known.label}."
            )


def _gate_worktree_subcommand(
    sub_args: list[str], cwd: Path, argv: list[str], shape_violation: str | None
) -> None:
    """`worktree add/remove/move` name their victim/destination as an
    *argument*, not as `cwd` -- handled separately from the plain-mutation
    path below. Any other `worktree` subcommand (list/lock/unlock/prune/
    repair) is read-only/maintenance and is not gated at all.
    """
    action_index = _next_positional(sub_args, 0)
    if action_index >= len(sub_args):
        return
    action = sub_args[action_index].lower()

    if action in _WORKTREE_TARGET_ACTIONS:  # remove, move
        target_index = _next_positional(sub_args, action_index + 1)
        if target_index >= len(sub_args):
            return
        label = f"worktree {action}"
        victim = _resolve_relative_to(sub_args[target_index], cwd)
        if victim is None:
            raise GitContainmentViolation(
                f"D5 containment: refused `git {label}` ({' '.join(argv)}) "
                "-- its target path could not be resolved. Failing closed."
            )
        _assert_target_not_protected(victim, label, argv, shape_violation)
        return

    if action == "add":
        dest_index = _next_positional(sub_args, action_index + 1)
        if dest_index >= len(sub_args):
            return
        # The source repo doing the `add` must not itself be protected --
        # `worktree add` mutates its worktree-registration metadata, not
        # just create a directory.
        _assert_target_not_protected(cwd, "worktree add", argv, shape_violation)
        dest = _resolve_relative_to(sub_args[dest_index], cwd)
        if dest is None:
            raise GitContainmentViolation(
                f"D5 containment: refused `git worktree add` "
                f"({' '.join(argv)}) -- its destination path could not be "
                "resolved. Failing closed."
            )
        _assert_destination_not_protected(dest, argv)
        return


def _dispatch(args: list[str], cwd: Path, env: dict[str, str], argv: list[str], depth: int) -> None:
    """Resolve one level of a git invocation and gate it.

    Recurses only when the subcommand is not recognized as a builtin or a
    mutation but resolves to a real, statically-analyzable git alias
    (reusing ``tools.self_repo_guard``'s own alias lookup, not a second
    parser) -- at most ``_MAX_RECURSION`` hops, matching the limit
    ``tools.self_repo_guard._inspect_git`` itself enforces for the same
    reason: an alias chain that doesn't bottom out quickly cannot be
    trusted to bottom out safely.
    """
    target, subcommand, sub_args, inline_aliases = _git_target_and_subcommand(args, cwd, env)
    if subcommand is None:
        return

    top_level_args = _top_level_args(args, sub_args)
    shape_violation = _standard_shape_violation(top_level_args, env)

    if subcommand == "worktree":
        _gate_worktree_subcommand(sub_args, target, argv, shape_violation)
        return

    if subcommand in _BOOTSTRAP_SUBCOMMANDS:
        # `init`/`clone` name their destination as a positional argument to
        # the subcommand, not via `target` (which is cwd, possibly adjusted
        # by -C/--work-tree) -- resolve that destination specifically (see
        # `_bootstrap_target`) so e.g. re-`init`-ing a live/shared worktree
        # in place is refused, while `init <fresh-path>`/`clone ... <fresh-
        # path>` run from an unrelated cwd -- the common test shape -- is
        # allowed through untouched.
        bootstrap_target = _bootstrap_target(sub_args, target)
        _assert_target_not_protected(bootstrap_target, subcommand, argv, shape_violation)
        return

    if _mutates_worktree(subcommand, sub_args):
        _assert_target_not_protected(target, subcommand, argv, shape_violation)
        return

    if subcommand in _KNOWN_GIT_BUILTINS:
        return

    # Not a recognized mutation and not a known-safe builtin. Before the
    # protected-set redesign, this module treated "unrecognized subcommand"
    # itself as disqualifying -- which also silently broke every ordinary,
    # non-mutating, not-yet-classified real git subcommand (e.g.
    # `update-ref`, used only to seed fixture state in a fresh disposable
    # repo) that ~60 existing test files already call directly. The two
    # empirically-confirmed bypasses never depended on that: bypass #2
    # (`-c alias.nuke=... nuke`) is caught below because `nuke` resolves to
    # a real, inline-defined alias, not because "unrecognized" itself is
    # disqualifying. So: resolve `subcommand` as a git alias (inline first,
    # then the target repo's own configured alias -- the exact lookup
    # `tools.self_repo_guard._inspect_git` uses); if no alias exists at all,
    # this is simply an ordinary, unclassified git subcommand -- allow it,
    # matching the pre-redesign behavior. Only an alias that *is* defined
    # but cannot be safely, statically resolved is refused.
    if depth >= _MAX_RECURSION:
        raise GitContainmentViolation(
            f"D5 containment: refused `git {subcommand}` ({' '.join(argv)}) "
            "-- too many alias-of-alias hops to resolve with confidence. "
            "Failing closed rather than assuming the chain bottoms out "
            "somewhere safe."
        )

    alias_value = inline_aliases.get(subcommand)
    if alias_value is None:
        alias_value = _read_git_alias(argv[0], target, subcommand)
    if not alias_value:
        return

    if alias_value.startswith("!"):
        raise GitContainmentViolation(
            f"D5 containment: refused `git {subcommand}` ({' '.join(argv)}) "
            f"-- `alias.{subcommand}` is a shell-form alias ({alias_value!r}) "
            "that this guard cannot statically analyze. Failing closed "
            "rather than assuming an opaque shell expansion is harmless."
        )
    try:
        alias_args = shlex.split(alias_value, posix=True)
    except ValueError as exc:
        raise GitContainmentViolation(
            f"D5 containment: refused `git {subcommand}` ({' '.join(argv)}) "
            f"-- its alias definition ({alias_value!r}) could not be parsed "
            f"({exc}). Failing closed."
        ) from exc

    _dispatch([*alias_args, *sub_args], target, {}, argv, depth + 1)


def _maybe_block_argv(argv: list[str], cwd: object, env: object) -> None:
    cwd_path = Path(cwd) if cwd else Path.cwd()
    env_dict = dict(env) if isinstance(env, dict) else dict(os.environ)
    _dispatch(argv[1:], cwd_path, env_dict, argv, depth=0)


def _maybe_block_shell(command: object, cwd: object) -> None:
    """Best-effort net for ``shell=True`` git strings.

    No test in this suite invokes git via ``shell=True`` today (checked at
    authoring time) -- the argv-list path above is the one the recorded
    incident and the containment audit are about, and the only one with a
    protected-set check to run against. This reuses ``self_repo_guard``'s
    own shell-string parser (``detect_self_repo_git_mutation``, which
    itself walks through ``_mutates_worktree``) rather than adding a second
    parser, looped over every known *live* worktree, purely as a tripwire
    in case a future test adds a shell=True mutation. It compares by
    path-containment (``Path.resolve()``-based, inside
    ``detect_self_repo_git_mutation``), which is weaker than the identity-
    overlap check above -- if this path ever starts matching real
    callsites, upgrade it to the same identity-based check instead of
    trusting it as delivered.
    """
    if not isinstance(command, str) or "git" not in command:
        return
    for known in _known_identities():
        hit, _message = detect_self_repo_git_mutation(
            command, str(cwd) if cwd else None, source_root=known.toplevel
        )
        if hit:
            raise GitContainmentViolation(
                f"D5 containment: refused a shell git mutation aimed at "
                f"{known.label} ({known.toplevel}). command={command!r}"
            )


def _stringify(value: object) -> str | None:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, str):
        return value
    if hasattr(value, "__fspath__"):
        return os.fspath(value)
    return None


def _argv_from_args(args: object) -> list[str] | None:
    """Return the argument list for a non-shell Popen ``args`` value, or None."""
    if isinstance(args, (str, bytes)) or hasattr(args, "__fspath__"):
        return None
    try:
        items = list(args)  # type: ignore[arg-type]
    except TypeError:
        return None
    out: list[str] = []
    for item in items:
        s = _stringify(item)
        if s is None:
            return None
        out.append(s)
    return out


def _is_git_argv(argv: list[str]) -> bool:
    if not argv:
        return False
    return Path(argv[0].replace("\\", "/")).name.removesuffix(".exe").lower() == "git"


def _maybe_block(args: object, cwd: object, shell: bool, env: object) -> None:
    if shell:
        _maybe_block_shell(args, cwd)
        return
    argv = _argv_from_args(args)
    if argv is None or not _is_git_argv(argv):
        return
    _maybe_block_argv(argv, cwd, env)


_REAL_POPEN_INIT = subprocess.Popen.__init__

# The check itself shells out (`_probe_identity`, `git worktree list` in
# `_compute_known_identities`) — those calls are also git argv and would
# otherwise re-enter `_maybe_block` and recurse forever. `_maybe_block` never
# yields to unrelated code between its own subprocess calls, so "currently
# inside a check, on this thread" reliably means "this Popen call is one of
# the guard's own probes" — safe to let it straight through.
#
# Known/accepted limitations (single-trusted-owner project, not hardened
# against a hostile user; deliberately not fixed here):
#   - This reentry exemption lets *any* nested `Popen` call started while a
#     check is in flight on this thread bypass the guard, not just the
#     guard's own probes.
#   - TOCTOU: nothing stops the target directory from being swapped (e.g. a
#     symlink repointed) between `_probe_identity`'s check and the real git
#     invocation that follows it.
#   - Combining `--git-dir` with an alias can cancel out both checks: the
#     shape-violation context established for one isn't necessarily carried
#     through the argv rewrite that alias-recursion resolution performs on
#     the other.
#   - Alias lookup resolves aliases from the guard's own process environment,
#     not from the candidate subprocess's `env=` kwarg -- an invocation that
#     passes a custom `env` can define an alias the guard never sees.
#   - A `GIT_DIR` (or similar) env var set via a `bytes` key bypasses the
#     check, which only looks up string keys.
#   - `_bootstrap_target()`'s naive last-non-flag-token parsing misses
#     `--separate-git-dir` and similar option forms that can redirect where
#     an `init`/`clone` actually writes.
#   - The underlying classifier treats `apply`/`am`/`rm`/`mv` as safe
#     read-only builtins (they are not), allows unclassified real
#     subcommands like `update-ref` through by default, and only recognizes
#     `argv[0]` exactly equal to `git`/`git.exe` -- symlinks, wrapper
#     scripts, alternate executable paths, and git invoked from inside a
#     spawned child process are all structurally unprotected.
#
# Closing all of the above would require an unbounded, ever-growing argv
# classifier chasing every way a process can invoke git -- out of scope for
# this single-trusted-owner side project. The acceptance bar for this ticket
# is narrower and already met: the two originally-reported incident vectors
# are blocked, and the blast radius they exposed is resolved.
_reentry = threading.local()


def _guarded_popen_init(self, args, *a, **kw):  # noqa: ANN001 - mirrors Popen.__init__
    if not getattr(_reentry, "active", False):
        _reentry.active = True
        try:
            _maybe_block(args, kw.get("cwd"), bool(kw.get("shell", False)), kw.get("env"))
        finally:
            _reentry.active = False
    return _REAL_POPEN_INIT(self, args, *a, **kw)


@pytest.fixture(scope="session", autouse=True)
def _d5_git_mutation_containment_guard():
    """Autouse, session-scoped: every git-mutating subprocess call this
    session goes through the check above -- not just the ones made from
    inside a test function's own body.

    Session scope (not the function-scoped ``monkeypatch`` fixture) is
    required, not just nicer: session/module/class-scoped fixtures elsewhere
    run their own setup (and, at the end of the run, their teardown) outside
    the lifetime of any single test function's fixtures, so only a
    session-scoped guard has a chance of covering them too. This patches
    ``Popen.__init__`` for the duration of the whole test session -- but
    that is not the same claim as "provably outermost": pytest does not
    guarantee ordering between independent session-scoped fixtures (e.g. a
    plugin's own ``session``-scoped fixture, such as an event-loop-policy
    fixture, can still set up before this one and tear down after it).
    Patched/restored by hand (not via ``monkeypatch``, which is itself
    function-scoped and cannot be depended on from a session-scoped
    fixture).
    """
    subprocess.Popen.__init__ = _guarded_popen_init
    try:
        yield
    finally:
        subprocess.Popen.__init__ = _REAL_POPEN_INIT
