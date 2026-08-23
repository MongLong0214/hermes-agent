"""Run a pinned property against a MUTATED copy of the tree, and require it to die.

A pin that passes is not a pin. It is a pin only when there is a change to the
source that makes it fail — otherwise it may be asserting something the code
cannot do anyway, and it will keep passing after the guard is deleted. That is
not hypothetical here: the first foreign-root pin written against this harness
survived its own mutation, because it was refused by a holder comparison one
line below the root comparison it claimed to be testing.

Three properties, each because the obvious version is wrong:

* Mutations are keyed by an exact source SUBSTRING that must match EXACTLY ONCE,
  never by a line number. A line-number anchor goes stale the moment anything
  above it grows, and it goes stale silently.
* Every row extracts its OWN tree and runs with ``PYTHONDONTWRITEBYTECODE``. A
  shared directory lets the second row import the first row's compiled module
  and report a removed guard as covered.
* Every row runs clean / mutated / restored. Clean proves the extract works at
  all; restored proves the failure came from the mutation and not from a
  fixture that was broken from the start.

Not a test module (no ``test_`` prefix), so pytest imports it only through the
files that use it.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap
from dataclasses import dataclass

import pytest

from tests.state.test_turn_lease_generation_trigger import (
    BASE_TREE_PATHSPEC,
    _git_dir,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Enough of the package to open a store, plus the two test paths every pin
#: module imports at module scope, plus this harness. Derived from the pathspec
#: the base-binary fixture already maintains rather than a second copy of it, so
#: a module that fixture starts needing arrives here too.
BASE_EXTRACT_PATHSPEC = tuple(BASE_TREE_PATHSPEC) + (
    "tests/__init__.py",
    "tests/state/test_turn_lease_generation_trigger.py",
    "tests/state/lease_mutation_harness.py",
)


@dataclass(frozen=True)
class Mutation:
    """One guard removal, and the pin that must not survive it."""

    pin: str
    module: str
    find: str
    replace: str
    why: str


class OwnerRefused(str):
    """A refusal of the OWNER's own call, carried as a value rather than raised.

    For the "and the owner can still do X" leg every refusal pin needs. Left
    bare, such a leg dies under its mutation by an escaping
    ``SessionTurnLeaseLostError``: the property IS violated, but the pin never
    states what it expected, so the failure reads as a broken fixture rather
    than as "the fence refused the conversation's own holder".

    That distinction is the whole of :func:`assert_mutation_kills_the_pin`'s
    discrimination check. A row whose pin dies without its own assertion firing
    proves the mutation runs; it does not prove the pin can see what the guard
    protects, and those are the rows that go on passing after the guard is
    deleted.
    """


def owner_call(fn):
    """Run *fn*, returning its result or the refusal as :class:`OwnerRefused`."""
    from hermes_state import SessionTurnLeaseLostError

    try:
        return fn()
    except SessionTurnLeaseLostError as exc:
        return OwnerRefused(f"REFUSED THE OWNER: {exc}")


def assert_the_owner_was_not_refused(outcome, what: str) -> None:
    """The assertion half of :func:`owner_call`, so the message is uniform."""
    assert not isinstance(outcome, OwnerRefused), (
        f"the fence refused the conversation's OWN holder on {what}. A fence "
        f"that refuses the owner is a different failure from the one the "
        f"refusal legs check, not a fix for it.\n  {outcome}"
    )


def extract_tree(tmp_path: pathlib.Path, *extra: str, ref: str = "HEAD") -> pathlib.Path:
    """A private, byte-fresh copy of *ref*'s tree under *tmp_path*.

    HEAD by default, not the working tree: an immutable object, so a row cannot
    be measuring uncommitted state it did not describe.

    *ref* exists for one caller and one reason. A row written but not yet
    committed cannot be measured against HEAD — the guard it names is not there
    yet, so it reports "0 matches" and looks stale rather than unproven. A
    pre-commit check hands in the commit object ``git stash create`` makes from
    the working tree, which is precisely what is about to become HEAD. Every
    other caller leaves this alone and keeps measuring an immutable object.
    """
    git_dir = _git_dir()
    if git_dir is None:
        pytest.skip(
            "no git repository to extract a tree from; the pins themselves "
            "still run without it"
        )
    out = tmp_path / "tree"
    out.mkdir()
    archive = subprocess.run(
        [
            "git", "-C", git_dir, "archive", ref, "--",
            *BASE_EXTRACT_PATHSPEC, *extra,
        ],
        capture_output=True,
    )
    assert archive.returncode == 0, archive.stderr.decode(errors="replace")
    extract = subprocess.run(
        ["tar", "-x", "-C", str(out)], input=archive.stdout, capture_output=True
    )
    assert extract.returncode == 0, extract.stderr.decode(errors="replace")
    return out


def run_pin(
    tree: pathlib.Path,
    pin_module: str,
    pin: str,
    scratch: pathlib.Path,
) -> subprocess.CompletedProcess:
    """Run one check inside *tree*, importing it BY PATH from that tree."""
    scratch.mkdir(parents=True, exist_ok=True)
    probe = textwrap.dedent(
        f"""
        import importlib.util, pathlib, sys
        sys.path.insert(0, {str(tree)!r})
        spec = importlib.util.spec_from_file_location(
            "pins_under_mutation", {str(tree / pin_module)!r}
        )
        module = importlib.util.module_from_spec(spec)
        # Registered BEFORE exec: @dataclass resolves annotations through
        # sys.modules[cls.__module__], and an unregistered module makes that
        # None. That failure looks like the pin not holding on a clean extract,
        # i.e. like the harness measuring nothing.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        import hermes_state
        loaded = pathlib.Path(hermes_state.__file__).resolve()
        assert loaded.is_relative_to(pathlib.Path({str(tree)!r}).resolve()), (
            "the probe imported %s, not the extracted tree" % loaded
        )
        module.PINS[{pin!r}](pathlib.Path({str(scratch)!r}))
        print("PIN-HELD")
        """
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(scratch),
        "PYTHONPATH": str(tree),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    # A pin that needs the BASE object has to be able to find it from inside
    # the extract, and the extract is a bare tree with no `.git`. This names an
    # immutable commit in the real repository, which is the one thing that
    # cannot smuggle uncommitted state into the fixture — the same reason
    # ``HERMES_BASE_TREE_GIT_DIR`` exists at all. Without it those pins would
    # skip inside the subprocess, and a skip in the CLEAN run reads as "the pin
    # does not hold", i.e. as a harness fault rather than a missing fixture.
    git_dir = _git_dir()
    if git_dir is not None:
        env["HERMES_BASE_TREE_GIT_DIR"] = git_dir
    return subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(tree),
        env=env,
        capture_output=True, text=True, timeout=300,
    )


def assert_mutation_kills_the_pin(
    mutation: Mutation,
    pin_module: str,
    tmp_path: pathlib.Path,
    *extra_paths: str,
    ref: str = "HEAD",
) -> None:
    """Clean, mutated, restored. Anything else is a row that measures nothing.

    *extra_paths* widens the extract for a pin whose enforcement seam is not in
    the state layer. ``BASE_EXTRACT_PATHSPEC`` is deliberately narrow — enough
    to open a store — and a pin that drives a gateway command needs the rest of
    the package, which it asks for as ``"."`` rather than by naming the modules
    it happens to import today.
    """
    tree = extract_tree(tmp_path, pin_module, *extra_paths, ref=ref)
    target = tree / mutation.module
    original = target.read_text(encoding="utf-8")

    occurrences = original.count(mutation.find)
    assert occurrences == 1, (
        f"the mutation for {mutation.pin} matches {occurrences} places in "
        f"{mutation.module}, so it no longer names one guard. Keyed by content "
        f"on purpose — a line number would have gone stale silently. Re-derive "
        f"the anchor:\n{mutation.find!r}"
    )

    clean = run_pin(tree, pin_module, mutation.pin, tmp_path / "clean")
    assert clean.returncode == 0, (
        f"{mutation.pin} does not hold on the UNMUTATED extract, so this row "
        f"measures nothing:\n{clean.stdout}\n{clean.stderr}"
    )
    assert "PIN-HELD" in clean.stdout

    target.write_text(original.replace(mutation.find, mutation.replace, 1))
    killed = run_pin(tree, pin_module, mutation.pin, tmp_path / "mutated")
    assert killed.returncode != 0, (
        f"{mutation.pin} still passed with its guard removed ({mutation.why}). "
        f"It is asserting something the code cannot do anyway, and it will keep "
        f"passing after that guard is deleted:\n{killed.stdout}\n{killed.stderr}"
    )
    # AND IT HAS TO HAVE DIED BY OBSERVING, NOT BY FALLING OVER.
    #
    # A non-zero exit is not a kill. A mutation that breaks an import, moves a
    # signature, or trips a TypeError on the way in makes the probe exit
    # non-zero without the pin ever reading the state the guard protects — so
    # the row scores as discrimination while discriminating nothing. That is
    # the same blind spot review found in the base-binary attempt table, where
    # an attempt scored "did not change the state" because the snapshot did not
    # read the column the write lands in.
    #
    # So the mutated run must end in the PIN'S OWN AssertionError. A pin that
    # dies any other way is either measuring a crash or has a mutation aimed at
    # the wrong seam, and both are rows that would keep passing after the guard
    # they name is deleted.
    trace = f"{killed.stdout}\n{killed.stderr}"
    assert "AssertionError" in trace, (
        f"{mutation.pin} failed under its mutation, but NOT by its own "
        f"assertion — nothing in the failure names an AssertionError, so the "
        f"pin never reached the observation it exists to make. A row that "
        f"kills its pin by crashing proves the mutation runs, not that the pin "
        f"can see what the guard protects.\n"
        f"mutation: {mutation.module} :: {mutation.find!r} -> "
        f"{mutation.replace!r}\n{trace}"
    )
    assert "PIN-HELD" not in killed.stdout, (
        f"{mutation.pin} printed PIN-HELD under its mutation and then exited "
        f"non-zero anyway, so the non-zero exit came from something after the "
        f"pin passed:\n{trace}"
    )

    target.write_text(original)
    restored = run_pin(tree, pin_module, mutation.pin, tmp_path / "restored")
    assert restored.returncode == 0, (
        f"{mutation.pin} did not recover when the guard was put back, so the "
        f"failure above was not caused by the mutation:\n{restored.stdout}\n"
        f"{restored.stderr}"
    )


def assert_every_pin_has_a_killer(pins, mutations) -> None:
    """No pin without a killer, and no killer without a pin.

    A property added to ``PINS`` with no row in the table reads as coverage and
    is not, because nothing has ever shown it can fail.
    """
    pinned = set(pins)
    mutated = {mutation.pin for mutation in mutations}
    assert pinned == mutated, (
        f"pins without a mutation row: {sorted(pinned - mutated)}; "
        f"mutation rows naming no pin: {sorted(mutated - pinned)}"
    )
