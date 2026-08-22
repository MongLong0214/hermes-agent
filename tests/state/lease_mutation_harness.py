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


def extract_tree(tmp_path: pathlib.Path, *extra: str) -> pathlib.Path:
    """A private, byte-fresh copy of HEAD's tree under *tmp_path*.

    HEAD, not the working tree: an immutable object, so a row cannot be
    measuring uncommitted state it did not describe.
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
            "git", "-C", git_dir, "archive", "HEAD", "--",
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
    return subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(tree),
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(scratch),
            "PYTHONPATH": str(tree),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True, text=True, timeout=300,
    )


def assert_mutation_kills_the_pin(
    mutation: Mutation,
    pin_module: str,
    tmp_path: pathlib.Path,
) -> None:
    """Clean, mutated, restored. Anything else is a row that measures nothing."""
    tree = extract_tree(tmp_path, pin_module)
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
