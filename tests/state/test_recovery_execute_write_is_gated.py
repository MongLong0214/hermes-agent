"""Every `_execute_write` in the recovery lane is reachable only through the gate.

WHAT THIS FILE IS FOR, AND WHAT IT IS NOT DUPLICATING
    ``SessionDB._execute_write`` takes an arbitrary callable and runs it on the
    registered, current-generation connection. Generation registration proves
    the binary is current — old-binary fencing — and says nothing about the
    canonical root, the holder or the epoch. So a caller that hands it a raw
    ``UPDATE sessions …`` is admitted by nothing.

    Nine production call sites do exactly that, all in the recovery lane::

        hermes_cli/session_recovery.py:441   _copy_table
        hermes_cli/session_recovery.py:698   recover_exact_rowid
        hermes_cli/session_recovery.py:742   copy_range
        hermes_cli/session_recovery.py:855   _copy_state_meta
        hermes_cli/session_recovery.py:979   _cleanup_partial_orphans
        hermes_cli/session_recovery.py:1405  _finalize_derived_metadata
        hermes_cli/session_lost_and_found.py:490  map_lost_and_found_rows
        hermes_cli/session_lost_and_found.py:581  stub_missing_parent_sessions
        hermes_cli/session_lost_and_found.py:604  rebuild_fts_indexes

    None of them carries a gate of its own, and reading thirty lines above any
    of them shows no gate. Their safety rests entirely on a gate TWO frames up:
    both production entry points wrap the whole rebuild in
    ``destination_db.offline_rebuild(...)`` and pass the yielded
    ``destination_store`` down. That is safety resting on a different
    component, which is precisely the shape that must be pinned rather than
    argued.

    :mod:`tests.state.test_recovery_writes_through_sessiondb` already pins the
    GATE's behaviour on rows — refused while a conversation is live-owned, and
    the row does not move; admitted once released, and it does. This file
    deliberately does not repeat that. What it pins is the half that rots: that
    every one of those nine sites is REACHABLE ONLY from inside the gate, so a
    tenth site added next to them fails here instead of shipping.

WHY THE STRUCTURAL HALF IS THE ONE THAT IS REACHABLE TODAY
    Both gated entry points create their destination fresh —
    ``recover_session_database``'s docstring states ``output_path`` must not
    exist — so in production the gate's ownership refusal can only fire when an
    operator points a rebuild at a store that already exists. The live risk is
    therefore not "recovery rewrites an owned conversation"; it is "the next
    ``_execute_write`` in these modules is written outside the gate, and
    nothing says so".

DERIVED, NOT LISTED
    The call sites above are printed in the failure message, not used as the
    input. :func:`_execute_write_sites` reads them out of the modules, and
    :func:`_gate_reachable` computes the set of functions reachable from a
    function that opens ``offline_rebuild``, to a fixpoint. Adding a site to a
    function nothing gated can reach fails; adding one to a gated function
    passes. Neither outcome can be reached by editing this file.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from tests.state.lease_mutation_harness import (
    Mutation,
    assert_every_pin_has_a_killer,
    assert_mutation_kills_the_pin,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: This file, so a mutated extract can run the same checks it defines.
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)

#: The extract needs the recovery package, which is outside the state layer.
_EXTRA_EXTRACT = (".",)

#: The two modules the census names as raw `sessions` writers.
RECOVERY_MODULES = (
    "hermes_cli/session_recovery.py",
    "hermes_cli/session_lost_and_found.py",
)


def _functions(tree):
    return [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _enclosing(funcs, line):
    best = None
    for f in funcs:
        if f.lineno <= line <= f.end_lineno and (best is None or f.lineno > best.lineno):
            best = f
    return best.name if best else "<module>"


def _called_names(node):
    """Every name this function calls, by attribute or bare name.

    Deliberately name-based rather than resolved: ``_copy_table`` is reached
    through ``copy_function = (_copy_table_salvage if allow_partial else
    _copy_table)`` and then ``copy_function(...)``, so a resolver that followed
    only direct calls would report those two as unreachable and this check
    would pass by not seeing them. Matching the NAME wherever it appears as a
    call target or a bare load inside a gated function over-approximates
    reachability, which is the direction that cannot hide an ungated site.
    """
    names = set()
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            fn = inner.func
            if isinstance(fn, ast.Name):
                names.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                names.add(fn.attr)
        elif isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load):
            names.add(inner.id)
    return names


def _recovery_index(root: pathlib.Path):
    """``(sites, calls, gates)`` read out of the recovery modules under *root*."""
    sites, calls, gates = [], {}, set()
    for rel in RECOVERY_MODULES:
        path = root / rel
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        funcs = _functions(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_execute_write"
            ):
                sites.append((rel, node.lineno, _enclosing(funcs, node.lineno)))
            if isinstance(node, ast.With):
                for item in node.items:
                    ctx = item.context_expr
                    if (
                        isinstance(ctx, ast.Call)
                        and isinstance(ctx.func, ast.Attribute)
                        and ctx.func.attr == "offline_rebuild"
                    ):
                        gates.add(_enclosing(funcs, node.lineno))
        for f in funcs:
            calls.setdefault(f.name, set()).update(_called_names(f))
    return sites, calls, gates


def _gate_reachable(calls, gates) -> set:
    """Functions reachable from a gate opener, to a fixpoint."""
    reachable = set(gates)
    changed = True
    while changed:
        changed = False
        for name in list(reachable):
            for callee in calls.get(name, ()):  # noqa: B007
                if callee in calls and callee not in reachable:
                    reachable.add(callee)
                    changed = True
    return reachable


# ---------------------------------------------------------------------------
# The pin.
# ---------------------------------------------------------------------------

def check_every_recovery_execute_write_is_reachable_only_through_the_gate(
    tmpdir,
) -> None:
    """No ``_execute_write`` in the recovery lane sits outside the rebuild gate."""
    sites, calls, gates = _recovery_index(REPO_ROOT)

    assert sites, (
        "no _execute_write call sites found in the recovery modules — the "
        "derivation stopped seeing its own subject, which reads as coverage "
        "and is not"
    )
    assert gates, (
        "no `with <db>.offline_rebuild(...)` block found in the recovery "
        "modules, so every one of the following raw-write sites is admitted "
        f"by nothing: {sites}"
    )

    reachable = _gate_reachable(calls, gates)
    ungated = [
        (rel, line, func) for rel, line, func in sites if func not in reachable
    ]
    assert not ungated, (
        f"these _execute_write sites are not reachable from any "
        f"offline_rebuild scope, so they run raw DML on the store's own "
        f"connection with no admission of any kind:\n"
        + "\n".join(f"  {rel}:{line} in {func}()" for rel, line, func in ungated)
        + f"\ngate openers: {sorted(gates)}\n"
        f"all sites: {[(r, l, f) for r, l, f in sites]}"
    )


PINS = {
    "check_every_recovery_execute_write_is_reachable_only_through_the_gate":
        check_every_recovery_execute_write_is_reachable_only_through_the_gate,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_recovery_execute_write_property(name, tmp_path):
    PINS[name](tmp_path)


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_every_recovery_execute_write_is_reachable_only_through_the_gate",
        module="hermes_cli/session_recovery.py",
        find=(
            "            with destination_db.offline_rebuild(\n"
            "                reason=\"session recovery\"\n"
            "            ) as destination_store:\n"
        ),
        replace=(
            "            with contextlib.nullcontext(destination_db) "
            "as destination_store:\n"
        ),
        why="replacing the gate with a no-op context is exactly how this lane "
            "regressed before — the destination was opened a second time and "
            "written from a handle that registered nothing. The six sites "
            "under this block become raw writers admitted by nobody, and the "
            "fail-closed ownership refusal never runs",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    assert_mutation_kills_the_pin(mutation, str(_SELF), tmp_path, *_EXTRA_EXTRACT)


def test_every_pin_has_a_mutation_that_kills_it():
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)
