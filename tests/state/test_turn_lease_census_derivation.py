"""The census denominator has to be DERIVED, not typed.

Review finding A, restated as a test.

``tests/state/test_turn_lease_writer_census.py`` states its denominator, asserts
its premise and enumerates every production call site — all of which the
previous check did not do. It bought that by making membership of
"context-bearing mutator" a ``frozenset`` literal, and every stage of the scan
filters on that literal first:

    _self_fencing_sweeps()   `if node.name not in CONTEXT_BEARING_MUTATORS`
    census()                 `if fn.attr not in CONTEXT_BEARING_MUTATORS`

So a mutator whose name nobody typed into the literal is invisible to the whole
file — it is not in the denominator, it is not in the derived sweep set, and its
production call sites are not counted. "Every production transcript writer goes
through the fence" then means "every transcript writer *someone remembered*".

THE CONSTRUCTION BELOW IS THE REVIEWER'S, VERBATIM
    A new ``SessionDB`` method that calls the sweep-admission helper and throws
    the answer away, then deletes the transcript; and a production caller for
    it. On the tree that shipped, ``pytest tests/state/test_turn_lease_writer_census.py``
    reported 5 passed with both of those present.

    It is not a hypothetical shape. It is the exact shape the census docstring
    claims to catch: *"adding a decorative call to the admission helper puts a
    mutator in the derived set and then fails right here"*. That sentence is
    true only for a name already in the literal.

WHAT THIS FILE ASSERTS
    Both halves of the pipeline take a tree, and both find the new mutator in
    it without anyone editing a set:

    1. the context-bearing set derived from ``<root>/hermes_state.py`` contains
       ``decorative_sweep``;
    2. the census over ``<root>`` reaches ``gateway/mirror.py``'s call to it.

    (1) is the load-bearing one. (2) follows from it, and is asserted separately
    so that a derivation which is right but unwired still fails.

WHY THE TREE IS SYNTHETIC AND THE SOURCE IS REAL
    The fixture writes the REAL ``hermes_state.py`` with the method spliced into
    the real ``class SessionDB`` body, so the derivation is exercised against
    the actual file it has to work on — its imports, its decorators, its nested
    ``_do(conn)`` closures — rather than a toy that happens to match whatever
    pattern the derivation looks for. Nothing here imports the synthetic module;
    the derivation is static, which is what lets it run on a tree that is not
    the one under the interpreter.
"""

from __future__ import annotations

import ast
import pathlib
import textwrap

import pytest

from tests.state import test_turn_lease_writer_census as census_mod


#: The reviewer's construction, verbatim. A write transaction that calls the
#: admission helper, discards the answer, and deletes every transcript row.
DECORATIVE_SWEEP = textwrap.dedent(
    '''
    def decorative_sweep(self, older_than):
        def _do(conn):
            rows = [r[0] for r in conn.execute("SELECT id FROM sessions").fetchall()]
            self._skip_leased_conversations(conn, rows)   # called, result discarded
            conn.execute("DELETE FROM messages")          # transcript deleted, unfenced
        return self._execute_write(_do)
    '''
)

#: Its production caller, verbatim.
HOUSEKEEPING_CALLER = textwrap.dedent(
    '''
    """Synthetic production module for the census derivation test."""

    def _housekeeping(db, older_than):
        return db.decorative_sweep(older_than)
    '''
).lstrip()


def _splice_into_sessiondb(source: str, method_src: str) -> str:
    """Return *source* with *method_src* added to the end of ``class SessionDB``.

    Located by symbol: the class is found in the AST and the insertion point is
    the last line of its last member, so nothing here depends on a line number
    or on what happens to sit above or below the class.
    """
    tree = ast.parse(source)
    classes = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "SessionDB"
    ]
    assert len(classes) == 1, (
        f"expected exactly one `class SessionDB` in the source, found "
        f"{len(classes)} — the splice point is no longer unambiguous"
    )
    body = classes[0].body
    assert body, "class SessionDB has an empty body"
    after = body[-1].end_lineno
    lines = source.splitlines(keepends=True)
    indented = textwrap.indent(method_src, "    ")
    return "".join(lines[:after]) + "\n" + indented + "\n" + "".join(lines[after:])


@pytest.fixture
def tree_with_an_untyped_mutator(tmp_path):
    """A tree that differs from this one by one method and one caller."""
    real = census_mod.REPO_ROOT / "hermes_state.py"
    spliced = _splice_into_sessiondb(
        real.read_text(encoding="utf-8"), DECORATIVE_SWEEP
    )
    # It has to still parse, or the derivation would "miss" it for the wrong
    # reason and this test would pass on a broken fixture.
    ast.parse(spliced)
    (tmp_path / "hermes_state.py").write_text(spliced, encoding="utf-8")
    (tmp_path / "gateway").mkdir()
    (tmp_path / "gateway" / "mirror.py").write_text(
        HOUSEKEEPING_CALLER, encoding="utf-8"
    )
    (tmp_path / "tui_gateway").mkdir()
    (tmp_path / "tui_gateway" / "__init__.py").write_text("", encoding="utf-8")
    return tmp_path


def _derived_mutators(root: pathlib.Path) -> frozenset:
    """The census's own answer to "what is a context-bearing mutator here?".

    ``getattr`` rather than a plain call so that the failure is the finding
    rather than an ``AttributeError``: a census that has no derivation can only
    answer with its literal, and answering with a literal is the defect.
    """
    derive = getattr(census_mod, "derive_context_bearing_mutators", None)
    if derive is None:
        return frozenset(census_mod.CONTEXT_BEARING_MUTATORS)
    return frozenset(derive(root))


def _census_of(root: pathlib.Path) -> list:
    """The census over *root*, or over whatever it insists on scanning."""
    try:
        return census_mod.census(root)
    except TypeError:
        return census_mod.census()


def test_a_new_transcript_deleter_enters_the_denominator_without_being_typed(
    tree_with_an_untyped_mutator,
):
    """A mutator nobody listed is still a mutator."""
    derived = _derived_mutators(tree_with_an_untyped_mutator)
    assert "decorative_sweep" in derived, (
        "`decorative_sweep` deletes every row of `messages` inside a SessionDB "
        "write transaction and the census does not consider it a "
        "context-bearing mutator, because membership is a set literal in the "
        "test file rather than a property of the code. Everything downstream "
        "filters on that set first, so this method and every production call "
        "to it are outside the denominator the census reports coverage of.\n"
        f"derived set ({len(derived)}): {sorted(derived)}"
    )


def test_the_census_reaches_the_production_caller_of_an_untyped_mutator(
    tree_with_an_untyped_mutator,
):
    """Derivation is only half of it; the scan has to use the result."""
    sites = _census_of(tree_with_an_untyped_mutator)
    reached = [
        s for s in sites
        if s["mutator"] == "decorative_sweep"
        and s["location"].startswith("gateway/mirror.py")
    ]
    assert reached, (
        "gateway/mirror.py calls decorative_sweep(), which deletes the "
        "transcript, and the census did not count the call site. The census "
        "reports a coverage fraction over a denominator this call is not in.\n"
        f"sites seen ({len(sites)}): "
        f"{sorted({s['location'] for s in sites})[:20]}"
    )


def test_the_derived_set_still_contains_every_writer_that_was_listed_by_hand(
    tree_with_an_untyped_mutator,
):
    """The derivation must be a superset of the literal it replaces.

    Losing an entry while gaining ``decorative_sweep`` would trade one hole for
    another, and the census would keep reporting full coverage either way. The
    literal is the previous round's hand-checked answer, so it is the floor.
    """
    derived = _derived_mutators(tree_with_an_untyped_mutator)
    hand_written = {
        "append_message", "append_messages_batch", "replace_messages",
        "archive_and_compact", "publish_compression_child", "rewind_to_message",
        "restore_rewound", "clear_messages", "set_latest_user_api_content",
        "set_message_reaction", "take_unseen_reactions",
        "set_latest_matching_message_display_kind",
        "purge_stale_tool_call_markers", "delete_session", "delete_sessions",
        "delete_empty_sessions", "prune_sessions",
    }
    lost = hand_written - derived
    assert not lost, (
        f"the derivation dropped writers the hand-written set had: "
        f"{sorted(lost)}"
    )
