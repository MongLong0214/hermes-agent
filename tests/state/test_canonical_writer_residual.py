"""Residual zero over the canonical writers — the ratchet, moved into the repo.

WHY THIS FILE EXISTS
    The residual-zero number this branch has been reporting for the canonical
    ``SessionDB`` writers came from a DERIVATION SCRIPT RUN BY HAND in a
    scratchpad. Nothing in the repository re-derived it, and the only ratchet
    over that surface was the replay-column one in
    ``test_turn_lease_replay_column_writers``, which is keyed on a specific set
    of columns. So a seventy-second writer that did not touch a replay column
    could land with nothing failing — and "nothing failed" would have been read
    as "residual is still zero", which is exactly the substitution
    ``test_raw_sink_census`` was written to stop on the other half of the
    surface.

    A number nobody re-derives is a number that was true once.

THE DENOMINATOR, STATED SO IT CAN BE CHECKED
    Every method of ``SessionDB`` and its mixins whose own SQL writes a table on
    :data:`hermes_state_common.TURN_FENCE_SURFACE`. The table set is READ FROM
    THE DECLARATION, not typed here, so widening the fence widens this census in
    the same commit — which is how ``compression_locks`` and
    ``session_turn_leases`` entered it.

    The other half of the surface — statements that are not ``SessionDB``
    methods at all — belongs to ``test_raw_sink_census``.
    :func:`test_every_fenced_table_has_a_census_that_counts_its_writers` asserts
    the two together leave no fenced table uncounted.

FOUR CLASSES, AND EVERY ONE OF THEM IS A PREDICATE OVER THE SOURCE
    admits          the method's transaction reaches the canonical turn-lease
                    admission: it raises ``SessionTurnLeaseLostError``, or it
                    hands its connection to something that does. Same closure
                    ``test_turn_fence_surface`` derives the fence surface from,
                    imported rather than reimplemented.
    inner           the method is only ever handed ANOTHER covered method's
                    open transaction, so its statements run inside a decision
                    that was already made. ``_store_system_prompt`` is this.
    schema-repair   every call site of the method, transitively, is inside
                    ``_init_schema``. Connection setup: no caller holds a handle
                    yet, so there is no call site at which a grant could be
                    presented. ``_init_schema`` itself is the census's one
                    argued exemption, and ``test_turn_lease_writer_census``
                    already asserts it cannot reach a row this generation wrote.
    token-scoped    every statement the method aims at a fenced table either
                    constrains ``holder`` — a token the caller had to be given —
                    or is an ``INSERT OR IGNORE`` that cannot overwrite a row
                    that exists. The three compression-lock writers are this:
                    a caller with no token can neither take a lock somebody
                    holds nor free it. Same shape as the ``claim-scoped`` ground
                    in ``test_raw_sink_census``, and checked the same way — by
                    reading the SQL, not by being on a list.

    Anything else is RESIDUAL and fails. There is no list of names anywhere in
    this file: a method is classified by what its source does, so a new writer
    has to earn a class rather than inherit one.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

import hermes_state_common
from tests.state.lease_mutation_harness import (
    Mutation,
    assert_every_pin_has_a_killer,
    assert_mutation_kills_the_pin,
)
from tests.state import test_turn_fence_surface as surface_mod
from tests.state import test_turn_lease_writer_census as census_mod

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)
#: The ratchet reads the SessionDB source; it needs the tree, not a store.
_EXTRA_EXTRACT = (".",)

#: Any statement that writes a table, and which table.
WRITE_STATEMENT = re.compile(
    r"\b(?:insert(?:\s+or\s+\w+)?\s+into|update|delete\s+from)\s+"
    r"([A-Za-z_][A-Za-z_0-9]*)",
    re.IGNORECASE,
)

#: One statement, from the first keyword to the end of the literal. Used to ask
#: what a single statement is scoped by without a SQL parser: the grounds below
#: are about the presence of a scoping predicate, and a substring from the verb
#: onward is enough to answer that without pretending to parse.
STATEMENT_SPAN = re.compile(
    r"\b(insert(?:\s+or\s+\w+)?\s+into|update|delete\s+from)\s+"
    r"([A-Za-z_][A-Za-z_0-9]*)(?P<rest>.*?)(?=\b(?:insert\s+into|update\s+"
    r"[A-Za-z_]|delete\s+from)\b|$)",
    re.IGNORECASE | re.DOTALL,
)

#: The schema-init entry point. It is not an exemption list: it is the ONE
#: method ``test_turn_lease_writer_census`` already exempts with an argued and
#: asserted ground, read from that census rather than restated.
SCHEMA_INIT = "_init_schema"


def fenced_tables() -> frozenset:
    return frozenset(
        table.lower() for table, _op in hermes_state_common.TURN_FENCE_SURFACE
    )


def _sql_literals(node: ast.AST):
    for inner in ast.walk(node):
        if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
            yield inner.value


def tables_written_by(node: ast.AST, tables: frozenset) -> set:
    found = set()
    for value in _sql_literals(node):
        for match in WRITE_STATEMENT.finditer(value):
            name = match.group(1).lower()
            if name in tables:
                found.add(name)
    return found


def _self_calls(node: ast.AST) -> set:
    """Every ``self.<name>(…)`` under *node*, connection or not.

    Wider than the transaction-hop rule on purpose: this one answers "who can
    invoke this at all", which is the question the schema-repair ground turns
    on. A method reachable from a live caller by ANY route is not connection
    setup, whatever it is handed.
    """
    return {
        inner.func.attr
        for inner in ast.walk(node)
        if isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Attribute)
        and isinstance(inner.func.value, ast.Name)
        and inner.func.value.id == "self"
    }


def schema_repair_closure(methods: dict) -> frozenset:
    """Methods whose every caller is, transitively, the schema init.

    Derived from the call graph. A method that grows a second caller outside
    connection setup leaves this set on the next run, which is the point: the
    ground is "nobody can present a grant here", and a new caller is exactly
    what makes that false.
    """
    callers: dict = {name: set() for name in methods}
    for name, (_module, node) in methods.items():
        for callee in _self_calls(node):
            if callee in callers:
                callers[callee].add(name)

    repair = {SCHEMA_INIT}
    changed = True
    while changed:
        changed = False
        for name in methods:
            if name in repair:
                continue
            who = callers[name]
            if who and who <= repair:
                repair.add(name)
                changed = True
    return frozenset(repair - {SCHEMA_INIT})


def _statement_is_token_scoped(verb: str, rest: str) -> bool:
    """One statement that cannot take or free what the caller was not given.

    Two shapes, both read off the SQL:

    * ``INSERT OR IGNORE`` — creates a row only where none exists, so it can
      never overwrite a holder;
    * anything else must constrain ``holder``, the token the caller had to be
      handed by whoever granted it.

    A bare ``INSERT`` is NOT token-scoped, and neither is ``INSERT OR REPLACE``:
    both can put a row where another holder's row was.
    """
    verb = " ".join(verb.lower().split())
    if verb.startswith("insert"):
        return verb == "insert or ignore into"
    return bool(re.search(r"\bholder\s*=", rest, re.IGNORECASE))


def is_token_scoped(node: ast.AST, tables: frozenset) -> bool:
    """True when EVERY fenced-table statement in *node* is token-scoped."""
    seen = False
    for value in _sql_literals(node):
        for match in STATEMENT_SPAN.finditer(value):
            if match.group(2).lower() not in tables:
                continue
            seen = True
            if not _statement_is_token_scoped(match.group(1), match.group("rest")):
                return False
    return seen


def census() -> dict:
    """``{method: classification}`` over the canonical writers of fenced tables."""
    methods, missing = census_mod._sessiondb_implementation(census_mod.REPO_ROOT)
    assert not missing, (
        f"the SessionDB implementation could not be read in full ({missing}); "
        f"every count below would be short by whatever those modules hold"
    )
    tables = fenced_tables()
    assert tables, "the fence declares no tables; this census has no denominator"

    writers = {
        name for name, (_module, node) in methods.items()
        if tables_written_by(node, tables)
    }

    full = surface_mod.admitted_transaction_closure(methods)
    admits = set(
        name for name, (_module, node) in methods.items()
        if surface_mod._raises_the_turn_lease_refusal(node)
    )
    changed = True
    while changed:
        changed = False
        for name, (_module, node) in methods.items():
            if name in admits:
                continue
            calls = set(
                surface_mod._self_calls_handing_over_the_connection(node)
            )
            if calls & admits:
                admits.add(name)
                changed = True
    inner = full - admits
    repair = schema_repair_closure(methods)
    argued = set(census_mod.NOT_CONTEXT_BEARING)

    verdicts = {}
    for name in sorted(writers):
        node = methods[name][1]
        if name in admits:
            verdicts[name] = "admits"
        elif name in inner:
            verdicts[name] = "inner"
        elif name in argued:
            verdicts[name] = "exempt(argued)"
        elif name in repair:
            verdicts[name] = "schema-repair"
        elif is_token_scoped(node, tables):
            verdicts[name] = "token-scoped"
        else:
            verdicts[name] = "RESIDUAL"
    return verdicts


def _report(verdicts: dict) -> str:
    counts: dict = {}
    for verdict in verdicts.values():
        counts[verdict] = counts.get(verdict, 0) + 1
    lines = [
        f"canonical SessionDB/mixin writers of a fenced table: {len(verdicts)}",
        f"fenced tables: {sorted(fenced_tables())}",
    ]
    for verdict in sorted(counts):
        lines.append(f"  {verdict:<16}{counts[verdict]}")
    residual = sorted(n for n, v in verdicts.items() if v == "RESIDUAL")
    lines.append(f"RESIDUAL: {len(residual)}")
    lines.extend(f"    {name}" for name in residual)
    return "\n".join(lines)


def check_the_canonical_writer_residual_is_zero(_tmpdir=None) -> None:
    """The ratchet. A writer with no class fails here and names itself.

    The counts are PRINTED rather than written into a docstring for the reason
    the raw-sink census prints its own: three places in this branch once stated
    three different numbers for one set, because a number in prose is a number
    nobody re-derives.

    Takes an ignored directory argument so the mutation harness can drive it
    with the same signature every other pin has; it needs no store, because it
    reads source rather than rows.
    """
    verdicts = census()
    report = _report(verdicts)
    print("\n" + report)

    assert verdicts, "the census found no writers at all; the scan is broken"
    residual = sorted(n for n, v in verdicts.items() if v == "RESIDUAL")
    assert not residual, (
        "these SessionDB writers touch a fenced table and reach no admission, "
        "run inside no admitted transaction, are not connection setup, and "
        "carry no token scope. Each one can move a row on a conversation a "
        "live turn owns:\n  " + "\n  ".join(residual) + "\n\n" + report
    )


def test_every_ground_is_a_predicate_that_can_reject(capsys):
    """A ground that nothing can fail is not a ground.

    Each non-admitted class is checked against a synthetic method that ALMOST
    qualifies, and must be rejected. Without this the grounds could be widened
    until everything passed and the residual would stay zero by construction —
    which is the shape of "fenced or on a list", and the reason the writer
    census was rebuilt in the first place.
    """
    tables = frozenset({"compression_locks"})

    holder_scoped = ast.parse(
        "def m(self, session_id, holder):\n"
        "    conn.execute('DELETE FROM compression_locks "
        "WHERE session_id = ? AND holder = ?')\n"
    ).body[0]
    assert is_token_scoped(holder_scoped, tables)

    # Same method with the token dropped from the WHERE: it can now free a lock
    # the caller was never given.
    unscoped = ast.parse(
        "def m(self, session_id, holder):\n"
        "    conn.execute('DELETE FROM compression_locks WHERE session_id = ?')\n"
    ).body[0]
    assert not is_token_scoped(unscoped, tables), (
        "a DELETE with no holder predicate was accepted as token-scoped; the "
        "ground would then absolve a writer that frees anybody's lock"
    )

    # INSERT OR IGNORE cannot overwrite; INSERT OR REPLACE can, and must not
    # be mistaken for it.
    assert is_token_scoped(ast.parse(
        "def m(self):\n"
        "    conn.execute('INSERT OR IGNORE INTO compression_locks "
        "(session_id, holder) VALUES (?, ?)')\n"
    ).body[0], tables)
    assert not is_token_scoped(ast.parse(
        "def m(self):\n"
        "    conn.execute('INSERT OR REPLACE INTO compression_locks "
        "(session_id, holder) VALUES (?, ?)')\n"
    ).body[0], tables), (
        "INSERT OR REPLACE was accepted as token-scoped; it puts a row where "
        "another holder's row was, which is the theft the ground denies"
    )

    # One unscoped statement is enough to lose the ground, even beside a
    # scoped one — a method is only as scoped as its loosest statement.
    assert not is_token_scoped(ast.parse(
        "def m(self, holder):\n"
        "    conn.execute('DELETE FROM compression_locks WHERE holder = ?')\n"
        "    conn.execute('UPDATE compression_locks SET expires_at = ?')\n"
    ).body[0], tables)

    # And a method that writes nothing fenced cannot claim the ground at all,
    # which is what stops it from becoming a blanket pass.
    assert not is_token_scoped(ast.parse(
        "def m(self):\n    return 1\n"
    ).body[0], tables)

    # schema-repair: a method reachable from a LIVE caller is not connection
    # setup, however many of its callers are inside the schema init.
    methods = {
        SCHEMA_INIT: ("m.py", ast.parse(
            "def _init_schema(self):\n    self._heal(cursor)\n"
        ).body[0]),
        "_heal": ("m.py", ast.parse("def _heal(self, cursor):\n    pass\n").body[0]),
        "_also_called_live": ("m.py", ast.parse(
            "def _also_called_live(self, cursor):\n    pass\n"
        ).body[0]),
        "append_message": ("m.py", ast.parse(
            "def append_message(self):\n    self._also_called_live(conn)\n"
        ).body[0]),
    }
    methods[SCHEMA_INIT][1].body.append(
        ast.parse("self._also_called_live(cursor)").body[0]
    )
    repair = schema_repair_closure(methods)
    assert "_heal" in repair
    assert "_also_called_live" not in repair, (
        "a method with a live caller was classified as connection setup; the "
        "ground is 'no caller can present a grant here' and a live caller can"
    )


def test_every_fenced_table_has_a_census_that_counts_its_writers():
    """No fenced table may be outside BOTH censuses.

    The two halves of the surface are counted in two files: ``SessionDB``
    methods here, statements that are not ``SessionDB`` methods in
    ``test_raw_sink_census``. A table written from neither is a table whose
    writers nobody enumerates — the exact gap that let ``tools/async_delegation``
    hold fifteen statements disposed of by a docstring.
    """
    from tests.state import test_raw_sink_census as raw_mod

    tables = fenced_tables()
    methods, _missing = census_mod._sessiondb_implementation(census_mod.REPO_ROOT)
    counted_here = set()
    for _name, (_module, node) in methods.items():
        counted_here |= tables_written_by(node, tables)

    statements, _funcs = raw_mod.derive_raw_sink_statements(raw_mod.REPO_ROOT)
    counted_raw = {statement["table"].lower() for statement in statements}

    uncounted = sorted(tables - counted_here - counted_raw)
    assert not uncounted, (
        f"these tables are on the turn-fence surface and neither census "
        f"enumerates a writer for them: {uncounted}. A fenced table with no "
        f"counted writers is a table whose next writer arrives unclassified.\n"
        f"  counted by this file: {sorted(counted_here)}\n"
        f"  counted by the raw-sink census: {sorted(counted_raw)}"
    )


PINS = {
    "check_the_canonical_writer_residual_is_zero":
        check_the_canonical_writer_residual_is_zero,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_canonical_writer_residual_property(name, capsys):
    with capsys.disabled():
        PINS[name]()


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_the_canonical_writer_residual_is_zero",
        module="hermes_state.py",
        find=(
            "            self._admit_routing_write(\n"
            "                conn, [(scope, session_key)], entry_json,\n"
            "                turn_lease_holder, turn_lease_ttl_seconds,\n"
            "            )\n"
        ),
        replace="",
        why="save_gateway_routing_entry then writes the routing index — which "
            "decides where a platform reply lands — inside a transaction that "
            "consults nothing. It reaches no admission, extends no admitted "
            "transaction, is not connection setup and carries no token scope, "
            "so the census must name it. A ratchet that cannot see a writer "
            "losing its admission is the hand-run script this file replaced",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    assert_mutation_kills_the_pin(
        mutation, str(_SELF), tmp_path, *_EXTRA_EXTRACT
    )


def test_every_pin_has_a_mutation_that_kills_it():
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)
