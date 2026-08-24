"""The raw-SQLite sinks, derived — the half the SessionDB census cannot see.

WHY A SEPARATE CENSUS EXISTS AT ALL
    :mod:`tests.state.test_turn_lease_writer_census` is a census of CALL SITES
    of ``SessionDB`` mutators, and its exclusion proof —
    ``test_the_denominator_states_what_it_left_out`` — works by asserting that
    an excluded module cannot obtain a ``SessionDB``. Both of those are only
    meaningful for code that goes THROUGH ``SessionDB``.

    ``tools/async_delegation.py`` writes ``async_delegations`` in the live
    ``state.db`` through module-level functions, not through ``SessionDB``
    methods, so that census cannot see it **by construction** — not by
    oversight. Its fifteen statements were therefore disposed of by a module
    docstring and one pin, and a sixteenth added tomorrow would fail nothing.
    The symptom was already visible: three different places in this branch
    stated three different counts for one set, which is what prose does and
    derivation does not.

    THE HANDLE IS GONE; THE STATEMENTS ARE NOT.
    That module used to hold its own ``sqlite3.connect`` on the store. It no
    longer does — ``async_delegations`` joined ``TURN_FENCE_SURFACE`` and an
    unmarked connection cannot write a fenced table, so the writes moved onto
    ``SessionDB.write_transaction``. What did NOT change is why this census
    exists: the statements still live outside any ``SessionDB`` method, so the
    call-site census still cannot reach them and they still need a derived
    disposition of their own.

    So the set is derived here, at test time, and the counts are PRINTED rather
    than written down anywhere.

THE RULE, AND WHY EACH ARM IS CHECKABLE
    A statement in a raw sink is admitted, or it carries a ground this file
    verifies. Three grounds, all read off the SQL:

    admitted        the enclosing function calls the module's admission
                    helpers — ``_admit_delegations`` (refuse) or
                    ``_leased_delegation_ids`` (sweep). Those in turn borrow
                    ``SessionDB.admit_on_connection`` /
                    ``skip_leased_on_connection``, so there is one rule, not a
                    second implementation.
    claim-scoped    the statement's WHERE carries ``delivery_claim = ?``. The
                    claim token is a different and correct mutual exclusion:
                    the actor is the consumer INJECTING the completion into the
                    parent's turn, not a bystander. That ground is not taken on
                    trust — ``check_the_claim_protocol_leaves_the_payload_pending``
                    in the sink's own pin file asserts both halves of it: a
                    claim and its release leave the payload ``pending``, and a
                    holder of the WRONG claim cannot move it at all.
    non-destructive the statement cannot remove a turn from the parent's
                    future: it is not a DELETE, and it does not assign
                    ``delivery_state`` anything but ``'pending'``. Creating a
                    record, or moving one TOWARD pending, cannot take work away
                    from a conversation.

    Anything else is ungrounded and fails here. That is the sixteenth-statement
    guard: a new DELETE, or a new ``delivery_state='dropped'`` without either
    the admission or the claim scope, has nowhere to land.

THE OTHER HALF, WHICH USED TO BE MISSING
    This file used to end by saying that the SessionDB surface was NOT derived
    by any test — that its residual-zero number came from a derivation script
    run by hand, and that the only ratchet in the repository over that surface
    was the replay-column one in ``test_turn_lease_replay_column_writers``.
    That gap is now closed by ``test_canonical_writer_residual``, which derives
    the canonical writers of every fenced table, classifies each one, prints its
    counts and fails on anything it cannot ground.

    The two files split the surface between them: statements that are
    ``SessionDB`` methods there, statements that are not here.
    ``test_every_fenced_table_has_a_census_that_counts_its_writers`` asserts the
    split leaves no fenced table uncounted, so neither file can quietly stop
    covering something by assuming the other does.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from tests.state.lease_mutation_harness import (
    Mutation,
    assert_every_pin_has_a_killer,
    assert_mutation_kills_the_pin,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: This file, so a mutated extract can run the same checks it defines.
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)

#: The extract needs the tools package, which is outside the state layer.
_EXTRA_EXTRACT = (".",)

#: The module that holds its own read-write handle on the live store. Asserted
#: to be the only one by :func:`test_only_one_module_holds_a_raw_handle_on_the_live_store`
#: rather than assumed here.
RAW_SINK = "tools/async_delegation.py"

#: Tables whose rows decide what a later turn replays.
FENCED_TABLES = (
    "sessions", "messages", "message_reactions", "system_prompts",
    "session_model_usage", "gateway_routing", "async_delegations",
)

_DML = re.compile(
    r"\b(?P<verb>insert(?:\s+or\s+\w+)?\s+into|replace\s+into|update|delete\s+from)"
    r"\s+(?P<table>" + "|".join(FENCED_TABLES) + r")\b",
    re.IGNORECASE,
)

#: ``delivery_state='...'`` anywhere in a SET clause.
_SETS_DELIVERY_STATE = re.compile(
    r"delivery_state\s*=\s*'(?P<value>[a-z_]+)'", re.IGNORECASE
)

#: The claim token in a predicate. ``delivery_claim=?`` with any spacing.
_CLAIM_SCOPED = re.compile(r"delivery_claim\s*=\s*\?", re.IGNORECASE)

#: The module's own admission helpers. Derived membership would be circular —
#: these are the names the classifier looks for, and
#: :func:`test_the_admission_helpers_reach_the_canonical_rule` proves each one
#: actually reaches SessionDB rather than re-implementing the predicate.
ADMISSION_HELPERS = ("_admit_delegations", "_leased_delegation_ids")


def _docstrings(tree) -> set:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                out.add(doc)
    return out


def _enclosing(funcs, line):
    best = None
    for f in funcs:
        if f.lineno <= line <= f.end_lineno and (best is None or f.lineno > best.lineno):
            best = f
    return best


def derive_raw_sink_statements(root: pathlib.Path, module: str = RAW_SINK):
    """Every fenced-table DML statement in *module*, with what it can do.

    Read off the source: the verb, the table, whether the statement moves
    ``delivery_state`` and to what, whether it is claim-scoped, and which
    function contains it. Docstrings are excluded — this module's docstrings
    quote its own SQL.
    """
    path = root / module
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    docs = _docstrings(tree)
    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if node.value in docs:
            continue
        for match in _DML.finditer(node.value):
            owner = _enclosing(funcs, node.lineno)
            state = _SETS_DELIVERY_STATE.search(node.value)
            out.append({
                "function": owner.name if owner else "<module>",
                "line": node.lineno,
                "verb": " ".join(match.group("verb").lower().split()),
                "table": match.group("table").lower(),
                "delivery_state": state.group("value").lower() if state else None,
                "claim_scoped": bool(_CLAIM_SCOPED.search(node.value)),
                "sql": " ".join(node.value.split())[:110],
            })
    return out, {f.name: f for f in funcs}


def classify(statement, funcs) -> str:
    """``admitted`` / ``claim-scoped`` / ``non-destructive`` / ``UNGROUNDED``."""
    owner = funcs.get(statement["function"])
    if owner is not None:
        called = {
            inner.func.id for inner in ast.walk(owner)
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
        }
        if called & set(ADMISSION_HELPERS):
            return "admitted"
    destructive = (
        statement["verb"].startswith("delete")
        or (statement["delivery_state"] not in (None, "pending"))
    )
    if not destructive:
        return "non-destructive"
    if statement["claim_scoped"]:
        return "claim-scoped"
    return "UNGROUNDED"


def _report(root: pathlib.Path) -> str:
    statements, funcs = derive_raw_sink_statements(root)
    rows, counts = [], {}
    for s in statements:
        verdict = classify(s, funcs)
        counts[verdict] = counts.get(verdict, 0) + 1
        rows.append(f"  {s['function']}:{s['line']:<5} {verdict:<16} {s['sql']}")
    header = (
        f"{RAW_SINK}: {len(statements)} fenced-table DML statements — "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    return header + "\n" + "\n".join(rows)


# ---------------------------------------------------------------------------
# The pin.
# ---------------------------------------------------------------------------

def check_every_raw_sink_statement_is_admitted_or_grounded(tmpdir) -> None:
    """No statement in the raw sink is disposed of by prose."""
    statements, funcs = derive_raw_sink_statements(REPO_ROOT)
    assert statements, (
        "the derivation found no DML in the raw sink, so it stopped seeing "
        "its own subject — which reads as coverage and is not"
    )
    ungrounded = [s for s in statements if classify(s, funcs) == "UNGROUNDED"]
    assert not ungrounded, (
        "these raw-SQLite statements are neither admitted nor grounded — they "
        "remove a turn from a parent conversation's future on a connection no "
        "fence can see:\n"
        + "\n".join(f"  {s['function']}:{s['line']} {s['sql']}" for s in ungrounded)
        + "\n\n" + _report(REPO_ROOT)
    )


PINS = {
    "check_every_raw_sink_statement_is_admitted_or_grounded":
        check_every_raw_sink_statement_is_admitted_or_grounded,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_raw_sink_property(name, tmp_path):
    PINS[name](tmp_path)


def test_the_counts_are_derived_and_printed(capsys):
    """The counts live here, computed, and nowhere else as prose.

    Three places in this branch once stated three different totals for this one
    set. Printing the derived table is what stops a fourth.
    """
    report = _report(REPO_ROOT)
    print("\n" + report)
    statements, funcs = derive_raw_sink_statements(REPO_ROOT)
    verdicts = {classify(s, funcs) for s in statements}
    assert verdicts <= {"admitted", "claim-scoped", "non-destructive"}, report
    assert "UNGROUNDED" not in report


def test_the_admission_helpers_reach_the_canonical_rule():
    """The sink's helpers BORROW SessionDB's rule; they do not re-implement it.

    This is the property that keeps the raw sink from becoming a second door.
    A helper that grew its own lineage walk and its own liveness predicate
    would satisfy the classifier above while drifting from the rule every other
    writer obeys.
    """
    source = (REPO_ROOT / RAW_SINK).read_text(encoding="utf-8")
    tree = ast.parse(source)
    funcs = {n.name: n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    borrowed = {
        "_admit_delegations": "admit_on_connection",
        "_leased_delegation_ids": "skip_leased_on_connection",
    }
    for helper, canonical in borrowed.items():
        assert helper in funcs, f"{helper} is gone; the classifier now matches nothing"
        attrs = {
            inner.func.attr for inner in ast.walk(funcs[helper])
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
        }
        assert canonical in attrs, (
            f"{helper} no longer calls SessionDB.{canonical}. If it grew its "
            f"own root walk or its own liveness check, the raw sink is a "
            f"SECOND implementation of the admission rule and the two will "
            f"drift — which is the failure this whole family exists to stop."
        )
    # And the canonical side must still exist to be borrowed.
    state = (REPO_ROOT / "hermes_state.py").read_text(encoding="utf-8")
    for canonical in borrowed.values():
        assert f"def {canonical}(" in state, (
            f"SessionDB.{canonical} is gone, so the sink is borrowing nothing"
        )


def test_only_one_module_holds_a_raw_handle_on_the_live_store():
    """A second raw sink must announce itself, not inherit this file's silence.

    Derived: a production module that resolves ``<hermes home>/state.db`` AND
    opens it read-write with its own ``sqlite3.connect``. The read-only probes
    (doctor, backup) and the rollback lane (which routes through
    ``offline_rebuild``) are excluded by the ``mode=ro`` / gate they already
    carry, not by being named here.
    """
    from tests.state import test_turn_lease_writer_census as census

    impl = {"hermes_state.py", "hermes_state_schema.py", "hermes_state_common.py",
            "hermes_state_search.py", "hermes_state_portability.py"}
    offenders = []
    for rel, path in census._production_files(REPO_ROOT):
        # RAW_SINK used to be exempt here, because it WAS the raw handle this
        # check counted to one. It is not any more: its writes run on
        # SessionDB's transaction. Leaving the carve-out would have made this
        # the one place a reopened private handle in that exact module could
        # hide, so the count is now to ZERO and the module is scanned like
        # every other.
        if rel.name in impl:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if '"state.db"' not in text and "'state.db'" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        docs = _docstrings(tree)
        writes = any(
            isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value not in docs and _DML.search(n.value)
            for n in ast.walk(tree)
        )
        if not writes:
            continue
        module_funcs = {
            n.name: n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        def _names_state_db(node) -> bool:
            """True when *node*'s source, or one hop into a local helper, names the store.

            The one hop matters and the loose version was measured wrong:
            `tools/async_delegation.py` connects to `_db_path()`, so a check on
            the call site alone misses it — while a check on "does this MODULE
            mention state.db anywhere" flagged `gateway/platforms/api_server.py`,
            which opens `response_store.db` and writes `sessions` through
            SessionDB's own `_execute_write`. Three unrelated facts in one file
            are not a sink.
            """
            text = ast.unparse(node)
            if "state.db" in text:
                return True
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                    helper = module_funcs.get(inner.func.id)
                    if helper is not None and "state.db" in ast.unparse(helper):
                        return True
            return False

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "connect"
                    and getattr(node.func.value, "id", "") == "sqlite3"):
                continue
            arg = node.args[0] if node.args else None
            if arg is None:
                continue
            literal = ast.unparse(arg)
            if "mode=ro" in literal:
                continue
            enclosing = _enclosing(list(module_funcs.values()), node.lineno)
            if not _names_state_db(arg) and not (
                enclosing is not None and _names_state_db(enclosing)
            ):
                continue
            offenders.append(f"{rel}:{node.lineno}: sqlite3.connect({literal[:60]})")
    assert not offenders, (
        "a second production module writes a fenced table through its own "
        "read-write handle on the live store. It is outside the SessionDB "
        "census by construction, exactly as tools/async_delegation.py is, so "
        "it needs its own derived disposition in this file rather than "
        f"inheriting silence:\n  " + "\n  ".join(offenders)
    )


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_every_raw_sink_statement_is_admitted_or_grounded",
        module="tools/async_delegation.py",
        find=(
            "        _admit_delegations(\n"
            "            store, conn, [delegation_id], turn_lease_holder,\n"
            "            f\"refusing to delete the durable record for \"\n"
            "            f\"{delegation_id!r}: it belongs to\",\n"
            "        )\n"
        ),
        replace="",
        why="the durable DELETE is destructive and not claim-scoped, so with "
            "its admission gone it falls through every ground the classifier "
            "knows and this census must name it",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    assert_mutation_kills_the_pin(mutation, str(_SELF), tmp_path, *_EXTRA_EXTRACT)


def test_every_pin_has_a_mutation_that_kills_it():
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)
