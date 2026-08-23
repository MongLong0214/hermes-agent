"""The DB-level fence surface is DERIVED from source and mutation-tested.

WHY THIS FILE EXISTS
    The first version of the generation fence put triggers on ``messages`` only,
    and the probe against the base binary only called ``append_message``. Both
    halves of that were too narrow, and the narrowness was invisible because the
    tests that existed all passed:

        end_session, update_session_model, update_system_prompt,
        set_session_title, patch_session_model_config

    all WROTE, from the exact binary at the base commit, against a conversation
    this generation held the lease on. Not one of them touches ``messages``.
    They rewrite the model, the system prompt and the end state that the next
    turn replays under — provider-visible state, changed by a process that has
    never heard of the lease.

    THE EMPTY-SESSION DELETE, AND A CORRECTION I OWE THE RECORD
    I first reported that the predicted empty-session counterexample "does not
    reproduce", and that was wrong because I tested the wrong thing. Driving it
    through the old binary's ``delete_session`` IS refused — that method issues
    ``DELETE FROM messages WHERE session_id = ?`` and SQLite resolves a trigger
    program when it PREPARES a statement, so the refusal happens whether or not
    a row would have matched. But a foreign writer does not have to go through
    that method. A plain

        DELETE FROM sessions WHERE id = 'empty'

    never names ``messages`` at all, so nothing prepares a message trigger and
    the row is gone. Measured at cb61320a: ``empty_survives: false``,
    ``raw_empty_session_delete: accepted``. The prediction was right and my
    check was aimed one layer too high.
    :func:`test_the_measured_counterexample_at_cb61320a` is that measurement,
    kept verbatim as a test rather than as a paragraph.

HOW THE SURFACE IS DECIDED — AND THE SEED THAT WAS WRONG
    Not by a list in production that somebody keeps in step. Production
    DECLARES :data:`hermes_state_common.TURN_FENCE_SURFACE`; this file DERIVES
    the same set from the source, and fails when they differ. A new mutator that
    writes a new table therefore fails here until the declaration follows it.

    THE PREVIOUS DERIVATION SEEDED FROM THE WRONG SET, AND IT WAS MEASURABLE.
    It seeded from ``derive_context_bearing_mutators()`` — the MESSAGE-TABLE
    derivation — and then collected the tables those methods touch. So a writer
    reachable only from the session row never entered the seed at all:
    ``_store_system_prompt`` writes ``system_prompts`` and never names
    ``messages``, and rule 2 of that derivation adds CALLERS of derived members,
    not CALLEES. Four tables came out unfenced and a foreign connection wrote
    every one of them — see ``tests/state/test_turn_fence_adjunct_surface``,
    which keeps the measurement.

    THE DENOMINATOR IS THE TRANSACTION, NOT THE TABLE.
    A table belongs on the surface when production writes it inside a
    transaction that consults the canonical turn-lease admission. That is the
    property the barrier is for: whatever this generation decided it had to ask
    permission before writing is exactly what an old binary must not write
    without asking. Stated that way the derivation needs no list of tables, no
    list of mutators, and no exemption:

    1. SEED — the methods that raise the production refusal itself,
       ``hermes_state.SessionTurnLeaseLostError``. Anchored on the refusal
       rather than on a table name so that renaming a table, adding one, or
       moving a guard cannot quietly shrink the seed.
    2. UPWARD — a method that hands its own transaction's connection to a
       member is running that member's decision inside its transaction.
    3. INNER — a method handed a member's connection runs its DML inside that
       member's admitted transaction, so its tables are on the surface too.
       This is the direction the old seed did not have.
    4. BORROWED — a production module outside ``SessionDB`` that calls one of
       the closure's ``(self, conn, …)`` entry points is running the canonical
       decision on its own transaction, so every table IT writes is on the
       surface. ``tools/async_delegation`` is that module and
       ``async_delegations`` is that table.

    Rules 1-3 run to a joint fixpoint, and the result is intersected with the
    tables that actually exist in the schema, so that prose in a docstring
    cannot invent a table name.

WHY EACH TRIGGER IS MUTATION-TESTED
    A trigger that is never the reason a write fails contributes nothing and
    still reads as coverage. So for every derived ``(table, operation)`` the
    table below drops THAT ONE trigger from a real store and shows the foreign
    write then succeeds — and with it in place, fails. A trigger whose removal
    changes nothing fails this test.
"""

from __future__ import annotations

import ast
import os
import pathlib
import re
import sqlite3

import pytest

import hermes_state_common
from hermes_state_common import TURN_FENCE_FUNCTION_NAME
from tests.state.lease_mutation_harness import (
    Mutation,
    assert_every_pin_has_a_killer,
    assert_mutation_kills_the_pin,
)
from tests.state import test_turn_fence_adjunct_surface as adjunct
from tests.state import test_turn_lease_writer_census as census_mod

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)
#: The migration pin opens a real store and drives `tools/async_delegation`
#: through the shared fixture; take the whole tree rather than naming the
#: modules it happens to import today.
_EXTRA_EXTRACT = (".",)


def turn_fence_trigger_name(table: str, operation: str) -> str:
    """The trigger production names for one surface entry.

    Read from production when it exposes the helper. The fallback reproduces
    the name production already uses, so a tree that has not grown the helper
    yet is measured against its real triggers rather than failing to import —
    a collection error is not a result.
    """
    helper = getattr(hermes_state_common, "turn_fence_trigger_name", None)
    if helper is not None:
        return helper(table, operation)
    return f"hermes_turn_fence_{table}_{operation.lower()}"


def declared_fence_surface() -> tuple:
    """What production says the fence covers, as ``(table, operation)`` pairs.

    ``TURN_FENCE_SURFACE`` when production declares one; otherwise recovered
    from the trigger NAMES it does declare, which is the same claim in a less
    convenient form. Either way this returns production's real answer, so a
    difference from the derivation is a finding rather than an artefact.
    """
    declared = getattr(hermes_state_common, "TURN_FENCE_SURFACE", None)
    if declared is not None:
        return tuple(declared)
    recovered = []
    for name in getattr(hermes_state_common, "TURN_FENCE_TRIGGERS", ()):
        stem = name.removeprefix("hermes_turn_fence_")
        table, _, verb = stem.rpartition("_")
        if table and verb:
            recovered.append((table, verb.upper()))
    return tuple(recovered)


TURN_FENCE_SURFACE = declared_fence_surface()

#: Any statement that writes a table, and which table. The verb is captured so
#: the surface is per (table, operation) rather than per table: a fence that
#: covers INSERT and not DELETE is exactly the shape being ruled out.
WRITE_STATEMENT = re.compile(
    r"\b(insert(?:\s+or\s+\w+)?\s+into|update|delete\s+from)\s+"
    r"([A-Za-z_][A-Za-z_0-9]*)",
    re.IGNORECASE,
)

TURN_LEASE_TABLE = "session_turn_leases"

VERB_TO_OPERATION = {"insert": "INSERT", "update": "UPDATE", "delete": "DELETE"}


def _real_tables(tmp_path) -> frozenset:
    """Table names in a store this generation creates.

    The derivation reads string literals, and a docstring that says "update is
    …" parses as ``UPDATE is``. Intersecting with the live schema is what stops
    prose from inventing a table, and it is derived rather than filtered by a
    list of words to ignore.
    """
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "schema-probe.db")
    with db._read_ctx() as conn:
        names = {
            row["name"] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    db.close()
    return frozenset(names)


#: The production refusal a turn-lease admission raises. The seed is anchored
#: HERE and not on a table name: the question the surface answers is "what did
#: this generation decide it must ask permission before writing", and the raise
#: site IS that decision. A table can be renamed and a guard can move; the
#: exception is the thing that cannot change without the refusal changing.
TURN_LEASE_REFUSAL = "SessionTurnLeaseLostError"


def _raises_the_turn_lease_refusal(node: ast.AST) -> bool:
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Raise) or inner.exc is None:
            continue
        raised = inner.exc.func if isinstance(inner.exc, ast.Call) else inner.exc
        name = getattr(raised, "id", None) or getattr(raised, "attr", None)
        if name == TURN_LEASE_REFUSAL:
            return True
    return False


def _self_calls_handing_over_the_connection(node: ast.AST):
    """``self.<name>(<a name bound in this method>, …)`` — the transaction hop.

    The same shape the writer census uses for its rule 2, and for the same
    reason: the transaction's connection is a parameter, so a call that passes
    a bound name as its first positional argument is the call that puts the
    callee's statements inside this method's transaction.
    """
    bound = census_mod._names_bound_in(node)
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call):
            continue
        fn = inner.func
        if not (isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "self"):
            continue
        first = inner.args[0] if inner.args else None
        if isinstance(first, ast.Name) and first.id in bound:
            yield fn.attr


def admitted_transaction_closure(methods: dict) -> frozenset:
    """Methods whose transaction consults the canonical turn-lease admission.

    Rules 1-3 of HOW THE SURFACE IS DECIDED, run to a JOINT fixpoint. Running
    them to separate fixpoints is not the same answer: an entry point that
    hands its connection to a helper which is only pulled in by the inner rule
    (``skip_leased_on_connection`` -> ``_skip_leased_conversations``) is
    reachable only when the upward pass gets to run again afterwards.
    """
    covered = {name for name, (_module, node) in methods.items()
               if _raises_the_turn_lease_refusal(node)}
    assert covered, (
        f"no method raises {TURN_LEASE_REFUSAL}; the seed itself is broken and "
        f"every answer below would be vacuous"
    )
    changed = True
    while changed:
        changed = False
        for name, (_module, node) in methods.items():
            if name in covered:
                continue
            if set(_self_calls_handing_over_the_connection(node)) & covered:
                covered.add(name)
                changed = True
        grown = set()
        for name in sorted(covered):
            grown |= set(
                _self_calls_handing_over_the_connection(methods[name][1])
            )
        inner = (grown & set(methods)) - covered
        if inner:
            covered |= inner
            changed = True
    return frozenset(covered)


def _tables_written_in(node: ast.AST, real_tables: frozenset) -> set:
    found = set()
    for inner in ast.walk(node):
        if not (isinstance(inner, ast.Constant) and isinstance(inner.value, str)):
            continue
        for match in WRITE_STATEMENT.finditer(inner.value):
            table = match.group(2).lower()
            if table in real_tables:
                found.add(table)
    return found


def borrowed_admission_entry_points(methods: dict, closure) -> frozenset:
    """Closure members a FOREIGN module can call on its own transaction.

    A method whose first parameter after ``self`` is the connection is one that
    decides on the caller's transaction rather than on this store's — which is
    exactly what a module holding its own ``BEGIN IMMEDIATE`` has to call. The
    set is read off the signatures, so a new borrowed entry point is picked up
    without anybody adding it here.
    """
    entry_points = set()
    for name in closure:
        node = methods[name][1]
        args = [a.arg for a in node.args.args]
        if args and args[0] == "self":
            args = args[1:]
        if args and args[0] == "conn":
            entry_points.add(name)
    return frozenset(entry_points)


def modules_borrowing_the_admission(entry_points) -> dict:
    """``{relative path: {tables it writes}}`` for the borrowing modules.

    A production module outside the ``SessionDB`` implementation that calls one
    of *entry_points* is running the canonical decision inside its own
    transaction. Its DML is therefore in an admitted transaction as surely as
    ``SessionDB``'s is, and its tables belong on the surface.
    """
    owners = set(census_mod.SESSIONDB_IMPLEMENTATION_MODULES)
    borrowing = {}
    for rel, path in census_mod._production_files():
        if str(rel) in owners:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:  # pragma: no cover - a broken tree is not our finding
            continue
        borrows = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in entry_points
            and not (isinstance(node.func.value, ast.Name)
                     and node.func.value.id == "self")
            for node in ast.walk(tree)
        )
        if borrows:
            borrowing[str(rel)] = tree
    return borrowing


def derive_turn_fence_surface(tmp_path) -> frozenset:
    """``{(table, operation)}`` the fence has to cover, read off the source."""
    methods, _missing = census_mod._sessiondb_implementation(census_mod.REPO_ROOT)
    real_tables = frozenset(
        name.lower() for name in _real_tables(tmp_path)
    )

    closure = admitted_transaction_closure(methods)
    fenced_tables = set()
    for name in closure:
        fenced_tables |= _tables_written_in(methods[name][1], real_tables)

    entry_points = borrowed_admission_entry_points(methods, closure)
    for _rel, tree in modules_borrowing_the_admission(entry_points).items():
        fenced_tables |= _tables_written_in(tree, real_tables)

    # ALL THREE operations on every table the derivation reaches, not only the
    # ones this generation happens to perform. The surface is about what a
    # FOREIGN writer can do, and "no code path of ours deletes a lease row" is
    # not "a lease row cannot be deleted" — a fence keyed per operation is a
    # fence with the other doors open. Deriving tables and then covering the
    # operations also removes the only place a judgement call could live.
    return frozenset(
        (table, operation)
        for table in fenced_tables
        for operation in ("INSERT", "UPDATE", "DELETE")
    )


def test_the_measured_counterexample_at_cb61320a(tmp_path):
    """The exact review measurement, kept as a test rather than as prose.

    Two LIVE-OWNED sessions created by this generation — ``meta`` carrying a
    ``model_config``, and ``empty`` with no ``messages`` row at all — then one
    plain ``sqlite3.connect`` with no generation function registered, running
    two statements that never name ``messages``::

        UPDATE sessions SET model_config = '{"model": "FOREIGN"}' WHERE id = 'meta'
        DELETE FROM sessions WHERE id = 'empty'

    Measured at cb61320a5510cc7fb4cc8e3da3ebf9ac8aab6c2b::

        {"empty_survives": false,
         "meta_model_config_after": "{\\"model\\": \\"FOREIGN\\"}",
         "raw_empty_session_delete": "accepted",
         "raw_model_config_update": "accepted"}

    A live-owned provider-visible config mutation and a live-owned destructive
    lifecycle mutation, both accepted. The empty session is the load-bearing
    half: with no ``messages`` row there is nothing for a cascade or a message
    trigger to act on, so it removes the last defence a transcript-only fence
    has.

    Asserted on the four keys, and on the row VALUES rather than on an
    exception — "it raised" and "the row is unchanged" are different claims and
    only the second is the property.
    """
    import json

    from hermes_state import SessionDB

    store = tmp_path / "state.db"
    db = SessionDB(store)
    db.create_session("meta", source="test",
                      model_config={"model": "current"})
    db.create_session("empty", source="test")
    for sid in ("meta", "empty"):
        grant = db.try_acquire_session_turn_lease(
            sid, f"pid={os.getpid()}:turn=live-{sid}:platform=test",
            ttl_seconds=600,
        )
        assert grant, f"could not take the lease on {sid}"
    db.close()

    foreign = sqlite3.connect(str(store))
    outcome = {}
    for key, sql in (
        ("raw_model_config_update",
         "UPDATE sessions SET model_config = '{\"model\": \"FOREIGN\"}' "
         "WHERE id = 'meta'"),
        ("raw_empty_session_delete",
         "DELETE FROM sessions WHERE id = 'empty'"),
    ):
        try:
            foreign.execute(sql)
        except sqlite3.OperationalError as exc:
            outcome[key] = f"refused: {exc}"
        else:
            outcome[key] = "accepted"
    foreign.commit()
    foreign.close()

    reopened = SessionDB(store)
    with reopened._read_ctx() as conn:
        row = conn.execute(
            "SELECT model_config FROM sessions WHERE id = 'meta'"
        ).fetchone()
        outcome["meta_model_config_after"] = row["model_config"] if row else None
        outcome["empty_survives"] = conn.execute(
            "SELECT 1 FROM sessions WHERE id = 'empty'"
        ).fetchone() is not None
    reopened.close()

    assert outcome["empty_survives"] is True, (
        f"a foreign connection deleted a LIVE-OWNED empty session. It has no "
        f"`messages` row, so no message trigger and no cascade can act on it — "
        f"a transcript-only fence has nothing left to stop this.\n"
        f"{json.dumps(outcome, sort_keys=True)}"
    )
    assert json.loads(outcome["meta_model_config_after"] or "{}").get(
        "model"
    ) == "current", (
        f"a foreign connection rewrote the model_config of a LIVE-OWNED "
        f"session. `model_config` is what the next turn is dispatched under "
        f"and it never touches `messages`.\n"
        f"{json.dumps(outcome, sort_keys=True)}"
    )
    assert outcome["raw_model_config_update"].startswith("refused"), (
        f"the UPDATE was accepted: {json.dumps(outcome, sort_keys=True)}"
    )
    assert outcome["raw_empty_session_delete"].startswith("refused"), (
        f"the DELETE was accepted: {json.dumps(outcome, sort_keys=True)}"
    )


def test_the_declared_fence_surface_is_the_one_the_source_needs(tmp_path):
    """Production's declaration must equal the derivation, both ways.

    Missing entries are the hole this file was written for. EXTRA entries
    matter too: a declared trigger with nothing behind it is a claim the source
    does not support, and it makes the next reader believe a table is written
    where it is not.
    """
    derived = derive_turn_fence_surface(tmp_path)
    declared = frozenset(
        (table.lower(), op.upper()) for table, op in TURN_FENCE_SURFACE
    )
    assert derived, "the derivation found nothing; the scan itself is broken"
    missing = derived - declared
    extra = declared - derived
    assert not missing, (
        f"these (table, operation) pairs are written inside a transaction that "
        f"CONSULTS the canonical turn-lease admission — this generation decided "
        f"it had to ask permission before writing them — and no generation "
        f"trigger covers them, so an old binary performs them unrefused: "
        f"{sorted(missing)}"
    )
    assert not extra, (
        f"these triggers are declared but no derived mutator writes them: "
        f"{sorted(extra)}"
    )


def test_every_declared_trigger_exists_in_a_real_store(tmp_path):
    """The declaration has to reach the file, not just the constant."""
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    with db._read_ctx() as conn:
        installed = {
            row["name"] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }
    db.close()
    for table, op in TURN_FENCE_SURFACE:
        name = turn_fence_trigger_name(table, op)
        assert name in installed, (
            f"{name} is declared but was not created on a store this "
            f"generation opened; installed triggers: {sorted(installed)}"
        )


#: One foreign statement per surface entry, and the seed it needs. The statement
#: has to be one the schema would otherwise accept, or "it failed" proves
#: nothing about the trigger.
FOREIGN_STATEMENTS = {
    ("messages", "INSERT"):
        "INSERT INTO messages (session_id, role, content, timestamp, active) "
        "VALUES ('s', 'assistant', 'foreign', 1.0, 1)",
    ("messages", "UPDATE"):
        "UPDATE messages SET content = 'clobbered' WHERE session_id = 's'",
    ("messages", "DELETE"):
        "DELETE FROM messages WHERE session_id = 's'",
    ("sessions", "INSERT"):
        "INSERT INTO sessions (id, source, started_at) "
        "VALUES ('smuggled', 'foreign', 1.0)",
    ("sessions", "UPDATE"):
        "UPDATE sessions SET model = 'evil-model' WHERE id = 's'",
    ("sessions", "DELETE"):
        "DELETE FROM sessions WHERE id = 's'",
    (TURN_LEASE_TABLE, "INSERT"):
        "INSERT INTO session_turn_leases (conversation_id, holder, "
        "acquired_at, expires_at, epoch) VALUES ('smuggled', 'x', 0, 0, 1)",
    (TURN_LEASE_TABLE, "UPDATE"):
        "UPDATE session_turn_leases SET holder = '' WHERE conversation_id = 's'",
    (TURN_LEASE_TABLE, "DELETE"):
        "DELETE FROM session_turn_leases WHERE conversation_id = 's'",
    ("compression_locks", "INSERT"):
        "INSERT INTO compression_locks (session_id, holder, acquired_at, "
        "expires_at) VALUES ('smuggled', 'foreign', 1.0, 4102444800.0)",
    ("compression_locks", "UPDATE"):
        "UPDATE compression_locks SET holder = 'foreign' WHERE session_id = 's'",
    ("compression_locks", "DELETE"):
        "DELETE FROM compression_locks WHERE session_id = 's'",
    # The four adjunct tables, taken from the file that MEASURED them rather
    # than restated here. Two copies of a foreign statement is two things to
    # keep in step, and the copy that drifts is the one that stops proving
    # anything.
    **adjunct.ADJUNCT_STATEMENTS,
}


def _store_with_a_live_lease(tmp_path, name="state.db"):
    """A live-owned store carrying a row in EVERY table on the surface.

    Built by :func:`tests.state.test_turn_fence_adjunct_surface._live_owned_store`
    so the mutation table below runs against the same fixture the counterexample
    measurement runs against. An UPDATE or DELETE that matches no row is still
    REFUSED by a trigger (SQLite resolves the trigger program when it PREPARES
    the statement), so a fixture with empty tables would let a mutation row pass
    while proving nothing about rows — the drop leg would 'succeed' by changing
    nothing.
    """
    from hermes_state import SessionDB

    store, grant = adjunct._live_owned_store(tmp_path, name)
    db = SessionDB(store)
    try:
        # The grant is presented because taking the compression lock is
        # admitted: the fixture's compressor is the turn's compressor, and a
        # holderless acquire on a live-owned conversation is precisely what
        # tests/state/test_turn_lease_compression_lock_admission refuses.
        assert db.try_acquire_compression_lock(
            "s", "holder-1", ttl_seconds=600, turn_lease_holder=grant
        ), "the compression lock this fixture needs was not taken"
    finally:
        db.close()
    return store


def test_every_declared_trigger_is_the_reason_its_write_fails(tmp_path):
    """The mutation table. Drop one trigger; that one write gets through.

    A trigger something else already guarantees the property for is redundant
    and reports coverage it does not have. This is the check that can tell the
    difference: with the trigger installed the foreign write must be refused,
    and with THAT trigger and no other removed it must succeed.
    """
    missing_statements = {
        (t.lower(), o.upper()) for t, o in TURN_FENCE_SURFACE
    } - set(FOREIGN_STATEMENTS)
    assert not missing_statements, (
        f"these surface entries have no foreign statement to exercise them, "
        f"so their triggers are unproven: {sorted(missing_statements)}"
    )

    for index, (table, op) in enumerate(TURN_FENCE_SURFACE):
        key = (table.lower(), op.upper())
        sql = FOREIGN_STATEMENTS[key]
        trigger = turn_fence_trigger_name(table, op)

        store = _store_with_a_live_lease(tmp_path, f"intact-{index}.db")
        intact = sqlite3.connect(str(store))
        try:
            with pytest.raises(sqlite3.OperationalError) as caught:
                intact.execute(sql)
            assert TURN_FENCE_FUNCTION_NAME in str(caught.value), (
                f"{key} was refused, but not by {trigger}: {caught.value!r}"
            )
        finally:
            intact.rollback()
            intact.close()

        # Same store, same statement, that one trigger removed.
        store = _store_with_a_live_lease(tmp_path, f"mutated-{index}.db")
        surgeon = sqlite3.connect(str(store))
        surgeon.create_function(TURN_FENCE_FUNCTION_NAME, 0, lambda: 1)
        surgeon.execute(f"DROP TRIGGER {trigger}")
        surgeon.commit()
        surgeon.close()

        mutated = sqlite3.connect(str(store))
        try:
            mutated.execute("PRAGMA foreign_keys=OFF")
            mutated.execute(sql)
        except sqlite3.OperationalError as exc:
            pytest.fail(
                f"{trigger} was removed and {key} was STILL refused "
                f"({exc}). Something else already guarantees this write "
                f"cannot happen, so the trigger is reporting coverage it does "
                f"not have — or the drop did not take."
            )
        finally:
            mutated.rollback()
            mutated.close()


def test_no_production_module_writes_a_fenced_table_outside_sessiondb():
    """No SECOND MINTER: one place registers the marker, and it is SessionDB.

    The marker proves "current generation" and nothing else. It does not prove
    the conversation root, the holder, or the epoch, so it is not authority to
    mutate a live-owned conversation — and the barrier's real question was never
    "is this connection marked" but "who is allowed to mint the marker". A
    connection-local scalar is not a database boundary.

    So the fix for a module that this test names is NOT to call
    ``register_turn_fence_function`` on its connection. That mints a second
    admitted door around the token validator: it is the original defect
    (a foreign handle registers the same name and walks past the trigger)
    performed deliberately, by us, in production. The fix for the blind spot
    would carry the blind spot.

    There are exactly two closures for a module named here:

    * route the write through ``SessionDB``, so it runs on the canonical
      transaction that the token validator sits in
      (``plugins/platforms/a2a/adapter.py`` was closed this way); or
    * refuse deterministically while a live lease exists.

    "It registers the fence function" is not one of them.

    The writer census asserts SessionDB ownership for ``messages``. The barrier
    needs it for every table on the surface, so it is asserted for every table
    on the surface — derived from the same declaration the triggers come from,
    never from a list of modules someone keeps in step.
    """
    tables = {table.lower() for table, _op in TURN_FENCE_SURFACE}
    statement = re.compile(
        r"\b(?:insert(?:\s+or\s+\w+)?\s+into|update|delete\s+from)\s+"
        r"(" + "|".join(sorted(re.escape(t) for t in tables)) + r")\b",
        re.IGNORECASE,
    )
    owners = set(census_mod.SESSIONDB_IMPLEMENTATION_MODULES)
    offenders = []
    for rel, path in census_mod._production_files():
        if str(rel) in owners:
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc is not None:
                    docstrings.add(doc)
        # Deliberately NO escape hatch for "the module registers the marker".
        # Registering is the trap this check exists to keep out of the tree, so
        # a module cannot answer this test by minting the marker — only by
        # routing through SessionDB or by refusing while a lease is live.
        # Does the module open its own connections at all? A module that only
        # ever writes on a connection handed to it (SessionDB's
        # `_execute_write(lambda conn: ...)` shape) has nothing to register:
        # registration belongs where the connection is opened.
        opens_its_own = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "connect"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sqlite3"
            for node in ast.walk(tree)
        )
        if not opens_its_own:
            continue
        # A write that runs inside a callback the module hands to
        # SessionDB._execute_write is on SESSIONDB's connection, whatever else
        # the module opens elsewhere. gateway/platforms/api_server.py is this
        # shape: it opens response_store.db on one path, and writes `sessions`
        # from an `_atomic(conn)` closure passed to `db._execute_write`.
        # Recovery is NOT this shape — it opens the destination itself and
        # passes that handle down — which is why the carve-out is keyed on the
        # callback reaching _execute_write rather than on "the receiver is a
        # parameter".
        canonical = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_execute_write"):
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        canonical.add(arg.id)
        canonical_spans = [
            (n.lineno, n.end_lineno)
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name in canonical
        ]
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if node.value in docstrings:
                continue
            found = statement.search(node.value)
            if not found:
                continue
            if any(lo <= node.lineno <= hi for lo, hi in canonical_spans):
                continue
            offenders.append(
                f"{rel}:{node.lineno}: writes {found.group(1)} on a connection "
                f"this module opened, outside any SessionDB transaction"
            )
    assert not offenders, (
        "these production modules open their own SQLite connections and write a "
        "fenced table outside any SessionDB transaction. Each one needs ONE of "
        "the two closures — route the write through SessionDB, or refuse "
        "deterministically while a live lease exists. Calling "
        "register_turn_fence_function on the connection is NOT a closure: it "
        "mints a second admitted door around the token validator, which is the "
        "defect this barrier exists to stop, performed on purpose.\n  "
        + "\n  ".join(sorted(set(offenders)))
    )


def test_this_generation_writes_every_surface_table_normally(tmp_path):
    """The fence must not be a fence on us, on any of its tables."""
    from hermes_state import SessionDB

    store = tmp_path / "state.db"
    db = SessionDB(store)
    db.create_session("s", source="test")
    grant = db.try_acquire_session_turn_lease(
        "s", f"pid={os.getpid()}:turn=live:platform=test", ttl_seconds=600
    )
    assert grant
    db.append_message(
        session_id="s", role="user", content="one", turn_lease_holder=grant
    )
    # update_session_model presents the grant for the same reason
    # append_message above and delete_session below do: it writes `model`,
    # `model_config`, `system_prompt` and `system_prompt_hash`, which are what
    # the next turn replays under, so it now takes the turn-lease admission
    # too. Leaving it holderless here would assert that the model switch is
    # exempt from the fence this file exists to check, and it is the write the
    # surface docstring names first.
    #
    # end_session presents the grant now too: it moves `ended_at` /
    # `end_reason`, and `end_reason = 'compression'` is the value
    # _check_transcript_write_guards enforces against the APPENDER — so an
    # unfenced writer of it closes the transcript against the holder of a
    # still-valid grant. See tests/state/test_turn_lease_session_lifecycle.
    #
    # set_session_title presents it too: the title pair shares the list-flag
    # admission helper, and `archived` in that family is what prune reads as
    # a do-not-collect marker. See tests/state/test_turn_lease_broad_writer_closure.
    db.update_session_model("s", "a-model", turn_lease_holder=grant)
    db.set_session_title("s", "a title", turn_lease_holder=grant)
    db.end_session("s", "completed", turn_lease_holder=grant)
    assert db.refresh_session_turn_lease("s", grant, ttl_seconds=600) is True
    db.release_session_turn_lease("s", grant)
    regrant = db.try_acquire_session_turn_lease(
        "s", f"pid={os.getpid()}:turn=again:platform=test", ttl_seconds=600
    )
    assert regrant is not None and regrant.epoch > grant.epoch
    db.delete_session("s", turn_lease_holder=regrant)
    db.close()


def test_the_surface_covers_sessions_not_only_messages(tmp_path):
    """A named regression for the narrowness this file was opened for.

    Stated separately from the derivation check so that a derivation which
    silently stops finding `sessions` writers fails with the reason rather than
    with a set difference.
    """
    declared = {(t.lower(), o.upper()) for t, o in TURN_FENCE_SURFACE}
    for op in ("INSERT", "UPDATE", "DELETE"):
        assert ("sessions", op) in declared, (
            f"sessions {op} is unfenced. The model, the system prompt, the "
            f"title and the end state all live in `sessions` and none of them "
            f"touch `messages`, so a transcript-only fence lets a mixed-version "
            f"writer rewrite what the next turn replays under."
        )
    for op in ("INSERT", "UPDATE", "DELETE"):
        assert ("messages", op) in declared


#: The pre-scope shape of ``gateway_routing`` (#59203) and the pre-``task``
#: shape of ``session_model_usage`` (#73823). Both are rebuilt on open by a
#: heal that SQLite cannot express as an ALTER, and both tables are now on the
#: fence surface — so the migration has to run inside the barrier and the
#: rebuilt table has to come out carrying it.
LEGACY_GATEWAY_ROUTING_SQL = """
CREATE TABLE gateway_routing (
    session_key TEXT PRIMARY KEY,
    entry_json TEXT NOT NULL,
    updated_at REAL NOT NULL
)
"""

LEGACY_SESSION_MODEL_USAGE_SQL = """
CREATE TABLE session_model_usage (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    billing_provider TEXT NOT NULL DEFAULT '',
    billing_base_url TEXT NOT NULL DEFAULT '',
    billing_mode TEXT NOT NULL DEFAULT '',
    api_call_count INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0,
    actual_cost_usd REAL NOT NULL DEFAULT 0,
    cost_status TEXT,
    cost_source TEXT,
    first_seen REAL,
    last_seen REAL,
    PRIMARY KEY (session_id, model, billing_provider, billing_base_url,
                 billing_mode)
)
"""


def _regress_to_a_legacy_store(store):
    """Put a fenced store back into the shapes the open-time heals repair.

    The fence triggers are dropped FIRST and by name from the declaration.
    Dropping them is what makes this a legacy store rather than a current one
    with old tables, and doing it without registering the generation marker is
    deliberate: a test that mints the marker to build its fixture is a test
    that has quietly proved the marker is mintable.
    """
    conn = sqlite3.connect(str(store), isolation_level=None)
    try:
        for name in hermes_state_common.TURN_FENCE_TRIGGERS:
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.execute("DROP TABLE gateway_routing")
        conn.executescript(LEGACY_GATEWAY_ROUTING_SQL)
        conn.execute(
            "INSERT INTO gateway_routing (session_key, entry_json, updated_at) "
            "VALUES ('legacy-key', '{\"session_id\": \"s\"}', 1.0)"
        )
        conn.execute("DROP TABLE session_model_usage")
        conn.executescript(LEGACY_SESSION_MODEL_USAGE_SQL)
        conn.execute(
            "INSERT INTO session_model_usage "
            "(session_id, model, input_tokens, output_tokens) "
            "VALUES ('s', 'legacy-model', 41, 1)"
        )
        conn.execute(
            "INSERT INTO async_delegations "
            "(delegation_id, origin_session, origin_ui_session_id, "
            " parent_session_id, state, dispatched_at, updated_at, "
            " delivery_state, delivery_attempts) "
            "VALUES ('legacy-d', 's', 's', 's', 'running', 1.0, 1.0, "
            "'pending', 0)"
        )
    finally:
        conn.close()


def check_a_legacy_store_migrates_under_the_fence_and_comes_out_fenced(
    tmp_path,
) -> None:
    """The other direction of the downgrade contract, and it is not symmetric.

    The rollback side proves an old binary is REFUSED. This side proves the
    current binary is not — that widening the surface to the tables the
    open-time heals rebuild did not turn a migration into a store that cannot
    open. That is a live risk rather than a formality: ``_heal_gateway_routing_pk``
    and ``_heal_session_model_usage_pk`` rebuild a table by RENAME + CREATE +
    INSERT ... SELECT, so every one of those statements now runs against a
    fenced table, and the CREATE produces a table with no trigger on it.

    Two things are therefore asserted, and the second is the one that would
    catch a reordering:

    * the rows survive the migration — the heal's own ``INSERT ... SELECT``
      is admitted, so the store is not silently emptied by the barrier;
    * every declared trigger is installed on the migrated store. The fence DDL
      runs LAST in ``_init_schema`` precisely so a rebuilt table is covered in
      the same open. Move it earlier and the store comes out of its migration
      with ``gateway_routing`` and ``session_model_usage`` unfenced until some
      later reopen, which is a hole nothing else here would notice.
    """
    from hermes_state import SessionDB

    store = _store_with_a_live_lease(tmp_path)
    _regress_to_a_legacy_store(store)

    migrated = SessionDB(store)
    try:
        with migrated._read_ctx() as conn:
            routing = conn.execute(
                "SELECT scope, session_key, entry_json FROM gateway_routing "
                "ORDER BY session_key"
            ).fetchall()
            usage = conn.execute(
                "SELECT session_id, model, input_tokens, task "
                "FROM session_model_usage ORDER BY model"
            ).fetchall()
            delegation = conn.execute(
                "SELECT delegation_id, origin_session_id FROM async_delegations "
                "WHERE delegation_id = 'legacy-d'"
            ).fetchone()
            installed = {
                row["name"] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                ).fetchall()
            }
    finally:
        migrated.close()

    assert [(r["scope"], r["session_key"]) for r in routing] == [
        ("", "legacy-key")
    ], (
        f"the gateway_routing rebuild lost its rows under the fence: "
        f"{[tuple(r) for r in routing]}"
    )
    assert [(r["model"], r["input_tokens"], r["task"]) for r in usage] == [
        ("legacy-model", 41, "")
    ], (
        f"the session_model_usage rebuild lost its rows under the fence: "
        f"{[tuple(r) for r in usage]}"
    )
    assert delegation is not None and delegation["origin_session_id"] == "", (
        "the reconciler did not add async_delegations.origin_session_id, which "
        "moved into SCHEMA_SQL when tools/async_delegation stopped opening a "
        "connection it could run DDL on"
    )

    missing = [
        turn_fence_trigger_name(table, op)
        for table, op in TURN_FENCE_SURFACE
        if turn_fence_trigger_name(table, op) not in installed
    ]
    assert not missing, (
        f"these triggers are not on the MIGRATED store: {missing}. A table an "
        f"open-time heal rebuilt comes out of the rebuild with no trigger on "
        f"it, so the fence DDL has to run after every heal — and this is the "
        f"only place that would notice it moving."
    )


#: The one property in this file that is not already its own mutation table.
#: Everything else here either DERIVES an answer and compares it to production
#: (a derivation that stops deriving fails on its own), or drops a trigger and
#: requires the write through — ``test_every_declared_trigger_is_the_reason_its_
#: write_fails`` is twenty-four mutation rows in a loop. The migration property
#: is different: it depends on an ORDERING inside ``_init_schema`` that nothing
#: else observes, so it needs a row that moves that ordering.
PINS = {
    "check_a_legacy_store_migrates_under_the_fence_and_comes_out_fenced":
        check_a_legacy_store_migrates_under_the_fence_and_comes_out_fenced,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_turn_fence_surface_property(name, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    PINS[name](tmp_path / "work")


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_a_legacy_store_migrates_under_the_fence_and_comes_out_fenced",
        module="hermes_state_schema.py",
        find="        cursor.executescript(TURN_FENCE_TRIGGER_SQL)\n",
        replace="",
        why="_init_schema's last statement is what puts the barrier on the "
            "store, and a MIGRATED store is where that matters most: "
            "_heal_gateway_routing_pk and _heal_session_model_usage_pk rebuild "
            "a table by RENAME + CREATE + INSERT...SELECT, and the CREATE "
            "produces a table with no trigger on it. Removed, the pin fails "
            "naming every missing trigger.\n"
            "WHAT THIS ROW DOES NOT DISTINGUISH, stated so nobody reads more "
            "into it: it removes the DDL rather than moving it. A single "
            "content-keyed substitution cannot relocate a statement, and "
            "`CREATE TRIGGER IF NOT EXISTS` at the end is idempotent, so "
            "installing it EARLIER as well changes nothing and cannot be the "
            "row. The ordering half was measured directly instead — moving the "
            "executescript in front of the two heals makes this same pin fail "
            "with exactly six triggers missing, the three on gateway_routing "
            "and the three on session_model_usage.",
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
