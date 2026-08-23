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

HOW THE SURFACE IS DECIDED
    Not by a list in production that somebody keeps in step. Production
    DECLARES :data:`hermes_state_common.TURN_FENCE_SURFACE`; this file DERIVES
    the same set from the source of ``SessionDB`` and its mixins, and fails when
    they differ. A new mutator that writes a new table therefore fails here
    until the declaration follows it.

    The derivation is: every ``(table, operation)`` written inside the write
    transaction of a method that either

    * writes model context (:func:`derive_context_bearing_mutators`, the same
      derivation the writer census uses), or
    * writes the turn-lease table itself — a process that can free the fence
      can defeat it, so the fence's own table is part of the surface.

    minus the one argued exemption (``_init_schema``, which installs the
    triggers and cannot be fenced by them), and intersected with the tables that
    actually exist in the schema so that prose in a docstring cannot invent a
    table name.

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
from tests.state import test_turn_lease_writer_census as census_mod


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


def derive_turn_fence_surface(tmp_path) -> frozenset:
    """``{(table, operation)}`` the fence has to cover, read off the source."""
    methods, _missing = census_mod._sessiondb_implementation(census_mod.REPO_ROOT)
    context_bearing = set(census_mod.derive_context_bearing_mutators())

    lease_writers = set()
    for name, (_module, node) in methods.items():
        for inner in ast.walk(node):
            if not (isinstance(inner, ast.Constant)
                    and isinstance(inner.value, str)):
                continue
            for match in WRITE_STATEMENT.finditer(inner.value):
                if match.group(2).lower() == TURN_LEASE_TABLE:
                    lease_writers.add(name)

    exempt = set(census_mod.NOT_CONTEXT_BEARING)
    tables = {name.lower() for name in _real_tables(tmp_path)}
    fenced_tables = set()
    for name in (context_bearing | lease_writers) - exempt:
        entry = methods.get(name)
        if entry is None:
            continue
        for inner in ast.walk(entry[1]):
            if not (isinstance(inner, ast.Constant)
                    and isinstance(inner.value, str)):
                continue
            for match in WRITE_STATEMENT.finditer(inner.value):
                table = match.group(2).lower()
                if table in tables:
                    fenced_tables.add(table)
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
        f"these (table, operation) pairs are written inside the write "
        f"transaction of a context-bearing or lease-table mutator and no "
        f"generation trigger covers them, so an old binary performs them "
        f"unrefused: {sorted(missing)}"
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
}


def _store_with_a_live_lease(tmp_path, name="state.db"):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / name)
    db.create_session("s", source="test")
    grant = db.try_acquire_session_turn_lease(
        "s", f"pid={os.getpid()}:turn=live:platform=test", ttl_seconds=600
    )
    assert grant
    db.append_message(
        session_id="s", role="user", content="current", turn_lease_holder=grant
    )
    db.close()
    return tmp_path / name


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
