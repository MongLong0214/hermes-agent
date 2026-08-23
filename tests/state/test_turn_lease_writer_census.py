"""Slice 4 — the census counts PRODUCTION CALL SITES, and names its denominator.

Why this file replaces the previous census outright.

The earlier check was called "every writer is classified". What it actually
enumerated was *SessionDB methods whose body contains message-table SQL*. Three
separate failures came out of that one substitution:

1. **The mechanism was verified and the list was not.** "Fenced or on a list"
   passes trivially by growing the list. Two entries on it (destructive session
   deletion; reaction write and consume) could remove or consume model context.
2. **The denominator was never stated, so it was never checked.** It silently
   excluded every *caller*. A call flow that reads state, mutates holderlessly
   and then rewrites memory is invisible to a scan of ``SessionDB``.
3. **The package set was wrong and the name hid it.** ``gateway/`` and
   ``tui_gateway/`` are two different packages that both exist; a rebuild
   covered the first and reported coverage under a name that claimed both.

So the denominator is written down here, in the check, rather than implied by
what the scan happens to reach.

THE DENOMINATOR
    Every call, in non-test Python under the repository root, to one of the
    mutators :func:`derive_context_bearing_mutators` reads out of the
    ``SessionDB`` implementation, on any receiver.

MEMBERSHIP IS READ OFF THE CODE, NOT TYPED HERE
    The first version of this file made membership a ``frozenset`` literal, and
    every stage filtered on it before doing anything else. A mutator nobody had
    typed into the literal was therefore outside the denominator, outside the
    derived sweep set, and its production call sites were not counted — so
    "every production transcript writer goes through the fence" meant "every
    transcript writer someone remembered". Review demonstrated it on a copy of
    that tree with a method that discards the admission helper's answer and then
    ``DELETE``\\s the transcript: the file reported 5 passed.

    So membership is now a property of ``hermes_state.py`` and its mixins:
    a method is a context-bearing mutator when its own WRITE TRANSACTION
    contains message-table SQL. "Its own write transaction" is the whole of the
    rule and it is what keeps the set from collapsing into "every method":

    * the method's body (its nested ``_do(conn)`` closure included) contains a
      string that writes ``messages`` or ``message_reactions``; or
    * it hands the transaction's connection to a helper that is already in the
      set — ``self._insert_message_rows(conn, …)``. Passing ``conn`` is what
      makes the helper's SQL part of THIS transaction, and it is the difference
      between a real extension of the write and the plain ``self._execute_write``
      / ``self._init_schema`` calls that every method makes. Following those
      instead puts ``__init__`` in the set, and then everything.

    :mod:`tests.state.test_turn_lease_census_derivation` is the test for this:
    it splices review's method into the real ``hermes_state.py`` source and
    requires the derivation to find it and the scan to reach its caller.

WHY "ON ANY RECEIVER" NEEDED A DISCRIMINATOR
    Matching the attribute name on any receiver is what closes the aliasing
    hole (``self._db``, ``store.db``, a handle passed in as ``db``). It also
    catches an unrelated class that happens to share a method name, and it did:
    ``plugins/platforms/telegram/adapter.py`` calls
    ``self._bot.set_message_reaction(chat_id=…, message_id=…, reaction=…)`` —
    python-telegram-bot's HTTP API, in a module that never touches
    ``SessionDB`` at all. Counting those two as unfenced transcript writers made
    the denominator wrong in the other direction.

    A call leaves the denominator only on TWO independent proofs, both required
    (:func:`_provably_not_sessiondb`):

    1. it passes a keyword argument that the real ``SessionDB`` method has no
       parameter for, so the call would ``TypeError`` if it ever reached it —
       read off ``SessionDB`` with :mod:`inspect`, not written down here; and
    2. the module never obtains a ``SessionDB`` at all (no import, no mention).

    Neither can be used to hide a real writer: a real writer's keywords are all
    valid, and a module that writes the transcript has to get a handle from
    somewhere. Excluded calls are counted and printed in the failure report, so
    they are visible rather than silently dropped.

WHY THAT IS THE WHOLE SET
    Model context lives in exactly one place: the ``messages`` rows of a
    session, plus the ``sessions`` rows that own them. Those rows are reachable
    only through ``SessionDB``'s mutators — the class owns the sole connection
    and no production module opens its own handle to write them (asserted by
    :func:`test_no_production_module_writes_messages_outside_sessiondb`).
    So a call to one of those names is necessary and sufficient to change what
    a later turn replays, and enumerating the calls enumerates the risk.

WHAT COUNTS AS GOING THROUGH THE FENCE
    Four forms, and every one of them is read off the code:

    grant   the call passes ``turn_lease_holder=``.
    scope   the call is lexically inside a ``with ... session_turn_lease(...)``
            / ``turn_lease_scope(...)`` block that binds the grant it presents.
    sweep   the mutator does its own per-row admission inside its write
            transaction (see below).
    inner   the call is inside a ``SessionDB`` implementation module, lexically
            within another derived mutator. It is not a separate production
            entry point: the enclosing method extends its transaction into the
            callee, so the risk belongs to the enclosing method — which the
            derivation guarantees is itself in the denominator, with its own
            call sites counted here. ``self._insert_message_rows(conn, …)``
            inside ``append_messages_batch`` is the shape.

    Anything else is an unfenced production writer, and this test names it.

THE ONE PREDICATE CHANGE, AND WHY IT IS NOT AN EXEMPTION
    The third form was added when the sweeps were reached, and it is the place
    to attack this file. The argument for it: ``prune_sessions`` picks its
    victims from a filter, ``delete_empty_sessions`` from a table scan,
    ``purge_stale_tool_call_markers`` from every session in the store. Nobody —
    not the operator, not the scheduler — can name those conversations in
    advance, so there is no grant a caller could have held and no call site at
    which to hold it. Requiring one would leave exactly two outcomes: refuse
    every sweep while any conversation anywhere is busy, or run sweeps outside
    the fence. The admission therefore moves inside the sweep, per row, in the
    same transaction as the delete.

    What keeps it from being an exemption in disguise:

    * the set is DERIVED from the implementation (:func:`_self_fencing_sweeps`)
      rather than listed here, so it cannot be grown by editing this file;
    * the marker it derives from is the call to the real admission helper, and
      :func:`test_every_self_fencing_sweep_actually_skips_an_owned_conversation`
      proves BEHAVIOURALLY that each derived member leaves an owned
      conversation alone — a decorative call to the helper fails that test;
    * membership is still restricted to the derived mutator set.

    A single-target mutator can never qualify: it has one conversation, so it
    has a call site that can hold a grant, and every one of them does.

EXEMPTIONS ARE ARGUED, NOT ASSUMED
    :data:`NOT_CONTEXT_BEARING` holds the residue the derivation finds and the
    fence forms do not cover. Every entry has to answer one question — *can this
    write change what the provider will see on the next turn, remove it, or
    consume it?* — in its comment, AND have a test that proves the answer.
    ``_init_schema`` is the only one, and
    :func:`test_the_only_exemption_cannot_touch_a_row_this_generation_wrote`
    is its evidence.
"""

from __future__ import annotations

import ast
import os
import re
import time
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: The modules that implement ``SessionDB``: the class itself plus the mixins it
#: inherits. Located by parsing ``class SessionDB(...)`` rather than listed, and
#: :func:`test_the_derivation_reads_every_module_sessiondb_is_made_of` asserts
#: that every base named there was actually found in one of them.
SESSIONDB_IMPLEMENTATION_MODULES = (
    "hermes_state.py",
    "hermes_state_schema.py",
    "hermes_state_search.py",
    "hermes_state_portability.py",
)

#: A statement that writes model context. ``\b`` after the table name is what
#: keeps ``messages_fts`` — the search shadow table, rebuilt constantly and
#: carrying no context of its own — out of the set.
MESSAGE_TABLE_WRITE = re.compile(
    r"\b(?:insert(?:\s+or\s+\w+)?\s+into|update|delete\s+from)\s+"
    r"(?:messages|message_reactions)\b",
    re.IGNORECASE,
)

#: The residue: derived mutators whose call sites no fence form covers, with the
#: question answered per entry. See EXEMPTIONS ARE ARGUED, NOT ASSUMED.
NOT_CONTEXT_BEARING: dict = {
    # `UPDATE messages SET active = 1 WHERE active IS NULL`, run on every open.
    #
    # Can it change what the provider sees next?  Only by making rows visible
    # that a *previous generation* of the writer hid by accident (#51646: an
    # older reconciler added `active` without its NOT NULL DEFAULT 1, so INSERTs
    # that omitted the column wrote NULL and every `WHERE active = 1` transcript
    # loader dropped the row). It cannot remove or consume anything — the
    # predicate is `active IS NULL` and the direction is NULL → 1.
    #
    # Can it reach a row this generation wrote?  No: every INSERT now sets
    # active explicitly, so `active IS NULL` matches nothing this binary
    # produced. That is the part that is asserted rather than argued —
    # test_the_only_exemption_cannot_touch_a_row_this_generation_wrote.
    #
    # And its two call sites are `__init__` and `_reconnect_after_notadb`,
    # i.e. connection setup: there is no caller holding a handle yet, so there
    # is no call site at which a grant could be presented.
    "_init_schema": (
        "schema repair; heals `active IS NULL` rows left by a previous "
        "generation and cannot reach a row this one wrote"
    ),
}


def _sessiondb_implementation(root: pathlib.Path):
    """Return ``(methods, missing_bases)`` for the ``SessionDB`` under *root*.

    ``methods`` maps a method name to ``(module_relative_name, ast node)`` for
    ``class SessionDB`` and every base class named in its declaration that can
    be found in :data:`SESSIONDB_IMPLEMENTATION_MODULES`. ``missing_bases`` is
    the bases that could not be found — reported rather than raised, so the
    derivation still runs on a partial tree (the fixture in
    ``test_turn_lease_census_derivation`` is one), and asserted empty against
    the real tree by
    :func:`test_the_derivation_reads_every_module_sessiondb_is_made_of`.
    """
    trees = {}
    for name in SESSIONDB_IMPLEMENTATION_MODULES:
        path = root / name
        if not path.is_file():
            continue
        try:
            trees[name] = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:  # pragma: no cover - a broken tree is not our finding
            continue

    def _classes(want):
        for module, tree in trees.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == want:
                    yield module, node

    owner = list(_classes("SessionDB"))
    assert owner, f"no `class SessionDB` under {root}"
    wanted = ["SessionDB"]
    for _module, node in owner:
        for base in node.bases:
            name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
            if name:
                wanted.append(name)

    methods, missing = {}, []
    for want in wanted:
        found = list(_classes(want))
        if not found:
            missing.append(want)
            continue
        for module, node in found:
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.setdefault(member.name, (module, member))
    return methods, missing


def _writes_message_rows(node: ast.AST) -> bool:
    """True when a string literal anywhere under *node* writes model context."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
            if MESSAGE_TABLE_WRITE.search(inner.value):
                return True
    return False


def _names_bound_in(method: ast.AST) -> set:
    """Every parameter name in scope anywhere inside *method*.

    The transaction connection is a parameter — of ``_do(conn)``, or of the
    method itself for the ``(self, conn, …)`` helpers. Collecting all of them
    over-approximates in the safe direction: a call that passes a parameter as
    its first positional argument might not be passing the connection, and
    counting it anyway can only make the denominator larger.
    """
    names = set()
    for inner in ast.walk(method):
        if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = inner.args
            for group in (args.posonlyargs, args.args, args.kwonlyargs):
                names.update(a.arg for a in group)
            for extra in (args.vararg, args.kwarg):
                if extra is not None:
                    names.add(extra.arg)
    return names


def derive_context_bearing_mutators(root: pathlib.Path = REPO_ROOT) -> frozenset:
    """Methods whose own write transaction writes model context, read off *root*.

    See MEMBERSHIP IS READ OFF THE CODE in the module docstring. Two rules,
    applied to a fixpoint:

    1. the method's body contains message-table write SQL; or
    2. it calls ``self.<already in the set>(<a parameter>, …)`` — handing the
       transaction's connection to a helper, which is what puts that helper's
       SQL in this method's transaction.

    Rule 2 is deliberately NOT "calls something that can eventually write":
    ``self._execute_write`` reaches ``_init_schema`` through the reconnect path,
    so following plain calls puts every writer, ``__init__`` and ``close`` in
    the set, and a denominator containing everything measures nothing.
    """
    methods, _missing = _sessiondb_implementation(root)
    derived = {name for name, (_m, node) in methods.items()
               if _writes_message_rows(node)}
    changed = True
    while changed:
        changed = False
        for name, (_module, node) in methods.items():
            if name in derived:
                continue
            bound = _names_bound_in(node)
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Call):
                    continue
                fn = inner.func
                if not (isinstance(fn, ast.Attribute)
                        and isinstance(fn.value, ast.Name)
                        and fn.value.id == "self"
                        and fn.attr in derived):
                    continue
                first = inner.args[0] if inner.args else None
                if isinstance(first, ast.Name) and first.id in bound:
                    derived.add(name)
                    changed = True
                    break
    return frozenset(derived)

#: Trees that are not shipped production Python.
EXCLUDED_DIRS = frozenset({
    "tests", "tests-js", "evals", "node_modules", ".git", "web", "website",
    "docs", "native", "ui-tui", "contributors", "locales", "assets",
})

FENCE_HELPERS = ("session_turn_lease", "turn_lease_scope")


def _production_files(root: pathlib.Path = REPO_ROOT):
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if set(rel.parts) & EXCLUDED_DIRS:
            continue
        if rel.name.startswith("test_") or rel.name == "conftest.py":
            continue
        yield rel, path


def _fenced_line_ranges(tree: ast.AST) -> list[tuple[int, int]]:
    """Line spans of `with ...session_turn_lease(...)/turn_lease_scope(...)`."""
    spans = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            fn = call.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in FENCE_HELPERS:
                spans.append((node.lineno, node.end_lineno))
    return spans


#: The marker the sweep form is derived from: the in-transaction, per-row
#: admission helper. Naming the helper rather than the methods is the point —
#: a name added to a list here would prove nothing.
SWEEP_ADMISSION_HELPER = "_skip_leased_conversations"


def _self_fencing_sweeps(root: pathlib.Path = REPO_ROOT) -> frozenset:
    """Derived mutators that admit per row inside their own write transaction.

    Derived twice over, and never listed. A method qualifies by being a
    context-bearing mutator at all (:func:`derive_context_bearing_mutators` —
    which no longer consults a name anybody typed) and by calling
    :data:`SWEEP_ADMISSION_HELPER` somewhere in its body, including the nested
    ``_do(conn)`` closure that runs in the write transaction.
    """
    methods, _missing = _sessiondb_implementation(root)
    mutators = derive_context_bearing_mutators(root)
    found = set()
    for name in mutators:
        entry = methods.get(name)
        if entry is None:
            continue
        for inner in ast.walk(entry[1]):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == SWEEP_ADMISSION_HELPER
            ):
                found.add(name)
                break
    return frozenset(found)


def _mutator_call_sites_inside_the_implementation(root: pathlib.Path = REPO_ROOT):
    """``{(module, lineno)}`` for mutator calls nested in another mutator.

    The ``inner`` fence form. See WHAT COUNTS AS GOING THROUGH THE FENCE: such
    a call is the enclosing mutator's own transaction reaching further, not a
    new production entry point, and the enclosing mutator is in the denominator
    by construction — rule 2 of the derivation is what put it there.
    """
    methods, _missing = _sessiondb_implementation(root)
    mutators = derive_context_bearing_mutators(root)
    inner_sites = set()
    for name in mutators:
        entry = methods.get(name)
        if entry is None:
            continue
        module, node = entry
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in mutators
            ):
                inner_sites.add((module, call.lineno))
    return inner_sites


def _mutator_parameters(root: pathlib.Path = REPO_ROOT) -> dict:
    """Parameter names of each derived mutator, read off ``SessionDB`` itself."""
    import inspect

    from hermes_state import SessionDB

    params = {}
    for name in derive_context_bearing_mutators(root):
        fn = getattr(SessionDB, name, None)
        if fn is None:
            continue
        try:
            params[name] = set(inspect.signature(fn).parameters)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            continue
    return params


def _provably_not_sessiondb(node: ast.Call, mutator: str, source: str, params) -> str:
    """Reason this call CANNOT be the SessionDB method, or '' when it could be.

    Both proofs are required. See the module docstring.
    """
    known = params.get(mutator)
    if known is None:
        return ""
    passed = {k.arg for k in node.keywords if k.arg}
    impossible = passed - known
    if not impossible:
        return ""
    if "SessionDB" in source or "session_db" in source or "hermes_state" in source:
        # The module CAN get a handle, so a surprising keyword is a reason to
        # look, not a reason to stop counting it.
        return ""
    return f"passes {sorted(impossible)}, which SessionDB.{mutator} has no parameter for"


def census(root: pathlib.Path = REPO_ROOT) -> list[dict]:
    """Every production call site in the denominator, with its fence status."""
    mutators = derive_context_bearing_mutators(root)
    sweeps = _self_fencing_sweeps(root)
    inner_sites = _mutator_call_sites_inside_the_implementation(root)
    params = _mutator_parameters(root)
    sites = []
    for rel, path in _production_files(root):
        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        fenced_spans = _fenced_line_ranges(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            # Attribute calls only: a bare `append_message(messages, ...)` is
            # the in-memory list helper in agent/, not a durable write.
            if not isinstance(fn, ast.Attribute):
                continue
            if fn.attr not in mutators:
                continue
            excluded = _provably_not_sessiondb(node, fn.attr, source, params)
            carries = any(k.arg == "turn_lease_holder" for k in node.keywords)
            inside = any(lo <= node.lineno <= hi for lo, hi in fenced_spans)
            sweep = fn.attr in sweeps
            # Keyed on the module's path relative to the root, so a same-named
            # file in a package cannot inherit an implementation module's
            # `inner` classification.
            inner = (str(rel), node.lineno) in inner_sites
            sites.append({
                "location": f"{rel}:{node.lineno}",
                "package": rel.parts[0] if len(rel.parts) > 1 else "(root)",
                "mutator": fn.attr,
                "excluded": excluded,
                "fenced": carries or inside or sweep or inner,
                "how": (
                    "grant" if carries
                    else "scope" if inside
                    else "sweep" if sweep
                    else "inner" if inner
                    else ""
                ),
            })
    return sites


def _seed_owned_and_free(db, tmp_path):
    """Two ended, empty, old sessions; a live turn owns the first one.

    Both are sweep candidates by every filter the sweeps use. The only thing
    that differs is who owns the conversation.
    """
    old = time.time() - 400 * 86400
    for sid in ("owned", "free"):
        db.create_session(sid, source="test")
        db.end_session(sid, "completed")
        with db._lock:
            db._conn.execute(
                "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
                (old, old, sid),
            )
            db._conn.commit()
    grant = db.try_acquire_session_turn_lease(
        "owned", f"pid={os.getpid()}:turn=census-proof:platform=test",
        ttl_seconds=600,
    )
    assert grant, "could not take the lease the proof depends on"
    return grant


def _seed_ghost_owner_and_free(db):
    """A live lease on a conversation with no ``sessions`` row, and nothing else.

    ``import_sessions`` skips every id that already exists, inside the same
    write transaction that would have inserted it, so an existing transcript is
    already out of its reach. What is left is the id it CAN create: one whose
    row is absent while the lease on that conversation is live — a turn that
    took the lease before its session row landed, or whose row was removed
    under it. Seeding it any other way would test the `exists` check instead of
    the fence.
    """
    grant = db.try_acquire_session_turn_lease(
        "owned", f"pid={os.getpid()}:turn=census-proof:platform=test",
        ttl_seconds=600,
    )
    assert grant, "could not take the lease the proof depends on"
    with db._read_ctx() as conn:
        assert not conn.execute(
            "SELECT 1 FROM sessions WHERE id = 'owned'"
        ).fetchone(), "the ghost owner must not have a sessions row"
    return grant


def test_every_self_fencing_sweep_actually_skips_an_owned_conversation(tmp_path):
    """The derived sweep set is proven, not asserted.

    This is what stops the third fence form from being an exemption: adding a
    decorative call to the admission helper puts a mutator in the derived set
    and then fails right here.
    """
    from hermes_state import SessionDB

    derived = _self_fencing_sweeps()
    assert derived, (
        "no derived mutator performs sweep admission, so the third fence form "
        "is vacuous — either the helper was renamed or the sweeps lost their "
        "guard"
    )
    unknown = derived - derive_context_bearing_mutators()
    assert not unknown, f"sweep set escaped the denominator: {sorted(unknown)}"

    # Every derived member, exercised through its real public entry point.
    exercises = {
        "prune_sessions": lambda db: db.prune_sessions(older_than_days=1),
        "delete_empty_sessions": lambda db: db.delete_empty_sessions(),
        "delete_sessions": lambda db: db.delete_sessions(["owned", "free"]),
        "purge_stale_tool_call_markers": lambda db: db.purge_stale_tool_call_markers(
            backup=False
        ),
        # import_sessions never touches a session that already exists, so the
        # conversation it CAN reach while a turn owns it is one whose row is
        # gone (or not written yet) while the lease on its id is live. That is
        # what this seeds; see _seed_ghost_owner_and_free.
        "import_sessions": lambda db: db.import_sessions([
            {"id": "owned", "source": "test", "started_at": 1.0,
             "messages": [{"role": "user", "content": "imported"}]},
            {"id": "free", "source": "test", "started_at": 1.0,
             "messages": [{"role": "user", "content": "imported"}]},
        ]),
    }
    missing = derived - set(exercises)
    assert not missing, (
        f"these mutators claim sweep admission but this test does not exercise "
        f"them, so their claim is unproven: {sorted(missing)}"
    )

    for name in sorted(derived):
        db = SessionDB(tmp_path / f"{name}.db")
        if name == "import_sessions":
            _seed_ghost_owner_and_free(db)
            result = exercises[name](db)
            with db._read_ctx() as conn:
                imported = {
                    r["session_id"] for r in conn.execute(
                        "SELECT DISTINCT session_id FROM messages"
                    ).fetchall()
                }
            assert "free" in imported, (
                "import_sessions imported nothing at all, so this proves "
                f"nothing about what it skipped: {result}"
            )
            assert "owned" not in imported, (
                "import_sessions wrote a transcript onto a conversation a live "
                "turn owns; its call sites are counted as fenced BY it, so "
                "this makes the census a lie"
            )
            db.close()
            continue
        grant = _seed_owned_and_free(db, tmp_path)
        if name == "purge_stale_tool_call_markers":
            for sid in ("owned", "free"):
                db.append_message(
                    session_id=sid, role="assistant", content="[memory]",
                    tool_calls=[{"id": "1", "function": {"name": "x",
                                                         "arguments": "{}"}}],
                    turn_lease_holder=grant if sid == "owned" else None,
                )
        exercises[name](db)
        with db._read_ctx() as conn:
            survivors = {
                r["id"] for r in conn.execute("SELECT id FROM sessions").fetchall()
            }
            owned_content = conn.execute(
                "SELECT content FROM messages WHERE session_id = 'owned'"
            ).fetchall()
        if name == "purge_stale_tool_call_markers":
            assert [r["content"] for r in owned_content] == ["[memory]"], (
                f"{name} cleared content on a conversation a live turn owns"
            )
        else:
            assert "owned" in survivors, (
                f"{name} deleted a conversation a live turn owns; its call sites "
                f"are counted as fenced BY it, so this makes the census a lie"
            )
        db.close()


def test_the_derivation_reads_every_module_sessiondb_is_made_of():
    """The derivation's own premise, asserted rather than assumed.

    ``SessionDB`` is a class plus three mixins in three other files. If a base
    moves to a module the derivation does not parse, its methods silently leave
    the denominator and the census keeps reporting full coverage over a smaller
    set — the exact failure mode the hand-written literal had.
    """
    _methods, missing = _sessiondb_implementation(REPO_ROOT)
    assert not missing, (
        f"SessionDB inherits {sorted(missing)}, and none of "
        f"{list(SESSIONDB_IMPLEMENTATION_MODULES)} defines them. Every method "
        f"of those bases is outside the derived denominator until this list is "
        f"updated deliberately."
    )


def test_the_only_exemption_cannot_touch_a_row_this_generation_wrote(tmp_path):
    """Evidence for the one entry in NOT_CONTEXT_BEARING.

    ``_init_schema`` runs ``UPDATE messages SET active = 1 WHERE active IS
    NULL`` on every open. The argument for exempting it is that the predicate
    cannot match a row this binary produced. That is checkable, so it is
    checked, on the real write paths rather than on a hand-built row.
    """
    from hermes_state import SessionDB

    assert set(NOT_CONTEXT_BEARING) == {"_init_schema"}, (
        f"the exemption list changed; every entry needs the question answered "
        f"and a test that proves the answer: {sorted(NOT_CONTEXT_BEARING)}"
    )
    db = SessionDB(tmp_path / "exemption.db")
    db.create_session("s", source="test")
    db.append_message(session_id="s", role="user", content="one")
    db.append_messages_batch("s", [{"role": "assistant", "content": "two"}])
    with db._read_ctx() as conn:
        nulls = conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE active IS NULL"
        ).fetchone()["n"]
    assert nulls == 0, (
        "a write path in this generation left `active` NULL, so the schema "
        "repair CAN reach a row this binary wrote and the exemption argument "
        "for _init_schema no longer holds"
    )
    db.close()

    # And re-opening (which is what runs the repair) leaves the transcript
    # exactly as it was, including while the conversation is owned.
    reopened = SessionDB(tmp_path / "exemption.db")
    grant = reopened.try_acquire_session_turn_lease(
        "s", f"pid={os.getpid()}:turn=exemption:platform=test", ttl_seconds=600
    )
    assert grant, "could not take the lease this check depends on"
    again = SessionDB(tmp_path / "exemption.db")
    with again._read_ctx() as conn:
        rows = [r["content"] for r in conn.execute(
            "SELECT content FROM messages WHERE session_id = 's' ORDER BY id"
        ).fetchall()]
    assert rows == ["one", "two"], f"the schema repair changed the transcript: {rows}"
    again.close()
    reopened.close()


def test_the_denominator_spans_every_production_package(tmp_path):
    """The set is checked against reality, not against what a scan reached.

    `gateway/` and `tui_gateway/` are two different packages that both exist.
    Naming one and reporting coverage of both is the specific mistake this
    assertion exists to make impossible.
    """
    sites = census()
    assert sites, "the census found nothing — the scan itself is broken"
    packages = {s["package"] for s in sites}
    for required in ("gateway", "tui_gateway"):
        assert (REPO_ROOT / required).is_dir(), (
            f"{required}/ is not a directory in this tree; if it was renamed, "
            f"update the denominator deliberately rather than dropping it"
        )
        assert required in packages, (
            f"{required}/ holds transcript writers but the census reached none "
            f"of them; packages seen: {sorted(packages)}"
        )


def test_the_denominator_states_what_it_left_out():
    """Exclusions stay visible on the GREEN path, not just in a failure report.

    A denominator that quietly shrinks is the failure this whole file was
    rebuilt around. So every excluded call is re-proved here, independently of
    :func:`census`: its module must genuinely never obtain a ``SessionDB``, and
    the keywords it passes must genuinely not exist on the real method.
    """
    params = _mutator_parameters()
    assert params, "SessionDB exposes none of the mutators — the scan is broken"
    excluded = [s for s in census() if s["excluded"]]
    for site in excluded:
        rel = site["location"].rsplit(":", 1)[0]
        source = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
        assert "SessionDB" not in source, (
            f"{site['location']} was left out of the denominator, but its "
            f"module can obtain a SessionDB — the exclusion is not proved"
        )
        assert "hermes_state" not in source, (
            f"{site['location']} imports hermes_state; it cannot be excluded"
        )
    # Reported, not asserted away: the number is part of the claim this file
    # makes, and a reader has to be able to see it move.
    print(
        "census exclusions (same name, provably not SessionDB): "
        + (", ".join(f"{s['location']} — {s['excluded']}" for s in excluded)
           or "none")
    )


def test_no_production_module_writes_messages_outside_sessiondb():
    """The premise the denominator rests on: SessionDB owns the rows.

    If some module grew its own connection and wrote `messages` directly, then
    enumerating SessionDB mutator calls would no longer enumerate the risk, and
    every coverage number in this file would be measuring the wrong set.
    """
    offenders = []
    for rel, path in _production_files():
        if rel.name in ("hermes_state.py", "hermes_state_schema.py",
                        "hermes_state_common.py", "hermes_state_search.py"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        # Docstrings talk ABOUT the SQL (shutdown_flush explains what happens
        # when `INSERT INTO messages` fails); only executable literals count.
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc is not None:
                    docstrings.add(doc)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if node.value in docstrings:
                continue
            lowered = node.value.lower()
            for stmt in ("insert into messages", "update messages set",
                         "delete from messages"):
                if stmt in lowered:
                    offenders.append(f"{rel}:{node.lineno}: {stmt}")
    assert not offenders, (
        "production modules write the messages table outside SessionDB, so the "
        f"census denominator no longer covers the risk: {offenders}"
    )


def test_every_production_transcript_writer_goes_through_the_fence():
    """The census proper."""
    sites = census()
    in_denominator = [s for s in sites if not s["excluded"]]
    excluded = [s for s in sites if s["excluded"]]
    unfenced = [s for s in in_denominator if not s["fenced"]]
    unfenced = [
        s for s in unfenced
        if s["mutator"] not in NOT_CONTEXT_BEARING
    ]
    if unfenced:
        by_package: dict[str, list[str]] = {}
        for s in unfenced:
            by_package.setdefault(s["package"], []).append(
                f"{s['location']} {s['mutator']}()"
            )
        report = "\n".join(
            f"  {pkg} ({len(rows)}):\n" + "\n".join(f"    {r}" for r in sorted(rows))
            for pkg, rows in sorted(by_package.items())
        )
        excluded_report = "\n".join(
            f"    {s['location']} {s['mutator']}(): {s['excluded']}"
            for s in sorted(excluded, key=lambda s: s["location"])
        )
        pytest.fail(
            f"{len(unfenced)} of {len(in_denominator)} production transcript "
            f"writers do not present a turn-lease grant and are not inside a "
            f"fence scope:\n{report}\n"
            f"  ({len(excluded)} same-named calls proved not to be SessionDB "
            f"and left out of the denominator:\n{excluded_report})"
        )
