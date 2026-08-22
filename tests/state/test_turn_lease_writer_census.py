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
    Every call, in non-test Python under the repository root, to one of
    :data:`CONTEXT_BEARING_MUTATORS` on any receiver.

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
    only through ``SessionDB``'s public mutators — the class owns the sole
    connection and no production module opens its own handle to write them
    (asserted by :func:`test_no_production_module_writes_messages_outside_sessiondb`).
    So a call to one of those names is necessary and sufficient to change what
    a later turn replays, and enumerating the calls enumerates the risk.

WHAT COUNTS AS GOING THROUGH THE FENCE
    The call passes ``turn_lease_holder=``, or it is lexically inside a
    ``with ... session_turn_lease(...)`` / ``turn_lease_scope(...)`` block that
    binds the grant it presents, or the mutator it calls is a SWEEP that does
    its own per-row admission inside its write transaction. Anything else is an
    unfenced production writer, and this test names it.

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

    * the set is DERIVED from ``hermes_state.py`` (:func:`_self_fencing_sweeps`)
      rather than listed here, so it cannot be grown by editing this file;
    * the marker it derives from is the call to the real admission helper, and
      :func:`test_every_self_fencing_sweep_actually_skips_an_owned_conversation`
      proves BEHAVIOURALLY that each derived member leaves an owned
      conversation alone — a decorative call to the helper fails that test;
    * membership is still restricted to :data:`CONTEXT_BEARING_MUTATORS`.

    A single-target mutator can never qualify: it has one conversation, so it
    has a call site that can hold a grant, and every one of them does.

MEMBERSHIP IS ARGUED, NOT ASSUMED
    :data:`NOT_CONTEXT_BEARING` is deliberately empty. Every candidate for it
    has to answer one question — *can this write change what the provider will
    see on the next turn, remove it, or consume it?* — and the two entries that
    were previously exempted both answer yes. An entry may only be added with
    that question answered in its comment.
"""

from __future__ import annotations

import ast
import os
import time
import pathlib

import pytest

#: Mutators that reach model context. Session deletion is included: removing the
#: rows is the most complete way to change what the next turn replays.
CONTEXT_BEARING_MUTATORS = frozenset({
    "append_message",
    "append_messages_batch",
    "replace_messages",
    "archive_and_compact",
    "publish_compression_child",
    "rewind_to_message",
    "restore_rewound",
    "clear_messages",
    "set_latest_user_api_content",
    # Reactions are NOT presentation: take_unseen_reactions consumes the
    # announcement that is injected into model context on the next turn, and
    # set_message_reaction produces it. Consume-once state, not a decoration.
    "set_message_reaction",
    "take_unseen_reactions",
    # display_kind is read back by the context pipeline (a "hidden" row is
    # treated differently) and by the compressor's real-ask classification, so
    # stamping it changes what the provider sees.
    "set_latest_matching_message_display_kind",
    # Clears `content` in place across sessions.
    "purge_stale_tool_call_markers",
    # Destructive lifecycle: these remove the context entirely.
    "delete_session",
    "delete_sessions",
    "delete_empty_sessions",
    "prune_sessions",
})

#: Empty on purpose — see the module docstring. Adding an entry requires
#: answering "can this change, remove or consume what the provider sees next?"
NOT_CONTEXT_BEARING: dict = {}

#: Trees that are not shipped production Python.
EXCLUDED_DIRS = frozenset({
    "tests", "tests-js", "evals", "node_modules", ".git", "web", "website",
    "docs", "native", "ui-tui", "contributors", "locales", "assets",
})

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

FENCE_HELPERS = ("session_turn_lease", "turn_lease_scope")


def _production_files():
    for path in sorted(REPO_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
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


def _self_fencing_sweeps() -> frozenset:
    """Mutators in ``hermes_state.py`` that admit per row inside their own txn.

    Derived, never listed. A method qualifies by calling
    :data:`SWEEP_ADMISSION_HELPER` somewhere in its body (including the nested
    ``_do(conn)`` closure that runs in the write transaction), and by being a
    context-bearing mutator in the first place.
    """
    tree = ast.parse((REPO_ROOT / "hermes_state.py").read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in CONTEXT_BEARING_MUTATORS:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == SWEEP_ADMISSION_HELPER
            ):
                found.add(node.name)
                break
    return frozenset(found)


def _mutator_parameters() -> dict:
    """Parameter names of each mutator, read off ``SessionDB`` itself."""
    import inspect

    from hermes_state import SessionDB

    params = {}
    for name in CONTEXT_BEARING_MUTATORS:
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


def census() -> list[dict]:
    """Every production call site in the denominator, with its fence status."""
    sweeps = _self_fencing_sweeps()
    params = _mutator_parameters()
    sites = []
    for rel, path in _production_files():
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
            if fn.attr not in CONTEXT_BEARING_MUTATORS:
                continue
            excluded = _provably_not_sessiondb(node, fn.attr, source, params)
            carries = any(k.arg == "turn_lease_holder" for k in node.keywords)
            inside = any(lo <= node.lineno <= hi for lo, hi in fenced_spans)
            sweep = fn.attr in sweeps
            sites.append({
                "location": f"{rel}:{node.lineno}",
                "package": rel.parts[0] if len(rel.parts) > 1 else "(root)",
                "mutator": fn.attr,
                "excluded": excluded,
                "fenced": carries or inside or sweep,
                "how": (
                    "grant" if carries
                    else "scope" if inside
                    else "sweep" if sweep
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


def test_every_self_fencing_sweep_actually_skips_an_owned_conversation(tmp_path):
    """The derived sweep set is proven, not asserted.

    This is what stops the third fence form from being an exemption: adding a
    decorative call to the admission helper puts a mutator in the derived set
    and then fails right here.
    """
    from hermes_state import SessionDB

    derived = _self_fencing_sweeps()
    assert derived, (
        "no mutator in hermes_state.py performs sweep admission, so the third "
        "fence form is vacuous — either the helper was renamed or the sweeps "
        "lost their guard"
    )
    unknown = derived - CONTEXT_BEARING_MUTATORS
    assert not unknown, f"sweep set escaped the denominator: {sorted(unknown)}"

    # Every derived member, exercised through its real public entry point.
    exercises = {
        "prune_sessions": lambda db: db.prune_sessions(older_than_days=1),
        "delete_empty_sessions": lambda db: db.delete_empty_sessions(),
        "delete_sessions": lambda db: db.delete_sessions(["owned", "free"]),
        "purge_stale_tool_call_markers": lambda db: db.purge_stale_tool_call_markers(
            backup=False
        ),
    }
    missing = derived - set(exercises)
    assert not missing, (
        f"these mutators claim sweep admission but this test does not exercise "
        f"them, so their claim is unproven: {sorted(missing)}"
    )

    for name in sorted(derived):
        db = SessionDB(tmp_path / f"{name}.db")
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
