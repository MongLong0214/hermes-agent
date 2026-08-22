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
    binds the grant it presents. Anything else is an unfenced production writer,
    and this test names it.

MEMBERSHIP IS ARGUED, NOT ASSUMED
    :data:`NOT_CONTEXT_BEARING` is deliberately empty. Every candidate for it
    has to answer one question — *can this write change what the provider will
    see on the next turn, remove it, or consume it?* — and the two entries that
    were previously exempted both answer yes. An entry may only be added with
    that question answered in its comment.
"""

from __future__ import annotations

import ast
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


def census() -> list[dict]:
    """Every production call site in the denominator, with its fence status."""
    sites = []
    for rel, path in _production_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
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
            carries = any(k.arg == "turn_lease_holder" for k in node.keywords)
            inside = any(lo <= node.lineno <= hi for lo, hi in fenced_spans)
            sites.append({
                "location": f"{rel}:{node.lineno}",
                "package": rel.parts[0] if len(rel.parts) > 1 else "(root)",
                "mutator": fn.attr,
                "fenced": carries or inside,
            })
    return sites


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
    unfenced = [s for s in sites if not s["fenced"]]
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
        pytest.fail(
            f"{len(unfenced)} of {len(sites)} production transcript writers do "
            f"not present a turn-lease grant and are not inside a fence "
            f"scope:\n{report}"
        )
