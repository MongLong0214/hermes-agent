"""The fence is bypassable by the method next door, and the set is derived.

WHAT THE PREVIOUS SLICE LEFT OPEN
    ``update_session_model`` is fenced. ``update_session_runtime_lock``, eleven
    hundred lines away, executes::

        UPDATE sessions SET
           model_config = ?,
           model = COALESCE(?, model),
           system_prompt = NULL,
           system_prompt_hash = NULL
         WHERE id = ?

    — a superset of what the fenced method writes, with zero
    ``_check_turn_lease_guard`` calls. A fence one method can walk around is
    not a fence, and this one additionally NULLs the assembled prompt, so a
    bystander can delete the bytes a running turn is replaying under.

    Three more sat in the same state: ``update_system_prompt`` (writes the
    prompt hash and NULLs the inline snapshot), ``update_session_billing_route``
    (NULLs both prompt columns so a stale ``Model:`` footer cannot lie) and
    ``set_session_yolo`` (rewrites ``model_config``). All four are named in the
    fence surface's own declaration as things the next turn replays under.

HOW THE SET IS DECIDED — DERIVED, NOT LISTED
    :func:`derive_replay_column_overwriters` reads ``SessionDB`` and its mixins
    and returns every method that ASSIGNS a replay column an expression which
    is not that column's own NULL-backfill. The distinction is read off the
    SQL and it is the one that matters:

        model = COALESCE(model, ?)          fills a hole; cannot change a route
        model = COALESCE(?, model)          replaces the route
        system_prompt = NULL                deletes what the turn is replaying

    The first form is why ``update_token_counts`` is not simply "an unfenced
    writer of ``model``" — its ordinary statement cannot move a model that is
    already set. (It has a second statement that can; see THE RESIDUE.)

    Each overwriter is then classified by the fence form it reaches, all three
    of them derived from the source rather than named here:

    grant   it calls the admission helper inside its write transaction.
    sweep   it calls the per-row sweep admission (``import_sessions`` does —
            its targets come from a payload, so no single grant authorises it).
    inner   it is handed another method's open transaction as its first
            positional argument, so its SQL belongs to that method's write.
            ``_init_schema`` -> ``_dedupe_legacy_system_prompts(cursor)`` is
            the shape, and ``_init_schema`` is the census's one argued
            exemption.

THE RESIDUE, AND WHY IT IS A RATCHET RATHER THAN A LIST
    Two overwriters are still open after this slice and neither can be closed
    from here:

    ``_insert_session_row``   ``update_token_counts`` calls it on EVERY
                              accounted API call, to make sure the row exists.
                              Fencing it holderless refuses the owner's own
                              accounting mid-turn, so closing it means giving
                              ``update_token_counts`` — and ``create_session``,
                              and every one of their callers — a grant to
                              present. That is a slice, not a line.
    ``update_token_counts``   its ``first_accounted_route`` branch runs
                              ``UPDATE sessions SET model = ?`` unconditionally
                              when ``api_call_count == 0``. Same call sites,
                              same problem.

    So the check here is not "the open set is empty" and it is not a list of
    names to forgive. It is a RATCHET against
    :data:`BASELINE_COMMIT`: the derivation runs against a ``git archive`` of
    that commit and against the working tree, and the tree's open set must be a
    strict subset. Adding method 64 with an unfenced ``UPDATE sessions SET
    model = ?`` fails here; fencing one of the two remaining shrinks it further
    and still passes. Nothing is written down that a future author can grow.

WHAT THE PINS ASSERT
    Rows and values. For each newly fenced method: the bystander's write is
    refused AND the four replay columns are byte-identical afterwards; the
    owner's own write with its grant LANDS (a fence that refuses everybody
    passes a refusal test perfectly); and a holderless write on a FREE
    conversation still lands, which is every single-writer install.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import subprocess

import pytest

from tests.state.lease_mutation_harness import (
    Mutation,
    assert_every_pin_has_a_killer,
    assert_mutation_kills_the_pin,
)
from tests.state.test_turn_lease_generation_trigger import _git_dir
from tests.state import test_turn_lease_writer_census as census_mod

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: This file, so a mutated extract can run the same checks it defines.
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)

#: The tree the ratchet is measured against: the commit at which the turn-lease
#: fence covered update_session_model / patch_session_model_config /
#: update_session_meta and nothing else in `sessions`. An immutable object, so
#: the baseline cannot be edited into agreement with a regression.
BASELINE_COMMIT = "ee1de21518ec834a58ac820ee11388788920d31e"

#: The modules ``SessionDB`` is made of, so the baseline extract is the same
#: shape as the working tree the derivation reads.
IMPLEMENTATION_MODULES = census_mod.SESSIONDB_IMPLEMENTATION_MODULES

#: What the next turn is dispatched under. Read off the statement
#: ``update_session_model`` executes — the method whose entire purpose is
#: "change what the model sees".
REPLAY_COLUMNS = ("model", "model_config", "system_prompt", "system_prompt_hash")

_SET_CLAUSE = re.compile(
    r"update\s+sessions\s+set\s+(.*?)(?:\bwhere\b|$)", re.IGNORECASE | re.DOTALL
)
_ON_CONFLICT = re.compile(
    r"on\s+conflict\s*\([^)]*\)\s*do\s+update\s+set\s+(.*)$",
    re.IGNORECASE | re.DOTALL,
)
_INSERT_COLUMNS = re.compile(
    r"insert(?:\s+or\s+\w+)?\s+into\s+sessions\s*\(([^)]*)\)",
    re.IGNORECASE | re.DOTALL,
)
#: ``col = COALESCE(col, ...)`` — the column guards itself, so a value that is
#: already there survives. Position matters: ``COALESCE(?, col)`` mentions the
#: same column and overwrites.
_SELF_BACKFILL = re.compile(
    r"^coalesce\s*\(\s*(?:sessions\s*\.\s*)?(\w+)\s*,", re.IGNORECASE
)

#: The admission helpers, by the name the source calls them.
_GRANT_HELPERS = ("_check_turn_lease_guard", "_check_transcript_write_guards")
_SWEEP_HELPER = census_mod.SWEEP_ADMISSION_HELPER


def _split_assignments(body: str):
    """``(column, expression)`` for each top-level ``col = expr`` in a SET body.

    Split at depth zero so ``COALESCE(a, b)`` and ``CASE ... END`` survive
    intact; a plain ``str.split(",")`` cuts them in half and every classifier
    downstream then reads the fragments.
    """
    parts, depth, buf = [], 0, ""
    for ch in body:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    parts.append(buf)
    out = []
    for part in parts:
        match = re.match(r"\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*)", part, re.DOTALL)
        if match:
            out.append((match.group(1).lower(), match.group(2).strip()))
    return out


def _overwritten_replay_columns(sql: str) -> set:
    """Replay columns *sql* can move off a value that is already there."""
    overwritten = set()
    bodies = [m.group(1) for m in _SET_CLAUSE.finditer(sql)]
    bodies += [m.group(1) for m in _ON_CONFLICT.finditer(sql)]
    for body in bodies:
        for column, expression in _split_assignments(body):
            if column not in REPLAY_COLUMNS:
                continue
            backfill = _SELF_BACKFILL.match(expression)
            if not (backfill and backfill.group(1).lower() == column):
                overwritten.add(column)
    for match in _INSERT_COLUMNS.finditer(sql):
        for column in match.group(1).split(","):
            if column.strip().lower() in REPLAY_COLUMNS:
                overwritten.add(column.strip().lower())
    return overwritten


def _string_literals(node):
    for inner in ast.walk(node):
        if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
            yield inner.value


def _calls_any(node, names) -> bool:
    for inner in ast.walk(node):
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr in names
        ):
            return True
    return False


def _names_bound_in(method) -> set:
    """Every parameter AND local name bound anywhere inside *method*.

    Wider than the census's parameter-only version on purpose:
    ``_init_schema`` opens its cursor with ``cursor = self._conn.cursor()`` and
    hands it to ``_dedupe_legacy_system_prompts(cursor)``. A parameter-only
    rule cannot see that the helper's SQL runs in ``_init_schema``'s
    transaction, and the widening can only make the covered set larger, which
    is the direction that cannot hide a writer.
    """
    names = set(census_mod._names_bound_in(method))
    for inner in ast.walk(method):
        if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Store):
            names.add(inner.id)
    return names


def derive_replay_column_overwriters(root: pathlib.Path) -> dict:
    """``{method: sorted(columns)}`` for every replay-column overwriter under *root*."""
    methods, _missing = census_mod._sessiondb_implementation(root)
    found = {}
    for name, (_module, node) in methods.items():
        columns = set()
        for sql in _string_literals(node):
            columns |= _overwritten_replay_columns(sql)
        if columns:
            found[name] = sorted(columns)
    return found


def derive_open_overwriters(root: pathlib.Path) -> frozenset:
    """Overwriters under *root* that reach no fence form at all.

    The three forms are derived: a call to an admission helper, a call to the
    per-row sweep admission, or being handed another covered method's open
    transaction (applied to a fixpoint, so a helper of a helper is covered too).
    """
    methods, _missing = census_mod._sessiondb_implementation(root)
    overwriters = set(derive_replay_column_overwriters(root))

    covered = {
        name for name in overwriters
        if _calls_any(methods[name][1], _GRANT_HELPERS)
        or _calls_any(methods[name][1], (_SWEEP_HELPER,))
    }
    covered |= set(census_mod.NOT_CONTEXT_BEARING) & set(methods)

    while True:
        grown = set()
        for name in covered:
            entry = methods.get(name)
            if entry is None:
                continue
            bound = _names_bound_in(entry[1])
            for inner in ast.walk(entry[1]):
                if not isinstance(inner, ast.Call):
                    continue
                fn = inner.func
                if not (
                    isinstance(fn, ast.Attribute)
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "self"
                ):
                    continue
                first = inner.args[0] if inner.args else None
                if isinstance(first, ast.Name) and first.id in bound:
                    grown.add(fn.attr)
        new = (grown & overwriters) - covered
        if not new:
            break
        covered |= new
    return frozenset(overwriters - covered)


def _baseline_tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """A bare extract of :data:`BASELINE_COMMIT`, implementation modules only."""
    git_dir = _git_dir()
    if git_dir is None:
        pytest.skip("no git repository to read the baseline commit from")
    out = tmp_path / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "-C", git_dir, "archive", BASELINE_COMMIT, "--",
         *IMPLEMENTATION_MODULES],
        capture_output=True,
    )
    assert archive.returncode == 0, (
        f"could not read {BASELINE_COMMIT}: "
        f"{archive.stderr.decode(errors='replace')}"
    )
    extract = subprocess.run(
        ["tar", "-x", "-C", str(out)], input=archive.stdout, capture_output=True
    )
    assert extract.returncode == 0, extract.stderr.decode(errors="replace")
    assert (out / "hermes_state.py").is_file()
    return out


# ---------------------------------------------------------------------------
# The derivation checks.
# ---------------------------------------------------------------------------

def test_the_named_hole_is_a_replay_column_overwriter(tmp_path):
    """``update_session_runtime_lock`` writes a superset of the fenced method.

    Stated as a comparison of the two derived column sets rather than as prose,
    so a future edit that narrows either statement is visible here.
    """
    derived = derive_replay_column_overwriters(REPO_ROOT)
    fenced = set(derived.get("update_session_model", ()))
    adjacent = set(derived.get("update_session_runtime_lock", ()))
    assert fenced, "update_session_model no longer overwrites any replay column"
    assert fenced <= adjacent, (
        f"update_session_runtime_lock no longer writes a superset of "
        f"update_session_model: {sorted(adjacent)} vs {sorted(fenced)}"
    )


def test_no_new_unfenced_replay_column_writer_since_the_fence_landed(tmp_path):
    """The ratchet. Strictly fewer open overwriters than at the baseline.

    Not "the open set is empty" — two writers cannot be closed from this slice
    (see THE RESIDUE) — and not a list of names, which is the thing a later
    author can quietly grow. A new unfenced writer of ``model`` /
    ``model_config`` / ``system_prompt`` / ``system_prompt_hash`` appears in
    ``current`` and not in ``baseline``, and fails here.
    """
    baseline = derive_open_overwriters(_baseline_tree(tmp_path))
    current = derive_open_overwriters(REPO_ROOT)

    assert current <= baseline, (
        f"a replay-column writer is unfenced that was not unfenced at "
        f"{BASELINE_COMMIT[:10]}: {sorted(current - baseline)}\n"
        f"Every one of `model`, `model_config`, `system_prompt` and "
        f"`system_prompt_hash` is what the next turn is dispatched under, so a "
        f"writer that can move one of them while another process owns the "
        f"conversation is the hole this family exists to close. Either present "
        f"a grant and call the admission helper, or hand the write another "
        f"fenced method's open transaction.\n"
        f"baseline: {sorted(baseline)}\ncurrent:  {sorted(current)}"
    )
    assert current < baseline, (
        f"this slice was supposed to fence four of them and the open set did "
        f"not shrink: {sorted(current)}"
    )


# ---------------------------------------------------------------------------
# The behavioural pins.
# ---------------------------------------------------------------------------

def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _store(tmpdir, name="state.db"):
    from hermes_state import SessionDB

    return SessionDB(db_path=pathlib.Path(tmpdir) / name)


def _replay_state(db, session_id):
    row = db.get_session(session_id)
    if row is None:
        return None
    return tuple(row[column] for column in REPLAY_COLUMNS)


def _owned_session(db, session_id="s", *, tag="owner"):
    """A conversation with a LIVE owner, carrying a prompt and a route."""
    db.create_session(session_id, "test")
    db.append_message(session_id, "user", f"{session_id} context")
    db.update_session_model(session_id, "anthropic/claude-before")
    db.update_system_prompt(session_id, "THE PROMPT THE TURN IS REPLAYING")
    grant = db.try_acquire_session_turn_lease(
        session_id, _holder(tag), ttl_seconds=600
    )
    assert grant, f"could not take the lease on {session_id!r}"
    return grant


def _refused_then_owned_then_free(
    tmpdir, bystander, owner_write, free_write, observe
):
    """The three-part shape every pin below uses.

    * a holderless write while somebody owns the conversation is refused AND
      leaves all four replay columns byte-identical;
    * the owner's own write, presenting its grant, LANDS — otherwise the
      refusal above is indistinguishable from a fence that refuses everybody;
    * once the lease is released, a holderless write lands again, which is
      every single-writer install, every fresh session and every import.

    *observe* returns the value a landed write must MOVE, and it is per-pin
    rather than the replay tuple for a reason found by running this:
    ``update_session_billing_route``'s only replay-column effect is NULLing two
    columns that the previous step already NULLed, so "the replay tuple
    changed" reads as "the write was refused" on the second call. A landedness
    probe that a legitimate write cannot move is a check that fails on correct
    code, which is the same fault as one that passes on broken code.
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        grant = _owned_session(db)
        before = _replay_state(db, "s")
        before_observed = observe(db)
        assert before[0] == "anthropic/claude-before"
        assert before[3], "the fixture has no prompt hash to lose"

        try:
            bystander(db)
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a holderless write was admitted while another writer held "
                "the conversation's turn lease"
            )
        assert _replay_state(db, "s") == before, (
            f"the refused write changed the replay columns anyway: "
            f"{_replay_state(db, 's')!r} != {before!r}"
        )
        assert observe(db) == before_observed, (
            f"the refused write landed anyway: {observe(db)!r} != "
            f"{before_observed!r}"
        )

        owner_write(db, grant)
        after_owner = observe(db)
        assert after_owner != before_observed, (
            f"the owner's own write was refused: {after_owner!r} == "
            f"{before_observed!r}"
        )

        db.release_session_turn_lease("s", grant)
        free_write(db)
        assert observe(db) != after_owner, (
            f"a holderless write is refused even on a FREE conversation, which "
            f"breaks every single-writer install: {observe(db)!r}"
        )
    finally:
        db.close()


def _model_config(db):
    return (db.get_session("s") or {}).get("model_config")


def _prompt_hash(db):
    return (db.get_session("s") or {}).get("system_prompt_hash")


def _billing_provider(db):
    return (db.get_session("s") or {}).get("billing_provider")


def check_a_runtime_lock_write_is_refused_while_a_live_owner_holds_it(
    tmpdir,
) -> None:
    """The named hole: a superset of ``update_session_model``, unfenced.

    ``UPDATE sessions SET model_config = ?, model = COALESCE(?, model),
    system_prompt = NULL, system_prompt_hash = NULL`` — every column the fenced
    method writes, plus it drops the assembled prompt. Reached from
    ``gateway/platforms/api_server.py`` on any request carrying a runtime lock,
    which is a different process from the one running the turn as often as not.
    """
    _refused_then_owned_then_free(
        tmpdir,
        lambda db: db.update_session_runtime_lock(
            "s", model="anthropic/claude-stolen", provider="smuggler",
            confirmed=True,
        ),
        lambda db, grant: db.update_session_runtime_lock(
            "s", model="anthropic/claude-owner", provider="owner",
            confirmed=True, turn_lease_holder=grant,
        ),
        lambda db: db.update_session_runtime_lock(
            "s", model="anthropic/claude-free", provider="free", confirmed=True,
        ),
        _model_config,
    )


def check_a_system_prompt_rewrite_is_refused_while_a_live_owner_holds_it(
    tmpdir,
) -> None:
    """The prompt snapshot IS the context; replacing it is not bookkeeping.

    ``update_system_prompt`` stores a new hash and NULLs the inline column, so
    a bystander replaces the exact bytes the running turn resumes from.
    """
    _refused_then_owned_then_free(
        tmpdir,
        lambda db: db.update_system_prompt("s", "SMUGGLED PROMPT"),
        lambda db, grant: db.update_system_prompt(
            "s", "OWNER PROMPT", turn_lease_holder=grant
        ),
        lambda db: db.update_system_prompt("s", "FREE PROMPT"),
        _prompt_hash,
    )


def check_a_billing_route_write_is_refused_when_the_lease_lands_mid_call(
    tmpdir,
) -> None:
    """The IN-TRANSACTION guard, isolated from the advisory one.

    ``update_session_billing_route`` NULLs ``system_prompt`` /
    ``system_prompt_hash`` on the way past, so it is a context write however it
    is named — the running turn's prompt is gone either way. But it carries TWO
    admission points: ``_refuse_before_side_effects`` before the token-queue
    barrier, and ``_check_turn_lease_guard`` inside ``BEGIN IMMEDIATE``. A
    plain "a bystander is refused" pin is satisfied by either, so no single
    mutation can kill it and it measures neither.

    So this pin opens the window the advisory check cannot see. The
    conversation is FREE when the call starts, and the lease is taken DURING
    ``flush_token_counts`` — the exact interval between the advisory read and
    the write transaction. The advisory check has already said "no finding";
    only the guard inside the transaction can refuse now, and that is the
    property: the early refusal saves side effects, it is not the authority.
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        db.create_session("s", "test")
        db.append_message("s", "user", "s context")
        db.update_session_model("s", "anthropic/claude-before")
        db.update_system_prompt("s", "THE PROMPT THE TURN IS REPLAYING")

        before = _replay_state(db, "s")
        before_provider = _billing_provider(db)
        assert db.get_session_turn_lease("s") is None, (
            "the conversation must start FREE or the advisory check refuses "
            "first and this pin measures the wrong guard"
        )

        taken = {}
        original_flush = db.flush_token_counts

        def _flush_and_take_the_lease(*args, **kwargs):
            # Restored immediately so the owner's own call below flushes
            # normally, and so the lease is taken exactly once.
            db.flush_token_counts = original_flush
            taken["grant"] = db.try_acquire_session_turn_lease(
                "s", _holder("landed-mid-call"), ttl_seconds=600
            )
            return original_flush(*args, **kwargs)

        db.flush_token_counts = _flush_and_take_the_lease

        try:
            db.update_session_billing_route(
                "s", provider="smuggler", base_url="https://smuggled.example",
            )
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a write was admitted although the conversation was taken "
                "between the advisory read and the write transaction. The "
                "advisory check cannot see that, which is why it is not the "
                "authority"
            )
        assert taken.get("grant"), (
            "the lease was never taken, so the window was never opened and "
            "this pin proves nothing"
        )
        assert _replay_state(db, "s") == before, (
            f"the refused write changed the replay columns anyway: "
            f"{_replay_state(db, 's')!r} != {before!r}"
        )
        assert _billing_provider(db) == before_provider, (
            f"the refused write landed anyway: {_billing_provider(db)!r}"
        )

        db.update_session_billing_route(
            "s", provider="owner", base_url="https://owner.example",
            turn_lease_holder=taken["grant"],
        )
        assert _billing_provider(db) == "owner", (
            f"the owner's own write was refused: {_billing_provider(db)!r}"
        )
    finally:
        db.close()


def check_a_yolo_toggle_is_refused_while_a_live_owner_holds_it(tmpdir) -> None:
    """``model_config`` is one column and the whole of it is replayed.

    ``set_session_yolo`` merges one key into it — through the same merge
    helper ``update_session_model`` uses — and writes the merged document back.
    A merge is a read-modify-write, so it carries whatever the reader saw over
    whatever the owner wrote in between.
    """
    _refused_then_owned_then_free(
        tmpdir,
        lambda db: db.set_session_yolo("s", True),
        lambda db, grant: db.set_session_yolo(
            "s", True, turn_lease_holder=grant
        ),
        lambda db: db.set_session_yolo("s", False),
        _model_config,
    )


PINS = {
    "check_a_runtime_lock_write_is_refused_while_a_live_owner_holds_it":
        check_a_runtime_lock_write_is_refused_while_a_live_owner_holds_it,
    "check_a_system_prompt_rewrite_is_refused_while_a_live_owner_holds_it":
        check_a_system_prompt_rewrite_is_refused_while_a_live_owner_holds_it,
    "check_a_billing_route_write_is_refused_when_the_lease_lands_mid_call":
        check_a_billing_route_write_is_refused_when_the_lease_lands_mid_call,
    "check_a_yolo_toggle_is_refused_while_a_live_owner_holds_it":
        check_a_yolo_toggle_is_refused_while_a_live_owner_holds_it,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_replay_column_writer_property(name, tmp_path):
    """The pin. Each property, asserted against the tree under test."""
    PINS[name](tmp_path)


def _guard_block(comment: str) -> str:
    """The admission call as it appears in one mutator, with its own comment.

    Keyed by the comment line so each row names ONE guard: the call itself is
    character-identical in every mutator, and an anchor that matches five
    places names none of them.
    """
    return (
        f"            # {comment}\n"
        "            self._check_turn_lease_guard(\n"
        "                conn,\n"
        "                session_id,\n"
        "                turn_lease_holder,\n"
        "                turn_lease_ttl_seconds=turn_lease_ttl_seconds,\n"
        "            )\n"
    )


#: This module derives through the census's parsing helpers, so a mutated
#: extract has to contain them or the pin module cannot be imported at all —
#: and a clean run that dies on ImportError reads as "the pin does not hold".
#: An extract pathspec, which is what BASE_EXTRACT_PATHSPEC already is; not a
#: list of methods or exemptions.
_EXTRA_EXTRACT = ("tests/state/test_turn_lease_writer_census.py",)


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_a_runtime_lock_write_is_refused_while_a_live_owner_holds_it",
        module="hermes_state.py",
        find=_guard_block(
            "This rewrites the whole route the next turn replays under."
        ),
        replace="",
        why="this is the method the fence was bypassable through: it writes "
            "every column update_session_model writes and drops the prompt",
    ),
    Mutation(
        pin="check_a_system_prompt_rewrite_is_refused_while_a_live_owner_holds_it",
        module="hermes_state.py",
        find=_guard_block(
            "The prompt snapshot is what the next turn resumes from."
        ),
        replace="",
        why="unfenced, a bystander replaces the exact bytes a running turn is "
            "replaying under",
    ),
    Mutation(
        pin="check_a_billing_route_write_is_refused_when_the_lease_lands_mid_call",
        module="hermes_state.py",
        find=_guard_block(
            "Nulling the prompt snapshot changes what the next turn replays."
        ),
        replace="",
        why="with the in-transaction guard gone the only admission left is "
            "the advisory read, which ran while the conversation was still "
            "free — a lease taken during the flush is invisible to it",
    ),
    Mutation(
        pin="check_a_yolo_toggle_is_refused_while_a_live_owner_holds_it",
        module="hermes_state.py",
        find=_guard_block(
            "The yolo flag rides in model_config, which the next turn replays."
        ),
        replace="",
        why="a merge into model_config is a read-modify-write on a column the "
            "next turn is dispatched under",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    """Clean, mutated, restored. A pin that survives its mutation is not a pin."""
    assert_mutation_kills_the_pin(mutation, str(_SELF), tmp_path, *_EXTRA_EXTRACT)


def test_every_pin_has_a_mutation_that_kills_it():
    """No pin without a killer, and no killer without a pin."""
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)
