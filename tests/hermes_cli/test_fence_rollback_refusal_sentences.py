"""A refusal may not describe work the run never did.

``_human_lines`` picks the sentence that follows a refusal from a four-way
chain. The third arm reads::

    rehearsal = payload.get("rehearsal") or {}
    ...
    elif rehearsal.get("changed") is None:
        "The rehearsal's own commit on a disposable copy did not report back,
         which is a fault in this run and not in your store."

``{}.get("changed")`` IS ``None``. So the arm fires whenever the payload has no
``rehearsal`` key at all — and no payload in this tree ever has one:
``_emit_refusal`` publishes the key only under ``if rehearsal:``, its
``rehearsal=`` parameter has ZERO callers anywhere in the repository, and both
refusal call sites (the rehearsal reporter and the real run) omit it. One of
them says so in a comment: *"NO `rehearsal` BLOCK. Nothing was rehearsed, so
there are no rehearsal facts — not false ones."*

The consequence is the whole of this file. EVERY ``changed: false`` refusal —
including ``store-missing``, which refuses before opening anything — tells the
operator that a rehearsal ran on a disposable copy and faulted. It also tells
them the fault is "in this run and not in your store", which is a diagnosis
about an event that did not occur. The honest sentence for that payload is the
fourth arm, ``"Nothing was changed."``, and it is unreachable.

WHY THIS IS THE OUTPUT-TRUTH RULE THE MODULE ALREADY HOLDS ITSELF TO
    ``_emit_refusal``'s own docstring draws exactly this distinction for the
    STRUCTURED half — "ABSENT AND ``false`` ARE DIFFERENT STATEMENTS … When the
    run refuses before deriving anything, no backup step ran at all — so
    rendering ``false`` is a claim about an event that never had one." The
    structured half obeys it. The sentence half, rendered from the same payload
    two functions later, collapses absent into a value and then narrates it.

WHY THE PIN DRIVES THE VERB AND READS STDERR
    The defect is only visible in the sentence a person reads. A check against
    ``_human_lines`` with a hand-built dict would pass a payload the verb never
    produces, and a check on the JSON would see nothing wrong at all — the JSON
    is correct. So this goes through ``cmd_sessions`` and asserts on the bytes
    that reach the operator's terminal.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
from argparse import Namespace
from dataclasses import dataclass

import pytest

from tests.state.lease_mutation_harness import (
    Mutation,
    assert_every_pin_has_a_killer,
    assert_mutation_kills_the_pin,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: This file, so a mutated extract can run the same checks it defines.
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)

#: The verb lives in the CLI package and drives the state layer, so the narrow
#: state-only extract the harness defaults to is not enough.
_EXTRA_EXTRACT = (".",)

_REHEARSAL_SENTENCE = "The rehearsal's own commit on a disposable copy"
_NOTHING_CHANGED = "Nothing was changed."


@dataclass(frozen=True)
class VerbRun:
    """One invocation, reduced to values so a crash is an observation."""

    rc: object
    stdout: str
    stderr: str
    crash: str


def _run_verb(store, backup, *, dry_run=False, work_dir=None) -> VerbRun:
    """Drive the verb THROUGH ``cmd_sessions`` — the operator surface."""
    from hermes_cli.sessions_cmd import cmd_sessions

    args = Namespace(
        sessions_action="fence-rollback",
        store=store,
        backup=backup,
        dry_run=dry_run,
        work_dir=work_dir,
    )
    out, err = io.StringIO(), io.StringIO()
    crash, rc = "", None
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cmd_sessions(args)
    except BaseException as exc:  # noqa: BLE001 - carried, not swallowed
        crash = f"{type(exc).__name__}: {exc}"
    return VerbRun(rc=rc, stdout=out.getvalue(), stderr=err.getvalue(), crash=crash)


def _payload(run: VerbRun):
    try:
        return json.loads(run.stdout)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The property.
# ---------------------------------------------------------------------------

def check_a_refusal_that_never_rehearsed_does_not_claim_one(tmpdir) -> None:
    """The sentence after a not-started refusal describes the not-started run.

    ``store-missing`` is the strongest available case: the verb refuses on a
    path that is not a file, so nothing was opened, copied, rehearsed or
    committed. Whatever it says about a rehearsal is necessarily invented.
    """
    where = pathlib.Path(tmpdir)
    run = _run_verb(where / "not-a-store.db", where / "backup.db")

    assert not run.crash, f"the verb crashed: {run.crash}"
    payload = _payload(run)
    assert payload is not None, (
        f"the verb printed no structured report:\n{run.stdout}\n{run.stderr}"
    )

    # The premise, asserted rather than assumed: this payload carries no
    # rehearsal facts, because no rehearsal happened.
    assert payload.get("ok") is False
    assert payload.get("changed") is False, (
        f"this fixture no longer produces a changed=False refusal, so it is "
        f"not exercising the arm under test: changed={payload.get('changed')!r}"
    )
    assert payload.get("outcome") == "not-started", payload.get("outcome")
    assert "rehearsal" not in payload, (
        "the payload carries rehearsal facts, so this fixture is no longer the "
        "no-rehearsal case the check is about"
    )

    assert _REHEARSAL_SENTENCE not in run.stderr, (
        "a refusal that never rehearsed anything told the operator that a "
        "rehearsal ran on a disposable copy and did not report back — and "
        "diagnosed it as 'a fault in this run and not in your store'. The "
        "payload has no 'rehearsal' key (asserted above) and nothing in the "
        "repository ever passes one, so `payload.get('rehearsal') or {}` is "
        "always {} and `{}.get('changed') is None` is always True. This is the "
        "same absent-is-not-false rule _emit_refusal's own docstring states for "
        "the structured half of the very same payload.\n"
        f"  stderr: {run.stderr!r}"
    )
    assert _NOTHING_CHANGED in run.stderr, (
        "the honest sentence for a not-started refusal is the chain's last "
        "arm, and it never renders — the rehearsal arm above it matches every "
        "payload first, which makes that arm dead code.\n"
        f"  stderr: {run.stderr!r}"
    )


PINS = {
    "check_a_refusal_that_never_rehearsed_does_not_claim_one":
        check_a_refusal_that_never_rehearsed_does_not_claim_one,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_refusal_sentence_property(name, tmp_path):
    """The pin. Asserted against the tree under test."""
    PINS[name](tmp_path)


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_a_refusal_that_never_rehearsed_does_not_claim_one",
        module="hermes_cli/session_fence_rollback_cmd.py",
        find="        elif rehearsal and rehearsal.get(\"changed\") is None:\n",
        replace="        elif rehearsal.get(\"changed\") is None:\n",
        why="without the presence test, an ABSENT rehearsal renders as a "
            "rehearsal that ran and faulted — {}.get('changed') is None is "
            "true for every payload in the tree, so the arm fires on all of "
            "them and the honest last arm never renders at all",
        kills_by="a refusal that never rehearsed anything told the operator",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    """Clean, mutated, restored. A pin that survives its mutation is not a pin."""
    assert_mutation_kills_the_pin(
        mutation, str(_SELF), tmp_path, *_EXTRA_EXTRACT
    )


def test_every_pin_has_a_mutation_that_kills_it():
    """No pin without a killer, and no killer without a pin."""
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)
