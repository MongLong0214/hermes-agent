"""The four adjunct tables a foreign writer reached, kept as measurements.

WHAT WAS MEASURED, AND WHY IT IS NOT A BOOKKEEPING RESIDUAL
    One ``sqlite3.connect`` with NO generation function registered, against a
    store this generation created while a conversation is LIVE-OWNED, one write
    per table, the verdict read off the rows::

        table                  in TURN_FENCE_SURFACE   foreign write
        sessions               yes                     refused
        messages               yes                     refused
        system_prompts         no                      ACCEPTED
        session_model_usage    no                      ACCEPTED
        gateway_routing        no                      ACCEPTED
        async_delegations      no                      ACCEPTED

    And the consequence, measured separately, is not "a counter drifted"::

        owner prompt BEFORE : "THE PROMPT THE TURN IS REPLAYING"  hash 4e9cbc79…
        owner prompt AFTER  : None                                hash 4e9cbc79…

    A foreign ``DELETE FROM system_prompts`` removes the BYTES and leaves
    ``sessions.system_prompt_hash`` pointing at them. The schema declares that
    reference — ``FOREIGN KEY (system_prompt_hash) REFERENCES
    system_prompts(hash)`` — and a raw connection has ``PRAGMA foreign_keys``
    off, so the delete is accepted and the store is left with a DANGLING HASH.
    The next turn resolves its system prompt through
    ``LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash`` and gets
    NULL: it resumes with no system prompt, and nothing anywhere raises.

WHY "REFUSED" IS NOT THE ASSERTION
    "The statement raised" and "the prompt still resolves" are different claims
    and only the second is the property. So every case below asserts on ROWS —
    the full contents of the table before and after — and the system-prompt case
    additionally asserts the hash/bytes relationship through the production
    reader, plus ``PRAGMA foreign_key_check``, which is the schema's own
    statement of the integrity that a dangling hash breaks.

ROOT CAUSE OF THE OMISSION
    ``derive_turn_fence_surface`` seeded from ``derive_context_bearing_mutators``
    — the MESSAGE-table derivation — and then collected the tables those methods
    touch. ``_store_system_prompt`` writes ``system_prompts`` and never touches
    ``messages``, so it never entered the seed, and rule 2 of that derivation
    adds CALLERS of derived members, not CALLEES. The surface was therefore
    everything reachable from the transcript and nothing reachable only from the
    session row. That is an omission, not a decision, which is why this file
    states the counterexamples rather than an exemption.
"""

from __future__ import annotations

import json
import os
import pathlib
import sqlite3

import pytest

from tests.state.lease_mutation_harness import (
    Mutation,
    assert_every_pin_has_a_killer,
    assert_mutation_kills_the_pin,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)
#: The pins drive `tools/async_delegation` and the store; take the whole tree
#: rather than naming the modules they happen to import today.
_EXTRA_EXTRACT = (".",)

#: The conversation every fixture row belongs to, and the one that is owned.
SESSION_ID = "s"

#: The bytes the owner's next turn replays under.
OWNER_PROMPT = "THE PROMPT THE TURN IS REPLAYING"

#: One foreign statement per ``(table, operation)``, each one the schema would
#: otherwise accept. A statement the schema rejects proves nothing about a
#: trigger, so every INSERT below names a full, legal row.
ADJUNCT_STATEMENTS = {
    ("system_prompts", "INSERT"):
        "INSERT INTO system_prompts (hash, prompt) "
        "VALUES ('smuggled-hash', 'FOREIGN PROMPT')",
    ("system_prompts", "UPDATE"):
        "UPDATE system_prompts SET prompt = 'FOREIGN PROMPT'",
    ("system_prompts", "DELETE"):
        "DELETE FROM system_prompts",
    ("session_model_usage", "INSERT"):
        "INSERT INTO session_model_usage "
        "(session_id, model, billing_provider, billing_base_url, "
        " billing_mode, task, api_call_count, input_tokens) "
        f"VALUES ('{SESSION_ID}', 'FOREIGN-MODEL', '', '', '', '', 1, 1)",
    ("session_model_usage", "UPDATE"):
        "UPDATE session_model_usage SET input_tokens = 999999, "
        f"api_call_count = 999999 WHERE session_id = '{SESSION_ID}'",
    ("session_model_usage", "DELETE"):
        f"DELETE FROM session_model_usage WHERE session_id = '{SESSION_ID}'",
    ("gateway_routing", "INSERT"):
        "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) "
        "VALUES ('probe', 'smuggled-key', '{\"session_id\": \"FOREIGN\"}', 1.0)",
    ("gateway_routing", "UPDATE"):
        "UPDATE gateway_routing SET entry_json = '{\"session_id\": \"FOREIGN\"}' "
        "WHERE scope = 'probe'",
    ("gateway_routing", "DELETE"):
        "DELETE FROM gateway_routing WHERE scope = 'probe'",
    ("async_delegations", "INSERT"):
        "INSERT INTO async_delegations "
        "(delegation_id, origin_session, origin_ui_session_id, "
        " parent_session_id, state, dispatched_at, updated_at, delivery_state) "
        f"VALUES ('smuggled-d', '{SESSION_ID}', '{SESSION_ID}', "
        f"'{SESSION_ID}', 'running', 1.0, 1.0, 'pending')",
    ("async_delegations", "UPDATE"):
        "UPDATE async_delegations SET delivery_state = 'delivered', "
        "delivery_claim = 'FOREIGN' WHERE delegation_id = 'd-1'",
    ("async_delegations", "DELETE"):
        "DELETE FROM async_delegations WHERE delegation_id = 'd-1'",
}

ADJUNCT_TABLES = tuple(sorted({table for table, _op in ADJUNCT_STATEMENTS}))


def _live_owned_store(tmp_path, name="state.db"):
    """A store with a row in every adjunct table and a LIVE lease on ``s``.

    Every row is written through the production writer that owns it, so the
    fixture is the shape a running turn leaves behind rather than a hand-built
    one that might not be reachable.
    """
    from hermes_state import SessionDB

    tmp_path = pathlib.Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = tmp_path / name
    db = SessionDB(store)
    db.create_session(SESSION_ID, source="test", system_prompt=OWNER_PROMPT)
    grant = db.try_acquire_session_turn_lease(
        SESSION_ID, f"pid={os.getpid()}:turn=live:platform=test", ttl_seconds=600
    )
    assert grant, "could not take the lease this measurement depends on"
    db.append_message(
        session_id=SESSION_ID, role="user", content="current",
        turn_lease_holder=grant,
    )
    db.save_gateway_routing_entry(
        "key-1", json.dumps({"session_id": SESSION_ID}), scope="probe",
        turn_lease_holder=grant,
    )

    def _usage(conn):
        db._record_model_usage(
            conn, SESSION_ID, model="owner-model", billing_provider="p",
            billing_base_url="u", billing_mode="b", input_tokens=7,
            output_tokens=11, cache_read_tokens=0, cache_write_tokens=0,
            reasoning_tokens=0, estimated_cost_usd=0.0, actual_cost_usd=0.0,
            cost_status="ok", cost_source="test", api_call_count=1,
        )

    db._execute_write(_usage)
    db.close()

    _seed_delegation(store)
    return store, grant


def _seed_delegation(store):
    """One durable delegation row, written by the module that owns the table."""
    import tools.async_delegation as ad

    previous_path, previous_store = ad._db_path, dict(ad._STORE)
    ad._db_path = lambda: store
    ad._STORE.clear()
    try:
        ad._persist_dispatch({
            "delegation_id": "d-1",
            "session_key": SESSION_ID,
            "origin_ui_session_id": SESSION_ID,
            "parent_session_id": SESSION_ID,
            "origin_session_id": SESSION_ID,
            "dispatched_at": 1.0,
        })
    finally:
        ad._db_path = previous_path
        ad._STORE.clear()
        ad._STORE.update(previous_store)


def _rows(store, table):
    conn = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        return sorted(tuple(row) for row in conn.execute(f'SELECT * FROM "{table}"'))
    finally:
        conn.close()


def _resolved_prompt(store):
    """What the next turn would replay under, read the way production reads it."""
    from hermes_state import SessionDB

    db = SessionDB(store)
    try:
        row = db.get_session(SESSION_ID)
        return (row or {}).get("system_prompt")
    finally:
        db.close()


def _stored_hash(store):
    conn = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT system_prompt_hash FROM sessions WHERE id = ?", (SESSION_ID,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _check_one_table_refuses_a_foreign_writer(tmpdir, table):
    """Rows, not exceptions. Every operation on one adjunct table.

    The assertion is that the table is byte-for-byte what the owner left, and
    that the foreign connection performed NO changes at all — a refusal that
    happens after a row has moved is not a refusal.
    """
    tmpdir = pathlib.Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    findings, exercised = [], 0
    for (candidate, operation), sql in sorted(ADJUNCT_STATEMENTS.items()):
        if candidate != table:
            continue
        exercised += 1
        store, _grant = _live_owned_store(tmpdir, f"{table}-{operation}.db")
        before = _rows(store, table)

        foreign = sqlite3.connect(str(store))
        try:
            try:
                foreign.execute(sql)
            except sqlite3.OperationalError as exc:
                outcome = f"refused: {exc}"
            else:
                outcome = "ACCEPTED"
            changes = foreign.total_changes
            foreign.commit()
        finally:
            foreign.close()

        after = _rows(store, table)
        if after != before:
            findings.append(
                f"{operation}: the rows MOVED\n"
                f"      statement: {sql}\n"
                f"      outcome:   {outcome}\n"
                f"      before:    {before}\n"
                f"      after:     {after}"
            )
        elif changes:
            findings.append(
                f"{operation}: refused, but total_changes == {changes} — rows "
                f"moved before the refusal"
            )
        elif outcome == "ACCEPTED":
            findings.append(f"{operation}: ACCEPTED ({sql})")
    assert exercised, (
        f"no statement in ADJUNCT_STATEMENTS names `{table}`, so this check "
        f"exercised nothing and would pass whatever the fence did"
    )
    assert not findings, (
        f"a foreign connection with no generation function reached `{table}` "
        f"while conversation {SESSION_ID!r} was LIVE-OWNED. `{table}` is "
        f"written inside the same admitted transaction as the transcript, so a "
        f"writer that has never heard of the lease must not reach it:\n    "
        + "\n    ".join(findings)
    )


def check_system_prompts_refuses_a_foreign_writer(tmpdir) -> None:
    _check_one_table_refuses_a_foreign_writer(tmpdir, "system_prompts")


def check_session_model_usage_refuses_a_foreign_writer(tmpdir) -> None:
    _check_one_table_refuses_a_foreign_writer(tmpdir, "session_model_usage")


def check_gateway_routing_refuses_a_foreign_writer(tmpdir) -> None:
    _check_one_table_refuses_a_foreign_writer(tmpdir, "gateway_routing")


def check_async_delegations_refuses_a_foreign_writer(tmpdir) -> None:
    _check_one_table_refuses_a_foreign_writer(tmpdir, "async_delegations")


def check_the_owners_system_prompt_survives_a_foreign_delete(tmp_path) -> None:
    """The dangling hash. This is the provider-visible half of the defect.

    ``DELETE FROM system_prompts`` names neither ``sessions`` nor ``messages``,
    so no fence this branch had installed prepares against it. It removes the
    BYTES; ``sessions.system_prompt_hash`` still names them. The measurement
    that forced this slice, verbatim::

        owner prompt BEFORE : "THE PROMPT THE TURN IS REPLAYING"  hash 4e9cbc79…
        owner prompt AFTER  : None                                hash 4e9cbc79…

    Asserted three ways, because "refused" would be satisfied by any of them
    alone and the property is all three: the resolved prompt through the
    production reader, the hash still resolving to the bytes, and the schema's
    own ``FOREIGN KEY (system_prompt_hash) REFERENCES system_prompts(hash)``
    being satisfied — ``PRAGMA foreign_key_check`` is that statement, and a
    dangling hash is exactly what it reports.
    """
    store, _grant = _live_owned_store(tmp_path)
    prompt_before, hash_before = _resolved_prompt(store), _stored_hash(store)
    assert prompt_before == OWNER_PROMPT, "the fixture did not store the prompt"
    assert hash_before, "the fixture did not store a prompt hash"

    foreign = sqlite3.connect(str(store))
    try:
        try:
            foreign.execute("DELETE FROM system_prompts")
        except sqlite3.OperationalError as exc:
            outcome = f"refused: {exc}"
        else:
            outcome = "ACCEPTED"
        foreign.commit()
    finally:
        foreign.close()

    prompt_after, hash_after = _resolved_prompt(store), _stored_hash(store)

    checker = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        dangling = checker.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        checker.close()

    assert prompt_after == OWNER_PROMPT, (
        f"the owner's system prompt is gone while its conversation is "
        f"LIVE-OWNED. The next turn resumes with no system prompt and nothing "
        f"raises anywhere.\n"
        f"  BEFORE : {prompt_before!r}  hash {hash_before}\n"
        f"  AFTER  : {prompt_after!r}  hash {hash_after}\n"
        f"  foreign DELETE: {outcome}"
    )
    assert hash_after == hash_before, (
        f"the session's prompt hash moved: {hash_before} -> {hash_after}"
    )
    assert not dangling, (
        f"`sessions.system_prompt_hash` no longer resolves: the schema "
        f"declares FOREIGN KEY (system_prompt_hash) REFERENCES "
        f"system_prompts(hash) and PRAGMA foreign_key_check reports "
        f"{dangling}. A raw connection has foreign_keys OFF, so the delete "
        f"was accepted and the store is left with a dangling hash."
    )
    assert outcome.startswith("refused"), (
        f"the foreign DELETE FROM system_prompts was {outcome}"
    )


def check_the_owner_still_writes_every_adjunct_table_normally(tmp_path) -> None:
    """A fence that refuses the owner is a different failure, not a fix.

    Every adjunct table, written by its production writer, under the grant the
    owner holds, after the fence is in place.

    Refusals are CAUGHT and turned into values rather than allowed to escape,
    so this reads as "the owner's write did not land, and here is what happened
    instead" rather than as a traceback. A pin that dies by raising cannot be
    told apart from a pin that died because the fixture broke.
    """
    from hermes_state import SessionDB, SessionTurnLeaseLostError

    import tools.async_delegation as ad

    store, grant = _live_owned_store(tmp_path)
    outcomes = {}

    def _record(label, call):
        try:
            outcomes[label] = call()
        except SessionTurnLeaseLostError as exc:
            outcomes[label] = f"REFUSED THE OWNER: {exc}"

    db = SessionDB(store)
    try:
        _record("system_prompt", lambda: db.update_system_prompt(
            SESSION_ID, "A SECOND PROMPT", turn_lease_holder=grant
        ))
        _record("gateway_routing", lambda: db.save_gateway_routing_entry(
            "key-1", json.dumps({"session_id": SESSION_ID, "v": 2}),
            scope="probe", turn_lease_holder=grant,
        ))

        def _usage(conn):
            db._record_model_usage(
                conn, SESSION_ID, model="owner-model", billing_provider="p",
                billing_base_url="u", billing_mode="b", input_tokens=3,
                output_tokens=5, cache_read_tokens=0, cache_write_tokens=0,
                reasoning_tokens=0, estimated_cost_usd=0.0,
                actual_cost_usd=0.0, cost_status="ok", cost_source="test",
                api_call_count=1,
            )

        _record("session_model_usage", lambda: db._execute_write(_usage))
    finally:
        db.close()

    previous_path, previous_store = ad._db_path, dict(ad._STORE)
    ad._db_path = lambda: store
    ad._STORE.clear()
    try:
        _record("delegation_completion", lambda: ad._persist_completion(
            {"delegation_id": "d-1", "status": "completed", "completed_at": 2.0},
            {"ok": True},
        ))
        _record("delegation_delivery", lambda: ad.mark_completion_delivered(
            "d-1", turn_lease_holder=grant
        ))
    finally:
        ad._db_path = previous_path
        ad._STORE.clear()
        ad._STORE.update(previous_store)

    refused = sorted(
        label for label, value in outcomes.items()
        if isinstance(value, str) and value.startswith("REFUSED THE OWNER")
    )
    assert not refused, (
        f"the fence refused the conversation's OWN holder on {refused}. That "
        f"is a different failure from the one this file exists for, not a "
        f"fix for it.\n  {json.dumps(outcomes, default=str, indent=4)}"
    )
    assert outcomes.get("delegation_delivery") is True, (
        f"the owner's delivery acknowledgement did not land: "
        f"{outcomes.get('delegation_delivery')!r}"
    )

    landed = {"system_prompt": _resolved_prompt(store)}
    conn = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        landed["session_model_usage"] = conn.execute(
            "SELECT input_tokens FROM session_model_usage "
            "WHERE session_id = ? AND task = ''", (SESSION_ID,)
        ).fetchone()[0]
        landed["gateway_routing"] = json.loads(conn.execute(
            "SELECT entry_json FROM gateway_routing WHERE scope = 'probe'"
        ).fetchone()[0])["v"]
        landed["async_delegations"] = conn.execute(
            "SELECT delivery_state FROM async_delegations "
            "WHERE delegation_id = 'd-1'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert landed == {
        "system_prompt": "A SECOND PROMPT",
        "session_model_usage": 10,
        "gateway_routing": 2,
        "async_delegations": "delivered",
    }, (
        f"the owner's own writes did not all land under the widened fence: "
        f"{json.dumps(landed, default=str, sort_keys=True)}"
    )


def check_a_current_generation_marker_is_not_authority(tmp_path) -> None:
    """A current-generation marker alone never creates authority.

    The generation trigger asks ONE question — did this connection register the
    marker — and every connection this package opens registers it. So a second
    process running THIS build is admitted by the trigger on all four adjunct
    tables and must still be refused, by the token validator, when it does not
    hold the conversation's grant. If it were not, widening the surface would
    have bought mixed-version safety and nothing else, and the fence would read
    as a permission it is not.

    The assertions are the three the atomicity family uses, because "it raised"
    is satisfied by a refusal that has already spent something:

        total_changes   across the refused call on the writer's OWN connection.
                        Zero, or the refusal wrote on the way to refusing.
        the rows        every adjunct table, byte-equal before and after.
        deep state      ``tools.async_delegation._records``, the in-process
                        registry the refused delivery ack would have moved.
    """
    from hermes_state import SessionDB, SessionTurnLeaseLostError

    import tools.async_delegation as ad

    store, _owner_grant = _live_owned_store(tmp_path)
    before = {table: _rows(store, table) for table in ADJUNCT_TABLES}
    records_before = dict(ad._records)

    bystander = SessionDB(store)
    refusals = {}
    changes = {}
    try:
        # No grant at all, and a grant for a DIFFERENT conversation: both are
        # "not the holder of this one", and only the first was ever measured.
        bystander.create_session("other", source="test")
        foreign_grant = bystander.try_acquire_session_turn_lease(
            "other", f"pid={os.getpid()}:turn=bystander:platform=test",
            ttl_seconds=600,
        )
        assert foreign_grant, "the bystander could not take its own lease"

        attempts = {
            "system_prompt (no grant)":
                lambda: bystander.update_system_prompt(SESSION_ID, "FOREIGN"),
            "system_prompt (wrong conversation's grant)":
                lambda: bystander.update_system_prompt(
                    SESSION_ID, "FOREIGN", turn_lease_holder=foreign_grant
                ),
            "gateway_routing (no grant)":
                lambda: bystander.save_gateway_routing_entry(
                    "key-1", json.dumps({"session_id": SESSION_ID, "v": 99}),
                    scope="probe",
                ),
            "gateway_routing (wrong conversation's grant)":
                lambda: bystander.save_gateway_routing_entry(
                    "key-1", json.dumps({"session_id": SESSION_ID, "v": 99}),
                    scope="probe", turn_lease_holder=foreign_grant,
                ),
        }
        for label, call in attempts.items():
            started = bystander._conn.total_changes
            try:
                call()
            except SessionTurnLeaseLostError as exc:
                refusals[label] = f"refused: {exc}"
            else:
                refusals[label] = "ACCEPTED"
            changes[label] = bystander._conn.total_changes - started
    finally:
        bystander.close()

    previous_path, previous_store = ad._db_path, dict(ad._STORE)
    ad._db_path = lambda: store
    ad._STORE.clear()
    try:
        delegation_store = ad._session_store()
        started = delegation_store._conn.total_changes
        label = "async_delegations delivery ack (no grant)"
        try:
            ad.mark_completion_delivered("d-1")
        except SessionTurnLeaseLostError as exc:
            refusals[label] = f"refused: {exc}"
        else:
            refusals[label] = "ACCEPTED"
        changes[label] = delegation_store._conn.total_changes - started
    finally:
        ad._db_path = previous_path
        ad._STORE.clear()
        ad._STORE.update(previous_store)

    after = {table: _rows(store, table) for table in ADJUNCT_TABLES}
    moved = sorted(t for t in ADJUNCT_TABLES if after[t] != before[t])
    assert not moved, (
        f"a CURRENT-GENERATION writer holding no grant for {SESSION_ID!r} "
        f"moved rows in {moved}. The generation marker admits it at the "
        f"trigger — every connection this build opens registers it — so the "
        f"token validator is the only thing between a bystander process and "
        f"these rows.\n"
        f"  outcomes: {json.dumps(refusals, indent=4, sort_keys=True)}\n"
        + "\n".join(
            f"  {t}\n    before {before[t]}\n    after  {after[t]}"
            for t in moved
        )
    )
    spent = {label: n for label, n in changes.items() if n}
    assert not spent, (
        f"these refused calls changed rows on the way to being refused: "
        f"{spent}. total_changes must be 0 across a refusal — a guard that "
        f"runs after the first statement is not a guard.\n"
        f"  outcomes: {json.dumps(refusals, indent=4, sort_keys=True)}"
    )
    assert dict(ad._records) == records_before, (
        "the refused delivery acknowledgement moved the in-process delegation "
        "registry. Deep in-memory state has to be invariant across a refusal "
        "for the same reason the rows do: a caller that catches the refusal "
        "and retries is otherwise working from state the refusal already spent"
    )
    accepted = sorted(k for k, v in refusals.items() if v == "ACCEPTED")
    assert not accepted, (
        f"these writes were ACCEPTED from a process that holds no grant for "
        f"{SESSION_ID!r}: {accepted}\n"
        f"  outcomes: {json.dumps(refusals, indent=4, sort_keys=True)}"
    )


PINS = {
    "check_system_prompts_refuses_a_foreign_writer":
        check_system_prompts_refuses_a_foreign_writer,
    "check_session_model_usage_refuses_a_foreign_writer":
        check_session_model_usage_refuses_a_foreign_writer,
    "check_gateway_routing_refuses_a_foreign_writer":
        check_gateway_routing_refuses_a_foreign_writer,
    "check_async_delegations_refuses_a_foreign_writer":
        check_async_delegations_refuses_a_foreign_writer,
    "check_the_owners_system_prompt_survives_a_foreign_delete":
        check_the_owners_system_prompt_survives_a_foreign_delete,
    "check_the_owner_still_writes_every_adjunct_table_normally":
        check_the_owner_still_writes_every_adjunct_table_normally,
    "check_a_current_generation_marker_is_not_authority":
        check_a_current_generation_marker_is_not_authority,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_adjunct_surface_property(name, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    PINS[name](tmp_path / "work")


#: Each row removes ONE thing from production and requires the pin to fail by
#: its own assertion. Every anchor is an exact source substring that must match
#: once — a line number would go stale silently, and silently is the only way
#: this table can lie.
SOURCE_MUTATIONS = (
    Mutation(
        pin="check_system_prompts_refuses_a_foreign_writer",
        module="hermes_state_common.py",
        find='        "system_prompts",\n',
        replace="",
        why="the table drops off TURN_FENCE_SURFACE, so no trigger is created "
            "for it and a connection with no generation marker writes the "
            "prompt bytes the next turn replays",
    ),
    Mutation(
        pin="check_session_model_usage_refuses_a_foreign_writer",
        module="hermes_state_common.py",
        find='        "session_model_usage",\n',
        replace="",
        why="the per-model accounting the turn is billed and routed on becomes "
            "writable by a process that has never heard of the lease",
    ),
    Mutation(
        pin="check_gateway_routing_refuses_a_foreign_writer",
        module="hermes_state_common.py",
        find='        "gateway_routing",\n',
        replace="",
        why="the routing index decides which conversation a platform reply "
            "lands in; unfenced, a foreign writer redirects a live one",
    ),
    Mutation(
        pin="check_async_delegations_refuses_a_foreign_writer",
        module="hermes_state_common.py",
        find='        "async_delegations",\n',
        replace="",
        why="delivery_state and delivery_claim decide who may deliver a "
            "subagent's result into a turn; unfenced, anybody can",
    ),
    Mutation(
        pin="check_the_owners_system_prompt_survives_a_foreign_delete",
        module="hermes_state_common.py",
        find='    for operation in ("INSERT", "UPDATE", "DELETE")\n',
        replace='    for operation in ("INSERT", "UPDATE")\n',
        why="a fence keyed per operation with DELETE left off is a fence with "
            "the other door open: DELETE FROM system_prompts removes the bytes "
            "and leaves sessions.system_prompt_hash pointing at them, which is "
            "the dangling hash this pin measures",
    ),
    Mutation(
        pin="check_the_owner_still_writes_every_adjunct_table_normally",
        module="hermes_state.py",
        find=(
            "        named = next(\n"
            "            (sid for sid in ids\n"
            "             if self._session_turn_lease_key_on_conn(conn, sid) "
            "== granted_root),\n"
            "            None,\n"
            "        )\n"
        ),
        replace="        named = None\n",
        why="without resolving WHICH conversation the caller's grant names, "
            "the borrowed admission treats the holder's own conversation as a "
            "bystander's and refuses the owner — the failure this pin exists "
            "to tell apart from a working fence",
    ),
    Mutation(
        pin="check_a_current_generation_marker_is_not_authority",
        module="hermes_state.py",
        find=(
            "            self._check_turn_lease_guard(\n"
            "                conn,\n"
            "                session_id,\n"
            "                turn_lease_holder,\n"
            "                turn_lease_ttl_seconds=turn_lease_ttl_seconds,\n"
            "            )\n"
            "            system_prompt_hash = self._store_system_prompt("
            "conn, system_prompt)\n"
        ),
        replace=(
            "            system_prompt_hash = self._store_system_prompt("
            "conn, system_prompt)\n"
        ),
        why="the trigger admits every connection this build opens, so the "
            "token validator is the only thing standing between a bystander "
            "PROCESS OF THIS GENERATION and the prompt the owner's next turn "
            "replays under",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    assert_mutation_kills_the_pin(mutation, str(_SELF), tmp_path, *_EXTRA_EXTRACT)


def test_every_pin_has_a_mutation_that_kills_it():
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)
