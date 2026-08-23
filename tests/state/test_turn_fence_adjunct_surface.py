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
import sqlite3

import pytest

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


@pytest.mark.parametrize(
    "table,operation",
    sorted(ADJUNCT_STATEMENTS),
    ids=lambda value: str(value),
)
def test_a_foreign_writer_cannot_change_an_adjunct_table(
    tmp_path, table, operation
):
    """Rows, not exceptions. One foreign statement per (table, operation).

    The assertion is that the table is byte-for-byte what the owner left, and
    that the foreign connection performed NO changes at all — a refusal that
    happens after a row has moved is not a refusal.
    """
    store, _grant = _live_owned_store(tmp_path, f"{table}-{operation}.db")
    before = _rows(store, table)

    foreign = sqlite3.connect(str(store))
    try:
        try:
            foreign.execute(ADJUNCT_STATEMENTS[(table, operation)])
        except sqlite3.OperationalError as exc:
            outcome = f"refused: {exc}"
        else:
            outcome = "ACCEPTED"
        changes = foreign.total_changes
        foreign.commit()
    finally:
        foreign.close()

    after = _rows(store, table)
    assert after == before, (
        f"a foreign connection with no generation function performed "
        f"{operation} on `{table}` while conversation {SESSION_ID!r} was "
        f"LIVE-OWNED, and the rows moved.\n"
        f"  statement: {ADJUNCT_STATEMENTS[(table, operation)]}\n"
        f"  outcome:   {outcome}\n"
        f"  before:    {before}\n"
        f"  after:     {after}"
    )
    assert changes == 0, (
        f"the foreign {operation} on `{table}` was refused but "
        f"total_changes == {changes}: rows moved before the refusal."
    )
    assert outcome.startswith("refused"), (
        f"the foreign {operation} on `{table}` was {outcome}. `{table}` is "
        f"written inside the same admitted transaction as the transcript, so "
        f"a writer that has never heard of the lease must not reach it."
    )


def test_a_foreign_delete_leaves_the_owner_resuming_with_no_system_prompt(
    tmp_path,
):
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


def test_the_owner_still_writes_every_adjunct_table_normally(tmp_path):
    """A fence that refuses the owner is a different failure, not a fix.

    Every adjunct table, written by its production writer, under the grant the
    owner holds, after the fence is in place.
    """
    from hermes_state import SessionDB

    import tools.async_delegation as ad

    store, grant = _live_owned_store(tmp_path)
    db = SessionDB(store)
    try:
        db.update_system_prompt(
            SESSION_ID, "A SECOND PROMPT", turn_lease_holder=grant
        )
        db.save_gateway_routing_entry(
            "key-1", json.dumps({"session_id": SESSION_ID, "v": 2}),
            scope="probe", turn_lease_holder=grant,
        )

        def _usage(conn):
            db._record_model_usage(
                conn, SESSION_ID, model="owner-model", billing_provider="p",
                billing_base_url="u", billing_mode="b", input_tokens=3,
                output_tokens=5, cache_read_tokens=0, cache_write_tokens=0,
                reasoning_tokens=0, estimated_cost_usd=0.0,
                actual_cost_usd=0.0, cost_status="ok", cost_source="test",
                api_call_count=1,
            )

        db._execute_write(_usage)
    finally:
        db.close()

    previous_path, previous_store = ad._db_path, dict(ad._STORE)
    ad._db_path = lambda: store
    ad._STORE.clear()
    try:
        ad._persist_completion(
            {"delegation_id": "d-1", "status": "completed", "completed_at": 2.0},
            {"ok": True},
        )
        assert ad.mark_completion_delivered("d-1", turn_lease_holder=grant) is True
    finally:
        ad._db_path = previous_path
        ad._STORE.clear()
        ad._STORE.update(previous_store)

    assert _resolved_prompt(store) == "A SECOND PROMPT"
    conn = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        assert conn.execute(
            "SELECT input_tokens FROM session_model_usage "
            "WHERE session_id = ? AND task = ''", (SESSION_ID,)
        ).fetchone()[0] == 10, "the owner's usage delta was lost"
        assert json.loads(conn.execute(
            "SELECT entry_json FROM gateway_routing WHERE scope = 'probe'"
        ).fetchone()[0])["v"] == 2, "the owner's routing write was lost"
        assert conn.execute(
            "SELECT delivery_state FROM async_delegations "
            "WHERE delegation_id = 'd-1'"
        ).fetchone()[0] == "delivered", "the owner's delegation write was lost"
    finally:
        conn.close()
