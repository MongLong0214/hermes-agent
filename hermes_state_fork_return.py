"""Return this build's fenced state.db (1033) to the live fork's schema (1030), as a new file.

The inverse of :func:`hermes_state_fence.migrate_fenced_store`. The source is never written: it is read
immutably (or through a byte copy when a WAL still holds pages the main file lacks), and the result is
built in a second byte copy that is published under the output path only after every check passed.

Everything the fork's schema has is kept byte for byte, rowids included: sessions and messages, the
fork's session-process authorities and their append-only event log, turn leases, conversation
generations, routing, delegations, delivery obligations, state_meta. What upstream schema 31-33 added
leaves the store with a stated reason (``converted``), column by column. Work this build accepted and
has not finished (a queued or started admission, a registered or running worker) has no representation
in the fork, so the export refuses rather than drop it. A terminal admission is input, not bookkeeping:
one that ran must find its text as a kept user message of its sessions, and one that never ran (cancelled)
exists nowhere else, so either refuses as a lossless return that is impossible. So does any object,
column or value this module has no rule for.

Loss is measured, not assumed: after the conversion every kept table is compared row by row against
the source on every column the result has, every removed table, column and state_meta key is counted
against its disposition, and an unaccounted difference refuses the export with nothing published.

Paths are checked before anything is written: the output and the report are new files, and neither may
be, alias or link to the source, each other or any of their SQLite files. Working copies live in
directories created for this run alone and are removed on any exit; the output is published with an
exclusive link, so a file that appears there meanwhile is never replaced.

Copying the canonical tables into a fresh fork schema, as the fork's own ``sessions recover`` does, is
ruled out: it leaves the authorities, their events, turn leases and conversation generations behind
and still reports no loss. This build's tables are removed rather than kept beside the fork's tables:
the fork writes with foreign keys on, and their RESTRICT keys to sessions would refuse its deletes.

    python -m hermes_state_fork_return SOURCE OUTPUT [--work-dir DIR] [--report PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Optional

import hermes_state_fence as fence

FORK_STORED_SCHEMA_VERSION = fence.FORK_STORED_SCHEMA_VERSION

# The fork's state.db objects (fork build 420c58a13e: a fresh store with its lazily created tables, plus the
# legacy session-process objects and shapes a long-lived fork store still carries). Fence triggers, sqlite_
# internals and FTS5 shadow tables are not listed; they follow their tables.
_FORK_TABLES = {
    "async_delegations": (
        "delegation_id origin_session origin_ui_session_id parent_session_id state dispatched_at "
        "completed_at updated_at event_json result_json delivery_state delivery_attempts delivered_at "
        "owner_pid owner_started_at task_json delivery_claim delivery_claimed_at origin_session_id"
    ),
    "compression_locks": "session_id holder acquired_at expires_at",
    "conversation_generations": "source session_key generation",
    "delivery_obligations": (
        "obligation_id session_key platform chat_id thread_id content state attempts created_at "
        "updated_at owner_pid owner_started_at last_error adapter_profile runtime_claim_token "
        "acp_update_id delivered_message_ids"
    ),
    "gateway_heartbeats": "backend_id pid started_at last_heartbeat profile host",
    "gateway_hygiene_state": "session_key failure_streak",
    "gateway_routing": "scope session_key entry_json updated_at",
    "messages": (
        "id session_id role content tool_call_id tool_calls tool_name effect_disposition timestamp "
        "token_count finish_reason reasoning reasoning_content reasoning_details codex_reasoning_items "
        "codex_message_items platform_message_id observed active compacted api_content display_kind "
        "display_metadata _compressed_summary display_identity display_order"
    ),
    "messages_fts": "content tool_name tool_calls messages_fts rank",
    "messages_fts_trigram": "content tool_name tool_calls messages_fts_trigram rank",
    "schema_version": "version",
    "session_model_usage": (
        "session_id model billing_provider billing_base_url billing_mode task api_call_count input_tokens "
        "output_tokens cache_read_tokens cache_write_tokens reasoning_tokens estimated_cost_usd "
        "actual_cost_usd cost_status cost_source first_seen last_seen"
    ),
    "session_process_authorities": (
        "session_id session_generation state_db_id state_family authority_token status issued_at "
        "terminal_at"
    ),
    "session_process_authority_events": (
        "id session_id session_generation state_db_id state_family event_type reservation_id occurred_at"
    ),
    "session_process_reservations": (
        "reservation_id reservation_token_sha256 session_id session_generation state_db_id state_family "
        "status reserved_at expires_at consumed_at process_pid process_started_at process_argv_json "
        "process_purpose"
    ),
    "session_turn_leases": "conversation_id holder acquired_at expires_at owner_pid owner_pid_start",
    "sessions": (
        "id source user_id session_key chat_id chat_type thread_id display_name origin_json "
        "expiry_finalized model model_config system_prompt system_prompt_hash parent_session_id "
        "started_at ended_at end_reason message_count tool_call_count input_tokens output_tokens "
        "cache_read_tokens cache_write_tokens reasoning_tokens cwd git_branch git_repo_root "
        "billing_provider billing_base_url billing_mode estimated_cost_usd actual_cost_usd cost_status "
        "cost_source pricing_version title title_source last_activity_at last_activity_description "
        "last_activity_provenance api_call_count handoff_state handoff_platform handoff_error "
        "compression_failure_cooldown_until compression_failure_error compression_fallback_streak "
        "compression_ineffective_count profile_name rewind_count archived pinned last_read_at "
        "git_metadata_generation hidden session_generation compression_recovery_deadline "
        "transport_profile tool_names"
    ),
    "state_meta": "key value",
    "system_prompts": "hash prompt",
    "telegram_dm_topic_bindings": (
        "profile_name chat_id thread_id user_id session_key session_id managed_mode linked_at updated_at"
    ),
    "telegram_dm_topic_mode": (
        "profile_name chat_id user_id enabled activated_at updated_at has_topics_enabled "
        "allows_users_to_create_topics capability_checked_at intro_message_id pinned_message_id"
    ),
    "turn_receipts": (
        "turn_request_id session_id binding_digest receipt_identity_json receipt_identity_digest "
        "target_bind_receipt_json target_bind_receipt_digest status claim_token terminal_message_id "
        "response_digest abort_receipt_id abort_evidence_digest abort_reason_code created_at claimed_at "
        "completed_at aborted_at"
    ),
}
_FORK_INDEXES = frozenset((
    "idx_async_delegations_delivery idx_compression_locks_expires idx_messages_active_null "
    "idx_messages_assistant_calls_by_session idx_messages_display_backfill "
    "idx_messages_display_identity idx_messages_display_page idx_messages_platform_msg_id "
    "idx_messages_session idx_messages_session_active idx_messages_session_id "
    "idx_session_model_usage_model idx_session_model_usage_session "
    "idx_session_process_authorities_current idx_session_process_reservations_current "
    "idx_session_turn_leases_expires idx_sessions_effective_activity idx_sessions_gateway_peer "
    "idx_sessions_handoff_state idx_sessions_parent idx_sessions_session_key idx_sessions_source "
    "idx_sessions_source_id idx_sessions_started idx_sessions_system_prompt_hash "
    "idx_sessions_title_unique idx_sessions_tool_names idx_telegram_dm_topic_bindings_session "
    "idx_telegram_dm_topic_bindings_user idx_turn_receipts_completed_prune "
    "idx_turn_receipts_session_status"
).split())
_FORK_TRIGGERS = frozenset((
    "messages_display_identity_delete messages_display_identity_update messages_display_order_insert "
    "messages_display_visibility_update messages_fts_delete messages_fts_insert "
    "messages_fts_trigram_delete messages_fts_trigram_insert messages_fts_trigram_update "
    "messages_fts_update session_process_authority_events_append_only_delete "
    "session_process_authority_events_append_only_update "
    "session_process_authority_generation_insert_guard "
    "session_process_authority_generation_monotonic_guard "
    "session_process_authority_identity_immutable_delete "
    "session_process_authority_identity_immutable_update session_process_authority_sessions_close "
    "session_process_authority_sessions_issue session_process_authority_sessions_reopen "
    "session_process_reservation_issue session_process_reservation_transition_event "
    "session_process_reservation_transition_guard"
).split())
_FORK_VIEWS = frozenset(("messages_fts_src", "messages_fts_trigram_src"))
_FTS5_SHADOW_SUFFIXES = ("_config", "_content", "_data", "_docsize", "_idx")

# What upstream schema 31-33 added, and why the fork's representation of each is its absence. Every column of
# these tables needs a rule here; a column without one refuses the export, set or not.
_RUNTIME_TABLES = {
    "worker_receipts": "replay answers of terminal workers' steps; the steps' effects are rows of kept tables",
    "worker_executions": "terminal workers of this build's runtime; their effects are rows of kept tables",
    "session_admissions": "terminal input whose run's user message is a kept row (matched, not assumed)",
    "runtime_epoch": "this build's runtime owner epoch; the fork has no epoch-fenced owner",
}
_RETRY = "exact-retry identity this build's ledger answers; the fork keeps no admission ledger"
_FENCE = "runtime fence of a finished claim; no unfinished work reads it"
_RUNTIME_COLUMNS = {
    "runtime_epoch": {"singleton": "the epoch row's key", "epoch": _FENCE, "instance_id": _FENCE},
    "session_admissions": {
        "seq": "FIFO position; every row is terminal, so no queue is left", "admission_id": _RETRY,
        "request_id": _RETRY, "principal_id": _RETRY, "payload_digest": _RETRY,
        "target_session_id": "the kept session the input was matched in",
        "lineage_json": "the kept sessions the input was matched in",
        "payload_json": "the input: its text is a kept user message of those sessions",
        "intent": "queue, steer or redirect of input that already ran",
        "status": "terminal", "outcome": "how the run ended; the run's transcript is kept",
        "owner_epoch": _FENCE, "generation": _FENCE,
    },
    "worker_executions": {
        "execution_id": "the worker's key", "session_id": "the kept session it worked for",
        "kind": "cron, child, compute or kanban", "status": "terminal", "adoption_digest": _FENCE,
        "owner_epoch": _FENCE, "generation": _FENCE, "last_sequence": "the last step it applied",
    },
    "worker_receipts": {
        "execution_id": "the terminal worker it answered", "sequence": "the step it answered",
        "payload_digest": "fingerprint of the step a late duplicate must match",
        "result_json": "the answer a late duplicate replays; the step's effect is kept",
    },
}
# Unfinished work has no representation in the fork; values outside these sets have no rule either.
_UNFINISHED = {"session_admissions": ("queued", "started", "unknown"),
               "worker_executions": ("registered", "running", "unknown")}
_RUNTIME_VALUES = {
    "session_admissions": {"status": ("terminal",), "intent": ("queue", "steer", "redirect")},
    "worker_executions": {"status": ("terminal",), "kind": ("cron", "child", "compute", "kanban")},
}
# Terminal admission outcomes. A run handed the input to an agent, so its user message must be a kept row; an
# input that never ran exists nowhere but its admission, and the fork has no place that gives it meaning.
_ADMISSION_RAN = ("completed", "interrupted", "failed", "rejected", "steered")
_ADMISSION_NEVER_RAN = {"cancelled": "cancelled before it ran", "steering": "handed to a turn that never settled it"}
_COLUMNS = {
    "async_delegations": {
        "owner_execution_id": "execution fence of a terminal worker; owner_pid/owner_started_at are kept",
    },
    "messages": {
        "message_uid": "row identity this build mints; the fork keys rows by id, which is kept",
        "absorbed_message_uids": "uids folded into this row for display dedupe; their content is kept",
        "tool_call_uid": "tool-call identity this build mints; tool_call_id is kept",
        "tool_call_uids": "tool-call identities this build mints; tool_calls is kept",
    },
    "sessions": {
        "compression_overload_streak": "this build's compression backoff counter",
        "runtime_revision": "this build's runtime CAS counter; no unfinished worker reads it",
        "runtime_generation": "this build's runtime generation; no unfinished worker reads it",
        "transcript_revision": "cache key this build derives from message writes",
        "created_source": "listing label for local sessions; source is kept",
        "auto_archived": "sweep provenance of archived; archived is kept",
    },
}
_TRIGGERS = (
    "messages_display_revision_update", "messages_message_uid_insert", "messages_transcript_revision_update",
    "sessions_runtime_ledger_delete_guard",
)
_INDEXES = ("idx_messages_session_uid", "idx_sessions_local_created_started")


def _meta_keys() -> dict:
    from hermes_state_legacy_refusal import _PENDING, _SCAN_COMPLETE, _SCAN_CURSOR, LEGACY_REFUSAL_REPAIR_DONE
    from hermes_state_schema import _MESSAGE_UID_BACKFILL_CURSOR, _MESSAGE_UID_BACKFILL_DONE

    uid = "progress of the message_uid backfill; the column leaves, so a later forward migration redoes it"
    refusal = ("progress of the legacy-refusal repair; the fork can write the old shape again, so a later "
               "forward migration rescans")
    return {
        _MESSAGE_UID_BACKFILL_DONE: uid, _MESSAGE_UID_BACKFILL_CURSOR: uid,
        LEGACY_REFUSAL_REPAIR_DONE: refusal, _SCAN_CURSOR: refusal, _SCAN_COMPLETE: refusal, _PENDING: refusal,
    }


class ForkReturnError(RuntimeError):
    """The export could not run (paths, I/O); nothing was published."""


class ForkReturnRefused(ForkReturnError):
    """The source cannot be returned without losing data or semantics; nothing was published."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__("; ".join(report["refused"]))


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sidecar(path: Path, suffix: str) -> Path:
    return path.with_name(path.name + suffix)


def _ro_uri(path: Path) -> str:
    return Path(path).resolve().as_uri() + "?mode=ro&immutable=1"


def _is_wal_file(path: Path) -> bool:
    with open(path, "rb") as handle:
        header = handle.read(20)
    return len(header) == 20 and header[18:20] == b"\x02\x02"


# ── Schema inventory and the plan ──

def _inventory(conn, schema: str = "main") -> dict:
    objects = {"table": {}, "index": {}, "trigger": {}, "view": {}}
    for kind, name, table, sql in conn.execute(f"SELECT type, name, tbl_name, sql FROM {schema}.sqlite_master"):
        objects[kind][name] = (table, sql or "")
    virtual = {n for n, (_, sql) in objects["table"].items() if sql.upper().startswith("CREATE VIRTUAL TABLE")}
    tables = {}
    for name, (_, sql) in objects["table"].items():
        tables[name] = {
            "sql": sql, "virtual": name in virtual,
            "shadow_of": next((v for v in virtual for s in _FTS5_SHADOW_SUFFIXES if name == v + s), None),
            "columns": {row[1]: row[4] for row in conn.execute(f"PRAGMA {schema}.table_xinfo({_q(name)})")},
            "rows": None if name in virtual else conn.execute(f"SELECT count(*) FROM {schema}.{_q(name)}").fetchone()[0],
        }
    return {"tables": tables, "indexes": objects["index"], "triggers": objects["trigger"], "views": objects["view"]}


def _non_default_rows(conn, table: str, column: str, default) -> int:
    differs = f"{_q(column)} IS NOT NULL" if default is None else f"{_q(column)} IS NOT ({default})"
    return conn.execute(f"SELECT count(*) FROM {_q(table)} WHERE {differs}").fetchone()[0]


def _admission_input(payload_json, target, lineage_json) -> tuple:
    """``(text, sessions, None)`` for an input this export can match, else ``(None, None, why not)``."""
    try:
        payload, lineage = json.loads(payload_json), json.loads(lineage_json)
    except (TypeError, ValueError):
        return None, None, "payload or lineage is not JSON"
    if not isinstance(payload, dict) or set(payload) != {"text"} or not isinstance(payload["text"], str):
        shape = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        return None, None, f"payload {shape} is more than its text"
    if not isinstance(lineage, list) or not all(isinstance(s, str) for s in lineage):
        return None, None, "lineage is not a list of session ids"
    return payload["text"], sorted({target, *lineage}), None


def _plan_admissions(conn, refused: list) -> dict:
    """Each terminal admission's input against the user messages the output keeps: a run's input must be one
    of them (each message answers one admission), an input that never ran refuses the export."""
    taken, matched, lost, unruled = set(), 0, [], []
    for admission_id, target, lineage, payload, outcome in conn.execute(
            "SELECT admission_id, target_session_id, lineage_json, payload_json, outcome FROM session_admissions "
            "WHERE status = 'terminal' ORDER BY seq").fetchall():
        if outcome in _ADMISSION_NEVER_RAN:
            lost.append(f"{admission_id} ({_ADMISSION_NEVER_RAN[outcome]})")
            continue
        if outcome not in _ADMISSION_RAN:
            unruled.append(f"{admission_id}: outcome {outcome!r}")
            continue
        text, sessions, why = _admission_input(payload, target, lineage)
        if why:
            unruled.append(f"{admission_id}: {why}")
            continue
        hits = conn.execute(
            f"SELECT id FROM messages WHERE role = 'user' AND typeof(content) = 'text' AND content = ? "
            f"AND session_id IN ({', '.join('?' * len(sessions))}) ORDER BY id", (text, *sessions)).fetchall()
        hit = next((mid for (mid,) in hits if mid not in taken), None)
        if hit is None:
            lost.append(f"{admission_id} ({outcome}; no kept user message of its sessions carries its input)")
            continue
        taken.add(hit)
        matched += 1
    if lost:
        refused.append(f"lossless return impossible: {len(lost)} session_admissions inputs exist only in "
                       f"session_admissions: {'; '.join(lost[:10])}{' ...' if len(lost) > 10 else ''}")
    if unruled:
        refused.append(f"session_admissions rows no rule carries to the fork: "
                       f"{'; '.join(unruled[:10])}{' ...' if len(unruled) > 10 else ''}")
    return {"matched_to_messages": matched, "lossless_return_impossible": len(lost), "no_rule": len(unruled)}


def _plan_runtime_table(conn, name: str, info: dict, columns: dict, refused: list) -> dict:
    entry = {"disposition": "converted", "rows": info["rows"], "reason": _RUNTIME_TABLES[name]}
    known, before = _RUNTIME_COLUMNS[name], len(refused)
    for column, default in info["columns"].items():
        col = {"non_default_rows": _non_default_rows(conn, name, column, default)}
        if column in known:
            col.update(disposition="converted", reason=known[column])
        else:
            col["disposition"] = "refused"
            refused.append(f"column {name}.{column} ({col['non_default_rows']} rows set) has no representation "
                           "in the fork")
        columns[f"{name}.{column}"] = col
    if missing := sorted(set(known) - set(info["columns"])):
        refused.append(f"{name} lacks {missing}; its rows are not the shape this export reads")
        entry["disposition"] = "refused"
        return entry
    for column, allowed in _RUNTIME_VALUES.get(name, {}).items():
        entry[f"by_{column}"] = counts = dict(conn.execute(
            f"SELECT {_q(column)}, count(*) FROM {_q(name)} GROUP BY 1 ORDER BY 1").fetchall())
        pending = {v: n for v, n in counts.items() if column == "status" and v in _UNFINISHED[name]}
        if pending:
            refused.append(f"{name} holds unfinished work the fork cannot carry: {pending}")
        if unknown := {v: n for v, n in counts.items() if v not in allowed and v not in pending}:
            refused.append(f"{name}.{column} holds values no rule carries to the fork: {unknown}")
    if name == "worker_receipts":
        try:
            orphans = conn.execute("SELECT count(*) FROM worker_receipts AS r WHERE NOT EXISTS (SELECT 1 FROM "
                                   "worker_executions AS w WHERE w.execution_id = r.execution_id "
                                   "AND w.status = 'terminal')").fetchone()[0]
        except sqlite3.OperationalError as exc:
            orphans = f"unknown ({exc})"
        if orphans:
            refused.append(f"worker_receipts rows of no terminal worker: {orphans}")
    if name == "session_admissions":
        entry["inputs"] = _plan_admissions(conn, refused)
    if len(refused) > before:
        entry["disposition"] = "refused"
    return entry


def _plan_fork_table(conn, name: str, info: dict, columns: dict, drop_columns: dict, refused: list) -> None:
    known, ours = set(_FORK_TABLES[name].split()), _COLUMNS.get(name, {})
    for column, default in info["columns"].items():
        if column in ours:
            drop_columns.setdefault(name, []).append(column)
            columns[f"{name}.{column}"] = {"disposition": "converted", "reason": ours[column],
                                           "non_default_rows": _non_default_rows(conn, name, column, default)}
        elif column not in known:
            columns[f"{name}.{column}"] = {"disposition": "refused"}
            refused.append(f"column {name}.{column} has no representation in the fork")


def _plan_tables(conn, inventory: dict, stamp, refused: list) -> dict:
    tables, columns, drop_tables, drop_columns = {}, {}, [], {}
    for name, info in sorted(inventory["tables"].items()):
        if name.startswith("sqlite_") or info["shadow_of"] in _FORK_TABLES:
            continue  # SQLite's own tables, and FTS5 shadows that follow a kept virtual table
        if name in _RUNTIME_TABLES:
            tables[name] = _plan_runtime_table(conn, name, info, columns, refused)
            drop_tables.append(name)
        elif name == "schema_version":
            tables[name] = {"disposition": "converted", "rows": info["rows"],
                            "reason": f"stamp {stamp} -> {FORK_STORED_SCHEMA_VERSION}"}
            if set(info["columns"]) != {"version"}:
                tables[name]["disposition"] = "refused"
                refused.append(f"schema_version has columns {sorted(info['columns'])}; only its version converts")
        elif name in _FORK_TABLES:
            tables[name] = {"disposition": "preserved", "rows": info["rows"]}
            _plan_fork_table(conn, name, info, columns, drop_columns, refused)
        elif info["rows"] == 0 and not info["shadow_of"]:
            tables[name] = {"disposition": "converted", "rows": 0, "reason": "empty table the fork does not have"}
            for column in info["columns"]:
                columns[f"{name}.{column}"] = {"disposition": "converted", "non_default_rows": 0,
                                               "reason": "column of an empty table"}
            drop_tables.append(name)
        elif not info["shadow_of"]:
            tables[name] = {"disposition": "refused", "rows": info["rows"]}
            refused.append(f"table {name} ({'virtual' if info['virtual'] else info['rows']} rows) "
                           "has no representation in the fork")
    return {"tables": tables, "columns": columns, "drop_tables": drop_tables, "drop_columns": drop_columns}


def _plan_objects(inventory: dict, fences: list, gone: set, refused: list) -> dict:
    objects = {}
    for kind, known, ours in (("triggers", _FORK_TRIGGERS, _TRIGGERS), ("indexes", _FORK_INDEXES, _INDEXES),
                              ("views", _FORK_VIEWS, ())):
        for name, (table, _sql) in sorted(inventory[kind].items()):
            if name.startswith("sqlite_autoindex_") or name.lower() in fences:
                continue
            if name in known:
                objects[name] = "preserved"
            elif name in ours or table in gone:
                objects[name] = "converted"
            else:
                objects[name] = "refused"
                refused.append(f"{kind[:-1]} {name} on {table} has no representation in the fork")
    return objects


def _plan(conn, inventory: dict) -> dict:
    """Every source object's disposition, decided on the read-only source view; every refusal collected."""
    refused = []
    try:
        lineage = fence.decode_store_lineage(conn.cursor())
        fences = sorted(n.lower() for n in fence.owned_turn_fence_literals(conn.cursor()))
    except fence.IncompatibleSchemaError as exc:
        lineage, fences = None, []
        refused.append(f"source lineage is unreadable to this build: {exc}")
    if lineage is not None and not lineage.settled:
        refused.append(f"source is {lineage.lineage} at schema_version {lineage.stored}; only this build's "
                       f"settled {fence.STORED_SCHEMA_VERSION} store returns to the fork")
    plan = _plan_tables(conn, inventory, lineage.stored if lineage else None, refused)
    gone = set(plan["drop_tables"])
    objects = _plan_objects(inventory, fences, gone, refused)
    keys = _meta_keys()
    meta = {} if "state_meta" not in inventory["tables"] else {
        key: {"disposition": "converted", "reason": keys[key]} for (key,) in conn.execute(
            f"SELECT key FROM state_meta WHERE key IN ({', '.join('?' * len(keys))}) ORDER BY key", tuple(keys))}
    plan.update(
        refused=refused, objects=objects, state_meta=meta, fences=fences,
        drop_triggers=[n for n, (table, _) in inventory["triggers"].items()
                       if objects.get(n) == "converted" and table not in gone],
        drop_indexes=[n for n, (table, _) in inventory["indexes"].items()
                      if objects.get(n) == "converted" and table not in gone])
    return plan


# ── The conversion, on the build copy ──

def _convert(conn, plan: dict) -> None:
    conn.execute("BEGIN IMMEDIATE")
    # Triggers first: every later DDL re-parses the schema, and no trigger may name a dropped object.
    for name in plan["fences"] + plan["drop_triggers"]:
        conn.execute(f"DROP TRIGGER {_q(name)}")
    for name in plan["drop_indexes"]:
        conn.execute(f"DROP INDEX {_q(name)}")
    for name in plan["drop_tables"]:
        conn.execute(f"DROP TABLE {_q(name)}")
    for table, cols in plan["drop_columns"].items():
        for column in cols:
            conn.execute(f"ALTER TABLE {_q(table)} DROP COLUMN {_q(column)}")
    if plan["state_meta"]:
        conn.execute(f"DELETE FROM state_meta WHERE key IN ({', '.join('?' * len(plan['state_meta']))})",
                     tuple(plan["state_meta"]))
    governed = fence._fork_governed(fence._existing_tables(conn.cursor()))
    for sql in fence.turn_fence_trigger_definitions(governed, generation=FORK_STORED_SCHEMA_VERSION).values():
        conn.execute(sql)
    conn.execute("UPDATE schema_version SET version = ?", (FORK_STORED_SCHEMA_VERSION,))
    conn.execute("COMMIT")


# ── Measuring loss: the output against the attached source ``src``, table by table ──

def _row_differences(conn, table: str, columns: list, sql: str, exclude) -> tuple:
    """``(source rows, output rows, source rows with no identical output row)`` on *columns*."""
    key, values = exclude
    skip = f"{_q(key)} NOT IN ({', '.join('?' * len(values))})" if values else ""
    count = f"SELECT count(*) FROM {{}}.{_q(table)}" + (f" WHERE {skip}" if skip else "")
    n_src, n_out = (conn.execute(count.format(schema), values).fetchone()[0] for schema in ("src", "main"))
    if "WITHOUT ROWID" in " ".join(sql.upper().split()):
        listed = ", ".join(_q(c) for c in columns)
        where = f" WHERE {skip}" if skip else ""
        absent = conn.execute(f"SELECT count(*) FROM (SELECT {listed} FROM src.{_q(table)}{where} "
                              f"EXCEPT SELECT {listed} FROM main.{_q(table)}{where})", values * 2).fetchone()[0]
    else:
        # Same rowid, same value and same storage class in every kept column (``IS`` alone equates 1 and 1.0).
        same = " AND ".join(f"d.{_q(c)} IS s.{_q(c)} AND typeof(d.{_q(c)}) = typeof(s.{_q(c)})" for c in columns)
        where = f" AND s.{skip}" if skip else ""
        absent = conn.execute(f"SELECT count(*) FROM src.{_q(table)} AS s WHERE NOT EXISTS (SELECT 1 FROM "
                              f"main.{_q(table)} AS d WHERE d.rowid = s.rowid AND {same}){where}", values).fetchone()[0]
    return n_src, n_out, absent


def _compare(conn, plan: dict, source: dict) -> dict:
    output = _inventory(conn)
    differences = {}
    exclude = {"state_meta": ("key", tuple(plan["state_meta"])),
               "sqlite_sequence": ("name", tuple(plan["drop_tables"]))}
    for name, info in sorted(source["tables"].items()):
        if name.startswith("sqlite_stat"):
            continue  # planner statistics, derived
        if name == "schema_version":
            continue  # the stamp is the conversion itself; the output lineage decode checks it
        if name in plan["drop_tables"]:
            if name in output["tables"]:
                differences[name] = "still present"
            continue
        if name not in output["tables"]:
            differences[name] = f"missing ({info['rows']} rows)"
            continue
        kept = set(info["columns"]) - set(plan["drop_columns"].get(name, ()))
        if set(output["tables"][name]["columns"]) != kept:
            differences[name] = f"columns differ: {sorted(set(output['tables'][name]['columns']) ^ kept)}"
            continue
        if info["virtual"]:
            continue  # FTS5 content is its shadow tables, compared as tables
        n_src, n_out, absent = _row_differences(
            conn, name, list(output["tables"][name]["columns"]), info["sql"], exclude.get(name, (None, ())))
        entry = plan["tables"].setdefault(name, {"disposition": "preserved", "rows": info["rows"]})
        entry.update(source_rows=n_src, output_rows=n_out, absent_or_changed_rows=absent)
        if absent or n_src != n_out:
            differences[name] = f"{absent} source rows absent or changed; {n_src} source vs {n_out} output rows"
    if extra := sorted(set(output["tables"]) - set(source["tables"])):
        differences["(tables)"] = f"not in the source: {extra}"
    for kind in ("indexes", "triggers", "views"):
        expected = {n for n, (table, _sql) in source[kind].items()
                    if plan["objects"].get(n, "preserved") == "preserved" and n.lower() not in plan["fences"]
                    and table not in plan["drop_tables"]}
        actual = {n for n in output[kind] if not n.startswith("turn_fence_")}
        rewritten = sorted(n for n in expected & actual if source[kind][n] != output[kind][n])
        if expected != actual or rewritten:
            differences[f"({kind})"] = {"missing": sorted(expected - actual), "unexpected": sorted(actual - expected),
                                        "rewritten": rewritten}
    return differences


def _integrity(conn) -> list:
    return [row[0] for row in conn.execute("PRAGMA main.integrity_check")]


# ── The entry point ──

_SIDECARS = ("", "-wal", "-shm", "-journal")


def _bundle(path: Path) -> list:
    return [_sidecar(path, suffix) for suffix in _SIDECARS]


def _same_path(a: Path, b: Path) -> bool:
    """Equal paths, equal on a case- and normalization-insensitive filesystem, or one file (hard/symbolic link)."""
    if unicodedata.normalize("NFC", str(a)).casefold() == unicodedata.normalize("NFC", str(b)).casefold():
        return True
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _directory(path, what: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ForkReturnError(f"{what} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise ForkReturnError(f"{what} is not a directory: {resolved}")
    return resolved


def _new_path(path, what: str) -> Path:
    path = Path(path).expanduser()
    path = _directory(path.parent, f"the {what}'s directory") / path.name
    for candidate in (_bundle(path) if what == "output" else [path]):
        if os.path.lexists(candidate):
            raise ForkReturnError(f"refusing to overwrite {candidate}: the {what} must be a new file")
    return path


def _validated(source, output, work_dir, report_path) -> tuple:
    """Every path resolved and checked against every other before anything is written: the output and report
    are new files, and neither is, aliases or links to the source, each other, or any of their SQLite files."""
    try:
        source = Path(source).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ForkReturnError(f"source does not exist: {source}") from exc
    if not source.is_file():
        raise ForkReturnError(f"source is not a file: {source}")
    named = {"the source": _bundle(source)}
    for what, path in (("output", output), ("report", report_path)):
        if what == "output" or path:
            path = _new_path(path, what)
            for owner, files in named.items():
                if clash := next((f for f in files for p in (_bundle(path) if what == "output" else [path])
                                  if _same_path(f, p)), None):
                    raise ForkReturnError(f"the {what} {path} would be {owner}'s file {clash}")
            named[f"the {what}"] = _bundle(path) if what == "output" else [path]
    output = named["the output"][0]
    report = named["the report"][0] if "the report" in named else None
    work = _directory(work_dir, "the work directory") if work_dir else output.parent
    if _sidecar(source, "-journal").exists():
        raise ForkReturnError("the source has a hot rollback journal; let its writer finish first")
    return source, output, work, report


def _source_state(source: Path) -> dict:
    """The source's bytes: a sha256 of the database and of each journal file it has."""
    state = {}
    for path in (source, _sidecar(source, "-wal"), _sidecar(source, "-journal")):
        if path is source or path.exists():
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(chunk)
            state[path.name] = digest.hexdigest()
    return state


_STAGED, _BUILDING = "source.db", "build.db"  # the only names this run creates, inside its own scratch


def _scratch(parent: Path, created: list) -> Path:
    """A new private directory, created exclusively; the only place this run creates its working files."""
    path = Path(tempfile.mkdtemp(prefix=".fork-return-", dir=parent))
    created.append(path)
    return path


def _remove_scratch(directory: Path) -> Optional[str]:
    """Delete the files this run made in *directory*, then the directory if that empties it; never anything else."""
    for name in (_STAGED, _BUILDING):
        for path in _bundle(directory / name):
            path.unlink(missing_ok=True)
    try:
        directory.rmdir()
    except OSError:
        return str(directory)
    return None



def _write_report(path: Path, report: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ForkReturnError(f"refusing to overwrite {path}: it appeared during the export") from exc
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")


def return_to_fork(source, output, *, work_dir=None, report_path=None) -> dict:
    """Write *source* (a settled 1033 store) to *output* as the live fork's fenced 1030 store; return the report.

    Raises ForkReturnRefused (carrying the report) when anything cannot be carried, ForkReturnError on unusable
    paths. Either way *output* is not created, *source* is not written and no existing file is touched; the
    report's ``source_unchanged`` is true only when the source's bytes were re-read and found unchanged."""
    source, output, work, report_file = _validated(source, output, work_dir, report_path)
    try:
        before = _source_state(source)
    except OSError as exc:
        raise ForkReturnError(f"cannot read the source: {exc}") from exc
    wal_mode = _is_wal_file(source)
    report = {"operation": "return_to_fork", "source": str(source), "output": str(output),
              "target_schema_version": FORK_STORED_SCHEMA_VERSION, "published": False, "refused": [],
              "source_unchanged": False}
    scratch = []
    try:
        view = source
        if _sidecar(source, "-wal").exists() and _sidecar(source, "-wal").stat().st_size:
            # The WAL holds pages the main file lacks: fold them into a private copy, never into the source.
            staged = _scratch(work, scratch) / _STAGED
            shutil.copyfile(source, staged)
            shutil.copyfile(_sidecar(source, "-wal"), _sidecar(staged, "-wal"))
            conn = sqlite3.connect(staged, isolation_level=None)
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.execute("PRAGMA journal_mode=DELETE")
            finally:
                conn.close()
            view = staged
        conn = sqlite3.connect(_ro_uri(view), uri=True, isolation_level=None)
        try:
            inventory = _inventory(conn)
            plan = _plan(conn, inventory)
        finally:
            conn.close()
        report.update(tables=plan["tables"], columns=plan["columns"], state_meta=plan["state_meta"],
                      schema_objects=plan["objects"])
        if plan["refused"]:
            report["refused"] = plan["refused"]
            raise ForkReturnRefused(report)

        building = _scratch(output.parent, scratch) / _BUILDING
        shutil.copyfile(view, building)
        os.chmod(building, 0o600)
        conn = sqlite3.connect(building.as_uri(), uri=True, isolation_level=None)
        try:
            # The build copy is disposable until it is published: no journal, and any failure discards it.
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA foreign_keys=OFF")
            _convert(conn, plan)
            conn.execute("ATTACH DATABASE ? AS src", (_ro_uri(view),))
            lineage = fence.decode_store_lineage(conn.cursor())
            differences = _compare(conn, plan, inventory)
            integrity = _integrity(conn)
            conn.execute("DETACH DATABASE src")
            conn.execute(f"PRAGMA journal_mode={'WAL' if wal_mode else 'DELETE'}")
        finally:
            conn.close()
        report.update(
            output_lineage={"lineage": lineage.lineage, "schema_version": lineage.stored,
                            "fence_generations": sorted(lineage.fence_literals), "fork_store": lineage.migratable},
            differences=differences, loss_detected=bool(differences), integrity_check=integrity,
            source_unchanged=_source_state(source) == before)
        if differences:
            report["refused"].append(f"the output differs from the source: {sorted(differences)}")
        if not lineage.migratable:
            report["refused"].append("the output is not exactly the fork's fenced 1030 store")
        if integrity != ["ok"]:
            report["refused"].append(f"integrity_check: {integrity[:5]}")
        if not report["source_unchanged"]:
            report["refused"].append("the source changed while it was read")
        if report["refused"]:
            raise ForkReturnRefused(report)
        with open(building, "rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(building, output)  # exclusive: a file that appeared at the output since validation is kept
        except FileExistsError as exc:
            raise ForkReturnError(f"refusing to overwrite {output}: it appeared during the export") from exc
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        report["published"] = True
        return report
    finally:
        if left := [path for path in map(_remove_scratch, scratch) if path]:
            report["scratch_left"] = left
        try:
            report["source_unchanged"] = _source_state(source) == before
        except OSError as exc:
            report["source_unchanged"], report["source_check"] = False, f"unverifiable: {exc}"
        if report_file:
            _write_report(report_file, report)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hermes_state_fork_return",
                                     description="Return this build's state.db to the live fork's schema 1030.")
    parser.add_argument("source", help="this build's state.db (schema_version 1033); never written")
    parser.add_argument("output", help="new file for the fork's 1030 store; must not exist")
    parser.add_argument("--work-dir", help="where a WAL-bearing source is staged (default: the output's directory)")
    parser.add_argument("--report", help="write the JSON report here")
    args = parser.parse_args(argv)

    def cancel(signum, _frame):
        raise SystemExit(128 + signum)  # a terminated export still removes its scratch and reports

    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        report = return_to_fork(args.source, args.output, work_dir=args.work_dir, report_path=args.report)
    except ForkReturnRefused as exc:
        print("Refused; nothing was published:", *(f"  - {r}" for r in exc.report["refused"]), sep="\n",
              file=sys.stderr)
        return 2
    except ForkReturnError as exc:
        print(f"Refused; nothing was published: {exc}", file=sys.stderr)
        return 2
    finally:
        signal.signal(signal.SIGTERM, previous)
    tables = report["tables"]
    print(f"Returned to the fork's schema {FORK_STORED_SCHEMA_VERSION}: {report['output']}")
    print(f"  tables preserved: {sum(v['disposition'] == 'preserved' for v in tables.values())}; converted: "
          + (", ".join(f"{k} ({v['rows']} rows)" for k, v in tables.items() if v["disposition"] == "converted")
             or "none"))
    print("  columns converted: " + ", ".join(f"{k} ({v['non_default_rows']} set)" for k, v in report["columns"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
