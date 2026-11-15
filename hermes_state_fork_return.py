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
one that ran must find its input in the kept rows (its text as a user message of its sessions, client-supplied
history as the kept messages before it), and one that never ran (cancelled) exists nowhere else. Every key of its
payload, at every depth, is on an allowlist of lossless mappings; a key no kept row holds returns only while it
carries nothing, and a key or shape the list lacks refuses with its path. So does any object, column or value this
module has no rule for, and a session this build's lookups (its label aliases) find under other labels than the
fork's, which filter on sessions.source alone.

Loss is measured, not assumed: after the conversion every kept table is compared row by row against
the source on every column the result has, every removed table, column and state_meta key is counted
against its disposition, and an unaccounted difference refuses the export with nothing published.

Paths are checked before anything is written: the output and the report are new files, and neither may
be, alias or link to the source, each other or any of their SQLite files. Working copies live in
directories created for this run alone and are removed on any exit; the output is published with an
exclusive link, so a file that appears there meanwhile is never replaced. The source is read again before
publishing and once more after; a change then (or any failure or cancel before the commit point) takes the output
back, and only while it is still the file this run linked. The commit point is the exclusive link of the output's
commit marker (OUTPUT.fork-return, naming its bytes): only an output whose marker verifies is a return
(``--verify``), and nothing is taken back once it is linked.

Copying the canonical tables into a fresh fork schema, as the fork's own ``sessions recover`` does, is
ruled out: it leaves the authorities, their events, turn leases and conversation generations behind
and still reports no loss. This build's tables are removed rather than kept beside the fork's tables:
the fork writes with foreign keys on, and their RESTRICT keys to sessions would refuse its deletes.

    python -m hermes_state_fork_return SOURCE OUTPUT [--work-dir DIR] [--report PATH]
    python -m hermes_state_fork_return --verify OUTPUT
"""

from __future__ import annotations

import argparse
import functools
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
        "payload_json": "the input and what it ran under: every key a lossless mapping into kept rows, or nothing",
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
        "created_source": "creation label: unset or equal to source, and found under the same labels by the fork",
        "auto_archived": "sweep provenance of archived; archived is kept",
    },
}
# A converted column whose rows must satisfy a condition, else the fork reads them differently: (the rows that
# break it, what that means, the columns that name such a row). This build lists a ``source='local'`` session
# under the surface that created it (created_source); the fork finds every session by its source alone, so a
# row whose two labels differ would leave that listing. Unset is the fork-era rows' own state.
_COLUMN_CONDITIONS = {
    ("sessions", "created_source"): ("created_source IS NOT NULL AND created_source IS NOT source",
                                     "differs from source; the fork finds a session by its source alone",
                                     ("id", "source", "created_source")),
}
_TRIGGERS = (
    "messages_display_revision_update", "messages_message_uid_insert", "messages_transcript_revision_update",
    "sessions_runtime_ledger_delete_guard",
)
_INDEXES = ("idx_messages_session_uid", "idx_sessions_local_created_started")
# Every rule in this module is for this pair: this build's settled store, the fork's schema and its build. Either
# side at another version refuses until the rules are written for it.
_RULES = {"source_schema_version": 1033, "target_schema_version": 1030, "fork_build": "420c58a13e"}


# ── Session labels: the same sessions must be found under every label by this build's lookups and the fork's ──
#
# This build's listings, search and counts (hermes_cli.gateway_session_store.SurfaceLabelReader) expand a requested
# label through its alias table, gateway.session_policy.SURFACE_GROUPS: ``gui`` and ``desktop`` name one
# conversation, and a ``source='local'`` row is found under the group of its created_source. The fork filters on
# sessions.source alone (``s.source IN (...)``). scripts/review_kit_fork_return.sh measures both, with the real
# binaries, for every label below.

def lookup_labels() -> list:
    """Every label this build's session lookups resolve through its alias table, and ``local``."""
    from gateway.session_policy import SURFACE_GROUPS, SURFACES

    return sorted(set(SURFACE_GROUPS) | set(SURFACES) | set(SURFACES.values()) | {"local"})


def label_lookups(source, created_source) -> tuple:
    """``(labels this build's lookups find a session under, labels the fork's find it under)``."""
    from gateway.session_policy import SURFACE_GROUPS

    found = {label for label, group in SURFACE_GROUPS.items()
             if (created_source if source == "local" else source) in group}
    if source is not None and source not in SURFACE_GROUPS:
        found.add(source)  # a label outside the alias table is a plain ``source = ?`` filter on both sides
    return sorted(found), [] if source is None else [source]


def _plan_labels(conn, inventory: dict, refused: list) -> list:
    """Each (source, created_source) pair the sessions hold, with the labels each side's lookups find it under; a
    pair the two find under different labels refuses: the session would leave or join a listing on return."""
    if "sessions" not in inventory["tables"]:
        return []
    created = "created_source" if "created_source" in inventory["tables"]["sessions"]["columns"] else "NULL"
    labels = []
    for source, created_source, count, ids in conn.execute(
            f"SELECT source, {created}, count(*), json_group_array(id) FROM (SELECT * FROM sessions ORDER BY rowid) "
            "GROUP BY 1, 2 ORDER BY 1, 2").fetchall():
        this_build, fork = label_lookups(source, created_source)
        labels.append({"source": source, "created_source": created_source, "sessions": count,
                       "this_build_finds_under": this_build, "the_fork_finds_under": fork})
        if this_build != fork:
            shown = json.loads(ids)
            refused.append(f"lossless return impossible: sessions with source={source!r}, created_source="
                           f"{created_source!r} are found under {this_build} by this build's lookups and under {fork} "
                           f"by the fork's: {count} rows: {', '.join(map(str, shown[:10]))}"
                           f"{' ...' if count > 10 else ''}")
    return labels


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
    """The export did not succeed (paths, I/O); unless it is a ForkReturnResidual, nothing it published is left."""


class ForkReturnRefused(ForkReturnError):
    """The source cannot be returned without losing data or semantics, or changed while it was; nothing this run
    published is left at the output (``report["retraction"]`` says what was taken back)."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__("; ".join(report["refused"]))


class ForkReturnResidual(ForkReturnError):
    """The export failed after publishing and its output could not be taken back: ``report["residual"]`` names
    the path left and what it is."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__(f"{report['residual']['path']}: {report['residual']['state']}")


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


# ── Finished admissions: an allowlist of lossless mappings, checked against the rows the output keeps ──
#
# A finished admission's payload is the turn's input and what the turn ran under, and this build reads it after the
# run too (an API turn's terminal replay projects settings.ephemeral_system_prompt as its ``instructions``; a webhook
# delivery is looked up by its source and webhook_delivery). So every key, at every depth, is listed with the one
# mapping that carries it into the fork, and a key or shape the table does not list refuses with its exact path.
# Only these mappings carry a set value; each is verified against the rows the output keeps:
_PRIOR = "client-supplied history: each entry a kept message of its sessions before the input's row, in order"
_SHARE = "a re-run of an unanswered turn: it adopted its failed attempt's tail row instead of writing it again"
# Every other listed key has no place in the fork's schema (it keeps no admission ledger). It returns only while it
# carries nothing: unset (null, an empty list or object), or this build's own declared default for that field.
_HELD = "no kept row holds it, and the fork has no place for it: it returns only while it carries nothing"
# A gateway session's source (SessionSource.to_dict, plus is_bot) and the platform event it carried.
_SOURCE = dict.fromkeys((
    "platform", "chat_id", "chat_name", "chat_type", "user_id", "user_name", "thread_id", "chat_topic", "user_id_alt",
    "chat_id_alt", "scope_id", "guild_id", "parent_chat_id", "message_id", "profile", "auto_thread_created",
    "auto_thread_initial_name", "prospective_thread_id", "shared_session", "is_bot"), _HELD)
_EVENT = dict.fromkeys((
    "user_id", "user_name", "message_id", "platform_update_id", "reply_to_message_id", "reply_to_text",
    "reply_to_author_id", "reply_to_author_name", "reply_to_is_own_message", "allow_gateway_control", "auto_skill",
    "channel_prompt", "channel_context", "reply_expected", "reply_anchor_override", "edit_revision",
    "inherit_channel_pin", "metadata", "media_types", "media_text_inlined", "media_filenames"), _HELD)
_ENVELOPES = {
    "native_text_v1": {
        "source": _SOURCE, "event": _EVENT,
        **dict.fromkeys(("media", "route", "timestamp", "webhook_delivery", "webhook_route", "reauthorize",
                         "provenance", "message_type", "automation", "ingress_sha256"), _HELD),
    },
    "api_turn_v1": {
        "history": _PRIOR, "resume_unanswered": _SHARE,
        "settings": dict.fromkeys((
            "ephemeral_system_prompt", "requested_model", "requested_provider", "model_options", "route",
            "session_model", "confirmed_runtime_lock", "requested_runtime", "route_source", "room_dispatch",
            "room_execution_policy", "session_history_delivery"), _HELD),
        # Client JSON (the request body's ``metadata``) the turn handed to its agent: no key of it has a mapping.
        "relay_metadata": {},
        **dict.fromkeys(("media", "turn_author", "run_owner_scope", "browser_control"), _HELD),
    },
}
_NO_DEFAULT = object()


@functools.lru_cache(maxsize=1)
def _declared_defaults() -> dict:
    """This build's own defaults of the native source and event fields, by payload path: a value equal to one says
    nothing the field's absence does not (``allow_gateway_control`` defaults to true, so false is data)."""
    import dataclasses

    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource

    defaults = {}
    for path, cls in (("native_text_v1.source", SessionSource), ("native_text_v1.event", MessageEvent)):
        for field in dataclasses.fields(cls):
            if field.default is not dataclasses.MISSING:
                defaults[f"{path}.{field.name}"] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                defaults[f"{path}.{field.name}"] = field.default_factory()
    return defaults


def _carries_nothing(at: str, value) -> bool:
    if value is None or (isinstance(value, (list, dict)) and not value):
        return True
    default = _declared_defaults().get(at, _NO_DEFAULT)
    return default is not _NO_DEFAULT and type(value) is type(default) and value == default


def _fits(rule, value) -> bool:
    """Whether *value* has the shape *rule* reads; anything else has no rule."""
    if isinstance(rule, dict):
        return value is None or isinstance(value, dict)
    if rule == _SHARE:
        return value is True
    if rule == _PRIOR:
        return value is None or isinstance(value, list) and all(
            isinstance(e, dict) and set(e) == {"role", "content"} and all(isinstance(v, str) for v in e.values())
            for e in value)
    return True


def _walk(value: dict, rules: dict, path: str, found: dict) -> None:
    """Sort every leaf of *value* under its mapping into *found*: ``no_rule`` and ``held`` (paths), ``prior``
    (path, role, content) and ``share``."""
    for key, item in value.items():
        at, rule = f"{path}.{key}", rules.get(key)
        if rule is None:
            found["no_rule"].append(f"no rule for {at}")
        elif not _fits(rule, item):
            found["no_rule"].append(f"{at} is not the shape its rule reads: {json.dumps(item, default=str)[:80]}")
        elif isinstance(rule, dict):
            if item is not None:
                _walk(item, rule, at, found)
        elif rule == _HELD:
            if not _carries_nothing(at, item):
                found["held"].append(at)
        elif rule == _SHARE:
            found["share"] = True
        elif rule == _PRIOR:
            found["prior"].extend((f"{at}[{n}]", e["role"], e["content"]) for n, e in enumerate(item or ()))


def _admission_input(payload_json, target, lineage_json) -> tuple:
    """``(kind, input, None)`` for an input every leaf of which has a mapping, else ``(kind, None, why not)``.
    *input* holds its sessions, text, whether it may share a row, the history entries that must precede it, and
    the set leaves no kept row holds."""
    try:
        payload, lineage = json.loads(payload_json), json.loads(lineage_json)
    except (TypeError, ValueError):
        return "unreadable", None, "payload or lineage is not JSON"
    if not isinstance(payload, dict):
        return "unreadable", None, f"payload is a {type(payload).__name__}"
    extra = sorted(set(payload) - {"text"})
    kind = "+".join(extra) or "text"
    if not isinstance(lineage, list) or not all(isinstance(s, str) for s in lineage):
        return kind, None, "lineage is not a list of session ids"
    if not isinstance(payload.get("text"), str):
        return kind, None, f"its text is a {type(payload.get('text')).__name__}"
    found = {"no_rule": [], "held": [], "prior": [], "share": False}
    if extra:
        envelope = payload[extra[0]]
        if (len(extra) > 1 or extra[0] not in _ENVELOPES or not isinstance(envelope, dict)
                or extra[0] == "native_text_v1" and not isinstance(envelope.get("event"), dict)):
            return kind, None, f"payload {sorted(payload)} has no rule"
        _walk(envelope, _ENVELOPES[extra[0]], extra[0], found)
    if found["no_rule"]:
        return kind, None, "; ".join(found["no_rule"])
    return kind, {"sessions": sorted({target, *lineage}), "text": payload["text"], **found}, None


def _admission_row(conn, match: dict, taken: set):
    """The kept user row that is this input: a user message of its sessions that is exactly its text. Each row is
    one admission's, unless shared."""
    sessions = match["sessions"]
    hits = conn.execute(f"SELECT id FROM messages WHERE role = 'user' AND typeof(content) = 'text' AND content = ? "
                        f"AND session_id IN ({', '.join('?' * len(sessions))}) ORDER BY id",
                        (match["text"], *sessions)).fetchall()
    return next((mid for (mid,) in hits if match["share"] or mid not in taken), None)


def _unmatched_history(conn, match: dict, row: int) -> Optional[str]:
    """The first history entry that is no kept message of its sessions before *row* (in order), or None."""
    marks, after = ", ".join("?" * len(match["sessions"])), 0
    for path, role, content in match["prior"]:
        hit = conn.execute(f"SELECT min(id) FROM messages WHERE session_id IN ({marks}) AND id > ? AND id < ? AND "
                           "role = ? AND typeof(content) = 'text' AND content = ?",
                           (*match["sessions"], after, row, role, content)).fetchone()[0]
        if hit is None:
            return f"{path} ({role}) is no kept message of its sessions before its input"
        after = hit
    return None


def _plan_admissions(conn, refused: list) -> dict:
    """Each terminal admission's input against the user rows the output keeps: a run's input must be one of them,
    with every leaf of its payload carried by a rule; an input that never ran refuses the export. Counted per
    payload kind."""
    taken, kinds, lost, unruled = set(), {}, [], []
    for admission_id, target, lineage, payload, outcome in conn.execute(
            "SELECT admission_id, target_session_id, lineage_json, payload_json, outcome FROM session_admissions "
            "WHERE status = 'terminal' ORDER BY seq").fetchall():
        kind, match, why = _admission_input(payload, target, lineage)
        counts = kinds.setdefault(kind, {"matched": 0, "lossless_return_impossible": 0, "no_rule": 0})
        if outcome in _ADMISSION_NEVER_RAN:
            lost.append(f"{admission_id} ({kind}; {_ADMISSION_NEVER_RAN[outcome]})")
        elif outcome not in _ADMISSION_RAN or why:
            unruled.append(f"{admission_id} ({kind}): {why or f'outcome {outcome!r}'}")
            counts["no_rule"] += 1
            continue
        elif match["held"]:
            lost.append(f"{admission_id} ({kind}, {outcome}; {', '.join(match['held'])} only its admission holds)")
        elif (row := _admission_row(conn, match, taken)) is None:
            lost.append(f"{admission_id} ({kind}, {outcome}; no kept user message of its sessions is its input)")
        elif history := _unmatched_history(conn, match, row):
            lost.append(f"{admission_id} ({kind}, {outcome}; {history})")
        else:
            taken.add(row)
            counts["matched"] += 1
            continue
        counts["lossless_return_impossible"] += 1
    if lost:
        refused.append(f"lossless return impossible: {len(lost)} session_admissions inputs exist only in "
                       f"session_admissions: {'; '.join(lost[:10])}{' ...' if len(lost) > 10 else ''}")
    if unruled:
        refused.append(f"session_admissions rows no rule carries to the fork: "
                       f"{'; '.join(unruled[:10])}{' ...' if len(unruled) > 10 else ''}")
    return {"matched_to_messages": sum(c["matched"] for c in kinds.values()), "lossless_return_impossible": len(lost),
            "no_rule": len(unruled), "by_kind": kinds}


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
            col = columns[f"{name}.{column}"] = {"disposition": "converted", "reason": ours[column],
                                                 "non_default_rows": _non_default_rows(conn, name, column, default)}
            if (name, column) in _COLUMN_CONDITIONS:
                breaks, meaning, shown = _COLUMN_CONDITIONS[name, column]
                rows = conn.execute(f"SELECT {', '.join(map(_q, shown))} FROM {_q(name)} WHERE {breaks} "
                                    "ORDER BY rowid").fetchall()
                col["condition_broken_rows"] = len(rows)
                if rows:
                    col["disposition"] = "refused"
                    listed = "; ".join(", ".join(f"{k}={v!r}" for k, v in zip(shown, row)) for row in rows[:10])
                    refused.append(f"lossless return impossible: {name}.{column} {meaning}: {len(rows)} rows: "
                                   f"{listed}{' ...' if len(rows) > 10 else ''}")
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
    built_for = (_RULES["source_schema_version"], _RULES["target_schema_version"])
    if (fence.STORED_SCHEMA_VERSION, FORK_STORED_SCHEMA_VERSION) != built_for:
        refused.append(f"these conversion rules are for schema {built_for[0]} -> {built_for[1]} (fork "
                       f"{_RULES['fork_build']}); this build stores {fence.STORED_SCHEMA_VERSION} and returns to "
                       f"{FORK_STORED_SCHEMA_VERSION}")
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
    plan["labels"] = _plan_labels(conn, inventory, refused)
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
MARKER_SUFFIX = ".fork-return"  # OUTPUT + this: the commit marker (see verify_return)


def _marker(output: Path) -> Path:
    return _sidecar(output, MARKER_SUFFIX)


def _bundle(path: Path) -> list:
    return [_sidecar(path, suffix) for suffix in _SIDECARS]


def _output_files(path: Path) -> list:
    """The output's SQLite files and its commit marker: all new files, none another path's."""
    return [*_bundle(path), _marker(path)]


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
    for candidate in (_output_files(path) if what == "output" else [path]):
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
                if clash := next((f for f in files for p in (_output_files(path) if what == "output" else [path])
                                  if _same_path(f, p)), None):
                    raise ForkReturnError(f"the {what} {path} would be {owner}'s file {clash}")
            named[f"the {what}"] = _output_files(path) if what == "output" else [path]
    output = named["the output"][0]
    report = named["the report"][0] if "the report" in named else None
    work = _directory(work_dir, "the work directory") if work_dir else output.parent
    if _sidecar(source, "-journal").exists():
        raise ForkReturnError("the source has a hot rollback journal; let its writer finish first")
    return source, output, work, report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_state(source: Path) -> dict:
    """The source's bytes: a sha256 of the database and of each journal file it has."""
    return {path.name: _sha256(path) for path in (source, _sidecar(source, "-wal"), _sidecar(source, "-journal"))
            if path is source or path.exists()}


_STAGED, _BUILDING, _MARKING = "source.db", "build.db", "marker.json"  # the only names this run creates, in its scratch


def _scratch(parent: Path) -> Path:
    """A new private directory, created exclusively; the only place this run creates its working files."""
    return Path(tempfile.mkdtemp(prefix=".fork-return-", dir=parent))


def _remove_scratch(directory: Path) -> Optional[str]:
    """Delete the files this run made in *directory*, then the directory if that empties it; never anything else."""
    for name in (_STAGED, _BUILDING, _MARKING):
        for path in _bundle(directory / name):
            path.unlink(missing_ok=True)
    try:
        directory.rmdir()
    except OSError:
        return str(directory)
    return None


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_report(path: Path, report: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ForkReturnError(f"refusing to overwrite {path}: it appeared during the export") from exc
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")


def _verify_source(source: Path, before: dict, report: dict) -> Optional[str]:
    """Read the source's bytes again: None when they are the ones first read, else why the export fails."""
    try:
        report["source_unchanged"] = _source_state(source) == before
    except OSError as exc:
        report["source_unchanged"], report["source_check"] = False, f"unverifiable: {exc}"
        return f"the source could not be read again to verify it: {exc}"
    return None if report["source_unchanged"] else "the source changed while it was read"


def _ours(path: Path, identity: tuple) -> bool:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False
    return (st.st_dev, st.st_ino) == identity


def _retract(output: Path, identity: tuple, report: dict) -> None:
    """Take back the output this run published, and only that file: it is moved into a new private directory and
    removed there once it is confirmed to be the file this run linked; a file another process put at the output is
    put back. ``published`` says whether this run's output is still at the output path; what could not be taken
    back is named in ``residual``."""
    if not _ours(output, identity):
        report["published"] = False
        report["retraction"] = ("the output path no longer holds this run's output; whatever is there now is "
                                "another process's and was left untouched")
        return
    aside = None
    try:
        aside = _scratch(output.parent) / _BUILDING
        os.rename(output, aside)  # atomic: what is checked below is exactly what left the output path
        if not _ours(aside, identity):  # replaced between the check and the move: put it back, never delete it
            os.link(aside, output)
            aside.unlink()
            report["published"] = False
            report["retraction"] = "the output path holds another process's file; it was left in place"
        else:
            aside.unlink()
            report["published"] = False
            report["retraction"] = "retracted: this run's output was removed"
    except OSError as exc:
        report["published"] = _ours(output, identity)
        where = output if report["published"] else aside
        report["residual"] = {"path": str(where), "state": (
            f"this run's output, published before the export failed and not removed ({exc}); it is not a "
            "verified return of the source" if report["published"] or aside and _ours(aside, identity) else
            f"another process's file, moved aside while retracting and not put back at {output} ({exc})")}
    finally:
        if aside is not None and report.get("residual", {}).get("path") != str(aside):
            _remove_scratch(aside.parent)
    try:
        _fsync_directory(output.parent)
    except OSError:
        pass  # durability of the removal only; the path's state is already what the report says


# ── The commit point ──
#
# An export succeeded exactly when its commit marker, OUTPUT + MARKER_SUFFIX, exists and names OUTPUT's bytes. The
# marker is linked once, exclusively, after the output is published and every check (the source read again
# included) passed: that link is the commit point. Before it nothing the run left counts: an output without its
# marker (a run killed after publishing it, or one whose retraction failed) is not a return, whatever any report
# says. After it nothing is taken back: a cancel, or a report that cannot be written, leaves the committed output and
# its marker. The report is written after the marker and is an account of the run, never the verdict. A consumer
# runs verify_return (``--verify OUTPUT``) before the fork first opens OUTPUT: the fork's first write changes it.

_MARKER_FORMAT = "hermes_state_fork_return/1"


def _stage_marker(output: Path, record: dict) -> tuple:
    """The marker's content, durable in a new private directory beside the output (so it links on the output's
    filesystem): ``(directory, identity of the staged file)``."""
    directory = _scratch(output.parent)
    try:
        fd = os.open(directory / _MARKING, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            st = os.fstat(handle.fileno())
    except BaseException:
        _remove_scratch(directory)
        raise
    return directory, (st.st_dev, st.st_ino)


def verify_return(output) -> dict:
    """The rule a consumer applies: *output* is a successful return only when its commit marker exists and names its
    exact bytes, and no journal beside it holds more. Returns the marker's record; raises ForkReturnError with the
    reason otherwise (no marker means the export never reached its commit point)."""
    output = Path(output)
    marker = _marker(output)
    try:
        fd = os.open(marker, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ForkReturnError(f"{output} has no commit marker {marker} ({exc.strerror}): the export that made it did "
                              "not reach its commit point") from exc
    with os.fdopen(fd) as handle:
        try:
            record = json.load(handle)
        except ValueError as exc:
            raise ForkReturnError(f"the commit marker {marker} is not this tool's record: {exc}") from exc
    if not isinstance(record, dict) or record.get("format") != _MARKER_FORMAT or record.get("output") != output.name:
        raise ForkReturnError(f"the commit marker {marker} is not this tool's record for {output.name}")
    for suffix in ("-wal", "-journal"):
        if (journal := _sidecar(output, suffix)).exists() and (suffix == "-journal" or journal.stat().st_size):
            raise ForkReturnError(f"{journal} exists: {output} has been opened for writing since it was committed")
    try:
        size, digest = output.stat().st_size, _sha256(output)
    except OSError as exc:
        raise ForkReturnError(f"{output} cannot be read: {exc}") from exc
    if (size, digest) != (record.get("size"), record.get("sha256")):
        raise ForkReturnError(f"{output} is not the file its commit marker names (sha256 {digest}, {size} bytes; the "
                              f"marker names {record.get('sha256')}, {record.get('size')} bytes)")
    return record


def _stage_wal(source: Path, work: Path) -> tuple:
    """``(staging directory or None, the file to read)``: a source whose WAL holds pages the main file lacks is read
    through a private copy with them folded in, never folded into the source."""
    if not (_sidecar(source, "-wal").exists() and _sidecar(source, "-wal").stat().st_size):
        return None, source
    stage = _scratch(work)
    try:
        view = stage / _STAGED
        shutil.copyfile(source, view)
        shutil.copyfile(_sidecar(source, "-wal"), _sidecar(view, "-wal"))
        conn = sqlite3.connect(view, isolation_level=None)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("PRAGMA journal_mode=DELETE")
        finally:
            conn.close()
    except BaseException:
        _remove_scratch(stage)
        raise
    return stage, view


def _record_build(report: dict, lineage, differences: dict, integrity: list) -> None:
    report.update(
        output_lineage={"lineage": lineage.lineage, "schema_version": lineage.stored,
                        "fence_generations": sorted(lineage.fence_literals), "fork_store": lineage.migratable},
        differences=differences, loss_detected=bool(differences), integrity_check=integrity)
    if differences:
        report["refused"].append(f"the output differs from the source: {sorted(differences)}")
    if not lineage.migratable:
        report["refused"].append("the output is not exactly the fork's fenced 1030 store")
    if integrity != ["ok"]:
        report["refused"].append(f"integrity_check: {integrity[:5]}")


def _check_published(source: Path, before: dict, output: Path, identity: tuple, digest: str, report: dict) -> None:
    """The last check before the commit point: the source is still the bytes this output was made from, and the
    output path still holds the file this run linked, unwritten."""
    if why := _verify_source(source, before, report):
        report["refused"].append(why)
    if not _ours(output, identity):
        report["published"] = False
        report["refused"].append(f"the output {output} was replaced or removed by another process")
    elif _sha256(output) != digest:
        report["refused"].append(f"the output {output} was written by another process after it was published")


def _clean(report: dict, *directories) -> None:
    for directory in directories:
        if directory is not None and (left := _remove_scratch(directory)):
            report.setdefault("scratch_left", []).append(left)


def _final_report(report_file: Optional[Path], report: dict) -> None:
    """Write the report once, last. Before the commit point a report that cannot be written fails the export;
    after it the committed return stands and only its account is missing (``report_error``)."""
    if not report_file:
        return
    try:
        _write_report(report_file, report)
    except ForkReturnError as error:
        if not report["committed"]:
            raise
        report["report_error"] = str(error)


def return_to_fork(source, output, *, work_dir=None, report_path=None) -> dict:
    """Write *source* (a settled 1033 store) to *output* as the live fork's fenced 1030 store; return the report.

    The output is built and verified under a private name, published by one exclusive link once every check passed
    and the source was read again unchanged, then checked again (the source, and that the output path holds this
    run's unchanged file) and committed by linking its marker. Any failure or cancel before the commit point takes
    the published output back, only while it is still the file this run linked; none after it does. Raises
    ForkReturnRefused (carrying the report) when anything cannot be carried or the source changed, ForkReturnError on
    unusable paths, ForkReturnResidual when a published output could not be taken back. The report's ``committed``
    (and ``succeeded``) is whether the marker is this run's, ``published`` the output path's state; a report that
    cannot be written after the commit point is named in ``report_error``."""
    source, output, work, report_file = _validated(source, output, work_dir, report_path)
    try:
        before = _source_state(source)
    except OSError as exc:
        raise ForkReturnError(f"cannot read the source: {exc}") from exc
    wal_mode = _is_wal_file(source)
    marker = _marker(output)
    report = {"operation": "return_to_fork", "source": str(source), "output": str(output), "marker": str(marker),
              "rules": _RULES, "target_schema_version": FORK_STORED_SCHEMA_VERSION, "published": False,
              "committed": False, "succeeded": False, "refused": [], "source_unchanged": False}
    stage = build = identity = pinned = marking = marker_id = None
    reported = verified = False
    try:
        stage, view = _stage_wal(source, work)
        conn = sqlite3.connect(_ro_uri(view), uri=True, isolation_level=None)
        try:
            inventory = _inventory(conn)
            plan = _plan(conn, inventory)
        finally:
            conn.close()
        report.update(tables=plan["tables"], columns=plan["columns"], state_meta=plan["state_meta"],
                      schema_objects=plan["objects"], labels=plan["labels"])
        if plan["refused"]:
            report["refused"] = plan["refused"]
            raise ForkReturnRefused(report)

        build = _scratch(output.parent)
        building = build / _BUILDING
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
        _record_build(report, lineage, differences, integrity)
        # The staging copy is done with: removed before the source is read again, so that read is after it.
        _clean(report, stage)
        stage = None
        if why := _verify_source(source, before, report):
            report["refused"].append(why)
        if report["refused"]:
            raise ForkReturnRefused(report)

        # Publish: one exclusive link of the verified file, pinned open so its identity cannot be reused.
        digest = _sha256(building)
        pinned = os.open(building, os.O_RDONLY)
        os.fsync(pinned)
        st = os.fstat(pinned)
        try:
            os.link(building, output)  # exclusive: a file that appeared at the output since validation is kept
        except FileExistsError as exc:
            raise ForkReturnError(f"refusing to overwrite {output}: it appeared during the export") from exc
        identity, report["published"] = (st.st_dev, st.st_ino), True
        _fsync_directory(output.parent)
        _clean(report, build)
        build = None
        _check_published(source, before, output, identity, digest, report)
        verified = True
        if report["refused"]:
            raise ForkReturnRefused(report)
        marking, marker_id = _stage_marker(output, {
            "format": _MARKER_FORMAT, "output": output.name, "size": st.st_size, "sha256": digest,
            "source": str(source), "source_state": before, "rules": _RULES})
        try:
            os.link(marking / _MARKING, marker)  # the commit point
        except FileExistsError as exc:
            raise ForkReturnError(f"refusing to overwrite {marker}: it appeared during the export") from exc
        report["committed"] = report["succeeded"] = True
        _fsync_directory(output.parent)
        return report
    except BaseException as exc:
        if marker_id is not None and _ours(marker, marker_id):
            # Past the commit point (the link may have landed just before the interruption): nothing is taken back.
            report["committed"] = report["succeeded"] = True
            report["after_commit"] = f"{type(exc).__name__}: {exc}"
            raise
        report["committed"] = report["succeeded"] = False
        if report["published"]:
            _retract(output, identity, report)
            if report.get("residual"):
                if report_file and not reported:
                    reported = True
                    try:
                        _write_report(report_file, report)
                    except ForkReturnError as error:  # the residual is what must reach the caller
                        report["report_error"] = str(error)
                raise ForkReturnResidual(report) from exc
        raise
    finally:
        if pinned is not None:
            os.close(pinned)
        _clean(report, stage, build, marking)
        if not verified:
            _verify_source(source, before, report)  # what the report says is a real read, on every exit
        if not reported:
            _final_report(report_file, report)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hermes_state_fork_return",
                                     description="Return this build's state.db to the live fork's schema 1030.")
    parser.add_argument("source", nargs="?", help="this build's state.db (schema_version 1033); never written")
    parser.add_argument("output", nargs="?", help="new file for the fork's 1030 store; must not exist")
    parser.add_argument("--work-dir", help="where a WAL-bearing source is staged (default: the output's directory)")
    parser.add_argument("--report", help="write the JSON report here")
    parser.add_argument("--verify", metavar="OUTPUT", help="exit 0 only when OUTPUT is a committed return: its "
                        "commit marker names its exact bytes (check before the fork first opens it)")
    args = parser.parse_args(argv)
    if args.verify:
        if args.source or args.output or args.work_dir or args.report:
            parser.error("--verify takes only OUTPUT")
        try:
            record = verify_return(args.verify)
        except ForkReturnError as exc:
            print(f"Not a committed return: {exc}", file=sys.stderr)
            return 2
        print(f"Committed return of {record['source']}: {args.verify} (sha256 {record['sha256']})")
        return 0
    if not (args.source and args.output):
        parser.error("SOURCE and OUTPUT are required")

    def cancel(signum, _frame):
        raise SystemExit(128 + signum)  # a terminated export still removes its scratch and reports

    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        report = return_to_fork(args.source, args.output, work_dir=args.work_dir, report_path=args.report)
    except ForkReturnResidual as exc:
        residual = exc.report["residual"]
        print(f"Failed, and what it published is still there: {residual['path']}: {residual['state']}",
              *(f"  - {r}" for r in exc.report["refused"]), sep="\n", file=sys.stderr)
        return 3
    except ForkReturnRefused as exc:
        done = exc.report.get("retraction") or "nothing was published"
        print(f"Refused; {done}:", *(f"  - {r}" for r in exc.report["refused"]), sep="\n", file=sys.stderr)
        return 2
    except ForkReturnError as exc:
        print(f"Refused; nothing was published: {exc}", file=sys.stderr)
        return 2
    finally:
        signal.signal(signal.SIGTERM, previous)
    tables = report["tables"]
    print(f"Returned to the fork's schema {FORK_STORED_SCHEMA_VERSION}: {report['output']} (committed: "
          f"{report['marker']})")
    print(f"  tables preserved: {sum(v['disposition'] == 'preserved' for v in tables.values())}; converted: "
          + (", ".join(f"{k} ({v['rows']} rows)" for k, v in tables.items() if v["disposition"] == "converted")
             or "none"))
    print("  columns converted: " + ", ".join(f"{k} ({v['non_default_rows']} set)" for k, v in report["columns"].items()))
    if report.get("report_error"):
        print(f"The return is committed, but its report was not written: {report['report_error']}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
