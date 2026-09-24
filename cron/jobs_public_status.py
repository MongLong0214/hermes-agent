"""What of a cron job record may leave the host: closed failure text and the served field sets.

Raw failure text carries provider response bodies, filesystem paths, script stderr tails and
ids, and a deny-list scrubber only removes the shapes it already knows. So ``last_error``,
``last_delivery_error``, ``last_fire_error`` and the cause clause of the chat failure notice are
always one of the fixed values built from the tables below. The raw text stays where the operator
already reads it on this machine: the private run output, the executions ledger
(``hermes cron runs``) and the redacting logs.
"""

from __future__ import annotations

import functools
import re
from copy import deepcopy
from typing import Any, Dict, Optional

RUNS_POINTER = "Details: `hermes cron runs`."
DELIVERY_FAILED = "Delivery failed. Details: the gateway log (`hermes logs --level WARNING`)."
FIRE_FORWARD_FAILED = (
    "The scheduled fire could not be handed to the gateway: it may be down, or its API server "
    "is not enabled (API_SERVER_KEY)")
NEXT_RUN_UNCOMPUTABLE = (
    "Failed to compute next run for recurring schedule (is the 'croniter' package "
    "installed in the gateway's Python env?)")

# Served by the bearer-token ``/api/jobs`` routes. A record field leaves the host only by being
# listed: fire/run claims (owner pid and host), ``origin`` (chat and user ids), workdir, script and
# model routing stay.
PUBLIC_CRON_JOB_FIELDS = (
    "id", "name", "prompt", "skill", "skills", "schedule", "schedule_display", "repeat", "deliver",
    "enabled", "state", "paused_at", "paused_reason", "next_run_at", "last_run_at", "last_status",
    "last_delivery_error", "last_fire_error",
)
# The dashboard edits a job's execution config and shows its run error (web/src/lib/api.ts
# ``CronJob``), so it serves those too; claims and ``origin`` still stay on the host.
DASHBOARD_CRON_JOB_FIELDS = PUBLIC_CRON_JOB_FIELDS + (
    "script", "no_agent", "model", "provider", "base_url", "context_from", "enabled_toolsets",
    "workdir", "last_error",
)

PROVIDER_KIND_PREFIX = "provider:"

# Kind -> what happened, with the job as subject. Provider kinds are not listed: their clause
# comes from the gloss table shared with chat turns (agent/turn_failure_copy.py).
_CAUSES: dict[str, str] = {
    "blocked_config": "the pre-run configuration check found a problem",
    "missing_credential": "the provider credential for this job is missing",
    "delivery_unconfigured": "its delivery target is not configured on this gateway",
    "skill_not_ready": "an attached skill is not ready",
    "mcp_unresolved": "an MCP server it needs has no tools for this profile",
    "drift": "its inference config drifted since it was created, so it was skipped to prevent "
             "unintended spend",
    "script_timeout": "its script timed out",
    "inactivity": "it stopped doing anything for too long and was cut off",
    "shutdown": "the gateway shut down before the run finished",
    "forced_release": "an earlier run never released the scheduler's in-flight guard",
    "worker_dispatch": "the restart-safe cron worker could not be started",
    "unrunnable": "it has nothing to run, so it was paused",
    "import_error": "one of Hermes's own modules failed to import",
    "agent_declared": "the job reported its own failure",
    "script_failed": "its script exited with an error",
    "unknown": "the run ended with an error",
}

_BLOCKED_MARKER = re.compile(r"\[blocked_config[^\]]*\]\s*")

# Pre-run check verdict prefix (cron/scheduler_preflight.py) -> kind; anything else blocked is
# the generic ``blocked_config``.
_PREFLIGHT_KINDS: tuple[tuple[str, str], ...] = (
    ("provider credential missing", "missing_credential"),
    ("delivery platform ", "delivery_unconfigured"),
    ("attached skill ", "skill_not_ready"),
    ("MCP server(s) ", "mcp_unresolved"),
)

# First match wins. These are scheduler/script shapes whose text can contain "timed out" or a
# status code, so they are matched before the provider classifier could blame the model service.
_KIND_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("drift", re.compile(r"skipped to prevent unintended spend\b", re.I)),
    ("script_timeout", re.compile(r"script timed out\b", re.I)),
    ("inactivity", re.compile(r".*?\bidle for \d+s\s*\(limit \d+s\)", re.I | re.S)),
    ("shutdown", re.compile(r"interrupted by (?:gateway )?shutdown\b|gateway shutdown\b", re.I)),
    ("forced_release", re.compile(r"stale in-flight claim force-released\b", re.I)),
    ("worker_dispatch", re.compile(r"restart-safe cron worker dispatch failed\b", re.I)),
)
_IMPORT_FAILURE = re.compile(r".*?(?:cannot import name|modulenotfounderror|importerror)", re.I | re.S)


def _unrunnable_reasons() -> tuple[str, ...]:
    from cron.jobs import EMPTY_PAYLOAD_ERROR, NO_AGENT_WITHOUT_SCRIPT_ERROR

    return EMPTY_PAYLOAD_ERROR, NO_AGENT_WITHOUT_SCRIPT_ERROR


def _provider_cause(reason: str) -> Optional[str]:
    from agent.turn_failure_copy import failure_cause_gloss

    return failure_cause_gloss(reason, subject="this job", possessive="the job's")


def failure_kind(text: str, *, no_agent: bool = False) -> str:
    """Closed kind for raw failure *text*; ``provider:<reason>`` for a glossed provider verdict."""
    text = (text or "").strip()
    blocked = _BLOCKED_MARKER.search(text)
    if blocked:
        verdict = text[blocked.end():]
        return next((kind for prefix, kind in _PREFLIGHT_KINDS if verdict.startswith(prefix)),
                    "blocked_config")
    if text in _unrunnable_reasons():
        return "unrunnable"
    for kind, pattern in _KIND_PATTERNS:
        if pattern.match(text):
            return kind
    # no_agent jobs never reach a model or Hermes's own imports: a script's "429", "timed out" or
    # ImportError is the script's own problem.
    if not no_agent:
        if _IMPORT_FAILURE.match(text):
            return "import_error"
        from cron.scheduler_failure_copy import classify_cron_failure_reason

        reason = classify_cron_failure_reason(text)
        if _provider_cause(reason) is not None:
            return PROVIDER_KIND_PREFIX + reason
    return "script_failed" if no_agent else "unknown"


def provider_reason(kind: str) -> Optional[str]:
    """The ``FailoverReason`` value of a provider kind, else None."""
    return kind[len(PROVIDER_KIND_PREFIX):] if kind.startswith(PROVIDER_KIND_PREFIX) else None


def failure_cause(kind: str) -> str:
    """Plain clause for *kind* (never raw text); unknown kinds read as a generic error."""
    reason = provider_reason(kind)
    cause = _provider_cause(reason) if reason is not None else _CAUSES.get(kind)
    return cause or _CAUSES["unknown"]


def _label(kind: str) -> str:
    cause = failure_cause(kind)
    return f"{cause[0].upper()}{cause[1:]}. {RUNS_POINTER}"


@functools.lru_cache(maxsize=1)
def _closed_run_labels() -> frozenset:
    from agent.turn_failure_copy import FAILURE_CAUSE_GLOSS

    kinds = [*_CAUSES, *(PROVIDER_KIND_PREFIX + reason for reason in FAILURE_CAUSE_GLOSS)]
    return frozenset(_label(kind) for kind in kinds) | {NEXT_RUN_UNCOMPUTABLE}


def public_run_error(
    value: Any, *, no_agent: bool = False, agent_declared: bool = False,
) -> Optional[str]:
    """The persisted/public ``last_error`` for a run's raw error: None stays None, a closed
    label stays itself, anything else becomes the fixed label of its kind."""
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    if text in _closed_run_labels():
        return text
    return _label("agent_declared" if agent_declared else failure_kind(text, no_agent=no_agent))


def public_delivery_error(value: Any) -> Optional[str]:
    """The persisted/public ``last_delivery_error``: None/blank stays None, any failure is the one
    fixed label (the adapter text is logged when the outcome is recorded)."""
    if value is None or not str(value).strip():
        return None
    return DELIVERY_FAILED


def public_fire_error(value: Any) -> Optional[Dict[str, Any]]:
    """The persisted/public ``last_fire_error``: no stamp (the CLI's test: a dict with a detail)
    stays None, a stamp keeps only its time (the forwarder's reason is logged when recorded)."""
    if not (isinstance(value, dict) and value.get("detail")):
        return None
    at = value.get("at")
    return {"at": at if isinstance(at, str) else None, "detail": FIRE_FORWARD_FAILED}


def closed_status_fields(job: Dict[str, Any]) -> Dict[str, Any]:
    """*job*'s failure fields as served anywhere off the host, including records written raw
    before the store closed them."""
    return {
        "last_error": public_run_error(job.get("last_error"), no_agent=bool(job.get("no_agent"))),
        "last_delivery_error": public_delivery_error(job.get("last_delivery_error")),
        "last_fire_error": public_fire_error(job.get("last_fire_error")),
    }


def project_cron_job(
    record: Dict[str, Any], fields: tuple[str, ...] = PUBLIC_CRON_JOB_FIELDS,
) -> Dict[str, Any]:
    """A fresh copy of *record* holding only *fields*, its failure fields closed."""
    closed = closed_status_fields(record)
    return {field: closed[field] if field in closed else deepcopy(record[field])
            for field in fields if field in record}
