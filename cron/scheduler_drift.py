"""Legacy snapshot drift guard: spend guard + alert-once bookkeeping for snapshot-era job records.

New jobs follow the main model (upstream semantics); records created before that still carry
``provider_snapshot`` / ``model_snapshot`` as their effective pin, so while ``cron.model_drift_guard``
is on they are skipped (no model call) with one alert per drift episode instead of silently moving
to a changed main model.

Split out of ``cron.scheduler``, which calls it as ``_drift.<name>``. Facade names are reached
late-bound through ``_sched`` (bottom import) and store helpers through ``cron.jobs`` at call time,
so monkeypatching the defining module works.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from cron.scheduler import _CronJobConfig

# Log-record parity with the origin module.
logger = logging.getLogger("cron.scheduler")

# The drift alert travels on this private job field (popped by the alert commit), never as a marker
# in error text: an agent's failure prose can quote any marker, and must neither impersonate nor
# silence the alert.
_PRIVATE_CRON_DRIFT_SIGNAL = "_private_cron_drift_signal"
_DRIFT_FLEET_DEFAULT_KEY = {"provider": "model_provider", "model": "model"}


def _cron_model_drift_guard_enabled(cfg: dict) -> bool:
    """Spend guard: ON unless ``cron.model_drift_guard`` is literally ``false``."""
    cron_cfg = (cfg or {}).get("cron")
    return not isinstance(cron_cfg, dict) or cron_cfg.get("model_drift_guard", True) is not False


def _primary_provider(resolve_exc: BaseException, requested: Any, jc: _CronJobConfig) -> Any:
    """The route a failed primary resolve asked for; drift is judged on it, not on the fallback."""
    configured = jc.model_cfg.get("provider") if isinstance(jc.model_cfg, dict) else ""
    return getattr(resolve_exc, "provider", "") or requested or configured


def _claim_owner(job: dict) -> Optional[str]:
    claim = job.get("fire_claim")
    return str(claim.get("by") or "") if isinstance(claim, dict) else None


def _legacy_snapshot_drift(job: dict, jc: _CronJobConfig, primary_provider: Any) -> list:
    """``"<axis> '<snapshot>' -> '<current>'"`` for each axis a snapshot-era record would silently
    move on. Following a changed main model would spend on a provider/model nobody chose for it.
    An axis pinned on the job or supplied by the cron fleet default is the operator's explicit
    routing, not drift. ``primary_provider`` is the pre-fallback route: a fallback reached because
    the primary failed is the configured recovery, not drift."""
    if not _cron_model_drift_guard_enabled(jc.cfg):
        return []
    cron_cfg = jc.cfg.get("cron") if isinstance(jc.cfg.get("cron"), dict) else {}
    current = {"provider": primary_provider, "model": jc.model}
    drift = []
    for axis, fleet_key in _DRIFT_FLEET_DEFAULT_KEY.items():
        snapshot = str(job.get(f"{axis}_snapshot") or "").strip().lower()
        now = str(current[axis] or "").strip().lower()
        exempt = str(job.get(axis) or "").strip() or str(cron_cfg.get(fleet_key) or "").strip()
        if snapshot and now and snapshot != now and not exempt:
            drift.append(f"{axis} '{snapshot}' -> '{now}'")
    return drift


def _legacy_drift_block(
    job: dict, job_id: str, job_name: str, jc: _CronJobConfig, runtime: dict,
) -> Optional[tuple]:
    """Skip a drifted legacy record before any agent machinery exists (no model spend), or re-arm
    the alert when the condition healed. Delivery owns the alert-once bit: it is consumed only
    after a target accepted the alert, so neither a failed send nor a stale worker burns it.
    Pops the fallback path's ``runtime["_primary_provider"]``."""
    drift = _legacy_snapshot_drift(
        job, jc, runtime.pop("_primary_provider", runtime.get("provider")))
    if not drift:
        if job.get("drift_alerted"):
            try:
                from cron.jobs import set_drift_alert
                set_drift_alert(job_id, False, expected_fire_owner=_claim_owner(job))
            except Exception:
                logger.warning("Job '%s': could not re-arm the drift alert", job_id, exc_info=True)
        return None
    changes = "; ".join(drift)
    if (job.get("schedule") or {}).get("kind") == "once":
        remedy = ("This one-shot job is used up by this skipped run, so it cannot be edited; on the "
                  "host running Hermes create a new one with `hermes cron create ... --provider "
                  "<provider> --model <model>`.")
    else:
        remedy = ("To run on the new config, on the host running Hermes pin it explicitly: "
                  f"`hermes cron edit {job_id} --provider <provider> --model <model>` (or pin the "
                  "original values to keep them). This alert is sent once; the job stays skipped "
                  "until it is pinned or the config is restored.")
    notice = (f"⛔ Cron '{job.get('name') or job_id}' did not run: its inference config drifted since "
              f"it was created ({changes}) and it is not pinned. Nothing was charged. {remedy} "
              "`cron.model_drift_guard: false` in config.yaml lets such jobs follow the main model.")
    job[_PRIVATE_CRON_DRIFT_SIGNAL] = notice
    logger.warning(
        "Job '%s' (ID: %s): skipped, legacy snapshot drift (%s); no LLM call was made",
        job_name, job_id, changes)
    doc = (f"# Cron Job: {job_name}\n\n**Job ID:** {job_id}\n"
           f"**Run Time:** {_sched._hermes_now().strftime('%Y-%m-%d %H:%M:%S')}\n"
           f"**Status:** SKIPPED (model drift)\n\n{notice}\n")
    return False, doc, "", (
        f"Skipped to prevent unintended spend: legacy snapshot drift ({changes}); "
        "no inference call was made.")


# Bot Chat receipt status -> accepted (True) / still open (None); every other status re-arms.
_BOT_CHAT_OUTCOME = {"settled": True, "queued": None, "claimed": None}


def _parked_outcome(refs: list) -> Optional[bool]:
    """True once any parked handoff was accepted, None while one is still open, else False."""
    outcomes = [_parked_ref_outcome(ref) for ref in refs]
    return True if True in outcomes else None if None in outcomes else False


def _parked_ref_outcome(ref: dict) -> Optional[bool]:
    if "queue" in ref:
        from cron.delivery_queue import get_status
        row = get_status(ref["queue"])
        if row is None:
            return False
        if row["status"] in ("pending", "delivering"):
            return None
        # A tombstoned row keeps only its status.
        if row.get("accepted", row["status"] == "delivered"):
            return True
        return _parked_outcome(json.loads(row.get("parked") or "[]"))
    from cron.bot_chat_delivery import read_pending
    from hermes_cli.profiles import get_profile_dir
    from hermes_constants import get_hermes_home
    from tools.bot_live_delivery import read_delivery_result
    # Same lookup order as the send: a live-owner receipt is authoritative over a deferred record.
    home = (get_profile_dir(ref["profile"]) if ref["profile"] else get_hermes_home()).resolve()
    record = read_delivery_result(home, ref["bot_chat"]) or read_pending(ref["bot_chat"])
    return _BOT_CHAT_OUTCOME.get(record["status"], False) if record else False


def _pending_drift_alert(job: dict) -> Optional[str]:
    """The drift notice this failed run owes; "" when the episode's alert is already out or still
    open, None for any other run. Settles an alert parked on handoffs no target had finished
    (durable-queue send, Bot Chat receipt) through their own lanes: accepted -> consumed; still open
    -> wait (a re-send to Bot Chat is a new, paid turn); any other outcome (failed, cancelled,
    ambiguous, suppressed, ``unknown`` after a gateway died mid-send, a lost record) -> re-armed and
    owed again. Only this job's claim owner writes; a claim lost meanwhile is caught by the fence."""
    notice = job.get(_PRIVATE_CRON_DRIFT_SIGNAL)
    if notice is None:
        return None
    from cron.jobs import get_job, set_drift_alert
    current = get_job(job["id"]) or {}
    if not current.get("drift_alerted"):
        return notice
    parked = current.get("drift_alert_parked")
    if not parked:
        return ""
    try:
        accepted = _parked_outcome(parked)
        if accepted is None:
            return ""
        set_drift_alert(job["id"], accepted, expected_fire_owner=_claim_owner(job))
    except Exception:
        logger.warning("Job '%s': could not settle the parked drift alert", job["id"], exc_info=True)
        return ""
    return "" if accepted else notice


def _commit_drift_alert(job: dict, owner: Optional[str]) -> None:
    """After a delivery: consume the alert-once bit when a target accepted the drift alert, or park
    it on the handoffs no target has finished (settled next tick). A failed write only risks one
    duplicate alert next tick, never a lost one or a failed delivery record."""
    accepted = job.pop("_delivery_accepted", False)
    parked = job.pop("_delivery_parked", None)
    if job.pop(_PRIVATE_CRON_DRIFT_SIGNAL, None) is None or not (accepted or parked):
        return
    try:
        from cron.jobs import set_drift_alert
        set_drift_alert(job["id"], True, parked=None if accepted else parked,
                        expected_fire_owner=owner)
    except Exception:
        logger.warning("Job '%s': could not record the sent drift alert", job["id"], exc_info=True)


# Late-bound origin namespace (see module docstring). Imported LAST so this module is fully
# populated before ``scheduler`` imports from it.
from cron import scheduler as _sched  # noqa: E402
