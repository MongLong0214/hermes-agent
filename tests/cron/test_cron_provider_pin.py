"""Unpinned cron jobs run on the main agent model at fire time; ``pinned`` locks it; legacy
snapshot records fail closed while ``cron.model_drift_guard`` is on.

Contract:
  - run_job() resolves per-job pin > cron.model / cron.model_provider > the main agent model
    (config ``model:``). New jobs carry no creation-time snapshot.
  - A legacy record that still carries ``provider_snapshot`` / ``model_snapshot`` is skipped
    before any agent is built (no model spend) when an axis that is unpinned and not supplied by
    the cron fleet default drifted from its snapshot. The provider axis compares the PRIMARY
    route, never a fallback reached because the primary failed. With
    ``cron.model_drift_guard: false`` such records follow the main model like new jobs.
  - create_job(pinned=True) / update_job({"pinned": True}) lock the CURRENT main provider+model
    onto the job as an ordinary per-job pin; ``pinned=False`` releases both.

These tests exercise the full run_job path (real imports, mocked AIAgent +
resolve_runtime_provider against a temp HERMES_HOME) and the job-store pin helpers.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import run_job


def _base_job(**overrides):
    job = {
        "id": "pin-test",
        "name": "pin test",
        "prompt": "hello",
        "model": None,
        "provider": None,
        "base_url": None,
    }
    job.update(overrides)
    return job


def _run(job, tmp_path, *, current_provider="openrouter", current_model=None, cron_model=None,
         cron_model_provider=None, drift_guard=None, extra_yaml="", resolver=None):
    """Drive run_job against a temp config.yaml whose ``model.default`` / ``model.provider`` are
    the CURRENT global defaults. Returns ``(success, error, agent_kwargs, resolve_kwargs)`` where
    the last two are the kwargs AIAgent / resolve_runtime_provider were called with (None when
    never called). ``resolver`` replaces the default resolve_runtime_provider stand-in."""
    config_yaml = ""
    if current_model or current_provider:
        config_yaml += "model:\n"
        if current_model:
            config_yaml += f"  default: {current_model}\n"
        if current_provider:
            config_yaml += f"  provider: {current_provider}\n"
    cron_lines = []
    if cron_model is not None:
        cron_lines.append(f"  model: {cron_model}")
    if cron_model_provider is not None:
        cron_lines.append(f"  model_provider: {cron_model_provider}")
    if drift_guard is not None:
        cron_lines.append(f"  model_drift_guard: {str(drift_guard).lower()}")
    if cron_lines:
        config_yaml += "cron:\n" + "\n".join(cron_lines) + "\n"
    (tmp_path / "config.yaml").write_text(config_yaml + extra_yaml)

    resolve_kwargs = {}

    def _resolve(**kwargs):
        resolve_kwargs.update(kwargs)
        if resolver is not None:
            return resolver(**kwargs)
        return {
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "provider": kwargs.get("requested") or current_provider,
            "api_mode": "chat_completions",
        }

    fake_db = MagicMock()
    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._get_hermes_home", return_value=tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=fake_db), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_resolve), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent
        success, _output, _final, error = run_job(job)
        agent_kwargs = mock_agent_cls.call_args.kwargs if mock_agent_cls.called else None
    return success, error, agent_kwargs, (resolve_kwargs or None)


class TestLegacySnapshotDriftGuard:
    def test_legacy_snapshot_fails_closed_while_guard_on_and_follows_main_when_off(self, tmp_path):
        """A snapshot-era record never silently moves to a changed main model while the guard
        is on (the default); switching the guard off gives it the upstream follow-main
        semantics."""
        job = _base_job(provider_snapshot="old-provider", model_snapshot="old-model")
        success, error, agent_kwargs, _ = _run(
            job, tmp_path, current_provider="new-provider", current_model="new-model")
        assert success is False
        assert agent_kwargs is None
        assert "no inference call" in (error or "").lower()

        success, error, agent_kwargs, resolve_kwargs = _run(
            _base_job(provider_snapshot="old-provider", model_snapshot="old-model"), tmp_path,
            current_provider="new-provider", current_model="new-model", drift_guard=False)
        assert success is True, error
        assert agent_kwargs["model"] == "new-model"
        assert resolve_kwargs["requested"] is None

    def test_legacy_model_snapshot_axis_fails_closed(self, tmp_path):
        """The model axis is guarded on its own: same provider, different main model."""
        job = _base_job(provider_snapshot="openrouter", model_snapshot="old-model")
        success, _error, agent_kwargs, _ = _run(
            job, tmp_path, current_provider="openrouter", current_model="new-model")
        assert success is False
        assert agent_kwargs is None

    def test_fleet_default_axes_are_exempt_from_the_legacy_drift_guard(self, tmp_path):
        """cron.model_provider / cron.model are the operator's explicit cron routing, not drift."""
        job = _base_job(provider_snapshot="openrouter", model_snapshot="old-model")
        success, error, agent_kwargs, resolve_kwargs = _run(
            job, tmp_path, current_provider="openrouter", current_model="main-model",
            cron_model="fleet-model", cron_model_provider="nous")
        assert success is True, error
        assert (agent_kwargs["model"], resolve_kwargs["requested"]) == ("fleet-model", "nous")

    def test_legacy_drift_compares_the_primary_route_not_the_fallback(self, tmp_path):
        """A primary that matches the snapshot but fails auth may fall back; that is not drift."""
        from hermes_cli.auth import AuthError

        def resolver(**kwargs):
            if kwargs.get("requested") is None:
                raise AuthError("openrouter token expired", provider="openrouter")
            return {"api_key": "fb-key", "base_url": "https://example.invalid/v1",
                    "provider": kwargs["requested"], "api_mode": "chat_completions"}

        job = _base_job(provider_snapshot="openrouter", model_snapshot="main-model")
        success, error, agent_kwargs, _ = _run(
            job, tmp_path, current_provider="openrouter", current_model="main-model",
            resolver=resolver,
            extra_yaml="fallback_providers:\n  - provider: nous\n    model: fb-model\n")
        assert success is True, error
        assert (agent_kwargs["provider"], agent_kwargs["model"]) == ("nous", "fb-model")

    def test_heal_clear_failure_does_not_fail_a_healthy_run(self, tmp_path, monkeypatch):
        """Re-arming the alert is bookkeeping; a store error there must not cost the run."""
        def broken_store(*_a, **_k):
            raise OSError("jobs.json is read-only")

        monkeypatch.setattr("cron.jobs.set_drift_alert", broken_store)
        job = _base_job(provider_snapshot="openrouter", drift_alerted=True)
        success, error, agent_kwargs, _ = _run(
            job, tmp_path, current_provider="openrouter", current_model="main-model")
        assert success is True, error
        assert agent_kwargs is not None


class TestUnpinnedJobsFollowTheMainModel:
    def test_explicit_pin_then_fleet_default_beat_the_main_model(self, tmp_path):
        pinned = _base_job(provider="pinned-provider", model="pinned-model")
        success, error, agent_kwargs, resolve_kwargs = _run(
            pinned, tmp_path, current_provider="new-provider", current_model="new-model",
            cron_model="fleet-model", cron_model_provider="fleet-provider")
        assert success is True, error
        assert (agent_kwargs["model"], resolve_kwargs["requested"]) == ("pinned-model", "pinned-provider")

        success, error, agent_kwargs, resolve_kwargs = _run(
            _base_job(), tmp_path, current_provider="new-provider", current_model="new-model",
            cron_model="fleet-model", cron_model_provider="fleet-provider")
        assert success is True, error
        assert (agent_kwargs["model"], resolve_kwargs["requested"]) == ("fleet-model", "fleet-provider")

    def test_missing_model_guides_to_user_owned_cli(self, tmp_path, monkeypatch):
        """A missing-model failure cannot advertise agent-owned pinning."""
        monkeypatch.delenv("HERMES_MODEL", raising=False)
        success, error, agent_kwargs, _ = _run(
            _base_job(), tmp_path, current_provider="openrouter", current_model=None)

        assert success is False
        assert agent_kwargs is None
        assert "hermes cron edit pin-test --model <name>" in error
        assert "cronjob action=update" not in error


class TestPinnedLocksTheMainModel:
    """``pinned`` is a lock on the main model at the time it is set, stored as a plain pin."""

    @staticmethod
    def _store(monkeypatch, tmp_path, main_model="main-model", main_provider="openrouter"):
        import cron.jobs as jobs
        (tmp_path / "config.yaml").write_text(f"model:\n  default: {main_model}\n")
        monkeypatch.setattr(jobs, "get_hermes_home", lambda: tmp_path, raising=True)
        state = {"jobs": []}
        monkeypatch.setattr(jobs, "load_jobs", lambda: list(state["jobs"]), raising=True)
        monkeypatch.setattr(jobs, "save_jobs", lambda j: state.__setitem__("jobs", list(j)), raising=True)
        monkeypatch.setattr(jobs, "resolve_job_ref", lambda ref: next(
            (j for j in state["jobs"] if j["id"] == ref), None), raising=True)
        resolver = MagicMock(return_value={"provider": main_provider})
        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolver)
        return jobs, resolver

    def test_pinned_true_locks_then_pinned_false_releases(self, monkeypatch, tmp_path):
        jobs, _ = self._store(monkeypatch, tmp_path)

        from tools.cronjob_job_args import _format_job

        unpinned = jobs.create_job(prompt="do a thing", schedule="every 1 hour")
        assert (unpinned["model"], unpinned["provider"]) == (None, None)
        assert _format_job(unpinned)["pinned"] is False

        locked = jobs.update_job(unpinned["id"], {"pinned": True})
        assert (locked["model"], locked["provider"]) == ("main-model", "openrouter")
        assert _format_job(locked)["pinned"] is True
        assert "pinned" not in jobs.load_jobs()[0]  # derived, never stored

        # The main model moves on; the locked job does not.
        (tmp_path / "config.yaml").write_text("model:\n  default: newer-model\n")
        assert jobs.update_job(locked["id"], {"name": "renamed"})["model"] == "main-model"

        released = jobs.update_job(locked["id"], {"pinned": False})
        assert (released["model"], released["provider"]) == (None, None)

    def test_pinned_never_overrides_an_explicit_model(self, monkeypatch, tmp_path):
        jobs, resolver = self._store(monkeypatch, tmp_path)

        job = jobs.create_job(prompt="do a thing", schedule="every 1 hour", model="my-model",
                              provider="nous", pinned=True)
        assert (job["model"], job["provider"]) == ("my-model", "nous")
        resolver.assert_not_called()

        still = jobs.update_job(job["id"], {"pinned": True, "model": "other-model"})
        assert still["model"] == "other-model"


class TestRuntimeResolutionTargetModel:
    """run_job must resolve the primary provider against the model the job will actually run
    (per-job pin > cron.model > the main agent model), so providers with model-specific
    api_mode routing pick the mode for that model instead of the stale persisted default."""

    def test_primary_resolution_passes_effective_model(self, tmp_path):
        job = _base_job(model="my-pinned-model", provider="openrouter")
        success, error, _agent_kwargs, resolve_kwargs = _run(
            job, tmp_path, current_provider="openrouter", current_model="other-model")

        assert success is True, error
        assert resolve_kwargs["target_model"] == "my-pinned-model"
        assert resolve_kwargs["requested"] == "openrouter"
