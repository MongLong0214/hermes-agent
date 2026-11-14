"""Platform activation evidence comes only from the profile's own secret scope.

``TELEGRAM_*`` in ANOTHER profile's ``.env`` says nothing about this profile: a multiplexing (or
profile-routing) process holds the launch profile's ``.env`` in ``os.environ``, and a scope bound for a
different home is that home's. Neither may activate the plugin for the profile being configured, while
the profile's own ``.env``, its own bound scope and a single-profile process env still do.
"""

import os

import pytest

from agent.secret_scope import reset_multiplex_context, reset_secret_scope, set_multiplex_context, set_secret_scope
from gateway.platform_activation import activation_predicate
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_TOKEN = "TELEGRAM_BOT_TOKEN"


@pytest.fixture
def homes(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("TELEGRAM_"):
            monkeypatch.delenv(name)
    launch, work, other = (tmp_path / name for name in ("launch", "work", "other"))
    for home in (launch, work, other):
        home.mkdir()
        (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))  # the process's own (launch) profile
    return launch, work, other


def _activates_telegram(home) -> bool:
    return activation_predicate(yaml_cfg={}, home=home)("telegram")


def _launch_profile_loaded_its_dotenv(monkeypatch, launch):
    (launch / ".env").write_text(f"{_TOKEN}=123456:launch-only\n", encoding="utf-8")
    monkeypatch.setenv(_TOKEN, "123456:launch-only")  # the launch process loads its .env into os.environ


@pytest.mark.parametrize("mode", ["multiplexed", "routed-override"])
def test_launch_profiles_telegram_env_does_not_activate_another_profile(homes, monkeypatch, mode):
    launch, work, _ = homes
    _launch_profile_loaded_its_dotenv(monkeypatch, launch)
    token = set_multiplex_context(True) if mode == "multiplexed" else set_hermes_home_override(str(work))
    try:
        assert _activates_telegram(work) is False
        assert _activates_telegram(launch) is True  # still its own evidence
    finally:
        (reset_multiplex_context if mode == "multiplexed" else reset_hermes_home_override)(token)


def test_scope_bound_for_another_profile_is_not_evidence(homes):
    _, work, other = homes
    token = set_secret_scope({_TOKEN: "123456:other-only"}, profile_home=str(other))
    try:
        assert _activates_telegram(work) is False
        assert _activates_telegram(other) is True
    finally:
        reset_secret_scope(token)
    token = set_secret_scope({_TOKEN: "123456:unstamped"})
    try:
        assert _activates_telegram(work) is False  # an unstamped scope belongs to no profile we can name
    finally:
        reset_secret_scope(token)


def test_profiles_own_evidence_still_activates(homes, monkeypatch):
    launch, work, _ = homes
    (work / ".env").write_text(f"{_TOKEN}=123456:work\n", encoding="utf-8")
    token = set_multiplex_context(True)
    try:
        assert _activates_telegram(work) is True  # its own .env
        assert _activates_telegram(launch) is False  # the secondary's .env is not the launch profile's
    finally:
        reset_multiplex_context(token)
    (work / ".env").write_text("", encoding="utf-8")
    token = set_secret_scope({"TELEGRAM_ALLOWED_USERS": "1"}, profile_home=str(work))  # e.g. an external source
    try:
        assert _activates_telegram(work) is True
    finally:
        reset_secret_scope(token)


def test_single_profile_process_env_still_activates(homes, monkeypatch):
    """systemd ``Environment=`` / ``docker -e`` / ``op run``: no file holds the token, only the env."""
    launch, work, _ = homes
    monkeypatch.setenv(_TOKEN, "123456:env-only")
    assert _activates_telegram(launch) is True
    assert _activates_telegram(None) is True  # the active (process) home
    assert _activates_telegram(work) is False
