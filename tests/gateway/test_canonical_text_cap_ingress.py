"""An event refused at canonical ingress for its length is named in the gateway's log.

Regression for agent-control-plane #1111: on 2026-10-16 a 17,475-char thread reply was answered
400 ``canonical_invalid_request`` and the gateway recorded nothing, so the refusal was visible only
in the bridge's own log. The answer the bridge reads stays the same (it tells the sender from it);
the gateway's log names the refused event by its correlation and sizes, never by its text.
"""

import asyncio
import json
import logging
import os
import secrets
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.hermes_home_pin import import_home, pinned_hermes_home, tmp_root

try:
    import pwd  # the account database names the real home even when HOME is redirected
except ImportError:  # not POSIX
    pwd = None

# As in test_canonical_record_cas.py: HOME, HERMES_HOME and the home override name a fresh tmp home
# while this module imports Hermes and while its tests run, and a resolver that still names the
# real Hermes home fails the test before anything is written.
_REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir) if pwd is not None else Path.home()
_LIVE_ROOT = (_REAL_HOME / ".hermes").resolve()
_PINNED_ROOT = tmp_root("hermes-text-cap-")  # removed at exit by this process only
if _PINNED_ROOT.is_relative_to(_LIVE_ROOT) or _LIVE_ROOT.is_relative_to(_PINNED_ROOT):
    raise RuntimeError(f"the pinned test home {_PINNED_ROOT} overlaps the real home")

# The imports run pinned to the tmp home all pinned modules import under, so a path they cache names
# no module's test home; the pin is put back (in ``finally``) as soon as they have run: collecting
# this module leaves no home, override or home cache behind for the modules collected after it.
with pinned_hermes_home(import_home(), importer=__name__):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    from gateway.canonical_surface import _MAX_TEXT_CHARS
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter, _api_request_profile


@pytest.fixture(autouse=True, scope="module")
def _pin_the_module_home():
    """The same pin while this module's tests run, owned by this fixture and put back on teardown."""
    with pinned_hermes_home(_PINNED_ROOT):
        yield


_ROUTE = "/v1/canonical-surface/events"


@pytest.fixture
def post(tmp_path, monkeypatch):
    """POST one canonical event through the real handler; no runner is attached, so an event that
    passes ingress answers 503 rather than running anything."""
    home = tmp_path / "hermes-home"
    if home.resolve().is_relative_to(_LIVE_ROOT):
        pytest.fail(f"the test home resolves into the live Hermes home: {home}")
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    override = set_hermes_home_override(home)
    import hermes_state
    from hermes_cli.config import get_config_path

    for path in (get_config_path(), hermes_state._default_db_path()):
        if Path(path).resolve().is_relative_to(_LIVE_ROOT):
            reset_hermes_home_override(override)
            pytest.fail(f"a resolver names the live Hermes home: {path}")
    key = secrets.token_hex(32)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key}))
    [handler] = [h for method, path, h in adapter._http_route_table() if (method, path) == ("POST", _ROUTE)]

    def send(text):
        raw = json.dumps({"binding": "canonical", "event_id": "evt-1", "author_id": "author",
                          "channel_id": "channel", "text": text}).encode()

        async def read():
            return raw

        async def go():
            request = SimpleNamespace(headers={"Authorization": f"Bearer {key}"}, read=read, method="POST",
                                      path_qs=_ROUTE, transport=None)
            token = _api_request_profile.set(None)
            try:
                response = await handler(request)
            finally:
                _api_request_profile.reset(token)
            return response.status, response.text
        return asyncio.run(go())

    try:
        yield send
    finally:
        reset_hermes_home_override(override)
        adapter._response_store.close()


def _cap_records(caplog):
    return [r.getMessage() for r in caplog.records if "text_over_limit" in r.getMessage()]


def test_an_event_over_the_cap_is_refused_unchanged_and_logged_by_correlation_never_text(post, caplog):
    caplog.set_level(logging.INFO)
    text = "Q" * (_MAX_TEXT_CHARS + 1)

    status, body = post(text)

    assert (status, body) == (400, json.dumps(
        {"error": {"code": "canonical_invalid_request", "message": "Canonical request rejected."}}))
    [message] = _cap_records(caplog)
    for part in ("binding='canonical'", "event_id='evt-1'", f"chars={len(text)}", f"limit={_MAX_TEXT_CHARS}"):
        assert part in message, message
    assert "Q" * 16 not in message


def test_control_an_event_at_the_cap_passes_ingress_and_is_not_logged(post, caplog):
    caplog.set_level(logging.INFO)

    status, body = post("Q" * _MAX_TEXT_CHARS)

    assert (status, json.loads(body)["error"]["code"]) == (503, "canonical_unavailable")
    assert _cap_records(caplog) == []
