"""The bridge tells the relaying CTO, never the owner, why a relayed owner instruction did not run.

The gateway refuses a relay its CTO's transcript does not prove (``canonical_owner_relay_unverified``,
gateway/owner_relay.py) before any claim, naming a reason from a closed vocabulary. The shim turns that
into a fixed notice for the thread: the reason, what to check, and that the owner must not be asked to
approve or retype it again. The refusal is pre-claim: nothing ran.
"""
import importlib.util
import os
from pathlib import Path

import pytest

from tests.hermes_home_pin import pinned_hermes_home, tmp_root

_PINNED_ROOT = tmp_root("hermes-shim-relay-")  # removed at exit by this process only


@pytest.fixture(autouse=True, scope="module")
def _pin_the_module_home():
    """HOME and the Hermes home name a fresh tmp dir while this module's tests run (tests/bridge/conftest.py)."""
    with pinned_hermes_home(_PINNED_ROOT):
        yield


_SHIM = Path(os.environ.get("SHIM_PATH") or
             Path(__file__).resolve().parents[2] / "bridge" / "canonical" / "acp-canonical-shim.py")


@pytest.fixture
def shim(tmp_path, monkeypatch):
    env = {
        "CANONICAL_SURFACE_URL": "http://127.0.0.1:9", "API_SERVER_KEY": "fixture-key",
        "CANONICAL_BINDING": "acp-canonical-ceo", "CANONICAL_AUTHOR_ID": "a" * 64,
        "CANONICAL_CHANNEL_ID": "00000000-0000-4000-8000-000000000000", "CANONICAL_FORWARD": "live",
        "CANONICAL_SHIM_LOG": str(tmp_path / "shim.log"), "CANONICAL_QUARANTINE_LOG": str(tmp_path / "q.jsonl"),
        "CANONICAL_CANARY_FILE": str(tmp_path / "canary.json"),
        "CANONICAL_PUBLISHED_DIR": str(tmp_path / "published"),
        "CANONICAL_MENTIONS_FILE": str(tmp_path / "required-mentions.json"),
        "CANONICAL_BUZZ_CLI": str(tmp_path / "buzz"),
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location("shim_owner_relay_under_test", str(_SHIM))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_unverified_relay_tells_the_cto_its_reason_and_never_to_ask_the_owner_again(shim):
    notice = shim.refusal_notice(409, "canonical_owner_relay_unverified", {"reason": "text_mismatch"},
                                 "event-1", 120)
    assert "사유: text_mismatch" in notice and "owner 에게 다시 승인하거나 다시 입력해 달라고 요청하지 마세요" in notice
    assert "(event event-1)" in notice
    # A reason outside the gateway's closed vocabulary is never echoed.
    assert "사유: malformed" in shim.refusal_notice(
        409, "canonical_owner_relay_unverified", {"reason": "<script>"}, "event-2", 1)
    assert shim.OWNER_RELAY_REASONS == __import__("gateway.owner_relay", fromlist=["REASONS"]).REASONS
