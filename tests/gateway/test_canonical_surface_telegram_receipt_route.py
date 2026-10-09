"""U4 H1 — ``GET /v1/canonical-surface/receipts/telegram/{update_id}``.

Read-only: never opens a turn, a lease, or a write handle. Resolves the sole Telegram-origin
canonical binding and answers from whatever ``gateway.acp_turn_receipts`` finds in its
``state_meta``, written directly here since the H2 seam that would write it live is not wired yet.
"""

from __future__ import annotations

import asyncio
import secrets
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import acp_turn_receipts as receipts
from gateway.canonical_surface import CanonicalSurfaceBinding
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
from gateway.run import GatewayRunner
from gateway.session import SessionSource


_ROUTE_METHOD = "GET"


@pytest.fixture
def ingress(tmp_path, monkeypatch):
    home, user_home = tmp_path / "hermes-home", tmp_path / "user"
    home.mkdir()
    user_home.mkdir()
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setattr(Path, "home", lambda: user_home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user")
    entry = runner.session_store.get_or_create_session(source)
    db = runner.session_store._db
    binding = CanonicalSurfaceBinding(
        name="canonical", session_key=entry.session_key, session_id=entry.session_id,
        telegram_chat_id="chat", telegram_chat_type="dm", telegram_user_id="user",
        telegram_thread_id=None, allowed_author_ids=("author",), allowed_channel_ids=("channel",),
    )
    runner.config.canonical_surface_bindings = {binding.name: binding}
    key = secrets.token_hex(32)

    def proof():
        return {"proven_db_path": db.db_path, "proven_db_identity": db._db_file_identity}

    yield SimpleNamespace(runner=runner, key=key, db=db, proof=proof, binding=binding)
    runner.session_store.close_all_db_handles()


@pytest.fixture
def get_receipt(ingress):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": ingress.key}))
    adapter.gateway_runner = ingress.runner
    route = "/v1/canonical-surface/receipts/telegram/{update_id}"
    [handler] = [h for method, path, h in adapter._http_route_table() if (method, path) == (_ROUTE_METHOD, route)]

    async def get(update_id, *, bearer=ingress.key, profile=None):
        async def read():
            return b""

        headers = {"Authorization": f"Bearer {bearer}"} if bearer is not None else {}
        request = SimpleNamespace(
            headers=headers, read=read, method=_ROUTE_METHOD,
            path_qs=f"/v1/canonical-surface/receipts/telegram/{update_id}", transport=None,
            match_info={"update_id": str(update_id)},
        )
        token = _api_request_profile.set(profile)
        try:
            response = await handler(request)
        finally:
            _api_request_profile.reset(token)
        import json
        return response.status, json.loads(response.text)

    get.adapter = adapter
    yield get
    adapter._response_store.close()


def test_an_unclaimed_update_answers_never_found(get_receipt):
    async def exercise():
        status, body = await get_receipt("999")
        assert status == 200
        assert body["status"] == "NEVER_FOUND"
        assert body["schema"] == "hermes.gateway-turn-receipt/v1"
    asyncio.run(exercise())


def test_a_pending_claim_answers_pending_with_its_identity(get_receipt, ingress):
    async def exercise():
        receipts.claim_pending(
            ingress.db, "42", message_id="m1", turn_request_id="t1",
            receipt_identity={"targetActorId": "a1"}, **ingress.proof(),
        )
        status, body = await get_receipt("42")
        assert status == 200
        assert body["status"] == "PENDING"
        assert body["turnRequestId"] == "t1"
        assert body["receiptIdentity"] == {"targetActorId": "a1"}
    asyncio.run(exercise())


def test_a_completed_receipt_answers_with_its_digest_and_content(get_receipt, ingress):
    async def exercise():
        receipts.claim_pending(
            ingress.db, "42", message_id="m1", turn_request_id="t1", receipt_identity={}, **ingress.proof(),
        )
        receipts.settle_completed(
            ingress.db, "42", receipt_id="hermes-tg:ob1", evidence_digest="sha256:abc",
            content="final text", delivery={"chat_id": "chat", "state": "delivered"}, **ingress.proof(),
        )
        status, body = await get_receipt("42")
        assert status == 200
        assert body["status"] == "COMPLETED"
        assert body["receiptId"] == "hermes-tg:ob1"
        assert body["evidenceDigest"] == "sha256:abc"
        assert body["content"] == "final text"
        assert body["delivery"] == {"chat_id": "chat", "state": "delivered"}
    asyncio.run(exercise())


_TURN = {
    "turnRequestId": "t1", "targetActorId": "actor-1", "promptDigest": "sha256:" + "ab" * 32,
    "bindingGeneration": 4, "targetBindingId": "bind-1", "targetAttestationId": "att-1",
    "executorSessionId": "ses-1", "executorSessionIncarnation": "inc-1",
}


def test_an_aborted_receipt_answers_aborted_with_its_reason(get_receipt, ingress):
    # An ABORTED receipt is served only with the evidence ACP compares preserved whole (the eight
    # identity fields, a numeric message id) and a ledger that can show no answer was delivered.
    from gateway import delivery_ledger

    delivery_ledger.sweep_recoverable(deliverable_platforms=set())  # the startup sweep opens the ledger

    async def exercise():
        receipts.claim_pending(
            ingress.db, "42", message_id="55", turn_request_id="t1", receipt_identity=dict(_TURN),
            **ingress.proof(),
        )
        receipts.settle_aborted(ingress.db, "42", reason_code="HERMES_PROCESS_DIED_BEFORE_ANSWER", **ingress.proof())
        status, body = await get_receipt("42")
        assert status == 200
        assert body["status"] == "ABORTED" and body["receiptId"] == "hermes-tg:aborted:42"
        assert body["reasonCode"] == "HERMES_PROCESS_DIED_BEFORE_ANSWER"
    asyncio.run(exercise())


def test_an_aborted_receipt_without_its_identity_is_refused_not_guessed(get_receipt, ingress):
    async def exercise():
        receipts.claim_pending(
            ingress.db, "42", message_id="m1", turn_request_id="t1", receipt_identity={}, **ingress.proof(),
        )
        receipts.settle_aborted(ingress.db, "42", reason_code="HERMES_PROCESS_DIED_BEFORE_ANSWER", **ingress.proof())
        status, body = await get_receipt("42")
        assert status == 409 and body["error"]["code"] == "canonical_receipt_unprovable"
    asyncio.run(exercise())


def test_missing_bearer_is_refused_before_any_lookup(get_receipt):
    async def exercise():
        status, _ = await get_receipt("42", bearer=None)
        assert status == 401
    asyncio.run(exercise())


def test_wrong_bearer_is_refused(get_receipt):
    async def exercise():
        status, _ = await get_receipt("42", bearer="not-the-key")
        assert status == 401
    asyncio.run(exercise())


def test_an_unresolvable_store_answers_uncertain_not_never_found(get_receipt, ingress, monkeypatch):
    """Review R1: the resolver returns None for "no store yet" AND for a transient filesystem
    error on a binding that WAS previously claimed. Collapsing both to NEVER_FOUND would tell a
    caller it is safe to resend an update whose outcome this process simply cannot read right
    now. Demonstrated with a receipt already claimed on the real store, then made unresolvable."""
    async def exercise():
        from gateway.canonical_surface import ExistingCanonicalBindingResolver

        receipts.claim_pending(
            ingress.db, "42", message_id="m1", turn_request_id="t1", receipt_identity={}, **ingress.proof(),
        )
        monkeypatch.setattr(ExistingCanonicalBindingResolver, "_existing_db_path", lambda self, key: None)
        status, body = await get_receipt("42")
        assert status == 409
        assert body["error"]["code"] == "canonical_event_uncertain"
    asyncio.run(exercise())


def test_zero_telegram_bindings_answers_binding_unknown(get_receipt, ingress):
    async def exercise():
        ingress.runner.config.canonical_surface_bindings = {}
        status, body = await get_receipt("42")
        assert status == 404
        assert body["error"]["code"] == "canonical_binding_unknown"
    asyncio.run(exercise())


def test_two_telegram_bindings_is_ambiguous_and_answers_binding_unknown(get_receipt, ingress):
    async def exercise():
        second = CanonicalSurfaceBinding(
            name="second", session_key="agent:main:telegram:dm:other", session_id="other-session",
            telegram_chat_id="other-chat", telegram_chat_type="dm", telegram_user_id="other-user",
            telegram_thread_id=None, allowed_author_ids=("a",), allowed_channel_ids=("c",),
        )
        ingress.runner.config.canonical_surface_bindings = {
            ingress.binding.name: ingress.binding, second.name: second,
        }
        status, body = await get_receipt("42")
        assert status == 404
        assert body["error"]["code"] == "canonical_binding_unknown"
    asyncio.run(exercise())


def test_a_scoped_profile_request_never_reaches_the_default_profiles_binding(get_receipt, monkeypatch):
    """Mirrors the POST route's profile isolation: a request scoped to another served profile,
    authenticated with ITS OWN key, must not resolve or read the launch profile's binding."""
    async def exercise():
        other_key = secrets.token_hex(32)
        monkeypatch.setenv("API_SERVER_KEY", other_key)
        status, body = await get_receipt("42", bearer=other_key, profile="other-profile")
        assert status == 404
        assert body["error"]["code"] == "canonical_binding_unknown"
    asyncio.run(exercise())
