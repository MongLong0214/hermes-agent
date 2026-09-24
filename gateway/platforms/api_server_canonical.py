"""Authenticated canonical-surface ingress: ``POST /v1/canonical-surface/events``.

One request carries one existing-only canonical event and is answered with the durable terminal
``CanonicalReceiptCoordinator`` holds for it. The terminal goes only to this HTTP response: the
payload names no destination, so reply authority stays request-local.
"""

import logging
from typing import Any

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]


logger = logging.getLogger("gateway.platforms.api_server")

ROUTE = "/v1/canonical-surface/events"

# Refusals the coordinator raises before it claims the event, and only once the binding's own state.db
# holds no receipt for the event: nothing ran under its id, so a resend under a new id is safe.
_PRE_CLAIM_CODES = frozenset({
    "canonical_binding_stale",
    "canonical_agent_replaced",
    "canonical_turn_busy",
})


def _refusal(code: str, status: int, message: str = "Canonical request rejected.") -> "web.Response":
    return web.json_response({"error": {"code": code, "message": message}}, status=status)


def _http_routes(self) -> list[tuple[str, str, Any]]:
    async def handle(request: "web.Request") -> "web.Response":
        return await handle_canonical_surface_event(self, request)

    return [("POST", ROUTE, handle)]


def _refusal_for(code: str) -> "web.Response":
    if code == "canonical_principal_rejected":
        return _refusal(code, 403)
    if code == "canonical_receipt_conflict":
        return _refusal("canonical_event_conflict", 409)
    if code in _PRE_CLAIM_CODES:
        return _refusal(code, 409)
    # Anything else may come from past the claim, where the receipt is pending: it is never
    # re-executed, so it is reported as uncertainty rather than as a refusal inviting a resend.
    return _refusal("canonical_event_uncertain", 409)


async def handle_canonical_surface_event(self, request: "web.Request") -> "web.Response":
    """Serve one authenticated canonical event through the durable receipt coordinator."""
    auth_error = self._check_auth(request)
    if auth_error:
        return auth_error
    from gateway.canonical_surface import CanonicalIngressEvent, CanonicalReceiptCoordinator
    from gateway.platforms.api_server import _api_request_profile

    try:
        event = CanonicalIngressEvent.from_json_bytes(await request.read())
    except ValueError:
        return _refusal("canonical_invalid_request", 400)
    runner = self.gateway_runner
    if runner is None:
        return _refusal("canonical_unavailable", 503, "Canonical request unavailable.")
    # Bindings are the launch profile's config; a request scoped to another served profile
    # (/p/<name>/ under multiplexing, authenticated with that profile's key) must not drive them.
    profile = _api_request_profile.get()
    bindings = getattr(runner.config, "canonical_surface_bindings", None) or {}
    binding = bindings.get(event.binding) if profile in (None, "default") else None
    if binding is None:
        return _refusal("canonical_binding_unknown", 404)
    try:
        receipt = await CanonicalReceiptCoordinator(runner).submit(binding, event)
    except ValueError as exc:
        return _refusal_for(str(exc))
    except Exception:
        # The coordinator exposes no claim boundary, so an unexpected failure cannot prove the
        # turn did not run; report uncertainty, never detail, and never invite a re-execution.
        logger.exception("Canonical surface event failed; reporting uncertainty")
        return _refusal("canonical_event_uncertain", 409)
    if receipt.status != "terminal" or receipt.terminal_text is None:
        return _refusal("canonical_event_uncertain", 409)
    return web.json_response({"event_id": event.event_id, "text": receipt.terminal_text})
