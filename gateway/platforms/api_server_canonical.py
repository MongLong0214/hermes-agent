"""Authenticated canonical-surface ingress: ``POST /v1/canonical-surface/events``.

One request carries one existing-only canonical event and is answered with the durable terminal
``CanonicalReceiptCoordinator`` holds for it. The terminal goes only to this HTTP response: the
payload names no destination, so reply authority stays request-local.
"""

import ctypes
import logging
import os
import sqlite3
import sys
from contextlib import closing
from types import SimpleNamespace
from typing import Any

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]


logger = logging.getLogger("gateway.platforms.api_server")

ROUTE = "/v1/canonical-surface/events"
IDENTITY_ROUTE = "/v1/canonical-surface/identity"


class _ProcBSDInfo(ctypes.Structure):
    """Darwin sys/proc_info.h's proc_bsdinfo (including the native timeval)."""

    _fields_ = [
        (name, ctypes.c_uint32) for name in (
            "pbi_flags", "pbi_status", "pbi_xstatus", "pbi_pid", "pbi_ppid",
            "pbi_uid", "pbi_gid", "pbi_ruid", "pbi_rgid", "pbi_svuid", "pbi_svgid", "rfu_1",
        )
    ] + [
        ("pbi_comm", ctypes.c_char * 16), ("pbi_name", ctypes.c_char * 32),
    ] + [
        (name, ctypes.c_uint32) for name in (
            "pbi_nfiles", "pbi_pgid", "pbi_pjobc", "e_tdev", "e_tpgid", "pbi_nice",
        )
    ] + [("pbi_start_tvsec", ctypes.c_uint64), ("pbi_start_tvusec", ctypes.c_uint64)]


def _process_started_at(pid: int) -> str | None:
    """Use ACP's kernel process-start source; unreadable is never replaced with clock time."""
    if sys.platform == "darwin":
        try:
            libproc = ctypes.CDLL("libproc.dylib")
            proc_pidinfo = libproc.proc_pidinfo
            proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                     ctypes.c_void_p, ctypes.c_int]
            proc_pidinfo.restype = ctypes.c_int
            info = _ProcBSDInfo()
            size = ctypes.sizeof(info)
            # PROC_PIDTBSDINFO = 3 (sys/proc_info.h); require the full native record.
            if proc_pidinfo(pid, 3, 0, ctypes.byref(info), size) != size or info.pbi_pid != pid:
                return None
            if not info.pbi_start_tvsec or info.pbi_start_tvusec >= 1_000_000:
                return None
            return f"darwin-tv:{info.pbi_start_tvsec}.{info.pbi_start_tvusec:06d}"
        except (OSError, AttributeError):
            return None
    if sys.platform == "linux":
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as stat:
                raw = stat.read()
            fields = raw[raw.rindex(")") + 1:].split()
            starttime = fields[19]
            if starttime.isascii() and starttime.isdecimal():
                return f"linux-clk:{starttime}"
        except (OSError, ValueError, IndexError):
            pass
    return None

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

    async def identity(request: "web.Request") -> "web.Response":
        return await handle_canonical_surface_identity(self, request)

    return [("POST", ROUTE, handle), ("GET", IDENTITY_ROUTE, identity)]


async def handle_canonical_surface_identity(self, request: "web.Request") -> "web.Response":
    """Read the sole configured binding's existing live head without claiming an event."""
    from gateway.platforms.api_server import _api_request_profile

    if _api_request_profile.get() not in (None, "default"):
        return _refusal("canonical_binding_unknown", 404)
    if not self._expected_api_key():
        return self._auth_failed_response()
    auth_error = self._check_auth(request)
    if auth_error:
        return auth_error
    from gateway.canonical_surface import ExistingCanonicalBindingResolver
    from hermes_state_target_bind import (
        TargetBindReceiptFenceError, _lineage_root_digest, _resolve_lineage_root,
    )

    # The router matches the mirror, and the prefix middleware validates its profile.
    # Keep the query/path check exact: no extra parameters or alternate profile aliases.
    if request.path_qs != IDENTITY_ROUTE and not (
        request.path_qs == f"/p/default{IDENTITY_ROUTE}"
        and request.match_info.get("profile") == "default"
    ):
        return _refusal("canonical_invalid_request", 400)
    runner = self.gateway_runner
    if runner is None:
        return _refusal("canonical_unavailable", 503, "Canonical request unavailable.")
    bindings = getattr(runner.config, "canonical_surface_bindings", None) or {}
    if len(bindings) != 1:
        return _refusal("canonical_binding_unknown", 404)
    binding = next(iter(bindings.values()))
    resolver = ExistingCanonicalBindingResolver(runner.session_store)
    try:
        entry = resolver.resolve(binding, SimpleNamespace(
            author_id=binding.allowed_author_ids[0], channel_id=binding.allowed_channel_ids[0]))
        path = resolver._existing_db_path(binding.session_key)
        if path is None:
            raise ValueError("canonical_binding_stale")
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            root_id = _resolve_lineage_root(conn, entry.session_id)
    except (ValueError, IndexError, sqlite3.Error, TargetBindReceiptFenceError):
        return _refusal("canonical_binding_stale", 409)
    pid = os.getpid()
    started_at = _process_started_at(pid)
    if started_at is None:
        return _refusal("canonical_unavailable", 503, "Canonical request unavailable.")
    return web.json_response({
        "session_id": entry.session_id,
        "lineage_root_digest": _lineage_root_digest(root_id),
        "process_pid": pid,
        "process_started_at": started_at,
    })


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
