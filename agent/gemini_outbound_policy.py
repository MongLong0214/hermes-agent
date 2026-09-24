"""Default-deny policy for Gemini / Google AI Studio / Vertex outbound routes.

Every boundary that could reach a Google-hosted inference or speech endpoint (runtime
resolution, auxiliary clients, client construction, ``/model`` switches, primary restore,
trajectory compression, MiniSWE, the native Gemini / Vertex adapters, TTS) calls
:func:`deny_gemini_outbound` with the route's non-secret facts BEFORE any credential read,
cache lookup or client construction. There is no opt-out.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse


class GeminiOutboundDenied(RuntimeError):
    """Stable, metadata-free rejection: it carries no route, key or model detail."""

    code = "gemini_outbound_denied"
    public_message = "Gemini outbound requests are disabled."

    def __init__(self) -> None:
        super().__init__(self.public_message)


_PROVIDER_ALIASES = frozenset({
    "gemini", "google", "google-gemini", "google-ai-studio",
    "vertex", "vertexai", "google-vertex", "vertex-ai", "gcp-vertex",
})
_INFERENCE_HOSTS = frozenset({
    "generativelanguage.googleapis.com", "aiplatform.googleapis.com", "vertexai.googleapis.com",
})
_API_MODE_HINTS = frozenset({"gemini", "gemini_native", "vertex", "vertex_ai"})


def _normalized(value: object) -> str:
    return str(value or "").strip().lower()


def _base_url_hostname(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    raw = value.strip()
    try:
        parsed = urlparse(raw if "://" in raw else f"//{raw}")
        return (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _is_gemini_model(model: object) -> bool:
    value = _normalized(model)
    if "nano-banana" in value or "nanobanana" in value:
        return True
    return any(segment == "gemini" or segment.startswith(("gemini-", "gemini_"))
               for segment in re.split(r"[/:]", value))


def _is_inference_host(host: str) -> bool:
    return (host in _INFERENCE_HOSTS or host.endswith(".aiplatform.googleapis.com")
            or host.endswith("-aiplatform.googleapis.com"))


def is_gemini_outbound(
    *, canonical_provider: object = "", model: object = "", base_url: object = "",
    api_mode: object = "", routing_hint: object = "", endpoint_authority: bool = False,
) -> bool:
    """Classify a route from its non-secret facts; reads no mutable state.

    Default: any Google signal denies (provider or routing-hint alias, a Gemini-family model,
    a Google inference host, a Gemini/Vertex api_mode). ``endpoint_authority=True`` is for
    boundaries that hold the concrete selected endpoint (client construction, a live agent's
    model switch, auxiliary clients): a present ``base_url`` alone decides, so ``google/gemini-*`` served by
    OpenRouter or another non-Google endpoint stays allowed; without one the provider, hint and
    api_mode decide; with none of those the default classification applies."""
    if endpoint_authority:
        if isinstance(base_url, str) and base_url.strip():
            return _is_inference_host(_base_url_hostname(base_url))
        if _normalized(canonical_provider) or _normalized(routing_hint):
            model = ""
    return (
        _normalized(canonical_provider) in _PROVIDER_ALIASES
        or _normalized(routing_hint) in _PROVIDER_ALIASES
        or _is_gemini_model(model)
        or _is_inference_host(_base_url_hostname(base_url))
        or _normalized(api_mode) in _API_MODE_HINTS
    )


def deny_gemini_outbound(**route: object) -> None:
    """Raise :class:`GeminiOutboundDenied` when *route* (see :func:`is_gemini_outbound`) is Google-bound."""
    if is_gemini_outbound(**route):
        raise GeminiOutboundDenied()
