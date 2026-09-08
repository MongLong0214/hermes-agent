"""Engine-owned default-deny policy for Gemini-family outbound routes."""

from __future__ import annotations

import re
from urllib.parse import urlparse


class GeminiOutboundDenied(RuntimeError):
    """Stable, metadata-safe rejection for Gemini-family outbound routes."""

    code = "gemini_outbound_denied"
    public_message = "Gemini outbound requests are disabled."

    def __init__(self) -> None:
        super().__init__(self.public_message)


_PROVIDER_ALIASES = frozenset(
    {
        "gemini",
        "google",
        "google-gemini",
        "google-ai-studio",
        "vertex",
        "vertexai",
        "google-vertex",
        "vertex-ai",
        "gcp-vertex",
    }
)

_INFERENCE_HOSTS = frozenset(
    {
        "generativelanguage.googleapis.com",
        "aiplatform.googleapis.com",
        "vertexai.googleapis.com",
    }
)

_API_MODE_HINTS = frozenset({"gemini", "gemini_native", "vertex", "vertex_ai"})


def _normalized(value: object) -> str:
    return str(value or "").strip().lower()


def _base_url_hostname(value: object) -> str:
    if not isinstance(value, str):
        return ""
    raw = value.strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw if "://" in raw else f"//{raw}")
        return (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _is_gemini_model(model: object) -> bool:
    value = _normalized(model)
    if "nano-banana" in value or "nanobanana" in value:
        return True
    return any(
        segment == "gemini" or segment.startswith(("gemini-", "gemini_"))
        for segment in re.split(r"[/:]", value)
    )


def _is_inference_host(host: object) -> bool:
    value = _normalized(host).rstrip(".")
    return (
        value in _INFERENCE_HOSTS
        or value.endswith(".aiplatform.googleapis.com")
        or value.endswith("-aiplatform.googleapis.com")
    )


def is_gemini_outbound(
    *,
    canonical_provider: object = "",
    model: object = "",
    base_url_host: object = "",
    base_url: object = "",
    api_mode: object = "",
    routing_hint: object = "",
) -> bool:
    """Classify a fully-described runtime route without reading mutable state."""
    provider = _normalized(canonical_provider)
    hint = _normalized(routing_hint)
    return (
        provider in _PROVIDER_ALIASES
        or hint in _PROVIDER_ALIASES
        or _is_gemini_model(model)
        or _is_inference_host(base_url_host)
        or _is_inference_host(_base_url_hostname(base_url))
        or _normalized(api_mode) in _API_MODE_HINTS
    )


def deny_gemini_outbound(**route: object) -> None:
    """Raise the stable public denial before credentials or clients are created."""
    if is_gemini_outbound(**route):
        raise GeminiOutboundDenied()
