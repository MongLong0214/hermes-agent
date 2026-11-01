"""Single resolver for every media source (data:/http(s)/file/local/container) -> bytes + mime.

Everything funnels through :func:`resolve_image_source` so size and magic-byte checks run
exactly once. Images are the default; video callers opt in via ``permitted=("video",)``.
Security (GHSA-gpxw-6wxv-w3qq): under a non-local backend vision is confined like the file
tools — host-read only inside a media cache (bind-mounted into the sandbox), anything else is
exec-read *inside the sandbox*, so ``vision_analyze('/etc/passwd')`` never reads the host.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import contextvars
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urljoin, urlsplit

# Raw-bytes INGEST budget: deliberately the 50MB download cap, NOT the 20MB provider
# payload cap — that one is enforced post-resize at the call sites.
_MAX_INGEST_BYTES = 50 * 1024 * 1024


class ImageResolutionError(Exception):
    def __init__(self, message: str, *, src: str = "", origin: str = ""):
        super().__init__(message)
        self.src, self.origin = src, origin


class UnsupportedScheme(ImageResolutionError): ...
class SourceUnsafe(ImageResolutionError): ...  # SSRF / path-allowlist
class SourceTooLarge(ImageResolutionError): ...
class SourceNotFound(ImageResolutionError): ...
class NotAnImage(ImageResolutionError): ...


@dataclass
class ResolveContext:
    task_id: Optional[str] = None


@dataclass
class ResolvedImage:
    data: bytes
    mime: str
    origin: str  # one of: data | http | file | local | container


# Explicit URL scheme ("ftp://", "s3://"). Bare Windows drive paths lack the "//".
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")

_DEFAULT_PORTS = {"http": 80, "https": 443}
_MAX_SCOPED_REDIRECTS = 5


def _url_parts(url: Any) -> Optional[tuple]:
    """``(scheme, host, port, path)`` of an http(s) URL; None for any other scheme, userinfo, a bad
    port, whitespace, a control character, a backslash or a ``.``/``..`` path segment (also encoded)."""
    if not isinstance(url, str) or any(c.isspace() or ord(c) < 0x20 or c in "\\\x7f" for c in url):
        return None
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not parts.hostname or "@" in parts.netloc:
        return None
    path = parts.path or "/"
    if any(segment in (".", "..") for segment in unquote(path).split("/")):
        return None
    return scheme, parts.hostname, port or _DEFAULT_PORTS[scheme], path


@dataclass(frozen=True)
class MediaScope:
    """What a scoped resolve (a canonical peer turn's ``vision_analyze``) may read: URLs whose
    scheme, host and port equal a base's and whose path sits under its path prefix, and files whose
    realpath sits under a root. Under :func:`scoped_media` anything else is refused before any read
    or fetch, and every redirect hop is checked against the same bases."""

    bases: tuple = ()  # (scheme, host, port, path prefix ending in "/")
    roots: tuple = ()  # realpath'd root Paths

    @classmethod
    def parse(cls, urls: Any, roots: Any) -> Optional["MediaScope"]:
        """The scope of a configured ``urls``/``roots`` pair; None when either is malformed."""
        urls, roots = urls or [], roots or []
        if not (isinstance(urls, list) and isinstance(roots, list) and all(isinstance(v, str) for v in urls + roots)):
            return None
        bases = []
        for url in urls:
            parts = _url_parts(url)
            if parts is None or urlsplit(url).query or urlsplit(url).fragment:
                return None
            bases.append(parts[:3] + (parts[3] if parts[3].endswith("/") else parts[3] + "/",))
        real_roots = []
        for root in roots:
            expanded = os.path.expanduser(root)
            if not os.path.isabs(expanded):
                return None
            real_roots.append(Path(os.path.realpath(os.path.normpath(expanded))))
        return cls(tuple(bases), tuple(real_roots))

    def admits_url(self, url: Any) -> bool:
        parts = _url_parts(url)
        return parts is not None and any(parts[:3] == base[:3] and parts[3].startswith(base[3]) for base in self.bases)

    def local_target(self, src: Any) -> Optional[Path]:
        """The realpath to read for a path source (``file://`` stripped, ``~`` expanded, normalised),
        or None when it is not absolute or does not sit under a root."""
        if not isinstance(src, str):
            return None
        s = src.strip()
        s = s[len("file://"):] if s.lower().startswith("file://") else s
        expanded = os.path.expanduser(s)
        if not os.path.isabs(expanded):
            return None
        real = Path(os.path.realpath(os.path.normpath(expanded)))
        return real if any(real.is_relative_to(root) for root in self.roots) else None

    def admits(self, src: Any) -> bool:
        s = src.strip() if isinstance(src, str) else src
        return self.admits_url(s) or self.local_target(s) is not None


_media_scope: contextvars.ContextVar[Optional[MediaScope]] = contextvars.ContextVar(
    "image_source_media_scope", default=None)


@contextlib.contextmanager
def scoped_media(scope: MediaScope):
    """Confine every :func:`resolve_image_source` in this context to ``scope``."""
    token = _media_scope.set(scope)
    try:
        yield
    finally:
        _media_scope.reset(token)


def _scoped_http_client(timeout: float):
    import httpx

    # Redirects are followed by hand in _fetch_scoped, each hop checked first.
    return httpx.AsyncClient(follow_redirects=False, timeout=timeout)


async def _fetch_scoped(url: str, scope: MediaScope) -> bytes:
    """GET ``url`` with redirects off: each ``Location`` is checked against ``scope`` before it is
    requested. The configured host is dialed even when it resolves to a private (tailnet) address."""
    from tools.vision_tools import _DOWNLOAD_USER_AGENT, _VISION_DOWNLOAD_TIMEOUT
    from tools.website_policy import check_website_access

    async with _scoped_http_client(_VISION_DOWNLOAD_TIMEOUT) as client:
        for _ in range(_MAX_SCOPED_REDIRECTS + 1):
            if not scope.admits_url(url):
                raise SourceUnsafe("blocked: outside the media base this turn may fetch", src=url, origin="http")
            if blocked := check_website_access(url):
                raise SourceUnsafe(blocked.get("message") or "blocked by website policy", src=url, origin="http")
            async with client.stream("GET", url, headers={"User-Agent": _DOWNLOAD_USER_AGENT,
                                                          "Accept": "image/*,*/*;q=0.8"}) as response:
                if response.is_redirect:
                    url = urljoin(str(response.url), response.headers["location"])
                    continue
                response.raise_for_status()
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > _MAX_INGEST_BYTES:
                        raise SourceTooLarge("media exceeds size limit", src=url, origin="http")
                return bytes(data)
    raise SourceUnsafe("blocked: too many redirects", src=url, origin="http")


async def _resolve_scoped(src: str, scope: MediaScope, permitted: tuple) -> ResolvedImage:
    """Only an allowed URL (fetched hop by hop) or a file under an allowed root (read at its
    realpath); anything else is refused before any read or fetch."""
    if scope.admits_url(src):
        return _finalize(await _fetch_scoped(src, scope), "", "http", src, permitted)
    target = scope.local_target(src)
    if target is None:
        raise SourceUnsafe("blocked: outside the media this turn may read", src=src)
    if not target.is_file():
        raise SourceNotFound(f"media file not found: '{target}'", src=src, origin="file")
    _guard_credential_read(target, src)
    return _finalize(await asyncio.to_thread(target.read_bytes), "", "file", src, permitted)


async def resolve_image_source(
    src: str, ctx: ResolveContext, *, permitted: tuple = ("image",)) -> ResolvedImage:
    if not isinstance(src, str) or not src.strip():
        raise SourceNotFound("image_url is required", src=str(src))
    s = src.strip()
    scope = _media_scope.get()
    if scope is not None:
        return await _resolve_scoped(s, scope, permitted)
    if s.startswith("data:"):
        data, mime = _resolve_data_url(s)
        return _finalize(data, mime, "data", s, permitted)
    if s.startswith(("http://", "https://")):
        reason = _http_block_reason(s)
        if reason:
            raise SourceUnsafe(reason, src=s)
        return _finalize(await _download_to_bytes(s), "", "http", s, permitted)
    if _SCHEME_RE.match(s) and not s.lower().startswith("file://"):
        raise UnsupportedScheme(
            "Unrecognized image source scheme. Use an http(s) URL, a local "
            "file path, a file:// URI, or a data: URL.",
            src=s)
    # Everything else is a filesystem path — including bare relative names like "pic.png"
    # (a path-shape gate here regressed them once).
    candidate = s[len("file://"):] if s.lower().startswith("file://") else s
    p = Path(os.path.expanduser(candidate))
    host_target = _permitted_host_read_target(p, ctx)
    if host_target is not None and host_target.is_file():
        _guard_credential_read(host_target, s)
        data = await asyncio.to_thread(host_target.read_bytes)
        return _finalize(data, "", "file", s, permitted)
    if _is_local_terminal_backend():
        # Any path was host-readable, so a miss means the file doesn't exist.
        raise SourceNotFound(f"media file not found: '{p}'", src=s, origin="file")
    return await _resolve_container_fallback(p, ctx, s, permitted)


def _guard_credential_read(host_target: Path, src: str) -> None:
    """Shared credential-read guard: refuse secret-bearing files (.env, auth.json) with a specific
    error. Guard import is best-effort; a real block always propagates."""
    try:
        # Shared credential-read guard (agent.file_safety, #57698): refuse secret-bearing files (.env,
        # auth.json, ...) with an intentional, specific error instead of relying on the magic-byte sniff to
        # reject them incidentally. Same chokepoint the image-gen/video-gen provider plugins enforce on
        # model-supplied local paths. Import is best-effort (guard unavailability must not break image
        # loading); a real block always propagates.
        from agent.file_safety import raise_if_read_blocked
    except Exception:  # noqa: BLE001 — guard unavailable: proceed
        return
    try:
        raise_if_read_blocked(str(host_target))
    except ValueError as exc:
        raise SourceUnsafe(str(exc), src=src, origin="file")


def _resolve_data_url(s: str) -> tuple[bytes, str]:
    header, _, payload = s.partition(",")
    if ";base64" not in header:
        raise NotAnImage("data: URL must be base64-encoded", src=s[:64])
    declared = header[len("data:"):].split(";", 1)[0].strip() or "application/octet-stream"
    # Cheap pre-decode size gate on the encoded length (~4/3 expansion).
    if (len(payload) * 3) // 4 > _MAX_INGEST_BYTES:
        raise SourceTooLarge("data: URL exceeds size limit", src=s[:64])
    try:
        data = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise NotAnImage(f"invalid base64 in data: URL: {exc}", src=s[:64])
    return data, declared  # real mime verified in _finalize via magic bytes


def _http_block_reason(url: str) -> Optional[str]:
    """Block reason, or None when allowed. Refuses policy-blocked URLs BEFORE any network I/O;
    ``_download_image`` re-checks per attempt and against the final redirect target (intentional)."""
    from tools.url_safety import is_safe_url
    from tools.website_policy import check_website_access
    if not is_safe_url(url):
        return "blocked: unsafe or private URL"
    if blocked := check_website_access(url):
        return blocked.get("message") or "blocked by website policy"
    return None


async def _download_to_bytes(url: str) -> bytes:
    import tempfile
    from tools.vision_tools import _download_image
    with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as tf:
        tmp = Path(tf.name)
    try:
        # Enforces the 50MB stream cap, redirect SSRF guard, and website policy.
        await _download_image(url, tmp)
        return await asyncio.to_thread(tmp.read_bytes)
    except PermissionError as exc:  # website policy block
        raise SourceUnsafe(str(exc), src=url, origin="http")
    finally:
        tmp.unlink(missing_ok=True)


def _is_local_terminal_backend() -> bool:
    """True when the terminal backend runs directly on the host (keys off ``TERMINAL_ENV``, read
    through the per-turn terminal scope so a routed multiplex profile sees ITS backend)."""
    from tools.terminal_scope import terminal_env
    return terminal_env("TERMINAL_ENV", "local").strip().lower() in ("local", "")


# Host-side media caches: the only host paths vision may read under a non-local backend
# (gateway inbound media + the tools' own download temp dirs).
_MEDIA_CACHE_SUBDIRS = (
    "cache",  # cache/images, cache/vision, cache/video(s), cache/audio
    "images",  # desktop/clipboard/PDF uploads (tui_gateway)
    "image_cache", "audio_cache", "video_cache", "temp_vision_images", "temp_video_files")


def _media_cache_roots() -> list:
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    return [home / sub for sub in _MEDIA_CACHE_SUBDIRS]


def _permitted_host_read_target(p: Path, ctx: ResolveContext) -> Optional[Path]:
    """Host path to read, or ``None`` (caller exec-reads inside the sandbox instead).

    Local backend: any path. Non-local: only paths inside a media cache root (a
    container-visible cache path is first translated back to its host mount).
    """
    if _is_local_terminal_backend():
        try:
            return p.resolve()
        except Exception:  # noqa: BLE001 — unresolved path: let is_file() fail downstream
            return p
    from tools.credential_files import from_agent_visible_cache_path
    try:
        real = Path(from_agent_visible_cache_path(str(p))).resolve()
    except Exception:  # noqa: BLE001 — cannot resolve -> not a safe host read
        return None
    if any(real.is_relative_to(root.resolve()) for root in _media_cache_roots()):
        return real
    return None


def _get_active_env(task_id: Optional[str]):
    if not task_id:
        return None
    try:
        from tools.terminal_tool_lifecycle import get_active_env
        return get_active_env(task_id)
    except Exception:
        return None


def _ensure_container_env(task_id: Optional[str]) -> None:
    """Lazily bring up the sandbox before an in-sandbox read (vision may be a session's first
    action). Best-effort: failure leaves the env absent and the caller hits the fail-closed error.

    Unlike the terminal tool, vision never triggered environment creation, so a session whose first action
    is ``vision_analyze`` on a container-only path under a non-local backend found no active env and failed
    — until a terminal command happened to create one (issue #62825).
    """
    if not task_id:
        return
    try:
        from tools.terminal_tool import ensure_task_env
        ensure_task_env(task_id)
    except Exception:
        pass


async def _resolve_container_fallback(
    p: Path, ctx: ResolveContext, src: str, permitted: tuple = ("image",)) -> ResolvedImage:
    """Read the bytes inside the sandbox; fail-closed when no env exists (a non-cache host
    path under a sandbox must never leak via a host fallback).

    Cold-start retry: under Docker the first exec against a fresh container can fail (empty
    pipe) while a second succeeds. On final failure the container's output is folded into the
    error so "no such file" / "permission denied" / "never came up" are distinguishable.

    We retry once with a short delay before giving up, so callers don't see "could not read inside the
    sandbox" on a file that is verifiably readable on the immediate retry. See #76566.
    """
    import shlex
    # Bring the sandbox up on demand: without this, the first vision_analyze of a session (before any
    # terminal command) has no active env to read from under a non-local backend (issue #62825).
    _ensure_container_env(ctx.task_id)
    env = _get_active_env(ctx.task_id)
    if env is None:
        raise SourceNotFound(
            f"'{p}' is not reachable inside the sandbox and no active sandbox "
            f"session is available to read it",
            src=src, origin="container")
    # Bound the read INSIDE the sandbox: head -c caps at ingest-limit+1 (+1 distinguishes "at the
    # cap" from "over") so /dev/zero can't stream unbounded base64 into host memory. The input
    # redirect avoids argv (leading-dash paths); tr -d instead of GNU-only base64 -w0 (BusyBox).
    cmd = f"head -c {_MAX_INGEST_BYTES + 1} < {shlex.quote(str(p))} | base64 | tr -d '\\n'"
    last_res: dict = {"returncode": 1, "output": ""}
    for attempt in range(2):
        last_res = await asyncio.to_thread(env.execute, cmd)
        if last_res.get("returncode", 1) == 0:
            break
        if attempt == 0:
            await asyncio.sleep(0.15)  # covers Docker exec warm-up in practice
    if last_res.get("returncode", 1) != 0:
        diag = (last_res.get("output") or "").strip().splitlines()
        first = next((ln.strip() for ln in diag if ln.strip()), "")
        suffix = f" ({first[:200]})" if first else ""
        raise SourceNotFound(f"could not read '{p}' inside the sandbox{suffix}", src=src, origin="container")
    try:
        data = base64.b64decode(last_res.get("output", ""), validate=True)
    except Exception as exc:
        raise NotAnImage(f"sandbox returned non-image data for '{p}': {exc}", src=src)
    if len(data) > _MAX_INGEST_BYTES:
        raise SourceTooLarge("media exceeds size limit", src=src, origin="container")
    return _finalize(data, "", "container", src, permitted)


def _finalize(
    data: bytes, declared_mime: str, origin: str, src: str, permitted: tuple = ("image",)
) -> ResolvedImage:
    """Chokepoint: 50MB ingest cap + type check. Images by magic bytes; video (opt-in) by
    extension + mp4 sniff — enough because every downstream consumer re-validates."""
    from tools.vision_tools_image_prep import _detect_image_mime_type_from_bytes
    if len(data) > _MAX_INGEST_BYTES:
        raise SourceTooLarge("media exceeds size limit", src=src, origin=origin)
    sniffed = _detect_image_mime_type_from_bytes(data)
    if sniffed is not None:
        if "image" not in permitted:
            raise NotAnImage("source is an image, but this argument takes a video", src=src, origin=origin)
        return ResolvedImage(data=data, mime=sniffed, origin=origin)
    if "image" in permitted and b"<svg" in data[:4096].lower():
        # Pass SVG through — call sites rasterize it to PNG before embedding.
        return ResolvedImage(data=data, mime="image/svg+xml", origin=origin)
    if "video" in permitted:
        video_mime = _detect_video_mime(data, src)
        if video_mime is not None:
            return ResolvedImage(data=data, mime=video_mime, origin=origin)
        raise NotAnImage("source is not a recognized video (mp4 expected)", src=src, origin=origin)
    raise NotAnImage("source is not a recognized image", src=src, origin=origin)


def _detect_video_mime(data: bytes, src: str) -> Optional[str]:
    """Video MIME from the extension table, else the ISO base-media ``ftyp`` magic at
    offset 4 (covers extensionless data: URLs / query-string URLs)."""
    from urllib.parse import urlsplit
    from tools.vision_tools import _detect_video_mime_type
    path_part = urlsplit(src).path if _SCHEME_RE.match(src) else src
    by_extension = _detect_video_mime_type(Path(path_part))
    if by_extension is not None:
        return by_extension
    if len(data) > 12 and data[4:8] == b"ftyp":
        return "video/mp4"
    return None


async def resolve_local_source_to_data_url(
    src: str, task_id: Optional[str], *, permitted: tuple = ("image",)) -> str:
    """Convert a path-like media source into a ``data:`` URL via the resolver.

    Dispatch-layer chokepoint for generation tools so providers never read model-supplied paths
    off the HOST under a sandbox. URL-shaped sources pass through; callers apply this only under
    a non-local backend.
    """
    s = (src or "").strip()
    if not s or s.lower().startswith(("http://", "https://", "data:")):
        return src
    resolved = await resolve_image_source(s, ResolveContext(task_id=task_id), permitted=permitted)
    encoded = base64.b64encode(resolved.data).decode("ascii")
    mime = resolved.mime or "application/octet-stream"
    return f"data:{mime};base64,{encoded}"
