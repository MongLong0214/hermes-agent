"""The outcome of a Codex Responses stream that lost transport after an accepted event.

``run_codex_stream`` returns ``_CodexStreamTerminalFailure`` instead of raising: the request is
billed, so nothing may replay it — not generic retry, not failover, not a watchdog reconnect.
These helpers keep the stream-end hook, the client pool and the stale breaker from treating it
as a completed call. No module-level ``agent`` imports: the request modules import this at load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class _CodexStreamTerminalFailure:
    """A Responses decoder accepted an event before transport was lost.

    Returning an object lets the invocation boundary distinguish a billed,
    non-replayable stream from an ordinary pre-stream transport failure.  The
    caller validates object identity before it suppresses generic recovery.
    """

    error: BaseException


def is_accepted_stream_failure(response: Any) -> bool:
    return isinstance(response, _CodexStreamTerminalFailure)


def stream_end_fields(response: Any, final_text: Callable[[Any], Any]) -> Dict[str, Any]:
    """``_emit_stream_end`` kwargs: the returned failure ends the stream unfinished, with its error."""
    if is_accepted_stream_failure(response):
        return {"final_text": "", "finished": False, "error": str(response.error)}
    return {"final_text": final_text(response), "finished": True, "error": None}
