"""Regression contracts for #35809 under non-destructive exhaustion handling.

Compression exhaustion must block the request while retaining the same actor,
history, routing, and conversation scope; it no longer authorizes an automatic
reset or topic rebind. An explicit SessionStore.reset_session still yields an
empty next-turn transcript and preserves the old searchable history.
"""

from __future__ import annotations

import ast
import inspect

from gateway import run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# AST invariant: exhaustion preserves actor, history, binding, and scope
# ---------------------------------------------------------------------------
def _find_compression_exhausted_reset_block() -> ast.If:
    """Return the ``if agent_result.get('compression_exhausted') ...`` block."""
    tree = ast.parse(inspect.getsource(gateway_run))

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        consts = [
            n.value
            for n in ast.walk(node.test)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        ]
        # Identify the result-consumer branch by its flag and session guards.
        if "compression_exhausted" in consts:
            names = {sub.id for sub in ast.walk(node.test) if isinstance(sub, ast.Name)}
            if {"agent_result", "session_entry", "session_key"} <= names:
                return node
    raise AssertionError(
        "Could not locate the compression-exhausted result-consumer block "
        "in gateway/run.py — the structure changed or the AST walker is stale."
    )


class TestAutoResetBlockReSyncsBinding:
    def test_exhaustion_retains_actor_history_and_scope(self):
        """No destructive session operation is authorized by exhaustion."""
        block = _find_compression_exhausted_reset_block()
        references = {node.attr for node in ast.walk(block) if isinstance(node, ast.Attribute)}
        assert not references & {
            "reset_session", "switch_session", "_evict_cached_agent",
            "_clear_conversation_scope", "_sync_telegram_topic_binding",
            "replace_messages", "delete_session",
        }
        stores = {node.id for node in ast.walk(block)
                  if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        assert not stores & {"session_entry", "session_key", "source", "messages", "history"}

    def test_exhaustion_reports_blocked_same_session(self):
        """The response describes preservation, not a fresh-session promise."""
        block = _find_compression_exhausted_reset_block()
        response_assignments = [node for node in ast.walk(block)
                                if isinstance(node, ast.Assign)
                                and any(isinstance(target, ast.Name) and target.id == "response"
                                        for target in node.targets)]
        assert response_assignments
        text = " ".join(node.value for assignment in response_assignments
                        for node in ast.walk(assignment.value)
                        if isinstance(node, ast.Constant) and isinstance(node.value, str)).lower()
        assert "blocked" in text
        assert "history" in text and "routing" in text and "same session" in text
        assert "cooldown" in text and "/compress" in text
        assert "start a fresh session" not in text


# ---------------------------------------------------------------------------
# Behavioral contract: reset yields a clean next-turn transcript
# ---------------------------------------------------------------------------
def _make_store(tmp_path):
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    # Isolate the SQLite transcript store so we exercise per-session_id
    # transcripts without touching the developer's real state.db.
    store._db = SessionDB(db_path=tmp_path / "state.db")
    return store


def _make_source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1")


def _bloat(n):
    # Stand-in for the oversized, post-compression "child" transcript that
    # could not be compressed any further (#35809). Alternates roles so the
    # fixture is a valid conversation: load_transcript is a live-replay
    # restore site and heals alternation violations on load (#64934), so a
    # degenerate all-user transcript would be merged into one message.
    return [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": "x" * 2000,
        }
        for i in range(n)
    ]


class TestAutoResetLoadsCleanContext:
    """An explicit reset still starts empty without deleting old history."""

    def test_next_turn_transcript_is_empty_after_auto_reset(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()

        entry = store.get_or_create_session(source)
        session_key = entry.session_key
        bloated_sid = entry.session_id
        store._db.create_session(
            session_id=bloated_sid, source="telegram", user_id="u1"
        )
        store._db.replace_messages(bloated_sid, _bloat(120))
        assert len(store.load_transcript(bloated_sid)) == 120  # precondition

        new_entry = store.reset_session(session_key)
        assert new_entry is not None
        assert new_entry.session_id != bloated_sid

        resolved = store.get_or_create_session(source)
        assert resolved.session_id == new_entry.session_id
        loaded = store.load_transcript(resolved.session_id)

        assert loaded == [], (
            f"Auto-reset must yield an empty context, got {len(loaded)} "
            f"messages — the bloated compressed child leaked into the new session."
        )
        # The old transcript is still searchable, not destroyed.
        assert len(store.load_transcript(bloated_sid)) == 120

