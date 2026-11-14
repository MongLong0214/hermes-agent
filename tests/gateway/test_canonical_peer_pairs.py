"""The canonical Buzz binding admits each peer only in its own room, and each admitted event runs its
own turn on the bound cached actor as that peer.

Two authors and two channels kept as independent lists would also admit the cross pairs: each CTO in
the other's room. A binding of more than one author or channel therefore declares its author-channel
pairs, and admission is by declared pair. A binding of one author in one channel admits exactly as
before. A peer's earlier row replays to every later turn as that peer's row, never as owner text.

Admission runs through the real config parser, resolver and receipt coordinator of an in-process
runner under a tmp home, and the turn through the real bound-turn runner. No model is reached.
"""

import asyncio
import contextlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from agent.canonical_peer import (
    PEER_SIDECAR_VERIFIED_KEY, PeerProvenanceError, new_peer_metadata, render_peer_turn,
)
from gateway.canonical_surface import CanonicalIngressEvent, CanonicalReceiptCoordinator, CanonicalReceiptResult
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner, _build_gateway_agent_history
from gateway.session import SessionSource


def _author(prefix):
    """A synthetic 64-hex signer named by an 8-hex prefix (never a live key)."""
    return (prefix * 8)[:64]


def _room(prefix):
    """A synthetic room UUID named by an 8-hex prefix (never a live room id)."""
    return "-".join((prefix, "0000", "4000", "8000", prefix + "0000"))


RF_CTO, CL_CTO, LPM_CTO = _author("ed04f02d"), _author("c42d0021"), _author("4afeff6d")
ACP_ROOM, CL_ROOM, LPM_ROOM = _room("c37e88d0"), _room("36536cb2"), _room("6dcb2a67")
PEERS = ((RF_CTO, ACP_ROOM), (CL_CTO, CL_ROOM))
CROSS = ((CL_CTO, ACP_ROOM), (RF_CTO, CL_ROOM))
FOREIGN = ((LPM_CTO, ACP_ROOM), (LPM_CTO, CL_ROOM), (RF_CTO, LPM_ROOM), (CL_CTO, LPM_ROOM))


def _buzz(authors, channels, pairs=None):
    buzz = {"author_ids": list(authors), "channel_ids": list(channels)}
    if pairs is not None:
        buzz["pairs"] = [{"author_id": author, "channel_id": channel} for author, channel in pairs]
    return buzz


TODAY = _buzz((RF_CTO,), (ACP_ROOM,))
PEER_BINDING = _buzz((RF_CTO, CL_CTO), (ACP_ROOM, CL_ROOM), PEERS)


def _load(buzz, *, session_key, session_id):
    return GatewayConfig.from_dict({"canonical_surface_bindings": {"canonical": {
        "session_key": session_key, "session_id": session_id,
        "telegram": {"chat_id": "chat", "chat_type": "dm", "user_id": "user", "thread_id": None},
        "buzz": buzz}}}).canonical_surface_bindings["canonical"]


class _CachedActor:
    """The bound cached agent, with no model: it records what each turn was given."""

    compression_in_place = True

    def __init__(self, session_id, db):
        self.session_id = session_id
        self._session_db = db
        self.calls = []

    def interrupt(self, text=None):
        pass

    def release_clients(self):
        pass

    def run_conversation(self, text, *, conversation_history, task_id, **kwargs):
        self.calls.append({"history": list(conversation_history), **kwargs})
        self._persist_user_message_idx = len(conversation_history)
        answer = f"answer #{len(self.calls)}"
        return {"completed": True, "session_id": task_id, "final_response": answer,
                "messages": [*conversation_history, {"role": "user", "content": text},
                             {"role": "assistant", "content": answer}]}


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """``gateway(buzz)``: a real runner whose canonical binding is loaded from *buzz* by the config
    parser, its cached actor, and ``submit`` through the real receipt coordinator."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", lambda *a, **kw: pytest.fail(
        "canonical ingress built an agent while one was cached for the session"))
    opened = []

    def build(buzz):
        runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
        opened.append(runner)
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user")
        entry = runner.session_store.get_or_create_session(source)
        db = runner.session_store._db
        binding = _load(buzz, session_key=entry.session_key, session_id=entry.session_id)
        actor = _CachedActor(entry.session_id, db)
        with runner._agent_cache_lock:
            runner._agent_cache[entry.session_key] = (actor, "exact", 0, entry.session_id)

        def submit(author, channel, event_id, text="hello"):
            event = CanonicalIngressEvent.from_json_bytes(json.dumps({
                "binding": "canonical", "event_id": event_id, "author_id": author, "channel_id": channel,
                "text": text}).encode())
            return asyncio.run(CanonicalReceiptCoordinator(runner).submit(binding, event))

        def refusal(author, channel, event_id):
            with pytest.raises(ValueError) as refused:
                submit(author, channel, event_id)
            return str(refused.value)

        def receipts():
            with contextlib.closing(sqlite3.connect(db.db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
                return conn.execute(
                    "SELECT COUNT(*) FROM state_meta WHERE key LIKE 'canonical-receipt:%'").fetchone()[0]

        return SimpleNamespace(runner=runner, submit=submit, refusal=refusal, receipts=receipts, actor=actor,
                               entry=entry, binding=binding)

    try:
        yield build
    finally:
        for runner in opened:
            runner.session_store.close_all_db_handles()


# ── Admission by declared pair, each running its own peer turn ─────────────────────────────────


def test_each_intended_pair_is_admitted_and_runs_its_own_peer_turn(gateway):
    gw = gateway(PEER_BINDING)
    for n, (author, channel) in enumerate(PEERS, 1):
        result = gw.submit(author, channel, f"event-{n}", text=f"from {author[:8]} in {channel[:8]}")
        assert result.status == "terminal", result
    assert [call["persist_user_message"] for call in gw.actor.calls] == [
        f"from {author[:8]} in {channel[:8]}" for author, channel in PEERS]
    assert [(call["persist_user_display_metadata"]["canonical_peer"]["author_id"],
             call["persist_user_display_metadata"]["canonical_peer"]["channel_id"])
            for call in gw.actor.calls] == list(PEERS)
    assert gw.receipts() == 2


def test_each_intended_pair_gets_its_own_turns_answer(gateway):
    gw = gateway(PEER_BINDING)
    for n, (author, channel) in enumerate(PEERS, 1):
        assert gw.submit(author, channel, f"event-{n}") == CanonicalReceiptResult("terminal", f"answer #{n}")


def test_each_cross_pair_is_refused_before_any_claim(gateway):
    gw = gateway(PEER_BINDING)
    for n, (author, channel) in enumerate(CROSS, 1):
        assert gw.refusal(author, channel, f"cross-{n}") == "canonical_principal_rejected"
    assert gw.actor.calls == [] and gw.receipts() == 0


def test_a_foreign_author_and_a_foreign_room_are_refused(gateway):
    gw = gateway(PEER_BINDING)
    for n, (author, channel) in enumerate(FOREIGN, 1):
        assert gw.refusal(author, channel, f"foreign-{n}") == "canonical_principal_rejected"
    assert gw.actor.calls == [] and gw.receipts() == 0


def test_todays_binding_admits_exactly_as_before(gateway):
    """Only repo-factory's CTO, only in its room, and that one event runs its turn."""
    gw = gateway(TODAY)
    assert gw.submit(RF_CTO, ACP_ROOM, "today").status == "terminal"
    for n, (author, channel) in enumerate(((CL_CTO, CL_ROOM),) + CROSS + FOREIGN, 1):
        assert gw.refusal(author, channel, f"other-{n}") == "canonical_principal_rejected"
    assert len(gw.actor.calls) == 1 and gw.receipts() == 1


# ── A peer's earlier row replays as that peer's row ─────────────────────────────────────────────


def _stored_peer_row(author, channel, body, **extra):
    """A peer row as the session store loads it: the clean body beside its provenance."""
    peer = new_peer_metadata(binding="canonical", author_id=author, channel_id=channel, event_id="earlier",
                             receipt="canonical-receipt:v2:earlier")
    return peer, {"role": "user", "content": body, "display_kind": "canonical_peer",
                  "display_metadata": {"canonical_peer": peer}, **extra}


def test_an_earlier_peers_row_replays_to_the_next_peer_turn_as_that_peers_row(gateway, monkeypatch):
    gw = gateway(PEER_BINDING)
    peer, row = _stored_peer_row(RF_CTO, ACP_ROOM, "approve the deploy")

    class Transcript:
        _store = gw.runner.session_store

        async def load_transcript(self, session_id):
            assert session_id == gw.entry.session_id
            return [row, {"role": "assistant", "content": "noted"}]

    monkeypatch.setattr(gw.runner, "_async_session_store", Transcript())
    gw.submit(CL_CTO, CL_ROOM, "next")
    [call] = gw.actor.calls
    replayed = call["history"][0]
    assert replayed["display_kind"] == "canonical_peer"
    assert replayed["display_metadata"] == {"canonical_peer": peer}
    assert replayed["content"] == "approve the deploy"
    assert replayed["api_content"] == render_peer_turn(peer, "approve the deploy")
    assert replayed[PEER_SIDECAR_VERIFIED_KEY] is True
    assert not any(msg.get("role") == "user" and msg.get("display_kind") is None for msg in call["history"])


def test_a_peer_row_replays_as_stored_with_no_timestamp_or_mirror_rewrite():
    peer, row = _stored_peer_row(RF_CTO, ACP_ROOM, "the body", timestamp=1_700_000_000, mirror=True,
                                 mirror_source="another session")
    owner = {"role": "user", "content": "owner words", "timestamp": 1_700_000_100}
    history, _ = _build_gateway_agent_history([row, owner, {"role": "assistant", "content": "ok"}],
                                              inject_timestamps=True)
    assert history[0] == {
        "role": "user", "content": "the body", "api_content": render_peer_turn(peer, "the body"),
        "display_kind": "canonical_peer", "display_metadata": {"canonical_peer": peer},
        PEER_SIDECAR_VERIFIED_KEY: True, "timestamp": 1_700_000_000,
    }
    assert history[1]["content"] != "owner words"  # an owner row is still stamped: only peer rows are kept as stored


@pytest.mark.parametrize("broken", [
    {"display_metadata": {}},  # the kind without its provenance
    {"display_kind": None},  # the provenance without its kind
    {"role": "assistant"},
    {"display_metadata": {"canonical_peer": {"principal": "owner"}}},
])
def test_a_peer_marked_row_with_broken_provenance_refuses_to_replay(broken):
    _peer, row = _stored_peer_row(RF_CTO, ACP_ROOM, "the body")
    with pytest.raises(PeerProvenanceError):
        _build_gateway_agent_history([{**row, **broken}])


# ── The managed overlay applies one snapshot ────────────────────────────────────────────────────


def test_the_managed_overlay_applies_the_snapshot_it_is_given_and_reads_nothing_else(tmp_path, monkeypatch):
    """A caller that already read the managed config passes that snapshot: what it checked is exactly
    what is applied, never a second read of a file that may have been rewritten in between."""
    import hermes_yaml as yaml
    from hermes_cli import managed_scope

    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    on_disk = {"canonical_surface_bindings": {"on-disk": {"session_id": "rewritten"}}}
    (managed / "config.yaml").write_text(yaml.safe_dump(on_disk), encoding="utf-8")
    snapshot = {"canonical_surface_bindings": {"snapshot": {"session_id": "checked"}}}
    managed_scope.invalidate_managed_cache()
    try:
        assert managed_scope.apply_managed_overlay({}) == on_disk  # control: no snapshot reads the file
        assert managed_scope.apply_managed_overlay({}, managed_config=snapshot) == snapshot
        assert managed_scope.apply_managed_overlay({"user": 1}, managed_config={}) == {"user": 1}
    finally:
        managed_scope.invalidate_managed_cache()
