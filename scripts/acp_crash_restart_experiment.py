#!/usr/bin/env python3
"""U4 crash/restart experiment for the /acp managed ingress, with real process death.

An isolated gateway (temporary HERMES_HOME; no live gateway, Telegram, ACP or Keychain) admits an
``/acp`` task through a stand-in ACP lane and starts the turn through the real
``GatewayRunner._handle_message`` (session marker, marker binding and turn lease included). The
child process is SIGKILLed at a chosen point. A second process then opens the same home and runs
the startup recovery in production order: ``_recover_unclean_sessions``, the ACP receipt sweep and
``_redeliver_pending_obligations``, with a recording adapter in place of Telegram.

Scenarios:
  managed-before-send   killed after the managed answer is in the transcript, before its final send
  managed-failed-send   killed after the managed answer's send was rejected (ledger row ``failed``)
  chain-followup        killed while an ordinary follow-up runs inside the managed chain, after its
                        answer is in the transcript (the managed answer itself was delivered)
  chain-withheld-before-followup
                        the managed answer's obligation could not be recorded (withheld, receipt
                        ABORTED); killed as the follow-up starts, before its prompt is persisted
  chain-followup-unanswered
                        killed after the follow-up's prompt is persisted, before it has an answer
  pre-prompt-after-bind the session already holds a completed, delivered ordinary turn; killed after
                        the marker is bound, before the admitted prompt is persisted

Usage: acp_crash_restart_experiment.py [scenario ...]   (all scenarios when none is given)
Exit status 0 only when every scenario meets its expectation.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCENARIOS = ("managed-before-send", "managed-failed-send", "chain-followup", "chain-withheld-before-followup",
             "chain-followup-unanswered", "pre-prompt-after-bind")
UPDATE_ID = 901
CHAT = "100200300"


def _bootstrap(home: Path):
    os.environ["HERMES_HOME"] = str(home / "hermes-home")
    os.environ["HOME"] = str(home / "user")
    Path(os.environ["HERMES_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HOME"]).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO))
    import hermes_state

    hermes_state.DEFAULT_DB_PATH = Path(os.environ["HERMES_HOME"]) / "state.db"
    from gateway.canonical_surface import CanonicalSurfaceBinding
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    runner = GatewayRunner(GatewayConfig(sessions_dir=Path(os.environ["HERMES_HOME"]) / "sessions"))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=CHAT, chat_type="dm", user_id=CHAT)
    entry = runner.session_store.get_or_create_session(source)
    binding = CanonicalSurfaceBinding(
        name="acp-canonical-ceo", session_key=entry.session_key, session_id=entry.session_id,
        telegram_chat_id=CHAT, telegram_chat_type="dm", telegram_user_id=CHAT, telegram_thread_id=None,
        allowed_author_ids=("a",), allowed_channel_ids=("c",))
    runner.config.canonical_surface_bindings = {binding.name: binding}
    return runner, source, entry


def _recording_adapter(runner, log: Path, *, succeed: bool):
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    class Recording(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)

        async def connect(self):
            return True

        async def disconnect(self):
            pass

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            with open(log, "a") as handle:
                handle.write(json.dumps({"content": content, "reply_to": reply_to, "ok": succeed}) + "\n")
            return SendResult(success=succeed, message_id="777" if succeed else None,
                              error=None if succeed else "flood")

        async def get_chat_info(self, chat_id):
            return {"id": chat_id}

    adapter = Recording()
    adapter.gateway_runner = runner
    return adapter


def child(scenario: str, home: Path) -> None:
    import asyncio

    runner, source, entry = _bootstrap(home)
    from gateway import acp_managed_ingress as ingress
    from gateway.platforms.event import MessageEvent
    from gateway.turn_context import TurnContext

    ready = home / "ready"
    sends = home / "child-sends.jsonl"
    db = runner.session_store._db_for_key(entry.session_key)
    from hermes_state_target_bind import _lineage_root_digest

    lineage = _lineage_root_digest(db._session_lineage_root_to_tip(entry.session_id)[0])
    sock = Path(tempfile.mkdtemp(dir="/tmp", prefix="acpx-")) / "lane.sock"

    async def lane(reader, writer):
        envelope = json.loads(await reader.readline())
        writer.write(json.dumps({
            "allowed": True, "replayed": False,
            "turn": {"turnRequestId": "turn-1", "targetActorId": "actor-1", "promptDigest": "sha256:p",
                     "bindingGeneration": 4, "targetBindingId": "bind-1", "targetAttestationId": "att-1",
                     "executorSessionId": "ses-1", "executorSessionIncarnation": "inc-1"},
            "source": {"channel": "telegram", "nonce": f"update:{envelope['update']['update_id']}"},
            "targetBind": {"requested_session_id": entry.session_id, "lineage_root_digest": lineage},
        }).encode() + b"\n")
        await writer.drain()
        writer.close()

    ingress.socket_path = lambda: sock
    ingress.read_secret = lambda: "lane-secret"
    adapter = _recording_adapter(runner, sends, succeed=scenario != "managed-failed-send")
    runner.adapters[source.platform] = adapter

    async def admitted(event):
        return event, event.source, False

    runner._hm_admit_event = admitted

    def persist(role, content, metadata=None):
        row = {"role": role, "content": content}
        if metadata:
            row["display_metadata"] = metadata
        runner.session_store.append_to_transcript(entry.session_id, row)

    if scenario == "pre-prompt-after-bind":  # an earlier, completed and delivered ordinary turn
        from gateway import delivery_ledger

        persist("user", "earlier question")
        persist("assistant", "earlier answer")
        delivery_ledger.record_obligation(obligation_id="earlier", session_key=entry.session_key,
                                          platform="telegram", chat_id=CHAT, thread_id=None,
                                          content="earlier answer")
        delivery_ledger.mark_delivered("earlier", message_ids=["1"])

    async def park():
        ready.write_text(scenario)
        await asyncio.sleep(3600)  # SIGKILL lands here

    async def run_agent(*args, **kwargs):
        if scenario == "pre-prompt-after-bind" and not kwargs.get("_interrupt_depth", 0):
            await park()  # marker bound, admitted prompt not yet persisted
        if not kwargs.get("_interrupt_depth", 0):  # the agent persists the admitted prompt first
            persist("user", kwargs.get("persist_user_message") or kwargs.get("message") or "/acp deploy the fix",
                    kwargs.get("persist_user_display_metadata"))
        if kwargs.get("_interrupt_depth", 0):  # the ordinary follow-up inside the chain
            if scenario == "chain-withheld-before-followup":
                await park()  # before the follow-up's prompt is durable
            persist("user", kwargs["message"])
            if scenario == "chain-followup-unanswered":
                await park()
            persist("assistant", "ordinary follow-up answer")
            await park()
        if scenario == "managed-before-send":
            persist("assistant", "managed answer")
            await park()
        if scenario.startswith("chain-"):
            event = runner._experiment_event
            ctx = TurnContext(source=source, session_key=entry.session_key, session_id=entry.session_id,
                              history=[], acp_admission=event._acp_admission,
                              inbound_message_id="55", event_message_id="55")

            async def first_response(*a):
                persist("assistant", "managed answer")
                if scenario == "chain-withheld-before-followup":
                    from gateway import delivery_ledger

                    real_record = delivery_ledger.record_obligation

                    def failing_record(**k):
                        raise OSError("disk I/O error")

                    delivery_ledger.record_obligation = failing_record
                try:
                    await runner._send_queued_final_text(adapter, source, "managed answer", None, "55",
                                                         entry.session_key, "55",
                                                         acp_admission=event._acp_admission)
                finally:
                    if scenario == "chain-withheld-before-followup":
                        delivery_ledger.record_obligation = real_record

            runner._run_agent_deliver_first_response = first_response

            async def no_refresh(*a, **k):
                return None

            runner._refresh_agent_cache_message_count = no_refresh
            return await runner._run_agent_queued_followup(
                ctx, adapter, "ordinary leftover steer", None, {"final_response": "managed answer"},
                {"messages": []}, None)
        persist("assistant", "managed answer")
        return {"final_response": "managed answer", "messages": [], "api_calls": 1}

    runner._run_agent = run_agent

    async def main():
        server = await asyncio.start_unix_server(lane, path=str(sock))
        event = MessageEvent(text="/acp deploy the fix", source=source, message_id="55",
                             platform_update_id=UPDATE_ID)
        runner._experiment_event = event
        original = runner._handle_message_with_agent

        async def capture(ev, *a, **k):
            runner._experiment_event = ev
            return await original(ev, *a, **k)

        runner._handle_message_with_agent = capture
        response = await runner._handle_message(event)
        if scenario == "managed-failed-send":
            ev = runner._experiment_event
            await adapter.send_final_ledgered(ev, entry.session_key, response, {}, reply_to="55")
            await park()
        server.close()

    asyncio.run(main())


def restart(home: Path) -> dict:
    import asyncio

    runner, source, entry = _bootstrap(home)
    from gateway import acp_managed_ingress as ingress
    from gateway import acp_turn_receipts as receipts

    sends = home / "restart-sends.jsonl"
    adapter = _recording_adapter(runner, sends, succeed=True)
    runner.adapters[source.platform] = adapter
    runner._running = True

    async def main():
        resumed, ledgered = await runner._recover_unclean_sessions()
        await asyncio.to_thread(ingress.sweep_at_startup, runner)
        redelivered = await runner._redeliver_pending_obligations()
        return resumed, ledgered, redelivered

    resumed, ledgered, redelivered = asyncio.run(main())
    db = runner.session_store._db_for_key(entry.session_key)
    e = runner.session_store._entries[entry.session_key]
    receipt = receipts.lookup(db, UPDATE_ID)
    restart_sends = [json.loads(l) for l in sends.read_text().splitlines()] if sends.exists() else []
    return {"resumed": resumed, "ledgered": ledgered, "redelivered": redelivered,
            "resume_pending": bool(e.resume_pending), "marker_left": bool(e.active_turn_token),
            "receipt": receipt.status, "restart_sends": [s["content"] for s in restart_sends],
            "ledger": receipts.ledger_answers(db, UPDATE_ID)}


EXPECT = {
    # An admitted task is never re-run and never re-sent after a crash.
    "managed-before-send": lambda r: (r["resume_pending"] is False and r["restart_sends"] == []
                                      and r["receipt"] in ("PENDING", "ABORTED")),
    "managed-failed-send": lambda r: (r["resume_pending"] is False and r["restart_sends"] == []
                                      and r["receipt"] == "PENDING"),
    # The ordinary follow-up's answer is owed exactly once (ordinary crash recovery decorates it as a
    # recovered reply); the managed answer is not re-sent.
    "chain-followup": lambda r: (r["resume_pending"] is False
                                 and sum("ordinary follow-up answer" in s for s in r["restart_sends"]) == 1
                                 and not any("managed answer" in s for s in r["restart_sends"])
                                 and r["receipt"] == "COMPLETED"),
    # A turn that completed before this chain is never adopted as this chain's reply.
    "pre-prompt-after-bind": lambda r: (r["resume_pending"] is False and r["restart_sends"] == []),
    # A withheld managed answer stays unsent after the crash (the receipt stays ABORTED).
    "chain-withheld-before-followup": lambda r: (r["resume_pending"] is False and r["restart_sends"] == []
                                                 and r["receipt"] == "ABORTED"),
    # An unanswered follow-up has no answer to deliver; nothing re-runs, nothing is re-sent.
    "chain-followup-unanswered": lambda r: (r["resume_pending"] is False and r["restart_sends"] == []
                                            and r["receipt"] == "COMPLETED"),
}


def _persisted_state(home: Path) -> dict:
    """Read, from disk and read-only, what the running child has durably written: the session's
    active-turn marker (state.db gateway_routing), the marker binding and the transcript rows (state.db), and
    the ledger rows for the update."""
    import sqlite3

    hermes_home = home / "hermes-home"
    con = sqlite3.connect(f"file:{hermes_home / 'state.db'}?mode=ro", uri=True)
    try:
        marker = any(json.loads(r[0]).get("active_turn_token") for r in con.execute(
            "SELECT entry_json FROM gateway_routing"))
        bound = con.execute("SELECT value FROM state_meta WHERE key LIKE 'acp_turn_marker:%'").fetchall()
        rows = [r[0] for r in con.execute("SELECT content FROM messages WHERE role='assistant' ORDER BY id")]
        users = [json.loads(r[0] or "{}").get("acp_update_id") for r in con.execute(
            "SELECT display_metadata FROM messages WHERE role='user' ORDER BY id")]
        try:
            ledger = [tuple(r) for r in con.execute(
                "SELECT content, state FROM delivery_obligations WHERE acp_update_id = ?", (str(UPDATE_ID),))]
        except sqlite3.OperationalError:
            ledger = []
    finally:
        con.close()
    return {"marker": bool(marker), "bound": [b[0] for b in bound], "assistant_rows": rows,
            "user_acp_ids": users, "ledger": ledger}


def _followup_bound(p):
    """The marker carries the follow-up start recorded when the chain's follow-up started."""
    return len(p["bound"]) == 1 and p["bound"][0].startswith("followup:")


EXPECT_PERSISTED = {
    "managed-before-send": lambda p: str(UPDATE_ID) in p["user_acp_ids"] and p["marker"] and p["bound"] == [str(UPDATE_ID)]
    and "managed answer" in p["assistant_rows"],
    "managed-failed-send": lambda p: str(UPDATE_ID) in p["user_acp_ids"] and p["bound"] == [str(UPDATE_ID)]
    and ("managed answer", "failed") in p["ledger"],
    "chain-followup": lambda p: str(UPDATE_ID) in p["user_acp_ids"] and p["marker"] and _followup_bound(p)
    and "ordinary follow-up answer" in p["assistant_rows"]
    and ("managed answer", "delivered") in p["ledger"],
    "chain-withheld-before-followup": lambda p: str(UPDATE_ID) in p["user_acp_ids"] and p["marker"] and _followup_bound(p)
    and p["assistant_rows"] == ["managed answer"] and p["ledger"] == [],
    "pre-prompt-after-bind": lambda p: p["marker"] and p["bound"] == [str(UPDATE_ID)]
    and p["assistant_rows"] == ["earlier answer"] and str(UPDATE_ID) not in p["user_acp_ids"],
    "chain-followup-unanswered": lambda p: str(UPDATE_ID) in p["user_acp_ids"] and p["marker"] and _followup_bound(p)
    and p["assistant_rows"] == ["managed answer"] and ("managed answer", "delivered") in p["ledger"],
}


def run(scenario: str) -> bool:
    home = Path(tempfile.mkdtemp(prefix=f"acpx-{scenario}-"))
    try:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        proc = subprocess.Popen([sys.executable, __file__, "--child", scenario, str(home)], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        deadline = time.time() + 120
        while not (home / "ready").exists():
            if proc.poll() is not None or time.time() > deadline:
                out = proc.stdout.read().decode(errors="replace")[-3000:]
                print(f"[{scenario}] child never reached its kill point (rc={proc.poll()}):\n{out}")
                proc.kill()
                return False
            time.sleep(0.2)
        persisted = _persisted_state(home)
        if not EXPECT_PERSISTED[scenario](persisted):
            print(f"[{scenario}] running state not on disk before the kill: {persisted}")
            proc.kill()
            return False
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
        result = subprocess.run([sys.executable, __file__, "--restart", str(home)], env=env,
                                capture_output=True, text=True, timeout=300)
        line = next((l for l in result.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            print(f"[{scenario}] restart failed:\n{result.stdout[-2000:]}\n{result.stderr[-3000:]}")
            return False
        observed = json.loads(line[len("RESULT "):])
        ok = EXPECT[scenario](observed)
        print(f"[{scenario}] on disk before kill: {json.dumps(persisted, ensure_ascii=False)}")
        print(f"[{scenario}] killed pid {proc.pid} (SIGKILL) -> {'PASS' if ok else 'FAIL'} {json.dumps(observed, ensure_ascii=False)}")
        return ok
    finally:
        shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--child":
        child(sys.argv[2], Path(sys.argv[3]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "--restart":
        print("RESULT " + json.dumps(restart(Path(sys.argv[2])), ensure_ascii=False))
    else:
        chosen = sys.argv[1:] or list(SCENARIOS)
        results = [run(s) for s in chosen]
        sys.exit(0 if all(results) else 1)
