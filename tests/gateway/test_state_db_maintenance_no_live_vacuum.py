"""Gateway housekeeping never runs a full-store VACUUM on the store its turns are writing.

``_housekeeping_state_db_maintenance`` acquires the SAME shared SessionDB the gateway's Telegram turns
write through, and the auto-VACUUM holder gate excludes that very instance. With the default
``vacuum_after_prune: true``, a prune that left more than 25% of pages free ran ``VACUUM`` plus a
TRUNCATE checkpoint while holding that instance's lock, so every reply's transcript write waited
out a full rewrite of the store (5.7 GB live). The rewrite belongs to the explicit offline
``hermes sessions optimize``; the gateway still prunes.

Real store, real config file, real shared-instance registry: the VACUUM is observed on the
connection's own statement trace.
"""

from __future__ import annotations

import time

import gateway.run as gateway_run
import hermes_state_registry as registry


def _seed(db) -> None:
    """One long-expired session whose pages dominate the file, plus one live chat."""
    old = time.time() - 400 * 86400

    def _do(conn):
        conn.execute("INSERT INTO sessions (id, source, started_at, ended_at, end_reason, message_count)"
                     " VALUES ('old', 'cli', ?, ?, 'done', 300)", (old, old + 1))
        conn.executemany("INSERT INTO messages (session_id, role, content, timestamp) VALUES ('old', 'user', ?, ?)",
                         [(f"{i} " + "x" * 4000, old) for i in range(300)])
    db._execute_write(_do)
    db.create_session("live", "telegram")


def test_gateway_maintenance_prunes_but_never_vacuums_the_served_store(monkeypatch):
    import hermes_state
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "sessions:\n"
        "  auto_prune: true\n"
        "  retention_days: 90\n"
        "  min_interval_hours: 0\n"
        "  min_vacuum_interval_days: 0\n"
        "  vacuum_after_prune: true\n",
        encoding="utf-8")
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")

    served = registry.acquire()  # the handle the gateway's turns write through
    try:
        _seed(served)
        statements: list = []
        served._conn.set_trace_callback(statements.append)

        gateway_run._housekeeping_state_db_maintenance()

        served._conn.set_trace_callback(None)
        assert served.get_session("old") is None, "housekeeping did not prune the expired session"
        vacuums = [s for s in statements if s.strip().upper().startswith("VACUUM")]
        assert not vacuums, f"gateway housekeeping rewrote the store it serves: {vacuums}"
        # The prune left a freelist the automatic gate would have VACUUMed for (> 25%).
        ratio = served._freelist_ratio()
        assert ratio is not None and ratio > 0.25, ratio
        served.append_message("live", "user", "still writable")
    finally:
        registry.release(served)
