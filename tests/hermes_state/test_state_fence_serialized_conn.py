"""One thread at a time inside SQLite per state.db connection.

While thread A runs a Python UDF mid-statement (as every fenced write does), thread B's execute
on the same connection must wait for A's statement to return. Unserialized, B takes the GIL and
blocks on the connection mutex A holds while A's callback waits for the GIL: a process-wide
deadlock. It runs in a subprocess so the unserialized case fails by timeout instead of hanging
the runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_CHILD = textwrap.dedent("""
    import json, threading, time
    from hermes_state import SessionDB
    from hermes_constants import get_hermes_home

    db = SessionDB(db_path=get_hermes_home() / "state.db")
    conn = db._conn
    in_udf = threading.Event()
    # Only the invariant pair: once A's statement releases the connection, A's own return and
    # B's may land in either order, so A's return is recorded apart.
    order = []
    a_result = []

    def slow_udf():
        in_udf.set()
        time.sleep(2.0)
        order.append("udf_done")
        return 1

    conn.create_function("slow_udf", 0, slow_udf)
    conn.execute("SELECT ?", (0,)).fetchone()  # cached statement: B's bind path takes the mutex under the GIL

    def thread_a():
        a_result.append(conn.execute("SELECT slow_udf()").fetchone()[0])

    def thread_b():
        in_udf.wait(10)
        conn.execute("SELECT ?", (2,)).fetchone()
        order.append("b_returned")

    threads = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    db.close()
    print(json.dumps({"order": order, "a_result": a_result}))
""")


def test_second_thread_waits_for_a_statement_inside_a_udf(tmp_path):
    home, hermes_home = tmp_path / "home", tmp_path / "hermes_home"
    home.mkdir()
    hermes_home.mkdir()
    env = {**os.environ, "HOME": str(home), "HERMES_HOME": str(hermes_home), "PYTHONPATH": str(REPO_ROOT)}
    try:
        result = subprocess.run([sys.executable, "-c", _CHILD], cwd=REPO_ROOT, env=env, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise AssertionError("connection deadlocked: thread B entered SQLite while A's UDF waited for the GIL")

    assert result.returncode == 0, result.stderr[-2000:]
    # B's statement returns only after A's UDF has finished: B never ran inside A's statement.
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"order": ["udf_done", "b_returned"], "a_result": [1]}
