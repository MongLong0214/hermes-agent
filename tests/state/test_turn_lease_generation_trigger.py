"""Blocker (b): the store itself has to refuse an old binary's transcript write.

WHAT ``epoch NOT NULL`` DOES AND DOES NOT DO
    The epoch column stops a binary that predates it from *creating a lease
    row*: the INSERT has no value for a NOT NULL column and fails. That is a
    fence on the lease table.

    It is not a fence on ``messages``. An old binary never wanted a lease row —
    it writes the transcript holderlessly, which is what every writer did before
    this work, and the schema has nothing to say about that. So the exact
    binary at the base commit can still open a store this generation created,
    append to a conversation this generation holds the lease on, and be told
    nothing. Every guard added so far lives in Python that the old binary is not
    running.

THE ONE PLACE THAT IS COMMON TO BOTH BINARIES
    The database file. A trigger on ``messages`` whose body calls an
    application-defined function turns "did this connection register the
    function" into a precondition of the statement: SQLite resolves the trigger
    program when it PREPARES the write, so a connection that did not register
    it fails before any row is touched, with ``no such function``.

    Registration is done by this generation's connect path, so "this generation"
    is exactly the set of processes that can write.

TWO TESTS, DELIBERATELY
    :func:`test_a_foreign_connection_cannot_write_messages` is the mechanism,
    with no old binary and no git: a plain ``sqlite3.connect`` IS the general
    case of a process that did not register the function, and the assertion is
    about statement preparation. It runs anywhere.

    :func:`test_the_base_binary_cannot_write_a_store_this_generation_created`
    is the claim: the actual module tree at the base commit, extracted with
    ``git archive`` and imported in a subprocess whose ``sys.path`` and cwd are
    that tree, asserting its own ``hermes_state.__file__`` lives under the
    extract so the candidate cannot leak in through an inherited path.
"""

from __future__ import annotations

import os
import pathlib
import sqlite3
import subprocess
import sys

import pytest

#: The base commit this branch is measured against. The exact binary, not "an
#: older one" — the whole point is that this is checkable.
BASE_COMMIT = "261a4efb90d7dbe4e71786861858f721b4ab730c"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Enough of the base tree to `import hermes_state` and open a store. Extracting
#: the whole commit works and costs 14s and 173MB; this costs 0.15s and 1MB,
#: and the subprocess asserts it really is running the extracted module.
BASE_TREE_PATHSPEC = (
    "hermes_state.py",
    "hermes_state_common.py",
    "hermes_state_portability.py",
    "hermes_state_schema.py",
    "hermes_state_search.py",
    "hermes_constants.py",
    "hermes_logging.py",
    "hermes_time.py",
    "utils.py",
    "hermes_cli/__init__.py",
    "hermes_cli/sqlite_safe_read.py",
)


def _git_dir() -> str | None:
    """Where to read :data:`BASE_COMMIT` from, or None when it is unreachable.

    ``HERMES_BASE_TREE_GIT_DIR`` exists so this test can also run from a bare
    ``git archive`` extract of its own commit, which is not a repository. It
    names an immutable object, not a working tree, so it cannot smuggle
    uncommitted state into the fixture.
    """
    override = os.environ.get("HERMES_BASE_TREE_GIT_DIR")
    if override:
        return override
    probe = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--git-dir"],
        capture_output=True, text=True,
    )
    return str(REPO_ROOT) if probe.returncode == 0 else None


@pytest.fixture
def base_binary_tree(tmp_path):
    git_dir = _git_dir()
    if git_dir is None:
        pytest.skip(
            "no git repository to read the base commit from; set "
            "HERMES_BASE_TREE_GIT_DIR to one. The MECHANISM this test checks "
            "is covered without git by "
            "test_a_foreign_connection_cannot_write_messages."
        )
    out = tmp_path / "base"
    out.mkdir()
    archive = subprocess.run(
        ["git", "-C", git_dir, "archive", BASE_COMMIT, "--", *BASE_TREE_PATHSPEC],
        capture_output=True,
    )
    assert archive.returncode == 0, (
        f"could not read {BASE_COMMIT} out of {git_dir}: "
        f"{archive.stderr.decode(errors='replace')}"
    )
    extract = subprocess.run(
        ["tar", "-x", "-C", str(out)], input=archive.stdout, capture_output=True
    )
    assert extract.returncode == 0, extract.stderr.decode(errors="replace")
    assert (out / "hermes_state.py").is_file(), "the base extract has no hermes_state"
    return out


def _new_generation_store(path: pathlib.Path):
    """A store written by THIS generation, with a live lease on 's'."""
    from hermes_state import SessionDB

    db = SessionDB(path)
    db.create_session("s", source="test")
    grant = db.try_acquire_session_turn_lease(
        "s", f"pid={os.getpid()}:turn=live:platform=test", ttl_seconds=600
    )
    assert grant, "could not take the lease this proof depends on"
    db.append_message(
        session_id="s", role="user", content="current", turn_lease_holder=grant
    )
    db.close()
    return grant


def test_a_foreign_connection_cannot_write_messages(tmp_path):
    """A connection that did not register this generation's function is refused.

    This is the mechanism, stated without reference to any particular binary: a
    plain ``sqlite3.connect`` is precisely a process that opened the file and
    registered nothing, which is what an old binary is. The failure has to come
    from statement PREPARATION — before any row is touched — or a writer that
    ignores errors still lands the write.
    """
    store = tmp_path / "state.db"
    _new_generation_store(store)

    foreign = sqlite3.connect(str(store))
    try:
        for sql, params in (
            ("INSERT INTO messages (session_id, role, content, timestamp, active) "
             "VALUES ('s', 'assistant', 'foreign', 1.0, 1)", ()),
            ("UPDATE messages SET content = 'clobbered' WHERE session_id = 's'", ()),
            ("DELETE FROM messages WHERE session_id = 's'", ()),
        ):
            with pytest.raises(sqlite3.OperationalError) as caught:
                foreign.execute(sql, params)
            assert "no such function" in str(caught.value).lower(), (
                f"the statement was refused, but not by the generation fence: "
                f"{caught.value!r} for {sql!r}"
            )
    finally:
        foreign.close()

    # And the transcript is untouched: refusing at prepare means nothing partial
    # landed, which a refusal at COMMIT would not give.
    from hermes_state import SessionDB

    db = SessionDB(store)
    assert [m["content"] for m in db.get_messages("s")] == ["current"]
    db.close()


def test_this_generation_still_writes_normally(tmp_path):
    """The fence must not be a fence on us.

    Named separately from the census because a mechanism that stops everyone is
    trivially safe and completely useless, and the two failures look identical
    from the outside once a test only checks the refusal.
    """
    from hermes_state import SessionDB

    store = tmp_path / "state.db"
    grant = _new_generation_store(store)
    db = SessionDB(store)
    db.append_message(
        session_id="s", role="assistant", content="second", turn_lease_holder=grant
    )
    db.rewind_to_message("s", 1, turn_lease_holder=grant)
    db.clear_messages("s", turn_lease_holder=grant)
    db.close()

    # Reopening runs schema init, which itself writes `messages`.
    reopened = SessionDB(store)
    reopened.append_message(
        session_id="s", role="user", content="after reopen",
        turn_lease_holder=grant,
    )
    assert [m["content"] for m in reopened.get_messages("s")] == ["after reopen"]
    reopened.close()


def test_the_base_binary_cannot_write_a_store_this_generation_created(
    tmp_path, base_binary_tree
):
    """The claim, against the exact module tree at the base commit."""
    store = tmp_path / "state.db"
    _new_generation_store(store)

    probe = f'''
import pathlib, sys, traceback
import hermes_state

here = pathlib.Path({str(base_binary_tree)!r}).resolve()
loaded = pathlib.Path(hermes_state.__file__).resolve()
assert loaded.is_relative_to(here), (
    "this subprocess imported %s, not the extracted base tree under %s -- "
    "the candidate leaked in and the result would be meaningless"
    % (loaded, here)
)

db = hermes_state.SessionDB(pathlib.Path({str(store)!r}))
try:
    db.append_message(session_id="s", role="assistant", content="OLD BINARY WROTE")
except Exception as exc:
    print("REFUSED:", type(exc).__name__, exc)
else:
    print("WROTE")
finally:
    try:
        db.close()
    except Exception:
        traceback.print_exc()
'''
    home = tmp_path / "home"
    home.mkdir()
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(base_binary_tree),
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
            "PYTHONPATH": str(base_binary_tree),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, (
        f"the base-binary probe did not run to completion, so it proves "
        f"nothing:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "REFUSED:" in result.stdout, (
        f"the binary at {BASE_COMMIT[:10]} appended to a conversation this "
        f"generation holds the lease on, and nothing stopped it. epoch NOT NULL "
        f"fences the lease table; it says nothing about a holderless transcript "
        f"write.\nprobe stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )

    from hermes_state import SessionDB

    db = SessionDB(store)
    assert [m["content"] for m in db.get_messages("s")] == ["current"], (
        "the base binary's write landed even though it reported a refusal"
    )
    db.close()
