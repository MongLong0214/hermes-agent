"""Pin HOME, HERMES_HOME and the Hermes home override to a tmp home, then put them back.

A test module that must never resolve the real Hermes home pins twice: while it imports Hermes (so
no import-time path names the real home) and, with its own tmp home, while its tests run (owned by a
module-scoped fixture). Neither pin outlives its scope: a module collected or run later in the same
pytest process sees the HOME, HERMES_HOME, override and ``hermes_constants`` home caches it would
have seen had the pinned module never been collected. Every exit from a pin checks that, and counts
it in ``RESTORE_CHECKS``.

Constants a Hermes module computes from the home when it is first imported (for example
``hermes_state.DEFAULT_DB_PATH`` or ``gateway.run._hermes_home``) cannot be put back: they keep naming
the home in force when they were first imported. Every import-time pin therefore uses one tmp home
per process, ``import_home()``, which no module's tests use: a test that writes through such a
constant (an ordinary gateway turn saving ``config.yaml``, for one) writes there, never into another
module's test home.

Every tmp home here (``tmp_root()``, so ``import_home()`` too) is checked before anything is written:
the dir it would be made in, with symlinks resolved, must not be the real Hermes home or inside it;
if it is, nothing is made. It belongs to the process that made it: only that process (by PID)
removes it, at its exit. A forked child inherits that exit hook, which then
removes nothing, and gets an import home of its own. Pins are NOT supported in a process forked after
an import-time pin ran in its parent: the Hermes modules it inherited keep naming the parent's import
home (``gateway.run._hermes_home``, for one), so what the child reads or writes through them lands
there, and no pin can redirect them. ``import_home()`` and ``pinned_hermes_home()`` raise
``ForkedPinError`` in such a child; a test that needs a home of its own in another process runs in a
fresh interpreter (a subprocess) instead.

Nothing else here runs at interpreter exit. Hooks Hermes registers for exit (the browser tool's cleanup,
for one) resolve the home only then, after every pin is put back, so they see the HOME the process
started with. A test that runs such hooks does so in a subprocess whose HOME is a tmp dir from the
start (tests/agent/test_canonical_peer_tools_isolation.py).
"""

from __future__ import annotations

import atexit
import contextlib
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

try:
    import pwd  # the account database names the real home even when HOME is redirected
except ImportError:  # not POSIX
    pwd = None

_ENV = ("HOME", "HERMES_HOME")
# Module-level values in ``hermes_constants`` that remember a resolved home (or a check of one).
_HOME_CACHES = ("_default_hermes_root_memo", "_PINNED_PROCESS_HERMES_HOME", "_profile_fallback_warned")


def home_state() -> tuple[str | None, str | None, str | None]:
    """HOME and HERMES_HOME (None when unset) and the active home override. Imports nothing from
    Hermes: before ``hermes_constants`` is loaded no override can be active."""
    constants = sys.modules.get("hermes_constants")
    override = constants.get_hermes_home_override() if constants is not None else None
    return os.environ.get("HOME"), os.environ.get("HERMES_HOME"), override


#: The ``__name__`` of each module whose import-time pin has been entered, in order.
IMPORT_PINNED: list[str] = []
_IMPORT_PIN_PIDS: set[int] = set()  # the process(es) those import-time pins ran in
_IMPORT_HOME: dict[int, Path] = {}  # owning PID -> that process's import home


class ForkedPinError(RuntimeError):
    """A pin used in a process forked after an import-time pin ran in its parent (module docstring)."""


def _refuse_a_child_forked_after_import_pins() -> None:
    parents = sorted(_IMPORT_PIN_PIDS - {os.getpid()})
    if parents:
        raise ForkedPinError(
            f"process {os.getpid()} was forked from process {parents[0]} after import-time pins ran there "
            f"({', '.join(IMPORT_PINNED)}); the Hermes modules it inherited still name that process's import "
            f"home, and no pin can redirect them: run this in a fresh interpreter (a subprocess) instead")


def _remove_if_owner(root: Path, owner_pid: int) -> None:
    if os.getpid() == owner_pid:  # in a forked child, which inherits this exit hook, removes nothing
        shutil.rmtree(root, ignore_errors=True)


def _real_hermes_home() -> Path:
    return (Path(pwd.getpwuid(os.getuid()).pw_dir) if pwd is not None else Path.home()) / ".hermes"


def _tmp_parent() -> Path:
    """The dir a tmp home is made in, found without writing anything: ``tempfile.tempdir`` once set,
    else the first of ``tempfile.gettempdir()``'s candidates, in its order, that is a dir this process
    may write in. (``gettempdir()`` itself, until it has cached an answer, writes a probe file into
    each candidate it tries.)"""
    if tempfile.tempdir is not None:
        return Path(tempfile.tempdir)
    candidates = tempfile._candidate_tempdir_list()
    for candidate in candidates:
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK | os.X_OK):
            return Path(candidate)
    raise FileNotFoundError(f"no tmp dir this process may write in among {candidates}")


def _is_in(path: Path, home: Path) -> bool:
    """Whether ``path`` is ``home`` or inside it. Symlinks are resolved and paths compared part by part
    (``.hermes2`` is not in ``.hermes``), and each dir from ``path`` up is also compared with ``home``
    by identity, so another spelling of the same dir (another case, on a case-insensitive file
    system) counts as ``home``."""
    path, home = path.resolve(), home.resolve()
    if path.is_relative_to(home):
        return True
    try:
        home_stat = home.stat()
    except OSError:  # no such home: nothing is inside it
        return False
    for dir_ in (path, *path.parents):
        with contextlib.suppress(OSError):
            if os.path.samestat(dir_.stat(), home_stat):
                return True
    return False


def tmp_root(prefix: str) -> Path:
    """A fresh tmp dir outside the real Hermes home, made by this process and removed at this
    process's exit, only by this process. The dir it would be made in is checked first: one that is
    the real Hermes home or inside it raises, and nothing is written (a fresh dir cannot hold the
    real home, so only where it goes is checked)."""
    parent = _tmp_parent().resolve()
    if _is_in(parent, _real_hermes_home()):
        raise RuntimeError(f"a test home made in {parent} would be inside the real home; nothing was made")
    root = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    atexit.register(_remove_if_owner, root, os.getpid())
    return root


def import_home() -> Path:
    """The tmp home every import-time pin in this process uses; made on first use in each process (a
    forked child included), removed at that process's exit. Raises ``ForkedPinError`` in a process
    forked after an import-time pin ran in its parent."""
    _refuse_a_child_forked_after_import_pins()
    pid = os.getpid()
    if pid not in _IMPORT_HOME:
        _IMPORT_HOME[pid] = tmp_root("hermes-test-imports-")
    return _IMPORT_HOME[pid]


#: How many pin exits found each part put back as it was on entry: HOME and HERMES_HOME, the home
#: override, and the ``hermes_constants`` home caches. A part that is not put back raises instead.
RESTORE_CHECKS: dict[str, int] = {"env": 0, "override": 0, "home_caches": 0}


def _home_caches(constants) -> tuple:
    key_cache = getattr(constants, "_HOME_KEY_CACHE", None)
    return (dict(key_cache) if key_cache is not None else None,
            {name: getattr(constants, name) for name in _HOME_CACHES if hasattr(constants, name)})


def _restore_home_caches(constants, saved) -> None:
    saved_keys, saved_values = saved
    if saved_keys is not None:
        constants._HOME_KEY_CACHE.clear()
        constants._HOME_KEY_CACHE.update(saved_keys)
    for name, value in saved_values.items():
        setattr(constants, name, value)


@contextlib.contextmanager
def pinned_hermes_home(root: Path, *, importer: str | None = None) -> Iterator[Path]:
    """Inside the block HOME is ``root`` and HERMES_HOME and the context-local home override are
    ``root/.hermes``. On the way out, an exception included, both variables (set or unset), the
    override and the home caches of ``hermes_constants`` are what they were on the way in; each is
    checked, and counted in ``RESTORE_CHECKS``.

    ``importer``: the pinning module's ``__name__``, for an import-time pin (see ``IMPORT_PINNED``).
    Raises ``ForkedPinError``, changing nothing, in a process forked after an import-time pin ran in
    its parent.
    """
    _refuse_a_child_forked_after_import_pins()
    hermes_home = root / ".hermes"
    saved_env = {name: os.environ.get(name) for name in _ENV}
    saved_caches = saved_override = token = constants = None
    os.environ["HOME"] = str(root)
    os.environ["HERMES_HOME"] = str(hermes_home)
    try:
        import hermes_constants as constants  # only once HOME and HERMES_HOME name the tmp home

        saved_caches = _home_caches(constants)
        saved_override = constants.get_hermes_home_override()
        token = constants.set_hermes_home_override(hermes_home)
        if importer is not None:
            IMPORT_PINNED.append(importer)
            _IMPORT_PIN_PIDS.add(os.getpid())
        yield hermes_home
    finally:
        if token is not None:
            constants.reset_hermes_home_override(token)
        if saved_caches is not None:
            _restore_home_caches(constants, saved_caches)
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _check_put_back(saved_env, constants, token, saved_override, saved_caches)


def _check_put_back(saved_env, constants, token, saved_override, saved_caches) -> None:
    left = []
    if {name: os.environ.get(name) for name in _ENV} == saved_env:
        RESTORE_CHECKS["env"] += 1
    else:
        left.append("HOME/HERMES_HOME")
    if token is not None:
        if constants.get_hermes_home_override() == saved_override:
            RESTORE_CHECKS["override"] += 1
        else:
            left.append("home override")
    if saved_caches is not None:
        if _home_caches(constants) == saved_caches:
            RESTORE_CHECKS["home_caches"] += 1
        else:
            left.append("hermes_constants home caches")
    if left:
        raise RuntimeError(f"a Hermes home pin was not put back: {', '.join(left)}")
