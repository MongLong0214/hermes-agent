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

Nothing here runs at interpreter exit. Hooks Hermes registers for exit (the browser tool's cleanup,
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
_IMPORT_HOME: list[Path] = []


def import_home() -> Path:
    """The tmp home every import-time pin in this process uses; created on first use, removed at exit."""
    if not _IMPORT_HOME:
        real_hermes_home = ((Path(pwd.getpwuid(os.getuid()).pw_dir) if pwd is not None else Path.home())
                            / ".hermes").resolve()
        root = Path(tempfile.mkdtemp(prefix="hermes-test-imports-")).resolve()
        if root.is_relative_to(real_hermes_home) or real_hermes_home.is_relative_to(root):
            raise RuntimeError(f"the test import home {root} overlaps the real home")
        atexit.register(shutil.rmtree, root, True)
        _IMPORT_HOME.append(root)
    return _IMPORT_HOME[0]


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
    """
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
