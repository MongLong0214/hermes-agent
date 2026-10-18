"""Guard for the bridge shim's tests: none reaches a real canonical surface or runs in the real home.

The shim's defaults name the live gateway (``127.0.0.1:8642``) and paths under ``~/.hermes``. A test
that routes the shim's HTTP to its own in-process gateway replaces ``urlopen`` itself; any other
request fails the test before it leaves the process.
"""

import os
import urllib.request
from pathlib import Path

import pytest

try:
    import pwd  # the account database names the real home even when HOME is redirected
except ImportError:  # not POSIX
    pwd = None

_REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve() if pwd is not None else None


@pytest.fixture(autouse=True)
def _no_real_canonical_surface(monkeypatch):
    if _REAL_HOME is not None and Path(os.path.expanduser("~")).resolve() == _REAL_HOME:
        pytest.fail("HOME is the real home: the shim's default paths would be live")

    def refuse(request, *args, **kwargs):
        pytest.fail(f"a shim test reached for a real canonical surface: {getattr(request, 'full_url', request)}")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
