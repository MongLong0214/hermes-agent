"""Packaging: the wheel ships ``bridge.canonical``, which the gateway's canonical surface imports.

``gateway.canonical_surface`` imports ``bridge.canonical.canonical_roster`` at runtime. ``bridge`` and
``bridge.canonical`` are namespace packages (no ``__init__.py``), so setuptools ships them only when
the pyproject ``packages.find`` include list names them; an installed gateway otherwise fails that
import. The package list is the one setuptools itself expands from pyproject.toml.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from setuptools.config.pyprojecttoml import read_configuration

REPO_ROOT = Path(__file__).resolve().parent.parent


def _shipped_packages() -> set[str]:
    config = read_configuration(REPO_ROOT / "pyproject.toml", expand=True)
    return set(config["tool"]["setuptools"]["packages"])


def test_the_wheel_ships_the_bridge_canonical_package_the_gateway_imports():
    spec = importlib.util.find_spec("bridge.canonical.canonical_roster")
    assert spec is not None and Path(spec.origin).is_relative_to(REPO_ROOT)
    shipped = _shipped_packages()
    assert "gateway" in shipped  # the importer ships, so its import must resolve in the wheel
    assert {"bridge", "bridge.canonical"} <= shipped
