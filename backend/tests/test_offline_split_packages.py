"""Smoke tests for the three monolith-split package paths.

The Wave 5 refactor (see ``OFFLINE_PIPELINE_AUDIT.md``) split three
~1,500--3,500-line script monoliths into proper packages while
preserving the original module names as back-compat shims:

* ``xgb_model``               -> ``backend/scripts/xgb/``
* ``backtest_strategy``       -> ``backend/scripts/backtest/``
* ``generate_training_data``  -> ``backend/scripts/training/``

Existing tests still import via the *shim* names, which means a
mistake that breaks the *new* package-path entry point (e.g. somebody
removes a re-export from ``xgb/__init__.py`` or renames a submodule)
would not surface in CI; only the operator-driven CSV regression in
``tools/csv_regression/`` and the manual bytecode regression in
``tools/monolith_split_regression.py`` would catch it.

These tests close that gap: for each split they import the *package*
directly, walk every submodule, and verify that the package's public
symbol surface matches the shim's exactly (modulo the submodule
attributes themselves, which are package-only).  They run in <1s and
require no external data.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Put backend/scripts/ on sys.path so the flat-name shim and the
# package can both be imported by their canonical names (mirrors how
# the rest of the test suite reaches scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


# (package_name, shim_name, expected submodules) for each Wave 5 split.
# The submodule list is asserted both as "present in the package's
# vars()" and as "the only diff between package and shim public
# surface".  Any drift raises a clear AssertionError.
_SPLITS = [
    ("xgb", "xgb_model", {"features", "training", "walkforward", "cli"}),
    (
        "backtest",
        "backtest_strategy",
        {"engine", "optimizer", "analysis", "cli"},
    ),
    (
        "training",
        "generate_training_data",
        {"bs_gex_spot", "io_loaders", "candidates", "labeling", "cli"},
    ),
]


@pytest.mark.parametrize("pkg_name,shim_name,expected_submods", _SPLITS)
def test_package_imports_and_matches_shim_surface(
    pkg_name: str,
    shim_name: str,
    expected_submods: set[str],
) -> None:
    """Importing the new package path must succeed and surface the same public API as the shim.

    Asserts (for each of the three Wave 5 splits):

    1. ``import <pkg_name>`` succeeds.
    2. ``import <shim_name>`` succeeds (back-compat unchanged).
    3. ``set(dir(pkg)) - set(dir(shim))`` is exactly ``expected_submods``
       (the package exposes its submodules as attributes; the shim is
       a flat module so it doesn't).
    4. ``set(dir(shim)) - set(dir(pkg))`` is empty (no shim-only
       symbols -- the shim must re-export everything from the package).
    5. Every expected submodule is also importable as a stand-alone
       module via the package (e.g. ``from xgb import features``).
    """
    pkg = importlib.import_module(pkg_name)
    shim = importlib.import_module(shim_name)

    pkg_public = {n for n in dir(pkg) if not n.startswith("_")}
    shim_public = {n for n in dir(shim) if not n.startswith("_")}

    only_in_pkg = pkg_public - shim_public
    only_in_shim = shim_public - pkg_public

    # The only legitimate difference is the package exposing its
    # submodules as attributes; the shim is flat so it doesn't.
    assert only_in_pkg == expected_submods, (
        f"{pkg_name}: unexpected package-only symbols: "
        f"{sorted(only_in_pkg - expected_submods)} "
        f"(missing expected: {sorted(expected_submods - only_in_pkg)})"
    )
    assert only_in_shim == set(), (
        f"{shim_name}: shim-only symbols (shim must not add new public names): "
        f"{sorted(only_in_shim)}"
    )

    # Every expected submodule must be reachable via the package.
    for submod in expected_submods:
        full_name = f"{pkg_name}.{submod}"
        importlib.import_module(full_name)
        assert hasattr(pkg, submod), (
            f"{pkg_name}.{submod} exists as a module but is not "
            f"exposed as an attribute on the parent package"
        )
