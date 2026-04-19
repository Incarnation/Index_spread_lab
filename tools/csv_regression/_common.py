"""Shared helpers for the operator-driven CSV regression scripts.

These helpers are intentionally tiny and self-contained so the regression
scripts can be read top-to-bottom by an operator who is verifying a
pre-merge split.  They live next to the scripts (under ``tools/``) rather
than in ``backend/`` so they don't accidentally end up imported by app
code.
"""
from __future__ import annotations

import hashlib
import importlib
import io
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def repo_root() -> Path:
    """Return the repository root.

    Walks upward from this file looking for the marker ``backend/scripts``
    directory so the script works no matter where it is invoked from.
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "backend" / "scripts").is_dir():
            return parent
    raise RuntimeError(
        f"Could not locate repository root above {here}; expected to find "
        "a directory containing backend/scripts/"
    )


def setup_import_paths() -> Path:
    """Add ``backend/`` and ``backend/scripts/`` to ``sys.path``.

    Returns the repo root for convenience.  The double insert is what
    lets ``import backtest_strategy`` (the shim) and
    ``import backtest`` (the new package) both resolve from the same
    process; without it, the operator gets a confusing ``ModuleNotFound``.
    """
    root = repo_root()
    for entry in (root / "backend" / "scripts", root / "backend"):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def fresh_import(module_name: str) -> Any:
    """Import ``module_name`` with no cached state.

    Drops every cached entry whose dotted name starts with the prefix so
    the parity check is not fooled by a partially-initialised module
    that was imported by an earlier ``_run`` call.  This matters for the
    proxy-shim training split where the shim's submodules are
    eagerly attached to ``sys.modules`` at first import.
    """
    for cached in list(sys.modules):
        if cached == module_name or cached.startswith(module_name + "."):
            del sys.modules[cached]
    return importlib.import_module(module_name)


def df_to_canonical_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise ``df`` to CSV bytes with a fixed line terminator.

    Pandas defaults to the platform line terminator which would make
    the SHA differ between macOS and Linux.  Force ``\\n`` and
    UTF-8 so a CI box and a developer laptop produce the same hash.
    """
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8")


def sha256(payload: bytes) -> str:
    """Return the hex SHA-256 of ``payload``.

    Trivial wrapper kept so each script's main loop can be read as
    "hash this, hash that, compare" without importing :mod:`hashlib`.
    """
    return hashlib.sha256(payload).hexdigest()


def report_parity(
    label: str,
    shim_runs: list[bytes],
    pkg_runs: list[bytes],
) -> int:
    """Pretty-print determinism + parity, return exit code (0 = pass).

    ``shim_runs`` / ``pkg_runs`` are lists of CSV byte payloads, one per
    independent invocation.  Two invocations per path are required to
    detect runtime non-determinism that would otherwise make the parity
    check brittle.  Returns 0 only when both paths are internally
    deterministic *and* their hashes match.
    """
    if len(shim_runs) < 2 or len(pkg_runs) < 2:
        raise ValueError(
            "report_parity needs >= 2 runs per path to assess determinism"
        )

    shim_hashes = [sha256(b) for b in shim_runs]
    pkg_hashes = [sha256(b) for b in pkg_runs]

    print(f"\n=== {label} ===")
    for i, h in enumerate(shim_hashes, 1):
        print(f"  shim    run {i}: {h}  ({len(shim_runs[i - 1])} bytes)")
    for i, h in enumerate(pkg_hashes, 1):
        print(f"  package run {i}: {h}  ({len(pkg_runs[i - 1])} bytes)")

    shim_det = len({*shim_hashes}) == 1
    pkg_det = len({*pkg_hashes}) == 1
    parity = shim_hashes[0] == pkg_hashes[0]

    print(f"\n  shim determinism:    {'PASS' if shim_det else 'FAIL'}")
    print(f"  package determinism: {'PASS' if pkg_det else 'FAIL'}")
    print(f"  shim == package:     {'PASS' if parity else 'FAIL'}")

    return 0 if (shim_det and pkg_det and parity) else 1


def require_input(path: Path, hint: str) -> Path:
    """Resolve ``path``, exit with a friendly message when missing.

    Operator-driven scripts run on real data that often only exists on
    the engineer's laptop.  This helper turns a confusing
    :class:`FileNotFoundError` deep inside pandas into an actionable
    "you need to run X first" message before any work happens.
    """
    p = path.expanduser().resolve()
    if not p.exists():
        print(
            f"ERROR: required input not found at {p}\n"
            f"       hint: {hint}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return p
