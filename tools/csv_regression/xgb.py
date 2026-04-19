"""JSON regression check for the xgb_model split (Phase 3.1).

Runs ``walk_forward_rolling`` on a pinned slice of the candidate CSV
twice via the back-compat shim ``xgb_model`` and twice via the new
``xgb`` package, then verifies the JSON-serialised results dict is
byte-identical across both paths and across runs.

XGBoost is forced into deterministic mode (``random_state=42``,
``n_jobs=1``, ``tree_method='exact'``) so the parity check isn't
defeated by histogram-binning thread-order non-determinism.  Early
stopping is disabled because the trigger threshold depends on
val-loss differences that themselves vary by 1-2 ULP; without that
pin two independent runs disagree on tree count and the output JSON
structurally drifts.  Floats in the output dict are rounded to 6
decimal places before serialisation so library-level numerical drift
(between, say, ``xgboost==2.x`` patch versions) doesn't mask a real
semantic regression.

Usage::

    python tools/csv_regression/xgb.py \\
        --candidates data/training_candidates.csv \\
        --start 2024-01-01 --end 2024-12-31 \\
        --train-months 6 --test-months 2

Operator-driven; not part of the pytest suite.  Exits 0 on PASS, 1 on
FAIL, 2 on missing input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    fresh_import,
    report_parity,
    require_input,
    setup_import_paths,
)


# Single-threaded + fixed-seed override so two independent runs hash
# the same.  ``exact`` is slower than ``hist`` but produces identical
# splits across runs (``hist`` is the XGBoost default but its histogram
# reduction has thread-order sensitivity even with n_jobs=1 on some
# CPUs).  ``early_stopping_rounds=None`` + a low ``n_estimators`` keeps
# the run from drifting on tree count when val-loss ties shift by 1
# ULP between runs.
DETERMINISTIC_PARAMS = {
    "random_state": 42,
    "n_jobs": 1,
    "tree_method": "exact",
    "n_estimators": 50,
    "early_stopping_rounds": None,
}


# Fields that are inherently non-deterministic (wall clock, machine
# fingerprints, etc.) and would defeat the parity check even when the
# underlying training is byte-identical.  Drop them before hashing.
_DROP_KEYS = frozenset({"train_time_s", "wall_time_s", "elapsed_s"})


def _normalize(obj: Any, ndigits: int = 6) -> Any:
    """Recursively round floats and drop wall-clock keys.

    ``walk_forward_rolling`` returns a dict that nests numpy floats,
    lists of dicts, etc.  Without rounding, library-level numeric
    jitter (e.g. one xgboost patch tweaking a leaf-pruning tiebreaker)
    would dominate the hash and hide real semantic drift.  The
    ``_DROP_KEYS`` filter removes wall-clock timing fields so the
    parity check measures *what* was computed, not *how long* it took.
    """
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {
            k: _normalize(v, ndigits)
            for k, v in obj.items()
            if k not in _DROP_KEYS
        }
    if isinstance(obj, (list, tuple)):
        return [_normalize(x, ndigits) for x in obj]
    if hasattr(obj, "tolist"):
        try:
            return _normalize(obj.tolist(), ndigits)
        except Exception:
            pass
    return obj


def _run(via: str, df: pd.DataFrame, train_months: int, test_months: int) -> bytes:
    """Run walk_forward via ``via`` and return canonical JSON bytes.

    Each call drops cached modules so the proxy-shim training split's
    eager-attached submodules don't leak across ``_run`` invocations
    (xgb itself doesn't use a proxy shim, but the helper is symmetric).
    Also pokes ``DEFAULT_REG_PARAMS`` into deterministic mode -- the
    public ``walk_forward_rolling`` API only accepts ``cls_params``,
    so we have to mutate the module-level constant to make the
    auxiliary regressor that runs alongside the classifier respect
    the determinism contract.
    """
    if via == "shim":
        mod = fresh_import("xgb_model")
    elif via == "package":
        mod = fresh_import("xgb")
    else:
        raise ValueError(f"unknown via: {via!r}")

    cls_params = dict(mod.DEFAULT_CLS_PARAMS) if hasattr(mod, "DEFAULT_CLS_PARAMS") else {}
    cls_params.update(DETERMINISTIC_PARAMS)

    # Override the auxiliary regressor's defaults so the per-fold
    # train_xgb_models call isn't a determinism leak.  The proxy-shim
    # training split forwards setattr to submodules; the xgb shim
    # re-exports DEFAULT_REG_PARAMS directly so writing to the shim
    # is sufficient.
    if hasattr(mod, "DEFAULT_REG_PARAMS"):
        mod.DEFAULT_REG_PARAMS = {**mod.DEFAULT_REG_PARAMS, **DETERMINISTIC_PARAMS}

    result = mod.walk_forward_rolling(
        df,
        train_months=train_months,
        test_months=test_months,
        step_months=1,
        cls_params=cls_params,
    )
    serial = _normalize(result)
    # ``sort_keys`` ensures dict insertion order doesn't affect the
    # hash; ``default=str`` handles datetime / numpy scalars that
    # aren't natively JSON-serialisable.
    return json.dumps(
        serial, sort_keys=True, default=str, ensure_ascii=False,
    ).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path,
        default=Path("data/training_candidates.csv"),
        help="Path to training_candidates.csv (default: data/...)",
    )
    parser.add_argument(
        "--start", default="2024-01-01",
        help="First day (inclusive) of slice loaded for walk-forward.",
    )
    parser.add_argument(
        "--end", default="2024-12-31",
        help="Last day (inclusive) of slice loaded for walk-forward.",
    )
    parser.add_argument(
        "--train-months", type=int, default=6,
        help="walk_forward train window length in months (default 6).",
    )
    parser.add_argument(
        "--test-months", type=int, default=2,
        help="walk_forward test window length in months (default 2).",
    )
    args = parser.parse_args(argv)

    setup_import_paths()
    csv_path = require_input(
        args.candidates,
        "run backend/scripts/generate_training_data.py to produce it",
    )
    print(
        f"XGB walk-forward regression: {csv_path.name} "
        f"window {args.start} .. {args.end} "
        f"(train={args.train_months}m / test={args.test_months}m)"
    )

    df_full = pd.read_csv(csv_path)
    df = df_full[(df_full["day"] >= args.start) & (df_full["day"] <= args.end)].copy()
    if df.empty:
        print(
            f"FAIL: no rows in {csv_path} between {args.start} and {args.end}",
            file=sys.stderr,
        )
        return 1

    shim_runs = [
        _run("shim", df, args.train_months, args.test_months) for _ in range(2)
    ]
    pkg_runs = [
        _run("package", df, args.train_months, args.test_months) for _ in range(2)
    ]
    return report_parity("xgb_model", shim_runs, pkg_runs)


if __name__ == "__main__":
    sys.exit(main())
