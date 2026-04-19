"""JSON regression check for the generate_training_data split (Phase 3.3).

Runs the most-load-bearing pure helpers from the training package on a
synthetic options-chain fixture twice via the back-compat shim
``generate_training_data`` and twice via the new ``training`` package.
Verifies the JSON-serialised output is byte-identical across both
paths and across runs.

The full end-to-end candidate generation requires a production database
+ Databento parquets + GEX OI snapshots, which makes a true ``--day``
regression operator-laptop-only.  This script is the cheaper companion
that runs anywhere and exercises the BS pricing, BS delta, IV bisection,
and outcome-evaluation helpers -- the same code paths that audit
findings M3 / M4 / M9 modified.

Usage::

    python tools/csv_regression/training.py            # default fixture
    python tools/csv_regression/training.py --seed 7   # other fixture seed

Operator-driven; not part of the pytest suite (the proxy-shim setattr
forwarding tests live there).  Exits 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    fresh_import,
    report_parity,
    setup_import_paths,
)


def _build_fixture(seed: int) -> dict:
    """Return a deterministic synthetic options chain.

    Uses a fixed numpy ``Generator`` so two invocations of this helper
    with the same ``seed`` produce identical inputs; that's what makes
    the downstream parity check meaningful (any drift is in the
    helpers, not the fixture).
    """
    rng = np.random.default_rng(seed)
    n = 64
    spot = 5500.0
    strikes = np.linspace(spot * 0.8, spot * 1.2, n)
    is_call = np.array([(i % 2 == 0) for i in range(n)], dtype=bool)
    sigma = rng.uniform(0.10, 0.40, size=n)
    T = 7.0 / 365.0
    r = 0.05
    return {
        "S": np.full(n, spot),
        "K": strikes,
        "sigma": sigma,
        "is_call": is_call,
        "T": T,
        "r": r,
    }


def _build_outcome_marks(seed: int) -> tuple[float, list[dict]]:
    """Return a deterministic ``(entry_credit, marks)`` pair.

    Used by ``_evaluate_outcome``; produces a credit-spread quote
    series (short_bid/short_ask, long_bid/long_ask) where the spread
    width drifts according to a seeded random walk.  Audit M4 fixed
    the SL/TP ordering inside this evaluator, so we want a trajectory
    that crosses both TP and SL thresholds during the path.
    """
    rng = np.random.default_rng(seed + 1000)
    n = 120
    short_mid_walk = 1.50 + np.cumsum(rng.normal(0.0, 0.04, size=n))
    long_mid_walk = 0.10 + np.cumsum(rng.normal(0.0, 0.02, size=n))
    half_spread = 0.05
    marks: list[dict] = []
    for i in range(n):
        sm = float(round(max(0.05, short_mid_walk[i]), 4))
        lm = float(round(max(0.01, long_mid_walk[i]), 4))
        marks.append({
            "t": float(i),
            "short_bid": float(round(sm - half_spread, 4)),
            "short_ask": float(round(sm + half_spread, 4)),
            "long_bid": float(round(lm - half_spread, 4)),
            "long_ask": float(round(lm + half_spread, 4)),
        })
    entry_credit = 1.40
    return entry_credit, marks


def _run(via: str, fixture: dict, entry_credit: float, marks: list[dict]) -> bytes:
    """Run the helpers via ``via`` and return canonical JSON bytes.

    The serialised payload bundles BS prices, BS deltas, IV recovery,
    and the outcome-evaluation dict so a single SHA detects drift in
    any of the four code paths.  Float values are rounded to 9 digits
    -- enough to spot semantic regressions while tolerating the last
    couple of ULPs of FP non-determinism that scipy's normal-CDF
    sometimes emits between SciPy patch versions.
    """
    if via == "shim":
        mod = fresh_import("generate_training_data")
    elif via == "package":
        mod = fresh_import("training")
    else:
        raise ValueError(f"unknown via: {via!r}")

    bs_prices = mod.bs_price_vec(
        fixture["S"], fixture["K"], fixture["T"], fixture["r"],
        fixture["sigma"], fixture["is_call"],
    )
    bs_deltas = mod.bs_delta_vec(
        fixture["S"], fixture["K"], fixture["T"], fixture["r"],
        fixture["sigma"], fixture["is_call"],
    )
    # Recover IV from the prices we just computed; should round-trip
    # back to ``fixture['sigma']`` modulo the bisection tolerance.
    ivs = mod.implied_vol_vec(
        bs_prices, fixture["S"], fixture["K"], fixture["T"], fixture["r"],
        fixture["is_call"],
    )
    outcome = mod._evaluate_outcome(entry_credit, marks)

    payload = {
        "bs_prices": [round(float(x), 9) for x in bs_prices],
        "bs_deltas": [round(float(x), 9) for x in bs_deltas],
        "ivs": [round(float(x), 9) for x in ivs],
        "outcome": {
            k: (round(v, 9) if isinstance(v, float) else v)
            for k, v in outcome.items()
        },
    }
    return json.dumps(
        payload, sort_keys=True, default=str, ensure_ascii=False,
    ).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for the synthetic options chain (default 42).",
    )
    args = parser.parse_args(argv)

    setup_import_paths()
    print(f"Training helper regression: synthetic chain seed={args.seed}")
    fixture = _build_fixture(args.seed)
    entry_credit, marks = _build_outcome_marks(args.seed)

    shim_runs = [_run("shim", fixture, entry_credit, marks) for _ in range(2)]
    pkg_runs = [_run("package", fixture, entry_credit, marks) for _ in range(2)]
    return report_parity("generate_training_data", shim_runs, pkg_runs)


if __name__ == "__main__":
    sys.exit(main())
