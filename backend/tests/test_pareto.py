"""Tests for the shared Pareto-frontier helper (M6 in OFFLINE_PIPELINE_AUDIT.md).

Verifies that ``compute_pareto_mask`` and ``extract_pareto_frontier`` from
``_pareto.py`` are byte-identical to the legacy in-script implementations
in ``backtest_strategy.py`` and ``ingest_optimizer_results.py`` -- a
regression here would silently change DB ingest semantics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure backend/scripts is on sys.path so the helper is importable
# without depending on test ordering or CI working directory.
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _pareto import (  # noqa: E402  (sys.path manipulation above)
    compute_pareto_mask,
    extract_pareto_frontier,
)


def _make_df(points):
    """Build an optimizer results DataFrame from (sharpe, max_dd_pct) tuples."""
    return pd.DataFrame(points, columns=["sharpe", "max_dd_pct"])


class TestComputeParetoMask:
    """Boolean-mask correctness across canonical shapes."""

    def test_empty_dataframe_returns_empty_series(self):
        """Empty in => empty bool Series, no exception."""
        df = pd.DataFrame({"sharpe": [], "max_dd_pct": []})
        mask = compute_pareto_mask(df)
        assert isinstance(mask, pd.Series)
        assert len(mask) == 0
        assert mask.dtype == bool

    def test_single_row_is_always_pareto(self):
        """A single-element frontier is trivially optimal."""
        df = _make_df([(1.0, 10.0)])
        mask = compute_pareto_mask(df)
        assert mask.tolist() == [True]

    def test_strictly_dominated_row_is_excluded(self):
        """When B has both higher Sharpe and lower DD, A drops out."""
        # B (sharpe=2, dd=10) dominates A (sharpe=1, dd=20):
        # 10% drawdown is less risky than 20%.
        df = _make_df([(1.0, 20.0), (2.0, 10.0)])
        mask = compute_pareto_mask(df)
        assert mask.tolist() == [False, True]

    def test_equal_rows_are_both_pareto(self):
        """Strict dominance is required -- ties keep both points."""
        df = _make_df([(1.0, 10.0), (1.0, 10.0)])
        mask = compute_pareto_mask(df)
        # Neither row strictly dominates the other so both survive.
        assert mask.tolist() == [True, True]

    def test_classic_two_axis_frontier(self):
        """Two non-dominated points + two dominated rows."""
        # Row 0 (0.5, 5.0) is dominated by row 1 (1.0, 5.0): same DD,
        # higher Sharpe. Row 2 (1.5, 8.0) is dominated by row 1 too:
        # row 1 has lower DD (5 < 8) and higher-or-equal Sharpe? No,
        # row 1 sharpe=1.0 < 1.5, so row 1 does NOT dominate row 2.
        # Frontier: row 1 (low DD), row 2 (mid), row 3 (high Sharpe).
        df = _make_df(
            [
                (0.5, 5.0),
                (1.0, 5.0),
                (1.5, 8.0),
                (2.0, 10.0),
            ]
        )
        mask = compute_pareto_mask(df)
        assert mask.tolist() == [False, True, True, True]

    def test_mask_index_matches_input(self):
        """Mask must align to input index even with non-default index labels."""
        df = _make_df([(1.0, 20.0), (2.0, 10.0)])
        df.index = ["alpha", "beta"]
        mask = compute_pareto_mask(df)
        assert list(mask.index) == ["alpha", "beta"]

    def test_missing_column_raises_keyerror(self):
        """Missing input column should fail loudly, not silently return False."""
        df = pd.DataFrame({"sharpe": [1.0]})
        with pytest.raises(KeyError, match="max_dd_pct"):
            compute_pareto_mask(df)


class TestExtractParetoFrontier:
    """The DataFrame-returning convenience wrapper."""

    def test_returns_dataframe_sorted_by_sharpe_desc(self):
        """Frontier rows must come out best-Sharpe-first."""
        # All three points have different (sharpe, dd) pairs:
        # row a (0.5, 5.0) -- low Sharpe but lowest DD, on the frontier
        # row b (2.0, 10.0) -- high Sharpe but higher DD, on the frontier
        # row c (1.0, 8.0) -- middle Sharpe & DD, on the frontier
        df = _make_df(
            [
                (0.5, 5.0),
                (2.0, 10.0),
                (1.0, 8.0),
            ]
        )
        out = extract_pareto_frontier(df)
        assert list(out["sharpe"]) == [2.0, 1.0, 0.5]
        # Index is reset so downstream consumers can iterate by position.
        assert list(out.index) == [0, 1, 2]

    def test_dropped_row_is_excluded(self):
        """Dominated points must not appear in the output frame."""
        # Row 1 (1.0, 20.0) is dominated by row 0 (1.0, 10.0):
        # equal Sharpe but lower DD. Row 2 (2.0, 25.0) is on the
        # frontier (highest Sharpe, even though it has the worst DD).
        df = _make_df([(1.0, 10.0), (1.0, 20.0), (2.0, 25.0)])
        out = extract_pareto_frontier(df)
        dd_vals = sorted(out["max_dd_pct"].tolist())
        assert dd_vals == sorted([10.0, 25.0])


class TestParity:
    """The two legacy entry points must agree with the shared helper."""

    def test_backtest_wrapper_calls_shared_impl(self):
        """``extract_pareto_frontier`` re-export from backtest_strategy
        must produce the same rows as the shared module.
        """
        from backtest_strategy import (
            extract_pareto_frontier as backtest_extract,
        )

        rng = np.random.default_rng(seed=42)
        df = pd.DataFrame(
            {
                "sharpe": rng.uniform(-1, 3, size=50),
                "max_dd_pct": rng.uniform(1, 50, size=50),
            }
        )
        from_shared = extract_pareto_frontier(df)
        from_wrapper = backtest_extract(df)
        # Same number of rows, same Sharpe set (sort-stable).
        assert len(from_shared) == len(from_wrapper)
        pd.testing.assert_series_equal(
            from_shared["sharpe"].reset_index(drop=True),
            from_wrapper["sharpe"].reset_index(drop=True),
            check_names=False,
        )

    def test_ingest_wrapper_calls_shared_impl(self):
        """``_compute_pareto`` in ingest_optimizer_results must agree
        with the shared mask, byte-for-byte.
        """
        from ingest_optimizer_results import _compute_pareto

        rng = np.random.default_rng(seed=7)
        df = pd.DataFrame(
            {
                "sharpe": rng.uniform(-1, 3, size=30),
                "max_dd_pct": rng.uniform(1, 50, size=30),
            }
        )
        legacy = _compute_pareto(df)
        shared = compute_pareto_mask(df)
        pd.testing.assert_series_equal(legacy, shared, check_names=False)
