"""Tests for ``backend/scripts/regime_utils.py``.

Covers:

* ``classify_vix``                – VIX-level bucketing + boundary handling.
* ``classify_spx_move``           – decimal-fraction SPX return bucketing.
* ``classify_term_structure``     – inverted vs. normal threshold at 1.0.
* ``classify_day_regime``         – combined per-day tagger for one row.
* ``compute_regime_metrics``      – per-regime trades / pnl / win-rate / sharpe
  aggregation given an equity curve and a daily-signals frame.

The unit-convention asserts (Wave 3 / M2) live in the SPX section: every
threshold in this module is a decimal fraction (``-0.02`` = 2% drop), so
the tests use that scale and would catch any regression to the legacy
percent convention.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the scripts/ folder importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from regime_utils import (  # noqa: E402  -- after sys.path edit
    REGIME_DIMENSIONS,
    SPX_THRESHOLDS,
    VIX_THRESHOLDS,
    classify_day_regime,
    classify_spx_move,
    classify_term_structure,
    classify_vix,
    compute_regime_metrics,
)


# ---------------------------------------------------------------------------
# classify_vix
# ---------------------------------------------------------------------------


class TestClassifyVix:
    """The VIX classifier returns one of vix_low/medium/high/extreme.

    Threshold convention is *strictly* less-than (``vix < threshold``), so
    a VIX of exactly 15.0 falls into the next bucket (``vix_medium``).
    Tests pin both sides of every boundary to lock the convention.
    """

    def test_below_15_is_low(self):
        assert classify_vix(10.0) == "vix_low"
        assert classify_vix(14.99) == "vix_low"

    def test_15_falls_into_medium(self):
        # Boundary case: 15.0 is NOT < 15.0, so we move to the next bucket.
        assert classify_vix(15.0) == "vix_medium"

    def test_between_15_and_20_is_medium(self):
        assert classify_vix(17.5) == "vix_medium"
        assert classify_vix(19.99) == "vix_medium"

    def test_20_falls_into_high(self):
        assert classify_vix(20.0) == "vix_high"

    def test_between_20_and_25_is_high(self):
        assert classify_vix(22.5) == "vix_high"
        assert classify_vix(24.99) == "vix_high"

    def test_25_and_above_is_extreme(self):
        assert classify_vix(25.0) == "vix_extreme"
        assert classify_vix(40.0) == "vix_extreme"
        assert classify_vix(100.0) == "vix_extreme"

    def test_none_returns_unknown(self):
        assert classify_vix(None) == "vix_unknown"

    def test_nan_returns_unknown(self):
        assert classify_vix(float("nan")) == "vix_unknown"

    def test_thresholds_constant_matches_implementation(self):
        # Defensive: if someone re-orders the threshold list, the
        # classifier's behaviour changes silently.  Pin the structure.
        assert VIX_THRESHOLDS[0] == (15.0, "vix_low")
        assert VIX_THRESHOLDS[1] == (20.0, "vix_medium")
        assert VIX_THRESHOLDS[2] == (25.0, "vix_high")
        assert VIX_THRESHOLDS[3][1] == "vix_extreme"
        assert math.isinf(VIX_THRESHOLDS[3][0])


# ---------------------------------------------------------------------------
# classify_spx_move
# ---------------------------------------------------------------------------


class TestClassifySpxMove:
    """SPX classifier uses the **decimal-fraction** convention (M2).

    -0.02 => 2% drop.  The bucket boundaries are at -2%, -0.5%, +0.5%.
    A value at exactly -0.02 is NOT < -0.02 so it lands in the next
    bucket (``spx_small_drop``).  Tests pin every boundary.
    """

    def test_big_drop_below_minus_2_pct(self):
        assert classify_spx_move(-0.05) == "spx_big_drop"
        assert classify_spx_move(-0.0201) == "spx_big_drop"

    def test_minus_2_pct_falls_into_small_drop(self):
        # Boundary: -0.02 is NOT < -0.02, advances to next bucket.
        assert classify_spx_move(-0.02) == "spx_small_drop"

    def test_small_drop_between_minus_2_pct_and_minus_05_pct(self):
        assert classify_spx_move(-0.01) == "spx_small_drop"
        assert classify_spx_move(-0.0051) == "spx_small_drop"

    def test_minus_05_pct_falls_into_flat(self):
        assert classify_spx_move(-0.005) == "spx_flat"

    def test_flat_between_minus_05_pct_and_plus_05_pct(self):
        assert classify_spx_move(0.0) == "spx_flat"
        assert classify_spx_move(0.0049) == "spx_flat"
        assert classify_spx_move(-0.0049) == "spx_flat"

    def test_plus_05_pct_falls_into_rally(self):
        # Boundary: 0.005 is NOT < 0.005, advances to next bucket.
        assert classify_spx_move(0.005) == "spx_rally"

    def test_rally_above_plus_05_pct(self):
        assert classify_spx_move(0.01) == "spx_rally"
        assert classify_spx_move(0.05) == "spx_rally"

    def test_none_returns_unknown(self):
        assert classify_spx_move(None) == "spx_unknown"

    def test_nan_returns_unknown(self):
        assert classify_spx_move(float("nan")) == "spx_unknown"

    def test_decimal_convention_locked(self):
        # M2 regression guard.  If anyone ever switches back to percent
        # units (where "5" means 5% rally), passing a 2 here would be
        # interpreted as a 200% rally and bucket as ``spx_rally``.  The
        # current decimal convention says 2 = +200% which still rallies,
        # but the boundary at -0.02 would behave totally differently if
        # passed -2 (would land in ``spx_small_drop`` under decimal but
        # ``spx_big_drop`` under percent).  Lock the decimal contract.
        assert classify_spx_move(-2.0) == "spx_big_drop"  # any -big number
        assert classify_spx_move(-0.5) == "spx_big_drop"  # 50% drop
        # Most importantly: a "-1" input must NOT mean "1% drop".  Under
        # the decimal contract it's a 100% drop, which is way past the
        # -2% threshold and so still ``spx_big_drop``.
        assert classify_spx_move(-1.0) == "spx_big_drop"

    def test_thresholds_constant_matches_implementation(self):
        assert SPX_THRESHOLDS[0] == (-0.02, "spx_big_drop")
        assert SPX_THRESHOLDS[1] == (-0.005, "spx_small_drop")
        assert SPX_THRESHOLDS[2] == (0.005, "spx_flat")
        assert SPX_THRESHOLDS[3][1] == "spx_rally"


# ---------------------------------------------------------------------------
# classify_term_structure
# ---------------------------------------------------------------------------


class TestClassifyTermStructure:
    """Term structure = VIX9D / VIX.  >= 1.0 means short-term IV is
    above long-term IV (inverted/stressed)."""

    def test_below_1_is_normal(self):
        assert classify_term_structure(0.85) == "ts_normal"
        assert classify_term_structure(0.999) == "ts_normal"

    def test_exactly_1_is_inverted(self):
        # Boundary: classifier uses ``>=`` so 1.0 itself counts as
        # inverted.  This matches the live event-signal logic.
        assert classify_term_structure(1.0) == "ts_inverted"

    def test_above_1_is_inverted(self):
        assert classify_term_structure(1.01) == "ts_inverted"
        assert classify_term_structure(1.5) == "ts_inverted"

    def test_none_and_nan_are_unknown(self):
        assert classify_term_structure(None) == "ts_unknown"
        assert classify_term_structure(float("nan")) == "ts_unknown"


# ---------------------------------------------------------------------------
# classify_day_regime
# ---------------------------------------------------------------------------


class TestClassifyDayRegime:
    """Combined per-day classifier.

    The function must read three columns out of a Series row (``vix``,
    ``prev_spx_return``, ``term_structure``) and return a dict with one
    label per dimension in ``REGIME_DIMENSIONS``.
    """

    def test_returns_one_label_per_dimension(self):
        row = pd.Series({"vix": 18.0, "prev_spx_return": 0.001, "term_structure": 0.9})
        out = classify_day_regime(row)
        assert set(out.keys()) == set(REGIME_DIMENSIONS)
        assert out["vix_regime"] == "vix_medium"
        assert out["spx_regime"] == "spx_flat"
        assert out["ts_regime"] == "ts_normal"

    def test_extreme_vix_inverted_curve_big_drop(self):
        # All three dimensions push toward stressed labels.
        row = pd.Series({"vix": 35.0, "prev_spx_return": -0.05, "term_structure": 1.2})
        out = classify_day_regime(row)
        assert out == {
            "vix_regime": "vix_extreme",
            "spx_regime": "spx_big_drop",
            "ts_regime": "ts_inverted",
        }

    def test_missing_columns_yield_unknown(self):
        # Empty row -> all unknowns.  This is what happens when a day
        # is missing from the daily_signals frame.
        row = pd.Series(dtype=float)
        out = classify_day_regime(row)
        assert out["vix_regime"] == "vix_unknown"
        assert out["spx_regime"] == "spx_unknown"
        assert out["ts_regime"] == "ts_unknown"

    def test_nan_columns_yield_unknown(self):
        row = pd.Series({
            "vix": float("nan"),
            "prev_spx_return": float("nan"),
            "term_structure": float("nan"),
        })
        out = classify_day_regime(row)
        assert out["vix_regime"] == "vix_unknown"
        assert out["spx_regime"] == "spx_unknown"
        assert out["ts_regime"] == "ts_unknown"


# ---------------------------------------------------------------------------
# compute_regime_metrics
# ---------------------------------------------------------------------------


class TestComputeRegimeMetrics:
    """End-to-end check on the per-regime aggregation pipeline.

    We construct a 4-day equity curve where two days were in a high-VIX
    regime (one win, one loss) and two were in a low-VIX regime (one win,
    one no-trade).  The output dict must contain the per-regime trade
    counts, total PnL, win rate, and sharpe placeholders we expect.
    """

    @staticmethod
    def _make_curve_and_signals():
        # Days 1-2: high VIX (vix=22, > 20 < 25), neutral SPX, contango.
        # Days 3-4: low VIX (vix=12, < 15), neutral SPX, contango.
        days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        signals = pd.DataFrame(
            {
                "vix": [22.0, 22.0, 12.0, 12.0],
                "prev_spx_return": [0.0, 0.0, 0.0, 0.0],
                "term_structure": [0.85, 0.85, 0.85, 0.85],
            },
            index=days,
        )
        curve = pd.DataFrame(
            {
                "day": days,
                "equity": [100_000.0, 100_500.0, 100_300.0, 100_600.0],
                "daily_pnl": [500.0, -200.0, 300.0, 0.0],
                "n_trades": [1, 1, 1, 0],
                "lots": [1, 1, 1, 0],
                "status": ["traded", "traded", "traded", "noop"],
                "event_signals": [None, None, None, None],
            }
        )
        return curve, signals

    def test_returns_empty_dict_when_curve_empty(self):
        # Defensive: the helper must not throw on an empty backtest.
        out = compute_regime_metrics(pd.DataFrame(), pd.DataFrame())
        assert out == {}

    def test_per_vix_regime_trades_pnl_and_winrate(self):
        curve, signals = self._make_curve_and_signals()
        out = compute_regime_metrics(curve, signals)

        # ``vix_high`` regime (days 1-2): 2 trade-days, +500 + (-200) =
        # +300 pnl, 1 win out of 2 -> win_rate 0.5.
        assert out["vix_high_trades"] == 2
        assert out["vix_high_pnl"] == 300.0
        assert out["vix_high_win_rate"] == 0.5

        # ``vix_low`` regime (days 3-4): only day 3 traded (n_trades=1
        # on day 3, n_trades=0 on day 4 means no trade-day).  But
        # ``daily_pnl`` is summed over the full slice (includes the
        # 0.0 from day 4), so pnl == 300.0.  Win rate uses *trade-days*
        # only (1 trade-day, 1 win) -> 1.0.
        assert out["vix_low_trades"] == 1
        assert out["vix_low_pnl"] == 300.0
        assert out["vix_low_win_rate"] == 1.0

    def test_per_spx_and_ts_regimes_emit_keys(self):
        curve, signals = self._make_curve_and_signals()
        out = compute_regime_metrics(curve, signals)

        # All four days are flat (prev_spx_return == 0.0), so we expect
        # exactly one ``spx_flat`` group with non-zero counts.
        assert "spx_flat_trades" in out
        assert out["spx_flat_trades"] == 3  # 3 trade-days

        # All four days have ts_normal (0.85 < 1.0).
        assert "ts_normal_trades" in out
        assert out["ts_normal_trades"] == 3

    def test_unknown_regime_when_day_missing_from_signals(self):
        # If a day in the curve is not in the signals index, the helper
        # tags it as 'unknown' for every dimension.  We then expect
        # ``<prefix>_unknown_*`` keys to appear.
        curve, signals = self._make_curve_and_signals()
        # Drop day 4 from signals so it falls through to unknown.
        signals = signals.drop(signals.index[-1])
        out = compute_regime_metrics(curve, signals)

        # The legacy splitting in the helper produces keys like
        # ``vix_unknown_trades`` -- it splits the regime label on the
        # first underscore and uses the second half as the suffix.
        # Just check that something with ``unknown`` showed up.
        unknown_keys = [k for k in out if "unknown" in k]
        assert unknown_keys, f"Expected 'unknown' regime keys, got {sorted(out)}"

    def test_sharpe_zero_when_only_one_pnl_point(self):
        # Sharpe needs len > 1 and std > 0 to be non-zero.
        days = pd.to_datetime(["2024-01-02"])
        signals = pd.DataFrame(
            {"vix": [22.0], "prev_spx_return": [0.0], "term_structure": [0.85]},
            index=days,
        )
        curve = pd.DataFrame(
            {
                "day": days,
                "equity": [100_500.0],
                "daily_pnl": [500.0],
                "n_trades": [1],
                "lots": [1],
                "status": ["traded"],
                "event_signals": [None],
            }
        )
        out = compute_regime_metrics(curve, signals)
        assert out["vix_high_sharpe"] == 0.0

    def test_sharpe_uses_annualization_factor_252(self):
        # Two daily PnL points with mean 0 and a definite std --
        # sharpe = mean/std * sqrt(252).  Use mean != 0 to get a
        # non-zero sharpe.
        days = pd.to_datetime(["2024-01-02", "2024-01-03"])
        signals = pd.DataFrame(
            {"vix": [22.0, 22.0], "prev_spx_return": [0.0, 0.0], "term_structure": [0.85, 0.85]},
            index=days,
        )
        # PnL = [+100, +200]: mean = 150, std (sample) = ~70.71
        curve = pd.DataFrame(
            {
                "day": days,
                "equity": [100_100.0, 100_300.0],
                "daily_pnl": [100.0, 200.0],
                "n_trades": [1, 1],
                "lots": [1, 1],
                "status": ["traded", "traded"],
                "event_signals": [None, None],
            }
        )
        out = compute_regime_metrics(curve, signals)
        # Expected: 150/70.7107 * sqrt(252) ≈ 33.67
        expected = (150.0 / np.std([100.0, 200.0], ddof=1)) * np.sqrt(252)
        assert out["vix_high_sharpe"] == round(expected, 2)
