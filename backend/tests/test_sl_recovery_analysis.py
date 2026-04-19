"""Tests for sl_recovery_analysis.py.

Covers:

* ``compute_trade_pnl``  -- per-trade PnL given (SL level, exit rule).
* ``_fallback_pnl``      -- legacy fallback when trajectory cols missing.
* ``evaluate_strategy``  -- aggregate metrics (total/avg PnL, sharpe, …).
* ``slice_by_dimension`` -- per-bucket slicing of evaluate_strategy.
* ``recovery_analysis``  -- prints SL-recovery breakdowns; we assert it
  doesn't blow up on representative inputs.
* ``main``               -- CLI smoke test: build a tiny enhanced CSV,
  point ``--csv`` at it, and confirm the script runs end-to-end without
  raising.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sl_recovery_analysis as sra  # noqa: E402  -- after sys.path edit
from sl_recovery_analysis import (  # noqa: E402
    _fallback_pnl,
    _isnan,
    compute_trade_pnl,
    evaluate_strategy,
    main,
    recovery_analysis,
    slice_by_dimension,
)
from _constants import CONTRACT_MULT, CONTRACTS  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────


def _make_row(**overrides) -> pd.Series:
    """Build a minimal candidate row with trajectory columns."""
    defaults = {
        "resolved": True,
        "entry_credit": 1.0,
        "realized_pnl": 50.0,
        "exit_reason": "TAKE_PROFIT_50",
        "min_pnl_before_tp50": -20.0,
        "first_tp50_pnl": 50.0,
        "final_pnl_at_expiry": 100.0,
        "hold_realized_pnl": 80.0,
        "recovered_after_sl": False,
        "hold_hit_tp50": True,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


MAX_PROFIT = 1.0 * CONTRACT_MULT * CONTRACTS


# ── compute_trade_pnl ────────────────────────────────────────────


class TestComputeTradePnl:
    def test_unresolved_returns_none(self):
        """Unresolved rows return None."""
        row = _make_row(resolved=False)
        assert compute_trade_pnl(row, sl_mult=2.0, exit_rule="tp50") is None

    def test_zero_credit_returns_none(self):
        """Zero entry_credit returns None."""
        row = _make_row(entry_credit=0)
        assert compute_trade_pnl(row, sl_mult=2.0, exit_rule="tp50") is None

    def test_tp50_fires_before_sl(self):
        """When SL doesn't fire before TP50, TP50 PnL is returned."""
        row = _make_row(min_pnl_before_tp50=-10.0, first_tp50_pnl=50.0)
        result = compute_trade_pnl(row, sl_mult=2.0, exit_rule="tp50")
        assert result == 50.0

    def test_sl_fires_before_tp50(self):
        """When SL fires before TP50, SL threshold is returned as loss."""
        sl_threshold = MAX_PROFIT * 1.5
        row = _make_row(min_pnl_before_tp50=-sl_threshold - 1)
        result = compute_trade_pnl(row, sl_mult=1.5, exit_rule="tp50")
        assert result == pytest.approx(-sl_threshold)

    def test_no_sl(self):
        """sl_mult=None means no stop-loss; trade follows normal exit."""
        row = _make_row(min_pnl_before_tp50=-9999)
        result = compute_trade_pnl(row, sl_mult=None, exit_rule="tp50")
        assert result == 50.0

    def test_expiry_exit_rule_uses_final_pnl(self):
        """Expiry exit rule returns final_pnl when TP50 fires first."""
        row = _make_row(first_tp50_pnl=50.0, final_pnl_at_expiry=90.0,
                        min_pnl_before_tp50=-5.0)
        result = compute_trade_pnl(row, sl_mult=2.0, exit_rule="expiry")
        assert result == 90.0


# ── _fallback_pnl ────────────────────────────────────────────────


class TestFallbackPnl:
    def test_tp50_trade(self):
        """TP50 exit uses original realized_pnl."""
        row = _make_row(exit_reason="TAKE_PROFIT_50", realized_pnl=45.0,
                        min_pnl_before_tp50=None)
        result = _fallback_pnl(row, sl_mult=2.0, exit_rule="tp50", max_profit=MAX_PROFIT)
        assert result == 45.0

    def test_stop_loss_with_sl_mult(self):
        """STOP_LOSS with sl_mult returns -sl_threshold counterfactual."""
        row = _make_row(exit_reason="STOP_LOSS", realized_pnl=-80.0,
                        min_pnl_before_tp50=None)
        result = _fallback_pnl(row, sl_mult=2.0, exit_rule="tp50", max_profit=MAX_PROFIT)
        assert result == pytest.approx(-MAX_PROFIT * 2.0)

    def test_stop_loss_without_sl_mult(self):
        """STOP_LOSS with sl_mult=None uses hold_realized_pnl if available."""
        row = _make_row(exit_reason="STOP_LOSS", realized_pnl=-80.0,
                        hold_realized_pnl=20.0, min_pnl_before_tp50=None)
        result = _fallback_pnl(row, sl_mult=None, exit_rule="tp50", max_profit=MAX_PROFIT)
        assert result == 20.0

    def test_stop_loss_without_sl_mult_no_hold(self):
        """STOP_LOSS with sl_mult=None and no hold_realized_pnl falls back to orig_pnl."""
        row = _make_row(exit_reason="STOP_LOSS", realized_pnl=-80.0,
                        hold_realized_pnl=None, min_pnl_before_tp50=None)
        result = _fallback_pnl(row, sl_mult=None, exit_rule="tp50", max_profit=MAX_PROFIT)
        assert result == -80.0

    def test_other_exit_reason(self):
        """Unknown exit reason returns realized_pnl."""
        row = _make_row(exit_reason="OTHER", realized_pnl=10.0,
                        min_pnl_before_tp50=None)
        result = _fallback_pnl(row, sl_mult=2.0, exit_rule="tp50", max_profit=MAX_PROFIT)
        assert result == 10.0


# ── evaluate_strategy ─────────────────────────────────────────────


def _make_enhanced_df(n_rows: int = 20, *, with_sl_trade: bool = True) -> pd.DataFrame:
    """Build a small enhanced trades DataFrame with trajectory columns.

    The frame contains a mix of TP50 wins and (optionally) one
    STOP_LOSS row so the recovery + sweep paths both exercise data.
    """
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_rows):
        # Half winners, half non-winners so win_rate is well-defined.
        is_win = i % 2 == 0
        rows.append({
            "resolved": True,
            "entry_credit": 1.0,
            "realized_pnl": 50.0 if is_win else -10.0,
            "exit_reason": "TAKE_PROFIT_50" if is_win else "EXPIRY",
            "min_pnl_before_tp50": -float(rng.uniform(0, 30)),
            "first_tp50_pnl": 50.0 if is_win else None,
            "final_pnl_at_expiry": 100.0 if is_win else -50.0,
            "hold_realized_pnl": 80.0 if is_win else -20.0,
            "recovered_after_sl": False,
            "hold_hit_tp50": is_win,
            # Slicing dimensions used by recovery_analysis / main:
            "vix": 18.0 + i * 0.5,
            "dte_target": 5 if i % 3 == 0 else 7,
            "spread_side": "PUT" if i % 2 == 0 else "CALL",
            "delta_target": 0.20 if i % 2 == 0 else 0.30,
            "entry_dt": pd.Timestamp("2024-01-02 14:30", tz="UTC")
                + pd.Timedelta(days=i),
            "offline_gex_net": float(rng.uniform(-1e9, 1e9)),
        })
    df = pd.DataFrame(rows)
    if with_sl_trade:
        # One canonical SL row so recovery_analysis has data to print.
        sl_row = {
            "resolved": True,
            "entry_credit": 1.0,
            "realized_pnl": -150.0,
            "exit_reason": "STOP_LOSS",
            "min_pnl_before_tp50": -200.0,
            "first_tp50_pnl": None,
            "final_pnl_at_expiry": -10.0,
            "hold_realized_pnl": 30.0,
            "recovered_after_sl": True,
            "hold_hit_tp50": True,
            "vix": 25.0,
            "dte_target": 7,
            "spread_side": "PUT",
            "delta_target": 0.20,
            "entry_dt": pd.Timestamp("2024-01-15 14:30", tz="UTC"),
            "offline_gex_net": 0.0,
        }
        df = pd.concat([df, pd.DataFrame([sl_row])], ignore_index=True)
    return df


class TestEvaluateStrategy:
    """``evaluate_strategy`` aggregates per-trade PnLs into one dict."""

    def test_empty_dataframe_returns_zero_trades(self):
        # Defensive: a frame whose rows are all unresolved must report
        # n_trades=0 (and not crash trying to mean() an empty array).
        df = _make_enhanced_df(2)
        df["resolved"] = False
        out = evaluate_strategy(df, sl_mult=2.0, exit_rule="tp50")
        assert out["n_trades"] == 0
        # Sweep code keys off n_trades; ensure no follow-on stats are
        # present so a downstream consumer can't accidentally read 0.
        assert "total_pnl" not in out

    def test_basic_metrics_present(self):
        df = _make_enhanced_df(10, with_sl_trade=False)
        out = evaluate_strategy(df, sl_mult=None, exit_rule="tp50")
        # Required fields the leaderboard relies on.
        for key in (
            "strategy", "sl_mult", "exit_rule", "n_trades", "total_pnl",
            "avg_pnl", "win_rate", "sharpe", "max_drawdown", "tail_5_pct",
        ):
            assert key in out, f"missing key {key} in result"
        assert out["n_trades"] == 10
        # win_rate must lie in [0, 1] regardless of params.
        assert 0.0 <= out["win_rate"] <= 1.0

    def test_strategy_label_format(self):
        # The leaderboard groups by ``strategy``; the format string is
        # consumer-facing and must not regress.
        out = evaluate_strategy(_make_enhanced_df(4), sl_mult=2.5, exit_rule="tp50")
        assert out["strategy"] == "SL=2.5x | exit=tp50"

        out_no_sl = evaluate_strategy(_make_enhanced_df(4), sl_mult=None, exit_rule="expiry")
        assert out_no_sl["strategy"] == "SL=none | exit=expiry"

    def test_sharpe_zero_when_std_zero(self):
        # All wins -> std=0 -> sharpe must be 0 (not divide-by-zero).
        df = _make_enhanced_df(4, with_sl_trade=False)
        df["realized_pnl"] = 50.0
        # Force every row to take the TP50 branch with the same PnL.
        df["min_pnl_before_tp50"] = -1.0
        df["first_tp50_pnl"] = 50.0
        out = evaluate_strategy(df, sl_mult=2.0, exit_rule="tp50")
        assert out["sharpe"] == 0.0

    def test_max_drawdown_nonpositive(self):
        # Max drawdown is reported as the most-negative point on the
        # cumulative-PnL curve, so it must always be <= 0.
        out = evaluate_strategy(_make_enhanced_df(20), sl_mult=2.0, exit_rule="tp50")
        assert out["max_drawdown"] <= 0.0


# ── slice_by_dimension ────────────────────────────────────────────


class TestSliceByDimension:
    """``slice_by_dimension`` runs ``evaluate_strategy`` per bucket."""

    def test_slices_by_unique_values_when_no_buckets(self):
        df = _make_enhanced_df(20, with_sl_trade=False)
        out = slice_by_dimension(df, sl_mult=2.0, exit_rule="tp50",
                                 dim_col="dte_target")
        # The fixture uses dte_target ∈ {5, 7}; expect both slices.
        slices = sorted({r["slice"] for r in out})
        assert slices == [5, 7]
        # Each result should carry its dimension label for downstream
        # joining.
        assert all(r["dimension"] == "dte_target" for r in out)

    def test_slices_by_explicit_buckets(self):
        df = _make_enhanced_df(10, with_sl_trade=False)
        # Hand-rolled boolean masks (mimics build_vix_buckets / GEX).
        buckets = {
            "low": df["vix"] < 21.0,
            "high": df["vix"] >= 21.0,
        }
        out = slice_by_dimension(df, sl_mult=None, exit_rule="tp50",
                                 dim_col="vix", buckets=buckets)
        names = sorted(r["slice"] for r in out)
        assert names == ["high", "low"]

    def test_skips_empty_buckets(self):
        df = _make_enhanced_df(4, with_sl_trade=False)
        # ``impossible`` mask has no rows; helper must skip it without
        # emitting an empty result row (would dilute the leaderboard).
        buckets = {
            "all": pd.Series([True] * len(df)),
            "impossible": pd.Series([False] * len(df)),
        }
        out = slice_by_dimension(df, sl_mult=None, exit_rule="tp50",
                                 dim_col="vix", buckets=buckets)
        assert len(out) == 1
        assert out[0]["slice"] == "all"


# ── recovery_analysis ──────────────────────────────────────────────


class TestRecoveryAnalysis:
    """Smoke-test the print-only ``recovery_analysis`` helper.

    The function dumps multiple breakdowns to stdout; we just assert it
    runs without raising and writes some recognizable text.  This is the
    cheapest insurance against e.g. a column-rename breaking the script.
    """

    def test_runs_with_representative_data(self):
        df = _make_enhanced_df(8, with_sl_trade=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            recovery_analysis(df)
        out = buf.getvalue()
        assert "STOP-LOSS RECOVERY ANALYSIS" in out
        # The SL-row fixture has dte=7 / side=PUT / delta=0.20 — make
        # sure the per-dimension breakdowns at least mention each.
        assert "Recovery by DTE" in out
        assert "Recovery by Side" in out
        assert "Recovery by Delta" in out

    def test_prints_skip_message_when_no_sl_rows(self):
        df = _make_enhanced_df(4, with_sl_trade=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            recovery_analysis(df)
        out = buf.getvalue()
        assert "No STOP_LOSS trades found." in out


# ── main (CLI smoke test) ──────────────────────────────────────────


class TestMainSmoke:
    """End-to-end CLI smoke test for ``sl_recovery_analysis.main``.

    We write a tiny enhanced CSV, monkey-patch ``sys.argv`` to point
    ``--csv`` at it, and run ``main()`` capturing stdout.  The intent
    is to catch refactor bugs (missing columns, broken imports, etc.)
    long before they hit the operator running the script by hand.
    """

    def test_runs_end_to_end_with_minimal_csv(self, tmp_path, monkeypatch):
        df = _make_enhanced_df(10, with_sl_trade=True)
        csv_path = tmp_path / "tiny_enhanced.csv"
        df.to_csv(csv_path, index=False)

        # Patch argv so argparse picks up our temp CSV.
        monkeypatch.setattr(sys, "argv", [
            "sl_recovery_analysis.py",
            "--csv", str(csv_path),
        ])

        buf = io.StringIO()
        with redirect_stdout(buf):
            main()
        out = buf.getvalue()

        # Headers we expect in any successful run.
        assert "Loaded" in out
        assert "STRATEGY SWEEP" in out
        assert "DONE" in out
        # The recovery section runs because the CSV has trajectory
        # columns; it requires at least one STOP_LOSS row.
        assert "STOP-LOSS RECOVERY ANALYSIS" in out

    def test_runs_without_trajectory_columns(self, tmp_path, monkeypatch):
        # Backward-compat path: if the CSV lacks trajectory columns,
        # main() must still complete via the _fallback_pnl branch.
        df = _make_enhanced_df(6, with_sl_trade=False)
        df = df.drop(columns=["hold_realized_pnl"])
        # The script keys ``has_trajectory`` off the presence of
        # ``hold_realized_pnl``; recovery + dimensional sections will
        # be skipped in this branch.
        csv_path = tmp_path / "legacy.csv"
        df.to_csv(csv_path, index=False)

        monkeypatch.setattr(sys, "argv", [
            "sl_recovery_analysis.py",
            "--csv", str(csv_path),
        ])

        buf = io.StringIO()
        with redirect_stdout(buf):
            main()
        out = buf.getvalue()
        assert "Loaded" in out
        assert "STRATEGY SWEEP" in out
        # Recovery is gated on has_trajectory, so it must be silent.
        assert "STOP-LOSS RECOVERY ANALYSIS" not in out
