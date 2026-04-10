"""Tests for sl_recovery_analysis.py — compute_trade_pnl and _fallback_pnl."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sl_recovery_analysis import compute_trade_pnl, _fallback_pnl, _isnan
from _constants import CONTRACT_MULT, CONTRACTS


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
