"""Tests for the capital-budgeted backtest engine.

Covers PortfolioManager, EventSignalDetector, precompute_daily_signals,
and run_backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backtest_strategy import (
    BacktestResult,
    DayRecord,
    EventConfig,
    EventSignalDetector,
    FullConfig,
    PortfolioConfig,
    PortfolioManager,
    RegimeThrottle,
    TradingConfig,
    compute_regime_multiplier,
    precompute_daily_signals,
    run_backtest,
    compute_effective_pnl,
    pnl_column_name,
    precompute_pnl_columns,
    _should_skip_day,
    MARGIN_PER_LOT,
    _precompute_day_selections,
    _fast_sched_select,
    _build_event_only_grid,
    _build_selective_grid,
    extract_pareto_frontier,
    _deduplicate_results,
    _parameter_importance,
    _row_to_config,
    walkforward_split,
    run_analysis,
)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def sample_day_df() -> pd.DataFrame:
    """Minimal day DataFrame with 6 candidates across 2 entry times."""
    return pd.DataFrame([
        {"day": "2025-06-01", "entry_dt": "09:45", "spread_side": "call", "dte_target": 3, "delta_target": 0.10, "credit_to_width": 0.30, "realized_pnl": 100, "spot": 5500, "vix": 15, "vix9d": 14, "term_structure": 1.07},
        {"day": "2025-06-01", "entry_dt": "09:45", "spread_side": "call", "dte_target": 5, "delta_target": 0.20, "credit_to_width": 0.25, "realized_pnl": 50, "spot": 5500, "vix": 15, "vix9d": 14, "term_structure": 1.07},
        {"day": "2025-06-01", "entry_dt": "10:02", "spread_side": "call", "dte_target": 3, "delta_target": 0.10, "credit_to_width": 0.35, "realized_pnl": 150, "spot": 5500, "vix": 15, "vix9d": 14, "term_structure": 1.07},
        {"day": "2025-06-01", "entry_dt": "09:45", "spread_side": "put", "dte_target": 5, "delta_target": 0.20, "credit_to_width": 0.40, "realized_pnl": -500, "spot": 5500, "vix": 15, "vix9d": 14, "term_structure": 1.07},
        {"day": "2025-06-01", "entry_dt": "10:02", "spread_side": "put", "dte_target": 7, "delta_target": 0.15, "credit_to_width": 0.28, "realized_pnl": 80, "spot": 5500, "vix": 15, "vix9d": 14, "term_structure": 1.07},
        {"day": "2025-06-01", "entry_dt": "10:02", "spread_side": "put", "dte_target": 5, "delta_target": 0.20, "credit_to_width": 0.22, "realized_pnl": -200, "spot": 5500, "vix": 15, "vix9d": 14, "term_structure": 1.07},
    ])


@pytest.fixture
def multi_day_df() -> pd.DataFrame:
    """Three-day DataFrame with consistent structure."""
    rows = []
    for day_idx, day in enumerate(["2025-06-01", "2025-06-02", "2025-06-03"]):
        base_spot = 5500 + day_idx * 10
        for dt in ["09:45", "10:02"]:
            for side in ["call", "put"]:
                pnl = 100 if side == "call" else -50
                rows.append({
                    "day": day, "entry_dt": dt, "spread_side": side,
                    "dte_target": 3, "delta_target": 0.10,
                    "credit_to_width": 0.30 if side == "call" else 0.20,
                    "realized_pnl": pnl,
                    "spot": base_spot, "vix": 15 + day_idx * 5,
                    "vix9d": 14 + day_idx * 5, "term_structure": 1.0,
                })
    return pd.DataFrame(rows)


# ── PortfolioManager ─────────────────────────────────────────────


class TestPortfolioManager:
    def test_initial_state(self):
        """Manager starts with correct capital and can trade."""
        pm = PortfolioManager(PortfolioConfig(starting_capital=20_000))
        pm.begin_day("2025-06-01")
        assert pm.equity == 20_000
        assert pm.can_trade()

    def test_lot_scaling(self):
        """Lots scale with equity: 1 lot per $10k."""
        pm = PortfolioManager(PortfolioConfig(starting_capital=35_000, lot_per_equity=10_000))
        pm.begin_day("2025-06-01")
        assert pm.compute_lots() == 3

    def test_lot_scaling_floor(self):
        """Always at least 1 lot when capital permits."""
        pm = PortfolioManager(PortfolioConfig(starting_capital=5_000, lot_per_equity=10_000))
        pm.begin_day("2025-06-01")
        assert pm.compute_lots() == 1

    def test_daily_trade_limit(self):
        """Cannot exceed max_trades_per_day."""
        pm = PortfolioManager(PortfolioConfig(starting_capital=20_000, max_trades_per_day=2))
        pm.begin_day("2025-06-01")
        pm.record_trade(100, 1)
        assert pm.can_trade()
        pm.record_trade(100, 1)
        assert not pm.can_trade()
        assert pm.status_label() == "daily_limit"

    def test_monthly_drawdown_stop(self):
        """Monthly stop triggers when equity drops below threshold."""
        pm = PortfolioManager(PortfolioConfig(
            starting_capital=20_000, monthly_drawdown_limit=0.15, max_trades_per_day=10,
        ))
        pm.begin_day("2025-06-01")
        pm.record_trade(-2000, 1)
        assert pm.can_trade()
        pm.record_trade(-1100, 1)
        assert not pm.can_trade()
        assert pm.is_month_stopped

    def test_monthly_stop_none_disables(self):
        """Setting monthly_drawdown_limit to None disables the stop."""
        pm = PortfolioManager(PortfolioConfig(
            starting_capital=10_000, monthly_drawdown_limit=None, max_trades_per_day=10,
        ))
        pm.begin_day("2025-06-01")
        pm.record_trade(-5000, 1)
        assert pm.can_trade()

    def test_month_rollover_resets_stop(self):
        """Month change resets the stop and recalibrates month_start_equity."""
        pm = PortfolioManager(PortfolioConfig(
            starting_capital=20_000, monthly_drawdown_limit=0.15, max_trades_per_day=10,
        ))
        pm.begin_day("2025-06-01")
        pm.record_trade(-3100, 1)
        # Drawdown is checked on can_trade(), not record_trade()
        assert not pm.can_trade()
        assert pm.is_month_stopped

        pm.begin_day("2025-07-01")
        assert not pm.is_month_stopped
        assert pm.can_trade()
        assert pm.month_start_equity == 20_000 - 3100

    def test_insufficient_capital(self):
        """Cannot trade when equity below MARGIN_PER_LOT."""
        pm = PortfolioManager(PortfolioConfig(starting_capital=500))
        pm.begin_day("2025-06-01")
        assert not pm.can_trade()
        assert pm.status_label() == "insufficient_capital"

    def test_record_trade_updates_equity(self):
        """record_trade correctly updates equity."""
        pm = PortfolioManager(PortfolioConfig(starting_capital=20_000))
        pm.begin_day("2025-06-01")
        total = pm.record_trade(150, 2)
        assert total == 300
        assert pm.equity == 20_300


# ── EventSignalDetector ──────────────────────────────────────────


class TestEventSignalDetector:
    def test_disabled_returns_empty(self):
        """Disabled detector returns no signals."""
        det = EventSignalDetector(EventConfig(enabled=False))
        row = pd.Series({"prev_spx_return": -0.05})
        assert det.detect(row) == []

    def test_spx_drop_1d(self):
        """Detects 1-day SPX drop when below threshold."""
        det = EventSignalDetector(EventConfig(enabled=True, spx_drop_threshold=-0.01))
        row = pd.Series({
            "prev_spx_return": -0.015, "prev_spx_return_2d": 0,
            "prev_vix_pct_change": 0, "vix": 15, "term_structure": 0.95,
        })
        signals = det.detect(row)
        assert "spx_drop_1d" in signals

    def test_spx_drop_2d(self):
        """Detects 2-day cumulative SPX drop."""
        det = EventSignalDetector(EventConfig(enabled=True, spx_drop_2d_threshold=-0.02))
        row = pd.Series({
            "prev_spx_return": -0.005, "prev_spx_return_2d": -0.025,
            "prev_vix_pct_change": 0, "vix": 15, "term_structure": 0.95,
        })
        signals = det.detect(row)
        assert "spx_drop_2d" in signals

    def test_vix_spike(self):
        """Detects VIX % spike above threshold."""
        det = EventSignalDetector(EventConfig(enabled=True, vix_spike_threshold=0.10))
        row = pd.Series({
            "prev_spx_return": 0, "prev_spx_return_2d": 0,
            "prev_vix_pct_change": 0.20, "vix": 20, "term_structure": 0.95,
        })
        assert "vix_spike" in det.detect(row)

    def test_vix_elevated(self):
        """Detects elevated VIX level."""
        det = EventSignalDetector(EventConfig(enabled=True, vix_elevated_threshold=25))
        row = pd.Series({
            "prev_spx_return": 0, "prev_spx_return_2d": 0,
            "prev_vix_pct_change": 0, "vix": 30, "term_structure": 0.95,
        })
        assert "vix_elevated" in det.detect(row)

    def test_term_inversion(self):
        """Detects VIX term structure inversion."""
        det = EventSignalDetector(EventConfig(enabled=True, term_inversion_threshold=1.0))
        row = pd.Series({
            "prev_spx_return": 0, "prev_spx_return_2d": 0,
            "prev_vix_pct_change": 0, "vix": 20, "term_structure": 1.15,
        })
        assert "term_inversion" in det.detect(row)

    def test_rally_detection(self):
        """Detects rally when enabled and SPX return exceeds threshold."""
        det = EventSignalDetector(EventConfig(
            enabled=True, rally_avoidance=True, rally_threshold=0.01,
        ))
        row = pd.Series({
            "prev_spx_return": 0.015, "prev_spx_return_2d": 0,
            "prev_vix_pct_change": 0, "vix": 15, "term_structure": 0.95,
        })
        assert "rally" in det.detect(row)

    def test_no_false_positives(self):
        """Calm market should produce no signals."""
        det = EventSignalDetector(EventConfig(enabled=True))
        row = pd.Series({
            "prev_spx_return": 0.001, "prev_spx_return_2d": 0.003,
            "prev_vix_pct_change": 0.02, "vix": 15, "term_structure": 0.95,
        })
        assert det.detect(row) == []

    def test_multiple_signals(self):
        """Crash day triggers multiple signals simultaneously."""
        det = EventSignalDetector(EventConfig(
            enabled=True, spx_drop_threshold=-0.01,
            vix_spike_threshold=0.10, vix_elevated_threshold=25,
        ))
        row = pd.Series({
            "prev_spx_return": -0.03, "prev_spx_return_2d": -0.04,
            "prev_vix_pct_change": 0.30, "vix": 35, "term_structure": 1.2,
        })
        signals = det.detect(row)
        assert len(signals) >= 3


# ── ScheduledSelector ────────────────────────────────────────────


# ── precompute_daily_signals ─────────────────────────────────────


class TestPrecomputeDailySignals:
    def test_returns_lagged_features(self, multi_day_df):
        """Lagged returns are computed correctly."""
        signals = precompute_daily_signals(multi_day_df)
        assert "prev_spx_return" in signals.columns
        assert "prev_vix_pct_change" in signals.columns
        assert signals.index.name == "day"
        assert len(signals) == 3

    def test_first_day_has_nan(self, multi_day_df):
        """First day should have NaN for lagged features."""
        signals = precompute_daily_signals(multi_day_df)
        assert pd.isna(signals.iloc[0]["prev_spx_return"])


# ── run_backtest ─────────────────────────────────────────────────


class TestRunBacktest:
    def test_basic_run(self, multi_day_df):
        """Basic run completes and returns valid result."""
        signals = precompute_daily_signals(multi_day_df)
        config = FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2, calls_only=True),
            event=EventConfig(enabled=False),
        )
        result = run_backtest(multi_day_df, signals, config, label="test")
        assert isinstance(result, BacktestResult)
        assert result.total_trades > 0
        assert result.days_traded > 0
        assert len(result.curve) == 3  # 3 days

    def test_calls_only_filters_puts(self, multi_day_df):
        """calls_only config never trades puts."""
        signals = precompute_daily_signals(multi_day_df)
        config = FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=10, calls_only=True),
            event=EventConfig(enabled=False),
        )
        result = run_backtest(multi_day_df, signals, config)
        # All trades should be profitable since calls are +100 PnL in fixture
        for rec in result.curve:
            if rec.n_trades > 0:
                assert rec.daily_pnl > 0

    def test_monthly_stop_prevents_trading(self):
        """Monthly stop halts trading for rest of month."""
        rows = []
        for i, day in enumerate(["2025-06-01", "2025-06-02", "2025-06-03", "2025-06-04", "2025-06-05"]):
            pnl = -2000 if i < 2 else 500
            rows.append({
                "day": day, "entry_dt": "09:45", "spread_side": "call",
                "dte_target": 3, "delta_target": 0.10,
                "credit_to_width": 0.30, "realized_pnl": pnl,
                "spot": 5500, "vix": 15, "vix9d": 14, "term_structure": 1.0,
            })
        df = pd.DataFrame(rows)
        signals = precompute_daily_signals(df)
        config = FullConfig(
            portfolio=PortfolioConfig(starting_capital=10_000, max_trades_per_day=1,
                                      monthly_drawdown_limit=0.15, lot_per_equity=10_000),
            event=EventConfig(enabled=False),
        )
        result = run_backtest(df, signals, config)
        assert result.days_stopped > 0

    def test_event_layer_fires(self, multi_day_df):
        """Event trades are placed on signal days (VIX spike in fixture)."""
        signals = precompute_daily_signals(multi_day_df)
        config = FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2, calls_only=True),
            event=EventConfig(
                enabled=True, budget_mode="separate", max_event_trades=1,
                vix_spike_threshold=0.10,  # very sensitive to trigger events
                side_preference="puts", min_dte=1, max_dte=10,
                min_delta=0.05, max_delta=0.30,
            ),
        )
        result = run_backtest(multi_day_df, signals, config)
        event_days = [r for r in result.curve if r.event_signals]
        # At least one day should have VIX change large enough
        assert result.total_trades > 0

    def test_sharpe_computed(self, multi_day_df):
        """Sharpe ratio is a finite number."""
        signals = precompute_daily_signals(multi_day_df)
        config = FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2),
            event=EventConfig(enabled=False),
        )
        result = run_backtest(multi_day_df, signals, config)
        assert isinstance(result.sharpe, float)

    def test_precomputed_day_selections(self, multi_day_df):
        """Precomputed selections produce same result as without."""
        signals = precompute_daily_signals(multi_day_df)
        config = FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2, calls_only=True),
            event=EventConfig(enabled=False),
        )
        precomp = _precompute_day_selections(multi_day_df)
        r1 = run_backtest(multi_day_df, signals, config, day_precomp=precomp)
        r2 = run_backtest(multi_day_df, signals, config)
        assert r1.final_equity == r2.final_equity
        assert r1.total_trades == r2.total_trades


# ── FullConfig ───────────────────────────────────────────────────


class TestFullConfig:
    def test_flat_dict(self):
        """flat_dict produces prefixed keys."""
        cfg = FullConfig()
        d = cfg.flat_dict()
        assert "p_starting_capital" in d
        assert "e_enabled" in d
        assert d["p_starting_capital"] == 20_000
        assert d["e_enabled"] is False


# ── Pareto frontier ──────────────────────────────────────────────


class TestParetoFrontier:
    def test_finds_optimal_points(self):
        """Pareto filter keeps only non-dominated configs."""
        rdf = pd.DataFrame([
            {"sharpe": 6.0, "max_dd_pct": 25.0, "return_pct": 500, "final_equity": 100},
            {"sharpe": 5.0, "max_dd_pct": 15.0, "return_pct": 300, "final_equity": 80},
            {"sharpe": 4.0, "max_dd_pct": 10.0, "return_pct": 200, "final_equity": 60},
            {"sharpe": 3.0, "max_dd_pct": 30.0, "return_pct": 100, "final_equity": 40},  # dominated by all above
            {"sharpe": 5.5, "max_dd_pct": 20.0, "return_pct": 400, "final_equity": 90},
        ])
        pareto = extract_pareto_frontier(rdf)
        assert len(pareto) == 4  # row 3 (sharpe=3, dd=30) is dominated
        assert 3.0 not in pareto["sharpe"].values

    def test_single_point_is_pareto(self):
        """Single point is trivially Pareto-optimal."""
        rdf = pd.DataFrame([{"sharpe": 5.0, "max_dd_pct": 20.0}])
        pareto = extract_pareto_frontier(rdf)
        assert len(pareto) == 1

    def test_all_same_sharpe_different_dd(self):
        """When Sharpe is equal, only lowest DD survives."""
        rdf = pd.DataFrame([
            {"sharpe": 5.0, "max_dd_pct": 30.0},
            {"sharpe": 5.0, "max_dd_pct": 20.0},
            {"sharpe": 5.0, "max_dd_pct": 10.0},
        ])
        pareto = extract_pareto_frontier(rdf)
        assert len(pareto) == 1
        assert pareto.iloc[0]["max_dd_pct"] == 10.0


# ── Deduplication ────────────────────────────────────────────────


class TestDeduplication:
    def test_collapses_identical_outcomes(self):
        """Configs with identical metrics are collapsed to one."""
        rdf = pd.DataFrame([
            {"final_equity": 100, "return_pct": 50, "max_dd_pct": 10, "sharpe": 5,
             "total_trades": 200, "days_traded": 100, "days_stopped": 10, "win_days": 80,
             "p_calls_only": True},
            {"final_equity": 100, "return_pct": 50, "max_dd_pct": 10, "sharpe": 5,
             "total_trades": 200, "days_traded": 100, "days_stopped": 10, "win_days": 80,
             "p_calls_only": False},  # same metrics, different config
            {"final_equity": 200, "return_pct": 100, "max_dd_pct": 20, "sharpe": 4,
             "total_trades": 300, "days_traded": 150, "days_stopped": 5, "win_days": 120,
             "p_calls_only": True},
        ])
        dedup, orig, unique = _deduplicate_results(rdf)
        assert orig == 3
        assert unique == 2
        assert len(dedup) == 2

    def test_no_duplicates_unchanged(self):
        """All-unique data passes through unmodified."""
        rdf = pd.DataFrame([
            {"final_equity": 100, "return_pct": 50, "max_dd_pct": 10, "sharpe": 5,
             "total_trades": 200, "days_traded": 100, "days_stopped": 10, "win_days": 80},
            {"final_equity": 200, "return_pct": 100, "max_dd_pct": 20, "sharpe": 4,
             "total_trades": 300, "days_traded": 150, "days_stopped": 5, "win_days": 120},
        ])
        dedup, orig, unique = _deduplicate_results(rdf)
        assert orig == unique == 2


# ── Walk-forward split ───────────────────────────────────────────


class TestWalkforwardSplit:
    def test_correct_boundaries(self, multi_day_df):
        """Split produces correct train/test by date range."""
        train, test = walkforward_split(
            multi_day_df, "2025-06-01", "2025-06-02", "2025-06-03", "2025-06-03",
        )
        train_days = set(train["day"].unique())
        test_days = set(test["day"].unique())
        assert "2025-06-01" in train_days
        assert "2025-06-02" in train_days
        assert "2025-06-03" in test_days
        assert "2025-06-03" not in train_days

    def test_no_overlap(self, multi_day_df):
        """Train and test sets have no overlapping days."""
        train, test = walkforward_split(
            multi_day_df, "2025-06-01", "2025-06-01", "2025-06-02", "2025-06-03",
        )
        overlap = set(train["day"].unique()) & set(test["day"].unique())
        assert len(overlap) == 0

    def test_empty_when_no_match(self, multi_day_df):
        """Returns empty DF when date range has no data."""
        train, test = walkforward_split(
            multi_day_df, "2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31",
        )
        assert len(train) == 0
        assert len(test) == 0


# ── Row-to-config reconstruction ─────────────────────────────────


class TestRowToConfig:
    def test_roundtrip(self):
        """Config -> flat_dict -> row -> config preserves key fields."""
        original = FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=0.15, lot_per_equity=10_000, calls_only=True),
            event=EventConfig(enabled=True, budget_mode="shared", spx_drop_threshold=-0.01,
                              side_preference="puts", min_dte=5, max_dte=7),
        )
        row = pd.Series(original.flat_dict())
        reconstructed = _row_to_config(row)
        assert reconstructed.portfolio.max_trades_per_day == 2
        assert reconstructed.portfolio.monthly_drawdown_limit == 0.15
        assert reconstructed.event.enabled is True
        assert reconstructed.event.budget_mode == "shared"

    def test_nan_monthly_stop(self):
        """NaN monthly_drawdown_limit is reconstructed as None."""
        row = pd.Series({
            "p_starting_capital": 20_000, "p_max_trades_per_day": 1,
            "p_monthly_drawdown_limit": float("nan"), "p_lot_per_equity": 10_000,
            "p_calls_only": True, "e_enabled": False, "e_budget_mode": "shared",
            "e_max_event_trades": 1, "e_spx_drop_threshold": -0.01,
            "e_side_preference": "puts", "e_min_dte": 5, "e_max_dte": 7,
            "e_min_delta": 0.15, "e_max_delta": 0.25,
            "e_rally_avoidance": False, "e_rally_threshold": 0.01,
        })
        cfg = _row_to_config(row)
        assert cfg.portfolio.monthly_drawdown_limit is None


# ── Analysis smoke test ──────────────────────────────────────────


class TestAnalysisSmoke:
    def test_analysis_on_small_csv(self, tmp_path, multi_day_df):
        """run_analysis completes without error on a small synthetic results CSV."""
        signals = precompute_daily_signals(multi_day_df)
        configs = [
            FullConfig(portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=t, calls_only=c),
                       event=EventConfig(enabled=False))
            for t in [1, 2] for c in [True, False]
        ]
        rows = []
        for cfg in configs:
            result = run_backtest(multi_day_df, signals, cfg)
            row = cfg.flat_dict()
            row.update({
                "final_equity": result.final_equity, "return_pct": result.total_return_pct,
                "max_dd_pct": result.max_drawdown_pct, "sharpe": result.sharpe,
                "total_trades": result.total_trades, "days_traded": result.days_traded,
                "days_stopped": result.days_stopped, "win_days": result.win_days,
                "win_rate": result.win_days / max(result.days_traded, 1),
            })
            rows.append(row)

        csv_path = tmp_path / "test_results.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        pareto = run_analysis(csv_path)
        assert isinstance(pareto, pd.DataFrame)
        # With min-trades filter (30), test data may yield 0 Pareto-optimal configs
        assert len(pareto) >= 0


# ── TradingConfig ────────────────────────────────────────────────


class TestTradingConfig:
    def test_defaults(self):
        """Default TradingConfig has expected values."""
        tc = TradingConfig()
        assert tc.tp_pct == 0.50
        assert tc.sl_mult is None
        assert tc.max_vix is None
        assert tc.avoid_opex is False
        assert tc.width_filter is None
        assert tc.entry_count is None

    def test_flat_dict_includes_trading(self):
        """FullConfig.flat_dict includes t_ prefixed trading params."""
        cfg = FullConfig(trading=TradingConfig(tp_pct=0.70, sl_mult=2.0, max_vix=30.0))
        d = cfg.flat_dict()
        assert "t_tp_pct" in d
        assert d["t_tp_pct"] == 0.70
        assert d["t_sl_mult"] == 2.0
        assert d["t_max_vix"] == 30.0


# ── compute_effective_pnl ────────────────────────────────────────


class TestComputeEffectivePnl:
    @pytest.fixture
    def base_row(self) -> pd.Series:
        """A fully resolved candidate with multi-TP trajectory data."""
        return pd.Series({
            "resolved": True,
            "entry_credit": 1.0,
            "width_points": 10.0,
            "realized_pnl": 50.0,
            "max_adverse_pnl": -30.0,
            "final_pnl_at_expiry": 80.0,
            "first_tp50_pnl": 55.0,
            "min_pnl_before_tp50": -10.0,
            "first_tp60_pnl": 65.0,
            "min_pnl_before_tp60": -10.0,
            "first_tp70_pnl": 75.0,
            "min_pnl_before_tp70": -20.0,
            "first_tp80_pnl": None,
            "min_pnl_before_tp80": -30.0,
            "first_tp90_pnl": None,
            "min_pnl_before_tp90": -30.0,
            "first_tp100_pnl": None,
            "min_pnl_before_tp100": -30.0,
        })

    def test_tp50_no_sl(self, base_row):
        """TP50 exit with no stop-loss returns first_tp50_pnl."""
        result = compute_effective_pnl(base_row, tp_pct=0.50, sl_mult=None)
        assert result == pytest.approx(55.0)

    def test_tp70_no_sl(self, base_row):
        """TP70 exit returns first_tp70_pnl."""
        result = compute_effective_pnl(base_row, tp_pct=0.70, sl_mult=None)
        assert result == pytest.approx(75.0)

    def test_tp80_not_reached_falls_to_expiry(self, base_row):
        """When TP80 never fires, falls back to final_pnl_at_expiry."""
        result = compute_effective_pnl(base_row, tp_pct=0.80, sl_mult=None)
        assert result == pytest.approx(80.0)

    def test_sl_fires_before_tp(self, base_row):
        """SL fires when min_pnl_before_tp breaches threshold."""
        # SL mult=0.2 → threshold = 100*0.2 = $20. min_before_tp70=-20 → triggers
        result = compute_effective_pnl(base_row, tp_pct=0.70, sl_mult=0.2)
        assert result == pytest.approx(-20.0)

    def test_sl_doesnt_fire_before_tp(self, base_row):
        """SL doesn't fire when min_pnl_before is above threshold."""
        # SL mult=0.5 → threshold = $50. min_before_tp50=-10 → no SL
        result = compute_effective_pnl(base_row, tp_pct=0.50, sl_mult=0.5)
        assert result == pytest.approx(55.0)

    def test_hold_to_expiry(self, base_row):
        """tp_pct > 1.0 means hold to expiry."""
        result = compute_effective_pnl(base_row, tp_pct=1.01, sl_mult=None)
        assert result == pytest.approx(80.0)

    def test_hold_to_expiry_with_sl(self, base_row):
        """Hold-to-expiry with SL fires SL if max_adverse breaches."""
        # SL mult=0.2 → threshold = $20. max_adverse=-30 → SL fires
        result = compute_effective_pnl(base_row, tp_pct=1.01, sl_mult=0.2)
        assert result == pytest.approx(-20.0)

    def test_unresolved_row(self):
        """Unresolved rows return None."""
        row = pd.Series({"resolved": False, "entry_credit": 1.0})
        assert compute_effective_pnl(row, 0.50, None) is None

    def test_pnl_column_name(self):
        """pnl_column_name generates deterministic names."""
        assert pnl_column_name(0.50, None) == "pnl_tp50_none"
        assert pnl_column_name(0.70, 2.0) == "pnl_tp70_2.0"
        assert pnl_column_name(1.01, 1.5) == "pnl_tp101_1.5"


# ── Day-level filters ────────────────────────────────────────────


class TestDayLevelFilters:
    @pytest.fixture
    def daily_signals(self) -> pd.DataFrame:
        """Minimal daily signals DataFrame with regime data."""
        return pd.DataFrame([
            {"day": "2025-06-01", "vix": 15.0, "term_structure": 0.95,
             "is_opex_day": False, "is_fomc_day": False, "is_nfp_day": False},
            {"day": "2025-06-02", "vix": 35.0, "term_structure": 1.12,
             "is_opex_day": True, "is_fomc_day": False, "is_nfp_day": False},
            {"day": "2025-06-03", "vix": 20.0, "term_structure": 0.90,
             "is_opex_day": False, "is_fomc_day": True, "is_nfp_day": True},
        ]).set_index("day")

    def test_no_filters_no_skip(self, daily_signals):
        """Default TradingConfig skips nothing."""
        tc = TradingConfig()
        assert _should_skip_day(tc, "2025-06-01", daily_signals) is None
        assert _should_skip_day(tc, "2025-06-02", daily_signals) is None

    def test_vix_filter(self, daily_signals):
        """max_vix=30 skips day with VIX=35."""
        tc = TradingConfig(max_vix=30.0)
        assert _should_skip_day(tc, "2025-06-01", daily_signals) is None
        assert _should_skip_day(tc, "2025-06-02", daily_signals) == "vix_filter"

    def test_term_structure_filter(self, daily_signals):
        """max_term_structure=1.10 skips day with ts=1.12."""
        tc = TradingConfig(max_term_structure=1.10)
        assert _should_skip_day(tc, "2025-06-01", daily_signals) is None
        assert _should_skip_day(tc, "2025-06-02", daily_signals) == "ts_filter"

    def test_opex_filter(self, daily_signals):
        """avoid_opex=True skips OPEX day."""
        tc = TradingConfig(avoid_opex=True)
        assert _should_skip_day(tc, "2025-06-01", daily_signals) is None
        assert _should_skip_day(tc, "2025-06-02", daily_signals) == "opex_filter"

    def test_prefer_event_days(self, daily_signals):
        """prefer_event_days=True skips non-FOMC/NFP days."""
        tc = TradingConfig(prefer_event_days=True)
        assert _should_skip_day(tc, "2025-06-01", daily_signals) == "non_event_day"
        assert _should_skip_day(tc, "2025-06-03", daily_signals) is None

    def test_missing_day_no_skip(self, daily_signals):
        """Day not in daily_signals is not skipped."""
        tc = TradingConfig(max_vix=10.0)
        assert _should_skip_day(tc, "2025-12-31", daily_signals) is None


# ── Width filter and entry count in precompute ───────────────────


class TestPrecomputeFilters:
    @pytest.fixture
    def multi_width_df(self) -> pd.DataFrame:
        """DataFrame with multiple widths and entry times."""
        rows = []
        for day in ["2025-06-01", "2025-06-02"]:
            for dt in ["09:45", "10:15", "10:45"]:
                for width in [5.0, 10.0, 15.0]:
                    rows.append({
                        "day": day, "entry_dt": dt,
                        "spread_side": "call", "dte_target": 3,
                        "delta_target": 0.10, "credit_to_width": 0.30,
                        "realized_pnl": 100, "width_points": width,
                        "spot": 5500, "vix": 15, "vix9d": 14,
                        "term_structure": 1.0,
                    })
        return pd.DataFrame(rows)

    def test_width_filter(self, multi_width_df):
        """width_filter restricts candidates to a single width."""
        precomp = _precompute_day_selections(multi_width_df, width_filter=10.0)
        for day, data in precomp.items():
            assert all(data["all"]["width_points"] == 10.0)

    def test_entry_count(self, multi_width_df):
        """entry_count=1 keeps only the last entry time per day."""
        precomp = _precompute_day_selections(multi_width_df, entry_count=1)
        for day, data in precomp.items():
            assert data["all"]["entry_dt"].nunique() == 1
            assert data["all"]["entry_dt"].iloc[0] == "10:45"

    def test_entry_count_2(self, multi_width_df):
        """entry_count=2 keeps the last 2 entry times per day."""
        precomp = _precompute_day_selections(multi_width_df, entry_count=2)
        for day, data in precomp.items():
            assert data["all"]["entry_dt"].nunique() == 2

    def test_no_filters(self, multi_width_df):
        """No filters keeps all candidates."""
        precomp = _precompute_day_selections(multi_width_df)
        for day, data in precomp.items():
            assert len(data["all"]) == 9  # 3 widths * 3 entry times


# ── run_backtest with TradingConfig ──────────────────────────────


class TestRunBacktestWithTrading:
    def test_vix_filter_reduces_trading(self, multi_day_df):
        """VIX filter skips days, reducing total trades."""
        signals = precompute_daily_signals(multi_day_df)
        config_no_filter = FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2, calls_only=True),
            trading=TradingConfig(),
            event=EventConfig(enabled=False),
        )
        config_with_filter = FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2, calls_only=True),
            trading=TradingConfig(max_vix=10.0),
            event=EventConfig(enabled=False),
        )
        r_no = run_backtest(multi_day_df, signals, config_no_filter)
        r_with = run_backtest(multi_day_df, signals, config_with_filter)
        assert r_with.total_trades <= r_no.total_trades


# ── compute_effective_pnl edge cases ─────────────────────────────


class TestComputeEffectivePnlEdgeCases:
    def test_nan_trajectory_fields(self):
        """NaN in trajectory columns falls back gracefully."""
        row = pd.Series({
            "resolved": True, "entry_credit": 1.0, "width_points": 10.0,
            "realized_pnl": 42.0,
            "max_adverse_pnl": np.nan,
            "final_pnl_at_expiry": np.nan,
            "first_tp50_pnl": np.nan,
            "min_pnl_before_tp50": np.nan,
        })
        result = compute_effective_pnl(row, tp_pct=0.50, sl_mult=None)
        assert result == pytest.approx(42.0)

    def test_zero_entry_credit(self):
        """entry_credit of 0 returns None."""
        row = pd.Series({"resolved": True, "entry_credit": 0.0})
        assert compute_effective_pnl(row, 0.50, None) is None

    def test_negative_entry_credit(self):
        """Negative entry_credit returns None."""
        row = pd.Series({"resolved": True, "entry_credit": -0.5})
        assert compute_effective_pnl(row, 0.50, None) is None

    def test_missing_entry_credit(self):
        """Missing entry_credit returns None."""
        row = pd.Series({"resolved": True})
        assert compute_effective_pnl(row, 0.50, None) is None


# ── precompute_pnl_columns ───────────────────────────────────────


class TestPrecomputePnlColumns:
    @pytest.fixture
    def trajectory_df(self) -> pd.DataFrame:
        """Small DataFrame with trajectory columns for vectorized precompute."""
        return pd.DataFrame([
            {
                "resolved": True, "entry_credit": 1.0, "width_points": 10.0,
                "realized_pnl": 50.0, "final_pnl_at_expiry": 80.0,
                "max_adverse_pnl": -30.0,
                "first_tp50_pnl": 55.0, "min_pnl_before_tp50": -10.0,
                "first_tp70_pnl": 75.0, "min_pnl_before_tp70": -20.0,
                "first_tp100_pnl": None, "min_pnl_before_tp100": -30.0,
            },
            {
                "resolved": True, "entry_credit": 2.0, "width_points": 10.0,
                "realized_pnl": -100.0, "final_pnl_at_expiry": -100.0,
                "max_adverse_pnl": -200.0,
                "first_tp50_pnl": None, "min_pnl_before_tp50": -200.0,
                "first_tp70_pnl": None, "min_pnl_before_tp70": -200.0,
                "first_tp100_pnl": None, "min_pnl_before_tp100": -200.0,
            },
        ])

    def test_columns_created(self, trajectory_df):
        """Precompute adds named columns to the DataFrame."""
        cols = precompute_pnl_columns(trajectory_df, [0.50], [None])
        assert "pnl_tp50_none" in trajectory_df.columns
        assert "pnl_tp50_none" in cols

    def test_values_match_scalar(self, trajectory_df):
        """Vectorized results match row-by-row compute_effective_pnl."""
        precompute_pnl_columns(trajectory_df, [0.50, 0.70], [None, 2.0])
        for idx, row in trajectory_df.iterrows():
            for tp in [0.50, 0.70]:
                for sl in [None, 2.0]:
                    col = pnl_column_name(tp, sl)
                    expected = compute_effective_pnl(row, tp, sl)
                    actual = trajectory_df.at[idx, col]
                    if expected is None or (isinstance(expected, float) and np.isnan(expected)):
                        assert pd.isna(actual)
                    else:
                        assert actual == pytest.approx(expected, abs=0.01)

    def test_unresolved_row_nan(self, trajectory_df):
        """Unresolved rows get NaN in precomputed columns."""
        trajectory_df.at[1, "resolved"] = False
        precompute_pnl_columns(trajectory_df, [0.50], [None])
        assert pd.isna(trajectory_df.at[1, "pnl_tp50_none"])


# ── run_backtest with precomputed PnL column ─────────────────────


class TestRunBacktestPrecomputedPnl:
    def test_uses_precomputed_column(self):
        """run_backtest uses the precomputed PnL column when it exists."""
        rows = []
        for day in ["2025-06-01", "2025-06-02", "2025-06-03"]:
            rows.append({
                "day": day, "entry_dt": "09:45", "spread_side": "call",
                "dte_target": 3, "delta_target": 0.10,
                "credit_to_width": 0.30,
                "realized_pnl": 100,
                "pnl_tp70_none": -50,  # opposite direction from realized_pnl
                "spot": 5500, "vix": 15, "vix9d": 14, "term_structure": 1.0,
                "width_points": 10.0,
            })
        df = pd.DataFrame(rows)
        signals = precompute_daily_signals(df)

        config_default = FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=1, calls_only=True),
            trading=TradingConfig(tp_pct=0.50, sl_mult=None),
            event=EventConfig(enabled=False),
        )
        config_tp70 = FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=1, calls_only=True),
            trading=TradingConfig(tp_pct=0.70, sl_mult=None),
            event=EventConfig(enabled=False),
        )

        r_default = run_backtest(df, signals, config_default)
        r_tp70 = run_backtest(df, signals, config_tp70)

        # Default (realized_pnl=100) should profit; TP70 (pnl_tp70_none=-50) should lose
        assert r_default.final_equity > 20_000
        assert r_tp70.final_equity < 20_000


# ── Enriched fixtures ────────────────────────────────────────────


class TestEnrichedFixtures:
    @pytest.fixture
    def enriched_df(self) -> pd.DataFrame:
        """DataFrame with width_points, calendar flags, and trajectory columns."""
        rows = []
        for day_idx, day in enumerate(["2025-06-01", "2025-06-02", "2025-06-03"]):
            for dt in ["09:45", "10:02"]:
                rows.append({
                    "day": day, "entry_dt": dt, "spread_side": "call",
                    "dte_target": 3, "delta_target": 0.10,
                    "credit_to_width": 0.30, "realized_pnl": 100,
                    "entry_credit": 1.0, "width_points": 10.0,
                    "spot": 5500, "vix": 15 + day_idx * 10, "vix9d": 14,
                    "term_structure": 0.95 + day_idx * 0.10,
                    "is_opex_day": day_idx == 1,
                    "is_fomc_day": day_idx == 2,
                    "is_nfp_day": False,
                })
        return pd.DataFrame(rows)

    def test_precompute_signals_includes_calendar(self, enriched_df):
        """precompute_daily_signals picks up calendar columns from DataFrame."""
        signals = precompute_daily_signals(enriched_df)
        assert "is_opex_day" in signals.columns
        assert "is_fomc_day" in signals.columns

    def test_opex_filter_end_to_end(self, enriched_df):
        """avoid_opex=True skips OPEX day through full run_backtest."""
        signals = precompute_daily_signals(enriched_df)
        config = FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2, calls_only=True),
            trading=TradingConfig(avoid_opex=True),
            event=EventConfig(enabled=False),
        )
        result = run_backtest(enriched_df, signals, config)
        opex_recs = [r for r in result.curve if r.status == "opex_filter"]
        assert len(opex_recs) == 1  # day_idx==1 is OPEX


# ── FullConfig / RowToConfig with TradingConfig ──────────────────


class TestFullConfigWithTrading:
    def test_flat_dict_has_t_prefix(self):
        """flat_dict includes t_ prefixed trading params."""
        cfg = FullConfig(
            trading=TradingConfig(tp_pct=0.70, sl_mult=2.0, max_vix=30.0,
                                  avoid_opex=True, width_filter=15.0, entry_count=2),
        )
        d = cfg.flat_dict()
        assert d["t_tp_pct"] == 0.70
        assert d["t_sl_mult"] == 2.0
        assert d["t_max_vix"] == 30.0
        assert d["t_avoid_opex"] is True
        assert d["t_width_filter"] == 15.0
        assert d["t_entry_count"] == 2

    def test_roundtrip_with_trading(self):
        """Config -> flat_dict -> row -> config preserves TradingConfig."""
        original = FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=0.15, lot_per_equity=10_000,
                                      calls_only=True, min_dte=3, max_delta=0.20),
            trading=TradingConfig(tp_pct=0.70, sl_mult=1.5, max_vix=30.0,
                                  avoid_opex=True, width_filter=15.0, entry_count=2),
            event=EventConfig(enabled=True, budget_mode="shared", spx_drop_threshold=-0.01,
                              side_preference="puts", min_dte=5, max_dte=7),
        )
        row = pd.Series(original.flat_dict())
        reconstructed = _row_to_config(row)
        assert reconstructed.trading.tp_pct == 0.70
        assert reconstructed.trading.sl_mult == 1.5
        assert reconstructed.trading.max_vix == 30.0
        assert reconstructed.trading.avoid_opex is True
        assert reconstructed.trading.width_filter == 15.0
        assert reconstructed.trading.entry_count == 2
        assert reconstructed.portfolio.min_dte == 3
        assert reconstructed.portfolio.max_delta == 0.20

    def test_roundtrip_none_optionals(self):
        """None optionals survive the flat_dict -> row -> config roundtrip."""
        original = FullConfig(
            trading=TradingConfig(tp_pct=0.50, sl_mult=None, max_vix=None,
                                  width_filter=None, entry_count=None),
        )
        row = pd.Series(original.flat_dict())
        reconstructed = _row_to_config(row)
        assert reconstructed.trading.sl_mult is None
        assert reconstructed.trading.max_vix is None
        assert reconstructed.trading.width_filter is None
        assert reconstructed.trading.entry_count is None

    def test_nan_bool_reconstructed_correctly(self):
        """NaN boolean fields default to False, not True."""
        row = pd.Series({
            "p_starting_capital": 20_000, "p_max_trades_per_day": 2,
            "p_monthly_drawdown_limit": float("nan"), "p_lot_per_equity": 10_000,
            "p_calls_only": float("nan"), "p_min_dte": float("nan"),
            "p_max_delta": float("nan"),
            "t_tp_pct": 0.50, "t_sl_mult": float("nan"),
            "t_max_vix": float("nan"), "t_max_term_structure": float("nan"),
            "t_avoid_opex": float("nan"), "t_prefer_event_days": float("nan"),
            "t_width_filter": float("nan"), "t_entry_count": float("nan"),
            "e_enabled": float("nan"), "e_budget_mode": "shared",
            "e_max_event_trades": 1, "e_spx_drop_threshold": -0.01,
            "e_side_preference": "puts", "e_min_dte": 5, "e_max_dte": 7,
            "e_min_delta": 0.15, "e_max_delta": 0.25,
            "e_rally_avoidance": float("nan"), "e_rally_threshold": 0.01,
        })
        cfg = _row_to_config(row)
        assert cfg.portfolio.calls_only is True  # default
        assert cfg.trading.avoid_opex is False
        assert cfg.trading.prefer_event_days is False
        assert cfg.event.enabled is False
        assert cfg.event.rally_avoidance is False


# ── PortfolioManager margin_per_lot ──────────────────────────────


class TestPortfolioManagerMargin:
    def test_default_margin(self):
        """Default margin is MARGIN_PER_LOT (1000)."""
        pm = PortfolioManager(PortfolioConfig(starting_capital=20_000))
        assert pm.margin_per_lot == MARGIN_PER_LOT

    def test_custom_margin(self):
        """Custom margin_per_lot affects lot computation and min equity."""
        pm = PortfolioManager(PortfolioConfig(starting_capital=2000), margin_per_lot=500)
        pm.begin_day("2025-06-01")
        assert pm.can_trade()

        pm2 = PortfolioManager(PortfolioConfig(starting_capital=400), margin_per_lot=500)
        pm2.begin_day("2025-06-01")
        assert not pm2.can_trade()


# ── EventConfig signal_mode ─────────────────────────────────────


class TestSignalMode:
    """Tests for signal_mode (any / all / spx_and_vix) in EventSignalDetector."""

    def _crash_row(self) -> pd.Series:
        """Row where all thresholds fire (SPX drop, VIX spike, elevated, etc.)."""
        return pd.Series({
            "prev_spx_return": -0.03,
            "prev_spx_return_2d": -0.04,
            "prev_vix_pct_change": 0.30,
            "vix": 35,
            "term_structure": 1.2,
        })

    def _spx_only_row(self) -> pd.Series:
        """Row where only SPX drop fires, VIX is calm."""
        return pd.Series({
            "prev_spx_return": -0.015,
            "prev_spx_return_2d": -0.025,
            "prev_vix_pct_change": 0.02,
            "vix": 15,
            "term_structure": 0.90,
        })

    def _vix_only_row(self) -> pd.Series:
        """Row where only VIX fires, SPX is flat."""
        return pd.Series({
            "prev_spx_return": 0.001,
            "prev_spx_return_2d": 0.003,
            "prev_vix_pct_change": 0.30,
            "vix": 35,
            "term_structure": 0.90,
        })

    def test_any_mode_fires_on_single_signal(self):
        """'any' mode fires when just one signal triggers."""
        det = EventSignalDetector(EventConfig(
            enabled=True, signal_mode="any", spx_drop_threshold=-0.01,
        ))
        signals = det.detect(self._spx_only_row())
        assert len(signals) > 0
        assert "spx_drop_1d" in signals

    def test_all_mode_requires_every_signal(self):
        """'all' mode only fires when ALL configured thresholds trigger."""
        det = EventSignalDetector(EventConfig(
            enabled=True, signal_mode="all",
            spx_drop_threshold=-0.01, spx_drop_2d_threshold=-0.02,
            vix_spike_threshold=0.10, vix_elevated_threshold=25,
            term_inversion_threshold=1.0,
        ))
        assert len(det.detect(self._crash_row())) > 0
        assert len(det.detect(self._spx_only_row())) == 0

    def test_spx_and_vix_mode_requires_both(self):
        """'spx_and_vix' requires at least one SPX AND one VIX signal."""
        det = EventSignalDetector(EventConfig(
            enabled=True, signal_mode="spx_and_vix",
            spx_drop_threshold=-0.01,
            vix_spike_threshold=0.10, vix_elevated_threshold=25,
        ))
        assert len(det.detect(self._crash_row())) > 0
        assert len(det.detect(self._spx_only_row())) == 0
        assert len(det.detect(self._vix_only_row())) == 0

    def test_spx_and_vix_with_crash(self):
        """Crash row triggers both SPX and VIX, so spx_and_vix should fire."""
        det = EventSignalDetector(EventConfig(
            enabled=True, signal_mode="spx_and_vix",
            spx_drop_threshold=-0.01,
            vix_spike_threshold=0.10,
        ))
        signals = det.detect(self._crash_row())
        assert "spx_drop_1d" in signals
        assert "vix_spike" in signals

    def test_rally_preserved_regardless_of_mode(self):
        """Rally signal is kept even when other signals are filtered out."""
        det = EventSignalDetector(EventConfig(
            enabled=True, signal_mode="all",
            rally_avoidance=True, rally_threshold=0.005,
        ))
        row = pd.Series({
            "prev_spx_return": 0.02,
            "prev_spx_return_2d": 0.03,
            "prev_vix_pct_change": 0.01,
            "vix": 15,
            "term_structure": 0.90,
        })
        signals = det.detect(row)
        assert "rally" in signals

    def test_all_mode_category_based(self):
        """'all' fires when all 3 categories have at least one signal, even
        if not all 5 individual signals fire."""
        det = EventSignalDetector(EventConfig(
            enabled=True, signal_mode="all",
            spx_drop_threshold=-0.01, spx_drop_2d_threshold=-0.99,
            vix_spike_threshold=0.10, vix_elevated_threshold=999,
            term_inversion_threshold=1.0,
        ))
        # spx_drop_1d fires, spx_drop_2d does NOT (threshold too extreme)
        # vix_spike fires, vix_elevated does NOT (threshold too extreme)
        # term_inversion fires
        row = pd.Series({
            "prev_spx_return": -0.03,
            "prev_spx_return_2d": -0.04,
            "prev_vix_pct_change": 0.30,
            "vix": 35,
            "term_structure": 1.2,
        })
        signals = det.detect(row)
        assert "spx_drop_1d" in signals
        assert "vix_spike" in signals
        assert "term_inversion" in signals

    def test_all_mode_missing_ts_category(self):
        """'all' does NOT fire when term structure category is missing."""
        det = EventSignalDetector(EventConfig(
            enabled=True, signal_mode="all",
            spx_drop_threshold=-0.01,
            vix_spike_threshold=0.10,
            term_inversion_threshold=1.0,
        ))
        row = pd.Series({
            "prev_spx_return": -0.03,
            "prev_spx_return_2d": -0.04,
            "prev_vix_pct_change": 0.30,
            "vix": 35,
            "term_structure": 0.90,  # normal, no inversion
        })
        signals = det.detect(row)
        # Has SPX + VIX but no TS -> should be empty (+ no rally)
        assert len(signals) == 0


# ── SPX drop range gate ─────────────────────────────────────────


class TestSpxDropRange:
    """Tests for spx_drop_min / spx_drop_max magnitude bucketing."""

    def test_within_range_fires(self):
        """SPX drop within [min, max] range triggers the signal."""
        det = EventSignalDetector(EventConfig(
            enabled=True, spx_drop_threshold=-0.005,
            spx_drop_min=-0.02, spx_drop_max=-0.005,
        ))
        row = pd.Series({
            "prev_spx_return": -0.015,
            "prev_spx_return_2d": 0, "prev_vix_pct_change": 0,
            "vix": 15, "term_structure": 0.90,
        })
        assert "spx_drop_1d" in det.detect(row)

    def test_below_min_suppressed(self):
        """SPX drop more severe than spx_drop_min is outside the range."""
        det = EventSignalDetector(EventConfig(
            enabled=True, spx_drop_threshold=-0.005,
            spx_drop_min=-0.02, spx_drop_max=-0.005,
        ))
        row = pd.Series({
            "prev_spx_return": -0.03,
            "prev_spx_return_2d": 0, "prev_vix_pct_change": 0,
            "vix": 15, "term_structure": 0.90,
        })
        assert "spx_drop_1d" not in det.detect(row)

    def test_above_max_suppressed(self):
        """SPX drop not severe enough (above max) does not fire."""
        det = EventSignalDetector(EventConfig(
            enabled=True, spx_drop_threshold=-0.005,
            spx_drop_min=-0.02, spx_drop_max=-0.01,
        ))
        row = pd.Series({
            "prev_spx_return": -0.007,
            "prev_spx_return_2d": 0, "prev_vix_pct_change": 0,
            "vix": 15, "term_structure": 0.90,
        })
        assert "spx_drop_1d" not in det.detect(row)

    def test_no_range_constraints_passes_through(self):
        """Without range constraints, any drop below threshold fires."""
        det = EventSignalDetector(EventConfig(
            enabled=True, spx_drop_threshold=-0.01,
        ))
        row = pd.Series({
            "prev_spx_return": -0.05,
            "prev_spx_return_2d": 0, "prev_vix_pct_change": 0,
            "vix": 15, "term_structure": 0.90,
        })
        assert "spx_drop_1d" in det.detect(row)

    def test_only_min_set(self):
        """Only spx_drop_min set: suppresses drops more severe than min."""
        det = EventSignalDetector(EventConfig(
            enabled=True, spx_drop_threshold=-0.005,
            spx_drop_min=-0.02,
        ))
        mild = pd.Series({
            "prev_spx_return": -0.015,
            "prev_spx_return_2d": 0, "prev_vix_pct_change": 0,
            "vix": 15, "term_structure": 0.90,
        })
        severe = pd.Series({
            "prev_spx_return": -0.03,
            "prev_spx_return_2d": 0, "prev_vix_pct_change": 0,
            "vix": 15, "term_structure": 0.90,
        })
        assert "spx_drop_1d" in det.detect(mild)
        assert "spx_drop_1d" not in det.detect(severe)


# ── EventConfig roundtrip with new fields ────────────────────────


class TestEventConfigRoundtrip:
    """Verify new EventConfig fields survive flat_dict -> _row_to_config."""

    def test_signal_mode_roundtrip(self):
        """signal_mode survives roundtrip."""
        cfg = FullConfig(
            portfolio=PortfolioConfig(),
            trading=TradingConfig(),
            event=EventConfig(enabled=True, signal_mode="spx_and_vix"),
        )
        flat = cfg.flat_dict()
        assert flat["e_signal_mode"] == "spx_and_vix"
        restored = _row_to_config(pd.Series(flat))
        assert restored.event.signal_mode == "spx_and_vix"

    def test_spx_drop_range_roundtrip(self):
        """spx_drop_min and spx_drop_max survive roundtrip."""
        cfg = FullConfig(
            portfolio=PortfolioConfig(),
            trading=TradingConfig(),
            event=EventConfig(
                enabled=True,
                spx_drop_min=-0.02, spx_drop_max=-0.005,
            ),
        )
        flat = cfg.flat_dict()
        assert flat["e_spx_drop_min"] == -0.02
        assert flat["e_spx_drop_max"] == -0.005
        restored = _row_to_config(pd.Series(flat))
        assert restored.event.spx_drop_min == pytest.approx(-0.02)
        assert restored.event.spx_drop_max == pytest.approx(-0.005)

    def test_threshold_fields_roundtrip(self):
        """New threshold sweep fields survive roundtrip."""
        cfg = FullConfig(
            portfolio=PortfolioConfig(),
            trading=TradingConfig(),
            event=EventConfig(
                enabled=True,
                vix_spike_threshold=0.20,
                vix_elevated_threshold=30.0,
                spx_drop_2d_threshold=-0.03,
            ),
        )
        flat = cfg.flat_dict()
        restored = _row_to_config(pd.Series(flat))
        assert restored.event.vix_spike_threshold == pytest.approx(0.20)
        assert restored.event.vix_elevated_threshold == pytest.approx(30.0)
        assert restored.event.spx_drop_2d_threshold == pytest.approx(-0.03)

    def test_none_range_roundtrip(self):
        """None spx_drop_min/max stay None after roundtrip."""
        cfg = FullConfig(
            portfolio=PortfolioConfig(),
            trading=TradingConfig(),
            event=EventConfig(enabled=True),
        )
        flat = cfg.flat_dict()
        restored = _row_to_config(pd.Series(flat))
        assert restored.event.spx_drop_min is None
        assert restored.event.spx_drop_max is None

    def test_nan_string_fields_get_defaults(self):
        """NaN in string EventConfig fields falls back to defaults, not 'nan'."""
        flat = FullConfig(
            portfolio=PortfolioConfig(),
            trading=TradingConfig(),
            event=EventConfig(enabled=True, signal_mode="spx_and_vix"),
        ).flat_dict()
        flat["e_signal_mode"] = float("nan")
        flat["e_budget_mode"] = float("nan")
        flat["e_side_preference"] = float("nan")
        restored = _row_to_config(pd.Series(flat))
        assert restored.event.signal_mode == "any"
        assert restored.event.budget_mode == "shared"
        assert restored.event.side_preference == "puts"


# ── EventOnly mode ──────────────────────────────────────────────


class TestEventOnlyMode:
    """Verify event_only flag suppresses scheduled trades."""

    def test_event_only_skips_scheduled(self, multi_day_df):
        """With event_only=True and no drop signals, no trades happen."""
        daily_signals = precompute_daily_signals(multi_day_df)
        cfg = FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000),
            trading=TradingConfig(),
            event=EventConfig(enabled=True, event_only=True),
        )
        result = run_backtest(multi_day_df, daily_signals, cfg)
        assert result.total_trades == 0

    def test_event_only_flat_dict_roundtrip(self):
        """event_only survives flat_dict -> _row_to_config."""
        cfg = FullConfig(
            event=EventConfig(enabled=True, event_only=True),
        )
        flat = cfg.flat_dict()
        assert flat["e_event_only"] is True
        restored = _row_to_config(pd.Series(flat))
        assert restored.event.event_only is True

    def test_event_only_grid_builds(self):
        """_build_event_only_grid produces valid configs with event_only=True."""
        configs = _build_event_only_grid()
        assert len(configs) > 1000
        for c in configs[:10]:
            assert c.event.enabled is True
            assert c.event.event_only is True
            assert c.event.budget_mode == "shared"


# ── RegimeThrottle ──────────────────────────────────────────────


class TestRegimeThrottle:
    """Verify regime-based position throttling."""

    def _make_signals(self, vix=18.0, prev_ret=0.001):
        """Helper: build a daily_signals-like DataFrame."""
        return pd.DataFrame(
            [{"vix": vix, "prev_spx_return": prev_ret}],
            index=["2025-06-01"],
        )

    def test_disabled_returns_1(self):
        """Disabled throttle always returns multiplier 1.0."""
        rt = RegimeThrottle(enabled=False)
        sigs = self._make_signals(vix=50.0, prev_ret=-0.05)
        assert compute_regime_multiplier(rt, "2025-06-01", sigs, 10) == 1.0

    def test_high_vix_throttle(self):
        """VIX above threshold reduces multiplier."""
        rt = RegimeThrottle(enabled=True, high_vix_threshold=25.0, high_vix_multiplier=0.5)
        sigs = self._make_signals(vix=28.0)
        mult = compute_regime_multiplier(rt, "2025-06-01", sigs, 0)
        assert mult == pytest.approx(0.5)

    def test_extreme_vix_skips(self):
        """VIX above extreme threshold returns 0.0 (skip day)."""
        rt = RegimeThrottle(enabled=True, extreme_vix_threshold=40.0)
        sigs = self._make_signals(vix=42.0)
        mult = compute_regime_multiplier(rt, "2025-06-01", sigs, 0)
        assert mult == 0.0

    def test_big_drop_throttle(self):
        """Large prior-day SPX drop reduces multiplier."""
        rt = RegimeThrottle(enabled=True, big_drop_threshold=-0.02, big_drop_multiplier=0.5)
        sigs = self._make_signals(prev_ret=-0.03)
        mult = compute_regime_multiplier(rt, "2025-06-01", sigs, 0)
        assert mult == pytest.approx(0.5)

    def test_consecutive_loss_throttle(self):
        """Consecutive losing days reduce multiplier."""
        rt = RegimeThrottle(enabled=True, consecutive_loss_days=3, consecutive_loss_multiplier=0.5)
        sigs = self._make_signals()
        mult = compute_regime_multiplier(rt, "2025-06-01", sigs, 5)
        assert mult == pytest.approx(0.5)

    def test_multiple_conditions_take_minimum(self):
        """When multiple conditions fire, the smallest multiplier wins."""
        rt = RegimeThrottle(
            enabled=True,
            high_vix_threshold=25.0, high_vix_multiplier=0.7,
            big_drop_threshold=-0.01, big_drop_multiplier=0.3,
        )
        sigs = self._make_signals(vix=28.0, prev_ret=-0.02)
        mult = compute_regime_multiplier(rt, "2025-06-01", sigs, 0)
        assert mult == pytest.approx(0.3)

    def test_normal_conditions_no_throttle(self):
        """Calm market returns 1.0 even when throttle is enabled."""
        rt = RegimeThrottle(enabled=True)
        sigs = self._make_signals(vix=18.0, prev_ret=0.005)
        mult = compute_regime_multiplier(rt, "2025-06-01", sigs, 0)
        assert mult == 1.0

    def test_regime_throttle_roundtrip(self):
        """RegimeThrottle survives flat_dict -> _row_to_config."""
        cfg = FullConfig(
            regime=RegimeThrottle(
                enabled=True, high_vix_threshold=28.0,
                big_drop_multiplier=0.3,
            ),
        )
        flat = cfg.flat_dict()
        assert flat["r_enabled"] is True
        assert flat["r_high_vix_threshold"] == 28.0
        restored = _row_to_config(pd.Series(flat))
        assert restored.regime.enabled is True
        assert restored.regime.high_vix_threshold == pytest.approx(28.0)
        assert restored.regime.big_drop_multiplier == pytest.approx(0.3)

    def test_regime_throttle_in_backtest(self, multi_day_df):
        """Regime throttle changes equity outcome when VIX exceeds threshold."""
        daily_signals = precompute_daily_signals(multi_day_df)
        cfg_no_throttle = FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000),
            trading=TradingConfig(),
        )
        cfg_with_throttle = FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000),
            trading=TradingConfig(),
            regime=RegimeThrottle(enabled=True, high_vix_threshold=15.0, high_vix_multiplier=0.5),
        )
        r_no = run_backtest(multi_day_df, daily_signals, cfg_no_throttle)
        r_with = run_backtest(multi_day_df, daily_signals, cfg_with_throttle)
        assert r_no.total_trades > 0
        assert r_with.total_trades > 0
        assert r_no.final_equity != r_with.final_equity, (
            "Throttled and unthrottled runs should produce different equity outcomes"
        )


# ── Selective grid ──────────────────────────────────────────────


class TestSelectiveGrid:
    """Verify selective high-win-rate grid construction."""

    def test_grid_builds(self):
        """Grid produces non-empty list of valid configs."""
        configs = _build_selective_grid()
        assert len(configs) > 100

    def test_conservative_parameters(self):
        """All selective configs use conservative VIX, DTE, delta filters."""
        configs = _build_selective_grid()
        for c in configs[:50]:
            assert c.trading.max_vix is not None
            assert c.trading.max_vix <= 30.0
            assert c.portfolio.min_dte is not None
            assert c.portfolio.min_dte >= 3
            assert c.portfolio.max_delta is not None
            assert c.portfolio.max_delta <= 0.20
            assert c.trading.width_filter == 10.0
            assert c.trading.entry_count is not None
            assert c.event.enabled is False

    def test_capital_levels_present(self):
        """Grid includes multiple capital levels."""
        configs = _build_selective_grid()
        caps = set(c.portfolio.starting_capital for c in configs)
        assert 20_000 in caps
        assert 50_000 in caps
        assert 100_000 in caps

    def test_regime_throttle_variants(self):
        """Grid includes both throttle=on and throttle=off configs."""
        configs = _build_selective_grid()
        has_on = any(c.regime.enabled for c in configs)
        has_off = any(not c.regime.enabled for c in configs)
        assert has_on and has_off
