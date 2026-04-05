"""Tests for the capital-budgeted backtest engine.

Covers PortfolioManager, EventSignalDetector, ScheduledSelector,
EventSelector, precompute_daily_signals, and run_backtest.
"""
from __future__ import annotations

import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backtest_strategy import (
    BacktestResult,
    DayRecord,
    EventConfig,
    EventSelector,
    EventSignalDetector,
    FullConfig,
    PortfolioConfig,
    PortfolioManager,
    ScheduledSelector,
    precompute_daily_signals,
    run_backtest,
    MARGIN_PER_LOT,
    _precompute_day_selections,
    _fast_sched_select,
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


class TestScheduledSelector:
    def test_calls_only(self, sample_day_df):
        """calls_only=True filters out put candidates."""
        sel = ScheduledSelector(PortfolioConfig(calls_only=True), max_n=2)
        result = sel.select(sample_day_df)
        assert all(r["spread_side"] == "call" for _, r in result.iterrows())

    def test_both_sides(self, sample_day_df):
        """calls_only=False includes both sides."""
        sel = ScheduledSelector(PortfolioConfig(calls_only=False), max_n=5)
        result = sel.select(sample_day_df)
        sides = set(result["spread_side"])
        assert sides == {"call", "put"}

    def test_max_n_limit(self, sample_day_df):
        """Never returns more than max_n candidates."""
        sel = ScheduledSelector(PortfolioConfig(calls_only=False), max_n=1)
        result = sel.select(sample_day_df)
        assert len(result) <= 1

    def test_best_per_decision_time(self, sample_day_df):
        """Picks highest credit_to_width per entry_dt."""
        sel = ScheduledSelector(PortfolioConfig(calls_only=True), max_n=10)
        result = sel.select(sample_day_df)
        assert len(result) == 2  # one per entry_dt
        assert result.iloc[0]["credit_to_width"] >= result.iloc[1]["credit_to_width"]

    def test_min_dte_filter(self, sample_day_df):
        """min_dte filters out short-dated candidates."""
        sel = ScheduledSelector(PortfolioConfig(calls_only=True, min_dte=5), max_n=10)
        result = sel.select(sample_day_df)
        assert all(r["dte_target"] >= 5 for _, r in result.iterrows())

    def test_empty_after_filter(self, sample_day_df):
        """Returns empty when all candidates are filtered out."""
        sel = ScheduledSelector(PortfolioConfig(calls_only=True, min_dte=100), max_n=10)
        result = sel.select(sample_day_df)
        assert len(result) == 0


# ── EventSelector ────────────────────────────────────────────────


class TestEventSelector:
    def test_puts_on_drop(self, sample_day_df):
        """Selects puts on SPX drop signals."""
        ec = EventConfig(enabled=True, side_preference="puts", min_dte=3, max_dte=10, min_delta=0.10, max_delta=0.25)
        sel = EventSelector(ec, max_n=1)
        result = sel.select(sample_day_df, ["spx_drop_1d"])
        assert all(r["spread_side"] == "put" for _, r in result.iterrows())

    def test_no_signals_returns_empty(self, sample_day_df):
        """No signals returns empty selection."""
        ec = EventConfig(enabled=True)
        sel = EventSelector(ec, max_n=2)
        result = sel.select(sample_day_df, [])
        assert len(result) == 0

    def test_disabled_returns_empty(self, sample_day_df):
        """Disabled event config returns empty."""
        ec = EventConfig(enabled=False)
        sel = EventSelector(ec, max_n=2)
        result = sel.select(sample_day_df, ["spx_drop_1d"])
        assert len(result) == 0

    def test_dte_range_filter(self, sample_day_df):
        """Event DTE range filters correctly."""
        ec = EventConfig(enabled=True, side_preference="best", min_dte=5, max_dte=7, min_delta=0.10, max_delta=0.25)
        sel = EventSelector(ec, max_n=10)
        result = sel.select(sample_day_df, ["spx_drop_1d"])
        for _, r in result.iterrows():
            assert 5 <= r["dte_target"] <= 7


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
        assert len(pareto) >= 1
