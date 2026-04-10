"""Capital-budgeted backtest engine with scheduled + event-driven layers.

Simulates a fixed-capital portfolio with configurable:
  - Position sizing (gradual lot scaling or fixed)
  - Monthly drawdown stop-loss
  - Scheduled trade selection (calls-only or both, by credit_to_width)
  - Event-driven signals (SPX drop, VIX spike, rally avoidance, etc.)
  - Exhaustive parameter grid search with CSV export

Usage::

    # Single run with defaults ($20k, 2/day, 15% monthly stop)
    python scripts/backtest_strategy.py

    # Custom single run
    python scripts/backtest_strategy.py --capital 20000 --max-trades 2 --monthly-stop 0.15

    # Exhaustive optimizer (sweeps ~1000+ configs, exports CSV)
    python scripts/backtest_strategy.py --optimize

    # Quick comparison of preset configs
    python scripts/backtest_strategy.py --compare

Continuous Improvement Workflow
-------------------------------
**Weekly** -- Compare live paper-trade PnL against backtest expectations.
Check that win-rate and average trade PnL are within 1 standard deviation
of the backtest distribution.  Investigate if 3+ consecutive losing days
occur that the backtest did not predict.

**Monthly** -- After accumulating a full calendar month of live data:

1. Export latest production data (chains, underlying quotes, calendar)::

       python scripts/export_production_data.py --tables all

2. Regenerate training candidates.  The pipeline auto-merges Databento
   files (historical) with production exports (recent), so no separate
   download step is needed for dates covered by the production DB::

       python scripts/generate_training_data.py --workers 4

3. Re-run the optimizer with the expanded dataset::

       python scripts/backtest_strategy.py --optimize

4. Run walk-forward validation with auto-generated windows::

       python scripts/backtest_strategy.py --walkforward --wf-auto

5. If the current config's out-of-sample Sharpe drops >30% vs the
   original walkforward result for 2+ consecutive months, re-select
   the best config from the updated Pareto frontier and update
   ``.env`` PORTFOLIO_*/EVENT_* values accordingly.

**On new data sources** -- When adding FRD data for earlier years (2022-2024),
regenerate the full training set and re-run ``--optimize --walkforward``
to validate the strategy on the longer history.  Larger datasets
reduce overfitting risk and increase confidence in parameter choices.

**Production DB as sole data source** -- Once the production system has
accumulated enough data, Databento/FRD downloads are no longer needed.
The ``export_production_data.py`` script exports ``chain_snapshots``,
``option_chain_rows``, and ``underlying_quotes`` as per-day Parquet files
that the training pipeline consumes transparently via auto-fallback
loaders.  Run ``--wf-auto`` to auto-size walk-forward windows to the
available data range.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from _constants import CONTRACT_MULT, CONTRACTS, MARGIN_PER_LOT

DATA_DIR = _SCRIPTS_DIR.parents[1] / "data"
DEFAULT_CSV = DATA_DIR / "training_candidates.csv"
RESULTS_CSV = DATA_DIR / "backtest_results.csv"


def _opt_val(v):
    """Return *v* unchanged if it is not NaN/None, else None."""
    return v if pd.notna(v) else None


def _opt_str(v, default: str = "") -> str:
    """Convert *v* to str, returning *default* when v is NaN or missing."""
    return str(v) if pd.notna(v) else default


def _safe_bool(v, default: bool = False) -> bool:
    """Convert *v* to bool, returning *default* when v is NaN or missing."""
    if pd.isna(v):
        return default
    return bool(v)


# ===================================================================
# Configuration dataclasses
# ===================================================================


@dataclass
class PortfolioConfig:
    """All portfolio-level parameters.

    Every field is configurable for the grid-search optimizer.
    Defaults reflect the recommended starting configuration.
    """

    starting_capital: float = 20_000
    max_trades_per_day: int = 2
    monthly_drawdown_limit: float | None = 0.15
    lot_per_equity: float = 10_000
    max_equity_risk_pct: float = 0.10
    max_margin_pct: float = 0.30
    calls_only: bool = True
    min_dte: int | None = None
    max_delta: float | None = None


@dataclass
class EventConfig:
    """All event-driven signal parameters.

    When ``enabled`` is False the event layer is skipped entirely
    and all other fields are ignored by the optimizer.

    ``signal_mode`` controls how multiple signals combine:
      - ``"any"``  (default) -- fire if ANY signal is active (OR logic)
      - ``"all"``  -- fire only if ALL configured signals are active (AND)
      - ``"spx_and_vix"`` -- require at least one SPX drop signal AND
        at least one VIX signal (spike or elevated)

    ``spx_drop_min`` / ``spx_drop_max`` (optional) restrict the event
    trigger to SPX drops within a specific magnitude range, e.g.
    ``spx_drop_min=-0.02, spx_drop_max=-0.005`` fires only on 0.5%-2% drops.
    """

    enabled: bool = False

    # Signal combination logic
    signal_mode: str = "any"           # "any", "all", or "spx_and_vix"

    # Budget allocation
    budget_mode: str = "shared"        # "shared" or "separate"
    max_event_trades: int = 1          # only used in "separate" mode

    # SPX drop triggers (previous-day return)
    spx_drop_threshold: float = -0.01  # e.g. -0.01 = drop > 1%
    spx_drop_2d_threshold: float = -0.02
    spx_drop_min: float | None = None  # floor of SPX drop range (e.g. -0.02)
    spx_drop_max: float | None = None  # ceiling of SPX drop range (e.g. -0.005)

    # VIX triggers
    vix_spike_threshold: float = 0.15  # VIX % change > 15%
    vix_elevated_threshold: float = 25.0

    # Term structure
    term_inversion_threshold: float = 1.0  # VIX/VIX9D > 1.0

    # Event-driven trade parameters
    side_preference: str = "puts"      # "puts", "calls", or "best"
    min_dte: int = 5
    max_dte: int = 7
    min_delta: float = 0.15
    max_delta: float = 0.25

    # Rally avoidance
    rally_avoidance: bool = False
    rally_threshold: float = 0.01

    # Event-only mode: suppress all scheduled trades, only trade on signal days
    event_only: bool = False


@dataclass
class TradingConfig:
    """Trade-level exit rules and regime filters.

    These parameters control *how* individual trades are managed
    (TP/SL exit logic) and *when* the strategy trades (regime filters),
    independently of portfolio sizing and event signal detection.
    """

    tp_pct: float = 0.50
    """Take-profit as fraction of max profit (0.50 = 50%). Use 1.01 for hold-to-expiry."""

    sl_mult: float | None = None
    """Stop-loss as multiple of credit (e.g. 2.0 = close at -200% of credit). None = no SL."""

    max_vix: float | None = None
    """Skip trading on days where VIX exceeds this level. None = no filter."""

    max_term_structure: float | None = None
    """Skip trading on days where VIX9D/VIX exceeds this (heavily inverted). None = no filter."""

    avoid_opex: bool = False
    """Skip trading on OPEX / options-expiration days."""

    prefer_event_days: bool = False
    """Only trade on FOMC, NFP, or CPI days (aggressive filter for high-IV environments)."""

    width_filter: float | None = None
    """Only use candidates with this spread width (points). None = use all widths."""

    entry_count: int | None = None
    """Limit to the last N entry times per day. None = use all available entry times."""


@dataclass
class RegimeThrottle:
    """Regime-based position sizing throttle.

    Reduces lot count (or skips trading) on days where market conditions
    suggest elevated risk.  Each condition is checked independently and
    the most aggressive throttle wins (i.e. smallest multiplier).
    """

    enabled: bool = False

    high_vix_threshold: float = 30.0
    """VIX above this level triggers the high-VIX throttle."""
    high_vix_multiplier: float = 0.5
    """Lot multiplier when VIX exceeds ``high_vix_threshold``.  0.0 = skip day."""

    extreme_vix_threshold: float = 40.0
    """VIX above this stops trading entirely (lots = 0)."""

    big_drop_threshold: float = -0.02
    """Prior-day SPX return below this triggers the big-drop throttle."""
    big_drop_multiplier: float = 0.5
    """Lot multiplier when prior-day SPX return exceeds ``big_drop_threshold``."""

    consecutive_loss_days: int = 3
    """After this many consecutive losing days, throttle lots."""
    consecutive_loss_multiplier: float = 0.5
    """Lot multiplier during consecutive-loss streaks."""


def compute_regime_multiplier(
    throttle: "RegimeThrottle",
    day: str,
    daily_signals: pd.DataFrame,
    consecutive_losses: int,
) -> float:
    """Return the lot-size multiplier for the given day under regime throttling.

    Checks VIX level, prior-day SPX drop, and consecutive loss count.
    Returns the minimum (most conservative) multiplier across all active rules.
    A return value of 0.0 means skip the day entirely.
    """
    if not throttle.enabled:
        return 1.0

    mult = 1.0

    if day in daily_signals.index:
        row = daily_signals.loc[day]
        vix = row.get("vix")
        if pd.notna(vix):
            if vix >= throttle.extreme_vix_threshold:
                return 0.0
            if vix >= throttle.high_vix_threshold:
                mult = min(mult, throttle.high_vix_multiplier)

        prev_ret = row.get("prev_spx_return")
        if pd.notna(prev_ret) and prev_ret <= throttle.big_drop_threshold:
            mult = min(mult, throttle.big_drop_multiplier)

    if consecutive_losses >= throttle.consecutive_loss_days:
        mult = min(mult, throttle.consecutive_loss_multiplier)

    return mult


@dataclass
class FullConfig:
    """Combines portfolio, trading, event, and regime-throttle configs."""

    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    event: EventConfig = field(default_factory=EventConfig)
    regime: RegimeThrottle = field(default_factory=RegimeThrottle)

    def flat_dict(self) -> dict[str, Any]:
        """Flatten all configs into a single dict for CSV export."""
        d: dict[str, Any] = {}
        for k, v in asdict(self.portfolio).items():
            d[f"p_{k}"] = v
        for k, v in asdict(self.trading).items():
            d[f"t_{k}"] = v
        for k, v in asdict(self.event).items():
            d[f"e_{k}"] = v
        for k, v in asdict(self.regime).items():
            d[f"r_{k}"] = v
        return d


# ===================================================================
# Effective PnL computation
# ===================================================================


def pnl_column_name(tp_pct: float, sl_mult: float | None) -> str:
    """Deterministic column name for a (tp_pct, sl_mult) combo."""
    sl_tag = "none" if sl_mult is None else f"{sl_mult:.1f}"
    return f"pnl_tp{int(tp_pct * 100)}_{sl_tag}"


def compute_effective_pnl(
    row: pd.Series,
    tp_pct: float,
    sl_mult: float | None,
) -> float | None:
    """Derive per-trade PnL from trajectory columns under a given exit policy.

    Uses the multi-TP columns (first_tpXX_pnl, min_pnl_before_tpXX) to
    determine whether SL fires before TP, whether TP fires at all, and
    falls back to final_pnl_at_expiry otherwise.

    Parameters
    ----------
    row : One row from the training CSV with trajectory columns.
    tp_pct : Take-profit fraction (0.50-1.00). Values > 1.0 mean hold-to-expiry.
    sl_mult : Stop-loss multiple of credit. None means no stop-loss.

    Returns
    -------
    float | None
        Trade PnL in dollars, or None if unresolved.
    """
    if not row.get("resolved", False):
        return None

    entry_credit = row.get("entry_credit")
    if entry_credit is None or entry_credit <= 0:
        return None

    max_profit = entry_credit * CONTRACT_MULT * CONTRACTS
    sl_thr = max_profit * sl_mult if sl_mult is not None else float("inf")

    # Hold-to-expiry path (tp_pct > 1.0)
    if tp_pct > 1.0:
        max_adverse = row.get("max_adverse_pnl")
        if sl_mult is not None and max_adverse is not None and max_adverse <= -sl_thr:
            return -sl_thr
        final = row.get("final_pnl_at_expiry")
        return final if final is not None and not _isnan(final) else row.get("realized_pnl")

    tp_key = int(tp_pct * 100)
    first_tp = row.get(f"first_tp{tp_key}_pnl")
    min_before = row.get(f"min_pnl_before_tp{tp_key}")

    has_tp_data = first_tp is not None and not _isnan(first_tp)
    has_min_data = min_before is not None and not _isnan(min_before)

    # SL fires before TP?
    if sl_mult is not None and has_min_data and min_before <= -sl_thr:
        return -sl_thr

    # TP fires
    if has_tp_data:
        return first_tp

    # Neither TP nor SL hit before expiry -- check SL during hold
    max_adverse = row.get("max_adverse_pnl")
    if sl_mult is not None and max_adverse is not None and not _isnan(max_adverse):
        if max_adverse <= -sl_thr:
            return -sl_thr

    final = row.get("final_pnl_at_expiry")
    if final is not None and not _isnan(final):
        return final
    return row.get("realized_pnl")


def _isnan(val) -> bool:
    """Check if a value is NaN (safe for non-float types)."""
    try:
        return pd.isna(val)
    except (TypeError, ValueError):
        return False


def precompute_pnl_columns(
    df: pd.DataFrame,
    tp_values: list[float],
    sl_values: list[float | None],
) -> list[str]:
    """Add pre-computed PnL columns to df for each (tp, sl) combo.

    Uses vectorized numpy operations for performance. Modifies df in place.

    Parameters
    ----------
    df : Training candidates DataFrame (must have trajectory columns).
    tp_values : List of TP fractions to pre-compute.
    sl_values : List of SL multiples to pre-compute (None = no SL).

    Returns
    -------
    List of column names that were added.
    """
    combos = list(itertools.product(tp_values, sl_values))
    print(f"Pre-computing {len(combos)} PnL columns ...", end=" ", flush=True)
    t0 = time.time()

    max_profit = df["entry_credit"] * CONTRACT_MULT * CONTRACTS
    fallback = df["final_pnl_at_expiry"].fillna(df["realized_pnl"])
    resolved = df.get("resolved", pd.Series(True, index=df.index))
    valid_entry = df["entry_credit"].notna() & (df["entry_credit"] > 0) & (resolved == True)  # noqa: E712 — intentional for NaN-safe Pandas comparison
    max_adverse = df.get("max_adverse_pnl", pd.Series(np.nan, index=df.index))

    col_names: list[str] = []
    for tp, sl in combos:
        col = pnl_column_name(tp, sl)
        col_names.append(col)

        sl_thr = max_profit * sl if sl is not None else pd.Series(np.inf, index=df.index)

        if tp > 1.0:
            # Hold-to-expiry: only SL via max_adverse
            adverse_sl = (sl is not None) & max_adverse.notna() & (max_adverse <= -sl_thr)
            result = np.where(adverse_sl, -sl_thr, fallback)
        else:
            tp_key = int(tp * 100)
            tp_col = f"first_tp{tp_key}_pnl"
            min_col = f"min_pnl_before_tp{tp_key}"
            tp_vals = df.get(tp_col, pd.Series(np.nan, index=df.index))
            min_vals = df.get(min_col, pd.Series(np.nan, index=df.index))

            has_tp = tp_vals.notna()
            sl_before_tp = (sl is not None) & min_vals.notna() & (min_vals <= -sl_thr)
            adverse_sl = (sl is not None) & max_adverse.notna() & (max_adverse <= -sl_thr)

            result = np.where(
                sl_before_tp, -sl_thr,
                np.where(has_tp, tp_vals,
                    np.where(adverse_sl, -sl_thr, fallback)))

        df[col] = np.where(valid_entry, result, np.nan)

    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s", flush=True)
    return col_names


# ===================================================================
# Core classes
# ===================================================================


@dataclass
class DayRecord:
    """One row in the equity-curve output."""

    day: str
    equity: float
    daily_pnl: float
    n_trades: int
    lots: int
    status: str
    month_start_equity: float
    event_signals: str = ""


class PortfolioManager:
    """Tracks equity, enforces risk limits, computes lot sizes.

    Parameters
    ----------
    config : PortfolioConfig with all strategy parameters.
    margin_per_lot : Override for margin requirement per lot.
        Defaults to ``MARGIN_PER_LOT`` (1000, i.e. 10-pt × $100/pt).
        When ``TradingConfig.width_filter`` is set, pass
        ``width_filter * 100`` to scale margin to the actual spread width.
    """

    def __init__(self, config: PortfolioConfig, margin_per_lot: float = MARGIN_PER_LOT) -> None:
        """Initialise budget tracker with starting capital from *config*."""
        self.cfg = config
        self.margin_per_lot = margin_per_lot
        self.equity = config.starting_capital
        self.month_start_equity = config.starting_capital
        self._current_month: str | None = None
        self._month_stopped = False
        self._trades_today = 0
        self._lots_today: int | None = None

    def begin_day(self, day: str) -> None:
        """Reset daily counters; handle month rollover."""
        month_key = day[:7]
        if month_key != self._current_month:
            self._current_month = month_key
            self.month_start_equity = self.equity
            self._month_stopped = False
        self._trades_today = 0
        self._lots_today = None

    def can_trade(self) -> bool:
        """Return True if a new trade is allowed right now."""
        if self._month_stopped:
            return False
        if self.equity < self.margin_per_lot:
            return False
        limit = self.cfg.monthly_drawdown_limit
        if limit is not None and self.equity < self.month_start_equity * (1 - limit):
            self._month_stopped = True
            return False
        if self._trades_today >= self.cfg.max_trades_per_day:
            return False
        return True

    @property
    def is_month_stopped(self) -> bool:
        """True when the monthly drawdown stop-loss has been triggered."""
        return self._month_stopped

    def compute_lots(self) -> int:
        """Lot count for the next trade, cached per day for consistency."""
        if self._lots_today is not None:
            return self._lots_today

        raw = max(1, int(self.equity / self.cfg.lot_per_equity))
        max_by_risk = max(1, int(self.equity * self.cfg.max_equity_risk_pct / self.margin_per_lot))
        self._lots_today = min(raw, max_by_risk)
        return self._lots_today

    def record_trade(self, pnl_per_lot: float, lots: int) -> float:
        """Apply a settled trade. Returns total PnL."""
        total_pnl = pnl_per_lot * lots
        self.equity += total_pnl
        self._trades_today += 1
        return total_pnl

    def status_label(self) -> str:
        """Return a human-readable label describing why trading may be blocked."""
        if self._month_stopped:
            return "monthly_stop"
        if self.equity < self.margin_per_lot:
            return "insufficient_capital"
        if self._trades_today >= self.cfg.max_trades_per_day:
            return "daily_limit"
        return "ok"


class EventSignalDetector:
    """Evaluate market conditions and return active event signals.

    Requires a ``day_signals`` DataFrame precomputed from the daily
    aggregates (SPX return, VIX change, etc.).

    Parameters
    ----------
    config : EventConfig with all trigger thresholds.
    """

    def __init__(self, config: EventConfig) -> None:
        """Bind the detector to the given event configuration."""
        self.cfg = config

    def detect(self, day_row: pd.Series) -> list[str]:
        """Return list of active signal names for today.

        Parameters
        ----------
        day_row : Series with keys ``prev_spx_return``, ``prev_spx_return_2d``,
            ``prev_vix_pct_change``, ``vix``, ``term_structure``.
        """
        if not self.cfg.enabled:
            return []

        signals: list[str] = []
        prev_ret = day_row.get("prev_spx_return")
        prev_ret_2d = day_row.get("prev_spx_return_2d")
        prev_vix_chg = day_row.get("prev_vix_pct_change")
        vix = day_row.get("vix")
        ts = day_row.get("term_structure")

        if prev_ret is not None and prev_ret < self.cfg.spx_drop_threshold:
            # Optional magnitude-range gate
            dmin = self.cfg.spx_drop_min
            dmax = self.cfg.spx_drop_max
            in_range = True
            if dmin is not None and prev_ret < dmin:
                in_range = False
            if dmax is not None and prev_ret > dmax:
                in_range = False
            if in_range:
                signals.append("spx_drop_1d")

        if prev_ret_2d is not None and prev_ret_2d < self.cfg.spx_drop_2d_threshold:
            signals.append("spx_drop_2d")
        if prev_vix_chg is not None and prev_vix_chg > self.cfg.vix_spike_threshold:
            signals.append("vix_spike")
        if vix is not None and vix > self.cfg.vix_elevated_threshold:
            signals.append("vix_elevated")
        if ts is not None and ts > self.cfg.term_inversion_threshold:
            signals.append("term_inversion")
        if self.cfg.rally_avoidance and prev_ret is not None and prev_ret > self.cfg.rally_threshold:
            signals.append("rally")

        # Apply signal_mode filtering (rally is always kept separately)
        mode = self.cfg.signal_mode
        non_rally = [s for s in signals if s != "rally"]
        has_rally = "rally" in signals

        if mode == "all":
            # Require all three categories: SPX drop, VIX, and term structure
            spx_ok = any(s.startswith("spx_drop") for s in non_rally)
            vix_ok = any(s in ("vix_spike", "vix_elevated") for s in non_rally)
            ts_ok = "term_inversion" in non_rally
            if not (spx_ok and vix_ok and ts_ok):
                non_rally = []
        elif mode == "spx_and_vix":
            spx_signals = [s for s in non_rally if s.startswith("spx_drop")]
            vix_signals = [s for s in non_rally if s in ("vix_spike", "vix_elevated")]
            if not spx_signals or not vix_signals:
                non_rally = []

        result = non_rally
        if has_rally:
            result.append("rally")
        return result


# ===================================================================
# Precompute daily signals
# ===================================================================


def precompute_daily_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-day DataFrame with lagged market features for event detection.

    Includes calendar flags (is_opex_day, is_fomc_day, is_nfp_day) needed
    by TradingConfig regime filters.

    Parameters
    ----------
    df : Full training_candidates DataFrame.

    Returns
    -------
    DataFrame indexed by ``day`` with columns used by EventSignalDetector
    and TradingConfig day-level filters.
    """
    agg_dict: dict[str, tuple] = {
        "spot": ("spot", "first"),
        "vix": ("vix", "first"),
        "vix9d": ("vix9d", "first"),
        "term_structure": ("term_structure", "first"),
    }
    for cal_col in ("is_opex_day", "is_fomc_day", "is_nfp_day", "is_cpi_day", "is_triple_witching"):
        if cal_col in df.columns:
            agg_dict[cal_col] = (cal_col, "first")

    daily = (
        df.groupby("day")
        .agg(**agg_dict)
        .reset_index()
        .sort_values("day")
        .reset_index(drop=True)
    )
    daily["spx_return"] = daily["spot"].pct_change()
    daily["spx_return_2d"] = daily["spot"].pct_change(2)
    daily["vix_pct_change"] = daily["vix"].pct_change()

    daily["prev_spx_return"] = daily["spx_return"].shift(1)
    daily["prev_spx_return_2d"] = daily["spx_return_2d"].shift(1)
    daily["prev_vix_pct_change"] = daily["vix_pct_change"].shift(1)

    return daily.set_index("day")


# ===================================================================
# Backtest engine
# ===================================================================


@dataclass
class BacktestResult:
    """Complete output from a single backtest run."""

    config: FullConfig
    label: str
    curve: list[DayRecord]
    final_equity: float
    total_return_pct: float
    annualised_return_pct: float
    max_drawdown_pct: float
    trough: float
    total_trades: int
    days_traded: int
    days_stopped: int
    win_days: int
    sharpe: float
    monthly: pd.DataFrame


def _precompute_day_selections(
    df: pd.DataFrame,
    width_filter: float | None = None,
    entry_count: int | None = None,
) -> dict[str, dict]:
    """Pre-group candidates by day and pre-sort by credit_to_width.

    Parameters
    ----------
    df : Training candidates DataFrame.
    width_filter : If set, only keep candidates with this spread width.
    entry_count : If set, restrict to the last N entry times per day.

    Returns a dict keyed by day with pre-filtered views for fast selection.
    """
    filtered = df
    if width_filter is not None and "width_points" in df.columns:
        filtered = filtered[filtered["width_points"] == width_filter]

    df_sorted = filtered.sort_values("credit_to_width", ascending=False)
    result: dict[str, dict] = {}

    for day, grp in df_sorted.groupby("day"):
        if entry_count is not None and "entry_dt" in grp.columns:
            unique_times = sorted(grp["entry_dt"].unique())
            keep_times = unique_times[-entry_count:]
            grp = grp[grp["entry_dt"].isin(keep_times)]

        calls = grp[grp["spread_side"] == "call"]
        puts = grp[grp["spread_side"] == "put"]

        call_best = calls.groupby("entry_dt").head(1) if not calls.empty else calls
        put_best = puts.groupby("entry_dt").head(1) if not puts.empty else puts
        all_best = grp.groupby("entry_dt").head(1) if not grp.empty else grp

        result[day] = {
            "all": grp,
            "call": calls,
            "put": puts,
            "call_best_per_dt": call_best,
            "put_best_per_dt": put_best,
            "all_best_per_dt": all_best,
        }
    return result


def _fast_sched_select(precomp: dict, pc: PortfolioConfig, max_n: int) -> pd.DataFrame:
    """Pick top-N scheduled candidates from pre-sorted data."""
    if pc.calls_only:
        cands = precomp["call_best_per_dt"]
    else:
        cands = precomp["all_best_per_dt"]
    if pc.min_dte is not None:
        cands = cands[cands["dte_target"] >= pc.min_dte]
    if pc.max_delta is not None:
        cands = cands[cands["delta_target"] <= pc.max_delta]
    return cands.head(max_n)


def _fast_event_select(precomp: dict, ec: EventConfig, signals: list[str], max_n: int) -> pd.DataFrame:
    """Pick top-N event candidates from pre-sorted data."""
    has_drop = any(s.startswith("spx_drop") or s in ("vix_spike", "vix_elevated") for s in signals)

    has_rally = "rally" in signals
    if has_drop and ec.side_preference == "puts":
        cands = precomp["put"]
    elif has_rally and ec.side_preference == "calls":
        cands = precomp["call"]
    else:
        cands = precomp["all"]

    mask = (
        (cands["dte_target"] >= ec.min_dte) & (cands["dte_target"] <= ec.max_dte) &
        (cands["delta_target"] >= ec.min_delta) & (cands["delta_target"] <= ec.max_delta)
    )
    filtered = cands[mask]
    if filtered.empty:
        return filtered.head(0)
    return filtered.groupby("entry_dt").head(1).head(max_n)


def _should_skip_day(
    tc: TradingConfig,
    day: str,
    daily_signals: pd.DataFrame,
) -> str | None:
    """Check TradingConfig day-level regime filters.

    Returns a skip reason string, or None if the day should be traded.
    """
    if day not in daily_signals.index:
        return None

    day_row = daily_signals.loc[day]

    if tc.max_vix is not None:
        vix = day_row.get("vix")
        if vix is not None and not pd.isna(vix) and vix > tc.max_vix:
            return "vix_filter"

    if tc.max_term_structure is not None:
        ts = day_row.get("term_structure")
        if ts is not None and not pd.isna(ts) and ts > tc.max_term_structure:
            return "ts_filter"

    def _flag(col: str) -> bool:
        """Return a boolean from *day_row[col]*, defaulting to False on NaN."""
        v = day_row.get(col, False)
        return bool(v) if pd.notna(v) else False

    if tc.avoid_opex:
        if _flag("is_opex_day"):
            return "opex_filter"

    if tc.prefer_event_days:
        if not _flag("is_fomc_day") and not _flag("is_nfp_day") and not _flag("is_cpi_day"):
            return "non_event_day"

    return None


def run_backtest(
    df: pd.DataFrame,
    daily_signals: pd.DataFrame,
    config: FullConfig,
    label: str = "",
    day_precomp: dict[str, dict] | None = None,
) -> BacktestResult:
    """Simulate the full strategy on historical candidates.

    Parameters
    ----------
    df : Full training_candidates DataFrame.
    daily_signals : Precomputed daily signal features (from ``precompute_daily_signals``).
    config : Combined portfolio + trading + event configuration.
    label : Human-readable name for this run.
    day_precomp : Optional pre-computed per-day candidate selections
        (from ``_precompute_day_selections``). Avoids repeated sorting.
        Must be built with matching width_filter / entry_count.

    Returns
    -------
    BacktestResult with equity curve, metrics, and monthly breakdown.
    """
    pc = config.portfolio
    tc = config.trading
    ec = config.event
    effective_pc = pc
    if ec.enabled and ec.budget_mode == "separate":
        effective_pc = PortfolioConfig(**{**vars(pc),
            "max_trades_per_day": pc.max_trades_per_day + ec.max_event_trades})

    margin = tc.width_filter * CONTRACT_MULT if tc.width_filter else MARGIN_PER_LOT
    pm = PortfolioManager(effective_pc, margin_per_lot=margin)
    event_det = EventSignalDetector(ec)
    rt = config.regime

    # Determine PnL column: use pre-computed column if it exists, else fallback
    pnl_col = pnl_column_name(tc.tp_pct, tc.sl_mult)
    use_precomputed = pnl_col in df.columns

    if day_precomp is None:
        day_precomp = _precompute_day_selections(
            df, width_filter=tc.width_filter, entry_count=tc.entry_count,
        )

    days = sorted(day_precomp.keys())
    curve: list[DayRecord] = []
    _consecutive_losses = 0

    for day in days:
        pm.begin_day(day)

        if not pm.can_trade():
            curve.append(DayRecord(
                day=day, equity=pm.equity, daily_pnl=0, n_trades=0,
                lots=0, status=pm.status_label(),
                month_start_equity=pm.month_start_equity,
            ))
            continue

        # --- TradingConfig day-level regime filters ---
        skip_reason = _should_skip_day(tc, day, daily_signals)
        if skip_reason:
            curve.append(DayRecord(
                day=day, equity=pm.equity, daily_pnl=0, n_trades=0,
                lots=0, status=skip_reason,
                month_start_equity=pm.month_start_equity,
            ))
            continue

        # --- Regime throttle ---
        regime_mult = compute_regime_multiplier(rt, day, daily_signals, _consecutive_losses)
        if regime_mult <= 0.0:
            curve.append(DayRecord(
                day=day, equity=pm.equity, daily_pnl=0, n_trades=0,
                lots=0, status="regime_throttle",
                month_start_equity=pm.month_start_equity,
            ))
            continue

        precomp = day_precomp[day]

        signals: list[str] = []
        if day in daily_signals.index:
            day_row = daily_signals.loc[day]
            signals = event_det.detect(day_row)

        skip_scheduled = ec.event_only or (ec.rally_avoidance and "rally" in signals)

        raw_lots = pm.compute_lots()
        lots = max(1, int(raw_lots * regime_mult))
        daily_pnl = 0.0
        trades_placed = 0

        # Resolve which PnL column to read from candidates
        effective_pnl_col = pnl_col if use_precomputed else "realized_pnl"

        # --- Event trades first ---
        drop_signals = [s for s in signals if s != "rally"]
        event_trades_placed = 0
        event_candidates = pd.DataFrame()
        if ec.enabled and drop_signals:
            evt_max = ec.max_event_trades
            event_candidates = _fast_event_select(precomp, ec, signals, evt_max)

            for pnl_val in event_candidates[effective_pnl_col].values:
                if pd.isna(pnl_val):
                    continue
                if not pm.can_trade() or event_trades_placed >= evt_max:
                    break
                trade_pnl = pm.record_trade(float(pnl_val), lots)
                daily_pnl += trade_pnl
                trades_placed += 1
                event_trades_placed += 1

        # --- Scheduled trades ---
        if not skip_scheduled:
            if ec.enabled and ec.budget_mode == "shared":
                sched_limit = max(0, pc.max_trades_per_day - event_trades_placed)
            else:
                sched_limit = pc.max_trades_per_day
            sched_candidates = _fast_sched_select(precomp, pc, sched_limit)
            if len(event_candidates) > 0 and len(sched_candidates) > 0:
                sched_candidates = sched_candidates[
                    ~sched_candidates.index.isin(event_candidates.index)
                ]
            for pnl_val in sched_candidates[effective_pnl_col].values:
                if pd.isna(pnl_val):
                    continue
                if not pm.can_trade():
                    break
                trade_pnl = pm.record_trade(float(pnl_val), lots)
                daily_pnl += trade_pnl
                trades_placed += 1

        status = "traded" if trades_placed > 0 else ("rally_skip" if skip_scheduled else "no_candidates")
        curve.append(DayRecord(
            day=day, equity=pm.equity, daily_pnl=daily_pnl,
            n_trades=trades_placed, lots=lots, status=status,
            month_start_equity=pm.month_start_equity,
            event_signals=",".join(signals) if signals else "",
        ))

        if trades_placed > 0:
            _consecutive_losses = _consecutive_losses + 1 if daily_pnl < 0 else 0

    # --- Compute summary metrics ---
    cap = pc.starting_capital
    if not curve:
        return BacktestResult(
            config=config, label=label, curve=[],
            final_equity=cap, total_return_pct=0.0,
            annualised_return_pct=0.0, max_drawdown_pct=0.0,
            trough=cap, total_trades=0, days_traded=0,
            days_stopped=0, win_days=0, sharpe=0.0,
            monthly=pd.DataFrame(),
        )

    ec_df = pd.DataFrame([vars(r) for r in curve])
    final = pm.equity
    n_days = len(days)

    cummax = ec_df["equity"].cummax()
    dd_series = (cummax - ec_df["equity"]) / cummax
    max_dd_pct = float(dd_series.max()) * 100 if len(dd_series) > 0 else 0

    traded_mask = ec_df["n_trades"] > 0
    stopped_mask = ec_df["status"] == "monthly_stop"
    days_traded = int(traded_mask.sum())
    days_stopped = int(stopped_mask.sum())
    win_days = int((ec_df.loc[traded_mask, "daily_pnl"] > 0).sum()) if days_traded > 0 else 0

    all_pnl = ec_df["daily_pnl"]
    sharpe = 0.0
    if len(all_pnl) > 1 and all_pnl.std() > 0:
        sharpe = float(all_pnl.mean() / all_pnl.std() * np.sqrt(252))

    total_ret = (final / cap - 1) * 100 if cap > 0 else 0
    ann_ret = ((final / cap) ** (252 / max(n_days, 1)) - 1) * 100 if cap > 0 else 0

    # Monthly breakdown
    ec_df["month"] = pd.to_datetime(ec_df["day"]).dt.to_period("M")
    monthly_rows: list[dict[str, Any]] = []
    running_eq = cap
    for month, grp in ec_df.groupby("month"):
        m_pnl = grp["daily_pnl"].sum()
        m_ret = m_pnl / running_eq * 100 if running_eq > 0 else 0
        m_traded = (grp["n_trades"] > 0).sum()
        m_stopped = (grp["status"] == "monthly_stop").sum()
        m_win = (grp.loc[grp["n_trades"] > 0, "daily_pnl"] > 0).sum()
        m_worst = grp["daily_pnl"].min()
        m_end = grp["equity"].iloc[-1]
        m_lots = grp.loc[grp["n_trades"] > 0, "lots"]
        m_avg_lots = float(m_lots.mean()) if len(m_lots) > 0 else 0
        m_events = (grp["event_signals"] != "").sum()

        monthly_rows.append({
            "month": str(month), "pnl": m_pnl, "return_pct": m_ret,
            "end_equity": m_end, "avg_lots": m_avg_lots,
            "days_traded": int(m_traded), "days_stopped": int(m_stopped),
            "win_days": int(m_win), "worst_day": m_worst, "event_days": int(m_events),
        })
        running_eq = m_end

    return BacktestResult(
        config=config, label=label, curve=curve,
        final_equity=final, total_return_pct=total_ret,
        annualised_return_pct=ann_ret, max_drawdown_pct=max_dd_pct,
        trough=float(ec_df["equity"].min()),
        total_trades=int(ec_df["n_trades"].sum()),
        days_traded=days_traded, days_stopped=days_stopped,
        win_days=win_days, sharpe=sharpe,
        monthly=pd.DataFrame(monthly_rows),
    )


# ===================================================================
# Grid-search optimizer
# ===================================================================


def _build_optimizer_grid() -> list[FullConfig]:
    """Generate all valid parameter combinations for the exhaustive search.

    Returns
    -------
    List of FullConfig objects covering the full parameter space.
    Invalid combos (e.g. event params when event is disabled) are pruned.
    """
    configs: list[FullConfig] = []

    portfolio_grid = list(itertools.product(
        [20_000],                          # starting_capital
        [1, 2, 3],                         # max_trades_per_day
        [0.10, 0.15, 0.20, None],         # monthly_drawdown_limit
        [10_000, 20_000, 999_999],         # lot_per_equity
        [True, False],                     # calls_only
        [None],                            # min_dte (None = no filter)
        [None],                            # max_delta (None = no filter)
    ))

    # No-event configs
    for cap, trades, mstop, lpe, co, mdte, mdelta in portfolio_grid:
        pc = PortfolioConfig(
            starting_capital=cap, max_trades_per_day=trades,
            monthly_drawdown_limit=mstop, lot_per_equity=lpe,
            calls_only=co, min_dte=mdte, max_delta=mdelta,
        )
        configs.append(FullConfig(portfolio=pc, event=EventConfig(enabled=False)))

    # Event configs (only meaningful subsets)
    event_grid = list(itertools.product(
        ["shared", "separate"],            # budget_mode
        [1, 2],                            # max_event_trades
        [-0.005, -0.01, -0.015],           # spx_drop_threshold
        ["puts", "best"],                  # side_preference
        [3, 5],                            # event min_dte
        [5, 7, 10],                        # event max_dte
        [0.10, 0.15],                      # event min_delta
        [0.20, 0.25],                      # event max_delta
        [True, False],                     # rally_avoidance
    ))

    # Pair event configs with a focused set of portfolio configs
    portfolio_for_events = list(itertools.product(
        [20_000],                          # starting_capital
        [2, 3],                            # max_trades_per_day
        [0.15],                            # monthly_drawdown_limit
        [10_000],                          # lot_per_equity
        [True, False],                     # calls_only
    ))

    for (cap, trades, mstop, lpe, co) in portfolio_for_events:
        for (bmode, evt_max, drop_thr, side, emin_dte, emax_dte,
             emin_d, emax_d, rally) in event_grid:

            if emin_dte > emax_dte:
                continue
            if emin_d > emax_d:
                continue
            if bmode == "shared" and evt_max > 1:
                continue  # shared mode ignores max_event_trades

            pc = PortfolioConfig(
                starting_capital=cap, max_trades_per_day=trades,
                monthly_drawdown_limit=mstop, lot_per_equity=lpe,
                calls_only=co,
            )
            evc = EventConfig(
                enabled=True, budget_mode=bmode,
                max_event_trades=evt_max,
                spx_drop_threshold=drop_thr,
                side_preference=side,
                min_dte=emin_dte, max_dte=emax_dte,
                min_delta=emin_d, max_delta=emax_d,
                rally_avoidance=rally,
            )
            configs.append(FullConfig(portfolio=pc, event=evc))

    return configs


# TP / SL / regime values for the staged optimizer
OPTIMIZER_TP_VALUES = [0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
OPTIMIZER_SL_VALUES: list[float | None] = [None, 1.5, 2.0, 3.0]


def _build_staged_grid_stage1() -> list[FullConfig]:
    """Stage 1: Sweep trading params with fixed portfolio baseline.

    Fixes portfolio to current best (2/day, 15% DD, gradual sizing, calls-only)
    and sweeps all TradingConfig dimensions.
    """
    configs: list[FullConfig] = []

    trading_grid = itertools.product(
        OPTIMIZER_TP_VALUES,                # tp_pct
        OPTIMIZER_SL_VALUES,                # sl_mult
        [None, 30.0, 35.0],                # max_vix
        [None, 1.05, 1.10],                # max_term_structure
        [True, False],                      # avoid_opex
        [None, 10.0],                       # width_filter (None=all, 10=original)
        [None, 1, 2],                       # entry_count
        [None, 3, 5],                       # min_dte (via PortfolioConfig)
        [None, 0.20],                       # max_delta (via PortfolioConfig)
    )

    for (tp, sl, mvix, mts, aopex, wf, ec_n, mdte, mdelta) in trading_grid:
        tc = TradingConfig(
            tp_pct=tp, sl_mult=sl, max_vix=mvix,
            max_term_structure=mts, avoid_opex=aopex,
            width_filter=wf, entry_count=ec_n,
        )
        pc_with_dte = PortfolioConfig(
            starting_capital=20_000, max_trades_per_day=2,
            monthly_drawdown_limit=0.15, lot_per_equity=10_000,
            calls_only=True, min_dte=mdte, max_delta=mdelta,
        )
        configs.append(FullConfig(portfolio=pc_with_dte, trading=tc,
                                  event=EventConfig(enabled=False)))

    return configs


_StagedTradingWinner = tuple[TradingConfig, int | None, float | None]
"""(TradingConfig, best_min_dte, best_max_delta) from stage 1."""


def _build_staged_grid_stage2(
    top_trading_configs: list[_StagedTradingWinner],
) -> list[FullConfig]:
    """Stage 2: Sweep portfolio params using top trading configs from Stage 1.

    Each winner carries the best (min_dte, max_delta) from stage 1 so
    portfolio sweeping preserves DTE/delta filtering.
    """
    configs: list[FullConfig] = []

    portfolio_grid = itertools.product(
        [20_000],                          # starting_capital
        [1, 2, 3],                         # max_trades_per_day
        [0.10, 0.15, 0.20, None],         # monthly_drawdown_limit
        [10_000, 20_000, 999_999],         # lot_per_equity
        [True, False],                     # calls_only
    )

    for (cap, trades, mstop, lpe, co) in portfolio_grid:
        for tc, mdte, mdelta in top_trading_configs:
            pc = PortfolioConfig(
                starting_capital=cap, max_trades_per_day=trades,
                monthly_drawdown_limit=mstop, lot_per_equity=lpe,
                calls_only=co, min_dte=mdte, max_delta=mdelta,
            )
            configs.append(FullConfig(portfolio=pc, trading=tc,
                                      event=EventConfig(enabled=False)))

    return configs


def _build_staged_grid_stage3(
    top_combined: list[tuple[PortfolioConfig, TradingConfig]],
) -> list[FullConfig]:
    """Stage 3: Sweep event params using top portfolio+trading combos.

    Includes broader threshold sweeps for vix_spike_threshold,
    vix_elevated_threshold, spx_drop_2d_threshold, and signal_mode.
    """
    configs: list[FullConfig] = []

    event_grid = itertools.product(
        ["any", "spx_and_vix"],            # signal_mode
        ["shared", "separate"],            # budget_mode
        [1, 2],                            # max_event_trades
        [-0.005, -0.01, -0.015, -0.02, -0.03, -0.05],  # spx_drop_threshold
        [-0.01, -0.02, -0.03, -0.05],     # spx_drop_2d_threshold
        [0.10, 0.20],                      # vix_spike_threshold
        [20.0, 30.0],                      # vix_elevated_threshold
        ["puts", "best"],                  # side_preference
        [3, 5],                            # event min_dte
        [5, 7, 10],                        # event max_dte
        [0.10, 0.15],                      # event min_delta
        [0.20, 0.25],                      # event max_delta
        [True, False],                     # rally_avoidance
    )

    event_combos = list(event_grid)

    for (pc, tc) in top_combined:
        for (sig_mode, bmode, evt_max, drop_thr, drop_2d_thr,
             vix_spike_thr, vix_elev_thr,
             side, emin_dte, emax_dte,
             emin_d, emax_d, rally) in event_combos:

            if emin_dte > emax_dte:
                continue
            if emin_d > emax_d:
                continue
            if bmode == "shared" and evt_max > 1:
                continue

            evc = EventConfig(
                enabled=True,
                signal_mode=sig_mode,
                budget_mode=bmode,
                max_event_trades=evt_max,
                spx_drop_threshold=drop_thr,
                spx_drop_2d_threshold=drop_2d_thr,
                vix_spike_threshold=vix_spike_thr,
                vix_elevated_threshold=vix_elev_thr,
                side_preference=side,
                min_dte=emin_dte, max_dte=emax_dte,
                min_delta=emin_d, max_delta=emax_d,
                rally_avoidance=rally,
            )
            configs.append(FullConfig(portfolio=pc, trading=tc, event=evc))

    return configs


EVENT_ONLY_TP_VALUES = [0.50, 0.60, 0.70, 0.80]
EVENT_ONLY_SL_VALUES: list[float | None] = [None, 2.0, 3.0]


def _build_event_only_grid() -> list[FullConfig]:
    """Build a grid for event-only strategies that ONLY trade on SPX drop days.

    Sets ``EventConfig.event_only=True`` which suppresses all scheduled trades
    in ``run_backtest``, combined with wide SPX drop thresholds [-0.5% to -5%]
    and event-driven put credit spread parameters.  The result is a pure
    "sell puts after a market drop" strategy.
    """
    configs: list[FullConfig] = []

    grid = itertools.product(
        EVENT_ONLY_TP_VALUES,                          # tp_pct
        EVENT_ONLY_SL_VALUES,                          # sl_mult
        [-0.005, -0.01, -0.015, -0.02, -0.03, -0.05], # spx_drop_threshold (1d)
        [-0.01, -0.02, -0.03, -0.05],                 # spx_drop_2d_threshold
        ["puts", "best"],                              # side_preference
        [3, 5],                                        # event min_dte
        [5, 7, 10],                                    # event max_dte
        [0.10, 0.15],                                  # event min_delta
        [0.20, 0.25],                                  # event max_delta
        [1, 2, 3],                                     # max_event_trades
        [True, False],                                 # rally_avoidance
        [30.0, None],                                  # max_vix
    )

    for (tp, sl, drop_1d, drop_2d, side,
         emin_dte, emax_dte, emin_d, emax_d,
         evt_max, rally, mvix) in grid:

        if emin_dte > emax_dte:
            continue
        if emin_d > emax_d:
            continue

        tc = TradingConfig(
            tp_pct=tp, sl_mult=sl, max_vix=mvix,
            avoid_opex=True, width_filter=10.0,
        )
        pc = PortfolioConfig(
            starting_capital=20_000,
            max_trades_per_day=3,
            monthly_drawdown_limit=0.15,
            lot_per_equity=20_000,
            calls_only=False,
        )
        evc = EventConfig(
            enabled=True,
            event_only=True,
            signal_mode="any",
            budget_mode="shared",
            max_event_trades=evt_max,
            spx_drop_threshold=drop_1d,
            spx_drop_2d_threshold=drop_2d,
            side_preference=side,
            min_dte=emin_dte, max_dte=emax_dte,
            min_delta=emin_d, max_delta=emax_d,
            rally_avoidance=rally,
        )
        configs.append(FullConfig(portfolio=pc, trading=tc, event=evc))

    return configs


def run_event_only_optimizer(
    df: pd.DataFrame,
    daily_signals: pd.DataFrame,
    output_csv: Path = RESULTS_CSV,
) -> pd.DataFrame:
    """Run a focused optimizer sweep for event-only (SPX drop) strategies.

    Builds a grid of configs that only trade on SPX drop days, sweeping
    drop thresholds from -0.5% to -5%, DTE/delta ranges, TP/SL, and
    event parameters.  Results are appended to any existing results CSV
    so they can be analyzed alongside the staged optimizer output.

    Parameters
    ----------
    df : Training candidates DataFrame.
    daily_signals : Precomputed daily signal features.
    output_csv : Path to write/append results CSV.

    Returns
    -------
    DataFrame with event-only optimizer results.
    """
    precompute_pnl_columns(df, EVENT_ONLY_TP_VALUES, EVENT_ONLY_SL_VALUES)

    configs = _build_event_only_grid()
    print(f"  Event-only grid: {len(configs):,} configs", flush=True)
    results = _run_grid(configs, df, daily_signals, "Event-Only")

    if results.empty:
        logger.warning("Event-only optimizer produced no results")
        return results

    if output_csv.exists():
        try:
            existing = pd.read_csv(output_csv)
            combined = pd.concat([existing, results], ignore_index=True)
            combined.to_csv(output_csv, index=False)
            print(f"\n  Appended {len(results):,} event-only results to {output_csv}"
                  f" (total: {len(combined):,})", flush=True)
        except Exception as exc:
            logger.warning("Failed to append to existing CSV, writing standalone: %s", exc)
            fallback = output_csv.with_stem(output_csv.stem + "_event_only")
            results.to_csv(fallback, index=False)
            print(f"\n  Exported {len(results):,} event-only results to {fallback}",
                  flush=True)
    else:
        results.to_csv(output_csv, index=False)
        print(f"\n  Exported {len(results):,} event-only results to {output_csv}",
              flush=True)

    return results.sort_values("sharpe", ascending=False).reset_index(drop=True)


# ===================================================================
# SELECTIVE HIGH-WIN-RATE OPTIMIZER
# ===================================================================

SELECTIVE_TP_VALUES = [0.50, 0.60, 0.70]
SELECTIVE_SL_VALUES: list[float | None] = [None, 2.0, 3.0]


def _build_selective_grid() -> list[FullConfig]:
    """Build a grid targeting 90%+ win rate with selective trading.

    Key design choices for high win rate:
    - Tighter delta (0.05-0.15): further OTM spreads = higher probability of profit
    - Longer DTE (5-10): more time for theta decay to work
    - VIX filter (max 25-30): avoid chaotic high-vol environments
    - Conservative TP (0.50-0.70): take profit quickly
    - Width filter (10 pts): consistent spread geometry
    - Entry count filter (1-2): only best entries per day
    - Regime throttle enabled: reduce exposure in hostile environments
    """
    configs: list[FullConfig] = []

    grid = itertools.product(
        SELECTIVE_TP_VALUES,               # tp_pct
        SELECTIVE_SL_VALUES,               # sl_mult
        [20_000, 50_000, 100_000],         # starting_capital
        [1, 2],                            # max_trades_per_day
        [0.10, 0.15, 0.20],               # monthly_drawdown_limit
        [True, False],                     # calls_only
        [3, 5],                            # min_dte
        [0.15, 0.20],                      # max_delta
        [25.0, 30.0],                      # max_vix
        [True, False],                     # avoid_opex
        [1, 2],                            # entry_count
        [True, False],                     # regime_throttle enabled
    )

    for (tp, sl, cap, trades, mstop, co,
         mdte, mdelta, mvix, avopex, ecnt,
         rt_on) in grid:

        tc = TradingConfig(
            tp_pct=tp, sl_mult=sl, max_vix=mvix,
            avoid_opex=avopex, width_filter=10.0,
            entry_count=ecnt,
        )
        pc = PortfolioConfig(
            starting_capital=cap,
            max_trades_per_day=trades,
            monthly_drawdown_limit=mstop,
            lot_per_equity=cap,
            calls_only=co,
            min_dte=mdte,
            max_delta=mdelta,
        )
        rt = RegimeThrottle(enabled=rt_on)
        configs.append(FullConfig(
            portfolio=pc, trading=tc,
            event=EventConfig(enabled=False),
            regime=rt,
        ))

    return configs


def run_selective_optimizer(
    df: pd.DataFrame,
    daily_signals: pd.DataFrame,
    output_csv: Path = RESULTS_CSV,
) -> pd.DataFrame:
    """Run a focused optimizer sweep targeting high win-rate selective strategies.

    Uses conservative parameters (tight delta, long DTE, VIX filter) to find
    configs that achieve 90%+ day win-rate with at least 50 trades.  Results
    are appended to any existing results CSV.

    Parameters
    ----------
    df : Training candidates DataFrame.
    daily_signals : Precomputed daily signal features.
    output_csv : Path to write/append results CSV.

    Returns
    -------
    DataFrame with selective optimizer results.
    """
    precompute_pnl_columns(df, SELECTIVE_TP_VALUES, SELECTIVE_SL_VALUES)

    configs = _build_selective_grid()
    print(f"  Selective grid: {len(configs):,} configs", flush=True)
    results = _run_grid(configs, df, daily_signals, "Selective-HiWR")

    if results.empty:
        logger.warning("Selective optimizer produced no results")
        return results

    if output_csv.exists():
        try:
            existing = pd.read_csv(output_csv)
            combined = pd.concat([existing, results], ignore_index=True)
            combined.to_csv(output_csv, index=False)
            print(f"\n  Appended {len(results):,} selective results to {output_csv}"
                  f" (total: {len(combined):,})", flush=True)
        except Exception as exc:
            logger.warning("Failed to append to existing CSV, writing standalone: %s", exc)
            fallback = output_csv.with_stem(output_csv.stem + "_selective")
            results.to_csv(fallback, index=False)
            print(f"\n  Exported {len(results):,} selective results to {fallback}",
                  flush=True)
    else:
        results.to_csv(output_csv, index=False)
        print(f"\n  Exported {len(results):,} selective results to {output_csv}",
              flush=True)

    hi_wr = results[
        (results["win_rate"] >= 0.90) & (results["total_trades"] >= 50)
    ]
    print(f"\n  Configs with 90%+ WR and 50+ trades: {len(hi_wr):,}", flush=True)
    if not hi_wr.empty:
        best = hi_wr.nlargest(5, "sharpe")
        print("  Top 5 high-WR configs:", flush=True)
        for _, r in best.iterrows():
            print(f"    sharpe={r['sharpe']:.2f}  WR={r['win_rate']:.1%}  "
                  f"ret={r['return_pct']:.0f}%  DD={r['max_dd_pct']:.1f}%  "
                  f"trades={int(r['total_trades'])}  "
                  f"tp={r['t_tp_pct']}  sl={r['t_sl_mult']}  "
                  f"vix<={r['t_max_vix']}", flush=True)

    return results.sort_values("sharpe", ascending=False).reset_index(drop=True)


def _run_grid(
    configs: list[FullConfig],
    df: pd.DataFrame,
    daily_signals: pd.DataFrame,
    stage_name: str,
) -> pd.DataFrame:
    """Execute a list of configs and return results DataFrame."""
    print(f"\n[{stage_name}] {len(configs):,} configurations to evaluate", flush=True)
    rows: list[dict[str, Any]] = []
    t0 = time.time()

    # Pre-compute day selections per unique (width_filter, entry_count)
    precomp_cache: dict[tuple, dict] = {}

    for i, cfg in enumerate(configs):
        tc = cfg.trading
        cache_key = (tc.width_filter, tc.entry_count)
        if cache_key not in precomp_cache:
            precomp_cache[cache_key] = _precompute_day_selections(
                df, width_filter=tc.width_filter, entry_count=tc.entry_count,
            )

        result = run_backtest(
            df, daily_signals, cfg, day_precomp=precomp_cache[cache_key],
        )
        row = cfg.flat_dict()
        row.update({
            "final_equity": result.final_equity,
            "return_pct": result.total_return_pct,
            "ann_return_pct": result.annualised_return_pct,
            "max_dd_pct": result.max_drawdown_pct,
            "trough": result.trough,
            "sharpe": result.sharpe,
            "total_trades": result.total_trades,
            "days_traded": result.days_traded,
            "days_stopped": result.days_stopped,
            "win_days": result.win_days,
            "win_rate": result.win_days / max(result.days_traded, 1),
        })
        rows.append(row)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(configs) - i - 1) / rate
            print(f"  {i+1:,}/{len(configs):,} done  "
                  f"({rate:.0f}/sec, ~{remaining:.0f}s remaining)", flush=True)

    elapsed = time.time() - t0
    print(f"  [{stage_name}] Completed {len(configs):,} configs in {elapsed:.1f}s "
          f"({len(configs)/max(elapsed, 0.01):.0f}/sec)", flush=True)

    return pd.DataFrame(rows)


def run_optimizer(
    df: pd.DataFrame,
    daily_signals: pd.DataFrame,
    output_csv: Path = RESULTS_CSV,
) -> pd.DataFrame:
    """Run exhaustive grid search (original mode) and export results.

    Parameters
    ----------
    df : Training candidates DataFrame.
    daily_signals : Precomputed daily signal features.
    output_csv : Path to write the full results CSV.

    Returns
    -------
    DataFrame with one row per config, sorted by Sharpe descending.
    """
    configs = _build_optimizer_grid()
    results_df = _run_grid(configs, df, daily_signals, "Optimizer")
    results_df.to_csv(output_csv, index=False)
    print(f"\n  Full results exported to {output_csv}", flush=True)
    return results_df.sort_values("sharpe", ascending=False).reset_index(drop=True)


def run_staged_optimizer(
    df: pd.DataFrame,
    daily_signals: pd.DataFrame,
    output_csv: Path = RESULTS_CSV,
    top_n_trading: int = 10,
    top_n_combined: int = 5,
) -> pd.DataFrame:
    """Run 3-stage optimizer: trading params -> portfolio params -> event params.

    Stage 1 sweeps all TradingConfig dimensions with fixed portfolio baseline.
    Stage 2 takes the top trading configs and sweeps portfolio params.
    Stage 3 takes the top combined configs and sweeps event params.

    Parameters
    ----------
    df : Training candidates DataFrame (must have multi-TP trajectory columns).
    daily_signals : Precomputed daily signal features.
    output_csv : Path to write combined results CSV.
    top_n_trading : Number of top trading configs to carry into Stage 2.
    top_n_combined : Number of top combined configs to carry into Stage 3.

    Returns
    -------
    Combined DataFrame with all stage results, sorted by Sharpe descending.
    """
    # Pre-compute PnL columns for all (tp, sl) combos
    precompute_pnl_columns(df, OPTIMIZER_TP_VALUES, OPTIMIZER_SL_VALUES)

    # --- Stage 1: Trading param sweep ---
    s1_configs = _build_staged_grid_stage1()
    s1_results = _run_grid(s1_configs, df, daily_signals, "Stage-1 Trading")
    s1_sorted = s1_results.sort_values("sharpe", ascending=False)

    # Extract top trading configs (carrying min_dte / max_delta from stage 1)
    top_trading: list[_StagedTradingWinner] = []
    seen: set[str] = set()
    for _, row in s1_sorted.iterrows():
        key = (f"{row.get('t_tp_pct')}_{row.get('t_sl_mult')}_{row.get('t_max_vix')}_"
               f"{row.get('t_max_term_structure')}_{row.get('t_avoid_opex')}_"
               f"{row.get('t_width_filter')}_{row.get('t_entry_count')}_"
               f"{row.get('p_min_dte')}_{row.get('p_max_delta')}")
        if key in seen:
            continue
        seen.add(key)

        top_trading.append((
            TradingConfig(
                tp_pct=float(row.get("t_tp_pct", 0.50)),
                sl_mult=_opt_val(row.get("t_sl_mult")),
                max_vix=_opt_val(row.get("t_max_vix")),
                max_term_structure=_opt_val(row.get("t_max_term_structure")),
                avoid_opex=_safe_bool(row.get("t_avoid_opex")),
                width_filter=_opt_val(row.get("t_width_filter")),
                entry_count=int(row.get("t_entry_count")) if pd.notna(row.get("t_entry_count")) else None,
            ),
            int(row.get("p_min_dte")) if pd.notna(row.get("p_min_dte")) else None,
            float(row.get("p_max_delta")) if pd.notna(row.get("p_max_delta")) else None,
        ))
        if len(top_trading) >= top_n_trading:
            break

    print(f"\n  Top {len(top_trading)} trading configs selected for Stage 2", flush=True)

    # --- Stage 2: Portfolio param sweep ---
    s2_configs = _build_staged_grid_stage2(top_trading)
    s2_results = _run_grid(s2_configs, df, daily_signals, "Stage-2 Portfolio")
    s2_sorted = s2_results.sort_values("sharpe", ascending=False)

    top_combined: list[tuple[PortfolioConfig, TradingConfig]] = []
    seen_combined: set[str] = set()
    _s2_key_cols = [c for c in s2_sorted.columns if c.startswith(("p_", "t_"))]
    for _, row in s2_sorted.iterrows():
        key = "|".join(str(row.get(c)) for c in _s2_key_cols)
        if key in seen_combined:
            continue
        seen_combined.add(key)

        dd_limit = _opt_val(row.get("p_monthly_drawdown_limit"))
        pc = PortfolioConfig(
            starting_capital=float(row.get("p_starting_capital", 20_000)),
            max_trades_per_day=int(row.get("p_max_trades_per_day", 2)),
            monthly_drawdown_limit=dd_limit,
            lot_per_equity=float(row.get("p_lot_per_equity", 10_000)),
            calls_only=_safe_bool(row.get("p_calls_only"), True),
            min_dte=int(row.get("p_min_dte")) if pd.notna(row.get("p_min_dte")) else None,
            max_delta=float(row.get("p_max_delta")) if pd.notna(row.get("p_max_delta")) else None,
        )
        tc = TradingConfig(
            tp_pct=float(row.get("t_tp_pct", 0.50)),
            sl_mult=_opt_val(row.get("t_sl_mult")),
            max_vix=_opt_val(row.get("t_max_vix")),
            max_term_structure=_opt_val(row.get("t_max_term_structure")),
            avoid_opex=_safe_bool(row.get("t_avoid_opex")),
            width_filter=_opt_val(row.get("t_width_filter")),
            entry_count=int(row.get("t_entry_count")) if pd.notna(row.get("t_entry_count")) else None,
        )
        top_combined.append((pc, tc))
        if len(top_combined) >= top_n_combined:
            break

    print(f"\n  Top {len(top_combined)} combined configs selected for Stage 3", flush=True)

    # --- Stage 3: Event param sweep ---
    s3_configs = _build_staged_grid_stage3(top_combined)
    s3_results = _run_grid(s3_configs, df, daily_signals, "Stage-3 Event")

    # Combine all stages
    all_results = pd.concat([s1_results, s2_results, s3_results], ignore_index=True)
    all_results.to_csv(output_csv, index=False)
    print(f"\n  Combined results ({len(all_results):,} rows) exported to {output_csv}",
          flush=True)

    return all_results.sort_values("sharpe", ascending=False).reset_index(drop=True)


# ===================================================================
# Preset comparison configs
# ===================================================================


def _build_comparison_configs() -> list[tuple[str, FullConfig]]:
    """Hand-picked configs for quick side-by-side comparison."""
    return [
        ("$20k | 2/day | 15% stop | gradual | calls", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=0.15, lot_per_equity=10_000, calls_only=True),
            event=EventConfig(enabled=False),
        )),
        ("$20k | 2/day | 20% stop | gradual | calls", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=0.20, lot_per_equity=10_000, calls_only=True),
            event=EventConfig(enabled=False),
        )),
        ("$20k | 2/day | no stop  | gradual | calls", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=None, lot_per_equity=10_000, calls_only=True),
            event=EventConfig(enabled=False),
        )),
        ("$20k | 1/day | 15% stop | gradual | calls", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=1,
                                      monthly_drawdown_limit=0.15, lot_per_equity=10_000, calls_only=True),
            event=EventConfig(enabled=False),
        )),
        ("$20k | 2/day | 15% stop | fixed 1 | calls", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=0.15, lot_per_equity=999_999, calls_only=True),
            event=EventConfig(enabled=False),
        )),
        ("$20k | 2/day | 15% stop | gradual | both sides", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=0.15, lot_per_equity=10_000, calls_only=False),
            event=EventConfig(enabled=False),
        )),
        ("$20k | 2+1ev | 15% stop | shared  | drop puts", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=0.15, lot_per_equity=10_000, calls_only=True),
            event=EventConfig(enabled=True, budget_mode="shared", spx_drop_threshold=-0.01,
                              side_preference="puts", min_dte=5, max_dte=7, min_delta=0.15, max_delta=0.25),
        )),
        ("$20k | 2+1ev | 15% stop | separate | drop puts", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=0.15, lot_per_equity=10_000, calls_only=True),
            event=EventConfig(enabled=True, budget_mode="separate", max_event_trades=1,
                              spx_drop_threshold=-0.01, side_preference="puts",
                              min_dte=5, max_dte=7, min_delta=0.15, max_delta=0.25),
        )),
        ("$20k | 2/day | 15% stop | gradual | calls + rally avoid", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=0.15, lot_per_equity=10_000, calls_only=True),
            event=EventConfig(enabled=True, rally_avoidance=True, rally_threshold=0.01),
        )),
    ]


# ===================================================================
# Display helpers
# ===================================================================


def print_summary(result: BacktestResult) -> None:
    """Print a one-block summary of a backtest result."""
    r = result
    cap = r.config.portfolio.starting_capital
    print(f"\n{'=' * 80}")
    print(f"  {r.label}")
    print(f"{'=' * 80}")
    print(f"  Start: ${cap:,.0f}  ->  Final: ${r.final_equity:,.0f}  "
          f"({r.total_return_pct:+.1f}%)")
    print(f"  Annualised: {r.annualised_return_pct:+.0f}%  |  "
          f"Max DD: {r.max_drawdown_pct:.1f}%  |  Trough: ${r.trough:,.0f}")
    print(f"  Sharpe: {r.sharpe:.2f}  |  "
          f"Trades: {r.total_trades}  |  "
          f"Traded {r.days_traded}d  Stopped {r.days_stopped}d  "
          f"Win {r.win_days}/{r.days_traded} "
          f"({r.win_days / max(r.days_traded, 1) * 100:.0f}%)")


def print_monthly(result: BacktestResult) -> None:
    """Print month-by-month breakdown."""
    print(f"\n{'Month':>8}  {'PnL':>10}  {'Return':>8}  {'Equity':>10}  "
          f"{'Lots':>5}  {'Traded':>7}  {'Stopped':>8}  {'Events':>7}  {'Worst Day':>10}")
    print("-" * 90)
    for _, row in result.monthly.iterrows():
        print(f"{row['month']:>8}  ${row['pnl']:>9,.0f}  "
              f"{row['return_pct']:>+6.1f}%  ${row['end_equity']:>9,.0f}  "
              f"{row['avg_lots']:>5.1f}  {int(row['days_traded']):>5}d  "
              f"{int(row['days_stopped']):>6}d  {int(row['event_days']):>5}d  "
              f"${row['worst_day']:>9,.0f}")


def print_comparison_table(results: list[BacktestResult]) -> None:
    """Print a compact comparison of multiple configs."""
    print(f"\n{'=' * 115}")
    print("  CONFIGURATION COMPARISON")
    print(f"{'=' * 115}")
    print(f"{'Configuration':<50} {'Final':>10} {'Return':>8} "
          f"{'MaxDD':>6} {'Trough':>10} {'Sharpe':>7} "
          f"{'Trades':>7} {'Stopped':>8}")
    print("-" * 115)
    for r in results:
        print(f"{r.label:<50} ${r.final_equity:>9,.0f} "
              f"{r.total_return_pct:>+6.0f}% "
              f"{r.max_drawdown_pct:>5.0f}% "
              f"${r.trough:>9,.0f} "
              f"{r.sharpe:>7.2f} "
              f"{r.total_trades:>7} "
              f"{r.days_stopped:>6}d")


def print_optimizer_top(results_df: pd.DataFrame, metric: str, n: int = 10) -> None:
    """Print the top-N configs from the optimizer by a given metric."""
    asc = metric == "max_dd_pct"
    top = results_df.sort_values(metric, ascending=asc).head(n)

    print(f"\n{'=' * 115}")
    print(f"  TOP {n} BY {metric.upper()}")
    print(f"{'=' * 115}")
    cols = ["p_max_trades_per_day", "p_monthly_drawdown_limit", "p_lot_per_equity",
            "p_calls_only", "t_tp_pct", "t_sl_mult", "t_max_vix",
            "t_avoid_opex", "t_width_filter", "t_entry_count",
            "e_enabled", "e_budget_mode", "e_spx_drop_threshold",
            "e_rally_avoidance", "final_equity", "return_pct", "max_dd_pct",
            "sharpe", "total_trades", "days_stopped"]
    display_cols = [c for c in cols if c in top.columns]
    print(top[display_cols].to_string(index=False))


# ===================================================================
# Analysis: parameter importance, Pareto frontier, robustness
# ===================================================================


PARETO_CSV = DATA_DIR / "pareto_frontier.csv"
WALKFORWARD_CSV = DATA_DIR / "walkforward_results.csv"

# Parameters that vary in the optimizer grid and are worth analyzing
ANALYSIS_PARAMS = [
    "p_max_trades_per_day",
    "p_monthly_drawdown_limit",
    "p_lot_per_equity",
    "p_calls_only",
    "p_min_dte",
    "p_max_delta",
    "t_tp_pct",
    "t_sl_mult",
    "t_max_vix",
    "t_max_term_structure",
    "t_avoid_opex",
    "t_prefer_event_days",
    "t_width_filter",
    "t_entry_count",
    "e_enabled",
    "e_signal_mode",
    "e_budget_mode",
    "e_spx_drop_threshold",
    "e_spx_drop_2d_threshold",
    "e_vix_spike_threshold",
    "e_vix_elevated_threshold",
    "e_rally_avoidance",
    "e_side_preference",
]


def _deduplicate_results(rdf: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Collapse configs that produce identical outcome metrics.

    Parameters
    ----------
    rdf : Full optimizer results DataFrame.

    Returns
    -------
    Tuple of (deduplicated DataFrame keeping first of each group,
    original count, unique count).
    """
    metric_cols = ["final_equity", "return_pct", "max_dd_pct", "sharpe",
                   "total_trades", "days_traded", "days_stopped", "win_days"]
    rounded = rdf[metric_cols].round(4)
    dedup = rdf.loc[~rounded.duplicated(keep="first")]
    return dedup, len(rdf), len(dedup)


def _parameter_importance(rdf: pd.DataFrame) -> None:
    """Print grouped-mean tables showing how each parameter affects outcomes.

    For each parameter in ANALYSIS_PARAMS, groups the results by that
    parameter's values and prints the mean Sharpe, return, and max DD,
    plus the spread (max-min) across values.
    """
    metrics = ["sharpe", "return_pct", "max_dd_pct"]
    print(f"\n{'=' * 100}")
    print("  PARAMETER IMPORTANCE (mean metric by parameter value)")
    print(f"{'=' * 100}")

    importance: list[tuple[str, float]] = []

    for param in ANALYSIS_PARAMS:
        if param not in rdf.columns:
            continue
        grouped = rdf.groupby(param)[metrics].mean()
        if len(grouped) < 2:
            continue

        sharpe_spread = grouped["sharpe"].max() - grouped["sharpe"].min()
        importance.append((param, sharpe_spread))

        print(f"\n  {param} (Sharpe spread: {sharpe_spread:.2f})")
        print(f"  {'Value':<25} {'Sharpe':>8} {'Return%':>10} {'MaxDD%':>8}")
        print("  " + "-" * 55)
        for val, row in grouped.iterrows():
            val_str = str(val) if val is not None else "None"
            print(f"  {val_str:<25} {row['sharpe']:>8.2f} "
                  f"{row['return_pct']:>+9.0f}% {row['max_dd_pct']:>7.1f}%")

    importance.sort(key=lambda x: -x[1])
    print(f"\n{'=' * 100}")
    print("  PARAMETER RANKING (by Sharpe spread across values)")
    print(f"{'=' * 100}")
    for rank, (param, spread) in enumerate(importance, 1):
        print(f"  {rank}. {param:<35} spread = {spread:.2f}")


def extract_pareto_frontier(rdf: pd.DataFrame) -> pd.DataFrame:
    """Find Pareto-optimal configs: no other config has both higher Sharpe AND lower max DD.

    Parameters
    ----------
    rdf : Optimizer results DataFrame with ``sharpe`` and ``max_dd_pct`` columns.

    Returns
    -------
    DataFrame with only Pareto-optimal rows, sorted by Sharpe descending.
    """
    is_pareto = np.ones(len(rdf), dtype=bool)
    sharpe_vals = rdf["sharpe"].values
    dd_vals = rdf["max_dd_pct"].values

    for i in range(len(rdf)):
        if not is_pareto[i]:
            continue
        # A point is dominated if any other point has >= sharpe AND <= dd (and strictly better on at least one)
        for j in range(len(rdf)):
            if i == j or not is_pareto[j]:
                continue
            if sharpe_vals[j] >= sharpe_vals[i] and dd_vals[j] <= dd_vals[i]:
                if sharpe_vals[j] > sharpe_vals[i] or dd_vals[j] < dd_vals[i]:
                    is_pareto[i] = False
                    break

    return rdf[is_pareto].sort_values("sharpe", ascending=False).reset_index(drop=True)


def _print_pareto(pareto: pd.DataFrame) -> None:
    """Print the Pareto frontier in a readable table."""
    print(f"\n{'=' * 130}")
    print(f"  PARETO FRONTIER (Sharpe vs Max DD) -- {len(pareto)} optimal configs")
    print(f"{'=' * 130}")
    cols = ["p_max_trades_per_day", "p_monthly_drawdown_limit", "p_lot_per_equity",
            "p_calls_only", "t_tp_pct", "t_sl_mult", "t_max_vix",
            "t_avoid_opex", "t_width_filter",
            "e_enabled", "e_budget_mode", "e_spx_drop_threshold",
            "e_rally_avoidance", "final_equity", "return_pct", "max_dd_pct",
            "sharpe", "win_rate"]
    display_cols = [c for c in cols if c in pareto.columns]
    print(pareto[display_cols].to_string(index=False))


def _robustness_check(
    df: pd.DataFrame,
    daily_signals: pd.DataFrame,
    rdf: pd.DataFrame,
    top_n: int = 20,
) -> None:
    """Re-run top-N configs and report monthly consistency.

    Flags configs that depend on 1-2 monster months vs. those that
    are consistently profitable across the period.

    Parameters
    ----------
    df : Training candidates DataFrame.
    daily_signals : Precomputed daily signal features.
    rdf : Optimizer results DataFrame.
    top_n : Number of top configs to re-evaluate.
    """
    print(f"\n{'=' * 140}")
    print(f"  ROBUSTNESS CHECK (monthly consistency for top-{top_n} by Sharpe)")
    print(f"{'=' * 140}")

    top = rdf.sort_values("sharpe", ascending=False).head(top_n)

    precomp_cache: dict[tuple, dict] = {}

    print(f"  {'#':>3} {'Trades/d':>8} {'Stop':>6} {'Lots':>8} {'Calls':>6} "
          f"{'Events':>7} {'Sharpe':>7} {'Return':>8} {'MaxDD':>6} "
          f"{'Profit Mo':>10} {'Worst Mo':>10} {'Best Mo':>9} {'Consistency':>12}")
    print("  " + "-" * 135)

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        cfg = _row_to_config(row)
        key = (cfg.trading.width_filter, cfg.trading.entry_count)
        if key not in precomp_cache:
            precomp_cache[key] = _precompute_day_selections(
                df, width_filter=key[0], entry_count=key[1],
            )
        result = run_backtest(df, daily_signals, cfg, day_precomp=precomp_cache[key])
        m = result.monthly
        n_months = len(m)
        profitable_months = int((m["pnl"] > 0).sum())
        worst_month_pct = float(m["return_pct"].min())
        best_month_pct = float(m["return_pct"].max())
        consistency = profitable_months / max(n_months, 1)

        stop_str = f"{row.get('p_monthly_drawdown_limit', '')}"
        lots_str = "grad" if row.get("p_lot_per_equity", 0) < 100_000 else "fixed"
        evt_str = "yes" if row.get("e_enabled") else "no"

        print(f"  {rank:>3} {int(row.get('p_max_trades_per_day', 0)):>8} "
              f"{stop_str:>6} {lots_str:>8} {str(row.get('p_calls_only', '')):>6} "
              f"{evt_str:>7} {row['sharpe']:>7.2f} {row['return_pct']:>+7.0f}% "
              f"{row['max_dd_pct']:>5.0f}% "
              f"{profitable_months:>5}/{n_months:<4} "
              f"{worst_month_pct:>+9.1f}% {best_month_pct:>+8.1f}% "
              f"{consistency:>11.0%}")


def _row_to_config(row: pd.Series) -> FullConfig:
    """Reconstruct a FullConfig from an optimizer results CSV row.

    Parameters
    ----------
    row : One row from the optimizer results DataFrame.

    Returns
    -------
    FullConfig with portfolio, trading, and event settings from the row.
    """
    dd_limit = row.get("p_monthly_drawdown_limit")
    if pd.isna(dd_limit):
        dd_limit = None

    min_dte = row.get("p_min_dte")
    if pd.notna(min_dte):
        min_dte = int(min_dte)
    else:
        min_dte = None

    max_delta = row.get("p_max_delta")
    if pd.notna(max_delta):
        max_delta = float(max_delta)
    else:
        max_delta = None

    def _opt_float(key, default=None):
        """Extract a float from *row[key]*, returning *default* on NaN."""
        v = row.get(key)
        return float(v) if pd.notna(v) else default

    def _opt_int(key, default=None):
        """Extract an int from *row[key]*, returning *default* on NaN."""
        v = row.get(key)
        return int(v) if pd.notna(v) else default

    def _opt_bool(key, default=False):
        """Extract a bool from *row[key]*, returning *default* on NaN."""
        v = row.get(key, default)
        if pd.isna(v):
            return default
        return bool(v)

    pc = PortfolioConfig(
        starting_capital=float(row.get("p_starting_capital", 20_000)),
        max_trades_per_day=int(row.get("p_max_trades_per_day", 2)),
        monthly_drawdown_limit=dd_limit,
        lot_per_equity=float(row.get("p_lot_per_equity", 10_000)),
        calls_only=_opt_bool("p_calls_only", True),
        min_dte=min_dte,
        max_delta=max_delta,
    )

    tc = TradingConfig(
        tp_pct=float(row.get("t_tp_pct", 0.50)),
        sl_mult=_opt_float("t_sl_mult"),
        max_vix=_opt_float("t_max_vix"),
        max_term_structure=_opt_float("t_max_term_structure"),
        avoid_opex=_opt_bool("t_avoid_opex"),
        prefer_event_days=_opt_bool("t_prefer_event_days"),
        width_filter=_opt_float("t_width_filter"),
        entry_count=_opt_int("t_entry_count"),
    )

    ec = EventConfig(
        enabled=_opt_bool("e_enabled"),
        signal_mode=_opt_str(row.get("e_signal_mode"), "any"),
        budget_mode=_opt_str(row.get("e_budget_mode"), "shared"),
        max_event_trades=int(row.get("e_max_event_trades", 1)),
        spx_drop_threshold=float(row.get("e_spx_drop_threshold", -0.01)),
        spx_drop_2d_threshold=float(row.get("e_spx_drop_2d_threshold", -0.02)),
        spx_drop_min=_opt_float("e_spx_drop_min"),
        spx_drop_max=_opt_float("e_spx_drop_max"),
        vix_spike_threshold=float(row.get("e_vix_spike_threshold", 0.15)),
        vix_elevated_threshold=float(row.get("e_vix_elevated_threshold", 25.0)),
        term_inversion_threshold=float(row.get("e_term_inversion_threshold", 1.0)),
        side_preference=_opt_str(row.get("e_side_preference"), "puts"),
        min_dte=int(row.get("e_min_dte", 5)),
        max_dte=int(row.get("e_max_dte", 7)),
        min_delta=float(row.get("e_min_delta", 0.15)),
        max_delta=float(row.get("e_max_delta", 0.25)),
        rally_avoidance=_opt_bool("e_rally_avoidance"),
        rally_threshold=float(row.get("e_rally_threshold", 0.01)),
        event_only=_opt_bool("e_event_only"),
    )

    rt = RegimeThrottle(
        enabled=_opt_bool("r_enabled"),
        high_vix_threshold=float(row.get("r_high_vix_threshold", 30.0)),
        high_vix_multiplier=float(row.get("r_high_vix_multiplier", 0.5)),
        extreme_vix_threshold=float(row.get("r_extreme_vix_threshold", 40.0)),
        big_drop_threshold=float(row.get("r_big_drop_threshold", -0.02)),
        big_drop_multiplier=float(row.get("r_big_drop_multiplier", 0.5)),
        consecutive_loss_days=int(row.get("r_consecutive_loss_days", 3)),
        consecutive_loss_multiplier=float(row.get("r_consecutive_loss_multiplier", 0.5)),
    )
    return FullConfig(portfolio=pc, trading=tc, event=ec, regime=rt)


def run_analysis(results_csv: Path) -> pd.DataFrame:
    """Load optimizer CSV and print full analysis (A1-A4).

    Parameters
    ----------
    results_csv : Path to the optimizer results CSV.

    Returns
    -------
    The Pareto-frontier DataFrame (also exported to CSV).
    """
    rdf = pd.read_csv(results_csv)
    print(f"\nLoaded {len(rdf):,} optimizer results from {results_csv}")

    # A4: Dedup
    dedup, orig, unique = _deduplicate_results(rdf)
    print(f"  {unique} unique result profiles out of {orig} configs "
          f"({orig - unique} duplicates)")

    # A1: Parameter importance
    _parameter_importance(dedup)

    # Filter out configs with too few trades for statistical significance
    min_trades = 30
    before_len = len(dedup)
    dedup = dedup[dedup["total_trades"] >= min_trades]
    if len(dedup) < before_len:
        logger.info(
            "Filtered %d configs with < %d total trades",
            before_len - len(dedup), min_trades,
        )

    # A2: Pareto frontier (drop rows with missing Sharpe/DD to avoid NaN in dominance check)
    dedup = dedup.dropna(subset=["sharpe", "max_dd_pct"])
    pareto = extract_pareto_frontier(dedup)
    _print_pareto(pareto)
    PARETO_CSV.parent.mkdir(parents=True, exist_ok=True)
    pareto.to_csv(PARETO_CSV, index=False)
    print(f"\n  Pareto frontier exported to {PARETO_CSV}")

    return pareto


# ===================================================================
# Walk-forward validation
# ===================================================================

WALKFORWARD_WINDOWS = [
    {
        "name": "W1",
        "train_start": "2025-03-01", "train_end": "2025-08-31",
        "test_start": "2025-09-01", "test_end": "2025-11-30",
    },
    {
        "name": "W2",
        "train_start": "2025-06-01", "train_end": "2025-11-30",
        "test_start": "2025-12-01", "test_end": "2026-02-28",
    },
    {
        "name": "W3",
        "train_start": "2025-09-01", "train_end": "2026-02-28",
        "test_start": "2026-03-01", "test_end": "2026-04-30",
    },
]


def generate_auto_windows(
    min_day: str,
    max_day: str,
    *,
    train_months: int = 3,
    test_months: int = 1,
    step_months: int = 1,
) -> list[dict[str, str]]:
    """Auto-generate rolling walk-forward windows from a data date range.

    Produces non-overlapping test windows, each preceded by a contiguous
    training window.  Windows that would extend beyond the data range are
    truncated.  Windows with less than one month of training or test data
    are dropped.

    Parameters
    ----------
    min_day : str
        Earliest date in the dataset (ISO format ``YYYY-MM-DD``).
    max_day : str
        Latest date in the dataset (ISO format ``YYYY-MM-DD``).
    train_months : int
        Length of each training window in months.
    test_months : int
        Length of each test window in months.
    step_months : int
        Stride between successive windows in months.

    Returns
    -------
    list[dict[str, str]]
        List of window dicts with keys ``name``, ``train_start``,
        ``train_end``, ``test_start``, ``test_end``.
    """
    from dateutil.relativedelta import relativedelta

    start = pd.Timestamp(min_day).date()
    end = pd.Timestamp(max_day).date()

    windows: list[dict[str, str]] = []
    cursor = start
    idx = 1

    while True:
        train_start = cursor
        train_end_raw = cursor + relativedelta(months=train_months) - timedelta(days=1)
        test_start = train_end_raw + timedelta(days=1)
        test_end_raw = test_start + relativedelta(months=test_months) - timedelta(days=1)

        # Truncate to data range
        train_end = min(train_end_raw, end)
        test_end = min(test_end_raw, end)

        # Need at least 30 days of train and 15 days of test
        if (train_end - train_start).days < 30:
            break
        if test_start > end or (test_end - test_start).days < 15:
            break

        windows.append({
            "name": f"Auto-W{idx}",
            "train_start": str(train_start),
            "train_end": str(train_end),
            "test_start": str(test_start),
            "test_end": str(test_end),
        })
        idx += 1
        cursor += relativedelta(months=step_months)

        if cursor >= end:
            break

    return windows


def walkforward_split(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training candidates into train/test by date range.

    Parameters
    ----------
    df : Full training candidates DataFrame with ``day`` column.
    train_start, train_end : Inclusive date boundaries for training slice.
    test_start, test_end : Inclusive date boundaries for test slice.

    Returns
    -------
    Tuple of (train_df, test_df).
    """
    train = df[(df["day"] >= train_start) & (df["day"] <= train_end)]
    test = df[(df["day"] >= test_start) & (df["day"] <= test_end)]
    return train, test


def _run_window_optimizer(
    df_slice: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Run the full optimizer grid on a data slice and return top-N rows.

    Parameters
    ----------
    df_slice : Subset of training candidates for this window.
    top_n : Number of top configs to return.

    Returns
    -------
    DataFrame with top-N rows by Sharpe, including all config columns.
    """
    daily_signals = precompute_daily_signals(df_slice)
    configs = _build_optimizer_grid()

    precomp_cache: dict[tuple, dict] = {}
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        key = (cfg.trading.width_filter, cfg.trading.entry_count)
        if key not in precomp_cache:
            precomp_cache[key] = _precompute_day_selections(
                df_slice, width_filter=key[0], entry_count=key[1],
            )
        result = run_backtest(df_slice, daily_signals, cfg, day_precomp=precomp_cache[key])
        row = cfg.flat_dict()
        row.update({
            "final_equity": result.final_equity,
            "return_pct": result.total_return_pct,
            "ann_return_pct": result.annualised_return_pct,
            "max_dd_pct": result.max_drawdown_pct,
            "trough": result.trough,
            "sharpe": result.sharpe,
            "total_trades": result.total_trades,
            "days_traded": result.days_traded,
            "days_stopped": result.days_stopped,
            "win_days": result.win_days,
            "win_rate": result.win_days / max(result.days_traded, 1),
        })
        rows.append(row)

    results_df = pd.DataFrame(rows)
    return results_df.sort_values("sharpe", ascending=False).head(top_n).reset_index(drop=True)


def _config_signature(row: pd.Series) -> str:
    """Build a short human-readable label for a config row."""
    trades = int(row.get("p_max_trades_per_day", 0))
    stop = row.get("p_monthly_drawdown_limit")
    stop_s = f"{stop:.0%}" if pd.notna(stop) else "none"
    lots = "grad" if row.get("p_lot_per_equity", 0) < 100_000 else "fixed"
    calls = "C" if row.get("p_calls_only") else "B"
    evt = "E" if row.get("e_enabled") else ""
    bm = row.get("e_budget_mode", "")[0].upper() if row.get("e_enabled") else ""
    return f"{trades}/d|{stop_s}|{lots}|{calls}{evt}{bm}"


def _config_key(row: pd.Series) -> tuple:
    """Build a hashable key from the config columns of a results row.

    NaN values are normalized to None so that two otherwise-identical
    configs always produce equal keys (``NaN != NaN`` in Python).
    """
    return tuple(
        _opt_val(row.get(c)) for c in [
            "p_max_trades_per_day", "p_monthly_drawdown_limit", "p_lot_per_equity",
            "p_calls_only", "p_min_dte", "p_max_delta",
            "t_tp_pct", "t_sl_mult", "t_max_vix", "t_max_term_structure",
            "t_avoid_opex", "t_prefer_event_days", "t_width_filter", "t_entry_count",
            "e_enabled", "e_signal_mode", "e_budget_mode", "e_max_event_trades",
            "e_spx_drop_threshold", "e_spx_drop_2d_threshold",
            "e_spx_drop_min", "e_spx_drop_max",
            "e_vix_spike_threshold", "e_vix_elevated_threshold",
            "e_term_inversion_threshold",
            "e_side_preference", "e_min_dte", "e_max_dte",
            "e_min_delta", "e_max_delta", "e_rally_avoidance", "e_rally_threshold",
            "e_event_only",
            "r_enabled", "r_high_vix_threshold", "r_high_vix_multiplier",
            "r_extreme_vix_threshold", "r_big_drop_threshold", "r_big_drop_multiplier",
            "r_consecutive_loss_days", "r_consecutive_loss_multiplier",
        ]
    )


def run_walkforward(
    df: pd.DataFrame,
    output_csv: Path = WALKFORWARD_CSV,
    top_n: int = 10,
    *,
    windows: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Run rolling walk-forward validation across all windows.

    For each window, runs the full optimizer on the train slice, then
    evaluates the top-N train configs on the test slice (out-of-sample).

    Parameters
    ----------
    df : Full training candidates DataFrame.
    output_csv : Path to export detailed results.
    top_n : Number of top configs per window.
    windows : Optional custom window list.  When ``None`` the hard-coded
        ``WALKFORWARD_WINDOWS`` are used.  Pass the output of
        ``generate_auto_windows()`` for data-driven window placement.

    Returns
    -------
    DataFrame with per-window, per-config train vs test comparison.
    """
    effective_windows = windows if windows is not None else WALKFORWARD_WINDOWS
    all_rows: list[dict[str, Any]] = []
    config_appearances: dict[tuple, list[str]] = {}

    for window in effective_windows:
        wname = window["name"]
        print(f"\n{'=' * 100}")
        print(f"  WALK-FORWARD {wname}: "
              f"Train {window['train_start']} to {window['train_end']}  |  "
              f"Test {window['test_start']} to {window['test_end']}")
        print(f"{'=' * 100}")

        train_df, test_df = walkforward_split(
            df, window["train_start"], window["train_end"],
            window["test_start"], window["test_end"],
        )
        train_days = train_df["day"].nunique()
        test_days = test_df["day"].nunique()
        print(f"  Train: {len(train_df):,} candidates, {train_days} days")
        print(f"  Test:  {len(test_df):,} candidates, {test_days} days")

        if train_days < 10 or test_days < 5:
            print("  SKIP: insufficient data in this window")
            continue

        print(f"  Optimizing on train set ({len(_build_optimizer_grid()):,} configs) ...")
        t0 = time.time()
        top_train = _run_window_optimizer(train_df, top_n=top_n)
        train_elapsed = time.time() - t0
        print(f"  Train optimization done in {train_elapsed:.0f}s")

        test_signals = precompute_daily_signals(test_df)
        test_precomp_cache: dict[tuple, dict] = {}

        print(f"\n  {'#':>3} {'Config':<30} {'Tr Sharpe':>10} {'Te Sharpe':>10} "
              f"{'Tr Ret':>8} {'Te Ret':>8} {'Tr DD':>6} {'Te DD':>6} {'Verdict':>10}")
        print("  " + "-" * 100)

        for rank, (_, trow) in enumerate(top_train.iterrows(), 1):
            cfg = _row_to_config(trow)
            tpc_key = (cfg.trading.width_filter, cfg.trading.entry_count)
            if tpc_key not in test_precomp_cache:
                test_precomp_cache[tpc_key] = _precompute_day_selections(
                    test_df, width_filter=tpc_key[0], entry_count=tpc_key[1],
                )
            test_result = run_backtest(test_df, test_signals, cfg, day_precomp=test_precomp_cache[tpc_key])

            train_sharpe = float(trow["sharpe"])
            test_sharpe = test_result.sharpe
            train_ret = float(trow["return_pct"])
            test_ret = test_result.total_return_pct
            train_dd = float(trow["max_dd_pct"])
            test_dd = test_result.max_drawdown_pct

            # Verdict: how well does train performance predict test
            if test_sharpe > 0 and test_ret > 0:
                if train_sharpe <= 0:
                    verdict = "WEAK"
                else:
                    verdict = "PASS" if test_sharpe >= train_sharpe * 0.3 else "WEAK"
            elif test_ret > 0:
                verdict = "MARGINAL"
            else:
                verdict = "FAIL"

            sig = _config_signature(trow)
            key = _config_key(trow)
            if key not in config_appearances:
                config_appearances[key] = []
            config_appearances[key].append(wname)

            print(f"  {rank:>3} {sig:<30} {train_sharpe:>10.2f} {test_sharpe:>10.2f} "
                  f"{train_ret:>+7.0f}% {test_ret:>+7.0f}% "
                  f"{train_dd:>5.0f}% {test_dd:>5.0f}% {verdict:>10}")

            row_out = trow.to_dict()
            row_out.update({
                "window": wname,
                "rank_in_window": rank,
                "train_sharpe": train_sharpe,
                "test_sharpe": test_sharpe,
                "train_return_pct": train_ret,
                "test_return_pct": test_ret,
                "train_max_dd_pct": train_dd,
                "test_max_dd_pct": test_dd,
                "verdict": verdict,
            })
            all_rows.append(row_out)

    # Cross-window summary
    print(f"\n{'=' * 100}")
    print("  CROSS-WINDOW SUMMARY: configs appearing in top-10 across multiple windows")
    print(f"{'=' * 100}")

    multi_window = {k: v for k, v in config_appearances.items() if len(v) > 1}
    if multi_window:
        wf_df = pd.DataFrame(all_rows)
        print(f"\n  {len(multi_window)} configs appear in 2+ windows:\n")

        for key, windows in sorted(multi_window.items(), key=lambda x: -len(x[1])):
            matches = wf_df[wf_df.apply(lambda r: _config_key(r) == key, axis=1)]
            if matches.empty:
                continue
            sig = _config_signature(matches.iloc[0])
            avg_test_sharpe = matches["test_sharpe"].mean()
            avg_test_ret = matches["test_return_pct"].mean()
            wlist = ",".join(windows)
            print(f"  {sig:<35} windows=[{wlist}]  "
                  f"avg_test_sharpe={avg_test_sharpe:.2f}  "
                  f"avg_test_return={avg_test_ret:+.0f}%")
    else:
        print("\n  No config appeared in the top-10 of more than one window.")
        print("  This suggests high regime-dependence; consider simpler / more robust configs.")

    results_df = pd.DataFrame(all_rows)
    if not results_df.empty:
        results_df.to_csv(output_csv, index=False)
        print(f"\n  Walk-forward results exported to {output_csv}")

    return results_df


# ===================================================================
# CLI
# ===================================================================


def main() -> None:
    """Entry point for the backtest CLI."""
    parser = argparse.ArgumentParser(
        description="Capital-budgeted backtest with scheduled + event-driven layers",
    )
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV),
                        help="Path to training_candidates.csv")
    parser.add_argument("--capital", type=float, default=20_000)
    parser.add_argument("--max-trades", type=int, default=2)
    parser.add_argument("--monthly-stop", type=float, default=0.15,
                        help="Monthly drawdown stop (0 to disable)")
    parser.add_argument("--lot-per-equity", type=float, default=10_000)
    parser.add_argument("--both-sides", action="store_true", default=False)
    parser.add_argument("--min-dte", type=int, default=None)
    parser.add_argument("--max-delta", type=float, default=None)

    # Event params
    parser.add_argument("--event", action="store_true", default=False,
                        help="Enable event-driven layer")
    parser.add_argument("--event-budget", type=str, default="shared",
                        choices=["shared", "separate"])
    parser.add_argument("--event-max-trades", type=int, default=1)
    parser.add_argument("--event-drop-threshold", type=float, default=-0.01)
    parser.add_argument("--event-side", type=str, default="puts",
                        choices=["puts", "calls", "best"])
    parser.add_argument("--event-min-dte", type=int, default=5)
    parser.add_argument("--event-max-dte", type=int, default=7)
    parser.add_argument("--event-min-delta", type=float, default=0.15)
    parser.add_argument("--event-max-delta", type=float, default=0.25)
    parser.add_argument("--rally-avoid", action="store_true", default=False)
    parser.add_argument("--rally-threshold", type=float, default=0.01)

    # Modes
    parser.add_argument("--compare", action="store_true", default=False,
                        help="Run preset comparison configs")
    parser.add_argument("--optimize", action="store_true", default=False,
                        help="Run exhaustive grid-search optimizer (original mode)")
    parser.add_argument("--optimize-staged", action="store_true", default=False,
                        help="Run 3-stage optimizer: trading -> portfolio -> event "
                             "(requires multi-TP trajectory columns)")
    parser.add_argument("--optimize-event-only", action="store_true", default=False,
                        help="Run event-only optimizer: sweep SPX-drop strategies "
                             "that only trade on drop days (appends to results CSV)")
    parser.add_argument("--optimize-selective", action="store_true", default=False,
                        help="Run selective high-win-rate optimizer: conservative "
                             "params targeting 90%+ day win rate (appends to results CSV)")
    parser.add_argument("--analyze", action="store_true", default=False,
                        help="Deep-dive optimizer results (param importance, Pareto, robustness)")
    parser.add_argument("--walkforward", action="store_true", default=False,
                        help="Rolling walk-forward validation (train/test splits)")
    parser.add_argument("--wf-auto", action="store_true", default=False,
                        help="Auto-generate walk-forward windows from data date range")
    parser.add_argument("--wf-train-months", type=int, default=3,
                        help="Walk-forward train window in months (default: 3)")
    parser.add_argument("--wf-test-months", type=int, default=1,
                        help="Walk-forward test window in months (default: 1)")
    parser.add_argument("--wf-step-months", type=int, default=1,
                        help="Walk-forward step size in months (default: 1)")
    parser.add_argument("--top-n-trading", type=int, default=10,
                        help="Staged optimizer: top-N trading configs to carry into Stage 2 (default: 10)")
    parser.add_argument("--top-n-combined", type=int, default=5,
                        help="Staged optimizer: top-N combined configs to carry into Stage 3 (default: 5)")
    parser.add_argument("--output-csv", type=str, default=str(RESULTS_CSV),
                        help="CSV output path for optimizer results")

    args = parser.parse_args()

    # --analyze only needs the results CSV, not the full training data
    if args.analyze:
        results_path = Path(args.output_csv)
        if not results_path.exists():
            logger.error("Results CSV not found: %s. Run --optimize first to generate it.", results_path)
            sys.exit(1)
        run_analysis(results_path)

        csv_path = Path(args.csv)
        if csv_path.exists():
            print("\nLoading training data for robustness check ...")
            df = pd.read_csv(csv_path)
            daily_signals = precompute_daily_signals(df)
            rdf = pd.read_csv(results_path)
            _robustness_check(df, daily_signals, rdf, top_n=20)
        else:
            print("\n  (skipping robustness check -- training CSV not found)")
        return

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Validate required and optional columns
    required_cols = {"day", "entry_credit", "realized_pnl", "spread_side",
                     "dte_target", "delta_target", "credit_to_width", "spot", "vix"}
    optional_cols = {"width_points", "is_opex_day", "is_fomc_day", "is_nfp_day",
                     "is_cpi_day", "hold_realized_pnl", "recovered_after_sl",
                     "hold_hit_tp50", "exit_reason", "final_pnl_at_expiry"}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        logger.error("CSV missing required columns: %s", sorted(missing_required))
        sys.exit(2)
    missing_optional = optional_cols - set(df.columns)
    if missing_optional:
        logger.warning("CSV missing optional columns (will be ignored): %s", sorted(missing_optional))

    print(f"  {len(df):,} candidates across {df['day'].nunique()} trading days")
    print(f"  Period: {df['day'].min()} to {df['day'].max()}")

    print("Precomputing daily signals ...")
    daily_signals = precompute_daily_signals(df)

    if args.walkforward:
        wf_windows = None
        if args.wf_auto:
            min_day = df["day"].min()
            max_day = df["day"].max()
            wf_windows = generate_auto_windows(
                min_day, max_day,
                train_months=args.wf_train_months,
                test_months=args.wf_test_months,
                step_months=args.wf_step_months,
            )
            print(f"Auto-generated {len(wf_windows)} walk-forward windows from {min_day} to {max_day}")
        wf_output = Path(args.output_csv).parent / "walkforward_results.csv"
        run_walkforward(df, output_csv=wf_output, windows=wf_windows)

    elif args.optimize_staged:
        results_df = run_staged_optimizer(
            df, daily_signals, Path(args.output_csv),
            top_n_trading=args.top_n_trading,
            top_n_combined=args.top_n_combined,
        )
        for metric in ["sharpe", "return_pct", "max_dd_pct"]:
            print_optimizer_top(results_df, metric)

    elif args.optimize_event_only:
        results_df = run_event_only_optimizer(
            df, daily_signals, Path(args.output_csv),
        )
        for metric in ["sharpe", "return_pct", "max_dd_pct"]:
            print_optimizer_top(results_df, metric)

    elif args.optimize_selective:
        results_df = run_selective_optimizer(
            df, daily_signals, Path(args.output_csv),
        )
        for metric in ["sharpe", "return_pct", "max_dd_pct"]:
            print_optimizer_top(results_df, metric)

    elif args.optimize:
        results_df = run_optimizer(df, daily_signals, Path(args.output_csv))
        for metric in ["sharpe", "return_pct", "max_dd_pct"]:
            print_optimizer_top(results_df, metric)

    elif args.compare:
        presets = _build_comparison_configs()
        results = []
        for label, cfg in presets:
            r = run_backtest(df, daily_signals, cfg, label)
            results.append(r)
        print_comparison_table(results)
        print("\n\nDetailed monthly breakdown for recommended config:")
        print_summary(results[0])
        print_monthly(results[0])

    else:
        pc = PortfolioConfig(
            starting_capital=args.capital,
            max_trades_per_day=args.max_trades,
            monthly_drawdown_limit=args.monthly_stop if args.monthly_stop > 0 else None,
            lot_per_equity=args.lot_per_equity,
            calls_only=not args.both_sides,
            min_dte=args.min_dte,
            max_delta=args.max_delta,
        )
        evc = EventConfig(
            enabled=args.event,
            budget_mode=args.event_budget,
            max_event_trades=args.event_max_trades,
            spx_drop_threshold=args.event_drop_threshold,
            side_preference=args.event_side,
            min_dte=args.event_min_dte,
            max_dte=args.event_max_dte,
            min_delta=args.event_min_delta,
            max_delta=args.event_max_delta,
            rally_avoidance=args.rally_avoid,
            rally_threshold=args.rally_threshold,
        )
        cfg = FullConfig(portfolio=pc, event=evc)

        stop_label = f"{pc.monthly_drawdown_limit:.0%} stop" if pc.monthly_drawdown_limit else "no stop"
        lots_label = "gradual" if pc.lot_per_equity < 100_000 else "fixed"
        event_label = " + events" if evc.enabled else ""
        label = (f"${pc.starting_capital/1000:.0f}k | "
                 f"{pc.max_trades_per_day}/day | "
                 f"{stop_label} | {lots_label}{event_label}")

        result = run_backtest(df, daily_signals, cfg, label)
        print_summary(result)
        print_monthly(result)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal: %s", exc, exc_info=True)
        sys.exit(1)
