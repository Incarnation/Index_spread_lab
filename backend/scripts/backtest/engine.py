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
import os
import pickle
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import timedelta
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)

def _locate_scripts_dir() -> Path:
    """Find ``backend/scripts/`` from this file's location.

    Works whether the file is the original monolith (``backend/scripts/
    backtest_strategy.py``) or a submodule of the post-split package
    (``backend/scripts/backtest/<sub>.py``).  We walk up the parent
    chain looking for a directory named ``scripts`` whose sibling
    ``spx_backend`` package exists -- that's the canonical scripts
    directory regardless of nesting.
    """
    for parent in Path(__file__).resolve().parents:
        if parent.name == "scripts" and (parent.parent / "spx_backend").is_dir():
            return parent
    # Fall back to direct parent so behaviour is unchanged when running
    # from an unfamiliar layout (e.g. a test fixture); spx_backend
    # imports below will fail loudly rather than silently mis-resolve.
    return Path(__file__).resolve().parent


_SCRIPTS_DIR = _locate_scripts_dir()
_BACKEND_DIR = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))
# Add backend root so ``spx_backend.config`` is importable for the
# M10 alignment-warn helper.  Falls back gracefully when the package
# is unavailable (e.g. minimal CI sandboxes).
sys.path.insert(0, str(_BACKEND_DIR))

from _constants import CONTRACT_MULT, CONTRACTS, MARGIN_PER_LOT
from _pareto import extract_pareto_frontier as _pareto_extract
from regime_utils import compute_regime_metrics

# Shared event-signal evaluator (Wave 5 of OFFLINE_PIPELINE_AUDIT.md).
# Both the live ``EventSignalDetector`` and the backtest's wrapper now
# delegate to ``evaluate_event_signals`` so signal-firing rules stay
# byte-identical between paths.  A failed import (e.g. a Python sandbox
# without the spx_backend package on sys.path) is treated as a hard
# error -- a divergent backtest is worse than a missing one.
from spx_backend.services.event_signals import (
    EventThresholds,
    evaluate_event_signals,
)

# Shared candidate-leg-identity dedup helper (audit M3).  Both the live
# ``DecisionJob`` and this backtest engine now use the same key shape
# so a candidate row that maps to the exact same (side, expiration,
# short_symbol, long_symbol) as another row in the same day-window is
# never double-counted regardless of DataFrame index.
from spx_backend.services.candidate_dedupe import candidate_dedupe_key

DATA_DIR = _SCRIPTS_DIR.parents[1] / "data"
DEFAULT_CSV = DATA_DIR / "training_candidates.csv"
RESULTS_CSV = DATA_DIR / "backtest_results.csv"

# Backtest-local mirror of the live ``_SPX_2D_MAX_CALENDAR_GAP_DAYS``.
# Kept as a module constant for any non-detector code paths that want
# to introspect or document the gap threshold; the actual gating now
# lives in ``evaluate_event_signals`` (single source of truth).
_SPX_2D_MAX_CALENDAR_GAP_DAYS = 4


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
    final_pnl = df.get("final_pnl_at_expiry", pd.Series(np.nan, index=df.index))
    fallback = final_pnl.fillna(df["realized_pnl"])
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

@dataclass
class DayRecord:
    """One row in the equity-curve output.

    ``winning_trades`` tracks the per-day count of individual trades
    whose realised PnL was strictly > 0 (Tier 1 win-rate disambiguation
    fix from E2E pipeline review).  Used to compute ``win_trade_rate``
    alongside the existing day-level ``win_day_rate``.
    """

    day: str
    equity: float
    daily_pnl: float
    n_trades: int
    lots: int
    status: str
    month_start_equity: float
    event_signals: str = ""
    winning_trades: int = 0


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
        """Lot count for the next trade, cached per day for consistency.

        The lot count is the minimum of three caps:

        1. ``lot_per_equity`` -- equity-bucketed sizing (e.g. one lot per
           $10k of equity).
        2. ``max_equity_risk_pct`` -- caps single-trade catastrophic loss
           exposure (``equity * pct / margin_per_lot``).
        3. ``max_margin_pct`` -- caps margin dollars committed for the
           trade as a fraction of equity (``equity * pct / margin_per_lot``).

        Cap 3 was previously declared on ``PortfolioConfig`` but never
        enforced (M4 in OFFLINE_PIPELINE_AUDIT.md), making it a no-op
        optimizer dimension.  It is now enforced symmetrically with cap 2
        so the optimizer CSV reflects real behaviour.

        Caps 2 and 3 are functionally similar for narrow credit spreads
        where ``max_loss ≈ margin``; they diverge for wide spreads or
        when ``max_margin_pct < max_equity_risk_pct``.
        """
        if self._lots_today is not None:
            return self._lots_today

        raw = max(1, int(self.equity / self.cfg.lot_per_equity))
        max_by_risk = max(1, int(self.equity * self.cfg.max_equity_risk_pct / self.margin_per_lot))
        # M4 fix: enforce max_margin_pct so the optimizer dimension is
        # not silently inert.  Margin per trade = lots * margin_per_lot,
        # so the lot cap from a margin-fraction cap is symmetric to the
        # risk cap formula above.
        max_by_margin = max(1, int(self.equity * self.cfg.max_margin_pct / self.margin_per_lot))
        self._lots_today = min(raw, max_by_risk, max_by_margin)
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

    Thin wrapper around the shared ``evaluate_event_signals`` evaluator
    in ``spx_backend.services.event_signals``: the *integration*
    responsibility (taking a precomputed ``pandas.Series`` row, packing
    EventConfig fields into the shared ``EventThresholds`` carrier)
    lives here, but the actual signal-firing rules live in one place
    so the backtest can never silently drift from live.  See the C1,
    H1, M3, M7, M9 finding cluster in OFFLINE_PIPELINE_AUDIT.md for
    the divergence history that motivated this extraction.

    Parameters
    ----------
    config : EventConfig with all trigger thresholds.
    """

    def __init__(self, config: EventConfig) -> None:
        """Bind the detector to the given event configuration."""
        self.cfg = config

    def _build_thresholds(self) -> "EventThresholds":
        """Pack the EventConfig fields into a shared EventThresholds.

        Local import keeps backtest_strategy importable without forcing
        the live ``spx_backend`` package onto the path during minimal
        test collection (the live package is already on the path in
        normal usage; this is only insurance).
        """
        return EventThresholds(
            spx_drop_threshold=self.cfg.spx_drop_threshold,
            spx_drop_2d_threshold=self.cfg.spx_drop_2d_threshold,
            vix_spike_threshold=self.cfg.vix_spike_threshold,
            vix_elevated_threshold=self.cfg.vix_elevated_threshold,
            term_inversion_threshold=self.cfg.term_inversion_threshold,
            rally_avoidance=self.cfg.rally_avoidance,
            rally_threshold=self.cfg.rally_threshold,
            signal_mode=self.cfg.signal_mode,
            spx_drop_min=self.cfg.spx_drop_min,
            spx_drop_max=self.cfg.spx_drop_max,
        )

    def detect(self, day_row: pd.Series) -> list[str]:
        """Return list of active signal names for today.

        Parameters
        ----------
        day_row : Series with keys ``prev_spx_return``, ``prev_spx_return_2d``,
            ``prev_vix_pct_change``, ``vix``, ``term_structure``,
            ``prev_spx_return_2d_gap_days``.
        """
        if not self.cfg.enabled:
            return []
        # Convert the Series to a plain dict so the shared evaluator
        # doesn't need to know about pandas.  ``.to_dict()`` preserves
        # NaN values which the shared ``_is_nan_safe`` helper handles.
        ctx = day_row.to_dict() if hasattr(day_row, "to_dict") else dict(day_row)
        # Suppress the per-day "spx_drop_2d suppressed" warning -- the
        # backtest sweeps thousands of days and the warning is more
        # useful in live where the gap is unexpected.
        return evaluate_event_signals(ctx, self._build_thresholds(), log_warnings=False)


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

    # Calendar gap (in days) between the row 2 trading days back and the
    # current row.  Used downstream to suppress "2-day return" signals
    # that actually span a long weekend or data outage; mirrors the live
    # ``prev_spx_return_2d_gap_days`` field in event_signals.py
    # (see H1 in OFFLINE_PIPELINE_AUDIT.md).
    day_dt = pd.to_datetime(daily["day"], errors="coerce")
    daily["spx_return_2d_gap_days"] = (day_dt - day_dt.shift(2)).dt.days

    daily["prev_spx_return"] = daily["spx_return"].shift(1)
    daily["prev_spx_return_2d"] = daily["spx_return_2d"].shift(1)
    daily["prev_spx_return_2d_gap_days"] = daily["spx_return_2d_gap_days"].shift(1)
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
    regime_metrics: dict[str, Any] = field(default_factory=dict)
    # Tier 1 win-rate disambiguation: count of individual trades with
    # PnL > 0 (versus ``win_days`` which counts days whose net PnL was
    # > 0).  Day-level WR conflates a 2-win-1-loss day with a 1-win
    # day; trade-level WR distinguishes them.  Default 0 keeps any
    # external constructors that pre-date this field working.
    win_trades: int = 0


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
        # Per-day count of winning individual trades (PnL > 0).  Required
        # for win_trade_rate disambiguation: previously every output
        # column called ``win_rate`` was actually win_day_rate, which
        # masks signal quality on multi-trade days.
        winning_trades_today = 0

        # Resolve which PnL column to read from candidates
        effective_pnl_col = pnl_col if use_precomputed else "realized_pnl"

        # --- Event trades first ---
        drop_signals = [s for s in signals if s != "rally"]
        event_trades_placed = 0
        event_candidates = pd.DataFrame()
        # Day-scoped set of leg-identity keys already executed.  Replaces
        # the old DataFrame.index dedup so two rows that map to the same
        # (side, expiration, short_symbol, long_symbol) are not both
        # taken even if their indices differ (audit M3).
        seen_leg_keys: set[tuple[str, str, str, str]] = set()
        if ec.enabled and drop_signals:
            evt_max = ec.max_event_trades
            event_candidates = _fast_event_select(precomp, ec, signals, evt_max)

            for _idx, row in event_candidates.iterrows():
                if event_trades_placed >= evt_max or not pm.can_trade():
                    break
                pnl_val = row[effective_pnl_col]
                if pd.isna(pnl_val):
                    continue
                key = candidate_dedupe_key(row)
                # Skip leg pairs we've already executed today.  Empty
                # short/long symbols (unknown legs) collapse to a single
                # bucket ``("", "", "", "")``; in that case dedup
                # degenerates to "take only one unknown-leg trade per
                # day," which is intentional and safe.
                if key in seen_leg_keys:
                    continue
                seen_leg_keys.add(key)
                trade_pnl = pm.record_trade(float(pnl_val), lots)
                daily_pnl += trade_pnl
                trades_placed += 1
                event_trades_placed += 1
                if trade_pnl > 0:
                    winning_trades_today += 1

        # --- Scheduled trades ---
        if not skip_scheduled:
            if ec.enabled and ec.budget_mode == "shared":
                sched_limit = max(0, pc.max_trades_per_day - event_trades_placed)
            else:
                sched_limit = pc.max_trades_per_day
            sched_candidates = _fast_sched_select(precomp, pc, sched_limit)
            if len(event_candidates) > 0 and len(sched_candidates) > 0:
                # Index-based pre-filter is still useful as a fast
                # short-circuit (avoids the per-row Python loop below
                # when event/scheduled overlap completely), but the
                # leg-identity ``seen_leg_keys`` check inside the loop
                # is now the source of truth.
                sched_candidates = sched_candidates[
                    ~sched_candidates.index.isin(event_candidates.index)
                ]
            for _idx, row in sched_candidates.iterrows():
                if not pm.can_trade():
                    break
                pnl_val = row[effective_pnl_col]
                if pd.isna(pnl_val):
                    continue
                key = candidate_dedupe_key(row)
                if key in seen_leg_keys:
                    continue
                seen_leg_keys.add(key)
                trade_pnl = pm.record_trade(float(pnl_val), lots)
                daily_pnl += trade_pnl
                trades_placed += 1
                if trade_pnl > 0:
                    winning_trades_today += 1

        status = "traded" if trades_placed > 0 else ("rally_skip" if skip_scheduled else "no_candidates")
        curve.append(DayRecord(
            day=day, equity=pm.equity, daily_pnl=daily_pnl,
            n_trades=trades_placed, lots=lots, status=status,
            month_start_equity=pm.month_start_equity,
            event_signals=",".join(signals) if signals else "",
            winning_trades=winning_trades_today,
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
    # Tier 1 win-rate disambiguation: aggregate per-day winning-trade
    # counts.  Defensive default to 0 if the column is missing (e.g.
    # the empty-curve early-return below already returns BacktestResult
    # with win_trades=0).
    win_trades = int(ec_df["winning_trades"].sum()) if "winning_trades" in ec_df.columns else 0

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

    regime_met = compute_regime_metrics(ec_df, daily_signals)

    return BacktestResult(
        config=config, label=label, curve=curve,
        final_equity=final, total_return_pct=total_ret,
        annualised_return_pct=ann_ret, max_drawdown_pct=max_dd_pct,
        trough=float(ec_df["equity"].min()),
        total_trades=int(ec_df["n_trades"].sum()),
        days_traded=days_traded, days_stopped=days_stopped,
        win_days=win_days, sharpe=sharpe,
        monthly=pd.DataFrame(monthly_rows),
        regime_metrics=regime_met,
        win_trades=win_trades,
    )
