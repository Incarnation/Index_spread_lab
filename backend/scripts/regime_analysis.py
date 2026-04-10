#!/usr/bin/env python3
"""Regime-based performance analysis for SPX credit-spread candidates.

Answers questions like: "When SPX drops 1-2% AND VIX spikes >15%,
how do put credit spreads perform at 3 DTE vs 7 DTE?"

Loads training_candidates.csv, enriches each row with prior-day SPX
return and VIX change, then slices the data across configurable
dimensions and reports performance metrics per bucket.

Usage::

    # 1D breakdown: put performance by SPX drop bucket
    python regime_analysis.py --group-by spx_drop --side put

    # 2D cross-tab: SPX drop × DTE for puts
    python regime_analysis.py --cross spx_drop dte --side put

    # Conditional: filter to stress days, break down by DTE
    python regime_analysis.py --filter "spx_drop<-0.5,vix_spike>10" --group-by dte --side put

    # Full pre-built regime report
    python regime_analysis.py --regime-report

    # Export to CSV
    python regime_analysis.py --regime-report --output regime_results.csv
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)


_BACKEND = Path(__file__).resolve().parents[1]
DATA_DIR = _BACKEND.parent / "data"
DEFAULT_CSV = DATA_DIR / "training_candidates.csv"


# ===================================================================
# Data enrichment
# ===================================================================


def enrich_with_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge prior-day SPX return and VIX change onto every candidate row.

    Computes per-day aggregates (last observation of spot/vix sorted by
    entry_dt for determinism), derives lagged returns, and left-joins
    back to the candidate DataFrame.

    Parameters
    ----------
    df : Training candidates with ``day``, ``spot``, ``vix`` columns.

    Returns
    -------
    DataFrame with added columns: ``prev_spx_return``, ``prev_spx_return_2d``,
    ``prev_vix_pct_change``, ``vix_change_abs``.

    .. note::
        Values are in **percentage units** (e.g. ``-1.0`` = a 1% drop).
        This differs from ``backtest_strategy.precompute_daily_signals()``
        which uses decimal fractions (``-0.01`` = a 1% drop).  When
        translating regime analysis findings into EventConfig thresholds,
        divide by 100.
    """
    sorted_df = df.sort_values("entry_dt") if "entry_dt" in df.columns else df
    agg: dict[str, tuple] = {
        "spot": ("spot", "last"),
        "vix": ("vix", "last"),
    }
    daily = (
        sorted_df.groupby("day")
        .agg(**agg)
        .reset_index()
        .sort_values("day")
        .reset_index(drop=True)
    )
    daily["spx_return"] = daily["spot"].pct_change() * 100
    daily["spx_return_2d"] = daily["spot"].pct_change(2) * 100
    daily["vix_pct_change"] = daily["vix"].pct_change() * 100

    daily["prev_spx_return"] = daily["spx_return"].shift(1)
    daily["prev_spx_return_2d"] = daily["spx_return_2d"].shift(1)
    daily["prev_vix_pct_change"] = daily["vix_pct_change"].shift(1)
    daily["vix_change_abs"] = daily["vix"].diff().shift(1)

    merge_cols = [
        "day", "prev_spx_return", "prev_spx_return_2d",
        "prev_vix_pct_change", "vix_change_abs",
    ]
    return df.merge(daily[merge_cols], on="day", how="left")


# ===================================================================
# Bucket builders
# ===================================================================

# Each returns a dict mapping a human-readable label to a boolean mask
# over the DataFrame index, following the pattern in sl_recovery_analysis.py.


def build_spx_drop_buckets(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Prior-day SPX return buckets (in %)."""
    r = df["prev_spx_return"]
    return {
        "SPX < -2%":       r < -2,
        "-2% <= SPX < -1%": (r >= -2) & (r < -1),
        "-1% <= SPX < -0.5%": (r >= -1) & (r < -0.5),
        "-0.5% <= SPX < 0%": (r >= -0.5) & (r < 0),
        "0% <= SPX < 0.5%": (r >= 0) & (r < 0.5),
        "SPX >= 0.5%":     r >= 0.5,
    }


def build_vix_level_buckets(df: pd.DataFrame) -> dict[str, pd.Series]:
    """VIX level at entry."""
    v = df["vix"]
    return {
        "VIX < 15":       v < 15,
        "15 <= VIX < 20":  (v >= 15) & (v < 20),
        "20 <= VIX < 25":  (v >= 20) & (v < 25),
        "25 <= VIX < 30":  (v >= 25) & (v < 30),
        "VIX >= 30":       v >= 30,
    }


def build_vix_spike_buckets(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Prior-day VIX % change buckets."""
    v = df["prev_vix_pct_change"]
    return {
        "VIX chg < 0%":      v < 0,
        "0% <= VIX chg < 5%": (v >= 0) & (v < 5),
        "5% <= VIX chg < 10%": (v >= 5) & (v < 10),
        "10% <= VIX chg < 15%": (v >= 10) & (v < 15),
        "VIX chg >= 15%":    v >= 15,
    }


def build_dte_buckets(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Individual DTE target values."""
    return {
        f"DTE={d}": df["dte_target"] == d
        for d in sorted(df["dte_target"].unique())
    }


def build_side_buckets(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Spread side."""
    return {s: df["spread_side"] == s for s in sorted(df["spread_side"].unique())}


def build_delta_buckets(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Delta target values."""
    return {
        f"delta={d:.2f}": df["delta_target"] == d
        for d in sorted(df["delta_target"].unique())
    }


def build_width_buckets(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Spread width values."""
    if "width_points" not in df.columns:
        return {"all": pd.Series(True, index=df.index)}
    return {
        f"width={w:.0f}": df["width_points"] == w
        for w in sorted(df["width_points"].unique())
    }


def build_calendar_buckets(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Calendar event day flags."""
    buckets: dict[str, pd.Series] = {}
    for col, label in [
        ("is_opex_day", "OPEX"),
        ("is_fomc_day", "FOMC"),
        ("is_nfp_day", "NFP"),
        ("is_cpi_day", "CPI"),
        ("is_triple_witching", "TripleWitch"),
    ]:
        if col in df.columns:
            mask = df[col].fillna(False).astype(bool)
            buckets[label] = mask
            buckets[f"non-{label}"] = ~mask
    return buckets


def build_term_structure_buckets(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Term structure regime (VIX9D / VIX)."""
    if "term_structure" not in df.columns:
        return {}
    ts = df["term_structure"]
    return {
        "Normal (TS < 1.0)": ts < 1.0,
        "Inverted (TS >= 1.0)": ts >= 1.0,
    }


DIMENSION_BUILDERS: dict[str, Any] = {
    "spx_drop":       build_spx_drop_buckets,
    "vix_level":      build_vix_level_buckets,
    "vix_spike":      build_vix_spike_buckets,
    "dte":            build_dte_buckets,
    "side":           build_side_buckets,
    "delta":          build_delta_buckets,
    "width":          build_width_buckets,
    "calendar":       build_calendar_buckets,
    "term_structure": build_term_structure_buckets,
}


# ===================================================================
# Metrics computation
# ===================================================================


def compute_cell_metrics(
    df: pd.DataFrame,
    pnl_col: str = "realized_pnl",
) -> dict[str, Any]:
    """Compute performance metrics for a slice of candidates.

    Parameters
    ----------
    df : Filtered DataFrame slice.
    pnl_col : Column to use for PnL values.

    Returns
    -------
    Dict with n_trades, avg_pnl, total_pnl, win_rate, pnl_ratio
    (per-trade mean/std, not annualized Sharpe), avg_max_adverse,
    avg_max_favorable, tp hit rates, tail_5pct.
    """
    n_total = len(df)
    pnl = df[pnl_col].dropna()
    n = len(pnl)
    if n == 0:
        return _empty_metrics(n_total)

    arr = pnl.values.astype(float)
    avg = float(np.mean(arr))
    std = float(np.std(arr)) if n > 1 else 0.0
    # Mean of worst ~5% of trades (not the 5th-percentile quantile).
    # For small n this may be a single trade.
    tail_count = max(1, n // 20)
    tail_5 = float(np.mean(np.sort(arr)[:tail_count]))

    result: dict[str, Any] = {
        "n_total": n_total,
        "n_trades": n,
        "avg_pnl": avg,
        "total_pnl": float(np.sum(arr)),
        "win_rate": float(np.mean(arr > 0)),
        "pnl_ratio": avg / std if std > 0 else 0.0,
        "tail_5pct": tail_5,
    }

    if "max_adverse_pnl" in df.columns:
        adv = df["max_adverse_pnl"].dropna()
        result["avg_max_adverse"] = float(adv.mean()) if len(adv) > 0 else None
    else:
        result["avg_max_adverse"] = None

    if "max_favorable_pnl" in df.columns:
        fav = df["max_favorable_pnl"].dropna()
        result["avg_max_favorable"] = float(fav.mean()) if len(fav) > 0 else None
    else:
        result["avg_max_favorable"] = None

    for tp in [50, 60, 70, 80, 90, 100]:
        col = f"first_tp{tp}_pnl"
        if col in df.columns:
            tp_hits = int(df[col].notna().sum())
            result[f"tp{tp}_hit_rate"] = tp_hits / n if n > 0 else 0.0
        else:
            result[f"tp{tp}_hit_rate"] = None

    return result


def _empty_metrics(n_total: int = 0) -> dict[str, Any]:
    """Return a metrics dict for an empty slice."""
    m: dict[str, Any] = {
        "n_total": n_total, "n_trades": 0, "avg_pnl": None, "total_pnl": None,
        "win_rate": None, "pnl_ratio": None, "tail_5pct": None,
        "avg_max_adverse": None, "avg_max_favorable": None,
    }
    for tp in [50, 60, 70, 80, 90, 100]:
        m[f"tp{tp}_hit_rate"] = None
    return m


# ===================================================================
# Filter expression parser
# ===================================================================


# Maps user-facing names to (column, scale_factor).
# scale_factor converts user units (%) to column units.
FILTER_COLUMNS: dict[str, tuple[str, float]] = {
    "spx_drop":     ("prev_spx_return", 1.0),
    "spx_drop_2d":  ("prev_spx_return_2d", 1.0),
    "vix":          ("vix", 1.0),
    "vix_spike":    ("prev_vix_pct_change", 1.0),
    "vix_abs":      ("vix_change_abs", 1.0),
    "dte":          ("dte_target", 1.0),
    "delta":        ("delta_target", 1.0),
    "width":        ("width_points", 1.0),
    "term_structure": ("term_structure", 1.0),
}

_FILTER_RE = re.compile(r"^(\w+)\s*([<>=!]+)\s*(-?[\d.]+)$")


def parse_filter_expr(expr: str) -> list[tuple[str, str, float]]:
    """Parse a comma-separated filter expression.

    Parameters
    ----------
    expr : e.g. ``"spx_drop<-0.5,vix>25,vix_spike>10"``

    Returns
    -------
    List of (column_name, operator, threshold) tuples.

    Raises
    ------
    ValueError
        If a token cannot be parsed.
    """
    tokens = [t.strip() for t in expr.split(",") if t.strip()]
    parsed: list[tuple[str, str, float]] = []
    for tok in tokens:
        m = _FILTER_RE.match(tok)
        if not m:
            raise ValueError(f"Cannot parse filter token: {tok!r}")
        name, op, val_str = m.group(1), m.group(2), m.group(3)
        if name not in FILTER_COLUMNS:
            raise ValueError(
                f"Unknown filter dimension: {name!r}. "
                f"Available: {', '.join(sorted(FILTER_COLUMNS))}"
            )
        col, scale = FILTER_COLUMNS[name]
        parsed.append((col, op, float(val_str) * scale))
    return parsed


def apply_filters(df: pd.DataFrame, filters: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Apply parsed filter conditions to a DataFrame.

    Parameters
    ----------
    df : Enriched candidates DataFrame.
    filters : Output of ``parse_filter_expr``.

    Returns
    -------
    Filtered DataFrame.
    """
    mask = pd.Series(True, index=df.index)
    for col, op, val in filters:
        if col not in df.columns:
            logger.warning("Filter column %r not found in data — skipping filter %s%s%s", col, col, op, val)
            continue
        s = df[col]
        if op == "<":
            mask &= s < val
        elif op == "<=":
            mask &= s <= val
        elif op == ">":
            mask &= s > val
        elif op == ">=":
            mask &= s >= val
        elif op in ("==", "="):
            mask &= s == val
        elif op in ("!=", "<>"):
            mask &= s != val
        else:
            raise ValueError(f"Unknown operator: {op!r}")
    return df[mask]


# ===================================================================
# Analysis modes
# ===================================================================


def run_1d_breakdown(
    df: pd.DataFrame,
    dimension: str,
    pnl_col: str = "realized_pnl",
    min_trades: int = 10,
) -> pd.DataFrame:
    """Group by one dimension and compute metrics per bucket.

    Parameters
    ----------
    df : Enriched + filtered candidates.
    dimension : Key into ``DIMENSION_BUILDERS``.
    pnl_col : PnL column to use for metrics.
    min_trades : Minimum trades to include a bucket in the result.

    Returns
    -------
    DataFrame with one row per bucket plus metrics columns.
    """
    builder = DIMENSION_BUILDERS.get(dimension)
    if builder is None:
        raise ValueError(
            f"Unknown dimension: {dimension!r}. "
            f"Available: {', '.join(sorted(DIMENSION_BUILDERS))}"
        )
    buckets = builder(df)
    rows: list[dict[str, Any]] = []
    for label, mask in buckets.items():
        sub = df[mask]
        metrics = compute_cell_metrics(sub, pnl_col)
        if metrics["n_trades"] < min_trades:
            continue
        rows.append({"bucket": label, **metrics})
    return pd.DataFrame(rows)


def run_2d_crosstab(
    df: pd.DataFrame,
    dim1: str,
    dim2: str,
    pnl_col: str = "realized_pnl",
    min_trades: int = 10,
) -> pd.DataFrame:
    """Two-dimensional cross-tabulation of metrics.

    Parameters
    ----------
    df : Enriched + filtered candidates.
    dim1, dim2 : Dimension keys.
    pnl_col : PnL column.
    min_trades : Minimum trades per cell.

    Returns
    -------
    DataFrame with one row per (dim1_bucket, dim2_bucket) combination.
    """
    b1_fn = DIMENSION_BUILDERS.get(dim1)
    b2_fn = DIMENSION_BUILDERS.get(dim2)
    if b1_fn is None or b2_fn is None:
        raise ValueError(f"Unknown dimension(s): {dim1}, {dim2}")
    buckets1 = b1_fn(df)
    buckets2 = b2_fn(df)
    rows: list[dict[str, Any]] = []
    for l1, m1 in buckets1.items():
        for l2, m2 in buckets2.items():
            sub = df[m1 & m2]
            metrics = compute_cell_metrics(sub, pnl_col)
            if metrics["n_trades"] < min_trades:
                continue
            rows.append({dim1: l1, dim2: l2, **metrics})
    return pd.DataFrame(rows)


def run_regime_report(
    df: pd.DataFrame,
    pnl_col: str = "realized_pnl",
    min_trades: int = 10,
) -> list[tuple[str, pd.DataFrame]]:
    """Generate a comprehensive set of pre-built regime analyses.

    Returns a list of (title, DataFrame) pairs covering the most
    important regime explorations for put credit spread analysis.

    Parameters
    ----------
    df : Enriched candidates (all sides/DTEs).
    pnl_col : PnL column.
    min_trades : Minimum trades per cell.

    Returns
    -------
    List of (section_title, results_dataframe) tuples.
    """
    puts = df[df["spread_side"] == "put"]
    calls = df[df["spread_side"] == "call"]

    report: list[tuple[str, pd.DataFrame]] = []

    report.append((
        "PUT CREDIT SPREADS: Performance by SPX Prior-Day Return",
        run_1d_breakdown(puts, "spx_drop", pnl_col, min_trades),
    ))
    report.append((
        "PUT CREDIT SPREADS: Performance by VIX Level",
        run_1d_breakdown(puts, "vix_level", pnl_col, min_trades),
    ))
    report.append((
        "PUT CREDIT SPREADS: Performance by VIX Prior-Day Change",
        run_1d_breakdown(puts, "vix_spike", pnl_col, min_trades),
    ))
    report.append((
        "PUT CREDIT SPREADS: Performance by DTE",
        run_1d_breakdown(puts, "dte", pnl_col, min_trades),
    ))
    report.append((
        "PUT CREDIT SPREADS: SPX Drop x DTE Cross-Tab",
        run_2d_crosstab(puts, "spx_drop", "dte", pnl_col, min_trades),
    ))
    report.append((
        "PUT CREDIT SPREADS: VIX Level x DTE Cross-Tab",
        run_2d_crosstab(puts, "vix_level", "dte", pnl_col, min_trades),
    ))
    report.append((
        "PUT CREDIT SPREADS: SPX Drop x VIX Level Cross-Tab",
        run_2d_crosstab(puts, "spx_drop", "vix_level", pnl_col, min_trades),
    ))
    report.append((
        "PUT CREDIT SPREADS: VIX Spike x DTE Cross-Tab",
        run_2d_crosstab(puts, "vix_spike", "dte", pnl_col, min_trades),
    ))
    report.append((
        "PUT CREDIT SPREADS: Term Structure Regime x DTE",
        run_2d_crosstab(puts, "term_structure", "dte", pnl_col, min_trades),
    ))
    report.append((
        "PUT CREDIT SPREADS: Performance by Delta",
        run_1d_breakdown(puts, "delta", pnl_col, min_trades),
    ))
    report.append((
        "PUT CREDIT SPREADS: Performance by Width",
        run_1d_breakdown(puts, "width", pnl_col, min_trades),
    ))
    report.append((
        "PUT CREDIT SPREADS: SPX Drop x Width Cross-Tab",
        run_2d_crosstab(puts, "spx_drop", "width", pnl_col, min_trades),
    ))
    report.append((
        "CALL CREDIT SPREADS: Performance by SPX Prior-Day Return",
        run_1d_breakdown(calls, "spx_drop", pnl_col, min_trades),
    ))
    report.append((
        "CALL CREDIT SPREADS: Performance by DTE",
        run_1d_breakdown(calls, "dte", pnl_col, min_trades),
    ))
    report.append((
        "ALL SIDES: Calendar Event Breakdown",
        run_1d_breakdown(df, "calendar", pnl_col, min_trades),
    ))

    return report


# ===================================================================
# Output formatting
# ===================================================================


_CORE_COLS = ["n_total", "n_trades", "avg_pnl", "total_pnl", "win_rate", "pnl_ratio", "tail_5pct"]
_EXTENDED_COLS = [
    "avg_max_adverse", "avg_max_favorable",
    "tp50_hit_rate", "tp60_hit_rate", "tp70_hit_rate",
    "tp80_hit_rate", "tp90_hit_rate", "tp100_hit_rate",
]


def format_table(result_df: pd.DataFrame, title: str = "") -> str:
    """Format a results DataFrame as a readable CLI table.

    Parameters
    ----------
    result_df : Output from ``run_1d_breakdown`` or ``run_2d_crosstab``.
    title : Optional section title.

    Returns
    -------
    Formatted string for printing.
    """
    if result_df.empty:
        return f"\n{title}\n  (no data meets the minimum trades threshold)\n" if title else ""

    lines: list[str] = []
    if title:
        lines.append(f"\n{'=' * 120}")
        lines.append(f"  {title}")
        lines.append(f"{'=' * 120}")

    label_cols = [c for c in result_df.columns if c not in _CORE_COLS + _EXTENDED_COLS]
    core_visible = []
    for c in _CORE_COLS:
        if c not in result_df.columns:
            continue
        # Only show n_total when it differs from n_trades (TP column in use)
        if c == "n_total" and (result_df["n_total"] == result_df["n_trades"]).all():
            continue
        core_visible.append(c)
    show_cols = label_cols + core_visible
    show_extended = [c for c in _EXTENDED_COLS if c in result_df.columns and result_df[c].notna().any()]
    show_cols += show_extended

    header_parts: list[str] = []
    for col in show_cols:
        if col in label_cols:
            header_parts.append(f"{col:<25}")
        elif col == "n_total":
            header_parts.append(f"{'Total':>8}")
        elif col == "n_trades":
            header_parts.append(f"{'Trades':>8}")
        elif col == "avg_pnl":
            header_parts.append(f"{'AvgPnL':>10}")
        elif col == "total_pnl":
            header_parts.append(f"{'TotalPnL':>12}")
        elif col == "win_rate" or col.endswith("_hit_rate"):
            header_parts.append(f"{col:>12}")
        elif col == "pnl_ratio":
            header_parts.append(f"{'PnlRatio':>10}")
        elif col == "tail_5pct":
            header_parts.append(f"{'Tail5%':>10}")
        elif col in ("avg_max_adverse", "avg_max_favorable"):
            header_parts.append(f"{col:>16}")
        else:
            header_parts.append(f"{col:>12}")

    lines.append("  " + " ".join(header_parts))
    lines.append("  " + "-" * (len(" ".join(header_parts))))

    for _, row in result_df.iterrows():
        parts: list[str] = []
        for col in show_cols:
            val = row.get(col)
            if col in label_cols:
                parts.append(f"{str(val):<25}")
            elif val is None or (isinstance(val, float) and np.isnan(val)):
                if col in label_cols:
                    parts.append(f"{'':>25}")
                elif col in ("n_total", "n_trades"):
                    parts.append(f"{'--':>8}")
                elif col in ("avg_pnl", "tail_5pct"):
                    parts.append(f"{'--':>10}")
                elif col == "total_pnl":
                    parts.append(f"{'--':>12}")
                elif col in ("avg_max_adverse", "avg_max_favorable"):
                    parts.append(f"{'--':>16}")
                else:
                    parts.append(f"{'--':>12}")
            elif col in ("n_total", "n_trades"):
                parts.append(f"{int(val):>8}")
            elif col == "avg_pnl":
                parts.append(f"${val:>9,.0f}")
            elif col == "total_pnl":
                parts.append(f"${val:>11,.0f}")
            elif col == "tail_5pct":
                parts.append(f"${val:>9,.0f}")
            elif col == "win_rate" or col.endswith("_hit_rate"):
                parts.append(f"{val * 100:>11.1f}%")
            elif col == "pnl_ratio":
                parts.append(f"{val:>10.2f}")
            elif col in ("avg_max_adverse", "avg_max_favorable"):
                parts.append(f"${val:>15,.0f}")
            else:
                parts.append(f"{val:>12}")
        lines.append("  " + " ".join(parts))

    return "\n".join(lines)


# ===================================================================
# CLI
# ===================================================================


def main() -> None:
    """Entry point for regime analysis CLI."""
    parser = argparse.ArgumentParser(
        description="Regime-based performance analysis for SPX credit spreads",
    )
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV),
                        help="Path to training_candidates.csv")
    parser.add_argument("--side", type=str, default="both",
                        choices=["put", "call", "both"],
                        help="Filter to spread side")
    parser.add_argument("--dte", type=str, default=None,
                        help="Comma-separated DTEs to include (e.g. '3,5,7')")
    parser.add_argument("--delta", type=str, default=None,
                        help="Comma-separated deltas (e.g. '0.10,0.15')")
    parser.add_argument("--width", type=float, default=None,
                        help="Filter to specific spread width")
    parser.add_argument("--tp", type=float, default=None,
                        help="TP level for PnL (e.g. 0.50 uses first_tp50_pnl; "
                             "omit to use realized_pnl)")
    parser.add_argument("--group-by", type=str, default=None, dest="group_by",
                        help=f"1D breakdown dimension ({', '.join(sorted(DIMENSION_BUILDERS))})")
    parser.add_argument("--cross", nargs=2, default=None, metavar=("DIM1", "DIM2"),
                        help="2D cross-tabulation of two dimensions")
    parser.add_argument("--filter", type=str, default=None, dest="filter_expr",
                        help="Comma-separated filters (e.g. 'spx_drop<-0.5,vix>25')")
    parser.add_argument("--regime-report", action="store_true", default=False,
                        help="Generate full pre-built regime report")
    parser.add_argument("--output", type=str, default=None,
                        help="CSV export path")
    parser.add_argument("--min-trades", type=int, default=10,
                        help="Minimum trades per cell to display")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(str(csv_path))
    print(f"  {len(df):,} candidates across {df['day'].nunique()} trading days")

    resolved_mask = df.get("resolved", pd.Series(True, index=df.index))
    df = df[resolved_mask.fillna(False).astype(bool)]
    print(f"  {len(df):,} resolved candidates")

    print("Enriching with daily features ...")
    df = enrich_with_daily_features(df)

    # --- Pre-filters ---
    if args.side != "both":
        df = df[df["spread_side"] == args.side]
        print(f"  Filtered to {args.side}: {len(df):,} candidates")

    if args.dte:
        dte_vals = [int(d.strip()) for d in args.dte.split(",")]
        df = df[df["dte_target"].isin(dte_vals)]
        print(f"  Filtered to DTE {dte_vals}: {len(df):,} candidates")

    if args.delta:
        delta_vals = [float(d.strip()) for d in args.delta.split(",")]
        df = df[df["delta_target"].isin(delta_vals)]
        print(f"  Filtered to delta {delta_vals}: {len(df):,} candidates")

    if args.width is not None and "width_points" in df.columns:
        df = df[df["width_points"] == args.width]
        print(f"  Filtered to width {args.width}: {len(df):,} candidates")

    if args.filter_expr:
        filters = parse_filter_expr(args.filter_expr)
        df = apply_filters(df, filters)
        print(f"  After filters: {len(df):,} candidates")

    # Determine PnL column
    pnl_col = "realized_pnl"
    if args.tp is not None:
        tp_key = int(args.tp * 100)
        candidate_col = f"first_tp{tp_key}_pnl"
        if candidate_col in df.columns:
            pnl_col = candidate_col
            print(f"  Using PnL column: {pnl_col}")
        else:
            logger.warning("%s not found, using realized_pnl", candidate_col)

    print(f"  Period: {df['day'].min()} to {df['day'].max()}")

    if len(df) == 0:
        print("No candidates remaining after filters.")
        return

    # --- Analysis ---
    all_results: list[pd.DataFrame] = []

    if args.regime_report:
        report = run_regime_report(df, pnl_col, args.min_trades)
        for title, result_df in report:
            print(format_table(result_df, title))
            if not result_df.empty:
                all_results.append(result_df.assign(_section=title))

    elif args.cross:
        dim1, dim2 = args.cross
        title = f"{dim1.upper()} x {dim2.upper()} Cross-Tab"
        result_df = run_2d_crosstab(df, dim1, dim2, pnl_col, args.min_trades)
        print(format_table(result_df, title))
        all_results.append(result_df)

    elif args.group_by:
        title = f"Breakdown by {args.group_by.upper()}"
        result_df = run_1d_breakdown(df, args.group_by, pnl_col, args.min_trades)
        print(format_table(result_df, title))
        all_results.append(result_df)

    else:
        overall = compute_cell_metrics(df, pnl_col)
        print(f"\n  OVERALL: {overall['n_trades']} trades, "
              f"avg ${overall['avg_pnl']:,.0f}, "
              f"win {overall['win_rate'] * 100:.1f}%, "
              f"pnl_ratio {overall['pnl_ratio']:.2f}")

        for dim in ["spx_drop", "vix_level", "dte"]:
            result_df = run_1d_breakdown(df, dim, pnl_col, args.min_trades)
            title = f"Breakdown by {dim.upper()}"
            print(format_table(result_df, title))
            all_results.append(result_df)

    # --- CSV export ---
    if args.output and all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\n  Results exported to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal: %s", exc, exc_info=True)
        sys.exit(1)
