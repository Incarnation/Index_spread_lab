"""Tests for the regime analysis explorer.

Covers data enrichment, bucket builders, metrics computation,
filter parsing, and analysis modes (1D, 2D, regime report).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from regime_analysis import (
    enrich_with_daily_features,
    build_spx_drop_buckets,
    build_vix_level_buckets,
    build_vix_spike_buckets,
    build_dte_buckets,
    build_side_buckets,
    build_delta_buckets,
    build_width_buckets,
    build_calendar_buckets,
    build_term_structure_buckets,
    compute_cell_metrics,
    parse_filter_expr,
    apply_filters,
    run_1d_breakdown,
    run_2d_crosstab,
    run_regime_report,
    format_table,
    DIMENSION_BUILDERS,
)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Multi-day DataFrame with realistic candidate rows."""
    rows = []
    for day_idx, day in enumerate([
        "2025-06-01", "2025-06-02", "2025-06-03",
        "2025-06-04", "2025-06-05",
    ]):
        for side in ["put", "call"]:
            for dte in [3, 5, 7]:
                rows.append({
                    "day": day,
                    "spread_side": side,
                    "dte_target": dte,
                    "delta_target": 0.10,
                    "width_points": 10.0,
                    "entry_credit": 1.0,
                    "credit_to_width": 0.10,
                    "spot": 5500 - day_idx * 50,
                    "vix": 18 + day_idx * 3,
                    "vix9d": 17 + day_idx * 2,
                    "term_structure": (17 + day_idx * 2) / (18 + day_idx * 3),
                    "resolved": True,
                    "realized_pnl": 50 - day_idx * 30 + (10 if side == "put" else -10),
                    "max_adverse_pnl": -(day_idx * 20 + 5),
                    "max_favorable_pnl": 60 + day_idx * 10,
                    "first_tp50_pnl": 55.0 if day_idx < 3 else None,
                    "first_tp70_pnl": 75.0 if day_idx < 2 else None,
                    "first_tp100_pnl": None,
                    "is_opex_day": day_idx == 2,
                    "is_fomc_day": day_idx == 4,
                    "is_nfp_day": False,
                    "is_cpi_day": False,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def enriched_df(sample_df) -> pd.DataFrame:
    """Sample DataFrame with daily features merged."""
    return enrich_with_daily_features(sample_df)


# ── Enrichment ───────────────────────────────────────────────────


class TestEnrichment:
    def test_columns_added(self, enriched_df):
        """Enrichment adds prev_spx_return and prev_vix_pct_change."""
        for col in ["prev_spx_return", "prev_spx_return_2d",
                     "prev_vix_pct_change", "vix_change_abs"]:
            assert col in enriched_df.columns

    def test_first_day_nan(self, enriched_df):
        """First day has NaN for lagged features (no prior day)."""
        day1 = enriched_df[enriched_df["day"] == "2025-06-01"]
        assert day1["prev_spx_return"].isna().all()

    def test_subsequent_days_have_values(self, enriched_df):
        """Days after the first have non-NaN lagged features."""
        day3 = enriched_df[enriched_df["day"] == "2025-06-03"]
        assert day3["prev_spx_return"].notna().all()
        assert day3["prev_vix_pct_change"].notna().all()

    def test_spx_return_sign(self, enriched_df):
        """Spot decreases day-over-day, so returns should be negative."""
        day3 = enriched_df[enriched_df["day"] == "2025-06-03"]
        assert (day3["prev_spx_return"] < 0).all()

    def test_vix_change_sign(self, enriched_df):
        """VIX increases day-over-day, so changes should be positive."""
        day3 = enriched_df[enriched_df["day"] == "2025-06-03"]
        assert (day3["prev_vix_pct_change"] > 0).all()

    def test_row_count_preserved(self, sample_df, enriched_df):
        """Enrichment does not add or drop rows."""
        assert len(enriched_df) == len(sample_df)


# ── Bucket builders ──────────────────────────────────────────────


class TestBuckets:
    def test_spx_drop_buckets_cover_all(self, enriched_df):
        """SPX drop buckets are mutually exclusive and exhaustive (for non-NaN rows)."""
        has_return = enriched_df["prev_spx_return"].notna()
        sub = enriched_df[has_return]
        buckets = build_spx_drop_buckets(sub)
        total = sum(m.sum() for m in buckets.values())
        assert total == len(sub)

    def test_vix_level_buckets_cover_all(self, enriched_df):
        """VIX level buckets are exhaustive."""
        buckets = build_vix_level_buckets(enriched_df)
        total = sum(m.sum() for m in buckets.values())
        assert total == len(enriched_df)

    def test_vix_spike_buckets_cover_all(self, enriched_df):
        """VIX spike buckets are exhaustive for non-NaN rows."""
        has_val = enriched_df["prev_vix_pct_change"].notna()
        sub = enriched_df[has_val]
        buckets = build_vix_spike_buckets(sub)
        total = sum(m.sum() for m in buckets.values())
        assert total == len(sub)

    def test_dte_buckets_match_values(self, enriched_df):
        """DTE buckets have one entry per unique DTE value."""
        buckets = build_dte_buckets(enriched_df)
        assert len(buckets) == enriched_df["dte_target"].nunique()

    def test_side_buckets(self, enriched_df):
        """Side buckets split into put and call."""
        buckets = build_side_buckets(enriched_df)
        assert "put" in buckets
        assert "call" in buckets

    def test_width_buckets(self, enriched_df):
        """Width buckets match unique widths."""
        buckets = build_width_buckets(enriched_df)
        assert len(buckets) == enriched_df["width_points"].nunique()

    def test_calendar_buckets(self, enriched_df):
        """Calendar buckets include OPEX and FOMC."""
        buckets = build_calendar_buckets(enriched_df)
        assert "OPEX" in buckets
        assert "FOMC" in buckets
        assert buckets["OPEX"].sum() > 0

    def test_term_structure_buckets(self, enriched_df):
        """Term structure splits into normal and inverted."""
        buckets = build_term_structure_buckets(enriched_df)
        assert len(buckets) == 2
        total = sum(m.sum() for m in buckets.values())
        assert total == len(enriched_df)

    def test_all_dimensions_registered(self):
        """All bucket builders are in DIMENSION_BUILDERS."""
        expected = {
            "spx_drop", "vix_level", "vix_spike", "dte", "side",
            "delta", "width", "calendar", "term_structure",
        }
        assert set(DIMENSION_BUILDERS.keys()) == expected


# ── Metrics computation ──────────────────────────────────────────


class TestMetrics:
    def test_basic_metrics(self, enriched_df):
        """Metrics are computed correctly on a non-empty slice."""
        m = compute_cell_metrics(enriched_df)
        assert m["n_trades"] == len(enriched_df)
        assert m["n_total"] == len(enriched_df)
        assert isinstance(m["avg_pnl"], float)
        assert 0 <= m["win_rate"] <= 1
        assert isinstance(m["sharpe"], float)

    def test_empty_slice(self):
        """Empty DataFrame produces zero-count metrics."""
        empty = pd.DataFrame({"realized_pnl": pd.Series(dtype=float)})
        m = compute_cell_metrics(empty)
        assert m["n_trades"] == 0
        assert m["n_total"] == 0
        assert m["avg_pnl"] is None

    def test_tp_hit_rates(self, enriched_df):
        """TP hit rates are computed when trajectory columns exist."""
        m = compute_cell_metrics(enriched_df)
        assert m["tp50_hit_rate"] is not None
        assert 0 <= m["tp50_hit_rate"] <= 1

    def test_max_adverse_favorable(self, enriched_df):
        """Max adverse/favorable averages are computed."""
        m = compute_cell_metrics(enriched_df)
        assert m["avg_max_adverse"] is not None
        assert m["avg_max_adverse"] < 0
        assert m["avg_max_favorable"] is not None
        assert m["avg_max_favorable"] > 0

    def test_n_total_differs_with_tp_column(self, enriched_df):
        """n_total counts all rows while n_trades counts non-NaN PnL rows."""
        m = compute_cell_metrics(enriched_df, pnl_col="first_tp50_pnl")
        assert m["n_total"] == len(enriched_df)
        assert m["n_trades"] <= m["n_total"]
        assert m["n_trades"] < m["n_total"]  # some rows have NaN tp50


# ── Filter parsing ───────────────────────────────────────────────


class TestFilterParsing:
    def test_simple_filter(self):
        """Single condition parses correctly."""
        result = parse_filter_expr("spx_drop<-0.5")
        assert len(result) == 1
        col, op, val = result[0]
        assert col == "prev_spx_return"
        assert op == "<"
        assert val == -0.5

    def test_multiple_filters(self):
        """Comma-separated conditions parse correctly."""
        result = parse_filter_expr("spx_drop<-0.5,vix>25")
        assert len(result) == 2

    def test_ge_operator(self):
        """Greater-than-or-equal parses correctly."""
        result = parse_filter_expr("vix>=30")
        assert result[0] == ("vix", ">=", 30.0)

    def test_unknown_dimension_raises(self):
        """Unknown dimension name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown filter dimension"):
            parse_filter_expr("unknown_dim>5")

    def test_bad_syntax_raises(self):
        """Malformed token raises ValueError."""
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_filter_expr("bad token here")

    def test_apply_filters(self, enriched_df):
        """apply_filters reduces the DataFrame."""
        filters = parse_filter_expr("vix>20")
        filtered = apply_filters(enriched_df, filters)
        assert len(filtered) < len(enriched_df)
        assert (filtered["vix"] > 20).all()

    def test_apply_multiple_filters(self, enriched_df):
        """Multiple AND filters further reduce the set."""
        f1 = apply_filters(enriched_df, parse_filter_expr("vix>20"))
        f2 = apply_filters(enriched_df, parse_filter_expr("vix>20,dte==5"))
        assert len(f2) <= len(f1)
        if len(f2) > 0:
            assert (f2["dte_target"] == 5).all()


# ── Analysis modes ───────────────────────────────────────────────


class TestAnalysisModes:
    def test_1d_breakdown_returns_dataframe(self, enriched_df):
        """1D breakdown returns a non-empty DataFrame."""
        result = run_1d_breakdown(enriched_df, "dte", min_trades=1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "bucket" in result.columns
        assert "n_trades" in result.columns

    def test_1d_min_trades_filter(self, enriched_df):
        """Buckets with fewer trades than min_trades are excluded."""
        all_result = run_1d_breakdown(enriched_df, "dte", min_trades=1)
        strict_result = run_1d_breakdown(enriched_df, "dte", min_trades=999)
        assert len(strict_result) <= len(all_result)

    def test_2d_crosstab_returns_dataframe(self, enriched_df):
        """2D cross-tab returns a non-empty DataFrame."""
        result = run_2d_crosstab(enriched_df, "side", "dte", min_trades=1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "side" in result.columns
        assert "dte" in result.columns

    def test_regime_report_returns_list(self, enriched_df):
        """Regime report returns a list of (title, DataFrame) tuples."""
        report = run_regime_report(enriched_df, min_trades=1)
        assert isinstance(report, list)
        assert len(report) > 0
        for title, rdf in report:
            assert isinstance(title, str)
            assert isinstance(rdf, pd.DataFrame)

    def test_regime_report_includes_width_delta(self, enriched_df):
        """Regime report includes width and delta breakdown sections."""
        report = run_regime_report(enriched_df, min_trades=1)
        titles = [t for t, _ in report]
        assert any("Delta" in t for t in titles)
        assert any("Width" in t for t in titles)
        assert any("SPX Drop x Width" in t for t in titles)

    def test_unknown_dimension_raises(self, enriched_df):
        """Unknown dimension in 1D breakdown raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dimension"):
            run_1d_breakdown(enriched_df, "nonexistent")


# ── Output formatting ────────────────────────────────────────────


class TestFormatting:
    def test_format_table_nonempty(self, enriched_df):
        """format_table produces non-empty output for valid data."""
        result = run_1d_breakdown(enriched_df, "dte", min_trades=1)
        output = format_table(result, "Test Title")
        assert "Test Title" in output
        assert "Trades" in output

    def test_format_table_empty(self):
        """format_table handles empty DataFrame gracefully."""
        output = format_table(pd.DataFrame(), "Empty Section")
        assert "no data" in output
