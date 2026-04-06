"""Tests for the production data -> backtest bridge.

Covers:
- export_production_data: chain export, underlying parquet, calendar merge
- generate_training_data: production adapters, auto-fallback loaders,
  merged day discovery
- backtest_strategy: dynamic walk-forward window generation
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


# ── Production adapter tests ─────────────────────────────────────

from generate_training_data import (
    _symbol_to_iid,
    _production_chain_cache,
    definitions_from_production,
    cbbo_from_production,
    statistics_from_production,
    load_production_chain,
    PRODUCTION_CHAINS_DIR,
    PRODUCTION_UNDERLYING_DIR,
)


def _sample_chain_df() -> pd.DataFrame:
    """Build a minimal production chain DataFrame for adapter tests."""
    ts = pd.Timestamp("2026-04-03 14:30:00", tz="UTC")
    return pd.DataFrame([
        {
            "ts": ts,
            "option_symbol": "SPXW  260403C05700000",
            "expiration": date(2026, 4, 3),
            "strike": 5700.0,
            "option_right": "C",
            "bid": 12.50,
            "ask": 13.00,
            "open_interest": 1500,
            "delta": 0.25,
            "gamma": 0.003,
        },
        {
            "ts": ts,
            "option_symbol": "SPXW  260403P05690000",
            "expiration": date(2026, 4, 3),
            "strike": 5690.0,
            "option_right": "P",
            "bid": 8.20,
            "ask": 8.70,
            "open_interest": 800,
            "delta": -0.15,
            "gamma": 0.002,
        },
        {
            "ts": ts + timedelta(minutes=5),
            "option_symbol": "SPXW  260403C05700000",
            "expiration": date(2026, 4, 3),
            "strike": 5700.0,
            "option_right": "C",
            "bid": 12.60,
            "ask": 13.10,
            "open_interest": 1500,
            "delta": 0.26,
            "gamma": 0.003,
        },
    ])


class TestSymbolToIid:
    def test_deterministic(self):
        sym = "SPXW  260403C05700000"
        assert _symbol_to_iid(sym) == _symbol_to_iid(sym)

    def test_different_symbols_differ(self):
        a = _symbol_to_iid("SPXW  260403C05700000")
        b = _symbol_to_iid("SPXW  260403P05700000")
        assert a != b

    def test_returns_positive_int(self):
        iid = _symbol_to_iid("SPXW  260403C05700000")
        assert isinstance(iid, int)
        assert iid > 0


class TestDefinitionsFromProduction:
    def test_returns_expected_columns(self):
        chain = _sample_chain_df()
        defs = definitions_from_production(chain)
        assert "instrument_id" in defs.columns
        assert "raw_symbol" in defs.columns
        assert "strike_price" in defs.columns
        assert "expiration" in defs.columns
        assert "instrument_class" in defs.columns

    def test_unique_per_symbol(self):
        chain = _sample_chain_df()
        defs = definitions_from_production(chain)
        # 3 rows but only 2 unique option_symbols
        assert len(defs) == 2

    def test_instrument_id_is_int(self):
        chain = _sample_chain_df()
        defs = definitions_from_production(chain)
        for iid in defs["instrument_id"]:
            assert isinstance(iid, (int, np.integer))


class TestCbboFromProduction:
    def test_returns_expected_columns(self):
        chain = _sample_chain_df()
        cbbo = cbbo_from_production(chain)
        assert set(cbbo.columns) == {"instrument_id", "ts", "bid_px_00", "ask_px_00"}

    def test_row_count_preserved(self):
        chain = _sample_chain_df()
        cbbo = cbbo_from_production(chain)
        assert len(cbbo) == 3

    def test_prices_mapped_correctly(self):
        chain = _sample_chain_df()
        cbbo = cbbo_from_production(chain)
        assert cbbo.iloc[0]["bid_px_00"] == 12.50
        assert cbbo.iloc[0]["ask_px_00"] == 13.00


class TestStatisticsFromProduction:
    def test_returns_oi_columns(self):
        chain = _sample_chain_df()
        stats = statistics_from_production(chain)
        assert stats is not None
        assert set(stats.columns) == {"instrument_id", "oi"}

    def test_keeps_latest_oi(self):
        chain = _sample_chain_df()
        stats = statistics_from_production(chain)
        # Should have 2 unique symbols
        assert len(stats) == 2

    def test_returns_none_on_empty_oi(self):
        chain = _sample_chain_df()
        chain["open_interest"] = None
        result = statistics_from_production(chain)
        assert result is None


class TestLoadProductionChain:
    def setup_method(self):
        _production_chain_cache.clear()

    def test_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "generate_training_data.PRODUCTION_CHAINS_DIR", tmp_path
        )
        assert load_production_chain("20260403") is None

    def test_reads_parquet(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "generate_training_data.PRODUCTION_CHAINS_DIR", tmp_path
        )
        chain = _sample_chain_df()
        chain.to_parquet(tmp_path / "20260403.parquet", index=False)
        loaded = load_production_chain("20260403")
        assert loaded is not None
        assert len(loaded) == 3

    def test_cache_avoids_redundant_reads(self, tmp_path, monkeypatch):
        """Second call for the same day should return cached result."""
        monkeypatch.setattr(
            "generate_training_data.PRODUCTION_CHAINS_DIR", tmp_path
        )
        chain = _sample_chain_df()
        chain.to_parquet(tmp_path / "20260403.parquet", index=False)

        first = load_production_chain("20260403")
        assert first is not None
        assert "20260403" in _production_chain_cache

        second = load_production_chain("20260403")
        assert second is first  # same object from cache


# ── Auto-fallback loader tests ──────────────────────────────────

from generate_training_data import (
    load_definitions,
    load_cbbo,
    load_statistics,
    load_spy_equity,
    load_frd_quotes,
    _available_day_files,
)


class TestAutoFallbackDefinitions:
    def setup_method(self):
        _production_chain_cache.clear()

    def test_falls_back_to_production(self, tmp_path, monkeypatch):
        """When no .dbn.zst exists, should read production parquet."""
        monkeypatch.setattr("generate_training_data.SPXW_DEFS", tmp_path / "empty_defs")
        monkeypatch.setattr("generate_training_data.SPX_DEFS", tmp_path / "empty_defs2")
        monkeypatch.setattr("generate_training_data.PRODUCTION_CHAINS_DIR", tmp_path)

        chain = _sample_chain_df()
        chain.to_parquet(tmp_path / "20260403.parquet", index=False)

        defs = load_definitions("20260403")
        assert defs is not None
        assert "instrument_id" in defs.columns
        assert len(defs) == 2

    def test_returns_none_when_no_source(self, tmp_path, monkeypatch):
        monkeypatch.setattr("generate_training_data.SPXW_DEFS", tmp_path / "empty1")
        monkeypatch.setattr("generate_training_data.SPX_DEFS", tmp_path / "empty2")
        monkeypatch.setattr("generate_training_data.PRODUCTION_CHAINS_DIR", tmp_path / "empty3")
        assert load_definitions("20260403") is None


class TestAutoFallbackCbbo:
    def setup_method(self):
        _production_chain_cache.clear()

    def test_falls_back_to_production(self, tmp_path, monkeypatch):
        monkeypatch.setattr("generate_training_data.SPXW_CBBO", tmp_path / "empty1")
        monkeypatch.setattr("generate_training_data.SPX_CBBO", tmp_path / "empty2")
        monkeypatch.setattr("generate_training_data.PRODUCTION_CHAINS_DIR", tmp_path)

        chain = _sample_chain_df()
        chain.to_parquet(tmp_path / "20260403.parquet", index=False)

        cbbo = load_cbbo("20260403")
        assert cbbo is not None
        assert "instrument_id" in cbbo.columns
        assert "bid_px_00" in cbbo.columns


class TestAutoFallbackStatistics:
    def setup_method(self):
        _production_chain_cache.clear()

    def test_falls_back_to_production(self, tmp_path, monkeypatch):
        monkeypatch.setattr("generate_training_data.SPXW_STATS", tmp_path / "empty1")
        monkeypatch.setattr("generate_training_data.SPX_STATS", tmp_path / "empty2")
        monkeypatch.setattr("generate_training_data.PRODUCTION_CHAINS_DIR", tmp_path)

        chain = _sample_chain_df()
        chain.to_parquet(tmp_path / "20260403.parquet", index=False)

        stats = load_statistics("20260403")
        assert stats is not None
        assert "oi" in stats.columns


# ── Day discovery tests ──────────────────────────────────────────

class TestAvailableDayFiles:
    def test_merges_dbn_and_parquet(self, tmp_path, monkeypatch):
        """Should include days from both .dbn.zst and .parquet sources."""
        monkeypatch.setattr("generate_training_data.PRODUCTION_CHAINS_DIR", tmp_path / "prod")
        (tmp_path / "prod").mkdir()

        # Databento day
        (tmp_path / "20260401.dbn.zst").touch()
        (tmp_path / "20260402.dbn.zst").touch()
        # Production day (no overlap)
        (tmp_path / "prod" / "20260403.parquet").touch()
        # Production day (overlaps with Databento)
        (tmp_path / "prod" / "20260401.parquet").touch()

        days = _available_day_files(tmp_path)
        assert "20260401" in days
        assert "20260402" in days
        assert "20260403" in days
        # No duplicates
        assert len(days) == 3

    def test_sorted_output(self, tmp_path, monkeypatch):
        monkeypatch.setattr("generate_training_data.PRODUCTION_CHAINS_DIR", tmp_path / "prod")
        (tmp_path / "prod").mkdir()

        (tmp_path / "20260403.dbn.zst").touch()
        (tmp_path / "20260401.dbn.zst").touch()
        (tmp_path / "prod" / "20260402.parquet").touch()

        days = _available_day_files(tmp_path)
        assert days == ["20260401", "20260402", "20260403"]


# ── Underlying quotes fallback tests ─────────────────────────────

class TestLoadFrdQuotesFallback:
    def test_loads_production_when_frd_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("generate_training_data.PRODUCTION_UNDERLYING_DIR", tmp_path)

        prod_df = pd.DataFrame({
            "ts": pd.to_datetime(["2026-04-03 14:30:00+00:00", "2026-04-03 14:35:00+00:00"]),
            "close": [18.5, 18.6],
        })
        prod_df.to_parquet(tmp_path / "VIX_1min.parquet", index=False)

        result = load_frd_quotes(tmp_path / "nonexistent.parquet", "VIX")
        assert len(result) == 2
        assert "last" in result.columns
        assert result.iloc[0]["last"] == 18.5

    def test_merges_both_sources(self, tmp_path, monkeypatch):
        monkeypatch.setattr("generate_training_data.PRODUCTION_UNDERLYING_DIR", tmp_path / "prod")
        (tmp_path / "prod").mkdir()

        # FRD data (older)
        frd_df = pd.DataFrame({
            "ts": pd.to_datetime(["2026-04-01 14:30:00+00:00"]),
            "close": [17.0],
        })
        frd_path = tmp_path / "vix_1min.parquet"
        frd_df.to_parquet(frd_path, index=False)

        # Production data (newer)
        prod_df = pd.DataFrame({
            "ts": pd.to_datetime(["2026-04-03 14:30:00+00:00"]),
            "close": [18.5],
        })
        prod_df.to_parquet(tmp_path / "prod" / "VIX_1min.parquet", index=False)

        result = load_frd_quotes(frd_path, "VIX")
        assert len(result) == 2


class TestLoadSpyEquityFallback:
    def test_loads_production_when_databento_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("generate_training_data.SPY_EQUITY_PATH", tmp_path / "nonexistent.parquet")
        monkeypatch.setattr("generate_training_data.PRODUCTION_UNDERLYING_DIR", tmp_path)

        prod_df = pd.DataFrame({
            "ts": pd.to_datetime(["2026-04-03 14:30:00+00:00", "2026-04-03 14:35:00+00:00"]),
            "close": [570.0, 570.5],
        })
        prod_df.to_parquet(tmp_path / "SPY_1min.parquet", index=False)

        result = load_spy_equity()
        assert len(result) == 2
        assert "close" in result.columns

    def test_returns_empty_when_nothing_available(self, tmp_path, monkeypatch):
        monkeypatch.setattr("generate_training_data.SPY_EQUITY_PATH", tmp_path / "nope.parquet")
        monkeypatch.setattr("generate_training_data.PRODUCTION_UNDERLYING_DIR", tmp_path / "nope")

        result = load_spy_equity()
        assert len(result) == 0

    def test_merges_both_spy_sources(self, tmp_path, monkeypatch):
        """When both Databento and production SPY exist, merge them."""
        dbn_dir = tmp_path / "databento"
        dbn_dir.mkdir()
        prod_dir = tmp_path / "prod"
        prod_dir.mkdir()

        # Databento SPY (older dates, uses ts_event column)
        dbn_df = pd.DataFrame({
            "ts_event": pd.to_datetime(["2026-04-01 14:30:00+00:00"]),
            "close": [565.0],
        })
        dbn_path = dbn_dir / "spy_equity_1m.parquet"
        dbn_df.to_parquet(dbn_path, index=False)

        # Production SPY (newer dates, uses ts column)
        prod_df = pd.DataFrame({
            "ts": pd.to_datetime(["2026-04-03 14:30:00+00:00"]),
            "close": [570.0],
        })
        prod_df.to_parquet(prod_dir / "SPY_1min.parquet", index=False)

        monkeypatch.setattr("generate_training_data.SPY_EQUITY_PATH", dbn_path)
        monkeypatch.setattr("generate_training_data.PRODUCTION_UNDERLYING_DIR", prod_dir)

        result = load_spy_equity()
        assert len(result) == 2
        assert result.iloc[0]["close"] == 565.0
        assert result.iloc[1]["close"] == 570.0


# ── Export function tests ────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_production_data import (
    export_chain_data,
    export_underlying_parquet,
    export_economic_calendar_merged,
)


class TestExportChainData:
    def test_groups_by_day(self, tmp_path):
        """Verify per-day parquet creation from a mock engine."""
        ts1 = pd.Timestamp("2026-04-01 14:30:00", tz="UTC")
        ts2 = pd.Timestamp("2026-04-02 14:30:00", tz="UTC")
        mock_df = pd.DataFrame([
            {"ts": ts1, "option_symbol": "SPXW  260403C05700000",
             "expiration": "2026-04-03", "strike": 5700.0, "option_right": "C",
             "bid": 12.5, "ask": 13.0, "open_interest": 1000, "delta": 0.25, "gamma": 0.003},
            {"ts": ts2, "option_symbol": "SPXW  260403C05700000",
             "expiration": "2026-04-03", "strike": 5700.0, "option_right": "C",
             "bid": 12.6, "ask": 13.1, "open_interest": 1000, "delta": 0.26, "gamma": 0.003},
        ])

        class MockConn:
            def __enter__(self): return self
            def __exit__(self, *a): pass

        class MockEngine:
            def connect(self):
                return MockConn()

        # Patch pd.read_sql to return our mock data
        import export_production_data as epd
        original_read_sql = pd.read_sql

        def mock_read_sql(query, conn, params=None):
            return mock_df.copy()

        pd.read_sql = mock_read_sql
        try:
            days, rows = epd.export_chain_data(MockEngine(), output_dir=tmp_path)
        finally:
            pd.read_sql = original_read_sql

        assert days == 2
        assert rows == 2
        assert (tmp_path / "20260401.parquet").exists()
        assert (tmp_path / "20260402.parquet").exists()

        day1 = pd.read_parquet(tmp_path / "20260401.parquet")
        assert len(day1) == 1


class TestExportEconomicCalendarMerged:
    def test_merges_with_existing(self, tmp_path):
        existing = pd.DataFrame({
            "date": ["2026-01-15", "2026-01-28"],
            "event_type": ["OPEX", "FOMC"],
            "has_projections": [False, False],
            "is_triple_witching": [False, False],
        })
        out_path = tmp_path / "economic_calendar.csv"
        existing.to_csv(out_path, index=False)

        prod_df = pd.DataFrame({
            "date": ["2026-04-03", "2026-01-28"],
            "event_type": ["OPEX", "FOMC"],
            "has_projections": [False, True],
            "is_triple_witching": [False, False],
        })

        class MockConn:
            def __enter__(self): return self
            def __exit__(self, *a): pass

        class MockEngine:
            def connect(self): return MockConn()

        import export_production_data as epd
        original_read_sql = pd.read_sql

        def mock_read_sql(query, conn, params=None):
            return prod_df.copy()

        pd.read_sql = mock_read_sql
        try:
            n = epd.export_economic_calendar_merged(MockEngine(), output=out_path)
        finally:
            pd.read_sql = original_read_sql

        result = pd.read_csv(out_path)
        assert n == 3
        assert len(result) == 3
        # The 2026-01-28 FOMC should be updated (keep="last" from production)
        fomc_row = result[result["event_type"] == "FOMC"]
        assert len(fomc_row) == 1


# ── Dynamic walk-forward window tests ────────────────────────────

from backtest_strategy import generate_auto_windows


class TestGenerateAutoWindows:
    def test_basic_generation(self):
        windows = generate_auto_windows("2025-03-01", "2026-04-30")
        assert len(windows) >= 1
        for w in windows:
            assert "name" in w
            assert "train_start" in w
            assert "train_end" in w
            assert "test_start" in w
            assert "test_end" in w
            assert w["train_start"] < w["train_end"]
            assert w["test_start"] <= w["test_end"]
            assert w["train_end"] < w["test_start"]

    def test_window_names_sequential(self):
        windows = generate_auto_windows("2025-01-01", "2026-12-31")
        for i, w in enumerate(windows, 1):
            assert w["name"] == f"Auto-W{i}"

    def test_short_data_produces_few_windows(self):
        windows = generate_auto_windows("2026-01-01", "2026-06-30")
        # ~6 months should produce at most 1 window (6m train + 3m test = 9m)
        # but data is only 6 months, so test truncation
        assert len(windows) <= 1

    def test_very_short_data_produces_no_windows(self):
        windows = generate_auto_windows("2026-04-01", "2026-04-30")
        assert len(windows) == 0

    def test_windows_do_not_exceed_data_range(self):
        windows = generate_auto_windows("2025-03-01", "2026-04-30")
        for w in windows:
            assert w["train_start"] >= "2025-03-01"
            assert w["test_end"] <= "2026-04-30"

    def test_custom_window_sizes(self):
        windows = generate_auto_windows(
            "2025-01-01", "2026-12-31",
            train_months=3,
            test_months=1,
            step_months=2,
        )
        assert len(windows) >= 3
