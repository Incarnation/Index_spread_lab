"""Tests for config validation and backtest engine path sanitization."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from spx_backend.config import Settings
from spx_backend.backtest.engine import BacktestConfig, BacktestEngine, sanitize_parquet_path


# ---------------------------------------------------------------------------
# Config parsing tests
# ---------------------------------------------------------------------------


class TestConfigParsing:
    """Validate Settings parsing helpers handle edge cases."""

    def _settings(self, **overrides):
        """Build a Settings instance with minimal required values."""
        defaults = {
            "DATABASE_URL": "postgresql+asyncpg://u:p@localhost/test",
            "TRADIER_ACCESS_TOKEN": "fake",
            "TRADIER_ACCOUNT_ID": "fake",
        }
        defaults.update(overrides)
        with patch.dict(os.environ, defaults, clear=False):
            return Settings()

    def test_dte_targets_list_normal(self):
        s = self._settings(SNAPSHOT_DTE_TARGETS="3,5,7")
        assert s.dte_targets_list() == [3, 5, 7]

    def test_dte_targets_list_empty(self):
        s = self._settings(SNAPSHOT_DTE_TARGETS="")
        assert s.dte_targets_list() == []

    def test_dte_targets_list_malformed(self):
        """Non-integer entries are silently dropped."""
        s = self._settings(SNAPSHOT_DTE_TARGETS="3,abc,7")
        assert s.dte_targets_list() == [3, 7]

    def test_decision_entry_times_normal(self):
        s = self._settings(DECISION_ENTRY_TIMES="09:45,10:30")
        assert s.decision_entry_times_list() == [(9, 45), (10, 30)]

    def test_decision_entry_times_empty(self):
        s = self._settings(DECISION_ENTRY_TIMES="")
        assert s.decision_entry_times_list() == []

    def test_decision_entry_times_malformed(self):
        """Malformed time entries are silently skipped."""
        s = self._settings(DECISION_ENTRY_TIMES="09:45,badtime,10:30")
        assert s.decision_entry_times_list() == [(9, 45), (10, 30)]

    def test_cors_origins_list(self):
        s = self._settings(CORS_ORIGINS="http://localhost:5173,https://app.example.com")
        assert s.cors_origins_list() == ["http://localhost:5173", "https://app.example.com"]

    def test_quote_symbols_list(self):
        s = self._settings(QUOTE_SYMBOLS="SPX,VIX,SPY")
        assert s.quote_symbols_list() == ["SPX", "VIX", "SPY"]

    def test_cboe_gex_underlyings_dedup(self):
        """Duplicate symbols are removed while preserving order."""
        s = self._settings(CBOE_GEX_UNDERLYINGS="SPX,SPY,SPX")
        result = s.cboe_gex_underlyings_list()
        assert result == ["SPX", "SPY"]

    def test_cboe_gex_underlyings_bad_symbols(self):
        """Symbols with invalid characters are dropped."""
        s = self._settings(CBOE_GEX_UNDERLYINGS="SPX,drop!,SPY")
        result = s.cboe_gex_underlyings_list()
        assert result == ["SPX", "SPY"]

    def test_decision_spread_sides(self):
        s = self._settings(DECISION_SPREAD_SIDES="put,call")
        assert s.decision_spread_sides_list() == ["put", "call"]

    def test_decision_spread_sides_invalid(self):
        """Only 'put' and 'call' are accepted."""
        s = self._settings(DECISION_SPREAD_SIDES="put,iron_condor,call")
        assert s.decision_spread_sides_list() == ["put", "call"]

    def test_decision_delta_targets(self):
        s = self._settings(DECISION_DELTA_TARGETS="0.10,0.15,0.20")
        assert s.decision_delta_targets_list() == [0.10, 0.15, 0.20]


# ---------------------------------------------------------------------------
# Backtest engine path sanitization
# ---------------------------------------------------------------------------

_DANGEROUS_PATHS = [
    "../../etc/passwd",
    "data'; DROP TABLE cbbo; --",
    "path/with;semicolons",
    "'; SELECT 1; --",
    "data/INSERT.parquet",
]


class TestBacktestPathSanitization:
    """Ensure dangerous paths are rejected by sanitize_parquet_path."""

    @pytest.mark.parametrize("path", _DANGEROUS_PATHS)
    def test_dangerous_paths_rejected(self, path: str):
        with pytest.raises(ValueError):
            sanitize_parquet_path(path)

    def test_safe_path_accepted(self):
        safe = "data/opra/cbbo_2024.parquet"
        assert sanitize_parquet_path(safe) == safe

    def test_glob_path_accepted(self):
        glob_path = "data/opra/cbbo_*.parquet"
        assert sanitize_parquet_path(glob_path) == glob_path


class TestBacktestConfigDefaults:
    """Verify BacktestConfig defaults are sensible."""

    def test_defaults(self):
        cfg = BacktestConfig(
            cbbo_parquet_glob="a.parquet",
            definitions_parquet_glob="b.parquet",
            underlying_parquet_glob="c.parquet",
        )
        assert cfg.spread_side in ("put", "call")
        assert cfg.take_profit_pct > 0
        assert cfg.stop_loss_pct > 0
        assert cfg.dte_targets == [3, 5, 7]
        assert cfg.contract_multiplier == 100
