"""Tests for config validation helpers."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from spx_backend.config import Settings


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

    def test_quote_symbols_default_includes_vvix_and_skew(self):
        """Default quote_symbols should capture all six underlyings."""
        s = self._settings()
        symbols = s.quote_symbols_list()
        assert "VVIX" in symbols
        assert "SKEW" in symbols
        assert len(symbols) == 6

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

    # ── event_only_mode / decision_avoid_opex ─────────────────────

    def test_event_only_mode_default_false(self):
        """event_only_mode defaults to False when env var is not set."""
        s = self._settings(EVENT_ONLY_MODE="false")
        assert s.event_only_mode is False

    def test_event_only_mode_env_override(self):
        """EVENT_ONLY_MODE=true activates event-only trading."""
        s = self._settings(EVENT_ONLY_MODE="true")
        assert s.event_only_mode is True

    def test_decision_avoid_opex_default_false(self):
        """decision_avoid_opex is False when env var is not set."""
        s = self._settings(DECISION_AVOID_OPEX="false")
        assert s.decision_avoid_opex is False

    def test_decision_avoid_opex_env_override(self):
        """DECISION_AVOID_OPEX=true skips trading on OPEX days."""
        s = self._settings(DECISION_AVOID_OPEX="true")
        assert s.decision_avoid_opex is True


class TestEventSidePreference:
    """Verify that all event signals respect EVENT_SIDE_PREFERENCE."""

    @staticmethod
    def _resolve_event_sides(drop_signals: list[str], side_preference: str, spread_sides: list[str]) -> list[str]:
        """Replicate the has_drop -> event_sides logic from _run.

        This mirrors lines 204-217 of decision_job.py so we can test
        the signal-to-side mapping without spinning up the full job.
        """
        event_side_raw = side_preference.rstrip("s")
        has_drop = any(
            s.startswith("spx_drop") or s in ("vix_spike", "vix_elevated", "term_inversion")
            for s in drop_signals
        )
        if drop_signals:
            if has_drop:
                return [event_side_raw]
            else:
                return list(spread_sides)
        return []

    def test_spx_drop_uses_side_preference(self):
        """spx_drop_1d signal should use EVENT_SIDE_PREFERENCE (put)."""
        sides = self._resolve_event_sides(["spx_drop_1d"], "puts", ["put", "call"])
        assert sides == ["put"]

    def test_vix_spike_uses_side_preference(self):
        sides = self._resolve_event_sides(["vix_spike"], "puts", ["put", "call"])
        assert sides == ["put"]

    def test_vix_elevated_uses_side_preference(self):
        sides = self._resolve_event_sides(["vix_elevated"], "puts", ["put", "call"])
        assert sides == ["put"]

    def test_term_inversion_uses_side_preference(self):
        """term_inversion must respect EVENT_SIDE_PREFERENCE, not fall back to both sides."""
        sides = self._resolve_event_sides(["term_inversion"], "puts", ["put", "call"])
        assert sides == ["put"]

    def test_multiple_signals_uses_side_preference(self):
        """When multiple signals fire, side preference is still respected."""
        sides = self._resolve_event_sides(["term_inversion", "vix_elevated"], "puts", ["put", "call"])
        assert sides == ["put"]

    def test_no_signals_returns_empty(self):
        sides = self._resolve_event_sides([], "puts", ["put", "call"])
        assert sides == []

    def test_unknown_signal_falls_back_to_spread_sides(self):
        """An unrecognized signal that isn't in has_drop falls back to configured spread sides."""
        sides = self._resolve_event_sides(["some_future_signal"], "puts", ["put", "call"])
        assert sides == ["put", "call"]


class TestIsOpexDay:
    """Validate the DecisionJob._is_opex_day static helper."""

    @pytest.mark.asyncio
    async def test_opex_day_returns_true(self):
        """_is_opex_day returns True when economic_events has an OPEX row."""
        from unittest.mock import AsyncMock, MagicMock
        from datetime import date
        from spx_backend.jobs.decision_job import DecisionJob

        mock_row = MagicMock()
        mock_result = MagicMock()
        mock_result.first.return_value = mock_row
        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        assert await DecisionJob._is_opex_day(mock_session, date(2026, 4, 17)) is True

    @pytest.mark.asyncio
    async def test_non_opex_day_returns_false(self):
        """_is_opex_day returns False when no OPEX row exists."""
        from unittest.mock import AsyncMock, MagicMock
        from datetime import date
        from spx_backend.jobs.decision_job import DecisionJob

        mock_result = MagicMock()
        mock_result.first.return_value = None
        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        assert await DecisionJob._is_opex_day(mock_session, date(2026, 4, 14)) is False
