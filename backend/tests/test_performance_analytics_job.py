from __future__ import annotations

from datetime import datetime, timezone

import pytest

from spx_backend.jobs.performance_analytics_job import (
    TradeAnalyticsRow,
    _bucket_delta,
    _bucket_dte,
    _derive_spread_side,
    _dimension_values,
    _mode_pnl_points,
)


def test_bucket_dte_groups_ranges() -> None:
    """Bucket DTE values into the expected canonical dashboard ranges."""
    assert _bucket_dte(None) == "unknown"
    assert _bucket_dte(0) == "0"
    assert _bucket_dte(3) == "1-3"
    assert _bucket_dte(7) == "4-7"
    assert _bucket_dte(14) == "8-14"
    assert _bucket_dte(30) == "15+"


def test_bucket_delta_uses_absolute_delta_ranges() -> None:
    """Bucket delta values by absolute magnitude into stable risk bands."""
    assert _bucket_delta(None) == "unknown"
    assert _bucket_delta(0.05) == "0.00-0.10"
    assert _bucket_delta(-0.18) == "0.11-0.20"
    assert _bucket_delta(0.25) == "0.21-0.30"
    assert _bucket_delta(0.38) == "0.31-0.40"
    assert _bucket_delta(0.55) == "0.41+"


def test_mode_points_include_realized_and_combined_paths() -> None:
    """Emit realized+combined points for closed trades and combined for open trades."""
    closed_trade = TradeAnalyticsRow(
        trade_id=1,
        status="CLOSED",
        trade_source="live",
        strategy_type="PUT_CREDIT_SPREAD",
        entry_time=datetime(2026, 2, 10, 15, 0, tzinfo=timezone.utc),
        exit_time=datetime(2026, 2, 11, 19, 0, tzinfo=timezone.utc),
        target_dte=3,
        delta_target=0.15,
        current_pnl=None,
        realized_pnl=50.0,
    )
    open_trade = TradeAnalyticsRow(
        trade_id=2,
        status="OPEN",
        trade_source="paper",
        strategy_type="CALL_CREDIT_SPREAD",
        entry_time=datetime(2026, 2, 10, 15, 0, tzinfo=timezone.utc),
        exit_time=None,
        target_dte=5,
        delta_target=0.20,
        current_pnl=12.5,
        realized_pnl=None,
    )

    closed_points = _mode_pnl_points(closed_trade, as_of_date=datetime(2026, 2, 16, tzinfo=timezone.utc).date())
    open_points = _mode_pnl_points(open_trade, as_of_date=datetime(2026, 2, 16, tzinfo=timezone.utc).date())

    assert ("realized", 50.0, datetime(2026, 2, 11, tzinfo=timezone.utc).date()) in closed_points
    assert ("combined", 50.0, datetime(2026, 2, 11, tzinfo=timezone.utc).date()) in closed_points
    assert open_points == [("combined", 12.5, datetime(2026, 2, 16, tzinfo=timezone.utc).date())]


def test_dimension_values_cover_side_dte_delta_weekday_hour_source() -> None:
    """Build all dashboard breakdown dimensions from one normalized trade row."""
    row = TradeAnalyticsRow(
        trade_id=7,
        status="OPEN",
        trade_source="live",
        strategy_type="PUT_CREDIT_SPREAD",
        entry_time=datetime(2026, 2, 10, 15, 45, tzinfo=timezone.utc),
        exit_time=None,
        target_dte=5,
        delta_target=-0.17,
        current_pnl=10.0,
        realized_pnl=None,
    )

    dims = _dimension_values(row)

    assert _derive_spread_side(row.strategy_type) == "put"
    assert dims["side"] == "put"
    assert dims["dte_bucket"] == "4-7"
    assert dims["delta_bucket"] == "0.11-0.20"
    assert dims["weekday"] == "Tue"
    assert dims["hour"] == "15"
    assert dims["source"] == "live"
