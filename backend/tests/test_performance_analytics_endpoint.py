from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from spx_backend.web.app import get_performance_analytics


class _FakeExecResult:
    """Minimal async execute result wrapper for endpoint unit tests."""

    def __init__(self, rows):
        """Store one batch of rows returned by a fake DB call."""
        self._rows = rows

    def fetchone(self):
        """Return first row to mimic SQLAlchemy result behavior."""
        return self._rows[0] if self._rows else None

    def fetchall(self):
        """Return all rows to mimic SQLAlchemy result behavior."""
        return self._rows


class _FakeSession:
    """Small fake async DB session that records SQL and params."""

    def __init__(self, row_batches):
        """Preload row batches consumed by successive execute() calls."""
        self._row_batches = list(row_batches)
        self.calls: list[tuple[str, dict]] = []

    async def execute(self, stmt, params=None):
        """Capture SQL text and return the next prepared result batch."""
        self.calls.append((str(stmt), params or {}))
        rows = self._row_batches.pop(0) if self._row_batches else []
        return _FakeExecResult(rows)


def _snapshot_row() -> SimpleNamespace:
    """Build a deterministic latest-snapshot row for endpoint tests."""
    return SimpleNamespace(
        analytics_snapshot_id=42,
        as_of_ts=datetime(2026, 2, 16, 15, 40, tzinfo=timezone.utc),
        source_trade_count=8,
        source_closed_count=6,
        source_open_count=2,
    )


def _mode_rows() -> list[SimpleNamespace]:
    """Build mixed realized/combined equity rows used in KPI checks."""
    return [
        SimpleNamespace(
            mode="realized",
            bucket_date=date(2026, 2, 10),
            trade_count=2,
            win_count=1,
            loss_count=1,
            pnl_sum=50.0,
            win_pnl_sum=120.0,
            loss_pnl_sum=-70.0,
        ),
        SimpleNamespace(
            mode="realized",
            bucket_date=date(2026, 2, 11),
            trade_count=1,
            win_count=0,
            loss_count=1,
            pnl_sum=-30.0,
            win_pnl_sum=0.0,
            loss_pnl_sum=-30.0,
        ),
        SimpleNamespace(
            mode="combined",
            bucket_date=date(2026, 2, 10),
            trade_count=2,
            win_count=1,
            loss_count=1,
            pnl_sum=50.0,
            win_pnl_sum=120.0,
            loss_pnl_sum=-70.0,
        ),
        SimpleNamespace(
            mode="combined",
            bucket_date=date(2026, 2, 11),
            trade_count=1,
            win_count=0,
            loss_count=1,
            pnl_sum=-20.0,
            win_pnl_sum=0.0,
            loss_pnl_sum=-20.0,
        ),
    ]


def _breakdown_rows() -> list[SimpleNamespace]:
    """Build grouped breakdown rows for response-shape assertions."""
    return [
        SimpleNamespace(
            dimension_type="side",
            dimension_value="put",
            trade_count=2,
            win_count=1,
            loss_count=1,
            pnl_sum=20.0,
            win_pnl_sum=70.0,
            loss_pnl_sum=-50.0,
        ),
        SimpleNamespace(
            dimension_type="side",
            dimension_value="call",
            trade_count=1,
            win_count=0,
            loss_count=1,
            pnl_sum=-10.0,
            win_pnl_sum=0.0,
            loss_pnl_sum=-10.0,
        ),
        SimpleNamespace(
            dimension_type="source",
            dimension_value="live",
            trade_count=3,
            win_count=1,
            loss_count=2,
            pnl_sum=10.0,
            win_pnl_sum=70.0,
            loss_pnl_sum=-60.0,
        ),
    ]


@pytest.mark.asyncio
async def test_get_performance_analytics_shapes_combined_response() -> None:
    """Verify combined-mode summary math and grouped breakdown payload shape."""
    session = _FakeSession(row_batches=[[_snapshot_row()], _mode_rows(), _breakdown_rows()])

    result = await get_performance_analytics(lookback_days=30, mode="combined", db=session)

    assert result["mode"] == "combined"
    assert result["snapshot"]["analytics_snapshot_id"] == 42
    assert result["summary"]["net_pnl"] == pytest.approx(30.0)
    assert result["summary"]["realized_net_pnl"] == pytest.approx(20.0)
    assert result["summary"]["unrealized_net_pnl"] == pytest.approx(10.0)
    assert result["summary"]["trade_count"] == 3
    assert result["summary"]["max_drawdown"] == pytest.approx(20.0)
    assert len(result["equity_curve"]) == 2
    assert result["equity_curve"][0]["date"] == "2026-02-10"
    assert len(result["breakdowns"]["side"]) == 2
    assert result["breakdowns"]["dte_bucket"] == []
    assert "FROM trade_performance_snapshots" in session.calls[0][0]
    assert "FROM trade_performance_equity_curve" in session.calls[1][0]
    assert "FROM trade_performance_breakdowns" in session.calls[2][0]


@pytest.mark.asyncio
async def test_get_performance_analytics_realized_mode_uses_realized_rows() -> None:
    """Ensure realized mode picks realized aggregates for headline KPI fields."""
    session = _FakeSession(row_batches=[[_snapshot_row()], _mode_rows(), _breakdown_rows()])

    result = await get_performance_analytics(lookback_days=30, mode="realized", db=session)

    assert result["mode"] == "realized"
    assert result["summary"]["net_pnl"] == pytest.approx(20.0)
    assert result["summary"]["combined_net_pnl"] == pytest.approx(30.0)
    assert result["summary"]["profit_factor"] == pytest.approx(1.2)


@pytest.mark.asyncio
async def test_get_performance_analytics_rejects_invalid_inputs() -> None:
    """Reject invalid lookback and mode inputs with 400 HTTP errors."""
    session = _FakeSession(row_batches=[])
    with pytest.raises(HTTPException) as lookback_exc:
        await get_performance_analytics(lookback_days=0, mode="combined", db=session)
    assert lookback_exc.value.status_code == 400
    assert lookback_exc.value.detail == "invalid_lookback_days"

    with pytest.raises(HTTPException) as mode_exc:
        await get_performance_analytics(lookback_days=10, mode="bad", db=session)
    assert mode_exc.value.status_code == 400
    assert mode_exc.value.detail == "invalid_mode"


@pytest.mark.asyncio
async def test_get_performance_analytics_returns_empty_payload_when_no_snapshot() -> None:
    """Return an empty analytics payload when aggregate tables are uninitialized."""
    session = _FakeSession(row_batches=[[]])

    result = await get_performance_analytics(lookback_days=30, mode="combined", db=session)

    assert result["snapshot"] is None
    assert result["summary"] is None
    assert result["equity_curve"] == []
