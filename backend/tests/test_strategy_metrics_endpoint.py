from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from spx_backend.web.app import get_strategy_metrics


class _FakeExecResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeSession:
    def __init__(self, row_batches):
        self._row_batches = list(row_batches)
        self.calls: list[tuple[str, dict]] = []

    async def execute(self, stmt, params=None):
        self.calls.append((str(stmt), params or {}))
        rows = self._row_batches.pop(0) if self._row_batches else []
        return _FakeExecResult(rows)


@pytest.mark.asyncio
async def test_get_strategy_metrics_shapes_response() -> None:
    rows = [
        SimpleNamespace(
            ts=datetime(2026, 2, 1, 15, 0, 0, tzinfo=timezone.utc),
            realized_pnl=50.0,
            hit_tp50=True,
            hit_tp100=False,
            spread_side="put",
            max_loss=8.0,
            contracts=1,
        ),
        SimpleNamespace(
            ts=datetime(2026, 2, 1, 16, 0, 0, tzinfo=timezone.utc),
            realized_pnl=-20.0,
            hit_tp50=False,
            hit_tp100=False,
            spread_side="call",
            max_loss=10.0,
            contracts=1,
        ),
        SimpleNamespace(
            ts=datetime(2026, 2, 2, 15, 0, 0, tzinfo=timezone.utc),
            realized_pnl=30.0,
            hit_tp50=True,
            hit_tp100=True,
            spread_side="put",
            max_loss=7.5,
            contracts=1,
        ),
    ]
    session = _FakeSession(row_batches=[rows])

    result = await get_strategy_metrics(lookback_days=90, db=session)

    assert result["lookback_days"] == 90
    assert result["summary"]["resolved"] == 3
    assert result["summary"]["tp50"] == 2
    assert result["summary"]["tp100_at_expiry"] == 1
    assert result["summary"]["expectancy"] == pytest.approx((50.0 - 20.0 + 30.0) / 3)
    assert len(result["by_side"]) == 2
    put_row = next(r for r in result["by_side"] if r["spread_side"] == "put")
    assert put_row["resolved"] == 2
    assert put_row["tp50_rate"] == 1.0
    assert "FROM trade_candidates" in session.calls[0][0]


@pytest.mark.asyncio
async def test_get_strategy_metrics_rejects_invalid_lookback() -> None:
    session = _FakeSession(row_batches=[])
    with pytest.raises(HTTPException) as exc_info:
        await get_strategy_metrics(lookback_days=0, db=session)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_lookback_days"
