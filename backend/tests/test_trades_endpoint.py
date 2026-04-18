from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from spx_backend.web.app import list_trades


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
async def test_list_trades_rejects_invalid_status() -> None:
    session = _FakeSession(row_batches=[])
    with pytest.raises(HTTPException) as exc_info:
        await list_trades(limit=10, status="bad", db=session)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_status"


@pytest.mark.asyncio
async def test_list_trades_shapes_response() -> None:
    session = _FakeSession(
        row_batches=[
            [
                SimpleNamespace(
                    trade_id=101,
                    decision_id=91,
                    status="OPEN",
                    trade_source="live",
                    strategy_type="PUT_CREDIT_SPREAD",
                    underlying="SPX",
                    entry_time=datetime(2026, 2, 1, 15, 0, 0),
                    exit_time=None,
                    last_mark_ts=datetime(2026, 2, 1, 15, 5, 0),
                    target_dte=3,
                    expiration=date(2026, 2, 4),
                    contracts=1,
                    contract_multiplier=100,
                    spread_width_points=25.0,
                    entry_credit=1.2,
                    current_exit_cost=0.8,
                    current_pnl=40.0,
                    realized_pnl=None,
                    max_profit=120.0,
                    max_loss=2380.0,
                    take_profit_target=60.0,
                    stop_loss_target=120.0,
                    exit_reason=None,
                    mark_count=2,
                    legs_json=[
                        {
                            "leg_index": 0,
                            "option_symbol": "SPXW260204P05700000",
                            "side": "STO",
                            "qty": 1,
                            "entry_price": 1.2,
                            "exit_price": None,
                            "strike": 5700.0,
                            "expiration": "2026-02-04",
                            "option_right": "P",
                        },
                        {
                            "leg_index": 1,
                            "option_symbol": "SPXW260204P05675000",
                            "side": "BTO",
                            "qty": 1,
                            "entry_price": 0.0,
                            "exit_price": None,
                            "strike": 5675.0,
                            "expiration": "2026-02-04",
                            "option_right": "P",
                        },
                    ],
                )
            ]
        ]
    )

    result = await list_trades(limit=25, status="OPEN", db=session)

    assert len(session.calls) == 1
    sql, params = session.calls[0]
    assert "FROM trades t" in sql
    assert "jsonb_agg" in sql
    assert params["status"] == "OPEN"
    assert params["limit"] == 25
    assert result["items"][0]["trade_id"] == 101
    assert result["items"][0]["mark_count"] == 2
    assert result["items"][0]["legs"][0]["option_symbol"] == "SPXW260204P05700000"


@pytest.mark.asyncio
async def test_list_trades_without_status_omits_status_bind_param() -> None:
    session = _FakeSession(row_batches=[[]])

    result = await list_trades(limit=100, status=None, db=session)

    assert result == {"items": []}
    assert len(session.calls) == 1
    sql, params = session.calls[0]
    assert "FROM trades t" in sql
    assert ":status" not in sql
    assert "status" not in params
    assert params["limit"] == 100
