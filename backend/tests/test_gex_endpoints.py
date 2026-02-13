from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from spx_backend.web.app import get_gex_curve, list_gex_dtes, list_gex_expirations


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
async def test_list_gex_dtes_shapes_response() -> None:
    session = _FakeSession(
        row_batches=[
            [
                SimpleNamespace(dte_days=0),
                SimpleNamespace(dte_days=1),
                SimpleNamespace(dte_days=3),
            ]
        ]
    )

    result = await list_gex_dtes(snapshot_id=123, db=session)

    assert result == {"snapshot_id": 123, "dte_days": [0, 1, 3]}


@pytest.mark.asyncio
async def test_list_gex_expirations_shapes_response() -> None:
    session = _FakeSession(
        row_batches=[
            [
                SimpleNamespace(expiration=date(2026, 2, 17), dte_days=1),
                SimpleNamespace(expiration=date(2026, 2, 18), dte_days=2),
            ]
        ]
    )

    result = await list_gex_expirations(snapshot_id=456, db=session)

    assert result == {
        "snapshot_id": 456,
        "items": [
            {"expiration": "2026-02-17", "dte_days": 1},
            {"expiration": "2026-02-18", "dte_days": 2},
        ],
    }


@pytest.mark.asyncio
async def test_get_gex_curve_custom_invalid_dates_returns_empty_without_query() -> None:
    session = _FakeSession(row_batches=[])

    result = await get_gex_curve(snapshot_id=1, dte_days=None, expirations_csv="bad-date", db=session)

    assert result["points"] == []
    assert result["expirations"] == []
    assert session.calls == []


@pytest.mark.asyncio
async def test_get_gex_curve_custom_dates_uses_expiration_filter_query() -> None:
    session = _FakeSession(
        row_batches=[
            [SimpleNamespace(strike=6800.0, gex_net=10.0, gex_calls=15.0, gex_puts=-5.0)],
        ]
    )

    result = await get_gex_curve(
        snapshot_id=2,
        dte_days=None,
        expirations_csv="2026-02-17,2026-02-18",
        db=session,
    )

    assert len(session.calls) == 1
    sql, params = session.calls[0]
    assert "WITH anchor AS" in sql
    assert "expiration IN (" in sql
    assert params["snapshot_id"] == 2
    assert params["expirations"] == [date(2026, 2, 17), date(2026, 2, 18)]
    assert result["points"] == [{"strike": 6800.0, "gex_net": 10.0, "gex_calls": 15.0, "gex_puts": -5.0}]


@pytest.mark.asyncio
async def test_get_gex_curve_all_mode_falls_back_to_gex_by_strike_when_needed() -> None:
    session = _FakeSession(
        row_batches=[
            [],
            [SimpleNamespace(strike=6800.0, gex_net=5.0, gex_calls=8.0, gex_puts=-3.0)],
        ]
    )

    result = await get_gex_curve(snapshot_id=3, dte_days=None, expirations_csv=None, db=session)

    assert len(session.calls) == 2
    first_sql, _ = session.calls[0]
    second_sql, _ = session.calls[1]
    assert "FROM gex_by_expiry_strike" in first_sql
    assert "FROM gex_by_strike" in second_sql
    assert result["points"] == [{"strike": 6800.0, "gex_net": 5.0, "gex_calls": 8.0, "gex_puts": -3.0}]
