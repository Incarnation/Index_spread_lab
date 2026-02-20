from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from spx_backend.web.routers.public import get_gex_curve, list_gex_dtes, list_gex_expirations, list_gex_snapshots


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
async def test_list_gex_snapshots_shapes_response_without_filter() -> None:
    """List GEX snapshots returns shaped rows and no symbol WHERE clause by default."""
    session = _FakeSession(
        row_batches=[
            [
                SimpleNamespace(
                    snapshot_id=862,
                    ts=datetime(2026, 2, 17, 8, 51, tzinfo=timezone.utc),
                    underlying="SPX",
                    source="TRADIER",
                    spot_price=6819.4,
                    gex_net=-2010.5,
                    gex_calls=120.0,
                    gex_puts=-2130.5,
                    gex_abs=2250.5,
                    zero_gamma_level=500.0,
                    method="test_method",
                )
            ]
        ]
    )

    result = await list_gex_snapshots(limit=10, db=session)

    assert len(session.calls) == 1
    sql, params = session.calls[0]
    assert "FROM gex_snapshots" in sql
    assert "UPPER(TRIM(underlying)) = :underlying" not in sql
    assert params == {"limit": 10}
    assert result["items"][0]["underlying"] == "SPX"
    assert result["items"][0]["source"] == "TRADIER"
    assert result["items"][0]["snapshot_id"] == 862


@pytest.mark.asyncio
async def test_list_gex_snapshots_applies_underlying_filter_case_insensitive() -> None:
    """Underlying filter is normalized to uppercase and applied in SQL."""
    session = _FakeSession(row_batches=[[]])

    await list_gex_snapshots(limit=25, underlying="spy", db=session)

    assert len(session.calls) == 1
    sql, params = session.calls[0]
    assert "UPPER(TRIM(underlying)) = :underlying" in sql
    assert params["underlying"] == "SPY"
    assert params["limit"] == 25


@pytest.mark.asyncio
async def test_list_gex_snapshots_applies_source_filter_case_insensitive() -> None:
    """Source filter is normalized to uppercase and applied in SQL."""
    session = _FakeSession(row_batches=[[]])

    await list_gex_snapshots(limit=12, source="cboe", db=session)

    assert len(session.calls) == 1
    sql, params = session.calls[0]
    assert "UPPER(TRIM(source)) = :source" in sql
    assert params["source"] == "CBOE"
    assert params["limit"] == 12


@pytest.mark.asyncio
async def test_list_gex_snapshots_applies_underlying_filter_with_surrounding_spaces() -> None:
    """Underlying filter trims spaces and still binds the normalized symbol."""
    session = _FakeSession(row_batches=[[]])

    await list_gex_snapshots(limit=15, underlying="  spx  ", db=session)

    assert len(session.calls) == 1
    _, params = session.calls[0]
    assert params["underlying"] == "SPX"
    assert params["limit"] == 15


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
    assert "gs.source = a.source" in sql
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
    assert "gs.source = a.source" in first_sql
    assert "FROM gex_by_expiry_strike" in first_sql
    assert "FROM gex_by_strike" in second_sql
    assert result["points"] == [{"strike": 6800.0, "gex_net": 5.0, "gex_calls": 8.0, "gex_puts": -3.0}]
