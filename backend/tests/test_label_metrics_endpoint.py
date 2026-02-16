from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from spx_backend.web.app import get_label_metrics


class _FakeExecResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

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
async def test_get_label_metrics_shapes_response() -> None:
    session = _FakeSession(
        row_batches=[
            [
                SimpleNamespace(
                    resolved_count=10,
                    tp50_count=6,
                    tp100_count=3,
                    avg_realized_pnl=42.5,
                )
            ],
            [
                SimpleNamespace(
                    spread_side="put",
                    resolved_count=7,
                    tp50_count=5,
                    tp100_count=2,
                    avg_realized_pnl=50.0,
                ),
                SimpleNamespace(
                    spread_side="call",
                    resolved_count=3,
                    tp50_count=1,
                    tp100_count=1,
                    avg_realized_pnl=25.0,
                ),
            ],
        ]
    )

    result = await get_label_metrics(lookback_days=90, db=session)

    assert result["lookback_days"] == 90
    assert result["summary"]["resolved"] == 10
    assert result["summary"]["tp50"] == 6
    assert result["summary"]["tp100_at_expiry"] == 3
    assert result["summary"]["tp50_rate"] == 0.6
    assert result["summary"]["tp100_at_expiry_rate"] == 0.3
    assert len(result["by_side"]) == 2
    assert result["by_side"][0]["spread_side"] == "put"
    assert result["by_side"][0]["tp50_rate"] == pytest.approx(5 / 7)
    assert result["by_side"][1]["spread_side"] == "call"
    assert result["by_side"][1]["tp100_at_expiry_rate"] == pytest.approx(1 / 3)
    assert len(session.calls) == 2
    assert "FROM trade_candidates" in session.calls[0][0]
    assert "GROUP BY spread_side" in session.calls[1][0]


@pytest.mark.asyncio
async def test_get_label_metrics_rejects_invalid_lookback() -> None:
    session = _FakeSession(row_batches=[])
    with pytest.raises(HTTPException) as exc_info:
        await get_label_metrics(lookback_days=0, db=session)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_lookback_days"
