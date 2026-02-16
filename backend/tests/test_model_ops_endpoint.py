from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from spx_backend.web.app import get_model_ops


class _FakeExecResult:
    """Minimal async execute result wrapper for endpoint unit tests."""

    def __init__(self, rows):
        """Store one batch of rows returned by a fake DB call."""
        self._rows = rows

    def fetchone(self):
        """Return first row to mimic SQLAlchemy result behavior."""
        return self._rows[0] if self._rows else None


class _FakeSession:
    """Small fake async DB session that records SQL and params."""

    def __init__(self, row_batches):
        """Preload result batches consumed by successive execute() calls."""
        self._row_batches = list(row_batches)
        self.calls: list[tuple[str, dict]] = []

    async def execute(self, stmt, params=None):
        """Capture SQL text and return the next prepared result batch."""
        self.calls.append((str(stmt), params or {}))
        rows = self._row_batches.pop(0) if self._row_batches else []
        return _FakeExecResult(rows)


@pytest.mark.asyncio
async def test_get_model_ops_shapes_response() -> None:
    """Verify /api/model-ops response structure and model-ops payload mapping."""
    session = _FakeSession(
        row_batches=[
            [
                SimpleNamespace(
                    model_versions_count=3,
                    training_runs_count=2,
                    model_predictions_count=400,
                    model_predictions_24h_count=50,
                    latest_prediction_ts=datetime(2026, 2, 15, 15, 0, 0, tzinfo=timezone.utc),
                )
            ],
            [
                SimpleNamespace(
                    model_version_id=11,
                    version="wf_20260215150000",
                    rollout_status="shadow",
                    is_active=False,
                    created_at=datetime(2026, 2, 15, 15, 1, 0, tzinfo=timezone.utc),
                    promoted_at=None,
                    metrics_json={
                        "tp50_rate_test": 0.58,
                        "expectancy_test": 15.2,
                        "max_drawdown_test": 4200.0,
                        "tail_loss_proxy_test": -350.0,
                        "avg_margin_usage_test": 1700.0,
                    },
                )
            ],
            [
                SimpleNamespace(
                    training_run_id=21,
                    model_version_id=11,
                    status="COMPLETED",
                    started_at=datetime(2026, 2, 15, 14, 0, 0, tzinfo=timezone.utc),
                    finished_at=datetime(2026, 2, 15, 15, 0, 0, tzinfo=timezone.utc),
                    rows_train=300,
                    rows_test=80,
                    notes="gate_passed",
                    metrics_json={"gate": {"passed": True, "checks": {"tp50_rate": {"pass": True}}}},
                )
            ],
            [
                SimpleNamespace(
                    model_version_id=9,
                    version="wf_20260208150000",
                    rollout_status="active",
                    is_active=True,
                    created_at=datetime(2026, 2, 8, 15, 1, 0, tzinfo=timezone.utc),
                    promoted_at=datetime(2026, 2, 10, 15, 1, 0, tzinfo=timezone.utc),
                )
            ],
        ]
    )

    result = await get_model_ops(model_name="cand_bucket_v1", db=session)

    assert result["model_name"] == "cand_bucket_v1"
    assert result["counts"]["model_versions"] == 3
    assert result["counts"]["training_runs"] == 2
    assert result["counts"]["model_predictions"] == 400
    assert result["counts"]["model_predictions_24h"] == 50
    assert result["latest_model_version"]["model_version_id"] == 11
    assert result["latest_model_version"]["metrics"]["tp50_rate_test"] == 0.58
    assert result["latest_training_run"]["training_run_id"] == 21
    assert result["latest_training_run"]["gate"]["passed"] is True
    assert result["active_model_version"]["is_active"] is True
    assert result["warnings"] == []
    assert len(session.calls) == 4
    # Regression guard: fully qualify created_at to avoid Postgres ambiguity in joined subquery.
    assert "MAX(mp.created_at)" in session.calls[0][0]
    assert "MAX(created_at)" not in session.calls[0][0]


@pytest.mark.asyncio
async def test_get_model_ops_returns_warnings_when_no_data() -> None:
    """Ensure warnings are emitted when model-ops tables are empty."""
    session = _FakeSession(
        row_batches=[
            [
                SimpleNamespace(
                    model_versions_count=0,
                    training_runs_count=0,
                    model_predictions_count=0,
                    model_predictions_24h_count=0,
                    latest_prediction_ts=None,
                )
            ],
            [],
            [],
            [],
        ]
    )

    result = await get_model_ops(model_name="cand_bucket_v1", db=session)

    assert result["latest_model_version"] is None
    assert result["latest_training_run"] is None
    assert result["active_model_version"] is None
    assert "no_model_versions" in result["warnings"]
    assert "no_training_runs" in result["warnings"]
    assert "no_model_predictions" in result["warnings"]
