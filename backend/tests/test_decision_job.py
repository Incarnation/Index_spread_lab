from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from spx_backend.config import settings
from spx_backend.jobs.decision_job import DecisionJob


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeSession:
    def __init__(self, rows):
        self._rows = list(rows)
        self.calls: list[tuple[str, dict]] = []

    async def execute(self, stmt, params=None):
        self.calls.append((str(stmt), params or {}))
        row = self._rows.pop(0) if self._rows else None
        return _FakeResult(row)


def _set_default_decision_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "snapshot_underlying", "SPX")
    monkeypatch.setattr(settings, "decision_contracts", 1)
    monkeypatch.setattr(settings, "decision_snapshot_max_age_minutes", 15)
    monkeypatch.setattr(settings, "decision_dte_tolerance_days", 1)
    monkeypatch.setattr(settings, "snapshot_range_fallback_enabled", True)


def test_build_candidate_returns_credit_spread(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_default_decision_settings(monkeypatch)
    job = DecisionJob()
    options = [
        {"symbol": "P1", "strike": 6820.0, "bid": 18.0, "ask": 18.4, "delta": -0.20},
        {"symbol": "P2", "strike": 6795.0, "bid": 14.8, "ask": 15.2, "delta": -0.16},
        {"symbol": "P3", "strike": 6770.0, "bid": 11.9, "ask": 12.3, "delta": -0.12},
    ]

    candidate = job._build_candidate(
        options=options,
        target_dte=3,
        delta_target=0.20,
        spread_side="put",
        width_points=25.0,
        snapshot_id=123,
        expiration=date(2026, 2, 18),
        spot=6830.0,
        context=None,
    )

    assert candidate is not None
    assert candidate["snapshot_id"] == 123
    assert candidate["chosen_legs_json"]["short"]["symbol"] == "P1"
    assert candidate["chosen_legs_json"]["long"]["symbol"] == "P2"
    assert candidate["credit"] > 0


def test_build_candidate_returns_none_when_credit_non_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_default_decision_settings(monkeypatch)
    job = DecisionJob()
    options = [
        {"symbol": "P1", "strike": 6820.0, "bid": 10.0, "ask": 10.2, "delta": -0.20},
        {"symbol": "P2", "strike": 6795.0, "bid": 10.4, "ask": 10.8, "delta": -0.16},
    ]

    candidate = job._build_candidate(
        options=options,
        target_dte=3,
        delta_target=0.20,
        spread_side="put",
        width_points=25.0,
        snapshot_id=123,
        expiration=date(2026, 2, 18),
        spot=6830.0,
        context=None,
    )

    assert candidate is None


def test_context_score_supports_put_spreads() -> None:
    job = DecisionJob()
    score, flags = job._context_score(
        context={"gex_net": 1000.0, "zero_gamma_level": 6800.0, "vix": 18.0, "term_structure": 0.95},
        spread_side="put",
        spot=6830.0,
    )

    assert score > 0
    assert "gex_support" in flags
    assert "spot_above_zero_gamma" in flags


@pytest.mark.asyncio
async def test_snapshot_selection_queries_are_freshness_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_default_decision_settings(monkeypatch)
    job = DecisionJob()
    now_et = datetime(2026, 2, 12, 12, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    snapshot_row = (10, datetime(2026, 2, 12, 16, 58, 0, tzinfo=ZoneInfo("UTC")), date(2026, 2, 20), 8)
    session = _FakeSession(rows=[None, snapshot_row])

    result = await job._get_latest_snapshot_for_dte(session, now_et, target_dte=0, force=False)

    assert result is not None
    assert result["snapshot_id"] == 10
    assert len(session.calls) == 2
    for sql, params in session.calls:
        assert "ts <= :now_ts" in sql
        assert "ts >= :min_ts" in sql
        assert "now_ts" in params
        assert "min_ts" in params


@pytest.mark.asyncio
async def test_snapshot_selection_force_mode_disables_freshness_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_default_decision_settings(monkeypatch)
    job = DecisionJob()
    now_et = datetime(2026, 2, 12, 12, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    snapshot_row = (11, datetime(2026, 2, 12, 10, 0, 0, tzinfo=ZoneInfo("UTC")), date(2026, 2, 20), 8)
    session = _FakeSession(rows=[snapshot_row])

    result = await job._get_latest_snapshot_for_dte(session, now_et, target_dte=0, force=True)

    assert result is not None
    sql, params = session.calls[0]
    assert "ts <= :now_ts" in sql
    assert "ts >= :min_ts" not in sql
    assert "now_ts" in params
