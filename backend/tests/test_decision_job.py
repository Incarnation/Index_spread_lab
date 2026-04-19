from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
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
    """Patch common decision settings defaults used across unit tests.

    All decision runs are portfolio-managed; tests that need to bypass
    that should monkeypatch ``DecisionJob._run`` directly.
    """
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


def test_cboe_gex_underlyings_list_prefers_plural_env_and_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plural CBOE underlyings should override legacy single-symbol config."""
    monkeypatch.setattr(settings, "cboe_gex_underlyings", "spx, spy ,VIX,SPX,@@bad@@")
    monkeypatch.setattr(settings, "cboe_gex_underlying", "RUT")

    assert settings.cboe_gex_underlyings_list() == ["SPX", "SPY", "VIX"]


def test_cboe_gex_underlyings_list_falls_back_to_legacy_single_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy single-symbol CBOE config should still work when plural is empty."""
    monkeypatch.setattr(settings, "cboe_gex_underlyings", "")
    monkeypatch.setattr(settings, "cboe_gex_underlying", "spx")

    assert settings.cboe_gex_underlyings_list() == ["SPX"]


@pytest.mark.asyncio
async def test_snapshot_selection_queries_are_freshness_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_default_decision_settings(monkeypatch)
    job = DecisionJob()
    now_et = datetime(2026, 2, 12, 12, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    snapshot_row = (10, datetime(2026, 2, 12, 16, 58, 0, tzinfo=ZoneInfo("UTC")), date(2026, 2, 20), 8)
    session = _FakeSession(rows=[None, snapshot_row])

    result = await job._get_latest_snapshot_for_dte(session, now_et, target_dte=0)

    assert result is not None
    assert result["snapshot_id"] == 10
    assert len(session.calls) == 2
    for sql, params in session.calls:
        assert "ts <= :now_ts" in sql
        assert "ts >= :min_ts" in sql
        assert "now_ts" in params
        assert "min_ts" in params


@pytest.mark.asyncio
async def test_snapshot_selection_always_applies_freshness_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freshness must always be enforced.

    Replaces the previous ``force=True`` test which used to verify the
    bypass.  After the correctness fix, ``force`` no longer participates
    in snapshot selection (it remains a top-level run_once flag for
    bypassing entry-time / RTH checks only).
    """
    _set_default_decision_settings(monkeypatch)
    job = DecisionJob()
    now_et = datetime(2026, 2, 12, 12, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    snapshot_row = (11, datetime(2026, 2, 12, 10, 0, 0, tzinfo=ZoneInfo("UTC")), date(2026, 2, 20), 8)
    session = _FakeSession(rows=[snapshot_row])

    result = await job._get_latest_snapshot_for_dte(session, now_et, target_dte=0)

    assert result is not None
    sql, params = session.calls[0]
    assert "ts <= :now_ts" in sql
    # Freshness clause is now mandatory; this is the regression guard.
    assert "ts >= :min_ts" in sql
    assert "now_ts" in params
    assert "min_ts" in params


def test_decision_spread_sides_list_parses_csv_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "decision_spread_side", "put")
    monkeypatch.setattr(settings, "decision_spread_sides", "put, call, invalid, put")
    assert settings.decision_spread_sides_list() == ["put", "call"]

    monkeypatch.setattr(settings, "decision_spread_sides", "")
    monkeypatch.setattr(settings, "decision_spread_side", "call")
    assert settings.decision_spread_sides_list() == ["call"]


def test_decision_dte_targets_list_parses_zero_and_ten(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "decision_dte_targets", "0,3,5,7,10")
    assert settings.decision_dte_targets_list() == [0, 3, 5, 7, 10]


def test_build_candidate_call_credit_spread(monkeypatch: pytest.MonkeyPatch) -> None:
    """_build_candidate selects correct short (lower) and long (higher) strikes for calls."""
    _set_default_decision_settings(monkeypatch)
    job = DecisionJob()
    options = [
        {"symbol": "C1", "strike": 6850.0, "bid": 16.0, "ask": 16.4, "delta": 0.30},
        {"symbol": "C2", "strike": 6875.0, "bid": 12.5, "ask": 13.0, "delta": 0.22},
        {"symbol": "C3", "strike": 6900.0, "bid": 9.5, "ask": 10.0, "delta": 0.15},
    ]

    candidate = job._build_candidate(
        options=options,
        target_dte=3,
        delta_target=0.30,
        spread_side="call",
        width_points=25.0,
        snapshot_id=200,
        expiration=date(2026, 5, 15),
        spot=6830.0,
        context=None,
    )

    assert candidate is not None
    legs = candidate["chosen_legs_json"]
    assert legs["short"]["symbol"] == "C1"
    assert legs["long"]["symbol"] == "C2"
    assert legs["short"]["strike"] < legs["long"]["strike"], "Call short must be lower strike"
    assert candidate["credit"] > 0
    actual_width = abs(legs["short"]["strike"] - legs["long"]["strike"])
    assert legs["width_points"] == actual_width


def test_build_candidate_stores_actual_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """width_points in chosen_legs_json reflects actual strike distance, not config."""
    _set_default_decision_settings(monkeypatch)
    job = DecisionJob()
    options = [
        {"symbol": "P1", "strike": 6820.0, "bid": 18.0, "ask": 18.4, "delta": -0.20},
        {"symbol": "P2", "strike": 6800.0, "bid": 14.8, "ask": 15.2, "delta": -0.16},
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
    assert candidate["chosen_legs_json"]["width_points"] == 20.0
    assert candidate["chosen_legs_json"]["requested_width_points"] == 25.0


def test_build_candidate_rejects_inverted_put_spread(monkeypatch: pytest.MonkeyPatch) -> None:
    """A put spread where long strike >= short strike is rejected by integrity check."""
    _set_default_decision_settings(monkeypatch)
    job = DecisionJob()
    options = [
        {"symbol": "P1", "strike": 6800.0, "bid": 15.0, "ask": 15.4, "delta": -0.20},
        {"symbol": "P2", "strike": 6825.0, "bid": 14.0, "ask": 14.4, "delta": -0.25},
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

    assert candidate is None, "Should reject: long strike (6825) >= short strike (6800) for put"


def test_build_candidate_rejects_inverted_call_spread(monkeypatch: pytest.MonkeyPatch) -> None:
    """A call spread where long strike <= short strike is rejected by integrity check."""
    _set_default_decision_settings(monkeypatch)
    job = DecisionJob()
    options = [
        {"symbol": "C1", "strike": 6850.0, "bid": 12.0, "ask": 12.4, "delta": 0.25},
        {"symbol": "C2", "strike": 6825.0, "bid": 11.0, "ask": 11.4, "delta": 0.30},
    ]

    candidate = job._build_candidate(
        options=options,
        target_dte=3,
        delta_target=0.25,
        spread_side="call",
        width_points=25.0,
        snapshot_id=200,
        expiration=date(2026, 5, 15),
        spot=6830.0,
        context=None,
    )

    assert candidate is None, "Should reject: long strike (6825) <= short strike (6850) for call"


def test_context_score_supports_call_spreads() -> None:
    """Call spreads get positive GEX score when gex_net <= 0 and spot below zero gamma."""
    job = DecisionJob()
    score, flags = job._context_score(
        context={"gex_net": -500.0, "zero_gamma_level": 6850.0, "vix": 18.0, "term_structure": 0.95},
        spread_side="call",
        spot=6830.0,
    )

    assert score > 0
    assert "gex_support" in flags
    assert "spot_below_zero_gamma" in flags


def test_context_score_call_headwind_when_gex_positive() -> None:
    """Call spreads penalized when gex_net > 0 (dealer long gamma = headwind for calls)."""
    job = DecisionJob()
    score, flags = job._context_score(
        context={"gex_net": 1000.0, "zero_gamma_level": 6800.0, "vix": 18.0, "term_structure": 0.95},
        spread_side="call",
        spot=6830.0,
    )

    assert "gex_headwind" in flags
    assert "spot_above_zero_gamma" in flags


@pytest.mark.asyncio
async def test_get_latest_context_filters_by_underlying(monkeypatch) -> None:
    """_get_latest_context should only return rows where underlying='SPX'."""
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock
    from collections import namedtuple

    ContextRow = namedtuple("ContextRow", [
        "ts", "spx_price", "spy_price", "vix", "vix9d",
        "term_structure", "vvix", "skew", "gex_net", "zero_gamma_level",
        "gex_net_tradier", "zero_gamma_level_tradier",
        "gex_net_cboe", "zero_gamma_level_cboe",
    ])

    spx_row = ContextRow(
        ts=datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc),
        spx_price=5500.0, spy_price=550.0, vix=16.0, vix9d=15.0,
        term_structure=0.94, vvix=80.0, skew=130.0, gex_net=5e9, zero_gamma_level=5450.0,
        gex_net_tradier=1e12, zero_gamma_level_tradier=5400.0,
        gex_net_cboe=5e9, zero_gamma_level_cboe=5450.0,
    )

    captured_params: list[dict] = []

    class _FakeResult:
        """Captures params and returns the SPX row."""
        def fetchone(self):
            return spx_row

    class _FakeSession:
        """Session that captures SQL parameters for assertion."""
        async def execute(self, stmt, params=None):
            captured_params.append(dict(params or {}))
            return _FakeResult()

    job = DecisionJob()
    now_et = datetime(2026, 4, 8, 10, 30, 0, tzinfo=timezone.utc)
    result = await job._get_latest_context(_FakeSession(), now_et)

    assert result is not None
    assert result["spx_price"] == 5500.0

    # Verify the SQL was called with underlying='SPX'
    assert len(captured_params) == 1
    assert captured_params[0]["underlying"] == "SPX"
