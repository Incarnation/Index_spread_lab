from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import spx_backend.jobs.decision_job as decision_module
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


class _NoopSession:
    """Minimal async session stub for run_once orchestration tests."""

    async def commit(self) -> None:
        """No-op commit used by patched decision-job workflows."""
        return None


class _SessionFactory:
    """Async context-manager factory that returns one fake session."""

    def __init__(self, session: _NoopSession):
        """Store the fake session instance yielded to job code."""
        self._session = session

    def __call__(self):
        """Return self to mimic SessionLocal callable behavior."""
        return self

    async def __aenter__(self) -> _NoopSession:
        """Yield fake session to the decision job."""
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        """Do not suppress exceptions raised inside the context."""
        return False


def _set_default_decision_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch common decision settings defaults used across unit tests.

    Pins ``portfolio_enabled=False`` so tests that target the legacy
    (non-portfolio) ``run_once`` path are not routed into
    ``_run_portfolio_managed`` when the env has PORTFOLIO_ENABLED=true.
    """
    monkeypatch.setattr(settings, "portfolio_enabled", False)
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


@pytest.mark.asyncio
async def test_side_limit_helpers_use_strategy_type_pattern(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_default_decision_settings(monkeypatch)
    job = DecisionJob()
    session = _FakeSession(rows=[SimpleNamespace(cnt=2), SimpleNamespace(cnt=3)])
    now_et = datetime(2026, 2, 12, 12, 0, 0, tzinfo=ZoneInfo("America/New_York"))

    open_count = await job._max_open_trades_by_side(session, "call")
    daily_count = await job._trades_today_by_side(session, now_et, "put")

    assert open_count == 2
    assert daily_count == 3

    first_sql, first_params = session.calls[0]
    second_sql, second_params = session.calls[1]
    assert "strategy_type LIKE :strategy_type_pattern" in first_sql
    assert first_params["strategy_type_pattern"] == "%_call"
    assert "strategy_type LIKE :strategy_type_pattern" in second_sql
    assert second_params["strategy_type_pattern"] == "%_put"


def _selection(
    *,
    short_symbol: str,
    long_symbol: str,
    score: float,
    target_dte: int,
    delta_target: float,
    spread_side: str,
) -> dict:
    """Build one ranked-selection payload for run_once top-N tests."""
    candidate = {
        "target_dte": target_dte,
        "delta_target": delta_target,
        "snapshot_id": 501,
        "expiration": "2026-02-20",
        "credit": 1.2,
        "delta_diff": 0.01,
        "score": score,
        "context_score": 0.0,
        "chosen_legs_json": {
            "short": {"symbol": short_symbol},
            "long": {"symbol": long_symbol},
            "spread_side": spread_side,
        },
        "strategy_params_json": {"spread_side": spread_side},
    }
    return {
        "chosen": candidate,
        "candidate_ref": {"candidate_id": 99, "feature_snapshot_id": 77},
        "decision_source": "rules",
        "model_prediction": None,
    }


@pytest.mark.asyncio
async def test_run_once_creates_top_n_with_dedupe(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_once should dedupe repeated legs and insert up to per-run cap."""
    _set_default_decision_settings(monkeypatch)
    monkeypatch.setattr(settings, "decision_dte_targets", "0")
    monkeypatch.setattr(settings, "decision_delta_targets", "0.10")
    monkeypatch.setattr(settings, "decision_spread_sides", "put")
    monkeypatch.setattr(settings, "decision_max_trades_per_run", 3)
    monkeypatch.setattr(settings, "decision_max_trades_per_day", 20)
    monkeypatch.setattr(settings, "decision_max_open_trades", 20)
    monkeypatch.setattr(settings, "decision_max_trades_per_side_per_day", 0)
    monkeypatch.setattr(settings, "decision_max_open_trades_per_side", 0)
    monkeypatch.setattr(decision_module, "SessionLocal", _SessionFactory(_NoopSession()))

    ranked = [
        _selection(short_symbol="SPXW_DUP", long_symbol="SPXW_DUPL", score=5.0, target_dte=0, delta_target=0.10, spread_side="put"),
        _selection(short_symbol="SPXW_DUP", long_symbol="SPXW_DUPL", score=4.9, target_dte=0, delta_target=0.10, spread_side="put"),
        _selection(short_symbol="SPXW_A", long_symbol="SPXW_AL", score=4.8, target_dte=3, delta_target=0.10, spread_side="put"),
        _selection(short_symbol="SPXW_B", long_symbol="SPXW_BL", score=4.7, target_dte=7, delta_target=0.20, spread_side="call"),
        _selection(short_symbol="SPXW_C", long_symbol="SPXW_CL", score=4.6, target_dte=10, delta_target=0.20, spread_side="call"),
    ]

    async def fake_max_open(self, session):  # noqa: ANN001
        """Return zero open trades for deterministic clipping checks."""
        return 0

    async def fake_trades_today(self, session, now_et):  # noqa: ANN001
        """Return zero trades today so per-run cap controls selection."""
        return 0

    async def fake_snapshot(self, session, now_et, target_dte, force=False):  # noqa: ANN001
        """Return one available snapshot so candidate generation continues."""
        return {
            "snapshot_id": 500 + int(target_dte),
            "ts": now_et.astimezone(ZoneInfo("UTC")),
            "expiration": date(2026, 2, 20),
            "target_dte": int(target_dte),
        }

    async def fake_option_rows(self, session, snapshot_id, spread_side):  # noqa: ANN001
        """Return one option row placeholder for candidate builder path."""
        return [{"symbol": f"{spread_side}_{snapshot_id}", "strike": 5000.0, "bid": 1.0, "ask": 1.1, "delta": -0.1}]

    def fake_build_candidate(
        self,
        *,
        options,
        target_dte,
        delta_target,
        spread_side,
        width_points,
        snapshot_id,
        expiration,
        spot,
        context,
    ):  # noqa: ANN001
        """Return a placeholder candidate; final ranking is monkeypatched."""
        return {
            "target_dte": target_dte,
            "delta_target": delta_target,
            "snapshot_id": snapshot_id,
            "expiration": expiration,
            "score": 1.0,
            "delta_diff": 0.01,
            "chosen_legs_json": {"short": {"symbol": "S"}, "long": {"symbol": "L"}, "spread_side": spread_side},
            "strategy_params_json": {"spread_side": spread_side},
        }

    async def fake_rank(self, session, now_et, candidates):  # noqa: ANN001
        """Return deterministic ranked list including one duplicate leg pair."""
        return ranked

    async def fake_spot(self, session, ts, underlying):  # noqa: ANN001
        """Return stable spot value for scoring payloads."""
        return 6000.0

    async def fake_context(self, session, now_et):  # noqa: ANN001
        """Return no context so score depends on candidate payload only."""
        return None

    decision_ids: list[int] = []
    trade_ids: list[int] = []

    async def fake_insert_decision(self, **kwargs):  # noqa: ANN001
        """Return incrementing decision IDs for created trade decisions."""
        decision_id = 900 + len(decision_ids) + 1
        decision_ids.append(decision_id)
        return decision_id

    async def fake_create_trade(self, **kwargs):  # noqa: ANN001
        """Return incrementing trade IDs for created trades."""
        trade_id = 700 + len(trade_ids) + 1
        trade_ids.append(trade_id)
        return trade_id

    monkeypatch.setattr(DecisionJob, "_max_open_trades", fake_max_open)
    monkeypatch.setattr(DecisionJob, "_trades_today", fake_trades_today)
    monkeypatch.setattr(DecisionJob, "_get_latest_snapshot_for_dte", fake_snapshot)
    monkeypatch.setattr(DecisionJob, "_get_option_rows", fake_option_rows)
    monkeypatch.setattr(DecisionJob, "_build_candidate", fake_build_candidate)
    monkeypatch.setattr(DecisionJob, "_rank_candidates_with_policy", fake_rank)
    monkeypatch.setattr(DecisionJob, "_insert_decision", fake_insert_decision)
    monkeypatch.setattr(DecisionJob, "_create_trade_from_decision", fake_create_trade)
    monkeypatch.setattr(DecisionJob, "_get_spot_price", fake_spot)
    monkeypatch.setattr(DecisionJob, "_get_latest_context", fake_context)

    result = await DecisionJob().run_once(force=True)

    assert result["skipped"] is False
    assert result["decisions_created_count"] == 3
    assert result["trades_created_count"] == 3
    assert len(result["decisions_created"]) == 3
    assert len(result["trades_created"]) == 3
    assert result["selection_meta"]["candidates_ranked"] == 5
    assert result["selection_meta"]["candidates_after_dedupe"] == 4
    assert result["selection_meta"]["duplicates_removed"] == 1


@pytest.mark.asyncio
async def test_run_once_clips_by_day_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_once should clip selected trades when daily capacity is limited."""
    _set_default_decision_settings(monkeypatch)
    monkeypatch.setattr(settings, "decision_dte_targets", "0")
    monkeypatch.setattr(settings, "decision_delta_targets", "0.10")
    monkeypatch.setattr(settings, "decision_spread_sides", "put")
    monkeypatch.setattr(settings, "decision_max_trades_per_run", 4)
    monkeypatch.setattr(settings, "decision_max_trades_per_day", 3)
    monkeypatch.setattr(settings, "decision_max_open_trades", 20)
    monkeypatch.setattr(settings, "decision_max_trades_per_side_per_day", 0)
    monkeypatch.setattr(settings, "decision_max_open_trades_per_side", 0)
    monkeypatch.setattr(decision_module, "SessionLocal", _SessionFactory(_NoopSession()))

    ranked = [
        _selection(short_symbol="SPXW_A", long_symbol="SPXW_AL", score=5.0, target_dte=0, delta_target=0.10, spread_side="put"),
        _selection(short_symbol="SPXW_B", long_symbol="SPXW_BL", score=4.9, target_dte=3, delta_target=0.10, spread_side="put"),
        _selection(short_symbol="SPXW_C", long_symbol="SPXW_CL", score=4.8, target_dte=7, delta_target=0.20, spread_side="call"),
    ]

    async def fake_max_open(self, session):  # noqa: ANN001
        """Return no open trades so daily cap is the only clip factor."""
        return 0

    async def fake_trades_today(self, session, now_et):  # noqa: ANN001
        """Return two existing trades so only one slot remains today."""
        return 2

    async def fake_snapshot(self, session, now_et, target_dte, force=False):  # noqa: ANN001
        """Return one available snapshot so candidate generation continues."""
        return {
            "snapshot_id": 510 + int(target_dte),
            "ts": now_et.astimezone(ZoneInfo("UTC")),
            "expiration": date(2026, 2, 20),
            "target_dte": int(target_dte),
        }

    async def fake_option_rows(self, session, snapshot_id, spread_side):  # noqa: ANN001
        """Return one option row placeholder for candidate builder path."""
        return [{"symbol": f"{spread_side}_{snapshot_id}", "strike": 5000.0, "bid": 1.0, "ask": 1.1, "delta": -0.1}]

    def fake_build_candidate(
        self,
        *,
        options,
        target_dte,
        delta_target,
        spread_side,
        width_points,
        snapshot_id,
        expiration,
        spot,
        context,
    ):  # noqa: ANN001
        """Return a placeholder candidate; ranking is monkeypatched in this test."""
        return {
            "target_dte": target_dte,
            "delta_target": delta_target,
            "snapshot_id": snapshot_id,
            "expiration": expiration,
            "score": 1.0,
            "delta_diff": 0.01,
            "chosen_legs_json": {"short": {"symbol": "S"}, "long": {"symbol": "L"}, "spread_side": spread_side},
            "strategy_params_json": {"spread_side": spread_side},
        }

    async def fake_rank(self, session, now_et, candidates):  # noqa: ANN001
        """Return deterministic ranked list with enough items to hit day cap."""
        return ranked

    async def fake_spot(self, session, ts, underlying):  # noqa: ANN001
        """Return stable spot value for candidate payloads."""
        return 6000.0

    async def fake_context(self, session, now_et):  # noqa: ANN001
        """Return no context so ranking remains deterministic in test."""
        return None

    created_trade_ids: list[int] = []

    async def fake_insert_decision(self, **kwargs):  # noqa: ANN001
        """Return incrementing decision IDs for one expected insertion."""
        return 1200 + len(created_trade_ids) + 1

    async def fake_create_trade(self, **kwargs):  # noqa: ANN001
        """Return incrementing trade IDs for one expected insertion."""
        trade_id = 2200 + len(created_trade_ids) + 1
        created_trade_ids.append(trade_id)
        return trade_id

    monkeypatch.setattr(DecisionJob, "_max_open_trades", fake_max_open)
    monkeypatch.setattr(DecisionJob, "_trades_today", fake_trades_today)
    monkeypatch.setattr(DecisionJob, "_get_latest_snapshot_for_dte", fake_snapshot)
    monkeypatch.setattr(DecisionJob, "_get_option_rows", fake_option_rows)
    monkeypatch.setattr(DecisionJob, "_build_candidate", fake_build_candidate)
    monkeypatch.setattr(DecisionJob, "_rank_candidates_with_policy", fake_rank)
    monkeypatch.setattr(DecisionJob, "_insert_decision", fake_insert_decision)
    monkeypatch.setattr(DecisionJob, "_create_trade_from_decision", fake_create_trade)
    monkeypatch.setattr(DecisionJob, "_get_spot_price", fake_spot)
    monkeypatch.setattr(DecisionJob, "_get_latest_context", fake_context)

    result = await DecisionJob().run_once(force=True)

    assert result["skipped"] is False
    assert result["trades_created_count"] == 1
    assert result["decisions_created_count"] == 1
    assert result["selection_meta"]["clipped_by"] == "day_cap"
