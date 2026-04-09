from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from spx_backend.config import settings
from spx_backend.jobs.feature_builder_job import FeatureBuilderJob, build_candidate_hash


def _candidate_json(short_symbol: str = "P_SHORT") -> dict:
    return {
        "underlying": "SPX",
        "snapshot_id": 101,
        "expiration": "2026-02-20",
        "target_dte": 3,
        "delta_target": 0.1,
        "spread_side": "put",
        "width_points": 25.0,
        "contracts": 1,
        "legs": {
            "short": {"symbol": short_symbol, "strike": 6800.0},
            "long": {"symbol": "P_LONG", "strike": 6775.0},
        },
    }


def test_candidate_hash_is_stable_for_equivalent_payloads() -> None:
    a = _candidate_json()
    b = {
        # same values, different key order
        "target_dte": 3,
        "underlying": "SPX",
        "expiration": "2026-02-20",
        "snapshot_id": 101,
        "spread_side": "put",
        "delta_target": 0.1,
        "contracts": 1,
        "width_points": 25.0,
        "legs": {
            "long": {"strike": 6775.0, "symbol": "P_LONG"},
            "short": {"strike": 6800.0, "symbol": "P_SHORT"},
        },
    }
    assert build_candidate_hash(a) == build_candidate_hash(b)


def test_candidate_hash_changes_when_legs_change() -> None:
    base_hash = build_candidate_hash(_candidate_json(short_symbol="P_SHORT_A"))
    changed_hash = build_candidate_hash(_candidate_json(short_symbol="P_SHORT_B"))
    assert base_hash != changed_hash


def test_build_candidate_json_includes_cboe_context() -> None:
    job = FeatureBuilderJob()
    candidate = {
        "snapshot_id": 101,
        "expiration": "2026-02-20",
        "target_dte": 3,
        "delta_target": 0.1,
        "credit": 1.2,
        "score": 2.5,
        "delta_diff": 0.01,
        "chosen_legs_json": {
            "spread_side": "put",
            "width_points": 10.0,
            "short": {"symbol": "P_SHORT", "strike": 6800.0, "qty": 1},
            "long": {"symbol": "P_LONG", "strike": 6790.0, "qty": 1},
            "context": {"vix": 18.0, "term_structure": 1.01, "spy_price": 600.0, "spx_price": 6000.0},
        },
    }
    cboe_context = {"expiry_gex_net": 1234.5, "gamma_wall_distance_ratio": 0.004}

    payload = job._build_candidate_json(candidate, cboe_context=cboe_context)

    assert payload["cboe_context"] == cboe_context


# ── run_once skip-path tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_run_once_skips_when_disabled(monkeypatch) -> None:
    """run_once returns skipped when feature_builder_enabled is False."""
    monkeypatch.setattr(settings, "feature_builder_enabled", False)
    job = FeatureBuilderJob()
    result = await job.run_once()
    assert result["skipped"] is True
    assert result["reason"] == "feature_builder_disabled"


@pytest.mark.asyncio
async def test_run_once_skips_when_not_entry_time(monkeypatch) -> None:
    """run_once returns skipped outside configured entry times."""
    monkeypatch.setattr(settings, "feature_builder_enabled", True)
    monkeypatch.setattr(settings, "decision_entry_times", "09:31,10:01")

    # Freeze time to 03:00 ET — not an entry time
    fixed = datetime(2026, 4, 9, 3, 0, tzinfo=ZoneInfo("America/New_York"))
    import spx_backend.jobs.feature_builder_job as fb_mod
    monkeypatch.setattr(fb_mod, "datetime", type("_DT", (), {"now": staticmethod(lambda tz=None: fixed)}))

    job = FeatureBuilderJob()
    result = await job.run_once()
    assert result["skipped"] is True
    assert result["reason"] == "not_entry_time"


@pytest.mark.asyncio
async def test_run_once_skips_when_missing_targets(monkeypatch) -> None:
    """run_once returns skipped when DTE or delta targets are empty."""
    monkeypatch.setattr(settings, "feature_builder_enabled", True)
    monkeypatch.setattr(settings, "decision_dte_targets", "")
    monkeypatch.setattr(settings, "decision_delta_targets", "0.20")

    job = FeatureBuilderJob()
    result = await job.run_once(force=True)
    assert result["skipped"] is True
    assert result["reason"] == "missing_targets"


@pytest.mark.asyncio
async def test_run_once_skips_when_missing_spread_sides(monkeypatch) -> None:
    """run_once returns skipped when no spread sides are configured."""
    monkeypatch.setattr(settings, "feature_builder_enabled", True)
    monkeypatch.setattr(settings, "decision_dte_targets", "3")
    monkeypatch.setattr(settings, "decision_delta_targets", "0.20")
    monkeypatch.setattr(settings, "decision_spread_sides", "")
    monkeypatch.setattr(settings, "decision_spread_side", "")

    job = FeatureBuilderJob()
    result = await job.run_once(force=True)
    assert result["skipped"] is True
    assert result["reason"] == "missing_spread_sides"
