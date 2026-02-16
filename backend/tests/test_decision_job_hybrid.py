from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from spx_backend.config import settings
from spx_backend.jobs.decision_job import DecisionJob


def _candidate(*, short_symbol: str, long_symbol: str, score: float) -> dict:
    return {
        "target_dte": 3,
        "delta_target": 0.2,
        "snapshot_id": 100,
        "expiration": "2026-02-18",
        "credit": 1.2,
        "delta_diff": 0.01,
        "score": score,
        "context_score": 0.0,
        "chosen_legs_json": {
            "short": {"symbol": short_symbol},
            "long": {"symbol": long_symbol},
            "spread_side": "put",
        },
        "strategy_params_json": {"spread_side": "put"},
    }


@pytest.mark.asyncio
async def test_select_candidate_with_policy_uses_rules_when_hybrid_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    job = DecisionJob()
    c1 = _candidate(short_symbol="S1", long_symbol="L1", score=2.0)
    c2 = _candidate(short_symbol="S2", long_symbol="L2", score=1.0)
    now_et = datetime(2026, 2, 12, 10, 0, 0, tzinfo=ZoneInfo("America/New_York"))

    monkeypatch.setattr(settings, "decision_hybrid_enabled", False)

    async def fake_find_candidate_reference(self, session, now_et_arg, chosen):
        if chosen["chosen_legs_json"]["short"]["symbol"] == "S1":
            return {"candidate_id": 1, "feature_snapshot_id": 11}
        return {"candidate_id": 2, "feature_snapshot_id": 22}

    monkeypatch.setattr(DecisionJob, "_find_candidate_reference", fake_find_candidate_reference)

    selection = await job._select_candidate_with_policy(session=object(), now_et=now_et, candidates=[c2, c1])
    assert selection["decision_source"] == "rules"
    assert selection["chosen"]["chosen_legs_json"]["short"]["symbol"] == "S1"


@pytest.mark.asyncio
async def test_select_candidate_with_policy_prefers_hybrid_rank_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    job = DecisionJob()
    c_rules_best = _candidate(short_symbol="S1", long_symbol="L1", score=3.0)
    c_model_best = _candidate(short_symbol="S2", long_symbol="L2", score=2.0)
    now_et = datetime(2026, 2, 12, 10, 0, 0, tzinfo=ZoneInfo("America/New_York"))

    monkeypatch.setattr(settings, "decision_hybrid_enabled", True)
    monkeypatch.setattr(settings, "decision_hybrid_min_probability", 0.5)
    monkeypatch.setattr(settings, "decision_hybrid_min_expected_pnl", 0.0)

    async def fake_find_candidate_reference(self, session, now_et_arg, chosen):
        short_symbol = chosen["chosen_legs_json"]["short"]["symbol"]
        if short_symbol == "S1":
            return {"candidate_id": 101, "feature_snapshot_id": 11}
        return {"candidate_id": 202, "feature_snapshot_id": 22}

    async def fake_get_hybrid_model_version(self, session):
        return {"model_version_id": 9, "model_name": "cand_bucket_v1", "version": "wf_x", "rollout_status": "active"}

    async def fake_get_candidate_prediction(self, session, candidate_id, model_version_id):
        if candidate_id == 101:
            return {"prediction_id": 1, "score_raw": 0.2, "probability_win": 0.6, "expected_value": 5.0, "rank_in_snapshot": 2}
        return {"prediction_id": 2, "score_raw": 0.9, "probability_win": 0.7, "expected_value": 12.0, "rank_in_snapshot": 1}

    monkeypatch.setattr(DecisionJob, "_find_candidate_reference", fake_find_candidate_reference)
    monkeypatch.setattr(DecisionJob, "_get_hybrid_model_version", fake_get_hybrid_model_version)
    monkeypatch.setattr(DecisionJob, "_get_candidate_prediction", fake_get_candidate_prediction)

    selection = await job._select_candidate_with_policy(
        session=object(),
        now_et=now_et,
        candidates=[c_rules_best, c_model_best],
    )
    assert selection["decision_source"] == "hybrid_model"
    assert selection["chosen"]["chosen_legs_json"]["short"]["symbol"] == "S2"
    assert selection["model_prediction"]["model_version_id"] == 9
