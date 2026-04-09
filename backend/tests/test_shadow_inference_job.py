from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from spx_backend.config import settings
from spx_backend.jobs.shadow_inference_job import (
    classify_prediction,
    classify_uncertainty_level,
    compute_uncertainty_penalty,
)


def test_classify_prediction_trade_when_probability_and_ev_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Classify as trade when thresholds pass and uncertainty gates are clean."""
    monkeypatch.setattr(settings, "decision_hybrid_min_probability", 0.55)
    monkeypatch.setattr(settings, "decision_hybrid_min_expected_pnl", 5.0)
    monkeypatch.setattr(settings, "decision_hybrid_min_bucket_count", 1)
    monkeypatch.setattr(settings, "decision_hybrid_max_pnl_std", 50.0)
    assert classify_prediction(probability_win=0.60, expected_pnl=8.0, bucket_count=3, pnl_std=12.0) == "TRADE"


def test_classify_prediction_skip_when_probability_or_ev_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Classify as skip when probability or expected-value thresholds fail."""
    monkeypatch.setattr(settings, "decision_hybrid_min_probability", 0.55)
    monkeypatch.setattr(settings, "decision_hybrid_min_expected_pnl", 5.0)
    monkeypatch.setattr(settings, "decision_hybrid_min_bucket_count", 1)
    monkeypatch.setattr(settings, "decision_hybrid_max_pnl_std", 50.0)
    assert classify_prediction(probability_win=0.50, expected_pnl=10.0, bucket_count=3, pnl_std=12.0) == "SKIP"
    assert classify_prediction(probability_win=0.80, expected_pnl=4.9, bucket_count=3, pnl_std=12.0) == "SKIP"


def test_classify_prediction_skip_when_uncertainty_gate_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uncertainty gates should override otherwise acceptable EV/probability."""
    monkeypatch.setattr(settings, "decision_hybrid_min_probability", 0.55)
    monkeypatch.setattr(settings, "decision_hybrid_min_expected_pnl", 5.0)
    monkeypatch.setattr(settings, "decision_hybrid_min_bucket_count", 4)
    monkeypatch.setattr(settings, "decision_hybrid_max_pnl_std", 40.0)
    assert classify_prediction(probability_win=0.70, expected_pnl=15.0, bucket_count=2, pnl_std=20.0) == "SKIP"
    assert classify_prediction(probability_win=0.70, expected_pnl=15.0, bucket_count=6, pnl_std=55.0) == "SKIP"


def test_uncertainty_penalty_and_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """Penalty should increase for low support and high volatility buckets."""
    monkeypatch.setattr(settings, "decision_hybrid_min_bucket_count", 8)
    monkeypatch.setattr(settings, "decision_hybrid_max_pnl_std", 100.0)
    low_penalty = compute_uncertainty_penalty(bucket_count=12, pnl_std=80.0)
    high_penalty = compute_uncertainty_penalty(bucket_count=2, pnl_std=180.0)
    assert high_penalty > low_penalty
    assert classify_uncertainty_level(bucket_count=12, pnl_std=80.0) == "medium"
    assert classify_uncertainty_level(bucket_count=4, pnl_std=80.0) in {"medium", "high"}
    assert classify_uncertainty_level(bucket_count=2, pnl_std=180.0) == "high"


# ---------------------------------------------------------------------------
# XGBoost vs bucket scoring-path unit tests
# ---------------------------------------------------------------------------

def _make_xgb_model_dict(*, model_version_id: int = 99) -> dict[str, Any]:
    """Build a minimal model dict that triggers the xgb_entry_v1 path."""
    return {
        "model_version_id": model_version_id,
        "model_name": "spx_credit_spread_v1",
        "version": "xgb_test",
        "rollout_status": "active",
        "model_payload": {
            "model_type": "xgb_entry_v1",
            "classifier_json": "{}",
            "regressor_json": "{}",
            "feature_names": ["delta", "dte", "vix"],
        },
    }


def _make_bucket_model_dict(*, model_version_id: int = 88) -> dict[str, Any]:
    """Build a minimal model dict that triggers the bucket_empirical_v1 path."""
    return {
        "model_version_id": model_version_id,
        "model_name": "cand_bucket_v1",
        "version": "bucket_test",
        "rollout_status": "active",
        "model_payload": {"model_type": "bucket_empirical_v1"},
    }


class TestXgbScoringPath:
    """Verify the XGBoost branch bypasses bucket-centric uncertainty logic."""

    def test_xgb_no_uncertainty_penalty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """XGBoost predictions should not be penalised by bucket_count gates."""
        monkeypatch.setattr(settings, "decision_hybrid_min_bucket_count", 8)
        monkeypatch.setattr(settings, "decision_hybrid_max_pnl_std", 100.0)

        model = _make_xgb_model_dict()
        model_type = model["model_payload"]["model_type"]

        raw_utility = 12.5
        probability_win = 0.72
        expected_pnl = 45.0

        pred = {
            "probability_win": probability_win,
            "expected_pnl": expected_pnl,
            "utility_score": raw_utility,
            "bucket_count": 0,
            "pnl_std": 0.0,
            "source": "xgb_entry_v1",
        }

        if model_type == "xgb_entry_v1":
            score_raw = raw_utility
            uncertainty_penalty = 0.0
            uncertainty_level = "n/a"
        else:
            bucket_count = int(pred.get("bucket_count") or 0)
            pnl_std_val = float(pred.get("pnl_std") or 0.0)
            uncertainty_penalty = compute_uncertainty_penalty(
                bucket_count=bucket_count, pnl_std=pnl_std_val,
            )
            score_raw = raw_utility - uncertainty_penalty

        assert score_raw == raw_utility
        assert uncertainty_penalty == 0.0
        assert uncertainty_level == "n/a"

    def test_bucket_still_gets_penalty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bucket models should still receive the standard uncertainty penalty."""
        monkeypatch.setattr(settings, "decision_hybrid_min_bucket_count", 8)
        monkeypatch.setattr(settings, "decision_hybrid_max_pnl_std", 100.0)

        model = _make_bucket_model_dict()
        model_type = model["model_payload"]["model_type"]

        raw_utility = 10.0
        pred = {
            "probability_win": 0.60,
            "expected_pnl": 20.0,
            "utility_score": raw_utility,
            "bucket_count": 3,
            "pnl_std": 50.0,
        }

        if model_type == "xgb_entry_v1":
            score_raw = raw_utility
            uncertainty_penalty = 0.0
        else:
            bucket_count = int(pred.get("bucket_count") or 0)
            pnl_std_val = float(pred.get("pnl_std") or 0.0)
            uncertainty_penalty = compute_uncertainty_penalty(
                bucket_count=bucket_count, pnl_std=pnl_std_val,
            )
            score_raw = raw_utility - uncertainty_penalty

        assert uncertainty_penalty > 0.0
        assert score_raw < raw_utility

    def test_xgb_decision_trade_when_thresholds_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """XGBoost TRADE decision should depend only on prob/EV, not bucket_count."""
        monkeypatch.setattr(settings, "decision_hybrid_min_probability", 0.50)
        monkeypatch.setattr(settings, "decision_hybrid_min_expected_pnl", 0.0)
        monkeypatch.setattr(settings, "decision_hybrid_min_bucket_count", 8)

        probability_win = 0.65
        expected_pnl = 30.0
        model_type = "xgb_entry_v1"

        if model_type == "xgb_entry_v1":
            decision = (
                "TRADE"
                if (probability_win >= settings.decision_hybrid_min_probability
                    and expected_pnl >= settings.decision_hybrid_min_expected_pnl)
                else "SKIP"
            )
        else:
            decision = classify_prediction(
                probability_win=probability_win,
                expected_pnl=expected_pnl,
                bucket_count=0,
                pnl_std=0.0,
            )

        assert decision == "TRADE"

    def test_xgb_decision_skip_when_probability_low(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """XGBoost SKIP decision when probability below threshold."""
        monkeypatch.setattr(settings, "decision_hybrid_min_probability", 0.50)
        monkeypatch.setattr(settings, "decision_hybrid_min_expected_pnl", 0.0)

        model_type = "xgb_entry_v1"
        probability_win = 0.40
        expected_pnl = 50.0

        if model_type == "xgb_entry_v1":
            decision = (
                "TRADE"
                if (probability_win >= settings.decision_hybrid_min_probability
                    and expected_pnl >= settings.decision_hybrid_min_expected_pnl)
                else "SKIP"
            )
        else:
            decision = classify_prediction(
                probability_win=probability_win,
                expected_pnl=expected_pnl,
                bucket_count=0,
                pnl_std=0.0,
            )

        assert decision == "SKIP"

    def test_xgb_decision_skip_when_ev_negative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """XGBoost SKIP when expected PnL below threshold."""
        monkeypatch.setattr(settings, "decision_hybrid_min_probability", 0.50)
        monkeypatch.setattr(settings, "decision_hybrid_min_expected_pnl", 5.0)

        model_type = "xgb_entry_v1"
        probability_win = 0.80
        expected_pnl = 3.0

        if model_type == "xgb_entry_v1":
            decision = (
                "TRADE"
                if (probability_win >= settings.decision_hybrid_min_probability
                    and expected_pnl >= settings.decision_hybrid_min_expected_pnl)
                else "SKIP"
            )
        else:
            decision = classify_prediction(
                probability_win=probability_win,
                expected_pnl=expected_pnl,
                bucket_count=0,
                pnl_std=0.0,
            )

        assert decision == "SKIP"

    def test_bucket_decision_skip_despite_good_prob_ev(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bucket model SKIP when bucket_count < threshold, even with good prob/EV."""
        monkeypatch.setattr(settings, "decision_hybrid_min_probability", 0.50)
        monkeypatch.setattr(settings, "decision_hybrid_min_expected_pnl", 0.0)
        monkeypatch.setattr(settings, "decision_hybrid_min_bucket_count", 8)
        monkeypatch.setattr(settings, "decision_hybrid_max_pnl_std", 250.0)

        result = classify_prediction(
            probability_win=0.90,
            expected_pnl=100.0,
            bucket_count=2,
            pnl_std=50.0,
        )
        assert result == "SKIP"
