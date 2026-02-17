from __future__ import annotations

import pytest

from spx_backend.config import settings
from spx_backend.jobs.shadow_inference_job import classify_prediction, classify_uncertainty_level, compute_uncertainty_penalty


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
