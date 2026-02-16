from __future__ import annotations

import pytest

from spx_backend.config import settings
from spx_backend.jobs.shadow_inference_job import classify_prediction


def test_classify_prediction_trade_when_probability_and_ev_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "decision_hybrid_min_probability", 0.55)
    monkeypatch.setattr(settings, "decision_hybrid_min_expected_pnl", 5.0)
    assert classify_prediction(probability_win=0.60, expected_pnl=8.0) == "TRADE"


def test_classify_prediction_skip_when_probability_or_ev_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "decision_hybrid_min_probability", 0.55)
    monkeypatch.setattr(settings, "decision_hybrid_min_expected_pnl", 5.0)
    assert classify_prediction(probability_win=0.50, expected_pnl=10.0) == "SKIP"
    assert classify_prediction(probability_win=0.80, expected_pnl=4.9) == "SKIP"
