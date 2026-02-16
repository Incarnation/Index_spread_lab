from __future__ import annotations

from spx_backend.jobs.modeling import (
    extract_candidate_features,
    predict_with_bucket_model,
    summarize_strategy_quality,
    train_bucket_model,
)


def test_train_bucket_model_and_predict_returns_prob_ev_and_utility() -> None:
    rows = [
        {
            "features": {
                "spread_side": "put",
                "target_dte": 3,
                "delta_bucket": 0.15,
                "credit_bucket": 0.10,
                "context_regime": "support",
                "margin_usage": 1200.0,
            },
            "realized_pnl": 55.0,
            "hit_tp50": True,
            "margin_usage": 1200.0,
        },
        {
            "features": {
                "spread_side": "put",
                "target_dte": 3,
                "delta_bucket": 0.15,
                "credit_bucket": 0.10,
                "context_regime": "support",
                "margin_usage": 1300.0,
            },
            "realized_pnl": 35.0,
            "hit_tp50": True,
            "margin_usage": 1300.0,
        },
        {
            "features": {
                "spread_side": "call",
                "target_dte": 5,
                "delta_bucket": 0.20,
                "credit_bucket": 0.05,
                "context_regime": "headwind",
                "margin_usage": 1500.0,
            },
            "realized_pnl": -20.0,
            "hit_tp50": False,
            "margin_usage": 1500.0,
        },
    ]
    model = train_bucket_model(rows=rows, min_bucket_size=1, prior_strength=2.0)
    pred = predict_with_bucket_model(
        model_payload=model,
        features={
            "spread_side": "put",
            "target_dte": 3,
            "delta_bucket": 0.15,
            "credit_bucket": 0.10,
            "context_regime": "support",
            "margin_usage": 1200.0,
        },
    )
    assert 0.0 <= pred["probability_win"] <= 1.0
    assert isinstance(pred["expected_pnl"], float)
    assert isinstance(pred["utility_score"], float)
    assert pred["source"] in {"bucket", "bucket_low_sample", "global_fallback"}


def test_extract_candidate_features_and_quality_summary() -> None:
    features = extract_candidate_features(
        candidate_json={
            "spread_side": "put",
            "target_dte": 5,
            "delta_target": 0.2,
            "entry_credit": 1.5,
            "width_points": 10.0,
            "contracts": 1,
            "context_flags": ["gex_support"],
        },
        max_loss_points=8.5,
        contract_multiplier=100,
    )
    assert features["spread_side"] == "put"
    assert features["target_dte"] == 5
    assert features["credit_to_width"] == 0.15
    assert features["context_regime"] == "support"
    assert features["margin_usage"] == 850.0

    summary = summarize_strategy_quality(
        realized_pnls=[40.0, -20.0, 25.0, -10.0],
        margin_usages=[900.0, 900.0, 950.0, 950.0],
        hit_tp50_count=2,
        hit_tp100_count=1,
    )
    assert summary["resolved"] == 4
    assert summary["tp50"] == 2
    assert summary["tp100_at_expiry"] == 1
    assert isinstance(summary["expectancy"], float)
    assert isinstance(summary["max_drawdown"], float)
