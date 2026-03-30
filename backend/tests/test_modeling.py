from __future__ import annotations

from spx_backend.jobs.modeling import (
    _classify_skew_regime,
    _classify_vvix_regime,
    build_bucket_key,
    build_bucket_key_levels,
    build_legacy_bucket_key,
    extract_candidate_features,
    predict_with_bucket_model,
    summarize_strategy_quality,
    train_bucket_model,
)


def test_train_bucket_model_and_predict_returns_prob_ev_and_utility() -> None:
    """Training/prediction should emit utility and hierarchy metadata."""
    rows = [
        {
            "features": {
                "spread_side": "put",
                "target_dte": 3,
                "delta_bucket": 0.15,
                "credit_bucket": 0.10,
                "context_regime": "support",
                "vix_regime": "normal",
                "term_structure_regime": "flat",
                "spy_spx_ratio_regime": "parity",
                "vix_delta_interaction_bucket": "normal:0.15",
                "dte_credit_interaction_bucket": "3:0.10",
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
                "vix_regime": "normal",
                "term_structure_regime": "flat",
                "spy_spx_ratio_regime": "parity",
                "vix_delta_interaction_bucket": "normal:0.15",
                "dte_credit_interaction_bucket": "3:0.10",
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
                "vix_regime": "high",
                "term_structure_regime": "backwardation",
                "spy_spx_ratio_regime": "premium",
                "vix_delta_interaction_bucket": "high:0.20",
                "dte_credit_interaction_bucket": "5:0.05",
                "margin_usage": 1500.0,
            },
            "realized_pnl": -20.0,
            "hit_tp50": False,
            "margin_usage": 1500.0,
        },
    ]
    model = train_bucket_model(
        rows=rows,
        min_bucket_size=1,
        prior_strength=2.0,
        adaptive_prior_enabled=True,
        adaptive_prior_reference_rows=200,
        adaptive_prior_min=2.0,
        adaptive_prior_max=24.0,
    )
    pred = predict_with_bucket_model(
        model_payload=model,
        features={
            "spread_side": "put",
            "target_dte": 3,
            "delta_bucket": 0.15,
            "credit_bucket": 0.10,
            "context_regime": "support",
            "vix_regime": "normal",
            "term_structure_regime": "flat",
            "spy_spx_ratio_regime": "parity",
            "vix_delta_interaction_bucket": "normal:0.15",
            "dte_credit_interaction_bucket": "3:0.10",
            "margin_usage": 1200.0,
        },
    )
    assert "bucket_hierarchy" in model
    assert model["prior_strength"] >= 2.0
    assert model["prior_strength_base"] == 2.0
    assert 0.0 <= pred["probability_win"] <= 1.0
    assert isinstance(pred["expected_pnl"], float)
    assert isinstance(pred["utility_score"], float)
    assert pred["source"] in {"full_bucket", "full_low_sample", "global_fallback"}
    assert pred["bucket_level"] in {"full", "relaxed_market", "core", "global", "legacy_full"}


def test_extract_candidate_features_and_quality_summary() -> None:
    """Feature extraction should include sparse-data interaction buckets."""
    features = extract_candidate_features(
        candidate_json={
            "spread_side": "put",
            "target_dte": 5,
            "delta_target": 0.2,
            "entry_credit": 1.5,
            "width_points": 10.0,
            "contracts": 1,
            "context_flags": ["gex_support"],
            "context": {"vix": 30.0, "term_structure": 1.12, "spy_price": 620.0, "spx_price": 6200.0},
        },
        max_loss_points=8.5,
        contract_multiplier=100,
    )
    assert features["spread_side"] == "put"
    assert features["target_dte"] == 5
    assert features["credit_to_width"] == 0.15
    assert features["context_regime"] == "support"
    assert features["vix_regime"] == "high"
    assert features["term_structure_regime"] == "backwardation"
    assert features["spy_spx_ratio_regime"] == "parity"
    assert features["cboe_regime"] == "unknown"
    assert features["cboe_wall_proximity"] == "unknown"
    assert features["vix_delta_interaction_bucket"] == "high:0.20"
    assert features["dte_credit_interaction_bucket"] == "5:0.10"
    assert features["margin_usage"] == 850.0
    bucket_key = build_bucket_key(features)
    assert "support|high|backwardation|parity|unknown|unknown" in bucket_key
    assert bucket_key.endswith("high:0.20|5:0.10")
    assert build_legacy_bucket_key(features).endswith("support|high|backwardation|parity")
    levels = build_bucket_key_levels(features)
    assert set(levels.keys()) == {"full", "relaxed_market", "core"}

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


def test_extract_candidate_features_reads_cboe_context_buckets() -> None:
    """CBOE context should map into deterministic regime/proximity buckets."""
    features = extract_candidate_features(
        candidate_json={
            "spread_side": "call",
            "target_dte": 3,
            "delta_target": 0.15,
            "entry_credit": 1.0,
            "width_points": 5.0,
            "contracts": 1,
            "context_flags": [],
            "context": {"vix": 18.0, "term_structure": 1.0, "spy_price": 600.0, "spx_price": 6000.0},
            "cboe_context": {"expiry_gex_net": 1250.0, "gamma_wall_distance_ratio": 0.002},
        },
        max_loss_points=4.0,
        contract_multiplier=100,
    )
    assert features["cboe_regime"] == "support"
    assert features["cboe_wall_proximity"] == "near"


def test_predict_with_bucket_model_uses_relaxed_hierarchy_before_global() -> None:
    """Prediction should use relaxed hierarchy when full key has no support."""
    rows = [
        {
            "features": {
                "spread_side": "put",
                "target_dte": 3,
                "delta_bucket": 0.15,
                "credit_bucket": 0.10,
                "context_regime": "support",
                "vix_regime": "normal",
                "term_structure_regime": "flat",
                "spy_spx_ratio_regime": "parity",
                "vix_delta_interaction_bucket": "normal:0.15",
                "dte_credit_interaction_bucket": "3:0.10",
                "margin_usage": 1200.0,
            },
            "realized_pnl": 30.0,
            "hit_tp50": True,
            "margin_usage": 1200.0,
        }
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
            "vix_regime": "normal",
            # Switch market regimes to make full key miss, but relaxed key match.
            "term_structure_regime": "backwardation",
            "spy_spx_ratio_regime": "premium",
            "vix_delta_interaction_bucket": "normal:0.15",
            "dte_credit_interaction_bucket": "3:0.10",
            "margin_usage": 1100.0,
        },
    )
    assert pred["bucket_level"] in {"relaxed_market", "core", "global"}
    assert pred["bucket_level"] != "full"


# ===================================================================
# SKEW regime classifier
# ===================================================================


class TestSkewRegime:
    """SKEW regime classification thresholds."""

    def test_none_returns_unknown(self) -> None:
        assert _classify_skew_regime(None) == "unknown"

    def test_low(self) -> None:
        assert _classify_skew_regime(110.0) == "low"

    def test_normal(self) -> None:
        assert _classify_skew_regime(130.0) == "normal"

    def test_elevated(self) -> None:
        assert _classify_skew_regime(155.0) == "elevated"

    def test_boundary_120(self) -> None:
        assert _classify_skew_regime(120.0) == "normal"

    def test_boundary_145(self) -> None:
        assert _classify_skew_regime(145.0) == "normal"

    def test_just_above_145(self) -> None:
        assert _classify_skew_regime(145.01) == "elevated"


# ===================================================================
# VVIX regime classifier
# ===================================================================


class TestVvixRegime:
    """VVIX regime classification thresholds."""

    def test_none_returns_unknown(self) -> None:
        assert _classify_vvix_regime(None) == "unknown"

    def test_low(self) -> None:
        assert _classify_vvix_regime(70.0) == "low"

    def test_normal(self) -> None:
        assert _classify_vvix_regime(90.0) == "normal"

    def test_elevated(self) -> None:
        assert _classify_vvix_regime(120.0) == "elevated"

    def test_boundary_80(self) -> None:
        assert _classify_vvix_regime(80.0) == "normal"

    def test_boundary_105(self) -> None:
        assert _classify_vvix_regime(105.0) == "normal"


# ===================================================================
# New features in extract_candidate_features
# ===================================================================


class TestExtractNewFeatures:
    """extract_candidate_features should handle SKEW, VVIX, and calendar flags."""

    def test_skew_and_vvix_regimes(self) -> None:
        features = extract_candidate_features(
            candidate_json={
                "spread_side": "put",
                "target_dte": 3,
                "delta_target": 0.10,
                "credit_to_width": 0.06,
                "skew": 135.0,
                "vvix": 92.0,
            },
            max_loss_points=24.0,
        )
        assert features["skew_regime"] == "normal"
        assert features["vvix_regime"] == "normal"

    def test_missing_skew_vvix_default_unknown(self) -> None:
        features = extract_candidate_features(
            candidate_json={
                "spread_side": "put",
                "target_dte": 3,
                "delta_target": 0.10,
                "credit_to_width": 0.06,
            },
            max_loss_points=24.0,
        )
        assert features["skew_regime"] == "unknown"
        assert features["vvix_regime"] == "unknown"

    def test_calendar_flags_present(self) -> None:
        features = extract_candidate_features(
            candidate_json={
                "spread_side": "put",
                "target_dte": 3,
                "is_opex_day": True,
                "is_fomc_day": False,
                "is_triple_witching": True,
            },
            max_loss_points=24.0,
        )
        assert features["is_opex_day"] is True
        assert features["is_fomc_day"] is False
        assert features["is_triple_witching"] is True

    def test_calendar_flags_default_false(self) -> None:
        features = extract_candidate_features(
            candidate_json={"spread_side": "put", "target_dte": 3},
            max_loss_points=24.0,
        )
        assert features["is_opex_day"] is False
        assert features["is_fomc_day"] is False
        assert features["is_triple_witching"] is False
