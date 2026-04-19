from __future__ import annotations

import math

import pytest

from spx_backend.jobs.modeling import (
    _classify_skew_regime,
    _classify_vvix_regime,
    build_bucket_key,
    build_bucket_key_levels,
    build_legacy_bucket_key,
    extract_candidate_features,
    extract_xgb_features,
    predict_with_bucket_model,
    predict_xgb_entry,
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


# ===================================================================
# XGBoost entry feature extraction (V1 / V2) -- C5 regression
# ===================================================================


# Stable candidate payload reused by extract_xgb_features tests.
_SAMPLE_CANDIDATE_JSON = {
    "spread_side": "call",
    "target_dte": 3,
    "delta_target": 0.20,
    "credit_to_width": 0.30,
    "entry_credit": 3.0,
    "width_points": 10.0,
    "max_loss": 7.0,
    "vix": 18.0,
    "vix9d": 17.0,
    "term_structure": 17.0 / 18.0,
    "vvix": 90.0,
    "skew": 130.0,
    "spot": 5500.0,
    "spy_price": 549.5,
    "short_iv": 0.32,
    "long_iv": 0.28,
    "short_delta": 0.20,
    "long_delta": 0.10,
    "offline_gex_net": -1.5e8,
    "offline_zero_gamma": 5450.0,
    "is_opex_day": False,
    "is_fomc_day": True,
    "is_triple_witching": False,
    "is_cpi_day": False,
    "is_nfp_day": False,
    "entry_dt": "2025-06-01T13:30:00+00:00",
}


class TestExtractXgbFeaturesV1:
    """Default ``feature_set='v1'`` should match the V1 column set used by
    the existing entry model.  V2 extras must NOT appear so live inference
    against a V1 model picks up exactly the columns it was trained on."""

    def test_v1_includes_engineered_columns(self) -> None:
        features = extract_xgb_features(_SAMPLE_CANDIDATE_JSON)
        # Spot-check a few fields driven by the candidate payload.
        assert features["vix"] == pytest.approx(18.0)
        assert features["term_structure"] == pytest.approx(17.0 / 18.0)
        assert features["dte_target"] == 3
        assert features["is_put"] == 0
        assert features["is_fomc_day"] == 1
        # Engineered interactions live in the V1 set.
        assert features["vix_x_delta"] == pytest.approx(18.0 * 0.20)
        assert features["dte_x_credit"] == pytest.approx(3 * 0.30)
        assert features["gex_sign"] == -1.0

    def test_v1_excludes_v2_extras(self) -> None:
        features = extract_xgb_features(_SAMPLE_CANDIDATE_JSON)
        # V2 extras must NOT leak into the V1 schema.
        for key in (
            "is_call",
            "iv_skew_ratio",
            "credit_to_max_loss",
            "gex_abs",
            "vix_change_1d",
            "recent_loss_rate_5d",
        ):
            assert key not in features


class TestExtractXgbFeaturesV2:
    """``feature_set='v2'`` adds the loss-avoidance extras the V2 entry
    model expects.  Mirrors xgb_model._add_v2_features (see C5)."""

    def test_v2_adds_extras_from_payload(self) -> None:
        features = extract_xgb_features(
            _SAMPLE_CANDIDATE_JSON,
            feature_set="v2",
            vix_change_1d=0.5,
            recent_loss_rate_5d=0.20,
        )
        assert features["is_call"] == 1
        assert features["iv_skew_ratio"] == pytest.approx(0.32 / 0.28)
        assert features["credit_to_max_loss"] == pytest.approx(3.0 / 7.0)
        assert features["gex_abs"] == pytest.approx(1.5e8)
        assert features["vix_change_1d"] == pytest.approx(0.5)
        assert features["recent_loss_rate_5d"] == pytest.approx(0.20)

    def test_v2_handles_missing_history_with_none(self) -> None:
        # vix_change_1d / recent_loss_rate_5d need history; when callers
        # don't supply them the columns must still exist (set to None) so
        # the booster sees the expected schema and routes them through
        # XGBoost's missing-value default path.
        features = extract_xgb_features(_SAMPLE_CANDIDATE_JSON, feature_set="v2")
        assert "vix_change_1d" in features
        assert "recent_loss_rate_5d" in features
        assert features["vix_change_1d"] is None
        assert features["recent_loss_rate_5d"] is None

    def test_v2_iv_skew_ratio_handles_zero_long_iv(self) -> None:
        payload = {**_SAMPLE_CANDIDATE_JSON, "long_iv": 0.0}
        features = extract_xgb_features(payload, feature_set="v2")
        assert features["iv_skew_ratio"] is None

    def test_unknown_feature_set_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_xgb_features(_SAMPLE_CANDIDATE_JSON, feature_set="v3")


# ===================================================================
# predict_xgb_entry V1 / V2 dispatch -- C5 regression
# ===================================================================


def _train_dummy_xgb_payload(
    *,
    feature_names: list[str],
    classifier_label_for_hot_row: int,
    model_type: str,
) -> dict[str, object]:
    """Train a small but learnable XGBoost model and serialise it the
    same way upload_xgb_model.py would.

    A 2-row training set is too sparse for XGBoost to learn anything
    meaningful (the booster collapses to 0.5).  Instead we synthesise
    ~100 rows where the label is a clean function of the feature mean,
    so the test can assert the prediction direction reliably.

    ``classifier_label_for_hot_row`` controls the mapping from
    "high-feature" rows to label, letting the V2 inversion test trigger
    a high P(big_loss) prediction on the hot test row.
    """
    import numpy as np
    import xgboost as xgb

    rng = np.random.RandomState(0)
    n_per_class = 50
    # Hot rows: features sampled around 1.0 (noisy, so trees have signal).
    hot = rng.normal(loc=1.0, scale=0.05, size=(n_per_class, len(feature_names)))
    # Cold rows: features sampled around 0.0.
    cold = rng.normal(loc=0.0, scale=0.05, size=(n_per_class, len(feature_names)))
    X = np.vstack([hot, cold])
    y_cls = np.concatenate([
        np.full(n_per_class, classifier_label_for_hot_row, dtype=int),
        np.full(n_per_class, 1 - classifier_label_for_hot_row, dtype=int),
    ])
    y_pnl = np.concatenate([
        np.full(n_per_class, 200.0),
        np.full(n_per_class, -200.0),
    ])

    cls = xgb.XGBClassifier(
        n_estimators=20, max_depth=2,
        objective="binary:logistic", verbosity=0, random_state=42,
        eval_metric="logloss",
    )
    cls.fit(X, y_cls)
    reg = xgb.XGBRegressor(
        n_estimators=20, max_depth=2, objective="reg:squarederror",
        verbosity=0, random_state=42,
    )
    reg.fit(X, y_pnl)

    # Use a temp file because xgb.Booster.save_model writes to disk.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cls_path = f"{tmp}/cls.json"
        reg_path = f"{tmp}/reg.json"
        cls.get_booster().save_model(cls_path)
        reg.get_booster().save_model(reg_path)
        with open(cls_path) as f:
            cls_json = f.read()
        with open(reg_path) as f:
            reg_json = f.read()

    return {
        "model_type": model_type,
        "feature_names": feature_names,
        "classifier_json": cls_json,
        "regressor_json": reg_json,
    }


class TestPredictXgbEntrySemantics:
    """C5 regression: V1 returns the classifier output as
    ``probability_win``; V2 inverts (classifier predicts P(big_loss))."""

    @staticmethod
    def _hot_features(names: list[str]) -> dict[str, float]:
        return {n: 1.0 for n in names}

    def test_v1_probability_win_matches_classifier_output(self) -> None:
        feature_names = ["f1", "f2", "f3"]
        payload = _train_dummy_xgb_payload(
            feature_names=feature_names,
            classifier_label_for_hot_row=1,
            model_type="xgb_entry_v1",
        )
        out = predict_xgb_entry(payload, self._hot_features(feature_names))
        # The hot row was labelled 1, so the classifier should produce a
        # high probability and probability_win should equal that
        # (no inversion in V1).
        assert out["source"] == "xgb_entry_v1"
        assert out["probability_win"] > 0.5
        assert "p_big_loss" not in out
        assert out["utility_score"] == pytest.approx(
            out["probability_win"] * max(out["expected_pnl"], 0.0)
        )

    def test_v2_inverts_p_big_loss(self) -> None:
        feature_names = ["f1", "f2", "f3"]
        # In V2 the classifier predicts P(big_loss); the hot row is the
        # "bad" row (label 1).  predict_xgb_entry must invert it so that
        # probability_win = 1 - P(big_loss) and utility ranks the bad
        # row LOWER than a clean row.
        payload = _train_dummy_xgb_payload(
            feature_names=feature_names,
            classifier_label_for_hot_row=1,
            model_type="xgb_entry_v2",
        )
        out_hot = predict_xgb_entry(payload, self._hot_features(feature_names))
        out_cold = predict_xgb_entry(payload, {n: 0.0 for n in feature_names})

        assert out_hot["source"] == "xgb_entry_v2"
        assert "p_big_loss" in out_hot
        # Hot row was labelled "big loss" so P(big_loss) should be high.
        assert out_hot["p_big_loss"] > 0.5
        assert out_hot["probability_win"] == pytest.approx(1.0 - out_hot["p_big_loss"])
        # Cold row should look strictly safer than the hot row (lower
        # P(big_loss) -> higher probability_win).
        assert out_cold["probability_win"] > out_hot["probability_win"]

    def test_default_model_type_is_v1(self) -> None:
        # A payload with NO model_type field should dispatch to V1
        # (backward-compat with pre-stamp artifacts).
        feature_names = ["f1", "f2"]
        payload = _train_dummy_xgb_payload(
            feature_names=feature_names,
            classifier_label_for_hot_row=1,
            model_type="xgb_entry_v1",
        )
        del payload["model_type"]
        out = predict_xgb_entry(payload, {"f1": 1.0, "f2": 1.0})
        # No exception, returns V1 source label.
        assert out["source"] == "xgb_entry_v1"
        assert "p_big_loss" not in out
