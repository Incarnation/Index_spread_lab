"""Tests for the XGBoost model training module (xgb_model.py).

Covers feature matrix construction, model training, prediction,
serialization roundtrip, and walk-forward validation logic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from xgb_model import (  # noqa: E402
    BIG_LOSS_THRESHOLD,
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    ORDINAL_FEATURES,
    V2_EXTRA_FEATURES,
    _add_v2_features,
    _calibration_by_decile,
    _extract_entry_rules,
    _max_drawdown,
    _resolve_targets,
    build_entry_feature_matrix,
    build_entry_v2_feature_matrix,
    build_feature_matrix,
    load_model,
    predict_xgb,
    save_model,
    train_xgb_models,
    walk_forward_rolling,
    walk_forward_validate_xgb,
)

from backtest_entry import (  # noqa: E402
    _compute_metrics,
    collect_oos_predictions,
    run_strategies,
)


def _make_synthetic_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic training CSV DataFrame with realistic structure.

    Parameters
    ----------
    n    : number of rows.
    seed : random seed for reproducibility.

    Returns
    -------
    pd.DataFrame mimicking training_candidates.csv columns.
    """
    rng = np.random.RandomState(seed)
    days = pd.date_range("2025-06-01", periods=n, freq="B").strftime("%Y-%m-%d")

    entry_dts = pd.date_range("2025-06-01", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame({
        "day": days,
        "entry_dt": entry_dts.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "expiry": days,
        "spread_side": rng.choice(["put", "call"], n),
        "dte_target": rng.choice([0, 3, 5, 7, 10], n),
        "delta_target": rng.uniform(0.05, 0.25, n),
        "entry_credit": rng.uniform(0.3, 3.0, n),
        "width_points": rng.choice([10.0, 25.0], n),
        "credit_to_width": rng.uniform(0.02, 0.20, n),
        "spot": rng.uniform(5500, 6200, n),
        "spy_price": rng.uniform(550, 620, n),
        "vix": rng.uniform(12, 35, n),
        "vix9d": rng.uniform(14, 40, n),
        "term_structure": rng.uniform(0.85, 1.25, n),
        "vvix": rng.uniform(70, 120, n),
        "skew": rng.uniform(120, 160, n),
        "short_iv": rng.uniform(0.10, 0.50, n),
        "long_iv": rng.uniform(0.10, 0.50, n),
        "short_delta": -rng.uniform(0.05, 0.30, n),
        "long_delta": -rng.uniform(0.01, 0.15, n),
        "max_loss": rng.uniform(5, 25, n),
        "offline_gex_net": rng.uniform(-2e10, 2e10, n),
        "offline_zero_gamma": rng.uniform(5500, 6200, n),
        "is_opex_day": rng.choice([True, False], n),
        "is_fomc_day": rng.choice([True, False], n, p=[0.05, 0.95]),
        "is_triple_witching": rng.choice([True, False], n, p=[0.02, 0.98]),
        "contracts": 1,
        "resolved": True,
        "hit_tp50": rng.choice([True, False], n, p=[0.7, 0.3]),
        "hit_tp100_at_expiry": rng.choice([True, False], n, p=[0.15, 0.85]),
        "realized_pnl": rng.uniform(-500, 200, n),
        "exit_reason": rng.choice(["TAKE_PROFIT_50", "EXPIRY_OR_LAST_MARK", "STOP_LOSS"], n),
        "hold_hit_tp50": rng.choice([True, False], n, p=[0.75, 0.25]),
        "hold_realized_pnl": rng.uniform(-400, 250, n),
        "hold_exit_reason": rng.choice(["TAKE_PROFIT_50", "EXPIRY_OR_LAST_MARK"], n),
    })
    return df


# ===================================================================
# Feature matrix
# ===================================================================


class TestBuildFeatureMatrix:
    """build_feature_matrix should produce correct shape and types."""

    def test_shape(self) -> None:
        df = _make_synthetic_df(50)
        X, y_tp50, y_pnl = build_feature_matrix(df)
        expected_cols = (
            len(CONTINUOUS_FEATURES)
            + len(ORDINAL_FEATURES)
            + len(BINARY_FEATURES)
            + 1  # is_put (one-hot for spread_side)
            + 3  # engineered: vix_x_delta, dte_x_credit, gex_sign
        )
        assert X.shape == (50, expected_cols)
        assert len(y_tp50) == 50
        assert len(y_pnl) == 50

    def test_target_types(self) -> None:
        df = _make_synthetic_df(30)
        _, y_tp50, y_pnl = build_feature_matrix(df)
        assert y_tp50.dtype in (np.int64, np.int32, int)
        assert y_pnl.dtype == np.float64

    def test_is_put_encoding(self) -> None:
        """spread_side='put' should map to is_put=1."""
        df = _make_synthetic_df(20)
        X, _, _ = build_feature_matrix(df)
        for i, row in df.iterrows():
            expected = 1 if row["spread_side"] == "put" else 0
            assert X.loc[i, "is_put"] == expected

    def test_engineered_features(self) -> None:
        """Interaction features should be computed correctly."""
        df = _make_synthetic_df(10)
        X, _, _ = build_feature_matrix(df)
        np.testing.assert_allclose(
            X["vix_x_delta"].values,
            X["vix"].values * X["delta_target"].values,
            rtol=1e-10,
        )

    def test_missing_columns_produce_nan(self) -> None:
        """Missing optional columns should fill with NaN, not crash."""
        df = _make_synthetic_df(10)
        df = df.drop(columns=["vvix", "skew"])
        X, _, _ = build_feature_matrix(df)
        assert X["vvix"].isna().all()
        assert X["skew"].isna().all()

    def test_gex_sign(self) -> None:
        df = _make_synthetic_df(20)
        X, _, _ = build_feature_matrix(df)
        for i, row in df.iterrows():
            expected = np.sign(row["offline_gex_net"])
            assert X.loc[i, "gex_sign"] == expected


# ===================================================================
# Training and prediction
# ===================================================================


class TestTrainAndPredict:
    """Train and predict should produce valid outputs."""

    def test_train_returns_models(self) -> None:
        df = _make_synthetic_df(100)
        X, y_tp50, y_pnl = build_feature_matrix(df)
        models = train_xgb_models(X, y_tp50, y_pnl)
        assert "classifier" in models
        assert "regressor" in models
        assert "feature_names" in models
        assert len(models["feature_names"]) == X.shape[1]

    def test_predict_returns_correct_columns(self) -> None:
        df = _make_synthetic_df(100)
        X, y_tp50, y_pnl = build_feature_matrix(df)
        models = train_xgb_models(X, y_tp50, y_pnl)
        preds = predict_xgb(models, X)
        assert "prob_tp50" in preds.columns
        assert "predicted_pnl" in preds.columns
        assert len(preds) == len(X)

    def test_probabilities_in_range(self) -> None:
        """Predicted probabilities must be in [0, 1]."""
        df = _make_synthetic_df(100)
        X, y_tp50, y_pnl = build_feature_matrix(df)
        models = train_xgb_models(X, y_tp50, y_pnl)
        preds = predict_xgb(models, X)
        assert (preds["prob_tp50"] >= 0).all()
        assert (preds["prob_tp50"] <= 1).all()

    def test_train_with_validation_set(self) -> None:
        """Early stopping with a validation set should work."""
        df = _make_synthetic_df(150)
        X, y_tp50, y_pnl = build_feature_matrix(df)
        X_train, X_val = X.iloc[:100], X.iloc[100:]
        y_tp50_t, y_tp50_v = y_tp50.iloc[:100], y_tp50.iloc[100:]
        y_pnl_t, y_pnl_v = y_pnl.iloc[:100], y_pnl.iloc[100:]
        models = train_xgb_models(
            X_train, y_tp50_t, y_pnl_t,
            X_val, y_tp50_v, y_pnl_v,
        )
        preds = predict_xgb(models, X_val)
        assert len(preds) == 50


# ===================================================================
# Serialization
# ===================================================================


class TestSerialization:
    """Model save/load should produce identical predictions."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        df = _make_synthetic_df(80)
        X, y_tp50, y_pnl = build_feature_matrix(df)
        models = train_xgb_models(X, y_tp50, y_pnl)

        save_dir = tmp_path / "xgb_test"
        save_model(models, save_dir)

        assert (save_dir / "classifier.json").exists()
        assert (save_dir / "regressor.json").exists()
        assert (save_dir / "metadata.json").exists()

        loaded = load_model(save_dir)
        assert loaded["feature_names"] == models["feature_names"]

        preds_orig = predict_xgb(models, X)
        preds_loaded = predict_xgb(loaded, X)
        np.testing.assert_allclose(
            preds_orig["prob_tp50"].values,
            preds_loaded["prob_tp50"].values,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            preds_orig["predicted_pnl"].values,
            preds_loaded["predicted_pnl"].values,
            rtol=1e-6,
        )


# ===================================================================
# Walk-forward validation
# ===================================================================


class TestWalkForward:
    """Walk-forward should split by time and return valid metrics."""

    def test_returns_metrics(self) -> None:
        df = _make_synthetic_df(300)
        result = walk_forward_validate_xgb(df)
        assert "error" not in result
        assert "xgb_metrics" in result
        assert "test_summary" in result
        assert "feature_importance" in result
        assert 0 <= result["xgb_metrics"]["brier_score_tp50"] <= 1
        assert result["xgb_metrics"]["mae_pnl"] >= 0

    def test_train_test_no_overlap(self) -> None:
        """Train days must be strictly before test days."""
        df = _make_synthetic_df(300)
        result = walk_forward_validate_xgb(df)
        train_end = result["train_days"].split(" .. ")[1]
        test_start = result["test_days"].split(" .. ")[0]
        assert train_end <= test_start

    def test_insufficient_data(self) -> None:
        df = _make_synthetic_df(20)
        result = walk_forward_validate_xgb(df)
        assert "error" in result

    def test_calibration_has_10_deciles(self) -> None:
        df = _make_synthetic_df(300)
        result = walk_forward_validate_xgb(df)
        cal = result["xgb_metrics"]["calibration_by_decile"]
        assert len(cal) == 10
        for row in cal:
            assert 0 <= row["actual_rate"] <= 1
            assert row["count"] > 0


# ===================================================================
# Utility helpers
# ===================================================================


class TestCalibrationByDecile:
    """Decile calibration should partition predictions correctly."""

    def test_basic(self) -> None:
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
        y_prob = np.linspace(0.1, 0.9, 10)
        rows = _calibration_by_decile(y_true, y_prob)
        assert len(rows) == 10
        total_count = sum(r["count"] for r in rows)
        assert total_count == 10


class TestMaxDrawdown:
    """Max drawdown computation."""

    def test_no_drawdown(self) -> None:
        assert _max_drawdown([10, 20, 30]) == 0.0

    def test_full_drawdown(self) -> None:
        dd = _max_drawdown([100, -200])
        assert dd == 200.0

    def test_empty(self) -> None:
        assert _max_drawdown([]) == 0.0

    def test_recovery(self) -> None:
        """Drawdown should capture the peak-to-trough, not net."""
        dd = _max_drawdown([100, -50, -50, 200])
        # cumulative: 100, 50, 0, 200
        # peak:       100, 100, 100, 200
        # drawdown:   0, 50, 100, 0
        assert dd == 100.0


# ===================================================================
# Entry model features
# ===================================================================


class TestEntryHourFeature:
    """entry_hour should be derived from entry_dt in Eastern Time."""

    def test_entry_hour_extracted(self) -> None:
        df = _make_synthetic_df(20)
        X, _, _ = build_feature_matrix(df)
        assert "entry_hour" in X.columns
        assert X["entry_hour"].notna().all()

    def test_entry_hour_in_entry_matrix(self) -> None:
        df = _make_synthetic_df(20)
        X, _, _ = build_entry_feature_matrix(df)
        assert "entry_hour" in X.columns


class TestResolveTargets:
    """_resolve_targets should prefer hold labels when available."""

    def test_prefers_hold_labels(self) -> None:
        df = _make_synthetic_df(10)
        cls_col, reg_col = _resolve_targets(df)
        assert cls_col == "hold_hit_tp50"
        assert reg_col == "hold_realized_pnl"

    def test_falls_back_to_original(self) -> None:
        df = _make_synthetic_df(10).drop(columns=["hold_hit_tp50", "hold_realized_pnl"])
        cls_col, reg_col = _resolve_targets(df)
        assert cls_col == "hit_tp50"
        assert reg_col == "realized_pnl"


class TestBuildEntryFeatureMatrix:
    """build_entry_feature_matrix should use hold-based labels."""

    def test_shape_matches_build_feature_matrix(self) -> None:
        df = _make_synthetic_df(50)
        X1, _, _ = build_feature_matrix(df)
        X2, _, _ = build_entry_feature_matrix(df)
        assert X1.shape == X2.shape

    def test_uses_hold_labels(self) -> None:
        df = _make_synthetic_df(50)
        _, y_cls, y_pnl = build_entry_feature_matrix(df)
        expected_cls = df["hold_hit_tp50"].astype(int)
        pd.testing.assert_series_equal(y_cls, expected_cls, check_names=False)


class TestRollingWalkForward:
    """Rolling walk-forward should produce valid windowed results."""

    @staticmethod
    def _make_dense_df(n: int = 1200, seed: int = 42) -> pd.DataFrame:
        """Generate dense data spanning ~12 months with multiple rows per day."""
        rng = np.random.RandomState(seed)
        days = pd.date_range("2025-03-01", periods=n // 4, freq="B")
        chosen = pd.to_datetime(rng.choice(days, n)).sort_values()
        df = _make_synthetic_df(n)
        df["day"] = chosen.strftime("%Y-%m-%d").values
        df["entry_dt"] = [f"{d.strftime('%Y-%m-%d')} 13:00:00+00:00" for d in chosen]
        return df

    def test_returns_windows(self) -> None:
        df = self._make_dense_df(1200)
        result = walk_forward_rolling(df, train_months=4, test_months=2, step_months=2)
        assert "error" not in result
        assert result["n_windows"] >= 1
        assert "aggregated_metrics" in result
        assert "threshold_tuning" in result
        assert "recommended_threshold" in result

    def test_threshold_tuning_populated(self) -> None:
        df = self._make_dense_df(1200)
        result = walk_forward_rolling(df, train_months=4, test_months=2, step_months=2)
        if "error" in result:
            pytest.skip("Insufficient data for walk-forward")
        thr = result["threshold_tuning"]
        assert len(thr) >= 4
        for t in thr:
            assert "threshold" in t
            assert "trades_taken" in t
            assert "win_rate" in t

    def test_insufficient_data(self) -> None:
        df = _make_synthetic_df(20)
        result = walk_forward_rolling(df, train_months=8, test_months=2, step_months=1)
        assert "error" in result


class TestExtractEntryRules:
    """Entry rule extraction should produce ENTER/SKIP recommendations."""

    def test_returns_rules(self) -> None:
        df = _make_synthetic_df(200)
        rules = _extract_entry_rules(df)
        assert len(rules) > 0
        for r in rules:
            assert r["action"] in ("ENTER", "SKIP")
            assert "tp50_rate" in r
            assert "avg_pnl" in r
            assert "n_trades" in r

    def test_rules_sorted_by_pnl(self) -> None:
        df = _make_synthetic_df(200)
        rules = _extract_entry_rules(df)
        if len(rules) >= 2:
            pnls = [r["avg_pnl"] for r in rules]
            assert pnls == sorted(pnls, reverse=True)


# ===================================================================
# Production integration: extract_xgb_features + predict_xgb_entry
# ===================================================================


class TestExtractXgbFeatures:
    """extract_xgb_features should extract continuous features from candidate_json."""

    def test_basic_extraction(self) -> None:
        from spx_backend.jobs.modeling import extract_xgb_features

        cj = {
            "vix": 18.5, "vix9d": 20.0, "term_structure": 1.08,
            "vvix": 95.0, "skew": 135.0, "delta_target": 0.10,
            "credit_to_width": 0.08, "entry_credit": 0.80,
            "width_points": 10.0, "spot": 5800.0, "spy_price": 580.0,
            "short_iv": 0.20, "long_iv": 0.18, "short_delta": -0.10,
            "long_delta": -0.05, "offline_gex_net": 1e9,
            "offline_zero_gamma": 5900.0, "max_loss": 9.2,
            "target_dte": 7, "spread_side": "put",
            "is_opex_day": False, "is_fomc_day": True, "is_triple_witching": False,
        }
        features = extract_xgb_features(cj)
        assert features["vix"] == 18.5
        assert features["is_put"] == 1
        assert features["is_fomc_day"] == 1
        assert features["gex_sign"] == 1.0
        assert features["vix_x_delta"] == pytest.approx(18.5 * 0.10)

    def test_missing_fields_return_none(self) -> None:
        from spx_backend.jobs.modeling import extract_xgb_features

        features = extract_xgb_features({})
        assert features["vix"] is None
        assert features["is_put"] == 0
        assert features["entry_hour"] is None

    def test_entry_hour_from_candidate_ts(self) -> None:
        from datetime import datetime, timezone
        from spx_backend.jobs.modeling import extract_xgb_features

        ts = datetime(2025, 6, 15, 17, 0, 0, tzinfo=timezone.utc)
        features = extract_xgb_features({}, candidate_ts=ts)
        assert features["entry_hour"] is not None
        assert isinstance(features["entry_hour"], int)

    def test_spot_fallback_to_spx_price(self) -> None:
        """When 'spot' is absent, should fall back to 'spx_price' at top level or context."""
        from spx_backend.jobs.modeling import extract_xgb_features

        cj = {"spx_price": 5850.0, "context": {"spy_price": 585.0}}
        features = extract_xgb_features(cj)
        assert features["spot"] == 5850.0

        cj2 = {"context": {"spx_price": 5900.0}}
        features2 = extract_xgb_features(cj2)
        assert features2["spot"] == 5900.0

    def test_nested_leg_iv_extraction(self) -> None:
        """IV should be extracted from legs.short.iv / legs.long.iv when flat keys are absent."""
        from spx_backend.jobs.modeling import extract_xgb_features

        cj = {
            "legs": {
                "short": {"strike": 5800, "delta": -0.12, "iv": 0.22, "bid": 2.0, "ask": 2.5},
                "long": {"strike": 5790, "delta": -0.06, "iv": 0.19, "bid": 0.8, "ask": 1.2},
            },
        }
        features = extract_xgb_features(cj)
        assert features["short_iv"] == pytest.approx(0.22)
        assert features["long_iv"] == pytest.approx(0.19)
        assert features["short_delta"] == pytest.approx(-0.12)
        assert features["long_delta"] == pytest.approx(-0.06)

    def test_flat_keys_take_priority_over_nested_legs(self) -> None:
        """Top-level flat keys should be preferred over nested leg values."""
        from spx_backend.jobs.modeling import extract_xgb_features

        cj = {
            "short_iv": 0.30, "long_iv": 0.25,
            "short_delta": -0.15, "long_delta": -0.08,
            "legs": {
                "short": {"iv": 0.22, "delta": -0.12},
                "long": {"iv": 0.19, "delta": -0.06},
            },
        }
        features = extract_xgb_features(cj)
        assert features["short_iv"] == pytest.approx(0.30)
        assert features["long_iv"] == pytest.approx(0.25)
        assert features["short_delta"] == pytest.approx(-0.15)
        assert features["long_delta"] == pytest.approx(-0.08)

    def test_zero_values_not_treated_as_missing(self) -> None:
        """A flat key with value 0.0 should not fall through to nested legs."""
        from spx_backend.jobs.modeling import extract_xgb_features

        cj = {
            "short_delta": 0.0,
            "legs": {"short": {"delta": -0.12}, "long": {}},
        }
        features = extract_xgb_features(cj)
        assert features["short_delta"] == 0.0

    def test_production_shaped_candidate_json(self) -> None:
        """Integration test with a realistic production candidate_json payload.

        Exercises context fallbacks, nested leg extraction, calendar flags,
        and CBOE context -- the full path that feature_builder_job produces.
        """
        from spx_backend.jobs.modeling import extract_xgb_features

        production_cj = {
            "schema_version": 2,
            "underlying": "SPXW",
            "snapshot_id": 42,
            "expiration": "2026-03-27",
            "target_dte": 3,
            "delta_target": 0.10,
            "spread_side": "put",
            "width_points": 10.0,
            "contracts": 1,
            "entry_credit": 0.85,
            "score": 0.92,
            "delta_diff": 0.003,
            "spot": 5820.0,
            "spy_price": 582.0,
            "spx_price": 5820.0,
            "vix": 17.2,
            "vix9d": 19.5,
            "term_structure": 1.134,
            "vvix": 88.5,
            "skew": 142.3,
            "is_opex_day": True,
            "is_fomc_day": False,
            "is_triple_witching": False,
            "spy_spx_ratio": 0.1,
            "legs": {
                "short": {
                    "symbol": "SPXW260327P05810",
                    "strike": 5810.0,
                    "delta": -0.10,
                    "iv": 0.185,
                    "bid": 1.50,
                    "ask": 2.00,
                    "mid": 1.75,
                    "qty": 1,
                    "side": "STO",
                },
                "long": {
                    "symbol": "SPXW260327P05800",
                    "strike": 5800.0,
                    "delta": -0.06,
                    "iv": 0.175,
                    "bid": 0.60,
                    "ask": 1.10,
                    "mid": 0.85,
                    "qty": 1,
                    "side": "BTO",
                },
            },
            "context": {
                "ts": "2026-03-25T14:30:00+00:00",
                "spx_price": 5820.0,
                "spy_price": 582.0,
                "vix": 17.2,
                "vix9d": 19.5,
                "term_structure": 1.134,
                "vvix": 88.5,
                "skew": 142.3,
                "gex_net": -5e9,
                "zero_gamma_level": 5850.0,
            },
            "cboe_context": {
                "offline_gex_net": 2.5e9,
                "offline_zero_gamma": 5860.0,
            },
        }

        features = extract_xgb_features(production_cj)

        assert features["spot"] == 5820.0
        assert features["spy_price"] == 582.0
        assert features["vix"] == 17.2
        assert features["vix9d"] == 19.5
        assert features["term_structure"] == pytest.approx(1.134)
        assert features["vvix"] == 88.5
        assert features["skew"] == 142.3
        assert features["delta_target"] == 0.10
        assert features["entry_credit"] == 0.85
        assert features["width_points"] == 10.0
        assert features["credit_to_width"] == pytest.approx(0.085)
        assert features["short_iv"] == pytest.approx(0.185)
        assert features["long_iv"] == pytest.approx(0.175)
        assert features["short_delta"] == pytest.approx(-0.10)
        assert features["long_delta"] == pytest.approx(-0.06)
        assert features["offline_gex_net"] == pytest.approx(2.5e9)
        assert features["offline_zero_gamma"] == 5860.0
        assert features["dte_target"] == 3
        assert features["is_put"] == 1
        assert features["is_opex_day"] == 1
        assert features["is_fomc_day"] == 0
        assert features["is_triple_witching"] == 0
        assert features["vix_x_delta"] == pytest.approx(17.2 * 0.10)
        assert features["dte_x_credit"] == pytest.approx(3 * 0.085)
        assert features["gex_sign"] == 1.0


class TestPredictXgbEntry:
    """predict_xgb_entry should score candidates using a model payload."""

    def test_returns_prediction_dict(self) -> None:
        from spx_backend.jobs.modeling import predict_xgb_entry

        df = _make_synthetic_df(100)
        X, y_cls, y_pnl = build_feature_matrix(df)
        models = train_xgb_models(X, y_cls, y_pnl)

        save_dir = Path("/tmp/test_xgb_predict_entry")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_model(models, save_dir)

        cls_json = (save_dir / "classifier.json").read_text()
        reg_json = (save_dir / "regressor.json").read_text()

        payload = {
            "model_type": "xgb_entry_v1",
            "classifier_json": cls_json,
            "regressor_json": reg_json,
            "feature_names": models["feature_names"],
        }

        features = {fn: float(X.iloc[0][fn]) if pd.notna(X.iloc[0][fn]) else None
                    for fn in models["feature_names"]}

        pred = predict_xgb_entry(payload, features)
        assert "probability_win" in pred
        assert "expected_pnl" in pred
        assert "utility_score" in pred
        assert 0 <= pred["probability_win"] <= 1
        assert pred["source"] == "xgb_entry_v1"


# ===================================================================
# V2 loss-avoidance model
# ===================================================================


class TestAddV2Features:
    """_add_v2_features should add the expected extra columns."""

    def test_adds_extra_columns(self) -> None:
        df = _make_synthetic_df(50)
        X, _, _ = build_entry_feature_matrix(df)
        n_before = X.shape[1]
        X = _add_v2_features(X, df)
        assert X.shape[1] > n_before
        for col in ["is_call", "iv_skew_ratio", "credit_to_max_loss", "gex_abs",
                     "vix_change_1d", "recent_loss_rate_5d"]:
            assert col in X.columns, f"Missing column: {col}"

    def test_is_call_complement_of_is_put(self) -> None:
        df = _make_synthetic_df(30)
        X, _, _ = build_entry_feature_matrix(df)
        X = _add_v2_features(X, df)
        assert ((X["is_put"] + X["is_call"]) == 1).all()

    def test_iv_skew_ratio_positive(self) -> None:
        df = _make_synthetic_df(30)
        X, _, _ = build_entry_feature_matrix(df)
        X = _add_v2_features(X, df)
        valid = X["iv_skew_ratio"].dropna()
        assert (valid > 0).all()


class TestBuildEntryV2FeatureMatrix:
    """V2 feature matrix should have extra features and loss target."""

    def test_shape_larger_than_v1(self) -> None:
        df = _make_synthetic_df(50)
        X1, _, _ = build_entry_feature_matrix(df)
        X2, _, _ = build_entry_v2_feature_matrix(df)
        assert X2.shape[1] > X1.shape[1]

    def test_loss_target_is_binary(self) -> None:
        df = _make_synthetic_df(100)
        _, y_loss, _ = build_entry_v2_feature_matrix(df)
        assert set(y_loss.unique()).issubset({0, 1})

    def test_loss_target_matches_threshold(self) -> None:
        df = _make_synthetic_df(100)
        _, y_loss, _ = build_entry_v2_feature_matrix(df)
        pnl = pd.to_numeric(df["hold_realized_pnl"], errors="coerce")
        expected = (pnl < BIG_LOSS_THRESHOLD).astype(int)
        pd.testing.assert_series_equal(y_loss, expected, check_names=False)

    def test_v2_trains_and_predicts(self) -> None:
        df = _make_synthetic_df(200)
        X, y_cls, y_pnl = build_entry_v2_feature_matrix(df)
        models = train_xgb_models(X, y_cls, y_pnl)
        preds = predict_xgb(models, X)
        assert 0 <= preds["prob_tp50"].min()
        assert preds["prob_tp50"].max() <= 1


class TestWalkForwardV2:
    """Rolling walk-forward with V2 features should work."""

    @staticmethod
    def _make_dense_df(n: int = 1200, seed: int = 42) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        days = pd.date_range("2025-03-01", periods=n // 4, freq="B")
        chosen = pd.to_datetime(rng.choice(days, n)).sort_values()
        df = _make_synthetic_df(n)
        df["day"] = chosen.strftime("%Y-%m-%d").values
        df["entry_dt"] = [f"{d.strftime('%Y-%m-%d')} 13:00:00+00:00" for d in chosen]
        return df

    def test_v2_walk_forward_runs(self) -> None:
        df = self._make_dense_df(1200)
        result = walk_forward_rolling(
            df,
            train_months=4, test_months=2, step_months=2,
            build_features_fn=build_entry_v2_feature_matrix,
        )
        assert "error" not in result
        assert result["n_windows"] >= 1

    def test_v2_threshold_inverted_for_loss(self) -> None:
        """With is_big_loss as target (mean < 0.5), thresholds should be inverted."""
        df = self._make_dense_df(1200)
        result = walk_forward_rolling(
            df,
            train_months=4, test_months=2, step_months=2,
            build_features_fn=build_entry_v2_feature_matrix,
        )
        if "error" in result:
            pytest.skip("Insufficient data")
        thresholds = [t["threshold"] for t in result["threshold_tuning"]]
        assert min(thresholds) < 0.30


# ===================================================================
# Backtest framework
# ===================================================================


class TestComputeMetrics:
    """_compute_metrics should produce valid strategy results."""

    def test_basic_metrics(self) -> None:
        pnls = np.array([100, 50, -200, 75, 30])
        days = np.array(["2025-06-01", "2025-06-01", "2025-06-02", "2025-06-02", "2025-06-03"])
        r = _compute_metrics(pnls, days, "test", 10)
        assert r.trades_taken == 5
        assert r.trades_skipped == 5
        assert r.total_pnl == 55.0
        assert r.win_rate == 0.8
        assert r.profit_factor > 0
        assert len(r.equity_curve) == 5

    def test_empty_produces_zeros(self) -> None:
        r = _compute_metrics(np.array([]), np.array([]), "empty", 5)
        assert r.trades_taken == 0
        assert r.total_pnl == 0

    def test_all_winners_infinite_pf(self) -> None:
        pnls = np.array([100, 200, 50])
        days = np.array(["2025-06-01"] * 3)
        r = _compute_metrics(pnls, days, "winners", 3)
        assert r.profit_factor == float("inf")


class TestRunStrategies:
    """run_strategies should produce multiple strategy results."""

    def test_produces_results(self) -> None:
        df = _make_synthetic_df(100)
        df["prob_tp50"] = np.random.RandomState(42).uniform(0, 1, 100)
        df["predicted_pnl"] = np.random.RandomState(42).uniform(-200, 200, 100)
        results = run_strategies(df)
        assert len(results) > 3
        for r in results:
            assert r.trades_taken >= 0

    def test_v2_strategies(self) -> None:
        df = _make_synthetic_df(100)
        df["prob_tp50"] = np.random.RandomState(42).uniform(0, 0.3, 100)
        df["predicted_pnl"] = np.random.RandomState(42).uniform(-200, 200, 100)
        results = run_strategies(df, is_v2=True)
        v2_names = [r.name for r in results if r.name.startswith("V2")]
        assert len(v2_names) >= 5
