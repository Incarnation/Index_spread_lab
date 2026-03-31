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
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    ORDINAL_FEATURES,
    _calibration_by_decile,
    _max_drawdown,
    build_feature_matrix,
    load_model,
    predict_xgb,
    save_model,
    train_xgb_models,
    walk_forward_validate_xgb,
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

    df = pd.DataFrame({
        "day": days,
        "entry_dt": days,
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
