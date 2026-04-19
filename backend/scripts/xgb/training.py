"""XGBoost model training, prediction, and walk-forward validation.

Trains dual models on the offline training CSV produced by
``generate_training_data.py``:

* **Classifier** — ``XGBClassifier`` predicting ``P(hit_tp50)``.
* **Regressor** — ``XGBRegressor`` predicting ``realized_pnl``.

Both consume continuous features directly (VIX as a float, not a regime
bucket), letting the tree learner discover optimal splits instead of
relying on hand-coded thresholds.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
from sklearn.metrics import brier_score_loss, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

from .features import build_feature_matrix, get_feature_names


DEFAULT_CLS_PARAMS: dict[str, Any] = {
    "max_depth": 6,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}
DEFAULT_REG_PARAMS: dict[str, Any] = {
    "max_depth": 6,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "early_stopping_rounds": 30,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}
def train_xgb_models(
    X_train: pd.DataFrame,
    y_tp50_train: pd.Series,
    y_pnl_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_tp50_val: pd.Series | None = None,
    y_pnl_val: pd.Series | None = None,
    cls_params: dict | None = None,
    reg_params: dict | None = None,
) -> dict[str, Any]:
    """Train XGBClassifier (TP50) and XGBRegressor (PnL).

    Parameters
    ----------
    X_train, y_tp50_train, y_pnl_train : training data.
    X_val, y_tp50_val, y_pnl_val : optional validation set for early stopping.
    cls_params, reg_params : override default hyperparameters.

    Returns
    -------
    dict with keys:
        classifier  : fitted XGBClassifier
        regressor   : fitted XGBRegressor
        feature_names : list[str]
        cls_params  : dict of params used
        reg_params  : dict of params used
    """
    cp = {**DEFAULT_CLS_PARAMS, **(cls_params or {})}
    rp = {**DEFAULT_REG_PARAMS, **(reg_params or {})}

    es_cls = cp.pop("early_stopping_rounds", None)
    es_reg = rp.pop("early_stopping_rounds", None)

    clf = XGBClassifier(**cp)
    reg = XGBRegressor(**rp)

    fit_cls_kw: dict[str, Any] = {}
    fit_reg_kw: dict[str, Any] = {}

    if X_val is not None and y_tp50_val is not None and es_cls:
        fit_cls_kw["eval_set"] = [(X_val, y_tp50_val)]
        fit_cls_kw["verbose"] = False
    if X_val is not None and y_pnl_val is not None and es_reg:
        fit_reg_kw["eval_set"] = [(X_val, y_pnl_val)]
        fit_reg_kw["verbose"] = False

    # XGBoost >= 2.0 uses callbacks for early stopping
    if es_cls and fit_cls_kw.get("eval_set"):
        from xgboost.callback import EarlyStopping
        clf.set_params(callbacks=[EarlyStopping(rounds=es_cls, save_best=True)])
    if es_reg and fit_reg_kw.get("eval_set"):
        from xgboost.callback import EarlyStopping
        reg.set_params(callbacks=[EarlyStopping(rounds=es_reg, save_best=True)])

    clf.fit(X_train, y_tp50_train, **fit_cls_kw)
    reg.fit(X_train, y_pnl_train, **fit_reg_kw)

    return {
        "classifier": clf,
        "regressor": reg,
        "feature_names": get_feature_names(X_train),
        "cls_params": cp,
        "reg_params": rp,
    }
def predict_xgb(
    models: dict[str, Any],
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predictions from trained XGBoost models.

    Handles both sklearn wrappers (from ``train_xgb_models``) and raw
    Boosters (from ``load_model``).

    Parameters
    ----------
    models : dict returned by ``train_xgb_models`` or ``load_model``.
    X : feature matrix with same columns as training.

    Returns
    -------
    pd.DataFrame with columns ``prob_tp50`` and ``predicted_pnl``.
    """
    import xgboost as xgb

    clf = models["classifier"]
    reg = models["regressor"]

    if isinstance(clf, xgb.Booster):
        dm = xgb.DMatrix(X, feature_names=models.get("feature_names"))
        prob = clf.predict(dm)
        pnl = reg.predict(dm)
    else:
        prob = clf.predict_proba(X)[:, 1]
        pnl = reg.predict(X)

    return pd.DataFrame({
        "prob_tp50": prob,
        "predicted_pnl": pnl,
    }, index=X.index)
def save_model(
    models: dict[str, Any],
    path: Path,
    *,
    model_type: str = "xgb_entry_v1",
) -> None:
    """Save XGBoost models to a directory (two .json booster files + metadata).

    Uses the native Booster serialization to avoid sklearn wrapper issues.

    Parameters
    ----------
    models : dict
        Output of ``train_xgb_models`` / ``_fit_full_after_early_stopping``.
    path : Path
        Directory to save into (created if absent).
    model_type : str
        Model type stamp written into ``metadata.json`` so live inference can
        select the correct semantics.  Acceptable values:

        - ``"xgb_v1"``: original ``hit_tp50`` classifier (legacy main path).
        - ``"xgb_entry_v1"``: hold-based ``hit_tp50`` entry classifier.
        - ``"xgb_entry_v2"``: loss-avoidance entry model where the
          classifier predicts ``P(big_loss)`` (probability semantics are
          inverted at inference time).

    See C5 in OFFLINE_PIPELINE_AUDIT.md for why the stamp is required.
    """
    path.mkdir(parents=True, exist_ok=True)
    models["classifier"].get_booster().save_model(str(path / "classifier.json"))
    models["regressor"].get_booster().save_model(str(path / "regressor.json"))
    meta = {
        "model_type": model_type,
        "feature_names": models["feature_names"],
        "cls_params": models["cls_params"],
        "reg_params": models["reg_params"],
    }
    (path / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
def load_model(path: Path) -> dict[str, Any]:
    """Load XGBoost models from a directory saved by ``save_model``.

    Returns raw ``xgb.Booster`` objects which ``predict_xgb`` handles
    transparently alongside sklearn wrappers.

    Parameters
    ----------
    path : directory containing classifier.json, regressor.json, metadata.json.

    Returns
    -------
    dict compatible with ``predict_xgb``.
    """
    import xgboost as xgb

    meta = json.loads((path / "metadata.json").read_text())

    cls_booster = xgb.Booster()
    cls_booster.load_model(str(path / "classifier.json"))

    reg_booster = xgb.Booster()
    reg_booster.load_model(str(path / "regressor.json"))

    return {
        "classifier": cls_booster,
        "regressor": reg_booster,
        "feature_names": meta["feature_names"],
        "cls_params": meta.get("cls_params", {}),
        "reg_params": meta.get("reg_params", {}),
    }
def _fit_full_after_early_stopping(
    *,
    X_full: pd.DataFrame,
    y_cls_full: pd.Series,
    y_pnl_full: pd.Series,
    cls_params: dict | None,
    reg_params: dict | None,
) -> dict[str, Any]:
    """Train on 90% with early stopping, then refit on 100% with locked rounds.

    The two-step recipe addresses C2 in OFFLINE_PIPELINE_AUDIT.md: the
    legacy implementation trained only on the first 90% of chronology and
    used the most-recent 10% as an early-stopping holdout that was never
    re-incorporated.  The shipped booster therefore missed the most-recent
    decile -- exactly the period whose distribution is most relevant to
    live inference.

    Step 1 fits with early stopping to discover ``best_iteration``.
    Step 2 refits on the full dataset with ``n_estimators`` locked to the
    best iteration and **no** ``eval_set`` (early stopping is disabled).

    Parameters
    ----------
    X_full, y_cls_full, y_pnl_full:
        Time-sorted full training matrix and targets.
    cls_params, reg_params:
        Optional overrides forwarded to ``train_xgb_models``.

    Returns
    -------
    dict
        Same shape as ``train_xgb_models`` output, but the returned
        boosters were fit on every row in ``X_full``.  Adds
        ``best_iteration_classifier`` / ``best_iteration_regressor`` keys
        to record what early stopping chose; ``trained_rows`` records the
        number of rows seen during the final fit so callers (and tests)
        can verify no holdout was left out.
    """
    val_split = int(len(X_full) * 0.9)
    X_train = X_full.iloc[:val_split]
    y_cls_train = y_cls_full.iloc[:val_split]
    y_pnl_train = y_pnl_full.iloc[:val_split]
    X_val = X_full.iloc[val_split:]
    y_cls_val = y_cls_full.iloc[val_split:]
    y_pnl_val = y_pnl_full.iloc[val_split:]

    # Step 1: discover best_iteration via early stopping on the 90/10 split.
    es_models = train_xgb_models(
        X_train, y_cls_train, y_pnl_train,
        X_val, y_cls_val, y_pnl_val,
        cls_params=cls_params,
        reg_params=reg_params,
    )

    def _resolve_best(model: Any, default: int) -> int:
        # XGBoost surfaces best_iteration after EarlyStopping fires; on
        # tiny synthetic test data the callback may never trip and the
        # attribute is absent.  Fall back to the configured n_estimators
        # so the refit still trains on the full dataset.
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is not None and hasattr(booster, "best_iteration"):
            return int(booster.best_iteration) + 1
        if hasattr(model, "best_iteration") and model.best_iteration is not None:
            return int(model.best_iteration) + 1
        return default

    best_n_cls = _resolve_best(es_models["classifier"], DEFAULT_CLS_PARAMS["n_estimators"])
    best_n_reg = _resolve_best(es_models["regressor"], DEFAULT_REG_PARAMS["n_estimators"])

    # Step 2: refit on the full dataset with the discovered iteration count
    # and *no* early stopping (no eval_set) -- this is the conventional
    # XGBoost recipe for "use early stopping to choose hyperparams, train
    # final model on all data".
    final_cls_params = {
        **DEFAULT_CLS_PARAMS,
        **(cls_params or {}),
        "n_estimators": best_n_cls,
        "early_stopping_rounds": None,
    }
    final_reg_params = {
        **DEFAULT_REG_PARAMS,
        **(reg_params or {}),
        "n_estimators": best_n_reg,
        "early_stopping_rounds": None,
    }
    final_models = train_xgb_models(
        X_full, y_cls_full, y_pnl_full,
        cls_params=final_cls_params,
        reg_params=final_reg_params,
    )
    final_models["best_iteration_classifier"] = best_n_cls
    final_models["best_iteration_regressor"] = best_n_reg
    final_models["trained_rows"] = len(X_full)
    return final_models
def train_final_model(df: pd.DataFrame) -> dict[str, Any]:
    """Train XGBoost on the full dataset using the two-step recipe.

    Step 1: 90% / 10% chronological split, train classifier + regressor with
    early stopping to discover ``best_iteration`` for each booster.
    Step 2: refit on **all 100%** of rows with ``n_estimators`` locked to the
    discovered best iteration and no early-stopping callback.

    The 10% holdout is therefore used only for hyperparameter (round-count)
    selection; the shipped booster sees every training row.  See C2 in
    OFFLINE_PIPELINE_AUDIT.md for context on the prior 90/10-only behaviour.

    Parameters
    ----------
    df : full training CSV DataFrame.

    Returns
    -------
    dict with trained models and metadata, including ``trained_rows`` and
    the per-booster ``best_iteration_*`` values discovered in step 1.
    """
    df_sorted = df.sort_values("day").reset_index(drop=True)
    X_full, y_tp50, y_pnl = build_feature_matrix(df_sorted)
    return _fit_full_after_early_stopping(
        X_full=X_full,
        y_cls_full=y_tp50,
        y_pnl_full=y_pnl,
        cls_params=None,
        reg_params=None,
    )
