"""Split manifest for backend/scripts/xgb_model.py -> backend/scripts/xgb/."""
from __future__ import annotations

SOURCE = "backend/scripts/xgb_model.py"
PACKAGE_DIR = "backend/scripts/xgb/"

# Lines 1-31: docstring + stdlib/third-party imports + logger setup +
# sklearn/xgboost imports.  Copied verbatim into every submodule so each
# is self-contained.
HEADER_LINES = 31

# (filename, [(start, end), ...], extra_imports) -- ranges 1-based inclusive.
SUBMODULES = [
    (
        "features.py",
        [
            (37, 44),    # CONTINUOUS_FEATURES
            (46, 46),    # ORDINAL_FEATURES
            (48, 48),    # BINARY_FEATURES
            (50, 50),    # CATEGORICAL_FEATURES
            (52, 53),    # TARGET_CLS, TARGET_REG
            (95, 156),   # build_feature_matrix, get_feature_names
            (630, 631),  # HOLD_TARGET_CLS, HOLD_TARGET_REG
            (634, 687),  # _resolve_targets, build_entry_feature_matrix
            (694, 694),  # BIG_LOSS_THRESHOLD
            (696, 699),  # V2_EXTRA_FEATURES
            (702, 773),  # _add_v2_features, build_entry_v2_feature_matrix
            (1096, 1096),  # HOLD_VS_CLOSE_TARGET (defined-only, kept for surface parity)
            (1099, 1144),  # build_hold_vs_close_targets
        ],
        # No cross-package imports; features.py is the leaf layer.
        "",
    ),
    (
        "training.py",
        [
            (56, 71),     # DEFAULT_CLS_PARAMS
            (73, 88),     # DEFAULT_REG_PARAMS
            (163, 226),   # train_xgb_models
            (233, 267),   # predict_xgb
            (274, 311),   # save_model
            (314, 344),   # load_model
            (498, 591),   # _fit_full_after_early_stopping
            (594, 623),   # train_final_model
        ],
        # Cross-package: features symbols used by training functions.
        "from .features import build_feature_matrix, get_feature_names",
    ),
    (
        "walkforward.py",
        [
            (351, 385),   # _calibration_by_decile
            (388, 481),   # walk_forward_validate_xgb
            (484, 491),   # _max_drawdown
            (776, 1001),  # walk_forward_rolling
            (1004, 1089), # _extract_entry_rules
            (1147, 1309), # walk_forward_hold_vs_close
            (1312, 1385), # _extract_rules
        ],
        # Cross-package: features + training symbols used by walk-forward fns.
        (
            "from .features import (\n"
            "    BINARY_FEATURES,\n"
            "    build_entry_feature_matrix,\n"
            "    build_feature_matrix,\n"
            "    build_hold_vs_close_targets,\n"
            "    _resolve_targets,\n"
            ")\n"
            "from .training import (\n"
            "    DEFAULT_CLS_PARAMS,\n"
            "    predict_xgb,\n"
            "    train_xgb_models,\n"
            ")"
        ),
    ),
    (
        "cli.py",
        [
            (1392, 1436),  # main
            (1439, 1485),  # _run_tp50
            (1488, 1540),  # _run_hold_vs_close
            (1543, 1644),  # _run_entry
            (1647, 1746),  # _run_entry_v2
        ],
        (
            "from .features import (\n"
            "    BIG_LOSS_THRESHOLD,\n"
            "    build_entry_feature_matrix,\n"
            "    build_entry_v2_feature_matrix,\n"
            "    _resolve_targets,\n"
            ")\n"
            "from .training import (\n"
            "    DEFAULT_CLS_PARAMS,\n"
            "    save_model,\n"
            "    train_final_model,\n"
            "    _fit_full_after_early_stopping,\n"
            ")\n"
            "from .walkforward import (\n"
            "    walk_forward_hold_vs_close,\n"
            "    walk_forward_rolling,\n"
            "    walk_forward_validate_xgb,\n"
            "    _extract_entry_rules,\n"
            ")"
        ),
    ),
]

# Symbol owner per submodule (drives __init__.py re-exports).
# Includes module-level imports (Any, Path, XGBClassifier, etc.) that the
# baseline snapshot picks up so the shim's surface stays byte-identical.
SYMBOL_OWNERS = {
    "features": [
        "Any", "Path", "annotations", "logger",
        "BIG_LOSS_THRESHOLD", "BINARY_FEATURES", "CATEGORICAL_FEATURES",
        "CONTINUOUS_FEATURES", "HOLD_TARGET_CLS", "HOLD_TARGET_REG",
        "HOLD_VS_CLOSE_TARGET", "ORDINAL_FEATURES", "TARGET_CLS",
        "TARGET_REG", "V2_EXTRA_FEATURES",
        "_add_v2_features", "_resolve_targets",
        "build_entry_feature_matrix", "build_entry_v2_feature_matrix",
        "build_feature_matrix", "build_hold_vs_close_targets",
        "get_feature_names",
    ],
    "training": [
        "DEFAULT_CLS_PARAMS", "DEFAULT_REG_PARAMS",
        "XGBClassifier", "XGBRegressor",
        "_fit_full_after_early_stopping",
        "load_model", "predict_xgb", "save_model",
        "train_final_model", "train_xgb_models",
    ],
    "walkforward": [
        "brier_score_loss", "mean_absolute_error",
        "_calibration_by_decile", "_extract_entry_rules", "_extract_rules",
        "_max_drawdown",
        "walk_forward_hold_vs_close", "walk_forward_rolling",
        "walk_forward_validate_xgb",
    ],
    "cli": [
        "main",
        "_run_entry", "_run_entry_v2", "_run_hold_vs_close", "_run_tp50",
    ],
}

# All public symbols the original module exposed (must match the
# pre-split snapshot exactly).  Computed as the union of SYMBOL_OWNERS.
REEXPORTS = sorted({n for names in SYMBOL_OWNERS.values() for n in names})

# Shim CLI dispatch -- preserves ``python backend/scripts/xgb_model.py``.
CLI_DISPATCH = ("cli", "main")
