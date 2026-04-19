"""Split manifest for backend/scripts/generate_training_data.py -> backend/scripts/training/.

Mechanical 5-way split into a back-compat ``ProxyShim`` re-export:

* ``bs_gex_spot.py`` -- Black-Scholes math + GEX/SPX parity helpers
* ``io_loaders.py``  -- Databento / production / FRD data adapters and
                       loaders, GEX-cache CSV merge, intraday lookups
* ``candidates.py``  -- per-snapshot candidate construction, worker
                       processes, and per-day candidate cache
* ``labeling.py``    -- forward-mark labeling (single + parallel),
                       label cache, training-row builder, walk-forward
                       evaluation, deploy + relabel CSV helpers
* ``cli.py``         -- ``run_pipeline``, ``precompute_offline_gex_to_csv``,
                       ``main`` (argparse + entry-point)

The shim uses ``PROXY_SHIM = True`` so the generated
``generate_training_data.py`` is a ``_ProxyShim`` module subclass that
forwards ``setattr`` calls to every submodule.  This preserves the
existing ``monkeypatch.setattr(gtd, "SPXW_DEFS", ...)`` pattern in
``backend/tests/test_generate_training_data.py`` -- without forwarding,
those patches would only rebind the shim-level name and the consumer
function (which lives in a submodule that copied the constant from the
verbatim header) would still see the original value.

The split is verified end-to-end by
``python tools/monolith_split_regression.py diff`` -- all 131 public
symbols hash byte-identical pre/post split.
"""
from __future__ import annotations

SOURCE = "backend/scripts/generate_training_data.py"
PACKAGE_DIR = "backend/scripts/training/"

# Lines 1-209 are the verbatim header copied into every submodule:
# imports, ``logger``, the self-locating ``_locate_backend_dir`` helper,
# ``_BACKEND``, sys.path bootstrap, ``DATA_DIR``,
# ``_resolve_databento_dir`` + DATABENTO_DIR, all SPXW/SPX/SPY/FRD path
# constants, the pipeline grid constants (DTE/DELTA/SPREAD/WIDTH +
# TAKE_PROFIT_PCT etc.), ``_ACTIVE_GRID``, ``_get_grid_param`` and
# ``_assert_sl_alignment_with_live_settings``.  Header constants live in
# every submodule's globals; the proxy shim forwards monkeypatch writes
# so each submodule's copy stays consistent at runtime.  The monolith
# inserted ``_locate_backend_dir`` (19 lines) ahead of the original
# ``_BACKEND = ...`` assignment so submodules at depth 2 still resolve
# the canonical backend dir for the spx_backend / scripts sys.path bootstrap.
HEADER_LINES = 209

PROXY_SHIM = True

# (filename, [(start, end), ...], extra_imports) -- ranges 1-based inclusive.
# All line numbers are shifted +19 vs. the pre-helper monolith.
SUBMODULES = [
    (
        "bs_gex_spot.py",
        [
            (216, 288),    # _bs_d1, bs_price_vec, bs_delta_vec, implied_vol_vec
            (1010, 1011),  # GEX_MAX_DTE_DAYS, GEX_STRIKE_RANGE_PCT
            (1014, 1162),  # compute_offline_gex
            (1169, 1179),  # build_dte_lookup
            (1182, 1191),  # find_expiry_for_dte
            (1198, 1270),  # derive_spx_from_parity
            (1277, 1282),  # _time_to_expiry_years
            (1285, 1297),  # get_cbbo_snapshot_at
        ],
        # Leaf math/spot module; no cross-package imports needed.
        "",
    ),
    (
        "io_loaders.py",
        [
            (295, 312),    # _available_day_files
            (319, 326),    # _symbol_to_iid
            (329, 330),    # _production_chain_cache, _PRODUCTION_CHAIN_CACHE_MAX
            (333, 365),    # load_production_chain
            (368, 423),    # definitions_from_production
            (426, 445),    # cbbo_from_production
            (448, 474),    # statistics_from_production
            (477, 489),    # _load_dbn
            (492, 519),    # load_definitions
            (522, 571),    # build_instrument_map
            (574, 600),    # load_cbbo
            (603, 632),    # load_statistics
            (635, 663),    # load_spy_equity
            (666, 697),    # load_daily_parquet
            (700, 727),    # lag_daily_to_next_session
            (730, 776),    # load_economic_calendar
            (779, 806),    # lookup_intraday_value
            (809, 852),    # load_frd_quotes
            (855, 884),    # merge_underlying_quotes
            (887, 904),    # _load_gex_csv
            (907, 945),    # _load_merged_gex
            (948, 1007),   # lookup_gex_context
        ],
        # Pure data-IO module; no cross-package imports needed.
        "",
    ),
    (
        "candidates.py",
        [
            (1300, 1457),  # build_candidates_for_snapshot
            (2401, 2401),  # _WORKER_REF
            (2404, 2425),  # _init_candidate_worker
            (2428, 2552),  # _generate_candidates_for_day
            (2560, 2572),  # _compute_code_version
            (2575, 2596),  # _atomic_write_csv
            (2599, 2648),  # _input_data_fingerprint
            (2651, 2659),  # _load_cache_manifest
            (2662, 2668),  # _save_cache_manifest
            (2671, 2683),  # _cache_day_candidates
            (2686, 2699),  # _load_cached_day
        ],
        # Cross-package: candidate construction depends on data loaders
        # (definitions / cbbo / statistics / FRD lookups) and the BS+spot
        # helpers (delta/IV vectors + DTE lookup + SPX parity).
        (
            "from .bs_gex_spot import (\n"
            "    bs_delta_vec,\n"
            "    build_dte_lookup,\n"
            "    derive_spx_from_parity,\n"
            "    find_expiry_for_dte,\n"
            "    get_cbbo_snapshot_at,\n"
            "    implied_vol_vec,\n"
            "    _time_to_expiry_years,\n"
            ")\n"
            "from .io_loaders import (\n"
            "    _available_day_files,\n"
            "    build_instrument_map,\n"
            "    load_cbbo,\n"
            "    load_definitions,\n"
            "    load_economic_calendar,\n"
            "    load_frd_quotes,\n"
            "    load_spy_equity,\n"
            "    load_statistics,\n"
            "    lookup_gex_context,\n"
            "    lookup_intraday_value,\n"
            "    merge_underlying_quotes,\n"
            ")"
        ),
    ),
    (
        "labeling.py",
        [
            (1473, 1494),  # _downsample_marks
            (1497, 1497),  # TP_LEVELS
            (1500, 1691),  # _evaluate_outcome
            (1694, 1804),  # label_candidates
            (1808, 1808),  # label_candidates_sequential alias
            (1811, 1887),  # _extract_marks_for_day
            (1890, 2025),  # label_candidates_fast
            (2032, 2032),  # LABELS_CACHE_DIR
            (2035, 2044),  # _load_labels_manifest
            (2047, 2053),  # _save_labels_manifest
            (2056, 2062),  # _compute_label_code_hash
            (2065, 2071),  # _compute_days_hash
            (2074, 2137),  # _determine_relabel_days
            (2140, 2153),  # _load_labeled_cache_day
            (2156, 2165),  # _save_labeled_cache_day
            (2172, 2296),  # build_training_rows
            (2303, 2354),  # walk_forward_validate
            (2361, 2394),  # deploy_model
            (3305, 3313),  # TRAJECTORY_COLUMNS (uses TP_LEVELS in same module)
            (3316, 3426),  # relabel_from_csv
        ],
        # Cross-package: labeling re-loads CBBO and definitions per day
        # via the IO loaders, and uses the candidate-cache + atomic-write
        # helpers from candidates.py for relabel_from_csv.
        (
            "from .io_loaders import (\n"
            "    load_cbbo,\n"
            "    load_definitions,\n"
            "    build_instrument_map,\n"
            ")\n"
            "from .candidates import (\n"
            "    _atomic_write_csv,\n"
            ")"
        ),
    ),
    (
        "cli.py",
        [
            (2706, 3181),  # run_pipeline
            (3188, 3298),  # precompute_offline_gex_to_csv
            (3433, 3546),  # main
        ],
        # CLI orchestrates everything.  Each submodule contributes a few
        # entry points (and the ``LABELS_CACHE_DIR`` constant) used by
        # ``run_pipeline`` and ``main``.
        (
            "from .bs_gex_spot import (\n"
            "    compute_offline_gex,\n"
            "    derive_spx_from_parity,\n"
            "    get_cbbo_snapshot_at,\n"
            ")\n"
            "from .io_loaders import (\n"
            "    _available_day_files,\n"
            "    build_instrument_map,\n"
            "    load_cbbo,\n"
            "    load_definitions,\n"
            "    load_economic_calendar,\n"
            "    load_frd_quotes,\n"
            "    load_spy_equity,\n"
            "    load_statistics,\n"
            "    merge_underlying_quotes,\n"
            ")\n"
            "from .candidates import (\n"
            "    _atomic_write_csv,\n"
            "    _cache_day_candidates,\n"
            "    _compute_code_version,\n"
            "    _generate_candidates_for_day,\n"
            "    _init_candidate_worker,\n"
            "    _input_data_fingerprint,\n"
            "    _load_cache_manifest,\n"
            "    _load_cached_day,\n"
            "    _save_cache_manifest,\n"
            "    build_candidates_for_snapshot,\n"
            ")\n"
            "from .labeling import (\n"
            "    LABELS_CACHE_DIR,\n"
            "    TP_LEVELS,\n"
            "    TRAJECTORY_COLUMNS,\n"
            "    _compute_days_hash,\n"
            "    _compute_label_code_hash,\n"
            "    _determine_relabel_days,\n"
            "    _load_labeled_cache_day,\n"
            "    _load_labels_manifest,\n"
            "    _save_labeled_cache_day,\n"
            "    _save_labels_manifest,\n"
            "    build_training_rows,\n"
            "    deploy_model,\n"
            "    label_candidates,\n"
            "    label_candidates_fast,\n"
            "    relabel_from_csv,\n"
            "    walk_forward_validate,\n"
            ")"
        ),
    ),
]

# Per-submodule symbol ownership; drives __init__.py re-exports.  Header
# imports / module state attributable to ``bs_gex_spot.py`` since it's
# the first submodule in the package; they exist in every submodule but
# appear only once in ``__init__``.
SYMBOL_OWNERS = {
    "bs_gex_spot": [
        # Header imports, module state, paths, grid constants.
        "Any", "CANDIDATES_CACHE_DIR", "CONTEXT_SNAPSHOTS_CSV",
        "CONTRACTS", "CONTRACT_MULT",
        "DATABENTO_DIR", "DATA_DIR", "DECISION_MINUTES_ET",
        "DELTA_TARGETS", "DTE_TARGETS",
        "ECONOMIC_CALENDAR_CSV", "ET",
        "FRD_DIR", "FRD_SKEW", "FRD_SPX", "FRD_VIX", "FRD_VIX9D", "FRD_VVIX",
        "GEX_MAX_DTE_DAYS", "GEX_STRIKE_RANGE_PCT",
        "IV_BISECT_ITERS",
        "LABEL_MARK_INTERVAL_MINUTES",
        "MAX_IV", "MIN_IV", "MIN_MID_PRICE",
        "OFFLINE_GEX_CSV", "OUTPUT_CSV",
        "PRODUCTION_CHAINS_DIR", "PRODUCTION_UNDERLYING_DIR",
        "Path", "Pool",
        "RISK_FREE_RATE",
        "SPREAD_SIDES", "SPXW_CBBO", "SPXW_DEFS", "SPXW_STATS",
        "SPX_CBBO", "SPX_DEFS", "SPX_STATS",
        "SPY_EQUITY_PATH", "SPY_SPX_RATIO",
        "STOP_LOSS_PCT", "TAKE_PROFIT_PCT",
        "WIDTH_POINTS", "WIDTH_TARGETS",
        "ZoneInfo",
        "_ACTIVE_GRID", "_BACKEND",
        "_assert_sl_alignment_with_live_settings", "_get_grid_param",
        "_locate_backend_dir",
        "_resolve_databento_dir",
        "annotations", "date", "datetime", "defaultdict",
        "extract_candidate_features", "logger", "mid_price", "norm",
        "predict_with_bucket_model", "summarize_strategy_quality",
        "timedelta", "timezone", "train_bucket_model",
        # Body-defined symbols.
        "_bs_d1", "_time_to_expiry_years",
        "bs_delta_vec", "bs_price_vec",
        "build_dte_lookup", "compute_offline_gex",
        "derive_spx_from_parity", "find_expiry_for_dte",
        "get_cbbo_snapshot_at", "implied_vol_vec",
    ],
    "io_loaders": [
        "_PRODUCTION_CHAIN_CACHE_MAX", "_available_day_files",
        "_load_dbn", "_load_gex_csv", "_load_merged_gex",
        "_production_chain_cache", "_symbol_to_iid",
        "build_instrument_map",
        "cbbo_from_production", "definitions_from_production",
        "lag_daily_to_next_session",
        "load_cbbo", "load_daily_parquet", "load_definitions",
        "load_economic_calendar", "load_frd_quotes",
        "load_production_chain", "load_spy_equity", "load_statistics",
        "lookup_gex_context", "lookup_intraday_value",
        "merge_underlying_quotes",
        "statistics_from_production",
    ],
    "candidates": [
        "_WORKER_REF", "_atomic_write_csv",
        "_cache_day_candidates", "_compute_code_version",
        "_generate_candidates_for_day", "_init_candidate_worker",
        "_input_data_fingerprint",
        "_load_cache_manifest", "_load_cached_day",
        "_save_cache_manifest",
        "build_candidates_for_snapshot",
    ],
    "labeling": [
        "LABELS_CACHE_DIR", "TP_LEVELS", "TRAJECTORY_COLUMNS",
        "_compute_days_hash", "_compute_label_code_hash",
        "_determine_relabel_days", "_downsample_marks",
        "_evaluate_outcome", "_extract_marks_for_day",
        "_load_labeled_cache_day", "_load_labels_manifest",
        "_save_labeled_cache_day", "_save_labels_manifest",
        "build_training_rows", "deploy_model",
        "label_candidates", "label_candidates_fast",
        "label_candidates_sequential",
        "relabel_from_csv", "walk_forward_validate",
    ],
    "cli": [
        "main", "precompute_offline_gex_to_csv", "run_pipeline",
    ],
}

# Public symbols the original module exposed (must match the pre-split
# snapshot exactly).  Computed as the union of SYMBOL_OWNERS.
REEXPORTS = sorted({n for names in SYMBOL_OWNERS.values() for n in names})

# Shim CLI dispatch -- preserves ``python backend/scripts/generate_training_data.py``.
CLI_DISPATCH = ("cli", "main")
