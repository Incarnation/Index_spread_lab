"""Split manifest for backend/scripts/backtest_strategy.py -> backend/scripts/backtest/.

Mechanical 4-way split:

* ``engine.py``  -- dataclasses, regime/PnL helpers, run_backtest
* ``optimizer.py`` -- grid builders, _backtest_worker, run_*_optimizer
* ``analysis.py`` -- print/pareto/comparison/walkforward/holdout
* ``cli.py``     -- _alignment_warn_against_live_settings + main

The manifest is verified end-to-end by
``python tools/monolith_split_regression.py diff`` -- all 95 public
symbols hash byte-identical pre/post split.
"""
from __future__ import annotations

SOURCE = "backend/scripts/backtest_strategy.py"
PACKAGE_DIR = "backend/scripts/backtest/"

# Lines 1-150: docstring + stdlib/third-party imports + logger setup +
# _locate_scripts_dir helper + sys.path bootstrap + module-level path
# constants + the local _SPX_2D_MAX_CALENDAR_GAP_DAYS mirror.  Copied
# verbatim into every submodule so each is self-contained; the
# self-locating ``_locate_scripts_dir`` walks up the parent chain so
# the submodule discovers ``backend/scripts/`` regardless of nesting.
HEADER_LINES = 150

# (filename, [(start, end), ...], extra_imports) -- ranges 1-based inclusive.
SUBMODULES = [
    (
        "engine.py",
        [
            (153, 167),    # _opt_val, _opt_str, _safe_bool
            (174, 369),    # @dataclass classes (PortfolioConfig, EventConfig,
                           # TradingConfig, RegimeThrottle), compute_regime_multiplier,
                           # @dataclass FullConfig
            (377, 519),    # pnl_column_name, compute_effective_pnl,
                           # _isnan, precompute_pnl_columns
            (526, 1153),   # @dataclass DayRecord, PortfolioManager,
                           # EventSignalDetector, precompute_daily_signals,
                           # @dataclass BacktestResult, _precompute_day_selections,
                           # _fast_*, _should_skip_day, run_backtest
        ],
        # No cross-submodule imports; engine is the leaf layer.
        "",
    ),
    (
        "optimizer.py",
        [
            (1161, 1239),  # _build_optimizer_grid
            (1243, 1244),  # OPTIMIZER_TP_VALUES, OPTIMIZER_SL_VALUES
            (1247, 1281),  # _build_staged_grid_stage1
            (1284, 1284),  # _StagedTradingWinner alias
            (1288, 1316),  # _build_staged_grid_stage2
            (1319, 1376),  # _build_staged_grid_stage3
            (1379, 1380),  # EVENT_ONLY_TP_VALUES, EVENT_ONLY_SL_VALUES
            (1383, 1443),  # _build_event_only_grid
            (1446, 1499),  # run_event_only_optimizer
            (1506, 1507),  # SELECTIVE_TP_VALUES, SELECTIVE_SL_VALUES
            (1510, 1564),  # _build_selective_grid
            (1567, 1633),  # run_selective_optimizer
            (1640, 1640),  # _BACKTEST_WORKER_REF
            (1643, 1656),  # _init_backtest_worker
            (1659, 1689),  # _backtest_worker
            (1692, 1709),  # _build_precomp_cache
            (1712, 1805),  # _run_grid
            (1808, 1832),  # run_optimizer
            (1835, 1952),  # run_staged_optimizer
        ],
        # Cross-package: every config dataclass + run_backtest + the
        # precompute helpers live in engine.py.  ``_opt_val`` /
        # ``_safe_bool`` are needed by ``_build_*_grid`` row->config
        # round-trips inside this module; ``_precompute_day_selections``
        # is called by ``_build_precomp_cache`` for the parallel grid.
        (
            "from .engine import (\n"
            "    BacktestResult,\n"
            "    EventConfig,\n"
            "    FullConfig,\n"
            "    PortfolioConfig,\n"
            "    RegimeThrottle,\n"
            "    TradingConfig,\n"
            "    _opt_val,\n"
            "    _precompute_day_selections,\n"
            "    _safe_bool,\n"
            "    precompute_daily_signals,\n"
            "    precompute_pnl_columns,\n"
            "    run_backtest,\n"
            ")"
        ),
    ),
    (
        "analysis.py",
        [
            (1960, 2011),  # _build_comparison_configs
            (2019, 2034),  # print_summary
            (2037, 2047),  # print_monthly
            (2050, 2066),  # print_comparison_table
            (2069, 2084),  # print_optimizer_top
            (2092, 2093),  # PARETO_CSV, WALKFORWARD_CSV
            (2096, 2120),  # ANALYSIS_PARAMS
            (2123, 2139),  # _deduplicate_results
            (2142, 2179),  # _parameter_importance
            (2182, 2197),  # extract_pareto_frontier
            (2200, 2212),  # _print_pareto
            (2215, 2271),  # _robustness_check
            (2274, 2371),  # _row_to_config
            (2374, 2414),  # run_analysis
            (2421, 2437),  # WALKFORWARD_WINDOWS
            (2440, 2512),  # generate_auto_windows
            (2515, 2536),  # walkforward_split
            (2539, 2597),  # _run_window_optimizer
            (2600, 2610),  # _config_signature
            (2613, 2638),  # _config_key
            (2641, 2804),  # run_walkforward
            (2812, 2916),  # run_holdout_evaluation
        ],
        # Cross-package: every dataclass / run_backtest / engine helper +
        # the optimizer entry points used by walkforward / holdout.
        # ``_opt_str`` / ``_opt_val`` / ``_safe_bool`` are pulled in for
        # ``_row_to_config`` (analysis.py owns the row->config inverse).
        (
            "from .engine import (\n"
            "    BacktestResult,\n"
            "    EventConfig,\n"
            "    FullConfig,\n"
            "    PortfolioConfig,\n"
            "    RegimeThrottle,\n"
            "    TradingConfig,\n"
            "    _opt_str,\n"
            "    _opt_val,\n"
            "    _safe_bool,\n"
            "    compute_effective_pnl,\n"
            "    precompute_daily_signals,\n"
            "    precompute_pnl_columns,\n"
            "    run_backtest,\n"
            ")\n"
            "from .optimizer import (\n"
            "    OPTIMIZER_SL_VALUES,\n"
            "    OPTIMIZER_TP_VALUES,\n"
            "    _build_optimizer_grid,\n"
            "    _run_grid,\n"
            "    run_event_only_optimizer,\n"
            "    run_optimizer,\n"
            "    run_selective_optimizer,\n"
            "    run_staged_optimizer,\n"
            ")"
        ),
    ),
    (
        "cli.py",
        [
            (2924, 2977),  # _alignment_warn_against_live_settings
            (2980, 3336),  # main
        ],
        # CLI ties everything together.  ``_run_grid`` (YAML-grid path)
        # and ``_robustness_check`` (analysis path) are referenced at
        # runtime even though they live in sibling submodules.
        (
            "from .engine import (\n"
            "    BacktestResult,\n"
            "    EventConfig,\n"
            "    FullConfig,\n"
            "    PortfolioConfig,\n"
            "    RegimeThrottle,\n"
            "    TradingConfig,\n"
            "    compute_effective_pnl,\n"
            "    precompute_daily_signals,\n"
            "    precompute_pnl_columns,\n"
            "    run_backtest,\n"
            ")\n"
            "from .optimizer import (\n"
            "    _run_grid,\n"
            "    run_event_only_optimizer,\n"
            "    run_optimizer,\n"
            "    run_selective_optimizer,\n"
            "    run_staged_optimizer,\n"
            ")\n"
            "from .analysis import (\n"
            "    ANALYSIS_PARAMS,\n"
            "    PARETO_CSV,\n"
            "    WALKFORWARD_CSV,\n"
            "    WALKFORWARD_WINDOWS,\n"
            "    _build_comparison_configs,\n"
            "    _robustness_check,\n"
            "    _row_to_config,\n"
            "    extract_pareto_frontier,\n"
            "    generate_auto_windows,\n"
            "    print_comparison_table,\n"
            "    print_monthly,\n"
            "    print_optimizer_top,\n"
            "    print_summary,\n"
            "    run_analysis,\n"
            "    run_holdout_evaluation,\n"
            "    run_walkforward,\n"
            "    walkforward_split,\n"
            ")"
        ),
    ),
]

# Symbol owner per submodule (drives __init__.py re-exports).
# Header-imported symbols (Any, Path, asdict, dataclass, etc.) are
# attributed to ``engine.py`` since that is the first submodule in the
# package; they exist in every submodule but each appears once in
# __init__.
SYMBOL_OWNERS = {
    "engine": [
        # Header import / module state symbols.
        "Any", "CONTRACTS", "CONTRACT_MULT", "DATA_DIR", "DEFAULT_CSV",
        "EventThresholds", "MARGIN_PER_LOT", "Path", "Pool",
        "RESULTS_CSV", "_BACKEND_DIR", "_SCRIPTS_DIR",
        "_SPX_2D_MAX_CALENDAR_GAP_DAYS",
        "_locate_scripts_dir",
        "annotations", "asdict", "candidate_dedupe_key", "compute_regime_metrics",
        "cpu_count", "dataclass", "evaluate_event_signals", "field",
        "logger", "timedelta",
        "_pareto_extract",
        # Body-defined symbols.
        "BacktestResult", "DayRecord", "EventConfig", "EventSignalDetector",
        "FullConfig", "PortfolioConfig", "PortfolioManager",
        "RegimeThrottle", "TradingConfig",
        "_fast_event_select", "_fast_sched_select", "_isnan",
        "_opt_str", "_opt_val", "_precompute_day_selections",
        "_safe_bool", "_should_skip_day",
        "compute_effective_pnl", "compute_regime_multiplier",
        "pnl_column_name", "precompute_daily_signals",
        "precompute_pnl_columns", "run_backtest",
    ],
    "optimizer": [
        "EVENT_ONLY_SL_VALUES", "EVENT_ONLY_TP_VALUES",
        "OPTIMIZER_SL_VALUES", "OPTIMIZER_TP_VALUES",
        "SELECTIVE_SL_VALUES", "SELECTIVE_TP_VALUES",
        "_BACKTEST_WORKER_REF", "_StagedTradingWinner",
        "_backtest_worker", "_build_event_only_grid",
        "_build_optimizer_grid", "_build_precomp_cache",
        "_build_selective_grid",
        "_build_staged_grid_stage1", "_build_staged_grid_stage2",
        "_build_staged_grid_stage3",
        "_init_backtest_worker", "_run_grid",
        "run_event_only_optimizer", "run_optimizer",
        "run_selective_optimizer", "run_staged_optimizer",
    ],
    "analysis": [
        "ANALYSIS_PARAMS", "PARETO_CSV", "WALKFORWARD_CSV",
        "WALKFORWARD_WINDOWS",
        "_build_comparison_configs", "_config_key", "_config_signature",
        "_deduplicate_results", "_parameter_importance",
        "_print_pareto", "_robustness_check", "_row_to_config",
        "_run_window_optimizer",
        "extract_pareto_frontier", "generate_auto_windows",
        "print_comparison_table", "print_monthly",
        "print_optimizer_top", "print_summary",
        "run_analysis", "run_holdout_evaluation",
        "run_walkforward", "walkforward_split",
    ],
    "cli": [
        "_alignment_warn_against_live_settings",
        "main",
    ],
}

# All public symbols the original module exposed (must match the
# pre-split snapshot exactly).  Computed as the union of SYMBOL_OWNERS.
REEXPORTS = sorted({n for names in SYMBOL_OWNERS.values() for n in names})

# Shim CLI dispatch -- preserves ``python backend/scripts/backtest_strategy.py``.
CLI_DISPATCH = ("cli", "main")
