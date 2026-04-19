# CSV / JSON regression scripts -- operator pre-merge gate

The three scripts in this directory verify that the recently-landed
monolith splits (`xgb_model` -> `scripts/xgb/`, `backtest_strategy` ->
`scripts/backtest/`, `generate_training_data` -> `scripts/training/`)
produce **byte-identical output** when invoked through the back-compat
shim vs the new package path.

They are **not** part of the pytest suite -- two of them require local
data that isn't shipped in the repo (`data/training_candidates.csv`,
~300 MB), and pinning XGBoost into deterministic mode adds runtime that
isn't worth paying on every CI run when the
`tools/monolith_split_regression.py` bytecode check already proves the
mechanical equivalence.

Run them as a manual gate before merging any change that touches:

- the back-compat shim (`backend/scripts/{xgb_model,backtest_strategy,
  generate_training_data}.py`),
- any submodule of the three new packages,
- or `tools/split_monolith.py` / `tools/_split_manifests/*.py`.

## Quick start

```bash
# Backtest -- equity-curve + summary parity on Q1 2025
python tools/csv_regression/backtest.py \
    --candidates data/training_candidates.csv \
    --start 2025-01-01 --end 2025-03-31

# XGB walk-forward -- selection + per-window metrics parity on 2024
python tools/csv_regression/xgb.py \
    --candidates data/training_candidates.csv \
    --start 2024-01-01 --end 2024-12-31 \
    --train-months 6 --test-months 2

# Training helpers -- BS pricing + IV recovery + outcome eval (no data needed)
python tools/csv_regression/training.py
```

Each script prints four hashes (two shim runs, two package runs) and a
PASS/FAIL line for `shim determinism`, `package determinism`, and
`shim == package`.  Exit code is `0` only when **all three** pass.

## What gets compared

| Script         | Input                                        | Hashed payload                                                                 |
|----------------|----------------------------------------------|---------------------------------------------------------------------------------|
| `backtest.py`  | pinned date slice of `training_candidates.csv` | equity-curve CSV (per-day equity, daily PnL, n_trades, lots, status) + summary metrics CSV (final_equity, total_return_pct, max_drawdown_pct, sharpe, total_trades, win_days, days_traded) |
| `xgb.py`       | pinned date slice of `training_candidates.csv` | full `walk_forward_rolling` results dict, JSON-serialised with floats rounded to 6 digits, sorted keys |
| `training.py`  | synthetic options chain (seeded RNG)         | BS price vector + BS delta vector + IV bisection vector + `_evaluate_outcome` dict, JSON-serialised with floats rounded to 9 digits |

XGBoost is forced into deterministic mode (`random_state=42`,
`n_jobs=1`, `tree_method='hist'`) by `xgb.py` so two independent runs
hash the same; without those overrides the histogram-binning thread
order can drift and the parity check would be non-decidable.

## Adding a new regression script

1. Drop the script under `tools/csv_regression/<name>.py`.
2. Reuse `_common.py` for `setup_import_paths`, `fresh_import`,
   `report_parity`, `df_to_canonical_csv_bytes`, and `require_input` so
   the operator-facing CLI feels uniform across all scripts.
3. The convention is: run the same pipeline **twice** via shim and
   **twice** via package; pass both lists to `report_parity`.  Two
   runs per path is the minimum needed to detect runtime
   non-determinism.
4. Document the script in the table above and the example
   invocation in the `Quick start` section.
