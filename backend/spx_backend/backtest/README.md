## Backtest Folder

This folder contains the local backtest engine code and optional sample data artifacts.

Current status:
- Backtest helpers are available for local research workflows.
- Live production pipeline (quotes/snapshots/gex/decision/feature/label/trainer/shadow/gates) is documented in the top-level and backend READMEs.
- Backtest orchestration is still intentionally separate from the live scheduler path.

### Files

- `engine.py`: DuckDB-based backtest engine.
- `run_backtest.py`: simple local runner/entrypoint.
- `data/samples/`: sample CSV files kept for reference.

### Notes

- The backtest engine is configured for Parquet inputs (Databento downloads).
- Sample CSV files in `data/samples/` are not required by the engine.
- Update paths in `run_backtest.py` to point at your own local Parquet datasets.
- This folder does not currently run as part of the automated live predeploy gate.

