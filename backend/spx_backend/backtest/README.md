## Backtest Folder

This folder contains the local backtest engine code and optional sample data artifacts.

### Files

- `engine.py`: DuckDB-based backtest engine.
- `run_backtest.py`: simple local runner/entrypoint.
- `data/samples/`: sample CSV files kept for reference.

### Notes

- The backtest engine is configured for Parquet inputs (Databento downloads).
- Sample CSV files in `data/samples/` are not required by the engine.
- Update paths in `run_backtest.py` to point at your own local Parquet datasets.

