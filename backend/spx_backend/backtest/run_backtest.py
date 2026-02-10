from __future__ import annotations

from spx_backend.backtest.engine import BacktestConfig, BacktestEngine


def main() -> None:
    # Update these paths to match your local Databento downloads.
    cfg = BacktestConfig(
        cbbo_parquet_glob="data/cbbo_1m/*.parquet",
        definitions_parquet_glob="data/definitions/*.parquet",
        underlying_parquet_glob="data/underlying/*.parquet",
        spread_width_points=25.0,
        spread_side="put",
    )

    engine = BacktestEngine(cfg)
    result = engine.run()

    print(f"trades: {len(result.trades)}")
    if result.summary:
        print("summary:")
        for key, value in result.summary.items():
            print(f"  {key}: {value}")
    for t in result.trades[:5]:
        print(
            f"entry={t.entry_ts} exit={t.exit_ts} exp={t.expiration} credit={t.entry_credit:.2f} "
            f"reason={t.exit_reason}"
        )


if __name__ == "__main__":
    main()

