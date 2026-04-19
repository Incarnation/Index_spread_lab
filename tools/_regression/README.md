# Monolith-split bytecode regression baselines

These JSON files are the bytecode-parity snapshots that protect the
Wave-5 monolith splits documented in
[`backend/scripts/OFFLINE_PIPELINE_AUDIT.md`](../../backend/scripts/OFFLINE_PIPELINE_AUDIT.md).
Each `<module>_before.json` is the pre-split monolith snapshot; each
`<module>_after.json` is the post-split (and, where applicable,
post-fix) snapshot of the back-compat shim that re-exports from the
new package.

## Usage

```bash
# Verify the live shim still matches its post-split baseline:
cd backend && python3 ../tools/monolith_split_regression.py snapshot \
  --module xgb_model            --import-path scripts --out /tmp/xgb_now.json
cd backend && python3 ../tools/monolith_split_regression.py snapshot \
  --module backtest_strategy    --import-path scripts --out /tmp/backtest_now.json
cd backend && python3 ../tools/monolith_split_regression.py snapshot \
  --module generate_training_data --import-path scripts --out /tmp/training_now.json

# Each diff should report `byte-identical`:
python3 tools/monolith_split_regression.py diff tools/_regression/xgb_after.json      /tmp/xgb_now.json
python3 tools/monolith_split_regression.py diff tools/_regression/backtest_after.json /tmp/backtest_now.json
python3 tools/monolith_split_regression.py diff tools/_regression/training_after.json /tmp/training_now.json
```

## Expected `before` → `after` diffs

| Module                    | Symbols | `before` → `after`                                                                                                                                                                                                                                                                                                                                                                                                                  |
|---------------------------|--------:|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `xgb_model`               |      46 | byte-identical                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `generate_training_data`  |     134 | byte-identical                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `backtest_strategy`       |      95 | **1 changed function (`main`)** -- deliberate. The Wave-5 close-out commit fixed a pre-existing latent argparse bug in [`backend/scripts/backtest/cli.py:291`](../../backend/scripts/backtest/cli.py) (literal `90%+` → `90%%+` so `--help` no longer crashes with `TypeError: %d format: a real number is required`). The fix changes the `co_consts` of `main()` and therefore its bytecode hash; `backtest_after.json` was re-cut to absorb the one-character change. The `before.json` snapshot is left as the historical pre-split baseline so the diff visibly reports the deliberate change. |

If a `before → after` diff ever shows changes other than the row
above, treat it as a regression and bisect against the offending
commit. The `<module>_after.json` snapshots are intended to drift
**only** when the live code is intentionally edited and the operator
re-cuts them in the same commit (with a justification in the commit
message and an entry in this table).
