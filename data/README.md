# `data/` directory

Operational guide to every artifact under `data/`.  This file is the
single source of truth for "what is this file, who writes it, who reads
it, and is it safe to delete?".  Keep it accurate when you add a new
exporter, cache, or pipeline output.

> **Generated/managed**: most files in this tree are produced by scripts
> under `backend/scripts/` and are *not* checked into git
> (`.gitignore` excludes everything except this README and `archive/`
> tombstones).  Treat the entire tree as reproducible from the
> production database + the offline pipeline.

## Quick reference

| Path                                     | Producer (script)                                      | Consumer(s)                                                | Format            | Safe to delete?                                            |
| ---------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------- | ----------------- | ---------------------------------------------------------- |
| `archive/`                               | manual (supersession tombstones; see `archive/README.md`) | --                                                       | mixed             | yes (historical only)                                      |
| `backtest_results.csv`                   | `scripts/backtest_strategy.py`                         | `scripts/ingest_optimizer_results.py`, manual review       | CSV               | yes (rerun backtest to regenerate)                         |
| `candidates_cache/<YYYYMMDD>.parquet`    | `scripts/generate_training_data.py` (per-day cache)    | same script (next run skips already-cached days)           | Parquet           | yes (slow rebuild — entire history will be re-ingested)    |
| `context_snapshots_export.csv`           | `scripts/export_production_data.py --tables context_snapshots` | `scripts/generate_training_data.py`, regime + GEX joins   | CSV               | yes (rerun exporter)                                       |
| `databento/`                             | `scripts/download_databento.py`                        | `scripts/generate_training_data.py` (chains, underlying)   | DBN + Parquet     | **no** without re-paying for re-download (cost!)            |
| `economic_calendar.csv`                  | `scripts/export_production_data.py --tables calendar_merge` (preferred) or `scripts/generate_economic_calendar.py` | training pipeline + entry signals    | CSV               | yes (rerun exporter)                                       |
| `economic_events_export.csv`             | `scripts/export_production_data.py --tables economic_events` (legacy) | none currently — superseded by `economic_calendar.csv`    | CSV               | yes (legacy; kept for backward compat with old tooling)    |
| `experiments/run_<ts>_<git>_<hash>/`     | `scripts/experiment_tracker.py`                        | manual review, `scripts/ingest_optimizer_results.py`       | per-run dir       | yes (immutable run records — archive first if interesting) |
| `fed_funds_rate.csv`                     | manual / one-time download (FRED)                      | training pipeline (BS pricing)                             | CSV               | **no** without re-downloading from FRED                    |
| `firstratedata/`                         | manual one-time vendor delivery                        | training pipeline (intraday SKEW + benchmarks)             | Parquet           | **no** — vendor data, not regenerable                      |
| `labels_cache/<YYYY-MM-DD>.parquet`      | `scripts/generate_training_data.py` (label step)       | same script                                                | Parquet           | yes (slow to rebuild)                                      |
| `offline_gex_cache.csv`                  | `scripts/generate_training_data.py` (GEX backfill)     | same script                                                | CSV               | yes (rerun GEX backfill)                                   |
| `optimizer_event_only_v2.csv`            | `scripts/backtest_strategy.py --grid` (event-only V2)  | `scripts/ingest_optimizer_results.py`                      | CSV               | yes (rerun grid)                                           |
| `optimizer_event_only_v2_explore.csv`    | `scripts/backtest_strategy.py --grid` (broad explore)  | `scripts/ingest_optimizer_results.py`, manual analysis     | CSV               | yes (rerun grid)                                           |
| `optimizer_selective.csv`                | `scripts/backtest_strategy.py --grid` (selective grid) | `scripts/ingest_optimizer_results.py`                      | CSV               | yes (rerun grid)                                           |
| `pareto_frontier.csv`                    | `scripts/ingest_optimizer_results.py`                  | manual review, optimizer router                            | CSV               | yes (rerun ingest)                                         |
| `pipeline_log_<ts>.json`                 | `scripts/run_pipeline.py`                              | manual review                                              | JSON              | yes (logs only)                                            |
| `production_exports/`                    | `scripts/export_production_data.py`                    | training + forensics + ML re-entry (see subtree below)     | mixed             | yes (rerun exporter)                                       |
| `regime_report.csv`                      | `scripts/regime_analysis.py`                           | manual review                                              | CSV               | yes (rerun)                                                |
| `regime_results.csv`                     | `scripts/regime_analysis.py`                           | `scripts/sl_recovery_analysis.py`, manual review           | CSV               | yes (rerun)                                                |
| `trade_candidates_export.csv`            | **deprecated** — table dropped in migration 015        | none (kept only as historical reference)                   | CSV               | yes (frozen — table no longer exists in prod)              |
| `training_candidates.csv`                | `scripts/generate_training_data.py`                    | `scripts/xgb_model.py`, `scripts/backtest_entry.py`        | CSV (gigabytes)   | yes (very slow rebuild — full history sweep)               |
| `treasury_10y2y_spread.csv`              | manual / one-time download                             | regime context features                                    | CSV               | **no** without re-downloading                              |
| `underlying_quotes_export.csv`           | `scripts/export_production_data.py --tables underlying_quotes` (legacy) | none currently — superseded by `production_exports/underlying/` | CSV     | yes (legacy)                                               |
| `walkforward_results.csv`                | `scripts/backtest_strategy.py --walkforward`           | `scripts/ingest_optimizer_results.py`                      | CSV               | yes (rerun)                                                |
| `walkforward_v2_results.csv`             | `scripts/backtest_strategy.py --walkforward` (V2)      | same                                                       | CSV               | yes (rerun)                                                |

## `production_exports/` subtree (H7)

Everything captured from the live PostgreSQL database lives here so a
single subtree contains every "from prod" artifact.  All files are
written **atomically** via a temp-then-rename pattern (L9), so a crash
mid-export never leaves a torn file for downstream consumers.

```
data/production_exports/
├── chains/                    per-day SPX/SPXW chain Parquet
│   └── YYYYMMDD.parquet
├── underlying/                per-symbol intraday Parquet
│   └── {SPX,SPY,VIX,VIX9D,VVIX,SKEW}_1min.parquet
├── trade_decisions.csv        decision_job snapshots
├── trades/
│   ├── trades.csv             one row per opened spread
│   └── trade_legs.csv         one row per leg (joined to trades on trade_id)
├── trade_marks/               per-day mark-to-market timeseries (chunked Parquet)
│   └── YYYYMMDD.parquet
├── orders/
│   ├── orders.csv             order submissions (PII-redacted, see below)
│   └── fills.csv              fill events
├── gex_snapshots/             per-day net-GEX snapshots (Parquet)
│   └── YYYYMMDD.parquet
├── gex_by_strike/             per-day per-strike GEX (chunked Parquet)
│   └── YYYYMMDD.parquet
├── gex_by_expiry_strike/      per-day per-(expiry, strike) GEX (chunked Parquet)
│   └── YYYYMMDD.parquet
├── portfolio_state.csv        portfolio_state timeseries
├── portfolio_trades.csv       portfolio_trades timeseries
├── optimizer_runs.csv         optimizer_runs metadata
├── optimizer_results.csv      optimizer_results rows (joined to runs by run_id)
└── optimizer_walkforward.csv  optimizer_walkforward rows
```

### PII / secrets policy

* `users`, `auth_audit_log`, and any `user_id` columns are intentionally
  **never** exported.
* `orders.request_json` and `orders.response_json` are JSONB blobs that
  may contain broker account IDs, API tokens, and client IPs.  They are
  filtered by `_redact_order_payload` (in
  `backend/scripts/export_production_data.py`) using an explicit
  allowlist of safe keys (`class`, `symbol`, `side`, `quantity`,
  `price`, `status`, `id`, …) — anything not on the allowlist is
  dropped *before* the value reaches the CSV.
* If you add a new export, follow the same allowlist pattern.  Never
  use a blocklist (new fields default to "redacted").

### Refresh cadence

* **Forensics tables** (`trade_decisions`, `trades`, `trade_legs`,
  `trade_marks`, `orders`, `fills`, `portfolio_*`): refresh after every
  trading day for incremental ML / audits.
* **GEX tables** (`gex_snapshots`, `gex_by_strike`,
  `gex_by_expiry_strike`): refresh as needed for archaeology; expensive
  and rarely accessed.
* **Optimizer history**: refresh after every grid run to keep the
  Pareto frontier current.

Use the `--start` / `--end` filters to do incremental exports — every
exporter merges with the existing CSV on its natural key (and dedupes)
when a date filter is provided, so you never need to re-export the
full history.

## `databento/` subtree

```
data/databento/
├── _meta/                  per-day session metadata (open/close, holidays)
├── spx/                    SPX cash session DBN files
├── spxw/                   SPXW expirations
├── spy/                    SPY (used as proxy when SPX is illiquid)
└── underlying/             SPX, VIX, VIX9D, VVIX intraday DBN
```

Path is configurable via the `DATABENTO_DIR` environment variable
(`spx_backend.config.settings.databento_dir`, default
`data/databento`).  Re-download via
`scripts/download_databento.py --start <ISO> --end <ISO>` — note that
Databento charges per pull, so prefer the cached files when possible.

## Caches you can safely blow away

If you suspect a corrupt cache or want to force a full rebuild:

```bash
# Wipe per-day candidate features cache (regenerated by generate_training_data.py)
rm -rf data/candidates_cache/

# Wipe per-day labels cache (regenerated by generate_training_data.py)
rm -rf data/labels_cache/

# Wipe GEX backfill cache (regenerated from production_exports + chains)
rm -f data/offline_gex_cache.csv

# Wipe full training CSV (regenerated by generate_training_data.py — SLOW)
rm -f data/training_candidates.csv
```

The candidate cache uses an input-data fingerprint (size + mtime of
upstream Parquets, see H3 in `OFFLINE_PIPELINE_AUDIT.md`) so it will
auto-invalidate when the underlying data changes — manual deletion is
only needed in unusual cases (corrupted Parquet, schema migration).

## Files to **never** check into git

* `.env*` (secrets)
* Anything in `databento/` (vendor data; multi-GB; not yours to share)
* Anything in `firstratedata/` (vendor data; not yours to share)
* `production_exports/orders/orders.csv` (even after PII redaction —
  treat as sensitive forensics)

`.gitignore` at the repo root already enforces this; double-check
`git status` before any commit involving this directory.

## See also

* `backend/scripts/OFFLINE_PIPELINE_AUDIT.md` — full audit (Wave 0)
  including findings IDs (C1-C5, H1-H7, M1-M12, L1-L10) referenced
  throughout this file.
* `backend/scripts/run_pipeline.py` — orchestrates the offline tasks
  end-to-end.
* `backend/scripts/export_production_data.py --help` — list all
  available `--tables` choices.
