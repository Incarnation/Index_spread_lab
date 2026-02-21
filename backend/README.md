# IndexSpreadLab Backend

This backend is a FastAPI service that captures options market data, computes context/GEX, and runs a rules-first plus model-assisted paper execution pipeline.

It is built for observability and reproducibility:
- every chain snapshot is stored raw + normalized
- market clock states are audited
- decision runs are persisted (TRADE and SKIP) with reasons
- preflight endpoint provides one-call pipeline health

---

## 1) Service Responsibilities

The backend performs ten continuous tasks:
- Quote ingestion (`underlying_quotes`, `context_snapshots`)
- Option chain snapshots (`chain_snapshots`, `option_chain_rows`)
- GEX computation (`gex_snapshots`, strike/expiry detail tables)
- Feature generation (`feature_snapshots`, `trade_candidates`)
- Decision generation (`trade_decisions`, rules/hybrid policy)
- Trade mark-to-market and exits (`trades`, `trade_legs`, `trade_marks`)
- Label resolution (`trade_candidates.label_*`, `realized_pnl`)
- Weekly training (`training_runs`, `model_versions`)
- Shadow inference (`model_predictions`)
- Promotion gate evaluation (model rollout status updates)

And exposes APIs for:
- dashboard data reads
- admin run controls
- quick health and diagnostics

---

## 2) Runtime Lifecycle

Startup flow:
1) Load settings from env (`spx_backend/config.py`).
2) Initialize database schema from `spx_backend/database/sql/db_schema.sql`.
3) Build Tradier client + market clock cache.
4) Start APScheduler jobs for quote/SPX snapshot/(enabled-by-default) SPY snapshot/(optional) VIX snapshot/gex/feature-builder/decision/trade-pnl/labeler/trainer/shadow-inference/promotion-gates.
5) Optionally run immediate first cycles to warm data.

Shutdown flow:
- Scheduler stops with FastAPI lifespan shutdown.

Entrypoint:
- `python -m spx_backend.main`
- Honors Railway `PORT` automatically.

---

## 3) Core Modules

- `spx_backend/web/app.py`
  - FastAPI app and scheduler wiring.
- `spx_backend/web/routers/`
  - `public.py`: public dashboard/data endpoints.
  - `admin.py`: admin run/delete/preflight endpoints.
- `spx_backend/database/`
  - `connection.py`: async engine/session dependency.
  - `schema.py`: schema init/reset helpers.
  - `sql/`: schema and reset SQL files.
  - `reset_ml_schema.py`, `reset_all_schema.py`: DB reset CLIs.
- `spx_backend/jobs/snapshot_job.py`
  - Pulls expirations/chains, applies DTE policy, stores snapshots + rows.
- `spx_backend/jobs/quote_job.py`
  - Pulls quote symbols and updates context snapshot.
- `spx_backend/jobs/gex_job.py`
  - Computes GEX summary and curves.
- `spx_backend/jobs/decision_job.py`
  - Builds/scoring candidate spreads and writes TRADE/SKIP rows.
- `spx_backend/market_clock.py`
  - Tradier clock cache with DB audit rows and fallback behavior.
- `spx_backend/backtest/`
  - Local backtest engine code (`engine.py`, `run_backtest.py`) and `data/samples/`.
- `spx_backend/dte.py`
  - Trading-session DTE lookup and expiration chooser helpers.
- `spx_backend/database/sql/db_schema.sql`
  - Complete schema bootstrap for ingestion, decision, ML scaffolding.

---

## 4) Data Flow In Detail

### 4.1 Quote Job

Input:
- Tradier quotes for `QUOTE_SYMBOLS` (default `SPX,VIX,VIX9D,SPY`).

Output:
- One row per symbol in `underlying_quotes`.
- One upserted row in `context_snapshots` at current timestamp:
  - `spx_price`, `spy_price`, `vix`, `vix9d`
  - `term_structure = vix9d / vix` when available
  - `notes_json` with source metadata

RTH behavior:
- If outside regular hours and `ALLOW_QUOTES_OUTSIDE_RTH=false`, job skips unless forced.
- Market-open checks prefer cached Tradier clock; fallback to simple weekday/time logic.

### 4.2 Snapshot Job

Input:
- Tradier expirations + option chain payloads.

Selection:
- DTE mode:
  - `range`: capture expirations between min/max DTE.
  - `targets`: capture nearest expirations for specific targets.
- DTE semantics are trading-session based, not calendar difference.
- Strikes are trimmed around spot (`SNAPSHOT_STRIKES_EACH_SIDE`).

Output:
- `chain_snapshots` raw payload with checksum.
- `option_chain_rows` normalized rows (bid/ask/greeks/open interest/etc).

Dual-stream behavior:
- SPX snapshot stream remains the trading/decision source.
- SPY snapshot stream (enabled by default) writes the same tables with `underlying='SPY'` for future SPY model fitting.
- Optional VIX snapshot stream writes the same tables with `underlying='VIX'` for model features/training context.
- Decision execution continues to use SPX snapshots only.

Fallback mode:
- Optional `SNAPSHOT_RANGE_FALLBACK_ENABLED=true` can capture nearest expirations if strict range has none (useful in sandbox; usually off in production).

### 4.3 GEX Job

Input:
- Recent chain snapshots + option rows + recent spot quote.

Formula (per option row):
- `gex = sign * gamma * open_interest * contract_multiplier * spot^2`
- sign is negative for puts when `GEX_PUTS_NEGATIVE=true`.

Output tables:
- `gex_snapshots`: aggregate totals and `zero_gamma_level`.
- `gex_by_strike`: net/call/put GEX by strike.
- `gex_by_expiry_strike`: same by expiration+strike with `dte_days`.

Consistency behavior:
- `gex_by_expiry_strike` upserts update values on conflict, so recalculations self-heal stale DTE labels.

### 4.4 Decision Job

Input:
- Latest snapshots filtered by target DTE and freshness.
- Latest context fields (VIX/GEX/zero-gamma when available).

Core process:
1) For each target DTE, choose freshest eligible snapshot.
2) Build candidate vertical spread(s) for each enabled side (`put`/`call`) using delta and configured width.
3) Compute candidate credit and validate viability.
4) Apply context score adjustments.
5) Enforce guardrails (`DECISION_MAX_TRADES_PER_DAY`, `DECISION_MAX_OPEN_TRADES`, per-side daily/open caps).
6) Apply execution policy:
   - default: rules-ranked selection (`decision_source='rules'`)
   - optional hybrid: model-ranked selection after rules safety checks (`decision_source='hybrid_model'`)
7) Insert one `trade_decisions` row with final action and create paper trade rows for TRADE decisions.

Why SKIP rows are stored:
- They provide full auditability for why no trade occurred.
- They are required to evaluate decision quality over time.

### 4.5 Feature Builder Job

Input:
- Same live decision-time context (fresh snapshot + option rows + spot + context).

Output:
- One `feature_snapshots` row per target DTE.
- Ranked candidate rows in `trade_candidates` with deterministic `candidate_hash`.

### 4.6 Labeler Job

Input:
- Pending candidates from `trade_candidates`.
- Forward option marks from later snapshots for the same expiration.

Output:
- Resolved labels in `trade_candidates.label_json`.
- Status updates in `label_status`.
- Scalar outcomes in `realized_pnl`, `hit_tp50_before_sl_or_expiry`, and `hit_tp100_at_expiry`.
- Separate expiry counterfactual fields in label payload (`expiry_pnl`, `expiry_exit_cost`, `expiry_ts_utc`).

### 4.7 Trade PnL Job

Input:
- Open rows in `trades` and `trade_legs`.
- Latest chain marks for both legs from `option_chain_rows`.

Output:
- Rolling mark-to-market updates in `trades.current_pnl` every interval.
- Mark history in `trade_marks`.
- Auto-close updates (`status`, `exit_time`, `realized_pnl`, `exit_reason`) when TP/SL/expiry rules are met.

### 4.8 Trainer Job

Input:
- Resolved `trade_candidates` over configured lookback/test windows.

Output:
- `training_runs` lifecycle rows (RUNNING/COMPLETED/FAILED).
- `model_versions` rows with serialized model payload and walk-forward metrics.

### 4.9 Shadow Inference Job

Input:
- Most recent eligible model (`shadow`/`canary`/`active`) and recent candidates without predictions.

Output:
- `model_predictions` inserts/updates with `probability_win`, `expected_value`, and utility metadata.

### 4.10 Promotion Gate Job

Input:
- Most recent completed training run for configured model name.

Output:
- Gate pass/fail evaluation persisted into run/model metrics JSON.
- `model_versions.rollout_status` updates (`shadow`/`canary`) and optional activation behavior.

---

## 5) Trading-Day DTE Semantics

The backend maps DTE using trading sessions from available expirations, not raw date subtraction.

Implications:
- Holidays/weekends do not count as trading days.
- A calendar +3 day expiration can still be 2DTE or 3DTE depending on closures.
- Snapshot selection and decision selection both use this logic.

Helpers:
- `trading_dte_lookup(...)`
- `choose_expiration_for_trading_dte(...)`
- `closest_expiration_for_trading_dte(...)`

---

## 6) API Reference

### Public Endpoints

- `GET /health`
  - liveness probe
- `GET /api/chain-snapshots?limit=...`
  - recent chain snapshot metadata
- `GET /api/trade-decisions?limit=...`
  - recent decisions (TRADE/SKIP)
- `GET /api/gex/snapshots?limit=...`
  - recent GEX snapshot batches
- `GET /api/gex/dtes?snapshot_id=...`
  - available DTE options for selected GEX capture batch
- `GET /api/gex/expirations?snapshot_id=...`
  - available expiration dates for selected capture batch
- `GET /api/gex/curve?snapshot_id=...&dte_days=...`
- `GET /api/gex/curve?snapshot_id=...&expirations_csv=YYYY-MM-DD,YYYY-MM-DD`
  - strike curve for all, one DTE, or custom expiration set
- `GET /api/trades?status=...&limit=...`
  - recent/open/closed paper trade rows with legs and PnL fields
- `GET /api/label-metrics?lookback_days=...`
  - TP50/TP100 and realized PnL summary from resolved labels
- `GET /api/strategy-metrics?lookback_days=...`
  - strategy quality/risk metrics (win rates, expectancy, drawdown, tail-loss proxy, margin usage, side breakdown)
- `GET /api/model-ops`
  - model/training/prediction operational summary

Batch-scoped GEX note:
- DTE/expiration/curve endpoints use all snapshots in the same capture batch (`same ts + underlying + source`), not only one row.

### Admin Endpoints

All admin endpoints require a valid authenticated user token.

Routes:
- `POST /api/admin/run-snapshot`
- `POST /api/admin/run-quotes`
- `POST /api/admin/run-gex`
- `POST /api/admin/run-cboe-gex`
- `POST /api/admin/run-decision`
- `POST /api/admin/run-feature-builder`
- `POST /api/admin/run-labeler`
- `POST /api/admin/run-trade-pnl`
- `POST /api/admin/run-trainer`
- `POST /api/admin/run-shadow-inference`
- `POST /api/admin/run-promotion-gates`
- `DELETE /api/admin/trade-decisions/{decision_id}`
- `GET /api/admin/expirations?symbol=SPX`
- `GET /api/admin/preflight`

Preflight includes:
- counts of key tables
- latest timestamps for quote/snapshot/gex/decision/clock
- latest detail object for snapshot/gex/decision
- latest quote by symbol
- warning tags (for missing data domains)

---

## 7) Configuration Reference

Read from `.env` in repo root.

Required:
- `DATABASE_URL` (`postgresql+asyncpg://...`)
- `TRADIER_ACCESS_TOKEN`
- `TRADIER_ACCOUNT_ID`

High-impact settings:
- Snapshot:
  - `SNAPSHOT_INTERVAL_MINUTES`
  - `SNAPSHOT_UNDERLYING`
  - `SNAPSHOT_DTE_MODE`
  - `SNAPSHOT_DTE_MIN_DAYS`, `SNAPSHOT_DTE_MAX_DAYS`
  - `SNAPSHOT_DTE_TARGETS`, `SNAPSHOT_DTE_TOLERANCE_DAYS`
  - `SNAPSHOT_STRIKES_EACH_SIDE`
  - `ALLOW_SNAPSHOT_OUTSIDE_RTH`
  - Optional SPY snapshot stream (enabled by default):
    - `SPY_SNAPSHOT_ENABLED`
    - `SPY_SNAPSHOT_INTERVAL_MINUTES`
    - `SPY_SNAPSHOT_UNDERLYING` (default `SPY`)
    - `SPY_SNAPSHOT_DTE_MODE`
    - `SPY_SNAPSHOT_DTE_MIN_DAYS`, `SPY_SNAPSHOT_DTE_MAX_DAYS`
    - `SPY_SNAPSHOT_DTE_TARGETS`, `SPY_SNAPSHOT_DTE_TOLERANCE_DAYS`
    - `SPY_SNAPSHOT_STRIKES_EACH_SIDE`
    - `SPY_ALLOW_SNAPSHOT_OUTSIDE_RTH`
  - Optional VIX snapshot stream:
    - `VIX_SNAPSHOT_ENABLED`
    - `VIX_SNAPSHOT_INTERVAL_MINUTES`
    - `VIX_SNAPSHOT_UNDERLYING` (default `VIX`)
    - `VIX_SNAPSHOT_DTE_MODE`
    - `VIX_SNAPSHOT_DTE_MIN_DAYS`, `VIX_SNAPSHOT_DTE_MAX_DAYS`
    - `VIX_SNAPSHOT_DTE_TARGETS`, `VIX_SNAPSHOT_DTE_TOLERANCE_DAYS`
    - `VIX_SNAPSHOT_STRIKES_EACH_SIDE`
    - `VIX_ALLOW_SNAPSHOT_OUTSIDE_RTH`
- Decision:
  - `DECISION_ENTRY_TIMES`
  - `DECISION_DTE_TARGETS`, `DECISION_DTE_TOLERANCE_DAYS`
  - `DECISION_DELTA_TARGETS`
  - `DECISION_SPREAD_SIDE`, `DECISION_SPREAD_SIDES`, `DECISION_SPREAD_WIDTH_POINTS`
  - `DECISION_SNAPSHOT_MAX_AGE_MINUTES`
  - `DECISION_MAX_TRADES_PER_DAY`, `DECISION_MAX_OPEN_TRADES`
  - `DECISION_MAX_TRADES_PER_SIDE_PER_DAY`, `DECISION_MAX_OPEN_TRADES_PER_SIDE`
- Hybrid execution policy:
  - `DECISION_HYBRID_ENABLED`
  - `DECISION_HYBRID_MODEL_NAME`
  - `DECISION_HYBRID_MIN_PROBABILITY`
  - `DECISION_HYBRID_MIN_EXPECTED_PNL`
  - `DECISION_HYBRID_REQUIRE_ACTIVE_MODEL`
- Feature Builder:
  - `FEATURE_BUILDER_ENABLED`
  - `FEATURE_BUILDER_ALLOW_OUTSIDE_RTH`
  - `FEATURE_SCHEMA_VERSION`
  - `CANDIDATE_SCHEMA_VERSION`
- Labeler:
  - `LABELER_ENABLED`
  - `LABELER_BATCH_LIMIT`
  - `LABELER_MIN_AGE_MINUTES`
  - `LABELER_MAX_WAIT_HOURS`
  - `LABELER_TAKE_PROFIT_PCT`
  - `LABEL_SCHEMA_VERSION`
  - `LABEL_CONTRACT_MULTIPLIER`
  - Schedule: daily after close (Mon-Fri) at 16:15 ET
- Weekly trainer:
  - `TRAINER_ENABLED`
  - `TRAINER_WEEKDAY`, `TRAINER_HOUR`, `TRAINER_MINUTE`
  - `TRAINER_MODEL_NAME`
  - `TRAINER_LOOKBACK_DAYS`, `TRAINER_TEST_DAYS`
  - `TRAINER_MIN_ROWS`, `TRAINER_MIN_TRAIN_ROWS`, `TRAINER_MIN_TEST_ROWS`
- Shadow inference:
  - `SHADOW_INFERENCE_ENABLED`
  - `SHADOW_INFERENCE_BATCH_LIMIT`
  - `SHADOW_INFERENCE_LOOKBACK_MINUTES`
  - `SHADOW_INFERENCE_MODEL_NAME`
  - Schedule: daily after close (Mon-Fri) at 16:20 ET
- Promotion gates:
  - `PROMOTION_GATE_ENABLED`
  - `PROMOTION_GATE_MODEL_NAME`
  - `PROMOTION_GATE_MIN_RESOLVED`
  - `PROMOTION_GATE_MIN_TP50_RATE`
  - `PROMOTION_GATE_MIN_EXPECTANCY`
  - `PROMOTION_GATE_MAX_DRAWDOWN`
  - `PROMOTION_GATE_MIN_TAIL_LOSS_PROXY`
  - `PROMOTION_GATE_MAX_AVG_MARGIN_USAGE`
  - `PROMOTION_GATE_AUTO_ACTIVATE`
  - Schedule: weekly after trainer (trainer cron + 60 minutes)
- Trade PnL:
  - `TRADE_PNL_ENABLED`
  - `TRADE_PNL_INTERVAL_MINUTES`
  - `TRADE_PNL_ALLOW_OUTSIDE_RTH`
  - `TRADE_PNL_MARK_MAX_AGE_MINUTES`
  - `TRADE_PNL_TAKE_PROFIT_PCT`
  - `TRADE_PNL_STOP_LOSS_PCT`
  - `TRADE_PNL_CONTRACT_MULTIPLIER`
- Performance analytics:
  - `PERFORMANCE_ANALYTICS_ENABLED`
  - `PERFORMANCE_ANALYTICS_INTERVAL_MINUTES`
- GEX:
  - `GEX_ENABLED`, `GEX_INTERVAL_MINUTES`
  - `GEX_SNAPSHOT_BATCH_LIMIT` (default `20`)
  - `GEX_STRIKE_LIMIT` (default `150`)
  - `GEX_MAX_DTE_DAYS` (default `10`)
  - `GEX_SPOT_MAX_AGE_SECONDS`
- CBOE precomputed GEX (parallel stream):
  - `MZDATA_BASE_URL`
  - `CBOE_GEX_ENABLED`
  - `CBOE_GEX_UNDERLYING` (default `SPX`)
  - `CBOE_GEX_INTERVAL_MINUTES` (default `15`)
  - `CBOE_GEX_ALLOW_OUTSIDE_RTH` (default `false`)
- Ops/Safety:
  - `ALLOW_QUOTES_OUTSIDE_RTH`
  - `MARKET_CLOCK_CACHE_SECONDS`
  - `CORS_ORIGINS`
  - `TZ`

Recommended SPY model-fitting profile:
- `SPY_SNAPSHOT_ENABLED=true`
- `SPY_SNAPSHOT_INTERVAL_MINUTES=10`
- `SPY_SNAPSHOT_DTE_MODE=range`
- `SPY_SNAPSHOT_DTE_MIN_DAYS=0`
- `SPY_SNAPSHOT_DTE_MAX_DAYS=10`
- `SPY_SNAPSHOT_STRIKES_EACH_SIDE=75`
- Keep `SPY_ALLOW_SNAPSHOT_OUTSIDE_RTH=false` in production.

Recommended VIX model-fitting profile:
- `VIX_SNAPSHOT_ENABLED=true`
- `VIX_SNAPSHOT_INTERVAL_MINUTES=10`
- `VIX_SNAPSHOT_DTE_MODE=range`
- `VIX_SNAPSHOT_DTE_MIN_DAYS=7`
- `VIX_SNAPSHOT_DTE_MAX_DAYS=45`
- `VIX_SNAPSHOT_STRIKES_EACH_SIDE=50`
- Keep `VIX_ALLOW_SNAPSHOT_OUTSIDE_RTH=false` unless running controlled backfill/sandbox cycles.

---

## 8) Database Schema Guide

Schema file:
- `spx_backend/database/sql/db_schema.sql`

Logical groups:

Ingestion:
- `option_instruments`
- `chain_snapshots`
- `option_chain_rows`
- `underlying_quotes`
- `context_snapshots`
- `market_clock_audit`

GEX analytics:
- `gex_snapshots`
- `gex_by_strike`
- `gex_by_expiry_strike`

Decision/trading:
- `trade_decisions`
- `orders`, `fills`
- `trades`, `trade_legs`

ML/backtest scaffolding:
- `strategy_versions`
- `model_versions`
- `training_runs`
- `feature_snapshots`
- `trade_candidates`
- `model_predictions`
- `backtest_runs`
- `strategy_recommendations`

Initialization behavior:
- schema is idempotent (`IF NOT EXISTS`)
- executed statement-by-statement for asyncpg compatibility
- ML tables now include explicit schema-version and labeling fields to support feature/candidate/prediction lineage.

Destructive ML reset:
- reset SQL: `spx_backend/database/sql/db_reset_ml_tables.sql`
- CLI: `python -m spx_backend.database.reset_ml_schema`
- This drops and recreates ML/decision/trade tables only (market-data ingestion tables are preserved).

Destructive full reset:
- reset SQL: `spx_backend/database/sql/db_reset_all_tables.sql`
- CLI: `python -m spx_backend.database.reset_all_schema`
- This drops and recreates all app tables, including market-data ingestion history.

---

## 9) Local Run Instructions

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
cd backend
python -m spx_backend.main
```

Reset ML schema (destructive; keeps snapshots/quotes/GEX tables):

```bash
cd backend
python -m spx_backend.database.reset_ml_schema
```

Reset all schema (destructive; drops all app tables):

```bash
cd backend
python -m spx_backend.database.reset_all_schema
```

Smoke test:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/admin/preflight
```

Manual pipeline test:

```bash
curl -X POST http://localhost:8000/api/admin/run-quotes
curl -X POST http://localhost:8000/api/admin/run-snapshot
curl -X POST http://localhost:8000/api/admin/run-gex
curl -X POST http://localhost:8000/api/admin/run-feature-builder
curl -X POST http://localhost:8000/api/admin/run-labeler
curl -X POST http://localhost:8000/api/admin/run-decision
curl -X POST http://localhost:8000/api/admin/run-trade-pnl
curl http://localhost:8000/api/admin/preflight
```

---

## 10) Testing

Install test dependencies:

```bash
cd backend
python -m pip install -r requirements-dev.txt
```

Run:

```bash
python -m pytest -q
```

DB-backed integration tests (safe mode):

1) Start dedicated local Postgres test DB:

```bash
cd ..
docker compose -f docker-compose.test.yml up -d
```

2) Set `DATABASE_URL_TEST` (example in `backend/.env.test.example`):

```bash
export DATABASE_URL_TEST="postgresql+asyncpg://spx_test:spx_test_pw@localhost:5434/index_spread_lab_test"
```

3) Run only integration tests:

```bash
cd backend
python -m pytest -q -m integration
```

Convenience Make targets (run from repo root):

```bash
make test-e2e-up
make test-e2e-mocked
make test-e2e-db
make test-e2e
make test-e2e-regression
make test-predeploy
make test-e2e-down
```

If your default `python` is not your backend runtime, override interpreter:

```bash
make PYTHON_BIN=python3.11 test-e2e
make PYTHON_BIN=python3.11 test-predeploy
```

Safety guard behavior:
- Integration tests skip entirely when `DATABASE_URL_TEST` is unset.
- Integration tests fail fast if `DATABASE_URL_TEST` host is not `localhost`/`127.0.0.1` or DB name does not include `test`.
- `make test-e2e-regression` runs only failure/edge regression checks (`-m "integration and regression"`).

Current coverage domains:
- DTE helper behavior
- Tradier expiration request params
- Snapshot strike-selection helper
- Decision candidate/scoring/freshness logic
- GEX zero-gamma helper
- GEX API output behavior (including custom expiration filter and fallback)
- HTTP-level mocked E2E router workflows
- DB-backed integration smoke tests (`-m integration`)
- DB-backed regression failure pack (quote fetch fail, no expirations, no shadow model, promotion gate fail/pass)
- DB-backed trainer/shadow run-once integration coverage

---

## 11) Railway Deployment Notes

Required:
- valid `DATABASE_URL`
- `CORS_ORIGINS` includes deployed frontend

Tradier mode:
- sandbox: `TRADIER_BASE_URL=https://sandbox.tradier.com/v1`
- live: `TRADIER_BASE_URL=https://api.tradier.com/v1`

Recommended production behavior:
- `SNAPSHOT_RANGE_FALLBACK_ENABLED=false` (strict)
- keep snapshot/quote outside-RTH flags disabled unless intentionally testing

---

## 12) Operational Debugging Queries

Use API first:
- `GET /api/admin/preflight` for fast status.

Useful SQL checks:

```sql
-- latest snapshots
SELECT snapshot_id, ts, target_dte, expiration
FROM chain_snapshots
ORDER BY ts DESC
LIMIT 20;

-- latest decisions
SELECT decision_id, ts, decision, reason, target_dte, delta_target, score
FROM trade_decisions
ORDER BY ts DESC
LIMIT 20;

-- latest gex summary
SELECT snapshot_id, ts, gex_net, zero_gamma_level
FROM gex_snapshots
ORDER BY ts DESC
LIMIT 20;
```

---

## 13) Known Limitations

- Hybrid decision path is implemented, but defaults to rules-first with model ranking gated by config/promotions.
- Order placement/fill lifecycle tables exist but workflow remains staged.
- Frontend test coverage has not yet been added.
- Full historical replay/backtest orchestration is still pending.
