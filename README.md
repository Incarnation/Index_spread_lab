# IndexSpreadLab

IndexSpreadLab is a research and paper-execution platform for index options with a practical focus on short-dated credit spread workflows.

The current stack is:
- Backend: FastAPI + APScheduler + PostgreSQL
- Frontend: React (Vite) + Mantine + Recharts
- Data source: Tradier REST APIs (expirations, chain, quotes, market clock)

This repository is designed so live capture, analytics (GEX), and decision logs share one consistent schema that can later feed backtesting and ML.

---

## Current Product Scope

What is implemented now:
- Scheduled SPX option-chain snapshot capture (0-10 DTE range profile).
- Scheduled SPY snapshot stream (enabled by default) and optional VIX snapshot stream.
- Scheduled quote capture (SPX, VIX, VIX9D, SPY by default).
- Scheduled GEX computation and persistence (Tradier-computed and CBOE precomputed streams).
- Feature/candidate generation for both `put` and `call` credit spreads.
- Label resolution with both live-style TP50 outcome and expiry counterfactual outcome.
- Weekly walk-forward trainer (with sparse CV fallback) that writes `training_runs` and `model_versions`.
- Shadow inference writer to `model_predictions`.
- Promotion gate evaluator for rollout safety checks.
- Hybrid execution policy support (rules guardrails first, model ranking second) with safe default `decision_source='rules'`.
- Pipeline staleness monitoring with SendGrid email alerting (RTH-only, cooldown-gated).
- Performance analytics aggregation (win rates, expectancy, drawdown, tail loss, margin usage).
- Admin APIs to manually trigger each pipeline stage.
- React dashboard with ErrorBoundary, strategy quality/risk cards, GEX panels, and live trade PnL.
- Data retention CLI for exporting and purging old chain/GEX data.
- Backend unit/integration/E2E suites with a predeploy gate and CI workflow.

Still intentionally limited:
- Live broker order/fill automation remains staged (schema and paper workflow are available).
- Backtest runner orchestration and full historical backfill are still evolving.

---

## How The System Works

The pipeline runs in this order:

1) Quote job
- Pulls latest quotes for configured symbols (`QUOTE_SYMBOLS`).
- Stores raw quote rows in `underlying_quotes`.
- Updates `context_snapshots` (`spx_price`, `vix`, `term_structure`, etc).

2) Snapshot job
- Gets SPX expirations from Tradier (including all roots for weeklies/dailies).
- Selects expirations by DTE policy (`range` or `targets`).
- Pulls option chains and writes:
  - `chain_snapshots` (metadata + checksum; raw payload cleared to save storage)
  - `option_chain_rows` (normalized per-option rows)
- Dedicated SPY snapshot stream (enabled by default) runs in parallel and stores `underlying='SPY'` rows for future SPY model fitting.
- Optional VIX snapshot stream can run in parallel and writes the same tables with `underlying='VIX'` for model fitting features.
- Decision/trade execution remains SPX-only.

3) GEX job
- Reads option rows + latest eligible spot quote.
- Computes gamma exposure aggregates.
- Writes:
  - `gex_snapshots`
  - `gex_by_strike`
  - `gex_by_expiry_strike`

4) Feature builder + candidate generation
- At configured entry times, builds feature snapshots and ranked candidates.
- Generates candidates for both `put` and `call` sides.
- Writes:
  - `feature_snapshots`
  - `trade_candidates` (ranked, hashed candidates)

5) Decision + execution policy
- Enforces hard risk guardrails first (day/open caps and per-side caps).
- Selects candidate by rules score by default.
- If hybrid is enabled and eligible model predictions exist, applies model ranking subject to safety thresholds.
- Writes one decision row per run (`TRADE`/`SKIP`) in `trade_decisions`, then creates/updates paper trade rows.

6) Labeler + trade PnL
- Trade PnL job marks open trades continuously and closes on TP/SL/expiry policy.
- Labeler resolves candidate outcomes from forward marks and stores:
  - `hit_tp50_before_sl_or_expiry`
  - `hit_tp100_at_expiry`
  - `realized_pnl` and separate expiry counterfactual fields.

7) Weekly model loop
- Trainer runs walk-forward evaluation and writes `training_runs` + `model_versions`.
- Falls back to sparse cross-validation when walk-forward split has insufficient rows.
- Shadow inference scores fresh candidates and writes `model_predictions`.
- Promotion gates evaluate quality/risk thresholds and update rollout status.

8) Staleness monitor
- Periodically checks latest timestamps in `underlying_quotes`, `chain_snapshots`, `gex_snapshots`, `trade_decisions`.
- Runs only during RTH (skips evenings, weekends, exchange holidays).
- Sends email alert via SendGrid when any source exceeds its configured staleness threshold.
- Cooldown period prevents duplicate alerts.

9) Performance analytics
- Aggregates win rates, expectancy, drawdown, tail loss proxy, and margin usage.
- Refreshes on a configurable interval during RTH.

Important semantics:
- DTE handling is trading-session based (weekends and market holidays are skipped).
- GEX API data for UI is batch-scoped (same timestamp + underlying + source), not only one snapshot row.

---

## DTE Semantics (Critical)

The project does not use raw calendar-day difference for target selection.
It uses trading-session progression inferred from Tradier expiration sessions.

Example from Thu 2026-02-12:
- 0DTE -> 2026-02-12
- 1DTE -> 2026-02-13
- 2DTE -> 2026-02-17
- 3DTE -> 2026-02-18

Why 3DTE is 2026-02-18:
- 2026-02-16 is a market holiday (Presidents' Day), so it is skipped as a trading session.

---

## Repository Layout

- `backend/`
  - `spx_backend/config.py`: env-backed settings.
  - `spx_backend/main.py`: app entrypoint (`PORT` aware for Railway).
  - `spx_backend/web/app.py`: FastAPI app + scheduler wiring.
  - `spx_backend/web/routers/`: public/admin route modules.
  - `spx_backend/database/`: DB package (`connection.py`, `schema.py`, reset CLIs).
  - `spx_backend/database/sql/migrations/`: idempotent SQL migrations (run automatically on startup).
  - `spx_backend/backtest/`: local backtest engine with path-sanitization + sample data docs.
  - `spx_backend/jobs/`: snapshot, quote, gex, feature-builder, decision, labeler, trade-pnl, trainer, shadow-inference, promotion-gate, staleness-monitor, performance-analytics jobs.
  - `spx_backend/market_clock.py`: Tradier clock cache with DB audit and RTH fallback.
  - `spx_backend/dte.py`: trading-day DTE helper logic.
  - `scripts/data_retention.py`: CLI to export and purge old chain/GEX data.
  - `requirements.txt`: runtime dependencies.
  - `requirements-dev.txt`: test dependencies.
  - `tests/`: backend automated tests.
- `frontend/`
  - `src/DashboardApp.tsx`: top-level container.
  - `src/components/`: UI panels, widgets, and `ErrorBoundary`.
  - `src/hooks/`: data and action hooks.
  - `src/api.ts`: typed API client with `safeJson` response parsing.
  - `src/contexts/AuthContext.tsx`: JWT auth state with 401 event handling.
- `.dockerignore`: slimmed Docker build context (excludes frontend, docs, data files).

---

## Configuration

Copy `.env.example` to `.env` and fill values.

Required:
- `DATABASE_URL` (`postgresql+asyncpg://...`)
- `TRADIER_ACCESS_TOKEN`
- `TRADIER_ACCOUNT_ID`

Primary knobs:
- Snapshot:
  - `SNAPSHOT_DTE_MODE=range|targets`
  - `SNAPSHOT_DTE_MIN_DAYS`, `SNAPSHOT_DTE_MAX_DAYS`
  - `SNAPSHOT_STRIKES_EACH_SIDE`
  - `SPY_SNAPSHOT_ENABLED=true|false`
  - `SPY_SNAPSHOT_INTERVAL_MINUTES`
  - `SPY_SNAPSHOT_DTE_MODE=range|targets`
  - `SPY_SNAPSHOT_DTE_MIN_DAYS`, `SPY_SNAPSHOT_DTE_MAX_DAYS`
  - `SPY_SNAPSHOT_DTE_TARGETS`
  - `SPY_SNAPSHOT_STRIKES_EACH_SIDE`
  - `SPY_ALLOW_SNAPSHOT_OUTSIDE_RTH`
  - `VIX_SNAPSHOT_ENABLED=true|false`
  - `VIX_SNAPSHOT_INTERVAL_MINUTES`
  - `VIX_SNAPSHOT_DTE_MODE=range|targets`
  - `VIX_SNAPSHOT_DTE_MIN_DAYS`, `VIX_SNAPSHOT_DTE_MAX_DAYS`
  - `VIX_SNAPSHOT_DTE_TARGETS`
  - `VIX_SNAPSHOT_STRIKES_EACH_SIDE`
  - `VIX_ALLOW_SNAPSHOT_OUTSIDE_RTH`
- Decision:
  - `DECISION_ENTRY_TIMES`
  - `DECISION_DTE_TARGETS`
  - `DECISION_DTE_TOLERANCE_DAYS`
  - `DECISION_DELTA_TARGETS`
  - `DECISION_SPREAD_WIDTH_POINTS`
  - `DECISION_SPREAD_SIDES`
  - `DECISION_SNAPSHOT_MAX_AGE_MINUTES`
  - `DECISION_MAX_TRADES_PER_SIDE_PER_DAY`
  - `DECISION_MAX_OPEN_TRADES_PER_SIDE`
- Hybrid decision policy:
  - `DECISION_HYBRID_ENABLED`
  - `DECISION_HYBRID_MODEL_NAME`
  - `DECISION_HYBRID_MIN_PROBABILITY`
  - `DECISION_HYBRID_MIN_EXPECTED_PNL`
  - `DECISION_HYBRID_REQUIRE_ACTIVE_MODEL`
- ML (feature + label pipeline):
  - `FEATURE_BUILDER_ENABLED`
  - `FEATURE_BUILDER_ALLOW_OUTSIDE_RTH`
  - `FEATURE_SCHEMA_VERSION`
  - `CANDIDATE_SCHEMA_VERSION`
  - `LABELER_ENABLED`
  - `LABELER_BATCH_LIMIT`
  - `LABELER_TAKE_PROFIT_PCT`
  - `LABEL_SCHEMA_VERSION`
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
- Live trade PnL:
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
  - `GEX_ENABLED=true`
  - `GEX_MAX_DTE_DAYS`
  - `GEX_STRIKE_LIMIT`
  - `GEX_SNAPSHOT_BATCH_LIMIT` (recommended >= number of expirations captured per cycle; default `20`)
- CBOE precomputed GEX (parallel stream):
  - `MZDATA_BASE_URL`
  - `CBOE_GEX_ENABLED`
  - `CBOE_GEX_UNDERLYINGS` (default `SPX,SPY,VIX`)
  - `CBOE_GEX_UNDERLYING` (legacy fallback during migration)
  - `CBOE_GEX_INTERVAL_MINUTES` (default `15`)
  - `CBOE_GEX_ALLOW_OUTSIDE_RTH` (default `false`)
- Staleness alerting:
  - `STALENESS_ALERT_ENABLED` (default `false`)
  - `STALENESS_ALERT_INTERVAL_MINUTES` (default `30`)
  - `STALENESS_QUOTES_MAX_MINUTES`, `STALENESS_SNAPSHOTS_MAX_MINUTES`, `STALENESS_GEX_MAX_MINUTES` (default `120`)
  - `STALENESS_DECISIONS_MAX_MINUTES` (default `480`)
  - `STALENESS_COOLDOWN_MINUTES` (default `360`)
  - `SENDGRID_API_KEY`, `EMAIL_ALERT_RECIPIENT`, `EMAIL_ALERT_SENDER`
- Ops:
  - `CORS_ORIGINS`
  - `ALLOW_SNAPSHOT_OUTSIDE_RTH`
  - `ALLOW_QUOTES_OUTSIDE_RTH`
  - `MARKET_CLOCK_CACHE_SECONDS`

Production recommendation:
- Do not include quotes in Railway variable values (use raw values, e.g. `false`, not `"false"`).
- Keep `SPY_ALLOW_SNAPSHOT_OUTSIDE_RTH=false` for production scheduling.
- If enabling VIX snapshots, keep `VIX_ALLOW_SNAPSHOT_OUTSIDE_RTH=false` by default; use VIX chains for model features/training context, not direct execution signals.

---

## Local Development

### 1) Backend

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
cd backend
python -m spx_backend.main
```

Backend default URL:
- `http://localhost:8000`

### 2) Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend default URL:
- `http://localhost:5173`

---

## API Surface

Public read endpoints:
- `GET /health`
- `GET /api/chain-snapshots`
- `GET /api/gex/snapshots`
- `GET /api/gex/dtes?snapshot_id=...`
- `GET /api/gex/expirations?snapshot_id=...`
- `GET /api/gex/curve?snapshot_id=...`
- `GET /api/trade-decisions`
- `GET /api/trades`
- `GET /api/label-metrics`
- `GET /api/strategy-metrics`
- `GET /api/model-ops`

Admin endpoints (authenticated user required):
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

---

## Database Model Overview

Core ingestion:
- `chain_snapshots`: one row per captured chain (metadata + checksum; raw payload cleared to save storage).
- `option_chain_rows`: normalized options from each snapshot.
- `underlying_quotes`: raw quote history.
- `context_snapshots`: derived market context per timestamp.
- `market_clock_audit`: Tradier clock states/errors.

GEX:
- `gex_snapshots`: summary values per snapshot.
- `gex_by_strike`: strike curve per snapshot.
- `gex_by_expiry_strike`: expiration-strike curve with DTE labels.

Decisions/trading:
- `trade_decisions`: TRADE/SKIP decisions and metadata.
- `orders`, `fills`, `trades`, `trade_legs`: lifecycle schema scaffolding.

ML/backtest scaffolding:
- `strategy_versions`, `model_versions`, `training_runs`
- `feature_snapshots`, `trade_candidates`, `model_predictions`
- `backtest_runs`, `strategy_recommendations`
- these tables now include schema-version and label-tracking fields for ML lineage.

Destructive ML reset command (keeps ingestion tables):

```bash
cd backend
python -m spx_backend.database.reset_ml_schema
```

Destructive full reset command (drops all app tables):

```bash
cd backend
python -m spx_backend.database.reset_all_schema
```

---

## Testing

Backend tests:

```bash
cd backend
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

DB-backed integration tests (separate local test DB):

```bash
docker compose -f docker-compose.test.yml up -d
export DATABASE_URL_TEST="postgresql+asyncpg://spx_test:spx_test_pw@localhost:5434/index_spread_lab_test"
cd backend
python -m pytest -q -m integration
```

Convenience Make targets (from repo root):

```bash
make test-e2e-up
make test-e2e-mocked
make test-e2e-db
make test-e2e
make test-e2e-regression
make test-predeploy
make test-e2e-down
```

If needed, specify interpreter explicitly:

```bash
make PYTHON_BIN=python3.11 test-e2e
make PYTHON_BIN=python3.11 test-predeploy
```

Safety behavior:
- integration tests skip if `DATABASE_URL_TEST` is not set
- they fail fast if DB host is not local or DB name does not include `test`
- `make test-e2e-regression` runs only targeted failure/edge-path checks (`-m "integration and regression"`)

Current test coverage includes:
- Trading-day DTE mapping and expiration selection.
- Decision candidate construction and snapshot freshness behavior.
- GEX endpoint output modes (all / DTE / custom expirations).
- Snapshot strike window helper.
- Tradier expirations request parameter correctness.
- Trade PnL: mark-to-market, TP/SL/expiry close, bulk leg loading, expired trade handling.
- Staleness monitor: freshness detection, cooldown, SendGrid alerting, RTH guard.
- Market clock: `is_rth` boundary cases, `MarketClockCache` with mock Tradier and fallback.
- Quote job: market gate, symbol parsing, fetch failure, successful insertion.
- Trainer: walk-forward windows, sparse CV folds, walk-forward-to-sparse-CV fallback.
- Config parsing: list fields, dedup, bad symbols, delta targets.
- Backtest engine: parquet path sanitization (traversal, semicolons, SQL keywords).
- HTTP-level mocked E2E API workflows.
- DB-backed integration smoke tests behind `-m integration`.
- DB-backed regression failure-pack (quote fetch fail, no-expiration snapshot, no-shadow-model, promotion gate fail/pass branches).

---

## Deployment (Railway)

Backend service:
- Uses root `Dockerfile` with `.dockerignore` for a lean build context.
- Reads `PORT` automatically.
- Requires valid `DATABASE_URL` and Tradier credentials.
- SQL migrations run automatically on startup (idempotent).

Checklist:
1) Provision Railway Postgres.
2) Set backend env vars from `.env.example`.
3) Set `CORS_ORIGINS` to your frontend domain.
4) Keep secrets only in Railway variables.
5) Run `make test-predeploy` before shipping changes.

Frontend service:
- Deploy `frontend/` as static Vite app.
- Set `VITE_API_BASE_URL=https://<backend-domain>`.

---

## Ops Runbook

Recommended manual sequence for diagnostics:
1) `POST /api/admin/run-quotes`
2) `POST /api/admin/run-snapshot`
3) `POST /api/admin/run-gex`
4) `POST /api/admin/run-cboe-gex` (if CBOE stream enabled)
5) `POST /api/admin/run-feature-builder`
6) `POST /api/admin/run-labeler`
7) `POST /api/admin/run-decision`
8) `POST /api/admin/run-trade-pnl`
9) `GET /api/admin/preflight`

If decisions are skipping:
- Check `preflight.latest.snapshot_ts` freshness.
- Check `DECISION_DTE_TARGETS` and tolerance.
- Check `DECISION_SNAPSHOT_MAX_AGE_MINUTES`.
- Inspect last `trade_decisions` reason.

If model loop is not producing predictions yet:
- After a fresh DB reset, `no_model_versions` and `no_model_predictions` warnings are expected initially.
- Wait for enough resolved labels, or temporarily reduce trainer minimum-row thresholds to bootstrap first model.

---

## Common Troubleshooting

No near-term expirations:
- Verify `TRADIER_BASE_URL` and token.
- Use `GET /api/admin/expirations?symbol=SPX`.

GEX backlog:
- Increase `GEX_SNAPSHOT_BATCH_LIMIT`.
- Ensure quote job is running so spot is available.

Only one date in custom DTE dropdown:
- Ensure multiple expirations were captured in the same timestamp batch.
- Verify `GET /api/gex/expirations?snapshot_id=...`.

Unexpected DTE mapping:
- Remember DTE is trading-session based, not calendar-day diff.

---

## Next Improvements

- Split CI predeploy checks into faster parallel jobs while preserving gate quality.
- Expand backtest orchestration and historical replay tooling.
- Add frontend automated tests (Vitest + React Testing Library).
- Add richer decision skip taxonomy views in dashboard.
- Live broker order/fill automation beyond paper trading.
- Periodic data retention automation (scheduled purge of old chain/GEX data).
