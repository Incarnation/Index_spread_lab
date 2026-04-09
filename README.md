# IndexSpreadLab

IndexSpreadLab is a research and paper-execution platform for index options with a practical focus on short-dated credit spread workflows.

The current stack is:
- Backend: FastAPI + APScheduler + PostgreSQL
- Frontend: React 18 (Vite) + Radix UI + Tailwind CSS v4 + Recharts
- Data sources: Tradier REST APIs (expirations, chain, quotes, market clock) + CBOE precomputed GEX
- Auth: JWT-based authentication with bcrypt password hashing

This repository is designed so live capture, analytics (GEX), and decision logs share one consistent schema that can later feed backtesting and ML.

---

## Current Product Scope

What is implemented now:
- Scheduled SPX option-chain snapshot capture (0-16 DTE range profile).
- Scheduled SPY snapshot stream (enabled by default) and optional VIX snapshot stream.
- Scheduled quote capture (SPX, VIX, VIX9D, SPY by default).
- Dual-source GEX computation: Tradier-computed and CBOE precomputed streams with source-specific columns.
- Feature/candidate generation for both `put` and `call` credit spreads.
- Label resolution with both live-style TP50 outcome and expiry counterfactual outcome.
- Weekly walk-forward trainer (with sparse CV fallback) that writes `training_runs` and `model_versions`.
- Shadow inference writer to `model_predictions`.
- Promotion gate evaluator for rollout safety checks.
- Hybrid execution policy support (rules guardrails first, model ranking second) with safe default `decision_source='rules'`.
- Portfolio management with capital budgeting, lot scaling, drawdown stops, and trade-source tracking.
- Event-driven trading layer (VIX spikes, SPX drops, term-structure inversion) with configurable budget mode.
- EOD events job for end-of-day signal capture and economic calendar integration.
- Automated data retention job (configurable purge of old chain/GEX data with cascade safety).
- Pipeline staleness monitoring with SendGrid email alerting (RTH-only, cooldown-gated).
- Performance analytics aggregation (win rates, expectancy, drawdown, tail loss, margin usage).
- Admin APIs to manually trigger each pipeline stage.
- JWT authentication with user registration, login, logout, and auth audit logging.
- React dashboard with lazy-loaded pages, error boundaries, responsive mobile sidebar, and live trade PnL.
- Data retention CLI for exporting and purging old chain/GEX data.
- Backend unit/integration/E2E suites with a predeploy gate and CI workflow.
- Frontend test suite (Vitest + Testing Library) with page-level tests.

Still intentionally limited:
- Live broker order/fill automation remains staged (schema and paper workflow are available).
- Backtest runner orchestration and full historical backfill are still evolving.

---

## How The System Works

The pipeline runs in this order:

1) **Quote job**
- Pulls latest quotes for configured symbols (`QUOTE_SYMBOLS`).
- Stores raw quote rows in `underlying_quotes`.
- Updates `context_snapshots` (`spx_price`, `vix`, `term_structure`, etc).

2) **Snapshot job**
- Gets SPX expirations from Tradier (including all roots for weeklies/dailies).
- Selects expirations by DTE policy (`range` or `targets`).
- Pulls option chains and writes:
  - `chain_snapshots` (metadata + checksum; raw payload cleared to save storage)
  - `option_chain_rows` (normalized per-option rows)
- Dedicated SPY snapshot stream (enabled by default) runs in parallel and stores `underlying='SPY'` rows for future SPY model fitting.
- Optional VIX snapshot stream can run in parallel and writes the same tables with `underlying='VIX'` for model features/training context.
- Decision/trade execution remains SPX-only.

3) **GEX job (Tradier)**
- Reads option rows + latest eligible spot quote.
- Computes gamma exposure aggregates.
- Writes `gex_snapshots`, `gex_by_strike`, `gex_by_expiry_strike`.
- Populates `gex_net_tradier` / `zero_gamma_level_tradier` source columns in `context_snapshots`.

4) **CBOE GEX job**
- Fetches precomputed GEX data from CBOE/MZData vendor API.
- Writes the same GEX tables with `source='CBOE'`.
- Populates `gex_net_cboe` / `zero_gamma_level_cboe` source columns in `context_snapshots`.
- CBOE is the preferred source for canonical `gex_net`; Tradier is the fallback.

5) **Feature builder + candidate generation**
- At configured entry times, builds feature snapshots and ranked candidates.
- Generates candidates for both `put` and `call` sides.
- Writes `feature_snapshots` and `trade_candidates` (ranked, hashed candidates).

6) **Decision + execution policy**
- Enforces hard risk guardrails first (day/open caps and per-side caps).
- Selects candidate by rules score by default.
- If hybrid is enabled and eligible model predictions exist, applies model ranking subject to safety thresholds.
- Portfolio manager handles capital budgeting, lot sizing, and drawdown stops when portfolio mode is active.
- Writes one decision row per run (`TRADE`/`SKIP`) in `trade_decisions`, then creates/updates paper trade rows.

7) **Labeler + trade PnL**
- Trade PnL job marks open trades continuously and closes on TP/SL/expiry policy.
- Labeler resolves candidate outcomes from forward marks and stores labels.
- Orphan feature snapshots (no linked candidates) are automatically expired.

8) **Weekly model loop**
- Trainer runs walk-forward evaluation and writes `training_runs` + `model_versions`.
- Falls back to sparse cross-validation when walk-forward split has insufficient rows.
- Shadow inference scores fresh candidates and writes `model_predictions`.
- Promotion gates evaluate quality/risk thresholds and update rollout status.

9) **EOD events job**
- Runs after market close to capture end-of-day signals and economic calendar events.
- Writes to `economic_events` table for next-day decision context.

10) **Performance analytics**
- Aggregates win rates, expectancy, drawdown, tail loss proxy, and margin usage.
- Refreshes on a configurable interval during RTH.

11) **Staleness monitor**
- Periodically checks latest timestamps in key tables.
- Runs only during RTH (skips evenings, weekends, exchange holidays).
- Sends email alert via SendGrid when any source exceeds its configured staleness threshold.

12) **Retention job**
- Runs daily at 3 AM ET when enabled.
- Deletes old `chain_snapshots` and cascaded children in batches.
- Excludes snapshots referenced by open trades as a safety measure.
- Configurable retention window (default 60 days).

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
  - `spx_backend/web/routers/`: route modules (`public.py`, `admin.py`, `auth.py`, `portfolio.py`).
  - `spx_backend/database/`: DB package (`connection.py`, `schema.py`, reset CLIs).
  - `spx_backend/database/sql/migrations/`: idempotent SQL migrations (run automatically on startup).
  - `spx_backend/services/portfolio_manager.py`: capital budgeting, lot scaling, drawdown stops.
  - `spx_backend/backtest/`: local backtest engine with path-sanitization + sample data docs.
  - `spx_backend/jobs/`: all pipeline jobs (snapshot, quote, gex, cboe_gex, feature_builder, decision, labeler, trade_pnl, trainer, shadow_inference, promotion_gate, staleness_monitor, performance_analytics, eod_events, retention).
  - `spx_backend/market_clock.py`: Tradier clock cache with DB audit and RTH fallback.
  - `spx_backend/dte.py`: trading-day DTE helper logic.
  - `spx_backend/scheduler_builder.py`: APScheduler construction and job registration.
  - `scripts/data_retention.py`: CLI to export and purge old chain/GEX data.
  - `requirements.txt`: runtime dependencies.
  - `requirements-dev.txt`: test dependencies.
  - `tests/`: backend automated tests (45 test files).
- `frontend/`
  - `src/main.tsx`: entry point with lazy-loaded routing and auth provider.
  - `src/app/`: layout shell (`AppShell`, `Sidebar` with mobile drawer, `Navbar`).
  - `src/pages/`: per-route page components (Overview, Portfolio, Trades, Decisions, ModelMonitor, Performance, GEX, Admin, AuthAudit, StrategyConfig).
  - `src/components/`: shared UI (`ProtectedRoute`, `ErrorBoundary`, `DataTable`, `StatCard`, Radix primitives).
  - `src/hooks/`: `useAutoRefresh` with market-hours awareness.
  - `src/api.ts`: typed API client with `safeJson` response parsing.
  - `src/contexts/AuthContext.tsx`: JWT auth state with 401 event handling.
  - `src/test/`: test setup and page-level tests.
- `Makefile`: E2E and predeploy test targets.
- `Dockerfile`: backend Docker image.
- `docker-compose.test.yml`: Postgres test DB for integration tests.
- `.dockerignore`: slimmed Docker build context (excludes frontend, docs, data files).

---

## Configuration

Copy `.env.example` to `.env` and fill values. A separate `frontend/.env.example` exists for frontend-specific settings.

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
  - `VIX_SNAPSHOT_ENABLED=true|false`
- Decision:
  - `DECISION_ENTRY_TIMES`
  - `DECISION_DTE_TARGETS`
  - `DECISION_DELTA_TARGETS`, `DECISION_PUT_DELTA_TARGETS`, `DECISION_CALL_DELTA_TARGETS`
  - `DECISION_SPREAD_WIDTH_POINTS`, `DECISION_SPREAD_SIDES`
  - `DECISION_MAX_TRADES_PER_SIDE_PER_DAY`, `DECISION_MAX_OPEN_TRADES_PER_SIDE`
- Hybrid decision policy:
  - `DECISION_HYBRID_ENABLED`
  - `DECISION_HYBRID_MODEL_NAME`
  - `DECISION_HYBRID_MIN_PROBABILITY`, `DECISION_HYBRID_MIN_EXPECTED_PNL`
- Portfolio management:
  - `PORTFOLIO_ENABLED`, `PORTFOLIO_STARTING_CAPITAL`
  - `PORTFOLIO_MAX_TRADES_PER_DAY`, `PORTFOLIO_MAX_TRADES_PER_RUN`
  - `PORTFOLIO_MONTHLY_DRAWDOWN_LIMIT`, `PORTFOLIO_LOT_PER_EQUITY`
  - `PORTFOLIO_CALLS_ONLY`
- Event-driven trading:
  - `EVENT_ENABLED`, `EVENT_BUDGET_MODE`
  - `EVENT_SPX_DROP_THRESHOLD`, `EVENT_VIX_SPIKE_THRESHOLD`
  - `EVENT_MIN_DTE`, `EVENT_MAX_DTE`, `EVENT_MIN_DELTA`, `EVENT_MAX_DELTA`
- ML (feature + label pipeline):
  - `FEATURE_BUILDER_ENABLED`, `LABELER_ENABLED`
  - `TRAINER_ENABLED`, `SHADOW_INFERENCE_ENABLED`
  - `PROMOTION_GATE_ENABLED`
- GEX:
  - `GEX_ENABLED`, `GEX_SNAPSHOT_BATCH_LIMIT`
  - `CBOE_GEX_ENABLED`, `CBOE_GEX_UNDERLYINGS`
- Data retention:
  - `RETENTION_ENABLED` (default `false`)
  - `RETENTION_DAYS` (default `60`)
  - `RETENTION_BATCH_SIZE` (default `500`)
- EOD events:
  - `EOD_EVENTS_ENABLED` (default `true`)
  - `EOD_EVENTS_HOUR`, `EOD_EVENTS_MINUTE`
- Trade PnL:
  - `TRADE_PNL_ENABLED`, `TRADE_PNL_INTERVAL_MINUTES`
  - `TRADE_PNL_TAKE_PROFIT_PCT`, `TRADE_PNL_STOP_LOSS_PCT`
  - `TRADE_PNL_STOP_LOSS_BASIS` (`max_profit` or `max_loss`)
- Performance analytics:
  - `PERFORMANCE_ANALYTICS_ENABLED`, `PERFORMANCE_ANALYTICS_INTERVAL_MINUTES`
- Staleness alerting:
  - `STALENESS_ALERT_ENABLED` (default `true`)
  - `STALENESS_ALERT_INTERVAL_MINUTES`
  - `SENDGRID_API_KEY`, `EMAIL_ALERT_RECIPIENT`, `EMAIL_ALERT_SENDER`
- Ops:
  - `CORS_ORIGINS`
  - `ALLOW_SNAPSHOT_OUTSIDE_RTH`, `ALLOW_QUOTES_OUTSIDE_RTH`
  - `MARKET_CLOCK_CACHE_SECONDS`

See `backend/spx_backend/config.py` for the full list of settings with defaults.

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

### Authentication

- `POST /api/auth/login` -- login with username/password, returns JWT
- `POST /api/auth/register` -- create new user account
- `GET /api/auth/me` -- current user info (authenticated)
- `POST /api/auth/logout` -- logout (authenticated)

### Health

- `GET /health` -- liveness probe (unauthenticated)

### Data endpoints (authenticated)

- `GET /api/pipeline-status` -- pipeline freshness and warnings
- `GET /api/chain-snapshots` -- recent chain snapshot metadata
- `GET /api/trade-decisions` -- recent decisions (TRADE/SKIP)
- `GET /api/trades` -- recent/open/closed paper trade rows with legs and PnL
- `GET /api/label-metrics` -- TP50/TP100 and realized PnL summary
- `GET /api/strategy-metrics` -- strategy quality/risk metrics
- `GET /api/performance-analytics` -- performance analytics with breakdowns
- `GET /api/model-ops` -- model/training/prediction operational summary
- `GET /api/model-predictions` -- paginated prediction browser
- `GET /api/model-accuracy` -- accuracy/precision/recall over time windows
- `GET /api/model-calibration` -- calibration curve bins
- `GET /api/model-pnl-attribution` -- model PnL attribution vs baseline
- `GET /api/gex/snapshots` -- recent GEX snapshot batches
- `GET /api/gex/dtes` -- available DTE options for a GEX batch
- `GET /api/gex/expirations` -- available expirations for a GEX batch
- `GET /api/gex/curve` -- strike curve for all, one DTE, or custom expirations
- `GET /api/backtest-results` -- backtest run results

### Portfolio endpoints (authenticated, prefix `/api/portfolio`)

- `GET /api/portfolio/status` -- current equity, lots, drawdown state
- `GET /api/portfolio/history` -- daily equity history
- `GET /api/portfolio/trades` -- portfolio trade log with source tracking
- `GET /api/portfolio/config` -- active portfolio + event + decision config

### Admin endpoints (authenticated)

- `POST /api/admin/run-snapshot`
- `POST /api/admin/run-quotes`
- `POST /api/admin/run-gex`
- `POST /api/admin/run-cboe-gex`
- `POST /api/admin/run-decision`
- `POST /api/admin/run-feature-builder`
- `POST /api/admin/run-labeler`
- `POST /api/admin/run-trade-pnl`
- `POST /api/admin/run-performance-analytics`
- `POST /api/admin/run-trainer`
- `POST /api/admin/run-shadow-inference`
- `POST /api/admin/run-promotion-gates`
- `DELETE /api/admin/trade-decisions/{decision_id}`
- `GET /api/admin/expirations?symbol=SPX`
- `GET /api/admin/auth-audit` -- authentication audit log
- `GET /api/admin/preflight` -- one-call pipeline health check

---

## Database Model Overview

Auth:
- `users`: user accounts with hashed passwords.
- `auth_audit_log`: login/logout/register events.

Core ingestion:
- `chain_snapshots`: one row per captured chain (metadata + checksum; raw payload cleared to save storage).
- `option_chain_rows`: normalized options from each snapshot.
- `underlying_quotes`: raw quote history.
- `context_snapshots`: derived market context per timestamp (includes source-specific GEX columns).
- `market_clock_audit`: Tradier clock states/errors.
- `option_instruments`: option metadata registry.

GEX:
- `gex_snapshots`: summary values per snapshot.
- `gex_by_strike`: strike curve per snapshot.
- `gex_by_expiry_strike`: expiration-strike curve with DTE labels.

Decisions/trading:
- `trade_decisions`: TRADE/SKIP decisions and metadata.
- `orders`, `fills`: broker order/fill lifecycle scaffolding.
- `trades`, `trade_legs`: multi-leg trade state.
- `trade_marks`: rolling mark-to-market history.

Portfolio:
- `portfolio_state`: daily equity, lot sizing, drawdown stops, event signals.
- `portfolio_trades`: trade-level capital tracking with source attribution.

Performance:
- `trade_performance_snapshots`: aggregated performance snapshots.
- `trade_performance_breakdowns`: per-dimension breakdowns (side, DTE, delta, etc).
- `trade_performance_equity_curve`: daily equity curve data.

ML/backtest scaffolding:
- `strategy_versions`, `model_versions`, `training_runs`
- `feature_snapshots`, `trade_candidates`, `model_predictions`
- `backtest_runs`, `strategy_recommendations`

Other:
- `economic_events`: economic calendar events for EOD signal context.
- `schema_migrations`: migration tracking.

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

Backend tests (45 test files):

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

Frontend tests:

```bash
cd frontend
npm install
npm run test
```

Safety behavior:
- Integration tests skip if `DATABASE_URL_TEST` is not set.
- They fail fast if DB host is not local or DB name does not include `test`.

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
- Add richer decision skip taxonomy views in dashboard.
- Live broker order/fill automation beyond paper trading.
