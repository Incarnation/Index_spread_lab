# IndexSpreadLab

IndexSpreadLab is a research and paper-execution platform for index options with a practical focus on short-dated credit spread workflows. It combines a live pipeline (data capture, event-signal detection, portfolio-managed trade execution) with an offline optimizer for systematic strategy development.

The current stack is:
- Backend: FastAPI + APScheduler + PostgreSQL
- Frontend: React 18 (Vite) + Radix UI + Tailwind CSS v4 + Recharts
- Data sources: Tradier REST APIs (expirations, chain, quotes, market clock) + CBOE precomputed GEX
- Auth: JWT-based authentication with bcrypt password hashing
- Notifications: Twilio SMS for trade open/close alerts

---

## System Architecture

These diagrams split the architecture by concern: **system context** (who talks to whom), **live pipeline** (causal data and trade flow), **scheduler timing** (when jobs run), and **offline optimizer** (research loop). For table-level data flow (which jobs write which tables), see [backend/README.md](backend/README.md) (Data Flow).

### System context

```mermaid
flowchart LR
  subgraph clients [Clients]
    browser[Browser]
  end
  subgraph ext [External services]
    tradier[Tradier API]
    cboe[CBOE MZData]
    twilio[Twilio SMS]
    sendgrid[SendGrid email]
  end
  subgraph app [IndexSpreadLab]
    spa[React SPA]
    api[FastAPI]
    sched[APScheduler jobs]
  end
  pg[(PostgreSQL)]
  browser -->|HTTPS| spa
  spa -->|JWT REST| api
  api --> pg
  sched --> pg
  tradier --> sched
  cboe --> sched
  sched --> twilio
  sched --> sendgrid
  pg --> api
```

### Live pipeline (causal workflow)

The live pipeline is rules-only. Candidates are constructed inline by `decision_job` from the latest chain snapshot using the portfolio-managed `credit_to_width` ranker. ML training tooling lives offline under `backend/scripts/` and registers artifacts in `model_versions`; re-entry onto the portfolio path is a configuration change.

```mermaid
flowchart TB
  subgraph capture [Quotes and chains]
    quoteJob[Quote job]
    snapJob[Snapshot job]
  end
  subgraph gamma [GEX dual source]
    gexJob[GEX job]
    cboeJob[CBOE GEX job]
  end
  subgraph policy [Execution]
    eventSig[Event signals from quotes]
    portMgr[Portfolio manager]
    decJob["Decision job rules + portfolio"]
  end
  subgraph lifecycle [Marks and exits]
    pnlJob[Trade PnL job]
  end
  subgraph ops [Operations]
    perfJob[Performance analytics]
    eodJob[EOD events]
    staleJob[Staleness monitor]
    retJob[Retention job]
  end
  db[(PostgreSQL)]
  sms[Twilio SMS]
  email[SendGrid]
  quoteJob --> db
  snapJob --> db
  snapJob --> gexJob
  db --> gexJob
  cboeJob --> db
  gexJob --> db
  db --> eventSig
  db --> portMgr
  eventSig --> decJob
  portMgr --> decJob
  db --> decJob
  decJob --> db
  decJob --> sms
  db --> pnlJob
  pnlJob --> db
  pnlJob --> sms
  db --> perfJob
  perfJob --> db
  eodJob --> db
  retJob --> db
  staleJob --> email
```

### Scheduler timing (when jobs run)

```mermaid
flowchart TB
  subgraph rth [RTH interval jobs]
    r_quote[quote_job]
    r_snap[snapshot_job]
    r_spy[spy_snapshot optional]
    r_vix[vix_snapshot optional]
    r_gex[gex_job]
    r_cboe[cboe_gex_job]
    r_perf[performance_analytics_job]
  end
  subgraph entry [Entry time jobs]
    e_dec["decision_job rules + portfolio"]
  end
  subgraph cadence [Interval daily]
    c_pnl[trade_pnl_job interval]
    c_stale[staleness_monitor_job interval]
    c_eod[eod_events_job scheduled]
    c_ret[retention_job daily 0300 ET]
  end
```

Job registration and guards (market-open, RTH windows, cooldowns) are described in [backend/README.md](backend/README.md) under **Scheduler Architecture**.

### Offline optimizer pipeline

```mermaid
flowchart LR
  subgraph inputs [Inputs]
    prodDb[(Production DB)]
    yamlCfg[YAML configs]
  end
  gen[generate_training_data.py]
  backtest[backtest_strategy.py]
  wf["Walk forward optional"]
  regime["regime_analysis optional"]
  ingest[ingest_optimizer_results.py]
  pg[(PostgreSQL)]
  dash[Optimizer dashboard]
  prodDb --> gen
  gen --> backtest
  yamlCfg --> backtest
  backtest --> wf
  backtest --> regime
  backtest --> ingest
  wf -.->|optional outputs| ingest
  regime -.->|optional outputs| ingest
  ingest --> pg
  pg --> dash
```

Walk-forward and regime analysis are optional steps that consume optimizer outputs; result CSVs are loaded into PostgreSQL via `ingest_optimizer_results.py` for the dashboard.

---

## Current Product Scope

What is implemented now:

**Live Data Capture**
- Scheduled SPX option-chain snapshot capture (0-16 DTE range profile).
- Scheduled SPY snapshot stream (enabled by default) and optional VIX snapshot stream.
- Scheduled quote capture (SPX, VIX, VIX9D, SPY by default).
- Dual-source GEX computation: Tradier-computed and CBOE precomputed streams.

**Trade Execution**
- Rules-based, capital-budgeted decision job: ranks live chain candidates by `credit_to_width` and applies portfolio gates (max trades, drawdown, lot scaling, margin caps).
- Event-driven trading layer: detects VIX spikes, SPX drops (1d/2d), VIX elevated, term-structure inversion, and rally conditions. Configurable signal mode (`any` or `spx_and_vix`).
- Event-only mode: suppresses scheduled trades entirely, only trading on event signals.
- Portfolio management with capital budgeting, lot scaling, drawdown stops, and trade-source tracking.
- SMS trade notifications via Twilio (open/close alerts with spread details).

**Offline ML Toolkit (re-entry path; not wired into the live pipeline)**
- `generate_training_data.py` exports production trades + simulated PnL trajectories.
- `xgb_model.py` trains an XGBoost entry model with walk-forward validation.
- `upload_xgb_model.py` registers a trained artifact into `model_versions`.
- The `model_versions` table is the registration target for offline-trained artifacts and is the re-entry point onto the portfolio decision path.

**Optimizer / Backtest System**
- Offline capital-budgeted backtester with parallel execution (`backtest_strategy.py`).
- YAML-driven parameter sweeps across portfolio, trading, event, and regime dimensions.
- Multiple optimizer modes: staged (3-phase), event-only, selective, exhaustive, YAML-config.
- Walk-forward validation with auto-generated rolling windows.
- Regime analysis for performance breakdown by market conditions.
- Result ingestion into PostgreSQL for the interactive optimizer dashboard.

**Operations & Monitoring**
- Performance analytics aggregation (win rates, expectancy, drawdown, tail loss, margin usage).
- Pipeline staleness monitoring with SendGrid email alerting (RTH-only, cooldown-gated).
- Automated data retention job (configurable purge of old chain/GEX data).
- EOD events job for economic calendar integration.
- Admin APIs to manually trigger each pipeline stage.

**Dashboard**
- React dashboard with lazy-loaded pages, error boundaries, responsive mobile sidebar.
- Pages: Overview, Portfolio, Trades, Decisions, Performance, GEX, Strategy Config, Optimizer, Admin, Auth Audit.
- Optimizer dashboard with run comparison, Pareto scatter, walk-forward charts, and paginated results.

**Testing & CI**
- Backend unit/integration/E2E suites (53 test files) with predeploy gate and CI workflow.
- Frontend test suite (Vitest + Testing Library) with page-level tests.

Still intentionally limited:
- Live broker order/fill automation remains staged (schema and paper workflow are available).

---

## How The System Works

### Live Pipeline

The pipeline runs in this order during market hours:

1. **Quote job** -- Pulls latest quotes for configured symbols. Stores in `underlying_quotes` and updates `context_snapshots`.

2. **Snapshot job** -- Gets SPX expirations from Tradier, selects by DTE policy, pulls option chains. Writes `chain_snapshots` and `option_chain_rows`. Parallel SPY/VIX streams optional.

3. **GEX jobs** -- Tradier GEX computes gamma exposure from chain data. CBOE GEX fetches precomputed data from vendor API. Both write to `gex_snapshots` tables with source discrimination.

4. **Decision + execution** -- At configured entry times the decision job constructs candidates inline from the latest chain snapshot, ranks them by `credit_to_width`, and runs portfolio gates (capital, drawdown, lot scaling). Event signal detector evaluates market conditions (SPX drops, VIX spikes, term structure) for the event overlay. Writes `trade_decisions` and opens trades. SMS notification sent on trade open.

5. **Trade PnL** -- Marks open trades continuously. Closes on take-profit, stop-loss, or expiry. SMS notification sent on trade close.

6. **EOD events** -- Captures end-of-day signals and economic calendar events.

7. **Performance analytics** -- Aggregates win rates, expectancy, drawdown, tail loss, margin usage.

8. **Staleness monitor** -- Checks pipeline freshness during RTH. Sends SendGrid alerts on stale data.

9. **Retention job** -- Daily 3 AM purge of old chain/GEX data (excludes open-trade references).

### Offline Optimizer Pipeline

The optimizer runs outside the live system for strategy research:

1. **Generate training data** -- `generate_training_data.py` exports production DB data to `training_candidates.csv` with simulated P&L trajectories.

2. **Run optimizer** -- `backtest_strategy.py --optimize --config <yaml>` sweeps all parameter combinations in parallel. Precomputes P&L columns for each TP/SL pair, then runs capital-budgeted backtests.

3. **Walk-forward validation** -- `--walkforward --wf-auto` tests top configs across rolling train/test windows to check for overfitting.

4. **Regime analysis** -- `regime_analysis.py` breaks down performance by VIX level, term structure, calendar events, etc.

5. **Ingest results** -- `ingest_optimizer_results.py` loads CSVs into `optimizer_runs/results/walkforward` tables for the dashboard.

### Current Live Configuration (C3)

The optimized C3 configuration deployed to production:

| Parameter | Value |
|-----------|-------|
| Entry times | 10:01, 11:01, 12:01, 13:01, 14:01, 15:01, 16:01 (7 hourly checks) |
| Spread width | 15 points |
| Take profit | 50% of credit received |
| Stop loss | Disabled |
| Max event trades | 2 per day |
| VIX elevated threshold | 20.0 |
| SPX 2-day drop threshold | -1.0% |
| Term inversion | Disabled (threshold 99.0) |
| Event-only mode | Yes (no scheduled trades) |

Backtest performance: Sharpe 10.78, return 215.8%, max drawdown 0.0%, 208 trades, 100% win rate. Walk-forward validated across 9 rolling windows (all PASS).

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

```
├── backend/
│   ├── spx_backend/
│   │   ├── config.py              Pydantic Settings (env-backed)
│   │   ├── main.py                Uvicorn entrypoint (PORT-aware for Railway)
│   │   ├── dte.py                 Trading-day DTE helper
│   │   ├── market_clock.py        Tradier clock cache with RTH fallback
│   │   ├── scheduler_builder.py   APScheduler construction and job registration
│   │   ├── web/
│   │   │   ├── app.py             FastAPI app, lifespan, CORS
│   │   │   └── routers/           public, admin, auth, portfolio, optimizer
│   │   ├── jobs/                  Pipeline jobs (snapshot, quote, gex, decision, trade_pnl, etc.) + offline modeling.py
│   │   ├── services/
│   │   │   ├── portfolio_manager.py   Capital budgeting, lot scaling, drawdown stops
│   │   │   ├── event_signals.py       Event-driven signal detection
│   │   │   └── sms_notifier.py        Twilio SMS trade notifications
│   │   ├── ingestion/
│   │   │   └── tradier_client.py      Tradier REST wrapper
│   │   └── database/
│   │       ├── connection.py      Async engine + session factory
│   │       ├── schema.py          Schema bootstrap + migration runner
│   │       └── sql/               DDL, 15 numbered idempotent migrations
│   ├── configs/optimizer/         YAML parameter sweep definitions (8 configs)
│   ├── scripts/                   21 offline CLI tools (see below)
│   ├── tests/                     53 test files (unit + integration)
│   ├── requirements.txt           Runtime dependencies
│   └── requirements-dev.txt       Test dependencies
├── frontend/
│   ├── src/
│   │   ├── main.tsx               Entry point with routing and auth
│   │   ├── app/                   Layout shell (AppShell, Sidebar, Navbar)
│   │   ├── pages/                 12 page components
│   │   ├── components/            Shared UI (DataTable, StatCard, Radix primitives)
│   │   ├── hooks/                 useAutoRefresh (market-hours-aware polling)
│   │   ├── api.ts                 Typed API client with auto-auth
│   │   └── contexts/              AuthContext (JWT state)
│   ├── package.json
│   └── vite.config.ts
├── data/                          CSV exports, training data, optimizer results (gitignored)
├── Dockerfile                     Backend Docker image
├── Makefile                       E2E and predeploy test targets
├── docker-compose.test.yml        Postgres test DB for integration tests
└── .env.example                   Full configuration reference
```

### Key Scripts (`backend/scripts/`)

| Script | Purpose |
|--------|---------|
| `backtest_strategy.py` | Capital-budgeted backtester with optimizer modes |
| `generate_training_data.py` | Offline training pipeline for SPX credit-spread models |
| `ingest_optimizer_results.py` | Ingest optimizer CSVs into PostgreSQL for the dashboard |
| `regime_analysis.py` | Regime-based performance analysis |
| `xgb_model.py` | XGBoost training, prediction, walk-forward validation |
| `upload_xgb_model.py` | Upload trained XGB model into `model_versions` |
| `export_production_data.py` | Export production DB tables to CSV/Parquet |
| `download_databento.py` | Download historical options from Databento |
| `generate_economic_calendar.py` | Unified economic-event calendar CSV |
| `sl_recovery_analysis.py` | Stop-loss recovery analysis and sweeps |
| `experiment_tracker.py` | Experiment tracking for optimizer runs |
| `backtest_entry.py` | Backtest framework for SPX credit spread entries |
| `data_retention.py` | CLI to export and purge old chain/GEX data |
| `clean_start.py` | One-time DB cleanup for a fresh portfolio start |
| `health_check.py` | Production health check |
| `create_allowed_users.py` | Insert allowed user with bcrypt hash |
| `set_admin.py` | Promote user to admin |
| `test_sms.py` | Twilio SMS smoke test |

### Optimizer Configs (`backend/configs/optimizer/`)

| Config | Purpose |
|--------|---------|
| `staged_stage1.yaml` | Stage 1: trading parameter sweep (TP, SL, VIX, width) |
| `staged_stage2.yaml` | Stage 2: portfolio parameter sweep |
| `staged_stage3.yaml` | Stage 3: event signal parameter sweep |
| `event_only.yaml` | Event-only mode full sweep |
| `event_only_v2.yaml` | Phase 1: term_inversion ON/OFF validation |
| `event_only_v2_explore.yaml` | Phase 2: broad exploration (signal mode, VIX, entries, width) |
| `selective.yaml` | Selective high-win-rate configuration sweep |
| `portfolio_sweep.yaml` | Portfolio-only parameter sweep |

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
- Decision (rules-based, capital-budgeted):
  - `DECISION_ENTRY_TIMES`
  - `DECISION_DTE_TARGETS`
  - `DECISION_DELTA_TARGETS`, `DECISION_PUT_DELTA_TARGETS`, `DECISION_CALL_DELTA_TARGETS`
  - `DECISION_SPREAD_WIDTH_POINTS`, `DECISION_SPREAD_SIDES`
  - `DECISION_MAX_TRADES_PER_SIDE_PER_DAY`, `DECISION_MAX_OPEN_TRADES_PER_SIDE`
- Portfolio management (always on; `decision_job` is unconditionally portfolio-managed):
  - `PORTFOLIO_STARTING_CAPITAL`
  - `PORTFOLIO_MAX_TRADES_PER_DAY`, `PORTFOLIO_MAX_TRADES_PER_RUN`
  - `PORTFOLIO_MONTHLY_DRAWDOWN_LIMIT`, `PORTFOLIO_LOT_PER_EQUITY`
  - `PORTFOLIO_CALLS_ONLY`
- Event-driven trading:
  - `EVENT_ENABLED`, `EVENT_BUDGET_MODE`, `EVENT_SIGNAL_MODE`
  - `EVENT_SPX_DROP_THRESHOLD`, `EVENT_SPX_DROP_2D_THRESHOLD`
  - `EVENT_VIX_SPIKE_THRESHOLD`, `EVENT_VIX_ELEVATED_THRESHOLD`
  - `EVENT_TERM_INVERSION_THRESHOLD`, `EVENT_SIDE_PREFERENCE`
  - `EVENT_MIN_DTE`, `EVENT_MAX_DTE`, `EVENT_MIN_DELTA`, `EVENT_MAX_DELTA`
  - `EVENT_MAX_TRADES`, `EVENT_EVENT_ONLY`
  - `EVENT_RALLY_AVOIDANCE`, `EVENT_RALLY_THRESHOLD`
- Trade PnL:
  - `TRADE_PNL_ENABLED`, `TRADE_PNL_INTERVAL_MINUTES`
  - `TRADE_PNL_TAKE_PROFIT_PCT`, `TRADE_PNL_STOP_LOSS_ENABLED`
  - `TRADE_PNL_STOP_LOSS_PCT`, `TRADE_PNL_STOP_LOSS_BASIS`
- SMS notifications:
  - `SMS_ENABLED`, `SMS_ACCOUNT_SID`, `SMS_AUTH_TOKEN`
  - `SMS_FROM_NUMBER`, `SMS_TO_NUMBER`
  - `SMS_NOTIFY_SOURCES` (comma-separated: `portfolio_scheduled`, `portfolio_event`)
- GEX:
  - `GEX_ENABLED`, `GEX_SNAPSHOT_BATCH_LIMIT`
  - `CBOE_GEX_ENABLED`, `CBOE_GEX_UNDERLYINGS`
- Performance analytics:
  - `PERFORMANCE_ANALYTICS_ENABLED`, `PERFORMANCE_ANALYTICS_INTERVAL_MINUTES`
- Data retention:
  - `RETENTION_ENABLED` (default `false`)
  - `RETENTION_DAYS` (default `60`)
  - `RETENTION_BATCH_SIZE` (default `500`)
- EOD events:
  - `EOD_EVENTS_ENABLED` (default `true`)
  - `EOD_EVENTS_HOUR`, `EOD_EVENTS_MINUTE`
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

### 3) Run Optimizer (offline)

```bash
cd backend
# Generate training data from production DB
PYTHONPATH=. python scripts/generate_training_data.py

# Run YAML-driven parameter sweep
PYTHONPATH=. python scripts/backtest_strategy.py \
  --optimize --config configs/optimizer/event_only_v2_explore.yaml \
  --output-csv ../data/optimizer_results.csv

# Walk-forward validation
PYTHONPATH=. python scripts/backtest_strategy.py \
  --walkforward --wf-auto --wf-train-months 3 --wf-test-months 1

# Ingest results into database for dashboard
PYTHONPATH=. python scripts/ingest_optimizer_results.py \
  --run-name "my-run" --mode yaml-config \
  --results-csv ../data/optimizer_results.csv
```

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
- `GET /api/performance-analytics` -- performance analytics with breakdowns
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

### Optimizer endpoints (authenticated, prefix `/api/optimizer`)

- `GET /api/optimizer/runs` -- list all optimizer runs with summary stats
- `GET /api/optimizer/runs/{run_id}` -- single run details
- `GET /api/optimizer/runs/{run_id}/results` -- paginated/sorted results for a run
- `GET /api/optimizer/runs/{run_id}/pareto` -- Pareto frontier configs
- `GET /api/optimizer/runs/{run_id}/equity-curve` -- equity curve data for a config
- `GET /api/optimizer/regime-breakdown` -- regime performance breakdown
- `GET /api/optimizer/compare` -- multi-config comparison
- `GET /api/optimizer/walkforward/{run_id}` -- walk-forward validation results

### Admin endpoints (authenticated)

- `POST /api/admin/run-snapshot`
- `POST /api/admin/run-quotes`
- `POST /api/admin/run-gex`
- `POST /api/admin/run-cboe-gex`
- `POST /api/admin/run-decision`
- `POST /api/admin/run-trade-pnl`
- `POST /api/admin/run-performance-analytics`
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
- `chain_snapshots`: one row per captured chain (metadata + checksum).
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

Optimizer:
- `optimizer_runs`: optimizer run metadata (name, mode, config count, status).
- `optimizer_results`: per-config backtest metrics with Pareto flags.
- `optimizer_walkforward`: walk-forward validation results per window.

ML/backtest scaffolding:
- `strategy_versions`, `model_versions` (preserved for offline ML re-entry; the online-ML tables `training_runs`, `feature_snapshots`, `trade_candidates`, `model_predictions` were dropped by migration 015 in Track A)
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

Backend tests (53 test files):

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
5) `POST /api/admin/run-decision`
6) `POST /api/admin/run-trade-pnl`
7) `POST /api/admin/run-performance-analytics`
8) `GET /api/admin/preflight`

If decisions are skipping:
- Check `preflight.latest.snapshot_ts` freshness.
- Check `DECISION_DTE_TARGETS` and tolerance.
- Check `DECISION_SNAPSHOT_MAX_AGE_MINUTES`.
- Inspect last `trade_decisions` reason.

`no_model_versions` warnings on a fresh DB are expected: the decision pipeline is rules-based and does not require an active model version. Upload an XGB artifact via `backend/scripts/upload_xgb_model.py` only when you actually want to start using the offline ML re-entry path.

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

- Live broker order/fill automation beyond paper trading.
- Automated optimizer scheduling (run sweeps on a cron, auto-ingest results).
- Richer regime-conditional position sizing in the live system.
- Split CI predeploy checks into faster parallel jobs.
- Add richer decision skip taxonomy views in dashboard.
