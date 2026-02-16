# SPX Tools

SPX Tools is a research and paper-execution platform for SPX index options with a practical focus on short-dated credit spread workflows.

The current stack is:
- Backend: FastAPI + APScheduler + PostgreSQL
- Frontend: React (Vite) + Mantine + Recharts
- Data source: Tradier REST APIs (expirations, chain, quotes, market clock)

This repository is designed so live capture, analytics (GEX), and decision logs share one consistent schema that can later feed backtesting and ML.

---

## Current Product Scope

What is implemented now:
- Scheduled option-chain snapshot capture.
- Scheduled quote capture (SPX, VIX, VIX9D, SPY by default).
- Scheduled GEX computation and persistence.
- Rules-based decision engine that writes TRADE/SKIP decisions.
- Admin APIs to manually trigger each pipeline stage.
- React dashboard to inspect snapshots, GEX curves, and decisions.
- Backend test suite for core business logic.

What is not implemented yet:
- Full broker order lifecycle in production usage (schema exists, execution flow is still intentionally limited).
- Full historical backfill pipeline and walk-forward orchestration.
- ML training/serving loop execution (schema scaffolding exists).

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
  - `chain_snapshots` (raw payload + checksum)
  - `option_chain_rows` (normalized per-option rows)

3) GEX job
- Reads option rows + latest eligible spot quote.
- Computes gamma exposure aggregates.
- Writes:
  - `gex_snapshots`
  - `gex_by_strike`
  - `gex_by_expiry_strike`

4) Decision job
- At configured entry times, evaluates current snapshots.
- Builds spread candidates and scores them.
- Writes one decision record per run in `trade_decisions`:
  - `TRADE` with chosen legs and params, or
  - `SKIP` with reason.

Important semantics:
- DTE handling is trading-session based (weekends and market holidays are skipped).
- GEX API data for UI is batch-scoped (same timestamp + underlying), not only one snapshot row.

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
  - `spx_backend/web/app.py`: FastAPI routes + scheduler wiring.
  - `spx_backend/jobs/`: snapshot, quote, gex, decision jobs.
  - `spx_backend/dte.py`: trading-day DTE helper logic.
  - `spx_backend/db_schema.sql`: schema bootstrap.
  - `spx_backend/db_init.py`: idempotent schema initialization.
  - `requirements.txt`: runtime dependencies.
  - `requirements-dev.txt`: test dependencies.
  - `tests/`: backend automated tests.
- `frontend/`
  - `src/DashboardApp.tsx`: top-level container.
  - `src/components/`: UI panels and widgets.
  - `src/hooks/`: data and action hooks.
  - `src/api.ts`: typed API client.

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
- Decision:
  - `DECISION_ENTRY_TIMES`
  - `DECISION_DTE_TARGETS`
  - `DECISION_DTE_TOLERANCE_DAYS`
  - `DECISION_DELTA_TARGETS`
  - `DECISION_SPREAD_WIDTH_POINTS`
  - `DECISION_SNAPSHOT_MAX_AGE_MINUTES`
- ML (feature + label pipeline):
  - `FEATURE_BUILDER_ENABLED`
  - `FEATURE_BUILDER_ALLOW_OUTSIDE_RTH`
  - `FEATURE_SCHEMA_VERSION`
  - `CANDIDATE_SCHEMA_VERSION`
  - `LABELER_ENABLED`
  - `LABELER_INTERVAL_MINUTES`
  - `LABELER_BATCH_LIMIT`
  - `LABELER_TAKE_PROFIT_PCT`
  - `LABEL_SCHEMA_VERSION`
- Live trade PnL:
  - `TRADE_PNL_ENABLED`
  - `TRADE_PNL_INTERVAL_MINUTES`
  - `TRADE_PNL_ALLOW_OUTSIDE_RTH`
  - `TRADE_PNL_MARK_MAX_AGE_MINUTES`
  - `TRADE_PNL_TAKE_PROFIT_PCT`
  - `TRADE_PNL_STOP_LOSS_PCT`
  - `TRADE_PNL_CONTRACT_MULTIPLIER`
- GEX:
  - `GEX_ENABLED=true`
  - `GEX_MAX_DTE_DAYS`
  - `GEX_STRIKE_LIMIT`
  - `GEX_SNAPSHOT_BATCH_LIMIT` (recommended >= number of expirations captured per cycle; default `20`)
- Ops:
  - `ADMIN_API_KEY`
  - `CORS_ORIGINS`
  - `ALLOW_SNAPSHOT_OUTSIDE_RTH`
  - `ALLOW_QUOTES_OUTSIDE_RTH`

Production recommendation:
- Do not include quotes in Railway variable values (use raw values, e.g. `false`, not `"false"`).

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

Admin endpoints (`X-API-Key` required if `ADMIN_API_KEY` is configured):
- `POST /api/admin/run-snapshot`
- `POST /api/admin/run-quotes`
- `POST /api/admin/run-gex`
- `POST /api/admin/run-decision`
- `POST /api/admin/run-feature-builder`
- `POST /api/admin/run-labeler`
- `POST /api/admin/run-trade-pnl`
- `DELETE /api/admin/trade-decisions/{decision_id}`
- `GET /api/admin/expirations?symbol=SPX`
- `GET /api/admin/preflight`

---

## Database Model Overview

Core ingestion:
- `chain_snapshots`: one row per captured chain payload.
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
python -m spx_backend.reset_ml_schema
```

Destructive full reset command (drops all app tables):

```bash
cd backend
python -m spx_backend.reset_all_schema
```

---

## Testing

Backend tests:

```bash
cd backend
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Current test coverage includes:
- Trading-day DTE mapping and expiration selection.
- Decision candidate construction and snapshot freshness behavior.
- GEX endpoint output modes (all / DTE / custom expirations).
- Snapshot strike window helper.
- Tradier expirations request parameter correctness.

---

## Deployment (Railway)

Backend service:
- Uses root `Dockerfile`.
- Reads `PORT` automatically.
- Requires valid `DATABASE_URL` and Tradier credentials.

Checklist:
1) Provision Railway Postgres.
2) Set backend env vars from `.env.example`.
3) Set `APP_ENV=production`.
4) Set `ADMIN_API_KEY`.
5) Set `CORS_ORIGINS` to your frontend domain.
6) Keep secrets only in Railway variables.

Frontend service:
- Deploy `frontend/` as static Vite app.
- Set `VITE_API_BASE_URL=https://<backend-domain>`.

---

## Ops Runbook

Recommended manual sequence for diagnostics:
1) `POST /api/admin/run-quotes`
2) `POST /api/admin/run-snapshot`
3) `POST /api/admin/run-gex`
4) `POST /api/admin/run-feature-builder`
5) `POST /api/admin/run-labeler`
6) `POST /api/admin/run-decision`
7) `POST /api/admin/run-trade-pnl`
8) `GET /api/admin/preflight`

If decisions are skipping:
- Check `preflight.latest.snapshot_ts` freshness.
- Check `DECISION_DTE_TARGETS` and tolerance.
- Check `DECISION_SNAPSHOT_MAX_AGE_MINUTES`.
- Inspect last `trade_decisions` reason.

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

- Add CI workflow (pytest on push/PR).
- Expand integration tests with temp Postgres fixtures.
- Add frontend test suite (Vitest + React Testing Library).
- Add explicit decision skip reason taxonomy in API/UI.
