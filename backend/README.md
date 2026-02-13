# SPX Tools Backend

This backend is a FastAPI service that captures SPX options market data, computes GEX context, and produces rules-based trade decisions for paper workflow analysis.

It is built for observability and reproducibility:
- every chain snapshot is stored raw + normalized
- market clock states are audited
- decision runs are persisted (TRADE and SKIP) with reasons
- preflight endpoint provides one-call pipeline health

---

## 1) Service Responsibilities

The backend performs four continuous tasks:
- Quote ingestion (`underlying_quotes`, `context_snapshots`)
- Option chain snapshots (`chain_snapshots`, `option_chain_rows`)
- GEX computation (`gex_snapshots`, strike/expiry detail tables)
- Decision generation (`trade_decisions`)

And exposes APIs for:
- dashboard data reads
- admin run controls
- quick health and diagnostics

---

## 2) Runtime Lifecycle

Startup flow:
1) Load settings from env (`spx_backend/config.py`).
2) Initialize database schema from `spx_backend/db_schema.sql`.
3) Build Tradier client + market clock cache.
4) Start APScheduler jobs for quote/snapshot/gex/decision.
5) Optionally run immediate first cycles to warm data.

Shutdown flow:
- Scheduler stops with FastAPI lifespan shutdown.

Entrypoint:
- `python -m spx_backend.main`
- Honors Railway `PORT` automatically.

---

## 3) Core Modules

- `spx_backend/web/app.py`
  - FastAPI app, route handlers, scheduler wiring, auth guard.
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
- `spx_backend/dte.py`
  - Trading-session DTE lookup and expiration chooser helpers.
- `spx_backend/db_schema.sql`
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
2) Build candidate vertical spread(s) using delta and configured width.
3) Compute candidate credit and validate viability.
4) Apply context score adjustments.
5) Enforce guardrails (`DECISION_MAX_TRADES_PER_DAY`, `DECISION_MAX_OPEN_TRADES`).
6) Insert one `trade_decisions` row with final action.

Why SKIP rows are stored:
- They provide full auditability for why no trade occurred.
- They are required to evaluate decision quality over time.

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

Batch-scoped GEX note:
- DTE/expiration/curve endpoints use all snapshots in the same capture batch (`same ts + underlying`), not only one row.

### Admin Endpoints

If `ADMIN_API_KEY` is set, include header:
- `X-API-Key: <key>`

Routes:
- `POST /api/admin/run-snapshot`
- `POST /api/admin/run-quotes`
- `POST /api/admin/run-gex`
- `POST /api/admin/run-decision`
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
  - `SNAPSHOT_DTE_MODE`
  - `SNAPSHOT_DTE_MIN_DAYS`, `SNAPSHOT_DTE_MAX_DAYS`
  - `SNAPSHOT_DTE_TARGETS`, `SNAPSHOT_DTE_TOLERANCE_DAYS`
  - `SNAPSHOT_STRIKES_EACH_SIDE`
- Decision:
  - `DECISION_ENTRY_TIMES`
  - `DECISION_DTE_TARGETS`, `DECISION_DTE_TOLERANCE_DAYS`
  - `DECISION_DELTA_TARGETS`
  - `DECISION_SPREAD_SIDE`, `DECISION_SPREAD_WIDTH_POINTS`
  - `DECISION_SNAPSHOT_MAX_AGE_MINUTES`
  - `DECISION_MAX_TRADES_PER_DAY`, `DECISION_MAX_OPEN_TRADES`
- GEX:
  - `GEX_ENABLED`, `GEX_INTERVAL_MINUTES`
  - `GEX_SNAPSHOT_BATCH_LIMIT` (default `20`)
  - `GEX_STRIKE_LIMIT` (default `150`)
  - `GEX_MAX_DTE_DAYS` (default `10`)
  - `GEX_SPOT_MAX_AGE_SECONDS`
- Ops/Safety:
  - `ALLOW_SNAPSHOT_OUTSIDE_RTH`
  - `ALLOW_QUOTES_OUTSIDE_RTH`
  - `MARKET_CLOCK_CACHE_SECONDS`
  - `ADMIN_API_KEY`
  - `CORS_ORIGINS`
  - `TZ`

---

## 8) Database Schema Guide

Schema file:
- `spx_backend/db_schema.sql`

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
curl -X POST http://localhost:8000/api/admin/run-decision
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

Current coverage domains:
- DTE helper behavior
- Tradier expiration request params
- Snapshot strike-selection helper
- Decision candidate/scoring/freshness logic
- GEX zero-gamma helper
- GEX API output behavior (including custom expiration filter and fallback)

---

## 11) Railway Deployment Notes

Required:
- `APP_ENV=production`
- valid `DATABASE_URL`
- `ADMIN_API_KEY` enabled
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

- Decision engine is rules-only today (no live ML inference).
- Order placement/fill lifecycle tables exist but workflow remains staged.
- Frontend test coverage has not yet been added.
- Full backtest runner orchestration is still pending.
