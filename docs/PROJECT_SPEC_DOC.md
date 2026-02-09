## SPX Tools — Detailed Project Specification & Design Doc

This document expands `PROJECT_SPEC.md` into an implementation-level reference: architecture, data flows, database schema, API contracts, configuration, and an execution/backtest/ML roadmap.

**Project goal**: Build a reproducible research + paper execution platform for **SPX index options**, focusing initially on **credit vertical spreads** with **3/5/7 DTE** and entry times **10:00/11:00/12:00 ET**.

---

## Status (what exists today)

### Implemented
- **Backend** (`backend/spx_backend/`)
  - FastAPI service with:
    - DB schema initialization on startup (statement-by-statement)
    - Snapshot scheduler (options chains)
    - Quote-only scheduler (SPX/VIX/VIX9D/SPY)
    - Snapshot ingestion from Tradier option chain REST API
    - Underlying quote capture from Tradier `get quotes`
    - Minimal API to list stored snapshots
    - Admin endpoints to force a snapshot and inspect expirations
- **Frontend** (`frontend/`)
  - React (Vite) dashboard to:
    - View latest snapshots
    - Trigger “Run snapshot now” and display the result

### Not implemented yet (planned)
- Strategy selection for vertical spreads (delta targeting for live)
- Tradier sandbox multi-leg order placement + fill ingestion into `orders`/`fills`
- Trades normalization (`trades`, `trade_legs`) and PnL tracking
- ML model training pipeline (meta-labeler)

### Partially implemented
- **Backtest engine** (Databento OPRA.PILLAR SPX CBBO-1m)
  - Implemented baseline backtest (minute-by-minute, strike-distance spread selection, TP/SL exits)
  - Delta-based selection and IV/Greeks are not implemented yet
- **Decision engine (rules-only)**
  - Scheduled at 10/11/12 ET
  - Writes `trade_decisions` (no order placement yet)

---

## Glossary
- **SPX**: S&P 500 index options (cash settled, index options mechanics)
- **DTE**: Days to expiration (calendar days; MVP uses calendar-day distance)
- **Credit vertical**: Sell-to-open one option and buy-to-open another farther OTM at same expiration; net credit received.
- **NBBO**: National Best Bid and Offer
- **Snapshot**: A stored, timestamped option chain response (raw JSON payload + checksum)
- **RTH**: Regular trading hours (MVP definition: Mon–Fri 09:30–16:00 ET)
- **Paper trading**: Using Tradier sandbox to place orders and receive simulated fills

---

## System architecture

### High-level components

```text
                 ┌──────────────────────────────────────────┐
                 │                  Frontend                │
                 │        React (Vite) dashboard             │
                 │  - snapshot list                          │
                 │  - run snapshot now (admin)               │
                 └───────────────────────┬───────────────────┘
                                         │ HTTP (CORS / proxy)
                                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                                  Backend                                 │
│                              FastAPI (Uvicorn)                           │
│                                                                          │
│  Lifespan startup:                                                       │
│   - init_db() → executes db_schema.sql                                   │
│   - start APScheduler interval job                                       │
│                                                                          │
│  Jobs:                                                                   │
│   - snapshot_job.run_once()                                              │
│                                                                          │
│  APIs:                                                                   │
│   - /api/chain-snapshots                                                 │
│   - /api/admin/run-snapshot                                              │
│   - /api/admin/expirations                                               │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │        Postgres         │
                    │  chain_snapshots, ...   │
                    └────────────────────────┘

                                ▲
                                │ HTTPS (REST)
                                │
                    ┌────────────────────────┐
                    │         Tradier         │
                    │  option expirations     │
                    │  option chain + greeks  │
                    └────────────────────────┘
```

### Design constraints
- **No full-chain streaming**: We do not attempt to stream every SPX option symbol. Instead, we store periodic chain snapshots (scalable, reproducible).
- **Reproducibility first**: Each decision should reference the exact snapshot + checksum used.
- **Live/backtest parity**: Trade construction and risk logic should be shared between live and backtest.

---

## Repo layout and responsibilities

```text
spx_tools/
  backend/
    requirements.txt
    spx_backend/
      main.py                # Uvicorn entrypoint
      config.py              # env + defaults
      db.py                  # async SQLAlchemy engine/session
      db_init.py             # executes db_schema.sql on startup
      db_schema.sql          # MVP schema
      backtest/
        engine.py            # minimal backtest engine (DuckDB + Parquet)
        run_backtest.py      # CLI runner example
      ingestion/
        tradier_client.py    # REST client (expirations/chains/quotes)
      jobs/
        snapshot_job.py      # snapshot ingestion job
      web/
        app.py               # FastAPI app + endpoints + scheduler wiring
  frontend/
    package.json
    vite.config.ts           # proxies /api to backend
    src/
      api.ts                 # backend API client
      App.tsx                # dashboard UI
      main.tsx               # React entrypoint
  docs/
    PROJECT_SPEC_DOC.md      # this document
  PROJECT_SPEC.md
  README.md
  Dockerfile
  .env.example
  .gitignore
```

---

## Configuration (env vars)

Backend configuration is loaded via `pydantic-settings` from `.env` (repo root) or `../.env` when running from `backend/`.

### Required
- **`DATABASE_URL`**
  - Format: `postgresql+asyncpg://USER:PASSWORD@HOST:PORT/DBNAME`
  - Used by SQLAlchemy async engine.
- **`TRADIER_ACCESS_TOKEN`**
  - Tradier bearer token.
- **`TRADIER_ACCOUNT_ID`**
  - Paper account id (used later for order placement).

### Recommended
- **`TRADIER_BASE_URL`**
  - Default: `https://sandbox.tradier.com/v1`
  - Use sandbox for paper trading.
- **`TZ`**
  - Default: `America/New_York`

### Snapshot controls
- **`SNAPSHOT_INTERVAL_MINUTES`**
  - Default: `5`
  - Scheduler interval.
- **`SNAPSHOT_UNDERLYING`**
  - Default: `SPX`
- **`SNAPSHOT_DTE_TARGETS`**
  - Default: `3,5,7`
- **`SNAPSHOT_DTE_MODE`**
  - Default: `range` (use `targets` to honor `SNAPSHOT_DTE_TARGETS`)
- **`SNAPSHOT_DTE_MIN_DAYS`**
  - Default: `0`
- **`SNAPSHOT_DTE_MAX_DAYS`**
  - Default: `10`
- **`SNAPSHOT_DTE_TOLERANCE_DAYS`**
  - Default: `1`
  - Used when selecting an expiration for a target DTE.
- **`SNAPSHOT_STRIKES_EACH_SIDE`**
  - Default: `100` (store strikes near spot; 100 below, 100 above)
- **`QUOTE_SYMBOLS`**
  - Default: `SPX,VIX,VIX9D,SPY`
  - Symbols fetched via Tradier `get quotes` each snapshot.
- **`QUOTE_INTERVAL_MINUTES`**
  - Default: `5`
  - Quote-only scheduler interval.
- **`GEX_ENABLED`**
  - Default: `true`
- **`GEX_INTERVAL_MINUTES`**
  - Default: `5`
- **`GEX_STORE_BY_EXPIRY`**
  - Default: `true`
- **`GEX_SPOT_MAX_AGE_SECONDS`**
  - Default: `600`
- **`GEX_CONTRACT_MULTIPLIER`**
  - Default: `100`
- **`GEX_PUTS_NEGATIVE`**
  - Default: `true`
- **`GEX_SNAPSHOT_BATCH_LIMIT`**
  - Default: `5`
- **`GEX_STRIKE_LIMIT`**
  - Default: `150`
- **`GEX_MAX_DTE_DAYS`**
  - Default: `10`

### Dev/ops controls
- **`ALLOW_SNAPSHOT_OUTSIDE_RTH`**
  - Default: `false`
  - If false, scheduled snapshots only run during RTH.
  - Manual “force run” ignores RTH regardless.
- **`ALLOW_QUOTES_OUTSIDE_RTH`**
  - Default: `false`
  - Controls quote-only scheduler gating.
- **`ADMIN_API_KEY`**
  - Default: unset
  - If set, admin endpoints require `X-API-Key` header.
- **`MARKET_CLOCK_CACHE_SECONDS`**
  - Default: `300`
  - Cache TTL for Tradier market clock responses.
- **`CORS_ORIGINS`**
  - Default: `http://localhost:5173`
  - Comma-separated list of allowed origins.

---

## Scheduling and time model

### Snapshot cadence
- Default schedule: every **5 minutes**
- Stored timestamp:
  - `now_et` in response payload (for operator visibility)
  - `ts` stored in DB as UTC `TIMESTAMPTZ`

### Quote cadence
- Default schedule: every **5 minutes**
- Uses Tradier market clock (cached) to skip closed markets

### RTH definition (MVP)
- Monday–Friday
- 09:30–16:00 America/New_York

**Note**: The scheduler now uses Tradier’s market clock for holiday-aware open/close checks and falls back to RTH logic if the clock call fails.

---

## Data ingestion: Tradier snapshots

### Inputs
1) **Option expirations** for underlying `SPX`
2) **Option chain** for selected expiration(s) including Greeks/IV (Tradier-provided)

### Expiration selection logic
For each configured `target_dte`:
1) Compute `target_date = today + target_dte`
2) Find expirations within ±`SNAPSHOT_DTE_TOLERANCE_DAYS`
3) Choose closest by absolute day difference

**Range mode**: when `SNAPSHOT_DTE_MODE=range`, the job selects **all expirations** whose DTE is within `SNAPSHOT_DTE_MIN_DAYS..SNAPSHOT_DTE_MAX_DAYS` (inclusive).

### Strike filtering (storage optimization)
To reduce storage, the job can filter `option_chain_rows` to strikes near spot:
- Keep `SNAPSHOT_STRIKES_EACH_SIDE` below and above spot (e.g., 100/100)
- Raw `chain_snapshots` are still stored for reproducibility

#### Force-mode fallback (for testing)
Manual runs call `run_once(force=True)`.
If no expiration is within tolerance, the job chooses the **closest available expiration** and records:
- `actual_dte_days` (the real distance to expiration)

This is specifically to make end-to-end testing possible outside market hours (when the sandbox expiration list may not match near-term DTEs).

---

## Database schema (Postgres)

The backend runs `backend/spx_backend/db_schema.sql` at startup and creates tables if missing.

### `option_instruments`
Stores normalized option metadata when/if we start persisting instrument definitions separately.

Columns:
- `option_symbol` (PK): vendor symbol
- `root`: SPX
- `expiration`: date
- `strike`: float
- `option_right`: `C` or `P`
- `style`: optional
- `created_at`

### `chain_snapshots` (core table today)
Stores the raw Tradier chain response plus checksum for reproducibility.

Columns:
- `snapshot_id` (PK, bigserial)
- `ts` (TIMESTAMPTZ): snapshot time (stored in UTC)
- `underlying` (TEXT): e.g. `SPX`
- `target_dte` (INT): configured bucket (3/5/7)
- `expiration` (DATE): expiration used for this snapshot
- `payload_json` (JSONB): raw Tradier response
- `checksum` (TEXT): SHA-256 hash of the JSON (stable serialization)

### `option_chain_rows`
Stores per-option fields extracted from chain snapshots, including **open interest** and Greeks (when available).

Indexes:
- `idx_chain_snapshots_ts` on `(ts desc)`
- `idx_chain_snapshots_exp` on `(expiration, ts desc)`

### `context_snapshots` (planned)
Stores derived context signals aligned to timestamps (SPX/ SPY price, VIX9D, term structure, GEX approximation, etc.).

### `underlying_quotes`
Stores raw quote snapshots for underlying indices/ETFs (SPX, VIX, VIX9D, SPY).

### `market_clock_audit`
Stores raw Tradier market clock responses for holiday/audit visibility.

### `gex_snapshots` / `gex_by_strike` / `gex_by_expiry_strike`
Stores GEX aggregates computed from option_chain_rows using open_interest × gamma × spot².

### ML + strategy versioning (new)
- `strategy_versions`
  - Versioned strategy definitions + parameters (reproducible live/backtest)
- `model_versions`
  - ML model registry (feature spec, training windows, metrics, artifact URI)
- `training_runs`
  - Training job history + metrics for auditability
- `feature_snapshots`
  - Decision-time features used for ML (with optional labels)
- `trade_candidates`
  - Candidate legs/params scored by rules/ML
- `model_predictions`
  - Model scores per candidate (traceable by model version)
- `strategy_recommendations`
  - Proposed parameter changes + approval state
- `backtest_runs`
  - Backtest run metadata + config + aggregate stats

### `trade_decisions` (planned)
Stores discrete decision events (TRADE/SKIP), with references to snapshots.
Includes optional `strategy_params_json` to support future strategies beyond delta-based verticals.

### `orders` + `fills` (planned)
Stores broker orders and execution fills (paper sandbox initially).

### `trades` + `trade_legs`
Normalized multi-leg trades (supports verticals, condors, butterflies).

---

## Backend API contracts

### Public
- `GET /health`
  - Returns `{ "status": "ok" }`

- `GET /api/chain-snapshots?limit=50`
  - Response shape:
    - `items[]`: `{ snapshot_id, ts, underlying, target_dte, expiration, checksum }`

### Admin
Admin endpoints are optionally protected by `ADMIN_API_KEY` via `X-API-Key` header.

- `POST /api/admin/run-snapshot`
  - Forces a snapshot run immediately.
  - Returns:
    - `skipped` (bool)
    - `reason` (string | null)
    - `now_et` (string)
    - `inserted[]` array:
      - `target_dte`
      - `expiration`
      - `actual_dte_days`
      - `checksum`

- `POST /api/admin/run-quotes`
  - Forces a quote-only run (SPX/VIX/VIX9D/SPY).
  - Returns:
    - `skipped`, `reason`, `now_et`, `quotes_inserted`

- `POST /api/admin/run-gex`
  - Forces a GEX computation run.
  - Returns:
    - `skipped`, `reason`, `computed_snapshots`

- `GET /api/admin/expirations?symbol=SPX`
  - Returns a list of expirations from Tradier for debugging DTE selection.

---

## Frontend behavior (React)

### Data sources
- Reads snapshots via `GET /api/chain-snapshots`
- Can trigger snapshots via `POST /api/admin/run-snapshot`

### Local development proxy
`frontend/vite.config.ts` proxies:
- `/api` → `http://localhost:8000`
- `/health` → `http://localhost:8000`

This avoids CORS issues during local development, even though the backend also enables CORS.

---

## Trading strategy spec (MVP target)

### Instruments
- Underlying: **SPX options**
- Strategy: **credit vertical spreads**
- DTE targets: **3, 5, 7**
- Entry times: **10:00, 11:00, 12:00 ET**
- Short delta targets: **0.10** and **0.20** (absolute delta)

### Position management (close-only)
- Take profit at **50% of max profit**
- Stop loss at **-100% of max profit**
- No rolling in v1

### Risk constraints (MVP defaults)
- Max open trades: 1
- Max new trades/day: 1
- Daily stop after one stop-loss event

---

## Pricing and fills

### Live (paper) execution truth
When we add order placement:
- Place multi-leg orders in Tradier sandbox (even for 2 legs for uniformity).
- Store:
  - request payload
  - broker response
  - order status lifecycle
  - fills (including partial fills)

### Backtest fill model (initial)
Using NBBO mid with conservative slippage:
- Credit entry fill: `mid - 0.30 * spread`
- Credit exit fill: `mid + 0.30 * spread`
- No partial fills in v1 backtest

---

## Backtesting (Databento)

### Dataset selection
- Databento dataset: `OPRA.PILLAR`, filtered to **SPX options**
- Schema: **CBBO-1m**
- Also include instrument definitions; optionally trades later.

### Backtest engine behavior (current)
- Replay minute-by-minute using CBBO-1m
- **Construct spreads by strike distance** (not delta) using spot price
- Evaluate PnL and management rules (TP/SL) identical to live
- Outputs a list of trades (future: persist into Postgres)

### Backtest engine goals (next iteration)
- **As-of quote joins** (use last known quote <= timestamp, no exact-match gaps)
- **Cash-settled expiration handling** for SPX (settlement rules at expiration)
- **Fees + slippage model** (commissions, contract fees, realistic spread impact)
- **Leakage guardrails** (features strictly from data available at decision time)
- **Walk-forward evaluation** (rolling train/validate/out-of-sample windows)
- Delta-based selection (requires IV/Greeks)
- Optional use of vendor greeks if available, or local IV calculation
- Persist backtest trades into Postgres for UI/ML parity

---

## ML (phase 2)

### Target
Increase **win rate** for the defined management rules.

### Labeling (recommended)
For each trade decision:
- `win = 1` if trade hits **+50% max profit before stop**
- `win = 0` otherwise

### Feature store
Record at decision time:
- DTE bucket, delta target, spread width, credit, bid/ask spread metrics
- Market context features (VIX, VIX9D, term structure, GEX approximation)
- Calendar features (days to OPEX, event blackout flags)

### Model lifecycle
- Start with rules-only gating
- Add XGBoost classifier once consistent data collection exists (100s–1000s of labeled trades)
- Train with **walk-forward splits** and reserve true out-of-sample windows
- Use ML to **rank candidates**, not to bypass risk controls
- Require a **policy gate** before any strategy change is activated

### Decision flow (recommended)
1) Generate candidates → store `trade_candidates`
2) Build features → store `feature_snapshots`
3) Score candidates → store `model_predictions`
4) Choose decision → store `trade_decisions`
5) Evaluate outcomes → update labels/metrics

---

## Deployment (Railway)

### Backend
- Uses `Dockerfile` at repo root.
- Container runs:
  - `python -m spx_backend.main`
  - Listens on port 8000
- Requires env vars in Railway service settings.

### Frontend
Recommended: deploy separately as a static site (or later serve built assets from backend).

---

## Implementation roadmap (suggested)

### Milestone 1 — Snapshot reliability (done + polish)
- Snapshot scheduler + forced snapshot endpoint
- Expiration visibility endpoint
- React dashboard controls and feedback

### Milestone 2 — Decision engine
- Implement entry windows (10/11/12 ET) and write `trade_decisions`
- Add “active trade exists” and “max trades/day” rules

### Milestone 3 — Paper execution
- Tradier sandbox multi-leg order placement for 2-leg spreads
- Poll and ingest fills into `orders`/`fills`

### Milestone 4 — Trades + PnL
- Normalize `trades` + `trade_legs`
- Mark-to-market snapshots and TP/SL closes

### Milestone 5 — Backtesting (baseline done)
- Databento download tooling + DuckDB loader (partial)
- Baseline backtest engine implemented (strike-distance selection)
- Remaining: delta-based selection + DB persistence

### Milestone 6 — ML
- Feature/label pipeline
- XGBoost baseline and monitoring

