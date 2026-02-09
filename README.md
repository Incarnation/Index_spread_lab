## SPX Tools

A research + paper execution platform for **SPX index options** with a focus on **0–14 DTE credit spreads**.

Current MVP capabilities:
- **Backend**: FastAPI service that initializes a Postgres schema on startup and runs a scheduled job to snapshot the SPX option chain from Tradier.
- **Frontend**: React (Vite) dashboard that reads from the backend API.

Planned next steps (per `PROJECT_SPEC.md`):
- Decision engine (rules-only) is wired; next is broker order placement and fills.
- Backtesting pipeline (Databento `OPRA.PILLAR` SPX, `CBBO-1m`) with live/backtest parity.
- ML pipeline: feature snapshots, model versioning, and walk-forward evaluation.

---

## Repo layout
- `backend/`
  - `spx_backend/`: FastAPI app + scheduler + DB code
  - `requirements.txt`: backend Python dependencies
- `frontend/`
  - Vite + React dashboard

---

## Configuration

Create a `.env` in the repo root (copy `.env.example`) and fill in:
- **`DATABASE_URL`**: Postgres connection string for SQLAlchemy async
  - format: `postgresql+asyncpg://USER:PASSWORD@HOST:PORT/DBNAME`
- **`TRADIER_BASE_URL`**: default `https://sandbox.tradier.com/v1` (paper)
- **`TRADIER_ACCESS_TOKEN`**: your Tradier access token
- **`TRADIER_ACCOUNT_ID`**: your paper account id (e.g. `VAxxxxxx`)
- Optional:
  - `SNAPSHOT_INTERVAL_MINUTES` (default 5)
  - `SNAPSHOT_UNDERLYING` (default SPX)
  - `SNAPSHOT_DTE_TARGETS` (default `3,5,7`)
  - `SNAPSHOT_DTE_MODE` (default `range`, use `targets` for list mode)
  - `SNAPSHOT_DTE_MIN_DAYS` (default `0`)
  - `SNAPSHOT_DTE_MAX_DAYS` (default `10`)
  - `SNAPSHOT_DTE_TOLERANCE_DAYS` (default `1`)
  - `SNAPSHOT_STRIKES_EACH_SIDE` (default `100`)
  - `QUOTE_SYMBOLS` (default `SPX,VIX,VIX9D,SPY`)
  - `QUOTE_INTERVAL_MINUTES` (default `5`)
  - `DECISION_ENTRY_TIMES` (default `10:00,11:00,12:00`)
  - `DECISION_DTE_TARGETS` (default `3,5,7`)
  - `DECISION_DTE_TOLERANCE_DAYS` (default `1`)
  - `DECISION_DELTA_TARGETS` (default `0.10,0.20`)
  - `DECISION_SPREAD_SIDE` (default `put`)
  - `DECISION_SPREAD_WIDTH_POINTS` (default `25`)
  - `DECISION_CONTRACTS` (default `1`)
  - `DECISION_SNAPSHOT_MAX_AGE_MINUTES` (default `15`)
  - `DECISION_MAX_TRADES_PER_DAY` (default `1`)
  - `DECISION_MAX_OPEN_TRADES` (default `1`)
  - `DECISION_RULESET_VERSION` (default `rules_v1`)
  - `DECISION_ALLOW_OUTSIDE_RTH` (default `false`)
  - `GEX_ENABLED` (default `true`)
  - `GEX_INTERVAL_MINUTES` (default `5`)
  - `GEX_STORE_BY_EXPIRY` (default `true`)
  - `GEX_SPOT_MAX_AGE_SECONDS` (default `600`)
  - `GEX_CONTRACT_MULTIPLIER` (default `100`)
  - `GEX_PUTS_NEGATIVE` (default `true`)
  - `GEX_SNAPSHOT_BATCH_LIMIT` (default `5`)
  - `GEX_STRIKE_LIMIT` (default `150`)
  - `GEX_MAX_DTE_DAYS` (default `10`)
  - `ALLOW_SNAPSHOT_OUTSIDE_RTH` (default `false`)
  - `ALLOW_QUOTES_OUTSIDE_RTH` (default `false`)
  - `MARKET_CLOCK_CACHE_SECONDS` (default `300`)
  - `ADMIN_API_KEY` (optional admin auth)
  - `CORS_ORIGINS` (default `http://localhost:5173`)

---

## Database setup

### Recommended: Railway Postgres (cloud)
Use Railway Postgres for anything you want running 24/7 (snapshots, paper orders, dashboard history).

1) Create/add a Postgres service in Railway.
2) Copy the Postgres connection string from Railway.
3) Set `.env`:
- If Railway gives you `postgresql://...`, convert it to:
  - `postgresql+asyncpg://...`

The backend auto-creates required tables on startup from:
- `backend/spx_backend/db_schema.sql`

### Alternative: local Postgres (development)
If you prefer local DB for dev, you can run Postgres via Docker:

```bash
docker run --name spx-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=spx_tools -p 5432:5432 -d postgres:16
```

Then set:
- `DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/spx_tools`

---

## Running locally

### Backend (FastAPI + scheduler)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

cd backend
python -m spx_backend.main
```

Backend endpoints:
- `GET http://localhost:8000/health`
- `GET http://localhost:8000/api/chain-snapshots?limit=50`

Notes:
- The scheduler runs two jobs:
  - **Snapshot job**: stores option chain snapshots.
  - **Quote job**: stores SPX/VIX/VIX9D/SPY quotes every `QUOTE_INTERVAL_MINUTES`.
- Market open/close checks use Tradier market clock with a short cache to reduce calls.
- Market clock responses are stored in `market_clock_audit` for audit/debug.

### Frontend (React)
In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:
- `http://localhost:5173/`

The Vite dev server proxies `/api/*` to the backend on `http://localhost:8000`.

---

## Deployment (Railway)

### Backend
This repo includes a `Dockerfile` that starts the backend. Typical Railway setup:
- Create a service from this repo
- Set environment variables (same as `.env`)
- Ensure a Postgres plugin/service is attached and `DATABASE_URL` is set appropriately

### Frontend
Two options:
- **Separate service**: deploy `frontend/` as a static site (recommended when you want CDN/static hosting).
- **Single service**: later we can configure the backend to serve the built React assets (good for “one container” simplicity).

---

## Troubleshooting

### “No snapshots yet”
- The scheduler skips outside RTH.
- Also verify Tradier token/permissions and that the underlying symbol (`SPX`) is supported in your account.

### Postgres schema errors on boot
- The schema is executed statement-by-statement in `backend/spx_backend/db_init.py`.
- If you manually created tables earlier, you may need one-time migrations (we’ll add Alembic once the schema stabilizes).

### Railway DB SSL issues
If you see SSL errors connecting to Railway Postgres, paste the error message and we’ll add the correct asyncpg SSL settings.


