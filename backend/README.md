## SPX Tools Backend

FastAPI backend for the SPX options platform. This service:
- Initializes the Postgres schema on startup
- Runs the Tradier snapshot scheduler (options chains)
- Runs a separate quote-only scheduler (SPX/VIX/SPY/VIX9D)
- Exposes APIs for snapshots and admin controls

---

## How it works (high level)

1) **Snapshot scheduler (options chains)**
   - Every `SNAPSHOT_INTERVAL_MINUTES`, the job requests SPX option expirations,
     selects expirations based on DTE mode (range or targets), and pulls option chains.
   - Each response is stored as a row in `chain_snapshots` (raw JSON + checksum).

2) **Quote worker (underlying/context)**
   - Every `QUOTE_INTERVAL_MINUTES`, fetches quotes for `QUOTE_SYMBOLS`
     (default: `SPX,VIX,VIX9D,SPY`).
   - Quotes are stored in `underlying_quotes`.
   - A `context_snapshots` row is updated with `spx_price`, `spy_price`, `vix`, `vix9d`,
     and `term_structure`.

3) **Holiday‑aware scheduling**
   - Scheduler checks Tradier market clock and skips when the market is closed.
   - Falls back to simple RTH logic if the market clock call fails.

4) **Decision engine (rules‑based)**
   - At `DECISION_ENTRY_TIMES`, evaluates the latest snapshots to decide TRADE/SKIP.
   - Writes decisions into `trade_decisions` (no order placement yet).

---

## Configuration (.env)

Backend reads from `.env` in repo root (or `../.env` when running from `backend/`):

Required:
- `DATABASE_URL`  
  Example: `postgresql+asyncpg://USER:PASSWORD@HOST:PORT/DBNAME`
- `TRADIER_ACCESS_TOKEN`
- `TRADIER_ACCOUNT_ID`

Optional:
- `TRADIER_BASE_URL` (default `https://sandbox.tradier.com/v1`)
- `SNAPSHOT_INTERVAL_MINUTES` (default `5`)
- `SNAPSHOT_UNDERLYING` (default `SPX`)
- `SNAPSHOT_DTE_TARGETS` (default `3,5,7`)
- `SNAPSHOT_DTE_MODE` (default `range`, use `targets` for list mode)
- `SNAPSHOT_DTE_MIN_DAYS` (default `0`)
- `SNAPSHOT_DTE_MAX_DAYS` (default `10`)
- `SNAPSHOT_RANGE_FALLBACK_ENABLED` (default `false`)
- `SNAPSHOT_RANGE_FALLBACK_COUNT` (default `3`)
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
- `ADMIN_API_KEY` (if set, admin endpoints require `X-API-Key`)
- `CORS_ORIGINS` (default `http://localhost:5173`)
- `TZ` (default `America/New_York`)

---

## Run locally

From repo root:

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

---

## Trigger a snapshot manually (any time)

If `ADMIN_API_KEY` is **not** set:

```bash
curl -X POST http://localhost:8000/api/admin/run-snapshot
```

If `ADMIN_API_KEY` **is** set:

```bash
curl -X POST http://localhost:8000/api/admin/run-snapshot \
  -H "X-API-Key: your_key_here"
```

## Trigger quotes manually (any time)

```bash
curl -X POST http://localhost:8000/api/admin/run-quotes
```

## Trigger GEX computation manually

```bash
curl -X POST http://localhost:8000/api/admin/run-gex
```

## Trigger decision engine manually

```bash
curl -X POST http://localhost:8000/api/admin/run-decision
```

To see which expirations Tradier is returning:

```bash
curl "http://localhost:8000/api/admin/expirations?symbol=SPX"
```

## Preflight health summary

```bash
curl "http://localhost:8000/api/admin/preflight"
```

Returns one-call pipeline health: counts and latest timestamps for quotes, snapshots, GEX, and decisions.

---

## Railway deployment checklist

- Set `APP_ENV=production`
- Set `ADMIN_API_KEY` to protect admin endpoints
- Set `CORS_ORIGINS` to your deployed frontend URL
- Choose `TRADIER_BASE_URL` intentionally:
  - sandbox: `https://sandbox.tradier.com/v1`
  - live: `https://api.tradier.com/v1`
- Use strict range mode in production:
  - `SNAPSHOT_RANGE_FALLBACK_ENABLED=false`

The backend process reads Railway `PORT` automatically.

---

## Database schema (MVP)

Executed on startup from `spx_backend/db_schema.sql`.

Core tables:
- `chain_snapshots`: raw Tradier chain payloads + checksum
- `option_chain_rows`: per-option rows extracted from chain snapshots (incl. greeks + OI when available)
- `underlying_quotes`: SPX/VIX/SPY quotes captured by quote worker
- `context_snapshots`: context features (VIX, term structure, GEX placeholders)
- `market_clock_audit`: cached market clock responses for holiday audit
- `gex_snapshots` / `gex_by_strike` / `gex_by_expiry_strike`: GEX aggregates
- `trade_decisions`: future decision events (TRADE / SKIP)
- `orders` / `fills`: future paper broker order lifecycle
- `trades` / `trade_legs`: normalized multi‑leg trade records (verticals/condors/butterflies)

