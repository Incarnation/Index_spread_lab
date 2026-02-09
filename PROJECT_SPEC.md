## Project Spec: SPX 0–14 DTE Options Trading Platform (MVP: 3/5/7 DTE Credit Verticals)

### Executive summary
- **Goal**: Build a reproducible research + paper execution platform for **SPX index options** (not SPY), focusing on **credit vertical spreads** with 0–14 DTE (MVP uses 3/5/7 DTE).
- **Primary outputs**:
  - A live system that **snapshots SPX option chains every 5 minutes**, computes context + signals, and can **place paper multi-leg orders via Tradier sandbox**.
  - A backtester that uses **the same trade definitions, risk rules, and pricing conventions** as live.
  - A dataset suitable for iterative ML improvements (stored features, decisions, fills, and outcomes).

### What we are (and are not) building in v1
- **In scope (v1 / MVP)**:
  - SPX **credit vertical spreads only** (2 legs).
  - DTE targets: **3 DTE, 5 DTE, 7 DTE**.
  - Entry times: **10:00, 11:00, 12:00 America/New_York** (no other entries).
  - Delta targeting: **short-leg absolute delta targets 0.10 and 0.20** (two candidate setups).
  - Management: **close-only** with:
    - **Take profit**: close at **50% of max profit**.
    - **Stop loss**: close at **-100% of max profit** (i.e., loss equals collected credit).
  - Fixed position sizing (fixed contracts).
  - Minimal web dashboard for monitoring.
- **Out of scope (v1)**:
  - Iron condors / iron butterflies.
  - Rolling rules.
  - Tick-level replay.
  - Full-chain WebSocket streaming of all SPX options symbols.

### Key feasibility constraints (design choices to avoid failure modes)
- **Live chain capture**: Do not attempt to subscribe to the entire SPX chain via streaming. Instead, **pull REST snapshots every 5 minutes** and store them.
- **Paper fills**: Prefer **Tradier sandbox order placement** so fills/partials/rejects are broker-derived. Maintain a fallback fill simulator for offline backtests.
- **Reproducibility**: Store the **exact chain snapshot and context used** to make each decision.

---

## Tech stack
- **Language**: Python 3.10+
- **Live host**: Railway (Docker)
- **Live DB**: PostgreSQL (Railway)
- **Backtest DB**: DuckDB (local)
- **ML**: XGBoost (optional in MVP; allow rule-based gating first)
- **Web UI**: FastAPI + simple frontend (or server-rendered pages) backed by Postgres

---

## Data strategy

### Live data (Tradier)
- **Option chain snapshots** (REST): SPX options chain for each target expiration (3/5/7 DTE as available) with Greeks/IV as provided by Tradier.
  - Optional: range-based capture (e.g., 0–10 DTE) for broader research coverage.
- **Macro/context inputs** (REST where possible):
  - VIX and VIX9D quotes (if available via Tradier symbols)
  - Optional additional proxies (rates, etc.) only if reliable
- **Scheduling**:
  - Every **5 minutes** during regular trading hours (RTH).
  - Additionally, ensure snapshots exist at **10:00, 11:00, 12:00 ET** (entry decision times). If a scheduled tick lands exactly on these, reuse it.

### Historical data (Databento)
- **Dataset**: `OPRA.PILLAR` filtered to **SPX options**
- **Schema**: **`CBBO-1m`** as primary (NBBO + last sale at 1-minute intervals; sufficient for bar-level backtests)
- **Also download**:
  - **Instrument definitions** for options symbology/metadata (expiry/strike/right)
  - Optional: **Trades** schema for later liquidity/slippage research

---

## Trading rules (MVP)

### Trade type: SPX credit vertical spreads
- Two legs:
  - **Short** option (sell to open)
  - **Long** option (buy to open) at same expiration, farther OTM (defines risk)

### DTE targets
- Attempt to trade at **3, 5, 7 DTE** only.
- If no exact match exists due to calendar/holidays, select the closest expiration within a tolerance (configurable; default ±1 DTE) or skip.

### Entry times
- Consider entries only at:
  - **10:00 ET**
  - **11:00 ET**
  - **12:00 ET**
- At each entry time, if an active trade is open, **do not open a new trade**.

### Leg selection (delta-based)
- For each candidate setup (0.10 and 0.20 absolute delta):
  - Choose the option whose absolute delta is closest to target.
  - Define long leg farther OTM such that the spread width is consistent (configurable, e.g., 25/50 points) or derived from delta distance.
- Select the better candidate by score (rule-based initially; ML later).

### Management (close-only)
- **Profit target**: close when current PnL ≥ 0.50 × max_profit.
- **Stop loss**: close when current PnL ≤ -1.00 × max_profit.
- **Session exit**: optionally close all open positions near end of day (configurable; default off for 3/5/7 DTE unless needed).

### Position sizing
- **Fixed contracts** (configurable integer).

### Portfolio-level risk limits (MVP defaults)
- **Max open trades**: 1
- **Max new trades/day**: 1
- **Daily stop**: if a stop loss occurs, disable new entries for remainder of day
- **Event blackout**: if a high-impact event is within N hours, skip entries (optional table-driven)

---

## Pricing & fill assumptions (backtest and analytics)

### Live paper execution
- Place paper orders via Tradier sandbox:
  - Prefer **multi-leg order** API (`class=multileg`) even for 2 legs for consistency.
  - Use **credit limit orders** where supported; otherwise a controlled market/credit type with guardrails.
- Store:
  - preview response (if used)
  - submission payload
  - order id(s)
  - fills (including partial fills) and timestamps

### Backtest fill model (default)
- Use NBBO mid with conservative slippage:
  - Credit entry fill: `mid - 0.30 * (ask - bid)`
  - Credit exit fill: `mid + 0.30 * (ask - bid)`
- **No partial fills** in v1 backtest.

### Mark-to-market
- Mark each leg at mid; compute spread value from legs; compute PnL from entry credit vs current value.

---

## Database schema (Postgres “hot” tables)

### Reference / instruments
- `option_instruments`
  - option_symbol (primary key; vendor symbol)
  - root, expiration, strike, right (C/P), style (if available)
  - created_at

### Snapshots (reproducibility)
- `chain_snapshots`
  - snapshot_id (pk)
  - timestamp (ET normalized or UTC with timezone column)
  - underlying (SPX)
  - target_dte (3/5/7)
  - expiration
  - payload_json (raw chain response or normalized subset)
  - checksum (hash of payload)
- `context_snapshots`
  - timestamp (pk)
  - vix, vix9d, term_structure
  - gex_net, zero_gamma_level (approx)
  - notes_json

### Decisions (what the system intended to do)
- `trade_decisions`
  - decision_id (pk)
  - timestamp
  - target_dte
  - entry_slot (10/11/12)
  - delta_target (0.10 or 0.20)
  - chosen_legs_json (symbols/qty/sides)
  - strategy_params_json (strategy-specific inputs)
  - model_version / ruleset_version
  - score
  - decision (TRADE / SKIP) + reason
  - chain_snapshot_id (fk)

### Orders / fills (paper broker truth)
- `orders`
  - order_id (pk, broker order id)
  - decision_id (fk)
  - status
  - submitted_at, updated_at
  - request_json, response_json
- `fills`
  - fill_id (pk)
  - order_id (fk)
  - timestamp
  - option_symbol
  - qty
  - price

### Trades (normalized multi-leg state)
- `trades`
  - trade_id (pk)
  - decision_id (fk)
  - status (OPEN/CLOSED)
  - entry_time, exit_time
  - entry_credit
  - max_profit
  - exit_reason (TAKE_PROFIT_50 / STOP_LOSS / MANUAL / EXPIRED)
- `trade_legs`
  - trade_id (fk)
  - option_symbol
  - side (STO/BTO/STC/BTC)
  - qty
  - entry_price, exit_price

### ML + strategy versioning (phase 2)
- `strategy_versions`, `model_versions`, `training_runs`
- `feature_snapshots`, `trade_candidates`, `model_predictions`
- `strategy_recommendations`, `backtest_runs`

---

## Services / modules

### `ingestion/tradier_client.py`
- REST wrappers:
  - get_option_expirations(underlying)
  - get_option_chain(underlying, expiration, greeks=true)
  - get_quotes(symbols)
- Auth, retries, rate limiting, structured logging.

### `jobs/snapshot_job.py`
- Every 5 minutes:
  - fetch chain snapshots for target expirations
  - compute/store context snapshot (including GEX approximation)

### `strategy/vertical_credit.py`
- Build candidate vertical spreads from chain snapshot.
- Apply delta targeting, spread width constraints, and basic sanity checks (liquidity/spread thresholds).

### `engine/decision_engine.py`
- At 10/11/12 ET:
  - load latest snapshot
  - run ruleset/ML gating
  - write `trade_decisions`
  - if TRADE: create order payload for paper execution and submit

### `broker/tradier_paper.py`
- Place multileg orders in sandbox.
- Poll order status and ingest fills.

### `risk/rules.py`
- Enforce max open trades, max trades/day, daily stop, event blackout.

### `backtest/engine.py`
- DuckDB-backed replay using Databento `CBBO-1m`.
- Re-uses the same strategy selection + management rules.
- Produces the same `trades`/`trade_legs` shape as live for training parity.

---

## ML approach (phase 2)
- Build features strictly from data available at decision time (no leakage).
- Use **walk-forward evaluation** and keep true out-of-sample windows.
- ML ranks candidate trades; risk rules remain authoritative.
- Strategy adjustments are gated by policy + paper-trade shadow mode.

---

## Web dashboard (MVP pages)
- **Overview**: current status, next run, last snapshot timestamp, config summary.
- **Decisions**: list of decision events with reasons, scores, links to snapshots.
- **Orders/Fills**: broker status, fills, rejects.
- **Trades**: open/closed trades, PnL, exit reasons, equity curve.
- **Context**: VIX/VIX9D/term structure + GEX series charts.
- **Snapshots viewer**: inspect the option chain snapshot used for a decision.

---

## Notes / future extensions
- Add condors/butterflies after the vertical pipeline is stable.
- Add rolling rules after fill + PnL parity is validated between paper and backtest.
- Consider higher-frequency NBBO schemas only if the 1-minute model proves insufficient.

