# IndexSpreadLab -- Backend

FastAPI + APScheduler service that captures SPX/SPY/VIX option data, computes GEX, generates trade candidates, runs ML pipelines, and manages paper-trade execution with portfolio budgeting.

---

## Module Map

```
spx_backend/
  config.py             Pydantic Settings (env-backed, single source of truth)
  main.py               Uvicorn entrypoint (PORT-aware for Railway)
  dte.py                Trading-day DTE helper (skips weekends + holidays)
  market_clock.py       Tradier clock cache with DB audit and RTH fallback
  scheduler_builder.py  APScheduler construction and all job registration

  web/
    app.py              FastAPI app, lifespan, middleware, CORS
    routers/
      public.py         Read-only data endpoints (chains, GEX, trades, analytics)
      admin.py          Manual-trigger and ops endpoints
      auth.py           JWT login/register/logout + audit log
      portfolio.py      Portfolio status, history, trades, config

  ingestion/
    tradier_client.py   Tradier REST wrapper (expirations, chains, quotes, clock)

  jobs/
    quote_job.py        Pulls quotes for configured symbols
    snapshot_job.py     SPX/SPY/VIX chain capture with DTE policy
    gex_job.py          Tradier-computed GEX aggregation
    cboe_gex_job.py     CBOE precomputed GEX fetch
    feature_builder_job.py  Feature snapshots + ranked candidate generation
    decision_job.py     Execution policy (rules + optional hybrid ML ranking)
    labeler_job.py      Outcome labeling + orphan cascade
    trade_pnl_job.py    Mark-to-market + TP/SL/expiry close
    trainer_job.py      Walk-forward XGBoost trainer (sparse CV fallback)
    shadow_inference_job.py  Shadow model scoring
    promotion_gate_job.py    Quality/risk gate evaluation
    performance_analytics_job.py  Win rate, expectancy, drawdown aggregation
    staleness_monitor_job.py     Pipeline freshness alerting (SendGrid)
    eod_events_job.py   End-of-day signal capture + economic calendar
    retention_job.py    Batch purge of old chain/GEX data
    modeling.py         XGBoost model utilities (train, predict, feature importance)

  services/
    portfolio_manager.py  Capital budgeting, lot scaling, drawdown stops
    event_signals.py      Event-driven signal detection (VIX spikes, SPX drops, term inversion)

  database/
    connection.py       Async engine + session factory
    schema.py           Startup schema bootstrap + migration runner
    sql/
      db_schema.sql     Base table definitions
      migrations/       Numbered idempotent SQL migrations (run on startup)
      db_reset_all_tables.sql   Full destructive reset
      db_reset_ml_tables.sql    ML-only destructive reset

  backtest/
    strategy.py         Local backtest engine
    sample_data/        Documentation for sample data formats

scripts/
  data_retention.py     CLI to export and purge old chain/GEX data

tests/                  45 test files (unit, integration, E2E)
```

---

## Data Flow

```
Tradier API                  CBOE Vendor API
    |                              |
    v                              v
quote_job -----> underlying_quotes
    |                              |
    +-----> context_snapshots <----+
    |                              |
snapshot_job -> chain_snapshots    |
    |           option_chain_rows  |
    v                              v
gex_job ------> gex_snapshots     cboe_gex_job --> gex_snapshots
                gex_by_strike                      gex_by_strike
                gex_by_expiry_strike               gex_by_expiry_strike
                    |
                    v
feature_builder_job --> feature_snapshots
                        trade_candidates
                            |
                            v
decision_job ---------> trade_decisions
    |                   trades / trade_legs
    |                       |
    v                       v
portfolio_manager       trade_pnl_job --> trade_marks
    |                       |
    v                       v
portfolio_state         labeler_job --> labels on feature_snapshots
portfolio_trades            |
                            v
                    trainer_job --> training_runs / model_versions
                        |
                        v
                    shadow_inference_job --> model_predictions
                        |
                        v
                    promotion_gate_job --> model_versions (rollout status)
                        |
                        v
                    performance_analytics_job --> trade_performance_*
```

---

## Scheduler Architecture

All jobs are registered in `scheduler_builder.py`. The scheduler uses these patterns:

**RTH-window jobs** (snapshot, quote, GEX, CBOE GEX, performance analytics):
- Fire every N minutes from 09:31-15:55 ET on weekdays.
- A separate 16:00 ET `force=True` trigger runs the final capture.
- Market-open guard skips holidays by checking Tradier clock state; a date is only marked as a trading day when an earlier guarded run observes `is_open=True`.

**Entry-time jobs** (feature builder, decision):
- Fire at configured `DECISION_ENTRY_TIMES` (e.g., `10:00,11:00,13:00`).
- Market-open guarded.

**After-close jobs** (labeler at 16:15, shadow inference at 16:20, EOD events at configurable time):
- Guarded so they only run on dates that had an observed open market earlier.

**Weekly jobs** (trainer on configured weekday, promotion gate 60 min after trainer):
- Standard cron triggers.

**Interval jobs** (trade PnL, staleness monitor):
- Simple interval triggers (not RTH-gated for trade PnL so marks update continuously).

**Daily jobs** (retention at 03:00 ET):
- Standard cron trigger.

**Job failure alerting**:
- APScheduler `EVENT_JOB_ERROR` and `EVENT_JOB_MISSED` events trigger email alerts via SendGrid.
- Per-job cooldown prevents alert storms.

**Startup warmup**:
- On boot, quote -> snapshot -> SPY snapshot -> VIX snapshot -> GEX -> CBOE GEX -> performance analytics run once sequentially to populate fresh data.

---

## Key Design Decisions

**DTE semantics**: Trading-session based, not calendar-day. Weekends and exchange holidays are skipped. See `dte.py` and the root README for examples.

**GEX dual-source**: Both Tradier-computed and CBOE precomputed GEX are stored with `source` discrimination. CBOE is preferred for canonical `gex_net`; Tradier is the fallback. `context_snapshots` has separate `gex_net_tradier` / `gex_net_cboe` columns to avoid overwrite races.

**Decision policy**: Hard risk guardrails run first (day caps, open-trade caps, per-side caps). If all pass, rules-based scoring selects the best candidate. When hybrid mode is enabled and eligible model predictions exist, model ranking is applied subject to minimum probability and expected PnL thresholds.

**Portfolio management**: When `PORTFOLIO_ENABLED=true`, the decision job delegates to `PortfolioManager` for capital budgeting. Lot sizing scales with equity (`PORTFOLIO_LOT_PER_EQUITY`). Monthly drawdown stops halt new trades when cumulative monthly loss exceeds `PORTFOLIO_MONTHLY_DRAWDOWN_LIMIT`. Event-driven trades (VIX spikes, SPX drops) share or have separate budget depending on `EVENT_BUDGET_MODE`.

**Labeler orphan cascade**: Feature snapshots with no linked candidates after a configurable grace period are automatically expired to prevent unbounded growth.

**Trainer sparse fallback**: When walk-forward split has insufficient rows for time-series CV, the trainer falls back to sparse cross-validation to still produce a model version.

**Schema migrations**: Numbered SQL files in `database/sql/migrations/` run automatically on startup. Each migration is idempotent (uses `IF NOT EXISTS`, `IF NOT EXISTS ADD COLUMN`, etc). The `schema_migrations` table tracks which have been applied.

**Retention safety**: The retention job excludes chain snapshots referenced by open trades to prevent data loss during active positions.

---

## Database Tables

Auth:
- `users`, `auth_audit_log`

Core ingestion:
- `chain_snapshots`, `option_chain_rows`, `option_instruments`
- `underlying_quotes`, `context_snapshots`, `market_clock_audit`

GEX:
- `gex_snapshots`, `gex_by_strike`, `gex_by_expiry_strike`

Decisions/trading:
- `trade_decisions`, `orders`, `fills`, `trades`, `trade_legs`, `trade_marks`

Portfolio:
- `portfolio_state`, `portfolio_trades`

Performance:
- `trade_performance_snapshots`, `trade_performance_breakdowns`, `trade_performance_equity_curve`

ML/backtest:
- `strategy_versions`, `model_versions`, `training_runs`
- `feature_snapshots`, `trade_candidates`, `model_predictions`
- `backtest_runs`, `strategy_recommendations` (deprecated)

Other:
- `economic_events`, `schema_migrations`

---

## API Reference

### Public endpoints (authenticated)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe (unauthenticated) |
| GET | `/api/pipeline-status` | Pipeline freshness and warnings |
| GET | `/api/chain-snapshots` | Recent chain snapshot metadata |
| GET | `/api/trade-decisions` | Recent TRADE/SKIP decisions |
| GET | `/api/trades` | Paper trades with legs and PnL |
| GET | `/api/label-metrics` | TP50/TP100 and realized PnL summary |
| GET | `/api/strategy-metrics` | Strategy quality/risk metrics |
| GET | `/api/performance-analytics` | Performance with breakdowns |
| GET | `/api/model-ops` | Model/training/prediction ops summary |
| GET | `/api/model-predictions` | Paginated prediction browser |
| GET | `/api/model-accuracy` | Accuracy/precision/recall over windows |
| GET | `/api/model-calibration` | Calibration curve bins |
| GET | `/api/model-pnl-attribution` | Model PnL attribution vs baseline |
| GET | `/api/gex/snapshots` | Recent GEX snapshot batches |
| GET | `/api/gex/dtes` | Available DTEs for a GEX batch |
| GET | `/api/gex/expirations` | Available expirations for a GEX batch |
| GET | `/api/gex/curve` | Strike curve (all, one DTE, or custom expirations) |
| GET | `/api/backtest-results` | Backtest run results |

### Portfolio endpoints (prefix `/api/portfolio`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/portfolio/status` | Current equity, lots, drawdown state |
| GET | `/api/portfolio/history` | Daily equity history |
| GET | `/api/portfolio/trades` | Portfolio trade log with source tracking |
| GET | `/api/portfolio/config` | Active portfolio + event + decision config |

### Admin endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/admin/run-snapshot` | Trigger snapshot job |
| POST | `/api/admin/run-quotes` | Trigger quote job |
| POST | `/api/admin/run-gex` | Trigger GEX job |
| POST | `/api/admin/run-cboe-gex` | Trigger CBOE GEX job |
| POST | `/api/admin/run-decision` | Trigger decision job |
| POST | `/api/admin/run-feature-builder` | Trigger feature builder |
| POST | `/api/admin/run-labeler` | Trigger labeler |
| POST | `/api/admin/run-trade-pnl` | Trigger trade PnL job |
| POST | `/api/admin/run-performance-analytics` | Trigger performance analytics |
| POST | `/api/admin/run-trainer` | Trigger trainer |
| POST | `/api/admin/run-shadow-inference` | Trigger shadow inference |
| POST | `/api/admin/run-promotion-gates` | Trigger promotion gates |
| DELETE | `/api/admin/trade-decisions/{id}` | Delete a decision |
| GET | `/api/admin/expirations` | Fetch expirations from Tradier |
| GET | `/api/admin/auth-audit` | Authentication audit log |
| GET | `/api/admin/preflight` | One-call pipeline health check |

### Auth endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/login` | Login, returns JWT |
| POST | `/api/auth/register` | Create user account |
| GET | `/api/auth/me` | Current user info |
| POST | `/api/auth/logout` | Logout |

---

## Configuration

All settings live in `config.py` as a Pydantic `Settings` class backed by environment variables. See the root README for the full configuration reference.

Key backend-specific settings:

- `DATABASE_URL`: PostgreSQL connection string (asyncpg).
- `TRADIER_ACCESS_TOKEN`, `TRADIER_ACCOUNT_ID`: Tradier API credentials.
- `SNAPSHOT_INTERVAL_MINUTES`, `QUOTE_INTERVAL_MINUTES`, `GEX_INTERVAL_MINUTES`: RTH cadences.
- `DECISION_ENTRY_TIMES`: comma-separated `HH:MM` times for candidate generation and execution.
- `TRADE_PNL_TAKE_PROFIT_PCT`, `TRADE_PNL_STOP_LOSS_PCT`: exit thresholds.
- `TRAINER_WEEKDAY`, `TRAINER_HOUR`, `TRAINER_MINUTE`: weekly trainer schedule.
- `SENDGRID_API_KEY`, `EMAIL_ALERT_RECIPIENT`: alerting credentials.

---

## Running Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spx_backend.main
```

Default: `http://localhost:8000`

---

## Testing

Install dev dependencies:

```bash
pip install -r requirements-dev.txt
```

Run all unit tests:

```bash
python -m pytest -q
```

Run DB-backed integration tests (requires local test Postgres):

```bash
docker compose -f ../docker-compose.test.yml up -d
export DATABASE_URL_TEST="postgresql+asyncpg://spx_test:spx_test_pw@localhost:5434/index_spread_lab_test"
python -m pytest -q -m integration
```

Safety behavior:
- Integration tests skip if `DATABASE_URL_TEST` is not set.
- They fail fast if DB host is not local or DB name does not include `test`.

Convenience targets (from repo root):

```bash
make test-e2e-up        # Start test DB
make test-e2e-mocked    # Run mocked tests
make test-e2e-db        # Run DB integration tests
make test-e2e           # Run all tests
make test-predeploy     # Full predeploy gate
make test-e2e-down      # Stop test DB
```

---

## Known Limitations

- Live broker order/fill automation is scaffolded (`orders`, `fills` tables) but not wired to a real broker yet.
- Backtest orchestration and historical backfill tooling are still evolving.
- The trainer requires a minimum number of resolved labels before producing a first model version. After a fresh DB reset, `no_model_versions` and `no_model_predictions` warnings are expected initially.
- CBOE GEX data availability depends on the vendor API; outages are logged but do not block other pipeline stages.
