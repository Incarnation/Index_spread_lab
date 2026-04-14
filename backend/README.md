# IndexSpreadLab -- Backend

FastAPI + APScheduler service that captures SPX/SPY/VIX option data, computes GEX, generates trade candidates, runs ML pipelines, manages paper-trade execution with portfolio budgeting, and serves the optimizer dashboard.

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
      optimizer.py      Optimizer run history, results, Pareto, walk-forward

  ingestion/
    tradier_client.py   Tradier REST wrapper (expirations, chains, quotes, clock)

  jobs/
    quote_job.py             Pulls quotes for configured symbols
    snapshot_job.py          SPX/SPY/VIX chain capture with DTE policy
    gex_job.py               Tradier-computed GEX aggregation
    cboe_gex_job.py          CBOE precomputed GEX fetch
    feature_builder_job.py   Feature snapshots + ranked candidate generation
    decision_job.py          Execution policy (rules + optional hybrid ML)
    labeler_job.py           Outcome labeling + orphan cascade
    trade_pnl_job.py         Mark-to-market + TP/SL/expiry close + SMS
    trainer_job.py           Walk-forward XGBoost trainer (sparse CV fallback)
    shadow_inference_job.py  Shadow model scoring
    promotion_gate_job.py    Quality/risk gate evaluation
    performance_analytics_job.py  Win rate, expectancy, drawdown aggregation
    staleness_monitor_job.py      Pipeline freshness alerting (SendGrid)
    eod_events_job.py        End-of-day signal capture + economic calendar
    retention_job.py         Batch purge of old chain/GEX data
    modeling.py              XGBoost model utilities (train, predict, features)

  services/
    portfolio_manager.py  Capital budgeting, lot scaling, drawdown stops
    event_signals.py      Event-driven signal detection (VIX, SPX drops, term structure)
    sms_notifier.py       Twilio SMS trade open/close notifications

  database/
    connection.py       Async engine + session factory
    schema.py           Startup schema bootstrap + migration runner
    sql/
      db_schema.sql              Base table definitions (30+ tables)
      migrations/                14 numbered idempotent SQL migrations
      db_reset_all_tables.sql    Full destructive reset
      db_reset_ml_tables.sql     ML-only destructive reset

configs/
  optimizer/
    schema.py                  Pydantic models for YAML config validation
    staged_stage1.yaml         Stage 1: trading param sweep (TP, SL, VIX, width)
    staged_stage2.yaml         Stage 2: portfolio param sweep
    staged_stage3.yaml         Stage 3: event signal param sweep
    event_only.yaml            Event-only mode full sweep
    event_only_v2.yaml         Phase 1: term_inversion ON/OFF validation
    event_only_v2_explore.yaml Phase 2: broad exploration (signal mode, VIX, entries)
    selective.yaml             Selective high-win-rate config sweep
    portfolio_sweep.yaml       Portfolio-only param sweep

scripts/                       21 offline CLI tools (see Scripts section below)

tests/                         53 test files (unit + integration + E2E)
```

---

## Data Flow

For runtime order of scheduled jobs and high-level workflows, see the root [README.md](../README.md) section **System Architecture**.

```mermaid
flowchart TD
  subgraph ingest [Data Ingestion]
    Tradier["Tradier API"] --> QuoteJob["quote_job"]
    Tradier --> SnapshotJob["snapshot_job"]
    CBOE["CBOE / MZData"] --> CboeGexJob["cboe_gex_job"]
  end

  subgraph db [PostgreSQL]
    UQ["underlying_quotes"]
    CS["context_snapshots"]
    ChS["chain_snapshots"]
    OCR["option_chain_rows"]
    GS["gex_snapshots"]
    FS["feature_snapshots"]
    TC["trade_candidates"]
    TD["trade_decisions"]
    TR["trades"]
    TM["trade_marks"]
    PS["portfolio_state"]
    PT["portfolio_trades"]
    MV["model_versions"]
    MP["model_predictions"]
    TP["trade_performance_*"]
    EE["economic_events"]
    OR["optimizer_*"]
  end

  subgraph execution [Trade Execution]
    EventSignals["EventSignalDetector"]
    PortfolioMgr["PortfolioManager"]
    DecisionJob["decision_job"]
    SmsNotifier["SmsNotifier"]
  end

  subgraph ml [ML Pipeline]
    FeatureBuilder["feature_builder_job"]
    Labeler["labeler_job"]
    Trainer["trainer_job"]
    Shadow["shadow_inference_job"]
    Promotion["promotion_gate_job"]
  end

  subgraph monitoring [Monitoring]
    TradePnl["trade_pnl_job"]
    PerfAnalytics["performance_analytics_job"]
    Staleness["staleness_monitor_job"]
    EodEvents["eod_events_job"]
    Retention["retention_job"]
  end

  subgraph offline [Offline Optimizer]
    GenTraining["generate_training_data.py"]
    BacktestEngine["backtest_strategy.py"]
    YamlConfig["YAML configs"]
    IngestScript["ingest_optimizer_results.py"]
  end

  QuoteJob --> UQ & CS
  SnapshotJob --> ChS & OCR
  SnapshotJob --> GexJob["gex_job"] --> GS
  CboeGexJob --> GS
  GS & CS --> FeatureBuilder --> FS & TC
  EventSignals --> DecisionJob
  PortfolioMgr --> DecisionJob
  TC --> DecisionJob --> TD & TR
  DecisionJob --> SmsNotifier
  TR --> TradePnl --> TM
  TradePnl --> SmsNotifier
  TradePnl --> PS & PT
  TC --> Labeler
  Labeler --> Trainer --> MV
  MV --> Shadow --> MP
  MP --> Promotion
  TM --> PerfAnalytics --> TP
  EodEvents --> EE

  GenTraining --> BacktestEngine
  YamlConfig --> BacktestEngine
  BacktestEngine --> IngestScript --> OR
```

---

## Scheduler Architecture

All jobs are registered in `scheduler_builder.py`. The scheduler uses these patterns:

**RTH-window jobs** (snapshot, quote, GEX, CBOE GEX, performance analytics):
- Fire every N minutes from 09:31-15:55 ET on weekdays.
- A separate 16:00 ET `force=True` trigger runs the final capture.
- Market-open guard skips holidays by checking Tradier clock state; a date is only marked as a trading day when an earlier guarded run observes `is_open=True`.

**Entry-time jobs** (feature builder, decision):
- Fire at configured `DECISION_ENTRY_TIMES` (e.g., `10:01,11:01,12:01,13:01,14:01,15:01,16:01`).
- Market-open guarded.
- Decision job evaluates event signals at each entry time, allowing intraday signal changes (e.g., a midday SPX drop triggers event trades at the next entry).

**After-close jobs** (labeler at 16:15, shadow inference at 16:20, EOD events at configurable time):
- Guarded so they only run on dates that had an observed open market earlier.

**Weekly jobs** (trainer on configured weekday, promotion gate 60 min after trainer):
- Standard cron triggers.

**Interval jobs** (trade PnL, staleness monitor):
- Simple interval triggers (not RTH-gated for trade PnL so marks update continuously).
- Trade PnL job triggers SMS notifications on trade open/close events.

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

**Event-only mode**: When `EVENT_EVENT_ONLY=true`, scheduled trades are suppressed entirely. Trades are only placed when the event signal detector fires (VIX spike, SPX drop, elevated VIX, etc.). This is the current production mode.

**Event signal detection**: `EventSignalDetector` reads the latest `underlying_quotes` at each entry time to compute SPX returns (1d, 2d) and VIX levels/changes. Signals can appear or disappear intraday as prices move. Configurable `signal_mode`: `any` (OR of all signals) or `spx_and_vix` (requires both SPX drop and VIX condition).

**Term inversion disabled**: Backtesting showed term-structure inversion was not a profitable signal on its own — it triggered put credit spreads on non-drop days that underperformed. The C3 configuration disables it by setting the threshold to 99.0 (unreachable).

**Portfolio management**: When `PORTFOLIO_ENABLED=true`, the decision job delegates to `PortfolioManager` for capital budgeting. Lot sizing scales with equity (`PORTFOLIO_LOT_PER_EQUITY`). Monthly drawdown stops halt new trades when cumulative monthly loss exceeds `PORTFOLIO_MONTHLY_DRAWDOWN_LIMIT`. Event-driven trades share or have separate budget depending on `EVENT_BUDGET_MODE`.

**SMS notifications**: `SmsNotifier` sends Twilio SMS on trade open and close events. Errors never block the pipeline (fire-and-forget with logging). Configurable per source via `SMS_NOTIFY_SOURCES`.

**Optimizer YAML configs**: Parameter sweeps are defined in YAML files (`configs/optimizer/`). The `schema.py` Pydantic model validates configs and builds the Cartesian product of all list-valued fields, applying filter rules to prune invalid combinations.

**Labeler orphan cascade**: Feature snapshots with no linked candidates after a configurable grace period are automatically expired to prevent unbounded growth.

**Trainer sparse fallback**: When walk-forward split has insufficient rows for time-series CV, the trainer falls back to sparse cross-validation to still produce a model version.

**Schema migrations**: Numbered SQL files in `database/sql/migrations/` run automatically on startup. Each migration is idempotent (uses `IF NOT EXISTS`, `IF NOT EXISTS ADD COLUMN`, etc). The `schema_migrations` table tracks which have been applied.

**Retention safety**: The retention job excludes chain snapshots referenced by open trades to prevent data loss during active positions.

---

## Offline Scripts

All scripts live in `backend/scripts/` and are run with `PYTHONPATH=.` from the `backend/` directory.

| Script | Purpose |
|--------|---------|
| `backtest_strategy.py` | Capital-budgeted backtester with optimizer modes (staged, event-only, selective, exhaustive, YAML-config) |
| `generate_training_data.py` | Offline training pipeline: exports production data to `training_candidates.csv` with P&L trajectories |
| `ingest_optimizer_results.py` | Ingest optimizer/walk-forward CSV results into PostgreSQL for the dashboard |
| `regime_analysis.py` | Regime-based performance analysis (VIX level, term structure, calendar) |
| `xgb_model.py` | XGBoost training, prediction, walk-forward validation |
| `upload_xgb_model.py` | Upload trained XGB model into `model_versions` table |
| `export_production_data.py` | Export production DB tables to CSV/Parquet for offline analysis |
| `download_databento.py` | Download historical SPX/SPY options from Databento |
| `generate_economic_calendar.py` | Unified economic-event calendar CSV (shared with `EodEventsJob`) |
| `sl_recovery_analysis.py` | Stop-loss recovery analysis and parameter sweeps |
| `experiment_tracker.py` | CSV-based experiment tracking for optimizer runs |
| `backtest_entry.py` | Backtest framework for SPX credit spread entries |
| `data_retention.py` | CLI to export and purge old chain/GEX data |
| `clean_start.py` | One-time DB cleanup for a fresh portfolio start |
| `health_check.py` | Production health check utility |
| `create_allowed_users.py` | Insert allowed user with bcrypt hash |
| `set_admin.py` | Promote existing user to admin |
| `reset_auth_audit_log.py` | Clear the auth audit log table |
| `test_sms.py` | Twilio SMS smoke test |
| `_env.py` | Central `.env` loader for offline scripts |
| `_constants.py` | Shared constants (margin, lot sizing) for offline scripts |

---

## Optimizer Pipeline

```mermaid
flowchart LR
  subgraph inputs [Inputs]
    ProdDB["Production DB"]
    YAML["YAML Config"]
  end

  subgraph generate [Data Generation]
    GenScript["generate_training_data.py"]
    CSV["training_candidates.csv — 233MB"]
  end

  subgraph optimize [Optimizer]
    Precomp["Precompute PnL columns"]
    Grid["Build config grid — Cartesian product"]
    Workers["Parallel backtests — multiprocessing.Pool"]
  end

  subgraph validate [Validation]
    WF["Walk-forward — rolling windows"]
    Regime["Regime analysis"]
  end

  subgraph output [Output]
    ResultsCSV["optimizer_results.csv"]
    IngestScript["ingest_optimizer_results.py"]
    Dashboard["Optimizer Dashboard"]
  end

  ProdDB --> GenScript --> CSV
  YAML --> Grid
  CSV --> Precomp --> Workers
  Grid --> Workers
  Workers --> ResultsCSV
  Workers --> WF
  Workers --> Regime
  ResultsCSV --> IngestScript --> Dashboard
```

### Optimizer Modes

| Mode | CLI Flag | Description |
|------|----------|-------------|
| YAML-config | `--optimize --config <yaml>` | Sweep all combos defined in a YAML file |
| Staged | `--optimize-staged` | 3-phase pipeline: trading → portfolio → event |
| Event-only | `--optimize-event-only` | Event signal parameter sweep |
| Selective | `--optimize-selective` | High-win-rate focused sweep |
| Exhaustive | `--optimize` (no config) | Hardcoded grid from `_build_optimizer_grid()` |
| Walk-forward | `--walkforward --wf-auto` | Rolling train/test window validation |
| Compare | `--compare` | Head-to-head preset config comparison |
| Analyze | `--analyze` | Pareto frontier and robustness from existing CSV |

### YAML Config Schema

YAML configs define list values for each parameter dimension. The `schema.py` Pydantic model validates and builds the Cartesian product, then applies filter rules:

- `min_dte_lt_max_dte`: Drop combos where event min DTE >= max DTE
- `min_delta_lt_max_delta`: Drop combos where event min delta >= max delta
- `shared_single_event_trade`: Drop `shared` budget + `max_event_trades > 1`
- `lot_per_equity_eq_capital`: Force `lot_per_equity == starting_capital`

Example: `event_only_v2_explore.yaml` generates 20,736 configs by sweeping signal mode, VIX thresholds, entry count, spread width, and TP/SL parameters.

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

Optimizer:
- `optimizer_runs`: run metadata (name, mode, git hash, config count, timestamps)
- `optimizer_results`: per-config backtest metrics with Pareto flags
- `optimizer_walkforward`: walk-forward validation results per window per config

ML/backtest:
- `strategy_versions`, `model_versions`, `training_runs`
- `feature_snapshots`, `trade_candidates`, `model_predictions`
- `backtest_runs`, `strategy_recommendations`

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

### Optimizer endpoints (prefix `/api/optimizer`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/optimizer/runs` | List all optimizer runs with summary stats |
| GET | `/api/optimizer/runs/{run_id}` | Single run details |
| GET | `/api/optimizer/runs/{run_id}/results` | Paginated/sorted results for a run |
| GET | `/api/optimizer/runs/{run_id}/pareto` | Pareto frontier configs |
| GET | `/api/optimizer/runs/{run_id}/equity-curve` | Equity curve for a config |
| GET | `/api/optimizer/regime-breakdown` | Regime performance breakdown |
| GET | `/api/optimizer/compare` | Multi-config comparison |
| GET | `/api/optimizer/walkforward/{run_id}` | Walk-forward validation results |

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
- `TRADE_PNL_TAKE_PROFIT_PCT`, `TRADE_PNL_STOP_LOSS_PCT`, `TRADE_PNL_STOP_LOSS_ENABLED`: exit thresholds.
- `EVENT_ENABLED`, `EVENT_SIGNAL_MODE`, `EVENT_MAX_TRADES`, `EVENT_EVENT_ONLY`: event trading.
- `SMS_ENABLED`, `SMS_ACCOUNT_SID`, `SMS_AUTH_TOKEN`, `SMS_FROM_NUMBER`, `SMS_TO_NUMBER`: Twilio SMS.
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
- The trainer requires a minimum number of resolved labels before producing a first model version. After a fresh DB reset, `no_model_versions` and `no_model_predictions` warnings are expected initially.
- CBOE GEX data availability depends on the vendor API; outages are logged but do not block other pipeline stages.
- Walk-forward validation mode (`--walkforward`) uses a hardcoded grid from `_build_optimizer_grid()`; it does not read YAML configs. For YAML-driven walk-forward, use a custom script that calls `run_backtest` directly.
