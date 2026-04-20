-- Minimal schema for MVP (expand as needed).
-- Safe to run multiple times (IF NOT EXISTS used).

CREATE TABLE IF NOT EXISTS users (
  id BIGSERIAL PRIMARY KEY,
  username TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  is_admin BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);

CREATE TABLE IF NOT EXISTS auth_audit_log (
  id BIGSERIAL PRIMARY KEY,
  event_type TEXT NOT NULL,
  user_id BIGINT NULL REFERENCES users(id) ON DELETE SET NULL,
  username TEXT NULL,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ip_address INET NULL,
  user_agent TEXT NULL,
  country TEXT NULL,
  geo_json JSONB NULL,
  details JSONB NULL
);

CREATE INDEX IF NOT EXISTS idx_auth_audit_log_occurred_at ON auth_audit_log (occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_auth_audit_log_user_id ON auth_audit_log (user_id);
CREATE INDEX IF NOT EXISTS idx_auth_audit_log_event_type ON auth_audit_log (event_type);

CREATE TABLE IF NOT EXISTS option_instruments (
  option_symbol TEXT PRIMARY KEY,
  root TEXT NOT NULL,
  expiration DATE NOT NULL,
  strike DOUBLE PRECISION NOT NULL,
  option_right TEXT NOT NULL, -- 'C' or 'P'
  style TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chain_snapshots (
  snapshot_id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
  underlying TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'TRADIER',
  target_dte INTEGER NOT NULL,
  expiration DATE NOT NULL,
  checksum TEXT NOT NULL,
  -- Audit Wave 4 M1: discriminate full chain payloads (snapshot_job)
  -- from FK-anchor-only rows (cboe_gex_job). See migration 023.
  payload_kind TEXT NOT NULL DEFAULT 'options_chain'
    CHECK (payload_kind IN ('options_chain', 'gex_anchor'))
);

CREATE INDEX IF NOT EXISTS idx_chain_snapshots_ts ON chain_snapshots (ts DESC);
CREATE INDEX IF NOT EXISTS idx_chain_snapshots_exp ON chain_snapshots (expiration, ts DESC);
CREATE INDEX IF NOT EXISTS idx_chain_snapshots_underlying_ts ON chain_snapshots (underlying, ts DESC);
CREATE INDEX IF NOT EXISTS idx_chain_snapshots_source_ts ON chain_snapshots (source, ts DESC);
CREATE INDEX IF NOT EXISTS idx_chain_snapshots_underlying_source_ts ON chain_snapshots (underlying, source, ts DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_chain_snapshots_ts_und_exp_src ON chain_snapshots (ts, underlying, expiration, source);

CREATE TABLE IF NOT EXISTS option_chain_rows (
  snapshot_id BIGINT NOT NULL REFERENCES chain_snapshots(snapshot_id) ON DELETE CASCADE,
  option_symbol TEXT NOT NULL,
  underlying TEXT NOT NULL,
  expiration DATE NOT NULL,
  strike DOUBLE PRECISION NULL,
  option_right TEXT NULL, -- 'C' or 'P'
  bid DOUBLE PRECISION NULL,
  ask DOUBLE PRECISION NULL,
  last DOUBLE PRECISION NULL,
  volume BIGINT NULL,
  open_interest BIGINT NULL,
  contract_size BIGINT NULL,
  delta DOUBLE PRECISION NULL,
  gamma DOUBLE PRECISION NULL,
  theta DOUBLE PRECISION NULL,
  vega DOUBLE PRECISION NULL,
  rho DOUBLE PRECISION NULL,
  bid_iv DOUBLE PRECISION NULL,
  mid_iv DOUBLE PRECISION NULL,
  ask_iv DOUBLE PRECISION NULL,
  greeks_updated_at TEXT NULL,
  raw_json JSONB NULL,
  PRIMARY KEY (snapshot_id, option_symbol)
);

CREATE INDEX IF NOT EXISTS idx_option_chain_rows_exp_strike ON option_chain_rows (expiration, strike, option_right);
CREATE INDEX IF NOT EXISTS idx_option_chain_rows_symbol ON option_chain_rows (option_symbol);
CREATE INDEX IF NOT EXISTS idx_option_chain_rows_underlying_exp_strike
  ON option_chain_rows (underlying, expiration, strike, option_right);
CREATE INDEX IF NOT EXISTS idx_option_chain_rows_underlying_snapshot
  ON option_chain_rows (underlying, snapshot_id);
-- Audit Wave 4 M8: small partial indexes for incident-response diagnostics.
-- See migration 024 for rationale.
CREATE INDEX IF NOT EXISTS idx_option_chain_rows_null_right
  ON option_chain_rows (snapshot_id) WHERE option_right IS NULL;
CREATE INDEX IF NOT EXISTS idx_option_chain_rows_null_delta
  ON option_chain_rows (snapshot_id) WHERE delta IS NULL;
CREATE INDEX IF NOT EXISTS idx_option_chain_rows_null_bidask
  ON option_chain_rows (snapshot_id) WHERE bid IS NULL AND ask IS NULL;

CREATE TABLE IF NOT EXISTS gex_snapshots (
  snapshot_id BIGINT PRIMARY KEY REFERENCES chain_snapshots(snapshot_id) ON DELETE CASCADE,
  ts TIMESTAMPTZ NOT NULL,
  underlying TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'TRADIER',
  spot_price DOUBLE PRECISION NULL,
  gex_net DOUBLE PRECISION NULL,
  gex_calls DOUBLE PRECISION NULL,
  gex_puts DOUBLE PRECISION NULL,
  gex_abs DOUBLE PRECISION NULL,
  zero_gamma_level DOUBLE PRECISION NULL,
  method TEXT NOT NULL DEFAULT 'oi_gamma_spot'
);

CREATE INDEX IF NOT EXISTS idx_gex_snapshots_source_ts ON gex_snapshots (source, ts DESC);
CREATE INDEX IF NOT EXISTS idx_gex_snapshots_underlying_source_ts ON gex_snapshots (underlying, source, ts DESC);

CREATE TABLE IF NOT EXISTS gex_by_strike (
  snapshot_id BIGINT NOT NULL REFERENCES chain_snapshots(snapshot_id) ON DELETE CASCADE,
  strike DOUBLE PRECISION NOT NULL,
  gex_net DOUBLE PRECISION NULL,
  gex_calls DOUBLE PRECISION NULL,
  gex_puts DOUBLE PRECISION NULL,
  oi_total BIGINT NULL,
  method TEXT NOT NULL DEFAULT 'oi_gamma_spot',
  PRIMARY KEY (snapshot_id, strike)
);

CREATE TABLE IF NOT EXISTS gex_by_expiry_strike (
  snapshot_id BIGINT NOT NULL REFERENCES chain_snapshots(snapshot_id) ON DELETE CASCADE,
  expiration DATE NOT NULL,
  dte_days INTEGER NULL,
  strike DOUBLE PRECISION NOT NULL,
  gex_net DOUBLE PRECISION NULL,
  gex_calls DOUBLE PRECISION NULL,
  gex_puts DOUBLE PRECISION NULL,
  oi_total BIGINT NULL,
  method TEXT NOT NULL DEFAULT 'oi_gamma_spot',
  PRIMARY KEY (snapshot_id, expiration, strike)
);

CREATE INDEX IF NOT EXISTS idx_gex_by_expiry_snapshot_dte ON gex_by_expiry_strike (snapshot_id, dte_days, strike);

CREATE TABLE IF NOT EXISTS context_snapshots (
  ts TIMESTAMPTZ NOT NULL,
  underlying TEXT NOT NULL DEFAULT 'SPX',
  spx_price DOUBLE PRECISION NULL,
  spy_price DOUBLE PRECISION NULL,
  vix DOUBLE PRECISION NULL,
  vix9d DOUBLE PRECISION NULL,
  term_structure DOUBLE PRECISION NULL,
  vvix DOUBLE PRECISION NULL,
  skew DOUBLE PRECISION NULL,
  -- Audit Wave 2 M2: gex_net is derived from the source-tagged columns
  -- so concurrent writers can't race on which one wins. See migration 020.
  -- Writers MUST NOT touch this column directly (PG rejects with 428C9).
  zero_gamma_level DOUBLE PRECISION NULL,
  gex_net_tradier DOUBLE PRECISION NULL,
  zero_gamma_level_tradier DOUBLE PRECISION NULL,
  gex_net_cboe DOUBLE PRECISION NULL,
  zero_gamma_level_cboe DOUBLE PRECISION NULL,
  notes_json JSONB NULL,
  gex_net DOUBLE PRECISION
    GENERATED ALWAYS AS (COALESCE(gex_net_cboe, gex_net_tradier)) STORED,
  PRIMARY KEY (ts, underlying)
);

CREATE TABLE IF NOT EXISTS underlying_quotes (
  quote_id BIGSERIAL PRIMARY KEY,
  -- ts is INGEST time (set by quote_job from datetime.now). vendor_ts holds
  -- the vendor's reported observation timestamp (Tradier trade_date) when
  -- available. Audit Wave 2 H6 -- monitoring uses MAX(ts), as-of consumers
  -- use COALESCE(vendor_ts, ts).
  ts TIMESTAMPTZ NOT NULL,
  vendor_ts TIMESTAMPTZ NULL,
  symbol TEXT NOT NULL,
  last DOUBLE PRECISION NULL,
  bid DOUBLE PRECISION NULL,
  ask DOUBLE PRECISION NULL,
  open DOUBLE PRECISION NULL,
  high DOUBLE PRECISION NULL,
  low DOUBLE PRECISION NULL,
  close DOUBLE PRECISION NULL,
  volume BIGINT NULL,
  change DOUBLE PRECISION NULL,
  change_percent DOUBLE PRECISION NULL,
  prevclose DOUBLE PRECISION NULL,
  source TEXT NOT NULL DEFAULT 'tradier',
  raw_json JSONB NULL
);

CREATE INDEX IF NOT EXISTS idx_underlying_quotes_symbol_ts ON underlying_quotes (symbol, ts DESC);
-- Audit Wave 2 H5: idempotency for quote_job retries via UNIQUE on
-- (symbol, minute_bucket). Targeted by ON CONFLICT (symbol,
-- (date_bin('1 minute', ts, TIMESTAMPTZ '2000-01-01 00:00:00+00')))
-- DO NOTHING. We use date_bin rather than date_trunc because PG
-- requires unique-index expressions to be IMMUTABLE; date_trunc on
-- timestamptz is STABLE. See migration 018.
CREATE UNIQUE INDEX IF NOT EXISTS uq_underlying_quotes_symbol_ts_minute
  ON underlying_quotes (symbol, (date_bin('1 minute', ts, TIMESTAMPTZ '2000-01-01 00:00:00+00')));
-- Audit Wave 2 H6: functional index supporting
-- ORDER BY COALESCE(vendor_ts, ts) DESC for as-of spot lookups.
CREATE INDEX IF NOT EXISTS idx_underlying_quotes_symbol_vendor_or_ts
  ON underlying_quotes (symbol, (COALESCE(vendor_ts, ts)) DESC);

CREATE TABLE IF NOT EXISTS market_clock_audit (
  clock_id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
  state TEXT NULL,
  is_open BOOLEAN NULL,
  source TEXT NOT NULL DEFAULT 'tradier',
  raw_json JSONB NULL,
  error TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_market_clock_audit_ts ON market_clock_audit (ts DESC);

CREATE TABLE IF NOT EXISTS strategy_versions (
  strategy_version_id BIGSERIAL PRIMARY KEY,
  strategy_name TEXT NOT NULL,
  version TEXT NOT NULL,
  params_json JSONB NOT NULL,
  code_hash TEXT NULL,
  parent_strategy_version_id BIGINT NULL REFERENCES strategy_versions(strategy_version_id),
  rollout_status TEXT NOT NULL DEFAULT 'draft', -- draft/shadow/active/retired
  is_active BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  approved_by TEXT NULL,
  approved_at TIMESTAMPTZ NULL,
  notes TEXT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_versions_name_version ON strategy_versions (strategy_name, version);
CREATE INDEX IF NOT EXISTS idx_strategy_versions_rollout_status ON strategy_versions (rollout_status, created_at DESC);

CREATE TABLE IF NOT EXISTS model_versions (
  model_version_id BIGSERIAL PRIMARY KEY,
  model_name TEXT NOT NULL,
  version TEXT NOT NULL,
  algorithm TEXT NOT NULL,
  target_label TEXT NOT NULL DEFAULT 'hit_tp50_before_sl_or_expiry',
  prediction_type TEXT NOT NULL DEFAULT 'classification', -- classification/regression/ranking
  strategy_version_id BIGINT NULL REFERENCES strategy_versions(strategy_version_id),
  feature_schema_version TEXT NOT NULL DEFAULT 'fs_v1',
  candidate_schema_version TEXT NOT NULL DEFAULT 'cand_v1',
  label_schema_version TEXT NOT NULL DEFAULT 'label_v1',
  feature_spec_json JSONB NOT NULL,
  calibration_method TEXT NULL,
  decision_threshold DOUBLE PRECISION NULL,
  data_snapshot_json JSONB NULL,
  train_start TIMESTAMPTZ NULL,
  train_end TIMESTAMPTZ NULL,
  val_start TIMESTAMPTZ NULL,
  val_end TIMESTAMPTZ NULL,
  metrics_json JSONB NULL,
  artifact_uri TEXT NULL,
  rollout_status TEXT NOT NULL DEFAULT 'shadow', -- shadow/canary/active/retired
  is_active BOOLEAN NOT NULL DEFAULT false,
  promoted_at TIMESTAMPTZ NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  notes TEXT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_versions_name_version ON model_versions (model_name, version);
CREATE INDEX IF NOT EXISTS idx_model_versions_rollout_status ON model_versions (rollout_status, created_at DESC);

-- The online ML schema lives in `model_versions` only; legacy
-- `feature_snapshots`, `trade_candidates`, `model_predictions`, and
-- `training_runs` were dropped by migration
-- `015_decommission_online_ml_schema.sql`.  `model_versions` is the
-- registration target for offline-trained artifacts (e.g.
-- `upload_xgb_model.py`).

CREATE TABLE IF NOT EXISTS backtest_runs (
  run_id BIGSERIAL PRIMARY KEY,
  started_at TIMESTAMPTZ NOT NULL,
  finished_at TIMESTAMPTZ NULL,
  run_type TEXT NOT NULL DEFAULT 'backtest', -- backtest/walk_forward/simulation
  status TEXT NOT NULL, -- RUNNING/COMPLETED/FAILED
  strategy_version_id BIGINT NULL REFERENCES strategy_versions(strategy_version_id),
  model_version_id BIGINT NULL REFERENCES model_versions(model_version_id),
  train_window_start TIMESTAMPTZ NULL,
  train_window_end TIMESTAMPTZ NULL,
  test_window_start TIMESTAMPTZ NULL,
  test_window_end TIMESTAMPTZ NULL,
  data_source_json JSONB NULL,
  rows_evaluated BIGINT NULL,
  checksum TEXT NULL,
  config_json JSONB NOT NULL,
  stats_json JSONB NULL,
  notes TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_backtest_runs_started_at ON backtest_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_status ON backtest_runs (status, started_at DESC);

CREATE TABLE IF NOT EXISTS strategy_recommendations (
  recommendation_id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  recommendation_type TEXT NOT NULL DEFAULT 'param_tune', -- param_tune/model_select/risk_update
  strategy_version_id BIGINT NULL REFERENCES strategy_versions(strategy_version_id),
  model_version_id BIGINT NULL REFERENCES model_versions(model_version_id),
  backtest_run_id BIGINT NULL REFERENCES backtest_runs(run_id),
  proposed_params_json JSONB NOT NULL,
  confidence DOUBLE PRECISION NULL,
  expected_uplift DOUBLE PRECISION NULL,
  input_snapshot_json JSONB NULL,
  status TEXT NOT NULL DEFAULT 'PENDING', -- PENDING/APPROVED/REJECTED/APPLIED
  approved_at TIMESTAMPTZ NULL,
  applied_at TIMESTAMPTZ NULL,
  expires_at TIMESTAMPTZ NULL,
  metrics_json JSONB NULL,
  reason TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_strategy_recommendations_status ON strategy_recommendations (status, created_at DESC);

CREATE TABLE IF NOT EXISTS trade_decisions (
  decision_id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
  target_dte INTEGER NOT NULL,
  entry_slot INTEGER NOT NULL, -- 10/11/12
  delta_target DOUBLE PRECISION NOT NULL,
  chosen_legs_json JSONB NULL,
  strategy_params_json JSONB NULL,
  ruleset_version TEXT NOT NULL,
  score DOUBLE PRECISION NULL,
  decision TEXT NOT NULL, -- 'TRADE'/'SKIP'
  reason TEXT NULL,
  -- Online-ML legacy columns (`model_score`, `expected_value`, `policy_version`,
  -- `risk_gate_json`, `experiment_tag`, `feature_snapshot_id`, `candidate_id`,
  -- `prediction_id`) were dropped by migration 015.  `model_version_id` is
  -- retained for offline ML re-entry.
  chain_snapshot_id BIGINT NULL REFERENCES chain_snapshots(snapshot_id) ON DELETE SET NULL,
  strategy_version_id BIGINT NULL REFERENCES strategy_versions(strategy_version_id),
  model_version_id BIGINT NULL REFERENCES model_versions(model_version_id),
  decision_source TEXT NOT NULL DEFAULT 'rules'
);

CREATE INDEX IF NOT EXISTS idx_trade_decisions_ts ON trade_decisions (ts DESC);
CREATE INDEX IF NOT EXISTS idx_trade_decisions_strategy_ts ON trade_decisions (strategy_version_id, ts DESC);

CREATE TABLE IF NOT EXISTS orders (
  order_id TEXT PRIMARY KEY,
  decision_id BIGINT NULL REFERENCES trade_decisions(decision_id),
  status TEXT NOT NULL,
  submitted_at TIMESTAMPTZ NULL,
  updated_at TIMESTAMPTZ NULL,
  request_json JSONB NOT NULL,
  response_json JSONB NULL
);

CREATE TABLE IF NOT EXISTS fills (
  fill_id BIGSERIAL PRIMARY KEY,
  order_id TEXT NOT NULL REFERENCES orders(order_id),
  ts TIMESTAMPTZ NOT NULL,
  option_symbol TEXT NOT NULL,
  qty INTEGER NOT NULL,
  price DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fills_order_ts ON fills (order_id, ts);

CREATE TABLE IF NOT EXISTS trades (
  trade_id BIGSERIAL PRIMARY KEY,
  decision_id BIGINT NULL REFERENCES trade_decisions(decision_id),
  -- `candidate_id` and `feature_snapshot_id` were dropped by migration 015
  -- when their referenced tables (trade_candidates / feature_snapshots) were
  -- removed.  `_create_trade_from_decision` never populated them post-A.4.
  entry_snapshot_id BIGINT NULL REFERENCES chain_snapshots(snapshot_id) ON DELETE SET NULL,
  last_snapshot_id BIGINT NULL REFERENCES chain_snapshots(snapshot_id) ON DELETE SET NULL,
  backtest_run_id BIGINT NULL REFERENCES backtest_runs(run_id),
  trade_source TEXT NOT NULL DEFAULT 'live', -- live/paper/backtest
  strategy_version_id BIGINT NULL REFERENCES strategy_versions(strategy_version_id),
  model_version_id BIGINT NULL REFERENCES model_versions(model_version_id),
  strategy_type TEXT NOT NULL,
  status TEXT NOT NULL, -- OPEN/CLOSED/ROLLED
  underlying TEXT NOT NULL,
  entry_time TIMESTAMPTZ NOT NULL,
  exit_time TIMESTAMPTZ NULL,
  last_mark_ts TIMESTAMPTZ NULL,
  target_dte INTEGER NULL,
  expiration DATE NULL,
  contracts INTEGER NOT NULL DEFAULT 1,
  contract_multiplier INTEGER NOT NULL DEFAULT 100,
  spread_width_points DOUBLE PRECISION NULL,
  entry_credit DOUBLE PRECISION NULL,
  max_profit DOUBLE PRECISION NULL,
  max_loss DOUBLE PRECISION NULL,
  take_profit_target DOUBLE PRECISION NULL,
  stop_loss_target DOUBLE PRECISION NULL,
  current_exit_cost DOUBLE PRECISION NULL,
  current_pnl DOUBLE PRECISION NULL,
  realized_pnl DOUBLE PRECISION NULL,
  exit_reason TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades (entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_backtest_run ON trades (backtest_run_id, entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_strategy_entry ON trades (strategy_version_id, entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_status_entry ON trades (status, entry_time DESC);

CREATE TABLE IF NOT EXISTS trade_legs (
  trade_id BIGINT NOT NULL REFERENCES trades(trade_id),
  leg_index INTEGER NOT NULL,
  option_symbol TEXT NOT NULL,
  side TEXT NOT NULL, -- STO/BTO/STC/BTC
  qty INTEGER NOT NULL,
  entry_price DOUBLE PRECISION NULL,
  exit_price DOUBLE PRECISION NULL,
  strike DOUBLE PRECISION NULL,
  expiration DATE NULL,
  option_right TEXT NULL,
  PRIMARY KEY (trade_id, leg_index)
);

CREATE TABLE IF NOT EXISTS trade_marks (
  mark_id BIGSERIAL PRIMARY KEY,
  trade_id BIGINT NOT NULL REFERENCES trades(trade_id) ON DELETE CASCADE,
  snapshot_id BIGINT NULL REFERENCES chain_snapshots(snapshot_id) ON DELETE SET NULL,
  ts TIMESTAMPTZ NOT NULL,
  short_mid DOUBLE PRECISION NULL,
  long_mid DOUBLE PRECISION NULL,
  exit_cost DOUBLE PRECISION NULL,
  pnl DOUBLE PRECISION NULL,
  status TEXT NOT NULL DEFAULT 'OPEN', -- OPEN/CLOSED
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (trade_id, ts)
);

CREATE INDEX IF NOT EXISTS idx_trade_marks_trade_ts ON trade_marks (trade_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_trade_marks_ts ON trade_marks (ts DESC);

CREATE TABLE IF NOT EXISTS trade_performance_snapshots (
  analytics_snapshot_id BIGSERIAL PRIMARY KEY,
  as_of_ts TIMESTAMPTZ NOT NULL,
  job_started_at TIMESTAMPTZ NULL,
  job_finished_at TIMESTAMPTZ NULL,
  source_trade_count BIGINT NOT NULL DEFAULT 0,
  source_closed_count BIGINT NOT NULL DEFAULT 0,
  source_open_count BIGINT NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_trade_performance_snapshots_asof ON trade_performance_snapshots (as_of_ts DESC);
CREATE INDEX IF NOT EXISTS idx_trade_performance_snapshots_created ON trade_performance_snapshots (created_at DESC);

CREATE TABLE IF NOT EXISTS trade_performance_breakdowns (
  analytics_snapshot_id BIGINT NOT NULL REFERENCES trade_performance_snapshots(analytics_snapshot_id) ON DELETE CASCADE,
  mode TEXT NOT NULL CHECK (mode IN ('realized', 'combined')),
  bucket_date DATE NOT NULL,
  dimension_type TEXT NOT NULL, -- side/dte_bucket/delta_bucket/weekday/hour/source
  dimension_value TEXT NOT NULL,
  trade_count BIGINT NOT NULL DEFAULT 0,
  win_count BIGINT NOT NULL DEFAULT 0,
  loss_count BIGINT NOT NULL DEFAULT 0,
  pnl_sum DOUBLE PRECISION NOT NULL DEFAULT 0,
  win_pnl_sum DOUBLE PRECISION NOT NULL DEFAULT 0,
  loss_pnl_sum DOUBLE PRECISION NOT NULL DEFAULT 0,
  PRIMARY KEY (analytics_snapshot_id, mode, bucket_date, dimension_type, dimension_value)
);

CREATE INDEX IF NOT EXISTS idx_trade_performance_breakdowns_snapshot_dimension
  ON trade_performance_breakdowns (analytics_snapshot_id, dimension_type, mode);
CREATE INDEX IF NOT EXISTS idx_trade_performance_breakdowns_dimension_value_date
  ON trade_performance_breakdowns (dimension_type, dimension_value, bucket_date DESC);

CREATE TABLE IF NOT EXISTS trade_performance_equity_curve (
  analytics_snapshot_id BIGINT NOT NULL REFERENCES trade_performance_snapshots(analytics_snapshot_id) ON DELETE CASCADE,
  mode TEXT NOT NULL CHECK (mode IN ('realized', 'combined')),
  bucket_date DATE NOT NULL,
  trade_count BIGINT NOT NULL DEFAULT 0,
  win_count BIGINT NOT NULL DEFAULT 0,
  loss_count BIGINT NOT NULL DEFAULT 0,
  pnl_sum DOUBLE PRECISION NOT NULL DEFAULT 0,
  win_pnl_sum DOUBLE PRECISION NOT NULL DEFAULT 0,
  loss_pnl_sum DOUBLE PRECISION NOT NULL DEFAULT 0,
  cumulative_pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
  peak_pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
  drawdown DOUBLE PRECISION NOT NULL DEFAULT 0,
  PRIMARY KEY (analytics_snapshot_id, mode, bucket_date)
);

CREATE INDEX IF NOT EXISTS idx_trade_performance_equity_curve_snapshot_mode_date
  ON trade_performance_equity_curve (analytics_snapshot_id, mode, bucket_date ASC);
CREATE INDEX IF NOT EXISTS idx_trade_performance_equity_curve_mode_date
  ON trade_performance_equity_curve (mode, bucket_date DESC);

-- Economic event calendar (FOMC, CPI, NFP, OPEX) for feature derivation.
CREATE TABLE IF NOT EXISTS economic_events (
    date               DATE    NOT NULL,
    event_type         TEXT    NOT NULL,
    has_projections    BOOLEAN NOT NULL DEFAULT false,
    is_triple_witching BOOLEAN NOT NULL DEFAULT false,
    -- Audit Wave 2 M7: written by eod_events_job's ON CONFLICT DO UPDATE
    -- so a corrected has_projections / is_triple_witching value can land
    -- and the operator can audit which rows were rewritten. NULL for
    -- pre-M7 rows. See migration 021. Defensive DEFAULT now() added by
    -- migration 025 so out-of-band INSERTs cannot silently land NULL.
    updated_at         TIMESTAMPTZ NULL DEFAULT now(),
    PRIMARY KEY (date, event_type)
);

-- Audit Wave 3 H7: DB-backed cooldown registry for services.alerts.send_alert
-- so multi-replica deployments share one cooldown window per alert key.
-- See migration 022.
CREATE TABLE IF NOT EXISTS alert_cooldowns (
    cooldown_key TEXT PRIMARY KEY,
    last_alert_ts TIMESTAMPTZ NOT NULL
);

-- Portfolio state and trade tracking for the capital-budgeted strategy.
-- Mirrors migration 007_portfolio_state.sql so that fresh databases
-- bootstrapped from db_schema.sql (which seeds migrations as already-applied
-- without re-executing them) still get these tables.  Both tables also
-- appear in db_reset_all_tables.sql's drop list and in ALL_APP_TABLES.
CREATE TABLE IF NOT EXISTS portfolio_state (
    id            BIGSERIAL       PRIMARY KEY,
    date          DATE            NOT NULL UNIQUE,
    equity_start  DOUBLE PRECISION,
    equity_end    DOUBLE PRECISION,
    month_start_equity DOUBLE PRECISION,
    trades_placed INTEGER         DEFAULT 0,
    lots_per_trade INTEGER        DEFAULT 1,
    daily_pnl     DOUBLE PRECISION DEFAULT 0,
    monthly_stop_active BOOLEAN   DEFAULT false,
    event_signals JSONB           NULL,
    created_at    TIMESTAMPTZ     DEFAULT now()
);

CREATE TABLE IF NOT EXISTS portfolio_trades (
    id                BIGSERIAL       PRIMARY KEY,
    trade_id          BIGINT          REFERENCES trades(trade_id),
    portfolio_state_id BIGINT         REFERENCES portfolio_state(id),
    trade_source      TEXT            NOT NULL DEFAULT 'scheduled',
    event_signal      TEXT            NULL,
    lots              INTEGER         NOT NULL,
    margin_committed  DOUBLE PRECISION,
    realized_pnl      DOUBLE PRECISION,
    equity_before     DOUBLE PRECISION,
    equity_after      DOUBLE PRECISION,
    created_at        TIMESTAMPTZ     DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_portfolio_state_date ON portfolio_state (date);
CREATE INDEX IF NOT EXISTS idx_portfolio_trades_trade_id ON portfolio_trades (trade_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_trades_source ON portfolio_trades (trade_source);

-- Optimizer dashboard: run history, per-config results, walk-forward validation.
CREATE TABLE IF NOT EXISTS optimizer_runs (
    id              SERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL UNIQUE,
    run_name        TEXT,
    git_hash        TEXT,
    config_file     TEXT,
    optimizer_mode  TEXT NOT NULL,
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    num_configs     INTEGER,
    status          TEXT DEFAULT 'running',
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS optimizer_results (
    id              SERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES optimizer_runs(run_id) ON DELETE CASCADE,
    p_starting_capital          DOUBLE PRECISION,
    p_max_trades_per_day        INTEGER,
    p_monthly_drawdown_limit    DOUBLE PRECISION,
    p_lot_per_equity            DOUBLE PRECISION,
    p_max_equity_risk_pct       DOUBLE PRECISION,
    p_max_margin_pct            DOUBLE PRECISION,
    p_calls_only                BOOLEAN,
    p_min_dte                   INTEGER,
    p_max_delta                 DOUBLE PRECISION,
    t_tp_pct                    DOUBLE PRECISION,
    t_sl_mult                   DOUBLE PRECISION,
    t_max_vix                   DOUBLE PRECISION,
    t_max_term_structure        DOUBLE PRECISION,
    t_avoid_opex                BOOLEAN,
    t_prefer_event_days         BOOLEAN,
    t_width_filter              DOUBLE PRECISION,
    t_entry_count               INTEGER,
    e_enabled                   BOOLEAN,
    e_signal_mode               TEXT,
    e_budget_mode               TEXT,
    e_max_event_trades          INTEGER,
    e_spx_drop_threshold        DOUBLE PRECISION,
    e_spx_drop_2d_threshold     DOUBLE PRECISION,
    e_spx_drop_min              DOUBLE PRECISION,
    e_spx_drop_max              DOUBLE PRECISION,
    e_vix_spike_threshold       DOUBLE PRECISION,
    e_vix_elevated_threshold    DOUBLE PRECISION,
    e_term_inversion_threshold  DOUBLE PRECISION,
    e_side_preference           TEXT,
    e_min_dte                   INTEGER,
    e_max_dte                   INTEGER,
    e_min_delta                 DOUBLE PRECISION,
    e_max_delta                 DOUBLE PRECISION,
    e_rally_avoidance           BOOLEAN,
    e_rally_threshold           DOUBLE PRECISION,
    e_event_only                BOOLEAN,
    r_enabled                       BOOLEAN,
    r_high_vix_threshold            DOUBLE PRECISION,
    r_high_vix_multiplier           DOUBLE PRECISION,
    r_extreme_vix_threshold         DOUBLE PRECISION,
    r_big_drop_threshold            DOUBLE PRECISION,
    r_big_drop_multiplier           DOUBLE PRECISION,
    r_consecutive_loss_days         INTEGER,
    r_consecutive_loss_multiplier   DOUBLE PRECISION,
    final_equity        DOUBLE PRECISION,
    return_pct          DOUBLE PRECISION,
    ann_return_pct      DOUBLE PRECISION,
    max_dd_pct          DOUBLE PRECISION,
    trough              DOUBLE PRECISION,
    sharpe              DOUBLE PRECISION,
    total_trades        INTEGER,
    days_traded         INTEGER,
    days_stopped        INTEGER,
    win_days            INTEGER,
    win_rate            DOUBLE PRECISION,
    is_pareto           BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_optimizer_results_run_id
    ON optimizer_results(run_id);
CREATE INDEX IF NOT EXISTS idx_optimizer_results_sharpe
    ON optimizer_results(sharpe DESC);
CREATE INDEX IF NOT EXISTS idx_optimizer_results_pareto
    ON optimizer_results(is_pareto) WHERE is_pareto = TRUE;

CREATE TABLE IF NOT EXISTS optimizer_walkforward (
    id              SERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES optimizer_runs(run_id) ON DELETE CASCADE,
    config_key      TEXT,
    window_label    TEXT,
    train_start     TEXT,
    train_end       TEXT,
    test_start      TEXT,
    test_end        TEXT,
    train_sharpe    DOUBLE PRECISION,
    test_sharpe     DOUBLE PRECISION,
    train_return    DOUBLE PRECISION,
    test_return     DOUBLE PRECISION,
    train_trades    INTEGER,
    test_trades     INTEGER,
    decay_ratio     DOUBLE PRECISION,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_optimizer_walkforward_run_id
    ON optimizer_walkforward(run_id);

-- Tracks which migration files have been applied so they are not replayed.
CREATE TABLE IF NOT EXISTS schema_migrations (
  version    TEXT        PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

