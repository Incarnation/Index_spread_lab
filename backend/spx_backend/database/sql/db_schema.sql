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
  checksum TEXT NOT NULL
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
  gex_net DOUBLE PRECISION NULL,
  zero_gamma_level DOUBLE PRECISION NULL,
  notes_json JSONB NULL,
  PRIMARY KEY (ts, underlying)
);

CREATE TABLE IF NOT EXISTS underlying_quotes (
  quote_id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
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

CREATE TABLE IF NOT EXISTS training_runs (
  training_run_id BIGSERIAL PRIMARY KEY,
  model_version_id BIGINT NULL REFERENCES model_versions(model_version_id),
  started_at TIMESTAMPTZ NOT NULL,
  finished_at TIMESTAMPTZ NULL,
  run_type TEXT NOT NULL DEFAULT 'walk_forward', -- walk_forward/retrain/backfill
  status TEXT NOT NULL, -- RUNNING/COMPLETED/FAILED
  walkforward_fold INTEGER NULL,
  train_window_start TIMESTAMPTZ NULL,
  train_window_end TIMESTAMPTZ NULL,
  test_window_start TIMESTAMPTZ NULL,
  test_window_end TIMESTAMPTZ NULL,
  rows_train BIGINT NULL,
  rows_val BIGINT NULL,
  rows_test BIGINT NULL,
  leakage_checks_json JSONB NULL,
  artifacts_json JSONB NULL,
  config_json JSONB NOT NULL,
  metrics_json JSONB NULL,
  notes TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_training_runs_started_at ON training_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs (status, started_at DESC);

CREATE TABLE IF NOT EXISTS feature_snapshots (
  feature_snapshot_id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
  snapshot_id BIGINT NULL REFERENCES chain_snapshots(snapshot_id) ON DELETE SET NULL,
  context_ts TIMESTAMPTZ NULL,
  underlying TEXT NOT NULL,
  target_dte INTEGER NULL,
  entry_slot INTEGER NULL,
  strategy_version_id BIGINT NULL REFERENCES strategy_versions(strategy_version_id),
  data_source TEXT NOT NULL DEFAULT 'live', -- live/backtest
  feature_schema_version TEXT NOT NULL DEFAULT 'fs_v1',
  feature_hash TEXT NULL,
  source_job TEXT NULL,
  source_run_id BIGINT NULL,
  features_json JSONB NOT NULL,
  label_json JSONB NULL,
  label_schema_version TEXT NULL,
  label_status TEXT NOT NULL DEFAULT 'pending', -- pending/resolved/error/expired
  label_horizon TEXT NULL,
  resolved_at TIMESTAMPTZ NULL,
  label_error TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feature_snapshots_ts ON feature_snapshots (ts DESC);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_snapshot ON feature_snapshots (snapshot_id);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_strategy_ts ON feature_snapshots (strategy_version_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_label_status ON feature_snapshots (label_status, ts DESC);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_feature_hash ON feature_snapshots (feature_hash);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_underlying_ts ON feature_snapshots (underlying, ts DESC);

CREATE TABLE IF NOT EXISTS trade_candidates (
  candidate_id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
  feature_snapshot_id BIGINT NULL REFERENCES feature_snapshots(feature_snapshot_id) ON DELETE SET NULL,
  snapshot_id BIGINT NULL REFERENCES chain_snapshots(snapshot_id) ON DELETE SET NULL,
  strategy_version_id BIGINT NULL REFERENCES strategy_versions(strategy_version_id),
  candidate_hash TEXT NOT NULL,
  candidate_schema_version TEXT NOT NULL DEFAULT 'cand_v1',
  candidate_rank INTEGER NULL,
  entry_credit DOUBLE PRECISION NULL,
  max_loss DOUBLE PRECISION NULL,
  credit_to_width DOUBLE PRECISION NULL,
  candidate_json JSONB NOT NULL,
  constraints_json JSONB NULL,
  label_json JSONB NULL,
  label_schema_version TEXT NULL,
  label_status TEXT NOT NULL DEFAULT 'pending', -- pending/resolved/error/expired
  label_horizon TEXT NULL,
  resolved_at TIMESTAMPTZ NULL,
  realized_pnl DOUBLE PRECISION NULL,
  hit_tp50_before_sl_or_expiry BOOLEAN NULL,
  hit_sl_before_tp_or_expiry BOOLEAN NULL,
  hit_tp100_at_expiry BOOLEAN NULL,
  label_error TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_trade_candidates_ts ON trade_candidates (ts DESC);
CREATE INDEX IF NOT EXISTS idx_trade_candidates_feature ON trade_candidates (feature_snapshot_id);
CREATE INDEX IF NOT EXISTS idx_trade_candidates_label_status ON trade_candidates (label_status, ts DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_trade_candidates_snapshot_hash ON trade_candidates (feature_snapshot_id, candidate_hash);

CREATE TABLE IF NOT EXISTS model_predictions (
  prediction_id BIGSERIAL PRIMARY KEY,
  candidate_id BIGINT NOT NULL REFERENCES trade_candidates(candidate_id) ON DELETE CASCADE,
  model_version_id BIGINT NOT NULL REFERENCES model_versions(model_version_id) ON DELETE CASCADE,
  prediction_schema_version TEXT NOT NULL DEFAULT 'pred_v1',
  score_raw DOUBLE PRECISION NOT NULL,
  score_calibrated DOUBLE PRECISION NULL,
  probability_win DOUBLE PRECISION NULL,
  expected_value DOUBLE PRECISION NULL,
  threshold_used DOUBLE PRECISION NULL,
  rank_in_snapshot INTEGER NULL,
  decision TEXT NULL,
  meta_json JSONB NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (candidate_id, model_version_id)
);

CREATE INDEX IF NOT EXISTS idx_model_predictions_created_at ON model_predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_predictions_candidate_created ON model_predictions (candidate_id, created_at DESC);

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
  model_score DOUBLE PRECISION NULL,
  expected_value DOUBLE PRECISION NULL,
  decision TEXT NOT NULL, -- 'TRADE'/'SKIP'
  reason TEXT NULL,
  policy_version TEXT NULL,
  risk_gate_json JSONB NULL,
  experiment_tag TEXT NULL,
  chain_snapshot_id BIGINT NULL REFERENCES chain_snapshots(snapshot_id),
  strategy_version_id BIGINT NULL REFERENCES strategy_versions(strategy_version_id),
  model_version_id BIGINT NULL REFERENCES model_versions(model_version_id),
  feature_snapshot_id BIGINT NULL REFERENCES feature_snapshots(feature_snapshot_id),
  candidate_id BIGINT NULL REFERENCES trade_candidates(candidate_id),
  prediction_id BIGINT NULL REFERENCES model_predictions(prediction_id),
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
  candidate_id BIGINT NULL REFERENCES trade_candidates(candidate_id),
  feature_snapshot_id BIGINT NULL REFERENCES feature_snapshots(feature_snapshot_id),
  entry_snapshot_id BIGINT NULL REFERENCES chain_snapshots(snapshot_id),
  last_snapshot_id BIGINT NULL REFERENCES chain_snapshots(snapshot_id),
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
    PRIMARY KEY (date, event_type)
);

-- Tracks which migration files have been applied so they are not replayed.
CREATE TABLE IF NOT EXISTS schema_migrations (
  version    TEXT        PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

