-- Minimal schema for MVP (expand as needed).
-- Safe to run multiple times (IF NOT EXISTS used).

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
  target_dte INTEGER NOT NULL,
  expiration DATE NOT NULL,
  payload_json JSONB NOT NULL,
  checksum TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chain_snapshots_ts ON chain_snapshots (ts DESC);
CREATE INDEX IF NOT EXISTS idx_chain_snapshots_exp ON chain_snapshots (expiration, ts DESC);

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

CREATE TABLE IF NOT EXISTS gex_snapshots (
  snapshot_id BIGINT PRIMARY KEY REFERENCES chain_snapshots(snapshot_id) ON DELETE CASCADE,
  ts TIMESTAMPTZ NOT NULL,
  underlying TEXT NOT NULL,
  spot_price DOUBLE PRECISION NULL,
  gex_net DOUBLE PRECISION NULL,
  gex_calls DOUBLE PRECISION NULL,
  gex_puts DOUBLE PRECISION NULL,
  gex_abs DOUBLE PRECISION NULL,
  zero_gamma_level DOUBLE PRECISION NULL,
  method TEXT NOT NULL DEFAULT 'oi_gamma_spot'
);

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
  ts TIMESTAMPTZ PRIMARY KEY,
  spx_price DOUBLE PRECISION NULL,
  spy_price DOUBLE PRECISION NULL,
  vix DOUBLE PRECISION NULL,
  vix9d DOUBLE PRECISION NULL,
  term_structure DOUBLE PRECISION NULL,
  gex_net DOUBLE PRECISION NULL,
  zero_gamma_level DOUBLE PRECISION NULL,
  notes_json JSONB NULL
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
  is_active BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  notes TEXT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_versions_name_version ON strategy_versions (strategy_name, version);

CREATE TABLE IF NOT EXISTS model_versions (
  model_version_id BIGSERIAL PRIMARY KEY,
  model_name TEXT NOT NULL,
  version TEXT NOT NULL,
  algorithm TEXT NOT NULL,
  strategy_version_id BIGINT NULL REFERENCES strategy_versions(strategy_version_id),
  feature_spec_json JSONB NOT NULL,
  train_start TIMESTAMPTZ NULL,
  train_end TIMESTAMPTZ NULL,
  val_start TIMESTAMPTZ NULL,
  val_end TIMESTAMPTZ NULL,
  metrics_json JSONB NULL,
  artifact_uri TEXT NULL,
  is_active BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  notes TEXT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_versions_name_version ON model_versions (model_name, version);

CREATE TABLE IF NOT EXISTS training_runs (
  training_run_id BIGSERIAL PRIMARY KEY,
  model_version_id BIGINT NULL REFERENCES model_versions(model_version_id),
  started_at TIMESTAMPTZ NOT NULL,
  finished_at TIMESTAMPTZ NULL,
  status TEXT NOT NULL, -- RUNNING/COMPLETED/FAILED
  config_json JSONB NOT NULL,
  metrics_json JSONB NULL,
  notes TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_training_runs_started_at ON training_runs (started_at DESC);

CREATE TABLE IF NOT EXISTS feature_snapshots (
  feature_snapshot_id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
  snapshot_id BIGINT NULL REFERENCES chain_snapshots(snapshot_id) ON DELETE SET NULL,
  context_ts TIMESTAMPTZ NULL REFERENCES context_snapshots(ts) ON DELETE SET NULL,
  underlying TEXT NOT NULL,
  target_dte INTEGER NULL,
  entry_slot INTEGER NULL,
  strategy_version_id BIGINT NULL REFERENCES strategy_versions(strategy_version_id),
  data_source TEXT NOT NULL DEFAULT 'live', -- live/backtest
  features_json JSONB NOT NULL,
  label_json JSONB NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feature_snapshots_ts ON feature_snapshots (ts DESC);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_snapshot ON feature_snapshots (snapshot_id);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_strategy_ts ON feature_snapshots (strategy_version_id, ts DESC);

CREATE TABLE IF NOT EXISTS trade_candidates (
  candidate_id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
  feature_snapshot_id BIGINT NULL REFERENCES feature_snapshots(feature_snapshot_id) ON DELETE SET NULL,
  snapshot_id BIGINT NULL REFERENCES chain_snapshots(snapshot_id) ON DELETE SET NULL,
  strategy_version_id BIGINT NULL REFERENCES strategy_versions(strategy_version_id),
  candidate_json JSONB NOT NULL,
  constraints_json JSONB NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_trade_candidates_ts ON trade_candidates (ts DESC);
CREATE INDEX IF NOT EXISTS idx_trade_candidates_feature ON trade_candidates (feature_snapshot_id);

CREATE TABLE IF NOT EXISTS model_predictions (
  prediction_id BIGSERIAL PRIMARY KEY,
  candidate_id BIGINT NOT NULL REFERENCES trade_candidates(candidate_id) ON DELETE CASCADE,
  model_version_id BIGINT NOT NULL REFERENCES model_versions(model_version_id) ON DELETE CASCADE,
  score DOUBLE PRECISION NOT NULL,
  decision TEXT NULL,
  meta_json JSONB NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (candidate_id, model_version_id)
);

CREATE INDEX IF NOT EXISTS idx_model_predictions_created_at ON model_predictions (created_at DESC);

CREATE TABLE IF NOT EXISTS backtest_runs (
  run_id BIGSERIAL PRIMARY KEY,
  started_at TIMESTAMPTZ NOT NULL,
  finished_at TIMESTAMPTZ NULL,
  status TEXT NOT NULL, -- RUNNING/COMPLETED/FAILED
  config_json JSONB NOT NULL,
  stats_json JSONB NULL,
  notes TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_backtest_runs_started_at ON backtest_runs (started_at DESC);

CREATE TABLE IF NOT EXISTS strategy_recommendations (
  recommendation_id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  strategy_version_id BIGINT NULL REFERENCES strategy_versions(strategy_version_id),
  model_version_id BIGINT NULL REFERENCES model_versions(model_version_id),
  backtest_run_id BIGINT NULL REFERENCES backtest_runs(run_id),
  proposed_params_json JSONB NOT NULL,
  status TEXT NOT NULL DEFAULT 'PENDING', -- PENDING/APPROVED/REJECTED/APPLIED
  approved_at TIMESTAMPTZ NULL,
  applied_at TIMESTAMPTZ NULL,
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
  backtest_run_id BIGINT NULL REFERENCES backtest_runs(run_id),
  trade_source TEXT NOT NULL DEFAULT 'live', -- live/paper/backtest
  strategy_version_id BIGINT NULL REFERENCES strategy_versions(strategy_version_id),
  model_version_id BIGINT NULL REFERENCES model_versions(model_version_id),
  strategy_type TEXT NOT NULL,
  status TEXT NOT NULL, -- OPEN/CLOSED/ROLLED
  underlying TEXT NOT NULL,
  entry_time TIMESTAMPTZ NOT NULL,
  exit_time TIMESTAMPTZ NULL,
  target_dte INTEGER NULL,
  entry_credit DOUBLE PRECISION NULL,
  max_profit DOUBLE PRECISION NULL,
  current_pnl DOUBLE PRECISION NULL,
  exit_reason TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades (entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_backtest_run ON trades (backtest_run_id, entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_strategy_entry ON trades (strategy_version_id, entry_time DESC);

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

