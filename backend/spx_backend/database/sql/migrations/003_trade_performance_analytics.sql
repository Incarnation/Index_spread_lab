-- Add aggregate tables backing the performance analytics dashboard.
-- Non-destructive and idempotent for safe repeated startup migrations.

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
