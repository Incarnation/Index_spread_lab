-- Portfolio state and trade tracking for capital-budgeted strategy.
-- Tracks daily equity, lot sizing, drawdown stops, and event signals.

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

COMMENT ON TABLE strategy_recommendations IS 'DEPRECATED: superseded by portfolio_trades + decision_job portfolio-manager integration.';
