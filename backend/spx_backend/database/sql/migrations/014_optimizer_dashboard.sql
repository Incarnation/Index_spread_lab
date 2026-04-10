-- Optimizer dashboard tables for storing backtest run history, results,
-- and walk-forward validation outcomes.  Enables the interactive
-- dashboard to query historical optimizer runs from the database.

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

    -- Portfolio config
    p_starting_capital          DOUBLE PRECISION,
    p_max_trades_per_day        INTEGER,
    p_monthly_drawdown_limit    DOUBLE PRECISION,
    p_lot_per_equity            DOUBLE PRECISION,
    p_max_equity_risk_pct       DOUBLE PRECISION,
    p_max_margin_pct            DOUBLE PRECISION,
    p_calls_only                BOOLEAN,
    p_min_dte                   INTEGER,
    p_max_delta                 DOUBLE PRECISION,

    -- Trading config
    t_tp_pct                    DOUBLE PRECISION,
    t_sl_mult                   DOUBLE PRECISION,
    t_max_vix                   DOUBLE PRECISION,
    t_max_term_structure        DOUBLE PRECISION,
    t_avoid_opex                BOOLEAN,
    t_prefer_event_days         BOOLEAN,
    t_width_filter              DOUBLE PRECISION,
    t_entry_count               INTEGER,

    -- Event config
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

    -- Regime throttle config
    r_enabled                       BOOLEAN,
    r_high_vix_threshold            DOUBLE PRECISION,
    r_high_vix_multiplier           DOUBLE PRECISION,
    r_extreme_vix_threshold         DOUBLE PRECISION,
    r_big_drop_threshold            DOUBLE PRECISION,
    r_big_drop_multiplier           DOUBLE PRECISION,
    r_consecutive_loss_days         INTEGER,
    r_consecutive_loss_multiplier   DOUBLE PRECISION,

    -- Result metrics
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

    -- Pareto flag
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

    -- Config identifier (flat key from the results)
    config_key      TEXT,

    -- Walk-forward window
    window_label    TEXT,
    train_start     TEXT,
    train_end       TEXT,
    test_start      TEXT,
    test_end        TEXT,

    -- Metrics
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
