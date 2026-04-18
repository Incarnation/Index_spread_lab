-- Destructive reset of all application tables.
-- Recreate by running init_db() after this script.

DROP TABLE IF EXISTS trade_marks;
DROP TABLE IF EXISTS trade_performance_equity_curve;
DROP TABLE IF EXISTS trade_performance_breakdowns;
DROP TABLE IF EXISTS trade_performance_snapshots;
DROP TABLE IF EXISTS portfolio_trades;
DROP TABLE IF EXISTS portfolio_state;
DROP TABLE IF EXISTS trade_legs;
DROP TABLE IF EXISTS fills;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS trades;
DROP TABLE IF EXISTS trade_decisions;
DROP TABLE IF EXISTS strategy_recommendations;
DROP TABLE IF EXISTS backtest_runs;
DROP TABLE IF EXISTS model_versions;
DROP TABLE IF EXISTS strategy_versions;

DROP TABLE IF EXISTS gex_by_expiry_strike;
DROP TABLE IF EXISTS gex_by_strike;
DROP TABLE IF EXISTS gex_snapshots;

DROP TABLE IF EXISTS option_chain_rows;
DROP TABLE IF EXISTS option_instruments;
DROP TABLE IF EXISTS chain_snapshots;

DROP TABLE IF EXISTS context_snapshots;
DROP TABLE IF EXISTS underlying_quotes;
DROP TABLE IF EXISTS market_clock_audit;
DROP TABLE IF EXISTS economic_events;

DROP TABLE IF EXISTS optimizer_walkforward;
DROP TABLE IF EXISTS optimizer_results;
DROP TABLE IF EXISTS optimizer_runs;

DROP TABLE IF EXISTS auth_audit_log;
DROP TABLE IF EXISTS users;
