-- Destructive reset of ML/decision/trade tables.
-- This keeps ingestion/market-data tables intact (snapshots, option rows, quotes, GEX, context).

DROP TABLE IF EXISTS trade_performance_equity_curve;
DROP TABLE IF EXISTS trade_performance_breakdowns;
DROP TABLE IF EXISTS trade_performance_snapshots;
DROP TABLE IF EXISTS trade_legs;
DROP TABLE IF EXISTS trade_marks;
DROP TABLE IF EXISTS fills;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS trades;
DROP TABLE IF EXISTS trade_decisions;
DROP TABLE IF EXISTS model_predictions;
DROP TABLE IF EXISTS trade_candidates;
DROP TABLE IF EXISTS feature_snapshots;
DROP TABLE IF EXISTS strategy_recommendations;
DROP TABLE IF EXISTS backtest_runs;
DROP TABLE IF EXISTS training_runs;
DROP TABLE IF EXISTS model_versions;
DROP TABLE IF EXISTS strategy_versions;
