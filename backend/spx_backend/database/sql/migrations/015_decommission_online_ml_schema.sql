-- Decommission the online ML pipeline schema (Track A.7).
--
-- Earlier subtasks (A.1 - A.6) deleted the online ML jobs (feature_builder,
-- labeler, trainer, shadow_inference, promotion_gate), their REST routes,
-- the ModelMonitorPage frontend, and the manual-run admin triggers.  That
-- left four tables (`feature_snapshots`, `trade_candidates`,
-- `model_predictions`, `training_runs`) with no live writers and a set of
-- columns on `trade_decisions` / `trades` that are now always NULL.
--
-- This migration removes both: drop the FK constraints, drop the dead
-- columns, drop the dead tables.  `model_versions` is preserved so the
-- offline ML pipeline (`backend/scripts/upload_xgb_model.py`) can still
-- register artifacts when ML is re-introduced on the portfolio path.
--
-- Strict ordering matters:
--   1. Drop FK constraints in preserved tables that reference dropped
--      tables (PostgreSQL forbids dropping a referenced table while the
--      FK still exists).
--   2. Drop the now-orphaned columns in preserved tables.
--   3. Drop the four dead tables in FK-safe order
--      (`model_predictions` -> `trade_candidates` -> `feature_snapshots`,
--      with `training_runs` independent).
--
-- Defense-in-depth: bound how long the migration can sit waiting for
-- ACCESS EXCLUSIVE locks on the live `trade_decisions` / `trades`
-- tables.  Without these guards, a long-running query on either table
-- would block the migration (and, if it ever runs at app boot via
-- `init_db()`, the entire deploy) indefinitely.  These SETs are
-- session-scoped and apply only inside this migration's transaction.
SET lock_timeout = '5s';
SET statement_timeout = '60s';

-- Step 1: drop FK constraints from preserved tables.
ALTER TABLE trade_decisions DROP CONSTRAINT IF EXISTS trade_decisions_candidate_id_fkey;
ALTER TABLE trade_decisions DROP CONSTRAINT IF EXISTS trade_decisions_prediction_id_fkey;
ALTER TABLE trade_decisions DROP CONSTRAINT IF EXISTS trade_decisions_feature_snapshot_id_fkey;
ALTER TABLE trades          DROP CONSTRAINT IF EXISTS trades_candidate_id_fkey;
ALTER TABLE trades          DROP CONSTRAINT IF EXISTS trades_feature_snapshot_id_fkey;

-- Step 2: drop legacy ML columns from `trade_decisions`.  These were
-- only populated by the deprecated online inference path and are always
-- NULL now that the live producer (`_run`) writes only
-- the rules-based decision columns.
ALTER TABLE trade_decisions DROP COLUMN IF EXISTS candidate_id;
ALTER TABLE trade_decisions DROP COLUMN IF EXISTS prediction_id;
ALTER TABLE trade_decisions DROP COLUMN IF EXISTS feature_snapshot_id;
ALTER TABLE trade_decisions DROP COLUMN IF EXISTS model_score;
ALTER TABLE trade_decisions DROP COLUMN IF EXISTS expected_value;
ALTER TABLE trade_decisions DROP COLUMN IF EXISTS policy_version;
ALTER TABLE trade_decisions DROP COLUMN IF EXISTS experiment_tag;
ALTER TABLE trade_decisions DROP COLUMN IF EXISTS risk_gate_json;

-- Step 3: drop legacy ML columns from `trades`.  `_create_trade_from_decision`
-- never writes either column post-decommission.
ALTER TABLE trades DROP COLUMN IF EXISTS candidate_id;
ALTER TABLE trades DROP COLUMN IF EXISTS feature_snapshot_id;

-- Step 4: drop the four dead tables.  Order respects intra-set FKs:
--   model_predictions  -> trade_candidates -> feature_snapshots
--   training_runs       (independent of the candidate/prediction chain)
DROP TABLE IF EXISTS model_predictions;
DROP TABLE IF EXISTS trade_candidates;
DROP TABLE IF EXISTS training_runs;
DROP TABLE IF EXISTS feature_snapshots;
