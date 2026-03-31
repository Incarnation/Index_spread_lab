-- Add VVIX and SKEW columns to context_snapshots for XGBoost feature parity.
-- These indices are captured in underlying_quotes via Tradier and stored in
-- context_snapshots makes them available to feature_builder_job without an
-- extra join at decision time.

ALTER TABLE context_snapshots ADD COLUMN IF NOT EXISTS vvix DOUBLE PRECISION NULL;
ALTER TABLE context_snapshots ADD COLUMN IF NOT EXISTS skew DOUBLE PRECISION NULL;
