-- Add VVIX and SKEW columns to context_snapshots for XGBoost feature parity.
-- These indices are captured in underlying_quotes via Tradier and stored on
-- context_snapshots so the offline ML toolkit (and any future on-line
-- consumer of context features) can read them without an extra join at
-- decision time.  Originally added for the now-decommissioned
-- feature_builder_job; the columns are still useful as offline training
-- features so the migration is preserved as-is.

ALTER TABLE context_snapshots ADD COLUMN IF NOT EXISTS vvix DOUBLE PRECISION NULL;
ALTER TABLE context_snapshots ADD COLUMN IF NOT EXISTS skew DOUBLE PRECISION NULL;
