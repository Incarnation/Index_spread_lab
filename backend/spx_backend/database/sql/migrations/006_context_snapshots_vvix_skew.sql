-- Add VVIX and SKEW columns to context_snapshots for XGBoost feature parity.
-- These indices are captured in underlying_quotes via Tradier and stored on
-- context_snapshots so the offline ML toolkit (and any future on-line
-- consumer of context features) can read them without an extra join at
-- decision time.  Originally added for the online ML feature builder; the
-- columns remain useful as offline training inputs.

ALTER TABLE context_snapshots ADD COLUMN IF NOT EXISTS vvix DOUBLE PRECISION NULL;
ALTER TABLE context_snapshots ADD COLUMN IF NOT EXISTS skew DOUBLE PRECISION NULL;
