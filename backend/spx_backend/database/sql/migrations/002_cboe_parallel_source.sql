-- Parallel source migration for Tradier + CBOE ingestion.
-- This migration is additive/non-destructive and safe to rerun.

ALTER TABLE chain_snapshots ADD COLUMN IF NOT EXISTS source TEXT;
UPDATE chain_snapshots
SET source = 'TRADIER'
WHERE source IS NULL OR BTRIM(source) = '';
UPDATE chain_snapshots
SET source = UPPER(BTRIM(source))
WHERE source IS NOT NULL;
ALTER TABLE chain_snapshots ALTER COLUMN source SET DEFAULT 'TRADIER';
ALTER TABLE chain_snapshots ALTER COLUMN source SET NOT NULL;

ALTER TABLE gex_snapshots ADD COLUMN IF NOT EXISTS source TEXT;
UPDATE gex_snapshots
SET source = 'TRADIER'
WHERE source IS NULL OR BTRIM(source) = '';
UPDATE gex_snapshots
SET source = UPPER(BTRIM(source))
WHERE source IS NOT NULL;
ALTER TABLE gex_snapshots ALTER COLUMN source SET DEFAULT 'TRADIER';
ALTER TABLE gex_snapshots ALTER COLUMN source SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_chain_snapshots_source_ts ON chain_snapshots (source, ts DESC);
CREATE INDEX IF NOT EXISTS idx_chain_snapshots_underlying_source_ts ON chain_snapshots (underlying, source, ts DESC);
CREATE INDEX IF NOT EXISTS idx_gex_snapshots_source_ts ON gex_snapshots (source, ts DESC);
CREATE INDEX IF NOT EXISTS idx_gex_snapshots_underlying_source_ts ON gex_snapshots (underlying, source, ts DESC);
