-- Migration 008: Add underlying column to context_snapshots, change PK to (ts, underlying).
-- Existing rows default to 'SPX'. The FK from feature_snapshots.context_ts is dropped
-- because ts alone is no longer unique after this migration.

ALTER TABLE feature_snapshots DROP CONSTRAINT IF EXISTS feature_snapshots_context_ts_fkey;

ALTER TABLE context_snapshots ADD COLUMN IF NOT EXISTS underlying TEXT NOT NULL DEFAULT 'SPX';

ALTER TABLE context_snapshots DROP CONSTRAINT IF EXISTS context_snapshots_pkey;

ALTER TABLE context_snapshots ADD PRIMARY KEY (ts, underlying);
