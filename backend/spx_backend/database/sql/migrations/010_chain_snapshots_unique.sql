-- Migration 010: Add unique constraint on chain_snapshots to prevent duplicate
-- ingestion of the same underlying/expiration/source at the same timestamp.
-- Existing duplicates (keeping highest snapshot_id) should be removed first.

DELETE FROM chain_snapshots WHERE snapshot_id IN (
  SELECT snapshot_id FROM (
    SELECT snapshot_id, ROW_NUMBER() OVER (
      PARTITION BY ts, underlying, expiration, source ORDER BY snapshot_id DESC
    ) AS rn FROM chain_snapshots
  ) sub WHERE rn > 1
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_chain_snapshots_ts_und_exp_src
  ON chain_snapshots (ts, underlying, expiration, source);
