-- Migration 009: Drop legacy payload_json column from chain_snapshots.
-- Data was fully normalized into option_chain_rows. Migration 005 already
-- cleared the blobs. This removes the column entirely.

ALTER TABLE chain_snapshots DROP COLUMN IF EXISTS payload_json;
