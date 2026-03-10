-- Migration 005: Drop redundant index and clear legacy payload_json blobs.
-- The unique constraint on users(username) already provides an index;
-- idx_users_username is redundant.
-- payload_json in chain_snapshots is no longer written (data is normalized
-- into option_chain_rows), so existing blobs are replaced with empty JSON
-- to reclaim storage.

DROP INDEX IF EXISTS idx_users_username;

UPDATE chain_snapshots
SET payload_json = '{}'::jsonb
WHERE payload_json != '{}'::jsonb;
