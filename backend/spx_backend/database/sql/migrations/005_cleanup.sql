-- Migration 005: Drop redundant index.
-- The unique constraint on users(username) already provides an index,
-- so idx_users_username is redundant.
--
-- Historical note: this migration originally also cleared legacy
-- payload_json blobs in chain_snapshots. That column has since been
-- removed from the base schema and formally dropped by migration 009,
-- so the UPDATE is no longer needed (and would fail on fresh DBs).

DROP INDEX IF EXISTS idx_users_username;
