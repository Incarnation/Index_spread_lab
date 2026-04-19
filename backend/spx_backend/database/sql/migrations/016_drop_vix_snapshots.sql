-- Audit Wave 1 finding H3: hard-delete legacy VIX snapshot + GEX rows.
--
-- Wave 1 removes the VIX snapshot/GEX pipeline entirely (vix_snapshot_job
-- factory, 12 VIX_SNAPSHOT_* settings, scheduler wiring, CBOE_GEX_UNDERLYINGS
-- default). Forward-going writes will no longer produce underlying='VIX'
-- rows in chain_snapshots, gex_snapshots, gex_by_strike, gex_by_expiry_strike,
-- or option_chain_rows. This migration removes the stale historical rows so
-- they do not pollute downstream queries (regime ingester, optimizer scans,
-- staleness monitor) and so that the Wave 1 C1 unit-correction migration
-- (017_correct_cboe_gex_units.sql) does not waste UPDATEs on rows that are
-- about to be dropped.
--
-- IMPORTANT scope:
--   * `underlying_quotes` rows where symbol='VIX' are KEPT. Those are the
--     spot VIX index quotes consumed by quote_job and read by
--     context_snapshots.vix. The user explicitly requested preserving them.
--   * `context_snapshots.vix` (a column on the SPX-keyed row, not a
--     separate VIX-keyed row) is also KEPT for the same reason. context_snapshots
--     is keyed on (ts, underlying) with default underlying='SPX', so there
--     are no underlying='VIX' rows in that table to worry about.
--
-- Cascade strategy: chain_snapshots is the parent of option_chain_rows,
-- gex_snapshots, gex_by_strike, and gex_by_expiry_strike via FK ON DELETE
-- CASCADE (see db_schema.sql lines 58-130). A single DELETE FROM
-- chain_snapshots WHERE underlying='VIX' therefore removes all four child
-- tables in one statement and guarantees no orphan rows.
--
-- Idempotency: the schema_migrations tracker (see schema.py) ensures this
-- migration runs exactly once. Running it twice would be a no-op (the
-- WHERE clause matches nothing on the second pass) but should not occur.
--
-- Lock budget: chain_snapshots VIX subset is small (DTE 14/21 daily writes
-- for a few months) compared to the SPX side, so the cascade fan-out into
-- option_chain_rows is the dominant cost. Bounded conservatively below.
SET lock_timeout = '5s';
SET statement_timeout = '120s';

-- Single statement: delete all VIX-underlying chain_snapshots rows.
-- ON DELETE CASCADE on the four child tables propagates the delete to:
--   * option_chain_rows
--   * gex_snapshots         (both source='TRADIER' and source='CBOE')
--   * gex_by_strike
--   * gex_by_expiry_strike
DELETE FROM chain_snapshots
 WHERE underlying = 'VIX';
