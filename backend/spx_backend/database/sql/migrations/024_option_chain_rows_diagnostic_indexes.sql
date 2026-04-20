-- +migrate-no-transaction
-- Audit Wave 4 finding M8: add three small partial indexes on
-- option_chain_rows to make incident-response diagnostic queries
-- cheap (sub-second) instead of full-scan (multi-minute on
-- 10.5M rows / 16GB -- live probe value, see "Lock budget" below).
--
-- Background
-- ----------
-- The Wave 1 audit's evidence-gathering queries (Q23 -- empty
-- snapshot detection; Q24 -- NULL option_right counting; Q26 -- NULL
-- greek detection) all required full scans of option_chain_rows
-- because no index covers the (snapshot_id) column let alone the
-- partial WHERE filters. The original audit estimated 280k rows;
-- the live probe (Waves 2-6 execution log, snag #2) found the
-- production table at ~10.5M rows / 16GB, so unindexed scans are
-- now multi-minute and effectively unusable during an incident.
--
-- Resolution (per audit decision)
-- -------------------------------
-- Three partial indexes targeting the exact WHERE clauses the audit's
-- diagnostic SQL uses:
--   * (snapshot_id) WHERE option_right IS NULL  -- Q24 evidence
--   * (snapshot_id) WHERE delta IS NULL          -- Q26 evidence
--   * (snapshot_id) WHERE bid IS NULL AND ask IS NULL -- mid-quote check
-- All are partial indexes so the index size stays small (only the
-- "bad" rows are indexed; ~5% of the table per Q24 evidence).
--
-- Idempotency: schema_migrations tracker ensures single-execution.
-- IF NOT EXISTS makes a re-run safe in case the migration is
-- replayed manually outside the framework.
--
-- Lock budget (revised after live-DB probe)
-- ----------------------------------------
-- The live table is 10.5M rows / 16GB. A regular CREATE INDEX would
-- hold a SHARE lock and exceed any reasonable statement_timeout (the
-- first attempt timed out at 60s). Switching to CREATE INDEX
-- CONCURRENTLY builds without blocking writers; the trade-off is
-- that CONCURRENTLY is forbidden inside a transaction.
--
-- The ``-- +migrate-no-transaction`` marker on the very first line
-- opts this file into the runner's autocommit code path
-- (``_execute_sql_file`` in ``backend/spx_backend/database/schema.py``),
-- so the in-app deploy-time migrator can apply this file safely
-- without manual intervention. Each statement commits independently;
-- the runner records the file as applied in ``schema_migrations``
-- only after all statements succeed.
--
-- IMPORTANT: do NOT add ``SET statement_timeout`` here -- CONCURRENTLY
-- can take many minutes on a 16GB table and the runner uses the
-- session default (typically unlimited) instead.

-- Diagnostic index #1: snapshots with at least one row missing the
-- call/put discriminator. Q24 evidence shows ~5% of option_chain_rows
-- have option_right NULL (Tradier emits sparse rows for far-OTM
-- weeklies). Used by the operator to spot snapshot-level data drift.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_option_chain_rows_null_right
    ON option_chain_rows (snapshot_id)
    WHERE option_right IS NULL;

-- Diagnostic index #2: snapshots with NULL delta on at least one row.
-- Used by the operator to spot greek-pipeline regressions; gex_job's
-- Tradier writer gates on delta IS NOT NULL so missing greeks reduce
-- the GEX coverage of a snapshot.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_option_chain_rows_null_delta
    ON option_chain_rows (snapshot_id)
    WHERE delta IS NULL;

-- Diagnostic index #3: snapshots where Tradier returned a row with
-- both bid AND ask NULL (no two-sided market). Used to spot
-- post-hours / illiquid-strike noise in pre-market / after-hours
-- snapshots when allow_outside_rth is on.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_option_chain_rows_null_bidask
    ON option_chain_rows (snapshot_id)
    WHERE bid IS NULL AND ask IS NULL;
