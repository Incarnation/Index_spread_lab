-- Audit Wave 2 finding H6: split ingest-time from vendor-time on
-- underlying_quotes by adding a nullable vendor_ts column.
--
-- Background
-- ----------
-- ``underlying_quotes.ts`` is currently set by quote_job to
-- ``datetime.now(tz=ET).astimezone(UTC)``, i.e. the moment the row
-- LANDED in the DB rather than the moment Tradier observed the quote.
-- This conflation has two side effects:
--   * 5-call-site ``MAX(ts)`` staleness probes report freshness
--     against ingest time (right answer for monitoring; see
--     staleness_monitor_job).
--   * 6-call-site ``WHERE ts <= :ts ORDER BY ts DESC LIMIT 1`` as-of
--     spot lookups (snapshot_job, gex_job, cboe_gex_job context
--     upserts, trade_pnl_job, decision_job) use ingest time too,
--     which can pick a slightly newer row than the vendor's actual
--     observation cutoff. Acceptable in practice but technically wrong.
--
-- Resolution (per audit decision)
-- -------------------------------
-- Add ``vendor_ts TIMESTAMPTZ NULL``, populated by quote_job from
-- Tradier's ``trade_date`` ms-since-epoch field. ``ts`` keeps its
-- ingest-time semantics so the existing 5 monitoring sites stay
-- correct without code change. As-of consumers switch to
-- ``COALESCE(vendor_ts, ts)`` so historical rows (vendor_ts NULL)
-- continue to work and forward-going reads use the more accurate
-- vendor timestamp.
--
-- A functional index on (symbol, COALESCE(vendor_ts, ts) DESC)
-- supports the new ORDER BY shape; without it the as-of query plan
-- would degrade to a sequential scan post-cutover.
--
-- Idempotency: schema_migrations tracker enforces single-execution.
-- Lock budget: ALTER TABLE ... ADD COLUMN with a NULLABLE column
-- and no default is metadata-only in PG 11+; the index build is the
-- dominant cost (~13.4k rows, sub-second).
SET lock_timeout = '5s';
SET statement_timeout = '60s';

-- Step 1: add the new vendor_ts column. NULLABLE on purpose; backfill
-- of historical rows would require Tradier replay (we don't have that
-- data) so historical rows keep vendor_ts=NULL and the COALESCE
-- fallback makes them indistinguishable from "ingest-time IS the
-- vendor time" for downstream readers.
ALTER TABLE underlying_quotes
    ADD COLUMN vendor_ts TIMESTAMPTZ NULL;

-- Step 2: functional index supporting ``ORDER BY COALESCE(vendor_ts, ts) DESC``.
-- The expression in the index must match the expression in the query
-- exactly (down to argument order and parenthesization) for PG to use
-- the index. Mirrors the existing idx_underlying_quotes_symbol_ts but
-- on the COALESCE expression.
CREATE INDEX idx_underlying_quotes_symbol_vendor_or_ts
    ON underlying_quotes (symbol, (COALESCE(vendor_ts, ts)) DESC);
