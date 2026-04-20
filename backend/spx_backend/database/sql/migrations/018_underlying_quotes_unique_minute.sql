-- Audit Wave 2 finding H5: deduplicate underlying_quotes near-duplicates
-- and add a UNIQUE expression index on (symbol, minute_bucket).
--
-- Background
-- ----------
-- The Wave 1 audit live-DB probe (Q14) confirmed 10+ near-duplicate
-- (symbol, minute_bucket) rows on 2026-04-06 alone, all from quote_job
-- retries that landed within the same minute window. The table has only
-- a BIGSERIAL surrogate PK and a non-unique (symbol, ts DESC) helper
-- index, so quote_job retries are NOT idempotent: a transient connection
-- blip + APScheduler retry can produce two rows for the same (SPX,
-- 14:31) bucket, both committed.
--
-- This migration:
--   1. Hard-deletes the older (lower quote_id) row in any (symbol,
--      minute_bucket) duplicate cluster, keeping the most recent
--      ingest. Acceptable because the rows are functionally identical
--      (same symbol, same minute, snapshot of the same quote stream)
--      and downstream readers all use ORDER BY ts DESC LIMIT 1.
--   2. Adds a UNIQUE expression index on
--      (symbol, date_bin('1 minute', ts, TIMESTAMPTZ '2000-01-01 00:00:00+00')).
--      Forward-going writes from quote_job use the matching ON CONFLICT
--      expression so retries become idempotent without code-side
--      coordination.
--
-- IMMUTABLE-required note
-- -----------------------
-- PostgreSQL forbids STABLE/VOLATILE expressions in unique indexes.
-- ``date_trunc('minute', ts)`` is STABLE on ``timestamp with time
-- zone`` (its result depends on the session's TimeZone), so it cannot
-- be used here. ``date_bin(stride, source, origin)`` (PG 14+) is
-- IMMUTABLE: it always bins the absolute moment relative to the fixed
-- ``origin`` regardless of session timezone. Same semantic as
-- date_trunc('minute', ...) for sub-day strides, no timezone gotcha.
--
-- Idempotency: schema_migrations tracker (see schema.py) ensures this
-- runs exactly once. Re-running the DELETE would be a no-op (the
-- cluster has been collapsed); re-running the CREATE UNIQUE INDEX
-- would fail (index already exists), but the framework prevents that.
--
-- Lock budget: underlying_quotes is small (~13.4k rows pre-dedup; we
-- expect ~50-100 row deletions). The bulk DELETE + concurrent-safe
-- CREATE UNIQUE INDEX (note: NOT using CONCURRENTLY since the
-- migration runner wraps in a transaction) take milliseconds; the
-- whole migration is well under one second on production.
SET lock_timeout = '5s';
SET statement_timeout = '60s';

-- Step 1: collapse near-duplicate clusters by deleting all-but-the-most-recent
-- row in each (symbol, minute_bucket) group. ``a.quote_id < b.quote_id``
-- removes the older rows; the latest row (largest quote_id) survives.
-- Reading consumers (snapshot_job._get_spot_price, decision_job, etc.) use
-- ORDER BY ts DESC LIMIT 1 so dropping older rows in the same minute is a
-- no-op for them. We use date_bin (IMMUTABLE) here for symmetry with the
-- index expression in step 2; date_trunc would also work in DELETE because
-- the planner doesn't require IMMUTABILITY in WHERE clauses, but using the
-- same expression in both places keeps the migration readable.
DELETE FROM underlying_quotes a
      USING underlying_quotes b
      WHERE a.quote_id < b.quote_id
        AND a.symbol = b.symbol
        AND date_bin('1 minute', a.ts, TIMESTAMPTZ '2000-01-01 00:00:00+00')
          = date_bin('1 minute', b.ts, TIMESTAMPTZ '2000-01-01 00:00:00+00');

-- Step 2: enforce idempotency forward-going via a UNIQUE expression index.
-- Targeted by ON CONFLICT (symbol, (date_bin('1 minute', ts, TIMESTAMPTZ
-- '2000-01-01 00:00:00+00'))) DO NOTHING in quote_job. The expression
-- must match the index expression EXACTLY (down to argument order and
-- parenthesization) so the planner picks this index for upsert.
CREATE UNIQUE INDEX uq_underlying_quotes_symbol_ts_minute
    ON underlying_quotes (symbol, (date_bin('1 minute', ts, TIMESTAMPTZ '2000-01-01 00:00:00+00')));
