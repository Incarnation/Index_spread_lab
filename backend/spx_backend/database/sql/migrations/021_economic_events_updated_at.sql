-- Audit Wave 2 finding M7: add updated_at column to economic_events
-- so eod_events_job can switch from ON CONFLICT DO NOTHING to
-- ON CONFLICT DO UPDATE without losing change-tracking telemetry.
--
-- Background
-- ----------
-- ``economic_events`` is a static seed table holding FOMC, CPI, OPEX,
-- triple-witching, and similar calendar markers. eod_events_job calls
-- ``services.economic_calendar.generate_rows()`` and writes new rows.
-- Today the upsert is ON CONFLICT (event_date, event_type) DO NOTHING,
-- which means a corrected ``has_projections`` flag (or a re-classified
-- ``is_triple_witching``) for an existing date silently never lands.
--
-- Resolution (per audit decision)
-- -------------------------------
-- Add a nullable ``updated_at TIMESTAMPTZ`` column so the eod_events_job
-- code change (audit Wave 2 M7) can write ON CONFLICT DO UPDATE with
-- ``updated_at = NOW()`` and the operator can audit which rows were
-- ever rewritten.
--
-- Idempotency: schema_migrations tracker ensures single-execution.
-- Lock budget: ``ALTER TABLE ... ADD COLUMN`` with NULL/no default is
-- a metadata-only operation in PG 11+. economic_events is 211 rows;
-- migration completes in milliseconds.
SET lock_timeout = '5s';
SET statement_timeout = '30s';

ALTER TABLE economic_events
    ADD COLUMN updated_at TIMESTAMPTZ NULL;
