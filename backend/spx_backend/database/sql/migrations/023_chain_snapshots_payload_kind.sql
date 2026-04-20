-- Audit Wave 4 finding M1: disambiguate chain_snapshots payload semantics
-- by introducing a payload_kind column.
--
-- Background
-- ----------
-- Today ``chain_snapshots`` stores rows from two writers with
-- structurally different payloads:
--   * snapshot_job (source='TRADIER') writes a row PER expiration with a
--     full per-option payload in option_chain_rows (~280 rows / snapshot).
--   * cboe_gex_job (source='CBOE') writes a row PER expiration with NO
--     option_chain_rows; the row exists purely as the FK anchor for
--     gex_snapshots / gex_by_strike / gex_by_expiry_strike.
--
-- The ``source`` column is a partial discriminator (TRADIER vs CBOE)
-- but a future writer (e.g. polygon, dxfeed, manual replay) might also
-- be source='TRADIER'-shaped (full chain) or source='CBOE'-shaped
-- (anchor-only), so source alone can't drive read-side dispatch.
--
-- Resolution (per audit decision)
-- -------------------------------
-- Add a ``payload_kind`` column with values:
--   * 'options_chain'  -- has option_chain_rows; snapshot_job writes
--   * 'gex_anchor'     -- no option_chain_rows; cboe_gex_job writes
--
-- Existing rows are seeded by ``source``: TRADIER -> options_chain,
-- CBOE -> gex_anchor. Future writers MUST set payload_kind explicitly
-- (the new ``_chain_snapshot_dao.get_or_insert_anchor`` requires it).
-- A CHECK constraint enforces the closed set so a typo can't slip in.
--
-- Idempotency: schema_migrations tracker ensures single-execution.
-- Lock budget: ALTER TABLE ADD COLUMN with NOT NULL DEFAULT requires
-- a brief table rewrite in PG <= 10, but in PG 11+ a constant default
-- is metadata-only (the rewrite is deferred / avoided). chain_snapshots
-- is ~5.5k rows; even a full rewrite completes in well under 30s.
-- The UPDATE for CBOE rows is bounded to ~1.4k rows (Q5 evidence).
SET lock_timeout = '10s';
SET statement_timeout = '60s';

-- Step 1: add payload_kind defaulted to 'options_chain' (the majority
-- shape). The default lets existing TRADIER rows slot in without an
-- UPDATE; we then explicitly UPDATE the CBOE subset.
ALTER TABLE chain_snapshots
    ADD COLUMN payload_kind TEXT NOT NULL DEFAULT 'options_chain';

-- Step 2: re-classify existing CBOE rows. Future CBOE writes set
-- payload_kind='gex_anchor' explicitly via the DAO; this UPDATE
-- backfills the historical rows so reads can rely on the column from
-- day 1 of the new shape.
UPDATE chain_snapshots
   SET payload_kind = 'gex_anchor'
 WHERE source = 'CBOE';

-- Step 3: lock the discriminator down. A typo in app code (e.g.
-- 'option_chain' singular vs 'options_chain' plural) becomes a hard
-- INSERT failure rather than a silent shape drift.
ALTER TABLE chain_snapshots
    ADD CONSTRAINT chk_chain_snapshots_payload_kind
    CHECK (payload_kind IN ('options_chain', 'gex_anchor'));
