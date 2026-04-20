-- Audit Wave 2 finding M2: make context_snapshots.gex_net a GENERATED
-- column derived from the source-tagged columns.
--
-- Background
-- ----------
-- ``context_snapshots`` carries three GEX columns:
--   * gex_net_tradier  -- written by gex_job (Tradier-chain-derived)
--   * gex_net_cboe     -- written by cboe_gex_job (mzdata-derived)
--   * gex_net          -- canonical aggregate consumed by decision_job,
--                         OFFLINE_ML feature builder, exporter, etc.
--
-- Today the canonical column is filled by app code:
--   * quote_job upserts a new row with ``gex_net = NULL`` and uses
--     ``COALESCE(context_snapshots.gex_net, EXCLUDED.gex_net)`` to
--     preserve any prior value.
--   * gex_job (Tradier writer) sets ``gex_net = EXCLUDED.gex_net_tradier``.
--   * cboe_gex_job (CBOE writer) sets ``gex_net = EXCLUDED.gex_net_cboe``.
--
-- Race window: when CBOE finishes BEFORE Tradier for the same
-- (ts, underlying) bucket (~33% of rows per Q25 evidence), the
-- canonical ``gex_net`` becomes whichever writer landed second,
-- overwriting the other source. Decision_job then reads a
-- non-deterministic source-mix.
--
-- Resolution (per audit decision)
-- -------------------------------
-- Promote the precedence rule into the schema as a STORED generated
-- column: ``gex_net = COALESCE(gex_net_cboe, gex_net_tradier)``. This
-- makes CBOE the canonical source whenever it has a value, with
-- Tradier as the fallback. App writers must stop touching ``gex_net``
-- directly (PG rejects writes to generated columns with error 428C9).
--
-- Migration steps:
--   1. DROP the existing ``gex_net`` column (and its mirror NULL data).
--   2. ADD a new ``gex_net DOUBLE PRECISION GENERATED ALWAYS AS
--      (COALESCE(gex_net_cboe, gex_net_tradier)) STORED`` column.
--      All existing rows are back-populated automatically.
--
-- Idempotency: schema_migrations tracker ensures single-execution.
-- Re-running would error on the DROP (column doesn't exist) but the
-- runner prevents that.
--
-- Lock budget: context_snapshots is ~10.2k rows. The DROP is a
-- metadata operation; the ADD with a STORED expression rewrites the
-- table column-wise (PG 12+ optimization). Both complete in well
-- under 60 seconds.
--
-- Compatibility: any view/index referencing the OLD ``gex_net``
-- column would block the DROP. live-DB inspection (Q5) confirmed
-- no views reference context_snapshots and no index includes
-- ``gex_net``, so the DROP is safe.
SET lock_timeout = '10s';
SET statement_timeout = '120s';

-- Step 1: drop the existing column. PG cascades to dependent objects
-- but, per pre-flight inspection, none exist. CASCADE is conservative
-- belt-and-braces.
ALTER TABLE context_snapshots
    DROP COLUMN gex_net CASCADE;

-- Step 2: re-add as GENERATED. The expression mirrors the precedence
-- the app writers used to enforce; CBOE wins when both are present,
-- otherwise Tradier. Previously NULL rows where neither source has
-- written keep gex_net NULL (COALESCE of two NULLs).
ALTER TABLE context_snapshots
    ADD COLUMN gex_net DOUBLE PRECISION
        GENERATED ALWAYS AS (COALESCE(gex_net_cboe, gex_net_tradier)) STORED;
