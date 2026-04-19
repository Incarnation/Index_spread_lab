-- Audit Wave 1 finding C1: cross-source GEX magnitude divergence.
--
-- The TRADIER writer computes per-strike GEX via the canonical
-- SqueezeMetrics formula `OI * gamma * 100 * S^2 * 0.01`; a live-DB
-- probe (2026-04-17 SPX snapshot 91708, spot 7126) confirmed the
-- stored gex_net (5.96e10) equals the canonical recomputation
-- exactly, so TRADIER rows need NO correction.
--
-- The CBOE writer stores mzdata's per-strike `netGamma` series
-- directly. Empirical comparison against the same-date TRADIER value
-- (CBOE gex_net 5.879e8 vs TRADIER 5.96e10 -> ratio 101.4x) showed
-- mzdata publishes exposures in dollars-per-share-per-1%-move and
-- omits the standard 100-share contract multiplier. The Phase 2 code
-- change (services/gex_math.apply_vendor_units, scalar=100) corrected
-- forward-going writes; this migration backfills historical CBOE
-- rows with the same x100 scalar so the time series is continuous
-- across the cutover.
--
-- Scope: gex_snapshots + gex_by_strike + gex_by_expiry_strike for
-- source='CBOE' rows ONLY (TRADIER untouched). Plus context_snapshots
-- columns sourced from CBOE: gex_net_cboe (always) and gex_net (when
-- the row's gex_net_cboe is non-NULL, since cboe_gex_job upserts
-- gex_net = gex_net_cboe in that case).
--
-- VIX rows in CBOE-source tables are dropped by migration 016 BEFORE
-- this migration runs, so the predicates below correctly skip
-- already-deleted data.
--
-- Idempotency: the schema_migrations tracker (see schema.py) ensures
-- this migration runs exactly once. There is no in-SQL idempotency
-- guard; running it twice would 10000x CBOE values and require manual
-- recovery. The framework's run-once guarantee is the safety net.
--
-- Bound migration runtime so it cannot block the deploy indefinitely
-- if locks are contended. The largest target table is
-- gex_by_expiry_strike (~600k rows) and the UPDATE is a simple scalar
-- multiply; well under 60 seconds in production.
SET lock_timeout = '5s';
SET statement_timeout = '120s';

-- Step 1: scale the per-snapshot CBOE aggregates.
UPDATE gex_snapshots
   SET gex_net = gex_net * 100.0,
       gex_calls = gex_calls * 100.0,
       gex_puts = gex_puts * 100.0,
       gex_abs = gex_abs * 100.0
 WHERE source = 'CBOE'
   AND gex_net IS NOT NULL;

-- Step 2: scale the per-snapshot per-strike CBOE rows.
-- Joined to gex_snapshots via snapshot_id so we only touch the strikes
-- attached to a CBOE snapshot. gex_by_strike has no `source` column;
-- the join is the source filter.
UPDATE gex_by_strike AS gbs
   SET gex_net = gbs.gex_net * 100.0,
       gex_calls = gbs.gex_calls * 100.0,
       gex_puts = gbs.gex_puts * 100.0
  FROM gex_snapshots AS gs
 WHERE gs.snapshot_id = gbs.snapshot_id
   AND gs.source = 'CBOE';

-- Step 3: scale the per-snapshot per-expiry-per-strike CBOE rows.
-- Same source-via-join pattern as Step 2.
UPDATE gex_by_expiry_strike AS gbes
   SET gex_net = gbes.gex_net * 100.0,
       gex_calls = gbes.gex_calls * 100.0,
       gex_puts = gbes.gex_puts * 100.0
  FROM gex_snapshots AS gs
 WHERE gs.snapshot_id = gbes.snapshot_id
   AND gs.source = 'CBOE';

-- Step 4: scale context_snapshots' CBOE-derived columns. The
-- gex_net column is also rebased when it was sourced from CBOE
-- (cboe_gex_job upserts `gex_net = EXCLUDED.gex_net_cboe`); the SET
-- expressions in PostgreSQL reference the row's pre-update values, so
-- multiplying both columns by 100 in one statement is safe.
-- zero_gamma_level_cboe and zero_gamma_level are strike-level (in
-- price units, not dollar-GEX) and are NOT multiplied; their
-- correctness is tracked separately in audit Wave 2.
UPDATE context_snapshots
   SET gex_net_cboe = gex_net_cboe * 100.0,
       gex_net = CASE
                   WHEN gex_net_cboe IS NOT NULL THEN gex_net_cboe * 100.0
                   ELSE gex_net
                 END
 WHERE gex_net_cboe IS NOT NULL;
