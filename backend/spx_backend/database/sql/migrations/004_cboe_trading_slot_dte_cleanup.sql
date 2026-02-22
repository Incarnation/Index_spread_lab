-- Align CBOE snapshot DTE semantics to Tradier-style trading-slot indexing.
-- Then prune CBOE history outside the supported 0..10 window.
-- Safe to rerun: updates are deterministic and delete is bounded.

-- Reset CBOE target_dte to sentinel -1 so stale pre-migration values are replaced
-- deterministically before we upsert computed trading-slot DTE values.
UPDATE chain_snapshots
SET target_dte = -1
WHERE source = 'CBOE';

-- Recompute CBOE target_dte using trading-slot indexing per batch:
-- key = (ts, underlying), as_of date = ET trading date of batch timestamp.
WITH source_rows AS (
  SELECT
    cs.snapshot_id,
    cs.ts,
    cs.underlying,
    cs.expiration,
    (cs.ts AT TIME ZONE 'America/New_York')::date AS as_of_date
  FROM chain_snapshots cs
  WHERE cs.source = 'CBOE'
),
future_expirations AS (
  SELECT DISTINCT
    sr.ts,
    sr.underlying,
    sr.as_of_date,
    sr.expiration
  FROM source_rows sr
  WHERE sr.expiration >= sr.as_of_date
),
ranked_expirations AS (
  SELECT
    fe.ts,
    fe.underlying,
    fe.as_of_date,
    fe.expiration,
    DENSE_RANK() OVER (PARTITION BY fe.ts, fe.underlying ORDER BY fe.expiration) AS expiration_rank,
    MAX(CASE WHEN fe.expiration = fe.as_of_date THEN 1 ELSE 0 END)
      OVER (PARTITION BY fe.ts, fe.underlying) AS has_same_day
  FROM future_expirations fe
),
resolved_dte AS (
  SELECT
    re.ts,
    re.underlying,
    re.expiration,
    CASE
      WHEN re.has_same_day = 1 THEN re.expiration_rank - 1
      ELSE re.expiration_rank
    END AS trading_slot_dte
  FROM ranked_expirations re
)
UPDATE chain_snapshots cs
SET target_dte = rd.trading_slot_dte
FROM resolved_dte rd
WHERE cs.source = 'CBOE'
  AND cs.ts = rd.ts
  AND cs.underlying = rd.underlying
  AND cs.expiration = rd.expiration;

-- Keep expiry-level labels in sync with recomputed chain snapshot DTE.
UPDATE gex_by_expiry_strike gbes
SET dte_days = cs.target_dte
FROM chain_snapshots cs
WHERE cs.source = 'CBOE'
  AND gbes.snapshot_id = cs.snapshot_id;

-- Prune legacy CBOE rows outside supported window (FK cascades clear child rows).
DELETE FROM chain_snapshots
WHERE source = 'CBOE'
  AND (target_dte < 0 OR target_dte > 10);
