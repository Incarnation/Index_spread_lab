-- Add source-specific GEX columns to context_snapshots so Tradier and CBOE
-- values are stored independently.  The existing gex_net / zero_gamma_level
-- columns become "canonical" (CBOE-preferred, Tradier-fallback).

ALTER TABLE context_snapshots
    ADD COLUMN IF NOT EXISTS gex_net_tradier       DOUBLE PRECISION NULL,
    ADD COLUMN IF NOT EXISTS zero_gamma_level_tradier DOUBLE PRECISION NULL,
    ADD COLUMN IF NOT EXISTS gex_net_cboe          DOUBLE PRECISION NULL,
    ADD COLUMN IF NOT EXISTS zero_gamma_level_cboe DOUBLE PRECISION NULL;
