-- Allow chain_snapshots to be deleted without FK violations from
-- trades and trade_decisions.  All three columns are already NULLABLE,
-- so ON DELETE SET NULL is safe.  Requires dropping the old constraint
-- and recreating it (PostgreSQL cannot ALTER a constraint in-place).

-- trades.entry_snapshot_id
ALTER TABLE trades DROP CONSTRAINT IF EXISTS trades_entry_snapshot_id_fkey;
ALTER TABLE trades
    ADD CONSTRAINT trades_entry_snapshot_id_fkey
    FOREIGN KEY (entry_snapshot_id) REFERENCES chain_snapshots(snapshot_id)
    ON DELETE SET NULL;

-- trades.last_snapshot_id
ALTER TABLE trades DROP CONSTRAINT IF EXISTS trades_last_snapshot_id_fkey;
ALTER TABLE trades
    ADD CONSTRAINT trades_last_snapshot_id_fkey
    FOREIGN KEY (last_snapshot_id) REFERENCES chain_snapshots(snapshot_id)
    ON DELETE SET NULL;

-- trade_decisions.chain_snapshot_id
ALTER TABLE trade_decisions DROP CONSTRAINT IF EXISTS trade_decisions_chain_snapshot_id_fkey;
ALTER TABLE trade_decisions
    ADD CONSTRAINT trade_decisions_chain_snapshot_id_fkey
    FOREIGN KEY (chain_snapshot_id) REFERENCES chain_snapshots(snapshot_id)
    ON DELETE SET NULL;
