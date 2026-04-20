-- Audit Wave 3 finding H7: add a DB-backed table for alert cooldowns
-- so multi-replica deployments share a single cooldown window per
-- alert key.
--
-- Background
-- ----------
-- ``services.alerts.send_alert`` (the SendGrid dispatcher reused by
-- snapshot_job partial-batch alerts, trade_pnl_job split-brain alerts,
-- and the staleness monitor in Wave 3+) keeps a per-key cooldown
-- registry in a module-level dict ``_last_alert_at: dict[str, datetime]``.
--
-- The dict is process-local. Once Railway scales the backend beyond
-- one replica (or restarts a worker), each replica has its own
-- registry and the same alert can fire ``N`` times per cooldown
-- window. We're single-replica today, but the audit wants this gap
-- closed before the next traffic-driven scale-up.
--
-- Resolution (per audit decision)
-- -------------------------------
-- Add a tiny key/value table that ``send_alert`` reads + upserts
-- inside the cooldown gate. The PK on ``cooldown_key`` enforces one
-- row per gate; the ``last_alert_ts`` column stores the most recent
-- delivery. The app code does:
--   1. SELECT last_alert_ts FROM alert_cooldowns WHERE cooldown_key = $1
--   2. If row exists AND now - last_alert_ts < cooldown_minutes -> skip
--   3. Else send the email
--   4. INSERT ... ON CONFLICT (cooldown_key) DO UPDATE
--      SET last_alert_ts = NOW()
--
-- The DB roundtrip is small (PK lookup + small upsert) and only
-- happens on the alert-fire path, so the latency cost is negligible.
-- Falls back to the in-process dict when the DB is unreachable so
-- alerts still go out during DB outages.
--
-- Idempotency: schema_migrations tracker ensures single-execution.
-- Lock budget: CREATE TABLE on an empty table is instant.
SET lock_timeout = '5s';
SET statement_timeout = '30s';

CREATE TABLE IF NOT EXISTS alert_cooldowns (
    cooldown_key TEXT PRIMARY KEY,
    last_alert_ts TIMESTAMPTZ NOT NULL
);
