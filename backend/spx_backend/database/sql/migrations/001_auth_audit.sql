-- Auth audit: add is_admin to users; drop and recreate auth_audit_log with final schema.
-- Safe when table is empty. Run after db_schema.sql (which also creates auth_audit_log for new installs).

ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT false;

DROP TABLE IF EXISTS auth_audit_log;

CREATE TABLE auth_audit_log (
  id BIGSERIAL PRIMARY KEY,
  event_type TEXT NOT NULL,
  user_id BIGINT NULL REFERENCES users(id) ON DELETE SET NULL,
  username TEXT NULL,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ip_address INET NULL,
  user_agent TEXT NULL,
  country TEXT NULL,
  geo_json JSONB NULL,
  details JSONB NULL
);

CREATE INDEX IF NOT EXISTS idx_auth_audit_log_occurred_at ON auth_audit_log (occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_auth_audit_log_user_id ON auth_audit_log (user_id);
CREATE INDEX IF NOT EXISTS idx_auth_audit_log_event_type ON auth_audit_log (event_type);
