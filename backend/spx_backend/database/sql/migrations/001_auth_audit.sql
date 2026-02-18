-- Auth audit migration: ensure users.is_admin and auth_audit_log schema exist.
-- Non-destructive by design so startup init_db() does not wipe audit history.

ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT false;

CREATE TABLE IF NOT EXISTS auth_audit_log (
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
