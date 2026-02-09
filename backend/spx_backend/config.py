from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # When running locally, we typically run from repo root or from backend/.
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), env_file_encoding="utf-8")

    app_env: str = "local"
    log_level: str = "INFO"
    tz: str = "America/New_York"

    database_url: str

    tradier_base_url: str = "https://sandbox.tradier.com/v1"
    tradier_access_token: str
    tradier_account_id: str

    snapshot_interval_minutes: int = 5
    snapshot_underlying: str = "SPX"
    snapshot_dte_targets: str = "3,5,7"
    snapshot_dte_mode: str = "range"  # "range" or "targets"
    snapshot_dte_min_days: int = 0
    snapshot_dte_max_days: int = 10
    snapshot_dte_tolerance_days: int = 1
    snapshot_strikes_each_side: int = 100
    quote_symbols: str = "SPX,VIX,VIX9D,SPY"
    quote_interval_minutes: int = 5
    gex_enabled: bool = True
    gex_interval_minutes: int = 5
    gex_store_by_expiry: bool = True
    gex_spot_max_age_seconds: int = 600
    gex_contract_multiplier: int = 100
    gex_puts_negative: bool = True
    gex_snapshot_batch_limit: int = 5
    gex_strike_limit: int = 150
    gex_max_dte_days: int = 10

    cors_origins: str = "http://localhost:5173"

    # Testing/ops controls
    allow_snapshot_outside_rth: bool = False
    allow_quotes_outside_rth: bool = False
    admin_api_key: str | None = None
    market_clock_cache_seconds: int = 300

    def dte_targets_list(self) -> list[int]:
        parts = [p.strip() for p in self.snapshot_dte_targets.split(",") if p.strip()]
        return [int(p) for p in parts]

    def quote_symbols_list(self) -> list[str]:
        parts = [p.strip() for p in self.quote_symbols.split(",") if p.strip()]
        return parts

    def cors_origins_list(self) -> list[str]:
        parts = [p.strip() for p in self.cors_origins.split(",") if p.strip()]
        return parts


settings = Settings()

