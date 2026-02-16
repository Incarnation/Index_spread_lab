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
    snapshot_range_fallback_enabled: bool = False
    snapshot_range_fallback_count: int = 3
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
    # Keep this >= number of expirations captured per snapshot cycle (e.g., 0-10 DTE => up to 11 snapshots).
    gex_snapshot_batch_limit: int = 20
    gex_strike_limit: int = 150
    gex_max_dte_days: int = 10

    decision_entry_times: str = "10:00,11:00,12:00"
    decision_dte_targets: str = "3,5,7"
    decision_dte_tolerance_days: int = 1
    decision_delta_targets: str = "0.10,0.20"
    decision_spread_side: str = "put"
    decision_spread_width_points: float = 25.0
    decision_contracts: int = 1
    decision_snapshot_max_age_minutes: int = 15
    decision_max_trades_per_day: int = 1
    decision_max_open_trades: int = 1
    decision_ruleset_version: str = "rules_v1"
    decision_allow_outside_rth: bool = False

    # ML feature/candidate generation (step 1)
    feature_builder_enabled: bool = True
    feature_builder_allow_outside_rth: bool = False
    feature_schema_version: str = "fs_v1"
    candidate_schema_version: str = "cand_v1"

    # ML label resolver (step 2)
    labeler_enabled: bool = True
    labeler_interval_minutes: int = 15
    labeler_batch_limit: int = 200
    labeler_min_age_minutes: int = 5
    labeler_max_wait_hours: int = 336
    labeler_take_profit_pct: float = 0.50
    label_schema_version: str = "label_v1"
    label_contract_multiplier: int = 100

    # Live trade PnL mark-to-market job
    trade_pnl_enabled: bool = True
    trade_pnl_interval_minutes: int = 5
    trade_pnl_allow_outside_rth: bool = False
    trade_pnl_mark_max_age_minutes: int = 30
    trade_pnl_take_profit_pct: float = 0.50
    trade_pnl_stop_loss_pct: float = 1.00
    trade_pnl_contract_multiplier: int = 100

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

    def decision_entry_times_list(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for part in self.decision_entry_times.split(","):
            p = part.strip()
            if not p:
                continue
            try:
                h_str, m_str = p.split(":")
                out.append((int(h_str), int(m_str)))
            except Exception:
                continue
        return out

    def decision_dte_targets_list(self) -> list[int]:
        parts = [p.strip() for p in self.decision_dte_targets.split(",") if p.strip()]
        return [int(p) for p in parts]

    def decision_delta_targets_list(self) -> list[float]:
        parts = [p.strip() for p in self.decision_delta_targets.split(",") if p.strip()]
        return [float(p) for p in parts]


settings = Settings()

