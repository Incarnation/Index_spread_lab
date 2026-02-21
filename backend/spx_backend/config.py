from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # When running locally, we typically run from repo root or from backend/.
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), env_file_encoding="utf-8")

    log_level: str = "INFO"
    """If True, skip running quote/snapshot/gex jobs once on startup (server becomes ready immediately). Scheduler still runs jobs on their intervals."""
    skip_startup_warmup: bool = False
    tz: str = "America/New_York"

    database_url: str

    tradier_base_url: str = "https://sandbox.tradier.com/v1"
    tradier_access_token: str
    tradier_account_id: str
    mzdata_base_url: str = "https://mztrading-data.deno.dev"

    snapshot_interval_minutes: int = 10
    snapshot_underlying: str = "SPX"
    snapshot_dte_targets: str = "3,5,7"
    snapshot_dte_mode: str = "range"  # "range" or "targets"
    snapshot_dte_min_days: int = 0
    snapshot_dte_max_days: int = 10
    snapshot_range_fallback_enabled: bool = False
    snapshot_range_fallback_count: int = 3
    snapshot_dte_tolerance_days: int = 1
    snapshot_strikes_each_side: int = 75
    spy_snapshot_enabled: bool = True
    spy_snapshot_interval_minutes: int = 10
    spy_snapshot_underlying: str = "SPY"
    spy_snapshot_dte_targets: str = "3,5,7"
    spy_snapshot_dte_mode: str = "range"  # "range" or "targets"
    spy_snapshot_dte_min_days: int = 0
    spy_snapshot_dte_max_days: int = 10
    spy_snapshot_range_fallback_enabled: bool = False
    spy_snapshot_range_fallback_count: int = 3
    spy_snapshot_dte_tolerance_days: int = 1
    spy_snapshot_strikes_each_side: int = 75
    spy_allow_snapshot_outside_rth: bool = False
    vix_snapshot_enabled: bool = False
    vix_snapshot_interval_minutes: int = 10
    vix_snapshot_underlying: str = "VIX"
    vix_snapshot_dte_targets: str = "14,21,28"
    vix_snapshot_dte_mode: str = "range"  # "range" or "targets"
    vix_snapshot_dte_min_days: int = 7
    vix_snapshot_dte_max_days: int = 45
    vix_snapshot_range_fallback_enabled: bool = False
    vix_snapshot_range_fallback_count: int = 3
    vix_snapshot_dte_tolerance_days: int = 2
    vix_snapshot_strikes_each_side: int = 50
    vix_allow_snapshot_outside_rth: bool = False
    quote_symbols: str = "SPX,VIX,VIX9D,SPY"
    quote_interval_minutes: int = 10
    gex_enabled: bool = True
    gex_interval_minutes: int = 5
    gex_allow_outside_rth: bool = False
    gex_store_by_expiry: bool = True
    gex_spot_max_age_seconds: int = 600
    gex_contract_multiplier: int = 100
    gex_puts_negative: bool = True
    # Keep this >= number of expirations captured per snapshot cycle (e.g., 0-10 DTE => up to 11 snapshots).
    gex_snapshot_batch_limit: int = 20
    gex_strike_limit: int = 150
    gex_max_dte_days: int = 10
    cboe_gex_enabled: bool = False
    cboe_gex_underlying: str = "SPX"
    cboe_gex_interval_minutes: int = 15
    cboe_gex_allow_outside_rth: bool = False

    decision_entry_times: str = "10:00,11:00,12:00"
    decision_dte_targets: str = "0,3,5,7,10"
    decision_dte_tolerance_days: int = 1
    decision_delta_targets: str = "0.10,0.20"
    decision_spread_side: str = "put"
    decision_spread_sides: str = ""
    decision_spread_width_points: float = 25.0
    decision_contracts: int = 1
    decision_snapshot_max_age_minutes: int = 15
    decision_max_trades_per_run: int = 4
    decision_max_trades_per_day: int = 20
    decision_max_open_trades: int = 15
    decision_max_trades_per_side_per_day: int = 0
    decision_max_open_trades_per_side: int = 0
    decision_ruleset_version: str = "rules_v1"
    decision_allow_outside_rth: bool = False

    # ML feature/candidate generation (step 1)
    feature_builder_enabled: bool = True
    feature_builder_allow_outside_rth: bool = False
    feature_schema_version: str = "fs_v2"
    candidate_schema_version: str = "cand_v2"

    # ML label resolver (step 2)
    labeler_enabled: bool = True
    labeler_batch_limit: int = 200
    labeler_min_age_minutes: int = 5
    labeler_max_wait_hours: int = 336
    labeler_take_profit_pct: float = 0.50
    label_schema_version: str = "label_v1"
    label_contract_multiplier: int = 100

    # Weekly trainer (step 3)
    trainer_enabled: bool = True
    trainer_weekday: str = "sat"  # mon..sun accepted by APScheduler cron.
    trainer_hour: int = 9
    trainer_minute: int = 0
    trainer_model_name: str = "cand_bucket_v1"
    trainer_lookback_days: int = 365
    trainer_test_days: int = 28
    trainer_min_rows: int = 200
    trainer_min_train_rows: int = 100
    trainer_min_test_rows: int = 50
    trainer_sparse_cv_enabled: bool = True
    trainer_sparse_cv_min_rows: int = 30
    trainer_sparse_cv_folds: int = 4
    trainer_sparse_cv_min_train_rows: int = 12
    trainer_sparse_cv_min_test_rows: int = 6
    trainer_min_bucket_size: int = 12
    trainer_prior_strength: float = 8.0
    trainer_adaptive_prior_enabled: bool = True
    trainer_adaptive_prior_reference_rows: int = 200
    trainer_adaptive_prior_min: float = 2.0
    trainer_adaptive_prior_max: float = 24.0
    trainer_utility_prob_weight: float = 0.35
    trainer_utility_tail_penalty: float = 0.20
    trainer_utility_margin_penalty: float = 0.02

    # Shadow inference (step 4)
    shadow_inference_enabled: bool = True
    shadow_inference_batch_limit: int = 500
    shadow_inference_lookback_minutes: int = 1440
    shadow_inference_model_name: str = "cand_bucket_v1"

    # Promotion gate evaluator (step 5)
    promotion_gate_enabled: bool = True
    promotion_gate_model_name: str = "cand_bucket_v1"
    promotion_gate_min_resolved: int = 100
    promotion_gate_min_tp50_rate: float = 0.50
    promotion_gate_min_expectancy: float = 0.0
    promotion_gate_max_drawdown: float = 15000.0
    promotion_gate_min_tail_loss_proxy: float = -5000.0
    promotion_gate_max_avg_margin_usage: float = 5000.0
    promotion_gate_auto_activate: bool = False

    # Hybrid execution policy (step 6)
    decision_hybrid_enabled: bool = False
    decision_hybrid_model_name: str = "cand_bucket_v1"
    decision_hybrid_min_probability: float = 0.50
    decision_hybrid_min_expected_pnl: float = 0.0
    decision_hybrid_min_bucket_count: int = 8
    decision_hybrid_max_pnl_std: float = 250.0
    decision_hybrid_require_active_model: bool = True

    # Live trade PnL mark-to-market job
    trade_pnl_enabled: bool = True
    trade_pnl_interval_minutes: int = 5
    trade_pnl_allow_outside_rth: bool = False
    trade_pnl_mark_max_age_minutes: int = 30
    trade_pnl_take_profit_pct: float = 0.50
    trade_pnl_stop_loss_pct: float = 1.00
    trade_pnl_contract_multiplier: int = 100

    # Aggregate PnL analytics refresher
    performance_analytics_enabled: bool = False
    performance_analytics_interval_minutes: int = 10

    cors_origins: str = "http://localhost:5173"

    # Testing/ops controls
    allow_snapshot_outside_rth: bool = False
    allow_quotes_outside_rth: bool = False
    market_clock_cache_seconds: int = 300

    # Auth: JWT and user registration (multiple users, in-house auth).
    jwt_secret: str = ""  # set JWT_SECRET in env for auth; auth endpoints error if empty
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 24h
    auth_register_enabled: bool = False  # only allowed (pre-created) users can log in

    def _parse_int_csv(self, value: str) -> list[int]:
        """Parse comma-separated integer values, ignoring malformed items."""
        out: list[int] = []
        for part in value.split(","):
            token = part.strip()
            if not token:
                continue
            try:
                out.append(int(token))
            except ValueError:
                continue
        return out

    def dte_targets_list(self) -> list[int]:
        """Parse snapshot DTE targets from comma-separated env configuration."""
        return self._parse_int_csv(self.snapshot_dte_targets)

    def vix_snapshot_dte_targets_list(self) -> list[int]:
        """Parse VIX snapshot DTE targets from comma-separated env configuration."""
        return self._parse_int_csv(self.vix_snapshot_dte_targets)

    def spy_snapshot_dte_targets_list(self) -> list[int]:
        """Parse SPY snapshot DTE targets from comma-separated env configuration."""
        return self._parse_int_csv(self.spy_snapshot_dte_targets)

    def quote_symbols_list(self) -> list[str]:
        """Parse quote symbol list from comma-separated env configuration."""
        parts = [p.strip() for p in self.quote_symbols.split(",") if p.strip()]
        return parts

    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into a normalized list for FastAPI middleware."""
        parts = [p.strip() for p in self.cors_origins.split(",") if p.strip()]
        return parts

    def decision_entry_times_list(self) -> list[tuple[int, int]]:
        """Parse decision cron times (HH:MM) into hour/minute tuples."""
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
        """Parse decision DTE targets from comma-separated env configuration."""
        parts = [p.strip() for p in self.decision_dte_targets.split(",") if p.strip()]
        return [int(p) for p in parts]

    def decision_delta_targets_list(self) -> list[float]:
        """Parse decision delta targets from comma-separated env configuration."""
        parts = [p.strip() for p in self.decision_delta_targets.split(",") if p.strip()]
        return [float(p) for p in parts]

    def decision_spread_sides_list(self) -> list[str]:
        """Parse allowed spread sides with fallback to single-side config."""
        raw = self.decision_spread_sides.strip()
        source = raw if raw else self.decision_spread_side
        out: list[str] = []
        for part in source.split(","):
            side = part.strip().lower()
            if side not in {"put", "call"}:
                continue
            if side not in out:
                out.append(side)
        return out


settings = Settings()

