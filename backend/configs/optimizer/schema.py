"""YAML-to-FullConfig grid builder with Pydantic validation.

Loads optimizer parameter sweep definitions from YAML files and generates
the Cartesian product of all specified values as a list of ``FullConfig``
objects.  Falls back to the hardcoded ``_build_*_grid()`` functions when
no YAML file is provided.

Usage::

    from configs.optimizer.schema import build_configs_from_yaml
    configs = build_configs_from_yaml("backend/configs/optimizer/staged_stage1.yaml")
"""
from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for YAML validation
# ---------------------------------------------------------------------------

class PortfolioGrid(BaseModel):
    """Allowed sweep values for PortfolioConfig fields."""
    model_config = ConfigDict(extra="forbid")
    starting_capital: list[float] = [20_000]
    max_trades_per_day: list[int] = [2]
    monthly_drawdown_limit: list[float | None] = [0.15]
    lot_per_equity: list[float] = [10_000]
    max_equity_risk_pct: list[float] = [0.10]
    max_margin_pct: list[float] = [0.30]
    calls_only: list[bool] = [True]
    min_dte: list[int | None] = [None]
    max_delta: list[float | None] = [None]


class TradingGrid(BaseModel):
    """Allowed sweep values for TradingConfig fields."""
    model_config = ConfigDict(extra="forbid")
    tp_pct: list[float] = [0.50]
    sl_mult: list[float | None] = [None]
    max_vix: list[float | None] = [None]
    max_term_structure: list[float | None] = [None]
    avoid_opex: list[bool] = [False]
    prefer_event_days: list[bool] = [False]
    width_filter: list[float | None] = [None]
    entry_count: list[int | None] = [None]


class EventGrid(BaseModel):
    """Allowed sweep values for EventConfig fields."""
    model_config = ConfigDict(extra="forbid")
    enabled: list[bool] = [False]
    signal_mode: list[str] = ["any"]
    budget_mode: list[str] = ["shared"]
    max_event_trades: list[int] = [1]
    spx_drop_threshold: list[float] = [-0.01]
    spx_drop_2d_threshold: list[float] = [-0.02]
    spx_drop_min: list[float | None] = [None]
    spx_drop_max: list[float | None] = [None]
    vix_spike_threshold: list[float] = [0.15]
    vix_elevated_threshold: list[float] = [25.0]
    term_inversion_threshold: list[float] = [1.0]
    side_preference: list[str] = ["puts"]
    min_dte: list[int] = [5]
    max_dte: list[int] = [7]
    min_delta: list[float] = [0.15]
    max_delta: list[float] = [0.25]
    rally_avoidance: list[bool] = [False]
    rally_threshold: list[float] = [0.01]
    event_only: list[bool] = [False]


class RegimeGrid(BaseModel):
    """Allowed sweep values for RegimeThrottle fields."""
    model_config = ConfigDict(extra="forbid")
    enabled: list[bool] = [False]
    high_vix_threshold: list[float] = [30.0]
    high_vix_multiplier: list[float] = [0.5]
    extreme_vix_threshold: list[float] = [40.0]
    big_drop_threshold: list[float] = [-0.02]
    big_drop_multiplier: list[float] = [0.5]
    consecutive_loss_days: list[int] = [3]
    consecutive_loss_multiplier: list[float] = [0.5]


class FilterRules(BaseModel):
    """Post-generation filters to prune invalid combos."""
    model_config = ConfigDict(extra="forbid")
    min_dte_lt_max_dte: bool = True
    min_delta_lt_max_delta: bool = True
    shared_single_event_trade: bool = True
    lot_per_equity_eq_capital: bool = False


class OptimizerGridConfig(BaseModel):
    """Top-level YAML schema for an optimizer grid definition."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    portfolio: PortfolioGrid = PortfolioGrid()
    trading: TradingGrid = TradingGrid()
    event: EventGrid = EventGrid()
    regime: RegimeGrid = RegimeGrid()
    filters: FilterRules = FilterRules()

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Grid config 'name' must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> OptimizerGridConfig:
    """Parse and validate a YAML grid definition file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return OptimizerGridConfig(**raw)


def build_configs_from_yaml(path: str | Path) -> "list[FullConfig]":
    """Build a list of FullConfig objects from a YAML grid definition.

    Generates the Cartesian product of all parameter lists, then applies
    the configured filter rules to prune invalid combinations.

    Parameters
    ----------
    path : Path to the YAML grid definition file.

    Returns
    -------
    List of valid ``FullConfig`` instances.
    """
    import sys as _sys
    _scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(_scripts) not in _sys.path:
        _sys.path.insert(0, str(_scripts))
    from backtest_strategy import (
        PortfolioConfig, TradingConfig, EventConfig,
        RegimeThrottle, FullConfig,
    )

    grid_cfg = _load_yaml(Path(path))
    p = grid_cfg.portfolio
    t = grid_cfg.trading
    e = grid_cfg.event
    r = grid_cfg.regime
    flt = grid_cfg.filters

    p_fields = list(p.model_dump().values())
    t_fields = list(t.model_dump().values())
    e_fields = list(e.model_dump().values())
    r_fields = list(r.model_dump().values())

    p_keys = list(p.model_dump().keys())
    t_keys = list(t.model_dump().keys())
    e_keys = list(e.model_dump().keys())
    r_keys = list(r.model_dump().keys())

    configs: list[FullConfig] = []

    for combo in itertools.product(
        *p_fields, *t_fields, *e_fields, *r_fields,
    ):
        idx = 0
        p_vals = {}
        for k in p_keys:
            p_vals[k] = combo[idx]; idx += 1
        t_vals = {}
        for k in t_keys:
            t_vals[k] = combo[idx]; idx += 1
        e_vals = {}
        for k in e_keys:
            e_vals[k] = combo[idx]; idx += 1
        r_vals = {}
        for k in r_keys:
            r_vals[k] = combo[idx]; idx += 1

        if flt.min_dte_lt_max_dte and e_vals.get("enabled", False):
            if e_vals["min_dte"] > e_vals["max_dte"]:
                continue
        if flt.min_delta_lt_max_delta and e_vals.get("enabled", False):
            if e_vals["min_delta"] > e_vals["max_delta"]:
                continue
        if flt.shared_single_event_trade and e_vals.get("enabled", False):
            if e_vals["budget_mode"] == "shared" and e_vals["max_event_trades"] > 1:
                continue
        if flt.lot_per_equity_eq_capital:
            if p_vals.get("lot_per_equity") != p_vals.get("starting_capital"):
                continue

        pc = PortfolioConfig(**p_vals)
        tc = TradingConfig(**t_vals)
        ec = EventConfig(**e_vals)
        rt = RegimeThrottle(**r_vals)
        configs.append(FullConfig(portfolio=pc, trading=tc, event=ec, regime=rt))

    logger.info("Loaded grid '%s': %d configs from %s", grid_cfg.name, len(configs), path)
    return configs
