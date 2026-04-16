"""Pydantic schema for training candidate grid configuration.

Loads YAML files that define which entry times, DTEs, deltas, widths,
and spread sides to use when generating the training candidate universe
in ``generate_training_data.py``.

Usage::

    from configs.training.schema import load_training_config, TrainingGridConfig
    cfg = load_training_config("backend/configs/training/default.yaml")
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

logger = logging.getLogger(__name__)

# Must stay in sync with TP_LEVELS in generate_training_data.py
VALID_TP_LEVELS = {50, 60, 70, 80, 90, 100}


class TrainingGridConfig(BaseModel):
    """Top-level schema for a training candidate grid definition.

    Every field has a default matching the current hardcoded values in
    ``generate_training_data.py``, so omitting a section in YAML falls
    back to the production defaults.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "default"
    description: str = ""

    decision_minutes_et: list[list[int]] = [
        [10, 0], [11, 0], [12, 0], [13, 0], [14, 0], [15, 0], [16, 0],
    ]
    dte_targets: list[int] = [0, 1, 3, 5, 7, 10]
    delta_targets: list[float] = [0.10, 0.15, 0.20, 0.25]
    spread_sides: list[str] = ["put", "call"]
    width_targets: list[float] = [5.0, 10.0, 15.0, 20.0]

    take_profit_pct: float = 0.50
    stop_loss_pct: float = 2.00
    label_mark_interval_minutes: int = 5

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Training config 'name' must not be empty")
        return v.strip()

    @field_validator("decision_minutes_et")
    @classmethod
    def _valid_times(cls, v: list[list[int]]) -> list[list[int]]:
        """Each entry must be [hour, minute] with valid ranges."""
        if not v:
            raise ValueError("decision_minutes_et must not be empty")
        for pair in v:
            if len(pair) != 2:
                raise ValueError(f"Each entry time must be [hour, minute], got {pair}")
            if not (0 <= pair[0] <= 23 and 0 <= pair[1] <= 59):
                raise ValueError(f"Invalid time: {pair}")
        return v

    @field_validator("spread_sides")
    @classmethod
    def _valid_sides(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("spread_sides must not be empty")
        allowed = {"put", "call"}
        for s in v:
            if s not in allowed:
                raise ValueError(f"Invalid spread side: {s!r}. Must be 'put' or 'call'.")
        return v

    @field_validator("take_profit_pct")
    @classmethod
    def _valid_tp(cls, v: float) -> float:
        """Ensure take_profit_pct maps to a supported TP level (50-100%)."""
        lvl = round(v * 100)
        if lvl not in VALID_TP_LEVELS:
            raise ValueError(
                f"take_profit_pct={v} maps to TP level {lvl} which is not in "
                f"supported levels {sorted(VALID_TP_LEVELS)}. "
                f"Use one of: {[l/100 for l in sorted(VALID_TP_LEVELS)]}"
            )
        return v

    @field_validator("stop_loss_pct")
    @classmethod
    def _valid_sl(cls, v: float) -> float:
        """Stop-loss multiplier must be positive."""
        if v <= 0:
            raise ValueError(f"stop_loss_pct must be > 0, got {v}")
        return v

    @field_validator("label_mark_interval_minutes")
    @classmethod
    def _valid_interval(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"label_mark_interval_minutes must be >= 1, got {v}")
        return v

    @field_validator("dte_targets")
    @classmethod
    def _valid_dte(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("dte_targets must not be empty")
        for d in v:
            if d < 0:
                raise ValueError(f"DTE target must be >= 0, got {d}")
        return v

    @field_validator("delta_targets")
    @classmethod
    def _valid_deltas(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("delta_targets must not be empty")
        for d in v:
            if not (0 < d <= 1.0):
                raise ValueError(f"Delta target must be in (0, 1.0], got {d}")
        return v

    @field_validator("width_targets")
    @classmethod
    def _valid_widths(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("width_targets must not be empty")
        for w in v:
            if w <= 0:
                raise ValueError(f"Width target must be > 0, got {w}")
        return v

    def as_tuples(self) -> list[tuple[int, int]]:
        """Return decision_minutes_et as list of (hour, minute) tuples."""
        return [(pair[0], pair[1]) for pair in self.decision_minutes_et]

    def content_hash(self) -> str:
        """Deterministic hash of all grid parameters for cache keying.

        Changing any parameter (entry times, DTEs, widths, etc.) produces
        a different hash, which invalidates the candidate cache.
        """
        canon = (
            f"times={self.decision_minutes_et}|"
            f"dte={self.dte_targets}|"
            f"delta={self.delta_targets}|"
            f"sides={self.spread_sides}|"
            f"widths={self.width_targets}|"
            f"tp={self.take_profit_pct}|"
            f"sl={self.stop_loss_pct}|"
            f"interval={self.label_mark_interval_minutes}"
        )
        return hashlib.sha256(canon.encode()).hexdigest()[:16]


def load_training_config(path: str | Path) -> TrainingGridConfig:
    """Load and validate a training grid YAML file.

    Parameters
    ----------
    path : Path to the YAML file.

    Returns
    -------
    Validated TrainingGridConfig instance.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = TrainingGridConfig(**raw)
    logger.info(
        "Loaded training config '%s': %d times x %d DTEs x %d deltas "
        "x %d sides x %d widths = %d combos per day",
        cfg.name,
        len(cfg.decision_minutes_et),
        len(cfg.dte_targets),
        len(cfg.delta_targets),
        len(cfg.spread_sides),
        len(cfg.width_targets),
        (len(cfg.decision_minutes_et) * len(cfg.dte_targets) *
         len(cfg.delta_targets) * len(cfg.spread_sides) * len(cfg.width_targets)),
    )
    return cfg
