"""Tests for YAML config-as-code grid loader (optimizer and training)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.optimizer.schema import build_configs_from_yaml, _load_yaml
from configs.training.schema import (
    TrainingGridConfig,
    load_training_config,
    VALID_TP_LEVELS,
)

OPTIMIZER_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "optimizer"
TRAINING_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "training"


class TestYamlLoading:
    """Verify optimizer YAML files parse into valid OptimizerGridConfig objects."""

    @pytest.mark.parametrize("filename", [
        "staged_stage1.yaml",
        "staged_stage2.yaml",
        "staged_stage3.yaml",
        "event_only.yaml",
        "event_only_v2.yaml",
        "event_only_v2_explore.yaml",
        "selective.yaml",
        "portfolio_sweep.yaml",
    ])
    def test_yaml_parses(self, filename: str) -> None:
        """Each optimizer YAML file loads without validation errors."""
        cfg = _load_yaml(OPTIMIZER_CONFIGS_DIR / filename)
        assert cfg.name

    def test_staged_stage1_grid_size(self) -> None:
        """Stage 1 YAML produces the same grid size as hardcoded function."""
        from backtest_strategy import _build_staged_grid_stage1
        yaml_configs = build_configs_from_yaml(OPTIMIZER_CONFIGS_DIR / "staged_stage1.yaml")
        hardcoded_configs = _build_staged_grid_stage1()
        assert len(yaml_configs) == len(hardcoded_configs)

    def test_selective_grid_size(self) -> None:
        """Selective YAML produces the same grid size as hardcoded function."""
        from backtest_strategy import _build_selective_grid
        yaml_configs = build_configs_from_yaml(OPTIMIZER_CONFIGS_DIR / "selective.yaml")
        hardcoded_configs = _build_selective_grid()
        assert len(yaml_configs) == len(hardcoded_configs)


class TestGridContents:
    """Verify built configs have correct field values."""

    def test_event_only_all_event_enabled(self) -> None:
        """Every event-only config has events enabled and event_only=True."""
        configs = build_configs_from_yaml(OPTIMIZER_CONFIGS_DIR / "event_only.yaml")
        for c in configs:
            assert c.event.enabled is True
            assert c.event.event_only is True

    def test_selective_all_have_width_filter(self) -> None:
        """All selective configs use width_filter=10."""
        configs = build_configs_from_yaml(OPTIMIZER_CONFIGS_DIR / "selective.yaml")
        for c in configs:
            assert c.trading.width_filter == 10.0

    def test_filters_prune_invalid_dte(self) -> None:
        """Configs with min_dte > max_dte are filtered out."""
        configs = build_configs_from_yaml(OPTIMIZER_CONFIGS_DIR / "event_only.yaml")
        for c in configs:
            if c.event.enabled:
                assert c.event.min_dte <= c.event.max_dte

    def test_filters_prune_shared_multi_trade(self) -> None:
        """Shared budget mode configs never have max_event_trades > 1."""
        configs = build_configs_from_yaml(OPTIMIZER_CONFIGS_DIR / "staged_stage3.yaml")
        for c in configs:
            if c.event.budget_mode == "shared":
                assert c.event.max_event_trades <= 1


# ===================================================================
#  Training Config Schema Tests
# ===================================================================


class TestTrainingYamlLoading:
    """Verify training YAML files parse into valid TrainingGridConfig objects."""

    @pytest.mark.parametrize("filename", [
        "default.yaml",
        "narrow.yaml",
    ])
    def test_yaml_parses(self, filename: str) -> None:
        """Each training YAML file loads without validation errors."""
        cfg = load_training_config(TRAINING_CONFIGS_DIR / filename)
        assert cfg.name

    def test_default_content_hash_stable(self) -> None:
        """Same config produces the same hash on repeated loads."""
        cfg1 = load_training_config(TRAINING_CONFIGS_DIR / "default.yaml")
        cfg2 = load_training_config(TRAINING_CONFIGS_DIR / "default.yaml")
        assert cfg1.content_hash() == cfg2.content_hash()

    def test_different_configs_different_hashes(self) -> None:
        """default and narrow configs produce different hashes (assuming they differ)."""
        cfg_default = load_training_config(TRAINING_CONFIGS_DIR / "default.yaml")
        cfg_narrow = load_training_config(TRAINING_CONFIGS_DIR / "narrow.yaml")
        if cfg_default.model_dump() != cfg_narrow.model_dump():
            assert cfg_default.content_hash() != cfg_narrow.content_hash()


class TestTrainingSchemaValidation:
    """Verify training schema validators catch bad input."""

    def test_invalid_tp_rejected(self) -> None:
        """take_profit_pct outside TP_LEVELS raises ValidationError."""
        with pytest.raises(Exception, match="take_profit_pct"):
            TrainingGridConfig(take_profit_pct=0.55)

    def test_valid_tp_accepted(self) -> None:
        """All valid TP levels are accepted."""
        for lvl in sorted(VALID_TP_LEVELS):
            cfg = TrainingGridConfig(take_profit_pct=lvl / 100)
            assert cfg.take_profit_pct == lvl / 100

    def test_negative_sl_rejected(self) -> None:
        """Negative stop_loss_pct raises ValidationError."""
        with pytest.raises(Exception, match="stop_loss_pct"):
            TrainingGridConfig(stop_loss_pct=-1.0)

    def test_zero_interval_rejected(self) -> None:
        """label_mark_interval_minutes < 1 raises ValidationError."""
        with pytest.raises(Exception, match="label_mark_interval_minutes"):
            TrainingGridConfig(label_mark_interval_minutes=0)

    def test_empty_dte_rejected(self) -> None:
        """Empty dte_targets list raises ValidationError."""
        with pytest.raises(Exception, match="dte_targets"):
            TrainingGridConfig(dte_targets=[])

    def test_negative_delta_rejected(self) -> None:
        """Negative delta target raises ValidationError."""
        with pytest.raises(Exception, match="Delta target"):
            TrainingGridConfig(delta_targets=[-0.1])

    def test_zero_width_rejected(self) -> None:
        """Zero width target raises ValidationError."""
        with pytest.raises(Exception, match="Width target"):
            TrainingGridConfig(width_targets=[0.0])

    def test_extra_field_rejected(self) -> None:
        """Unknown fields are rejected by extra='forbid'."""
        with pytest.raises(Exception):
            TrainingGridConfig(unknown_field="oops")

    def test_empty_spread_sides_rejected(self) -> None:
        """Empty spread_sides list raises ValidationError."""
        with pytest.raises(Exception, match="spread_sides"):
            TrainingGridConfig(spread_sides=[])


class TestTPLevelsSync:
    """Ensure TP_LEVELS in generate_training_data stays in sync with VALID_TP_LEVELS in schema."""

    def test_tp_levels_match(self) -> None:
        """The set of supported TP levels must be identical in both modules."""
        from generate_training_data import TP_LEVELS
        assert set(TP_LEVELS) == VALID_TP_LEVELS
