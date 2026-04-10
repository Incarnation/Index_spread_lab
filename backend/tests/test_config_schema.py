"""Tests for YAML config-as-code grid loader."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.optimizer.schema import build_configs_from_yaml, _load_yaml

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "optimizer"


class TestYamlLoading:
    """Verify YAML files parse into valid OptimizerGridConfig objects."""

    @pytest.mark.parametrize("filename", [
        "staged_stage1.yaml",
        "staged_stage2.yaml",
        "staged_stage3.yaml",
        "event_only.yaml",
        "selective.yaml",
        "portfolio_sweep.yaml",
    ])
    def test_yaml_parses(self, filename: str) -> None:
        """Each YAML file loads without validation errors."""
        cfg = _load_yaml(CONFIGS_DIR / filename)
        assert cfg.name

    def test_staged_stage1_grid_size(self) -> None:
        """Stage 1 YAML produces the same grid size as hardcoded function."""
        from backtest_strategy import _build_staged_grid_stage1
        yaml_configs = build_configs_from_yaml(CONFIGS_DIR / "staged_stage1.yaml")
        hardcoded_configs = _build_staged_grid_stage1()
        assert len(yaml_configs) == len(hardcoded_configs)

    def test_selective_grid_size(self) -> None:
        """Selective YAML produces the same grid size as hardcoded function."""
        from backtest_strategy import _build_selective_grid
        yaml_configs = build_configs_from_yaml(CONFIGS_DIR / "selective.yaml")
        hardcoded_configs = _build_selective_grid()
        assert len(yaml_configs) == len(hardcoded_configs)


class TestGridContents:
    """Verify built configs have correct field values."""

    def test_event_only_all_event_enabled(self) -> None:
        """Every event-only config has events enabled and event_only=True."""
        configs = build_configs_from_yaml(CONFIGS_DIR / "event_only.yaml")
        for c in configs:
            assert c.event.enabled is True
            assert c.event.event_only is True

    def test_selective_all_have_width_filter(self) -> None:
        """All selective configs use width_filter=10."""
        configs = build_configs_from_yaml(CONFIGS_DIR / "selective.yaml")
        for c in configs:
            assert c.trading.width_filter == 10.0

    def test_filters_prune_invalid_dte(self) -> None:
        """Configs with min_dte > max_dte are filtered out."""
        configs = build_configs_from_yaml(CONFIGS_DIR / "event_only.yaml")
        for c in configs:
            if c.event.enabled:
                assert c.event.min_dte <= c.event.max_dte

    def test_filters_prune_shared_multi_trade(self) -> None:
        """Shared budget mode configs never have max_event_trades > 1."""
        configs = build_configs_from_yaml(CONFIGS_DIR / "staged_stage3.yaml")
        for c in configs:
            if c.event.budget_mode == "shared":
                assert c.event.max_event_trades <= 1
