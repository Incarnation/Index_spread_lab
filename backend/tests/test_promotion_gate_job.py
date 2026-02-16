from __future__ import annotations

from spx_backend.jobs.promotion_gate_job import evaluate_promotion_gates


def test_evaluate_promotion_gates_passes_when_metrics_meet_thresholds() -> None:
    result = evaluate_promotion_gates(
        metrics={
            "resolved_test": 220,
            "tp50_rate_test": 0.62,
            "expectancy_test": 18.0,
            "max_drawdown_test": 3200.0,
            "tail_loss_proxy_test": -350.0,
            "avg_margin_usage_test": 1600.0,
        },
        min_resolved=100,
        min_tp50_rate=0.55,
        min_expectancy=0.0,
        max_drawdown=5000.0,
        min_tail_loss_proxy=-1000.0,
        max_avg_margin_usage=5000.0,
    )
    assert result["passed"] is True
    assert all(check["pass"] for check in result["checks"].values())


def test_evaluate_promotion_gates_fails_when_any_gate_fails() -> None:
    result = evaluate_promotion_gates(
        metrics={
            "resolved_test": 40,
            "tp50_rate_test": 0.45,
            "expectancy_test": -5.0,
            "max_drawdown_test": 12000.0,
            "tail_loss_proxy_test": -6000.0,
            "avg_margin_usage_test": 8200.0,
        },
        min_resolved=100,
        min_tp50_rate=0.55,
        min_expectancy=0.0,
        max_drawdown=5000.0,
        min_tail_loss_proxy=-1000.0,
        max_avg_margin_usage=5000.0,
    )
    assert result["passed"] is False
    assert result["checks"]["resolved_count"]["pass"] is False
    assert result["checks"]["tp50_rate"]["pass"] is False
    assert result["checks"]["expectancy"]["pass"] is False
