"""Unit tests for EventSignalDetector threshold evaluation logic."""

from __future__ import annotations

import pytest

from spx_backend.services.event_signals import EventSignalDetector


@pytest.fixture()
def detector() -> EventSignalDetector:
    """Detector with explicit thresholds independent of env config."""
    return EventSignalDetector(
        spx_drop_threshold=-0.01,
        spx_drop_2d_threshold=-0.02,
        vix_spike_threshold=0.15,
        vix_elevated_threshold=25.0,
        term_inversion_threshold=1.0,
        rally_avoidance=True,
        rally_threshold=0.01,
    )


class TestEvaluate:
    """Tests for _evaluate() threshold logic."""

    def test_no_signals_on_calm_market(self, detector: EventSignalDetector) -> None:
        ctx = {
            "prev_spx_return": 0.002,
            "prev_spx_return_2d": 0.004,
            "prev_vix_pct_change": 0.02,
            "vix": 15.0,
            "term_structure": 0.9,
        }
        assert detector._evaluate(ctx) == []

    def test_spx_drop_1d(self, detector: EventSignalDetector) -> None:
        ctx = {
            "prev_spx_return": -0.015,
            "prev_spx_return_2d": 0.0,
            "prev_vix_pct_change": 0.0,
            "vix": 15.0,
            "term_structure": 0.9,
        }
        signals = detector._evaluate(ctx)
        assert "spx_drop_1d" in signals
        assert "spx_drop_2d" not in signals

    def test_spx_drop_2d(self, detector: EventSignalDetector) -> None:
        ctx = {
            "prev_spx_return": 0.0,
            "prev_spx_return_2d": -0.025,
            "prev_vix_pct_change": 0.0,
            "vix": 15.0,
            "term_structure": 0.9,
        }
        signals = detector._evaluate(ctx)
        assert "spx_drop_2d" in signals
        assert "spx_drop_1d" not in signals

    def test_vix_spike(self, detector: EventSignalDetector) -> None:
        ctx = {
            "prev_spx_return": 0.0,
            "prev_spx_return_2d": 0.0,
            "prev_vix_pct_change": 0.20,
            "vix": 15.0,
            "term_structure": 0.9,
        }
        assert "vix_spike" in detector._evaluate(ctx)

    def test_vix_elevated(self, detector: EventSignalDetector) -> None:
        ctx = {
            "prev_spx_return": 0.0,
            "prev_spx_return_2d": 0.0,
            "prev_vix_pct_change": 0.0,
            "vix": 30.0,
            "term_structure": 0.9,
        }
        assert "vix_elevated" in detector._evaluate(ctx)

    def test_term_inversion(self, detector: EventSignalDetector) -> None:
        ctx = {
            "prev_spx_return": 0.0,
            "prev_spx_return_2d": 0.0,
            "prev_vix_pct_change": 0.0,
            "vix": 15.0,
            "term_structure": 1.15,
        }
        assert "term_inversion" in detector._evaluate(ctx)

    def test_rally_avoidance(self, detector: EventSignalDetector) -> None:
        ctx = {
            "prev_spx_return": 0.02,
            "prev_spx_return_2d": 0.0,
            "prev_vix_pct_change": 0.0,
            "vix": 15.0,
            "term_structure": 0.9,
        }
        assert "rally" in detector._evaluate(ctx)

    def test_rally_avoidance_disabled(self) -> None:
        det = EventSignalDetector(
            spx_drop_threshold=-0.01,
            spx_drop_2d_threshold=-0.02,
            vix_spike_threshold=0.15,
            vix_elevated_threshold=25.0,
            term_inversion_threshold=1.0,
            rally_avoidance=False,
            rally_threshold=0.01,
        )
        ctx = {"prev_spx_return": 0.05, "prev_spx_return_2d": 0.0,
               "prev_vix_pct_change": 0.0, "vix": 15.0, "term_structure": 0.9}
        assert "rally" not in det._evaluate(ctx)

    def test_multiple_signals_fire_together(self, detector: EventSignalDetector) -> None:
        """Stress day: big SPX drop + VIX spike + VIX elevated + term inversion."""
        ctx = {
            "prev_spx_return": -0.03,
            "prev_spx_return_2d": -0.05,
            "prev_vix_pct_change": 0.25,
            "vix": 35.0,
            "term_structure": 1.2,
        }
        signals = detector._evaluate(ctx)
        assert set(signals) == {"spx_drop_1d", "spx_drop_2d", "vix_spike", "vix_elevated", "term_inversion"}

    def test_none_values_are_safe(self, detector: EventSignalDetector) -> None:
        """Missing data should not trigger any signals."""
        ctx = {
            "prev_spx_return": None,
            "prev_spx_return_2d": None,
            "prev_vix_pct_change": None,
            "vix": None,
            "term_structure": None,
        }
        assert detector._evaluate(ctx) == []

    def test_empty_context(self, detector: EventSignalDetector) -> None:
        assert detector._evaluate({}) == []

    def test_boundary_values_not_triggered(self, detector: EventSignalDetector) -> None:
        """Values exactly at threshold should NOT fire (strict inequality)."""
        ctx = {
            "prev_spx_return": -0.01,       # == threshold, not < threshold
            "prev_spx_return_2d": -0.02,     # == threshold
            "prev_vix_pct_change": 0.15,     # == threshold, not > threshold
            "vix": 25.0,                     # == threshold
            "term_structure": 1.0,           # == threshold
        }
        assert detector._evaluate(ctx) == []


class TestDetectFromDict:
    """Tests for the synchronous detect_from_dict() public API."""

    def test_returns_signal_list(self, detector: EventSignalDetector) -> None:
        ctx = {"prev_spx_return": -0.02, "prev_spx_return_2d": 0.0,
               "prev_vix_pct_change": 0.0, "vix": 15.0, "term_structure": 0.9}
        result = detector.detect_from_dict(ctx)
        assert isinstance(result, list)
        assert "spx_drop_1d" in result

    def test_calm_market_returns_empty(self, detector: EventSignalDetector) -> None:
        ctx = {"prev_spx_return": 0.001, "prev_spx_return_2d": 0.002,
               "prev_vix_pct_change": 0.01, "vix": 14.0, "term_structure": 0.85}
        assert detector.detect_from_dict(ctx) == []
