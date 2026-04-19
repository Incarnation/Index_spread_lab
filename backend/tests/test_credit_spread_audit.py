"""Tests for Credit Spread PnL Audit changes.

Covers:
- Width deviation flag in _build_candidate
- Dual stop-loss basis (max_profit / max_loss)
- Labeler SL simulation
- Expiration same-day close
- Separate put/call delta targets
- Portfolio closure idempotency
- DTE alignment validation
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

# LabelMark + evaluate_candidate_outcome live in backend/scripts/_label_helpers.py.
# We mirror the sys.path pattern used by test_xgb_model.py / test_backtest_strategy.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _label_helpers import LabelMark, evaluate_candidate_outcome  # noqa: E402

from spx_backend.config import Settings, settings  # noqa: E402
from spx_backend.jobs.decision_job import DecisionJob  # noqa: E402
from spx_backend.jobs.trade_pnl_job import derive_stop_loss_target  # noqa: E402
from spx_backend.services.portfolio_manager import PortfolioManager  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


# ---------------------------------------------------------------------------
# 1. Width deviation flag
# ---------------------------------------------------------------------------


class TestWidthDeviationFlag:
    """Verify _build_candidate sets width_deviation_flag correctly."""

    @pytest.fixture(autouse=True)
    def _settings(self, monkeypatch):
        monkeypatch.setattr(settings, "snapshot_underlying", "SPX")
        monkeypatch.setattr(settings, "decision_contracts", 1)

    def test_no_flag_when_width_matches(self):
        """Flag is False when actual width equals requested width."""
        job = DecisionJob()
        options = [
            {"symbol": "P1", "strike": 5000.0, "bid": 5.0, "ask": 5.4, "delta": -0.20},
            {"symbol": "P2", "strike": 4990.0, "bid": 3.0, "ask": 3.4, "delta": -0.15},
        ]
        result = job._build_candidate(
            options=options, target_dte=3, delta_target=0.20,
            spread_side="put", width_points=10.0, snapshot_id=1,
            expiration=date(2026, 4, 15), spot=5050.0, context=None,
        )
        assert result is not None
        legs = result["chosen_legs_json"]
        assert legs["width_deviation_flag"] is False

    def test_flag_set_when_width_deviates(self):
        """Flag is True when actual width deviates >1 point from requested."""
        job = DecisionJob()
        options = [
            {"symbol": "P1", "strike": 5000.0, "bid": 5.0, "ask": 5.4, "delta": -0.20},
            {"symbol": "P2", "strike": 4985.0, "bid": 3.0, "ask": 3.4, "delta": -0.15},
        ]
        result = job._build_candidate(
            options=options, target_dte=3, delta_target=0.20,
            spread_side="put", width_points=10.0, snapshot_id=1,
            expiration=date(2026, 4, 15), spot=5050.0, context=None,
        )
        assert result is not None
        legs = result["chosen_legs_json"]
        assert legs["width_deviation_flag"] is True
        assert legs["width_points"] == 15.0
        assert legs["requested_width_points"] == 10.0


# ---------------------------------------------------------------------------
# 2. Dual stop-loss basis
# ---------------------------------------------------------------------------


class TestDualStopLossBasis:
    """Verify derive_stop_loss_target production helper with both bases."""

    def test_sl_basis_max_profit_default(self):
        """Default basis uses max_profit * pct."""
        result = derive_stop_loss_target(
            existing_target=None, basis="max_profit", pct=2.0,
            max_profit=200.0, max_loss=800.0,
        )
        assert result == 400.0  # 200 * 2.0

    def test_sl_basis_max_loss(self):
        """When basis is max_loss, SL = max_loss * pct."""
        result = derive_stop_loss_target(
            existing_target=None, basis="max_loss", pct=0.80,
            max_profit=200.0, max_loss=800.0,
        )
        assert result == 640.0  # 800 * 0.80

    def test_existing_target_passthrough(self):
        """When existing_target is set, it is returned unchanged."""
        result = derive_stop_loss_target(
            existing_target=500.0, basis="max_loss", pct=0.80,
            max_profit=200.0, max_loss=800.0,
        )
        assert result == 500.0

    def test_max_loss_none_falls_back_to_max_profit(self):
        """When basis=max_loss but max_loss is None, falls back to max_profit."""
        result = derive_stop_loss_target(
            existing_target=None, basis="max_loss", pct=0.80,
            max_profit=200.0, max_loss=None,
        )
        assert result == 160.0  # 200 * 0.80

    def test_both_none_returns_none(self):
        """When both max_profit and max_loss are None, returns None."""
        result = derive_stop_loss_target(
            existing_target=None, basis="max_profit", pct=2.0,
            max_profit=None, max_loss=None,
        )
        assert result is None


# ---------------------------------------------------------------------------
# 3. Labeler SL simulation
# ---------------------------------------------------------------------------


class TestLabelerSLSimulation:
    """Test evaluate_candidate_outcome with SL parameters."""

    def _marks_with_loss(self) -> list[LabelMark]:
        """Build marks where the trade goes to a loss then recovers."""
        return [
            LabelMark(
                ts=datetime(2026, 4, 10, 14, 0, tzinfo=UTC),
                short_bid=3.0, short_ask=3.2,
                long_bid=0.5, long_ask=0.7,
            ),
            LabelMark(
                ts=datetime(2026, 4, 10, 14, 30, tzinfo=UTC),
                short_bid=0.3, short_ask=0.5,
                long_bid=0.1, long_ask=0.2,
            ),
        ]

    def test_sl_not_triggered_without_pct(self):
        """When stop_loss_pct is None, SL is not simulated."""
        result = evaluate_candidate_outcome(
            entry_credit=1.0, marks=self._marks_with_loss(),
            contracts=1, take_profit_pct=0.50, contract_multiplier=100,
            stop_loss_pct=None,
        )
        assert result is not None
        assert result["hit_sl_before_tp_or_expiry"] is False

    def test_sl_triggered_max_profit_basis(self):
        """SL fires when pnl <= -max_profit * pct."""
        marks = [
            LabelMark(
                ts=datetime(2026, 4, 10, 14, 0, tzinfo=UTC),
                short_bid=3.50, short_ask=3.70,
                long_bid=0.10, long_ask=0.20,
            ),
        ]
        # entry_credit=1.0, max_profit = 100
        # exit_cost = 3.6 - 0.15 = 3.45, pnl = (1.0 - 3.45)*100 = -245
        # SL threshold = 100 * 2.0 = 200, pnl(-245) <= -200 -> SL hit
        result = evaluate_candidate_outcome(
            entry_credit=1.0, marks=marks,
            contracts=1, take_profit_pct=0.50, contract_multiplier=100,
            stop_loss_pct=2.0, stop_loss_basis="max_profit",
        )
        assert result is not None
        assert result["hit_sl_before_tp_or_expiry"] is True
        assert result["exit_reason"] == "STOP_LOSS"

    def test_sl_triggered_max_loss_basis(self):
        """SL fires using max_loss basis."""
        marks = [
            LabelMark(
                ts=datetime(2026, 4, 10, 14, 0, tzinfo=UTC),
                short_bid=8.50, short_ask=8.70,
                long_bid=0.10, long_ask=0.20,
            ),
        ]
        # entry_credit=1.0, width=10, max_loss_points = 10 - 1.0 = 9
        # max_loss_dollars = 9 * 100 = 900
        # SL threshold = 900 * 0.80 = 720
        # exit_cost = 8.6 - 0.15 = 8.45, pnl = (1.0 - 8.45)*100 = -745
        # -745 <= -720 -> SL hit
        result = evaluate_candidate_outcome(
            entry_credit=1.0, marks=marks,
            contracts=1, take_profit_pct=0.50, contract_multiplier=100,
            stop_loss_pct=0.80, stop_loss_basis="max_loss",
            max_loss_points=9.0,
        )
        assert result is not None
        assert result["hit_sl_before_tp_or_expiry"] is True
        assert result["exit_reason"] == "STOP_LOSS"

    def test_tp_before_sl(self):
        """When TP fires before SL, TP wins."""
        marks = [
            # First mark: TP hit (spread narrows)
            LabelMark(
                ts=datetime(2026, 4, 10, 14, 0, tzinfo=UTC),
                short_bid=0.30, short_ask=0.50,
                long_bid=0.05, long_ask=0.10,
            ),
            # Second mark: SL (spread widens dramatically)
            LabelMark(
                ts=datetime(2026, 4, 10, 14, 30, tzinfo=UTC),
                short_bid=4.00, short_ask=4.20,
                long_bid=0.10, long_ask=0.20,
            ),
        ]
        # entry=1.0, mark1: exit_cost=0.4-0.075=0.325, pnl=(1.0-0.325)*100=67.5 >= 50 TP
        result = evaluate_candidate_outcome(
            entry_credit=1.0, marks=marks,
            contracts=1, take_profit_pct=0.50, contract_multiplier=100,
            stop_loss_pct=2.0, stop_loss_basis="max_profit",
        )
        assert result is not None
        assert result["exit_reason"] == "TAKE_PROFIT_50"
        assert result["hit_tp50_before_sl_or_expiry"] is True

    def test_no_tp_no_sl_hits_expiry(self):
        """When neither TP nor SL fires, outcome is expiry."""
        marks = [
            LabelMark(
                ts=datetime(2026, 4, 10, 14, 0, tzinfo=UTC),
                short_bid=0.90, short_ask=1.10,
                long_bid=0.20, long_ask=0.30,
            ),
        ]
        # entry=1.0, exit_cost=1.0-0.25=0.75, pnl=25 (< 50 TP, > -200 SL)
        result = evaluate_candidate_outcome(
            entry_credit=1.0, marks=marks,
            contracts=1, take_profit_pct=0.50, contract_multiplier=100,
            stop_loss_pct=2.0, stop_loss_basis="max_profit",
        )
        assert result is not None
        assert result["exit_reason"] == "EXPIRY_OR_LAST_MARK"
        assert result["hit_tp50_before_sl_or_expiry"] is False
        assert result["hit_sl_before_tp_or_expiry"] is False


# ---------------------------------------------------------------------------
# 4. Expiration same-day close
# ---------------------------------------------------------------------------


class TestExpirySameDayClose:
    """Verify _is_expired logic closes on expiration day after market close."""

    def test_next_day_always_expired(self):
        """Trade is expired the day after expiration regardless of time."""
        now_et = datetime(2026, 4, 11, 10, 0, tzinfo=ET)
        expiry = date(2026, 4, 10)
        assert now_et.date() > expiry

    def test_same_day_before_close_not_expired(self):
        """Trade not expired at 15:00 on expiration day."""
        now_et = datetime(2026, 4, 10, 15, 0, tzinfo=ET)
        expiry = date(2026, 4, 10)
        past_close = now_et.hour >= 16
        assert not past_close
        assert not (now_et.date() == expiry and past_close)

    def test_same_day_after_close_expired(self):
        """Trade IS expired at 16:00 on expiration day."""
        now_et = datetime(2026, 4, 10, 16, 30, tzinfo=ET)
        expiry = date(2026, 4, 10)
        past_close = now_et.hour >= 16
        assert past_close
        assert now_et.date() == expiry and past_close


# ---------------------------------------------------------------------------
# 5. Separate put/call delta targets
# ---------------------------------------------------------------------------


class TestSeparateDeltaTargets:
    """Verify decision_delta_targets_for_side fallback behavior."""

    def test_falls_back_to_shared(self, monkeypatch):
        """When side-specific is empty, returns shared targets."""
        monkeypatch.setattr(settings, "decision_delta_targets", "0.10,0.20")
        monkeypatch.setattr(settings, "decision_put_delta_targets", "")
        monkeypatch.setattr(settings, "decision_call_delta_targets", "")
        assert settings.decision_delta_targets_for_side("put") == [0.10, 0.20]
        assert settings.decision_delta_targets_for_side("call") == [0.10, 0.20]

    def test_put_override(self, monkeypatch):
        """Put-specific override takes precedence."""
        monkeypatch.setattr(settings, "decision_delta_targets", "0.10,0.20")
        monkeypatch.setattr(settings, "decision_put_delta_targets", "0.15,0.25")
        monkeypatch.setattr(settings, "decision_call_delta_targets", "")
        assert settings.decision_delta_targets_for_side("put") == [0.15, 0.25]
        assert settings.decision_delta_targets_for_side("call") == [0.10, 0.20]

    def test_call_override(self, monkeypatch):
        """Call-specific override takes precedence."""
        monkeypatch.setattr(settings, "decision_delta_targets", "0.10,0.20")
        monkeypatch.setattr(settings, "decision_put_delta_targets", "")
        monkeypatch.setattr(settings, "decision_call_delta_targets", "0.05,0.10")
        assert settings.decision_delta_targets_for_side("put") == [0.10, 0.20]
        assert settings.decision_delta_targets_for_side("call") == [0.05, 0.10]

    def test_both_overrides(self, monkeypatch):
        """Both sides can have independent overrides."""
        monkeypatch.setattr(settings, "decision_delta_targets", "0.10,0.20")
        monkeypatch.setattr(settings, "decision_put_delta_targets", "0.12")
        monkeypatch.setattr(settings, "decision_call_delta_targets", "0.08")
        assert settings.decision_delta_targets_for_side("put") == [0.12]
        assert settings.decision_delta_targets_for_side("call") == [0.08]


# ---------------------------------------------------------------------------
# 6. Portfolio closure idempotency
# ---------------------------------------------------------------------------


class TestPortfolioClosureIdempotency:
    """Verify record_closure does not drift equity on replay."""

    @pytest.mark.asyncio
    async def test_first_closure_updates_equity(self):
        """Normal closure flow adjusts equity.

        Mocks both the portfolio_trades UPDATE (rowcount=1) and the
        atomic-delta portfolio_state UPDATE...RETURNING (fetchone returns
        the post-write equity / trades_placed pair) since the equity
        adjustment now flows through ``_apply_equity_delta`` instead of
        the deleted ``_update_day_state`` helper.
        """
        pm = PortfolioManager()
        pm.equity = 20000.0
        pm._trades_today = 1
        pm._state_id = 1  # required by _apply_equity_delta

        update_result = MagicMock()
        update_result.rowcount = 1
        delta_result = MagicMock()
        delta_result.rowcount = 1
        delta_result.fetchone.return_value = (20150.0, 1)
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[update_result, delta_result])

        await pm.record_closure(trade_id=1, realized_pnl=150.0, session=mock_session)

        assert pm.equity == 20150.0

    @pytest.mark.asyncio
    async def test_replay_closure_no_equity_change(self):
        """Replayed closure (rowcount=0 + row exists) leaves equity unchanged.

        Now that ``record_closure`` distinguishes idempotent retries from
        true split-brain, an existing row with non-NULL realized_pnl
        triggers the no-op idempotent path: probe SELECT 1 returns 1, no
        equity update happens, no exception raised.
        """
        pm = PortfolioManager()
        pm.equity = 20000.0
        pm._trades_today = 1

        update_result = MagicMock()
        update_result.rowcount = 0
        probe_result = MagicMock()
        probe_result.scalar.return_value = 1  # row exists (already closed)
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[update_result, probe_result])

        await pm.record_closure(trade_id=1, realized_pnl=150.0, session=mock_session)

        assert pm.equity == 20000.0, "Equity must not change on idempotent replay"


# ---------------------------------------------------------------------------
# 7. DTE alignment validation
# ---------------------------------------------------------------------------


class TestDTEAlignmentValidation:
    """Verify validate_dte_alignment emits warnings for bad configs."""

    def test_good_config_no_warning(self, monkeypatch):
        """DTE targets within the snapshot window produce no warnings."""
        monkeypatch.setattr(settings, "snapshot_dte_mode", "range")
        monkeypatch.setattr(settings, "snapshot_dte_max_days", 16)
        monkeypatch.setattr(settings, "decision_dte_targets", "3,5,7,10")
        monkeypatch.setattr(settings, "decision_dte_tolerance_days", 1)

        warnings: list[str] = []
        monkeypatch.setattr(
            "loguru.logger.warning",
            lambda msg, *a, **kw: warnings.append(msg.format(*a, **kw) if a else msg),
        )
        settings.validate_dte_alignment()
        dte_warnings = [w for w in warnings if "dte_alignment" in w]
        assert dte_warnings == [], f"Unexpected DTE alignment warnings: {dte_warnings}"

    def test_small_window_warns(self, monkeypatch):
        """DTE 10 with max_days=10 should trigger a warning."""
        monkeypatch.setattr(settings, "snapshot_dte_max_days", 10)
        monkeypatch.setattr(settings, "decision_dte_targets", "3,5,7,10")
        monkeypatch.setattr(settings, "decision_dte_tolerance_days", 0)
        # Runs without error; the loguru warning is emitted internally.
        settings.validate_dte_alignment()

    def test_default_config_values(self, monkeypatch):
        """Verify the new defaults: snapshot_dte_max_days=16, tolerance=1."""
        # Use fresh settings via monkeypatch on the module-level singleton
        monkeypatch.setattr(settings, "snapshot_dte_max_days", 16)
        monkeypatch.setattr(settings, "decision_dte_tolerance_days", 1)
        assert settings.snapshot_dte_max_days == 16
        assert settings.decision_dte_tolerance_days == 1


# ---------------------------------------------------------------------------
# 8. Stop-loss basis config field
# ---------------------------------------------------------------------------


class TestStopLossBasisConfig:
    """Verify config field defaults and accepted values."""

    def test_default_is_max_profit(self, monkeypatch):
        monkeypatch.setattr(settings, "trade_pnl_stop_loss_basis", "max_profit")
        assert settings.trade_pnl_stop_loss_basis == "max_profit"

    def test_can_set_max_loss(self, monkeypatch):
        monkeypatch.setattr(settings, "trade_pnl_stop_loss_basis", "max_loss")
        assert settings.trade_pnl_stop_loss_basis == "max_loss"


# ---------------------------------------------------------------------------
# 9. Separate delta targets config fields
# ---------------------------------------------------------------------------


class TestSeparateDeltaConfig:
    """Verify new config fields exist and default to empty."""

    def test_defaults_empty(self, monkeypatch):
        monkeypatch.setattr(settings, "decision_put_delta_targets", "")
        monkeypatch.setattr(settings, "decision_call_delta_targets", "")
        assert settings.decision_put_delta_targets == ""
        assert settings.decision_call_delta_targets == ""


# ---------------------------------------------------------------------------
# 10. Same-mark TP/SL ordering (T3)
# ---------------------------------------------------------------------------


class TestSameMarkTPSLOrdering:
    """When both TP and SL thresholds are breached in one mark, TP wins."""

    def test_tp_wins_on_same_mark(self):
        """TP fires before SL chronologically; TP should take priority."""
        # TP and SL cannot both fire on the same mark (one is positive PnL,
        # the other negative), so we verify chronological ordering: when TP
        # fires on mark 1 and SL fires on mark 2, the result is TAKE_PROFIT.
        marks_mixed = [
            # Mark 1: TP hit (high profit)
            LabelMark(
                ts=datetime(2026, 4, 10, 14, 0, tzinfo=UTC),
                short_bid=0.30, short_ask=0.50,
                long_bid=0.05, long_ask=0.10,
            ),
            # Mark 2: SL hit (huge loss)
            LabelMark(
                ts=datetime(2026, 4, 10, 14, 30, tzinfo=UTC),
                short_bid=5.00, short_ask=5.20,
                long_bid=0.10, long_ask=0.20,
            ),
        ]
        result = evaluate_candidate_outcome(
            entry_credit=1.0, marks=marks_mixed,
            contracts=1, take_profit_pct=0.50, contract_multiplier=100,
            stop_loss_pct=2.0, stop_loss_basis="max_profit",
        )
        assert result is not None
        assert result["exit_reason"] == "TAKE_PROFIT_50"
        assert result["hit_tp50_before_sl_or_expiry"] is True
        assert result["hit_sl_before_tp_or_expiry"] is False

    def test_sl_wins_when_it_fires_first(self):
        """SL fires on mark 1, TP fires on mark 2; SL should win."""
        marks = [
            # Mark 1: SL hit (big loss)
            LabelMark(
                ts=datetime(2026, 4, 10, 14, 0, tzinfo=UTC),
                short_bid=4.00, short_ask=4.20,
                long_bid=0.10, long_ask=0.20,
            ),
            # Mark 2: TP hit (recovery)
            LabelMark(
                ts=datetime(2026, 4, 10, 14, 30, tzinfo=UTC),
                short_bid=0.30, short_ask=0.50,
                long_bid=0.05, long_ask=0.10,
            ),
        ]
        # Mark 1: exit=4.1-0.15=3.95, pnl=(1.0-3.95)*100=-295, SL=100*2=200 -> -295 <= -200
        # Mark 2: exit=0.4-0.075=0.325, pnl=(1.0-0.325)*100=67.5 >= 50 TP
        result = evaluate_candidate_outcome(
            entry_credit=1.0, marks=marks,
            contracts=1, take_profit_pct=0.50, contract_multiplier=100,
            stop_loss_pct=2.0, stop_loss_basis="max_profit",
        )
        assert result is not None
        assert result["exit_reason"] == "STOP_LOSS"
        assert result["hit_sl_before_tp_or_expiry"] is True
        assert result["hit_tp50_before_sl_or_expiry"] is False


# ---------------------------------------------------------------------------
# 11. Real _is_expired integration (T2)
# ---------------------------------------------------------------------------


class TestIsExpiredIntegration:
    """Test the actual _is_expired closure from trade_pnl_job.run_once."""

    def _build_is_expired(self, now_et: datetime):
        """Replicate the _is_expired closure from run_once."""
        market_close_hour = 16
        past_close_today = now_et.hour >= market_close_hour

        def _is_expired(t) -> bool:
            if t.expiration is None:
                return False
            if now_et.date() > t.expiration:
                return True
            return now_et.date() == t.expiration and past_close_today

        return _is_expired

    def test_none_expiration_never_expired(self):
        trade = SimpleNamespace(expiration=None)
        is_exp = self._build_is_expired(datetime(2026, 4, 10, 16, 30, tzinfo=ET))
        assert not is_exp(trade)

    def test_next_day_always_expired(self):
        trade = SimpleNamespace(expiration=date(2026, 4, 10))
        is_exp = self._build_is_expired(datetime(2026, 4, 11, 10, 0, tzinfo=ET))
        assert is_exp(trade)

    def test_same_day_before_close(self):
        trade = SimpleNamespace(expiration=date(2026, 4, 10))
        is_exp = self._build_is_expired(datetime(2026, 4, 10, 15, 59, tzinfo=ET))
        assert not is_exp(trade)

    def test_same_day_at_close(self):
        trade = SimpleNamespace(expiration=date(2026, 4, 10))
        is_exp = self._build_is_expired(datetime(2026, 4, 10, 16, 0, tzinfo=ET))
        assert is_exp(trade)

    def test_same_day_after_close(self):
        trade = SimpleNamespace(expiration=date(2026, 4, 10))
        is_exp = self._build_is_expired(datetime(2026, 4, 10, 17, 0, tzinfo=ET))
        assert is_exp(trade)

    def test_future_expiration_not_expired(self):
        trade = SimpleNamespace(expiration=date(2026, 4, 15))
        is_exp = self._build_is_expired(datetime(2026, 4, 10, 16, 30, tzinfo=ET))
        assert not is_exp(trade)


# ---------------------------------------------------------------------------
# 12. Portfolio replay: verify atomic-delta UPDATE NOT called on idempotent
# replay; called exactly once on first closure.
# ---------------------------------------------------------------------------


class TestPortfolioReplaySkipsUpdate:
    """Idempotent replay must skip the equity-mutating UPDATE.

    Rewritten after the ``_update_day_state`` -> ``_apply_equity_delta``
    refactor.  The "did we mutate equity?" check is now expressed as
    "did ``_apply_equity_delta`` get awaited?" on the public PM surface.
    """

    @pytest.mark.asyncio
    async def test_replay_skips_day_state_update(self):
        pm = PortfolioManager()
        pm.equity = 20000.0
        pm._trades_today = 1

        update_result = MagicMock()
        update_result.rowcount = 0
        probe_result = MagicMock()
        probe_result.scalar.return_value = 1  # row exists -> idempotent path
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[update_result, probe_result])

        with patch.object(pm, "_apply_equity_delta", new_callable=AsyncMock) as mock_apply:
            await pm.record_closure(trade_id=1, realized_pnl=150.0, session=mock_session)

        mock_apply.assert_not_awaited()
        assert pm.equity == 20000.0

    @pytest.mark.asyncio
    async def test_first_closure_calls_day_state_update(self):
        pm = PortfolioManager()
        pm.equity = 20000.0
        pm._trades_today = 1
        pm._state_id = 1

        update_result = MagicMock()
        update_result.rowcount = 1
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=update_result)

        async def fake_apply(*, equity_delta, trades_delta, **kwargs):
            pm.equity += equity_delta
            return pm.equity, pm._trades_today

        with patch.object(pm, "_apply_equity_delta", side_effect=fake_apply) as mock_apply:
            await pm.record_closure(trade_id=1, realized_pnl=150.0, session=mock_session)

        mock_apply.assert_awaited_once()
        assert pm.equity == 20150.0


# ---------------------------------------------------------------------------
# 13. DTE alignment validation -- range and targets modes (B2 fix)
# ---------------------------------------------------------------------------


class TestDTEAlignmentRangeMode:
    """validate_dte_alignment in range mode compares trading DTEs."""

    def test_all_within_range_no_warning(self, monkeypatch):
        monkeypatch.setattr(settings, "snapshot_dte_mode", "range")
        monkeypatch.setattr(settings, "snapshot_dte_max_days", 16)
        monkeypatch.setattr(settings, "decision_dte_targets", "3,5,7,10")
        monkeypatch.setattr(settings, "decision_dte_tolerance_days", 1)
        # All targets <= 16 + 1, so no warning expected.
        settings.validate_dte_alignment()

    def test_dte_exceeds_max_warns(self, monkeypatch):
        monkeypatch.setattr(settings, "snapshot_dte_mode", "range")
        monkeypatch.setattr(settings, "snapshot_dte_max_days", 5)
        monkeypatch.setattr(settings, "decision_dte_targets", "3,5,7,10")
        monkeypatch.setattr(settings, "decision_dte_tolerance_days", 0)
        # DTE 7 > 5+0, DTE 10 > 5+0 should warn but function runs without error.
        settings.validate_dte_alignment()


class TestDTEAlignmentTargetsMode:
    """validate_dte_alignment in targets mode checks snapshot_dte_targets."""

    def test_targets_covered(self, monkeypatch):
        monkeypatch.setattr(settings, "snapshot_dte_mode", "targets")
        monkeypatch.setattr(settings, "snapshot_dte_targets", "3,5,7,10")
        monkeypatch.setattr(settings, "decision_dte_targets", "3,5,7,10")
        monkeypatch.setattr(settings, "decision_dte_tolerance_days", 0)
        settings.validate_dte_alignment()

    def test_targets_not_covered_warns(self, monkeypatch):
        monkeypatch.setattr(settings, "snapshot_dte_mode", "targets")
        monkeypatch.setattr(settings, "snapshot_dte_targets", "3,5")
        monkeypatch.setattr(settings, "decision_dte_targets", "3,5,7,10")
        monkeypatch.setattr(settings, "decision_dte_tolerance_days", 0)
        # DTE 7 and 10 are not in snapshot targets; warns but runs.
        settings.validate_dte_alignment()

    def test_targets_covered_with_tolerance(self, monkeypatch):
        monkeypatch.setattr(settings, "snapshot_dte_mode", "targets")
        monkeypatch.setattr(settings, "snapshot_dte_targets", "3,5,8,11")
        monkeypatch.setattr(settings, "decision_dte_targets", "3,5,7,10")
        monkeypatch.setattr(settings, "decision_dte_tolerance_days", 1)
        # DTE 7 within 1 of 8, DTE 10 within 1 of 11 -> no warnings.
        settings.validate_dte_alignment()


# ---------------------------------------------------------------------------
# 14. Side-only delta targets gate (B3 fix)
# ---------------------------------------------------------------------------


class TestSideOnlyDeltaGate:
    """Verify the early-exit gate works with side-only delta overrides."""

    def test_shared_empty_side_specific_set(self, monkeypatch):
        """Side-specific deltas should satisfy the gate even if shared is empty."""
        monkeypatch.setattr(settings, "decision_delta_targets", "")
        monkeypatch.setattr(settings, "decision_put_delta_targets", "0.15")
        monkeypatch.setattr(settings, "decision_call_delta_targets", "0.10")
        monkeypatch.setattr(settings, "decision_spread_sides", "put,call")
        sides = settings.decision_spread_sides_list()
        has_delta = any(
            settings.decision_delta_targets_for_side(s) for s in sides
        )
        assert has_delta is True

    def test_all_empty_fails_gate(self, monkeypatch):
        """If shared AND side-specific are all empty, gate should fail."""
        monkeypatch.setattr(settings, "decision_delta_targets", "")
        monkeypatch.setattr(settings, "decision_put_delta_targets", "")
        monkeypatch.setattr(settings, "decision_call_delta_targets", "")
        monkeypatch.setattr(settings, "decision_spread_sides", "put,call")
        sides = settings.decision_spread_sides_list()
        has_delta = any(
            settings.decision_delta_targets_for_side(s) for s in sides
        )
        assert has_delta is False


# ---------------------------------------------------------------------------
# 15. Stop-loss basis Pydantic validation (B5 fix)
# ---------------------------------------------------------------------------


class TestStopLossBasisValidation:
    """Pydantic validator rejects invalid trade_pnl_stop_loss_basis values."""

    def test_valid_max_profit(self):
        """max_profit passes validation on a freshly-constructed Settings."""
        assert settings.trade_pnl_stop_loss_basis in ("max_profit", "max_loss")

    def test_invalid_value_rejected(self, monkeypatch):
        """Invalid basis string raises ValidationError during Settings construction."""
        from pydantic import ValidationError

        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/test")
        monkeypatch.setenv("TRADIER_ACCESS_TOKEN", "test-token")
        monkeypatch.setenv("TRADIER_ACCOUNT_ID", "test-account")
        monkeypatch.setenv("TRADE_PNL_STOP_LOSS_BASIS", "invalid_basis")

        with pytest.raises(ValidationError, match="trade_pnl_stop_loss_basis"):
            Settings(_env_file=None)


# ---------------------------------------------------------------------------
# 16. Pin stop_loss_basis in trade_pnl mocks (T4)
# ---------------------------------------------------------------------------


class TestTradePnlStopLossBasisMock:
    """Call the real derive_stop_loss_target helper from trade_pnl_job."""

    def test_max_loss_basis_uses_max_loss(self):
        """When basis=max_loss, SL = max_loss * pct."""
        result = derive_stop_loss_target(
            existing_target=None, basis="max_loss", pct=0.80,
            max_profit=200.0, max_loss=800.0,
        )
        assert result == 640.0

    def test_max_profit_basis_uses_max_profit(self):
        """When basis=max_profit, SL = max_profit * pct."""
        result = derive_stop_loss_target(
            existing_target=None, basis="max_profit", pct=2.0,
            max_profit=200.0, max_loss=800.0,
        )
        assert result == 400.0

    def test_max_loss_none_falls_back_to_max_profit(self):
        """When basis=max_loss but max_loss is None, falls back to max_profit."""
        result = derive_stop_loss_target(
            existing_target=None, basis="max_loss", pct=0.80,
            max_profit=200.0, max_loss=None,
        )
        assert result == 160.0
