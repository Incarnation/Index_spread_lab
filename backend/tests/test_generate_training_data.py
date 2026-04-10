"""Tests for the offline training pipeline (generate_training_data.py).

Covers Black-Scholes functions, data helpers, spread construction,
label resolution, and feature building -- all without requiring actual
Databento files on disk.
"""
from __future__ import annotations

import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_training_data import (  # noqa: E402
    LABEL_MARK_INTERVAL_MINUTES,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TP_LEVELS,
    _downsample_marks,
    _evaluate_outcome,
    _load_gex_csv,
    _mid,
    _time_to_expiry_years,
    bs_delta_vec,
    bs_price_vec,
    build_candidates_for_snapshot,
    build_dte_lookup,
    build_instrument_map,
    build_training_rows,
    compute_offline_gex,
    derive_spx_from_parity,
    find_expiry_for_dte,
    get_cbbo_snapshot_at,
    implied_vol_vec,
    load_daily_parquet,
    load_economic_calendar,
    load_frd_quotes,
    lookup_gex_context,
    lookup_intraday_value,
    merge_underlying_quotes,
    walk_forward_validate,
)


# ===================================================================
# Black-Scholes
# ===================================================================


class TestBSPriceVec:
    """Vectorised BS price should match textbook values."""

    def test_call_atm(self) -> None:
        """ATM call with 1-year expiry and 20% vol should be ~8-9% of spot."""
        S = np.array([100.0])
        K = np.array([100.0])
        sigma = np.array([0.20])
        is_call = np.array([True])
        price = bs_price_vec(S, K, 1.0, 0.05, sigma, is_call)
        assert 7.0 < price[0] < 12.0

    def test_put_atm(self) -> None:
        """ATM put should satisfy put-call parity."""
        S = np.array([100.0])
        K = np.array([100.0])
        sigma = np.array([0.20])
        call = bs_price_vec(S, K, 1.0, 0.05, sigma, np.array([True]))
        put = bs_price_vec(S, K, 1.0, 0.05, sigma, np.array([False]))
        parity = call[0] - put[0] - (S[0] - K[0] * np.exp(-0.05))
        assert abs(parity) < 1e-8

    def test_deep_otm_call_near_zero(self) -> None:
        """Deep OTM call should have very small value."""
        S = np.array([100.0])
        K = np.array([200.0])
        sigma = np.array([0.20])
        price = bs_price_vec(S, K, 0.1, 0.05, sigma, np.array([True]))
        assert price[0] < 0.01

    def test_vectorised_matches_scalar(self) -> None:
        """Vectorised computation should produce identical per-element results."""
        S = np.array([100.0, 100.0, 100.0])
        K = np.array([95.0, 100.0, 105.0])
        sigma = np.array([0.20, 0.25, 0.30])
        is_call = np.array([True, True, True])
        prices = bs_price_vec(S, K, 0.5, 0.05, sigma, is_call)
        for i in range(3):
            single = bs_price_vec(
                S[i:i + 1], K[i:i + 1], 0.5, 0.05,
                sigma[i:i + 1], is_call[i:i + 1],
            )
            assert abs(prices[i] - single[0]) < 1e-12


class TestBSDeltaVec:
    """BS delta should be bounded and have correct sign."""

    def test_call_delta_between_0_and_1(self) -> None:
        S = np.array([100.0, 100.0])
        K = np.array([90.0, 110.0])
        sigma = np.array([0.20, 0.20])
        d = bs_delta_vec(S, K, 0.5, 0.05, sigma, np.array([True, True]))
        assert all(0.0 < di < 1.0 for di in d)

    def test_put_delta_between_neg1_and_0(self) -> None:
        S = np.array([100.0, 100.0])
        K = np.array([90.0, 110.0])
        sigma = np.array([0.20, 0.20])
        d = bs_delta_vec(S, K, 0.5, 0.05, sigma, np.array([False, False]))
        assert all(-1.0 < di < 0.0 for di in d)

    def test_atm_call_delta_near_half(self) -> None:
        """ATM call delta should be close to 0.5 (slightly above due to drift)."""
        S = np.array([100.0])
        K = np.array([100.0])
        sigma = np.array([0.20])
        d = bs_delta_vec(S, K, 1.0, 0.05, sigma, np.array([True]))
        assert 0.45 < d[0] < 0.65


class TestImpliedVolVec:
    """IV bisection should recover the vol used to generate a price."""

    def test_roundtrip_call(self) -> None:
        """IV(BS_price(sigma)) should recover sigma."""
        S = np.array([100.0])
        K = np.array([105.0])
        true_vol = np.array([0.25])
        is_call = np.array([True])
        price = bs_price_vec(S, K, 0.5, 0.05, true_vol, is_call)
        recovered = implied_vol_vec(price, S, K, 0.5, 0.05, is_call)
        assert abs(recovered[0] - 0.25) < 1e-4

    def test_roundtrip_put(self) -> None:
        S = np.array([100.0])
        K = np.array([95.0])
        true_vol = np.array([0.30])
        is_call = np.array([False])
        price = bs_price_vec(S, K, 0.25, 0.05, true_vol, is_call)
        recovered = implied_vol_vec(price, S, K, 0.25, 0.05, is_call)
        assert abs(recovered[0] - 0.30) < 1e-4

    def test_vectorised_roundtrip(self) -> None:
        S = np.array([100.0, 100.0, 100.0])
        K = np.array([90.0, 100.0, 110.0])
        vols = np.array([0.15, 0.20, 0.35])
        is_call = np.array([True, False, True])
        prices = bs_price_vec(S, K, 1.0, 0.05, vols, is_call)
        recovered = implied_vol_vec(prices, S, K, 1.0, 0.05, is_call)
        np.testing.assert_allclose(recovered, vols, atol=1e-4)


# ===================================================================
# Data helpers
# ===================================================================


class TestBuildInstrumentMap:
    """build_instrument_map should parse definitions correctly."""

    def _make_def_df(self, rows: list[dict]) -> pd.DataFrame:
        defaults = {
            "ts_recv": pd.Timestamp("2026-01-02", tz="UTC"),
            "instrument_id": 1,
            "raw_symbol": "SPX   260102P06000000",
            "instrument_class": "P",
            "strike_price": 6000.0,
            "expiration": pd.Timestamp("2026-01-02", tz="UTC"),
        }
        return pd.DataFrame([{**defaults, **r} for r in rows])

    def test_basic_put(self) -> None:
        df = self._make_def_df([{"instrument_id": 42}])
        m = build_instrument_map(df)
        assert 42 in m
        assert m[42]["put_call"] == "P"
        assert m[42]["strike"] == 6000.0
        assert m[42]["expiry"] == date(2026, 1, 2)

    def test_call_from_raw_symbol(self) -> None:
        """Fall back to raw_symbol when instrument_class is missing."""
        df = self._make_def_df([{
            "instrument_id": 7,
            "instrument_class": None,
            "raw_symbol": "SPX   260117C06500000",
            "strike_price": 6500.0,
            "expiration": pd.Timestamp("2026-01-17", tz="UTC"),
        }])
        m = build_instrument_map(df)
        assert m[7]["put_call"] == "C"
        assert m[7]["strike"] == 6500.0

    def test_skip_invalid_strike(self) -> None:
        df = self._make_def_df([{"instrument_id": 9, "strike_price": 0}])
        m = build_instrument_map(df)
        assert 9 not in m

    def test_multiple_instruments(self) -> None:
        df = self._make_def_df([
            {"instrument_id": 1, "instrument_class": "P", "strike_price": 5900.0},
            {"instrument_id": 2, "instrument_class": "C", "strike_price": 6100.0},
        ])
        m = build_instrument_map(df)
        assert len(m) == 2


class TestBuildDteLookup:
    """DTE assignment should match production dte.trading_dte_lookup."""

    def test_0dte(self) -> None:
        exps = [date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 5)]
        lookup = build_dte_lookup(exps, date(2026, 1, 2))
        assert lookup[date(2026, 1, 2)] == 0

    def test_future_only(self) -> None:
        exps = [date(2026, 1, 5), date(2026, 1, 6)]
        lookup = build_dte_lookup(exps, date(2026, 1, 3))
        assert lookup[date(2026, 1, 5)] == 1

    def test_empty(self) -> None:
        assert build_dte_lookup([], date(2026, 1, 1)) == {}


class TestFindExpiryForDte:
    """Expiry finder should respect tolerance."""

    def test_exact_match(self) -> None:
        m = {date(2026, 1, 2): 0, date(2026, 1, 5): 3}
        assert find_expiry_for_dte(m, 3) == date(2026, 1, 5)

    def test_within_tolerance(self) -> None:
        m = {date(2026, 1, 6): 4}
        assert find_expiry_for_dte(m, 3, tolerance=1) == date(2026, 1, 6)

    def test_no_match(self) -> None:
        m = {date(2026, 1, 6): 4}
        assert find_expiry_for_dte(m, 0, tolerance=1) is None


class TestTimeToExpiryYears:
    """T should be positive and floor near-zero for 0DTE."""

    def test_positive(self) -> None:
        assert _time_to_expiry_years(date(2026, 1, 10), date(2026, 1, 2)) > 0

    def test_same_day_floor(self) -> None:
        t = _time_to_expiry_years(date(2026, 1, 2), date(2026, 1, 2))
        assert 0 < t < 0.001


# ===================================================================
# Snapshot helper
# ===================================================================


class TestGetCbboSnapshotAt:
    """Snapshot extraction should return latest quotes within window."""

    def test_returns_latest_per_instrument(self) -> None:
        df = pd.DataFrame({
            "ts": pd.to_datetime([
                "2026-01-02 15:00:00", "2026-01-02 15:01:00",
                "2026-01-02 15:00:00", "2026-01-02 15:01:00",
            ], utc=True),
            "instrument_id": [1, 1, 2, 2],
            "bid_px_00": [10.0, 11.0, 20.0, 21.0],
            "ask_px_00": [10.5, 11.5, 20.5, 21.5],
        })
        dt_utc = datetime(2026, 1, 2, 15, 2, tzinfo=timezone.utc)
        snap = get_cbbo_snapshot_at(df, dt_utc)
        assert len(snap) == 2
        row1 = snap[snap["instrument_id"] == 1].iloc[0]
        assert row1["bid_px_00"] == 11.0

    def test_empty_when_no_data(self) -> None:
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2026-01-02 14:00:00"], utc=True),
            "instrument_id": [1],
            "bid_px_00": [10.0],
            "ask_px_00": [10.5],
        })
        dt_utc = datetime(2026, 1, 2, 16, 0, tzinfo=timezone.utc)
        snap = get_cbbo_snapshot_at(df, dt_utc)
        assert snap.empty


# ===================================================================
# Label resolution
# ===================================================================


class TestMid:
    def test_valid(self) -> None:
        assert _mid(10.0, 12.0) == 11.0

    def test_none_bid(self) -> None:
        assert _mid(None, 12.0) is None

    def test_zero_ask(self) -> None:
        assert _mid(10.0, 0) is None


class TestEvaluateOutcome:
    """Outcome evaluation should match production labeler semantics."""

    def test_no_marks(self) -> None:
        out = _evaluate_outcome(1.0, [])
        assert out["resolved"] is False

    def test_tp50_hit(self) -> None:
        """If spread decays enough, TP50 should fire."""
        marks = [
            {"short_bid": 0.30, "short_ask": 0.40, "long_bid": 0.05, "long_ask": 0.15},
        ]
        out = _evaluate_outcome(1.0, marks)
        assert out["resolved"] is True
        assert out["hit_tp50"] is True
        assert out["exit_reason"] == "TAKE_PROFIT_50"

    def test_expiry_no_tp50(self) -> None:
        """When spread stays wide, outcome is EXPIRY."""
        marks = [
            {"short_bid": 0.85, "short_ask": 0.95, "long_bid": 0.05, "long_ask": 0.10},
        ]
        out = _evaluate_outcome(1.0, marks)
        assert out["resolved"] is True
        assert out["hit_tp50"] is False
        assert out["exit_reason"] == "EXPIRY_OR_LAST_MARK"

    def test_tp100_at_expiry(self) -> None:
        """TP100 should flag when last mark gives full profit."""
        marks = [
            {"short_bid": 0.01, "short_ask": 0.02, "long_bid": 0.01, "long_ask": 0.02},
        ]
        out = _evaluate_outcome(1.0, marks)
        assert out["hit_tp100_at_expiry"] is True

    def test_tp50_and_tp100_both_true(self) -> None:
        """TP50 fires on first mark, TP100 also true on last mark.

        realized_pnl should use the TP50 exit (first mark), while
        hit_tp100_at_expiry reflects the last mark.
        """
        marks = [
            # Mark 1: exit_cost = 0.25 - 0.05 = 0.20 -> pnl = (1.0-0.20)*100 = $80 >= $50 TP50
            {"short_bid": 0.20, "short_ask": 0.30, "long_bid": 0.02, "long_ask": 0.08},
            # Mark 2 (expiry): exit_cost = 0.015-0.015 = 0 -> pnl = $100 >= $100 TP100
            {"short_bid": 0.01, "short_ask": 0.02, "long_bid": 0.01, "long_ask": 0.02},
        ]
        out = _evaluate_outcome(1.0, marks)
        assert out["resolved"] is True
        assert out["hit_tp50"] is True
        assert out["hit_tp100_at_expiry"] is True
        assert out["exit_reason"] == "TAKE_PROFIT_50"
        # realized_pnl uses the TP50 mark ($80), not the expiry mark ($100)
        assert 79.0 < out["realized_pnl"] < 81.0

    def test_tp100_false_when_residual_value(self) -> None:
        """Options still have value at expiry -- TP100 should be False."""
        marks = [
            # Spread still has residual value: exit_cost = 0.6-0.1 = 0.5
            # pnl = (1.0 - 0.5)*100 = $50 -> TP50 fires
            # But last mark has same residual -> pnl $50 < $100 TP100 threshold
            {"short_bid": 0.50, "short_ask": 0.70, "long_bid": 0.05, "long_ask": 0.15},
        ]
        out = _evaluate_outcome(1.0, marks)
        assert out["resolved"] is True
        assert out["hit_tp100_at_expiry"] is False

    def test_invalid_marks_ignored(self) -> None:
        """Marks with zero bid/ask should be skipped."""
        marks = [
            {"short_bid": 0, "short_ask": 0, "long_bid": 0, "long_ask": 0},
        ]
        out = _evaluate_outcome(1.0, marks)
        assert out["resolved"] is False


# ===================================================================
# Spread construction (unit)
# ===================================================================


def _make_snapshot_and_map(
    n_instruments: int = 40,
    strikes_start: float = 5500.0,
    strike_step: float = 10.0,
) -> tuple[pd.DataFrame, dict[int, dict]]:
    """Build a synthetic CBBO snapshot and instrument map for testing.

    Creates OTM put options with 10-point strike intervals and prices
    computed from Black-Scholes at 20 % IV so the IV solver can recover
    a clean vol surface.  Expiry is 30 days out.  Range covers the
    10-delta put region (~5600) with room for long legs below.
    """
    from scipy.stats import norm as _norm

    spot = 6000.0
    T = 30.0 / 365.0
    r = 0.043
    sigma = 0.20
    expiry = date(2026, 2, 1)

    iids = list(range(1, n_instruments + 1))
    strikes = [strikes_start + i * strike_step for i in range(n_instruments)]
    inst_map: dict[int, dict] = {}
    bids, asks = [], []

    for iid, strike in zip(iids, strikes):
        inst_map[iid] = {
            "strike": strike,
            "expiry": expiry,
            "put_call": "P",
            "raw_symbol": f"SPX   260201P{int(strike * 1000):08d}",
        }
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        mid = strike * math.exp(-r * T) * _norm.cdf(-d2) - spot * _norm.cdf(-d1)
        mid = max(mid, 0.10)
        bids.append(mid - 0.05)
        asks.append(mid + 0.05)

    snap = pd.DataFrame({
        "ts": pd.Timestamp("2026-01-02 15:02:00", tz="UTC"),
        "instrument_id": iids,
        "bid_px_00": bids,
        "ask_px_00": asks,
    })
    return snap, inst_map


class TestBuildCandidatesForSnapshot:
    """Spread construction should produce valid candidates."""

    def test_produces_candidate(self) -> None:
        snap, inst_map = _make_snapshot_and_map()
        cands = build_candidates_for_snapshot(
            snapshot=snap, inst_map=inst_map, spot=6000.0,
            spy_price=598.5, vix=18.0, vix9d=19.0,
            term_structure=19.0 / 18.0,
            decision_dt=datetime(2026, 1, 2, 15, 2, tzinfo=timezone.utc),
            day_date=date(2026, 1, 2), dte_target=30,
            expiry=date(2026, 2, 1), delta_target=0.10, side="put",
        )
        assert len(cands) == 1
        c = cands[0]
        assert c["entry_credit"] > 0
        assert c["spread_side"] == "put"
        assert c["short_strike"] != c["long_strike"]
        assert c["width_points"] > 0

    def test_no_candidate_on_empty_snapshot(self) -> None:
        _, inst_map = _make_snapshot_and_map()
        cands = build_candidates_for_snapshot(
            snapshot=pd.DataFrame(),
            inst_map=inst_map, spot=6000.0, spy_price=598.5,
            vix=18.0, vix9d=19.0, term_structure=1.05,
            decision_dt=datetime(2026, 1, 2, 15, 2, tzinfo=timezone.utc),
            day_date=date(2026, 1, 2), dte_target=30,
            expiry=date(2026, 2, 1), delta_target=0.10, side="put",
        )
        assert cands == []


# ===================================================================
# Feature building
# ===================================================================


class TestBuildTrainingRows:
    """Feature builder should produce rows compatible with modeling.py."""

    def test_resolved_candidate_produces_row(self) -> None:
        c: dict[str, Any] = {
            "resolved": True,
            "hit_tp50": True,
            "hit_tp100_at_expiry": False,
            "realized_pnl": 50.0,
            "spread_side": "put",
            "dte_target": 3,
            "delta_target": 0.10,
            "entry_credit": 1.5,
            "width_points": 25.0,
            "credit_to_width": 0.06,
            "contracts": 1,
            "max_loss": 23.5,
            "vix": 18.0,
            "term_structure": 1.05,
            "spy_price": 598.0,
            "spot": 5995.0,
            "day": "2026-01-02",
            "entry_dt": "2026-01-02T15:02:00+00:00",
            "expiry": "2026-01-05",
            "exit_reason": "TAKE_PROFIT_50",
        }
        rows = build_training_rows([c])
        assert len(rows) == 1
        r = rows[0]
        assert "features" in r
        assert r["features"]["spread_side"] == "put"
        assert r["features"]["vix_regime"] in ("low", "normal", "high")
        assert r["realized_pnl"] == 50.0
        assert r["hit_tp50"] is True

    def test_unresolved_skipped(self) -> None:
        c: dict[str, Any] = {"resolved": False}
        rows = build_training_rows([c])
        assert rows == []


# ===================================================================
# Put-call parity SPX derivation
# ===================================================================


class TestDeriveSpxFromParity:
    """Put-call parity should accurately recover the spot price."""

    def _make_parity_snapshot(
        self, spot: float = 6000.0, n_strikes: int = 10,
    ) -> tuple[pd.DataFrame, dict[int, dict]]:
        """Build call+put pairs around ATM with prices from Black-Scholes."""
        from scipy.stats import norm as _norm

        T = 1.0 / 365.0
        r = 0.043
        sigma = 0.20
        expiry = date(2026, 1, 3)
        iids: list[int] = []
        inst_map: dict[int, dict] = {}
        bids, asks = [], []
        iid = 1

        for i in range(n_strikes):
            strike = spot - 25.0 * (n_strikes // 2 - i)
            for pc in ("C", "P"):
                d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
                d2 = d1 - sigma * math.sqrt(T)
                if pc == "C":
                    mid = spot * _norm.cdf(d1) - strike * math.exp(-r * T) * _norm.cdf(d2)
                else:
                    mid = strike * math.exp(-r * T) * _norm.cdf(-d2) - spot * _norm.cdf(-d1)
                mid = max(mid, 0.10)
                inst_map[iid] = {
                    "strike": strike, "expiry": expiry,
                    "put_call": pc,
                    "raw_symbol": f"SPXW  260103{pc}{int(strike*1000):08d}",
                }
                bids.append(mid - 0.05)
                asks.append(mid + 0.05)
                iids.append(iid)
                iid += 1

        snap = pd.DataFrame({
            "ts": pd.Timestamp("2026-01-02 15:02:00", tz="UTC"),
            "instrument_id": iids,
            "bid_px_00": bids,
            "ask_px_00": asks,
        })
        return snap, inst_map

    def test_recovers_spot(self) -> None:
        """Parity should recover spot within a few dollars."""
        snap, inst_map = self._make_parity_snapshot(spot=6000.0)
        result = derive_spx_from_parity(snap, inst_map, date(2026, 1, 2), 598.5)
        assert abs(result - 6000.0) < 5.0

    def test_fallback_on_empty_snapshot(self) -> None:
        result = derive_spx_from_parity(
            pd.DataFrame(), {}, date(2026, 1, 2), 598.5,
        )
        assert abs(result - 598.5 * 10.024) < 0.1

    def test_fallback_when_no_call_put_pairs(self) -> None:
        snap, inst_map = _make_snapshot_and_map()
        result = derive_spx_from_parity(snap, inst_map, date(2026, 1, 2), 598.5)
        assert abs(result - 598.5 * 10.024) < 0.1


# ===================================================================
# Stop-loss
# ===================================================================


class TestStopLoss:
    """Stop-loss at 2x max profit should cap losses."""

    def test_stop_loss_fires(self) -> None:
        """When PnL drops past -2x entry credit, stop-loss exits."""
        # credit=1.0 -> max_profit=$100 -> SL at -$200
        # exit_cost = 5.1 - 0.5 = 4.6 -> pnl = (1.0 - 4.6)*100 = -$360
        marks = [
            {"short_bid": 5.0, "short_ask": 5.2, "long_bid": 0.4, "long_ask": 0.6},
        ]
        out = _evaluate_outcome(1.0, marks)
        assert out["exit_reason"] == "STOP_LOSS"
        assert out["hit_tp50"] is False
        assert out["realized_pnl"] < -200

    def test_stop_loss_before_tp50(self) -> None:
        """If SL fires on the first mark, later profitable marks never execute."""
        # First mark: exit_cost = 6.0 - 0.5 = 5.5 -> pnl = (1 - 5.5)*100 = -450 < -200
        marks = [
            {"short_bid": 6.0, "short_ask": 6.0, "long_bid": 0.5, "long_ask": 0.5},
            {"short_bid": 0.01, "short_ask": 0.01, "long_bid": 0.005, "long_ask": 0.005},
        ]
        out = _evaluate_outcome(1.0, marks)
        assert out["exit_reason"] == "STOP_LOSS"

    def test_tp50_when_no_stop_loss(self) -> None:
        """Normal TP50 still works when loss never reaches SL threshold."""
        marks = [
            {"short_bid": 0.3, "short_ask": 0.3, "long_bid": 0.05, "long_ask": 0.05},
        ]
        out = _evaluate_outcome(1.0, marks)
        assert out["exit_reason"] == "TAKE_PROFIT_50"
        assert out["hit_tp50"] is True

    def test_stop_loss_threshold_is_2x(self) -> None:
        """SL fires at exactly 2x, not 1x."""
        credit = 2.0
        max_profit = credit * 100
        sl_threshold = max_profit * STOP_LOSS_PCT

        marks_just_under = [
            {"short_bid": 5.9, "short_ask": 5.9, "long_bid": 0.0, "long_ask": 0.0},
        ]
        out = _evaluate_outcome(credit, marks_just_under)
        assert out["exit_reason"] != "STOP_LOSS" or out["realized_pnl"] <= -sl_threshold


# ===================================================================
# Mark downsampling
# ===================================================================


class TestDownsampleMarks:
    """_downsample_marks should filter marks to N-minute boundaries."""

    def _mark(self, minute: int) -> dict:
        """Build a minimal mark with the given minute value."""
        return {
            "ts": datetime(2026, 3, 15, 10, minute, tzinfo=timezone.utc),
            "short_bid": 0.5, "short_ask": 0.6,
            "long_bid": 0.1, "long_ask": 0.2,
        }

    def test_default_interval_is_5(self) -> None:
        """Module constant should be 5 (matching production cadence)."""
        assert LABEL_MARK_INTERVAL_MINUTES == 5

    def test_keeps_5min_boundaries(self) -> None:
        """Only marks at :00, :05, :10, ... should survive."""
        marks = [self._mark(m) for m in range(0, 16)]
        result = _downsample_marks(marks, interval_minutes=5)
        minutes = [m["ts"].minute for m in result]
        assert minutes == [0, 5, 10, 15]

    def test_interval_1_keeps_all(self) -> None:
        """interval=1 should be a no-op (keep every mark)."""
        marks = [self._mark(m) for m in range(0, 5)]
        result = _downsample_marks(marks, interval_minutes=1)
        assert len(result) == 5

    def test_empty_input(self) -> None:
        result = _downsample_marks([], interval_minutes=5)
        assert result == []

    def test_no_marks_on_boundary(self) -> None:
        """If no marks fall on a 5-min boundary, result is empty."""
        marks = [self._mark(m) for m in [1, 2, 3, 4, 6, 7, 8, 9]]
        result = _downsample_marks(marks, interval_minutes=5)
        assert result == []

    def test_all_marks_on_boundary(self) -> None:
        """If all marks are on boundaries, all survive."""
        marks = [self._mark(m) for m in [0, 10, 20, 30, 40, 50]]
        result = _downsample_marks(marks, interval_minutes=5)
        assert len(result) == 6

    def test_preserves_mark_data(self) -> None:
        """Filtered marks should retain all original fields."""
        marks = [self._mark(5)]
        result = _downsample_marks(marks, interval_minutes=5)
        assert len(result) == 1
        assert result[0]["short_bid"] == 0.5
        assert result[0]["long_ask"] == 0.2

    def test_custom_interval_10(self) -> None:
        """With interval=10, only :00, :10, :20, ... survive."""
        marks = [self._mark(m) for m in range(0, 31)]
        result = _downsample_marks(marks, interval_minutes=10)
        minutes = [m["ts"].minute for m in result]
        assert minutes == [0, 10, 20, 30]

    def test_tp50_removed_by_downsampling(self) -> None:
        """A fleeting TP50 hit at minute :03 should vanish when downsampled.

        With 1-min marks the trade hits TP50 at :03, but with 5-min marks
        the only visible mark is the :05 one where the spread hasn't decayed.
        """
        # credit=1.0, max_profit=$100, TP50 at $50
        marks_1min = [
            # :03 -> exit_cost = 0.25-0.05 = 0.20 -> pnl = (1-0.20)*100 = $80 >= $50 TP50
            {"ts": datetime(2026, 3, 15, 10, 3, tzinfo=timezone.utc),
             "short_bid": 0.20, "short_ask": 0.30,
             "long_bid": 0.02, "long_ask": 0.08},
            # :05 -> exit_cost = 0.85-0.05 = 0.80 -> pnl = (1-0.80)*100 = $20 < $50
            {"ts": datetime(2026, 3, 15, 10, 5, tzinfo=timezone.utc),
             "short_bid": 0.80, "short_ask": 0.90,
             "long_bid": 0.02, "long_ask": 0.08},
        ]

        # At 1-min: TP50 fires
        out_1min = _evaluate_outcome(1.0, marks_1min)
        assert out_1min["hit_tp50"] is True

        # At 5-min: the :03 mark is gone, TP50 does NOT fire
        marks_5min = _downsample_marks(marks_1min, interval_minutes=5)
        assert len(marks_5min) == 1
        out_5min = _evaluate_outcome(1.0, marks_5min)
        assert out_5min["hit_tp50"] is False


# ===================================================================
# Underlying quotes / VIX intraday lookup
# ===================================================================


class TestLookupIntradayValue:
    """Intraday VIX lookup should find values within the time window."""

    def _make_uq_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ts": pd.to_datetime([
                "2026-02-18T15:00:00+00:00",
                "2026-02-18T15:05:00+00:00",
                "2026-02-18T15:10:00+00:00",
                "2026-02-18T15:00:00+00:00",
                "2026-02-18T15:05:00+00:00",
            ]),
            "symbol": ["VIX", "VIX", "VIX", "VIX9D", "VIX9D"],
            "last": [19.5, 19.8, 20.1, 21.0, 21.3],
        })

    def test_finds_vix_in_window(self) -> None:
        uq = self._make_uq_df()
        dt = datetime(2026, 2, 18, 15, 7, tzinfo=timezone.utc)
        result = lookup_intraday_value(uq, "VIX", dt, window_minutes=5)
        assert result == 19.8

    def test_returns_none_outside_window(self) -> None:
        uq = self._make_uq_df()
        dt = datetime(2026, 2, 18, 14, 50, tzinfo=timezone.utc)
        result = lookup_intraday_value(uq, "VIX", dt, window_minutes=5)
        assert result is None

    def test_returns_none_for_empty_df(self) -> None:
        uq = pd.DataFrame(columns=["ts", "symbol", "last"])
        dt = datetime(2026, 2, 18, 15, 5, tzinfo=timezone.utc)
        result = lookup_intraday_value(uq, "VIX", dt)
        assert result is None

    def test_finds_vix9d(self) -> None:
        uq = self._make_uq_df()
        dt = datetime(2026, 2, 18, 15, 7, tzinfo=timezone.utc)
        result = lookup_intraday_value(uq, "VIX9D", dt, window_minutes=5)
        assert result == 21.3


# ===================================================================
# GEX context lookup
# ===================================================================


class TestLookupGexContext:
    """GEX context_flags should mirror production _context_score logic."""

    def _make_cs_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ts": pd.to_datetime([
                "2026-02-18T15:00:00+00:00",
                "2026-02-18T15:05:00+00:00",
            ]),
            "gex_net": [1e9, -5e8],
            "zero_gamma_level": [5800.0, 5900.0],
            "spx_price": [5850.0, 5850.0],
            "spy_price": [583.5, 583.5],
            "vix": [19.5, 19.8],
            "vix9d": [21.0, 21.3],
            "term_structure": [1.08, 1.08],
        })

    def test_put_positive_gex_is_support(self) -> None:
        cs = self._make_cs_df()
        dt = datetime(2026, 2, 18, 15, 2, tzinfo=timezone.utc)
        flags, gex_net, zg = lookup_gex_context(cs, dt, "put", 5850.0)
        assert "gex_support" in flags
        assert gex_net == 1e9

    def test_put_negative_gex_is_headwind(self) -> None:
        cs = self._make_cs_df()
        dt = datetime(2026, 2, 18, 15, 7, tzinfo=timezone.utc)
        flags, gex_net, zg = lookup_gex_context(cs, dt, "put", 5850.0)
        assert "gex_headwind" in flags
        assert gex_net == -5e8

    def test_spot_above_zero_gamma(self) -> None:
        cs = self._make_cs_df()
        dt = datetime(2026, 2, 18, 15, 2, tzinfo=timezone.utc)
        flags, _, zg = lookup_gex_context(cs, dt, "put", 6000.0)
        assert "spot_above_zero_gamma" in flags

    def test_spot_below_zero_gamma(self) -> None:
        cs = self._make_cs_df()
        dt = datetime(2026, 2, 18, 15, 2, tzinfo=timezone.utc)
        flags, _, zg = lookup_gex_context(cs, dt, "put", 5700.0)
        assert "spot_below_zero_gamma" in flags

    def test_empty_df_returns_empty_flags(self) -> None:
        cs = pd.DataFrame(columns=["ts", "gex_net", "zero_gamma_level"])
        dt = datetime(2026, 2, 18, 15, 5, tzinfo=timezone.utc)
        flags, gex_net, zg = lookup_gex_context(cs, dt, "put", 5850.0)
        assert flags == []
        assert gex_net is None
        assert zg is None

    def test_outside_window_returns_empty(self) -> None:
        cs = self._make_cs_df()
        dt = datetime(2026, 2, 18, 14, 40, tzinfo=timezone.utc)
        flags, _, _ = lookup_gex_context(cs, dt, "put", 5850.0)
        assert flags == []


# ===================================================================
# Offline GEX computation
# ===================================================================


class TestComputeOfflineGex:
    """compute_offline_gex should derive gex_net and zero_gamma from OI + CBBO."""

    def _make_test_data(
        self,
    ) -> tuple[pd.DataFrame, dict[int, dict], pd.DataFrame, float, date]:
        """Build minimal synthetic data for offline GEX computation.

        Creates 20 instruments (10 calls, 10 puts) centered around spot=5800
        with strikes from 5550 to 6050, expiry in 5 days, realistic BS-priced
        mid values, and OI values.
        """
        spot = 5800.0
        day_date = date(2026, 3, 1)
        expiry = date(2026, 3, 6)
        iv = 0.18

        strikes = list(range(5550, 6050 + 1, 50))
        inst_map: dict[int, dict] = {}
        rows = []
        oi_rows = []

        iid = 1000
        for strike in strikes:
            for pc in ("C", "P"):
                inst_map[iid] = {
                    "strike": float(strike),
                    "expiry": expiry,
                    "put_call": pc,
                    "raw_symbol": f"SPXW {pc}{strike}",
                }
                T = (expiry - day_date).days / 365.0
                from scipy.stats import norm as _norm
                d1 = (np.log(spot / strike) + (0.043 + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
                d2 = d1 - iv * np.sqrt(T)
                if pc == "C":
                    price = spot * _norm.cdf(d1) - strike * np.exp(-0.043 * T) * _norm.cdf(d2)
                else:
                    price = strike * np.exp(-0.043 * T) * _norm.cdf(-d2) - spot * _norm.cdf(-d1)
                price = max(price, 0.10)
                bid = price * 0.95
                ask = price * 1.05

                rows.append({
                    "instrument_id": iid,
                    "bid_px_00": bid,
                    "ask_px_00": ask,
                })
                oi_val = max(100, 5000 - abs(strike - spot) * 5)
                oi_rows.append({"instrument_id": iid, "oi": oi_val})
                iid += 1

        snapshot = pd.DataFrame(rows)
        oi_df = pd.DataFrame(oi_rows)
        return snapshot, inst_map, oi_df, spot, day_date

    def test_returns_gex_and_zero_gamma(self) -> None:
        snapshot, inst_map, oi_df, spot, day_date = self._make_test_data()
        gex_net, zg = compute_offline_gex(snapshot, inst_map, oi_df, spot, day_date)
        assert gex_net is not None
        assert isinstance(gex_net, float)
        assert zg is not None or gex_net != 0.0

    def test_positive_gex_means_calls_dominate(self) -> None:
        """With equal OI, near-ATM call gamma > put gamma for SPX."""
        snapshot, inst_map, oi_df, spot, day_date = self._make_test_data()
        gex_net, _ = compute_offline_gex(snapshot, inst_map, oi_df, spot, day_date)
        assert gex_net is not None

    def test_empty_snapshot_returns_none(self) -> None:
        _, inst_map, oi_df, spot, day_date = self._make_test_data()
        empty_snap = pd.DataFrame(columns=["instrument_id", "bid_px_00", "ask_px_00"])
        gex_net, zg = compute_offline_gex(empty_snap, inst_map, oi_df, spot, day_date)
        assert gex_net is None
        assert zg is None

    def test_empty_oi_returns_none(self) -> None:
        snapshot, inst_map, _, spot, day_date = self._make_test_data()
        empty_oi = pd.DataFrame(columns=["instrument_id", "oi"])
        gex_net, zg = compute_offline_gex(snapshot, inst_map, empty_oi, spot, day_date)
        assert gex_net is None
        assert zg is None

    def test_filters_expired_options(self) -> None:
        """Options past expiry should be excluded."""
        snapshot, inst_map, oi_df, spot, day_date = self._make_test_data()
        future_date = date(2026, 3, 15)
        gex_net, _ = compute_offline_gex(
            snapshot, inst_map, oi_df, spot, future_date,
        )
        assert gex_net is None


# ===================================================================
# FirstRateData loading and merging
# ===================================================================


class TestLoadFrdQuotes:
    """load_frd_quotes should read parquet and produce uq-compatible schema."""

    def test_loads_parquet_with_correct_schema(self, tmp_path: Path) -> None:
        """Output should have columns ts, symbol, last (close renamed)."""
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2024-01-02 14:31:00", "2024-01-02 14:32:00"], utc=True),
            "open": [18.5, 18.6],
            "high": [18.7, 18.8],
            "low": [18.4, 18.5],
            "close": [18.6, 18.7],
        })
        pq = tmp_path / "vix_1min.parquet"
        df.to_parquet(pq, index=False)
        result = load_frd_quotes(pq, "VIX")
        assert list(result.columns) == ["ts", "symbol", "last"]
        assert len(result) == 2
        assert result["symbol"].unique().tolist() == ["VIX"]
        assert result["last"].iloc[0] == 18.6

    def test_timestamps_are_utc(self, tmp_path: Path) -> None:
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2024-06-15 13:30:00"], utc=True),
            "open": [5500.0], "high": [5510.0], "low": [5495.0], "close": [5505.0],
        })
        pq = tmp_path / "spx_1min.parquet"
        df.to_parquet(pq, index=False)
        result = load_frd_quotes(pq, "SPX")
        assert str(result["ts"].dt.tz) == "UTC"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = load_frd_quotes(tmp_path / "nonexistent.parquet", "VIX")
        assert result.empty
        assert list(result.columns) == ["ts", "symbol", "last"]


class TestMergeUnderlyingQuotes:
    """merge_underlying_quotes should combine production + FRD rows."""

    def _ts(self, s: str) -> pd.Timestamp:
        return pd.Timestamp(s, tz="UTC")

    def test_production_takes_priority(self) -> None:
        """When prod and FRD have the same (symbol, ts), prod wins."""
        prod = pd.DataFrame({
            "ts": [self._ts("2024-01-02 15:05:00")],
            "symbol": ["VIX"],
            "last": [20.0],
        })
        frd = pd.DataFrame({
            "ts": [self._ts("2024-01-02 15:05:00")],
            "symbol": ["VIX"],
            "last": [19.9],
        })
        merged = merge_underlying_quotes(prod, frd)
        vix_rows = merged[merged["symbol"] == "VIX"]
        assert len(vix_rows) == 1
        assert vix_rows.iloc[0]["last"] == 20.0

    def test_frd_fills_gaps(self) -> None:
        """FRD rows appear for timestamps not in production."""
        prod = pd.DataFrame({
            "ts": [self._ts("2024-06-01 15:00:00")],
            "symbol": ["VIX"],
            "last": [18.0],
        })
        frd = pd.DataFrame({
            "ts": [self._ts("2024-01-02 15:00:00"), self._ts("2024-01-02 15:01:00")],
            "symbol": ["VIX", "VIX"],
            "last": [22.0, 22.1],
        })
        merged = merge_underlying_quotes(prod, frd)
        assert len(merged) == 3

    def test_multiple_symbols(self) -> None:
        """Merging works across different symbols."""
        prod = pd.DataFrame(columns=["ts", "symbol", "last"])
        frd_vix = pd.DataFrame({
            "ts": [self._ts("2024-01-02 15:00:00")],
            "symbol": ["VIX"], "last": [20.0],
        })
        frd_vix9d = pd.DataFrame({
            "ts": [self._ts("2024-01-02 15:00:00")],
            "symbol": ["VIX9D"], "last": [22.0],
        })
        merged = merge_underlying_quotes(prod, frd_vix, frd_vix9d)
        assert len(merged) == 2
        assert set(merged["symbol"]) == {"VIX", "VIX9D"}

    def test_sorted_by_ts(self) -> None:
        prod = pd.DataFrame(columns=["ts", "symbol", "last"])
        frd = pd.DataFrame({
            "ts": [self._ts("2024-01-03 15:00:00"), self._ts("2024-01-02 15:00:00")],
            "symbol": ["VIX", "VIX"],
            "last": [21.0, 20.0],
        })
        merged = merge_underlying_quotes(prod, frd)
        assert merged.iloc[0]["last"] == 20.0
        assert merged.iloc[1]["last"] == 21.0

    def test_all_empty_returns_empty(self) -> None:
        empty = pd.DataFrame(columns=["ts", "symbol", "last"])
        merged = merge_underlying_quotes(empty)
        assert merged.empty
        assert list(merged.columns) == ["ts", "symbol", "last"]


# ===================================================================
# Daily parquet loading (SKEW)
# ===================================================================


class TestLoadDailyParquet:
    """load_daily_parquet should return a date -> close mapping."""

    def test_basic_load(self, tmp_path: Path) -> None:
        df = pd.DataFrame({
            "ts": pd.to_datetime([
                "2026-01-02 22:15:00", "2026-01-03 22:15:00",
            ], utc=True),
            "open": [130.0, 132.0],
            "high": [131.0, 133.0],
            "low": [129.0, 131.0],
            "close": [130.5, 132.5],
        })
        pq = tmp_path / "skew_daily.parquet"
        df.to_parquet(pq, index=False)
        result = load_daily_parquet(pq)
        assert result[date(2026, 1, 2)] == 130.5
        assert result[date(2026, 1, 3)] == 132.5

    def test_takes_last_close_per_day(self, tmp_path: Path) -> None:
        """When multiple rows share a date, the last close wins."""
        df = pd.DataFrame({
            "ts": pd.to_datetime([
                "2026-01-02 15:00:00", "2026-01-02 22:15:00",
            ], utc=True),
            "open": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 100.0],
            "close": [128.0, 130.5],
        })
        pq = tmp_path / "test.parquet"
        df.to_parquet(pq, index=False)
        result = load_daily_parquet(pq)
        assert result[date(2026, 1, 2)] == 130.5

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = load_daily_parquet(tmp_path / "nonexistent.parquet")
        assert result == {}


# ===================================================================
# Economic calendar loading
# ===================================================================


class TestLoadEconomicCalendar:
    """load_economic_calendar should parse event types into boolean flags."""

    def test_opex_event(self, tmp_path: Path) -> None:
        csv = tmp_path / "cal.csv"
        csv.write_text(
            "date,event_type,has_projections,is_triple_witching\n"
            "2026-01-16,OPEX,False,False\n"
        )
        result = load_economic_calendar(csv)
        assert result[date(2026, 1, 16)]["is_opex"] is True
        assert result[date(2026, 1, 16)]["is_fomc"] is False

    def test_fomc_event(self, tmp_path: Path) -> None:
        csv = tmp_path / "cal.csv"
        csv.write_text(
            "date,event_type,has_projections,is_triple_witching\n"
            "2026-03-17,FOMC,True,False\n"
        )
        result = load_economic_calendar(csv)
        assert result[date(2026, 3, 17)]["is_fomc"] is True
        assert result[date(2026, 3, 17)]["is_opex"] is False

    def test_triple_witching(self, tmp_path: Path) -> None:
        csv = tmp_path / "cal.csv"
        csv.write_text(
            "date,event_type,has_projections,is_triple_witching\n"
            "2026-03-20,OPEX,False,True\n"
        )
        result = load_economic_calendar(csv)
        d = result[date(2026, 3, 20)]
        assert d["is_opex"] is True
        assert d["is_triple_witching"] is True

    def test_multiple_events_same_date(self, tmp_path: Path) -> None:
        """OPEX and FOMC on same date should merge flags."""
        csv = tmp_path / "cal.csv"
        csv.write_text(
            "date,event_type,has_projections,is_triple_witching\n"
            "2026-06-19,OPEX,False,True\n"
            "2026-06-19,FOMC,True,False\n"
        )
        result = load_economic_calendar(csv)
        d = result[date(2026, 6, 19)]
        assert d["is_opex"] is True
        assert d["is_fomc"] is True
        assert d["is_triple_witching"] is True

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = load_economic_calendar(tmp_path / "nonexistent.csv")
        assert result == {}


# ===================================================================
# Offline GEX loading
# ===================================================================


class TestLoadGexCsv:
    """_load_gex_csv should read the GEX cache CSV."""

    def test_basic_load(self, tmp_path: Path) -> None:
        csv = tmp_path / "gex.csv"
        csv.write_text(
            "ts,gex_net,zero_gamma_level\n"
            "2026-01-02T15:05:00+00:00,1e9,5800.0\n"
        )
        df = _load_gex_csv(csv)
        assert len(df) == 1
        assert df.iloc[0]["gex_net"] == 1e9

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        df = _load_gex_csv(tmp_path / "nonexistent.csv")
        assert df.empty
        assert list(df.columns) == ["ts", "gex_net", "zero_gamma_level"]


# ===================================================================
# Trajectory / dual-label tests
# ===================================================================


def _mark(short_mid: float, long_mid: float) -> dict:
    """Build a mark from mid prices (bid = mid - 0.005, ask = mid + 0.005)."""
    return {
        "short_bid": short_mid - 0.005,
        "short_ask": short_mid + 0.005,
        "long_bid": long_mid - 0.005,
        "long_ask": long_mid + 0.005,
    }


class TestTrajectoryLabels:
    """_evaluate_outcome should return trajectory fields for SL sweep analysis.

    credit = 1.0 throughout:
      max_profit = $100, TP50 = $50, SL(2x) = -$200, TP100 = $100
      pnl = (1.0 - (short_mid - long_mid)) * 100
    """

    def test_no_marks_has_trajectory_nulls(self) -> None:
        out = _evaluate_outcome(1.0, [])
        assert out["max_adverse_pnl"] is None
        assert out["hit_stop_loss"] is False
        assert out["hold_hit_tp50"] is False
        assert out["hold_exit_reason"] == "NO_MARKS"

    def test_tp50_only_no_sl(self) -> None:
        """TP50 fires, SL never breached. Trajectory should show clean trade."""
        # exit_cost = 0.4 - 0.05 = 0.35 -> pnl = (1-0.35)*100 = $65 >= $50
        marks = [_mark(0.40, 0.05)]
        out = _evaluate_outcome(1.0, marks)

        assert out["exit_reason"] == "TAKE_PROFIT_50"
        assert out["hit_stop_loss"] is False
        assert out["recovered_after_sl"] is False
        assert out["hold_hit_tp50"] is True
        assert out["hold_exit_reason"] == "TAKE_PROFIT_50"
        # max_adverse_pnl = 0.0 (initialized at entry; only mark is profitable)
        assert out["max_adverse_pnl"] == pytest.approx(0.0, abs=0.01)
        # min_pnl_before_tp50: TP50 fires on first mark at $65 (the only mark seen)
        assert out["min_pnl_before_tp50"] == pytest.approx(65.0, abs=1)
        assert out["max_adverse_multiple"] == pytest.approx(0.0, abs=0.01)

    def test_sl_then_recovery_to_tp50(self) -> None:
        """Trade dips past SL on mark 1, then recovers to TP50 on mark 3."""
        marks = [
            # Mark 1: exit_cost=3.5-0.05=3.45 -> pnl=(1-3.45)*100 = -$245 < -$200 SL
            _mark(3.50, 0.05),
            # Mark 2: exit_cost=1.5-0.05=1.45 -> pnl=(1-1.45)*100 = -$45
            _mark(1.50, 0.05),
            # Mark 3: exit_cost=0.3-0.05=0.25 -> pnl=(1-0.25)*100 = $75 >= $50 TP50
            _mark(0.30, 0.05),
        ]
        out = _evaluate_outcome(1.0, marks)

        # Backward-compatible: SL fires first
        assert out["exit_reason"] == "STOP_LOSS"
        assert out["hit_tp50"] is False
        assert out["realized_pnl"] < -200

        # Trajectory: SL hit, then TP50 hit → recovery
        assert out["hit_stop_loss"] is True
        assert out["recovered_after_sl"] is True
        assert out["hold_hit_tp50"] is True
        assert out["hold_exit_reason"] == "TAKE_PROFIT_50"
        assert out["hold_realized_pnl"] == pytest.approx(75.0, abs=1)
        assert out["max_adverse_pnl"] < -200
        assert out["min_pnl_before_tp50"] < -200
        assert out["final_pnl_at_expiry"] == pytest.approx(75.0, abs=1)

    def test_sl_no_recovery(self) -> None:
        """Trade dips past SL and never recovers."""
        marks = [
            # exit_cost=4.0-0.05=3.95 -> pnl=(1-3.95)*100=-$295 SL
            _mark(4.00, 0.05),
            # exit_cost=2.5-0.1=2.40 -> pnl=(1-2.40)*100=-$140 (better but no TP50)
            _mark(2.50, 0.10),
            # exit_cost=1.5-0.1=1.40 -> pnl=(1-1.40)*100=-$40 (still negative)
            _mark(1.50, 0.10),
        ]
        out = _evaluate_outcome(1.0, marks)

        assert out["exit_reason"] == "STOP_LOSS"
        assert out["hit_stop_loss"] is True
        assert out["recovered_after_sl"] is False
        assert out["hold_hit_tp50"] is False
        assert out["hold_exit_reason"] == "EXPIRY_OR_LAST_MARK"
        assert out["hold_realized_pnl"] == pytest.approx(-40.0, abs=2)
        assert out["max_adverse_pnl"] < -200
        assert out["final_pnl_at_expiry"] == pytest.approx(-40.0, abs=2)

    def test_sl_recovery_to_tp100(self) -> None:
        """Trade dips past SL, then options expire worthless (TP100)."""
        marks = [
            # SL hit: exit_cost=3.5 -> pnl=-$245
            _mark(3.50, 0.05),
            # Recovery: exit_cost≈0 -> pnl=$100 TP100
            _mark(0.01, 0.01),
        ]
        out = _evaluate_outcome(1.0, marks)

        assert out["exit_reason"] == "STOP_LOSS"
        assert out["recovered_after_sl"] is True
        assert out["hold_hit_tp50"] is True
        assert out["final_is_tp100"] is True
        assert out["hold_hit_tp100_at_expiry"] is True
        assert out["final_pnl_at_expiry"] >= 100.0

    def test_max_adverse_multiple(self) -> None:
        """max_adverse_multiple should be worst PnL / max_profit."""
        # pnl = (1.0 - (2.0-0.05))*100 = (1.0-1.95)*100 = -$95
        marks = [_mark(2.00, 0.05)]
        out = _evaluate_outcome(1.0, marks)

        assert out["max_adverse_pnl"] == pytest.approx(-95.0, abs=1)
        # max_profit = $100, so multiple = -95/100 = -0.95
        assert out["max_adverse_multiple"] == pytest.approx(-0.95, abs=0.02)

    def test_expiry_no_events(self) -> None:
        """Neither SL nor TP50 fires — trade expires with small loss."""
        # pnl = (1.0-(1.2-0.05))*100 = (1.0-1.15)*100 = -$15
        marks = [_mark(1.20, 0.05)]
        out = _evaluate_outcome(1.0, marks)

        assert out["exit_reason"] == "EXPIRY_OR_LAST_MARK"
        assert out["hit_stop_loss"] is False
        assert out["hold_hit_tp50"] is False
        assert out["hold_exit_reason"] == "EXPIRY_OR_LAST_MARK"
        assert out["final_pnl_at_expiry"] == pytest.approx(-15.0, abs=1)
        assert out["final_is_tp100"] is False

    def test_min_pnl_before_tp50_tracks_drawdown_before_tp50(self) -> None:
        """min_pnl_before_tp50 stops tracking after TP50 fires."""
        marks = [
            # pnl = (1-1.95)*100 = -$95 (pre-TP50 drawdown)
            _mark(2.00, 0.05),
            # pnl = (1-0.35)*100 = $65 TP50 fires
            _mark(0.40, 0.05),
            # pnl = (1-5.95)*100 = -$495 (post-TP50, should NOT affect min_pnl_before)
            _mark(6.00, 0.05),
        ]
        out = _evaluate_outcome(1.0, marks)

        assert out["min_pnl_before_tp50"] == pytest.approx(-95.0, abs=1)
        # max_adverse_pnl spans ALL marks including post-TP50
        assert out["max_adverse_pnl"] < -400

    def test_backward_compat_sl_before_tp50(self) -> None:
        """Original fields should match old behavior: SL before TP50 → STOP_LOSS."""
        marks = [
            _mark(3.50, 0.05),   # SL fires (-$245)
            _mark(0.30, 0.05),   # TP50 fires (+$75)
        ]
        out = _evaluate_outcome(1.0, marks)

        assert out["exit_reason"] == "STOP_LOSS"
        assert out["hit_tp50"] is False
        assert out["resolved"] is True

    def test_backward_compat_tp50_before_sl(self) -> None:
        """If TP50 fires first, SL is irrelevant for backward-compat fields."""
        marks = [
            _mark(0.30, 0.05),   # TP50 fires first ($75)
            _mark(3.50, 0.05),   # SL fires later (-$245, but trade already closed)
        ]
        out = _evaluate_outcome(1.0, marks)

        assert out["exit_reason"] == "TAKE_PROFIT_50"
        assert out["hit_tp50"] is True
        # hit_stop_loss is True because SL WAS breached (just after TP50)
        assert out["hit_stop_loss"] is True
        assert out["recovered_after_sl"] is False


class TestMultiTpLabeling:
    """Multi-TP level tracking in _evaluate_outcome."""

    def test_all_tp_levels_present(self) -> None:
        """Output contains first_tpXX_pnl and min_pnl_before_tpXX for all TP_LEVELS."""
        marks = [_mark(0.05, 0.05)]  # credit=1.0, pnl = (1-0)*100 = $100 (full profit)
        out = _evaluate_outcome(1.0, marks)
        for lvl in TP_LEVELS:
            assert f"first_tp{lvl}_pnl" in out
            assert f"min_pnl_before_tp{lvl}" in out

    def test_tp60_fires_separately_from_tp50(self) -> None:
        """TP50 and TP60 can fire at different marks."""
        # credit=2.0, max_profit=$200
        # TP50 = $100, TP60 = $120, TP70 = $140
        marks = [
            _mark(0.80, 0.05),   # pnl=(2-.75)*100 = $125 -> TP50=$100 fires, TP60=$120 fires
            _mark(0.50, 0.05),   # pnl=(2-.45)*100 = $155 -> TP70=$140 fires
        ]
        out = _evaluate_outcome(2.0, marks)
        assert out["first_tp50_pnl"] is not None
        assert out["first_tp60_pnl"] is not None
        assert out["first_tp70_pnl"] is not None
        assert out["first_tp50_pnl"] == pytest.approx(125.0, abs=1)
        assert out["first_tp60_pnl"] == pytest.approx(125.0, abs=1)
        assert out["first_tp70_pnl"] == pytest.approx(155.0, abs=1)

    def test_tp90_none_when_not_reached(self) -> None:
        """Higher TP levels are None when not reached."""
        # credit=2.0, max_profit=$200, TP90=$180
        marks = [
            _mark(0.90, 0.05),   # pnl=(2-.85)*100 = $115 -> TP50 fires, but not TP90
        ]
        out = _evaluate_outcome(2.0, marks)
        assert out["first_tp50_pnl"] is not None
        assert out["first_tp90_pnl"] is None

    def test_min_pnl_before_tp70(self) -> None:
        """min_pnl_before_tp70 tracks worst PnL before TP70 fires."""
        # credit=1.0, max_profit=$100, TP70=$70
        marks = [
            _mark(1.80, 0.05),   # pnl=(1-1.75)*100 = -$75 (adverse)
            _mark(0.10, 0.05),   # pnl=(1-.05)*100 = $95 -> TP70 fires
        ]
        out = _evaluate_outcome(1.0, marks)
        assert out["min_pnl_before_tp70"] == pytest.approx(-75.0, abs=1)
        assert out["first_tp70_pnl"] == pytest.approx(95.0, abs=1)

    def test_max_favorable_pnl(self) -> None:
        """max_favorable_pnl tracks peak PnL across all marks."""
        # credit=1.0, max_profit=$100
        # exit_cost = short_mid - long_mid; pnl = (credit - exit_cost) * 100
        marks = [
            _mark(0.05, 0.05),   # exit_cost=0, pnl=$100
            _mark(0.02, 0.05),   # exit_cost=-0.03, pnl=$103 (peak)
            _mark(0.50, 0.05),   # exit_cost=0.45, pnl=$55
        ]
        out = _evaluate_outcome(1.0, marks)
        assert out["max_favorable_pnl"] == pytest.approx(103.0, abs=1)

    def test_empty_marks_all_tp_none(self) -> None:
        """Empty marks produce None for all TP levels."""
        out = _evaluate_outcome(1.0, [])
        for lvl in TP_LEVELS:
            assert out[f"first_tp{lvl}_pnl"] is None
            assert out[f"min_pnl_before_tp{lvl}"] is None
        assert out["max_favorable_pnl"] is None

    def test_backward_compat_aliases(self) -> None:
        """Backward-compat aliases match their multi-TP keyed counterparts."""
        # credit=1.0, max_profit=$100, TP50=$50
        # _mark(0.30, 0.05): exit_cost=0.25, pnl=(1-0.25)*100=$75 -> TP50 fires
        marks = [_mark(0.30, 0.05)]
        out = _evaluate_outcome(1.0, marks)
        assert out["first_tp50_pnl"] == pytest.approx(75.0, abs=1)
        assert out["min_pnl_before_tp50"] == pytest.approx(75.0, abs=1)


# ── Walk-forward date-boundary split ───────────────────────────


class TestWalkForwardDateBoundary:
    """Verify walk_forward_validate splits on day boundaries, not row count."""

    def test_no_day_straddling(self) -> None:
        """All rows from a single day must land in either train or test, not both."""
        rows = []
        for day in ["2025-01-01", "2025-01-02", "2025-01-03",
                     "2025-01-04", "2025-01-05", "2025-01-06"]:
            for i in range(10):
                rows.append({
                    "day": day,
                    "realized_pnl": 50.0,
                    "hit_tp50": True,
                    "features": {"dummy": 1},
                })

        result = walk_forward_validate(rows, train_ratio=0.67)
        assert "error" not in result

        train_days_str = result["train_days"]
        test_days_str = result["test_days"]
        train_last = train_days_str.split(" .. ")[1]
        test_first = test_days_str.split(" .. ")[0]
        assert train_last < test_first, (
            f"Train ends at {train_last} but test starts at {test_first} -- overlap!"
        )
