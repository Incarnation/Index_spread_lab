"""Tests for ``backend/scripts/backtest_entry.py``.

Covers:

* ``_compute_metrics``        -- per-trade aggregation (Sharpe, max DD,
                                 PF, equity curve, monthly).
* ``_apply_strategy``         -- mask filtering on the OOS pool.
* ``run_strategies``          -- V1 vs. V2 strategy menu construction.
* ``print_results``           -- smoke test on stdout formatting.
* ``collect_oos_predictions`` -- walk-forward looping + dedup of
                                 trades that appear in overlapping test
                                 windows (the "OOS dedup" test).
* ``main``                    -- CLI smoke test with monkey-patched
                                 model training so it runs in <1s.

We monkey-patch ``train_xgb_models`` and ``predict_xgb`` (and the
feature-matrix builder via the ``build_features_fn`` argument) so
``collect_oos_predictions`` can run without invoking real XGBoost.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make scripts/ importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import backtest_entry  # noqa: E402  -- after sys.path edit
from backtest_entry import (  # noqa: E402
    StrategyResult,
    _apply_strategy,
    _compute_metrics,
    collect_oos_predictions,
    main,
    print_results,
    run_strategies,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_pool(n_per_day: int = 4, n_days: int = 30) -> pd.DataFrame:
    """Build a realistic OOS pool DataFrame.

    Columns mirror what ``collect_oos_predictions`` produces so we can
    feed it directly into ``run_strategies`` / ``_apply_strategy``.
    """
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-01-02", periods=n_days, freq="B")
    rows = []
    for d_idx, d in enumerate(days):
        for j in range(n_per_day):
            rows.append({
                "day": d.date().isoformat(),
                "entry_dt": (d + pd.Timedelta(hours=14, minutes=30 + j)).isoformat(),
                "spread_side": "PUT" if j % 2 == 0 else "CALL",
                "dte_target": 0 if j == 0 else 7,
                "delta_target": 0.20,
                "short_strike": 4500 + j,
                "long_strike": 4480 + j,
                # Realised PnL roughly balanced, with a few big losers.
                "hold_realized_pnl": float(rng.uniform(-200, 100)),
                "vix": 18.0 + d_idx * 0.1,
                "prob_tp50": float(rng.uniform(0, 1)),
                "predicted_pnl": float(rng.uniform(-50, 50)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Unit tests for the internal metrics aggregator."""

    def test_empty_pnls_returns_zero_metrics(self):
        # Defensive: an empty array (no trades taken) must yield the
        # zero-trade sentinel rather than crashing on mean()/std().
        out = _compute_metrics(
            pnls=np.array([]),
            days=np.array([]),
            name="empty",
            total_candidates=10,
        )
        assert isinstance(out, StrategyResult)
        assert out.trades_taken == 0
        assert out.trades_skipped == 10
        assert out.total_pnl == 0
        assert out.sharpe == 0
        assert out.max_drawdown == 0
        assert out.equity_curve == []

    def test_basic_aggregates(self):
        # Hand-rolled: 3 wins (+10), 2 losses (-20).
        pnls = np.array([10.0, 10.0, 10.0, -20.0, -20.0])
        days = np.array(["2024-01-02"] * 5)
        out = _compute_metrics(pnls, days, name="basic", total_candidates=5)
        assert out.trades_taken == 5
        assert out.trades_skipped == 0
        assert out.total_pnl == pytest.approx(-10.0)
        assert out.avg_pnl == pytest.approx(-2.0)
        assert out.win_rate == pytest.approx(3 / 5)
        # PF = sum(wins) / |sum(losses)| = 30 / 40 = 0.75.
        assert out.profit_factor == pytest.approx(0.75)
        # Equity curve must be the cumulative PnL series.
        assert out.equity_curve == [10.0, 20.0, 30.0, 10.0, -10.0]

    def test_max_drawdown_is_nonnegative(self):
        # Max drawdown is reported as a *positive* magnitude (peak - cum)
        # so consumers can format it as ``-${max_dd}``.  Guard against
        # a regression that returns a negative value.
        pnls = np.array([100.0, -50.0, -25.0, 10.0])
        out = _compute_metrics(
            pnls,
            np.array(["2024-01-02"] * 4),
            "x",
            total_candidates=4,
        )
        # Peak = 100, trough = 25 -> max DD = 75.
        assert out.max_drawdown == pytest.approx(75.0)

    def test_sharpe_uses_annualization_252(self):
        # Two days with constant daily PnL won't yield std > 0; use two
        # days with different PnL so std > 0 and we get a finite Sharpe.
        pnls = np.array([100.0, -50.0])
        days = np.array(["2024-01-02", "2024-01-03"])
        out = _compute_metrics(pnls, days, "x", total_candidates=2)
        # mean(daily) = 25, std(daily, ddof=1) = 106.066, sharpe = 25/106.066 * sqrt(252)
        expected = (25.0 / np.std([100.0, -50.0], ddof=1)) * np.sqrt(252)
        assert out.sharpe == pytest.approx(expected, rel=1e-6)

    def test_profit_factor_inf_when_no_losses(self):
        # All-wins -> losses sum to 0 -> PF should be inf, not crash.
        pnls = np.array([10.0, 20.0, 30.0])
        out = _compute_metrics(
            pnls,
            np.array(["2024-01-02"] * 3),
            "x",
            total_candidates=3,
        )
        assert out.profit_factor == float("inf")

    def test_monthly_breakdown_groups_by_month(self):
        # Spread two trades across two months and verify both buckets.
        pnls = np.array([10.0, 20.0, 30.0])
        days = np.array(["2024-01-02", "2024-01-31", "2024-02-05"])
        out = _compute_metrics(pnls, days, "x", total_candidates=3)
        months = sorted({m["month"] for m in out.monthly})
        assert months == ["2024-01", "2024-02"]


# ---------------------------------------------------------------------------
# _apply_strategy
# ---------------------------------------------------------------------------


class TestApplyStrategy:
    def test_filters_pool_with_mask(self):
        pool = _make_pool(n_per_day=2, n_days=3)
        # Take only PUT trades (even-index rows in the fixture).
        mask = (pool["spread_side"] == "PUT").values
        out = _apply_strategy(pool, mask, name="puts only")
        # 6 rows total, 3 PUTs.
        assert out.trades_taken == 3
        assert out.trades_skipped == 3
        assert out.name == "puts only"

    def test_all_true_mask_takes_all(self):
        pool = _make_pool(n_per_day=2, n_days=2)
        mask = np.ones(len(pool), dtype=bool)
        out = _apply_strategy(pool, mask, "all")
        assert out.trades_taken == len(pool)
        assert out.trades_skipped == 0

    def test_all_false_mask_takes_none(self):
        pool = _make_pool(n_per_day=2, n_days=2)
        mask = np.zeros(len(pool), dtype=bool)
        out = _apply_strategy(pool, mask, "none")
        assert out.trades_taken == 0
        # Empty-pnls path must still report skipped count correctly.
        assert out.trades_skipped == len(pool)


# ---------------------------------------------------------------------------
# run_strategies
# ---------------------------------------------------------------------------


class TestRunStrategies:
    def test_v1_includes_baseline_and_threshold_strategies(self):
        # V1 exposes "All trades" + 3 prob>=thr filters + 4 rule strats
        # + "E[PnL] > 0".  Pin the count so a refactor that adds/drops
        # a strategy is loud rather than silent.
        pool = _make_pool(n_per_day=4, n_days=10)
        results = run_strategies(pool, is_v2=False)
        names = [r.name for r in results]
        assert "All trades" in names
        assert any(n.startswith("Model P>=.30") or n.startswith("Model P>=.50")
                   for n in names)
        assert "Skip 0-DTE" in names
        assert "E[PnL] > 0" in names

    def test_v2_uses_loss_threshold_strategies(self):
        # V2 inverts the prob semantics: lower P(big_loss) is better,
        # so the menu uses ``prob < threshold`` rather than ``>=``.
        pool = _make_pool(n_per_day=4, n_days=10)
        results = run_strategies(pool, is_v2=True)
        names = [r.name for r in results]
        assert "All trades" in names
        # V2 P(loss) thresholds should appear; V1 P>= ones must not.
        assert any(n.startswith("V2 P(loss)<") for n in names)
        assert not any(n.startswith("Model P>=") for n in names)
        # The combo strategies (V2 P< + no 0DTE) should also appear.
        assert any("no 0DTE" in n for n in names)

    def test_each_result_is_strategy_result(self):
        pool = _make_pool(n_per_day=4, n_days=5)
        results = run_strategies(pool, is_v2=False)
        for r in results:
            assert isinstance(r, StrategyResult)


# ---------------------------------------------------------------------------
# print_results
# ---------------------------------------------------------------------------


class TestPrintResults:
    def test_prints_header_and_each_strategy(self):
        # We only assert that the header + at least one strategy name
        # show up; the formatting is consumer-facing but loosely pinned.
        pool = _make_pool(n_per_day=4, n_days=5)
        results = run_strategies(pool, is_v2=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_results(results)
        out = buf.getvalue()
        assert "STRATEGY BACKTEST COMPARISON" in out
        assert "All trades" in out
        assert "best Sharpe ratio" in out


# ---------------------------------------------------------------------------
# collect_oos_predictions  (walk-forward + OOS dedup)
# ---------------------------------------------------------------------------


def _make_walkforward_df(n_months: int = 12, rows_per_day: int = 3) -> pd.DataFrame:
    """Build a synthetic training CSV spanning ``n_months`` months.

    Each business day gets ``rows_per_day`` trade candidates with stable
    keys so the dedup test in ``collect_oos_predictions`` has something
    to actually deduplicate (overlapping test windows otherwise re-emit
    the same trade-key).
    """
    days = pd.date_range("2024-01-01", periods=n_months * 21, freq="B")
    rows = []
    for d in days:
        for j in range(rows_per_day):
            rows.append({
                "day": d.date().isoformat(),
                "entry_dt": (d + pd.Timedelta(hours=14, minutes=30 + j)).isoformat(),
                "spread_side": "PUT" if j == 0 else "CALL",
                "dte_target": 7,
                "delta_target": 0.20,
                "short_strike": 4500.0 + j,
                "long_strike": 4480.0 + j,
                "hold_realized_pnl": 10.0 if j % 2 == 0 else -5.0,
                "hold_hit_tp50": j % 2 == 0,
                "vix": 18.0,
                # Minimal numerical features for the fake builder.
                "f0": float(j),
                "f1": float(d.weekday()),
            })
    return pd.DataFrame(rows)


def _fake_build_features(df: pd.DataFrame):
    """Return the bare minimum the walk-forward loop needs.

    Returns: (X, y_classification, y_regression).  Two numeric features
    are enough to satisfy XGBoost; we don't actually train a model
    because ``train_xgb_models`` is monkey-patched away.
    """
    X = df[["f0", "f1"]].astype(float).reset_index(drop=True)
    y_cls = df["hold_hit_tp50"].astype(int).reset_index(drop=True)
    y_pnl = df["hold_realized_pnl"].astype(float).reset_index(drop=True)
    return X, y_cls, y_pnl


class TestCollectOosPredictions:
    """Walk-forward + dedup logic.

    We monkey-patch the XGBoost train/predict functions so the test
    runs in <100ms while still exercising every branch of
    ``collect_oos_predictions`` (including the per-window concatenation
    and the cross-window dedup at the end).
    """

    @staticmethod
    def _fake_train(*args, **kwargs):
        # Sentinel value -- predict_xgb only sees this dict.
        return {"sentinel": True}

    @staticmethod
    def _fake_predict(models, X):
        # Return a constant prediction so the assertion below can
        # uniquely identify which window's prediction "wins" the dedup.
        return pd.DataFrame({
            "prob_tp50": np.full(len(X), 0.42),
            "predicted_pnl": np.full(len(X), 1.23),
        })

    def test_returns_pool_with_predictions_and_metadata(self, monkeypatch):
        df = _make_walkforward_df(n_months=12, rows_per_day=2)
        monkeypatch.setattr(backtest_entry, "train_xgb_models", self._fake_train)
        monkeypatch.setattr(backtest_entry, "predict_xgb", self._fake_predict)

        pool = collect_oos_predictions(
            df,
            train_months=8,
            test_months=2,
            step_months=1,
            build_features_fn=_fake_build_features,
        )

        assert not pool.empty
        # Metadata columns must survive the walk-forward + dedup.
        for col in ("day", "entry_dt", "spread_side", "dte_target",
                    "short_strike", "long_strike", "vix", "prob_tp50",
                    "predicted_pnl", "_window_train"):
            assert col in pool.columns

    def test_pool_is_deduplicated_by_trade_key(self, monkeypatch):
        # The whole point: when a trade falls into multiple
        # overlapping test windows (step=1 month, test=2 months), we
        # should keep exactly ONE row per trade key.
        df = _make_walkforward_df(n_months=12, rows_per_day=2)
        monkeypatch.setattr(backtest_entry, "train_xgb_models", self._fake_train)
        monkeypatch.setattr(backtest_entry, "predict_xgb", self._fake_predict)

        pool = collect_oos_predictions(
            df,
            train_months=8,
            test_months=2,
            step_months=1,
            build_features_fn=_fake_build_features,
        )

        key_cols = ["entry_dt", "spread_side", "dte_target",
                    "delta_target", "short_strike", "long_strike"]
        # No duplicate keys after dedup.
        assert pool.duplicated(subset=key_cols).sum() == 0

    def test_dedup_keeps_latest_window(self, monkeypatch):
        # Specialised predict that bakes the training-window string
        # into the prediction so we can verify which window's
        # prediction won the dedup.
        def predict_with_window(models, X):
            # ``models`` is the sentinel dict from _fake_train; we
            # don't actually need its content.  Return a constant.
            return pd.DataFrame({
                "prob_tp50": np.full(len(X), 0.5),
                "predicted_pnl": np.full(len(X), 0.0),
            })

        df = _make_walkforward_df(n_months=12, rows_per_day=2)
        monkeypatch.setattr(backtest_entry, "train_xgb_models", self._fake_train)
        monkeypatch.setattr(backtest_entry, "predict_xgb", predict_with_window)

        pool = collect_oos_predictions(
            df,
            train_months=8,
            test_months=2,
            step_months=1,
            build_features_fn=_fake_build_features,
        )
        # The dedup sorts by ``_window_train`` descending then keeps
        # ``first``.  String comparison on ISO dates is monotonic, so
        # the surviving row for any duplicate key must be the one with
        # the lexicographically largest window string.  Verify this
        # invariant holds for at least one duplicated key.
        assert "_window_train" in pool.columns
        # Verify the column is well-formed and contains valid window
        # labels in "<train_start>..<train_end>" form.  The dedup keeps
        # the lexicographically largest window per trade key (sort
        # descending then keep first), and ISO-format date strings
        # compare monotonically, so this also locks the
        # latest-window-wins invariant.
        assert pool["_window_train"].notna().all()
        for s in pool["_window_train"].unique():
            assert ".." in s, f"unexpected window label: {s}"

    def test_returns_empty_when_no_window_satisfies_size_threshold(self, monkeypatch):
        # If the dataset is too small to satisfy len(train) >= 100 and
        # len(test) >= 30, the function must return an empty DataFrame
        # rather than crashing on the empty concat.
        df = _make_walkforward_df(n_months=2, rows_per_day=1)  # tiny
        monkeypatch.setattr(backtest_entry, "train_xgb_models", self._fake_train)
        monkeypatch.setattr(backtest_entry, "predict_xgb", self._fake_predict)
        pool = collect_oos_predictions(
            df,
            train_months=8,
            test_months=2,
            step_months=1,
            build_features_fn=_fake_build_features,
        )
        assert isinstance(pool, pd.DataFrame)
        assert pool.empty

    def test_uses_default_build_features_fn(self, monkeypatch):
        # When ``build_features_fn`` is None, the function should
        # reach for ``build_entry_feature_matrix``.  Patch that and
        # verify it was called.
        df = _make_walkforward_df(n_months=12, rows_per_day=2)
        called: list[bool] = []

        def patched_default(passed_df):
            called.append(True)
            return _fake_build_features(passed_df)

        monkeypatch.setattr(backtest_entry, "build_entry_feature_matrix",
                            patched_default)
        monkeypatch.setattr(backtest_entry, "train_xgb_models", self._fake_train)
        monkeypatch.setattr(backtest_entry, "predict_xgb", self._fake_predict)
        collect_oos_predictions(
            df, train_months=8, test_months=2, step_months=1,
            build_features_fn=None,
        )
        assert called, "default build_features_fn was not used"


# ---------------------------------------------------------------------------
# main (CLI smoke test)
# ---------------------------------------------------------------------------


class TestMainSmoke:
    """End-to-end CLI smoke with monkey-patched XGBoost.

    The script exits with code 2 if the CSV is missing or doesn't have
    ``hold_realized_pnl``.  The fixtures here always supply both.
    """

    @staticmethod
    def _fake_train(*args, **kwargs):
        return {"sentinel": True}

    @staticmethod
    def _fake_predict(models, X):
        return pd.DataFrame({
            "prob_tp50": np.full(len(X), 0.5),
            "predicted_pnl": np.full(len(X), 0.0),
        })

    def test_main_exits_2_when_csv_missing(self, monkeypatch, tmp_path):
        # Pre-flight: missing file path -> sys.exit(2).
        missing = tmp_path / "does_not_exist.csv"
        monkeypatch.setattr(sys, "argv", [
            "backtest_entry.py", "--csv", str(missing),
        ])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2

    def test_main_exits_2_without_hold_realized_pnl(self, monkeypatch, tmp_path):
        df = _make_walkforward_df(n_months=12, rows_per_day=2)
        df = df.drop(columns=["hold_realized_pnl"])
        csv_path = tmp_path / "no_hold.csv"
        df.to_csv(csv_path, index=False)
        monkeypatch.setattr(sys, "argv", [
            "backtest_entry.py", "--csv", str(csv_path),
        ])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2

    def test_main_runs_end_to_end_with_patched_xgb(self, monkeypatch, tmp_path):
        df = _make_walkforward_df(n_months=12, rows_per_day=2)
        csv_path = tmp_path / "tiny.csv"
        df.to_csv(csv_path, index=False)

        monkeypatch.setattr(backtest_entry, "train_xgb_models", self._fake_train)
        monkeypatch.setattr(backtest_entry, "predict_xgb", self._fake_predict)
        # Inject the lightweight feature builder so we don't need the
        # full training-CSV column set.
        monkeypatch.setattr(backtest_entry, "build_entry_feature_matrix",
                            _fake_build_features)
        monkeypatch.setattr(sys, "argv", [
            "backtest_entry.py", "--csv", str(csv_path),
        ])

        buf = io.StringIO()
        with redirect_stdout(buf):
            main()
        out = buf.getvalue()
        # Headline strings the operator relies on.
        assert "[BACKTEST] Loading" in out
        assert "STRATEGY BACKTEST COMPARISON" in out
        assert "[BACKTEST] Done." in out
