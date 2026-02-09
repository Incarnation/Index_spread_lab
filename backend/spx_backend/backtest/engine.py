from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd


@dataclass
class BacktestConfig:
    # Data locations (Parquet). These should be pre-downloaded from Databento.
    cbbo_parquet_glob: str
    definitions_parquet_glob: str
    underlying_parquet_glob: str

    # Column mappings for cbbo schema
    cbbo_ts_col: str = "ts_event"
    cbbo_symbol_col: str = "symbol"
    cbbo_bid_col: str = "bid_px"
    cbbo_ask_col: str = "ask_px"

    # Column mappings for definitions schema
    def_symbol_col: str = "symbol"
    def_root_col: str = "root"
    def_expiration_col: str = "expiration"
    def_strike_col: str = "strike_price"
    def_right_col: str = "option_right"  # "C" or "P"

    # Column mappings for underlying series
    underlying_ts_col: str = "ts"
    underlying_price_col: str = "price"

    # Trading parameters
    underlying: str = "SPX"
    entry_times_et: list[str] = field(default_factory=lambda: ["10:00", "11:00", "12:00"])
    dte_targets: list[int] = field(default_factory=lambda: [3, 5, 7])
    dte_tolerance_days: int = 1
    spread_width_points: float = 25.0
    spread_side: str = "put"  # "put" or "call"

    # Risk & management
    take_profit_pct: float = 0.50
    stop_loss_pct: float = 1.00
    max_open_trades: int = 1

    # Fill model
    slippage_fraction: float = 0.30

    # Timezone
    tz: str = "America/New_York"


@dataclass
class TradeLeg:
    symbol: str
    side: str  # "SHORT" or "LONG"
    qty: int
    entry_price: float
    exit_price: float | None = None


@dataclass
class Trade:
    entry_ts: datetime
    exit_ts: datetime | None
    expiration: date
    entry_credit: float
    max_profit: float
    exit_reason: str | None
    legs: list[TradeLeg]


@dataclass
class BacktestResult:
    trades: list[Trade]


class BacktestEngine:
    """
    Minimal backtest engine for SPX credit verticals using Databento OPRA.PILLAR CBBO-1m.

    Notes:
    - This is a baseline implementation intended for correctness and reproducibility.
    - Performance can be improved later (vectorized queries, caching).
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.tz = ZoneInfo(config.tz)
        self.conn = duckdb.connect()

    def _load_views(self) -> None:
        cfg = self.config
        self.conn.execute(f"CREATE OR REPLACE VIEW cbbo AS SELECT * FROM read_parquet('{cfg.cbbo_parquet_glob}')")
        self.conn.execute(f"CREATE OR REPLACE VIEW defs AS SELECT * FROM read_parquet('{cfg.definitions_parquet_glob}')")
        self.conn.execute(f"CREATE OR REPLACE VIEW underlying AS SELECT * FROM read_parquet('{cfg.underlying_parquet_glob}')")

    def _get_expirations(self) -> list[date]:
        cfg = self.config
        q = f"""
        SELECT DISTINCT {cfg.def_expiration_col} AS exp
        FROM defs
        WHERE {cfg.def_root_col} = ?
        ORDER BY exp
        """
        rows = self.conn.execute(q, [cfg.underlying]).fetchall()
        return [r[0] for r in rows if r and r[0] is not None]

    def _choose_expiration(self, expirations: list[date], target_dte: int, now_et: datetime) -> date | None:
        target_date = now_et.date() + timedelta(days=target_dte)
        candidates = [e for e in expirations if abs((e - target_date).days) <= self.config.dte_tolerance_days]
        if not candidates:
            return None
        return min(candidates, key=lambda e: abs((e - target_date).days))

    def _get_leg_mid(self, ts: datetime, symbol: str) -> float | None:
        cfg = self.config
        q = f"""
        SELECT {cfg.cbbo_bid_col} AS bid, {cfg.cbbo_ask_col} AS ask
        FROM cbbo
        WHERE {cfg.cbbo_ts_col} = ? AND {cfg.cbbo_symbol_col} = ?
        LIMIT 1
        """
        row = self.conn.execute(q, [ts, symbol]).fetchone()
        if not row:
            return None
        bid, ask = row
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def _find_spread_legs(self, ts: datetime, expiration: date, spot: float) -> tuple[str, str] | None:
        """
        Select short and long leg symbols based on nearest strikes.
        Uses strike distance rather than delta (delta requires IV computation).
        """
        cfg = self.config
        right = "P" if cfg.spread_side == "put" else "C"

        q = f"""
        SELECT {cfg.def_symbol_col} AS symbol,
               {cfg.def_strike_col} AS strike
        FROM defs
        WHERE {cfg.def_root_col} = ?
          AND {cfg.def_expiration_col} = ?
          AND {cfg.def_right_col} = ?
        """
        rows = self.conn.execute(q, [cfg.underlying, expiration, right]).fetchall()
        if not rows:
            return None

        # Choose the short strike closest to spot on the appropriate side.
        strikes = sorted([(sym, float(strike)) for sym, strike in rows], key=lambda x: x[1])
        if cfg.spread_side == "put":
            candidates = [s for s in strikes if s[1] <= spot]
            short = candidates[-1] if candidates else strikes[0]
            long_strike = short[1] - cfg.spread_width_points
            long = min(strikes, key=lambda s: abs(s[1] - long_strike))
        else:
            candidates = [s for s in strikes if s[1] >= spot]
            short = candidates[0] if candidates else strikes[-1]
            long_strike = short[1] + cfg.spread_width_points
            long = min(strikes, key=lambda s: abs(s[1] - long_strike))

        return short[0], long[0]

    def _entry_credit(self, short_mid: float, long_mid: float) -> float:
        spread_mid = short_mid - long_mid
        slippage = abs(spread_mid) * self.config.slippage_fraction
        return spread_mid - slippage

    def run(self) -> BacktestResult:
        cfg = self.config
        self._load_views()

        expirations = self._get_expirations()
        if not expirations:
            return BacktestResult(trades=[])

        # Load underlying series for timeline.
        uq = f"SELECT {cfg.underlying_ts_col} AS ts, {cfg.underlying_price_col} AS price FROM underlying ORDER BY ts"
        df = self.conn.execute(uq).df()
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(cfg.tz)

        entry_times = {time.fromisoformat(t) for t in cfg.entry_times_et}
        trades: list[Trade] = []
        open_trade: Trade | None = None

        for _, row in df.iterrows():
            ts: datetime = row["ts"]
            spot: float = float(row["price"])

            # Mark-to-market and management for an open trade.
            if open_trade is not None:
                short_leg = next(l for l in open_trade.legs if l.side == "SHORT")
                long_leg = next(l for l in open_trade.legs if l.side == "LONG")
                short_mid = self._get_leg_mid(ts, short_leg.symbol)
                long_mid = self._get_leg_mid(ts, long_leg.symbol)
                if short_mid is not None and long_mid is not None:
                    spread_value = short_mid - long_mid
                    pnl = (open_trade.entry_credit - spread_value) * 100.0
                    if pnl >= open_trade.max_profit * cfg.take_profit_pct:
                        open_trade.exit_ts = ts
                        open_trade.exit_reason = "TAKE_PROFIT_50"
                        short_leg.exit_price = short_mid
                        long_leg.exit_price = long_mid
                        trades.append(open_trade)
                        open_trade = None
                        continue
                    if pnl <= -open_trade.max_profit * cfg.stop_loss_pct:
                        open_trade.exit_ts = ts
                        open_trade.exit_reason = "STOP_LOSS"
                        short_leg.exit_price = short_mid
                        long_leg.exit_price = long_mid
                        trades.append(open_trade)
                        open_trade = None
                        continue

            # Entry at configured times, one open trade max.
            if open_trade is None and ts.timetz().replace(tzinfo=None) in entry_times:
                # Pick first DTE target with a valid expiration.
                chosen_exp: date | None = None
                for dte in cfg.dte_targets:
                    exp = self._choose_expiration(expirations, dte, ts)
                    if exp is not None:
                        chosen_exp = exp
                        break
                if chosen_exp is None:
                    continue

                legs = self._find_spread_legs(ts, chosen_exp, spot)
                if not legs:
                    continue
                short_sym, long_sym = legs

                short_mid = self._get_leg_mid(ts, short_sym)
                long_mid = self._get_leg_mid(ts, long_sym)
                if short_mid is None or long_mid is None:
                    continue

                entry_credit = self._entry_credit(short_mid, long_mid)
                max_profit = max(entry_credit, 0.0) * 100.0
                if max_profit <= 0:
                    continue

                open_trade = Trade(
                    entry_ts=ts,
                    exit_ts=None,
                    expiration=chosen_exp,
                    entry_credit=entry_credit,
                    max_profit=max_profit,
                    exit_reason=None,
                    legs=[
                        TradeLeg(symbol=short_sym, side="SHORT", qty=1, entry_price=short_mid),
                        TradeLeg(symbol=long_sym, side="LONG", qty=1, entry_price=long_mid),
                    ],
                )

        # Close any open trade at end of dataset.
        if open_trade is not None:
            open_trade.exit_ts = df["ts"].iloc[-1] if not df.empty else None
            open_trade.exit_reason = "EOD_CLOSE"
            trades.append(open_trade)

        return BacktestResult(trades=trades)

