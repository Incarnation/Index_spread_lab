from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import statistics
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd


@dataclass
class BacktestConfig:
    """Configuration for the DuckDB-based backtest engine."""
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
    commission_per_contract: float = 0.65
    exchange_fee_per_contract: float = 0.00
    contract_multiplier: int = 100
    use_asof_quotes: bool = True
    settlement_time_et: str = "16:00"

    # Optional date filters (walk-forward / out-of-sample splits)
    start_date: str | None = None
    end_date: str | None = None

    # Timezone
    tz: str = "America/New_York"


@dataclass
class TradeLeg:
    """Single trade leg representation for backtest results."""
    symbol: str
    side: str  # "SHORT" or "LONG"
    qty: int
    entry_price: float
    strike: float | None = None
    right: str | None = None
    exit_price: float | None = None


@dataclass
class Trade:
    """Trade-level result for backtest output."""
    entry_ts: datetime
    exit_ts: datetime | None
    expiration: date
    entry_credit: float
    max_profit: float
    legs: list[TradeLeg]
    exit_reason: str | None = None
    entry_fees: float = 0.0
    exit_fees: float = 0.0
    exit_cost: float | None = None
    realized_pnl: float | None = None


@dataclass
class BacktestResult:
    """Container for backtest results."""
    trades: list[Trade]
    summary: dict[str, float | int | None]


class BacktestEngine:
    """
    Minimal backtest engine for SPX credit verticals using Databento OPRA.PILLAR CBBO-1m.

    Notes:
    - This is a baseline implementation intended for correctness and reproducibility.
    - Performance can be improved later (vectorized queries, caching).
    """

    def __init__(self, config: BacktestConfig) -> None:
        """Initialize engine and DuckDB connection."""
        self.config = config
        self.tz = ZoneInfo(config.tz)
        self.conn = duckdb.connect()

    def _load_views(self) -> None:
        """Load Parquet datasets into DuckDB views."""
        cfg = self.config
        self.conn.execute(f"CREATE OR REPLACE VIEW cbbo AS SELECT * FROM read_parquet('{cfg.cbbo_parquet_glob}')")
        self.conn.execute(f"CREATE OR REPLACE VIEW defs AS SELECT * FROM read_parquet('{cfg.definitions_parquet_glob}')")
        self.conn.execute(f"CREATE OR REPLACE VIEW underlying AS SELECT * FROM read_parquet('{cfg.underlying_parquet_glob}')")

    def _get_expirations(self) -> list[date]:
        """Get distinct expirations from the definitions file."""
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
        """Choose an expiration within the DTE tolerance window."""
        target_date = now_et.date() + timedelta(days=target_dte)
        candidates = [e for e in expirations if abs((e - target_date).days) <= self.config.dte_tolerance_days]
        if not candidates:
            return None
        return min(candidates, key=lambda e: abs((e - target_date).days))

    def _get_leg_quote(self, ts: datetime, symbol: str) -> tuple[float, float] | None:
        """Fetch bid/ask at or before ts (as-of) to avoid leakage."""
        cfg = self.config
        ts_query = ts.astimezone(timezone.utc)
        op = "<=" if cfg.use_asof_quotes else "="
        q = f"""
        SELECT {cfg.cbbo_bid_col} AS bid, {cfg.cbbo_ask_col} AS ask
        FROM cbbo
        WHERE {cfg.cbbo_symbol_col} = ?
          AND {cfg.cbbo_ts_col} {op} ?
        ORDER BY {cfg.cbbo_ts_col} DESC
        LIMIT 1
        """
        row = self.conn.execute(q, [symbol, ts_query]).fetchone()
        if not row:
            return None
        bid, ask = row
        if bid is None or ask is None:
            return None
        return float(bid), float(ask)

    def _get_leg_mid(self, ts: datetime, symbol: str) -> float | None:
        """Compute mid price from the latest available bid/ask."""
        quote = self._get_leg_quote(ts, symbol)
        if not quote:
            return None
        bid, ask = quote
        return (bid + ask) / 2.0

    def _find_spread_legs(self, ts: datetime, expiration: date, spot: float) -> tuple[tuple[str, float], tuple[str, float]] | None:
        """Select short and long leg symbols + strikes based on distance."""
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

        return (short[0], short[1]), (long[0], long[1])

    def _spread_stats(self, ts: datetime, short_symbol: str, long_symbol: str) -> tuple[float, float, float, float] | None:
        """Compute spread mid and width using as-of leg quotes."""
        short_quote = self._get_leg_quote(ts, short_symbol)
        long_quote = self._get_leg_quote(ts, long_symbol)
        if not short_quote or not long_quote:
            return None
        short_bid, short_ask = short_quote
        long_bid, long_ask = long_quote
        if short_bid is None or short_ask is None or long_bid is None or long_ask is None:
            return None
        short_mid = (short_bid + short_ask) / 2.0
        long_mid = (long_bid + long_ask) / 2.0
        spread_mid = short_mid - long_mid
        spread_width = max((short_ask - short_bid) + (long_ask - long_bid), 0.0)
        return short_mid, long_mid, spread_mid, spread_width

    def _entry_credit(self, spread_mid: float, spread_width: float) -> float:
        """Apply spread-width slippage to compute entry credit."""
        slippage = spread_width * self.config.slippage_fraction
        return spread_mid - slippage

    def _exit_cost(self, spread_mid: float, spread_width: float) -> float:
        """Apply spread-width slippage to compute exit cost."""
        slippage = spread_width * self.config.slippage_fraction
        return spread_mid + slippage

    def _calc_fees(self, legs: list[TradeLeg]) -> float:
        """Calculate per-contract fees for the spread."""
        per_contract = self.config.commission_per_contract + self.config.exchange_fee_per_contract
        contracts = sum(abs(l.qty) for l in legs)
        return contracts * per_contract

    def _build_settlement_prices(self, df: pd.DataFrame) -> dict[date, float]:
        """Precompute settlement prices at or before the configured settlement time."""
        settlement_time = time.fromisoformat(self.config.settlement_time_et)
        df = df.copy()
        df["day"] = df["ts"].dt.date
        df["tod"] = df["ts"].dt.time
        eligible = df[df["tod"] <= settlement_time]
        if eligible.empty:
            return {}
        last_rows = eligible.groupby("day").tail(1)
        return {row["day"]: float(row["price"]) for _, row in last_rows.iterrows()}

    def _settlement_price_for_date(self, settlement_prices: dict[date, float], day: date) -> float | None:
        """Return settlement price for a date, falling back to prior available."""
        if day in settlement_prices:
            return settlement_prices[day]
        prior_days = sorted(d for d in settlement_prices.keys() if d <= day)
        if not prior_days:
            return None
        return settlement_prices[prior_days[-1]]

    def _intrinsic_value(self, right: str | None, strike: float | None, spot: float) -> float:
        """Compute intrinsic value for a call/put at spot."""
        if right is None or strike is None:
            return 0.0
        if right.upper() == "C":
            return max(0.0, spot - strike)
        if right.upper() == "P":
            return max(0.0, strike - spot)
        return 0.0

    def _settlement_value(self, trade: Trade, settlement_price: float) -> float:
        """Compute cash-settled value of the spread at expiration."""
        total = 0.0
        for leg in trade.legs:
            intrinsic = self._intrinsic_value(leg.right, leg.strike, settlement_price)
            leg_value = intrinsic * leg.qty * self.config.contract_multiplier
            if leg.side == "SHORT":
                leg_value *= -1.0
            total += leg_value
        return total

    def _trade_exit_cost(self, trade: Trade) -> float | None:
        """Compute exit cost per contract from leg exit prices."""
        if len(trade.legs) < 2:
            return None
        short_leg = next((l for l in trade.legs if l.side == "SHORT"), None)
        long_leg = next((l for l in trade.legs if l.side == "LONG"), None)
        if not short_leg or not long_leg:
            return None
        if short_leg.exit_price is None or long_leg.exit_price is None:
            return None
        return float(short_leg.exit_price) - float(long_leg.exit_price)

    def _build_summary(self, trades: list[Trade]) -> dict[str, float | int | None]:
        """Compute a simple performance summary from realized PnL."""
        pnls = [t.realized_pnl for t in trades if t.realized_pnl is not None]
        trades_total = len(trades)
        trades_closed = len(pnls)
        wins = len([p for p in pnls if p > 0])
        losses = len([p for p in pnls if p < 0])
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = sum(p for p in pnls if p < 0)
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / trades_closed if trades_closed else 0.0
        median_pnl = statistics.median(pnls) if pnls else 0.0
        win_rate = wins / trades_closed if trades_closed else 0.0
        avg_win = gross_profit / wins if wins else 0.0
        avg_loss = gross_loss / losses if losses else 0.0
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None

        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for trade in sorted([t for t in trades if t.realized_pnl is not None], key=lambda t: t.exit_ts or t.entry_ts):
            equity += float(trade.realized_pnl or 0.0)
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)

        return {
            "trades_total": trades_total,
            "trades_closed": trades_closed,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "median_pnl": median_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
        }

    def run(self) -> BacktestResult:
        """Run the backtest and return trade results."""
        cfg = self.config
        self._load_views()

        expirations = self._get_expirations()
        if not expirations:
            return BacktestResult(trades=[])

        # Load underlying series for timeline.
        uq = f"SELECT {cfg.underlying_ts_col} AS ts, {cfg.underlying_price_col} AS price FROM underlying ORDER BY ts"
        df = self.conn.execute(uq).df()
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(cfg.tz)
        # Apply optional date filters for walk-forward splits.
        if cfg.start_date:
            start_day = date.fromisoformat(cfg.start_date)
            df = df[df["ts"].dt.date >= start_day]
        if cfg.end_date:
            end_day = date.fromisoformat(cfg.end_date)
            df = df[df["ts"].dt.date <= end_day]
        df = df.reset_index(drop=True)

        settlement_prices = self._build_settlement_prices(df)

        entry_times = {time.fromisoformat(t) for t in cfg.entry_times_et}
        trades: list[Trade] = []
        open_trade: Trade | None = None

        for _, row in df.iterrows():
            ts: datetime = row["ts"]
            spot: float = float(row["price"])

            # Force cash-settled expiration handling once we pass expiration date.
            if open_trade is not None and ts.date() > open_trade.expiration:
                settlement_price = self._settlement_price_for_date(settlement_prices, open_trade.expiration)
                if settlement_price is None:
                    settlement_price = spot
                open_trade.exit_ts = datetime.combine(
                    open_trade.expiration,
                    time.fromisoformat(cfg.settlement_time_et),
                    tzinfo=self.tz,
                )
                open_trade.exit_reason = "EXPIRED"
                open_trade.exit_fees = self._calc_fees(open_trade.legs)
                # Settlement value is a cash payout for each leg; store intrinsic as exit price.
                for leg in open_trade.legs:
                    leg.exit_price = self._intrinsic_value(leg.right, leg.strike, settlement_price)
                exit_cost = self._trade_exit_cost(open_trade)
                if exit_cost is not None:
                    open_trade.exit_cost = exit_cost
                    open_trade.realized_pnl = (open_trade.entry_credit - exit_cost) * cfg.contract_multiplier - (
                        open_trade.entry_fees + open_trade.exit_fees
                    )
                trades.append(open_trade)
                open_trade = None
                continue

            # Mark-to-market and management for an open trade.
            if open_trade is not None:
                short_leg = next(l for l in open_trade.legs if l.side == "SHORT")
                long_leg = next(l for l in open_trade.legs if l.side == "LONG")
                spread_stats = self._spread_stats(ts, short_leg.symbol, long_leg.symbol)
                if spread_stats is not None:
                    short_mid, long_mid, spread_mid, spread_width = spread_stats
                    exit_cost = self._exit_cost(spread_mid, spread_width)
                    exit_fees = self._calc_fees(open_trade.legs)
                    pnl = (open_trade.entry_credit - exit_cost) * cfg.contract_multiplier - (
                        open_trade.entry_fees + exit_fees
                    )
                    if pnl >= open_trade.max_profit * cfg.take_profit_pct:
                        open_trade.exit_ts = ts
                        open_trade.exit_reason = "TAKE_PROFIT_50"
                        open_trade.exit_fees = exit_fees
                        open_trade.exit_cost = exit_cost
                        open_trade.realized_pnl = pnl
                        short_leg.exit_price = short_mid
                        long_leg.exit_price = long_mid
                        trades.append(open_trade)
                        open_trade = None
                        continue
                    if pnl <= -open_trade.max_profit * cfg.stop_loss_pct:
                        open_trade.exit_ts = ts
                        open_trade.exit_reason = "STOP_LOSS"
                        open_trade.exit_fees = exit_fees
                        open_trade.exit_cost = exit_cost
                        open_trade.realized_pnl = pnl
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
                (short_sym, short_strike), (long_sym, long_strike) = legs

                spread_stats = self._spread_stats(ts, short_sym, long_sym)
                if spread_stats is None:
                    continue
                short_mid, long_mid, spread_mid, spread_width = spread_stats

                entry_credit = self._entry_credit(spread_mid, spread_width)
                if entry_credit <= 0:
                    continue
                entry_fees = self._calc_fees(
                    [
                        TradeLeg(symbol=short_sym, side="SHORT", qty=1, entry_price=short_mid),
                        TradeLeg(symbol=long_sym, side="LONG", qty=1, entry_price=long_mid),
                    ]
                )
                exit_fees_est = entry_fees
                max_profit = max(entry_credit, 0.0) * cfg.contract_multiplier - entry_fees - exit_fees_est
                if max_profit <= 0:
                    continue

                right = "P" if cfg.spread_side == "put" else "C"
                open_trade = Trade(
                    entry_ts=ts,
                    exit_ts=None,
                    expiration=chosen_exp,
                    entry_credit=entry_credit,
                    max_profit=max_profit,
                    entry_fees=entry_fees,
                    exit_fees=0.0,
                    exit_reason=None,
                    legs=[
                        TradeLeg(
                            symbol=short_sym,
                            side="SHORT",
                            qty=1,
                            entry_price=short_mid,
                            strike=short_strike,
                            right=right,
                        ),
                        TradeLeg(
                            symbol=long_sym,
                            side="LONG",
                            qty=1,
                            entry_price=long_mid,
                            strike=long_strike,
                            right=right,
                        ),
                    ],
                )

        # Close any open trade at end of dataset.
        if open_trade is not None:
            last_ts = df["ts"].iloc[-1] if not df.empty else None
            if last_ts is not None:
                spread_stats = self._spread_stats(last_ts, open_trade.legs[0].symbol, open_trade.legs[1].symbol)
                if spread_stats is not None:
                    short_mid, long_mid, spread_mid, spread_width = spread_stats
                    open_trade.exit_ts = last_ts
                    open_trade.exit_reason = "EOD_CLOSE"
                    open_trade.exit_fees = self._calc_fees(open_trade.legs)
                    open_trade.exit_cost = self._exit_cost(spread_mid, spread_width)
                    if open_trade.exit_cost is not None:
                        open_trade.realized_pnl = (open_trade.entry_credit - open_trade.exit_cost) * cfg.contract_multiplier - (
                            open_trade.entry_fees + open_trade.exit_fees
                        )
                    open_trade.legs[0].exit_price = short_mid
                    open_trade.legs[1].exit_price = long_mid
                    trades.append(open_trade)
                else:
                    open_trade.exit_ts = last_ts
                    open_trade.exit_reason = "EOD_NO_QUOTE"
                    trades.append(open_trade)

        summary = self._build_summary(trades)
        return BacktestResult(trades=trades, summary=summary)

