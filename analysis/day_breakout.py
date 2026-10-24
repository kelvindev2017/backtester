"""Day-breakout rule: buy a break of the prior N-day high, decide at each close whether to hold.

This is the rule that came out of reviewing Kelvin's 2026 IBKR trades (fit on Mar-Jul, checked on Aug-Oct):

* long only; enter on the first day the price trades above the highest high of the prior ``lookback`` days
  (default 10), filled at that level, or at the open when the stock gaps above it;
* at every close, including the entry day, keep the position only if the close is in the top part of the
  day's range (default the top third: (close - low) / (high - low) >= 2/3); otherwise sell at the close;
* a fixed dollar amount per trade (default $10,000), so no single trade dominates the result.

Daily bars stand in for the intraday fill: the entry assumes a stop order resting at the breakout level, and
every exit happens at a close, so the rule never needs a price the bar did not print. Fees are a flat amount
per round trip.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


EXIT_MODES = {
    "Hold only on a top-of-range close": "top_range",
    "Always sell at the close": "same_day",
    "Hold while the close is above the 9-day EMA": "above_ema9",
}
TRADE_COLUMNS = ["Ticker", "Entry date", "Entry", "Exit date", "Exit", "Days held", "Shares", "P&L", "Return %",
                 "Status"]


@dataclass
class DayBreakoutSettings:
    lookback: int = 10
    exit_mode: str = "top_range"
    keep_fraction: float = 2 / 3
    max_hold: int = 10
    position_size: float = 10_000.0
    fee_per_trade: float = 1.5
    trend_sma: int = 0


def close_location(daily):
    """Where each close sits in its day's range: 0 at the low, 1 at the high (0.5 when high equals low)."""

    span = daily["High"] - daily["Low"]
    location = (daily["Close"] - daily["Low"]) / span.where(span > 0)
    return location.fillna(0.5)


def keep_overnight(mode, location, close, ema9, keep_fraction):
    if mode == "same_day":
        return False
    if mode == "above_ema9":
        return close > ema9
    return location >= keep_fraction


def day_breakout_trades(daily, settings=None, ticker="", start=None, end=None):
    """Trades of the rule on one ticker's daily OHLC. Only entries from ``start`` to ``end`` are taken.

    A trade still held on the last bar is valued at that close and marked ``Open``.
    """

    settings = settings or DayBreakoutSettings()
    if daily is None or daily.empty or len(daily) <= settings.lookback:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    daily = daily.sort_index()
    opens, highs, closes = (daily[c].to_numpy(dtype=float) for c in ("Open", "High", "Close"))
    level = daily["High"].shift(1).rolling(settings.lookback).max().to_numpy()
    location = close_location(daily).to_numpy()
    ema9 = daily["Close"].ewm(span=9, adjust=False).mean().to_numpy()
    trend = (daily["Close"].rolling(settings.trend_sma).mean().shift(1).to_numpy()
             if settings.trend_sma > 0 else None)
    dates = daily.index
    first = 0 if start is None else int(dates.searchsorted(pd.Timestamp(start)))
    last = len(dates) - 1 if end is None else int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1

    rows = []
    i = max(first, settings.lookback)
    while i <= last:
        if not (np.isfinite(level[i]) and highs[i] > level[i]):
            i += 1
            continue
        if trend is not None and not (np.isfinite(trend[i]) and closes[i - 1] > trend[i]):
            i += 1
            continue
        entry = max(opens[i], level[i])
        shares = math.floor(settings.position_size / entry) if entry > 0 else 0
        if shares < 1:
            i += 1
            continue
        j, status = i, "Closed"
        while True:
            held = j - i + 1
            if settings.max_hold and held >= settings.max_hold:
                break
            if not keep_overnight(settings.exit_mode, location[j], closes[j], ema9[j], settings.keep_fraction):
                break
            if j + 1 >= len(dates):
                status = "Open"
                break
            j += 1
        exit_price = closes[j]
        pnl = shares * (exit_price - entry) - settings.fee_per_trade
        rows.append({
            "Ticker": ticker, "Entry date": dates[i], "Entry": entry, "Exit date": dates[j], "Exit": exit_price,
            "Days held": j - i, "Shares": shares, "P&L": pnl, "Return %": pnl / (shares * entry) * 100,
            "Status": status,
        })
        # A new breakout can be taken from the day after the exit.
        i = j + 1
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def run_day_breakout(prices, settings=None, start=None, end=None):
    """Trades of the rule on every ticker in ``prices`` ({ticker: daily OHLC}), in exit-date order."""

    frames = [day_breakout_trades(frame, settings, ticker, start, end) for ticker, frame in prices.items()]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["Exit date", "Entry date"], ignore_index=True)


def summarize_trades(trades):
    """Headline statistics of closed trades: count, win rate, net, average win and loss, drawdown."""

    closed = trades[trades["Status"] == "Closed"] if len(trades) else trades
    pnl = closed["P&L"].astype(float) if len(closed) else pd.Series(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    equity = pnl.cumsum()
    drawdown = (equity - equity.cummax().clip(lower=0)).min() if len(equity) else 0.0
    gross_loss = -losses.sum()
    return {
        "Trades": int(len(pnl)),
        "Win rate %": float(len(wins) / len(pnl) * 100) if len(pnl) else float("nan"),
        "Net P&L": float(pnl.sum()),
        "Avg / trade": float(pnl.mean()) if len(pnl) else float("nan"),
        "Avg win": float(wins.mean()) if len(wins) else 0.0,
        "Avg loss": float(losses.mean()) if len(losses) else 0.0,
        "Profit factor": float(wins.sum() / gross_loss) if gross_loss > 0 else float("nan"),
        "Max drawdown": float(min(drawdown, 0.0)),
        "Held overnight %": float((closed["Days held"] > 0).mean() * 100) if len(closed) else float("nan"),
    }


def split_summary(trades, split_date):
    """Summary before the split date (where a rule is fitted), from it (the check), and over all trades."""

    split = pd.Timestamp(split_date)
    parts = {
        f"Before {split:%b %d, %Y}": trades[trades["Entry date"] < split],
        f"From {split:%b %d, %Y}": trades[trades["Entry date"] >= split],
        "All": trades,
    }
    return pd.DataFrame({name: summarize_trades(part) for name, part in parts.items()}).T


def monthly_pnl(trades):
    """Net P&L of closed trades by exit month."""

    closed = trades[trades["Status"] == "Closed"]
    if closed.empty:
        return pd.Series(dtype=float)
    return closed.groupby(pd.to_datetime(closed["Exit date"]).dt.to_period("M"))["P&L"].sum()


def ticker_summary(trades):
    """Trades, win rate and net P&L per ticker, best first."""

    closed = trades[trades["Status"] == "Closed"]
    if closed.empty:
        return pd.DataFrame(columns=["Trades", "Win rate %", "Net P&L"])
    grouped = closed.groupby("Ticker")["P&L"]
    table = pd.DataFrame({
        "Trades": grouped.size(),
        "Win rate %": grouped.apply(lambda s: (s > 0).mean() * 100),
        "Net P&L": grouped.sum(),
    })
    return table.sort_values("Net P&L", ascending=False)
