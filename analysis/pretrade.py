"""Pre-trade checks: Kelvin's rules, checked for one stock before an order is placed, plus position sizing.

The rules come from the review of his IBKR trades and the breakout backtests:

* buy a break of the N-day high (default 20), only when the price is above its 50-day average;
* do not buy back a stock that already lost money today;
* after a losing day, do not carry a new position overnight (overnight holds lost the most);
* no Hong Kong stocks (fees ate the gains);
* each trade is a fixed slot of the account (default 4%) with a stop a fixed % below the entry (default 5%);
* an open position is sold on a close below its 20-day average or when the stop is hit.
"""

from dataclasses import dataclass
import math
import re

import pandas as pd

PASS, FAIL, WARN = "pass", "fail", "warn"


@dataclass
class Check:
    name: str
    status: str
    detail: str


def is_hk(symbol):
    """Hong Kong listings: Yahoo's .HK suffix, IBKR's SEHK tag, or an all-digit code such as 700 or 0700."""

    text = str(symbol or "").strip().upper()
    return text.endswith(".HK") or "SEHK" in text or bool(re.fullmatch(r"\d{1,5}", text))


def price_levels(daily, high_days=20, trend_days=50, exit_days=20):
    """Latest price and the levels the checks use, from daily OHLC bars (the last bar may be today's, still open).

    The N-day high is the highest High of the N bars *before* the last one, so today's own high does not raise
    the bar it has to break.
    """

    bars = daily.dropna(subset=["Close"]) if daily is not None else pd.DataFrame()
    if len(bars) < max(high_days, trend_days, exit_days) + 1:
        return None
    close = bars["Close"].astype(float)
    return {
        "date": bars.index[-1],
        "price": float(close.iloc[-1]),
        "prev_close": float(close.iloc[-2]),
        "high_n": float(bars["High"].iloc[-high_days - 1:-1].max()),
        "sma_trend": float(close.iloc[-trend_days:].mean()),
        "sma_exit": float(close.iloc[-exit_days:].mean()),
        "high_days": high_days, "trend_days": trend_days, "exit_days": exit_days,
    }


def closed_frame(closed):
    frame = pd.DataFrame(closed or [], columns=["Symbol", "Shares", "Entry price", "Entry date", "Exit price",
                                                "Exit date", "P&L", "Note"])
    frame["Symbol"] = frame["Symbol"].astype(str).str.strip().str.upper()
    frame["Exit date"] = pd.to_datetime(frame["Exit date"], errors="coerce").dt.normalize()
    frame["P&L"] = pd.to_numeric(frame["P&L"], errors="coerce").fillna(0.0)
    return frame


def day_pnl(closed, day):
    """Realised P&L of the trades closed on ``day``."""

    frame = closed_frame(closed)
    return float(frame.loc[frame["Exit date"] == pd.Timestamp(day).normalize(), "P&L"].sum())


def size_trade(account, slot_pct, stop_pct, price):
    """Shares for one slot of the account at ``price``, the stop price, and the dollars lost if it is hit."""

    if not price or price <= 0 or not account or account <= 0:
        return {"shares": 0, "value": 0.0, "stop": 0.0, "risk": 0.0}
    shares = int(math.floor(account * slot_pct / 100 / price))
    stop = round(price * (1 - stop_pct / 100), 2)
    return {"shares": shares, "value": shares * price, "stop": stop, "risk": shares * (price - stop)}


def run_checks(symbol, levels, closed, today, entry_price=None, open_positions=0, max_positions=None):
    """Pass, fail or warn for each rule. ``entry_price`` defaults to the latest price."""

    symbol = str(symbol or "").strip().upper()
    checks = []
    if is_hk(symbol):
        checks.append(Check("Not a Hong Kong stock", FAIL, "You stopped trading HK stocks: the fees ate the gains."))
    else:
        checks.append(Check("Not a Hong Kong stock", PASS, "US listing."))

    if levels is None:
        checks.append(Check("Price data", FAIL, "No price history could be loaded, so the chart rules can't be checked."))
    else:
        price = entry_price or levels["price"]
        high, n = levels["high_n"], levels["high_days"]
        if price > high:
            checks.append(Check(f"Breaks its {n}-day high", PASS,
                                f"${price:,.2f} is {100 * (price / high - 1):.1f}% above the {n}-day high of ${high:,.2f}."))
        else:
            checks.append(Check(f"Breaks its {n}-day high", FAIL,
                                f"${price:,.2f} is {100 * (1 - price / high):.1f}% below the {n}-day high of ${high:,.2f}. "
                                f"No breakout yet: a buy stop just above ${high:,.2f} waits for it."))
        sma, t = levels["sma_trend"], levels["trend_days"]
        checks.append(Check(f"Above its {t}-day average", PASS if price > sma else FAIL,
                            f"${price:,.2f} vs the {t}-day average ${sma:,.2f} "
                            f"({100 * (price / sma - 1):+.1f}%)."))

    frame = closed_frame(closed)
    today = pd.Timestamp(today).normalize()
    lost_today = frame[(frame["Symbol"] == symbol) & (frame["Exit date"] == today) & (frame["P&L"] < 0)]
    if len(lost_today):
        checks.append(Check("Not a stock you lost on today", FAIL,
                            f"You lost ${-lost_today['P&L'].sum():,.0f} on {symbol} today. Leave it until tomorrow."))
    else:
        checks.append(Check("Not a stock you lost on today", PASS, f"No losing {symbol} trade logged today."))

    pnl = float(frame.loc[frame["Exit date"] == today, "P&L"].sum())
    if pnl < 0:
        checks.append(Check("No overnight hold after a losing day", WARN,
                            f"Today is down ${-pnl:,.0f} on closed trades. Only a same-day trade: sell it by the close."))
    else:
        checks.append(Check("No overnight hold after a losing day", PASS,
                            f"Closed trades today: ${pnl:+,.0f}. Holding overnight is allowed."))

    if max_positions:
        status = PASS if open_positions < max_positions else FAIL
        checks.append(Check("A free slot", status, f"{open_positions} of {max_positions} positions open."))
    return checks


def verdict(checks):
    """'go' when every check passes, 'caution' when only warnings are left, else 'stop'."""

    if any(check.status == FAIL for check in checks):
        return "stop"
    if any(check.status == WARN for check in checks):
        return "caution"
    return "go"


def position_status(position, levels):
    """Live view of one open position: P&L and whether its exit rules have fired."""

    shares = float(position.get("Shares") or 0)
    entry = float(position.get("Entry price") or 0)
    stop = position.get("Stop")
    stop = float(stop) if stop not in (None, "") and not pd.isna(stop) else None
    row = {"Symbol": str(position.get("Symbol") or "").upper(), "Shares": shares, "Entry price": entry, "Stop": stop,
           "Price": None, "P&L": None, "P&L %": None, "Exit average": None, "Action": "No price data"}
    if levels is None or not entry:
        return row
    price = levels["price"]
    row.update({"Price": price, "P&L": shares * (price - entry), "P&L %": 100 * (price / entry - 1),
                "Exit average": levels["sma_exit"]})
    if stop is not None and price <= stop:
        row["Action"] = f"Sell: at or below the stop ${stop:,.2f}"
    elif price < levels["sma_exit"]:
        row["Action"] = f"Sell: below the {levels['exit_days']}-day average ${levels['sma_exit']:,.2f}"
    else:
        room = 100 * (price / stop - 1) if stop else None
        row["Action"] = "Hold" + (f" ({room:.1f}% above the stop)" if room is not None else "")
    return row
