"""First-bar day trade: the first bar of the session (1, 5 or 15 minutes) decides whether the day is a long day.

Kelvin's rule (2026-10-29):

* the user picks a stock and allocates capital to the rule;
* if the first 5-minute bar closes at least ``min_move`` points (dollars) above its open, the day is a long day;
  otherwise it is a short day and the rule stays out, unless ``allow_short`` is on: then a first bar that closes
  at least ``min_move`` points below its open is shorted, the mirror image of the long trade (sell at its close,
  stop at its high, the pullback exit on a close above the latest bounce, the lower-high exit on a higher low);
* on a long day, buy at the first bar's close with the stop at the first bar's low; size the position so that a
  stop-out loses at most ``risk_pct`` of the allocated capital (and never more shares than the capital buys);
* ignore the next ``skip_bars`` bars for the exit (the stop still applies), then sell everything on one of two
  exits: the pullback exit (every close below the previous close is a pullback; the first later close below the
  latest pullback sells), or the lower-high exit (the first bar whose high is below the previous bar's high sells
  at that bar's close), or a custom exit that compares two values the user picks (e.g. the close below the
  previous bar's low) and sells at the close of the first bar where the comparison holds;
* anything still held is sold 5 minutes before the close (at the open of the first bar that starts 5 minutes or
  less before the close; with 15-minute bars that is the last bar, 15 minutes before).

Costs follow IBKR Pro for US stocks with market orders (see ``order_costs``), plus a slippage of a few cents per
share on every fill.

Sizing is a separate function (``position_shares``) and the backtest takes a list of tickers, so a portfolio risk
system or the stock screener can feed them later.
"""

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd


EXIT_MODES = {
    "Pullback: next close below the last pullback close": "pullback",
    "Lower high: a bar's high below the previous bar's high": "lower_high",
    "Custom: compare two values": "custom",
}
# Values a custom exit can compare, each a price on every bar. "Since entry" means the bars after the first bar,
# before the bar being checked (starting at the entry price).
EXIT_VALUES = {
    "Close": "close",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Previous close": "prev_close",
    "Previous open": "prev_open",
    "Previous high": "prev_high",
    "Previous low": "prev_low",
    "Highest high since entry": "max_high",
    "Lowest low since entry": "min_low",
    "Entry price": "entry",
    "First bar high": "first_high",
    "First bar low": "first_low",
    "VWAP (this session)": "vwap",
    "EMA of closes (this session)": "ema",
}
EXIT_OPERATORS = ["<", "<=", ">", ">=", "crosses below", "crosses above"]
EXIT_FILLS = {
    "At the close that broke the pullback (market order)": "close",
    "At the pullback price (as written)": "pullback",
}
TRADE_COLUMNS = ["Ticker", "Date", "Side", "Entry time", "Entry", "Stop", "Exit time", "Exit", "Exit reason", "Shares",
                 "Risk $", "Costs $", "P&L", "Return %", "R"]
COMMISSION_PLANS = {
    "IBKR Pro Tiered": "tiered",
    "IBKR Pro Fixed": "fixed",
    "No commission (IBKR Lite)": "none",
}
# IBKR Pro, US stocks (published rates, 2026). Tiered is the lowest-volume tier (up to 300,000 shares a month);
# market orders take liquidity, so they pay the exchange's remove-liquidity fee (about $0.003 a share).
TIERED_PER_SHARE, TIERED_MIN, MAX_PCT_OF_VALUE = 0.0035, 0.35, 0.01
FIXED_PER_SHARE, FIXED_MIN = 0.005, 1.00
EXCHANGE_PER_SHARE, CLEARING_PER_SHARE = 0.003, 0.0002
PASS_THROUGH_OF_COMMISSION = 0.000175 + 0.000565  # NYSE and FINRA pass-through, as a share of the commission
SEC_FEE_OF_SALES, FINRA_TAF_PER_SHARE, FINRA_TAF_MAX, CAT_PER_SHARE = 0.0000206, 0.000195, 9.79, 0.000003
DAY_COLUMNS = ["Ticker", "Date", "First bar open", "First bar close", "Move", "Long day", "Short day"]


@dataclass
class CustomExit:
    """Sell at the close of the first bar where ``left operator right + offset`` holds."""

    left: str = "close"
    operator: str = "<"
    right: str = "prev_low"
    offset: float = 0.0
    ema_period: int = 9

    def describe(self):
        names = {key: label for label, key in EXIT_VALUES.items()}

        def name(key):
            return f"EMA {self.ema_period} of closes" if key == "ema" else names[key]

        offset = f" {'+' if self.offset >= 0 else '-'} {abs(self.offset):g}" if self.offset else ""
        return f"{name(self.left)} {self.operator} {name(self.right)}{offset}"


@dataclass
class OpeningBarSettings:
    min_move: float = 0.0
    skip_bars: int = 2
    exit_mode: str = "pullback"
    risk_pct: float = 10.0
    capital: float = 10_000.0
    cap_at_capital: bool = True
    exit_fill: str = "close"
    commission_plan: str = "tiered"
    slippage: float = 0.01
    allow_short: bool = False
    bar_minutes: int = 5
    custom_exit: CustomExit = field(default_factory=CustomExit)


def order_costs(shares, price, sell, plan="tiered"):
    """IBKR Pro cost of one market order for US stock: commission, exchange, clearing and regulatory fees.

    Tiered: $0.0035 a share (min $0.35, max 1% of the value) plus exchange, clearing and pass-through fees. Fixed:
    $0.005 a share (min $1, max 1%), which includes exchange and clearing fees. Sales also pay the SEC fee and the
    FINRA trading activity fee, on either plan. A short sale pays them on the opening sell.
    """

    if shares <= 0:
        return 0.0
    value = shares * price
    if plan == "fixed":
        cost = min(max(FIXED_PER_SHARE * shares, FIXED_MIN), MAX_PCT_OF_VALUE * value)
    elif plan == "tiered":
        commission = min(max(TIERED_PER_SHARE * shares, TIERED_MIN), MAX_PCT_OF_VALUE * value)
        cost = (commission * (1 + PASS_THROUGH_OF_COMMISSION) + (EXCHANGE_PER_SHARE + CLEARING_PER_SHARE) * shares)
    else:
        cost = 0.0
    if sell:
        cost += SEC_FEE_OF_SALES * value + min(FINRA_TAF_PER_SHARE * shares, FINRA_TAF_MAX) + CAT_PER_SHARE * shares
    return cost


def is_long_day(first_bar, min_move=0.0):
    """Long day when the first bar closes above its open by at least ``min_move`` points."""

    move = first_bar["Close"] - first_bar["Open"]
    # The tolerance keeps a move of exactly N points (100.3 - 100 is 0.2999... in floating point) a long day.
    return bool(move > 0 and move >= min_move - 1e-9)


def is_short_day(first_bar, min_move=0.0):
    """The mirror image: the first bar closes below its open by at least ``min_move`` points."""

    return is_long_day({"Open": -first_bar["Open"], "Close": -first_bar["Close"]}, min_move)


def mirrored(session):
    """Prices turned upside down (high becomes low), so a short is traded by the long rules."""

    frame = pd.DataFrame({"Open": -session["Open"], "High": -session["Low"], "Low": -session["High"],
                          "Close": -session["Close"]}, index=session.index)
    if "Volume" in session:
        frame["Volume"] = session["Volume"]
    return frame


def exit_values(session, key, entry, ema_period=9):
    """One of the ``EXIT_VALUES`` on every bar of the session (the first bar is the entry bar)."""

    opens, highs, lows, closes = (session[c].to_numpy(dtype=float) for c in ("Open", "High", "Low", "Close"))
    current = {"close": closes, "open": opens, "high": highs, "low": lows}
    if key in current:
        return current[key]
    if key.startswith("prev_"):
        values = current[key[5:]]
        return np.concatenate([[np.nan], values[:-1]])
    if key in ("max_high", "min_low"):
        # Bars after the entry bar, up to the one before the bar being checked, starting at the entry price.
        source = highs if key == "max_high" else lows
        running = (np.maximum if key == "max_high" else np.minimum).accumulate(np.concatenate([[entry], source[1:]]))
        return np.concatenate([[entry], running[:-1]])
    if key == "entry":
        return np.full(len(closes), float(entry))
    if key in ("first_high", "first_low"):
        return np.full(len(closes), highs[0] if key == "first_high" else lows[0])
    if key == "vwap":
        typical = (highs + lows + closes) / 3
        volume = session["Volume"].to_numpy(dtype=float) if "Volume" in session else np.ones(len(closes))
        volume = np.where(np.isfinite(volume) & (volume > 0), volume, 0.0)
        weight = np.cumsum(volume)
        with np.errstate(invalid="ignore", divide="ignore"):
            vwap = np.cumsum(typical * volume) / weight
        return np.where(weight > 0, vwap, np.cumsum(typical) / np.arange(1, len(closes) + 1))
    if key == "ema":
        return pd.Series(closes).ewm(span=max(int(ema_period), 1), adjust=False).mean().to_numpy()
    raise ValueError(f"Unknown exit value: {key}")


def custom_exit_bars(session, entry, custom):
    """True on every bar where the custom exit's comparison holds."""

    left = exit_values(session, custom.left, entry, custom.ema_period)
    right = exit_values(session, custom.right, entry, custom.ema_period) + custom.offset
    previous_left, previous_right = np.roll(left, 1), np.roll(right, 1)
    with np.errstate(invalid="ignore"):
        hit = {
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
            "crosses below": (previous_left >= previous_right) & (left < right),
            "crosses above": (previous_left <= previous_right) & (left > right),
        }[custom.operator]
    hit[0] = False
    return hit


def position_shares(capital, entry, stop, risk_pct=10.0, cap_at_capital=True):
    """Shares (long or short) such that a stop-out loses at most ``risk_pct`` % of ``capital`` (0 when there is no risk to size on)."""

    risk_per_share = abs(entry - stop)
    if not (np.isfinite(risk_per_share) and risk_per_share > 0 and entry > 0):
        return 0
    shares = math.floor(capital * risk_pct / 100 / risk_per_share)
    if cap_at_capital:
        shares = min(shares, math.floor(capital / entry))
    return max(shares, 0)


def exit_session(session, entry, stop, skip_bars=2, exit_fill="close", exit_mode="pullback", flat_bar=None,
                 custom=None):
    """Exit of a long entered at the close of the session's first bar: (bar position, price, reason).

    The stop is checked first on every bar (a gap below it fills at the open). The pullback, lower-high or custom
    exit starts after the skipped bars; ``flat_bar`` (default the last bar) is the flat-before-the-close exit, at its
    open. A custom exit sells at the close of the bar where its comparison holds.
    """

    opens, highs, lows, closes = (session[c].to_numpy(dtype=float) for c in ("Open", "High", "Low", "Close"))
    last = len(session) - 1 if flat_bar is None else flat_bar
    pullback = np.nan
    custom_hits = custom_exit_bars(session, entry, custom or CustomExit()) if exit_mode == "custom" else None
    for k in range(1, last + 1):
        if lows[k] <= stop:
            return k, min(opens[k], stop), "Stop"
        if k == last:
            return k, opens[k], "End of day"
        if k <= skip_bars:
            continue
        if exit_mode == "custom":
            if custom_hits[k]:
                return k, closes[k], "Custom"
            continue
        if exit_mode == "lower_high":
            if highs[k] < highs[k - 1]:
                return k, closes[k], "Lower high"
            continue
        if np.isfinite(pullback) and closes[k] < pullback:
            return k, (pullback if exit_fill == "pullback" else closes[k]), "Pullback"
        if closes[k] < closes[k - 1]:
            pullback = closes[k]
    return last, closes[last], "End of day"


def flat_bar_position(index, bar_minutes=5, minutes_before_close=5):
    """Position of the bar whose open is the flat-before-the-close sale: the first bar starting at most
    ``minutes_before_close`` before the session's end, else the last bar."""

    close_time = index[-1] + pd.Timedelta(minutes=bar_minutes)
    late = np.flatnonzero(index >= close_time - pd.Timedelta(minutes=minutes_before_close))
    return int(late[0]) if len(late) else len(index) - 1


def opening_bar_trades(bars, settings=None, ticker=""):
    """Trades of the rule on one ticker's regular-session intraday bars, plus the long/short call of every day."""

    settings = settings or OpeningBarSettings()
    trades, days = [], []
    if bars is None or bars.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS), pd.DataFrame(columns=DAY_COLUMNS)
    bars = bars.sort_index()
    for day, session in bars.groupby(bars.index.normalize()):
        # A session that does not start at the first bar (missing data) has no first bar to read.
        if len(session) < 3 or session.index[0].strftime("%H:%M") != "09:30":
            continue
        first = session.iloc[0]
        long_day = is_long_day(first, settings.min_move)
        short_day = is_short_day(first, settings.min_move)
        days.append({"Ticker": ticker, "Date": day, "First bar open": first["Open"], "First bar close": first["Close"],
                     "Move": first["Close"] - first["Open"], "Long day": long_day, "Short day": short_day})
        if not (long_day or (short_day and settings.allow_short)):
            continue
        side = 1 if long_day else -1
        entry, stop = float(first["Close"]), float(first["Low"] if long_day else first["High"])
        bar_length = pd.Timedelta(minutes=settings.bar_minutes)
        shares = position_shares(settings.capital, entry, stop, settings.risk_pct, settings.cap_at_capital)
        if shares < 1:
            continue
        # A short is the long trade on upside-down prices.
        k, exit_price, reason = exit_session(session if long_day else mirrored(session), side * entry, side * stop,
                                              settings.skip_bars, settings.exit_fill, settings.exit_mode,
                                              flat_bar_position(session.index, settings.bar_minutes),
                                              settings.custom_exit)
        exit_price *= side
        risk = shares * abs(entry - stop)
        # Slippage: every fill is a few cents worse than the bar price, in the trade's direction.
        entry_fill, exit_fill = entry + side * settings.slippage, exit_price - side * settings.slippage
        costs = (order_costs(shares, entry_fill, sell=not long_day, plan=settings.commission_plan)
                 + order_costs(shares, exit_fill, sell=long_day, plan=settings.commission_plan))
        pnl = side * shares * (exit_fill - entry_fill) - costs
        trades.append({
            "Ticker": ticker, "Date": day, "Side": "Long" if long_day else "Short",
            "Entry time": session.index[0] + bar_length, "Entry": entry_fill, "Stop": stop,
            "Exit time": session.index[k] + (pd.Timedelta(0) if reason == "End of day" else bar_length),
            "Exit": exit_fill, "Exit reason": reason, "Shares": shares, "Risk $": risk, "Costs $": costs, "P&L": pnl,
            "Return %": pnl / settings.capital * 100, "R": pnl / risk if risk > 0 else np.nan,
        })
    return pd.DataFrame(trades, columns=TRADE_COLUMNS), pd.DataFrame(days, columns=DAY_COLUMNS)


def run_opening_bar(bars_by_ticker, settings=None):
    """Trades and day calls of the rule on every ticker ({ticker: 5-minute bars}), each with its own capital."""

    results = [opening_bar_trades(frame, settings, ticker) for ticker, frame in bars_by_ticker.items()]
    trades = [t for t, _ in results if not t.empty]
    days = [d for _, d in results if not d.empty]
    trades = (pd.concat(trades, ignore_index=True).sort_values(["Date", "Ticker"], ignore_index=True) if trades
              else pd.DataFrame(columns=TRADE_COLUMNS))
    days = pd.concat(days, ignore_index=True) if days else pd.DataFrame(columns=DAY_COLUMNS)
    return trades, days


def summarize_opening_bar(trades, days=None):
    """Headline statistics: trades, win rate, net, average win and loss, profit factor, drawdown, exit mix."""

    pnl = trades["P&L"].astype(float) if len(trades) else pd.Series(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    equity = pnl.cumsum()
    drawdown = (equity - equity.cummax().clip(lower=0)).min() if len(equity) else 0.0
    gross_loss = -losses.sum()
    reasons = trades["Exit reason"].value_counts(normalize=True) * 100 if len(trades) else pd.Series(dtype=float)
    return {
        "Days": int(len(days)) if days is not None else np.nan,
        "Long days %": float(days["Long day"].mean() * 100) if days is not None and len(days) else np.nan,
        "Trades": int(len(pnl)),
        "Win rate %": float(len(wins) / len(pnl) * 100) if len(pnl) else np.nan,
        "Net P&L": float(pnl.sum()),
        "Costs": float(trades["Costs $"].sum()) if len(trades) else 0.0,
        "Avg / trade": float(pnl.mean()) if len(pnl) else np.nan,
        "Avg win": float(wins.mean()) if len(wins) else 0.0,
        "Avg loss": float(losses.mean()) if len(losses) else 0.0,
        "Avg R": float(trades["R"].mean()) if len(trades) else np.nan,
        "Profit factor": float(wins.sum() / gross_loss) if gross_loss > 0 else np.nan,
        "Max drawdown": float(min(drawdown, 0.0)),
        "Stopped out %": float(reasons.get("Stop", 0.0)),
        "Pullback exit %": float(reasons.get("Pullback", 0.0)),
        "Lower high exit %": float(reasons.get("Lower high", 0.0)),
        "Custom exit %": float(reasons.get("Custom", 0.0)),
        "End of day %": float(reasons.get("End of day", 0.0)),
    }


def ticker_table(trades, days):
    """Per-ticker summary, best net P&L first."""

    rows = {}
    for ticker in days["Ticker"].unique() if len(days) else []:
        rows[ticker] = summarize_opening_bar(trades[trades["Ticker"] == ticker], days[days["Ticker"] == ticker])
    table = pd.DataFrame(rows).T
    return table.sort_values("Net P&L", ascending=False) if len(table) else table
