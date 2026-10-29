"""First-5-minute day trade: the first bar of the session decides whether the day is a long day.

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
  at that bar's close);
* anything still held is sold 5 minutes before the close (at the open of the session's last 5-minute bar).

Sizing is a separate function (``position_shares``) and the backtest takes a list of tickers, so a portfolio risk
system or the stock screener can feed them later.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


EXIT_MODES = {
    "Pullback: next close below the last pullback close": "pullback",
    "Lower high: a bar's high below the previous bar's high": "lower_high",
}
EXIT_FILLS = {
    "At the close that broke the pullback (market order)": "close",
    "At the pullback price (as written)": "pullback",
}
TRADE_COLUMNS = ["Ticker", "Date", "Side", "Entry time", "Entry", "Stop", "Exit time", "Exit", "Exit reason", "Shares",
                 "Risk $", "P&L", "Return %", "R"]
DAY_COLUMNS = ["Ticker", "Date", "First bar open", "First bar close", "Move", "Long day", "Short day"]


@dataclass
class OpeningBarSettings:
    min_move: float = 0.0
    skip_bars: int = 2
    exit_mode: str = "pullback"
    risk_pct: float = 10.0
    capital: float = 10_000.0
    cap_at_capital: bool = True
    exit_fill: str = "close"
    fee_per_trade: float = 2.0
    allow_short: bool = False


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

    return pd.DataFrame({"Open": -session["Open"], "High": -session["Low"], "Low": -session["High"],
                         "Close": -session["Close"]}, index=session.index)


def position_shares(capital, entry, stop, risk_pct=10.0, cap_at_capital=True):
    """Shares (long or short) such that a stop-out loses at most ``risk_pct`` % of ``capital`` (0 when there is no risk to size on)."""

    risk_per_share = abs(entry - stop)
    if not (np.isfinite(risk_per_share) and risk_per_share > 0 and entry > 0):
        return 0
    shares = math.floor(capital * risk_pct / 100 / risk_per_share)
    if cap_at_capital:
        shares = min(shares, math.floor(capital / entry))
    return max(shares, 0)


def exit_session(session, entry, stop, skip_bars=2, exit_fill="close", exit_mode="pullback"):
    """Exit of a long entered at the close of the session's first bar: (bar position, price, reason).

    The stop is checked first on every bar (a gap below it fills at the open). The pullback or lower-high exit starts
    after the skipped bars; the session's last bar is the flat-by-the-close exit, at its open.
    """

    opens, highs, lows, closes = (session[c].to_numpy(dtype=float) for c in ("Open", "High", "Low", "Close"))
    last = len(session) - 1
    pullback = np.nan
    for k in range(1, last + 1):
        if lows[k] <= stop:
            return k, min(opens[k], stop), "Stop"
        if k == last:
            return k, opens[k], "End of day"
        if k <= skip_bars:
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


def opening_bar_trades(bars, settings=None, ticker=""):
    """Trades of the rule on one ticker's regular-session 5-minute bars, plus the long/short call of every day."""

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
        shares = position_shares(settings.capital, entry, stop, settings.risk_pct, settings.cap_at_capital)
        if shares < 1:
            continue
        # A short is the long trade on upside-down prices.
        k, exit_price, reason = exit_session(session if long_day else mirrored(session), side * entry, side * stop,
                                              settings.skip_bars, settings.exit_fill, settings.exit_mode)
        exit_price *= side
        risk = shares * abs(entry - stop)
        pnl = side * shares * (exit_price - entry) - settings.fee_per_trade
        trades.append({
            "Ticker": ticker, "Date": day, "Side": "Long" if long_day else "Short", "Entry time": session.index[0] + pd.Timedelta(minutes=5), "Entry": entry,
            "Stop": stop, "Exit time": session.index[k] + (pd.Timedelta(0) if reason == "End of day"
                                                            else pd.Timedelta(minutes=5)),
            "Exit": exit_price, "Exit reason": reason, "Shares": shares, "Risk $": risk, "P&L": pnl,
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
        "Avg / trade": float(pnl.mean()) if len(pnl) else np.nan,
        "Avg win": float(wins.mean()) if len(wins) else 0.0,
        "Avg loss": float(losses.mean()) if len(losses) else 0.0,
        "Avg R": float(trades["R"].mean()) if len(trades) else np.nan,
        "Profit factor": float(wins.sum() / gross_loss) if gross_loss > 0 else np.nan,
        "Max drawdown": float(min(drawdown, 0.0)),
        "Stopped out %": float(reasons.get("Stop", 0.0)),
        "Pullback exit %": float(reasons.get("Pullback", 0.0)),
        "Lower high exit %": float(reasons.get("Lower high", 0.0)),
        "End of day %": float(reasons.get("End of day", 0.0)),
    }


def ticker_table(trades, days):
    """Per-ticker summary, best net P&L first."""

    rows = {}
    for ticker in days["Ticker"].unique() if len(days) else []:
        rows[ticker] = summarize_opening_bar(trades[trades["Ticker"] == ticker], days[days["Ticker"] == ticker])
    table = pd.DataFrame(rows).T
    return table.sort_values("Net P&L", ascending=False) if len(table) else table
