"""Portfolio backtest: held core assets plus a sleeve of breakout trades spread over many stocks, rebalanced.

The breakout sleeve gives every stock its own slot of equal size. A slot holds its stock only while that stock is in
a breakout trade and sits in cash otherwise, so a slot's daily return is the trade's return or 0. Rule (from the
2026-11 IBKR trade review, long only, daily bars):

* buy when the price trades above the prior ``lookback``-day high while the prior close is above its ``trend_sma``-day
  SMA, filled at that high or at the open on a gap above it;
* sell at the first close below the ``exit_sma``-day SMA (the entry day included), or during the day at a stop
  ``stop_pct`` % below the entry (at the open on a gap below it);
* ``cost_pct`` % of the traded amount is paid on every buy and every sell.

Rebalancing puts every holding (each core asset, each breakout slot, cash) back to its target share of the total:
on a calendar (yearly, quarterly, monthly) or when any holding has drifted more than a band of percentage points
from its target. A slot that is in a trade at a rebalance keeps the trade and is resized, which is what adding to or
trimming the position does.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


REBALANCE_RULES = {
    "Once a year (first trading day of January)": "yearly",
    "Every quarter": "quarterly",
    "Every month": "monthly",
    "When a holding drifts more than a band from its target": "band",
    "Never": "never",
}


@dataclass
class BreakoutRule:
    lookback: int = 20
    trend_sma: int = 50
    exit_sma: int = 20
    stop_pct: float = 5.0
    cost_pct: float = 0.05


@dataclass
class PortfolioResult:
    equity: pd.Series
    weights: pd.DataFrame
    rebalance_dates: list
    trades: pd.DataFrame
    slot_exposure: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def breakout_slot_returns(daily, rule=None):
    """Daily returns of one breakout slot on ``daily`` OHLC, and the slot's trades."""

    rule = rule or BreakoutRule()
    opens, highs, lows, closes = (daily[c].to_numpy(dtype=float) for c in ("Open", "High", "Low", "Close"))
    n = len(closes)
    returns = np.zeros(n)
    if n == 0:
        return pd.Series(returns, index=daily.index), pd.DataFrame()
    close = pd.Series(closes)
    level = pd.Series(highs).shift(1).rolling(rule.lookback, min_periods=rule.lookback).max().to_numpy()
    trend = (close.rolling(rule.trend_sma).mean().shift(1).to_numpy() if rule.trend_sma
             else np.full(n, -np.inf))
    exit_ma = close.rolling(rule.exit_sma).mean().to_numpy()
    cost = rule.cost_pct / 100
    trades = []
    i = max(rule.lookback, rule.trend_sma, rule.exit_sma) + 1
    while i < n:
        if not (np.isfinite(level[i]) and highs[i] > level[i] and closes[i - 1] > trend[i]):
            i += 1
            continue
        entry = max(opens[i], level[i])
        stop = entry * (1 - rule.stop_pct / 100) if rule.stop_pct else 0.0
        returns[i] += closes[i] / entry - 1 - cost
        j, exit_price, reason = i, closes[i], "close below SMA"
        if closes[i] < exit_ma[i]:
            returns[i] -= cost
        else:
            while True:
                j += 1
                if j >= n:
                    j, exit_price, reason = n - 1, closes[n - 1], "open"
                    break
                if stop and lows[j] <= stop:
                    exit_price = min(opens[j], stop)
                    returns[j] += exit_price / closes[j - 1] - 1 - cost
                    reason = "stop"
                    break
                returns[j] += closes[j] / closes[j - 1] - 1
                if closes[j] < exit_ma[j]:
                    returns[j] -= cost
                    exit_price = closes[j]
                    break
        trades.append({"Entry date": daily.index[i], "Entry": entry, "Exit date": daily.index[j], "Exit": exit_price,
                       "Days held": j - i, "Return %": (exit_price / entry - 1) * 100, "Exit reason": reason})
        i = j + 1
    return pd.Series(returns, index=daily.index), pd.DataFrame(trades)


def _rebalance_due(rule, date, previous, values, targets, band_pct):
    if previous is None or rule == "never":
        return False
    if rule == "yearly":
        return date.year != previous.year
    if rule == "quarterly":
        return (date.year, (date.month - 1) // 3) != (previous.year, (previous.month - 1) // 3)
    if rule == "monthly":
        return (date.year, date.month) != (previous.year, previous.month)
    total = values.sum()
    return total > 0 and bool(np.any(np.abs(values / total - targets) > band_pct / 100))


def run_portfolio(core_returns, core_weights, slot_returns, breakout_weight, cash_return=None,
                  rebalance="yearly", band_pct=5.0, start=None, rebalance_cost_pct=0.05):
    """Equity (starting at 1) of the mix, with its rebalances.

    ``core_returns``: DataFrame of daily returns of the held assets; ``core_weights``: {column: share 0-1}.
    ``slot_returns``: DataFrame of daily breakout-slot returns, one column per stock, sharing ``breakout_weight``
    equally. Whatever share is left is cash, earning ``cash_return`` (a daily Series) or nothing. The band rule
    rebalances when any holding drifts more than ``band_pct`` points; it does not count a breakout slot's own drift,
    since a slot that is out of a trade is just cash. Each rebalance pays ``rebalance_cost_pct`` % of the amount
    moved.
    """

    index = core_returns.index.union(slot_returns.index).sort_values()
    if start is not None:
        index = index[index >= pd.Timestamp(start)]
    core = core_returns.reindex(index).fillna(0.0)
    slots = slot_returns.reindex(index).fillna(0.0)
    cash = (cash_return.reindex(index).fillna(0.0) if cash_return is not None
            else pd.Series(0.0, index=index))
    names = list(core.columns) + list(slots.columns) + ["Cash"]
    n_slots = max(len(slots.columns), 1)
    cash_weight = max(0.0, 1.0 - sum(core_weights.values()) - breakout_weight)
    targets = np.array([core_weights.get(c, 0.0) for c in core.columns]
                       + [breakout_weight / n_slots] * len(slots.columns) + [cash_weight])
    targets = targets / targets.sum()
    returns = np.column_stack([core.to_numpy(), slots.to_numpy(), cash.to_numpy()])
    # For the band test, the sleeve counts as one holding.
    groups = list(range(len(core.columns))) + [len(core.columns)] * len(slots.columns) + [len(core.columns) + 1]
    group_targets = np.bincount(groups, weights=targets)

    values = targets.copy()
    equity, weights, rebalances = [], [], []
    previous = None
    for row, date in enumerate(index):
        grouped = np.bincount(groups, weights=values)
        if _rebalance_due(rebalance, date, previous, grouped, group_targets, band_pct):
            new_values = values.sum() * targets
            moved = np.abs(new_values - values).sum() / 2  # each dollar moved is one sell and one buy
            values = new_values * (1 - 2 * moved * rebalance_cost_pct / 100 / max(values.sum(), 1e-12))
            rebalances.append(date)
        previous = date
        values = values * (1 + returns[row])
        equity.append(values.sum())
        weights.append(values / values.sum())
    weights = pd.DataFrame(weights, index=index, columns=names)
    sleeve = weights[list(slots.columns)].sum(axis=1) if len(slots.columns) else pd.Series(0.0, index=index)
    shown = weights[list(core.columns) + ["Cash"]].copy()
    shown.insert(len(core.columns), "Breakout trades", sleeve)
    exposure = (slots.ne(0).mean(axis=1) if len(slots.columns) else pd.Series(0.0, index=index))
    return PortfolioResult(pd.Series(equity, index=index), shown, rebalances, pd.DataFrame(), exposure)


def summarize(equity, periods_per_year=252):
    """Yearly return (CAGR), worst drop, Sharpe and total return of an equity curve."""

    equity = equity.dropna()
    if len(equity) < 2:
        return {"CAGR %": np.nan, "Worst drop %": np.nan, "Sharpe": np.nan, "Total return %": np.nan}
    returns = equity.pct_change().dropna()
    years = len(returns) / periods_per_year
    total = equity.iloc[-1] / equity.iloc[0]
    std = returns.std()
    return {
        "CAGR %": (total ** (1 / years) - 1) * 100 if years > 0 and total > 0 else np.nan,
        "Worst drop %": float((equity / equity.cummax() - 1).min() * 100),
        "Sharpe": float(returns.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0,
        "Total return %": (total - 1) * 100,
    }


def yearly_returns(equity):
    """Return % of each calendar year (the first and last years are partial)."""

    year_end = equity.groupby(equity.index.year).last()
    start = pd.concat([pd.Series([equity.iloc[0]], index=[year_end.index[0] - 1]), year_end]).shift(1).iloc[1:]
    return (year_end / start - 1) * 100
