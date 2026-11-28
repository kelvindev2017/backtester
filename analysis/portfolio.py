"""Portfolio checker: a mix of funds or stocks in sleeves, with optional trend and volatility rules, rebalanced on a
schedule and compared with buy & hold.

A portfolio is a list of holdings (ticker, sleeve, share of the portfolio, trend filter, stand-ins for the years
before the ticker existed) and a rule per sleeve:

- Weighting "Fixed": the sleeve's holdings keep the shares typed in.
- Weighting "Inverse volatility": the sleeve's total share is split by 1 / each holding's recent volatility
  (63-day EWMA), so calmer assets get more (risk parity).
- Trend filter (per holding, in trading days, 0 = off): the holding is kept only while its close is above its
  moving average of that many days; otherwise its share sits in cash.
- Volatility target (per sleeve, % a year, 0 = off): the sleeve is scaled so its recent volatility (20-day EWMA of
  the sleeve as held) is the target, at most "Max exposure" of its share. The rest sits in cash.

Target weights are decided at a close and traded at the next rebalance day's close (Daily, Weekly = last trading
day of the week, Monthly, Quarterly, Yearly = last trading day of the period). Between trades holdings drift with
prices. Cash earns the 3-month T-bill; leverage borrows at the T-bill + a spread. Costs are a share of turnover.

Prices are total-return closes (dividends in). A ticker with a short history can borrow the returns of one or more
stand-ins (for example IJS before AVUV existed) for the years before it started.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252
INVERSE_VOL_SPAN = 63
VOL_TARGET_SPAN = 20
REBALANCE_CHOICES = {"Daily": "D", "Weekly (Friday)": "W", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
WEIGHTINGS = ["Fixed", "Inverse volatility"]


@dataclass
class Holding:
    ticker: str
    weight: float                 # share of the portfolio, 0..1
    sleeve: str = ""
    trend_days: int = 0
    stand_ins: tuple = ()
    note: str = ""


@dataclass
class SleeveRule:
    weighting: str = "Fixed"
    vol_target: float = 0.0       # a year, 0..1; 0 = off
    max_exposure: float = 1.0     # of the sleeve's share


@dataclass
class PortfolioSpec:
    holdings: list
    sleeves: dict = field(default_factory=dict)
    rebalance: str = "W"
    leverage: float = 1.0
    cost: float = 0.001
    borrow_spread: float = 0.005
    band: float = 0.0             # skip a rebalance when no weight is further than this from its target

    def sleeve_of(self, holding):
        return holding.sleeve or holding.ticker

    def rule(self, sleeve):
        return self.sleeves.get(sleeve) or SleeveRule()

    def tickers(self):
        return list(dict.fromkeys(h.ticker for h in self.holdings))

    def all_symbols(self):
        symbols = []
        for h in self.holdings:
            symbols += [h.ticker, *h.stand_ins]
        return list(dict.fromkeys(s for s in symbols if s))


def parse_stand_ins(text):
    """'IJS, VBR' -> ('IJS', 'VBR')."""

    return tuple(part.strip().upper() for part in str(text or "").replace(";", ",").split(",") if part.strip())


# ------------------------------------------------------------------ prices
def _filled(series):
    """Prices carried over days the series did not trade (a holiday on its exchange), only inside its history."""

    first, last = series.first_valid_index(), series.last_valid_index()
    if first is None:
        return series
    inside = (series.index >= first) & (series.index <= last)
    return series.ffill(limit=5).where(inside)


def splice(primary, stand_ins):
    """``primary``'s daily returns where it has prices, and before that the first stand-in that has a return that
    day. Returns (price index starting at 1 the day before the first return, Series naming each day's source)."""

    index = primary.index
    primary = _filled(primary)
    returns = primary.pct_change(fill_method=None)
    source = pd.Series(np.where(returns.notna(), primary.name, None), index=index, dtype=object)
    first = primary.first_valid_index()
    # The first day of its own prices has no return of its own yet, so the stand-in covers that day too.
    before = (index <= first) if first is not None else np.ones(len(index), bool)
    for stand in stand_ins:
        if stand is None or stand.dropna().empty:
            continue
        r = _filled(stand.reindex(index)).pct_change(fill_method=None)
        fill = before & returns.isna().to_numpy() & r.notna().to_numpy()
        returns = returns.where(~fill, r)
        source = source.where(~fill, stand.name)
    start = returns.first_valid_index()
    if start is None:
        return pd.Series(np.nan, index=index), source
    position = index.get_loc(start)
    level = (1 + returns.fillna(0)).cumprod()
    level.iloc[:max(position - 1, 0)] = np.nan
    if position == 0:
        level = level / (1 + returns.iloc[0])
    last = primary.last_valid_index()
    return (level.where(index <= last) if last is not None else level), source


def build_prices(closes, spec):
    """Spliced price index per holding ticker, plus a table of where each ticker's history comes from."""

    prices, rows = {}, []
    for holding in spec.holdings:
        if holding.ticker in prices:
            continue
        own = closes[holding.ticker] if holding.ticker in closes else pd.Series(np.nan, index=closes.index)
        own = own.rename(holding.ticker)
        stands = [closes[s].rename(s) for s in holding.stand_ins if s in closes]
        level, source = splice(own, stands)
        prices[holding.ticker] = level
        own_start = own.first_valid_index()
        start = level.first_valid_index()
        used = [s for s in holding.stand_ins if (source == s).any()]
        rows.append({"Ticker": holding.ticker, "Own prices from": own_start, "With stand-ins from": start,
                     "Stand-ins used": ", ".join(used)})
    return pd.DataFrame(prices, index=closes.index), pd.DataFrame(rows)


# ------------------------------------------------------------------ rules
def ewma_vol(returns, span):
    return np.sqrt((returns ** 2).ewm(span=span, min_periods=span).mean() * TRADING_DAYS)


def warmup_days(spec):
    days = [h.trend_days for h in spec.holdings]
    for sleeve in {spec.sleeve_of(h) for h in spec.holdings}:
        rule = spec.rule(sleeve)
        if rule.weighting == "Inverse volatility":
            days.append(INVERSE_VOL_SPAN)
        if rule.vol_target > 0:
            days.append(VOL_TARGET_SPAN + (INVERSE_VOL_SPAN if rule.weighting == "Inverse volatility" else 0))
    return max(days + [0])


def target_weights(prices, spec):
    """Target weight of each ticker decided at each close (the leverage applied), one column per ticker.

    Two rows of the same ticker in one sleeve add up; the same ticker in two sleeves adds up too.
    """

    returns = prices.pct_change(fill_method=None)
    total = pd.DataFrame(0.0, index=prices.index, columns=spec.tickers())
    sleeves = {}
    for holding in spec.holdings:
        sleeves.setdefault(spec.sleeve_of(holding), []).append(holding)
    for name, members in sleeves.items():
        rule = spec.rule(name)
        share = sum(h.weight for h in members)
        if share <= 0:
            continue
        tickers = [h.ticker for h in members]
        if rule.weighting == "Inverse volatility":
            vols = ewma_vol(returns[list(dict.fromkeys(tickers))], INVERSE_VOL_SPAN)
            inverse = (1 / vols).replace([np.inf], np.nan)
            base = pd.DataFrame({i: inverse[h.ticker] for i, h in enumerate(members)})
            base = base.div(base.sum(axis=1), axis=0)
        else:
            base = pd.DataFrame({i: pd.Series(h.weight / share, index=prices.index) for i, h in enumerate(members)})
        # Trend filter: kept only while above the average; its part goes to cash.
        for i, h in enumerate(members):
            if h.trend_days > 0:
                above = prices[h.ticker] > prices[h.ticker].rolling(h.trend_days, min_periods=h.trend_days).mean()
                base[i] = base[i] * above.astype(float)
        base = base.fillna(0.0)
        if rule.vol_target > 0:
            book = sum((base[i].shift(1) * returns[h.ticker]).fillna(0) for i, h in enumerate(members))
            book = book.where(returns[tickers].notna().all(axis=1))
            book_vol = ewma_vol(book.to_frame("b"), VOL_TARGET_SPAN)["b"]
            scale = (rule.vol_target / book_vol).replace([np.inf], np.nan)
            cap = rule.max_exposure / base.sum(axis=1).replace(0, np.nan)
            base = base.mul(np.minimum(scale, cap).fillna(0), axis=0)
        for i, h in enumerate(members):
            total[h.ticker] += base[i] * share
    return total * spec.leverage


def rebalance_flags(index, freq):
    """True on the trading days the portfolio trades: every day, or the last trading day of each week/month/..."""

    if freq == "D":
        return np.ones(len(index), bool)
    period = {"W": "W-FRI", "M": "M", "Q": "Q", "Y": "Y"}[freq]
    days = pd.Series(index, index=index)
    last = days.groupby(index.to_period(period)).transform("max")
    flags = (days == last).to_numpy().copy()
    # The newest period may not be over yet: its last day so far trades only if the next weekday starts a new one.
    if len(index):
        flags[-1] = (index[-1] + pd.offsets.BDay(1)).to_period(period) != index[-1].to_period(period)
    return flags


def simulate(prices, cash, spec, start=None):
    """Daily returns of the portfolio, the weights held at each close (after trading) and the targets.

    ``cash``: daily T-bill return (a fraction per day) by date. ``start``: first day to hold anything; by
    default the first day every ticker has a price and every rule has its look-back.
    """

    targets = target_weights(prices, spec)
    ready = prices[spec.tickers()].notna().all(axis=1)
    if not ready.any():
        return pd.Series(dtype=float), pd.DataFrame(), targets, None
    first = ready.idxmax()
    position = prices.index.get_loc(first) + warmup_days(spec)
    if position >= len(prices.index) - 1:
        return pd.Series(dtype=float), pd.DataFrame(), targets, None
    begin = prices.index[position]
    if start is not None and pd.Timestamp(start) > begin:
        begin = prices.index[prices.index.searchsorted(pd.Timestamp(start))] if pd.Timestamp(start) <= prices.index[-1] else None
        if begin is None:
            return pd.Series(dtype=float), pd.DataFrame(), targets, None
    index = prices.index[prices.index >= begin]
    r = prices[targets.columns].pct_change(fill_method=None).reindex(index).fillna(0).to_numpy()
    c = cash.reindex(index).ffill().fillna(0).to_numpy()
    T = targets.reindex(index).fillna(0).to_numpy()
    flags = rebalance_flags(index, spec.rebalance)
    held = np.zeros_like(T)
    out = np.zeros(len(index))
    h = T[0].copy()                                          # bought at the first day's close
    out[0] = -spec.cost * np.abs(h).sum()
    held[0] = h
    spread = spec.borrow_spread / TRADING_DAYS
    for t in range(1, len(index)):
        gross = h.sum()
        ret = (h * r[t]).sum() + max(1 - gross, 0) * c[t] - max(gross - 1, 0) * (c[t] + spread)
        value = 1 + ret
        h = h * (1 + r[t]) / value if value > 0 else h * 0
        cost = 0.0
        if flags[t]:
            target = T[t - 1]                                # decided at the previous close
            if np.abs(target - h).max() > spec.band:
                cost = spec.cost * np.abs(target - h).sum()
                h = target.copy()
        out[t] = ret - cost
        held[t] = h
    return (pd.Series(out, index=index), pd.DataFrame(held, index=index, columns=targets.columns), targets, begin)


def buy_and_hold(price, index):
    """Daily returns of holding ``price`` over ``index`` (0 on the first day)."""

    r = price.reindex(index).pct_change(fill_method=None).fillna(0)
    if len(r):
        r.iloc[0] = 0.0
    return r


# ------------------------------------------------------------------ results
def drawdown(returns):
    equity = (1 + returns).cumprod()
    return equity / equity.cummax().clip(lower=1) - 1


def yearly(returns):
    """Return and worst drawdown inside each calendar year (from the year's start or a later high), in %."""

    groups = returns.groupby(returns.index.year)
    return pd.DataFrame({
        "Return %": groups.apply(lambda s: (1 + s).prod() - 1) * 100,
        "Worst drawdown in the year %": groups.apply(lambda s: drawdown(s).min()) * 100,
    })


def stats(returns):
    """Headline numbers of a daily return series, or {} when it is too short."""

    returns = returns.dropna()
    if len(returns) < 20:
        return {}
    equity = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1 if equity.iloc[-1] > 0 else -1.0
    dd = drawdown(returns)
    by_year = yearly(returns)
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    return {
        "From": returns.index[0], "To": returns.index[-1], "Years": years,
        "Return %/yr": cagr * 100, "Total return %": (equity.iloc[-1] - 1) * 100,
        "Max drawdown %": dd.min() * 100,
        "Worst year's drawdown %": by_year["Worst drawdown in the year %"].min(),
        "Worst year %": by_year["Return %"].min(),
        "Volatility %/yr": vol * 100,
        "Sharpe": returns.mean() / returns.std() * np.sqrt(TRADING_DAYS) if returns.std() > 0 else np.nan,
        "Return / max drawdown": cagr / -dd.min() if dd.min() < 0 else np.nan,
    }


def target_check(row, min_return, max_dd, max_year_dd):
    """Which of Kelvin's three targets a stats row meets: return above, both drawdowns smaller than the limits."""

    return {
        "Return": row.get("Return %/yr", np.nan) > min_return,
        "Max drawdown": row.get("Max drawdown %", np.nan) > -max_dd,
        "Yearly drawdown": row.get("Worst year's drawdown %", np.nan) > -max_year_dd,
    }


def next_rebalance_day(last_day, freq):
    """The calendar day of the next scheduled trade after ``last_day`` (a weekday; holidays not known)."""

    last_day = pd.Timestamp(last_day)
    if freq == "D":
        return last_day + pd.offsets.BDay(1)
    offset = {"W": pd.offsets.Week(weekday=4), "M": pd.offsets.BMonthEnd(), "Q": pd.offsets.BQuarterEnd(),
              "Y": pd.offsets.BYearEnd()}[freq]
    return offset.rollforward(last_day + pd.Timedelta(days=1))


# ------------------------------------------------------------------ presets
def preset_spec(name, rebalance=None, leverage=None):
    data = PRESETS[name]
    holdings = [Holding(ticker=r["Ticker"], weight=r["Weight %"] / 100, sleeve=r.get("Sleeve", ""),
                        trend_days=int(r.get("Trend days", 0) or 0), stand_ins=parse_stand_ins(r.get("Stand-ins", "")),
                        note=r.get("Note", "")) for r in data["holdings"]]
    sleeves = {s["Sleeve"]: SleeveRule(weighting=s["Weighting"], vol_target=s["Vol target %"] / 100,
                                       max_exposure=s["Max exposure %"] / 100) for s in data.get("sleeves", [])}
    return PortfolioSpec(holdings=holdings, sleeves=sleeves,
                         rebalance=REBALANCE_CHOICES[rebalance or data["rebalance"]],
                         leverage=leverage if leverage is not None else data.get("leverage", 1.0),
                         band=data.get("band", 0.0))


_TWO_SLEEVE = {
    "holdings": [
        {"Sleeve": "Nasdaq trend", "Ticker": "QQQ", "Weight %": 50.0, "Trend days": 200, "Stand-ins": "",
         "Note": "Nasdaq-100"},
        {"Sleeve": "Risk parity", "Ticker": "SPY", "Weight %": 50 / 3, "Trend days": 100, "Stand-ins": "",
         "Note": "S&P 500"},
        {"Sleeve": "Risk parity", "Ticker": "TLT", "Weight %": 50 / 3, "Trend days": 100, "Stand-ins": "",
         "Note": "Long Treasuries"},
        {"Sleeve": "Risk parity", "Ticker": "GLD", "Weight %": 50 / 3, "Trend days": 100, "Stand-ins": "GC=F",
         "Note": "Gold (gold futures before 2004)"},
    ],
    "sleeves": [
        {"Sleeve": "Nasdaq trend", "Weighting": "Fixed", "Vol target %": 10.0, "Max exposure %": 100.0},
        {"Sleeve": "Risk parity", "Weighting": "Inverse volatility", "Vol target %": 0.0, "Max exposure %": 100.0},
    ],
    "rebalance": "Weekly (Friday)",
    "band": 0.02,
}

_FIVE_ETF = [
    {"Sleeve": "", "Ticker": "SPYM", "Weight %": 45.0, "Trend days": 0, "Stand-ins": "SPY", "Note": "Base"},
    {"Sleeve": "", "Ticker": "SPMO", "Weight %": 15.0, "Trend days": 0, "Stand-ins": "MTUM", "Note": "Momentum"},
    {"Sleeve": "", "Ticker": "AVUV", "Weight %": 20.0, "Trend days": 0, "Stand-ins": "IJS", "Note": "Non-semi/AI"},
    {"Sleeve": "", "Ticker": "SOXX", "Weight %": 10.0, "Trend days": 0, "Stand-ins": "", "Note": "Semi and AI"},
    {"Sleeve": "", "Ticker": "VXUS", "Weight %": 10.0, "Trend days": 0, "Stand-ins": "VEU, EFA", "Note": "Non-US"},
]

PRESETS = {
    "Two-sleeve low drawdown (study)": {**_TWO_SLEEVE, "leverage": 1.0},
    "Two-sleeve low drawdown at 1.5x": {**_TWO_SLEEVE, "leverage": 1.5},
    "5-ETF mix": {"holdings": _FIVE_ETF, "sleeves": [], "rebalance": "Quarterly"},
    "5-ETF mix, each with 200-day trend": {
        "holdings": [{**row, "Trend days": 200} for row in _FIVE_ETF], "sleeves": [], "rebalance": "Weekly (Friday)",
        "band": 0.02},
    "60/40 SPY / IEF": {
        "holdings": [{"Sleeve": "", "Ticker": "SPY", "Weight %": 60.0, "Trend days": 0, "Stand-ins": "", "Note": "S&P 500"},
                     {"Sleeve": "", "Ticker": "IEF", "Weight %": 40.0, "Trend days": 0, "Stand-ins": "",
                      "Note": "7-10y Treasuries"}],
        "sleeves": [], "rebalance": "Monthly"},
}
