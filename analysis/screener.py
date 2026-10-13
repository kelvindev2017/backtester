"""Stock screening: market regime, market breadth, industry leaders and
per-stock technical filters.

Everything here is pure pandas on daily OHLCV frames (one per ticker), so it
can be tested offline. Data loading lives in ``App.py``.
"""

import numpy as np
import pandas as pd

from analysis.indicators import ema, sma
from analysis.signal_groups import MOVING_AVERAGE, evaluate_group_condition, group_condition_label

TRADING_DAYS_PER_YEAR = 252
OPERATORS = [">", "<", ">=", "<=", "Cross Above", "Cross Below"]
LINE_KINDS = ["Close", "SMA", "EMA", "Value"]


# ============================================================
# Lines (Close, moving averages, constants)
# ============================================================

def line_series(df, kind, period=None):
    """The series a condition compares: the close, an SMA/EMA, or a constant."""

    close = df["Close"]
    if kind == "Close":
        return close
    if kind == "SMA":
        return sma(close, int(period))
    if kind == "EMA":
        return ema(close, int(period))
    if kind == "Value":
        return pd.Series(float(period), index=close.index)
    raise ValueError(f"Unknown line kind: {kind}")


def line_label(kind, period=None):
    if kind == "Close":
        return "Close"
    if kind == "Value":
        return f"{float(period):g}"
    return f"{kind} {int(period)}"


def condition_label(condition):
    if condition.get("group", MOVING_AVERAGE) != MOVING_AVERAGE:
        return group_condition_label(condition)
    return (
        f"{line_label(condition['left_kind'], condition.get('left_period'))} {condition['operator']} "
        f"{line_label(condition['right_kind'], condition.get('right_period'))}"
    )


def evaluate_condition(df, condition, cross_window=5):
    """True when ``condition`` holds on the last bar.

    ``condition`` is a dict with ``left_kind``, ``left_period``, ``operator``,
    ``right_kind`` and ``right_period``. Cross Above / Cross Below are true
    when the cross happened within the last ``cross_window`` bars and the
    lines are still on the crossed side. Missing data (e.g. a 200-bar MA on a
    150-bar history) counts as not passing.
    """

    left = line_series(df, condition["left_kind"], condition.get("left_period"))
    right = line_series(df, condition["right_kind"], condition.get("right_period"))
    if left.empty or pd.isna(left.iloc[-1]) or pd.isna(right.iloc[-1]):
        return False
    operator = condition["operator"]
    last_left, last_right = float(left.iloc[-1]), float(right.iloc[-1])
    if operator == ">":
        return last_left > last_right
    if operator == "<":
        return last_left < last_right
    if operator == ">=":
        return last_left >= last_right
    if operator == "<=":
        return last_left <= last_right
    if operator in ("Cross Above", "Cross Below"):
        above = left > right
        valid = left.notna() & right.notna()
        if operator == "Cross Above":
            crossed = above & ~above.shift(1, fill_value=True) & valid & valid.shift(1, fill_value=False)
            still = last_left > last_right
        else:
            crossed = ~above & above.shift(1, fill_value=False) & valid & valid.shift(1, fill_value=False)
            still = last_left < last_right
        return bool(still and crossed.iloc[-int(cross_window):].any())
    raise ValueError(f"Unknown operator: {operator}")


# ============================================================
# Per-ticker snapshot
# ============================================================

def stock_snapshot(df, ma_periods=(20, 50, 200), high_window=TRADING_DAYS_PER_YEAR, recent_days=20, volume_period=50):
    """Latest price, MAs, volume and 52-week-range figures for one ticker."""

    df = df.dropna(subset=["Close"])
    if df.empty:
        return {}
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]
    last_close = float(close.iloc[-1])
    previous_close = float(close.iloc[-2]) if len(close) > 1 else np.nan
    window = min(high_window, len(df))
    high_52 = float(high.iloc[-window:].max())
    low_52 = float(low.iloc[-window:].min())
    rolling_high = high.rolling(high_window, min_periods=1).max()
    rolling_low = low.rolling(high_window, min_periods=1).min()
    recent = int(recent_days)

    snapshot = {
        "Last date": df.index[-1],
        "Close": last_close,
        "Prev close": previous_close,
        "Change %": (last_close / previous_close - 1) * 100 if previous_close else np.nan,
        "52w high": high_52,
        "52w low": low_52,
        "% from 52w high": (last_close / high_52 - 1) * 100 if high_52 else np.nan,
        "% above 52w low": (last_close / low_52 - 1) * 100 if low_52 else np.nan,
        f"New 52w high in last {recent}d": bool((high.iloc[-recent:] >= rolling_high.iloc[-recent:]).any()),
        f"New 52w low in last {recent}d": bool((low.iloc[-recent:] <= rolling_low.iloc[-recent:]).any()),
        "Volume": float(volume.iloc[-1]),
        f"Avg volume {volume_period}d": float(volume.iloc[-volume_period:].mean()),
    }
    average_volume = snapshot[f"Avg volume {volume_period}d"]
    snapshot["Volume / avg"] = snapshot["Volume"] / average_volume if average_volume else np.nan
    for label, bars in [("1w", 5), ("1m", 21), ("3m", 63), ("6m", 126), ("1y", TRADING_DAYS_PER_YEAR)]:
        snapshot[f"Return {label} %"] = (
            (last_close / float(close.iloc[-bars - 1]) - 1) * 100 if len(close) > bars else np.nan
        )
    prior_year = close.loc[close.index < pd.Timestamp(year=df.index[-1].year, month=1, day=1)]
    snapshot["Return YTD %"] = (last_close / float(prior_year.iloc[-1]) - 1) * 100 if not prior_year.empty else np.nan
    for period in ma_periods:
        value = sma(close, period).iloc[-1]
        snapshot[f"SMA {period}"] = float(value) if pd.notna(value) else np.nan
        snapshot[f"Close vs SMA {period} %"] = (last_close / value - 1) * 100 if pd.notna(value) and value else np.nan
    return snapshot


def snapshot_table(prices, **kwargs):
    """``stock_snapshot`` for every ticker in ``prices`` (dict ticker -> OHLCV)."""

    rows = {}
    for ticker, df in prices.items():
        if df is None or df.empty:
            continue
        row = stock_snapshot(df, **kwargs)
        if row:
            rows[ticker] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Ticker"
    return table


# ============================================================
# Market regime and breadth
# ============================================================

def index_trend(df, fast=50, slow=200):
    """Short trend verdict for an index: close vs the MAs and MA order."""

    close = df["Close"].dropna()
    if close.empty:
        return "No data"
    fast_ma, slow_ma = sma(close, fast).iloc[-1], sma(close, slow).iloc[-1]
    last = close.iloc[-1]
    if pd.isna(slow_ma) or pd.isna(fast_ma):
        return "Not enough history"
    if last > fast_ma > slow_ma:
        return "Uptrend"
    if last < fast_ma < slow_ma:
        return "Downtrend"
    return "Mixed"


def breadth_history(prices, ma_periods=(50, 200), high_window=TRADING_DAYS_PER_YEAR, recent_days=20, min_history=60):
    """Daily market-breadth series over a universe.

    Columns: ``Stocks`` (with data that day), ``% above SMA n`` for each
    period, ``% new 52w high today`` / ``% new 52w low today``, and
    ``% new 52w high in last Nd`` / ``% new 52w low in last Nd`` (share of
    stocks that set a new 52-week high/low at least once in the trailing
    ``recent_days`` bars). Tickers with fewer than ``min_history`` bars are
    skipped.
    """

    closes, highs, lows = {}, {}, {}
    for ticker, df in prices.items():
        if df is None or len(df.dropna(subset=["Close"])) < min_history:
            continue
        closes[ticker], highs[ticker], lows[ticker] = df["Close"], df["High"], df["Low"]
    if not closes:
        return pd.DataFrame()
    close = pd.DataFrame(closes).sort_index()
    high = pd.DataFrame(highs).reindex(close.index)
    low = pd.DataFrame(lows).reindex(close.index)
    has_data = close.notna()

    result = pd.DataFrame(index=close.index)
    result["Stocks"] = has_data.sum(axis=1)
    for period in ma_periods:
        average = close.rolling(period, min_periods=period).mean()
        result[f"% above SMA {period}"] = (close > average).sum(axis=1) / average.notna().sum(axis=1).replace(0, np.nan) * 100

    # A new high needs a full 52-week window behind it, otherwise every
    # recently listed stock would "make a new high" on its first days.
    full_window = high.notna().rolling(high_window, min_periods=1).sum() >= high_window * 0.9
    new_high = (high >= high.rolling(high_window, min_periods=1).max()) & full_window & has_data
    new_low = (low <= low.rolling(high_window, min_periods=1).min()) & full_window & has_data
    eligible = (full_window & has_data).sum(axis=1).replace(0, np.nan)
    result["% new 52w high today"] = new_high.sum(axis=1) / eligible * 100
    result["% new 52w low today"] = new_low.sum(axis=1) / eligible * 100
    recent = int(recent_days)
    recent_high = new_high.astype(float).rolling(recent, min_periods=1).max() > 0
    recent_low = new_low.astype(float).rolling(recent, min_periods=1).max() > 0
    result[f"% new 52w high in last {recent}d"] = (recent_high & full_window).sum(axis=1) / eligible * 100
    result[f"% new 52w low in last {recent}d"] = (recent_low & full_window).sum(axis=1) / eligible * 100
    return result


def market_gate(index_rows, breadth_latest, rules):
    """Evaluate the market filter.

    ``index_rows``: dict index name -> snapshot dict (from ``stock_snapshot``).
    ``breadth_latest``: last row of ``breadth_history`` (a Series) or None.
    ``rules``: dict with optional keys ``index`` (name), ``index_ma`` (period,
    close must be above that SMA), ``index_ma_order`` ((fast, slow), fast SMA
    must be above slow SMA), ``min_pct_above_ma`` ((period, pct)) and
    ``min_pct_new_highs`` ((column, pct)).

    Returns (passed, list of (description, passed or None when no data)).
    """

    checks = []
    name = rules.get("index")
    row = index_rows.get(name, {}) if name else {}
    if name and rules.get("index_ma"):
        period = rules["index_ma"]
        value = row.get(f"Close vs SMA {period} %")
        checks.append((f"{name} close above its SMA {period}", None if value is None or pd.isna(value) else value > 0))
    if name and rules.get("index_ma_order"):
        fast, slow = rules["index_ma_order"]
        fast_value, slow_value = row.get(f"SMA {fast}"), row.get(f"SMA {slow}")
        ok = None
        if fast_value is not None and slow_value is not None and pd.notna(fast_value) and pd.notna(slow_value):
            ok = fast_value > slow_value
        checks.append((f"{name} SMA {fast} above SMA {slow}", ok))
    if rules.get("min_pct_above_ma") and breadth_latest is not None:
        period, threshold = rules["min_pct_above_ma"]
        value = breadth_latest.get(f"% above SMA {period}")
        checks.append((
            f"At least {threshold:g}% of stocks above their SMA {period} (now {value:.1f}%)" if pd.notna(value) else
            f"At least {threshold:g}% of stocks above their SMA {period}",
            None if pd.isna(value) else value >= threshold,
        ))
    if rules.get("min_pct_new_highs") and breadth_latest is not None:
        column, threshold = rules["min_pct_new_highs"]
        value = breadth_latest.get(column)
        checks.append((
            f"At least {threshold:g}% of stocks with a {column[2:]} (now {value:.1f}%)" if pd.notna(value) else
            f"At least {threshold:g}% of stocks with a {column[2:]}",
            None if pd.isna(value) else value >= threshold,
        ))
    checks = [(text, None if ok is None else bool(ok)) for text, ok in checks]
    passed = all(ok is True for _, ok in checks)
    return passed, checks


# ============================================================
# Industry leaders
# ============================================================

def _span_days(series):
    return (series.index[-1] - series.index[0]).days


def fundamental_trend(fundamentals):
    """Revenue / net income trend from a quarterly frame (Revenue, Net_Income columns).

    Returns latest quarter values, year-over-year growth versus the same
    quarter a year earlier, trailing-twelve-month figures and their growth,
    and the net margin. Comparisons go by date, not position, because some
    sources (Yahoo for some stocks) skip quarters: a TTM sum is only
    reported when the four quarters are consecutive.
    """

    result = {}
    if fundamentals is None or fundamentals.empty:
        return result
    frame = fundamentals.sort_index()
    for column, label in [("Revenue", "Revenue"), ("Net_Income", "Net income")]:
        if column not in frame.columns:
            continue
        series = frame[column].dropna()
        if series.empty:
            continue
        last_date = series.index[-1]
        result["Last quarter"] = max(result.get("Last quarter", last_date), last_date)
        result[f"{label} (last Q)"] = float(series.iloc[-1])
        year_ago = series.loc[
            (series.index >= last_date - pd.Timedelta(days=400)) & (series.index <= last_date - pd.Timedelta(days=330))
        ]
        if not year_ago.empty:
            base = float(year_ago.iloc[-1])
            result[f"{label} YoY %"] = (series.iloc[-1] / base - 1) * 100 if base > 0 else np.nan
        if len(series) >= 4 and _span_days(series.iloc[-4:]) <= 300:
            ttm = float(series.iloc[-4:].sum())
            result[f"{label} TTM"] = ttm
            if len(series) >= 8 and _span_days(series.iloc[-8:]) <= 670:
                previous_ttm = float(series.iloc[-8:-4].sum())
                result[f"{label} TTM growth %"] = (ttm / previous_ttm - 1) * 100 if previous_ttm > 0 else np.nan
        if len(series) >= 3 and _span_days(series.iloc[-3:]) <= 210:
            last_three = series.iloc[-3:].to_numpy()
            result[f"{label} rising 2Q"] = bool(last_three[2] > last_three[1] > last_three[0])
    revenue = result.get("Revenue (last Q)")
    net_income = result.get("Net income (last Q)")
    if revenue and net_income is not None:
        result["Net margin %"] = net_income / revenue * 100
    return result


def leaders_health(snapshots, ma_periods=(50, 200)):
    """How many industry leaders trade above each SMA, and their average distance from the 52w high."""

    if snapshots is None or snapshots.empty:
        return {}
    health = {"Leaders": len(snapshots)}
    for period in ma_periods:
        column = f"Close vs SMA {period} %"
        if column in snapshots:
            health[f"Above SMA {period}"] = int((snapshots[column] > 0).sum())
    if "% from 52w high" in snapshots:
        health["Avg % from 52w high"] = float(snapshots["% from 52w high"].mean())
    if "Return 3m %" in snapshots:
        health["Avg 3m return %"] = float(snapshots["Return 3m %"].mean())
    return health


# ============================================================
# Stock screen
# ============================================================

def screen_stocks(
    prices,
    conditions,
    max_pct_from_high=None,
    min_price=None,
    min_avg_volume=None,
    min_volume_ratio=None,
    cross_window=5,
    ma_periods=(20, 50, 200),
    recent_days=20,
    extra=None,
    today=None,
):
    """Apply the stock filters to every ticker.

    ``conditions`` mix moving-average conditions (checked on the price
    history) and signal-group conditions (a ``group`` key, checked against a
    column of the snapshot table joined with ``extra``: fundamentals, sector,
    index membership ... indexed by ticker).

    Returns the snapshot table with one boolean column per condition and a
    ``Pass`` column (all filters met), sorted with passing stocks first and
    nearest-to-52w-high first.
    """

    table = snapshot_table(prices, ma_periods=ma_periods, recent_days=recent_days)
    if table.empty:
        return table
    if extra is not None and not extra.empty:
        extra = extra[[column for column in extra.columns if column not in table.columns]]
        table = table.join(extra, how="left")
    passes = pd.Series(True, index=table.index)
    for condition in conditions:
        label = condition_label(condition)
        if label in table.columns:
            label = f"{label} "
        if condition.get("group", MOVING_AVERAGE) == MOVING_AVERAGE:
            table[label] = [evaluate_condition(prices[ticker], condition, cross_window) for ticker in table.index]
        else:
            table[label] = [evaluate_group_condition(table.loc[ticker], condition, today) for ticker in table.index]
        passes &= table[label]
    if max_pct_from_high is not None:
        label = f"Within {max_pct_from_high:g}% of 52w high"
        table[label] = table["% from 52w high"] >= -abs(max_pct_from_high)
        passes &= table[label]
    if min_price is not None:
        passes &= table["Close"] >= min_price
    if min_avg_volume is not None:
        volume_column = next(column for column in table.columns if column.startswith("Avg volume"))
        passes &= table[volume_column] >= min_avg_volume
    if min_volume_ratio is not None:
        passes &= table["Volume / avg"] >= min_volume_ratio
    table.insert(0, "Pass", passes.fillna(False).astype(bool))
    return table.sort_values(["Pass", "% from 52w high"], ascending=[False, False])
