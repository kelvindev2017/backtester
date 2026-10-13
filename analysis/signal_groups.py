"""Signal groups for the stock screener and the industry index list.

A screen condition belongs to one group, modelled on TradingView's screener
filters (Price, Chg %, Mkt cap, P/E, Sector, Analyst rating ...). The group
decides which inputs a condition row needs and which column of the screener
table it is checked against. ``Moving average`` conditions are evaluated on
the price history by ``analysis.screener.evaluate_condition``; every other
group compares one column of the per-ticker table built here.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

MOVING_AVERAGE = "Moving average"
NUMBER_OPERATORS = [">", "<", ">=", "<=", "Between"]
CATEGORY_OPERATORS = ["is any of", "is none of"]

GICS_SECTORS = [
    "Communication Services", "Consumer Discretionary", "Consumer Staples", "Energy", "Financials",
    "Health Care", "Industrials", "Information Technology", "Materials", "Real Estate", "Utilities",
]
# Yahoo uses Morningstar-style sector names; map them onto GICS so one
# sector list works for index members (GICS from Wikipedia) and Yahoo data.
YAHOO_TO_GICS_SECTOR = {
    "Basic Materials": "Materials",
    "Communication Services": "Communication Services",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Energy": "Energy",
    "Financial Services": "Financials",
    "Healthcare": "Health Care",
    "Industrials": "Industrials",
    "Real Estate": "Real Estate",
    "Technology": "Information Technology",
    "Utilities": "Utilities",
}
ANALYST_RATINGS = ["Strong buy", "Buy", "Hold", "Underperform", "Sell"]
_RATING_KEYS = {"strong_buy": "Strong buy", "buy": "Buy", "hold": "Hold", "underperform": "Underperform", "sell": "Sell"}
PERF_PERIODS = {
    "1W": "Return 1w %", "1M": "Return 1m %", "3M": "Return 3m %", "6M": "Return 6m %",
    "YTD": "Return YTD %", "1Y": "Return 1y %",
}

# Group -> how its condition rows work.
#   kind "ma":       Left / period / operator / Right / period-or-value (price history)
#   kind "number":   operator and one or two numbers, compared with ``column``
#   kind "category": "is any of" / "is none of" a set of values in ``column``
#   kind "past" / "future": an event date within the last / next N days
# ``fundamental`` marks columns that need Yahoo's per-ticker quote summary.
SIGNAL_GROUPS = {
    MOVING_AVERAGE: {"kind": "ma"},
    "Index": {"kind": "category", "column": "Index"},
    "Price": {"kind": "number", "column": "Close", "default": (">", 5.0)},
    "Chg %": {"kind": "number", "column": "Change %", "default": (">", 0.0)},
    "Mkt cap": {"kind": "number", "column": "Mkt cap ($B)", "fundamental": True, "default": (">", 10.0),
                "help": "Market capitalisation in billions."},
    "P/E": {"kind": "number", "column": "P/E", "fundamental": True, "default": ("Between", 0.0, 30.0),
            "help": "Trailing twelve-month price / earnings."},
    "EPS dil growth": {"kind": "number", "column": "EPS growth YoY %", "fundamental": True, "default": (">", 15.0),
                       "help": "Diluted EPS growth of the latest quarter versus the same quarter a year earlier, in %."},
    "Div yield %": {"kind": "number", "column": "Div yield %", "fundamental": True, "default": (">", 2.0)},
    "Sector": {"kind": "category", "column": "Sector"},
    "Industry": {"kind": "category", "column": "Industry"},
    "Analyst rating": {"kind": "category", "column": "Analyst rating"},
    "Perf %": {"kind": "number", "column": None, "periods": PERF_PERIODS, "default": (">", 0.0),
               "help": "Price change over the chosen period, in %."},
    "Revenue growth": {"kind": "number", "column": "Revenue growth YoY %", "fundamental": True, "default": (">", 10.0),
                       "help": "Revenue growth of the latest quarter versus the same quarter a year earlier, in %."},
    "PEG": {"kind": "number", "column": "PEG", "fundamental": True, "default": ("Between", 0.0, 2.0)},
    "ROE": {"kind": "number", "column": "ROE %", "fundamental": True, "default": (">", 15.0),
            "help": "Return on equity, trailing twelve months, in %."},
    "Beta": {"kind": "number", "column": "Beta", "fundamental": True, "default": ("<", 1.5)},
    "Recent earnings date": {"kind": "past", "column": "Last earnings", "fundamental": True, "default": 30},
    "Upcoming earnings date": {"kind": "future", "column": "Next earnings", "fundamental": True, "default": 14},
}

CATEGORY_OPTIONS = {"Sector": GICS_SECTORS, "Analyst rating": ANALYST_RATINGS}

# Main US index (or ETF, where the index has no Yahoo history) per industry.
INDUSTRY_INDEXES = {
    "S&P 500 (benchmark)": "^GSPC",
    "Semiconductors (PHLX ^SOX)": "^SOX",
    "Technology (XLK)": "XLK",
    "Software (IGV)": "IGV",
    "Internet (FDN)": "FDN",
    "Cybersecurity (CIBR)": "CIBR",
    "Communication Services (XLC)": "XLC",
    "Financials (XLF)": "XLF",
    "Banks (KBW ^BKX)": "^BKX",
    "Regional Banks (KRE)": "KRE",
    "Health Care (XLV)": "XLV",
    "Biotechnology (Nasdaq ^NBI)": "^NBI",
    "Energy (XLE)": "XLE",
    "Oil & Gas E&P (XOP)": "XOP",
    "Industrials (XLI)": "XLI",
    "Aerospace & Defense (ITA)": "ITA",
    "Transportation (Dow ^DJT)": "^DJT",
    "Consumer Discretionary (XLY)": "XLY",
    "Retail (XRT)": "XRT",
    "Homebuilders (ITB)": "ITB",
    "Consumer Staples (XLP)": "XLP",
    "Utilities (XLU)": "XLU",
    "Real Estate (XLRE)": "XLRE",
    "Materials (XLB)": "XLB",
    "Gold Miners (NYSE Arca ^HUI)": "^HUI",
    "Solar (TAN)": "TAN",
    "Uranium (URA)": "URA",
}
DEFAULT_INDUSTRIES = [
    "Semiconductors (PHLX ^SOX)", "Software (IGV)", "Financials (XLF)", "Energy (XLE)", "Health Care (XLV)",
]


def _number(value, scale=1.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number * scale if np.isfinite(number) else np.nan


def _timestamp(value):
    number = _number(value)
    if np.isnan(number) or number <= 0:
        return None
    return pd.Timestamp(datetime.fromtimestamp(number, tz=timezone.utc).date())


def earnings_dates(info, today=None):
    """(last reported, next scheduled) earnings dates from Yahoo's quote fields.

    Yahoo keeps ``earningsTimestamp`` on the last report until it schedules
    the next one, so the fields together give the latest date on or before
    today and the earliest one after it. Either can be None.
    """

    today = pd.Timestamp(today or date_today()).normalize()
    dates = {_timestamp(info.get(key)) for key in (
        "earningsTimestamp", "earningsTimestampStart", "earningsTimestampEnd", "earningsCallTimestampStart")}
    dates.discard(None)
    past = [value for value in dates if value <= today]
    future = [value for value in dates if value > today]
    return (max(past) if past else None), (min(future) if future else None)


def date_today():
    return pd.Timestamp.today().normalize()


FUNDAMENTAL_COLUMNS = [
    "Mkt cap ($B)", "P/E", "EPS growth YoY %", "Div yield %", "Revenue growth YoY %", "PEG", "ROE %", "Beta",
    "Yahoo sector", "Yahoo industry", "Analyst rating", "Last earnings", "Next earnings",
]


def fundamental_row(info, today=None):
    """Screener columns from a ``yf.Ticker(...).info`` dict; {} when Yahoo returned nothing useful."""

    if not info or info.get("quoteType") not in (None, "EQUITY"):
        return {}
    last_earnings, next_earnings = earnings_dates(info, today)
    peg = _number(info.get("trailingPegRatio"))
    if np.isnan(peg):
        peg = _number(info.get("pegRatio"))
    dividend = _number(info.get("dividendYield"))
    row = {
        "Mkt cap ($B)": _number(info.get("marketCap"), 1e-9),
        "P/E": _number(info.get("trailingPE")),
        "EPS growth YoY %": _number(info.get("earningsGrowth", info.get("earningsQuarterlyGrowth")), 100),
        # yfinance reports dividendYield already in percent; non-payers have no field.
        "Div yield %": 0.0 if np.isnan(dividend) else dividend,
        "Revenue growth YoY %": _number(info.get("revenueGrowth"), 100),
        "PEG": peg,
        "ROE %": _number(info.get("returnOnEquity"), 100),
        "Beta": _number(info.get("beta")),
        "Yahoo sector": YAHOO_TO_GICS_SECTOR.get(info.get("sector"), info.get("sector") or ""),
        "Yahoo industry": info.get("industry") or "",
        "Analyst rating": _RATING_KEYS.get(str(info.get("recommendationKey", "")).lower(), ""),
        "Last earnings": last_earnings,
        "Next earnings": next_earnings,
    }
    if all(pd.isna(row[key]) for key in ("Mkt cap ($B)", "P/E", "Beta")) and not row["Yahoo sector"]:
        return {}
    return row


def needs_fundamentals(conditions):
    return any(SIGNAL_GROUPS.get(condition.get("group"), {}).get("fundamental") for condition in conditions)


def condition_column(condition):
    """The screener-table column a non-MA condition reads."""

    spec = SIGNAL_GROUPS[condition["group"]]
    if spec.get("periods"):
        return spec["periods"][condition.get("period", "1M")]
    return spec["column"]


def group_condition_label(condition):
    group = condition["group"]
    kind = SIGNAL_GROUPS[group]["kind"]
    column = condition_column(condition)
    if kind == "number":
        if condition["operator"] == "Between":
            return f"{column} {condition['value']:g} to {condition['value2']:g}"
        return f"{column} {condition['operator']} {condition['value']:g}"
    if kind == "category":
        values = ", ".join(condition.get("values", [])) or "(none)"
        return f"{column} {'is' if condition['operator'] == 'is any of' else 'not'} {values}"
    if kind == "past":
        return f"Earnings in the last {int(condition['days'])} days"
    if kind == "future":
        return f"Earnings in the next {int(condition['days'])} days"
    raise ValueError(f"Unknown group: {group}")


def evaluate_group_condition(row, condition, today=None):
    """True when one ticker's table ``row`` (a Series) meets a non-MA condition. Missing data fails."""

    kind = SIGNAL_GROUPS[condition["group"]]["kind"]
    value = row.get(condition_column(condition))
    if kind == "number":
        number = _number(value)
        if np.isnan(number):
            return False
        operator, target = condition["operator"], float(condition["value"])
        if operator == "Between":
            low, high = sorted((target, float(condition["value2"])))
            return low <= number <= high
        return {">": number > target, "<": number < target, ">=": number >= target, "<=": number <= target}[operator]
    if kind == "category":
        chosen = set(condition.get("values", []))
        if not chosen:
            return True
        present = set(value) if isinstance(value, (list, tuple, set)) else ({value} if isinstance(value, str) and value else set())
        if not present:
            return False
        hit = bool(chosen & present)
        return hit if condition["operator"] == "is any of" else not hit
    if kind in ("past", "future"):
        if value is None or pd.isna(value):
            return False
        today = pd.Timestamp(today or date_today()).normalize()
        days = (pd.Timestamp(value).normalize() - today).days
        window = int(condition["days"])
        return -window <= days <= 0 if kind == "past" else 0 < days <= window
    raise ValueError(f"Unknown group kind: {kind}")


def overlay_frame(prices, start=None, normalize=True):
    """Closes of several tickers on one date index from ``start``, rebased to 100 when ``normalize``."""

    closes = pd.DataFrame({ticker: frame["Close"] for ticker, frame in prices.items() if frame is not None and not frame.empty})
    if closes.empty:
        return closes
    closes = closes.sort_index().ffill()
    if start is not None:
        closes = closes.loc[closes.index >= pd.Timestamp(start)]
    closes = closes.dropna(how="all")
    if normalize and not closes.empty:
        first = closes.apply(lambda column: column.dropna().iloc[0] if column.notna().any() else np.nan)
        closes = closes / first * 100
    return closes


def period_start(last_date, period):
    """First date of a chart period ending at ``last_date`` (1M, 3M, 6M, YTD, 1Y, 2Y, 3Y, 5Y)."""

    last_date = pd.Timestamp(last_date)
    if period == "YTD":
        return pd.Timestamp(year=last_date.year, month=1, day=1)
    offsets = {"1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3), "6M": pd.DateOffset(months=6),
               "1Y": pd.DateOffset(years=1), "2Y": pd.DateOffset(years=2), "3Y": pd.DateOffset(years=3),
               "5Y": pd.DateOffset(years=5)}
    return last_date - offsets[period]
