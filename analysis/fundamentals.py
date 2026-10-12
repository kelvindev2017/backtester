"""Fundamental confirmation: valuation versus growth, and whether the price
move is backed by earnings.

The score is bullish-positive: cheap relative to growth and to the stock's
own P/E history, with improving revenue, EPS and margins, scores high.
"""

import numpy as np
import pandas as pd

from analysis.indicators import clamp
from analysis.scoring import weighted_score


SNAPSHOT_FIELDS = {
    "trailingPE": "Trailing P/E (Yahoo)",
    "forwardPE": "Forward P/E",
    "trailingPegRatio": "PEG (Yahoo, trailing)",
    "pegRatio": "PEG (Yahoo)",
    "earningsGrowth": "Earnings growth (YoY, Yahoo)",
    "revenueGrowth": "Revenue growth (YoY, Yahoo)",
    "profitMargins": "Profit margin",
    "priceToBook": "Price / book",
    "debtToEquity": "Debt / equity",
    "sector": "Sector",
    "industry": "Industry",
}


def value_as_of(series, timestamp):
    """Last non-NaN value at or before ``timestamp``."""

    if series is None or series.empty:
        return np.nan
    clean = series.dropna()
    clean = clean.loc[clean.index <= timestamp]
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def ordinal(number):
    """1 -> '1st', 3 -> '3rd', 12 -> '12th'."""

    number = int(number)
    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _finite(value):
    try:
        return value is not None and np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def quarterly_growth(fundamentals_df, column, as_of):
    """Latest quarter versus the same quarter a year earlier."""

    if fundamentals_df is None or fundamentals_df.empty or column not in fundamentals_df.columns:
        return np.nan, None
    series = fundamentals_df[column].dropna()
    series = series.loc[series.index <= as_of]
    if len(series) < 5:
        return np.nan, None
    latest_date = series.index[-1]
    year_ago = series.loc[
        (series.index >= latest_date - pd.Timedelta(days=400))
        & (series.index <= latest_date - pd.Timedelta(days=330))
    ]
    if year_ago.empty or year_ago.iloc[-1] <= 0:
        return np.nan, latest_date
    return float(series.iloc[-1] / year_ago.iloc[-1] - 1), latest_date


def ttm_margin_change(fundamentals_df, as_of):
    """Change in TTM net margin versus four quarters earlier."""

    if fundamentals_df is None or fundamentals_df.empty:
        return np.nan, np.nan
    if not {"Revenue", "Net_Income"}.issubset(fundamentals_df.columns):
        return np.nan, np.nan
    frame = fundamentals_df.loc[fundamentals_df.index <= as_of, ["Revenue", "Net_Income"]].dropna()
    if len(frame) < 8:
        return np.nan, np.nan
    ttm = frame.rolling(4).sum()
    margin = ttm["Net_Income"] / ttm["Revenue"].replace(0, np.nan)
    return float(margin.iloc[-1]), float(margin.iloc[-1] - margin.iloc[-5])


def assess_fundamentals(as_of, valuation_df, fundamentals_df=None, snapshot=None, rerating_bars=126):
    """Return ``(score, details, metrics)`` for the stock as of ``as_of``.

    ``fundamentals_df`` must be indexed by the date each quarter became public
    (its earnings announcement), not the fiscal quarter end, or a historical
    analysis would see results before they were released.

    ``snapshot`` holds point-in-time fields from Yahoo (forward P/E, PEG ...).
    Pass ``None`` when analysing a historical date, because those fields
    describe *today* and would leak future information.
    """

    as_of = pd.Timestamp(as_of)
    details, metrics, scored = [], {}, []
    snapshot = snapshot or {}

    valuation = valuation_df.loc[valuation_df.index <= as_of] if valuation_df is not None else pd.DataFrame()
    if not valuation.empty:
        as_of = min(as_of, pd.Timestamp(valuation.index.max()))
    trailing_pe = value_as_of(valuation.get("Trailing_PE"), as_of) if not valuation.empty else np.nan
    ttm_eps = value_as_of(valuation.get("TTM_Reported_EPS"), as_of) if not valuation.empty else np.nan
    metrics["Trailing P/E"] = trailing_pe
    metrics["TTM EPS"] = ttm_eps

    if _finite(ttm_eps) and ttm_eps <= 0:
        details.append(f"TTM EPS is {ttm_eps:.2f} (loss-making) - P/E and PEG are not meaningful.")
        scored.append((-0.5, 0.2))

    # P/E versus the stock's own history.
    if _finite(trailing_pe) and not valuation.empty:
        history = valuation["Trailing_PE"].dropna()
        if len(history) >= 120:
            percentile = float((history < trailing_pe).mean())
            metrics["P/E percentile vs own history"] = percentile
            scored.append((clamp((0.5 - percentile) * 2.0), 0.20))
            details.append(
                f"Trailing P/E {trailing_pe:.1f}x sits at the {ordinal(round(percentile * 100))} percentile of its "
                f"{history.index.min():%Y}-{history.index.max():%Y} range (median {history.median():.1f}x)."
            )

    # EPS growth from the TTM series.
    eps_growth = np.nan
    if not valuation.empty:
        eps_year_ago = value_as_of(valuation["TTM_Reported_EPS"], as_of - pd.Timedelta(days=365))
        if _finite(eps_year_ago) and eps_year_ago > 0 and _finite(ttm_eps):
            eps_growth = ttm_eps / eps_year_ago - 1
            metrics["TTM EPS growth (YoY)"] = eps_growth
            scored.append((clamp(eps_growth / 0.25), 0.20))
            details.append(f"TTM EPS grew {eps_growth * 100:+.1f}% year over year.")

    # PEG: prefer Yahoo's forward-looking figure when it is point-in-time valid.
    peg = np.nan
    peg_source = ""
    for key in ["trailingPegRatio", "pegRatio"]:
        if _finite(snapshot.get(key)):
            peg, peg_source = float(snapshot[key]), "Yahoo"
            break
    if not _finite(peg) and _finite(trailing_pe) and _finite(eps_growth) and eps_growth > 0:
        peg, peg_source = trailing_pe / (eps_growth * 100.0), "computed from TTM EPS growth"
    if _finite(peg):
        metrics["PEG"] = peg
        scored.append((clamp((1.5 - peg) / 1.0) if peg > 0 else -0.5, 0.20))
        verdict = "growth looks cheap" if 0 < peg < 1 else "fairly priced for growth" if 0 < peg <= 2 else "expensive for its growth"
        details.append(f"PEG {peg:.2f} ({peg_source}) - {verdict}.")

    forward_pe = snapshot.get("forwardPE")
    if _finite(forward_pe) and forward_pe > 0:
        metrics["Forward P/E"] = float(forward_pe)
        reference_pe = trailing_pe if _finite(trailing_pe) else snapshot.get("trailingPE")
        if _finite(reference_pe) and reference_pe > 0:
            ratio = reference_pe / forward_pe
            scored.append((clamp((ratio - 1.0) * 2.5), 0.10))
            details.append(
                f"Forward P/E {forward_pe:.1f}x vs trailing {reference_pe:.1f}x - analysts expect earnings to "
                f"{'grow' if ratio > 1 else 'shrink'}."
            )

    revenue_growth, revenue_date = quarterly_growth(fundamentals_df, "Revenue", as_of)
    if _finite(revenue_growth):
        metrics["Revenue growth (latest quarter YoY)"] = revenue_growth
        scored.append((clamp(revenue_growth / 0.15), 0.15))
        details.append(f"Revenue for the quarter ending {revenue_date:%Y-%m-%d} grew {revenue_growth * 100:+.1f}% YoY.")

    margin, margin_change = ttm_margin_change(fundamentals_df, as_of)
    if _finite(margin_change):
        metrics["TTM net margin"] = margin
        metrics["TTM net margin change (YoY)"] = margin_change
        scored.append((clamp(margin_change / 0.03), 0.10))
        details.append(f"TTM net margin {margin * 100:.1f}% ({margin_change * 100:+.1f} pts YoY).")

    # Is the recent price move backed by earnings, or by a re-rating of the multiple?
    if not valuation.empty and len(valuation) > rerating_bars:
        start = valuation.iloc[-rerating_bars - 1]
        end = valuation.iloc[-1]
        if _finite(start["TTM_Reported_EPS"]) and start["TTM_Reported_EPS"] > 0 and _finite(end["TTM_Reported_EPS"]):
            price_change = end["Close"] / start["Close"] - 1
            eps_change = end["TTM_Reported_EPS"] / start["TTM_Reported_EPS"] - 1
            multiple_change = (1 + price_change) / (1 + eps_change) - 1 if eps_change > -1 else np.nan
            if _finite(multiple_change):
                metrics["Price change (window)"] = price_change
                metrics["P/E re-rating (window)"] = multiple_change
                scored.append((clamp(-multiple_change / 0.30), 0.05))
                driver = "multiple expansion" if multiple_change > 0.1 else "multiple compression" if multiple_change < -0.1 else "earnings"
                details.append(
                    f"Over the last {rerating_bars} bars price moved {price_change * 100:+.1f}% while TTM EPS moved "
                    f"{eps_change * 100:+.1f}%: the move is mainly driven by {driver}."
                )

    for key, label in SNAPSHOT_FIELDS.items():
        if key in snapshot and snapshot[key] not in (None, ""):
            metrics.setdefault(label, snapshot[key])

    score = weighted_score(scored)
    if score is None:
        details.append("No usable fundamental data (ETF, index, non-US listing or data outage).")
        return None, details, metrics
    return score, details, metrics
