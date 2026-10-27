"""The US economy and other asset classes next to the S&P 500.

Two stacked-panel charts on a shared date axis, each panel with its own legend under it:

- Assets: the S&P 500 against Treasuries, gold, oil and bitcoin, rebased to 100 on the first day all of
  them have a price, then the drawdown from each one's peak and its rolling correlation with the S&P 500.
- Macro: inflation, unemployment, the Fed funds rate and other FRED series, one per panel, each with the
  S&P 500 on a right-hand log axis and US recessions shaded.

Macro levels are context, not timing signals: see analysis/macro-signals-2026-10-27.md in the project files.
"""

import io
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.valuation_chart import add_close_overlay, legends_under_panels

BENCHMARK = "S&P 500"
# Name -> (Yahoo symbol, line color). ETF prices are dividend-adjusted, so the lines are total returns.
ASSETS = {
    BENCHMARK: ("SPY", "black"),
    "Treasuries 7-10y": ("IEF", "royalblue"),
    "Treasuries 20y+": ("TLT", "lightskyblue"),
    "Gold": ("GLD", "goldenrod"),
    "Oil (Brent)": ("BZ=F", "saddlebrown"),
    "Bitcoin": ("BTC-USD", "mediumorchid"),
}
DEFAULT_ASSETS = [BENCHMARK, "Treasuries 7-10y", "Gold", "Oil (Brent)", "Bitcoin"]
SP500_INDEX = "^GSPC"

# Name -> (FRED series, transform, unit, color). "yoy" turns an index level into a 12-month % change.
INDICATORS = {
    "Inflation (CPI, YoY)": ("CPIAUCSL", "yoy", "%", "indianred"),
    "Unemployment rate": ("UNRATE", "level", "%", "purple"),
    "Fed funds rate": ("FEDFUNDS", "level", "%", "seagreen"),
    "Core inflation (PCE ex food & energy, YoY)": ("PCEPILFE", "yoy", "%", "lightcoral"),
    "10-year real yield (TIPS)": ("DFII10", "level", "%", "teal"),
    "Yield curve (10y minus 3m)": ("T10Y3M", "level", "pt", "steelblue"),
}
DEFAULT_INDICATORS = ["Inflation (CPI, YoY)", "Unemployment rate", "Fed funds rate"]
RECESSION_SERIES = "USREC"
RECESSION_FILL = "rgba(128, 128, 128, 0.18)"
HISTORY_CHOICES = {"1y": 1, "3y": 3, "5y": 5, "10y": 10, "20y": 20, "30y": 30, "Max": None}
CORRELATION_WEEKS = 52


def fred_series(series_id, timeout=30):
    """A FRED series as a float Series by date (missing values dropped), from FRED's public CSV download."""

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        raw = response.read()
    frame = pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True, na_values=".")
    return pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna().astype(float)


def indicator_values(raw, transform):
    """Apply an indicator's transform to its raw FRED values."""

    if transform == "yoy":
        # Monthly index levels: % change from the same month a year earlier.
        return (raw.pct_change(12, fill_method=None) * 100).dropna()
    return raw.dropna()


def window_start(end, years):
    """First date shown for a history of ``years`` (None = everything) ending at ``end``."""

    return None if years is None else pd.Timestamp(end) - pd.DateOffset(years=years)


def rebase(closes, start=None):
    """Rebase each column to 100 on the first day every column has a price (on or after ``start``).

    Assets that trade on weekends (bitcoin) are sampled on the other assets' trading days. Returns the
    rebased frame and the common first day (None if the columns never overlap).
    """

    frame = closes.copy()
    if start is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(start)]
    weekdays = frame.index[frame.index.dayofweek < 5]
    frame = frame.ffill().reindex(weekdays)
    complete = frame.dropna()
    if complete.empty:
        return pd.DataFrame(columns=frame.columns), None
    first = complete.index[0]
    frame = frame.loc[first:]
    return frame / frame.iloc[0] * 100, first


def drawdowns(rebased):
    """Percent below each column's running peak."""

    return (rebased / rebased.cummax() - 1) * 100


def rolling_correlation(rebased, benchmark=BENCHMARK, weeks=CORRELATION_WEEKS):
    """Rolling correlation of weekly returns with ``benchmark`` (weekly, so time zones and holidays matter less)."""

    weekly = rebased.resample("W-FRI").last().pct_change(fill_method=None)
    others = [column for column in weekly.columns if column != benchmark]
    return weekly[others].rolling(weeks, min_periods=weeks // 2).corr(weekly[benchmark]).dropna(how="all")


def asset_table(rebased, benchmark=BENCHMARK):
    """Return, risk and correlation with the benchmark for each rebased column."""

    rows = {}
    days = (rebased.index[-1] - rebased.index[0]).days
    years = days / 365.25
    weekly = rebased.resample("W-FRI").last().pct_change(fill_method=None)
    for column in rebased.columns:
        series = rebased[column].dropna()
        total = series.iloc[-1] / series.iloc[0] - 1
        weekly_returns = weekly[column].dropna()
        rows[column] = {
            "Total return %": total * 100,
            "Per year %": ((1 + total) ** (1 / years) - 1) * 100 if years >= 1 else np.nan,
            "Volatility %/yr": weekly_returns.std() * np.sqrt(52) * 100,
            "Worst fall %": drawdowns(series.to_frame()).iloc[:, 0].min(),
            "Below peak now %": drawdowns(series.to_frame()).iloc[-1, 0],
            f"Correlation with {benchmark}": weekly_returns.corr(weekly[benchmark]) if column != benchmark else 1.0,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def dates_on_every_panel(fig, rows, height):
    """Date tick labels under every panel (not only the bottom one), each panel's legend below them."""

    fig.update_xaxes(showticklabels=True)
    legends_under_panels(fig, rows, height, dates_every_row=True)


def assets_figure(rebased, colors=None, benchmark=BENCHMARK, height=None, gap_px=108):
    """Rebased growth (log), drawdown and rolling correlation with the benchmark, stacked on one date axis.

    ``gap_px`` is the room between panels for a panel's legend and the next panel's title; a narrow
    screen wraps the legend onto more lines and needs more.
    """

    colors = colors or {name: color for name, (_, color) in ASSETS.items()}
    correlations = rolling_correlation(rebased, benchmark) if benchmark in rebased and rebased.shape[1] > 1 else pd.DataFrame()
    panels = [("Growth of 100, log scale (dividends reinvested)", 0.45), ("Below previous peak, %", 0.27)]
    if not correlations.empty:
        panels.append((f"{CORRELATION_WEEKS}-week correlation with the {benchmark}", 0.28))
    rows = len(panels)
    height = height or {2: 640, 3: 900}[rows] + (gap_px - 80) * (rows - 1)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=min(0.18, gap_px / height),
                        row_heights=[weight for _, weight in panels], subplot_titles=[title for title, _ in panels])
    for column in rebased.columns:
        color = colors.get(column, "gray")
        width = 2.2 if column == benchmark else 1.4
        series = rebased[column]
        fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name=column, line=dict(color=color, width=width),
                                 hovertemplate=f"{column} %{{x|%Y-%m-%d}}: %{{y:,.0f}}<extra></extra>"), row=1, col=1)
    for column, series in drawdowns(rebased).items():
        fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name=column,
                                 line=dict(color=colors.get(column, "gray"), width=2 if column == benchmark else 1.2),
                                 hovertemplate=f"{column} %{{x|%Y-%m-%d}}: %{{y:.0f}}%<extra></extra>"), row=2, col=1)
    if not correlations.empty:
        for column, series in correlations.items():
            fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name=column,
                                     line=dict(color=colors.get(column, "gray"), width=1.4),
                                     hovertemplate=f"{column} week to %{{x|%Y-%m-%d}}: %{{y:+.2f}}<extra></extra>"),
                          row=3, col=1)
        fig.add_hline(y=0, line=dict(color="gray", width=1), row=3, col=1)
        fig.update_yaxes(range=[-1, 1], row=3, col=1)
    fig.update_yaxes(type="log", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=60), hovermode="x unified")
    dates_on_every_panel(fig, rows, height)
    return fig


def recession_periods(recession):
    """(start, end) pairs for the months a 0/1 recession series is 1."""

    flags = recession.fillna(0).astype(int)
    periods, start = [], None
    for day, flag in flags.items():
        if flag and start is None:
            start = day
        elif not flag and start is not None:
            periods.append((start, day))
            start = None
    if start is not None:
        periods.append((start, flags.index[-1] + pd.offsets.MonthEnd(1)))
    return periods


def macro_figure(indicators, sp500, recession=None, start=None, height=None, gap_px=108):
    """One panel per indicator, each with the S&P 500 on a right-hand log axis and recessions shaded.

    ``indicators`` maps a name from :data:`INDICATORS` (or any name) to a Series of values. ``gap_px`` is
    as in :func:`assets_figure`.
    """

    indicators = {name: series for name, series in indicators.items() if series is not None and not series.empty}
    if start is not None:
        indicators = {name: series.loc[series.index >= pd.Timestamp(start)] for name, series in indicators.items()}
        sp500 = sp500.loc[sp500.index >= pd.Timestamp(start)]
    indicators = {name: series for name, series in indicators.items() if not series.empty}
    rows = len(indicators)
    if rows == 0:
        return None
    height = height or 220 + (180 + gap_px) * rows
    titles = [f"{name}, {INDICATORS.get(name, (None, None, '%'))[2]}" for name in indicators]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=min(0.18, gap_px / height),
                        subplot_titles=titles)
    for row, (name, series) in enumerate(indicators.items(), start=1):
        color = INDICATORS.get(name, (None, None, None, "royalblue"))[3]
        unit = INDICATORS.get(name, (None, None, "%"))[2]
        suffix = "%" if unit == "%" else ""
        fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name=name, line=dict(color=color, width=2),
                                 hovertemplate=f"%{{x|%Y-%m}}: %{{y:.2f}}{suffix}<extra></extra>"), row=row, col=1)
        if series.min() < 0 < series.max():
            fig.add_hline(y=0, line=dict(color="gray", width=1), row=row, col=1)
        fig.update_yaxes(ticksuffix=suffix, row=row, col=1)
    # Legend ranks follow the panels, before the overlays add their own axes.
    for position, trace in enumerate(fig.data):
        trace.legendrank = (position + 1) * 100
    for row in range(1, rows + 1):
        add_close_overlay(fig, sp500, row, log_price=True, name="S&P 500 (right axis, log)")
        fig.data[-1].legendrank = row * 100 + 1
    if recession is not None and not recession.empty:
        first = min(series.index.min() for series in indicators.values())
        periods = [(begin, end) for begin, end in recession_periods(recession) if end >= first]
        for begin, end in periods:
            fig.add_vrect(x0=max(begin, first), x1=end, fillcolor=RECESSION_FILL, line_width=0, layer="below",
                          row="all", col=1)
        if periods:
            # A legend entry for the shading, in the first panel's legend.
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="US recession (shaded)",
                                     marker=dict(symbol="square", size=12, color=RECESSION_FILL)), row=1, col=1)
            fig.data[-1].legendrank = 102
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=60), hovermode="x unified")
    dates_on_every_panel(fig, rows, height)
    return fig


def latest_text(name, series):
    """'Unemployment rate: 4.3% in Sep 2026 (4.1% a year earlier)'."""

    series = series.dropna()
    if series.empty:
        return f"{name}: no data"
    unit = INDICATORS.get(name, (None, None, "%"))[2]
    suffix = "%" if unit == "%" else " pt"
    last_day = series.index[-1]
    text = f"{name}: {series.iloc[-1]:.2f}{suffix} ({last_day:%b %Y}"
    year_ago = series.loc[:last_day - pd.DateOffset(years=1)]
    if not year_ago.empty:
        text += f"; {year_ago.iloc[-1]:.2f}{suffix} a year earlier"
    return text + ")"
