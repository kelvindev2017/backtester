"""What happened next: how a stock did after past dates when its trailing P/E sat
at a similar point of its own history.

Each observation date t uses only data known at t: trailing P/E (split-adjusted
close / last 4 reported quarters' EPS) and where it ranks among the P/E of every
trading day in the look-back window before t. The outcome is the total return
(with dividends) over the look-ahead after t, minus the benchmark's.

With daily or weekly checks and a long look-ahead, neighbouring observations
share most of their future window, so they are not independent; the table
counts how many of them do not overlap.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.valuation_chart import add_close_overlay, legends_under_panels

CHECK_FREQUENCIES = {"Day": None, "Week": "W-FRI", "Month": "ME"}
LOOK_AHEADS = {
    "1 week": pd.DateOffset(weeks=1),
    "1 month": pd.DateOffset(months=1),
    "3 months": pd.DateOffset(months=3),
    "6 months": pd.DateOffset(months=6),
    "12 months": pd.DateOffset(months=12),
}
LOOKBACK_CHOICES = [3, 5, 10]
DEFAULT_LOOKBACK_YEARS = 5
DEFAULT_BAND = 10
# Fewer days than this in the look-back window and the percentile is left blank.
MIN_LOOKBACK_SHARE = 0.4
GROWTH_BUCKETS = [(-np.inf, 0, "falling"), (0, 50, "growing up to 50%"), (50, np.inf, "growing over 50%")]
OTHER_COLOR = "#a3a3a3"
SIMILAR_COLOR = "#2563eb"
INK = "#1f2937"
MUTED = "#6b7280"
CLOSE_OVERLAY_COLOR = "rgba(124, 58, 237, 0.5)"  # violet: the P/E panel already uses gray for the pre-window line
LOOKBACK_FILL = "rgba(37, 99, 235, 0.06)"
AHEAD_FILL = "rgba(245, 158, 11, 0.10)"


def check_dates(index, frequency):
    """The last trading day of each day, week or month in ``index`` (the last day is always included)."""

    index = pd.DatetimeIndex(index).sort_values()
    if len(index) == 0:
        return index
    rule = CHECK_FREQUENCIES[frequency]
    if rule is None:
        return index
    marker = pd.Series(index, index=index)
    dates = pd.DatetimeIndex(marker.resample(rule).last().dropna())
    return dates.union(index[-1:])


def growth_bucket(growth):
    if growth is None or not np.isfinite(growth):
        return None
    return next(label for low, high, label in GROWTH_BUCKETS if low <= growth < high)


def observations(valuation, total_return, benchmark, frequency="Month", look_ahead="12 months",
                 lookback_years=DEFAULT_LOOKBACK_YEARS, start=None):
    """One row per check date: P/E, its percentile in the look-back, EPS growth and what came next.

    ``valuation`` holds daily ``Trailing_PE`` and ``TTM_Reported_EPS`` (from ``build_valuation_data``);
    ``total_return`` and ``benchmark`` are dividend-adjusted closes. Rows whose look-ahead has not
    finished yet keep NaN outcomes; the last row is today.
    """

    columns = ["pe", "percentile", "eps_growth", "end", "return", "benchmark_return", "excess"]
    pe = pd.to_numeric(valuation["Trailing_PE"], errors="coerce")
    eps = pd.to_numeric(valuation["TTM_Reported_EPS"], errors="coerce")
    total_return = total_return.dropna()
    benchmark = benchmark.dropna()
    if pe.dropna().empty or total_return.empty:
        return pd.DataFrame(columns=columns)
    window = pd.Timedelta(days=round(365.25 * lookback_years))
    horizon = LOOK_AHEADS[look_ahead]
    dates = check_dates(pe.index, frequency)
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    values = pe.to_numpy()
    positions = pe.index.searchsorted(dates, side="right")
    first = pe.index.searchsorted(dates - window, side="left")
    min_days = MIN_LOOKBACK_SHARE * 252 * lookback_years
    last_day = min(total_return.index[-1], benchmark.index[-1]) if len(benchmark) else total_return.index[-1]
    percentiles = []
    for lo, hi in zip(first, positions):
        current, history = values[hi - 1], values[lo:hi]
        history = history[np.isfinite(history)]
        percentiles.append((history < current).mean() * 100 if np.isfinite(current) and len(history) >= min_days else np.nan)
    year_ago = eps.asof(dates - pd.Timedelta(days=365)).to_numpy()
    eps_now = eps.reindex(dates).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = np.where(year_ago > 0, (eps_now / year_ago - 1) * 100, np.nan)
    ends = pd.DatetimeIndex([when + horizon for when in dates])
    finished = ends <= last_day
    outcome = np.where(finished, (total_return.asof(ends).to_numpy() / total_return.asof(dates).to_numpy() - 1) * 100, np.nan)
    market = (benchmark.asof(ends).to_numpy() / benchmark.asof(dates).to_numpy() - 1) * 100 if len(benchmark) else np.nan
    market = np.where(finished, market, np.nan)
    rows = {"pe": values[positions - 1], "percentile": percentiles, "eps_growth": growth, "end": ends,
            "return": outcome, "benchmark_return": market, "excess": outcome - market}
    return pd.DataFrame(rows, index=dates, columns=columns)


def non_overlapping_count(dates, look_ahead):
    """How many of ``dates`` remain when each one must start after the previous one's look-ahead ended."""

    count, free_from = 0, None
    for when in sorted(dates):
        if free_from is None or when >= free_from:
            count += 1
            free_from = when + LOOK_AHEADS[look_ahead]
    return count


def similar_mask(obs, today, band=DEFAULT_BAND):
    """Past observations whose P/E percentile is within ``band`` of today's."""

    return (obs["percentile"] - today["percentile"]).abs() <= band


def summary_table(obs, look_ahead, band=DEFAULT_BAND):
    """Rows: every past check, those with a P/E percentile near today's, and those also in today's growth bucket."""

    today = obs.iloc[-1]
    past = obs.iloc[:-1].dropna(subset=["return", "percentile"])
    similar = past[similar_mask(past, today, band)]
    rows = [("Every past check", past), (f"P/E percentile within {band} of today's ({today['percentile']:.0f})", similar)]
    bucket = growth_bucket(today["eps_growth"])
    if bucket:
        rows.append((f"…and EPS {bucket}, like today", similar[similar["eps_growth"].map(growth_bucket) == bucket]))
    table = []
    for name, group in rows:
        has = len(group) > 0
        table.append({
            "Past checks like today": name,
            "Checks": len(group),
            "Non-overlapping": non_overlapping_count(group.index, look_ahead),
            "Median return %": group["return"].median() if has else np.nan,
            "Median minus benchmark, pts": group["excess"].median() if has else np.nan,
            "Beat benchmark %": (group["excess"] > 0).mean() * 100 if has else np.nan,
            "Worst %": group["return"].min() if has else np.nan,
            "Best %": group["return"].max() if has else np.nan,
        })
    return pd.DataFrame(table)


def today_text(obs, lookback_years):
    today = obs.iloc[-1]
    if not np.isfinite(today["percentile"]):
        return f"Not enough P/E history for a {lookback_years}-year comparison yet."
    growth = f", trailing EPS growth {today['eps_growth']:+.0f}% year on year" if np.isfinite(today["eps_growth"]) else ""
    return (f"Today ({obs.index[-1]:%d %b %Y}): trailing P/E {today['pe']:.1f}x, P/E percentile {today['percentile']:.0f} "
            f"(that share of trading days in the past {lookback_years} years had a lower P/E){growth}.")


def _hover(frame):
    return np.c_[frame.index.strftime("%d %b %Y"), frame["pe"], frame["eps_growth"]]


def scatter_figure(obs, ticker, look_ahead, benchmark_name, band=DEFAULT_BAND, lookback_years=DEFAULT_LOOKBACK_YEARS,
                   example=None, height=None):
    """Each past check: P/E percentile (x) against the return over the look-ahead minus the benchmark's (y)."""

    today = obs.iloc[-1]
    past = obs.iloc[:-1].dropna(subset=["excess", "percentile"])
    mask = similar_mask(past, today, band)
    hover = ("%{customdata[0]}<br>P/E %{customdata[1]:.1f}x, percentile %{x:.0f}<br>EPS growth %{customdata[2]:+.0f}%"
             f"<br>Next {look_ahead}: minus {benchmark_name} " + "%{y:+.1f} pts<extra></extra>")
    size = 5 if len(past) > 600 else 8
    fig = go.Figure()
    for frame, name, color, opacity in [(past[~mask], "Other past checks", OTHER_COLOR, 0.5),
                                        (past[mask], "Checks with a similar P/E percentile", SIMILAR_COLOR, 0.9)]:
        fig.add_trace(go.Scatter(x=frame["percentile"], y=frame["excess"], mode="markers", name=name,
                                 marker=dict(size=size, color=color, opacity=opacity, line=dict(width=0.5, color="white")),
                                 customdata=_hover(frame), hovertemplate=hover))
    if np.isfinite(today["percentile"]):
        fig.add_vrect(x0=max(0, today["percentile"] - band), x1=min(100, today["percentile"] + band),
                      fillcolor=SIMILAR_COLOR, opacity=0.07, line_width=0)
        fig.add_vline(x=today["percentile"], line=dict(color=INK, width=2, dash="dash"))
        fig.add_annotation(x=today["percentile"], y=1, yref="paper", text=f"Today: {today['pe']:.0f}x, percentile "
                           f"{today['percentile']:.0f}", showarrow=False, xanchor="left", xshift=6, yanchor="top")
    if example is not None and example in past.index:
        point = past.loc[example]
        fig.add_annotation(x=point["percentile"], y=point["excess"], text=f"{example:%d %b %Y}, explained below",
                           showarrow=True, arrowhead=0, ax=60, ay=-30)
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(
        title=f"{ticker}: what happened in the {look_ahead} after each check",
        height=height or 480, template="plotly_white", font=dict(color=INK),
        xaxis=dict(title=f"P/E percentile: % of days in the prior {lookback_years} years with a lower P/E (0 = cheapest)",
                   range=[-2, 102]),
        yaxis_title=f"Next {look_ahead}: return minus {benchmark_name}, pts",
        legend=dict(orientation="h", y=-0.25), margin=dict(t=50, l=60, r=20, b=90),
    )
    return fig


def walkthrough_figure(valuation, total_return, benchmark, obs, example, ticker, look_ahead, benchmark_name,
                       lookback_years=DEFAULT_LOOKBACK_YEARS, height=None):
    """How one dot is calculated: close, trailing EPS, P/E in its look-back window, then the look-ahead return."""

    row = obs.loc[example]
    window_start = example - pd.Timedelta(days=round(365.25 * lookback_years))
    end_day = total_return.loc[:row["end"]].index[-1]
    pad = pd.Timedelta(days=60)
    view = slice(window_start - pad, min(valuation.index[-1], end_day + pad))
    close = valuation["Close"].loc[view]
    eps = valuation["TTM_Reported_EPS"].loc[view]
    pe = valuation["Trailing_PE"].loc[view]
    history = valuation["Trailing_PE"].loc[window_start:example].dropna()
    below = int((history < row["pe"]).sum())
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=[
        "1. Daily close, split-adjusted",
        "2. Trailing EPS: sum of the last 4 reported quarters",
        f"3. Trailing P/E = close ÷ trailing EPS; the shaded {lookback_years} years set the percentile",
        f"4. Growth of 100 over the next {look_ahead}, with dividends",
    ])
    fig.add_trace(go.Scatter(x=close.index, y=close, name="Close", line=dict(color=INK, width=1.5), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=eps.index, y=eps, name="Trailing EPS", line=dict(color=INK, width=2, shape="hv"),
                             showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=pe.index, y=pe, name="Trailing P/E", line=dict(color=OTHER_COLOR, width=1.5),
                             showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=history.index, y=history, name="P/E in the look-back", line=dict(color=INK, width=1.5),
                             showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=[window_start, example], y=[row["pe"], row["pe"]], mode="lines", hoverinfo="skip",
                             line=dict(color=SIMILAR_COLOR, width=2, dash="dot"), showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=[example], y=[row["pe"]], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(size=11, color=SIMILAR_COLOR, line=dict(width=2, color="white"))), 3, 1)
    fig.add_annotation(x=window_start, y=row["pe"], xanchor="left", yanchor="top", yshift=-4, showarrow=False, row=3, col=1,
                       text=f"{row['pe']:.1f}x on {example:%d %b %Y}: {below:,} of {len(history):,} days were lower "
                            f"= percentile {row['percentile']:.0f} (the dot's x)")
    stock = total_return.loc[example:end_day]
    market = benchmark.loc[example:end_day]
    stock = stock / stock.iloc[0] * 100
    market = market / market.iloc[0] * 100
    fig.add_trace(go.Scatter(x=stock.index, y=stock, name=ticker, line=dict(color=SIMILAR_COLOR, width=2)), 4, 1)
    fig.add_trace(go.Scatter(x=market.index, y=market, name=benchmark_name, line=dict(color=OTHER_COLOR, width=2)), 4, 1)
    # Nudge the end labels apart (the higher one up) so they never sit on top of each other.
    stock_above = stock.iloc[-1] >= market.iloc[-1]
    for series, name, color, up in [(stock, ticker, INK, stock_above), (market, benchmark_name, MUTED, not stock_above)]:
        fig.add_annotation(x=series.index[-1], y=series.iloc[-1], text=f"{name} {series.iloc[-1] - 100:+.1f}%",
                           showarrow=False, xanchor="left", xshift=6, yshift=8 if up else -8, font=dict(color=color),
                           row=4, col=1)
    fig.add_annotation(x=example, y=1, yref="y4 domain", xanchor="right", yanchor="top", xshift=-8, showarrow=False,
                       text=f"{row['return']:+.1f}% minus {row['benchmark_return']:+.1f}% = {row['excess']:+.1f} pts "
                            "(the dot's y)", row=4, col=1)
    for panel in range(1, 5):
        fig.add_vrect(x0=window_start, x1=example, fillcolor=LOOKBACK_FILL, line_width=0, row=panel, col=1)
        fig.add_vrect(x0=example, x1=end_day, fillcolor=AHEAD_FILL, line_width=0, row=panel, col=1)
    fig.add_vline(x=example, line=dict(color=INK, width=1.5, dash="dash"))
    # The close on a right-hand axis in the EPS and P/E panels (panel 1 is the close, panel 4 is its return).
    for panel in (2, 3):
        add_close_overlay(fig, close, panel, color=CLOSE_OVERLAY_COLOR)
    fig.add_annotation(x=window_start, y=1, yref="y domain", text="Look-back: only data known by then", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(color=MUTED, size=11), row=1, col=1)
    fig.add_annotation(x=example, y=1, yref="y domain", text=f"next {look_ahead}", showarrow=False, xanchor="left",
                       yanchor="top", xshift=6, font=dict(color=MUTED, size=11), row=1, col=1)
    height = height or 1000
    fig.update_layout(title=f"How one dot is calculated: {ticker}, {example:%d %b %Y}", height=height,
                      template="plotly_white", font=dict(color=INK), margin=dict(t=80, l=60, r=100, b=60))
    legends_under_panels(fig, 4, height)
    return fig
