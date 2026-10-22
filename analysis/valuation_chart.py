"""Revenue and valuation (P/E) history for one company, drawn as two stacked
panels on a shared date axis.

The panels are stacked rather than overlaid on twin y-axes on purpose: with
two y-scales the visual crossing of the P/E line and the revenue bars depends
only on how each axis was scaled, and reads as a signal it is not.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.fundamentals import ordinal

VALUATION_HISTORY_YEARS = 8
ACTUAL_COLOR = "royalblue"
ESTIMATE_COLOR = "rgba(65, 105, 225, 0.35)"
PE_COLOR = "goldenrod"


def revenue_estimate_quarters(estimates, last_quarter_end):
    """Analyst average revenue for the current and next fiscal quarter as a Series by quarter end.

    ``estimates`` is Yahoo's ``revenue_estimate`` table (rows ``0q``, ``+1q``, ...,
    column ``avg``). Yahoo gives no dates, so the quarters are placed one and two
    quarters after the last reported quarter end.
    """

    if estimates is None or getattr(estimates, "empty", True) or "avg" not in estimates.columns:
        return pd.Series(dtype=float)
    if last_quarter_end is None or pd.isna(last_quarter_end):
        return pd.Series(dtype=float)
    values = {}
    for step, period in [(1, "0q"), (2, "+1q")]:
        if period in estimates.index:
            value = pd.to_numeric(estimates.at[period, "avg"], errors="coerce")
            if pd.notna(value) and value > 0:
                values[pd.Timestamp(last_quarter_end) + pd.DateOffset(months=3 * step)] = float(value)
    return pd.Series(values, dtype=float)


def pe_summary(trailing_pe):
    """Current trailing P/E against its own history: current, median, percentile, low and its date."""

    series = trailing_pe.dropna() if trailing_pe is not None else pd.Series(dtype=float)
    if series.empty:
        return {}
    current = float(series.iloc[-1])
    return {
        "current": current,
        "median": float(series.median()),
        "percentile": float((series < current).mean()),
        "low": float(series.min()),
        "low_date": series.idxmin(),
        "start": series.index.min(),
    }


def pe_summary_text(summary, forward_pe=None):
    """One caption line for :func:`pe_summary`, plus today's forward P/E when Yahoo has one."""

    if not summary:
        return "No trailing P/E history (no reported EPS, or EPS was negative)."
    text = (
        f"Trailing P/E {summary['current']:.1f}x sits at the {ordinal(round(summary['percentile'] * 100))} percentile "
        f"since {summary['start']:%Y} (median {summary['median']:.1f}x, low {summary['low']:.1f}x in "
        f"{summary['low_date']:%b %Y})."
    )
    if forward_pe is not None and np.isfinite(forward_pe) and forward_pe > 0:
        text += (
            f" Forward P/E today is {forward_pe:.1f}x (Yahoo, on next fiscal year's estimated EPS). "
            "It has no free history, so it is not comparable to the line."
        )
    return text


def revenue_pe_figure(revenue, trailing_pe, revenue_estimates=None, height=480):
    """Trailing P/E line above quarterly revenue bars, sharing one date axis.

    ``revenue`` is indexed by fiscal quarter end, in reporting currency units.
    ``revenue_estimates`` (optional) holds analyst averages for coming quarters
    and is drawn as faded bars, labelled as estimates, so it never passes for
    reported revenue.
    """

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.45, 0.55],
                        subplot_titles=("Trailing P/E (price / last 4 quarters' reported EPS)",
                                        "Revenue by quarter (billions, reporting currency)"))
    pe = trailing_pe.dropna() if trailing_pe is not None else pd.Series(dtype=float)
    if not pe.empty:
        fig.add_trace(go.Scatter(x=pe.index, y=pe, mode="lines", name="Trailing P/E",
                                 line=dict(color=PE_COLOR, width=2),
                                 hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}x<extra></extra>"), row=1, col=1)
        median = float(pe.median())
        fig.add_hline(y=median, line=dict(color="gray", width=1, dash="dash"), row=1, col=1,
                      annotation_text=f"median {median:.1f}x", annotation_position="top left")
    else:
        fig.add_annotation(x=0.5, y=0.5, xref="x domain", yref="y domain", showarrow=False,
                           text="No trailing P/E (no reported EPS, or EPS was negative)", font=dict(color="gray"))

    actual = revenue.dropna() / 1e9 if revenue is not None else pd.Series(dtype=float)
    fig.add_trace(go.Bar(x=actual.index, y=actual, name="Revenue (reported)", marker_color=ACTUAL_COLOR,
                         hovertemplate="Quarter to %{x|%Y-%m-%d}: %{y:,.2f}B<extra></extra>"), row=2, col=1)
    estimated = revenue_estimates.dropna() / 1e9 if revenue_estimates is not None else pd.Series(dtype=float)
    if not estimated.empty:
        fig.add_trace(go.Bar(x=estimated.index, y=estimated, name="Revenue (analyst estimate)",
                             marker=dict(color=ESTIMATE_COLOR, line=dict(color=ACTUAL_COLOR, width=1),
                                         pattern=dict(shape="/", fgcolor=ACTUAL_COLOR, size=6)),
                             hovertemplate="Quarter to %{x|%Y-%m-%d} (estimate): %{y:,.2f}B<extra></extra>"),
                      row=2, col=1)
        if not actual.empty:
            fig.add_vline(x=actual.index[-1] + pd.Timedelta(days=45), line=dict(color="gray", width=1, dash="dot"))

    fig.update_yaxes(ticksuffix="x", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(tickformat=",.0f", rangemode="tozero", row=2, col=1)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=60, b=10), bargap=0.15,
                      legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0))
    return fig
