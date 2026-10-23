"""Revenue and valuation (P/E, PEG) history for one company, drawn as two stacked
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
PEG_COLOR = "seagreen"
PEG_AXIS_CAP = 4
EPS_COLOR = "darkorange"
EPS_ESTIMATE_COLOR = "rgba(255, 140, 0, 0.35)"


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


def trailing_peg(valuation):
    """Daily trailing PEG: trailing P/E divided by TTM EPS growth (%) versus a year earlier.

    ``valuation`` holds ``Trailing_PE`` and ``TTM_Reported_EPS`` by day. PEG is left
    empty while EPS growth is zero or negative, where the ratio has no meaning.
    """

    if valuation is None or valuation.empty or not {"Trailing_PE", "TTM_Reported_EPS"}.issubset(valuation.columns):
        return pd.Series(dtype=float)
    eps = valuation["TTM_Reported_EPS"]
    clean = eps.dropna()
    if clean.empty:
        return pd.Series(np.nan, index=valuation.index)
    year_ago = clean.reindex(valuation.index - pd.Timedelta(days=365), method="ffill")
    year_ago.index = valuation.index
    growth = (eps / year_ago.where(year_ago > 0) - 1) * 100
    return (valuation["Trailing_PE"] / growth.where(growth > 0)).replace([np.inf, -np.inf], np.nan)


def peg_summary_text(peg, forward_pe=None, forward_growth=None):
    """One caption line: today's trailing PEG against its own history, plus a forward PEG when estimates exist."""

    series = peg.dropna() if peg is not None else pd.Series(dtype=float)
    parts = []
    if not series.empty:
        current = float(series.iloc[-1])
        parts.append(
            f"Trailing PEG {current:.2f} (P/E / last year's EPS growth %), {ordinal(round((series < current).mean() * 100))} "
            f"percentile since {series.index.min():%Y} (median {series.median():.2f}, low {series.min():.2f} in "
            f"{series.idxmin():%b %Y})."
        )
    else:
        parts.append("No trailing PEG (EPS growth was zero or negative).")
    if all(value is not None and np.isfinite(value) and value > 0 for value in [forward_pe, forward_growth]):
        parts.append(
            f"Forward PEG {forward_pe / (forward_growth * 100):.2f} (forward P/E {forward_pe:.1f}x / next fiscal year's "
            f"expected EPS growth {forward_growth * 100:.0f}%)."
        )
    parts.append("Below 1 reads as cheap for the growth, but only if that growth lasts.")
    return " ".join(parts)


def quarterly_eps(earnings, quarter_ends, max_lag_days=95, fallback_lag_days=30):
    """Reported EPS per quarter, moved from its announcement date to the fiscal quarter end it reports.

    ``earnings`` is indexed by announcement date with a ``Reported EPS`` column.
    Each announcement is matched to the latest of ``quarter_ends`` (the revenue
    quarters) within ``max_lag_days`` before it, so EPS and revenue bars line up.
    An announcement with no matching revenue quarter (the revenue source skipped
    it) is placed ``fallback_lag_days`` before its date. ``max_lag_days`` stays
    near one quarter so a skipped quarter's EPS doesn't land on the one before it.
    """

    if earnings is None or earnings.empty or "Reported EPS" not in earnings.columns:
        return pd.Series(dtype=float)
    ends = pd.DatetimeIndex(sorted(quarter_ends))
    values = {}
    for announced, eps in pd.to_numeric(earnings["Reported EPS"], errors="coerce").dropna().items():
        candidates = ends[(ends < announced) & (ends >= announced - pd.Timedelta(days=max_lag_days))]
        quarter = candidates[-1] if len(candidates) else announced.normalize() - pd.Timedelta(days=fallback_lag_days)
        values[quarter] = float(eps)
    return pd.Series(values, dtype=float).sort_index()


def eps_estimate_quarters(estimates, last_quarter_end):
    """Analyst average EPS for the current and next fiscal quarter, placed like :func:`revenue_estimate_quarters`."""

    if estimates is None or getattr(estimates, "empty", True) or "avg" not in estimates.columns:
        return pd.Series(dtype=float)
    if last_quarter_end is None or pd.isna(last_quarter_end):
        return pd.Series(dtype=float)
    values = {}
    for step, period in [(1, "0q"), (2, "+1q")]:
        if period in estimates.index:
            value = pd.to_numeric(estimates.at[period, "avg"], errors="coerce")
            if pd.notna(value):
                values[pd.Timestamp(last_quarter_end) + pd.DateOffset(months=3 * step)] = float(value)
    return pd.Series(values, dtype=float)


def revenue_pe_figure(revenue, trailing_pe, revenue_estimates=None, peg=None, eps=None, eps_estimates=None, height=None):
    """Trailing P/E (and PEG, when given) above quarterly revenue (and EPS) bars, sharing one date axis.

    ``revenue`` is indexed by fiscal quarter end, in reporting currency units.
    ``revenue_estimates`` (optional) holds analyst averages for coming quarters
    and is drawn as faded bars, labelled as estimates, so it never passes for
    reported revenue.
    """

    peg = peg.dropna() if peg is not None else pd.Series(dtype=float)
    titles = ["Trailing P/E (price / last 4 quarters' reported EPS)"]
    if not peg.empty:
        capped = peg.max() > PEG_AXIS_CAP
        titles.append("Trailing PEG (P/E / EPS growth %, blank when growth <= 0"
                      + (f", axis capped at {PEG_AXIS_CAP})" if capped else ")"))
    titles.append("Revenue by quarter (billions, reporting currency)")
    revenue_row = len(titles)
    eps = eps.dropna() if eps is not None else pd.Series(dtype=float)
    if not eps.empty:
        titles.append("EPS by quarter (Yahoo reported, adjusted, per share)")
    rows = len(titles)
    weights = {"Trailing P/E": 0.3, "Trailing PEG": 0.22, "Revenue": 0.3, "EPS": 0.22}
    heights = [next(w for key, w in weights.items() if title.startswith(key)) for title in titles]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[h / sum(heights) for h in heights], subplot_titles=titles)
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

    if not peg.empty:
        # Gaps (growth <= 0) stay gaps: connecting them would invent PEG values.
        full = peg.reindex(pd.date_range(peg.index.min(), peg.index.max(), freq="B"))
        fig.add_trace(go.Scatter(x=full.index, y=full, mode="lines", name="Trailing PEG", connectgaps=False,
                                 line=dict(color=PEG_COLOR, width=2),
                                 hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>"), row=2, col=1)
        fig.add_hline(y=1, line=dict(color="gray", width=1, dash="dash"), row=2, col=1,
                      annotation_text="PEG 1", annotation_position="top left")
        # Right after earnings turn up, growth is tiny and PEG spikes; capping keeps the usual range readable.
        fig.update_yaxes(range=[0, min(PEG_AXIS_CAP, float(peg.max()) * 1.1)], row=2, col=1)

    actual = revenue.dropna() / 1e9 if revenue is not None else pd.Series(dtype=float)
    fig.add_trace(go.Bar(x=actual.index, y=actual, name="Revenue (reported)", marker_color=ACTUAL_COLOR,
                         hovertemplate="Quarter to %{x|%Y-%m-%d}: %{y:,.2f}B<extra></extra>"), row=revenue_row, col=1)
    estimated = revenue_estimates.dropna() / 1e9 if revenue_estimates is not None else pd.Series(dtype=float)
    if not estimated.empty:
        fig.add_trace(go.Bar(x=estimated.index, y=estimated, name="Revenue (analyst estimate)",
                             marker=dict(color=ESTIMATE_COLOR, line=dict(color=ACTUAL_COLOR, width=1),
                                         pattern=dict(shape="/", fgcolor=ACTUAL_COLOR, size=6)),
                             hovertemplate="Quarter to %{x|%Y-%m-%d} (estimate): %{y:,.2f}B<extra></extra>"),
                      row=revenue_row, col=1)

    if not eps.empty:
        eps_row = revenue_row + 1
        fig.add_trace(go.Bar(x=eps.index, y=eps, name="EPS (reported)", marker_color=EPS_COLOR,
                             hovertemplate="Quarter to %{x|%Y-%m-%d}: %{y:.2f}<extra></extra>"), row=eps_row, col=1)
        eps_ahead = eps_estimates.dropna() if eps_estimates is not None else pd.Series(dtype=float)
        if not eps_ahead.empty:
            fig.add_trace(go.Bar(x=eps_ahead.index, y=eps_ahead, name="EPS (analyst estimate)",
                                 marker=dict(color=EPS_ESTIMATE_COLOR, line=dict(color=EPS_COLOR, width=1),
                                             pattern=dict(shape="/", fgcolor=EPS_COLOR, size=6)),
                                 hovertemplate="Quarter to %{x|%Y-%m-%d} (estimate): %{y:.2f}<extra></extra>"),
                          row=eps_row, col=1)
        fig.update_yaxes(tickformat=".2f", row=eps_row, col=1)

    if not estimated.empty and not actual.empty:
        # Added last: add_vline only reaches the panels that already hold traces.
        fig.add_vline(x=actual.index[-1] + pd.Timedelta(days=45), line=dict(color="gray", width=1, dash="dot"))

    fig.update_yaxes(ticksuffix="x", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(tickformat=",.0f", rangemode="tozero", row=revenue_row, col=1)
    fig.update_layout(height=height or {2: 480, 3: 620}.get(rows, 760), margin=dict(l=10, r=10, t=60, b=10),
                      bargap=0.15, legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0))
    return fig
