"""Charts for the Fair Value tab: revenue and growth, costs as a share of revenue, and price against value."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.fair_value import COST_LINES, yoy_growth

REVENUE_COLOR = "#4C78A8"
GROWTH_COLOR = "#F58518"
LINE_COLORS = {"Cost of sales": "#9D755D", "R&D": "#54A24B", "SG&A": "#B279A2",
               "Gross margin": "#4C78A8", "Operating margin": "#E45756", "Net margin": "#72B7B2"}
SCENARIO_COLORS = {"Bear": "#D62728", "Base": "#7F7F7F", "Bull": "#2CA02C", "Risk-adjusted": "#1F77B4"}


def _layout(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0))
    return fig


def income_figure(frame, scale=1e6):
    """Quarterly revenue (bars, millions) with its year-on-year growth (line, right axis)."""

    revenue = frame["Revenue"].dropna()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=revenue.index, y=revenue / scale, name="Revenue (M)", marker_color=REVENUE_COLOR,
                         hovertemplate="%{y:,.0f}M"), secondary_y=False)
    growth = yoy_growth(revenue)
    if not growth.empty:
        fig.add_trace(go.Scatter(x=growth.index, y=growth, name="YoY % (right)", mode="lines+markers",
                                 line=dict(color=GROWTH_COLOR), hovertemplate="%{y:+.1f}%"), secondary_y=True)
    fig.update_yaxes(title_text=None, secondary_y=False)
    fig.update_yaxes(ticksuffix="%", showgrid=False, secondary_y=True)
    return _layout(fig, 300)


def margin_figure(shares):
    """Cost lines and margins as % of revenue per period."""

    fig = go.Figure()
    for column in COST_LINES + ["Gross margin", "Operating margin", "Net margin"]:
        if column not in shares or shares[column].dropna().empty:
            continue
        series = shares[column].dropna()
        dash = "dot" if column in COST_LINES else "solid"
        fig.add_trace(go.Scatter(x=series.index, y=series, name=column, mode="lines+markers",
                                 line=dict(color=LINE_COLORS[column], dash=dash), hovertemplate="%{y:.1f}%"))
    fig.update_yaxes(ticksuffix="%")
    return _layout(fig, 320)


def value_figure(close, values, risk_adjusted):
    """Daily close against the bear / base / bull values and the risk-adjusted value (flat lines: the values
    change once a quarter, when the inputs do)."""

    close = close.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, name="Close", line=dict(color="#333333", width=1.5),
                             hovertemplate="%{y:,.2f}"))
    if close.empty:
        return _layout(fig, 320)
    span = [close.index[0], close.index[-1]]
    for scenario, value in values.items():
        if value is None or pd.isna(value):
            continue
        fig.add_trace(go.Scatter(x=span, y=[value, value], name=f"{scenario} {value:,.2f}", mode="lines",
                                 line=dict(color=SCENARIO_COLORS.get(scenario, "#999999"), dash="dash", width=1),
                                 hovertemplate=f"{scenario} %{{y:,.2f}}"))
    if risk_adjusted is not None and pd.notna(risk_adjusted):
        fig.add_trace(go.Scatter(x=span, y=[risk_adjusted, risk_adjusted], name=f"Risk-adjusted {risk_adjusted:,.2f}",
                                 mode="lines", line=dict(color=SCENARIO_COLORS["Risk-adjusted"], width=2.5),
                                 hovertemplate="Risk-adjusted %{y:,.2f}"))
    return _layout(fig, 340)
