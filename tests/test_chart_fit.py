import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.chart_fit import fit_y_axes


def dates(n=5):
    return pd.date_range("2026-01-01", periods=n)


def test_every_panel_zooms_dates_only_and_lists_its_axes():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Candlestick(x=dates(), open=[1] * 5, high=[2] * 5, low=[0.5] * 5, close=[1.5] * 5), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates(), y=[1, 2, 3, 4, 5]), row=2, col=1)
    fig.update_yaxes(range=[0, 10], row=2, col=1)
    fig.update_layout(meta=dict(axis_hover=True))

    fit_y_axes(fig)

    assert fig.layout.yaxis.fixedrange and fig.layout.yaxis2.fixedrange
    assert not fig.layout.xaxis.fixedrange
    assert fig.layout.meta == {"axis_hover": True, "fit_y": {"yaxis": {"fit": True}, "yaxis2": {"cap": [0.0, 10.0]}}}


def test_volume_overlay_and_axes_left_alone():
    fig = go.Figure(go.Scatter(x=dates(), y=[1, 2, 3, 4, 5]))
    fig.add_trace(go.Bar(x=dates(), y=[5] * 5, yaxis="y2"))
    fig.add_trace(go.Scatter(x=dates(), y=[0.5] * 5, yaxis="y3"))
    fig.add_trace(go.Bar(x=[1, 2], y=["a", "b"], orientation="h", xaxis="x2", yaxis="y4"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right"), yaxis3=dict(overlaying="y", fixedrange=True),
                      yaxis4=dict(anchor="x2"), xaxis2=dict(anchor="y4"))

    fit_y_axes(fig)

    assert fig.layout.meta["fit_y"] == {"yaxis": {"fit": True}, "yaxis2": {"volume": True}}
    assert not fig.layout.yaxis4.fixedrange


def test_charts_without_dragging_are_untouched():
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
    fig.update_layout(dragmode=False)
    fit_y_axes(fig)
    assert fig.layout.meta is None and not fig.layout.yaxis.fixedrange
