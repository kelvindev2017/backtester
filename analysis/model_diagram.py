"""Pictures of the valuation model (analysis.valuation_model): its block diagram, the signals of one block and the
model price against the close.

The diagram follows the two sample rates: the quarterly blocks sit in the upper region, the daily blocks in the
lower one, with a hold block on every signal that crosses. Each block shows its latest output, each wire the value
it carries, and an invisible marker on each block makes it clickable.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.valuation_model import BLOCKS, MODELLED, SCENARIOS, SIGNALS, latest_values, scenario_weights

BLOCK_FILL = "rgba(65, 105, 225, 0.12)"
BLOCK_LINE = "royalblue"
PLAIN_FILL = "rgba(128, 128, 128, 0.10)"
INPUT_FILL = "rgba(128, 128, 128, 0.22)"
OUTPUT_FILL = "rgba(46, 139, 87, 0.22)"
OUTPUT_LINE = "seagreen"
WIRE = "rgba(110, 110, 110, 0.9)"
PARKED_WIRE = "rgba(110, 110, 110, 0.6)"
REGION_LINE = "rgba(128, 128, 128, 0.6)"
SCENARIO_COLORS = {"Bear": "indianred", "Base": "royalblue", "Bull": "seagreen"}
AXIS_TITLES = {"amount": "Reporting currency", "eps": "Per share", "price": "Price", "percent": "%", "share": "%",
               "multiple": "P/E (x)", "points": "P/E points", "count": "Items"}

LEFT, RIGHT = 2.8, 7.4          # column centres
HALF_WIDTH, HALF_HEIGHT = 1.9, 0.5
# Block centres (x, y); y grows downwards.
POSITIONS = {
    "documents": (LEFT, 1.2), "income_map": (LEFT, 2.7), "item_tracker": (RIGHT, 2.7),
    "earnings_quality": (LEFT, 4.2), "triage_gate": (RIGHT, 4.2),
    "scenario_engine": (LEFT, 5.7), "item_router": (RIGHT, 5.7),
    "risk_weighting": (LEFT, 7.2),
    "hold_eps": (LEFT, 8.7), "hold_pe": (RIGHT, 8.7),
    "product": (LEFT, 10.4), "multiple_builder": (RIGHT, 10.4),
    "market_data": (RIGHT, 11.9),
    "price": (LEFT, 13.4), "market_check": (RIGHT, 13.4),
}
PLAIN_BLOCKS = {"documents": "Filings, calls, news", "market_data": "Market data", "hold_eps": "Hold",
                "hold_pe": "Hold", "price": "Price per share"}
QUARTERLY_REGION, DAILY_REGION = (0.3, 7.95), (9.55, 14.2)


def format_value(value, unit="", axis=""):
    """A signal value for a label: ``12.3%``, ``17.5x``, ``-6 pts``, ``1.23B`` or ``2.49``."""

    if value is None or not np.isfinite(value):
        return "n/a"
    if unit == "%":
        return f"{value:+.1f}%" if axis == "percent" else f"{value:.1f}%"
    if unit == "x":
        return f"{value:.1f}x"
    if unit == "pts":
        return f"{value:+.1f} pts"
    if axis == "count":
        return f"{value:.0f}"
    if abs(value) >= 1e9:
        return f"{value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    return f"{value:,.2f}"


def signal_text(values, name):
    _, label, unit, axis = SIGNALS[name]
    return format_value(values.get(name, np.nan), unit, axis)


def block_summary(block, values):
    """One line for the block's face: its main output on the chosen day."""

    lines = {
        "income_map": lambda: f"Growth {signal_text(values, 'revenue_yoy')} · margin {signal_text(values, 'net_margin')}",
        "earnings_quality": lambda: f"Clean EPS {signal_text(values, 'clean_eps')}",
        "item_tracker": lambda: f"{signal_text(values, 'items_modelled')} modelled · {signal_text(values, 'items_parked')} parked",
        "triage_gate": lambda: f"{signal_text(values, 'items_modelled')} passed",
        "item_router": lambda: f"EPS {signal_text(values, 'event_effect')} · P/E {signal_text(values, 'pe_adjust')}",
        "scenario_engine": lambda: " / ".join(signal_text(values, f"eps_{name.lower()}") for name in SCENARIOS),
        "risk_weighting": lambda: f"EPS {signal_text(values, 'risk_adjusted_eps')}",
        "multiple_builder": lambda: f"P/E {signal_text(values, 'acceptable_pe')}",
        "product": lambda: f"{signal_text(values, 'held_eps')} x {signal_text(values, 'acceptable_pe')}",
        "market_check": lambda: f"Model vs close {signal_text(values, 'residual_pct')}",
        "price": lambda: signal_text(values, "model_price"),
        "market_data": lambda: f"Close {signal_text(values, 'close')}",
    }
    return lines[block]() if block in lines else ""


def parameter_table(result, block):
    """The parameters one block owns, with the values in use."""

    params = result["params"]
    rows = [{"Parameter": label, "Value": params[key]} for key, label in BLOCKS[block]["params"].items()]
    if block == "risk_weighting":
        rows += [{"Parameter": f"{name} weight, normalised %", "Value": round(float(weight) * 100, 1)}
                 for name, weight in zip(SCENARIOS, scenario_weights(params))]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def block_hover(result, block, values):
    """Hover text of a block: what it reads, what it writes and its parameters."""

    spec = BLOCKS[block]
    lines = [f"<b>{spec['title']}</b> · runs {'each earnings release' if spec['rate'] == 'quarterly' else 'each close'}"]
    for title, names in [("In", spec["inputs"]), ("Out", spec["outputs"])]:
        for name in names:
            lines.append(f"{title}: {SIGNALS[name][1]} {signal_text(values, name)}")
    for row in parameter_table(result, block).to_dict("records"):
        lines.append(f"Param: {row['Parameter']} = {row['Value']:g}")
    return "<br>".join(lines)


def _box(fig, centre, fill, line, width=1.0, pill=False):
    x, y = centre
    fig.add_shape(type="rect", x0=x - HALF_WIDTH, x1=x + HALF_WIDTH, y0=y - (0.36 if pill else HALF_HEIGHT),
                  y1=y + (0.36 if pill else HALF_HEIGHT), fillcolor=fill, line=dict(color=line, width=width),
                  layer="below")


def _wire(fig, points, text=None, dashed=False, label_at=None, size=10):
    """A wire through ``points`` with an arrowhead on its last leg and, optionally, the value it carries."""

    color = PARKED_WIRE if dashed else WIRE
    for (x0, y0), (x1, y1) in zip(points[:-2], points[1:-1]):
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, layer="below",
                      line=dict(color=color, width=1.3, dash="dash" if dashed else "solid"))
    (x0, y0), (x1, y1) = points[-2], points[-1]
    if dashed:
        # An annotation arrow cannot be dashed: draw the leg as a dashed line and keep only a short arrow tip.
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, layer="below",
                      line=dict(color=color, width=1.3, dash="dash"))
        length = max(abs(x1 - x0), abs(y1 - y0)) or 1.0
        x0, y0 = x1 - (x1 - x0) / length * 0.25, y1 - (y1 - y0) / length * 0.25
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", text="",
                       showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.3, arrowcolor=color)
    if text:
        x, y = label_at if label_at is not None else ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(size=size), xanchor="left", xshift=4)


def model_diagram(result, selected=None, on=None, mobile=False):
    """The block diagram with the values of ``on`` (default: the last close) on its blocks and wires.

    ``selected`` is the block drawn with a heavy border. Each block has an invisible marker whose ``customdata`` is
    the block's key, so a click on it can be read back as a selection.
    """

    values = latest_values(result, on)
    fig = go.Figure()
    small = 9 if mobile else 11
    title_size = 10 if mobile else 12

    for (top, bottom), label in [(QUARTERLY_REGION, "<b>Quarterly rate</b> · each earnings release"),
                                 (DAILY_REGION, "<b>Daily rate</b> · each close")]:
        fig.add_shape(type="rect", x0=0.2, x1=9.95, y0=top, y1=bottom, fillcolor="rgba(0,0,0,0)", layer="below",
                      line=dict(color=REGION_LINE, width=1, dash="dash"))
        fig.add_annotation(x=0.35, y=top + 0.22, text=label, showarrow=False, xanchor="left", yanchor="middle",
                           font=dict(size=small))

    def edge(block, side):
        x, y = POSITIONS[block]
        half = 0.36 if block in ("documents", "market_data", "price") else HALF_HEIGHT
        return {"top": (x, y - half), "bottom": (x, y + half), "left": (x - HALF_WIDTH, y),
                "right": (x + HALF_WIDTH, y)}[side]

    def text(name):
        return signal_text(values, name)

    # Quarterly part.
    _wire(fig, [edge("documents", "bottom"), edge("income_map", "top")], "results")
    _wire(fig, [edge("documents", "right"), (RIGHT, POSITIONS["documents"][1]), edge("item_tracker", "top")], "items, stories",
          label_at=(LEFT + HALF_WIDTH + 0.1, POSITIONS["documents"][1] - 0.22))
    _wire(fig, [edge("income_map", "bottom"), edge("earnings_quality", "top")], f"growth {text('revenue_yoy')}")
    _wire(fig, [edge("earnings_quality", "bottom"), edge("scenario_engine", "top")], f"clean EPS {text('clean_eps')}")
    _wire(fig, [edge("scenario_engine", "bottom"), edge("risk_weighting", "top")], "bear / base / bull EPS")
    _wire(fig, [edge("item_tracker", "bottom"), edge("triage_gate", "top")], f"{text('items_modelled')} + {text('items_parked')} items")
    _wire(fig, [edge("triage_gate", "bottom"), edge("item_router", "top")], f"{text('items_modelled')} modelled")
    _wire(fig, [edge("item_router", "left"), edge("scenario_engine", "right")], f"EPS {text('event_effect')}",
          label_at=(LEFT + HALF_WIDTH - 0.05, POSITIONS["item_router"][1] - 0.25))
    _wire(fig, [edge("item_router", "bottom"), edge("hold_pe", "top")], f"P/E {text('pe_adjust')}")
    _wire(fig, [edge("risk_weighting", "bottom"), edge("hold_eps", "top")], f"EPS {text('risk_adjusted_eps')}")
    # Daily part.
    _wire(fig, [edge("hold_eps", "bottom"), edge("product", "top")], f"EPS {text('held_eps')}")
    _wire(fig, [edge("hold_pe", "bottom"), edge("multiple_builder", "top")], f"P/E {text('held_pe_adjust')}")
    _wire(fig, [edge("multiple_builder", "left"), edge("product", "right")], f"P/E {text('acceptable_pe')}",
          label_at=(LEFT + HALF_WIDTH - 0.05, POSITIONS["product"][1] - 0.25))
    _wire(fig, [edge("market_data", "top"), edge("multiple_builder", "bottom")], "peer P/E")
    _wire(fig, [edge("market_data", "bottom"), edge("market_check", "top")], f"close {text('close')}")
    _wire(fig, [edge("product", "bottom"), edge("price", "top")], f"price {text('model_price')}")
    # The model price branches to the market check.
    branch_y = POSITIONS["market_data"][1]
    middle_x = (LEFT + RIGHT) / 2 + 0.15
    fig.add_trace(go.Scatter(x=[LEFT], y=[branch_y + 0.55], mode="markers", marker=dict(size=7, color=WIRE),
                             hoverinfo="skip", showlegend=False))
    _wire(fig, [(LEFT, branch_y + 0.55), (middle_x, branch_y + 0.55), (middle_x, POSITIONS["market_check"][1]),
                edge("market_check", "left")])
    # Parked items reach the market check without touching the price.
    tracker_right = edge("item_tracker", "right")
    _wire(fig, [tracker_right, (9.75, tracker_right[1]), (9.75, POSITIONS["market_check"][1]), edge("market_check", "right")],
          dashed=True)

    for block, centre in POSITIONS.items():
        chosen = block == selected
        if block in BLOCKS:
            plain = block == "product"
            _box(fig, centre, PLAIN_FILL if plain else BLOCK_FILL, "gray" if plain else BLOCK_LINE, 3.0 if chosen else 1.0)
            spec = BLOCKS[block]
            label = f"<b>{spec['title']}</b><br><i>{spec['role']}</i><br>{block_summary(block, values)}"
        else:
            output = block == "price"
            pill = block in ("documents", "market_data", "price")
            _box(fig, centre, OUTPUT_FILL if output else (INPUT_FILL if pill else PLAIN_FILL),
                 OUTPUT_LINE if output else "gray", 1.0, pill=pill)
            summary = block_summary(block, values)
            label = f"<b>{PLAIN_BLOCKS[block]}</b>" + (f"<br>{summary}" if summary else "")
        fig.add_annotation(x=centre[0], y=centre[1], text=label, showarrow=False, font=dict(size=title_size),
                           align="center")

    keys = list(BLOCKS)
    fig.add_trace(go.Scatter(
        x=[POSITIONS[key][0] for key in keys], y=[POSITIONS[key][1] for key in keys], mode="markers",
        marker=dict(size=46 if mobile else 60, color="rgba(0,0,0,0)", symbol="square"),
        customdata=[[key] for key in keys], hovertext=[block_hover(result, key, values) for key in keys],
        hoverinfo="text", showlegend=False, name="blocks"))

    hidden = dict(visible=False, fixedrange=True)
    fig.update_layout(height=760 if mobile else 900, margin=dict(l=5, r=5, t=10, b=10), dragmode=False,
                      clickmode="event+select", xaxis=dict(range=[0, 10.1], **hidden),
                      yaxis=dict(range=[14.4, 0.1], **hidden), hoverlabel=dict(align="left"), showlegend=False)
    return fig


def _held(frame, index):
    """Quarterly values held on a daily index."""

    if frame.empty or len(index) == 0:
        return pd.DataFrame(index=index, columns=frame.columns, dtype=float)
    return frame.reindex(frame.index.union(index)).ffill().reindex(index)


def _signal_trace(result, name, color=None):
    frame, label, unit, _ = SIGNALS[name]
    series = result[frame][name].dropna() if name in result[frame] else pd.Series(dtype=float)
    suffix = unit if unit in ("%", "x") else (f" {unit}" if unit else "")
    quarterly = frame == "quarterly"
    return go.Scatter(
        x=series.index, y=series.values, name=label, mode="lines+markers" if quarterly else "lines",
        line=dict(width=1.6, shape="hv" if quarterly else "linear", color=color), marker=dict(size=5),
        hovertemplate=f"{label}: %{{y:,.2f}}{suffix}<extra></extra>")


def block_signals_figure(result, block, mobile=False):
    """Every input and output signal of one block over time: inputs in the upper panels, outputs in the lower ones.

    Signals on the same scale share a panel. Quarterly signals are drawn as steps (their value is held between
    updates), daily ones as lines. Returns None for a block with no signals to show.
    """

    spec = BLOCKS[block]
    panels = []
    for side, names in [("Input", spec["inputs"]), ("Output", spec["outputs"])]:
        groups = {}
        for name in names:
            groups.setdefault(SIGNALS[name][3], []).append(name)
        panels += [(side, axis, group) for axis, group in groups.items()]
    if not panels:
        return None
    titles = [f"{side}: " + ", ".join(SIGNALS[name][1] for name in group) for side, _, group in panels]
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=min(0.09, 0.4 / len(panels)),
                        subplot_titles=titles)
    for row, (_, axis, group) in enumerate(panels, start=1):
        for name in group:
            scenario = next((case for case in SCENARIOS if name.endswith(f"_{case.lower()}")), None)
            fig.add_trace(_signal_trace(result, name, SCENARIO_COLORS.get(scenario)), row=row, col=1)
        fig.update_yaxes(title_text=None if mobile else AXIS_TITLES.get(axis), row=row, col=1)
    for annotation in fig.layout.annotations:
        annotation.update(font=dict(size=10 if mobile else 12), x=0, xanchor="left")
    fig.update_layout(height=max(320, (190 if mobile else 220) * len(panels) + 80), hovermode="x unified",
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="left", x=0))
    return fig


def model_price_figure(result, ticker="", mobile=False):
    """The model's output over time: the model price and its bear-to-bull range against the close, the P/E the
    market pays for the risk-adjusted EPS against the acceptable P/E, and the gap between model and close."""

    daily, quarterly = result["daily"], result["quarterly"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=[f"{ticker} model price per share and close".strip(),
                                        "P/E: what the market pays and what the model accepts",
                                        "Model price vs close, %"])
    held = _held(quarterly[[f"eps_{name.lower()}" for name in SCENARIOS]] if not quarterly.empty else pd.DataFrame(
        columns=[f"eps_{name.lower()}" for name in SCENARIOS]), daily.index)
    bear = held["eps_bear"].where(held["eps_bear"] > 0) * daily["acceptable_pe"]
    bull = held["eps_bull"].where(held["eps_bull"] > 0) * daily["acceptable_pe"]
    fig.add_trace(go.Scatter(x=daily.index, y=bull, name="Bull case", mode="lines",
                             line=dict(width=0.6, color=SCENARIO_COLORS["Bull"]),
                             hovertemplate="Bull case: %{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily.index, y=bear, name="Bear case", mode="lines", fill="tonexty",
                             fillcolor="rgba(65, 105, 225, 0.10)", line=dict(width=0.6, color=SCENARIO_COLORS["Bear"]),
                             hovertemplate="Bear case: %{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily.index, y=daily["model_price"], name="Model price", mode="lines",
                             line=dict(width=2, color="royalblue"),
                             hovertemplate="Model price: %{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily.index, y=daily["close"], name="Close", mode="lines",
                             line=dict(width=1.6, color="dimgray"),
                             hovertemplate="Close: %{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily.index, y=daily["market_pe"], name="Market price / risk-adjusted EPS", mode="lines",
                             line=dict(width=1.4, color="dimgray"),
                             hovertemplate="Market P/E on model EPS: %{y:.1f}x<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily.index, y=daily["acceptable_pe"], name="Acceptable P/E", mode="lines",
                             line=dict(width=1.6, color="goldenrod"),
                             hovertemplate="Acceptable P/E: %{y:.1f}x<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily.index, y=daily["residual_pct"], name="Model vs close", mode="lines",
                             line=dict(width=1.4, color="teal"),
                             hovertemplate="Model vs close: %{y:+.1f}%<extra></extra>"), row=3, col=1)
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"), row=3, col=1)
    # A dotted line at every update of the quarterly part that falls in the window.
    if len(daily):
        for when in quarterly.index[(quarterly.index >= daily.index[0]) & (quarterly.index <= daily.index[-1])]:
            fig.add_shape(type="line", x0=when, x1=when, y0=0, y1=1, xref="x", yref="paper", layer="below",
                          line=dict(color="rgba(120, 120, 120, 0.35)", width=1, dash="dot"))
    for annotation in fig.layout.annotations:
        annotation.update(font=dict(size=10 if mobile else 12), x=0, xanchor="left")
    fig.update_layout(height=760 if mobile else 820, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="left", x=0))
    return fig


def items_table(result, tracked=None):
    """The register after the gate and router, for display: where each item goes and what it does."""

    items = result["items"].copy()
    if items.empty:
        return pd.DataFrame(columns=["Item", "Type", "Goes to", "Effect", "Gate"])
    unit = np.where(items["Route"] == "Scenario engine", "% of EPS", "P/E points")
    items["Effect"] = [f"{effect:+.1f} {text}" if np.isfinite(effect) else "not sized" for effect, text in zip(items["Effect"], unit)]
    table = items.rename(columns={"Route": "Goes to"})[["Item", "Type", "Goes to", "Effect", "Gate"]]
    if tracked is not None and len(tracked) == len(table):
        table["Mentions"] = tracked["Mentions"].to_numpy()
        table["Stale"] = np.where(tracked["Stale"].to_numpy(), "not mentioned lately", "")
    table["In the model"] = items["Gate"].to_numpy() == MODELLED
    return table
