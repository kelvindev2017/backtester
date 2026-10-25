"""Illustrations (made-up prices) of how each signal rule enters and exits.

They sit under the rule's settings so the numbers in the settings and the labels in the picture match. The
prices are invented to show the mechanism clearly; periods in the labels come from the settings, while the
lines are drawn with short periods so the pattern fits in a few bars.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

UP, DOWN = "seagreen", "indianred"
LEVEL, AVERAGE, SLOW_AVERAGE, TEXT = "royalblue", "darkorange", "mediumpurple", "#333"
LABEL_BACKGROUND = "rgba(255,255,255,0.85)"


def bars_from_closes(closes, wick=0.6, gaps=None):
    """Candles around a close path: each bar opens at the prior close (plus an optional gap)."""

    closes = np.asarray(closes, dtype=float)
    gaps = np.zeros(len(closes)) if gaps is None else np.asarray(gaps, dtype=float)
    opens = np.concatenate([[closes[0] - 0.5], closes[:-1]]) + gaps
    highs = np.maximum(opens, closes) + wick
    lows = np.minimum(opens, closes) - wick
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes})


def candles(fig, frame, **position):
    fig.add_trace(go.Candlestick(
        x=list(range(1, len(frame) + 1)), open=frame["Open"], high=frame["High"], low=frame["Low"],
        close=frame["Close"], showlegend=False, increasing_line_color=UP, decreasing_line_color=DOWN,
        hoverinfo="skip",
    ), **position)


def line(fig, values, color, dash=None, **position):
    fig.add_trace(go.Scatter(x=list(range(1, len(values) + 1)), y=values, mode="lines", showlegend=False,
                             hoverinfo="skip", line=dict(color=color, width=1.6, dash=dash)), **position)


def note(fig, x, y, text, color=TEXT, ax=0, ay=-40, arrow=True, **extra):
    fig.add_annotation(x=x, y=y, text=text, ax=ax, ay=ay, showarrow=arrow, arrowhead=2, arrowcolor=color,
                       font=dict(size=10, color=color), bgcolor=LABEL_BACKGROUND, **extra)


def finish(fig, y_range=None, height=290, title="How the rule trades (illustration)"):
    fig.update_layout(height=height, margin=dict(l=5, r=5, t=30, b=5), title=dict(text=title, font=dict(size=13)),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showticklabels=False, showgrid=False, rangeslider_visible=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    return fig


def day_breakout_figure(lookback=10, exit_mode="top_range", keep_fraction=2 / 3, ma_label="SMA 20"):
    """Day breakout: buy-stop above the prior N-day high, then the chosen overnight exit."""

    # (open, high, low, close): a base under 104, the breakout day, two strong closes, then a pullback.
    bars = [(100, 102, 98.5, 101), (101, 103, 99.5, 99.8), (99.8, 101.5, 98, 100.6), (100.6, 104, 100, 103.2),
            (103.2, 103.6, 100.5, 101.4), (101.4, 102.4, 99.8, 100.2), (100.2, 102.8, 99.6, 102.2),
            (102.2, 103.4, 101, 101.8), (101.8, 102.6, 100.4, 102.1), (102.1, 103.5, 101.2, 102.9),
            (103.4, 108, 102.8, 107.6), (107.6, 110.5, 106.8, 110.1), (110.1, 112.4, 109.4, 112.0),
            (112.0, 113.0, 108.6, 109.4), (109.4, 110.2, 106.5, 107.0), (107.0, 108.0, 104.2, 104.6)]
    frame = pd.DataFrame(bars, columns=["Open", "High", "Low", "Close"])
    breakout = 10
    level = frame["High"].iloc[:breakout].max()
    ma = frame["Close"].rolling(6, min_periods=1).mean()
    location = (frame["Close"] - frame["Low"]) / (frame["High"] - frame["Low"])
    exit_pct = (1 - keep_fraction) * 100
    if exit_mode == "same_day":
        exit_index, exit_text = breakout, "Sell at the close<br>(never held overnight)"
    elif exit_mode == "above_ma":
        exit_index = next((i for i in range(breakout, len(frame)) if frame["Close"].iloc[i] < ma.iloc[i]), len(frame) - 1)
        exit_text = f"First close below<br>the {ma_label}: sell"
    else:
        exit_index = next((i for i in range(breakout, len(frame)) if location.iloc[i] < keep_fraction), len(frame) - 1)
        exit_text = f"Close more than {exit_pct:.0f}%<br>below the high: sell"

    fig = go.Figure()
    candles(fig, frame)
    fig.add_shape(type="line", x0=0.5, x1=breakout + 1.5, y0=level, y1=level, line=dict(color=LEVEL, dash="dash"))
    note(fig, 3, level, f"Prior {lookback}-day high", LEVEL, arrow=False, yshift=9)
    if exit_mode == "above_ma":
        line(fig, ma, AVERAGE)
        note(fig, 3, ma.iloc[2], ma_label, AVERAGE, arrow=False, yshift=-14)
    elif exit_mode == "top_range":
        # Shade the hold zone of the breakout bar: within the exit % of the day's high.
        high, low = frame["High"].iloc[breakout], frame["Low"].iloc[breakout]
        fig.add_shape(type="rect", x0=breakout + 0.6, x1=breakout + 1.4, y0=low + keep_fraction * (high - low),
                      y1=high, fillcolor=UP, opacity=0.35, line_width=0)
    note(fig, breakout + 1, level, "Buy-stop fills<br>at the high", ax=-50, ay=55)
    hold_text = (f"Close within {exit_pct:.0f}% of the high: hold" if exit_mode == "top_range"
                 else f"Close above the {ma_label}: hold")
    if exit_index > breakout:
        note(fig, breakout + 1, frame["High"].iloc[breakout], hold_text, UP, ax=-45, ay=-28)
    same_day = exit_index == breakout
    note(fig, exit_index + 1, frame["Close"].iloc[exit_index], exit_text, DOWN,
         ax=-50 if same_day else 0, ay=-45 if same_day else 60)
    return finish(fig, [93, 115])


def ma_rule_figure(fast_label="EMA 20", slow_label="SMA 50", buy_text="Fast MA crosses above slow MA",
                   sell_text="Fast MA crosses below slow MA"):
    """Moving-average rule: buy and sell when the rule's conditions first become true."""

    steps = np.r_[np.linspace(100, 94, 8), np.linspace(94, 110, 14), np.linspace(110, 98, 12)]
    closes = steps + np.sin(np.arange(len(steps)) * 1.7) * 0.8
    frame = bars_from_closes(closes)
    fast = pd.Series(closes).ewm(span=4, adjust=False).mean()
    slow = pd.Series(closes).rolling(9, min_periods=1).mean()
    above = fast > slow
    buy = next(i for i in range(9, len(closes)) if above[i] and not above[i - 1])
    sell = next(i for i in range(buy + 1, len(closes)) if not above[i] and above[i - 1])

    fig = go.Figure()
    candles(fig, frame)
    line(fig, fast, AVERAGE)
    line(fig, slow, SLOW_AVERAGE)
    note(fig, 2, fast.iloc[1], fast_label, AVERAGE, arrow=False, yshift=12)
    note(fig, 6, slow.iloc[5], slow_label, SLOW_AVERAGE, arrow=False, yshift=-16)
    note(fig, buy + 1, frame["Low"].iloc[buy], f"BUY: {buy_text}", UP, ax=20, ay=45)
    note(fig, sell + 1, frame["High"].iloc[sell], f"SELL: {sell_text}", DOWN, ax=-30, ay=-45)
    return finish(fig)


def breakout_figure(lookback=20, volume_multiple=1.5, exit_ma=200):
    """Breakout: first close above the prior N-bar high; sell on a close below the exit SMA."""

    closes = np.r_[[100, 101.5, 99.5, 101, 102.5, 100.5, 99, 100.8, 102.2, 101, 100, 101.6],
                   [105.2, 106.5, 107.2, 105.8, 104, 102.4, 101.2, 99.6]]
    frame = bars_from_closes(closes, wick=0.7)
    volume = np.r_[np.full(12, 1.0) + np.sin(np.arange(12)) * 0.15, [2.3, 1.4, 1.2, 1.0, 1.1, 1.2, 1.3, 1.5]]
    breakout = 12
    high = frame["High"].iloc[:breakout].max()
    average = pd.Series(np.r_[np.full(4, 99.0), closes]).rolling(10).mean().iloc[4:].bfill().reset_index(drop=True)
    exit_index = next(i for i in range(breakout + 1, len(frame)) if frame["Close"].iloc[i] < average.iloc[i])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    candles(fig, frame, row=1, col=1)
    line(fig, average, SLOW_AVERAGE, row=1, col=1)
    fig.add_shape(type="line", x0=0.5, x1=breakout + 1.5, y0=high, y1=high, line=dict(color=LEVEL, dash="dash"),
                  row=1, col=1)
    fig.add_annotation(x=3, y=high, text=f"Prior {lookback}-bar high", showarrow=False, yshift=9,
                       font=dict(size=10, color=LEVEL), row=1, col=1)
    fig.add_annotation(x=4, y=average.iloc[3], text=f"{exit_ma}-bar SMA", showarrow=False, yshift=-12,
                       font=dict(size=10, color=SLOW_AVERAGE), bgcolor=LABEL_BACKGROUND, row=1, col=1)
    fig.add_trace(go.Bar(x=list(range(1, len(frame) + 1)), y=volume, showlegend=False, hoverinfo="skip",
                         marker_color=["darkorange" if i == breakout else "lightgray" for i in range(len(frame))]),
                  row=2, col=1)
    fig.add_annotation(x=breakout + 1, y=frame["Close"].iloc[breakout],
                       text=f"BUY: close above the high<br>(Signal Analysis: on ≥{volume_multiple:g}x volume)",
                       ax=-65, ay=-40, arrowhead=2, arrowcolor=UP, font=dict(size=10, color=UP),
                       bgcolor=LABEL_BACKGROUND, row=1, col=1)
    fig.add_annotation(x=exit_index + 1, y=frame["Close"].iloc[exit_index], text=f"SELL: close below<br>the {exit_ma}-bar SMA",
                       ax=-45, ay=35, arrowhead=2, arrowcolor=DOWN, font=dict(size=10, color=DOWN),
                       bgcolor=LABEL_BACKGROUND, row=1, col=1)
    fig.add_annotation(x=2, y=1.6, text="Volume", showarrow=False, font=dict(size=9, color="gray"), row=2, col=1)
    return finish(fig, height=320)


def pullback_figure(pullback_ema=20, trend_mid=50, trend_slow=200, min_depth_atr=1.5):
    """Pullback in an uptrend: a dip to the EMA, then the first close above the prior bar's high."""

    closes = np.r_[np.linspace(95, 108, 12), [109.5, 108.2, 106.6, 105.3, 104.6, 104.9, 106.4, 107.8, 109.6, 111]]
    frame = bars_from_closes(closes, wick=0.6)
    ema = pd.Series(closes).ewm(span=8, adjust=False).mean()
    trend = pd.Series(np.linspace(90, 101, len(closes)))
    swing = 12
    trigger = 18  # first close above the prior bar's high after the dip

    fig = go.Figure()
    candles(fig, frame)
    line(fig, ema, AVERAGE)
    line(fig, trend, SLOW_AVERAGE, dash="dot")
    note(fig, 4, ema.iloc[3], f"{pullback_ema} EMA", AVERAGE, arrow=False, yshift=12)
    note(fig, 9, trend.iloc[8], f"{trend_mid}-bar MA rising, above the {trend_slow}", SLOW_AVERAGE, arrow=False,
         yshift=-12)
    note(fig, swing + 1, frame["High"].iloc[swing], "Swing high", TEXT, ax=-25, ay=-30)
    note(fig, 17, frame["Low"].iloc[16], f"Dip of ≥{min_depth_atr:g} ATR<br>to the EMA, light volume", TEXT,
         ax=-10, ay=45)
    note(fig, trigger + 1, frame["High"].iloc[trigger], "BUY: close above<br>the prior bar's high", UP,
         ax=30, ay=-50)
    return finish(fig)


def reversal_figure(lookback=20, trend_mid=50, trend_slow=200, max_hold=10):
    """Reversal: in a long-term uptrend, buy the first close below the prior N-bar low; sell on the bounce."""

    closes = np.r_[np.linspace(100, 109, 10), [108.4, 109.2, 108.6, 107.5, 106.8, 105.2, 104.4, 105.6, 107.2, 108.8]]
    frame = bars_from_closes(closes, wick=0.6)
    trend = pd.Series(np.linspace(94, 101, len(closes)))
    signal = 16
    low = frame["Low"].iloc[signal - 10:signal].min()
    exit_index = next(i for i in range(signal + 1, len(frame)) if frame["Close"].iloc[i] > frame["High"].iloc[i - 1])

    fig = go.Figure()
    candles(fig, frame)
    line(fig, trend, SLOW_AVERAGE, dash="dot")
    fig.add_shape(type="line", x0=signal - 9.5, x1=signal + 1.5, y0=low, y1=low, line=dict(color=LEVEL, dash="dash"))
    note(fig, signal - 6, low, f"Prior {lookback}-bar low", LEVEL, arrow=False, yshift=-10)
    note(fig, 5, trend.iloc[4], f"Above the {trend_slow}-bar MA, {trend_mid} above it", SLOW_AVERAGE, arrow=False,
         yshift=-12)
    note(fig, signal + 1, frame["Low"].iloc[signal], "BUY: first close below<br>the low, in an uptrend", UP,
         ax=-60, ay=40)
    note(fig, exit_index + 1, frame["High"].iloc[exit_index],
         f"SELL: first close above the prior<br>bar's high, or after {max_hold} bars", DOWN, ax=-30, ay=-45)
    return finish(fig)
