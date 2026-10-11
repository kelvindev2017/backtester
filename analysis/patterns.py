"""Trend classification, breakout/breakdown detection, volume behaviour and
the historical edge of breakout signals.

Scores returned here are *bullish-positive* (+1 bullish, -1 bearish). The
engine flips them for a SELL proposal.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis.indicators import adx, atr, clamp, relative_volume, sma
from analysis.levels import find_pivots
from analysis.scoring import mean_score, weighted_score


# ============================================================
# Trend
# ============================================================

@dataclass
class TrendAssessment:
    timeframe: str
    label: str
    score: float
    details: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def assess_trend(df, timeframe="Daily", fast=50, slow=200, slope_bars=20, pivot_bars=5):
    """Classify trend from moving averages, MA slope, swing structure and ADX."""

    if df is None or len(df) < max(fast, 20):
        return TrendAssessment(timeframe, "Insufficient data", np.nan, [f"{timeframe}: not enough bars to judge trend."])

    close = df["Close"]
    last_close = float(close.iloc[-1])
    fast_ma = sma(close, fast)
    slow_ma = sma(close, slow)
    details, parts, metrics = [], [], {}

    slow_available = pd.notna(slow_ma.iloc[-1])
    reference_ma = slow_ma if slow_available else fast_ma
    reference_name = f"{slow}-bar MA" if slow_available else f"{fast}-bar MA"

    reference_value = float(reference_ma.iloc[-1])
    above = last_close > reference_value
    parts.append(1.0 if above else -1.0)
    details.append(f"{timeframe}: close {last_close:,.2f} is {'above' if above else 'below'} the {reference_name} ({reference_value:,.2f}).")

    if slow_available and pd.notna(fast_ma.iloc[-1]):
        golden = fast_ma.iloc[-1] > slow_ma.iloc[-1]
        parts.append(1.0 if golden else -1.0)
        details.append(f"{timeframe}: {fast}-bar MA is {'above' if golden else 'below'} the {slow}-bar MA.")

    reference_clean = reference_ma.dropna()
    if len(reference_clean) > slope_bars:
        slope = reference_clean.iloc[-1] / reference_clean.iloc[-1 - slope_bars] - 1
        metrics["ma_slope_pct"] = round(float(slope) * 100, 2)
        parts.append(clamp(slope / 0.05))
        details.append(f"{timeframe}: {reference_name} changed {slope * 100:+.1f}% over {slope_bars} bars.")

    pivots = find_pivots(df.iloc[-250:], pivot_bars, pivot_bars)
    swing_highs = pivots.loc[pivots["Type"] == "high", "Price"]
    swing_lows = pivots.loc[pivots["Type"] == "low", "Price"]
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        high_step = np.sign(swing_highs.iloc[-1] - swing_highs.iloc[-2])
        low_step = np.sign(swing_lows.iloc[-1] - swing_lows.iloc[-2])
        structure = 0.5 * (high_step + low_step)
        parts.append(float(structure))
        names = {
            1.0: "higher highs and higher lows",
            -1.0: "lower highs and lower lows",
        }
        details.append(f"{timeframe}: swing structure shows {names.get(structure, 'mixed highs/lows (range-bound)')}.")

    adx_frame = adx(df)
    adx_value = adx_frame["ADX"].iloc[-1]
    strength_factor = 1.0
    if pd.notna(adx_value):
        metrics["adx"] = round(float(adx_value), 1)
        if adx_value < 20:
            strength_factor = 0.6
            details.append(f"{timeframe}: ADX {adx_value:.0f} indicates a weak / non-trending market.")
        else:
            details.append(f"{timeframe}: ADX {adx_value:.0f} indicates a {'strong' if adx_value >= 25 else 'developing'} trend.")

    score = mean_score(parts)
    score = np.nan if score is None else score * strength_factor
    if not np.isfinite(score):
        label = "Insufficient data"
    elif score > 0.35:
        label = "Uptrend"
    elif score < -0.35:
        label = "Downtrend"
    else:
        label = "Sideways"
    return TrendAssessment(timeframe, label, score, details, metrics)


# ============================================================
# Breakouts and breakdowns
# ============================================================

def detect_breakouts(df, lookback=20, volume_period=50, volume_multiple=1.5, atr_period=14):
    """Flag the first close beyond the prior ``lookback``-bar high/low.

    Only data available at each bar's close is used, so the result can be
    replayed historically without look-ahead bias.
    """

    result = pd.DataFrame(index=df.index)
    prior_high = df["High"].shift(1).rolling(lookback, min_periods=lookback).max()
    prior_low = df["Low"].shift(1).rolling(lookback, min_periods=lookback).min()
    atr_series = atr(df, atr_period)
    previous_atr = atr_series.shift(1)
    close = df["Close"]

    above = close > prior_high
    below = close < prior_low
    result["Breakout_Up"] = above & ~above.shift(1, fill_value=False)
    result["Breakout_Down"] = below & ~below.shift(1, fill_value=False)
    result["Prior_High"] = prior_high
    result["Prior_Low"] = prior_low
    result["RVOL"] = relative_volume(df["Volume"], volume_period)
    result["ATR"] = atr_series
    result["Base_Width_ATR"] = (prior_high - prior_low) / previous_atr
    bar_range = (df["High"] - df["Low"]).replace(0, np.nan)
    result["Close_Location"] = ((close - df["Low"]) / bar_range).fillna(0.5)
    result["Volume_Confirmed"] = result["RVOL"] >= volume_multiple
    return result


def breakout_quality(row, close, direction):
    """Score one breakout bar in [-1, 1] and explain it.

    ``direction`` is +1 for an upside breakout and -1 for a downside breakdown.
    """

    trigger = row["Prior_High"] if direction > 0 else row["Prior_Low"]
    atr_value = row["ATR"]
    details = []

    rvol = row["RVOL"]
    volume_score = clamp((rvol - 1.0) / 1.0)
    if np.isfinite(rvol):
        details.append(
            f"Volume was {rvol:.1f}x its 50-bar average"
            + (" (volume-confirmed)." if row["Volume_Confirmed"] else " (NOT volume-confirmed).")
        )

    location = row["Close_Location"] if direction > 0 else 1 - row["Close_Location"]
    location_score = clamp((location - 0.5) * 2.5)
    details.append(f"Signal bar closed in the {'top' if direction > 0 else 'bottom'} {(1 - location) * 100:.0f}% of its range.")

    extension_score = np.nan
    if np.isfinite(atr_value) and atr_value > 0:
        extension = direction * (close - trigger) / atr_value
        extension_score = clamp(1.0 - max(0.0, extension - 1.5))
        details.append(
            f"Close is {extension:.2f} ATR beyond the trigger level {trigger:,.2f}"
            + (" (extended - chasing risk)." if extension > 2.5 else ".")
        )

    width = row["Base_Width_ATR"]
    base_score = clamp((10.0 - width) / 5.0)
    if np.isfinite(width):
        details.append(f"Prior base was {width:.1f} ATR wide ({'tight' if width <= 6 else 'loose' if width >= 12 else 'moderate'} consolidation).")

    score = weighted_score(
        [(volume_score, 0.35), (location_score, 0.20), (extension_score, 0.25), (base_score, 0.20)]
    )
    return score, details, float(trigger)


def latest_active_signal(df, breakouts, window=3):
    """Return the most recent breakout within ``window`` bars that still holds.

    The dict contains ``direction``, ``date``, ``bars_ago`` and ``trigger``.
    A breakout whose price has since moved back through the trigger is
    reported as failed instead.
    """

    recent = breakouts.iloc[-int(window):]
    last_close = float(df["Close"].iloc[-1])
    for timestamp in reversed(recent.index):
        row = recent.loc[timestamp]
        for direction, column, trigger_column in [(1, "Breakout_Up", "Prior_High"), (-1, "Breakout_Down", "Prior_Low")]:
            if not row[column]:
                continue
            trigger = float(row[trigger_column])
            holding = direction * (last_close - trigger) > 0
            return {
                "direction": direction,
                "date": timestamp,
                "bars_ago": len(df) - 1 - df.index.get_loc(timestamp),
                "trigger": trigger,
                "holding": holding,
            }
    return None


# ============================================================
# Volume / accumulation
# ============================================================

def assess_volume(df, window=20, obv_window=50):
    """Accumulation/distribution evidence from up/down volume, OBV trend and CMF."""

    if df is None or len(df) < window + 1:
        return np.nan, ["Not enough bars to assess volume behaviour."], {}

    recent = df.iloc[-window:]
    change = df["Close"].diff().iloc[-window:]
    up_volume = recent["Volume"].where(change > 0, 0).sum()
    down_volume = recent["Volume"].where(change < 0, 0).sum()
    details, metrics, parts = [], {}, []

    if up_volume > 0 and down_volume > 0:
        ratio = up_volume / down_volume
        metrics["up_down_volume_ratio"] = round(float(ratio), 2)
        parts.append(clamp(np.log(ratio) / np.log(2)))
        label = "accumulation" if ratio > 1.2 else "distribution" if ratio < 0.83 else "balanced"
        details.append(f"Up-day / down-day volume over {window} bars = {ratio:.2f} ({label}).")

    span = df.iloc[-(obv_window + 1):]
    total_volume = span["Volume"].iloc[1:].sum()
    if total_volume > 0:
        signed = (np.sign(span["Close"].diff()) * span["Volume"]).iloc[1:].sum()
        obv_score = clamp(3.0 * signed / total_volume)
        price_change = span["Close"].iloc[-1] / span["Close"].iloc[0] - 1
        metrics["obv_net_pct_of_volume"] = round(float(signed / total_volume) * 100, 1)
        parts.append(obv_score)
        if np.sign(signed) != np.sign(price_change) and abs(price_change) > 0.02:
            details.append(
                f"OBV diverges from price over {obv_window} bars (price {price_change * 100:+.1f}%, "
                f"OBV {'rising' if signed > 0 else 'falling'}) - volume does not confirm the move."
            )
        else:
            details.append(f"OBV over {obv_window} bars is {'rising' if signed > 0 else 'falling'}, in line with price.")

    bar_range = (recent["High"] - recent["Low"]).replace(0, np.nan)
    money_flow_multiplier = ((recent["Close"] - recent["Low"]) - (recent["High"] - recent["Close"])) / bar_range
    if recent["Volume"].sum() > 0:
        cmf = float((money_flow_multiplier.fillna(0) * recent["Volume"]).sum() / recent["Volume"].sum())
        metrics["chaikin_money_flow"] = round(cmf, 3)
        parts.append(clamp(cmf * 4.0))
        details.append(f"Chaikin Money Flow ({window}) = {cmf:+.2f}.")

    score = mean_score(parts)
    return (np.nan if score is None else score), details, metrics


# ============================================================
# Historical edge of the same signal on this ticker
# ============================================================

def historical_signal_stats(df, signal_dates, direction, horizons=(5, 10, 20), adverse_bars=10):
    """Forward returns after past signals, entering at the next bar's open.

    Returns ``(events, summary)``. Returns are oriented to ``direction`` so a
    positive number is always a profitable outcome.
    """

    opens = df["Open"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    records = []

    for timestamp in signal_dates:
        position = df.index.get_loc(timestamp)
        entry_position = position + 1
        if entry_position >= len(df):
            continue
        entry = opens[entry_position]
        if not np.isfinite(entry) or entry <= 0:
            continue
        record = {"Signal_Date": timestamp, "Entry_Open": entry}
        for horizon in horizons:
            exit_position = position + horizon
            record[f"Return_{horizon}"] = (
                direction * (closes[exit_position] / entry - 1) if exit_position < len(df) else np.nan
            )
        window_end = min(len(df), entry_position + adverse_bars)
        if direction > 0:
            record["Max_Adverse"] = lows[entry_position:window_end].min() / entry - 1
        else:
            record["Max_Adverse"] = entry / highs[entry_position:window_end].max() - 1
        records.append(record)

    events = pd.DataFrame(records)
    summary_rows = []
    for horizon in horizons:
        column = f"Return_{horizon}"
        values = events[column].dropna() if not events.empty else pd.Series(dtype=float)
        all_forward = direction * (df["Close"].shift(-horizon) / df["Open"].shift(-1) - 1)
        all_forward = all_forward.dropna()
        summary_rows.append(
            {
                "Horizon (bars)": horizon,
                "Signals": int(len(values)),
                "Win rate": float((values > 0).mean()) if len(values) else np.nan,
                "Average return": float(values.mean()) if len(values) else np.nan,
                "Median return": float(values.median()) if len(values) else np.nan,
                "Base win rate (any day)": float((all_forward > 0).mean()) if len(all_forward) else np.nan,
                "Base average return": float(all_forward.mean()) if len(all_forward) else np.nan,
            }
        )
    return events, pd.DataFrame(summary_rows)


def historical_edge_score(summary, horizon=10, min_signals=3):
    """Turn the historical summary into a shrunk score in [-1, 1]."""

    rows = summary.loc[summary["Horizon (bars)"] == horizon]
    if rows.empty:
        return None, ["No historical signals to evaluate."]
    row = rows.iloc[0]
    count = int(row["Signals"])
    if count < min_signals:
        return None, [f"Only {count} comparable past signal(s) with {horizon}-bar outcomes; too few to judge."]

    win_rate = row["Win rate"]
    base = row["Base win rate (any day)"]
    absolute = clamp((win_rate - 0.5) * 4.0)
    excess = clamp((win_rate - base) * 4.0) if np.isfinite(base) else np.nan
    raw = mean_score([absolute, excess, clamp(row["Average return"] / 0.04)])
    shrink = count / (count + 8.0)
    score = raw * shrink
    details = [
        f"{count} past volume-confirmed signals: {win_rate * 100:.0f}% were profitable after {horizon} bars "
        f"(average {row['Average return'] * 100:+.1f}%), versus a {base * 100:.0f}% base rate for any day.",
        f"Score shrunk by {(1 - shrink) * 100:.0f}% for the small sample size.",
    ]
    return score, details
