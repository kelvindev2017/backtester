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
    # Standard signal columns shared by every setup.
    result["Signal_Up"] = result["Breakout_Up"]
    result["Signal_Down"] = result["Breakout_Down"]
    result["Trigger_Up"] = prior_high
    result["Trigger_Down"] = prior_low
    result["Invalidation_Up"] = prior_high
    result["Invalidation_Down"] = prior_low
    return result


def detect_pullbacks(
    df,
    fast=20,
    mid=50,
    slow=200,
    swing_bars=20,
    pullback_bars=5,
    min_depth_atr=1.5,
    volume_period=50,
    atr_period=14,
):
    """Pullback entries in an established trend.

    Long: 50-bar MA above the 200-bar MA, close above the 200-bar MA and the
    50-bar MA rising; price has pulled back at least ``min_depth_atr`` ATR
    from the recent ``swing_bars`` high to within 0.25 ATR of the ``fast``
    EMA without closing more than 1 ATR under the 50-bar MA; the signal is
    the first close above the prior bar's high, in the upper half of the bar.
    Short is the mirror image. Only past data is used at each bar.
    """

    close, high, low = df["Close"], df["High"], df["Low"]
    atr_series = atr(df, atr_period)
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    mid_ma = sma(close, mid)
    slow_ma = sma(close, slow)
    rvol = relative_volume(df["Volume"], volume_period)
    bar_range = (high - low).replace(0, np.nan)
    location = ((close - low) / bar_range).fillna(0.5)

    swing_high = high.shift(1).rolling(swing_bars, min_periods=swing_bars).max()
    swing_low = low.shift(1).rolling(swing_bars, min_periods=swing_bars).min()
    pullback_low = low.rolling(pullback_bars, min_periods=pullback_bars).min()
    pullback_high = high.rolling(pullback_bars, min_periods=pullback_bars).max()
    pullback_rvol = rvol.shift(1).rolling(pullback_bars - 1, min_periods=2).mean()
    mid_rising = mid_ma > mid_ma.shift(10)
    mid_falling = mid_ma < mid_ma.shift(10)

    up_trend = (mid_ma > slow_ma) & (close > slow_ma) & mid_rising
    down_trend = (mid_ma < slow_ma) & (close < slow_ma) & mid_falling
    long_pullback = (
        ((swing_high - pullback_low) >= min_depth_atr * atr_series)
        & (pullback_low <= fast_ema + 0.25 * atr_series)
        & (close >= mid_ma - atr_series)
    )
    short_pullback = (
        ((pullback_high - swing_low) >= min_depth_atr * atr_series)
        & (pullback_high >= fast_ema - 0.25 * atr_series)
        & (close <= mid_ma + atr_series)
    )
    long_trigger = (close > high.shift(1)) & (location >= 0.5)
    short_trigger = (close < low.shift(1)) & (location <= 0.5)

    raw_up = (up_trend & long_pullback & long_trigger).fillna(False)
    raw_down = (down_trend & short_pullback & short_trigger).fillna(False)

    result = pd.DataFrame(index=df.index)
    result["Signal_Up"] = raw_up & ~raw_up.shift(1, fill_value=False)
    result["Signal_Down"] = raw_down & ~raw_down.shift(1, fill_value=False)
    result["Trigger_Up"] = high.shift(1)
    result["Trigger_Down"] = low.shift(1)
    result["Invalidation_Up"] = pullback_low
    result["Invalidation_Down"] = pullback_high
    result["Depth_ATR_Up"] = (swing_high - pullback_low) / atr_series
    result["Depth_ATR_Down"] = (pullback_high - swing_low) / atr_series
    result["Pullback_RVOL"] = pullback_rvol
    result["RVOL"] = rvol
    result["ATR"] = atr_series
    result["Close_Location"] = location
    # For a pullback, confirmation is *light* volume while price pulled back.
    result["Volume_Confirmed"] = pullback_rvol < 1.0
    return result


def detect_reversals(df, lookback=20, mid=50, slow=200, volume_period=50, atr_period=14):
    """Short-term reversal: buy a breakdown inside a long-term uptrend (long only).

    Long-term uptrend: close above the 200-bar MA and the 50-bar MA above the
    200-bar MA. Signal: the first close below the prior ``lookback``-bar low.
    Large, liquid stocks tend to recover such short-term drops; the
    validation tab tests this on the user's own tickers. The signal day's
    low is the invalidation level. There is no bearish counterpart.
    """

    close = df["Close"]
    prior_low = df["Low"].shift(1).rolling(lookback, min_periods=lookback).min()
    mid_ma, slow_ma = sma(close, mid), sma(close, slow)
    up_trend = (close > slow_ma) & (mid_ma > slow_ma)
    below = (close < prior_low).fillna(False)
    fresh = below & ~below.shift(1, fill_value=False)
    atr_series = atr(df, atr_period)
    bar_range = (df["High"] - df["Low"]).replace(0, np.nan)

    result = pd.DataFrame(index=df.index)
    result["Signal_Up"] = (fresh & up_trend).fillna(False)
    result["Signal_Down"] = False
    result["Trigger_Up"] = close
    result["Trigger_Down"] = np.nan
    result["Invalidation_Up"] = df["Low"]
    result["Invalidation_Down"] = np.nan
    result["Prior_Low"] = prior_low
    result["Drop_ATR"] = (close.shift(5) - close) / atr_series
    result["Distance_200_ATR"] = (close - slow_ma) / atr_series
    result["RVOL"] = relative_volume(df["Volume"], volume_period)
    result["ATR"] = atr_series
    result["Close_Location"] = ((close - df["Low"]) / bar_range).fillna(0.5)
    result["Volume_Confirmed"] = True
    return result


def reversal_quality(row, close, direction):
    """Describe a reversal entry. Scored neutrally on purpose: validation showed
    no reliable link between bar-level quality features and outcomes."""

    details = [
        f"Close {close:,.2f} broke below the prior 20-bar low {row['Prior_Low']:,.2f} while the long-term trend is up.",
    ]
    if np.isfinite(row["Drop_ATR"]):
        details.append(f"Price fell {row['Drop_ATR']:.1f} ATR over the last 5 bars.")
    if np.isfinite(row["Distance_200_ATR"]):
        details.append(f"Close is still {row['Distance_200_ATR']:.1f} ATR above the 200-bar MA.")
    if np.isfinite(row["RVOL"]):
        details.append(f"Volume was {row['RVOL']:.1f}x its 50-bar average.")
    details.append(f"Bar closed {row['Close_Location'] * 100:.0f}% of the way up its range.")
    return 0.0, details, float(row["Trigger_Up"])


# ============================================================
# Moving-average rule (the Backtest tab's buy/sell rule)
# ============================================================

MA_RULE_SERIES = ["Close", "Fast_MA", "Slow_MA"]
MA_RULE_OPERATORS = [">", "<", ">=", "<=", "Cross Above", "Cross Below"]
DEFAULT_MA_RULE = {
    "fast_type": "EMA",
    "fast_period": 20,
    "slow_type": "SMA",
    "slow_period": 50,
    "buy": ("Fast_MA", "Cross Above", "Slow_MA"),
    "sell": ("Fast_MA", "Cross Below", "Slow_MA"),
}


def moving_average(series, ma_type, period):
    """Exponential or simple moving average."""

    period = int(period)
    if ma_type == "EMA":
        return series.ewm(span=period, adjust=False).mean()
    return series.rolling(window=period, min_periods=period).mean()


def rule_condition(left, operator, right):
    """Boolean condition ``left <operator> right`` on two aligned series."""

    if operator == ">":
        condition = left > right
    elif operator == "<":
        condition = left < right
    elif operator == ">=":
        condition = left >= right
    elif operator == "<=":
        condition = left <= right
    elif operator == "Cross Above":
        condition = (left > right) & (left.shift(1) <= right.shift(1))
    elif operator == "Cross Below":
        condition = (left < right) & (left.shift(1) >= right.shift(1))
    else:
        condition = pd.Series(False, index=left.index)
    return condition.fillna(False).astype(bool)


def ma_rule_conditions(df, rule):
    """Moving averages plus the buy and sell conditions of an MA rule."""

    rule = {**DEFAULT_MA_RULE, **(rule or {})}
    close = df["Close"]
    series = {
        "Close": close,
        "Fast_MA": moving_average(close, rule["fast_type"], rule["fast_period"]),
        "Slow_MA": moving_average(close, rule["slow_type"], rule["slow_period"]),
    }
    buy_left, buy_operator, buy_right = rule["buy"]
    sell_left, sell_operator, sell_right = rule["sell"]
    buy = rule_condition(series[buy_left], buy_operator, series[buy_right])
    sell = rule_condition(series[sell_left], sell_operator, series[sell_right])
    return series["Fast_MA"], series["Slow_MA"], buy, sell


def describe_ma_rule(rule, side="buy"):
    rule = {**DEFAULT_MA_RULE, **(rule or {})}
    names = {
        "Close": "Close",
        "Fast_MA": f"{rule['fast_type']} {rule['fast_period']}",
        "Slow_MA": f"{rule['slow_type']} {rule['slow_period']}",
    }
    left, operator, right = rule[side]
    return f"{names[left]} {operator.lower()} {names[right]}"


def detect_ma_rule(df, rule=None, volume_period=50, atr_period=14):
    """Signals from a moving-average rule.

    A signal is the first bar on which the buy (or sell) condition becomes
    true, so level rules such as ``Close > Slow_MA`` fire once per episode,
    like the other setups. The slow MA (or the signal bar's extreme, if it
    lies beyond the MA) is the invalidation level.
    """

    close = df["Close"]
    fast_ma, slow_ma, buy, sell = ma_rule_conditions(df, rule)
    atr_series = atr(df, atr_period)
    bar_range = (df["High"] - df["Low"]).replace(0, np.nan)

    result = pd.DataFrame(index=df.index)
    result["Signal_Up"] = buy & ~buy.shift(1, fill_value=False)
    result["Signal_Down"] = sell & ~sell.shift(1, fill_value=False)
    result["Trigger_Up"] = close
    result["Trigger_Down"] = close
    result["Invalidation_Up"] = np.fmin(slow_ma, df["Low"])
    result["Invalidation_Down"] = np.fmax(slow_ma, df["High"])
    result["Fast_MA"] = fast_ma
    result["Slow_MA"] = slow_ma
    result["RVOL"] = relative_volume(df["Volume"], volume_period)
    result["ATR"] = atr_series
    result["Close_Location"] = ((close - df["Low"]) / bar_range).fillna(0.5)
    result["Volume_Confirmed"] = True
    return result


def ma_rule_quality(row, close, direction, rule=None):
    """Describe an MA-rule entry. Scored neutrally: the rule has no quality grading."""

    details = [f"Rule fired: {describe_ma_rule(rule, 'buy' if direction > 0 else 'sell')}."]
    if np.isfinite(row["Fast_MA"]) and np.isfinite(row["Slow_MA"]):
        details.append(f"Fast MA {row['Fast_MA']:,.2f}, slow MA {row['Slow_MA']:,.2f}, close {close:,.2f}.")
    if np.isfinite(row["RVOL"]):
        details.append(f"Volume was {row['RVOL']:.1f}x its 50-bar average.")
    trigger = row["Trigger_Up"] if direction > 0 else row["Trigger_Down"]
    return 0.0, details, float(trigger)


def detect_signals(df, setup="breakout", lookback=20, volume_multiple=1.5, volume_period=50, ma_rule=None):
    """Dispatch to the detector for ``setup`` ("breakout", "pullback", "reversal" or "ma_rule")."""

    if setup == "ma_rule":
        return detect_ma_rule(df, ma_rule, volume_period=volume_period)
    if setup == "pullback":
        return detect_pullbacks(df, volume_period=volume_period)
    if setup == "reversal":
        return detect_reversals(df, lookback, volume_period=volume_period)
    return detect_breakouts(df, lookback, volume_period=volume_period, volume_multiple=volume_multiple)


def pullback_quality(row, close, direction):
    """Score one pullback entry bar in [-1, 1] and explain it."""

    details = []
    trigger = row["Trigger_Up"] if direction > 0 else row["Trigger_Down"]
    invalidation = row["Invalidation_Up"] if direction > 0 else row["Invalidation_Down"]
    atr_value = row["ATR"]

    depth = row["Depth_ATR_Up"] if direction > 0 else row["Depth_ATR_Down"]
    depth_score = clamp(1.0 - abs(depth - 3.0) / 2.0) if np.isfinite(depth) else np.nan
    if np.isfinite(depth):
        details.append(f"Pullback depth {depth:.1f} ATR from the swing {'high' if direction > 0 else 'low'} (2-4 ATR is ideal).")

    pullback_rvol = row["Pullback_RVOL"]
    volume_score = clamp((1.0 - pullback_rvol) / 0.4) if np.isfinite(pullback_rvol) else np.nan
    if np.isfinite(pullback_rvol):
        details.append(
            f"Volume during the pullback averaged {pullback_rvol:.2f}x normal "
            + ("(light - sellers not in control)." if pullback_rvol < 1.0 else "(heavy - distribution risk).")
            if direction > 0 else
            f"Volume during the bounce averaged {pullback_rvol:.2f}x normal "
            + ("(light - buyers not in control)." if pullback_rvol < 1.0 else "(heavy - accumulation risk).")
        )

    location = row["Close_Location"] if direction > 0 else 1 - row["Close_Location"]
    location_score = clamp((location - 0.5) * 2.5)
    details.append(f"Trigger bar closed in the {'top' if direction > 0 else 'bottom'} {(1 - location) * 100:.0f}% of its range.")

    risk_score = np.nan
    if np.isfinite(atr_value) and atr_value > 0 and np.isfinite(invalidation):
        risk_atr = direction * (close - invalidation) / atr_value
        risk_score = clamp(1.0 - max(0.0, risk_atr - 1.5) / 1.5)
        details.append(f"Close is {risk_atr:.2f} ATR from the pullback {'low' if direction > 0 else 'high'} {invalidation:,.2f} (the invalidation level).")

    score = weighted_score([(depth_score, 0.25), (volume_score, 0.30), (location_score, 0.20), (risk_score, 0.25)])
    return score, details, float(trigger)


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


def latest_active_signal(df, signals, window=3):
    """Return the most recent signal within ``window`` bars and whether it still holds.

    The dict contains ``direction``, ``date``, ``bars_ago``, ``trigger``,
    ``invalidation`` and ``holding``. A signal whose price has since moved
    back through its invalidation level is reported as not holding.
    """

    recent = signals.iloc[-int(window):]
    last_close = float(df["Close"].iloc[-1])
    for timestamp in reversed(recent.index):
        row = recent.loc[timestamp]
        for direction, suffix in [(1, "Up"), (-1, "Down")]:
            if not row[f"Signal_{suffix}"]:
                continue
            trigger = float(row[f"Trigger_{suffix}"])
            invalidation = float(row[f"Invalidation_{suffix}"])
            return {
                "direction": direction,
                "date": timestamp,
                "bars_ago": len(df) - 1 - df.index.get_loc(timestamp),
                "trigger": trigger,
                "invalidation": invalidation,
                "holding": direction * (last_close - invalidation) > 0,
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
