"""Price and volume indicators shared by the signal-analysis modules.

All functions are causal: the value at bar ``t`` only uses data up to and
including bar ``t``.
"""

import numpy as np
import pandas as pd


def sma(series, period):
    """Simple moving average that stays NaN until ``period`` bars exist."""

    return series.rolling(window=int(period), min_periods=int(period)).mean()


def ema(series, period):
    """Exponential moving average."""

    return series.ewm(span=int(period), adjust=False).mean()


def true_range(df):
    """Wilder's true range."""

    previous_close = df["Close"].shift(1)
    ranges = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df, period=14):
    """Average true range using Wilder smoothing."""

    return true_range(df).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()


def rsi(close, period=14):
    """Relative strength index using Wilder smoothing."""

    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    result = 100 - 100 / (1 + relative_strength)
    return result.where(loss != 0, 100.0)


def adx(df, period=14):
    """Return ADX, +DI and -DI as a DataFrame."""

    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    smoothed_tr = true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean() / smoothed_tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean() / smoothed_tr

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx_line = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return pd.DataFrame({"ADX": adx_line, "Plus_DI": plus_di, "Minus_DI": minus_di})


def obv(df):
    """On-balance volume."""

    direction = np.sign(df["Close"].diff()).fillna(0.0)
    return (direction * df["Volume"]).cumsum()


def relative_volume(volume, period=50):
    """Volume divided by the average of the *previous* ``period`` bars."""

    baseline = volume.shift(1).rolling(window=period, min_periods=max(5, period // 2)).mean()
    return volume / baseline.replace(0, np.nan)


def normalized_slope(series, window):
    """Least-squares slope of the last ``window`` values per bar, as a
    fraction of the window's mean absolute level. Returns NaN if unavailable."""

    values = pd.Series(series).dropna().iloc[-int(window):]
    if len(values) < max(3, int(window) // 2):
        return np.nan
    x = np.arange(len(values), dtype=float)
    slope = np.polyfit(x, values.to_numpy(dtype=float), 1)[0]
    scale = np.abs(values).mean()
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    return float(slope / scale)


def clamp(value, low=-1.0, high=1.0):
    """Clamp a number into ``[low, high]``; NaN stays NaN."""

    if value is None or not np.isfinite(value):
        return np.nan
    return float(min(high, max(low, value)))


def resample_ohlcv(daily, rule="W-FRI"):
    """Aggregate daily OHLCV bars (weekly bars ending Friday by default)."""

    return (
        daily.resample(rule)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
    )
