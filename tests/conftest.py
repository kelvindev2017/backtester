import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_bars(closes, volumes=None, spread=0.01, index=None, seed=0):
    """Build OHLCV bars around a close path."""

    rng = np.random.default_rng(seed)
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) * (1 + spread * rng.uniform(0.2, 1.0, len(closes)))
    lows = np.minimum(opens, closes) * (1 - spread * rng.uniform(0.2, 1.0, len(closes)))
    if volumes is None:
        volumes = rng.uniform(0.8e6, 1.2e6, len(closes))
    if index is None:
        index = pd.bdate_range("2023-01-02", periods=len(closes))
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": np.asarray(volumes, dtype=float)},
        index=index,
    )


def to_weekly(daily):
    return (
        daily.resample("W-FRI")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
    )


@pytest.fixture
def breakout_daily():
    """An uptrend, a ~60-bar base between ~95 and ~105, then a high-volume breakout."""

    rng = np.random.default_rng(42)
    trend = np.linspace(60, 100, 260) + rng.normal(0, 0.8, 260)
    base = 100 + 4.5 * np.sin(np.linspace(0, 6 * np.pi, 60)) + rng.normal(0, 0.4, 60)
    closes = np.concatenate([trend, base, [107.5]])
    volumes = rng.uniform(0.8e6, 1.2e6, len(closes))
    volumes[-1] = 3.2e6
    bars = make_bars(closes, volumes, seed=1)
    # Make the breakout bar close near its high.
    bars.iloc[-1, bars.columns.get_loc("High")] = 107.8
    bars.iloc[-1, bars.columns.get_loc("Low")] = 104.6
    bars.iloc[-1, bars.columns.get_loc("Open")] = 104.9
    return bars


def make_session(day, closes, volumes=None, start="09:30"):
    index = pd.date_range(f"{day} {start}", periods=len(closes), freq="5min")
    return make_bars(closes, volumes, spread=0.0015, index=index, seed=3)


@pytest.fixture
def intraday_breakout():
    """Two quiet prior sessions, then a session that breaks out above 108 after the opening range."""

    prior_1 = make_session("2024-03-04", np.full(78, 104.0) + np.sin(np.arange(78) / 5) * 0.3)
    prior_2 = make_session("2024-03-05", np.full(78, 105.0) + np.sin(np.arange(78) / 5) * 0.3)
    opening = np.linspace(107.0, 107.6, 6)
    rally = np.linspace(107.7, 109.5, 30)
    closes = np.concatenate([opening, rally])
    volumes = np.full(len(closes), 1.0e6)
    volumes[6:10] = 3.0e6
    session = make_session("2024-03-06", closes, volumes)
    return pd.concat([prior_1, prior_2, session])
