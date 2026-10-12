"""Intraday entry confirmation on 5-minute and 15-minute bars.

Once a daily breakout (or breakdown) has closed, the entry is timed on the
following session:

* 5-minute: opening-range breakout. After the first ``opening_range_minutes``,
  the first 5-minute close beyond both the opening range and the daily
  trigger level, on the correct side of session VWAP, is the entry.
* 15-minute: the latest 15-minute close must hold beyond the trigger level
  and VWAP, with the 9/20 EMA stack pointing in the trade direction.

The 15-minute bars are resampled from the 5-minute bars so both views use
the same data.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis.indicators import ema


@dataclass
class IntradayResult:
    status: str
    score: float = None
    session_date: object = None
    entry_time: pd.Timestamp = None
    entry_price: float = None
    opening_range_high: float = None
    opening_range_low: float = None
    details: list = field(default_factory=list)
    session_bars: pd.DataFrame = None


def session_vwap(df):
    """Volume-weighted average price that resets every session."""

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    session = df.index.normalize()
    cumulative_value = (typical_price * df["Volume"]).groupby(session).cumsum()
    cumulative_volume = df["Volume"].groupby(session).cumsum().replace(0, np.nan)
    return cumulative_value / cumulative_volume


def resample_session_bars(df, rule="15min"):
    """Resample intraday bars without letting a bar span two sessions."""

    frames = []
    for _, session in df.groupby(df.index.normalize()):
        frames.append(
            session.resample(rule, label="left", closed="left", origin="start")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna(subset=["Close"])
        )
    return pd.concat(frames) if frames else df.iloc[0:0]


def evaluate_intraday_entry(
    bars_5m,
    trigger_level,
    direction,
    signal_date=None,
    opening_range_minutes=30,
    volume_multiple=1.2,
):
    """Evaluate the latest session for an entry in ``direction`` (+1 long, -1 short)."""

    if bars_5m is None or bars_5m.empty:
        return IntradayResult("unavailable", None, details=["No 5-minute data available (Yahoo keeps ~60 days)."])

    bars = bars_5m.sort_index().copy()
    bars["VWAP"] = session_vwap(bars)
    sessions = bars.index.normalize()
    session_date = sessions.max()
    session = bars.loc[sessions == session_date].copy()
    prior = bars.loc[sessions < session_date]

    if signal_date is not None and pd.Timestamp(session_date) < pd.Timestamp(signal_date).normalize():
        return IntradayResult("unavailable", None, details=["Intraday data ends before the signal date."])

    on_signal_day = signal_date is not None and pd.Timestamp(session_date) == pd.Timestamp(signal_date).normalize()

    opening_bars = max(1, int(opening_range_minutes // 5))
    if len(session) <= opening_bars:
        return IntradayResult(
            "pending",
            0.0,
            session_date=session_date.date(),
            details=[f"Opening range ({opening_range_minutes} min) not complete yet - wait for the trigger."],
            session_bars=session,
        )

    opening_range = session.iloc[:opening_bars]
    range_high = float(opening_range["High"].max())
    range_low = float(opening_range["Low"].min())

    # Relative volume per bar: compare with the same time-of-day in earlier sessions.
    if not prior.empty:
        time_of_day_average = prior.groupby(prior.index.time)["Volume"].mean()
        baseline = pd.Series(session.index.time, index=session.index).map(time_of_day_average)
        session["RVOL"] = session["Volume"] / baseline.replace(0, np.nan)
    else:
        session["RVOL"] = session["Volume"] / session["Volume"].expanding().mean()

    details = [
        f"Session {session_date:%Y-%m-%d}: opening range {range_low:,.2f}-{range_high:,.2f}, "
        f"daily trigger {trigger_level:,.2f}."
    ]

    after_range = session.iloc[opening_bars:]
    if direction > 0:
        level = max(range_high, trigger_level)
        triggered = after_range.loc[(after_range["Close"] > level) & (after_range["Close"] > after_range["VWAP"])]
    else:
        level = min(range_low, trigger_level)
        triggered = after_range.loc[(after_range["Close"] < level) & (after_range["Close"] < after_range["VWAP"])]

    bars_15m = resample_session_bars(session[["Open", "High", "Low", "Close", "Volume"]], "15min")
    fifteen_ok = None
    if len(bars_15m) >= 2:
        last_15 = bars_15m.iloc[-1]
        vwap_15 = float(session["VWAP"].iloc[-1])
        fast = ema(bars_15m["Close"], 9).iloc[-1]
        slow = ema(bars_15m["Close"], 20).iloc[-1]
        beyond_trigger = direction * (last_15["Close"] - trigger_level) > 0
        beyond_vwap = direction * (last_15["Close"] - vwap_15) > 0
        ema_aligned = direction * (fast - slow) > 0
        fifteen_ok = beyond_trigger and beyond_vwap
        details.append(
            f"15-min: last close {last_15['Close']:,.2f} is {'beyond' if beyond_trigger else 'NOT beyond'} the trigger, "
            f"{'on the right side of' if beyond_vwap else 'on the wrong side of'} VWAP {vwap_15:,.2f}; "
            f"EMA 9/20 {'aligned' if ema_aligned else 'not aligned'}."
        )

    result = IntradayResult(
        "pending",
        0.0,
        session_date=session_date.date(),
        opening_range_high=range_high,
        opening_range_low=range_low,
        details=details,
        session_bars=session,
    )

    if not triggered.empty:
        entry_bar = triggered.iloc[0]
        result.entry_time = triggered.index[0]
        result.entry_price = float(entry_bar["Close"])
        volume_ok = pd.notna(entry_bar["RVOL"]) and entry_bar["RVOL"] >= volume_multiple
        details.append(
            f"5-min trigger at {result.entry_time:%H:%M}: close {result.entry_price:,.2f} beyond "
            f"{'max' if direction > 0 else 'min'}(opening range, daily trigger) = {level:,.2f} and VWAP; "
            f"bar volume {entry_bar['RVOL']:.1f}x typical for that time of day."
        )
        if fifteen_ok is False:
            result.status = "triggered, 15-min not holding"
            result.score = 0.0
            details.append("The 5-min trigger fired, but the latest 15-min bar has not held - treat as unconfirmed.")
        else:
            result.status = "confirmed"
            result.score = 1.0 if volume_ok else 0.6
            if not volume_ok:
                details.append("Trigger bar volume was below the required multiple - partial confirmation only.")
    else:
        last_close = float(session["Close"].iloc[-1])
        last_vwap = float(session["VWAP"].iloc[-1])
        failed = direction * (last_close - trigger_level) < 0 and direction * (last_close - last_vwap) < 0
        if failed:
            result.status = "rejected"
            result.score = -0.75
            details.append(
                f"Price ({last_close:,.2f}) is back through the daily trigger and on the wrong side of VWAP - "
                "the breakout is failing intraday."
            )
        else:
            details.append("No 5-min entry trigger yet - wait for a close beyond the opening range and VWAP.")

    if on_signal_day:
        # The daily signal is only confirmed at this session's close, so the
        # entry belongs to the next session. Keep the evidence, halve its weight.
        result.score = None if result.score is None else result.score * 0.5
        result.status = f"signal day ({result.status}); enter next session"
        details.append("This session IS the daily signal bar - the entry trigger applies to the next session.")

    return result
