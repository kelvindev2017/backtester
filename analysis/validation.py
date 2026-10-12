"""Walk-forward validation of the signal engine.

For every historical day on which the selected setup fires, the full engine
is run on the data available up to that day only. The resulting trade plan
is then simulated on the following bars:

* entry at the next bar's open;
* exit at the stop or the target, whichever is touched first (the stop is
  assumed first when both fall inside the same bar - the conservative
  choice), at the open if price gaps through either level, or at the close
  after ``max_hold`` bars;
* the result is measured in R: profit or loss divided by the risk per share
  at entry. +1R means the trade made what it risked.

The summaries answer the questions that matter before trusting the tool:
does the strategy make money after realistic exits, and does a higher
confidence score actually mean better trades?
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.engine import AnalysisSettings, analyze
from analysis.indicators import atr, resample_ohlcv
from analysis.patterns import detect_signals
from analysis.risk import RiskSettings


TAKEN_ACTIONS = {"BUY", "SELL", "BUY SETUP", "SELL SETUP"}
CONFIDENCE_BINS = [0, 50, 60, 70, 80, 101]
COMPONENT_COLUMNS = [
    "Signal quality",
    "Trend alignment",
    "Volume / accumulation",
    "Fundamentals",
    "Historical edge",
]


@dataclass
class ValidationSettings:
    max_hold: int = 20
    horizons: tuple = (5, 10, 20)
    history_bars: int = 1260
    min_history_bars: int = 260
    random_entries_per_signal: int = 5


def simulate_trade(daily, signal_position, direction, stop, target, max_hold=20, risk_per_share=None):
    """Simulate one trade entered at the open after ``signal_position``.

    ``risk_per_share`` is the planned risk the position was sized on (signal
    close to stop). R is measured against it, because the order is sized
    before the next open; a gap toward the stop then shows up as a smaller
    loss or larger gain in R instead of shrinking the denominator. Without
    it, the risk from the actual entry is used.

    Returns a dict, or None when there is no next bar to enter on.
    """

    entry_position = signal_position + 1
    if entry_position >= len(daily):
        return None

    opens = daily["Open"].to_numpy(dtype=float)
    highs = daily["High"].to_numpy(dtype=float)
    lows = daily["Low"].to_numpy(dtype=float)
    closes = daily["Close"].to_numpy(dtype=float)
    entry = opens[entry_position]

    exit_price, exit_reason, exit_position = None, None, None
    last_position = min(len(daily), entry_position + max_hold) - 1

    for position in range(entry_position, last_position + 1):
        bar_open, bar_high, bar_low = opens[position], highs[position], lows[position]
        if direction > 0:
            if bar_open <= stop:
                exit_price, exit_reason = bar_open, "stop (gap)"
            elif bar_low <= stop:
                exit_price, exit_reason = stop, "stop"
            elif bar_open >= target:
                exit_price, exit_reason = bar_open, "target (gap)"
            elif bar_high >= target:
                exit_price, exit_reason = target, "target"
        else:
            if bar_open >= stop:
                exit_price, exit_reason = bar_open, "stop (gap)"
            elif bar_high >= stop:
                exit_price, exit_reason = stop, "stop"
            elif bar_open <= target:
                exit_price, exit_reason = bar_open, "target (gap)"
            elif bar_low <= target:
                exit_price, exit_reason = target, "target"
        if exit_price is not None:
            exit_position = position
            break

    complete = True
    if exit_price is None:
        exit_position = last_position
        exit_price = closes[last_position]
        complete = last_position - entry_position + 1 >= max_hold
        exit_reason = "time" if complete else "open (data ends)"

    if risk_per_share is not None and np.isfinite(risk_per_share) and risk_per_share > 0:
        risk = risk_per_share
    else:
        risk = direction * (entry - stop)
        if not np.isfinite(risk) or risk <= 0:
            # Price opened beyond the stop: the trade is stopped at the entry bar.
            risk = abs(entry - stop) or np.nan
    pnl = direction * (exit_price - entry)
    return {
        "Entry_Date": daily.index[entry_position],
        "Entry": entry,
        "Exit_Date": daily.index[exit_position],
        "Exit": exit_price,
        "Exit_Reason": exit_reason,
        "Bars_Held": exit_position - entry_position + 1,
        "Return": pnl / entry,
        "R": pnl / risk if np.isfinite(risk) and risk > 0 else np.nan,
        "Complete": complete,
    }


def walk_forward(
    symbol,
    daily,
    valuation_df=None,
    fundamentals_df=None,
    settings=None,
    risk_settings=None,
    start=None,
    end=None,
    validation_settings=None,
    progress=None,
):
    """Replay the engine on every signal day between ``start`` and ``end``.

    ``fundamentals_df`` must be indexed by release date (see
    ``assess_fundamentals``). ``progress`` is an optional callback taking a
    fraction in [0, 1].
    """

    settings = settings or AnalysisSettings()
    risk_settings = risk_settings or RiskSettings()
    validation_settings = validation_settings or ValidationSettings()
    if daily is None or daily.empty:
        return pd.DataFrame()

    signals = detect_signals(daily, settings.setup, settings.breakout_lookback, settings.volume_multiple)
    fired = (signals["Signal_Up"] | signals["Signal_Down"]).to_numpy()
    in_window = np.ones(len(daily), dtype=bool)
    if start is not None:
        in_window &= daily.index >= pd.Timestamp(start)
    if end is not None:
        in_window &= daily.index <= pd.Timestamp(end)
    positions = [
        position for position in np.flatnonzero(fired & in_window)
        if position >= validation_settings.min_history_bars and position + 1 < len(daily)
    ]

    closes = daily["Close"].to_numpy(dtype=float)
    opens = daily["Open"].to_numpy(dtype=float)
    atr_values = atr(daily).to_numpy(dtype=float)
    rows = []
    for count, position in enumerate(positions):
        if progress is not None:
            progress(count / max(1, len(positions)))
        history = daily.iloc[max(0, position - validation_settings.history_bars + 1): position + 1]
        as_of = history.index[-1]
        valuation = valuation_df.loc[:as_of] if valuation_df is not None else None
        report = analyze(
            symbol,
            history,
            resample_ohlcv(history),
            None,
            valuation,
            fundamentals_df,
            None,
            settings,
            risk_settings,
        )
        if report.signal is None or report.signal["date"] != as_of or report.plan is None:
            continue

        direction = report.direction
        trade = simulate_trade(
            daily,
            position,
            direction,
            report.plan.stop,
            report.plan.target,
            validation_settings.max_hold,
            risk_per_share=report.plan.risk_per_share,
        )
        if trade is None:
            continue

        row = {
            "Ticker": symbol,
            "Signal_Date": as_of,
            "Direction": direction,
            "Action": report.action,
            "Taken": report.action in TAKEN_ACTIONS,
            "Confidence": report.confidence,
            "Planned_RR": report.plan.reward_risk,
            "Stop": report.plan.stop,
            "Target": report.plan.target,
            "Signal_Close": closes[position],
            "Signal_ATR": atr_values[position],
        }
        row.update(trade)
        for horizon in validation_settings.horizons:
            exit_position = position + horizon
            row[f"Fwd_{horizon}"] = (
                direction * (closes[exit_position] / opens[position + 1] - 1) if exit_position < len(daily) else np.nan
            )
        for component in report.components:
            row[component.name] = component.score if component.available else np.nan
        rows.append(row)

    if progress is not None:
        progress(1.0)
    trades = pd.DataFrame(rows)
    if not trades.empty and validation_settings.random_entries_per_signal > 0:
        trades["Random_R"] = matched_random_r(
            daily,
            trades,
            in_window,
            validation_settings,
            seed=sum(ord(character) for character in symbol),
        )
    return trades


def matched_random_r(daily, trades, in_window, validation_settings, seed=0):
    """Average R of random entries that copy each signal's stop and target distances.

    For every replayed signal, ``random_entries_per_signal`` random days are
    drawn from the same ticker and test window. Each gets the same stop and
    target distance, measured in ATR, in the same direction, and is
    simulated identically. If the signal's timing has value, its R must beat
    this baseline - otherwise the result comes from market drift and the exit
    structure, not from the signal.
    """

    rng = np.random.default_rng(seed)
    closes = daily["Close"].to_numpy(dtype=float)
    atr_values = atr(daily).to_numpy(dtype=float)
    candidates = np.flatnonzero(in_window)
    candidates = candidates[
        (candidates >= validation_settings.min_history_bars)
        & (candidates < len(daily) - 1)
        & np.isfinite(atr_values[candidates])
        & (atr_values[candidates] > 0)
    ]
    if len(candidates) == 0:
        return pd.Series(np.nan, index=trades.index)

    results = []
    for _, trade in trades.iterrows():
        direction = trade["Direction"]
        signal_atr = trade["Signal_ATR"]
        if not np.isfinite(signal_atr) or signal_atr <= 0:
            results.append(np.nan)
            continue
        stop_atr = direction * (trade["Signal_Close"] - trade["Stop"]) / signal_atr
        target_atr = direction * (trade["Target"] - trade["Signal_Close"]) / signal_atr
        samples = []
        for position in rng.choice(candidates, size=validation_settings.random_entries_per_signal):
            close, atr_value = closes[position], atr_values[position]
            stop = close - direction * stop_atr * atr_value
            target = close + direction * target_atr * atr_value
            simulated = simulate_trade(
                daily, position, direction, stop, target, validation_settings.max_hold,
                risk_per_share=stop_atr * atr_value,
            )
            if simulated is not None and np.isfinite(simulated["R"]):
                samples.append(simulated["R"])
        results.append(float(np.mean(samples)) if samples else np.nan)
    return pd.Series(results, index=trades.index)


def base_rates(daily, start=None, end=None, horizon=10):
    """Long-side outcome of buying at *any* open in the window and holding ``horizon`` bars."""

    forward = daily["Close"].shift(-horizon) / daily["Open"].shift(-1) - 1
    mask = pd.Series(True, index=daily.index)
    if start is not None:
        mask &= daily.index >= pd.Timestamp(start)
    if end is not None:
        mask &= daily.index <= pd.Timestamp(end)
    values = forward[mask].dropna()
    return values


# ============================================================
# Summaries
# ============================================================

def trade_stats(trades):
    """Headline statistics for a set of simulated trades."""

    r = trades["R"].dropna() if "R" in trades else pd.Series(dtype=float)
    n = len(r)
    if n == 0:
        return {"Trades": 0}
    wins, losses = r[r > 0], r[r <= 0]
    std = r.std(ddof=1) if n > 1 else np.nan
    t_stat = r.mean() / (std / np.sqrt(n)) if n > 1 and std > 0 else np.nan
    reasons = trades.loc[r.index, "Exit_Reason"].str.split(" ").str[0]
    edge, edge_t, random_average = np.nan, np.nan, np.nan
    if "Random_R" in trades:
        paired = (trades.loc[r.index, "R"] - trades.loc[r.index, "Random_R"]).dropna()
        random_average = float(trades.loc[r.index, "Random_R"].mean())
        if len(paired) > 1 and paired.std(ddof=1) > 0:
            edge = float(paired.mean())
            edge_t = float(paired.mean() / (paired.std(ddof=1) / np.sqrt(len(paired))))
    return {
        "Trades": n,
        "Win rate": float((r > 0).mean()),
        "Average R": float(r.mean()),
        "Median R": float(r.median()),
        "t-stat (avg R)": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "Profit factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else np.nan,
        "Average return": float(trades.loc[r.index, "Return"].mean()),
        "Target hit": float((reasons == "target").mean()),
        "Stopped out": float((reasons == "stop").mean()),
        "Time exit": float((reasons == "time").mean()),
        "Avg bars held": float(trades.loc[r.index, "Bars_Held"].mean()),
        "Random-entry avg R": random_average,
        "Edge vs random (R)": edge,
        "t-stat (edge)": edge_t,
    }


def grouped_stats(trades, by):
    rows = []
    for key, group in trades.groupby(by, observed=True, dropna=False):
        stats = trade_stats(group)
        stats = {(by if isinstance(by, str) else "Group"): key, **stats}
        rows.append(stats)
    return pd.DataFrame(rows)


def rank_correlation(x, y):
    """Spearman rank correlation without SciPy."""

    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[mask].rank(), y[mask].rank())[0, 1])


def noise_band(n, trials=400, seed=0):
    """95th percentile of |rank correlation| between independent random series of length ``n``."""

    if n < 3:
        return np.nan
    rng = np.random.default_rng(seed)
    base = np.arange(n, dtype=float)
    values = [abs(np.corrcoef(base, rng.permutation(base))[0, 1]) for _ in range(trials)]
    return float(np.percentile(values, 95))


def component_predictiveness(trades):
    """How well each score ranks the trade outcomes (R)."""

    rows = []
    for column in COMPONENT_COLUMNS + ["Confidence"]:
        if column not in trades:
            continue
        available = trades[column].notna() & trades["R"].notna()
        n = int(available.sum())
        correlation = rank_correlation(trades.loc[available, column], trades.loc[available, "R"])
        band = noise_band(n)
        verdict = "n/a"
        if np.isfinite(correlation) and np.isfinite(band):
            if abs(correlation) <= band:
                verdict = "no detectable effect"
            elif correlation > 0:
                verdict = "helps (higher score -> better trades)"
            else:
                verdict = "HURTS (higher score -> worse trades)"
        rows.append(
            {"Score": column, "Trades": n, "Rank correlation with R": correlation, "95% noise band": band, "Verdict": verdict}
        )
    return pd.DataFrame(rows)


def threshold_table(trades, thresholds=(0, 50, 55, 60, 65, 70, 75, 80)):
    """Results if only signals at or above each confidence threshold were traded."""

    rows = []
    for threshold in thresholds:
        subset = trades.loc[trades["Confidence"].fillna(-1) >= threshold]
        stats = trade_stats(subset)
        rows.append({"Min confidence": threshold, **stats})
    return pd.DataFrame(rows)


def verdict(stats):
    """Plain-language judgement on whether the results show a real edge.

    When a random-entry baseline is available, the judgement is about the
    edge over random entries with the same stops and targets: positive R
    that random entries also achieve is market drift, not signal skill.
    """

    n = stats.get("Trades", 0)
    if n < 30:
        return "Too few trades to judge (fewer than 30). Add tickers or widen the date range."
    edge, edge_t = stats.get("Edge vs random (R)", np.nan), stats.get("t-stat (edge)", np.nan)
    if edge is not None and np.isfinite(edge_t):
        average, random_average = stats.get("Average R", np.nan), stats.get("Random-entry avg R", np.nan)
        context = (
            f"signals averaged {average:+.2f}R per trade vs {random_average:+.2f}R for random entry days with "
            f"the same stops and targets"
        )
        if edge_t >= 2:
            return f"Positive edge: {context} - an edge of {edge:+.2f}R that is unlikely to be luck (t = {edge_t:.1f})."
        if edge_t <= -2:
            return f"Negative edge: {context} - the signal timing is reliably worse than random (t = {edge_t:.1f})."
        return (
            f"No reliable edge: {context}. The difference ({edge:+.2f}R, t = {edge_t:.1f}) is within luck; "
            "any profit comes from market drift and the exit rules, not from the signal timing."
        )
    t_stat = stats.get("t-stat (avg R)", np.nan)
    average = stats.get("Average R", np.nan)
    if not np.isfinite(t_stat):
        return "Results are inconclusive."
    if t_stat >= 2:
        return f"Positive edge: average {average:+.2f}R per trade, statistically distinguishable from zero (t = {t_stat:.1f})."
    if t_stat <= -2:
        return f"Negative edge: average {average:+.2f}R per trade, reliably losing (t = {t_stat:.1f})."
    return (
        f"No reliable edge: average {average:+.2f}R per trade cannot be distinguished from luck (t = {t_stat:.1f}; "
        "|t| >= 2 is needed)."
    )


def exit_signal_stats(trades, base_forward=None, horizon=10):
    """How the stock moved after bearish signals, i.e. the value of exiting a long on them."""

    bearish = trades.loc[trades["Direction"] < 0]
    column = f"Fwd_{horizon}"
    if bearish.empty or column not in bearish:
        return {}
    stock_move = -bearish[column].dropna()  # Fwd is oriented to the short side.
    result = {
        "Bearish signals": int(len(stock_move)),
        f"Stock {horizon}-bar return after signal": float(stock_move.mean()),
        "Share of signals followed by a rise": float((stock_move > 0).mean()),
    }
    if base_forward is not None and len(base_forward):
        result[f"Any-day {horizon}-bar return"] = float(base_forward.mean())
        result["Exit helped vs holding"] = bool(stock_move.mean() < base_forward.mean())
    return result


def summarize(trades):
    """Bundle every summary table used by the app."""

    if trades is None or trades.empty:
        return {}
    trades = trades.copy()
    trades["Confidence bucket"] = pd.cut(trades["Confidence"], CONFIDENCE_BINS, right=False)
    trades["Year"] = pd.to_datetime(trades["Signal_Date"]).dt.year
    taken = trades.loc[trades["Taken"]]
    return {
        "exits": exit_signal_stats(trades),
        "all": trade_stats(trades),
        "taken": trade_stats(taken),
        "by_action": grouped_stats(trades, "Action"),
        "by_direction": grouped_stats(trades, "Direction"),
        "by_confidence": grouped_stats(trades, "Confidence bucket"),
        "by_year": grouped_stats(trades, "Year"),
        "by_ticker": grouped_stats(trades, "Ticker"),
        "thresholds": threshold_table(taken),
        "components": component_predictiveness(trades),
    }
