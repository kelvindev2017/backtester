"""Combine pattern, volume, intraday, fundamental and risk analysis into a
single pre-trade report with an action, a confidence level and reasoning.

Workflow
--------
1. Daily (and weekly) bars: is there a fresh, volume-confirmed breakout or
   breakdown, and does it agree with the daily/weekly trend?
2. Volume: is there accumulation (for a buy) or distribution (for a sell)?
3. Intraday: on the session after the daily signal, has the 5-minute /
   15-minute entry trigger fired?
4. Fundamentals: do valuation and growth support the direction?
5. History: how did the same signal perform on this ticker before?
6. Risk: stop beyond the nearest support/resistance zone, target at the
   next zone, position size from the account risk budget and confidence.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis.fundamentals import assess_fundamentals
from analysis.indicators import atr
from analysis.intraday import evaluate_intraday_entry
from analysis.levels import PriceZone, support_resistance, zone_near
from analysis.patterns import (
    assess_trend,
    assess_volume,
    breakout_quality,
    detect_breakouts,
    detect_signals,
    historical_edge_score,
    historical_signal_stats,
    describe_ma_rule,
    latest_active_signal,
    ma_rule_quality,
    pullback_quality,
    reversal_quality,
)
from analysis.risk import RiskSettings, plan_trade
from analysis.scoring import COMPONENT_WEIGHTS, ComponentResult, combine_confidence


@dataclass
class AnalysisSettings:
    setup: str = "breakout"
    allow_short: bool = False
    breakout_lookback: int = 20
    weekly_breakout_lookback: int = 10
    volume_multiple: float = 1.5
    require_volume_confirmation: bool = True
    signal_window: int = 3
    level_lookback: int = 250
    pivot_bars: int = 5
    level_tolerance_atr: float = 0.5
    opening_range_minutes: int = 30
    intraday_volume_multiple: float = 1.2
    history_horizon: int = 10
    ma_rule: dict = None


@dataclass
class SignalReport:
    symbol: str
    as_of: pd.Timestamp
    action: str
    direction: int = 0
    confidence: float = None
    coverage: float = 0.0
    headline: str = ""
    signal: dict = None
    trends: list = field(default_factory=list)
    components: list = field(default_factory=list)
    plan: object = None
    intraday: object = None
    supports: list = field(default_factory=list)
    resistances: list = field(default_factory=list)
    history_summary: pd.DataFrame = None
    history_events: pd.DataFrame = None
    signals: pd.DataFrame = None
    reasons_for: list = field(default_factory=list)
    reasons_against: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


SETUP_NAMES = {
    "breakout": ("breakout above", "breakdown below"),
    "pullback": ("pullback entry above", "bounce-failure entry below"),
    "reversal": ("reversal buy after breakdown at", "-"),
    "ma_rule": ("MA rule buy at", "MA rule sell at"),
}


def _direction_word(direction, allow_short=True):
    if direction > 0:
        return "BUY"
    return "SELL" if allow_short else "EXIT"


def analyze(
    symbol,
    daily,
    weekly=None,
    bars_5m=None,
    valuation_df=None,
    fundamentals_df=None,
    snapshot=None,
    settings=None,
    risk_settings=None,
):
    """Run the full analysis on data that ends at the analysis date."""

    settings = settings or AnalysisSettings()
    risk_settings = risk_settings or RiskSettings()

    if daily is None or len(daily) < max(60, settings.breakout_lookback + 15):
        return SignalReport(symbol, None, "INSUFFICIENT DATA", headline="At least ~60 daily bars are needed.")

    as_of = pd.Timestamp(daily.index[-1])
    signals = detect_signals(
        daily, settings.setup, settings.breakout_lookback, settings.volume_multiple, ma_rule=settings.ma_rule
    )
    atr_value = float(atr(daily).iloc[-1])
    supports, resistances = support_resistance(
        daily,
        atr_value,
        lookback=settings.level_lookback,
        pivot_bars=settings.pivot_bars,
        tolerance_atr=settings.level_tolerance_atr,
    )

    daily_trend = assess_trend(daily, "Daily", fast=50, slow=200, slope_bars=20, pivot_bars=settings.pivot_bars)
    trends = [daily_trend]
    weekly_breakouts = None
    if weekly is not None and len(weekly) >= 20:
        weekly_trend = assess_trend(weekly, "Weekly", fast=10, slow=40, slope_bars=4, pivot_bars=3)
        trends.append(weekly_trend)
        weekly_breakouts = detect_breakouts(
            weekly, settings.weekly_breakout_lookback, volume_period=20, volume_multiple=settings.volume_multiple
        )

    report = SignalReport(
        symbol=symbol,
        as_of=as_of,
        action="NO SIGNAL",
        trends=trends,
        supports=supports,
        resistances=resistances,
        signals=signals,
    )

    signal = latest_active_signal(daily, signals, settings.signal_window)
    report.signal = signal

    if signal is None:
        last = signals.iloc[-1]
        if settings.setup == "ma_rule":
            report.headline = (
                f"No moving-average rule signal in the last {settings.signal_window} bars. A BUY needs "
                f"{describe_ma_rule(settings.ma_rule, 'buy')}; a {_direction_word(-1, settings.allow_short)} needs "
                f"{describe_ma_rule(settings.ma_rule, 'sell')}."
            )
        elif settings.setup == "reversal":
            report.headline = (
                f"No reversal buy in the last {settings.signal_window} bars. A BUY needs a long-term uptrend "
                f"(close above the 200-bar MA, 50-bar MA above it) and a daily close below the prior "
                f"{settings.breakout_lookback}-bar low (currently {last['Prior_Low']:,.2f})."
            )
        elif settings.setup == "pullback":
            report.headline = (
                f"No pullback entry in the last {settings.signal_window} bars. A BUY needs an uptrend (50-bar MA above "
                "a rising 200-bar MA), a dip of at least 1.5 ATR toward the 20-bar EMA, then a close above the prior "
                "day's high."
            )
        else:
            report.headline = (
                f"No fresh breakout in the last {settings.signal_window} bars. Watch a daily close above "
                f"{last['Trigger_Up']:,.2f} (BUY trigger) or below {last['Trigger_Down']:,.2f} "
                f"({_direction_word(-1, settings.allow_short)} trigger) on at least {settings.volume_multiple:g}x average volume."
            )
        return report

    direction = signal["direction"]
    report.direction = direction
    word = _direction_word(direction, settings.allow_short)
    signal_row = signals.loc[signal["date"]]
    signal_close = float(daily.at[signal["date"], "Close"])
    setup_phrase = SETUP_NAMES.get(settings.setup, SETUP_NAMES["breakout"])[0 if direction > 0 else 1]

    if not signal["holding"]:
        report.action = "NO TRADE"
        report.headline = (
            f"Failed signal: price is back through {signal['invalidation']:,.2f} since the "
            f"{signal['date']:%Y-%m-%d} {setup_phrase.split(' ')[0]} - the setup is invalidated."
        )
        report.warnings.append(report.headline)
        return report

    components = []

    # 1. Signal quality --------------------------------------------------
    if settings.setup == "ma_rule":
        quality, quality_details, trigger = ma_rule_quality(signal_row, signal_close, direction, settings.ma_rule)
        signal_text = (
            f"Moving-average rule fired on {signal['date']:%Y-%m-%d} ({signal['bars_ago']} bar(s) ago) at "
            f"{trigger:,.2f}, invalidated {'below' if direction > 0 else 'above'} {signal['invalidation']:,.2f}."
        )
        quality_metrics = {"RVOL": float(signal_row["RVOL"])}
    elif settings.setup == "reversal":
        quality, quality_details, trigger = reversal_quality(signal_row, signal_close, direction)
        signal_text = (
            f"Daily breakdown inside an uptrend on {signal['date']:%Y-%m-%d} ({signal['bars_ago']} bar(s) ago): "
            f"a short-term reversal buy, invalidated below {signal['invalidation']:,.2f}."
        )
        quality_metrics = {"Drop over 5 bars (ATR)": float(signal_row["Drop_ATR"]), "RVOL": float(signal_row["RVOL"])}
    elif settings.setup == "pullback":
        quality, quality_details, trigger = pullback_quality(signal_row, signal_close, direction)
        signal_text = (
            f"Daily {setup_phrase} {signal['trigger']:,.2f} on {signal['date']:%Y-%m-%d} "
            f"({signal['bars_ago']} bar(s) ago), invalidated below {signal['invalidation']:,.2f}."
            if direction > 0 else
            f"Daily {setup_phrase} {signal['trigger']:,.2f} on {signal['date']:%Y-%m-%d} "
            f"({signal['bars_ago']} bar(s) ago), invalidated above {signal['invalidation']:,.2f}."
        )
        quality_metrics = {"Pullback RVOL": float(signal_row["Pullback_RVOL"])}
    else:
        quality, quality_details, trigger = breakout_quality(signal_row, signal_close, direction)
        signal_text = (
            f"Daily {settings.breakout_lookback}-bar {'breakout' if direction > 0 else 'breakdown'} on "
            f"{signal['date']:%Y-%m-%d} ({signal['bars_ago']} bar(s) ago) through {trigger:,.2f}."
        )
        quality_metrics = {"RVOL": float(signal_row["RVOL"]), "Base width (ATR)": float(signal_row["Base_Width_ATR"])}
    bonus = 0.0
    tolerance = settings.level_tolerance_atr * atr_value
    known_level = zone_near([zone for zone in supports + resistances if zone.source == "pivot"], trigger, tolerance)
    if known_level is not None and settings.setup == "breakout":
        bonus += 0.15
        quality_details.append(f"The trigger coincides with a known {known_level.touches}-touch level {known_level.describe()}.")
    if weekly_breakouts is not None and settings.setup == "breakout":
        column = "Breakout_Up" if direction > 0 else "Breakout_Down"
        if weekly_breakouts[column].iloc[-2:].any():
            bonus += 0.15
            quality_details.append(f"The weekly chart also shows a fresh {settings.weekly_breakout_lookback}-week {'breakout' if direction > 0 else 'breakdown'}.")
    components.append(
        ComponentResult(
            "Signal quality",
            None if quality is None else float(np.clip(quality + bonus, -1, 1)),
            [signal_text] + quality_details,
            quality_metrics,
        )
    )

    # 2. Trend alignment -------------------------------------------------
    trend_scores = [(trend.score, 0.4 if trend.timeframe == "Daily" else 0.6) for trend in trends if np.isfinite(trend.score)]
    trend_score = None
    if trend_scores:
        trend_score = direction * sum(s * w for s, w in trend_scores) / sum(w for _, w in trend_scores)
    trend_details = [f"{trend.timeframe} trend: {trend.label} (score {trend.score:+.2f})." for trend in trends if np.isfinite(trend.score)]
    for trend in trends:
        trend_details.extend(trend.details)
    components.append(ComponentResult("Trend alignment", trend_score, trend_details))

    # 3. Volume / accumulation --------------------------------------------
    volume_score, volume_details, volume_metrics = assess_volume(daily)
    components.append(
        ComponentResult(
            "Volume / accumulation",
            None if not np.isfinite(volume_score) else direction * volume_score,
            volume_details,
            volume_metrics,
        )
    )

    # 4. Intraday entry ------------------------------------------------------
    intraday = None
    if bars_5m is not None and not bars_5m.empty:
        intraday = evaluate_intraday_entry(
            bars_5m,
            trigger,
            direction,
            signal_date=signal["date"],
            opening_range_minutes=settings.opening_range_minutes,
            volume_multiple=settings.intraday_volume_multiple,
        )
        components.append(ComponentResult("Intraday entry", intraday.score, [f"Status: {intraday.status}."] + intraday.details))
    else:
        components.append(ComponentResult("Intraday entry", None, ["No 5-minute data for the session after the signal."]))
    report.intraday = intraday

    # 5. Fundamentals -------------------------------------------------------
    if valuation_df is not None:
        fundamental_score, fundamental_details, fundamental_metrics = assess_fundamentals(
            as_of, valuation_df, fundamentals_df, snapshot
        )
    else:
        fundamental_score, fundamental_details, fundamental_metrics = None, ["No earnings / valuation data supplied."], {}
    if fundamental_score is not None:
        fundamental_details.insert(
            0,
            f"Fundamental score {fundamental_score:+.2f} (bullish-positive) "
            f"{'supports' if direction * fundamental_score > 0 else 'argues against'} a {word}.",
        )
    components.append(
        ComponentResult(
            "Fundamentals",
            None if fundamental_score is None else direction * fundamental_score,
            fundamental_details,
            fundamental_metrics,
        )
    )

    # 6. Historical edge of this signal on this ticker ---------------------
    column = "Signal_Up" if direction > 0 else "Signal_Down"
    mask = signals[column]
    if settings.require_volume_confirmation:
        mask = mask & signals["Volume_Confirmed"].fillna(False).astype(bool)
    past_dates = signals.index[mask.to_numpy()]
    past_dates = past_dates[past_dates < signal["date"]]
    events, summary = historical_signal_stats(daily, past_dates, direction)
    edge_score, edge_details = historical_edge_score(summary, settings.history_horizon)
    report.history_events, report.history_summary = events, summary
    components.append(ComponentResult("Historical edge", edge_score, edge_details))

    confidence, coverage = combine_confidence(components, COMPONENT_WEIGHTS)
    report.components, report.confidence, report.coverage = components, confidence, coverage

    # Trade plan -----------------------------------------------------------
    if intraday is not None and intraday.status == "confirmed" and intraday.entry_price:
        entry = intraday.entry_price
        entry_basis = f"5-min trigger at {intraday.entry_time:%Y-%m-%d %H:%M}"
    else:
        entry = float(daily["Close"].iloc[-1])
        entry_basis = "last daily close (indicative - final entry comes from the intraday trigger)"

    if settings.setup in ("pullback", "reversal", "ma_rule"):
        # The pullback / signal-bar extreme is the level that invalidates the setup.
        invalidation = signal["invalidation"]
        broken_level = PriceZone(
            low=invalidation if direction > 0 else invalidation - 0.25 * atr_value,
            high=invalidation + 0.25 * atr_value if direction > 0 else invalidation,
            price=invalidation,
            touches=1,
            strength=1.0,
            source="pullback",
        )
        # Stop goes beyond the invalidation level even if a nearer zone exists.
        supports = [zone for zone in supports if zone.price < invalidation] if direction > 0 else supports
        resistances = [zone for zone in resistances if zone.price > invalidation] if direction < 0 else resistances
    else:
        broken_level = PriceZone(
            low=trigger - 0.25 * atr_value,
            high=trigger + 0.25 * atr_value,
            price=trigger,
            touches=1,
            strength=1.0,
            source="breakout",
        )
    if direction > 0:
        stop_zones = sorted([zone for zone in supports + [broken_level] if zone.price < entry], key=lambda z: z.price, reverse=True)
        plan = plan_trade(direction, entry, atr_value, stop_zones, resistances, risk_settings, confidence)
    else:
        stop_zones = sorted([zone for zone in resistances + [broken_level] if zone.price > entry], key=lambda z: z.price)
        plan = plan_trade(direction, entry, atr_value, supports, stop_zones, risk_settings, confidence)
    plan.notes.insert(0, f"Entry basis: {entry_basis}.")
    report.plan = plan

    # Action ------------------------------------------------------------------
    volume_ok = bool(signal_row["Volume_Confirmed"])
    intraday_status = intraday.status if intraday is not None else "unavailable"
    if settings.require_volume_confirmation and not volume_ok:
        report.action = "WATCH"
        report.warnings.append(
            "Volume during the pullback was heavy - sellers may still be in control."
            if settings.setup == "pullback" else
            "Breakout volume is below the confirmation threshold - price-only breakouts fail more often."
        )
    elif confidence is None or confidence < risk_settings.half_size_confidence:
        report.action = "NO TRADE"
    elif intraday_status == "rejected":
        report.action = "NO TRADE"
        report.warnings.append("Intraday price action rejected the breakout level.")
    elif intraday_status == "confirmed":
        report.action = word
    else:
        report.action = f"{word} SETUP"

    size_text = {1.0: "full size", 0.5: "half size", 0.0: "no position"}.get(plan.size_multiplier, "")
    if report.action in ("WATCH", "NO TRADE") and plan.shares > 0:
        # Not actionable yet: keep the levels, but do not present a live position size.
        plan.notes.append(
            f"Size once the signal is confirmed: {plan.shares:,} shares ({size_text}), "
            f"${plan.dollar_risk:,.0f} at risk."
        )
        plan.shares, plan.position_value, plan.dollar_risk = 0, 0.0, 0.0
        size_text = "no position until confirmed" if report.action == "WATCH" else "no position"
    if direction < 0 and not settings.allow_short:
        # Long-only mode: a bearish signal means exit or avoid, never a new short.
        if plan.shares > 0:
            plan.notes.append("Long-only mode: no short position is opened; the levels are for reference.")
        plan.shares, plan.position_value, plan.dollar_risk = 0, 0.0, 0.0
        size_text = "close or reduce existing longs; no new short"
    report.headline = (
        f"{report.action}: {symbol} {setup_phrase} {trigger:,.2f} "
        f"- confidence {confidence:.0f}/100 ({size_text})."
        if confidence is not None
        else f"{report.action}: {symbol}"
    )
    if report.action.endswith("SETUP"):
        report.headline += " Wait for the 5-min / 15-min entry trigger before executing."
    if coverage < 0.75:
        report.warnings.append(
            f"Only {coverage * 100:.0f}% of the confidence model's inputs were available; treat the score with caution."
        )
    if plan.blocking_zone is not None:
        report.warnings.append(
            f"Limited room: entry is pressed against {'resistance' if direction > 0 else 'support'} "
            f"{plan.blocking_zone.describe()}. Prefer an entry after it is cleared, or a pullback toward the trigger."
        )
    if risk_settings.min_reward_risk and np.isfinite(plan.reward_risk) and plan.reward_risk < risk_settings.min_reward_risk:
        report.warnings.append(f"Reward/risk {plan.reward_risk:.2f} is below the {risk_settings.min_reward_risk:g} minimum.")

    for component in components:
        if not component.available:
            continue
        summary_line = component.details[0] if component.details else component.name
        if component.score >= 0.25:
            report.reasons_for.append(f"{component.name} ({component.score:+.2f}): {summary_line}")
        elif component.score <= -0.1:
            report.reasons_against.append(f"{component.name} ({component.score:+.2f}): {summary_line}")

    return report


# ============================================================
# Rendering
# ============================================================

def _format_metric(value):
    if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
        return f"{value:,.3f}" if abs(value) < 10 else f"{value:,.1f}"
    return str(value)


def report_to_markdown(report):
    """Plain Markdown version of the report, for download or notifications."""

    lines = [f"# {report.symbol} signal report - {report.action}", ""]
    if report.as_of is not None:
        lines.append(f"*Data as of {report.as_of:%Y-%m-%d}.*")
        lines.append("")
    lines.append(f"**{report.headline}**")
    lines.append("")
    if report.confidence is not None:
        lines.append(f"- Confidence: **{report.confidence:.0f}/100** (inputs available: {report.coverage * 100:.0f}%)")
    for trend in report.trends:
        if np.isfinite(trend.score):
            lines.append(f"- {trend.timeframe} trend: {trend.label} ({trend.score:+.2f})")
    lines.append("")

    if report.plan is not None:
        plan = report.plan
        lines += [
            "## Trade plan",
            "",
            f"| Item | Value |",
            f"|---|---|",
            f"| Side | {plan.side} |",
            f"| Entry | {plan.entry:,.2f} |",
            f"| Stop | {plan.stop:,.2f} - {plan.stop_basis} |",
            f"| Target | {plan.target:,.2f} - {plan.target_basis} |",
            f"| Reward / risk | {plan.reward_risk:.2f} |",
            f"| Shares | {plan.shares:,} ({plan.size_multiplier:g}x size) |",
            f"| Position value | ${plan.position_value:,.0f} |",
            f"| Dollar risk | ${plan.dollar_risk:,.0f} |",
            "",
        ]
        lines += [f"- {note}" for note in plan.notes]
        lines.append("")

    if report.reasons_for:
        lines += ["## Supporting evidence", ""] + [f"- {reason}" for reason in report.reasons_for] + [""]
    if report.reasons_against or report.warnings:
        lines += ["## Risks and conflicts", ""]
        lines += [f"- {reason}" for reason in report.reasons_against + report.warnings]
        lines.append("")

    if report.components:
        lines += ["## Component scores", "", "| Component | Weight | Score |", "|---|---|---|"]
        for component in report.components:
            score = f"{component.score:+.2f}" if component.available else "n/a"
            lines.append(f"| {component.name} | {COMPONENT_WEIGHTS.get(component.name, 0) * 100:.0f}% | {score} |")
        lines.append("")
        for component in report.components:
            lines += [f"### {component.name}", ""] + [f"- {detail}" for detail in component.details]
            if component.metrics:
                lines += [f"- {key}: {_format_metric(value)}" for key, value in component.metrics.items()]
            lines.append("")

    if report.supports or report.resistances:
        lines += ["## Support / resistance zones", ""]
        lines += [f"- Resistance: {zone.describe()}" for zone in report.resistances[:4]]
        lines += [f"- Support: {zone.describe()}" for zone in report.supports[:4]]
        lines.append("")

    lines.append("*Generated by the backtester signal engine. Not investment advice; verify before trading.*")
    return "\n".join(lines)


def report_to_dict(report):
    """JSON-serialisable summary of the report."""

    plan = report.plan
    return {
        "symbol": report.symbol,
        "as_of": None if report.as_of is None else report.as_of.strftime("%Y-%m-%d"),
        "action": report.action,
        "direction": report.direction,
        "confidence": report.confidence,
        "coverage": round(report.coverage, 3),
        "headline": report.headline,
        "trends": {trend.timeframe: {"label": trend.label, "score": None if not np.isfinite(trend.score) else round(trend.score, 3)} for trend in report.trends},
        "components": {
            component.name: None if not component.available else round(component.score, 3)
            for component in report.components
        },
        "plan": None if plan is None else {
            "side": plan.side,
            "entry": round(plan.entry, 4),
            "stop": round(plan.stop, 4),
            "target": round(plan.target, 4),
            "reward_risk": round(plan.reward_risk, 3) if np.isfinite(plan.reward_risk) else None,
            "shares": plan.shares,
            "dollar_risk": round(plan.dollar_risk, 2),
            "stop_basis": plan.stop_basis,
            "target_basis": plan.target_basis,
        },
        "reasons_for": report.reasons_for,
        "reasons_against": report.reasons_against,
        "warnings": report.warnings,
    }
