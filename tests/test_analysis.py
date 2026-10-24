import numpy as np
import pandas as pd
import pytest

from analysis.engine import AnalysisSettings, analyze, report_to_dict, report_to_markdown
from analysis.fundamentals import assess_fundamentals
from analysis.intraday import evaluate_intraday_entry, resample_session_bars, session_vwap
from analysis.levels import PriceZone, find_pivots, support_resistance
from analysis.notify import build_webhook_payload, short_summary
from analysis.patterns import (
    assess_trend,
    detect_breakouts,
    historical_edge_score,
    historical_signal_stats,
    latest_active_signal,
)
from analysis.risk import RiskSettings, plan_trade, size_multiplier_for
from analysis.scoring import ComponentResult, combine_confidence
from tests.conftest import make_bars, make_session, to_weekly


# ------------------------------------------------------------------ patterns

def test_breakout_detection_has_no_lookahead(breakout_daily):
    full = detect_breakouts(breakout_daily)
    for cut in [120, 200, 300, len(breakout_daily)]:
        partial = detect_breakouts(breakout_daily.iloc[:cut])
        pd.testing.assert_frame_equal(partial.iloc[-1:], full.iloc[cut - 1:cut])


def test_latest_signal_is_volume_confirmed_upside_breakout(breakout_daily):
    breakouts = detect_breakouts(breakout_daily)
    signal = latest_active_signal(breakout_daily, breakouts, window=3)
    assert signal is not None
    assert signal["direction"] == 1
    assert signal["bars_ago"] == 0
    assert signal["holding"]
    assert 104 < signal["trigger"] < 107.5
    assert bool(breakouts["Volume_Confirmed"].iloc[-1])


def test_failed_breakout_is_flagged(breakout_daily):
    failed = pd.concat([breakout_daily, make_bars([101.0], [1e6], index=[breakout_daily.index[-1] + pd.offsets.BDay()])])
    signal = latest_active_signal(failed, detect_breakouts(failed), window=3)
    assert signal["direction"] == 1 and not signal["holding"]


def test_trend_classification():
    up = make_bars(np.linspace(50, 150, 300) + np.sin(np.arange(300) / 4) * 2)
    down = make_bars(np.linspace(150, 50, 300) + np.sin(np.arange(300) / 4) * 2)
    assert assess_trend(up).label == "Uptrend"
    assert assess_trend(down).label == "Downtrend"


def test_historical_stats_enter_next_open():
    bars = make_bars(np.arange(100, 130, dtype=float))
    events, summary = historical_signal_stats(bars, [bars.index[5]], direction=1, horizons=(5,))
    expected = bars["Close"].iloc[10] / bars["Open"].iloc[6] - 1
    assert events["Return_5"].iloc[0] == pytest.approx(expected)
    assert summary["Win rate"].iloc[0] == 1.0


def test_historical_edge_needs_samples():
    summary = pd.DataFrame([{"Horizon (bars)": 10, "Signals": 2, "Win rate": 1.0, "Average return": 0.05,
                             "Median return": 0.05, "Base win rate (any day)": 0.5, "Base average return": 0.0}])
    score, _ = historical_edge_score(summary)
    assert score is None


# ------------------------------------------------------------------ levels

def test_pivots_are_confirmed_after_right_bars():
    bars = make_bars([1, 2, 3, 4, 5, 9, 5, 4, 3, 2, 1, 2, 3], spread=0.0)
    pivots = find_pivots(bars, left=3, right=3)
    high = pivots.loc[pivots["Type"] == "high"].iloc[0]
    assert high["Price"] == pytest.approx(9)
    assert high["Confirmed_At"] == bars.index[5 + 3]


def test_range_produces_support_and_resistance():
    rng = np.random.default_rng(7)
    closes = 100 + 5 * np.sin(np.linspace(0, 10 * np.pi, 250)) + rng.normal(0, 0.3, 250)
    bars = make_bars(closes, seed=7)
    supports, resistances = support_resistance(bars, atr_value=1.5)
    assert any(93 < zone.price < 97 for zone in supports)
    assert any(103 < zone.price < 107 for zone in resistances)


# ------------------------------------------------------------------ risk

def test_plan_trade_long_uses_support_and_sizes_by_risk():
    supports = [PriceZone(low=95, high=96, price=95.5, touches=3, strength=3, source="pivot")]
    resistances = [PriceZone(low=110, high=111, price=110.5, touches=2, strength=2, source="pivot")]
    settings = RiskSettings(account_equity=100_000, risk_pct=1.0, max_position_pct=100)
    plan = plan_trade(1, 100.0, 2.0, supports, resistances, settings, confidence=80)
    assert plan.stop == pytest.approx(94.0)  # 95 - 0.5 * ATR
    assert plan.target == pytest.approx(110.0)
    assert plan.shares == 1000 // 6
    assert plan.dollar_risk <= 1000
    assert plan.reward_risk == pytest.approx(10 / 6)


def test_plan_trade_caps_position_and_falls_back_to_atr_stop():
    settings = RiskSettings(account_equity=100_000, risk_pct=2.0, max_position_pct=10)
    plan = plan_trade(1, 50.0, 0.5, [], [], settings, confidence=90)
    assert plan.stop == pytest.approx(49.0)
    assert plan.shares == 200  # 10% cap: 10,000 / 50
    assert "2R" in plan.target_basis


def test_plan_trade_short_mirrors_long():
    resistances = [PriceZone(low=104, high=105, price=104.5, touches=2, strength=2, source="pivot")]
    supports = [PriceZone(low=90, high=91, price=90.5, touches=2, strength=2, source="pivot")]
    plan = plan_trade(-1, 100.0, 2.0, supports, resistances, RiskSettings(), confidence=80)
    assert plan.stop == pytest.approx(106.0)
    assert plan.target == pytest.approx(91.0)


def test_size_tiers():
    settings = RiskSettings()
    assert size_multiplier_for(80, settings) == 1.0
    assert size_multiplier_for(65, settings) == 0.5
    assert size_multiplier_for(50, settings) == 0.0
    assert size_multiplier_for(None, settings) == 0.0


# ------------------------------------------------------------------ scoring

def test_confidence_renormalises_missing_components():
    components = [ComponentResult("Signal quality", 1.0), ComponentResult("Intraday entry", None)]
    confidence, coverage = combine_confidence(components)
    assert confidence == 100.0
    assert coverage == pytest.approx(0.25 / 0.40)


# ------------------------------------------------------------------ intraday

def test_vwap_resets_each_session(intraday_breakout):
    vwap = session_vwap(intraday_breakout)
    first_bar = intraday_breakout.index.normalize() != intraday_breakout.index.normalize().to_series().shift(1).to_numpy()
    typical = (intraday_breakout["High"] + intraday_breakout["Low"] + intraday_breakout["Close"]) / 3
    assert np.allclose(vwap[first_bar], typical[first_bar])


def test_fifteen_minute_resample_stays_inside_session(intraday_breakout):
    bars_15 = resample_session_bars(intraday_breakout)
    assert bars_15.index.min().strftime("%H:%M") == "09:30"
    assert bars_15.groupby(bars_15.index.normalize()).size().max() == 26


def test_intraday_confirmed_entry(intraday_breakout):
    result = evaluate_intraday_entry(intraday_breakout, trigger_level=107.0, direction=1, signal_date=pd.Timestamp("2024-03-05"))
    assert result.status == "confirmed"
    assert result.score == 1.0
    assert result.entry_time.strftime("%H:%M") == "10:00"
    assert result.entry_price > max(107.0, result.opening_range_high)


def test_intraday_rejection():
    prior = make_session("2024-03-05", np.full(78, 105.0))
    fade = make_session("2024-03-06", np.linspace(107.5, 103.0, 40))
    result = evaluate_intraday_entry(pd.concat([prior, fade]), 106.0, 1, signal_date=pd.Timestamp("2024-03-05"))
    assert result.status == "rejected"
    assert result.score < 0


def test_intraday_on_signal_day_is_half_weight(intraday_breakout):
    result = evaluate_intraday_entry(intraday_breakout, 107.0, 1, signal_date=pd.Timestamp("2024-03-06"))
    assert result.status.startswith("signal day")
    assert result.score == 0.5


# ------------------------------------------------------------------ fundamentals

def _valuation(closes, eps):
    index = pd.bdate_range("2021-01-01", periods=len(closes))
    frame = pd.DataFrame({"Close": closes, "TTM_Reported_EPS": eps}, index=index)
    frame["Trailing_PE"] = frame["Close"] / frame["TTM_Reported_EPS"]
    return frame


def test_fundamentals_reward_growth_at_reasonable_price():
    n = 800
    eps = np.linspace(2.0, 4.0, n)          # EPS doubling
    closes = eps * np.linspace(30, 22, n)   # multiple compressing
    score, details, metrics = assess_fundamentals(pd.Timestamp("2030-01-01"), _valuation(closes, eps))
    assert score > 0.3
    assert metrics["PEG"] < 1.5
    assert any("driven by earnings" in detail for detail in details)


def test_fundamentals_penalise_expensive_shrinking_earnings():
    n = 800
    eps = np.linspace(4.0, 3.0, n)
    closes = eps * np.linspace(15, 45, n)
    snapshot = {"forwardPE": 60.0, "trailingPegRatio": 4.0}
    score, _, _ = assess_fundamentals(pd.Timestamp("2030-01-01"), _valuation(closes, eps), snapshot=snapshot)
    assert score < -0.3


# ------------------------------------------------------------------ engine

def test_engine_end_to_end_buy(breakout_daily, intraday_breakout):
    daily = breakout_daily.copy()
    # Align the intraday sessions so the last one is the session after the daily signal.
    daily.index = pd.bdate_range(end="2024-03-05", periods=len(daily))
    eps = np.linspace(3.0, 4.5, len(daily))
    valuation = pd.DataFrame({"Close": daily["Close"], "TTM_Reported_EPS": eps}, index=daily.index)
    valuation["Trailing_PE"] = valuation["Close"] / valuation["TTM_Reported_EPS"]

    report = analyze(
        "TEST",
        daily,
        to_weekly(daily),
        bars_5m=intraday_breakout,
        valuation_df=valuation,
        settings=AnalysisSettings(),
        risk_settings=RiskSettings(account_equity=50_000, risk_pct=1.0, full_size_confidence=60, half_size_confidence=50),
    )
    assert report.direction == 1
    assert report.signal["holding"]
    assert report.confidence is not None and report.confidence > 55
    assert report.action == "BUY"
    assert report.plan.stop < report.plan.entry < report.plan.target
    assert report.plan.dollar_risk <= 500 + 1e-6
    assert {c.name for c in report.components} >= {"Signal quality", "Trend alignment", "Intraday entry", "Fundamentals"}

    markdown = report_to_markdown(report)
    assert "Trade plan" in markdown and "Component scores" in markdown
    assert report_to_dict(report)["action"] == "BUY"
    assert "TEST" in short_summary(report)


def test_engine_reports_watch_levels_without_signal():
    bars = make_bars(np.linspace(50, 80, 300) + np.sin(np.arange(300)) * 0.1)
    bars = bars.iloc[:-5]
    quiet = pd.concat([bars, make_bars(np.full(5, bars["Close"].iloc[-1] - 3), index=pd.bdate_range(bars.index[-1] + pd.offsets.BDay(), periods=5))])
    report = analyze("FLAT", quiet)
    assert report.action == "NO SIGNAL"
    assert "Watch a daily close above" in report.headline


def test_webhook_payload_shapes():
    assert "content" in build_webhook_payload("https://discord.com/api/webhooks/x", "hi")
    assert build_webhook_payload("https://hooks.slack.com/services/x", "hi") == {"text": "hi"}


def test_zone_pressed_against_entry_is_reported_not_ignored():
    resistances = [
        PriceZone(low=100.2, high=101.0, price=100.6, touches=3, strength=3, source="pivot"),
        PriceZone(low=108.0, high=108.5, price=108.2, touches=1, strength=1, source="period_high"),
    ]
    plan = plan_trade(1, 100.0, 2.0, [], resistances, RiskSettings(), confidence=80)
    assert plan.blocking_zone is resistances[0]
    assert plan.target == pytest.approx(108.0)
    assert any("pressed against" in note for note in plan.notes)


def test_period_high_becomes_resistance_even_with_one_touch():
    closes = np.concatenate([np.linspace(100, 130, 60), np.linspace(130, 110, 60)])
    bars = make_bars(closes, seed=11)
    _, resistances = support_resistance(bars, atr_value=1.5, min_touches=2)
    assert any(zone.source == "period_high" or zone.high >= bars["High"].max() - 1.5 for zone in resistances)


def test_ordinal_wording():
    from analysis.fundamentals import ordinal
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 100)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "100th"]


def test_unconfirmed_breakout_is_watch_with_no_live_size(breakout_daily):
    weak = breakout_daily.copy()
    weak.iloc[-1, weak.columns.get_loc("Volume")] = 1.0e6  # breakout on ordinary volume
    report = analyze("WEAK", weak, to_weekly(weak))
    assert report.action == "WATCH"
    assert report.plan.shares == 0 and report.plan.dollar_risk == 0
    assert "half size" not in report.headline and "full size" not in report.headline
    assert any("Size once the signal is confirmed" in note for note in report.plan.notes)


# ------------------------------------------------------------------ validation

from analysis.validation import simulate_trade, summarize, trade_stats, verdict, walk_forward


def _bars(rows, start="2024-01-01"):
    frame = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=pd.bdate_range(start, periods=len(rows)))
    frame["Volume"] = 1e6
    return frame


def test_simulate_long_target_and_stop():
    bars = _bars([[100, 101, 99, 100], [100, 103, 99.5, 102], [102, 106, 101, 105], [105, 106, 104, 105]])
    win = simulate_trade(bars, 0, 1, stop=98.0, target=105.0, max_hold=10)
    assert win["Exit_Reason"] == "target" and win["Exit"] == 105.0
    assert win["R"] == pytest.approx(5 / 2)
    loss = simulate_trade(bars, 0, 1, stop=99.6, target=110.0, max_hold=10)
    assert loss["Exit_Reason"] == "stop" and loss["R"] == pytest.approx(-1.0)


def test_simulate_stop_assumed_first_when_both_hit_same_bar():
    bars = _bars([[100, 100, 100, 100], [100, 110, 90, 100]])
    trade = simulate_trade(bars, 0, 1, stop=95.0, target=105.0, max_hold=5)
    assert trade["Exit_Reason"] == "stop"


def test_simulate_gap_through_stop_exits_at_open():
    bars = _bars([[100, 100, 100, 100], [100, 101, 99, 100], [90, 91, 89, 90]])
    trade = simulate_trade(bars, 0, 1, stop=97.0, target=110.0, max_hold=5)
    assert trade["Exit_Reason"] == "stop (gap)" and trade["Exit"] == 90
    assert trade["R"] == pytest.approx(-10 / 3)


def test_simulate_short_and_time_exit():
    bars = _bars([[100, 100, 100, 100], [100, 101, 99, 99], [99, 100, 97, 98], [98, 99, 96, 97]])
    trade = simulate_trade(bars, 0, -1, stop=103.0, target=80.0, max_hold=3)
    assert trade["Exit_Reason"] == "time" and trade["Exit"] == 97
    assert trade["R"] == pytest.approx(3 / 3)


def test_trade_stats_and_verdict():
    trades = pd.DataFrame({"R": [1.0, -1.0, 2.0, -1.0], "Return": [0.01, -0.01, 0.02, -0.01],
                           "Exit_Reason": ["target", "stop", "target", "stop"], "Bars_Held": [3, 2, 5, 1]})
    stats = trade_stats(trades)
    assert stats["Win rate"] == 0.5 and stats["Average R"] == pytest.approx(0.25)
    assert stats["Profit factor"] == pytest.approx(1.5)
    assert "Too few trades" in verdict(stats)
    assert "Positive edge" in verdict({"Trades": 100, "t-stat (avg R)": 2.5, "Average R": 0.3})
    assert "No reliable edge" in verdict({"Trades": 100, "t-stat (avg R)": 0.5, "Average R": 0.05})


def test_walk_forward_uses_only_past_data(breakout_daily):
    daily = pd.concat([breakout_daily, make_bars(np.linspace(107.5, 115, 25), index=pd.bdate_range(
        breakout_daily.index[-1] + pd.offsets.BDay(), periods=25), seed=9)])
    trades = walk_forward("TEST", daily)
    assert not trades.empty
    row = trades.loc[trades["Signal_Date"] == breakout_daily.index[-1]].iloc[0]
    # The engine on the truncated history must give the same confidence as the replay.
    direct = analyze("TEST", breakout_daily, to_weekly(breakout_daily))
    assert row["Confidence"] == pytest.approx(direct.confidence)
    assert row["Entry"] == daily["Open"].iloc[len(breakout_daily)]
    summary = summarize(trades)
    assert {"all", "taken", "by_confidence", "components", "thresholds"} <= set(summary)


def test_pullback_detector_fires_in_uptrend_after_dip():
    rng = np.random.default_rng(3)
    rise = np.linspace(50, 120, 280) + rng.normal(0, 0.3, 280)
    dip = np.linspace(120, 113, 6)
    closes = np.concatenate([rise, dip, [116.0]])
    volumes = np.full(len(closes), 1e6)
    volumes[-7:-1] = 0.6e6
    bars = make_bars(closes, volumes, seed=4)
    bars.iloc[-1, bars.columns.get_loc("High")] = 116.3
    from analysis.patterns import detect_pullbacks
    signals = detect_pullbacks(bars)
    assert bool(signals["Signal_Up"].iloc[-1])
    assert bool(signals["Volume_Confirmed"].iloc[-1])
    assert not signals["Signal_Down"].any()
    report = analyze("PB", bars, to_weekly(bars), settings=AnalysisSettings(setup="pullback"))
    assert report.direction == 1
    assert report.plan.stop < signals["Invalidation_Up"].iloc[-1]


def test_long_only_bearish_signal_is_exit_without_short():
    rng = np.random.default_rng(8)
    closes = np.concatenate([100 + rng.normal(0, 0.5, 300), [94.0]])
    volumes = np.full(len(closes), 1e6); volumes[-1] = 3e6
    bars = make_bars(closes, volumes, seed=2)
    report = analyze("DN", bars, to_weekly(bars))
    assert report.direction == -1
    assert report.action.startswith("EXIT") or report.action in ("NO TRADE", "WATCH")
    assert report.plan.shares == 0
    short = analyze("DN", bars, to_weekly(bars), settings=AnalysisSettings(allow_short=True))
    assert not short.action.startswith("EXIT")


def test_r_uses_planned_risk_when_given():
    bars = _bars([[100, 100, 100, 100], [98.5, 99, 97, 98], [98, 99, 97, 98]])
    # Planned at the 100 close with a 97 stop (3 risk); next open gaps to 98.5, then the stop is hit.
    trade = simulate_trade(bars, 0, 1, stop=97.0, target=110.0, max_hold=5, risk_per_share=3.0)
    assert trade["Exit_Reason"] == "stop"
    assert trade["R"] == pytest.approx(-1.5 / 3.0)


def test_matched_random_baseline_is_attached_and_deterministic(breakout_daily):
    daily = pd.concat([breakout_daily, make_bars(np.linspace(107.5, 115, 25), index=pd.bdate_range(
        breakout_daily.index[-1] + pd.offsets.BDay(), periods=25), seed=9)])
    first = walk_forward("TEST", daily)
    second = walk_forward("TEST", daily)
    assert "Random_R" in first and first["Random_R"].notna().all()
    pd.testing.assert_series_equal(first["Random_R"], second["Random_R"])
    stats = trade_stats(first)
    assert np.isfinite(stats["Random-entry avg R"])


def test_verdict_judges_edge_over_random_not_raw_r():
    drift_only = {"Trades": 200, "Average R": 0.20, "t-stat (avg R)": 4.0,
                  "Random-entry avg R": 0.22, "Edge vs random (R)": -0.02, "t-stat (edge)": -0.3}
    assert verdict(drift_only).startswith("No reliable edge")
    real = dict(drift_only, **{"Edge vs random (R)": 0.25, "t-stat (edge)": 3.1})
    assert verdict(real).startswith("Positive edge")


def test_exit_signal_stats_reports_stock_move():
    from analysis.validation import exit_signal_stats
    trades = pd.DataFrame({"Direction": [-1, -1, 1], "Fwd_10": [-0.02, 0.01, 0.05]})
    stats = exit_signal_stats(trades, base_forward=pd.Series([0.001, 0.002]))
    assert stats["Bearish signals"] == 2
    assert stats["Stock 10-bar return after signal"] == pytest.approx(0.005)
    assert stats["Exit helped vs holding"] is False


def test_reversal_detector_buys_breakdown_only_in_uptrend():
    from analysis.patterns import detect_reversals
    rng = np.random.default_rng(1)
    up = make_bars(np.concatenate([np.linspace(50, 120, 300) + rng.normal(0, 0.6, 300), [112.0]]), seed=5)
    signals = detect_reversals(up)
    assert bool(signals["Signal_Up"].iloc[-1]) and not signals["Signal_Down"].any()
    down = make_bars(np.concatenate([np.linspace(120, 50, 300) + rng.normal(0, 0.6, 300), [44.0]]), seed=5)
    assert not detect_reversals(down)["Signal_Up"].iloc[-1]
    report = analyze("RV", up, to_weekly(up), settings=AnalysisSettings(setup="reversal"))
    assert report.direction == 1 and report.plan.stop < up["Low"].iloc[-1]


def test_day_breakout_signal_needs_a_strong_close_and_marks_weak_closes_as_exits():
    from analysis.patterns import detect_day_breakouts, detect_signals
    rng = np.random.default_rng(2)
    up = make_bars(np.linspace(50, 120, 300) + rng.normal(0, 0.6, 300), seed=5)
    last = up.index[-1]
    up.loc[last, ["Open", "High", "Low"]] = [up["High"].iloc[-11:-1].max() - 1, up["High"].iloc[-11:-1].max() + 3,
                                              up["High"].iloc[-11:-1].max() - 2]
    up.loc[last, "Close"] = up.loc[last, "High"] - 0.5
    signals = detect_day_breakouts(up, 10)
    assert bool(signals["Signal_Up"].iloc[-1]) and not signals["Signal_Down"].any()
    assert signals["Trigger_Up"].iloc[-1] == pytest.approx(up["High"].iloc[-11:-1].max())
    assert signals["Invalidation_Up"].iloc[-1] == up["Low"].iloc[-1]
    assert (signals["Exit_Long"] == (signals["Close_Location"] < 2 / 3)).all()
    pd.testing.assert_frame_equal(detect_signals(up, "day_breakout", 10, 1.5), signals)

    weak = up.copy()
    weak.loc[last, "Close"] = weak.loc[last, "Low"] + 0.5
    assert not detect_day_breakouts(weak, 10)["Signal_Up"].iloc[-1]

    report = analyze("DB", up, to_weekly(up), settings=AnalysisSettings(setup="day_breakout", breakout_lookback=10))
    assert report.direction == 1 and report.plan.stop < up["Low"].iloc[-1]
    quiet = analyze("DB", weak, to_weekly(weak), settings=AnalysisSettings(setup="day_breakout", breakout_lookback=10))
    assert quiet.direction != 1 or "day breakout" in quiet.headline.lower()


def test_ma_rule_fires_once_per_cross_and_matches_backtest_rule():
    from analysis.patterns import detect_signals, ma_rule_conditions

    closes = np.concatenate([np.linspace(100, 80, 80), np.linspace(80, 120, 80), np.linspace(120, 90, 80)])
    bars = make_bars(closes, seed=5)
    rule = {"fast_type": "EMA", "fast_period": 10, "slow_type": "SMA", "slow_period": 30}
    signals = detect_signals(bars, "ma_rule", ma_rule=rule)
    _, _, buy, sell = ma_rule_conditions(bars, rule)
    pd.testing.assert_series_equal(signals["Signal_Up"], buy, check_names=False)
    pd.testing.assert_series_equal(signals["Signal_Down"], sell, check_names=False)
    assert signals["Signal_Up"].sum() >= 1 and signals["Signal_Down"].sum() >= 1
    # Long invalidation sits at or below the signal bar's low.
    row = signals[signals["Signal_Up"]].iloc[0]
    assert row["Invalidation_Up"] <= bars.loc[signals["Signal_Up"]].iloc[0]["Low"] + 1e-9


def test_ma_rule_level_condition_fires_on_first_bar_only():
    from analysis.patterns import detect_ma_rule

    closes = np.concatenate([np.full(60, 100.0), np.linspace(100, 130, 40)])
    bars = make_bars(closes, seed=6)
    rule = {"buy": ("Close", ">", "Slow_MA"), "sell": ("Close", "<", "Slow_MA")}
    signals = detect_ma_rule(bars, rule)
    fired = signals.index[signals["Signal_Up"]]
    assert len(fired) >= 1
    # Close stays above the slow MA through the rally, so no repeated buys on consecutive bars.
    positions = [bars.index.get_loc(ts) for ts in fired]
    assert all(b - a > 1 for a, b in zip(positions, positions[1:]))


def test_engine_runs_ma_rule_setup(breakout_daily):
    rule = {"fast_type": "EMA", "fast_period": 5, "slow_type": "SMA", "slow_period": 20,
            "buy": ("Close", "Cross Above", "Slow_MA"), "sell": ("Close", "Cross Below", "Slow_MA")}
    report = analyze("TEST", breakout_daily, to_weekly(breakout_daily),
                     settings=AnalysisSettings(setup="ma_rule", ma_rule=rule, signal_window=400))
    assert report.signal is not None
    assert report.action not in ("INSUFFICIENT DATA",)
    text = report_to_markdown(report)
    assert "TEST" in text
    no_signal = analyze("TEST", breakout_daily, None,
                        settings=AnalysisSettings(setup="ma_rule", ma_rule={"buy": ("Close", ">", "Close"), "sell": ("Close", "<", "Close")}))
    assert no_signal.action == "NO SIGNAL"
    assert "moving-average rule" in no_signal.headline


def test_setup_periods_reach_the_detectors(breakout_daily):
    from analysis.patterns import detect_pullbacks, detect_reversals, detect_signals

    settings = AnalysisSettings(setup="pullback", pullback_ema=10, trend_mid=30, trend_slow=100,
                                swing_bars=15, pullback_bars=4, min_depth_atr=1.0)
    via_settings = detect_signals(breakout_daily, "pullback", 20, 1.5, **settings.detector_kwargs())
    direct = detect_pullbacks(breakout_daily, fast=10, mid=30, slow=100, swing_bars=15, pullback_bars=4, min_depth_atr=1.0)
    pd.testing.assert_frame_equal(via_settings, direct)

    via_settings = detect_signals(breakout_daily, "reversal", 12, 1.5, trend_mid=30, trend_slow=100)
    pd.testing.assert_frame_equal(via_settings, detect_reversals(breakout_daily, 12, mid=30, slow=100))
    assert not via_settings.equals(detect_signals(breakout_daily, "reversal", 20, 1.5))


def test_no_signal_headlines_use_chosen_periods(breakout_daily):
    flat = breakout_daily.copy()
    report = analyze("TEST", flat.iloc[:200], None, settings=AnalysisSettings(
        setup="pullback", pullback_ema=13, trend_mid=40, trend_slow=150, swing_bars=17, min_depth_atr=2.5))
    if report.action == "NO SIGNAL":
        assert "13-bar EMA" in report.headline and "150-bar MA" in report.headline and "2.5 ATR" in report.headline
    report = analyze("TEST", flat.iloc[:200], None, settings=AnalysisSettings(
        setup="reversal", breakout_lookback=9, trend_mid=40, trend_slow=150))
    if report.action == "NO SIGNAL":
        assert "9-bar low" in report.headline and "150-bar MA" in report.headline
