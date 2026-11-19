import numpy as np
import pandas as pd
import pytest

from analysis.portfolio import BreakoutRule, breakout_slot_returns, run_portfolio, summarize, yearly_returns


def bars(closes, highs=None, lows=None, opens=None):
    closes = np.asarray(closes, dtype=float)
    if opens is None:  # each day opens at the prior close
        opens = np.r_[closes[0], closes[:-1]]
    index = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({"Open": closes if opens is None else opens, "High": closes if highs is None else highs,
                         "Low": closes if lows is None else lows, "Close": closes}, index=index)


def test_breakout_slot_buys_the_break_and_sells_below_the_sma():
    closes = [10.0] * 30 + [11.0, 11.5, 12.0, 9.0]
    frame = bars(closes, highs=[c + 0.1 for c in closes])
    rule = BreakoutRule(lookback=5, trend_sma=0, exit_sma=5, stop_pct=0, cost_pct=0)
    returns, trades = breakout_slot_returns(frame, rule)
    trade = trades.iloc[0]
    assert trade["Entry date"] == frame.index[30] and trade["Entry"] == pytest.approx(10.1)
    assert trade["Exit date"] == frame.index[33] and trade["Exit reason"] == "close below SMA"
    assert (1 + returns).prod() == pytest.approx(9.0 / 10.1)


def test_breakout_slot_stop_fills_at_the_stop_or_the_gap_open():
    closes = [10.0] * 30 + [11.0, 11.5, 11.0]
    lows = list(closes)
    lows[32] = 9.5
    frame = bars(closes, highs=[c + 0.1 for c in closes], lows=lows)
    rule = BreakoutRule(lookback=5, trend_sma=0, exit_sma=5, stop_pct=5, cost_pct=0)
    _, trades = breakout_slot_returns(frame, rule)
    assert trades.iloc[0]["Exit reason"] == "stop"
    assert trades.iloc[0]["Exit"] == pytest.approx(10.1 * 0.95)


def test_breakout_slot_needs_the_trend():
    closes = list(np.linspace(20, 10, 60)) + [10.5, 11.0]
    frame = bars(closes, highs=[c + 0.1 for c in closes])
    rule = BreakoutRule(lookback=3, trend_sma=50, exit_sma=5, stop_pct=0, cost_pct=0)
    assert breakout_slot_returns(frame, rule)[1].empty


def test_portfolio_rebalances_back_to_target():
    index = pd.bdate_range("2023-12-27", "2024-01-05")
    core = pd.DataFrame({"A": 0.0, "B": 0.0}, index=index)
    core.loc[index[1], "A"] = 1.0  # A doubles before the year end
    result = run_portfolio(core, {"A": 0.5, "B": 0.5}, pd.DataFrame(index=index), 0.0, rebalance="yearly",
                           rebalance_cost_pct=0)
    assert result.rebalance_dates == [pd.Timestamp("2024-01-01")]
    assert result.weights.loc["2023-12-29", "A"] == pytest.approx(2 / 3)
    assert result.weights.loc["2024-01-02", "A"] == pytest.approx(0.5)
    assert result.equity.iloc[-1] == pytest.approx(1.5)
    never = run_portfolio(core, {"A": 0.5, "B": 0.5}, pd.DataFrame(index=index), 0.0, rebalance="never")
    assert never.rebalance_dates == []


def test_band_rebalance_and_cash():
    index = pd.bdate_range("2024-01-01", periods=5)
    core = pd.DataFrame({"A": [0.0, 0.1, 0.1, 0.0, 0.0]}, index=index)
    slots = pd.DataFrame({"X": 0.0, "Y": 0.0}, index=index)
    result = run_portfolio(core, {"A": 0.4}, slots, 0.4, rebalance="band", band_pct=3, rebalance_cost_pct=0)
    # A grows from 40% to about 42.3% after one day (inside the band), to about 44.6% after two (outside it).
    assert result.rebalance_dates == [index[3]]
    assert result.weights["Cash"].iloc[0] == pytest.approx(0.2)
    assert list(result.weights.columns) == ["A", "Breakout trades", "Cash"]


def test_summary_and_yearly_returns():
    index = pd.bdate_range("2023-01-02", "2024-12-31")
    equity = pd.Series(np.linspace(100, 121, len(index)), index=index)
    stats = summarize(equity)
    assert stats["Total return %"] == pytest.approx(21.0)
    assert stats["Worst drop %"] == pytest.approx(0.0)
    years = yearly_returns(equity)
    assert list(years.index) == [2023, 2024]
    assert (1 + years / 100).prod() == pytest.approx(1.21)
