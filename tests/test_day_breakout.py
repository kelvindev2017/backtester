import numpy as np
import pandas as pd
import pytest

from analysis.day_breakout import (
    DayBreakoutSettings,
    close_location,
    day_breakout_trades,
    monthly_pnl,
    prior_average_close,
    run_day_breakout,
    split_summary,
    ticker_summary,
)


def bars(rows, start="2026-03-02"):
    """Daily bars from (open, high, low, close) tuples."""

    frame = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=pd.bdate_range(start, periods=len(rows)))
    frame["Volume"] = 1e6
    return frame


FLAT = [(100, 101, 99, 100)] * 10


def test_close_location():
    frame = bars([(10, 12, 8, 11), (10, 10, 10, 10)])
    assert close_location(frame).tolist() == [0.75, 0.5]


def test_entry_at_breakout_level_and_exit_at_close_when_close_is_weak():
    frame = bars(FLAT + [(100.5, 104, 100, 101), (101, 101.5, 100, 100.5)])
    trades = day_breakout_trades(frame, DayBreakoutSettings(fee_per_trade=0), "X")
    assert len(trades) == 1
    trade = trades.iloc[0]
    # Prior 10-day high is 101; the stock opened below it, so the buy-stop fills at 101.
    assert trade["Entry"] == 101
    assert trade["Exit"] == 101 and trade["Days held"] == 0
    assert trade["Shares"] == 99 and trade["P&L"] == 0


def test_gap_up_fills_at_open():
    frame = bars(FLAT + [(103, 106, 102.5, 103.5)])
    trade = day_breakout_trades(frame, DayBreakoutSettings(), "X").iloc[0]
    assert trade["Entry"] == 103


def test_strong_close_is_held_until_a_weak_close():
    frame = bars(FLAT + [(100.5, 105, 100, 104.8), (105, 108, 104, 107.9), (108, 109, 105, 105.5), (105, 106, 104, 105)])
    trade = day_breakout_trades(frame, DayBreakoutSettings(fee_per_trade=1.5), "X").iloc[0]
    assert trade["Entry date"] == frame.index[10]
    assert trade["Exit date"] == frame.index[12] and trade["Days held"] == 2
    assert trade["Exit"] == 105.5
    assert trade["P&L"] == pytest.approx(99 * (105.5 - 101) - 1.5)
    assert trade["Status"] == "Closed"


def test_same_day_mode_and_max_hold():
    strong = FLAT + [(100.5, 105, 100, 104.8), (105, 108, 104, 107.9), (108, 112, 107, 111.9)]
    frame = bars(strong + [(110, 111, 104, 105)])
    assert day_breakout_trades(frame, DayBreakoutSettings(exit_mode="same_day"), "X").iloc[0]["Days held"] == 0
    assert day_breakout_trades(frame, DayBreakoutSettings(max_hold=2), "X").iloc[0]["Days held"] == 1


def test_trade_held_on_last_bar_is_open_and_left_out_of_stats():
    frame = bars(FLAT + [(100.5, 105, 100, 104.8)])
    trades = day_breakout_trades(frame, DayBreakoutSettings(), "X")
    assert trades.iloc[0]["Status"] == "Open"
    assert split_summary(trades, "2026-01-01").loc["All", "Trades"] == 0


def test_trend_filter_and_start_date_limit_entries():
    frame = bars(FLAT + [(100.5, 104, 100, 101), (101, 101.5, 100, 100.5)])
    rising = bars([(100 + i, 101 + i, 99 + i, 100 + i) for i in range(10)] + [(109.5, 112, 109, 110)])
    assert len(day_breakout_trades(rising, DayBreakoutSettings(trend_sma=5), "X")) == 1
    falling = bars([(110 - i, 111 - i, 109 - i, 110 - i) for i in range(10)] + [(100.5, 112, 100, 101)])
    assert day_breakout_trades(falling, DayBreakoutSettings(trend_sma=5), "X").empty
    assert day_breakout_trades(frame, DayBreakoutSettings(), "X", start=frame.index[-1]).empty


def test_multi_ticker_summaries():
    rng = np.random.default_rng(1)
    prices = {}
    for ticker in ("A", "B"):
        closes = 100 + np.cumsum(rng.normal(0.2, 1.5, 200))
        opens = np.r_[closes[0], closes[:-1]]
        highs = np.maximum(opens, closes) + rng.uniform(0, 1, 200)
        lows = np.minimum(opens, closes) - rng.uniform(0, 1, 200)
        prices[ticker] = bars(list(zip(opens, highs, lows, closes)), start="2025-06-02")
    trades = run_day_breakout(prices, start="2025-08-01")
    assert set(trades["Ticker"]) == {"A", "B"}
    assert trades["Exit date"].is_monotonic_increasing
    table = split_summary(trades, "2026-01-01")
    closed = trades[trades["Status"] == "Closed"]
    assert table.loc["All", "Trades"] == table.iloc[0]["Trades"] + table.iloc[1]["Trades"] == len(closed)
    assert table.loc["All", "Net P&L"] == pytest.approx(closed["P&L"].sum())
    assert table.loc["All", "Max drawdown"] <= 0
    assert monthly_pnl(trades).sum() == pytest.approx(closed["P&L"].sum())
    assert ticker_summary(trades)["Trades"].sum() == len(closed)


def test_moving_average_exit_holds_until_a_close_below_the_average():
    rising = [(100 + i, 101 + i, 99 + i, 100 + i) for i in range(10)]
    frame = bars(rising + [(110, 113, 109.5, 112.5), (112.5, 114, 111, 111.5), (111, 112, 104, 104.5)])
    settings = DayBreakoutSettings(exit_mode="above_ma", exit_ma_type="SMA", exit_ma_period=5, max_hold=0)
    trade = day_breakout_trades(frame, settings, "X").iloc[0]
    # SMA 5 is about 108-109 on the last bars; the close at 104.5 is the first one below it.
    assert trade["Exit date"] == frame.index[-1] and trade["Exit"] == 104.5
    ema = DayBreakoutSettings(exit_mode="above_ma", exit_ma_type="EMA", exit_ma_period=5, max_hold=0)
    assert day_breakout_trades(frame, ema, "X").iloc[0]["Exit date"] == frame.index[-1]
    # Until the average has enough history, it cannot hold a position.
    long_ma = DayBreakoutSettings(exit_mode="above_ma", exit_ma_period=60, max_hold=0)
    assert day_breakout_trades(frame, long_ma, "X").iloc[0]["Days held"] == 0


def test_daily_break_exit_sells_on_a_close_below_the_prior_average_close():
    rising = [(100 + i, 101 + i, 99 + i, 100 + i) for i in range(10)]
    # Breakout closes 112.5, then 113 (higher), 112.8 (below yesterday), 111 (below the prior 3-day average).
    frame = bars(rising + [(110, 113, 109.5, 112.5), (112.5, 114, 112, 113), (113, 114, 112, 112.8),
                           (112.8, 113, 110, 111)])
    yesterday = DayBreakoutSettings(exit_mode="daily_break", max_hold=0)
    trade = day_breakout_trades(frame, yesterday, "X").iloc[0]
    assert trade["Exit date"] == frame.index[12] and trade["Exit"] == 112.8
    # Prior 3-day average before day 12 is (109 + 112.5 + 113) / 3 = 111.5, so 112.8 holds; 111 < 112.77 sells.
    three_days = DayBreakoutSettings(exit_mode="daily_break", break_days=3, max_hold=0)
    trade = day_breakout_trades(frame, three_days, "X").iloc[0]
    assert trade["Exit date"] == frame.index[13] and trade["Exit"] == 111


def test_prior_average_close_excludes_today():
    frame = bars([(1, 2, 0.5, 1), (1, 3, 1, 2), (2, 4, 2, 3), (3, 5, 3, 10)])
    assert prior_average_close(frame, 1).tolist()[1:] == [1, 2, 3]
    assert prior_average_close(frame, 2).tolist()[2:] == [1.5, 2.5]
