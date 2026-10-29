import numpy as np
import pandas as pd

from analysis.opening_bar import (
    OpeningBarSettings,
    exit_session,
    flat_bar_position,
    order_costs,
    is_long_day,
    opening_bar_trades,
    position_shares,
    run_opening_bar,
    summarize_opening_bar,
)
from analysis.rule_diagrams import opening_bar_figure

# No slippage, so fills are the bar prices; the IBKR fees still apply and are checked through "Costs $".
EXACT = dict(slippage=0.0)


def session(day, bars):
    """5-minute bars from 09:30 given (open, high, low, close) tuples."""

    index = pd.date_range(f"{day} 09:30", periods=len(bars), freq="5min")
    frame = pd.DataFrame(bars, columns=["Open", "High", "Low", "Close"], index=index)
    frame["Volume"] = 1e5
    return frame


# First bar green (100 -> 101, low 99.5), a run up, a pullback close at 102.5, then a close below it.
RUN = [(100, 101.2, 99.5, 101), (101, 101.8, 100.9, 101.5), (101.5, 102.4, 101.4, 102.2), (102.2, 103.1, 102, 103),
       (103, 103.2, 102.4, 102.5), (102.5, 103.6, 102.4, 103.4), (103.4, 103.5, 102.1, 102.2),
       (102.2, 102.5, 101.9, 102), (102, 102.3, 101.8, 102.1)]


def test_long_day_needs_the_move():
    bar = pd.Series({"Open": 100.0, "Close": 100.3})
    assert is_long_day(bar) and is_long_day(bar, 0.3)
    assert not is_long_day(bar, 0.5)
    assert not is_long_day(pd.Series({"Open": 100.0, "Close": 100.0}))


def test_sizing_caps_the_loss_and_the_capital():
    # 10% of $10,000 is $1,000 of risk; $2 a share risk allows 500 shares, but $10,000 buys only 100 at $100.
    assert position_shares(10_000, 100, 98, 10) == 100
    assert position_shares(10_000, 100, 98, 10, cap_at_capital=False) == 500
    assert position_shares(10_000, 100, 80, 10) == 50
    assert position_shares(10_000, 100, 100, 10) == 0


def test_pullback_exit():
    frame = session("2026-10-01", RUN)
    # Bar 4 closes below bar 3: pullback 102.5. Bar 6 closes at 102.2, below it: sell.
    assert exit_session(frame, 101, 99.5, skip_bars=0) == (6, 102.2, "Pullback")
    assert exit_session(frame, 101, 99.5, skip_bars=0, exit_fill="pullback") == (6, 102.5, "Pullback")


def test_skipped_bars_do_not_set_a_pullback():
    frame = session("2026-10-01", RUN)
    # With 4 bars skipped the 102.5 dip is ignored; bar 6 (102.2 < 103.4) is the first pullback, bar 7 breaks it.
    assert exit_session(frame, 101, 99.5, skip_bars=4) == (7, 102.0, "Pullback")


def test_lower_high_exit():
    frame = session("2026-10-01", RUN)
    # Bar 4's high 103.2 is above bar 3's 103.1; bar 6's high 103.5 is below bar 5's 103.6: sell at its close.
    assert exit_session(frame, 101, 99.5, skip_bars=0, exit_mode="lower_high") == (6, 102.2, "Lower high")


def test_stop_and_end_of_day():
    bars = RUN[:3] + [(102.2, 102.3, 99.0, 99.2)] + RUN[4:]
    frame = session("2026-10-01", bars)
    assert exit_session(frame, 101, 99.5, skip_bars=0) == (3, 99.5, "Stop")
    gap = RUN[:3] + [(99.0, 99.3, 98.5, 99.1)] + RUN[4:]
    assert exit_session(session("2026-10-01", gap), 101, 99.5)[1:] == (99.0, "Stop")
    # Only rising closes: sold at the open of the last bar, 5 minutes before the close.
    rising = [(100 + i, 101.5 + i, 99.5 + i, 101 + i) for i in range(6)]
    assert exit_session(session("2026-10-01", rising), 101, 99.5) == (5, 105, "End of day")


def test_trades_only_on_long_days():
    long_day = session("2026-10-01", RUN)
    short_day = session("2026-10-02", [(100, 100.5, 99, 99.5)] + RUN[1:])
    trades, days = opening_bar_trades(pd.concat([long_day, short_day]), OpeningBarSettings(skip_bars=0, **EXACT),
                                       "TEST")
    assert list(days["Long day"]) == [True, False]
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["Entry"] == 101 and trade["Stop"] == 99.5 and trade["Shares"] == 99
    assert np.isclose(trade["Costs $"], order_costs(99, 101, False) + order_costs(99, 102.2, True))
    assert np.isclose(trade["P&L"], 99 * (102.2 - 101) - trade["Costs $"])
    assert trade["Exit time"] == pd.Timestamp("2026-10-01 10:05")


def test_run_and_summary():
    bars = {"A": session("2026-10-01", RUN), "B": session("2026-10-01", RUN)}
    trades, days = run_opening_bar(bars, OpeningBarSettings(skip_bars=0, **EXACT))
    summary = summarize_opening_bar(trades, days)
    assert summary["Trades"] == 2 and summary["Long days %"] == 100
    assert np.isclose(summary["Net P&L"], 2 * 99 * 1.2 - summary["Costs"])
    assert summary["Pullback exit %"] == 100


def test_diagram_follows_the_settings():
    text = lambda fig: " ".join(a.text for a in fig.layout.annotations)
    assert "Next 3 bars ignored" in text(opening_bar_figure(0, 3))
    assert "open + 0.5" in text(opening_bar_figure(0.5, 2))
    assert "pullback" in text(opening_bar_figure(0, 2, "pullback"))
    assert "prior<br>bar's high" in text(opening_bar_figure(0, 12, "lower_high"))


def test_short_days_trade_the_mirror_image():
    # The long day turned upside down: price 200 minus each long price, so the short makes what the long made.
    flipped = [(200 - o, 200 - l, 200 - h, 200 - c) for o, h, l, c in RUN]
    frame = session("2026-10-01", flipped)
    off, _ = opening_bar_trades(frame, OpeningBarSettings(skip_bars=0), "TEST")
    assert off.empty
    trades, days = opening_bar_trades(frame, OpeningBarSettings(skip_bars=0, allow_short=True, **EXACT), "TEST")
    assert list(days["Short day"]) == [True]
    trade = trades.iloc[0]
    assert trade["Side"] == "Short" and trade["Entry"] == 99 and trade["Stop"] == 100.5
    assert trade["Exit"] == 97.8 and trade["Exit reason"] == "Pullback"
    # The short pays the sale fees when it opens.
    assert np.isclose(trade["Costs $"], order_costs(trade["Shares"], 99, True) + order_costs(trade["Shares"], 97.8, False))
    assert np.isclose(trade["P&L"], trade["Shares"] * (99 - 97.8) - trade["Costs $"])
    lower, _ = opening_bar_trades(frame, OpeningBarSettings(skip_bars=0, allow_short=True, exit_mode="lower_high",
                                                            **EXACT))
    assert lower.iloc[0]["Exit"] == 97.8


def test_short_stop_is_the_first_bar_high():
    flipped = [(200 - o, 200 - l, 200 - h, 200 - c) for o, h, l, c in RUN]
    flipped[3] = (97.8, 101.0, 97.7, 100.8)
    trades, _ = opening_bar_trades(session("2026-10-01", flipped),
                                   OpeningBarSettings(skip_bars=0, allow_short=True, **EXACT))
    assert trades.iloc[0]["Exit reason"] == "Stop" and trades.iloc[0]["Exit"] == 100.5
    assert trades.iloc[0]["P&L"] < 0


def test_ibkr_costs():
    # Tiered, 50 shares at $200: the $0.35 minimum, plus exchange and clearing at $0.0032 a share.
    assert np.isclose(order_costs(50, 200, False), 0.35 * 1.00074 + 50 * 0.0032)
    # The sale adds the SEC fee on the value and the FINRA fees per share.
    assert np.isclose(order_costs(50, 200, True) - order_costs(50, 200, False), 10_000 * 0.0000206 + 50 * 0.000198)
    assert np.isclose(order_costs(1_000, 50, False, "fixed"), 5.0)
    assert np.isclose(order_costs(50, 200, False, "fixed"), 1.0)
    assert order_costs(50, 200, False, "none") == 0
    # A $10,000 round trip on Tiered comes to about $1.30.
    assert 1.0 < order_costs(50, 200, False) + order_costs(50, 200, True) < 1.5


def test_slippage_worsens_both_fills():
    trades, _ = opening_bar_trades(session("2026-10-01", RUN), OpeningBarSettings(skip_bars=0, slippage=0.02))
    trade = trades.iloc[0]
    assert np.isclose(trade["Entry"], 101.02) and np.isclose(trade["Exit"], 102.18)


def test_flat_bar_by_timeframe():
    day = lambda minutes: pd.date_range("2026-10-01 09:30", "2026-10-01 15:59", freq=f"{minutes}min")
    assert day(5)[flat_bar_position(day(5), 5)] == pd.Timestamp("2026-10-01 15:55")
    assert day(1)[flat_bar_position(day(1), 1)] == pd.Timestamp("2026-10-01 15:55")
    # No 15-minute bar starts in the last 5 minutes: sold at the open of the last bar.
    assert day(15)[flat_bar_position(day(15), 15)] == pd.Timestamp("2026-10-01 15:45")
