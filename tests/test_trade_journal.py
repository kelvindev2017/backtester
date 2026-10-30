import io

import numpy as np
import pandas as pd
import pytest

from analysis.trade_journal import (
    BREAKDOWNS,
    WhatIfRules,
    apply_filters,
    breakdown,
    equity_curve,
    load_ibkr_trades,
    round_trips,
    simulate,
    simulate_trip,
    stop_sweep,
    summarize,
)
from analysis.opening_bar import order_costs


def ibkr_frame(rows):
    """IBKR Flex rows: (symbol, yyyymmdd, side, quantity, price, commission, order type[, currency])."""

    records = []
    for number, row in enumerate(rows):
        symbol, day, side, quantity, price, commission, order_type, *currency = row
        records.append({
            "CurrencyPrimary": currency[0] if currency else "USD", "Symbol": symbol, "TradeID": 1000 + number,
            "OrderID": 500 + number, "TradeDate": day, "Buy/Sell": side,
            "Quantity": quantity if side == "BUY" else -quantity, "Price": price,
            "Amount": (quantity if side == "BUY" else -quantity) * price, "Commission": commission,
            "OrderType": order_type,
        })
    return pd.DataFrame(records)


def hourly(day, rows):
    """Hourly regular-session bars for one day from (open, high, low, close) tuples, starting 09:30."""

    index = pd.date_range(f"{day} 09:30", periods=len(rows), freq="60min")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=index).assign(Volume=1e5)


def trip_fills(frame):
    trades, _ = load_ibkr_trades(frame)
    fills, trips = round_trips(trades)
    return fills, trips


# ------------------------------------------------------------------ import

def test_import_keeps_one_currency_drops_forex_and_signs_quantities():
    frame = ibkr_frame([
        ("AAPL", 20260504, "BUY", 10, 200.0, -1.0, "LMT"),
        ("AAPL", 20260505, "SELL", 10, 210.0, -1.2, "LMT"),
        ("1347", 20260504, "BUY", 1000, 50.0, -40.0, "LMT", "HKD"),
        ("USD.HKD", 20260504, "BUY", 1000, 7.84, -2.0, "LMT", "HKD"),
    ])
    frame["Quantity"] = frame["Quantity"].abs()  # some exports give unsigned quantities with Buy/Sell
    trades, notes = load_ibkr_trades(frame)
    assert trades["Symbol"].tolist() == ["AAPL", "AAPL"]
    assert trades["Quantity"].tolist() == [10, -10]
    assert trades["Cost"].tolist() == [1.0, 1.2]
    assert notes["other_currency"] == 2 and notes["kept"] == 2 and not notes["has_times"]


def test_import_reads_csv_with_datetime_and_rejects_missing_columns():
    csv = ("Symbol,DateTime,TradeDate,Buy/Sell,Quantity,Price,Commission\n"
           "MSFT,20260504;093512,20260504,BUY,5,400,-1\nMSFT,20260504;150001,20260504,SELL,-5,404,-1\n")
    trades, notes = load_ibkr_trades(io.StringIO(csv), name="trades.csv")
    assert notes["has_times"]
    assert trades["Time"].iloc[0] == pd.Timestamp("2026-05-04 09:35:12")
    with pytest.raises(ValueError, match="Price"):
        load_ibkr_trades(pd.DataFrame({"Symbol": ["X"], "TradeDate": [20260504], "Quantity": [1]}))


# ------------------------------------------------------------------ round trips

def test_round_trip_pnl_fees_and_added_to_loser():
    fills, trips = trip_fills(ibkr_frame([
        ("NVDA", 20260504, "BUY", 10, 100.0, -1.0, "LMT"),
        ("NVDA", 20260505, "BUY", 10, 95.0, -1.0, "LMT"),     # added below the first buy
        ("NVDA", 20260506, "SELL", 20, 90.0, -2.0, "STP"),
    ]))
    trip = trips.iloc[0]
    assert len(trips) == 1 and trip["Status"] == "Closed"
    assert trip["Gross P&L"] == pytest.approx(20 * 90 - 10 * 100 - 10 * 95)
    assert trip["Fees"] == pytest.approx(4.0) and trip["Net P&L"] == pytest.approx(-154.0)
    assert trip["Added to loser"] == "Yes" and trip["Entry orders"] == "Added (2+ buys)"
    assert trip["Days held"] == 2 and trip["Held"] == "1-2 days" and trip["Last exit type"] == "STP"


def test_a_fill_through_zero_closes_one_trip_and_opens_a_short():
    fills, trips = trip_fills(ibkr_frame([
        ("TSLA", 20260504, "BUY", 10, 100.0, -1.0, "LMT"),
        ("TSLA", 20260504, "SELL", 30, 101.0, -3.0, "LMT"),   # sells 10 to close, shorts 20
        ("TSLA", 20260505, "BUY", 20, 99.0, -2.0, "LMT"),
    ]))
    assert trips["Side"].tolist() == ["Long", "Short"]
    assert trips["Net P&L"].tolist() == pytest.approx([10 - 2.0, 40 - 4.0])  # commission split 1:2 on the flip fill


def test_open_position_is_excluded_from_the_summary():
    fills, trips = trip_fills(ibkr_frame([
        ("AMD", 20260504, "BUY", 10, 100.0, -1.0, "LMT"),
        ("AMD", 20260505, "SELL", 10, 104.0, -1.0, "LMT"),
        ("AMD", 20260506, "BUY", 10, 105.0, -1.0, "LMT"),
    ]))
    assert trips["Status"].tolist() == ["Closed", "Open"]
    stats = summarize(trips)
    assert stats["Trades"] == 1 and stats["Net P&L"] == pytest.approx(38.0)


def test_summary_breakdowns_filters_and_equity():
    fills, trips = trip_fills(ibkr_frame([
        ("A", 20260504, "BUY", 10, 10.0, 0, "LMT"), ("A", 20260504, "SELL", 10, 11.0, 0, "LMT"),
        ("B", 20260505, "BUY", 10, 10.0, 0, "LMT"), ("B", 20260507, "SELL", 10, 8.0, 0, "STP"),
        ("A", 20260508, "BUY", 10, 10.0, 0, "LMT"), ("A", 20260508, "SELL", 10, 10.5, 0, "LMT"),
    ]))
    stats = summarize(trips)
    assert stats["Win rate %"] == pytest.approx(200 / 3) and stats["Payoff"] == pytest.approx(7.5 / 20)
    assert stats["Breakeven win rate %"] == pytest.approx(100 / 1.375)
    assert stats["Profit factor"] == pytest.approx(15 / 20)
    by_symbol = breakdown(trips, "Symbol")
    assert by_symbol.loc["B", "Net P&L"] == -20 and by_symbol.index[0] == "B"
    for column in BREAKDOWNS.values():
        assert breakdown(trips, column)["Trades"].sum() == 3
    assert apply_filters(trips, symbols=["A"])["Trip"].tolist() == ["A #1", "A #2"]
    assert apply_filters(trips, exclude=["A"], side="Long")["Trip"].tolist() == ["B #1"]
    assert equity_curve(trips).iloc[-1] == pytest.approx(-5.0)


# ------------------------------------------------------------------ what-if

def long_trip(prices_by_day, fills_rows):
    fills, trips = trip_fills(ibkr_frame(fills_rows))
    bars = pd.concat([hourly(day, rows) for day, rows in prices_by_day])
    return fills, trips, bars


DAY1 = [(100, 101, 99.5, 100.5), (100.5, 101, 99, 99.5), (99.5, 100, 98, 99)]
DAY2 = [(97, 97.5, 92, 93), (93, 94, 90, 91), (91, 92, 89, 90)]
LOSER = [("X", 20260504, "BUY", 100, 100.0, -1.0, "LMT"), ("X", 20260505, "SELL", 100, 90.0, -1.0, "STP")]


def test_no_rules_reproduces_your_actual_result():
    fills, trips, bars = long_trip([("2026-05-04", DAY1), ("2026-05-05", DAY2)], LOSER)
    result = simulate(fills, trips, {"X": bars}, WhatIfRules())
    assert result["What-if P&L"].iloc[0] == pytest.approx(result["Net P&L"].iloc[0])
    assert result["What-if exit"].iloc[0] == "Your exit"


def test_stop_fills_at_the_open_when_the_next_day_gaps_past_it():
    fills, trips, bars = long_trip([("2026-05-04", DAY1), ("2026-05-05", DAY2)], LOSER)
    result = simulate_trip(fills, bars, WhatIfRules(stop_pct=2, slippage_bps=0))
    # Stop at 98: day 1 touches 98 on its last bar but the buy's bar is the first one; the last bar's low equals
    # the stop, so it fills at 98 that afternoon.
    assert result["What-if exit"] == "Stop" and result["What-if exit price"] == pytest.approx(98)
    gap = simulate_trip(fills, bars, WhatIfRules(stop_pct=3, slippage_bps=0))
    # Stop at 97 is never touched on day 1; day 2 opens at 97 (at the stop): filled at the open.
    assert gap["What-if exit"] == "Stop (gap)" and gap["What-if exit price"] == pytest.approx(97)
    expected = (97 - 100) * 100 - 1.0 - order_costs(100, 97, sell=True)
    assert gap["What-if P&L"] == pytest.approx(expected)


def test_slippage_makes_the_stop_fill_worse():
    fills, trips, bars = long_trip([("2026-05-04", DAY1), ("2026-05-05", DAY2)], LOSER)
    result = simulate_trip(fills, bars, WhatIfRules(stop_pct=3, slippage_bps=10))
    assert result["What-if exit price"] == pytest.approx(97 * (1 - 0.001))


def test_entry_bar_low_alone_does_not_trigger_the_stop_but_a_close_below_it_does():
    rows = [("X", 20260504, "BUY", 100, 100.0, 0, "LMT"), ("X", 20260505, "SELL", 100, 101.0, 0, "LMT")]
    dip_then_recover = [(100, 100.5, 95, 100.2), (100.2, 101, 100, 100.8)]
    fills, trips, bars = long_trip([("2026-05-04", dip_then_recover), ("2026-05-05", [(100.8, 101.5, 100.5, 101)])],
                                   rows)
    kept = simulate_trip(fills, bars, WhatIfRules(stop_pct=2, slippage_bps=0))
    assert kept["What-if exit"] == "Your exit" and kept["What-if P&L"] == pytest.approx(100.0)
    closes_below = [(100, 100.5, 95, 97), (97, 101, 96.5, 100.8)]
    fills, trips, bars = long_trip([("2026-05-04", closes_below), ("2026-05-05", [(100.8, 101.5, 100.5, 101)])], rows)
    stopped = simulate_trip(fills, bars, WhatIfRules(stop_pct=2, slippage_bps=0))
    assert stopped["What-if exit"] == "Stop" and stopped["What-if exit price"] == pytest.approx(98)


def test_target_and_stop_in_one_bar_counts_the_stop():
    rows = [("X", 20260504, "BUY", 10, 100.0, 0, "LMT"), ("X", 20260505, "SELL", 10, 100.0, 0, "LMT")]
    fills, trips, bars = long_trip([("2026-05-04", [(100, 100.2, 99.8, 100)]),
                                    ("2026-05-05", [(100, 106, 96, 100)])], rows)
    result = simulate_trip(fills, bars, WhatIfRules(stop_pct=2, target_pct=5, slippage_bps=0))
    assert result["What-if exit"] == "Stop"
    target_only = simulate_trip(fills, bars, WhatIfRules(target_pct=5, slippage_bps=0))
    assert target_only["What-if exit"] == "Target" and target_only["What-if exit price"] == pytest.approx(105)


def test_no_averaging_down_skips_the_lower_buy_and_sells_only_what_is_held():
    rows = [("X", 20260504, "BUY", 100, 100.0, -1.0, "LMT"), ("X", 20260505, "BUY", 100, 94.0, -1.0, "LMT"),
            ("X", 20260505, "SELL", 200, 92.0, -2.0, "LMT")]
    fills, trips, bars = long_trip([("2026-05-04", DAY1), ("2026-05-05", [(95, 95, 93.5, 94), (94, 94, 91.5, 92)])],
                                   rows)
    actual = trips["Net P&L"].iloc[0]
    result = simulate_trip(fills, bars, WhatIfRules(no_averaging_down=True))
    assert result["Skipped buys"] == 1
    assert result["What-if P&L"] == pytest.approx(100 * (92 - 100) - 1.0 - 1.0)  # half the sale's commission
    assert actual == pytest.approx(100 * (92 - 100) + 100 * (92 - 94) - 4.0)


def test_fixed_size_scales_every_fill():
    fills, trips, bars = long_trip([("2026-05-04", DAY1), ("2026-05-05", DAY2)], LOSER)
    result = simulate_trip(fills, bars, WhatIfRules(fixed_size=5_000))
    assert result["What-if P&L"] == pytest.approx(trips["Net P&L"].iloc[0] / 2)


def test_time_limit_sells_at_the_close_of_the_entry_day():
    fills, trips, bars = long_trip([("2026-05-04", DAY1), ("2026-05-05", DAY2)], LOSER)
    result = simulate_trip(fills, bars, WhatIfRules(max_days=0))
    assert result["What-if exit"] == "Time limit" and result["What-if exit price"] == 99
    assert result["What-if exit date"] == pd.Timestamp("2026-05-04")


def test_short_trip_stop_is_above_the_entry():
    rows = [("X", 20260504, "SELL", 100, 100.0, 0, "LMT"), ("X", 20260505, "BUY", 100, 108.0, 0, "STP")]
    up = [(100, 100.5, 99.5, 100), (100, 101, 99.8, 100.9)]
    fills, trips, bars = long_trip([("2026-05-04", up), ("2026-05-05", [(103.5, 109, 103, 108)])], rows)
    result = simulate_trip(fills, bars, WhatIfRules(stop_pct=3, slippage_bps=0))
    assert result["What-if exit"] == "Stop (gap)" and result["What-if exit price"] == pytest.approx(103.5)
    assert result["What-if P&L"] == pytest.approx(-350 - order_costs(100, 103.5, sell=False))


def test_trade_times_place_fills_in_their_bars():
    csv = ("Symbol,DateTime,TradeDate,Buy/Sell,Quantity,Price,Commission\n"
           "X,20260504;113000,20260504,BUY,100,99.2,0\nX,20260505;100000,20260505,SELL,-100,96,0\n")
    trades, _ = load_ibkr_trades(io.StringIO(csv), name="t.csv")
    fills, trips = round_trips(trades)
    bars = pd.concat([hourly("2026-05-04", [(100, 101, 99, 99.5), (99.5, 99.6, 99, 99.3), (99.3, 99.4, 98.7, 99)]),
                      hourly("2026-05-05", [(98.5, 98.6, 95.5, 96)])])
    result = simulate_trip(fills, bars, WhatIfRules(stop_pct=1, slippage_bps=0))
    # The buy is in the 10:30 bar; the 11:30 bar's low 98.7 is below the 98.21 stop? No: 98.7 > 98.21, so the stop
    # fills on day 2 at the open, 98.5 >= 98.21 so it is touched intrabar at the stop price.
    assert not result["Times estimated"]
    assert result["What-if exit"] == "Stop" and result["What-if exit price"] == pytest.approx(99.2 * 0.99)


def test_missing_bars_leave_the_trip_unsimulated_and_sweep_runs():
    fills, trips, bars = long_trip([("2026-05-04", DAY1), ("2026-05-05", DAY2)], LOSER)
    missing = simulate(fills, trips, {}, WhatIfRules(stop_pct=2))
    assert missing["What-if exit"].iloc[0] == "No price data" and np.isnan(missing["What-if P&L"].iloc[0])
    sweep = stop_sweep(fills, trips, {"X": bars}, WhatIfRules(slippage_bps=0), stops=(2, 3))
    assert sweep["Stop %"].tolist() == ["None", "2%", "3%"]
    assert sweep["Net P&L"].iloc[0] == pytest.approx(trips["Net P&L"].iloc[0])
    assert sweep["Net P&L"].iloc[1] > sweep["Net P&L"].iloc[0]
