import numpy as np
import pandas as pd

from analysis.screener import (
    breadth_history,
    evaluate_condition,
    fundamental_trend,
    index_trend,
    leaders_health,
    market_gate,
    screen_stocks,
    stock_snapshot,
)
from analysis.universe import (
    members_from_tables,
    normalize_symbol,
    normalize_ticker,
    parse_ticker_list,
)
from tests.conftest import make_bars


def rising(n=400, start=50.0, end=150.0, seed=0):
    return make_bars(np.linspace(start, end, n), seed=seed)


def falling(n=400, seed=0):
    return make_bars(np.linspace(150.0, 50.0, n), seed=seed)


def cond(left, left_period, operator, right, right_period):
    return {"left_kind": left, "left_period": left_period, "operator": operator,
            "right_kind": right, "right_period": right_period}


def test_conditions_on_trending_stocks():
    up, down = rising(), falling()
    assert evaluate_condition(up, cond("Close", None, ">", "SMA", 50))
    assert evaluate_condition(up, cond("SMA", 50, ">", "SMA", 200))
    assert not evaluate_condition(down, cond("SMA", 50, ">", "SMA", 200))
    assert evaluate_condition(down, cond("Close", None, "<", "EMA", 20))
    assert evaluate_condition(up, cond("Close", None, ">", "Value", 100))
    # Not enough history for a 500-bar MA counts as failing, not as an error.
    assert not evaluate_condition(up, cond("Close", None, ">", "SMA", 500))


def test_cross_above_within_window():
    closes = np.concatenate([np.linspace(120, 80, 150), np.linspace(80, 110, 40)])
    bars = make_bars(closes, seed=2)
    condition = cond("Close", None, "Cross Above", "SMA", 50)
    close, average = bars["Close"], bars["Close"].rolling(50).mean()
    first_above = int(np.argmax((close > average).to_numpy()))
    bars_since = len(bars) - 1 - first_above
    assert evaluate_condition(bars, condition, cross_window=bars_since + 1)
    assert not evaluate_condition(bars, condition, cross_window=max(1, bars_since - 1))


def test_snapshot_52_week_figures():
    bars = rising()
    snap = stock_snapshot(bars, ma_periods=(50, 200), recent_days=10)
    assert snap["% from 52w high"] <= 0 and snap["% from 52w high"] > -2
    assert snap["New 52w high in last 10d"]
    assert not snap["New 52w low in last 10d"]
    assert snap["Close vs SMA 200 %"] > 0
    assert index_trend(bars, 50, 200) == "Uptrend"
    assert index_trend(falling(), 50, 200) == "Downtrend"


def test_breadth_counts_new_highs_and_ma():
    prices = {"UP1": rising(seed=1), "UP2": rising(seed=2), "DOWN": falling(seed=3), "NEW": rising(n=100, seed=4)}
    breadth = breadth_history(prices, ma_periods=(50, 200), recent_days=10)
    last = breadth.iloc[-1]
    assert np.isclose(last["% above SMA 200"], 200 / 3)  # NEW has no 200-bar MA yet, so 2 of 3 eligible
    # NEW is too young for a 52-week high; UP1 and UP2 made one, DOWN made a low.
    assert np.isclose(last["% new 52w high in last 10d"], 200 / 3)
    assert np.isclose(last["% new 52w low in last 10d"], 100 / 3)


def test_market_gate_reports_each_check():
    index_rows = {"S&P 500": {"Close vs SMA 200 %": 5.0, "SMA 50": 110.0, "SMA 200": 100.0}}
    breadth_latest = pd.Series({"% above SMA 200": 40.0, "% new 52w high in last 20d": 8.0})
    passed, checks = market_gate(index_rows, breadth_latest, {
        "index": "S&P 500", "index_ma": 200, "index_ma_order": (50, 200),
        "min_pct_above_ma": (200, 50.0), "min_pct_new_highs": ("% new 52w high in last 20d", 5.0),
    })
    assert not passed
    assert [ok for _, ok in checks] == [True, True, False, True]
    assert market_gate(index_rows, breadth_latest, {"index": "S&P 500", "index_ma": 200})[0]
    # Values straight from pandas are numpy scalars; they must still count as passing.
    numpy_rows = {"S&P 500": {"Close vs SMA 200 %": np.float64(5.0)}}
    numpy_breadth = pd.Series({"% above SMA 200": np.float64(60.0)})
    assert market_gate(numpy_rows, numpy_breadth, {"index": "S&P 500", "index_ma": 200, "min_pct_above_ma": (200, 50.0)})[0]


def test_screen_stocks_filters_and_sorts():
    prices = {"UP": rising(seed=1), "DOWN": falling(seed=2), "CHEAP": rising(start=1, end=3, seed=3)}
    conditions = [cond("Close", None, ">", "SMA", 50), cond("SMA", 50, ">", "SMA", 200)]
    result = screen_stocks(prices, conditions, max_pct_from_high=10, min_price=5)
    assert result.loc["UP", "Pass"]
    assert not result.loc["DOWN", "Pass"]
    assert not result.loc["CHEAP", "Pass"]  # below the minimum price
    assert result.index[0] == "UP"
    assert "Close > SMA 50" in result.columns and "Within 10% of 52w high" in result.columns


def test_fundamental_trend_growth():
    quarters = pd.date_range("2023-03-31", periods=8, freq="QE")
    frame = pd.DataFrame({"Revenue": np.arange(100, 180, 10, dtype=float),
                          "Net_Income": np.arange(10, 18, 1, dtype=float)}, index=quarters)
    trend = fundamental_trend(frame)
    assert np.isclose(trend["Revenue YoY %"], (170 / 130 - 1) * 100)
    assert np.isclose(trend["Revenue TTM growth %"], ((140 + 150 + 160 + 170) / (100 + 110 + 120 + 130) - 1) * 100)
    assert trend["Net income rising 2Q"]
    assert np.isclose(trend["Net margin %"], 17 / 170 * 100)
    assert fundamental_trend(pd.DataFrame()) == {}


def test_leaders_health():
    table = pd.DataFrame({"Close vs SMA 50 %": [5, -2, 3], "Close vs SMA 200 %": [10, 1, -4],
                          "% from 52w high": [-1, -10, -4], "Return 3m %": [10, -5, 4]}, index=["A", "B", "C"])
    health = leaders_health(table, (50, 200))
    assert health["Above SMA 50"] == 2 and health["Above SMA 200"] == 2 and health["Leaders"] == 3


def test_ticker_normalisation():
    assert normalize_ticker("brk.b") == "BRK-B"
    assert normalize_ticker("700", "HK") == "0700.HK"
    assert normalize_ticker("9988.hk", "HK") == "9988.HK"
    assert parse_ticker_list("aapl, msft\nAAPL;  nvda") == ["AAPL", "MSFT", "NVDA"]


def test_members_from_wikipedia_style_table():
    table = pd.DataFrame({
        "Symbol": ["AAPL", "BRK.B"] * 3, "Security": ["Apple", "Berkshire"] * 3,
        "GICS Sector": ["Information Technology", "Financials"] * 3, "GICS Sub-Industry": ["Hardware", "Insurance"] * 3,
    })
    other = pd.DataFrame({"Date": [1, 2], "Added": ["X", "Y"]})
    members = members_from_tables([other, table], min_rows=2)
    assert members["Ticker"].tolist() == ["AAPL", "BRK-B"]
    assert members.loc[1, "Sector"] == "Financials"


def test_fundamental_trend_skips_comparisons_across_missing_quarters():
    # Yahoo often returns HK quarters with gaps: Q2 2025, then Q2 2026 only.
    dates = pd.to_datetime(["2025-06-30", "2025-09-30", "2026-06-30"])
    frame = pd.DataFrame({"Revenue": [100.0, np.nan, 120.0], "Net_Income": [10.0, 11.0, 12.0]}, index=dates)
    trend = fundamental_trend(frame)
    assert np.isclose(trend["Revenue YoY %"], 20.0)
    assert np.isclose(trend["Net income YoY %"], 20.0)
    assert "Net income TTM" not in trend  # only three quarters, not consecutive
    assert "Net income rising 2Q" not in trend


def test_symbol_normalisation_without_a_market():
    assert normalize_symbol(" brk.b ") == "BRK-B"
    assert normalize_symbol("700") == "0700.HK"
    assert normalize_symbol("0700.hk") == "0700.HK"
    assert normalize_symbol("SHOP.TO") == "SHOP.TO"
    assert normalize_symbol("^GSPC") == "^GSPC"


def test_members_table_with_footnoted_headers():
    table = pd.DataFrame({
        "Ticker": ["ADBE", "AMD"] * 3, "Company": ["Adobe", "AMD"] * 3,
        "ICB Industry[1]": ["Technology"] * 6, "ICB Subsector[1]": ["Software", "Semiconductors"] * 3,
    })
    members = members_from_tables([table], min_rows=2)
    assert members.loc[0, "Sector"] == "Technology"
    assert members.loc[1, "Industry"] == "Semiconductors"
