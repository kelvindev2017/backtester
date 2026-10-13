import numpy as np
import pandas as pd

from analysis.screener import condition_label, screen_stocks, stock_snapshot
from analysis.signal_groups import (
    earnings_dates,
    evaluate_group_condition,
    fundamental_row,
    needs_fundamentals,
    overlay_frame,
    period_start,
)
from tests.conftest import make_bars

TODAY = pd.Timestamp("2026-10-13")


def epoch(day):
    return pd.Timestamp(day, tz="UTC").timestamp()


def test_fundamental_row_scales_and_maps_yahoo_fields():
    info = {
        "quoteType": "EQUITY", "marketCap": 4.98e12, "trailingPE": 39.1, "earningsGrowth": 0.287,
        "dividendYield": 0.32, "revenueGrowth": 0.164, "trailingPegRatio": 2.7, "returnOnEquity": 1.49,
        "beta": 1.07, "sector": "Technology", "industry": "Consumer Electronics", "recommendationKey": "strong_buy",
        "earningsTimestamp": epoch("2026-11-02"),
    }
    row = fundamental_row(info, today=TODAY)
    assert round(row["Mkt cap ($B)"]) == 4980
    assert round(row["EPS growth YoY %"], 1) == 28.7
    assert row["Div yield %"] == 0.32
    assert round(row["ROE %"]) == 149
    assert row["Yahoo sector"] == "Information Technology"
    assert row["Analyst rating"] == "Strong buy"
    assert row["Last earnings"] is None and row["Next earnings"] == pd.Timestamp("2026-11-02")
    no_dividend = fundamental_row({k: v for k, v in info.items() if k != "dividendYield"}, today=TODAY)
    assert no_dividend["Div yield %"] == 0.0
    assert fundamental_row({"quoteType": "ETF", "marketCap": 1e9}) == {}
    assert fundamental_row({}) == {}


def test_earnings_dates_split_past_and_future():
    info = {"earningsTimestamp": epoch("2026-09-30"), "earningsTimestampStart": epoch("2026-12-18"),
            "earningsTimestampEnd": epoch("2026-12-18")}
    assert earnings_dates(info, TODAY) == (pd.Timestamp("2026-09-30"), pd.Timestamp("2026-12-18"))


def test_number_category_and_date_conditions():
    row = pd.Series({"P/E": 22.0, "Return 3m %": -4.0, "Sector": "Energy", "Index": ["S&P 500", "Dow Jones 30"],
                     "Last earnings": pd.Timestamp("2026-10-01"), "Next earnings": pd.Timestamp("2026-10-20"), "PEG": np.nan})
    assert evaluate_group_condition(row, {"group": "P/E", "operator": "Between", "value": 30, "value2": 10})
    assert not evaluate_group_condition(row, {"group": "P/E", "operator": ">", "value": 25})
    assert not evaluate_group_condition(row, {"group": "PEG", "operator": "<", "value": 5})  # missing data fails
    assert evaluate_group_condition(row, {"group": "Perf %", "period": "3M", "operator": "<", "value": 0})
    assert evaluate_group_condition(row, {"group": "Sector", "operator": "is any of", "values": ["Energy", "Utilities"]})
    assert not evaluate_group_condition(row, {"group": "Sector", "operator": "is none of", "values": ["Energy"]})
    assert evaluate_group_condition(row, {"group": "Index", "operator": "is any of", "values": ["Dow Jones 30"]})
    assert evaluate_group_condition(row, {"group": "Recent earnings date", "days": 14}, today=TODAY)
    assert not evaluate_group_condition(row, {"group": "Recent earnings date", "days": 7}, today=TODAY)
    assert evaluate_group_condition(row, {"group": "Upcoming earnings date", "days": 7}, today=TODAY)
    assert needs_fundamentals([{"group": "P/E"}]) and not needs_fundamentals([{"group": "Price"}, {"group": "Sector"}])
    assert condition_label({"group": "P/E", "operator": "Between", "value": 10, "value2": 30}) == "P/E 10 to 30"


def test_screen_stocks_with_group_conditions_and_extra_columns():
    prices = {name: make_bars(np.linspace(50, 150, 300), seed=i) for i, name in enumerate(["AAA", "BBB", "CCC"])}
    extra = pd.DataFrame({"P/E": [12.0, 40.0, np.nan], "Sector": ["Energy", "Energy", "Financials"]},
                         index=["AAA", "BBB", "CCC"])
    conditions = [
        {"group": "Moving average", "left_kind": "Close", "left_period": None, "operator": ">", "right_kind": "SMA", "right_period": 50},
        {"group": "P/E", "operator": "<", "value": 20},
        {"group": "Sector", "operator": "is any of", "values": ["Energy"]},
    ]
    table = screen_stocks(prices, conditions, extra=extra)
    assert table["Pass"].to_dict() == {"AAA": True, "BBB": False, "CCC": False}
    assert {"P/E < 20", "Sector is Energy", "P/E", "Sector"} <= set(table.columns)


def test_snapshot_ytd_and_one_year_returns():
    index = pd.bdate_range("2025-06-02", periods=330)
    frame = make_bars(np.linspace(100, 200, 330), index=index, seed=1)
    snapshot = stock_snapshot(frame)
    last_2025 = frame["Close"].loc[:"2025-12-31"].iloc[-1]
    assert abs(snapshot["Return YTD %"] - (frame["Close"].iloc[-1] / last_2025 - 1) * 100) < 1e-9
    assert snapshot["Return 1y %"] > 0


def test_overlay_rebases_each_line_to_100():
    index = pd.bdate_range("2025-01-01", periods=300)
    prices = {"A": make_bars(np.linspace(10, 20, 300), index=index), "B": make_bars(np.linspace(1000, 500, 300), index=index)}
    start = period_start(index[-1], "6M")
    overlay = overlay_frame(prices, start=start)
    assert overlay.index[0] >= start
    assert np.allclose(overlay.iloc[0], 100)
    assert overlay["A"].iloc[-1] > 100 > overlay["B"].iloc[-1]
    raw = overlay_frame(prices, start=start, normalize=False)
    assert raw["B"].iloc[0] > 500
    assert period_start(pd.Timestamp("2026-10-13"), "YTD") == pd.Timestamp("2026-01-01")


def test_fundamental_columns_match_row():
    from analysis.signal_groups import FUNDAMENTAL_COLUMNS, fundamental_row

    row = fundamental_row({"quoteType": "EQUITY", "marketCap": 2e12, "sector": "Technology"})
    assert list(row) == FUNDAMENTAL_COLUMNS
