"""Run App.py headlessly with Yahoo Finance and SEC calls replaced by synthetic data."""

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
import yfinance as yf

from tests.conftest import make_bars

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP_PATH = str(Path(__file__).resolve().parents[1] / "App.py")


def fake_daily(start, end):
    index = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
    rng = np.random.default_rng(5)
    n = len(index)
    closes = np.linspace(80, 140, n) + 6 * np.sin(np.arange(n) / 15) + rng.normal(0, 0.8, n)
    volumes = rng.uniform(0.8e6, 1.2e6, n)
    # Breakout on the second-to-last bar, held on the last one, so the signal
    # survives whether or not the app drops today's in-progress bar.
    closes[-2] = closes[-26:-2].max() * 1.03
    closes[-1] = closes[-2] * 1.003
    volumes[-2] = 3.5e6
    return make_bars(closes, volumes, index=index, seed=5)


def fake_intraday():
    frames = []
    for day in pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=3):
        index = pd.date_range(f"{day:%Y-%m-%d} 09:30", periods=78, freq="5min", tz="America/New_York")
        frames.append(make_bars(np.linspace(140, 146, 78), index=index, spread=0.001, seed=day.day))
    return pd.concat(frames)


def fake_download(tickers, start=None, end=None, interval="1d", **kwargs):
    if interval == "5m":
        return fake_intraday()
    if isinstance(tickers, list):
        frames = {symbol: fake_daily(start, end) for symbol in tickers}
        return pd.concat(frames, axis=1).swaplevel(0, 1, axis=1)
    return fake_daily(start, end)


class FakeTicker:
    def __init__(self, symbol):
        self.symbol = symbol
        self.info = {"forwardPE": 25.0, "trailingPegRatio": 1.4, "sector": "Technology"}
        self.quarterly_income_stmt = pd.DataFrame()
        self.quarterly_cashflow = pd.DataFrame()

    def get_earnings_dates(self, limit=100):
        dates = pd.date_range(end=pd.Timestamp.today(), periods=28, freq="QS-FEB", tz="America/New_York")
        eps = np.linspace(0.8, 2.0, len(dates))
        return pd.DataFrame(
            {"EPS Estimate": eps * 0.97, "Reported EPS": eps, "Surprise(%)": 3.0},
            index=pd.Index(dates, name="Earnings Date"),
        )


def offline_urlopen(*args, **kwargs):
    raise OSError("network disabled in tests")


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(yf, "download", fake_download)
    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    monkeypatch.setattr(urllib.request, "urlopen", offline_urlopen)


def start_app(timeout=180):
    app = AppTest.from_file(APP_PATH, default_timeout=timeout)
    app.secrets["password"] = "pw"
    app.run()
    app.text_input[0].input("pw").run()
    return app


def open_tab(app, label):
    # AppTest cannot click a tab, and each run resets the tabs to the first one, so set the open tab every run.
    app.session_state["main_tab"] = label
    app.run()
    assert not app.exception, app.exception
    return app


def test_app_runs_both_tabs(offline):
    app = start_app()
    assert [tab.label for tab in app.tabs] == ["Stock Screener", "Market Analysis", "Industry Analysis",
                                               "Semicon Supply Chain", "Backtest",
                                               "Signal Analysis & Trade Plan", "Validate"]
    open_tab(app, "Signal Analysis & Trade Plan")
    next(box for box in app.selectbox if box.label == "Signal rule").select("Breakout (new high/low on volume)")
    open_tab(app, "Signal Analysis & Trade Plan")
    assert not app.exception, app.exception
    rendered = " ".join(
        [element.value for element in app.success] + [element.value for element in app.warning]
        + [element.value for element in app.info] + [element.value for element in app.error]
    )
    assert "AAPL" in rendered
    assert any("Trade plan" in element.value for element in app.subheader)


def test_validate_tab_runs(offline):
    app = open_tab(start_app(300), "Validate")
    next(box for box in app.text_input if box.label.startswith("Tickers")).input("AAA, BBB")
    open_tab(app, "Validate")
    next(button for button in app.button if button.label == "Run validation").click()
    open_tab(app, "Validate")
    assert not app.exception, app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert int(metrics["Signals replayed"].replace(",", "")) > 0
    assert any("Does the confidence score mean anything?" in element.value for element in app.subheader)


def test_one_signal_rule_drives_every_tab(offline):
    app = open_tab(start_app(), "Backtest")
    rule = next(box for box in app.selectbox if box.label == "Signal rule")
    assert rule.value == "Moving-average rule"
    assert any(box.label == "Fast MA Type" for box in app.selectbox)
    assert [box.label for box in app.number_input].count("Account size ($)") == 1

    rule.select("Breakout (new high/low on volume)")

    open_tab(app, "Backtest")
    assert not app.exception, app.exception
    assert not any(box.label == "Fast MA Type" for box in app.selectbox)
    assert any("Breakout (new high/low on volume) signals" in element.value for element in app.info)

    # The setting survives a visit to another tab and is shared by the strategy tabs.
    open_tab(app, "Market Analysis")
    assert not any(box.label == "Signal rule" for box in app.selectbox)
    open_tab(app, "Signal Analysis & Trade Plan")
    assert next(box for box in app.selectbox if box.label == "Signal rule").value == "Breakout (new high/low on volume)"


def test_setup_periods_are_editable(offline):
    app = open_tab(start_app(), "Backtest")
    rule = next(box for box in app.selectbox if box.label == "Signal rule")
    assert not any(char.isdigit() for option in rule.options for char in option)

    rule.select("Pullback in trend (dip to the EMA, then reversal)")

    open_tab(app, "Backtest")
    labels = [box.label for box in app.number_input]
    for label in ["Trend fast MA (bars)", "Trend slow MA (bars)", "Pullback EMA (bars)", "Swing high lookback (bars)",
                  "Pullback window (bars)", "Min dip depth (ATR)"]:
        assert label in labels
    next(box for box in app.number_input if box.label == "Pullback EMA (bars)").set_value(12)
    open_tab(app, "Backtest")
    assert not app.exception, app.exception

    next(box for box in app.selectbox if box.label == "Signal rule").select("Reversal (buy a breakdown in an uptrend)")

    open_tab(app, "Backtest")
    next(box for box in app.number_input if box.label == "Breakdown lookback (bars)").set_value(10)
    open_tab(app, "Backtest")
    assert not app.exception, app.exception
    assert any("prior 10-bar low" in element.value for element in app.caption)


def test_screener_tab_scans_a_custom_list(offline):
    app = start_app()
    assert app.sidebar.header[0].value == "Stock Screener"
    next(box for box in app.selectbox if box.label == "Universe").select("Custom list").run()
    next(button for button in app.button if button.key == "screener_load_btn").click().run()
    assert not app.exception, app.exception
    assert "stocks pass" in " ".join(element.value for element in app.success)
    assert [box.value for box in app.selectbox if box.label == "Signal group"] == ["Moving average"] * 3

    # Switching a row's group swaps the inputs after it.
    next(box for box in app.selectbox if box.label == "Signal group").select("P/E").run()
    assert not app.exception, app.exception
    assert any(box.label.startswith("P/E") for box in app.number_input)
    assert not any(box.label == "Left" for box in app.selectbox if (box.key or "").startswith("screen_0_"))

    next(button for button in app.button if button.label == "+ Add condition").click().run()
    assert [box.label for box in app.selectbox].count("Signal group") == 4
    next(button for button in app.button if button.key == "screen_0_remove_btn").click().run()
    assert [box.label for box in app.selectbox].count("Signal group") == 3

    # Conditions and loaded data survive a visit to another tab.
    open_tab(app, "Industry Analysis")
    open_tab(app, "Stock Screener")
    assert [box.label for box in app.selectbox].count("Signal group") == 3
    assert "stocks pass" in " ".join(element.value for element in app.success)


def test_market_and_industry_tabs(offline):
    app = open_tab(start_app(), "Market Analysis")
    assert app.sidebar.header[0].value == "Market Analysis"
    # Index trends come from the data preloaded at start; breadth waits for its button.
    assert any("Market filter" in element.value for element in list(app.success) + list(app.error))
    next(box for box in app.selectbox if box.label == "Breadth universe").select("Dow Jones 30")
    open_tab(app, "Market Analysis")
    next(button for button in app.button if button.key == "market_load_btn").click()
    open_tab(app, "Market Analysis")
    assert not app.exception, app.exception
    assert any(metric.label.startswith("Above SMA") for metric in app.metric)

    open_tab(app, "Industry Analysis")
    assert any("Industry indexes" in chart.proto.spec for chart in app.get("plotly_chart"))
    next(toggle for toggle in app.toggle if toggle.key == "industry_normalize").set_value(False)
    open_tab(app, "Industry Analysis")
    assert not app.exception, app.exception


def test_supply_chain_tab(offline):
    app = open_tab(start_app(), "Semicon Supply Chain")
    assert app.sidebar.header[0].value == "Semicon Supply Chain"
    assert any("End market" in box.label for box in app.selectbox)
    assert any('"AAPL"' in chart.proto.spec for chart in app.get("plotly_chart"))

    # Focusing a company opens its detail panel with the companies it buys from and sells to.
    next(box for box in app.selectbox if box.key == "supply_focus").select("TSM")
    open_tab(app, "Semicon Supply Chain")
    assert any(element.value.startswith("TSM") for element in app.subheader)
    rendered = " ".join(element.value for element in app.markdown)
    assert "Buys from" in rendered and "ASML" in rendered and "Sells to" in rendered

    next(box for box in app.selectbox if box.key == "supply_metric").select("% from 52w high")
    open_tab(app, "Semicon Supply Chain")
    assert not app.exception, app.exception

    # Renaming a group relabels its box and its index row.
    assert any(element.value == "Group indexes" for element in app.subheader)
    next(box for box in app.selectbox if box.key == "supply_rename_from").select("Equipment")
    next(box for box in app.text_input if box.key == "supply_rename_to").input("Tool makers")
    next(button for button in app.button if button.key == "supply_rename_btn").click()
    open_tab(app, "Semicon Supply Chain")
    assert not app.exception, app.exception
    assert "Tool makers" in set(app.session_state["supply_group_base"]["Group"])
    assert any("Tool makers" in chart.proto.spec for chart in app.get("plotly_chart"))
    next(button for button in app.button if button.key == "supply_reset_btn").click()
    open_tab(app, "Semicon Supply Chain")
    assert "Equipment" in set(app.session_state["supply_group_base"]["Group"])

    for chain in ["AI / data center", "Automotive", "All end markets"]:
        next(box for box in app.selectbox if box.key == "supply_chain").select(chain)
        open_tab(app, "Semicon Supply Chain")
        assert not app.exception, app.exception
        assert any('"NVDA"' in chart.proto.spec for chart in app.get("plotly_chart"))


def test_mobile_layout_has_no_sidebar(offline):
    app = start_app()
    app.session_state["layout_mode"] = "Mobile"
    app.run()
    assert not app.exception, app.exception
    assert len(app.sidebar.children) == 0
    expanders = [element.label for element in app.get("expander")]
    assert any("Conditions" in label for label in expanders)


def test_screener_survives_yahoo_blocking_fundamentals(offline, monkeypatch):
    class BlockedTicker(FakeTicker):
        @property
        def info(self):
            raise RuntimeError("Too Many Requests")

        @info.setter
        def info(self, value):
            pass

    monkeypatch.setattr(yf, "Ticker", BlockedTicker)
    st.cache_resource.clear()
    app = start_app()
    next(box for box in app.selectbox if box.label == "Universe").select("Custom list").run()
    next(button for button in app.button if button.key == "screener_load_btn").click().run()
    next(box for box in app.selectbox if box.label == "Signal group").select("Mkt cap").run()
    next(box for box in app.number_input if box.label.startswith("Mkt cap")).set_value(10.0).run()
    assert not app.exception, app.exception
    assert any("only 0 of" in element.value for element in app.warning)
    st.cache_resource.clear()
