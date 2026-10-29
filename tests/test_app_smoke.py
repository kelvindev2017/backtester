"""Run App.py headlessly with Yahoo Finance and SEC calls replaced by synthetic data."""

import json
from types import SimpleNamespace
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
    if interval in ("1m", "5m", "15m"):
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
        self.revenue_estimate = pd.DataFrame({"avg": [12e9, 13e9]}, index=pd.Index(["0q", "+1q"], name="period"))
        self.earnings_estimate = pd.DataFrame({"avg": [2.1, 9.0], "growth": [0.2, 0.25]}, index=["0q", "+1y"])

    def get_earnings_dates(self, limit=100):
        # Midnight dates: a time of day between 1 and 2 am lands in the ambiguous hour when DST ends on Nov 1.
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=28, freq="QS-FEB", tz="America/New_York")
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
    assert [tab.label for tab in app.tabs] == ["Economic analysis", "Market Analysis", "Industry Analysis",
                                               "Stock Fundamentals", "Semicon Supply Chain", "Stock Screener", "Backtest",
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


def test_economic_analysis_tab_opens_first(offline, monkeypatch):
    import analysis.economy
    fred = {
        "CPIAUCSL": pd.Series(np.linspace(250, 320, 120), index=pd.date_range("2016-01-01", periods=120, freq="MS")),
        "UNRATE": pd.Series(np.linspace(5, 4, 120), index=pd.date_range("2016-01-01", periods=120, freq="MS")),
        "USREC": pd.Series([0] * 50 + [1] * 3 + [0] * 67, index=pd.date_range("2016-01-01", periods=120, freq="MS")),
    }

    def fake_fred(series_id, timeout=30):
        if series_id not in fred:
            raise OSError("not in the test data")
        return fred[series_id]

    monkeypatch.setattr(analysis.economy, "fred_series", fake_fred)
    app = start_app()
    assert not app.exception, app.exception
    assert app.sidebar.header[0].value == "Economic analysis"
    assert [element.value for element in app.subheader] == ["Asset classes compared", "S&P 500 vs the US economy"]
    assert any("Correlation with Stock (S&P 500)" in str(frame.value.columns) for frame in app.dataframe)
    # FEDFUNDS is missing from the fake data: a warning, and the other panels still draw.
    assert any("FEDFUNDS" in element.value for element in app.warning)
    assert any("Unemployment rate: 4.00%" in element.value for element in app.markdown)

    # Each asset against its own high.
    next(radio for radio in app.radio if radio.key == "economy_asset_scale").set_value("Highest = 100%")
    app = open_tab(app, "Economic analysis")
    assert any("highest value" in chart.proto.spec for chart in app.get("plotly_chart"))

    # Stock (S&P 500) can be removed like any other asset; the correlation panel and column go with it.
    next(box for box in app.multiselect if box.key == "economy_assets").unselect("Stock (S&P 500)")
    app = open_tab(app, "Economic analysis")
    assert not any("Correlation" in str(frame.value.columns) for frame in app.dataframe)
    assert any("Gold (GLD)" in str(frame.value.index) for frame in app.dataframe)


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
    app = open_tab(start_app(), "Stock Screener")
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


def test_market_and_industry_tabs(offline, monkeypatch):
    import analysis.universe
    # Only the Dow list is built in; the others come from the web, so give Nasdaq-100 a small fake list.
    real_members = analysis.universe.load_us_index_members
    monkeypatch.setattr(analysis.universe, "load_us_index_members",
                        lambda name: real_members("Dow Jones 30").head(8) if name == "Nasdaq-100" else real_members(name))
    app = open_tab(start_app(), "Market Analysis")
    assert app.sidebar.header[0].value == "Market Analysis"
    # Index trends come from the data preloaded at start; breadth waits for its button.
    assert any("Market filter" in element.value for element in list(app.success) + list(app.error))
    next(box for box in app.selectbox if box.key == "market_index").select("Dow Jones")
    open_tab(app, "Market Analysis")
    next(button for button in app.button if button.key == "market_load_btn").click()
    open_tab(app, "Market Analysis")
    assert not app.exception, app.exception
    assert any(metric.label.startswith("Above SMA") for metric in app.metric)
    assert any(header.value == "Market breadth: Dow Jones" for header in app.subheader)
    # Once breadth was loaded, switching index loads and shows that index's breadth without the button.
    next(box for box in app.selectbox if box.key == "market_index").select("Nasdaq-100")
    open_tab(app, "Market Analysis")
    assert not app.exception, app.exception
    assert any(header.value == "Market breadth: Nasdaq-100" for header in app.subheader)
    assert any(metric.label.startswith("Above SMA") for metric in app.metric)
    next(box for box in app.selectbox if box.key == "market_index").select("Russell 2000")
    open_tab(app, "Market Analysis")
    assert not app.exception, app.exception
    assert not any(metric.label.startswith("Above SMA") for metric in app.metric)

    open_tab(app, "Industry Analysis")
    assert any("Industry indexes" in chart.proto.spec for chart in app.get("plotly_chart"))
    # Time charts get the TradingView-style value labels, as in Economic analysis.
    chart = next(chart for chart in app.get("plotly_chart") if "Industry indexes" in chart.proto.spec)
    assert json.loads(chart.proto.spec)["layout"]["meta"] == {"axis_hover": True}
    next(toggle for toggle in app.toggle if toggle.key == "industry_normalize").set_value(False)
    open_tab(app, "Industry Analysis")
    assert not app.exception, app.exception


def test_industry_indexes_can_be_added_and_removed(offline):
    app = open_tab(start_app(), "Industry Analysis")
    chosen = lambda: next(box for box in app.multiselect if box.key == "industry_indexes").value
    assert "Software (IGV)" in chosen()

    next(box for box in app.text_input if box.key == "industry_add_symbol").input("smh")
    next(box for box in app.text_input if box.key == "industry_add_name").input("Semiconductor ETF")
    next(button for button in app.button if button.key == "industry_add_btn").click()
    app = open_tab(app, "Industry Analysis")
    assert "Semiconductor ETF (SMH)" in chosen()
    assert next(box for box in app.text_input if box.key == "industry_add_symbol").value == ""
    # A built-in index typed by its symbol is just chosen, not added twice.
    next(box for box in app.text_input if box.key == "industry_add_symbol").input("XLK")
    next(button for button in app.button if button.key == "industry_add_btn").click()
    app = open_tab(app, "Industry Analysis")
    assert "Technology (XLK)" in chosen()

    next(box for box in app.multiselect if box.key == "industry_indexes").unselect("Software (IGV)")
    app = open_tab(app, "Industry Analysis")
    assert "Software (IGV)" not in chosen()
    assert json.loads(app.session_state["industry_saved"])["added"] == {"Semiconductor ETF (SMH)": "SMH"}

    next(button for button in app.button if button.key == "industry_reset_btn").click()
    app = open_tab(app, "Industry Analysis")
    assert "Software (IGV)" in chosen() and "Semiconductor ETF (SMH)" not in chosen()


def test_leading_companies_are_found_for_a_fund_and_for_a_stock(offline, monkeypatch):
    class Ticker(FakeTicker):
        @property
        def funds_data(self):
            if self.symbol != "SOXX":
                raise ValueError("not a fund")
            holdings = pd.DataFrame({"Holding Percent": [0.1, 0.09, 0.08, 0.07]}, index=["INTC", "AMD", "MU", "NVDA"])
            return SimpleNamespace(top_holdings=holdings)

        @property
        def info(self):
            return {**FakeTicker(self.symbol).info, "industryKey": "space"}

        @info.setter
        def info(self, value):
            pass

    monkeypatch.setattr(yf, "Ticker", Ticker)
    monkeypatch.setattr(yf, "Industry", lambda key: SimpleNamespace(
        top_companies=pd.DataFrame({"market weight": [0.4, 0.3, 0.2]}, index=["LMT", "RKLB", "NOC"])), raising=False)
    app = open_tab(start_app(), "Industry Analysis")
    leaders = lambda: next(box for box in app.text_input if box.key == "industry_leaders").value

    # An index stands in for the ETF that tracks it.
    next(box for box in app.selectbox if box.key == "industry_leaders_of").select("Semiconductors (PHLX ^SOX)")
    app = open_tab(app, "Industry Analysis")
    assert leaders() == "INTC, AMD, MU"
    assert any(header.value == "Leading companies: Semiconductors (PHLX ^SOX)" for header in app.subheader)

    # A stock: itself and the two largest others in its industry.
    next(box for box in app.text_input if box.key == "industry_add_symbol").input("RKLB")
    next(button for button in app.button if button.key == "industry_add_btn").click()
    app = open_tab(app, "Industry Analysis")
    next(box for box in app.selectbox if box.key == "industry_leaders_of").select("RKLB")
    app = open_tab(app, "Industry Analysis")
    assert leaders() == "RKLB, LMT, NOC"


def test_leading_companies_fall_back_to_the_built_in_list(offline):
    # Yahoo gives no holdings (FakeTicker has none): the built-in list for the built-in industries.
    app = open_tab(start_app(), "Industry Analysis")
    next(box for box in app.selectbox if box.key == "industry_leaders_of").select("Financials (XLF)")
    app = open_tab(app, "Industry Analysis")
    assert next(box for box in app.text_input if box.key == "industry_leaders").value == "BRK-B, JPM, V"


def test_industry_index_without_history_is_refused(offline, monkeypatch):
    # Yahoo has only today's quote for some indexes, such as ^SPACE.
    def download(tickers, *args, **kwargs):
        prices = fake_download(tickers, *args, **kwargs)
        return prices.tail(1) if "^SPACE" in tickers else prices
    monkeypatch.setattr(yf, "download", download)
    app = open_tab(start_app(), "Industry Analysis")
    next(box for box in app.text_input if box.key == "industry_add_symbol").input("^SPACE")
    next(button for button in app.button if button.key == "industry_add_btn").click()
    app = open_tab(app, "Industry Analysis")
    assert "^SPACE" not in next(box for box in app.multiselect if box.key == "industry_indexes").value
    assert any("no daily price history for ^SPACE" in element.value for element in app.warning)

    # One saved before this check is listed as not charted.
    app.session_state["industry_added"] = {"^SPACE": "^SPACE"}
    app.session_state["industry_indexes"] = ["Software (IGV)", "^SPACE"]
    app = open_tab(app, "Industry Analysis")
    assert any("Not charted" in element.value and "^SPACE" in element.value for element in app.warning)


class FakeTickerWithRevenue(FakeTicker):
    def __init__(self, symbol):
        super().__init__(symbol)
        quarters = pd.date_range(end=pd.Timestamp.today(), periods=8, freq="QE")
        self.quarterly_income_stmt = pd.DataFrame(
            [np.linspace(8e9, 11e9, 8), np.linspace(2e9, 3e9, 8)], index=["Total Revenue", "Net Income"], columns=quarters
        )


def test_supply_chain_tab(offline, monkeypatch):
    monkeypatch.setattr(yf, "Ticker", FakeTickerWithRevenue)
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
    # Revenue and trailing P/E share a date axis; analyst estimates are drawn as their own, labelled bars.
    assert any("Revenue (reported)" in chart.proto.spec and "analyst estimate" in chart.proto.spec
               for chart in app.get("plotly_chart"))
    assert any("percentile" in element.value for element in app.caption)
    assert any("Forward PEG 1.00" in element.value for element in app.caption)
    assert any("EPS (reported)" in chart.proto.spec and "Price YoY" in chart.proto.spec
               for chart in app.get("plotly_chart"))

    next(box for box in app.selectbox if box.key == "supply_metric").select("% from 52w high")
    open_tab(app, "Semicon Supply Chain")
    assert not app.exception, app.exception

    for chain in ["AI / data center", "Automotive", "All end markets"]:
        next(box for box in app.selectbox if box.key == "supply_chain").select(chain)
        open_tab(app, "Semicon Supply Chain")
        assert not app.exception, app.exception
        assert any('"NVDA"' in chart.proto.spec for chart in app.get("plotly_chart"))


def test_stock_fundamentals_tab(offline, monkeypatch):
    monkeypatch.setattr(yf, "Ticker", FakeTickerWithRevenue)
    app = open_tab(start_app(), "Stock Fundamentals")
    assert app.sidebar.header[0].value == "Stock Fundamentals"
    assert app.subheader[0].value == "NVDA"
    assert any("Price YoY" in chart.proto.spec and "Daily OHLC" in chart.proto.spec and "Trailing P" in chart.proto.spec
               for chart in app.get("plotly_chart"))

    # Any ticker and history length.
    next(box for box in app.text_input if box.key == "fundamentals_ticker").input(" msft ")
    next(slider for slider in app.select_slider if slider.key == "fundamentals_years").set_value(3)
    next(box for box in app.checkbox if box.key == "fundamentals_log_price").check()
    open_tab(app, "Stock Fundamentals")
    assert app.subheader[0].value == "MSFT"
    assert any("last 3 years" in element.value for element in app.markdown)

    next(box for box in app.text_input if box.key == "fundamentals_ticker").input("")
    open_tab(app, "Stock Fundamentals")
    assert any("Enter a ticker" in element.value for element in app.info)

    # A symbol Yahoo doesn't know returns no prices: a warning, not a crash.
    monkeypatch.setattr(yf, "download", lambda tickers, *args, **kwargs: (
        pd.DataFrame() if tickers == "SKNX" else fake_download(tickers, *args, **kwargs)))
    next(box for box in app.text_input if box.key == "fundamentals_ticker").input("SKNX")
    open_tab(app, "Stock Fundamentals")
    assert any("no price history for SKNX" in element.value for element in app.warning)


def test_mobile_layout_has_no_sidebar(offline):
    app = start_app()
    app.session_state["layout_mode"] = "Mobile"
    open_tab(app, "Stock Screener")
    assert not app.exception, app.exception
    assert len(app.sidebar.children) == 0
    expanders = [element.label for element in app.get("expander")]
    assert any("Conditions" in label for label in expanders)


def test_mobile_charts_keep_their_zoom_for_touch_gestures(offline):
    # Swipes scroll the page and two fingers zoom in the browser (CHART_TOUCH_JS), which needs Plotly's
    # scroll zoom; the charts themselves keep Plotly's zoom and pan, and there is no Display setting for it.
    app = start_app()
    app.session_state["layout_mode"] = "Mobile"
    app = open_tab(app, "Industry Analysis")
    chart = app.get("plotly_chart")[0].proto
    layout = json.loads(chart.spec)["layout"]
    assert "dragmode" not in layout and not layout.get("xaxis", {}).get("fixedrange")
    assert json.loads(chart.config)["scrollZoom"] is True
    assert not any(radio.key == "chart_touch" for radio in app.radio)

    # On desktop a mouse wheel still scrolls the page.
    app.session_state["layout_mode"] = "Desktop"
    app = open_tab(app, "Industry Analysis")
    assert not json.loads(app.get("plotly_chart")[0].proto.config or "{}").get("scrollZoom")


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
    app = open_tab(start_app(), "Stock Screener")
    next(box for box in app.selectbox if box.label == "Universe").select("Custom list").run()
    next(button for button in app.button if button.key == "screener_load_btn").click().run()
    next(box for box in app.selectbox if box.label == "Signal group").select("Mkt cap").run()
    next(box for box in app.number_input if box.label.startswith("Mkt cap")).set_value(10.0).run()
    assert not app.exception, app.exception
    assert any("only 0 of" in element.value for element in app.warning)
    st.cache_resource.clear()


def test_what_happened_next_settings(offline, monkeypatch):
    monkeypatch.setattr(yf, "Ticker", FakeTickerWithRevenue)
    app = open_tab(start_app(), "Stock Fundamentals")
    assert any("What happened next" in element.value for element in app.markdown)
    assert any("P/E percentile" in element.value for element in app.caption)
    assert any("How one dot is calculated" in chart.proto.spec for chart in app.get("plotly_chart"))

    for frequency, look_ahead in [("Week", "1 month"), ("Day", "1 week")]:
        next(radio for radio in app.radio if radio.key == "next_frequency").set_value(frequency)
        next(slider for slider in app.select_slider if slider.key == "next_look_ahead").set_value(look_ahead)
        open_tab(app, "Stock Fundamentals")
        assert not app.exception, app.exception
        assert any(f"in the {look_ahead} after each check" in chart.proto.spec for chart in app.get("plotly_chart"))


def test_day_breakout_signal_rule(offline):
    app = open_tab(start_app(), "Backtest")
    next(box for box in app.selectbox if box.label == "Signal rule").select(
        "Day breakout (above the N-day high, hold on a top-third close)")
    open_tab(app, "Backtest")
    assert any(box.key == "signal_day_breakout_lookback" for box in app.number_input)
    assert any("Day breakout" in element.value for element in app.info)
    open_tab(app, "Signal Analysis & Trade Plan")
    open_tab(app, "Validate")


def test_settings_show_only_what_the_tab_and_rule_use(offline):
    app = open_tab(start_app(), "Backtest")
    labels = lambda: [w.label for w in list(app.number_input) + list(app.slider) + list(app.selectbox) + list(app.checkbox)]
    assert "Risk per trade (% of equity)" not in labels()
    assert "Allow short trades (otherwise sell signals mean exit)" not in labels()

    open_tab(app, "Signal Analysis & Trade Plan")
    assert {"Risk per trade (% of equity)", "Full-size confidence threshold", "Intraday opening range (minutes)",
            "Allow short trades (otherwise sell signals mean exit)"} <= set(labels())

    open_tab(app, "Validate")
    assert "Full-size confidence threshold" in labels()
    assert "Intraday opening range (minutes)" not in labels() and "Account size ($)" not in labels()

    next(box for box in app.selectbox if box.label == "Signal rule").select(
        "Day breakout (above the N-day high, hold on a top-third close)")
    open_tab(app, "Signal Analysis & Trade Plan")
    assert "Allow short trades (otherwise sell signals mean exit)" not in labels()
    assert any("How the rule trades" in chart.proto.spec for chart in app.get("plotly_chart"))
    for rule in ["Moving-average rule", "Breakout (new high/low on volume)",
                 "Pullback in trend (dip to the EMA, then reversal)", "Reversal (buy a breakdown in an uptrend)"]:
        next(box for box in app.selectbox if box.label == "Signal rule").select(rule)
        open_tab(app, "Signal Analysis & Trade Plan")
        assert any("How the rule trades" in chart.proto.spec for chart in app.get("plotly_chart")), rule


def test_sliders_have_a_number_box_in_sync(offline):
    app = open_tab(start_app(), "Backtest")
    next(box for box in app.selectbox if box.label == "Signal rule").select(
        "Day breakout (above the N-day high, hold on a top-third close)")
    open_tab(app, "Backtest")
    slider = lambda: next(widget for widget in app.slider if widget.key == "signal_day_breakout_exit_pct")
    box = lambda: next(widget for widget in app.number_input if widget.key == "signal_day_breakout_exit_pct_input")
    assert slider().value == box().value == 33

    box().set_value(40)
    open_tab(app, "Backtest")
    assert slider().value == 40 and not app.exception

    slider().set_value(20)
    open_tab(app, "Backtest")
    assert box().value == 20

    open_tab(app, "Signal Analysis & Trade Plan")
    full = next(widget for widget in app.number_input if widget.key == "full_size_confidence_input")
    full.set_value(80)
    open_tab(app, "Signal Analysis & Trade Plan")
    assert next(widget for widget in app.slider if widget.key == "full_size_confidence").value == 80


def test_first_bar_day_trade_is_a_backtest_rule(offline):
    app = open_tab(start_app(), "Backtest")
    next(box for box in app.selectbox if box.label == "Signal rule").select(
        "First-bar day trade (the first five-minute bar decides the day)")
    open_tab(app, "Backtest")
    labels = lambda: [w.label for w in list(app.selectbox) + list(app.date_input)]
    # The timeframe offers intraday bars and the dates do not apply; the Account size is its capital.
    assert next(box for box in app.selectbox if box.label == "Timeframe").options == ["1 min", "5 min", "15 min"]
    assert "Date Range" not in labels()
    assert any(box.label == "Account size ($)" for box in app.number_input)
    assert any(box.key == "opening_bar_fill" for box in app.selectbox)
    assert any(box.key == "opening_bar_commissions" for box in app.selectbox)
    assert any("long days" in block.value for block in app.markdown)
    # The usual Backtest chart, with the day trades as its signals.
    spec = next(chart.proto.spec for chart in app.get("plotly_chart") if "Price, Indicators and Trade Signals" in chart.proto.spec)
    # Every intraday bar is drawn as a candle.
    assert '"candlestick"' in spec and "5-min bars" in spec
    next(box for box in app.selectbox if box.label == "Timeframe").select("15 min")
    open_tab(app, "Backtest")
    assert any("15-minute bars" in block.value for block in app.markdown)
    next(box for box in app.selectbox if box.key == "opening_bar_exit").select(
        "Lower high: a bar's high below the previous bar's high")
    open_tab(app, "Backtest")
    # The fill choice belongs to the pullback exit only.
    assert not any(box.key == "opening_bar_fill" for box in app.selectbox)
    next(box for box in app.checkbox if box.key == "opening_bar_short").check()
    open_tab(app, "Backtest")
    assert any("short days" in block.value for block in app.markdown)
    # A custom exit: two values and how they compare.
    next(box for box in app.selectbox if box.key == "opening_bar_exit").select(
        "Custom: compare two values")
    open_tab(app, "Backtest")
    assert any("Close < Previous low" in block.value for block in app.caption)
    assert not any(box.key == "opening_bar_custom_ema" for box in app.number_input)
    next(box for box in app.selectbox if box.key == "opening_bar_custom_operator").select("crosses below")
    next(box for box in app.selectbox if box.key == "opening_bar_custom_right").select("EMA of closes (this session)")
    open_tab(app, "Backtest")
    assert any(box.key == "opening_bar_custom_ema" for box in app.number_input)
    assert any("Close crosses below EMA 9 of closes" in block.value for block in app.caption)
    # The other strategy tabs cannot replay it and say so.
    open_tab(app, "Validate")
    assert any("Backtest tab only" in element.value for element in app.info)
    open_tab(app, "Signal Analysis & Trade Plan")
    assert any("Backtest tab only" in element.value for element in app.info)
    next(box for box in app.selectbox if box.label == "Signal rule").select("Moving-average rule")
    open_tab(app, "Backtest")
    assert next(box for box in app.selectbox if box.label == "Timeframe").value == "Daily"
