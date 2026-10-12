"""Run App.py headlessly with Yahoo Finance and SEC calls replaced by synthetic data."""

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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


def test_app_runs_both_tabs(offline):
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.secrets["password"] = "pw"
    app.run()
    app.text_input[0].input("pw").run()
    next(box for box in app.selectbox if box.label == "Signal rule").select("Breakout (new high/low on volume)").run()
    assert not app.exception, app.exception
    tab_labels = [tab.label for tab in app.tabs]
    assert tab_labels == ["Backtest", "Signal Analysis & Trade Plan", "Validate", "Stock Screener"]
    rendered = " ".join(
        [element.value for element in app.success] + [element.value for element in app.warning]
        + [element.value for element in app.info] + [element.value for element in app.error]
    )
    assert "TEST" not in rendered  # sanity: ticker default is AAPL
    assert "AAPL" in rendered
    assert any("Trade plan" in element.value for element in app.subheader)


def test_validate_tab_runs(offline):
    app = AppTest.from_file(APP_PATH, default_timeout=300)
    app.secrets["password"] = "pw"
    app.run()
    app.text_input[0].input("pw").run()
    tickers = next(box for box in app.text_input if box.label.startswith("Tickers"))
    tickers.input("AAA, BBB").run()
    next(button for button in app.button if button.label == "Run validation").click().run()
    assert not app.exception, app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert int(metrics["Signals replayed"].replace(",", "")) > 0
    assert any("Does the confidence score mean anything?" in element.value for element in app.subheader)


def test_one_signal_rule_drives_every_tab(offline):
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.secrets["password"] = "pw"
    app.run()
    app.text_input[0].input("pw").run()
    rule = next(box for box in app.selectbox if box.label == "Signal rule")
    assert rule.value == "Moving-average rule"
    assert any(box.label == "Fast MA Type" for box in app.selectbox)
    assert [box.label for box in app.number_input].count("Account size ($)") == 1

    rule.select("Breakout (new high/low on volume)").run()
    assert not app.exception, app.exception
    assert not any(box.label == "Fast MA Type" for box in app.selectbox)
    assert any("Breakout (new high/low on volume) signals" in element.value for element in app.info)


def test_setup_periods_are_editable(offline):
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.secrets["password"] = "pw"
    app.run()
    app.text_input[0].input("pw").run()
    rule = next(box for box in app.selectbox if box.label == "Signal rule")
    assert not any(char.isdigit() for option in rule.options for char in option)

    rule.select("Pullback in trend (dip to the EMA, then reversal)").run()
    labels = [box.label for box in app.number_input]
    for label in ["Trend fast MA (bars)", "Trend slow MA (bars)", "Pullback EMA (bars)", "Swing high lookback (bars)",
                  "Pullback window (bars)", "Min dip depth (ATR)"]:
        assert label in labels
    next(box for box in app.number_input if box.label == "Pullback EMA (bars)").set_value(12).run()
    assert not app.exception, app.exception

    next(box for box in app.selectbox if box.label == "Signal rule").select("Reversal (buy a breakdown in an uptrend)").run()
    next(box for box in app.number_input if box.label == "Breakdown lookback (bars)").set_value(10).run()
    assert not app.exception, app.exception
    assert any("prior 10-bar low" in element.value for element in app.caption)


def test_screener_tab_scans_a_custom_list(offline):
    app = AppTest.from_file(APP_PATH, default_timeout=180)
    app.secrets["password"] = "pw"
    app.run()
    app.text_input[0].input("pw").run()
    next(box for box in app.selectbox if box.label == "Stocks to scan").select("Custom list").run()
    load = next(button for button in app.button if button.label.startswith("Load data for"))
    load.click().run()
    assert not app.exception, app.exception
    headers = [element.value for element in app.header]
    assert [h for h in headers if h[0].isdigit()] == ["1. Market", "2. Industry", "3. Stock"]
    outcome = " ".join([element.value for element in app.success] + [element.value for element in app.error])
    assert "Market filter" in outcome
    assert "stocks pass" in outcome

    next(radio for radio in app.radio if radio.label == "Market").set_value("HK").run()
    assert not app.exception, app.exception
