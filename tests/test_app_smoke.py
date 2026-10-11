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
    closes[-1] = closes[-25:-1].max() * 1.03
    volumes[-1] = 3.5e6
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
    assert not app.exception, app.exception
    tab_labels = [tab.label for tab in app.tabs]
    assert tab_labels == ["Backtest", "Signal Analysis & Trade Plan", "Validate"]
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
