import base64
import io
import json
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest

from analysis import trade_store
from analysis.pretrade import (
    FAIL, PASS, WARN, day_pnl, is_hk, position_status, price_levels, run_checks, size_trade, verdict,
)
from analysis.trade_store import GitHubStore, LocalStore, StoreConflict, StoreError, make_store
from tests.conftest import make_bars

TODAY = pd.Timestamp("2026-11-19")


def breakout_bars(breaks=True):
    closes = np.linspace(80, 100, 80)
    closes[-1] = 104 if breaks else 98
    return make_bars(closes, spread=0.002, index=pd.bdate_range(end=TODAY, periods=80))


def by_name(checks):
    return {check.name: check for check in checks}


def test_hk_symbols():
    assert is_hk("0700.HK") and is_hk("700") and is_hk("9988") and is_hk("700 SEHK")
    assert not is_hk("NVDA") and not is_hk("BRK.B")


def test_levels_use_the_bars_before_today():
    levels = price_levels(breakout_bars())
    bars = breakout_bars()
    assert levels["high_n"] == pytest.approx(bars["High"].iloc[-21:-1].max())
    assert levels["price"] == pytest.approx(104)
    assert levels["sma_trend"] == pytest.approx(bars["Close"].iloc[-50:].mean())
    assert price_levels(bars.tail(30)) is None


def test_all_rules_pass_on_a_clean_breakout():
    checks = run_checks("nvda", price_levels(breakout_bars()), [], TODAY, open_positions=3, max_positions=15)
    assert all(check.status == PASS for check in checks), checks
    assert verdict(checks) == "go"


def test_each_rule_can_fail():
    closed = [
        {"Symbol": "NVDA", "Shares": 10, "Entry price": 100, "Exit price": 95, "Exit date": "2026-11-19", "P&L": -50},
        {"Symbol": "AMD", "Shares": 10, "Entry price": 100, "Exit price": 101, "Exit date": "2026-11-18", "P&L": 10},
    ]
    checks = by_name(run_checks("NVDA", price_levels(breakout_bars(False)), closed, TODAY, open_positions=15,
                                max_positions=15))
    assert checks["Breaks its 20-day high"].status == FAIL
    assert "buy stop" in checks["Breaks its 20-day high"].detail
    assert checks["Not a stock you lost on today"].status == FAIL
    assert checks["No overnight hold after a losing day"].status == WARN
    assert checks["A free slot"].status == FAIL
    assert verdict(checks.values()) == "stop"
    # Yesterday's win and a different stock today do not count against AMD.
    amd = by_name(run_checks("AMD", price_levels(breakout_bars()), closed, TODAY))
    assert amd["Not a stock you lost on today"].status == PASS
    assert verdict(amd.values()) == "caution"
    assert day_pnl(closed, TODAY) == -50
    hk = by_name(run_checks("0700.HK", None, [], TODAY))
    assert hk["Not a Hong Kong stock"].status == FAIL and hk["Price data"].status == FAIL


def test_planned_price_overrides_latest():
    levels = price_levels(breakout_bars())
    checks = by_name(run_checks("NVDA", levels, [], TODAY, entry_price=levels["high_n"] * 0.99))
    assert checks["Breaks its 20-day high"].status == FAIL


def test_sizing():
    sizing = size_trade(100_000, 4, 5, 123.0)
    assert sizing["shares"] == 32 and sizing["stop"] == pytest.approx(116.85)
    assert sizing["risk"] == pytest.approx(32 * (123 - 116.85))
    assert size_trade(100_000, 4, 5, 0)["shares"] == 0


def test_position_status_actions():
    levels = price_levels(breakout_bars())
    assert position_status({"Symbol": "x", "Shares": 10, "Entry price": 100, "Stop": 95}, levels)["Action"].startswith("Hold")
    assert "stop" in position_status({"Symbol": "x", "Shares": 10, "Entry price": 120, "Stop": 110}, levels)["Action"]
    weak = price_levels(make_bars(np.linspace(100, 80, 80), index=pd.bdate_range(end=TODAY, periods=80)))
    row = position_status({"Symbol": "x", "Shares": 10, "Entry price": 90, "Stop": None}, weak)
    assert row["Action"].startswith("Sell: below the 20-day average") and row["P&L"] < 0
    assert position_status({"Symbol": "x", "Shares": 1, "Entry price": 5}, None)["Action"] == "No price data"


def test_local_store_round_trip(tmp_path):
    store = make_store({}, tmp_path / "data.json")
    assert isinstance(store, LocalStore) and not store.durable
    data, _ = store.load()
    assert data["positions"] == [] and data["settings"]["slot_pct"] == 4.0
    data["positions"].append({"Symbol": "NVDA", "Shares": 5})
    store.save(data)
    assert store.load()[0]["positions"] == [{"Symbol": "NVDA", "Shares": 5}]


class FakeGitHub:
    """Contents API for one file, with SHA checks like GitHub's."""

    def __init__(self):
        self.content, self.sha, self.requests = None, None, []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        assert request.get_header("Authorization") == "Bearer tok"
        if request.get_method() == "GET":
            if self.content is None:
                raise HTTPError(request.full_url, 404, "Not Found", {}, None)
            body = {"content": base64.b64encode(self.content).decode(), "sha": self.sha}
        else:
            sent = json.loads(request.data)
            if sent.get("sha") != self.sha:
                raise HTTPError(request.full_url, 409 if self.sha else 422, "Conflict", {}, None)
            self.content = base64.b64decode(sent["content"])
            self.sha = f"sha{len(self.requests)}"
            body = {"content": {"sha": self.sha}}
        return io.BytesIO(json.dumps(body).encode())


def test_github_store(monkeypatch):
    fake = FakeGitHub()
    monkeypatch.setattr(trade_store, "urlopen", fake)
    store = make_store({"github_token": "tok", "repo": "me/data"}, "unused.json")
    assert isinstance(store, GitHubStore) and store.durable
    assert fake.requests == []
    data, version = store.load()
    assert version is None and data["closed"] == []
    data["closed"].append({"Symbol": "AMD", "P&L": -12.5})
    version = store.save(data, version)
    assert fake.requests[-1].full_url == "https://api.github.com/repos/me/data/contents/trading.json"
    loaded, loaded_version = store.load()
    assert loaded["closed"] == [{"Symbol": "AMD", "P&L": -12.5}] and loaded_version == version
    # A save from a stale copy (another device saved in between) is refused, not overwritten.
    store.save(loaded, loaded_version)
    with pytest.raises(StoreConflict):
        store.save(loaded, loaded_version)


def test_github_store_errors(monkeypatch):
    def unauthorized(request, timeout=None):
        raise HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(trade_store, "urlopen", unauthorized)
    store = GitHubStore("tok", "me/data")
    with pytest.raises(StoreError, match="expired"):
        store.load()

    def offline(request, timeout=None):
        raise OSError("no network")

    monkeypatch.setattr(trade_store, "urlopen", offline)
    with pytest.raises(StoreError, match="reach GitHub"):
        store.save({}, None)


def test_pretrade_tab(monkeypatch, tmp_path):
    from tests.test_app_smoke import fake_download, FakeTicker, offline_urlopen, open_tab, start_app
    import urllib.request
    import yfinance as yf

    monkeypatch.setattr(yf, "download", fake_download)
    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    monkeypatch.setattr(urllib.request, "urlopen", offline_urlopen)
    store_file = tmp_path / "trading.json"
    monkeypatch.setenv("TRADE_STORE_PATH", str(store_file))
    store_file.write_text(json.dumps({"settings": {"account": 50_000, "slot_pct": 4, "stop_pct": 5,
                                                   "max_positions": 15},
                                      "positions": [{"Symbol": "AAA", "Shares": 10, "Entry price": 100.0,
                                                     "Entry date": "2026-11-01", "Stop": 95.0, "Note": ""}],
                                      "closed": []}))

    app = open_tab(start_app(), "Pre-trade Check")
    assert any("only kept until the app restarts" in element.value for element in app.warning)
    assert next(box for box in app.number_input if box.key == "pretrade_account").value == 50_000
    assert any(header.value == "Open positions (1)" for header in app.subheader)

    next(box for box in app.text_input if box.key == "pretrade_symbol").input("nvda")
    app = open_tab(app, "Pre-trade Check")
    text = " ".join(element.value for element in app.markdown)
    assert "Breaks its 20-day high" in text and "Above its 50-day average" in text
    assert "Not a Hong Kong stock" in text
    assert any("NVDA" in element.value for element in [*app.success, *app.warning, *app.error])
    labels = {metric.label: metric.value for metric in app.metric}
    assert "Shares" in labels and labels["Shares"] != "0"

    next(button for button in app.button if button.key == "pretrade_buy_btn").click()
    app = open_tab(app, "Pre-trade Check")
    saved = json.loads(store_file.read_text())
    assert [row["Symbol"] for row in saved["positions"]] == ["AAA", "NVDA"]
    assert saved["positions"][1]["Stop"] < saved["positions"][1]["Entry price"]

    # Changing a setting saves it straight away.
    next(box for box in app.number_input if box.key == "pretrade_slot_pct").set_value(5.0)
    app = open_tab(app, "Pre-trade Check")
    assert json.loads(store_file.read_text())["settings"]["slot_pct"] == 5.0

    # Closing AAA at a loss logs it, and the check then blocks AAA for the rest of the day.
    next(box for box in app.number_input if box.key == "pretrade_close_price_0_draft").set_value(90.0)
    next(button for button in app.button if button.key == "pretrade_close_btn").click()
    app = open_tab(app, "Pre-trade Check")
    saved = json.loads(store_file.read_text())
    assert [row["Symbol"] for row in saved["positions"]] == ["NVDA"]
    assert saved["closed"][0]["P&L"] == pytest.approx(-100.0)
    next(box for box in app.text_input if box.key == "pretrade_symbol").input("AAA")
    app = open_tab(app, "Pre-trade Check")
    text = " ".join(element.value for element in app.markdown)
    assert "You lost $100 on AAA today" in text
    assert any("don't buy" in element.value for element in app.error)
