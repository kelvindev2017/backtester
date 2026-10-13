from datetime import date

from analysis.settings_store import COOKIE_PREFIX, MAX_CHUNKS, cookie_script, pack, settings_snapshot, unpack


def cookies_for(chunks):
    cookies = {f"{COOKIE_PREFIX}{index}": chunk for index, chunk in enumerate(chunks)}
    cookies[f"{COOKIE_PREFIX}n"] = str(len(chunks))
    return cookies


def test_round_trip_keeps_types():
    state = {
        "screener_universe": "S&P 500",
        "screener_max_from_high": 25.0,
        "screener_use_high": True,
        "industry_indexes": ["^SOX", "XLK"],
        "start_date": date(2020, 1, 2),
        "date_range": (date(2020, 1, 2), date(2021, 1, 4)),
        "screen_rows": {0: {"group": "Mkt cap"}, 3: {"group": "P/E"}},
    }
    assert unpack(cookies_for(pack(settings_snapshot(state)))) == state


def test_snapshot_skips_private_keys_and_data():
    import pandas as pd

    state = {"_settings_saved": ["x"], "screener_data": pd.DataFrame({"a": [1]}), "main_tab": "Backtest", "ok": 1}
    assert settings_snapshot(state, skip=lambda key: key == "main_tab") == {"ok": 1}


def test_large_settings_span_several_cookies():
    tickers = " ".join(f"T{index:05d}{index * 7919 % 100003}" for index in range(3000))
    chunks = pack({"screener_custom": tickers})
    assert 1 < len(chunks) <= MAX_CHUNKS
    assert unpack(cookies_for(chunks)) == {"screener_custom": tickers}


def test_missing_or_broken_cookies_restore_nothing():
    assert unpack({}) == {}
    assert unpack({f"{COOKIE_PREFIX}n": "2", f"{COOKIE_PREFIX}0": "abc"}) == {}
    assert unpack({f"{COOKIE_PREFIX}n": "1", f"{COOKIE_PREFIX}0": "not base64!"}) == {}


def test_cookie_script_clears_leftover_chunks():
    script = cookie_script(["aaa", "bbb"])
    assert f"{COOKIE_PREFIX}1=bbb" in script
    assert f"{COOKIE_PREFIX}n=2" in script
    assert f'{COOKIE_PREFIX}2=; path=/; max-age=0' in script
