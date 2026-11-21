"""Check the fundamental P/E's growth rules against real data for many stocks (run in CI, which has the network).

A. Fade test (the 'growth half-life'): from SEC annual revenue, at each year with 3 years behind and 3 ahead,
   predict the next 3 years' revenue growth from the past 3 years' with each fade rule, and compare with what
   happened. Rules: no fade, gross-margin half-life only, gross margin or steadiness (the longer), a fixed 1-year
   half-life, and an instant fade to the terminal rate.
B. Cross-section today: the model's fundamental P/E (rules off / on) against the market's forward P/E.
"""

import json
import math
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from analysis.valuation_model import fundamental_pe, growth_half_life, model_params, steady_half_life  # noqa: E402

TICKERS = ["COST", "WMT", "KO", "PEP", "PG", "JNJ", "MCD", "HD", "NKE", "SBUX", "AAPL", "MSFT", "GOOGL", "AMZN",
           "META", "NVDA", "AVGO", "ADBE", "CRM", "ORCL", "INTC", "TXN", "AMAT", "KLAC", "LRCX", "TSLA", "XOM", "CVX",
           "UNH", "LLY", "MRK", "ABBV", "CAT", "DE", "HON", "NFLX", "ISRG", "TGT", "LOW", "QCOM"]
UA = {"User-Agent": "backtester validation research kelvindev2017@users.noreply.github.com"}
REVENUE = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
           "RevenueFromContractWithCustomerIncludingAssessedTax"]
COST = ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"]
TERMINAL = 3.0


def get(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as response:
                return json.loads(response.read())
        except Exception:
            time.sleep(1 + attempt)
    return None


def annual(facts, tags):
    """Fiscal-year values (10-K, ~12-month periods) by period end, the first tag that has each year."""

    out = pd.Series(dtype=float)
    for tag in tags:
        units = facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {}).get("USD", [])
        rows = [u for u in units if u.get("form") == "10-K" and u.get("start") and u.get("end")]
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        frame["days"] = (pd.to_datetime(frame["end"]) - pd.to_datetime(frame["start"])).dt.days
        frame = frame[(frame["days"] > 340) & (frame["days"] < 390)].sort_values("filed")
        series = frame.groupby("end")["val"].last()
        series.index = pd.to_datetime(series.index)
        out = out.combine_first(series.astype(float))
    return out.sort_index()


def yearly(series):
    """One value per fiscal year (the last period ending in each calendar year)."""

    series = series.dropna()
    return series.groupby(series.index.year).last()


def fade_cagr(growth, half_life, years=3, terminal=TERMINAL):
    """Average yearly growth over the next ``years`` years if the growth above terminal halves every half_life."""

    start, end = math.log1p(growth / 100), math.log1p(terminal / 100)
    if half_life == 0:
        logs = [end] * years
    elif not np.isfinite(half_life):
        logs = [start] * years
    else:
        logs = [end + (start - end) * 0.5 ** (k / half_life) for k in range(1, years + 1)]
    return (math.exp(sum(logs) / years) - 1) * 100


def main():
    params = model_params({})
    ciks = get("https://www.sec.gov/files/company_tickers.json")
    cik = {row["ticker"]: int(row["cik_str"]) for row in ciks.values()}
    fade_rows, now_rows = [], []
    for ticker in TICKERS:
        facts = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.get(ticker, 0):010d}.json") if ticker in cik else None
        time.sleep(0.15)
        if not facts:
            print(f"{ticker}: no SEC facts")
            continue
        revenue = yearly(annual(facts, REVENUE))
        gross = yearly(annual(facts, ["GrossProfit"]))
        cost = yearly(annual(facts, COST))
        gross = gross.combine_first(revenue - cost)
        revenue = revenue[revenue > 0]
        if len(revenue) < 7:
            print(f"{ticker}: only {len(revenue)} years of revenue")
            continue
        yoy = (revenue / revenue.shift(1) - 1) * 100
        years = list(revenue.index)
        for position in range(4, len(years) - 3):
            year = years[position]
            if years[position] - years[position - 3] != 3 or years[position + 3] - year != 3:
                continue
            past = ((revenue[year] / revenue[years[position - 3]]) ** (1 / 3) - 1) * 100
            spread = yoy.loc[years[position - 4]:year].std()
            margin = gross.get(year, np.nan) / revenue[year] * 100 if year in gross else np.nan
            actual = ((revenue[years[position + 3]] / revenue[year]) ** (1 / 3) - 1) * 100
            margin_h = growth_half_life(margin, params)
            steady_h = steady_half_life(spread, past, params)
            fade_rows.append({
                "ticker": ticker, "year": year, "past": past, "spread": spread, "gross_margin": margin,
                "actual": actual, "no_fade": fade_cagr(past, np.inf), "margin_rule": fade_cagr(past, margin_h),
                "margin_or_steady": fade_cagr(past, max(margin_h, steady_h) if np.isfinite(steady_h) else margin_h),
                "fixed_1y": fade_cagr(past, 1.0), "instant": fade_cagr(past, 0)})
        last = years[-1]
        if len(years) >= 5:
            past = ((revenue[last] / revenue[years[-4]]) ** (1 / 3) - 1) * 100
            now_rows.append({"ticker": ticker, "year": last, "cagr3": past, "yoy": yoy[last],
                             "spread": yoy.iloc[-5:].std(),
                             "gross_margin": gross.get(last, np.nan) / revenue[last] * 100 if last in gross else np.nan})
    fade = pd.DataFrame(fade_rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(f"\n=== A. Fade test: {len(fade)} stock-years, {fade['ticker'].nunique()} stocks ===")
    rules = ["no_fade", "margin_rule", "margin_or_steady", "fixed_1y", "instant"]
    summary = pd.DataFrame({rule: {
        "median abs error (pts)": (fade[rule] - fade["actual"]).abs().median(),
        "mean abs error (pts)": (fade[rule] - fade["actual"]).abs().mean(),
        "bias (pts, + = too high)": (fade[rule] - fade["actual"]).mean()} for rule in rules}).T
    print(summary.round(2).to_string())
    fade["excess_past"] = fade["past"] - TERMINAL
    fade["excess_next"] = fade["actual"] - TERMINAL
    slope = np.polyfit(fade["excess_past"].clip(-30, 100), fade["excess_next"].clip(-30, 100), 1)[0]
    print(f"\nPersistence: next-3y excess growth = {slope:.2f} x past-3y excess growth "
          f"(implied half-life over 3 years ~ {(-3 * math.log(2) / math.log(max(slope, 1e-3))) if 0 < slope < 1 else float('nan'):.1f} y "
          f"for a one-step decay, roughly)")
    for name, mask in [("steady (spread <= 3)", fade["spread"] <= 3), ("choppy (spread > 10)", fade["spread"] > 10),
                       ("high gross margin (>= 60%)", fade["gross_margin"] >= 60),
                       ("low gross margin (<= 25%)", fade["gross_margin"] <= 25)]:
        part = fade[mask]
        if len(part) >= 10:
            s = np.polyfit(part["excess_past"].clip(-30, 100), part["excess_next"].clip(-30, 100), 1)[0]
            errors = {rule: (part[rule] - part["actual"]).abs().median() for rule in rules}
            print(f"{name:28s} n={len(part):3d} persistence={s:.2f}  median abs error: "
                  + ", ".join(f"{rule} {value:.1f}" for rule, value in errors.items()))
    print("\nPer stock, median abs error of each rule (pts):")
    per = fade.groupby("ticker").apply(lambda part: pd.Series({rule: (part[rule] - part["actual"]).abs().median()
                                                               for rule in rules}), include_groups=False)
    print(per.round(1).to_string())

    # B. Cross-section today.
    import yfinance as yf
    spy = yf.Ticker("SPY").history(period="4y", interval="1d")["Close"]
    tnx = yf.Ticker("^TNX").history(period="5d")["Close"]
    ten_year = float(tnx.iloc[-1]) if len(tnx) else 4.0
    rows = []
    for row in now_rows:
        ticker = row["ticker"]
        try:
            info = yf.Ticker(ticker).info
            prices = yf.Ticker(ticker).history(period="4y", interval="1d")["Close"]
        except Exception as error:
            print(f"{ticker}: yahoo {error}")
            continue
        both = pd.concat([prices.tz_localize(None), spy.tz_localize(None)], axis=1, join="inner").dropna()
        both = both[both.index >= both.index.max() - pd.DateOffset(years=3)]
        weekly = np.log(both.resample("W-FRI").last()).diff().dropna()
        beta = weekly.iloc[:, 0].cov(weekly.iloc[:, 1]) / weekly.iloc[:, 1].var()
        growth_off = row["cagr3"]
        growth_on = 0.5 * row["cagr3"] + 0.5 * row["yoy"]
        margin_h = growth_half_life(row["gross_margin"], params)
        steady_h = steady_half_life(row["spread"], growth_on, params)
        half_on = max(margin_h, steady_h) if np.isfinite(steady_h) else margin_h
        erp_on = float(np.clip(4.5 * np.clip(beta, 0.5, 1.8), 2.5, 8.0))
        off = fundamental_pe(growth_off, None, ten_year, params, half_life=margin_h)
        on = fundamental_pe(growth_on, None, ten_year, {**params, "equity_premium": erp_on}, half_life=half_on)
        rows.append({"ticker": ticker, "market fwd P/E": info.get("forwardPE"), "rules off": off, "rules on": on,
                     "growth on": growth_on, "half-life on": half_on, "beta": beta, "GM": row["gross_margin"],
                     "spread": row["spread"]})
        time.sleep(0.3)
    now = pd.DataFrame(rows).dropna(subset=["market fwd P/E"])
    now = now[now["market fwd P/E"] > 0]
    print(f"\n=== B. Cross-section today ({len(now)} stocks, 10-year yield {ten_year:.2f}%) ===")
    print(now.round(2).sort_values("market fwd P/E").to_string(index=False))
    for column in ["rules off", "rules on"]:
        ratio = np.log(now["market fwd P/E"] / now[column])
        corr = np.corrcoef(np.log(now["market fwd P/E"]), np.log(now[column]))[0, 1]
        print(f"{column:9s}: corr(log P/E) {corr:.2f}, median market/model {math.exp(ratio.median()):.2f}x, "
              f"spread of log ratio {ratio.std():.2f}")
    fade.to_csv("validation_fade.csv", index=False)
    now.to_csv("validation_now.csv", index=False)


if __name__ == "__main__":
    main()
