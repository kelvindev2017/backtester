"""Daily log of company guidance and analysts' estimates for the S&P 500 members, ANT and SPCX.

Run each weekday evening by .github/workflows/log_guidance.yml, which keeps the files on the ``data-log`` branch
so they survive app updates and reboots:

- ``guidance_numbers.csv``: guided figures (revenue, revenue growth, EPS for the next quarter or the fiscal year)
  read by analysis/guidance.py from the 8-K press releases the companies filed. Only the days not yet done are
  read: the SEC's daily filing list says which companies filed an 8-K that day, and only those filings are opened.
  ``guidance_days.csv`` records the days done. The history back to 2012 was read once by
  scripts/backfill_guidance.py.
- ``estimates/YYYY-MM-DD.csv.gz``: Yahoo's analyst estimates as they stand that day (Yahoo keeps no history).

Usage: python scripts/log_guidance.py OUT_DIR [--tickers A,B] [--skip-yahoo]
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.guidance import GUIDANCE_COLUMNS, day_filings, filing_guidance, sec_get  # noqa: E402

WATCHED = ["ANT", "SPCX"]
SEC_AGENT = os.getenv("SEC_USER_AGENT") or "StreamlitBacktester/1.0 kelvin@example.com"  # same default as App.py
MAX_DAYS = 10  # days caught up in one run after missed runs


def universe():
    from analysis.universe import load_us_index_members, normalize_ticker

    members = load_us_index_members("S&P 500")["Ticker"].map(normalize_ticker).tolist()
    return sorted(set(members) | set(WATCHED))


def sec_ciks(session):
    lookup = sec_get("https://www.sec.gov/files/company_tickers.json", session, SEC_AGENT)
    return {} if lookup is None else {entry["ticker"].upper(): int(entry["cik_str"]) for entry in lookup.json().values()}


def new_guidance(tickers, days, session):
    """Guidance rows from the 8-K releases the tickers filed on ``days``; returns (rows, days whose list was read)."""

    ciks = sec_ciks(session)
    by_cik = {ciks[t]: t for t in tickers if t in ciks}
    rows, done, year_ends = [], [], {}
    for day in days:
        filings = day_filings(day, session, SEC_AGENT)
        if filings is None:
            continue  # not published yet (or no filings that day): tried again on the next run
        done.append(day)
        for filing in filings[filings["cik"].isin(by_cik)].itertuples():
            if filing.cik not in year_ends:
                submissions = sec_get(f"https://data.sec.gov/submissions/CIK{filing.cik:010d}.json", session, SEC_AGENT)
                year_ends[filing.cik] = (submissions.json().get("fiscalYearEnd") if submissions else None) or "1231"
            found = filing_guidance(filing.cik, filing.accession, filing.filed, year_ends[filing.cik], session,
                                    by_cik[filing.cik], SEC_AGENT, items_check=True)
            rows += found
            print(f"{day}: {by_cik[filing.cik]} {filing.accession}: {len(found)} figures")
    return rows, done


def yahoo_rows(ticker, today):
    import yfinance as yf

    stock, rows = yf.Ticker(ticker), []
    for table in ["earnings_estimate", "revenue_estimate", "eps_trend", "eps_revisions"]:
        try:
            frame = getattr(stock, table)
        except Exception as error:
            print(f"{ticker} {table}: {type(error).__name__}")
            continue
        if frame is None or getattr(frame, "empty", True):
            continue
        for period, values in frame.iterrows():
            for field, value in values.items():
                if pd.notna(value):
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        value = str(value)  # e.g. the currency of the estimates
                    rows.append({"date": today, "ticker": ticker, "table": table, "period": str(period),
                                 "field": str(field), "value": value})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--tickers", default="")
    parser.add_argument("--skip-yahoo", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or universe()
    today = datetime.now(timezone.utc).date()

    # Guidance: the days since the last one done (today included once its filing list is out).
    numbers_path = os.path.join(args.out_dir, "guidance_numbers.csv")
    days_path = os.path.join(args.out_dir, "guidance_days.csv")
    last = pd.Timestamp(pd.read_csv(days_path)["day"].max()).date() if os.path.exists(days_path) else today - timedelta(days=3)
    days = [last + timedelta(days=n) for n in range(1, (today - last).days + 1) if (last + timedelta(days=n)).weekday() < 5]
    rows, done = new_guidance(tickers, [d.isoformat() for d in days[-MAX_DAYS:]], requests.Session())
    if rows:
        seen = set(pd.read_csv(numbers_path, usecols=["accession"])["accession"]) if os.path.exists(numbers_path) else set()
        frame = pd.DataFrame(rows, columns=GUIDANCE_COLUMNS)
        frame = frame[~frame["accession"].isin(seen)]
        frame.to_csv(numbers_path, mode="a", header=not os.path.exists(numbers_path), index=False)
    if done:
        pd.DataFrame({"day": done}).to_csv(days_path, mode="a", header=not os.path.exists(days_path), index=False)
    print(f"guidance: {len(done)} days read, {len(rows)} figures")

    # Analysts' estimates as they stand today, one compressed file a day.
    estimates_path = os.path.join(args.out_dir, "estimates", f"{today.isoformat()}.csv.gz")
    if not args.skip_yahoo and not os.path.exists(estimates_path):
        estimates = []
        for ticker in tickers:
            estimates += yahoo_rows(ticker, today.isoformat())
            time.sleep(0.3)
        if not estimates:
            sys.exit("Yahoo returned no estimates at all")
        os.makedirs(os.path.dirname(estimates_path), exist_ok=True)
        pd.DataFrame(estimates).to_csv(estimates_path, index=False, compression="gzip")
        print(f"estimates: {len(estimates)} values for {len({r['ticker'] for r in estimates})} tickers")


if __name__ == "__main__":
    main()
