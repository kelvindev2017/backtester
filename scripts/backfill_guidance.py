"""One-off: read the guidance history (8-K releases since 2012) for the daily log's tickers into
``guidance_numbers.csv``, and the tickers done into ``guidance_read.csv`` (the app reads a ticker's guidance from
the log when it is listed there). After this, scripts/log_guidance.py only adds each day's new releases.

Usage: python scripts/backfill_guidance.py OUT_DIR [--tickers A,B] [--since 2012-01-01]
"""

import argparse
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from analysis.guidance import GUIDANCE_COLUMNS, company_guidance  # noqa: E402
from log_guidance import SEC_AGENT, sec_ciks, universe  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--tickers", default="")
    parser.add_argument("--since", default="2012-01-01")
    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or universe()
    path = os.path.join(args.out_dir, "guidance_numbers.csv")
    read_path = os.path.join(args.out_dir, "guidance_read.csv")
    done = set(pd.read_csv(read_path)["ticker"]) if os.path.exists(read_path) else set()
    session = requests.Session()
    ciks = sec_ciks(session)
    for ticker in [t for t in tickers if t in ciks and t not in done]:
        started = time.time()
        rows, read = company_guidance(ticker, ciks[ticker], session, SEC_AGENT, since=args.since)
        pd.DataFrame(rows, columns=GUIDANCE_COLUMNS).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        pd.DataFrame({"ticker": [ticker], "read": [pd.Timestamp.today().date().isoformat()]}).to_csv(
            read_path, mode="a", header=not os.path.exists(read_path), index=False)
        print(f"{ticker}: {len(read)} filings, {len(rows)} figures, {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
