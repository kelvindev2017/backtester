"""Daily log of company guidance and analysts' estimates (S&P 500 members, the study and supply chain stocks and a
watch list), so they build a history that can be tested later.

Two files, appended to on each run (run by .github/workflows/log_guidance.yml, which keeps them on the
``data-log`` branch so they survive app updates and reboots):

- ``estimates/YYYY-MM-DD.csv.gz``: Yahoo's analyst estimates as they stand that day, one row per ticker, table, period and field
  (EPS and revenue estimates, EPS trend 7/30/60/90 days ago, revisions up/down). Yahoo keeps no history of these,
  so this file is the history.
- ``guidance.csv``: company guidance from SEC 8-K filings (Item 2.02 results and 7.01 updates): the sentences of the
  press release exhibit that state an outlook with a number, one row per sentence, by filing. Each filing is logged
  once.

Usage: python scripts/log_guidance.py OUT_DIR [--tickers A,B] [--days 120]
"""

import argparse
import html
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

STUDY = ["COST", "WMT", "KO", "PEP", "PG", "JNJ", "MCD", "HD", "NKE", "SBUX", "AAPL", "MSFT", "GOOGL", "AMZN", "META",
         "NVDA", "AVGO", "ADBE", "CRM", "ORCL", "INTC", "TXN", "AMAT", "KLAC", "LRCX", "TSLA", "UNH", "LLY", "MRK",
         "ABBV", "CAT", "DE", "HON", "NFLX", "ISRG", "TGT", "LOW", "QCOM", "CVX", "XOM"]
WATCHED = ["BSX", "PLTR", "AVAV", "ASMVY", "BESIY", "TSM", "SKHY", "ANT", "SPCX"]
SEC_AGENT = os.getenv("SEC_USER_AGENT") or "StreamlitBacktester/1.0 kelvin@example.com"  # same default as App.py
GUIDANCE_WORDS = re.compile(r"\b(outlook|guidance|expects?|expected|anticipates?|forecasts?|now estimates?|"
                            r"estimates? (?:net |total )?(?:sales|revenue|earnings|eps)|reaffirm\w*|"
                            r"(?:raises?|raised|lowers?|lowered|narrows?|narrowed|updates?|updated) (?:its |the )?"
                            r"(?:full[- ]year |fiscal )?(?:outlook|guidance|forecast))\b", re.I)
NUMBER = re.compile(r"(\$\s?\d[\d,.]*|\d[\d,.]*\s?(?:%|percent|billion|million))", re.I)


def universe():
    tickers = list(STUDY) + list(WATCHED)
    try:
        from analysis.universe import load_us_index_members, normalize_ticker
        members = load_us_index_members("S&P 500")["Ticker"].map(normalize_ticker).tolist()
        print(f"S&P 500: {len(members)} members")
        tickers += members
    except Exception as error:
        print(f"S&P 500 list not loaded: {type(error).__name__}: {error}")
    try:
        from analysis.supply_chain import COMPANIES
        tickers += [t for t in COMPANIES if isinstance(t, str)]
    except Exception as error:
        print(f"supply chain list not loaded: {error}")
    return sorted({t.strip().upper() for t in tickers if t and re.fullmatch(r"[A-Z0-9.\-]{1,8}", t.strip().upper())})


def yahoo_rows(ticker, today):
    import yfinance as yf

    stock = yf.Ticker(ticker)
    rows = []
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


def sec_get(url, session):
    for attempt in range(4):
        response = session.get(url, headers={"User-Agent": SEC_AGENT, "Accept-Encoding": "gzip, deflate"}, timeout=30)
        time.sleep(0.15)  # SEC asks for at most 10 requests a second
        if response.status_code == 200:
            return response
        if response.status_code in (403, 429, 503):
            time.sleep(2 * (attempt + 1))
            continue
        return None
    return None


def exhibit_names(index_page):
    """Press release exhibits (EX-99.x) listed in a filing's index page; companies name the files freely."""

    names = []
    for row in re.findall(r"(?is)<tr.*?</tr>", index_page):
        cells = re.findall(r"(?is)<td[^>]*>(.*?)</td>", row)
        if len(cells) >= 4 and re.fullmatch(r"\s*EX-99(\.\d+)?\s*", re.sub(r"<[^>]+>", "", cells[3])):
            link = re.search(r'href="([^"]+)"', cells[2])
            if link and link.group(1).lower().endswith((".htm", ".html", ".txt")):
                names.append(link.group(1).rsplit("/", 1)[-1])
    return names


def text_of(document):
    document = re.sub(r"(?is)<(script|style).*?</\1>", " ", document)
    document = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>|</li>", "\n", document)
    document = html.unescape(re.sub(r"<[^>]+>", " ", document))
    return re.sub(r"[ \t\xa0]+", " ", document)


def guidance_sentences(text):
    """Sentences stating an outlook with a number, plus the numeric lines of an "Outlook" or "Guidance" section
    (many releases give guidance as a table), prefixed with that section's heading."""

    sentences = re.split(r"(?<=[.;])\s+(?=[A-Z(•])|\n+", text)
    kept, heading, left = [], "", 0
    for sentence in sentences:
        sentence = re.sub(r"\s+", " ", sentence).strip(" •-")
        if not sentence:
            continue
        if len(sentence) <= 80 and re.search(r"\b(outlook|guidance)\b", sentence, re.I) \
                and not re.search(r"\b(reports?|results|raises|lowers)\b", sentence, re.I):
            heading, left = sentence, 25
            continue
        if left and re.match(r"(about |conference call|forward-looking|notes?\b|non-gaap financial measures)", sentence, re.I):
            left = 0
        if re.search(r"forward-looking statements|safe harbor|undue reliance|not able to reconcile", sentence, re.I):
            continue
        if left:
            left -= 1
            if 8 <= len(sentence) <= 700 and NUMBER.search(sentence):
                kept.append(f"[{heading}] {sentence}")
                continue
        if 40 <= len(sentence) <= 700 and GUIDANCE_WORDS.search(sentence) and NUMBER.search(sentence):
            kept.append(sentence)
    return kept


def sec_rows(tickers, since, logged, session):
    lookup = sec_get("https://www.sec.gov/files/company_tickers.json", session)
    if lookup is None:
        print("SEC ticker list not reachable")
        return []
    ciks = {entry["ticker"].upper(): int(entry["cik_str"]) for entry in lookup.json().values()}
    rows = []
    for ticker in tickers:
        cik = ciks.get(ticker)
        if cik is None:
            continue  # foreign filers on 6-K and ADRs without their own filings are not covered
        submissions = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", session)
        if submissions is None:
            continue
        recent = submissions.json().get("filings", {}).get("recent", {})
        for form, filed, accession, items in zip(recent.get("form", []), recent.get("filingDate", []),
                                                 recent.get("accessionNumber", []), recent.get("items", [])):
            if form != "8-K" or filed < since or accession in logged:
                continue
            if "2.02" not in (items or "") and "7.01" not in (items or ""):
                continue
            folder = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}"
            index = sec_get(f"{folder}/{accession}-index.html", session)
            if index is None:
                continue
            exhibits = exhibit_names(index.text)
            found, seen = 0, set()
            for name in exhibits:
                document = sec_get(f"{folder}/{name}", session)
                if document is None:
                    continue
                for sentence in guidance_sentences(text_of(document.text)):
                    if sentence in seen:
                        continue
                    seen.add(sentence)
                    rows.append({"filed": filed, "ticker": ticker, "accession": accession, "items": items,
                                 "url": f"{folder}/{name}", "sentence": sentence})
                    found += 1
            if not found:  # logged anyway, so a release without guidance is not fetched again
                rows.append({"filed": filed, "ticker": ticker, "accession": accession, "items": items,
                             "url": folder, "sentence": ""})
        print(f"{ticker}: SEC done")
    return rows


def append(path, rows, columns):
    if not rows:
        return 0
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    return len(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--tickers", default="")
    parser.add_argument("--days", type=int, default=120, help="8-K look-back on the first run")
    parser.add_argument("--skip-yahoo", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or universe()
    today = datetime.now(timezone.utc).date().isoformat()
    # One compressed file a day (about 60,000 values for 520 stocks), so the log stays small in git.
    estimates_path = os.path.join(args.out_dir, "estimates", f"{today}.csv.gz")
    guidance_path = os.path.join(args.out_dir, "guidance.csv")

    yahoo = []
    if not args.skip_yahoo and not os.path.exists(estimates_path):
        for ticker in tickers:
            yahoo += yahoo_rows(ticker, today)
            time.sleep(0.3)
    added_estimates = 0
    if yahoo:
        os.makedirs(os.path.dirname(estimates_path), exist_ok=True)
        frame = pd.DataFrame(yahoo, columns=["date", "ticker", "table", "period", "field", "value"])
        frame.to_csv(estimates_path, index=False, compression="gzip")
        added_estimates = len(frame)

    logged = set()
    if os.path.exists(guidance_path):
        logged = set(pd.read_csv(guidance_path, usecols=["accession"])["accession"])
    since = (date.today() - timedelta(days=args.days)).isoformat()
    session = requests.Session()
    guidance = sec_rows(tickers, since, logged, session)
    added_guidance = append(guidance_path, guidance, ["filed", "ticker", "accession", "items", "url", "sentence"])
    with_numbers = sum(1 for row in guidance if row["sentence"])
    print(f"{len(tickers)} tickers; {added_estimates} estimate rows "
          f"({len({row['ticker'] for row in yahoo})} tickers with Yahoo estimates); {added_guidance} guidance rows "
          f"({with_numbers} outlook sentences from {len({row['accession'] for row in guidance})} new 8-K filings)")
    if not args.skip_yahoo and not yahoo and not os.path.exists(estimates_path):
        sys.exit("Yahoo returned no estimates at all")


if __name__ == "__main__":
    main()
