"""US stock universes for the screener.

Index members are read from Wikipedia at run time (cached by the app), with
a public CSV as a fallback for the S&P 500.
"""

import io
import re
import urllib.request
from urllib.request import Request

import pandas as pd

USER_AGENT = "Mozilla/5.0 (compatible; KelvinBacktester/1.0)"

MARKETS = {
    "US": {
        "indexes": {
            "S&P 500": "^GSPC",
            "Nasdaq Composite": "^IXIC",
            "Nasdaq-100": "^NDX",
            "Dow Jones": "^DJI",
            "Russell 2000": "^RUT",
        },
        "universes": ["S&P 500", "Nasdaq-100", "Dow Jones 30", "Custom list"],
    },
}

DOW_JONES_30 = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]


def normalize_ticker(raw):
    """Yahoo symbol for a user-typed US ticker: ``BRK.B`` -> ``BRK-B``."""

    ticker = str(raw).strip().upper()
    if not ticker:
        return ""
    if ticker.startswith("^") or "=" in ticker:
        return ticker
    return ticker.replace(".", "-")


def normalize_symbol(raw):
    """Yahoo symbol for a sidebar ticker: ``brk.b`` -> ``BRK-B``.

    Exchange suffixes (``SHOP.TO``, ``VOD.L``) are kept; a one-letter suffix
    is a US share class, which Yahoo writes with a dash.
    """

    ticker = str(raw).strip().upper()
    if re.fullmatch(r"[A-Z]+\.[A-Z]", ticker):
        return ticker.replace(".", "-")
    return ticker


def parse_ticker_list(text):
    """Unique Yahoo symbols from comma / space / newline separated text."""

    tickers = [normalize_ticker(part) for part in re.split(r"[\s,;]+", text or "")]
    return list(dict.fromkeys(ticker for ticker in tickers if ticker))


def _read_html_tables(url):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        html = response.read().decode("utf-8", errors="replace")
    return pd.read_html(io.StringIO(html))


def members_from_tables(tables, min_rows):
    """Pick the constituents table (a Symbol/Ticker column with enough rows) and standardise it."""

    for table in tables:
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = [" ".join(str(part) for part in column).strip() for column in table.columns]
        # Wikipedia headers can carry footnote markers, e.g. "ICB Industry[1]".
        columns = {re.sub(r"\[[^\]]*\]", "", str(column)).strip(): column for column in table.columns}
        symbol_column = next((columns[name] for name in ("Symbol", "Ticker", "Ticker symbol") if name in columns), None)
        if symbol_column is None or len(table) < min_rows:
            continue
        name_column = next((columns[name] for name in ("Security", "Company", "Company Name", "Name") if name in columns), None)
        sector_column = next((columns[name] for name in ("GICS Sector", "Sector", "ICB Industry") if name in columns), None)
        industry_column = next((columns[name] for name in ("GICS Sub-Industry", "Sub-Industry", "ICB Subsector", "Industry") if name in columns), None)
        frame = pd.DataFrame({
            "Ticker": [normalize_ticker(value) for value in table[symbol_column]],
            "Name": table[name_column].astype(str) if name_column is not None else "",
            "Sector": table[sector_column].astype(str) if sector_column is not None else "",
            "Industry": table[industry_column].astype(str) if industry_column is not None else "",
        })
        frame = frame[frame["Ticker"].str.match(r"^[A-Z0-9\-]+$")]
        return frame.drop_duplicates("Ticker").reset_index(drop=True)
    return pd.DataFrame(columns=["Ticker", "Name", "Sector", "Industry"])


def load_us_index_members(name):
    """Members of a US index as a DataFrame (Ticker, Name, Sector, Industry). Needs network."""

    if name == "Dow Jones 30":
        return pd.DataFrame({"Ticker": DOW_JONES_30, "Name": "", "Sector": "", "Industry": ""})
    if name == "S&P 500":
        try:
            members = members_from_tables(_read_html_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"), 400)
            if not members.empty:
                return members
        except Exception:
            pass
        csv_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        request = Request(csv_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=20) as response:
            table = pd.read_csv(io.BytesIO(response.read()))
        return members_from_tables([table], 400)
    if name == "Nasdaq-100":
        # The constituents moved from the index article to a list page in 2026.
        for url in ["https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies", "https://en.wikipedia.org/wiki/Nasdaq-100"]:
            members = members_from_tables(_read_html_tables(url), 90)
            if not members.empty:
                return members
        return members
    raise ValueError(f"Unknown universe: {name}")
