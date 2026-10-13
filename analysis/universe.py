"""Stock universes for the screener (US and Hong Kong).

Index members are read from Wikipedia at run time (cached by the app), with
a public CSV as a fallback for the S&P 500 and a bundled list as a fallback
for the Hang Seng Index; edit ``HANG_SENG_MEMBERS`` when the index is
rebalanced.
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
    "HK": {
        "indexes": {
            "Hang Seng Index": "^HSI",
            "Hang Seng China Enterprises": "^HSCE",
            # Yahoo has no usable history for the index itself (HSTECH.HK
            # returns one bar), so track it through the CSOP tracker ETF.
            "Hang Seng TECH (3033 ETF)": "3033.HK",
        },
        "universes": ["Hang Seng Index", "Custom list"],
    },
}

DOW_JONES_30 = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]

# Hang Seng Index members (Yahoo symbols), matching Wikipedia in October 2026.
# Only used when the Wikipedia table cannot be read; may lag index rebalances.
HANG_SENG_MEMBERS = {
    "0001.HK": "CK Hutchison", "0002.HK": "CLP Holdings", "0003.HK": "HK & China Gas",
    "0005.HK": "HSBC Holdings", "0006.HK": "Power Assets", "0012.HK": "Henderson Land",
    "0016.HK": "SHK Properties", "0027.HK": "Galaxy Entertainment", "0066.HK": "MTR Corporation",
    "0101.HK": "Hang Lung Properties", "0175.HK": "Geely Automobile", "0241.HK": "Alibaba Health",
    "0267.HK": "CITIC", "0285.HK": "BYD Electronic", "0288.HK": "WH Group",
    "0291.HK": "China Resources Beer", "0300.HK": "Midea Group", "0316.HK": "Orient Overseas",
    "0322.HK": "Tingyi", "0386.HK": "Sinopec", "0388.HK": "HKEX",
    "0669.HK": "Techtronic Industries", "0688.HK": "China Overseas Land", "0700.HK": "Tencent",
    "0762.HK": "China Unicom", "0823.HK": "Link REIT", "0836.HK": "China Resources Power",
    "0857.HK": "PetroChina", "0868.HK": "Xinyi Glass", "0881.HK": "Zhongsheng Group",
    "0883.HK": "CNOOC", "0939.HK": "China Construction Bank", "0941.HK": "China Mobile",
    "0960.HK": "Longfor Group", "0968.HK": "Xinyi Solar", "0981.HK": "SMIC",
    "0992.HK": "Lenovo Group", "1024.HK": "Kuaishou", "1038.HK": "CK Infrastructure",
    "1044.HK": "Hengan International", "1088.HK": "China Shenhua", "1093.HK": "CSPC Pharmaceutical",
    "1099.HK": "Sinopharm Group", "1109.HK": "China Resources Land", "1113.HK": "CK Asset",
    "1177.HK": "Sino Biopharmaceutical", "1209.HK": "China Resources Mixc", "1211.HK": "BYD Company",
    "1299.HK": "AIA Group", "1378.HK": "China Hongqiao", "1398.HK": "ICBC",
    "1810.HK": "Xiaomi", "1876.HK": "Budweiser APAC", "1928.HK": "Sands China",
    "1929.HK": "Chow Tai Fook", "1997.HK": "Wharf REIC", "2015.HK": "Li Auto",
    "2020.HK": "ANTA Sports", "2057.HK": "ZTO Express", "2269.HK": "WuXi Biologics",
    "2313.HK": "Shenzhou International", "2318.HK": "Ping An Insurance", "2319.HK": "Mengniu Dairy",
    "2331.HK": "Li Ning", "2359.HK": "WuXi AppTec", "2382.HK": "Sunny Optical",
    "2388.HK": "BOC Hong Kong", "2618.HK": "JD Logistics", "2628.HK": "China Life",
    "2688.HK": "ENN Energy", "2899.HK": "Zijin Mining", "3690.HK": "Meituan",
    "3692.HK": "Hansoh Pharmaceutical", "3968.HK": "China Merchants Bank", "3988.HK": "Bank of China",
    "6618.HK": "JD Health", "6690.HK": "Haier Smart Home", "6862.HK": "Haidilao",
    "9618.HK": "JD.com", "9633.HK": "Nongfu Spring", "9888.HK": "Baidu",
    "9961.HK": "Trip.com", "9988.HK": "Alibaba", "9992.HK": "Pop Mart",
    "9999.HK": "NetEase",
}


def normalize_ticker(raw, market="US"):
    """Yahoo symbol for a user-typed ticker: ``BRK.B`` -> ``BRK-B``, ``700`` -> ``0700.HK``."""

    ticker = str(raw).strip().upper()
    if not ticker:
        return ""
    if market == "HK":
        digits = ticker[:-3] if ticker.endswith(".HK") else ticker
        if digits.isdigit():
            return f"{int(digits):04d}.HK"
        return ticker
    if ticker.startswith("^") or "=" in ticker:
        return ticker
    return ticker.replace(".", "-")


def normalize_symbol(raw):
    """Yahoo symbol for a ticker typed without a market: ``brk.b`` -> ``BRK-B``, ``700`` -> ``0700.HK``.

    Exchange suffixes (``0700.HK``, ``SHOP.TO``, ``VOD.L``) are kept; a
    one-letter suffix is a US share class, which Yahoo writes with a dash.
    """

    ticker = str(raw).strip().upper()
    if ticker.isdigit() or (ticker.endswith(".HK") and ticker[:-3].isdigit()):
        return normalize_ticker(ticker, "HK")
    if re.fullmatch(r"[A-Z]+\.[A-Z]", ticker):
        return ticker.replace(".", "-")
    return ticker


def parse_ticker_list(text, market="US"):
    """Unique Yahoo symbols from comma / space / newline separated text."""

    tickers = [normalize_ticker(part, market) for part in re.split(r"[\s,;]+", text or "")]
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


def hang_seng_members_from_tables(tables, min_rows=60):
    """Hang Seng Index members from Wikipedia's constituents table ("SEHK: 5", Name, Sub-index)."""

    for table in tables:
        columns = {re.sub(r"\[[^\]]*\]", "", str(column)).strip(): column for column in table.columns}
        if "Ticker" not in columns or "Name" not in columns or len(table) < min_rows:
            continue
        codes = table[columns["Ticker"]].astype(str).str.extract(r"(\d+)\s*$")[0]
        frame = pd.DataFrame({
            "Ticker": [normalize_ticker(code, "HK") if isinstance(code, str) else "" for code in codes],
            "Name": table[columns["Name"]].astype(str),
            "Sector": table[columns["Sub-index"]].astype(str) if "Sub-index" in columns else "",
            "Industry": "",
        })
        frame = frame[frame["Ticker"] != ""]
        return frame.drop_duplicates("Ticker").reset_index(drop=True)
    return pd.DataFrame(columns=["Ticker", "Name", "Sector", "Industry"])


def load_hang_seng_members():
    """Hang Seng Index members from Wikipedia. Needs network."""

    return hang_seng_members_from_tables(_read_html_tables("https://en.wikipedia.org/wiki/Hang_Seng_Index"))


def hang_seng_members():
    """The bundled Hang Seng Index list (offline fallback)."""

    return pd.DataFrame({
        "Ticker": list(HANG_SENG_MEMBERS),
        "Name": list(HANG_SENG_MEMBERS.values()),
        "Sector": "",
        "Industry": "",
    })
