"""Stock universes for the screener (US and Hong Kong).

US index members are read from Wikipedia at run time (cached by the app),
with a public CSV as a fallback for the S&P 500. The Hang Seng Index list is
bundled because its Wikipedia table is not reliably machine-readable; edit
``HANG_SENG_MEMBERS`` when the index is rebalanced.
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
            "Hang Seng TECH": "HSTECH.HK",
        },
        "universes": ["Hang Seng Index", "Custom list"],
    },
}

DOW_JONES_30 = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]

# Hang Seng Index members (Yahoo symbols). Bundled; may lag index rebalances.
HANG_SENG_MEMBERS = {
    "0001.HK": "CK Hutchison", "0002.HK": "CLP Holdings", "0003.HK": "HK & China Gas",
    "0005.HK": "HSBC Holdings", "0006.HK": "Power Assets", "0011.HK": "Hang Seng Bank",
    "0012.HK": "Henderson Land", "0016.HK": "SHK Properties", "0027.HK": "Galaxy Entertainment",
    "0066.HK": "MTR Corporation", "0101.HK": "Hang Lung Properties", "0175.HK": "Geely Automobile",
    "0241.HK": "Alibaba Health", "0267.HK": "CITIC", "0285.HK": "BYD Electronic",
    "0288.HK": "WH Group", "0291.HK": "China Resources Beer", "0316.HK": "Orient Overseas",
    "0322.HK": "Tingyi", "0386.HK": "Sinopec", "0388.HK": "HKEX",
    "0669.HK": "Techtronic Industries", "0688.HK": "China Overseas Land", "0700.HK": "Tencent",
    "0762.HK": "China Unicom", "0823.HK": "Link REIT", "0836.HK": "China Resources Power",
    "0857.HK": "PetroChina", "0868.HK": "Xinyi Glass", "0881.HK": "Zhongsheng Group",
    "0883.HK": "CNOOC", "0939.HK": "China Construction Bank", "0941.HK": "China Mobile",
    "0960.HK": "Longfor Group", "0968.HK": "Xinyi Solar", "0981.HK": "SMIC",
    "0992.HK": "Lenovo Group", "1024.HK": "Kuaishou", "1038.HK": "CK Infrastructure",
    "1044.HK": "Hengan International", "1088.HK": "China Shenhua", "1093.HK": "CSPC Pharmaceutical",
    "1109.HK": "China Resources Land", "1113.HK": "CK Asset", "1177.HK": "Sino Biopharmaceutical",
    "1209.HK": "China Resources Mixc", "1211.HK": "BYD Company", "1299.HK": "AIA Group",
    "1378.HK": "China Hongqiao", "1398.HK": "ICBC", "1810.HK": "Xiaomi",
    "1876.HK": "Budweiser APAC", "1928.HK": "Sands China", "1929.HK": "Chow Tai Fook",
    "1997.HK": "Wharf REIC", "2015.HK": "Li Auto", "2020.HK": "ANTA Sports",
    "2269.HK": "WuXi Biologics", "2313.HK": "Shenzhou International", "2318.HK": "Ping An Insurance",
    "2319.HK": "Mengniu Dairy", "2331.HK": "Li Ning", "2382.HK": "Sunny Optical",
    "2388.HK": "BOC Hong Kong", "2628.HK": "China Life", "2688.HK": "ENN Energy",
    "2899.HK": "Zijin Mining", "3328.HK": "Bank of Communications", "3690.HK": "Meituan",
    "3692.HK": "Hansoh Pharmaceutical", "3968.HK": "China Merchants Bank", "3988.HK": "Bank of China",
    "6098.HK": "Country Garden Services", "6618.HK": "JD Health", "6690.HK": "Haier Smart Home",
    "6862.HK": "Haidilao", "9618.HK": "JD.com", "9633.HK": "Nongfu Spring",
    "9888.HK": "Baidu", "9961.HK": "Trip.com", "9988.HK": "Alibaba",
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
        columns = {str(column).strip(): column for column in table.columns}
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
        return members_from_tables(_read_html_tables("https://en.wikipedia.org/wiki/Nasdaq-100"), 90)
    raise ValueError(f"Unknown universe: {name}")


def hang_seng_members():
    return pd.DataFrame({
        "Ticker": list(HANG_SENG_MEMBERS),
        "Name": list(HANG_SENG_MEMBERS.values()),
        "Sector": "",
        "Industry": "",
    })
