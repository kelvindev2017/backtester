import hashlib
from datetime import date, timedelta
import json
import os
import re
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Trade Strategy Backtest",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("Trade Strategy Backtester")


# ============================================================
# Password protection
# ============================================================

def check_password():
    """Return True after the correct password has been entered."""

    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    password = st.text_input("Enter Password", type="password")

    if password:
        try:
            correct_password = st.secrets["password"]
        except KeyError:
            st.error("The password is not configured in Streamlit secrets.")
            return False

        if password == correct_password:
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("Password incorrect")

    return False


# ============================================================
# Indicators and signal conditions
# ============================================================

def compute_ma(series, ma_type, period):
    """Calculate an exponential or simple moving average."""

    period = int(period)
    if ma_type == "EMA":
        return series.ewm(span=period, adjust=False).mean()
    return series.rolling(window=period, min_periods=period).mean()


def build_condition(left, operator, right):
    """Build a Boolean condition from two aligned Pandas Series."""

    if operator == ">":
        condition = left > right
    elif operator == "<":
        condition = left < right
    elif operator == ">=":
        condition = left >= right
    elif operator == "<=":
        condition = left <= right
    elif operator == "Cross Above":
        condition = (left > right) & (left.shift(1) <= right.shift(1))
    elif operator == "Cross Below":
        condition = (left < right) & (left.shift(1) >= right.shift(1))
    else:
        condition = pd.Series(False, index=left.index)

    return condition.fillna(False).astype(bool)


# ============================================================
# Price-data loader
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def load_data(symbol, start_date_str, end_date_str, timeframe):
    """Download prices. Yahoo Finance treats the end date as exclusive."""

    start_timestamp = pd.Timestamp(start_date_str).normalize()
    selected_end_timestamp = pd.Timestamp(end_date_str).normalize()
    download_end_timestamp = selected_end_timestamp + pd.Timedelta(days=1)

    try:
        price_data = yf.download(
            symbol,
            start=start_timestamp.strftime("%Y-%m-%d"),
            end=download_end_timestamp.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if price_data is None or price_data.empty:
            return pd.DataFrame()

        price_data = price_data.copy()

        if isinstance(price_data.columns, pd.MultiIndex):
            final_level = price_data.columns.get_level_values(-1)
            if symbol in final_level:
                price_data = price_data.xs(symbol, axis=1, level=-1)
            else:
                price_data.columns = price_data.columns.get_level_values(0)

        price_data.index = pd.to_datetime(price_data.index)
        if getattr(price_data.index, "tz", None) is not None:
            price_data.index = price_data.index.tz_localize(None)

        price_data = price_data.sort_index()
        price_data = price_data.loc[~price_data.index.duplicated(keep="last")]
        price_data = price_data.loc[
            (price_data.index >= start_timestamp)
            & (price_data.index <= selected_end_timestamp)
        ]

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        if any(column not in price_data.columns for column in required_columns):
            return pd.DataFrame()

        price_data = price_data[required_columns].copy()
        price_data = price_data.dropna(subset=["Close"])

        if timeframe == "Weekly":
            price_data = (
                price_data.resample("W-FRI")
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
                .dropna(subset=["Close"])
            )
            price_data = price_data.loc[price_data.index <= selected_end_timestamp]

        elif timeframe == "Monthly":
            price_data = (
                price_data.resample("ME")
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
                .dropna(subset=["Close"])
            )
            price_data = price_data.loc[price_data.index <= selected_end_timestamp]

        return price_data

    except Exception:
        return pd.DataFrame()


# ============================================================
# Earnings loader
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_earnings_data(symbol, start_date_str, end_date_str):
    """Retrieve historical earnings dates and EPS values."""

    try:
        earnings = yf.Ticker(symbol).get_earnings_dates(limit=100)
        if earnings is None or earnings.empty:
            return pd.DataFrame()

        earnings = earnings.copy().reset_index()
        date_column = earnings.columns[0]
        earnings[date_column] = pd.to_datetime(
            earnings[date_column], errors="coerce", utc=True
        ).dt.tz_convert(None)
        earnings = earnings.rename(columns={date_column: "Earnings_Date"})

        for column in ["EPS Estimate", "Reported EPS", "Surprise(%)"]:
            if column in earnings.columns:
                earnings[column] = pd.to_numeric(earnings[column], errors="coerce")

        start_timestamp = pd.Timestamp(start_date_str).normalize()
        end_exclusive = pd.Timestamp(end_date_str).normalize() + pd.Timedelta(days=1)
        earnings = earnings.loc[
            (earnings["Earnings_Date"] >= start_timestamp)
            & (earnings["Earnings_Date"] < end_exclusive)
        ]
        earnings = earnings.dropna(subset=["Earnings_Date"])
        earnings = earnings.sort_values("Earnings_Date")
        earnings = earnings.drop_duplicates(subset=["Earnings_Date"], keep="last")
        return earnings.set_index("Earnings_Date")

    except Exception:
        return pd.DataFrame()


# ============================================================
# Quarterly fundamentals loader
# ============================================================

def first_available_statement(stock, attribute_names):
    """Return the first non-empty financial statement available."""

    for attribute_name in attribute_names:
        try:
            statement = getattr(stock, attribute_name)
            if callable(statement):
                statement = statement()
            if statement is not None and not statement.empty:
                return statement.copy()
        except Exception:
            continue
    return pd.DataFrame()


def extract_statement_row(statement, candidate_names):
    """Extract the first available statement row using name fallbacks."""

    if statement is None or statement.empty:
        return pd.Series(dtype=float)

    normalized_lookup = {
        str(index_value).strip().lower(): index_value
        for index_value in statement.index
    }

    for candidate in candidate_names:
        original_name = normalized_lookup.get(candidate.strip().lower())
        if original_name is not None:
            values = statement.loc[original_name]
            if isinstance(values, pd.DataFrame):
                values = values.iloc[0]
            values = pd.to_numeric(values, errors="coerce")
            values.index = pd.to_datetime(values.index, errors="coerce")
            values = values.loc[~values.index.isna()]
            return values.sort_index()

    return pd.Series(dtype=float)


def load_yahoo_quarterly_fundamentals(symbol):
    """Load the recent quarters available through Yahoo Finance."""

    try:
        stock = yf.Ticker(symbol)
        income_statement = first_available_statement(
            stock,
            ["quarterly_income_stmt", "quarterly_financials"],
        )
        cash_flow_statement = first_available_statement(
            stock,
            ["quarterly_cashflow", "quarterly_cash_flow"],
        )

        revenue = extract_statement_row(
            income_statement,
            ["Total Revenue", "Operating Revenue", "Revenue"],
        )
        operating_expense = extract_statement_row(
            income_statement,
            ["Operating Expense", "Total Operating Expenses", "Operating Expenses"],
        )
        total_expense = extract_statement_row(
            income_statement,
            ["Total Expenses", "Costs And Expenses", "Total Costs And Expenses"],
        )
        operating_income = extract_statement_row(
            income_statement,
            ["Operating Income", "Operating Income Loss"],
        )
        net_income = extract_statement_row(
            income_statement,
            [
                "Net Income",
                "Net Income Common Stockholders",
                "Net Income Including Noncontrolling Interests",
            ],
        )
        capex = extract_statement_row(
            cash_flow_statement,
            [
                "Capital Expenditure",
                "Capital Expenditures",
                "Purchase Of PPE",
                "Investments In Property Plant And Equipment",
            ],
        ).abs()

        all_dates = revenue.index.union(operating_expense.index)
        all_dates = all_dates.union(total_expense.index).union(capex.index)
        all_dates = all_dates.union(net_income.index).sort_values()
        if len(all_dates) == 0:
            return pd.DataFrame()

        result = pd.DataFrame(index=all_dates)
        result["Revenue"] = revenue.reindex(all_dates)
        result["Operating_Expense"] = operating_expense.reindex(all_dates)
        result["Total_Expense"] = total_expense.reindex(all_dates)
        result["CapEx"] = capex.reindex(all_dates)
        result["Net_Income"] = net_income.reindex(all_dates)

        # If Yahoo omits total expenses, derive it as revenue minus operating income.
        derived_total_expense = revenue.reindex(all_dates) - operating_income.reindex(all_dates)
        result["Total_Expense"] = result["Total_Expense"].combine_first(
            derived_total_expense
        )
        return result.dropna(how="all")

    except Exception:
        return pd.DataFrame()


def get_sec_user_agent():
    """Return an SEC-compliant identifying User-Agent."""

    try:
        configured_value = st.secrets.get("SEC_USER_AGENT", "")
    except Exception:
        configured_value = ""

    return (
        configured_value
        or os.getenv("SEC_USER_AGENT", "")
        or "StreamlitBacktester/1.0 kelvin@example.com"
    )


def sec_request_json(url):
    """Read JSON from an SEC endpoint with an identifying User-Agent."""

    request = Request(
        url,
        headers={
            "User-Agent": get_sec_user_agent(),
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        },
    )

    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def get_sec_cik(symbol):
    """Resolve a US-listed ticker to its SEC CIK."""

    ticker_map = sec_request_json(
        "https://www.sec.gov/files/company_tickers.json"
    )

    requested_symbol = symbol.upper().replace(".", "-")

    for item in ticker_map.values():
        sec_symbol = str(item.get("ticker", "")).upper()
        if sec_symbol == requested_symbol:
            return int(item["cik_str"])

    return None


def get_sec_fact_entries(company_facts, candidate_tags):
    """Return USD fact entries for the first SEC tag containing usable data."""

    us_gaap_facts = company_facts.get("facts", {}).get("us-gaap", {})

    for tag in candidate_tags:
        fact = us_gaap_facts.get(tag, {})
        entries = fact.get("units", {}).get("USD", [])
        if entries:
            return entries

    return []


def prepare_sec_duration_entries(entries):
    """Clean SEC duration facts and keep quarterly/annual filing observations."""

    records = []

    for entry in entries:
        if entry.get("form") not in {"10-Q", "10-K", "20-F", "40-F"}:
            continue

        start_date = pd.to_datetime(entry.get("start"), errors="coerce")
        end_date = pd.to_datetime(entry.get("end"), errors="coerce")
        filed_date = pd.to_datetime(entry.get("filed"), errors="coerce")
        value = pd.to_numeric(entry.get("val"), errors="coerce")
        fiscal_year = pd.to_numeric(entry.get("fy"), errors="coerce")
        fiscal_period = str(entry.get("fp", "")).upper()

        if (
            pd.isna(start_date)
            or pd.isna(end_date)
            or pd.isna(value)
            or pd.isna(fiscal_year)
        ):
            continue

        duration_days = int((end_date - start_date).days) + 1

        if duration_days < 60 or duration_days > 390:
            continue

        records.append(
            {
                "Start_Date": start_date,
                "End_Date": end_date,
                "Filed_Date": filed_date,
                "Value": float(value),
                "Fiscal_Year": int(fiscal_year),
                "Fiscal_Period": fiscal_period,
                "Duration_Days": duration_days,
                "Form": entry.get("form"),
            }
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values(
        ["Fiscal_Year", "End_Date", "Duration_Days", "Filed_Date"]
    )


def select_latest_shortest(group, minimum_days, maximum_days):
    """Select the latest-filed value among the shortest matching durations."""

    candidates = group.loc[
        (group["Duration_Days"] >= minimum_days)
        & (group["Duration_Days"] <= maximum_days)
    ].copy()

    if candidates.empty:
        return None

    shortest_duration = candidates["Duration_Days"].min()
    candidates = candidates.loc[
        candidates["Duration_Days"] <= shortest_duration + 7
    ]
    candidates = candidates.sort_values("Filed_Date")
    return candidates.iloc[-1]


def select_latest_longest(group, minimum_days, maximum_days):
    """Select the latest-filed value among the longest matching durations."""

    candidates = group.loc[
        (group["Duration_Days"] >= minimum_days)
        & (group["Duration_Days"] <= maximum_days)
    ].copy()

    if candidates.empty:
        return None

    longest_duration = candidates["Duration_Days"].max()
    candidates = candidates.loc[
        candidates["Duration_Days"] >= longest_duration - 7
    ]
    candidates = candidates.sort_values("Filed_Date")
    return candidates.iloc[-1]


def extract_sec_fiscal_quarters(company_facts, candidate_tags):
    """
    Build fiscal-quarter values from SEC duration facts.

    Q1-Q3 use standalone approximately three-month facts whenever available.
    Q4 is derived from the full fiscal-year value less Q1, Q2 and Q3. This
    avoids relying on CYyyyyQn frame labels, which omit many fiscal quarters.
    """

    entries = get_sec_fact_entries(company_facts, candidate_tags)
    facts = prepare_sec_duration_entries(entries)

    if facts.empty:
        return pd.Series(dtype=float)

    quarter_records = []

    for fiscal_year, fiscal_year_group in facts.groupby("Fiscal_Year"):
        selected_quarters = {}

        for quarter_name in ["Q1", "Q2", "Q3"]:
            period_group = fiscal_year_group.loc[
                fiscal_year_group["Fiscal_Period"] == quarter_name
            ]
            selected = select_latest_shortest(period_group, 60, 120)

            if selected is not None:
                selected_quarters[quarter_name] = selected
                quarter_records.append(
                    (
                        selected["End_Date"],
                        selected["Value"],
                        fiscal_year,
                        quarter_name,
                    )
                )

        annual_group = fiscal_year_group.loc[
            fiscal_year_group["Fiscal_Period"] == "FY"
        ]
        annual = select_latest_longest(annual_group, 300, 390)

        if annual is not None and all(
            quarter_name in selected_quarters
            for quarter_name in ["Q1", "Q2", "Q3"]
        ):
            q4_value = annual["Value"] - sum(
                selected_quarters[quarter_name]["Value"]
                for quarter_name in ["Q1", "Q2", "Q3"]
            )

            # Accept ordinary positive/negative accounting values but reject
            # implausibly large derivation errors caused by mixed contexts.
            comparison_scale = max(abs(annual["Value"]), 1.0)
            if abs(q4_value) <= comparison_scale * 1.5:
                quarter_records.append(
                    (annual["End_Date"], q4_value, fiscal_year, "Q4")
                )

    if not quarter_records:
        return pd.Series(dtype=float)

    quarter_df = pd.DataFrame(
        quarter_records,
        columns=["End_Date", "Value", "Fiscal_Year", "Fiscal_Quarter"],
    )
    quarter_df = quarter_df.sort_values(
        ["End_Date", "Fiscal_Year", "Fiscal_Quarter"]
    )
    quarter_df = quarter_df.drop_duplicates(
        subset=["End_Date"],
        keep="last",
    )

    return pd.Series(
        quarter_df["Value"].to_numpy(),
        index=pd.DatetimeIndex(quarter_df["End_Date"]),
        dtype=float,
    ).sort_index()


def extract_sec_cumulative_quarters(company_facts, candidate_tags):
    """
    Convert cumulative SEC cash-flow facts into individual fiscal quarters.

    This is used for CapEx because 10-Q cash-flow statements commonly report
    year-to-date values rather than standalone Q2 and Q3 amounts.
    """

    entries = get_sec_fact_entries(company_facts, candidate_tags)
    facts = prepare_sec_duration_entries(entries)

    if facts.empty:
        return pd.Series(dtype=float)

    quarter_records = []

    for fiscal_year, fiscal_year_group in facts.groupby("Fiscal_Year"):
        cumulative = {}

        period_ranges = {
            "Q1": (60, 120),
            "Q2": (120, 220),
            "Q3": (200, 310),
            "FY": (300, 390),
        }

        for period_name, (minimum_days, maximum_days) in period_ranges.items():
            period_group = fiscal_year_group.loc[
                fiscal_year_group["Fiscal_Period"] == period_name
            ]
            selected = select_latest_longest(
                period_group,
                minimum_days,
                maximum_days,
            )
            if selected is not None:
                cumulative[period_name] = selected

        if "Q1" in cumulative:
            quarter_records.append(
                (cumulative["Q1"]["End_Date"], cumulative["Q1"]["Value"])
            )

        if "Q2" in cumulative and "Q1" in cumulative:
            quarter_records.append(
                (
                    cumulative["Q2"]["End_Date"],
                    cumulative["Q2"]["Value"] - cumulative["Q1"]["Value"],
                )
            )

        if "Q3" in cumulative and "Q2" in cumulative:
            quarter_records.append(
                (
                    cumulative["Q3"]["End_Date"],
                    cumulative["Q3"]["Value"] - cumulative["Q2"]["Value"],
                )
            )

        if "FY" in cumulative and "Q3" in cumulative:
            quarter_records.append(
                (
                    cumulative["FY"]["End_Date"],
                    cumulative["FY"]["Value"] - cumulative["Q3"]["Value"],
                )
            )

    if not quarter_records:
        return pd.Series(dtype=float)

    quarter_df = pd.DataFrame(
        quarter_records,
        columns=["End_Date", "Value"],
    ).sort_values("End_Date")
    quarter_df = quarter_df.drop_duplicates(subset=["End_Date"], keep="last")

    return pd.Series(
        quarter_df["Value"].abs().to_numpy(),
        index=pd.DatetimeIndex(quarter_df["End_Date"]),
        dtype=float,
    ).sort_index()


def load_sec_quarterly_fundamentals(symbol):
    """Load extended quarterly financial history from SEC Company Facts."""

    try:
        cik = get_sec_cik(symbol)
        if cik is None:
            return pd.DataFrame(), "Ticker is not mapped to an SEC CIK."

        company_facts = sec_request_json(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
        )

        revenue = extract_sec_fiscal_quarters(
            company_facts,
            [
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet",
                "Revenues",
            ],
        )
        operating_expense = extract_sec_fiscal_quarters(
            company_facts,
            ["OperatingExpenses"],
        )
        total_expense = extract_sec_fiscal_quarters(
            company_facts,
            ["CostsAndExpenses"],
        )
        operating_income = extract_sec_fiscal_quarters(
            company_facts,
            ["OperatingIncomeLoss"],
        )
        capex = extract_sec_cumulative_quarters(
            company_facts,
            [
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "PaymentsForAdditionsToPropertyPlantAndEquipment",
            ],
        )
        net_income = extract_sec_fiscal_quarters(
            company_facts,
            ["NetIncomeLoss", "ProfitLoss"],
        )

        all_dates = revenue.index.union(operating_expense.index)
        all_dates = all_dates.union(total_expense.index).union(capex.index)
        all_dates = all_dates.union(net_income.index).sort_values()

        if len(all_dates) == 0:
            return pd.DataFrame(), "SEC returned no usable quarterly duration facts."

        result = pd.DataFrame(index=all_dates)
        result["Revenue"] = revenue.reindex(all_dates)
        result["Operating_Expense"] = operating_expense.reindex(all_dates)
        result["Total_Expense"] = total_expense.reindex(all_dates)
        result["CapEx"] = capex.reindex(all_dates)
        result["Net_Income"] = net_income.reindex(all_dates)

        derived_total_expense = (
            revenue.reindex(all_dates) - operating_income.reindex(all_dates)
        )
        result["Total_Expense"] = result["Total_Expense"].combine_first(
            derived_total_expense
        )

        return result.dropna(how="all"), ""

    except Exception as error:
        return pd.DataFrame(), f"{type(error).__name__}: {error}"


@st.cache_data(ttl=21600, show_spinner=False)
def load_quarterly_fundamentals(
    symbol,
    start_date_str,
    end_date_str,
    cache_version="2026-09-29-v4",
):
    """Combine long SEC history with recent Yahoo Finance statements."""

    del cache_version

    sec_data, sec_error = load_sec_quarterly_fundamentals(symbol)
    yahoo_data = load_yahoo_quarterly_fundamentals(symbol)

    if sec_data.empty and yahoo_data.empty:
        return pd.DataFrame(), "No SEC or Yahoo quarterly fundamentals were returned.", "None"

    if sec_data.empty:
        fundamentals = yahoo_data.copy()
        source = "Yahoo Finance only"
    elif yahoo_data.empty:
        fundamentals = sec_data.copy()
        source = "SEC Company Facts only"
    else:
        # Quarter-end dates can differ by a few days between data suppliers.
        # Keep SEC as the historical base, then use Yahoo only for dates not
        # already represented within 14 days of an SEC fiscal quarter end.
        fundamentals = sec_data.copy()

        for yahoo_date, yahoo_row in yahoo_data.iterrows():
            if fundamentals.empty:
                fundamentals.loc[yahoo_date] = yahoo_row
                continue

            date_distance = (fundamentals.index - yahoo_date).to_series().abs()
            nearest_date = date_distance.idxmin()
            nearest_distance_days = abs((nearest_date - yahoo_date).days)

            if nearest_distance_days <= 14:
                for column in yahoo_data.columns:
                    if pd.notna(yahoo_row.get(column)):
                        fundamentals.loc[nearest_date, column] = yahoo_row[column]
            else:
                fundamentals.loc[yahoo_date] = yahoo_row

        source = "SEC Company Facts plus Yahoo Finance"

    start_timestamp = pd.Timestamp(start_date_str).normalize()
    end_exclusive = pd.Timestamp(end_date_str).normalize() + pd.Timedelta(days=1)
    fundamentals = fundamentals.loc[
        (fundamentals.index >= start_timestamp)
        & (fundamentals.index < end_exclusive)
    ]
    fundamentals = fundamentals.sort_index()
    fundamentals = fundamentals.loc[
        ~fundamentals.index.duplicated(keep="last")
    ]

    return fundamentals.dropna(how="all"), sec_error, source


def choose_fundamental_scale(fundamentals_df):
    """Choose USD millions or billions based on the largest absolute value."""

    if fundamentals_df is None or fundamentals_df.empty:
        return 1e9, "USD billions"
    numeric_values = fundamentals_df.select_dtypes(include=[np.number])
    if numeric_values.empty:
        return 1e9, "USD billions"
    maximum_value = numeric_values.abs().max().max()
    if pd.isna(maximum_value) or maximum_value >= 1e9:
        return 1e9, "USD billions"
    return 1e6, "USD millions"


# ============================================================
# EPS and valuation calculations
# ============================================================

def build_valuation_data(price_df, earnings_history_df):
    """Calculate TTM EPS, trailing P/E and an estimate-based P/E proxy."""

    valuation = pd.DataFrame(index=price_df.index)
    valuation["Close"] = pd.to_numeric(price_df["Close"], errors="coerce")
    valuation["TTM_Reported_EPS"] = np.nan
    valuation["TTM_Estimated_EPS"] = np.nan
    valuation["Trailing_PE"] = np.nan
    valuation["Estimate_Based_PE_Proxy"] = np.nan

    if earnings_history_df is None or earnings_history_df.empty:
        valuation["PE_Reconstruction_Error"] = np.nan
        return valuation, pd.DataFrame()

    earnings = earnings_history_df.copy().sort_index()

    for column in ["Reported EPS", "EPS Estimate"]:
        if column in earnings.columns:
            earnings[column] = pd.to_numeric(earnings[column], errors="coerce")
        else:
            earnings[column] = np.nan

    earnings["TTM_Reported_EPS"] = (
        earnings["Reported EPS"].rolling(window=4, min_periods=4).sum()
    )
    earnings["TTM_Estimated_EPS"] = (
        earnings["EPS Estimate"].rolling(window=4, min_periods=4).sum()
    )

    eps_columns = earnings[["TTM_Reported_EPS", "TTM_Estimated_EPS"]].copy()
    combined_index = valuation.index.union(eps_columns.index).sort_values()
    earnings_daily = eps_columns.reindex(combined_index).ffill().reindex(valuation.index)

    valuation["TTM_Reported_EPS"] = earnings_daily["TTM_Reported_EPS"]
    valuation["TTM_Estimated_EPS"] = earnings_daily["TTM_Estimated_EPS"]
    valuation["Trailing_PE"] = np.where(
        valuation["TTM_Reported_EPS"] > 0,
        valuation["Close"] / valuation["TTM_Reported_EPS"],
        np.nan,
    )
    valuation["Estimate_Based_PE_Proxy"] = np.where(
        valuation["TTM_Estimated_EPS"] > 0,
        valuation["Close"] / valuation["TTM_Estimated_EPS"],
        np.nan,
    )
    valuation = valuation.replace([np.inf, -np.inf], np.nan)
    valuation["Implied_Close_From_PE"] = (
        valuation["Trailing_PE"] * valuation["TTM_Reported_EPS"]
    )
    valuation["PE_Reconstruction_Error"] = (
        valuation["Implied_Close_From_PE"] - valuation["Close"]
    ).abs()
    return valuation, earnings


# ============================================================
# Benchmark loader
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def load_benchmark_data(start_date_str, end_date_str, timeframe):
    """Download S&P 500 and Nasdaq-100 and normalize each to 100."""

    symbols = ["^GSPC", "^NDX"]
    start_timestamp = pd.Timestamp(start_date_str).normalize()
    selected_end_timestamp = pd.Timestamp(end_date_str).normalize()
    download_end_timestamp = selected_end_timestamp + pd.Timedelta(days=1)

    try:
        data = yf.download(
            symbols,
            start=start_timestamp.strftime("%Y-%m-%d"),
            end=download_end_timestamp.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if data is None or data.empty:
            return pd.DataFrame()

        close = data["Close"].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
        close.index = pd.to_datetime(close.index)
        if getattr(close.index, "tz", None) is not None:
            close.index = close.index.tz_localize(None)
        close = close.sort_index()
        close = close.loc[
            (close.index >= start_timestamp) & (close.index <= selected_end_timestamp)
        ]
        close = close.rename(columns={"^GSPC": "S&P 500", "^NDX": "Nasdaq-100"})

        if timeframe == "Weekly":
            close = close.resample("W-FRI").last().dropna(how="all")
            close = close.loc[close.index <= selected_end_timestamp]
        elif timeframe == "Monthly":
            close = close.resample("ME").last().dropna(how="all")
            close = close.loc[close.index <= selected_end_timestamp]

        normalized = pd.DataFrame(index=close.index)
        for column in close.columns:
            valid = close[column].dropna()
            if not valid.empty and valid.iloc[0] != 0:
                normalized[column] = close[column] / valid.iloc[0] * 100
        return normalized

    except Exception:
        return pd.DataFrame()


# ============================================================
# Position engine
# ============================================================

def run_position_engine(price_df, buy_condition, sell_condition):
    """Generate long-only positions and entry/exit markers."""

    positions, buy_markers, sell_markers = [], [], []
    in_trade = False

    for row_number in range(len(price_df)):
        buy_signal = False
        sell_signal = False
        if not in_trade and bool(buy_condition.iloc[row_number]):
            in_trade = True
            buy_signal = True
        elif in_trade and bool(sell_condition.iloc[row_number]):
            in_trade = False
            sell_signal = True

        positions.append(1 if in_trade else 0)
        buy_markers.append(buy_signal)
        sell_markers.append(sell_signal)

    result = price_df.copy()
    result["Raw_Position"] = pd.Series(positions, index=result.index, dtype=int)
    result["Buy_Signal"] = pd.Series(buy_markers, index=result.index, dtype=bool)
    result["Sell_Signal"] = pd.Series(sell_markers, index=result.index, dtype=bool)
    result["Position"] = result["Raw_Position"].shift(1).fillna(0).astype(int)
    return result


# ============================================================
# Main application
# ============================================================

if check_password():
    st.write("Welcome to the protected app!")

    st.sidebar.header("Strategy Settings")
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    timeframe = st.sidebar.selectbox("Timeframe", ["Daily", "Weekly", "Monthly"], index=0)
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(date(2022, 1, 1), date.today()),
        max_value=date.today(),
        format="YYYY-MM-DD",
    )
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)", min_value=1.0, value=10000.0, step=1000.0
    )

    st.sidebar.subheader("Moving Averages")
    fast_type = st.sidebar.selectbox("Fast MA Type", ["EMA", "SMA"], index=0)
    fast_period = int(st.sidebar.number_input("Fast MA Period", min_value=1, value=20, step=1))
    slow_type = st.sidebar.selectbox("Slow MA Type", ["EMA", "SMA"], index=1)
    slow_period = int(st.sidebar.number_input("Slow MA Period", min_value=1, value=50, step=1))

    signal_options = ["Close", "Fast_MA", "Slow_MA"]
    operator_options = [">", "<", ">=", "<=", "Cross Above", "Cross Below"]

    st.sidebar.subheader("Buy Signal")
    buy_col_left, buy_col_operator, buy_col_right = st.sidebar.columns([1.25, 1.1, 1.25])
    with buy_col_left:
        buy_left = st.selectbox("Buy left signal", signal_options, index=1, key="buy_left", label_visibility="collapsed")
    with buy_col_operator:
        buy_operator = st.selectbox("Buy operator", operator_options, index=4, key="buy_operator", label_visibility="collapsed")
    with buy_col_right:
        buy_right = st.selectbox("Buy right signal", signal_options, index=2, key="buy_right", label_visibility="collapsed")
    st.sidebar.caption(f"Buy when: `{buy_left} {buy_operator} {buy_right}`")

    st.sidebar.divider()
    st.sidebar.subheader("Sell Signal")
    sell_col_left, sell_col_operator, sell_col_right = st.sidebar.columns([1.25, 1.1, 1.25])
    with sell_col_left:
        sell_left = st.selectbox("Sell left signal", signal_options, index=1, key="sell_left", label_visibility="collapsed")
    with sell_col_operator:
        sell_operator = st.selectbox("Sell operator", operator_options, index=5, key="sell_operator", label_visibility="collapsed")
    with sell_col_right:
        sell_right = st.selectbox("Sell right signal", signal_options, index=2, key="sell_right", label_visibility="collapsed")
    st.sidebar.caption(f"Sell when: `{sell_left} {sell_operator} {sell_right}`")

    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.info("Please select both a start date and an end date.")
        st.stop()

    start_date, end_date = date_range
    if start_date > end_date:
        st.error("The start date must be earlier than the end date.")
        st.stop()

    ticker = ticker.strip().upper()
    if not ticker:
        st.error("Please enter a ticker symbol.")
        st.stop()

    start_date_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_date_str = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    earnings_lookback_start = (
        pd.Timestamp(start_date) - pd.DateOffset(years=2)
    ).strftime("%Y-%m-%d")

    with st.spinner(f"Loading {ticker} data from {start_date_str} to {end_date_str}..."):
        df = load_data(ticker, start_date_str, end_date_str, timeframe)
        earnings_history_df = load_earnings_data(ticker, earnings_lookback_start, end_date_str)
        fundamentals_df, fundamentals_sec_error, fundamentals_source = load_quarterly_fundamentals(
            ticker,
            start_date_str,
            end_date_str,
        )
        benchmark_df = load_benchmark_data(start_date_str, end_date_str, timeframe)

    if df.empty:
        st.error(f"No price data was returned for {ticker} between {start_date_str} and {end_date_str}.")
        st.stop()

    st.caption(
        f"Loaded {len(df):,} {timeframe.lower()} bars: "
        f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}"
    )


    if fundamentals_df.empty:
        st.warning(
            "No quarterly fundamentals were available for the selected period. "
            f"SEC diagnostic: {fundamentals_sec_error or 'No SEC error reported.'}"
        )
    else:
        st.caption(
            f"Quarterly fundamentals source: {fundamentals_source}. "
            f"Loaded {len(fundamentals_df)} quarter-end records from "
            f"{fundamentals_df.index.min():%Y-%m-%d} to "
            f"{fundamentals_df.index.max():%Y-%m-%d}."
        )
        if fundamentals_sec_error:
            st.caption(f"SEC fallback diagnostic: {fundamentals_sec_error}")

    df["Fast_MA"] = compute_ma(df["Close"], fast_type, fast_period)
    df["Slow_MA"] = compute_ma(df["Close"], slow_type, slow_period)

    if len(df) < max(fast_period, slow_period):
        st.warning(
            f"The selected period contains only {len(df)} bars, but the longest "
            f"moving average requires {max(fast_period, slow_period)} bars."
        )

    valuation_df, earnings_calculated_df = build_valuation_data(df, earnings_history_df)

    if earnings_calculated_df.empty:
        earnings_display_df = pd.DataFrame()
    else:
        display_start = pd.Timestamp(start_date).normalize()
        display_end = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
        earnings_display_df = earnings_calculated_df.loc[
            (earnings_calculated_df.index >= display_start)
            & (earnings_calculated_df.index < display_end)
        ].copy()

    pe_error = valuation_df["PE_Reconstruction_Error"].dropna()
    if not pe_error.empty:
        st.caption(f"Maximum internal P/E calculation error: ${pe_error.max():.8f}")

    signal_map = {"Close": df["Close"], "Fast_MA": df["Fast_MA"], "Slow_MA": df["Slow_MA"]}
    buy_condition = build_condition(signal_map[buy_left], buy_operator, signal_map[buy_right])
    sell_condition = build_condition(signal_map[sell_left], sell_operator, signal_map[sell_right])
    df = run_position_engine(df, buy_condition, sell_condition)

    df["Returns"] = df["Close"].pct_change().fillna(0.0)
    df["Strat_Returns"] = df["Position"] * df["Returns"]
    df["Equity"] = float(initial_capital) * (1.0 + df["Strat_Returns"]).cumprod()
    df["Peak"] = df["Equity"].cummax()
    df["Drawdown"] = df["Equity"] / df["Peak"] - 1.0

    final_equity = df["Equity"].iloc[-1]
    total_return = (final_equity / float(initial_capital) - 1.0) * 100.0
    max_drawdown = df["Drawdown"].min() * 100.0
    active_returns = df.loc[df["Strat_Returns"] != 0, "Strat_Returns"]
    positive_active_bars = 0.0 if active_returns.empty else active_returns.gt(0).mean() * 100.0

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Final Equity", f"${final_equity:,.2f}", f"{total_return:.2f}%")
    metric_col2.metric("Max Drawdown", f"{max_drawdown:.2f}%")
    metric_col3.metric("Positive Active Bars", f"{positive_active_bars:.1f}%")
    st.info(
        f"**BUY:** `{buy_left} {buy_operator} {buy_right}`\n\n"
        f"**SELL:** `{sell_left} {sell_operator} {sell_right}`"
    )

    fig = make_subplots(
        rows=7,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        subplot_titles=(
            f"{ticker} Price, Indicators and Trade Signals",
            f"{ticker} Quarterly and TTM EPS",
            f"{ticker} Quarterly Revenue, OpEx, Total Expense, CapEx and Net Income",
            f"{ticker} Trailing P/E and Estimate-Based P/E Proxy",
            "Market Benchmark Performance, Normalized to 100",
            "Portfolio Equity Curve ($)",
            "Drawdown Profile (%)",
        ),
        row_heights=[0.27, 0.14, 0.16, 0.13, 0.12, 0.11, 0.07],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )

    # Row 1: price, moving averages and signals
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="gray", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Fast_MA"], name=f"{fast_type} {fast_period}", line=dict(color="orange", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Slow_MA"], name=f"{slow_type} {slow_period}", line=dict(color="blue", width=1.5)), row=1, col=1)

    buy_points = df.loc[df["Buy_Signal"]]
    sell_points = df.loc[df["Sell_Signal"]]
    fig.add_trace(
        go.Scatter(
            x=buy_points.index, y=buy_points["Close"], mode="markers", name="BUY signal",
            marker=dict(symbol="triangle-up", size=12, color="lime", line=dict(color="darkgreen", width=1)),
            hovertemplate="BUY signal<br>Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=sell_points.index, y=sell_points["Close"], mode="markers", name="SELL signal",
            marker=dict(symbol="triangle-down", size=12, color="red", line=dict(color="darkred", width=1)),
            hovertemplate="SELL signal<br>Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
        ), row=1, col=1
    )

    # Row 2: quarterly and TTM EPS
    eps_trace_added = False
    eps_trace_definitions = [
        ("Reported EPS", "Quarterly Reported EPS", "bar", "royalblue", False),
        ("EPS Estimate", "Quarterly EPS Estimate", "scatter", "orange", False),
        ("TTM_Reported_EPS", "TTM Reported EPS", "scatter", "limegreen", True),
        ("TTM_Estimated_EPS", "TTM EPS Estimate Proxy", "scatter", "cyan", True),
    ]

    for column, trace_name, trace_type, color, use_secondary in eps_trace_definitions:
        if not earnings_display_df.empty and column in earnings_display_df.columns:
            series = earnings_display_df[column].dropna()
            if series.empty:
                continue
            eps_trace_added = True
            if trace_type == "bar":
                trace = go.Bar(x=series.index, y=series, name=trace_name, marker_color=color, opacity=0.75)
            else:
                trace = go.Scatter(
                    x=series.index,
                    y=series,
                    mode="lines+markers",
                    name=trace_name,
                    line=dict(color=color, width=2, dash="dot" if "Estimate" in trace_name else "solid"),
                    marker=dict(size=7, color=color),
                )
            fig.add_trace(trace, row=2, col=1, secondary_y=use_secondary)

    if not eps_trace_added:
        fig.add_annotation(
            x=0.5, y=0.5, xref="x2 domain", yref="y2 domain",
            text="No quarterly EPS event falls within the selected display period",
            showarrow=False, font=dict(color="gray", size=12)
        )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray", row=2, col=1, secondary_y=False)

    # Row 3: quarterly fundamentals
    fundamentals_trace_added = False
    scale_divisor, fundamentals_axis_title = choose_fundamental_scale(fundamentals_df)
    fundamental_definitions = [
        ("Revenue", "Revenue", "royalblue"),
        ("Total_Expense", "Total expenses", "crimson"),
        ("Net_Income", "Net income", "limegreen"),
        ("Operating_Expense", "Operating expenses", "orange"),
        ("CapEx", "CapEx spending", "purple"),
    ]

    for column, trace_name, color in fundamental_definitions:
        if not fundamentals_df.empty and column in fundamentals_df.columns:
            series = fundamentals_df[column].dropna() / scale_divisor
            if series.empty:
                continue
            fundamentals_trace_added = True
            fig.add_trace(
                go.Bar(
                    x=series.index,
                    y=series,
                    name=trace_name,
                    marker_color=color,
                    opacity=0.78,
                    hovertemplate=(
                        f"{trace_name}<br>Date: %{{x|%Y-%m-%d}}<br>"
                        f"Value: %{{y:,.2f}} {fundamentals_axis_title}<extra></extra>"
                    ),
                ),
                row=3,
                col=1,
            )

    if not fundamentals_trace_added:
        fig.add_annotation(
            x=0.5, y=0.5, xref="x3 domain", yref="y3 domain",
            text="Quarterly revenue, OpEx, total expense, CapEx and net income data are unavailable",
            showarrow=False, font=dict(color="gray", size=12)
        )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray", row=3, col=1)

    # Row 4: P/E
    trailing_pe = valuation_df["Trailing_PE"].dropna()
    estimate_based_pe = valuation_df["Estimate_Based_PE_Proxy"].dropna()

    if not trailing_pe.empty:
        fig.add_trace(
            go.Scatter(x=trailing_pe.index, y=trailing_pe, mode="lines", name="Trailing P/E", line=dict(color="gold", width=2)),
            row=4, col=1
        )
        median_value = trailing_pe.median()
        fig.add_hline(
            y=median_value, line_width=1, line_dash="dash", line_color="goldenrod",
            annotation_text=f"Median trailing P/E: {median_value:.1f}x",
            annotation_position="top left", row=4, col=1
        )

    if not estimate_based_pe.empty:
        fig.add_trace(
            go.Scatter(x=estimate_based_pe.index, y=estimate_based_pe, mode="lines", name="Estimate-Based P/E Proxy", line=dict(color="cyan", width=2, dash="dot")),
            row=4, col=1
        )
        median_value = estimate_based_pe.median()
        fig.add_hline(
            y=median_value, line_width=1, line_dash="dash", line_color="darkcyan",
            annotation_text=f"Median estimate-based P/E: {median_value:.1f}x",
            annotation_position="bottom left", row=4, col=1
        )

    if trailing_pe.empty and estimate_based_pe.empty:
        fig.add_annotation(
            x=0.5, y=0.5, xref="x4 domain", yref="y4 domain",
            text="Insufficient historical EPS data to calculate P/E",
            showarrow=False, font=dict(color="gray", size=12)
        )

    # Row 5: benchmarks
    benchmark_trace_added = False
    for column, color in [("S&P 500", "deepskyblue"), ("Nasdaq-100", "magenta")]:
        if not benchmark_df.empty and column in benchmark_df.columns:
            series = benchmark_df[column].dropna()
            if not series.empty:
                benchmark_trace_added = True
                fig.add_trace(
                    go.Scatter(x=series.index, y=series, name=column, line=dict(color=color, width=2)),
                    row=5, col=1
                )

    if not benchmark_trace_added:
        fig.add_annotation(
            x=0.5, y=0.5, xref="x5 domain", yref="y5 domain",
            text="Benchmark data is unavailable", showarrow=False,
            font=dict(color="gray", size=12)
        )
    fig.add_hline(y=100, line_width=1, line_dash="dot", line_color="gray", row=5, col=1)

    # Row 6: equity
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Equity"], name="Strategy Equity", line=dict(color="green", width=2),
            hovertemplate="Strategy Equity<br>Date: %{x|%Y-%m-%d}<br>Equity: $%{y:,.2f}<extra></extra>",
        ), row=6, col=1
    )

    # Row 7: drawdown
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Drawdown"] * 100, name="Drawdown %", fill="tozeroy",
            line=dict(color="red", width=1),
            hovertemplate="Drawdown<br>Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
        ), row=7, col=1
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Quarterly EPS", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="TTM EPS", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text=fundamentals_axis_title, row=3, col=1)
    fig.update_yaxes(title_text="P/E (x)", row=4, col=1)
    fig.update_yaxes(title_text="Indexed", row=5, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=6, col=1)
    fig.update_yaxes(title_text="DD (%)", row=7, col=1)

    fig.update_xaxes(
        range=[df.index.min(), df.index.max()],
        autorange=False,
        rangeslider_visible=False,
    )

    chart_state_text = "|".join(
        [
            ticker, start_date_str, end_date_str, timeframe,
            fast_type, str(fast_period), slow_type, str(slow_period),
            buy_left, buy_operator, buy_right,
            sell_left, sell_operator, sell_right, str(initial_capital),
        ]
    )
    chart_revision = hashlib.sha256(chart_state_text.encode("utf-8")).hexdigest()[:16]

    fig.update_layout(
        height=1725,
        margin=dict(l=20, r=70, t=50, b=20),
        showlegend=True,
        hovermode="x unified",
        barmode="group",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"backtest_chart_{chart_revision}",
        config={"responsive": True, "displaylogo": False},
    )
