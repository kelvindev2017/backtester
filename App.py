import hashlib
from datetime import date, timedelta
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from analysis.engine import AnalysisSettings, analyze, report_to_dict, report_to_markdown
from analysis.indicators import resample_ohlcv
from analysis.notify import send_webhook, short_summary
from analysis.patterns import (
    MA_RULE_OPERATORS,
    MA_RULE_SERIES,
    detect_signals,
    ma_rule_conditions,
    moving_average,
    rule_condition,
)
from analysis.risk import RiskSettings
from analysis.screener import (
    LINE_KINDS as SCREEN_LINE_KINDS,
    OPERATORS as SCREEN_OPERATORS,
    TRADING_DAYS_PER_YEAR,
    breadth_history,
    condition_label,
    fundamental_trend,
    index_trend,
    leaders_health,
    market_gate,
    screen_stocks,
    snapshot_table,
    stock_snapshot,
)
from analysis.scoring import COMPONENT_WEIGHTS
from analysis.supply_chain import (
    CHAINS as SUPPLY_CHAINS,
    COMPANIES as SUPPLY_COMPANIES,
    LEVEL_SHORT_NAMES as SUPPLY_LEVEL_SHORT_NAMES,
    LEVELS as SUPPLY_LEVELS,
    NODE_METRICS as SUPPLY_NODE_METRICS,
    chain_tickers,
    linked_tickers,
    neighbors,
    node_positions,
    node_table,
)
from analysis.signal_groups import (
    ANALYST_RATINGS,
    CATEGORY_OPERATORS,
    DEFAULT_INDUSTRIES,
    FUNDAMENTAL_COLUMNS,
    GICS_SECTORS,
    INDUSTRY_INDEXES,
    MOVING_AVERAGE,
    NUMBER_OPERATORS,
    SIGNAL_GROUPS,
    condition_column,
    fundamental_row,
    overlay_frame,
    period_start,
)
from analysis.valuation_chart import (
    VALUATION_HISTORY_YEARS,
    correlation_text,
    eps_estimate_quarters,
    growth_price_correlation,
    pe_summary,
    pe_summary_text,
    peg_summary_text,
    price_yoy,
    quarterly_eps,
    revenue_estimate_quarters,
    revenue_pe_figure,
    trailing_peg,
    yoy_change,
)
from analysis.universe import (
    MARKETS as SCREENER_MARKETS,
    load_us_index_members,
    normalize_symbol,
    parse_ticker_list,
)
from analysis.validation import ValidationSettings, base_rates, exit_signal_stats, summarize, verdict, walk_forward


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Trade Strategy Backtest",
    layout="wide",
    initial_sidebar_state="auto",
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

compute_ma = moving_average
build_condition = rule_condition


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


@st.cache_data(ttl=86400, show_spinner=False)
def load_sec_ticker_map():
    """SEC ticker-to-CIK map (about 1 MB), downloaded once a day."""

    return sec_request_json("https://www.sec.gov/files/company_tickers.json")


def get_sec_cik(symbol):
    """Resolve a US-listed ticker to its SEC CIK."""

    ticker_map = load_sec_ticker_map()

    requested_symbol = symbol.upper().replace(".", "-")

    for item in ticker_map.values():
        sec_symbol = str(item.get("ticker", "")).upper()
        if sec_symbol == requested_symbol:
            return int(item["cik_str"])

    return None


def get_sec_fact_entries(company_facts, tag):
    """Return the USD fact entries reported under one SEC tag."""

    us_gaap_facts = company_facts.get("facts", {}).get("us-gaap", {})
    return us_gaap_facts.get(tag, {}).get("units", {}).get("USD", [])


def combine_sec_tags(company_facts, candidate_tags, extract):
    """Quarterly series merged across tags, earlier tags taking priority.

    Companies switch tags over time (NVIDIA reported revenue under
    RevenueFromContractWithCustomerExcludingAssessedTax until 2022, then
    Revenues), so taking only the first tag with data drops whole years.
    """

    combined = pd.Series(dtype=float)
    for tag in candidate_tags:
        series = extract(get_sec_fact_entries(company_facts, tag))
        if series.empty:
            continue
        # Treat quarter ends a few days apart as the same quarter.
        new_dates = [
            timestamp for timestamp in series.index
            if combined.empty or np.abs(combined.index - timestamp).min() > pd.Timedelta(days=14)
        ]
        combined = pd.concat([combined, series.loc[new_dates]]).sort_index()
    return combined


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
    """Fiscal-quarter values from SEC duration facts, merged across ``candidate_tags``."""

    return combine_sec_tags(company_facts, candidate_tags, fiscal_quarters_from_entries)


def fiscal_quarters_from_entries(entries):
    """
    Build fiscal-quarter values from SEC duration facts.

    Q1-Q3 use standalone approximately three-month facts whenever available.
    Q4 is derived from the full fiscal-year value less Q1, Q2 and Q3. This
    avoids relying on CYyyyyQn frame labels, which omit many fiscal quarters.
    """

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
    """Fiscal quarters from cumulative SEC facts, merged across ``candidate_tags``."""

    return combine_sec_tags(company_facts, candidate_tags, cumulative_quarters_from_entries)


def cumulative_quarters_from_entries(entries):
    """
    Convert cumulative SEC cash-flow facts into individual fiscal quarters.

    This is used for CapEx because 10-Q cash-flow statements commonly report
    year-to-date values rather than standalone Q2 and Q3 amounts.
    """

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
                # Banks report net revenue (after interest expense) here.
                "RevenuesNetOfInterestExpense",
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
    cache_version="2026-10-12-v6",
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

            # Convert the Yahoo date into a timezone-naive Timestamp.
            yahoo_timestamp = pd.Timestamp(yahoo_date)
            
            if yahoo_timestamp.tzinfo is not None:
                yahoo_timestamp = yahoo_timestamp.tz_localize(None)
            
            # Ensure the SEC fundamentals index contains timezone-naive timestamps.
            fundamentals.index = pd.DatetimeIndex(
                pd.to_datetime(
                    fundamentals.index,
                    errors="coerce"
                )
            )
            
            # Remove invalid dates.
            fundamentals = fundamentals.loc[
                ~fundamentals.index.isna()
            ]
            
            if fundamentals.index.tz is not None:
                fundamentals.index = (
                    fundamentals.index.tz_localize(None)
                )
            
            # Calculate distance from every SEC quarter date to the Yahoo date.
            # The index of this Series must contain the actual quarter dates.
            date_distances = pd.Series(
                np.abs(
                    fundamentals.index
                    - yahoo_timestamp
                ),
                index=fundamentals.index
            )
            
            # idxmin now returns the actual nearest Timestamp.
            nearest_date = pd.Timestamp(
                date_distances.idxmin()
            )
            
            nearest_distance_days = abs(
                (
                    nearest_date
                    - yahoo_timestamp
                ).days
            )
            
            # Match Yahoo and SEC records when their quarter-end dates
            # differ by no more than 14 days.
            if nearest_distance_days <= 14:
                for column in yahoo_data.columns:
                    yahoo_value = yahoo_row.get(column)
            
                    if pd.notna(yahoo_value):
                        fundamentals.loc[
                            nearest_date,
                            column
                        ] = yahoo_value
            
            else:
                # No nearby SEC quarter exists, so add the Yahoo row.
                fundamentals.loc[
                    yahoo_timestamp
                ] = yahoo_row

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

def align_fundamentals_to_earnings_dates(
    fundamentals_df,
    earnings_df,
    maximum_days=120,
):
    """
    Align fiscal quarter-end fundamentals with earnings announcement dates.

    Fundamentals from SEC and Yahoo statements normally use fiscal
    quarter-end dates, while the EPS plot uses earnings announcement
    dates. This function maps each fundamentals quarter to the first
    earnings announcement occurring after that quarter-end date.

    If no suitable earnings announcement is found, the original
    fiscal quarter-end date is retained.
    """

    if fundamentals_df is None or fundamentals_df.empty:
        return pd.DataFrame()

    aligned_fundamentals = (
        fundamentals_df
        .copy()
        .sort_index()
    )

    aligned_fundamentals.index = pd.DatetimeIndex(
        pd.to_datetime(
            aligned_fundamentals.index,
            errors="coerce",
        )
    )

    aligned_fundamentals = aligned_fundamentals.loc[
        ~aligned_fundamentals.index.isna()
    ]

    if aligned_fundamentals.index.tz is not None:
        aligned_fundamentals.index = (
            aligned_fundamentals.index.tz_localize(None)
        )

    # Keep the original fiscal quarter-end date for hover information.
    aligned_fundamentals["Fiscal_Quarter_End"] = (
        aligned_fundamentals.index
    )

    if earnings_df is None or earnings_df.empty:
        aligned_fundamentals["Display_Date"] = (
            aligned_fundamentals.index
        )

        return aligned_fundamentals.set_index(
            "Display_Date"
        )

    earnings_dates = pd.DatetimeIndex(
        pd.to_datetime(
            earnings_df.index,
            errors="coerce",
        )
    )

    earnings_dates = earnings_dates[
        ~earnings_dates.isna()
    ]

    if earnings_dates.tz is not None:
        earnings_dates = (
            earnings_dates.tz_localize(None)
        )

    earnings_dates = (
        earnings_dates
        .sort_values()
        .unique()
    )

    display_dates = []

    for fiscal_quarter_end in aligned_fundamentals.index:
        # An earnings announcement should normally occur after
        # the corresponding fiscal quarter has ended.
        later_earnings_dates = earnings_dates[
            earnings_dates >= fiscal_quarter_end
        ]

        if len(later_earnings_dates) == 0:
            display_dates.append(
                fiscal_quarter_end
            )
            continue

        matching_earnings_date = pd.Timestamp(
            later_earnings_dates[0]
        )

        date_difference_days = (
            matching_earnings_date
            - fiscal_quarter_end
        ).days

        if (
            0
            <= date_difference_days
            <= maximum_days
        ):
            display_dates.append(
                matching_earnings_date
            )
        else:
            display_dates.append(
                fiscal_quarter_end
            )

    aligned_fundamentals["Display_Date"] = (
        pd.DatetimeIndex(display_dates)
    )

    aligned_fundamentals = (
        aligned_fundamentals
        .set_index("Display_Date")
        .sort_index()
    )

    return aligned_fundamentals


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
# Signal analysis: data loaders
# ============================================================

def get_secret(name, default=""):
    """Read a Streamlit secret or environment variable without raising."""

    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return value or os.getenv(name, default)


@st.cache_data(ttl=300, show_spinner=False)
def load_intraday_data(symbol, interval="5m", period="10d"):
    """Download regular-session intraday bars (Yahoo keeps ~60 days of 5m data).

    The bar that is still forming is dropped, and timestamps are kept in the
    exchange's local time so that 09:30 is the open for US listings.
    """

    try:
        bars = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            prepost=False,
            progress=False,
            threads=False,
        )
        if bars is None or bars.empty:
            return pd.DataFrame()

        bars = bars.copy()
        if isinstance(bars.columns, pd.MultiIndex):
            final_level = bars.columns.get_level_values(-1)
            if symbol in final_level:
                bars = bars.xs(symbol, axis=1, level=-1)
            else:
                bars.columns = bars.columns.get_level_values(0)

        bars.index = pd.to_datetime(bars.index)
        if bars.index.tz is not None:
            bar_length = pd.Timedelta(interval.replace("m", "min"))
            now = pd.Timestamp.now(tz=bars.index.tz)
            bars = bars.loc[bars.index + bar_length <= now]
            bars.index = bars.index.tz_localize(None)

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        if any(column not in bars.columns for column in required_columns):
            return pd.DataFrame()
        bars = bars[required_columns].dropna(subset=["Close"]).sort_index()
        return bars.loc[~bars.index.duplicated(keep="last")]

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamental_snapshot(symbol):
    """Yahoo valuation fields; raises when Yahoo returns nothing so failures are not cached."""

    keys = [
        "trailingPE", "forwardPE", "pegRatio", "trailingPegRatio", "earningsGrowth",
        "revenueGrowth", "profitMargins", "priceToBook", "debtToEquity", "sector", "industry", "shortName",
    ]
    info = yf.Ticker(symbol).info or {}
    snapshot = {key: info.get(key) for key in keys if info.get(key) is not None}
    if not snapshot:
        raise ValueError(f"Yahoo returned no valuation fields for {symbol}")
    return snapshot


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_revenue_estimates(symbol):
    """Yahoo's analyst revenue estimates (rows 0q, +1q, 0y, +1y); raises when empty so failures are not cached."""

    estimates = yf.Ticker(symbol).revenue_estimate
    if estimates is None or estimates.empty:
        raise ValueError(f"Yahoo returned no revenue estimates for {symbol}")
    return estimates


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_earnings_estimates(symbol):
    """Yahoo's analyst EPS estimates (rows 0q, +1q, 0y, +1y; columns avg, growth ...); raises when empty."""

    estimates = yf.Ticker(symbol).earnings_estimate
    if estimates is None or estimates.empty:
        raise ValueError(f"Yahoo returned no earnings estimates for {symbol}")
    return estimates


def load_earnings_estimates(symbol):
    """Analyst EPS estimates from Yahoo, or an empty frame when unavailable."""

    try:
        return fetch_earnings_estimates(symbol)
    except Exception:
        return pd.DataFrame()


def next_year_eps_growth(symbol):
    """Analysts' expected EPS growth for next fiscal year (0.70 = +70%), or None when unavailable."""

    try:
        value = float(fetch_earnings_estimates(symbol).at["+1y", "growth"])
    except Exception:
        return None
    return value if np.isfinite(value) else None


def load_revenue_estimates(symbol):
    """Analyst revenue estimates from Yahoo, or an empty frame when unavailable."""

    try:
        return fetch_revenue_estimates(symbol)
    except Exception:
        return pd.DataFrame()


def load_fundamental_snapshot(symbol):
    """Current valuation fields from Yahoo (forward P/E, PEG, growth ...), or {} when unavailable."""

    try:
        return fetch_fundamental_snapshot(symbol)
    except Exception:
        return {}


def fundamentals_by_release_date(fundamentals_df, earnings_df, fallback_lag_days=45):
    """Re-index quarterly fundamentals by the date they became public.

    Quarters are matched to their earnings announcement; a quarter with no
    matching announcement is assumed public ``fallback_lag_days`` after its
    fiscal quarter end. This keeps historical analysis free of look-ahead.
    """

    if fundamentals_df is None or fundamentals_df.empty:
        return pd.DataFrame()
    aligned = align_fundamentals_to_earnings_dates(fundamentals_df, earnings_df, maximum_days=120)
    unmatched = aligned.index == pd.DatetimeIndex(aligned["Fiscal_Quarter_End"])
    release_dates = aligned.index.where(
        ~unmatched, aligned.index + pd.Timedelta(days=fallback_lag_days)
    )
    aligned.index = pd.DatetimeIndex(release_dates)
    aligned = aligned.drop(columns=["Fiscal_Quarter_End"]).sort_index()
    return aligned.loc[~aligned.index.duplicated(keep="last")]


def drop_in_progress_daily_bar(daily):
    """Exclude today's daily bar while the US session is still open."""

    if daily.empty:
        return daily, False
    now_new_york = pd.Timestamp.now(tz="America/New_York")
    today = now_new_york.normalize().tz_localize(None)
    if daily.index[-1].normalize() == today and now_new_york.hour < 16:
        return daily.iloc[:-1], True
    return daily, False


def resample_weekly(daily):
    """Weekly bars ending Friday."""

    return resample_ohlcv(daily, "W-FRI")


# ============================================================
# Signal analysis: charts and report rendering
# ============================================================

def build_signal_chart(daily, report, bars_to_show=180):
    """Daily candles with S/R zones, breakout markers and the trade plan."""

    view = daily.iloc[-bars_to_show:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(
        go.Candlestick(
            x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
            name="Daily", increasing_line_color="seagreen", decreasing_line_color="indianred",
        ),
        row=1, col=1,
    )

    visible_low = view["Low"].min() * 0.9
    visible_high = view["High"].max() * 1.1
    for zones, color, label in [(report.supports, "green", "Support"), (report.resistances, "red", "Resistance")]:
        for zone in zones[:4]:
            if zone.high < visible_low or zone.low > visible_high:
                continue
            fig.add_hrect(
                y0=zone.low, y1=zone.high, fillcolor=color, opacity=0.12, line_width=0,
                annotation_text=f"{label} {zone.low:,.2f}-{zone.high:,.2f}", annotation_position="top left",
                annotation_font_size=10, row=1, col=1,
            )

    signals = report.signals.reindex(view.index) if report.signals is not None else None
    if signals is not None:
        for column, symbol, color, name in [
            ("Signal_Up", "triangle-up", "lime", "Bullish signal (vol. confirmed)"),
            ("Signal_Down", "triangle-down", "red", "Bearish signal (vol. confirmed)"),
        ]:
            mask = signals[column].fillna(False).astype(bool) & signals["Volume_Confirmed"].fillna(False).astype(bool)
            points = view.loc[mask]
            if not points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=points.index, y=points["Close"], mode="markers", name=name,
                        marker=dict(symbol=symbol, size=12, color=color, line=dict(color="black", width=1)),
                    ),
                    row=1, col=1,
                )

    if report.signal:
        fig.add_hline(
            y=report.signal["trigger"], line_dash="dot", line_color="orange",
            annotation_text=f"Trigger {report.signal['trigger']:,.2f}", row=1, col=1,
        )
    if report.plan is not None:
        for value, color, label in [
            (report.plan.entry, "royalblue", "Entry"),
            (report.plan.stop, "red", "Stop"),
            (report.plan.target, "green", "Target"),
        ]:
            fig.add_hline(
                y=value, line_dash="dash", line_color=color,
                annotation_text=f"{label} {value:,.2f}", annotation_position="bottom right", row=1, col=1,
            )

    volume_colors = np.where(view["Close"] >= view["Open"], "seagreen", "indianred")
    fig.add_trace(go.Bar(x=view.index, y=view["Volume"], name="Volume", marker_color=volume_colors, opacity=0.7), row=2, col=1)
    average_volume = daily["Volume"].rolling(50).mean().reindex(view.index)
    fig.add_trace(go.Scatter(x=view.index, y=average_volume, name="50-bar avg volume", line=dict(color="gray", width=1)), row=2, col=1)

    fig.update_xaxes(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_layout(height=680, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified", showlegend=True)
    return fig


def build_intraday_chart(intraday, trigger_level):
    """5-minute candles for the evaluated session with VWAP and opening range."""

    session = intraday.session_bars
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=session.index, open=session["Open"], high=session["High"], low=session["Low"], close=session["Close"],
            name="5-min", increasing_line_color="seagreen", decreasing_line_color="indianred",
        )
    )
    if "VWAP" in session.columns:
        fig.add_trace(go.Scatter(x=session.index, y=session["VWAP"], name="VWAP", line=dict(color="purple", width=1.5)))
    for value, color, label in [
        (intraday.opening_range_high, "gray", "OR high"),
        (intraday.opening_range_low, "gray", "OR low"),
        (trigger_level, "orange", "Daily trigger"),
    ]:
        if value is not None:
            fig.add_hline(y=value, line_dash="dot", line_color=color, annotation_text=f"{label} {value:,.2f}")
    if intraday.entry_time is not None:
        fig.add_trace(
            go.Scatter(
                x=[intraday.entry_time], y=[intraday.entry_price], mode="markers", name="Entry trigger",
                marker=dict(symbol="star", size=16, color="gold", line=dict(color="black", width=1)),
            )
        )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
    return fig


def format_metric_value(value):
    if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
        return f"{value:,.3f}" if abs(value) < 10 else f"{value:,.1f}"
    return str(value)


def render_signal_analysis(ticker, end_date, signal_settings, risk_settings):
    """Pattern/trend signal, intraday entry, fundamentals, sizing and report."""

    end_timestamp = pd.Timestamp(end_date).normalize()
    end_str = end_timestamp.strftime("%Y-%m-%d")
    analysis_start = (end_timestamp - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    fundamentals_start = (end_timestamp - pd.DateOffset(years=7)).strftime("%Y-%m-%d")
    is_live = end_timestamp >= pd.Timestamp(date.today()) - pd.Timedelta(days=5)

    st.info(
        "**Validation note:** replayed on 52 large US stocks over 2016-2026, the breakout, pullback and reversal "
        "setups did not beat random entry days with the same stops and targets, and the confidence score did not "
        "rank trades better than chance. The moving-average rule was not part of that study. Treat signals as a structured checklist, and use the **Validate** tab on "
        "your own tickers before acting on them."
    )
    st.caption(
        "Daily/weekly breakout and trend analysis with volume confirmation, a 5-min/15-min entry "
        "trigger, fundamental confirmation and risk-based sizing. The analysis date is the end of the "
        "date range in the settings; it always uses up to 5 years of daily history."
    )

    with st.spinner(f"Running signal analysis for {ticker} as of {end_str}..."):
        daily = load_data(ticker, analysis_start, end_str, "Daily")
        earnings = load_earnings_data(ticker, fundamentals_start, end_str)
        fundamentals, _, fundamentals_source = load_quarterly_fundamentals(ticker, fundamentals_start, end_str)
        snapshot = load_fundamental_snapshot(ticker) if is_live else None
        bars_5m = load_intraday_data(ticker) if is_live else pd.DataFrame()

    if daily.empty:
        st.error(f"No daily data was returned for {ticker}.")
        return

    dropped_today = False
    if is_live:
        daily, dropped_today = drop_in_progress_daily_bar(daily)
    else:
        st.info(
            "Historical analysis date: intraday entry and the live valuation snapshot (forward P/E, Yahoo PEG) "
            "are skipped to avoid look-ahead bias. Fundamentals use only data reported up to that date."
        )
    if dropped_today:
        st.caption("Today's daily bar is still forming and was excluded - daily signals use completed bars only.")

    valuation_df, earnings_calculated = build_valuation_data(daily, earnings)
    fundamentals = fundamentals_by_release_date(fundamentals, earnings_calculated)
    report = analyze(
        ticker,
        daily,
        resample_weekly(daily),
        bars_5m=bars_5m,
        valuation_df=valuation_df,
        fundamentals_df=fundamentals,
        snapshot=snapshot,
        settings=signal_settings,
        risk_settings=risk_settings,
    )

    # Headline -------------------------------------------------------------
    if report.action in ("BUY", "SELL"):
        st.success(f"**{report.headline}**")
    elif report.action.endswith("SETUP") or report.action in ("WATCH", "EXIT"):
        st.warning(f"**{report.headline}**")
    elif report.action == "NO TRADE":
        st.error(f"**{report.headline}**")
    else:
        st.info(f"**{report.headline}**")

    trend_lookup = {trend.timeframe: trend for trend in report.trends}
    metric_columns = st.columns(6)
    metric_columns[0].metric("Confidence", "n/a" if report.confidence is None else f"{report.confidence:.0f}/100")
    for column, timeframe in zip(metric_columns[1:3], ["Daily", "Weekly"]):
        trend = trend_lookup.get(timeframe)
        if trend is not None and np.isfinite(trend.score):
            column.metric(f"{timeframe} trend", trend.label, f"{trend.score:+.2f}", delta_color="off")
        else:
            column.metric(f"{timeframe} trend", "n/a")
    if report.plan is not None:
        metric_columns[3].metric("Reward / risk", f"{report.plan.reward_risk:.2f}")
        metric_columns[4].metric(
            "Shares",
            f"{report.plan.shares:,}",
            f"{report.plan.size_multiplier:g}x size" if report.plan.shares > 0 else None,
            delta_color="off",
        )
        metric_columns[5].metric("Dollar risk", f"${report.plan.dollar_risk:,.0f}")

    for warning in report.warnings:
        if warning != report.headline:
            st.warning(warning)

    # Trade plan and reasoning ---------------------------------------------
    if report.plan is not None:
        plan = report.plan
        plan_column, reasons_column = st.columns([1, 1.3])
        exit_only = plan.direction < 0 and not signal_settings.allow_short
        with plan_column:
            st.subheader("Exit guidance" if exit_only else "Trade plan")
        if exit_only:
            invalidation = report.signal["invalidation"] if report.signal else plan.stop
            plan_column.info(
                f"Bearish signal in long-only mode: no new position. If you hold {ticker}, consider closing or "
                f"reducing it, or raise its stop to about {plan.stop:,.2f}. The signal is invalidated if price "
                f"closes back above {invalidation:,.2f}. The next support zone is near {plan.target:,.2f}."
            )
        else:
            with plan_column:
                st.table(
                    pd.DataFrame(
                        {
                            "Value": [
                                plan.side, f"{plan.entry:,.2f}", f"{plan.stop:,.2f}", f"{plan.target:,.2f}",
                                f"{plan.risk_per_share:,.2f}", f"{plan.reward_risk:.2f}", f"{plan.shares:,}",
                                f"${plan.position_value:,.0f}", f"${plan.dollar_risk:,.0f}",
                            ],
                            "Basis": [
                                "", plan.notes[0] if plan.notes else "", plan.stop_basis, plan.target_basis,
                                "", "", f"{risk_settings.risk_pct:g}% risk x {plan.size_multiplier:g} (confidence tier)",
                                f"cap {risk_settings.max_position_pct:g}% of equity", "",
                            ],
                        },
                        index=["Side", "Entry", "Stop", "Target", "Risk / share", "Reward / risk", "Shares", "Position value", "Dollar risk"],
                    )
                )
                for note in plan.notes[1:]:
                    st.caption(note)
        with reasons_column:
            st.subheader("Reasoning")
            st.markdown("**Supporting evidence**")
            st.markdown("\n".join(f"- {reason}" for reason in report.reasons_for) or "- None")
            st.markdown("**Risks and conflicts**")
            risks = report.reasons_against + [warning for warning in report.warnings if warning != report.headline]
            st.markdown("\n".join(f"- {reason}" for reason in risks) or "- None identified")

    show_chart(build_signal_chart(daily, report), width="stretch", config={"displaylogo": False})

    if report.components:
        st.subheader("Confidence breakdown")
        st.caption(
            "Each component scores from -1 (argues against the trade) to +1 (supports it). "
            "Confidence = 50 + 50 x weighted average; unavailable components are left out and the weights re-normalised. "
            "The weights are a starting point and have not been validated against outcomes yet - treat the score as a "
            "structured checklist, not a probability."
        )
        for component in report.components:
            weight = COMPONENT_WEIGHTS.get(component.name, 0) * 100
            score_text = f"{component.score:+.2f}" if component.available else "n/a"
            with st.expander(f"{component.name} - weight {weight:.0f}% - score {score_text}"):
                st.markdown("\n".join(f"- {detail}" for detail in component.details))
                if component.metrics:
                    st.dataframe(
                        pd.DataFrame(
                            {"Value": [format_metric_value(value) for value in component.metrics.values()]},
                            index=list(component.metrics.keys()),
                        ),
                        width="stretch",
                    )

    if report.intraday is not None and report.intraday.session_bars is not None and not report.intraday.session_bars.empty:
        st.subheader(f"Intraday entry - session {report.intraday.session_date}")
        st.caption(f"Status: {report.intraday.status}. 15-minute bars are resampled from the 5-minute data.")
        show_chart(
            build_intraday_chart(report.intraday, report.signal["trigger"]),
            width="stretch",
            config={"displaylogo": False},
        )

    if report.history_summary is not None and not report.history_summary.empty:
        st.subheader("Historical edge of this signal on this ticker")
        st.caption("Past volume-confirmed signals in the same direction, entering at the next bar's open.")
        summary = report.history_summary.copy()
        for column in ["Win rate", "Average return", "Median return", "Base win rate (any day)", "Base average return"]:
            summary[column] = summary[column].map(lambda value: "n/a" if pd.isna(value) else f"{value * 100:.1f}%")
        st.dataframe(summary, hide_index=True, width="stretch")
        if report.history_events is not None and not report.history_events.empty:
            with st.expander(f"{len(report.history_events)} past signals"):
                st.dataframe(report.history_events, hide_index=True, width="stretch")

    if fundamentals is not None and not fundamentals.empty:
        st.caption(f"Quarterly fundamentals source: {fundamentals_source}.")

    # Report export and notification ----------------------------------------
    st.subheader("Pre-trade report")
    markdown_report = report_to_markdown(report)
    export_columns = st.columns(3)
    export_columns[0].download_button(
        "Download report (Markdown)", markdown_report,
        file_name=f"{ticker}_{end_str}_signal_report.md", mime="text/markdown",
    )
    export_columns[1].download_button(
        "Download report (JSON)", json.dumps(report_to_dict(report), indent=2, default=str),
        file_name=f"{ticker}_{end_str}_signal_report.json", mime="application/json",
    )
    webhook_url = get_secret("ALERT_WEBHOOK_URL")
    if webhook_url:
        if export_columns[2].button("Send notification"):
            ok, message = send_webhook(webhook_url, short_summary(report))
            (st.success if ok else st.error)(f"Notification {'sent' if ok else 'failed'}: {message}")
    else:
        export_columns[2].caption("Add `ALERT_WEBHOOK_URL` (Slack/Discord webhook) to Streamlit secrets to send notifications.")
    with st.expander("Report preview"):
        st.markdown(markdown_report)


# ============================================================
# Validation: walk-forward replay of the signal engine
# ============================================================

DEFAULT_VALIDATION_TICKERS = "AAPL, MSFT, NVDA, AMZN, META, GOOGL, JPM, XOM, KO, COST, UNH, CAT"


@st.cache_data(ttl=21600, show_spinner=False)
def run_ticker_validation(ticker, start_str, end_str, settings_dict, risk_dict, max_hold, include_fundamentals):
    """Walk-forward replay for one ticker. Cached, so re-runs are instant."""

    data_start = (pd.Timestamp(start_str) - pd.DateOffset(years=6)).strftime("%Y-%m-%d")
    data_end = pd.Timestamp(date.today()).strftime("%Y-%m-%d")
    daily = load_data(ticker, data_start, data_end, "Daily")
    if daily.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    daily, _ = drop_in_progress_daily_bar(daily)

    valuation_df, fundamentals_df = None, None
    if include_fundamentals:
        earnings = load_earnings_data(ticker, data_start, data_end)
        valuation_df, earnings_calculated = build_valuation_data(daily, earnings)
        fundamentals, _, _ = load_quarterly_fundamentals(ticker, data_start, data_end)
        fundamentals_df = fundamentals_by_release_date(fundamentals, earnings_calculated)

    trades = walk_forward(
        ticker,
        daily,
        valuation_df,
        fundamentals_df,
        AnalysisSettings(**settings_dict),
        RiskSettings(**risk_dict),
        start_str,
        end_str,
        ValidationSettings(max_hold=int(max_hold)),
    )
    return trades, base_rates(daily, start_str, end_str, horizon=10)


def format_stats_table(frame):
    """Format R, percentage and count columns for display."""

    formatted = frame.copy()
    percent_columns = ["Win rate", "Average return", "Target hit", "Stopped out", "Time exit"]
    for column in formatted.columns:
        if column in percent_columns:
            formatted[column] = formatted[column].map(lambda v: "" if pd.isna(v) else f"{v * 100:.1f}%")
        elif column in ("Average R", "Median R", "Random-entry avg R", "Edge vs random (R)"):
            formatted[column] = formatted[column].map(lambda v: "" if pd.isna(v) else f"{v:+.2f}R")
        elif column in ("t-stat (avg R)", "t-stat (edge)", "Profit factor", "Avg bars held", "Rank correlation with R", "95% noise band"):
            formatted[column] = formatted[column].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    for column in formatted.columns:
        if formatted[column].dtype.name == "category" or column == "Confidence bucket":
            formatted[column] = formatted[column].astype(str)
    return formatted


def render_validation(signal_settings, risk_settings, default_end):
    """Validate tab: does the engine make money, and does confidence mean anything?"""

    st.caption(
        "Replays the Signal Analysis engine on every past day the selected setup fired, using only data available "
        "on that day, then simulates the trade plan: entry at the next open, exit at the stop or target (stop assumed "
        "first if both are hit in one bar; gaps fill at the open) or after the maximum holding period. Results are "
        "in R - profit divided by the amount risked. Setup, volume, risk and confidence settings come from the "
        "settings. The intraday trigger cannot be replayed (Yahoo keeps ~60 days of 5-minute data), so taken trades "
        "enter at the next open."
    )

    with settings_panel("Validation"):
        tickers_text = st.text_input("Tickers (comma-separated)", DEFAULT_VALIDATION_TICKERS, key="validation_tickers")
        date_columns = st.columns(2)
        test_start = date_columns[0].date_input(
            "Test from", value=(pd.Timestamp(default_end) - pd.DateOffset(years=2)).date(), max_value=date.today(), key="validation_start"
        )
        test_end = date_columns[1].date_input("Test to", value=default_end, max_value=date.today(), key="validation_end")
        max_hold = int(st.number_input("Max holding period (bars)", min_value=3, max_value=120, value=20, step=1,
                                       key="validation_max_hold"))
        include_fundamentals = st.checkbox(
            "Include fundamentals (slower: loads SEC and earnings history for each ticker)", value=True,
            key="validation_fundamentals",
        )
        run_clicked = st.button("Run validation", type="primary", key="validation_run_btn")

    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers_text.replace(";", ",").split(",") if t.strip()))
    settings_dict = dict(signal_settings.__dict__)
    risk_dict = dict(risk_settings.__dict__)
    start_str = pd.Timestamp(test_start).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(test_end).strftime("%Y-%m-%d")
    run_key = json.dumps([tickers, start_str, end_str, settings_dict, risk_dict, max_hold, include_fundamentals], default=str)

    if run_clicked and tickers and test_start < test_end:
        progress = st.progress(0.0)
        frames, bases, failed = [], [], []
        for index, ticker in enumerate(tickers):
            progress.progress(index / len(tickers), text=f"Replaying {ticker} ({index + 1}/{len(tickers)})...")
            try:
                trades, base = run_ticker_validation(
                    ticker, start_str, end_str, settings_dict, risk_dict, max_hold, include_fundamentals
                )
            except Exception as error:
                trades, base = pd.DataFrame(), pd.Series(dtype=float)
                failed.append(f"{ticker} ({type(error).__name__})")
            if trades.empty and base.empty:
                failed.append(ticker)
            if not trades.empty:
                frames.append(trades)
            if not base.empty:
                bases.append(base)
        progress.empty()
        st.session_state["validation_result"] = {
            "key": run_key,
            "trades": pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
            "base": pd.concat(bases) if bases else pd.Series(dtype=float),
            "failed": failed,
            "setup": signal_settings.setup,
        }

    result = st.session_state.get("validation_result")
    if result is None:
        st.info("Choose tickers and a test period, then press **Run validation**. Expect roughly 5-15 seconds per ticker.")
        return
    if result["key"] != run_key:
        st.warning("The settings changed since this run - press **Run validation** to refresh.")
    if result["failed"]:
        st.caption(f"No data for: {', '.join(result['failed'])}")

    trades = result["trades"]
    if trades.empty:
        st.warning("No signals fired for these tickers and dates.")
        return

    summary = summarize(trades)
    taken_stats = summary["taken"]
    message = verdict(taken_stats)
    if message.startswith("Positive"):
        st.success(f"**Trades the engine would take:** {message}")
    elif message.startswith("Negative"):
        st.error(f"**Trades the engine would take:** {message}")
    else:
        st.warning(f"**Trades the engine would take:** {message}")

    metric_columns = st.columns(6)
    metric_columns[0].metric("Signals replayed", f"{len(trades):,}")
    metric_columns[1].metric("Trades taken", f"{taken_stats.get('Trades', 0):,}")
    if taken_stats.get("Trades", 0):
        metric_columns[2].metric("Win rate", f"{taken_stats['Win rate'] * 100:.1f}%")
        metric_columns[3].metric("Average R", f"{taken_stats['Average R']:+.2f}R")
        random_average = taken_stats.get("Random-entry avg R", np.nan)
        metric_columns[4].metric("Random entries", "n/a" if pd.isna(random_average) else f"{random_average:+.2f}R")
        edge, edge_t = taken_stats.get("Edge vs random (R)", np.nan), taken_stats.get("t-stat (edge)", np.nan)
        metric_columns[5].metric(
            "Edge vs random" if pd.isna(edge_t) else f"Edge vs random (t = {edge_t:.1f})",
            "n/a" if pd.isna(edge) else f"{edge:+.2f}R",
        )
    st.caption(
        "**Random entries** copy each signal's stop and target distances (in ATR) onto random days of the same "
        "ticker and period. Positive R that random entries also earn comes from market drift and the exit rules - "
        "only the **edge vs random** shows whether the signal's timing adds value (|t| >= 2 to trust it)."
    )

    base = result["base"]
    if not base.empty:
        st.caption(
            f"Benchmark - buying these tickers on *any* day in the window and holding 10 bars: "
            f"{(base > 0).mean() * 100:.1f}% of holds were profitable, average {base.mean() * 100:+.2f}%. "
            f"Taken trades averaged {taken_stats.get('Average return', np.nan) * 100:+.2f}% per trade."
        )

    # Cumulative R: what following every taken signal (1R risk each) would have produced.
    curve = go.Figure()
    taken_trades = trades.loc[trades["Taken"]].dropna(subset=["R"]).sort_values("Exit_Date")
    for label, column, subset, color, dash in [
        ("Trades taken", "R", taken_trades, "seagreen", "solid"),
        ("Random entries, same stops/targets", "Random_R", taken_trades, "darkorange", "dot"),
        ("All signals (incl. WATCH / NO TRADE / EXIT)", "R", trades.dropna(subset=["R"]).sort_values("Exit_Date"), "gray", "solid"),
    ]:
        if column in subset and not subset.empty:
            curve.add_trace(
                go.Scatter(
                    x=subset["Exit_Date"], y=subset[column].fillna(0).cumsum(), name=label, mode="lines",
                    line=dict(color=color, width=2, dash=dash),
                )
            )
    curve.add_hline(y=0, line_dash="dot", line_color="gray")
    curve.update_layout(
        height=380, margin=dict(l=20, r=20, t=40, b=20), title="Cumulative R (each trade risks 1R)",
        yaxis_title="Cumulative R", hovermode="x unified",
    )
    show_chart(curve, width="stretch", config={"displaylogo": False})

    exits = exit_signal_stats(trades, base_forward=base)
    if exits:
        move = exits["Stock 10-bar return after signal"]
        any_day = exits.get("Any-day 10-bar return", np.nan)
        helped = exits.get("Exit helped vs holding")
        st.subheader("Were the bearish (EXIT) signals worth acting on?")
        (st.success if helped else st.warning)(
            f"After {exits['Bearish signals']:,} bearish signals the stock moved {move * 100:+.2f}% over the next "
            f"10 bars (rose {exits['Share of signals followed by a rise'] * 100:.0f}% of the time), versus "
            f"{any_day * 100:+.2f}% after any day. "
            + ("Exiting on them avoided below-average returns." if helped else
               "Exiting on them gave up above-average returns - treat EXIT signals with caution.")
        )

    st.subheader("Does the confidence score mean anything?")
    st.caption(
        "If confidence works, higher buckets should show a higher average R, and its rank correlation with R should "
        "sit outside the 95% noise band (the range a random score would produce)."
    )
    st.dataframe(format_stats_table(summary["by_confidence"]), hide_index=True, width="stretch")
    st.dataframe(format_stats_table(summary["components"]), hide_index=True, width="stretch")
    with st.expander("Results if only trades at or above a confidence threshold were taken"):
        st.dataframe(format_stats_table(summary["thresholds"]), hide_index=True, width="stretch")

    for title, key in [
        ("By action", "by_action"),
        ("By direction (1 = bullish, -1 = bearish)", "by_direction"),
        ("By year", "by_year"),
        ("By ticker", "by_ticker"),
    ]:
        with st.expander(title):
            st.dataframe(format_stats_table(summary[key]), hide_index=True, width="stretch")

    with st.expander(f"All {len(trades):,} replayed signals"):
        st.dataframe(trades, hide_index=True, width="stretch")
    st.download_button(
        "Download replayed signals (CSV)",
        trades.to_csv(index=False),
        file_name=f"validation_{result['setup']}_{start_str}_{end_str}.csv",
        mime="text/csv",
    )


# ============================================================
# Stock screener: data loaders
# ============================================================

def split_download(data, tickers):
    """Turn a multi-ticker ``yf.download`` frame into {ticker: OHLCV frame}."""

    frames = {}
    if data is None or data.empty:
        return frames
    fields = ["Open", "High", "Low", "Close", "Volume"]
    if not isinstance(data.columns, pd.MultiIndex):
        if len(tickers) == 1:
            frames[tickers[0]] = data
    else:
        level_0 = set(data.columns.get_level_values(0))
        for ticker in tickers:
            try:
                if ticker in level_0:
                    frame = data[ticker]
                else:
                    frame = data.xs(ticker, axis=1, level=1)
            except KeyError:
                continue
            frames[ticker] = frame
    cleaned = {}
    for ticker, frame in frames.items():
        if any(field not in frame.columns for field in fields):
            continue
        frame = frame[fields].copy()
        frame.index = pd.to_datetime(frame.index)
        if getattr(frame.index, "tz", None) is not None:
            frame.index = frame.index.tz_localize(None)
        frame = frame.sort_index()
        frame = frame.loc[~frame.index.duplicated(keep="last")].dropna(subset=["Close"])
        if not frame.empty:
            cleaned[ticker] = frame
    return cleaned


@st.cache_data(ttl=21600, show_spinner=False)
def load_price_chunk(tickers, start_str, end_str):
    """Daily OHLCV for a batch of tickers. Cached for 6 hours."""

    try:
        data = yf.download(
            list(tickers),
            start=start_str,
            end=end_str,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception:
        return {}
    return split_download(data, list(tickers))


def load_price_panel(tickers, chunk_size=50, progress=None, days=1100):
    """Daily OHLCV for many tickers (``days`` calendar days, about 3 years by default), downloaded in batches."""

    end = pd.Timestamp(date.today()) + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days)
    tickers = list(tickers)
    prices = {}
    for first in range(0, len(tickers), chunk_size):
        chunk = tuple(tickers[first:first + chunk_size])
        prices.update(load_price_chunk(chunk, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
        if progress is not None:
            progress(min(1.0, (first + chunk_size) / max(1, len(tickers))))
    return prices


@st.cache_data(ttl=86400, show_spinner=False)
def load_universe_members(market, universe):
    """Members of a screener universe. Cached for a day."""

    try:
        return load_us_index_members(universe), ""
    except Exception as error:
        return pd.DataFrame(columns=["Ticker", "Name", "Sector", "Industry"]), f"{type(error).__name__}: {error}"


# ============================================================
# Stock screener: page
# ============================================================

def editor_conditions(frame):
    """Valid screen conditions from the data-editor rows (incomplete rows are skipped)."""

    conditions = []
    for _, row in frame.iterrows():
        left, operator, right = row.get("Left"), row.get("Operator"), row.get("Right")
        left_period, right_value = row.get("Left period"), row.get("Right period / value")
        if left not in ("Close", "SMA", "EMA") or right not in SCREEN_LINE_KINDS or operator not in SCREEN_OPERATORS:
            continue
        if left != "Close" and (pd.isna(left_period) or int(left_period) < 1):
            continue
        if right != "Close" and pd.isna(right_value):
            continue
        if right in ("SMA", "EMA") and int(right_value) < 1:
            continue
        conditions.append({
            "left_kind": left,
            "left_period": None if left == "Close" else int(left_period),
            "operator": operator,
            "right_kind": right,
            "right_period": None if right == "Close" else (float(right_value) if right == "Value" else int(right_value)),
        })
    return conditions


def price_lines_chart(frame, title, ma_periods=(), bars=TRADING_DAYS_PER_YEAR, height=380):
    """Last ``bars`` closes with SMAs (computed on the full history so they start filled)."""

    view = frame.iloc[-bars:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.index, y=view["Close"], name="Close", line=dict(color="gray", width=1.4)))
    for color, period in zip(["orange", "royalblue", "purple"], ma_periods):
        moving = frame["Close"].rolling(period, min_periods=period).mean().iloc[-bars:]
        fig.add_trace(go.Scatter(x=moving.index, y=moving, name=f"SMA {period}", line=dict(color=color, width=1.2)))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
    return fig


@st.cache_resource(show_spinner=False)
def fundamentals_store():
    """Yahoo quote-summary rows shared across reruns: {ticker: (fetched at, row)}."""

    return {}


def fetch_fundamental_row(ticker):
    """Screener fundamentals for one ticker, or None when Yahoo fails (so the failure is not stored)."""

    try:
        return fundamental_row(yf.Ticker(ticker).info or {})
    except Exception:
        return None


def load_fundamental_table(tickers, progress=None, max_age_hours=12, retry_failed_minutes=15):
    """Fundamentals (market cap, P/E, growth, earnings dates ...) for many tickers, fetched 8 at a time."""

    store = fundamentals_store()
    now = time.time()
    def stale(ticker):
        fetched_at, row = store[ticker]
        return now - fetched_at > (max_age_hours * 3600 if row is not None else retry_failed_minutes * 60)

    missing = [ticker for ticker in tickers if ticker not in store or stale(ticker)]
    if missing:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for done, (ticker, row) in enumerate(zip(missing, pool.map(fetch_fundamental_row, missing)), start=1):
                # Failed lookups are stored too (as None) so every rerun does not refetch them; they retry sooner.
                store[ticker] = (now, row)
                if progress is not None:
                    progress(done / len(missing))
    rows = {ticker: store[ticker][1] for ticker in tickers if ticker in store and store[ticker][1]}
    return pd.DataFrame.from_dict(rows, orient="index")


def index_membership(market, tickers):
    """{ticker: [index names it belongs to]} over the market's built-in universes."""

    membership = {ticker: [] for ticker in tickers}
    for name in SCREENER_MARKETS[market]["universes"]:
        if name == "Custom list":
            continue
        members, _ = load_universe_members(market, name)
        for ticker in set(members["Ticker"]) & set(tickers):
            membership[ticker].append(name)
    return membership


# ============================================================
# Layout: desktop sidebar or mobile menus
# ============================================================

SUPPLY_CHAIN_TAB = "Semicon Supply Chain"
FUNDAMENTALS_TAB = "Stock Fundamentals"
FUNDAMENTALS_YEARS = [3, 5, 8, 10]
TAB_NAMES = ["Stock Screener", "Market Analysis", "Industry Analysis", FUNDAMENTALS_TAB, SUPPLY_CHAIN_TAB, "Backtest", "Signal Analysis & Trade Plan", "Validate"]
SIGNAL_TAB = "Signal Analysis & Trade Plan"
LAYOUT_MODES = ["Auto", "Desktop", "Mobile"]
MOBILE_AGENT = re.compile(r"Mobi|Android|iPhone|iPad|iPod", re.IGNORECASE)
# Session keys that are actions or display state rather than settings; they are not carried across tab switches.
TRANSIENT_KEY_SUFFIXES = ("_btn", "_table", "_chart")
TRANSIENT_KEYS = {"main_tab", "password_correct"}
MAIN_INDEX_SYMBOLS = list(dict.fromkeys(list(SCREENER_MARKETS["US"]["indexes"].values()) + list(INDUSTRY_INDEXES.values())))

# Writing a widget's own value back into Session State is how its setting survives a tab switch (see
# keep_settings_across_tabs); Streamlit logs a warning for every such widget that also has a default.
logging.getLogger("streamlit.elements.lib.policies").setLevel(logging.ERROR)


def keep_settings_across_tabs():
    """Only the open tab runs, and Streamlit forgets the value of any widget that did not run.

    Re-storing every keyed setting at the start of each run keeps it, so a tab's settings are still
    there after a visit to another tab.
    """

    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key not in TRANSIENT_KEYS and not key.endswith(TRANSIENT_KEY_SUFFIXES):
            st.session_state[key] = st.session_state[key]


def detect_mobile():
    """Mobile layout when chosen in Display, else when the browser's user agent is a phone or tablet.

    Streamlit has no device check of its own; the user agent is the signal the server can see.
    """

    mode = st.session_state.get("layout_mode", "Auto")
    if mode != "Auto":
        return mode == "Mobile"
    try:
        agent = st.context.headers.get("User-Agent", "") or ""
    except Exception:
        agent = ""
    return bool(MOBILE_AGENT.search(agent))


MOBILE = False


def settings_panel(title, expanded=True):
    """Where a tab's settings go: the sidebar on desktop, a collapsible menu at the top of the tab on mobile."""

    if MOBILE:
        return st.expander(title, expanded=False)
    return st.sidebar.expander(title, expanded=expanded)


def show_chart(fig, **kwargs):
    """``st.plotly_chart`` that, on mobile, moves the legend under the plot so the plot gets the full width."""

    if MOBILE:
        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0, font=dict(size=10),
                        entrywidth=0.5, entrywidthmode="fraction"),
            margin=dict(l=5, r=5),
        )
    st.plotly_chart(fig, **kwargs)


@st.cache_data(ttl=21600, show_spinner=False)
def load_main_indexes():
    """Daily prices (about 5 years) for the main US and industry indexes, loaded when the app starts."""

    return load_price_panel(MAIN_INDEX_SYMBOLS, days=1900)


# ============================================================
# Market Analysis tab
# ============================================================

MARKET_ANALYSIS_UNIVERSES = ["S&P 500", "Nasdaq-100", "Dow Jones 30"]


def render_market_analysis(index_prices):
    """US index trends, market breadth and the market filter. Stores the filter verdict for the screener."""

    indexes = SCREENER_MARKETS["US"]["indexes"]
    with settings_panel("Breadth data"):
        universe = st.selectbox("Breadth universe", MARKET_ANALYSIS_UNIVERSES, key="market_universe")
        recent_days = int(st.number_input(
            "New-high window (days)", min_value=1, max_value=120, value=20, step=1, key="market_recent_days",
            help="Breadth counts stocks that made a new 52-week high (or low) at least once in this many trading days.",
        ))
        load_clicked = st.button("Load breadth data", type="primary", key="market_load_btn",
                                 help="Downloads every member of the universe; about a minute the first time, then cached.")
    with settings_panel("Moving averages"):
        ma_columns = st.columns(3)
        ma_fast = int(ma_columns[0].number_input("Short", min_value=2, max_value=400, value=20, step=1, key="market_ma_fast"))
        ma_mid = int(ma_columns[1].number_input("Medium", min_value=2, max_value=400, value=50, step=1, key="market_ma_mid"))
        ma_slow = int(ma_columns[2].number_input("Long", min_value=2, max_value=400, value=200, step=1, key="market_ma_slow"))
        chart_index = st.selectbox("Index chart", list(indexes), key="market_index_chart")
    ma_periods = tuple(dict.fromkeys([ma_fast, ma_mid, ma_slow]))
    with settings_panel("Market filter", expanded=False):
        gate_index = st.selectbox("Index", list(indexes), key="market_gate_index")
        use_index_ma = st.checkbox(f"Index close above its SMA {ma_slow}", value=True, key="market_gate_close")
        use_index_order = st.checkbox(f"Index SMA {ma_mid} above SMA {ma_slow}", value=False, key="market_gate_order")
        use_breadth = st.checkbox(f"Minimum % of stocks above SMA {ma_slow}", value=True, key="market_gate_breadth")
        breadth_threshold = st.number_input("% above", 0.0, 100.0, 50.0, 5.0, key="market_gate_breadth_pct", disabled=not use_breadth)
        use_highs = st.checkbox(f"Minimum % of stocks with a new 52w high in the last {recent_days} days", value=False, key="market_gate_highs")
        highs_threshold = st.number_input("% with new highs", 0.0, 100.0, 5.0, 1.0, key="market_gate_highs_pct", disabled=not use_highs)

    st.caption(
        "The US market as a whole: index trends (loaded when the app starts) and the breadth of an index's members "
        "(share above their moving averages, share making new 52-week highs or lows). The market filter can gate "
        "the Stock Screener."
    )
    st.subheader("US indexes")
    index_rows = {}
    for name, symbol in indexes.items():
        if symbol in index_prices:
            row = stock_snapshot(index_prices[symbol], ma_periods=ma_periods, recent_days=recent_days)
            row["Trend"] = index_trend(index_prices[symbol], ma_mid, ma_slow)
            index_rows[name] = row
    if index_rows:
        index_table = pd.DataFrame.from_dict(index_rows, orient="index")
        index_columns = ["Trend", "Close", "Change %"] + [f"Close vs SMA {p} %" for p in ma_periods] + [
            "% from 52w high", "Return 1m %", "Return 3m %", "Return YTD %"]
        st.dataframe(index_table[index_columns].style.format(precision=2), width="stretch")
        if chart_index in index_rows:
            show_chart(price_lines_chart(index_prices[indexes[chart_index]], chart_index, ma_periods), width="stretch")
    else:
        st.warning("No index data was returned. Yahoo Finance may be rate-limiting; try again in a few minutes.")

    members, members_error = load_universe_members("US", universe)
    tickers = members["Ticker"].tolist()
    data_key = json.dumps([universe, tickers])
    loaded = st.session_state.get("market_data")
    if load_clicked and tickers:
        progress = st.progress(0.0, text="Downloading prices...")
        prices = load_price_panel(tickers, progress=lambda value: progress.progress(value, text="Downloading prices..."))
        progress.empty()
        loaded = {"key": data_key, "prices": prices}
        st.session_state["market_data"] = loaded

    breadth_latest = pd.Series(dtype=float)
    st.subheader(f"Market breadth: {universe}")
    if members.empty:
        st.error(f"Could not load the members of {universe}. {members_error}")
    elif loaded is None or loaded["key"] != data_key:
        st.info(f"Press **Load breadth data** in the settings to download the {len(tickers)} members of {universe}.")
    elif not loaded["prices"]:
        st.error("No price data was returned. Yahoo Finance may be rate-limiting; try again in a few minutes.")
    else:
        breadth = breadth_history(loaded["prices"], ma_periods=(ma_mid, ma_slow), recent_days=recent_days)
        if not breadth.empty:
            breadth_latest = breadth.dropna(how="all").iloc[-1]
            st.caption(f"{int(breadth_latest['Stocks'])} stocks with price data.")
            tiles = st.columns(4)
            tiles[0].metric(f"Above SMA {ma_mid}", f"{breadth_latest[f'% above SMA {ma_mid}']:.0f}%")
            tiles[1].metric(f"Above SMA {ma_slow}", f"{breadth_latest[f'% above SMA {ma_slow}']:.0f}%")
            tiles[2].metric(f"New 52w high, last {recent_days}d", f"{breadth_latest[f'% new 52w high in last {recent_days}d']:.1f}%")
            tiles[3].metric(f"New 52w low, last {recent_days}d", f"{breadth_latest[f'% new 52w low in last {recent_days}d']:.1f}%")
            recent_breadth = breadth.iloc[-TRADING_DAYS_PER_YEAR:]
            fig = go.Figure()
            for column, color in [
                (f"% above SMA {ma_mid}", "royalblue"), (f"% above SMA {ma_slow}", "purple"),
                (f"% new 52w high in last {recent_days}d", "seagreen"), (f"% new 52w low in last {recent_days}d", "indianred"),
            ]:
                fig.add_trace(go.Scatter(x=recent_breadth.index, y=recent_breadth[column], name=column, line=dict(color=color, width=1.4)))
            fig.update_layout(title="Market breadth (% of stocks)", height=340, margin=dict(l=10, r=10, t=40, b=10),
                              hovermode="x unified", yaxis=dict(range=[0, 100]))
            show_chart(fig, width="stretch")

    market_rules = {"index": gate_index}
    if use_index_ma:
        market_rules["index_ma"] = ma_slow
    if use_index_order:
        market_rules["index_ma_order"] = (ma_mid, ma_slow)
    if use_breadth:
        market_rules["min_pct_above_ma"] = (ma_slow, float(breadth_threshold))
    if use_highs:
        market_rules["min_pct_new_highs"] = (f"% new 52w high in last {recent_days}d", float(highs_threshold))
    market_ok, market_checks = market_gate(index_rows, breadth_latest, market_rules)
    check_lines = "\n".join(
        f"- {'✅' if ok else ('⚠️ no data:' if ok is None else '❌')} {text}" for text, ok in market_checks
    )
    (st.success if market_ok else st.error)(f"**Market filter: {'PASS' if market_ok else 'FAIL'}**\n\n{check_lines}")
    st.session_state["market_filter_ok"] = market_ok


# ============================================================
# Industry Analysis tab
# ============================================================

INDUSTRY_PERIODS = ["1M", "3M", "6M", "YTD", "1Y", "2Y", "3Y", "5Y"]


def render_industry_analysis(index_prices):
    """Overlay the main index of each chosen industry, plus an optional look at an industry's leading companies."""

    with settings_panel("Industries"):
        chosen = st.multiselect("Industry indexes", list(INDUSTRY_INDEXES), default=DEFAULT_INDUSTRIES, key="industry_indexes")
        extra_text = st.text_input("Other tickers", value="", key="industry_extra",
                                   help="Any Yahoo symbols to add, e.g. ^RUT, SMH, XBI.")
    with settings_panel("Chart"):
        period = st.select_slider("Period", INDUSTRY_PERIODS, value="1Y", key="industry_period")
        normalize = st.toggle("Normalize (start = 100)", value=True, key="industry_normalize")
        log_scale = st.checkbox("Log scale", value=False, key="industry_log")
    with settings_panel("Leading companies", expanded=False):
        leaders_on = st.checkbox("Compare the leading companies of one industry", value=False, key="industry_leaders_on")
        leaders_text = st.text_input("Industry leaders (up to 5 tickers)", value="NVDA, AMD, AVGO, TSM, INTC",
                                     key="industry_leaders", disabled=not leaders_on)
        leaders_metric = st.radio("Quarterly chart", ["Revenue", "Net_Income"], horizontal=True, key="industry_fund_metric",
                                  format_func=lambda value: value.replace("_", " "), disabled=not leaders_on)

    st.caption(
        "Compare industries through their main index (or the ETF that tracks it, where Yahoo has no index history). "
        "With **Normalize** on, every line starts at 100 so moves are comparable."
    )
    symbols = {label: INDUSTRY_INDEXES[label] for label in chosen}
    for ticker in parse_ticker_list(extra_text):
        symbols.setdefault(ticker, ticker)
    if not symbols:
        st.info("Pick at least one industry index in the settings.")
    else:
        raw_prices = {symbol: index_prices[symbol] for symbol in symbols.values() if symbol in index_prices}
        to_fetch = [symbol for symbol in dict.fromkeys(symbols.values()) if symbol not in raw_prices]
        if to_fetch:
            with st.spinner("Loading prices..."):
                raw_prices.update(load_price_panel(to_fetch, days=1900))
        prices = {label: raw_prices[symbol] for label, symbol in symbols.items() if symbol in raw_prices}
        missing = [label for label in symbols if label not in prices]
        if missing:
            st.caption(f"No price data for: {', '.join(missing)}")
        if prices:
            last_date = max(frame.index[-1] for frame in prices.values())
            start = period_start(last_date, period)
            overlay = overlay_frame(prices, start=start, normalize=normalize)
            fig = go.Figure()
            for label in overlay.columns:
                benchmark = symbols[label] == "^GSPC"
                fig.add_trace(go.Scatter(x=overlay.index, y=overlay[label], name=label,
                                         line=dict(width=2.2 if benchmark else 1.5, dash="dash" if benchmark else None)))
            if normalize:
                fig.add_hline(y=100, line=dict(color="gray", width=1, dash="dot"))
            fig.update_layout(
                title=f"Industry indexes, {period} ({'rebased to 100' if normalize else 'price'})", height=460,
                margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                yaxis=dict(type="log" if log_scale else "linear"),
            )
            show_chart(fig, width="stretch")

            table = snapshot_table(prices, ma_periods=(50, 200))
            raw_overlay = overlay_frame(prices, start=start, normalize=True)
            period_column = f"Return {period} %"
            table[period_column] = (raw_overlay.ffill().iloc[-1] - 100).reindex(table.index)
            table.insert(0, "Trend", [index_trend(prices[label]) for label in table.index])
            table.insert(1, "Symbol", [symbols[label] for label in table.index])
            columns = ["Trend", "Symbol", "Close", "Change %", period_column, "Return 1m %", "Return 3m %", "Return YTD %",
                       "Return 1y %", "% from 52w high", "Close vs SMA 50 %", "Close vs SMA 200 %"]
            columns = [column for column in dict.fromkeys(columns)
                       if column == period_column or column.lower() != period_column.lower()]
            table.index.name = "Industry"
            st.dataframe(table.sort_values(period_column, ascending=False)[columns].style.format(precision=2), width="stretch")
            st.caption("Trend: Uptrend when the close is above its 50-day SMA and that is above the 200-day SMA; "
                       "Downtrend when both are reversed; otherwise Mixed.")

    if leaders_on:
        st.subheader("Leading companies")
        render_industry_leaders(leaders_text, leaders_metric)


def render_industry_leaders(leaders_text, metric):
    leaders = parse_ticker_list(leaders_text)
    if len(leaders) > 5:
        st.warning(f"Only the first 5 of {len(leaders)} tickers are compared.")
        leaders = leaders[:5]
    if not leaders:
        st.info("Type up to 5 tickers in the settings.")
        return
    ma_periods = (20, 50, 200)
    leader_prices = load_price_panel(leaders)
    leader_table = snapshot_table(leader_prices, ma_periods=ma_periods)
    if leader_table.empty:
        st.warning("No price data for these tickers.")
        return
    leader_columns = ["Close", "Prev close", "Change %"] + [f"SMA {p}" for p in ma_periods] + [
        f"Close vs SMA {p} %" for p in ma_periods] + ["Volume / avg", "% from 52w high", "Return 1m %", "Return 3m %", "Return 6m %"]
    st.dataframe(leader_table[leader_columns].style.format(precision=2), width="stretch")

    rebased = overlay_frame(leader_prices, start=period_start(max(f.index[-1] for f in leader_prices.values()), "1Y"))
    fig = go.Figure([go.Scatter(x=rebased.index, y=rebased[column], name=column) for column in rebased.columns])
    fig.update_layout(title="Price over the last year (rebased to 100)", height=340,
                      margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
    show_chart(fig, width="stretch")

    fundamentals_start = (pd.Timestamp(date.today()) - pd.DateOffset(years=4)).strftime("%Y-%m-%d")
    today_str = pd.Timestamp(date.today()).strftime("%Y-%m-%d")
    fundamental_rows, quarterly = {}, {}
    with st.spinner("Loading quarterly fundamentals..."):
        for ticker in leader_table.index:
            fundamentals, _, _ = load_quarterly_fundamentals(ticker, fundamentals_start, today_str)
            trend = fundamental_trend(fundamentals)
            if trend:
                fundamental_rows[ticker] = trend
                quarterly[ticker] = fundamentals
    if fundamental_rows:
        fundamental_table = pd.DataFrame.from_dict(fundamental_rows, orient="index")
        for column in fundamental_table.columns:
            if column.endswith("(last Q)") or column.endswith("TTM"):
                fundamental_table[column] = fundamental_table[column] / 1e6
        fundamental_table = fundamental_table.rename(columns=lambda c: f"{c} ($M)" if c.endswith("(last Q)") or c.endswith("TTM") else c)
        st.dataframe(fundamental_table.style.format(precision=1), width="stretch")
        st.caption("Amounts in millions of the reporting currency. YoY compares with the same quarter a year earlier.")
        fig = go.Figure()
        for ticker, frame in quarterly.items():
            if metric in frame:
                series = frame[metric].dropna().iloc[-8:] / 1e6
                fig.add_trace(go.Bar(x=series.index, y=series, name=ticker))
        fig.update_layout(title=f"{metric.replace('_', ' ')} by quarter (millions)", barmode="group", height=340,
                          margin=dict(l=10, r=10, t=40, b=10))
        show_chart(fig, width="stretch")
    else:
        st.caption("No quarterly fundamentals were available for these tickers.")

    health = leaders_health(leader_table, (50, 200))
    health_columns = st.columns(4)
    health_columns[0].metric("Leaders above SMA 50", f"{health.get('Above SMA 50', 0)} / {health['Leaders']}")
    health_columns[1].metric("Leaders above SMA 200", f"{health.get('Above SMA 200', 0)} / {health['Leaders']}")
    health_columns[2].metric("Avg % from 52w high", f"{health.get('Avg % from 52w high', float('nan')):.1f}%")
    health_columns[3].metric("Avg 3-month return", f"{health.get('Avg 3m return %', float('nan')):.1f}%")


# ============================================================
# Semicon Supply Chain tab
# ============================================================

NO_FOCUS = "(none)"


def supply_graph_key():
    return f"supply_graph_{st.session_state.get('supply_graph_round', 0)}_chart"


def focus_from_graph():
    """A tap or click on a company node in the supply-chain graph makes it the focused company.

    Every click also gives the chart a new key, so the next run draws it fresh instead of keeping Plotly's own
    selection, which would grey out every other point and hide the highlighted chain.
    """

    event = st.session_state.get(supply_graph_key())
    st.session_state["supply_graph_round"] = st.session_state.get("supply_graph_round", 0) + 1
    points = event.selection.points if event and event.selection else []
    for point in points:
        customdata = point.get("customdata")
        if customdata and customdata[0] in SUPPLY_COMPANIES:
            st.session_state["supply_focus"] = customdata[0]
            return


def wrap_text(text, width=48):
    words, lines, line = text.split(), [], ""
    for word in words:
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return "<br>".join(lines + [line])


def node_colors(values, metric):
    """Color range for the node metric: symmetric around 0 for returns, from the lowest value to 0 for % from the high."""

    finite = values[np.isfinite(values)]
    if metric == "% from 52w high":
        return dict(cmin=min(-1.0, float(finite.min())) if finite.size else -30.0, cmax=0.0)
    bound = max(1.0, float(np.nanpercentile(np.abs(finite), 90))) if finite.size else 20.0
    return dict(cmin=-bound, cmax=bound)


def supply_chain_figure(chain, table, metric, focus, highlight):
    positions = node_positions(chain)
    linked = linked_tickers(chain, focus) if focus and highlight else set(positions)
    fig = go.Figure()

    for active in (False, True):
        xs, ys = [], []
        for source, target, _ in SUPPLY_CHAINS[chain]["edges"]:
            if (source in linked and target in linked) == active:
                xs += [positions[source][0], positions[target][0], None]
                ys += [positions[source][1], positions[target][1], None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(width=1.6 if active and focus else 1.0,
                      color="rgba(65,105,225,0.75)" if active and focus else "rgba(128,128,128,0.35)"),
        ))

    # Invisible markers along each edge carry the edge's hover text. They sit 30% of the way from the customer,
    # which never lands on a company (companies sit on whole-number columns, edges span 1 to 3 columns).
    edges = SUPPLY_CHAINS[chain]["edges"]
    fig.add_trace(go.Scatter(
        x=[positions[t][0] + 0.3 * (positions[s][0] - positions[t][0]) for s, t, _ in edges],
        y=[positions[t][1] + 0.3 * (positions[s][1] - positions[t][1]) for s, t, _ in edges],
        mode="markers", marker=dict(size=10, color="rgba(0,0,0,0)"), showlegend=False,
        hovertext=[f"<b>{s} → {t}</b><br>{wrap_text(note)}" for s, t, note in edges], hoverinfo="text",
    ))

    tickers = list(positions)
    values = table.loc[tickers, metric].astype(float).to_numpy()
    labels = [f"<b>{t}</b><br>{v:+.1f}%" if np.isfinite(v) else f"<b>{t}</b><br>n/a" for t, v in zip(tickers, values)]
    customdata = [
        [t, SUPPLY_COMPANIES[t][0], table.at[t, "Level"], wrap_text(SUPPLY_COMPANIES[t][1]), table.at[t, "Trend"],
         *[f"{table.at[t, column]:+.1f}%" if pd.notna(table.at[t, column]) else "n/a"
           for column in ["Return 1m %", "Return 3m %", "Return YTD %", "% from 52w high"]],
         f"{table.at[t, 'Close']:.2f}" if pd.notna(table.at[t, "Close"]) else "n/a"]
        for t in tickers
    ]
    fig.add_trace(go.Scatter(
        x=[positions[t][0] for t in tickers], y=[positions[t][1] for t in tickers],
        mode="markers+text", text=labels, textposition="middle right",
        textfont=dict(size=10 if MOBILE else 12), customdata=customdata, showlegend=False,
        marker=dict(
            size=[26 if t == focus else 18 for t in tickers],
            color=np.where(np.isfinite(values), values, np.nan), colorscale="RdYlGn", **node_colors(values, metric),
            opacity=[1.0 if t in linked else 0.25 for t in tickers],
            line=dict(width=[3 if t == focus else 1 for t in tickers], color="gray"),
            showscale=not MOBILE, colorbar=dict(title=dict(text=metric, side="right"), thickness=10, len=0.6),
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b> · %{customdata[1]}<br><i>%{customdata[2]}</i><br>%{customdata[3]}<br><br>"
            "Close %{customdata[9]} · %{customdata[4]}<br>1m %{customdata[5]} · 3m %{customdata[6]} · "
            "YTD %{customdata[7]}<br>From 52w high %{customdata[8]}<extra></extra>"
        ),
    ))

    for x, level in enumerate(SUPPLY_LEVEL_SHORT_NAMES if MOBILE else SUPPLY_LEVELS):
        fig.add_annotation(x=x, y=1.09, text=f"<b>{level}</b>", showarrow=False, xanchor="left",
                           font=dict(size=11 if MOBILE else 12))
    hidden = dict(visible=False, fixedrange=True)
    fig.update_layout(
        height=max(640, 44 * max(len(level) for level in SUPPLY_CHAINS[chain]["levels"]) + 100),
        margin=dict(l=5, r=5, t=30, b=10), dragmode=False, clickmode="event+select",
        xaxis=dict(range=[-0.15, len(SUPPLY_LEVELS) - 0.35], **hidden), yaxis=dict(range=[-0.06, 1.14], **hidden),
        hoverlabel=dict(align="left"),
    )
    return fig


def render_fundamentals(ticker, years=VALUATION_HISTORY_YEARS, log_price=False):
    """Growth vs price, revenue, EPS, trailing P/E and PEG for one ticker, with their captions."""

    today = pd.Timestamp(date.today()).strftime("%Y-%m-%d")
    history_start = (pd.Timestamp(date.today()) - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    with st.spinner("Loading quarterly revenue and P/E history..."):
        fundamentals, _, _ = load_quarterly_fundamentals(ticker, history_start, today)
        # Load two extra years so TTM EPS, and its year-on-year growth for PEG, exist from the first day shown.
        earnings_start = (pd.Timestamp(history_start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        earnings = load_earnings_data(ticker, earnings_start, today)
        valuation, _ = build_valuation_data(load_data(ticker, earnings_start, today, "Daily"), earnings)
    trend = fundamental_trend(fundamentals)
    if trend and "Revenue" in fundamentals:
        revenue = fundamentals["Revenue"].dropna()
        yoy = trend.get("Revenue YoY %")
        margin = trend.get("Net margin %")
        st.caption(
            f"Revenue last quarter: {revenue.iloc[-1] / 1e6:,.0f}M in its reporting currency"
            + (f", {yoy:+.1f}% year on year" if yoy is not None and pd.notna(yoy) else "")
            + (f". Net margin {margin:.1f}%." if margin is not None and pd.notna(margin) else ".")
        )
        st.markdown(f"**{ticker} fundamentals: growth, revenue, EPS and valuation, last {years} years**")
        peg = trailing_peg(valuation).loc[history_start:]
        trailing_pe = valuation["Trailing_PE"].loc[history_start:]
        forward_pe = load_fundamental_snapshot(ticker).get("forwardPE")
        estimates = revenue_estimate_quarters(load_revenue_estimates(ticker), revenue.index[-1])
        all_eps = quarterly_eps(earnings, revenue.index)
        eps = all_eps.loc[history_start:]
        eps_estimates = eps_estimate_quarters(load_earnings_estimates(ticker), revenue.index[-1])
        price_change = price_yoy(valuation["Close"]).loc[history_start:]
        growth = {"Revenue": yoy_change(revenue), "EPS": yoy_change(all_eps)}
        # Captions follow the panel order: growth vs price above the chart, valuation below it.
        st.caption(correlation_text({name: growth_price_correlation(series.loc[history_start:], price_change)
                                     for name, series in growth.items()}))
        show_chart(revenue_pe_figure(revenue, trailing_pe, estimates, peg=peg, eps=eps, eps_estimates=eps_estimates,
                                     price_change=price_change, growth=growth,
                                     close=valuation["Close"].loc[history_start:], log_price=log_price),
                   width="stretch")
        st.caption(pe_summary_text(pe_summary(trailing_pe), forward_pe))
        st.caption(peg_summary_text(peg, forward_pe, next_year_eps_growth(ticker)))
        st.caption(
            "YoY change: the price daily, revenue and EPS by quarter, left empty when the year-ago value is zero or "
            "negative. Revenue is in billions of the reporting currency. EPS is Yahoo's reported (adjusted) EPS per "
            "share, on the quarter it reports rather than the announcement date. The daily close is split-adjusted "
            "(not dividend-adjusted), the same price the P/E uses. Trailing P/E is price / last 4 "
            "quarters' EPS, so it can differ from GAAP P/E; a spike usually means earnings fell, not that the price "
            "ran. PEG is that P/E / EPS growth %, blank while growth is zero or negative. Hatched bars are analyst "
            "estimates, not results."
        )
    else:
        st.caption("No quarterly revenue was available for this company.")



def render_stock_fundamentals():
    """Fundamentals of any ticker: growth vs price, revenue, EPS, trailing P/E and PEG."""

    with settings_panel("Stock"):
        ticker = st.text_input("Ticker", value="NVDA", key="fundamentals_ticker",
                               help="Any Yahoo symbol with quarterly results, e.g. NVDA, TSM, BESIY.").strip().upper()
        years = st.select_slider("History (years)", FUNDAMENTALS_YEARS, value=VALUATION_HISTORY_YEARS,
                                 key="fundamentals_years")
        log_price = st.checkbox("Log scale for the price", value=False, key="fundamentals_log_price",
                                help="Equal % moves look the same size, so early years aren't flattened.")
    st.caption(
        "Is the price following the business? The top panel compares the year-over-year change in revenue and EPS "
        "with the price's own change; below it are revenue, EPS and the valuation those results imply."
    )
    if not ticker:
        st.info("Enter a ticker in the settings.")
        return
    name = load_fundamental_snapshot(ticker).get("shortName")
    st.subheader(f"{ticker} · {name}" if name else ticker)
    render_fundamentals(ticker, years, log_price)


def render_company_detail(chain, ticker, table, prices, index_prices):
    name, role = SUPPLY_COMPANIES[ticker]
    st.subheader(f"{ticker} · {name}")
    st.caption(f"{table.at[ticker, 'Level']}. {role}")
    tiles = st.columns(4)
    close = table.at[ticker, "Close"]
    tiles[0].metric("Close", f"{close:.2f}" if pd.notna(close) else "n/a")
    for tile, column in zip(tiles[1:3], ["Return 3m %", "% from 52w high"]):
        value = table.at[ticker, column]
        tile.metric(column.replace(" %", ""), f"{value:+.1f}%" if pd.notna(value) else "n/a")
    tiles[3].metric("Trend", table.at[ticker, "Trend"])

    suppliers, customers = neighbors(chain, ticker)
    columns = st.columns(2)
    for column, title, links in [(columns[0], "Buys from", suppliers), (columns[1], "Sells to", customers)]:
        with column:
            st.markdown(f"**{title}**")
            st.markdown("\n".join(f"- **{other}** ({SUPPLY_COMPANIES[other][0]}): {note}" for other, note in links)
                        if links else "_None in this chain._")

    if ticker in prices:
        lines = {ticker: prices[ticker]}
        if "^SOX" in index_prices:
            lines["^SOX"] = index_prices["^SOX"]
        rebased = overlay_frame(lines, start=period_start(prices[ticker].index[-1], "1Y"))
        fig = go.Figure([go.Scatter(x=rebased.index, y=rebased[column], name=column,
                                    line=dict(width=1.2, dash="dot") if column == "^SOX" else dict(width=1.8))
                         for column in rebased.columns])
        fig.add_hline(y=100, line=dict(color="gray", width=1, dash="dot"))
        fig.update_layout(title=f"{ticker} vs the semiconductor index, last year (rebased to 100)", height=320,
                          margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
        show_chart(fig, width="stretch")

    render_fundamentals(ticker)


def render_supply_chain(index_prices):
    """Bottom-up supply chain of one end market as a graph: hover a company for details, tap or click it to focus."""

    with settings_panel("Supply chain"):
        chain = st.selectbox("End market", list(SUPPLY_CHAINS), key="supply_chain")
        metric = st.selectbox("Node metric", SUPPLY_NODE_METRICS, index=1, key="supply_metric",
                              help="Shown on each company and used for its color.")
        highlight = st.checkbox("Highlight the focused company's chain", value=True, key="supply_highlight",
                                help="Dims everything that is not upstream or downstream of the focused company.")

    tickers = chain_tickers(chain)
    if st.session_state.get("supply_focus") not in [NO_FOCUS] + tickers:
        st.session_state["supply_focus"] = NO_FOCUS
    focus = st.session_state.get("supply_focus", NO_FOCUS)
    focus = None if focus == NO_FOCUS else focus

    st.caption(
        f"**{chain}**: traced back from the end market, US-listed companies only. A line means the "
        "left company buys from the right one. Hover a company or a line for details, or tap a company to focus it. "
        "Links are curated from public reporting and show the main relationships, not every one."
    )
    with st.spinner("Loading prices..."):
        prices = load_price_panel(tickers)
    if not prices:
        st.warning("No price data was returned. Yahoo Finance may be rate-limiting; try again in a few minutes.")
    table = node_table(chain, prices)

    show_chart(supply_chain_figure(chain, table, metric, focus, highlight), width="stretch", key=supply_graph_key(),
               on_select=focus_from_graph, selection_mode="points", config={"displayModeBar": False})
    st.selectbox("Focused company", [NO_FOCUS] + tickers, key="supply_focus",
                 format_func=lambda t: t if t == NO_FOCUS else f"{t} · {SUPPLY_COMPANIES[t][0]}")

    if focus:
        render_company_detail(chain, focus, table, prices, index_prices)

    with st.expander("All companies", expanded=not focus):
        columns = ["Company", "Level", "Trend", "Close"] + SUPPLY_NODE_METRICS
        st.dataframe(table[columns].style.format(precision=2), width="stretch")


# ============================================================
# Stock Screener tab
# ============================================================

DEFAULT_SCREEN_ROWS = [
    {"group": MOVING_AVERAGE, "left": "Close", "left_period": 0, "operator": ">", "right": "SMA", "right_value": 50.0},
    {"group": MOVING_AVERAGE, "left": "SMA", "left_period": 50, "operator": ">", "right": "SMA", "right_value": 200.0},
    {"group": MOVING_AVERAGE, "left": "Close", "left_period": 0, "operator": ">", "right": "SMA", "right_value": 200.0},
]


def screen_rows():
    """Condition rows as {row id: seed values}; ids keep widget keys stable when rows are removed."""

    if "screen_rows" not in st.session_state:
        st.session_state["screen_rows"] = {index: dict(seed) for index, seed in enumerate(DEFAULT_SCREEN_ROWS)}
    return st.session_state["screen_rows"]


def add_screen_row():
    rows = screen_rows()
    rows[max(rows, default=-1) + 1] = {"group": MOVING_AVERAGE}


def remove_screen_row(row_id):
    screen_rows().pop(row_id, None)


def row_group(row_id, seed):
    return st.session_state.get(f"screen_{row_id}_group", seed.get("group", MOVING_AVERAGE))


def render_condition_row(row_id, seed, category_options):
    """One condition row: the signal group first, then inputs that fit the group. Returns a condition dict or None.

    Laid out for a narrow column (the sidebar, or a phone): the group and a remove button on one line, the
    group's inputs below.
    """

    groups = list(SIGNAL_GROUPS)
    with st.container(border=True):
        top = st.columns([3, 1], vertical_alignment="bottom")
        group = top[0].selectbox("Signal group", groups, index=groups.index(seed.get("group", MOVING_AVERAGE)),
                                 key=f"screen_{row_id}_group")
        top[1].button("✕", key=f"screen_{row_id}_remove_btn", on_click=remove_screen_row, args=(row_id,),
                      help="Remove this condition")
        spec = SIGNAL_GROUPS[group]
        key = f"screen_{row_id}_{group}"
        kind = spec["kind"]

        if kind == "ma":
            cells = st.columns([3, 2])
            left = cells[0].selectbox("Left", ["Close", "SMA", "EMA"], index=["Close", "SMA", "EMA"].index(seed.get("left", "Close")),
                                      key=f"{key}_left")
            left_period = cells[1].number_input("Period", min_value=0, max_value=400, value=int(seed.get("left_period", 0)), step=1,
                                                key=f"{key}_left_period", disabled=left == "Close", help="MA length (ignored for Close).")
            operator = st.selectbox("Operator", SCREEN_OPERATORS, index=SCREEN_OPERATORS.index(seed.get("operator", ">")),
                                    key=f"{key}_operator")
            cells = st.columns([3, 2])
            right = cells[0].selectbox("Right", SCREEN_LINE_KINDS, index=SCREEN_LINE_KINDS.index(seed.get("right", "SMA")),
                                       key=f"{key}_right")
            right_value = cells[1].number_input("Period / value", value=float(seed.get("right_value", 50.0)),
                                                key=f"{key}_right_value", disabled=right == "Close",
                                                help="MA length, or the number for Value.")
            frame = pd.DataFrame([{"Left": left, "Left period": left_period, "Operator": operator, "Right": right,
                                   "Right period / value": right_value}])
            conditions = editor_conditions(frame)
            if not conditions:
                return None
            return {"group": MOVING_AVERAGE, **conditions[0]}

        if kind == "number":
            default = spec.get("default", (">", 0.0))
            period = None
            if spec.get("periods"):
                period = st.selectbox("Period", list(spec["periods"]), index=1, key=f"{key}_period")
            label = spec["periods"][period] if period else spec["column"]
            operator = st.selectbox("Operator", NUMBER_OPERATORS, index=NUMBER_OPERATORS.index(default[0]), key=f"{key}_operator")
            cells = st.columns(2)
            value = cells[0].number_input(label if operator != "Between" else f"{label} from", value=float(default[1]),
                                          key=f"{key}_value", help=spec.get("help"))
            condition = {"group": group, "operator": operator, "value": float(value)}
            if period:
                condition["period"] = period
            if operator == "Between":
                upper = default[2] if len(default) > 2 else float(default[1]) + 10
                condition["value2"] = float(cells[1].number_input("to", value=float(upper), key=f"{key}_value2"))
            return condition

        if kind == "category":
            operator = st.selectbox("Operator", CATEGORY_OPERATORS, key=f"{key}_operator")
            options = category_options.get(group, [])
            if not options:
                st.caption(f"No {group.lower()} data yet; load the stocks first.")
                return None
            chosen = st.session_state.get(f"{key}_values")
            if chosen and any(value not in options for value in chosen):
                # Options follow the loaded stocks; drop picks that are no longer offered.
                st.session_state[f"{key}_values"] = [value for value in chosen if value in options]
            values = st.multiselect(group, options, key=f"{key}_values")
            if not values:
                return None
            return {"group": group, "operator": operator, "values": values}

        days = st.number_input("Within the last (days)" if kind == "past" else "Within the next (days)",
                               min_value=1, max_value=365, value=int(spec.get("default", 30)), step=1, key=f"{key}_days")
        return {"group": group, "days": int(days)}


def open_in_signal_analysis(ticker):
    """Send a screener result to the ticker box and switch to the Signal Analysis tab."""

    st.session_state["ticker_input"] = ticker
    st.session_state["main_tab"] = SIGNAL_TAB


def render_stock_screener():
    """Stock screen over a US index's members, with conditions picked by signal group."""

    market = "US"
    with settings_panel("Stocks to scan"):
        universe = st.selectbox("Universe", SCREENER_MARKETS[market]["universes"], key="screener_universe")
        if universe == "Custom list":
            custom_text = st.text_area("Tickers (comma or space separated)", value="AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA",
                                       key="screener_custom")
            members = pd.DataFrame({"Ticker": parse_ticker_list(custom_text), "Name": "", "Sector": "", "Industry": ""})
            members_error = ""
        else:
            members, members_error = load_universe_members(market, universe)
        load_clicked = st.button(f"Load data for {len(members)} stocks", type="primary", key="screener_load_btn",
                                 disabled=members.empty)

    tickers = members["Ticker"].tolist()
    data_key = json.dumps([market, universe, tickers])
    loaded = st.session_state.get("screener_data")
    if load_clicked:
        progress = st.progress(0.0, text="Downloading prices...")
        prices = load_price_panel(tickers, progress=lambda value: progress.progress(value, text="Downloading prices..."))
        progress.empty()
        loaded = {"key": data_key, "prices": prices}
        st.session_state["screener_data"] = loaded
    is_loaded = loaded is not None and loaded["key"] == data_key
    prices = loaded["prices"] if is_loaded else {}

    # Per-ticker columns the signal groups read, beyond the price snapshot.
    rows = screen_rows()
    groups_in_use = {row_group(row_id, seed) for row_id, seed in rows.items()}
    scanned = [ticker for ticker in tickers if ticker in prices] if is_loaded else tickers
    extra = members.drop_duplicates("Ticker").set_index("Ticker")[["Name", "Sector", "Industry"]].reindex(scanned)
    extra = extra.fillna("").replace("nan", "")
    fundamentals_note = fundamentals_warning = ""
    if is_loaded and prices:
        wants_yahoo = any(SIGNAL_GROUPS[group].get("fundamental") for group in groups_in_use) or (
            bool(groups_in_use & {"Sector", "Industry"}) and (extra["Sector"].eq("") | extra["Industry"].eq("")).any())
        if wants_yahoo:
            progress = st.progress(0.0, text="Loading fundamentals from Yahoo...")
            fundamentals = load_fundamental_table(scanned, progress=lambda value: progress.progress(value, text="Loading fundamentals from Yahoo..."))
            progress.empty()
            # Yahoo can rate-limit or block quote lookups; the columns must exist anyway so the conditions
            # simply fail for stocks without data instead of crashing the table.
            fundamentals = fundamentals.reindex(index=scanned, columns=FUNDAMENTAL_COLUMNS)
            extra = extra.join(fundamentals, how="left")
            extra["Sector"] = extra["Sector"].where(extra["Sector"] != "", extra["Yahoo sector"].fillna(""))
            extra["Industry"] = extra["Industry"].where(extra["Industry"] != "", extra["Yahoo industry"].fillna(""))
            found = int(fundamentals.notna().any(axis=1).sum())
            fundamentals_note = f"Fundamentals for {found} of {len(scanned)} stocks (Yahoo Finance, refreshed every 12 hours)."
            if found < len(scanned) / 2:
                fundamentals_warning = (f"Yahoo Finance returned fundamentals for only {found} of {len(scanned)} stocks, "
                                        "so conditions on them fail for the rest. Yahoo may be rate-limiting; it retries "
                                        "the missing ones after 15 minutes.")
        if "Index" in groups_in_use:
            membership = index_membership(market, scanned)
            extra["Index"] = [membership.get(ticker, []) for ticker in extra.index]
    category_options = {
        "Index": [name for name in SCREENER_MARKETS[market]["universes"] if name != "Custom list"],
        "Sector": sorted(set(GICS_SECTORS) | {value for value in extra["Sector"] if value}),
        "Industry": sorted({value for value in extra["Industry"] if value}),
        "Analyst rating": ANALYST_RATINGS,
    }

    with settings_panel("Conditions"):
        st.caption("A stock must meet every condition. Pick a signal group first; the inputs below it follow the group.")
        conditions = []
        for row_id, seed in list(rows.items()):
            condition = render_condition_row(row_id, seed, category_options)
            if condition is not None:
                conditions.append(condition)
        st.button("+ Add condition", on_click=add_screen_row, key="screen_add_row_btn")
    with settings_panel("More filters", expanded=False):
        use_high = st.checkbox("Within % of 52w high", value=True, key="screener_use_high")
        max_from_high = st.number_input("Max % below 52w high", 0.0, 100.0, 25.0, 1.0,
                                        key="screener_max_from_high", disabled=not use_high)
        min_price = st.number_input("Min price", 0.0, value=5.0, step=1.0, key="screener_min_price")
        min_volume = st.number_input("Min avg daily volume (shares)", 0.0, value=0.0, step=100000.0, key="screener_min_volume")
        cross_window = int(st.number_input("Cross window (bars)", 1, 60, 5, 1, key="screener_cross_window",
                                           help="Cross Above / Below counts when the cross happened within this many bars and still holds."))
        recent_days = int(st.number_input("New-high window (days)", min_value=1, max_value=120, value=20, step=1,
                                          key="screener_recent_days",
                                          help="The results show whether each stock made a new 52-week high in this many trading days."))
        ma_columns = st.columns(3)
        ma_fast = int(ma_columns[0].number_input("Short MA", min_value=2, max_value=400, value=20, step=1, key="screener_ma_fast"))
        ma_mid = int(ma_columns[1].number_input("Medium MA", min_value=2, max_value=400, value=50, step=1, key="screener_ma_mid"))
        ma_slow = int(ma_columns[2].number_input("Long MA", min_value=2, max_value=400, value=200, step=1, key="screener_ma_slow"))
        require_market = st.checkbox("Only list stocks when the Market Analysis filter passes", value=False,
                                     key="screener_require_gates")
    ma_periods = tuple(dict.fromkeys([ma_fast, ma_mid, ma_slow]))

    st.caption(
        "Scan an index's members for stocks that meet every condition. Each condition starts with a **signal group** "
        "(moving averages, price, valuation, growth, sector, analyst rating, earnings dates ...). Prices come from "
        "Yahoo Finance (cached for 6 hours); fundamentals are fetched only when a condition needs them."
    )
    if members.empty:
        st.error(f"Could not load the members of {universe}. {members_error} Try **Custom list** instead.")
        return
    if not is_loaded:
        st.info(f"Press **Load data for {len(tickers)} stocks** in the settings to download prices. A full index takes "
                "about a minute the first time, then it is cached.")
        return
    missing = [ticker for ticker in tickers if ticker not in prices]
    if missing:
        st.caption(f"No price data for {len(missing)} ticker(s): {', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}")
    if not prices:
        st.error("No price data was returned. Yahoo Finance may be rate-limiting; try again in a few minutes.")
        return
    if fundamentals_warning:
        st.warning(fundamentals_warning)
    elif fundamentals_note:
        st.caption(fundamentals_note)

    results = screen_stocks(
        prices,
        conditions,
        max_pct_from_high=float(max_from_high) if use_high else None,
        min_price=float(min_price) if min_price > 0 else None,
        min_avg_volume=float(min_volume) if min_volume > 0 else None,
        cross_window=cross_window,
        ma_periods=ma_periods,
        recent_days=recent_days,
        extra=extra,
    )
    if results.empty:
        st.warning("No stocks to screen.")
        return
    passing = results[results["Pass"]]
    if require_market:
        market_ok = st.session_state.get("market_filter_ok")
        if market_ok is None:
            st.info("Open the **Market Analysis** tab once to evaluate the market filter; it is ignored for now.")
        elif not market_ok:
            st.error(f"{len(passing)} stock(s) meet the conditions, but they are hidden because the market filter fails "
                     "(see Market Analysis).")
            return
    st.success(f"**{len(passing)} of {len(results)} stocks pass** "
               + ", ".join(condition_label(condition) for condition in conditions)
               + (f", within {max_from_high:g}% of the 52w high" if use_high else ""))
    show_all = st.checkbox("Show stocks that fail too", value=False, key="screener_show_all")
    shown = (results if show_all else passing).copy()
    if "Index" in shown:
        shown["Index"] = shown["Index"].apply(lambda value: ", ".join(value) if isinstance(value, list) else "")
    # The columns after Close / Change % follow the signal groups in use.
    group_columns = [condition_column(condition) for condition in conditions if condition["group"] != MOVING_AVERAGE]
    base_columns = ["Pass", "Name", "Close", "Change %"]
    display_columns = list(dict.fromkeys(base_columns + group_columns + [f"Close vs SMA {p} %" for p in ma_periods] + [
        "% from 52w high", f"New 52w high in last {recent_days}d", "Volume / avg", "Return 1m %", "Return 3m %"]))
    display_columns = [column for column in display_columns if column in results.columns]
    condition_columns = [column for column in results.columns if column not in display_columns and results[column].dtype == bool]
    st.caption("Select a row to analyze that stock in Signal Analysis.")
    event = st.dataframe(shown[display_columns + condition_columns].style.format(precision=2), width="stretch", height=420,
                         on_select="rerun", selection_mode="single-row", key="screener_results_table")
    selected = event.selection.rows if event is not None and hasattr(event, "selection") else []
    action_columns = st.columns([2, 1])
    if selected and selected[0] < len(shown):
        picked = shown.index[selected[0]]
        action_columns[0].button(f"Analyze {picked} in Signal Analysis", type="primary", key="screener_open_btn",
                                 on_click=open_in_signal_analysis, args=(picked,))
    action_columns[1].download_button(
        "Download results (CSV)",
        shown.to_csv().encode("utf-8"),
        file_name=f"screener_{market}_{pd.Timestamp(date.today()):%Y%m%d}.csv",
        mime="text/csv",
    )


# ============================================================
# Backtest / Signal Analysis / Validate: strategy settings
# ============================================================

SETUP_OPTIONS = {
    "Moving-average rule": "ma_rule",
    "Breakout (new high/low on volume)": "breakout",
    "Pullback in trend (dip to the EMA, then reversal)": "pullback",
    "Reversal (buy a breakdown in an uptrend)": "reversal",
}


def strategy_settings():
    """Ticker, dates, signal rule and risk settings shared by Backtest, Signal Analysis and Validate.

    Returns a namespace of the settings, or None (after saying why) when the ticker or dates are not usable.
    """

    with settings_panel("Ticker & dates"):
        ticker = normalize_symbol(st.text_input("Ticker Symbol", value="AAPL", key="ticker_input"))
        timeframe = st.selectbox("Timeframe", ["Daily", "Weekly", "Monthly"], index=0, key="timeframe")
        date_range = st.date_input(
            "Date Range",
            value=(date(2022, 1, 1), date.today()),
            max_value=date.today(),
            format="YYYY-MM-DD",
            key="date_range",
        )
        account_size = st.number_input(
            "Account size ($)",
            min_value=1000.0,
            value=100000.0,
            step=5000.0,
            key="account_size",
            help="Starting equity for the Backtest tab and the equity used for position sizing.",
        )

    # Moving-average rule defaults (also used for the Backtest chart's MA lines).
    fast_type, fast_period, slow_type, slow_period = "EMA", 20, "SMA", 50
    buy_left, buy_operator, buy_right = "Fast_MA", "Cross Above", "Slow_MA"
    sell_left, sell_operator, sell_right = "Fast_MA", "Cross Below", "Slow_MA"
    breakout_lookback, volume_multiple = 20, 1.5
    pullback_ema, trend_mid, trend_slow = 20, 50, 200
    swing_bars, pullback_bars, min_depth_atr = 20, 5, 1.5

    with settings_panel("Signal"):
        setup_label = st.selectbox(
            "Signal rule",
            list(SETUP_OPTIONS),
            index=0,
            key="signal_rule",
            help="Used by every tab: the Backtest buys and sells on it, Signal Analysis builds a trade plan from it, "
            "and Validate replays it.",
        )
        setup = SETUP_OPTIONS[setup_label]

        if setup == "ma_rule":
            ma_columns = st.columns(2)
            fast_type = ma_columns[0].selectbox("Fast MA Type", ["EMA", "SMA"], index=0, key="fast_type")
            fast_period = int(ma_columns[1].number_input("Fast MA Period", min_value=1, value=20, step=1, key="fast_period"))
            slow_type = ma_columns[0].selectbox("Slow MA Type", ["EMA", "SMA"], index=1, key="slow_type")
            slow_period = int(ma_columns[1].number_input("Slow MA Period", min_value=1, value=50, step=1, key="slow_period"))

            st.markdown("**Buy when**")
            buy_left = st.selectbox("Buy left signal", MA_RULE_SERIES, index=1, key="buy_left", label_visibility="collapsed")
            buy_operator = st.selectbox("Buy operator", MA_RULE_OPERATORS, index=4, key="buy_operator", label_visibility="collapsed")
            buy_right = st.selectbox("Buy right signal", MA_RULE_SERIES, index=2, key="buy_right", label_visibility="collapsed")

            st.markdown("**Sell / exit when**")
            sell_left = st.selectbox("Sell left signal", MA_RULE_SERIES, index=1, key="sell_left", label_visibility="collapsed")
            sell_operator = st.selectbox("Sell operator", MA_RULE_OPERATORS, index=5, key="sell_operator", label_visibility="collapsed")
            sell_right = st.selectbox("Sell right signal", MA_RULE_SERIES, index=2, key="sell_right", label_visibility="collapsed")
        elif setup == "breakout":
            breakout_lookback = int(st.number_input("Breakout lookback (bars)", min_value=2, max_value=250, value=20, step=1,
                                                    key="breakout_lookback"))
            volume_multiple = st.number_input("Breakout volume (x 50-bar average)", min_value=1.0, max_value=5.0, value=1.5, step=0.1,
                                              key="volume_multiple")
            st.caption(
                f"Buy: the first close above the prior {breakout_lookback}-bar high on at least {volume_multiple:g}x "
                f"average volume. Sell / exit: the first close below the prior {breakout_lookback}-bar low."
            )
        elif setup == "pullback":
            trend_columns = st.columns(2)
            trend_mid = int(trend_columns[0].number_input("Trend fast MA (bars)", min_value=5, max_value=250, value=50, step=1, key="trend_mid"))
            trend_slow = int(trend_columns[1].number_input("Trend slow MA (bars)", min_value=10, max_value=400, value=200, step=1, key="trend_slow"))
            pullback_columns = st.columns(2)
            pullback_ema = int(pullback_columns[0].number_input("Pullback EMA (bars)", min_value=2, max_value=100, value=20, step=1, key="pullback_ema"))
            swing_bars = int(pullback_columns[1].number_input("Swing high lookback (bars)", min_value=2, max_value=120, value=20, step=1, key="swing_bars"))
            pullback_bars = int(pullback_columns[0].number_input("Pullback window (bars)", min_value=2, max_value=30, value=5, step=1, key="pullback_bars"))
            min_depth_atr = float(pullback_columns[1].number_input("Min dip depth (ATR)", min_value=0.0, max_value=10.0, value=1.5, step=0.1, key="min_depth_atr"))
            st.caption(
                f"Buy: {trend_mid}-bar MA rising and above the {trend_slow}-bar MA, a dip of at least {min_depth_atr:g} ATR "
                f"from the {swing_bars}-bar high reaching the {pullback_ema} EMA within the last {pullback_bars} bars, then a "
                "close above the prior high. Sell / exit: the mirror image in a downtrend."
            )
        else:
            breakout_lookback = int(st.number_input("Breakdown lookback (bars)", min_value=2, max_value=250, value=20, step=1,
                                                    key="breakdown_lookback"))
            trend_columns = st.columns(2)
            trend_mid = int(trend_columns[0].number_input("Trend fast MA (bars)", min_value=5, max_value=250, value=50, step=1, key="trend_mid"))
            trend_slow = int(trend_columns[1].number_input("Trend slow MA (bars)", min_value=10, max_value=400, value=200, step=1, key="trend_slow"))
            st.caption(
                f"Buy: the first close below the prior {breakout_lookback}-bar low while the close is above the "
                f"{trend_slow}-bar MA and the {trend_mid}-bar MA is above it. There is no sell signal; trades exit at "
                "their stop or target."
            )

        allow_short = st.checkbox("Allow short trades (otherwise sell signals mean exit)", value=False, key="allow_short")

    with settings_panel("Risk & sizing", expanded=False):
        risk_pct = st.number_input("Risk per trade (% of equity)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="risk_pct")
        max_position_pct = st.number_input("Max position (% of equity)", min_value=1.0, max_value=100.0, value=25.0, step=1.0,
                                           key="max_position_pct")
        full_size_confidence = st.slider("Full-size confidence threshold", 50, 95, 75, key="full_size_confidence")
        half_size_confidence = st.slider("Half-size confidence threshold", 40, 90, 60, key="half_size_confidence")
        opening_range_minutes = st.selectbox("Intraday opening range (minutes)", [15, 30, 60], index=1, key="opening_range_minutes")

    ma_rule = {
        "fast_type": fast_type,
        "fast_period": int(fast_period),
        "slow_type": slow_type,
        "slow_period": int(slow_period),
        "buy": (buy_left, buy_operator, buy_right),
        "sell": (sell_left, sell_operator, sell_right),
    }
    signal_settings = AnalysisSettings(
        setup=setup,
        allow_short=allow_short,
        breakout_lookback=breakout_lookback,
        volume_multiple=float(volume_multiple),
        opening_range_minutes=int(opening_range_minutes),
        ma_rule=ma_rule if setup == "ma_rule" else None,
        pullback_ema=pullback_ema,
        trend_mid=trend_mid,
        trend_slow=trend_slow,
        swing_bars=swing_bars,
        pullback_bars=pullback_bars,
        min_depth_atr=min_depth_atr,
    )
    risk_settings = RiskSettings(
        account_equity=float(account_size),
        risk_pct=float(risk_pct),
        max_position_pct=float(max_position_pct),
        full_size_confidence=float(max(full_size_confidence, half_size_confidence)),
        half_size_confidence=float(min(full_size_confidence, half_size_confidence)),
    )

    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.info("Please select both a start date and an end date.")
        return None
    start_date, end_date = date_range
    if start_date > end_date:
        st.error("The start date must be earlier than the end date.")
        return None
    ticker = ticker.strip().upper()
    if not ticker:
        st.error("Please enter a ticker symbol.")
        return None

    return SimpleNamespace(
        ticker=ticker, timeframe=timeframe, start_date=start_date, end_date=end_date, account_size=account_size,
        setup=setup, setup_label=setup_label, fast_type=fast_type, fast_period=fast_period, slow_type=slow_type,
        slow_period=slow_period, buy_left=buy_left, buy_operator=buy_operator, buy_right=buy_right,
        sell_left=sell_left, sell_operator=sell_operator, sell_right=sell_right, breakout_lookback=breakout_lookback,
        volume_multiple=volume_multiple, ma_rule=ma_rule, signal_settings=signal_settings, risk_settings=risk_settings,
    )


# ============================================================
# Backtest tab
# ============================================================

def render_backtest(s):
    """Moving-average / signal-rule backtest of the chosen ticker with EPS, fundamentals, P/E and benchmark panels."""

    ticker, timeframe, start_date, end_date = s.ticker, s.timeframe, s.start_date, s.end_date
    account_size, setup, setup_label, ma_rule = s.account_size, s.setup, s.setup_label, s.ma_rule
    fast_type, fast_period, slow_type, slow_period = s.fast_type, s.fast_period, s.slow_type, s.slow_period
    buy_left, buy_operator, buy_right = s.buy_left, s.buy_operator, s.buy_right
    sell_left, sell_operator, sell_right = s.sell_left, s.sell_operator, s.sell_right
    breakout_lookback, volume_multiple, signal_settings = s.breakout_lookback, s.volume_multiple, s.signal_settings

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
        return

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

    if setup == "ma_rule" and len(df) < max(fast_period, slow_period):
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

    # --------------------------------------------------------
    # Align quarterly fundamentals with EPS announcement dates
    # --------------------------------------------------------
    fundamentals_plot_df = (
        align_fundamentals_to_earnings_dates(
            fundamentals_df=fundamentals_df,
            earnings_df=earnings_calculated_df,
            maximum_days=120,
        )
    )

    pe_error = valuation_df["PE_Reconstruction_Error"].dropna()
    if not pe_error.empty:
        st.caption(f"Maximum internal P/E calculation error: ${pe_error.max():.8f}")

    if setup == "ma_rule":
        _, _, buy_condition, sell_condition = ma_rule_conditions(df, ma_rule)
        rule_text = (
            f"**BUY:** `{buy_left} {buy_operator} {buy_right}`\n\n"
            f"**SELL:** `{sell_left} {sell_operator} {sell_right}`"
        )
    else:
        backtest_signals = detect_signals(
            df, setup, breakout_lookback, float(volume_multiple), **signal_settings.detector_kwargs()
        )
        buy_condition = backtest_signals["Signal_Up"].fillna(False).astype(bool)
        sell_condition = backtest_signals["Signal_Down"].fillna(False).astype(bool)
        rule_text = (
            f"**BUY / SELL:** {setup_label} signals on {timeframe.lower()} bars. The position is held until "
            "the opposite signal. Stops and targets are not simulated here; the **Validate** tab replays the "
            "full trade plan."
        )
    df = run_position_engine(df, buy_condition, sell_condition)

    df["Returns"] = df["Close"].pct_change().fillna(0.0)
    df["Strat_Returns"] = df["Position"] * df["Returns"]
    df["Equity"] = float(account_size) * (1.0 + df["Strat_Returns"]).cumprod()
    df["Peak"] = df["Equity"].cummax()
    df["Drawdown"] = df["Equity"] / df["Peak"] - 1.0

    final_equity = df["Equity"].iloc[-1]
    total_return = (final_equity / float(account_size) - 1.0) * 100.0
    max_drawdown = df["Drawdown"].min() * 100.0
    active_returns = df.loc[df["Strat_Returns"] != 0, "Strat_Returns"]
    positive_active_bars = 0.0 if active_returns.empty else active_returns.gt(0).mean() * 100.0

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Final Equity", f"${final_equity:,.2f}", f"{total_return:.2f}%")
    metric_col2.metric("Max Drawdown", f"{max_drawdown:.2f}%")
    metric_col3.metric("Positive Active Bars", f"{positive_active_bars:.1f}%")
    st.info(rule_text)

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
    if setup == "ma_rule":
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

    scale_divisor, fundamentals_axis_title = (
        choose_fundamental_scale(
            fundamentals_plot_df
        )
    )

    fundamental_definitions = [
        (
            "Revenue",
            "Revenue",
            "royalblue",
        ),
        (
            "Total_Expense",
            "Total expenses",
            "crimson",
        ),
        (
            "Net_Income",
            "Net income",
            "limegreen",
        ),
        (
            "Operating_Expense",
            "Operating expenses",
            "orange",
        ),
        (
            "CapEx",
            "CapEx spending",
            "purple",
        ),
    ]


    for (
        column,
        trace_name,
        color,
    ) in fundamental_definitions:

        if (
            fundamentals_plot_df.empty
            or column not in fundamentals_plot_df.columns
        ):
            continue

        trace_data = fundamentals_plot_df[
            [
                column,
                "Fiscal_Quarter_End",
            ]
        ].dropna(
            subset=[column]
        )

        if trace_data.empty:
            continue

        fundamentals_trace_added = True

        scaled_values = (
            trace_data[column]
            / scale_divisor
        )

        fiscal_quarter_end_text = (
            trace_data["Fiscal_Quarter_End"]
            .dt.strftime("%Y-%m-%d")
        )

        fig.add_trace(
            go.Bar(
                x=trace_data.index,
                y=scaled_values,
                name=trace_name,
                marker_color=color,
                opacity=0.78,
                customdata=np.column_stack(
                    [
                        fiscal_quarter_end_text,
                    ]
                ),
                hovertemplate=(
                    f"{trace_name}<br>"
                    "Earnings date: "
                    "%{x|%Y-%m-%d}<br>"
                    "Fiscal quarter end: "
                    "%{customdata[0]}<br>"
                    "Value: "
                    f"%{{y:,.2f}} "
                    f"{fundamentals_axis_title}"
                    "<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )


    if not fundamentals_trace_added:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="x3 domain",
            yref="y3 domain",
            text=(
                "Quarterly revenue, total expenses, "
                "net income, operating expenses and "
                "CapEx data are unavailable"
            ),
            showarrow=False,
            font=dict(
                color="gray",
                size=12,
            ),
        )


    fig.add_hline(
        y=0,
        line_width=1,
        line_dash="dot",
        line_color="gray",
        row=3,
        col=1,
    )

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

    # Apply the exact same time range to every subplot.
    common_x_start = pd.Timestamp(df.index.min())
    common_x_end = pd.Timestamp(df.index.max())
    for subplot_row in range(1, 8):
        fig.update_xaxes(
            range=[
                common_x_start,
                common_x_end,
            ],
            autorange=False,
            rangeslider_visible=False,
            row=subplot_row,
            col=1,
        )

    chart_state_text = "|".join(
        [
            ticker, start_date_str, end_date_str, timeframe, setup,
            fast_type, str(fast_period), slow_type, str(slow_period),
            buy_left, buy_operator, buy_right,
            sell_left, sell_operator, sell_right, str(account_size),
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

    show_chart(
        fig,
        use_container_width=True,
        key=f"backtest_chart_{chart_revision}",
        config={"responsive": True, "displaylogo": False},
    )


# ============================================================
# Main application
# ============================================================

if check_password():
    st.write("Welcome to the protected app!")
    keep_settings_across_tabs()
    MOBILE = detect_mobile()

    with st.spinner("Loading the main indexes..."):
        main_index_prices = load_main_indexes()

    # on_change="rerun" makes the tabs lazy: only the open tab runs, and its settings fill the sidebar.
    tabs = st.tabs(TAB_NAMES, key="main_tab", on_change="rerun")
    active_tab = st.session_state.get("main_tab") or TAB_NAMES[0]

    def tab_is_open(tab, label):
        return tab.open if tab.open is not None else label == active_tab

    if not MOBILE:
        st.sidebar.header(active_tab)
    for tab, label in zip(tabs, TAB_NAMES):
        with tab:
            if not tab_is_open(tab, label):
                continue
            if label == "Stock Screener":
                render_stock_screener()
            elif label == "Market Analysis":
                render_market_analysis(main_index_prices)
            elif label == "Industry Analysis":
                render_industry_analysis(main_index_prices)
            elif label == FUNDAMENTALS_TAB:
                render_stock_fundamentals()
            elif label == SUPPLY_CHAIN_TAB:
                render_supply_chain(main_index_prices)
            else:
                strategy = strategy_settings()
                if strategy is None:
                    continue
                if label == "Backtest":
                    render_backtest(strategy)
                elif label == SIGNAL_TAB:
                    render_signal_analysis(strategy.ticker, strategy.end_date, strategy.signal_settings, strategy.risk_settings)
                else:
                    render_validation(strategy.signal_settings, strategy.risk_settings, strategy.end_date)

    with settings_panel("Display", expanded=False):
        st.selectbox(
            "Layout", LAYOUT_MODES, key="layout_mode",
            help="Auto picks the mobile layout on phones and tablets. Desktop puts each tab's settings in the "
            "sidebar; Mobile puts them in collapsible menus above the charts.",
        )
