import hashlib
from datetime import date

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
    initial_sidebar_state="collapsed"
)

st.title("📈 Trade Strategy Backtester")


# ============================================================
# Password protection
# ============================================================

def check_password():
    """
    Return True when the correct password has been entered.
    """

    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    password = st.text_input(
        "Enter Password",
        type="password"
    )

    if password:
        try:
            correct_password = st.secrets["password"]
        except KeyError:
            st.error(
                "The password is not configured in Streamlit secrets."
            )
            return False

        if password == correct_password:
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("😕 Password incorrect")

    return False


# ============================================================
# Moving-average calculation
# ============================================================

def compute_ma(series, ma_type, period):
    """
    Calculate an exponential or simple moving average.
    """

    period = int(period)

    if ma_type == "EMA":
        return series.ewm(
            span=period,
            adjust=False
        ).mean()

    return series.rolling(
        window=period,
        min_periods=period
    ).mean()


# ============================================================
# Generic signal condition builder
# ============================================================

def build_condition(left, operator, right):
    """
    Build a Boolean condition from two aligned Pandas Series.
    """

    if operator == ">":
        condition = left > right

    elif operator == "<":
        condition = left < right

    elif operator == ">=":
        condition = left >= right

    elif operator == "<=":
        condition = left <= right

    elif operator == "Cross Above":
        condition = (
            (left > right)
            & (left.shift(1) <= right.shift(1))
        )

    elif operator == "Cross Below":
        condition = (
            (left < right)
            & (left.shift(1) >= right.shift(1))
        )

    else:
        condition = pd.Series(
            False,
            index=left.index
        )

    return condition.fillna(False).astype(bool)


# ============================================================
# Price-data loader
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def load_data(
    symbol,
    start_date_str,
    end_date_str,
    timeframe
):
    """
    Download historical price data.

    Yahoo Finance treats the supplied end date as exclusive.
    Therefore, one calendar day is added to the user-selected
    end date before downloading.
    """

    start_timestamp = pd.Timestamp(
        start_date_str
    ).normalize()

    selected_end_timestamp = pd.Timestamp(
        end_date_str
    ).normalize()

    download_end_timestamp = (
        selected_end_timestamp
        + pd.Timedelta(days=1)
    )

    try:
        price_data = yf.download(
            symbol,
            start=start_timestamp.strftime("%Y-%m-%d"),
            end=download_end_timestamp.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if price_data is None or price_data.empty:
            return pd.DataFrame()

        price_data = price_data.copy()

        # yfinance may return MultiIndex columns, even for one ticker.
        if isinstance(price_data.columns, pd.MultiIndex):
            ticker_level = (
                price_data.columns.get_level_values(-1)
            )

            if symbol in ticker_level:
                price_data = price_data.xs(
                    symbol,
                    axis=1,
                    level=-1
                )
            else:
                price_data.columns = (
                    price_data.columns.get_level_values(0)
                )

        price_data.index = pd.to_datetime(
            price_data.index
        )

        if getattr(price_data.index, "tz", None) is not None:
            price_data.index = (
                price_data.index.tz_localize(None)
            )

        price_data = price_data.sort_index()

        price_data = price_data.loc[
            ~price_data.index.duplicated(keep="last")
        ]

        # Explicitly keep only the user-selected period.
        price_data = price_data.loc[
            (
                price_data.index
                >= start_timestamp
            )
            & (
                price_data.index
                <= selected_end_timestamp
            )
        ]

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in price_data.columns
        ]

        if missing_columns:
            return pd.DataFrame()

        price_data = price_data[
            required_columns
        ].copy()

        price_data = price_data.dropna(
            subset=["Close"]
        )

        if price_data.empty:
            return pd.DataFrame()

        if timeframe == "Weekly":
            price_data = (
                price_data
                .resample("W-FRI")
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum"
                    }
                )
                .dropna(subset=["Close"])
            )

            # Remove a weekly label falling after the selected end date.
            price_data = price_data.loc[
                price_data.index
                <= selected_end_timestamp
            ]

        elif timeframe == "Monthly":
            price_data = (
                price_data
                .resample("ME")
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum"
                    }
                )
                .dropna(subset=["Close"])
            )

            # Remove a month-end label after the selected end date.
            price_data = price_data.loc[
                price_data.index
                <= selected_end_timestamp
            ]

        return price_data

    except Exception:
        return pd.DataFrame()


# ============================================================
# Historical quarterly EPS loader
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def load_earnings_data(
    symbol,
    start_date_str,
    end_date_str
):
    """
    Retrieve historical earnings dates, reported EPS,
    analyst EPS estimates and surprise percentages.

    Availability depends on Yahoo Finance and ticker coverage.
    """

    try:
        stock = yf.Ticker(symbol)

        earnings = stock.get_earnings_dates(
            limit=100
        )

        if earnings is None or earnings.empty:
            return pd.DataFrame()

        earnings = earnings.copy()
        earnings = earnings.reset_index()

        date_column = earnings.columns[0]

        earnings[date_column] = pd.to_datetime(
            earnings[date_column],
            errors="coerce",
            utc=True
        ).dt.tz_convert(None)

        earnings = earnings.rename(
            columns={
                date_column: "Earnings_Date"
            }
        )

        for column in [
            "EPS Estimate",
            "Reported EPS",
            "Surprise(%)"
        ]:
            if column in earnings.columns:
                earnings[column] = pd.to_numeric(
                    earnings[column],
                    errors="coerce"
                )

        start_timestamp = pd.Timestamp(
            start_date_str
        ).normalize()

        end_timestamp = pd.Timestamp(
            end_date_str
        ).normalize()

        # Include the complete selected end date.
        end_timestamp_inclusive = (
            end_timestamp
            + pd.Timedelta(days=1)
            - pd.Timedelta(microseconds=1)
        )

        earnings = earnings.loc[
            (
                earnings["Earnings_Date"]
                >= start_timestamp
            )
            & (
                earnings["Earnings_Date"]
                <= end_timestamp_inclusive
            )
        ]

        earnings = earnings.dropna(
            subset=["Earnings_Date"]
        )

        earnings = earnings.sort_values(
            "Earnings_Date"
        )

        earnings = earnings.drop_duplicates(
            subset=["Earnings_Date"],
            keep="last"
        )

        earnings = earnings.set_index(
            "Earnings_Date"
        )

        return earnings

    except Exception:
        return pd.DataFrame()


# ============================================================
# Historical valuation-data builder
# ============================================================

def build_valuation_data(
    price_df,
    earnings_history_df
):
    """
    Calculate:

    1. Trailing P/E:
       Close / rolling four-quarter reported EPS

    2. Forward P/E proxy:
       Close / rolling four-quarter analyst EPS estimates

    The forward P/E calculation is a proxy. It is not a complete
    point-in-time institutional forward P/E history.
    """

    valuation = pd.DataFrame(
        index=price_df.index
    )

    valuation["Close"] = price_df["Close"]

    valuation["TTM_Reported_EPS"] = np.nan
    valuation["TTM_Estimated_EPS"] = np.nan
    valuation["Trailing_PE"] = np.nan
    valuation["Forward_PE_Proxy"] = np.nan

    if (
        earnings_history_df is None
        or earnings_history_df.empty
    ):
        return valuation

    earnings = (
        earnings_history_df
        .copy()
        .sort_index()
    )

    if "Reported EPS" in earnings.columns:
        reported_eps = pd.to_numeric(
            earnings["Reported EPS"],
            errors="coerce"
        )

        earnings["TTM_Reported_EPS"] = (
            reported_eps
            .rolling(
                window=4,
                min_periods=4
            )
            .sum()
        )
    else:
        earnings["TTM_Reported_EPS"] = np.nan

    if "EPS Estimate" in earnings.columns:
        estimated_eps = pd.to_numeric(
            earnings["EPS Estimate"],
            errors="coerce"
        )

        earnings["TTM_Estimated_EPS"] = (
            estimated_eps
            .rolling(
                window=4,
                min_periods=4
            )
            .sum()
        )
    else:
        earnings["TTM_Estimated_EPS"] = np.nan

    eps_columns = earnings[
        [
            "TTM_Reported_EPS",
            "TTM_Estimated_EPS"
        ]
    ].copy()

    # Combine earnings dates and price dates before forward-filling.
    # This ensures that an EPS value before the selected price period
    # can propagate into the selected period.
    combined_index = (
        eps_columns.index
        .union(valuation.index)
        .sort_values()
    )

    earnings_daily = (
        eps_columns
        .reindex(combined_index)
        .ffill()
        .reindex(valuation.index)
    )

    valuation["TTM_Reported_EPS"] = (
        earnings_daily["TTM_Reported_EPS"]
    )

    valuation["TTM_Estimated_EPS"] = (
        earnings_daily["TTM_Estimated_EPS"]
    )

    valuation["Trailing_PE"] = np.where(
        valuation["TTM_Reported_EPS"] > 0,
        (
            valuation["Close"]
            / valuation["TTM_Reported_EPS"]
        ),
        np.nan
    )

    valuation["Forward_PE_Proxy"] = np.where(
        valuation["TTM_Estimated_EPS"] > 0,
        (
            valuation["Close"]
            / valuation["TTM_Estimated_EPS"]
        ),
        np.nan
    )

    valuation = valuation.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return valuation


# ============================================================
# Benchmark-data loader
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def load_benchmark_data(
    start_date_str,
    end_date_str,
    timeframe
):
    """
    Download S&P 500 and Nasdaq-100 history and normalize
    both series to 100.
    """

    benchmark_symbols = [
        "^GSPC",
        "^NDX"
    ]

    start_timestamp = pd.Timestamp(
        start_date_str
    ).normalize()

    selected_end_timestamp = pd.Timestamp(
        end_date_str
    ).normalize()

    download_end_timestamp = (
        selected_end_timestamp
        + pd.Timedelta(days=1)
    )

    try:
        benchmark_data = yf.download(
            benchmark_symbols,
            start=start_timestamp.strftime("%Y-%m-%d"),
            end=download_end_timestamp.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if (
            benchmark_data is None
            or benchmark_data.empty
        ):
            return pd.DataFrame()

        if isinstance(
            benchmark_data.columns,
            pd.MultiIndex
        ):
            benchmark_close = (
                benchmark_data["Close"].copy()
            )
        else:
            benchmark_close = (
                benchmark_data.copy()
            )

        benchmark_close.index = pd.to_datetime(
            benchmark_close.index
        )

        if getattr(
            benchmark_close.index,
            "tz",
            None
        ) is not None:
            benchmark_close.index = (
                benchmark_close.index.tz_localize(None)
            )

        benchmark_close = (
            benchmark_close
            .sort_index()
        )

        benchmark_close = benchmark_close.loc[
            (
                benchmark_close.index
                >= start_timestamp
            )
            & (
                benchmark_close.index
                <= selected_end_timestamp
            )
        ]

        benchmark_close = benchmark_close.rename(
            columns={
                "^GSPC": "S&P 500",
                "^NDX": "Nasdaq-100"
            }
        )

        if timeframe == "Weekly":
            benchmark_close = (
                benchmark_close
                .resample("W-FRI")
                .last()
                .dropna(how="all")
            )

            benchmark_close = benchmark_close.loc[
                benchmark_close.index
                <= selected_end_timestamp
            ]

        elif timeframe == "Monthly":
            benchmark_close = (
                benchmark_close
                .resample("ME")
                .last()
                .dropna(how="all")
            )

            benchmark_close = benchmark_close.loc[
                benchmark_close.index
                <= selected_end_timestamp
            ]

        benchmark_close = (
            benchmark_close
            .dropna(how="all")
        )

        normalized = pd.DataFrame(
            index=benchmark_close.index
        )

        for column in benchmark_close.columns:
            valid_values = (
                benchmark_close[column]
                .dropna()
            )

            if valid_values.empty:
                continue

            first_value = valid_values.iloc[0]

            if first_value == 0:
                continue

            normalized[column] = (
                benchmark_close[column]
                / first_value
                * 100
            )

        return normalized

    except Exception:
        return pd.DataFrame()


# ============================================================
# Position engine
# ============================================================

def run_position_engine(
    price_df,
    buy_condition,
    sell_condition
):
    """
    Generate raw positions and buy/sell markers.

    Signals are generated on the current bar. The actual position
    is shifted by one bar to reduce lookahead bias.
    """

    raw_positions = []
    buy_markers = []
    sell_markers = []

    in_trade = False

    for row_number in range(len(price_df)):
        buy_signal = False
        sell_signal = False

        if (
            not in_trade
            and bool(buy_condition.iloc[row_number])
        ):
            in_trade = True
            buy_signal = True

        elif (
            in_trade
            and bool(sell_condition.iloc[row_number])
        ):
            in_trade = False
            sell_signal = True

        raw_positions.append(
            1 if in_trade else 0
        )

        buy_markers.append(buy_signal)
        sell_markers.append(sell_signal)

    result = price_df.copy()

    result["Raw_Position"] = pd.Series(
        raw_positions,
        index=result.index,
        dtype=int
    )

    result["Buy_Signal"] = pd.Series(
        buy_markers,
        index=result.index,
        dtype=bool
    )

    result["Sell_Signal"] = pd.Series(
        sell_markers,
        index=result.index,
        dtype=bool
    )

    # Signal is observed at the close of one bar.
    # The position becomes active on the following bar.
    result["Position"] = (
        result["Raw_Position"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    return result


# ============================================================
# Main application
# ============================================================

if check_password():
    st.write("Welcome to the protected app!")

    # --------------------------------------------------------
    # Sidebar: general settings
    # --------------------------------------------------------

    st.sidebar.header("Strategy Settings")

    ticker = st.sidebar.text_input(
        "Ticker Symbol",
        value="AAPL"
    )

    timeframe = st.sidebar.selectbox(
        "Timeframe",
        [
            "Daily",
            "Weekly",
            "Monthly"
        ],
        index=0
    )

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(
            date(2022, 1, 1),
            date.today()
        ),
        max_value=date.today(),
        format="YYYY-MM-DD"
    )

    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=1.0,
        value=10000.0,
        step=1000.0
    )

    # --------------------------------------------------------
    # Sidebar: moving averages
    # --------------------------------------------------------

    st.sidebar.subheader("Moving Averages")

    fast_type = st.sidebar.selectbox(
        "Fast MA Type",
        [
            "EMA",
            "SMA"
        ],
        index=0
    )

    fast_period = st.sidebar.number_input(
        "Fast MA Period",
        min_value=1,
        value=20,
        step=1
    )

    slow_type = st.sidebar.selectbox(
        "Slow MA Type",
        [
            "EMA",
            "SMA"
        ],
        index=1
    )

    slow_period = st.sidebar.number_input(
        "Slow MA Period",
        min_value=1,
        value=50,
        step=1
    )

    fast_period = int(fast_period)
    slow_period = int(slow_period)

    # --------------------------------------------------------
    # Sidebar: trade logic
    # --------------------------------------------------------

    signal_options = [
        "Close",
        "Fast_MA",
        "Slow_MA"
    ]

    operator_options = [
        ">",
        "<",
        ">=",
        "<=",
        "Cross Above",
        "Cross Below"
    ]

    st.sidebar.subheader("Buy Signal")

    (
        buy_col_left,
        buy_col_operator,
        buy_col_right
    ) = st.sidebar.columns(
        [1.25, 1.1, 1.25]
    )

    with buy_col_left:
        buy_left = st.selectbox(
            "Buy left signal",
            signal_options,
            index=1,
            key="buy_left",
            label_visibility="collapsed"
        )

    with buy_col_operator:
        buy_operator = st.selectbox(
            "Buy operator",
            operator_options,
            index=4,
            key="buy_operator",
            label_visibility="collapsed"
        )

    with buy_col_right:
        buy_right = st.selectbox(
            "Buy right signal",
            signal_options,
            index=2,
            key="buy_right",
            label_visibility="collapsed"
        )

    st.sidebar.caption(
        f"Buy when: "
        f"`{buy_left} {buy_operator} {buy_right}`"
    )

    st.sidebar.divider()
    st.sidebar.subheader("Sell Signal")

    (
        sell_col_left,
        sell_col_operator,
        sell_col_right
    ) = st.sidebar.columns(
        [1.25, 1.1, 1.25]
    )

    with sell_col_left:
        sell_left = st.selectbox(
            "Sell left signal",
            signal_options,
            index=1,
            key="sell_left",
            label_visibility="collapsed"
        )

    with sell_col_operator:
        sell_operator = st.selectbox(
            "Sell operator",
            operator_options,
            index=5,
            key="sell_operator",
            label_visibility="collapsed"
        )

    with sell_col_right:
        sell_right = st.selectbox(
            "Sell right signal",
            signal_options,
            index=2,
            key="sell_right",
            label_visibility="collapsed"
        )

    st.sidebar.caption(
        f"Sell when: "
        f"`{sell_left} {sell_operator} {sell_right}`"
    )

    # --------------------------------------------------------
    # Validate date input
    # --------------------------------------------------------

    if not isinstance(date_range, (tuple, list)):
        st.info(
            "Please select both a start date and an end date."
        )
        st.stop()

    if len(date_range) != 2:
        st.info(
            "Please select both a start date and an end date."
        )
        st.stop()

    start_date, end_date = date_range

    if start_date > end_date:
        st.error(
            "The start date must be earlier than the end date."
        )
        st.stop()

    ticker = ticker.strip().upper()

    if not ticker:
        st.error("Please enter a ticker symbol.")
        st.stop()

    start_date_str = pd.Timestamp(
        start_date
    ).strftime("%Y-%m-%d")

    end_date_str = pd.Timestamp(
        end_date
    ).strftime("%Y-%m-%d")

    # Load two years of earnings before the displayed period.
    earnings_lookback_start = (
        pd.Timestamp(start_date)
        - pd.DateOffset(years=2)
    ).strftime("%Y-%m-%d")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    with st.spinner(
        f"Loading {ticker} data from "
        f"{start_date_str} to {end_date_str}..."
    ):
        df = load_data(
            ticker,
            start_date_str,
            end_date_str,
            timeframe
        )

        earnings_history_df = load_earnings_data(
            ticker,
            earnings_lookback_start,
            end_date_str
        )

        benchmark_df = load_benchmark_data(
            start_date_str,
            end_date_str,
            timeframe
        )

    if df.empty:
        st.error(
            f"No price data was returned for {ticker} between "
            f"{start_date_str} and {end_date_str}. "
            "The period may contain no completed trading bars, "
            "or Yahoo Finance may not have published the latest bar."
        )
        st.stop()

    st.caption(
        f"Loaded {len(df):,} "
        f"{timeframe.lower()} bars: "
        f"{df.index.min():%Y-%m-%d} to "
        f"{df.index.max():%Y-%m-%d}"
    )

    # --------------------------------------------------------
    # Calculate indicators
    # --------------------------------------------------------

    df["Fast_MA"] = compute_ma(
        df["Close"],
        fast_type,
        fast_period
    )

    df["Slow_MA"] = compute_ma(
        df["Close"],
        slow_type,
        slow_period
    )

    minimum_required_bars = max(
        fast_period,
        slow_period
    )

    if len(df) < minimum_required_bars:
        st.warning(
            f"The selected period contains only {len(df)} bars, "
            f"but the longest moving average uses "
            f"{minimum_required_bars} bars. "
            "Some moving averages and trade signals may be "
            "unavailable. Select an earlier start date or use "
            "shorter moving-average periods."
        )

    valuation_df = build_valuation_data(
        df,
        earnings_history_df
    )

    # Show only earnings events inside the visible period.
    if earnings_history_df.empty:
        earnings_display_df = pd.DataFrame()
    else:
        earnings_display_df = earnings_history_df.loc[
            (
                earnings_history_df.index
                >= pd.Timestamp(start_date)
            )
            & (
                earnings_history_df.index
                < (
                    pd.Timestamp(end_date)
                    + pd.Timedelta(days=1)
                )
            )
        ].copy()

    # --------------------------------------------------------
    # Build buy and sell conditions
    # --------------------------------------------------------

    signal_map = {
        "Close": df["Close"],
        "Fast_MA": df["Fast_MA"],
        "Slow_MA": df["Slow_MA"]
    }

    buy_condition = build_condition(
        signal_map[buy_left],
        buy_operator,
        signal_map[buy_right]
    )

    sell_condition = build_condition(
        signal_map[sell_left],
        sell_operator,
        signal_map[sell_right]
    )

    df = run_position_engine(
        df,
        buy_condition,
        sell_condition
    )

    # --------------------------------------------------------
    # Performance calculations
    # --------------------------------------------------------

    df["Returns"] = (
        df["Close"]
        .pct_change()
        .fillna(0.0)
    )

    df["Strat_Returns"] = (
        df["Position"]
        * df["Returns"]
    )

    df["Equity"] = (
        float(initial_capital)
        * (
            1.0
            + df["Strat_Returns"]
        ).cumprod()
    )

    df["Peak"] = (
        df["Equity"]
        .cummax()
    )

    df["Drawdown"] = (
        df["Equity"]
        / df["Peak"]
        - 1.0
    )

    final_equity = df["Equity"].iloc[-1]

    total_return = (
        final_equity
        / float(initial_capital)
        - 1.0
    ) * 100.0

    max_drawdown = (
        df["Drawdown"].min()
        * 100.0
    )

    active_returns = df.loc[
        df["Strat_Returns"] != 0,
        "Strat_Returns"
    ]

    if active_returns.empty:
        win_rate = 0.0
    else:
        win_rate = (
            active_returns.gt(0).mean()
            * 100.0
        )

    # --------------------------------------------------------
    # KPI display
    # --------------------------------------------------------

    metric_col1, metric_col2, metric_col3 = (
        st.columns(3)
    )

    metric_col1.metric(
        "Final Equity",
        f"${final_equity:,.2f}",
        f"{total_return:.2f}%"
    )

    metric_col2.metric(
        "Max Drawdown",
        f"{max_drawdown:.2f}%"
    )

    metric_col3.metric(
        "Positive Active Bars",
        f"{win_rate:.1f}%"
    )

    st.info(
        f"""
        **BUY:** `{buy_left} {buy_operator} {buy_right}`

        **SELL:** `{sell_left} {sell_operator} {sell_right}`
        """
    )

    # ========================================================
    # Plotly figure
    # ========================================================

    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f"{ticker} Price, Indicators and Trade Signals",
            f"{ticker} Quarterly EPS",
            f"{ticker} Historical and Estimated P/E",
            "Market Benchmark Performance, Normalized to 100",
            "Portfolio Equity Curve ($)",
            "Drawdown Profile (%)"
        ),
        row_heights=[
            0.30,
            0.16,
            0.15,
            0.15,
            0.15,
            0.09
        ]
    )

    # --------------------------------------------------------
    # Row 1: price, moving averages and trade markers
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Close",
            line=dict(
                color="gray",
                width=1
            )
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Fast_MA"],
            name=f"{fast_type} {fast_period}",
            line=dict(
                color="orange",
                width=1.5
            )
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Slow_MA"],
            name=f"{slow_type} {slow_period}",
            line=dict(
                color="blue",
                width=1.5
            )
        ),
        row=1,
        col=1
    )

    buy_points = df.loc[
        df["Buy_Signal"]
    ]

    fig.add_trace(
        go.Scatter(
            x=buy_points.index,
            y=buy_points["Close"],
            mode="markers",
            name="BUY signal",
            marker=dict(
                symbol="triangle-up",
                size=12,
                color="lime",
                line=dict(
                    color="darkgreen",
                    width=1
                )
            ),
            hovertemplate=(
                "BUY signal<br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Signal-bar price: %{y:.2f}"
                "<extra></extra>"
            )
        ),
        row=1,
        col=1
    )

    sell_points = df.loc[
        df["Sell_Signal"]
    ]

    fig.add_trace(
        go.Scatter(
            x=sell_points.index,
            y=sell_points["Close"],
            mode="markers",
            name="SELL signal",
            marker=dict(
                symbol="triangle-down",
                size=12,
                color="red",
                line=dict(
                    color="darkred",
                    width=1
                )
            ),
            hovertemplate=(
                "SELL signal<br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Signal-bar price: %{y:.2f}"
                "<extra></extra>"
            )
        ),
        row=1,
        col=1
    )

    # --------------------------------------------------------
    # Row 2: quarterly EPS
    # --------------------------------------------------------

    eps_trace_added = False

    if not earnings_display_df.empty:
        if (
            "Reported EPS"
            in earnings_display_df.columns
        ):
            reported_eps = (
                earnings_display_df["Reported EPS"]
                .dropna()
            )

            if not reported_eps.empty:
                eps_trace_added = True

                fig.add_trace(
                    go.Bar(
                        x=reported_eps.index,
                        y=reported_eps,
                        name="Reported EPS",
                        marker_color="royalblue",
                        opacity=0.75,
                        hovertemplate=(
                            "Reported EPS<br>"
                            "Date: %{x|%Y-%m-%d}<br>"
                            "EPS: %{y:.3f}"
                            "<extra></extra>"
                        )
                    ),
                    row=2,
                    col=1
                )

        if (
            "EPS Estimate"
            in earnings_display_df.columns
        ):
            estimated_eps = (
                earnings_display_df["EPS Estimate"]
                .dropna()
            )

            if not estimated_eps.empty:
                eps_trace_added = True

                fig.add_trace(
                    go.Scatter(
                        x=estimated_eps.index,
                        y=estimated_eps,
                        mode="lines+markers",
                        name="Analyst EPS Estimate",
                        line=dict(
                            color="orange",
                            width=2,
                            dash="dot"
                        ),
                        marker=dict(
                            symbol="diamond",
                            size=8,
                            color="orange"
                        ),
                        hovertemplate=(
                            "EPS Estimate<br>"
                            "Date: %{x|%Y-%m-%d}<br>"
                            "Estimate: %{y:.3f}"
                            "<extra></extra>"
                        )
                    ),
                    row=2,
                    col=1
                )

    if not eps_trace_added:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="x2 domain",
            yref="y2 domain",
            text=(
                "No quarterly EPS event falls within "
                "the selected display period"
            ),
            showarrow=False,
            font=dict(
                color="gray",
                size=12
            )
        )

    fig.add_hline(
        y=0,
        line_width=1,
        line_dash="dot",
        line_color="gray",
        row=2,
        col=1
    )

    # --------------------------------------------------------
    # Row 3: historical P/E
    # --------------------------------------------------------

    trailing_pe = (
        valuation_df["Trailing_PE"]
        .dropna()
    )

    forward_pe_proxy = (
        valuation_df["Forward_PE_Proxy"]
        .dropna()
    )

    if not trailing_pe.empty:
        fig.add_trace(
            go.Scatter(
                x=trailing_pe.index,
                y=trailing_pe,
                mode="lines",
                name="Trailing P/E",
                line=dict(
                    color="gold",
                    width=2
                ),
                hovertemplate=(
                    "Trailing P/E<br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "P/E: %{y:.2f}x"
                    "<extra></extra>"
                )
            ),
            row=3,
            col=1
        )

    if not forward_pe_proxy.empty:
        fig.add_trace(
            go.Scatter(
                x=forward_pe_proxy.index,
                y=forward_pe_proxy,
                mode="lines",
                name="Forward P/E Proxy",
                line=dict(
                    color="cyan",
                    width=2,
                    dash="dot"
                ),
                hovertemplate=(
                    "Forward P/E Proxy<br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "P/E: %{y:.2f}x"
                    "<extra></extra>"
                )
            ),
            row=3,
            col=1
        )

    if not trailing_pe.empty:
        median_trailing_pe = trailing_pe.median()

        fig.add_hline(
            y=median_trailing_pe,
            line_width=1,
            line_dash="dash",
            line_color="goldenrod",
            annotation_text=(
                f"Median trailing P/E: "
                f"{median_trailing_pe:.1f}x"
            ),
            annotation_position="top left",
            row=3,
            col=1
        )

    if not forward_pe_proxy.empty:
        median_forward_pe = (
            forward_pe_proxy.median()
        )

        fig.add_hline(
            y=median_forward_pe,
            line_width=1,
            line_dash="dash",
            line_color="darkcyan",
            annotation_text=(
                f"Median estimated P/E: "
                f"{median_forward_pe:.1f}x"
            ),
            annotation_position="bottom left",
            row=3,
            col=1
        )

    if (
        trailing_pe.empty
        and forward_pe_proxy.empty
    ):
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="x3 domain",
            yref="y3 domain",
            text=(
                "Insufficient Yahoo Finance EPS history "
                "to calculate P/E"
            ),
            showarrow=False,
            font=dict(
                color="gray",
                size=12
            )
        )

    # --------------------------------------------------------
    # Row 4: benchmarks
    # --------------------------------------------------------

    benchmark_trace_added = False

    if not benchmark_df.empty:
        if "S&P 500" in benchmark_df.columns:
            sp500_data = (
                benchmark_df["S&P 500"]
                .dropna()
            )

            if not sp500_data.empty:
                benchmark_trace_added = True

                fig.add_trace(
                    go.Scatter(
                        x=sp500_data.index,
                        y=sp500_data,
                        name="S&P 500",
                        line=dict(
                            color="deepskyblue",
                            width=2
                        ),
                        hovertemplate=(
                            "S&P 500<br>"
                            "Date: %{x|%Y-%m-%d}<br>"
                            "Normalized value: %{y:.2f}"
                            "<extra></extra>"
                        )
                    ),
                    row=4,
                    col=1
                )

        if "Nasdaq-100" in benchmark_df.columns:
            nasdaq_data = (
                benchmark_df["Nasdaq-100"]
                .dropna()
            )

            if not nasdaq_data.empty:
                benchmark_trace_added = True

                fig.add_trace(
                    go.Scatter(
                        x=nasdaq_data.index,
                        y=nasdaq_data,
                        name="Nasdaq-100",
                        line=dict(
                            color="magenta",
                            width=2
                        ),
                        hovertemplate=(
                            "Nasdaq-100<br>"
                            "Date: %{x|%Y-%m-%d}<br>"
                            "Normalized value: %{y:.2f}"
                            "<extra></extra>"
                        )
                    ),
                    row=4,
                    col=1
                )

    if not benchmark_trace_added:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="x4 domain",
            yref="y4 domain",
            text="Benchmark data is unavailable",
            showarrow=False,
            font=dict(
                color="gray",
                size=12
            )
        )

    fig.add_hline(
        y=100,
        line_width=1,
        line_dash="dot",
        line_color="gray",
        row=4,
        col=1
    )

    # --------------------------------------------------------
    # Row 5: strategy equity
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Equity"],
            name="Strategy Equity",
            line=dict(
                color="green",
                width=2
            ),
            hovertemplate=(
                "Strategy Equity<br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Equity: $%{y:,.2f}"
                "<extra></extra>"
            )
        ),
        row=5,
        col=1
    )

    # --------------------------------------------------------
    # Row 6: drawdown
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Drawdown"] * 100,
            name="Drawdown %",
            fill="tozeroy",
            line=dict(
                color="red",
                width=1
            ),
            hovertemplate=(
                "Drawdown<br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Drawdown: %{y:.2f}%"
                "<extra></extra>"
            )
        ),
        row=6,
        col=1
    )

    # --------------------------------------------------------
    # Axis titles
    # --------------------------------------------------------

    fig.update_yaxes(
        title_text="Price",
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text="EPS",
        row=2,
        col=1
    )

    fig.update_yaxes(
        title_text="P/E (x)",
        rangemode="tozero",
        row=3,
        col=1
    )

    fig.update_yaxes(
        title_text="Indexed",
        row=4,
        col=1
    )

    fig.update_yaxes(
        title_text="Equity ($)",
        row=5,
        col=1
    )

    fig.update_yaxes(
        title_text="DD (%)",
        row=6,
        col=1
    )

    # Use the actual available data range.
    plot_start = df.index.min()
    plot_end = df.index.max()

    fig.update_xaxes(
        range=[
            plot_start,
            plot_end
        ],
        autorange=False,
        rangeslider_visible=False
    )

    # --------------------------------------------------------
    # Generate a stable chart revision and Streamlit key
    # --------------------------------------------------------

    chart_state_text = "|".join(
        [
            ticker,
            start_date_str,
            end_date_str,
            timeframe,
            fast_type,
            str(fast_period),
            slow_type,
            str(slow_period),
            buy_left,
            buy_operator,
            buy_right,
            sell_left,
            sell_operator,
            sell_right,
            str(initial_capital)
        ]
    )

    chart_revision = hashlib.sha256(
        chart_state_text.encode("utf-8")
    ).hexdigest()[:16]

    # --------------------------------------------------------
    # Overall layout
    # --------------------------------------------------------

    fig.update_layout(
        height=1450,
        margin=dict(
            left=20,
            right=20,
            top=50,
            bottom=20
        ),
        showlegend=True,
        hovermode="x unified",
        barmode="group",

        # Change Plotly UI state when input settings change.
        uirevision=chart_revision
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"backtest_chart_{chart_revision}",
        config={
            "responsive": True,
            "displaylogo": False
        }
    )
