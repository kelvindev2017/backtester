import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Configuration for Mobile
st.set_page_config(page_title="Trade Strategy Backtest", layout="wide", initial_sidebar_state="collapsed")
st.title("📈 Trade Strategy Backtester")


# Helper: Compute MA
def compute_ma(series, ma_type, period):
    if ma_type == "EMA":
        return series.ewm(span=period, adjust=False).mean()
    return series.rolling(window=period).mean()

# Data Loader
@st.cache_data(ttl=3600)
def load_data(symbol, start, end, tf):
    df = yf.download(symbol, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if tf == "Weekly":
        df = df.resample("W-FRI").agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    elif tf == "Monthly":
        df = df.resample("ME").agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    return df

# Check password
def check_password():
    """Returns `True` if the user had the correct password."""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    # Show input for password
    password = st.text_input("Enter Password", type="password")
    if password:
        if password == st.secrets["password"]: # Store password in secrets.toml
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("😕 Password incorrect")
    return False

# Generic condition builder
def build_condition(left, operator, right):

    if operator == ">":
        return left > right

    elif operator == "<":
        return left < right

    elif operator == ">=":
        return left >= right

    elif operator == "<=":
        return left <= right

    elif operator == "Cross Above":
        return (
            (left > right)
            &
            (left.shift(1) <= right.shift(1))
        )

    elif operator == "Cross Below":
        return (
            (left < right)
            &
            (left.shift(1) >= right.shift(1))
        )

    return pd.Series(False, index=left.index)


# ============================================================
# Load historical quarterly EPS
# ============================================================

@st.cache_data(ttl=3600)
def load_earnings_data(symbol, start, end):
    """
    Retrieves historical earnings dates, reported EPS,
    analyst EPS estimates, and surprise percentage.

    Availability depends on Yahoo Finance and the selected ticker.
    """

    try:
        stock = yf.Ticker(symbol)

        earnings = stock.get_earnings_dates(limit=100)

        if earnings is None or earnings.empty:
            return pd.DataFrame()

        earnings = earnings.copy()

        # Convert the earnings-date index into a normal column
        earnings = earnings.reset_index()

        # The first column is normally named "Earnings Date"
        date_column = earnings.columns[0]

        earnings[date_column] = pd.to_datetime(
            earnings[date_column],
            errors="coerce",
            utc=True
        ).dt.tz_convert(None)

        earnings = earnings.rename(
            columns={date_column: "Earnings_Date"}
        )

        # Make sure EPS columns are numeric
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

        start_timestamp = pd.Timestamp(start)
        end_timestamp = pd.Timestamp(end)

        # Keep earnings within the selected date range
        earnings = earnings[
            (earnings["Earnings_Date"] >= start_timestamp)
            & (earnings["Earnings_Date"] <= end_timestamp)
        ]

        earnings = earnings.sort_values("Earnings_Date")

        earnings = earnings.set_index("Earnings_Date")

        return earnings

    except Exception:
        return pd.DataFrame()


# ============================================================
# Load S&P 500 and Nasdaq-100 data
# ============================================================

@st.cache_data(ttl=3600)
def load_benchmark_data(start, end, tf):
    """
    Downloads S&P 500 and Nasdaq-100 index history
    and normalizes both series to 100.
    """

    benchmark_symbols = ["^GSPC", "^NDX"]

    try:
        benchmark_data = yf.download(
            benchmark_symbols,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False
        )

        if benchmark_data is None or benchmark_data.empty:
            return pd.DataFrame()

        # Extract Close from the MultiIndex structure
        if isinstance(benchmark_data.columns, pd.MultiIndex):
            benchmark_close = benchmark_data["Close"].copy()
        else:
            benchmark_close = benchmark_data.copy()

        benchmark_close = benchmark_close.rename(
            columns={
                "^GSPC": "S&P 500",
                "^NDX": "Nasdaq-100"
            }
        )

        # Match the selected timeframe
        if tf == "Weekly":
            benchmark_close = (
                benchmark_close
                .resample("W-FRI")
                .last()
                .dropna(how="all")
            )

        elif tf == "Monthly":
            benchmark_close = (
                benchmark_close
                .resample("ME")
                .last()
                .dropna(how="all")
            )

        # Remove rows for which both indices are unavailable
        benchmark_close = benchmark_close.dropna(how="all")

        # Normalize each index to 100
        normalized = pd.DataFrame(
            index=benchmark_close.index
        )

        for column in benchmark_close.columns:
            valid_values = benchmark_close[column].dropna()

            if not valid_values.empty:
                first_value = valid_values.iloc[0]

                normalized[column] = (
                    benchmark_close[column]
                    / first_value
                    * 100
                )

        return normalized

    except Exception:
        return pd.DataFrame()


#############################################
# Main()
#############################################
if check_password():
    st.write("Welcome to the protected app!")
    # Sidebar Controls (Collapsible on Mobile)
    st.sidebar.header("Strategy Settings")
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    timeframe = st.sidebar.selectbox("Timeframe", ["Daily", "Weekly", "Monthly"], index=0)
    date_range = st.sidebar.date_input("Date Range", [pd.to_datetime("2022-01-01"), pd.to_datetime("2026-01-01")])
    initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000)
    
    st.sidebar.subheader("Moving Averages")
    fast_type = st.sidebar.selectbox("Fast MA Type", ["EMA", "SMA"], index=0)
    fast_period = st.sidebar.number_input("Fast MA Period", value=20, min_value=1)
    
    slow_type = st.sidebar.selectbox("Slow MA Type", ["EMA", "SMA"], index=1)
    slow_period = st.sidebar.number_input("Slow MA Period", value=50, min_value=1)

    #st.sidebar.subheader("Trade Logic")
    #logic_type = st.sidebar.selectbox("Entry Condition",
    #    [
    #        ">",
    #        "<",
    #        ">=",
    #        "<=",
    #        "Cross Above",
    #        "Cross Below"
    #    ],
    #    index=0
    #)

    # ===================================
    # Trade Logic
    # ===================================
    
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
    
    # -----------------------------------
    # Buy Signal
    # -----------------------------------
    
    st.sidebar.subheader("Buy Signal")
    
    buy_col_left, buy_col_operator, buy_col_right = st.sidebar.columns(
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
    
    # Show the selected buy rule clearly
    st.sidebar.caption(
        f"Buy when: `{buy_left} {buy_operator} {buy_right}`"
    )
    
    st.sidebar.divider()
    
    # -----------------------------------
    # Sell Signal
    # -----------------------------------
    
    st.sidebar.subheader("Sell Signal")
    
    sell_col_left, sell_col_operator, sell_col_right = st.sidebar.columns(
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
    
    # Show the selected sell rule clearly
    st.sidebar.caption(
        f"Sell when: `{sell_left} {sell_operator} {sell_right}`"
    )
    
    #######################################################################
    # Run Backtest
    if len(date_range) == 2:
        start_d, end_d = date_range
        df = load_data(ticker, start_d, end_d, timeframe)
        earnings_df = load_earnings_data(ticker, start_d, end_d)
        benchmark_df = load_benchmark_data(start_d, end_d, timeframe)
    
        if not df.empty:
            # Calculate MAs
            df['Fast_MA'] = compute_ma(df['Close'], fast_type, fast_period)
            df['Slow_MA'] = compute_ma(df['Close'], slow_type, slow_period)

            # Signal Logic: Long when Fast MA > Slow MA
            #df['Signal'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, 0)

            # ===================================
            # Signal Mapping
            # ===================================
            
            signal_map = {
                "Close": df["Close"],
                "Fast_MA": df["Fast_MA"],
                "Slow_MA": df["Slow_MA"]
            }
            
            # ===================================
            # Buy Condition
            # ===================================
            
            buy_condition = build_condition(
                signal_map[buy_left],
                buy_operator,
                signal_map[buy_right]
            )
            
            # ===================================
            # Sell Condition
            # ===================================
            
            sell_condition = build_condition(
                signal_map[sell_left],
                sell_operator,
                signal_map[sell_right]
            )
            
            # ===================================
            # Position Engine
            # ===================================
            
            position = []
            buy_markers = []
            sell_markers = []
            
            in_trade = False
            
            for i in range(len(df)):
            
                buy_signal = False
                sell_signal = False
            
                # Enter a long position when the buy condition is True
                if (not in_trade) and bool(buy_condition.iloc[i]):
                    in_trade = True
                    buy_signal = True
            
                # Exit the long position when the sell condition is True
                elif in_trade and bool(sell_condition.iloc[i]):
                    in_trade = False
                    sell_signal = True
            
                position.append(1 if in_trade else 0)
                buy_markers.append(buy_signal)
                sell_markers.append(sell_signal)
            
            df["Raw_Position"] = pd.Series(position, index=df.index)
            df["Buy_Signal"] = pd.Series(buy_markers, index=df.index)
            df["Sell_Signal"] = pd.Series(sell_markers, index=df.index)
            
            # Execute each signal on the following bar to avoid lookahead bias
            df["Position"] = df["Raw_Position"].shift(1).fillna(0).astype(int)
            
            #df["Buy_Signal"] = buy_markers
            #df["Sell_Signal"] = sell_markers


            ###############################################################################    
            
    
            # Performance Calculations
            df['Returns'] = df['Close'].pct_change()
            df['Strat_Returns'] = df['Position'] * df['Returns']
            df['Equity'] = initial_capital * (1 + df['Strat_Returns']).cumprod()
            df['Peak'] = df['Equity'].cummax()
            df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
    
            # Key Metrics Summary
            total_return = ((df['Equity'].iloc[-1] - initial_capital) / initial_capital) * 100
            max_dd = df['Drawdown'].min() * 100
            win_rate = (df['Strat_Returns'] > 0).sum() / (df['Strat_Returns'] != 0).sum() * 100 if (df['Strat_Returns'] != 0).sum() > 0 else 0
    
            # KPI Display Cards
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Equity", f"${df['Equity'].iloc[-1]:,.2f}", f"{total_return:.2f}%")
            col2.metric("Max Drawdown", f"{max_dd:.2f}%")
            col3.metric("Win Rate", f"{win_rate:.1f}%")

            st.info(
                f"""
                BUY : {buy_left} {buy_operator} {buy_right}
            
                SELL : {sell_left} {sell_operator} {sell_right}
                """
            )
    
            # ============================================================
            # Interactive Plotly Charts
            # ============================================================
            
            fig = make_subplots(
                rows=5,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.035,
                subplot_titles=(
                    f"{ticker} Price, Indicators and Trade Signals",
                    f"{ticker} Quarterly EPS",
                    "Market Benchmark Performance, Normalized to 100",
                    "Portfolio Equity Curve ($)",
                    "Drawdown Profile (%)"
                ),
                row_heights=[
                    0.36,
                    0.17,
                    0.17,
                    0.18,
                    0.12
                ]
            )
            
            # ============================================================
            # Row 1: Price, moving averages and trade markers
            # ============================================================
            
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
            
            # Actual entry markers generated by the position engine
            buy_points = df[df["Buy_Signal"]]
            
            fig.add_trace(
                go.Scatter(
                    x=buy_points.index,
                    y=buy_points["Close"],
                    mode="markers",
                    name="BUY",
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
                        "BUY<br>"
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Price: %{y:.2f}"
                        "<extra></extra>"
                    )
                ),
                row=1,
                col=1
            )
            
            # Actual exit markers generated by the position engine
            sell_points = df[df["Sell_Signal"]]
            
            fig.add_trace(
                go.Scatter(
                    x=sell_points.index,
                    y=sell_points["Close"],
                    mode="markers",
                    name="SELL",
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
                        "SELL<br>"
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Price: %{y:.2f}"
                        "<extra></extra>"
                    )
                ),
                row=1,
                col=1
            )
            
            # ============================================================
            # Row 2: Historical reported and estimated quarterly EPS
            # ============================================================
            
            if not earnings_df.empty:
            
                if "Reported EPS" in earnings_df.columns:
                    reported_eps = earnings_df["Reported EPS"].dropna()
            
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
            
                if "EPS Estimate" in earnings_df.columns:
                    estimated_eps = earnings_df["EPS Estimate"].dropna()
            
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
            
            # If EPS data is unavailable, add an annotation
            else:
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref="x2 domain",
                    yref="y2 domain",
                    text="Quarterly EPS data is unavailable for this ticker",
                    showarrow=False,
                    font=dict(
                        color="gray",
                        size=12
                    )
                )
            
            # Add horizontal zero line for positive/negative EPS
            fig.add_hline(
                y=0,
                line_width=1,
                line_dash="dot",
                line_color="gray",
                row=2,
                col=1
            )
            
            # ============================================================
            # Row 3: S&P 500 and Nasdaq-100
            # ============================================================
            
            if not benchmark_df.empty:
            
                if "S&P 500" in benchmark_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=benchmark_df.index,
                            y=benchmark_df["S&P 500"],
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
                        row=3,
                        col=1
                    )
            
                if "Nasdaq-100" in benchmark_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=benchmark_df.index,
                            y=benchmark_df["Nasdaq-100"],
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
                        row=3,
                        col=1
                    )
            
            else:
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref="x3 domain",
                    yref="y3 domain",
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
                row=3,
                col=1
            )
            
            # ============================================================
            # Row 4: Strategy equity
            # ============================================================
            
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
                row=4,
                col=1
            )
            
            # ============================================================
            # Row 5: Drawdown
            # ============================================================
            
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
                row=5,
                col=1
            )
            
            # ============================================================
            # Axis titles
            # ============================================================
            
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
                title_text="Indexed",
                row=3,
                col=1
            )
            
            fig.update_yaxes(
                title_text="Equity ($)",
                row=4,
                col=1
            )
            
            fig.update_yaxes(
                title_text="DD (%)",
                row=5,
                col=1
            )
            
            fig.update_xaxes(
                rangeslider_visible=False
            )
            
            # ============================================================
            # Overall chart layout
            # ============================================================
            
            fig.update_layout(
                height=1250,
                margin=dict(
                    l=20,
                    r=20,
                    t=50,
                    b=20
                ),
                showlegend=True,
                hovermode="x unified",
                barmode="group"
            )
            
            st.plotly_chart(
                fig,
                use_container_width=True
            )
    else:
        st.error("No data found for this ticker and date range.")

