import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Configuration for Mobile
st.set_page_config(page_title="Swing Backtest", layout="wide", initial_sidebar_state="collapsed")
st.title("📈 Swing Strategy Backtester")


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
    
    st.sidebar.subheader("Buy Signal")
    
    buy_left = st.sidebar.selectbox(
        "Buy Signal A",
        ["Close", "Fast_MA", "Slow_MA"],
        index=1
    )
    
    buy_operator = st.sidebar.selectbox(
        "Buy Operator",
        [
            ">",
            "<",
            ">=",
            "<=",
            "Cross Above",
            "Cross Below"
        ],
        index=4
    )
    
    buy_right = st.sidebar.selectbox(
        "Buy Signal B",
        ["Close", "Fast_MA", "Slow_MA"],
        index=2
    )
    
    st.sidebar.divider()
    
    st.sidebar.subheader("Sell Signal")
    
    sell_left = st.sidebar.selectbox(
        "Sell Signal A",
        ["Close", "Fast_MA", "Slow_MA"],
        index=1
    )
    
    sell_operator = st.sidebar.selectbox(
        "Sell Operator",
        [
            ">",
            "<",
            ">=",
            "<=",
            "Cross Above",
            "Cross Below"
        ],
        index=5
    )
    
    sell_right = st.sidebar.selectbox(
        "Sell Signal B",
        ["Close", "Fast_MA", "Slow_MA"],
        index=2
    )
    
    #######################################################################
    # Run Backtest
    if len(date_range) == 2:
        start_d, end_d = date_range
        df = load_data(ticker, start_d, end_d, timeframe)
    
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
            
                if (not in_trade) and buy_condition.ilocin_trade == True:
                    buy_signal = True
            
                elif in_trade and sell_condition.ilocin_trade == False:
                    sell_signal = True
            
                position.append(
                    1 if in_trade else 0
                )
            
                buy_markers.append(buy_signal)
                sell_markers.append(sell_signal)
            
            df["Position"] = position
            
            # Avoid lookahead bias
            df["Position"] = (
                df["Position"]
                .shift(1)
                .fillna(0)
            )
            
            df["Buy_Signal"] = buy_markers
            df["Sell_Signal"] = sell_markers


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
    
            # Interactive Mobile Plotly Charts
            fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.05,
                subplot_titles=(f"{ticker} Price & Indicators", "Portfolio Equity Curve ($)", "Drawdown Profile (%)"),
                row_heights=[0.5, 0.3, 0.2]
            )
    
            # 1. Price + MAs
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='gray', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Fast_MA'], name=f'{fast_type} {fast_period}', line=dict(color='orange', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Slow_MA'], name=f'{slow_type} {slow_period}', line=dict(color='blue', width=1.5)), row=1, col=1)
            buy_points = df[df["Buy_Signal"]]
            fig.add_trace(go.Scatter(x=buy_points.index, y=buy_points["Close"], mode="markers", name="BUY",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color="lime"
                    )), row=1, col=1)
            
            sell_points = df[df["Sell_Signal"]]
            fig.add_trace(go.Scatter(x=sell_points.index, y=sell_points["Close"], mode="markers", name="SELL",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color="red"
                    )), row=1, col=1)
    
            # 2. Equity Curve
            fig.add_trace(go.Scatter(x=df.index, y=df['Equity'], name='Equity', line=dict(color='green', width=2)), row=2, col=1)
    
            # 3. Drawdown
            fig.add_trace(go.Scatter(x=df.index, y=df['Drawdown'] * 100, name='Drawdown %', fill='tozeroy', line=dict(color='red', width=1)), row=3, col=1)
    
            fig.update_layout(height=750, margin=dict(l=10, r=10, t=40, b=10), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data found for this ticker and date range.")

