import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Configuration for Mobile
st.set_page_config(page_title="Swing Backtest", layout="wide", initial_sidebar_state="collapsed")

st.title("📈 Swing Strategy Backtester")

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

# Run Backtest
if len(date_range) == 2:
    start_d, end_d = date_range
    df = load_data(ticker, start_d, end_d, timeframe)

    if not df.empty:
        # Calculate MAs
        df['Fast_MA'] = compute_ma(df['Close'], fast_type, fast_period)
        df['Slow_MA'] = compute_ma(df['Close'], slow_type, slow_period)

        # Signal Logic: Long when Fast MA > Slow MA
        df['Signal'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, 0)
        df['Position'] = df['Signal'].shift(1).fillna(0)  # Avoid lookahead bias

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

        # 2. Equity Curve
        fig.add_trace(go.Scatter(x=df.index, y=df['Equity'], name='Equity', line=dict(color='green', width=2)), row=2, col=1)

        # 3. Drawdown
        fig.add_trace(go.Scatter(x=df.index, y=df['Drawdown'] * 100, name='Drawdown %', fill='tozeroy', line=dict(color='red', width=1)), row=3, col=1)

        fig.update_layout(height=750, margin=dict(l=10, r=10, t=40, b=10), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data found for this ticker and date range.")

