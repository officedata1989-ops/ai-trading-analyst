Python

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------
st.set_page_config(
    page_title="AI Trading Analyst",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI TRADING ANALYST")
st.subheader("NIFTY 50 | BANK NIFTY | MCX")

st.warning(
    "DEMO VERSION: All prices and signals are simulated. "
    "These are NOT live market prices or real trading recommendations."
)

# ---------------------------------
# SIDEBAR
# ---------------------------------
st.sidebar.title("⚙️ Trading Settings")

market = st.sidebar.selectbox(
    "Select Market",
    ["NIFTY 50", "BANK NIFTY", "MCX"]
)

if market == "MCX":
    instrument = st.sidebar.selectbox(
        "Select Commodity",
        ["Gold", "Silver", "Crude Oil", "Natural Gas"]
    )
else:
    instrument = market

period = st.sidebar.selectbox(
    "Chart Period",
    ["1 Month", "3 Months", "6 Months", "1 Year"]
)

period_days = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365
}

days = period_days[period]

# ---------------------------------
# SIMULATED MARKET DATA
# ---------------------------------
price_map = {
    "NIFTY 50": 26000,
    "BANK NIFTY": 59000,
    "Gold": 150000,
    "Silver": 180000,
    "Crude Oil": 5500,
    "Natural Gas": 300
}

base_price = price_map[instrument]

seed = sum(ord(c) for c in instrument) + days
rng = np.random.default_rng(seed)

dates = pd.bdate_range(
    end=pd.Timestamp.today().normalize(),
    periods=days
)

returns = rng.normal(0.0003, 0.012, len(dates))
close = base_price * np.exp(np.cumsum(returns))

open_price = np.concatenate(
    [[close[0]], close[:-1]]
)

high = np.maximum(open_price, close) * (
    1 + rng.uniform(0.001, 0.012, len(dates))
)

low = np.minimum(open_price, close) * (
    1 - rng.uniform(0.001, 0.012, len(dates))
)

volume = rng.integers(
    100000, 1000000, len(dates)
)

df = pd.DataFrame({
    "Open": open_price,
    "High": high,
    "Low": low,
    "Close": close,
    "Volume": volume
}, index=dates)

# ---------------------------------
# TECHNICAL INDICATORS
# ---------------------------------

# EMA
df["EMA_9"] = df["Close"].ewm(
    span=9, adjust=False
).mean()

df["EMA_21"] = df["Close"].ewm(
    span=21, adjust=False
).mean()

df["EMA_50"] = df["Close"].ewm(
    span=50, adjust=False
).mean()

# RSI
delta = df["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(
    alpha=1/14, adjust=False, min_periods=14
).mean()

avg_loss = loss.ewm(
    alpha=1/14, adjust=False, min_periods=14
).mean()

rs = avg_gain / avg_loss.replace(0, np.nan)

df["RSI"] = 100 - (100 / (1 + rs))

df.loc[
    (avg_loss == 0) & (avg_gain > 0), "RSI"
] = 100

df.loc[
    (avg_loss == 0) & (avg_gain == 0), "RSI"
] = 50

# MACD
ema12 = df["Close"].ewm(
    span=12, adjust=False
).mean()

ema26 = df["Close"].ewm(
    span=26, adjust=False
).mean()

df["MACD"] = ema12 - ema26

df["Signal_Line"] = df["MACD"].ewm(
    span=9, adjust=False
).mean()

df["MACD_Histogram"] = (
    df["MACD"] - df["Signal_Line"]
)

# ATR
previous_close = df["Close"].shift(1)

tr = pd.concat([
    df["High"] - df["Low"],
    (df["High"] - previous_close).abs(),
    (df["Low"] - previous_close).abs()
], axis=1).max(axis=1)

df["ATR"] = tr.ewm(
    alpha=1/14, adjust=False
).mean()

# ---------------------------------
# MARKET ANALYSIS
# ---------------------------------

latest = df.iloc[-1]

bullish = (
    latest["EMA_9"] > latest["EMA_21"]
    and latest["MACD"] > latest["Signal_Line"]
    and latest["RSI"] > 50
)

bearish = (
    latest["EMA_9"] < latest["EMA_21"]
    and latest["MACD"] < latest["Signal_Line"]
    and latest["RSI"] < 50
)

if bullish:
    signal = "BUY"
    trend = "BULLISH"
elif bearish:
    signal = "SELL"
    trend = "BEARISH"
else:
    signal = "WAIT"
    trend = "SIDEWAYS / MIXED"

entry = float(latest["Close"])
atr = float(latest["ATR"])

if signal == "BUY":
    stop_loss = entry - (1.5 * atr)
    targets = [
        entry + atr * 1.5,
        entry + atr * 3,
        entry + atr * 4.5
    ]

elif signal == "SELL":
    stop_loss = entry + (1.5 * atr)
    targets = [
        entry - atr * 1.5,
        entry - atr * 3,
        entry - atr * 4.5
    ]

else:
    stop_loss = np.nan
    targets = [np.nan, np.nan, np.nan]

# ---------------------------------
# DASHBOARD METRICS
# ---------------------------------

st.header(f"Market Overview: {instrument}")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Last Simulated Price",
    f"{entry:,.2f}"
)

col2.metric(
    "RSI (14)",
    f"{latest['RSI']:.2f}"
)

col3.metric(
    "MACD",
    f"{latest['MACD']:.2f}"
)

col4.metric(
    "Market Trend",
    trend
)

st.divider()

# ---------------------------------
# TRADING SIGNAL
# ---------------------------------

st.header("🎯 AI Trading Analysis")

if signal == "BUY":
    st.success("BUY SIGNAL – SIMULATED RULE-BASED SIGNAL")
elif signal == "SELL":
    st.error("SELL SIGNAL – SIMULATED RULE-BASED SIGNAL")
else:
    st.warning("WAIT – NO CLEAR SIGNAL")

st.caption(
    "This is a basic technical rules engine, not a trained AI model."
)

if signal != "WAIT":
    c1, c2, c3 = st.columns(3)

    c1.metric("Entry Price", f"{entry:,.2f}")
    c2.metric("Stop Loss", f"{stop_loss:,.2f}")
    c3.metric("Target 1", f"{targets[0]:,.2f}")

    c4, c5 = st.columns(2)

    c4.metric("Target 2", f"{targets[1]:,.2f}")
    c5.metric("Target 3", f"{targets[2]:,.2f}")

    risk = abs(entry - stop_loss)

    st.write(
        f"**Risk / Reward:** "
        f"1 : 1 | 1 : 2 | 1 : 3"
    )

else:
    st.info(
        "No trade setup. Wait for stronger confirmation."
    )

# ---------------------------------
# CANDLESTICK CHART
# ---------------------------------

st.header("📊 Candlestick Chart")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Candlestick"
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA_9"],
    name="EMA 9",
    line=dict(width=1)
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA_21"],
    name="EMA 21",
    line=dict(width=1)
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA_50"],
    name="EMA 50",
    line=dict(width=1)
))

fig.update_layout(
    height=500,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h")
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# RSI CHART
# ---------------------------------

st.header("RSI Indicator")

rsi_fig = go.Figure()

rsi_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["RSI"],
    name="RSI 14"
))

rsi_fig.add_hline(y=70, line_dash="dash")
rsi_fig.add_hline(y=30, line_dash="dash")

rsi_fig.update_layout(
    height=300,
    template="plotly_dark",
    yaxis_range=[0, 100]
)

st.plotly_chart(rsi_fig, use_container_width=True)

# ---------------------------------
# MACD CHART
# ---------------------------------

st.header("MACD Indicator")

macd_fig = go.Figure()

macd_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["MACD"],
    name="MACD"
))

macd_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Signal_Line"],
    name="Signal Line"
))

macd_fig.add_trace(go.Bar(
    x=df.index,
    y=df["MACD_Histogram"],
    name="Histogram"
))

macd_fig.update_layout(
    height=300,
    template="plotly_dark"
)

st.plotly_chart(macd_fig, use_container_width=True)

# ---------------------------------
# LATEST MARKET DATA
# ---------------------------------

st.header("📋 Recent Market Data")

st.dataframe(
    df.tail(10).round(2),
    use_container_width=True
)

st.caption(
    "AI Trading Analyst Version 1.0 | "
    "All market data is simulated for demonstration only."
)
