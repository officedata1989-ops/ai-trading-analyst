Python

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =====================================
# AI TRADING ANALYST - VERSION 1
# =====================================

st.set_page_config(
    page_title="AI Trading Analyst",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI TRADING ANALYST")
st.subheader("NIFTY 50 | BANK NIFTY | MCX")

st.warning(
    "DEMO MODE: All prices and signals are simulated. "
    "This dashboard does not show live market data."
)

# =====================================
# SIDEBAR SETTINGS
# =====================================

st.sidebar.header("Trading Settings")

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
    "Select Chart Period",
    ["1 Month", "3 Months", "6 Months", "1 Year"]
)

period_days = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365
}

days = period_days[period]

# =====================================
# SIMULATED MARKET DATA
# =====================================

price_map = {
    "NIFTY 50": 26000,
    "BANK NIFTY": 59000,
    "Gold": 150000,
    "Silver": 180000,
    "Crude Oil": 5500,
    "Natural Gas": 300
}

base_price = price_map[instrument]

seed = sum(ord(char) for char in instrument)
rng = np.random.default_rng(seed)

dates = pd.bdate_range(
    end=pd.Timestamp.today().normalize(),
    periods=days
)

returns = rng.normal(
    loc=0.0003,
    scale=0.012,
    size=len(dates)
)

close = base_price * np.exp(
    np.cumsum(returns)
)

open_price = np.concatenate(
    ([close[0]], close[:-1])
)

high = np.maximum(
    open_price, close
) * (1 + rng.uniform(0.001, 0.012, len(dates)))

low = np.minimum(
    open_price, close
) * (1 - rng.uniform(0.001, 0.012, len(dates)))

volume = rng.integers(
    100000, 1000000, len(dates)
)

df = pd.DataFrame(
    {
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    },
    index=dates
)

# =====================================
# EMA INDICATORS
# =====================================

df["EMA_9"] = df["Close"].ewm(
    span=9, adjust=False
).mean()

df["EMA_21"] = df["Close"].ewm(
    span=21, adjust=False
).mean()

df["EMA_50"] = df["Close"].ewm(
    span=50, adjust=False
).mean()

# =====================================
# RSI INDICATOR
# =====================================

delta = df["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(
    alpha=1/14,
    adjust=False,
    min_periods=14
).mean()

avg_loss = loss.ewm(
    alpha=1/14,
    adjust=False,
    min_periods=14
).mean()

rs = avg_gain / avg_loss.replace(0, np.nan)

df["RSI"] = 100 - (100 / (1 + rs))

df.loc[
    (avg_loss == 0) & (avg_gain > 0),
    "RSI"
] = 100

df.loc[
    (avg_loss == 0) & (avg_gain == 0),
    "RSI"
] = 50

# =====================================
# MACD INDICATOR
# =====================================

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

df["Histogram"] = (
    df["MACD"] - df["Signal_Line"]
)

# =====================================
# ATR INDICATOR
# =====================================

previous_close = df["Close"].shift(1)

tr1 = df["High"] - df["Low"]
tr2 = (df["High"] - previous_close).abs()
tr3 = (df["Low"] - previous_close).abs()

true_range = pd.concat(
    [tr1, tr2, tr3],
    axis=1
).max(axis=1)

df["ATR"] = true_range.ewm(
    alpha=1/14,
    adjust=False,
    min_periods=14
).mean()

# =====================================
# MARKET ANALYSIS
# =====================================

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

# =====================================
# ENTRY, STOP LOSS AND TARGETS
# =====================================

if signal == "BUY" and np.isfinite(atr) and atr > 0:

    stop_loss = entry - (1.5 * atr)

    target1 = entry + (1.5 * atr)
    target2 = entry + (3 * atr)
    target3 = entry + (4.5 * atr)

elif signal == "SELL" and np.isfinite(atr) and atr > 0:

    stop_loss = entry + (1.5 * atr)

    target1 = entry - (1.5 * atr)
    target2 = entry - (3 * atr)
    target3 = entry - (4.5 * atr)

else:
    signal = "WAIT"
    stop_loss = None
    target1 = None
    target2 = None
    target3 = None

# =====================================
# DASHBOARD OVERVIEW
# =====================================

st.header("📊 Market Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Simulated Price",
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

# =====================================
# TRADING SIGNAL
# =====================================

st.header("🎯 Trading Signal")

if signal == "BUY":
    st.success("BUY SIGNAL")

elif signal == "SELL":
    st.error("SELL SIGNAL")

else:
    st.warning("WAIT - NO CLEAR SIGNAL")

st.caption(
    "Signals are generated by basic technical rules, "
    "not by a trained AI model."
)

if signal != "WAIT":

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Entry Price",
        f"{entry:,.2f}"
    )

    c2.metric(
        "Stop Loss",
        f"{stop_loss:,.2f}"
    )

    c3.metric(
        "Target 1",
        f"{target1:,.2f}"
    )

    c4, c5 = st.columns(2)

    c4.metric(
        "Target 2",
        f"{target2:,.2f}"
    )

    c5.metric(
        "Target 3",
        f"{target3:,.2f}"
    )

    st.info(
        "Indicative risk-reward ratios: "
        "1:1, 1:2 and 1:3"
    )

else:
    st.info(
        "No clear trade setup. Wait for confirmation."
    )

# =====================================
# CANDLESTICK CHART
# =====================================

st.header("🕯️ Candlestick Chart")

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA_9"],
        name="EMA 9"
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA_21"],
        name="EMA 21"
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA_50"],
        name="EMA 50"
    )
)

fig.update_layout(
    height=500,
    template="plotly_dark",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# RSI CHART
# =====================================

st.header("RSI Indicator")

rsi_fig = go.Figure()

rsi_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["RSI"],
        name="RSI 14"
    )
)

rsi_fig.add_hline(y=70, line_dash="dash")
rsi_fig.add_hline(y=30, line_dash="dash")

rsi_fig.update_layout(
    height=300,
    template="plotly_dark",
    yaxis_range=[0, 100]
)

st.plotly_chart(
    rsi_fig,
    use_container_width=True
)

# =====================================
# MACD CHART
# =====================================

st.header("MACD Indicator")

macd_fig = go.Figure()

macd_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MACD"],
        name="MACD"
    )
)

macd_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Signal_Line"],
        name="Signal Line"
    )
)

macd_fig.add_trace(
    go.Bar(
        x=df.index,
        y=df["Histogram"],
        name="Histogram"
    )
)

macd_fig.update_layout(
    height=300,
    template="plotly_dark"
)

st.plotly_chart(
    macd_fig,
    use_container_width=True
)

# =====================================
# RECENT DATA TABLE
# =====================================

st.header("📋 Recent Market Data")

st.dataframe(
    df.tail(10).round(2),
    use_container_width=True
)

st.divider()

st.caption(
    "AI Trading Analyst V1.0 | "
    "Simulated data only | Not financial advice"
)
