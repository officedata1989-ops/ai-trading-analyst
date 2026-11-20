import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(
    page_title="AI Trading Analyst",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI TRADING ANALYST V2")
st.caption("Historical Market Analysis | Stock Scanner | Backtesting")

st.warning(
    "Yahoo Finance historical data. Not a live NSE feed. "
    "Signals are rule-based and are not guaranteed trading recommendations."
)

# ---------------------------------
# NIFTY 50 STOCK SYMBOLS
# ---------------------------------

STOCKS = {
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Bharat Electronics": "BEL.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Cipla": "CIPLA.NS",
    "Coal India": "COALINDIA.NS",
    "Dr Reddys": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Eternal": "ETERNAL.NS",
    "Grasim": "GRASIM.NS",
    "HCL Technologies": "HCLTECH.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Hindalco": "HINDALCO.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Infosys": "INFY.NS",
    "ITC": "ITC.NS",
    "Jio Financial": "JIOFIN.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Larsen and Toubro": "LT.NS",
    "Mahindra and Mahindra": "M&M.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Max Healthcare": "MAXHEALTH.NS",
    "Nestle India": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "Power Grid": "POWERGRID.NS",
    "Reliance": "RELIANCE.NS",
    "SBI Life": "SBILIFE.NS",
    "State Bank of India": "SBIN.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Tata Steel": "TATASTEEL.NS",
    "TCS": "TCS.NS",
    "Tech Mahindra": "TECHM.NS",
    "Titan": "TITAN.NS",
    "Trent": "TRENT.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Wipro": "WIPRO.NS"
}

# ---------------------------------
# DATA DOWNLOAD
# ---------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_data(symbol, period):
    try:
        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]
        df = df[required].copy()
        df = df.dropna()

        return df

    except Exception:
        return pd.DataFrame()


# ---------------------------------
# TECHNICAL INDICATORS
# ---------------------------------

def add_indicators(df):
    df = df.copy()

    df["EMA9"] = df["Close"].ewm(
        span=9, adjust=False
    ).mean()

    df["EMA21"] = df["Close"].ewm(
        span=21, adjust=False
    ).mean()

    df["EMA50"] = df["Close"].ewm(
        span=50, adjust=False
    ).mean()

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

    ema12 = df["Close"].ewm(
        span=12, adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26, adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    return df


# ---------------------------------
# CANDLESTICK PATTERNS
# ---------------------------------

def add_patterns(df):
    df = df.copy()

    body = (df["Close"] - df["Open"]).abs()

    candle_range = (
        df["High"] - df["Low"]
    ).replace(0, np.nan)

    upper = (
        df["High"] -
        df[["Open", "Close"]].max(axis=1)
    )

    lower = (
        df[["Open", "Close"]].min(axis=1) -
        df["Low"]
    )

    df["Pattern"] = "Normal Candle"

    doji = body <= candle_range * 0.10

    hammer = (
        (lower >= body * 2) &
        (upper <= body) &
        (body > 0)
    )

    shooting_star = (
        (upper >= body * 2) &
        (lower <= body) &
        (body > 0)
    )

    df.loc[doji, "Pattern"] = "Doji"
    df.loc[hammer, "Pattern"] = "Hammer"
    df.loc[
        shooting_star, "Pattern"
    ] = "Shooting Star"

    prev_open = df["Open"].shift(1)
    prev_close = df["Close"].shift(1)

    bullish_engulfing = (
        (prev_close < prev_open) &
        (df["Close"] > df["Open"]) &
        (df["Open"] <= prev_close) &
        (df["Close"] >= prev_open)
    )

    bearish_engulfing = (
        (prev_close > prev_open) &
        (df["Close"] < df["Open"]) &
        (df["Open"] >= prev_close) &
        (df["Close"] <= prev_open)
    )

    df.loc[
        bullish_engulfing, "Pattern"
    ] = "Bullish Engulfing"

    df.loc[
        bearish_engulfing, "Pattern"
    ] = "Bearish Engulfing"

    return df


# ---------------------------------
# BUY / SELL / WAIT RULES
# ---------------------------------

def add_signals(df):
    df = df.copy()

    buy = (
        (df["EMA9"] > df["EMA21"]) &
        (df["MACD"] > df["MACD_SIGNAL"]) &
        (df["RSI"] > 50)
    )

    sell = (
        (df["EMA9"] < df["EMA21"]) &
        (df["MACD"] < df["MACD_SIGNAL"]) &
        (df["RSI"] < 50)
    )

    df["Signal"] = "WAIT"
    df.loc[buy, "Signal"] = "BUY"
    df.loc[sell, "Signal"] = "SELL"

    return df


# ---------------------------------
# BACKTESTING ENGINE
# ---------------------------------

def run_backtest(df, capital, cost_pct):

    data = df.copy()

    # Signal at closing price.
    # Position starts from the following session.
    data["Position"] = (
        data["Signal"]
        .map({"BUY": 1, "SELL": 0, "WAIT": 0})
        .shift(1)
        .fillna(0)
    )

    data["Daily_Return"] = (
        data["Close"].pct_change().fillna(0)
    )

    data["Turnover"] = (
        data["Position"].diff().abs()
    )

    data["Turnover"] = (
        data["Turnover"].fillna(
            data["Position"].abs()
        )
    )

    data["Strategy_Return"] = (
        data["Position"] * data["Daily_Return"]
        - data["Turnover"] * cost_pct / 100
    )

    data["Equity"] = (
        capital *
        (1 + data["Strategy_Return"]).cumprod()
    )

    data["Buy_Hold"] = (
        capital *
        (1 + data["Daily_Return"]).cumprod()
    )

    # Closed trades and their returns
    trade_returns = []
    in_trade = False
    trade_return = 0.0

    for pos, ret in zip(
        data["Position"],
        data["Strategy_Return"]
    ):

        if pos == 1:
            trade_return += ret
            in_trade = True

        elif in_trade:
            trade_returns.append(trade_return)
            trade_return = 0.0
            in_trade = False

    if in_trade:
        trade_returns.append(trade_return)

    completed = len(trade_returns)

    wins = sum(
        r > 0 for r in trade_returns
    )

    win_rate = (
        wins / completed * 100
        if completed else 0
    )

    net_profit = (
        data["Equity"].iloc[-1] - capital
    )

    peak = data["Equity"].cummax()

    drawdown = (
        data["Equity"] / peak - 1
    ) * 100

    max_drawdown = drawdown.min()

    return (
        data,
        completed,
        win_rate,
        net_profit,
        max_drawdown
    )


# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.header("Trading Settings")

section = st.sidebar.selectbox(
    "Select Section",
    [
        "Single Stock Analysis",
        "Nifty 50 Scanner",
        "Backtesting"
    ]
)

period = st.sidebar.selectbox(
    "Historical Period",
    ["1 Year", "2 Years", "5 Years"]
)

period_map = {
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y"
}

selected_period = period_map[period]

stock_name = st.sidebar.selectbox(
    "Select Nifty 50 Stock",
    list(STOCKS.keys())
)

symbol = STOCKS[stock_name]


# ---------------------------------
# SINGLE STOCK ANALYSIS
# ---------------------------------

if section == "Single Stock Analysis":

    st.header(stock_name)

    with st.spinner("Downloading historical data..."):

        df = get_data(symbol, selected_period)

    if df.empty:
        st.error(
            "Data unavailable. Please try again later "
            "or select another stock."
        )
        st.stop()

    df = add_indicators(df)
    df = add_patterns(df)
    df = add_signals(df)

    latest = df.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Latest Close",
        f"₹{latest['Close']:,.2f}"
    )

    c2.metric(
        "RSI",
        f"{latest['RSI']:.2f}"
    )

    c3.metric(
        "MACD",
        f"{latest['MACD']:.2f}"
    )

    c4.metric(
        "Signal",
        latest["Signal"]
    )

    st.subheader("Latest Candlestick Pattern")

    st.info(
        f"{latest['Pattern']} | "
        f"Date: {df.index[-1].date()}"
    )

    if latest["Signal"] == "BUY":
        st.success("BUY setup according to selected rules")

    elif latest["Signal"] == "SELL":
        st.error("SELL / EXIT setup according to selected rules")

    else:
        st.warning("WAIT - No clear setup")

    # Candlestick chart

    st.subheader("Candlestick Chart")

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

    for col in ["EMA9", "EMA21", "EMA50"]:

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                name=col
            )
        )

    fig.update_layout(
        height=600,
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # RSI

    st.subheader("RSI")

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

    st.subheader("Recent Market Data")

    st.dataframe(
        df.tail(20).round(2),
        use_container_width=True
    )


# ---------------------------------
# NIFTY 50 SCANNER
# ---------------------------------

elif section == "Nifty 50 Scanner":

    st.header("Nifty 50 - All Stock Scanner")

    st.caption(
        "Scanning the latest available daily candles "
        "for the listed 50 symbols."
    )

    if st.button("Run Nifty 50 Scan"):

        results = []

        progress = st.progress(0)

        for i, (name, ticker) in enumerate(
            STOCKS.items()
        ):

            data = get_data(ticker, "1y")

            if not data.empty and len(data) >= 60:

                data = add_indicators(data)
                data = add_patterns(data)
                data = add_signals(data)

                last = data.iloc[-1]

                results.append({
                    "Stock": name,
                    "Close": round(float(last["Close"]), 2),
                    "RSI": round(float(last["RSI"]), 2),
                    "Pattern": last["Pattern"],
                    "Signal": last["Signal"],
                    "Date": str(data.index[-1].date())
                })

            progress.progress(
                (i + 1) / len(STOCKS)
            )

        if results:

            result_df = pd.DataFrame(results)

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Signal Summary")

            a, b, c = st.columns(3)

            a.metric(
                "BUY",
                int((result_df["Signal"] == "BUY").sum())
            )

            b.metric(
                "SELL",
                int((result_df["Signal"] == "SELL").sum())
            )

            c.metric(
                "WAIT",
                int((result_df["Signal"] == "WAIT").sum())
            )

        else:
            st.error(
                "No stock data could be downloaded. "
                "Please retry later."
            )


# ---------------------------------
# BACKTESTING
# ---------------------------------

elif section == "Backtesting":

    st.header("Strategy Backtesting")

    st.write(
        "Test the EMA 9/21 + MACD + RSI strategy "
        "on historical daily stock data."
    )

    capital = st.number_input(
        "Starting Capital (₹)",
        min_value=1000,
        value=100000,
        step=10000
    )

    cost_pct = st.number_input(
        "Estimated trading cost per position change (%)",
        min_value=0.0,
        max_value=2.0,
        value=0.10,
        step=0.05
    )

    if st.button("Run Backtest"):

        with st.spinner("Running backtest..."):

            df = get_data(symbol, selected_period)

            if df.empty or len(df) < 60:

                st.error("Not enough historical data.")

            else:

                df = add_indicators(df)
                df = add_patterns(df)
                df = add_signals(df)

                (
                    test,
                    total_trades,
                    win_rate,
                    net_profit,
                    max_drawdown
                ) = run_backtest(
                    df,
                    capital,
                    cost_pct
                )

                st.subheader(
                    f"Backtest Results: {stock_name}"
                )

                a, b, c = st.columns(3)

                a.metric(
                    "Net Profit / Loss",
                    f"₹{net_profit:,.2f}"
                )

                b.metric(
                    "Win Rate",
                    f"{win_rate:.2f}%"
                )

                c.metric(
                    "Completed Trades",
                    total_trades
                )

                d, e, f = st.columns(3)

                d.metric(
                    "Final Capital",
                    f"₹{test['Equity'].iloc[-1]:,.2f}"
                )

                e.metric(
                    "Maximum Drawdown",
                    f"{max_drawdown:.2f}%"
                )

                f.metric(
                    "Buy & Hold Capital",
                    f"₹{test['Buy_Hold'].iloc[-1]:,.2f}"
                )

                st.subheader("Equity Curve")

                equity_fig = go.Figure()

                equity_fig.add_trace(
                    go.Scatter(
                        x=test.index,
                        y=test["Equity"],
                        name="Strategy"
                    )
                )

                equity_fig.add_trace(
                    go.Scatter(
                        x=test.index,
                        y=test["Buy_Hold"],
                        name="Buy & Hold"
                    )
                )

                equity_fig.update_layout(
                    height=450,
                    template="plotly_dark"
                )

                st.plotly_chart(
                    equity_fig,
                    use_container_width=True
                )

                st.subheader("Historical Trade Data")

                st.dataframe(
                    test[
                        [
                            "Close",
                            "RSI",
                            "MACD",
                            "Signal",
                            "Position",
                            "Strategy_Return",
                            "Equity"
                        ]
                    ].tail(30).round(3),
                    use_container_width=True
                )

                st.caption(
                    "This is an educational daily-data backtest. "
                    "It uses a simplified long-only strategy. "
                    "Costs are approximate; taxes, exact brokerage, "
                    "market impact and all execution effects may differ."
                )

# ---------------------------------
# FOOTER
# ---------------------------------

st.divider()

st.caption(
    "AI Trading Analyst V2 | Historical data via Yahoo Finance | "
    "Educational analysis only"
)
