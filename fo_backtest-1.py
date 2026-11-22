"""F&O CSV Backtesting module. Add to app.py:
from fo_backtest import render_fo_backtest
render_fo_backtest()
CSV columns: date,underlying,instrument,expiry,option_type,strike,symbol,open,high,low,close,volume,lot_size
FUT rows: instrument=FUT, option_type=NA, strike=0. OPT rows: instrument=OPT, option_type=CE/PE.
Use only historical data you are authorized to use.
"""
import pandas as pd
import streamlit as st

REQUIRED = ["date","underlying","instrument","expiry","option_type","strike","symbol",
            "open","high","low","close","volume","lot_size"]
UNDERLYINGS = ["NIFTY","BANKNIFTY","FINNIFTY"]

def clean_csv(uploaded):
    df = pd.read_csv(uploaded)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    df = df[REQUIRED].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
    for c in ["strike","open","high","low","close","volume","lot_size"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["underlying","instrument","option_type","symbol"]:
        df[c] = df[c].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["date","expiry","strike","open","high","low","close","volume","lot_size"])
    df = df[(df.close > 0) & (df.lot_size > 0) & df.underlying.isin(UNDERLYINGS)]
    df = df[df.instrument.isin(["FUT","OPT"])]
    df = df[(df.instrument == "FUT") | df.option_type.isin(["CE","PE"])]
    df = df.sort_values(["symbol","date"]).drop_duplicates(["symbol","date"], keep="last")
    if df.empty:
        raise ValueError("No valid contract rows found.")
    return df

def render_fo_backtest():
    st.header("📊 F&O CSV Backtesting")
    st.warning("Historical research only. Upload only CSV data you are authorized to use. "
               "This is a daily-close backtest, not a live feed or trading recommendation.")
    template = pd.DataFrame(columns=REQUIRED)
    st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8"),
                       "fo_backtest_template.csv", "text/csv")
    uploaded = st.file_uploader("Upload historical F&O contract CSV", type=["csv"], key="fo_csv_upload")
    if uploaded is None:
        st.info("Upload a contract-wise CSV to begin. Options require actual historical option-premium OHLC.")
        return
    try:
        df = clean_csv(uploaded)
    except Exception as e:
        st.error(f"CSV validation failed: {e}")
        return
    st.success(f"Loaded {len(df):,} valid daily rows across {df.symbol.nunique()} contracts.")
    a,b,c = st.columns(3)
    a.metric("First date", df.date.min().date().isoformat())
    b.metric("Last date", df.date.max().date().isoformat())
    c.metric("Contracts", int(df.symbol.nunique()))

    underlying = st.selectbox("Index", UNDERLYINGS)
    kind = st.radio("Instrument", ["Futures","Options"], horizontal=True)
    sub = df[(df.underlying == underlying) & (df.instrument == ("FUT" if kind == "Futures" else "OPT"))]
    if kind == "Options":
        opt = st.radio("Option type", ["CE","PE"], horizontal=True)
        sub = sub[sub.option_type == opt]
        strikes = sorted(sub.strike.unique().tolist())
        if not strikes:
            st.error("No matching option contracts in this CSV.")
            return
        strike = st.selectbox("Strike", strikes)
        sub = sub[sub.strike == strike]
    if sub.empty:
        st.error("No matching contracts in this CSV.")
        return
    expiries = sorted(sub.expiry.unique())
    expiry = st.selectbox("Expiry", expiries, format_func=lambda x: pd.Timestamp(x).date().isoformat())
    sub = sub[sub.expiry == pd.Timestamp(expiry)]
    symbols = sorted(sub.symbol.unique())
    symbol = st.selectbox("Contract", symbols)
    contract = sub[sub.symbol == symbol].sort_values("date")
    first_date, last_date = contract.date.min().date(), contract.date.max().date()
    x,y = st.columns(2)
    entry_date = x.date_input("Entry date", first_date, min_value=first_date, max_value=last_date)
    exit_date = y.date_input("Last date to hold", last_date, min_value=first_date, max_value=last_date)
    p,q = st.columns(2)
    side = p.selectbox("Position", ["BUY","SELL"])
    lots = p.number_input("Lots", min_value=1, value=1, step=1)
    sl = q.number_input("Stop Loss % (0 = off)", min_value=0.0, value=20.0)
    target = q.number_input("Target % (0 = off)", min_value=0.0, value=40.0)
    brokerage = st.number_input("Brokerage per order ₹ (estimate)", min_value=0.0, value=20.0)
    other = st.number_input("Other charges per order ₹ (estimate)", min_value=0.0, value=10.0)

    if st.button("Run Backtest", type="primary"):
        if exit_date < entry_date:
            st.error("Exit date must be on or after entry date.")
            return
        d = contract[(contract.date >= pd.Timestamp(entry_date)) & (contract.date <= pd.Timestamp(exit_date))]
        if d.empty:
            st.error("No rows for selected date range.")
            return
        d = d.sort_values("date")
        first = d.iloc[0]
        entry = float(first.close)
        qty = int(lots) * int(first.lot_size)
        stop = entry * (1 - sl/100) if side == "BUY" and sl else entry * (1 + sl/100) if sl else None
        take = entry * (1 + target/100) if side == "BUY" and target else entry * (1 - target/100) if target else None
        out = d.iloc[-1]
        reason = "Last available daily close"
        for _, row in d.iloc[1:].iterrows():
            price = float(row.close)
            hit_sl = stop is not None and ((price <= stop) if side == "BUY" else (price >= stop))
            hit_tp = take is not None and ((price >= take) if side == "BUY" else (price <= take))
            if hit_sl:
                out, reason = row, "Stop Loss"; break
            if hit_tp:
                out, reason = row, "Target"; break
        exit_price = float(out.close)
        gross = (exit_price-entry)*qty if side == "BUY" else (entry-exit_price)*qty
        charges = 2*(float(brokerage)+float(other))
        net = gross-charges
        r,s,t = st.columns(3)
        r.metric("Gross P&L", f"₹{gross:,.2f}")
        s.metric("Estimated charges", f"₹{charges:,.2f}")
        t.metric("Net P&L", f"₹{net:,.2f}")
        st.write({"Contract":symbol,"Side":side,"Entry date":first.date.date().isoformat(),
                  "Entry price":entry,"Exit date":out.date.date().isoformat(),"Exit price":exit_price,
                  "Lots":int(lots),"Lot size":int(first.lot_size),"Quantity":qty,"Exit reason":reason,
                  "Gross P&L":gross,"Estimated charges":charges,"Net P&L":net})
        st.caption("Charges are estimates only, not a complete tax calculator. SL/target are checked on daily closing prices, "
                   "not intraday highs/lows. Expiry intrinsic settlement is not modeled; the result is not a guaranteed executable fill.")
