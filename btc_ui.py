import os
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

ENGINE_URL = os.getenv("ENGINE_URL", "http://localhost:8080").rstrip("/")
REFRESH_MS = int(os.getenv("REFRESH_MS", "10000"))  # 10s default

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")

st.title("🧠 BTC AI Dashboard")
st.caption("15m execution • 1h+6h bias • RSI(5m/15m) • MACD • ATR SL/TP • Peak/Dip watch • Paper/Real logs")

if st_autorefresh:
    st_autorefresh(interval=REFRESH_MS, key="btc_autorefresh")

def fetch_json(path: str, timeout: float = 12.0):
    try:
        r = requests.get(f"{ENGINE_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    page = st.radio("Navigation", ["Overview", "Trades"], horizontal=True, label_visibility="collapsed")
with c3:
    if st.button("🔄 Refresh now"):
        st.rerun()

with st.expander("Connection Debug", expanded=False):
    st.write("ENGINE_URL:", ENGINE_URL)
    colA, colB = st.columns(2)
    with colA:
        if st.button("Ping /health"):
            data, err = fetch_json("/health")
            if err:
                st.error(f"/health failed: {err}")
            else:
                st.json(data)
    with colB:
        if st.button("Ping /state"):
            data, err = fetch_json("/state")
            if err:
                st.error(f"/state failed: {err}")
            else:
                st.json(data)

state, err = fetch_json("/state")
if err or not state:
    st.error("Engine is NOT reachable from the UI right now.")
    st.write("Error:", err)
    st.stop()

if not state.get("ok"):
    st.warning("⏳ Engine is up, but market state isn't ready yet.")
    st.write("Engine error:", state.get("error"))
    st.stop()

price = state.get("price")
signal = state.get("signal", "WAIT")
conf = float(state.get("confidence", 0.0))
rsi5 = state.get("rsi_5m")
rsi15 = state.get("rsi_15m")
b1h = state.get("bias_1h", "UNKNOWN")
b6h = state.get("bias_6h", "UNKNOWN")
reason = state.get("reason", "")
sl = state.get("sl")
tp = state.get("tp")
iso = state.get("iso")
src = state.get("src", "")

if page == "Overview":
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("BTC Price", f"${price:,.2f}" if isinstance(price, (int, float)) else "--")
    m2.metric("Signal", signal)
    m3.metric("Confidence", f"{conf:.0f}%")
    m4.metric("RSI (5m)", f"{rsi5:.1f}" if isinstance(rsi5, (int, float)) else "--")
    m5.metric("RSI (15m)", f"{rsi15:.1f}" if isinstance(rsi15, (int, float)) else "--")
    m6.metric("Bias (1h/6h)", f"{b1h}/{b6h}")

    events = state.get("events", []) or []
    if events:
        for ev in events[:3]:
            if ev.get("type") == "PEAK_WATCH":
                st.warning(f"📈 Peak watch: price ${price:,.2f} near {ev.get('window_min')}m high ${ev.get('near_peak'):,.2f}")
            elif ev.get("type") == "DIP_WATCH":
                st.info(f"📉 Dip watch: price ${price:,.2f} near {ev.get('window_min')}m low ${ev.get('near_dip'):,.2f}")

    st.caption(f"Last update: {iso} • Source: {src}")

    if sl is not None and tp is not None and signal in ("BUY", "SELL"):
        st.success(f"ATR levels → SL: {sl:,.2f} • TP: {tp:,.2f}")
    st.write("Reason:", reason)

    st.subheader("📊 BTC Candlesticks (15m)")
    candles = state.get("candles_15m") or []
    if not candles or any(k not in candles[0] for k in ("time", "open", "high", "low", "close")):
        st.warning("Candle data incomplete — waiting for full feed.")
    else:
        df = pd.DataFrame(candles)
        df["time"] = pd.to_datetime(df["time"], utc=True)

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["time"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name="BTC",
                )
            ]
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

else:
    trades, terr = fetch_json("/trades")
    if terr or not trades:
        st.error(f"Couldn't load trades: {terr}")
        st.stop()

    st.subheader("📒 Trade Journal")
    paper_state = state.get("paper", {}) or {}

    col1, col2, col3 = st.columns(3)
    col1.metric("Paper USD", f"${paper_state.get('usd', 0.0):,.2f}")
    col2.metric("Paper Equity", f"${paper_state.get('equity', 0.0):,.2f}")
    col3.metric("Paper Unreal P/L", f"${paper_state.get('unreal_pnl', 0.0):,.2f}")

    st.markdown("### 🧪 Paper Trades")
    paper = trades.get("paper", []) or []
    if paper:
        st.dataframe(pd.DataFrame(paper), use_container_width=True, height=260)
    else:
        st.info("No paper trades yet.")

    st.markdown("### 💰 Real Trades (Logged)")
    real = trades.get("real", []) or []
    if real:
        st.dataframe(pd.DataFrame(real), use_container_width=True, height=260)
    else:
        st.info("No real trades logged yet.")
