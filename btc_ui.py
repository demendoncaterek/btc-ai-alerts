import os
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ================= CONFIG =================
ENGINE_URL = os.getenv("ENGINE_URL", "http://localhost:8080").rstrip("/")
REFRESH_MS = int(os.getenv("REFRESH_MS", "15000"))
# ==========================================

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")

# ================= NAV =================
page = st.query_params.get("page", "overview")

nav1, nav2 = st.columns(2)
with nav1:
    if st.button("🏠 Overview"):
        st.query_params.clear()
        st.rerun()

with nav2:
    if st.button("📒 Trades"):
        st.query_params["page"] = "trades"
        st.rerun()

st.divider()

# ================= FETCH =================
def fetch_state():
    try:
        r = requests.get(f"{ENGINE_URL}/state", timeout=6)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

def fetch_health():
    try:
        r = requests.get(f"{ENGINE_URL}/health", timeout=6)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

state = fetch_state()

# ================= ENGINE NOT READY =================
if not state.get("ok"):
    st.title("🧠 BTC AI Dashboard")
    st.warning("⏳ Engine not ready yet.")

    with st.expander("Debug", expanded=True):
        st.write("ENGINE_URL:", ENGINE_URL)
        st.json(state)

        if st.button("Ping /health"):
            st.json(fetch_health())

    st.stop()

# ================= CANDLE NORMALIZER =================
def normalize_candles(raw):
    """
    Supports:
    - Dict candles: {time, open, high, low, close}
    - List candles: [time, low, high, open, close, volume]
    """
    if not raw:
        return pd.DataFrame()

    first = raw[0]

    # Dict format
    if isinstance(first, dict):
        df = pd.DataFrame(raw)
        df.rename(columns={"t": "time", "timestamp": "time"}, inplace=True)
        return df

    # List format (Coinbase style)
    if isinstance(first, (list, tuple)) and len(first) >= 5:
        df = pd.DataFrame(
            raw,
            columns=["time", "low", "high", "open", "close", "volume"][: len(first)]
        )
        return df

    return pd.DataFrame()

# ================= OVERVIEW =================
if page == "overview":
    st.title("🧠 BTC AI Dashboard")
    st.caption("15m execution • 1h + 4h bias • ATR SL/TP • Telegram alerts")

    cols = st.columns(4)
    cols[0].metric("BTC Price", f"${state['price']:,.2f}")
    cols[1].metric("RSI (5m)", state.get("rsi_5m", "--"))
    cols[2].metric("Signal", state["signal"])
    cols[3].metric("Confidence", f"{state['confidence']}%")

    if state["signal"] == "WAIT":
        st.warning("⏳ Waiting for high-probability setup")
    elif state["signal"] == "BUY":
        st.success("🟢 BUY setup detected")
    elif state["signal"] == "SELL":
        st.error("🔴 SELL setup detected")

    # ================= CHART =================
    st.subheader("📊 BTC Candlesticks")

    candles = normalize_candles(state.get("candles", []))

    if candles.empty:
        st.info("No candle data yet.")
        st.stop()

    # Safe time column
    if "time" not in candles.columns:
        candles["time"] = range(len(candles))

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=candles["time"],
        open=candles["open"],
        high=candles["high"],
        low=candles["low"],
        close=candles["close"],
        name="BTC"
    ))

    # Momentum arrows
    for b in state.get("bullish_marks", []):
        fig.add_trace(go.Scatter(
            x=[b["x"]],
            y=[b["y"]],
            mode="markers",
            marker=dict(color="green", symbol="triangle-up", size=10),
            name="Bullish"
        ))

    for b in state.get("bearish_marks", []):
        fig.add_trace(go.Scatter(
            x=[b["x"]],
            y=[b["y"]],
            mode="markers",
            marker=dict(color="red", symbol="triangle-down", size=10),
            name="Bearish"
        ))

    fig.update_layout(
        height=520,
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

# ================= TRADES =================
elif page == "trades":
    st.title("📒 Trade Journal")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧪 Paper Trades")
        df = pd.DataFrame(state.get("paper_trades", []))
        st.dataframe(df if not df.empty else pd.DataFrame(), use_container_width=True)

    with col2:
        st.subheader("💰 Real Trades")
        df = pd.DataFrame(state.get("real_trades", []))
        st.dataframe(df if not df.empty else pd.DataFrame(), use_container_width=True)

    st.divider()

    cols = st.columns(4)
    cols[0].metric("Paper Equity", f"${state['paper_equity']:,.2f}")
    cols[1].metric("Paper P/L", f"${state['paper_pl']:,.2f}")
    cols[2].metric("Real BTC", f"{state['real_btc']:.6f}")
    cols[3].metric("Unrealized P/L", f"${state['real_unrealized_pl']:,.2f}")
