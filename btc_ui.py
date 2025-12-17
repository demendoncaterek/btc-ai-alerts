import os
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ================= CONFIG =================
ENGINE_URL = os.getenv("ENGINE_URL", "http://localhost:8080").rstrip("/")
REFRESH_MS = int(os.getenv("REFRESH_MS", "15000"))  # 15s
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
    if st.button("📒 Trades Journal"):
        st.query_params["page"] = "trades"
        st.rerun()

st.divider()

# ================= DATA =================
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
    st.caption("Check ENGINE_URL and that engine /health is reachable.")

    with st.expander("Debug", expanded=True):
        st.write("ENGINE_URL:", ENGINE_URL)
        st.json(state)

        if st.button("Ping /health"):
            st.json(fetch_health())

    st.stop()

# ================= OVERVIEW =================
if page == "overview":
    st.title("🧠 BTC AI Dashboard")
    st.caption("15m execution • 1h+4h bias • ATR SL/TP • Telegram alerts")

    cols = st.columns(4)
    cols[0].metric("BTC Price", f"${state['price']:,.2f}")
    cols[1].metric("RSI (5m)", state.get("rsi_5m", "--"))
    cols[2].metric("Signal", state["signal"])
    cols[3].metric("Confidence", f"{state['confidence']}%")

    if state["signal"] == "WAIT":
        st.warning("⏳ Waiting for strong setup")
    elif state["signal"] == "BUY":
        st.success("🟢 BUY setup detected")
    elif state["signal"] == "SELL":
        st.error("🔴 SELL setup detected")

    # ================= CANDLES =================
    st.subheader("📊 BTC Candlesticks")

    candles = pd.DataFrame(state["candles"])

    # 🔒 SAFE TIME COLUMN DETECTION
    if "time" in candles.columns:
        x_col = candles["time"]
    elif "timestamp" in candles.columns:
        x_col = candles["timestamp"]
    elif "t" in candles.columns:
        x_col = candles["t"]
    else:
        candles["index_time"] = range(len(candles))
        x_col = candles["index_time"]

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=x_col,
        open=candles["open"],
        high=candles["high"],
        low=candles["low"],
        close=candles["close"],
        name="BTC"
    ))

    # Momentum arrows (optional)
    if "bullish_marks" in state:
        fig.add_trace(go.Scatter(
            x=state["bullish_marks"]["x"],
            y=state["bullish_marks"]["y"],
            mode="markers",
            marker=dict(color="green", symbol="triangle-up", size=10),
            name="Bullish momentum"
        ))

    if "bearish_marks" in state:
        fig.add_trace(go.Scatter(
            x=state["bearish_marks"]["x"],
            y=state["bearish_marks"]["y"],
            mode="markers",
            marker=dict(color="red", symbol="triangle-down", size=10),
            name="Bearish momentum"
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
    st.caption("Paper + Real trades (logged by engine)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧪 Paper Trading")
        paper = pd.DataFrame(state.get("paper_trades", []))
        if paper.empty:
            st.info("No paper trades yet.")
        else:
            st.dataframe(paper, use_container_width=True)

    with col2:
        st.subheader("💰 Real Trades")
        real = pd.DataFrame(state.get("real_trades", []))
        if real.empty:
            st.info("No real trades logged yet.")
        else:
            st.dataframe(real, use_container_width=True)

    st.divider()

    cols = st.columns(4)
    cols[0].metric("Paper Equity", f"${state['paper_equity']:,.2f}")
    cols[1].metric("Paper P/L", f"${state['paper_pl']:,.2f}")
    cols[2].metric("Real BTC", f"{state['real_btc']:.6f}")
    cols[3].metric("Unrealized P/L", f"${state['real_unrealized_pl']:,.2f}")

else:
    st.error("Unknown page")
