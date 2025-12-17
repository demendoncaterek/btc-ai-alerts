import os
import requests
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# ================= CONFIG =================
ENGINE_URL = os.getenv("ENGINE_URL", "http://localhost:8080").rstrip("/")
REFRESH_MS = int(os.getenv("REFRESH_MS", "15000"))  # 15s
# ==========================================

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")

st.title("🧠 BTC AI Dashboard")
st.caption("5m + 1h bias • Telegram alerts • Paper + Real (logged) P/L")

# ---------- Auto refresh (safe) ----------
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_MS, key="btc_autorefresh")
except Exception:
    pass


# ============== DATA FETCH ==============
def fetch_state():
    try:
        r = requests.get(f"{ENGINE_URL}/state", timeout=6)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

state = fetch_state()

# ---------- If engine not ready ----------
if not state.get("ok"):
    st.warning("⏳ Waiting for engine data…")
    st.caption("Market will display once engine responds.")

    with st.expander("Debug (click to open)", expanded=True):
        st.write("ENGINE_URL:", ENGINE_URL)
        st.error(state.get("error", "No state yet"))
        if st.button("Ping engine /health"):
            try:
                r = requests.get(f"{ENGINE_URL}/health", timeout=6)
                st.json(r.json())
            except Exception as e:
                st.error(str(e))
    st.stop()

# ================= MARKET METRICS =================
price = state.get("price")
rsi = state.get("rsi")
confidence = state.get("confidence", 0)
trend = state.get("trend", "WAIT")
time_str = state.get("time", "--")
candles = state.get("candles", [])
momentum = state.get("momentum", 0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC Price", f"${price:,.2f}" if price else "--")
c2.metric("RSI (5m)", round(rsi, 1) if rsi else "--")
c3.metric("Trend", trend)
c4.metric("Confidence", f"{confidence}%")

st.caption(f"Last update: {time_str} • src=Coinbase")

# ================= SIGNAL STATUS =================
if confidence >= 80:
    st.success("🔥 High-confidence setup detected")
elif confidence >= 60:
    st.warning("⚠️ Medium-confidence setup")
else:
    st.info("⏳ Waiting for strong conditions")

# ================= CANDLE CHART =================
st.subheader("📊 BTC Candlesticks")

if not candles:
    st.info("Waiting for candle data…")
else:
    fig = go.Figure()

    # Candles
    fig.add_trace(go.Candlestick(
        x=[c["time"] for c in candles],
        open=[c["open"] for c in candles],
        high=[c["high"] for c in candles],
        low=[c["low"] for c in candles],
        close=[c["close"] for c in candles],
        name="BTC"
    ))

    # ---------- Momentum arrows (subtle style) ----------
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    buy_x, buy_y = [], []
    sell_x, sell_y = [], []

    for i in range(2, len(candles)):
        # Higher low = bullish momentum
        if lows[i] > lows[i - 1] > lows[i - 2]:
            buy_x.append(candles[i]["time"])
            buy_y.append(lows[i] * 0.999)

        # Lower high = bearish momentum
        if highs[i] < highs[i - 1] < highs[i - 2]:
            sell_x.append(candles[i]["time"])
            sell_y.append(highs[i] * 1.001)

    fig.add_trace(go.Scatter(
        x=buy_x,
        y=buy_y,
        mode="markers",
        marker=dict(symbol="triangle-up", color="lime", size=8),
        name="Bullish momentum"
    ))

    fig.add_trace(go.Scatter(
        x=sell_x,
        y=sell_y,
        mode="markers",
        marker=dict(symbol="triangle-down", color="red", size=8),
        name="Bearish momentum"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=10, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

# ================= FOOTER =================
if st.button("🔄 Refresh Now"):
    st.rerun()
