import os
import json
import requests
import streamlit as st
import plotly.graph_objects as go

ENGINE_URL = os.getenv("ENGINE_URL", "http://btc-engine.railway.internal").rstrip("/")

AUTO_REFRESH_MS = int(os.getenv("AUTO_REFRESH_MS", "5000"))  # 5s

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")
st.title("🧠 BTC AI Dashboard")
st.caption("Short-term • AI-filtered • Telegram alerts • Paper + Real (logged) P/L")

# ✅ Auto refresh (requires streamlit-autorefresh)
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=AUTO_REFRESH_MS, key="btc_refresh")
except Exception:
    st.info("Auto-refresh is off (add streamlit-autorefresh). Use the Refresh button for now.")

def fetch_state():
    try:
        r = requests.get(f"{ENGINE_URL}/state", timeout=5)
        return r.json()
    except:
        return None

def render_candles(candles):
    if not candles:
        st.info("⏳ Waiting for candle data…")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=[c["time"] for c in candles],
            open=[c["open"] for c in candles],
            high=[c["high"] for c in candles],
            low=[c["low"] for c in candles],
            close=[c["close"] for c in candles],
            name="BTC",
        )
    )

    # Momentum arrows (up/down close vs previous close)
    ups_x, ups_y = [], []
    downs_x, downs_y = [], []
    for i in range(1, len(candles)):
        prev_close = candles[i - 1]["close"]
        cur_close = candles[i]["close"]
        if cur_close >= prev_close:
            ups_x.append(candles[i]["time"])
            ups_y.append(candles[i]["low"])
        else:
            downs_x.append(candles[i]["time"])
            downs_y.append(candles[i]["high"])

    if ups_x:
        fig.add_trace(go.Scatter(
            x=ups_x, y=ups_y, mode="markers",
            marker=dict(symbol="triangle-up", size=9),
            name="Momentum Up",
            hoverinfo="skip"
        ))
    if downs_x:
        fig.add_trace(go.Scatter(
            x=downs_x, y=downs_y, mode="markers",
            marker=dict(symbol="triangle-down", size=9),
            name="Momentum Down",
            hoverinfo="skip"
        ))

    fig.update_layout(
        height=360,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig, use_container_width=True)

state = fetch_state()
if not state or ("error" in state and state.get("error") == "Starting…"):
    st.info("⏳ Waiting for engine data…")
    st.caption("Make sure ENGINE_URL points to your btc-engine service URL and /state works.")
    st.stop()

if state.get("error"):
    st.error(f"Engine error: {state['error']}")

# Top metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC Price", f"${state.get('price', 0):,.2f}")
c2.metric("RSI (1m)", state.get("rsi", 0))
c3.metric("Trend", state.get("trend", "WAIT"))
c4.metric("🧠 Confidence", f"{state.get('confidence', 0)}%")

st.caption(f"Last update: {state.get('time','--:--:--')}  •  {state.get('notes','')}")

conf = int(state.get("confidence", 0) or 0)
if conf >= 75:
    st.success("🔥 High-quality setup")
elif conf >= 60:
    st.warning("⚠️ Medium-quality signal")
else:
    st.info("⏳ Waiting for stronger conditions")

# Paper + Real P/L
paper = state.get("paper", {}) or {}
manual = state.get("manual", {}) or {}

st.subheader("💰 P/L Snapshot")

p1, p2, p3, p4 = st.columns(4)
p1.metric("Paper Equity", f"${paper.get('equity_usd', 0):,.2f}")
p2.metric("Paper Realized P/L", f"${paper.get('realized_pl_usd', 0):,.2f}")
p3.metric("Real Total P/L", f"${manual.get('total_pl_usd', 0):,.2f}")
p4.metric("Real BTC (logged)", f"{manual.get('qty_btc', 0):.8f}")

lt_paper = paper.get("last_trade")
lt_real = manual.get("last_trade")

colA, colB = st.columns(2)
with colA:
    st.write("**Last Paper Trade**")
    st.json(lt_paper if lt_paper else {})
with colB:
    st.write("**Last Real (Logged) Trade**")
    st.json(lt_real if lt_real else {})

st.subheader("📊 BTC 1-Minute Candlesticks (Last 30 min)")
render_candles(state.get("candles", []))

st.info(
    "Telegram commands:\n"
    "- /status\n"
    "- /logbuy 100\n"
    "- /logsell 100\n"
)

if st.button("🔄 Refresh Now"):
    st.rerun()
