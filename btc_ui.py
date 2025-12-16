import os
import requests
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

ENGINE_URL = os.getenv("ENGINE_URL", "").strip().rstrip("/")
UI_REFRESH_MS = int(os.getenv("UI_REFRESH_MS", "5000"))

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")
st.title("🧠 BTC AI Dashboard")
st.caption("Short-term • AI-filtered • Telegram alerts • Paper + Real (logged) P/L")

st_autorefresh(interval=UI_REFRESH_MS, key="btc_refresh")

def fetch_json(url: str, timeout=6):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_engine_state():
    if not ENGINE_URL:
        return None, "ENGINE_URL is not set."
    try:
        return fetch_json(f"{ENGINE_URL}/state", timeout=6), ""
    except Exception as e:
        return None, str(e)

def ping_health():
    if not ENGINE_URL:
        return None, "ENGINE_URL is not set."
    try:
        return fetch_json(f"{ENGINE_URL}/health", timeout=4), ""
    except Exception as e:
        return None, str(e)

def render_candles_with_momentum(candles):
    if not candles:
        st.info("⏳ Waiting for candle data…")
        return

    x = [c["time"] for c in candles]
    opens = [c["open"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]

    fig = go.Figure(
        data=[go.Candlestick(x=x, open=opens, high=highs, low=lows, close=closes, name="BTC")]
    )

    # Momentum arrows (5-candle momentum)
    up_x, up_y, dn_x, dn_y = [], [], [], []
    for i in range(len(closes)):
        if i < 5:
            continue
        prev = closes[i - 5]
        if prev == 0:
            continue
        mom = (closes[i] - prev) / prev
        if mom > 0.001:
            up_x.append(x[i]); up_y.append(highs[i] * 1.0005)
        elif mom < -0.001:
            dn_x.append(x[i]); dn_y.append(lows[i] * 0.9995)

    if up_x:
        fig.add_trace(go.Scatter(x=up_x, y=up_y, mode="markers",
                                 marker=dict(symbol="triangle-up", size=10),
                                 name="Momentum ↑"))
    if dn_x:
        fig.add_trace(go.Scatter(x=dn_x, y=dn_y, mode="markers",
                                 marker=dict(symbol="triangle-down", size=10),
                                 name="Momentum ↓"))

    fig.update_layout(
        height=380,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

state, err = get_engine_state()

with st.expander("Debug (click to open)", expanded=not bool(state)):
    st.write(f"ENGINE_URL: {ENGINE_URL or '(not set)'}")
    cdbg1, cdbg2 = st.columns([1, 3])
    with cdbg1:
        if st.button("Ping engine /health"):
            data, herr = ping_health()
            if herr:
                st.error(herr)
            else:
                st.success("Health OK")
                st.json(data)
    with cdbg2:
        if err:
            st.error(err)
        else:
            st.caption("Engine reachable.")

if not state:
    st.info("⏳ Waiting for engine data…")
    st.stop()

if state.get("error"):
    st.error(f"Engine error: {state['error']}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC Price", f"${state.get('price', 0):,.2f}")
c2.metric("RSI (1m)", state.get("rsi", 0))
c3.metric("Trend", state.get("trend", "WAIT"))
c4.metric("🧠 Confidence", f"{state.get('confidence', 0)}%")
st.caption(f"Last update: {state.get('time','--:--:--')}  •  {state.get('notes','')}")

conf = state.get("confidence", 0)
if conf >= 75:
    st.success("🔥 High-quality setup")
elif conf >= 60:
    st.warning("⚠️ Medium-quality signal")
else:
    st.info("⏳ Waiting for stronger conditions")

paper = state.get("paper", {}) or {}
manual = state.get("manual", {}) or {}

p1, p2, p3, p4 = st.columns(4)
p1.metric("📄 Paper P/L (total)", f"${paper.get('total_pl_usd', 0):,.2f}")
p2.metric("📄 Paper Position (BTC)", f"{paper.get('btc', 0):.6f}")
p3.metric("🧾 Real P/L (total)", f"${manual.get('total_pl_usd', 0):,.2f}")
p4.metric("🧾 Real Position (BTC)", f"{manual.get('btc', 0):.6f}")

st.subheader("📊 BTC 1-Minute Candlesticks (Last 30 min) + Momentum Arrows")
render_candles_with_momentum(state.get("candles", []))

st.subheader("🧾 Trades")
t1, t2 = st.columns(2)
with t1:
    st.write("📄 Paper trades (latest)")
    st.dataframe(paper.get("trades", [])[-20:], use_container_width=True, height=240)
with t2:
    st.write("🧾 Real (logged) trades (latest)")
    st.dataframe(manual.get("trades", [])[-20:], use_container_width=True, height=240)

st.markdown(
    """
**Telegram commands (manual logging):**
- `/status` — shows current price/RSI/trend + P/L  
- `/logbuy 100` — logs a real buy of **$100** at current price  
- `/logsell 100` — logs a real sell of **$100** at current price  

*(This dashboard is paper/record-keeping. It does not place real trades.)*
"""
)
