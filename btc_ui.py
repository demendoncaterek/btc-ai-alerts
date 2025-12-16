import os
import json
import requests
import streamlit as st
import plotly.graph_objects as go

# If you install streamlit-autorefresh, this will work.
# If not installed, the app still runs (just no auto-refresh).
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

ENGINE_URL = os.getenv("ENGINE_URL", "").strip().rstrip("/")  # e.g. https://your-btc-engine.up.railway.app
STATE_FILE = "btc_state.json"  # fallback if you run UI+engine in same service

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")
st.title("🧠 BTC AI Dashboard")
st.caption("Short-term • AI-filtered • Telegram alerts • Paper + Real (logged) P/L")

if HAS_AUTOREFRESH:
    st_autorefresh(interval=5000, key="btc_refresh")  # refresh every 5s


def fetch_state():
    # Preferred: HTTP from engine
    if ENGINE_URL:
        try:
            r = requests.get(f"{ENGINE_URL}/state", timeout=6)
            if r.status_code == 200:
                return r.json()
        except:
            return None

    # Fallback: local file
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                raw = f.read().strip()
                if raw:
                    return json.loads(raw)
        except:
            return None

    return None


def render_candles_with_momentum(candles):
    if not candles:
        st.info("⏳ Waiting for candle data…")
        return

    x = [c["time"] for c in candles]
    o = [c["open"] for c in candles]
    h = [c["high"] for c in candles]
    l = [c["low"] for c in candles]
    c = [c["close"] for c in candles]

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=o,
            high=h,
            low=l,
            close=c,
            name="BTC",
        )
    )

    # momentum arrows based on candle-to-candle close change
    ups_x, ups_y, downs_x, downs_y = [], [], [], []
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            ups_x.append(x[i])
            ups_y.append(c[i])
        elif c[i] < c[i - 1]:
            downs_x.append(x[i])
            downs_y.append(c[i])

    if ups_x:
        fig.add_trace(
            go.Scatter(
                x=ups_x,
                y=ups_y,
                mode="markers",
                name="Up momentum",
                marker=dict(symbol="triangle-up", size=10),
            )
        )

    if downs_x:
        fig.add_trace(
            go.Scatter(
                x=downs_x,
                y=downs_y,
                mode="markers",
                name="Down momentum",
                marker=dict(symbol="triangle-down", size=10),
            )
        )

    fig.update_layout(
        height=360,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


state = fetch_state()
if not state:
    st.info("⏳ Waiting for engine data…")
    st.caption("Make sure ENGINE_URL is set to your btc-engine service URL (and /state works).")
    st.stop()

if state.get("error"):
    st.error(f"Engine error: {state['error']}")

paper = state.get("paper", {}) or {}
real = state.get("real", {}) or {}

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("BTC Price", f"${state.get('price', 0):,.2f}")
c2.metric("RSI (1m)", state.get("rsi", 0))
c3.metric("Trend", state.get("trend", "WAIT"))
c4.metric("Confidence", f"{state.get('confidence', 0)}%")
c5.metric("Momentum (~5m)", f"{float(state.get('momentum', 0.0)):+.5f}")

st.caption(f"Last update: {state.get('time','--:--:--')}  •  {state.get('notes','')}")

pcol, rcol = st.columns(2)

with pcol:
    st.subheader("🧪 Paper Trading (Auto)")
    st.metric("Equity", f"${paper.get('equity', 0):,.2f}")
    st.metric("Total P/L", f"${paper.get('total_pl', 0):,.2f}")
    st.caption(f"Cash: ${paper.get('cash',0):,.2f} • BTC: {paper.get('btc',0)}")

with rcol:
    st.subheader("🧾 Real Trades (Logged via Telegram)")
    st.metric("Equity", f"${real.get('equity', 0):,.2f}")
    st.metric("Total P/L", f"${real.get('total_pl', 0):,.2f}")
    st.caption(f"Cash: ${real.get('cash',0):,.2f} • BTC: {real.get('btc',0)}")

conf = int(state.get("confidence", 0))
if conf >= 75:
    st.success("🔥 High-quality setup")
elif conf >= 60:
    st.warning("⚠️ Medium-quality signal")
else:
    st.info("⏳ Waiting for stronger conditions")

st.subheader("📊 BTC 1-Minute Candlesticks (Last 30 min)")
render_candles_with_momentum(state.get("candles", []))

st.subheader("📒 Recent Trades")
t1, t2 = st.columns(2)

with t1:
    st.write("Paper trades")
    st.dataframe((state.get("trades", {}) or {}).get("paper", []), use_container_width=True)

with t2:
    st.write("Real trades (logged)")
    st.dataframe((state.get("trades", {}) or {}).get("real", []), use_container_width=True)

with st.sidebar:
    st.header("Telegram Commands")
    st.code(
        "/status\n"
        "/logbuy 100\n"
        "/logsell 100\n"
        "/help",
        language="text",
    )
    if not HAS_AUTOREFRESH:
        st.warning("Auto-refresh not installed. Add `streamlit-autorefresh` to requirements.txt for live updates.")
