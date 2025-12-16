import subprocess
import sys
import os
import json
import time
import streamlit as st
import plotly.graph_objects as go

# =========================
# AUTO-START ENGINE (SAFE)
# =========================
ENGINE_FLAG = "/tmp/engine_started.flag"

if not os.path.exists(ENGINE_FLAG):
    subprocess.Popen([sys.executable, "btc_telegram_alerts.py"])
    open(ENGINE_FLAG, "w").close()

# =========================
# CONFIG
# =========================
STATE_FILE = "btc_state.json"
REFRESH_INTERVAL = 5  # seconds

# =========================
# STREAMLIT SETUP
# =========================
st.set_page_config(page_title="BTC AI Dashboard", layout="wide")
st.title("🧠 BTC AI Dashboard")
st.caption("Short-term • AI-filtered • Telegram alerts")

# =========================
# AUTO REFRESH LOGIC (SAFE)
# =========================
now = time.time()
last = st.session_state.get("last_refresh", 0)

if now - last > REFRESH_INTERVAL:
    st.session_state["last_refresh"] = now
    st.rerun()

# =========================
# HELPERS
# =========================
def safe_load_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if not raw:
                return None
            return json.loads(raw)
    except:
        return None


def render_candles(candles):
    if not candles or len(candles) < 2:
        st.info("⏳ Waiting for candle data…")
        return

    times = [c["time"] for c in candles]
    opens = [c["open"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=times,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="BTC"
        )
    )

    # =========================
    # MOMENTUM ARROWS
    # =========================
    arrow_x = []
    arrow_y = []
    arrow_color = []
    arrow_symbol = []

    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]

        # Ignore tiny noise
        if abs(delta) < closes[i] * 0.0002:
            continue

        arrow_x.append(times[i])

        if delta > 0:
            arrow_y.append(lows[i] * 0.999)
            arrow_color.append("lime")
            arrow_symbol.append("triangle-up")
        else:
            arrow_y.append(highs[i] * 1.001)
            arrow_color.append("red")
            arrow_symbol.append("triangle-down")

    fig.add_trace(
        go.Scatter(
            x=arrow_x,
            y=arrow_y,
            mode="markers",
            marker=dict(
                size=10,
                color=arrow_color,
                symbol=arrow_symbol
            ),
            name="Momentum"
        )
    )

    fig.update_layout(
        height=360,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# LOAD STATE
# =========================
state = safe_load_state()

if not state:
    st.info("⏳ Waiting for engine data…")
    st.stop()

if state.get("error"):
    st.error(f"Engine error: {state['error']}")

# =========================
# METRICS
# =========================
c1, c2, c3, c4 = st.columns(4)

c1.metric("BTC Price", f"${state.get('price', 0):,.2f}")
c2.metric("RSI (1m)", state.get("rsi", 0))
c3.metric("Trend", state.get("trend", "WAIT"))
c4.metric("🧠 Confidence", f"{state.get('confidence', 0)}%")

st.caption(
    f"Last update: {state.get('time','--:--:--')}  •  {state.get('notes','')}"
)

# =========================
# SIGNAL QUALITY
# =========================
conf = state.get("confidence", 0)

if conf >= 75:
    st.success("🔥 High-quality setup")
elif conf >= 60:
    st.warning("⚠️ Medium-quality signal")
else:
    st.info("⏳ Waiting for stronger conditions")

# =========================
# CHART
# =========================
st.subheader("📊 BTC 1-Minute Candlesticks (Last 30 min)")
render_candles(state.get("candles", []))

# =========================
# MANUAL REFRESH (OPTIONAL)
# =========================
if st.button("🔄 Refresh Now"):
    st.session_state["last_refresh"] = time.time()
    st.rerun()
