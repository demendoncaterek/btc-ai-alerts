import subprocess
import sys
import os

ENGINE_FLAG = "/tmp/engine_started.flag"

if not os.path.exists(ENGINE_FLAG):
    subprocess.Popen([sys.executable, "btc_telegram_alerts.py"])
    open(ENGINE_FLAG, "w").close()

import json
import os
import streamlit as st
import plotly.graph_objects as go

STATE_FILE = "btc_state.json"  # must match engine

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")
st.title("🧠 BTC AI Dashboard")
st.caption("Short-term • AI-filtered • Telegram alerts")


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
    if not candles:
        st.info("⏳ Waiting for candle data…")
        return

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=[c["time"] for c in candles],
                open=[c["open"] for c in candles],
                high=[c["high"] for c in candles],
                low=[c["low"] for c in candles],
                close=[c["close"] for c in candles],
            )
        ]
    )
    fig.update_layout(
        height=320,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, width="stretch")


state = safe_load_state()
if not state:
    st.info("⏳ Waiting for engine data… (start btc_telegram_alerts.py first)")
    st.stop()

if "error" in state and state["error"]:
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

st.subheader("📊 BTC 1-Minute Candlesticks (Last 30 min)")
render_candles(state.get("candles", []))

if st.button("🔄 Refresh Now"):
    st.rerun()
