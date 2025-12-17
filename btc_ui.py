import os
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

ENGINE_URL = os.getenv("ENGINE_URL", "http://localhost:8080").rstrip("/")
REFRESH_MS = int(os.getenv("REFRESH_MS", "15000"))  # 15s

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")

st.title("🧠 BTC AI Dashboard")
st.caption("5m + 1h bias • Telegram alerts • Paper + Real (logged) P/L")

if st_autorefresh:
    st_autorefresh(interval=REFRESH_MS, key="btc_autorefresh")

def fetch_state():
    try:
        r = requests.get(f"{ENGINE_URL}/state", timeout=6)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

state = fetch_state()

if not state.get("ok"):
    st.warning("⏳ Waiting for engine data…")
    st.write("Make sure ENGINE_URL points to your engine and /health works.")
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

price = float(state.get("price", 0))
signal = state.get("signal", "WAIT")
confidence = int(state.get("confidence", 0))
trend_bias = state.get("trend_bias", "NEUTRAL")
base_tf = int(state.get("base_granularity", 300)) // 60
rsi_v = state.get("base_rsi", None)
notes = state.get("notes", "")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("BTC Price", f"${price:,.2f}")
c2.metric(f"RSI ({base_tf}m)", f"{rsi_v}")
c3.metric("Trend (1h bias)", trend_bias)
c4.metric("Signal", signal)
c5.metric("Confidence", f"{confidence}%")
st.caption(f"Last update: {state.get('time')} • {notes}")

paper = state.get("paper_summary", {}) or {}
real = state.get("real_summary", {}) or {}
p1, p2, p3 = st.columns(3)
p1.metric("Paper Equity", f"${paper.get('equity', 0):,.2f}")
p2.metric("Paper P/L", f"${paper.get('pnl', 0):,.2f}")
p3.metric("Logged Real Unrealized P/L", f"${real.get('unrealized', 0):,.2f}")

st.subheader("📈 BTC Candlesticks (5m) + Momentum/Signal Markers")
candles = state.get("candles", []) or []

if candles:
    df = pd.DataFrame(candles)
    df["dt"] = pd.to_datetime(df["ts"], unit="s")

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["dt"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="BTC",
            )
        ]
    )

    events = state.get("events", []) or []
    if events:
        ev = pd.DataFrame(events)
        ev["dt"] = pd.to_datetime(ev["ts"])
        sym = []
        text = []
        for t, lbl in zip(ev["type"], ev.get("label", ev["type"])):
            t = str(t).upper()
            if t == "BUY":
                sym.append("triangle-up")
            elif t == "SELL":
                sym.append("triangle-down")
            elif t == "DIP":
                sym.append("triangle-left")
            elif t == "PEAK":
                sym.append("triangle-right")
            else:
                sym.append("circle")
            text.append(lbl)

        fig.add_trace(
            go.Scatter(
                x=ev["dt"],
                y=ev["price"],
                mode="markers",
                marker=dict(size=12, symbol=sym),
                name="Markers",
                text=text,
                hovertemplate="%{text}<br>$%{y:,.2f}<br>%{x}<extra></extra>",
            )
        )

    fig.update_layout(height=520, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No candle data yet.")

with st.expander("How to use (important)", expanded=False):
    st.write(
        """
This bot flags **setups** (not guaranteed wins).
- Telegram: `/status`
- Log real trades: `/logbuy 100` or `/logsell 100` (optional `[price]`)
- It will also alert “Dip Watch” / “Peak Watch” for strong moves over ~3 hours.
"""
    )

st.button("🔄 Refresh page")
