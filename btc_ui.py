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

# ---------- Helpers ----------
def safe_get(d, key, default=None):
    return d[key] if isinstance(d, dict) and key in d else default

def fetch_state():
    try:
        r = requests.get(f"{ENGINE_URL}/state", timeout=6)
        data = r.json()
        data["ok"] = True
        return data
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------- Header ----------
st.title("🧠 BTC AI Dashboard")
st.caption("15m execution • 1h+4h bias • ATR SL/TP • Telegram alerts")

# ---------- Fetch ----------
state = fetch_state()

if not state.get("ok"):
    st.warning("⏳ Engine not ready yet.")
    with st.expander("Debug", expanded=True):
        st.write("ENGINE_URL:", ENGINE_URL)
        st.error(state.get("error", "Unknown error"))
        if st.button("Ping /health"):
            try:
                st.json(requests.get(f"{ENGINE_URL}/health", timeout=5).json())
            except Exception as e:
                st.error(str(e))
    st.stop()

# ---------- Top Metrics ----------
c1, c2, c3, c4 = st.columns(4)

c1.metric("BTC Price", f"${safe_get(state,'price',0):,.2f}")
c2.metric("RSI (5m)", safe_get(state, "rsi", "--"))
c3.metric("Signal", safe_get(state, "signal", "WAIT"))
c4.metric("Confidence", f"{safe_get(state,'confidence',0)}%")

# ---------- Status Banner ----------
confidence = safe_get(state, "confidence", 0)
if confidence >= 75:
    st.success("🔥 High-probability setup")
elif confidence >= 60:
    st.warning("⚠️ Medium-confidence setup")
else:
    st.info("⏳ Waiting for high-probability setup")

# ---------- Candlestick Chart ----------
st.subheader("📊 BTC Candlesticks")

candles = safe_get(state, "candles", [])

if not candles:
    st.info("No candle data yet.")
else:
    df = pd.DataFrame(candles)

    required_cols = {"time", "open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        st.warning("Candle data incomplete — waiting for full feed.")
    else:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["time"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name="BTC"
                )
            ]
        )

        # Momentum markers (optional)
        if "bullish" in df.columns:
            fig.add_scatter(
                x=df[df["bullish"]]["time"],
                y=df[df["bullish"]]["low"],
                mode="markers",
                marker=dict(color="lime", size=8, symbol="triangle-up"),
                name="Bullish momentum"
            )

        if "bearish" in df.columns:
            fig.add_scatter(
                x=df[df["bearish"]]["time"],
                y=df[df["bearish"]]["high"],
                mode="markers",
                marker=dict(color="red", size=8, symbol="triangle-down"),
                name="Bearish momentum"
            )

        fig.update_layout(
            template="plotly_dark",
            height=420,
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=30, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

# ---------- Trade Journal ----------
st.divider()
st.header("📓 Trade Journal")

paper_equity = safe_get(state, "paper_equity", 0.0)
real_equity = safe_get(state, "real_equity", 0.0)

t1, t2 = st.columns(2)
t1.metric("Paper Equity", f"${paper_equity:,.2f}")
t2.metric("Real Equity (Logged)", f"${real_equity:,.2f}")

paper_trades = safe_get(state, "paper_trades", [])
real_trades = safe_get(state, "real_trades", [])

st.subheader("🧪 Paper Trades")
if paper_trades:
    st.dataframe(pd.DataFrame(paper_trades))
else:
    st.info("No paper trades yet.")

st.subheader("💰 Real Trades")
if real_trades:
    st.dataframe(pd.DataFrame(real_trades))
else:
    st.info("No real trades logged yet.")

# ---------- Footer ----------
st.caption(f"Last update: {safe_get(state,'time','--')} • Source: {safe_get(state,'src','')}")
