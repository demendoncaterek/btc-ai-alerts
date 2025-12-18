import os
import time
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# ================= CONFIG =================
ENGINE_URL = os.getenv("ENGINE_URL", "http://localhost:8080").rstrip("/")
REFRESH_MS = int(os.getenv("REFRESH_MS", "10000"))  # 10s default like before
# ==========================================

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")

def fetch_json(path: str):
    try:
        r = requests.get(f"{ENGINE_URL}{path}", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

def nav():
    params = st.query_params
    page = params.get("page", "overview")
    cols = st.columns([1,1,1])
    with cols[0]:
        if st.button("🏠 Overview", use_container_width=True):
            st.query_params["page"] = "overview"
            st.rerun()
    with cols[1]:
        if st.button("🧾 Trades", use_container_width=True):
            st.query_params["page"] = "trades"
            st.rerun()
    with cols[2]:
        if st.button("🔄 Refresh now", use_container_width=True):
            st.rerun()
    return page

st.title("🧠 BTC AI Dashboard")
st.caption("15m execution • 1h+ bias • RSI/MACD • ATR SL/TP • Peak/Dip watch • Paper/Real logs")

# Auto refresh
if st_autorefresh:
    st_autorefresh(interval=REFRESH_MS, key="btc_autorefresh")
else:
    # fallback: soft refresh every ~10s
    time.sleep(0.01)

page = nav()

# Always load state for header + debugging
state = fetch_json("/state")

# Connection debug expander
with st.expander("Connection Debug", expanded=False):
    st.write("ENGINE_URL:", ENGINE_URL)
    colA, colB = st.columns(2)
    with colA:
        if st.button("Ping /health", use_container_width=True):
            st.json(fetch_json("/health"))
    with colB:
        if st.button("Ping /state", use_container_width=True):
            st.json(fetch_json("/state"))

if not state.get("ok"):
    st.error("Engine is NOT reachable / not ready.")
    st.write(state.get("error", "No error text"))
    st.stop()

# ========== Overview page ==========
if page == "overview":
    price = state.get("price")
    signal = state.get("signal", "WAIT")
    conf = state.get("confidence", 0.0)
    rsi5 = state.get("rsi_5m")
    trend = state.get("trend_bias", "UNKNOWN")
    reason = state.get("reason", "")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BTC Price", f"${price:,.2f}" if isinstance(price,(int,float)) else "—")
    c2.metric("RSI (5m)", f"{rsi5:.1f}" if isinstance(rsi5,(int,float)) else "—")
    c3.metric("Signal", signal)
    c4.metric("Confidence", f"{conf*100:.2f}%")

    if signal == "WAIT":
        st.info("⏳ Waiting for high-probability setup")
    elif signal == "BUY":
        st.success("✅ BUY setup detected (paper engine may enter if confidence threshold is met)")
    elif signal == "SELL":
        st.warning("⚠️ SELL setup detected (paper engine may enter if confidence threshold is met)")

    if reason:
        st.caption(f"Reason: {reason} • Trend bias: {trend} • Last update: {state.get('time')} • Source: Coinbase")

    # Candlestick chart
    st.subheader("📊 BTC Candlesticks")
    candles = state.get("candles_exec", [])
    if not candles or len(candles) < 20:
        st.warning("Candle data incomplete — waiting for full feed.")
    else:
        df = pd.DataFrame(candles)
        # expected keys: t,o,h,l,c
        needed = {"t","o","h","l","c"}
        if not needed.issubset(set(df.columns)):
            st.error(f"Engine candle schema mismatch. Need {needed}, got {set(df.columns)}")
        else:
            df["t"] = pd.to_datetime(df["t"], utc=True)
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=df["t"],
                        open=df["o"],
                        high=df["h"],
                        low=df["l"],
                        close=df["c"],
                        name="BTC",
                    )
                ]
            )

            # Optional markers if engine provides later; safe to ignore for now
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Paper position snapshot
    st.subheader("📓 Paper Trading Snapshot")
    colp1, colp2 = st.columns(2)
    colp1.metric("Paper Equity", f"${state.get('paper_equity', 0.0):,.2f}")
    ot = state.get("paper_open_trade")
    if ot:
        colp2.metric("Open Paper Trade", f"{ot.get('side')} @ ${ot.get('entry'):,.2f}")
        st.write(ot)
    else:
        colp2.metric("Open Paper Trade", "None")

# ========== Trades page ==========
elif page == "trades":
    st.header("🧾 Trade Journal")
    data = fetch_json("/trades")
    if not data.get("ok"):
        st.error(data.get("error", "Failed to load trades"))
        st.stop()

    c1, c2 = st.columns(2)
    c1.metric("Paper Equity", f"${data.get('paper_equity', 0.0):,.2f}")
    c2.metric("Real Trades Logged", f"{len(data.get('real_trades', []))}")

    st.subheader("🧪 Paper Trades")
    paper = data.get("paper_trades", [])
    if not paper:
        st.info("No paper trades yet.")
    else:
        dfp = pd.DataFrame(paper)
        st.dataframe(dfp, use_container_width=True)

    st.subheader("💰 Real Trades (Logged)")
    real = data.get("real_trades", [])
    if not real:
        st.info("No real trades logged yet.")
    else:
        dfr = pd.DataFrame(real)
        st.dataframe(dfr, use_container_width=True)

else:
    st.error("Unknown page. Use ?page=overview or ?page=trades")
