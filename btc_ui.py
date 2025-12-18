import os
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

ENGINE_URL = os.getenv("ENGINE_URL", "").rstrip("/")
REFRESH_MS = int(os.getenv("REFRESH_MS", "10000"))  # 10s

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")
st.title("🧠 BTC AI Dashboard")

if st_autorefresh:
    st_autorefresh(interval=REFRESH_MS, key="btc_autorefresh")

def get_json(path, timeout=8):
    r = requests.get(f"{ENGINE_URL}{path}", timeout=timeout)
    return r.json()

# ---------- connection guard ----------
if not ENGINE_URL:
    st.error("ENGINE_URL env var is missing in the UI service.")
    st.stop()

state = get_json("/state")
if not state.get("ok"):
    st.warning("Engine not ready / error")
    st.write(state)
    st.stop()

metrics = get_json("/metrics")
trades_resp = get_json("/trades")
trades = trades_resp.get("trades", [])

tab_overview, tab_trades = st.tabs(["🏠 Overview", "🧾 Trades"])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${state['price']:,.2f}")
    c2.metric("Signal", state.get("signal", "—"))
    c3.metric("Confidence", f"{state.get('confidence', 0)*100:.1f}%")
    c4.metric("Equity", f"${metrics.get('equity', 0):,.2f}")

    st.caption(f"Min confidence (auto): **{state.get('min_conf', 0)*100:.1f}%** • Last update: {state.get('time','')}")

    st.subheader("🧭 Reason / Levels")
    st.write("**Reason:**", state.get("reason", "—"))
    st.json(state.get("levels", {}), expanded=False)

    st.subheader("📈 Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Win rate", f"{metrics.get('win_rate', 0)*100:.1f}%")
    m2.metric("Profit factor", f"{metrics.get('profit_factor', 0):.2f}")
    m3.metric("Avg R", f"{metrics.get('avg_r', 0):.2f}")
    m4.metric("Max DD", f"{metrics.get('max_dd', 0)*100:.1f}%")

    st.subheader("🕯️ Candlesticks (15m)")
    candles = get_json("/candles?tf=15m")
    if candles.get("ok"):
        df = pd.DataFrame(candles["candles"])
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
        fig.update_layout(height=520, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Candle data not available yet.")

    if state.get("open_trade"):
        ot = state["open_trade"]
        st.subheader("📌 Open Paper Trade")
        st.json(ot, expanded=False)

with tab_trades:
    st.subheader("🧾 Trade Journal")
    if not trades:
        st.info("No paper trades closed yet. Once one closes, you’ll see the table + equity curve here.")
    else:
        df = pd.DataFrame(trades)
        # equity curve from closed trades
        df2 = df.sort_values("opened")
        start_eq = 250.0
        df2["equity_curve"] = start_eq + df2["pnl"].cumsum()

        fig2 = go.Figure(go.Scatter(x=df2["opened"], y=df2["equity_curve"], mode="lines", name="Equity"))
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df, use_container_width=True)
