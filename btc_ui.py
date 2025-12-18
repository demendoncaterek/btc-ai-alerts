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
REFRESH_MS = int(os.getenv("REFRESH_MS", "10000"))  # 10s default

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")
st.title("🧠 BTC AI Dashboard")

def get_json(path, timeout=8):
    r = requests.get(f"{ENGINE_URL}{path}", timeout=timeout)
    return r.json()

def safe_get(path, timeout=8):
    try:
        return get_json(path, timeout=timeout)
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------- connection guard ----------
if not ENGINE_URL:
    st.error("ENGINE_URL env var is missing in the UI service.")
    st.stop()

# ---------- top controls ----------
cA, cB, cC = st.columns([1, 1, 6])
with cA:
    if st.button("🔄 Refresh now"):
        st.rerun()
with cB:
    st.caption(f"Auto-refresh: {int(REFRESH_MS/1000)}s" if st_autorefresh else "Auto-refresh: off (missing streamlit-autorefresh)")

if st_autorefresh:
    st_autorefresh(interval=REFRESH_MS, key="btc_autorefresh")

# ---------- Force calibrate section ----------
with st.expander("🧪 Force Sim / Calibrate (Debug)", expanded=False):
    st.write("This calls the engine **/calibrate** endpoint and shows what it picked.")
    if st.button("⚙️ Calibrate now"):
        cal = safe_get("/calibrate", timeout=12)
        if not cal.get("ok"):
            st.error("Calibrate failed")
            st.write(cal)
        else:
            st.success("Calibration ran.")
            st.write("**New min_conf:**", f"{cal.get('min_conf', 0)*100:.1f}%")
            if cal.get("picked_bucket"):
                st.write("**Picked bucket:**", cal["picked_bucket"])
            if cal.get("note"):
                st.info(cal["note"])
            stats = cal.get("bucket_stats", [])
            if stats:
                st.write("**Bucket stats (top results):**")
                st.dataframe(pd.DataFrame(stats), use_container_width=True)

# ---------- fetch engine data ----------
state = safe_get("/state")
if not state.get("ok"):
    st.error("Engine not reachable / error")
    st.write(state)
    st.stop()

metrics = safe_get("/metrics")
if not metrics.get("ok", True) and "error" in metrics:
    st.warning("Metrics endpoint error")
    st.write(metrics)

trades_resp = safe_get("/trades")
trades = trades_resp.get("trades", []) if trades_resp.get("ok") else []

tab_overview, tab_trades = st.tabs(["🏠 Overview", "🧾 Trades"])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${state.get('price', 0):,.2f}")
    c2.metric("Signal", state.get("signal", "—"))
    c3.metric("Confidence", f"{state.get('confidence', 0)*100:.1f}%")
    c4.metric("Equity", f"${metrics.get('equity', 0):,.2f}")

    st.caption(
        f"Min confidence (auto): **{state.get('min_conf', 0)*100:.1f}%** • "
        f"Last update: {state.get('time','')}"
    )

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
    candles = safe_get("/candles?tf=15m", timeout=12)
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
        st.write(candles)

    if state.get("open_trade"):
        st.subheader("📌 Open Paper Trade")
        st.json(state["open_trade"], expanded=False)

with tab_trades:
    st.subheader("🧾 Trade Journal")
    if not trades:
        st.info("No paper trades closed yet. Once one closes, you’ll see the table + equity curve here.")
    else:
        df = pd.DataFrame(trades)
        df2 = df.sort_values("opened")

        # equity curve using trade PnL (engine equity already exists; this is for visualization)
        start_eq = 250.0
        df2["equity_curve"] = start_eq + df2["pnl"].cumsum()

        fig2 = go.Figure(go.Scatter(x=df2["opened"], y=df2["equity_curve"], mode="lines", name="Equity"))
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df, use_container_width=True)
