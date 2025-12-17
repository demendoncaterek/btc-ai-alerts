"""
btc_ui.py
Streamlit UI for btc_engine.py

Overview:
- Price / RSI / Signal / Confidence
- Bias (1h + 6h), ATR SL/TP, Peak/Dip watch
- Candles + momentum arrows
Trades:
- Paper equity + trade tables
- Real trades table + manual log form
Connection Debug:
- Ping /health and /state
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


ENGINE_URL = os.getenv("ENGINE_URL", "http://localhost:8080").rstrip("/")
REFRESH_MS = int(os.getenv("REFRESH_MS", "10000"))  # 10s
REQ_TIMEOUT = float(os.getenv("REQ_TIMEOUT", "6"))

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")

page = st.query_params.get("page", "overview").lower()
if page not in ("overview", "trades"):
    page = "overview"


def get_json(path: str) -> Dict[str, Any]:
    r = requests.get(f"{ENGINE_URL}{path}", timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.json()


def post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{ENGINE_URL}{path}", json=payload, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.json()


def try_fetch_state() -> Dict[str, Any]:
    try:
        return get_json("/state")
    except Exception as e:
        return {"ok": False, "error": str(e)}


def connection_debug_box(err: Optional[str] = None) -> None:
    with st.expander("Connection Debug", expanded=True):
        st.write("ENGINE_URL:", ENGINE_URL)
        c1, c2 = st.columns(2)
        if c1.button("Ping /health"):
            try:
                st.json(get_json("/health"))
            except Exception as e:
                st.error(str(e))
        if c2.button("Ping /state"):
            try:
                st.json(get_json("/state"))
            except Exception as e:
                st.error(str(e))

        if err:
            st.error(err)
            st.markdown(
                """
**Fix checklist (Railway):**
1) Open your engine’s public URL in a browser and go to `/health`. It must return JSON.  
2) If it doesn't, your engine service is not running or start command is wrong.  
3) If public works but internal fails, set UI `ENGINE_URL` to the public engine URL temporarily.
                """
            )


def normalize_candles(candles: Any) -> pd.DataFrame:
    if not isinstance(candles, list) or len(candles) == 0:
        return pd.DataFrame()
    if isinstance(candles[0], dict):
        df = pd.DataFrame(candles)
        for col in ["iso", "open", "high", "low", "close"]:
            if col not in df.columns:
                return pd.DataFrame()
        df["iso"] = pd.to_datetime(df["iso"], errors="coerce")
        df = df.dropna(subset=["iso", "open", "high", "low", "close"])
        return df
    return pd.DataFrame()


def plot_candles(df: pd.DataFrame, markers: List[Dict[str, Any]]) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["iso"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="BTC",
            )
        ]
    )

    if markers:
        bulls_x, bulls_y, bears_x, bears_y = [], [], [], []
        for m in markers:
            t = m.get("t")
            p = m.get("price")
            if t is None or p is None:
                continue
            if m.get("type") == "bull":
                bulls_x.append(t); bulls_y.append(p)
            elif m.get("type") == "bear":
                bears_x.append(t); bears_y.append(p)

        if bulls_x:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(bulls_x),
                    y=bulls_y,
                    mode="markers",
                    name="Bullish momentum",
                    marker=dict(symbol="arrow-up", size=10),
                )
            )
        if bears_x:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(bears_x),
                    y=bears_y,
                    mode="markers",
                    name="Bearish momentum",
                    marker=dict(symbol="arrow-down", size=10),
                )
            )

    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
    return fig


if st_autorefresh:
    st_autorefresh(interval=REFRESH_MS, key="btc_autorefresh")


st.title("🧠 BTC AI Dashboard")
st.caption("15m execution • 1h + 6h bias • RSI/MACD • ATR SL/TP • Peak/Dip watch • Paper/Real logs")

nav1, nav2 = st.columns(2)
if nav1.button("🏠 Overview", use_container_width=True):
    st.query_params["page"] = "overview"
    st.rerun()
if nav2.button("🗒️ Trades", use_container_width=True):
    st.query_params["page"] = "trades"
    st.rerun()

if st.button("🔄 Refresh now"):
    st.rerun()

state = try_fetch_state()
if not state.get("ok"):
    connection_debug_box(state.get("error", "Engine not reachable"))
    st.stop()

price = float(state.get("price", 0.0) or 0.0)
signal = state.get("signal", "WAIT")
confidence = float(state.get("confidence", 0.0) or 0.0)
bias1h = state.get("bias_1h", "—")
bias6h = state.get("bias_6h", "—")
rsi_val = state.get("rsi", None)
atr_val = state.get("atr", None)
sl = state.get("sl", None)
tp = state.get("tp", None)
peak_watch = state.get("peak_watch", None)
iso = state.get("iso", "")

if page == "overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BTC Price", f"${price:,.2f}")
    c2.metric("RSI (base)", "--" if rsi_val is None else f"{float(rsi_val):.1f}")
    c3.metric("Signal", signal)
    c4.metric("Confidence", f"{confidence:.0f}%")

    st.info("⏳ Waiting for high-probability setup" if signal == "WAIT" else f"📌 Setup detected: **{signal}**")
    st.caption(f"Last update: {iso} • Source: Coinbase Exchange • Bias: 1h={bias1h}, 6h={bias6h}")

    b1, b2 = st.columns(2)
    with b1:
        st.subheader("Peak/Dip watch")
        if isinstance(peak_watch, dict):
            st.json(peak_watch)
        else:
            st.write("No active peak/dip watch.")
    with b2:
        st.subheader("Risk levels (ATR)")
        if atr_val is None:
            st.write("ATR not ready yet.")
        else:
            st.write(f"ATR: {float(atr_val):.2f}")
            if sl is not None and tp is not None:
                st.write(f"SL: {float(sl):,.2f}")
                st.write(f"TP: {float(tp):,.2f}")
            else:
                st.write("No SL/TP because signal is WAIT.")

    st.subheader("📊 BTC Candlesticks")
    df = normalize_candles(state.get("candles"))
    if df.empty:
        st.warning("Candle data incomplete — waiting for full feed.")
    else:
        st.plotly_chart(plot_candles(df, state.get("markers", []) or []), use_container_width=True)

    with st.expander("Why is it saying this? (/explain)"):
        try:
            st.json(get_json("/explain"))
        except Exception as e:
            st.error(str(e))

else:
    st.header("🗒️ Trade Journal")

    try:
        trades = get_json("/trades")
    except Exception as e:
        connection_debug_box(str(e))
        st.stop()

    paper = trades.get("paper", {}) or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Paper USD", f"${float(paper.get('usd', 0.0)):,.2f}")
    c2.metric("Paper BTC", f"{float(paper.get('btc', 0.0)):.6f}")
    c3.metric("Paper Equity", f"${float(paper.get('equity', 0.0)):,.2f}", f"P/L ${float(paper.get('pnl', 0.0)):,.2f}")

    st.subheader("🧪 Paper Trades")
    paper_rows = trades.get("paper_trades", []) or []
    if paper_rows:
        st.dataframe(pd.DataFrame(paper_rows), use_container_width=True)
    else:
        st.info("No paper trades yet.")

    st.subheader("💰 Real Trades (Logged)")
    real_rows = trades.get("real_trades", []) or []
    if real_rows:
        st.dataframe(pd.DataFrame(real_rows), use_container_width=True)
    else:
        st.info("No real trades logged yet.")

    st.subheader("Log a real trade (manual)")
    with st.form("log_real"):
        side = st.selectbox("Side", ["BUY", "SELL"])
        qty = st.number_input("Qty (BTC)", min_value=0.0, value=0.0, step=0.0001, format="%.6f")
        note = st.text_input("Note (optional)")
        if st.form_submit_button("Log trade"):
            try:
                out = post_json("/real/log", {"side": side, "qty": qty, "note": note})
                if out.get("ok"):
                    st.success("Logged.")
                    st.rerun()
                else:
                    st.error(out)
            except Exception as e:
                st.error(str(e))
