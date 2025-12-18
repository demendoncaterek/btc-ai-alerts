import os
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

APP_VERSION = "2025-12-17"
ENGINE_URL = os.getenv("ENGINE_URL", "http://localhost:8080").rstrip("/")
REFRESH_MS = int(os.getenv("REFRESH_MS", "10000"))

def fetch_json(path: str, timeout: float = 6.0):
    url = f"{ENGINE_URL}{path}"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None, f"{r.status_code}: {r.text[:200]}"
        return r.json(), None
    except Exception as e:
        return None, str(e)

def post_json(path: str, payload: Dict[str, Any], timeout: float = 10.0):
    url = f"{ENGINE_URL}{path}"
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code != 200:
            return None, f"{r.status_code}: {r.text[:200]}"
        return r.json(), None
    except Exception as e:
        return None, str(e)

def normalize_candles(raw: Any) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    rows: List[Dict[str, Any]] = []
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        for c in raw:
            rows.append({
                "time": c.get("time") or c.get("t") or c.get("timestamp"),
                "open": c.get("open"),
                "high": c.get("high"),
                "low": c.get("low"),
                "close": c.get("close"),
                "volume": c.get("volume", 0),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("time")
    return df

def candle_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=[go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="BTC"
    )])
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=20), xaxis_rangeslider_visible=False)
    return fig

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")
if st_autorefresh is not None:
    st_autorefresh(interval=REFRESH_MS, key="btc_autorefresh")

st.title("🧠 BTC AI Dashboard")
st.caption("15m execution • 1h + 6h bias • RSI(5m/15m) • MACD • ATR SL/TP • Peak/Dip watch • Paper/Real logs")

qp = st.query_params
page = qp.get("page", "overview")

nav1, nav2, nav3 = st.columns([1, 1, 2])
with nav1:
    if st.button("🏠 Overview", use_container_width=True):
        st.query_params["page"] = "overview"; st.rerun()
with nav2:
    if st.button("🧾 Trades", use_container_width=True):
        st.query_params["page"] = "trades"; st.rerun()
with nav3:
    if st.button("🔄 Refresh now", use_container_width=True):
        st.rerun()

with st.expander("Connection Debug", expanded=False):
    st.write("ENGINE_URL:", ENGINE_URL)
    h, h_err = fetch_json("/health")
    if h_err: st.error(f"/health failed: {h_err}")
    else: st.success(f"/health ok: {h.get('ok')} • {h.get('product')} • v{h.get('version')}")
    s, s_err = fetch_json("/state")
    if s_err: st.error(f"/state failed: {s_err}")
    else: st.info(f"/state ok: {s.get('ok')} • last: {s.get('iso')}")

state, err = fetch_json("/state")
if err or not state:
    st.error("Engine is NOT reachable from the UI right now.")
    st.code(err or "Unknown error")
    st.stop()

if page == "overview":
    c1, c2, c3, c4 = st.columns(4)
    price = state.get("price")
    rsi5 = state.get("rsi_5m")
    signal = state.get("signal", "WAIT")
    conf = state.get("confidence", 0)

    c1.metric("BTC Price", f"${price:,.2f}" if isinstance(price, (int, float)) else "—")
    c2.metric("RSI (5m)", f"{rsi5:.1f}" if isinstance(rsi5, (int, float)) else "—")
    c3.metric("Signal", signal)
    c4.metric("Confidence", f"{float(conf):.0f}%" if isinstance(conf, (int, float)) else "—")

    st.info(state.get("reason") or "Waiting for setup…")

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Trend", state.get("trend", "UNKNOWN"))
    t2.metric("Bias 1h", state.get("bias_1h", "UNKNOWN"))
    t3.metric("Bias 6h", state.get("bias_6h", "UNKNOWN"))
    atr15 = state.get("atr_15m")
    t4.metric("ATR (15m)", f"{atr15:,.2f}" if isinstance(atr15, (int, float)) else "—")

    sl = state.get("sl"); tp = state.get("tp")
    if isinstance(sl, (int, float)) or isinstance(tp, (int, float)):
        st.write("**Suggested levels (ATR-based, educational):**")
        e1, e2, e3 = st.columns(3)
        e1.metric("Entry", f"${price:,.2f}" if isinstance(price, (int, float)) else "—")
        e2.metric("Stop (SL)", f"${sl:,.2f}" if isinstance(sl, (int, float)) else "—")
        e3.metric("Target (TP)", f"${tp:,.2f}" if isinstance(tp, (int, float)) else "—")

    if bool(state.get("peak_watch")) or bool(state.get("dip_watch")):
        msg = []
        if state.get("peak_watch"): msg.append(f"📈 Peak Watch near {state.get('peak_180m')}")
        if state.get("dip_watch"): msg.append(f"📉 Dip Watch near {state.get('dip_180m')}")
        st.warning(" • ".join(msg))

    df = normalize_candles(state.get("candles_15m"))
    if df.empty:
        c, c_err = fetch_json("/candles?tf=15m&limit=200", timeout=10)
        if not c_err:
            df = normalize_candles(c.get("candles"))
    if not df.empty:
        st.plotly_chart(candle_chart(df), use_container_width=True)
    else:
        st.warning("No candle data yet.")

    cal = (state.get("calibration") or {})
    if cal.get("enabled"):
        with st.expander("Calibration (educational)", expanded=False):
            st.write(f"Horizon: {cal.get('horizon_bars')} bars • Samples: {cal.get('samples_total')}")
            st.write(f"Current confidence bucket: {cal.get('current_bucket')} • Win rate: {cal.get('current_bucket_win_rate')}")
            st.dataframe(pd.DataFrame(cal.get("buckets", [])), use_container_width=True, height=280)

    paper = state.get("paper") or {}
    st.subheader("Trade Journal (quick view)")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Paper USD", f"${paper.get('usd', 0):,.2f}")
    p2.metric("Paper BTC", f"{paper.get('btc', 0):.6f}")
    p3.metric("Paper Equity", f"${paper.get('equity', 0):,.2f}")
    p4.metric("Paper Realized PnL", f"${paper.get('realized_pnl', 0):,.2f}")

    st.caption(f"Last update: {state.get('iso')} • UI v{APP_VERSION}")

elif page == "trades":
    st.subheader("Trades")

    paper_resp, p_err = fetch_json("/trades/paper?limit=1000", timeout=10)
    real_resp, r_err = fetch_json("/trades/real?limit=1000", timeout=10)

    if p_err: st.error(f"Paper trades error: {p_err}")
    if r_err: st.error(f"Real trades error: {r_err}")

    if paper_resp and paper_resp.get("paper"):
        paper = paper_resp["paper"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Paper USD", f"${paper.get('usd', 0):,.2f}")
        c2.metric("Paper BTC", f"{paper.get('btc', 0):.6f}")
        c3.metric("Paper Equity", f"${paper.get('equity', 0):,.2f}")
        c4.metric("Realized PnL", f"${paper.get('realized_pnl', 0):,.2f}")

        st.write("**Paper Trades**")
        p_trades = paper_resp.get("trades", [])
        if p_trades:
            st.dataframe(pd.DataFrame(p_trades)[::-1], use_container_width=True, height=320)
        else:
            st.info("No paper trades yet.")

        if st.button("🧹 Reset paper trades", type="secondary"):
            _, e = post_json("/paper/reset", {})
            if e: st.error(e)
            else: st.success("Reset."); st.rerun()

    st.divider()
    st.write("**Real Trades (logged)**")
    if real_resp and real_resp.get("trades"):
        st.dataframe(pd.DataFrame(real_resp["trades"])[::-1], use_container_width=True, height=320)
    else:
        st.info("No real trades logged yet.")

    st.divider()
    st.write("**Log a real trade (manual)**")
    with st.form("log_trade"):
        side = st.selectbox("Side", ["BUY", "SELL"])
        qty = st.number_input("Qty (BTC)", min_value=0.0, value=0.0, step=0.0001, format="%.6f")
        price = st.number_input("Price (USD)", min_value=0.0, value=float(state.get("price") or 0.0), step=10.0)
        note = st.text_input("Note", value="")
        submitted = st.form_submit_button("Log trade")
        if submitted:
            resp, e = post_json("/trades/real", {"side": side, "qty": qty, "price": price, "note": note})
            if e: st.error(e)
            else: st.success("Logged."); st.json(resp.get("trade")); st.rerun()
