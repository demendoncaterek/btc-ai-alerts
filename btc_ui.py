#!/usr/bin/env python3
"""
btc_ui.py

Streamlit UI for the BTC engine.

Streamlit usually won't give you a true /trades route.
This UI uses a query param instead:
  - Overview: /
  - Trades:   /?page=trades
"""

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
REFRESH_MS = int(os.getenv("REFRESH_MS", "15000"))

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")

qp = st.query_params
page = (qp.get("page", "overview") or "overview").lower()

st.title("🧠 BTC AI Dashboard")
st.caption("15m execution • 1h+4h bias • ATR SL/TP • Telegram /status + /explain")

cols_top = st.columns([1, 1, 3])
with cols_top[0]:
    st.page_link(".", label="🏠 Overview")
with cols_top[1]:
    st.page_link("?page=trades", label="📒 Trades Journal")

if st_autorefresh:
    st_autorefresh(interval=REFRESH_MS, key="btc_autorefresh")


@st.cache_data(ttl=8)
def fetch_json(path: str):
    r = requests.get(f"{ENGINE_URL}{path}", timeout=8)
    return r.status_code, r.json()


code, state = fetch_json("/state")

if not isinstance(state, dict) or not state.get("ok"):
    st.warning("⏳ Engine not ready yet.")
    st.write("Check ENGINE_URL and that engine /health is reachable.")
    with st.expander("Debug", expanded=True):
        st.write("ENGINE_URL:", ENGINE_URL)
        if isinstance(state, dict):
            st.json(state)
        else:
            st.write("Bad response:", code, state)

        if st.button("Ping /health"):
            try:
                c2, j2 = fetch_json("/health")
                st.write("HTTP", c2)
                st.json(j2)
            except Exception as e:
                st.error(str(e))
    st.stop()


price = float(state.get("price") or 0.0)
signal = state.get("signal", "WAIT")
conf = float(state.get("confidence") or 0.0)
trend_1h = state.get("trend_1h", "UNKNOWN")
trend_4h = state.get("trend_4h", "UNKNOWN")
rsi_base = state.get("rsi_base")
atr = state.get("atr")
suggested = state.get("suggested") or {}
sl = suggested.get("sl")
tp = suggested.get("tp")

dip_watch = state.get("dip_watch")
peak_watch = state.get("peak_watch")

candles = state.get("candles") or []
markers = state.get("markers") or []


if page != "trades":
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BTC Price", f"${price:,.2f}")
    c2.metric("RSI (base)", f"{rsi_base:.1f}" if rsi_base is not None else "—")
    c3.metric("Signal", signal)
    c4.metric("Confidence", f"{conf:.0%}")
    c5.metric("Bias (1h / 4h)", f"{trend_1h} / {trend_4h}")

    if conf >= 0.72 and signal in ("BUY", "SELL"):
        st.success(f"High-confidence setup: **{signal}** ({conf:.0%})")
    elif conf >= 0.60:
        st.warning("⚠️ Medium-confidence setup")
    else:
        st.info("⏳ Waiting for stronger conditions")

    if peak_watch:
        st.warning(f"📈 Peak watch: price near {peak_watch.get('window_min')}m high.")
    if dip_watch:
        st.warning(f"📉 Dip watch: price near {dip_watch.get('window_min')}m low.")

    if sl and tp and atr:
        st.caption(f"Suggested SL/TP (ATR): SL **{sl:,.2f}** • TP **{tp:,.2f}** • ATR(base) {atr:.2f}")

    if candles:
        df = pd.DataFrame(candles)
        df["dt"] = pd.to_datetime(df["t"], unit="s")
        df = df.sort_values("dt")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df["dt"], open=df["o"], high=df["h"], low=df["l"], close=df["c"], name="BTC"
        ))

        if markers:
            mdf = pd.DataFrame(markers)
            mdf["dt"] = pd.to_datetime(mdf["t"], unit="s")
            bull = mdf[mdf["kind"] == "bull"]
            bear = mdf[mdf["kind"] == "bear"]

            if not bull.empty:
                fig.add_trace(go.Scatter(
                    x=bull["dt"], y=bull["price"], mode="markers",
                    marker=dict(symbol="arrow-up", size=10),
                    name="Bullish momentum"
                ))
            if not bear.empty:
                fig.add_trace(go.Scatter(
                    x=bear["dt"], y=bear["price"], mode="markers",
                    marker=dict(symbol="arrow-down", size=10),
                    name="Bearish momentum"
                ))

        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False,
            legend_orientation="h",
        )

        st.subheader("📊 BTC Candlesticks")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No candle data yet.")

    st.divider()
    st.subheader("🧾 Paper Trading Controls (optional)")
    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        if st.button("Paper BUY (all USD)"):
            try:
                r = requests.post(f"{ENGINE_URL}/paper/buy", timeout=8)
                st.json(r.json())
                st.cache_data.clear()
            except Exception as e:
                st.error(str(e))
    with cB:
        if st.button("Paper SELL (all BTC)"):
            try:
                r = requests.post(f"{ENGINE_URL}/paper/sell", timeout=8)
                st.json(r.json())
                st.cache_data.clear()
            except Exception as e:
                st.error(str(e))
    with cC:
        paper = state.get("paper") or {}
        st.caption(f"Paper equity: ${paper.get('equity', 0):,.2f} • USD: ${paper.get('usd', 0):,.2f} • BTC: {paper.get('btc', 0):.6f}")

else:
    st.subheader("📒 Trades Journal")

    code_t, trades = fetch_json("/trades")
    if code_t != 200 or not isinstance(trades, dict) or not trades.get("ok"):
        st.error("Could not load /trades from engine.")
        st.write("HTTP", code_t)
        st.json(trades)
        st.stop()

    paper = trades.get("paper") or []
    real = trades.get("real") or []

    tab1, tab2 = st.tabs(["Paper trades", "Logged real trades (manual)"])

    with tab1:
        if not paper:
            st.info("No paper trades yet.")
        else:
            pdf = pd.DataFrame(paper)
            st.dataframe(pdf, use_container_width=True, hide_index=True)
            st.download_button("Download paper trades CSV", pdf.to_csv(index=False), file_name="paper_trades.csv")

    with tab2:
        if not real:
            st.info("No logged real trades yet.")
        else:
            rdf = pd.DataFrame(real)
            st.dataframe(rdf, use_container_width=True, hide_index=True)
            st.download_button("Download real trades CSV", rdf.to_csv(index=False), file_name="real_trades.csv")

    st.caption("Bookmark: **/?page=trades** (Streamlit routing limitation).")
