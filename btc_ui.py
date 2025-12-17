import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# -------------------------
# CONFIG
# -------------------------
ENGINE_URL = os.getenv("ENGINE_URL", "http://localhost:8080").rstrip("/")
REFRESH_MS = int(os.getenv("REFRESH_MS", "15000"))

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")

# Auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_MS, key="btc_autorefresh")
except Exception:
    pass

def fetch_state():
    try:
        r = requests.get(f"{ENGINE_URL}/state", timeout=8)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

state = fetch_state()

# Query param page (Streamlit-safe “/trades” equivalent)
qp = st.query_params
page = (qp.get("page", "dashboard") or "dashboard").lower()

# Top header always
st.title("🧠 BTC AI Dashboard")
st.caption("15m execution • 1h+4h bias • ATR SL/TP • Telegram /status + /explain")

# If engine not ready, still show debug (don’t blank out)
if not state.get("ok"):
    st.warning("⏳ Engine not ready yet.")
    st.write("Check ENGINE_URL and that engine /health is reachable.")
    with st.expander("Debug", expanded=True):
        st.write("ENGINE_URL:", ENGINE_URL)
        st.json(state)
        if st.button("Ping /health"):
            try:
                r = requests.get(f"{ENGINE_URL}/health", timeout=8)
                st.json(r.json())
            except Exception as e:
                st.error(str(e))
    st.stop()

# Simple nav (works on mobile too)
nav = st.radio("View", ["Dashboard", "Trades"], horizontal=True, index=0 if page != "trades" else 1)
if nav == "Trades":
    st.query_params["page"] = "trades"
    page = "trades"
else:
    st.query_params["page"] = "dashboard"
    page = "dashboard"

# Convenience link (bookmark this for “/trades”)
st.caption("Bookmark Trades: add `?page=trades` to your URL.")

# -------------------------
# Shared fields
# -------------------------
price = float(state.get("price", 0))
signal = state.get("signal", "WAIT")
confidence = int(state.get("confidence", 0))
bias = state.get("bias", "NEUTRAL")
bias1 = state.get("bias_1h", "?")
bias4 = state.get("bias_4h", "?")
rsi_v = state.get("rsi", None)
atr_v = state.get("atr", None)
sl = state.get("sl", None)
tp = state.get("tp", None)
rr = state.get("rr", None)

paper = state.get("paper_summary", {}) or {}
real = state.get("real_summary", {}) or {}

# -------------------------
# DASHBOARD
# -------------------------
if page == "dashboard":
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BTC Price", f"${price:,.2f}")
    c2.metric("RSI (15m)", f"{rsi_v}")
    c3.metric("Bias", f"{bias} (1h={bias1}, 4h={bias4})")
    c4.metric("Signal", signal)
    c5.metric("Confidence", f"{confidence}%")

    a1, a2, a3 = st.columns(3)
    a1.metric("Paper Equity", f"${paper.get('equity', 0):,.2f}")
    a2.metric("Paper P/L", f"${paper.get('pnl', 0):,.2f}")
    a3.metric("Logged Real Unrealized P/L", f"${real.get('unrealized', 0):,.2f}")

    # SL/TP panel (suggestion only)
    st.subheader("🎯 Risk Plan (Suggestion)")
    if sl is not None and tp is not None:
        st.success(f"Suggested SL: ${sl:,.2f}  |  TP: ${tp:,.2f}  |  R:R≈{rr}")
    else:
        st.info("No SL/TP suggestion right now (no active BUY/SELL setup).")

    # Explain panel
    st.subheader("🧠 Why it thinks this")
    st.write(state.get("reason", ""))
    with st.expander("Full confidence breakdown"):
        for r in state.get("confidence_reasons", [])[:50]:
            st.write(f"• {r}")

    # Chart
    st.subheader("📈 BTC Candlesticks (15m) + Momentum Arrows + Signals")
    candles = state.get("candles", []) or []
    events = state.get("events", []) or []
    macd_hist_series = state.get("macd_hist_series", []) or []

    if not candles:
        st.info("Waiting for candle data…")
    else:
        df = pd.DataFrame(candles)
        df["dt"] = pd.to_datetime(df["ts"], unit="s")

        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df["dt"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="BTC",
            )
        )

        # Momentum arrows (like your old look): use MACD histogram sign per candle if available
        if len(macd_hist_series) == len(df):
            y_pos = df["low"] * 0.999  # arrows slightly below candle
            colors = ["lime" if m > 0 else "red" for m in macd_hist_series]
            symbols = ["triangle-up" if m > 0 else "triangle-down" for m in macd_hist_series]
            fig.add_trace(
                go.Scatter(
                    x=df["dt"],
                    y=y_pos,
                    mode="markers",
                    marker=dict(size=9, symbol=symbols, color=colors, opacity=0.85),
                    name="Momentum",
                    hoverinfo="skip",
                )
            )

        # Event markers (BUY/SELL/CRAZY)
        if events:
            ev = pd.DataFrame(events)
            ev["dt"] = pd.to_datetime(ev["ts"])

            sym = []
            col = []
            for t in ev["type"]:
                t = str(t).upper()
                if t == "BUY":
                    sym.append("triangle-up"); col.append("lime")
                elif t == "SELL":
                    sym.append("triangle-down"); col.append("red")
                elif "DIP" in t:
                    sym.append("triangle-left"); col.append("cyan")
                elif "PEAK" in t:
                    sym.append("triangle-right"); col.append("orange")
                else:
                    sym.append("circle"); col.append("white")

            fig.add_trace(
                go.Scatter(
                    x=ev["dt"],
                    y=ev["price"],
                    mode="markers",
                    marker=dict(size=14, symbol=sym, color=col),
                    name="Signals",
                    text=ev.get("label", ev["type"]),
                    hovertemplate="%{text}<br>$%{y:,.2f}<br>%{x}<extra></extra>",
                )
            )

        fig.update_layout(height=560, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# TRADES JOURNAL
# -------------------------
else:
    st.subheader("📒 Trade Journal")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### 📄 Paper Trades")
        ptrades = state.get("paper_trades", []) or []
        if ptrades:
            pdf = pd.DataFrame(ptrades)
            st.dataframe(pdf, use_container_width=True, height=360)
        else:
            st.info("No paper trades logged yet.")

    with colB:
        st.markdown("### 🧾 Logged Real Trades (manual)")
        rtrades = state.get("real_trades", []) or []
        if rtrades:
            rdf = pd.DataFrame(rtrades)
            st.dataframe(rdf, use_container_width=True, height=360)
        else:
            st.info("No real trades logged yet. Use Telegram: /logbuy 100 or /logsell 100")

    st.markdown("### Summary")
    s1, s2, s3 = st.columns(3)
    s1.metric("Paper Equity", f"${paper.get('equity', 0):,.2f}")
    s2.metric("Paper P/L", f"${paper.get('pnl', 0):,.2f}")
    s3.metric("Logged Real Unrealized P/L", f"${real.get('unrealized', 0):,.2f}")

    st.caption("Tip: Bookmark this page using `?page=trades`.")
