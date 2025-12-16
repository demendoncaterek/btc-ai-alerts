import os
import json
from datetime import datetime

import requests
import numpy as np
import streamlit as st
import plotly.graph_objects as go

try:
    from streamlit_autorefresh import st_autorefresh  # pip: streamlit-autorefresh
except Exception:
    st_autorefresh = None

# =========================
# CONFIG
# =========================
DEFAULT_ENGINE_URL = "http://btc-engine.railway.internal:8080"  # Railway internal + engine port
REFRESH_MS = int(os.getenv("UI_REFRESH_MS", "5000"))


def normalize_engine_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        raw = DEFAULT_ENGINE_URL

    # ensure scheme
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw

    raw = raw.rstrip("/")

    # If railway internal and no port, add :8080 (engine listens on 8080 in your setup)
    if "railway.internal" in raw:
        # very light parsing without extra deps
        # raw is like http://host or http://host:port
        try:
            after_scheme = raw.split("://", 1)[1]
            host_and_path = after_scheme.split("/", 1)
            hostport = host_and_path[0]
            path = "/" + host_and_path[1] if len(host_and_path) > 1 else ""

            if ":" not in hostport:
                hostport = f"{hostport}:8080"
                raw = raw.split("://", 1)[0] + "://" + hostport + path
        except Exception:
            pass

    return raw


ENGINE_URL = normalize_engine_url(os.getenv("ENGINE_URL", ""))

# =========================
# STREAMLIT SETUP
# =========================
st.set_page_config(page_title="BTC AI Dashboard", layout="wide")
st.title("🧠 BTC AI Dashboard")
st.caption("Short-term • AI-filtered • Telegram alerts • Paper + Real (logged) P/L")

# Auto-refresh
if st_autorefresh:
    st_autorefresh(interval=REFRESH_MS, key="btc_refresh")
else:
    st.warning("Auto-refresh helper not installed. Add `streamlit-autorefresh` to requirements.txt.")


# =========================
# ENGINE FETCH
# =========================
def fetch_json(path: str, timeout=4):
    url = f"{ENGINE_URL}{path}"
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} from {url}: {r.text[:200]}")
    return r.json()


def fetch_state():
    return fetch_json("/state", timeout=4)


def ping_health():
    return fetch_json("/health", timeout=4)


# =========================
# UI HELPERS
# =========================
def format_money(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"


def format_btc(x):
    try:
        return f"{float(x):.6f} BTC"
    except Exception:
        return "0.000000 BTC"


def safe_num(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def render_candles_with_momentum(candles, momentum=0.0, signal=None):
    if not candles:
        st.info("⏳ Waiting for candle data…")
        return

    x = [c["time"] for c in candles]
    opens = [c["open"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=x,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                name="BTC",
            )
        ]
    )

    # Momentum arrow (overall)
    last_x = x[-1]
    last_close = closes[-1]
    if momentum > 0:
        fig.add_trace(
            go.Scatter(
                x=[last_x],
                y=[last_close],
                mode="markers+text",
                text=["⬆️"],
                textposition="top center",
                name="Momentum Up",
            )
        )
    elif momentum < 0:
        fig.add_trace(
            go.Scatter(
                x=[last_x],
                y=[last_close],
                mode="markers+text",
                text=["⬇️"],
                textposition="bottom center",
                name="Momentum Down",
            )
        )

    # Signal marker
    if signal == "BUY":
        fig.add_trace(
            go.Scatter(
                x=[last_x],
                y=[last_close],
                mode="markers+text",
                text=["🟢 BUY"],
                textposition="top center",
                name="BUY",
            )
        )
    elif signal == "SELL":
        fig.add_trace(
            go.Scatter(
                x=[last_x],
                y=[last_close],
                mode="markers+text",
                text=["🔴 SELL"],
                textposition="bottom center",
                name="SELL",
            )
        )

    fig.update_layout(
        height=360,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================
# MAIN RENDER
# =========================
state = None
err = None

try:
    state = fetch_state()
except Exception as e:
    err = str(e)

if not state:
    st.info("⏳ Waiting for engine data…")

    st.caption("Make sure ENGINE_URL points to your btc-engine service URL and /state works.")
    with st.expander("Debug (click to open)"):
        st.write("ENGINE_URL:", ENGINE_URL)
        if err:
            st.error(err)

        colA, colB = st.columns(2)
        with colA:
            if st.button("Ping engine /health"):
                try:
                    health = ping_health()
                    st.success(f"Engine healthy: {health}")
                except Exception as e:
                    st.error(f"Health check failed: {e}")

        with colB:
            st.write("Expected in Railway UI service variables:")
            st.code("ENGINE_URL=http://btc-engine.railway.internal:8080")

    st.stop()

# Engine error banner
if state.get("error"):
    st.error(f"Engine error: {state['error']}")

# Top metrics
price = safe_num(state.get("price", 0))
rsi = state.get("rsi", 0)
trend = state.get("trend", "WAIT")
confidence = safe_num(state.get("confidence", 0))
momentum = safe_num(state.get("momentum", 0))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("BTC Price", format_money(price))
c2.metric("RSI (1m)", rsi)
c3.metric("Trend", trend)
c4.metric("🧠 Confidence", f"{int(confidence)}%")
c5.metric("Momentum", f"{momentum:+.5f}")

st.caption(f"Last update: {state.get('time','--:--:--')}  •  {state.get('notes','')}")

# Signal quality banner
if confidence >= 75:
    st.success("🔥 High-quality setup")
elif confidence >= 60:
    st.warning("⚠️ Medium-quality signal")
else:
    st.info("⏳ Waiting for stronger conditions")

# Paper trading + real logged P/L
paper = state.get("paper", {}) or {}
manual = state.get("manual", {}) or {}

st.subheader("🧪 Paper Trading + 💵 Manual (Logged) P/L")

pc1, pc2, pc3, pc4 = st.columns(4)
pc1.metric("Paper Enabled", "Yes ✅" if paper.get("enabled") else "No ❌")
pc2.metric("Paper Equity", format_money(paper.get("equity", 0)))
pc3.metric("Paper Position", format_btc(paper.get("position_btc", 0)))
pc4.metric("Paper P/L (Unreal.)", format_money(paper.get("unrealized_pl", 0)))

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Manual Position", format_btc(manual.get("position_btc", 0)))
mc2.metric("Manual Avg Entry", format_money(manual.get("avg_entry", 0)))
mc3.metric("Manual Realized P/L", format_money(manual.get("realized_pl", 0)))
mc4.metric("Manual Unreal. P/L", format_money(manual.get("unrealized_pl", 0)))

st.caption(
    "Telegram commands: /status • /logbuy 100 • /logsell 50 • /paper on/off • /resetpaper"
)

# Candles
st.subheader("📊 BTC 1-Minute Candlesticks (Last 30 min) + Momentum Arrows")
render_candles_with_momentum(
    state.get("candles", []),
    momentum=momentum,
    signal=("BUY" if trend == "BUY" else "SELL" if trend == "SELL" else None),
)

# Trades tables (optional)
col1, col2 = st.columns(2)
with col1:
    st.subheader("🧪 Paper Trades (latest)")
    paper_trades = paper.get("trades", []) or []
    if paper_trades:
        st.dataframe(paper_trades, use_container_width=True)
    else:
        st.caption("No paper trades yet.")

with col2:
    st.subheader("💵 Manual Logs (latest)")
    manual_logs = manual.get("logs", []) or []
    if manual_logs:
        st.dataframe(manual_logs, use_container_width=True)
    else:
        st.caption("No manual logs yet.")

# Manual refresh button (still useful)
if st.button("🔄 Refresh Now"):
    st.rerun()
