import os
import json
import time
import requests
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

ENGINE_URL = os.getenv("ENGINE_URL", "").strip()
AUTO_REFRESH_MS = int(os.getenv("AUTO_REFRESH_MS", "5000"))

def normalize_engine_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url
    # no trailing slash
    url = url.rstrip("/")
    # if they gave railway.internal without port, default to 8080
    if ".railway.internal" in url:
        # if no explicit port in hostname
        host = url.split("://", 1)[1]
        if ":" not in host:
            url = url + ":8080"
    return url

ENGINE_URL = normalize_engine_url(ENGINE_URL)

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")
st.title("🧠 BTC AI Dashboard")
st.caption("Short-term • AI-filtered • Telegram alerts • Paper + Real (logged) P/L")

st_autorefresh(interval=AUTO_REFRESH_MS, key="btc_refresh")

def fetch_state():
    if not ENGINE_URL:
        return None, "ENGINE_URL not set"
    try:
        r = requests.get(f"{ENGINE_URL}/state", timeout=4)
        r.raise_for_status()
        return r.json(), ""
    except Exception as e:
        return None, str(e)

def render_candles(candles, momentum):
    if not candles:
        st.info("⏳ Waiting for candle data…")
        return

    xs = [c["time"] for c in candles]
    closes = [c["close"] for c in candles]

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=xs,
                open=[c["open"] for c in candles],
                high=[c["high"] for c in candles],
                low=[c["low"] for c in candles],
                close=closes,
            )
        ]
    )

    # Momentum arrows (simple: show arrow on latest candle, direction from momentum sign)
    if closes and momentum is not None:
        last_x = xs[-1]
        last_y = closes[-1]
        arrow = "▲" if momentum > 0 else "▼" if momentum < 0 else "•"
        fig.add_annotation(
            x=last_x,
            y=last_y,
            text=arrow,
            showarrow=False,
            font=dict(size=22),
            yshift=18 if momentum > 0 else -18 if momentum < 0 else 0,
        )

    fig.update_layout(
        height=360,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

state, err = fetch_state()

if not state:
    st.info("⏳ Waiting for engine data…")
    st.caption("Make sure ENGINE_URL points to your btc-engine service URL and /health works.")
    with st.expander("Debug (click to open)"):
        st.write(f"ENGINE_URL: {ENGINE_URL or '(not set)'}")
        st.error(err)
        st.code("Expected ENGINE_URL examples:\n- https://<btc-engine>.up.railway.app\n- http://btc-engine.railway.internal:8080")
    st.stop()

if state.get("error"):
    st.error(f"Engine error: {state['error']}")

price = float(state.get("price", 0) or 0)
rsi = float(state.get("rsi", 0) or 0)
trend = state.get("trend", "WAIT")
conf = int(state.get("confidence", 0) or 0)
notes = state.get("notes", "")
t = state.get("time", "--:--:--")

# derive momentum from notes if present
momentum = None
try:
    # notes like: "src=Coinbase • momentum=+0.00123"
    if "momentum=" in notes:
        momentum = float(notes.split("momentum=")[-1].strip())
except Exception:
    momentum = None

c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC Price", f"${price:,.2f}")
c2.metric("RSI (1m)", f"{rsi:.1f}")
c3.metric("Trend", trend)
c4.metric("🧠 Confidence", f"{conf}%")

st.caption(f"Last update: {t}  •  {notes}")

if conf >= 75:
    st.success("🔥 High-quality setup")
elif conf >= 60:
    st.warning("⚠️ Medium-quality signal")
else:
    st.info("⏳ Waiting for stronger conditions")

# Paper + Manual P/L
paper = state.get("paper", {}) or {}
manual = state.get("manual", {}) or {}

p_cash = float(paper.get("cash_usd", 0) or 0)
p_qty = float(paper.get("qty_btc", 0) or 0)
p_real = float(paper.get("realized_pl_usd", 0) or 0)
p_entry = paper.get("entry_price", None)
p_unreal = 0.0
if p_entry and p_qty > 0 and price > 0:
    p_unreal = (price - float(p_entry)) * p_qty
p_equity = p_cash + (p_qty * price)

m_qty = float(manual.get("qty_btc", 0) or 0)
m_cost = float(manual.get("cost_basis_usd", 0) or 0)
m_real = float(manual.get("realized_pl_usd", 0) or 0)
m_unreal = (m_qty * price) - m_cost
m_total = m_real + m_unreal

st.subheader("📈 P/L")
pc1, pc2 = st.columns(2)
with pc1:
    st.markdown("**🧾 Paper**")
    st.write(f"Equity: **${p_equity:,.2f}**")
    st.write(f"Realized P/L: **${p_real:,.2f}**")
    st.write(f"Unrealized P/L: **${p_unreal:,.2f}**")
    st.write(f"Holdings: **{p_qty:.8f} BTC**  •  Cash: **${p_cash:,.2f}**")
with pc2:
    st.markdown("**🧍 Real (logged)**")
    st.write(f"Total P/L: **${m_total:,.2f}**")
    st.write(f"Realized P/L: **${m_real:,.2f}**")
    st.write(f"Unrealized P/L: **${m_unreal:,.2f}**")
    st.write(f"Holdings: **{m_qty:.8f} BTC**  •  Cost basis: **${m_cost:,.2f}**")
    st.caption("Log trades in Telegram: /logbuy 100 or /logsell 100")

st.subheader("📊 BTC 1-Minute Candlesticks (Last 30 min)")
render_candles(state.get("candles", []), momentum)

if st.button("🔄 Refresh Now"):
    st.rerun()
