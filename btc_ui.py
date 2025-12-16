import json
import os
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go

STATE_FILE = os.getenv("STATE_FILE", "btc_state.json")      # produced by engine
TRADE_FILE = os.getenv("TRADE_FILE", "btc_trades.json")     # produced by engine
UI_REFRESH_SEC = int(os.getenv("UI_REFRESH_SEC", "5"))      # page reload interval

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")
st.title("🧠 BTC AI Dashboard")
st.caption("Short-term • AI-filtered • Telegram alerts")

# ✅ Auto refresh without needing streamlit-extras
st.markdown(
    f"<meta http-equiv='refresh' content='{UI_REFRESH_SEC}'>",
    unsafe_allow_html=True
)


def safe_load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if not raw:
                return None
            return json.loads(raw)
    except:
        return None


def render_chart(candles, trades):
    if not candles:
        st.info("⏳ Waiting for candle data…")
        return

    x = [datetime.fromtimestamp(c["ts"]) for c in candles]

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=x,
                open=[c["open"] for c in candles],
                high=[c["high"] for c in candles],
                low=[c["low"] for c in candles],
                close=[c["close"] for c in candles],
                name="BTC",
            )
        ]
    )

    # ---- Momentum arrows (based on 5-candle momentum) ----
    closes = [c["close"] for c in candles]
    arrows_x = []
    arrows_y = []
    arrows_symbol = []

    for i in range(len(candles)):
        if i < 5:
            continue
        mom = (closes[i] - closes[i - 5]) / closes[i - 5] if closes[i - 5] else 0
        # only show arrows if momentum is "noticeable" to reduce clutter
        if abs(mom) < 0.0008:
            continue

        arrows_x.append(x[i])
        if mom > 0:
            arrows_y.append(candles[i]["high"] * 1.0005)
            arrows_symbol.append("triangle-up")
        else:
            arrows_y.append(candles[i]["low"] * 0.9995)
            arrows_symbol.append("triangle-down")

    if arrows_x:
        fig.add_trace(
            go.Scatter(
                x=arrows_x,
                y=arrows_y,
                mode="markers",
                marker=dict(size=10, symbol=arrows_symbol),
                name="Momentum",
            )
        )

    # ---- Trade markers (paper trades) ----
    if trades:
        bx, by, sx, sy = [], [], [], []
        for t in trades:
            ts = t.get("ts")
            if not ts:
                continue
            dt = datetime.fromtimestamp(int(ts))
            side = (t.get("side") or "").upper()
            price = t.get("price")
            if price is None:
                continue
            if side == "BUY":
                bx.append(dt); by.append(price)
            elif side == "SELL":
                sx.append(dt); sy.append(price)

        if bx:
            fig.add_trace(
                go.Scatter(
                    x=bx, y=by, mode="markers",
                    marker=dict(size=12, symbol="triangle-up"),
                    name="BUY",
                )
            )
        if sx:
            fig.add_trace(
                go.Scatter(
                    x=sx, y=sy, mode="markers",
                    marker=dict(size=12, symbol="triangle-down"),
                    name="SELL",
                )
            )

    fig.update_layout(
        height=380,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width="stretch")


state = safe_load_json(STATE_FILE)
trade = safe_load_json(TRADE_FILE) or {}

if not state:
    st.info("⏳ Waiting for engine data…")
    st.stop()

if state.get("error"):
    st.error(f"Engine error: {state['error']}")

# Top metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC Price", f"${state.get('price', 0):,.2f}")
c2.metric("RSI (1m)", state.get("rsi", 0))
c3.metric("Trend", state.get("trend", "WAIT"))
c4.metric("🧠 Confidence", f"{state.get('confidence', 0)}%")

st.caption(f"Last update: {state.get('time','--:--:--')}  •  {state.get('notes','')}")

conf = int(state.get("confidence", 0) or 0)
if conf >= 75:
    st.success("🔥 High-quality setup")
elif conf >= 60:
    st.warning("⚠️ Medium-quality signal")
else:
    st.info("⏳ Waiting for stronger conditions")

# Trading panel
st.subheader("💼 Trading (Paper)")
tc1, tc2, tc3, tc4 = st.columns(4)

enabled = trade.get("enabled", False)
mode = trade.get("mode", "paper")
btc_qty = float(trade.get("btc_qty", 0) or 0)
pos = "LONG" if btc_qty > 0 else "FLAT"

tc1.metric("Trading", "ON ✅" if enabled else "OFF ⛔")
tc2.metric("Mode", mode.upper())
tc3.metric("Position", pos)
tc4.metric("Equity", f"${float(trade.get('equity_usd', 0) or 0):,.2f}")

p1, p2, p3 = st.columns(3)
p1.metric("Realized P/L", f"${float(trade.get('realized_pnl', 0) or 0):,.2f}")
p2.metric("Unrealized P/L", f"${float(trade.get('unrealized_pnl', 0) or 0):,.2f}")
p3.metric("Updated", trade.get("updated_at", "--:--:--"))

st.subheader("📊 BTC 1-Minute Candlesticks (Last 30 min)")
render_chart(state.get("candles", []), trade.get("trades", []))

with st.expander("🧾 Recent Trades"):
    trades = trade.get("trades", [])
    if not trades:
        st.write("No trades yet.")
    else:
        st.dataframe(trades[::-1], use_container_width=True)

if st.button("🔄 Refresh Now"):
    st.rerun()
