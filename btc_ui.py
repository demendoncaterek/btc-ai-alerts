import os
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except:
    st_autorefresh = None

# ================= CONFIG =================
ENGINE_URL = os.getenv(
    "ENGINE_URL",
    "http://btc-engine.railway.internal:8080"
).rstrip("/")

REFRESH_MS = int(os.getenv("REFRESH_MS", "10000"))  # 10s
# =========================================

st.set_page_config(
    page_title="BTC AI Dashboard",
    layout="wide"
)

# ============== AUTO REFRESH ==============
if st_autorefresh:
    st_autorefresh(interval=REFRESH_MS, key="btc_refresh")

# ============== HELPERS ===================
def fetch_json(path: str):
    try:
        r = requests.get(f"{ENGINE_URL}{path}", timeout=8)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ============== ROUTING ===================
query = st.query_params
page = query.get("page", "overview")

# ============== NAV =======================
nav_cols = st.columns(2)
with nav_cols[0]:
    st.markdown("### 🧠 BTC AI Dashboard")
with nav_cols[1]:
    st.markdown(
        """
        <div style='text-align:right'>
        <a href='/?page=overview'>🏠 Overview</a> |
        <a href='/?page=trades'>📒 Trades</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# =========================================
# ============== OVERVIEW ==================
# =========================================
if page == "overview":

    state = fetch_json("/state")

    if not state.get("ok"):
        st.warning("⏳ Engine not ready yet.")
        st.code(state)
        st.stop()

    # ---------- SAFE GETTERS ----------
    price = state.get("price")
    signal = state.get("signal", "WAIT")
    confidence = state.get("confidence", 0)
    rsi_5m = state.get("rsi_5m")
    trend = state.get("htf_bias", "—")
    reason = state.get("reason", "")
    sl = state.get("sl_price")
    tp = state.get("tp_price")
    atr = state.get("atr_5m")
    div = state.get("divergence", {})
    watch = state.get("watch", {})
    conf_break = state.get("confidence_breakdown", {})

    # ---------- TOP METRICS ----------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BTC Price", f"${price:,.2f}")
    m2.metric("RSI (5m)", f"{rsi_5m:.1f}" if rsi_5m else "—")
    m3.metric("Signal", signal)
    m4.metric("Confidence", f"{confidence:.2f}%")

    st.info(reason)

    # ---------- RISK BOX ----------
    if sl and tp:
        st.success(
            f"""
            **Risk Setup (ATR-based)**  
            SL: **${sl:,.2f}**  
            TP: **${tp:,.2f}**  
            ATR (5m): {atr}
            """
        )

    # ---------- DIVERGENCE ----------
    if div.get("bullish"):
        st.success(f"🟢 **Bullish Divergence** — {div.get('detail')}")
    if div.get("bearish"):
        st.error(f"🔴 **Bearish Divergence** — {div.get('detail')}")

    if watch.get("peak_watch"):
        st.warning("📈 Peak Watch: momentum extended")
    if watch.get("dip_watch"):
        st.warning("📉 Dip Watch: selling pressure rising")

    # ---------- CONFIDENCE BREAKDOWN ----------
    with st.expander("🧠 Confidence Breakdown"):
        for k, v in conf_break.items():
            st.write(f"{k}: **{v}**")

    st.divider()

    # ---------- CANDLE CHART ----------
    st.subheader("📊 BTC Candlesticks")

    candles = fetch_json("/backtest?granularity=300&bars=120")

    if not candles.get("ok"):
        st.warning("Candle data incomplete — waiting for feed.")
    else:
        # Fetch raw candles again safely
        try:
            df = requests.get(
                f"{ENGINE_URL}/state",
                timeout=6
            ).json()
        except:
            df = None

        # We re-fetch candles directly to avoid KeyErrors
        try:
            c = requests.get(
                "https://api.exchange.coinbase.com/products/BTC-USD/candles",
                params={"granularity": 300},
                timeout=10,
            ).json()

            candles_df = pd.DataFrame(
                c,
                columns=["time", "low", "high", "open", "close", "volume"],
            )
            candles_df["time"] = pd.to_datetime(
                candles_df["time"], unit="s"
            )
            candles_df = candles_df.sort_values("time").tail(120)

            fig = go.Figure()

            fig.add_candlestick(
                x=candles_df["time"],
                open=candles_df["open"],
                high=candles_df["high"],
                low=candles_df["low"],
                close=candles_df["close"],
                name="BTC",
            )

            # Momentum arrows
            mom = candles_df["close"].diff()
            fig.add_scatter(
                x=candles_df["time"],
                y=candles_df["low"],
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=7,
                    color="green",
                ),
                name="Bullish momentum",
                visible=True,
            )
            fig.add_scatter(
                x=candles_df["time"],
                y=candles_df["high"],
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=7,
                    color="red",
                ),
                name="Bearish momentum",
                visible=True,
            )

            fig.update_layout(
                height=520,
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error("Chart error")
            st.code(str(e))

    st.caption(
        f"Last update: {state.get('time')} • Source: Coinbase • Trend: {trend}"
    )

# =========================================
# ============== TRADES ====================
# =========================================
elif page == "trades":

    st.header("📒 Trade Journal")

    state = fetch_json("/state")
    if not state.get("ok"):
        st.warning("Engine not ready.")
        st.stop()

    col1, col2 = st.columns(2)
    col1.metric("Paper Equity", "$0.00")
    col2.metric("Real Equity (Logged)", "$0.00")

    st.subheader("🧪 Paper Trades")
    st.info("No paper trades yet.")

    st.subheader("💰 Real Trades")
    st.info("No real trades logged yet.")

    st.caption("Trade logging will appear automatically once trades are recorded.")
