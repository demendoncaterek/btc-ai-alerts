import os
import io
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# ================= CONFIG =================
ENGINE_URL = os.getenv("ENGINE_URL", "http://localhost:8080").rstrip("/")
REFRESH_MS = int(os.getenv("REFRESH_MS", "10000"))  # 10s default
# =========================================


# ----------------- Helpers -----------------
def fetch_json(path: str, timeout: int = 8):
    try:
        r = requests.get(f"{ENGINE_URL}{path}", timeout=timeout)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _qp_get(key: str, default: str = "") -> str:
    try:
        qp = dict(st.query_params)
        v = qp.get(key, default)
        if isinstance(v, list):
            return v[0] if v else default
        return v
    except Exception:
        try:
            qp = st.experimental_get_query_params()
            v = qp.get(key, [default])
            return v[0] if isinstance(v, list) else v
        except Exception:
            return default

def _qp_set(**kwargs):
    try:
        st.query_params.update(kwargs)
    except Exception:
        st.experimental_set_query_params(**kwargs)

def normalize_candles(raw):
    """
    Accepts:
    - list of dicts {time, open, high, low, close} or {t,o,h,l,c}
    - list of lists Coinbase: [time, low, high, open, close, volume]
    Returns DataFrame(time, open, high, low, close)
    """
    if not raw or not isinstance(raw, list):
        return pd.DataFrame()

    first = raw[0]

    if isinstance(first, dict):
        df = pd.DataFrame(raw)
        # map short keys
        df = df.rename(columns={
            "t": "time",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
        })
    elif isinstance(first, (list, tuple)) and len(first) >= 5:
        df = pd.DataFrame(raw)
        # Coinbase: [time, low, high, open, close, volume]
        df = df.rename(columns={0: "time", 1: "low", 2: "high", 3: "open", 4: "close"})
    else:
        return pd.DataFrame()

    needed = {"time", "open", "high", "low", "close"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()

    # time conversion
    if pd.api.types.is_numeric_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
    else:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    df = df.dropna(subset=["time"]).sort_values("time")

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    return df

def add_momentum_arrows(fig, df: pd.DataFrame):
    """
    Recreates the triangle-up/triangle-down arrows based on short lookback momentum.
    """
    if df.empty or len(df) < 10:
        return fig

    lookback = 3
    thr = 0.0009  # ~0.09% over lookback
    mom = df["close"].pct_change(lookback)

    bull = mom > thr
    bear = mom < -thr

    if bull.any():
        fig.add_trace(go.Scatter(
            x=df.loc[bull, "time"],
            y=df.loc[bull, "low"] * 0.999,
            mode="markers",
            name="Bullish momentum",
            marker=dict(symbol="triangle-up", size=10, color="#22c55e"),
            hoverinfo="skip"
        ))
    if bear.any():
        fig.add_trace(go.Scatter(
            x=df.loc[bear, "time"],
            y=df.loc[bear, "high"] * 1.001,
            mode="markers",
            name="Bearish momentum",
            marker=dict(symbol="triangle-down", size=10, color="#ef4444"),
            hoverinfo="skip"
        ))

    return fig

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ----------------- Page setup -----------------
st.set_page_config(page_title="BTC AI Dashboard", layout="wide")

# Auto-refresh
if st_autorefresh:
    st_autorefresh(interval=REFRESH_MS, key="btc_autorefresh")
else:
    # fallback meta refresh
    sec = max(5, int(round(REFRESH_MS / 1000)))
    st.markdown(f"<meta http-equiv='refresh' content='{sec}'>", unsafe_allow_html=True)

page = (_qp_get("page", "overview") or "overview").lower().strip()
if page not in {"overview", "trades"}:
    page = "overview"

# ----------------- Header + Nav -----------------
top1, top2, top3 = st.columns([2, 2, 3])

with top1:
    st.title("🧠 BTC AI Dashboard")
with top2:
    st.caption("Disciplined • rules-based • logs + calibration-ready")
with top3:
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🏠 Overview", use_container_width=True):
            _qp_set(page="overview")
            st.rerun()
    with c2:
        if st.button("📒 Trades", use_container_width=True):
            _qp_set(page="trades")
            st.rerun()
    with c3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

st.divider()

# ----------------- Engine connectivity panel -----------------
health = fetch_json("/health", timeout=6)
if not health.get("ok"):
    st.warning("⏳ Can’t reach engine yet.")
    st.write("Make sure `ENGINE_URL` is correct in Railway Variables for the UI service.")
    with st.expander("Debug", expanded=True):
        st.write("ENGINE_URL:", ENGINE_URL)
        st.json(health)
    st.stop()

state = fetch_json("/state", timeout=8)
if not state.get("ok"):
    st.warning("⏳ Engine is reachable but not ready.")
    with st.expander("Debug", expanded=True):
        st.write("ENGINE_URL:", ENGINE_URL)
        st.json(state)
    st.stop()

# =========================
# OVERVIEW PAGE
# =========================
if page == "overview":
    price = state.get("price")
    signal = state.get("signal", "WAIT")
    confidence = state.get("confidence", 0.0)
    reason = state.get("reason", "")
    src = state.get("src", "—")
    ts = state.get("time", "—")

    rsi_5m = state.get("rsi_5m", None)
    rsi_1h = state.get("rsi_1h", None)
    htf_bias = state.get("htf_bias", "—")
    trend_1h = state.get("trend_1h", "—")
    trend_6h = state.get("trend_6h", "—")

    atr = state.get("atr_5m", None)
    sl = state.get("sl_price", None)
    tp = state.get("tp_price", None)

    div = state.get("divergence", {}) or {}
    watch = state.get("watch", {}) or {}
    conf_break = state.get("confidence_breakdown", {}) or {}
    params = state.get("params", {}) or {}
    paper = state.get("paper", {}) or {}
    paper_equity = (paper.get("equity") if isinstance(paper, dict) else None)

    # --- Top metrics ---
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("BTC Price", f"${price:,.2f}" if isinstance(price, (int, float)) else str(price))
    m2.metric("Signal", str(signal))
    m3.metric("Confidence", f"{confidence:.2f}%")
    m4.metric("RSI (5m)", f"{rsi_5m:.1f}" if isinstance(rsi_5m, (int, float)) else "—")
    m5.metric("RSI (1h)", f"{rsi_1h:.1f}" if isinstance(rsi_1h, (int, float)) else "—")

    st.caption(f"Last update: {ts} • Source: {src}")

    # --- Signal banner ---
    if str(signal).upper() == "BUY":
        st.success(f"✅ BUY setup (paper) • {reason}")
    elif str(signal).upper() == "SELL":
        st.error(f"⚠️ SELL setup (paper) • {reason}")
    else:
        st.info(f"⏳ WAIT • {reason}")

    # --- HTF / trend row ---
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("HTF Bias", str(htf_bias))
    t2.metric("Trend 1h", str(trend_1h))
    t3.metric("Trend 6h", str(trend_6h))
    t4.metric("Paper Equity", f"${paper_equity:,.2f}" if isinstance(paper_equity, (int, float)) else "—")

    # --- Risk (ATR SL/TP) ---
    if isinstance(sl, (int, float)) and isinstance(tp, (int, float)):
        st.markdown("### 🛡️ Risk (ATR-based suggestion)")
        r1, r2, r3 = st.columns(3)
        r1.metric("ATR (5m)", f"{atr:.2f}" if isinstance(atr, (int, float)) else "—")
        r2.metric("SL", f"${sl:,.2f}")
        r3.metric("TP", f"${tp:,.2f}")

    # --- Divergence + watch ---
    if div.get("bullish"):
        st.success(f"🟢 Bullish divergence — {div.get('detail','')}")
    if div.get("bearish"):
        st.warning(f"🔴 Bearish divergence — {div.get('detail','')}")

    if watch.get("peak_watch"):
        st.warning("📈 Peak Watch: momentum extended (heads-up, not an entry by itself).")
    if watch.get("dip_watch"):
        st.warning("📉 Dip Watch: selling pressure rising (heads-up, not an entry by itself).")

    # --- Calibration status panel ---
    st.divider()
    st.subheader("🧪 Calibration Status")

    cA, cB = st.columns([2, 3])
    with cA:
        st.write("**Current live params (engine):**")
        if params:
            st.json(params)
        else:
            st.info("No `params` block found in /state yet (engine may not be returning it).")

    with cB:
        # If you saved calibration metadata into params.json under "_calibration",
        # you can expose it later from the engine. For now, show what we have.
        min_conf = params.get("min_confidence", "—")
        st.metric("Min Confidence", f"{min_conf}" if isinstance(min_conf, (int, float)) else str(min_conf))
        st.caption(
            "Calibration workflow: logs → calibrate.py → params.json → engine auto-loads."
        )

    # --- Confidence breakdown ---
    with st.expander("🧠 Confidence breakdown (why confidence is what it is)", expanded=False):
        if conf_break:
            for k, v in conf_break.items():
                st.write(f"- **{k}**: {v}")
        else:
            st.info("No breakdown found yet.")

    # --- Candlestick chart + momentum arrows ---
    st.divider()
    st.subheader("📊 Candlesticks + Momentum Arrows")

    # Prefer engine-provided candles if you later add them.
    # For now, use recent_signals to build a lightweight “price line” fallback if candle feed isn’t available.
    # If you DO have candles in /state in the future, it will render them.
    raw_candles = state.get("candles", None)
    candles_df = normalize_candles(raw_candles) if raw_candles else pd.DataFrame()

    if candles_df.empty:
        # Fallback: show a lightweight line from recent signals (not true OHLC, but keeps it “live”)
        rs = fetch_json("/recent_signals?limit=180", timeout=10)
        rows = rs.get("rows", []) if rs.get("ok") else []
        if rows:
            sdf = pd.DataFrame(rows)
            if "ts_utc" in sdf.columns and "price" in sdf.columns:
                sdf["ts_utc"] = pd.to_datetime(sdf["ts_utc"], errors="coerce")
                sdf = sdf.dropna(subset=["ts_utc"]).sort_values("ts_utc")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sdf["ts_utc"],
                    y=pd.to_numeric(sdf["price"], errors="coerce"),
                    mode="lines",
                    name="BTC (from signals log)"
                ))
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No candle data yet (and signals log missing expected columns).")
        else:
            st.info("No candle data yet. (Optional upgrade: have engine provide `candles`.)")
    else:
        fig = go.Figure(data=[
            go.Candlestick(
                x=candles_df["time"],
                open=candles_df["open"],
                high=candles_df["high"],
                low=candles_df["low"],
                close=candles_df["close"],
                name="BTC"
            )
        ])
        fig = add_momentum_arrows(fig, candles_df)
        fig.update_layout(
            height=520,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Download logs section ---
    st.divider()
    st.subheader("⬇️ Download Logs")

    d1, d2, d3 = st.columns([2, 2, 3])

    with d1:
        rs = fetch_json("/recent_signals?limit=2000", timeout=15)
        if rs.get("ok") and rs.get("rows"):
            df = pd.DataFrame(rs["rows"])
            st.download_button(
                label="Download signals.csv",
                data=df_to_csv_bytes(df),
                file_name="signals.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(f"{len(df)} rows")
        else:
            st.button("Download signals.csv", disabled=True, use_container_width=True)
            st.caption("No rows yet")

    with d2:
        pt = fetch_json("/paper_trades?limit=2000", timeout=15)
        if pt.get("ok") and pt.get("rows"):
            df = pd.DataFrame(pt["rows"])
            st.download_button(
                label="Download paper_trades.csv",
                data=df_to_csv_bytes(df),
                file_name="paper_trades.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(f"{len(df)} rows")
        else:
            st.button("Download paper_trades.csv", disabled=True, use_container_width=True)
            st.caption("No trades yet")

    with d3:
        st.caption("Tip: Use these CSVs for calibration (`calibrate.py`) and performance review.")

# =========================
# TRADES PAGE
# =========================
else:
    st.header("📒 Trade Journal")

    # Paper equity + open position
    paper = state.get("paper", {}) or {}
    equity = paper.get("equity", None)
    pos = paper.get("open_position", None)
    recent = paper.get("recent_trades", []) if isinstance(paper.get("recent_trades", []), list) else []

    e1, e2, e3 = st.columns(3)
    e1.metric("Paper Equity", f"${equity:,.2f}" if isinstance(equity, (int, float)) else "—")
    e2.metric("Open Position", "YES" if pos else "NO")
    e3.metric("Last Update", str(state.get("time", "—")))

    if pos:
        st.subheader("🟦 Open Paper Position")
        st.json(pos)
    else:
        st.info("No open paper position right now.")

    st.divider()
    st.subheader("📈 Equity Curve (Paper)")
    pt = fetch_json("/paper_trades?limit=2000", timeout=15)
    if pt.get("ok") and pt.get("rows"):
        df = pd.DataFrame(pt["rows"])

        # Build equity curve from PnL
        # Start from paper_start_usd if present in params; else approximate from first equity in /state
        start_usd = None
        params = state.get("params", {}) or {}
        if isinstance(params.get("paper_start_usd"), (int, float)):
            start_usd = float(params["paper_start_usd"])
        elif isinstance(equity, (int, float)):
            # fallback: reverse engineer rough start
            start_usd = float(equity)

        df["ts_utc"] = pd.to_datetime(df.get("ts_utc"), errors="coerce")
        df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc")

        pnl_usd = pd.to_numeric(df.get("pnl_usd"), errors="coerce").fillna(0.0)
        if start_usd is None:
            start_usd = 1000.0

        df["equity_curve"] = start_usd + pnl_usd.cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["ts_utc"],
            y=df["equity_curve"],
            mode="lines",
            name="Equity"
        ))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧪 Paper Trades (latest)")
        st.dataframe(df.tail(200), use_container_width=True, hide_index=True)

        # downloads
        st.download_button(
            "Download paper_trades.csv",
            data=df_to_csv_bytes(df),
            file_name="paper_trades.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No paper trades yet — once TP/SL hits, they’ll show here.")

    st.divider()
    st.subheader("📌 Signals (latest)")
    rs = fetch_json("/recent_signals?limit=500", timeout=15)
    if rs.get("ok") and rs.get("rows"):
        df = pd.DataFrame(rs["rows"])
        st.dataframe(df.tail(200), use_container_width=True, hide_index=True)
        st.download_button(
            "Download signals.csv",
            data=df_to_csv_bytes(df),
            file_name="signals.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No signal logs yet.")
