import os
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


# ================= CONFIG =================
ENGINE_URL = os.getenv("ENGINE_URL", "http://localhost:8080").rstrip("/")
REFRESH_MS = int(os.getenv("REFRESH_MS", "10000"))  # default 10s
# ==========================================

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")


# ---------- Query param helpers (version-safe) ----------
def _get_query_params() -> dict:
    try:
        # Newer Streamlit
        qp = dict(st.query_params)
        # normalize values
        out = {}
        for k, v in qp.items():
            if isinstance(v, list):
                out[k] = v[0] if v else ""
            else:
                out[k] = v
        return out
    except Exception:
        # Older Streamlit
        qp = st.experimental_get_query_params()
        return {k: (v[0] if isinstance(v, list) and v else v) for k, v in qp.items()}


def _set_query_params(**kwargs):
    try:
        st.query_params.update(kwargs)
    except Exception:
        st.experimental_set_query_params(**kwargs)


# ---------- Auto refresh (works even if plugin missing) ----------
def setup_autorefresh():
    if st_autorefresh:
        st_autorefresh(interval=REFRESH_MS, key="btc_autorefresh")
    else:
        # fallback: HTML meta refresh (seconds)
        sec = max(5, int(round(REFRESH_MS / 1000)))
        st.markdown(
            f"<meta http-equiv='refresh' content='{sec}'>",
            unsafe_allow_html=True
        )


# ---------- Engine fetch ----------
def fetch_json(path: str, timeout: int = 6) -> dict:
    try:
        r = requests.get(f"{ENGINE_URL}{path}", timeout=timeout)
        # Try JSON even on non-200 (useful for debug)
        try:
            return r.json()
        except Exception:
            return {"ok": False, "error": f"Non-JSON response {r.status_code}: {r.text[:200]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def fetch_state() -> dict:
    s = fetch_json("/state", timeout=6)
    # Some engines return raw state without ok flag
    if isinstance(s, dict) and "ok" not in s:
        s["ok"] = True
    return s


# ---------- Candle normalization (prevents KeyError) ----------
def normalize_candles(raw):
    """
    Accepts:
    - list[dict] with time/open/high/low/close
    - list[dict] with t/o/h/l/c
    - Coinbase list[list]: [time, low, high, open, close, volume]
    Returns a DataFrame with columns: time, open, high, low, close
    """
    if not raw or not isinstance(raw, list):
        return pd.DataFrame()

    first = raw[0]

    # Case 1: list of dicts
    if isinstance(first, dict):
        df = pd.DataFrame(raw)

        # Map short keys if needed
        keymap = {}
        if "time" not in df.columns and "t" in df.columns:
            keymap["t"] = "time"
        if "open" not in df.columns and "o" in df.columns:
            keymap["o"] = "open"
        if "high" not in df.columns and "h" in df.columns:
            keymap["h"] = "high"
        if "low" not in df.columns and "l" in df.columns:
            keymap["l"] = "low"
        if "close" not in df.columns and "c" in df.columns:
            keymap["c"] = "close"

        if keymap:
            df = df.rename(columns=keymap)

    # Case 2: list of lists (Coinbase style)
    elif isinstance(first, (list, tuple)) and len(first) >= 5:
        # Coinbase candles are [time, low, high, open, close, volume]
        df = pd.DataFrame(raw)
        df = df.rename(columns={0: "time", 1: "low", 2: "high", 3: "open", 4: "close"})

    else:
        return pd.DataFrame()

    needed = {"time", "open", "high", "low", "close"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()

    # Convert time
    # If numeric -> assume seconds
    if pd.api.types.is_numeric_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
    else:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    df = df.dropna(subset=["time"]).sort_values("time")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def add_momentum_markers(fig, df: pd.DataFrame):
    """
    Re-creates the green/red triangle “momentum arrows” look from before.
    Simple rule: lookback pct-change crosses threshold.
    """
    if df.empty or len(df) < 8:
        return fig

    lookback = 3
    thr = 0.0009  # ~0.09% over lookback (tune if you want more/less arrows)

    mom = df["close"].pct_change(lookback)
    bull = mom > thr
    bear = mom < -thr

    if bull.any():
        fig.add_trace(
            go.Scatter(
                x=df.loc[bull, "time"],
                y=df.loc[bull, "low"] * 0.999,
                mode="markers",
                name="Bullish momentum",
                marker=dict(symbol="triangle-up", size=10, color="#22c55e"),
                hoverinfo="skip",
            )
        )
    if bear.any():
        fig.add_trace(
            go.Scatter(
                x=df.loc[bear, "time"],
                y=df.loc[bear, "high"] * 1.001,
                mode="markers",
                name="Bearish momentum",
                marker=dict(symbol="triangle-down", size=10, color="#ef4444"),
                hoverinfo="skip",
            )
        )

    return fig


# ---------- UI header + nav ----------
setup_autorefresh()

qp = _get_query_params()
page = (qp.get("page") or "overview").lower().strip()
if page not in {"overview", "trades"}:
    page = "overview"

st.title("🧠 BTC AI Dashboard")
st.caption("Disciplined, rules-based trading assistant • alerts + journal")

nav1, nav2, nav3 = st.columns([1, 1, 3])
with nav1:
    if st.button("🏠 Overview", use_container_width=True):
        _set_query_params(page="overview")
        st.rerun()
with nav2:
    if st.button("📒 Trades", use_container_width=True):
        _set_query_params(page="trades")
        st.rerun()
with nav3:
    if st.button("🔄 Refresh now", use_container_width=False):
        st.rerun()


# ---------- Load state ----------
state = fetch_state()

if not state.get("ok"):
    st.warning("⏳ Engine not ready yet.")
    st.write("Check ENGINE_URL and that engine /health is reachable.")
    with st.expander("Debug", expanded=True):
        st.write("ENGINE_URL:", ENGINE_URL)
        st.json(state)
        if st.button("Ping /health"):
            st.json(fetch_json("/health", timeout=6))
    st.stop()


# ---------- Pull fields safely ----------
def g(key, default=None):
    return state.get(key, default)


price = g("price")
signal = g("signal", g("action", "WAIT"))
confidence = g("confidence", 0.0)
trend = g("trend", g("bias", "—"))
rsi_5m = g("rsi_5m", g("rsi", None))
last_update = g("time", g("iso", g("updated", "")))
src = g("src", g("source", "Coinbase"))

# Top metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("BTC Price", f"${price:,.2f}" if isinstance(price, (int, float)) else str(price))
m2.metric("RSI", f"{rsi_5m:.1f}" if isinstance(rsi_5m, (int, float)) else "--")
m3.metric("Signal", str(signal))
m4.metric("Confidence", f"{float(confidence)*100:.2f}%" if isinstance(confidence, (int, float)) else str(confidence))

if isinstance(signal, str) and signal.upper() == "WAIT":
    st.info("⏳ Waiting for high-probability setup")
elif isinstance(signal, str) and signal.upper() in {"BUY", "SELL"}:
    st.warning("⚠️ High-confidence alert (confirm with your own rules + risk management)")

# ---------- Candles + chart ----------
raw_candles = g("candles", g("ohlc", g("candles_15m", g("candles_5m", []))))
candles = normalize_candles(raw_candles)

st.subheader("📊 BTC Candlesticks")
if candles.empty:
    st.warning("Candle data incomplete — waiting for full feed.")
    with st.expander("Candle debug"):
        st.write("State keys:", list(state.keys())[:60])
        st.write("Raw candles sample:", raw_candles[:2] if isinstance(raw_candles, list) else raw_candles)
else:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=candles["time"],
                open=candles["open"],
                high=candles["high"],
                low=candles["low"],
                close=candles["close"],
                name="BTC",
            )
        ]
    )
    fig = add_momentum_markers(fig, candles)
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption(f"Last update: {last_update} • Source: {src} • Trend: {trend}")


# ---------- Trades / Journal ----------
def _as_list(x):
    return x if isinstance(x, list) else []

paper_trades = _as_list(g("paper_trades", g("paper", [])))
real_trades = _as_list(g("real_trades", g("logged_real", [])))

paper_equity = float(g("paper_equity", g("paper_usd", 0.0)) or 0.0)
real_equity = float(g("real_equity", 0.0) or 0.0)

if page == "overview":
    st.subheader("📒 Trade Journal (Summary)")
else:
    st.title("📒 Trade Journal")

eq1, eq2 = st.columns(2)
eq1.metric("Paper Equity", f"${paper_equity:,.2f}")
eq2.metric("Real Equity (Logged)", f"${real_equity:,.2f}")

st.markdown("### 🧪 Paper Trades")
if paper_trades:
    st.dataframe(pd.DataFrame(paper_trades), use_container_width=True, hide_index=True)
else:
    st.info("No paper trades yet.")

st.markdown("### 💰 Real Trades")
if real_trades:
    st.dataframe(pd.DataFrame(real_trades), use_container_width=True, hide_index=True)
else:
    st.info("No real trades logged yet.")
