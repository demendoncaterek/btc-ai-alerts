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
REFRESH_MS = int(os.getenv("REFRESH_MS", "10000"))  # 10s

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")
st.title("🧠 BTC AI Dashboard")
st.caption("15m execution • 1h+4h bias • RSI(5m) • ATR SL/TP • Peak/Dip watch • Paper/Real logs")

def _get_qp():
    try: return dict(st.query_params)
    except Exception: return st.experimental_get_query_params()

def _set_qp(**kw):
    try:
        for k, v in kw.items(): st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**kw)

def _rerun():
    try: st.rerun()
    except Exception: st.experimental_rerun()

qp = _get_qp()
page = (qp.get("page", ["overview"])[0] if isinstance(qp.get("page", ["overview"]), list) else qp.get("page", "overview"))
page = str(page).lower().strip() or "overview"

if st_autorefresh:
    st_autorefresh(interval=REFRESH_MS, key="auto")
else:
    st.info("Install `streamlit-autorefresh` for auto-refresh (every 10s).")

def get_json(path: str, timeout: int = 8):
    r = requests.get(f"{ENGINE_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()

def parse_candles(raw):
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        df = pd.DataFrame(raw)
        if "time" in df.columns: df["time"] = pd.to_datetime(df["time"], errors="coerce")
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        need = {"time", "open", "high", "low", "close"}
        if not need.issubset(df.columns): return pd.DataFrame()
        return df.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
    return pd.DataFrame()

def fmt_money(x):
    try: return f"${float(x):,.2f}"
    except Exception: return "—"

def fmt_num(x, d=2):
    try: return f"{float(x):.{d}f}"
    except Exception: return "—"

def fmt_pct(x, d=2):
    try: return f"{float(x)*100:.{d}f}%"
    except Exception: return "—"

# nav
l, r = st.columns([1, 1])
with l:
    if st.button("🏠 Overview", use_container_width=True, disabled=(page == "overview")):
        _set_qp(page="overview"); _rerun()
with r:
    if st.button("📒 Trades", use_container_width=True, disabled=(page == "trades")):
        _set_qp(page="trades"); _rerun()

def add_momentum_marks(df):
    if df.empty or len(df) < 6: return pd.DataFrame(), pd.DataFrame()
    roc = df["close"].pct_change(3)
    up = (roc.shift(1) <= 0) & (roc > 0)
    dn = (roc.shift(1) >= 0) & (roc < 0)
    bull = df.loc[up.fillna(False), ["time", "low"]].copy()
    bear = df.loc[dn.fillna(False), ["time", "high"]].copy()
    return bull, bear

if page == "trades":
    st.subheader("📒 Trade Journal")
    try:
        state = get_json("/state", 8)
        t = get_json("/trades", 10)
    except Exception as e:
        st.error(str(e)); st.stop()

    paper = t.get("paper", {}) if isinstance(t.get("paper"), dict) else {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Paper USD", fmt_money(paper.get("usd", 0.0)))
    c2.metric("Paper BTC", fmt_num(paper.get("btc", 0.0), 8))
    c3.metric("Paper Equity", fmt_money(paper.get("equity", 0.0)))

    st.markdown("### 🧪 Paper Trades")
    pt = t.get("paper_trades", [])
    if pt: st.dataframe(pd.DataFrame(pt), use_container_width=True, hide_index=True)
    else: st.info("No paper trades yet.")

    st.markdown("### 💰 Real Trades (Logged)")
    rt = t.get("real_trades", [])
    if rt: st.dataframe(pd.DataFrame(rt), use_container_width=True, hide_index=True)
    else: st.info("No real trades logged yet.")

    st.caption(f"Last update: {state.get('ts','—')} • Auto-refresh: {REFRESH_MS/1000:.0f}s")
    if st.button("🔄 Refresh now"): _rerun()
    st.stop()

# overview
try:
    s = get_json("/state", 8)
except Exception as e:
    st.error(str(e))
    with st.expander("Debug"):
        st.write("ENGINE_URL:", ENGINE_URL)
        try: st.json(get_json("/health", 6))
        except Exception as ee: st.error(str(ee))
    st.stop()

if not s.get("ok"):
    st.warning("⏳ Engine is starting / waiting for data…")
    st.write(s.get("reason", "—"))
    st.stop()

price = s.get("price")
signal = s.get("signal", "WAIT")
conf = s.get("confidence", 0.0)
bias = s.get("trend_bias", "—")
rsi = s.get("rsi_5m")
mom = s.get("momentum")
peak = s.get("peak_watch", False)
dip = s.get("dip_watch", False)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("BTC Price", fmt_money(price))
c2.metric("RSI (5m)", "—" if rsi is None else fmt_num(rsi, 1))
c3.metric("Signal", str(signal))
c4.metric("Confidence", f"{fmt_num(conf, 0)}%")
c5.metric("Bias (1h+4h)", str(bias))
c6.metric("Momentum (ROC)", "—" if mom is None else fmt_pct(mom, 2))

if signal in ("BUY", "SELL"):
    st.success(f"✅ {signal} setup detected")
elif peak or dip:
    st.info("👀 Watch: possible swing point (peak/dip watch). If the signal is WAIT, treat this as a heads-up — not an entry.")
else:
    st.info("⏳ Waiting for a high-probability setup")

with st.expander("Why?", expanded=False):
    try: st.json(get_json("/explain", 8))
    except Exception as e: st.error(str(e))

st.subheader("📊 BTC Candlesticks (15m)")
df = parse_candles(s.get("candles_15m", []))
if df.empty or len(df) < 20:
    st.warning("Candle data incomplete — waiting for full feed.")
else:
    bull, bear = add_momentum_marks(df)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="BTC"
    ))
    if not bull.empty:
        fig.add_trace(go.Scatter(
            x=bull["time"], y=bull["low"], mode="markers",
            marker=dict(symbol="arrow-up", size=10), name="Bullish momentum"
        ))
    if not bear.empty:
        fig.add_trace(go.Scatter(
            x=bear["time"], y=bear["high"], mode="markers",
            marker=dict(symbol="arrow-down", size=10), name="Bearish momentum"
        ))
    fig.update_layout(height=540, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

paper = s.get("paper", {}) if isinstance(s.get("paper"), dict) else {}
p1, p2 = st.columns(2)
p1.metric("Paper Equity", fmt_money(paper.get("equity", 0.0)))
p2.metric("Paper P/L", fmt_money(paper.get("pl", 0.0)))

st.caption(f"Last update: {s.get('ts','—')} • Source: {s.get('src','—')} • Auto-refresh: {REFRESH_MS/1000:.0f}s")
if st.button("🔄 Refresh now"): _rerun()
