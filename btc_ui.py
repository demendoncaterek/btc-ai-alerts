import os
import time
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

# ---------- query params helpers ----------
def _get_qp():
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()

def _set_qp(**kw):
    try:
        for k, v in kw.items():
            st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**kw)

def _rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

qp = _get_qp()
page = qp.get("page", "overview")
if isinstance(page, list):
    page = page[0] if page else "overview"
page = str(page).lower().strip() or "overview"

# ---------- HTTP helpers ----------
SESSION = requests.Session()

def get_json(path: str, timeout=(2, 5)):
    """
    timeout=(connect, read) so we don't hang forever
    """
    url = f"{ENGINE_URL}{path}"
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def parse_candles(raw):
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        df = pd.DataFrame(raw)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        need = {"time", "open", "high", "low", "close"}
        if not need.issubset(df.columns):
            return pd.DataFrame()
        return df.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
    return pd.DataFrame()

def fmt_money(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "—"

def fmt_num(x, d=2):
    try:
        return f"{float(x):.{d}f}"
    except Exception:
        return "—"

def fmt_pct(x, d=2):
    try:
        return f"{float(x) * 100:.{d}f}%"
    except Exception:
        return "—"

# ---------- nav buttons ----------
l, r = st.columns([1, 1])
with l:
    if st.button("🏠 Overview", use_container_width=True, disabled=(page == "overview")):
        _set_qp(page="overview"); _rerun()
with r:
    if st.button("📒 Trades", use_container_width=True, disabled=(page == "trades")):
        _set_qp(page="trades"); _rerun()

# ---------- always-visible debug header ----------
with st.expander("Connection Debug", expanded=True):
    st.write("ENGINE_URL:", ENGINE_URL)

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Ping /health", use_container_width=True):
            try:
                t0 = time.time()
                h = get_json("/health", timeout=(2, 4))
                dt = (time.time() - t0) * 1000
                st.success(f"/health OK in {dt:.0f} ms")
                st.json(h)
            except Exception as e:
                st.error(f"/health failed: {e}")

    with colB:
        if st.button("Ping /state", use_container_width=True):
            try:
                t0 = time.time()
                s = get_json("/state", timeout=(2, 5))
                dt = (time.time() - t0) * 1000
                st.success(f"/state OK in {dt:.0f} ms")
                st.json({k: s.get(k) for k in ["ok","ts","price","signal","confidence","reason"]})
            except Exception as e:
                st.error(f"/state failed: {e}")

# ---------- try connecting BEFORE autorefresh ----------
status_box = st.empty()
status_box.info("Connecting to engine…")

try:
    health = get_json("/health", timeout=(2, 4))
except Exception as e:
    status_box.error(
        "Engine is NOT reachable from the UI right now.\n\n"
        f"Error: {e}\n\n"
        "Fix:\n"
        "1) In UI Railway Variables, set ENGINE_URL to: http://btc-engine.railway.internal:8080\n"
        "2) Make sure your ENGINE service is running and listening on port 8080.\n"
        "3) If your service name isn’t literally btc-engine, replace it in the URL.\n"
    )
    st.stop()

status_box.success("Engine reachable ✅")

# only start autorefresh AFTER engine is reachable
if st_autorefresh:
    st_autorefresh(interval=REFRESH_MS, key="auto")
else:
    st.warning("Auto-refresh disabled: install streamlit-autorefresh if you want 10s refresh.")

# ---------- pages ----------
if page == "trades":
    st.subheader("📒 Trade Journal")
    try:
        state = get_json("/state", timeout=(2, 6))
        t = get_json("/trades", timeout=(2, 8))
    except Exception as e:
        st.error(str(e))
        st.stop()

    paper = t.get("paper", {}) if isinstance(t.get("paper"), dict) else {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Paper USD", fmt_money(paper.get("usd", 0.0)))
    c2.metric("Paper BTC", fmt_num(paper.get("btc", 0.0), 8))
    c3.metric("Paper Equity", fmt_money(paper.get("equity", 0.0)))

    st.markdown("### 🧪 Paper Trades")
    pt = t.get("paper_trades", [])
    st.dataframe(pd.DataFrame(pt) if pt else pd.DataFrame(), use_container_width=True, hide_index=True)

    st.markdown("### 💰 Real Trades (Logged)")
    rt = t.get("real_trades", [])
    st.dataframe(pd.DataFrame(rt) if rt else pd.DataFrame(), use_container_width=True, hide_index=True)

    st.caption(f"Last update: {state.get('ts','—')} • Auto-refresh: {REFRESH_MS/1000:.0f}s")
    if st.button("🔄 Refresh now"): _rerun()
    st.stop()

# overview
try:
    s = get_json("/state", timeout=(2, 6))
except Exception as e:
    st.error(f"Failed to load /state: {e}")
    st.stop()

if not s.get("ok"):
    st.warning("⏳ Engine reachable, but not ready yet.")
    st.write("Reason:", s.get("reason", "—"))
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
    st.info("👀 Watch: possible swing point (peak/dip watch).")
else:
    st.info("⏳ Waiting for a high-probability setup")

with st.expander("Why?", expanded=False):
    try:
        st.json(get_json("/explain", timeout=(2, 6)))
    except Exception as e:
        st.error(str(e))

st.subheader("📊 BTC Candlesticks (15m)")
df = parse_candles(s.get("candles_15m", []))
if df.empty or len(df) < 20:
    st.warning("Candle data incomplete — waiting for full feed.")
else:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="BTC"
    ))
    fig.update_layout(height=540, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

paper = s.get("paper", {}) if isinstance(s.get("paper"), dict) else {}
p1, p2 = st.columns(2)
p1.metric("Paper Equity", fmt_money(paper.get("equity", 0.0)))
p2.metric("Paper P/L", fmt_money(paper.get("pl", 0.0)))

st.caption(f"Last update: {s.get('ts','—')} • Source: {s.get('src','—')} • Auto-refresh: {REFRESH_MS/1000:.0f}s")
if st.button("🔄 Refresh now"): _rerun()
