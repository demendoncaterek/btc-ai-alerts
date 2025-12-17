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
REFRESH_MS = int(os.getenv("REFRESH_MS", "15000"))  # 15s
# ==========================================

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")

st.title("🧠 BTC AI Dashboard")
st.caption("5m + 1h bias • Telegram alerts • Paper + Real (logged) P/L")

if st_autorefresh:
    st_autorefresh(interval=REFRESH_MS, key="btc_autorefresh")

# ============== DATA FETCH ==============
def fetch_state():
    try:
        r = requests.get(f"{ENGINE_URL}/state", timeout=6)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

state = fetch_state()

if not state.get("ok"):
    st.warning("⏳ Waiting for engine data…")
    st.write("Make sure ENGINE_URL points to your engine and /health works.")
    with st.expander("Debug (click to open)", expanded=True):
        st.write("ENGINE_URL:", ENGINE_URL)
        st.error(state.get("error", "No state yet"))
        if st.button("Ping engine /health"):
            try:
                r = requests.get(f"{ENGINE_URL}/health", timeout=6)
                st.json(r.json())
            except Exception as e:
                st.error(str(e))
    st.stop()
