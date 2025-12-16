import asyncio
import json
import os
import time
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

import numpy as np
import requests

# =========================
# CONFIG
# =========================
PRODUCT = "BTC-USD"
GRANULARITY = 60  # 1-minute candles
COINBASE_BASE = "https://api.exchange.coinbase.com"

STATE_FILE = "btc_state.json"          # optional debug file
PERSIST_FILE = "trade_state.json"      # stores paper + real trade logs

ENGINE_TICK_SECONDS = int(os.getenv("ENGINE_TICK_SECONDS", "5"))

ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", "300"))
MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "70"))

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = str(os.getenv("TELEGRAM_CHAT_ID", "")).strip()

# Heartbeat: 0 disables (ex: 60 = hourly)
HEARTBEAT_MINUTES = int(os.getenv("HEARTBEAT_MINUTES", "0"))

# Startup message toggle: 1 = on, 0 = off
STARTUP_MESSAGE = os.getenv("STARTUP_MESSAGE", "1").strip() == "1"

# Paper trading
PAPER_START_CASH = float(os.getenv("PAPER_START_CASH", "10000"))

# HTTP server
PORT = int(os.getenv("PORT", "8000"))  # Railway sets PORT automatically; this reads it if present


# =========================
# GLOBAL STATE (thread-safe)
# =========================
state_lock = threading.Lock()

STATE = {
    "price": 0.0,
    "rsi": 0.0,
    "trend": "WAIT",
    "state": "WAIT",
    "confidence": 0,
    "momentum": 0.0,
    "time": "--:--:--",
    "candles": [],
    "notes": "",
    "error": "",
    "paper": {},
    "real": {},
    "trades": {"paper": [], "real": []},
}

PERSIST = {
    "paper": {
        "start_cash": PAPER_START_CASH,
        "cash": PAPER_START_CASH,
        "btc": 0.0,
        "cost_basis": 0.0,          # total USD spent on current holdings
        "realized_pl": 0.0,
        "last_trade_ts": 0.0,
    },
    "real": {
        # purely tracked from your /log commands
        "cash": 0.0,
        "btc": 0.0,
        "cost_basis": 0.0,
        "realized_pl": 0.0,
    },
    "trades": {"paper": [], "real": []},
    "telegram": {"offset": 0},
    "engine": {"started_at": time.time()},
}

persist_lock = threading.Lock()


# =========================
# UTIL
# =========================
def now_str():
    return datetime.now().strftime("%H:%M:%S")

def safe_load_json(path, default):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return default
        return json.loads(raw)
    except:
        return default

def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)

def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=8,
        )
    except:
        pass

# Fix typo-safe: if you ever accidentally renamed the var
TELELEGRAM_CHAT_ID = TELEGRAM_CHAT_ID


# =========================
# MARKET DATA + SIGNALS
# =========================
def fetch_candles(limit=60):
    url = f"{COINBASE_BASE}/products/{PRODUCT}/candles"
    resp = requests.get(
        url,
        params={"granularity": GRANULARITY},
        headers={"Accept": "application/json", "User-Agent": "btc-engine"},
        timeout=10,
    )
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Coinbase API error: {data}")

    # Coinbase returns newest-first; sort oldest->newest
    data.sort(key=lambda x: x[0])
    data = data[-limit:]

    candles = []
    closes = []
    for c in data:
        ts = int(c[0])
        low = float(c[1])
        high = float(c[2])
        open_ = float(c[3])
        close = float(c[4])

        candles.append({
            "time": datetime.fromtimestamp(ts).strftime("%H:%M"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        })
        closes.append(close)

    return candles, closes

def compute_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0)
    losses = -np.minimum(deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def confidence_score(rsi, trend_strength, momentum):
    score = 0

    if rsi < 30 or rsi > 70:
        score += 35
    elif 35 <= rsi <= 65:
        score -= 10

    score += min(abs(momentum) * 2000, 25)
    score += min(trend_strength * 2000, 25)

    return max(0, min(100, int(score)))


# =========================
# PAPER + REAL P/L
# =========================
def compute_portfolio_snapshot(port, price):
    cash = float(port.get("cash", 0.0))
    btc = float(port.get("btc", 0.0))
    equity = cash + btc * price
    cost_basis = float(port.get("cost_basis", 0.0))
    realized = float(port.get("realized_pl", 0.0))

    unrealized = 0.0
    if btc > 0 and cost_basis > 0:
        avg_cost = cost_basis / btc
        unrealized = (price - avg_cost) * btc

    total_pl = realized + unrealized
    return {
        "cash": round(cash, 2),
        "btc": round(btc, 8),
        "equity": round(equity, 2),
        "cost_basis": round(cost_basis, 2),
        "realized_pl": round(realized, 2),
        "unrealized_pl": round(unrealized, 2),
        "total_pl": round(total_pl, 2),
    }

def paper_trade_logic(signal_state, confidence, price):
    """Simple paper strategy: BUY = all-in, SELL = all-out."""
    with persist_lock:
        p = PERSIST["paper"]
        now = time.time()

        if now - float(p.get("last_trade_ts", 0.0)) < ALERT_COOLDOWN:
            return

        if signal_state == "BUY" and confidence >= MIN_CONFIDENCE and p["cash"] > 1:
            usd = p["cash"]
            btc = usd / price
            p["cash"] -= usd
            p["btc"] += btc
            p["cost_basis"] += usd
            p["last_trade_ts"] = now

            PERSIST["trades"]["paper"].append({
                "time": now_str(),
                "side": "BUY",
                "usd": round(usd, 2),
                "btc": round(btc, 8),
                "price": round(price, 2),
            })

        elif signal_state == "SELL" and confidence >= MIN_CONFIDENCE and p["btc"] > 0:
            btc = p["btc"]
            proceeds = btc * price

            avg_cost = (p["cost_basis"] / btc) if btc > 0 else price
            realized = (price - avg_cost) * btc

            p["cash"] += proceeds
            p["btc"] = 0.0
            p["cost_basis"] = 0.0
            p["realized_pl"] += realized
            p["last_trade_ts"] = now

            PERSIST["trades"]["paper"].append({
                "time": now_str(),
                "side": "SELL",
                "usd": round(proceeds, 2),
                "btc": round(btc, 8),
                "price": round(price, 2),
                "realized_pl": round(realized, 2),
            })

def real_log_buy(usd, price):
    with persist_lock:
        r = PERSIST["real"]
        usd = float(usd)
        btc = usd / price

        r["cash"] -= usd
        r["btc"] += btc
        r["cost_basis"] += usd

        PERSIST["trades"]["real"].append({
            "time": now_str(),
            "side": "BUY",
            "usd": round(usd, 2),
            "btc": round(btc, 8),
            "price": round(price, 2),
        })

def real_log_sell(usd, price):
    with persist_lock:
        r = PERSIST["real"]
        usd = float(usd)

        if r["btc"] <= 0:
            return False, "You have 0 BTC logged in real trades."

        btc_to_sell = usd / price
        if btc_to_sell > r["btc"]:
            btc_to_sell = r["btc"]
            usd = btc_to_sell * price

        avg_cost = (r["cost_basis"] / r["btc"]) if r["btc"] > 0 and r["cost_basis"] > 0 else price
        realized = (price - avg_cost) * btc_to_sell

        r["btc"] -= btc_to_sell
        r["cash"] += usd
        r["cost_basis"] -= avg_cost * btc_to_sell
        r["realized_pl"] += realized

        if r["btc"] < 1e-12:
            r["btc"] = 0.0
            r["cost_basis"] = 0.0

        PERSIST["trades"]["real"].append({
            "time": now_str(),
            "side": "SELL",
            "usd": round(usd, 2
