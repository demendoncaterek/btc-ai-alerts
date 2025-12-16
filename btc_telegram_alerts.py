import asyncio
import json
import os
import time
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import numpy as np
import requests

# =========================
# CONFIG
# =========================
PRODUCT = os.getenv("PRODUCT", "BTC-USD")
GRANULARITY = int(os.getenv("GRANULARITY", "60"))  # seconds (1m)
COINBASE_BASE = os.getenv("COINBASE_BASE", "https://api.exchange.coinbase.com")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", "300"))
MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "70"))

# “Bot is alive” heartbeat
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "3600"))

# Manual trade logging
MANUAL_FILE = os.getenv("MANUAL_TRADES_FILE", "manual_trades.json")

# Paper trading
PAPER_STATE_FILE = os.getenv("PAPER_STATE_FILE", "paper_state.json")
PAPER_START_CASH = float(os.getenv("PAPER_START_CASH", "1000"))

PORT = int(os.getenv("PORT", "8080"))

# =========================
# STATE (shared)
# =========================
state_lock = threading.Lock()
latest_state = {
    "price": 0.0,
    "rsi": 0.0,
    "trend": "WAIT",
    "state": "WAIT",
    "confidence": 0,
    "time": "--:--:--",
    "candles": [],
    "notes": "",
    "error": "",
    "paper": {},
    "manual": {},
}

paper_state = {
    "cash_usd": PAPER_START_CASH,
    "qty_btc": 0.0,
    "entry_price": None,
    "realized_pl_usd": 0.0,
    "last_action": "WAIT",
    "last_action_time": None,
}

manual_state = {
    "qty_btc": 0.0,
    "cost_basis_usd": 0.0,
    "realized_pl_usd": 0.0,
    "trades": [],
}

# =========================
# HELPERS
# =========================
def atomic_write_json(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)

def safe_load_json(path: str, default_obj: dict):
    if not os.path.exists(path):
        return default_obj
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_obj

def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=8,
        )
    except Exception:
        pass

def fetch_candles(limit=60):
    url = f"{COINBASE_BASE}/products/{PRODUCT}/candles"
    resp = requests.get(
        url,
        params={"granularity": GRANULARITY},
        headers={"Accept": "application/json", "User-Agent": "btc-alerts"},
        timeout=10,
    )
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Coinbase API error: {data}")

    data.sort(key=lambda x: x[0])
    data = data[-limit:]

    candles = []
    closes = []
    for c in data:
        ts = int(c[0])
        # Coinbase candle order: [ time, low, high, open, close, volume ]
        candles.append({
            "time": datetime.fromtimestamp(ts).strftime("%H:%M"),
            "open": float(c[3]),
            "high": float(c[2]),
            "low": float(c[1]),
            "close": float(c[4]),
        })
        closes.append(float(c[4]))

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
# PAPER TRADER
# =========================
def paper_step(signal: str, price: float, now_str: str):
    # Simple “all in / all out” for demo paper trading
    global paper_state
    if signal == "BUY" and paper_state["cash_usd"] > 0:
        qty = paper_state["cash_usd"] / price
        paper_state["qty_btc"] += qty
        paper_state["entry_price"] = price
        paper_state["cash_usd"] = 0.0
        paper_state["last_action"] = "BUY"
        paper_state["last_action_time"] = now_str
    elif signal == "SELL" and paper_state["qty_btc"] > 0:
        proceeds = paper_state["qty_btc"] * price
        entry = paper_state["entry_price"] or price
        pnl = proceeds - (paper_state["qty_btc"] * entry)
        paper_state["realized_pl_usd"] += pnl
        paper_state["cash_usd"] = proceeds
        paper_state["qty_btc"] = 0.0
        paper_state["entry_price"] = None
        paper_state["last_action"] = "SELL"
        paper_state["last_action_time"] = now_str
    atomic_write_json(PAPER_STATE_FILE, paper_state)

# =========================
# MANUAL (LOGGED) TRADES
# =========================
def build_status_text():
    with state_lock:
        s = dict(latest_state)

    price = s.get("price", 0)
    rsi = s.get("rsi", 0)
    trend = s.get("trend", "WAIT")
    conf = s.get("confidence", 0)
    t = s.get("time", "--:--:--")

    # paper
    p_cash = paper_state["cash_usd"]
    p_qty = paper_state["qty_btc"]
    p_equity = p_cash + (p_qty * price)
    p_unreal = 0.0
    if paper_state.get("entry_price") and p_qty > 0:
        p_unreal = (price - float(paper_state["entry_price"])) * p_qty

    # manual
    m_qty = manual_state["qty_btc"]
    m_cost = manual_state["cost_basis_usd"]
    m_unreal = (m_qty * price) - m_cost
    m_total = manual_state["realized_pl_usd"] + m_unreal

    return (
        f"📊 BTC STATUS\n"
        f"Price: ${price:,.2f}\n"
        f"RSI(1m): {rsi}\n"
        f"Trend: {trend}\n"
        f"Confidence: {conf}%\n"
        f"Updated: {t}\n\n"
        f"🧾 PAPER\n"
        f"Equity: ${p_equity:,.2f}\n"
        f"Realized P/L: ${paper_state['realized_pl_usd']:,.2f}\n"
        f"Unrealized P/L: ${p_unreal:,.2f}\n\n"
        f"🧍 REAL (logged)\n"
        f"BTC: {m_qty:.8f}\n"
        f"Realized P/L: ${manual_state['realized_pl_usd']:,.2f}\n"
        f"Total P/L: ${m_total:,.2f}\n\n"
        f"Commands: /status, /logbuy 100, /logsell 100"
    )

def handle_manual_trade(side: str, usd_amount: float):
    with state_lock:
        price = float(latest_state.get("price", 0) or 0)

    if price <= 0:
        return "Engine has no price yet. Try again in a few seconds."
    if usd_amount <= 0:
        return "Amount must be > 0. Example: /logbuy 100"

    qty = usd_amount / price

    if side == "BUY":
        manual_state["qty_btc"] += qty
        manual_state["cost_basis_usd"] += usd_amount
        manual_state["trades"].append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "side": "BUY",
            "usd": round(usd_amount, 2),
            "price": round(price, 2),
            "qty_btc": qty,
        })
        atomic_write_json(MANUAL_FILE, manual_state)
        return f"✅ Logged BUY ${usd_amount:.2f} @ ${price:,.2f} (≈ {qty:.8f} BTC)"

    if side == "SELL":
        if manual_state["qty_btc"] <= 0:
            return "You have 0 BTC logged. Use /logbuy first."

        sell_qty = min(qty, manual_state["qty_btc"])
        avg_cost = (manual_state["cost_basis_usd"] / manual_state["qty_btc"]) if manual_state["qty_btc"] > 0 else 0
        cost_sold = avg_cost * sell_qty
        proceeds = sell_qty * price
        pnl = proceeds - cost_sold

        manual_state["qty_btc"] -= sell_qty
        manual_state["cost_basis_usd"] -= cost_sold
        manual_state["realized_pl_usd"] += pnl
        manual_state["trades"].append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "side": "SELL",
            "usd": round(proceeds, 2),
            "price": round(price, 2),
            "qty_btc": sell_qty,
            "pnl_usd": round(pnl, 2),
        })
        atomic_write_json(MANUAL_FILE, manual_state)
        return f"✅ Logged SELL ≈${proceeds:.2f} @ ${price:,.2f} (P/L {pnl:+.2f})"

    return "Unknown trade side."

# =========================
# TELEGRAM COMMANDS (polling)
# =========================
def telegram_delete_webhook():
    try:
        requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook",
            params={"drop_pending_updates": True},
            timeout=8,
        )
    except Exception:
        pass

def telegram_get_updates(offset: int):
    resp = requests.get(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
        params={"timeout": 10, "offset": offset},
        timeout=15,
    )
    data = resp.json()
    if not data.get("ok"):
        return [], offset
    updates = data.get("result", [])
    if updates:
        offset = updates[-1]["update_id"] + 1
    return updates, offset

def telegram_poll_loop():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    telegram_delete_webhook()
    offset = 0
    send_telegram("✅ BTC AI bot is live. Use /status")

    while True:
        try:
            updates, offset = telegram_get_updates(offset)
            for upd in updates:
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                chat_id = str(msg.get("chat", {}).get("id", ""))
                if chat_id != str(TELEGRAM_CHAT_ID):
                    continue

                text = (msg.get("text") or "").strip()
                if not text:
                    continue

                if text.startswith("/status"):
                    send_telegram(build_status_text())

                elif text.startswith("/logbuy"):
                    parts = text.split()
                    if len(parts) != 2:
                        send_telegram("Usage: /logbuy 100")
                        continue
                    try:
                        amt = float(parts[1])
                    except Exception:
                        send_telegram("Usage: /logbuy 100")
                        continue
                    send_telegram(handle_manual_trade("BUY", amt))

                elif text.startswith("/logsell"):
                    parts = text.split()
                    if len(parts) != 2:
                        send_telegram("Usage: /logsell 100")
                        continue
                    try:
                        amt = float(parts[1])
                    except Exception:
                        send_telegram("Usage: /logsell 100")
                        continue
                    send_telegram(handle_manual_trade("SELL", amt))

                elif text.startswith("/clearlogs"):
                    manual_state["qty_btc"] = 0.0
                    manual_state["cost_basis_usd"] = 0.0
                    manual_state["realized_pl_usd"] = 0.0
                    manual_state["trades"] = []
                    atomic_write_json(MANUAL_FILE, manual_state)
                    send_telegram("🧹 Cleared manual trade logs.")

        except Exception:
            pass

        time.sleep(2)

# =========================
# HTTP API
# =========================
class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/health":
            return self._json({"ok": True, "time": datetime.now().isoformat()})
        if p.path == "/state":
            with state_lock:
                return self._json(latest_state)
        return self._json({"error": "not found"}, code=404)

def run_http_server():
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"🌐 Serving HTTP on 0.0.0.0:{PORT}")
    server.serve_forever()

# =========================
# MAIN LOOP
# =========================
async def engine_loop():
    global paper_state, manual_state
    paper_state = safe_load_json(PAPER_STATE_FILE, paper_state)
    manual_state = safe_load_json(MANUAL_FILE, manual_state)

    last_alert = 0.0
    last_heartbeat = 0.0

    print("✅ BTC Alert Engine Running (AI-Filtered • Short-Term)")

    while True:
        try:
            candles, closes = fetch_candles(limit=60)
            price = closes[-1]
            rsi = compute_rsi(closes)

            momentum = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 6 else 0.0
            trend_strength = abs(momentum)

            trend = "WAIT"
            signal = "WAIT"
            if rsi < 30 and momentum > 0:
                trend = signal = "BUY"
            elif rsi > 70 and momentum < 0:
                trend = signal = "SELL"

            conf = confidence_score(rsi, trend_strength, momentum)
            now_str = datetime.now().strftime("%H:%M:%S")

            # paper trading step
            if signal in ("BUY", "SELL") and conf >= MIN_CONFIDENCE:
                paper_step(signal, price, now_str)

            # update shared state
            with state_lock:
                latest_state.update({
                    "price": round(price, 2),
                    "rsi": round(rsi, 1),
                    "trend": trend,
                    "state": signal,
                    "confidence": conf,
                    "time": now_str,
                    "candles": candles[-30:],
                    "notes": f"src=Coinbase • momentum={momentum:+.5f}",
                    "error": "",
                    "paper": dict(paper_state),
                    "manual": dict(manual_state),
                })

            # alerts (only on strong signals)
            now = time.time()
            if signal in ("BUY", "SELL") and conf >= MIN_CONFIDENCE and (now - last_alert) > ALERT_COOLDOWN:
                send_telegram(
                    f"📢 BTC {signal}\n"
                    f"Price: ${price:,.2f}\n"
                    f"RSI(1m): {round(rsi,1)}\n"
                    f"Confidence: {conf}%\n"
                    f"Momentum: {momentum:+.5f}"
                )
                last_alert = now

            # heartbeat
            if HEARTBEAT_INTERVAL_SECONDS > 0 and (now - last_heartbeat) > HEARTBEAT_INTERVAL_SECONDS:
                send_telegram(f"💓 BTC AI heartbeat  Price: ${price:,.2f}  RSI(1m): {round(rsi,1)}")
                last_heartbeat = now

        except Exception as e:
            with state_lock:
                latest_state.update({
                    "error": str(e),
                    "time": datetime.now().strftime("%H:%M:%S"),
                })

        await asyncio.sleep(5)

def main():
    # start HTTP server
    threading.Thread(target=run_http_server, daemon=True).start()

    # start telegram polling (if configured)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        threading.Thread(target=telegram_poll_loop, daemon=True).start()

    asyncio.run(engine_loop())

if __name__ == "__main__":
    main()
