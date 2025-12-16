import asyncio
import json
import time
import os
import requests
import numpy as np
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# =========================
# CONFIG
# =========================
PRODUCT = "BTC-USD"
GRANULARITY = 60  # 1 minute candles
COINBASE_BASE = "https://api.exchange.coinbase.com"

TELEGRAM_BOT_TOKEN = os.getenv("7586361018:AAFeb0aI25IjcR47dc9ysR7fkmL5godKXHs", "")
TELEGRAM_CHAT_ID = str(os.getenv("5698575104", "")).strip()

STATE_FILE = "btc_state.json"
ENGINE_INTERVAL_SECONDS = float(os.getenv("ENGINE_INTERVAL_SECONDS", "5"))

# Alerts (real Telegram alerts)
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", "300"))
MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "70"))

# Paper trading (sim)
PAPER_START_CASH = float(os.getenv("PAPER_START_CASH", "1000"))
PAPER_MIN_CONFIDENCE = int(os.getenv("PAPER_MIN_CONFIDENCE", "55"))

# Heartbeat (to know bot is alive)
HEARTBEAT_SECONDS = int(os.getenv("HEARTBEAT_SECONDS", "3600"))  # 1 hour default

# Files for persistence
MANUAL_FILE = "manual_trades.json"
PAPER_FILE = "paper_trades.json"

PORT = int(os.getenv("PORT", "8080"))

# =========================
# STORAGE (in memory)
# =========================
state_lock = threading.Lock()

latest_state = {
    "price": 0,
    "rsi": 0,
    "trend": "WAIT",
    "state": "WAIT",
    "confidence": 0,
    "time": "--:--:--",
    "candles": [],
    "notes": "",
    "error": "Starting…",
    "signal": "WAIT",
    "momentum": 0.0,
}

def load_json(path: str, default_obj):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
                if raw:
                    return json.loads(raw)
    except:
        pass
    return default_obj

def atomic_write_json(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)

manual_state = load_json(MANUAL_FILE, {
    "trades": [],
    "qty_btc": 0.0,
    "cost_basis_usd": 0.0,
    "realized_pl_usd": 0.0,
})

paper_state = load_json(PAPER_FILE, {
    "cash_usd": PAPER_START_CASH,
    "qty_btc": 0.0,
    "entry_price": None,
    "realized_pl_usd": 0.0,
    "trades": [],
})

# =========================
# TELEGRAM
# =========================
def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=10,
        )
    except:
        pass

def telegram_delete_webhook():
    if not TELEGRAM_BOT_TOKEN:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook",
            json={"drop_pending_updates": False},
            timeout=10,
        )
    except:
        pass

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
    # side: BUY / SELL
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
        # avg cost method
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

def telegram_poll_loop():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    telegram_delete_webhook()
    offset = 0

    while True:
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                params={"timeout": 50, "offset": offset},
                timeout=60,
            )
            data = resp.json()
            results = data.get("result", [])

            for upd in results:
                offset = max(offset, int(upd.get("update_id", 0)) + 1)
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue

                chat_id = str(msg.get("chat", {}).get("id", "")).strip()
                text = (msg.get("text") or "").strip()

                # only respond to your configured chat
                if chat_id != TELEGRAM_CHAT_ID:
                    continue

                if text.startswith("/status"):
                    send_telegram(build_status_text())

                elif text.startswith("/logbuy"):
                    parts = text.split()
                    if len(parts) != 2:
                        send_telegram("Usage: /logbuy 100")
                    else:
                        try:
                            amt = float(parts[1])
                            send_telegram(handle_manual_trade("BUY", amt))
                        except:
                            send_telegram("Usage: /logbuy 100")

                elif text.startswith("/logsell"):
                    parts = text.split()
                    if len(parts) != 2:
                        send_telegram("Usage: /logsell 100")
                    else:
                        try:
                            amt = float(parts[1])
                            send_telegram(handle_manual_trade("SELL", amt))
                        except:
                            send_telegram("Usage: /logsell 100")

        except:
            pass

        time.sleep(1)

# =========================
# MARKET DATA / SIGNALS
# =========================
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

def update_paper(signal: str, confidence: int, price: float):
    if price <= 0:
        return

    # only take paper actions when confidence is decent
    if confidence < PAPER_MIN_CONFIDENCE:
        return

    if signal == "BUY" and paper_state["qty_btc"] <= 0 and paper_state["cash_usd"] > 0:
        qty = paper_state["cash_usd"] / price
        paper_state["qty_btc"] = qty
        paper_state["entry_price"] = price
        paper_state["trades"].append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "side": "BUY",
            "price": round(price, 2),
            "qty_btc": qty,
        })
        paper_state["cash_usd"] = 0.0
        atomic_write_json(PAPER_FILE, paper_state)

    elif signal == "SELL" and paper_state["qty_btc"] > 0:
        qty = paper_state["qty_btc"]
        entry = float(paper_state["entry_price"] or price)
        proceeds = qty * price
        pnl = (price - entry) * qty

        paper_state["cash_usd"] = proceeds
        paper_state["qty_btc"] = 0.0
        paper_state["entry_price"] = None
        paper_state["realized_pl_usd"] += pnl
        paper_state["trades"].append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "side": "SELL",
            "price": round(price, 2),
            "qty_btc": qty,
            "pnl_usd": round(pnl, 2),
        })
        atomic_write_json(PAPER_FILE, paper_state)

# =========================
# HTTP API (for UI)
# =========================
class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/health":
            return self._json(200, {"ok": True})

        if path == "/state":
            with state_lock:
                s = dict(latest_state)

            # attach paper + manual summaries for UI
            price = float(s.get("price", 0) or 0)
            p_equity = paper_state["cash_usd"] + paper_state["qty_btc"] * price
            p_unreal = 0.0
            if paper_state.get("entry_price") and paper_state["qty_btc"] > 0:
                p_unreal = (price - float(paper_state["entry_price"])) * paper_state["qty_btc"]

            m_unreal = (manual_state["qty_btc"] * price) - manual_state["cost_basis_usd"]
            m_total = manual_state["realized_pl_usd"] + m_unreal

            s["paper"] = {
                "cash_usd": paper_state["cash_usd"],
                "qty_btc": paper_state["qty_btc"],
                "equity_usd": p_equity,
                "realized_pl_usd": paper_state["realized_pl_usd"],
                "unrealized_pl_usd": p_unreal,
                "last_trade": paper_state["trades"][-1] if paper_state["trades"] else None,
            }
            s["manual"] = {
                "qty_btc": manual_state["qty_btc"],
                "cost_basis_usd": manual_state["cost_basis_usd"],
                "realized_pl_usd": manual_state["realized_pl_usd"],
                "unrealized_pl_usd": m_unreal,
                "total_pl_usd": m_total,
                "last_trade": manual_state["trades"][-1] if manual_state["trades"] else None,
            }

            return self._json(200, s)

        return self._json(404, {"error": "Not found"})

# =========================
# MAIN LOOP
# =========================
async def main_loop():
    last_alert = 0.0
    last_heartbeat = 0.0

    # startup msg once (if configured)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        send_telegram("✅ BTC AI bot started. Use /status anytime.")

    while True:
        try:
            candles, closes = fetch_candles(limit=60)
            price = float(closes[-1])
            rsi = float(compute_rsi(closes))

            momentum = 0.0
            if len(closes) >= 6 and closes[-5] != 0:
                momentum = (closes[-1] - closes[-5]) / closes[-5]

            trend_strength = abs(momentum)

            signal = "WAIT"
            if rsi < 30 and momentum > 0:
                signal = "BUY"
            elif rsi > 70 and momentum < 0:
                signal = "SELL"

            confidence = confidence_score(rsi, trend_strength, momentum)

            with state_lock:
                latest_state.update({
                    "price": round(price, 2),
                    "rsi": round(rsi, 1),
                    "trend": signal if signal != "WAIT" else "WAIT",
                    "state": signal if signal != "WAIT" else "WAIT",
                    "signal": signal,
                    "confidence": int(confidence),
                    "momentum": float(momentum),
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "candles": candles[-30:],
                    "notes": f"src=Coinbase • momentum={momentum:.5f}",
                    "error": "",
                })

            atomic_write_json(STATE_FILE, latest_state)

            # paper sim
            update_paper(signal, confidence, price)

            # real alerts (filtered)
            now = time.time()
            if signal in ["BUY", "SELL"] and confidence >= MIN_CONFIDENCE and (now - last_alert) > ALERT_COOLDOWN:
                send_telegram(
                    f"📢 BTC {signal} ALERT\n"
                    f"Price: ${price:,.2f}\n"
                    f"RSI(1m): {round(rsi,1)}\n"
                    f"Confidence: {confidence}%\n"
                    f"Momentum(5m): {momentum:+.4f}"
                )
                last_alert = now

            # heartbeat (to prove it's alive)
            if HEARTBEAT_SECONDS > 0 and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                if (now - last_heartbeat) > HEARTBEAT_SECONDS:
                    send_telegram(f"💓 Heartbeat: ${price:,.2f} | RSI {round(rsi,1)} | {signal} | conf {confidence}%")
                    last_heartbeat = now

        except Exception as e:
            with state_lock:
                latest_state.update({
                    "error": str(e),
                    "time": datetime.now().strftime("%H:%M:%S"),
                })
            atomic_write_json(STATE_FILE, latest_state)

        await asyncio.sleep(ENGINE_INTERVAL_SECONDS)

def run_http_server():
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()

if __name__ == "__main__":
    # HTTP server thread (for /state)
    t = threading.Thread(target=run_http_server, daemon=True)
    t.start()

    # Telegram command poll thread
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        tp = threading.Thread(target=telegram_poll_loop, daemon=True)
        tp.start()

    asyncio.run(main_loop())
