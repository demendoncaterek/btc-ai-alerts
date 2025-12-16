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
            "usd": round(usd, 2),
            "btc": round(btc_to_sell, 8),
            "price": round(price, 2),
            "realized_pl": round(realized, 2),
        })
        return True, None


# =========================
# TELEGRAM COMMANDS
# =========================
def format_status():
    with state_lock:
        s = dict(STATE)

    price = s.get("price", 0.0)
    rsi = s.get("rsi", 0.0)
    trend = s.get("trend", "WAIT")
    conf = s.get("confidence", 0)
    mom = s.get("momentum", 0.0)
    t = s.get("time", "--:--:--")

    paper = s.get("paper", {})
    real = s.get("real", {})

    up = int(time.time() - PERSIST["engine"]["started_at"])
    return (
        f"📊 BTC Engine Status\n"
        f"Price: ${price:,.2f}\n"
        f"RSI(1m): {rsi}\n"
        f"Momentum(~5m): {mom:+.5f}\n"
        f"Signal: {trend} | Confidence: {conf}%\n"
        f"Last update: {t}\n"
        f"\n🧪 Paper\n"
        f"Equity: ${paper.get('equity', 0):,.2f}\n"
        f"P/L: ${paper.get('total_pl', 0):,.2f} (R {paper.get('realized_pl', 0):,.2f} / U {paper.get('unrealized_pl', 0):,.2f})\n"
        f"\n🧾 Real (logged)\n"
        f"Equity: ${real.get('equity', 0):,.2f}\n"
        f"P/L: ${real.get('total_pl', 0):,.2f} (R {real.get('realized_pl', 0):,.2f} / U {real.get('unrealized_pl', 0):,.2f})\n"
        f"\n⏱ Uptime: {up//3600}h {(up%3600)//60}m"
    )

async def telegram_loop():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    offset = int(PERSIST.get("telegram", {}).get("offset", 0))

    while True:
        try:
            r = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                params={"timeout": 30, "offset": offset},
                timeout=35,
            )
            data = r.json()
            if not data.get("ok"):
                await asyncio.sleep(2)
                continue

            for upd in data.get("result", []):
                offset = max(offset, int(upd.get("update_id", 0)) + 1)

                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue

                chat_id = str((msg.get("chat") or {}).get("id", ""))
                if chat_id != TELEGRAM_CHAT_ID:
                    continue

                text = (msg.get("text") or "").strip()
                if not text.startswith("/"):
                    continue

                parts = text.split()
                cmd = parts[0].lower()

                with state_lock:
                    price = float(STATE.get("price", 0.0))

                if cmd in ("/help", "/start"):
                    send_telegram(
                        "Commands:\n"
                        "/status\n"
                        "/logbuy <usd>\n"
                        "/logsell <usd>\n"
                        "Example: /logbuy 100"
                    )

                elif cmd == "/status":
                    send_telegram(format_status())

                elif cmd == "/logbuy":
                    if len(parts) != 2:
                        send_telegram("Usage: /logbuy 100")
                    else:
                        try:
                            usd = float(parts[1])
                            if usd <= 0:
                                raise ValueError()
                            if price <= 0:
                                send_telegram("Price not ready yet — try again in a few seconds.")
                            else:
                                real_log_buy(usd, price)
                                atomic_write_json(PERSIST_FILE, PERSIST)
                                send_telegram(f"✅ Logged REAL BUY ${usd:,.2f} @ ${price:,.2f}")
                        except:
                            send_telegram("Usage: /logbuy 100")

                elif cmd == "/logsell":
                    if len(parts) != 2:
                        send_telegram("Usage: /logsell 100")
                    else:
                        try:
                            usd = float(parts[1])
                            if usd <= 0:
                                raise ValueError()
                            if price <= 0:
                                send_telegram("Price not ready yet — try again in a few seconds.")
                            else:
                                ok, err = real_log_sell(usd, price)
                                if not ok:
                                    send_telegram(f"❌ {err}")
                                else:
                                    atomic_write_json(PERSIST_FILE, PERSIST)
                                    send_telegram(f"✅ Logged REAL SELL ${usd:,.2f} @ ${price:,.2f}")
                        except:
                            send_telegram("Usage: /logsell 100")

            PERSIST["telegram"]["offset"] = offset
            atomic_write_json(PERSIST_FILE, PERSIST)

        except:
            pass

        await asyncio.sleep(2)

async def heartbeat_loop():
    if HEARTBEAT_MINUTES <= 0:
        return
    while True:
        try:
            send_telegram("💚 Heartbeat\n" + format_status())
        except:
            pass
        await asyncio.sleep(HEARTBEAT_MINUTES * 60)


# =========================
# HTTP SERVER: /state
# =========================
class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path

        if path in ("/health", "/"):
            return self._send_json({"ok": True, "time": now_str()})

        if path == "/state":
            with state_lock:
                s = dict(STATE)
            return self._send_json(s)

        return self._send_json({"error": "not found"}, code=404)

def start_http_server():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()


# =========================
# ENGINE LOOP
# =========================
async def engine_loop():
    last_alert_ts = 0.0
    print("✅ BTC Engine running")

    while True:
        try:
            candles, closes = fetch_candles(limit=60)
            price = float(closes[-1])
            rsi = float(compute_rsi(closes, period=14))

            momentum = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 6 else 0.0
            trend_strength = abs(momentum)

            trend = "WAIT"
            signal_state = "WAIT"

            if (rsi < 30) and (momentum > 0):
                trend = signal_state = "BUY"
            elif (rsi > 70) and (momentum < 0):
                trend = signal_state = "SELL"

            confidence = confidence_score(rsi, trend_strength, momentum)

            paper_trade_logic(signal_state, confidence, price)

            with persist_lock:
                paper_snap = compute_portfolio_snapshot(PERSIST["paper"], price)
                real_snap = compute_portfolio_snapshot(PERSIST["real"], price)
                paper_trades = PERSIST["trades"]["paper"][-20:]
                real_trades = PERSIST["trades"]["real"][-20:]

            with state_lock:
                STATE.update({
                    "price": round(price, 2),
                    "rsi": round(rsi, 1),
                    "trend": trend,
                    "state": signal_state,
                    "confidence": int(confidence),
                    "momentum": float(momentum),
                    "time": now_str(),
                    "candles": candles[-30:],
                    "notes": f"src=Coinbase • momentum={momentum:+.5f}",
                    "error": "",
                    "paper": paper_snap,
                    "real": real_snap,
                    "trades": {"paper": paper_trades, "real": real_trades},
                })

            now = time.time()
            if (
                signal_state in ("BUY", "SELL")
                and confidence >= MIN_CONFIDENCE
                and (now - last_alert_ts) > ALERT_COOLDOWN
            ):
                send_telegram(
                    f"📢 BTC {signal_state} ALERT\n"
                    f"Price: ${price:,.2f}\n"
                    f"RSI(1m): {round(rsi,1)}\n"
                    f"Confidence: {confidence}%\n"
                    f"Momentum(~5m): {momentum:+.5f}"
                )
                last_alert_ts = now

            with persist_lock:
                atomic_write_json(PERSIST_FILE, PERSIST)
            atomic_write_json(STATE_FILE, STATE)

        except Exception as e:
            with state_lock:
                STATE["error"] = str(e)
                STATE["time"] = now_str()
            atomic_write_json(STATE_FILE, STATE)

        await asyncio.sleep(ENGINE_TICK_SECONDS)

async def main():
    loaded = safe_load_json(PERSIST_FILE, None)
    if isinstance(loaded, dict):
        with persist_lock:
            PERSIST.update(loaded)

    threading.Thread(target=start_http_server, daemon=True).start()

    if STARTUP_MESSAGE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        send_telegram("✅ BTC engine is live. Use /status, /logbuy 100, /logsell 100")

    await asyncio.gather(
        engine_loop(),
        telegram_loop(),
        heartbeat_loop(),
    )

if __name__ == "__main__":
    asyncio.run(main())
