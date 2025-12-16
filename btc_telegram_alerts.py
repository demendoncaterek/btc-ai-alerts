import json
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import numpy as np
import requests

# =========================
# CONFIG
# =========================
PRODUCT = os.getenv("PRODUCT", "BTC-USD")            # Coinbase product
GRANULARITY = int(os.getenv("GRANULARITY", "60"))   # seconds (60 = 1m candles)

STATE_FILE = os.getenv("STATE_FILE", "btc_state.json")
PAPER_FILE = os.getenv("PAPER_FILE", "paper_trades.json")
MANUAL_FILE = os.getenv("MANUAL_FILE", "manual_trades.json")

COINBASE_BASE = os.getenv("COINBASE_BASE", "https://api.exchange.coinbase.com")

ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", "300"))   # seconds
MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "70"))

ENGINE_LOOP_SECONDS = float(os.getenv("ENGINE_LOOP_SECONDS", "5"))
HEARTBEAT_SECONDS = int(os.getenv("HEARTBEAT_SECONDS", "21600"))  # 6 hours default

PAPER_START_CASH = float(os.getenv("PAPER_START_CASH", "1000"))
PAPER_TRADE_FRACTION = float(os.getenv("PAPER_TRADE_FRACTION", "1.0"))  # 1.0 = all-in

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Web server
PORT = int(os.getenv("PORT", "8080"))

# =========================
# STATE (in-memory)
# =========================
latest_state_lock = threading.Lock()
latest_state: dict = {}

# =========================
# JSON helpers
# =========================
def load_json(path: str, default):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if not raw:
                return default
            return json.loads(raw)
    except Exception:
        return default


def atomic_write_json(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


# =========================
# Telegram
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
    except Exception:
        pass


def format_status(state: dict) -> str:
    if not state:
        return "⚠️ Engine has no state yet."
    price = state.get("price", 0)
    rsi = state.get("rsi", 0)
    trend = state.get("trend", "WAIT")
    conf = state.get("confidence", 0)
    t = state.get("time", "--:--:--")
    paper = state.get("paper", {})
    manual = state.get("manual", {})
    paper_total = paper.get("total_pl_usd", 0)
    manual_total = manual.get("total_pl_usd", 0)

    return (
        f"🧠 BTC AI Status\n"
        f"Price: ${price:,.2f}\n"
        f"RSI(1m): {rsi}\n"
        f"Trend: {trend}\n"
        f"Confidence: {conf}%\n"
        f"Time: {t}\n\n"
        f"📄 Paper P/L: ${paper_total:,.2f}\n"
        f"🧾 Real (logged) P/L: ${manual_total:,.2f}"
    )


def apply_trade(tradebook: dict, side: str, usd_amount: float, price: float, label_time: str):
    """
    Very simple avg-cost position model (BTC only).
    tradebook fields:
      - btc
      - avg_entry
      - realized_pl_usd
      - trades (list)
    """
    btc = float(tradebook.get("btc", 0.0))
    avg_entry = float(tradebook.get("avg_entry", 0.0))
    realized = float(tradebook.get("realized_pl_usd", 0.0))

    side = side.upper().strip()
    usd_amount = float(usd_amount)

    if price <= 0 or usd_amount <= 0:
        return tradebook

    if side == "BUY":
        btc_bought = usd_amount / price
        new_btc = btc + btc_bought
        if new_btc > 0:
            avg_entry = ((btc * avg_entry) + (btc_bought * price)) / new_btc
        btc = new_btc
        trade = {"side": "BUY", "usd": round(usd_amount, 2), "price": round(price, 2), "btc": btc_bought, "time": label_time}

    elif side == "SELL":
        btc_sold = usd_amount / price
        btc_sold = min(btc_sold, btc)
        proceeds = btc_sold * price
        cost = btc_sold * avg_entry
        realized += (proceeds - cost)
        btc -= btc_sold
        if btc <= 1e-12:
            btc = 0.0
            avg_entry = 0.0
        trade = {"side": "SELL", "usd": round(proceeds, 2), "price": round(price, 2), "btc": btc_sold, "time": label_time}
    else:
        return tradebook

    trades = tradebook.get("trades", [])
    trades.append(trade)
    tradebook["btc"] = btc
    tradebook["avg_entry"] = avg_entry
    tradebook["realized_pl_usd"] = realized
    tradebook["trades"] = trades[-200:]
    return tradebook


def telegram_poll_loop():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    offset = 0
    send_telegram("✅ BTC engine started. Send /status anytime. (Paper trading only.)")

    while True:
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                params={"timeout": 25, "offset": offset},
                timeout=30,
            ).json()

            if not resp.get("ok"):
                time.sleep(2)
                continue

            for upd in resp.get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue

                chat_id = str(msg.get("chat", {}).get("id", ""))
                if TELEGRAM_CHAT_ID and chat_id != str(TELEGRAM_CHAT_ID):
                    continue

                text = (msg.get("text") or "").strip()
                if not text.startswith("/"):
                    continue

                parts = text.split()
                cmd = parts[0].lower()

                with latest_state_lock:
                    snap = dict(latest_state) if latest_state else {}

                if cmd in ["/help", "/start"]:
                    send_telegram(
                        "Commands:\n"
                        "/status — show current indicators + P/L\n"
                        "/logbuy <usd> — log a real buy amount (USD)\n"
                        "/logsell <usd> — log a real sell amount (USD)\n"
                        "/help — show this help"
                    )

                elif cmd == "/status":
                    send_telegram(format_status(snap))

                elif cmd in ["/logbuy", "/logsell"]:
                    if len(parts) < 2:
                        send_telegram("Usage: /logbuy 100  OR  /logsell 100")
                        continue
                    try:
                        usd = float(parts[1])
                    except Exception:
                        send_telegram("Amount must be a number, e.g. /logbuy 100")
                        continue

                    price = float(snap.get("price", 0) or 0)
                    if price <= 0:
                        send_telegram("⚠️ No live price yet. Try again in a few seconds.")
                        continue

                    manual = load_json(MANUAL_FILE, {"btc": 0.0, "avg_entry": 0.0, "realized_pl_usd": 0.0, "trades": []})
                    label_time = snap.get("time", datetime.now().strftime("%H:%M:%S"))
                    manual = apply_trade(manual, "BUY" if cmd == "/logbuy" else "SELL", usd, price, label_time)
                    atomic_write_json(MANUAL_FILE, manual)

                    send_telegram(f"🧾 Logged {cmd.replace('/log','').upper()} ${usd:,.2f} @ ${price:,.2f}\n\n" + format_status(snap))

        except Exception:
            time.sleep(2)


def heartbeat_loop():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    while True:
        try:
            with latest_state_lock:
                snap = dict(latest_state) if latest_state else {}
            if snap:
                send_telegram(
                    f"🫀 Heartbeat: ${snap.get('price',0):,.2f} • RSI {snap.get('rsi',0)} • "
                    f"{snap.get('trend','WAIT')} ({snap.get('confidence',0)}%)"
                )
        except Exception:
            pass
        time.sleep(max(60, HEARTBEAT_SECONDS))


# =========================
# Market data + indicators
# =========================
def fetch_candles(limit=60):
    url = f"{COINBASE_BASE}/products/{PRODUCT}/candles"
    resp = requests.get(
        url,
        params={"granularity": GRANULARITY},
        headers={"Accept": "application/json", "User-Agent": "btc-alerts"},
        timeout=12,
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
        candles.append(
            {
                "time": datetime.fromtimestamp(ts).strftime("%H:%M"),
                "open": float(c[3]),
                "high": float(c[2]),
                "low": float(c[1]),
                "close": float(c[4]),
            }
        )
        closes.append(float(c[4]))
    return candles, closes


def compute_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0)
    losses = -np.minimum(deltas, 0)
    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))
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
# Paper trading
# =========================
def init_paper():
    return {"cash_usd": PAPER_START_CASH, "btc": 0.0, "avg_entry": 0.0, "realized_pl_usd": 0.0, "trades": []}


def update_paper(paper: dict, signal: str, price: float, label_time: str):
    cash = float(paper.get("cash_usd", 0.0))
    btc = float(paper.get("btc", 0.0))

    if signal == "BUY" and cash > 1:
        usd_to_spend = cash * PAPER_TRADE_FRACTION
        paper = apply_trade(paper, "BUY", usd_to_spend, price, label_time)
        paper["cash_usd"] = cash - usd_to_spend

    if signal == "SELL" and btc > 0:
        usd_proceeds = btc * price
        paper = apply_trade(paper, "SELL", usd_proceeds, price, label_time)
        paper["cash_usd"] = cash + usd_proceeds

    return paper


def compute_pl(book: dict, price: float):
    btc = float(book.get("btc", 0.0))
    avg_entry = float(book.get("avg_entry", 0.0))
    realized = float(book.get("realized_pl_usd", 0.0))
    unrealized = (btc * (price - avg_entry)) if btc > 0 else 0.0
    total = realized + unrealized
    return realized, unrealized, total


# =========================
# HTTP server
# =========================
class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/health":
            self._json(200, {"ok": True, "service": "btc-engine", "time": datetime.now().isoformat()})
            return
        if path == "/state":
            with latest_state_lock:
                snap = dict(latest_state) if latest_state else {}
            self._json(200, snap)
            return
        self._json(404, {"error": "not_found"})


def start_http_server():
    srv = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"✅ Engine HTTP server listening on 0.0.0.0:{PORT}")
    srv.serve_forever()


# =========================
# Main engine loop
# =========================
def engine_loop():
    last_alert_ts = 0
    paper = load_json(PAPER_FILE, init_paper())
    manual = load_json(MANUAL_FILE, {"btc": 0.0, "avg_entry": 0.0, "realized_pl_usd": 0.0, "trades": []})

    print("✅ BTC Alert Engine Running (Paper + Real logging)")

    while True:
        try:
            candles, closes = fetch_candles(limit=60)
            price = float(closes[-1])
            rsi = float(compute_rsi(closes))
            momentum = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 6 else 0.0
            trend_strength = abs(momentum)

            trend = state = "WAIT"
            if rsi < 30 and momentum > 0:
                trend = state = "BUY"
            elif rsi > 70 and momentum < 0:
                trend = state = "SELL"

            conf = confidence_score(rsi, trend_strength, momentum)
            label_time = datetime.now().strftime("%H:%M:%S")

            if state in ("BUY", "SELL"):
                paper = update_paper(paper, state, price, label_time)
                atomic_write_json(PAPER_FILE, paper)

            manual = load_json(MANUAL_FILE, manual)

            paper_realized, paper_unreal, paper_total = compute_pl(paper, price)
            manual_realized, manual_unreal, manual_total = compute_pl(manual, price)

            state_obj = {
                "price": round(price, 2),
                "rsi": round(rsi, 1),
                "trend": trend,
                "state": state,
                "confidence": conf,
                "time": label_time,
                "candles": candles[-30:],
                "notes": f"src=Coinbase • momentum={momentum:.5f}",
                "error": "",
                "paper": {
                    "cash_usd": round(float(paper.get("cash_usd", 0.0)), 2),
                    "btc": float(paper.get("btc", 0.0)),
                    "avg_entry": round(float(paper.get("avg_entry", 0.0)), 2),
                    "realized_pl_usd": round(paper_realized, 2),
                    "unrealized_pl_usd": round(paper_unreal, 2),
                    "total_pl_usd": round(paper_total, 2),
                    "trades": paper.get("trades", [])[-50:],
                },
                "manual": {
                    "btc": float(manual.get("btc", 0.0)),
                    "avg_entry": round(float(manual.get("avg_entry", 0.0)), 2),
                    "realized_pl_usd": round(manual_realized, 2),
                    "unrealized_pl_usd": round(manual_unreal, 2),
                    "total_pl_usd": round(manual_total, 2),
                    "trades": manual.get("trades", [])[-50:],
                },
            }

            atomic_write_json(STATE_FILE, state_obj)
            with latest_state_lock:
                latest_state.clear()
                latest_state.update(state_obj)

            now = time.time()
            if state in ("BUY", "SELL") and conf >= MIN_CONFIDENCE and (now - last_alert_ts) > ALERT_COOLDOWN:
                send_telegram(
                    f"📢 BTC {state} ALERT\n"
                    f"Price: ${price:,.2f}\n"
                    f"RSI(1m): {round(rsi,1)}\n"
                    f"Confidence: {conf}%"
                )
                last_alert_ts = now

        except Exception as e:
            err = str(e)
            state_obj = {
                "price": 0,
                "rsi": 0,
                "trend": "WAIT",
                "state": "WAIT",
                "confidence": 0,
                "time": datetime.now().strftime("%H:%M:%S"),
                "candles": [],
                "notes": "",
                "error": err,
                "paper": load_json(PAPER_FILE, init_paper()),
                "manual": load_json(MANUAL_FILE, {"btc": 0.0, "avg_entry": 0.0, "realized_pl_usd": 0.0, "trades": []}),
            }
            atomic_write_json(STATE_FILE, state_obj)
            with latest_state_lock:
                latest_state.clear()
                latest_state.update(state_obj)

        time.sleep(ENGINE_LOOP_SECONDS)


def main():
    threading.Thread(target=start_http_server, daemon=True).start()
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        threading.Thread(target=telegram_poll_loop, daemon=True).start()
        threading.Thread(target=heartbeat_loop, daemon=True).start()
    engine_loop()


if __name__ == "__main__":
    main()
