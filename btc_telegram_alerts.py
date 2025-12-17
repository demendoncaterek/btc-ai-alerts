import asyncio
import json
import os
import time
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import numpy as np
import requests

# ============================================================
# BTC ENGINE (15m execution + 1h/4h bias + SL/TP + explanations)
# ============================================================

# -------------------------
# CONFIG
# -------------------------
PRODUCT = os.getenv("PRODUCT", "BTC-USD")
COINBASE_BASE = "https://api.exchange.coinbase.com"

# Execution timeframe: 15m (more stable than 1m/5m)
BASE_GRANULARITY = int(os.getenv("BASE_GRANULARITY", "900"))       # 900 = 15m
BIAS1_GRANULARITY = int(os.getenv("BIAS1_GRANULARITY", "3600"))    # 1h
BIAS2_GRANULARITY = int(os.getenv("BIAS2_GRANULARITY", "14400"))   # 4h

BASE_LIMIT = int(os.getenv("BASE_LIMIT", "240"))   # 240*15m ≈ 60h
BIAS1_LIMIT = int(os.getenv("BIAS1_LIMIT", "240")) # 240*1h ≈ 10d
BIAS2_LIMIT = int(os.getenv("BIAS2_LIMIT", "240")) # 240*4h ≈ 40d

STATE_FILE = os.getenv("STATE_FILE", "btc_state.json")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

PORT = int(os.getenv("PORT", "8080"))

# Alerts
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", "3600"))  # 60 min
MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "70"))

# “Crazy move” detectors (relative move over last N bars)
MOVE_WINDOW_BARS = int(os.getenv("MOVE_WINDOW_BARS", "12"))         # 12*15m = 3h
MOVE_PCT = float(os.getenv("MOVE_PCT", "1.0")) / 100.0              # 1.00%

# Paper trading
PAPER_START_USD = float(os.getenv("PAPER_START_USD", "250.0"))
PAPER_FEE_BPS = float(os.getenv("PAPER_FEE_BPS", "10")) / 10000.0   # 10 bps = 0.10%

# ATR risk model
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.5"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))

# Heartbeat (0 disables)
HEARTBEAT_MINUTES = int(os.getenv("HEARTBEAT_MINUTES", "120"))

# Loop speed
LOOP_SECONDS = int(os.getenv("LOOP_SECONDS", "20"))


# -------------------------
# Helpers
# -------------------------
def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def atomic_write_json(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def send_telegram(msg: str):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
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

def fetch_candles(granularity: int, limit: int):
    url = f"{COINBASE_BASE}/products/{PRODUCT}/candles"
    resp = requests.get(
        url,
        params={"granularity": granularity},
        headers={"Accept": "application/json", "User-Agent": "btc-ai-alerts"},
        timeout=15,
    )
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Coinbase API error: {data}")

    data.sort(key=lambda x: x[0])
    data = data[-limit:]

    candles = []
    closes = []
    highs = []
    lows = []
    for c in data:
        ts = int(c[0])
        # Coinbase format: [ time, low, high, open, close, volume ]
        candles.append({
            "ts": ts,
            "time": datetime.fromtimestamp(ts).strftime("%m-%d %H:%M"),
            "open": float(c[3]),
            "high": float(c[2]),
            "low": float(c[1]),
            "close": float(c[4]),
        })
        closes.append(float(c[4]))
        highs.append(float(c[2]))
        lows.append(float(c[1]))

    return candles, np.array(closes, dtype=float), np.array(highs, dtype=float), np.array(lows, dtype=float)


# -------------------------
# Indicators
# -------------------------
def ema(x: np.ndarray, period: int) -> np.ndarray:
    if len(x) == 0:
        return np.array([])
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out

def rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0.0)
    losses = -np.minimum(deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))

def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 0.0
    prev_close = closes[:-1]
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - prev_close), np.abs(lows[1:] - prev_close)),
    )
    return float(np.mean(tr[-period:]))

def bollinger(prices: np.ndarray, period: int = 20, std_mult: float = 2.0):
    if len(prices) < period:
        return (np.nan, np.nan, np.nan)
    window = prices[-period:]
    mid = float(np.mean(window))
    sd = float(np.std(window))
    upper = mid + std_mult * sd
    lower = mid - std_mult * sd
    return (lower, mid, upper)

def macd_series(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    if len(prices) < slow + signal:
        return np.zeros_like(prices)
    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)
    line = fast_ema - slow_ema
    sig = ema(line, signal)
    hist = line - sig
    return hist

def zscore(prices: np.ndarray, period: int = 50) -> float:
    if len(prices) < period:
        return 0.0
    w = prices[-period:]
    mu = np.mean(w)
    sd = np.std(w)
    if sd == 0:
        return 0.0
    return float((prices[-1] - mu) / sd)


# -------------------------
# Store (paper + real logs)
# -------------------------
def load_store():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip() or "{}"
            return json.loads(raw)
    except:
        return {}

def init_store(store: dict):
    store.setdefault("paper", {"usd": PAPER_START_USD, "btc": 0.0, "avg_entry": 0.0, "trades": []})
    store.setdefault("real", {"trades": [], "position_btc": 0.0, "avg_entry": 0.0})
    store.setdefault("events", [])
    store.setdefault("last_state", {})
    return store

def paper_buy(store: dict, usd_amount: float, price: float, reason: str):
    p = store["paper"]
    usd_amount = max(0.0, min(usd_amount, p["usd"]))
    if usd_amount <= 0:
        return False
    fee = usd_amount * PAPER_FEE_BPS
    usd_net = usd_amount - fee
    btc = usd_net / price
    new_btc = p["btc"] + btc
    if new_btc > 0:
        p["avg_entry"] = (p["avg_entry"] * p["btc"] + price * btc) / new_btc
    p["btc"] = new_btc
    p["usd"] -= usd_amount
    p["trades"].append({"ts": _now_iso(), "side": "BUY", "usd": usd_amount, "price": price, "fee": fee, "reason": reason})
    return True

def paper_sell_all(store: dict, price: float, reason: str):
    p = store["paper"]
    btc_amount = p["btc"]
    if btc_amount <= 0:
        return False
    usd_gross = btc_amount * price
    fee = usd_gross * PAPER_FEE_BPS
    usd_net = usd_gross - fee
    p["btc"] = 0.0
    p["avg_entry"] = 0.0
    p["usd"] += usd_net
    p["trades"].append({"ts": _now_iso(), "side": "SELL", "btc": btc_amount, "price": price, "fee": fee, "reason": reason})
    return True

def calc_paper_summary(store: dict, price: float):
    p = store["paper"]
    equity = p["usd"] + p["btc"] * price
    return {"equity": equity, "usd": p["usd"], "btc": p["btc"], "avg_entry": p["avg_entry"], "pnl": equity - PAPER_START_USD}

def real_log(store: dict, side: str, qty_btc: float, price: float, note: str):
    r = store["real"]
    side = side.upper()
    if side not in ("BUY", "SELL"):
        return False
    qty_btc = max(0.0, qty_btc)
    if qty_btc <= 0:
        return False

    if side == "BUY":
        new_pos = r["position_btc"] + qty_btc
        if new_pos > 0:
            r["avg_entry"] = (r["avg_entry"] * r["position_btc"] + price * qty_btc) / new_pos
        r["position_btc"] = new_pos
    else:
        r["position_btc"] = max(0.0, r["position_btc"] - qty_btc)
        if r["position_btc"] == 0:
            r["avg_entry"] = 0.0

    r["trades"].append({"ts": _now_iso(), "side": side, "btc": qty_btc, "price": price, "note": note})
    return True

def calc_real_summary(store: dict, price: float):
    r = store["real"]
    pos = r["position_btc"]
    avg = r["avg_entry"]
    unreal = (price - avg) * pos if pos > 0 else 0.0
    return {"position_btc": pos, "avg_entry": avg, "unrealized": unreal}


# -------------------------
# Decision logic (15m + 1h/4h bias)
# -------------------------
def bias_label(closes: np.ndarray):
    # EMA50 vs EMA200 for bias
    if len(closes) < 200:
        return "NEUTRAL"
    e50 = ema(closes, 50)[-1]
    e200 = ema(closes, 200)[-1]
    if e50 > e200:
        return "UP"
    if e50 < e200:
        return "DOWN"
    return "NEUTRAL"

def build_sl_tp(price: float, atr_v: float, side: str):
    if atr_v <= 0:
        return (None, None, None)
    if side == "BUY":
        sl = price - (SL_ATR_MULT * atr_v)
        tp = price + (TP_ATR_MULT * atr_v)
        rr = (tp - price) / max(1e-9, (price - sl))
    else:
        sl = price + (SL_ATR_MULT * atr_v)
        tp = price - (TP_ATR_MULT * atr_v)
        rr = (price - tp) / max(1e-9, (sl - price))
    return (round(sl, 2), round(tp, 2), round(rr, 2))

def decide(base_prices, base_highs, base_lows, bias1_prices, bias2_prices):
    price = float(base_prices[-1])

    b1 = bias_label(bias1_prices)
    b2 = bias_label(bias2_prices)
    bias = "NEUTRAL"
    if b1 == b2 and b1 in ("UP", "DOWN"):
        bias = b1

    base_rsi = rsi(base_prices, 14)
    lower, mid, upper = bollinger(base_prices, 20, 2.0)
    atr_v = atr(base_highs, base_lows, base_prices, ATR_PERIOD)
    macd_hist = macd_series(base_prices)
    macd_hist_now = float(macd_hist[-1])
    zs = zscore(base_prices, 50)

    # 3-hour move detector (crazy dip/peak)
    lookback = min(MOVE_WINDOW_BARS, len(base_prices))
    w = base_prices[-lookback:]
    w_high = float(np.max(w))
    w_low = float(np.min(w))
    crazy_dip = price <= w_high * (1.0 - MOVE_PCT)
    crazy_peak = price >= w_low * (1.0 + MOVE_PCT)

    # Confidence breakdown (explainable)
    reasons = []
    score = 50

    # Bias alignment (big deal for swing-style)
    if bias == "UP":
        score += 12
        reasons.append("1h+4h bias UP")
    elif bias == "DOWN":
        score += 12
        reasons.append("1h+4h bias DOWN")
    else:
        reasons.append("bias NEUTRAL")

    # RSI extremes
    if base_rsi <= 30:
        score += 12
        reasons.append("RSI oversold")
    elif base_rsi >= 70:
        score += 12
        reasons.append("RSI overbought")

    # Bollinger interaction
    if not np.isnan(lower) and price <= lower:
        score += 10
        reasons.append("at/below lower Bollinger")
    if not np.isnan(upper) and price >= upper:
        score += 10
        reasons.append("at/above upper Bollinger")

    # MACD momentum
    if macd_hist_now > 0:
        score += 6
        reasons.append("MACD momentum bullish")
    elif macd_hist_now < 0:
        score += 6
        reasons.append("MACD momentum bearish")

    # Z-score extremes
    if zs <= -1.2:
        score += 8
        reasons.append("price statistically low (z)")
    elif zs >= 1.2:
        score += 8
        reasons.append("price statistically high (z)")

    # Crazy move flags
    if crazy_dip:
        score += 10
        reasons.append("crazy dip (≈3h)")
    if crazy_peak:
        score += 10
        reasons.append("crazy peak (≈3h)")

    confidence = int(np.clip(score, 0, 100))

    signal = "WAIT"
    signal_reason = "No high-probability setup"

    # High-probability BUY: bias UP preferred, or deep mean reversion
    if (bias == "UP" and base_rsi <= 40 and (price <= lower or crazy_dip) and macd_hist_now > 0) or (base_rsi <= 28 and zs <= -1.2):
        signal = "BUY"
        signal_reason = "Dip-in-uptrend or deep oversold bounce"

    # High-probability SELL: bias DOWN preferred, or deep mean reversion
    if (bias == "DOWN" and base_rsi >= 60 and (price >= upper or crazy_peak) and macd_hist_now < 0) or (base_rsi >= 72 and zs >= 1.2):
        signal = "SELL"
        signal_reason = "Peak-in-downtrend or deep overbought fade"

    sl, tp, rr = (None, None, None)
    if signal in ("BUY", "SELL"):
        sl, tp, rr = build_sl_tp(price, atr_v, signal)

    return {
        "price": price,
        "signal": signal,
        "reason": signal_reason,
        "confidence": confidence,
        "confidence_reasons": reasons,
        "bias_1h": b1,
        "bias_4h": b2,
        "bias": bias,
        "rsi": round(base_rsi, 1),
        "atr": round(atr_v, 2),
        "bb_lower": None if np.isnan(lower) else round(lower, 2),
        "bb_mid": None if np.isnan(mid) else round(mid, 2),
        "bb_upper": None if np.isnan(upper) else round(upper, 2),
        "zscore": round(zs, 2),
        "macd_hist_now": round(macd_hist_now, 6),
        "macd_hist_series": macd_hist.tolist(),
        "crazy_dip": bool(crazy_dip),
        "crazy_peak": bool(crazy_peak),
        "sl": sl,
        "tp": tp,
        "rr": rr,
    }


# -------------------------
# Telegram commands
# -------------------------
LATEST_STATE = {"ok": False, "error": "starting"}

def build_status_text(s: dict):
    price = s.get("price", 0)
    msg = (
        f"🧠 BTC Bot Status\n"
        f"Price: ${price:,.2f}\n"
        f"Signal: {s.get('signal','WAIT')} (conf {s.get('confidence',0)}%)\n"
        f"Bias: {s.get('bias','NEUTRAL')} (1h={s.get('bias_1h','?')}, 4h={s.get('bias_4h','?')})\n"
        f"RSI(15m): {s.get('rsi','--')}\n"
    )
    if s.get("sl") and s.get("tp"):
        msg += f"SL: ${s['sl']:,.2f}  TP: ${s['tp']:,.2f}  R:R≈{s.get('rr','--')}\n"
    msg += f"Reason: {s.get('reason','')}\n"
    return msg

def build_explain_text(s: dict):
    lines = ["🧠 Why the bot thinks this:", f"Signal: {s.get('signal')} (conf {s.get('confidence')}%)", ""]
    for r in s.get("confidence_reasons", [])[:20]:
        lines.append(f"• {r}")
    if s.get("sl") and s.get("tp"):
        lines.append("")
        lines.append(f"Risk idea (ATR): SL ${s['sl']:,.2f} | TP ${s['tp']:,.2f} | R:R≈{s.get('rr')}")
    return "\n".join(lines)

def handle_command(text: str):
    if not text:
        return
    parts = text.strip().split()
    cmd = parts[0].lower()
    s = LATEST_STATE if isinstance(LATEST_STATE, dict) else {}

    if cmd in ("/start", "/help"):
        send_telegram("✅ Bot running.\nCommands: /status, /explain, /logbuy 100, /logsell 100")
        return

    if cmd == "/status":
        send_telegram(build_status_text(s))
        return

    if cmd == "/explain":
        send_telegram(build_explain_text(s))
        return

    if cmd in ("/logbuy", "/logsell"):
        if len(parts) < 2:
            send_telegram("Usage: /logbuy 100 [price] or /logsell 100 [price]")
            return
        usd_amount = safe_float(parts[1], 0.0)
        price = safe_float(parts[2], s.get("price", 0.0)) if len(parts) >= 3 else float(s.get("price", 0.0))
        if usd_amount <= 0 or price <= 0:
            send_telegram("Invalid amount/price.")
            return

        qty_btc = usd_amount / price
        store = init_store(load_store())
        side = "BUY" if cmd == "/logbuy" else "SELL"
        ok = real_log(store, side, qty_btc, price, note="manual log via telegram")
        if ok:
            atomic_write_json(STATE_FILE, store)
            send_telegram(f"✅ Logged {side} ${usd_amount:,.2f} (~{qty_btc:.6f} BTC) @ ${price:,.2f}")
        else:
            send_telegram("Could not log trade.")
        return

def telegram_poll_loop():
    if not TELEGRAM_BOT_TOKEN:
        return
    telegram_delete_webhook()
    offset = None

    while True:
        try:
            params = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset
            r = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                params=params,
                timeout=35,
            )
            data = r.json()
            if not data.get("ok"):
                time.sleep(2)
                continue

            for upd in data.get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or {}
                chat = msg.get("chat") or {}
                chat_id = str(chat.get("id", ""))

                if TELEGRAM_CHAT_ID and str(TELEGRAM_CHAT_ID) != chat_id:
                    continue

                handle_command(msg.get("text", ""))

        except Exception:
            time.sleep(2)


# -------------------------
# HTTP endpoints for UI
# -------------------------
class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._json(200, {"ok": True, "service": "btc-engine", "time": _now_iso()})
            return
        if path == "/state":
            self._json(200, LATEST_STATE)
            return
        self._json(404, {"ok": False, "error": "not found"})

def start_http_server():
    httpd = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"✅ Engine HTTP server listening on 0.0.0.0:{PORT}")
    httpd.serve_forever()


# -------------------------
# Main loop
# -------------------------
async def main():
    global LATEST_STATE

    threading.Thread(target=start_http_server, daemon=True).start()

    store = init_store(load_store())
    atomic_write_json(STATE_FILE, store)

    if TELEGRAM_BOT_TOKEN:
        threading.Thread(target=telegram_poll_loop, daemon=True).start()

    if TELEGRAM_BOT_TOKEN and HEARTBEAT_MINUTES > 0:
        send_telegram("✅ BTC engine restarted. Use /status or /explain anytime.")

    last_alert = 0.0
    last_move_alert = 0.0
    last_heartbeat = 0.0

    print("✅ BTC Alert Engine Running (15m exec • 1h+4h bias)")

    while True:
        try:
            base_candles, base_prices, base_highs, base_lows = fetch_candles(BASE_GRANULARITY, BASE_LIMIT)
            _, b1_prices, _, _ = fetch_candles(BIAS1_GRANULARITY, BIAS1_LIMIT)
            _, b2_prices, _, _ = fetch_candles(BIAS2_GRANULARITY, BIAS2_LIMIT)

            d = decide(base_prices, base_highs, base_lows, b1_prices, b2_prices)
            price = d["price"]

            store = init_store(load_store())

            # Event markers for UI
            events = store.get("events", [])
            now = time.time()

            # Crazy move alert (dip/peak)
            if (d["crazy_dip"] or d["crazy_peak"]) and now - last_move_alert > 3600:
                last_move_alert = now
                label = "CRAZY DIP" if d["crazy_dip"] else "CRAZY PEAK"
                events.append({"ts": _now_iso(), "type": label, "price": price, "label": label})
                send_telegram(
                    f"⚡ {label}\nPrice: ${price:,.2f}\n"
                    f"Window: ~{MOVE_WINDOW_BARS*BASE_GRANULARITY//60}m\nBias: {d['bias']}"
                )

            # High-confidence setup alert (+ optional paper sim)
            if d["signal"] in ("BUY", "SELL") and d["confidence"] >= MIN_CONFIDENCE and now - last_alert > ALERT_COOLDOWN:
                last_alert = now
                events.append({"ts": _now_iso(), "type": d["signal"], "price": price, "label": d["signal"]})

                msg = (
                    f"📢 BTC {d['signal']} (setup)\n"
                    f"Price: ${price:,.2f}\n"
                    f"Bias: {d['bias']} (1h={d['bias_1h']}, 4h={d['bias_4h']})\n"
                    f"RSI(15m): {d['rsi']} | Conf: {d['confidence']}%\n"
                    f"Reason: {d['reason']}\n"
                )
                if d["sl"] and d["tp"]:
                    msg += f"SL: ${d['sl']:,.2f} | TP: ${d['tp']:,.2f} | R:R≈{d.get('rr')}\n"
                msg += "Use /explain for the full breakdown."
                send_telegram(msg)

                # Conservative paper sim: buy only if flat, sell only if holding
                p = store["paper"]
                if d["signal"] == "BUY" and p["btc"] <= 0 and p["usd"] > 10:
                    paper_buy(store, usd_amount=min(50.0, p["usd"]), price=price, reason=d["reason"])
                elif d["signal"] == "SELL" and p["btc"] > 0:
                    paper_sell_all(store, price=price, reason=d["reason"])

            # Heartbeat
            if TELEGRAM_BOT_TOKEN and HEARTBEAT_MINUTES > 0 and now - last_heartbeat > HEARTBEAT_MINUTES * 60:
                last_heartbeat = now
                send_telegram(f"🫀 Heartbeat\nPrice: ${price:,.2f}\nSignal: {d['signal']} (conf {d['confidence']}%)\nBias: {d['bias']}")

            # Trim events
            if len(events) > 300:
                events = events[-300:]
            store["events"] = events

            # Build UI state
            state = {
                "ok": True,
                "time": datetime.now().strftime("%H:%M:%S"),
                "iso": _now_iso(),
                "product": PRODUCT,

                "price": round(price, 2),
                "signal": d["signal"],
                "reason": d["reason"],
                "confidence": d["confidence"],
                "confidence_reasons": d["confidence_reasons"],

                "base_granularity": BASE_GRANULARITY,
                "bias_1h": d["bias_1h"],
                "bias_4h": d["bias_4h"],
                "bias": d["bias"],

                "rsi": d["rsi"],
                "atr": d["atr"],
                "bb_lower": d["bb_lower"],
                "bb_mid": d["bb_mid"],
                "bb_upper": d["bb_upper"],
                "zscore": d["zscore"],
                "macd_hist_now": d["macd_hist_now"],
                "macd_hist_series": d["macd_hist_series"],

                "crazy_dip": d["crazy_dip"],
                "crazy_peak": d["crazy_peak"],

                "sl": d["sl"],
                "tp": d["tp"],
                "rr": d["rr"],

                "candles": base_candles[-120:],  # last ~30h on 15m
                "events": store.get("events", []),

                "paper_summary": calc_paper_summary(store, price),
                "real_summary": calc_real_summary(store, price),

                "paper_trades": store["paper"].get("trades", [])[-200:],
                "real_trades": store["real"].get("trades", [])[-200:],

                "notes": "src=Coinbase • 15m execution • 1h+4h bias • ATR SL/TP",
                "error": "",
            }

            store["last_state"] = state
            atomic_write_json(STATE_FILE, store)
            LATEST_STATE = state

        except Exception as e:
            LATEST_STATE = {"ok": False, "time": datetime.now().strftime("%H:%M:%S"), "iso": _now_iso(), "error": str(e)}

        await asyncio.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
