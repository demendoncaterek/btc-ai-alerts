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
# BTC ENGINE (alerts + paper trades + manual real trade logging)
# ============================================================

# -------------------------
# CONFIG (safe defaults)
# -------------------------
PRODUCT = os.getenv("PRODUCT", "BTC-USD")
COINBASE_BASE = "https://api.exchange.coinbase.com"

# "Longer-term" vs 1m: 5m base + 1h trend filter
BASE_GRANULARITY = int(os.getenv("BASE_GRANULARITY", "300"))   # 300 = 5m
TREND_GRANULARITY = int(os.getenv("TREND_GRANULARITY", "3600"))  # 3600 = 1h

BASE_LIMIT = int(os.getenv("BASE_LIMIT", "240"))      # 240 * 5m = 20h
TREND_LIMIT = int(os.getenv("TREND_LIMIT", "240"))    # 240 * 1h = 10d

STATE_FILE = os.getenv("STATE_FILE", "btc_state.json")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", "1800"))  # 30 min
MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "65"))

# "Crazy dip / peak" alerts (relative move in last window)
DIP_WINDOW_BARS = int(os.getenv("DIP_WINDOW_BARS", "36"))        # 36 * 5m ≈ 3h
DIP_PCT = float(os.getenv("DIP_PCT", "0.75")) / 100.0            # 0.75% by default
PEAK_WINDOW_BARS = int(os.getenv("PEAK_WINDOW_BARS", "36"))
PEAK_PCT = float(os.getenv("PEAK_PCT", "0.75")) / 100.0

# Heartbeat message (optional). Set HEARTBEAT_MINUTES=0 to disable
HEARTBEAT_MINUTES = int(os.getenv("HEARTBEAT_MINUTES", "120"))

# Paper trading
PAPER_START_USD = float(os.getenv("PAPER_START_USD", "250.0"))
PAPER_FEE_BPS = float(os.getenv("PAPER_FEE_BPS", "10")) / 10000.0  # 10 bps = 0.10%

# HTTP server
PORT = int(os.getenv("PORT", "8080"))

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
    """
    Coinbase Exchange candles endpoint (public).
    Returns candles ascending by time.
    """
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

    data.sort(key=lambda x: x[0])   # time asc
    data = data[-limit:]

    candles = []
    closes = []
    highs = []
    lows = []
    for c in data:
        ts = int(c[0])
        candles.append({
            "ts": ts,
            "time": datetime.fromtimestamp(ts).strftime("%H:%M"),
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
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - prev_close), np.abs(lows[1:] - prev_close)))
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

def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    if len(prices) < slow + signal:
        return (0.0, 0.0, 0.0)
    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)
    line = fast_ema - slow_ema
    sig = ema(line, signal)
    hist = line - sig
    return (float(line[-1]), float(sig[-1]), float(hist[-1]))

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
# Paper + real log store
# -------------------------
def load_store():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.loads(f.read() or "{}")
    except:
        return {}

def init_trading_state(store: dict):
    store.setdefault("paper", {"usd": PAPER_START_USD, "btc": 0.0, "avg_entry": 0.0, "trades": []})
    store.setdefault("real", {"trades": [], "position_btc": 0.0, "avg_entry": 0.0})
    store.setdefault("events", [])  # for UI markers (signals/dips/peaks)
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

def paper_sell(store: dict, btc_amount: float, price: float, reason: str):
    p = store["paper"]
    btc_amount = max(0.0, min(btc_amount, p["btc"]))
    if btc_amount <= 0:
        return False
    usd_gross = btc_amount * price
    fee = usd_gross * PAPER_FEE_BPS
    usd_net = usd_gross - fee
    p["btc"] -= btc_amount
    if p["btc"] == 0:
        p["avg_entry"] = 0.0
    p["usd"] += usd_net
    p["trades"].append({"ts": _now_iso(), "side": "SELL", "btc": btc_amount, "price": price, "fee": fee, "reason": reason})
    return True

def calc_paper_pnl(store: dict, price: float):
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

def calc_real_pnl(store: dict, price: float):
    r = store["real"]
    pos = r["position_btc"]
    avg = r["avg_entry"]
    unreal = (price - avg) * pos if pos > 0 else 0.0
    return {"position_btc": pos, "avg_entry": avg, "unrealized": unreal}

# -------------------------
# Signal logic
# -------------------------
def compute_confidence(*, base_rsi: float, macd_hist: float, zs: float, in_uptrend: bool, dip: bool, peak: bool) -> int:
    score = 50
    if base_rsi <= 30:
        score += 12
    if base_rsi >= 70:
        score += 12
    score += int(np.clip(abs(macd_hist) * 8000, 0, 15))
    score += int(np.clip(abs(zs) * 5, 0, 15))
    if in_uptrend:
        score += 5
    if dip or peak:
        score += 8
    return int(np.clip(score, 0, 100))

def decide_signal(base_prices: np.ndarray, base_highs: np.ndarray, base_lows: np.ndarray, trend_prices: np.ndarray):
    price = float(base_prices[-1])

    # Trend bias from 1h: EMA50 vs EMA200
    t_ema50 = ema(trend_prices, 50)
    t_ema200 = ema(trend_prices, 200) if len(trend_prices) >= 200 else np.full_like(trend_prices, np.nan, dtype=float)
    in_uptrend = bool(len(trend_prices) >= 200 and t_ema50[-1] > t_ema200[-1])
    in_downtrend = bool(len(trend_prices) >= 200 and t_ema50[-1] < t_ema200[-1])

    base_rsi = rsi(base_prices, 14)
    lower, mid, upper = bollinger(base_prices, 20, 2.0)
    a = atr(base_highs, base_lows, base_prices, 14)
    m_line, m_sig, m_hist = macd(base_prices, 12, 26, 9)
    zs = zscore(base_prices, 50)

    lookback = min(DIP_WINDOW_BARS, len(base_prices))
    window = base_prices[-lookback:]
    recent_high = float(np.max(window))
    recent_low = float(np.min(window))
    dip = (price <= recent_high * (1.0 - DIP_PCT))
    peak = (price >= recent_low * (1.0 + PEAK_PCT))

    signal = "WAIT"
    reason = "No setup"

    # Buy: dips in uptrend OR strong oversold mean reversion
    if (in_uptrend and (base_rsi <= 35 or dip) and price <= lower) or (base_rsi <= 28 and zs <= -1.2):
        signal = "BUY"
        reason = "Oversold dip (trend-filtered)" if in_uptrend else "Deep oversold mean reversion"

    # Sell: peaks in downtrend OR strong overbought mean reversion
    elif (in_downtrend and (base_rsi >= 65 or peak) and price >= upper) or (base_rsi >= 72 and zs >= 1.2):
        signal = "SELL"
        reason = "Overbought peak (trend-filtered)" if in_downtrend else "Deep overbought mean reversion"

    conf = compute_confidence(base_rsi=base_rsi, macd_hist=m_hist, zs=zs, in_uptrend=in_uptrend, dip=dip, peak=peak)

    return {
        "price": price,
        "base_rsi": round(base_rsi, 1),
        "bb_lower": None if np.isnan(lower) else round(lower, 2),
        "bb_mid": None if np.isnan(mid) else round(mid, 2),
        "bb_upper": None if np.isnan(upper) else round(upper, 2),
        "atr": round(a, 2),
        "macd_hist": round(m_hist, 5),
        "zscore": round(zs, 2),
        "trend_bias": "UP" if in_uptrend else ("DOWN" if in_downtrend else "NEUTRAL"),
        "dip": bool(dip),
        "peak": bool(peak),
        "signal": signal,
        "reason": reason,
        "confidence": conf,
    }

# -------------------------
# Telegram commands (polling)
# -------------------------
def build_status_text(state: dict) -> str:
    price = state.get("price", 0)
    rsi_v = state.get("base_rsi", 0)
    sig = state.get("signal", "WAIT")
    conf = state.get("confidence", 0)
    bias = state.get("trend_bias", "NEUTRAL")
    dip = "✅" if state.get("dip") else "—"
    peak = "✅" if state.get("peak") else "—"
    notes = state.get("reason", "")
    paper = state.get("paper_summary", {})
    real = state.get("real_summary", {})

    msg = (
        f"🧠 BTC Bot Status\n"
        f"Price: ${price:,.2f}\n"
        f"Signal: {sig} (conf {conf}%)\n"
        f"Trend bias (1h): {bias}\n"
        f"RSI({BASE_GRANULARITY//60}m): {rsi_v}\n"
        f"Dip(≈{DIP_WINDOW_BARS*BASE_GRANULARITY//60}m): {dip}   Peak: {peak}\n"
    )
    if notes:
        msg += f"Reason: {notes}\n"

    if paper:
        msg += (
            f"\n📄 Paper\n"
            f"Equity: ${paper.get('equity',0):,.2f}  P/L: ${paper.get('pnl',0):,.2f}\n"
            f"USD: ${paper.get('usd',0):,.2f}  BTC: {paper.get('btc',0):.6f}\n"
        )
    if real:
        msg += (
            f"\n🧾 Logged real\n"
            f"Pos BTC: {real.get('position_btc',0):.6f}\n"
            f"Avg entry: ${real.get('avg_entry',0):,.2f}\n"
            f"Unreal P/L: ${real.get('unrealized',0):,.2f}\n"
        )
    msg += "\nCommands: /status, /logbuy <usd> [price], /logsell <usd> [price]"
    return msg

def handle_command(text: str, latest_state: dict):
    if not text:
        return
    parts = text.strip().split()
    cmd = parts[0].lower()

    if cmd in ("/start", "/help"):
        send_telegram("✅ Bot is running.\nTry /status\nLog trades: /logbuy 100 or /logsell 100")
        return

    if cmd == "/status":
        send_telegram(build_status_text(latest_state))
        return

    if cmd in ("/logbuy", "/logsell"):
        if len(parts) < 2:
            send_telegram("Usage: /logbuy 100 [price] OR /logsell 100 [price]")
            return
        usd_amount = safe_float(parts[1], 0.0)
        price = safe_float(parts[2], latest_state.get("price", 0.0)) if len(parts) >= 3 else float(latest_state.get("price", 0.0))
        if usd_amount <= 0 or price <= 0:
            send_telegram("Invalid amount/price.")
            return
        qty_btc = usd_amount / price
        store = init_trading_state(load_store())
        ok = real_log(store, "BUY" if cmd == "/logbuy" else "SELL", qty_btc, price, note=f"via {cmd}")
        if ok:
            atomic_write_json(STATE_FILE, store)
            send_telegram(f"✅ Logged {cmd[4:].upper()} ${usd_amount:,.2f} (~{qty_btc:.6f} BTC) @ ${price:,.2f}")
        else:
            send_telegram("Could not log trade.")
        return

def telegram_poll_loop(get_latest_state_fn):
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
                text = msg.get("text", "")

                if TELEGRAM_CHAT_ID and str(TELEGRAM_CHAT_ID) != chat_id:
                    continue

                latest_state = get_latest_state_fn()
                handle_command(text, latest_state)

        except Exception:
            time.sleep(2)

# -------------------------
# HTTP endpoints for UI
# -------------------------
LATEST_STATE = {"ok": False, "error": "starting"}

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

    store = init_trading_state(load_store())
    atomic_write_json(STATE_FILE, store)

    def get_latest_state():
        return LATEST_STATE

    if TELEGRAM_BOT_TOKEN:
        threading.Thread(target=telegram_poll_loop, args=(get_latest_state,), daemon=True).start()

    if TELEGRAM_BOT_TOKEN and HEARTBEAT_MINUTES > 0:
        send_telegram("✅ BTC AI engine restarted and is running. Use /status anytime.")

    last_alert = 0.0
    last_dip_alert = 0.0
    last_peak_alert = 0.0
    last_heartbeat = 0.0

    print("✅ BTC Alert Engine Running (5m base • 1h trend filter)")

    while True:
        try:
            base_candles, base_prices, base_highs, base_lows = fetch_candles(BASE_GRANULARITY, BASE_LIMIT)
            _, trend_prices, _, _ = fetch_candles(TREND_GRANULARITY, TREND_LIMIT)

            s = decide_signal(base_prices, base_highs, base_lows, trend_prices)
            price = s["price"]

            store = init_trading_state(load_store())
            paper_summary = calc_paper_pnl(store, price)
            real_summary = calc_real_pnl(store, price)
            events = store.get("events", [])

            now = time.time()

            # Dip/Peak watches
            if s["dip"] and now - last_dip_alert > 1800:
                events.append({"ts": _now_iso(), "type": "DIP", "price": price, "label": "DIP"})
                last_dip_alert = now
                send_telegram(
                    f"🟦 BTC Dip Watch\nPrice: ${price:,.2f}\n"
                    f"Move: ≥{DIP_PCT*100:.2f}% down (≈{DIP_WINDOW_BARS*BASE_GRANULARITY//60}m window)\n"
                    f"Trend bias: {s['trend_bias']}"
                )

            if s["peak"] and now - last_peak_alert > 1800:
                events.append({"ts": _now_iso(), "type": "PEAK", "price": price, "label": "PEAK"})
                last_peak_alert = now
                send_telegram(
                    f"🟥 BTC Peak Watch\nPrice: ${price:,.2f}\n"
                    f"Move: ≥{PEAK_PCT*100:.2f}% up (≈{PEAK_WINDOW_BARS*BASE_GRANULARITY//60}m window)\n"
                    f"Trend bias: {s['trend_bias']}"
                )

            # High-confidence setup alert
            if s["signal"] in ("BUY", "SELL") and s["confidence"] >= MIN_CONFIDENCE and now - last_alert > ALERT_COOLDOWN:
                events.append({"ts": _now_iso(), "type": s["signal"], "price": price, "label": s["signal"]})
                last_alert = now
                send_telegram(
                    f"📢 BTC {s['signal']} (setup)\n"
                    f"Price: ${price:,.2f}\n"
                    f"Trend bias (1h): {s['trend_bias']}\n"
                    f"RSI({BASE_GRANULARITY//60}m): {s['base_rsi']}\n"
                    f"Confidence: {s['confidence']}%\n"
                    f"Reason: {s['reason']}"
                )

                # Optional: paper trade simulation (small + conservative)
                if PAPER_START_USD > 0:
                    p = store["paper"]
                    if s["signal"] == "BUY" and p["btc"] <= 0 and p["usd"] > 5:
                        paper_buy(store, usd_amount=min(50.0, p["usd"]), price=price, reason=s["reason"])
                    elif s["signal"] == "SELL" and p["btc"] > 0:
                        paper_sell(store, btc_amount=p["btc"], price=price, reason=s["reason"])

            # Heartbeat
            if TELEGRAM_BOT_TOKEN and HEARTBEAT_MINUTES > 0 and now - last_heartbeat > HEARTBEAT_MINUTES * 60:
                last_heartbeat = now
                send_telegram(f"🫀 Heartbeat\nPrice: ${price:,.2f}\nSignal: {s['signal']} (conf {s['confidence']}%)\nTrend: {s['trend_bias']}")

            if len(events) > 200:
                events = events[-200:]
            store["events"] = events

            state = {
                "ok": True,
                "time": datetime.now().strftime("%H:%M:%S"),
                "iso": _now_iso(),
                "product": PRODUCT,
                "price": round(price, 2),
                "signal": s["signal"],
                "reason": s["reason"],
                "confidence": s["confidence"],
                "trend_bias": s["trend_bias"],
                "base_granularity": BASE_GRANULARITY,
                "trend_granularity": TREND_GRANULARITY,
                "base_rsi": s["base_rsi"],
                "bb_lower": s["bb_lower"],
                "bb_mid": s["bb_mid"],
                "bb_upper": s["bb_upper"],
                "atr": s["atr"],
                "macd_hist": s["macd_hist"],
                "zscore": s["zscore"],
                "dip": s["dip"],
                "peak": s["peak"],
                "notes": f"src=Coinbase • 5m+1h • dip/peak window={DIP_WINDOW_BARS} bars",
                "candles": base_candles[-60:],  # last ~5h on 5m
                "events": store.get("events", []),
                "paper_summary": calc_paper_pnl(store, price),
                "real_summary": calc_real_pnl(store, price),
                "error": "",
            }

            store["last_state"] = state
            atomic_write_json(STATE_FILE, store)
            LATEST_STATE = state

        except Exception as e:
            LATEST_STATE = {"ok": False, "time": datetime.now().strftime("%H:%M:%S"), "iso": _now_iso(), "error": str(e)}

        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
