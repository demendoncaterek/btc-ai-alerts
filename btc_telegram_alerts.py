import os, json, time, threading, traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import requests
import numpy as np

VERSION = "btc-engine-stdlib-1.0"

PRODUCT_ID = os.getenv("PRODUCT_ID", "BTC-USD")
ENGINE_HOST = "0.0.0.0"
ENGINE_PORT = int(os.getenv("PORT", os.getenv("ENGINE_PORT", "8080")))

# Timeframes (Coinbase Exchange candles granularities must be one of these)
TF_EXEC_SEC   = int(os.getenv("TF_EXEC_SEC", "900"))    # 15m
TF_BIAS_1H_SEC = int(os.getenv("TF_BIAS_1H_SEC", "3600"))
TF_BIAS_6H_SEC = int(os.getenv("TF_BIAS_6H_SEC", "21600"))

POLL_SEC = float(os.getenv("POLL_SEC", "10"))
CANDLE_LIMIT = int(os.getenv("CANDLE_LIMIT", "300"))

PEAK_WINDOW_MIN = int(os.getenv("PEAK_WINDOW_MIN", "180"))
PEAK_NEAR_PCT = float(os.getenv("PEAK_NEAR_PCT", "0.25")) / 100.0
DIP_NEAR_PCT  = float(os.getenv("DIP_NEAR_PCT", "0.25")) / 100.0

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()  # needed for push alerts
TELEGRAM_POLL_SEC  = float(os.getenv("TELEGRAM_POLL_SEC", "2"))
TELEGRAM_STATE_PATH = os.getenv("TELEGRAM_STATE_PATH", "telegram_state.json")

DATA_PATH = os.getenv("DATA_PATH", "engine_data.json")

COINBASE_EXCHANGE_BASE = "https://api.exchange.coinbase.com"
ALLOWED_GRANULARITIES = {60, 300, 900, 3600, 21600, 86400}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def ema(arr, period):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return np.array([])
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(closes, period=14):
    closes = np.asarray(closes, dtype=float)
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        r = 100.0
    else:
        rs = avg_gain / avg_loss
        r = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            r = 100.0
        else:
            rs = avg_gain / avg_loss
            r = 100.0 - (100.0 / (1.0 + rs))

    return float(r)


def macd(closes, fast=12, slow=26, signal=9):
    closes = np.asarray(closes, dtype=float)
    if len(closes) < slow + signal + 1:
        return None
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    line = ema_fast - ema_slow
    sig = ema(line, signal)
    hist = line - sig
    return {
        "macd": float(line[-1]),
        "signal": float(sig[-1]),
        "hist": float(hist[-1]),
        "hist_prev": float(hist[-2]) if len(hist) >= 2 else None,
    }


def atr(highs, lows, closes, period=14):
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    if len(closes) < period + 1:
        return None
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))

    atr_val = np.mean(tr[1:period + 1])
    for i in range(period + 1, len(tr)):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
    return float(atr_val)


def fetch_candles(product_id, granularity, limit):
    if granularity not in ALLOWED_GRANULARITIES:
        raise ValueError(f"Unsupported granularity: {granularity}. Allowed: {sorted(ALLOWED_GRANULARITIES)}")

    url = f"{COINBASE_EXCHANGE_BASE}/products/{product_id}/candles"
    params = {"granularity": int(granularity)}
    headers = {"User-Agent": "btc-alert-engine/1.0"}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Coinbase HTTP {r.status_code}: {r.text[:200]}")

    data = r.json()
    if not isinstance(data, list) or (data and not isinstance(data[0], list)):
        raise RuntimeError(f"Unexpected candle payload: {str(data)[:200]}")

    # newest-first -> sort asc
    data = sorted(data, key=lambda x: x[0])[-limit:]
    out = []
    for row in data:
        t, low, high, open_, close, vol = row
        out.append({
            "time": datetime.fromtimestamp(int(t), tz=timezone.utc).isoformat(),
            "ts": int(t),
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(vol),
        })
    return out


def compute_bias(candles):
    if not candles or len(candles) < 60:
        return {"bias": "UNKNOWN", "strength": 0.0, "ema_fast": None, "ema_slow": None}

    closes = np.array([c["close"] for c in candles], dtype=float)
    fast_p = min(50, max(10, len(closes)//4))
    slow_p = min(200, max(30, len(closes)//2))

    e_fast = ema(closes, fast_p)
    e_slow = ema(closes, slow_p)
    bias = "UP" if e_fast[-1] > e_slow[-1] else "DOWN"
    strength = abs(e_fast[-1] - e_slow[-1]) / closes[-1]
    return {"bias": bias, "strength": float(clamp(strength, 0.0, 0.05)), "ema_fast": float(e_fast[-1]), "ema_slow": float(e_slow[-1])}


def make_marks(candles, macd_info):
    if not candles or macd_info is None:
        return []
    hist = macd_info.get("hist")
    hist_prev = macd_info.get("hist_prev")
    if hist is None or hist_prev is None:
        return []
    if hist_prev <= 0 and hist > 0:
        return [{"time": candles[-1]["time"], "type": "bull"}]
    if hist_prev >= 0 and hist < 0:
        return [{"time": candles[-1]["time"], "type": "bear"}]
    return []


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        return True
    except Exception:
        return False


class EngineState:
    def __init__(self):
        self.lock = threading.Lock()
        self.exec_candles = []
        self.bias1h_candles = []
        self.bias6h_candles = []
        self.last_update_iso = None

        persisted = load_json(DATA_PATH, {})
        self.paper = persisted.get("paper", {"cash_usd": 250.0, "pos_qty": 0.0, "avg_entry": 0.0, "trades": []})
        self.real  = persisted.get("real",  {"trades": [], "pos_qty": 0.0, "avg_entry": 0.0})

        self.alert_cooldown = {}
        self.state = {"ok": False, "error": "Starting…", "time": now_iso(), "product": PRODUCT_ID}

    def persist(self):
        save_json(DATA_PATH, {"paper": self.paper, "real": self.real})


ENGINE = EngineState()


def compute_paper_equity(price):
    cash = float(ENGINE.paper.get("cash_usd", 0.0))
    qty = float(ENGINE.paper.get("pos_qty", 0.0))
    return cash + (qty * price if price else 0.0)


def log_trade(kind, side, price, qty, note=""):
    trade = {"time": now_iso(), "side": side, "price": float(price), "qty": float(qty), "note": note}
    if kind == "paper":
        ENGINE.paper.setdefault("trades", []).append(trade)
    else:
        ENGINE.real.setdefault("trades", []).append(trade)
    ENGINE.persist()
    return trade


def paper_buy(price, usd=50.0):
    cash = float(ENGINE.paper.get("cash_usd", 0.0))
    usd = min(usd, cash)
    if usd <= 0:
        return None, "No paper cash available."
    qty = usd / price
    prev_qty = float(ENGINE.paper.get("pos_qty", 0.0))
    prev_avg = float(ENGINE.paper.get("avg_entry", 0.0))
    new_qty = prev_qty + qty
    new_avg = (prev_qty * prev_avg + qty * price) / new_qty if new_qty > 0 else 0.0
    ENGINE.paper["pos_qty"] = float(new_qty)
    ENGINE.paper["avg_entry"] = float(new_avg)
    ENGINE.paper["cash_usd"] = float(cash - usd)
    trade = log_trade("paper", "BUY", price, qty, note=f"usd={usd:.2f}")
    return trade, None


def paper_sell(price, qty=None):
    pos = float(ENGINE.paper.get("pos_qty", 0.0))
    if pos <= 0:
        return None, "No paper position to sell."
    if qty is None or qty <= 0 or qty > pos:
        qty = pos
    usd = qty * price
    ENGINE.paper["pos_qty"] = float(pos - qty)
    if ENGINE.paper["pos_qty"] <= 1e-12:
        ENGINE.paper["pos_qty"] = 0.0
        ENGINE.paper["avg_entry"] = 0.0
    ENGINE.paper["cash_usd"] = float(ENGINE.paper.get("cash_usd", 0.0) + usd)
    trade = log_trade("paper", "SELL", price, qty, note=f"usd={usd:.2f}")
    return trade, None


def send_telegram(text, chat_id=None):
    if not TELEGRAM_BOT_TOKEN:
        return False, "No TELEGRAM_BOT_TOKEN set"
    cid = chat_id or TELEGRAM_CHAT_ID
    if not cid:
        return False, "No TELEGRAM_CHAT_ID set (needed for push alerts)"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": cid, "text": text}
    try:
        r = requests.post(url, json=payload, timeout=10)
        ok = (r.status_code == 200)
        return ok, None if ok else r.text[:200]
    except Exception as e:
        return False, str(e)


def should_send(key, cooldown_sec):
    now = time.time()
    last = ENGINE.alert_cooldown.get(key, 0)
    if now - last >= cooldown_sec:
        ENGINE.alert_cooldown[key] = now
        return True
    return False


def format_status(s):
    price = s.get("price")
    sig = s.get("signal")
    conf = s.get("confidence")
    b1 = (s.get("bias_1h") or {}).get("bias")
    b6 = (s.get("bias_6h") or {}).get("bias")
    rsi_v = s.get("rsi_exec")

    lines = ["🧠 BTC Bot Status"]
    if price is not None:
        lines.append(f"Price: ${price:,.2f}")
    lines.append(f"Signal: {sig} (conf {conf}%)")
    lines.append(f"Bias (1h/6h): {b1}/{b6}")
    if rsi_v is not None:
        lines.append(f"RSI(15m): {rsi_v}")
    if s.get("dip_watch"):
        dw = s["dip_watch"]
        lines.append(f"Dip: near ${dw.get('near_low', 0):,.2f} ({dw.get('window_min')}m)")
    if s.get("peak_watch"):
        pw = s["peak_watch"]
        lines.append(f"Peak: near ${pw.get('near_high', 0):,.2f} ({pw.get('window_min')}m)")
    lines.append(f"Reason: {s.get('reason', '-')}")
    return "\n".join(lines)


def summarize_state(exec_candles, bias1h, bias6h):
    closes = [c["close"] for c in exec_candles] if exec_candles else []
    highs = [c["high"] for c in exec_candles] if exec_candles else []
    lows  = [c["low"]  for c in exec_candles] if exec_candles else []
    price = float(closes[-1]) if closes else None

    rsi_val = rsi(closes, 14) if closes else None
    macd_val = macd(closes) if closes else None
    atr_val = atr(highs, lows, closes, 14) if closes else None

    peak_watch = None
    dip_watch = None
    if exec_candles and price is not None:
        window_candles = max(5, int((PEAK_WINDOW_MIN * 60) / TF_EXEC_SEC))
        recent = exec_candles[-window_candles:]
        rc = [c["close"] for c in recent]
        hi = max(rc)
        lo = min(rc)
        if (hi - price) / hi <= PEAK_NEAR_PCT:
            peak_watch = {"near_high": float(hi), "window_min": PEAK_WINDOW_MIN}
        if (price - lo) / lo <= DIP_NEAR_PCT:
            dip_watch = {"near_low": float(lo), "window_min": PEAK_WINDOW_MIN}

    in_pos = float(ENGINE.paper.get("pos_qty", 0.0)) > 0.0
    bias_align_up = (bias1h["bias"] == "UP" and bias6h["bias"] == "UP")

    if price is None or macd_val is None:
        return {
            "signal": "WAIT",
            "confidence": 0.0,
            "reason": "Warming up candles/indicators…",
            "price": price, "rsi": rsi_val, "macd": macd_val, "atr": atr_val,
            "peak_watch": peak_watch, "dip_watch": dip_watch, "marks": []
        }

    hist = macd_val["hist"]
    hist_prev = macd_val.get("hist_prev")
    macd_cross_up = (hist_prev is not None and hist_prev <= 0 and hist > 0)
    macd_cross_dn = (hist_prev is not None and hist_prev >= 0 and hist < 0)

    conf = 0.0
    reasons = []

    if bias_align_up:
        conf += 0.35
        reasons.append("1h+6h bias aligned UP")
    else:
        reasons.append("bias mixed/unknown")

    if macd_cross_up:
        conf += 0.25
        reasons.append("MACD momentum flipped bullish")
    elif macd_cross_dn:
        conf += 0.15
        reasons.append("MACD momentum flipped bearish")

    if rsi_val is not None:
        if rsi_val < 30:
            conf += 0.10
            reasons.append("RSI oversold (<30)")
        elif rsi_val > 70:
            conf += 0.05
            reasons.append("RSI overbought (>70)")

    if dip_watch and bias_align_up:
        conf += 0.10
        reasons.append("Dip watch with UP bias")

    if peak_watch and in_pos:
        conf += 0.10
        reasons.append("Peak watch while in position")

    signal = "WAIT"
    if (not in_pos) and bias_align_up and (macd_cross_up or (rsi_val is not None and rsi_val < 35) or dip_watch):
        signal = "BUY"
    elif in_pos and (macd_cross_dn or (peak_watch and rsi_val is not None and rsi_val > 65)):
        signal = "SELL"
        conf = clamp(conf + 0.10, 0.0, 1.0)
    else:
        conf = clamp(conf * 0.6, 0.0, 1.0)

    return {
        "signal": signal,
        "confidence": float(round(clamp(conf, 0.0, 1.0) * 100.0, 2)),
        "reason": "; ".join(reasons) if reasons else "No setup",
        "price": price,
        "rsi": None if rsi_val is None else float(round(rsi_val, 2)),
        "macd": {k: float(v) for k, v in macd_val.items() if v is not None},
        "atr": None if atr_val is None else float(round(atr_val, 2)),
        "peak_watch": peak_watch,
        "dip_watch": dip_watch,
        "marks": make_marks(exec_candles, macd_val)
    }


def update_loop():
    while True:
        try:
            exec_c = fetch_candles(PRODUCT_ID, TF_EXEC_SEC, CANDLE_LIMIT)
            c1h = fetch_candles(PRODUCT_ID, TF_BIAS_1H_SEC, CANDLE_LIMIT)
            c6h = fetch_candles(PRODUCT_ID, TF_BIAS_6H_SEC, CANDLE_LIMIT)

            b1 = compute_bias(c1h)
            b6 = compute_bias(c6h)

            summary = summarize_state(exec_c, b1, b6)

            with ENGINE.lock:
                ENGINE.exec_candles = exec_c
                ENGINE.bias1h_candles = c1h
                ENGINE.bias6h_candles = c6h
                ENGINE.last_update_iso = now_iso()

                ENGINE.state = {
                    "ok": True,
                    "time": now_iso(),
                    "last_update": ENGINE.last_update_iso,
                    "product": PRODUCT_ID,
                    "src": "Coinbase",
                    "price": summary["price"],
                    "signal": summary["signal"],
                    "confidence": summary["confidence"],
                    "reason": summary["reason"],
                    "rsi_exec": summary["rsi"],
                    "macd_exec": summary["macd"],
                    "atr_exec": summary["atr"],
                    "bias_1h": b1,
                    "bias_6h": b6,
                    "peak_watch": summary["peak_watch"],
                    "dip_watch": summary["dip_watch"],
                    "marks": summary["marks"],
                    "paper": {
                        "cash_usd": ENGINE.paper.get("cash_usd", 0.0),
                        "pos_qty": ENGINE.paper.get("pos_qty", 0.0),
                        "avg_entry": ENGINE.paper.get("avg_entry", 0.0),
                        "equity": compute_paper_equity(summary["price"]),
                    },
                    "real": {
                        "pos_qty": ENGINE.real.get("pos_qty", 0.0),
                        "avg_entry": ENGINE.real.get("avg_entry", 0.0),
                    },
                }

            s = ENGINE.state
            if s.get("ok"):
                if s.get("peak_watch") and should_send("peak_watch", 300):
                    pw = s["peak_watch"]
                    send_telegram(
                        f"📈 BTC Peak Watch\nPrice: ${s['price']:,.2f}\nNear {pw['window_min']}m high: ${pw['near_high']:,.2f}\nBias: {s['bias_1h']['bias']}/{s['bias_6h']['bias']}"
                    )
                if s.get("dip_watch") and should_send("dip_watch", 300):
                    dw = s["dip_watch"]
                    send_telegram(
                        f"📉 BTC Dip Watch\nPrice: ${s['price']:,.2f}\nNear {dw['window_min']}m low: ${dw['near_low']:,.2f}\nBias: {s['bias_1h']['bias']}/{s['bias_6h']['bias']}"
                    )

                # Signal change
                prev = load_json("last_sig.json", {}).get("signal")
                if s.get("signal") != prev and should_send("sig_change", 120):
                    save_json("last_sig.json", {"signal": s.get("signal"), "t": now_iso()})
                    send_telegram(
                        f"⚡ Signal: {s['signal']} (conf {s['confidence']}%)\nPrice: ${s['price']:,.2f}\nReason: {s['reason']}"
                    )

        except Exception as e:
            with ENGINE.lock:
                ENGINE.state = {
                    "ok": False,
                    "time": now_iso(),
                    "product": PRODUCT_ID,
                    "error": str(e),
                    "trace": traceback.format_exc().splitlines()[-10:],
                }

        time.sleep(POLL_SEC)


def telegram_poll_loop():
    if not TELEGRAM_BOT_TOKEN:
        return

    offset = int(load_json(TELEGRAM_STATE_PATH, {}).get("offset", 0))

    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
            params = {"timeout": 20, "offset": offset}
            r = requests.get(url, params=params, timeout=30)
            if r.status_code != 200:
                time.sleep(TELEGRAM_POLL_SEC)
                continue

            payload = r.json()
            if not payload.get("ok"):
                time.sleep(TELEGRAM_POLL_SEC)
                continue

            for upd in payload.get("result", []):
                offset = max(offset, upd.get("update_id", 0) + 1)
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue

                chat_id = str((msg.get("chat") or {}).get("id", ""))
                text = (msg.get("text") or "").strip()

                with ENGINE.lock:
                    s = dict(ENGINE.state)

                reply = None
                if text.startswith("/status"):
                    reply = format_status(s)
                elif text.startswith("/explain"):
                    reply = "🧾 Explain\n" + format_status(s)
                elif text.startswith("/paperbuy") or text.startswith("/paper_buy"):
                    parts = text.split()
                    usd = safe_float(parts[1], 50.0) if len(parts) > 1 else 50.0
                    price = s.get("price")
                    if price:
                        _, err = paper_buy(price, usd=usd)
                        reply = f"🧪 Paper BUY @ ${price:,.2f} (usd {usd:.2f})" if not err else f"❌ {err}"
                    else:
                        reply = "❌ No price yet."
                elif text.startswith("/papersell") or text.startswith("/paper_sell"):
                    price = s.get("price")
                    if price:
                        _, err = paper_sell(price)
                        reply = f"🧪 Paper SELL @ ${price:,.2f}" if not err else f"❌ {err}"
                    else:
                        reply = "❌ No price yet."
                elif text.startswith("/logbuy"):
                    parts = text.split()
                    usd = safe_float(parts[1], 0.0) if len(parts) > 1 else 0.0
                    price = safe_float(parts[2], None) if len(parts) > 2 else s.get("price")
                    if price and usd > 0:
                        qty = usd / price
                        log_trade("real", "BUY", price, qty, note=f"usd={usd:.2f}")
                        reply = f"📝 Logged REAL BUY @ ${price:,.2f} (usd {usd:.2f})"
                    else:
                        reply = "Usage: /logbuy <usd> [price]"
                elif text.startswith("/logsell"):
                    parts = text.split()
                    qty = safe_float(parts[1], 0.0) if len(parts) > 1 else 0.0
                    price = safe_float(parts[2], None) if len(parts) > 2 else s.get("price")
                    if price and qty > 0:
                        log_trade("real", "SELL", price, qty, note="")
                        reply = f"📝 Logged REAL SELL @ ${price:,.2f} (qty {qty:.6f})"
                    else:
                        reply = "Usage: /logsell <qty> [price]"

                if reply:
                    send_telegram(reply, chat_id=chat_id)

            save_json(TELEGRAM_STATE_PATH, {"offset": offset})
        except Exception:
            pass

        time.sleep(TELEGRAM_POLL_SEC)


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/health":
            self._json(200, {"ok": True, "time": now_iso(), "version": VERSION, "product": PRODUCT_ID})
            return

        if path == "/state":
            with ENGINE.lock:
                self._json(200, dict(ENGINE.state))
            return

        if path == "/candles":
            qs = parse_qs(parsed.query or "")
            tf = (qs.get("tf", ["15m"])[0] or "15m").lower()

            with ENGINE.lock:
                if tf in ("exec", "15m", "15"):
                    candles = list(ENGINE.exec_candles)
                    tf_sec = TF_EXEC_SEC
                elif tf in ("1h", "60m"):
                    candles = list(ENGINE.bias1h_candles)
                    tf_sec = TF_BIAS_1H_SEC
                elif tf in ("6h", "360m"):
                    candles = list(ENGINE.bias6h_candles)
                    tf_sec = TF_BIAS_6H_SEC
                else:
                    self._json(400, {"ok": False, "error": "tf must be 15m|1h|6h"})
                    return

            self._json(200, {"ok": True, "tf": tf, "tf_sec": tf_sec, "candles": candles})
            return

        if path == "/trades":
            with ENGINE.lock:
                s = dict(ENGINE.state)
                paper = dict(ENGINE.paper)
                real = dict(ENGINE.real)
            price = s.get("price")
            paper_equity = compute_paper_equity(price) if price else paper.get("cash_usd", 0.0)
            self._json(200, {"ok": True, "paper": {**paper, "equity": paper_equity}, "real": real})
            return

        self._json(404, {"ok": False, "error": "Not found"})


def main():
    threading.Thread(target=update_loop, daemon=True).start()
    threading.Thread(target=telegram_poll_loop, daemon=True).start()
    server = ThreadingHTTPServer((ENGINE_HOST, ENGINE_PORT), Handler)
    print(f"[engine] listening on {ENGINE_HOST}:{ENGINE_PORT} product={PRODUCT_ID} poll={POLL_SEC}s")
    server.serve_forever()


if __name__ == "__main__":
    main()
