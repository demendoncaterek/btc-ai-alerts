import os, json, time, threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware

APP_VERSION = "2025-12-17"

# =========================
# Config (env vars)
# =========================
PRODUCT_ID = os.getenv("COINBASE_PRODUCT", "BTC-USD")

REFRESH_SEC = float(os.getenv("REFRESH_SEC", "10"))  # engine loop period

# Coinbase Exchange supports: 60, 300, 900, 3600, 21600, 86400
EXEC_GRAN_SEC = int(os.getenv("EXEC_GRAN_SEC", "900"))        # 15m execution
BIAS_1H_GRAN_SEC = int(os.getenv("BIAS_1H_GRAN_SEC", "3600"))  # 1h bias
BIAS_6H_GRAN_SEC = int(os.getenv("BIAS_6H_GRAN_SEC", "21600")) # 6h bias
RSI_5M_GRAN_SEC = int(os.getenv("RSI_5M_GRAN_SEC", "300"))     # 5m RSI confirm

CANDLES_LIMIT = int(os.getenv("CANDLES_LIMIT", "200"))

PEAK_WINDOW_MIN = int(os.getenv("PEAK_WINDOW_MIN", "180"))
PEAK_NEAR_PCT = float(os.getenv("PEAK_NEAR_PCT", "0.25"))

# Paper sim (educational)
PAPER_START_USD = float(os.getenv("PAPER_START_USD", "250"))
PAPER_MIN_CONF = float(os.getenv("PAPER_MIN_CONF", "65"))

# Confidence calibration (educational)
CAL_ENABLED = os.getenv("CAL_ENABLED", "1") == "1"
CAL_HORIZON_BARS = int(os.getenv("CAL_HORIZON_BARS", "4"))  # 4x15m = 1 hour
CAL_MAX_EVENTS = int(os.getenv("CAL_MAX_EVENTS", "2000"))

DATA_DIR = os.getenv("DATA_DIR", "/tmp")
PERSIST = os.getenv("PERSIST", "1") == "1"

# Telegram (optional)
ENABLE_TELEGRAM = os.getenv("ENABLE_TELEGRAM", "1") == "1"
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

COINBASE_EXCHANGE_BASE = os.getenv("COINBASE_EXCHANGE_BASE", "https://api.exchange.coinbase.com")

# =========================
# Indicators
# =========================
def ema(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(out[-1] + k * (v - out[-1]))
    return out

def rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(-period, 0):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))

def macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(closes) < slow + signal:
        return (None, None, None)
    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)
    macd_line = [f - s for f, s in zip(fast_ema[-len(slow_ema):], slow_ema)]
    signal_line = ema(macd_line, signal)
    hist = macd_line[-1] - signal_line[-1]
    return (macd_line[-1], signal_line[-1], hist)

def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1 or len(highs) < period + 1 or len(lows) < period + 1:
        return None
    trs: List[float] = []
    for i in range(-period, 0):
        hi, lo = highs[i], lows[i]
        prev_close = closes[i - 1]
        tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        trs.append(tr)
    return sum(trs) / len(trs)

# =========================
# Candles fetch
# =========================
def _iso(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

def fetch_candles(gran_sec: int, limit: int = CANDLES_LIMIT) -> List[Dict[str, Any]]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=gran_sec * limit)
    url = f"{COINBASE_EXCHANGE_BASE}/products/{PRODUCT_ID}/candles"
    params = {"start": _iso(start), "end": _iso(end), "granularity": str(gran_sec)}
    headers = {"User-Agent": "btc-ai-dashboard/1.0"}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Coinbase API {r.status_code}: {r.text[:200]}")
    data = r.json()

    out: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, list) or len(row) < 6:
            continue
        t, low, high, open_, close, vol = row[:6]
        dt = datetime.fromtimestamp(int(t), tz=timezone.utc)
        out.append(
            {"time": dt.isoformat(), "open": float(open_), "high": float(high), "low": float(low), "close": float(close), "volume": float(vol)}
        )
    out.sort(key=lambda x: x["time"])
    return out

# =========================
# State + persistence
# =========================
_state_lock = threading.Lock()
STATE: Dict[str, Any] = {
    "ok": False,
    "iso": None,
    "product": PRODUCT_ID,
    "price": None,
    "signal": "WAIT",
    "confidence": 0.0,
    "trend": "UNKNOWN",
    "bias_1h": "UNKNOWN",
    "bias_6h": "UNKNOWN",
    "rsi_5m": None,
    "rsi_15m": None,
    "macd_15m": None,
    "macd_signal_15m": None,
    "macd_hist_15m": None,
    "atr_15m": None,
    "sl": None,
    "tp": None,
    "peak_watch": False,
    "dip_watch": False,
    "peak_180m": None,
    "dip_180m": None,
    "reason": "",
    "source": "Coinbase Exchange",
    "version": APP_VERSION,
}

PAPER = {"usd": PAPER_START_USD, "btc": 0.0, "equity": PAPER_START_USD, "pos_entry": None, "pos_side": None, "realized_pnl": 0.0}
paper_trades: List[Dict[str, Any]] = []
real_trades: List[Dict[str, Any]] = []
signal_events: List[Dict[str, Any]] = []  # calibration events

def _persist_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)

def load_persisted():
    if not PERSIST:
        return
    for name, target in [("paper_trades.json", paper_trades), ("real_trades.json", real_trades), ("signal_events.json", signal_events), ("paper_state.json", None)]:
        path = _persist_path(name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if name == "paper_state.json" and isinstance(data, dict):
                PAPER.update(data)
            elif isinstance(data, list):
                target.clear()
                target.extend(data)
        except Exception:
            pass

def persist_all():
    if not PERSIST:
        return
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(_persist_path("paper_trades.json"), "w", encoding="utf-8") as f:
            json.dump(paper_trades[-2000:], f)
        with open(_persist_path("real_trades.json"), "w", encoding="utf-8") as f:
            json.dump(real_trades[-2000:], f)
        with open(_persist_path("signal_events.json"), "w", encoding="utf-8") as f:
            json.dump(signal_events[-CAL_MAX_EVENTS:], f)
        with open(_persist_path("paper_state.json"), "w", encoding="utf-8") as f:
            json.dump(PAPER, f)
    except Exception:
        pass

# =========================
# Decision logic
# =========================
def compute_bias(candles: List[Dict[str, Any]], fast: int = 20, slow: int = 50) -> str:
    closes = [c["close"] for c in candles]
    if len(closes) < slow + 5:
        return "UNKNOWN"
    e_fast = ema(closes, fast)[-1]
    e_slow = ema(closes, slow)[-1]
    if e_fast > e_slow:
        return "UP"
    if e_fast < e_slow:
        return "DOWN"
    return "FLAT"

def near_pct(a: float, b: float) -> float:
    if b == 0:
        return 999.0
    return abs(a - b) / b * 100.0

def decide(exec_c: List[Dict[str, Any]], bias1: str, bias6: str, rsi5: Optional[float]) -> Dict[str, Any]:
    closes = [c["close"] for c in exec_c]
    highs = [c["high"] for c in exec_c]
    lows = [c["low"] for c in exec_c]
    price = closes[-1] if closes else None

    rsi15 = rsi(closes, 14)
    m_line, m_sig, m_hist = macd(closes)
    a15 = atr(highs, lows, closes)

    if bias1 == "UP" and bias6 == "UP":
        trend = "UP"
    elif bias1 == "DOWN" and bias6 == "DOWN":
        trend = "DOWN"
    elif bias1 == "UNKNOWN" or bias6 == "UNKNOWN":
        trend = "UNKNOWN"
    else:
        trend = "MIXED"

    reasons: List[str] = []
    score = 0.0

    if trend == "UP":
        score += 30; reasons.append("Higher-timeframe bias bullish (1h & 6h).")
    elif trend == "DOWN":
        score += 30; reasons.append("Higher-timeframe bias bearish (1h & 6h).")
    elif trend == "MIXED":
        score += 10; reasons.append("Higher-timeframe bias mixed.")
    else:
        reasons.append("Higher-timeframe bias not ready.")

    if rsi15 is not None:
        if rsi15 <= 30:
            score += 25; reasons.append(f"RSI(15m) oversold ({rsi15:.1f}).")
        elif rsi15 >= 70:
            score += 25; reasons.append(f"RSI(15m) overbought ({rsi15:.1f}).")
        else:
            score += 10; reasons.append(f"RSI(15m) neutral ({rsi15:.1f}).")
    else:
        reasons.append("RSI(15m) not enough data yet.")

    if rsi5 is not None:
        score += 10 if (rsi5 <= 30 or rsi5 >= 70) else 5
        reasons.append(f"RSI(5m) {'extreme' if (rsi5 <= 30 or rsi5 >= 70) else 'normal'} ({rsi5:.1f}).")

    if m_hist is not None:
        score += 15
        reasons.append("MACD histogram " + ("positive (bullish momentum)." if m_hist > 0 else "negative (bearish momentum)."))
    else:
        reasons.append("MACD not enough data yet.")

    signal = "WAIT"
    if price is not None and rsi15 is not None and m_hist is not None:
        if trend == "UP" and rsi15 <= 35 and m_hist >= 0:
            signal = "BUY"; reasons.append("Setup: uptrend + pullback + momentum recovering.")
        elif trend == "DOWN" and rsi15 >= 65 and m_hist <= 0:
            signal = "SELL"; reasons.append("Setup: downtrend + bounce + momentum fading.")
        elif rsi15 <= 25 and m_hist >= 0:
            signal = "BUY"; reasons.append("Setup: deep oversold + momentum turning.")
        elif rsi15 >= 75 and m_hist <= 0:
            signal = "SELL"; reasons.append("Setup: deep overbought + momentum turning.")
        else:
            reasons.append("No high-probability setup yet.")

    confidence = max(0.0, min(99.0, score))

    sl = tp = None
    if price is not None and a15 is not None:
        if signal == "BUY":
            sl = price - a15
            tp = price + 2 * a15
        elif signal == "SELL":
            sl = price + a15
            tp = price - 2 * a15

    return {
        "price": price,
        "trend": trend,
        "bias_1h": bias1,
        "bias_6h": bias6,
        "rsi_5m": rsi5,
        "rsi_15m": rsi15,
        "macd_15m": m_line,
        "macd_signal_15m": m_sig,
        "macd_hist_15m": m_hist,
        "atr_15m": a15,
        "signal": signal,
        "confidence": confidence,
        "sl": sl,
        "tp": tp,
        "reason": " ".join(reasons),
    }

# =========================
# Paper sim (educational)
# =========================
def update_paper(price: Optional[float], signal: str, confidence: float):
    if price is None:
        return
    PAPER["equity"] = PAPER["usd"] + PAPER["btc"] * price
    if confidence < PAPER_MIN_CONF:
        return

    if signal == "BUY" and PAPER["btc"] == 0.0 and PAPER["usd"] > 1.0:
        usd_to_use = PAPER["usd"] * 0.95
        btc = usd_to_use / price
        PAPER["usd"] -= usd_to_use
        PAPER["btc"] += btc
        PAPER["pos_entry"] = price
        PAPER["pos_side"] = "LONG"
        paper_trades.append({"ts": datetime.now(timezone.utc).isoformat(), "side": "BUY", "price": price, "btc": btc, "usd": usd_to_use, "note": f"auto paper (conf {confidence:.0f}%)"})
    elif signal == "SELL" and PAPER["btc"] > 0.0:
        btc = PAPER["btc"]
        usd_out = btc * price
        PAPER["usd"] += usd_out
        PAPER["btc"] = 0.0
        entry = PAPER.get("pos_entry") or price
        pnl = (price - entry) * btc
        PAPER["realized_pnl"] += pnl
        PAPER["pos_entry"] = None
        PAPER["pos_side"] = None
        paper_trades.append({"ts": datetime.now(timezone.utc).isoformat(), "side": "SELL", "price": price, "btc": btc, "usd": usd_out, "pnl": pnl, "note": f"auto paper (conf {confidence:.0f}%)"})

# =========================
# Calibration (educational)
# =========================
def _conf_bucket(conf: float) -> int:
    c = int(max(0, min(99, conf)))
    return c // 10

def record_signal_event(now: datetime, signal: str, confidence: float, price: float):
    if not CAL_ENABLED or signal not in ("BUY", "SELL"):
        return
    resolve_at = now + timedelta(seconds=EXEC_GRAN_SEC * CAL_HORIZON_BARS)
    signal_events.append({"ts": now.isoformat(), "signal": signal, "confidence": float(confidence), "entry_price": float(price), "resolve_at_iso": resolve_at.isoformat(), "resolved": False})
    if len(signal_events) > CAL_MAX_EVENTS:
        del signal_events[: len(signal_events) - CAL_MAX_EVENTS]

def resolve_events(now: datetime, current_price: Optional[float]):
    if not CAL_ENABLED or current_price is None:
        return
    for ev in signal_events:
        if ev.get("resolved"):
            continue
        ra = ev.get("resolve_at_iso")
        if not ra:
            continue
        try:
            resolve_at = datetime.fromisoformat(ra)
        except Exception:
            continue
        if now < resolve_at:
            continue
        entry = float(ev.get("entry_price") or 0.0)
        sig = ev.get("signal")
        ret = (current_price - entry) / entry * 100.0 if entry else 0.0
        win = (current_price > entry) if sig == "BUY" else (current_price < entry)
        ev.update({"resolved": True, "exit_price": float(current_price), "ret_pct": float(ret), "win": bool(win), "resolved_at": now.isoformat()})

def calibration_summary(current_conf: float) -> Dict[str, Any]:
    if not CAL_ENABLED:
        return {"enabled": False}
    buckets = [{"bucket": i, "wins": 0, "total": 0} for i in range(10)]
    for ev in signal_events:
        if not ev.get("resolved"):
            continue
        b = _conf_bucket(float(ev.get("confidence") or 0.0))
        buckets[b]["total"] += 1
        if ev.get("win"):
            buckets[b]["wins"] += 1
    out = []
    for b in buckets:
        total = b["total"]; wins = b["wins"]
        out.append({"range": f"{b['bucket']*10}-{b['bucket']*10+9}", "wins": wins, "total": total, "win_rate": (wins / total) if total else None})
    cb = _conf_bucket(current_conf)
    cur = out[cb]
    return {"enabled": True, "horizon_bars": CAL_HORIZON_BARS, "bar_seconds": EXEC_GRAN_SEC, "buckets": out, "current_bucket": cur["range"], "current_bucket_win_rate": cur["win_rate"], "samples_total": sum(x["total"] for x in out)}

# =========================
# Peak/Dip watch
# =========================
def compute_peak_dip_watch(candles_5m: List[Dict[str, Any]], price: Optional[float]) -> Tuple[bool, bool, Optional[float], Optional[float]]:
    if price is None or not candles_5m:
        return (False, False, None, None)
    window_bars = max(2, int((PEAK_WINDOW_MIN * 60) / RSI_5M_GRAN_SEC))
    closes = [c["close"] for c in candles_5m][-window_bars:]
    if not closes:
        return (False, False, None, None)
    hi = max(closes); lo = min(closes)
    peak = near_pct(price, hi) <= PEAK_NEAR_PCT
    dip = near_pct(price, lo) <= PEAK_NEAR_PCT
    return peak, dip, hi, lo

# =========================
# Telegram (no extra deps)
# =========================
_last_update_id = 0

def tg_send(text: str, chat_id: Optional[str] = None):
    if not (TG_TOKEN and ENABLE_TELEGRAM):
        return
    cid = chat_id or TG_CHAT_ID
    if not cid:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", json={"chat_id": cid, "text": text}, timeout=10)
    except Exception:
        pass

def format_status() -> str:
    with _state_lock:
        s = dict(STATE)
    price = s.get("price"); conf = s.get("confidence"); sig = s.get("signal")
    r5 = s.get("rsi_5m"); r15 = s.get("rsi_15m")
    trend = s.get("trend"); b1 = s.get("bias_1h"); b6 = s.get("bias_6h")
    reason = s.get("reason", "")
    lines = [
        "🧠 BTC Bot Status",
        f"Price: ${price:,.2f}" if isinstance(price, (int, float)) else "Price: —",
        f"Signal: {sig} (conf {conf:.0f}%)" if isinstance(conf, (int, float)) else f"Signal: {sig}",
        f"Trend: {trend} (1h:{b1}, 6h:{b6})",
        f"RSI(5m): {r5:.1f}" if isinstance(r5, (int, float)) else "RSI(5m): —",
        f"RSI(15m): {r15:.1f}" if isinstance(r15, (int, float)) else "RSI(15m): —",
    ]
    if reason:
        lines.append(f"Reason: {reason[:350]}")
    return "\n".join(lines)

def telegram_poll_loop():
    global _last_update_id
    while True:
        if not (TG_TOKEN and ENABLE_TELEGRAM):
            time.sleep(5); continue
        try:
            r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates", params={"timeout": 10, "offset": _last_update_id + 1}, timeout=20)
            data = r.json()
            if not data.get("ok"):
                time.sleep(2); continue
            for upd in data.get("result", []):
                _last_update_id = max(_last_update_id, upd.get("update_id", 0))
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                chat_id = str(msg["chat"]["id"])
                text = (msg.get("text") or "").strip()
                if text.startswith("/start") or text.startswith("/help"):
                    tg_send("Commands: /status, /health, /explain", chat_id)
                elif text.startswith("/health"):
                    with _state_lock:
                        ok = bool(STATE.get("ok"))
                    tg_send(f"✅ Engine ok: {ok}", chat_id)
                elif text.startswith("/status"):
                    tg_send(format_status(), chat_id)
                elif text.startswith("/explain"):
                    with _state_lock:
                        reason = STATE.get("reason", "")
                    tg_send(reason or "No reason yet.", chat_id)
        except Exception:
            pass
        time.sleep(1)

# =========================
# Engine loop
# =========================
def engine_loop():
    load_persisted()
    prev_sig = "WAIT"
    prev_peak = prev_dip = False

    while True:
        start_t = time.time()
        try:
            exec_c = fetch_candles(EXEC_GRAN_SEC, CANDLES_LIMIT)
            bias1_c = fetch_candles(BIAS_1H_GRAN_SEC, max(120, CANDLES_LIMIT // 2))
            bias6_c = fetch_candles(BIAS_6H_GRAN_SEC, max(120, CANDLES_LIMIT // 2))
            rsi5_c = fetch_candles(RSI_5M_GRAN_SEC, max(220, CANDLES_LIMIT))

            bias1 = compute_bias(bias1_c)
            bias6 = compute_bias(bias6_c)
            rsi5_val = rsi([c["close"] for c in rsi5_c], 14)

            d = decide(exec_c, bias1, bias6, rsi5_val)
            peak, dip, hi, lo = compute_peak_dip_watch(rsi5_c, d.get("price"))

            now = datetime.now(timezone.utc)

            resolve_events(now, d.get("price"))
            if d.get("signal") != prev_sig and d.get("price") is not None:
                record_signal_event(now, d.get("signal"), float(d.get("confidence") or 0), float(d.get("price")))

            cal = calibration_summary(float(d.get("confidence") or 0.0))

            with _state_lock:
                STATE.update({
                    "ok": True,
                    "iso": now.isoformat(),
                    **d,
                    "peak_watch": peak,
                    "dip_watch": dip,
                    "peak_180m": hi,
                    "dip_180m": lo,
                    "candles_15m": exec_c[-200:],
                    "calibration": cal,
                })

            update_paper(d.get("price"), d.get("signal", "WAIT"), float(d.get("confidence") or 0))
            persist_all()

            # optional alerts (signal + peak/dip)
            if TG_TOKEN and ENABLE_TELEGRAM and TG_CHAT_ID:
                if d.get("signal") != prev_sig:
                    tg_send(f"🔔 Signal: {d.get('signal')} (conf {d.get('confidence'):.0f}%)\nPrice: ${d.get('price'):,.2f}\n{(d.get('reason') or '')[:300]}")
                if peak and not prev_peak:
                    tg_send(f"📈 Peak Watch • Price: ${d.get('price'):,.2f} • Near {PEAK_WINDOW_MIN}m high: {hi}")
                if dip and not prev_dip:
                    tg_send(f"📉 Dip Watch • Price: ${d.get('price'):,.2f} • Near {PEAK_WINDOW_MIN}m low: {lo}")

            prev_sig = d.get("signal", prev_sig)
            prev_peak, prev_dip = peak, dip

        except Exception as e:
            now = datetime.now(timezone.utc)
            with _state_lock:
                STATE.update({"ok": False, "iso": now.isoformat(), "error": str(e)})

        elapsed = time.time() - start_t
        time.sleep(max(0.5, REFRESH_SEC - elapsed))

# =========================
# FastAPI app
# =========================
app = FastAPI(title="BTC Engine", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    threading.Thread(target=engine_loop, daemon=True).start()
    if TG_TOKEN and ENABLE_TELEGRAM:
        threading.Thread(target=telegram_poll_loop, daemon=True).start()

@app.get("/health")
def health():
    with _state_lock:
        ok = bool(STATE.get("ok"))
        err = STATE.get("error")
    return {"ok": ok, "service": "btc-engine", "version": APP_VERSION, "product": PRODUCT_ID, "time": datetime.now(timezone.utc).isoformat(), "error": err}

@app.get("/state")
def state():
    with _state_lock:
        s = dict(STATE)
    price = s.get("price") if isinstance(s.get("price"), (int, float)) else None
    PAPER["equity"] = PAPER["usd"] + PAPER["btc"] * price if price else PAPER["usd"]
    s["paper"] = dict(PAPER)
    s["paper_trades_count"] = len(paper_trades)
    s["real_trades_count"] = len(real_trades)
    return s

@app.get("/candles")
def candles(tf: str = "15m", limit: int = 200):
    tf_map = {"5m": RSI_5M_GRAN_SEC, "15m": EXEC_GRAN_SEC, "1h": BIAS_1H_GRAN_SEC, "6h": BIAS_6H_GRAN_SEC}
    gran = tf_map.get(tf, EXEC_GRAN_SEC)
    data = fetch_candles(gran, max(10, min(500, int(limit))))
    return {"ok": True, "tf": tf, "gran_sec": gran, "candles": data}

@app.get("/trades/paper")
def get_paper_trades(limit: int = 500):
    return {"ok": True, "paper": dict(PAPER), "trades": paper_trades[-max(1, min(5000, int(limit))):]}

@app.get("/trades/real")
def get_real_trades(limit: int = 500):
    return {"ok": True, "trades": real_trades[-max(1, min(5000, int(limit))):]}

@app.post("/trades/real")
def log_real_trade(payload: Dict[str, Any] = Body(...)):
    trade = {"ts": payload.get("ts") or datetime.now(timezone.utc).isoformat(), "side": str(payload.get("side", "")).upper(), "qty": float(payload.get("qty") or 0), "price": float(payload.get("price") or 0), "note": payload.get("note") or ""}
    real_trades.append(trade)
    persist_all()
    return {"ok": True, "trade": trade, "count": len(real_trades)}

@app.post("/paper/reset")
def reset_paper():
    PAPER.update({"usd": PAPER_START_USD, "btc": 0.0, "equity": PAPER_START_USD, "pos_entry": None, "pos_side": None, "realized_pnl": 0.0})
    paper_trades.clear()
    persist_all()
    return {"ok": True, "paper": dict(PAPER)}
