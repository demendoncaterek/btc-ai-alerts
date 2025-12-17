#!/usr/bin/env python3
"""
btc_telegram_alerts.py

BTC "AI" Engine (rules-based). Exposes a tiny HTTP API for the UI + sends Telegram alerts.

IMPORTANT:
- This is NOT financial advice.
- This engine does NOT "know all strategies". It only evaluates the indicators/filters coded here.
- Default mode is PAPER trading (no real orders). Logged "real" trades are manual entries via /logbuy /logsell.
"""

import os
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
from flask import Flask, jsonify, request


# -------------------------
# Config
# -------------------------

COINBASE_PRODUCT = os.getenv("COINBASE_PRODUCT", "BTC-USD")
COINBASE_BASE = os.getenv("COINBASE_BASE", "https://api.exchange.coinbase.com").rstrip("/")

# Base execution timeframe: typically "15m" for slower signals.
BASE_TF = os.getenv("BASE_TF", "15m").strip().lower()  # examples: "1m", "5m", "15m", "1h"
REFRESH_SECS = int(os.getenv("REFRESH_SECS", "15"))    # engine recalculates this often (lightweight)

# Candle history (Coinbase endpoint returns up to ~300 candles per request)
CANDLES_LIMIT = int(os.getenv("CANDLES_LIMIT", "250"))

# Trend/bias timeframes (resampled from BASE_TF)
BIAS_TF_1 = os.getenv("BIAS_TF_1", "1h").strip().lower()
BIAS_TF_2 = os.getenv("BIAS_TF_2", "4h").strip().lower()

# Alerting
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
ALERT_COOLDOWN_SECS = int(os.getenv("ALERT_COOLDOWN_SECS", "900"))  # 15m default

# Confidence thresholds (0..1)
BUY_CONF = float(os.getenv("BUY_CONF", "0.72"))
SELL_CONF = float(os.getenv("SELL_CONF", "0.72"))

# Risk tuning (ATR multiple)
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.5"))
ATR_TP_MULT = float(os.getenv("ATR_TP_MULT", "2.0"))

# "Watch" notifications (not trade signals)
WATCH_WINDOW_MIN = int(os.getenv("WATCH_WINDOW_MIN", "180"))
WATCH_EDGE_PCT = float(os.getenv("WATCH_EDGE_PCT", "0.25"))  # within 0.25% of window high/low triggers watch


# -------------------------
# Helpers: timeframes / indicators
# -------------------------

_ALLOWED_GRANULARITIES = {60, 300, 900, 3600, 21600, 86400}  # seconds (Coinbase Exchange candles)

def _tf_to_seconds(tf: str) -> int:
    """
    Parses tf strings like '15m', '1h', '60s'. Returns seconds.
    Also accepts bare numbers:
      - if not supported, but (n*60) is supported, we treat it as minutes.
      - otherwise snap to nearest supported value.
    """
    tf = str(tf).strip().lower()
    if tf.endswith("ms"):
        return max(1, int(float(tf[:-2]) / 1000))
    if tf.endswith("s"):
        return int(float(tf[:-1]))
    if tf.endswith("m"):
        return int(float(tf[:-1]) * 60)
    if tf.endswith("h"):
        return int(float(tf[:-1]) * 3600)
    if tf.endswith("d"):
        return int(float(tf[:-1]) * 86400)

    try:
        g = int(float(tf))
    except Exception:
        return 900

    if g in _ALLOWED_GRANULARITIES:
        return g

    if (g * 60) in _ALLOWED_GRANULARITIES:
        return g * 60

    return min(_ALLOWED_GRANULARITIES, key=lambda x: abs(x - g))

BASE_SEC = _tf_to_seconds(BASE_TF)

def _resample_factor(base_sec: int, target_tf: str) -> int:
    target_sec = _tf_to_seconds(target_tf)
    if target_sec <= 0:
        return 1
    return max(1, int(round(target_sec / base_sec)))

def ema(vals, n):
    if len(vals) < n:
        return None
    k = 2 / (n + 1)
    e = vals[0]
    for v in vals[1:]:
        e = v * k + e * (1 - k)
    return e

def rsi(closes, n=14):
    if len(closes) < n + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(-n, 0):
        d = closes[i] - closes[i - 1]
        if d >= 0:
            gains += d
        else:
            losses -= d
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def true_range(h, l, prev_c):
    return max(h - l, abs(h - prev_c), abs(l - prev_c))

def atr(highs, lows, closes, n=14):
    if len(closes) < n + 1:
        return None
    trs = []
    for i in range(-n, 0):
        trs.append(true_range(highs[i], lows[i], closes[i - 1]))
    return sum(trs) / n

def _chunk_ohlcv(candles, k):
    """Downsamples candles by factor k with OHLCV aggregation."""
    if k <= 1:
        return candles[:]
    out = []
    for i in range(0, len(candles) - k + 1, k):
        block = candles[i:i+k]
        out.append({
            "t": block[0]["t"],
            "o": block[0]["o"],
            "c": block[-1]["c"],
            "h": max(x["h"] for x in block),
            "l": min(x["l"] for x in block),
            "v": sum(x.get("v", 0.0) for x in block),
        })
    return out


# -------------------------
# Data fetching
# -------------------------

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "btc-ai-alerts/1.0"})

def fetch_candles(granularity_sec: int, limit: int):
    """
    Coinbase Exchange candles endpoint (public).
    Returns list of dicts sorted oldest->newest.
    """
    if granularity_sec not in _ALLOWED_GRANULARITIES:
        granularity_sec = min(_ALLOWED_GRANULARITIES, key=lambda x: abs(x - granularity_sec))

    url = f"{COINBASE_BASE}/products/{COINBASE_PRODUCT}/candles"
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=granularity_sec * min(limit, 300))
    params = {"granularity": granularity_sec, "start": start.isoformat(), "end": end.isoformat()}

    r = _SESSION.get(url, params=params, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Coinbase HTTP {r.status_code}: {r.text[:250]}")
    data = r.json()

    if isinstance(data, dict):
        raise RuntimeError(f"Coinbase API error: {data}")

    rows = []
    for t, low, high, open_, close, vol in data:
        rows.append({"t": int(t), "l": float(low), "h": float(high), "o": float(open_), "c": float(close), "v": float(vol)})
    rows.sort(key=lambda x: x["t"])
    return rows[-limit:]


# -------------------------
# Strategy / scoring (rules-based)
# -------------------------

@dataclass
class EngineState:
    ok: bool
    time: str
    iso: str
    src: str

    price: Optional[float] = None
    rsi_base: Optional[float] = None
    rsi_1h: Optional[float] = None
    rsi_4h: Optional[float] = None

    ema20: Optional[float] = None
    ema50: Optional[float] = None
    trend_1h: Optional[str] = None
    trend_4h: Optional[str] = None

    atr: Optional[float] = None
    signal: Optional[str] = None  # BUY/SELL/WAIT
    confidence: float = 0.0
    reason: str = ""

    dip_watch: Optional[dict] = None
    peak_watch: Optional[dict] = None

    candles: Optional[list] = None
    markers: Optional[list] = None

    paper: Optional[dict] = None
    real: Optional[dict] = None

    error: Optional[str] = None


paper_usd = float(os.getenv("PAPER_USD", "250"))
paper_btc = float(os.getenv("PAPER_BTC", "0"))
paper_entry = None
paper_trades = []

real_pos_btc = float(os.getenv("REAL_BTC", "0"))
real_avg_entry = None
real_trades = []

def _trend_label(candles, fast=20, slow=50):
    closes = [c["c"] for c in candles]
    ef = ema(closes[-max(len(closes), fast):], fast) if len(closes) >= fast else None
    es = ema(closes[-max(len(closes), slow):], slow) if len(closes) >= slow else None
    if ef is None or es is None:
        return None, None, "UNKNOWN"
    if ef > es:
        return ef, es, "UP"
    if ef < es:
        return ef, es, "DOWN"
    return ef, es, "FLAT"

def _watch_levels(candles, window_min: int):
    window_sec = window_min * 60
    newest_t = candles[-1]["t"]
    cutoff = newest_t - window_sec
    recent = [c for c in candles if c["t"] >= cutoff]
    if len(recent) < 5:
        return None, None
    return max(c["h"] for c in recent), min(c["l"] for c in recent)

def _score_signal(base_candles, c1h, c4h):
    closes = [c["c"] for c in base_candles]
    highs = [c["h"] for c in base_candles]
    lows  = [c["l"] for c in base_candles]
    price = closes[-1]

    rsi_base = rsi(closes, 14)
    rsi1 = rsi([x["c"] for x in c1h], 14) if c1h else None
    rsi4 = rsi([x["c"] for x in c4h], 14) if c4h else None

    ema20v = ema(closes[-max(len(closes), 20):], 20) if len(closes) >= 20 else None
    ema50v = ema(closes[-max(len(closes), 50):], 50) if len(closes) >= 50 else None
    atrv = atr(highs, lows, closes, 14)

    _, _, trend1 = _trend_label(c1h, 20, 50) if c1h else (None, None, "UNKNOWN")
    _, _, trend4 = _trend_label(c4h, 20, 50) if c4h else (None, None, "UNKNOWN")

    # Momentum markers
    markers = []
    if len(base_candles) >= 10:
        e9 = ema(closes[-20:], 9)
        for c in base_candles[-60:]:
            if e9 is None:
                break
            if c["c"] > e9 and c["c"] > c["o"]:
                markers.append({"t": c["t"], "price": c["c"], "kind": "bull"})
            elif c["c"] < e9 and c["c"] < c["o"]:
                markers.append({"t": c["t"], "price": c["c"], "kind": "bear"})

    score_buy = 0.0
    score_sell = 0.0
    reasons = []

    if trend4 == "UP":
        score_buy += 0.25; reasons.append("4h bias UP")
    elif trend4 == "DOWN":
        score_sell += 0.25; reasons.append("4h bias DOWN")

    if trend1 == "UP":
        score_buy += 0.15; reasons.append("1h bias UP")
    elif trend1 == "DOWN":
        score_sell += 0.15; reasons.append("1h bias DOWN")

    if rsi_base is not None:
        if rsi_base <= 30:
            score_buy += 0.25; reasons.append("RSI oversold")
        elif rsi_base >= 70:
            score_sell += 0.25; reasons.append("RSI overbought")
        if rsi_base >= 55:
            score_buy += 0.10
        if rsi_base <= 45:
            score_sell += 0.10

    if ema20v and ema50v:
        if price > ema20v > ema50v:
            score_buy += 0.15; reasons.append("Price above EMA20/50")
        if price < ema20v < ema50v:
            score_sell += 0.15; reasons.append("Price below EMA20/50")

    if atrv:
        score_buy += 0.05
        score_sell += 0.05

    buy_conf = max(0.0, min(1.0, score_buy))
    sell_conf = max(0.0, min(1.0, score_sell))

    if trend4 == "UP":
        sell_conf *= 0.6
    if trend4 == "DOWN":
        buy_conf *= 0.6

    signal = "WAIT"
    confidence = max(buy_conf, sell_conf)
    if buy_conf >= BUY_CONF and buy_conf > sell_conf:
        signal = "BUY"; confidence = buy_conf
    elif sell_conf >= SELL_CONF and sell_conf > buy_conf:
        signal = "SELL"; confidence = sell_conf

    reason = ", ".join(reasons) if reasons else "No setup"
    return signal, confidence, reason, markers, rsi_base, rsi1, rsi4, atrv, ema20v, ema50v, trend1, trend4

def _telegram_send(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        _SESSION.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}, timeout=10)
        return True
    except Exception:
        return False

_last_alert_ts = 0
_last_watch_ts = 0
_last_explain = "Not ready yet."

def build_state():
    global _last_explain
    now = datetime.now(timezone.utc)

    try:
        base = fetch_candles(BASE_SEC, CANDLES_LIMIT)
        price = base[-1]["c"]

        k1 = _resample_factor(BASE_SEC, BIAS_TF_1)
        k4 = _resample_factor(BASE_SEC, BIAS_TF_2)
        c1h = _chunk_ohlcv(base, k1)
        c4h = _chunk_ohlcv(base, k4)

        signal, conf, reason, markers, rsi_b, rsi1, rsi4, atrv, ema20v, ema50v, tr1, tr4 = _score_signal(base, c1h, c4h)

        win_hi, win_lo = _watch_levels(base, WATCH_WINDOW_MIN)
        dip_watch = peak_watch = None
        if win_hi and win_lo:
            if abs(price - win_lo) / win_lo * 100 <= WATCH_EDGE_PCT:
                dip_watch = {"window_min": WATCH_WINDOW_MIN, "low": win_lo, "price": price, "edge_pct": WATCH_EDGE_PCT}
            if abs(win_hi - price) / win_hi * 100 <= WATCH_EDGE_PCT:
                peak_watch = {"window_min": WATCH_WINDOW_MIN, "high": win_hi, "price": price, "edge_pct": WATCH_EDGE_PCT}

        sl = tp = None
        if atrv:
            if signal == "BUY":
                sl = price - ATR_SL_MULT * atrv
                tp = price + ATR_TP_MULT * atrv
            elif signal == "SELL":
                sl = price + ATR_SL_MULT * atrv
                tp = price - ATR_TP_MULT * atrv

        _last_explain = (
            f"Signal={signal} (conf={conf:.0%}) | price={price:.2f}\n"
            f"Bias: 1h={tr1}, 4h={tr4}\n"
            f"RSI: base={rsi_b if rsi_b is not None else 'n/a'} | 1h={rsi1 if rsi1 is not None else 'n/a'} | 4h={rsi4 if rsi4 is not None else 'n/a'}\n"
        )
        _last_explain += (f"ATR(base)={atrv:.2f}" if atrv else "ATR(base)=n/a")
        if sl and tp:
            _last_explain += f"\nSuggested SL={sl:.2f} TP={tp:.2f} (ATR x{ATR_SL_MULT}/{ATR_TP_MULT})"
        if reason:
            _last_explain += f"\nReason: {reason}"

        st = EngineState(
            ok=True,
            time=now.strftime("%H:%M:%S"),
            iso=now.isoformat(),
            src="Coinbase",
            price=price,
            rsi_base=rsi_b,
            rsi_1h=rsi1,
            rsi_4h=rsi4,
            ema20=ema20v,
            ema50=ema50v,
            trend_1h=tr1,
            trend_4h=tr4,
            atr=atrv,
            signal=signal,
            confidence=float(conf),
            reason=reason,
            dip_watch=dip_watch,
            peak_watch=peak_watch,
            candles=base[-120:],
            markers=markers,
            paper={
                "usd": paper_usd,
                "btc": paper_btc,
                "entry": paper_entry,
                "equity": (paper_usd + paper_btc * price),
                "trades": paper_trades[-100:],
            },
            real={
                "pos_btc": real_pos_btc,
                "avg_entry": real_avg_entry,
                "unreal_pl": (real_pos_btc * (price - real_avg_entry)) if (real_pos_btc and real_avg_entry) else 0.0,
                "trades": real_trades[-100:],
            },
        )
        return st, sl, tp

    except Exception as e:
        st = EngineState(ok=False, time=now.strftime("%H:%M:%S"), iso=now.isoformat(), src="Coinbase", error=str(e))
        return st, None, None

def _format_alert(state: EngineState, sl, tp):
    parts = [
        "🧠 BTC Alert",
        f"Price: ${state.price:,.2f}" if state.price is not None else "Price: n/a",
        f"Signal: {state.signal} (conf {state.confidence:.0%})",
        f"Bias: 1h={state.trend_1h} | 4h={state.trend_4h}",
        f"RSI(base)={state.rsi_base:.1f}" if state.rsi_base is not None else "RSI(base)=n/a",
    ]
    if state.atr is not None:
        parts.append(f"ATR(base): {state.atr:.2f}")
    if sl and tp:
        parts.append(f"SL: {sl:,.2f} | TP: {tp:,.2f}")
    if state.reason:
        parts.append(f"Reason: {state.reason}")
    return "\n".join(parts)

def _format_watch(state: EngineState):
    if state.peak_watch:
        w = state.peak_watch
        return f"📈 BTC Peak Watch\nPrice: ${w['price']:,.2f}\nNear {w['window_min']}m high: {w['high']:,.2f}\nBias: {state.trend_4h}"
    if state.dip_watch:
        w = state.dip_watch
        return f"📉 BTC Dip Watch\nPrice: ${w['price']:,.2f}\nNear {w['window_min']}m low: {w['low']:,.2f}\nBias: {state.trend_4h}"
    return None

def alerts_loop():
    global _last_alert_ts, _last_watch_ts
    while True:
        state, sl, tp = build_state()
        now = int(time.time())

        if state.ok and state.signal in ("BUY", "SELL"):
            if now - _last_alert_ts >= ALERT_COOLDOWN_SECS:
                _telegram_send(_format_alert(state, sl, tp))
                _last_alert_ts = now

        if state.ok and (state.peak_watch or state.dip_watch):
            if now - _last_watch_ts >= max(180, ALERT_COOLDOWN_SECS // 3):
                msg = _format_watch(state)
                if msg:
                    _telegram_send(msg)
                _last_watch_ts = now

        time.sleep(REFRESH_SECS)

app = Flask(__name__)
_state_cache = {"state": None, "ts": 0}

def get_state_cached():
    now = time.time()
    if _state_cache["state"] is None or (now - _state_cache["ts"]) > 10:
        s, sl, tp = build_state()
        d = asdict(s)
        d["suggested"] = {"sl": sl, "tp": tp}
        _state_cache["state"] = d
        _state_cache["ts"] = now
    return _state_cache["state"]

@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "btc-engine", "time": datetime.now(timezone.utc).isoformat()})

@app.get("/state")
def state():
    return jsonify(get_state_cached())

@app.get("/explain")
def explain():
    return jsonify({"ok": True, "explain": _last_explain})

@app.get("/trades")
def trades():
    return jsonify({"ok": True, "paper": paper_trades[-500:], "real": real_trades[-500:]})

@app.post("/paper/buy")
def paper_buy():
    global paper_usd, paper_btc, paper_entry
    d = get_state_cached()
    if not d.get("ok") or not d.get("price"):
        return jsonify({"ok": False, "error": "engine not ready"}), 400
    price = float(d["price"])
    if paper_usd <= 1:
        return jsonify({"ok": False, "error": "no paper USD available"}), 400
    qty = paper_usd / price
    paper_btc += qty
    paper_usd = 0.0
    paper_entry = price
    paper_trades.append({"ts": d["iso"], "side": "BUY", "price": price, "qty": qty, "mode": "paper"})
    return jsonify({"ok": True, "paper_usd": paper_usd, "paper_btc": paper_btc})

@app.post("/paper/sell")
def paper_sell():
    global paper_usd, paper_btc, paper_entry
    d = get_state_cached()
    if not d.get("ok") or not d.get("price"):
        return jsonify({"ok": False, "error": "engine not ready"}), 400
    price = float(d["price"])
    if paper_btc <= 0:
        return jsonify({"ok": False, "error": "no paper BTC to sell"}), 400
    qty = paper_btc
    paper_usd += qty * price
    paper_btc = 0.0
    paper_entry = None
    paper_trades.append({"ts": d["iso"], "side": "SELL", "price": price, "qty": qty, "mode": "paper"})
    return jsonify({"ok": True, "paper_usd": paper_usd, "paper_btc": paper_btc})

@app.post("/logbuy")
def logbuy():
    global real_pos_btc, real_avg_entry
    d = request.get_json(force=True, silent=True) or {}
    qty = float(d.get("qty", 0.0))
    price = float(d.get("price", 0.0))
    ts = d.get("ts") or datetime.now(timezone.utc).isoformat()
    if qty <= 0 or price <= 0:
        return jsonify({"ok": False, "error": "qty and price required"}), 400
    new_pos = real_pos_btc + qty
    if real_avg_entry is None:
        real_avg_entry = price
    else:
        real_avg_entry = (real_avg_entry * real_pos_btc + price * qty) / new_pos
    real_pos_btc = new_pos
    real_trades.append({"ts": ts, "side": "BUY", "price": price, "qty": qty, "mode": "real"})
    return jsonify({"ok": True})

@app.post("/logsell")
def logsell():
    global real_pos_btc, real_avg_entry
    d = request.get_json(force=True, silent=True) or {}
    qty = float(d.get("qty", 0.0))
    price = float(d.get("price", 0.0))
    ts = d.get("ts") or datetime.now(timezone.utc).isoformat()
    if qty <= 0 or price <= 0:
        return jsonify({"ok": False, "error": "qty and price required"}), 400
    if qty > real_pos_btc:
        qty = real_pos_btc
    real_pos_btc -= qty
    if real_pos_btc <= 0:
        real_pos_btc = 0.0
        real_avg_entry = None
    real_trades.append({"ts": ts, "side": "SELL", "price": price, "qty": qty, "mode": "real"})
    return jsonify({"ok": True})

def main():
    threading.Thread(target=alerts_loop, daemon=True).start()
    port = int(os.getenv("PORT", "8080"))
    print(f"✅ BTC Alert Engine Running ({BASE_TF} base • {BIAS_TF_1}+{BIAS_TF_2} bias • ATR SL/TP)")
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
