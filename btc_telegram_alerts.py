"""
btc_engine.py
Rules-based BTC market monitor with:
- Coinbase Exchange candles (15m + 1h + 6h bias by default)
- Indicators: EMA, RSI, MACD, ATR
- Multi-timeframe bias + signal + confidence scoring
- Peak/Dip watch
- Paper trade simulator + manual "real trade" logging
- HTTP API: /health /state /trades /explain /paper/execute /real/log

No FastAPI/uvicorn required (stdlib http.server) -> fewer deploy crashes.
"""

import os
import json
import time
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import requests


# -------------------- Config --------------------
HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8080"))

PRODUCT = os.getenv("PRODUCT", "BTC-USD")
CB_BASE = os.getenv("COINBASE_EXCHANGE_URL", "https://api.exchange.coinbase.com").rstrip("/")
USER_AGENT = os.getenv("CB_USER_AGENT", "btc-engine/1.0")

BASE_GRAN = int(os.getenv("BASE_GRANULARITY", "900"))          # 15m
BIAS_1H_GRAN = int(os.getenv("BIAS_1H_GRANULARITY", "3600"))   # 1h
BIAS_6H_GRAN = int(os.getenv("BIAS_6H_GRANULARITY", "21600"))  # 6h (Coinbase supports 6h, not 4h)

LIMIT_BASE = int(os.getenv("LIMIT_BASE", "180"))
LIMIT_BIAS = int(os.getenv("LIMIT_BIAS", "200"))

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "10"))  # update cadence

PEAK_WINDOW_MIN = int(os.getenv("PEAK_WINDOW_MIN", "180"))
PEAK_THRESHOLD_PCT = float(os.getenv("PEAK_THRESHOLD_PCT", "0.75"))

ATR_LEN = int(os.getenv("ATR_LEN", "14"))
RSI_LEN = int(os.getenv("RSI_LEN", "14"))

SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.5"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "2.5"))

PAPER_START_USD = float(os.getenv("PAPER_START_USD", "250"))
PAPER_AUTO_TRADE = os.getenv("PAPER_AUTO_TRADE", "false").lower() in ("1", "true", "yes")

DATA_DIR = os.getenv("DATA_DIR", "./data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "trades.db")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


SUPPORTED_GRANS = {60, 300, 900, 3600, 21600, 86400}


# -------------------- Utils --------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=6)
    except Exception:
        pass


# -------------------- Indicators --------------------
def ema(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def rsi(values: List[float], period: int) -> List[Optional[float]]:
    if len(values) < period + 1:
        return [None] * len(values)

    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period

    out: List[Optional[float]] = [None] * period
    out.append(100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss))))

    for i in range(period + 1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out.append(100.0)
        else:
            rs = avg_gain / avg_loss
            out.append(100.0 - (100.0 / (1.0 + rs)))
    return out


def atr(highs: List[float], lows: List[float], closes: List[float], period: int) -> List[Optional[float]]:
    n = len(closes)
    if n == 0:
        return []
    trs: List[float] = []
    for i in range(n):
        if i == 0:
            trs.append(highs[i] - lows[i])
        else:
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            trs.append(tr)

    if n < period:
        return [None] * n

    out: List[Optional[float]] = [None] * (period - 1)
    first = sum(trs[:period]) / period
    out.append(first)
    prev = first
    for i in range(period, n):
        prev = (prev * (period - 1) + trs[i]) / period
        out.append(prev)
    return out


def macd(values: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
    if not values:
        return [], [], []
    ef = ema(values, fast)
    es = ema(values, slow)
    line = [a - b for a, b in zip(ef, es)]
    sig = ema(line, signal)
    hist = [a - b for a, b in zip(line, sig)]
    return line, sig, hist


# -------------------- Coinbase fetch --------------------
def normalize_gran(g: int) -> int:
    if g in SUPPORTED_GRANS:
        return g
    return min(SUPPORTED_GRANS, key=lambda x: abs(x - g))


def fetch_candles(granularity: int, limit: int) -> List[Dict[str, Any]]:
    g = normalize_gran(granularity)
    url = f"{CB_BASE}/products/{PRODUCT}/candles"
    headers = {"User-Agent": USER_AGENT}
    params = {"granularity": g, "limit": limit}
    r = requests.get(url, headers=headers, params=params, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Coinbase API error {r.status_code}: {r.text[:200]}")
    data = r.json()
    if not isinstance(data, list) or not data:
        raise RuntimeError("Coinbase API returned empty candles")

    out: List[Dict[str, Any]] = []
    for row in reversed(data):  # ascending
        t, lo, hi, op, cl, vol = row
        iso = datetime.fromtimestamp(int(t), tz=timezone.utc).isoformat()
        out.append(
            {"time": int(t), "iso": iso, "open": float(op), "high": float(hi), "low": float(lo), "close": float(cl), "volume": float(vol)}
        )
    return out


# -------------------- Persistence --------------------
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            note TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS real_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            note TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS state_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            price REAL NOT NULL,
            signal TEXT NOT NULL,
            confidence REAL NOT NULL,
            rsi REAL,
            bias_1h TEXT,
            bias_6h TEXT,
            atr REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_state (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def kv_get(key: str, default: str) -> str:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT v FROM paper_state WHERE k=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row["v"] if row else default


def kv_set(key: str, val: str) -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO paper_state(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (key, val),
    )
    conn.commit()
    conn.close()


# -------------------- State --------------------
@dataclass
class EngineState:
    ok: bool
    iso: str
    price: float
    signal: str
    confidence: float
    bias_1h: str
    bias_6h: str
    rsi: Optional[float]
    atr: Optional[float]
    sl: Optional[float]
    tp: Optional[float]
    peak_watch: Optional[Dict[str, Any]]
    candles: List[Dict[str, Any]]
    markers: List[Dict[str, Any]]
    reason: str


STATE_LOCK = threading.Lock()
STATE: EngineState = EngineState(
    ok=False,
    iso=utc_now_iso(),
    price=0.0,
    signal="WAIT",
    confidence=0.0,
    bias_1h="—",
    bias_6h="—",
    rsi=None,
    atr=None,
    sl=None,
    tp=None,
    peak_watch=None,
    candles=[],
    markers=[],
    reason="Engine starting…",
)


# -------------------- Paper trading --------------------
def load_paper() -> Dict[str, float]:
    usd = float(kv_get("paper_usd", str(PAPER_START_USD)))
    btc = float(kv_get("paper_btc", "0"))
    avg = float(kv_get("paper_avg_entry", "0"))
    return {"usd": usd, "btc": btc, "avg": avg}


def save_paper(usd: float, btc: float, avg: float) -> None:
    kv_set("paper_usd", f"{usd:.8f}")
    kv_set("paper_btc", f"{btc:.8f}")
    kv_set("paper_avg_entry", f"{avg:.8f}")


def paper_equity(price: float) -> Tuple[float, float]:
    st = load_paper()
    equity = st["usd"] + st["btc"] * price
    pnl = equity - PAPER_START_USD
    return equity, pnl


def paper_execute(side: str, qty: float, price: float, note: str = "") -> Dict[str, Any]:
    side = side.upper()
    if qty <= 0:
        raise ValueError("qty must be > 0")

    st = load_paper()
    usd, btc, avg = st["usd"], st["btc"], st["avg"]

    if side == "BUY":
        cost = qty * price
        if cost > usd + 1e-9:
            raise ValueError("insufficient USD for BUY")
        new_btc = btc + qty
        new_avg = (avg * btc + price * qty) / new_btc if new_btc > 0 else 0.0
        usd -= cost
        btc = new_btc
        avg = new_avg
    elif side == "SELL":
        if qty > btc + 1e-9:
            raise ValueError("insufficient BTC for SELL")
        usd += qty * price
        btc -= qty
        if btc <= 1e-9:
            btc = 0.0
            avg = 0.0
    else:
        raise ValueError("side must be BUY or SELL")

    save_paper(usd, btc, avg)

    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO paper_trades(ts,side,qty,price,note) VALUES(?,?,?,?,?)",
        (utc_now_iso(), side, float(qty), float(price), note[:500]),
    )
    conn.commit()
    conn.close()

    equity, pnl = paper_equity(price)
    return {"ok": True, "paper": {"usd": usd, "btc": btc, "avg_entry": avg, "equity": equity, "pnl": pnl}}


def real_log(side: str, qty: float, price: float, note: str = "") -> Dict[str, Any]:
    side = side.upper()
    if side not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    if qty <= 0:
        raise ValueError("qty must be > 0")
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO real_trades(ts,side,qty,price,note) VALUES(?,?,?,?,?)",
        (utc_now_iso(), side, float(qty), float(price), note[:500]),
    )
    conn.commit()
    conn.close()
    return {"ok": True}


def read_trades(limit: int = 200) -> Dict[str, Any]:
    with STATE_LOCK:
        price = STATE.price

    peq, ppnl = paper_equity(price)
    paper = load_paper()

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT ts,side,qty,price,note FROM paper_trades ORDER BY id DESC LIMIT ?", (limit,))
    paper_rows = [dict(r) for r in cur.fetchall()]
    cur.execute("SELECT ts,side,qty,price,note FROM real_trades ORDER BY id DESC LIMIT ?", (limit,))
    real_rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    return {
        "ok": True,
        "paper": {"usd": paper["usd"], "btc": paper["btc"], "avg_entry": paper["avg"], "equity": peq, "pnl": ppnl},
        "paper_trades": list(reversed(paper_rows)),
        "real_trades": list(reversed(real_rows)),
    }


# -------------------- Decision logic --------------------
def trend_bias_from_close(closes: List[float]) -> str:
    if len(closes) < 60:
        return "—"
    e20 = ema(closes, 20)[-1]
    e50 = ema(closes, 50)[-1]
    if e20 > e50:
        return "UP"
    if e20 < e50:
        return "DOWN"
    return "FLAT"


def compute_markers(candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    closes = [c["close"] for c in candles]
    if len(closes) < 30:
        return []
    rsis = rsi(closes, RSI_LEN)
    e20 = ema(closes, 20)
    out: List[Dict[str, Any]] = []
    for i in range(1, len(candles)):
        if rsis[i] is None or rsis[i - 1] is None:
            continue
        if rsis[i - 1] < 50 <= rsis[i] and closes[i] > e20[i]:
            out.append({"t": candles[i]["iso"], "price": closes[i], "type": "bull"})
        if rsis[i - 1] > 50 >= rsis[i] and closes[i] < e20[i]:
            out.append({"t": candles[i]["iso"], "price": closes[i], "type": "bear"})
    return out[-120:]


def peak_dip_watch(candles: List[Dict[str, Any]], current_price: float) -> Optional[Dict[str, Any]]:
    if not candles:
        return None
    window_sec = PEAK_WINDOW_MIN * 60
    n = max(5, int(window_sec / max(60, BASE_GRAN)))
    window = candles[-n:]
    closes = [c["close"] for c in window]
    lo = min(closes)
    hi = max(closes)
    up_from_lo = (current_price - lo) / lo * 100 if lo > 0 else 0.0
    down_from_hi = (hi - current_price) / hi * 100 if hi > 0 else 0.0

    alert = None
    if up_from_lo >= PEAK_THRESHOLD_PCT:
        alert = {"kind": "PEAK_WATCH", "move_pct": round(up_from_lo, 3), "window_min": PEAK_WINDOW_MIN}
    elif down_from_hi >= PEAK_THRESHOLD_PCT:
        alert = {"kind": "DIP_WATCH", "move_pct": round(down_from_hi, 3), "window_min": PEAK_WINDOW_MIN}

    if alert:
        alert.update({"lo": lo, "hi": hi, "price": current_price})
    return alert


def decide(
    base: List[Dict[str, Any]],
    bias1h: List[Dict[str, Any]],
    bias6h: List[Dict[str, Any]],
) -> Tuple[str, float, str, str, Optional[float], Optional[float], Optional[float], Optional[float], str]:

    closes = [c["close"] for c in base]
    highs = [c["high"] for c in base]
    lows = [c["low"] for c in base]
    price = closes[-1] if closes else 0.0

    b1 = trend_bias_from_close([c["close"] for c in bias1h]) if bias1h else "—"
    b6 = trend_bias_from_close([c["close"] for c in bias6h]) if bias6h else "—"

    rsis = rsi(closes, RSI_LEN)
    r = rsis[-1] if rsis else None

    atrs = atr(highs, lows, closes, ATR_LEN)
    a = atrs[-1] if atrs else None

    _, _, macd_hist = macd(closes)
    h = macd_hist[-1] if macd_hist else 0.0

    e20 = ema(closes, 20)
    e50 = ema(closes, 50)
    above_fast = closes[-1] > e20[-1] if e20 else False
    above_slow = closes[-1] > e50[-1] if e50 else False

    aligned_up = (b1 == "UP" and b6 == "UP")
    aligned_down = (b1 == "DOWN" and b6 == "DOWN")

    signal = "WAIT"
    reason_parts: List[str] = []

    long_setup = aligned_up and r is not None and r <= 45 and above_fast and h > 0
    short_setup = aligned_down and r is not None and r >= 55 and (not above_fast) and h < 0

    if long_setup:
        signal = "BUY"
        reason_parts += ["Bias UP (1h+6h)", "RSI <= 45 (pullback)", "MACD hist > 0", "Price > EMA20"]
    elif short_setup:
        signal = "SELL"
        reason_parts += ["Bias DOWN (1h+6h)", "RSI >= 55 (bounce)", "MACD hist < 0", "Price < EMA20"]
    else:
        reason_parts.append("No high-probability setup")

    conf = 5.0
    if aligned_up or aligned_down:
        conf += 20
    if r is not None:
        conf += clamp(abs(r - 50) * 0.8, 0, 25)
    conf += clamp(abs(h) * 1500, 0, 20)
    if above_fast != above_slow:
        conf += 5
    if signal in ("BUY", "SELL"):
        conf += 25
    conf = clamp(conf, 0, 100)

    sl = tp = None
    if a is not None and price > 0:
        if signal == "BUY":
            sl = price - SL_ATR_MULT * a
            tp = price + TP_ATR_MULT * a
        elif signal == "SELL":
            sl = price + SL_ATR_MULT * a
            tp = price - TP_ATR_MULT * a

    return signal, float(conf), b1, b6, (float(r) if r is not None else None), (float(a) if a is not None else None), sl, tp, "; ".join(reason_parts)


# -------------------- Main loop --------------------
_last_peak_alert_kind = None
_last_peak_alert_ts = 0.0


def engine_loop() -> None:
    global _last_peak_alert_kind, _last_peak_alert_ts
    while True:
        try:
            base = fetch_candles(BASE_GRAN, LIMIT_BASE)
            bias1h = fetch_candles(BIAS_1H_GRAN, LIMIT_BIAS)
            bias6h = fetch_candles(BIAS_6H_GRAN, LIMIT_BIAS)

            price = base[-1]["close"] if base else 0.0

            sig, conf, b1, b6, r, a, sl, tp, reason = decide(base, bias1h, bias6h)
            markers = compute_markers(base)
            pd = peak_dip_watch(base, price)

            # Rate-limit Telegram alerts
            if pd and pd.get("kind"):
                now = time.time()
                kind = pd["kind"]
                if kind != _last_peak_alert_kind or (now - _last_peak_alert_ts) > 300:
                    send_telegram(f"🚨 BTC {kind}\nPrice: ${price:,.2f}\nMove: {pd['move_pct']}% (last {pd['window_min']}m)")
                    _last_peak_alert_kind = kind
                    _last_peak_alert_ts = now

            # Optional paper auto-trade (OFF by default)
            if PAPER_AUTO_TRADE and sig in ("BUY", "SELL"):
                paper = load_paper()
                if sig == "BUY":
                    equity = paper["usd"] + paper["btc"] * price
                    usd_to_use = equity * 0.10
                    qty = usd_to_use / price
                    if qty * price <= paper["usd"]:
                        try:
                            paper_execute("BUY", qty, price, note="auto")
                        except Exception:
                            pass
                elif sig == "SELL":
                    qty = paper["btc"] * 0.10
                    if qty > 0:
                        try:
                            paper_execute("SELL", qty, price, note="auto")
                        except Exception:
                            pass

            st = EngineState(
                ok=True,
                iso=utc_now_iso(),
                price=float(price),
                signal=sig,
                confidence=float(conf),
                bias_1h=b1,
                bias_6h=b6,
                rsi=r,
                atr=a,
                sl=sl,
                tp=tp,
                peak_watch=pd,
                candles=base,
                markers=markers,
                reason=reason,
            )

            # Log state row for calibration/backtesting later
            conn = db()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO state_log(ts,price,signal,confidence,rsi,bias_1h,bias_6h,atr) VALUES(?,?,?,?,?,?,?,?)",
                (st.iso, st.price, st.signal, st.confidence, st.rsi, st.bias_1h, st.bias_6h, st.atr),
            )
            conn.commit()
            conn.close()

            with STATE_LOCK:
                global STATE
                STATE = st

        except Exception as e:
            with STATE_LOCK:
                global STATE
                STATE = EngineState(
                    ok=False,
                    iso=utc_now_iso(),
                    price=STATE.price if STATE.price else 0.0,
                    signal="WAIT",
                    confidence=0.0,
                    bias_1h="—",
                    bias_6h="—",
                    rsi=None,
                    atr=None,
                    sl=None,
                    tp=None,
                    peak_watch=None,
                    candles=[],
                    markers=[],
                    reason=str(e),
                )

        time.sleep(POLL_SECONDS)


# -------------------- HTTP API --------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/health"):
            self._send(200, {"ok": True, "iso": utc_now_iso(), "product": PRODUCT, "base_gran": BASE_GRAN})
            return

        if path == "/state":
            with STATE_LOCK:
                self._send(200, asdict(STATE))
            return

        if path == "/trades":
            qs = parse_qs(parsed.query)
            limit = int(qs.get("limit", ["200"])[0])
            limit = int(clamp(limit, 1, 1000))
            self._send(200, read_trades(limit=limit))
            return

        if path == "/explain":
            with STATE_LOCK:
                st = STATE
            self._send(
                200,
                {
                    "ok": True,
                    "iso": st.iso,
                    "signal": st.signal,
                    "confidence": st.confidence,
                    "bias_1h": st.bias_1h,
                    "bias_6h": st.bias_6h,
                    "rsi": st.rsi,
                    "atr": st.atr,
                    "sl": st.sl,
                    "tp": st.tp,
                    "peak_watch": st.peak_watch,
                    "reason": st.reason,
                    "note": "Informational rules-based analysis, not financial advice.",
                },
            )
            return

        self._send(404, {"ok": False, "error": "not_found", "path": path})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        data = self._read_json()

        try:
            if path == "/paper/execute":
                side = str(data.get("side", "")).upper()
                qty = float(data.get("qty", 0))
                price = float(data.get("price") or 0)
                note = str(data.get("note", ""))
                if price <= 0:
                    with STATE_LOCK:
                        price = STATE.price
                self._send(200, paper_execute(side, qty, price, note=note))
                return

            if path == "/real/log":
                side = str(data.get("side", "")).upper()
                qty = float(data.get("qty", 0))
                price = float(data.get("price") or 0)
                note = str(data.get("note", ""))
                if price <= 0:
                    with STATE_LOCK:
                        price = STATE.price
                self._send(200, real_log(side, qty, price, note=note))
                return

            self._send(404, {"ok": False, "error": "not_found", "path": path})
        except Exception as e:
            self._send(400, {"ok": False, "error": str(e)})


def main() -> None:
    init_db()
    threading.Thread(target=engine_loop, daemon=True).start()
    print(f"✅ BTC engine up on http://{HOST}:{PORT} (product={PRODUCT}, base={BASE_GRAN}s)")
    HTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
