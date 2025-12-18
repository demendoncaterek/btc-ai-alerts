import os
import time
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

VERSION = "2025-12-17"

PRODUCT = os.getenv("PRODUCT", "BTC-USD")
EXCHANGE_BASE = os.getenv("EXCHANGE_BASE", "https://api.exchange.coinbase.com").rstrip("/")
REFRESH_SEC = int(os.getenv("REFRESH_SEC", "10"))  # background refresh cadence
PEAK_WINDOW_MIN = int(os.getenv("PEAK_WINDOW_MIN", "180"))
NEAR_PEAK_PCT = float(os.getenv("NEAR_PEAK_PCT", "0.003"))  # 0.3%
NEAR_DIP_PCT = float(os.getenv("NEAR_DIP_PCT", "0.003"))

AUTO_PAPER = os.getenv("AUTO_PAPER", "false").lower() in ("1", "true", "yes", "y")
PAPER_START_USD = float(os.getenv("PAPER_START_USD", "250"))
PAPER_RISK_PCT = float(os.getenv("PAPER_RISK_PCT", "0.01"))  # risk 1% of equity per trade (paper)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()  # if set, only respond to this chat id

app = FastAPI(title="BTC Engine", version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state_lock = threading.Lock()
_state: Dict[str, Any] = {"ok": False, "error": "starting"}
_last_broadcast_sig: Optional[str] = None

_paper = {
    "usd": PAPER_START_USD,
    "btc": 0.0,
    "pos": None,  # {"side": "LONG"/"SHORT", "qty": float, "entry": float, "sl": float, "tp": float, "ts": iso}
    "equity": PAPER_START_USD,
    "unreal_pnl": 0.0,
}
_paper_trades: List[Dict[str, Any]] = []
_real_trades: List[Dict[str, Any]] = []


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tg_api(method: str, payload: dict, timeout: float = 10.0) -> Any:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _tg_send(chat_id: str, text: str) -> None:
    if not TELEGRAM_TOKEN:
        return
    try:
        _tg_api("sendMessage", {"chat_id": chat_id, "text": text}, timeout=8.0)
    except Exception:
        pass


def _tg_allowed(chat_id: str) -> bool:
    return (not TELEGRAM_CHAT_ID) or (str(chat_id) == str(TELEGRAM_CHAT_ID))


def _tg_poll_loop() -> None:
    # Poll Telegram so the bot can reply to: /health, /status, /trades
    if not TELEGRAM_TOKEN:
        return

    offset = 0
    while True:
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                params={"timeout": 30, "offset": offset},
                timeout=35,
            ).json()

            for upd in resp.get("result", []):
                offset = max(offset, int(upd.get("update_id", 0)) + 1)
                msg = upd.get("message") or upd.get("edited_message") or {}
                text = (msg.get("text") or "").strip()
                chat = msg.get("chat") or {}
                chat_id = str(chat.get("id", ""))

                if not (chat_id and text):
                    continue
                if not _tg_allowed(chat_id):
                    continue

                cmd = text.split()[0].lower()
                if cmd == "/health":
                    with _state_lock:
                        ok = bool(_state.get("ok"))
                        iso = _state.get("iso", _utc_iso())
                        err = _state.get("error")
                    _tg_send(chat_id, f"✅ Health\nengine_ok: {ok}\niso: {iso}\nerror: {err}")
                elif cmd in ("/status", "/state"):
                    with _state_lock:
                        s = dict(_state)
                    if not s.get("ok"):
                        _tg_send(chat_id, f"⏳ Engine not ready.\nerror: {s.get('error')}\niso: {s.get('iso')}")
                    else:
                        _tg_send(
                            chat_id,
                            "📊 Status\n"
                            f"Price: ${s.get('price', 0):,.2f}\n"
                            f"Signal: {s.get('signal')} ({float(s.get('confidence', 0.0)):.0f}%)\n"
                            f"Bias(1h/6h): {s.get('bias_1h')}/{s.get('bias_6h')}\n"
                            f"RSI(5m/15m): {s.get('rsi_5m')} / {s.get('rsi_15m')}\n"
                            f"SL/TP: {s.get('sl')} / {s.get('tp')}\n"
                            f"iso: {s.get('iso')}",
                        )
                elif cmd == "/trades":
                    last = _paper_trades[-5:]
                    if not last:
                        _tg_send(chat_id, "No paper trades yet.")
                    else:
                        lines = [
                            f"{t.get('ts','')[-8:]} {t.get('type')} {t.get('side','')} @ {t.get('price','')}"
                            for t in last
                        ]
                        _tg_send(chat_id, "🧪 Last paper trades:\n" + "\n".join(lines))
                elif cmd in ("/start", "/help"):
                    _tg_send(chat_id, "Commands: /health /status /trades")
        except Exception:
            pass

        time.sleep(2)


def _fetch_json(url: str, params: Optional[dict] = None, timeout: float = 8.0) -> Any:
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "btc-engine/1.0"})
    r.raise_for_status()
    return r.json()


def fetch_spot_price() -> float:
    data = _fetch_json(f"{EXCHANGE_BASE}/products/{PRODUCT}/ticker", timeout=6.0)
    return float(data["price"])


def fetch_candles(granularity_sec: int) -> pd.DataFrame:
    raw = _fetch_json(
        f"{EXCHANGE_BASE}/products/{PRODUCT}/candles",
        params={"granularity": granularity_sec},
        timeout=10.0,
    )
    df = pd.DataFrame(raw, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    for c in ["low", "high", "open", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m = ema(series, fast) - ema(series, slow)
    s = ema(m, signal)
    h = m - s
    return m, s, h


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bias_from_ma(df: pd.DataFrame, span: int = 20) -> str:
    if len(df) < span + 2:
        return "UNKNOWN"
    ma = ema(df["close"], span)
    return "UP" if df["close"].iloc[-1] >= ma.iloc[-1] else "DOWN"


def _paper_mark_to_market(price: float) -> None:
    pos = _paper["pos"]
    if pos is None:
        _paper["equity"] = _paper["usd"] + _paper["btc"] * price
        _paper["unreal_pnl"] = 0.0
        return

    qty = float(pos["qty"])
    entry = float(pos["entry"])
    unreal = (price - entry) * qty if pos["side"] == "LONG" else (entry - price) * qty
    _paper["unreal_pnl"] = float(unreal)
    _paper["equity"] = float(_paper["usd"] + unreal)


def _paper_enter(side: str, price: float, atr_val: float, reason: str, confidence: float) -> None:
    if _paper["pos"] is not None:
        return

    equity = float(_paper.get("equity", _paper["usd"]))
    risk_dollars = max(1.0, equity * PAPER_RISK_PCT)
    sl_mult = 1.2
    tp_mult = 2.0
    if atr_val <= 0 or np.isnan(atr_val):
        atr_val = max(10.0, price * 0.001)

    if side == "BUY":
        sl = price - atr_val * sl_mult
        tp = price + atr_val * tp_mult
        per_unit_risk = max(1e-9, price - sl)
        qty = risk_dollars / per_unit_risk
        _paper["pos"] = {"side": "LONG", "qty": float(qty), "entry": float(price), "sl": float(sl), "tp": float(tp), "ts": _utc_iso()}
    else:
        sl = price + atr_val * sl_mult
        tp = price - atr_val * tp_mult
        per_unit_risk = max(1e-9, sl - price)
        qty = risk_dollars / per_unit_risk
        _paper["pos"] = {"side": "SHORT", "qty": float(qty), "entry": float(price), "sl": float(sl), "tp": float(tp), "ts": _utc_iso()}

    _paper_trades.append(
        {"ts": _utc_iso(), "type": "ENTRY", "side": side, "price": float(price), "qty": float(_paper["pos"]["qty"]),
         "sl": float(_paper["pos"]["sl"]), "tp": float(_paper["pos"]["tp"]), "reason": reason, "confidence": float(confidence)}
    )


def _paper_check_exit(price: float) -> None:
    pos = _paper["pos"]
    if pos is None:
        return
    sl = float(pos["sl"])
    tp = float(pos["tp"])
    entry = float(pos["entry"])
    qty = float(pos["qty"])

    hit = None
    pnl = 0.0
    if pos["side"] == "LONG":
        if price <= sl:
            hit = "SL"
        elif price >= tp:
            hit = "TP"
        if hit:
            pnl = (price - entry) * qty
    else:
        if price >= sl:
            hit = "SL"
        elif price <= tp:
            hit = "TP"
        if hit:
            pnl = (entry - price) * qty

    if hit:
        _paper["usd"] = float(_paper["usd"] + pnl)
        _paper_trades.append({"ts": _utc_iso(), "type": "EXIT", "hit": hit, "price": float(price), "pnl": float(pnl), "equity": float(_paper["usd"])})
        _paper["pos"] = None
        _paper_mark_to_market(price)


def compute_signal(
    price: float,
    rsi5: Optional[float],
    macd_hist_15: Optional[float],
    macd_hist_15_prev: Optional[float],
    b1h: str,
    b6h: str,
    atr15: Optional[float],
    near_peak: bool,
    near_dip: bool,
) -> Tuple[str, float, str]:
    reasons = []
    long_ok = (b1h == "UP" and b6h == "UP")
    short_ok = (b1h == "DOWN" and b6h == "DOWN")

    if b1h != "UNKNOWN":
        reasons.append(f"1h bias {b1h.lower()}")
    if b6h != "UNKNOWN":
        reasons.append(f"6h bias {b6h.lower()}")

    if rsi5 is not None:
        if rsi5 <= 35:
            reasons.append("RSI5 oversold")
        if rsi5 >= 65:
            reasons.append("RSI5 overbought")

    macd_up = macd_hist_15 is not None and macd_hist_15_prev is not None and macd_hist_15 > macd_hist_15_prev
    macd_down = macd_hist_15 is not None and macd_hist_15_prev is not None and macd_hist_15 < macd_hist_15_prev

    if near_peak:
        reasons.append("near rolling peak")
    if near_dip:
        reasons.append("near rolling dip")

    signal = "WAIT"
    if long_ok and (rsi5 is not None and rsi5 <= 35) and macd_up and not near_peak:
        signal = "BUY"
    elif short_ok and (rsi5 is not None and rsi5 >= 65) and macd_down and not near_dip:
        signal = "SELL"

    conf = 20.0 if signal == "WAIT" else 35.0
    if signal == "BUY":
        if rsi5 is not None and rsi5 <= 30:
            conf += 10
        if macd_up:
            conf += 15
        conf += (10 if b1h == "UP" else 0) + (10 if b6h == "UP" else 0)
        conf += 5 if (atr15 is not None and not np.isnan(atr15)) else 0
    elif signal == "SELL":
        if rsi5 is not None and rsi5 >= 70:
            conf += 10
        if macd_down:
            conf += 15
        conf += (10 if b1h == "DOWN" else 0) + (10 if b6h == "DOWN" else 0)
        conf += 5 if (atr15 is not None and not np.isnan(atr15)) else 0

    conf = float(max(0.0, min(100.0, conf)))
    return signal, conf, "; ".join(reasons[:6]) if reasons else "no setup"


def _build_state() -> Dict[str, Any]:
    t0 = time.time()

    c5 = fetch_candles(300)
    c15 = fetch_candles(900)
    c1h = fetch_candles(3600)
    c6h = fetch_candles(21600)

    price = float(c5["close"].iloc[-1]) if len(c5) else fetch_spot_price()

    rsi5_val = float(rsi(c5["close"]).iloc[-1]) if len(c5) >= 20 else None
    rsi15_val = float(rsi(c15["close"]).iloc[-1]) if len(c15) >= 20 else None

    _, _, h15 = macd(c15["close"]) if len(c15) >= 40 else (None, None, None)
    macd_hist_15 = float(h15.iloc[-1]) if h15 is not None else None
    macd_hist_15_prev = float(h15.iloc[-2]) if h15 is not None and len(h15) >= 2 else None

    atr15_series = atr(c15) if len(c15) >= 20 else None
    atr15_val = float(atr15_series.iloc[-1]) if atr15_series is not None else None

    b1h = bias_from_ma(c1h)
    b6h = bias_from_ma(c6h)

    n = max(5, int(PEAK_WINDOW_MIN / 5))
    peak = float(c5["high"].tail(n).max()) if len(c5) else price
    dip = float(c5["low"].tail(n).min()) if len(c5) else price
    near_peak = (peak - price) / peak <= NEAR_PEAK_PCT if peak > 0 else False
    near_dip = (price - dip) / price <= NEAR_DIP_PCT if price > 0 else False

    signal, conf, reason = compute_signal(price, rsi5_val, macd_hist_15, macd_hist_15_prev, b1h, b6h, atr15_val, near_peak, near_dip)

    sl = tp = None
    if atr15_val is not None and not np.isnan(atr15_val):
        sl_mult = 1.2
        tp_mult = 2.0
        if signal == "BUY":
            sl = price - atr15_val * sl_mult
            tp = price + atr15_val * tp_mult
        elif signal == "SELL":
            sl = price + atr15_val * sl_mult
            tp = price - atr15_val * tp_mult

    _paper_mark_to_market(price)
    _paper_check_exit(price)
    if AUTO_PAPER and signal in ("BUY", "SELL"):
        _paper_enter(signal, price, atr15_val or 0.0, reason=reason, confidence=conf)

    events: List[Dict[str, Any]] = []
    if near_peak:
        events.append({"type": "PEAK_WATCH", "price": price, "near_peak": peak, "window_min": PEAK_WINDOW_MIN})
    if near_dip:
        events.append({"type": "DIP_WATCH", "price": price, "near_dip": dip, "window_min": PEAK_WINDOW_MIN})

    def _serialize(df: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
        d = df.tail(limit).copy()
        out = []
        for _, row in d.iterrows():
            out.append(
                {"time": row["time"].isoformat(), "open": float(row["open"]), "high": float(row["high"]),
                 "low": float(row["low"]), "close": float(row["close"]), "volume": float(row.get("volume", 0.0))}
            )
        return out

    return {
        "ok": True,
        "version": VERSION,
        "iso": _utc_iso(),
        "latency_ms": int((time.time() - t0) * 1000),
        "src": "Coinbase Exchange",
        "product": PRODUCT,
        "price": price,
        "signal": signal,
        "confidence": conf,
        "reason": reason,
        "bias_1h": b1h,
        "bias_6h": b6h,
        "rsi_5m": rsi5_val,
        "rsi_15m": rsi15_val,
        "macd_hist_15m": macd_hist_15,
        "atr_15m": atr15_val,
        "sl": sl,
        "tp": tp,
        "peak": {"near": near_peak, "value": peak, "window_min": PEAK_WINDOW_MIN},
        "dip": {"near": near_dip, "value": dip, "window_min": PEAK_WINDOW_MIN},
        "events": events,
        "paper": {"usd": float(_paper["usd"]), "equity": float(_paper["equity"]), "unreal_pnl": float(_paper["unreal_pnl"]),
                  "pos": _paper["pos"], "trades": _paper_trades[-50:]},
        "real": {"trades": _real_trades[-50:]},
        "candles_15m": _serialize(c15, 120),
        "candles_5m": _serialize(c5, 240),
    }


def _engine_loop() -> None:
    global _last_broadcast_sig
    while True:
        try:
            new_state = _build_state()
            with _state_lock:
                _state.clear()
                _state.update(new_state)

            # Optional push updates to TELEGRAM_CHAT_ID when setup changes
            if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                sig = f"{new_state.get('signal')}|{new_state.get('bias_1h')}|{new_state.get('bias_6h')}|{new_state.get('peak',{}).get('near')}|{new_state.get('dip',{}).get('near')}"
                if sig != _last_broadcast_sig:
                    _last_broadcast_sig = sig
                    _tg_send(
                        TELEGRAM_CHAT_ID,
                        f"BTC Update\nPrice: ${new_state['price']:.2f}\nSignal: {new_state['signal']} ({new_state['confidence']:.0f}%)\nBias(1h/6h): {new_state['bias_1h']}/{new_state['bias_6h']}"
                    )
        except Exception as e:
            with _state_lock:
                _state.clear()
                _state.update({"ok": False, "error": str(e), "iso": _utc_iso()})
        time.sleep(max(2, REFRESH_SEC))


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_engine_loop, daemon=True).start()
    if TELEGRAM_TOKEN:
        threading.Thread(target=_tg_poll_loop, daemon=True).start()


@app.get("/health")
def health() -> Dict[str, Any]:
    # MUST be fast and never do network calls.
    with _state_lock:
        ok = bool(_state.get("ok"))
        iso = _state.get("iso", _utc_iso())
        err = _state.get("error")
    return {"ok": True, "engine_ok": ok, "version": VERSION, "iso": iso, "error": err}


@app.get("/state")
def state() -> Dict[str, Any]:
    with _state_lock:
        return dict(_state)


@app.get("/trades")
def trades() -> Dict[str, Any]:
    return {"paper": _paper_trades, "real": _real_trades}


@app.post("/paper/reset")
def paper_reset() -> Dict[str, Any]:
    _paper_trades.clear()
    _paper["usd"] = PAPER_START_USD
    _paper["btc"] = 0.0
    _paper["pos"] = None
    _paper["equity"] = PAPER_START_USD
    _paper["unreal_pnl"] = 0.0
    return {"ok": True}


@app.post("/real/log")
def real_log(trade: Dict[str, Any]) -> Dict[str, Any]:
    trade = dict(trade)
    trade.setdefault("ts", _utc_iso())
    _real_trades.append(trade)
    return {"ok": True, "count": len(_real_trades)}
