import os
import time
import json
import math
import csv
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ================= CONFIG =================
SYMBOL = "BTC-USD"
COINBASE_URL = "https://api.exchange.coinbase.com"

# Coinbase supported granularities: 60, 300, 900, 3600, 21600, 86400
G_5M = 300
G_1H = 3600
G_6H = 21600

RSI_PERIOD = 14
ATR_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200

# Defaults (can be overridden by params.json)
DEFAULT_PARAMS = {
    "min_confidence": 70.0,               # alerts / paper entries only above this
    "atr_sl_mult": 1.5,
    "atr_tp_mult": 2.5,
    "rsi_oversold": 35.0,
    "rsi_overbought": 65.0,
    # confidence weights
    "w_htf_alignment": 30.0,
    "w_rsi_signal": 25.0,
    "w_momentum": 20.0,
    "w_divergence_bonus": 10.0,
    # paper sizing
    "paper_start_usd": 1000.0,
    "paper_risk_fraction": 0.25,          # fraction of equity to allocate per trade (simple)
}

PARAMS_FILE = os.getenv("PARAMS_FILE", "params.json")

# Logs: for Railway, best is to mount a volume and point LOG_DIR to it.
LOG_DIR = os.getenv("LOG_DIR", ".")
SIGNAL_LOG = os.path.join(LOG_DIR, "signals.csv")
PAPER_LOG = os.path.join(LOG_DIR, "paper_trades.csv")
# =========================================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= UTIL =================
def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch_candles(granularity: int, limit: int = 300) -> pd.DataFrame:
    url = f"{COINBASE_URL}/products/{SYMBOL}/candles"
    r = requests.get(url, params={"granularity": granularity}, timeout=12)
    r.raise_for_status()

    df = pd.DataFrame(r.json(), columns=["time","low","high","open","close","volume"])
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df.tail(limit)

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def ema_trend(close: pd.Series) -> str:
    ema_fast = close.ewm(span=EMA_FAST).mean()
    ema_slow = close.ewm(span=EMA_SLOW).mean()
    return "UP" if float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1]) else "DOWN"

def momentum_lookback(close: pd.Series, n: int = 6) -> float:
    if len(close) < n + 1:
        return 0.0
    return float((close.iloc[-1] / close.iloc[-1-n]) - 1.0)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def find_swings(series: pd.Series, lookback: int = 5):
    if len(series) < (lookback * 2 + 10):
        return [], []
    lows, highs = [], []
    arr = series.values
    for i in range(lookback, len(arr) - lookback):
        w = arr[i-lookback:i+lookback+1]
        if arr[i] == np.min(w): lows.append(i)
        if arr[i] == np.max(w): highs.append(i)
    return lows[-2:], highs[-2:]

def detect_rsi_divergence(df: pd.DataFrame) -> dict:
    out = {"bullish": False, "bearish": False, "detail": ""}
    if df is None or df.empty or len(df) < 80:
        return out
    rsi = compute_rsi(df["close"], RSI_PERIOD)
    if rsi.isna().all():
        return out
    lows_idx, highs_idx = find_swings(df["close"], lookback=5)
    if len(lows_idx) == 2:
        i1, i2 = lows_idx
        p1, p2 = float(df["close"].iloc[i1]), float(df["close"].iloc[i2])
        r1, r2 = float(rsi.iloc[i1]), float(rsi.iloc[i2])
        if p2 < p1 and r2 > r1:
            out["bullish"] = True
            out["detail"] += f"Bull div: price LL ({p1:.0f}->{p2:.0f}) + RSI HL ({r1:.1f}->{r2:.1f}). "
    if len(highs_idx) == 2:
        i1, i2 = highs_idx
        p1, p2 = float(df["close"].iloc[i1]), float(df["close"].iloc[i2])
        r1, r2 = float(rsi.iloc[i1]), float(rsi.iloc[i2])
        if p2 > p1 and r2 < r1:
            out["bearish"] = True
            out["detail"] += f"Bear div: price HH ({p1:.0f}->{p2:.0f}) + RSI LH ({r1:.1f}->{r2:.1f}). "
    return out

# ================= PARAMS =================
def load_params():
    p = dict(DEFAULT_PARAMS)
    try:
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE, "r", encoding="utf-8") as f:
                userp = json.load(f)
            if isinstance(userp, dict):
                p.update(userp)
    except Exception:
        pass
    return p

# ================= CSV LOGGING =================
def ensure_csv(path: str, header: list):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

def append_csv(path: str, row: dict, header: list):
    ensure_csv(path, header)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writerow({k: row.get(k, "") for k in header})

SIGNAL_HEADER = [
    "ts_utc","symbol","price","signal","confidence","reason",
    "rsi_5m","rsi_1h","atr_5m","htf_bias","trend_1h","trend_6h",
    "momentum_30m","div_bull","div_bear"
]

PAPER_HEADER = [
    "ts_utc","trade_id","symbol","side","entry_price","exit_price","qty_btc",
    "sl","tp","result","pnl_usd","pnl_pct","hold_minutes"
]

# ================= PAPER SIM =================
PAPER = {
    "equity": None,          # USD
    "pos": None,             # dict or None
    "trade_id": 0,
    "trades": [],            # recent
}

def paper_init(params):
    if PAPER["equity"] is None:
        PAPER["equity"] = float(params["paper_start_usd"])

def paper_maybe_open(signal_pack, params):
    """
    Open a paper trade only on BUY/SELL signals above min_confidence,
    and only if no open position.
    """
    if PAPER["pos"] is not None:
        return

    signal = signal_pack["signal"]
    conf = float(signal_pack["confidence"])
    if signal not in ("BUY", "SELL"):
        return
    if conf < float(params["min_confidence"]):
        return
    if signal_pack["sl_price"] is None or signal_pack["tp_price"] is None:
        return

    price = float(signal_pack["price"])
    equity = float(PAPER["equity"])
    alloc = clamp(float(params["paper_risk_fraction"]), 0.05, 1.0) * equity
    qty = alloc / price

    PAPER["trade_id"] += 1
    PAPER["pos"] = {
        "trade_id": PAPER["trade_id"],
        "side": "LONG" if signal == "BUY" else "SHORT",
        "entry": price,
        "qty": qty,
        "sl": float(signal_pack["sl_price"]),
        "tp": float(signal_pack["tp_price"]),
        "opened_ts": time.time(),
        "opened_iso": now_utc_iso(),
    }

def paper_maybe_close(latest_price: float):
    pos = PAPER["pos"]
    if pos is None:
        return

    side = pos["side"]
    entry = float(pos["entry"])
    sl = float(pos["sl"])
    tp = float(pos["tp"])
    qty = float(pos["qty"])

    hit = None
    if side == "LONG":
        if latest_price <= sl: hit = "SL"
        if latest_price >= tp: hit = "TP"
        pnl_usd = (latest_price - entry) * qty
        pnl_pct = (latest_price / entry) - 1.0
    else:
        if latest_price >= sl: hit = "SL"
        if latest_price <= tp: hit = "TP"
        pnl_usd = (entry - latest_price) * qty
        pnl_pct = (entry / latest_price) - 1.0

    if hit is None:
        return

    PAPER["equity"] = float(PAPER["equity"]) + pnl_usd
    hold_minutes = (time.time() - float(pos["opened_ts"])) / 60.0

    row = {
        "ts_utc": now_utc_iso(),
        "trade_id": pos["trade_id"],
        "symbol": SYMBOL,
        "side": side,
        "entry_price": round(entry, 2),
        "exit_price": round(latest_price, 2),
        "qty_btc": qty,
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "result": hit,
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_pct * 100.0, 3),
        "hold_minutes": round(hold_minutes, 2),
    }
    append_csv(PAPER_LOG, row, PAPER_HEADER)

    PAPER["trades"] = ([row] + PAPER["trades"])[:50]
    PAPER["pos"] = None

# ================= DECISION ENGINE =================
def build_decision(c5, c1h, c6h, params):
    price = float(c5["close"].iloc[-1])

    rsi5s = compute_rsi(c5["close"], RSI_PERIOD)
    rsi1hs = compute_rsi(c1h["close"], RSI_PERIOD)
    atrs = compute_atr(c5, ATR_PERIOD)

    rsi_5m = float(rsi5s.iloc[-1]) if not math.isnan(float(rsi5s.iloc[-1])) else None
    rsi_1h = float(rsi1hs.iloc[-1]) if not math.isnan(float(rsi1hs.iloc[-1])) else None
    atr_5m = float(atrs.iloc[-1]) if not math.isnan(float(atrs.iloc[-1])) else None

    trend_1h = ema_trend(c1h["close"])
    trend_6h = ema_trend(c6h["close"])
    htf_bias = trend_1h if trend_1h == trend_6h else "NEUTRAL"

    mom = momentum_lookback(c5["close"], n=6)
    div = detect_rsi_divergence(c5)

    # Confidence breakdown
    w_htf = float(params["w_htf_alignment"])
    w_rsi = float(params["w_rsi_signal"])
    w_mom = float(params["w_momentum"])
    w_div = float(params["w_divergence_bonus"])

    conf_parts = {}
    conf_parts["htf_alignment"] = w_htf if htf_bias in ("UP","DOWN") else w_htf * 0.33

    if rsi_5m is None:
        conf_parts["rsi_signal"] = 0.0
    else:
        if htf_bias == "UP" and rsi_5m < float(params["rsi_oversold"]):
            conf_parts["rsi_signal"] = w_rsi
        elif htf_bias == "DOWN" and rsi_5m > float(params["rsi_overbought"]):
            conf_parts["rsi_signal"] = w_rsi
        else:
            conf_parts["rsi_signal"] = w_rsi * 0.4

    if htf_bias == "UP":
        conf_parts["momentum"] = w_mom if mom > 0 else w_mom * 0.4
    elif htf_bias == "DOWN":
        conf_parts["momentum"] = w_mom if mom < 0 else w_mom * 0.4
    else:
        conf_parts["momentum"] = w_mom * 0.4

    div_bonus = 0.0
    if div["bullish"]: div_bonus += w_div
    if div["bearish"]: div_bonus += w_div
    conf_parts["divergence_bonus"] = div_bonus

    confidence = clamp(sum(conf_parts.values()), 1.0, 95.0)

    # Signal rules
    signal = "WAIT"
    reason = "No high-probability setup."

    if htf_bias == "UP" and rsi_5m is not None:
        if (rsi_5m < float(params["rsi_oversold"])) and (mom > 0 or div["bullish"]):
            signal = "BUY"
            reason = "HTF UP + RSI oversold + confirmation (momentum/divergence)."

    if htf_bias == "DOWN" and rsi_5m is not None:
        if (rsi_5m > float(params["rsi_overbought"])) and (mom < 0 or div["bearish"]):
            signal = "SELL"
            reason = "HTF DOWN + RSI overbought + confirmation (momentum/divergence)."

    # ATR SL/TP
    sl_price = None
    tp_price = None
    if atr_5m is not None:
        sl_mult = float(params["atr_sl_mult"])
        tp_mult = float(params["atr_tp_mult"])
        if signal == "BUY":
            sl_price = price - sl_mult * atr_5m
            tp_price = price + tp_mult * atr_5m
        elif signal == "SELL":
            sl_price = price + sl_mult * atr_5m
            tp_price = price - tp_mult * atr_5m

    watch = {
        "peak_watch": bool(rsi_5m is not None and rsi_5m > 60 and mom > 0),
        "dip_watch": bool(rsi_5m is not None and rsi_5m < 40 and mom < 0),
    }

    return {
        "price": price,
        "signal": signal,
        "confidence": round(confidence, 2),
        "reason": reason,
        "rsi_5m": None if rsi_5m is None else round(rsi_5m, 2),
        "rsi_1h": None if rsi_1h is None else round(rsi_1h, 2),
        "atr_5m": None if atr_5m is None else round(atr_5m, 2),
        "sl_price": None if sl_price is None else round(sl_price, 2),
        "tp_price": None if tp_price is None else round(tp_price, 2),
        "trend_1h": trend_1h,
        "trend_6h": trend_6h,
        "htf_bias": htf_bias,
        "momentum_30m": round(mom, 6),
        "divergence": div,
        "watch": watch,
        "confidence_breakdown": conf_parts,
    }

# ================= STATE CACHE =================
LAST_STATE = {"ok": False, "error": "Not ready"}

# ================= API =================
@app.get("/health")
def health():
    return {"ok": True, "service": "btc-engine", "time": now_utc_iso()}

@app.get("/state")
def state():
    return LAST_STATE

@app.get("/recent_signals")
def recent_signals(limit: int = 50):
    # Return last N rows from signals.csv (if present)
    if not os.path.exists(SIGNAL_LOG):
        return {"ok": True, "rows": []}
    try:
        df = pd.read_csv(SIGNAL_LOG).tail(int(clamp(limit, 1, 500)))
        return {"ok": True, "rows": df.to_dict(orient="records")}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/paper_trades")
def paper_trades(limit: int = 50):
    if not os.path.exists(PAPER_LOG):
        return {"ok": True, "rows": []}
    try:
        df = pd.read_csv(PAPER_LOG).tail(int(clamp(limit, 1, 500)))
        return {"ok": True, "rows": df.to_dict(orient="records")}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ================= LOOP =================
def main_loop():
    global LAST_STATE
    while True:
        params = load_params()
        paper_init(params)

        try:
            c5 = fetch_candles(G_5M, limit=300)
            c1h = fetch_candles(G_1H, limit=300)
            c6h = fetch_candles(G_6H, limit=300)

            if c5.empty or c1h.empty or c6h.empty:
                LAST_STATE = {"ok": False, "error": "Empty candle feed from Coinbase."}
                time.sleep(5)
                continue

            d = build_decision(c5, c1h, c6h, params)

            # --- Log the signal snapshot every loop (so you can calibrate) ---
            row = {
                "ts_utc": now_utc_iso(),
                "symbol": SYMBOL,
                "price": d["price"],
                "signal": d["signal"],
                "confidence": d["confidence"],
                "reason": d["reason"],
                "rsi_5m": d["rsi_5m"],
                "rsi_1h": d["rsi_1h"],
                "atr_5m": d["atr_5m"],
                "htf_bias": d["htf_bias"],
                "trend_1h": d["trend_1h"],
                "trend_6h": d["trend_6h"],
                "momentum_30m": d["momentum_30m"],
                "div_bull": bool(d["divergence"].get("bullish")),
                "div_bear": bool(d["divergence"].get("bearish")),
            }
            append_csv(SIGNAL_LOG, row, SIGNAL_HEADER)

            # --- Paper sim: open/close based on the same rules ---
            paper_maybe_close(float(d["price"]))
            paper_maybe_open(d, params)

            LAST_STATE = {
                "ok": True,
                "time": now_utc_iso(),
                "src": "Coinbase",
                **d,
                "params": {k: params[k] for k in [
                    "min_confidence","atr_sl_mult","atr_tp_mult",
                    "rsi_oversold","rsi_overbought",
                    "w_htf_alignment","w_rsi_signal","w_momentum","w_divergence_bonus"
                ]},
                "paper": {
                    "equity": round(float(PAPER["equity"]), 2),
                    "open_position": PAPER["pos"],
                    "recent_trades": PAPER["trades"],
                }
            }

        except Exception as e:
            LAST_STATE = {"ok": False, "error": str(e), "time": now_utc_iso()}

        time.sleep(5)

# Run loop when this file starts (Railway/uvicorn will import module)
import threading
threading.Thread(target=main_loop, daemon=True).start()
