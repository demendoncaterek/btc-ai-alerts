import os, asyncio, math, requests
from datetime import datetime
from fastapi import FastAPI
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("btc-engine")

app = FastAPI()

# =========================
# ENV CONFIG
# =========================
SYMBOL = os.getenv("SYMBOL", "BTC-USD")
POLL_SEC = int(os.getenv("POLL_SEC", 10))
START_EQUITY = float(os.getenv("START_EQUITY", 250))
RISK_PCT = float(os.getenv("RISK_PCT", 0.01))

MIN_CONF_DEFAULT = float(os.getenv("MIN_CONF_DEFAULT", 0.55))
MIN_CONF_FLOOR = float(os.getenv("MIN_CONF_FLOOR", 0.48))
MIN_CONF_CAP = float(os.getenv("MIN_CONF_CAP", 0.72))
EXPLORATION_RATE = float(os.getenv("EXPLORATION_RATE", 0.06))

# =========================
# STATE
# =========================
equity = START_EQUITY
position = None
trades = []
confidence_floor = MIN_CONF_DEFAULT
last_state = {}

# =========================
# MARKET DATA
# =========================
def fetch_ohlc(limit=200):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "15m", "limit": limit}
    data = requests.get(url, params=params, timeout=10).json()

    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","vol",
        "_","_","_","_","_","_"
    ])
    for c in ["open","high","low","close"]:
        df[c] = df[c].astype(float)
    return df

def indicators(df):
    close = df["close"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    rsi = 100 - (100 / (1 + rs))

    macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    atr = (df["high"] - df["low"]).rolling(14).mean()

    return rsi.iloc[-1], macd.iloc[-1], atr.iloc[-1]

# =========================
# CONFIDENCE ENGINE
# =========================
def score_confidence(price, rsi, macd, atr, ema):
    score = 0.0

    if rsi < 30: score += 0.40
    elif rsi < 40: score += 0.25
    elif rsi < 50: score += 0.10

    if macd > 0: score += 0.25
    if price < ema: score += 0.20
    if atr / price > 0.002: score += 0.10

    return min(score, 1.0)

def adaptive_threshold():
    global confidence_floor

    if len(trades) < 5:
        return confidence_floor

    recent = trades[-10:]
    win_rate = sum(1 for t in recent if t["pnl"] > 0) / len(recent)

    if win_rate > 0.6:
        confidence_floor = min(confidence_floor + 0.01, MIN_CONF_CAP)
    elif win_rate < 0.4:
        confidence_floor = max(confidence_floor - 0.01, MIN_CONF_FLOOR)

    return confidence_floor

# =========================
# TRADE LOGIC
# =========================
def trade_tick():
    global equity, position, last_state

    df = fetch_ohlc()
    price = df["close"].iloc[-1]
    ema = df["close"].ewm(span=50).mean().iloc[-1]

    rsi, macd, atr = indicators(df)
    conf = score_confidence(price, rsi, macd, atr, ema)
    threshold = adaptive_threshold()

    # ---- HARD DIP CATCHER ----
    forced_buy = rsi < 25 and price < ema * 0.98

    # ---- BUY ----
    if position is None:
        explore = np.random.rand() < EXPLORATION_RATE

        if conf >= threshold or explore or forced_buy:
            risk = equity * RISK_PCT
            size = risk / atr if atr > 0 else 0

            position = {
                "entry": price,
                "size": size,
                "sl": price - atr * 1.5,
                "tp": price + atr * 2.5,
                "confidence": conf,
                "time": datetime.utcnow().isoformat()
            }
            log.info(f"BUY @ {price:.2f} conf={conf:.2f}")

    # ---- SELL ----
    else:
        if price <= position["sl"] or price >= position["tp"]:
            pnl = (price - position["entry"]) * position["size"]
            equity += pnl

            trades.append({
                "entry": position["entry"],
                "exit": price,
                "pnl": pnl,
                "confidence": position["confidence"],
                "time": datetime.utcnow().isoformat()
            })

            log.info(f"SELL @ {price:.2f} pnl={pnl:.2f}")
            position = None

    last_state = {
        "price": round(price, 2),
        "signal": "LONG" if position else "WAIT",
        "confidence": round(conf * 100, 2),
        "equity": round(equity, 2),
        "threshold": round(threshold * 100, 2)
    }

# =========================
# BACKGROUND LOOP (Railway-safe)
# =========================
async def engine_loop():
    while True:
        try:
            trade_tick()
            await asyncio.sleep(POLL_SEC)
        except asyncio.CancelledError:
            log.warning("Engine loop cancelled")
            break
        except Exception as e:
            log.exception("Engine error", exc_info=e)
            await asyncio.sleep(5)

@app.on_event("startup")
async def startup():
    app.state.task = asyncio.create_task(engine_loop())

@app.on_event("shutdown")
async def shutdown():
    task = getattr(app.state, "task", None)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

# =========================
# API
# =========================
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/state")
def state():
    return last_state

@app.get("/trades")
def get_trades():
    return trades

@app.get("/calibrate")
def calibrate():
    trade_tick()
    return {"status": "ok", "picked": last_state}
