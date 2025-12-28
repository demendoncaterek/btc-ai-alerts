import os, time, math, requests
from datetime import datetime
from fastapi import FastAPI
import pandas as pd
import numpy as np

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

# =========================
# MARKET DATA
# =========================
def fetch_ohlc(limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit={limit}"
    data = requests.get(url, timeout=10).json()
    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","vol",
        "_","_","_","_","_","_"
    ])
    for c in ["open","high","low","close"]:
        df[c] = df[c].astype(float)
    return df

def indicators(df):
    close = df["close"]

    rsi = 100 - (100 / (1 + close.diff().clip(lower=0).rolling(14).mean() /
                       close.diff().clip(upper=0).abs().rolling(14).mean()))

    macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    atr = (df["high"] - df["low"]).rolling(14).mean()

    return rsi.iloc[-1], macd.iloc[-1], atr.iloc[-1]

# =========================
# CONFIDENCE ENGINE
# =========================
def score_confidence(price, rsi, macd, atr, ema):
    score = 0.0

    if rsi < 35: score += 0.30
    elif rsi < 45: score += 0.15

    if macd > 0: score += 0.25

    if price < ema: score += 0.20

    if atr / price > 0.002: score += 0.10

    return min(score, 1.0)

def adaptive_threshold():
    global confidence_floor
    if len(trades) < 5:
        return confidence_floor

    wins = [t for t in trades[-10:] if t["pnl"] > 0]
    win_rate = len(wins) / len(trades[-10:])

    if win_rate > 0.6:
        confidence_floor = min(confidence_floor + 0.01, MIN_CONF_CAP)
    elif win_rate < 0.4:
        confidence_floor = max(confidence_floor - 0.01, MIN_CONF_FLOOR)

    return confidence_floor

# =========================
# TRADE LOOP
# =========================
def trade_tick():
    global equity, position

    df = fetch_ohlc()
    price = df["close"].iloc[-1]
    ema = df["close"].ewm(span=50).mean().iloc[-1]

    rsi, macd, atr = indicators(df)
    conf = score_confidence(price, rsi, macd, atr, ema)
    threshold = adaptive_threshold()

    # ---- BUY ----
    if position is None:
        explore = np.random.rand() < EXPLORATION_RATE

        if conf >= threshold or explore:
            risk = equity * RISK_PCT
            size = risk / atr if atr > 0 else 0

            position = {
                "entry": price,
                "size": size,
                "sl": price - atr * 1.5,
                "tp": price + atr * 2.5,
                "time": datetime.utcnow().isoformat(),
                "confidence": conf
            }

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

            position = None

    return {
        "price": price,
        "signal": "LONG" if position else "WAIT",
        "confidence": round(conf * 100, 2),
        "equity": round(equity, 2),
        "threshold": round(threshold * 100, 2)
    }

# =========================
# API
# =========================
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/state")
def state():
    return trade_tick()

@app.get("/trades")
def get_trades():
    return trades
