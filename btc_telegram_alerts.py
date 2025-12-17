import time
import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ================= CONFIG =================
SYMBOL = "BTC-USD"
COINBASE_URL = "https://api.exchange.coinbase.com"
RSI_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200
# =========================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= UTIL =================
def fetch_candles(granularity, limit=300):
    url = f"{COINBASE_URL}/products/{SYMBOL}/candles"
    params = {"granularity": granularity}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()

    df = pd.DataFrame(
        r.json(),
        columns=["time", "low", "high", "open", "close", "volume"],
    )
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time")
    return df.tail(limit)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def ema_trend(series):
    ema_fast = series.ewm(span=EMA_FAST).mean()
    ema_slow = series.ewm(span=EMA_SLOW).mean()
    return "UP" if ema_fast.iloc[-1] > ema_slow.iloc[-1] else "DOWN"

# ================= ENGINE =================
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/state")
def state():
    try:
        # --- Fetch candles ---
        candles_5m = fetch_candles(300)
        candles_1h = fetch_candles(3600)
        candles_4h = fetch_candles(14400)

        price = float(candles_5m["close"].iloc[-1])

        # --- Indicators ---
        rsi_5m = compute_rsi(candles_5m["close"], RSI_PERIOD)
        rsi_1h = compute_rsi(candles_1h["close"], RSI_PERIOD)

        trend_1h = ema_trend(candles_1h["close"])
        trend_4h = ema_trend(candles_4h["close"])

        # --- HTF bias ---
        if trend_1h == trend_4h:
            htf_bias = trend_1h
        else:
            htf_bias = "NEUTRAL"

        # --- Signal logic ---
        signal = "WAIT"
        reason = "No setup"

        if htf_bias == "UP" and rsi_5m < 35:
            signal = "BUY"
            reason = "HTF uptrend + RSI recovery"
        elif htf_bias == "DOWN" and rsi_5m > 65:
            signal = "SELL"
            reason = "HTF downtrend + RSI rollover"

        confidence = round(
            max(5, min(95, abs(50 - rsi_5m) * 1.5)), 2
        )

        return {
            "ok": True,
            "time": time.strftime("%H:%M:%S"),
            "price": price,
            "signal": signal,
            "confidence": confidence,
            "reason": reason,

            # indicators (THIS IS WHAT YOUR UI NEEDS)
            "rsi_5m": round(rsi_5m, 2),
            "rsi_1h": round(rsi_1h, 2),
            "trend_1h": trend_1h,
            "trend_4h": trend_4h,
            "htf_bias": htf_bias,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }
