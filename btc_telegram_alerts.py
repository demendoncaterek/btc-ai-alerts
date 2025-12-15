import asyncio
import json
import time
import os
import requests
import numpy as np
from datetime import datetime

# =========================
# CONFIG
# =========================
PRODUCT = "BTC-USD"          # Coinbase product (USD)
GRANULARITY = 60             # 60 seconds = 1 minute candles
STATE_FILE = "btc_state.json"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

ALERT_COOLDOWN = 300         # seconds
MIN_CONFIDENCE = 70          # only high-quality alerts

COINBASE_BASE = "https://api.exchange.coinbase.com"

# =========================
# HELPERS
# =========================
def send_telegram(msg: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=8,
        )
    except:
        pass


def fetch_candles(limit=60):
    url = f"{COINBASE_BASE}/products/{PRODUCT}/candles"
    resp = requests.get(
        url,
        params={"granularity": GRANULARITY},
        headers={"Accept": "application/json", "User-Agent": "btc-alerts"},
        timeout=10,
    )

    data = resp.json()

    if not isinstance(data, list):
        raise RuntimeError(f"Coinbase API error: {data}")

    data.sort(key=lambda x: x[0])
    data = data[-limit:]

    candles = []
    closes = []

    for c in data:
        ts = int(c[0])
        candles.append({
            "time": datetime.fromtimestamp(ts).strftime("%H:%M"),
            "open": float(c[3]),
            "high": float(c[2]),
            "low": float(c[1]),
            "close": float(c[4]),
        })
        closes.append(float(c[4]))

    return candles, closes


def compute_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0

    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0)
    losses = -np.minimum(deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def confidence_score(rsi, trend_strength, momentum):
    score = 0

    if rsi < 30 or rsi > 70:
        score += 35
    elif 35 <= rsi <= 65:
        score -= 10

    score += min(abs(momentum) * 2000, 25)
    score += min(trend_strength * 2000, 25)

    return max(0, min(100, int(score)))


def atomic_write_json(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# ✅ SAFE TEST MESSAGE (after function exists)
if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    send_telegram("✅ BTC AI bot is live and can send alerts.")

# =========================
# MAIN LOOP
# =========================
async def main():
    last_alert = 0
    print("✅ BTC Alert Engine Running (AI-Filtered • Short-Term)")

    while True:
        try:
            candles, closes = fetch_candles(limit=60)
            price = closes[-1]
            rsi = compute_rsi(closes)

            momentum = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 6 else 0
            trend_strength = abs(momentum)

            trend = "WAIT"
            state = "WAIT"

            if rsi < 30 and momentum > 0:
                trend = state = "BUY"
            elif rsi > 70 and momentum < 0:
                trend = state = "SELL"

            confidence = confidence_score(rsi, trend_strength, momentum)

            atomic_write_json(STATE_FILE, {
                "price": round(price, 2),
                "rsi": round(rsi, 1),
                "trend": trend,
                "state": state,
                "confidence": confidence,
                "time": datetime.now().strftime("%H:%M:%S"),
                "candles": candles[-30:],
                "notes": f"src=Coinbase • momentum={momentum:.5f}",
                "error": "",
            })

            now = time.time()
            if state in ["BUY", "SELL"] and confidence >= MIN_CONFIDENCE and now - last_alert > ALERT_COOLDOWN:
                send_telegram(
                    f"📢 BTC {state}\n"
                    f"Price: ${price:,.2f}\n"
                    f"RSI(1m): {round(rsi,1)}\n"
                    f"Confidence: {confidence}%"
                )
                last_alert = now

        except Exception as e:
            atomic_write_json(STATE_FILE, {
                "price": 0,
                "rsi": 0,
                "trend": "WAIT",
                "state": "WAIT",
                "confidence": 0,
                "time": datetime.now().strftime("%H:%M:%S"),
                "candles": [],
                "notes": "",
                "error": str(e),
            })

        await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
