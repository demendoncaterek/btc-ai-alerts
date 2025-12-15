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
PRODUCT = "BTC-USD"
GRANULARITY = 60
STATE_FILE = "btc_state.json"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

ALERT_COOLDOWN = 300
MIN_CONFIDENCE = 70

HEARTBEAT_INTERVAL = 6 * 60 * 60  # 6 hours
COINBASE_BASE = "https://api.exchange.coinbase.com"

# =========================
# TELEGRAM HELPERS
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


def fetch_telegram_commands(offset=None):
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
            params={"timeout": 10, "offset": offset},
            timeout=15,
        )
        return resp.json().get("result", [])
    except:
        return []


# =========================
# MARKET HELPERS
# =========================
def fetch_candles(limit=60):
    resp = requests.get(
        f"{COINBASE_BASE}/products/{PRODUCT}/candles",
        params={"granularity": GRANULARITY},
        headers={"Accept": "application/json"},
        timeout=10,
    )

    data = resp.json()
    data.sort(key=lambda x: x[0])
    data = data[-limit:]

    candles = []
    closes = []

    for c in data:
        candles.append({
            "time": datetime.fromtimestamp(c[0]).strftime("%H:%M"),
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
    score += min(abs(momentum) * 2000, 25)
    score += min(trend_strength * 2000, 25)
    return max(0, min(100, int(score)))


def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# =========================
# STARTUP MESSAGE
# =========================
if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    send_telegram("✅ BTC AI bot is live and can send alerts.")


# =========================
# MAIN LOOP
# =========================
async def main():
    last_alert = 0
    last_heartbeat = 0
    last_update_id = None

    print("✅ BTC Alert Engine Running")

    while True:
        try:
            candles, closes = fetch_candles()
            price = closes[-1]
            rsi = compute_rsi(closes)

            momentum = (closes[-1] - closes[-5]) / closes[-5]
            trend_strength = abs(momentum)

            trend = state = "WAIT"
            if rsi < 30 and momentum > 0:
                trend = state = "BUY"
            elif rsi > 70 and momentum < 0:
                trend = state = "SELL"

            confidence = confidence_score(rsi, trend_strength, momentum)

            state_data = {
                "price": round(price, 2),
                "rsi": round(rsi, 1),
                "trend": trend,
                "state": state,
                "confidence": confidence,
                "time": datetime.now().strftime("%H:%M:%S"),
                "candles": candles[-30:],
                "notes": f"src=Coinbase • momentum={momentum:.5f}",
                "error": "",
            }

            atomic_write_json(STATE_FILE, state_data)

            now = time.time()

            # 🚨 Alerts
            if state in ["BUY", "SELL"] and confidence >= MIN_CONFIDENCE and now - last_alert > ALERT_COOLDOWN:
                send_telegram(
                    f"📢 BTC {state}\n"
                    f"Price: ${price:,.2f}\n"
                    f"RSI(1m): {round(rsi,1)}\n"
                    f"Confidence: {confidence}%"
                )
                last_alert = now

            # 🫀 Heartbeat
            if now - last_heartbeat > HEARTBEAT_INTERVAL:
                send_telegram(
                    f"🫀 BTC AI heartbeat\n"
                    f"Price: ${price:,.2f}\n"
                    f"RSI(1m): {round(rsi,1)}"
                )
                last_heartbeat = now

            # 💬 Telegram /status command
            updates = fetch_telegram_commands(last_update_id)
            for u in updates:
                last_update_id = u["update_id"] + 1
                msg = u.get("message", {}).get("text", "")

                if msg.strip().lower() == "/status":
                    send_telegram(
                        "🧠 BTC AI Status\n"
                        "Status: Running\n"
                        f"Price: ${price:,.2f}\n"
                        f"RSI(1m): {round(rsi,1)}\n"
                        f"Trend: {trend}\n"
                        f"Confidence: {confidence}%\n"
                        f"Last update: {state_data['time']}"
                    )

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
