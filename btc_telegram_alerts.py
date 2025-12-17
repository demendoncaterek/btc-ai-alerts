import time
import math
import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ================= CONFIG =================
SYMBOL = "BTC-USD"
COINBASE_URL = "https://api.exchange.coinbase.com"

# Supported Coinbase granularities: 60, 300, 900, 3600, 21600, 86400
G_5M = 300
G_15M = 900
G_1H = 3600
G_6H = 21600

RSI_PERIOD = 14
ATR_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200

# Risk model (suggestions only)
ATR_SL_MULT = 1.5
ATR_TP_MULT = 2.5

# Signal thresholds
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

# Backtest defaults (Coinbase candle endpoint returns max ~300 per call)
BT_DEFAULT_GRAN = G_15M
BT_DEFAULT_BARS = 300
# =========================================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= UTIL =================
def fetch_candles(granularity: int, limit: int = 300) -> pd.DataFrame:
    """
    Coinbase candles: [ time, low, high, open, close, volume ]
    Returns sorted df with columns: time, low, high, open, close, volume
    """
    url = f"{COINBASE_URL}/products/{SYMBOL}/candles"
    params = {"granularity": granularity}
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()

    df = pd.DataFrame(
        r.json(),
        columns=["time", "low", "high", "open", "close", "volume"],
    )
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
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period).mean()
    return atr

def ema_trend(close: pd.Series) -> str:
    ema_fast = close.ewm(span=EMA_FAST).mean()
    ema_slow = close.ewm(span=EMA_SLOW).mean()
    return "UP" if float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1]) else "DOWN"

def pct(x: float) -> float:
    return round(float(x) * 100.0, 2)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def find_swings(series: pd.Series, lookback: int = 5):
    """
    Very simple swing finder:
    swing low if it's the min in a window; swing high if max in window.
    Returns indices of last two swing lows and highs.
    """
    if len(series) < (lookback * 2 + 5):
        return [], []

    lows = []
    highs = []
    arr = series.values

    for i in range(lookback, len(arr) - lookback):
        window = arr[i - lookback : i + lookback + 1]
        if arr[i] == np.min(window):
            lows.append(i)
        if arr[i] == np.max(window):
            highs.append(i)

    return lows[-2:], highs[-2:]

def detect_rsi_divergence(df: pd.DataFrame) -> dict:
    """
    Bullish divergence: price lower low + RSI higher low
    Bearish divergence: price higher high + RSI lower high
    Uses last two swing points.
    """
    out = {
        "bullish": False,
        "bearish": False,
        "detail": ""
    }

    if df is None or df.empty or len(df) < 60:
        return out

    rsi = compute_rsi(df["close"], RSI_PERIOD)
    if rsi.isna().all():
        return out

    lows_idx, highs_idx = find_swings(df["close"], lookback=5)
    # bullish divergence on swing lows
    if len(lows_idx) == 2:
        i1, i2 = lows_idx[0], lows_idx[1]
        p1, p2 = float(df["close"].iloc[i1]), float(df["close"].iloc[i2])
        r1, r2 = float(rsi.iloc[i1]), float(rsi.iloc[i2])

        if p2 < p1 and r2 > r1:
            out["bullish"] = True
            out["detail"] += f"Bull div: price LL ({p1:.0f}->{p2:.0f}) + RSI HL ({r1:.1f}->{r2:.1f}). "

    # bearish divergence on swing highs
    if len(highs_idx) == 2:
        i1, i2 = highs_idx[0], highs_idx[1]
        p1, p2 = float(df["close"].iloc[i1]), float(df["close"].iloc[i2])
        r1, r2 = float(rsi.iloc[i1]), float(rsi.iloc[i2])

        if p2 > p1 and r2 < r1:
            out["bearish"] = True
            out["detail"] += f"Bear div: price HH ({p1:.0f}->{p2:.0f}) + RSI LH ({r1:.1f}->{r2:.1f}). "

    return out

def momentum_score(close: pd.Series, n: int = 6) -> float:
    """
    Simple momentum: slope-ish using pct change over n bars.
    """
    if len(close) < n + 1:
        return 0.0
    return float((close.iloc[-1] / close.iloc[-1 - n]) - 1.0)

# ================= CORE STRATEGY =================
def build_decision(c5: pd.DataFrame, c1h: pd.DataFrame, c6h: pd.DataFrame):
    price = float(c5["close"].iloc[-1])

    rsi5 = compute_rsi(c5["close"], RSI_PERIOD)
    rsi1h = compute_rsi(c1h["close"], RSI_PERIOD)
    atr = compute_atr(c5, ATR_PERIOD)

    rsi_5m = float(rsi5.iloc[-1]) if not math.isnan(float(rsi5.iloc[-1])) else None
    rsi_1h = float(rsi1h.iloc[-1]) if not math.isnan(float(rsi1h.iloc[-1])) else None
    atr_now = float(atr.iloc[-1]) if not math.isnan(float(atr.iloc[-1])) else None

    trend_1h = ema_trend(c1h["close"])
    trend_6h = ema_trend(c6h["close"])

    if trend_1h == trend_6h:
        htf_bias = trend_1h
    else:
        htf_bias = "NEUTRAL"

    # divergence on 5m
    div = detect_rsi_divergence(c5)

    # momentum (5m)
    mom = momentum_score(c5["close"], n=6)  # ~30m momentum

    # ---- Confidence breakdown (0-100, but we’ll output percent) ----
    conf_parts = {}

    # Trend alignment
    conf_parts["htf_alignment"] = 30 if htf_bias in ["UP", "DOWN"] else 10

    # RSI quality
    if rsi_5m is None:
        conf_parts["rsi_signal"] = 0
    else:
        if htf_bias == "UP" and rsi_5m < RSI_OVERSOLD:
            conf_parts["rsi_signal"] = 25
        elif htf_bias == "DOWN" and rsi_5m > RSI_OVERBOUGHT:
            conf_parts["rsi_signal"] = 25
        else:
            conf_parts["rsi_signal"] = 10

    # Momentum confirmation
    if htf_bias == "UP":
        conf_parts["momentum"] = 20 if mom > 0 else 8
    elif htf_bias == "DOWN":
        conf_parts["momentum"] = 20 if mom < 0 else 8
    else:
        conf_parts["momentum"] = 8

    # Divergence bonus (counter-trend warning / reversal hint)
    div_bonus = 0
    if div["bullish"]:
        div_bonus += 10
    if div["bearish"]:
        div_bonus += 10
    conf_parts["divergence_bonus"] = div_bonus

    confidence_raw = sum(conf_parts.values())
    confidence = clamp(confidence_raw, 1, 95)

    # ---- Signal rules (disciplined) ----
    signal = "WAIT"
    reason = "No high-probability setup."

    # BUY: HTF UP + RSI oversold + momentum flips positive or bullish divergence
    if htf_bias == "UP" and rsi_5m is not None:
        if (rsi_5m < RSI_OVERSOLD) and (mom > 0 or div["bullish"]):
            signal = "BUY"
            reason = "HTF UP + RSI(5m) oversold + confirmation (momentum/divergence)."

    # SELL: HTF DOWN + RSI overbought + momentum flips negative or bearish divergence
    if htf_bias == "DOWN" and rsi_5m is not None:
        if (rsi_5m > RSI_OVERBOUGHT) and (mom < 0 or div["bearish"]):
            signal = "SELL"
            reason = "HTF DOWN + RSI(5m) overbought + confirmation (momentum/divergence)."

    # ---- ATR-based SL/TP suggestions (only if we have ATR) ----
    sl_price = None
    tp_price = None
    if atr_now is not None:
        if signal == "BUY":
            sl_price = price - ATR_SL_MULT * atr_now
            tp_price = price + ATR_TP_MULT * atr_now
        elif signal == "SELL":
            sl_price = price + ATR_SL_MULT * atr_now
            tp_price = price - ATR_TP_MULT * atr_now

    # helpful “watch” messages (does NOT override signal)
    peak_watch = False
    dip_watch = False
    if rsi_5m is not None:
        peak_watch = (rsi_5m > 60 and mom > 0)  # “getting extended”
        dip_watch = (rsi_5m < 40 and mom < 0)   # “falling hard”

    return {
        "price": price,
        "signal": signal,
        "confidence": round(confidence, 2),
        "reason": reason,
        "rsi_5m": None if rsi_5m is None else round(rsi_5m, 2),
        "rsi_1h": None if rsi_1h is None else round(rsi_1h, 2),
        "trend_1h": trend_1h,
        "trend_6h": trend_6h,
        "htf_bias": htf_bias,
        "momentum_30m": round(mom, 6),
        "divergence": div,
        "atr_5m": None if atr_now is None else round(atr_now, 2),
        "sl_price": None if sl_price is None else round(sl_price, 2),
        "tp_price": None if tp_price is None else round(tp_price, 2),
        "watch": {
            "peak_watch": peak_watch,
            "dip_watch": dip_watch,
        },
        "confidence_breakdown": conf_parts,
    }

# ================= BACKTEST =================
def backtest_simple(df: pd.DataFrame) -> dict:
    """
    Simple backtest on a single timeframe:
    - Uses EMA trend on the same df as HTF proxy (lightweight)
    - Entries are RSI extremes + momentum confirmation
    - Exits via ATR SL/TP
    This is NOT “perfect”, it’s a sanity-check + stats view.
    """
    if df is None or df.empty or len(df) < 80:
        return {"ok": False, "error": "Not enough bars for backtest."}

    close = df["close"]
    rsi = compute_rsi(close, RSI_PERIOD)
    atr = compute_atr(df, ATR_PERIOD)
    ema_fast = close.ewm(span=EMA_FAST).mean()
    ema_slow = close.ewm(span=EMA_SLOW).mean()

    trades = []
    equity = 1.0
    peak = 1.0
    dd_max = 0.0

    in_pos = False
    side = None
    entry = None
    sl = None
    tp = None
    entry_i = None

    for i in range(max(EMA_SLOW, RSI_PERIOD, ATR_PERIOD) + 5, len(df)):
        price = float(close.iloc[i])
        if np.isnan(rsi.iloc[i]) or np.isnan(atr.iloc[i]):
            continue

        trend = "UP" if ema_fast.iloc[i] > ema_slow.iloc[i] else "DOWN"
        mom = float((close.iloc[i] / close.iloc[i - 6]) - 1.0) if i >= 6 else 0.0
        r = float(rsi.iloc[i])
        a = float(atr.iloc[i])

        # Exit logic
        if in_pos:
            if side == "LONG":
                if price <= sl or price >= tp:
                    ret = (price / entry) - 1.0
                    equity *= (1.0 + ret)
                    trades.append(ret)
                    in_pos = False
            elif side == "SHORT":
                if price >= sl or price <= tp:
                    ret = (entry / price) - 1.0
                    equity *= (1.0 + ret)
                    trades.append(ret)
                    in_pos = False

        # Update drawdown
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        dd_max = max(dd_max, dd)

        # Entry logic (only if flat)
        if not in_pos:
            if trend == "UP" and r < RSI_OVERSOLD and (mom > 0):
                in_pos = True
                side = "LONG"
                entry = price
                sl = entry - ATR_SL_MULT * a
                tp = entry + ATR_TP_MULT * a
                entry_i = i
            elif trend == "DOWN" and r > RSI_OVERBOUGHT and (mom < 0):
                in_pos = True
                side = "SHORT"
                entry = price
                sl = entry + ATR_SL_MULT * a
                tp = entry - ATR_TP_MULT * a
                entry_i = i

    if not trades:
        return {
            "ok": True,
            "trades": 0,
            "winrate": 0,
            "expectancy": 0,
            "profit_factor": 0,
            "max_drawdown": pct(dd_max),
            "equity": round(equity, 4),
        }

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    winrate = len(wins) / len(trades)

    gross_win = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    expectancy = (sum(trades) / len(trades))

    return {
        "ok": True,
        "trades": len(trades),
        "winrate": pct(winrate),
        "expectancy": pct(expectancy),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "max_drawdown": pct(dd_max),
        "equity": round(equity, 4),
    }

# ================= API =================
@app.get("/health")
def health():
    return {"ok": True, "service": "btc-engine", "time": time.strftime("%H:%M:%S")}

@app.get("/state")
def state():
    try:
        c5 = fetch_candles(G_5M, limit=300)
        c1h = fetch_candles(G_1H, limit=300)
        c6h = fetch_candles(G_6H, limit=300)

        if c5.empty or c1h.empty or c6h.empty:
            return {"ok": False, "error": "Empty candle feed from Coinbase."}

        d = build_decision(c5, c1h, c6h)
        return {
            "ok": True,
            "time": time.strftime("%H:%M:%S"),
            "src": "Coinbase",
            **d,
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/explain")
def explain():
    """
    Convenience endpoint: same info as /state but grouped
    """
    s = state()
    if not s.get("ok"):
        return s

    return {
        "ok": True,
        "summary": {
            "price": s["price"],
            "signal": s["signal"],
            "confidence": s["confidence"],
            "reason": s["reason"],
        },
        "indicators": {
            "rsi_5m": s["rsi_5m"],
            "rsi_1h": s["rsi_1h"],
            "atr_5m": s["atr_5m"],
            "momentum_30m": s["momentum_30m"],
        },
        "trend": {
            "trend_1h": s["trend_1h"],
            "trend_6h": s["trend_6h"],
            "htf_bias": s["htf_bias"],
        },
        "risk": {
            "sl_price": s["sl_price"],
            "tp_price": s["tp_price"],
            "sl_mult_atr": ATR_SL_MULT,
            "tp_mult_atr": ATR_TP_MULT,
        },
        "divergence": s["divergence"],
        "watch": s["watch"],
        "confidence_breakdown": s["confidence_breakdown"],
    }

@app.get("/backtest")
def backtest(granularity: int = BT_DEFAULT_GRAN, bars: int = BT_DEFAULT_BARS):
    """
    Returns quick stats on the last N bars of the chosen granularity.
    Example:
      /backtest?granularity=900&bars=300  (15m)
    """
    try:
        bars = int(clamp(int(bars), 80, 300))
        df = fetch_candles(int(granularity), limit=bars)
        if df.empty:
            return {"ok": False, "error": "Empty candle feed from Coinbase."}

        stats = backtest_simple(df)
        return {
            "ok": True,
            "symbol": SYMBOL,
            "granularity": int(granularity),
            "bars": bars,
            "stats": stats,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
