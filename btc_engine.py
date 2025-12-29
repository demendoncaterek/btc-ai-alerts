# btc_engine.py
import os
import time
import math
import json
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple

import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

app = FastAPI(title="BTC Engine", version="3.0")

# ============================================================
# ENV CONFIG (Railway Variables)
# ============================================================
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")  # Binance symbol
EXEC_INTERVAL = os.getenv("EXEC_INTERVAL", "15m")  # execution timeframe
POLL_SEC = int(os.getenv("ENGINE_LOOP_SEC", os.getenv("POLL_SEC", "10")))

PAPER_START_USD = float(os.getenv("PAPER_START_USD", os.getenv("START_EQUITY", "250")))
RISK_PCT = float(os.getenv("RISK_PCT", "0.01"))  # equity fraction risked per trade baseline
MAX_LEVERAGE = float(os.getenv("MAX_LEVERAGE", "1.0"))  # paper sizing only; not real leverage
FEE_BPS = float(os.getenv("FEE_BPS", "6"))  # approx fees/slippage in basis points per side

# Adaptive decision threshold
MIN_CONF_DEFAULT = float(os.getenv("MIN_CONF_DEFAULT", "0.55"))
MIN_CONF_FLOOR = float(os.getenv("MIN_CONF_FLOOR", "0.45"))
MIN_CONF_CAP = float(os.getenv("MIN_CONF_CAP", "0.75"))
EXPLORATION_RATE = float(os.getenv("EXPLORATION_RATE", "0.05"))  # chance to take borderline trades

# Risk / exit config
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.6"))
ATR_MULT_TP = float(os.getenv("ATR_MULT_TP", "3.0"))

# Partial take profits (R multiples)
TP1_R = float(os.getenv("TP1_R", "1.0"))
TP2_R = float(os.getenv("TP2_R", "2.0"))
TP3_R = float(os.getenv("TP3_R", "3.0"))
TP1_PCT = float(os.getenv("TP1_PCT", "0.40"))  # sell 40% at TP1
TP2_PCT = float(os.getenv("TP2_PCT", "0.35"))  # sell 35% at TP2
TP3_PCT = float(os.getenv("TP3_PCT", "0.25"))  # sell rest at TP3

TRAIL_AFTER_R = float(os.getenv("TRAIL_AFTER_R", "1.0"))  # start trailing after +1R
TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "1.2"))  # trail distance in ATR

# Time exits / cadence
MAX_HOLD_MIN = int(os.getenv("MAX_HOLD_MIN", "180"))  # force close after N minutes
MIN_HOLD_MIN = int(os.getenv("MIN_HOLD_MIN", "5"))    # don't immediately flip
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", "3"))    # wait after close before next entry
MAX_TRADES_PER_HOUR = int(os.getenv("MAX_TRADES_PER_HOUR", "6"))

# Data
KLINES_LIMIT_EXEC = int(os.getenv("KLINES_LIMIT_EXEC", "240"))  # for 15m indicators
KLINES_LIMIT_BIAS = int(os.getenv("KLINES_LIMIT_BIAS", "240"))  # for 1h/4h bias

BINANCE_BASE = os.getenv("BINANCE_BASE", "https://api.binance.com")

# ============================================================
# HTTP SESSION (robust)
# ============================================================
_session = requests.Session()
_session.headers.update({"User-Agent": "btc-engine/3.0"})
_session_timeout = 12

def _now() -> datetime:
    return datetime.now(timezone.utc)

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _bps_to_frac(bps: float) -> float:
    return bps / 10000.0

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

# ============================================================
# MARKET DATA
# ============================================================
def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": int(limit)}
    r = _session.get(url, params=params, timeout=_session_timeout)
    r.raise_for_status()
    data = r.json()

    # Kline array format:
    # [
    #   0 open_time, 1 open, 2 high, 3 low, 4 close, 5 volume, 6 close_time, ...
    # ]
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "tbbav", "tbqav", "ignore"
    ])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill").fillna(50)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    line = fast_ema - slow_ema
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().fillna(method="bfill")

# ============================================================
# STRATEGY LAYER
#   - We compute 3 strategy "scores" in [0..1]
#   - We weight them based on their recent realized performance
#   - We combine with HTF bias (1h/4h) to decide entries
# ============================================================
@dataclass
class StrategyStats:
    name: str
    trades: int = 0
    wins: int = 0
    pnl_sum: float = 0.0

    def win_rate(self) -> float:
        return (self.wins / self.trades) if self.trades > 0 else 0.0

    def avg_pnl(self) -> float:
        return (self.pnl_sum / self.trades) if self.trades > 0 else 0.0

# initialize with neutral weights
strategy_stats: Dict[str, StrategyStats] = {
    "dip": StrategyStats("dip"),
    "pullback": StrategyStats("pullback"),
    "breakout": StrategyStats("breakout"),
}

def strategy_weights() -> Dict[str, float]:
    """
    Convert performance into weights.
    - start equal
    - small boost for better win rate & avg pnl
    - keep weights bounded and normalized
    """
    base = {"dip": 1.0, "pullback": 1.0, "breakout": 1.0}
    raw = {}
    for k, st in strategy_stats.items():
        # A smooth scoring: win_rate + scaled avg_pnl
        wr = st.win_rate()
        ap = st.avg_pnl()
        # convert ap into a small score; assume pnl in dollars is small vs equity
        ap_score = _clamp(ap / 10.0, -0.3, 0.3)  # bounded
        perf = (wr - 0.5) + ap_score  # centered around 0
        raw[k] = base[k] * (1.0 + _clamp(perf, -0.4, 0.6))

    s = sum(raw.values())
    if s <= 0:
        return {"dip": 1/3, "pullback": 1/3, "breakout": 1/3}
    return {k: v / s for k, v in raw.items()}

def htf_bias(symbol: str) -> Dict[str, Any]:
    """
    Determine higher time frame bias from 1h and 4h:
    - trend: close above EMA200 => bullish
    - momentum: MACD histogram sign
    """
    out = {"bias": "NEUTRAL", "score": 0.5}
    try:
        df1h = fetch_klines(symbol, "1h", KLINES_LIMIT_BIAS)
        df4h = fetch_klines(symbol, "4h", KLINES_LIMIT_BIAS)
        c1 = df1h["close"]
        c4 = df4h["close"]

        ema1 = ema(c1, 200).iloc[-1]
        ema4 = ema(c4, 200).iloc[-1]
        macd1, sig1, hist1 = macd(c1)
        macd4, sig4, hist4 = macd(c4)

        price1 = c1.iloc[-1]
        price4 = c4.iloc[-1]

        trend_score = 0.0
        trend_score += 0.5 if price1 > ema1 else 0.0
        trend_score += 0.5 if price4 > ema4 else 0.0

        mom_score = 0.0
        mom_score += 0.5 if hist1.iloc[-1] > 0 else 0.0
        mom_score += 0.5 if hist4.iloc[-1] > 0 else 0.0

        score = 0.5 * trend_score + 0.5 * mom_score  # 0..1
        if score >= 0.65:
            bias = "BULL"
        elif score <= 0.35:
            bias = "BEAR"
        else:
            bias = "NEUTRAL"

        out.update({"bias": bias, "score": float(score), "ema1h200": float(ema1), "ema4h200": float(ema4)})
        return out
    except Exception as e:
        out["error"] = str(e)
        return out

def score_dip(df: pd.DataFrame) -> float:
    """
    Dip mean-reversion: oversold + stretched below EMA.
    """
    c = df["close"]
    r = rsi(c, 14).iloc[-1]
    e = ema(c, 50).iloc[-1]
    price = c.iloc[-1]
    stretch = (e - price) / max(price, 1e-9)  # how far below EMA

    s = 0.0
    # RSI
    if r < 20: s += 0.55
    elif r < 25: s += 0.45
    elif r < 30: s += 0.30
    elif r < 35: s += 0.18

    # stretch below EMA
    if stretch > 0.02: s += 0.35
    elif stretch > 0.012: s += 0.25
    elif stretch > 0.007: s += 0.15

    return float(_clamp(s, 0.0, 1.0))

def score_pullback(df: pd.DataFrame) -> float:
    """
    Trend pullback: trend up + price pulled back to EMA zone + RSI reclaims.
    """
    c = df["close"]
    e20 = ema(c, 20).iloc[-1]
    e50 = ema(c, 50).iloc[-1]
    price = c.iloc[-1]
    r = rsi(c, 14).iloc[-1]
    macd_line, macd_sig, macd_hist = macd(c)
    h = macd_hist.iloc[-1]

    s = 0.0
    # trend
    if e20 > e50:
        s += 0.25
        # pullback proximity
        dist = abs(price - e20) / max(price, 1e-9)
        if dist < 0.004: s += 0.30
        elif dist < 0.008: s += 0.20
        elif dist < 0.012: s += 0.12

        # RSI in “healthy” zone
        if 45 <= r <= 60: s += 0.20
        elif 40 <= r <= 65: s += 0.12

        # MACD not collapsing
        if h > -0.5 * abs(h):  # basically don't punish too hard
            s += 0.15
        if h > 0:
            s += 0.10

    return float(_clamp(s, 0.0, 1.0))

def score_breakout(df: pd.DataFrame) -> float:
    """
    Momentum breakout: compression then expansion, price above EMA, MACD positive.
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]
    price = c.iloc[-1]
    e50 = ema(c, 50).iloc[-1]
    macd_line, macd_sig, macd_hist = macd(c)
    hist = macd_hist.iloc[-1]
    a = atr(df, 14).iloc[-1]

    # recent range / compression
    recent = df.tail(40)
    range_pct = (recent["high"].max() - recent["low"].min()) / max(price, 1e-9)
    last_range = (df["high"].iloc[-1] - df["low"].iloc[-1]) / max(price, 1e-9)

    s = 0.0
    if price > e50:
        s += 0.20

    if hist > 0:
        s += 0.25
    elif hist > -0.0001:
        s += 0.12

    # expansion candle
    if a > 0:
        if (df["high"].iloc[-1] - df["low"].iloc[-1]) > 1.2 * a:
            s += 0.25
        elif (df["high"].iloc[-1] - df["low"].iloc[-1]) > 0.9 * a:
            s += 0.15

    # compression then pop (rough heuristic)
    if range_pct < 0.03:
        s += 0.20
    if last_range > 0.004:
        s += 0.10

    return float(_clamp(s, 0.0, 1.0))

def combined_confidence(exec_df: pd.DataFrame, bias: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a combined confidence score with strategy weighting + HTF bias gating.
    """
    s_dip = score_dip(exec_df)
    s_pull = score_pullback(exec_df)
    s_break = score_breakout(exec_df)

    w = strategy_weights()
    raw = (
        w["dip"] * s_dip +
        w["pullback"] * s_pull +
        w["breakout"] * s_break
    )

    # Bias adjustment:
    # If BEAR bias, reduce long confidence
    bias_score = float(bias.get("score", 0.5))
    bias_label = bias.get("bias", "NEUTRAL")

    if bias_label == "BULL":
        adj = 0.92 + 0.16 * bias_score   # ~0.92..1.08
    elif bias_label == "BEAR":
        adj = 0.72 + 0.18 * bias_score   # ~0.72..0.90
    else:
        adj = 0.90 + 0.20 * bias_score   # ~0.90..1.10

    conf = _clamp(raw * adj, 0.0, 1.0)

    # Choose the "dominant" strategy for journaling
    strat_scores = {"dip": s_dip, "pullback": s_pull, "breakout": s_break}
    dominant = max(strat_scores, key=strat_scores.get)

    return {
        "confidence": float(conf),
        "dominant": dominant,
        "strategy_scores": {k: float(v) for k, v in strat_scores.items()},
        "weights": {k: float(v) for k, v in w.items()},
        "bias": bias,
    }

# ============================================================
# PAPER POSITION MODEL
# ============================================================
@dataclass
class Position:
    entry: float
    size: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    remaining: float
    opened_at: str
    strategy: str
    confidence: float
    atr: float
    r: float  # risk per unit (entry - sl)
    trail_active: bool = False
    trail_sl: Optional[float] = None
    tp1_done: bool = False
    tp2_done: bool = False
    tp3_done: bool = False

@dataclass
class Trade:
    entry: float
    exit: float
    size: float
    pnl: float
    opened_at: str
    closed_at: str
    strategy: str
    confidence: float
    reason: str

# ============================================================
# STATE
# ============================================================
equity = PAPER_START_USD
pos: Optional[Position] = None
trades: List[Trade] = []

confidence_floor = MIN_CONF_DEFAULT
last_close_ts: Optional[datetime] = None
recent_trade_times: List[datetime] = []  # to enforce MAX_TRADES_PER_HOUR

last_tick: Optional[Dict[str, Any]] = None

def _fee_cost(notional: float) -> float:
    # approximate round-trip will be applied at close, but we also apply partial fees
    return notional * _bps_to_frac(FEE_BPS)

def _can_trade_now(now: datetime) -> Tuple[bool, str]:
    global last_close_ts, recent_trade_times

    # cooldown after close
    if last_close_ts is not None:
        if now < last_close_ts + timedelta(minutes=COOLDOWN_MIN):
            return False, "cooldown"

    # cap trades per hour
    horizon = now - timedelta(hours=1)
    recent_trade_times = [t for t in recent_trade_times if t >= horizon]
    if len(recent_trade_times) >= MAX_TRADES_PER_HOUR:
        return False, "rate_limit"

    return True, "ok"

def adaptive_threshold_update() -> float:
    """
    Adjust threshold based on recent win rate (last 20 closed trades).
    """
    global confidence_floor
    if len(trades) < 6:
        return confidence_floor

    last = trades[-20:]
    wins = sum(1 for t in last if t.pnl > 0)
    wr = wins / len(last)

    if wr > 0.60:
        confidence_floor = min(confidence_floor + 0.01, MIN_CONF_CAP)
    elif wr < 0.45:
        confidence_floor = max(confidence_floor - 0.01, MIN_CONF_FLOOR)

    return confidence_floor

def update_strategy_stats(closed: Trade):
    st = strategy_stats.get(closed.strategy)
    if not st:
        return
    st.trades += 1
    if closed.pnl > 0:
        st.wins += 1
    st.pnl_sum += closed.pnl

# ============================================================
# CORE ENGINE TICK
# ============================================================
def trade_tick(force: bool = False) -> Dict[str, Any]:
    """
    One evaluation step.
    - updates pos / equity when exits occur
    - may open a new position when signals good
    """
    global equity, pos, last_close_ts, recent_trade_times, last_tick

    now = _now()

    # 1) fetch execution candles
    try:
        df = fetch_klines(SYMBOL, EXEC_INTERVAL, KLINES_LIMIT_EXEC)
    except Exception as e:
        last_tick = {"ok": False, "error": f"fetch_klines failed: {e}"}
        return last_tick

    price = float(df["close"].iloc[-1])
    a = float(atr(df, 14).iloc[-1])
    if not math.isfinite(a) or a <= 0:
        a = max(price * 0.002, 1.0)

    # 2) compute multi-timeframe bias + combined confidence
    bias = htf_bias(SYMBOL)
    combo = combined_confidence(df, bias)
    conf = float(combo["confidence"])
    threshold = adaptive_threshold_update()

    # exploration allows some borderline entries
    explore = (np.random.rand() < EXPLORATION_RATE)

    # 3) update open position (partials, trailing, time exit, SL/TP)
    event_log = []
    if pos is not None:
        opened_dt = datetime.fromisoformat(pos.opened_at)
        hold_min = (now - opened_dt).total_seconds() / 60.0

        # Build current "R" progress
        r_unit = max(pos.r, 1e-9)
        current_r = (price - pos.entry) / r_unit

        # Activate trailing after TRAIL_AFTER_R and after MIN_HOLD_MIN
        if (not pos.trail_active) and (current_r >= TRAIL_AFTER_R) and (hold_min >= MIN_HOLD_MIN):
            pos.trail_active = True
            pos.trail_sl = pos.sl  # start at original SL
            event_log.append("trail_activated")

        # Update trailing stop if active
        if pos.trail_active:
            # trailing stop follows price - TRAIL_ATR_MULT*ATR
            new_trail = price - TRAIL_ATR_MULT * pos.atr
            if pos.trail_sl is None:
                pos.trail_sl = new_trail
            else:
                pos.trail_sl = max(pos.trail_sl, new_trail)  # ratchet only upward

        # Partial take profits
        def do_partial(tag: str, pct: float, reason: str):
            nonlocal price
            global equity
            if pos is None:
                return
            sell_qty = pos.size * pct
            sell_qty = min(sell_qty, pos.remaining)
            if sell_qty <= 0:
                return

            # realized PnL for that slice
            pnl = (price - pos.entry) * sell_qty
            # fee for slice close (one side)
            fee = _fee_cost(abs(price * sell_qty))
            pnl -= fee
            equity += pnl
            pos.remaining -= sell_qty
            event_log.append(f"{tag}_partial")

        # hit TP1/TP2/TP3 as R targets, not fixed price
        if not pos.tp1_done and current_r >= TP1_R:
            do_partial("tp1", TP1_PCT, "tp1_partial")
            pos.tp1_done = True

        if not pos.tp2_done and current_r >= TP2_R:
            do_partial("tp2", TP2_PCT, "tp2_partial")
            pos.tp2_done = True

        if not pos.tp3_done and current_r >= TP3_R:
            # close whatever remains
            do_partial("tp3", 1.0, "tp3_final")
            pos.tp3_done = True

        # Determine active SL (trail or base)
        active_sl = pos.trail_sl if (pos.trail_active and pos.trail_sl is not None) else pos.sl

        # Stop loss hit
        stopped = (price <= active_sl)

        # Time exit
        time_exit = (hold_min >= MAX_HOLD_MIN)

        # Close if stopped OR time_exit OR fully scaled out
        if stopped or time_exit or pos.remaining <= 1e-12:
            if pos.remaining > 1e-12:
                # Close remainder
                pnl = (price - pos.entry) * pos.remaining
                fee = _fee_cost(abs(price * pos.remaining))
                pnl -= fee
                equity += pnl

            # one-time entry fee approximation (entry side), charged at close
            entry_fee = _fee_cost(abs(pos.entry * pos.size))
            equity -= entry_fee

            reason = "stop" if stopped else ("time_exit" if time_exit else "scaled_out")
            closed = Trade(
                entry=pos.entry,
                exit=price,
                size=pos.size,
                pnl=(equity - 0.0),  # placeholder, we will recompute below
                opened_at=pos.opened_at,
                closed_at=_iso(now),
                strategy=pos.strategy,
                confidence=pos.confidence,
                reason=reason
            )

            # recompute pnl properly (from notional)
            # We track pnl by summing equity changes is messy; compute approximate:
            gross = (price - pos.entry) * pos.size
            # approximate total fees: entry + exit
            total_fee = _fee_cost(abs(pos.entry * pos.size)) + _fee_cost(abs(price * pos.size))
            closed.pnl = gross - total_fee

            trades.append(closed)
            update_strategy_stats(closed)

            last_close_ts = now
            recent_trade_times.append(now)

            event_log.append(f"closed:{reason}")
            pos = None

    # 4) Open new position if none
    opened = False
    open_reason = "none"
    if pos is None:
        can, why = _can_trade_now(now)
        if can or force:
            # Confidence gating:
            # - require threshold unless explore
            # - if HTF bias is BEAR, require stronger confidence
            bias_label = combo["bias"].get("bias", "NEUTRAL")
            bias_penalty = 0.08 if bias_label == "BEAR" else 0.0
            effective_threshold = _clamp(threshold + bias_penalty, MIN_CONF_FLOOR, MIN_CONF_CAP)

            # "Opportunity" trigger: confidence OR exploration
            take = (conf >= effective_threshold) or explore or force

            if take:
                # position sizing:
                # risk dollars scales with confidence (more confident -> closer to baseline*1.4)
                risk_dollars = equity * RISK_PCT * (0.7 + 0.7 * conf)
                risk_dollars = max(risk_dollars, equity * 0.002)  # don't go microscopic
                risk_dollars = min(risk_dollars, equity * 0.03)   # don't go insane

                # stop distance from ATR
                sl_dist = max(a * ATR_MULT_SL, price * 0.0015)  # ensure minimum
                sl = price - sl_dist
                r_unit = price - sl  # risk per unit

                # compute size such that risk ~= risk_dollars
                size = risk_dollars / max(r_unit, 1e-9)

                # cap notional by equity*MAX_LEVERAGE
                notional = price * size
                max_notional = equity * MAX_LEVERAGE
                if notional > max_notional:
                    size = max_notional / max(price, 1e-9)

                # take profit ladders from R
                tp1 = price + TP1_R * r_unit
                tp2 = price + TP2_R * r_unit
                tp3 = price + TP3_R * r_unit

                strategy = combo["dominant"]

                pos = Position(
                    entry=price,
                    size=float(size),
                    sl=float(sl),
                    tp1=float(tp1),
                    tp2=float(tp2),
                    tp3=float(tp3),
                    remaining=float(size),
                    opened_at=_iso(now),
                    strategy=strategy,
                    confidence=float(conf),
                    atr=float(a),
                    r=float(r_unit),
                )

                opened = True
                open_reason = "force" if force else ("explore" if explore and conf < effective_threshold else "signal")
                recent_trade_times.append(now)
        else:
            open_reason = why

    # 5) Build response state
    # perf metrics
    closed_trades = trades
    wins = sum(1 for t in closed_trades if t.pnl > 0)
    total = len(closed_trades)
    win_rate = (wins / total) if total else 0.0

    # profit factor
    gains = sum(t.pnl for t in closed_trades if t.pnl > 0)
    losses = -sum(t.pnl for t in closed_trades if t.pnl < 0)
    profit_factor = (gains / losses) if losses > 1e-9 else (gains if gains > 0 else 0.0)

    # max drawdown from equity curve (closed-trade based)
    eq_curve = [PAPER_START_USD]
    running = PAPER_START_USD
    for t in closed_trades:
        running += t.pnl
        eq_curve.append(running)
    peak = -1e9
    max_dd = 0.0
    for x in eq_curve:
        peak = max(peak, x)
        if peak > 0:
            dd = (peak - x) / peak
            max_dd = max(max_dd, dd)

    # include some last indicator values for UI "Reason/Levels"
    c = df["close"]
    rsi15 = float(rsi(c, 14).iloc[-1])
    macd_line, macd_sig, macd_hist = macd(c)
    macd15 = float(macd_hist.iloc[-1])
    ema50 = float(ema(c, 50).iloc[-1])
    atr15 = float(a)

    out = {
        "ok": True,
        "ts": _iso(now),
        "symbol": SYMBOL,
        "interval": EXEC_INTERVAL,
        "price": round(price, 2),
        "signal": "LONG" if pos else "WAIT",
        "confidence": round(conf * 100, 2),
        "threshold": round(threshold * 100, 2),
        "effective_threshold": round((_clamp(threshold + (0.08 if combo["bias"].get("bias") == "BEAR" else 0.0),
                                             MIN_CONF_FLOOR, MIN_CONF_CAP)) * 100, 2),
        "equity": round(equity, 2),

        "opened": opened,
        "open_reason": open_reason,
        "events": event_log,

        "position": asdict(pos) if pos else None,

        "reason_levels": {
            "rsi15": rsi15,
            "macd15": macd15,
            "ema50": ema50,
            "atr15": atr15,
            "strategy_scores": combo["strategy_scores"],
            "strategy_weights": combo["weights"],
            "htf_bias": combo["bias"],
        },

        "performance": {
            "trades_closed": total,
            "win_rate": round(win_rate * 100, 2),
            "profit_factor": round(float(profit_factor), 2),
            "max_dd": round(max_dd * 100, 2),
        }
    }

    last_tick = out
    return out

# ============================================================
# BACKGROUND LOOP (so it trades even if UI isn’t open)
# ============================================================
async def engine_loop():
    while True:
        try:
            trade_tick()
        except Exception:
            # never let the loop die
            pass
        await asyncio.sleep(POLL_SEC)

@app.on_event("startup")
async def startup_event():
    # start background engine
    app.state.engine_task = asyncio.create_task(engine_loop())

# ============================================================
# API
# ============================================================
@app.get("/health")
def health():
    return {"ok": True, "ts": _iso(_now())}

@app.get("/state")
def state():
    return JSONResponse(trade_tick())

@app.get("/trades")
def get_trades(limit: int = Query(200, ge=1, le=500)):
    # most recent first
    data = [asdict(t) for t in trades[-limit:]][::-1]
    return {"count": len(data), "trades": data}

@app.post("/calibrate")
def calibrate():
    """
    Force a tick & allow opening even if cooldown/rate-limit would block.
    Useful to test that it can enter/exit.
    """
    picked = trade_tick(force=True)
    return {"status": "ok", "picked": picked}

@app.post("/reset")
def reset():
    """
    Reset paper account & journal.
    """
    global equity, pos, trades, confidence_floor, last_close_ts, recent_trade_times
    equity = PAPER_START_USD
    pos = None
    trades = []
    confidence_floor = MIN_CONF_DEFAULT
    last_close_ts = None
    recent_trade_times = []
    # reset strategy stats
    for k in strategy_stats.keys():
        strategy_stats[k] = StrategyStats(k)
    return {"ok": True, "equity": equity, "ts": _iso(_now())}
