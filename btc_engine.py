import os
import time
import math
import json
import uuid
import sqlite3
import threading
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# =========================
# Config (env)
# =========================
SYMBOL = os.getenv("SYMBOL", "BTC-USD")

# Timeframes in seconds supported by Coinbase Exchange candles API:
# 60, 300, 900, 3600, 21600, 86400
TF_5M = 300
TF_15M = 900
TF_1H = 3600

EXEC_TF = int(os.getenv("EXEC_TF", str(TF_15M)))  # execution timeframe (default 15m)
BIAS_TF = int(os.getenv("BIAS_TF", str(TF_1H)))   # bias timeframe (default 1h)
REFRESH_SEC = float(os.getenv("ENGINE_LOOP_SEC", "10"))  # engine loop

MIN_CONF_TRADE = float(os.getenv("MIN_CONF_TRADE", "0.60"))  # 0-1
PAPER_START_USD = float(os.getenv("PAPER_START_USD", "250"))

ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.8"))
ATR_MULT_TP = float(os.getenv("ATR_MULT_TP", "2.2"))

MAX_CANDLES = int(os.getenv("MAX_CANDLES", "200"))

COINBASE_BASE = os.getenv("COINBASE_BASE", "https://api.exchange.coinbase.com").rstrip("/")

DB_PATH = os.getenv("DB_PATH", "trades.db")

# =========================
# Helpers
# =========================
def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def fetch_candles(symbol: str, granularity: int, limit: int = 200) -> pd.DataFrame:
    """
    Coinbase Exchange candles endpoint returns [time, low, high, open, close, volume]
    time is unix seconds. We'll return ascending time DataFrame with columns:
    t, o, h, l, c, v
    """
    url = f"{COINBASE_BASE}/products/{symbol}/candles"
    # Coinbase returns newest-first; we reverse later
    params = {"granularity": granularity}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])

    # Each row: [time, low, high, open, close, volume]
    df = pd.DataFrame(data, columns=["t", "l", "h", "o", "c", "v"])
    df = df.sort_values("t").tail(limit).reset_index(drop=True)
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    # ensure numeric
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["o", "h", "l", "c"]).reset_index(drop=True)
    return df

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill").fillna(50)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["h"]
    low = df["l"]
    close = df["c"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    out = tr.ewm(alpha=1/period, adjust=False).mean()
    return out.fillna(method="bfill")

# =========================
# Trade DB (SQLite)
# =========================
def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def db_init():
    con = db_conn()
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS paper_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS paper_trades (
        id TEXT PRIMARY KEY,
        side TEXT NOT NULL,
        entry REAL NOT NULL,
        sl REAL NOT NULL,
        tp REAL NOT NULL,
        confidence REAL NOT NULL,
        opened_at TEXT NOT NULL,
        status TEXT NOT NULL,
        exit_price REAL,
        closed_at TEXT,
        pnl_usd REAL,
        pnl_r REAL,
        reason TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS real_trades (
        id TEXT PRIMARY KEY,
        side TEXT NOT NULL,
        entry REAL NOT NULL,
        exit_price REAL,
        opened_at TEXT NOT NULL,
        closed_at TEXT,
        pnl_usd REAL,
        note TEXT
    )
    """)
    con.commit()

    # initialize starting equity once
    cur.execute("SELECT value FROM paper_state WHERE key='equity_usd'")
    row = cur.fetchone()
    if row is None:
        cur.execute("INSERT INTO paper_state(key,value) VALUES(?,?)", ("equity_usd", json.dumps(PAPER_START_USD)))
        cur.execute("INSERT INTO paper_state(key,value) VALUES(?,?)", ("pos_btc", json.dumps(0.0)))
        con.commit()

    con.close()

def get_paper_equity():
    con = db_conn()
    cur = con.cursor()
    cur.execute("SELECT value FROM paper_state WHERE key='equity_usd'")
    row = cur.fetchone()
    con.close()
    return float(json.loads(row[0])) if row else PAPER_START_USD

def set_paper_equity(x: float):
    con = db_conn()
    cur = con.cursor()
    cur.execute("INSERT OR REPLACE INTO paper_state(key,value) VALUES(?,?)", ("equity_usd", json.dumps(float(x))))
    con.commit()
    con.close()

def get_open_paper_trade():
    con = db_conn()
    cur = con.cursor()
    cur.execute("SELECT id, side, entry, sl, tp, confidence, opened_at, status, reason FROM paper_trades WHERE status='OPEN' ORDER BY opened_at DESC LIMIT 1")
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return {
        "id": row[0],
        "side": row[1],
        "entry": row[2],
        "sl": row[3],
        "tp": row[4],
        "confidence": row[5],
        "opened_at": row[6],
        "status": row[7],
        "reason": row[8],
    }

def insert_open_trade(tr):
    con = db_conn()
    cur = con.cursor()
    cur.execute("""
    INSERT INTO paper_trades(id,side,entry,sl,tp,confidence,opened_at,status,reason)
    VALUES(?,?,?,?,?,?,?,?,?)
    """, (
        tr["id"], tr["side"], tr["entry"], tr["sl"], tr["tp"], tr["confidence"],
        tr["opened_at"], tr["status"], tr.get("reason","")
    ))
    con.commit()
    con.close()

def close_trade(trade_id: str, exit_price: float, pnl_usd: float, pnl_r: float):
    con = db_conn()
    cur = con.cursor()
    cur.execute("""
    UPDATE paper_trades
    SET status='CLOSED', exit_price=?, closed_at=?, pnl_usd=?, pnl_r=?
    WHERE id=?
    """, (float(exit_price), now_utc_iso(), float(pnl_usd), float(pnl_r), trade_id))
    con.commit()
    con.close()

def list_trades(table: str, limit: int = 200):
    con = db_conn()
    cur = con.cursor()
    if table == "paper_trades":
        cur.execute("""
        SELECT id, side, entry, sl, tp, confidence, opened_at, status, exit_price, closed_at, pnl_usd, pnl_r, reason
        FROM paper_trades
        ORDER BY opened_at DESC
        LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        con.close()
        out = []
        for r in rows:
            out.append({
                "id": r[0], "side": r[1], "entry": r[2], "sl": r[3], "tp": r[4],
                "confidence": r[5], "opened_at": r[6], "status": r[7],
                "exit_price": r[8], "closed_at": r[9], "pnl_usd": r[10], "pnl_r": r[11],
                "reason": r[12]
            })
        return out

    if table == "real_trades":
        cur.execute("""
        SELECT id, side, entry, exit_price, opened_at, closed_at, pnl_usd, note
        FROM real_trades
        ORDER BY opened_at DESC
        LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        con.close()
        out = []
        for r in rows:
            out.append({
                "id": r[0], "side": r[1], "entry": r[2], "exit_price": r[3],
                "opened_at": r[4], "closed_at": r[5], "pnl_usd": r[6], "note": r[7]
            })
        return out

    con.close()
    return []

# =========================
# Strategy & Paper execution
# =========================
def compute_signal(exec_df: pd.DataFrame, bias_df: pd.DataFrame):
    """
    Outputs:
      signal: "BUY" | "SELL" | "WAIT"
      confidence: 0..1
      trend_bias: "UP" | "DOWN" | "UNKNOWN"
      details dict for /explain
    """
    details = {}

    if len(exec_df) < 50 or len(bias_df) < 50:
        return "WAIT", 0.0, "UNKNOWN", {"reason": "Not enough candle data yet."}

    price = float(exec_df["c"].iloc[-1])
    details["price"] = price

    # indicators on exec timeframe
    rsi_exec = float(rsi(exec_df["c"], 14).iloc[-1])
    macd_line, macd_sig, macd_hist = macd(exec_df["c"])
    macd_hist_now = float(macd_hist.iloc[-1])
    atr_exec = float(atr(exec_df, 14).iloc[-1])

    details["rsi_exec"] = rsi_exec
    details["macd_hist_exec"] = macd_hist_now
    details["atr_exec"] = atr_exec

    # bias on 1h (or configured)
    # Simple bias: price vs EMA(50) and MACD histogram direction
    bias_ema = float(ema(bias_df["c"], 50).iloc[-1])
    bias_macd_hist = float(macd(bias_df["c"])[2].iloc[-1])

    trend_bias = "UNKNOWN"
    if bias_df["c"].iloc[-1] > bias_ema and bias_macd_hist >= 0:
        trend_bias = "UP"
    elif bias_df["c"].iloc[-1] < bias_ema and bias_macd_hist <= 0:
        trend_bias = "DOWN"

    details["bias_ema50"] = bias_ema
    details["bias_macd_hist"] = bias_macd_hist
    details["trend_bias"] = trend_bias

    # Scoring (heuristic, deterministic)
    # Momentum factors:
    bull = 0.0
    bear = 0.0

    # RSI: prefer buying when RSI recovering from low, selling when RSI falling from high
    if rsi_exec <= 30:
        bull += 0.25
    elif rsi_exec >= 70:
        bear += 0.25

    # MACD histogram sign
    if macd_hist_now > 0:
        bull += 0.25
    elif macd_hist_now < 0:
        bear += 0.25

    # Bias alignment boost
    if trend_bias == "UP":
        bull += 0.20
    elif trend_bias == "DOWN":
        bear += 0.20

    # Volatility sanity: very tiny ATR -> reduce confidence (chop); very huge -> also reduce a bit
    # normalize ATR by price
    atrp = atr_exec / price if price > 0 else 0
    details["atr_pct"] = atrp
    vol_penalty = 0.0
    if atrp < 0.0015:   # very low range
        vol_penalty = 0.10
    elif atrp > 0.02:   # extremely volatile
        vol_penalty = 0.07

    # Decide
    raw_conf = max(bull, bear) - vol_penalty
    raw_conf = clamp(raw_conf, 0.0, 0.95)

    if bull > bear and raw_conf >= 0.45:
        return "BUY", raw_conf, trend_bias, {**details, "bull": bull, "bear": bear, "vol_penalty": vol_penalty}
    if bear > bull and raw_conf >= 0.45:
        return "SELL", raw_conf, trend_bias, {**details, "bull": bull, "bear": bear, "vol_penalty": vol_penalty}

    return "WAIT", raw_conf, trend_bias, {**details, "bull": bull, "bear": bear, "vol_penalty": vol_penalty, "reason": "No high-probability setup."}

def maybe_open_paper_trade(signal: str, confidence: float, price: float, atr_value: float, reason: str):
    open_trade = get_open_paper_trade()
    if open_trade is not None:
        return None  # already in a trade

    if confidence < MIN_CONF_TRADE:
        return None

    side = None
    if signal == "BUY":
        side = "LONG"
    elif signal == "SELL":
        side = "SHORT"
    else:
        return None

    # ATR-based stops
    if atr_value is None or atr_value <= 0:
        return None

    if side == "LONG":
        sl = price - ATR_MULT_SL * atr_value
        tp = price + ATR_MULT_TP * atr_value
    else:
        sl = price + ATR_MULT_SL * atr_value
        tp = price - ATR_MULT_TP * atr_value

    tr = {
        "id": str(uuid.uuid4()),
        "side": side,
        "entry": float(price),
        "sl": float(sl),
        "tp": float(tp),
        "confidence": float(confidence),
        "opened_at": now_utc_iso(),
        "status": "OPEN",
        "reason": reason,
    }
    insert_open_trade(tr)
    return tr

def maybe_close_paper_trade(latest_price: float):
    tr = get_open_paper_trade()
    if tr is None:
        return None

    side = tr["side"]
    entry = tr["entry"]
    sl = tr["sl"]
    tp = tr["tp"]

    hit = None
    if side == "LONG":
        if latest_price <= sl:
            hit = "SL"
        elif latest_price >= tp:
            hit = "TP"
    else:  # SHORT
        if latest_price >= sl:
            hit = "SL"
        elif latest_price <= tp:
            hit = "TP"

    if not hit:
        return None

    # PnL model: use 1 unit notional sizing based on equity with fixed risk? (simple baseline)
    # We'll do a simple "1x" position size: invest 100% equity as notional on paper
    equity = get_paper_equity()
    notional = equity  # baseline; later we'll optimize sizing
    qty_btc = notional / entry if entry > 0 else 0

    if side == "LONG":
        pnl_usd = (latest_price - entry) * qty_btc
    else:
        pnl_usd = (entry - latest_price) * qty_btc

    # R-multiple: distance to SL in price terms
    risk_per_btc = abs(entry - sl)
    r = (abs(latest_price - entry) / risk_per_btc) if risk_per_btc > 0 else 0.0
    if hit == "SL":
        r = -abs(r)

    new_equity = max(0.0, equity + pnl_usd)
    set_paper_equity(new_equity)

    close_trade(tr["id"], latest_price, pnl_usd, r)
    return {"closed": tr["id"], "hit": hit, "exit": latest_price, "pnl_usd": pnl_usd, "equity": new_equity}

# =========================
# Engine state
# =========================
STATE_LOCK = threading.Lock()
ENGINE_STATE = {
    "ok": False,
    "time": "",
    "iso": "",
    "symbol": SYMBOL,
    "price": None,
    "signal": "WAIT",
    "confidence": 0.0,
    "trend_bias": "UNKNOWN",
    "rsi_5m": None,
    "rsi_exec": None,
    "atr_exec": None,
    "reason": "",
    "error": None,
    "candles_exec": [],
    "paper_equity": PAPER_START_USD,
    "paper_open_trade": None,
}

LAST_EXPLAIN = {}

def set_state(**kwargs):
    with STATE_LOCK:
        ENGINE_STATE.update(kwargs)

def get_state():
    with STATE_LOCK:
        return dict(ENGINE_STATE)

# =========================
# FastAPI app
# =========================
app = FastAPI(title="BTC Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    st = get_state()
    return {"ok": bool(st.get("ok")), "time": st.get("time"), "iso": st.get("iso"), "symbol": SYMBOL}

@app.get("/state")
def state():
    return get_state()

@app.get("/explain")
def explain():
    return {"ok": True, "explain": LAST_EXPLAIN, "min_conf_trade": MIN_CONF_TRADE}

@app.get("/trades")
def trades():
    paper = list_trades("paper_trades", limit=200)
    real = list_trades("real_trades", limit=200)
    return {
        "ok": True,
        "paper_equity": get_paper_equity(),
        "paper_trades": paper,
        "real_trades": real,
    }

# =========================
# Background engine loop
# =========================
def engine_loop():
    global LAST_EXPLAIN
    db_init()

    while True:
        t0 = time.time()
        try:
            # candles for exec & bias
            exec_df = fetch_candles(SYMBOL, EXEC_TF, limit=MAX_CANDLES)
            bias_df = fetch_candles(SYMBOL, BIAS_TF, limit=MAX_CANDLES)

            if len(exec_df) == 0 or len(bias_df) == 0:
                set_state(
                    ok=False,
                    time=datetime.now().strftime("%H:%M:%S"),
                    iso=now_utc_iso(),
                    error="No candle data yet.",
                    reason="Waiting for candles…"
                )
                time.sleep(REFRESH_SEC)
                continue

            # compute RSI(5m) separately (for display)
            df_5m = fetch_candles(SYMBOL, TF_5M, limit=MAX_CANDLES)
            rsi_5m_val = None
            if len(df_5m) >= 20:
                rsi_5m_val = float(rsi(df_5m["c"], 14).iloc[-1])

            # signal
            signal, conf, trend_bias, details = compute_signal(exec_df, bias_df)
            price = float(exec_df["c"].iloc[-1])
            atr_exec_val = float(atr(exec_df, 14).iloc[-1]) if len(exec_df) > 20 else None

            # paper exec
            closed = maybe_close_paper_trade(price)
            opened = None
            if signal in ("BUY", "SELL"):
                opened = maybe_open_paper_trade(signal, conf, price, atr_exec_val, reason=details.get("reason","signal"))

            open_trade = get_open_paper_trade()
            paper_equity = get_paper_equity()

            # package candles (UI-safe keys)
            candles_exec = []
            if len(exec_df) > 0:
                for _, row in exec_df.tail(120).iterrows():
                    candles_exec.append({
                        "t": row["t"].isoformat(),
                        "o": float(row["o"]),
                        "h": float(row["h"]),
                        "l": float(row["l"]),
                        "c": float(row["c"]),
                        "v": float(row["v"]) if not pd.isna(row["v"]) else 0.0,
                    })

            # update explain
            LAST_EXPLAIN = {
                "signal": signal,
                "confidence": conf,
                "trend_bias": trend_bias,
                "details": details,
                "paper_open_trade": open_trade,
                "paper_equity": paper_equity,
                "last_opened": opened,
                "last_closed": closed,
                "exec_tf": EXEC_TF,
                "bias_tf": BIAS_TF,
                "source": "Coinbase Exchange candles",
            }

            set_state(
                ok=True,
                time=datetime.now().strftime("%H:%M:%S"),
                iso=now_utc_iso(),
                symbol=SYMBOL,
                price=price,
                signal=("WAIT" if signal == "WAIT" else ("BUY" if signal == "BUY" else "SELL")),
                confidence=float(conf),
                trend_bias=trend_bias,
                rsi_5m=rsi_5m_val,
                rsi_exec=float(details.get("rsi_exec")) if details.get("rsi_exec") is not None else None,
                atr_exec=atr_exec_val,
                reason=details.get("reason", ""),
                error=None,
                candles_exec=candles_exec,
                paper_equity=paper_equity,
                paper_open_trade=open_trade,
            )

        except requests.HTTPError as e:
            # common: unsupported granularity, rate limit, etc
            msg = f"Coinbase API error: {str(e)}"
            set_state(ok=False, time=datetime.now().strftime("%H:%M:%S"), iso=now_utc_iso(), error=msg, reason="Engine not ready yet.")
        except Exception as e:
            set_state(ok=False, time=datetime.now().strftime("%H:%M:%S"), iso=now_utc_iso(), error=str(e), reason="Engine crashed in loop.")
        finally:
            dt = time.time() - t0
            sleep_for = max(0.5, REFRESH_SEC - dt)
            time.sleep(sleep_for)

def start_engine_thread():
    th = threading.Thread(target=engine_loop, daemon=True)
    th.start()

start_engine_thread()
