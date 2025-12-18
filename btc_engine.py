import os, time, uuid, sqlite3, threading, math
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ================= CONFIG =================
SYMBOL = os.getenv("SYMBOL", "BTC-USD")
COINBASE = "https://api.exchange.coinbase.com"

# Timeframes (Coinbase allowed granularities)
TF_5M  = 300
TF_15M = 900
TF_1H  = 3600

ENGINE_LOOP_SEC = int(os.getenv("ENGINE_LOOP_SEC", "10"))
DB_PATH = os.getenv("DB_PATH", "trades.db")

PAPER_START = float(os.getenv("PAPER_START", "250"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.10"))       # risk 10% of equity notionally (paper)
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.8"))
ATR_TP_MULT = float(os.getenv("ATR_TP_MULT", "2.2"))

# Auto-calibration bounds
MIN_CONF_DEFAULT = float(os.getenv("MIN_CONF", "0.60"))
MIN_CONF_FLOOR   = float(os.getenv("MIN_CONF_FLOOR", "0.52"))
MIN_CONF_CEIL    = float(os.getenv("MIN_CONF_CEIL", "0.78"))

# Behavior
COOLDOWN_BARS = int(os.getenv("COOLDOWN_BARS", "2"))  # wait N 15m candles after a close
MAX_BARS = int(os.getenv("MAX_BARS", "240"))          # candles stored/served

# ================= UTILS =================
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def fetch_candles(granularity, limit=200):
    # Coinbase returns: [ time, low, high, open, close, volume ]
    r = requests.get(
        f"{COINBASE}/products/{SYMBOL}/candles",
        params={"granularity": int(granularity)},
        timeout=10
    )
    data = r.json()
    if not isinstance(data, list) or (data and not isinstance(data[0], list)):
        raise RuntimeError(f"Coinbase API error: {data}")

    df = pd.DataFrame(data, columns=["t", "l", "h", "o", "c", "v"])
    df = df.sort_values("t").tail(limit)
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    for col in ["l","h","o","c","v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    return df

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(series, p=14):
    d = series.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    rs = up.ewm(alpha=1/p, adjust=False).mean() / down.ewm(alpha=1/p, adjust=False).mean()
    return 100 - (100 / (1 + rs))

def macd_hist(s):
    m = ema(s,12) - ema(s,26)
    sig = ema(m,9)
    return m - sig

def atr(df, p=14):
    prev = df["c"].shift()
    tr = pd.concat([
        df["h"] - df["l"],
        (df["h"] - prev).abs(),
        (df["l"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ================= DB =================
def db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    c = db()
    cur = c.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS state(
      k TEXT PRIMARY KEY,
      v REAL
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades(
      id TEXT PRIMARY KEY,
      side TEXT,
      entry REAL,
      exit REAL,
      sl REAL,
      tp REAL,
      conf REAL,
      reason TEXT,
      pnl REAL,
      r REAL,
      opened TEXT,
      closed TEXT,
      duration_sec INTEGER
    )""")

    # defaults
    if not cur.execute("SELECT 1 FROM state WHERE k='equity'").fetchone():
        cur.execute("INSERT INTO state VALUES('equity',?)", (PAPER_START,))
    if not cur.execute("SELECT 1 FROM state WHERE k='min_conf'").fetchone():
        cur.execute("INSERT INTO state VALUES('min_conf',?)", (MIN_CONF_DEFAULT,))
    if not cur.execute("SELECT 1 FROM state WHERE k='cooldown_until_t'").fetchone():
        cur.execute("INSERT INTO state VALUES('cooldown_until_t',?)", (0.0,))

    c.commit(); c.close()

def get_state_num(k, default=0.0):
    c = db()
    row = c.execute("SELECT v FROM state WHERE k=?", (k,)).fetchone()
    c.close()
    return float(row[0]) if row else float(default)

def set_state_num(k, v):
    c = db()
    c.execute("INSERT INTO state(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, float(v)))
    c.commit(); c.close()

def equity():
    return get_state_num("equity", PAPER_START)

def set_equity(x):
    set_state_num("equity", x)

def min_conf():
    return get_state_num("min_conf", MIN_CONF_DEFAULT)

def set_min_conf(x):
    set_state_num("min_conf", x)

def cooldown_until_t():
    return get_state_num("cooldown_until_t", 0.0)

def set_cooldown_until_t(x):
    set_state_num("cooldown_until_t", x)

# ================= METRICS / CALIBRATION =================
def load_trades_df():
    c = db()
    df = pd.read_sql("SELECT * FROM trades ORDER BY opened ASC", c)
    c.close()
    return df

def metrics():
    df = load_trades_df()
    if df.empty:
        return dict(ok=True, trades=0, win_rate=0.0, profit_factor=0.0, avg_r=0.0, max_dd=0.0, equity=equity())

    wins = df[df.pnl > 0]
    losses = df[df.pnl < 0]

    pf = float("inf") if losses.empty else (wins.pnl.sum() / abs(losses.pnl.sum()))
    eq_curve = PAPER_START + df.pnl.cumsum()
    dd = (eq_curve / eq_curve.cummax() - 1.0).min()

    return dict(
        ok=True,
        trades=int(len(df)),
        win_rate=float(len(wins)/len(df)),
        profit_factor=float(pf) if math.isfinite(pf) else 999.0,
        avg_r=float(df.r.mean()),
        max_dd=float(dd),
        equity=float(equity()),
    )

def calibrate_min_conf():
    """
    Simple, stable calibration:
    - bucket trades by conf ranges
    - pick the best bucket by average pnl (with a minimum sample)
    - set min_conf near that bucket's lower bound
    """
    df = load_trades_df()
    if len(df) < 12:
        return {"ok": True, "note": "Not enough trades yet to calibrate", "min_conf": min_conf()}

    # buckets: 0.50-0.55 ... 0.75-0.80
    bins = np.arange(0.50, 0.81, 0.05)
    df["bucket"] = pd.cut(df["conf"], bins=bins, include_lowest=True)

    grp = df.groupby("bucket").agg(
        n=("pnl","count"),
        avg_pnl=("pnl","mean"),
        win_rate=("pnl", lambda x: float((x>0).mean())),
    ).reset_index()

    grp = grp[grp["n"] >= 4]  # minimum samples per bucket
    if grp.empty:
        return {"ok": True, "note": "No bucket has enough samples yet", "min_conf": min_conf()}

    # choose best by avg_pnl, tie-break win_rate
    grp = grp.sort_values(["avg_pnl","win_rate"], ascending=False)
    best = grp.iloc[0]

    # parse bucket like "(0.6, 0.65]"
    b = str(best["bucket"])
    # extract lower number
    lo = float(b.split(",")[0].replace("(","").replace("[","").strip())

    new_min = clamp(lo, MIN_CONF_FLOOR, MIN_CONF_CEIL)
    set_min_conf(new_min)

    return {
        "ok": True,
        "min_conf": new_min,
        "picked_bucket": b,
        "bucket_stats": grp.to_dict("records")[:8]
    }

# ================= STRATEGY (SCORING) =================
def score_setup(df5, df15, df1h):
    """
    Returns: signal ("BUY"/"SELL"/"WAIT"), conf (0..1), reason (string), levels dict
    Uses:
      - Trend bias (1h EMA50)
      - Momentum (15m MACD hist)
      - Mean reversion trigger (5m RSI)
      - Volatility sanity (15m ATR)
    """
    price = float(df15["c"].iloc[-1])

    rsi5 = float(rsi(df5["c"]).iloc[-1])
    macd15 = float(macd_hist(df15["c"]).iloc[-1])
    ema1h50 = float(ema(df1h["c"], 50).iloc[-1])
    atr15 = float(atr(df15).iloc[-1])

    bias_up = price > ema1h50
    bias_dn = price < ema1h50

    # scoring
    score_buy = 0
    score_sell = 0
    notes = []

    # Bias
    if bias_up: score_buy += 2; notes.append("bias_up")
    if bias_dn: score_sell += 2; notes.append("bias_down")

    # RSI triggers
    if rsi5 < 28: score_buy += 3; notes.append("rsi5_oversold")
    elif rsi5 < 35: score_buy += 1; notes.append("rsi5_low")
    if rsi5 > 72: score_sell += 3; notes.append("rsi5_overbought")
    elif rsi5 > 65: score_sell += 1; notes.append("rsi5_high")

    # MACD momentum
    if macd15 > 0: score_buy += 2; notes.append("macd15_pos")
    if macd15 < 0: score_sell += 2; notes.append("macd15_neg")

    # volatility sanity: avoid insanely low atr (dead market) or extreme spikes
    atr_pct = atr15 / price if price else 0
    if atr_pct < 0.0008:
        score_buy -= 1; score_sell -= 1; notes.append("low_vol")
    if atr_pct > 0.01:
        score_buy -= 1; score_sell -= 1; notes.append("very_high_vol")

    # decide
    best = max(score_buy, score_sell)
    if best < 5:
        return "WAIT", 0.0, "no_high_prob_setup", {"price": price, "rsi5": rsi5, "macd15": macd15, "ema1h50": ema1h50, "atr15": atr15}

    if score_buy > score_sell:
        conf = clamp((score_buy / 10.0), 0.0, 1.0)
        return "BUY", conf, "|".join(notes), {"price": price, "rsi5": rsi5, "macd15": macd15, "ema1h50": ema1h50, "atr15": atr15}
    else:
        conf = clamp((score_sell / 10.0), 0.0, 1.0)
        return "SELL", conf, "|".join(notes), {"price": price, "rsi5": rsi5, "macd15": macd15, "ema1h50": ema1h50, "atr15": atr15}

def position_notional(eq):
    # simple: allocate a fraction of equity as notional for pnl scaling (paper)
    return max(25.0, eq * RISK_PCT)

# ================= FASTAPI =================
STATE = {}
CACHE = {"5m": None, "15m": None, "1h": None}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"ok": True, "time": now_iso(), "symbol": SYMBOL}

@app.get("/state")
def state():
    return STATE

@app.get("/metrics")
def metrics_route():
    return metrics()

@app.get("/calibrate")
def calibrate_route():
    return calibrate_min_conf()

@app.get("/trades")
def trades_route():
    df = load_trades_df()
    return {"ok": True, "trades": df.sort_values("opened", ascending=False).to_dict("records")}

@app.get("/candles")
def candles_route(tf: str = "15m"):
    # serves cached candles so UI is fast
    if tf not in CACHE:
        return {"ok": False, "error": "tf must be one of: 5m,15m,1h"}
    df = CACHE[tf]
    if df is None or df.empty:
        return {"ok": False, "error": "no candles yet"}
    out = df.tail(200).copy()
    out["t"] = out["t"].astype(str)
    return {"ok": True, "tf": tf, "candles": out.to_dict("records")}

# ================= ENGINE LOOP =================
def engine_loop():
    init_db()

    open_trade = None  # dict with keys: id, side, entry, sl, tp, opened_ts, conf, reason, notional
    last_closed_bar_t = None

    while True:
        try:
            df5  = fetch_candles(TF_5M,  limit=min(MAX_BARS, 400))
            df15 = fetch_candles(TF_15M, limit=min(MAX_BARS, 400))
            df1h = fetch_candles(TF_1H,  limit=min(MAX_BARS, 300))

            CACHE["5m"] = df5
            CACHE["15m"] = df15
            CACHE["1h"] = df1h

            price = float(df15["c"].iloc[-1])
            atr15 = float(atr(df15).iloc[-1]) if len(df15) > 20 else 0.0

            signal, conf, reason, levels = score_setup(df5, df15, df1h)
            mc = float(min_conf())

            # cooldown based on bar time
            current_bar_t = float(df15["t"].iloc[-1].timestamp())
            cd_until = float(cooldown_until_t())
            in_cooldown = current_bar_t < cd_until

            # manage open trade
            if open_trade is not None:
                side = open_trade["side"]
                entry = open_trade["entry"]
                sl = open_trade["sl"]
                tp = open_trade["tp"]
                notional = open_trade["notional"]

                hit = None
                if side == "LONG":
                    if price <= sl: hit = "SL"
                    elif price >= tp: hit = "TP"
                else:
                    if price >= sl: hit = "SL"
                    elif price <= tp: hit = "TP"

                if hit:
                    # pnl scaled by notional
                    direction = 1.0 if side == "LONG" else -1.0
                    pnl = (price - entry) * direction * (notional / entry)

                    # R multiple approx: pnl / (risk per unit)
                    risk_per_unit = abs(entry - sl)
                    r_mult = ( (price-entry)*direction ) / (risk_per_unit if risk_per_unit else 1e-9)

                    # persist
                    opened_ts = open_trade["opened_ts"]
                    closed_ts = datetime.now(timezone.utc).timestamp()
                    duration = int(max(0, closed_ts - opened_ts))

                    c = db()
                    c.execute(
                        "INSERT INTO trades VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            open_trade["id"],
                            side,
                            entry,
                            price,
                            sl,
                            tp,
                            float(open_trade["conf"]),
                            str(open_trade["reason"]),
                            float(pnl),
                            float(r_mult),
                            datetime.fromtimestamp(opened_ts, tz=timezone.utc).isoformat(),
                            now_iso(),
                            duration
                        )
                    )
                    c.commit(); c.close()

                    set_equity(equity() + pnl)

                    # cooldown: N bars
                    set_cooldown_until_t(current_bar_t + COOLDOWN_BARS * TF_15M)
                    open_trade = None

                    # recalibrate occasionally
                    if metrics().get("trades", 0) % 10 == 0:
                        calibrate_min_conf()

            # open new trade (only if none open)
            if open_trade is None and signal in ("BUY", "SELL") and not in_cooldown and atr15 > 0:
                if conf >= mc:
                    side = "LONG" if signal == "BUY" else "SHORT"
                    sl = price - ATR_SL_MULT * atr15 if side == "LONG" else price + ATR_SL_MULT * atr15
                    tp = price + ATR_TP_MULT * atr15 if side == "LONG" else price - ATR_TP_MULT * atr15
                    nt = position_notional(equity())

                    open_trade = {
                        "id": str(uuid.uuid4()),
                        "side": side,
                        "entry": price,
                        "sl": float(sl),
                        "tp": float(tp),
                        "opened_ts": datetime.now(timezone.utc).timestamp(),
                        "conf": float(conf),
                        "reason": reason,
                        "notional": float(nt),
                    }

            # update API state
            STATE.update({
                "ok": True,
                "time": now_iso(),
                "symbol": SYMBOL,
                "price": price,
                "signal": signal if not in_cooldown else "WAIT",
                "confidence": conf if not in_cooldown else 0.0,
                "min_conf": float(min_conf()),
                "reason": reason if not in_cooldown else "cooldown",
                "equity": float(equity()),
                "open_trade": open_trade,
                "levels": levels,
            })

        except Exception as e:
            STATE.update({"ok": False, "time": now_iso(), "error": str(e)})

        time.sleep(ENGINE_LOOP_SEC)

threading.Thread(target=engine_loop, daemon=True).start()
