import os, time, uuid, sqlite3, threading, math, json
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
RISK_PCT = float(os.getenv("RISK_PCT", "0.10"))       # 10% of equity notionally (paper)
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.8"))
ATR_TP_MULT = float(os.getenv("ATR_TP_MULT", "2.2"))

# Auto-calibration bounds
MIN_CONF_DEFAULT = float(os.getenv("MIN_CONF", "0.60"))
MIN_CONF_FLOOR   = float(os.getenv("MIN_CONF_FLOOR", "0.45"))
MIN_CONF_CEIL    = float(os.getenv("MIN_CONF_CEIL", "0.80"))

# Behavior
COOLDOWN_BARS = int(os.getenv("COOLDOWN_BARS", "2"))
MAX_BARS = int(os.getenv("MAX_BARS", "240"))

# Update #3 knobs (adaptive)
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD", "4"))   # was effectively 5; lower -> more trades
MODEL_MIN_TRADES = int(os.getenv("MODEL_MIN_TRADES", "25"))# start learning after N closed trades
LEARN_LR = float(os.getenv("LEARN_LR", "0.15"))
LEARN_L2 = float(os.getenv("LEARN_L2", "1e-4"))

# ================= UTILS =================
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def sigmoid(x):
    x = float(x)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

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

# ================= DB =================
def db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    c = db()
    cur = c.cursor()

    # numeric state
    cur.execute("""
    CREATE TABLE IF NOT EXISTS state(
      k TEXT PRIMARY KEY,
      v REAL
    )""")

    # text/json state
    cur.execute("""
    CREATE TABLE IF NOT EXISTS state_text(
      k TEXT PRIMARY KEY,
      v TEXT
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

    # store entry features separately (for learning without schema migrations)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trade_features(
      trade_id TEXT PRIMARY KEY,
      f_json TEXT
    )""")

    # defaults
    if not cur.execute("SELECT 1 FROM state WHERE k='equity'").fetchone():
        cur.execute("INSERT INTO state VALUES('equity',?)", (PAPER_START,))
    if not cur.execute("SELECT 1 FROM state WHERE k='min_conf'").fetchone():
        cur.execute("INSERT INTO state VALUES('min_conf',?)", (MIN_CONF_DEFAULT,))
    if not cur.execute("SELECT 1 FROM state WHERE k='cooldown_until_t'").fetchone():
        cur.execute("INSERT INTO state VALUES('cooldown_until_t',?)", (0.0,))

    # model weights (Update #3)
    if not cur.execute("SELECT 1 FROM state_text WHERE k='model_w'").fetchone():
        # weights: [bias, rsi5_z, macd_z, ema_dist_z, atrpct_z, side_is_long]
        cur.execute("INSERT INTO state_text VALUES('model_w',?)", (json.dumps([0,0,0,0,0,0]),))
    if not cur.execute("SELECT 1 FROM state_text WHERE k='model_stats'").fetchone():
        # running mean/std for z-normalization
        cur.execute("INSERT INTO state_text VALUES('model_stats',?)", (json.dumps({
            "n": 0,
            "mean": [0,0,0,0,0,0],
            "m2":   [1e-6,1e-6,1e-6,1e-6,1e-6,1e-6],
        }),))

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

def get_state_text(k, default=""):
    c = db()
    row = c.execute("SELECT v FROM state_text WHERE k=?", (k,)).fetchone()
    c.close()
    return str(row[0]) if row else str(default)

def set_state_text(k, v):
    c = db()
    c.execute("INSERT INTO state_text(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, str(v)))
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
    Stable calibration:
    - bucket trades by conf
    - pick best bucket by avg pnl (min samples)
    """
    df = load_trades_df()
    if len(df) < 12:
        return {"ok": True, "note": "Not enough trades yet to calibrate", "min_conf": min_conf()}

    bins = np.arange(0.40, 0.91, 0.05)
    df = df.copy()
    df["bucket"] = pd.cut(df["conf"], bins=bins, include_lowest=True)

    grp = df.groupby("bucket").agg(
        n=("pnl","count"),
        avg_pnl=("pnl","mean"),
        win_rate=("pnl", lambda x: float((x>0).mean())),
    ).reset_index()

    grp = grp[grp["n"] >= 4]
    if grp.empty:
        return {"ok": True, "note": "No bucket has enough samples yet", "min_conf": min_conf()}

    grp = grp.sort_values(["avg_pnl","win_rate"], ascending=False)
    best = grp.iloc[0]
    b = str(best["bucket"])
    lo = float(b.split(",")[0].replace("(","").replace("[","").strip())

    new_min = clamp(lo, MIN_CONF_FLOOR, MIN_CONF_CEIL)
    set_min_conf(new_min)

    return {
        "ok": True,
        "min_conf": new_min,
        "picked_bucket": b,
        "bucket_stats": grp.to_dict("records")[:8]
    }

# ================= Update #3: ADAPTIVE MODEL =================
def get_model_w():
    return np.array(json.loads(get_state_text("model_w", "[0,0,0,0,0,0]")), dtype=float)

def set_model_w(w):
    set_state_text("model_w", json.dumps([float(x) for x in w]))

def get_model_stats():
    return json.loads(get_state_text("model_stats"))

def set_model_stats(stats):
    set_state_text("model_stats", json.dumps(stats))

def _update_running_stats(stats, x):
    # Welford update for each dimension
    n = int(stats["n"])
    mean = np.array(stats["mean"], dtype=float)
    m2 = np.array(stats["m2"], dtype=float)

    n2 = n + 1
    delta = x - mean
    mean2 = mean + delta / n2
    delta2 = x - mean2
    m2_2 = m2 + delta * delta2

    stats["n"] = n2
    stats["mean"] = mean2.tolist()
    stats["m2"] = m2_2.tolist()
    return stats

def _z_norm(stats, x):
    n = max(1, int(stats["n"]))
    mean = np.array(stats["mean"], dtype=float)
    m2 = np.array(stats["m2"], dtype=float)
    var = m2 / max(1, (n - 1))
    std = np.sqrt(np.maximum(var, 1e-6))
    return (x - mean) / std

def features_from_levels(levels, side):
    """
    Build a compact feature vector from your indicators.
    Order must match model weights:
      [bias, rsi5_z, macd_z, ema_dist_z, atrpct_z, side_is_long]
    """
    price = float(levels.get("price", 0.0))
    rsi5v = float(levels.get("rsi5", 50.0))
    macdv = float(levels.get("macd15", 0.0))
    ema50 = float(levels.get("ema1h50", price))
    atr15v = float(levels.get("atr15", 0.0))

    # raw features (pre z)
    f = np.array([
        1.0,                       # bias term
        rsi5v,
        macdv,
        (price - ema50),
        (atr15v / price) if price else 0.0,
        1.0 if side == "LONG" else 0.0
    ], dtype=float)

    return f

def model_predict_prob(levels, side):
    stats = get_model_stats()
    w = get_model_w()

    f = features_from_levels(levels, side)

    # normalize all except bias
    f2 = f.copy()
    f2[1:] = _z_norm(stats, f[1:])

    logit = float(np.dot(w, f2))
    return sigmoid(logit)

def model_train_on_trade(trade_id, won):
    """
    Online logistic regression update using entry features saved at open.
    """
    c = db()
    row = c.execute("SELECT f_json FROM trade_features WHERE trade_id=?", (trade_id,)).fetchone()
    c.close()
    if not row:
        return {"ok": False, "note": "no features for trade"}

    f = np.array(json.loads(row[0]), dtype=float)
    y = 1.0 if won else 0.0

    stats = get_model_stats()
    w = get_model_w()

    # update running stats (for normalization) on f[1:]
    stats = _update_running_stats(stats, f[1:])
    set_model_stats(stats)

    # normalize f for training
    f2 = f.copy()
    f2[1:] = _z_norm(stats, f[1:])

    p = sigmoid(float(np.dot(w, f2)))
    grad = (p - y) * f2 + LEARN_L2 * w
    w2 = w - LEARN_LR * grad

    set_model_w(w2)

    return {"ok": True, "p_before": p, "y": y, "w": w2.tolist(), "n": int(stats["n"])}

# ================= STRATEGY (BASE RULES + CONF) =================
def score_setup(df5, df15, df1h):
    """
    Returns: signal ("BUY"/"SELL"/"WAIT"), base_conf (0..1), reason, levels dict
    """
    price = float(df15["c"].iloc[-1])

    rsi5v = float(rsi(df5["c"]).iloc[-1])
    macd15v = float(macd_hist(df15["c"]).iloc[-1])
    ema1h50 = float(ema(df1h["c"], 50).iloc[-1])
    atr15v = float(atr(df15).iloc[-1])

    bias_up = price > ema1h50
    bias_dn = price < ema1h50

    score_buy = 0
    score_sell = 0
    notes = []

    # Bias
    if bias_up:
        score_buy += 2; notes.append("bias_up")
    if bias_dn:
        score_sell += 2; notes.append("bias_down")

    # RSI triggers
    if rsi5v < 28:
        score_buy += 3; notes.append("rsi5_oversold")
    elif rsi5v < 35:
        score_buy += 1; notes.append("rsi5_low")

    if rsi5v > 72:
        score_sell += 3; notes.append("rsi5_overbought")
    elif rsi5v > 65:
        score_sell += 1; notes.append("rsi5_high")

    # MACD momentum
    if macd15v > 0:
        score_buy += 2; notes.append("macd15_pos")
    if macd15v < 0:
        score_sell += 2; notes.append("macd15_neg")

    # volatility sanity
    atr_pct = (atr15v / price) if price else 0.0
    if atr_pct < 0.0008:
        score_buy -= 1; score_sell -= 1; notes.append("low_vol")
    if atr_pct > 0.01:
        score_buy -= 1; score_sell -= 1; notes.append("very_high_vol")

    levels = {"price": price, "rsi5": rsi5v, "macd15": macd15v, "ema1h50": ema1h50, "atr15": atr15v}

    best = max(score_buy, score_sell)
    if best < SCORE_THRESHOLD:
        return "WAIT", 0.0, "no_high_prob_setup", levels

    if score_buy > score_sell:
        base_conf = clamp(score_buy / 10.0, 0.0, 1.0)
        return "BUY", base_conf, "|".join(notes), levels
    else:
        base_conf = clamp(score_sell / 10.0, 0.0, 1.0)
        return "SELL", base_conf, "|".join(notes), levels

def position_notional(eq):
    return max(25.0, eq * RISK_PCT)

def blended_confidence(base_conf, levels, side):
    """
    Update #3:
    - confidence is NOT only rules
    - if enough trade history exists, blend rule confidence with learned win-prob
    """
    n_trades = metrics().get("trades", 0)
    if n_trades < MODEL_MIN_TRADES:
        return base_conf, {"mode": "rules_only", "n_trades": n_trades}

    p_win = model_predict_prob(levels, side)
    # blend: 40% rules + 60% learned (tunable)
    conf = clamp(0.4 * base_conf + 0.6 * p_win, 0.0, 1.0)
    return conf, {"mode": "blended", "n_trades": n_trades, "p_win": p_win}

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

@app.get("/model")
def model_route():
    return {
        "ok": True,
        "w": json.loads(get_state_text("model_w")),
        "stats": get_model_stats(),
        "min_trades": MODEL_MIN_TRADES
    }

@app.get("/trades")
def trades_route():
    df = load_trades_df()
    return {"ok": True, "trades": df.sort_values("opened", ascending=False).to_dict("records")}

@app.get("/candles")
def candles_route(tf: str = "15m"):
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

    open_trade = None  # dict with keys: id, side, entry, sl, tp, opened_ts, conf, reason, notional, levels, model_meta
    while True:
        try:
            df5  = fetch_candles(TF_5M,  limit=min(MAX_BARS, 400))
            df15 = fetch_candles(TF_15M, limit=min(MAX_BARS, 400))
            df1h = fetch_candles(TF_1H,  limit=min(MAX_BARS, 300))

            CACHE["5m"] = df5
            CACHE["15m"] = df15
            CACHE["1h"] = df1h

            price = float(df15["c"].iloc[-1])
            atr15v = float(atr(df15).iloc[-1]) if len(df15) > 20 else 0.0

            signal, base_conf, reason, levels = score_setup(df5, df15, df1h)
            mc = float(min_conf())

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
                    direction = 1.0 if side == "LONG" else -1.0
                    pnl = (price - entry) * direction * (notional / entry)

                    risk_per_unit = abs(entry - sl)
                    r_mult = (((price - entry) * direction) / (risk_per_unit if risk_per_unit else 1e-9))

                    opened_ts = open_trade["opened_ts"]
                    closed_ts = datetime.now(timezone.utc).timestamp()
                    duration = int(max(0, closed_ts - opened_ts))

                    trade_id = open_trade["id"]

                    c = db()
                    c.execute(
                        "INSERT INTO trades VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            trade_id,
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

                    # train model (Update #3) on the result
                    won = pnl > 0
                    model_train_on_trade(trade_id, won)

                    # recalibrate occasionally
                    if metrics().get("trades", 0) % 10 == 0:
                        calibrate_min_conf()

                    open_trade = None

            # open new trade
            if open_trade is None and signal in ("BUY", "SELL") and not in_cooldown and atr15v > 0:
                side = "LONG" if signal == "BUY" else "SHORT"

                # Update #3: blended confidence (rules + learned probability)
                conf, model_meta = blended_confidence(base_conf, levels, side)

                if conf >= mc:
                    sl = price - ATR_SL_MULT * atr15v if side == "LONG" else price + ATR_SL_MULT * atr15v
                    tp = price + ATR_TP_MULT * atr15v if side == "LONG" else price - ATR_TP_MULT * atr15v
                    nt = position_notional(equity())

                    trade_id = str(uuid.uuid4())

                    open_trade = {
                        "id": trade_id,
                        "side": side,
                        "entry": price,
                        "sl": float(sl),
                        "tp": float(tp),
                        "opened_ts": datetime.now(timezone.utc).timestamp(),
                        "conf": float(conf),
                        "reason": reason,
                        "notional": float(nt),
                        "levels": levels,
                        "model_meta": model_meta,
                    }

                    # save entry features for learning later
                    f = features_from_levels(levels, side).tolist()
                    c = db()
                    c.execute(
                        "INSERT INTO trade_features(trade_id,f_json) VALUES(?,?) ON CONFLICT(trade_id) DO UPDATE SET f_json=excluded.f_json",
                        (trade_id, json.dumps([float(x) for x in f]))
                    )
                    c.commit(); c.close()

            # update API state
            out_signal = signal if not in_cooldown else "WAIT"
            out_conf = 0.0 if not in_cooldown else 0.0

            if not in_cooldown and signal in ("BUY", "SELL"):
                # show blended confidence for UI
                side_preview = "LONG" if signal == "BUY" else "SHORT"
                out_conf, meta = blended_confidence(base_conf, levels, side_preview)
            else:
                meta = {"mode": "cooldown_or_wait"}

            STATE.update({
                "ok": True,
                "time": now_iso(),
                "symbol": SYMBOL,
                "price": float(price),
                "signal": out_signal,
                "confidence": float(out_conf) if out_signal in ("BUY","SELL") else float(out_conf),
                "base_conf": float(base_conf),
                "min_conf": float(min_conf()),
                "reason": reason if not in_cooldown else "cooldown",
                "equity": float(equity()),
                "open_trade": open_trade,
                "levels": levels,
                "model_meta": meta,
                "score_threshold": SCORE_THRESHOLD,
            })

        except Exception as e:
            STATE.update({"ok": False, "time": now_iso(), "error": str(e)})

        time.sleep(ENGINE_LOOP_SEC)

threading.Thread(target=engine_loop, daemon=True).start()

# Allow "python btc_engine.py" locally
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
