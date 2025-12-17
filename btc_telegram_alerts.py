import os, json, time, math, threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------- config ----------
PRODUCT = os.getenv("PRODUCT", "BTC-USD")
CB_BASE = os.getenv("COINBASE_BASE", "https://api.exchange.coinbase.com").rstrip("/")
POLL_SEC = float(os.getenv("POLL_SEC", "10"))

# Coinbase-supported granularities (sec)
SUPPORTED = (60, 300, 900, 3600, 21600, 86400)
GRAN_5M  = int(os.getenv("GRAN_5M",  "300"))
GRAN_15M = int(os.getenv("GRAN_15M", "900"))
GRAN_1H  = int(os.getenv("GRAN_1H",  "3600"))

CONF_BUY   = float(os.getenv("CONF_BUY",   "72"))  # 0..100
CONF_SELL  = float(os.getenv("CONF_SELL",  "72"))
CONF_WATCH = float(os.getenv("CONF_WATCH", "55"))

COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "900"))

TG_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

DATA_DIR = os.getenv("DATA_DIR", ".")
SIGNAL_LOG_PATH = os.getenv("SIGNAL_LOG_PATH", os.path.join(DATA_DIR, "signals.jsonl"))
PAPER_LOG_PATH  = os.getenv("PAPER_LOG_PATH",  os.path.join(DATA_DIR, "paper_trades.jsonl"))
REAL_LOG_PATH   = os.getenv("REAL_LOG_PATH",   os.path.join(DATA_DIR, "real_trades.jsonl"))
PAPER_START_USD = float(os.getenv("PAPER_START_USD", "250"))

# ---------- utils ----------
def utcnow() -> datetime: return datetime.now(timezone.utc)
def iso(dt: datetime) -> str: return dt.astimezone(timezone.utc).isoformat()
def clamp(x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))
def fnum(x: Any, d: float = float("nan")) -> float:
    try: return float(x)
    except Exception: return d

def snap_gran(g: int) -> int:
    return g if g in SUPPORTED else min(SUPPORTED, key=lambda s: abs(s-g))

def jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f: f.write(json.dumps(obj) + "\n")
    except Exception:
        pass  # never crash engine due to logging

def jsonl_tail(path: str, n: int = 500) -> List[Dict[str, Any]]:
    if not os.path.exists(path): return []
    try:
        with open(path, "r", encoding="utf-8") as f: lines = f.readlines()[-n:]
        out=[]
        for ln in lines:
            ln=ln.strip()
            if not ln: continue
            try: out.append(json.loads(ln))
            except Exception: continue
        return out
    except Exception:
        return []

def tg_send(text: str) -> None:
    if not (TG_BOT_TOKEN and TG_CHAT_ID): return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": text},
            timeout=8,
        )
    except Exception:
        pass

# ---------- indicators ----------
def ema(s: pd.Series, span: int) -> pd.Series: return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    au = up.ewm(alpha=1/period, adjust=False).mean()
    ad = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([(df["high"]-df["low"]), (df["high"]-pc).abs(), (df["low"]-pc).abs()], axis=1).max(axis=1)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1/period, adjust=False).mean()

def macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, sig)
    return m, s, (m - s)

def bollinger(close: pd.Series, period: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return (mid - k*sd), mid, (mid + k*sd)

# ---------- coinbase ----------
def cb_get(path: str, params: Optional[Dict[str, Any]]=None, timeout: int=12) -> Any:
    r = requests.get(f"{CB_BASE}{path}", params=params, timeout=timeout, headers={"User-Agent":"btc-engine/2.0"})
    if r.status_code != 200: raise RuntimeError(f"Coinbase HTTP {r.status_code}: {r.text[:200]}")
    return r.json()

def fetch_ticker() -> float:
    return fnum(cb_get(f"/products/{PRODUCT}/ticker", timeout=8).get("price"))

def fetch_candles(gran: int, limit: int = 240) -> pd.DataFrame:
    gran = snap_gran(int(gran))
    end = utcnow()
    start = end - timedelta(seconds=gran*limit)
    raw = cb_get(
        f"/products/{PRODUCT}/candles",
        params={"granularity":gran, "start":iso(start), "end":iso(end)},
        timeout=14,
    )
    rows=[]
    for it in (raw if isinstance(raw, list) else []):
        if isinstance(it, (list, tuple)) and len(it) >= 6:
            t, lo, hi, op, cl, vol = it[:6]
            rows.append({
                "time": datetime.fromtimestamp(int(t), tz=timezone.utc),
                "low":float(lo),"high":float(hi),"open":float(op),"close":float(cl),"volume":float(vol)
            })
    if not rows: raise RuntimeError("No candles returned")
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    d=df.set_index("time")
    out=pd.concat([
        d["open"].resample(rule).first(),
        d["high"].resample(rule).max(),
        d["low"].resample(rule).min(),
        d["close"].resample(rule).last(),
        d["volume"].resample(rule).sum(),
    ], axis=1)
    out.columns=["open","high","low","close","volume"]
    return out.dropna().reset_index()

def df_to_candles(df: pd.DataFrame, limit: int = 180) -> List[Dict[str, Any]]:
    if df is None or df.empty: return []
    d=df.tail(limit)
    vals = d["volume"] if "volume" in d.columns else [0.0]*len(d)
    return [{
        "time": iso(t.to_pydatetime() if hasattr(t,"to_pydatetime") else t),
        "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": float(v),
    } for t,o,h,l,c,v in zip(d["time"], d["open"], d["high"], d["low"], d["close"], vals)]

# ---------- paper account ----------
def paper_account() -> Dict[str, float]:
    usd, btc = PAPER_START_USD, 0.0
    for t in jsonl_tail(PAPER_LOG_PATH, n=10000):
        side=str(t.get("side","")).upper()
        qty=fnum(t.get("qty"),0.0); px=fnum(t.get("price"),0.0)
        if qty<=0 or px<=0: continue
        if side=="BUY":
            cost=qty*px
            if cost<=usd+1e-9: usd-=cost; btc+=qty
        elif side=="SELL":
            if qty<=btc+1e-12: usd+=qty*px; btc-=qty
    return {"usd":usd,"btc":btc}

def paper_equity(price: float) -> Dict[str, float]:
    a=paper_account()
    eq=a["usd"]+a["btc"]*price
    return {"usd":a["usd"],"btc":a["btc"],"equity":eq,"pl":eq-PAPER_START_USD}

# ---------- signal engine ----------
def trend_bias(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> str:
    def one(df: pd.DataFrame) -> str:
        if df is None or df.empty or len(df) < 220: return "NEUTRAL"
        c=df["close"]
        e50=float(ema(c,50).iloc[-1]); e200=float(ema(c,200).iloc[-1])
        if e50>e200: return "UP"
        if e50<e200: return "DOWN"
        return "NEUTRAL"
    b1, b4 = one(df_1h), one(df_4h)
    if b1==b4 and b1!="NEUTRAL": return b1
    if b1!="NEUTRAL": return b1
    if b4!="NEUTRAL": return b4
    return "NEUTRAL"

def score(df5: pd.DataFrame, df15: pd.DataFrame, df1h: pd.DataFrame) -> Dict[str, Any]:
    if df5 is None or df5.empty or len(df5) < 60:
        return {"signal":"WAIT","confidence":0.0,"reason":"Not enough 5m candles"}
    if df15 is None or df15.empty or len(df15) < 60:
        return {"signal":"WAIT","confidence":0.0,"reason":"Not enough 15m candles"}

    df4h = resample_ohlcv(df1h, "4h") if df1h is not None and not df1h.empty else pd.DataFrame()
    bias = trend_bias(df1h, df4h)

    price=float(df5["close"].iloc[-1])
    r=float(rsi(df5["close"],14).iloc[-1])

    m_line, s_line, _ = macd(df5["close"])
    macd_up = (m_line.iloc[-2] < s_line.iloc[-2]) and (m_line.iloc[-1] > s_line.iloc[-1])
    macd_dn = (m_line.iloc[-2] > s_line.iloc[-2]) and (m_line.iloc[-1] < s_line.iloc[-1])

    bb_lo, _, bb_hi = bollinger(df5["close"],20,2.0)
    bb_lo=float(bb_lo.iloc[-1]); bb_hi=float(bb_hi.iloc[-1])

    roc=float(df5["close"].pct_change(3).iloc[-1]) if len(df5)>=4 else 0.0

    atr15=float(atr(df15,14).iloc[-1])
    sl = price - 1.3*atr15 if atr15>0 else None
    tp = price + 2.0*atr15 if atr15>0 else None

    # 180m window (36x 5m)
    recent=df5.tail(36)
    local_low=float(recent["low"].min()); local_high=float(recent["high"].max())
    peak_watch = (price/local_low - 1.0) >= 0.006 if local_low>0 else False
    dip_watch  = (local_high/price - 1.0) >= 0.006 if price>0 else False

    score=0.0; why=[]
    if bias=="UP": score+=18; why.append("bias UP (1h+4h)")
    elif bias=="DOWN": score+=18; why.append("bias DOWN (1h+4h)")
    else: score+=6; why.append("bias NEUTRAL")

    if r<=30: score+=25; why.append(f"RSI oversold {r:.1f}")
    elif r<=40: score+=12; why.append(f"RSI low {r:.1f}")
    elif r>=70: score+=25; why.append(f"RSI overbought {r:.1f}")
    elif r>=60: score+=12; why.append(f"RSI high {r:.1f}")

    if macd_up: score+=14; why.append("MACD cross up")
    if macd_dn: score+=14; why.append("MACD cross down")

    if price<=bb_lo: score+=12; why.append("at/under lower BB")
    if price>=bb_hi: score+=12; why.append("at/over upper BB")

    if abs(roc) >= 0.0025: score+=10; why.append(f"momentum {'up' if roc>0 else 'down'} {roc*100:.2f}%")

    score=clamp(score,0,100)

    buy_ok  = (bias in ("UP","NEUTRAL")   and (r<=35 or macd_up) and roc>=-0.002) or (bias=="UP" and price<=bb_lo and r<=45)
    sell_ok = (bias in ("DOWN","NEUTRAL") and (r>=65 or macd_dn) and roc<= 0.002) or (bias=="DOWN" and price>=bb_hi and r>=55)

    sig="WAIT"
    if buy_ok and score>=CONF_BUY: sig="BUY"
    elif sell_ok and score>=CONF_SELL: sig="SELL"

    return {
        "signal": sig,
        "confidence": float(round(score,2)),
        "trend_bias": bias,
        "rsi_5m": float(round(r,2)),
        "momentum": float(round(roc,6)),
        "peak_watch": bool(peak_watch),
        "dip_watch": bool(dip_watch),
        "atr_sl": None if sl is None else float(round(sl,2)),
        "atr_tp": None if tp is None else float(round(tp,2)),
        "reason": " • ".join(why[:6]) if why else "—",
    }

def calibrate(n: int = 3000) -> Dict[str, Any]:
    sigs = jsonl_tail(SIGNAL_LOG_PATH, n=n)
    if len(sigs) < 200: return {"ok": False, "error": "Not enough signal history yet"}
    sigs = sorted(sigs, key=lambda x: x.get("ts",""))
    def sim(th: float) -> float:
        cash=PAPER_START_USD; btc=0.0
        for s in sigs:
            px=fnum(s.get("price"),0.0); conf=fnum(s.get("confidence"),0.0); sg=str(s.get("signal","WAIT")).upper()
            if px<=0: continue
            if sg=="BUY"  and conf>=th and cash>1: btc=cash/px; cash=0.0
            if sg=="SELL" and conf>=th and btc>0:  cash=btc*px; btc=0.0
        last=fnum(sigs[-1].get("price"),0.0)
        return cash + btc*(last if last>0 else 0.0)
    best_th=None; best_eq=-1e18
    for th in range(55,86):
        eq=sim(float(th))
        if eq>best_eq: best_eq=eq; best_th=float(th)
    return {"ok": True, "recommended_threshold": best_th, "sim_equity": best_eq}

# ---------- api ----------
app = FastAPI(title="BTC AI Engine", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

LOCK = threading.Lock()
STATE: Dict[str, Any] = {
    "ok": False,
    "ts": iso(utcnow()),
    "src": "Coinbase",
    "product": PRODUCT,
    "price": None,
    "signal": "WAIT",
    "confidence": 0.0,
    "trend_bias": "NEUTRAL",
    "rsi_5m": None,
    "momentum": 0.0,
    "peak_watch": False,
    "dip_watch": False,
    "reason": "Starting…",
    "atr_sl": None,
    "atr_tp": None,
    "candles_15m": [],
    "candles_5m": [],
    "paper": {"usd": PAPER_START_USD, "btc": 0.0, "equity": PAPER_START_USD, "pl": 0.0},
    "calibration": {"ok": False},
}

_last_alert_at = 0.0
_last_alert_key = ""
_cache = {"ticker_ts":0.0,"price":None,"c5_ts":0.0,"c15_ts":0.0,"c1h_ts":0.0,"df5":None,"df15":None,"df1h":None}

def loop() -> None:
    global _last_alert_at, _last_alert_key
    g5, g15, g1h = snap_gran(GRAN_5M), snap_gran(GRAN_15M), snap_gran(GRAN_1H)
    if (g5,g15,g1h)!=(GRAN_5M,GRAN_15M,GRAN_1H):
        tg_send(f"⚠️ Granularity adjusted: 5m {GRAN_5M}->{g5}, 15m {GRAN_15M}->{g15}, 1h {GRAN_1H}->{g1h}")

    while True:
        try:
            now=time.time()
            if now-_cache["ticker_ts"]>=5:
                px=fetch_ticker()
                if px and px>0: _cache["price"]=float(px); _cache["ticker_ts"]=now

            if now-_cache["c5_ts"]>=30:
                _cache["df5"]=fetch_candles(g5, 240); _cache["c5_ts"]=now
            if now-_cache["c15_ts"]>=45:
                _cache["df15"]=fetch_candles(g15, 240); _cache["c15_ts"]=now
            if now-_cache["c1h_ts"]>=120:
                _cache["df1h"]=fetch_candles(g1h, 400); _cache["c1h_ts"]=now

            df5, df15, df1h = _cache["df5"], _cache["df15"], _cache["df1h"]
            s = score(df5, df15, df1h)

            price = _cache["price"]
            if (price is None or not (price>0)) and df5 is not None and not df5.empty:
                price=float(df5["close"].iloc[-1])

            paper = paper_equity(float(price) if price else 0.0)

            new = {
                "ok": True,
                "ts": iso(utcnow()),
                "src": "Coinbase",
                "product": PRODUCT,
                "price": float(price) if price else None,
                **s,
                "candles_15m": df_to_candles(df15, 180),
                "candles_5m": df_to_candles(df5, 180),
                "paper": {"usd": round(paper["usd"],2), "btc": round(paper["btc"],8), "equity": round(paper["equity"],2), "pl": round(paper["pl"],2)},
            }

            jsonl_append(SIGNAL_LOG_PATH, {
                "ts":new["ts"],"price":new["price"],"signal":new["signal"],"confidence":new["confidence"],
                "trend_bias":new["trend_bias"],"rsi_5m":new["rsi_5m"],"momentum":new["momentum"]
            })

            key=f"{new['signal']}|{int(new['confidence'])}|{new.get('peak_watch')}|{new.get('dip_watch')}|{new.get('trend_bias')}"
            if key!=_last_alert_key and (now-_last_alert_at)>=COOLDOWN_SEC:
                if new["signal"] in ("BUY","SELL") or ((new.get("peak_watch") or new.get("dip_watch")) and new["confidence"]>=CONF_WATCH):
                    _last_alert_key=key; _last_alert_at=now
                    px=new["price"]
                    if px:
                        tg_send(
                            f"🧠 BTC Bot\nPrice: ${px:,.2f}\nSignal: {new['signal']} (conf {new['confidence']:.0f}%)\n"
                            f"Bias: {new['trend_bias']} • RSI(5m): {new['rsi_5m']}\n{new['reason']}"
                        )

            with LOCK: STATE.update(new)

        except Exception as e:
            with LOCK:
                STATE["ok"]=False
                STATE["ts"]=iso(utcnow())
                STATE["reason"]=f"Engine error: {e}"

        time.sleep(POLL_SEC)

# start loop (can disable for local tests)
if os.getenv("DISABLE_LOOP","0") != "1":
    threading.Thread(target=loop, daemon=True).start()

@app.get("/health")
def health() -> Dict[str, Any]:
    with LOCK: st=dict(STATE)
    return {"ok": True, "engine_ok": bool(st.get("ok")), "ts": st.get("ts"), "product": st.get("product")}

@app.get("/state")
def state() -> Dict[str, Any]:
    with LOCK: return dict(STATE)

@app.get("/explain")
def explain() -> Dict[str, Any]:
    with LOCK: st=dict(STATE)
    keys=["ok","signal","confidence","trend_bias","rsi_5m","momentum","peak_watch","dip_watch","atr_sl","atr_tp","reason"]
    return {k: st.get(k) for k in keys}

@app.post("/paper/log")
def paper_log(payload: Dict[str, Any]) -> Dict[str, Any]:
    side=str(payload.get("side","")).upper()
    qty=fnum(payload.get("qty"),0.0)
    price=fnum(payload.get("price"),0.0)
    note=str(payload.get("note","")).strip()
    with LOCK: px=fnum(STATE.get("price"),0.0)
    if price<=0 and px>0: price=px
    if side not in ("BUY","SELL") or qty<=0 or price<=0:
        return {"ok": False, "error": "Need side BUY/SELL, qty>0, price>0"}
    obj={"ts":iso(utcnow()),"side":side,"qty":qty,"price":price,"note":note}
    jsonl_append(PAPER_LOG_PATH,obj)
    return {"ok": True, "logged": obj, "paper": paper_equity(price)}

@app.post("/real/log")
def real_log(payload: Dict[str, Any]) -> Dict[str, Any]:
    side=str(payload.get("side","")).upper()
    qty=fnum(payload.get("qty"),0.0)
    price=fnum(payload.get("price"),0.0)
    note=str(payload.get("note","")).strip()
    with LOCK: px=fnum(STATE.get("price"),0.0)
    if price<=0 and px>0: price=px
    if side not in ("BUY","SELL") or qty<=0 or price<=0:
        return {"ok": False, "error": "Need side BUY/SELL, qty>0, price>0"}
    obj={"ts":iso(utcnow()),"side":side,"qty":qty,"price":price,"note":note}
    jsonl_append(REAL_LOG_PATH,obj)
    return {"ok": True, "logged": obj}

@app.get("/trades")
def trades() -> Dict[str, Any]:
    with LOCK: px=fnum(STATE.get("price"),0.0)
    return {
        "ok": True,
        "paper": paper_equity(px if px>0 else 0.0),
        "paper_trades": jsonl_tail(PAPER_LOG_PATH, 200),
        "real_trades": jsonl_tail(REAL_LOG_PATH, 200),
    }

@app.post("/calibrate")
def calibrate_endpoint(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    n = int(payload.get("n", 3000)) if isinstance(payload, dict) else 3000
    rec = calibrate(n=n)
    with LOCK: STATE["calibration"]=rec
    return rec

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","8080")))
