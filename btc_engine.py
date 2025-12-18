import os, time, uuid, json, sqlite3, threading
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ================= CONFIG =================
SYMBOL = "BTC-USD"
EXEC_TF = 900        # 15m
BIAS_TF = 3600       # 1h
ENGINE_LOOP_SEC = 10
MIN_CONF = 0.60
PAPER_START = 250.0
DB_PATH = "trades.db"
COINBASE = "https://api.exchange.coinbase.com"

# ================= UTILS =================
def now(): return datetime.now(timezone.utc).isoformat()

def fetch_candles(tf, limit=200):
    r = requests.get(
        f"{COINBASE}/products/{SYMBOL}/candles",
        params={"granularity": tf},
        timeout=10
    )
    data = r.json()
    df = pd.DataFrame(data, columns=["t","l","h","o","c","v"])
    df = df.sort_values("t").tail(limit)
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    return df

def rsi(series, p=14):
    d = series.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    rs = up.ewm(alpha=1/p).mean() / down.ewm(alpha=1/p).mean()
    return 100 - (100 / (1 + rs))

def ema(s, n): return s.ewm(span=n).mean()

def macd(s):
    m = ema(s,12)-ema(s,26)
    sig = ema(m,9)
    return m-sig

def atr(df, p=14):
    tr = pd.concat([
        df["h"]-df["l"],
        (df["h"]-df["c"].shift()).abs(),
        (df["l"]-df["c"].shift()).abs()
    ],axis=1).max(axis=1)
    return tr.ewm(alpha=1/p).mean()

# ================= DB =================
def db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    c = db()
    cur = c.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades(
      id TEXT PRIMARY KEY,
      side TEXT,
      entry REAL,
      exit REAL,
      r REAL,
      pnl REAL,
      opened TEXT,
      closed TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS state(
      k TEXT PRIMARY KEY,
      v REAL
    )""")
    if not cur.execute("SELECT 1 FROM state WHERE k='equity'").fetchone():
        cur.execute("INSERT INTO state VALUES('equity',?)",(PAPER_START,))
    c.commit(); c.close()

def equity():
    c=db();v=c.execute("SELECT v FROM state WHERE k='equity'").fetchone()[0];c.close();return v

def set_equity(x):
    c=db();c.execute("UPDATE state SET v=? WHERE k='equity'",(x,));c.commit();c.close()

# ================= METRICS =================
def metrics():
    c=db()
    df=pd.read_sql("SELECT * FROM trades",c)
    c.close()
    if df.empty:
        return dict(ok=True,trades=0,win_rate=0,profit_factor=0,avg_r=0,max_dd=0,equity=equity())
    wins=df[df.pnl>0]
    losses=df[df.pnl<0]
    pf = wins.pnl.sum()/abs(losses.pnl.sum()) if not losses.empty else float("inf")
    eq_curve = PAPER_START + df.pnl.cumsum()
    dd = (eq_curve/eq_curve.cummax()-1).min()
    return dict(
        ok=True,
        trades=len(df),
        win_rate=len(wins)/len(df),
        profit_factor=pf,
        avg_r=df.r.mean(),
        max_dd=dd,
        equity=equity()
    )

# ================= ENGINE =================
STATE={}
app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

@app.get("/health")
def h(): return {"ok":True}

@app.get("/state")
def s(): return STATE

@app.get("/metrics")
def m(): return metrics()

@app.get("/trades")
def t():
    c=db()
    rows=pd.read_sql("SELECT * FROM trades ORDER BY opened DESC",c).to_dict("records")
    c.close()
    return {"ok":True,"trades":rows}

def loop():
    init_db()
    open_trade=None
    while True:
        try:
            df=fetch_candles(EXEC_TF)
            bias=fetch_candles(BIAS_TF)
            price=df.c.iloc[-1]
            r=rsi(df.c).iloc[-1]
            m=macd(df.c).iloc[-1]
            a=atr(df).iloc[-1]
            bias_ok=bias.c.iloc[-1]>ema(bias.c,50).iloc[-1]
            signal="WAIT"
            conf=0.0
            if r<30 and m>0 and bias_ok:
                signal="BUY"; conf=0.7
            elif r>70 and m<0 and not bias_ok:
                signal="SELL"; conf=0.7

            if open_trade:
                side,entry,sl,tp,id=open_trade
                hit=None
                if side=="LONG":
                    if price<=sl: hit="SL"
                    if price>=tp: hit="TP"
                else:
                    if price>=sl: hit="SL"
                    if price<=tp: hit="TP"
                if hit:
                    pnl=(price-entry)*(equity()/entry)*(1 if side=="LONG" else -1)
                    set_equity(equity()+pnl)
                    c=db()
                    c.execute("INSERT INTO trades VALUES(?,?,?,?,?,?,?,?)",
                              (id,side,entry,price,pnl/(abs(entry-sl)),pnl,now(),now()))
                    c.commit();c.close()
                    open_trade=None

            if not open_trade and conf>=MIN_CONF:
                sl=price-1.8*a if signal=="BUY" else price+1.8*a
                tp=price+2.2*a if signal=="BUY" else price-2.2*a
                open_trade=("LONG" if signal=="BUY" else "SHORT",price,sl,tp,str(uuid.uuid4()))

            STATE.update({
                "ok":True,"price":price,"signal":signal,"confidence":conf,
                "equity":equity(),"time":now()
            })
        except Exception as e:
            STATE.update({"ok":False,"error":str(e)})
        time.sleep(ENGINE_LOOP_SEC)

threading.Thread(target=loop,daemon=True).start()
