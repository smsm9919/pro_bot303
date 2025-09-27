# -*- coding: utf-8 -*-
"""
RF Bot — Exact TradingView Range Filter (Buy & Sell) — FULL READY
- إشارات مطابقة 1:1 لمؤشر TradingView Range Filter B&S
- لا SL ولا TP ولا أي حماية إضافية
- المتغيرات جاهزة جوه الكود
"""

import os, time, threading, requests
import pandas as pd
import ccxt
from flask import Flask
from datetime import datetime

# ======================= FIXED VARS =======================
API_KEY    = os.getenv("BINGX_API_KEY", "your_api_key")
API_SECRET = os.getenv("BINGX_API_SECRET", "your_api_secret")

SYMBOL     = "DOGE/USDT:USDT"
INTERVAL   = "15m"
LEVERAGE   = 10
RISK_ALLOC = 0.6
PORT       = 5000

RF_SOURCE  = "close"   # مصدر الحساب
RF_PERIOD  = 20        # الفتره
RF_MULT    = 3.5       # معامل الفلتر
USE_TV_BAR = True      # نفس البار زي تريدنج فيو
LOOP_SLEEP = 10        # تكرار كل 10 ثواني

MODE_LIVE  = bool(API_KEY and API_SECRET)

# ======================= UI =======================
def fmt(v, d=6, na="N/A"):
    try: return f"{float(v):.{d}f}"
    except: return na

def sep(): print("─"*70, flush=True)

def safe_symbol(s):
    if s.endswith(":USDT"): return s
    if "/USDT" in s: return s + ":USDT"
    return s

# ======================= EXCHANGE =======================
def make_exchange():
    return ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap", "defaultMarginMode": "isolated"}
    })

ex = make_exchange()
try: ex.load_markets()
except: pass

def balance_usdt():
    if not MODE_LIVE: return None
    try:
        b = ex.fetch_balance(params={"type":"swap"})
        return b.get("total",{}).get("USDT")
    except: return None

def price_now():
    try:
        t = ex.fetch_ticker(safe_symbol(SYMBOL))
        return t.get("last")
    except: return None

def fetch_ohlcv(limit=500):
    rows = ex.fetch_ohlcv(safe_symbol(SYMBOL), timeframe=INTERVAL, limit=limit, params={"type":"swap"})
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def compute_size(balance, price):
    bal = (balance if MODE_LIVE and balance is not None else 100.0)
    return (bal * RISK_ALLOC * LEVERAGE) / max(price, 1e-9)

# =================== Range Filter ===================
def _ema(s: pd.Series, n: int): return s.ewm(span=n, adjust=False).mean()
def _rng_size(src: pd.Series, qty: float, n: int):
    avrng = _ema((src - src.shift(1)).abs(), n)
    return _ema(avrng, (2*n)-1) * qty

def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf = [float(src.iloc[0])]
    for i in range(1, len(src)):
        prev = rf[-1]; x = float(src.iloc[i]); r = float(rsize.iloc[i]); cur = prev
        if x - r > prev: cur = x - r
        if x + r < prev: cur = x + r
        rf.append(cur)
    filt = pd.Series(rf, index=src.index)
    hi = filt + rsize; lo = filt - rsize
    return hi, lo, filt

def compute_tv_signals(df: pd.DataFrame):
    if df is None or len(df) < RF_PERIOD + 5: return None
    s = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(s, _rng_size(s, RF_MULT, RF_PERIOD))
    dfilt = filt - filt.shift(1)
    fdir = pd.Series(0.0, index=filt.index).mask(dfilt>0,1).mask(dfilt<0,-1).ffill().fillna(0.0)

    upward   = (fdir==1).astype(int)
    downward = (fdir==-1).astype(int)

    src_gt_f, src_lt_f = (s > filt), (s < filt)
    src_gt_p, src_lt_p = (s > s.shift(1)), (s < s.shift(1))

    longCond  = (src_gt_f & src_gt_p & (upward>0)) | (src_gt_f & src_lt_p & (upward>0))
    shortCond = (src_lt_f & src_lt_p & (downward>0)) | (src_lt_f & src_gt_p & (downward>0))

    CondIni = pd.Series(0, index=s.index)
    for i in range(1, len(s)):
        if longCond.iloc[i]: CondIni.iloc[i] = 1
        elif shortCond.iloc[i]: CondIni.iloc[i] = -1
        else: CondIni.iloc[i] = CondIni.iloc[i-1]

    longSig  = longCond & (CondIni.shift(1) == -1)
    shortSig = shortCond & (CondIni.shift(1) ==  1)

    i = len(df)-1 if USE_TV_BAR else len(df)-2
    return {
        "price": float(s.iloc[i]),
        "long": bool(longSig.iloc[i]),
        "short": bool(shortSig.iloc[i])
    }

# =================== STATE ===================
state = {"open": False, "side": None, "entry": None, "qty": None}
compound_pnl = 0.0

def snapshot(info):
    sep()
    print(f"📊 RF BOT {SYMBOL} {INTERVAL} {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Price: {fmt(info.get('price'))} | Long: {info.get('long')} | Short: {info.get('short')}")
    print(f"Status: {state['side'] if state['open'] else 'FLAT'} @ {fmt(state['entry'])}")
    sep()

# =================== Trading ===================
def open_market(side, qty, px):
    state.update({"open": True, "side": side, "entry": px, "qty": qty})
    print(f"✅ OPEN {side.upper()} {qty}@{px}")

def close_market(reason, px):
    global compound_pnl
    if not state["open"]: return
    pnl = (px - state["entry"]) * state["qty"] * (1 if state["side"]=="long" else -1)
    compound_pnl += pnl
    print(f"🔚 CLOSE {state['side']} pnl={fmt(pnl)} total={fmt(compound_pnl)} reason={reason}")
    state.update({"open": False, "side": None, "entry": None, "qty": None})

def trade_loop():
    while True:
        try:
            bal = balance_usdt()
            df  = fetch_ohlcv()
            info = compute_tv_signals(df)
            snapshot(info or {})

            if not info: 
                time.sleep(LOOP_SLEEP); continue

            side = "long" if info["long"] else ("short" if info["short"] else None)
            px = info["price"]
            qty = compute_size(bal, px)

            if side:
                if state["open"]:
                    if state["side"] != side:
                        close_market("opposite", px)
                        open_market(side, qty, px)
                else:
                    open_market(side, qty, px)
        except Exception as e:
            print(f"❌ loop error: {e}")
        time.sleep(LOOP_SLEEP)

# =================== KEEPALIVE + API ===================
def keepalive_loop():
    while True:
        try: requests.get("https://render.com", timeout=5)
        except: pass
        time.sleep(50)

app = Flask(__name__)
@app.route("/")
def home(): return "✅ RF Bot Running"

threading.Thread(target=trade_loop, daemon=True).start()
threading.Thread(target=keepalive_loop, daemon=True).start()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
