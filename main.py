# -*- coding: utf-8 -*-
"""
RF Bot — Range Filter Strategy (TradingView 1:1)
- Signal-only: يقفل الصفقة فقط عند الإشارة العكسية
- Logs: مؤشرات ملوّنة + Compound PnL + Effective Equity
- Reads keys/config from ENV
"""

import os, time, math, threading, requests, shutil
import pandas as pd
import ccxt
from flask import Flask, jsonify
from datetime import datetime

# ===== ENV =====
SYMBOL        = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL      = os.getenv("INTERVAL", "15m")
LEVERAGE      = int(float(os.getenv("LEVERAGE", "10")))
RISK_ALLOC    = float(os.getenv("RISK_ALLOC", "0.60"))
PORT          = int(float(os.getenv("PORT", "5000")))
SELF_URL      = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
API_KEY       = os.getenv("BINGX_API_KEY", "")
API_SECRET    = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE     = bool(API_KEY and API_SECRET)
RF_SOURCE     = os.getenv("RF_SOURCE", "close").lower()
RF_PERIOD     = int(float(os.getenv("RF_PERIOD", "20")))
RF_MULT       = float(os.getenv("RF_MULT", "3.5"))
USE_TV_BAR    = os.getenv("USE_TV_BAR", "false").lower() == "true"
DECISION_EVERY_S = int(float(os.getenv("DECISION_EVERY_S", "60")))
MAX_SLIPPAGE_PCT = float(os.getenv("MAX_SLIPPAGE_PCT", "0.004"))

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

def line(char="─", color="cyan"):
    print(colored(char*80, color), flush=True)
def fmt(v, d=6):
    try: return f"{float(v):.{d}f}"
    except Exception: return "N/A"
def colorize_by_side(text):
    if state.get("open"):
        return colored(text, "green" if state["side"]=="long" else "red")
    return text

def safe_symbol(s: str):
    if s.endswith(":USDT") or s.endswith(":USDC"): return s
    if "/USDT" in s and not s.endswith(":USDT"): return s + ":USDT"
    return s

def make_exchange():
    return ccxt.bingx({
        "apiKey": API_KEY, "secret": API_SECRET,
        "enableRateLimit": True, "timeout": 20000,
        "options": {"defaultType": "swap", "defaultMarginMode": "isolated"}
    })
ex = make_exchange()
try: ex.load_markets()
except Exception: pass

def balance_usdt():
    if not MODE_LIVE: return None
    try: return ex.fetch_balance(params={"type":"swap"}).get("total",{}).get("USDT")
    except: return None
def price_now():
    try: t = ex.fetch_ticker(safe_symbol(SYMBOL)); return t.get("last") or t.get("close")
    except: return None
def fetch_ohlcv(limit=400):
    rows = ex.fetch_ohlcv(safe_symbol(SYMBOL), timeframe=INTERVAL, limit=limit, params={"type":"swap"})
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
def market_amount(amount):
    try:
        m = ex.market(safe_symbol(SYMBOL))
        prec = int(m.get("precision",{}).get("amount",3))
        min_amt = m.get("limits",{}).get("amount",{}).get("min",0.001)
        return max(float(f"{amount:.{prec}f}"), float(min_amt))
    except: return float(amount)
def compute_size(balance, price):
    if not price: return 0
    bal = (balance if (MODE_LIVE and balance is not None) else 100.0)
    raw = (bal * RISK_ALLOC * LEVERAGE) / price
    return market_amount(raw)

# ===== Range Filter =====
def rf_rng_size(x, qty, n):
    avrng = x.diff().abs().ewm(span=n, adjust=False).mean()
    return avrng.ewm(span=(n*2)-1, adjust=False).mean() * qty
def rf_filter(x, r):
    filt_vals=[x.iloc[0]]
    for i in range(1,len(x)):
        prev=filt_vals[-1]; xi=x.iloc[i]; ri=r.iloc[i]
        if xi-ri>prev: cur=xi-ri
        elif xi+ri<prev: cur=xi+ri
        else: cur=prev
        filt_vals.append(cur)
    filt=pd.Series(filt_vals,index=x.index)
    return filt+r, filt-r, filt
def compute_rf(df):
    src=df[RF_SOURCE].astype(float); r=rf_rng_size(src,RF_MULT,RF_PERIOD)
    hi,lo,filt=rf_filter(src,r)
    dfilt=filt-filt.shift(1)
    fdir=pd.Series(0,index=df.index).mask(dfilt>0,1).mask(dfilt<0,-1).ffill().fillna(0)
    upward=(fdir==1).astype(int); downward=(fdir==-1).astype(int)
    longCond=((src>filt)&(src>src.shift(1))&(upward>0))|((src>filt)&(src<src.shift(1))&(upward>0))
    shortCond=((src<filt)&(src<src.shift(1))&(downward>0))|((src<filt)&(src>src.shift(1))&(downward>0))
    CondIni=pd.Series(0,index=df.index)
    for i in range(1,len(df)):
        if longCond.iloc[i]: CondIni.iloc[i]=1
        elif shortCond.iloc[i]: CondIni.iloc[i]=-1
        else: CondIni.iloc[i]=CondIni.iloc[i-1]
    longSig=longCond&(CondIni.shift(1)==-1)
    shortSig=shortCond&(CondIni.shift(1)==1)
    i=len(df)-1 if USE_TV_BAR else len(df)-2
    last=lambda s: None if pd.isna(s.iloc[i]) else float(s.iloc[i])
    return {"price":last(df["close"]), "filt":last(filt),"hi":last(hi),"lo":last(lo),
            "rsize":last(r),"fdir":1 if filt.iloc[i]-filt.iloc[i-1]>0 else (-1 if filt.iloc[i]-filt.iloc[i-1]<0 else 0),
            "long":bool(longSig.iloc[i]),"short":bool(shortSig.iloc[i])}

# ===== State =====
state={"open":False,"side":None,"entry":None,"qty":None,"pnl":0.0}
compound_pnl=0.0

def snapshot(balance, rf, position, total_pnl):
    line("═","cyan")
    now=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    mode="LIVE" if MODE_LIVE else "PAPER"
    print(colored(f"📊 RF BOT • {safe_symbol(SYMBOL)} • {INTERVAL} • {mode} • {now}","cyan"))
    line("─","cyan")
    print(colorize_by_side("📈 INDICATORS"))
    print(f"   💲 Price     : {fmt(rf.get('price'))}")
    print(f"   📏 Filter    : {fmt(rf.get('filt'))}")
    print(f"   🔼 Band Hi   : {fmt(rf.get('hi'))}")
    print(f"   🔽 Band Lo   : {fmt(rf.get('lo'))}")
    print(f"   📐 RangeSize : {fmt(rf.get('rsize'))}")
    dir_icon="🟢 ↑ Up" if rf.get("fdir")==1 else ("🔴 ↓ Down" if rf.get("fdir")==-1 else "⚪ Flat")
    print(f"   🧭 Direction : {dir_icon}")
    print(f"   🟩 LongSig   : {rf.get('long')}")
    print(f"   🟥 ShortSig  : {rf.get('short')}")
    print()
    print(colorize_by_side("🧭 POSITION"))
    bal_txt="N/A (paper)" if not MODE_LIVE else fmt(balance,2)
    print(f"   💰 Balance   : {bal_txt} USDT")
    if position["open"]:
        side_icon="🟩 LONG" if position['side']=="long" else "🟥 SHORT"
        print(f"   📌 Status    : {side_icon}")
        print(f"   🎯 Entry     : {fmt(position['entry'])}")
        print(f"   📦 Qty       : {fmt(position['qty'],4)}")
        print(f"   📊 PnL curr. : {fmt(position['pnl'])}")
    else:
        print("   📌 Status    : ⚪ FLAT")
    print()
    effective_equity=(balance or 0)+total_pnl if balance else total_pnl
    print("📦 RESULTS")
    print(f"   🧮 Compound PnL   : {fmt(total_pnl)}")
    print(f"   🚀 Effective Equity: {fmt(effective_equity)} USDT")
    line("─","cyan")

def open_market(side, qty, ref_price):
    global state
    if not MODE_LIVE:
        state.update({"open":True,"side":"long" if side=="buy" else "short","entry":ref_price,"qty":qty,"pnl":0.0})
        print(colorize_by_side(f"✅ OPEN {side.upper()} PAPER qty={fmt(qty,4)} @ {fmt(ref_price)}"))
        return
    try: ex.set_leverage(LEVERAGE,safe_symbol(SYMBOL),params={"side":"BOTH"})
    except: pass
    try:
        ex.create_order(safe_symbol(SYMBOL),"market",side,qty,params={"reduceOnly":False})
        state.update({"open":True,"side":"long" if side=="buy" else "short","entry":ref_price,"qty":qty,"pnl":0.0})
        print(colorize_by_side(f"✅ OPEN {side.upper()} CONFIRMED qty={fmt(qty,4)} @ {fmt(ref_price)}"))
    except Exception as e: print(colored(f"❌ open error: {e}","red"))

def close_market(reason):
    global state,compound_pnl
    if not state["open"]: return
    px=price_now() or state["entry"]; qty=state["qty"]
    side="sell" if state["side"]=="long" else "buy"
    if not MODE_LIVE: pnl=(px-state["entry"])*qty*(1 if state["side"]=="long" else -1)
    else:
        try: ex.create_order(safe_symbol(SYMBOL),"market",side,qty,params={"reduceOnly":True})
        except: pass
        pnl=(px-state["entry"])*qty*(1 if state["side"]=="long" else -1)
    compound_pnl+=pnl
    print(colored(f"🔚 CLOSE {state['side']} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
    state={"open":False,"side":None,"entry":None,"qty":None,"pnl":0.0}

def trade_loop():
    global state,compound_pnl
    while True:
        try:
            bal=balance_usdt(); px=price_now(); df=fetch_ohlcv()
            if df is None or len(df)<50: time.sleep(DECISION_EVERY_S); continue
            rf=compute_rf(df)
            if state["open"] and px:
                state["pnl"]=(px-state["entry"])*state["qty"] if state["side"]=="long" else (state["entry"]-px)*state["qty"]
            snapshot(bal,rf,state.copy(),compound_pnl)
            side="buy" if rf.get("long") else ("sell" if rf.get("short") else None)
            if not side: time.sleep(DECISION_EVERY_S); continue
            desired="long" if side=="buy" else "short"; ref=rf.get("price") or px
            if not ref or abs(px-ref)/ref>MAX_SLIPPAGE_PCT: time.sleep(DECISION_EVERY_S); continue
            qty=compute_size(bal,px)
            if state["open"]:
                if state["side"]!=desired: close_market("opposite_signal"); open_market(side,qty,px)
            else: open_market(side,qty,px)
        except Exception as e: print(colored(f"❌ loop error: {e}","red"))
        time.sleep(DECISION_EVERY_S)

def keepalive_loop():
    if not SELF_URL: return
    url=SELF_URL.rstrip("/")
    while True:
        try: requests.get(url,timeout=8)
        except: pass
        time.sleep(50)

app=Flask(__name__)
@app.route("/") 
def home(): return f"✅ RF Bot Running — {safe_symbol(SYMBOL)} {INTERVAL}"
@app.route("/metrics")
def metrics():
    return jsonify({"symbol":safe_symbol(SYMBOL),"interval":INTERVAL,"mode":"live" if MODE_LIVE else "paper",
                    "leverage":LEVERAGE,"risk_alloc":RISK_ALLOC,"balance":balance_usdt(),
                    "price":price_now(),"position":state,"compound_pnl":compound_pnl,
                    "time":datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")})

threading.Thread(target=trade_loop,daemon=True).start()
threading.Thread(target=keepalive_loop,daemon=True).start()
if __name__=="__main__": app.run(host="0.0.0.0",port=PORT)
