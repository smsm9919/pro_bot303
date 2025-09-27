# -*- coding: utf-8 -*-
"""
RF Bot — EXACT TradingView Range Filter (LIVE VERSION)
- 1:1 مع Pine Script اللي انت بعتلي
- يفتح/يقفل الصفقات فقط مع الإشارة العكسية
- لا ستوب لوس / لا تيك بروفيت / لا فلاتر إضافية
- يقرأ كل حاجة من ENV: BINGX_API_KEY, BINGX_API_SECRET, SYMBOL, INTERVAL, LEVERAGE, RISK_ALLOC, SELF_URL
"""

import os, time, threading, requests
import pandas as pd
import ccxt
from flask import Flask
from datetime import datetime

# ======================= ENV =======================
SYMBOL     = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(float(os.getenv("LEVERAGE", "10")))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", "0.6"))
SELF_URL   = os.getenv("SELF_URL", "")
PORT       = int(float(os.getenv("PORT", "5000")))

API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET)

# Indicator settings (TradingView Pine defaults)
RF_PERIOD  = int(float(os.getenv("RF_PERIOD", "20")))
RF_MULT    = float(os.getenv("RF_MULT", "3.5"))
RF_SOURCE  = os.getenv("RF_SOURCE", "close").lower()

# ======================= Exchange =======================
def make_exchange():
    return ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "defaultMarginMode": "isolated"}
    })
ex = make_exchange()
try: ex.load_markets()
except: pass

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b = ex.fetch_balance(params={"type": "swap"})
        return b.get("total", {}).get("USDT", 0)
    except: return 0

def price_now():
    try:
        t = ex.fetch_ticker(SYMBOL)
        return t.get("last") or t.get("close")
    except: return None

def fetch_ohlcv(limit=500):
    rows = ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"})
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def market_amount(amount):
    try:
        m = ex.market(SYMBOL)
        prec = int(m.get("precision",{}).get("amount",3))
        min_amt = m.get("limits",{}).get("amount",{}).get("min",0.001)
        amt = float(f"{amount:.{prec}f}")
        return max(amt, float(min_amt))
    except: return float(amount)

def compute_size(balance, price):
    raw = (balance * RISK_ALLOC * LEVERAGE) / max(price, 1e-9)
    return market_amount(raw)

# =================== Range Filter Logic ===================
def _ema(s: pd.Series, n: int): return s.ewm(span=n, adjust=False).mean()

def _rng_size(src: pd.Series, qty: float, n: int):
    avrng = _ema((src - src.shift(1)).abs(), n)
    return _ema(avrng, (n*2)-1) * qty

def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf = [float(src.iloc[0])]
    for i in range(1, len(src)):
        prev = rf[-1]; x = float(src.iloc[i]); r = float(rsize.iloc[i]); cur = prev
        if x - r > prev: cur = x - r
        if x + r < prev: cur = x + r
        rf.append(cur)
    filt = pd.Series(rf, index=src.index)
    return filt + rsize, filt - rsize, filt

def compute_tv_signals(df: pd.DataFrame):
    s = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(s, _rng_size(s, RF_MULT, RF_PERIOD))
    dfilt = filt - filt.shift(1)
    fdir = pd.Series(0.0, index=filt.index).mask(dfilt>0,1).mask(dfilt<0,-1).ffill().fillna(0.0)

    longCond  = (s>filt) & (((s>s.shift(1)) & (fdir==1)) | ((s<s.shift(1)) & (fdir==1)))
    shortCond = (s<filt) & (((s<s.shift(1)) & (fdir==-1))| ((s>s.shift(1)) & (fdir==-1)))

    CondIni = pd.Series(0, index=s.index)
    for i in range(1, len(s)):
        if longCond.iloc[i]: CondIni.iloc[i] = 1
        elif shortCond.iloc[i]: CondIni.iloc[i] = -1
        else: CondIni.iloc[i] = CondIni.iloc[i-1]

    longSignal  = longCond & (CondIni.shift(1) == -1)
    shortSignal = shortCond & (CondIni.shift(1) ==  1)

    i = len(df)-1
    return {
        "price": float(s.iloc[i]),
        "filter": float(filt.iloc[i]),
        "hi": float(hi.iloc[i]),
        "lo": float(lo.iloc[i]),
        "fdir": int(fdir.iloc[i]),
        "long": bool(longSignal.iloc[i]),
        "short": bool(shortSignal.iloc[i])
    }

# =================== State & Logging ===================
state = {"open": False, "side": None, "entry": None, "qty": None}
compound_pnl = 0.0

def snapshot(bal, info):
    print("─"*70)
    print(f"📊 RF BOT {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow()} UTC")
    print("─"*70)
    print(f"💲 Price     : {info.get('price')}")
    print(f"📏 Filter    : {info.get('filter')}")
    print(f"🔼 Band Hi   : {info.get('hi')}")
    print(f"🔽 Band Lo   : {info.get('lo')}")
    print(f"🧭 Direction : {info.get('fdir')}")
    print(f"🟩 LongSig   : {info.get('long')}")
    print(f"🟥 ShortSig  : {info.get('short')}")
    print()
    print(f"💰 Balance   : {bal} USDT")
    print(f"📌 Status    : {'LONG' if state['side']=='long' else ('SHORT' if state['side']=='short' else 'FLAT')}")
    print(f"🧮 CompoundPnL : {compound_pnl}")
    print("─"*70, flush=True)

# =================== Trading Exec ===================
def open_market(side, qty, price):
    global state
    if not MODE_LIVE:
        state.update({"open":True,"side":"long" if side=="buy" else "short","entry":price,"qty":qty})
        print(f"✅ OPEN {side.upper()} (PAPER) @ {price}")
        return
    ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
    ex.create_order(SYMBOL, "market", side, qty, params={"reduceOnly": False})
    state.update({"open":True,"side":"long" if side=="buy" else "short","entry":price,"qty":qty})
    print(f"✅ OPEN {side.upper()} CONFIRMED @ {price}")

def close_market(reason, price):
    global state, compound_pnl
    if not state["open"]: return
    side = "sell" if state["side"]=="long" else "buy"
    pnl = (price - state["entry"]) * state["qty"] if state["side"]=="long" else (state["entry"] - price) * state["qty"]
    if MODE_LIVE:
        ex.create_order(SYMBOL, "market", side, state["qty"], params={"reduceOnly": True})
    compound_pnl += pnl
    print(f"🔚 CLOSE {state['side']} reason={reason} pnl={pnl} total={compound_pnl}")
    state.update({"open":False,"side":None,"entry":None,"qty":None})

# =================== Main Loop ===================
def trade_loop():
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()
            info = compute_tv_signals(df)
            snapshot(bal, info)

            if info["long"]:
                if state["open"] and state["side"]=="short":
                    close_market("opposite_signal", px)
                    open_market("buy", compute_size(bal, px), px)
                elif not state["open"]:
                    open_market("buy", compute_size(bal, px), px)

            elif info["short"]:
                if state["open"] and state["side"]=="long":
                    close_market("opposite_signal", px)
                    open_market("sell", compute_size(bal, px), px)
                elif not state["open"]:
                    open_market("sell", compute_size(bal, px), px)

        except Exception as e:
            print(f"❌ loop error: {e}")
        time.sleep(30)

# =================== Keepalive + API ===================
def keepalive_loop():
    if not SELF_URL: return
    while True:
        try: requests.get(SELF_URL, timeout=5)
        except: pass
        time.sleep(50)

app = Flask(__name__)
@app.route("/")
def home(): return f"✅ RF Exact Bot — {SYMBOL} {INTERVAL} — {'LIVE' if MODE_LIVE else 'PAPER'}"

# =================== Boot ===================
threading.Thread(target=trade_loop, daemon=True).start()
threading.Thread(target=keepalive_loop, daemon=True).start()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
