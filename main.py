# -*- coding: utf-8 -*-
"""
RF Bot — Pretty Console / ENV / Signal-Only
- Strategy: Range Filter (old DW-lite) 1:1 (نفس Buy/Sell بتاع Pine)
- Execution: يقفل الصفقة فقط عند الإشارة العكسية، ويفتح الاتجاه الجديد فورًا
- Logs: لوحة مؤشرات ملوّنة + ASCII mini-chart + أقسام واضحة
- Safe: يعتمد على ENV فقط لمفاتيح BingX وغيرها

ENV المطلوبة (أمثلة):
  BINGX_API_KEY=xxxx
  BINGX_API_SECRET=yyyy
  SYMBOL=DOGE/USDT:USDT
  INTERVAL=15m
  LEVERAGE=10
  RISK_ALLOC=0.60
  RF_PERIOD=20
  RF_MULT=3.5
  RF_SOURCE=close
  USE_TV_BAR=false
  DECISION_EVERY_S=60
  MAX_SLIPPAGE_PCT=0.004
  PORT=5000
  SELF_URL=https://your-app.onrender.com   (اختياري)
"""

import os, time, math, threading, requests, shutil
import pandas as pd
import ccxt
from flask import Flask, jsonify
from datetime import datetime

# ============== ENV ==============
SYMBOL        = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL      = os.getenv("INTERVAL", "15m")
LEVERAGE      = int(float(os.getenv("LEVERAGE", "10")))
RISK_ALLOC    = float(os.getenv("RISK_ALLOC", "0.60"))
PORT          = int(float(os.getenv("PORT", "5000")))
SELF_URL      = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")

API_KEY       = os.getenv("BINGX_API_KEY", "")
API_SECRET    = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE     = bool(API_KEY and API_SECRET)

# Range Filter params
RF_SOURCE     = os.getenv("RF_SOURCE", "close").lower()  # close/open/high/low
RF_PERIOD     = int(float(os.getenv("RF_PERIOD", "20")))
RF_MULT       = float(os.getenv("RF_MULT", "3.5"))
USE_TV_BAR    = os.getenv("USE_TV_BAR", "false").lower() == "true"

# Loop / execution
DECISION_EVERY_S = int(float(os.getenv("DECISION_EVERY_S", "60")))
MAX_SLIPPAGE_PCT = float(os.getenv("MAX_SLIPPAGE_PCT", "0.004"))

# ============== UI helpers ==============
try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

def term_width():
    try: return shutil.get_terminal_size((120, 30)).columns
    except Exception: return 120

def line(char="─", color="cyan"):
    w = max(60, term_width()-2)
    print(colored(char*w, color), flush=True)

def fmt(v, d=6):
    try:
        return f"{float(v):.{d}f}"
    except Exception:
        return "N/A"

def colorize_by_side(text):
    if state.get("open"):
        if state["side"]=="long":  return colored(text, "green")
        if state["side"]=="short": return colored(text, "red")
    return colored(text, "white")

def kv(key, val, kcolor="white", vcolor="white"):
    return f"{colored(key, kcolor)} {colored(val, vcolor)}"

def sparkline(series, width=40):
    """ASCII mini chart for price."""
    try:
        s = pd.Series(series).dropna()
        if s.size < 2: return ""
        s = s.tail(width)
        lo, hi = float(s.min()), float(s.max())
        rng = max(hi-lo, 1e-12)
        blocks = "▁▂▃▄▅▆▇█"
        out = []
        for v in s:
            idx = int((v - lo) / rng * (len(blocks)-1))
            out.append(blocks[idx])
        return "".join(out)
    except Exception:
        return ""

# ============== Exchange ==============
def safe_symbol(s: str):
    if s.endswith(":USDT") or s.endswith(":USDC"): return s
    if "/USDT" in s and not s.endswith(":USDT"): return s + ":USDT"
    return s

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
except Exception: pass

def balance_usdt():
    if not MODE_LIVE: return None
    try:
        b = ex.fetch_balance(params={"type":"swap"})
        return b.get("total", {}).get("USDT")
    except Exception as e:
        print(colored(f"❌ balance error: {e}", "red"), flush=True); return None

def price_now():
    try:
        t = ex.fetch_ticker(safe_symbol(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception as e:
        print(colored(f"❌ ticker error: {e}", "red"), flush=True); return None

def fetch_ohlcv(limit=500):
    rows = ex.fetch_ohlcv(safe_symbol(SYMBOL), timeframe=INTERVAL, limit=limit, params={"type":"swap"})
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def market_amount(amount):
    try:
        m = ex.market(safe_symbol(SYMBOL))
        prec = int(m.get("precision",{}).get("amount", 3))
        min_amt = m.get("limits",{}).get("amount",{}).get("min", 0.001)
        amt = float(f"{float(amount):.{prec}f}")
        return max(amt, float(min_amt or 0.001))
    except Exception:
        return float(amount)

def compute_size(balance, price):
    if not price: return 0
    bal = (balance if (MODE_LIVE and balance is not None) else 100.0)
    raw = (bal * RISK_ALLOC * LEVERAGE) / price
    return market_amount(raw)

# ============== Range Filter (Pine 1:1) ==============
def rf_rng_size(x: pd.Series, qty: float, n: int):
    avrng = x.diff().abs().ewm(span=n, adjust=False).mean()
    wper  = (n*2) - 1
    return avrng.ewm(span=wper, adjust=False).mean() * qty

def rf_filter(x: pd.Series, r: pd.Series):
    filt_vals = [x.iloc[0]]
    for i in range(1, len(x)):
        prev = filt_vals[-1]; xi = x.iloc[i]; ri = r.iloc[i]
        if xi - ri > prev: cur = xi - ri
        elif xi + ri < prev: cur = xi + ri
        else: cur = prev
        filt_vals.append(cur)
    filt = pd.Series(filt_vals, index=x.index)
    hi   = filt + r
    lo   = filt - r
    return hi, lo, filt

def compute_rf(df: pd.DataFrame):
    src = df[RF_SOURCE].astype(float)
    r   = rf_rng_size(src, RF_MULT, RF_PERIOD)
    hi, lo, filt = rf_filter(src, r)

    dfilt = filt - filt.shift(1)
    fdir = pd.Series(0, index=df.index)
    fdir = fdir.mask(dfilt > 0, 1).mask(dfilt < 0, -1).ffill().fillna(0)
    upward   = (fdir == 1).astype(int)
    downward = (fdir == -1).astype(int)

    longCond  = ((src > filt) & (src > src.shift(1)) & (upward > 0)) | ((src > filt) & (src < src.shift(1)) & (upward > 0))
    shortCond = ((src < filt) & (src < src.shift(1)) & (downward > 0)) | ((src < filt) & (src > src.shift(1)) & (downward > 0))

    CondIni = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if longCond.iloc[i]:    CondIni.iloc[i] = 1
        elif shortCond.iloc[i]: CondIni.iloc[i] = -1
        else:                   CondIni.iloc[i] = CondIni.iloc[i-1]

    longSig  = longCond & (CondIni.shift(1) == -1)
    shortSig = shortCond & (CondIni.shift(1) == 1)

    i = len(df)-1 if USE_TV_BAR else len(df)-2
    def last(s):
        v = s.iloc[i]
        return None if pd.isna(v) else float(v)

    return {
        "idx": int(i),
        "price": last(df["close"].astype(float)),
        "filt":  last(filt),
        "hi":    last(hi),
        "lo":    last(lo),
        "rsize": last(r),
        "fdir":  1 if (filt.iloc[i] - filt.iloc[i-1]) > 0 else (-1 if (filt.iloc[i] - filt.iloc[i-1]) < 0 else 0),
        "long":  bool(longSig.iloc[i]),
        "short": bool(shortSig.iloc[i]),
        "series_tail": df["close"].astype(float).tail(80).tolist()
    }

# ============== State & Pretty Logs ==============
state = {"open": False, "side": None, "entry": None, "qty": None, "pnl": 0.0}
compound_pnl = 0.0

def snapshot(balance, rf, position):
    # Header
    line("═", "cyan")
    mode = "LIVE" if MODE_LIVE else "PAPER"
    hdr  = f"📊 RF BOT • {safe_symbol(SYMBOL)} • {INTERVAL} • {mode} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    print(colored(hdr, "cyan"), flush=True)
    line("─", "cyan")

    # Indicators Panel (left) + Position Panel (right)
    price = rf.get("price")
    fdir  = rf.get("fdir")
    updn  = colored("↑ Up", "green") if fdir==1 else (colored("↓ Down", "red") if fdir==-1 else colored("• Flat", "yellow"))
    print(colorize_by_side("▌ INDICATORS"))
    print(kv("   Price   :", fmt(price), "white", "green"))
    print(kv("   Filter  :", fmt(rf.get('filt')), "white", "cyan"))
    print(kv("   Band Hi :", fmt(rf.get('hi')), "white", "cyan"))
    print(kv("   Band Lo :", fmt(rf.get('lo')), "white", "cyan"))
    print(kv("   RangeSz :", fmt(rf.get('rsize')), "white", "magenta"))
    print(kv("   Dir     :", updn, "white", "white"))
    print(kv("   LongSig :", str(rf.get('long')), "white", "green"))
    print(kv("   ShortSig:", str(rf.get('short')), "white", "red"))
    print()

    print(colorize_by_side("▌ POSITION"))
    bal_txt = "N/A (paper)" if not MODE_LIVE else fmt(balance,2)
    print(kv("   Balance :", bal_txt, "white", "yellow"))
    if position["open"]:
        side = colored(position['side'].upper(), "green" if position['side']=="long" else "red")
        print(kv("   Status  :", f"OPEN {side}", "white", "white"))
        print(kv("   Entry   :", fmt(position['entry']), "white", "white"))
        print(kv("   Qty     :", fmt(position['qty'], 4), "white", "white"))
        print(kv("   PnL     :", fmt(position['pnl']), "white", "white"))
    else:
        print(kv("   Status  :", "FLAT", "white", "white"))
    print()

    # Mini Chart
    print(colorize_by_side("▌ PRICE MINI-CHART"))
    print("   " + colored(sparkline(rf.get("series_tail", []), width=60), "white"))
    line("─", "cyan")

def log_event(msg):
    print(colorize_by_side("• " + msg), flush=True)

# ============== Open/Close (Signal-only) ==============
def open_market(side, qty, ref_price):
    global state
    if not MODE_LIVE:
        state.update({"open": True, "side": "long" if side=="buy" else "short",
                      "entry": ref_price, "qty": qty, "pnl": 0.0})
        log_event(f"OPEN [{('BUY' if side=='buy' else 'SELL')}] PAPER qty={fmt(qty,4)} @ {fmt(ref_price)}")
        return
    try:
        ex.set_leverage(LEVERAGE, safe_symbol(SYMBOL), params={"side":"BOTH"})
    except Exception as e:
        print(colored(f"❌ set_leverage: {e}", "red"))
    try:
        ex.create_order(safe_symbol(SYMBOL), "market", side, qty, params={"reduceOnly": False})
        log_event(f"SUBMIT {('BUY' if side=='buy' else 'SELL')} qty={fmt(qty,4)}")
        time.sleep(0.9)
        poss = ex.fetch_positions([safe_symbol(SYMBOL)], params={"type":"swap"})
        entry = ref_price; size = qty
        for p in poss:
            if p.get("symbol") == safe_symbol(SYMBOL) and abs(float(p.get("contracts") or 0)) > 0:
                entry = float(p.get("entryPrice") or ref_price)
                size  = abs(float(p.get("contracts") or qty))
                break
        state.update({"open": True, "side": "long" if side=='buy' else 'short',
                      "entry": entry, "qty": size, "pnl": 0.0})
        log_event(f"✅ OPEN CONFIRMED entry={fmt(entry)} qty={fmt(size,4)}")
    except Exception as e:
        print(colored(f"❌ open error: {e}", "red"))

def close_market(reason):
    global state, compound_pnl
    if not state["open"]: return
    px  = price_now() or state["entry"]
    qty = state["qty"]
    side= "sell" if state["side"]=="long" else "buy"
    if not MODE_LIVE:
        pnl = (px - state["entry"]) * qty * (1 if state["side"]=="long" else -1)
    else:
        try:
            ex.create_order(safe_symbol(SYMBOL), "market", side, qty, params={"reduceOnly": True})
        except Exception as e:
            print(colored(f"❌ close error: {e}", "red"))
        pnl = (px - state["entry"]) * qty * (1 if state["side"]=="long" else -1)
    compound_pnl += pnl
    print(colored(f"🔚 CLOSE {state['side']} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}", "magenta"))
    state.update({"open": False, "side": None, "entry": None, "qty": None, "pnl": 0.0})

# ============== Loop ==============
def trade_loop():
    global state, compound_pnl
    last_side_print = None
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()
            if df is None or len(df) < 60:
                time.sleep(DECISION_EVERY_S); continue

            rf  = compute_rf(df)
            # Update PnL live
            if state["open"] and px:
                state["pnl"] = (px - state["entry"]) * state["qty"] if state["side"]=="long" else (state["entry"] - px) * state["qty"]

            snapshot(bal, rf, state.copy())

            # Decide
            side = "buy" if rf.get("long") else ("sell" if rf.get("short") else None)
            if side is None:
                log_event("no_signal")
                time.sleep(DECISION_EVERY_S); continue

            desired = "long" if side=="buy" else "short"
            ref = rf.get("price") or px
            if not px or not ref:
                log_event("skip: no price")
                time.sleep(DECISION_EVERY_S); continue
            if abs(px - ref)/ref > MAX_SLIPPAGE_PCT:
                log_event(f"skip: slippage px={fmt(px)} ref={fmt(ref)}")
                time.sleep(DECISION_EVERY_S); continue

            qty = compute_size(bal, px)
            if state["open"]:
                if state["side"] != desired:
                    close_market("opposite_signal")
                    open_market(side, qty, px)
                else:
                    if last_side_print != state["side"]:
                        log_event("already_in_position")
                        last_side_print = state["side"]
            else:
                open_market(side, qty, px)
                last_side_print = desired

        except Exception as e:
            print(colored(f"❌ loop error: {e}", "red"))

        time.sleep(DECISION_EVERY_S)

# ============== Keepalive / API ==============
def keepalive_loop():
    if not SELF_URL:
        print(colored("SELF_URL not set — keepalive disabled", "yellow")); return
    url = SELF_URL.rstrip("/")
    while True:
        try: requests.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

app = Flask(__name__)
@app.route("/")
def home(): return f"✅ RF Pretty Bot — {safe_symbol(SYMBOL)} {INTERVAL} — {'LIVE' if MODE_LIVE else 'PAPER'}"
@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": safe_symbol(SYMBOL),
        "interval": INTERVAL,
        "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE,
        "risk_alloc": RISK_ALLOC,
        "balance": balance_usdt(),
        "price": price_now(),
        "position": state,
        "compound_pnl": compound_pnl,
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    })

# ============== Boot ==============
threading.Thread(target=trade_loop, daemon=True).start()
threading.Thread(target=keepalive_loop, daemon=True).start()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
