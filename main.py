# -*- coding: utf-8 -*-
"""
Range Filter Bot — BingX Futures — ENV-driven
- Strategy: TradingView Range Filter (old DW-lite) 1:1
  longCondition / shortCondition مطابقان لكود Pine
- Execution: Close on opposite signal, then open new side immediately
- TP/SL: OCO based on ATR (تنفيذ فقط — لا يغير الإشارة)
- Security: يقرأ كل الإعدادات من ENV (لا نطبع مفاتيح/أسرار نهائياً)
- Logs: تعرض مؤشرات الاستراتيجية (filt/hi/lo/fdir/range_size/ATR) ملوّنة حسب الاتجاه

ENV الرئيسية المقترحة:
SYMBOL, INTERVAL, LEVERAGE, RISK_ALLOC, RF_PERIOD, RF_MULT, RF_SOURCE, USE_TV_BAR,
ATR_LEN, TP_ATR_MULT, SL_ATR_MULT, MAX_SLIPPAGE_PCT, DECISION_EVERY_S, PORT, SELF_URL,
BINGX_API_KEY, BINGX_API_SECRET
"""

import os, time, math, threading, requests
import pandas as pd
import ccxt
from flask import Flask, jsonify
from datetime import datetime

# ========= ENV =========
SYMBOL        = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL      = os.getenv("INTERVAL", "15m")
LEVERAGE      = int(float(os.getenv("LEVERAGE", "10")))
RISK_ALLOC    = float(os.getenv("RISK_ALLOC", "0.60"))
PORT          = int(float(os.getenv("PORT", "5000")))
SELF_URL      = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")

API_KEY       = os.getenv("BINGX_API_KEY", "")
API_SECRET    = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE     = bool(API_KEY and API_SECRET)

# Range Filter params (مطابقة للـ Pine)
RF_SOURCE     = os.getenv("RF_SOURCE", "close").lower()  # "close","open","high","low"
RF_PERIOD     = int(float(os.getenv("RF_PERIOD", "20")))
RF_MULT       = float(os.getenv("RF_MULT", "3.5"))

# Candle behavior (TV-compat)
USE_TV_BAR    = os.getenv("USE_TV_BAR", "false").lower() == "true"  # true=current bar, false=closed bar

# Execution guards
ATR_LEN           = int(float(os.getenv("ATR_LEN", "14")))
TP_ATR_MULT       = float(os.getenv("TP_ATR_MULT", "1.8"))
SL_ATR_MULT       = float(os.getenv("SL_ATR_MULT", "1.2"))
MAX_SLIPPAGE_PCT  = float(os.getenv("MAX_SLIPPAGE_PCT", "0.004"))  # 0.4%
DECISION_EVERY_S  = int(float(os.getenv("DECISION_EVERY_S", "60"))) # قرار كل دقيقة

# ========= UI / Colors =========
try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t
IC_HDR="📊"; IC_BAL="💰"; IC_PRC="💲"; IC_OK="✅"; IC_BAD="❌"; IC_SHD="🛡️"; IC_TRD="🚀"; IC_CLS="🔚"
SEP = colored("—"*96, "cyan")

def log(msg, color="white"): print(colored(msg, color), flush=True)
def fmt(v, d=6, na="N/A"):
    try:
        if v is None: return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

# أثناء BUY: أخضر، أثناء SELL: أحمر
def log_trade(msg, base_color="white"):
    if state.get("open"):
        if state["side"] == "long":  return log(msg, "green")
        if state["side"] == "short": return log(msg, "red")
    return log(msg, base_color)

def safe_symbol(s: str):
    if s.endswith(":USDT") or s.endswith(":USDC"): return s
    if "/USDT" in s and not s.endswith(":USDT"): return s + ":USDT"
    return s

# ========= Exchange =========
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

# ========= Account / Market =========
def balance_usdt():
    if not MODE_LIVE: return None  # ما نطلبش بالانس في الورقي
    try:
        b = ex.fetch_balance(params={"type":"swap"})
        return b.get("total", {}).get("USDT")
    except Exception as e:
        log(f"{IC_BAD} balance error: {e}", "red"); return None

def price_now():
    try:
        t = ex.fetch_ticker(safe_symbol(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception as e:
        log(f"{IC_BAD} ticker error: {e}", "red"); return None

def fetch_ohlcv(limit=400):
    rows = ex.fetch_ohlcv(safe_symbol(SYMBOL), timeframe=INTERVAL, limit=limit, params={"type":"swap"})
    df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
    return df

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
    bal = (balance if (MODE_LIVE and balance is not None) else 100.0)  # ورقي: افتراضي 100$
    raw = (bal * RISK_ALLOC * LEVERAGE) / price
    return market_amount(raw)

# ========= Wilder tools =========
def rma(series: pd.Series, length: int):
    r = [None]*len(series); s = series.astype(float).values
    acc = 0.0; n = 0
    for i, v in enumerate(s):
        if pd.isna(v): r[i] = math.nan; continue
        if n < length:
            acc += v; n += 1
            r[i] = math.nan if n < length else acc/length
        else:
            r[i] = (r[i-1]*(length-1) + v) / length
    return pd.Series(r, index=series.index, dtype="float64")

# ========= Range Filter (ترجمة Pine 1:1) =========
def rf_rng_size(x: pd.Series, qty: float, n: int):
    avrng = x.diff().abs().ewm(span=n, adjust=False).mean()
    wper  = (n*2) - 1
    AC    = avrng.ewm(span=wper, adjust=False).mean() * qty
    return AC  # = range_size

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
    fdir  = pd.Series(0, index=df.index)
    fdir = fdir.mask(dfilt > 0, 1).mask(dfilt < 0, -1).ffill().fillna(0)
    upward   = (fdir == 1).astype(int)
    downward = (fdir == -1).astype(int)

    longCond  = ((src > filt) & (src > src.shift(1)) & (upward > 0)) | ((src > filt) & (src < src.shift(1)) & (upward > 0))
    shortCond = ((src < filt) & (src < src.shift(1)) & (downward > 0)) | ((src < filt) & (src > src.shift(1)) & (downward > 0))

    CondIni = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if longCond.iloc[i]:      CondIni.iloc[i] = 1
        elif shortCond.iloc[i]:   CondIni.iloc[i] = -1
        else:                     CondIni.iloc[i] = CondIni.iloc[i-1]

    longSig  = longCond & (CondIni.shift(1) == -1)
    shortSig = shortCond & (CondIni.shift(1) == 1)

    # ATR لخدمة TP/SL فقط
    h=df["high"].astype(float); l=df["low"].astype(float); c=df["close"].astype(float)
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = rma(tr, ATR_LEN)

    i = len(df)-1 if USE_TV_BAR else len(df)-2
    def last(s): 
        v = s.iloc[i]; 
        return None if pd.isna(v) else float(v)
    out = {
        "price": last(c),
        "atr":   last(atr),
        "filt":  last(filt),
        "hi":    last(hi),
        "lo":    last(lo),
        "rsize": last(r),
        "fdir":  1 if (filt.iloc[i] - filt.iloc[i-1]) > 0 else (-1 if (filt.iloc[i] - filt.iloc[i-1]) < 0 else 0),
        "long":  bool(longSig.iloc[i]),
        "short": bool(shortSig.iloc[i]),
    }
    return out

# ========= OCO Protection =========
def attach_protection(side_opp, qty, tp, sl):
    try:
        ex.create_order(safe_symbol(SYMBOL), "take_profit_market", side_opp, qty,
                        params={"reduceOnly": True, "takeProfitPrice": tp})
        ex.create_order(safe_symbol(SYMBOL), "stop_market", side_opp, qty,
                        params={"reduceOnly": True, "stopLossPrice": sl})
        return True
    except Exception:
        try:
            ex.create_order(safe_symbol(SYMBOL), "limit", side_opp, qty, price=tp, params={"reduceOnly": True})
            ex.create_order(safe_symbol(SYMBOL), "stop",  side_opp, qty, params={"stopPrice": sl, "reduceOnly": True})
            return True
        except Exception as e2:
            log_trade(f"{IC_SHD} protection attach failed: {e2}", "yellow"); return False

def verify_oco(side_opp):
    try:
        oo = ex.fetch_open_orders(symbol=safe_symbol(SYMBOL))
        tp_ok = any(("take" in (o.get("type","").lower())) and o.get("side","").lower()==side_opp for o in oo)
        sl_ok = any(("stop" in (o.get("type","").lower())) and o.get("side","").lower()==side_opp for o in oo)
        return tp_ok and sl_ok
    except Exception as e:
        log_trade(f"{IC_SHD} verify_oco error: {e}", "yellow"); return False

# ========= State =========
state = {"open": False, "side": None, "entry": None, "qty": None, "tp": None, "sl": None, "pnl": 0.0}
compound_pnl = 0.0

def snapshot(balance, price, rf, pos, total_pnl):
    print()
    log_trade(SEP, "cyan")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    mode = "LIVE" if MODE_LIVE else "PAPER"
    log_trade(f"{IC_HDR} SNAPSHOT • {now} • mode={mode} • {safe_symbol(SYMBOL)} • {INTERVAL}", "cyan")
    log_trade("—", "cyan")
    log_trade(f"{IC_BAL} Balance (USDT): {fmt(balance,2) if MODE_LIVE else 'N/A (paper)'}", "yellow")
    if rf:
        log_trade(f"{IC_PRC} Price={fmt(rf.get('price'))} | filt={fmt(rf.get('filt'))} | hi={fmt(rf.get('hi'))} | lo={fmt(rf.get('lo'))}", "green")
        log_trade(f"ℹ️ fdir={rf.get('fdir')}  | range_size={fmt(rf.get('rsize'))}  | ATR({ATR_LEN})={fmt(rf.get('atr'))}")
    if pos["open"]:
        log_trade(f"🧭 Position: {pos['side'].upper()} | entry={fmt(pos['entry'])} qty={fmt(pos['qty'],4)}")
        log_trade(f"🎯/🛑 TP/SL : {fmt(pos['tp'])} / {fmt(pos['sl'])}")
        log_trade(f"📈 PnL     : {fmt(pos['pnl'])}")
    log_trade(f"📦 Compound PnL: {fmt(total_pnl)}", "yellow")
    log_trade(SEP, "cyan")

# ========= Open / Close =========
def place_protected_order(side, qty, ref_price, atr=None):
    global state
    # TP/SL based on ATR
    if not atr or atr <= 0:
        sl = ref_price * (0.985 if side=="buy" else 1.015)
        tp = ref_price * (1.015 if side=="buy" else 0.985)
    else:
        sl = ref_price - SL_ATR_MULT*atr if side=="buy" else ref_price + SL_ATR_MULT*atr
        tp = ref_price + TP_ATR_MULT*atr if side=="buy" else ref_price - TP_ATR_MULT*atr

    if not MODE_LIVE:
        state.update({"open": True, "side": "long" if side=="buy" else "short",
                      "entry": ref_price, "qty": qty, "tp": tp, "sl": sl, "pnl": 0.0})
        log_trade(f"{IC_TRD} {'BUY' if side=='buy' else 'SELL'} [PAPER] qty={fmt(qty,4)} entry={fmt(ref_price)} TP={fmt(tp)} SL={fmt(sl)}", "green")
        return

    try:
        ex.set_leverage(LEVERAGE, safe_symbol(SYMBOL), params={"side":"BOTH"})
    except Exception as e:
        log_trade(f"{IC_BAD} set_leverage: {e}", "red")

    try:
        ex.create_order(safe_symbol(SYMBOL), "market", side, qty, params={"reduceOnly": False})
        log_trade(f"{IC_TRD} submit {'BUY' if side=='buy' else 'SELL'} qty={fmt(qty,4)}", "green")
    except Exception as e:
        log_trade(f"{IC_BAD} open error: {e}", "red"); return

    try:
        time.sleep(0.9)
        poss = ex.fetch_positions([safe_symbol(SYMBOL)], params={"type":"swap"})
        pos = None
        for p in poss:
            if (p.get("symbol") == safe_symbol(SYMBOL)) and abs(float(p.get("contracts") or 0)) > 0:
                pos = p; break
        if not pos:
            log_trade(f"{IC_SHD} no real position detected; state not updated.", "yellow"); return

        entry = float(pos.get("entryPrice") or ref_price)
        size  = abs(float(pos.get("contracts") or qty))
        side_opp = "sell" if side=="buy" else "buy"

        attached = attach_protection(side_opp, size, tp, sl)
        if not attached or not verify_oco(side_opp):
            try: ex.create_order(safe_symbol(SYMBOL), "market", side_opp, size, params={"reduceOnly": True})
            except Exception as e3: log_trade(f"{IC_BAD} emergency close: {e3}", "red")
            log_trade(f"{IC_SHD} aborted entry since SL/TP not attached/verified.", "yellow")
            return

        state.update({"open": True, "side": "long" if side=="buy" else "short",
                      "entry": entry, "qty": size, "tp": tp, "sl": sl, "pnl": 0.0})
        log_trade(f"{IC_TRD} {'BUY' if side=='buy' else 'SELL'} CONFIRMED qty={fmt(size,4)} entry={fmt(entry)} TP={fmt(tp)} SL={fmt(sl)}","green")

    except Exception as e:
        log_trade(f"{IC_BAD} post-trade validation error: {e}", "red")

def close_position(reason):
    global state, compound_pnl
    if not state["open"]: return
    px   = price_now() or state["entry"]
    qty  = state["qty"]
    side = "sell" if state["side"]=="long" else "buy"

    if not MODE_LIVE:
        pnl = (px - state["entry"]) * qty * (1 if state["side"]=="long" else -1)
    else:
        try:
            ex.create_order(safe_symbol(SYMBOL), "market", side, qty, params={"reduceOnly": True})
        except Exception as e:
            log_trade(f"{IC_BAD} close error: {e}", "red")
        pnl = (px - state["entry"]) * qty * (1 if state["side"]=="long" else -1)

    compound_pnl += pnl
    log_trade(f"{IC_CLS} Close {state['side']} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta")
    state.update({"open": False, "side": None, "entry": None, "qty": None, "tp": None, "sl": None, "pnl": 0.0})

# ========= Main loop =========
def trade_loop():
    global state, compound_pnl
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()
            rf  = compute_rf(df) if df is not None and len(df) > 60 else {}

            if state["open"] and px:
                state["pnl"] = (px - state["entry"]) * state["qty"] if state["side"]=="long" else (state["entry"] - px) * state["qty"]

            snapshot(bal, px, rf or {}, state.copy(), compound_pnl)

            if not rf or not (px and (bal is not None or not MODE_LIVE)):
                time.sleep(DECISION_EVERY_S); continue

            side = "buy" if rf.get("long") else ("sell" if rf.get("short") else None)
            if side is None:
                log_trade(f"{IC_SHD} no_signal", "yellow")
                time.sleep(DECISION_EVERY_S); continue

            desired = "long" if side=="buy" else "short"
            ref = rf.get("price") or px
            if abs(px - ref)/ref > MAX_SLIPPAGE_PCT:
                log_trade(f"{IC_SHD} skip: slippage px={fmt(px)} ref={fmt(ref)}", "yellow")
                time.sleep(DECISION_EVERY_S); continue

            qty = compute_size(bal, px)
            if state["open"]:
                if state["side"] != desired:
                    close_position("opposite_signal")
                    place_protected_order(side, qty, px, atr=rf.get("atr"))
                else:
                    log_trade("ℹ️ already_in_position", "yellow")
            else:
                place_protected_order(side, qty, px, atr=rf.get("atr"))

        except Exception as e:
            log_trade(f"{IC_BAD} loop error: {e}", "red")

        time.sleep(DECISION_EVERY_S)

# ========= Keepalive / Flask =========
def keepalive_loop():
    if not SELF_URL:
        log("SELF_URL not set — keepalive disabled", "yellow"); return
    url = SELF_URL.rstrip("/")
    while True:
        try: requests.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

app = Flask(__name__)
@app.route("/")
def home(): return f"{IC_OK} RF Bot — {safe_symbol(SYMBOL)} {INTERVAL} — mode={'LIVE' if MODE_LIVE else 'PAPER'}"
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

# ========= Boot =========
threading.Thread(target=trade_loop, daemon=True).start()
threading.Thread(target=keepalive_loop, daemon=True).start()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
