# -*- coding: utf-8 -*-
"""
RF Bot — EXACT TradingView Range Filter (B&S) — Close-Bar + Confirm Bars
- مطابق 1:1 لكود Pine (المُرسل) لحساب (Filter / Bands / Long/Short conditions)
- تنفيذ على آخر شمعة مُقفلة (افتراضيًا) + تأكيد إشارات لعدد CONFIRM_BARS
- فتح/إغلاق فقط بالإشارة العكسية (Flip) — بدون SL/TP
- Compound PnL + Effective Equity في اللوجز
- لوجز احترافية بالألوان والأيقونات
"""

import os, time, threading, requests
import pandas as pd
import ccxt
from flask import Flask, jsonify
from datetime import datetime

# ======================= ENV / CONFIG =======================
SYMBOL       = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL     = os.getenv("INTERVAL", "15m")
LEVERAGE     = int(float(os.getenv("LEVERAGE", "10")))
RISK_ALLOC   = float(os.getenv("RISK_ALLOC", "0.60"))          # نسبة من الرصيد × الرافعة
SELF_URL     = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT         = int(float(os.getenv("PORT", "5000")))

BINGX_API_KEY    = os.getenv("BINGX_API_KEY", "")
BINGX_API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE = bool(BINGX_API_KEY and BINGX_API_SECRET)           # True=LIVE إذا المفاتيح موجودة

# إعدادات المؤشر (مطابقة Pine)
RF_SOURCE   = os.getenv("RF_SOURCE", "close").lower()          # close/open/high/low
RF_PERIOD   = int(float(os.getenv("RF_PERIOD", "20")))
RF_MULT     = float(os.getenv("RF_MULT", "3.5"))

# تنفيذ الإشارات
USE_TV_BAR     = os.getenv("USE_TV_BAR", "false").lower() == "true"  # True=اقرأ الشمعة الحالية. False=آخر شمعة مُقفلة (مُفضل)
CONFIRM_BARS   = int(float(os.getenv("CONFIRM_BARS", "1")))          # كم شمعة مقفلة متتالية لازم يكون عليها الإشارة
DECISION_EVERY = int(float(os.getenv("DECISION_EVERY_S", "30")))     # كل كام ثانية تتخذ قرار

# ======================= UI / LOGS =======================
try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

IC_HDR="🧭"; IC_IND="📈"; IC_POS="🧳"; IC_RES="📦"
IC_UP="🟢"; IC_DN="🔴"; IC_FL="⚪"
IC_BUY="🟩 BUY"; IC_SELL="🟥 SELL"
IC_BAL="💰"; IC_PRC="💲"; IC_FIL="📏"; IC_HI="🔼"; IC_LO="🔽"
IC_OK="✅"; IC_ERR="❌"; IC_INFO="ℹ️"

def sep():
    print(colored("─"*98, "cyan"), flush=True)

def safe_symbol(s):
    if s.endswith(":USDT") or s.endswith(":USDC"): return s
    if "/USDT" in s and not s.endswith(":USDT"): return s + ":USDT"
    return s

def fmt(v, d=6, na="N/A"):
    try:
        if v is None: return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

def log_side_line(text):
    if state["open"]:
        c = "green" if state["side"] == "long" else "red"
        print(colored(text, c), flush=True)
    else:
        print(text, flush=True)

print(colored(f"{IC_INFO} MODE: {'LIVE' if MODE_LIVE else 'PAPER'} | keys:{'✔' if MODE_LIVE else '✖'}", "yellow"))

# ======================= EXCHANGE =======================
def make_exchange():
    return ccxt.bingx({
        "apiKey": BINGX_API_KEY,
        "secret": BINGX_API_SECRET,
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
        b = ex.fetch_balance(params={"type": "swap"})
        return b.get("total", {}).get("USDT") or b.get("free", {}).get("USDT")
    except Exception as e:
        print(colored(f"{IC_ERR} balance error: {e}", "red")); return None

def price_now():
    try:
        t = ex.fetch_ticker(safe_symbol(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception:
        return None

def fetch_ohlcv(limit=600):
    rows = ex.fetch_ohlcv(safe_symbol(SYMBOL), timeframe=INTERVAL, limit=limit, params={"type":"swap"})
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def market_amount(amount):
    try:
        m = ex.market(safe_symbol(SYMBOL))
        prec = int(m.get("precision",{}).get("amount",3))
        min_amt = m.get("limits",{}).get("amount",{}).get("min",0.001)
        amt = float(f"{float(amount):.{prec}f}")
        return max(amt, float(min_amt or 0.001))
    except Exception:
        return float(amount)

def compute_size(balance, price):
    bal = (balance if (MODE_LIVE and balance is not None) else 100.0)
    notional = bal * RISK_ALLOC * LEVERAGE
    qty = notional / max(price, 1e-9)
    return market_amount(qty)

# =================== Range Filter (Pine 1:1) ===================
def _ema(s: pd.Series, n: int): return s.ewm(span=n, adjust=False).mean()

def _rng_size(src: pd.Series, qty: float, n: int) -> pd.Series:
    avrng = _ema((src - src.shift(1)).abs(), n)
    wper  = (n*2) - 1
    return _ema(avrng, wper) * qty

def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf = [float(src.iloc[0])]
    for i in range(1, len(src)):
        prev = rf[-1]; x = float(src.iloc[i]); r = float(rsize.iloc[i]); cur = prev
        if x - r > prev: cur = x - r
        if x + r < prev: cur = x + r
        rf.append(cur)
    filt = pd.Series(rf, index=src.index, dtype="float64")
    hi = filt + rsize
    lo = filt - rsize
    return hi, lo, filt

def compute_tv_signals(df: pd.DataFrame):
    if df is None or len(df) < RF_PERIOD + 5: return None
    src = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))

    dfilt = filt - filt.shift(1)
    fdir = pd.Series(0.0, index=filt.index).mask(dfilt>0,1).mask(dfilt<0,-1).ffill().fillna(0.0)
    upward   = (fdir==1).astype(int)
    downward = (fdir==-1).astype(int)

    src_gt_f = (src >  filt); src_lt_f = (src <  filt)
    src_gt_p = (src >  src.shift(1)); src_lt_p = (src <  src.shift(1))

    longCond  = (src_gt_f & src_gt_p & (upward  > 0)) | (src_gt_f & src_lt_p & (upward  > 0))
    shortCond = (src_lt_f & src_lt_p & (downward > 0)) | (src_lt_f & src_gt_p & (downward > 0))

    CondIni = pd.Series(0, index=src.index)
    for i in range(1, len(src)):
        if bool(longCond.iloc[i]):       CondIni.iloc[i] =  1
        elif bool(shortCond.iloc[i]):    CondIni.iloc[i] = -1
        else:                            CondIni.iloc[i] = CondIni.iloc[i-1]

    longSignal  = longCond  & (CondIni.shift(1) == -1)
    shortSignal = shortCond & (CondIni.shift(1) ==  1)

    # اختيار الشمعة: i = الأخيرة المقفولة أو الحالية حسب الإعداد
    i = len(df)-1 if USE_TV_BAR else len(df)-2

    def last_at(s: pd.Series):
        v = s.iloc[i]
        return None if pd.isna(v) else float(v)

    return {
        "index": i,
        "time": int(df["time"].iloc[i]),
        "price": last_at(df["close"].astype(float)),
        "filter": last_at(filt),
        "hi": last_at(hi),
        "lo": last_at(lo),
        "fdir":  1.0 if (filt.iloc[i] > filt.iloc[i-1]) else (-1.0 if (filt.iloc[i] < filt.iloc[i-1]) else float(fdir.iloc[i-1] if i-1>=0 else 0.0)),
        "longSigSeries":  longSignal.astype(bool),
        "shortSigSeries": shortSignal.astype(bool),
        "long": bool(longSignal.iloc[i]),
        "short": bool(shortSignal.iloc[i]),
    }

def signal_confirmed(sig_series: pd.Series, i: int, bars: int) -> bool:
    """يتأكد أن الإشارة True لعدد (bars) من الشموع المقفولة قبل الدخول."""
    if bars <= 0: return bool(sig_series.iloc[i])
    start = i - (bars - 1)
    if start < 0: return False
    window = sig_series.iloc[start:i+1]
    return bool(window.all())

# =================== STATE ===================
state = {"open": False, "side": None, "entry": None, "qty": None, "pnl": 0.0}
compound_pnl = 0.0

# =================== LOG SNAPSHOT ===================
def snapshot(balance, info):
    sep()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(colored(f"{IC_HDR} RF BOT {safe_symbol(SYMBOL)} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {now}", "cyan"))
    sep()
    log_side_line(f"{IC_IND} INDICATORS")
    print(f"  {IC_PRC} Price  : {fmt(info.get('price'))}")
    print(f"  {IC_FIL} Filter : {fmt(info.get('filter'))}")
    print(f"  {IC_HI} BandHi : {fmt(info.get('hi'))}")
    print(f"  {IC_LO} BandLo : {fmt(info.get('lo'))}")
    d = IC_UP+' Up' if info and info.get("fdir")==1 else (IC_DN+' Down' if info and info.get("fdir")==-1 else IC_FL+' Flat')
    print(f"  🧭 Direction  : {d}")
    if info:
        # إشارات مؤكدة حسب CONFIRM_BARS وعلى الشمعة المختارة
        i = info["index"]
        long_ok  = signal_confirmed(info["longSigSeries"],  i, CONFIRM_BARS)
        short_ok = signal_confirmed(info["shortSigSeries"], i, CONFIRM_BARS)
        print(f"  {IC_BUY} Confirmed : {long_ok}")
        print(f"  {IC_SELL} Confirmed: {short_ok}")
    print()
    log_side_line(f"{IC_POS} POSITION")
    bal_txt = "N/A (paper)" if not MODE_LIVE else fmt(balance, 2)
    print(f"  {IC_BAL} Balance : {bal_txt} USDT")
    if state["open"]:
        print(f"  📌 Status  : {'🟩 LONG' if state['side']=='long' else '🟥 SHORT'}")
        print(f"  🎯 Entry   : {fmt(state['entry'])}")
        print(f"  📦 Qty     : {fmt(state['qty'], 4)}")
        print(f"  📊 PnL     : {fmt(state['pnl'])}")
    else:
        print(f"  📌 Status  : {IC_FL} FLAT")
    print()
    print(f"{IC_RES} RESULTS")
    eff_eq = ((balance or 0.0) + compound_pnl) if MODE_LIVE else compound_pnl
    print(f"  🧮 Compound PnL     : {fmt(compound_pnl)}")
    print(f"  🚀 Effective Equity : {fmt(eff_eq)} USDT")
    sep()

# =================== EXECUTION ===================
def open_market(side, qty, ref_price):
    global state
    if not MODE_LIVE:
        state.update({"open": True, "side": "long" if side=='buy' else "short",
                      "entry": ref_price, "qty": qty, "pnl": 0.0})
        print(colored(f"{IC_OK} OPEN {side.upper()} [PAPER] qty={fmt(qty,4)} @ {fmt(ref_price)}", "green" if side=='buy' else "red"))
        return
    try:
        ex.set_leverage(LEVERAGE, safe_symbol(SYMBOL), params={"side":"BOTH"})
    except Exception as e:
        print(colored(f"⚠️ set_leverage: {e}", "yellow"))
    try:
        ex.create_order(safe_symbol(SYMBOL), "market", side, qty, params={"reduceOnly": False})
        time.sleep(0.7)
        entry = ref_price; size = qty
        poss = ex.fetch_positions([safe_symbol(SYMBOL)], params={"type":"swap"})
        for p in poss:
            if p.get("symbol")==safe_symbol(SYMBOL) and abs(float(p.get("contracts") or 0))>0:
                entry = float(p.get("entryPrice") or ref_price); size = abs(float(p.get("contracts") or qty)); break
        state.update({"open": True, "side": "long" if side=='buy' else "short",
                      "entry": entry, "qty": size, "pnl": 0.0})
        print(colored(f"{IC_OK} OPEN {side.upper()} CONFIRMED qty={fmt(size,4)} @ {fmt(entry)}", "green" if side=='buy' else "red"))
    except Exception as e:
        print(colored(f"{IC_ERR} open error: {e}", "red"))

def close_market(reason):
    global state, compound_pnl
    if not state["open"]: return
    px  = price_now() or state["entry"]
    qty = state["qty"]
    side = "sell" if state["side"]=="long" else "buy"
    if not MODE_LIVE:
        pnl = (px - state["entry"]) * qty * (1 if state["side"]=="long" else -1)
    else:
        try:
            ex.create_order(safe_symbol(SYMBOL), "market", side, qty, params={"reduceOnly": True})
        except Exception as e:
            print(colored(f"{IC_ERR} close error: {e}", "red"))
        pnl = (px - state["entry"]) * qty * (1 if state["side"]=="long" else -1)
    compound_pnl += pnl
    print(colored(f"🔚 CLOSE {state['side']} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}", "magenta"))
    state.update({"open": False, "side": None, "entry": None, "qty": None, "pnl": 0.0})

# =================== MAIN LOOP ===================
def trade_loop():
    global state
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()
            info = compute_tv_signals(df)

            # تحديث PnL الجاري
            if info and state["open"] and px:
                state["pnl"] = (px - state["entry"]) * state["qty"] if state["side"]=="long" else (state["entry"] - px) * state["qty"]

            snapshot(bal, info or {})

            if not info or not px:
                time.sleep(DECISION_EVERY); continue

            i = info["index"]
            # إشارات مؤكدة (على الشموع المختارة)
            long_ok  = signal_confirmed(info["longSigSeries"],  i, CONFIRM_BARS)
            short_ok = signal_confirmed(info["shortSigSeries"], i, CONFIRM_BARS)

            desired_side = "buy" if long_ok else ("sell" if short_ok else None)
            if desired_side:
                want = "long" if desired_side=="buy" else "short"
                qty  = compute_size(bal, px)

                if state["open"]:
                    if state["side"] != want:   # Flip على الإشارة العكسية
                        close_market("opposite_signal")
                        open_market(desired_side, qty, px)
                    else:
                        print(colored(f"{IC_INFO} already in same direction", "yellow"))
                else:
                    open_market(desired_side, qty, px)

        except Exception as e:
            print(colored(f"{IC_ERR} loop error: {e}", "red"))

        time.sleep(DECISION_EVERY)

# =================== KEEPALIVE & API ===================
def keepalive_loop():
    if not SELF_URL: return
    url = SELF_URL.rstrip("/")
    while True:
        try: requests.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

app = Flask(__name__)
@app.route("/")
def home():
    return f"{IC_OK} RF Exact Bot — {safe_symbol(SYMBOL)} {INTERVAL} — {'LIVE' if MODE_LIVE else 'PAPER'}"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": safe_symbol(SYMBOL),
        "interval": INTERVAL,
        "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE,
        "risk_alloc": RISK_ALLOC,
        "rf_period": RF_PERIOD,
        "rf_mult": RF_MULT,
        "rf_source": RF_SOURCE,
        "use_tv_bar": USE_TV_BAR,
        "confirm_bars": CONFIRM_BARS,
        "balance": balance_usdt(),
        "price": price_now(),
        "position": state,
        "compound_pnl": compound_pnl,
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    })

# =================== BOOT ===================
threading.Thread(target=trade_loop, daemon=True).start()
threading.Thread(target=keepalive_loop, daemon=True).start()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
