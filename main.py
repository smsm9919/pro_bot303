# -*- coding: utf-8 -*-
"""
RF Bot — TradingView Range Filter (Pine 1:1) + Signal Hold + ENV (Render/Heroku friendly)
- يقرأ المفاتيح والإعدادات من Environment Variables (لا يحتاج .env على Render)
- إشارات BUY/SELL مطابقة للـ Pine
- ينتظر 120 ثانية لتأكيد الإشارة (منع الإشارات الوهمية) قبل التنفيذ
- فتح/إغلاق فقط بالإشارة العكسية (بدون SL/TP)
- Logs ملوّنة ومنظمة + Compound PnL + Effective Equity
- حمايات خفيفة: Slippage / Daily Loss Limit / ATR Circuit Breaker
- /metrics API للمتابعة

ENV المطلوبة (حسب صورتك): 
BINGX_API_KEY, BINGX_API_SECRET, SYMBOL, INTERVAL, LEVERAGE, RISK_ALLOC, SELF_URL (اختياري), TRADE_MODE (اختياري/لن يستخدم)
يمكنك إضافة مفاتيح ضبط إضافية مذكورة أدناه (قِيَم افتراضية موجودة هنا).
"""

import os, time, threading, requests, shutil
import pandas as pd
import ccxt
from flask import Flask, jsonify
from datetime import datetime, date

# =====================[ قراءة ENV + افتراضات آمنة ]=====================
SYMBOL        = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL      = os.getenv("INTERVAL", "15m")
LEVERAGE      = int(float(os.getenv("LEVERAGE", "10")))
RISK_ALLOC    = float(os.getenv("RISK_ALLOC", "0.60"))
SELF_URL      = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT          = int(float(os.getenv("PORT", "5000")))

API_KEY       = os.getenv("BINGX_API_KEY", "")
API_SECRET    = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE     = bool(API_KEY and API_SECRET)   # لو في مفاتيح = LIVE، غير كده = PAPER

# Range Filter مطابق Pine
RF_SOURCE     = os.getenv("RF_SOURCE", "close").lower()   # close/open/high/low
RF_PERIOD     = int(float(os.getenv("RF_PERIOD", "20")))
RF_MULT       = float(os.getenv("RF_MULT", "3.5"))

# سلوك القرار
USE_TV_BAR        = os.getenv("USE_TV_BAR", "true").lower() == "true"   # true=إحساس TV
DECISION_EVERY_S  = int(float(os.getenv("DECISION_EVERY_S", "30")))
MAX_SLIPPAGE_PCT  = float(os.getenv("MAX_SLIPPAGE_PCT", "0.004"))       # 0.4%
COOLDOWN_BARS     = int(float(os.getenv("COOLDOWN_BARS", "0")))

# Signal Hold لتأكيد الإشارة (منع الإشارات الوهمية)
SIGNAL_HOLD_S     = int(float(os.getenv("SIGNAL_HOLD_S", "120")))       # 120 ثانية
SIGNAL_MAX_AGE_S  = int(float(os.getenv("SIGNAL_MAX_AGE_S", "900")))    # 15 دقيقة

# حمايات اختيارية
USE_CIRCUIT_BREAKER = os.getenv("USE_CIRCUIT_BREAKER", "true").lower() == "true"
CB_ATR_MULT         = float(os.getenv("CB_ATR_MULT", "3.5"))
CB_PAUSE_BARS       = int(float(os.getenv("CB_PAUSE_BARS", "2")))

USE_DAILY_LOSS_LIMIT = os.getenv("USE_DAILY_LOSS_LIMIT", "true").lower() == "true"
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.08"))

# =====================[ UI ]=====================
try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

def term_w():
    try: return shutil.get_terminal_size((120,30)).columns
    except Exception: return 120

def line(char="─", color="cyan"): print(colored(char*max(72, term_w()-2), color), flush=True)
def fmt(v, d=6, na="N/A"):
    try:
        if v is None: return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

def safe_symbol(s):
    if s.endswith(":USDT") or s.endswith(":USDC"): return s
    if "/USDT" in s and not s.endswith(":USDT"): return s + ":USDT"
    return s

def colorize_by_side(text):
    if state.get("open"):
        return colored(text, "green" if state["side"]=="long" else "red")
    return text

print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} | apiKey:{'✔' if API_KEY else '✖'} secret:{'✔' if API_SECRET else '✖'}", "yellow"))

# =====================[ Exchange ]=====================
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
    """قراءة مرنة للرصيد + طباعة سبب الفشل إن وجد."""
    if not MODE_LIVE: return None
    try:
        b = ex.fetch_balance(params={"type":"swap"})
        usdt = None
        if "USDT" in b.get("total", {}): usdt = b["total"]["USDT"]
        elif "USDT" in b.get("free", {}): usdt = b["free"]["USDT"]
        if usdt is None and "info" in b:
            info = b["info"]
            for key in ("data","balances","assets"):
                arr = info.get(key)
                if isinstance(arr, list):
                    for it in arr:
                        sym = (it.get("asset") or it.get("currency") or it.get("code") or "").upper()
                        if sym=="USDT":
                            usdt = float(it.get("total", it.get("balance", it.get("availableBalance", 0.0))))
                            break
        if usdt is None: print(colored(f"⚠️ balance parse fail: {b}", "yellow"))
        return usdt
    except Exception as e:
        print(colored(f"❌ balance error: {e}", "red"))
        return None

def price_now():
    try:
        t = ex.fetch_ticker(safe_symbol(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception:
        return None

def fetch_ohlcv(limit=500):
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
    if not price: return 0.0
    bal = (balance if (MODE_LIVE and balance is not None) else 100.0)
    raw = (bal * RISK_ALLOC * LEVERAGE) / price
    return market_amount(raw)

# =====================[ Range Filter Pine 1:1 ]=====================
def _ema(s: pd.Series, n: int): return s.ewm(span=n, adjust=False).mean()
def _rng_size(src: pd.Series, qty: float, n: int) -> pd.Series:
    avrng = _ema((src - src.shift(1)).abs(), n); wper = (n*2) - 1
    return _ema(avrng, wper) * qty
def _rng_filter(src: pd.Series, rsize: pd.Series):
    rfilt_vals = [float(src.iloc[0])]
    for i in range(1, len(src)):
        prev = rfilt_vals[-1]; x=float(src.iloc[i]); r=float(rsize.iloc[i]); cur=prev
        if x - r > prev: cur = x - r
        if x + r < prev: cur = x + r
        rfilt_vals.append(cur)
    rfilt = pd.Series(rfilt_vals, index=src.index, dtype="float64")
    hi = rfilt + rsize; lo = rfilt - rsize
    return hi, lo, rfilt

def _atr_last(df: pd.DataFrame, n: int = 14):
    h=df["high"].astype(float); l=df["low"].astype(float); c=df["close"].astype(float)
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean().iloc[-1] if len(tr) else None

def compute_rf(df: pd.DataFrame, use_tv_bar: bool, rf_source: str, rf_period: int, rf_mult: float):
    s = df[rf_source].astype(float)
    hi, lo, filt = _rng_filter(s, _rng_size(s, rf_mult, rf_period))
    dfilt = filt - filt.shift(1)
    fdir = pd.Series(0.0, index=filt.index).mask(dfilt>0,1).mask(dfilt<0,-1).ffill().fillna(0.0)
    upward=(fdir==1).astype(int); downward=(fdir==-1).astype(int)
    src_gt_f=(s>filt); src_lt_f=(s<filt); src_gt_p=(s>s.shift(1)); src_lt_p=(s<s.shift(1))
    longCond  = (src_gt_f & src_gt_p & (upward > 0)) | (src_gt_f & src_lt_p & (upward > 0))
    shortCond = (src_lt_f & src_lt_p & (downward > 0)) | (src_lt_f & src_gt_p & (downward > 0))
    CondIni = pd.Series(0, index=s.index)
    for i in range(1,len(s)):
        if  bool(longCond.iloc[i]):  CondIni.iloc[i]=1
        elif bool(shortCond.iloc[i]): CondIni.iloc[i]=-1
        else:                         CondIni.iloc[i]=CondIni.iloc[i-1]
    longCondition  = longCond & (CondIni.shift(1) == -1)
    shortCondition = shortCond & (CondIni.shift(1) == 1)
    i = len(df)-1 if use_tv_bar else len(df)-2
    def last_at(series: pd.Series):
        v=series.iloc[i]; return None if pd.isna(v) else float(v)
    return {
        "bar_index": int(i),
        "price": last_at(df["close"].astype(float)),
        "filt":  last_at(filt),
        "hi":    last_at(hi),
        "lo":    last_at(lo),
        "fdir":  1.0 if (filt.iloc[i] > filt.iloc[i-1]) else (-1.0 if (filt.iloc[i] < filt.iloc[i-1]) else float(fdir.iloc[i-1] if i-1>=0 else 0.0)),
        "longCond": bool(longCond.iloc[i]), "shortCond": bool(shortCond.iloc[i]),
        "long": bool(longCondition.iloc[i]), "short": bool(shortCondition.iloc[i]),
        "atr": _atr_last(df)
    }

# =====================[ State / Pending / Guards ]=====================
state = {"open": False, "side": None, "entry": None, "qty": None, "pnl": 0.0}
compound_pnl = 0.0
pending = {"side": None, "first_seen": None, "bar_index": None}
cooldown_left = 0
cb_pause_left = 0
_daily_anchor_equity = None
_daily_halted = False
_last_day = None

def reset_daily_limits(balance):
    global _daily_anchor_equity, _daily_halted
    _daily_anchor_equity = (balance or 0.0); _daily_halted = False

def check_daily_loss_guard(balance):
    global _daily_halted
    if not USE_DAILY_LOSS_LIMIT or balance is None: return True
    if _daily_anchor_equity is None: return True
    drawdown = (_daily_anchor_equity - balance) / max(_daily_anchor_equity, 1e-9)
    if drawdown >= DAILY_LOSS_LIMIT_PCT: _daily_halted = True
    return not _daily_halted

def set_pending(side, bar_idx):
    pending.update({"side": side, "first_seen": time.time(), "bar_index": bar_idx})
def clear_pending():
    pending.update({"side": None, "first_seen": None, "bar_index": None})
def pending_ok(side, bar_idx):
    if pending["side"] != side: return False
    if pending["first_seen"] is None: return False
    if bar_idx < pending["bar_index"]: return False
    return (time.time() - pending["first_seen"]) >= SIGNAL_HOLD_S

# =====================[ Logs ]=====================
def snapshot(balance, rf, total_pnl):
    line("═","cyan")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    mode = "LIVE" if MODE_LIVE else "PAPER"
    print(colored(f"📊 RF BOT • {safe_symbol(SYMBOL)} • {INTERVAL} • {mode} • {now}", "cyan"))
    line("─","cyan")
    print(colorize_by_side("📈 INDICATORS"))
    print(f"   💲 Price     : {fmt(rf.get('price'))}")
    print(f"   📏 Filter    : {fmt(rf.get('filt'))}")
    print(f"   🔼 Band Hi   : {fmt(rf.get('hi'))}")
    print(f"   🔽 Band Lo   : {fmt(rf.get('lo'))}")
    dir_icon = "🟢 ↑ Up" if rf.get("fdir")==1 else ("🔴 ↓ Down" if rf.get("fdir")==-1 else "⚪ Flat")
    print(f"   🧭 Direction : {dir_icon}")
    print(f"   🟩 LongSig   : {rf.get('long')}")
    print(f"   🟥 ShortSig  : {rf.get('short')}")
    if USE_CIRCUIT_BREAKER: print(f"   🧯 ATR       : {fmt(rf.get('atr'))}  (CB x{CB_ATR_MULT})")
    print()
    print(colorize_by_side("🧭 POSITION"))
    bal_txt = "N/A (paper)" if not MODE_LIVE else fmt(balance,2)
    print(f"   💰 Balance   : {bal_txt} USDT")
    if state["open"]:
        side_icon = "🟩 LONG" if state['side']=="long" else "🟥 SHORT"
        print(f"   📌 Status    : {side_icon}")
        print(f"   🎯 Entry     : {fmt(state['entry'])}")
        print(f"   📦 Qty       : {fmt(state['qty'],4)}")
        print(f"   📊 PnL curr. : {fmt(state['pnl'])}")
    else:
        print("   📌 Status    : ⚪ FLAT")
    print()
    effective_equity = ((balance or 0.0) + total_pnl) if MODE_LIVE else total_pnl
    print("📦 RESULTS")
    print(f"   🧮 Compound PnL    : {fmt(total_pnl)}")
    print(f"   🚀 Effective Equity: {fmt(effective_equity)} USDT")
    if USE_DAILY_LOSS_LIMIT:
        anchor = fmt(_daily_anchor_equity,2,"N/A")
        print(f"   🛡️ Daily Anchor    : {anchor}   Halted: {False if not _daily_halted else True}")
    if cooldown_left>0: print(f"   ⏳ Cooldown Bars    : {cooldown_left}")
    if cb_pause_left>0: print(f"   🧯 CB Pause Bars    : {cb_pause_left}")
    line("─","cyan")

def log_reason(prefix, rf, note=""):
    msg = (f"{prefix} | bar={rf['bar_index']} | price={fmt(rf['price'])} "
           f"filt={fmt(rf['filt'])} hi={fmt(rf['hi'])} lo={fmt(rf['lo'])} "
           f"fdir={rf['fdir']} longCond={rf['longCond']} shortCond={rf['shortCond']} "
           f"long={rf['long']} short={rf['short']} {note}")
    print(colored(msg, "cyan"))

# =====================[ Open/Close (Signal-only) ]=====================
def open_market(side, qty, ref_price):
    global state
    if not MODE_LIVE:
        state.update({"open": True, "side":"long" if side=="buy" else "short",
                      "entry": ref_price, "qty": qty, "pnl": 0.0})
        print(colorize_by_side(f"✅ OPEN {side.upper()} [PAPER] qty={fmt(qty,4)} @ {fmt(ref_price)}"))
        return
    try: ex.set_leverage(LEVERAGE, safe_symbol(SYMBOL), params={"side":"BOTH"})
    except Exception as e: print(colored(f"⚠️ set_leverage: {e}", "yellow"))
    try:
        ex.create_order(safe_symbol(SYMBOL), "market", side, qty, params={"reduceOnly": False})
        time.sleep(0.8)
        entry = ref_price; size = qty
        poss = ex.fetch_positions([safe_symbol(SYMBOL)], params={"type":"swap"})
        for p in poss:
            if p.get("symbol")==safe_symbol(SYMBOL) and abs(float(p.get("contracts") or 0))>0:
                entry = float(p.get("entryPrice") or ref_price); size=abs(float(p.get("contracts") or qty))
                break
        state.update({"open": True, "side":"long" if side=='buy' else 'short',
                      "entry": entry, "qty": size, "pnl": 0.0})
        print(colorize_by_side(f"✅ OPEN {side.upper()} CONFIRMED qty={fmt(size,4)} @ {fmt(entry)}"))
    except Exception as e:
        print(colored(f"❌ open error: {e}", "red"))

def close_market(reason):
    global state, compound_pnl
    if not state["open"]: return
    px = price_now() or state["entry"]; qty = state["qty"]
    side = "sell" if state["side"]=="long" else "buy"
    if not MODE_LIVE:
        pnl = (px - state["entry"])*qty*(1 if state["side"]=="long" else -1)
    else:
        try: ex.create_order(safe_symbol(SYMBOL), "market", side, qty, params={"reduceOnly": True})
        except Exception as e: print(colored(f"❌ close error: {e}", "red"))
        pnl = (px - state["entry"])*qty*(1 if state["side"]=="long" else -1)
    compound_pnl += pnl
    print(colored(f"🔚 CLOSE {state['side']} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
    state.update({"open": False, "side": None, "entry": None, "qty": None, "pnl": 0.0})

# =====================[ Main Loop ]=====================
cooldown_left = 0
cb_pause_left = 0

def trade_loop():
    global state, compound_pnl, cooldown_left, cb_pause_left, _last_day
    while True:
        try:
            # Daily anchor
            today = date.today()
            if _last_day != today:
                _last_day = today
                reset_daily_limits(balance_usdt())

            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()
            if df is None or len(df) < 60:
                time.sleep(DECISION_EVERY_S); continue

            rf  = compute_rf(df, USE_TV_BAR, RF_SOURCE, RF_PERIOD, RF_MULT)

            # PnL لحظي
            if state["open"] and px:
                state["pnl"] = (px - state["entry"])*state["qty"] if state["side"]=="long" else (state["entry"] - px)*state["qty"]

            snapshot(bal, rf, compound_pnl)
            log_reason("WHY", rf, f"pending={pending}")

            # حمايات
            if USE_DAILY_LOSS_LIMIT and not check_daily_loss_guard(bal):
                print(colored("🛑 Daily loss limit hit — trading halted for today.", "red"))
                time.sleep(DECISION_EVERY_S); continue

            if USE_CIRCUIT_BREAKER and rf.get("atr") and len(df) >= 2:
                rng = abs(float(df["close"].iloc[-1]) - float(df["open"].iloc[-1]))
                if rf["atr"] and (rng >= CB_ATR_MULT * rf["atr"]):
                    cb_pause_left = CB_PAUSE_BARS
                    print(colored(f"🧯 CircuitBreaker: rng={fmt(rng)} >= {CB_ATR_MULT}×ATR({fmt(rf['atr'])}). Pausing.", "yellow"))
                    time.sleep(DECISION_EVERY_S); continue

            if cb_pause_left > 0:
                cb_pause_left -= 1
                print(colored(f"🧯 CB Pause ({cb_pause_left} bars left)", "yellow"))
                time.sleep(DECISION_EVERY_S); continue

            if cooldown_left > 0:
                cooldown_left -= 1
                print(colored(f"⏳ Cooldown ({cooldown_left} bars left)", "yellow"))
                time.sleep(DECISION_EVERY_S); continue

            # الإشارة الخام من Pine
            raw_side = "buy" if rf["long"] else ("sell" if rf["short"] else None)

            # Signal Hold
            side = None
            if raw_side is None:
                if pending["side"]: print(colored("hold_cancel: signal disappeared", "yellow"))
                clear_pending()
            else:
                if pending["side"] is None:
                    set_pending(raw_side, rf["bar_index"])
                    print(colored(f"hold_start: {raw_side} t+{SIGNAL_HOLD_S}s", "yellow"))
                elif pending_ok(raw_side, rf["bar_index"]):
                    side = raw_side
                    clear_pending()
                else:
                    wait_left = max(0, SIGNAL_HOLD_S - int(time.time() - pending["first_seen"]))
                    print(colored(f"holding: {pending['side']} wait={wait_left}s", "yellow"))
                    if (time.time() - pending["first_seen"]) > SIGNAL_MAX_AGE_S:
                        print(colored("hold_timeout: pending cleared (max age)", "yellow"))
                        clear_pending()

            # تنفيذ بعد ثبوت الإشارة
            if side:
                ref = rf.get("price") or px
                if not (ref and px): time.sleep(DECISION_EVERY_S); continue
                if abs(px - ref)/ref > MAX_SLIPPAGE_PCT:
                    print(colored(f"skip: slippage px={fmt(px)} ref={fmt(ref)}", "yellow"))
                    time.sleep(DECISION_EVERY_S); continue

                qty = compute_size(bal, px)
                desired = "long" if side=="buy" else "short"
                if state["open"]:
                    if state["side"] != desired:
                        close_market("opposite_signal_confirmed")
                        open_market(side, qty, px)
                        cooldown_left = COOLDOWN_BARS
                    else:
                        print(colored("ℹ️ already_in_position", "yellow"))
                else:
                    open_market(side, qty, px)
                    cooldown_left = COOLDOWN_BARS

        except Exception as e:
            print(colored(f"❌ loop error: {e}", "red"))

        time.sleep(DECISION_EVERY_S)

# =====================[ Keepalive / API ]=====================
def keepalive_loop():
    if not SELF_URL: return
    url = SELF_URL.rstrip("/")
    while True:
        try: requests.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

from flask import Flask
app = Flask(__name__)
@app.route("/")
def home(): return f"✅ RF Signal-Hold Bot — {safe_symbol(SYMBOL)} {INTERVAL} — {'LIVE' if MODE_LIVE else 'PAPER'}"
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
        "pending": pending,
        "cooldown_left": cooldown_left,
        "cb_pause_left": cb_pause_left,
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    })

# =====================[ Boot ]=====================
threading.Thread(target=trade_loop, daemon=True).start()
threading.Thread(target=keepalive_loop, daemon=True).start()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
