# -*- coding: utf-8 -*-
"""
BingX Futures LIVE (15m) — Strategy (EMA5/15 + RSI + MACD + BB + Trend EMA200) + OCO Verification + Explanatory Logs
مراعاة: لا تغيير في الدوال الأساسية (الرصيد/الحجم/الأوامر/Flask/KeepAlive). كل التعديلات داخل الاستراتيجية والحماية فقط.
"""

import os, time, math, threading, requests
import pandas as pd
import ccxt
from flask import Flask, jsonify
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# ======== Config ========
SYMBOL       = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL     = os.getenv("INTERVAL", "15m")
LEVERAGE     = int(float(os.getenv("LEVERAGE", "10")))
RISK_ALLOC   = float(os.getenv("RISK_ALLOC", "0.60"))   # 60% of balance (لا ألمسه)
TRADE_MODE   = os.getenv("TRADE_MODE", "live").lower()  # live/paper
SELF_URL     = os.getenv("RENDER_EXTERNAL_URL", "") or os.getenv("SELF_URL", "")

# Strategy thresholds (تبقى من .env لو حبيت تغيّرها لاحقًا)
RSI_LONG_TH      = float(os.getenv("RSI_LONG_TH", "35"))  # مطابق للـ RSI Oversold في الصورة
RSI_SHORT_TH     = float(os.getenv("RSI_SHORT_TH", "65")) # مطابق للـ RSI Overbought
MIN_ADX          = float(os.getenv("MIN_ADX", "18"))      # نستخدمه كفلتر ثانوي اختياري
MAX_SLIPPAGE_PCT = float(os.getenv("MAX_SLIPPAGE_PCT", "0.004"))  # 0.4%
COOLDOWN_BARS    = int(os.getenv("COOLDOWN_BARS", "1"))

API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")

# ======== Icons / colors ========
try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

IC_HDR="📊"; IC_BAL="💰"; IC_PRC="💲"; IC_EMA="📈"; IC_RSI="📉"; IC_ADX="📐"; IC_ATR="📏"
IC_POS="🧭"; IC_TP="🎯"; IC_SL="🛑"; IC_TRD="🟢"; IC_CLS="🔴"; IC_OK="✅"; IC_BAD="❌"; IC_SHD="🛡️"
IC_BUY="🟩 BUY"; IC_SELL="🟥 SELL"
SEP = colored("—"*88, "cyan")

def log(msg, color="white"): print(colored(msg, color), flush=True)
def fmt(v, d=2, na="N/A"):
    try:
        if v is None: return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

def safe_symbol(s: str) -> str:
    if s.endswith(":USDT") or s.endswith(":USDC"): return s
    if "/USDT" in s and not s.endswith(":USDT"): return s + ":USDT"
    return s

# ======== Exchange ========
def make_exchange():
    return ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap", "defaultMarginMode": "isolated"}
    })
ex = make_exchange()

# ======== Market / Account (كما هي) ========
def balance_usdt():
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

def market_amount(amount):
    try:
        m = ex.market(safe_symbol(SYMBOL))
        prec = int(m.get("precision",{}).get("amount", 3))
        min_amt = m.get("limits",{}).get("amount",{}).get("min", 0.001)
        amt = float(f"{float(amount):.{prec}f}")
        return max(amt, float(min_amt or 0.001))
    except Exception:
        return float(amount)

# ======== OHLCV + Wilder tools (كما هي) ========
def fetch_ohlcv(limit=300):
    rows = ex.fetch_ohlcv(safe_symbol(SYMBOL), timeframe=INTERVAL, limit=limit, params={"type":"swap"})
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def rma(series: pd.Series, length: int):
    alpha = 1.0 / float(length)
    r = [None]*len(series)
    s = series.astype(float).values
    acc = 0.0; n = 0
    for i, v in enumerate(s):
        if pd.isna(v): r[i] = math.nan; continue
        if n < length:
            acc += v; n += 1
            r[i] = math.nan if n < length else acc/length
        else:
            r[i] = (r[i-1]*(length-1) + v) / length
    return pd.Series(r, index=series.index, dtype="float64")

def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

# ======== Extra indicators (EMA200 + MACD + BB) ========
def macd_series(c, fast=12, slow=26, signal=9):
    ema_fast = c.ewm(span=fast, adjust=False).mean()
    ema_slow = c.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal
    return macd, macd_signal, hist

def bollinger(c, length=20, mult=2.0):
    ma = c.rolling(length).mean()
    sd = c.rolling(length).std(ddof=0)
    upper = ma + mult*sd
    lower = ma - mult*sd
    return ma, upper, lower

# ======== مؤشرات الشمعة المُغلقة (اعتمادًا على الصورة) ========
def compute_indicators_closed(df: pd.DataFrame):
    if df is None or len(df) < 210: return None
    c = df['close'].astype(float); h=df['high'].astype(float); l=df['low'].astype(float)

    ema_fast = ema(c, 5)      # EMA السريع
    ema_slow = ema(c, 15)     # EMA البطيء
    ema200   = ema(c, 200)    # اتجاه

    # RSI(7) كما في استايل الصورة
    change = c.diff(); gain=change.clip(lower=0); loss=(-change.clip(upper=0))
    avg_gain=rma(gain,7); avg_loss=rma(loss,7).replace(0,1e-12); rs=avg_gain/avg_loss
    rsi = 100 - (100/(1+rs))

    # ATR/ADX الأساسيين (للـlogs والحماية)
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr = rma(tr,14)
    up=h.diff(); dn=-l.diff()
    plus_dm=((up>dn)&(up>0))*up; minus_dm=((dn>up)&(dn>0))*dn
    tr14=rma(tr,14); pdi14=100*rma(plus_dm,14)/tr14; mdi14=100*rma(minus_dm,14)/tr14
    dx=( (pdi14-mdi14).abs()/(pdi14+mdi14).replace(0,1e-12) )*100; adx=rma(dx,14)

    macd_line, macd_signal, _ = macd_series(c, 12, 26, 9)
    bb_mid, bb_u, bb_l = bollinger(c, 20, 2.0)

    i = len(df)-2  # نشتغل على آخر شمعة مغلقة
    last = lambda s: float(s.iloc[i]) if not pd.isna(s.iloc[i]) else None
    prev = lambda s: float(s.iloc[i-1]) if not pd.isna(s.iloc[i-1]) else None

    return {
        "ts": int(df['time'].iloc[i]),
        "price": last(c),
        "ema_fast": last(ema_fast), "ema_slow": last(ema_slow),
        "ema_fast_prev": prev(ema_fast), "ema_slow_prev": prev(ema_slow),
        "ema200": last(ema200),
        "rsi": last(rsi), "adx": last(adx), "atr": last(atr),
        "bb_mid": last(bb_mid), "bb_u": last(bb_u), "bb_l": last(bb_l),
        "macd": last(macd_line), "macd_signal": last(macd_signal),
    }

# ======== إشارات BUY/SELL + أسباب الرفض (الاستراتيجية الفعلية) ========
def signal_with_reason(ind):
    if not ind or any(ind.get(k) is None for k in ["price","ema_fast","ema_slow","ema_fast_prev","ema_slow_prev"]):
        return None, "indicators_not_ready"

    p = ind["price"]
    # شروط شراء (زي الصورة): قطع EMA5 فوق EMA15 + RSI<35 (oversold) + MACD>Signal + السعر تحت BB-L + أعلى من EMA200
    emaCrossBuy = (ind["ema_fast_prev"] <= ind["ema_slow_prev"]) and (ind["ema_fast"] > ind["ema_slow"])
    rsiBuy      = ind.get("rsi") is not None and ind["rsi"] < RSI_LONG_TH
    macdBuy     = (ind.get("macd") is not None and ind.get("macd_signal") is not None and ind["macd"] > ind["macd_signal"])
    bbBuy       = (ind.get("bb_l") is not None and p < ind["bb_l"])
    trendBuy    = (ind.get("ema200") is not None and p > ind["ema200"])
    longCondition = emaCrossBuy and rsiBuy and macdBuy and bbBuy and trendBuy

    # شروط بيع: قطع EMA5 تحت EMA15 + RSI>65 + MACD<Signal + السعر فوق BB-U + تحت EMA200
    emaCrossSell = (ind["ema_fast_prev"] >= ind["ema_slow_prev"]) and (ind["ema_fast"] < ind["ema_slow"])
    rsiSell      = ind.get("rsi") is not None and ind["rsi"] > RSI_SHORT_TH
    macdSell     = (ind.get("macd") is not None and ind.get("macd_signal") is not None and ind["macd"] < ind["macd_signal"])
    bbSell       = (ind.get("bb_u") is not None and p > ind["bb_u"])
    trendSell    = (ind.get("ema200") is not None and p < ind["ema200"])
    shortCondition = emaCrossSell and rsiSell and macdSell and bbSell and trendSell

    if longCondition:  return "buy",  None
    if shortCondition: return "sell", None

    # أسباب عدم الدخول (للوغز)
    reasons = []
    if not (emaCrossBuy or emaCrossSell): reasons.append("no_ema_cross")
    if ind.get("rsi") is not None and RSI_LONG_TH <= ind["rsi"] <= RSI_SHORT_TH: reasons.append("rsi_neutral")
    if ind.get("macd") is not None and ind.get("macd_signal") is not None and abs(ind["macd"]-ind["macd_signal"])<1e-12: reasons.append("macd_flat")
    if ind.get("bb_l") is not None and ind.get("bb_u") is not None and (ind["bb_l"] < p < ind["bb_u"]): reasons.append("price_inside_bb")
    if ind.get("ema200") is not None and abs(p - ind["ema200"])/p < 0.002: reasons.append("near_trend")
    return None, ("no_entry: " + (",".join(reasons) if reasons else "filters_not_met"))

def signal_balanced(ind):
    side, reason = signal_with_reason(ind)
    if side is None and reason:
        log(f"{IC_SHD} {reason}", "yellow")
    else:
        tag = IC_BUY if side=="buy" else IC_SELL
        log(f"{IC_TRD} {tag} signal ready", "green")
    return side

# ======== حالة وإظهار ========
state = {"open": False, "side": None, "entry": None, "qty": None, "tp": None, "sl": None, "pnl": 0.0}
compound_pnl = 0.0
cool_bars = 0
latest_bar_ts = None
latest_indicators = {}

def snapshot(balance, price, ind, pos, total_pnl):
    print()
    log(SEP, "cyan")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    log(f"{IC_HDR} SNAPSHOT • {now} • mode={TRADE_MODE.upper()} • {safe_symbol(SYMBOL)} • {INTERVAL}", "cyan")
    log("—", "cyan")
    log(f"{IC_BAL} Balance (USDT): {fmt(balance,2)}", "yellow")
    log(f"{IC_PRC} Price          : {fmt(price,6)}", "green")
    if ind:
        log(f"{IC_EMA} EMA5/15/200   : {fmt(ind.get('ema_fast'),6)} / {fmt(ind.get('ema_slow'),6)} / {fmt(ind.get('ema200'),6)}", "blue")
        log(f"{IC_RSI} RSI(7)        : {fmt(ind.get('rsi'),2)}  (buy<{RSI_LONG_TH} / sell>{RSI_SHORT_TH})", "magenta")
        log(f"{IC_ADX} ADX(14)       : {fmt(ind.get('adx'),2)}", "magenta")
        log(f"{IC_ATR} ATR(14)       : {fmt(ind.get('atr'),6)}", "magenta")
    if pos["open"]:
        side = pos["side"].upper()
        log(f"{IC_POS} Position      : {side} | entry={fmt(pos['entry'],6)} qty={fmt(pos['qty'],4)}", "white")
        log(f"{IC_TP}/{IC_SL} TP / SL       : {fmt(pos['tp'],6)} / {fmt(pos['sl'],6)}", "white")
        log(f"📈 PnL current  : {fmt(pos['pnl'],6)}", "white")
    log(f"📦 Compound PnL : {fmt(total_pnl,6)}", "yellow")
    log(SEP, "cyan")

# ======== Sizing / Helpers (كما هي) ========
def compute_size(balance, price):
    if not balance or not price: return 0
    raw = (balance * RISK_ALLOC * LEVERAGE) / price
    return market_amount(raw)

# ======== حماية: استخدام دوالك + Verify OCO ========
def attach_protection(side_opp, qty, tp, sl):
    """هذه الدالة كما في بوتك السابق. نتركها كما هي ونضيف Verify خارجي."""
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
            log(f"{IC_SHD} protection attach failed: {e2}", "yellow")
            return False

def verify_oco(side_opp):
    """تحقق أن SL وTP اتعلقوا بالفعل — لو فشل: هنقفل المركز فورًا لحماية الرصيد."""
    try:
        oo = ex.fetch_open_orders(symbol=safe_symbol(SYMBOL))
        tp_ok = any(("take" in (o.get("type","").lower())) and o.get("side","").lower()==side_opp for o in oo)
        sl_ok = any(("stop" in (o.get("type","").lower())) and o.get("side","").lower()==side_opp for o in oo)
        return tp_ok and sl_ok
    except Exception as e:
        log(f"{IC_SHD} verify_oco error: {e}", "yellow"); return False

# ======== Open / Close (لا تغيير في طريقة الحساب والدخول) ========
def place_protected_order(side, qty, ref_price, atr=None):
    global state
    # SL/TP بالـ ATR (كما في بوتك)
    if not atr or atr <= 0:
        sl = ref_price * (0.985 if side=="buy" else 1.015)
        tp = ref_price * (1.015 if side=="buy" else 0.985)
    else:
        sl = ref_price - 1.2*atr if side=="buy" else ref_price + 1.2*atr
        tp = ref_price + 1.8*atr if side=="buy" else ref_price - 1.8*atr

    if TRADE_MODE == "paper":
        state.update({"open": True, "side": "long" if side=="buy" else "short",
                      "entry": ref_price, "qty": qty, "tp": tp, "sl": sl, "pnl": 0.0})
        tag = IC_BUY if side=="buy" else IC_SELL
        log(f"{IC_TRD} {tag} [PAPER] qty={fmt(qty,4)} entry={fmt(ref_price,6)} TP={fmt(tp,6)} SL={fmt(sl,6)}","green")
        return

    try:
        ex.set_leverage(LEVERAGE, safe_symbol(SYMBOL), params={"side":"BOTH"})
    except Exception as e:
        log(f"{IC_BAD} set_leverage: {e}", "red")

    try:
        ex.create_order(safe_symbol(SYMBOL), "market", side, qty, params={"reduceOnly": False})
        tag = "🟢 BUY" if side=="buy" else "🔴 SELL"
        log(f"{IC_TRD} submit {tag} qty={fmt(qty,4)}", "green")
    except Exception as e:
        log(f"{IC_BAD} open error: {e}", "red"); return

    # تحقق بعد التنفيذ + إرفاق حماية + Verify OCO
    try:
        time.sleep(0.9)
        poss = ex.fetch_positions([safe_symbol(SYMBOL)], params={"type":"swap"})
        pos = None
        for p in poss:
            if (p.get("symbol") == safe_symbol(SYMBOL)) and abs(float(p.get("contracts") or 0)) > 0:
                pos = p; break
        if not pos:
            log(f"{IC_SHD} no real position detected; state not updated.", "yellow"); return

        entry = float(pos.get("entryPrice") or ref_price)
        size  = abs(float(pos.get("contracts") or qty))
        side_opp = "sell" if side=="buy" else "buy"

        attached = attach_protection(side_opp, size, tp, sl)
        if not attached or not verify_oco(side_opp):
            try: ex.create_order(safe_symbol(SYMBOL), "market", side_opp, size, params={"reduceOnly": True})
            except Exception as e3: log(f"{IC_BAD} emergency close: {e3}", "red")
            log(f"{IC_SHD} aborted entry since SL/TP not attached/verified.", "yellow")
            return

        state.update({"open": True, "side": "long" if side=="buy" else "short",
                      "entry": entry, "qty": size, "tp": tp, "sl": sl, "pnl": 0.0})
        tag2 = IC_BUY if side=="buy" else IC_SELL
        log(f"{IC_TRD} {tag2} CONFIRMED qty={fmt(size,4)} entry={fmt(entry,6)} TP={fmt(tp,6)} SL={fmt(sl,6)}","green")

    except Exception as e:
        log(f"{IC_BAD} post-trade validation error: {e}", "red")

def close_position(reason):
    global state, compound_pnl, cool_bars
    if not state["open"]: return
    px   = price_now() or state["entry"]
    qty  = state["qty"]
    side = "sell" if state["side"]=="long" else "buy"

    if TRADE_MODE == "paper":
        pnl = (px - state["entry"]) * qty * (1 if state["side"]=="long" else -1)
    else:
        try:
            ex.create_order(safe_symbol(SYMBOL), "market", side, qty, params={"reduceOnly": True})
        except Exception as e:
            log(f"{IC_BAD} close error: {e}", "red")
        pnl = (px - state["entry"]) * qty * (1 if state["side"]=="long" else -1)

    compound_pnl += pnl
    log(f"{IC_CLS} Close {state['side']} reason={reason} pnl={fmt(pnl,6)} total={fmt(compound_pnl,6)}","magenta")
    state.update({"open": False, "side": None, "entry": None, "qty": None, "tp": None, "sl": None, "pnl": 0.0})
    cool_bars = COOLDOWN_BARS

# ======== Main loop (كما هو مع لوج أسباب الرفض) ========
def trade_loop():
    global state, compound_pnl, cool_bars, latest_bar_ts, latest_indicators
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()
            # مؤشرات الشمعة المغلقة لإشارات مؤكدة
            ind = compute_indicators_closed(df) if df is not None and len(df) > 220 else None
            latest_indicators = ind or {}

            # تحديث PnL الجاري
            if state["open"] and px:
                state["pnl"] = (px - state["entry"]) * state["qty"] if state["side"]=="long" else (state["entry"] - px) * state["qty"]

            snapshot(bal, px, ind or {}, state.copy(), compound_pnl)

            if cool_bars > 0:
                cool_bars -= 1
                log(f"{IC_SHD} cooldown bars left: {cool_bars}", "yellow")
                time.sleep(60); continue

            # دخول محمي
            if not state["open"] and px and bal and ind:
                ref = ind.get("price")
                side = signal_balanced(ind)
                if side and ref and abs(px - ref)/ref <= MAX_SLIPPAGE_PCT:
                    qty = compute_size(bal, px)  # نفس معادلة 60% × 10x
                    if qty and qty > 0:
                        log(f"{IC_TRD} {(IC_BUY if side=='buy' else IC_SELL)} qty={fmt(qty,4)} entry={fmt(px,6)}", "green")
                        place_protected_order(side, qty, px, atr=ind.get("atr"))
                        cool_bars = COOLDOWN_BARS
                elif side and ref:
                    log(f"{IC_SHD} no_entry: slippage px={fmt(px,6)} ref={fmt(ref,6)}","yellow")

            # Soft-guard لمس TP/SL
            if state["open"] and px:
                if state["side"]=="long" and (px <= state["sl"] or px >= state["tp"]):
                    close_position("tp" if px >= state["tp"] else "sl")
                elif state["side"]=="short" and (px >= state["sl"] or px <= state["tp"]):
                    close_position("tp" if px <= state["tp"] else "sl")

        except Exception as e:
            log(f"{IC_BAD} loop error: {e}", "red")

        time.sleep(60)  # 1m tick for 15m strategy (قرارات على إغلاق 15m)

# ======== Keepalive / Flask (كما هي) ========
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
def home(): return f"{IC_OK} Bot Running — {safe_symbol(SYMBOL)} {INTERVAL}"
@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": safe_symbol(SYMBOL),
        "interval": INTERVAL,
        "mode": TRADE_MODE,
        "leverage": LEVERAGE,
        "risk_alloc": RISK_ALLOC,
        "balance": balance_usdt(),
        "price": price_now(),
        "position": state,
        "compound_pnl": compound_pnl,
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    })

# ======== Boot ========
threading.Thread(target=trade_loop, daemon=True).start()
threading.Thread(target=keepalive_loop, daemon=True).start()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
