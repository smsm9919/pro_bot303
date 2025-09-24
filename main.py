# -*- coding: utf-8 -*-
"""
MIME BAY — BingX Futures LIVE (15m)
Robust Strategy (EMA5/15 + RSI + MACD + BB + EMA200)
+ False-Signal Guards (ADX/ATR-body/EMA confirm) + Reverse-on-Signal (2–3m confirm)
+ OCO Verification + Color-coded Logs by Position

مهم:
- لا نلمس دوال المنصّة/الرصيد/الحجم/Flask/KeepAlive.
- كل التعديلات داخل الاستراتيجية والحمايات والـ logs فقط.
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
RISK_ALLOC   = float(os.getenv("RISK_ALLOC", "0.60"))         # 60% من الرصيد
TRADE_MODE   = os.getenv("TRADE_MODE", "live").lower()        # live/paper
SELF_URL     = os.getenv("RENDER_EXTERNAL_URL", "") or os.getenv("SELF_URL", "")

# — thresholds —
RSI_LONG_TH      = float(os.getenv("RSI_LONG_TH", "35"))      # شراء تحت 35
RSI_SHORT_TH     = float(os.getenv("RSI_SHORT_TH", "65"))     # بيع فوق 65
MIN_ADX          = float(os.getenv("MIN_ADX", "18"))          # للعرض فقط
MAX_SLIPPAGE_PCT = float(os.getenv("MAX_SLIPPAGE_PCT", "0.004"))  # 0.4%
COOLDOWN_BARS    = int(os.getenv("COOLDOWN_BARS", "1"))

# — guards —
USE_REVERSE     = os.getenv("USE_REVERSE", "true").lower() == "true"
MIN_ADX_TRADE   = float(os.getenv("MIN_ADX_TRADE", "18"))     # حدّ أدنى للـ ADX
CONFIRM_PCT     = float(os.getenv("CONFIRM_PCT", "0.0015"))   # 0.15% فوق/تحت EMA slow
MIN_BODY_ATR    = float(os.getenv("MIN_BODY_ATR", "0.20"))    # جسم الشمعة ≥ 20% من ATR

# — توافق TradingView (اختياري) —
USE_TV_BAR      = os.getenv("USE_TV_BAR", "false").lower() == "true"  # true = استخدم الشمعة الحالية
RSI_LEN_MAIN    = int(os.getenv("RSI_LEN_MAIN", "7"))                 # 6/7

# — Reverse confirm —
REV_CONFIRM_SECONDS = int(os.getenv("REV_CONFIRM_SECONDS", "180"))    # 2–3 دقائق
REV_REQUIRE_CONSEC  = int(os.getenv("REV_REQUIRE_CONSEC",  "2"))      # قراءات متتالية أثناء الانتظار

# — Circuit breaker —
CIRCUIT_BREAKER_ATR_PCT = float(os.getenv("CIRCUIT_BREAKER_ATR_PCT", "3.0"))

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
SEP = colored("—"*96, "cyan")

def log(msg, color="white"): print(colored(msg, color), flush=True)

def log_trade(msg, base_color="white"):
    if state.get("open"):
        if state["side"] == "long":  return log(msg, "green")
        if state["side"] == "short": return log(msg, "red")
    return log(msg, base_color)

def fmt(v, d=2, na="N/A"):
    try:
        if v is None: return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

def safe_symbol(s: str):
    # FIX: كان فيها "::USDT"
    if s.endswith(":USDT") or s.endswith(":USDC"): return s
    if "/USDT" in s and not s.endswith(":USDT"):
        return s + ":USDT"
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
try: ex.load_markets()
except Exception: pass

# ======== Market / Account ========
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

# ======== Data / Wilder tools ========
def fetch_ohlcv(limit=400):
    rows = ex.fetch_ohlcv(safe_symbol(SYMBOL), timeframe=INTERVAL, limit=limit, params={"type":"swap"})
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

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

def ema(series: pd.Series, span: int): return series.ewm(span=span, adjust=False).mean()

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
    return ma, ma + mult*sd, ma - mult*sd

# ======== Indicators (Wilder دقيق) ========
def compute_indicators(df: pd.DataFrame):
    if df is None or len(df) < 210: return None
    o = df['open'].astype(float); h=df['high'].astype(float)
    l=df['low'].astype(float);  c = df['close'].astype(float)

    ema_fast = ema(c, 5); ema_slow = ema(c, 15); ema200 = ema(c, 200)

    # RSI (Wilder)
    change = c.diff(); gain=change.clip(lower=0); loss=(-change.clip(upper=0))
    avg_gain=rma(gain,RSI_LEN_MAIN); avg_loss=rma(loss,RSI_LEN_MAIN).replace(0,1e-12)
    rs=avg_gain/avg_loss; rsi = 100 - (100/(1+rs))

    # ATR (Wilder)
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = rma(tr, 14)

    # ADX (Wilder) — صحيح
    up = h.diff(); dn = -l.diff()
    plus_dm  = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    tr14  = rma(tr, 14)
    pdi14 = 100 * rma(plus_dm, 14) / tr14
    mdi14 = 100 * rma(minus_dm, 14) / tr14
    dx    = ( (pdi14 - mdi14).abs() / (pdi14 + mdi14).replace(0, 1e-12) ) * 100
    adx   = rma(dx, 14)

    macd_line, macd_signal, _ = macd_series(c, 12, 26, 9)
    bb_mid, bb_u, bb_l = bollinger(c, 20, 2.0)

    # الشمعة المستخدمة
    i = len(df)-1 if USE_TV_BAR else len(df)-2
    last = lambda s: float(s.iloc[i]) if not pd.isna(s.iloc[i]) else None
    prev = lambda s: float(s.iloc[i-1]) if not pd.isna(s.iloc[i-1]) else None

    return {
        "ts": int(df['time'].iloc[i]),
        "price": last(c),
        "open":  last(o),
        "body":  abs(last(c) - last(o)),
        "ema_fast": last(ema_fast), "ema_slow": last(ema_slow),
        "ema_fast_prev": prev(ema_fast), "ema_slow_prev": prev(ema_slow),
        "ema200": last(ema200),
        "rsi": last(rsi), "adx": last(adx), "atr": last(atr),
        "bb_mid": last(bb_mid), "bb_u": last(bb_u), "bb_l": last(bb_l),
        "macd": last(macd_line), "macd_signal": last(macd_signal),
    }

# ======== Strategy ========
def signal_core(ind):
    if not ind: return None, "indicators_not_ready"
    need = ["price","ema_fast","ema_slow","ema_fast_prev","ema_slow_prev","ema200","rsi","bb_l","bb_u","macd","macd_signal"]
    if any(ind.get(k) is None for k in need): return None, "missing_indicators"

    p = ind["price"]
    # BUY
    emaCrossBuy  = (ind["ema_fast_prev"] <= ind["ema_slow_prev"]) and (ind["ema_fast"] > ind["ema_slow"])
    cond_buy  = all([emaCrossBuy, ind["rsi"] < RSI_LONG_TH,  ind["macd"] > ind["macd_signal"], p < ind["bb_l"], p > ind["ema200"]])
    # SELL
    emaCrossSell = (ind["ema_fast_prev"] >= ind["ema_slow_prev"]) and (ind["ema_fast"] < ind["ema_slow"])
    cond_sell = all([emaCrossSell, ind["rsi"] > RSI_SHORT_TH, ind["macd"] < ind["macd_signal"], p > ind["bb_u"], p < ind["ema200"]])

    if cond_buy:  return "buy",  None
    if cond_sell: return "sell", None

    reasons=[]
    if not (emaCrossBuy or emaCrossSell): reasons.append("no_ema_cross")
    if RSI_LONG_TH <= ind["rsi"] <= RSI_SHORT_TH: reasons.append("rsi_neutral")
    if abs(ind["macd"]-ind["macd_signal"])<1e-12: reasons.append("macd_flat")
    if ind["bb_l"] < p < ind["bb_u"]: reasons.append("price_inside_bb")
    if abs(p - ind["ema200"])/p < 0.002: reasons.append("near_ema200")
    return None, "no_entry:" + (",".join(reasons) if reasons else "filters_not_met")

def signal_valid(ind, side):
    if not ind: return False, "indicators_na"
    reasons=[]
    if ind.get("adx") is not None and ind["adx"] < MIN_ADX_TRADE:
        reasons.append("weak_adx")
    if ind.get("atr") and ind.get("body") is not None and ind["body"] < MIN_BODY_ATR * ind["atr"]:
        reasons.append("small_body_vs_atr")
    if side == "buy":
        if not (ind.get("price") and ind.get("ema_slow")) or not (ind["price"] > ind["ema_slow"]*(1+CONFIRM_PCT)):
            reasons.append("no_confirm_above_ema_slow")
    else:  # sell
        if not (ind.get("price") and ind.get("ema_slow")) or not (ind["price"] < ind["ema_slow"]*(1-CONFIRM_PCT)):
            reasons.append("no_confirm_below_ema_slow")
    if reasons: return False, ",".join(reasons)
    return True, ""

def signal_balanced(ind):
    side, why = signal_core(ind)
    if side is None:
        log_trade(f"{IC_SHD} {why}", "yellow"); return None
    ok, why2 = signal_valid(ind, side)
    if not ok:
        log_trade(f"{IC_SHD} no_entry:{why2}", "yellow"); return None
    log_trade(f"{IC_TRD} {(IC_BUY if side=='buy' else IC_SELL)} signal ready", "green")
    return side

# ======== Reverse Manager (2–3m confirm) ========
reverse_pending = {"active": False, "to": None, "since": 0, "hits": 0}
def start_reverse(to_side: str):
    reverse_pending.update({"active": True, "to": to_side, "since": int(time.time()), "hits": 0})
    log_trade(f"🌀 reverse_started → {to_side}", "yellow")
def feed_reverse(current_signal: str) -> bool:
    if not reverse_pending["active"]: return False
    if current_signal == reverse_pending["to"]: reverse_pending["hits"] += 1
    else: reverse_pending["hits"] = 0
    enough_time = int(time.time()) - reverse_pending["since"] >= REV_CONFIRM_SECONDS
    return bool(enough_time and reverse_pending["hits"] >= REV_REQUIRE_CONSEC)
def end_reverse(msg="done"):
    if reverse_pending["active"]: log_trade(f"🌀 reverse_end: {msg}", "yellow")
    reverse_pending.update({"active": False, "to": None, "since": 0, "hits": 0})

# ======== State ========
state = {"open": False, "side": None, "entry": None, "qty": None, "tp": None, "sl": None, "pnl": 0.0}
compound_pnl = 0.0
cool_bars = 0

def snapshot(balance, price, ind, pos, total_pnl):
    print()
    log_trade(SEP, "cyan")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    # FIX: upper() (مش UPPER)
    log_trade(f"{IC_HDR} SNAPSHOT • {now} • mode={TRADE_MODE.upper()} • {safe_symbol(SYMBOL)} • {INTERVAL}", "cyan")
    log_trade("—", "cyan")
    log_trade(f"{IC_BAL} Balance (USDT): {fmt(balance,2)}", "yellow")
    log_trade(f"{IC_PRC} Price          : {fmt(price,6)}", "green")
    if ind:
        log_trade(f"{IC_EMA} EMA5/15/200   : {fmt(ind.get('ema_fast'),6)} / {fmt(ind.get('ema_slow'),6)} / {fmt(ind.get('ema200'),6)}", "blue")
        log_trade(f"{IC_RSI} RSI({RSI_LEN_MAIN})  : {fmt(ind.get('rsi'),2)}  (buy<{RSI_LONG_TH} / sell>{RSI_SHORT_TH})", "magenta")
        log_trade(f"{IC_ADX} ADX(14)       : {fmt(ind.get('adx'),2)}  (min_trade={fmt(MIN_ADX_TRADE,0)})", "magenta")
        log_trade(f"{IC_ATR} ATR(14)       : {fmt(ind.get('atr'),6)}  | body={fmt(ind.get('body'),6)}", "magenta")
    if pos["open"]:
        side = pos["side"].upper()
        log_trade(f"{IC_POS} Position      : {side} | entry={fmt(pos['entry'],6)} qty={fmt(pos['qty'],4)}")
        log_trade(f"{IC_TP}/{IC_SL} TP / SL       : {fmt(pos['tp'],6)} / {fmt(pos['sl'],6)}")
        log_trade(f"📈 PnL current  : {fmt(pos['pnl'],6)}")
    log_trade(f"📦 Compound PnL : {fmt(total_pnl,6)}", "yellow")
    log_trade(SEP, "cyan")

# ======== Sizing ========
def compute_size(balance, price):
    if not balance or not price: return 0
    raw = (balance * RISK_ALLOC * LEVERAGE) / price
    return market_amount(raw)

# ======== Protection (OCO) ========
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

def place_protected_order(side, qty, ref_price, atr=None):
    global state
    if not atr or atr <= 0:
        sl = ref_price * (0.985 if side=="buy" else 1.015)
        tp = ref_price * (1.015 if side=="buy" else 0.985)
    else:
        sl = ref_price - 1.2*atr if side=="buy" else ref_price + 1.2*atr
        tp = ref_price + 1.8*atr if side=="buy" else ref_price - 1.8*atr

    if TRADE_MODE == "paper":
        state.update({"open": True, "side": "long" if side=="buy" else "short",
                      "entry": ref_price, "qty": qty, "tp": tp, "sl": sl, "pnl": 0.0})
        log_trade(f"{IC_TRD} {(IC_BUY if side=='buy' else IC_SELL)} [PAPER] qty={fmt(qty,4)} entry={fmt(ref_price,6)} TP={fmt(tp,6)} SL={fmt(sl,6)}","green")
        return

    try:
        ex.set_leverage(LEVERAGE, safe_symbol(SYMBOL), params={"side":"BOTH"})
    except Exception as e:
        log_trade(f"{IC_BAD} set_leverage: {e}", "red")

    try:
        ex.create_order(safe_symbol(SYMBOL), "market", side, qty, params={"reduceOnly": False})
        log_trade(f"{IC_TRD} submit {'🟢 BUY' if side=='buy' else '🔴 SELL'} qty={fmt(qty,4)}", "green")
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
        log_trade(f"{IC_TRD} {(IC_BUY if side=='buy' else IC_SELL)} CONFIRMED qty={fmt(size,4)} entry={fmt(entry,6)} TP={fmt(tp,6)} SL={fmt(sl,6)}","green")

    except Exception as e:
        log_trade(f"{IC_BAD} post-trade validation error: {e}", "red")

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
            log_trade(f"{IC_BAD} close error: {e}", "red")
        pnl = (px - state["entry"]) * qty * (1 if state["side"]=="long" else -1)

    compound_pnl += pnl
    log_trade(f"{IC_CLS} Close {state['side']} reason={reason} pnl={fmt(pnl,6)} total={fmt(compound_pnl,6)}","magenta")
    state.update({"open": False, "side": None, "entry": None, "qty": None, "tp": None, "sl": None, "pnl": 0.0})
    cool_bars = COOLDOWN_BARS

# ======== Main loop ========
def trade_loop():
    global state, compound_pnl, cool_bars
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()
            ind = compute_indicators(df)

            # PnL الحالي
            if state["open"] and px:
                state["pnl"] = (px - state["entry"]) * state["qty"] if state["side"]=="long" else (state["entry"] - px) * state["qty"]

            snapshot(bal, px, ind or {}, state.copy(), compound_pnl)

            # Circuit breaker
            if ind and ind.get("atr") and ind.get("price"):
                atr_pct = (ind["atr"] / ind["price"]) * 100.0
                if atr_pct >= CIRCUIT_BREAKER_ATR_PCT:
                    log_trade("🛑 circuit_breaker: high volatility", "yellow")
                    time.sleep(60); continue

            if cool_bars > 0:
                cool_bars -= 1
                log_trade(f"{IC_SHD} cooldown bars left: {cool_bars}", "yellow")
                time.sleep(60); continue

            if not ind or not (px and bal):
                time.sleep(60); continue

            side = signal_balanced(ind)

            # Flip مؤكد: اغلاق ثم انتظار تأكيد 2–3 دق
            if USE_REVERSE and state["open"] and side:
                opposite = (state["side"]=="long" and side=="sell") or (state["side"]=="short" and side=="buy")
                if opposite:
                    close_position("signal_flip")
                    start_reverse(side)

            # تنفيذ الـ Reverse بعد التأكيد (مع slippage guard)
            if reverse_pending["active"]:
                if feed_reverse(side):
                    ref = ind.get("price") or px
                    if abs(px - ref)/ref <= MAX_SLIPPAGE_PCT:
                        qty = compute_size(bal, px)
                        place_protected_order(reverse_pending["to"], qty, px, atr=ind.get("atr"))
                        end_reverse("opened")
                        cool_bars = COOLDOWN_BARS
                        time.sleep(60); continue
                    else:
                        log_trade(f"{IC_SHD} reverse blocked: slippage", "yellow")

            # دخول عادي
            if not state["open"] and side:
                ref = ind.get("price")
                if ref and abs(px - ref)/ref <= MAX_SLIPPAGE_PCT:
                    qty = compute_size(bal, px)
                    if qty and qty > 0:
                        log_trade(f"{IC_TRD} {(IC_BUY if side=='buy' else IC_SELL)} qty={fmt(qty,4)} entry={fmt(px,6)}", "green")
                        place_protected_order(side, qty, px, atr=ind.get("atr"))
                        cool_bars = COOLDOWN_BARS
                else:
                    log_trade(f"{IC_SHD} no_entry: slippage px={fmt(px,6)} ref={fmt(ref,6)}","yellow")

            # Soft-guard TP/SL
            if state["open"] and px:
                if state["side"]=="long" and (px <= state["sl"] or px >= state["tp"]):
                    close_position("tp" if px >= state["tp"] else "sl")
                elif state["side"]=="short" and (px >= state["sl"] or px <= state["tp"]):
                    close_position("tp" if px <= state["tp"] else "sl")

        except Exception as e:
            log_trade(f"{IC_BAD} loop error: {e}", "red")

        time.sleep(60)  # 15m — قرار كل دقيقة

# ======== Keepalive / Flask ========
def keepalive_loop():
    if not SELF_URL:
        log_trade("SELF_URL not set — keepalive disabled", "yellow"); return
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
