# -*- coding: utf-8 -*-
"""
BingX Futures LIVE (15m) — Multi-Engine Signals + OCO Protection + Trailing + Explanatory Logs
"""

# ======================= Imports / Setup =======================
import os, time, math, threading, requests
import pandas as pd
import ccxt
from flask import Flask, jsonify
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# ——— Icons / Colors ———
try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

IC_HDR="📊"; IC_BAL="💰"; IC_PRC="💲"; IC_EMA="📈"; IC_RSI="📉"; IC_ADX="📐"; IC_ATR="📏"
IC_POS="🧭"; IC_TP="🎯"; IC_SL="🛑"; IC_TRD="🟢"; IC_CLS="🔴"; IC_OK="✅"; IC_BAD="❌"; IC_SHD="🛡️"
IC_BUY="🟩 BUY"; IC_SELL="🟥 SELL"
SEP = colored("—"*92, "cyan")

def log(msg, color="white"):
    print(colored(msg, color), flush=True)

def fmt(v, d=2, na="N/A"):
    try:
        if v is None: return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

# ======================= Config =======================
SYMBOL       = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL     = os.getenv("INTERVAL", "15m")
LEVERAGE     = int(float(os.getenv("LEVERAGE", "10")))
RISK_ALLOC   = float(os.getenv("RISK_ALLOC", "0.60"))   # احتياطي (لن نستخدمه للحجم؛ نستخدم RISK_PCT)
TRADE_MODE   = os.getenv("TRADE_MODE", "live").lower()  # live/paper
SELF_URL     = os.getenv("RENDER_EXTERNAL_URL", "") or os.getenv("SELF_URL", "")

# Strategy thresholds (قابلة للتعديل من ENV)
RSI_LONG_TH      = float(os.getenv("RSI_LONG_TH", "55"))
RSI_SHORT_TH     = float(os.getenv("RSI_SHORT_TH", "45"))
MIN_ADX          = float(os.getenv("MIN_ADX", "18"))    # يسمح بإشارات MR/PB
MAX_SLIPPAGE_PCT = float(os.getenv("MAX_SLIPPAGE_PCT", "0.004"))  # 0.4%
COOLDOWN_BARS    = int(os.getenv("COOLDOWN_BARS", "2")) # كوولداون بالشموع
RISK_PCT         = float(os.getenv("RISK_PCT", "0.01")) # 1% مخاطرة/صفقة

# قواطع السوق
ADX_TREND = max(MIN_ADX, 22)
ADX_RANGE = 18
ATR_BREAKER_PCT = 3.0
SPIKE_MULT = 2.0

API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")

def safe_symbol(s: str) -> str:
    if s.endswith(":USDT") or s.endswith(":USDC"): return s
    if "/USDT" in s and not s.endswith(":USDT"): return s + ":USDT"
    return s

# ======================= Exchange =======================
def make_exchange():
    return ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap", "defaultMarginMode": "isolated"}
    })

ex = make_exchange()

# ======================= Market / Account =======================
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

# ======================= Data / Indicators =======================
def fetch_ohlcv(limit=400):
    rows = ex.fetch_ohlcv(safe_symbol(SYMBOL), timeframe=INTERVAL, limit=limit, params={"type":"swap"})
    df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
    return df

def rma(series: pd.Series, length: int):
    """Wilder RMA"""
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

def bollinger(c, length=20, mult=2.0):
    ma = c.rolling(length).mean()
    sd = c.rolling(length).std(ddof=0)
    upper = ma + mult*sd
    lower = ma - mult*sd
    return ma, upper, lower

def supertrend_series(df, period=10, mult=3.0):
    h, l, c = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
    hl2 = (h + l) / 2.0
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr_ = rma(tr, period)
    upper = hl2 + mult*atr_
    lower = hl2 - mult*atr_
    st = pd.Series(index=df.index, dtype=float)
    d  = pd.Series(index=df.index, dtype=int)
    st.iloc[0] = upper.iloc[0]; d.iloc[0] = 1
    for i in range(1, len(df)):
        prev = st.iloc[i-1]
        if c.iloc[i-1] > prev:
            cand = min(upper.iloc[i], prev); dir_=1
        else:
            cand = max(lower.iloc[i], prev); dir_=-1
        if dir_==1 and c.iloc[i] < cand:
            dir_=-1; st.iloc[i]=upper.iloc[i]
        elif dir_==-1 and c.iloc[i] > cand:
            dir_=1; st.iloc[i]=lower.iloc[i]
        else:
            st.iloc[i]=cand
        d.iloc[i]=dir_
    return st, d

def compute_indicators_closed(df: pd.DataFrame):
    """نستخدم آخر شمعة مُغلقة لتفادي إشارات كاذبة"""
    if df is None or len(df) < 210: return None
    c = df['close'].astype(float); h=df['high'].astype(float); l=df['low'].astype(float)

    ema20  = ema(c,20); ema50=ema(c,50); ema200=ema(c,200)
    change = c.diff(); gain=change.clip(lower=0); loss=(-change.clip(upper=0))
    avg_gain=rma(gain,14); avg_loss=rma(loss,14).replace(0,1e-12); rs=avg_gain/avg_loss
    rsi = 100 - (100/(1+rs))
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr = rma(tr,14)
    up=h.diff(); dn=-l.diff()
    plus_dm=((up>dn)&(up>0))*up; minus_dm=((dn>up)&(dn>0))*dn
    tr14=rma(tr,14); pdi14=100*rma(plus_dm,14)/tr14; mdi14=100*rma(minus_dm,14)/tr14
    dx=( (pdi14-mdi14).abs()/(pdi14+mdi14).replace(0,1e-12) )*100; adx=rma(dx,14)
    bb_mid, bb_u, bb_l = bollinger(c,20,2.0)
    st_line, st_dir = supertrend_series(df,10,3.0)

    i = len(df)-2  # آخر شمعة مغلقة
    last = lambda s: float(s.iloc[i]) if not pd.isna(s.iloc[i]) else None
    prev = lambda s: float(s.iloc[i-1]) if not pd.isna(s.iloc[i-1]) else None

    return {
        "ts": int(df['time'].iloc[i]),
        "price": last(c),
        "ema20": last(ema20), "ema50": last(ema50), "ema200": last(ema200),
        "ema20_prev": prev(ema20), "ema50_prev": prev(ema50),
        "rsi": last(rsi), "adx": last(adx), "atr": last(atr),
        "atr_pct": (last(atr)/last(c))*100 if last(atr) and last(c) else None,
        "bb_mid": last(bb_mid), "bb_u": last(bb_u), "bb_l": last(bb_l),
        "st_dir": int(st_dir.iloc[i]) if not pd.isna(st_dir.iloc[i]) else None,
        "st_line": last(st_line),
        "bar_range": float((h.iloc[i]-l.iloc[i])),
    }

# ======================= Signals (TC + PB + MR) =======================
def signal_with_reason(ind):
    """
    يرجع: (side, engine_tag, reason_if_none)
    side: "buy"/"sell"/None
    """
    if not ind: return None, None, "indicators_not_ready"
    p=ind["price"]; e20=ind["ema20"]; e50=ind["ema50"]; e200=ind["ema200"]
    e20p=ind["ema20_prev"]; e50p=ind["ema50_prev"]; rsi=ind["rsi"]; adx=ind["adx"]

    # قواطع
    if ind.get("atr_pct") and ind["atr_pct"] >= ATR_BREAKER_PCT:
        return None, None, f"atr_breaker_{fmt(ind['atr_pct'],2)}%"
    if ind.get("atr") and ind.get("bar_range") and ind["bar_range"] > SPIKE_MULT*ind["atr"]:
        return None, None, "spike_bar_gt_2xATR"
    if adx is None: return None, None, "adx_na"

    trending = adx >= ADX_TREND
    ranging  = adx < ADX_RANGE

    # Trend Continuation (قطع/بولباك + Supertrend + RSI)
    if trending and p and e20 and e50 and e200:
        long_ok  = (p>e200) and (e20>e50) and (ind["st_dir"]==1) and (rsi is not None and rsi>RSI_LONG_TH)
        short_ok = (p<e200) and (e20<e50) and (ind["st_dir"]==-1) and (rsi is not None and rsi<RSI_SHORT_TH)
        cross_up   = (e20p<=e50p) and (e20>e50)
        cross_down = (e20p>=e50p) and (e20<e50)
        pb_long  = p>e20 and ind["bb_mid"] and p>ind["bb_mid"] and not cross_down
        pb_short = p<e20 and ind["bb_mid"] and p<ind["bb_mid"] and not cross_up
        if long_ok and (cross_up or pb_long):  return "buy","TC",None
        if short_ok and (cross_down or pb_short): return "sell","TC",None
        return None, None, "tc_filters_not_met"

    # Pullback Scalps (ADX بين 18 و 22)
    if (ADX_RANGE <= adx < ADX_TREND) and p and e20 and e50:
        if p>e50 and rsi and rsi>52 and p>e20:  return "buy","PB",None
        if p<e50 and rsi and rsi<48 and p<e20:  return "sell","PB",None
        return None, None, "pb_filters_not_met"

    # Mean Reversion (رينج صريح)
    if ranging and ind["bb_l"] and ind["bb_u"] and rsi is not None:
        if p<=ind["bb_l"] and rsi<35: return "buy","MR",None
        if p>=ind["bb_u"] and rsi>65: return "sell","MR",None
        return None, None, "mr_filters_not_met"

    return None, None, "market_state_undefined"

# ======================= Risk Sizing / Helpers =======================
def compute_size_risk(balance, entry, sl):
    if not balance or not entry or not sl: return 0.0
    risk_usdt = balance * RISK_PCT
    stop_dist = abs(entry - sl)
    if stop_dist <= 0: return 0.0
    qty = (risk_usdt / stop_dist) * LEVERAGE / entry
    return market_amount(qty)

def attach_protection(side_opp, qty, tp, sl):
    """ضع TP/SL كـ reduceOnly. حاول أكثر من طريقة."""
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
    try:
        oo = ex.fetch_open_orders(symbol=safe_symbol(SYMBOL))
        tp_ok = any("take" in (o.get("type","").lower()) and o.get("side","").lower()==side_opp for o in oo)
        sl_ok = any("stop" in (o.get("type","").lower()) and o.get("side","").lower()==side_opp for o in oo)
        return tp_ok and sl_ok
    except Exception as e:
        log(f"{IC_SHD} verify_oco error: {e}", "yellow"); return False

def cancel_all_open_orders():
    try:
        for o in ex.fetch_open_orders(symbol=safe_symbol(SYMBOL)):
            try: ex.cancel_order(o['id'], symbol=safe_symbol(SYMBOL))
            except Exception: pass
    except Exception: pass

# ======================= State / Snapshot =======================
state = {"open": False, "side": None, "entry": None, "qty": None, "tp": None, "sl": None, "pnl": 0.0}
compound_pnl = 0.0
cool_bars = 0
latest_bar_ts = None
latest_indicators = {}
block_dir = None
block_bars_left = 0

def snapshot(balance, price, ind, pos, total_pnl):
    print()
    log(SEP, "cyan")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    log(f"{IC_HDR} SNAPSHOT • {now} • mode={TRADE_MODE.upper()} • {safe_symbol(SYMBOL)} • {INTERVAL}", "cyan")
    log("—", "cyan")
    log(f"{IC_BAL} Balance (USDT): {fmt(balance,2)}", "yellow")
    log(f"{IC_PRC} Price          : {fmt(price,6)}", "green")
    if ind:
        log(f"{IC_EMA} EMA20/50/200  : {fmt(ind.get('ema20'),6)} / {fmt(ind.get('ema50'),6)} / {fmt(ind.get('ema200'),6)}", "blue")
        log(f"{IC_RSI} RSI(14)       : {fmt(ind.get('rsi'),2)}", "magenta")
        log(f"{IC_ADX} ADX(14)       : {fmt(ind.get('adx'),2)}", "magenta")
        log(f"{IC_ATR} ATR(14)       : {fmt(ind.get('atr'),6)}  | ATR%={fmt(ind.get('atr_pct'),2)}", "magenta")
    if pos["open"]:
        side = pos["side"].upper()
        log(f"{IC_POS} Position      : {side} | entry={fmt(pos['entry'],6)} qty={fmt(pos['qty'],4)}", "white")
        log(f"{IC_TP}/{IC_SL} TP / SL       : {fmt(pos['tp'],6)} / {fmt(pos['sl'],6)}", "white")
        log(f"📈 PnL current  : {fmt(pos['pnl'],6)}", "white")
    log(f"📦 Compound PnL : {fmt(total_pnl,6)}", "yellow")
    log(SEP, "cyan")

# ======================= Open / Close / Trailing =======================
def update_trailing(px, ind):
    if not state["open"] or not px or not ind: return
    atr = ind.get("atr"); stl = ind.get("st_line")
    if not atr: return
    if state["side"]=="long":
        gain = px - state["entry"]; be = state["entry"] + 0.1*atr
        if gain >= 1.0*atr and state["sl"] < be:
            state["sl"] = be; log("🔒 move SL → BE", "yellow")
        trail = max(stl, px - 1.0*atr) if stl else (px - 1.0*atr)
        if trail > state["sl"]:
            state["sl"] = trail; log(f"↗️ trail SL {fmt(state['sl'],6)}","yellow")
    else:
        gain = state["entry"] - px; be = state["entry"] - 0.1*atr
        if gain >= 1.0*atr and state["sl"] > be:
            state["sl"] = be; log("🔒 move SL → BE", "yellow")
        trail = min(stl, px + 1.0*atr) if stl else (px + 1.0*atr)
        if trail < state["sl"]:
            state["sl"] = trail; log(f"↘️ trail SL {fmt(state['sl'],6)}","yellow")

def place_protected_order(side, qty, ref_price, atr=None):
    """يدخل الصفقة + يرفق SL/TP + يتحقق منهم، وإلا يلغي الدخول فورًا."""
    global state
    # مستويات من ATR (افتراضي 1.2/1.8)
    if not atr or atr <= 0:
        sl = ref_price * (0.985 if side=="buy" else 1.015)
        tp = ref_price * (1.015 if side=="buy" else 0.985)
    else:
        sl = ref_price - 1.2*atr if side=="buy" else ref_price + 1.2*atr
        tp = ref_price + 1.8*atr if side=="buy" else ref_price - 1.8*atr

    # PAPER
    if TRADE_MODE == "paper":
        state.update({"open": True, "side": "long" if side=="buy" else "short",
                      "entry": ref_price, "qty": qty, "tp": tp, "sl": sl, "pnl": 0.0})
        tag = IC_BUY if side=="buy" else IC_SELL
        log(f"{IC_TRD} {tag} [PAPER] qty={fmt(qty,4)} entry={fmt(ref_price,6)} TP={fmt(tp,6)} SL={fmt(sl,6)}","green")
        return

    # LIVE
    try:
        ex.set_leverage(LEVERAGE, safe_symbol(SYMBOL), params={"side":"BOTH"})
    except Exception as e:
        log(f"{IC_BAD} set_leverage: {e}", "red")

    try:
        ex.create_order(safe_symbol(SYMBOL), "market", side, qty, params={"reduceOnly": False})
        log(f"{IC_TRD} submit {('🟢 BUY' if side=='buy' else '🔴 SELL')} qty={fmt(qty,4)}", "green")
    except Exception as e:
        log(f"{IC_BAD} open error: {e}", "red"); return

    # تحقق/إرفاق حماية
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
        tag = IC_BUY if side=="buy" else IC_SELL
        log(f"{IC_TRD} {tag} CONFIRMED qty={fmt(size,4)} entry={fmt(entry,6)} TP={fmt(tp,6)} SL={fmt(sl,6)}","green")

    except Exception as e:
        log(f"{IC_BAD} post-trade validation error: {e}", "red")

def close_position(reason):
    """يغلق المركز + ينظّف أوامر OCO + يحدّث البلوك/الكوولداون"""
    global state, compound_pnl, cool_bars, block_dir, block_bars_left
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

    cancel_all_open_orders()
    compound_pnl += pnl
    log(f"{IC_CLS} Close {state['side'].upper()} reason={reason} pnl={fmt(pnl,6)} total={fmt(compound_pnl,6)}","magenta")
    if reason == "sl":
        block_dir = state["side"]; block_bars_left = max(block_bars_left, COOLDOWN_BARS if COOLDOWN_BARS>1 else 2)

    state.update({"open": False, "side": None, "entry": None, "qty": None, "tp": None, "sl": None, "pnl": 0.0})
    cool_bars = COOLDOWN_BARS

# ======================= Main Loop =======================
def trade_loop():
    global state, compound_pnl, cool_bars, latest_bar_ts, latest_indicators, block_dir, block_bars_left
    log(f"{IC_OK} Trading loop started on {safe_symbol(SYMBOL)} @ {INTERVAL}", "cyan")
    while True:
        try:
            df = fetch_ohlcv()
            if df is None or len(df) < 210:
                log(f"{IC_SHD} no_entry: insufficient_bars", "yellow"); time.sleep(30); continue

            bar_ts = int(df['time'].iloc[-2])  # آخر شمعة مغلقة
            new_bar = (latest_bar_ts is None) or (bar_ts != latest_bar_ts)

            # تحديث PnL الحالي
            px_now = price_now()
            if state["open"] and px_now:
                state["pnl"] = (px_now - state["entry"]) * state["qty"] if state["side"]=="long" else (state["entry"] - px_now) * state["qty"]

            if new_bar:
                latest_indicators = compute_indicators_closed(df)
                latest_bar_ts = bar_ts

                bal = balance_usdt(); px = price_now()
                snapshot(bal, px, latest_indicators, state.copy(), compound_pnl)

                # عدادات بالشموع
                if cool_bars > 0: cool_bars -= 1
                if block_bars_left > 0: block_bars_left -= 1

                # Trailing عند إغلاق الشمعة
                if state["open"] and px: update_trailing(px, latest_indicators)

                # قرارات الدخول على الشمعة المغلقة فقط
                if not state["open"] and cool_bars == 0:
                    side_sig, engine, no_reason = signal_with_reason(latest_indicators)
                    if side_sig is None:
                        log(f"{IC_SHD} no_entry: {no_reason}", "yellow")
                    else:
                        # حظر الاتجاه بعد خسارة
                        if block_bars_left>0 and ((block_dir=="long" and side_sig=="buy") or (block_dir=="short" and side_sig=="sell")):
                            log(f"{IC_SHD} no_entry: blocked_after_loss dir={block_dir} bars_left={block_bars_left}", "yellow")
                        else:
                            ref = latest_indicators["price"]; px = price_now(); bal = balance_usdt()
                            if not (px and ref and bal):
                                log(f"{IC_SHD} no_entry: px_or_balance_na", "yellow")
                            elif abs(px-ref)/ref > MAX_SLIPPAGE_PCT:
                                log(f"{IC_SHD} no_entry: slippage px={fmt(px,6)} ref={fmt(ref,6)}", "yellow")
                            else:
                                atr = latest_indicators.get("atr") or 0
                                sl = (px - 1.2*atr) if side_sig=="buy" else (px + 1.2*atr)
                                tp = (px + 1.8*atr) if side_sig=="buy" else (px - 1.8*atr)
                                qty = compute_size_risk(bal, px, sl)
                                if qty and qty > 0:
                                    tag = IC_BUY if side_sig=="buy" else IC_SELL
                                    log(f"{IC_TRD} {tag} signal engine={engine} qty={fmt(qty,4)} entry={fmt(px,6)} tp={fmt(tp,6)} sl={fmt(sl,6)}", "green")
                                    place_protected_order(side_sig, qty, px, atr=atr)
                                    cool_bars = COOLDOWN_BARS
                                else:
                                    log(f"{IC_SHD} no_entry: qty_zero risk_pct={RISK_PCT}", "yellow")
            else:
                # أثناء تكوّن الشمعة: متابعة Trailing وSoft-guards
                px = price_now()
                if state["open"] and px:
                    update_trailing(px, latest_indicators)
                    if state["side"]=="long" and (px <= state["sl"] or px >= state["tp"]):
                        close_position("tp" if px >= state["tp"] else "sl")
                    elif state["side"]=="short" and (px >= state["sl"] or px <= state["tp"]):
                        close_position("tp" if px <= state["tp"] else "sl")

        except Exception as e:
            log(f"{IC_BAD} loop error: {e}", "red")

        time.sleep(5)  # قرارات الدخول على إغلاق الشموع فقط

# ======================= Keepalive =======================
def keepalive_loop():
    if not SELF_URL:
        log("SELF_URL not set — keepalive disabled", "yellow"); return
    url = SELF_URL.rstrip("/")
    while True:
        try: requests.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# ======================= Flask (Dashboard) =======================
app = Flask(__name__)

@app.route("/")
def home():
    return f"{IC_OK} Bot Running — Signals OCO Protected — {safe_symbol(SYMBOL)} {INTERVAL}"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": safe_symbol(SYMBOL),
        "interval": INTERVAL,
        "mode": TRADE_MODE,
        "leverage": LEVERAGE,
        "risk_pct": RISK_PCT,
        "min_adx": MIN_ADX,
        "cooldown_bars": COOLDOWN_BARS,
        "balance": balance_usdt(),
        "price": price_now(),
        "position": state,
        "compound_pnl": compound_pnl,
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    })

# ======================= Boot =======================
threading.Thread(target=trade_loop, daemon=True).start()
threading.Thread(target=keepalive_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
