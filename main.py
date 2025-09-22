# -*- coding: utf-8 -*-
"""
BingX Futures LIVE (15m) — Wilder Indicators + Protected Orders + Filters + KeepAlive
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
RISK_ALLOC   = float(os.getenv("RISK_ALLOC", "0.60"))   # 60% of balance
TRADE_MODE   = os.getenv("TRADE_MODE", "live").lower()  # live/paper
SELF_URL     = os.getenv("RENDER_EXTERNAL_URL", "") or os.getenv("SELF_URL", "")

# Strategy thresholds
RSI_LONG_TH      = float(os.getenv("RSI_LONG_TH", "55"))
RSI_SHORT_TH     = float(os.getenv("RSI_SHORT_TH", "45"))
MIN_ADX          = float(os.getenv("MIN_ADX", "18"))
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
SEP = colored("—"*78, "cyan")

def log(msg, color="white"):
    print(colored(msg, color), flush=True)

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

# ======== Indicators (Wilder matching) ========
def fetch_ohlcv(limit=300):
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

def compute_indicators(df: pd.DataFrame):
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)

    ema20  = ema(c, 20)
    ema50  = ema(c, 50)
    ema200 = ema(c, 200)

    # RSI(14) Wilder
    change = c.diff()
    gain = change.clip(lower=0)
    loss = (-change.clip(upper=0))
    avg_gain = rma(gain, 14)
    avg_loss = rma(loss, 14).replace(0, 1e-12)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # ATR(14) Wilder
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = rma(tr, 14)

    # ADX(14) Wilder
    up_move = h.diff()
    dn_move = (-l.diff())
    plus_dm  = ((up_move > dn_move) & (up_move > 0)) * up_move
    minus_dm = ((dn_move > up_move) & (dn_move > 0)) * dn_move
    tr14  = rma(tr, 14)
    pdi14 = 100 * rma(plus_dm, 14) / tr14
    mdi14 = 100 * rma(minus_dm, 14) / tr14
    dx    = ( (pdi14 - mdi14).abs() / (pdi14 + mdi14).replace(0,1e-12) ) * 100
    adx14 = rma(dx, 14)

    last = lambda s: float(s.dropna().iloc[-1]) if s.dropna().size else None
    prev = lambda s: float(s.dropna().iloc[-2]) if s.dropna().size > 1 else None

    out = {
        "price": last(c),
        "ema20": last(ema20), "ema50": last(ema50), "ema200": last(ema200),
        "ema20_prev": prev(ema20), "ema50_prev": prev(ema50),
        "rsi": last(rsi), "adx": last(adx14), "atr": last(atr)
    }
    return out

def signal_balanced(ind):
    if not ind or any(ind[k] is None for k in ["price","ema20","ema50","ema200","ema20_prev","ema50_prev","rsi","adx"]):
        return None
    p, e20, e50, e200 = ind["price"], ind["ema20"], ind["ema50"], ind["ema200"]
    e20p, e50p = ind["ema20_prev"], ind["ema50_prev"]
    rsi, adx = ind["rsi"], ind["adx"]

    cross_up   = (e20p <= e50p) and (e20 > e50)
    cross_down = (e20p >= e50p) and (e20 < e50)
    long_ok  = p > e200
    short_ok = p < e200

    if adx is None or adx < MIN_ADX: return None
    if cross_up and long_ok and rsi is not None and rsi > RSI_LONG_TH:
        return "buy"
    if cross_down and short_ok and rsi is not None and rsi < RSI_SHORT_TH:
        return "sell"
    return None

# ======== State ========
state = {"open": False, "side": None, "entry": None, "qty": None, "tp": None, "sl": None, "pnl": 0.0}
compound_pnl = 0.0
cool_bars = 0

# ======== Snapshot ========
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
        log(f"{IC_ATR} ATR(14)       : {fmt(ind.get('atr'),6)}", "magenta")
    if pos["open"]:
        side = pos["side"].upper()
        log(f"{IC_POS} Position      : {side} | entry={fmt(pos['entry'],6)} qty={fmt(pos['qty'],4)}", "white")
        log(f"{IC_TP}/{IC_SL} TP / SL       : {fmt(pos['tp'],6)} / {fmt(pos['sl'],6)}", "white")
        log(f"📈 PnL current  : {fmt(pos['pnl'],6)}", "white")
    log(f"📦 Compound PnL : {fmt(total_pnl,6)}", "yellow")
    log(SEP, "cyan")

# ======== Sizing / Helpers ========
def compute_size(balance, price):
    if not balance or not price: return 0
    raw = (balance * RISK_ALLOC * LEVERAGE) / price
    return market_amount(raw)

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
            log(f"{IC_SHD} protection attach failed: {e2}", "yellow")
            return False

# ======== Open / Close ========
def place_protected_order(side, qty, ref_price, atr=None):
    global state
    # SL/TP من ATR أو fallback ~1.5%
    if not atr or atr <= 0:
        sl = ref_price * (0.985 if side=="buy" else 1.015)
        tp = ref_price * (1.015 if side=="buy" else 0.985)
    else:
        sl = ref_price - 1.2*atr if side=="buy" else ref_price + 1.2*atr
        tp = ref_price + 1.8*atr if side=="buy" else ref_price - 1.8*atr

    if TRADE_MODE == "paper":
        state.update({"open": True, "side": "long" if side=="buy" else "short",
                      "entry": ref_price, "qty": qty, "tp": tp, "sl": sl, "pnl": 0.0})
        log(f"{IC_TRD} [PAPER] Open {side} qty={fmt(qty,4)} entry={fmt(ref_price,6)} TP={fmt(tp,6)} SL={fmt(sl,6)}","green")
        return

    try:
        ex.set_leverage(LEVERAGE, safe_symbol(SYMBOL), params={"side":"BOTH"})
    except Exception as e:
        log(f"{IC_BAD} set_leverage: {e}", "red")

    try:
        ex.create_order(safe_symbol(SYMBOL), "market", side, qty, params={"reduceOnly": False})
        log(f"{IC_TRD} submit {side} qty={fmt(qty,4)}", "green")
    except Exception as e:
        log(f"{IC_BAD} open error: {e}", "red"); return

    # تحقق بعد التنفيذ
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
        if not attached:
            # إلغاء الدخول لحماية الحساب
            try: ex.create_order(safe_symbol(SYMBOL), "market", side_opp, size, params={"reduceOnly": True})
            except Exception as e3: log(f"{IC_BAD} emergency close: {e3}", "red")
            log(f"{IC_SHD} aborted entry since SL/TP not attached.", "yellow")
            return

        state.update({"open": True, "side": "long" if side=="buy" else "short",
                      "entry": entry, "qty": size, "tp": tp, "sl": sl, "pnl": 0.0})
        log(f"{IC_TRD} Open {side} (confirmed) qty={fmt(size,4)} entry={fmt(entry,6)} TP={fmt(tp,6)} SL={fmt(sl,6)}","green")

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

# ======== Main loop ========
def trade_loop():
    global state, compound_pnl, cool_bars
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()
            ind = compute_indicators(df) if df is not None and len(df) > 220 else {}
            side = signal_balanced(ind)
            atr  = ind.get("atr") if ind else None

            # تحديث PnL الجاري
            if state["open"] and px:
                if state["side"]=="long":
                    state["pnl"] = (px - state["entry"]) * state["qty"]
                else:
                    state["pnl"] = (state["entry"] - px) * state["qty"]

            snapshot(bal, px, ind, state.copy(), compound_pnl)

            if cool_bars > 0:
                cool_bars -= 1
                log(f"{IC_SHD} cooldown bars left: {cool_bars}", "yellow")
                time.sleep(60); continue

            # دخول محمي
            if not state["open"] and side and px and bal and ind:
                ref = ind["price"]
                if abs(px - ref)/ref <= MAX_SLIPPAGE_PCT:
                    qty = compute_size(bal, px)
                    if qty and qty > 0:
                        place_protected_order(side, qty, px, atr=atr)
                else:
                    log(f"{IC_SHD} skip entry (slippage) px={fmt(px,6)} ref={fmt(ref,6)}","yellow")

            # Soft-guard لمس TP/SL
            if state["open"] and px:
                if state["side"]=="long" and (px <= state["sl"] or px >= state["tp"]):
                    close_position("tp" if px >= state["tp"] else "sl")
                elif state["side"]=="short" and (px >= state["sl"] or px <= state["tp"]):
                    close_position("tp" if px <= state["tp"] else "sl")

        except Exception as e:
            log(f"{IC_BAD} loop error: {e}", "red")

        time.sleep(60)  # 1m tick for 15m strategy

# ======== Keepalive ========
def keepalive_loop():
    if not SELF_URL:
        log("SELF_URL not set — keepalive disabled", "yellow"); return
    url = SELF_URL.rstrip("/")
    while True:
        try: requests.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# ======== Flask ========
app = Flask(__name__)

@app.route("/")
def home():
    return f"{IC_OK} Bot Running — Dashboard Active"

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
