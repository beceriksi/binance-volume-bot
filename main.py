# main.py
# Binance USDT scanner -> EMA50/EMA200 (15m + 1h), RSI, volume spike, buy-wall -> Telegram alerts
import os, time, math, logging
from datetime import datetime

import ccxt
import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("binance-usdt-scanner")

# --- CONFIG (env üzerinden değiştirilebilir) ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # GitHub Secret
CHAT_ID        = os.getenv("CHAT_ID")         # GitHub Secret (string)

EXCHANGE_NAME  = os.getenv("EXCHANGE", "binance")
QUOTE          = "USDT"
TIMEFRAME_1H   = os.getenv("TIMEFRAME_1H", "1h")
TIMEFRAME_15M  = os.getenv("TIMEFRAME_15M", "15m")
LIMIT          = int(os.getenv("LIMIT", "300"))

# Strategy params
EMA_FAST = int(os.getenv("EMA_FAST", "50"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "200"))
VOLUME_MULTIPLIER = float(os.getenv("VOLUME_MULTIPLIER", "1.3"))  # 15m volume spike multiplier
RSI_LOW  = float(os.getenv("RSI_LOW", "40"))
RSI_HIGH = float(os.getenv("RSI_HIGH", "70"))

SCAN_PAUSE = int(os.getenv("SCAN_PAUSE", "5"))  # seconds between symbols (politeness)

# --- util ---
def send_telegram(text):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.info("Telegram not configured — skipping send")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        logger.info("Telegram message sent (%d chars)", len(text))
    except Exception as e:
        logger.exception("Failed to send telegram: %s", e)

# indicators
def ema(arr, window):
    arr = np.asarray(arr, dtype=float)
    alpha = 2.0 / (window + 1.0)
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

def rsi(series, window=14):
    series = np.asarray(series, dtype=float)
    deltas = np.diff(series)
    if len(deltas) < window:
        return np.array([50.0]*len(series))
    up = np.where(deltas>0, deltas, 0.0)
    down = np.where(deltas<0, -deltas, 0.0)
    up_ema = pd.Series(up).ewm(alpha=1/window, adjust=False).mean().to_numpy()
    down_ema = pd.Series(down).ewm(alpha=1/window, adjust=False).mean().to_numpy()
    rs = up_ema / (down_ema + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # pad to match series length
    rsi_full = np.concatenate(([rsi[0]] , rsi)) if len(rsi)>0 else np.array([50.0]*len(series))
    # ensure len match
    if len(rsi_full) < len(series):
        rsi_full = np.pad(rsi_full, (len(series)-len(rsi_full),0), 'edge')
    return rsi_full

def has_buy_wall(order_book, multiplier=2.0):
    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])
    if not bids or not asks:
        return False
    top_bid_vol = bids[0][1]
    top_asks = [a[1] for a in asks[:5]] if len(asks)>=1 else [1e-9]
    avg_asks = sum(top_asks)/max(len(top_asks),1)
    return top_bid_vol >= avg_asks * multiplier

# --- exchange setup ---
def make_exchange(name="binance"):
    ex = getattr(ccxt, name)({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    ex.load_markets()
    return ex

def fetch_ohlcv_safe(ex, symbol, timeframe, limit=200):
    try:
        data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return np.array(data) if data else None
    except Exception as e:
        logger.warning("fetch_ohlcv failed %s %s -> %s", symbol, timeframe, str(e))
        return None

# --- evaluation per symbol ---
def evaluate_symbol(ex, symbol):
    # returns dict with match bool and details
    res = {"symbol": symbol, "match": False, "reasons": [], "details": {}}

    kl15 = fetch_ohlcv_safe(ex, symbol, TIMEFRAME_15M, limit=200)
    kl1h = fetch_ohlcv_safe(ex, symbol, TIMEFRAME_1H, limit=200)
    if kl15 is None or kl1h is None:
        res["reasons"].append("ohlcv fetch failed")
        return res

    close15 = kl15[:,4].astype(float)
    vol15   = kl15[:,5].astype(float)
    close1h = kl1h[:,4].astype(float)

    # EMA checks
    if len(close15) < max(EMA_SLOW, EMA_FAST) or len(close1h) < max(EMA_SLOW, EMA_FAST):
        res["reasons"].append("not enough history")
        return res

    ema50_15 = ema(close15, EMA_FAST)[-1]
    ema200_15 = ema(close15, EMA_SLOW)[-1]
    ema50_1h = ema(close1h, EMA_FAST)[-1]
    ema200_1h = ema(close1h, EMA_SLOW)[-1]

    cond_15 = ema50_15 > ema200_15
    cond_1h = ema50_1h > ema200_1h
    if not cond_15:
        res["reasons"].append("15m EMA fail")
    if not cond_1h:
        res["reasons"].append("1h EMA fail")

    # volume spike on 15m
    avg_vol = np.mean(vol15[-20:]) if len(vol15)>=20 else np.mean(vol15)
    curr_vol = float(vol15[-1])
    vol_ok = curr_vol >= avg_vol * VOLUME_MULTIPLIER
    if not vol_ok:
        res["reasons"].append("vol not spike")

    # RSI 15m
    rsi_vals = rsi(close15)
    rsi_latest = float(rsi_vals[-1])
    rsi_ok = (RSI_LOW <= rsi_latest <= RSI_HIGH)
    if not rsi_ok:
        res["reasons"].append(f"rsi {rsi_latest:.1f} out of {RSI_LOW}-{RSI_HIGH}")

    # orderbook buy wall
    try:
        ob = ex.fetch_order_book(symbol, limit=10)
        wall = has_buy_wall(ob, multiplier=2.0)
        if not wall:
            res["reasons"].append("no buy wall")
    except Exception as e:
        res["reasons"].append("orderbook fail")
        wall = False

    match = cond_15 and cond_1h and vol_ok and rsi_ok and wall
    res["match"] = bool(match)
    res["details"] = {
        "ema50_15": float(ema50_15),
        "ema200_15": float(ema200_15),
        "ema50_1h": float(ema50_1h),
        "ema200_1h": float(ema200_1h),
        "avg_vol_15_20": float(avg_vol),
        "curr_vol_15": float(curr_vol),
        "rsi_15": float(rsi_latest),
        "buy_wall": bool(wall),
        "last_close_15": float(close15[-1]),
        "time_utc": datetime.utcnow().isoformat() + "Z"
    }
    if match:
        res["reasons"] = ["all green"]
    return res

# --- main runner (single scan) ---
def run_once():
    ex = make_exchange(EXCHANGE_NAME)
    # collect USDT markets
    markets = [s for s,m in ex.markets.items() if m.get("active") and m.get("spot") and m.get("quote")==QUOTE]
    markets = sorted(markets)
    logger.info("Found %d %s markets (scanning top %d)", len(markets), QUOTE, len(markets))
    hits = []
    for i, sym in enumerate(markets, 1):
        try:
            logger.info("Evaluating %s (%d/%d)", sym, i, len(markets))
            r = evaluate_symbol(ex, sym)
            if r["match"]:
                hits.append(r)
                logger.info("MATCH %s", sym)
            time.sleep(SCAN_PAUSE)
        except Exception as e:
            logger.exception("Error evaluating %s: %s", sym, str(e))
    # prepare message
    if hits:
        lines = ["🔥 *Binance USDT — Strategy Matches* 🔥", ""]
        for h in hits:
            d = h["details"]
            lines.append(f"*{h['symbol']}* | Close={d['last_close_15']:.6f}")
            lines.append(f"EMA15: {d['ema50_15']:.6f} / {d['ema200_15']:.6f}  | EMA1h: {d['ema50_1h']:.6f} / {d['ema200_1h']:.6f}")
            lines.append(f"RSI15: {d['rsi_15']:.1f}  Vol15: {int(d['curr_vol_15']):,} (avg20 {int(d['avg_vol_15_20']):,})")
            lines.append(f"BuyWall: {d['buy_wall']}  Time: {d['time_utc']}")
            lines.append("---")
        text = "\n".join(lines)
    else:
        text = "📭 Binance USDT scan: no matches at this run."

    send_telegram(text)

# allow running from CLI or as scheduled job
if __name__ == "__main__":
    run_once()
