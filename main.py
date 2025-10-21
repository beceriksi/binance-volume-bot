# main.py
# MEXC / USDT / 1H — EMA10/20/30 cross + RSI yükselişi + price cap -> Telegram alerts
# NOT: Bu kod sadece SİNYAL üretir. LIVE TRADE YAPMAZ.

import os, time, math, logging
from datetime import datetime
import ccxt
import pandas as pd
import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mexc-1h-strategy")

# ---------- CONFIG (env üzerinden değiştirebilirsiniz) ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # GitHub Secret
CHAT_ID        = os.getenv("CHAT_ID")         # GitHub Secret

EXCHANGE_NAME = os.getenv("EXCHANGE", "mexc")
QUOTE = "USDT"
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
LIMIT = int(os.getenv("LIMIT", "300"))

# Strategy params
EMA_FAST = int(os.getenv("EMA_FAST", "10"))
EMA_MID  = int(os.getenv("EMA_MID", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "30"))

RSI_WINDOW = int(os.getenv("RSI_WINDOW", "14"))
RSI_MIN = float(os.getenv("RSI_MIN", "50.0"))       # RSI must be > this
PRICE_MAX_CHANGE = float(os.getenv("PRICE_MAX_CHANGE", "0.05"))  # <= 5%
PRICE_MIN_CHANGE = float(os.getenv("PRICE_MIN_CHANGE", "-0.02")) # >= -2%

CROSS_LOOKBACK = int(os.getenv("CROSS_LOOKBACK", "2"))  # last N bars where cross may have occurred
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.03"))  # suggested stop-loss (3%)

SCAN_PAUSE = float(os.getenv("SCAN_PAUSE", "0.15"))  # seconds between symbol requests
CSV_OUT = os.getenv("CSV_OUT", "mexc_1h_strategy_hits.csv")

# ---------- HELPERS ----------
def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.info("Telegram credentials not set - skipping send.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        logger.info("Telegram message sent.")
    except Exception as e:
        logger.exception("Failed to send Telegram message: %s", e)

def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def recent_crossed_up(fast: pd.Series, slow: pd.Series, lookback: int = 2) -> bool:
    # Check if fast crossed above slow within last `lookback` bars
    n = len(fast)
    if n < lookback + 1:
        return False
    for i in range(1, lookback+1):
        # compare bar -i and bar -(i+1)
        cur = n - i
        prev = n - i - 1
        if prev < 0:
            continue
        if (fast.iloc[prev] <= slow.iloc[prev]) and (fast.iloc[cur] > slow.iloc[cur]):
            return True
    return False

# ---------- EXCHANGE ----------
def make_exchange(name="mexc"):
    ex = getattr(ccxt, name)({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"}
    })
    # load markets may raise if API blocked - let it bubble up
    ex.load_markets()
    return ex

def fetch_ohlcv_safe(ex, symbol, timeframe, limit=300):
    try:
        raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not raw:
            return None
        df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("time", inplace=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        return df
    except Exception as e:
        logger.warning("fetch_ohlcv failed for %s %s: %s", symbol, timeframe, str(e))
        return None

# ---------- EVALUATE SYMBOL ----------
def evaluate_symbol(ex, symbol):
    df = fetch_ohlcv_safe(ex, symbol, TIMEFRAME, limit=LIMIT)
    if df is None or len(df) < max(EMA_SLOW, RSI_WINDOW) + 5:
        return None

    close = df["close"]
    # indicators
    e_fast = ema(close, EMA_FAST)
    e_mid  = ema(close, EMA_MID)
    e_slow = ema(close, EMA_SLOW)
    rsi = compute_rsi(close, RSI_WINDOW)

    # latest indexes
    if len(close) < 3:
        return None
    last = -1
    prev = -2

    # Conditions:
    # 1) latest EMA order: fast > mid > slow
    cond_order = (e_fast.iloc[last] > e_mid.iloc[last]) and (e_mid.iloc[last] > e_slow.iloc[last])
    # 2) recent up-cross of fast over mid and fast over slow within CROSS_LOOKBACK bars
    cross_mid = recent_crossed_up(e_fast, e_mid, lookback=CROSS_LOOKBACK)
    cross_slow = recent_crossed_up(e_fast, e_slow, lookback=CROSS_LOOKBACK)
    cross_ok = cross_mid and cross_slow
    # 3) RSI > RSI_MIN and RSI increased vs previous bar
    rsi_ok = (rsi.iloc[last] > RSI_MIN) and (rsi.iloc[last] > rsi.iloc[prev])
    # 4) price change constraint
    change = (close.iloc[last] - close.iloc[prev]) / max(close.iloc[prev], 1e-12)
    price_ok = (change <= PRICE_MAX_CHANGE) and (change >= PRICE_MIN_CHANGE)

    reasons = []
    if not cond_order:
        reasons.append("EMA order fail")
    if not cross_ok:
        reasons.append("No recent EMA up-cross")
    if not rsi_ok:
        reasons.append("RSI condition fail")
    if not price_ok:
        reasons.append(f"price change {change:.3f} out of range")

    match = cond_order and cross_ok and rsi_ok and price_ok
    details = {
        "symbol": symbol,
        "close": float(close.iloc[last]),
        "change": float(change),
        "ema_fast": float(e_fast.iloc[last]),
        "ema_mid": float(e_mid.iloc[last]),
        "ema_slow": float(e_slow.iloc[last]),
        "rsi": float(rsi.iloc[last]),
        "rsi_prev": float(rsi.iloc[prev]),
        "time_utc": df.index[last].to_pydatetime().isoformat() + "Z"
    }
    return {"match": bool(match), "reasons": reasons, "details": details}

# ---------- MAIN RUN ----------
def run_once():
    ex = make_exchange(EXCHANGE_NAME)
    # build list of USDT spot markets
    markets = [s for s,m in ex.markets.items() if m.get("active") and m.get("spot") and m.get("quote") == QUOTE]
    markets = sorted(markets)
    logger.info("Scanning %d %s markets", len(markets), QUOTE)

    hits = []
    for i, sym in enumerate(markets, 1):
        try:
            logger.debug("Checking %s (%d/%d)", sym, i, len(markets))
            res = evaluate_symbol(ex, sym)
            if res and res["match"]:
                hits.append(res["details"])
                logger.info("MATCH %s | RSI %.2f | change %.2f%%", sym, res["details"]["rsi"], res["details"]["change"]*100)
            time.sleep(SCAN_PAUSE)
        except Exception as e:
            logger.exception("Error for %s: %s", sym, str(e))
            continue

    # prepare message
    if hits:
        lines = ["🔥 *MEXC 1H — EMA(10/20/30) + RSI >50 (Temiz)* 🔥", ""]
        for d in hits:
            stop_price = d["close"] * (1.0 - STOP_LOSS_PCT)
            lines.append(f"*{d['symbol']}* | Close={d['close']:.6f} | Δ={d['change']*100:.2f}%")
            lines.append(f"EMA10={d['ema_fast']:.6f} EMA20={d['ema_mid']:.6f} EMA30={d['ema_slow']:.6f}")
            lines.append(f"RSI={d['rsi']:.2f} (prev {d['rsi_prev']:.2f})")
            lines.append(f"Önerilen stop-loss: {stop_price:.6f}  (≈ {STOP_LOSS_PCT*100:.1f}% altında)")
            if d['rsi'] >= 65:
                lines.append("⚠️ RSI yaklaşmakta: overbought uyarısı (RSI ≥ 65)")
            lines.append("---")
        message = "\n".join(lines)
    else:
        message = "📭 MEXC 1H scan: kriterlere uyan coin yok."

    send_telegram(message)
    # also save CSV for artifact (if desired)
    try:
        import csv
        with open(CSV_OUT, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol","close","change","ema10","ema20","ema30","rsi","time"])
            for d in hits:
                w.writerow([d["symbol"], d["close"], d["change"], d["ema_fast"], d["ema_mid"], d["ema_slow"], d["rsi"], d["time_utc"]])
    except Exception:
        pass

if __name__ == "__main__":
    logger.info("Starting single-run MEXC scanner")
    run_once()
