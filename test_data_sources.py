#!/usr/bin/env python3
"""
test_data_sources_v2.py — 更直白版

用法：
  1. 把下面三組 key 貼在引號之間
  2. 存檔
  3. python test_data_sources_v2.py
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════
# ⚠️ ⚠️ ⚠️  把你的 key 貼在這三個引號裡 ⚠️ ⚠️ ⚠️
# ═══════════════════════════════════════════════════════

NASDAQ_KEY    = "xMQVLs5ussHBFoRAZzt8"
FRED_KEY      = "43c0811da53d9355e6a308a2a18589c6"
FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoia2Fpc2xpcGtub3QzNCIsImVtYWlsIjoicGllcmNlLmxlZS4zNEBnbWFpbC5jb20iLCJ0b2tlbl92ZXJzaW9uIjowfQ.9T3Y6zCiy5OlAlTyewu-p3THfao8Haz7jz1lj0o6J7Q"

# ═══════════════════════════════════════════════════════
# 範例（這是錯的，僅供格式示範）：
#   NASDAQ_KEY = "xMQVLs5ussHBFoRAZzt8"
#   FRED_KEY   = "abc123def4567890abc123def4567890"
# ═══════════════════════════════════════════════════════


# ─────────── 以下不用改 ───────────
# 環境變數覆蓋（沒設環境變數就用上面貼的值）
NASDAQ_KEY    = os.environ.get("NASDAQ_DATA_LINK_API_KEY", NASDAQ_KEY).strip()
FRED_KEY      = os.environ.get("FRED_API_KEY",             FRED_KEY).strip()
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN",            FINMIND_TOKEN).strip()

# 偵測還沒改的預設文字 → 視同未設定
if "把" in NASDAQ_KEY or "貼這裡" in NASDAQ_KEY:
    NASDAQ_KEY = ""
if "把" in FRED_KEY or "貼這裡" in FRED_KEY:
    FRED_KEY = ""

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
results = []


def run(name, func):
    try:
        msg = func()
        results.append((name, "✅ PASS", msg))
        print(f"✅ {name}")
        print(f"   {msg}\n")
    except Exception as e:
        results.append((name, "❌ FAIL", str(e)[:250]))
        print(f"❌ {name}")
        print(f"   Error: {str(e)[:250]}\n")


# ─── 1. Nasdaq Data Link：AAII ─────────────────────────
def t_nasdaq_aaii():
    if not NASDAQ_KEY:
        return "略過（無 key）"
    url = "https://data.nasdaq.com/api/v3/datasets/AAII/AAII_SENTIMENT/data.json"
    r = requests.get(url, params={"api_key": NASDAQ_KEY, "rows": 3}, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    d = r.json().get("dataset_data", {})
    rows = d.get("data", [])
    cols = d.get("column_names", [])
    if not rows:
        raise RuntimeError("無資料")
    return f"欄位={cols} | 最新={rows[0]}"


# ─── 2. Nasdaq Data Link：NAAIM ────────────────────────
def t_nasdaq_naaim():
    if not NASDAQ_KEY:
        return "略過（無 key）"
    url = "https://data.nasdaq.com/api/v3/datasets/NAAIM/NAAIM/data.json"
    r = requests.get(url, params={"api_key": NASDAQ_KEY, "rows": 3}, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    d = r.json().get("dataset_data", {})
    rows = d.get("data", [])
    cols = d.get("column_names", [])
    if not rows:
        raise RuntimeError("無資料")
    return f"欄位={cols} | 最新={rows[0]}"


# ─── 3. FRED：10Y-2Y 殖利率差（衰退領先指標）─────────
def t_fred_t10y2y():
    if not FRED_KEY:
        return "略過（無 key）"
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": "T10Y2Y", "api_key": FRED_KEY,
              "file_type": "json", "sort_order": "desc", "limit": 3}
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    obs = r.json().get("observations", [])
    if not obs:
        raise RuntimeError("無資料")
    return f"最新 {obs[0]['date']} = {obs[0]['value']}%（負值=倒掛）"


# ─── 4. FRED：High Yield 信用利差 ──────────────────────
def t_fred_hy():
    if not FRED_KEY:
        return "略過（無 key）"
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": "BAMLH0A0HYM2", "api_key": FRED_KEY,
              "file_type": "json", "sort_order": "desc", "limit": 3}
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    obs = r.json().get("observations", [])
    if not obs:
        raise RuntimeError("無資料")
    return f"最新 {obs[0]['date']} = {obs[0]['value']}%（>5% 高風險）"


# ─── 5. yfinance：VIX 期限結構 ─────────────────────────
def t_vix_term():
    import yfinance as yf
    vix = yf.Ticker("^VIX").history(period="5d")
    vix3m = yf.Ticker("^VIX3M").history(period="5d")
    if vix.empty or vix3m.empty:
        raise RuntimeError("yfinance VIX 無資料")
    v_now = float(vix["Close"].iloc[-1])
    v3m   = float(vix3m["Close"].iloc[-1])
    ratio = v_now / v3m
    state = "🔴 倒掛（恐慌）" if ratio > 1 else "🟢 正常"
    return f"VIX={v_now:.2f} / VIX3M={v3m:.2f} / 比值={ratio:.3f} {state}"


# ─── 6. yfinance：殖利率（無 FRED 也能用）──────────────
def t_yf_yields():
    import yfinance as yf
    tnx = yf.Ticker("^TNX").history(period="5d")
    irx = yf.Ticker("^IRX").history(period="5d")
    if tnx.empty:
        raise RuntimeError("^TNX 無資料")
    return (f"10Y={tnx['Close'].iloc[-1]:.2f}% / "
            f"13W={irx['Close'].iloc[-1]:.2f}%（粗略估算用）")


# ─── 7. yfinance：TSM ADR 溢價（外資先行指標）──────────
def t_tsm_adr():
    import yfinance as yf
    tsm  = yf.Ticker("TSM").history(period="5d")
    t2330 = yf.Ticker("2330.TW").history(period="5d")
    if tsm.empty or t2330.empty:
        raise RuntimeError("yfinance 取不到")
    twd = yf.Ticker("TWD=X").history(period="5d")
    fx = float(twd["Close"].iloc[-1]) if not twd.empty else 32.5
    tsm_implied = float(tsm["Close"].iloc[-1]) * fx / 5
    actual = float(t2330["Close"].iloc[-1])
    premium = (tsm_implied / actual - 1) * 100
    state = ("🔴 折價（外資搶逃）" if premium < -2 else
             "🟠 微折價" if premium < 0 else "🟢 溢價（外資進場）")
    return (f"TSM=${tsm['Close'].iloc[-1]:.2f} / "
            f"2330={actual:.0f} / 匯率={fx:.2f} / 溢價={premium:+.2f}% {state}")


# ─── 8. FinMind：當沖比率（散戶過熱）──────────────────
def t_finmind_daytrading():
    if not FINMIND_TOKEN:
        return "略過（無 token）"
    url = "https://api.finmindtrade.com/api/v4/data"
    start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    for ds in ["TaiwanStockDayTradingInfo", "TaiwanStockDayTrading"]:
        params = {"dataset": ds, "start_date": start, "token": FINMIND_TOKEN}
        try:
            r = requests.get(url, params=params, timeout=15).json()
            if r.get("msg") == "success" and r.get("data"):
                df = pd.DataFrame(r["data"])
                return f"dataset={ds} | 筆數={len(df)} | 欄位={list(df.columns)[:6]}"
        except Exception:
            continue
    raise RuntimeError("兩種 dataset 名稱都失敗（可能需付費版）")


# ─── 9. FinMind：台幣匯率 ──────────────────────────────
def t_finmind_fx():
    if not FINMIND_TOKEN:
        return "略過（無 token）"
    url = "https://api.finmindtrade.com/api/v4/data"
    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    params = {"dataset": "TaiwanExchangeRate", "data_id": "USD",
              "start_date": start, "token": FINMIND_TOKEN}
    r = requests.get(url, params=params, timeout=15).json()
    if r.get("msg") != "success" or not r.get("data"):
        raise RuntimeError(f"API msg={r.get('msg')}")
    df = pd.DataFrame(r["data"])
    last = df.iloc[-1].to_dict()
    return f"筆數={len(df)} | 最新={last}"


# ─── 10. CNN Fear & Greed Index ────────────────────────
def t_cnn_fg():
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {
        "User-Agent": UA,
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://edition.cnn.com",
        "Referer": "https://edition.cnn.com/",
    }
    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    j = r.json()
    fg = j.get("fear_and_greed", {})
    score = fg.get("score")
    rating = fg.get("rating")
    if score is None:
        raise RuntimeError(f"無分數欄位，結構: {list(j.keys())}")
    return f"score={float(score):.1f} | rating={rating}"


# ═══════════════════════════════════════════════════════
# 主程式
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 62)
    print("  資料源連線測試 v2 — test_data_sources_v2.py")
    print("=" * 62)
    print(f"  NASDAQ_KEY:    {'✓ 已設定 (' + NASDAQ_KEY[:6] + '...)' if NASDAQ_KEY else '✗ 未設定'}")
    print(f"  FRED_KEY:      {'✓ 已設定 (' + FRED_KEY[:6] + '...)' if FRED_KEY else '✗ 未設定'}")
    print(f"  FINMIND_TOKEN: {'✓ 已設定 (' + FINMIND_TOKEN[:6] + '...)' if FINMIND_TOKEN else '✗ 未設定'}")
    print("=" * 62)
    print()

    tests = [
        ("1. Nasdaq Data Link / AAII_SENTIMENT",   t_nasdaq_aaii),
        ("2. Nasdaq Data Link / NAAIM/NAAIM",      t_nasdaq_naaim),
        ("3. FRED / T10Y2Y（殖利率倒掛）",          t_fred_t10y2y),
        ("4. FRED / BAMLH0A0HYM2（HY 信用利差）",   t_fred_hy),
        ("5. yfinance / VIX 期限結構",              t_vix_term),
        ("6. yfinance / 美債殖利率（備援）",        t_yf_yields),
        ("7. yfinance / TSM ADR 溢價",              t_tsm_adr),
        ("8. FinMind / 當沖比率",                   t_finmind_daytrading),
        ("9. FinMind / 台幣匯率",                   t_finmind_fx),
        ("10. CNN Fear & Greed Index",              t_cnn_fg),
    ]

    for name, func in tests:
        run(name, func)

    print("=" * 62)
    print("  總結")
    print("=" * 62)
    passed = sum(1 for _, s, _ in results if "PASS" in s)
    print(f"  通過: {passed}/{len(results)}\n")
    for name, status, msg in results:
        print(f"  {status}  {name}")
    print()
    print("=" * 62)
    print("  把完整輸出複製回 Claude")
    print("=" * 62)